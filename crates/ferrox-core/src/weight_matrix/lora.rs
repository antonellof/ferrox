//! LoRA deltas on a [`WeightMatrix`](super::WeightMatrix): the low-rank
//! term of `W x + Σ_i s_i · B_i (A_i x)`.
//!
//! llama.cpp applies an adapter inside ONE function, `build_lora_mm`
//! (`src/llama-graph.cpp:1486-1514`): every projection in every graph
//! goes through it, and the adapter is a lookup on the weight tensor
//! plus two more `mul_mat`s and an `add`. ferrox has no graph, so the
//! equivalent seam is the type every projection already is: a
//! [`WeightMatrix`](super::WeightMatrix) that carries a [`LoraStack`]
//! computes the delta inside its own `apply` / `apply_batch` /
//! `apply_gpu` / `dequant_row`, and the CPU row body, the batched host
//! bodies and the per-matrix GPU launches cannot disagree about whether
//! the adapter was applied, because none of them can see the base
//! weights without going through the same methods.
//!
//! **Layouts.** A projection's base is `[rows][cols]` (`n_out x n_in`,
//! row-major, a contiguous input vector per row). Its `lora_a` is
//! `[rank][cols]` and its `lora_b` is `[rows][rank]` in the same sense,
//! which is exactly ggml's `ne = [n_in, rank]` and `ne = [rank, n_out]`
//! (`llama-adapter.cpp:362-367` checks precisely those three
//! equalities), so `B (A x)` is two matvecs of the same row-major kind
//! as `W x`. A token-embedding adapter is stored FLIPPED upstream --
//! `token_embd.weight.lora_a` is `ne = [rank, n_vocab]` because the
//! graph gathers a ROW of it per token and multiplies by `lora_b`
//! (`llama-graph.cpp:2296-2304`, `llama-adapter.cpp:355-359`); the
//! loader in `ferrox-models` transposes that pair into this module's
//! one layout, so `dequant_row(token)` here is the same formula as
//! every other row.
//!
//! **Scale.** `scale = adapter_scale * alpha / rank` when the file
//! carries a nonzero `adapter.lora.alpha`, else `adapter_scale` alone
//! (`llama-adapter.h:53-57`, `rank = b->ne[0]`). The adapter half of
//! that product is a [`LoraScale`] SHARED by every delta one adapter
//! attached, so `POST /lora-adapters` and a per-request `lora` list
//! change one atomic per adapter and nothing is recomputed.
//!
//! **Cost when absent.** A matrix with no stack is a different enum
//! variant, so the seam costs nothing: the base arms are untouched and
//! the logits are byte-identical (pinned by the fixture suite). A
//! stack whose every scale is zero adds nothing either -- the delta is
//! skipped before any arithmetic, which is what makes `scale 0` equal
//! to "no adapter" bit for bit, as it is in libllama (measured: the
//! `--lora x:0` golden is byte-identical to the base golden).
//!
//! **No per-token allocation.** The `A x` intermediate is `rank`
//! floats; it lives in a thread-local scratch that is sized once and
//! reused, so the decode path allocates nothing here.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// One adapter's runtime scale (`adapter_scale` in llama.cpp's terms),
/// shared by every [`LoraDelta`] that adapter attached.
///
/// An `f32` in an atomic so a server can change it between requests
/// without a lock on the model: every projection reads it once per
/// apply with a relaxed load.
#[derive(Debug)]
pub struct LoraScale(AtomicU32);

impl LoraScale {
    pub fn new(scale: f32) -> Arc<Self> {
        Arc::new(Self(AtomicU32::new(scale.to_bits())))
    }

    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }

    pub fn set(&self, scale: f32) {
        self.0.store(scale.to_bits(), Ordering::Relaxed);
    }
}

/// Why a pair of `lora_a` / `lora_b` matrices cannot decorate a base of
/// a given shape. The message names every dimension involved, because
/// the usual cause is an adapter converted for a different base model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraShapeError {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub a_len: usize,
    pub b_len: usize,
}

impl std::fmt::Display for LoraShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LoRA pair does not fit a [{} x {}] base at rank {}: lora_a has {} values \
             (want rank x cols = {}), lora_b has {} values (want rows x rank = {})",
            self.rows,
            self.cols,
            self.rank,
            self.a_len,
            self.rank * self.cols,
            self.b_len,
            self.rows * self.rank
        )
    }
}

impl std::error::Error for LoraShapeError {}

/// One adapter's `(A, B)` pair for one base matrix.
#[derive(Debug)]
pub struct LoraDelta {
    /// `[rank][cols]`, row-major.
    a: Vec<f32>,
    /// `[rows][rank]`, row-major.
    b: Vec<f32>,
    rank: usize,
    rows: usize,
    cols: usize,
    /// `alpha / rank` when the adapter declares a nonzero alpha, else
    /// `1.0`: the half of llama.cpp's `get_scale` that is fixed at load.
    alpha_over_rank: f32,
    scale: Arc<LoraScale>,
}

impl LoraDelta {
    /// `a` is `[rank][cols]`, `b` is `[rows][rank]`, both row-major;
    /// `alpha` is the file's `adapter.lora.alpha` (0 when absent, which
    /// upstream reads as "no alpha scaling").
    pub fn new(
        a: Vec<f32>,
        b: Vec<f32>,
        rank: usize,
        rows: usize,
        cols: usize,
        alpha: f32,
        scale: Arc<LoraScale>,
    ) -> Result<Self, LoraShapeError> {
        if rank == 0 || a.len() != rank * cols || b.len() != rows * rank {
            return Err(LoraShapeError {
                rows,
                cols,
                rank,
                a_len: a.len(),
                b_len: b.len(),
            });
        }
        let alpha_over_rank = if alpha != 0.0 {
            alpha / rank as f32
        } else {
            1.0
        };
        Ok(Self {
            a,
            b,
            rank,
            rows,
            cols,
            alpha_over_rank,
            scale,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The adapter's shared scale, for a caller that wants to change it.
    pub fn scale_handle(&self) -> &Arc<LoraScale> {
        &self.scale
    }

    /// Bytes this delta keeps resident.
    pub fn resident_bytes(&self) -> usize {
        (self.a.len() + self.b.len()) * 4
    }

    /// The whole of llama.cpp's `get_scale`: `adapter_scale * alpha /
    /// rank`, or `adapter_scale` when alpha is zero.
    #[inline]
    fn effective_scale(&self) -> f32 {
        self.scale.get() * self.alpha_over_rank
    }

    /// `y = A x`, written into `y` (`rank` long).
    #[inline]
    fn project(&self, x: &[f32], y: &mut [f32]) {
        for (k, yk) in y.iter_mut().enumerate() {
            let row = &self.a[k * self.cols..(k + 1) * self.cols];
            *yk = dot(row, x);
        }
    }

    /// `out[r] += s * b[r] . y` for every row.
    #[inline]
    fn accumulate(&self, s: f32, y: &[f32], out: &mut [f32]) {
        for (r, o) in out.iter_mut().enumerate() {
            let brow = &self.b[r * self.rank..(r + 1) * self.rank];
            *o += s * dot(brow, y);
        }
    }

    /// `out += s · B (A x)` for one activation. `out` is `rows` long.
    pub fn add_to(&self, x: &[f32], out: &mut [f32], scratch: &mut Vec<f32>) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(out.len(), self.rows);
        let s = self.effective_scale();
        if s == 0.0 {
            return;
        }
        scratch.clear();
        scratch.resize(self.rank, 0.0);
        self.project(x, scratch);
        self.accumulate(s, scratch, out);
    }

    /// `out_row += s · (b[r] A)`: the delta of ONE row of the adapted
    /// matrix, which is what a row gather (the token embedding) needs.
    pub fn add_row_to(&self, r: usize, out_row: &mut [f32]) {
        debug_assert!(r < self.rows);
        debug_assert_eq!(out_row.len(), self.cols);
        let s = self.effective_scale();
        if s == 0.0 {
            return;
        }
        let brow = &self.b[r * self.rank..(r + 1) * self.rank];
        for (k, &bk) in brow.iter().enumerate() {
            let arow = &self.a[k * self.cols..(k + 1) * self.cols];
            let sb = s * bk;
            for (o, &a) in out_row.iter_mut().zip(arow) {
                *o += sb * a;
            }
        }
    }

    /// `out[b] += s · B (A x_b)` for every position of a batch. `x_batch`
    /// is `[batch][cols]`, `out` is `[batch][rows]`, both row-major --
    /// the layouts `apply_batch` takes and returns.
    ///
    /// Parallel over positions: a prefill's `batch x rows x rank` is the
    /// only place the low-rank term is not negligible next to the base
    /// GEMM, and one thread per position keeps every write disjoint.
    pub fn add_batch_to(&self, x_batch: &[f32], batch: usize, out: &mut [f32]) {
        debug_assert_eq!(x_batch.len(), batch * self.cols);
        debug_assert_eq!(out.len(), batch * self.rows);
        let s = self.effective_scale();
        if s == 0.0 || batch == 0 {
            return;
        }
        let rows = self.rows;
        let cols = self.cols;
        crate::par::chunks_mut(out, rows, 1, |b, out_b| {
            SCRATCH.with(|cell| {
                let mut y = cell.borrow_mut();
                y.clear();
                y.resize(self.rank, 0.0);
                self.project(&x_batch[b * cols..(b + 1) * cols], &mut y);
                self.accumulate(s, &y, out_b);
            });
        });
    }
}

thread_local! {
    /// The `A x` intermediate, `rank` floats, reused across calls so
    /// the decode path never allocates here.
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Every adapter attached to one base matrix, applied in order. Two
/// adapters on one weight are two terms of the sum, as they are in
/// llama.cpp's `for (const auto & lora : *loras)`.
#[derive(Debug, Default)]
pub struct LoraStack {
    deltas: Vec<LoraDelta>,
}

impl LoraStack {
    pub fn new(delta: LoraDelta) -> Self {
        Self {
            deltas: vec![delta],
        }
    }

    pub fn push(&mut self, delta: LoraDelta) {
        self.deltas.push(delta);
    }

    pub fn len(&self) -> usize {
        self.deltas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }

    pub fn deltas(&self) -> &[LoraDelta] {
        &self.deltas
    }

    pub fn resident_bytes(&self) -> usize {
        self.deltas.iter().map(LoraDelta::resident_bytes).sum()
    }

    /// `out += Σ_i s_i · B_i (A_i x)`.
    pub fn add_to(&self, x: &[f32], out: &mut [f32]) {
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            for d in &self.deltas {
                d.add_to(x, out, &mut scratch);
            }
        });
    }

    /// The delta of one row, for a row gather.
    pub fn add_row_to(&self, r: usize, out_row: &mut [f32]) {
        for d in &self.deltas {
            d.add_row_to(r, out_row);
        }
    }

    /// `out[b] += Σ_i s_i · B_i (A_i x_b)` over a `[batch][cols]` input.
    pub fn add_batch_to(&self, x_batch: &[f32], batch: usize, out: &mut [f32]) {
        for d in &self.deltas {
            d.add_batch_to(x_batch, batch, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta(rows: usize, cols: usize, rank: usize, alpha: f32, scale: f32) -> LoraDelta {
        let a: Vec<f32> = (0..rank * cols).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..rows * rank).map(|i| (i as f32 * 0.53).cos()).collect();
        LoraDelta::new(a, b, rank, rows, cols, alpha, LoraScale::new(scale)).unwrap()
    }

    /// The reference: materialise `s * B A` and multiply.
    fn dense_delta(d: &LoraDelta, x: &[f32]) -> Vec<f32> {
        let s = d.effective_scale();
        (0..d.rows)
            .map(|r| {
                let mut acc = 0.0;
                for k in 0..d.rank {
                    let bk = d.b[r * d.rank + k];
                    for (c, &xc) in x.iter().enumerate() {
                        acc += s * bk * d.a[k * d.cols + c] * xc;
                    }
                }
                acc
            })
            .collect()
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn matvec_delta_matches_the_materialised_product() {
        let d = delta(7, 5, 3, 6.0, 0.8);
        let x: Vec<f32> = (0..5).map(|i| i as f32 - 2.0).collect();
        let mut out = vec![0.0; 7];
        d.add_to(&x, &mut out, &mut Vec::new());
        close(&out, &dense_delta(&d, &x));
    }

    #[test]
    fn alpha_over_rank_is_llama_cpp_s_get_scale() {
        // alpha 6 at rank 3 is a factor of 2; alpha 0 is a factor of 1.
        let with = delta(4, 4, 3, 6.0, 0.5);
        let without = delta(4, 4, 3, 0.0, 0.5);
        assert!((with.effective_scale() - 1.0).abs() < 1e-7);
        assert!((without.effective_scale() - 0.5).abs() < 1e-7);
    }

    #[test]
    fn row_delta_agrees_with_the_matvec_on_a_unit_vector() {
        let d = delta(6, 8, 2, 4.0, 1.3);
        for r in 0..6 {
            let mut row = vec![0.0; 8];
            d.add_row_to(r, &mut row);
            // Row r of (s B A) dotted with e_c is entry (r, c).
            for c in 0..8 {
                let mut e = vec![0.0; 8];
                e[c] = 1.0;
                let mut out = vec![0.0; 6];
                d.add_to(&e, &mut out, &mut Vec::new());
                assert!((out[r] - row[c]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn batch_delta_is_the_matvec_delta_per_position() {
        let d = delta(5, 6, 4, 8.0, 0.25);
        let batch = 3;
        let x: Vec<f32> = (0..batch * 6).map(|i| (i as f32 * 0.11).cos()).collect();
        let mut out = vec![0.0; batch * 5];
        d.add_batch_to(&x, batch, &mut out);
        for b in 0..batch {
            let mut one = vec![0.0; 5];
            d.add_to(&x[b * 6..(b + 1) * 6], &mut one, &mut Vec::new());
            close(&out[b * 5..(b + 1) * 5], &one);
        }
    }

    #[test]
    fn a_zero_scale_adds_nothing_bit_for_bit() {
        let d = delta(5, 6, 4, 8.0, 0.0);
        let x = vec![1.0; 6];
        let mut out = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let before = out.clone();
        d.add_to(&x, &mut out, &mut Vec::new());
        assert_eq!(out, before);
        let mut batch = vec![0.7; 10];
        d.add_batch_to(&[x.clone(), x.clone()].concat(), 2, &mut batch);
        assert_eq!(batch, vec![0.7; 10]);
    }

    #[test]
    fn the_scale_is_read_at_apply_time() {
        let d = delta(5, 6, 4, 0.0, 1.0);
        let x = vec![1.0; 6];
        let mut at_one = vec![0.0; 5];
        d.add_to(&x, &mut at_one, &mut Vec::new());
        d.scale_handle().set(0.5);
        let mut at_half = vec![0.0; 5];
        d.add_to(&x, &mut at_half, &mut Vec::new());
        let halved: Vec<f32> = at_one.iter().map(|v| v * 0.5).collect();
        close(&at_half, &halved);
    }

    #[test]
    fn a_stack_sums_its_adapters() {
        let d1 = delta(5, 6, 2, 0.0, 1.0);
        let d2 = delta(5, 6, 3, 0.0, 0.5);
        let x: Vec<f32> = (0..6).map(|i| i as f32 * 0.3 - 1.0).collect();
        let mut want = vec![0.0; 5];
        d1.add_to(&x, &mut want, &mut Vec::new());
        d2.add_to(&x, &mut want, &mut Vec::new());
        let mut stack = LoraStack::new(d1);
        stack.push(d2);
        let mut got = vec![0.0; 5];
        stack.add_to(&x, &mut got);
        close(&got, &want);
    }

    #[test]
    fn a_pair_of_the_wrong_shape_is_refused_with_every_dimension() {
        let err = LoraDelta::new(
            vec![0.0; 6],
            vec![0.0; 5],
            2,
            3,
            4,
            1.0,
            LoraScale::new(1.0),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("[3 x 4]"), "{msg}");
        assert!(msg.contains("rank 2"), "{msg}");
        assert!(msg.contains("want rank x cols = 8"), "{msg}");
        assert!(msg.contains("want rows x rank = 6"), "{msg}");
    }
}
