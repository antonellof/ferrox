//! The Hadamard rotation PrismML folds into Bonsai's weights, undone on
//! the activation side.
//!
//! A Bonsai checkpoint (`prism.hadamard.version = 1`) stores each
//! listed weight `W` as `W' = W S H` for a blockwise normalized
//! Sylvester (Walsh) Hadamard `H` and a diagonal `S` of `+/-1`, so that
//! the ternary quantizer sees a flatter distribution. The product is
//! recovered on the input, `W x = W' (H S x)`, in the order the
//! reference applies it (`llama-graph.cpp:1548-1575`, `build_lora_mm`):
//! an optional head-order permutation, then the sign flip, then the
//! rotation, then the matmul against the folded weight. `H` is
//! symmetric and its own inverse, so "rotation" means the same
//! transform both ways. The token-embedding table stores rotated ROWS
//! and is undone after the lookup instead: `e = S (H z)`
//! (`llama-graph.cpp:2425-2436`).
//!
//! The rotation is `H` of size `block` applied to every consecutive
//! `block`-wide slice of the vector (`llama_mul_mat_hadamard`,
//! `llama-impl.h:57-76`: the activation reshaped to `[block, rest]`
//! and multiplied by the `block x block` matrix whose entry is
//! `(-1)^popcount(row & col) / sqrt(block)`, `llama-model.cpp:
//! 2015-2027`). That matrix is the natural-order Walsh-Hadamard
//! transform, computed here in place by the butterfly rather than as
//! a matmul.
//!
//! The permutation is for the gated delta net's `ssm_out`
//! (`prism.hadamard.gdn_v_grouped`): the block's output arrives with
//! its V heads in the TILED order ferrox and llama.cpp both produce for
//! `qwen35` (`[hd, nk, rep]`, ggml `ne` order) and the fold was
//! computed in the grouped order `[hd, rep, nk]` (`llama-model.cpp:
//! 2083-2092`, `llama-graph.cpp:1562-1568`).
//!
//! What is deliberately NOT here: parsing the metadata (the loader's,
//! `ferrox_models::hadamard_fold`), and any fast path. The transform
//! costs `n log n` adds on a 5120- to 17408-wide vector, which next to
//! a 27B matmul is nothing; the day it shows in a profile it moves,
//! and its twin stays here.

use std::sync::Arc;

/// `[hd, nk, rep]` tiled to `[hd, rep, nk]` grouped, before the signs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadPerm {
    pub hd: usize,
    pub nk: usize,
    pub rep: usize,
}

/// Where a fold sits relative to the matrix it wraps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FoldSite {
    /// The matrix's INPUT is transformed before the product
    /// (`prism.hadamard.weight_names`).
    Input,
    /// The matrix's ROWS are stored rotated and undone after a lookup
    /// (`prism.hadamard.inverse_weight_names`, the token embedding).
    RowLookup,
}

/// One weight's activation-side transform.
#[derive(Debug, Clone)]
pub struct HadamardFold {
    /// A power of two dividing the width.
    pub block: usize,
    /// `+1.0 / -1.0` per input element; `None` is `sign_mode =
    /// identity`.
    pub signs: Option<Arc<[f32]>>,
    pub perm: Option<HeadPerm>,
    pub site: FoldSite,
}

impl HadamardFold {
    /// The width this fold is for, when it carries signs; a fold
    /// without signs fits any width the block divides.
    pub fn width(&self) -> Option<usize> {
        self.signs.as_ref().map(|s| s.len())
    }

    fn check(&self, n: usize) {
        assert!(
            self.block.is_power_of_two() && n.is_multiple_of(self.block),
            "Hadamard block {} does not divide a width of {n}",
            self.block
        );
        if let Some(s) = &self.signs {
            assert_eq!(s.len(), n, "sign vector width");
        }
        if let Some(p) = self.perm {
            assert_eq!(p.hd * p.nk * p.rep, n, "head permutation width");
        }
    }

    /// `x` as the folded matrix expects it: permuted, signed, rotated.
    /// The same for both sites: a row-lookup table used as a tied head
    /// stores `z = H S e`, and `e . h = (S H z) . h = z . (H S h)`, so
    /// the head's input takes the forward transform too (no
    /// permutation is ever set on a lookup table).
    pub fn transform_input(&self, x: &[f32]) -> Vec<f32> {
        self.check(x.len());
        let mut v = match self.perm {
            Some(p) => tiled_to_grouped(x, p),
            None => x.to_vec(),
        };
        if let Some(s) = &self.signs {
            for (a, b) in v.iter_mut().zip(s.iter()) {
                *a *= b;
            }
        }
        fwht_normalized(&mut v, self.block);
        v
    }

    /// [`Self::transform_input`] over `n` consecutive rows of `x`.
    pub fn transform_rows(&self, x: &[f32], n: usize) -> Vec<f32> {
        assert!(n > 0 && x.len().is_multiple_of(n));
        let w = x.len() / n;
        let mut out = vec![0f32; x.len()];
        // Rows are independent: one parallel region over them. Serial,
        // this was a quarter of a Bonsai-2-27B prefill step.
        crate::par::chunks_mut(&mut out, w, 1, |r, dst| {
            dst.copy_from_slice(&self.transform_input(&x[r * w..(r + 1) * w]));
        });
        out
    }

    /// A stored (rotated) row back to the primal basis: `e = S (H z)`.
    pub fn restore_row(&self, row: &mut [f32]) {
        self.check(row.len());
        fwht_normalized(row, self.block);
        if let Some(s) = &self.signs {
            for (a, b) in row.iter_mut().zip(s.iter()) {
                *a *= b;
            }
        }
    }
}

/// `[hd, nk, rep]` (ggml `ne` order: `hd` fastest) to `[hd, rep, nk]`.
fn tiled_to_grouped(x: &[f32], p: HeadPerm) -> Vec<f32> {
    let HeadPerm { hd, nk, rep } = p;
    let mut out = vec![0f32; x.len()];
    for r in 0..rep {
        for k in 0..nk {
            let src = &x[hd * (k + nk * r)..hd * (k + nk * r) + hd];
            out[hd * (r + rep * k)..hd * (r + rep * k) + hd].copy_from_slice(src);
        }
    }
    out
}

/// The natural-order Walsh-Hadamard transform of every `block`-wide
/// slice of `x`, scaled by `1 / sqrt(block)`: the matrix with entries
/// `(-1)^popcount(row & col) / sqrt(block)`, which is its own inverse.
pub fn fwht_normalized(x: &mut [f32], block: usize) {
    assert!(block.is_power_of_two() && x.len().is_multiple_of(block));
    let scale = 1.0 / (block as f32).sqrt();
    for chunk in x.chunks_exact_mut(block) {
        // Stages 1 and 2 (h = 1, 2) as explicit 4-wide butterflies, the
        // rest as two half-slices per pair so the inner loop is a plain
        // zip the compiler vectorises; a `chunk[j + h]` index form kept
        // every stage scalar.
        if block >= 4 {
            for q in chunk.as_chunks_mut::<4>().0 {
                let (a, b, c, d) = (q[0], q[1], q[2], q[3]);
                let (ab, amb, cd, cmd) = (a + b, a - b, c + d, c - d);
                q[0] = ab + cd;
                q[1] = amb + cmd;
                q[2] = ab - cd;
                q[3] = amb - cmd;
            }
        } else {
            let mut h = 1;
            while h < block {
                for i in (0..block).step_by(2 * h) {
                    for j in i..i + h {
                        let (a, b) = (chunk[j], chunk[j + h]);
                        chunk[j] = a + b;
                        chunk[j + h] = a - b;
                    }
                }
                h *= 2;
            }
        }
        let mut h = 4;
        while h < block {
            for pair in chunk.chunks_exact_mut(2 * h) {
                let (lo, hi) = pair.split_at_mut(h);
                for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                    let (x, y) = (*a, *b);
                    *a = x + y;
                    *b = x - y;
                }
            }
            h *= 2;
        }
        for v in chunk.iter_mut() {
            *v *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The butterfly is the matrix llama.cpp builds (`llama-model.cpp:
    /// 2015-2027`), entry for entry, and is its own inverse.
    #[test]
    fn the_butterfly_is_the_parity_matrix_and_its_own_inverse() {
        // Every block width both arms of `fwht_normalized` take (the
        // scalar one below 4, the 4-wide first two stages plus the
        // vectorised pairs from 4 up), and Bonsai's 1024.
        for n in [1usize, 2, 4, 8, 16, 64, 1024] {
            let scale = 1.0 / (n as f32).sqrt();
            for col in (0..n).step_by(if n > 64 { 37 } else { 1 }) {
                let mut e = vec![0f32; n];
                e[col] = 1.0;
                fwht_normalized(&mut e, n);
                for (row, v) in e.iter().enumerate() {
                    let want = if (row & col).count_ones() % 2 == 1 {
                        -scale
                    } else {
                        scale
                    };
                    assert!(
                        (v - want).abs() < 1e-6,
                        "n={n} H[{row}][{col}] = {v}, want {want}"
                    );
                }
            }
        }
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.7).sin()).collect();
        let mut y = x.clone();
        fwht_normalized(&mut y, 16);
        fwht_normalized(&mut y, 16);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    /// Folding a weight and unfolding the input give the same product:
    /// `W x == (W H S) (H S x)` per block, with the permutation applied
    /// to both sides the way the reference does.
    #[test]
    fn the_folded_product_equals_the_plain_one() {
        let (rows, cols, block) = (3usize, 8usize, 4usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.31).cos()).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.53).sin()).collect();
        let signs: Vec<f32> = (0..cols)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        // The reference computes W' (H S x), so W' = W (H S)^-1 = W S H:
        // row_r' = H (S row_r), signs first and then the rotation, the
        // same order the activation takes.
        let mut w_folded = w.clone();
        for r in 0..rows {
            let row = &mut w_folded[r * cols..(r + 1) * cols];
            for (a, s) in row.iter_mut().zip(&signs) {
                *a *= s;
            }
            fwht_normalized(row, block);
        }
        let fold = HadamardFold {
            block,
            signs: Some(signs.clone().into()),
            perm: None,
            site: FoldSite::Input,
        };
        let xt = fold.transform_input(&x);
        for r in 0..rows {
            let plain: f32 = w[r * cols..(r + 1) * cols]
                .iter()
                .zip(&x)
                .map(|(a, b)| a * b)
                .sum();
            let folded: f32 = w_folded[r * cols..(r + 1) * cols]
                .iter()
                .zip(&xt)
                .map(|(a, b)| a * b)
                .sum();
            assert!(
                (plain - folded).abs() < 1e-5,
                "row {r}: {plain} vs {folded}"
            );
        }
        // The row-lookup site restores a stored row: z = H S e  ->  S H z = e.
        let e: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.19).cos()).collect();
        let mut z: Vec<f32> = e.iter().zip(&signs).map(|(a, s)| a * s).collect();
        fwht_normalized(&mut z, block);
        let lookup = HadamardFold {
            site: FoldSite::RowLookup,
            ..fold.clone()
        };
        lookup.restore_row(&mut z);
        for (a, b) in e.iter().zip(&z) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn the_head_permutation_moves_tiled_heads_into_groups() {
        // hd 2, nk 2 K-heads, rep 2: tiled order (k0 r0)(k1 r0)(k0 r1)(k1 r1)
        // becomes grouped (k0 r0)(k0 r1)(k1 r0)(k1 r1).
        let p = HeadPerm {
            hd: 2,
            nk: 2,
            rep: 2,
        };
        let x: Vec<f32> = vec![0., 1., 10., 11., 20., 21., 30., 31.];
        assert_eq!(
            tiled_to_grouped(&x, p),
            vec![0., 1., 20., 21., 10., 11., 30., 31.]
        );
    }
}
