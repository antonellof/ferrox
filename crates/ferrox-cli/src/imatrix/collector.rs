//! The accumulator behind `ferrox imatrix`: llama.cpp's
//! `IMatrixCollector::collect_imatrix` (`tools/imatrix/imatrix.cpp:
//! 219-399`), fed by `ferrox_core::activation_tap` instead of a graph
//! callback.
//!
//! The rule it accumulates is one line and it is the whole tool: for
//! every row `x` of activations fed to a weight, and every column `j`,
//! `values[j] += x[j] * x[j]`, and the row count for that weight goes
//! up by one (`imatrix.cpp:365-372` for a dense weight, `:302-317` per
//! expert for a MUL_MAT_ID weight). The file then carries the sums and
//! the counts separately, and the quantizer divides.
//!
//! What is collected is decided by NAME, exactly as upstream does it
//! (`imatrix.cpp:229-237`): weights under `blk.`, and `output.weight`
//! only with `--process-output`. A `WeightMatrix` has no name, so the
//! caller registers each one it wants collected by address before the
//! run, and every observation for an unregistered matrix is dropped.
//! That is the norm weights, the embedding lookup, and on a tied model
//! the output head, which llama.cpp does not collect either because
//! its `src0` is `token_embd.weight`.
//!
//! The accumulation is contracted: `values[j] += x[j] * x[j]` in a C
//! file built with `-ffp-contract=on` is one fused multiply-add, and
//! `mul_add` here keeps the arithmetic identical for identical inputs.
//! The inputs are NOT identical -- the two engines' forward passes
//! diverge at the attention block, measured in the parent module --
//! so the two tools' files agree to the precision of the activations,
//! not bit for bit. `ferrox imatrix --compare` prints that bound.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use ferrox_core::WeightMatrix;

use super::file::Stats;

/// Where an observed matrix's rows go: which entry, and for an expert
/// stack, which matrix within it.
#[derive(Debug, Clone)]
struct Slot {
    name: String,
    /// `(expert index, expert count)` for a stacked expert weight;
    /// `None` for a dense one.
    expert: Option<(usize, usize)>,
    /// Columns the entry must have. Checked on every observation so a
    /// registration against the wrong matrix is refused at the first
    /// row rather than written into a file with the wrong width.
    cols: usize,
}

#[derive(Default)]
pub struct Collector {
    slots: HashMap<usize, Slot>,
    stats: Mutex<BTreeMap<String, Stats>>,
    /// The first non-finite sum seen, with the entry it was in.
    /// llama.cpp `exit(1)`s on the spot; a tap cannot, so the run
    /// checks this after every chunk.
    non_finite: Mutex<Option<String>>,
}

impl Collector {
    fn key(m: &WeightMatrix) -> usize {
        m as *const WeightMatrix as usize
    }

    /// Registers a dense weight: `n_mat = 1`, one count.
    pub fn register_dense(&mut self, m: &WeightMatrix, name: &str) {
        self.slots.insert(
            Self::key(m),
            Slot {
                name: name.to_string(),
                expert: None,
                cols: m.cols(),
            },
        );
        self.stats
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .entry(name.to_string())
            .or_insert_with(|| Stats {
                values: vec![0.0; m.cols()],
                counts: vec![0],
            });
    }

    /// Registers one expert of a stacked weight: the entry holds
    /// `n_experts` matrices and one count each.
    pub fn register_expert(
        &mut self,
        m: &WeightMatrix,
        name: &str,
        expert: usize,
        n_experts: usize,
    ) {
        assert!(expert < n_experts);
        self.slots.insert(
            Self::key(m),
            Slot {
                name: name.to_string(),
                expert: Some((expert, n_experts)),
                cols: m.cols(),
            },
        );
        self.stats
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .entry(name.to_string())
            .or_insert_with(|| Stats {
                values: vec![0.0; m.cols() * n_experts],
                counts: vec![0; n_experts],
            });
    }

    /// How many weights are registered.
    pub fn n_registered(&self) -> usize {
        self.slots.len()
    }

    /// The tap body. Rows are `[n_rows][cols]`.
    pub fn observe(&self, m: &WeightMatrix, rows: &[f32], n_rows: usize) {
        let Some(slot) = self.slots.get(&Self::key(m)) else {
            return;
        };
        let cols = slot.cols;
        debug_assert_eq!(rows.len(), n_rows * cols);
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        let e = stats
            .get_mut(&slot.name)
            .expect("every registered slot has an entry");
        let (mat, count_idx) = match slot.expert {
            Some((ex, _)) => (ex, ex),
            None => (0, 0),
        };
        let start = mat * cols;
        let values = &mut e.values[start..start + cols];
        for row in rows.chunks_exact(cols) {
            for (v, &x) in values.iter_mut().zip(row) {
                *v = x.mul_add(x, *v);
            }
        }
        e.counts[count_idx] += n_rows as i64;
        if let Some(bad) = values.iter().find(|v| !v.is_finite()) {
            let mut nf = self.non_finite.lock().unwrap_or_else(|e| e.into_inner());
            if nf.is_none() {
                *nf = Some(format!("{bad} detected in {}", slot.name));
            }
        }
    }

    /// `Some(message)` if any accumulated sum has gone non-finite.
    pub fn non_finite(&self) -> Option<String> {
        self.non_finite
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// A snapshot of the accumulated statistics.
    pub fn stats(&self) -> BTreeMap<String, Stats> {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_core::tensor::Tensor;

    fn matrix(rows: usize, cols: usize) -> WeightMatrix {
        WeightMatrix::F32(Tensor::new(vec![0.5; rows * cols], vec![rows, cols]))
    }

    /// The accumulation rule: per column, the sum of squares over every
    /// row observed, and the count is the number of rows, not the
    /// number of calls. Two calls of two rows must equal one call of
    /// four.
    #[test]
    fn dense_sums_squares_per_column_and_counts_rows() {
        let m = matrix(3, 4);
        let mut c = Collector::default();
        c.register_dense(&m, "blk.0.attn_q.weight");
        c.observe(&m, &[1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0], 2);
        c.observe(&m, &[0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0], 2);
        let s = &c.stats()["blk.0.attn_q.weight"];
        assert_eq!(s.values, vec![11.0, 5.0, 10.0, 21.0]);
        assert_eq!(s.counts, vec![4]);
    }

    /// An unregistered matrix is dropped, not accumulated under some
    /// default name: that is how norms and the embedding stay out of
    /// the file, exactly as llama.cpp's `blk.` filter keeps them out.
    #[test]
    fn an_unregistered_matrix_is_ignored() {
        let m = matrix(2, 4);
        let other = matrix(2, 4);
        let mut c = Collector::default();
        c.register_dense(&m, "blk.0.attn_q.weight");
        c.observe(&other, &[9.0; 4], 1);
        let s = c.stats();
        assert_eq!(s.len(), 1);
        assert_eq!(s["blk.0.attn_q.weight"].counts, vec![0]);
    }

    /// An expert stack is one entry with one matrix and one count per
    /// expert (`imatrix.cpp:302-317`): observing expert 1 leaves expert
    /// 0's sums and count untouched.
    #[test]
    fn experts_accumulate_into_their_own_matrix_and_count() {
        let e0 = matrix(2, 3);
        let e1 = matrix(2, 3);
        let mut c = Collector::default();
        c.register_expert(&e0, "blk.0.ffn_gate_exps.weight", 0, 2);
        c.register_expert(&e1, "blk.0.ffn_gate_exps.weight", 1, 2);
        c.observe(&e1, &[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], 2);
        c.observe(&e0, &[2.0, 0.0, 0.0], 1);
        let s = &c.stats()["blk.0.ffn_gate_exps.weight"];
        assert_eq!(s.values, vec![4.0, 0.0, 0.0, 2.0, 5.0, 10.0]);
        assert_eq!(s.counts, vec![1, 2]);
    }

    /// A sum that overflows to infinity is reported, because llama.cpp
    /// exits on it and a file with an `inf` in it makes the quantizer
    /// throw (`llama-quant.cpp:620-624`).
    #[test]
    fn a_non_finite_sum_is_reported_with_its_entry() {
        let m = matrix(1, 2);
        let mut c = Collector::default();
        c.register_dense(&m, "blk.3.ffn_down.weight");
        c.observe(&m, &[f32::MAX, 1.0], 1);
        c.observe(&m, &[f32::MAX, 1.0], 1);
        let msg = c.non_finite().expect("overflow must be reported");
        assert!(msg.contains("blk.3.ffn_down.weight"), "{msg}");
    }
}
