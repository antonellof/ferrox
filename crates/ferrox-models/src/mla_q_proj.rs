//! **HOW AN MLA LAYER PROJECTS Q** -- low-rank through `attn_q_a` /
//! `attn_q_a_norm` / `attn_q_b`, or DIRECT through one `attn_q`. One
//! enum every MLA attention body takes, so no forward pass can reach
//! the query without the file having answered which.
//!
//! # What it is
//!
//! `src/models/deepseek2.cpp:104-115` create `attn_q_a` {n_embd,
//! q_lora_rank}, `attn_q_a_norm` {q_lora_rank} and `attn_q_b`
//! {q_lora_rank, n_head * n_embd_head_k_mla} when `q_lora_rank > 0`,
//! and one `attn_q` {n_embd, n_head * n_embd_head_k_mla} otherwise;
//! the graph (`:283-296`) runs `q = wq_b(rms_norm(wq_a(x)))` or
//! `q = wq(x)`. What follows -- the per-head nope / pe split, the
//! rotation of `q_pe` -- reads the same `[n_head * (qk_nope + qk_rope)]`
//! vector either way. `plm.cpp:32,81` has ONLY the direct form and no
//! `q_lora_rank` key at all; `kimi-linear.cpp:120` has both on its own
//! engine.
//!
//! # Reach -- MEASURED
//!
//! `grep -l ATTN_KV_A_MQA src/models/*.cpp` over all 140 graphs is six
//! files (`deepseek2`, `deepseek32`, `glm-dsa`, `kimi-linear`,
//! `minicpm3`, `plm`); of those, `grep 'LLM_TENSOR_ATTN_Q,'` finds a
//! direct `wq` in `deepseek2.cpp:114`, `kimi-linear.cpp:120` and
//! `plm.cpp:32`. On the MLA engine that is two rows: `plm`, always
//! direct, and `deepseek2` for the "lite" checkpoints (`deepseek2.cpp:
//! 8,11-13`: `is_lite` is decided from the LAYER COUNT -- 27, 26, or 48
//! with a 128256 vocabulary -- and a lite file's `q_lora_rank` key is
//! never read, so DeepSeek-V2-Lite, GigaChat3-10B-A1.8B and
//! Kanana-2-30B-A3B all project Q directly). `crate::mla_arch` carries
//! that rule per architecture; this module is the type it produces.
//!
//! # Why an enum and not an `Option` on the low-rank fields
//!
//! `MlaAttnWeights` had `q_a_proj`, `q_a_layernorm` and `q_b_proj` as
//! three REQUIRED fields, so the loader for a direct-Q file had no
//! honest value to put in them and every real DeepSeek-V2-Lite export
//! failed on `missing hparam deepseek2.attention.q_lora_rank`, a true
//! statement about a key llama.cpp does not read for that file. Three
//! `Option`s would let a loader fill one and not the others; one enum
//! cannot be half-filled, and [`MlaQProj::apply`] is the ONE place the
//! two forms meet the hidden state.

use ferrox_core::matmul::rms_norm;
use ferrox_core::weight_matrix::WeightMatrix;

/// The query projection of one MLA layer.
pub enum MlaQProj {
    /// `attn_q_b(rms_norm(attn_q_a(x), q_a_norm))`
    /// (`deepseek2.cpp:283-296`, `q_lora_rank > 0`).
    LowRank {
        /// `attn_q_a`: `[q_lora_rank, hidden_dim]`.
        a: WeightMatrix,
        /// `attn_q_a_norm`: `[q_lora_rank]`.
        norm: Vec<f32>,
        /// `attn_q_b`: `[n_heads * (qk_nope + qk_rope), q_lora_rank]`.
        b: WeightMatrix,
    },
    /// `attn_q(x)` (`deepseek2.cpp:114,285`, `plm.cpp:32,81`):
    /// `[n_heads * (qk_nope + qk_rope), hidden_dim]`.
    Direct(WeightMatrix),
}

impl MlaQProj {
    /// The `[n_heads * (qk_nope + qk_rope)]` query, nope first within
    /// each head, BEFORE any rotation.
    pub fn apply(&self, hidden: &[f32], rms_norm_eps: f32) -> Vec<f32> {
        match self {
            MlaQProj::LowRank { a, norm, b } => {
                let q_a = a.apply(hidden);
                b.apply(&rms_norm(&q_a, norm, rms_norm_eps))
            }
            MlaQProj::Direct(w) => w.apply(hidden),
        }
    }

    /// Rows of the query the projection produces: `n_heads * (qk_nope +
    /// qk_rope)` for a file whose shapes agree with its keys.
    pub fn q_rows(&self) -> usize {
        match self {
            MlaQProj::LowRank { b, .. } => b.rows(),
            MlaQProj::Direct(w) => w.rows(),
        }
    }

    /// Input width: the hidden size.
    pub fn in_cols(&self) -> usize {
        match self {
            MlaQProj::LowRank { a, .. } => a.cols(),
            MlaQProj::Direct(w) => w.cols(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_core::tensor::Tensor;

    fn wm(data: &[f32], rows: usize, cols: usize) -> WeightMatrix {
        WeightMatrix::F32(Tensor::new(data.to_vec(), vec![rows, cols]))
    }

    /// The direct form IS the low-rank form with `a` the identity, a
    /// unit norm weight and `b` the direct matrix -- up to the RMS
    /// scale the norm applies, which this test cancels by feeding a
    /// unit-RMS input. A `Direct` arm that did anything else would be
    /// visible here.
    #[test]
    fn direct_equals_low_rank_through_the_identity() {
        let hidden = [1.0f32, -1.0, 1.0, -1.0]; // RMS exactly 1
        let w = [0.5f32, -0.25, 1.0, 2.0, -1.5, 0.75, 0.0, 3.0];
        let direct = MlaQProj::Direct(wm(&w, 2, 4));
        let identity = [
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let low_rank = MlaQProj::LowRank {
            a: wm(&identity, 4, 4),
            norm: vec![1.0; 4],
            b: wm(&w, 2, 4),
        };
        let got = direct.apply(&hidden, 0.0);
        let want = low_rank.apply(&hidden, 0.0);
        assert_eq!(got.len(), 2);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "{got:?} vs {want:?}");
        }
        assert_eq!(direct.q_rows(), 2);
        assert_eq!(direct.in_cols(), 4);
        assert_eq!(low_rank.q_rows(), 2);
        assert_eq!(low_rank.in_cols(), 4);
    }
}
