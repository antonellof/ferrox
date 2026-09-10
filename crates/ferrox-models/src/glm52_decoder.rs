//! A dedicated decoder for GLM-5.2's real architecture, separate from
//! `ferrox-models::decoder::Decoder` (the generic GQA path every other
//! preset uses) -- analogous to `kimi_decoder.rs`'s role for Kimi K3:
//! composes the already-independently-tested `glm_dsa` attention
//! module with a standard SwiGLU dense/MoE FFN into a real forward
//! pass, without touching the existing GQA decoder.
//!
//! Real per-layer flow, transcribed from `llama_model_glm_dsa::graph`'s
//! real per-layer loop in `src/models/glm-dsa.cpp` (confirmed against
//! both PR #23346's DeepSeek-V3.2 fork point and PR #25407's GLM-5.2
//! diff on top) -- notably **simpler** than Kimi K3's real per-layer
//! flow (`kimi_decoder.rs`'s module doc comment): no block-residual
//! scaffolding at all, an ordinary pre-norm transformer block:
//!
//! ```text
//! attn_in = rms_norm(hidden, attn_norm)
//! attn_out = glm52_attn_forward_token(attn_in)   // glm_dsa module
//! ffn_in = hidden + attn_out
//! ffn_out = rms_norm(ffn_in, ffn_norm) |> dense_or_moe_ffn
//! hidden = ffn_in + ffn_out
//! ```
//!
//! FFN: dense leading layers use ordinary SiLU-gated SwiGLU
//! (`ffn_gate`/`ffn_up`/`ffn_down`, `ggml`'s `LLM_FFN_SILU`/
//! `LLM_FFN_PAR` -- the same convention every other architecture's
//! dense FFN uses in this codebase, `ferrox_core::matmul::swiglu`). MoE
//! layers use real sigmoid gating with an aux-loss-free per-expert bias
//! (`noaux_tc`, confirmed against the real `config.json`:
//! `"scoring_func": "sigmoid"`, `"topk_method": "noaux_tc"` -- see
//! docs/MODELS.md) plus a shared expert, reusing
//! `ferrox_moe::route_top_k_sigmoid_with_bias`/`run_expert`/
//! `combine_expert_outputs` directly rather than re-deriving that math
//! here (it's already independently tested there, and it's exactly the
//! same real convention DeepSeek-V3/Kimi K3 use for their own
//! `noaux_tc` routing).
//!
//! Not yet run against a real GLM-5.2 checkpoint (~744B params, no
//! feasible download in this environment) -- tested here against
//! synthetic weights only, cross-validated for the attention math via
//! `glm_dsa`'s own independent Python cross-check;
//! this module's own test additionally confirms the full decoder
//! (attention + both dense and MoE FFN branches, across a full/shared
//! indexer-layer pair) composes into a finite, real forward pass end
//! to end, the same rigor `kimi_decoder.rs`'s own test applies for
//! Kimi K3.

use ferrox_core::matmul::{rms_norm, swiglu};
use ferrox_core::tensor::Tensor;
use ferrox_core::weight_matrix::WeightMatrix;
use ferrox_moe::{
    combine_expert_outputs, route_top_k_sigmoid_with_bias, run_expert, ExpertWeights, GluAct,
};

use crate::glm_dsa::{Glm52AttnState, Glm52AttnWeights, Glm52MlaConfig, IndexerConfig};

/// The dense leading layer's feed-forward block (real tensor names
/// `blk.{bid}.ffn_{gate,down,up}`).
pub struct Glm52DenseFfnWeights {
    pub gate_proj: WeightMatrix,
    pub up_proj: WeightMatrix,
    pub down_proj: WeightMatrix,
}

impl Glm52DenseFfnWeights {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let gate = self.gate_proj.apply(x);
        let up = self.up_proj.apply(x);
        let combined = swiglu(&gate, &up);
        self.down_proj.apply(&combined)
    }
}

/// One MoE layer's real weights: routed experts (sigmoid gating +
/// aux-loss-free bias) plus a shared expert, always active.
pub struct Glm52MoeFfnWeights {
    pub router_weight: WeightMatrix,
    pub e_score_correction_bias: Vec<f32>,
    pub experts: Vec<ExpertWeights>,
    pub shared_expert: ExpertWeights,
}

pub enum Glm52LayerFfn {
    Dense(Box<Glm52DenseFfnWeights>),
    Moe(Box<Glm52MoeFfnWeights>),
}

pub struct Glm52DecoderLayerWeights {
    pub attn_norm_weight: Vec<f32>,
    pub attn: Glm52AttnWeights,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn: Glm52LayerFfn,
    /// Whether this layer's indexer is "full" (computes its own top-k)
    /// or "shared" (reuses the nearest preceding "full" layer's top-k)
    /// -- real per-layer `indexer_types` array, see `glm_dsa`'s module
    /// doc comment point 1. The very first layer processed must always
    /// be `true` (the real architecture guarantees this).
    pub is_full_indexer_layer: bool,
}

pub struct Glm52DecoderWeights {
    pub embedding: Tensor, // [vocab_size, hidden_dim]
    pub layers: Vec<Glm52DecoderLayerWeights>,
    pub final_norm_weight: Vec<f32>,
    pub output_head: WeightMatrix, // [vocab_size, hidden_dim]
}

pub struct Glm52DecoderConfig {
    pub rms_norm_eps: f32,
    pub mla: Glm52MlaConfig,
    pub indexer: IndexerConfig,
    pub n_experts_active: usize,
    /// GLM-5.2's real `norm_topk_prob`/`routed_scaling_factor`
    /// (2.5 in the real published config -- see docs/MODELS.md).
    pub moe_renormalize: bool,
    pub routed_scaling_factor: f32,
}

pub struct Glm52DecodeState {
    layer_states: Vec<Glm52AttnState>,
}

impl Glm52DecodeState {
    pub fn new(weights: &Glm52DecoderWeights) -> Self {
        Glm52DecodeState {
            layer_states: weights
                .layers
                .iter()
                .map(|_| Glm52AttnState::new())
                .collect(),
        }
    }
}

fn moe_ffn_forward(weights: &Glm52MoeFfnWeights, cfg: &Glm52DecoderConfig, x: &[f32]) -> Vec<f32> {
    let router_logits = weights.router_weight.apply(x);
    let decision = route_top_k_sigmoid_with_bias(
        &router_logits,
        &weights.e_score_correction_bias,
        cfg.n_experts_active,
        cfg.moe_renormalize,
        cfg.routed_scaling_factor,
    );
    let routed_outputs: Vec<(Vec<f32>, f32)> = decision
        .expert_ids
        .iter()
        .zip(decision.weights.iter())
        .map(|(&e, &w)| (run_expert(x, &weights.experts[e], GluAct::Swiglu), w))
        .collect();
    let shared_out = run_expert(x, &weights.shared_expert, GluAct::Swiglu);
    combine_expert_outputs(&routed_outputs, &[shared_out], x.len())
}

/// One decode step across every layer. `prev_top_k` is reset to `None`
/// at the start of this function (per-token-forward-pass scope, see
/// `glm_dsa::glm52_attn_forward_token`'s doc comment) -- it must not be
/// threaded in from a previous token.
pub fn glm52_forward_token(
    weights: &Glm52DecoderWeights,
    cfg: &Glm52DecoderConfig,
    token_id: usize,
    state: &mut Glm52DecodeState,
) -> Vec<f32> {
    let hidden_dim = weights.embedding.cols();
    let mut hidden = weights.embedding.row(token_id).to_vec();
    let mut prev_top_k: Option<Vec<usize>> = None;

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let attn_in = rms_norm(&hidden, &layer.attn_norm_weight, cfg.rms_norm_eps);
        let attn_out = crate::glm_dsa::glm52_attn_forward_token(
            &layer.attn,
            &cfg.mla,
            &cfg.indexer,
            &attn_in,
            cfg.rms_norm_eps,
            layer.is_full_indexer_layer,
            &mut state.layer_states[layer_idx],
            &mut prev_top_k,
        );

        let mut ffn_in = hidden;
        for (f, a) in ffn_in.iter_mut().zip(attn_out.iter()) {
            *f += a;
        }

        let ffn_normed = rms_norm(&ffn_in, &layer.ffn_norm_weight, cfg.rms_norm_eps);
        let ffn_out = match &layer.ffn {
            Glm52LayerFfn::Dense(w) => w.forward(&ffn_normed),
            Glm52LayerFfn::Moe(w) => moe_ffn_forward(w, cfg, &ffn_normed),
        };

        let mut next_hidden = ffn_in;
        for (h, f) in next_hidden.iter_mut().zip(ffn_out.iter()) {
            *h += f;
        }
        hidden = next_hidden;
    }

    let final_normed = rms_norm(&hidden, &weights.final_norm_weight, cfg.rms_norm_eps);
    assert_eq!(final_normed.len(), hidden_dim);
    weights.output_head.apply(&final_normed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MlaRopeConfig;
    use crate::glm_dsa::IndexerWeights;

    const HIDDEN_DIM: usize = 8;
    const EPS: f32 = 1e-5;
    const NUM_HEADS: usize = 2;
    const QK_NOPE: usize = 4;
    const QK_ROPE: usize = 4;
    const KV_LORA: usize = 4;
    const Q_LORA: usize = 6;
    const V_HEAD_DIM: usize = 3;
    const IDX_N_HEADS: usize = 2;
    const IDX_HEAD_DIM: usize = 4;
    const IDX_ROPE_DIM: usize = 2;
    const TOP_K: usize = 2;
    const MOE_HIDDEN_DIM: usize = 8;
    const MOE_FFN_DIM: usize = 3;
    const N_EXPERTS: usize = 4;
    const N_EXPERTS_ACTIVE: usize = 2;
    const OUTPUT_VOCAB: usize = 5;
    const DENSE_FFN_DIM: usize = 5;

    fn wm(data: Vec<f32>, rows: usize, cols: usize) -> WeightMatrix {
        assert_eq!(data.len(), rows * cols);
        WeightMatrix::F32(Tensor::new(data, vec![rows, cols]))
    }

    // Small deterministic pseudo-random generator (no external RNG
    // dependency needed for a purely-structural synthetic test) --
    // same style used elsewhere in this codebase's synthetic fixtures.
    fn synth(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((seed * 131 + i * 17 + 7) % 23) as f32 * 0.05) - 0.55)
            .collect()
    }

    fn make_attn_weights(seed: usize, is_full: bool) -> Glm52AttnWeights {
        let q_head_dim = QK_NOPE + QK_ROPE;
        Glm52AttnWeights {
            q_a_proj: wm(synth(seed + 1, Q_LORA * HIDDEN_DIM), Q_LORA, HIDDEN_DIM),
            q_a_layernorm: vec![1.0; Q_LORA],
            q_b_proj: wm(
                synth(seed + 2, NUM_HEADS * q_head_dim * Q_LORA),
                NUM_HEADS * q_head_dim,
                Q_LORA,
            ),
            kv_a_proj_with_mqa: wm(
                synth(seed + 3, (KV_LORA + QK_ROPE) * HIDDEN_DIM),
                KV_LORA + QK_ROPE,
                HIDDEN_DIM,
            ),
            kv_a_layernorm: vec![1.0; KV_LORA],
            wk_b: (0..NUM_HEADS)
                .map(|h| wm(synth(seed + 4 + h, QK_NOPE * KV_LORA), QK_NOPE, KV_LORA))
                .collect(),
            wv_b: (0..NUM_HEADS)
                .map(|h| {
                    wm(
                        synth(seed + 6 + h, V_HEAD_DIM * KV_LORA),
                        V_HEAD_DIM,
                        KV_LORA,
                    )
                })
                .collect(),
            o_proj: wm(
                synth(seed + 8, HIDDEN_DIM * NUM_HEADS * V_HEAD_DIM),
                HIDDEN_DIM,
                NUM_HEADS * V_HEAD_DIM,
            ),
            indexer: if is_full {
                Some(IndexerWeights {
                    k_norm_weight: vec![1.0; IDX_HEAD_DIM],
                    k_norm_bias: vec![0.0; IDX_HEAD_DIM],
                    proj: wm(
                        synth(seed + 9, IDX_N_HEADS * HIDDEN_DIM),
                        IDX_N_HEADS,
                        HIDDEN_DIM,
                    ),
                    attn_k: wm(
                        synth(seed + 10, IDX_HEAD_DIM * HIDDEN_DIM),
                        IDX_HEAD_DIM,
                        HIDDEN_DIM,
                    ),
                    attn_q_b: wm(
                        synth(seed + 11, IDX_N_HEADS * IDX_HEAD_DIM * Q_LORA),
                        IDX_N_HEADS * IDX_HEAD_DIM,
                        Q_LORA,
                    ),
                })
            } else {
                None
            },
        }
    }

    fn make_weights() -> Glm52DecoderWeights {
        let layer0 = Glm52DecoderLayerWeights {
            attn_norm_weight: vec![1.0; HIDDEN_DIM],
            attn: make_attn_weights(100, true),
            ffn_norm_weight: vec![1.0; HIDDEN_DIM],
            ffn: Glm52LayerFfn::Dense(Box::new(Glm52DenseFfnWeights {
                gate_proj: wm(
                    synth(200, DENSE_FFN_DIM * HIDDEN_DIM),
                    DENSE_FFN_DIM,
                    HIDDEN_DIM,
                ),
                up_proj: wm(
                    synth(201, DENSE_FFN_DIM * HIDDEN_DIM),
                    DENSE_FFN_DIM,
                    HIDDEN_DIM,
                ),
                down_proj: wm(
                    synth(202, HIDDEN_DIM * DENSE_FFN_DIM),
                    HIDDEN_DIM,
                    DENSE_FFN_DIM,
                ),
            })),
            is_full_indexer_layer: true,
        };

        let expert = |seed: usize| ExpertWeights {
            gate: wm(
                synth(seed, MOE_FFN_DIM * MOE_HIDDEN_DIM),
                MOE_FFN_DIM,
                MOE_HIDDEN_DIM,
            ),
            up: wm(
                synth(seed + 1, MOE_FFN_DIM * MOE_HIDDEN_DIM),
                MOE_FFN_DIM,
                MOE_HIDDEN_DIM,
            ),
            down: wm(
                synth(seed + 2, MOE_HIDDEN_DIM * MOE_FFN_DIM),
                MOE_HIDDEN_DIM,
                MOE_FFN_DIM,
            ),
        };

        let layer1 = Glm52DecoderLayerWeights {
            attn_norm_weight: vec![1.0; HIDDEN_DIM],
            attn: make_attn_weights(300, false),
            ffn_norm_weight: vec![1.0; HIDDEN_DIM],
            ffn: Glm52LayerFfn::Moe(Box::new(Glm52MoeFfnWeights {
                router_weight: wm(synth(400, N_EXPERTS * HIDDEN_DIM), N_EXPERTS, HIDDEN_DIM),
                e_score_correction_bias: vec![0.0; N_EXPERTS],
                experts: (0..N_EXPERTS).map(|e| expert(500 + e * 10)).collect(),
                shared_expert: expert(900),
            })),
            is_full_indexer_layer: false,
        };

        Glm52DecoderWeights {
            embedding: Tensor::new(
                synth(1000, OUTPUT_VOCAB * HIDDEN_DIM),
                vec![OUTPUT_VOCAB, HIDDEN_DIM],
            ),
            layers: vec![layer0, layer1],
            final_norm_weight: vec![1.0; HIDDEN_DIM],
            output_head: wm(
                synth(1100, OUTPUT_VOCAB * HIDDEN_DIM),
                OUTPUT_VOCAB,
                HIDDEN_DIM,
            ),
        }
    }

    fn decoder_cfg() -> Glm52DecoderConfig {
        Glm52DecoderConfig {
            rms_norm_eps: EPS,
            mla: Glm52MlaConfig {
                num_heads: NUM_HEADS,
                q_lora_rank: Q_LORA,
                kv_lora_rank: KV_LORA,
                qk_nope_head_dim: QK_NOPE,
                qk_rope_head_dim: QK_ROPE,
                v_head_dim: V_HEAD_DIM,
                rope: MlaRopeConfig { theta: 8_000_000.0 },
            },
            indexer: IndexerConfig {
                n_heads: IDX_N_HEADS,
                head_dim: IDX_HEAD_DIM,
                rope_dim: IDX_ROPE_DIM,
                top_k: TOP_K,
                rope_theta: 8_000_000.0,
            },
            n_experts_active: N_EXPERTS_ACTIVE,
            moe_renormalize: true,
            routed_scaling_factor: 2.5,
        }
    }

    /// One GLM-5.2 decode step enters the CPU worker pool once.
    ///
    /// GLM-5.2 has no checkpoint this machine can load -- the `glm_5_2`
    /// preset is a sketch, not proof of real-checkpoint support -- so
    /// the instance is the same synthetic two-layer stack the tests
    /// above use. The count does not depend on the weights: it is how
    /// many times the driving thread crossed rayon's cold submission
    /// path, which before `engine/entry.rs` was once per parallel
    /// region and is now once per step.
    ///
    /// Sabotage: drop the `par::on_workers` from
    /// `Engine::forward_token` and this goes red with the region count.
    #[test]
    fn one_glm52_decode_step_enters_the_pool_once() {
        crate::engine::assert_one_pool_entry_per_step(
            &crate::Glm52Engine {
                weights: make_weights(),
                cfg: decoder_cfg(),
            },
            0,
        );
    }

    #[test]
    fn two_mixed_layers_run_end_to_end_across_three_tokens() {
        let weights = make_weights();
        let cfg = decoder_cfg();
        let mut state = Glm52DecodeState::new(&weights);

        for token_id in 0..3 {
            let logits = glm52_forward_token(&weights, &cfg, token_id % OUTPUT_VOCAB, &mut state);
            assert_eq!(logits.len(), OUTPUT_VOCAB);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "token {token_id}: logits must be finite, got {logits:?}"
            );
        }
    }

    #[test]
    fn shared_layer_without_a_prior_full_layer_in_the_same_token_panics() {
        // Build a decoder whose only layer is "shared" -- violates the
        // real architecture's invariant that the first layer processed
        // is always "full" (see `glm_dsa`'s module doc comment). Must
        // panic loudly, matching the real `GGML_ASSERT`, not silently
        // produce wrong output.
        let mut weights = make_weights();
        weights.layers.truncate(1);
        weights.layers[0].is_full_indexer_layer = false;
        weights.layers[0].attn.indexer = None;

        let cfg = decoder_cfg();
        let mut state = Glm52DecodeState::new(&weights);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            glm52_forward_token(&weights, &cfg, 0, &mut state)
        }));
        assert!(
            result.is_err(),
            "a lone \"shared\" layer with no preceding \"full\" layer must panic"
        );
    }
}
