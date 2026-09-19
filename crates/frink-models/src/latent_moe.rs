//! Kimi K3's "latent MoE" block (`KimiSparseMoeBlock` in the real
//! `modeling_kimi_linear.py`), a real architectural detail beyond a
//! standard top-k MoE FFN, discovered by reading the real source rather
//! than assuming the more common DeepSeek-V3-style MoE this crate's
//! `frink_moe::route_top_k` was originally written for: routed experts
//! operate on a *down-projected* latent space
//! (`routed_expert_hidden_size` = 3584, half of `hidden_size` = 7168 in
//! Kimi K3's real config — `use_latent_moe`/`latent_moe_use_norm` are
//! real, confirmed-active config fields, not a rare/optional path), not
//! the full hidden dimension. Real per-layer flow:
//!
//! ```text
//! identity = hidden
//! (topk_idx, topk_weight) = gate(hidden)              // on FULL hidden, not the latent
//! latent = down_proj(hidden)
//! combined = sum_i topk_weight[i] * expert[topk_idx[i]](latent)  // in latent space
//! combined = rms_norm(combined)                        // iff latent_moe_use_norm
//! routed_out = up_proj(combined)                        // back to hidden_size
//! output = routed_out + shared_expert(identity)         // shared expert on FULL hidden
//! ```
//!
//! Routing uses `frink_moe::route_top_k_sigmoid_with_bias` (the real
//! "aux-loss-free" per-expert bias affects selection only, not the
//! final combine weight — see that function's doc comment). Every
//! expert (`KimiBlockSparseMLP`) and the shared expert (`KimiMLP`) use
//! Kimi K3's real `situ` activation (`frink_core::situ_and_mul`), not
//! the more common SwiGLU.
//!
//! Not yet wired into `Decoder`. Tested here against synthetic weights,
//! cross-validated against an independent Python transcription of the
//! same real algorithm.

use frink_core::matmul::{rms_norm, situ_and_mul};
use frink_core::weight_matrix::WeightMatrix;
use frink_moe::route_top_k_sigmoid_with_bias;

/// One expert's gate/up/down projections
/// (`w1`=gate, `w3`=up, `w2`=down, matching `KimiBlockSparseMLP`'s real
/// naming). Used both for routed experts (in the latent `moe_hidden_dim`
/// space) and the shared expert (in the full `hidden_dim` space).
pub struct KimiExpertWeights {
    pub w1: WeightMatrix, // gate: [ffn_dim, in_dim]
    pub w2: WeightMatrix, // down: [in_dim, ffn_dim]
    pub w3: WeightMatrix, // up: [ffn_dim, in_dim]
}

impl KimiExpertWeights {
    pub fn forward(&self, x: &[f32], situ_beta: f32, situ_linear_beta: f32) -> Vec<f32> {
        let gate = self.w1.apply(x);
        let up = self.w3.apply(x);
        let combined = situ_and_mul(&gate, &up, situ_beta, situ_linear_beta);
        self.w2.apply(&combined)
    }
}

pub struct KimiLatentMoeWeights {
    pub router_weight: WeightMatrix,       // [n_experts, hidden_dim]
    pub e_score_correction_bias: Vec<f32>, // [n_experts]
    pub down_proj: WeightMatrix,           // [moe_hidden_dim, hidden_dim]
    pub up_proj: WeightMatrix,             // [hidden_dim, moe_hidden_dim]
    /// Present iff `latent_moe_use_norm` (true for Kimi K3's real
    /// config).
    pub routed_expert_norm_weight: Option<Vec<f32>>, // [moe_hidden_dim]
    pub experts: KimiExpertBacking,        // moe_hidden_dim <-> moe_intermediate_dim
    pub shared_expert: KimiExpertWeights,  // hidden_dim <-> (moe_intermediate_dim*num_shared)
}

/// How a Kimi layer's routed experts are held: `Resident` is the
/// original always-constructed form (zero-copy MXFP4 mmap views);
/// `Stored` holds only the per-layer byte layout and materializes one
/// expert at a time from a bounded, lease-protected store shared by
/// every layer -- same design (and same bit-equivalence argument) as
/// the GGUF path's `frink_models::decoder::ExpertBacking`. Every
/// expert in a Kimi layer has identical dims, so one layout serves the
/// whole layer.
pub enum KimiExpertBacking {
    Resident(Vec<KimiExpertWeights>),
    Stored {
        store: std::sync::Arc<
            frink_core::expert_store::ExpertStore<crate::kimi_loader::KimiExpertSource>,
        >,
        layout: crate::kimi_loader::KimiStoredExpertLayout,
        n_experts: usize,
        layer: u32,
    },
}

impl KimiExpertBacking {
    pub fn n_experts(&self) -> usize {
        match self {
            KimiExpertBacking::Resident(v) => v.len(),
            KimiExpertBacking::Stored { n_experts, .. } => *n_experts,
        }
    }

    /// Runs `f` against expert `e`, materializing it from the store
    /// first when store-backed (the lease pins the cache entry for
    /// exactly `f`'s borrow).
    pub fn with_expert<R>(&self, e: usize, f: impl FnOnce(&KimiExpertWeights) -> R) -> R {
        match self {
            KimiExpertBacking::Resident(v) => f(&v[e]),
            KimiExpertBacking::Stored {
                store,
                layout,
                layer,
                ..
            } => {
                let lease = store
                    .acquire(frink_core::expert_store::ExpertKey {
                        layer: *layer,
                        expert: e as u32,
                    })
                    .unwrap_or_else(|err| {
                        panic!(
                            "kimi expert store read failed for layer {layer} expert {e}: {err} \
                             (checkpoint file unreadable mid-decode)"
                        )
                    });
                let tmp = layout.materialize(&lease);
                f(&tmp)
            }
        }
    }
}

pub struct KimiMoeConfig {
    pub n_experts_active: usize,
    pub moe_renormalize: bool,
    pub routed_scaling_factor: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub rms_norm_eps: f32,
}

pub fn kimi_latent_moe_forward(
    weights: &KimiLatentMoeWeights,
    cfg: &KimiMoeConfig,
    hidden: &[f32],
) -> Vec<f32> {
    let router_logits = weights.router_weight.apply(hidden);
    let decision = route_top_k_sigmoid_with_bias(
        &router_logits,
        &weights.e_score_correction_bias,
        cfg.n_experts_active,
        cfg.moe_renormalize,
        cfg.routed_scaling_factor,
    );

    let latent = weights.down_proj.apply(hidden);

    let mut combined = vec![0f32; latent.len()];
    for (&eid, &w) in decision.expert_ids.iter().zip(decision.weights.iter()) {
        let out = weights.experts.with_expert(eid, |ex| {
            ex.forward(&latent, cfg.situ_beta, cfg.situ_linear_beta)
        });
        for (c, o) in combined.iter_mut().zip(out.iter()) {
            *c += w * o;
        }
    }

    let normed = match &weights.routed_expert_norm_weight {
        Some(w) => rms_norm(&combined, w, cfg.rms_norm_eps),
        None => combined,
    };

    let mut out = weights.up_proj.apply(&normed);
    let shared = weights
        .shared_expert
        .forward(hidden, cfg.situ_beta, cfg.situ_linear_beta);
    for (o, s) in out.iter_mut().zip(shared.iter()) {
        *o += s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_core::tensor::Tensor;

    const HIDDEN_DIM: usize = 6;
    const MOE_HIDDEN_DIM: usize = 4;
    const MOE_INTERMEDIATE: usize = 3;
    const N_EXPERTS: usize = 4;
    const TOP_K: usize = 2;
    const SHARED_INTERMEDIATE: usize = 3; // MOE_INTERMEDIATE * num_shared(1)
    const SITU_BETA: f32 = 4.0;
    const SITU_LINEAR_BETA: f32 = 25.0;
    const NORM_EPS: f32 = 1e-5;

    fn wm(data: &[f32], rows: usize, cols: usize) -> WeightMatrix {
        assert_eq!(data.len(), rows * cols);
        WeightMatrix::F32(Tensor::new(data.to_vec(), vec![rows, cols]))
    }

    // Generated by an independent Python reference -- do not hand-edit.
    const LMOE_ROUTER_WEIGHT: [f32; 24] = [
        0.0102578, 0.407924, 0.367416, -0.153092, -0.0893909, -0.158215, 0.170918, -0.0168193,
        0.224066, -0.554197, 0.469965, -0.0289296, 0.204114, -0.0409699, -0.11373, 0.138933,
        0.247354, -0.060759, -0.0458359, 0.20571, -0.261102, -0.454315, 0.118495, -0.20117,
    ];
    const LMOE_BIAS: [f32; 4] = [-0.96017, -0.407027, -0.233799, -0.596601];
    const LMOE_DOWN_PROJ: [f32; 24] = [
        -0.447739, 0.0109913, 0.269175, -0.0699396, -0.223079, 0.115498, 0.215171, -0.0900032,
        0.1634, 0.312863, -0.0620869, -0.244055, 0.104295, 0.0742637, 0.329644, -0.385374,
        -0.198484, -0.25145, -0.520204, 0.0379304, 0.158341, -0.221637, 0.415694, 0.246577,
    ];
    const LMOE_UP_PROJ: [f32; 24] = [
        0.188213,
        0.120512,
        0.286701,
        -0.399594,
        0.184179,
        0.180833,
        -0.530316,
        0.104109,
        -0.0751264,
        0.234457,
        -0.131719,
        -0.00547226,
        0.102855,
        -0.262879,
        0.179579,
        -0.031489,
        0.147745,
        -0.156551,
        0.32586,
        0.181561,
        -0.0534075,
        0.189587,
        0.377927,
        0.537353,
    ];
    const LMOE_ROUTED_NORM_W: [f32; 4] = [0.842642, 1.08831, 1.04651, 0.990614];
    const LMOE_E0_W1: [f32; 12] = [
        -0.302, 0.377157, -0.378521, 0.170084, 0.39056, -0.479901, -0.0907554, -0.392765,
        0.0732162, 0.454313, 0.607067, -0.533434,
    ];
    const LMOE_E0_W2: [f32; 12] = [
        -0.172485,
        0.211064,
        0.473812,
        0.126363,
        -0.223846,
        0.0891395,
        -0.00498576,
        -0.0611222,
        -0.220341,
        0.116178,
        0.0923639,
        -0.0278952,
    ];
    const LMOE_E0_W3: [f32; 12] = [
        -0.0665064, -0.385475, -0.145853, 0.361935, -0.0571675, -0.431909, 0.400333, 0.15908,
        0.632426, 0.0187537, -0.138415, -0.434293,
    ];
    const LMOE_E1_W1: [f32; 12] = [
        0.397152, 0.770853, -0.24628, -0.194126, 0.178857, -0.249104, -0.0811675, -0.104951,
        0.0575877, 0.328446, 0.00662028, 0.275673,
    ];
    const LMOE_E1_W2: [f32; 12] = [
        -0.125967, 0.0983396, -0.641468, -0.434984, 0.238774, -0.177045, 0.173974, 0.162703,
        0.396684, 0.243558, 0.305097, -0.0335014,
    ];
    const LMOE_E1_W3: [f32; 12] = [
        -0.209486, -0.219468, -0.146413, -0.338949, -0.164233, -0.0277707, 0.0754836, -0.101667,
        -0.577105, -0.0216849, 0.0676037, 0.325343,
    ];
    const LMOE_E2_W1: [f32; 12] = [
        0.173359, -0.193068, -0.217133, 0.603179, 0.226991, 0.549432, 0.638833, -0.245425,
        0.115581, 0.13745, 0.167884, 0.162571,
    ];
    const LMOE_E2_W2: [f32; 12] = [
        0.0606164, 0.0522388, -0.45075, -0.0496204, -0.224286, 0.0378262, -0.140264, 0.185556,
        0.245723, 0.0926077, 0.0948504, 0.027884,
    ];
    const LMOE_E2_W3: [f32; 12] = [
        -0.134339, -0.0493504, -0.148694, 0.116396, 0.00423402, 0.174387, -0.398595, 0.266334,
        -0.228809, -0.220285, -0.0592329, -0.169015,
    ];
    const LMOE_E3_W1: [f32; 12] = [
        0.0873376, -0.172249, -0.320843, -0.253741, 0.393595, 0.0132775, -0.350259, 0.00251322,
        -0.466784, 0.53687, -0.457343, 0.143626,
    ];
    const LMOE_E3_W2: [f32; 12] = [
        0.163088, -0.435002, 0.090083, 0.299152, 0.140273, 0.0783049, 0.284729, 0.0482714,
        0.100928, -0.0380135, 0.189542, -0.282415,
    ];
    const LMOE_E3_W3: [f32; 12] = [
        0.237529, -0.15159, -0.327196, 0.109575, 0.507887, 0.288503, -0.154695, 0.209418,
        -0.136263, -0.0372058, 0.0268585, -0.695166,
    ];
    const LMOE_SHARED_W1: [f32; 18] = [
        0.0574844,
        -0.308883,
        -0.20922,
        -0.442317,
        -0.454967,
        -0.282979,
        0.247679,
        0.499812,
        -0.00756153,
        0.327537,
        -0.0791942,
        -0.573592,
        0.044975,
        0.133694,
        -0.128585,
        0.0906673,
        0.171767,
        -0.259022,
    ];
    const LMOE_SHARED_W2: [f32; 18] = [
        -0.44305, -0.0663763, -0.0633182, -0.105799, 0.296174, 0.517677, -0.124994, 0.209666,
        0.282279, 0.213947, 0.314246, -0.116773, 0.221942, 0.607345, 0.2411, -0.185968, 0.182314,
        0.378714,
    ];
    const LMOE_SHARED_W3: [f32; 18] = [
        0.110696,
        -0.16902,
        0.462924,
        -0.37493,
        0.151028,
        -0.00485267,
        -0.3581,
        0.365835,
        0.104366,
        -0.350039,
        0.180519,
        -0.12948,
        -0.570837,
        -0.206672,
        0.0790654,
        0.190711,
        0.0496695,
        0.0135495,
    ];

    const LMOE_HIDDEN: [f32; 6] = [0.234995, -0.105488, 0.588477, 0.0743208, 0.146961, 0.270399];
    const LMOE_GOLDEN_OUT: [f32; 6] = [
        0.061351, 0.0969791, 0.0499038, -0.052412, -0.0592386, -0.0464002,
    ];

    fn expert(
        w1: &[f32],
        w2: &[f32],
        w3: &[f32],
        ffn_dim: usize,
        in_dim: usize,
    ) -> KimiExpertWeights {
        KimiExpertWeights {
            w1: wm(w1, ffn_dim, in_dim),
            w2: wm(w2, in_dim, ffn_dim),
            w3: wm(w3, ffn_dim, in_dim),
        }
    }

    fn make_weights() -> KimiLatentMoeWeights {
        KimiLatentMoeWeights {
            router_weight: wm(&LMOE_ROUTER_WEIGHT, N_EXPERTS, HIDDEN_DIM),
            e_score_correction_bias: LMOE_BIAS.to_vec(),
            down_proj: wm(&LMOE_DOWN_PROJ, MOE_HIDDEN_DIM, HIDDEN_DIM),
            up_proj: wm(&LMOE_UP_PROJ, HIDDEN_DIM, MOE_HIDDEN_DIM),
            routed_expert_norm_weight: Some(LMOE_ROUTED_NORM_W.to_vec()),
            experts: KimiExpertBacking::Resident(vec![
                expert(
                    &LMOE_E0_W1,
                    &LMOE_E0_W2,
                    &LMOE_E0_W3,
                    MOE_INTERMEDIATE,
                    MOE_HIDDEN_DIM,
                ),
                expert(
                    &LMOE_E1_W1,
                    &LMOE_E1_W2,
                    &LMOE_E1_W3,
                    MOE_INTERMEDIATE,
                    MOE_HIDDEN_DIM,
                ),
                expert(
                    &LMOE_E2_W1,
                    &LMOE_E2_W2,
                    &LMOE_E2_W3,
                    MOE_INTERMEDIATE,
                    MOE_HIDDEN_DIM,
                ),
                expert(
                    &LMOE_E3_W1,
                    &LMOE_E3_W2,
                    &LMOE_E3_W3,
                    MOE_INTERMEDIATE,
                    MOE_HIDDEN_DIM,
                ),
            ]),
            shared_expert: expert(
                &LMOE_SHARED_W1,
                &LMOE_SHARED_W2,
                &LMOE_SHARED_W3,
                SHARED_INTERMEDIATE,
                HIDDEN_DIM,
            ),
        }
    }

    fn cfg() -> KimiMoeConfig {
        KimiMoeConfig {
            n_experts_active: TOP_K,
            moe_renormalize: true,
            routed_scaling_factor: 1.0,
            situ_beta: SITU_BETA,
            situ_linear_beta: SITU_LINEAR_BETA,
            rms_norm_eps: NORM_EPS,
        }
    }

    #[test]
    fn matches_independent_python_reference() {
        let weights = make_weights();
        let cfg = cfg();
        let out = kimi_latent_moe_forward(&weights, &cfg, &LMOE_HIDDEN);
        assert_eq!(out.len(), LMOE_GOLDEN_OUT.len());
        for (i, (a, b)) in out.iter().zip(LMOE_GOLDEN_OUT.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "element {i}: rust={a} python={b}");
        }
    }

    #[test]
    fn without_routed_norm_output_still_finite_and_differs() {
        let mut weights = make_weights();
        weights.routed_expert_norm_weight = None;
        let cfg = cfg();
        let out = kimi_latent_moe_forward(&weights, &cfg, &LMOE_HIDDEN);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!((out[0] - LMOE_GOLDEN_OUT[0]).abs() > 1e-6);
    }
}
