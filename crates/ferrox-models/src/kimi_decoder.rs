//! A dedicated decoder for Kimi K3's real hybrid architecture, separate
//! from `ferrox-models::decoder::Decoder` (which every GQA-only preset
//! -- GLM-5.2, DeepSeek V4 Pro, the test fixtures -- uses and which
//! `ferrox-cli`, `ferrox-server`, `prefix_cache`, and `speculative` all
//! depend on): this module composes the already-independently-tested
//! `kda`, `mla`, `latent_moe`, `block_residual`, and
//! `ferrox_core::situ_and_mul` pieces into a real forward pass, without
//! touching any of that existing, production-quality GQA code path (a
//! shared, polymorphic `Decoder` was judged too risky to attempt
//! without a way to verify it end to end).
//!
//! Real per-layer flow, transcribed from `KimiDecoderLayer.forward`'s
//! `_forward_attn_residual` path (the one Kimi K3 actually runs, since
//! `attn_res_block_size`=12 is set in its real config) in
//! `modeling_kimi_linear.py`:
//!
//! ```text
//! prefix_sum = hidden
//! blended = block_residual.is_empty() ? prefix_sum : apply_attn_res(prefix_sum, block_residual, self_attn_res_*)
//! if layer_idx % attn_res_block_size == 0 { block_residual.push(prefix_sum); prefix_sum = None }
//! attn_out = kda_or_mla(rms_norm(blended, input_layernorm))
//! prefix_sum = (prefix_sum is None) ? attn_out : prefix_sum + attn_out
//! blended2 = apply_attn_res(prefix_sum, block_residual, mlp_res_*)   // block_residual may have just grown above
//! ffn_out = dense_or_moe(rms_norm(blended2, post_attention_layernorm))
//! hidden = prefix_sum + ffn_out
//! ```
//!
//! One real, non-obvious fact confirmed by reading
//! `KimiLinearModel.forward` (not just the per-layer code): `block_residual`
//! is freshly re-initialized to *empty* inside `forward()`, i.e. once
//! per forward call -- for single-token incremental decode (what this
//! module implements), that means every decode step starts with an
//! empty `block_residual`, not a value carried over from the previous
//! token. `KimiDecodeState` therefore only needs to carry each layer's
//! own attention state (`kda::KdaState` or MLA's growable K/V buffers),
//! not any block-residual bookkeeping across positions.

use ferrox_core::matmul::{rms_norm, situ_and_mul};
use ferrox_core::tensor::Tensor;
use ferrox_core::weight_matrix::WeightMatrix;

use crate::block_residual::apply_attn_res;
use crate::kda::{self, KdaAttnWeights, KdaState};
use crate::latent_moe::{self, KimiLatentMoeWeights, KimiMoeConfig};
use crate::mla::{self, MlaAttnWeights};

pub enum KimiLayerAttention {
    Kda(Box<KdaAttnWeights>),
    Mla(Box<MlaAttnWeights>),
}

/// `KimiMLP` used directly on the full hidden dimension -- the sole
/// dense leading layer's feed-forward block (`n_dense_leading_layers`=1
/// for Kimi K3).
pub struct DenseMlpWeights {
    pub gate_proj: WeightMatrix,
    pub up_proj: WeightMatrix,
    pub down_proj: WeightMatrix,
}

impl DenseMlpWeights {
    pub(crate) fn forward(&self, x: &[f32], situ_beta: f32, situ_linear_beta: f32) -> Vec<f32> {
        let gate = self.gate_proj.apply(x);
        let up = self.up_proj.apply(x);
        let combined = situ_and_mul(&gate, &up, situ_beta, situ_linear_beta);
        self.down_proj.apply(&combined)
    }
}

pub enum KimiLayerFfn {
    Dense(Box<DenseMlpWeights>),
    Moe(Box<KimiLatentMoeWeights>),
}

pub struct KimiDecoderLayerWeights {
    pub input_layernorm_weight: Vec<f32>,
    pub attn: KimiLayerAttention,
    pub post_attention_layernorm_weight: Vec<f32>,
    pub ffn: KimiLayerFfn,
    /// Block-residual weights -- present on every real layer (confirmed
    /// against a real shard header), used twice per layer (once before
    /// attention, once before the FFN).
    pub self_attention_res_norm_weight: Vec<f32>,
    pub self_attention_res_proj_weight: Vec<f32>,
    pub mlp_res_norm_weight: Vec<f32>,
    pub mlp_res_proj_weight: Vec<f32>,
}

pub struct KimiDecoderWeights {
    pub embedding: Tensor, // [vocab_size, hidden_dim]
    pub layers: Vec<KimiDecoderLayerWeights>,
    pub output_attn_res_norm_weight: Vec<f32>,
    pub output_attn_res_proj_weight: Vec<f32>,
    pub final_norm_weight: Vec<f32>,
    pub output_head: WeightMatrix, // [vocab_size, hidden_dim]
}

impl KimiDecoderWeights {
    /// The shared expert store's live counters when this model streams
    /// routed experts -- `None` when fully resident. Every store-backed
    /// layer shares one store, so the first found speaks for the model.
    pub fn expert_store_stats(&self) -> Option<ferrox_core::expert_store::ExpertStoreStats> {
        self.layers.iter().find_map(|l| match &l.ffn {
            KimiLayerFfn::Moe(moe) => match &moe.experts {
                crate::latent_moe::KimiExpertBacking::Stored { store, .. } => Some(store.stats()),
                crate::latent_moe::KimiExpertBacking::Resident(_) => None,
            },
            KimiLayerFfn::Dense(_) => None,
        })
    }
}

pub struct KimiDecoderConfig {
    pub attn_res_block_size: usize,
    pub rms_norm_eps: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub moe: KimiMoeConfig,
}

pub enum KimiLayerState {
    Kda(KdaState),
    Mla {
        k_cache: Vec<f32>,
        v_cache: Vec<f32>,
    },
}

pub struct KimiDecodeState {
    layer_states: Vec<KimiLayerState>,
}

impl KimiDecodeState {
    /// Builds fresh (all-zero/empty) per-layer state matching each
    /// layer's real attention kind, as declared by `weights.layers`.
    pub fn new(weights: &KimiDecoderWeights, kda_cfg: &crate::config::KdaConfig) -> Self {
        let layer_states = weights
            .layers
            .iter()
            .map(|l| match &l.attn {
                KimiLayerAttention::Kda(_) => KimiLayerState::Kda(KdaState::new(kda_cfg)),
                KimiLayerAttention::Mla(_) => KimiLayerState::Mla {
                    k_cache: Vec::new(),
                    v_cache: Vec::new(),
                },
            })
            .collect();
        KimiDecodeState { layer_states }
    }
}

/// One decode step across every layer.
pub fn kimi_forward_token(
    weights: &KimiDecoderWeights,
    cfg: &KimiDecoderConfig,
    mla_cfg: &crate::config::MlaConfig,
    kda_cfg: &crate::config::KdaConfig,
    token_id: usize,
    state: &mut KimiDecodeState,
) -> Vec<f32> {
    let hidden_dim = weights.embedding.cols();
    let mut hidden = weights.embedding.row(token_id).to_vec();
    // Fresh every call -- see module doc comment for why this is real,
    // not a simplification: `KimiLinearModel.forward` re-initializes
    // `block_residual` to empty on every forward call.
    let mut block_residual: Vec<f32> = Vec::new();

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let prefix_sum_pre_blend = hidden.clone();

        let blended = if block_residual.is_empty() {
            prefix_sum_pre_blend.clone()
        } else {
            apply_attn_res(
                &prefix_sum_pre_blend,
                &block_residual,
                &layer.self_attention_res_norm_weight,
                &layer.self_attention_res_proj_weight,
                cfg.rms_norm_eps,
            )
        };

        let mut prefix_sum: Option<Vec<f32>> = Some(prefix_sum_pre_blend.clone());
        if layer_idx % cfg.attn_res_block_size == 0 {
            block_residual.extend_from_slice(&prefix_sum_pre_blend);
            prefix_sum = None;
        }

        let normed = rms_norm(&blended, &layer.input_layernorm_weight, cfg.rms_norm_eps);
        let attn_out = match (&layer.attn, &mut state.layer_states[layer_idx]) {
            (KimiLayerAttention::Kda(w), KimiLayerState::Kda(s)) => {
                kda::kda_forward_token(w, kda_cfg, &normed, cfg.rms_norm_eps, s)
            }
            (KimiLayerAttention::Mla(w), KimiLayerState::Mla { k_cache, v_cache }) => {
                mla::mla_forward_token(w, mla_cfg, &normed, cfg.rms_norm_eps, k_cache, v_cache)
            }
            _ => unreachable!("layer attention kind and decode state kind must always match"),
        };

        let mut prefix_sum = match prefix_sum {
            Some(mut ps) => {
                for (p, a) in ps.iter_mut().zip(attn_out.iter()) {
                    *p += a;
                }
                ps
            }
            None => attn_out,
        };

        let blended2 = apply_attn_res(
            &prefix_sum,
            &block_residual,
            &layer.mlp_res_norm_weight,
            &layer.mlp_res_proj_weight,
            cfg.rms_norm_eps,
        );
        let normed2 = rms_norm(
            &blended2,
            &layer.post_attention_layernorm_weight,
            cfg.rms_norm_eps,
        );
        let ffn_out = match &layer.ffn {
            KimiLayerFfn::Dense(w) => w.forward(&normed2, cfg.situ_beta, cfg.situ_linear_beta),
            KimiLayerFfn::Moe(w) => latent_moe::kimi_latent_moe_forward(w, &cfg.moe, &normed2),
        };

        for (p, f) in prefix_sum.iter_mut().zip(ffn_out.iter()) {
            *p += f;
        }
        hidden = prefix_sum;
    }

    if !block_residual.is_empty() {
        hidden = apply_attn_res(
            &hidden,
            &block_residual,
            &weights.output_attn_res_norm_weight,
            &weights.output_attn_res_proj_weight,
            cfg.rms_norm_eps,
        );
    }

    let final_normed = rms_norm(&hidden, &weights.final_norm_weight, cfg.rms_norm_eps);
    assert_eq!(final_normed.len(), hidden_dim);
    weights.output_head.apply(&final_normed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{KdaConfig, MlaConfig};
    use crate::kda::KdaAttnWeights;
    use crate::latent_moe::KimiExpertBacking;
    use crate::latent_moe::KimiExpertWeights;
    use crate::mla::MlaAttnWeights;
    use ferrox_core::tensor::Tensor;

    const HIDDEN_DIM: usize = 8;
    const EPS: f32 = 1e-5;
    const SITU_BETA: f32 = 4.0;
    const SITU_LINEAR_BETA: f32 = 25.0;
    const ATTN_RES_BLOCK_SIZE: usize = 2;

    const KDA_NUM_HEADS: usize = 2;
    const KDA_HEAD_DIM: usize = 3;
    const KDA_PROJ: usize = KDA_NUM_HEADS * KDA_HEAD_DIM;
    const KDA_CONV_SIZE: usize = 4;
    const KDA_GATE_LOWER_BOUND: f32 = -5.0;

    const DENSE_INTERMEDIATE: usize = 5;

    const MLA_NUM_HEADS: usize = 2;
    const MLA_QK_NOPE: usize = 3;
    const MLA_QK_ROPE: usize = 2;
    const MLA_KV_LORA: usize = 4;
    const MLA_Q_LORA: usize = 6;
    const MLA_V_HEAD_DIM: usize = 3;
    const MLA_PROJ: usize = MLA_NUM_HEADS * MLA_V_HEAD_DIM;

    const MOE_HIDDEN_DIM: usize = 4;
    const MOE_INTERMEDIATE: usize = 3;
    const N_EXPERTS: usize = 4;
    const TOP_K: usize = 2;
    const SHARED_INTERMEDIATE: usize = 3;
    const OUTPUT_VOCAB: usize = 5;

    fn wm(data: &[f32], rows: usize, cols: usize) -> WeightMatrix {
        assert_eq!(data.len(), rows * cols);
        WeightMatrix::F32(Tensor::new(data.to_vec(), vec![rows, cols]))
    }

    // Generated by an independent Python reference -- do not hand-edit.
    const KDA_Q_PROJ: [f32; 48] = [
        0.0332731, -0.0251273, -0.241248, -0.645647, 0.363556, -0.144589, -0.058427, -0.264839,
        -0.174983, -0.313658, -0.0222095, 0.0329603, -0.0773659, 0.179789, -0.428436, 0.272643,
        -0.29953, -0.124363, -0.270533, -0.15906, -0.136326, 0.138484, 0.728104, -0.306624,
        -0.0415019, -0.62199, -0.0334468, 0.0501355, 0.374942, 0.438693, 0.398001, 0.0660859,
        0.146161, -0.129336, 0.701423, -0.0954704, -0.37886, -0.0410528, 0.653784, 0.193832,
        0.451906, 0.367486, 0.0999694, -0.498877, 0.495604, -0.203314, -0.326699, -0.410713,
    ];
    const KDA_K_PROJ: [f32; 48] = [
        -0.421999, -0.0844493, 0.0366557, -0.373529, -0.320309, -0.171037, -0.208623, -0.251102,
        -0.384313, -0.242582, -0.708695, -0.0410055, -0.163735, 0.045339, 0.0596934, -0.462932,
        0.344817, 0.314063, 0.183688, -0.215147, 0.0682644, -0.119572, 0.24693, -0.171698,
        -0.427691, -0.311782, -0.243235, 0.168478, -0.322684, -0.0423794, 0.296734, -0.106622,
        0.0607117, 0.900444, 0.214488, 0.390483, 0.0470901, -0.084555, -0.509912, 0.273387,
        -0.0251134, 0.459143, 0.34484, -0.102602, -0.315908, -0.311536, -0.154464, -0.067087,
    ];
    const KDA_V_PROJ: [f32; 48] = [
        0.106298, -0.296877, -0.441024, 0.346085, -0.211878, 0.208265, 0.286445, 0.254616,
        0.527007, 0.0238059, 0.0520139, -0.0428818, 0.547271, 0.0641891, 0.406677, -0.159912,
        0.315001, 0.090456, 0.665635, 0.0849947, 0.332725, 0.0468494, -0.340192, -0.57511,
        -0.199206, 0.00843515, 0.307353, -0.341596, -0.188955, -0.0262392, 0.711621, -0.12957,
        0.0589482, 0.6676, -0.0487213, 0.318632, -0.129497, 0.45287, 0.312848, 0.404807, 0.300039,
        -0.111872, 0.00619566, 0.337312, 0.203499, -0.615727, -0.426802, 0.00709124,
    ];
    const KDA_Q_CONV_W: [f32; 24] = [
        0.232949, -0.349125, -0.144423, -1.17421, 0.392014, -0.0551284, -0.188107, 0.34828,
        0.182368, -1.15816, 0.29745, -0.163951, -0.785389, -0.0799616, 0.0853995, 0.159346,
        0.438013, 0.16794, 0.11068, 0.222541, 0.27932, 0.584781, 0.0801223, 0.315161,
    ];
    const KDA_K_CONV_W: [f32; 24] = [
        0.0811582, 0.5723, 0.275913, 0.337176, 0.30982, -0.260437, -0.368242, 0.355531, -0.940038,
        -0.0380444, 0.559686, -0.632208, 0.0297131, -0.246906, 0.435212, -0.65012, -0.207435,
        -0.259419, -0.219341, 0.426728, -0.0195621, -0.164165, 0.231709, 0.363266,
    ];
    const KDA_V_CONV_W: [f32; 24] = [
        0.174442,
        0.16448,
        -0.11751,
        -0.00693412,
        0.22015,
        0.1184,
        0.59284,
        -0.100545,
        0.904578,
        -0.057162,
        -0.185486,
        -0.0155876,
        0.239106,
        -0.184512,
        0.887074,
        0.319565,
        0.518244,
        0.238721,
        -0.0932803,
        -0.614529,
        0.400545,
        0.458996,
        -0.217588,
        0.0540083,
    ];
    const KDA_A_LOG: [f32; 2] = [2.57153, 1.68862];
    const KDA_F_A_PROJ: [f32; 24] = [
        0.293008, 0.208071, -0.0458425, 0.26919, -0.0604267, -0.414273, -0.347327, 0.435059,
        -0.0875413, 0.260375, 0.78432, -0.0103574, 0.129559, 0.522936, 0.205997, 0.235916,
        0.129708, -0.515258, -0.251447, 0.0114761, -0.0608938, -0.142939, 0.0391045, -0.106787,
    ];
    const KDA_F_B_PROJ: [f32; 18] = [
        0.258232, 0.0799449, -0.128173, 0.374093, 0.0249372, 0.564439, 0.291732, -0.673602,
        0.300492, -0.312056, -0.0782346, -0.560855, 0.319259, -0.194854, -0.109685, -0.295024,
        -0.0275032, -0.287207,
    ];
    const KDA_DT_BIAS: [f32; 6] = [0.406797, 0.44378, -0.118682, -0.101149, 0.144153, 0.100643];
    const KDA_B_PROJ: [f32; 16] = [
        -0.557976, 0.101854, -0.529571, 0.437377, 0.282072, 0.0808985, 0.119817, -0.234031,
        -0.114455, -0.182255, 0.492746, -0.213926, -0.0890158, 0.520648, -0.0288543, 0.378583,
    ];
    const KDA_G_PROJ: [f32; 48] = [
        0.115836, 0.257017, -0.28702, 0.0285552, 0.0241214, -0.586101, -0.284745, -0.1296,
        0.369078, -0.275476, 0.352622, 0.235832, -0.13984, 0.230448, 0.440173, 0.224343, 0.103446,
        0.108157, -0.523653, -0.674568, 0.130624, 0.228334, -0.493531, 0.257466, -0.0398756,
        -0.385554, 0.209178, -0.414277, -0.364987, 0.515843, -0.486846, -0.128941, -0.0565001,
        0.480469, -0.0731606, 0.279074, -0.16879, -0.589812, 0.200213, -0.3698, 0.355666, 0.269551,
        0.450168, -0.0689023, -0.234073, -0.81206, 0.250412, 0.0890739,
    ];
    const KDA_O_NORM_W: [f32; 3] = [0.970175, 0.902756, 1.03511];
    const KDA_O_PROJ: [f32; 48] = [
        -0.148986, -0.157794, 0.225606, -0.302445, -0.0587451, -0.235495, -0.541087, -0.270422,
        0.604319, 0.371634, 0.144359, 0.409312, 0.108881, -0.352629, -0.129886, -0.258043,
        -0.168576, -0.241786, 0.0283778, -0.097169, -0.0738384, 0.102603, 0.302548, 0.219944,
        0.144096, -0.521365, -0.354098, -0.79218, -0.227326, -0.778447, -0.221429, -0.0319402,
        0.377724, 0.388695, 0.560278, 0.420781, -0.440711, 0.275919, 0.189938, 0.159768, 0.341554,
        0.594513, 0.333733, -0.0974569, -0.308295, 0.200531, 0.101979, -0.363293,
    ];
    const DENSE_GATE_PROJ: [f32; 40] = [
        0.0530662,
        -0.0224611,
        -0.468284,
        0.394925,
        -0.0289055,
        -0.138464,
        0.302004,
        -0.0823999,
        0.295465,
        0.391958,
        0.197625,
        0.236802,
        -0.000834978,
        0.539451,
        0.230436,
        0.0369415,
        0.235552,
        -0.0302823,
        0.216731,
        0.106894,
        -0.0388429,
        -0.0235654,
        0.0683206,
        -0.247831,
        -0.829944,
        -0.255354,
        -0.652798,
        -0.539486,
        -0.16364,
        0.212245,
        -0.263695,
        0.266948,
        0.474532,
        0.249119,
        -0.0215431,
        0.0568342,
        -0.0727135,
        -0.267156,
        0.0426981,
        -0.921182,
    ];
    const DENSE_UP_PROJ: [f32; 40] = [
        -0.236304,
        0.180069,
        -0.2151,
        0.222938,
        -0.0463798,
        -0.31709,
        -0.00577976,
        0.254963,
        -0.152982,
        0.736692,
        0.565269,
        -0.00259989,
        0.22078,
        0.0674972,
        0.193022,
        0.626185,
        0.0697913,
        0.489142,
        -0.0709509,
        0.135837,
        0.0604967,
        0.134483,
        0.15545,
        -0.715912,
        0.131568,
        0.447037,
        0.0964134,
        -0.568027,
        -0.539577,
        -0.385414,
        -0.308715,
        0.338456,
        -0.589399,
        -0.22471,
        -0.256634,
        -0.172552,
        0.229815,
        -0.232763,
        -0.15492,
        0.112469,
    ];
    const DENSE_DOWN_PROJ: [f32; 40] = [
        0.364035, -0.722506, -0.192076, -0.513796, 0.358074, 0.187812, -0.210479, -0.190417,
        0.0378089, 0.398016, 0.393775, 0.119131, 0.0944593, 0.160586, -0.302192, -0.0174533,
        -0.0131564, 0.17226, -0.0443057, -0.354608, 0.221662, 0.333016, -0.0158993, -0.0357403,
        -0.251197, 0.0164112, 0.210909, 0.292464, 0.201202, -0.121079, 0.407385, 0.275425,
        -0.537043, -0.0175377, -0.428206, 0.253178, 0.404171, -0.231798, 0.168481, -0.222966,
    ];
    const MLA_Q_A_PROJ: [f32; 48] = [
        0.451391, -0.126339, 0.514525, 0.104224, -0.482352, -0.12592, -0.140897, 0.570421,
        0.177444, 0.0637828, -0.219946, -0.449056, -0.0832418, -0.0932411, 0.141219, -0.688717,
        0.311497, -0.0927841, 0.0286042, 0.608332, -0.340392, 0.292946, 0.390898, -0.0359569,
        -0.157069, -0.194805, 0.595659, -0.129717, 0.435375, 0.707992, -0.267707, -0.446994,
        0.257943, 0.208935, 0.0124412, 0.0539018, 0.40146, -0.144905, -0.115103, -0.432397,
        -0.205624, -0.596541, 0.485434, 0.194575, -0.132356, -0.194361, -0.361851, -0.109006,
    ];
    const MLA_Q_A_NORM_W: [f32; 6] = [0.969939, 0.841643, 1.00562, 1.08525, 0.926626, 0.990575];
    const MLA_Q_B_PROJ: [f32; 60] = [
        0.1374,
        -0.135647,
        0.0887308,
        -0.0979333,
        -0.259915,
        -0.0779051,
        0.115564,
        0.228723,
        -0.230383,
        0.083073,
        0.192752,
        -0.0369039,
        -0.286092,
        -0.0686448,
        -0.307404,
        -0.0636238,
        0.211306,
        -0.146429,
        -0.0687791,
        0.250252,
        -0.156639,
        0.337944,
        0.178598,
        -0.616342,
        -0.0385348,
        0.382634,
        0.671192,
        -0.248981,
        0.121872,
        0.275898,
        0.188452,
        0.828256,
        -0.438505,
        0.621337,
        0.408834,
        -0.303079,
        0.0797437,
        -0.240968,
        0.0335185,
        0.602176,
        0.461823,
        -0.23303,
        0.0937693,
        -0.0384911,
        0.173248,
        -0.515262,
        0.0811063,
        0.543123,
        -0.148281,
        -0.599126,
        0.302461,
        -0.417772,
        0.064963,
        -0.000217146,
        0.188863,
        0.752821,
        -0.293606,
        0.169847,
        0.739423,
        0.435163,
    ];
    const MLA_KV_A_PROJ: [f32; 48] = [
        -0.421765, -0.232617, 0.203753, 0.166609, -0.045513, -0.0101532, 0.0955488, -0.165861,
        -0.0260061, -0.0220384, -0.0330121, -0.158364, -0.0111785, -0.345142, 0.755459, -0.0558389,
        -0.277054, -0.26961, 0.240628, 0.0947741, -0.167247, -0.23363, -0.157332, 0.345719,
        0.156487, 0.0865676, -0.192661, -0.214795, -0.567426, -0.0837593, -0.324576, 0.310762,
        -0.314536, -0.68736, -0.0154466, 0.315627, -0.150268, -0.0287802, 0.356274, -0.246495,
        -0.0825744, 0.204116, 0.559475, 0.640863, 0.456519, 0.173669, 0.185071, -0.115482,
    ];
    const MLA_KV_A_NORM_W: [f32; 4] = [1.10413, 1.0107, 0.871313, 1.16755];
    const MLA_KV_B_PROJ: [f32; 48] = [
        0.0866448,
        -0.423795,
        0.114522,
        -0.441771,
        -0.186988,
        0.0366587,
        0.114137,
        -0.0609123,
        0.199646,
        0.722449,
        0.704719,
        -0.460906,
        0.0629041,
        0.485761,
        0.328924,
        0.0319879,
        -0.298518,
        0.123369,
        -0.136789,
        -0.256657,
        0.131407,
        0.435331,
        0.0737087,
        -0.130659,
        0.130558,
        0.113592,
        -0.0490255,
        -0.292942,
        -0.0925113,
        -0.412935,
        -0.222763,
        -0.189368,
        0.202786,
        -0.128307,
        -0.215928,
        0.0669919,
        0.384514,
        0.637332,
        -0.0592113,
        0.0464926,
        -0.281259,
        0.107265,
        0.494055,
        -0.00978826,
        -0.398057,
        0.593258,
        0.0966843,
        0.013154,
    ];
    const MLA_O_PROJ: [f32; 48] = [
        -0.384195,
        0.195792,
        0.334816,
        -0.101664,
        0.103024,
        -0.142412,
        0.130369,
        -0.307274,
        -0.367605,
        0.559024,
        -0.128535,
        0.0330137,
        0.241127,
        0.0644755,
        -0.0776874,
        -0.231983,
        -0.909896,
        -0.268289,
        0.205435,
        -0.268665,
        -0.663843,
        0.00206543,
        -0.454986,
        0.226469,
        -0.539808,
        0.34166,
        7.24295e-05,
        0.339034,
        -0.244381,
        0.168123,
        -0.316009,
        0.00653252,
        0.107088,
        0.229998,
        0.279935,
        0.318718,
        -0.262585,
        0.528549,
        0.356373,
        0.217004,
        0.154669,
        -0.193219,
        -0.0690429,
        0.304651,
        0.554123,
        0.158962,
        0.525434,
        -0.20413,
    ];
    const MLA_G_PROJ: [f32; 48] = [
        0.378873, -0.362289, -0.172782, -0.305609, -0.15234, 0.13899, -0.358599, -0.786074,
        -0.168818, 0.168034, -0.27184, -0.103517, 0.34245, 0.0531821, 0.254431, -0.164725,
        -0.0816316, -0.0860996, -0.0865057, 0.0241667, 0.259761, 0.230357, -0.726831, 0.379549,
        0.171985, -0.378427, -0.199484, 0.138928, -0.238634, -0.55968, -0.405272, -0.184117,
        -0.184866, 0.0497178, 0.103438, 0.380376, 0.364498, -0.175124, 0.293164, 0.200305,
        -0.0883711, -0.155664, -0.011865, 0.304442, -0.0586883, -0.269584, -0.215651, 0.0801283,
    ];
    const MOE_ROUTER_WEIGHT: [f32; 32] = [
        -0.14075, 0.348724, 0.162851, 0.220414, -0.334635, -0.0441166, 0.291079, -0.16397,
        0.104033, 0.186523, -0.0732253, -0.18498, 0.0548991, 0.0572118, -0.212319, -0.114626,
        0.0438156, -0.291582, -0.364393, -0.456047, -0.153299, -0.175468, 0.0600189, -0.433395,
        -0.241496, -0.195144, 0.468446, -0.0704137, 0.0423066, 0.0625186, 0.125581, -0.384111,
    ];
    const MOE_BIAS: [f32; 4] = [-0.287968, -0.506168, -0.0823805, -1.16076];
    const MOE_DOWN_PROJ: [f32; 32] = [
        0.357856, -0.419139, 0.0951717, 0.486966, 0.138318, -0.321581, 0.023647, -0.287115,
        -0.218501, 0.466282, -0.674924, 0.358341, -0.651141, -0.190237, 0.154624, -0.131574,
        -0.0769183, 0.0423567, 0.768159, 0.659497, -0.106854, -0.338984, -0.0425253, -0.326632,
        -0.157781, -0.0021155, 0.259854, 0.132918, 0.305795, 0.418168, 0.0538999, 0.630383,
    ];
    const MOE_UP_PROJ: [f32; 32] = [
        0.0537259, -0.222431, 0.0517936, -0.328772, -0.218991, 0.0109666, -0.66681, 0.53133,
        -0.24925, -0.271987, -0.57234, 0.0120699, -0.288328, 0.131623, -0.132867, -0.105812,
        -0.284854, -0.456721, -0.150868, -0.102689, -0.245856, -0.0891634, 0.389166, 0.518893,
        -0.0414535, 0.157239, -0.0155488, 0.207036, 0.00431687, -0.261777, -0.541271, 0.211408,
    ];
    const MOE_ROUTED_NORM_W: [f32; 4] = [1.14805, 0.951891, 0.971543, 0.908736];
    const MOE_SHARED_W1: [f32; 24] = [
        -0.377193,
        0.27978,
        -0.0102447,
        0.229219,
        -0.380455,
        0.552381,
        -0.0980754,
        0.56176,
        -0.297,
        -0.142547,
        -0.204375,
        -0.338748,
        0.388348,
        0.489957,
        0.356719,
        0.0574536,
        0.110144,
        -0.00856218,
        0.426843,
        0.260044,
        0.148431,
        -0.184063,
        -0.342297,
        -0.0762749,
    ];
    const MOE_SHARED_W2: [f32; 24] = [
        -0.43575, 0.0521214, -0.10101, 0.072215, 0.337606, 0.0963424, -0.107468, -0.49745,
        0.364127, 0.217258, 0.359295, -0.0454641, -0.252116, 0.0389652, 0.178223, 0.0836014,
        0.124572, -0.252946, -0.227852, 0.0663797, 0.110113, 0.458224, 0.520268, -0.0199504,
    ];
    const MOE_SHARED_W3: [f32; 24] = [
        0.368003,
        -0.483325,
        -0.379792,
        0.100306,
        0.106976,
        -0.0896776,
        -0.316544,
        0.322652,
        -0.517715,
        0.249498,
        0.0612062,
        -0.601994,
        -0.193968,
        0.558544,
        -0.388051,
        -0.277637,
        0.122792,
        -0.185875,
        -0.00133169,
        0.0236871,
        -0.296105,
        0.0705485,
        -0.302294,
        -0.246982,
    ];
    const MOE_E0_W1: [f32; 12] = [
        0.0440567, 0.174659, -0.355788, -0.234582, 0.238399, 0.600469, -0.130771, 0.811262,
        0.130333, -0.0412095, -0.714084, 0.158573,
    ];
    const MOE_E0_W2: [f32; 12] = [
        -0.103864, -0.115033, 0.275029, -0.0808199, -0.348501, 0.30532, 0.310035, -0.0757431,
        -0.132877, 0.711211, 0.226243, 0.163931,
    ];
    const MOE_E0_W3: [f32; 12] = [
        -0.114776, 0.141509, 0.504018, -0.115836, 0.43992, 0.131565, -0.469145, 0.0490231,
        0.151157, -0.0375621, -0.17218, 0.231705,
    ];
    const MOE_E1_W1: [f32; 12] = [
        0.0488396,
        -0.341746,
        0.0418606,
        0.221871,
        -0.722022,
        -0.00220993,
        -0.0916067,
        -0.398679,
        -0.423551,
        -0.645019,
        -0.27434,
        0.0208012,
    ];
    const MOE_E1_W2: [f32; 12] = [
        -0.167076, 0.231248, -0.759916, 0.352397, -0.0142662, -0.24585, -0.0941462, 0.0286856,
        0.103257, -0.272298, 0.177667, -0.226165,
    ];
    const MOE_E1_W3: [f32; 12] = [
        0.189908, -0.207244, -0.0267815, 0.0907185, 0.191764, 0.105395, 0.0162554, -0.452242,
        0.430473, -0.291964, -0.261698, 0.0512003,
    ];
    const MOE_E2_W1: [f32; 12] = [
        -0.0189576, 0.0668481, 0.214585, 0.143547, -0.459021, 0.191771, -0.386651, 0.519018,
        -0.38456, -0.409715, 0.101121, -0.114501,
    ];
    const MOE_E2_W2: [f32; 12] = [
        0.329286, -0.122696, 0.385045, -0.309553, 0.205889, -0.0737322, 0.0664255, 0.243589,
        -0.101208, 0.615374, 0.791347, -0.288181,
    ];
    const MOE_E2_W3: [f32; 12] = [
        -0.34824, -0.0969774, -0.0404219, -0.142054, -0.140162, 0.168913, 0.329251, 0.568559,
        0.302453, 0.2389, 0.10173, 0.26265,
    ];
    const MOE_E3_W1: [f32; 12] = [
        -0.270246, -0.148207, 0.301809, 0.317732, 0.216907, 0.245138, 0.390181, -0.120865,
        0.115518, -0.398155, -0.0685247, -0.263379,
    ];
    const MOE_E3_W2: [f32; 12] = [
        -0.0315493, -0.147451, -0.471437, -0.149953, -0.0759736, -0.42393, 0.500115, 0.50571,
        -0.126137, -0.0207527, 0.0606406, -0.138175,
    ];
    const MOE_E3_W3: [f32; 12] = [
        0.0539619, -0.604005, -0.229171, 0.432303, 0.240478, 0.157986, 0.482184, -0.0937539,
        0.255604, 0.304551, -0.0564262, -0.17754,
    ];
    const L0_SELF_ATTN_RES_NORM_W: [f32; 8] = [
        1.09202, 0.89273, 1.09569, 0.903039, 0.86568, 0.985966, 0.904269, 0.901858,
    ];
    const L0_SELF_ATTN_RES_PROJ_W: [f32; 8] = [
        0.181373, -0.268475, -0.278623, -0.0176931, 0.15743, -0.189069, -0.0939655, 0.0588636,
    ];
    const L0_MLP_RES_NORM_W: [f32; 8] = [
        1.01577, 1.04681, 0.991541, 1.00423, 0.89548, 1.09467, 1.05601, 0.968893,
    ];
    const L0_MLP_RES_PROJ_W: [f32; 8] = [
        -0.388332, -0.132232, 0.21217, -0.337705, 0.246598, -0.20951, 0.154862, -0.225175,
    ];
    const L0_INPUT_LAYERNORM_W: [f32; 8] = [
        0.970917, 0.970733, 0.946098, 1.23423, 0.883455, 1.12311, 0.809911, 1.12998,
    ];
    const L0_POST_ATTN_LAYERNORM_W: [f32; 8] = [
        0.986205, 1.04826, 1.00761, 1.04983, 0.890071, 0.817534, 1.17858, 0.915493,
    ];
    const L1_SELF_ATTN_RES_NORM_W: [f32; 8] = [
        0.985561, 0.917139, 1.10889, 1.02053, 0.951296, 1.08218, 1.05087, 0.816958,
    ];
    const L1_SELF_ATTN_RES_PROJ_W: [f32; 8] = [
        -0.13164, -0.436846, -0.109316, 0.0236738, 0.0868917, 0.038431, 0.12599, 0.680195,
    ];
    const L1_MLP_RES_NORM_W: [f32; 8] = [
        1.123, 1.03367, 1.15067, 1.06599, 1.10144, 0.905083, 0.911655, 1.11872,
    ];
    const L1_MLP_RES_PROJ_W: [f32; 8] = [
        -0.0917493, -0.258613, 0.62677, 0.0330113, 0.0841344, -0.332195, -0.355091, 0.152981,
    ];
    const L1_INPUT_LAYERNORM_W: [f32; 8] = [
        0.878093, 0.911769, 1.15303, 1.1774, 0.961136, 1.01693, 0.935918, 1.08085,
    ];
    const L1_POST_ATTN_LAYERNORM_W: [f32; 8] = [
        0.982172, 0.868114, 0.876461, 1.00153, 1.10216, 1.01644, 0.784298, 1.0521,
    ];
    const OUTPUT_ATTN_RES_NORM_W: [f32; 8] = [
        0.978339, 0.909439, 1.00794, 0.900698, 1.11013, 1.13568, 1.00689, 1.06737,
    ];
    const OUTPUT_ATTN_RES_PROJ_W: [f32; 8] = [
        -0.0780474, 0.195513, -0.118552, 0.46788, 0.285518, -0.16824, -0.350209, -0.0767004,
    ];
    const FINAL_NORM_W: [f32; 8] = [
        0.969882, 0.931832, 1.10373, 1.16478, 0.891072, 1.0772, 0.85093, 0.91487,
    ];
    const EMBEDDING_ROW: [f32; 8] = [
        0.309738, 0.0378015, -0.337758, 0.323415, 0.162627, -0.685942, 0.134525, 0.42463,
    ];
    const OUTPUT_HEAD: [f32; 40] = [
        0.04209, 0.261934, -0.329344, 0.0625128, 0.272985, 0.290578, 0.0418593, 0.216545, -0.63005,
        -0.246383, 0.0668848, -0.179091, -0.277674, 0.285747, 0.0959162, -0.0944821, -0.542151,
        0.2749, 0.299618, -0.0826765, 0.136924, -0.193359, -0.132659, 0.306145, 0.0721566,
        0.0321218, -0.360207, -0.152256, -0.779984, 0.286993, -0.0684097, 0.142516, -0.124933,
        0.180322, 0.00527221, 0.0710325, 0.453557, -0.152982, 0.0701835, 0.314973,
    ];
    const GOLDEN_LOGITS: [f32; 5] = [0.487583, -1.04021, 0.695218, -0.214018, 1.12848];

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

    fn make_weights() -> KimiDecoderWeights {
        let layer0 = KimiDecoderLayerWeights {
            input_layernorm_weight: L0_INPUT_LAYERNORM_W.to_vec(),
            attn: KimiLayerAttention::Kda(Box::new(KdaAttnWeights {
                q_proj: wm(&KDA_Q_PROJ, KDA_PROJ, HIDDEN_DIM),
                k_proj: wm(&KDA_K_PROJ, KDA_PROJ, HIDDEN_DIM),
                v_proj: wm(&KDA_V_PROJ, KDA_PROJ, HIDDEN_DIM),
                q_conv_weight: KDA_Q_CONV_W.to_vec(),
                k_conv_weight: KDA_K_CONV_W.to_vec(),
                v_conv_weight: KDA_V_CONV_W.to_vec(),
                a_log: KDA_A_LOG.to_vec(),
                f_a_proj: wm(&KDA_F_A_PROJ, KDA_HEAD_DIM, HIDDEN_DIM),
                f_b_proj: wm(&KDA_F_B_PROJ, KDA_PROJ, KDA_HEAD_DIM),
                dt_bias: KDA_DT_BIAS.to_vec(),
                b_proj: wm(&KDA_B_PROJ, KDA_NUM_HEADS, HIDDEN_DIM),
                g_proj: wm(&KDA_G_PROJ, KDA_PROJ, HIDDEN_DIM),
                o_norm_weight: KDA_O_NORM_W.to_vec(),
                o_proj: wm(&KDA_O_PROJ, HIDDEN_DIM, KDA_PROJ),
            })),
            post_attention_layernorm_weight: L0_POST_ATTN_LAYERNORM_W.to_vec(),
            ffn: KimiLayerFfn::Dense(Box::new(DenseMlpWeights {
                gate_proj: wm(&DENSE_GATE_PROJ, DENSE_INTERMEDIATE, HIDDEN_DIM),
                up_proj: wm(&DENSE_UP_PROJ, DENSE_INTERMEDIATE, HIDDEN_DIM),
                down_proj: wm(&DENSE_DOWN_PROJ, HIDDEN_DIM, DENSE_INTERMEDIATE),
            })),
            self_attention_res_norm_weight: L0_SELF_ATTN_RES_NORM_W.to_vec(),
            self_attention_res_proj_weight: L0_SELF_ATTN_RES_PROJ_W.to_vec(),
            mlp_res_norm_weight: L0_MLP_RES_NORM_W.to_vec(),
            mlp_res_proj_weight: L0_MLP_RES_PROJ_W.to_vec(),
        };

        let layer1 = KimiDecoderLayerWeights {
            input_layernorm_weight: L1_INPUT_LAYERNORM_W.to_vec(),
            attn: KimiLayerAttention::Mla(Box::new(MlaAttnWeights {
                q_a_proj: wm(&MLA_Q_A_PROJ, MLA_Q_LORA, HIDDEN_DIM),
                q_a_layernorm: MLA_Q_A_NORM_W.to_vec(),
                q_b_proj: wm(
                    &MLA_Q_B_PROJ,
                    MLA_NUM_HEADS * (MLA_QK_NOPE + MLA_QK_ROPE),
                    MLA_Q_LORA,
                ),
                kv_a_proj_with_mqa: wm(&MLA_KV_A_PROJ, MLA_KV_LORA + MLA_QK_ROPE, HIDDEN_DIM),
                kv_a_layernorm: MLA_KV_A_NORM_W.to_vec(),
                kv_b_proj: wm(
                    &MLA_KV_B_PROJ,
                    MLA_NUM_HEADS * (MLA_QK_NOPE + MLA_V_HEAD_DIM),
                    MLA_KV_LORA,
                ),
                o_proj: wm(&MLA_O_PROJ, HIDDEN_DIM, MLA_PROJ),
                g_proj: Some(wm(&MLA_G_PROJ, MLA_PROJ, HIDDEN_DIM)),
            })),
            post_attention_layernorm_weight: L1_POST_ATTN_LAYERNORM_W.to_vec(),
            ffn: KimiLayerFfn::Moe(Box::new(KimiLatentMoeWeights {
                router_weight: wm(&MOE_ROUTER_WEIGHT, N_EXPERTS, HIDDEN_DIM),
                e_score_correction_bias: MOE_BIAS.to_vec(),
                down_proj: wm(&MOE_DOWN_PROJ, MOE_HIDDEN_DIM, HIDDEN_DIM),
                up_proj: wm(&MOE_UP_PROJ, HIDDEN_DIM, MOE_HIDDEN_DIM),
                routed_expert_norm_weight: Some(MOE_ROUTED_NORM_W.to_vec()),
                experts: KimiExpertBacking::Resident(vec![
                    expert(
                        &MOE_E0_W1,
                        &MOE_E0_W2,
                        &MOE_E0_W3,
                        MOE_INTERMEDIATE,
                        MOE_HIDDEN_DIM,
                    ),
                    expert(
                        &MOE_E1_W1,
                        &MOE_E1_W2,
                        &MOE_E1_W3,
                        MOE_INTERMEDIATE,
                        MOE_HIDDEN_DIM,
                    ),
                    expert(
                        &MOE_E2_W1,
                        &MOE_E2_W2,
                        &MOE_E2_W3,
                        MOE_INTERMEDIATE,
                        MOE_HIDDEN_DIM,
                    ),
                    expert(
                        &MOE_E3_W1,
                        &MOE_E3_W2,
                        &MOE_E3_W3,
                        MOE_INTERMEDIATE,
                        MOE_HIDDEN_DIM,
                    ),
                ]),
                shared_expert: expert(
                    &MOE_SHARED_W1,
                    &MOE_SHARED_W2,
                    &MOE_SHARED_W3,
                    SHARED_INTERMEDIATE,
                    HIDDEN_DIM,
                ),
            })),
            self_attention_res_norm_weight: L1_SELF_ATTN_RES_NORM_W.to_vec(),
            self_attention_res_proj_weight: L1_SELF_ATTN_RES_PROJ_W.to_vec(),
            mlp_res_norm_weight: L1_MLP_RES_NORM_W.to_vec(),
            mlp_res_proj_weight: L1_MLP_RES_PROJ_W.to_vec(),
        };

        KimiDecoderWeights {
            embedding: Tensor::new(EMBEDDING_ROW.to_vec(), vec![1, HIDDEN_DIM]),
            layers: vec![layer0, layer1],
            output_attn_res_norm_weight: OUTPUT_ATTN_RES_NORM_W.to_vec(),
            output_attn_res_proj_weight: OUTPUT_ATTN_RES_PROJ_W.to_vec(),
            final_norm_weight: FINAL_NORM_W.to_vec(),
            output_head: wm(&OUTPUT_HEAD, OUTPUT_VOCAB, HIDDEN_DIM),
        }
    }

    fn decoder_cfg() -> KimiDecoderConfig {
        KimiDecoderConfig {
            attn_res_block_size: ATTN_RES_BLOCK_SIZE,
            rms_norm_eps: EPS,
            situ_beta: SITU_BETA,
            situ_linear_beta: SITU_LINEAR_BETA,
            moe: KimiMoeConfig {
                n_experts_active: TOP_K,
                moe_renormalize: true,
                routed_scaling_factor: 1.0,
                situ_beta: SITU_BETA,
                situ_linear_beta: SITU_LINEAR_BETA,
                rms_norm_eps: EPS,
            },
        }
    }

    fn mla_cfg() -> MlaConfig {
        MlaConfig {
            num_heads: MLA_NUM_HEADS,
            q_lora_rank: MLA_Q_LORA,
            kv_lora_rank: MLA_KV_LORA,
            qk_nope_head_dim: MLA_QK_NOPE,
            qk_rope_head_dim: MLA_QK_ROPE,
            v_head_dim: MLA_V_HEAD_DIM,
            use_output_gate: true,
            rope: None,
        }
    }

    fn kda_cfg() -> KdaConfig {
        KdaConfig {
            num_heads: KDA_NUM_HEADS,
            head_dim: KDA_HEAD_DIM,
            short_conv_kernel_size: KDA_CONV_SIZE,
            gate_lower_bound: KDA_GATE_LOWER_BOUND,
            use_full_rank_gate: true,
        }
    }

    /// One Kimi K3 decode step enters the CPU worker pool once.
    ///
    /// Kimi has no checkpoint this machine can load, so the instance is
    /// the same synthetic two-layer stack every other test in this file
    /// uses. The count being asserted does not depend on the weights:
    /// it is how many times the driving thread crossed rayon's cold
    /// submission path, which before `engine/entry.rs` was once per
    /// parallel region and is now once per step.
    ///
    /// Sabotage: drop the `par::on_workers` from
    /// `Engine::forward_token` and this goes red with the region count.
    #[test]
    fn one_kimi_decode_step_enters_the_pool_once() {
        crate::engine::assert_one_pool_entry_per_step(
            &crate::KimiEngine {
                weights: make_weights(),
                cfg: decoder_cfg(),
                mla_cfg: mla_cfg(),
                kda_cfg: kda_cfg(),
            },
            0,
        );
    }

    #[test]
    fn two_mixed_layers_match_independent_python_reference() {
        let weights = make_weights();
        let cfg = decoder_cfg();
        let mla_cfg = mla_cfg();
        let kda_cfg = kda_cfg();
        let mut state = KimiDecodeState::new(&weights, &kda_cfg);

        let logits = kimi_forward_token(&weights, &cfg, &mla_cfg, &kda_cfg, 0, &mut state);
        assert_eq!(logits.len(), GOLDEN_LOGITS.len());
        for (i, (a, b)) in logits.iter().zip(GOLDEN_LOGITS.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "logit {i}: rust={a} python={b}");
        }
    }
}
