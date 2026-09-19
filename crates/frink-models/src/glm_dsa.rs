//! GLM-5.2's real DSA (DeepSeek Sparse Attention) attention layer:
//! RoPE-carrying MLA (`frink_models::mla`'s math, inlined here rather
//! than reused directly -- see below) plus the lightning indexer
//! (`frink_core::attention::lightning_indexer_topk`) selecting which
//! causal positions are visible, then sparse attention restricted to
//! exactly those (`frink_core::attention::causal_mla_attention_sparse`).
//!
//! Real tensor names/shapes/dispatch confirmed against llama.cpp PR
//! #23346 (DeepSeek-V3.2, `src/models/deepseek32.cpp`) and PR #25407
//! (GLM-5.2's `indexer_types`/interleaved-RoPE diff on top,
//! `src/models/glm-dsa.cpp`), both fetched live and read line-by-line
//! (`gh api -H "Accept: application/vnd.github.raw"
//! repos/ggerganov/llama.cpp/contents/src/models/glm-dsa.cpp`, since
//! `gh pr diff` alone doesn't show unchanged context for tensor
//! creation that predates PR #25407) -- see docs/MODELS.md.
//!
//! Not the same weight layout as `frink_models::mla::MlaAttnWeights`:
//! GLM-5.2's real GGUF main-attention K/V decompression uses separate
//! per-head `wk_b`/`wv_b` 3D tensors (`blk.N.attn_k_b`/`attn_v_b`), not
//! Kimi K3's combined `kv_b_proj` -- see [`Glm52AttnWeights`]'s doc
//! comment for the "absorbed vs. un-absorbed" direction this matters
//! for. That's why this is a new set of weight/forward structures
//! rather than a reuse of `mla::MlaAttnWeights`/`mla_forward_token`,
//! even though the underlying low-rank-compression math is the same
//! family.
//!
//! Two more real, non-obvious facts from that source, beyond what
//! `frink_models::mla`'s module doc comment already covers for the
//! shared MLA math:
//!
//! 1. Per-layer full/shared indexer dispatch is **not a fixed period**.
//!    GLM-5.2's real per-layer `indexer_types` array has layers 0-1 as
//!    "full", then a repeating 1-full+3-shared pattern -- but the *real
//!    mechanism* a "shared" layer uses is "reuse the top-k from the
//!    nearest preceding full layer," not "recompute every 4th layer."
//!    Confirmed directly from `glm-dsa.cpp`'s per-layer loop: a single
//!    `prev_top_k` local variable is reassigned only when a full layer
//!    runs, and carried forward unconditionally into every following
//!    shared layer until the next full layer reassigns it --
//!    `GGML_ASSERT(prev_top_k != nullptr && "shared indexer layer must
//!    follow a previous full indexer layer")` on the shared-layer
//!    branch confirms a shared layer can never be the first layer
//!    processed. [`glm52_attn_forward_token`]'s `prev_top_k` parameter
//!    mirrors this exactly: caller-threaded, per-token-forward-pass
//!    scoped (reset to `None` at the start of each new token, the same
//!    "one token, all layers in order" scope the real ggml local
//!    variable has), not persisted across tokens.
//! 2. The lightning indexer's own q/k split into rope/nope halves is
//!    **rope-FIRST, nope-second** -- read directly from `glm-dsa.cpp`'s
//!    `indexer_q_pe`/`indexer_q_nope` `ggml_view_3d` byte offsets
//!    (`indexer_q_pe` at offset 0, `indexer_q_nope` at
//!    `ggml_row_size(..., n_embd_indexer_head_nope)`), the **opposite**
//!    of the main attention's nope-first/rope-second convention.
//!    **GENUINE DISCLOSED CAVEAT**: GLM-5.2's own real `config.json`
//!    gives `index_head_dim=128` with the indexer's rope portion
//!    reusing the main attention's `n_rot()`=64, so
//!    `nope_dim == rope_dim == 64` for this specific model -- meaning
//!    this physical-order reading is **not numerically distinguishable
//!    from its opposite** by inspecting GLM-5.2's own hyperparameters
//!    alone (the offset expression's value is identical either way).
//!    This implementation commits to the literal "first view in the
//!    code is the rope view" reading rather than silently picking
//!    whichever seemed more consistent with the main attention's
//!    convention; see docs/MODELS.md for the same caveat recorded
//!    against the evidence ledger.
//!
//! Tested here against synthetic weights, cross-validated against an
//! independent Python transcription of one "full" indexer layer's
//! RoPE+indexer+sparse
//! math across four decode steps, including a step where top-k
//! sparsity actually excludes a causally-visible position) plus
//! dedicated Rust-only tests for the full/shared dispatch bookkeeping
//! itself (not real "math to get subtly wrong" the way RoPE/
//! indexer-scoring/sparse-selection are, so not re-derived in Python).

use frink_core::attention::{
    apply_rope_interleaved, causal_mla_attention_sparse, lightning_indexer_topk,
};
use frink_core::matmul::{layer_norm, rms_norm};
use frink_core::weight_matrix::WeightMatrix;

use crate::config::MlaRopeConfig;

/// GLM-5.2's real per-layer MLA hyperparameters. Unlike
/// `frink_models::mla::MlaConfig`, there is no `use_output_gate` (real
/// GLM-5.2 tensor list has no Kimi-K3-style `attn_gate` equivalent) and
/// rope is unconditional, not `Option` (every real GLM-5.2 layer's main
/// attention applies it -- `rope_interleave: true` in its
/// `config.json`, no per-layer exception unlike the indexer's
/// full/shared split).
#[derive(Debug, Clone)]
pub struct Glm52MlaConfig {
    pub num_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rope: MlaRopeConfig,
}

/// GLM-5.2's real lightning-indexer hyperparameters
/// (`index_head_dim`=128, `index_n_heads`=32, `index_topk`=2048 in the
/// real published `config.json` -- see docs/MODELS.md; kept
/// generic here, not hardcoded, so small synthetic tests can use tiny
/// values).
#[derive(Debug, Clone)]
pub struct IndexerConfig {
    pub n_heads: usize,
    /// Total per-(shared-)head dimension, rope + nope.
    pub head_dim: usize,
    /// The rope portion of `head_dim` -- real GLM-5.2 reuses the main
    /// attention's `n_rot()` for this (see module doc comment point 2).
    pub rope_dim: usize,
    pub top_k: usize,
    pub rope_theta: f32,
}

/// The real lightning indexer's weights for one layer (only present on
/// "full" indexer layers -- see [`glm52_attn_forward_token`]'s
/// `is_full_indexer_layer` parameter).
pub struct IndexerWeights {
    pub k_norm_weight: Vec<f32>,
    pub k_norm_bias: Vec<f32>,
    /// `n_embd -> n_heads` (one scalar weight per indexer head).
    pub proj: WeightMatrix,
    /// `n_embd -> head_dim`: a single shared (MQA-style) key, not
    /// per-head -- real `indexer.attn_k` tensor shape confirms this
    /// (no `n_heads` factor), matching Kimi K3's MLA `k_rot` MQA
    /// pattern (`frink_models::mla`'s module doc comment point 2).
    pub attn_k: WeightMatrix,
    /// `q_lora_rank -> n_heads*head_dim`.
    pub attn_q_b: WeightMatrix,
}

/// GLM-5.2's real per-layer MLA+indexer attention weights.
pub struct Glm52AttnWeights {
    pub q_a_proj: WeightMatrix,
    pub q_a_layernorm: Vec<f32>,
    pub q_b_proj: WeightMatrix,
    pub kv_a_proj_with_mqa: WeightMatrix,
    pub kv_a_layernorm: Vec<f32>,
    /// Per-head, in the DECOMPRESSION direction (`kv_lora_rank ->
    /// qk_nope_head_dim`) -- i.e. **already transposed** from the real
    /// GGUF's on-disk `attn_k_b` layout, which llama.cpp instead
    /// applies in the opposite ("absorbed": `qk_nope_head_dim ->
    /// kv_lora_rank`, pre-multiplying the *query* rather than
    /// decompressing K) direction as a compute optimization --
    /// mathematically equivalent, just not the form llama.cpp actually
    /// executes. `glm52_gguf_loader` performs this transpose at load
    /// time so this module can reuse the same direct-decompression
    /// math `frink_models::mla` already has, via
    /// `causal_mla_attention_sparse`, instead of implementing a second,
    /// absorbed-computation attention primitive. `wk_b[h]` is
    /// `[qk_nope_head_dim, kv_lora_rank]`.
    pub wk_b: Vec<WeightMatrix>,
    /// Per-head, natively in the decompression direction
    /// (`kv_lora_rank -> v_head_dim`) already -- no transpose needed
    /// (real GGUF `attn_v_b` is used by llama.cpp to decompress V
    /// directly too, unlike `wk_b`). `wv_b[h]` is
    /// `[v_head_dim, kv_lora_rank]`.
    pub wv_b: Vec<WeightMatrix>,
    pub o_proj: WeightMatrix,
    /// Only present on "full" indexer layers -- see
    /// [`glm52_attn_forward_token`]'s doc comment.
    pub indexer: Option<IndexerWeights>,
}

/// Growable per-layer decode state: the main K/V cache (same shape
/// convention as `frink_models::mla`'s `k_cache`/`v_cache`) plus,
/// for "full" indexer layers only, the indexer's own K cache (a
/// separate cache since `IndexerConfig::head_dim` generally differs
/// from the main attention's `qk_nope_head_dim + qk_rope_head_dim`).
/// "Shared" layers never touch `indexer_k_cache` -- they don't run
/// their own indexer at all, see module doc comment point 1.
#[derive(Debug, Clone, Default)]
pub struct Glm52AttnState {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub indexer_k_cache: Vec<f32>,
}

impl Glm52AttnState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// One decode step for one GLM-5.2 DSA attention layer.
///
/// `is_full_indexer_layer` selects which of the two real per-layer
/// behaviors this call uses (see module doc comment point 1):
/// - `true` ("full"): `weights.indexer` must be `Some`; this call
///   computes a fresh top-k from the indexer, appending to
///   `state.indexer_k_cache`, and writes the result into `*prev_top_k`
///   for later "shared" layers *within this same token's forward
///   pass* to reuse.
/// - `false` ("shared"): reuses `prev_top_k` as-is (must already be
///   `Some` -- the real architecture guarantees the first layer
///   processed is always "full," matching llama.cpp's
///   `GGML_ASSERT(prev_top_k != nullptr ...)`); `weights.indexer` is
///   ignored (may be `None`).
///
/// `prev_top_k` is caller-owned and must be reset to `None` at the
/// start of every new token's forward pass across all layers (not
/// persisted token-to-token) -- mirroring the real ggml local variable
/// of the same name and scope in `glm-dsa.cpp`'s per-token graph build.
#[allow(clippy::too_many_arguments)]
pub fn glm52_attn_forward_token(
    weights: &Glm52AttnWeights,
    cfg: &Glm52MlaConfig,
    indexer_cfg: &IndexerConfig,
    hidden: &[f32],
    rms_norm_eps: f32,
    is_full_indexer_layer: bool,
    state: &mut Glm52AttnState,
    prev_top_k: &mut Option<Vec<usize>>,
) -> Vec<f32> {
    let q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
    let pos = state.k_cache.len() / (cfg.num_heads * q_head_dim);

    let q_a = weights.q_a_proj.apply(hidden);
    let qr = rms_norm(&q_a, &weights.q_a_layernorm, rms_norm_eps);

    let visible: Vec<usize> = if is_full_indexer_layer {
        let indexer = weights
            .indexer
            .as_ref()
            .expect("a \"full\" indexer layer must carry indexer weights");
        let idx_head_dim = indexer_cfg.head_dim;
        let idx_rope_dim = indexer_cfg.rope_dim;

        let indexer_q_full = indexer.attn_q_b.apply(&qr); // [n_heads*head_dim]
        let indexer_q: Vec<Vec<f32>> = (0..indexer_cfg.n_heads)
            .map(|h| {
                let head_slice = &indexer_q_full[h * idx_head_dim..(h + 1) * idx_head_dim];
                // rope-first, nope-second -- see module doc comment point 2.
                let mut rope_part = head_slice[..idx_rope_dim].to_vec();
                apply_rope_interleaved(&mut rope_part, pos, indexer_cfg.rope_theta);
                rope_part.extend_from_slice(&head_slice[idx_rope_dim..]);
                rope_part
            })
            .collect();

        let indexer_k_raw = indexer.attn_k.apply(hidden); // [head_dim], shared (MQA)
        let indexer_k_normed = layer_norm(
            &indexer_k_raw,
            &indexer.k_norm_weight,
            &indexer.k_norm_bias,
            rms_norm_eps,
        );
        let mut idx_rope_part = indexer_k_normed[..idx_rope_dim].to_vec();
        apply_rope_interleaved(&mut idx_rope_part, pos, indexer_cfg.rope_theta);
        idx_rope_part.extend_from_slice(&indexer_k_normed[idx_rope_dim..]);
        state.indexer_k_cache.extend_from_slice(&idx_rope_part);

        let idx_seq_len = state.indexer_k_cache.len() / idx_head_dim;
        let indexer_keys: Vec<Vec<f32>> = (0..idx_seq_len)
            .map(|t| state.indexer_k_cache[t * idx_head_dim..(t + 1) * idx_head_dim].to_vec())
            .collect();

        let indexer_weights_vec = indexer.proj.apply(hidden); // [n_heads]
        let top_k = lightning_indexer_topk(
            &indexer_q,
            &indexer_keys,
            &indexer_weights_vec,
            indexer_cfg.top_k,
        );
        *prev_top_k = Some(top_k.clone());
        top_k
    } else {
        prev_top_k
            .clone()
            .expect("a \"shared\" indexer layer must follow a previous \"full\" layer's top-k within this token's forward pass")
    };

    let mut query = weights.q_b_proj.apply(&qr); // [n_heads*q_head_dim], nope first then rope
    for h in 0..cfg.num_heads {
        let q_rot_h = &mut query[h * q_head_dim + cfg.qk_nope_head_dim..(h + 1) * q_head_dim];
        apply_rope_interleaved(q_rot_h, pos, cfg.rope.theta);
    }

    let kv_cmpr_pe = weights.kv_a_proj_with_mqa.apply(hidden);
    let (kv_cmpr_raw, k_pe_raw) = kv_cmpr_pe.split_at(cfg.kv_lora_rank);
    let mut k_pe = k_pe_raw.to_vec();
    apply_rope_interleaved(&mut k_pe, pos, cfg.rope.theta);
    let kv_cmpr = rms_norm(kv_cmpr_raw, &weights.kv_a_layernorm, rms_norm_eps);

    let mut key_step = vec![0f32; cfg.num_heads * q_head_dim];
    let mut value_step = vec![0f32; cfg.num_heads * cfg.v_head_dim];
    for h in 0..cfg.num_heads {
        let k_pass = weights.wk_b[h].apply(&kv_cmpr); // [qk_nope_head_dim]
        let v_h = weights.wv_b[h].apply(&kv_cmpr); // [v_head_dim]

        let key_h = &mut key_step[h * q_head_dim..(h + 1) * q_head_dim];
        key_h[..cfg.qk_nope_head_dim].copy_from_slice(&k_pass);
        key_h[cfg.qk_nope_head_dim..].copy_from_slice(&k_pe);

        value_step[h * cfg.v_head_dim..(h + 1) * cfg.v_head_dim].copy_from_slice(&v_h);
    }

    state.k_cache.extend_from_slice(&key_step);
    state.v_cache.extend_from_slice(&value_step);
    let seq_len = state.k_cache.len() / (cfg.num_heads * q_head_dim);

    let attn_out = causal_mla_attention_sparse(
        &query,
        &state.k_cache,
        &state.v_cache,
        cfg.num_heads,
        q_head_dim,
        cfg.v_head_dim,
        seq_len,
        &visible,
    );

    weights.o_proj.apply(&attn_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MlaRopeConfig;
    use frink_core::tensor::Tensor;

    fn wm(data: &[f32], rows: usize, cols: usize) -> WeightMatrix {
        assert_eq!(data.len(), rows * cols);
        WeightMatrix::F32(Tensor::new(data.to_vec(), vec![rows, cols]))
    }

    const HIDDEN_SIZE: usize = 8;
    const NUM_HEADS: usize = 2;
    const QK_NOPE_HEAD_DIM: usize = 4;
    const QK_ROPE_HEAD_DIM: usize = 4;
    const KV_LORA_RANK: usize = 4;
    const Q_LORA_RANK: usize = 6;
    const V_HEAD_DIM: usize = 3;
    const EPS: f32 = 1e-5;
    const ROPE_THETA: f32 = 10000.0;

    const IDX_N_HEADS: usize = 2;
    const IDX_ROPE_DIM: usize = 2;
    const IDX_NOPE_DIM: usize = 2;
    const IDX_HEAD_DIM: usize = IDX_ROPE_DIM + IDX_NOPE_DIM;
    const TOP_K: usize = 2;

    // Generated by an independent Python reference -- do not hand-edit.
    const GLM_Q_A_PROJ: [f32; 48] = [
        -0.0282433, -0.200786, 0.229784, 0.162609, 0.161474, 0.190923, -0.28271, -0.259493,
        -0.299452, -0.445281, 0.438341, 0.223402, -0.114472, 0.108293, 0.138669, -0.0983588,
        -0.139929, -0.287077, 0.233259, -0.0924404, -0.48316, 0.236404, 0.487659, -0.22182,
        -0.425881, 0.474637, 0.224366, 0.227792, 0.155548, 0.139447, -0.0826791, -0.435336,
        0.438776, -0.328727, 0.415205, -0.280294, 0.25502, -0.264693, -0.374199, 0.626933,
        0.175746, 0.198436, -0.0882081, -0.131098, 0.416386, 0.0938253, -0.086288, -0.173465,
    ];
    const GLM_Q_A_LAYERNORM_W: [f32; 6] = [0.983826, 0.879749, 1.11838, 1.05149, 1.1818, 0.834347];
    const GLM_Q_B_PROJ: [f32; 96] = [
        0.291994,
        -0.728247,
        -0.316577,
        0.0813452,
        -0.266261,
        0.389674,
        0.0833719,
        0.351552,
        0.356938,
        0.350334,
        -0.0473987,
        -0.0266404,
        -0.169264,
        0.0701104,
        0.0207743,
        0.44759,
        0.372409,
        0.283663,
        0.161893,
        0.0691206,
        0.164776,
        -0.159844,
        0.244357,
        0.254148,
        0.781266,
        -0.0410461,
        0.00448047,
        0.167438,
        0.134256,
        -0.117364,
        0.613949,
        -0.207111,
        0.42746,
        0.45351,
        0.237126,
        -0.50974,
        0.328859,
        -0.250491,
        -0.356138,
        0.122879,
        0.254109,
        0.120117,
        -0.244618,
        0.090442,
        0.572282,
        -0.175117,
        0.150304,
        0.127176,
        -0.230927,
        -0.181049,
        0.0503238,
        -0.252932,
        -0.00813607,
        -0.169141,
        0.178562,
        -0.172518,
        -0.163208,
        -0.286795,
        0.358209,
        0.355661,
        -0.0321808,
        0.025399,
        -0.227651,
        -0.0153813,
        -0.0254572,
        -0.364581,
        -0.450488,
        0.155816,
        0.0033226,
        0.481021,
        -0.000260049,
        -0.230117,
        -0.0422523,
        0.269254,
        -0.225551,
        -0.265757,
        -0.192519,
        -0.300859,
        -0.152023,
        0.31445,
        -0.229592,
        -0.417754,
        -0.219984,
        0.0230321,
        0.162062,
        -0.162489,
        -0.504785,
        0.117479,
        -0.152083,
        0.203557,
        -0.232979,
        -0.537171,
        -0.131909,
        -0.0782392,
        0.187798,
        -0.364894,
    ];
    const GLM_KV_A_PROJ: [f32; 64] = [
        -0.0108888,
        0.122853,
        -0.388147,
        -0.320502,
        0.288834,
        0.0587081,
        0.0027565,
        -0.0303023,
        0.252204,
        0.103756,
        -0.563955,
        0.539224,
        -0.515732,
        0.475067,
        -0.179422,
        0.512039,
        0.564391,
        -0.309453,
        0.178157,
        0.4829,
        0.304922,
        -0.327155,
        0.235531,
        -0.223351,
        0.113775,
        -0.326219,
        0.129363,
        0.343847,
        -0.555633,
        -0.00371874,
        -0.480022,
        0.0622793,
        0.121396,
        0.902273,
        -0.271857,
        -0.0787809,
        -0.148056,
        -0.246381,
        -0.388923,
        -0.326308,
        0.754771,
        -0.188557,
        0.157124,
        -0.242718,
        -0.196856,
        0.168396,
        0.116464,
        0.406121,
        -0.0524445,
        -0.226537,
        -0.220791,
        -0.42747,
        -0.109609,
        0.327875,
        0.238249,
        0.262922,
        0.0603609,
        0.259383,
        -0.125942,
        0.0253563,
        -0.672037,
        -0.0822506,
        -0.313883,
        -0.079927,
    ];
    const GLM_KV_A_LAYERNORM_W: [f32; 4] = [1.11964, 1.00794, 0.927425, 0.974059];
    const GLM_WK_B_0: [f32; 16] = [
        0.152348, 0.223828, 0.104805, 0.146991, 0.167048, 0.0596911, 0.13428, -0.165432, 0.185469,
        -0.438659, 0.0505363, -0.378193, -0.367254, 0.291605, 0.25605, -0.278039,
    ];
    const GLM_WV_B_0: [f32; 12] = [
        -0.0451786, 0.016892, -0.503596, 0.38556, 0.0672434, -0.345241, 0.261026, 0.113356,
        0.195666, 0.124068, 0.169155, 0.0241996,
    ];
    const GLM_WK_B_1: [f32; 16] = [
        0.32183, 0.077576, 0.657456, -0.210592, -0.241762, -0.113545, 0.305935, -0.142537,
        -0.131723, -0.0698308, -0.231153, 0.0832406, -0.184562, -0.395013, -0.434206, 0.643208,
    ];
    const GLM_WV_B_1: [f32; 12] = [
        0.0460725, -0.199187, 0.525018, 0.704311, 0.214609, 0.155865, -0.13945, -0.361349,
        0.200727, 0.669041, -0.35694, 0.405635,
    ];
    const GLM_O_PROJ: [f32; 48] = [
        0.156636,
        0.435755,
        0.254836,
        -0.28038,
        -0.00686566,
        0.254093,
        0.13879,
        0.298608,
        -0.654407,
        0.544604,
        -0.40823,
        0.557235,
        -0.401607,
        0.0393622,
        -0.0108063,
        -0.425778,
        -0.0790213,
        0.183181,
        0.770074,
        0.431033,
        -0.191665,
        -0.321149,
        -0.243943,
        -0.0704616,
        0.180775,
        -0.216385,
        0.0824125,
        -0.320591,
        -0.182163,
        -0.0257085,
        -0.0184709,
        0.292862,
        -0.215734,
        0.652291,
        -0.0461593,
        0.249014,
        -0.205017,
        0.0634068,
        0.087137,
        0.529326,
        0.477227,
        0.171185,
        0.0539693,
        0.0189488,
        -0.138254,
        -0.173556,
        0.65771,
        -0.0616593,
    ];
    const GLM_INDEXER_PROJ: [f32; 16] = [
        -0.087587, -0.0520619, 0.169093, -0.459473, -0.354101, -0.151364, -0.238526, 0.783742,
        0.0472875, -0.286511, -0.0857637, 0.0416524, 0.383631, 0.309099, -0.0708246, -0.45329,
    ];
    const GLM_INDEXER_ATTN_K: [f32; 32] = [
        -0.0243618, 0.264064, 0.116963, 0.0329414, 0.16249, 0.374711, 0.0555387, 0.011557,
        0.720572, -0.288658, 0.660557, -0.0520769, -0.398689, 0.10919, -0.21608, -0.272101,
        0.324867, -0.0406344, 0.436622, -0.353795, 0.30306, -0.256956, 0.381955, 0.346494,
        0.579363, 0.0492111, 0.0489935, -0.110262, -0.359962, 0.0704509, 0.402797, -0.121248,
    ];
    const GLM_INDEXER_ATTN_Q_B: [f32; 48] = [
        0.423031,
        0.0939434,
        0.165178,
        -0.337296,
        -0.0761678,
        -0.0724417,
        -0.485867,
        0.693554,
        0.201884,
        0.544458,
        -0.0574976,
        0.2739,
        0.174839,
        0.602076,
        -0.00910648,
        0.12034,
        -0.0199685,
        0.238296,
        0.0537028,
        0.221498,
        -0.0995168,
        0.0551425,
        0.126781,
        0.14289,
        0.0625852,
        0.14437,
        0.00611794,
        0.166845,
        -0.566204,
        0.132155,
        -0.292038,
        -0.203268,
        -0.226073,
        0.2511,
        -0.339645,
        0.0101247,
        0.0409607,
        -0.369437,
        0.101741,
        -0.517995,
        0.150119,
        0.19386,
        0.115472,
        -0.196285,
        -0.0137043,
        0.291252,
        -0.0213699,
        0.00902874,
    ];
    const GLM_INDEXER_K_NORM_W: [f32; 4] = [1.01802, 0.864429, 0.781694, 1.0421];
    const GLM_INDEXER_K_NORM_B: [f32; 4] = [-0.00865097, -0.0181871, -0.0141045, 0.0431985];

    const GLM_HIDDEN_0: [f32; 8] = [
        0.173677, 0.403207, -0.819441, 0.0385262, -0.382448, 0.0666326, 0.107605, -0.50578,
    ];
    const GLM_HIDDEN_1: [f32; 8] = [
        0.0557989, -0.774157, 0.586028, -0.0179948, 0.00680277, 0.436953, 0.188515, -0.2977,
    ];
    const GLM_HIDDEN_2: [f32; 8] = [
        -0.718918, 1.18182, -0.484152, -0.465392, 0.221702, -0.438492, -0.388645, -0.164228,
    ];
    const GLM_HIDDEN_3: [f32; 8] = [
        0.650653, -0.256895, -0.94904, 0.929269, -0.585101, 0.0973952, 0.140374, 0.190254,
    ];

    const GLM_GOLDEN_OUT_0: [f32; 8] = [
        0.361853, -0.250738, 0.433283, -0.278145, 0.27569, -0.414117, 0.121914, 0.401593,
    ];
    const GLM_GOLDEN_OUT_1: [f32; 8] = [
        -0.013624, 0.187721, -0.050068, -0.116258, -0.0672519, 0.179838, 0.13011, -0.114857,
    ];
    const GLM_GOLDEN_OUT_2: [f32; 8] = [
        0.28988, -0.593942, 0.388018, 0.0189887, 0.317219, -0.64959, -0.143766, 0.524578,
    ];
    const GLM_GOLDEN_OUT_3: [f32; 8] = [
        0.212341, 0.238018, 0.206285, -0.35234, 0.114026, 0.0278299, 0.262389, 0.0396803,
    ];

    const GLM_GOLDEN_VISIBLE_0: [usize; 1] = [0];
    const GLM_GOLDEN_VISIBLE_1: [usize; 2] = [0, 1];
    const GLM_GOLDEN_VISIBLE_2: [usize; 2] = [0, 2];
    const GLM_GOLDEN_VISIBLE_3: [usize; 2] = [0, 3];

    fn cfg() -> Glm52MlaConfig {
        Glm52MlaConfig {
            num_heads: NUM_HEADS,
            q_lora_rank: Q_LORA_RANK,
            kv_lora_rank: KV_LORA_RANK,
            qk_nope_head_dim: QK_NOPE_HEAD_DIM,
            qk_rope_head_dim: QK_ROPE_HEAD_DIM,
            v_head_dim: V_HEAD_DIM,
            rope: MlaRopeConfig { theta: ROPE_THETA },
        }
    }

    fn indexer_cfg() -> IndexerConfig {
        IndexerConfig {
            n_heads: IDX_N_HEADS,
            head_dim: IDX_HEAD_DIM,
            rope_dim: IDX_ROPE_DIM,
            top_k: TOP_K,
            rope_theta: ROPE_THETA,
        }
    }

    fn make_weights() -> Glm52AttnWeights {
        let q_head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
        Glm52AttnWeights {
            q_a_proj: wm(&GLM_Q_A_PROJ, Q_LORA_RANK, HIDDEN_SIZE),
            q_a_layernorm: GLM_Q_A_LAYERNORM_W.to_vec(),
            q_b_proj: wm(&GLM_Q_B_PROJ, NUM_HEADS * q_head_dim, Q_LORA_RANK),
            kv_a_proj_with_mqa: wm(&GLM_KV_A_PROJ, KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN_SIZE),
            kv_a_layernorm: GLM_KV_A_LAYERNORM_W.to_vec(),
            wk_b: vec![
                wm(&GLM_WK_B_0, QK_NOPE_HEAD_DIM, KV_LORA_RANK),
                wm(&GLM_WK_B_1, QK_NOPE_HEAD_DIM, KV_LORA_RANK),
            ],
            wv_b: vec![
                wm(&GLM_WV_B_0, V_HEAD_DIM, KV_LORA_RANK),
                wm(&GLM_WV_B_1, V_HEAD_DIM, KV_LORA_RANK),
            ],
            o_proj: wm(&GLM_O_PROJ, HIDDEN_SIZE, NUM_HEADS * V_HEAD_DIM),
            indexer: Some(IndexerWeights {
                k_norm_weight: GLM_INDEXER_K_NORM_W.to_vec(),
                k_norm_bias: GLM_INDEXER_K_NORM_B.to_vec(),
                proj: wm(&GLM_INDEXER_PROJ, IDX_N_HEADS, HIDDEN_SIZE),
                attn_k: wm(&GLM_INDEXER_ATTN_K, IDX_HEAD_DIM, HIDDEN_SIZE),
                attn_q_b: wm(
                    &GLM_INDEXER_ATTN_Q_B,
                    IDX_N_HEADS * IDX_HEAD_DIM,
                    Q_LORA_RANK,
                ),
            }),
        }
    }

    #[test]
    fn full_layer_matches_independent_python_reference_across_four_decode_steps() {
        let weights = make_weights();
        let cfg = cfg();
        let idx_cfg = indexer_cfg();
        let mut state = Glm52AttnState::new();
        let mut prev_top_k: Option<Vec<usize>> = None;

        let hiddens = [
            &GLM_HIDDEN_0[..],
            &GLM_HIDDEN_1[..],
            &GLM_HIDDEN_2[..],
            &GLM_HIDDEN_3[..],
        ];
        let goldens = [
            &GLM_GOLDEN_OUT_0[..],
            &GLM_GOLDEN_OUT_1[..],
            &GLM_GOLDEN_OUT_2[..],
            &GLM_GOLDEN_OUT_3[..],
        ];
        let golden_visible: [&[usize]; 4] = [
            &GLM_GOLDEN_VISIBLE_0,
            &GLM_GOLDEN_VISIBLE_1,
            &GLM_GOLDEN_VISIBLE_2,
            &GLM_GOLDEN_VISIBLE_3,
        ];

        for (pos, ((hidden, golden), visible)) in hiddens
            .iter()
            .zip(goldens.iter())
            .zip(golden_visible.iter())
            .enumerate()
        {
            // Every layer here is "full" -- this test is specifically
            // about the RoPE+indexer+sparse math, not the full/shared
            // dispatch bookkeeping (covered separately below).
            let out = glm52_attn_forward_token(
                &weights,
                &cfg,
                &idx_cfg,
                hidden,
                EPS,
                true,
                &mut state,
                &mut prev_top_k,
            );
            assert_eq!(out.len(), golden.len());
            for (i, (a, b)) in out.iter().zip(golden.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3,
                    "position {pos} element {i}: rust={a} python={b}"
                );
            }
            // Cross-check the indexer's own top-k selection against the
            // Python reference too -- this is the sparsity mechanism
            // itself, not just its downstream numerical effect.
            assert_eq!(
                prev_top_k.as_deref(),
                Some(*visible),
                "position {pos}: indexer top-k selection mismatch"
            );
        }
    }

    #[test]
    #[should_panic(expected = "must follow a previous")]
    fn shared_layer_without_a_preceding_full_layer_panics() {
        let weights = make_weights();
        let cfg = cfg();
        let idx_cfg = indexer_cfg();
        let mut state = Glm52AttnState::new();
        let mut prev_top_k: Option<Vec<usize>> = None;

        // First layer processed is "shared" with no prior "full" layer
        // in this token's forward pass -- the real architecture's
        // GGML_ASSERT guarantees this can never happen for a real
        // checkpoint (layer 0 is always "full"), so this must panic
        // loudly rather than silently produce wrong output.
        glm52_attn_forward_token(
            &weights,
            &cfg,
            &idx_cfg,
            &GLM_HIDDEN_0,
            EPS,
            false,
            &mut state,
            &mut prev_top_k,
        );
    }

    #[test]
    fn shared_layer_reuses_the_nearest_preceding_full_layers_top_k_exactly() {
        // Three layers processed for the same token, in order:
        // full (layer A) -> shared (layer B) -> shared (layer C).
        // Real semantics: B and C must both reuse EXACTLY layer A's
        // top-k, not recompute their own -- confirmed structurally by
        // checking `prev_top_k` is untouched by B/C's calls (a "shared"
        // layer with `weights.indexer: None` would panic on `.expect()`
        // inside the `is_full_indexer_layer` branch if it ever tried to
        // compute its own top-k instead of reusing `prev_top_k`).
        let mut weights = make_weights();
        let cfg = cfg();
        let idx_cfg = indexer_cfg();
        let mut state = Glm52AttnState::new();
        let mut prev_top_k: Option<Vec<usize>> = None;

        // Layer A: full.
        glm52_attn_forward_token(
            &weights,
            &cfg,
            &idx_cfg,
            &GLM_HIDDEN_0,
            EPS,
            true,
            &mut state,
            &mut prev_top_k,
        );
        let top_k_after_full = prev_top_k.clone();
        assert!(top_k_after_full.is_some());

        // Now strip the indexer weights entirely -- a "shared" layer
        // must never touch them, so this proves it by construction:
        // if the shared-layer code path ever dereferenced
        // `weights.indexer`, this would panic on the `.expect()` inside
        // the (never-taken, for `is_full_indexer_layer=false`) full
        // branch's `Option::as_ref().expect(...)` -- but that branch is
        // gated on `is_full_indexer_layer`, so it's simply never
        // reached.
        weights.indexer = None;

        // Layer B: shared, reusing layer A's cached indexer state
        // implicitly via `prev_top_k` (its own `state.indexer_k_cache`
        // is a per-attention-instance field in this test setup, reused
        // across calls since it's the same `state`/`weights` -- a real
        // decoder would have one `Glm52AttnState` per *layer*, not
        // shared across layers the way this test reuses one for
        // brevity, since each layer has its own main K/V cache; what's
        // being verified here is purely the `prev_top_k` reuse
        // contract, which is decoder-scoped, not per-layer-state
        // scoped).
        glm52_attn_forward_token(
            &weights,
            &cfg,
            &idx_cfg,
            &GLM_HIDDEN_1,
            EPS,
            false,
            &mut state,
            &mut prev_top_k,
        );
        assert_eq!(
            prev_top_k, top_k_after_full,
            "shared layer B must not alter prev_top_k"
        );

        // Layer C: shared again, still reusing layer A's top-k.
        glm52_attn_forward_token(
            &weights,
            &cfg,
            &idx_cfg,
            &GLM_HIDDEN_2,
            EPS,
            false,
            &mut state,
            &mut prev_top_k,
        );
        assert_eq!(
            prev_top_k, top_k_after_full,
            "shared layer C must not alter prev_top_k"
        );
    }

    #[test]
    fn full_layer_grows_the_indexer_cache_but_shared_layer_does_not() {
        let mut weights = make_weights();
        let cfg = cfg();
        let idx_cfg = indexer_cfg();
        let mut state = Glm52AttnState::new();
        let mut prev_top_k: Option<Vec<usize>> = None;

        glm52_attn_forward_token(
            &weights,
            &cfg,
            &idx_cfg,
            &GLM_HIDDEN_0,
            EPS,
            true,
            &mut state,
            &mut prev_top_k,
        );
        let len_after_full = state.indexer_k_cache.len();
        assert_eq!(len_after_full, IDX_HEAD_DIM);

        weights.indexer = None;
        glm52_attn_forward_token(
            &weights,
            &cfg,
            &idx_cfg,
            &GLM_HIDDEN_1,
            EPS,
            false,
            &mut state,
            &mut prev_top_k,
        );
        assert_eq!(
            state.indexer_k_cache.len(),
            len_after_full,
            "a \"shared\" layer must not grow the indexer K cache"
        );
    }
}
