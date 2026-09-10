//! A dedicated decoder skeleton for DeepSeek V4's real architecture,
//! separate from `ferrox-models::decoder::Decoder` (the generic GQA path
//! every other preset uses) — analogous to `glm52_decoder.rs` /
//! `kimi_decoder.rs`: composes the already-independently-tested mHC,
//! CSA/HCA compression + attention, derope, grouped output projection,
//! and sqrtsoftplus MoE primitives into one synthetic forward pass.
//!
//! **Synthetic weights only — not a real checkpoint path.** No GGUF
//! loader, no `Engine` wiring, no claim of oracle-correct output against
//! a production DeepSeek V4 file. This module exists to prove the real
//! primitives compose into a finite forward pass end-to-end on tiny dims,
//! the same rigor `glm52_decoder.rs` applies before a real loader lands.
//!
//! Real per-layer flow (simplified to one layer here, transcribed from
//! llama.cpp PR #24162 `deepseek4.cpp`'s layer loop):
//!
//! ```text
//! attn_in  = mHC_pre(hc_streams, attn_hc_pre)
//! attn_out = rms_norm(attn_in) |> HCA/CSA attention |> derope |> grouped wo_a/wo_b
//! hc_streams = mHC_post(attn_out, ...)
//! ffn_in   = mHC_pre(hc_streams, ffn_hc_pre)
//! ffn_out  = rms_norm(ffn_in) |> MoE (sqrtsoftplus routing)
//! hc_streams = mHC_post(ffn_out, ...)
//! hidden   = mHC_head(hc_streams)
//! logits   = output_head(rms_norm(hidden))
//! ```
//!
//! Deliberately **not** implemented in this skeleton (real, cited scope
//! for later slices): incremental DSV4 KV/compressor state
//! (`llama-kv-cache-dsv4.cpp`), CSA's `coff=2` dual-role projection,
//! hash-based first-layer MoE selection, and multi-layer stacking.

use crate::deepseek_v4_budget::LayerCompressor;
use ferrox_core::attention::apply_rope_back;
use ferrox_core::csa_hca_compress::compress_block;
use ferrox_core::deepseek_v4_attention::{csa_attention, hca_attention};
use ferrox_core::matmul::rms_norm;
use ferrox_core::tensor::Tensor;
use ferrox_core::weight_matrix::WeightMatrix;
use ferrox_moe::{
    combine_expert_outputs, route_top_k, run_expert, ExpertWeights, GatingFunction, GluAct,
};

use crate::hyper_connections::{
    head as hc_head, post as hc_post, pre as hc_pre, HyperConnectionHeadWeights,
    HyperConnectionPreWeights, HC_MULT,
};
use crate::output_projection::grouped_output_projection;

/// A CSA layer's extras: the doubled role projection and the Lightning
/// Indexer. Present only on a [`LayerCompressor::Csa`] layer, and its
/// absence there is a configuration error rather than a default.
///
/// # The doubled projection, which is the whole point of the split
///
/// On a CSA layer each raw token is projected **twice**: once for its
/// role as the tail of the block ending at it, and once as the head of
/// the next, overlapping block -- two different learned projections of
/// the same token, not one reused twice (llama.cpp
/// `load_arch_tensors`' `coff = ratio == 4 ? 2 : 1`, and
/// `build_overlap_compressed_kv_from_state`'s
/// `GGML_ASSERT(kv_state->ne[0] == 2*n_embd_head)`). A stack that
/// applies one uniform ratio gives every CSA layer a single-width
/// projection, which runs and produces numbers.
pub struct DeepseekV4CsaWeights {
    /// `qk_head_dim -> 2 * qk_head_dim`: the head-role projection in the
    /// leading half, the tail-role projection in the trailing half.
    pub role_proj: WeightMatrix,
    /// The same split for `attn_comp_wgate`'s scores, so every block row
    /// carries the gate belonging to the role it was projected for.
    pub role_gate: WeightMatrix,
    /// One indexer key per compressed entry.
    ///
    /// **A skeleton simplification, named rather than hidden**: upstream
    /// runs a *separate* compressor (`indexer_comp_*`) over the raw
    /// indexer projections, where this projects the already-compressed
    /// entry. The indexer keys are compressed representations either
    /// way, which is what the top-k selection needs; the second
    /// compressor's own pooling is not reproduced here.
    pub indexer_key_proj: WeightMatrix,
    /// The query side, `qk_head_dim -> n_index_heads * index_head_dim`.
    pub indexer_q_proj: WeightMatrix,
    /// One weight per index head; its length is `n_index_heads`.
    pub indexer_head_weights: Vec<f32>,
    /// How many compressed entries survive selection.
    pub indexer_top_k: usize,
}

/// One layer's attention-side weights (synthetic, tiny-dim fixtures only).
pub struct DeepseekV4AttnWeights {
    pub q_proj: WeightMatrix,
    pub k_proj: WeightMatrix,
    pub v_proj: WeightMatrix,
    /// Per-group down-projections (`wo_a`), one per contiguous head block.
    pub group_down: Vec<WeightMatrix>,
    pub wo_b: WeightMatrix,
    /// Block-compression gate (`attn_comp_wgate`) and norm for HCA pooling.
    pub comp_gate: WeightMatrix,
    pub comp_norm: Vec<f32>,
    /// One learned logit per query head, joining the softmax denominator
    /// with a zero value vector so a head can decline to attend rather
    /// than being forced to spend a full unit of weight on the keys it
    /// has. `None` for a checkpoint that ships no sinks.
    pub attn_sinks: Option<Vec<f32>>,
    /// Present iff this layer's compressor is [`LayerCompressor::Csa`].
    pub csa: Option<DeepseekV4CsaWeights>,
}

/// MoE FFN weights for one layer (`sqrtsoftplus` gating, no hash routing).
pub struct DeepseekV4MoeFfnWeights {
    pub router_weight: WeightMatrix,
    pub experts: Vec<ExpertWeights>,
    pub shared_expert: ExpertWeights,
}

pub struct DeepseekV4DecoderLayerWeights {
    pub attn_hc_pre: HyperConnectionPreWeights,
    pub attn_norm_weight: Vec<f32>,
    pub attn: DeepseekV4AttnWeights,
    pub ffn_hc_pre: HyperConnectionPreWeights,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn: DeepseekV4MoeFfnWeights,
}

pub struct DeepseekV4DecoderWeights {
    pub embedding: Tensor,
    pub layer: DeepseekV4DecoderLayerWeights,
    pub hc_head: HyperConnectionHeadWeights,
    pub final_norm_weight: Vec<f32>,
    pub output_head: WeightMatrix,
}

pub struct DeepseekV4DecoderConfig {
    pub rms_norm_eps: f32,
    pub hc_sinkhorn_iters: u32,
    pub hc_eps: f32,
    pub n_heads: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
    pub qk_rope: usize,
    pub compress_rope_theta: f32,
    pub n_experts_active: usize,
    pub moe_renormalize: bool,
    /// Which compressor this layer runs.
    ///
    /// A mechanism, not a scalar ratio, and that is the correction:
    /// upstream ships a per-layer array (`0, 0, 4, 128, 4, 128, 4, 0`)
    /// where `0` means no compressor at all, `4` means CSA -- overlapping
    /// blocks, a doubled projection width, and a Lightning Indexer --
    /// and `128` means HCA, non-overlapping and dense with no indexer.
    /// One uniform ratio across the stack builds an indexer on the HCA
    /// layers or none on the CSA ones, and gives half of them the wrong
    /// compressor width; all three failures run and produce numbers.
    /// [`LayerCompressor`] derives every one of those parameters from
    /// the mechanism, so they cannot be set inconsistently here.
    pub compressor: LayerCompressor,
}

/// Minimal per-layer state: raw K/V for the SWA window plus optional
/// compressed entries. No incremental DSV4 cache yet — callers append
/// one token at a time and pool when the layer's compressor has enough
/// raw positions for its next block.
pub struct DeepseekV4LayerState {
    hc_streams: [Vec<f32>; HC_MULT],
    raw_k: Vec<f32>,
    raw_v: Vec<f32>,
    compressed_k: Vec<f32>,
    compressed_v: Vec<f32>,
    /// CSA only: each raw token's `2 * qk_head_dim` role projection and
    /// its matching role gate, kept because a CSA block reaches back
    /// into the *previous* half-block's head-role rows.
    role_kv: Vec<f32>,
    role_scores: Vec<f32>,
    token_count: usize,
}

impl DeepseekV4LayerState {
    pub fn new(hidden_dim: usize) -> Self {
        let zero = vec![0.0; hidden_dim];
        Self {
            hc_streams: std::array::from_fn(|_| zero.clone()),
            raw_k: Vec::new(),
            raw_v: Vec::new(),
            compressed_k: Vec::new(),
            compressed_v: Vec::new(),
            role_kv: Vec::new(),
            role_scores: Vec::new(),
            token_count: 0,
        }
    }

    fn reset_hc_from_hidden(&mut self, hidden: &[f32]) {
        for stream in self.hc_streams.iter_mut() {
            stream.copy_from_slice(hidden);
        }
    }
}

pub struct DeepseekV4DecodeState {
    layer: DeepseekV4LayerState,
}

impl DeepseekV4DecodeState {
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            layer: DeepseekV4LayerState::new(hidden_dim),
        }
    }
}

fn derope_attn_out(
    attn_out: &mut [f32],
    n_heads: usize,
    v_head_dim: usize,
    qk_rope: usize,
    pos: usize,
    theta: f32,
) {
    assert!(qk_rope <= v_head_dim);
    for h in 0..n_heads {
        let head_start = h * v_head_dim;
        let rope_start = head_start + v_head_dim - qk_rope;
        apply_rope_back(
            &mut attn_out[rope_start..head_start + v_head_dim],
            pos,
            theta,
        );
    }
}

/// Pools one HCA block: `ratio` consecutive raw positions, gated by
/// `attn_comp_wgate`, into a single compressed entry.
///
/// Non-overlapping, which is the whole difference from CSA: block `j`
/// reads only its own `ratio` positions and never reaches back.
fn hca_block(
    weights: &DeepseekV4AttnWeights,
    cfg: &DeepseekV4DecoderConfig,
    state: &mut DeepseekV4LayerState,
    ratio: usize,
) {
    let n_raw = state.token_count;
    let per_token_k = cfg.n_heads * cfg.qk_head_dim;
    let per_token_v = cfg.n_heads * cfg.v_head_dim;
    let block_index = n_raw / ratio;

    let kv_block: Vec<Vec<f32>> = state.raw_k[(n_raw - ratio) * per_token_k..n_raw * per_token_k]
        .chunks(cfg.qk_head_dim)
        .map(|row| row.to_vec())
        .collect();
    let score_block: Vec<Vec<f32>> = kv_block
        .iter()
        .map(|row| weights.comp_gate.apply(row))
        .collect();
    let compressed_k = compress_block(
        &kv_block,
        &score_block,
        &weights.comp_norm,
        cfg.rms_norm_eps,
        cfg.qk_rope,
        block_index,
        cfg.compress_rope_theta,
    );
    let v_block: Vec<Vec<f32>> = state.raw_v[(n_raw - ratio) * per_token_v..n_raw * per_token_v]
        .chunks(cfg.v_head_dim)
        .map(|row| row.to_vec())
        .collect();
    let compressed_v = compress_block(
        &v_block,
        &score_block,
        &weights.comp_norm,
        cfg.rms_norm_eps,
        cfg.qk_rope,
        block_index,
        cfg.compress_rope_theta,
    );
    state.compressed_k.extend_from_slice(&compressed_k);
    state.compressed_v.extend_from_slice(&compressed_v);
}

/// Pools one CSA block: the `2 * ratio`-row concatenation of the
/// PREVIOUS half-block's head-role rows and the CURRENT half-block's
/// tail-role rows.
///
/// # Where the extra `ratio` rows come from
///
/// This is the overlap. Block `j` covers raw positions
/// `[(j-1)*ratio, j*ratio)` through their head-role projection and
/// `[j*ratio, (j+1)*ratio)` through their tail-role projection -- the
/// same token contributing to two blocks under two different learned
/// projections, which is what `coff = 2` buys and what a uniform ratio
/// silently drops.
///
/// # The first block reaches off the start of the sequence
///
/// Block 0 has no previous half-block. The real implementation pads
/// out-of-range reads with a zero-KV, `-inf`-score phantom row
/// (`dsv4_append_zero_row`), and `ferrox_core::csa_hca_compress`'s
/// module docs require callers to apply the same convention by hand.
/// A `-inf` score contributes exactly zero weight after the softmax, so
/// the phantom rows are present in the block and absent from the
/// result -- which is not the same as shortening the block, because the
/// per-channel softmax normalizes over whatever rows it is given.
fn csa_block(
    weights: &DeepseekV4AttnWeights,
    cfg: &DeepseekV4DecoderConfig,
    state: &mut DeepseekV4LayerState,
    ratio: usize,
) {
    assert_eq!(
        cfg.n_heads, 1,
        "this skeleton compresses one head: a block is assembled as rows of qk_head_dim, so \
         more heads would pool across them into a single entry and leave n_compressed \
         fractional. Multi-head compression is real scope, not a silent approximation"
    );
    let n_raw = state.token_count;
    // `blocks_closed` counts blocks including this one, so the block
    // being built is ordinal `blocks_closed - 1` and its own tokens are
    // the LAST `ratio` appended -- not the next `ratio`, which is the
    // off-by-one that indexes past the end of the state.
    let blocks_closed = n_raw / ratio;
    let block_ord = blocks_closed - 1;
    let role_width = 2 * cfg.qk_head_dim;

    let mut kv_block: Vec<Vec<f32>> = Vec::with_capacity(2 * ratio);
    let mut score_block: Vec<Vec<f32>> = Vec::with_capacity(2 * ratio);
    let mut v_block: Vec<Vec<f32>> = Vec::with_capacity(2 * ratio);

    // The previous half-block, through its HEAD-role projection -- or
    // phantom rows when block 0 reaches off the start of the sequence.
    for step in 0..ratio {
        if block_ord == 0 {
            kv_block.push(vec![0.0; cfg.qk_head_dim]);
            score_block.push(vec![f32::NEG_INFINITY; cfg.qk_head_dim]);
            v_block.push(vec![0.0; cfg.v_head_dim]);
            continue;
        }
        let token = (block_ord - 1) * ratio + step;
        let at = token * role_width;
        kv_block.push(state.role_kv[at..at + cfg.qk_head_dim].to_vec());
        score_block.push(state.role_scores[at..at + cfg.qk_head_dim].to_vec());
        let v_at = token * cfg.v_head_dim;
        v_block.push(state.raw_v[v_at..v_at + cfg.v_head_dim].to_vec());
    }
    // The current half-block, through its TAIL-role projection.
    for step in 0..ratio {
        let token = block_ord * ratio + step;
        let at = token * role_width + cfg.qk_head_dim;
        kv_block.push(state.role_kv[at..at + cfg.qk_head_dim].to_vec());
        score_block.push(state.role_scores[at..at + cfg.qk_head_dim].to_vec());
        let v_at = token * cfg.v_head_dim;
        v_block.push(state.raw_v[v_at..v_at + cfg.v_head_dim].to_vec());
    }

    let compressed_k = compress_block(
        &kv_block,
        &score_block,
        &weights.comp_norm,
        cfg.rms_norm_eps,
        cfg.qk_rope,
        blocks_closed,
        cfg.compress_rope_theta,
    );
    // V follows the same index structure but is not role-projected: the
    // doubled state the reference asserts on is the shared MLA latent,
    // and `v_head_dim` need not equal `qk_head_dim`, so a role split
    // here would be this skeleton's invention rather than a
    // transcription.
    let compressed_v = compress_block(
        &v_block,
        &score_block,
        &weights.comp_norm,
        cfg.rms_norm_eps,
        cfg.qk_rope,
        blocks_closed,
        cfg.compress_rope_theta,
    );
    state.compressed_k.extend_from_slice(&compressed_k);
    state.compressed_v.extend_from_slice(&compressed_v);
}

fn attn_forward_token(
    weights: &DeepseekV4AttnWeights,
    cfg: &DeepseekV4DecoderConfig,
    attn_in: &[f32],
    state: &mut DeepseekV4LayerState,
) -> Vec<f32> {
    let q = weights.q_proj.apply(attn_in);
    let k = weights.k_proj.apply(attn_in);
    let v = weights.v_proj.apply(attn_in);
    debug_assert_eq!(q.len(), cfg.n_heads * cfg.qk_head_dim);
    debug_assert_eq!(k.len(), cfg.n_heads * cfg.qk_head_dim);
    debug_assert_eq!(v.len(), cfg.n_heads * cfg.v_head_dim);

    state.raw_k.extend_from_slice(&k);
    state.raw_v.extend_from_slice(&v);
    state.token_count += 1;
    let n_raw = state.token_count;

    // A CSA layer keeps every token's two role projections, because the
    // block that closes at token t reaches back into the head-role rows
    // of the tokens before it.
    if let Some(csa) = weights.csa.as_ref() {
        for head in k.chunks(cfg.qk_head_dim) {
            state.role_kv.extend_from_slice(&csa.role_proj.apply(head));
            state
                .role_scores
                .extend_from_slice(&csa.role_gate.apply(head));
        }
    }

    let ratio = cfg.compressor.ratio() as usize;
    if ratio > 0 && n_raw.is_multiple_of(ratio) {
        match cfg.compressor {
            LayerCompressor::Hca => hca_block(weights, cfg, state, ratio),
            LayerCompressor::Csa => {
                csa_block(weights, cfg, state, ratio);
            }
            LayerCompressor::None => unreachable!("ratio 0 is filtered above"),
        }
    }

    let n_compressed = if cfg.qk_head_dim > 0 {
        state.compressed_k.len() / (cfg.n_heads * cfg.qk_head_dim)
    } else {
        0
    };
    let sinks = weights.attn_sinks.as_deref();

    // The three-way dispatch. A layer with no compressor sees the raw
    // window and nothing else -- a real entry in the shipped schedule
    // rather than a disabled state, so it runs the dense path over an
    // empty compressed set rather than being skipped.
    let mut attn_out = match cfg.compressor {
        LayerCompressor::None | LayerCompressor::Hca => hca_attention(
            &q,
            &state.raw_k,
            &state.raw_v,
            n_raw,
            &state.compressed_k,
            &state.compressed_v,
            n_compressed,
            cfg.n_heads,
            cfg.qk_head_dim,
            cfg.v_head_dim,
            sinks,
        ),
        LayerCompressor::Csa => {
            let csa = weights
                .csa
                .as_ref()
                .expect("a CSA layer must carry its role projections and indexer");
            let n_index_heads = csa.indexer_head_weights.len();
            // Projected once. Splitting the width across the index
            // heads needs the length, and computing it by projecting a
            // second time would run the indexer's matvec twice on every
            // decode step of every CSA layer.
            let projected = csa.indexer_q_proj.apply(&q[..cfg.qk_head_dim]);
            // `checked_div` rather than a guarded `/`: a layer that
            // declares no index heads has no width to split, and zero
            // is the honest answer rather than a panic.
            let index_head_dim = projected.len().checked_div(n_index_heads).unwrap_or(0);
            let indexer_q: Vec<Vec<f32>> = projected
                .chunks(index_head_dim.max(1))
                .map(|c| c.to_vec())
                .collect();
            let indexer_keys: Vec<Vec<f32>> = state
                .compressed_k
                .chunks(cfg.qk_head_dim)
                .take(n_compressed)
                .map(|entry| csa.indexer_key_proj.apply(entry))
                .collect();
            csa_attention(
                &q,
                &state.raw_k,
                &state.raw_v,
                n_raw,
                &state.compressed_k,
                &state.compressed_v,
                n_compressed,
                &indexer_q,
                &indexer_keys,
                &csa.indexer_head_weights,
                csa.indexer_top_k,
                cfg.n_heads,
                cfg.qk_head_dim,
                cfg.v_head_dim,
                sinks,
            )
        }
    };

    derope_attn_out(
        &mut attn_out,
        cfg.n_heads,
        cfg.v_head_dim,
        cfg.qk_rope,
        state.token_count.saturating_sub(1),
        cfg.compress_rope_theta,
    );

    grouped_output_projection(&attn_out, &weights.group_down, &weights.wo_b)
}

fn moe_ffn_forward(
    weights: &DeepseekV4MoeFfnWeights,
    cfg: &DeepseekV4DecoderConfig,
    x: &[f32],
) -> Vec<f32> {
    let router_logits = weights.router_weight.apply(x);
    let decision = route_top_k(
        &router_logits,
        cfg.n_experts_active,
        GatingFunction::SqrtSoftplus,
        cfg.moe_renormalize,
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

/// One decode step through the single synthetic layer, then final norm +
/// output projection. `token_id` indexes the embedding table.
pub fn deepseek_v4_forward_token(
    weights: &DeepseekV4DecoderWeights,
    cfg: &DeepseekV4DecoderConfig,
    token_id: usize,
    state: &mut DeepseekV4DecodeState,
) -> Vec<f32> {
    let hidden_dim = weights.embedding.cols();
    let hidden = weights.embedding.row(token_id).to_vec();
    state.layer.reset_hc_from_hidden(&hidden);
    let layer = &weights.layer;

    let hc_residual = state.layer.hc_streams.clone();
    let (attn_in, attn_post, attn_comb) = hc_pre(
        &layer.attn_hc_pre,
        &hc_residual,
        cfg.rms_norm_eps,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
    );
    let attn_normed = rms_norm(&attn_in, &layer.attn_norm_weight, cfg.rms_norm_eps);
    let attn_out = attn_forward_token(&layer.attn, cfg, &attn_normed, &mut state.layer);
    state.layer.hc_streams = hc_post(&attn_out, &hc_residual, &attn_post, &attn_comb);

    let hc_residual = state.layer.hc_streams.clone();
    let (ffn_in, ffn_post, ffn_comb) = hc_pre(
        &layer.ffn_hc_pre,
        &hc_residual,
        cfg.rms_norm_eps,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
    );
    let ffn_normed = rms_norm(&ffn_in, &layer.ffn_norm_weight, cfg.rms_norm_eps);
    let ffn_out = moe_ffn_forward(&layer.ffn, cfg, &ffn_normed);
    state.layer.hc_streams = hc_post(&ffn_out, &hc_residual, &ffn_post, &ffn_comb);

    let collapsed = hc_head(
        &weights.hc_head,
        &state.layer.hc_streams,
        cfg.rms_norm_eps,
        cfg.hc_eps,
    );
    let final_normed = rms_norm(&collapsed, &weights.final_norm_weight, cfg.rms_norm_eps);
    assert_eq!(final_normed.len(), hidden_dim);
    weights.output_head.apply(&final_normed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyper_connections::HyperConnectionPreWeights;

    const HIDDEN_DIM: usize = 8;
    const EPS: f32 = 1e-5;
    const NUM_HEADS: usize = 1;
    const QK_HEAD_DIM: usize = 4;
    const V_HEAD_DIM: usize = 4;
    const QK_ROPE: usize = 2;
    const N_GROUPS: usize = 1;
    const O_LORA_RANK: usize = 2;
    const O_GROUP_DIM: usize = (NUM_HEADS * V_HEAD_DIM) / N_GROUPS;
    const N_EXPERTS: usize = 4;
    const N_EXPERTS_ACTIVE: usize = 2;
    const MOE_FFN_DIM: usize = 3;
    const OUTPUT_VOCAB: usize = 5;
    const HC_FLAT: usize = HC_MULT * HIDDEN_DIM;
    const N_INDEX_HEADS: usize = 2;
    const INDEX_HEAD_DIM: usize = 2;

    fn wm(data: Vec<f32>, rows: usize, cols: usize) -> WeightMatrix {
        assert_eq!(data.len(), rows * cols);
        WeightMatrix::F32(Tensor::new(data, vec![rows, cols]))
    }

    fn synth(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((seed * 131 + i * 17 + 7) % 23) as f32 * 0.05) - 0.55)
            .collect()
    }

    fn make_hc_pre(seed: usize) -> HyperConnectionPreWeights {
        HyperConnectionPreWeights {
            fn_proj: wm(
                synth(seed, (2 + HC_MULT) * HC_MULT * HC_FLAT),
                (2 + HC_MULT) * HC_MULT,
                HC_FLAT,
            ),
            scale: [0.5, 0.5, 0.5],
            base_pre: [0.1; HC_MULT],
            base_post: [0.2; HC_MULT],
            base_comb: [0.01; HC_MULT * HC_MULT],
        }
    }

    fn make_hc_head(seed: usize) -> HyperConnectionHeadWeights {
        HyperConnectionHeadWeights {
            fn_proj: wm(synth(seed, HC_MULT * HC_FLAT), HC_MULT, HC_FLAT),
            scale: 0.5,
            base: [0.1; HC_MULT],
        }
    }

    fn make_weights_for(csa: bool, sinks: Option<Vec<f32>>) -> DeepseekV4DecoderWeights {
        let expert = |seed: usize| ExpertWeights {
            gate: wm(
                synth(seed, MOE_FFN_DIM * HIDDEN_DIM),
                MOE_FFN_DIM,
                HIDDEN_DIM,
            ),
            up: wm(
                synth(seed + 1, MOE_FFN_DIM * HIDDEN_DIM),
                MOE_FFN_DIM,
                HIDDEN_DIM,
            ),
            down: wm(
                synth(seed + 2, HIDDEN_DIM * MOE_FFN_DIM),
                HIDDEN_DIM,
                MOE_FFN_DIM,
            ),
        };

        DeepseekV4DecoderWeights {
            embedding: Tensor::new(
                synth(1000, OUTPUT_VOCAB * HIDDEN_DIM),
                vec![OUTPUT_VOCAB, HIDDEN_DIM],
            ),
            layer: DeepseekV4DecoderLayerWeights {
                attn_hc_pre: make_hc_pre(100),
                attn_norm_weight: vec![1.0; HIDDEN_DIM],
                attn: DeepseekV4AttnWeights {
                    q_proj: wm(
                        synth(110, NUM_HEADS * QK_HEAD_DIM * HIDDEN_DIM),
                        NUM_HEADS * QK_HEAD_DIM,
                        HIDDEN_DIM,
                    ),
                    k_proj: wm(
                        synth(111, NUM_HEADS * QK_HEAD_DIM * HIDDEN_DIM),
                        NUM_HEADS * QK_HEAD_DIM,
                        HIDDEN_DIM,
                    ),
                    v_proj: wm(
                        synth(112, NUM_HEADS * V_HEAD_DIM * HIDDEN_DIM),
                        NUM_HEADS * V_HEAD_DIM,
                        HIDDEN_DIM,
                    ),
                    group_down: (0..N_GROUPS)
                        .map(|g| {
                            wm(
                                synth(120 + g, O_LORA_RANK * O_GROUP_DIM),
                                O_LORA_RANK,
                                O_GROUP_DIM,
                            )
                        })
                        .collect(),
                    wo_b: wm(
                        synth(130, HIDDEN_DIM * O_LORA_RANK * N_GROUPS),
                        HIDDEN_DIM,
                        O_LORA_RANK * N_GROUPS,
                    ),
                    comp_gate: wm(
                        synth(140, QK_HEAD_DIM * QK_HEAD_DIM),
                        QK_HEAD_DIM,
                        QK_HEAD_DIM,
                    ),
                    comp_norm: vec![1.0; QK_HEAD_DIM],
                    attn_sinks: sinks,
                    csa: csa.then(|| DeepseekV4CsaWeights {
                        role_proj: wm(
                            synth(150, 2 * QK_HEAD_DIM * QK_HEAD_DIM),
                            2 * QK_HEAD_DIM,
                            QK_HEAD_DIM,
                        ),
                        role_gate: wm(
                            synth(151, 2 * QK_HEAD_DIM * QK_HEAD_DIM),
                            2 * QK_HEAD_DIM,
                            QK_HEAD_DIM,
                        ),
                        indexer_key_proj: wm(
                            synth(152, INDEX_HEAD_DIM * QK_HEAD_DIM),
                            INDEX_HEAD_DIM,
                            QK_HEAD_DIM,
                        ),
                        indexer_q_proj: wm(
                            synth(153, N_INDEX_HEADS * INDEX_HEAD_DIM * QK_HEAD_DIM),
                            N_INDEX_HEADS * INDEX_HEAD_DIM,
                            QK_HEAD_DIM,
                        ),
                        indexer_head_weights: vec![1.0; N_INDEX_HEADS],
                        indexer_top_k: 1,
                    }),
                },
                ffn_hc_pre: make_hc_pre(200),
                ffn_norm_weight: vec![1.0; HIDDEN_DIM],
                ffn: DeepseekV4MoeFfnWeights {
                    router_weight: wm(synth(300, N_EXPERTS * HIDDEN_DIM), N_EXPERTS, HIDDEN_DIM),
                    experts: (0..N_EXPERTS).map(|e| expert(400 + e * 10)).collect(),
                    shared_expert: expert(900),
                },
            },
            hc_head: make_hc_head(500),
            final_norm_weight: vec![1.0; HIDDEN_DIM],
            output_head: wm(
                synth(1100, OUTPUT_VOCAB * HIDDEN_DIM),
                OUTPUT_VOCAB,
                HIDDEN_DIM,
            ),
        }
    }

    fn make_weights() -> DeepseekV4DecoderWeights {
        make_weights_for(false, None)
    }

    fn decoder_cfg_for(compressor: LayerCompressor) -> DeepseekV4DecoderConfig {
        DeepseekV4DecoderConfig {
            rms_norm_eps: EPS,
            hc_sinkhorn_iters: 4,
            hc_eps: 1e-6,
            n_heads: NUM_HEADS,
            qk_head_dim: QK_HEAD_DIM,
            v_head_dim: V_HEAD_DIM,
            qk_rope: QK_ROPE,
            compress_rope_theta: 1_000_000.0,
            n_experts_active: N_EXPERTS_ACTIVE,
            moe_renormalize: true,
            compressor,
        }
    }

    fn decoder_cfg() -> DeepseekV4DecoderConfig {
        decoder_cfg_for(LayerCompressor::Hca)
    }

    /// One DeepSeek-V4 decode step enters the CPU worker pool once.
    ///
    /// DeepSeek-V4 Pro has no checkpoint this machine can load -- the
    /// `deepseek_v4_pro` preset is a sketch, not proof of
    /// real-checkpoint support -- so the instance is the same synthetic
    /// stack the tests below use. The count does not depend on the
    /// weights: it is how many times the driving thread crossed rayon's
    /// cold submission path, which before `engine/entry.rs` was once
    /// per parallel region and is now once per step.
    ///
    /// Sabotage: drop the `par::on_workers` from
    /// `Engine::forward_token` and this goes red with the region count.
    #[test]
    fn one_deepseek_v4_decode_step_enters_the_pool_once() {
        crate::engine::assert_one_pool_entry_per_step(
            &crate::DeepseekV4Engine {
                weights: make_weights(),
                cfg: decoder_cfg(),
            },
            0,
        );
    }

    /// All three arms of the schedule run and stay finite. The point of
    /// the dispatch is that these are three different mechanisms, so
    /// each has to be exercised as itself rather than one standing in
    /// for the others.
    #[test]
    fn every_compressor_in_the_schedule_produces_finite_logits() {
        for compressor in [
            LayerCompressor::None,
            LayerCompressor::Csa,
            LayerCompressor::Hca,
        ] {
            let weights = make_weights_for(compressor == LayerCompressor::Csa, None);
            let cfg = decoder_cfg_for(compressor);
            let mut state = DeepseekV4DecodeState::new(HIDDEN_DIM);
            // Enough tokens for CSA to close two blocks; HCA's 128 needs
            // more than any of these, which is itself the correct
            // behaviour for a layer whose first block has not closed.
            for token_id in 0..9 {
                let logits =
                    deepseek_v4_forward_token(&weights, &cfg, token_id % OUTPUT_VOCAB, &mut state);
                assert!(
                    logits.iter().all(|v| v.is_finite()),
                    "{compressor:?} token {token_id}: {logits:?}"
                );
            }
        }
    }

    /// Ratio 0 is a real entry in the shipped schedule, not a disabled
    /// state: the layer runs, and it never builds a compressed entry at
    /// any length. A uniform ratio gives these layers a compressor they
    /// do not have.
    #[test]
    fn a_layer_with_no_compressor_never_builds_a_compressed_entry() {
        let weights = make_weights_for(false, None);
        let cfg = decoder_cfg_for(LayerCompressor::None);
        let mut state = DeepseekV4DecodeState::new(HIDDEN_DIM);
        for token_id in 0..20 {
            deepseek_v4_forward_token(&weights, &cfg, token_id % OUTPUT_VOCAB, &mut state);
        }
        assert_eq!(state.layer.token_count, 20);
        assert!(
            state.layer.compressed_k.is_empty() && state.layer.compressed_v.is_empty(),
            "a ratio-0 layer compressed something"
        );
    }

    /// A compressed entry appears exactly when a block closes, and one
    /// per closed block -- for both mechanisms, at their own ratios.
    /// `LayerCompressor::visible_compressed` is the same count stated
    /// independently, so the two agreeing is the check.
    #[test]
    fn one_compressed_entry_appears_per_closed_block() {
        let weights = make_weights_for(true, None);
        let cfg = decoder_cfg_for(LayerCompressor::Csa);
        let mut state = DeepseekV4DecodeState::new(HIDDEN_DIM);
        for token in 0..12 {
            deepseek_v4_forward_token(&weights, &cfg, token % OUTPUT_VOCAB, &mut state);
            let entries = state.layer.compressed_k.len() / (NUM_HEADS * QK_HEAD_DIM);
            assert_eq!(
                entries,
                LayerCompressor::Csa.visible_compressed(token),
                "after token {token}"
            );
        }
    }

    /// The overlap: a CSA block spans `2 * ratio` rows, reaching back
    /// into the previous half-block through the head-role projection.
    /// Block 0 has no previous half-block, so it is padded with
    /// zero-KV/`-inf`-score phantom rows -- present in the block and
    /// absent from the result, which is not the same as shortening it,
    /// because the per-channel softmax normalizes over whatever rows it
    /// is given.
    ///
    /// Checked by construction: a block that did NOT reach back would
    /// be unaffected by the earlier tokens' role projections, so
    /// changing only those must change the second compressed entry.
    #[test]
    fn a_csa_block_reaches_back_into_the_previous_half_block() {
        let cfg = decoder_cfg_for(LayerCompressor::Csa);
        let ratio = LayerCompressor::Csa.ratio() as usize;

        let run = |first_tokens: &[usize]| -> Vec<f32> {
            let weights = make_weights_for(true, None);
            let mut state = DeepseekV4DecodeState::new(HIDDEN_DIM);
            for &t in first_tokens {
                deepseek_v4_forward_token(&weights, &cfg, t, &mut state);
            }
            // Then a fixed second half-block, so only the FIRST block's
            // tokens differ between the two runs.
            for t in 0..ratio {
                deepseek_v4_forward_token(&weights, &cfg, t % OUTPUT_VOCAB, &mut state);
            }
            let per_entry = NUM_HEADS * QK_HEAD_DIM;
            assert_eq!(state.layer.compressed_k.len() / per_entry, 2);
            state.layer.compressed_k[per_entry..].to_vec()
        };

        let a = run(&[0, 1, 2, 3]);
        let b = run(&[4, 3, 2, 1]);
        assert_ne!(
            a, b,
            "the second block must depend on the first half-block it overlaps"
        );
    }

    /// A per-head sink changes the answer, which is the whole reason to
    /// carry it: without one, every head must spend a full unit of
    /// weight on the keys it has.
    #[test]
    fn a_per_head_attention_sink_changes_the_output() {
        let cfg = decoder_cfg_for(LayerCompressor::None);
        let logits_for = |sinks: Option<Vec<f32>>| {
            let weights = make_weights_for(false, sinks);
            let mut state = DeepseekV4DecodeState::new(HIDDEN_DIM);
            let mut last = Vec::new();
            for token in 0..4 {
                last = deepseek_v4_forward_token(&weights, &cfg, token % OUTPUT_VOCAB, &mut state);
            }
            last
        };

        let plain = logits_for(None);
        let negligible = logits_for(Some(vec![-40.0; NUM_HEADS]));
        let dominant = logits_for(Some(vec![40.0; NUM_HEADS]));

        for (p, n) in plain.iter().zip(negligible.iter()) {
            assert!(
                (p - n).abs() < 1e-4,
                "a sink far below every score: {p} vs {n}"
            );
        }
        assert!(
            plain
                .iter()
                .zip(dominant.iter())
                .any(|(p, d)| (p - d).abs() > 1e-3),
            "a dominant sink must change the answer"
        );
        assert!(dominant.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn one_layer_synthetic_forward_produces_finite_logits() {
        let weights = make_weights();
        let cfg = decoder_cfg();
        let mut state = DeepseekV4DecodeState::new(HIDDEN_DIM);

        for token_id in 0..3 {
            let logits =
                deepseek_v4_forward_token(&weights, &cfg, token_id % OUTPUT_VOCAB, &mut state);
            assert_eq!(logits.len(), OUTPUT_VOCAB);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "token {token_id}: logits must be finite, got {logits:?}"
            );
            assert!(
                !logits.iter().any(|v| v.is_nan()),
                "token {token_id}: logits must not contain NaN, got {logits:?}"
            );
        }
    }
}
