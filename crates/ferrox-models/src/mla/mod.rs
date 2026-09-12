//! DeepSeek-style Multi-head Latent Attention (MLA): low-rank Q/KV
//! compression, with an optional sigmoid output gate (Kimi K3's real
//! addition) and an optional RoPE rotation of the decoupled `q_rot`/
//! `k_rot` slices (`MlaConfig::rope`; GLM-5.2's real addition -- see
//! below). Transcribed directly from real reference code, not guessed
//! or derived by analogy:
//!
//! 1. **Kimi K3** (`moonshotai/Kimi-K3`'s `modeling_kimi_linear.py`,
//!    `KimiMLAAttention.forward`, fetched live from the model repo): no
//!    rotary embedding is actually applied. `q_rot`/`k_rot` are named
//!    for the historical DeepSeek "rope part" split, but the real
//!    module asserts `self.use_nope` and never calls a rotary
//!    embedding function in `forward()` — the "rot" slice is just
//!    extra head-dim content, never position-rotated. Represented here
//!    as `MlaConfig::rope: None`.
//! 2. **GLM-5.2** (`zai-org/GLM-5.2`'s real `config.json`, confirmed
//!    against llama.cpp PR #25407's `src/models/glm-dsa.cpp`) DOES
//!    rotate its decoupled `q_rot`/`k_rot` slices, with the interleaved
//!    convention (`rope_interleave: true`,
//!    `ferrox_core::attention::apply_rope_interleaved`) — the opposite
//!    of the natural-but-wrong assumption the Kimi K3 module doc above
//!    warns against, for a *different* real architecture. Represented
//!    here as `MlaConfig::rope: Some(MlaRopeConfig { theta })`. `k_rot`
//!    is MQA-style (one shared vector per position, broadcast to every
//!    head — see point 4) so it's rotated once, before broadcasting;
//!    rotating a shared vector once then copying it into every head
//!    is exactly equivalent to rotating each head's copy separately,
//!    since RoPE's rotation angle depends only on position, not on the
//!    vector's per-head value.
//! 3. `kv_b_proj` expands to `num_heads * (...)`, not
//!    `num_key_value_heads * (...)`, despite `num_key_value_heads`/
//!    `num_key_value_groups` being computed in Kimi K3's real
//!    `__init__` — they go unused in `forward()`. Every query head gets
//!    its own decompressed K/V; there is no GQA-style grouping layered
//!    on top of the latent compression, which is why this module uses
//!    `ferrox_core::attention::causal_mla_attention` rather than
//!    `causal_gqa_attention`.
//!
//! When `rope` is `None` (Kimi K3's real path), a further simplification
//! applies, not present in the reference code's literal structure but
//! mathematically identical to it: the real `forward()` splits
//! `q_b_proj`'s output into `q_pass`/`q_rot` and immediately
//! re-concatenates them in the same order to form `query_states`. Since
//! nothing is inserted between the split and the concat (no rotation),
//! that round-trip is a no-op — `concat(x[..a], x[a..]) == x` — so this
//! implementation uses `q_b_proj`'s raw output directly as the query in
//! that case. When `rope` is `Some` (GLM-5.2's real path), the split is
//! no longer a no-op (rotation happens in between), so the `q_rot`
//! slice is rotated in place before attention runs.
//!
//! Not yet wired into `Decoder`'s forward pass (`AttentionKind` doesn't
//! dispatch to this yet) or into `ferrox_core::cache::KvCache` (which
//! assumes K and V share one `head_dim`, whereas MLA's K head dim
//! `qk_nope_head_dim + qk_rope_head_dim` and V head dim `v_head_dim`
//! generally differ) — both are handled by `kimi_decoder` (Kimi K3;
//! `rope: None`) and `glm_dsa`/`glm52_decoder` (GLM-5.2; `rope: Some`),
//! the dedicated decoders that consume this module. Tested here against
//! synthetic weights, cross-validated against independent Python
//! transcriptions of the same real reference algorithms for both
//! rope-disabled and rope-enabled paths.

use ferrox_core::attention::{
    apply_rope_interleaved, apply_rope_interleaved_with_freq_factors, causal_mla_attention_scaled,
};
use ferrox_core::matmul::rms_norm;
use ferrox_core::mla_absorbed::causal_mla_absorbed_attention;
use ferrox_core::weight_matrix::WeightMatrix;

use crate::config::MlaConfig;
pub use crate::mla_q_proj::MlaQProj;
use crate::mla_yarn::{kq_scale, MlaYarn};

pub struct MlaAttnWeights {
    /// Low-rank (`attn_q_a` / `attn_q_a_norm` / `attn_q_b`) or direct
    /// (`attn_q`): `crate::mla_q_proj`.
    pub q: MlaQProj,
    pub kv_a_proj_with_mqa: WeightMatrix, // [kv_lora_rank+qk_rope_head_dim, hidden_dim]
    pub kv_a_layernorm: Vec<f32>,         // [kv_lora_rank]
    /// The KV decompression, combined or split; decides which
    /// attention form runs and what the cache holds.
    pub kv_b: MlaKvB,
    pub o_proj: WeightMatrix, // [hidden_dim, n_heads*v_head_dim]
    /// Present iff `MlaConfig::use_output_gate`.
    pub g_proj: Option<WeightMatrix>, // [n_heads*v_head_dim, hidden_dim]
}

/// How a layer decompresses its latent KV, which is ALSO which
/// attention form it runs and what its cache holds
/// (`src/models/deepseek2.cpp:563-635`, the `is_mla` branch and its
/// `else`).
///
/// A converter writes ONE of the two: the combined `attn_kv_b` for a
/// legacy export and for every `plm` (`plm.cpp:35`), the split
/// `attn_k_b` / `attn_v_b` for every DeepSeek export since the `_mla`
/// keys existed (`conversion/deepseek.py:420-427`, which transposes the
/// `k` half so that `wk_b` maps `qk_nope -> kv_lora_rank`). llama.cpp
/// decides by `is_mla()`, i.e. by the `_mla` keys, and creates the
/// matching tensors; a file with the other set fails to load there.
/// The two forms compute the same attention (`ferrox_core::
/// mla_absorbed` pins it) with different memory: the naive cache is
/// `n_heads * (qk_nope + qk_rope + v)` per position, the latent one
/// `kv_lora_rank + qk_rope`.
pub enum MlaKvB {
    /// `attn_kv_b`: `[n_heads * (qk_nope + v_head_dim), kv_lora_rank]`.
    /// Expanded per position into per-head K and V; the cache is
    /// `[seq, n_heads, qk_nope + qk_rope]` + `[seq, n_heads, v_head_dim]`.
    Combined(WeightMatrix),
    /// `attn_k_b` / `attn_v_b`, one matrix per head: `k_b[h]` is
    /// `[kv_lora_rank, qk_nope]` (the absorb direction), `v_b[h]` is
    /// `[v_head_dim, kv_lora_rank]`. The query is absorbed, attention
    /// runs over the latent, and the cache is `[seq, kv_lora_rank +
    /// qk_rope]` in `k_cache` with `v_cache` unused.
    Split {
        k_b: Vec<WeightMatrix>,
        v_b: Vec<WeightMatrix>,
    },
}

impl MlaKvB {
    /// Positions the layer's `k_cache` already holds, for the form the
    /// cache is in.
    pub fn cached_positions(&self, cfg: &MlaConfig, k_cache_len: usize) -> usize {
        match self {
            MlaKvB::Combined(_) => {
                k_cache_len / (cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim))
            }
            MlaKvB::Split { .. } => k_cache_len / (cfg.kv_lora_rank + cfg.qk_rope_head_dim),
        }
    }
}

/// One decode step. `k_cache`/`v_cache` are growable, caller-owned
/// buffers -- for [`MlaKvB::Combined`] in `[seq_len_so_far, n_heads,
/// head_dim]` layout (head_dim = `qk_nope_head_dim + qk_rope_head_dim`
/// for `k`, `v_head_dim` for `v`), for [`MlaKvB::Split`] the latent
/// `[seq_len_so_far, kv_lora_rank + qk_rope_head_dim]` in `k_cache`
/// alone -- plain `Vec<f32>`, not yet `ferrox_core::cache::KvCache`
/// (see module doc comment). This function appends the current
/// position before running attention over every position pushed so
/// far.
///
/// `yarn` is the file's YaRN (`crate::mla_yarn`), an argument rather
/// than a field of `cfg` so no caller reaches the rotation or the
/// softmax scale without having answered it: the `pe` bands are
/// divided by its factors, `q_pe` / `k_pe` multiplied by its magnitude
/// after rotation, and the scale is its `kq_scale`. `None` is a plain
/// file.
#[allow(clippy::too_many_arguments)]
pub fn mla_forward_token(
    weights: &MlaAttnWeights,
    cfg: &MlaConfig,
    yarn: Option<&MlaYarn>,
    hidden: &[f32],
    rms_norm_eps: f32,
    k_cache: &mut Vec<f32>,
    v_cache: &mut Vec<f32>,
) -> Vec<f32> {
    let q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
    // Position of the token being processed this call = how many
    // positions this layer's cache already holds, before this call
    // appends one more -- the same implicit convention `seq_len` below
    // already relies on (cache length as a proxy for "how many tokens
    // this layer has processed so far").
    let pos = weights.kv_b.cached_positions(cfg, k_cache.len());

    // Without rope: `query_states` == the raw projection output; see
    // module doc comment for why the reference's split+re-concat
    // round-trip is skipped in that case. With rope (GLM-5.2, PLM): the
    // split is no longer a no-op, so `q_rot` is rotated in place per
    // head before use.
    let mut query = weights.q.apply(hidden, rms_norm_eps); // [n_heads*q_head_dim]
    if let Some(rope) = &cfg.rope {
        for h in 0..cfg.num_heads {
            let q_rot_h = &mut query[h * q_head_dim + cfg.qk_nope_head_dim..(h + 1) * q_head_dim];
            rotate_pe(q_rot_h, pos, rope.theta, yarn);
        }
    }

    let compressed_kv = weights.kv_a_proj_with_mqa.apply(hidden);
    let (k_pass_c, k_rot_raw) = compressed_kv.split_at(cfg.kv_lora_rank);
    // `k_rot` is MQA-style: one shared vector broadcast to every head,
    // not a per-head projection (real `kv_a_proj_with_mqa` name says so
    // directly, and the reference `.expand(...)`s it across heads) --
    // so with rope enabled, it's rotated once here before broadcasting
    // (see module doc comment point 2 for why that's equivalent to
    // rotating each head's copy separately).
    let mut k_rot = k_rot_raw.to_vec();
    if let Some(rope) = &cfg.rope {
        rotate_pe(&mut k_rot, pos, rope.theta, yarn);
    }
    let k_pass_c_normed = rms_norm(k_pass_c, &weights.kv_a_layernorm, rms_norm_eps);
    let scale = kq_scale(yarn, q_head_dim);

    let attn_out = match &weights.kv_b {
        MlaKvB::Split { k_b, v_b } => absorbed_attention(
            cfg,
            k_b,
            v_b,
            &query,
            &k_pass_c_normed,
            &k_rot,
            k_cache,
            scale,
        ),
        MlaKvB::Combined(kv_b_proj) => naive_attention(
            cfg,
            kv_b_proj,
            &query,
            &k_pass_c_normed,
            &k_rot,
            k_cache,
            v_cache,
            scale,
        ),
    };

    let gated = match &weights.g_proj {
        Some(g_proj) => {
            let g = g_proj.apply(hidden);
            attn_out
                .iter()
                .zip(g.iter())
                .map(|(a, g)| a * (1.0 / (1.0 + (-g).exp())))
                .collect::<Vec<f32>>()
        }
        None => attn_out,
    };

    weights.o_proj.apply(&gated)
}

/// `ggml_rope_ext` on one `pe` slice (`deepseek2.cpp:320-328`): the
/// NORM-layout rotation with YaRN's per-band divisors when the file
/// declares it, then ggml's `rope_yarn` magnitude on the rotated
/// channels -- which is every channel of this slice, since `n_rot ==
/// qk_rope`.
fn rotate_pe(slice: &mut [f32], pos: usize, theta: f32, yarn: Option<&MlaYarn>) {
    match yarn {
        None => apply_rope_interleaved(slice, pos, theta),
        Some(y) => {
            apply_rope_interleaved_with_freq_factors(slice, pos, theta, &y.freq_factors);
            if y.pe_magnitude != 1.0 {
                for v in slice.iter_mut() {
                    *v *= y.pe_magnitude;
                }
            }
        }
    }
}

/// `deepseek2.cpp:563-598`: absorb `q_nope` through `wk_b`, attend as
/// MQA over the latent `concat(c, k_pe)`, pull the weighted latent
/// through `wv_b`. `scale` is `kq_scale`, the UNabsorbed head width's
/// with YaRN's `mscale^2` folded in (`ferrox_core::mla_absorbed`,
/// `crate::mla_yarn`). Returns `[n_heads * v_head_dim]`.
#[allow(clippy::too_many_arguments)]
fn absorbed_attention(
    cfg: &MlaConfig,
    k_b: &[WeightMatrix],
    v_b: &[WeightMatrix],
    query: &[f32],
    k_pass_c_normed: &[f32],
    k_rot: &[f32],
    k_cache: &mut Vec<f32>,
    scale: f32,
) -> Vec<f32> {
    let q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
    let width = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
    let mut q_abs = vec![0f32; cfg.num_heads * width];
    for h in 0..cfg.num_heads {
        let q_h = &query[h * q_head_dim..(h + 1) * q_head_dim];
        let absorbed = k_b[h].apply(&q_h[..cfg.qk_nope_head_dim]);
        q_abs[h * width..h * width + cfg.kv_lora_rank].copy_from_slice(&absorbed);
        q_abs[h * width + cfg.kv_lora_rank..(h + 1) * width]
            .copy_from_slice(&q_h[cfg.qk_nope_head_dim..]);
    }
    k_cache.extend_from_slice(k_pass_c_normed);
    k_cache.extend_from_slice(k_rot);
    let seq_len = k_cache.len() / width;
    let out_lat = causal_mla_absorbed_attention(
        &q_abs,
        k_cache,
        cfg.num_heads,
        cfg.kv_lora_rank,
        cfg.qk_rope_head_dim,
        seq_len,
        scale,
    );
    let mut out = vec![0f32; cfg.num_heads * cfg.v_head_dim];
    for h in 0..cfg.num_heads {
        let v_h = v_b[h].apply(&out_lat[h * cfg.kv_lora_rank..(h + 1) * cfg.kv_lora_rank]);
        out[h * cfg.v_head_dim..(h + 1) * cfg.v_head_dim].copy_from_slice(&v_h);
    }
    out
}

/// The naive form (`deepseek2.cpp:600-635`, `plm.cpp:120-166`): expand
/// the normed latent through the combined `kv_b` into this position's
/// per-head `k_nope` and `v`, append per-head K (the shared roped `k_pe`
/// on every head) and V, attend with per-head caches. Returns
/// `[n_heads * v_head_dim]`.
#[allow(clippy::too_many_arguments)]
fn naive_attention(
    cfg: &MlaConfig,
    kv_b_proj: &WeightMatrix,
    query: &[f32],
    k_pass_c_normed: &[f32],
    k_rot: &[f32],
    k_cache: &mut Vec<f32>,
    v_cache: &mut Vec<f32>,
    scale: f32,
) -> Vec<f32> {
    let q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
    let k_pass_full = kv_b_proj.apply(k_pass_c_normed); // [n_heads*(qk_nope_head_dim+v_head_dim)]

    let mut key_step = vec![0f32; cfg.num_heads * q_head_dim];
    let mut value_step = vec![0f32; cfg.num_heads * cfg.v_head_dim];
    let kpf_stride = cfg.qk_nope_head_dim + cfg.v_head_dim;
    for h in 0..cfg.num_heads {
        let k_pass = &k_pass_full[h * kpf_stride..h * kpf_stride + cfg.qk_nope_head_dim];
        let v_h = &k_pass_full[h * kpf_stride + cfg.qk_nope_head_dim..(h + 1) * kpf_stride];

        let key_h = &mut key_step[h * q_head_dim..(h + 1) * q_head_dim];
        key_h[..cfg.qk_nope_head_dim].copy_from_slice(k_pass);
        key_h[cfg.qk_nope_head_dim..].copy_from_slice(k_rot);

        value_step[h * cfg.v_head_dim..(h + 1) * cfg.v_head_dim].copy_from_slice(v_h);
    }

    k_cache.extend_from_slice(&key_step);
    v_cache.extend_from_slice(&value_step);
    let seq_len = k_cache.len() / (cfg.num_heads * q_head_dim);

    causal_mla_attention_scaled(
        query,
        k_cache,
        v_cache,
        cfg.num_heads,
        q_head_dim,
        cfg.v_head_dim,
        seq_len,
        scale,
    )
}

#[cfg(test)]
mod tests;
