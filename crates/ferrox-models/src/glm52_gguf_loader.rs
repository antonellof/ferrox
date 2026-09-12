//! Loads `ferrox-models::glm52_decoder` weights from a GLM-5.2 GGUF
//! checkpoint, via `ferrox_gguf::TensorSource` -- the same trait
//! `ferrox-models::loader`'s generic GQA path and
//! `kimi_gguf_loader`'s Kimi K3 path both use. Follows
//! `kimi_gguf_loader.rs`'s dedicated-loader pattern (a hand-written
//! loader for an architecture whose MLA+DSA structure doesn't fit the
//! generic GQA loader), not `loader.rs`'s generic path.
//!
//! Every tensor name/shape here is confirmed against real, inspectable
//! upstream source, not guessed: `ggerganov/llama.cpp` PR #23346
//! (DeepSeek-V3.2's DSA graph builder, `src/models/deepseek32.cpp`,
//! which GLM-5.2's own model file is a fork of) and PR #25407
//! (GLM-5.2's `indexer_types` diff on top,
//! `src/models/glm-dsa.cpp::load_arch_tensors`'s real
//! `create_tensor(tn(LLM_TENSOR_...))` calls) -- fetched live via
//! `gh api -H "Accept: application/vnd.github.raw"
//! repos/ggerganov/llama.cpp/contents/src/models/glm-dsa.cpp` (rather
//! than `gh pr diff`, since PR #25407 doesn't touch most of these
//! `create_tensor` calls -- they predate it, inherited unchanged from
//! the DeepSeek-V3.2 fork point) and cross-checked against
//! `src/llama-arch.cpp`'s real `LLM_TENSOR_NAMES` table for the exact
//! on-disk name strings. This loader has NOT been run against a real
//! GLM-5.2 GGUF file (~744B params, no feasible download/quant exists
//! at a size this environment could hold) -- it is real, inspectable
//! code built from real upstream evidence, tested here against a small
//! synthetic on-disk fixture, the same rigor `kimi_gguf_loader.rs`
//! documents for its own untested-against-a-real-file status.
//!
//! One real, non-obvious fact this loader has to account for:
//! **the real on-disk `attn_k_b`/`attn_v_b` tensors are separate
//! per-head 3D tensors, not a combined 2D `kv_b_proj` the way Kimi K3's
//! real checkpoint stores it.** `attn_k_b`'s real ggml `ne[]` is
//! `[qk_nope_head_dim, kv_lora_rank, n_head]` -- llama.cpp applies it in
//! the "absorbed" direction (`ggml_mul_mat(wk_b, q_nope)`, projecting
//! the *query* into compressed space, a compute optimization) rather
//! than decompressing K directly. This loader instead TRANSPOSES each
//! head's slice at load time into `[qk_nope_head_dim, kv_lora_rank]`
//! (the direct decompression direction), so `glm_dsa`'s attention
//! forward pass can reuse the same un-absorbed math
//! `ferrox_models::mla` already has via `causal_mla_attention_sparse`,
//! instead of implementing a second, absorbed-computation attention
//! primitive purely for compute-efficiency parity with llama.cpp (which
//! this CPU reference implementation doesn't need). `attn_v_b`'s real
//! `ne[]` is `[kv_lora_rank, v_head_dim, n_head]`, which is *already*
//! in the needed decompression direction per head (`[v_head_dim,
//! kv_lora_rank]`) -- no transpose needed there. The transpose is only
//! implemented for F32/BF16 (this loader dequantizes to f32 first, then
//! transposes elementwise) -- a real, disclosed gap for quantized
//! `attn_k_b` tensors specifically, not silently wrong: `load_wk_b_head`
//! returns a clear `LoadError::UnsupportedDtype` for any other quant
//! kind rather than guessing.

use ferrox_core::tensor::Tensor;
use ferrox_core::weight_matrix::quant_kind_for;
use ferrox_core::weight_matrix::{WeightBytes, WeightMatrix};
use ferrox_gguf::{GgmlType, TensorSource};

use crate::glm_dsa::{Glm52AttnWeights, Glm52MlaConfig, IndexerConfig, IndexerWeights};
use crate::loader::LoadError;
use crate::loader::{find_info, load_f32_vec, load_weight_matrix, split_expert_tensor};

/// Splits `blk.N.attn_k_b.weight` (real on-disk ggml `ne[]` =
/// `[qk_nope_head_dim, kv_lora_rank, n_head]`, i.e. per head, physically
/// `kv_lora_rank` rows of `qk_nope_head_dim` floats each) into per-head
/// `WeightMatrix`es, TRANSPOSED into `[qk_nope_head_dim, kv_lora_rank]`
/// -- see module doc comment for why. F32/BF16 only (a real, disclosed
/// gap for quantized `attn_k_b`, not a silent wrong-shape read).
fn load_wk_b_transposed(
    file: &impl TensorSource,
    name: &str,
    n_head: usize,
    qk_nope_head_dim: usize,
    kv_lora_rank: usize,
) -> Result<Vec<WeightMatrix>, LoadError> {
    let info = find_info(file, name)?;
    if info.shape.len() != 3
        || info.shape[0] as usize != qk_nope_head_dim
        || info.shape[1] as usize != kv_lora_rank
        || info.shape[2] as usize != n_head
    {
        return Err(LoadError::UnsupportedDtype(
            format!(
                "{name} (expected ne=[{qk_nope_head_dim}, {kv_lora_rank}, {n_head}], got {:?})",
                info.shape
            ),
            info.dtype,
        ));
    }
    if !matches!(info.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        return Err(LoadError::UnsupportedDtype(name.to_string(), info.dtype));
    }
    let all = load_f32_vec(file, name)?;
    let per_head = kv_lora_rank * qk_nope_head_dim;
    Ok((0..n_head)
        .map(|h| {
            let head_raw = &all[h * per_head..(h + 1) * per_head]; // [kv_lora_rank, qk_nope_head_dim] row-major
            let mut transposed = vec![0f32; per_head]; // [qk_nope_head_dim, kv_lora_rank] row-major
            for row in 0..kv_lora_rank {
                for col in 0..qk_nope_head_dim {
                    transposed[col * kv_lora_rank + row] = head_raw[row * qk_nope_head_dim + col];
                }
            }
            WeightMatrix::F32(Tensor::new(
                transposed,
                vec![qk_nope_head_dim, kv_lora_rank],
            ))
        })
        .collect())
}

/// Splits `blk.N.attn_v_b.weight` (real on-disk ggml `ne[]` =
/// `[kv_lora_rank, v_head_dim, n_head]`) into per-head `WeightMatrix`es
/// -- already in the needed decompression direction (`[v_head_dim,
/// kv_lora_rank]` per head), no transpose required, unlike `attn_k_b`.
fn load_wv_b(
    file: &impl TensorSource,
    name: &str,
    n_head: usize,
    kv_lora_rank: usize,
    v_head_dim: usize,
) -> Result<Vec<WeightMatrix>, LoadError> {
    let info = find_info(file, name)?;
    if info.shape.len() != 3
        || info.shape[0] as usize != kv_lora_rank
        || info.shape[1] as usize != v_head_dim
        || info.shape[2] as usize != n_head
    {
        return Err(LoadError::UnsupportedDtype(
            format!(
                "{name} (expected ne=[{kv_lora_rank}, {v_head_dim}, {n_head}], got {:?})",
                info.shape
            ),
            info.dtype,
        ));
    }
    match info.dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
            let all = load_f32_vec(file, name)?;
            let per_head = kv_lora_rank * v_head_dim;
            Ok((0..n_head)
                .map(|h| {
                    WeightMatrix::F32(Tensor::new(
                        all[h * per_head..(h + 1) * per_head].to_vec(),
                        vec![v_head_dim, kv_lora_rank],
                    ))
                })
                .collect())
        }
        other => match quant_kind_for(other) {
            Some(kind) => {
                let (mmap, full_range) = file.tensor_mapped_range(name)?;
                let bytes_per_head = (full_range.end - full_range.start) / n_head;
                Ok((0..n_head)
                    .map(|h| WeightMatrix::Quantized {
                        data: WeightBytes::Mapped {
                            mmap: std::sync::Arc::clone(&mmap),
                            range: (full_range.start + h * bytes_per_head)
                                ..(full_range.start + (h + 1) * bytes_per_head),
                        },
                        rows: v_head_dim,
                        cols: kv_lora_rank,
                        kind,
                    })
                    .collect())
            }
            None => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
        },
    }
}

/// Real per-layer/global hyperparameters needed to load a GLM-5.2 GGUF
/// file -- see docs/MODELS.md's "GLM-5.2 (Z.ai)" section for the
/// real, confirmed values (78 layers, `hidden_size`=6144, etc.).
pub struct Glm52GgufHparams {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rope_theta: f32,
    pub indexer_n_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_rope_dim: usize,
    pub indexer_top_k: usize,
    pub dense_ffn_dim: usize,
    pub moe_ffn_dim: usize,
    pub n_experts: usize,
    pub n_shared_experts: usize,
}

/// Loads one GLM-5.2 attention layer's weights. `is_full_indexer_layer`
/// controls whether the real (`TENSOR_NOT_REQUIRED`-flagged, but always
/// present for a "full" layer in a real checkpoint) indexer tensors are
/// read at all -- "shared" layers carry no indexer weights of their own
/// (see `glm_dsa`'s module doc comment point 1).
pub fn load_glm52_attn(
    file: &impl TensorSource,
    hp: &Glm52GgufHparams,
    layer_idx: usize,
    is_full_indexer_layer: bool,
) -> Result<Glm52AttnWeights, LoadError> {
    let l = layer_idx;
    let q_head_dim = hp.qk_nope_head_dim + hp.qk_rope_head_dim;

    let q_a_proj = load_weight_matrix(file, &format!("blk.{l}.attn_q_a.weight"))?;
    assert_eq!(
        q_a_proj.rows(),
        hp.q_lora_rank,
        "blk.{l}.attn_q_a.weight row count"
    );
    assert_eq!(
        q_a_proj.cols(),
        hp.hidden_dim,
        "blk.{l}.attn_q_a.weight col count"
    );

    let q_b_proj = load_weight_matrix(file, &format!("blk.{l}.attn_q_b.weight"))?;
    assert_eq!(
        q_b_proj.rows(),
        hp.num_heads * q_head_dim,
        "blk.{l}.attn_q_b.weight row count"
    );

    let kv_a_proj_with_mqa = load_weight_matrix(file, &format!("blk.{l}.attn_kv_a_mqa.weight"))?;
    assert_eq!(
        kv_a_proj_with_mqa.rows(),
        hp.kv_lora_rank + hp.qk_rope_head_dim,
        "blk.{l}.attn_kv_a_mqa.weight row count"
    );

    let wk_b = load_wk_b_transposed(
        file,
        &format!("blk.{l}.attn_k_b.weight"),
        hp.num_heads,
        hp.qk_nope_head_dim,
        hp.kv_lora_rank,
    )?;
    let wv_b = load_wv_b(
        file,
        &format!("blk.{l}.attn_v_b.weight"),
        hp.num_heads,
        hp.kv_lora_rank,
        hp.v_head_dim,
    )?;

    let o_proj = load_weight_matrix(file, &format!("blk.{l}.attn_output.weight"))?;
    assert_eq!(
        o_proj.rows(),
        hp.hidden_dim,
        "blk.{l}.attn_output.weight row count"
    );
    assert_eq!(
        o_proj.cols(),
        hp.num_heads * hp.v_head_dim,
        "blk.{l}.attn_output.weight col count"
    );

    let indexer = if is_full_indexer_layer {
        let k_norm_weight = load_f32_vec(file, &format!("blk.{l}.indexer.k_norm.weight"))?;
        let k_norm_bias = load_f32_vec(file, &format!("blk.{l}.indexer.k_norm.bias"))?;
        let proj = load_weight_matrix(file, &format!("blk.{l}.indexer.proj.weight"))?;
        assert_eq!(
            proj.rows(),
            hp.indexer_n_heads,
            "blk.{l}.indexer.proj.weight row count"
        );
        let attn_k = load_weight_matrix(file, &format!("blk.{l}.indexer.attn_k.weight"))?;
        assert_eq!(
            attn_k.rows(),
            hp.indexer_head_dim,
            "blk.{l}.indexer.attn_k.weight row count"
        );
        let attn_q_b = load_weight_matrix(file, &format!("blk.{l}.indexer.attn_q_b.weight"))?;
        assert_eq!(
            attn_q_b.rows(),
            hp.indexer_n_heads * hp.indexer_head_dim,
            "blk.{l}.indexer.attn_q_b.weight row count"
        );
        Some(IndexerWeights {
            k_norm_weight,
            k_norm_bias,
            proj,
            attn_k,
            attn_q_b,
        })
    } else {
        None
    };

    Ok(Glm52AttnWeights {
        q_a_proj,
        q_a_layernorm: load_f32_vec(file, &format!("blk.{l}.attn_q_a_norm.weight"))?,
        q_b_proj,
        kv_a_proj_with_mqa,
        kv_a_layernorm: load_f32_vec(file, &format!("blk.{l}.attn_kv_a_norm.weight"))?,
        wk_b,
        wv_b,
        o_proj,
        indexer,
    })
}

pub fn glm52_mla_config(hp: &Glm52GgufHparams) -> Glm52MlaConfig {
    Glm52MlaConfig {
        num_heads: hp.num_heads,
        q_lora_rank: hp.q_lora_rank,
        kv_lora_rank: hp.kv_lora_rank,
        qk_nope_head_dim: hp.qk_nope_head_dim,
        qk_rope_head_dim: hp.qk_rope_head_dim,
        v_head_dim: hp.v_head_dim,
        rope: crate::config::MlaRopeConfig {
            theta: hp.rope_theta,
        },
    }
}

pub fn glm52_indexer_config(hp: &Glm52GgufHparams) -> IndexerConfig {
    IndexerConfig {
        n_heads: hp.indexer_n_heads,
        head_dim: hp.indexer_head_dim,
        rope_dim: hp.indexer_rope_dim,
        top_k: hp.indexer_top_k,
        rope_theta: hp.rope_theta,
    }
}

/// Loads one dense leading layer's feed-forward block (real tensor
/// names `blk.{bid}.ffn_{gate,down,up}` -- same convention every other
/// architecture's dense FFN uses in this codebase).
pub struct Glm52DenseFfnWeights {
    pub gate_proj: WeightMatrix,
    pub up_proj: WeightMatrix,
    pub down_proj: WeightMatrix,
}

pub fn load_glm52_dense_ffn(
    file: &impl TensorSource,
    layer_idx: usize,
) -> Result<Glm52DenseFfnWeights, LoadError> {
    let l = layer_idx;
    Ok(Glm52DenseFfnWeights {
        gate_proj: load_weight_matrix(file, &format!("blk.{l}.ffn_gate.weight"))?,
        up_proj: load_weight_matrix(file, &format!("blk.{l}.ffn_up.weight"))?,
        down_proj: load_weight_matrix(file, &format!("blk.{l}.ffn_down.weight"))?,
    })
}

/// One routed expert's gate/up/down weights (SwiGLU FFN) --
/// `ferrox_moe::ExpertWeights`'s field names/order.
pub struct Glm52MoeFfnWeights {
    pub router_weight: WeightMatrix,
    pub e_score_correction_bias: Vec<f32>,
    pub experts: Vec<ferrox_moe::ExpertWeights>,
    pub shared_expert: ferrox_moe::ExpertWeights,
}

pub fn load_glm52_moe_ffn(
    file: &impl TensorSource,
    hp: &Glm52GgufHparams,
    layer_idx: usize,
) -> Result<Glm52MoeFfnWeights, LoadError> {
    let l = layer_idx;
    let gate_exps =
        split_expert_tensor(file, &format!("blk.{l}.ffn_gate_exps.weight"), hp.n_experts)?;
    let down_exps =
        split_expert_tensor(file, &format!("blk.{l}.ffn_down_exps.weight"), hp.n_experts)?;
    let up_exps = split_expert_tensor(file, &format!("blk.{l}.ffn_up_exps.weight"), hp.n_experts)?;
    let experts = gate_exps
        .into_iter()
        .zip(down_exps)
        .zip(up_exps)
        .map(|((gate, down), up)| ferrox_moe::ExpertWeights { gate, up, down })
        .collect();

    let shared_expert = ferrox_moe::ExpertWeights {
        gate: load_weight_matrix(file, &format!("blk.{l}.ffn_gate_shexp.weight"))?,
        up: load_weight_matrix(file, &format!("blk.{l}.ffn_up_shexp.weight"))?,
        down: load_weight_matrix(file, &format!("blk.{l}.ffn_down_shexp.weight"))?,
    };

    Ok(Glm52MoeFfnWeights {
        router_weight: load_weight_matrix(file, &format!("blk.{l}.ffn_gate_inp.weight"))?,
        // `LLM_TENSOR_FFN_EXP_PROBS_B` is spelled `blk.%d.exp_probs_b`
        // on disk -- no `ffn_` prefix (llama-arch.cpp:416,
        // gguf-py/gguf/constants.py:1240). This asked for a name no real
        // checkpoint carries, and the synthetic fixture below wrote the
        // same wrong name, so the tests agreed with the bug.
        e_score_correction_bias: load_f32_vec(file, &format!("blk.{l}.exp_probs_b.bias"))?,
        experts,
        shared_expert,
    })
}

fn meta_u64(file: &impl TensorSource, key: &str) -> Result<u64, LoadError> {
    file.metadata_u64(key)
        .ok_or_else(|| LoadError::MissingHparam(key.to_string()))
}

fn meta_f32(file: &impl TensorSource, key: &str, default: f32) -> f32 {
    file.metadata_f32(key).unwrap_or(default)
}

/// GGUF-side metadata needed alongside [`Glm52GgufHparams`] to build
/// [`crate::engine::Glm52Engine`].
pub struct Glm52FileMeta {
    pub arch: String,
    pub n_layer: usize,
    pub leading_dense: usize,
    pub rms_norm_eps: f32,
    pub n_experts_active: usize,
    pub moe_renormalize: bool,
    pub routed_scaling_factor: f32,
}

/// Read GLM-5.2 / GLM4 hparams from an opened GGUF (`glm-dsa`, `glm4`).
pub fn read_glm52_hparams(
    file: &impl TensorSource,
) -> Result<(Glm52GgufHparams, Glm52FileMeta), LoadError> {
    let arch = file
        .metadata_str("general.architecture")
        .ok_or_else(|| LoadError::MissingHparam("general.architecture".into()))?
        .to_string();
    // `glm4moe` is deliberately not accepted: it is plain GQA on the
    // generic path (tests/glm4moe_graphs.rs), and this loader's MLA keys
    // are ones it never carries.
    if !matches!(arch.as_str(), "glm-dsa" | "glm4") {
        return Err(LoadError::UnsupportedArchitecture(arch));
    }
    let p = |suffix: &str| format!("{arch}.{suffix}");
    // The trunk: `block_count` minus the NextN/MTP blocks llama.cpp
    // never runs, decided once for every loader in `crate::mtp_blocks`.
    let n_layer =
        crate::mtp_blocks::trunk_layers(file, &arch, meta_u64(file, &p("block_count"))? as usize)?
            .n_layers;
    let hidden_dim = meta_u64(file, &p("embedding_length"))? as usize;
    let dense_ffn_dim = meta_u64(file, &p("feed_forward_length"))? as usize;
    let moe_ffn_dim = file
        .metadata_u64(&p("expert_feed_forward_length"))
        .unwrap_or(dense_ffn_dim as u64) as usize;
    let n_heads = meta_u64(file, &p("attention.head_count"))? as usize;
    let q_lora_rank = meta_u64(file, &p("attention.q_lora_rank"))? as usize;
    let kv_lora_rank = meta_u64(file, &p("attention.kv_lora_rank"))? as usize;
    let qk_nope_head_dim = meta_u64(file, &p("attention.qk_nope_head_dim"))? as usize;
    let qk_rope_head_dim = meta_u64(file, &p("attention.qk_rope_head_dim"))? as usize;
    let v_head_dim = file
        .metadata_u64(&p("attention.v_head_dim"))
        .or_else(|| file.metadata_u64(&p("attention.key_length")))
        .unwrap_or(qk_nope_head_dim as u64) as usize;
    let leading_dense = file
        .metadata_u64(&p("leading_dense_block_count"))
        .unwrap_or(0) as usize;
    let n_experts = file.metadata_u64(&p("expert_count")).unwrap_or(0) as usize;
    let n_shared_experts = file.metadata_u64(&p("expert_shared_count")).unwrap_or(1) as usize;
    let n_experts_active = file.metadata_u64(&p("expert_used_count")).unwrap_or(8) as usize;
    let indexer_n_heads = file
        .metadata_u64(&p("attention.indexer_n_heads"))
        .unwrap_or(4) as usize;
    let indexer_head_dim = file
        .metadata_u64(&p("attention.indexer_head_dim"))
        .unwrap_or(128) as usize;
    let indexer_top_k = file
        .metadata_u64(&p("attention.indexer_top_k"))
        .unwrap_or(2048) as usize;
    let hp = Glm52GgufHparams {
        hidden_dim,
        num_heads: n_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        rope_theta: meta_f32(file, &p("rope.freq_base"), 1_000_000.0),
        indexer_n_heads,
        indexer_head_dim,
        indexer_rope_dim: qk_rope_head_dim,
        indexer_top_k,
        dense_ffn_dim,
        moe_ffn_dim,
        n_experts,
        n_shared_experts,
    };
    let meta = Glm52FileMeta {
        arch: arch.clone(),
        n_layer,
        leading_dense: leading_dense.min(n_layer),
        rms_norm_eps: meta_f32(file, &p("attention.layer_norm_rms_epsilon"), 1e-5),
        n_experts_active,
        moe_renormalize: file
            .metadata_u64(&p("expert_norm_topk_prob"))
            .is_some_and(|v| v != 0),
        routed_scaling_factor: meta_f32(file, &p("expert_routing_scale"), 2.5),
    };
    Ok((hp, meta))
}

fn is_full_indexer_layer(file: &impl TensorSource, layer_idx: usize) -> bool {
    file.find_tensor(&format!("blk.{layer_idx}.indexer.proj.weight"))
        .is_some()
}

fn is_dense_ffn_layer(file: &impl TensorSource, layer_idx: usize, leading_dense: usize) -> bool {
    if layer_idx < leading_dense {
        return true;
    }
    file.find_tensor(&format!("blk.{layer_idx}.ffn_gate.weight"))
        .is_some()
        && file
            .find_tensor(&format!("blk.{layer_idx}.ffn_gate_inp.weight"))
            .is_none()
}

fn load_embedding_tensor(
    file: &impl TensorSource,
    hidden_dim: usize,
) -> Result<ferrox_core::tensor::Tensor, LoadError> {
    let wm = load_weight_matrix(file, "token_embd.weight")?;
    let vocab = wm.rows();
    assert_eq!(wm.cols(), hidden_dim, "token_embd.weight col count");
    let mut data = vec![0f32; vocab * hidden_dim];
    for row in 0..vocab {
        let r = wm.dequant_row(row);
        data[row * hidden_dim..(row + 1) * hidden_dim].copy_from_slice(&r);
    }
    Ok(ferrox_core::tensor::Tensor::new(
        data,
        vec![vocab, hidden_dim],
    ))
}

/// Load a GLM-5.2 / GLM4-family GGUF into [`crate::engine::Glm52Engine`].
pub fn load_glm52_engine(
    file: &impl TensorSource,
) -> Result<crate::engine::Glm52Engine, LoadError> {
    use crate::glm52_decoder::{
        Glm52DecoderConfig, Glm52DecoderLayerWeights, Glm52DecoderWeights,
        Glm52DenseFfnWeights as DecDenseFfn, Glm52LayerFfn, Glm52MoeFfnWeights as DecMoeFfn,
    };

    let (hp, meta) = read_glm52_hparams(file)?;
    let embedding = load_embedding_tensor(file, hp.hidden_dim)?;
    let final_norm_weight = load_f32_vec(file, "output_norm.weight")?;
    let output_head = match load_weight_matrix(file, "output.weight") {
        Ok(w) => w,
        Err(_) => load_weight_matrix(file, "token_embd.weight")?,
    };

    let mut layers = Vec::with_capacity(meta.n_layer);
    for layer_idx in 0..meta.n_layer {
        let is_full = is_full_indexer_layer(file, layer_idx);
        let attn = load_glm52_attn(file, &hp, layer_idx, is_full)?;
        let ffn = if is_dense_ffn_layer(file, layer_idx, meta.leading_dense) {
            let d = load_glm52_dense_ffn(file, layer_idx)?;
            Glm52LayerFfn::Dense(Box::new(DecDenseFfn {
                gate_proj: d.gate_proj,
                up_proj: d.up_proj,
                down_proj: d.down_proj,
            }))
        } else {
            let m = load_glm52_moe_ffn(file, &hp, layer_idx)?;
            Glm52LayerFfn::Moe(Box::new(DecMoeFfn {
                router_weight: m.router_weight,
                e_score_correction_bias: m.e_score_correction_bias,
                experts: m.experts,
                shared_expert: m.shared_expert,
            }))
        };
        layers.push(Glm52DecoderLayerWeights {
            attn_norm_weight: load_f32_vec(file, &format!("blk.{layer_idx}.attn_norm.weight"))?,
            attn,
            ffn_norm_weight: load_f32_vec(file, &format!("blk.{layer_idx}.ffn_norm.weight"))?,
            ffn,
            is_full_indexer_layer: is_full,
        });
    }

    let weights = Glm52DecoderWeights {
        embedding,
        layers,
        final_norm_weight,
        output_head,
    };
    let cfg = Glm52DecoderConfig {
        rms_norm_eps: meta.rms_norm_eps,
        mla: glm52_mla_config(&hp),
        indexer: glm52_indexer_config(&hp),
        n_experts_active: meta.n_experts_active,
        moe_renormalize: meta.moe_renormalize,
        routed_scaling_factor: meta.routed_scaling_factor,
    };
    Ok(crate::engine::Glm52Engine { weights, cfg })
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    struct FixtureTensor {
        name: String,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    }

    fn f32_tensor(name: impl Into<String>, shape: Vec<u64>, values: Vec<f32>) -> FixtureTensor {
        FixtureTensor {
            name: name.into(),
            shape,
            bytes: f32_bytes(&values),
        }
    }

    /// Builds a real, parseable on-disk GGUF file from a flat list of
    /// tensors -- same builder pattern as
    /// `kimi_gguf_loader::tests::build_gguf` (duplicated, not shared,
    /// matching that module's own precedent), F32-only since none of
    /// the hparams under test here are quantized-format-sensitive.
    fn build_gguf(arch: &str, tensors: &[FixtureTensor]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count: just general.architecture

        let write_string = |buf: &mut Vec<u8>, s: &str| {
            buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
            buf.write_all(s.as_bytes()).unwrap();
        };
        write_string(&mut buf, "general.architecture");
        buf.write_u32::<LittleEndian>(8).unwrap(); // type = string
        write_string(&mut buf, arch);

        let mut offset = 0u64;
        let mut offsets = Vec::with_capacity(tensors.len());
        for t in tensors {
            write_string(&mut buf, &t.name);
            buf.write_u32::<LittleEndian>(t.shape.len() as u32).unwrap();
            for &d in t.shape.iter().rev() {
                buf.write_u64::<LittleEndian>(d).unwrap();
            }
            buf.write_u32::<LittleEndian>(0).unwrap(); // dtype tag: F32
            offsets.push(offset);
            buf.write_u64::<LittleEndian>(offset).unwrap();
            let padded = t.bytes.len().div_ceil(32) * 32;
            offset += padded as u64;
        }

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let data_start = buf.len();
        for (t, &off) in tensors.iter().zip(offsets.iter()) {
            let want_len = data_start + off as usize;
            while buf.len() < want_len {
                buf.push(0);
            }
            buf.extend_from_slice(&t.bytes);
            while buf.len() % 32 != 0 {
                buf.push(0);
            }
        }
        buf
    }

    struct Dims {
        hidden_dim: usize,
        num_heads: usize,
        q_lora_rank: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        indexer_n_heads: usize,
        indexer_head_dim: usize,
        dense_ffn_dim: usize,
        moe_ffn_dim: usize,
        n_experts: usize,
        n_shared_experts: usize,
    }

    fn push_layer_tensors(
        tensors: &mut Vec<FixtureTensor>,
        l: usize,
        is_full: bool,
        is_dense: bool,
        d: &Dims,
    ) {
        let h = d.hidden_dim;
        let q_head_dim = d.qk_nope_head_dim + d.qk_rope_head_dim;

        tensors.push(f32_tensor(
            format!("blk.{l}.attn_norm.weight"),
            vec![h as u64],
            vec![1.0; h],
        ));
        tensors.push(f32_tensor(
            format!("blk.{l}.ffn_norm.weight"),
            vec![h as u64],
            vec![1.0; h],
        ));
        tensors.push(f32_tensor(
            format!("blk.{l}.attn_q_a_norm.weight"),
            vec![d.q_lora_rank as u64],
            vec![1.0; d.q_lora_rank],
        ));
        tensors.push(f32_tensor(
            format!("blk.{l}.attn_kv_a_norm.weight"),
            vec![d.kv_lora_rank as u64],
            vec![1.0; d.kv_lora_rank],
        ));
        tensors.push(f32_tensor(
            format!("blk.{l}.attn_q_a.weight"),
            vec![d.q_lora_rank as u64, h as u64],
            vec![0.02; d.q_lora_rank * h],
        ));
        tensors.push(f32_tensor(
            format!("blk.{l}.attn_q_b.weight"),
            vec![(d.num_heads * q_head_dim) as u64, d.q_lora_rank as u64],
            vec![0.02; d.num_heads * q_head_dim * d.q_lora_rank],
        ));
        tensors.push(f32_tensor(
            format!("blk.{l}.attn_kv_a_mqa.weight"),
            vec![(d.kv_lora_rank + d.qk_rope_head_dim) as u64, h as u64],
            vec![0.02; (d.kv_lora_rank + d.qk_rope_head_dim) * h],
        ));
        // Real ne = [qk_nope_head_dim, kv_lora_rank, n_head]; this
        // builder reverses its `shape` arg once to produce the written
        // raw ne, so the argument here must be the REVERSE of that.
        tensors.push(f32_tensor(
            format!("blk.{l}.attn_k_b.weight"),
            vec![
                d.num_heads as u64,
                d.kv_lora_rank as u64,
                d.qk_nope_head_dim as u64,
            ],
            vec![0.02; d.num_heads * d.kv_lora_rank * d.qk_nope_head_dim],
        ));
        tensors.push(f32_tensor(
            format!("blk.{l}.attn_v_b.weight"),
            vec![
                d.num_heads as u64,
                d.v_head_dim as u64,
                d.kv_lora_rank as u64,
            ],
            vec![0.02; d.num_heads * d.v_head_dim * d.kv_lora_rank],
        ));
        tensors.push(f32_tensor(
            format!("blk.{l}.attn_output.weight"),
            vec![h as u64, (d.num_heads * d.v_head_dim) as u64],
            vec![0.02; h * d.num_heads * d.v_head_dim],
        ));

        if is_full {
            tensors.push(f32_tensor(
                format!("blk.{l}.indexer.k_norm.weight"),
                vec![d.indexer_head_dim as u64],
                vec![1.0; d.indexer_head_dim],
            ));
            tensors.push(f32_tensor(
                format!("blk.{l}.indexer.k_norm.bias"),
                vec![d.indexer_head_dim as u64],
                vec![0.0; d.indexer_head_dim],
            ));
            tensors.push(f32_tensor(
                format!("blk.{l}.indexer.proj.weight"),
                vec![d.indexer_n_heads as u64, h as u64],
                vec![0.02; d.indexer_n_heads * h],
            ));
            tensors.push(f32_tensor(
                format!("blk.{l}.indexer.attn_k.weight"),
                vec![d.indexer_head_dim as u64, h as u64],
                vec![0.02; d.indexer_head_dim * h],
            ));
            tensors.push(f32_tensor(
                format!("blk.{l}.indexer.attn_q_b.weight"),
                vec![
                    (d.indexer_n_heads * d.indexer_head_dim) as u64,
                    d.q_lora_rank as u64,
                ],
                vec![0.02; d.indexer_n_heads * d.indexer_head_dim * d.q_lora_rank],
            ));
        }

        if is_dense {
            for name in ["ffn_gate", "ffn_up"] {
                tensors.push(f32_tensor(
                    format!("blk.{l}.{name}.weight"),
                    vec![d.dense_ffn_dim as u64, h as u64],
                    vec![0.02; d.dense_ffn_dim * h],
                ));
            }
            tensors.push(f32_tensor(
                format!("blk.{l}.ffn_down.weight"),
                vec![h as u64, d.dense_ffn_dim as u64],
                vec![0.02; h * d.dense_ffn_dim],
            ));
        } else {
            let ff = d.moe_ffn_dim;
            let n = d.n_experts;
            tensors.push(f32_tensor(
                format!("blk.{l}.ffn_gate_inp.weight"),
                vec![n as u64, h as u64],
                vec![0.02; n * h],
            ));
            tensors.push(f32_tensor(
                format!("blk.{l}.exp_probs_b.bias"),
                vec![n as u64],
                vec![0.0; n],
            ));
            tensors.push(f32_tensor(
                format!("blk.{l}.ffn_gate_exps.weight"),
                vec![n as u64, ff as u64, h as u64],
                vec![0.02; h * ff * n],
            ));
            tensors.push(f32_tensor(
                format!("blk.{l}.ffn_down_exps.weight"),
                vec![n as u64, h as u64, ff as u64],
                vec![0.02; ff * h * n],
            ));
            tensors.push(f32_tensor(
                format!("blk.{l}.ffn_up_exps.weight"),
                vec![n as u64, ff as u64, h as u64],
                vec![0.02; h * ff * n],
            ));
            let shexp_dim = ff * d.n_shared_experts;
            tensors.push(f32_tensor(
                format!("blk.{l}.ffn_gate_shexp.weight"),
                vec![shexp_dim as u64, h as u64],
                vec![0.02; shexp_dim * h],
            ));
            tensors.push(f32_tensor(
                format!("blk.{l}.ffn_down_shexp.weight"),
                vec![h as u64, shexp_dim as u64],
                vec![0.02; h * shexp_dim],
            ));
            tensors.push(f32_tensor(
                format!("blk.{l}.ffn_up_shexp.weight"),
                vec![shexp_dim as u64, h as u64],
                vec![0.02; shexp_dim * h],
            ));
        }
    }

    fn dims() -> Dims {
        Dims {
            hidden_dim: 8,
            num_heads: 2,
            q_lora_rank: 6,
            kv_lora_rank: 4,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 3,
            indexer_n_heads: 2,
            indexer_head_dim: 4,
            dense_ffn_dim: 5,
            moe_ffn_dim: 4,
            n_experts: 3,
            n_shared_experts: 1,
        }
    }

    fn hp_from(d: &Dims) -> Glm52GgufHparams {
        Glm52GgufHparams {
            hidden_dim: d.hidden_dim,
            num_heads: d.num_heads,
            q_lora_rank: d.q_lora_rank,
            kv_lora_rank: d.kv_lora_rank,
            qk_nope_head_dim: d.qk_nope_head_dim,
            qk_rope_head_dim: d.qk_rope_head_dim,
            v_head_dim: d.v_head_dim,
            rope_theta: 8_000_000.0,
            indexer_n_heads: d.indexer_n_heads,
            indexer_head_dim: d.indexer_head_dim,
            indexer_rope_dim: d.qk_rope_head_dim,
            indexer_top_k: 2,
            dense_ffn_dim: d.dense_ffn_dim,
            moe_ffn_dim: d.moe_ffn_dim,
            n_experts: d.n_experts,
            n_shared_experts: d.n_shared_experts,
        }
    }

    #[test]
    fn loads_a_full_indexer_dense_layer_and_a_shared_indexer_moe_layer() {
        let d = dims();
        let mut tensors: Vec<FixtureTensor> = Vec::new();
        push_layer_tensors(&mut tensors, 0, true, true, &d);
        push_layer_tensors(&mut tensors, 1, false, false, &d);

        let bytes = build_gguf("glm-dsa", &tensors);
        let path = std::env::temp_dir().join(format!(
            "ferrox_glm52_gguf_test_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let file = ferrox_gguf::GgufFile::open(&path).expect("synthetic GGUF must parse");

        let hp = hp_from(&d);

        let layer0 = load_glm52_attn(&file, &hp, 0, true).expect("full-indexer layer must load");
        assert!(layer0.indexer.is_some());
        assert_eq!(layer0.wk_b.len(), d.num_heads);
        assert_eq!(layer0.wk_b[0].rows(), d.qk_nope_head_dim);
        assert_eq!(layer0.wk_b[0].cols(), d.kv_lora_rank);
        assert_eq!(layer0.wv_b[0].rows(), d.v_head_dim);
        assert_eq!(layer0.wv_b[0].cols(), d.kv_lora_rank);
        let dense0 = load_glm52_dense_ffn(&file, 0).expect("dense FFN must load");
        assert_eq!(dense0.gate_proj.rows(), d.dense_ffn_dim);

        let layer1 = load_glm52_attn(&file, &hp, 1, false).expect("shared-indexer layer must load");
        assert!(
            layer1.indexer.is_none(),
            "a \"shared\" layer must not load its own indexer weights"
        );
        let moe1 = load_glm52_moe_ffn(&file, &hp, 1).expect("MoE FFN must load");
        assert_eq!(moe1.experts.len(), d.n_experts);
        assert_eq!(moe1.e_score_correction_bias.len(), d.n_experts);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wk_b_transpose_matches_hand_computed_values() {
        // n_head=1, qk_nope_head_dim=2, kv_lora_rank=3. Real on-disk
        // layout (ne=[2,3,1], i.e. 3 rows of 2 floats): row0=[1,2],
        // row1=[3,4], row2=[5,6]. Transposed [2,3] result must be
        // row0=[1,3,5], row1=[2,4,6].
        let tensors = vec![f32_tensor(
            "blk.0.attn_k_b.weight",
            vec![1, 3, 2], // reversed real ne=[2,3,1]
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )];
        let bytes = build_gguf("glm-dsa", &tensors);
        let path = std::env::temp_dir().join(format!(
            "ferrox_glm52_wk_b_test_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let file = ferrox_gguf::GgufFile::open(&path).expect("synthetic GGUF must parse");

        let heads = load_wk_b_transposed(&file, "blk.0.attn_k_b.weight", 1, 2, 3)
            .expect("must load and transpose");
        std::fs::remove_file(&path).ok();

        assert_eq!(heads.len(), 1);
        let applied_e0 = heads[0].apply(&[1.0, 0.0, 0.0]);
        let applied_e1 = heads[0].apply(&[0.0, 1.0, 0.0]);
        let applied_e2 = heads[0].apply(&[0.0, 0.0, 1.0]);
        // Column c of the transposed [2,3] matrix (applying basis
        // vector e_c) must equal row c of the original [3,2] input.
        assert_eq!(applied_e0, vec![1.0, 2.0]);
        assert_eq!(applied_e1, vec![3.0, 4.0]);
        assert_eq!(applied_e2, vec![5.0, 6.0]);
    }
}
