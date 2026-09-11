//! DeepSeek-2 / Mistral-4 GGUF → [`crate::engine::MlaEngine`].
//!
//! Tensor names follow llama.cpp `deepseek2` / `mistral4` (same graph):
//! `blk.{i}.attn_q_a|attn_q_b|attn_kv_a_mqa|attn_kv_b|attn_output` plus
//! optional `attn_q_a_norm` / `attn_kv_a_norm`. Dense FFN:
//! `ffn_{gate,up,down}`. MoE after `leading_dense_block_count` uses
//! `ffn_gate_inp` + packed `ffn_{gate,up,down}_exps` + shared
//! `ffn_{gate,up,down}_shexp` (fail-closed if any are missing).
//!
//! `use_output_gate` is off (classic DeepSeek-2). RoPE uses interleaved
//! Norm layout via [`crate::config::MlaRopeConfig`].

use ferrox_gguf::TensorSource;
use ferrox_moe::GatingFunction;

use crate::config::{MlaConfig, MlaRopeConfig};
use crate::engine::{
    MlaDenseFfn, MlaEngine, MlaLayerFfn, MlaLayerWeights, MlaMoeFfn, MlaMoeRuntime,
};
use crate::loader::LoadError;
use crate::loader::{load_f32_vec, load_weight_matrix, split_expert_tensor};
use crate::mla::MlaAttnWeights;

/// Hyperparameters read from `{arch}.*` GGUF metadata.
#[derive(Debug, Clone)]
pub struct Deepseek2Hparams {
    pub arch: String,
    pub n_layer: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Layers `[0, leading_dense)` use dense SwiGLU; rest require MoE.
    pub leading_dense_block_count: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_shared_experts: usize,
    pub expert_ffn_dim: usize,
    pub gating: GatingFunction,
    pub norm_topk_prob: bool,
    pub expert_weights_scale: f32,
}

fn meta_u64(file: &impl TensorSource, key: &str) -> Result<u64, LoadError> {
    file.metadata_u64(key)
        .ok_or_else(|| LoadError::MissingHparam(key.to_string()))
}

fn meta_f32(file: &impl TensorSource, key: &str, default: f32) -> f32 {
    file.metadata_f32(key).unwrap_or(default)
}

/// Read DeepSeek-2 / Mistral-4 hparams from an opened GGUF.
pub fn read_deepseek2_hparams(file: &impl TensorSource) -> Result<Deepseek2Hparams, LoadError> {
    let arch = file
        .metadata_str("general.architecture")
        .ok_or_else(|| LoadError::MissingHparam("general.architecture".into()))?
        .to_string();
    if arch != "deepseek2" && arch != "mistral4" {
        return Err(LoadError::UnsupportedArchitecture(arch));
    }
    let p = |suffix: &str| format!("{arch}.{suffix}");
    // The trunk: `block_count` minus the NextN/MTP blocks llama.cpp
    // never runs, decided once for every loader in `crate::mtp_blocks`.
    let n_layer =
        crate::mtp_blocks::trunk_layers(file, &arch, meta_u64(file, &p("block_count"))? as usize)?
            .n_layers;
    let hidden_dim = meta_u64(file, &p("embedding_length"))? as usize;
    let ffn_dim = meta_u64(file, &p("feed_forward_length"))? as usize;
    let n_heads = meta_u64(file, &p("attention.head_count"))? as usize;
    let q_lora_rank = meta_u64(file, &p("attention.q_lora_rank"))? as usize;
    let kv_lora_rank = meta_u64(file, &p("attention.kv_lora_rank"))? as usize;
    // `attention.qk_nope_head_dim` and `attention.qk_rope_head_dim` ARE
    // NOT GGUF KEYS. Neither string appears in llama.cpp's
    // `LLM_KV_NAMES` or anywhere in `gguf-py`; they are HF
    // `config.json` field names. Requiring them meant DeepSeek-V2,
    // V2.5, V3 and R1 -- the largest open models people run -- all
    // failed with "missing hparam deepseek2.attention.qk_nope_head_dim",
    // a true statement about a key no converter has ever written. The
    // same shape as `glm4moe` being sent to an MLA loader for a
    // `q_lora_rank` it does not have.
    //
    // What a real file carries, and how llama.cpp derives the per-head
    // dims from it (`src/models/deepseek2.cpp:77-82`):
    //
    //   qk_rope = rope.dimension_count                  (`n_rot`)
    //   qk_nope = attention.key_length_mla - qk_rope
    //   v_head  = attention.value_length_mla
    //
    // `attention.key_length` / `value_length` are NOT these: for an MLA
    // checkpoint they hold the COMPRESSED MQA widths
    // (`kv_lora_rank + qk_rope` and `kv_lora_rank`), so reading them as
    // per-head dims silently builds a differently shaped model.
    //
    // The HF spellings are still accepted, because ferrox's own
    // synthetic fixtures were written against them, but they are the
    // fallback rather than the contract.
    let qk_rope_head_dim = meta_u64(file, &p("rope.dimension_count"))
        .or_else(|_| meta_u64(file, &p("attention.qk_rope_head_dim")))?
        as usize;
    let qk_nope_head_dim = match meta_u64(file, &p("attention.key_length_mla")) {
        Ok(k_mla) => (k_mla as usize)
            .checked_sub(qk_rope_head_dim)
            .filter(|&nope| nope >= 1)
            .ok_or_else(|| {
                LoadError::MissingHparam(format!(
                    "{arch}.attention.key_length_mla ({k_mla}) must exceed \
                     rope.dimension_count ({qk_rope_head_dim})"
                ))
            })?,
        Err(_) => meta_u64(file, &p("attention.qk_nope_head_dim"))? as usize,
    };
    let v_head_dim = meta_u64(file, &p("attention.value_length_mla"))
        .or_else(|_| meta_u64(file, &p("attention.v_head_dim")))
        .unwrap_or(qk_nope_head_dim as u64) as usize;
    let leading_dense = file
        .metadata_u64(&p("leading_dense_block_count"))
        .unwrap_or(n_layer as u64) as usize;
    let n_expert = file.metadata_u64(&p("expert_count")).unwrap_or(0) as usize;
    let n_expert_used = file
        .metadata_u64(&p("expert_used_count"))
        .unwrap_or(if n_expert > 0 { 6 } else { 0 }) as usize;
    let n_shared_experts = file.metadata_u64(&p("expert_shared_count")).unwrap_or(1) as usize;
    let expert_ffn_dim = file
        .metadata_u64(&p("expert_feed_forward_length"))
        .unwrap_or(ffn_dim as u64) as usize;
    let rms_norm_eps = meta_f32(file, &p("attention.layer_norm_rms_epsilon"), 1e-6);
    let rope_theta = meta_f32(file, &p("rope.freq_base"), 10000.0);
    // llama.cpp deepseek2: default Softmax unless expert_gating_func set
    // (1=softmax, 2=sigmoid); special-case GLM 4.7 Lite sigmoid when absent.
    let gating = match file.metadata_u64(&p("expert_gating_func")) {
        Some(2) => GatingFunction::Sigmoid,
        Some(1) => GatingFunction::Softmax,
        _ if (n_layer == 47 || n_layer == 48)
            && file
                .find_tensor("token_embd.weight")
                .map(|t| t.shape.last().copied().unwrap_or(0) == 154880)
                .unwrap_or(false) =>
        {
            GatingFunction::Sigmoid
        }
        _ => GatingFunction::Softmax,
    };
    let norm_topk_prob = file
        .metadata_u64(&p("expert_weights_norm"))
        .map(|v| v != 0)
        .unwrap_or(true);
    let expert_weights_scale = meta_f32(file, &p("expert_weights_scale"), 1.0);
    Ok(Deepseek2Hparams {
        arch,
        n_layer,
        hidden_dim,
        ffn_dim,
        n_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        rms_norm_eps,
        rope_theta,
        leading_dense_block_count: leading_dense.min(n_layer),
        n_expert,
        n_expert_used: n_expert_used.min(n_expert.max(1)),
        n_shared_experts: n_shared_experts.max(1),
        expert_ffn_dim,
        gating,
        norm_topk_prob,
        expert_weights_scale,
    })
}

fn load_f32_vec_optional(
    file: &impl TensorSource,
    name: &str,
) -> Result<Option<Vec<f32>>, LoadError> {
    if file.find_tensor(name).is_none() {
        return Ok(None);
    }
    Ok(Some(load_f32_vec(file, name)?))
}

fn load_mla_attn(
    file: &impl TensorSource,
    layer_idx: usize,
    hp: &Deepseek2Hparams,
) -> Result<MlaAttnWeights, LoadError> {
    let l = layer_idx;
    let q_head_dim = hp.qk_nope_head_dim + hp.qk_rope_head_dim;
    let q_a_proj = load_weight_matrix(file, &format!("blk.{l}.attn_q_a.weight"))?;
    let q_b_proj = load_weight_matrix(file, &format!("blk.{l}.attn_q_b.weight"))?;
    let kv_a = load_weight_matrix(file, &format!("blk.{l}.attn_kv_a_mqa.weight"))?;
    let o_proj = load_weight_matrix(file, &format!("blk.{l}.attn_output.weight"))?;

    // Prefer combined `attn_kv_b`; else refuse split k_b/v_b until concat lands.
    let kv_b_proj = if file
        .find_tensor(&format!("blk.{l}.attn_kv_b.weight"))
        .is_some()
    {
        load_weight_matrix(file, &format!("blk.{l}.attn_kv_b.weight"))?
    } else {
        return Err(LoadError::Gguf(ferrox_gguf::GgufError::TensorNotFound(
            format!(
                "blk.{l}.attn_kv_b.weight (split attn_k_b/attn_v_b not wired for MlaEngine yet)"
            ),
        )));
    };

    let q_a_ln = load_f32_vec_optional(file, &format!("blk.{l}.attn_q_a_norm.weight"))?
        .unwrap_or_else(|| vec![1.0; hp.q_lora_rank]);
    let kv_a_ln = load_f32_vec_optional(file, &format!("blk.{l}.attn_kv_a_norm.weight"))?
        .unwrap_or_else(|| vec![1.0; hp.kv_lora_rank]);

    let _ = (q_head_dim,);
    Ok(MlaAttnWeights {
        q_a_proj,
        q_a_layernorm: q_a_ln,
        q_b_proj,
        kv_a_proj_with_mqa: kv_a,
        kv_a_layernorm: kv_a_ln,
        kv_b_proj,
        o_proj,
        g_proj: None,
    })
}

fn require_tensor(file: &impl TensorSource, name: &str) -> Result<(), LoadError> {
    if file.find_tensor(name).is_none() {
        return Err(LoadError::Gguf(ferrox_gguf::GgufError::TensorNotFound(
            name.to_string(),
        )));
    }
    Ok(())
}

fn load_dense_ffn(file: &impl TensorSource, layer_idx: usize) -> Result<MlaDenseFfn, LoadError> {
    let l = layer_idx;
    Ok(MlaDenseFfn {
        gate: load_weight_matrix(file, &format!("blk.{l}.ffn_gate.weight"))?,
        up: load_weight_matrix(file, &format!("blk.{l}.ffn_up.weight"))?,
        down: load_weight_matrix(file, &format!("blk.{l}.ffn_down.weight"))?,
    })
}

fn load_moe_ffn(
    file: &impl TensorSource,
    layer_idx: usize,
    hp: &Deepseek2Hparams,
) -> Result<MlaMoeFfn, LoadError> {
    let l = layer_idx;
    // Fail-closed: every MoE tensor must be present (no silent dense fallback).
    for name in [
        format!("blk.{l}.ffn_gate_inp.weight"),
        format!("blk.{l}.ffn_gate_exps.weight"),
        format!("blk.{l}.ffn_up_exps.weight"),
        format!("blk.{l}.ffn_down_exps.weight"),
        format!("blk.{l}.ffn_gate_shexp.weight"),
        format!("blk.{l}.ffn_up_shexp.weight"),
        format!("blk.{l}.ffn_down_shexp.weight"),
    ] {
        require_tensor(file, &name)?;
    }
    let gate_exps =
        split_expert_tensor(file, &format!("blk.{l}.ffn_gate_exps.weight"), hp.n_expert)?;
    let up_exps = split_expert_tensor(file, &format!("blk.{l}.ffn_up_exps.weight"), hp.n_expert)?;
    let down_exps =
        split_expert_tensor(file, &format!("blk.{l}.ffn_down_exps.weight"), hp.n_expert)?;
    let experts = gate_exps
        .into_iter()
        .zip(up_exps)
        .zip(down_exps)
        .map(|((gate, up), down)| ferrox_moe::ExpertWeights { gate, up, down })
        .collect();
    let shared_expert = ferrox_moe::ExpertWeights {
        gate: load_weight_matrix(file, &format!("blk.{l}.ffn_gate_shexp.weight"))?,
        up: load_weight_matrix(file, &format!("blk.{l}.ffn_up_shexp.weight"))?,
        down: load_weight_matrix(file, &format!("blk.{l}.ffn_down_shexp.weight"))?,
    };
    // See the note in `glm52_gguf_loader`: the on-disk name has no
    // `ffn_` prefix. Optional here on purpose -- llama.cpp declares it
    // TENSOR_NOT_REQUIRED for `deepseek2`, which also covers V2-era
    // checkpoints with no routing bias at all -- which is exactly why the
    // wrong name was silent rather than a load error, and a real
    // DeepSeek-V3 checkpoint routed with its bias dropped.
    let exp_probs_bias = load_f32_vec_optional(file, &format!("blk.{l}.exp_probs_b.bias"))?;
    Ok(MlaMoeFfn {
        router: load_weight_matrix(file, &format!("blk.{l}.ffn_gate_inp.weight"))?,
        experts,
        shared_expert,
        exp_probs_bias,
    })
}

fn load_layer(
    file: &impl TensorSource,
    layer_idx: usize,
    hp: &Deepseek2Hparams,
) -> Result<MlaLayerWeights, LoadError> {
    let l = layer_idx;
    let ffn = if layer_idx < hp.leading_dense_block_count || hp.n_expert == 0 {
        MlaLayerFfn::Dense(load_dense_ffn(file, layer_idx)?)
    } else {
        MlaLayerFfn::Moe(load_moe_ffn(file, layer_idx, hp)?)
    };
    Ok(MlaLayerWeights {
        attn_norm: load_f32_vec(file, &format!("blk.{l}.attn_norm.weight"))?,
        attn: load_mla_attn(file, layer_idx, hp)?,
        ffn_norm: load_f32_vec(file, &format!("blk.{l}.ffn_norm.weight"))?,
        ffn,
    })
}

/// Load a DeepSeek-2 / Mistral-4 GGUF into [`MlaEngine`] (dense lead + MoE tail).
pub fn load_mla_engine(file: &impl TensorSource) -> Result<MlaEngine, LoadError> {
    let hp = read_deepseek2_hparams(file)?;
    if hp.n_expert > 0 && hp.leading_dense_block_count >= hp.n_layer {
        // Experts declared but every layer is still dense — ignore MoE.
    } else if hp.n_expert > 0 && hp.n_expert_used == 0 {
        return Err(LoadError::UnsupportedArchitecture(format!(
            "{}: expert_count={} but expert_used_count is 0",
            hp.arch, hp.n_expert
        )));
    }
    if hp.n_layer == 0 {
        return Err(LoadError::UnsupportedArchitecture(format!(
            "{}: no layers to load",
            hp.arch
        )));
    }

    let embedding = if file.find_tensor("token_embd.weight").is_some() {
        load_weight_matrix(file, "token_embd.weight")?
    } else {
        return Err(LoadError::Gguf(ferrox_gguf::GgufError::TensorNotFound(
            "token_embd.weight".into(),
        )));
    };
    let final_norm = load_f32_vec(file, "output_norm.weight")?;
    let output_head = match load_weight_matrix(file, "output.weight") {
        Ok(w) => w,
        Err(_) => load_weight_matrix(file, "token_embd.weight")?,
    };

    let mut layers = Vec::with_capacity(hp.n_layer);
    for i in 0..hp.n_layer {
        layers.push(load_layer(file, i, &hp)?);
    }
    let has_moe = layers.iter().any(|l| matches!(l.ffn, MlaLayerFfn::Moe(_)));
    let moe = if has_moe {
        Some(MlaMoeRuntime {
            n_experts_active: hp.n_expert_used,
            gating: hp.gating,
            norm_topk_prob: hp.norm_topk_prob,
            expert_weights_scale: hp.expert_weights_scale,
        })
    } else {
        None
    };

    Ok(MlaEngine {
        embedding,
        layers,
        final_norm,
        output_head,
        mla_cfg: MlaConfig {
            num_heads: hp.n_heads,
            q_lora_rank: hp.q_lora_rank,
            kv_lora_rank: hp.kv_lora_rank,
            qk_nope_head_dim: hp.qk_nope_head_dim,
            qk_rope_head_dim: hp.qk_rope_head_dim,
            v_head_dim: hp.v_head_dim,
            use_output_gate: false,
            rope: Some(MlaRopeConfig {
                theta: hp.rope_theta,
            }),
        },
        rms_norm_eps: hp.rms_norm_eps,
        hidden_dim: hp.hidden_dim,
        moe,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use byteorder::{LittleEndian, WriteBytesExt};
    use ferrox_gguf::GgufFile;
    use std::io::Write;

    struct FixtureTensor {
        name: String,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.write_f32::<LittleEndian>(*x).unwrap();
        }
        b
    }

    fn f32_tensor(name: &str, shape: Vec<u64>, values: Vec<f32>) -> FixtureTensor {
        FixtureTensor {
            name: name.into(),
            shape,
            bytes: f32_bytes(&values),
        }
    }

    fn build_gguf(
        arch: &str,
        kv: &[(&str, u64)],
        fkv: &[(&str, f32)],
        tensors: &[FixtureTensor],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
        // general.architecture + uint + float kvs
        let kv_count = 1 + kv.len() + fkv.len();
        buf.write_u64::<LittleEndian>(kv_count as u64).unwrap();

        let write_string = |buf: &mut Vec<u8>, s: &str| {
            buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
            buf.write_all(s.as_bytes()).unwrap();
        };
        write_string(&mut buf, "general.architecture");
        buf.write_u32::<LittleEndian>(8).unwrap();
        write_string(&mut buf, arch);
        for &(k, v) in kv {
            write_string(&mut buf, k);
            buf.write_u32::<LittleEndian>(10).unwrap(); // UINT64
            buf.write_u64::<LittleEndian>(v).unwrap();
        }
        for &(k, v) in fkv {
            write_string(&mut buf, k);
            buf.write_u32::<LittleEndian>(6).unwrap(); // FLOAT32
            buf.write_f32::<LittleEndian>(v).unwrap();
        }

        let mut offset = 0u64;
        let mut offsets = Vec::with_capacity(tensors.len());
        for t in tensors {
            write_string(&mut buf, &t.name);
            buf.write_u32::<LittleEndian>(t.shape.len() as u32).unwrap();
            for &d in t.shape.iter().rev() {
                buf.write_u64::<LittleEndian>(d).unwrap();
            }
            buf.write_u32::<LittleEndian>(0).unwrap();
            offsets.push(offset);
            buf.write_u64::<LittleEndian>(offset).unwrap();
            offset += (t.bytes.len().div_ceil(32) * 32) as u64;
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let data_start = buf.len();
        for (t, &off) in tensors.iter().zip(offsets.iter()) {
            while buf.len() < data_start + off as usize {
                buf.push(0);
            }
            buf.extend_from_slice(&t.bytes);
            while buf.len() % 32 != 0 {
                buf.push(0);
            }
        }
        buf
    }

    /// The two-layer dense fixture, with `n_mtp_blocks` NextN/MTP blocks
    /// declared after it: `block_count` counts them and
    /// `nextn_predict_layers` names them, and NO tensor is written for
    /// them -- llama.cpp's "trunk-only" split (`deepseek2.cpp:64-66`),
    /// which is also exactly the file that would fail to load if the
    /// blocks were treated as layers.
    fn synthetic_dense_deepseek2(n_mtp_blocks: u64) -> Vec<u8> {
        let h = 16usize;
        let n_heads = 2usize;
        let q_lora = 8usize;
        let kv_lora = 4usize;
        let qk_nope = 4usize;
        let qk_rope = 2usize;
        let v_dim = 4usize;
        let ffn = 32usize;
        let vocab = 8usize;
        let q_head = qk_nope + qk_rope;
        let arch = "deepseek2";

        let mut tensors = vec![
            f32_tensor(
                "token_embd.weight",
                vec![vocab as u64, h as u64],
                vec![0.01; h * vocab],
            ),
            f32_tensor("output_norm.weight", vec![h as u64], vec![1.0; h]),
            f32_tensor(
                "output.weight",
                vec![vocab as u64, h as u64],
                vec![0.02; h * vocab],
            ),
        ];
        for l in 0..2usize {
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.ffn_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_q_a.weight"),
                vec![q_lora as u64, h as u64],
                vec![0.01; h * q_lora],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_q_b.weight"),
                vec![(n_heads * q_head) as u64, q_lora as u64],
                vec![0.01; q_lora * n_heads * q_head],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_kv_a_mqa.weight"),
                vec![(kv_lora + qk_rope) as u64, h as u64],
                vec![0.01; h * (kv_lora + qk_rope)],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_kv_b.weight"),
                vec![(n_heads * (qk_nope + v_dim)) as u64, kv_lora as u64],
                vec![0.01; kv_lora * n_heads * (qk_nope + v_dim)],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_output.weight"),
                vec![h as u64, (n_heads * v_dim) as u64],
                vec![0.01; n_heads * v_dim * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.ffn_gate.weight"),
                vec![ffn as u64, h as u64],
                vec![0.01; h * ffn],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.ffn_up.weight"),
                vec![ffn as u64, h as u64],
                vec![0.01; h * ffn],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.ffn_down.weight"),
                vec![h as u64, ffn as u64],
                vec![0.01; ffn * h],
            ));
        }

        let kv = [
            ("deepseek2.block_count", 2u64 + n_mtp_blocks),
            ("deepseek2.nextn_predict_layers", n_mtp_blocks),
            ("deepseek2.embedding_length", h as u64),
            ("deepseek2.feed_forward_length", ffn as u64),
            ("deepseek2.attention.head_count", n_heads as u64),
            ("deepseek2.attention.q_lora_rank", q_lora as u64),
            ("deepseek2.attention.kv_lora_rank", kv_lora as u64),
            ("deepseek2.attention.qk_nope_head_dim", qk_nope as u64),
            ("deepseek2.attention.qk_rope_head_dim", qk_rope as u64),
            ("deepseek2.attention.v_head_dim", v_dim as u64),
            ("deepseek2.leading_dense_block_count", 2u64),
            ("deepseek2.expert_count", 0u64),
        ];
        let fkv = [
            ("deepseek2.attention.layer_norm_rms_epsilon", 1e-5f32),
            ("deepseek2.rope.freq_base", 10000.0f32),
        ];
        build_gguf(arch, &kv, &fkv, &tensors)
    }

    fn load_and_forward(bytes: &[u8], tag: &str) -> MlaEngine {
        let path =
            std::env::temp_dir().join(format!("ferrox_mla_gguf_{tag}_{}.gguf", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let engine = load_mla_engine(&file).expect("load mla");
        let mut state = engine.new_state();
        let logits = engine.forward_token(0, 0, &mut state);
        assert_eq!(logits.len(), engine.vocab_size());
        assert!(logits.iter().all(|x| x.is_finite()));
        let _ = std::fs::remove_file(&path);
        engine
    }

    #[test]
    fn load_synthetic_deepseek2_dense_and_forward() {
        let engine = load_and_forward(&synthetic_dense_deepseek2(0), "dense");
        assert_eq!(engine.layers.len(), 2);
        assert_eq!(engine.vocab_size(), 8);
    }

    /// A real DeepSeek-V3 / GLM-4.7-Flash export counts its MTP block
    /// in `block_count` (`conversion/deepseek.py:457,498`). This loader
    /// took `block_count` verbatim, so on such a file it would have
    /// either run the block as a 62nd layer or, on a trunk-only split,
    /// failed on its missing tensors. The trunk is what loads now, and
    /// the block's absent tensors are not asked for.
    #[test]
    fn a_deepseek2_file_with_an_mtp_block_loads_only_its_trunk() {
        let engine = load_and_forward(&synthetic_dense_deepseek2(1), "mtp");
        assert_eq!(engine.layers.len(), 2, "block_count 3 minus one MTP block");
    }

    #[allow(clippy::too_many_arguments)] // test fixture: mirrors the MLA tensor shape set
    fn push_mla_attn_tensors(
        tensors: &mut Vec<FixtureTensor>,
        l: usize,
        h: usize,
        n_heads: usize,
        q_lora: usize,
        kv_lora: usize,
        qk_nope: usize,
        qk_rope: usize,
        v_dim: usize,
    ) {
        let q_head = qk_nope + qk_rope;
        tensors.push(f32_tensor(
            &format!("blk.{l}.attn_norm.weight"),
            vec![h as u64],
            vec![1.0; h],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.ffn_norm.weight"),
            vec![h as u64],
            vec![1.0; h],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.attn_q_a.weight"),
            vec![q_lora as u64, h as u64],
            vec![0.01; h * q_lora],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.attn_q_b.weight"),
            vec![(n_heads * q_head) as u64, q_lora as u64],
            vec![0.01; q_lora * n_heads * q_head],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.attn_kv_a_mqa.weight"),
            vec![(kv_lora + qk_rope) as u64, h as u64],
            vec![0.01; h * (kv_lora + qk_rope)],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.attn_kv_b.weight"),
            vec![(n_heads * (qk_nope + v_dim)) as u64, kv_lora as u64],
            vec![0.01; kv_lora * n_heads * (qk_nope + v_dim)],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.attn_output.weight"),
            vec![h as u64, (n_heads * v_dim) as u64],
            vec![0.01; n_heads * v_dim * h],
        ));
    }

    #[test]
    fn load_synthetic_deepseek2_moe_after_dense_and_forward() {
        let h = 16usize;
        let n_heads = 2usize;
        let q_lora = 8usize;
        let kv_lora = 4usize;
        let qk_nope = 4usize;
        let qk_rope = 2usize;
        let v_dim = 4usize;
        let ffn = 32usize;
        let exp_ff = 16usize;
        let n_exp = 4usize;
        let vocab = 8usize;
        let arch = "deepseek2";

        let mut tensors = vec![
            f32_tensor(
                "token_embd.weight",
                vec![vocab as u64, h as u64],
                vec![0.01; h * vocab],
            ),
            f32_tensor("output_norm.weight", vec![h as u64], vec![1.0; h]),
            f32_tensor(
                "output.weight",
                vec![vocab as u64, h as u64],
                vec![0.02; h * vocab],
            ),
        ];
        // Layer 0: dense
        push_mla_attn_tensors(
            &mut tensors,
            0,
            h,
            n_heads,
            q_lora,
            kv_lora,
            qk_nope,
            qk_rope,
            v_dim,
        );
        tensors.push(f32_tensor(
            "blk.0.ffn_gate.weight",
            vec![ffn as u64, h as u64],
            vec![0.01; h * ffn],
        ));
        tensors.push(f32_tensor(
            "blk.0.ffn_up.weight",
            vec![ffn as u64, h as u64],
            vec![0.01; h * ffn],
        ));
        tensors.push(f32_tensor(
            "blk.0.ffn_down.weight",
            vec![h as u64, ffn as u64],
            vec![0.01; ffn * h],
        ));
        // Layer 1: MoE
        push_mla_attn_tensors(
            &mut tensors,
            1,
            h,
            n_heads,
            q_lora,
            kv_lora,
            qk_nope,
            qk_rope,
            v_dim,
        );
        tensors.push(f32_tensor(
            "blk.1.ffn_gate_inp.weight",
            vec![n_exp as u64, h as u64],
            vec![0.01; h * n_exp],
        ));
        // Packed expert tensors: logical [n_experts, out, in] → GGUF shape write uses rev
        // so pass shape as [n_experts, out, in] matching other fixtures' logical order.
        tensors.push(f32_tensor(
            "blk.1.ffn_gate_exps.weight",
            vec![n_exp as u64, exp_ff as u64, h as u64],
            vec![0.01; n_exp * exp_ff * h],
        ));
        tensors.push(f32_tensor(
            "blk.1.ffn_up_exps.weight",
            vec![n_exp as u64, exp_ff as u64, h as u64],
            vec![0.01; n_exp * exp_ff * h],
        ));
        tensors.push(f32_tensor(
            "blk.1.ffn_down_exps.weight",
            vec![n_exp as u64, h as u64, exp_ff as u64],
            vec![0.01; n_exp * h * exp_ff],
        ));
        tensors.push(f32_tensor(
            "blk.1.ffn_gate_shexp.weight",
            vec![exp_ff as u64, h as u64],
            vec![0.01; h * exp_ff],
        ));
        tensors.push(f32_tensor(
            "blk.1.ffn_up_shexp.weight",
            vec![exp_ff as u64, h as u64],
            vec![0.01; h * exp_ff],
        ));
        tensors.push(f32_tensor(
            "blk.1.ffn_down_shexp.weight",
            vec![h as u64, exp_ff as u64],
            vec![0.01; exp_ff * h],
        ));

        let kv = [
            ("deepseek2.block_count", 2u64),
            ("deepseek2.embedding_length", h as u64),
            ("deepseek2.feed_forward_length", ffn as u64),
            ("deepseek2.attention.head_count", n_heads as u64),
            ("deepseek2.attention.q_lora_rank", q_lora as u64),
            ("deepseek2.attention.kv_lora_rank", kv_lora as u64),
            ("deepseek2.attention.qk_nope_head_dim", qk_nope as u64),
            ("deepseek2.attention.qk_rope_head_dim", qk_rope as u64),
            ("deepseek2.attention.v_head_dim", v_dim as u64),
            ("deepseek2.leading_dense_block_count", 1u64),
            ("deepseek2.expert_count", n_exp as u64),
            ("deepseek2.expert_used_count", 2u64),
            ("deepseek2.expert_shared_count", 1u64),
            ("deepseek2.expert_feed_forward_length", exp_ff as u64),
            ("deepseek2.expert_gating_func", 1u64), // softmax
        ];
        let fkv = [
            ("deepseek2.attention.layer_norm_rms_epsilon", 1e-5f32),
            ("deepseek2.rope.freq_base", 10000.0f32),
            ("deepseek2.expert_weights_scale", 1.0f32),
        ];
        let bytes = build_gguf(arch, &kv, &fkv, &tensors);
        let path =
            std::env::temp_dir().join(format!("ferrox_mla_moe_gguf_{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let engine = load_mla_engine(&file).expect("load mla moe");
        assert_eq!(engine.layers.len(), 2);
        assert!(matches!(
            engine.layers[0].ffn,
            crate::engine::MlaLayerFfn::Dense(_)
        ));
        assert!(matches!(
            engine.layers[1].ffn,
            crate::engine::MlaLayerFfn::Moe(_)
        ));
        assert!(engine.moe.is_some());
        let mut state = engine.new_state();
        let logits = engine.forward_token(0, 0, &mut state);
        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|x| x.is_finite()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn moe_after_dense_fails_closed_without_expert_tensors() {
        let h = 16usize;
        let n_heads = 2usize;
        let q_lora = 8usize;
        let kv_lora = 4usize;
        let qk_nope = 4usize;
        let qk_rope = 2usize;
        let v_dim = 4usize;
        let ffn = 32usize;
        let vocab = 8usize;
        let arch = "deepseek2";

        let mut tensors = vec![
            f32_tensor(
                "token_embd.weight",
                vec![vocab as u64, h as u64],
                vec![0.01; h * vocab],
            ),
            f32_tensor("output_norm.weight", vec![h as u64], vec![1.0; h]),
            f32_tensor(
                "output.weight",
                vec![vocab as u64, h as u64],
                vec![0.02; h * vocab],
            ),
        ];
        for l in 0..2usize {
            push_mla_attn_tensors(
                &mut tensors,
                l,
                h,
                n_heads,
                q_lora,
                kv_lora,
                qk_nope,
                qk_rope,
                v_dim,
            );
            // Only dense FFN tensors — MoE layer 1 will fail closed.
            tensors.push(f32_tensor(
                &format!("blk.{l}.ffn_gate.weight"),
                vec![ffn as u64, h as u64],
                vec![0.01; h * ffn],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.ffn_up.weight"),
                vec![ffn as u64, h as u64],
                vec![0.01; h * ffn],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.ffn_down.weight"),
                vec![h as u64, ffn as u64],
                vec![0.01; ffn * h],
            ));
        }
        let kv = [
            ("deepseek2.block_count", 2u64),
            ("deepseek2.embedding_length", h as u64),
            ("deepseek2.feed_forward_length", ffn as u64),
            ("deepseek2.attention.head_count", n_heads as u64),
            ("deepseek2.attention.q_lora_rank", q_lora as u64),
            ("deepseek2.attention.kv_lora_rank", kv_lora as u64),
            ("deepseek2.attention.qk_nope_head_dim", qk_nope as u64),
            ("deepseek2.attention.qk_rope_head_dim", qk_rope as u64),
            ("deepseek2.attention.v_head_dim", v_dim as u64),
            ("deepseek2.leading_dense_block_count", 1u64),
            ("deepseek2.expert_count", 4u64),
            ("deepseek2.expert_used_count", 2u64),
        ];
        let fkv = [
            ("deepseek2.attention.layer_norm_rms_epsilon", 1e-5f32),
            ("deepseek2.rope.freq_base", 10000.0f32),
        ];
        let bytes = build_gguf(arch, &kv, &fkv, &tensors);
        let path = std::env::temp_dir().join(format!(
            "ferrox_mla_moe_missing_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let err = match load_mla_engine(&file) {
            Err(e) => e,
            Ok(_) => panic!("expected missing MoE tensors to fail closed"),
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("ffn_gate_inp") || msg.contains("TensorNotFound"),
            "unexpected error: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }
}
