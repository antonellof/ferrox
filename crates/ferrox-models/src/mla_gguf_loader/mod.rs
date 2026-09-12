//! DeepSeek-2 / Mistral-4 / PLM GGUF → [`crate::engine::MlaEngine`].
//!
//! Tensor names follow llama.cpp `deepseek2` / `mistral4` (same graph)
//! and `plm` (the same attention on a dense model): `blk.{i}.attn_q_a|
//! attn_q_b` OR `attn_q` (`crate::mla_q_proj`), `attn_kv_a_mqa|
//! attn_kv_b|attn_output` plus optional `attn_q_a_norm` /
//! `attn_kv_a_norm`. Dense FFN: `ffn_{gate,up,down}` (SwiGLU) or
//! `ffn_{up,down}` alone (PLM's ReLU-squared). MoE after
//! `leading_dense_block_count` uses `ffn_gate_inp` + packed
//! `ffn_{gate,up,down}_exps` + shared `ffn_{gate,up,down}_shexp`
//! (fail-closed if any are missing). The three places the
//! architectures differ are one table, `crate::mla_arch`.
//!
//! `use_output_gate` is off (classic DeepSeek-2). RoPE uses interleaved
//! Norm layout via [`crate::config::MlaRopeConfig`]; a file declaring a
//! `rope.scaling.type` is REFUSED, because `deepseek2.cpp:312-319`
//! folds YaRN's magnitude into `kq_scale` and this engine has neither
//! that nor the frequency rewrite.

use ferrox_gguf::TensorSource;
use ferrox_moe::GatingFunction;

use crate::config::{MlaConfig, MlaRopeConfig};
use crate::engine::{
    MlaDenseFfn, MlaEngine, MlaLayerFfn, MlaLayerWeights, MlaMoeFfn, MlaMoeRuntime,
};
use crate::loader::LoadError;
use crate::loader::{load_f32_vec, load_weight_matrix, split_expert_tensor};
use crate::mla::{MlaAttnWeights, MlaKvB, MlaQProj};
use crate::mla_arch::{mla_arch, MlaArch, MlaOutputHead, QProjRule};

/// Hyperparameters read from `{arch}.*` GGUF metadata.
#[derive(Debug, Clone)]
pub struct Deepseek2Hparams {
    pub arch: String,
    /// The architecture's row in `crate::mla_arch`.
    pub row: MlaArch,
    pub n_layer: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    /// `Some` for a low-rank Q (`attn_q_a` / `attn_q_b`), `None` for a
    /// direct `attn_q`: `QProjRule` applied to this file.
    pub q_lora_rank: Option<usize>,
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
    let Some(row) = mla_arch(&arch).copied() else {
        return Err(LoadError::UnsupportedArchitecture(arch));
    };
    let p = |suffix: &str| format!("{arch}.{suffix}");
    // The trunk: `block_count` minus the NextN/MTP blocks llama.cpp
    // never runs, decided once for every loader in `crate::mtp_blocks`.
    let n_layer =
        crate::mtp_blocks::trunk_layers(file, &arch, meta_u64(file, &p("block_count"))? as usize)?
            .n_layers;
    let hidden_dim = meta_u64(file, &p("embedding_length"))? as usize;
    let ffn_dim = meta_u64(file, &p("feed_forward_length"))? as usize;
    let n_heads = meta_u64(file, &p("attention.head_count"))? as usize;
    // `deepseek2.cpp:5`: `vocab_size`, else the token list's length.
    let n_vocab = file
        .metadata_u64(&p("vocab_size"))
        .map(|v| v as usize)
        .or_else(|| match file.metadata("tokenizer.ggml.tokens") {
            Some(ferrox_gguf::GgufValue::Array(items)) => Some(items.len()),
            _ => None,
        })
        .unwrap_or(0);
    // Low-rank or direct Q: `crate::mla_arch` for the rule and its
    // order (the lite layer counts are decided BEFORE the key is read).
    let q_lora_rank = match row.q_proj {
        QProjRule::Direct => None,
        QProjRule::LoraUnlessLite if crate::mla_arch::is_lite(n_layer, n_vocab) => None,
        QProjRule::LoraUnlessLite => match meta_u64(file, &p("attention.q_lora_rank"))? as usize {
            0 => None,
            rank => Some(rank),
        },
    };
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
    // `llama-hparams.cpp:259-265`: `n_embd_head_k_mla()` is the `_mla`
    // key when the file has one and `n_embd_head_k()` -- plain
    // `attention.key_length` -- otherwise, which is the only spelling
    // `plm` has (`conversion/plm.py:16-17`, `plm.cpp:16-17`). The HF
    // spellings are still accepted after both, because ferrox's own
    // synthetic fixtures were written against them, but they are the
    // fallback rather than the contract.
    let qk_rope_head_dim = meta_u64(file, &p("rope.dimension_count"))
        .or_else(|_| meta_u64(file, &p("attention.qk_rope_head_dim")))?
        as usize;
    let key_length = meta_u64(file, &p("attention.key_length_mla"))
        .map(|v| (v, "attention.key_length_mla"))
        .or_else(|_| {
            meta_u64(file, &p("attention.key_length")).map(|v| (v, "attention.key_length"))
        });
    let qk_nope_head_dim = match key_length {
        Ok((k, key)) => (k as usize)
            .checked_sub(qk_rope_head_dim)
            .filter(|&nope| nope >= 1)
            .ok_or_else(|| {
                LoadError::MissingHparam(format!(
                    "{arch}.{key} ({k}) must exceed rope.dimension_count ({qk_rope_head_dim})"
                ))
            })?,
        Err(_) => meta_u64(file, &p("attention.qk_nope_head_dim"))? as usize,
    };
    let v_head_dim = meta_u64(file, &p("attention.value_length_mla"))
        .or_else(|_| meta_u64(file, &p("attention.value_length")))
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
    // A RoPE scaling is two things this engine does not have: the
    // frequency rewrite (`ggml_rope_ext`'s `freq_scale` / `ext_factor`
    // on `q_pe` and `k_pe`, `deepseek2.cpp:320-328`) and YaRN's
    // magnitude folded into the softmax scale (`:312-319`: `kq_scale =
    // mscale^2 / sqrt(n_embd_head_k_mla)`). Every real DeepSeek-V2 /
    // V3 export declares `yarn` (`conversion/deepseek.py:352-361`), so
    // this used to run them at factor 1 with the wrong scale; the
    // generic path implements both (`crate::loader`, `crate::
    // yarn_magnitude`) and has the goldens to show it. A file that
    // declares `none` is what `plm` and `conversion/deepseek.py:140`'s
    // dense exports write, and is not a scaling.
    if let Some(kind) = file
        .metadata_str(&p("rope.scaling.type"))
        .filter(|k| *k != "none")
    {
        return Err(LoadError::UnsupportedFeature(
            arch.clone(),
            format!(
                "`{arch}.rope.scaling.type = \"{kind}\"`: llama.cpp's deepseek2 graph rewrites the \
                 `pe` frequencies through `ggml_rope_ext` and folds YaRN's mscale into `kq_scale` \
                 (src/models/deepseek2.cpp:312-328); the MLA engine has plain RoPE and \
                 `1/sqrt(n_embd_head_k)` only, so it stops rather than run this file at factor 1"
            ),
        ));
    }
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
    // The per-position attention temperature: `deepseek2.cpp:46-47`
    // reads `attention.temperature_scale` and `attention.temperature_length`
    // ("used by mistral-large", its own comment) and :595-598 / :632-635
    // multiply Q by the per-token scale after RoPE; `mistral4` reuses
    // both (`models.h:1311-1318`). This engine has no per-position Q
    // scale and, unlike the generic decoder, no libllama-golden fixture
    // to check one against, so a file that declares a nonzero scale
    // STOPS here rather than running Mistral-Large-3 at temperature 1.
    // The resolution is the generic path's (`crate::attn_temperature`),
    // so the two loaders cannot disagree about which values mean "on".
    let declared = crate::attn_temperature::DeclaredTemperature {
        scale: file.metadata_f32(&p("attention.temperature_scale")),
        length: file.metadata_u64(&p("attention.temperature_length")),
        n_ctx_orig_yarn: None,
    };
    match crate::attn_temperature::resolve_attn_temperature(&arch, declared) {
        Ok(None) => {}
        Ok(Some(t)) => {
            return Err(LoadError::UnsupportedFeature(
                arch.clone(),
                format!(
                    "`{arch}.attention.temperature_scale = {}` (floor \
                     `{arch}.attention.temperature_length = {}`): llama.cpp's deepseek2 graph \
                     multiplies Q by a per-position temperature (src/models/deepseek2.cpp:46-47, \
                     595-598, 632-635; `crate::attn_temperature`) and the MLA engine has no \
                     per-position Q scale. Implemented on the generic path for `mistral3`; \
                     this engine refuses rather than run Mistral-Large-3 at temperature 1",
                    t.scale,
                    t.floor_scale.get()
                ),
            ));
        }
        Err(e) => {
            return Err(LoadError::UnsupportedFeature(
                arch.clone(),
                e.message(&arch),
            ));
        }
    }
    Ok(Deepseek2Hparams {
        arch,
        row,
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
    let q = match hp.q_lora_rank {
        Some(rank) => MlaQProj::LowRank {
            a: load_weight_matrix(file, &format!("blk.{l}.attn_q_a.weight"))?,
            norm: load_f32_vec_optional(file, &format!("blk.{l}.attn_q_a_norm.weight"))?
                .unwrap_or_else(|| vec![1.0; rank]),
            b: load_weight_matrix(file, &format!("blk.{l}.attn_q_b.weight"))?,
        },
        None => {
            // `deepseek2.cpp:104-115` create ONE of the two forms, so a
            // low-rank pair beside a direct `attn_q` is a file no graph
            // reads whole; it is refused rather than left unread.
            for stray in ["attn_q_a", "attn_q_b", "attn_q_a_norm"] {
                let name = format!("blk.{l}.{stray}.weight");
                if file.find_tensor(&name).is_some() {
                    return Err(LoadError::UnsupportedFeature(
                        hp.arch.clone(),
                        format!(
                            "{name} is present but this file projects Q directly ({:?}, \
                             `crate::mla_arch`): llama.cpp creates `attn_q` OR the low-rank pair, \
                             never both (src/models/deepseek2.cpp:104-115; plm.cpp:32)",
                            hp.row.q_proj
                        ),
                    ));
                }
            }
            MlaQProj::Direct(load_weight_matrix(file, &format!("blk.{l}.attn_q.weight"))?)
        }
    };
    let kv_a = load_weight_matrix(file, &format!("blk.{l}.attn_kv_a_mqa.weight"))?;
    let o_proj = load_weight_matrix(file, &format!("blk.{l}.attn_output.weight"))?;

    // Combined `attn_kv_b` or split `attn_k_b` / `attn_v_b`
    // (`crate::mla::MlaKvB`): the file carries one set, and llama.cpp
    // decides which it EXPECTS from the `_mla` keys (`is_mla()`). Read
    // by presence here, and refused when the file has both or neither,
    // because each such file fails in llama.cpp's own loader.
    let combined = format!("blk.{l}.attn_kv_b.weight");
    let k_b_name = format!("blk.{l}.attn_k_b.weight");
    let v_b_name = format!("blk.{l}.attn_v_b.weight");
    let kv_b = match (
        file.find_tensor(&combined).is_some(),
        file.find_tensor(&k_b_name).is_some(),
        file.find_tensor(&v_b_name).is_some(),
    ) {
        (true, false, false) => MlaKvB::Combined(load_weight_matrix(file, &combined)?),
        (false, true, true) => {
            // ne = [qk_nope, kv_lora, n_head] and [kv_lora, v_head, n_head]
            // (`deepseek2.cpp:120-122`): the per-head leading split is
            // the same cut `split_expert_tensor` makes for experts, and
            // gives `k_b[h]` as `[kv_lora, qk_nope]` (the absorb
            // direction, transposed by `conversion/deepseek.py:426`)
            // and `v_b[h]` as `[v_head, kv_lora]`.
            let k_b = split_expert_tensor(file, &k_b_name, hp.n_heads)?;
            let v_b = split_expert_tensor(file, &v_b_name, hp.n_heads)?;
            for (h, (k, v)) in k_b.iter().zip(v_b.iter()).enumerate() {
                if k.rows() != hp.kv_lora_rank
                    || k.cols() != hp.qk_nope_head_dim
                    || v.rows() != hp.v_head_dim
                    || v.cols() != hp.kv_lora_rank
                {
                    return Err(LoadError::UnsupportedFeature(
                        hp.arch.clone(),
                        format!(
                            "{k_b_name} / {v_b_name} head {h}: k_b is {}x{}, v_b is {}x{}; expected \
                             k_b [kv_lora_rank {}, qk_nope {}] and v_b [v_head {}, kv_lora_rank {}] \
                             (deepseek2.cpp:120-122)",
                            k.rows(),
                            k.cols(),
                            v.rows(),
                            v.cols(),
                            hp.kv_lora_rank,
                            hp.qk_nope_head_dim,
                            hp.v_head_dim,
                            hp.kv_lora_rank
                        ),
                    ));
                }
            }
            MlaKvB::Split { k_b, v_b }
        }
        (false, false, false) => {
            return Err(LoadError::Gguf(ferrox_gguf::GgufError::TensorNotFound(
                format!("{combined} (or the split {k_b_name} / {v_b_name})"),
            )));
        }
        (has_combined, has_k, has_v) => {
            return Err(LoadError::UnsupportedFeature(
                hp.arch.clone(),
                format!(
                    "layer {l} carries attn_kv_b={has_combined}, attn_k_b={has_k}, \
                     attn_v_b={has_v}: llama.cpp creates the combined tensor OR the split pair \
                     (deepseek2.cpp:118-123), never a mix"
                ),
            ));
        }
    };

    let kv_a_ln = load_f32_vec_optional(file, &format!("blk.{l}.attn_kv_a_norm.weight"))?
        .unwrap_or_else(|| vec![1.0; hp.kv_lora_rank]);

    Ok(MlaAttnWeights {
        q,
        kv_a_proj_with_mqa: kv_a,
        kv_a_layernorm: kv_a_ln,
        kv_b,
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

fn load_dense_ffn(
    file: &impl TensorSource,
    layer_idx: usize,
    hp: &Deepseek2Hparams,
) -> Result<MlaDenseFfn, LoadError> {
    let l = layer_idx;
    let act = hp.row.dense_act;
    let gate_name = format!("blk.{l}.ffn_gate.weight");
    let up_name = format!("blk.{l}.ffn_up.weight");
    let gate = if act.ungated().is_some() {
        // `plm.cpp:39-40` create `ffn_up` and `ffn_down` and no gate; a
        // file carrying one describes a graph this architecture does
        // not compute. The alias is the generic loader's (`arcee`): the
        // same tensor read again, which `run_expert` never applies.
        if file.find_tensor(&gate_name).is_some() {
            return Err(LoadError::UnsupportedFeature(
                hp.arch.clone(),
                format!(
                    "{gate_name} is present but this architecture's dense FFN is ungated \
                     ({act:?}: LLM_FFN_RELU_SQR under LLM_FFN_SEQ with a null gate, plm.cpp:181-187)"
                ),
            ));
        }
        load_weight_matrix(file, &up_name)?
    } else {
        load_weight_matrix(file, &gate_name)?
    };
    Ok(MlaDenseFfn {
        weights: ferrox_moe::ExpertWeights {
            gate,
            up: load_weight_matrix(file, &up_name)?,
            down: load_weight_matrix(file, &format!("blk.{l}.ffn_down.weight"))?,
        },
        act,
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
        MlaLayerFfn::Dense(load_dense_ffn(file, layer_idx, hp)?)
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
    let output_head = match hp.row.output_head {
        MlaOutputHead::OutputOrTied => match load_weight_matrix(file, "output.weight") {
            Ok(w) => w,
            Err(_) => load_weight_matrix(file, "token_embd.weight")?,
        },
        MlaOutputHead::TiedOnly => {
            if file.find_tensor("output.weight").is_some() {
                // Measured: libllama refuses `plm_decoy_output_tiny.gguf`
                // with "done_getting_tensors: wrong number of tensors;
                // expected 30, got 29".
                return Err(LoadError::UnsupportedFeature(
                    hp.arch.clone(),
                    format!(
                        "output.weight is present but `{}`'s lm_head is `token_embd` DUPLICATED \
                         ({}): llama.cpp never creates the tensor and its loader refuses the file \
                         for it (llama-model-loader.cpp:1309-1313)",
                        hp.arch, hp.row.lines
                    ),
                ));
            }
            load_weight_matrix(file, "token_embd.weight")?
        }
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
            q_lora_rank: hp.q_lora_rank.unwrap_or(0),
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
mod tests;
