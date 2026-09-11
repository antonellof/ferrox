//! Qwen3.5 / Qwen3-Next hybrid GGUF → GDN weight loader skeleton (**P3**).
//!
//! Reads `{arch}.*` hparams for `qwen35` / `qwen35moe` / `qwen3next`,
//! classifies each block as GDN (linear) vs full attention from tensor
//! presence, and can materialize [`crate::gdn::GdnWeights`] for GDN layers.
//!
//! Full [`crate::hybrid_engine::HybridEngine`] assemble / serve is **not**
//! wired: [`try_load`] always returns
//! [`LoadError::UnsupportedFeature`] listing what is still missing.
//! Unit tests prove GDN tensor → [`gdn_forward_token`] without serve.

use ferrox_gguf::TensorSource;

use crate::gdn::{GdnConfig, GdnWeights};
use crate::hybrid_engine::HybridEngine;
use crate::loader::LoadError;
use crate::loader::{load_f32_vec, load_weight_matrix};

/// Architectures this loader accepts.
pub const HYBRID_GDN_ARCHES: &[&str] = &["qwen35", "qwen35moe", "qwen3next"];

/// Hyperparameters from `{arch}.*` GGUF metadata (fail-closed if required keys absent).
#[derive(Debug, Clone)]
pub struct HybridHparams {
    pub arch: String,
    pub n_layer: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Depthwise conv kernel (`{arch}.ssm.conv_kernel`).
    pub ssm_conv_kernel: usize,
    /// Total V width (`{arch}.ssm.inner_size`) = `num_value_heads ·
    /// value_head_dim`. This is the **only** place GGUF records the V head
    /// dim, so [`gdn_config_from_hparams`] divides it by the V head count;
    /// see the failure mode documented there.
    pub ssm_inner_size: usize,
    /// GDN **key** head dim (`{arch}.ssm.state_size`) — not the V head dim.
    pub ssm_state_size: usize,
    /// Number of V heads (`{arch}.ssm.time_step_rank`).
    pub ssm_time_step_rank: usize,
    /// Number of K heads (`{arch}.ssm.group_count`); `≤ time_step_rank`
    /// for the real GQA-shaped Qwen3.5 / Qwen3-Next geometry.
    pub ssm_group_count: usize,
    /// Default full-attn interval when `attention.recurrent_layers` absent.
    pub full_attention_interval: usize,
    pub n_expert: usize,
}

/// Per-block attention kind from tensor presence (not metadata alone).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridLayerKind {
    /// Gated Delta Net / linear SSM block.
    Gdn,
    /// Standard GQA (`attn_q`).
    FullAttn,
}

fn meta_u64(file: &impl TensorSource, key: &str) -> Result<u64, LoadError> {
    file.metadata_u64(key)
        .ok_or_else(|| LoadError::MissingHparam(key.to_string()))
}

fn meta_f32(file: &impl TensorSource, key: &str, default: f32) -> f32 {
    file.metadata_f32(key).unwrap_or(default)
}

/// Read hybrid GDN hparams; errors if arch is wrong or required keys missing.
pub fn read_hybrid_hparams(file: &impl TensorSource) -> Result<HybridHparams, LoadError> {
    let arch = file
        .metadata_str("general.architecture")
        .ok_or_else(|| LoadError::MissingHparam("general.architecture".into()))?
        .to_string();
    if !HYBRID_GDN_ARCHES.contains(&arch.as_str()) {
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
    let n_kv_heads = file
        .metadata_u64(&p("attention.head_count_kv"))
        .unwrap_or(n_heads as u64) as usize;
    let head_dim = file
        .metadata_u64(&p("attention.key_length"))
        .map(|v| v as usize)
        .unwrap_or_else(|| hidden_dim / n_heads.max(1));

    let ssm_conv_kernel = meta_u64(file, &p("ssm.conv_kernel"))? as usize;
    let ssm_inner_size = meta_u64(file, &p("ssm.inner_size"))? as usize;
    let ssm_state_size = meta_u64(file, &p("ssm.state_size"))? as usize;
    let ssm_time_step_rank = meta_u64(file, &p("ssm.time_step_rank"))? as usize;
    let ssm_group_count = meta_u64(file, &p("ssm.group_count"))? as usize;
    let full_attention_interval = file
        .metadata_u64(&p("full_attention_interval"))
        .unwrap_or(4) as usize;
    let n_expert = file.metadata_u64(&p("expert_count")).unwrap_or(0) as usize;
    let rms_norm_eps = meta_f32(file, &p("attention.layer_norm_rms_epsilon"), 1e-6);
    let rope_theta = meta_f32(file, &p("rope.freq_base"), 10000.0);

    Ok(HybridHparams {
        arch,
        n_layer,
        hidden_dim,
        ffn_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        rms_norm_eps,
        rope_theta,
        ssm_conv_kernel,
        ssm_inner_size,
        ssm_state_size,
        ssm_time_step_rank,
        ssm_group_count,
        full_attention_interval,
        n_expert,
    })
}

/// Detect layer type from tensors:
/// - GDN if `ssm_conv1d.weight` **or** (`attn_qkv.weight` + `ssm_a`) present
/// - full attn if `attn_q.weight` present
pub fn detect_layer_kind(
    file: &impl TensorSource,
    layer_idx: usize,
) -> Result<HybridLayerKind, LoadError> {
    let l = layer_idx;
    let has_conv = file
        .find_tensor(&format!("blk.{l}.ssm_conv1d.weight"))
        .is_some();
    let has_qkv = file
        .find_tensor(&format!("blk.{l}.attn_qkv.weight"))
        .is_some();
    let has_ssm_a = file.find_tensor(&format!("blk.{l}.ssm_a")).is_some();
    let has_attn_q = file
        .find_tensor(&format!("blk.{l}.attn_q.weight"))
        .is_some();

    if has_conv || (has_qkv && has_ssm_a) {
        return Ok(HybridLayerKind::Gdn);
    }
    if has_attn_q {
        return Ok(HybridLayerKind::FullAttn);
    }
    Err(LoadError::Gguf(ferrox_gguf::GgufError::TensorNotFound(
        format!(
            "blk.{l}: neither GDN (ssm_conv1d / attn_qkv+ssm_a) nor full-attn (attn_q) tensors found"
        ),
    )))
}

/// Classify every trunk layer.
pub fn classify_layers(
    file: &impl TensorSource,
    hp: &HybridHparams,
) -> Result<Vec<HybridLayerKind>, LoadError> {
    let mut out = Vec::with_capacity(hp.n_layer);
    for i in 0..hp.n_layer {
        out.push(detect_layer_kind(file, i)?);
    }
    Ok(out)
}

fn load_f32_vec_first_of(
    file: &impl TensorSource,
    names: &[&str],
) -> Result<(String, Vec<f32>), LoadError> {
    for &name in names {
        if file.find_tensor(name).is_some() {
            return Ok((name.to_string(), load_f32_vec(file, name)?));
        }
    }
    Err(LoadError::Gguf(ferrox_gguf::GgufError::TensorNotFound(
        names.join(" | "),
    )))
}

/// Map `{arch}.ssm.*` metadata onto the GQA-shaped [`GdnConfig`].
///
/// The four GDN geometry numbers come from four distinct keys:
///
/// | `GdnConfig` field | GGUF key |
/// |---|---|
/// | `num_key_heads` | `ssm.group_count` |
/// | `num_value_heads` | `ssm.time_step_rank` |
/// | `key_head_dim` | `ssm.state_size` |
/// | `value_head_dim` | `ssm.inner_size / ssm.time_step_rank` |
///
/// `ssm.inner_size` is the **whole** V width (`num_value_heads ·
/// value_head_dim`), which is why the V head dim is a division and not a
/// key of its own. Two ways that can go wrong, both refused here rather
/// than silently mis-slicing the fused QKV projection:
///
/// * `time_step_rank` not dividing `inner_size` — the file is either
///   truncated or uses a different `inner_size` convention (e.g. writing
///   the full `conv_dim`), so the V head dim is unknowable.
/// * `time_step_rank` not a positive multiple of `group_count` — no whole
///   `repeat_interleave` factor exists, so some V heads would have no K
///   head. [`crate::gdn::gdn_forward_token`] asserts the same invariant;
///   catching it at load time turns a decode-time panic into a load error.
///
/// A wrong-but-divisible `inner_size` is still caught downstream in
/// [`load_gdn_layer_weights`], which cross-checks the derived
/// `value_head_dim` against the real `ssm_norm.weight` length.
fn gdn_config_from_hparams(hp: &HybridHparams) -> Result<GdnConfig, LoadError> {
    let num_value_heads = hp.ssm_time_step_rank;
    let num_key_heads = hp.ssm_group_count;
    if num_key_heads == 0 || num_value_heads == 0 {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!(
                "GDN head counts must be non-zero (ssm.group_count={num_key_heads}, \
                 ssm.time_step_rank={num_value_heads})"
            ),
        ));
    }
    if !num_value_heads.is_multiple_of(num_key_heads) {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!(
                "GDN V heads ({num_value_heads}, ssm.time_step_rank) must be a whole multiple \
                 of K heads ({num_key_heads}, ssm.group_count) — repeat_interleave has no \
                 integer replication factor otherwise"
            ),
        ));
    }
    if hp.ssm_inner_size == 0 || !hp.ssm_inner_size.is_multiple_of(num_value_heads) {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!(
                "cannot derive GDN value_head_dim: ssm.inner_size={} is not a positive multiple \
                 of ssm.time_step_rank={num_value_heads} (inner_size must be the total V width, \
                 num_value_heads × value_head_dim)",
                hp.ssm_inner_size
            ),
        ));
    }
    Ok(GdnConfig {
        hidden_dim: hp.hidden_dim,
        num_key_heads,
        num_value_heads,
        key_head_dim: hp.ssm_state_size,
        value_head_dim: hp.ssm_inner_size / num_value_heads,
        conv_kernel_size: hp.ssm_conv_kernel,
        rms_norm_eps: hp.rms_norm_eps,
    })
}

/// Load one GDN layer into [`GdnWeights`] (qwen35 split layout).
pub fn load_gdn_layer_weights(
    file: &impl TensorSource,
    layer_idx: usize,
    hp: &HybridHparams,
) -> Result<(GdnConfig, GdnWeights), LoadError> {
    let kind = detect_layer_kind(file, layer_idx)?;
    if kind != HybridLayerKind::Gdn {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!("blk.{layer_idx} is {kind:?}, not GDN — cannot load into GdnWeights"),
        ));
    }
    let cfg = gdn_config_from_hparams(hp)?;
    let l = layer_idx;

    let attn_qkv = load_weight_matrix(file, &format!("blk.{l}.attn_qkv.weight"))?;
    let attn_gate = load_weight_matrix(file, &format!("blk.{l}.attn_gate.weight"))?;
    let ssm_conv1d = load_f32_vec(file, &format!("blk.{l}.ssm_conv1d.weight"))?;
    let (_, ssm_dt) = load_f32_vec_first_of(
        file,
        &[&format!("blk.{l}.ssm_dt.bias"), &format!("blk.{l}.ssm_dt")],
    )?;
    let ssm_a = load_f32_vec(file, &format!("blk.{l}.ssm_a"))?;
    let ssm_beta = load_weight_matrix(file, &format!("blk.{l}.ssm_beta.weight"))?;
    let ssm_alpha = load_weight_matrix(file, &format!("blk.{l}.ssm_alpha.weight"))?;
    let ssm_norm = load_f32_vec(file, &format!("blk.{l}.ssm_norm.weight"))?;
    let ssm_out = load_weight_matrix(file, &format!("blk.{l}.ssm_out.weight"))?;

    // Head geometry before the conv taps: `qkv_dim` is itself derived from
    // the head counts and head dims, so a wrong `value_head_dim` would
    // otherwise surface as a confusing conv-length mismatch.
    let qkv_dim = cfg.qkv_dim();
    if ssm_dt.len() != cfg.num_value_heads || ssm_a.len() != cfg.num_value_heads {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!(
                "blk.{l} ssm_dt/ssm_a length mismatch: dt={}, a={}, num_value_heads={}",
                ssm_dt.len(),
                ssm_a.len(),
                cfg.num_value_heads
            ),
        ));
    }
    // `ssm_norm` is per V head, so its length is the authoritative
    // `value_head_dim` — this is what catches an `ssm.inner_size` that
    // divides evenly but means something other than the total V width.
    if ssm_norm.len() != cfg.value_head_dim {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!(
                "blk.{l}.ssm_norm.weight has {} elements, expected value_head_dim={} \
                 (derived as ssm.inner_size={} / ssm.time_step_rank={})",
                ssm_norm.len(),
                cfg.value_head_dim,
                hp.ssm_inner_size,
                hp.ssm_time_step_rank
            ),
        ));
    }
    // Fused QKV rows are 2*key_dim + value_dim; the z gate is value_dim
    // wide. A mismatch here means the split offsets would land inside the
    // wrong tensor — finite, correctly shaped, and wrong — so refuse.
    if attn_qkv.rows() != qkv_dim || attn_gate.rows() != cfg.value_dim() {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!(
                "blk.{l} projection geometry mismatch: attn_qkv has {} rows (expected \
                 2*key_dim + value_dim = {qkv_dim}), attn_gate has {} rows (expected \
                 value_dim={}); K heads={}, V heads={}, key_head_dim={}, value_head_dim={}",
                attn_qkv.rows(),
                attn_gate.rows(),
                cfg.value_dim(),
                cfg.num_key_heads,
                cfg.num_value_heads,
                cfg.key_head_dim,
                cfg.value_head_dim
            ),
        ));
    }
    let expected_conv = qkv_dim * cfg.conv_kernel_size;
    if ssm_conv1d.len() != expected_conv {
        return Err(LoadError::UnsupportedFeature(
            hp.arch.clone(),
            format!(
                "blk.{l}.ssm_conv1d.weight has {} elements, expected {expected_conv} \
                 (qkv_dim={qkv_dim} × kernel={})",
                ssm_conv1d.len(),
                cfg.conv_kernel_size
            ),
        ));
    }

    Ok((
        cfg,
        GdnWeights {
            attn_qkv,
            attn_gate,
            ssm_conv1d,
            ssm_dt,
            ssm_a,
            ssm_beta,
            ssm_alpha,
            ssm_norm,
            ssm_out,
        },
    ))
}

fn serve_gap_message(hp: &HybridHparams, kinds: &[HybridLayerKind]) -> String {
    let n_gdn = kinds.iter().filter(|k| **k == HybridLayerKind::Gdn).count();
    let n_full = kinds
        .iter()
        .filter(|k| **k == HybridLayerKind::FullAttn)
        .count();
    let mut missing = vec![
        "HybridEngine layer assemble (GDN + full-attn residuals / post-norm / FFN)".into(),
        "token_embd / output_norm / lm_head serve path".into(),
        "hybrid KV + recurrent state scheduling".into(),
    ];
    if n_full > 0 {
        missing.push(format!(
            "full-attn GQA decode for {n_full} layer(s) (attn_q path not wired into HybridEngine)"
        ));
    }
    if hp.n_expert > 0 {
        missing.push(format!("MoE expert routing (expert_count={})", hp.n_expert));
    }
    if hp.arch == "qwen3next" {
        missing.push("qwen3next legacy fused tensors (ssm_ba / ssm_in) if present".into());
    }
    format!(
        "hybrid GGUF hparams OK (n_layer={}, GDN={n_gdn}, full_attn={n_full}); \
         GDN layer weights loadable via load_gdn_layer_weights; serve blocked — missing: {}",
        hp.n_layer,
        missing.join("; ")
    )
}

/// Attempt HybridEngine load for serve — fail-closed with an inventory of gaps.
///
/// Still validates hparams + layer classification so callers get a precise
/// error rather than a silent generic-Decoder path.
pub fn try_load(file: &impl TensorSource) -> Result<HybridEngine, LoadError> {
    let hp = read_hybrid_hparams(file)?;
    let kinds = classify_layers(file, &hp)?;
    // Prove at least one GDN layer's tensors parse when present. No
    // equal-head precondition any more: the GQA-shaped geometry
    // (group_count < time_step_rank, key_head_dim != value_head_dim) is
    // exactly what the real Qwen3.5 / Qwen3-Next checkpoints carry.
    if let Some(i) = kinds.iter().position(|k| *k == HybridLayerKind::Gdn) {
        let _ = load_gdn_layer_weights(file, i, &hp)?;
    }
    Err(LoadError::UnsupportedFeature(
        hp.arch.clone(),
        serve_gap_message(&hp, &kinds),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn::{gdn_forward_token, GdnState};
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
        assert_eq!(
            values.len(),
            shape.iter().product::<u64>() as usize,
            "{name}"
        );
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
            buf.write_u32::<LittleEndian>(10).unwrap();
            buf.write_u64::<LittleEndian>(v).unwrap();
        }
        for &(k, v) in fkv {
            write_string(&mut buf, k);
            buf.write_u32::<LittleEndian>(6).unwrap();
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
            buf.write_u32::<LittleEndian>(0).unwrap(); // F32
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

    /// One synthetic GDN block's geometry. Tensors are always built for
    /// the *true* geometry; `inner_size_override` lets a test write an
    /// `ssm.inner_size` that disagrees with them, which is how a
    /// checkpoint using a different `inner_size` convention would look.
    struct GdnFixtureSpec {
        hidden: usize,
        num_key_heads: usize,
        num_value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        conv: usize,
        inner_size_override: Option<usize>,
    }

    impl GdnFixtureSpec {
        /// The equal-head geometry the loader supported before the GQA
        /// generalization: 2 K heads, 2 V heads, both head dims 2.
        fn equal_heads() -> Self {
            Self {
                hidden: 4,
                num_key_heads: 2,
                num_value_heads: 2,
                key_head_dim: 2,
                value_head_dim: 2,
                conv: 2,
                inner_size_override: None,
            }
        }

        /// The real GDN shape: fewer K heads than V heads, and K/V head
        /// dims that differ.
        fn gqa_heads() -> Self {
            Self {
                hidden: 4,
                num_key_heads: 1,
                num_value_heads: 2,
                key_head_dim: 2,
                value_head_dim: 3,
                conv: 2,
                inner_size_override: None,
            }
        }

        fn key_dim(&self) -> usize {
            self.num_key_heads * self.key_head_dim
        }

        fn value_dim(&self) -> usize {
            self.num_value_heads * self.value_head_dim
        }

        fn qkv_dim(&self) -> usize {
            2 * self.key_dim() + self.value_dim()
        }
    }

    fn gdn_fixture(spec: &GdnFixtureSpec) -> (std::path::PathBuf, GgufFile, HybridHparams) {
        let h = spec.hidden;
        let conv = spec.conv;
        let qkv = spec.qkv_dim();
        let v = spec.value_dim();
        let nv = spec.num_value_heads;
        let arch = "qwen35";

        let mut tensors = Vec::new();
        // Fixture shape = WeightMatrix [rows, cols]; build_gguf+loader reverse to GGML ne order.
        tensors.push(f32_tensor(
            "blk.0.attn_qkv.weight",
            vec![qkv as u64, h as u64],
            vec![0.05; qkv * h],
        ));
        tensors.push(f32_tensor(
            "blk.0.attn_gate.weight",
            vec![v as u64, h as u64],
            vec![0.04; v * h],
        ));
        // Flat layout [qkv_dim, kernel] as gdn::causal_conv_step expects.
        tensors.push(f32_tensor(
            "blk.0.ssm_conv1d.weight",
            vec![conv as u64, qkv as u64],
            {
                let mut c = vec![0.1; qkv * conv];
                for d in 0..qkv {
                    c[d * conv + (conv - 1)] = 1.0;
                }
                c
            },
        ));
        tensors.push(f32_tensor(
            "blk.0.ssm_dt.bias",
            vec![nv as u64],
            (0..nv)
                .map(|i| if i % 2 == 0 { 0.1 } else { -0.05 })
                .collect(),
        ));
        tensors.push(f32_tensor(
            "blk.0.ssm_a",
            vec![nv as u64],
            (0..nv)
                .map(|i| if i % 2 == 0 { -0.5 } else { -0.75 })
                .collect(),
        ));
        tensors.push(f32_tensor(
            "blk.0.ssm_beta.weight",
            vec![nv as u64, h as u64],
            vec![0.1; nv * h],
        ));
        tensors.push(f32_tensor(
            "blk.0.ssm_alpha.weight",
            vec![nv as u64, h as u64],
            vec![0.08; nv * h],
        ));
        tensors.push(f32_tensor(
            "blk.0.ssm_norm.weight",
            vec![spec.value_head_dim as u64],
            vec![1.0; spec.value_head_dim],
        ));
        tensors.push(f32_tensor(
            "blk.0.ssm_out.weight",
            vec![h as u64, v as u64],
            vec![0.06; h * v],
        ));

        // ssm.inner_size is the total V width; ssm.state_size is the K head dim.
        let inner_size = spec.inner_size_override.unwrap_or(v);
        let kv = [
            ("qwen35.block_count", 1u64),
            ("qwen35.embedding_length", h as u64),
            ("qwen35.feed_forward_length", 8u64),
            ("qwen35.attention.head_count", 2u64),
            ("qwen35.attention.head_count_kv", 2u64),
            ("qwen35.attention.key_length", spec.key_head_dim as u64),
            ("qwen35.ssm.conv_kernel", conv as u64),
            ("qwen35.ssm.inner_size", inner_size as u64),
            ("qwen35.ssm.state_size", spec.key_head_dim as u64),
            ("qwen35.ssm.time_step_rank", nv as u64),
            ("qwen35.ssm.group_count", spec.num_key_heads as u64),
        ];
        let fkv = [
            ("qwen35.attention.layer_norm_rms_epsilon", 1e-5f32),
            ("qwen35.rope.freq_base", 10000.0f32),
        ];
        let bytes = build_gguf(arch, &kv, &fkv, &tensors);
        let path = std::env::temp_dir().join(format!(
            "ferrox_hybrid_gdn_test_{}_{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).expect("synthetic hybrid GGUF must parse");
        let hp = read_hybrid_hparams(&file).expect("hparams");
        (path, file, hp)
    }

    fn tiny_gdn_fixture() -> (std::path::PathBuf, GgufFile, HybridHparams) {
        gdn_fixture(&GdnFixtureSpec::equal_heads())
    }

    /// `GdnWeights` is not `Debug` (it owns mapped quant bytes), so
    /// `unwrap_err` is unavailable on a load result.
    fn expect_load_error(result: Result<(GdnConfig, GdnWeights), LoadError>) -> LoadError {
        match result {
            Ok((cfg, _)) => panic!("expected a load error, got config {cfg:?}"),
            Err(e) => e,
        }
    }

    #[test]
    fn read_hparams_fails_clear_when_ssm_keys_missing() {
        let tensors = [f32_tensor("token_embd.weight", vec![4, 4], vec![0.0; 16])];
        let bytes = build_gguf(
            "qwen35",
            &[
                ("qwen35.block_count", 1),
                ("qwen35.embedding_length", 4),
                ("qwen35.feed_forward_length", 8),
                ("qwen35.attention.head_count", 2),
            ],
            &[],
            &tensors,
        );
        let path = std::env::temp_dir().join(format!(
            "ferrox_hybrid_missing_ssm_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let err = read_hybrid_hparams(&file).unwrap_err();
        std::fs::remove_file(&path).ok();
        match err {
            LoadError::MissingHparam(k) => assert!(k.contains("ssm.conv_kernel"), "{k}"),
            other => panic!("expected MissingHparam, got {other:?}"),
        }
    }

    #[test]
    fn synthetic_gdn_layer_loads_and_forward_token() {
        let (path, file, hp) = tiny_gdn_fixture();
        assert_eq!(detect_layer_kind(&file, 0).unwrap(), HybridLayerKind::Gdn);
        let (cfg, weights) = load_gdn_layer_weights(&file, 0, &hp).expect("load GDN");
        std::fs::remove_file(&path).ok();

        let mut state = GdnState::new(&cfg);
        let hidden = [0.2f32, -0.1, 0.3, -0.4];
        let out = gdn_forward_token(&weights, &cfg, &hidden, &mut state);
        assert_eq!(out.len(), hp.hidden_dim);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn try_load_returns_unsupported_feature_inventory() {
        let (path, file, _) = tiny_gdn_fixture();
        let err = try_load(&file).unwrap_err();
        std::fs::remove_file(&path).ok();
        match err {
            LoadError::UnsupportedFeature(arch, msg) => {
                assert_eq!(arch, "qwen35");
                assert!(msg.contains("HybridEngine"), "{msg}");
                assert!(
                    msg.contains("serve blocked") || msg.contains("missing"),
                    "{msg}"
                );
                assert!(msg.contains("load_gdn_layer_weights"), "{msg}");
            }
            other => panic!("expected UnsupportedFeature, got {other:?}"),
        }
    }

    /// The geometry the loader used to refuse outright: `ssm.group_count`
    /// (K heads) below `ssm.time_step_rank` (V heads), with a
    /// `value_head_dim` that differs from `ssm.state_size`. The config it
    /// derives must carry the reference's four independent numbers and the
    /// `[key_dim, key_dim, value_dim]` split — under the old equal-head
    /// formula this file's `attn_qkv` (10 rows) would have been read as
    /// `3 * 2 * 2 = 12` rows split into equal thirds.
    #[test]
    fn unequal_gdn_head_geometry_loads_instead_of_being_refused() {
        let spec = GdnFixtureSpec::gqa_heads();
        let (path, file, hp) = gdn_fixture(&spec);
        let loaded = load_gdn_layer_weights(&file, 0, &hp);
        std::fs::remove_file(&path).ok();
        let (cfg, weights) = loaded.expect("GQA-shaped GDN layer must load");

        assert_eq!(cfg.num_key_heads, 1);
        assert_eq!(cfg.num_value_heads, 2);
        assert_eq!(cfg.key_head_dim, 2, "from ssm.state_size");
        assert_eq!(
            cfg.value_head_dim, 3,
            "from ssm.inner_size / ssm.time_step_rank"
        );
        assert_eq!(cfg.key_dim(), 2);
        assert_eq!(cfg.value_dim(), 6);
        assert_eq!(cfg.qkv_dim(), 10, "2*key_dim + value_dim, not 3*n*head_dim");
        assert_eq!(cfg.heads_per_key_group(), 2);

        let mut state = GdnState::new(&cfg);
        let hidden = [0.2f32, -0.1, 0.3, -0.4];
        let out = gdn_forward_token(&weights, &cfg, &hidden, &mut state);
        assert_eq!(out.len(), hp.hidden_dim);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    /// Regression: equal K/V heads and equal head dims must still derive
    /// exactly the config they did before the generalization, so no
    /// already-loadable checkpoint changes shape.
    #[test]
    fn equal_head_metadata_still_derives_the_pre_generalization_config() {
        let (path, file, hp) = tiny_gdn_fixture();
        let loaded = load_gdn_layer_weights(&file, 0, &hp);
        std::fs::remove_file(&path).ok();
        let (cfg, _) = loaded.expect("equal-head GDN layer must load");

        assert_eq!(cfg.num_key_heads, cfg.num_value_heads);
        assert_eq!(cfg.key_head_dim, cfg.value_head_dim);
        assert_eq!(cfg.num_value_heads, 2);
        assert_eq!(cfg.value_head_dim, 2);
        assert_eq!(cfg.qkv_dim(), 3 * 2 * 2, "equal heads: still three thirds");
        assert_eq!(cfg.heads_per_key_group(), 1);
    }

    /// `ssm.inner_size` is the only record of the V head dim, so a file
    /// whose `inner_size` is not a whole multiple of `time_step_rank`
    /// leaves it unknowable — refuse rather than round.
    #[test]
    fn inner_size_not_divisible_by_v_head_count_is_refused() {
        let mut spec = GdnFixtureSpec::gqa_heads();
        spec.inner_size_override = Some(7); // 7 % 2 != 0
        let (path, file, hp) = gdn_fixture(&spec);
        let err = expect_load_error(load_gdn_layer_weights(&file, 0, &hp));
        std::fs::remove_file(&path).ok();
        match err {
            LoadError::UnsupportedFeature(arch, msg) => {
                assert_eq!(arch, "qwen35");
                assert!(msg.contains("value_head_dim"), "{msg}");
                assert!(msg.contains("inner_size"), "{msg}");
            }
            other => panic!("expected UnsupportedFeature, got {other:?}"),
        }
    }

    /// A divisible-but-wrong `inner_size` (here: the whole fused QKV
    /// width, a plausible alternative convention) derives the wrong V head
    /// dim. `ssm_norm.weight` is per V head and settles it, so the load
    /// fails loudly instead of mis-splitting the projection.
    #[test]
    fn inner_size_disagreeing_with_ssm_norm_length_is_refused() {
        let mut spec = GdnFixtureSpec::gqa_heads();
        spec.inner_size_override = Some(spec.qkv_dim()); // 10 / 2 = 5 != 3
        let (path, file, hp) = gdn_fixture(&spec);
        let err = expect_load_error(load_gdn_layer_weights(&file, 0, &hp));
        std::fs::remove_file(&path).ok();
        match err {
            LoadError::UnsupportedFeature(_, msg) => {
                assert!(msg.contains("ssm_norm.weight"), "{msg}");
                assert!(msg.contains("value_head_dim"), "{msg}");
            }
            other => panic!("expected UnsupportedFeature, got {other:?}"),
        }
    }

    /// V heads must be a whole multiple of K heads — otherwise
    /// `repeat_interleave` has no integer factor and some V heads would
    /// pair with a K head that never fed them. Caught at load time so the
    /// forward pass never has to panic on it.
    #[test]
    fn v_heads_not_a_multiple_of_k_heads_is_refused_at_load() {
        let spec = GdnFixtureSpec {
            hidden: 4,
            num_key_heads: 3,
            num_value_heads: 4,
            key_head_dim: 2,
            value_head_dim: 2,
            conv: 2,
            inner_size_override: None,
        };
        let (path, file, hp) = gdn_fixture(&spec);
        let err = expect_load_error(load_gdn_layer_weights(&file, 0, &hp));
        std::fs::remove_file(&path).ok();
        match err {
            LoadError::UnsupportedFeature(_, msg) => {
                assert!(msg.contains("repeat_interleave"), "{msg}");
                assert!(msg.contains("whole multiple"), "{msg}");
            }
            other => panic!("expected UnsupportedFeature, got {other:?}"),
        }
    }

    /// `try_load` still fails closed on serve, but unequal K/V heads is no
    /// longer one of the reasons it lists — the layer loads now.
    #[test]
    fn try_load_no_longer_reports_unequal_gdn_heads_as_a_gap() {
        let (path, file, _) = gdn_fixture(&GdnFixtureSpec::gqa_heads());
        let err = try_load(&file).unwrap_err();
        std::fs::remove_file(&path).ok();
        match err {
            LoadError::UnsupportedFeature(_, msg) => {
                assert!(!msg.contains("unequal GDN K/V heads"), "{msg}");
                assert!(msg.contains("load_gdn_layer_weights"), "{msg}");
            }
            other => panic!("expected UnsupportedFeature, got {other:?}"),
        }
    }

    #[test]
    fn full_attn_layer_detected_from_attn_q() {
        let tensors = [f32_tensor(
            "blk.0.attn_q.weight",
            vec![4u64, 4u64],
            vec![0.0; 16],
        )];
        let bytes = build_gguf(
            "qwen35",
            &[
                ("qwen35.block_count", 1),
                ("qwen35.embedding_length", 4),
                ("qwen35.feed_forward_length", 8),
                ("qwen35.attention.head_count", 2),
                ("qwen35.ssm.conv_kernel", 2),
                ("qwen35.ssm.inner_size", 12),
                ("qwen35.ssm.state_size", 2),
                ("qwen35.ssm.time_step_rank", 2),
                ("qwen35.ssm.group_count", 2),
            ],
            &[],
            &tensors,
        );
        let path = std::env::temp_dir().join(format!(
            "ferrox_hybrid_full_attn_{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(
            detect_layer_kind(&file, 0).unwrap(),
            HybridLayerKind::FullAttn
        );
        std::fs::remove_file(&path).ok();
    }
}
