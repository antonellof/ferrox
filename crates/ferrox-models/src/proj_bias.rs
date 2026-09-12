//! **PROJECTION BIASES ON THE GENERIC PATH** -- `attn_output.bias` and
//! the dense FFN's `ffn_{up,gate,down}.bias`, which architectures'
//! graphs create them, and how the loader fills the two slots that
//! apply them.
//!
//! # What it is
//!
//! `build_attn(wo, wo_b, wo_s, ...)` adds `wo_b` right after the output
//! projection (after `wo_s`, which `build_lora_mm` multiplies inside),
//! and `build_ffn(up, up_b, gate, gate_b, down, down_b, ...)` adds each
//! bias right after its matmul, before the activation for `up_b` /
//! `gate_b` and after `down` for `down_b` (`llama-graph.cpp`). The
//! arithmetic is the same in every graph; what differs is whether the
//! graph CREATES the tensor -- required, optional, or not at all -- and
//! that is the per-architecture fact this module holds, because a bias
//! tensor in a file whose graph never creates it is a file llama.cpp
//! refuses (`done_getting_tensors: wrong number of tensors`), and
//! ferrox refuses it as unread.
//!
//! # Reach -- MEASURED
//!
//! Over all 140 `src/models/*.cpp` (2026-09-12):
//!
//! ```text
//! grep -l 'ATTN_OUT, *"bias"'   src/models/*.cpp   # 33 graphs
//! grep -l 'FFN_UP, *"bias"'     src/models/*.cpp   # 27 graphs
//! grep -l 'FFN_GATE, *"bias"'   src/models/*.cpp   # 11 graphs
//! ```
//!
//! restricted to the generic path, with `REQ` where `create_tensor(...,
//! 0)` and `opt` where `TENSOR_NOT_REQUIRED`: [`ATTN_OUT_BIAS_CREATORS`]
//! and [`FFN_BIAS_CREATORS`] below carry every row and its flag. The
//! rows that REQUIRE them and refused for nothing else -- `starcoder2`,
//! `codeshell`, `jais2` -- close on this module
//! (`tests/proj_bias_graphs.rs`); the rows that create them OPTIONAL
//! (`llama` itself, `granite`, `deci`, `mistral3`, `minicpm`, `nemotron`,
//! `apertus`, `ernie4_5`) used to refuse a file that carried them as
//! unread, where llama.cpp applies them.
//!
//! `gpt-oss` (`openai-moe.cpp:51`) REQUIRES `attn_output.bias`, and its
//! bias used to live on the gpt-oss side table (`GptOssLayer::o_bias`)
//! beside its router and expert biases, which spelled the rule as "arch
//! is gpt-oss"; it is [`crate::decoder::AttnWeights::o_bias`] now, the
//! slot this table fills for thirteen generic rows, applied in the ONE
//! attention tail.
//!
//! # Where the arithmetic is
//!
//! `AttnWeights::o_bias` in `Decoder::attn_out_to_residual_rows`, after
//! `o_scale` and before `attn_value_scale` and `post_attn_norm`, the
//! order `build_attn` has. `MoeWeights::dense_bias`
//! (`ferrox_moe::DenseBias`) in `Decoder::run_dense_expert`
//! (`ferrox_moe::run_expert_biased`) and `Decoder::dense_ffn_batch`,
//! whose fused Metal launch has no bias site and is fenced on the same
//! field; `Decoder::metal_attn_view` answers `None` for a layer with an
//! `o_bias`, so no fused attention launch drops it.

use ferrox_gguf::TensorSource;
use ferrox_moe::DenseBias;

use crate::loader::{load_f32_vec, LoadError};

/// Whether a graph creates a bias tensor as required or optional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    /// `create_tensor(..., 0)`: llama.cpp fails to load without it.
    Required,
    /// `TENSOR_NOT_REQUIRED`: applied when present.
    Optional,
}

/// Generic-path graphs that create `blk.N.attn_output.bias`.
pub const ATTN_OUT_BIAS_CREATORS: &[(&str, Presence)] = &[
    ("apertus", Presence::Optional),
    ("codeshell", Presence::Required),
    ("deci", Presence::Optional),
    ("ernie4_5", Presence::Optional),
    ("gpt-oss", Presence::Required),
    ("granite", Presence::Optional),
    ("granitemoe", Presence::Optional),
    ("granite-moe", Presence::Optional),
    ("jais2", Presence::Required),
    ("llama", Presence::Optional),
    ("minicpm", Presence::Optional),
    ("mistral3", Presence::Optional),
    ("nemotron", Presence::Optional),
    ("phimoe", Presence::Required),
    ("starcoder2", Presence::Required),
];

/// Generic-path graphs that create `blk.N.ffn_up.bias` and
/// `blk.N.ffn_down.bias` (the same flag for both in every graph), and
/// whether they also create `blk.N.ffn_gate.bias`.
pub const FFN_BIAS_CREATORS: &[(&str, Presence, bool)] = &[
    ("codeshell", Presence::Required, false),
    ("deci", Presence::Optional, true),
    ("granite", Presence::Optional, true),
    ("granitemoe", Presence::Optional, true),
    ("granite-moe", Presence::Optional, true),
    ("jais2", Presence::Required, false),
    ("llama", Presence::Optional, true),
    ("minicpm", Presence::Optional, true),
    ("mistral3", Presence::Optional, true),
    ("nemotron", Presence::Optional, false),
    ("starcoder2", Presence::Required, false),
];

fn attn_out_presence(arch: &str) -> Option<Presence> {
    ATTN_OUT_BIAS_CREATORS
        .iter()
        .find(|(n, _)| *n == arch)
        .map(|(_, p)| *p)
}

fn ffn_presence(arch: &str) -> Option<(Presence, bool)> {
    FFN_BIAS_CREATORS
        .iter()
        .find(|(n, _, _)| *n == arch)
        .map(|(_, p, g)| (*p, *g))
}

fn load_bias(
    file: &impl TensorSource,
    arch: &str,
    name: &str,
    presence: Presence,
    len: usize,
) -> Result<Option<Vec<f32>>, LoadError> {
    if file.find_tensor(name).is_none() {
        return match presence {
            Presence::Required => Err(LoadError::Gguf(ferrox_gguf::GgufError::TensorNotFound(
                format!("{name} (REQUIRED by `{arch}`'s graph)"),
            ))),
            Presence::Optional => Ok(None),
        };
    }
    let v = load_f32_vec(file, name)?;
    if v.len() != len {
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!("{name} has {} entries, expected {len}", v.len()),
        ));
    }
    Ok(Some(v))
}

/// Layer `l`'s `attn_output.bias`: `Some` when the architecture's graph
/// creates it and the file has it, an error when the graph requires it
/// and the file lacks it, `None` otherwise -- including for an
/// architecture whose graph never creates it, where a present tensor
/// stays unread and is refused as such.
pub fn load_attn_out_bias(
    file: &impl TensorSource,
    arch: &str,
    l: usize,
    hidden_dim: usize,
) -> Result<Option<Vec<f32>>, LoadError> {
    let Some(presence) = attn_out_presence(arch) else {
        return Ok(None);
    };
    load_bias(
        file,
        arch,
        &format!("blk.{l}.attn_output.bias"),
        presence,
        hidden_dim,
    )
}

/// Layer `l`'s dense FFN biases, on a DENSE layer only: `Some` when the
/// architecture's graph creates them and the file has at least one.
/// `ffn_dim` is the layer's `up` width; `ungated` says the gate is an
/// alias of `up` (`ModelConfig::ffn_is_ungated`), in which case no gate
/// bias is looked for -- the graph has no gate to bias.
pub fn load_dense_ffn_bias(
    file: &impl TensorSource,
    arch: &str,
    l: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    ungated: bool,
) -> Result<Option<DenseBias>, LoadError> {
    let Some((presence, has_gate)) = ffn_presence(arch) else {
        return Ok(None);
    };
    let up = load_bias(
        file,
        arch,
        &format!("blk.{l}.ffn_up.bias"),
        presence,
        ffn_dim,
    )?;
    let down = load_bias(
        file,
        arch,
        &format!("blk.{l}.ffn_down.bias"),
        presence,
        hidden_dim,
    )?;
    let gate = if has_gate && !ungated {
        // Every graph that creates a gate bias creates it OPTIONAL
        // beside an optional up/down pair; none requires it alone.
        load_bias(
            file,
            arch,
            &format!("blk.{l}.ffn_gate.bias"),
            Presence::Optional,
            ffn_dim,
        )?
    } else {
        None
    };
    let bias = DenseBias { gate, up, down };
    Ok((!bias.is_empty()).then_some(bias))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row is a generic-path architecture; the three REQUIRED
    /// rows that close on this module are audited; a required row that
    /// is not audited is refused for something else and says so in its
    /// reason (`tests/attn_bias.rs` pins the bias half of that).
    #[test]
    fn every_creator_is_a_generic_row() {
        let names: Vec<&str> = ATTN_OUT_BIAS_CREATORS
            .iter()
            .map(|(n, _)| *n)
            .chain(FFN_BIAS_CREATORS.iter().map(|(n, _, _)| *n))
            .collect();
        for name in names {
            let profile = crate::capability::resolve_profile(name)
                .unwrap_or_else(|| panic!("`{name}` is not a registered architecture"));
            assert!(
                matches!(
                    profile.path,
                    crate::capability::ArchPath::GenericGqa { .. }
                        | crate::capability::ArchPath::DedicatedOnly { .. }
                ),
                "`{name}` is {:?}",
                profile.path
            );
        }
        for name in ["starcoder2", "codeshell", "jais2"] {
            assert!(
                crate::capability::AUDITED_GENERIC_GQA.contains(&name),
                "{name}"
            );
            assert_eq!(attn_out_presence(name), Some(Presence::Required));
            assert!(matches!(
                ffn_presence(name),
                Some((Presence::Required, false))
            ));
        }
        assert_eq!(attn_out_presence("gpt-oss"), Some(Presence::Required));
        assert_eq!(attn_out_presence("qwen3"), None);
        assert_eq!(ffn_presence("qwen3"), None);
    }
}
