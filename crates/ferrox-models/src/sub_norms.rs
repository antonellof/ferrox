//! **THE TWO NORMS INSIDE THE BLOCKS** -- BitNet's `attn_sub_norm` and
//! `ffn_sub_norm`, as one fact a `ModelConfig` carries, one table that
//! says which architecture has them, and one loader for the pair.
//!
//! # What they are
//!
//! A generic decoder layer has four norm sites, all OUTSIDE the two
//! sublayers (`crate::norm_sites`): before and after attention, before
//! and after the FFN. `src/models/bitnet.cpp` adds two INSIDE them:
//!
//! ```text
//! attn:  x -> attn_norm -> qkv -> rope -> attend -> [attn_sub_norm] -> wo
//! ffn :  x -> ffn_norm -> gate, up -> silu(gate) * up -> [ffn_sub_norm] -> down
//! ```
//!
//! `bitnet.cpp:24` creates `attn_sub_norm` `{n_embd}` REQUIRED and
//! `:101-106` applies it, RMS, to the attention output -- the
//! concatenated heads after the softmax-weighted V sum -- and THEN
//! `:107` runs `wo`. That is the other side of a matmul from Gemma's
//! `post_attention_norm`, which `AttnWeights::post_attn_norm` applies
//! AFTER `wo`; reading one as the other moves a norm across a
//! projection. `:36` creates `ffn_sub_norm` `{n_ff}` REQUIRED, `:127-132`
//! call `build_ffn` with a NULL down projection so it returns the
//! `silu(gate) * up` product, `:135-140` norm that, and `:141` apply
//! `ffn_down` by hand. The LM head is `tok_embd` unconditionally
//! (`:164`; `:14-17` create no `output` tensor), which is the tied
//! embedding ferrox already takes when `output.weight` is absent.
//!
//! # Reach -- MEASURED, not read off one file
//!
//! `grep -l 'attn_sub_norm\|ffn_sub_norm' src/models/*.cpp` over all
//! 140 graphs (2026-09-12) is `bitnet.cpp`. `LLM_TENSOR_ATTN_SUB_NORM`
//! and `LLM_TENSOR_FFN_SUB_NORM` (`llama-arch.cpp:510-511`) are created
//! by no other `load_arch_tensors`. So [`SUB_NORM_ARCHS`] has one row,
//! and the fact is a `bool` rather than an enum: there is no second
//! shape to name, and a variant with no caller is the OLMo lesson
//! (`capability::WEIGHTED_LAYER_NORM`).
//!
//! # Why a model-wide fact and not two `Option`s alone
//!
//! The two tensors live on the layer (`AttnWeights::attn_sub_norm`,
//! `MoeWeights::ffn_sub_norm`), because that is where the arithmetic
//! reads them. But every fused Metal launch is admitted by
//! `Decoder::metal_can_serve_model`, which takes the CONFIG, and none
//! of those kernels has a norm between attention and `wo` or between
//! the activation and `down`. So the architecture's answer is resolved
//! once into `ModelConfig::block_sub_norms`, the loader reads it to
//! decide whether the tensors are REQUIRED, and the Metal predicate
//! reads it to refuse. One fact, two readers; a file whose tensors and
//! architecture disagree is refused either way -- missing tensors on a
//! `bitnet` file by name, unread tensors on any other by the
//! unconsumed-tensor gate.
//!
//! # Where they are applied
//!
//! `attn_sub_norm` in `Decoder::attn_out_to_residual_rows`, the ONE
//! tail every host attention body ends in, after the output gate (no
//! graph has both; the order is documented there) and before `o_proj`.
//! `ffn_sub_norm` in `ferrox_moe::run_expert_sub_normed` for a row and
//! in `Decoder::dense_ffn_batch` for a batch, which is every dense FFN
//! body there is; the routed-expert bodies never see it because no MoE
//! graph has one (`build_moe_ffn` has no such site), and the loader
//! refuses the pair on a MoE layer by construction of the table.
//!
//! # What ferrox does NOT do that llama.cpp does on a BitNet file
//!
//! `bitnet.cpp:27-43` also create OPTIONAL per-tensor `blk.N.<proj>.scale`
//! tensors that `build_lora_mm` multiplies each projection's output by
//! (`llama-graph.cpp:1492-1494`), and since `llama-model.cpp:1355-1400`
//! every architecture's loader picks them up. The current converter
//! writes none (`conversion/bitnet.py:23-32` folds the scale into the
//! ternary weights); older exports carry them, and libllama's logits
//! MOVE when they are present (measured: the two fixtures differ at the
//! first logit). ferrox does not apply them and refuses such a file by
//! name (`crate::weight_scales`), rather than running it at the wrong
//! scale under the unread-tensor gate's `FERROX_ALLOW_UNKNOWN_TENSORS`.

use crate::loader::load_f32_vec;
use crate::LoadError;
use ferrox_gguf::TensorSource;

/// Architectures whose blocks carry the two inner norms, with the
/// graph lines that create and apply them.
pub const SUB_NORM_ARCHS: &[(&str, &str)] =
    &[("bitnet", "src/models/bitnet.cpp:24,36,101-106,135-140")];

/// Whether this architecture's blocks norm INSIDE the two sublayers.
pub fn block_sub_norms(arch: &str) -> bool {
    SUB_NORM_ARCHS.iter().any(|(name, _)| *name == arch)
}

/// Layer `l`'s two inner norm weights, for a model whose config says it
/// has them: `attn_sub_norm` at `hidden_dim` and `ffn_sub_norm` at the
/// layer's FFN width, both REQUIRED as `bitnet.cpp:24,36` require them.
///
/// `None` for every other model, without touching the file, so a
/// tensor of that name on an architecture whose graph has no such
/// site stays UNREAD and is refused as such.
pub fn load_sub_norms(
    file: &impl TensorSource,
    arch: &str,
    block_sub_norms: bool,
    l: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> Result<Option<SubNorms>, LoadError> {
    if !block_sub_norms {
        return Ok(None);
    }
    let attn_name = format!("blk.{l}.attn_sub_norm.weight");
    let ffn_name = format!("blk.{l}.ffn_sub_norm.weight");
    let attn = load_f32_vec(file, &attn_name)?;
    let ffn = load_f32_vec(file, &ffn_name)?;
    let check = |name: &str, got: usize, want: usize| {
        if got == want {
            Ok(())
        } else {
            Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!("{name} has {got} entries, expected {want}"),
            ))
        }
    };
    check(&attn_name, attn.len(), hidden_dim)?;
    check(&ffn_name, ffn.len(), ffn_dim)?;
    Ok(Some(SubNorms { attn, ffn }))
}

/// One layer's pair, as the loader hands them out; the decoder keeps
/// each on the sublayer that reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct SubNorms {
    /// `blk.N.attn_sub_norm.weight`, `[hidden_dim]`.
    pub attn: Vec<f32>,
    /// `blk.N.ffn_sub_norm.weight`, `[ffn_dim]`.
    pub ffn: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one row, and the default everywhere else -- including the
    /// architectures whose norms sit on the OUTSIDE of the same
    /// sublayers (Gemma's post-norms, OLMo-2's), which this table must
    /// not be confused with.
    #[test]
    fn only_bitnet_norms_inside_the_blocks() {
        assert!(block_sub_norms("bitnet"));
        for arch in [
            "llama", "gemma2", "gemma3", "olmo2", "exaone4", "qwen3", "talkie",
        ] {
            assert!(!block_sub_norms(arch), "{arch}");
        }
    }

    /// Every table row is an architecture the generic loader can reach
    /// and has audited, so the seam it names is a seam something asks
    /// and something evidences.
    #[test]
    fn every_table_row_is_on_the_generic_path() {
        for (arch, line) in SUB_NORM_ARCHS {
            let profile = crate::capability::resolve_profile(arch)
                .unwrap_or_else(|| panic!("`{arch}` ({line}) is not a registered architecture"));
            assert!(
                matches!(profile.path, crate::capability::ArchPath::GenericGqa { .. }),
                "`{arch}` ({line}) is {:?}, and only the generic decoder reads this table",
                profile.path
            );
            assert!(
                crate::capability::AUDITED_GENERIC_GQA.contains(arch),
                "`{arch}` is served here and must be audited, or the seam is unevidenced"
            );
        }
    }
}
