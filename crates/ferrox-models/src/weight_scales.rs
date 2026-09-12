//! **PER-TENSOR WEIGHT SCALES** -- the optional `<tensor>.scale` and
//! `<tensor>.input_scale` companions llama.cpp multiplies a projection's
//! output by, which ferrox does not apply and therefore refuses by name.
//!
//! # What they are
//!
//! `build_lora_mm(w, cur, w_s)` (`llama-graph.cpp:1486-1494`) is
//! `res = mul_mat(w, cur); if (w_s) res = mul(res, w_s)`, and its
//! `mul_mat_id` twin does the same per expert (`:1521-1530`). Since
//! `llama-model.cpp:1355-1440` a GENERIC pass after every architecture's
//! `load_arch_tensors` creates `{1}` `scale` and `input_scale` tensors
//! as `TENSOR_NOT_REQUIRED` beside every projection it finds -- Q, K,
//! V, O, fused QKV, the attention gate, dense gate/up/down, shared
//! experts, routed experts (`{n_expert}`), the recurrent projections,
//! the NextN head -- and `:1506-1507` does it for `output`. So ANY
//! architecture's file may carry them, and llama.cpp honours them on
//! all of them.
//!
//! # Who writes them
//!
//! Two converters. `conversion/base.py:687,780` write `.scale` (the
//! NVFP4 per-tensor `scale2`) and `.input_scale` beside NVFP4-packed
//! weights, a quantization type ferrox has no kernel for, so such a
//! file stops on the dtype before it reaches here. And BitNet: `src/
//! models/bitnet.cpp:27-43` created the seven per-projection scales
//! BEFORE the generic pass existed, for the exports whose ternary
//! weights were stored unscaled; the CURRENT `conversion/bitnet.py:
//! 23-32` folds the scale into the weights and writes no tensor, but
//! older files carry them and libllama still applies them -- measured,
//! `tests/sub_norm_graphs.rs`: the same fixture with and without seven
//! `2.0` scales gives DIFFERENT logits under libllama. And `talkie`
//! (`conversion/talkie.py:26-31`), which writes `attn_output.scale` and
//! `ffn_down.scale` from its `attn_gain` / `mlp_gain` on every export
//! and closed on exactly those two: [`SERVED_SCALE_TENSORS`] are the
//! companions the decoder applies (`AttnWeights::o_scale`,
//! `MoeWeights::down_scale`), for any architecture, as upstream.
//!
//! # Why a refusal by name
//!
//! Without this, such a file died on the unread-tensor gate
//! (`loader::assert_every_tensor_consumed`) -- the right outcome with
//! the wrong message, and one `FERROX_ALLOW_UNKNOWN_TENSORS=1` turns
//! into a model running every projection at the wrong magnitude. A
//! multiply per projection is not hard to add; it is not added because
//! no file ferrox can otherwise run needs it today, and a seam with no
//! caller is the OLMo lesson (`capability::WEIGHTED_LAYER_NORM`). When
//! one does, this module is where the census already is.

use crate::LoadError;

/// The suffixes `llama-model.cpp:1355-1440` create under every
/// projection name.
const SCALE_SUFFIXES: [&str; 2] = [".scale", ".input_scale"];

/// The two companions ferrox APPLIES: `blk.N.attn_output.scale` (`wo_s`,
/// multiplied onto the attention branch after `wo`) and
/// `blk.N.ffn_down.scale` (`ffn_down_s`, onto the dense FFN after
/// `down`). They are the two `talkie`'s converter writes on every
/// export (`conversion/talkie.py:26-31`), and `build_lora_mm`'s
/// `res = mul(mul_mat(w, x), s)` is the same multiply for both; every
/// other companion is still refused below, because nothing ferrox can
/// otherwise run writes it.
pub const SERVED_SCALE_TENSORS: [&str; 2] = ["attn_output.scale", "ffn_down.scale"];

/// Whether `name` is one of the two companions the decoder applies.
fn is_served(name: &str) -> bool {
    SERVED_SCALE_TENSORS.iter().any(|s| {
        name.strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
            .is_some_and(|(_, tail)| tail == *s)
    })
}

/// Layer `l`'s `blk.N.<proj>.scale`, a `{1}` tensor, when the file has
/// one. Optional everywhere, as `TENSOR_NOT_REQUIRED` upstream.
pub fn load_projection_gain(
    file: &impl ferrox_gguf::TensorSource,
    arch: &str,
    l: usize,
    proj: &str,
) -> Result<Option<f32>, LoadError> {
    let name = format!("blk.{l}.{proj}.scale");
    if file.find_tensor(&name).is_none() {
        return Ok(None);
    }
    let v = crate::loader::load_f32_vec(file, &name)?;
    if v.len() != 1 {
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!("{name} has {} entries; a per-tensor scale is one", v.len()),
        ));
    }
    Ok(Some(v[0]))
}

/// Refuses a file carrying any per-tensor weight scale, naming the
/// first few.
///
/// `names` is every tensor in the file; the loader calls this once,
/// before it reads a layer, so the refusal names the feature rather
/// than the unread tensor.
pub fn refuse_weight_scale_tensors<'a>(
    arch: &str,
    names: impl Iterator<Item = &'a str>,
) -> Result<(), LoadError> {
    let mut scaled: Vec<&str> = names
        .filter(|n| SCALE_SUFFIXES.iter().any(|s| n.ends_with(s)) && !is_served(n))
        .collect();
    if scaled.is_empty() {
        return Ok(());
    }
    scaled.sort_unstable();
    let shown = scaled
        .iter()
        .take(4)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let listing = if scaled.len() > 4 {
        format!("{shown}, … (+{} more)", scaled.len() - 4)
    } else {
        shown
    };
    Err(LoadError::UnsupportedFeature(
        arch.to_string(),
        format!(
            "per-tensor weight scales ({} tensor(s): {listing}): llama.cpp multiplies each \
             projection's output by its `.scale` / `.input_scale` companion \
             (`build_lora_mm`, llama-graph.cpp:1492-1494), which ferrox does not apply; \
             running the file without them would put every scaled projection at the \
             wrong magnitude. Re-export without per-tensor scales (the current BitNet \
             converter folds them into the weights)",
            scaled.len()
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two `talkie` writes pass; the others beside them still do not.
    #[test]
    fn the_two_served_companions_pass_and_the_rest_are_still_refused() {
        let ok = [
            "blk.0.attn_output.scale",
            "blk.3.ffn_down.scale",
            "blk.0.attn_q.weight",
        ];
        assert!(refuse_weight_scale_tensors("talkie", ok.into_iter()).is_ok());
        let err = refuse_weight_scale_tensors(
            "talkie",
            ["blk.0.attn_output.scale", "blk.0.attn_q.scale"].into_iter(),
        )
        .expect_err("attn_q.scale is not applied");
        let msg = err.to_string();
        assert!(
            msg.contains("blk.0.attn_q.scale") && !msg.contains("attn_output.scale"),
            "{msg}"
        );
    }

    #[test]
    fn a_file_without_scale_companions_passes() {
        let names = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "output_norm.weight",
        ];
        assert!(refuse_weight_scale_tensors("bitnet", names.into_iter()).is_ok());
    }

    /// Both suffixes, on a layer tensor and on the lm_head, and the
    /// message names the tensor and the feature.
    #[test]
    fn either_suffix_anywhere_is_refused_by_name() {
        for name in [
            "blk.3.ffn_gate.scale",
            "blk.0.attn_q.input_scale",
            "output.scale",
        ] {
            let err =
                refuse_weight_scale_tensors("llama", [name, "blk.0.attn_q.weight"].into_iter())
                    .expect_err(name);
            let msg = err.to_string();
            assert!(msg.contains(name), "{msg}");
            assert!(msg.contains("per-tensor weight scales"), "{msg}");
        }
    }

    /// A weight whose NAME merely contains "scale" is not a scale
    /// companion; only the exact suffixes are.
    #[test]
    fn the_suffix_is_matched_not_the_substring() {
        let names = ["blk.0.attn_scale_gate.weight", "blk.0.scale_norm.weight"];
        assert!(refuse_weight_scale_tensors("llama", names.into_iter()).is_ok());
    }
}
