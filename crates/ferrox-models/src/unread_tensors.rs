//! Tensors llama.cpp CREATES for an architecture and then never reads
//! in its graph, so a file carrying them is served with them ignored --
//! exactly as upstream serves it -- rather than refused as "a term the
//! graph is missing".
//!
//! `assert_every_tensor_consumed` is the loader's last gate, and it is
//! right to be strict: an unread tensor is usually a feature this
//! engine has not implemented. But llama.cpp itself has a few
//! `create_tensor(..., TENSOR_NOT_REQUIRED)` slots that no line of the
//! graph consumes, and for those the honest answer is the one upstream
//! gives: load, ignore, and say so here. Each row below names the line
//! that creates the tensor, the line that would have read it and
//! passes `NULL` instead, and the fixture that MEASURED libllama's
//! logits byte-identical with and without it.
//!
//! This is not [`crate::mtp_blocks`]: those are whole blocks llama.cpp
//! skips by design (`TENSOR_SKIP`). These are single slots inside a
//! served layer.

use ferrox_gguf::ShardedGguf;

/// Per-architecture: the per-layer tensor suffixes (after `blk.N.`)
/// that llama.cpp creates and never reads, with the evidence.
///
/// * `apertus`: `apertus.cpp:50,52` create `attn_q_norm.bias` and
///   `attn_k_norm.bias` (`TENSOR_NOT_REQUIRED`); `:93,96` call
///   `build_norm(Qcur, attn_q_norm, NULL, LLM_NORM_RMS, il)` -- the
///   bias argument is the literal `NULL`, so the tensors are loaded
///   and never enter the graph. `conversion/llama.py:424-457` never
///   writes them (the checkpoint's QK-norms are RMSNorm, which has no
///   bias), so only a hand-written file can carry them.
///   `tests/fixtures/apertus_qknorm_bias_tiny.gguf` does, and
///   libllama's logits for it are byte-identical to the base file's
///   (`tests/per_layer_activation_graphs.rs`).
pub const UNREAD_LAYER_TENSORS: &[(&str, &[&str])] =
    &[("apertus", &["attn_q_norm.bias", "attn_k_norm.bias"])];

/// The suffixes llama.cpp leaves unread for `arch`, or an empty slice.
pub fn unread_layer_tensors(arch: &str) -> &'static [&'static str] {
    UNREAD_LAYER_TENSORS
        .iter()
        .find(|(a, _)| *a == arch)
        .map(|(_, t)| *t)
        .unwrap_or(&[])
}

/// Marks every such tensor the file carries as deliberately unread, and
/// returns their names so the caller can log what was ignored.
pub fn note_unread_layer_tensors(file: &ShardedGguf, arch: &str, n_layers: usize) -> Vec<String> {
    let mut noted = Vec::new();
    for suffix in unread_layer_tensors(arch) {
        for l in 0..n_layers {
            let name = format!("blk.{l}.{suffix}");
            // `find_tensor` already records the name as consumed; the
            // explicit note is here so the intent survives a change to
            // that accounting.
            if file.find_tensor(&name).is_some() {
                file.note_consumed(&name);
                noted.push(name);
            }
        }
    }
    noted
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table answers for the one row it has and for nobody else,
    /// and every suffix is a `.bias` -- a weight llama.cpp ignored
    /// would be a different finding.
    #[test]
    fn only_apertus_has_unread_slots_and_they_are_the_two_qk_norm_biases() {
        assert_eq!(
            unread_layer_tensors("apertus"),
            ["attn_q_norm.bias", "attn_k_norm.bias"]
        );
        for arch in ["llama", "qwen3", "arcee", "olmo2", "step35"] {
            assert!(unread_layer_tensors(arch).is_empty(), "{arch}");
        }
        for (_, suffixes) in UNREAD_LAYER_TENSORS {
            for s in *suffixes {
                assert!(s.ends_with(".bias"), "{s}");
            }
        }
    }
}
