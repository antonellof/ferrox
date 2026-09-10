//! `{arch}.attention.clamp_kqv`, for the architectures whose graph
//! applies it.
//!
//! `LLM_KV_ATTENTION_CLAMP_KQV` (`llama-arch.cpp:231`) is a symmetric
//! clamp on the Q, K and V projections, applied INSIDE the shared
//! `build_qkv` rather than in any one architecture's graph:
//!
//! ```text
//! // llama-graph.cpp:1611-1612, and again at :1631, :1641, :1651
//! if (hparams.f_clamp_kqv > 0.0f) {
//!     qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
//! }
//! ```
//!
//! The fused branch clamps the fused projection once; the split branch
//! clamps `Qcur`, `Kcur` and `Vcur` separately. Either way it happens
//! before the reshape, before any QK-norm and before RoPE.
//!
//! **Three architectures read the key**, and no other: `dbrx.cpp:5`
//! (REQUIRED), `mpt.cpp:5` and `olmo.cpp:5` (both optional). `mpt` and
//! `dbrx` do not reach the generic decoder for other reasons -- ALiBi
//! and a LayerNorm-plus-`attn_output_norm` graph respectively -- so
//! `olmo` is the only row this gate is live for today, and it is live
//! rather than theoretical: `conversion/olmo.py:23-25` writes the key
//! whenever the HF config has a `clip_qkv`, which OLMo-7B-Twin-2T and
//! OLMo-1.7-7B do (`8.0`) and the original OLMo-7B does not (`null`).
//!
//! **ferrox has no clamp on any projection.** Implementing one is not
//! three lines: the projections are computed in the CPU prefill body,
//! the decode body, the continuous-batching body and the fused Metal
//! launches, and a feature added to some of those and not the others is
//! this repo's single most expensive defect shape -- eight model
//! features have been lost that way, one at a time. So a file that
//! declares a positive clamp STOPS here. A file that omits the key, or
//! declares `0.0`, is llama.cpp's own "no clamp" and runs.

/// Architectures whose graph clamps Q, K and V by
/// `{arch}.attention.clamp_kqv`.
///
/// `mpt` and `dbrx` are deliberately absent even though they read the
/// key: both resolve to `ArchPath::DedicatedOnly` and never reach the
/// generic decoder, so listing them would be a second claim about a row
/// this list does not serve. That is the same rule
/// [`crate::rope_finetuned::ROPE_GATED_ON_FINETUNED`] follows for
/// `granitehybrid`.
pub const CLAMPED_QKV_ARCHITECTURES: &[&str] = &["olmo"];

/// The refusal reason when `arch` clamps its projections and the file
/// declares a positive clamp, or `None` when the file may be run.
///
/// `declared` is the file's value, `None` when the key is absent --
/// llama.cpp's `0.0f` default, i.e. no clamp.
pub fn clamped_refusal(arch: &str, declared: Option<f32>) -> Option<String> {
    if !CLAMPED_QKV_ARCHITECTURES.contains(&arch) {
        return None;
    }
    // `> 0.0f` is llama.cpp's own test (llama-graph.cpp:1611), so zero
    // and a negative value both mean "no clamp" there and must mean it
    // here. Reading zero as a clamp would zero every projection.
    let v = declared.filter(|v| *v > 0.0)?;
    Some(format!(
        "`{arch}.attention.clamp_kqv` = {v}. llama-graph.cpp:1611-1652 clamps the Q, K and V \
         projections to [-{v}, {v}] inside `build_qkv`, before the reshape, the QK-norm and \
         RoPE. ferrox has no clamp on any projection, and adding one to the CPU prefill body \
         while missing the decode body or a fused Metal launch is the defect shape that has \
         already cost this engine eight model features one at a time, so it stops instead. \
         `conversion/olmo.py:23-25` writes this key whenever the HF config carries a \
         `clip_qkv`: OLMo-7B-Twin-2T and OLMo-1.7-7B do, the original OLMo-7B does not"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate fires on exactly the values llama.cpp branches on.
    ///
    /// The absent and zero cases are the ones worth pinning: llama.cpp's
    /// default is `0.0f` and its test is `> 0.0f`, so treating either as
    /// a clamp would refuse the original OLMo-7B, whose `clip_qkv` is
    /// null and whose converted file carries no key at all.
    #[test]
    fn only_a_positive_clamp_refuses() {
        assert!(clamped_refusal("olmo", Some(8.0)).is_some());
        assert!(
            clamped_refusal("olmo", None).is_none(),
            "an absent key is llama.cpp's 0.0 default: no clamp"
        );
        assert!(
            clamped_refusal("olmo", Some(0.0)).is_none(),
            "`> 0.0f` is llama.cpp's own test; zero means no clamp"
        );
        assert!(
            clamped_refusal("olmo", Some(-1.0)).is_none(),
            "a negative clamp is not a clamp in llama.cpp either"
        );
    }

    /// An architecture that does not clamp is unaffected, including the
    /// two that read the key and never reach this decoder.
    #[test]
    fn an_architecture_whose_graph_does_not_clamp_is_unaffected() {
        for arch in ["llama", "qwen3", "olmo2", "mpt", "dbrx"] {
            assert!(
                clamped_refusal(arch, Some(8.0)).is_none(),
                "{arch} must not be gated by this key here"
            );
        }
    }

    /// The message names the key, the value and the llama.cpp line, so
    /// a user who hits it can check the claim rather than take it.
    #[test]
    fn the_refusal_names_the_key_the_value_and_the_line() {
        let msg = clamped_refusal("olmo", Some(8.0)).expect("refused");
        assert!(msg.contains("olmo.attention.clamp_kqv"), "{msg}");
        assert!(msg.contains("llama-graph.cpp:1611-1652"), "{msg}");
        assert!(msg.contains('8'), "{msg}");
    }
}
