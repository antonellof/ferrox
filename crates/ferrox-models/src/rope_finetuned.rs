//! `{arch}.rope.scaling.finetuned` as a SWITCH, for the one family that
//! reads it that way.
//!
//! `LLM_KV_ROPE_SCALING_FINETUNED` is a boolean llama.cpp otherwise only
//! prints. Granite is the exception, and says so in a comment:
//!
//! ```text
//! // src/models/granite.cpp:33-35
//! // Granite uses rope_finetuned as a switch for rope, so default to true
//! bool rope_finetuned = true;
//! ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
//! hparams.rope_finetuned = rope_finetuned;
//! ```
//!
//! and the graph then builds no `inp_pos` at all and skips both
//! `ggml_rope_ext` calls when it is false (`granite.cpp:130-133`,
//! `:206-219`). That is the ALiBi class of divergence: a file that
//! declares it would run through ferrox's generic decoder rotating every
//! Q and K head of every layer, fluently, from positions the checkpoint
//! does not encode that way. ferrox has no way to express "this
//! architecture, unrotated", so it stops.
//!
//! **Is this a gate that can fire?** Not from a converter. The only
//! `add_rope_scaling_finetuned` call in `conversion/granite.py` is at
//! :253, inside `GraniteHybridModel`, whose `model_arch` is
//! `GRANITE_HYBRID` -- a different GGUF string, and a `DedicatedOnly`
//! row in ferrox's catalog. `GraniteModel` and `GraniteMoeModel` never
//! write the key, so every converter-produced `granite` / `granitemoe`
//! file takes the `true` default and rotates. The gate is for the file
//! that does carry it, which llama.cpp will happily run unrotated, and
//! `tests/granite_family_graphs.rs` drives one through the loader rather
//! than asserting the refusal exists. That distinction is the whole
//! reason this module has a test: this repo has shipped a refusal keyed
//! on a GGUF spelling nothing writes, and it read as coverage for
//! months.

/// Architectures whose graph builds no positions and applies no RoPE
/// when `{arch}.rope.scaling.finetuned` is false.
///
/// `granitehybrid` is the one architecture whose CONVERTER writes the
/// key, and it is not here on purpose: it resolves to
/// `ArchPath::DedicatedOnly` and never reaches the generic decoder, so
/// listing it would be a second claim about a row this list does not
/// serve.
pub const ROPE_GATED_ON_FINETUNED: &[&str] = &["granite", "granitemoe", "granite-moe"];

/// The refusal reason when `arch` gates its RoPE on the key and the file
/// declares it false, or `None` when the file may be rotated normally.
///
/// `declared` is the file's value, `None` when the key is absent --
/// which is llama.cpp's `true` default for these rows, i.e. rotate.
pub fn unrotated_refusal(arch: &str, declared: Option<bool>) -> Option<String> {
    if !ROPE_GATED_ON_FINETUNED.contains(&arch) || declared != Some(false) {
        return None;
    }
    Some(format!(
        "`{arch}.rope.scaling.finetuned` = false. src/models/granite.cpp:33-35 reads this key \
         as a SWITCH FOR ROPE rather than as a note about the scaling, and :130-133,:206-219 \
         then build no positions and skip both ggml_rope_ext calls, so llama.cpp runs this \
         checkpoint with NO rotation at all. ferrox's generic decoder rotates every Q and K \
         head of every layer and has no per-model way to express `no RoPE`, so it would \
         answer fluently from positions this checkpoint never encodes -- the same failure \
         gpt2, mpt, refact, bloom and jais were caught in. No Granite converter writes this \
         key (conversion/granite.py:253 is GraniteHybridModel, a different architecture \
         string), so a real Granite export never reaches this message"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate fires on exactly the value llama.cpp branches on, and on
    /// nothing else.
    ///
    /// The absent case is the one worth pinning: `None` is llama.cpp's
    /// `true` default for these rows, so treating a missing key as "no
    /// RoPE" would refuse every real Granite checkpoint there is.
    #[test]
    fn only_an_explicit_false_refuses() {
        for arch in ROPE_GATED_ON_FINETUNED {
            assert!(
                unrotated_refusal(arch, Some(false)).is_some(),
                "{arch} must refuse an explicit false"
            );
            assert!(
                unrotated_refusal(arch, Some(true)).is_none(),
                "{arch} rotates when the file says so"
            );
            assert!(
                unrotated_refusal(arch, None).is_none(),
                "{arch} must take llama.cpp's `true` default when the key is absent"
            );
        }
    }

    /// An architecture that merely CARRIES the key is not gated by it.
    ///
    /// llama.cpp reads `rope_finetuned` for every architecture and
    /// branches on it in exactly one graph. Refusing everywhere the key
    /// appears would refuse legacy `llama` files converted by
    /// `examples/convert_legacy_llama.py:915`, which writes it and means
    /// nothing by it.
    #[test]
    fn an_architecture_that_does_not_gate_its_rope_is_unaffected() {
        for arch in ["llama", "qwen3", "granitehybrid"] {
            assert!(
                unrotated_refusal(arch, Some(false)).is_none(),
                "{arch} does not read this key as a RoPE switch"
            );
        }
    }

    /// The message names the key and the llama.cpp line, so a user who
    /// hits it can check the claim rather than take it.
    #[test]
    fn the_refusal_names_the_key_and_the_line_that_decides_it() {
        let msg = unrotated_refusal("granite", Some(false)).expect("refused");
        assert!(msg.contains("granite.rope.scaling.finetuned"), "{msg}");
        assert!(msg.contains("granite.cpp:33-35"), "{msg}");
    }
}
