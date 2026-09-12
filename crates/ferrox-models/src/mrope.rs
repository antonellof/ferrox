//! **`rope.dimension_sections` ON THE GENERIC PATH** -- what llama.cpp
//! does with M-RoPE sections on a text tower, per architecture, and what
//! that means for a rotation ferrox decides per architecture.
//!
//! # What it is
//!
//! `llama_hparams::use_mrope()` is `rope_sections[0] > 0 &&
//! rope_sections[1] > 0` (`llama-hparams.cpp:284-286`). The multimodal
//! converters write the sections for the text tower of a vision export
//! (`conversion/glm.py:26-27`), and `llama_model_rope_type` then answers
//! `LLAMA_ROPE_TYPE_MROPE` for the two GLM graphs instead of their text
//! layout (`llama-model.cpp:2698-2701`). With TEXT positions -- one
//! position for every M-RoPE component -- `ggml_rope_multi` in MROPE
//! mode rotates band `i` against band `i + n_dims/2` at angle `pos *
//! freq_i`, which is NEOX rotation band for band. So:
//!
//! - `glm4moe` is NEOX without sections and NEOX with them; libllama's
//!   logits on `tests/fixtures/glm4moe_mrope_tiny.gguf` are byte for
//!   byte the plain file's (measured, `tests/glm4moe_graphs.rs`). SERVED.
//! - `glm4` is NORM without sections (`:2699`), and the converter
//!   PERMUTES a sectioned file's Q/K weights to NEOX order
//!   (`glm.py:53-73,78-85`) because the M-RoPE kernel only speaks
//!   NEOX; libllama's logits on `tests/fixtures/glm4_mrope_tiny.gguf`
//!   differ from the plain file's by 0.72 (measured). A ferrox rope
//!   layout is a property of the architecture (`ArchPath::GenericGqa {
//!   rope }`), not of the file, so this file is REFUSED by name.
//!
//! # Reach -- MEASURED
//!
//! `grep -l ROPE_DIMENSION_SECTIONS src/models/*.cpp` over all 140
//! graphs is eleven files. On the generic path: `glm4` and `glm4moe`
//! (above) and `ernie4-5.cpp:5`, which reads the sections into hparams
//! and rotates with `ggml_rope_ext` unconditionally (`llama-model.cpp:
//! 2602` is an unconditional NORM arm), so for ERNIE they are dead
//! metadata and nothing here applies. The other eight are on other
//! engines or refused.

use ferrox_gguf::{GgufValue, TensorSource};

/// What the generic path does with a file whose sections declare
/// M-RoPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MropeOnText {
    /// The architecture's text layout is already NEOX, which is what
    /// M-RoPE computes on text positions: served, and the identity is
    /// pinned against libllama.
    SameAsNeox,
    /// The architecture's text layout is NORM and the converter
    /// permuted the file to NEOX: refused, with the lines.
    RefusedNormBase,
}

/// The two generic-path graphs whose rotation `use_mrope()` switches.
pub const MROPE_READERS: &[(&str, MropeOnText, &str)] = &[
    (
        "glm4moe",
        MropeOnText::SameAsNeox,
        "src/models/glm4-moe.cpp:6,145,188; llama-model.cpp:2700",
    ),
    (
        "glm4",
        MropeOnText::RefusedNormBase,
        "src/models/glm4.cpp:5,112-119; llama-model.cpp:2699; conversion/glm.py:53-85",
    ),
];

/// `llama_hparams::use_mrope()` for a file.
pub fn declares_mrope(file: &impl TensorSource, arch: &str) -> bool {
    match file.metadata(&format!("{arch}.rope.dimension_sections")) {
        Some(GgufValue::Array(items)) => {
            let at = |i: usize| items.get(i).and_then(GgufValue::as_u64).unwrap_or(0);
            at(0) > 0 && at(1) > 0
        }
        _ => false,
    }
}

/// The refusal reason for a file that declares M-RoPE on an
/// architecture whose text rotation is not what M-RoPE computes, or
/// `None` when the file may be run.
pub fn mrope_refusal(file: &impl TensorSource, arch: &str) -> Option<String> {
    let (_, what, lines) = MROPE_READERS.iter().find(|(n, _, _)| *n == arch)?;
    if *what != MropeOnText::RefusedNormBase || !declares_mrope(file, arch) {
        return None;
    }
    Some(format!(
        "`{arch}.rope.dimension_sections` declares M-RoPE (a vision export's text tower): \
         llama.cpp rotates this file with LLAMA_ROPE_TYPE_MROPE over Q/K weights the converter \
         permuted to NEOX order, where the text-only `{arch}` rotates NORM ({lines}). ferrox \
         decides the rotation per architecture, so it stops rather than rotate the wrong pairs \
         of every head; libllama's logits for such a file differ from the unpermuted file's by \
         0.72 (measured on tests/fixtures/glm4_mrope_tiny.gguf)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_reader_is_a_generic_row_with_the_matching_text_layout() {
        for (arch, what, line) in MROPE_READERS {
            let profile = crate::capability::resolve_profile(arch)
                .unwrap_or_else(|| panic!("`{arch}` ({line}) is not a registered architecture"));
            let crate::capability::ArchPath::GenericGqa { rope } = profile.path else {
                panic!("`{arch}` is not on the generic path");
            };
            // The decision follows from the text layout, and the table
            // must say the same thing the profile does.
            match what {
                MropeOnText::SameAsNeox => {
                    assert_eq!(rope, crate::config::RopeLayout::Neox, "{arch}")
                }
                MropeOnText::RefusedNormBase => {
                    assert_eq!(rope, crate::config::RopeLayout::Norm, "{arch}")
                }
            }
            assert!(
                crate::capability::AUDITED_GENERIC_GQA.contains(arch),
                "{arch}"
            );
        }
    }
}
