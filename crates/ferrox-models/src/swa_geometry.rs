//! The two ways a sliding-window layer can differ from a full-attention
//! layer in llama.cpp beyond its mask and its RoPE base, and why ferrox
//! refuses both by name rather than serving one geometry to both kinds.
//!
//! 1. **A second rotary width, and a second head width.** The base
//!    loader seeds `n_rot_swa`, `n_embd_head_k_swa` and
//!    `n_embd_head_v_swa` from the full-attention values and lets three
//!    optional keys override them (`llama-model.cpp:1215-1223`), and
//!    `llama_hparams::n_rot(il)` is `is_swa(il) ? n_rot_swa :
//!    n_rot_full` (`llama-hparams.cpp:85-91`). ferrox carries ONE
//!    `rope_dim` and ONE `head_dim`, read at every rotation site and in
//!    every cache geometry, so a file whose sliding layers rotate a
//!    different width -- `conversion/laguna.py:115` writes it for
//!    Laguna-XS.2 (full layers 64 of 128 dims, sliding layers all 128,
//!    `laguna.cpp:43-45`) and `conversion/gemma.py:673-674,700` for
//!    Gemma-4 -- would be rotated at the full width on every layer.
//!    [`swa_geometry_refusal`] names the key.
//!
//! 2. **YaRN switched off on the sliding layers.** Three graphs on the
//!    generic path rope their sliding layers with `freq_scale = 1`,
//!    `ext_factor = 0`, `attn_factor = 1` while their full layers use
//!    the model's scaling: `olmo2.cpp:120-134` (Olmo-3), `mellum.cpp:
//!    128-142`, `laguna.cpp:48,181-193`. Measured by grepping every
//!    `ggml_rope_ext` call under `src/models/` for a zeroed
//!    `ext_factor`; `deepseek4.cpp` is the fourth and is on its own
//!    engine. ferrox folds the frequency half of YaRN into
//!    [`crate::config::RopeFreqs`], which already keeps the sliding
//!    layers unscaled for these three (`swa_rope_scale_follows_model`
//!    is false for all of them), but `rope_attn_factor` -- YaRN's
//!    `mscale` -- is one value for the whole model, so a file with a
//!    window AND a scaling is refused. A window with no scaling, or a
//!    scaling with no window, reduces both branches to the same RoPE
//!    and is served. [`swa_layers_unscaled_rope`] is the table; it used
//!    to be `arch == "olmo2"` in the loader, one row of three.

/// The llama.cpp lines where `arch` ropes its sliding layers with the
/// scaling switched off, or `None` when its sliding layers inherit the
/// model's scaling like everyone else's.
pub fn swa_layers_unscaled_rope(arch: &str) -> Option<&'static str> {
    match arch {
        "olmo2" => Some("olmo2.cpp:120-134"),
        "mellum" => Some("mellum.cpp:128-142"),
        "laguna" => Some("laguna.cpp:48,181-193"),
        _ => None,
    }
}

/// The three SWA-geometry keys as the file declares them, beside the
/// full-attention values they would override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwaGeometry {
    /// `{arch}.rope.dimension_count_swa`.
    pub rope_dim_swa: Option<u64>,
    /// `{arch}.attention.key_length_swa`.
    pub key_length_swa: Option<u64>,
    /// `{arch}.attention.value_length_swa`.
    pub value_length_swa: Option<u64>,
    /// The full-attention rotary width, `rope.dimension_count` or the
    /// head width when the key is absent -- what `n_rot_swa` defaults
    /// to (`llama-model.cpp:1222`).
    pub rope_dim_full: u64,
    /// The full-attention head width, what both `_swa` head keys
    /// default to (`llama-model.cpp:1216-1217`).
    pub head_dim: u64,
}

/// Why this file cannot be served, or `None` when every declared key
/// restates the full-attention value (which llama.cpp treats exactly as
/// the key being absent).
///
/// Only meaningful for a model with at least one sliding layer; the
/// caller checks that, because a key nothing reads is not a hazard.
pub fn swa_geometry_refusal(arch: &str, g: SwaGeometry) -> Option<String> {
    let differing = [
        ("rope.dimension_count_swa", g.rope_dim_swa, g.rope_dim_full),
        ("attention.key_length_swa", g.key_length_swa, g.head_dim),
        ("attention.value_length_swa", g.value_length_swa, g.head_dim),
    ]
    .into_iter()
    .find_map(|(key, declared, full)| declared.filter(|v| *v != full).map(|v| (key, v, full)))?;
    let (key, declared, full) = differing;
    Some(format!(
        "`{arch}.{key}` = {declared} while the full-attention layers use {full}: llama.cpp \
         gives the sliding layers their own rotary and head widths (llama-model.cpp:1215-1223, \
         `n_rot(il)` at llama-hparams.cpp:85-91) and ferrox carries one of each for the whole \
         model, so honouring the file would rotate or cache half the layers at the wrong width"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> SwaGeometry {
        SwaGeometry {
            rope_dim_swa: None,
            key_length_swa: None,
            value_length_swa: None,
            rope_dim_full: 64,
            head_dim: 128,
        }
    }

    /// A key restating the full value is what an absent key means to
    /// llama.cpp, so it is served; a key that differs is refused naming
    /// that key and both numbers.
    #[test]
    fn a_swa_width_equal_to_the_full_one_is_served_and_a_different_one_names_the_key() {
        assert_eq!(swa_geometry_refusal("laguna", geometry()), None);
        let mut same = geometry();
        same.rope_dim_swa = Some(64);
        same.key_length_swa = Some(128);
        same.value_length_swa = Some(128);
        assert_eq!(swa_geometry_refusal("laguna", same), None);

        let mut rot = geometry();
        rot.rope_dim_swa = Some(128);
        let msg = swa_geometry_refusal("laguna", rot).expect("refused");
        assert!(
            msg.contains("`laguna.rope.dimension_count_swa` = 128"),
            "{msg}"
        );
        assert!(msg.contains("use 64"), "{msg}");

        let mut kv = geometry();
        kv.value_length_swa = Some(256);
        let msg = swa_geometry_refusal("gemma4", kv).expect("refused");
        assert!(msg.contains("value_length_swa"), "{msg}");
    }

    /// The three rows that zero YaRN on their sliding layers, each
    /// citing the lines, and a plain-SWA architecture answering `None`.
    #[test]
    fn the_unscaled_swa_table_cites_lines_for_exactly_the_three_generic_rows() {
        for arch in ["olmo2", "mellum", "laguna"] {
            let lines = swa_layers_unscaled_rope(arch).unwrap_or_else(|| panic!("{arch}"));
            assert!(lines.contains(".cpp:"), "{arch}: {lines}");
        }
        for arch in ["gemma3", "gpt-oss", "exaone4", "afmoe", "llama"] {
            assert_eq!(swa_layers_unscaled_rope(arch), None, "{arch}");
        }
    }
}
