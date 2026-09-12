//! **A V HEAD WIDTH THAT DIFFERS FROM THE K HEAD WIDTH** -- llama.cpp's
//! `n_embd_head_k` / `n_embd_head_v` pair, as one resolution the loader
//! makes and one table that says which architecture may declare them
//! apart.
//!
//! # What it is
//!
//! llama.cpp reads `attention.key_length` and `attention.value_length`
//! for every architecture (`llama-model.cpp:1195-1200`), sizes K rows
//! by the first and V rows by the second (`n_embd_k_gqa` /
//! `n_embd_v_gqa`), scores with the first (`1/sqrt(n_embd_head_k)`),
//! and reads `wo` as `{n_embd_head_v * n_head, n_embd}`. Eighty-nine of
//! the 140 graphs then assert the two EQUAL (`GGML_ASSERT(n_embd_head
//! == hparams.n_embd_head_v())`, measured 2026-09-12); the rest use
//! both names and would run either way.
//!
//! # Who declares them apart -- MEASURED
//!
//! `grep -n add_value_length conversion/*.py`: fourteen converters
//! write the key, and all but three write it from the SAME `head_dim`
//! they wrote `key_length` from. The three: `deepseek.py` and `plm.py`
//! (MLA, `qk_head_dim != v_head_dim`, on the MLA engine, which has
//! carried the pair since it existed); and `mimo.py:154`, which writes
//! `v_head_dim` -- `head_dim: 192, v_head_dim: 128` on MiMo-V2-Flash,
//! the same on V2.5 -- for the generic-path graph `mimo2.cpp`, whose
//! `:47-48,132-140,152-154` size and view K and V separately and whose
//! `wo` is `{n_embd_head_v * n_head, n_embd}` (`:52`). So
//! [`SPLIT_KV_HEAD_DIM_ARCHS`] has one row.
//!
//! # What ferrox does with it
//!
//! `ModelConfig::v_head_dim` is the resolved width and every consumer
//! reads it: `KvCache` / `PagedKvStore` size and index V by it
//! (`new_split`), `causal_gqa_attention_row` and the batched prefill
//! kernel accumulate over it, `check_gqa_projection_widths` sizes
//! `v_proj` and `o_proj` by it, and `qkv_fused::FusedQkvRows` cuts the
//! fused `attn_qkv` at it. A file whose two keys differ on an
//! architecture NOT in the table is refused here, by name, as before:
//! its graph would `GGML_ASSERT` upstream, and running it would be a
//! guess about a shape llama.cpp itself does not run.
//!
//! Every fused Metal launch takes ONE head width -- the KV buffers, the
//! attention kernel's tile, the `wo` fold -- so `metal_can_serve_model`
//! refuses a split model, the CUDA resident KV refuses it, and the KV
//! block-file format (one `head_dim` in its header) refuses to stamp
//! it (`ferrox_core::kv_signature::SignatureError::SplitKvHeadWidth`).

use crate::LoadError;

/// Architectures whose graph sizes K and V heads separately AND whose
/// converter writes the two keys apart, with the lines.
pub const SPLIT_KV_HEAD_DIM_ARCHS: &[(&str, &str)] = &[(
    "mimo2",
    "src/models/mimo2.cpp:47-48,52,132-140,152-154; conversion/mimo.py:154",
)];

/// Whether this architecture may declare `attention.value_length`
/// different from `attention.key_length`.
pub fn admits_split_kv_head_dims(arch: &str) -> bool {
    SPLIT_KV_HEAD_DIM_ARCHS
        .iter()
        .any(|(name, _)| *name == arch)
}

/// The V head width for a file: `attention.value_length` when present,
/// else the K width -- refused when the two differ on an architecture
/// whose graph asserts them equal.
pub fn resolve_v_head_dim(
    arch: &str,
    head_dim: usize,
    value_length: Option<usize>,
) -> Result<usize, LoadError> {
    let v_head_dim = value_length.unwrap_or(head_dim);
    if v_head_dim == head_dim || admits_split_kv_head_dims(arch) {
        if v_head_dim == 0 {
            return Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                "attention.value_length is 0".to_string(),
            ));
        }
        return Ok(v_head_dim);
    }
    Err(LoadError::UnsupportedFeature(
        arch.to_string(),
        format!(
            "split K/V head dims (key_length={head_dim}, value_length={v_head_dim}): \
             llama.cpp's `{arch}` graph asserts the two equal (`n_embd_head_k() == \
             n_embd_head_v()`), so a file declaring them apart runs there no more than \
             here; only `mimo2` sizes K and V heads separately on the generic path \
             (`ferrox_models::kv_head_dims`)"
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_widths_pass_for_everyone_and_a_missing_key_means_equal() {
        for arch in ["llama", "qwen3", "mimo2", "gemma3"] {
            assert_eq!(resolve_v_head_dim(arch, 128, None).unwrap(), 128, "{arch}");
            assert_eq!(
                resolve_v_head_dim(arch, 128, Some(128)).unwrap(),
                128,
                "{arch}"
            );
        }
    }

    /// The one row takes MiMo-V2-Flash's real pair; a Llama declaring
    /// the same pair is refused naming the assert upstream.
    #[test]
    fn only_the_table_admits_a_differing_value_length() {
        assert_eq!(resolve_v_head_dim("mimo2", 192, Some(128)).unwrap(), 128);
        let err = resolve_v_head_dim("llama", 192, Some(128)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("split K/V head dims"), "{msg}");
        assert!(msg.contains("key_length=192, value_length=128"), "{msg}");
        assert!(msg.contains("n_embd_head_v()"), "{msg}");
    }

    #[test]
    fn a_zero_value_length_is_refused_rather_than_sizing_empty_heads() {
        assert!(resolve_v_head_dim("mimo2", 192, Some(0)).is_err());
    }

    /// Every table row is a generic-path architecture, and audited: the
    /// seam is asked by something and evidenced by something.
    #[test]
    fn every_table_row_is_on_the_generic_path() {
        for (arch, line) in SPLIT_KV_HEAD_DIM_ARCHS {
            let profile = crate::capability::resolve_profile(arch)
                .unwrap_or_else(|| panic!("`{arch}` ({line}) is not a registered architecture"));
            assert!(
                matches!(profile.path, crate::capability::ArchPath::GenericGqa { .. }),
                "`{arch}` ({line}) is {:?}",
                profile.path
            );
            assert!(
                crate::capability::AUDITED_GENERIC_GQA.contains(arch),
                "`{arch}` is served here and must be audited, or the seam is unevidenced"
            );
        }
    }
}
