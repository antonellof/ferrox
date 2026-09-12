//! **YaRN ON THE MLA ENGINE** -- the frequency rewrite of the `pe`
//! slices, the magnitude on them, and the `mscale^2` folded into the
//! softmax scale, as `src/models/deepseek2.cpp:312-328,438-448` compute
//! them, so that a real DeepSeek-V2 / V3 export (every one declares
//! YaRN) runs at its declared factor instead of being refused.
//!
//! # What llama.cpp does, in order
//!
//! Three places, and all three matter:
//!
//! 1. `deepseek2.cpp:34-37` read `rope.scaling.yarn_log_multiplier`
//!    and DIVIDE it by 0.1, because `conversion/deepseek.py:363-368`
//!    writes `0.1 * mscale_all_dim` ("for legacy reasons"); so
//!    `hparams.rope_yarn_log_mul` is `mscale_all_dim` itself. Only
//!    this loader (which `mistral4` shares) reads the key; `plm`'s does
//!    not, and the generic path's one reader is `mistral3`
//!    (`crate::yarn_magnitude::YARN_LOG_MUL_READERS`).
//! 2. `llama-context.cpp:194-231` compute the `attn_factor` handed to
//!    `ggml_rope_ext`: with `L = rope_yarn_log_mul != 0`,
//!    `get_mscale(F, m) / get_mscale(F, L)` where `m = L` for
//!    `LLM_ARCH_DEEPSEEK2` when `L != 1` and `1` otherwise (`:202-215`,
//!    the DeepSeek-V2 special case: its config has `mscale ==
//!    mscale_all_dim == 0.707`); with `L == 0`, `get_mscale(F, 1)`;
//!    then `/= (1 + 0.1 ln F)` to cancel the term ggml's `rope_yarn`
//!    multiplies back in (`:223-228`); then `*= rope.scaling.attn_factor`
//!    (`:231`).
//! 3. `deepseek2.cpp:438-448` undo the cancel to get `attn_factor_org`
//!    (what the `pe` slices are really scaled by after `ggml_rope_ext`),
//!    take `mscale = attn_factor_org * (1 + 0.1 L ln F)`, and set
//!    `kq_scale = mscale^2 / sqrt(n_embd_head_k_mla)`.
//!
//! So the observable pieces are: per-band divisors on the `qk_rope`
//! bands (`ferrox_core::attention::yarn_freq_factors`, the same
//! function the generic path's goldens hold to 1e-14), ONE magnitude
//! on `q_pe` and `k_pe` -- `attn_factor_org` -- and ONE softmax scale.
//! [`MlaYarn`] is exactly those three, resolved once at load from the
//! keys, and [`crate::mla::mla_forward_token`] takes it as an argument
//! so no rotation site and no attention body can be reached without
//! it answered. For the real files the magnitude comes out as
//! `rope.scaling.attn_factor` (1.0): with `m == L` the ratio in step 2
//! is 1, and DeepSeek-V3 has `L == 1`, so `get_mscale(F,1)/get_mscale(F,1)`
//! is 1 too; what is NOT 1 is `kq_scale`, `(1 + 0.1 L ln F)^2 / sqrt(d)`,
//! which is what every real DeepSeek was missing here.
//!
//! # What is refused
//!
//! A `rope.scaling.type` that is neither absent, `none` nor `yarn`
//! (`linear` on an MLA graph has no caller), and a `yarn` without
//! `rope.scaling.original_context_length` -- llama.cpp would floor its
//! ramp on `context_length` (the EXTENDED length, `llama-model.cpp:
//! 1164-1165`), which every converter avoids by writing the key
//! (`conversion/base.py:1235`), and the generic path declines to guess
//! at too.

use ferrox_core::attention::yarn_freq_factors;
use ferrox_gguf::TensorSource;

use crate::LoadError;

/// The three YaRN quantities an MLA layer's attention needs.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaYarn {
    /// One divisor per rotation band of the `qk_rope` slice
    /// (`qk_rope / 2` entries): ggml's `rope_yarn` ramp between the
    /// interpolated and extrapolated angle, in the per-band form the
    /// generic path uses.
    pub freq_factors: Vec<f32>,
    /// `attn_factor_org`: what `q_pe` and `k_pe` are multiplied by
    /// after rotation (`deepseek2.cpp:440-444`).
    pub pe_magnitude: f32,
    /// `mscale^2 / sqrt(qk_nope + qk_rope)` (`deepseek2.cpp:446-448`).
    pub kq_scale: f32,
}

/// Whether an architecture's loader reads `yarn_log_multiplier`, and
/// whether it takes llama.cpp's DeepSeek-V2 `mscale == mscale_all_dim`
/// rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YarnLogMul {
    /// Not read: `rope_yarn_log_mul` stays 0 (`plm`).
    NotRead,
    /// `deepseek2.cpp:34-37`, with `llama-context.cpp:210-213`'s
    /// special case when `deepseek2_mscale_rule` is set (`deepseek2`
    /// itself; `mistral4` reads the key through the same loader but
    /// is not `LLM_ARCH_DEEPSEEK2`).
    Read { deepseek2_mscale_rule: bool },
}

/// `llama-context.cpp:195-197`, in `f32` as it is there.
fn get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// The three quantities for a file, or `None` when it declares no
/// scaling. `rope_attn_factor` is `rope.scaling.attn_factor` (1.0 when
/// absent); `log_mul_key` is the raw `rope.scaling.yarn_log_multiplier`
/// as written.
pub fn resolve_mla_yarn(
    file: &impl TensorSource,
    arch: &str,
    log_mul: YarnLogMul,
    qk_rope_head_dim: usize,
    qk_head_dim: usize,
    rope_theta: f32,
) -> Result<Option<MlaYarn>, LoadError> {
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let Some(kind) = file
        .metadata_str(&key("rope.scaling.type"))
        .filter(|k| !k.eq_ignore_ascii_case("none"))
    else {
        return Ok(None);
    };
    if !kind.eq_ignore_ascii_case("yarn") {
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!(
                "`{arch}.rope.scaling.type = \"{kind}\"`: the MLA engine implements YaRN \
                 (`crate::mla_yarn`) and no other scaling; no MLA converter writes one"
            ),
        ));
    }
    let orig_ctx = file
        .metadata_u64(&key("rope.scaling.original_context_length"))
        .map(|v| v as usize);
    if orig_ctx.is_none() {
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!(
                "`{arch}.rope.scaling.type = yarn` without \
                 `rope.scaling.original_context_length`: llama.cpp would floor the YaRN ramp \
                 on `context_length`, the EXTENDED length (llama-model.cpp:1164-1165); every \
                 converter writes the key (conversion/base.py:1235) and ferrox does not guess it"
            ),
        ));
    }
    let Some(scaling) = crate::loader::yarn_scaling_from_gguf(file, arch, orig_ctx) else {
        // `yarn` with a factor at or below 1: llama.cpp's `get_mscale`
        // floors at 1 and its ramp is the identity, so the file is a
        // plain one.
        return Ok(None);
    };
    if qk_rope_head_dim == 0 || !qk_rope_head_dim.is_multiple_of(2) {
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!("YaRN on a `pe` slice of width {qk_rope_head_dim}, which has no whole bands"),
        ));
    }
    let f = scaling.factor;
    // Step 1: `hparams.rope_yarn_log_mul`.
    let l = match log_mul {
        YarnLogMul::NotRead => 0.0,
        YarnLogMul::Read { .. } => file
            .metadata_f32(&key("rope.scaling.yarn_log_multiplier"))
            .map(|v| v / 0.1)
            .unwrap_or(0.0),
    };
    // Step 2: `cparams.yarn_attn_factor`, before the cancel.
    let rope_attn_factor = file
        .metadata_f32(&key("rope.scaling.attn_factor"))
        .unwrap_or(1.0);
    let pre_cancel = if l != 0.0 {
        let m = match log_mul {
            YarnLogMul::Read {
                deepseek2_mscale_rule: true,
            } if l != 1.0 => l,
            _ => 1.0,
        };
        get_mscale(f, m) / get_mscale(f, l)
    } else {
        get_mscale(f, 1.0)
    };
    // Step 3: the cancel and its undo are `* 1/(1 + 0.1 ln F)` then
    // `* (1 + 0.1 ln F)` in `f32`, and they do not cancel bit-exactly;
    // both are kept so the magnitude is llama.cpp's to the last bit.
    let cancel = 1.0 / (1.0 + 0.1 * f.ln());
    let attn_factor = pre_cancel * cancel * rope_attn_factor;
    let attn_factor_org = attn_factor * (1.0 + 0.1 * (1.0 / (1.0 / f)).ln());
    let mscale = attn_factor_org * (1.0 + 0.1 * l * (1.0 / (1.0 / f)).ln());
    let kq_scale = 1.0 * mscale * mscale / (qk_head_dim as f32).sqrt();
    Ok(Some(MlaYarn {
        freq_factors: yarn_freq_factors(scaling, qk_rope_head_dim, rope_theta),
        pe_magnitude: attn_factor_org,
        kq_scale,
    }))
}

/// The softmax scale for a layer with or without YaRN: `kq_scale`
/// when scaled, `1/sqrt(qk_nope + qk_rope)` otherwise
/// (`deepseek2.cpp:446-448`; every `mscale` term is 1 without it).
pub fn kq_scale(yarn: Option<&MlaYarn>, qk_head_dim: usize) -> f32 {
    match yarn {
        Some(y) => y.kq_scale,
        None => 1.0 / (qk_head_dim as f32).sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The DeepSeek-V2 shape (`mscale == mscale_all_dim == 0.707`):
    /// step 2's ratio is exactly 1, so the `pe` magnitude is
    /// `rope.scaling.attn_factor` and the whole effect is in
    /// `kq_scale = (1 + 0.1 * 0.707 * ln F)^2 / sqrt(d)`.
    #[test]
    fn the_v2_shape_puts_everything_in_kq_scale() {
        let f = 40.0f32;
        let l = 0.707f32;
        let m = l; // the DEEPSEEK2 rule
        let pre = get_mscale(f, m) / get_mscale(f, l);
        assert_eq!(pre, 1.0);
        let attn_factor_org = pre * (1.0 / (1.0 + 0.1 * f.ln())) * (1.0 + 0.1 * f.ln());
        assert!((attn_factor_org - 1.0).abs() < 1e-6);
        let mscale = attn_factor_org * (1.0 + 0.1 * l * f.ln());
        let expect = (1.0 + 0.1 * 0.707 * 40f32.ln()).powi(2) / (192f32).sqrt();
        assert!((mscale * mscale / 192f32.sqrt() - expect).abs() < 1e-6);
    }

    /// `mistral4` reads the key but is not `LLM_ARCH_DEEPSEEK2`, so with
    /// `L != 1` its step-2 ratio is NOT 1: the `pe` magnitude differs
    /// from `deepseek2`'s on the same file. The table column exists
    /// because of this line.
    #[test]
    fn the_mscale_rule_is_deepseek2s_alone() {
        let (f, l) = (4.0f32, 0.5f32);
        let ds2 = get_mscale(f, l) / get_mscale(f, l);
        let m4 = get_mscale(f, 1.0) / get_mscale(f, l);
        assert_eq!(ds2, 1.0);
        assert!((m4 - 1.0).abs() > 1e-3, "{m4}");
    }

    #[test]
    fn no_scaling_is_the_plain_scale() {
        assert_eq!(kq_scale(None, 16), 0.25);
        let y = MlaYarn {
            freq_factors: vec![1.0; 2],
            pe_magnitude: 1.0,
            kq_scale: 0.3,
        };
        assert_eq!(kq_scale(Some(&y), 16), 0.3);
    }
}
