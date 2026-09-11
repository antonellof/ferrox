//! **YaRN's magnitude term**, and the one key that adjusts it.
//!
//! YaRN has two halves. The FREQUENCY half -- which rotary bands are
//! interpolated toward the trained context and which stay extrapolated
//! -- lives in `ferrox_core::attention::yarn_freq_factors` and reaches
//! the kernels as per-band divisors. The MAGNITUDE half is a scalar on
//! the rotated channels of q and k, and until 2026-09-11 ferrox did not
//! apply it: `ModelConfig::rope_attn_factor` carried
//! `rope.scaling.attn_factor` alone, while llama.cpp multiplies that
//! key by a YaRN term it derives from the scaling factor. Every YaRN
//! checkpoint on the generic path -- the `*-128K` Qwen3 exports, every
//! Ministral-3 -- was therefore roped at the right frequencies and the
//! wrong magnitude, with attention logits low by the SQUARE of the
//! missing term (both q and k take it). Found reading
//! `mistral3.cpp:9` for `rope.scaling.yarn_log_multiplier`, whose
//! whole job is to adjust a term ferrox turned out not to have.
//!
//! # llama.cpp's arithmetic, in three places
//!
//! `llama-context.cpp:189-227`, when the scaling type is YaRN
//! (`yarn_ext_factor` defaults to 1.0 for it and 0.0 otherwise, :190):
//!
//! ```text
//! get_mscale(scale, m) = scale <= 1 ? 1 : 0.1 * m * ln(scale) + 1
//! factor = 1 / rope_freq_scale                          // = rope.scaling.factor
//! attn   = log_mul != 0 ? get_mscale(factor, 1) / get_mscale(factor, log_mul)
//!                       : get_mscale(factor, 1)         // :202-221
//! attn  *= 1 / (1 + 0.1 * ln(factor))                   // :227, "cancel this factor"
//! attn  *= rope_attn_factor                             // :231, the GGUF key
//! ```
//!
//! and then ggml's `rope_yarn` (`ggml-cpu/ops.cpp:5835-5841`) multiplies
//! `mscale` by `1 + 0.1 * ln(1 / freq_scale)` -- the term :227 cancelled
//! -- before folding it into `cos` and `sin`. The two cancel exactly,
//! so what reaches the rotated channels is
//!
//! ```text
//! rope_attn_factor * get_mscale(factor, 1) / get_mscale(factor, log_mul)   // log_mul != 0
//! rope_attn_factor * get_mscale(factor, 1)                                  // otherwise
//! ```
//!
//! which is [`yarn_attn_magnitude`], folded into `rope_attn_factor` at
//! load time so that it rides the existing `apply_rope_attn_factor`
//! helper and the Metal `mscale` uniform unchanged.
//!
//! The `DEEPSEEK2` special case at :210-212 (`mscale = mscale_all_dims`
//! when it is not 1) belongs to the MLA engine and is not written here:
//! that engine reads no YaRN key at all today.
//!
//! # Who reads `yarn_log_multiplier` -- MEASURED
//!
//! `grep -rn rope_yarn_log_mul src/` over llama.cpp, 2026-09-11: the
//! key is read by `mistral3.cpp:9` (verbatim), `deepseek2.cpp:34-37`
//! and `deepseek32.cpp:36-39` (both divide it by 0.1 for a legacy
//! converter, `[TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]`), and applied by
//! `glm-dsa.cpp:234,610` through deepseek2's hparams. The three
//! dedicated rows apply it INSIDE their graph's `kq_scale`
//! (`deepseek2.cpp:444-448`), a different formula on a different
//! engine. On the generic path `mistral3` is the only reader, so
//! `hparams.rope_yarn_log_mul` stays at its `llama-hparams.h:133`
//! default of 0 for every other architecture and the key is dead
//! metadata there -- [`yarn_log_mul_for`] returns 0 for it, and a test
//! pins that a `llama` file carrying the key is not changed by it.
//!
//! Real Ministral-3 files carry it: `conversion/mistral3.py:28` writes
//! `mscale_all_dim` from the HF `rope_parameters`, and
//! `conversion/mistral.py:99` writes `1.0` when the checkpoint's
//! `apply_scale` is false and `0.0` when true -- so a Ministral with
//! `yarn_log_multiplier = 1.0` has NO magnitude term at all (the ratio
//! is exactly 1) and one with `0.0` takes the plain `get_mscale`.
//! `mistral3_yarn_tiny.gguf` and `mistral3_yarn_logmul_tiny.gguf`
//! evidence both arms against libllama.

/// Generic-path architectures whose `load_arch_hparams` reads
/// `rope.scaling.yarn_log_multiplier`, with the line.
///
/// A census the resolver is checked against, like every other table
/// in this crate. The dedicated readers (`deepseek2`, `deepseek32`,
/// `glm-dsa`) are deliberately absent: their engines do not go through
/// `ModelConfig::rope_attn_factor`.
pub const YARN_LOG_MUL_READERS: &[(&str, &str)] = &[("mistral3", "src/models/mistral3.cpp:9")];

/// The `yarn_log_multiplier` llama.cpp's hparams hold for `arch` given
/// what the file declares: the declared value for an architecture that
/// reads the key, the `llama-hparams.h:133` default of `0.0` for every
/// other.
pub fn yarn_log_mul_for(arch: &str, declared: Option<f32>) -> f32 {
    if YARN_LOG_MUL_READERS.iter().any(|(name, _)| *name == arch) {
        declared.filter(|v| v.is_finite()).unwrap_or(0.0)
    } else {
        0.0
    }
}

/// `llama-context.cpp:196-199`, in `f32` as it is there.
fn get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// The magnitude every rotated channel of q and k is multiplied by
/// under YaRN scaling with the given `factor` (`rope.scaling.factor`)
/// and `log_mul` (`rope.scaling.yarn_log_multiplier`, 0 when the
/// architecture does not read it), BEFORE `rope.scaling.attn_factor`.
///
/// Exactly `1.0` for `factor <= 1`, which is llama.cpp's own
/// `get_mscale` floor and why a file with no scaling is untouched.
pub fn yarn_attn_magnitude(factor: f32, log_mul: f32) -> f32 {
    // `hparams.rope_yarn_log_mul != 0.0f` is the branch at :202.
    if log_mul != 0.0 {
        get_mscale(factor, 1.0) / get_mscale(factor, log_mul)
    } else {
        get_mscale(factor, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plain term, against a hand evaluation of `get_mscale(f, 1)`:
    /// factor 4 gives `1 + 0.1 * ln 4`, which squared is the 1.30x the
    /// attention logits of every 128K Qwen3 export were low by.
    #[test]
    fn without_a_log_multiplier_the_term_is_one_plus_a_tenth_of_the_log() {
        let got = yarn_attn_magnitude(4.0, 0.0);
        let want = 1.0 + 0.1 * 4f32.ln();
        assert!((got - want).abs() < 1e-7, "{got} vs {want}");
        assert!(got > 1.1, "not a rounding-error effect: {got}");
    }

    /// `log_mul = 1.0` -- what `conversion/mistral.py:99` writes for a
    /// checkpoint whose `apply_scale` is false -- makes the ratio exactly
    /// 1: the two `get_mscale` calls take the same `mscale`.
    #[test]
    fn a_log_multiplier_of_one_cancels_the_term_exactly() {
        assert_eq!(yarn_attn_magnitude(4.0, 1.0), 1.0);
        assert_eq!(yarn_attn_magnitude(16.0, 1.0), 1.0);
    }

    /// A fractional multiplier lands between the two: the ratio form
    /// from `:215`.
    #[test]
    fn a_fractional_log_multiplier_divides_by_its_own_mscale() {
        let got = yarn_attn_magnitude(4.0, 0.5);
        let want = (1.0 + 0.1 * 4f32.ln()) / (1.0 + 0.05 * 4f32.ln());
        assert!((got - want).abs() < 1e-7, "{got} vs {want}");
        assert!(got > 1.0 && got < yarn_attn_magnitude(4.0, 0.0));
    }

    /// `scale <= 1` is unscaled on both arms, so a file with a factor
    /// of 1 (or a defensive 0.5) is untouched whatever its multiplier.
    #[test]
    fn a_factor_at_or_below_one_is_no_magnitude_change() {
        for f in [1.0, 0.5, 0.0] {
            assert_eq!(yarn_attn_magnitude(f, 0.0), 1.0);
            assert_eq!(yarn_attn_magnitude(f, 0.5), 1.0);
        }
    }

    /// The census and the resolver agree, in both directions: every
    /// listed reader takes the declared value, and an architecture
    /// llama.cpp never reads the key for keeps the 0.0 default.
    #[test]
    fn only_a_listed_reader_takes_the_declared_multiplier() {
        for (arch, line) in YARN_LOG_MUL_READERS {
            assert_eq!(
                yarn_log_mul_for(arch, Some(0.5)),
                0.5,
                "{arch} reads the key at {line}"
            );
        }
        for arch in ["llama", "qwen3", "qwen2", "gemma3", "phi3"] {
            assert_eq!(
                yarn_log_mul_for(arch, Some(0.5)),
                0.0,
                "{arch}'s load_arch_hparams never reads LLM_KV_ROPE_SCALING_YARN_LOG_MUL"
            );
        }
        assert_eq!(yarn_log_mul_for("mistral3", None), 0.0);
    }
}
