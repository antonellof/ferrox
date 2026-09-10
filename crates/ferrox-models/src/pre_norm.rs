//! The pre-branch normalisation slot of one decoder sublayer, and the
//! post-norm-only residual topology that has none.
//!
//! Nearly every architecture on the generic path is PRE-norm: each
//! sublayer normalises the residual before it reads it, and the branch
//! output is added back raw.
//!
//! ```text
//! ffn_inp = x       + attn(rms(x, attn_norm))
//! out     = ffn_inp + ffn(rms(ffn_inp, ffn_norm))
//! ```
//!
//! Gemma-2 added a *sandwich*: the same two pre-norms, plus a norm on
//! each branch's OUTPUT before its residual add. ferrox has carried
//! those two as [`crate::decoder::AttnWeights::post_attn_norm`] and
//! [`crate::decoder::AttnWeights::post_ffn_norm`] for a long time.
//!
//! **`olmo2` and `exaone4` are the sandwich with the bread taken off.**
//! They have the two post-norms and NO pre-norms at all, and both
//! sublayers read the raw residual:
//!
//! ```text
//! ffn_inp = x       + post_attn_norm(attn(x))
//! out     = ffn_inp + post_ffn_norm(ffn(ffn_inp))
//! ```
//!
//! That claim is a reading of both files, not a family resemblance:
//!
//! | | `src/models/olmo2.cpp` | `src/models/exaone4.cpp` |
//! |---|---|---|
//! | per-layer norms created | `attn_q_norm`, `attn_k_norm`, `attn_post_norm`, `ffn_post_norm` (:45-52) | `attn_post_norm`, `attn_q_norm`, `attn_k_norm`, `ffn_post_norm` (:60-67) |
//! | `attn_norm` | absent | absent |
//! | `ffn_norm` | absent | absent |
//! | Q/K/V read | `cur = inpL` (:92) | `cur = inpL` (:118) |
//! | attention output | `build_norm(cur, attn_post_norm)` (:160-163) | `build_norm(cur, attn_post_norm)` (:152) |
//! | `ffn_inp` | `add(cur, inpSA)` (:165) | `add(cur, inpSA)` (:155) |
//! | FFN input | `build_ffn(ffn_inp, ...)` (:169) | `build_ffn(ffn_inp, ...)` (:159) |
//! | FFN output | `build_norm(cur, ffn_post_norm)` (:177-179) | `build_norm(cur, ffn_post_norm)` (:166) |
//! | residual | `add(cur, ffn_inp)` (:182) | `add(cur, ffn_inp)` (:169) |
//!
//! Line for line the same graph. So the two rows share ONE
//! implementation -- this module -- rather than getting one arm each.
//! What they do NOT share is their QK-norm style (`olmo2` norms the 2-D
//! projection over its whole width, `exaone4` per head after
//! `build_qkv` has reshaped), which is why each still needs its own
//! fixture: `tests/post_norm_only_graphs.rs`.
//!
//! # Why this is a type and not a `bool` on `ModelConfig`
//!
//! The pre-norm weights are handed to fused Metal kernels that apply
//! the norm INSIDE the kernel (`PrefillDenseLayerMetal::attn_norm_w`,
//! `MoeLayerMetal::ffn_norm_w`, `launch_decode_dense_layer`). A flag on
//! the config would leave every one of those launches free to keep
//! reading a `&[f32]` that no longer means anything, and nothing would
//! fail: that is precisely this repo's dominant bug shape, two
//! structures that must agree with nothing enforcing it, and it is how
//! `post_attn_norm` was lost from a decode path once already.
//!
//! Making the slot an enum instead means a fused launch cannot compile
//! until it has said what it does when there is no weight to hand over.
//! [`PreNorm::rms_weights`] returns `None` for [`PreNorm::None`], and
//! every GPU call site turns that into a fall-back to the host body,
//! which computes the right thing. The disagreement is a type error
//! rather than a silent wrong answer.

use ferrox_core::matmul::rms_norm;

/// One sublayer's pre-branch normalisation.
#[derive(Debug, Clone, PartialEq)]
pub enum PreNorm {
    /// RMSNorm with these learned weights, applied to the residual
    /// before the branch reads it. Every architecture on the generic
    /// path except the post-norm-only family.
    Rms(Vec<f32>),
    /// No pre-norm: the branch reads the raw residual.
    ///
    /// `olmo2` and `exaone4`. NOT "an RMSNorm whose weights are all
    /// ones" -- that would still divide by the RMS of the residual, and
    /// the whole point of this variant is that nothing is divided.
    None,
}

impl PreNorm {
    /// The sublayer's input: `rms_norm(x, w, eps)`, or `x` itself when
    /// there is no pre-norm.
    ///
    /// Returns an owned vector in both arms, which is what every caller
    /// already had: `rms_norm` allocates too, so the post-norm-only path
    /// costs the same copy the pre-norm path always paid.
    pub fn apply(&self, x: &[f32], eps: f32) -> Vec<f32> {
        match self {
            Self::Rms(w) => rms_norm(x, w, eps),
            Self::None => x.to_vec(),
        }
    }

    /// The weights a fused GPU kernel needs, or `None` when this
    /// sublayer has no pre-norm and the kernel therefore must not run.
    ///
    /// Every fused Metal launch that bakes the RMSNorm into its kernel
    /// goes through here and falls back to the host body on `None`.
    /// See the module docs for why that is a `?` and not a comment.
    pub fn rms_weights(&self) -> Option<&[f32]> {
        match self {
            Self::Rms(w) => Some(w),
            Self::None => None,
        }
    }
}

// There is deliberately no `is_present()` beside `rms_weights()`.
// "Does this sublayer norm?" and "what weights does the kernel get?"
// are the same fact, and two spellings of one fact is the shape this
// repo keeps shipping bugs in: `rms_weights().is_some()` is the only
// way to ask.

#[cfg(test)]
mod tests {
    use super::*;

    /// `PreNorm::None` is the identity, and an all-ones RMSNorm is not.
    ///
    /// The tempting shortcut for `olmo2` / `exaone4` was to load a
    /// vector of ones into the existing slot and change nothing else.
    /// It is wrong by exactly the RMS scale factor, which for a residual
    /// with any magnitude at all is not close to 1. This test is what
    /// stops somebody re-discovering that as a "simplification".
    #[test]
    fn no_pre_norm_is_the_identity_and_an_all_ones_rmsnorm_is_not() {
        let x = vec![3.0f32, -4.0, 12.0, 0.5];
        let eps = 1e-5;

        assert_eq!(PreNorm::None.apply(&x, eps), x);

        let ones = PreNorm::Rms(vec![1.0; x.len()]);
        let normed = ones.apply(&x, eps);
        let worst = x
            .iter()
            .zip(normed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst > 1.0,
            "an all-ones RMSNorm moved the vector by only {worst}; if it were the \
             identity the post-norm-only topology would not need a variant at all"
        );
    }

    /// A fused GPU launch cannot be handed weights that do not exist.
    #[test]
    fn only_the_rms_variant_offers_weights_to_a_fused_kernel() {
        assert_eq!(
            PreNorm::Rms(vec![2.0, 3.0]).rms_weights(),
            Some(&[2.0f32, 3.0][..])
        );
        assert_eq!(PreNorm::None.rms_weights(), None);
    }
}
