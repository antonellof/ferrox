//! The normalisation at ONE site in a decoder: before the attention
//! branch, before the FFN branch, or before the LM head.
//!
//! Three variants, because llama.cpp's `build_norm`
//! (`llama-graph.cpp`) has three answers at those sites and ferrox used
//! to have one. It takes a norm TYPE (`LLM_NORM` = LayerNorm,
//! `LLM_NORM_RMS` = RMSNorm) and a weight that may be null, and applies
//! the multiply only `if (mw)`. Everything below is a reading of that
//! function and of the three graphs that reach its corners.
//!
//! ```text
//! // the ordinary case, and nearly every architecture on the generic path
//! ffn_inp = x       + attn(rms(x, attn_norm))
//! out     = ffn_inp + ffn(rms(ffn_inp, ffn_norm))
//! ```
//!
//! Gemma-2 added a *sandwich*: the same two pre-norms, plus a norm on
//! each branch's OUTPUT before its residual add. ferrox has carried
//! those two as [`crate::decoder::AttnWeights::post_attn_norm`] and
//! [`crate::decoder::AttnWeights::post_ffn_norm`] for a long time, and
//! they are not this type -- they are `Option<Vec<f32>>` RMSNorms, and
//! no architecture has ever wanted anything else there.
//!
//! # [`NormOp::None`]: `olmo2` and `exaone4`
//!
//! **The sandwich with the bread taken off.** They have the two
//! post-norms and NO pre-norms at all, and both sublayers read the raw
//! residual:
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
//! # [`NormOp::LayerNormNoParams`]: `olmo`, and only `olmo`
//!
//! OLMo-1 is a THIRD shape, and it is not about the residual at all.
//! `src/models/olmo.cpp:27-35` creates Q/K/V, `attn_output` and
//! gate/up/down and **not one norm tensor** -- no `attn_norm`, no
//! `ffn_norm`, no `output_norm` -- and its graph normalises at all
//! three sites with a null weight AND a null bias:
//!
//! ```text
//! // olmo.cpp:65-67, :104-106, :128-130
//! cur = build_norm(inpL, NULL, NULL, LLM_NORM, il);
//! ```
//!
//! `LLM_NORM` is `ggml_norm` (`llama-graph.cpp`'s `build_norm`), which
//! subtracts the mean and divides by the standard deviation
//! (`ggml/src/ggml-cpu/ops.cpp:3716-3745`); with both weight and bias
//! null, `build_norm` does nothing further. So it is a pre-norm layer
//! like `llama`, with a different norm FUNCTION and no parameters at
//! all. `NormOp::None` is no help here and neither is `NormOp::Rms`.
//!
//! **This is the only architecture in llama.cpp that does it.** Scanned
//! over every `build_norm` call in all 140 `src/models/*.cpp` graphs,
//! extracting the weight argument: three calls pass a null weight to
//! `LLM_NORM`, and all three are `olmo.cpp`. (`talkie.cpp` passes a null
//! weight to `LLM_NORM_RMS` at five sites, which is a different
//! function and a different row.) So this variant closes exactly one
//! refusal and the hoped-for shared cause is not there -- see
//! `capability::NON_PARAMETRIC_LAYER_NORM`, which says so where the next
//! person will look.
//!
//! # Why this is a type and not a `bool` on `ModelConfig`
//!
//! The RMSNorm weights are handed to fused Metal kernels that apply the
//! norm INSIDE the kernel (`PrefillDenseLayerMetal::attn_norm_w`,
//! `MoeLayerMetal::ffn_norm_w`, `launch_decode_dense_layer`). A flag on
//! the config would leave every one of those launches free to keep
//! reading a `&[f32]` that no longer means anything, and nothing would
//! fail: that is precisely this repo's dominant bug shape, two
//! structures that must agree with nothing enforcing it, and it is how
//! `post_attn_norm` was lost from a decode path once already.
//!
//! Making the slot an enum instead means a fused launch cannot compile
//! until it has said what it does when there is no weight to hand over.
//! [`NormOp::rms_weights`] returns `None` for BOTH non-RMS variants, and
//! every GPU call site turns that into a fall-back to the host body,
//! which computes the right thing. The disagreement is a type error
//! rather than a silent wrong answer. `Decoder::final_norm` is this type
//! for the same reason: `olmo` is the first architecture whose FINAL
//! norm is not an RMSNorm either, and the fused stacks that fold
//! `final_norm + lm_head + argmax` had `Some(&self.final_norm)` written
//! into them unconditionally.

use ferrox_core::matmul::rms_norm;

/// The normalisation applied at one norm site.
#[derive(Debug, Clone, PartialEq)]
pub enum NormOp {
    /// RMSNorm with these learned weights. Every architecture on the
    /// generic path except the two families below.
    Rms(Vec<f32>),
    /// Non-parametric LayerNorm: subtract the mean, divide by the
    /// standard deviation, no learned weight and no bias.
    ///
    /// `olmo` (OLMo-1) and nothing else in llama.cpp. There is no
    /// `LayerNorm(Vec<f32>)` beside this because no architecture ferrox
    /// admits needs one: the weighted-LayerNorm rows (`dbrx`,
    /// `nemotron`, `orion`, `stablelm`, ...) all refuse for other
    /// reasons too, and a variant with no caller is a variant that
    /// silently rots.
    LayerNormNoParams,
    /// No norm at all: the branch reads the raw residual.
    ///
    /// `olmo2` and `exaone4`. NOT "an RMSNorm whose weights are all
    /// ones" -- that would still divide by the RMS of the residual, and
    /// the whole point of this variant is that nothing is divided.
    None,
}

impl NormOp {
    /// The site's output: `rms_norm(x, w, eps)`, the non-parametric
    /// LayerNorm, or `x` itself.
    ///
    /// Returns an owned vector in every arm, which is what every caller
    /// already had: `rms_norm` allocates too.
    ///
    /// `eps` is `ModelConfig::rms_norm_eps`, which for a LayerNorm
    /// architecture is read from `{arch}.attention.layer_norm_epsilon`
    /// -- llama.cpp's `f_norm_eps` rather than `f_norm_rms_eps`. The two
    /// are one field here because no architecture reads both keys, and
    /// `loader.rs` already accepted either spelling into that field
    /// before this variant existed.
    pub fn apply(&self, x: &[f32], eps: f32) -> Vec<f32> {
        match self {
            Self::Rms(w) => rms_norm(x, w, eps),
            Self::LayerNormNoParams => layer_norm_no_params(x, eps),
            Self::None => x.to_vec(),
        }
    }

    /// The weights a fused GPU kernel needs, or `None` when this site
    /// has no RMSNorm weights and the kernel therefore must not run.
    ///
    /// Every fused Metal launch that bakes an RMSNorm into its kernel
    /// goes through here and falls back to the host body on `None`.
    /// See the module docs for why that is a `?` and not a comment.
    pub fn rms_weights(&self) -> Option<&[f32]> {
        match self {
            Self::Rms(w) => Some(w),
            Self::LayerNormNoParams | Self::None => None,
        }
    }
}

/// `(x - mean) / sqrt(var + eps)`, with the BIASED variance.
///
/// `ggml_compute_forward_norm_f32` (`ggml/src/ggml-cpu/ops.cpp:3716-3745`)
/// divides the sum of squared deviations by `ne00`, not by `ne00 - 1`.
/// Bessel's correction on a 4096-wide hidden state is a factor of
/// 1.00012, which is far too small to fail a smoke test and far too
/// large to be right.
fn layer_norm_no_params(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    debug_assert!(n > 0.0, "a norm site with no elements");
    let mean = x.iter().sum::<f32>() / n;
    // The deviations are computed once and reused, which is also what
    // ggml does: it writes `x - mean` into the destination and takes
    // the variance from there.
    let mut out: Vec<f32> = x.iter().map(|v| v - mean).collect();
    let var = out.iter().map(|d| d * d).sum::<f32>() / n;
    let scale = 1.0 / (var + eps).sqrt();
    for v in out.iter_mut() {
        *v *= scale;
    }
    out
}

// There is deliberately no `is_present()` beside `rms_weights()`.
// "Does this site norm?" and "what weights does the kernel get?" are
// the same fact for a fused kernel, and two spellings of one fact is
// the shape this repo keeps shipping bugs in: `rms_weights().is_some()`
// is the only way to ask.

#[cfg(test)]
mod tests {
    use super::*;

    /// `NormOp::None` is the identity, and an all-ones RMSNorm is not.
    ///
    /// The tempting shortcut for `olmo2` / `exaone4` was to load a
    /// vector of ones into the existing slot and change nothing else.
    /// It is wrong by exactly the RMS scale factor, which for a residual
    /// with any magnitude at all is not close to 1. This test is what
    /// stops somebody re-discovering that as a "simplification".
    #[test]
    fn no_norm_is_the_identity_and_an_all_ones_rmsnorm_is_not() {
        let x = vec![3.0f32, -4.0, 12.0, 0.5];
        let eps = 1e-5;

        assert_eq!(NormOp::None.apply(&x, eps), x);

        let ones = NormOp::Rms(vec![1.0; x.len()]);
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

    /// The non-parametric LayerNorm is not the all-ones RMSNorm either,
    /// and the difference is the MEAN.
    ///
    /// This is the shortcut somebody will reach for next: "OLMo-1 just
    /// has no norm weights, so load ones". On a vector with a non-zero
    /// mean the two differ in every element, and the assertion below
    /// measures that rather than trusting it -- a centred input would
    /// make them agree and would prove nothing.
    #[test]
    fn the_layer_norm_subtracts_the_mean_and_an_all_ones_rmsnorm_does_not() {
        let x = vec![3.0f32, -4.0, 12.0, 0.5];
        let eps = 1e-5;
        let ln = NormOp::LayerNormNoParams.apply(&x, eps);
        let rms = NormOp::Rms(vec![1.0; x.len()]).apply(&x, eps);

        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        assert!(mean.abs() > 1.0, "the input must not be centred: {mean}");

        let out_mean: f32 = ln.iter().sum::<f32>() / ln.len() as f32;
        assert!(
            out_mean.abs() < 1e-5,
            "a LayerNorm's output is centred; got mean {out_mean}"
        );

        let worst = ln
            .iter()
            .zip(rms.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst > 0.1,
            "the two norms differ by only {worst}; an all-ones RMSNorm would then be a \
             legitimate stand-in for OLMo-1's norm and this variant would be decoration"
        );
    }

    /// The variance is BIASED (divide by n), which is ggml's.
    ///
    /// Checked against arithmetic written out here rather than against
    /// the implementation: the two spellings differ by sqrt(n/(n-1)),
    /// which on this 4-element vector is 1.155 and on a real hidden
    /// state is 1.0001 -- big enough to be wrong, small enough that no
    /// end-to-end smoke test would notice.
    #[test]
    fn the_variance_is_the_biased_one_ggml_uses() {
        let x = [1.0f32, 2.0, 3.0, 10.0];
        let got = NormOp::LayerNormNoParams.apply(&x, 0.0);

        let n = x.len() as f64;
        let mean = x.iter().map(|v| *v as f64).sum::<f64>() / n;
        let var = x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
        let want: Vec<f32> = x
            .iter()
            .map(|v| ((*v as f64 - mean) / var.sqrt()) as f32)
            .collect();

        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {got:?}, want {want:?}");
        }

        // ... and the SAMPLE variance would be visibly different here,
        // so this test can tell them apart.
        let sample = x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let worst = got
            .iter()
            .zip(x.iter())
            .map(|(g, v)| (g - ((*v as f64 - mean) / sample.sqrt()) as f32).abs())
            .fold(0f32, f32::max);
        assert!(worst > 0.1, "the two variances differ by only {worst}");
    }

    /// A fused GPU launch cannot be handed weights that do not exist,
    /// from EITHER non-RMS variant.
    #[test]
    fn only_the_rms_variant_offers_weights_to_a_fused_kernel() {
        assert_eq!(
            NormOp::Rms(vec![2.0, 3.0]).rms_weights(),
            Some(&[2.0f32, 3.0][..])
        );
        assert_eq!(NormOp::None.rms_weights(), None);
        assert_eq!(NormOp::LayerNormNoParams.rms_weights(), None);
    }
}
