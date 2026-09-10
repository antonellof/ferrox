//! [`SamplingParams`]: everything one generation request says about how
//! the sampler chain should behave.
//!
//! Split out of `sampling.rs` because it is a different concept from the
//! RNG and the chain runner that read it, and because it is the struct
//! the whole workspace agrees about: the CLI flags build one, the two
//! OpenAI routes and llama.cpp's native `/completion` build one, the
//! response cache destructures one EXHAUSTIVELY (`ferrox_server::
//! response_cache::sampling_key`) so that a knob added here and
//! forgotten there stops that crate compiling.
//!
//! Every field's default is the value that makes its sampler a NO-OP,
//! and where llama.cpp has a neutral value the two agree. llama.cpp's
//! own CLI numbers (`--temp 0.8`, `--top-k 40`, `--min-p 0.05`) live on
//! ferrox's CLI flags, where the person who typed them can see them, and
//! not here: `SamplingParams::default()` is the "do nothing the caller
//! did not ask for" baseline that an HTTP request with an empty body
//! resolves to.

use crate::dry::DryParams;
use crate::sampler_order::SamplerOrder;

/// Sampling parameters for one generation request. `temperature <= 0.0`
/// means "sample nothing, take the greedy argmax" -- the same
/// deterministic behavior ferrox always had before this module existed.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    /// Nucleus sampling threshold in (0.0, 1.0]. 1.0 disables top-p
    /// filtering (every token with nonzero probability is eligible).
    pub top_p: f32,
    /// Keep only candidates at least `min_p` times as likely as the most
    /// likely one. `0.0` disables it; llama.cpp's `--min-p`, whose
    /// default is **0.05** (`common/common.h:231`) rather than off.
    ///
    /// That default is why this is a parity item and not a feature:
    /// llama.cpp truncates with min-p on every run nobody configured,
    /// so without it ferrox could not reproduce llama.cpp's *own*
    /// out-of-the-box output for any prompt.
    ///
    /// The struct default here stays `0.0` (disabled) for the same
    /// reason `temperature` defaults to greedy: `SamplingParams::default`
    /// is ferrox's "do nothing the caller did not ask for" baseline, and
    /// llama.cpp's CLI numbers live on the CLI flags.
    pub min_p: f32,
    /// Keep only the `top_k` highest-probability tokens before
    /// sampling. 0 disables top-k filtering.
    pub top_k: usize,
    /// Locally typical sampling, llama.cpp's `typ_p`
    /// (`common/common.h:230`, default **1.0 = disabled**).
    ///
    /// Keeps the candidates whose surprisal is CLOSEST to the
    /// distribution's entropy, from the middle outward, rather than the
    /// most likely ones -- see
    /// [`crate::sampler_chain::Candidates::typical_p`].
    pub typical_p: f32,
    /// Truncate at `n` standard deviations of the logits below the
    /// maximum, llama.cpp's `top_n_sigma` (`common/common.h:250`,
    /// default **-1.0 = disabled**).
    pub top_n_sigma: f32,
    /// The probability that XTC removes the top candidates on any one
    /// token, llama.cpp's `xtc_probability` (`common/common.h:228`,
    /// default **0.0 = disabled**).
    pub xtc_probability: f32,
    /// The probability a candidate must reach to be a candidate XTC
    /// might remove, llama.cpp's `xtc_threshold` (`common/common.h:229`,
    /// default 0.1). **Above 0.5 disables XTC**, which is upstream's
    /// guard and not a range check: above 0.5 at most one candidate can
    /// ever clear it, and XTC never removes the last one.
    pub xtc_threshold: f32,
    /// The DRY sequence-repetition penalty. Disabled by default; see
    /// [`crate::dry`] for why its breakers are a type invariant rather
    /// than four more `f32`s here.
    pub dry: DryParams,
    /// > 1.0 discourages repeating a token already in the
    /// > [`crate::penalty_window::PenaltyWindow`] -- prompt included;
    /// > 1.0 disables repetition penalty. Uses the standard convention
    /// > (divide positive logits, multiply negative ones) so the penalty
    /// > always pushes toward *less* likely, regardless of logit sign.
    pub repetition_penalty: f32,
    /// How many of the most recent tokens the penalties look at, as
    /// llama.cpp's `penalty_last_n` (`common/common.h:238`, default 64).
    ///
    /// `0` disables the penalties entirely. ferrox had no window at all
    /// and scanned the WHOLE history, so on a long generation it
    /// penalised a steadily growing set of tokens where llama.cpp
    /// penalises the last 64 -- the divergence grew with output length,
    /// which is exactly when a repetition penalty matters most.
    pub penalty_last_n: usize,
    /// OpenAI-style presence penalty: subtract from logits of tokens
    /// that already appeared in the window (once per distinct token).
    pub presence_penalty: f32,
    /// OpenAI-style frequency penalty: subtract `frequency_penalty *
    /// count` from logits for each token id seen in the window.
    pub frequency_penalty: f32,
    /// The ORDER the chain above runs in, llama.cpp's `--samplers`.
    ///
    /// Not a cosmetic setting. Each filter renormalises over the
    /// survivors of the last one, so moving a step changes which
    /// candidates the next step can see -- ferrox has already shipped
    /// that bug once, with temperature running first.
    ///
    /// The default is llama.cpp's own default chain
    /// (`penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`),
    /// and every step ferrox added to it is a no-op at the neutral
    /// values above. See [`crate::sampler_order`].
    pub sampler_order: SamplerOrder,
}

impl Default for SamplingParams {
    /// Greedy decoding: identical behavior to ferrox's original
    /// argmax-only generation loop.
    fn default() -> Self {
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            min_p: 0.0,
            top_k: 0,
            typical_p: 1.0,
            top_n_sigma: -1.0,
            xtc_probability: 0.0,
            xtc_threshold: 0.1,
            dry: DryParams::off(),
            repetition_penalty: 1.0,
            penalty_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            sampler_order: SamplerOrder::default(),
        }
    }
}

impl SamplingParams {
    /// The single predicate for "XTC can remove something".
    ///
    /// llama.cpp tests the same two conditions in two places --
    /// `llama_sampler_init_xtc` returns an empty sampler at `:2208` and
    /// `llama_sample_xtc_apply` returns early at `:2139` -- and this is
    /// one function because ferrox reads it in two places too: the RNG
    /// draw ([`super::Sampler::xtc_roll`]) and the filter itself. If
    /// those disagreed, either the seeded stream would advance on a run
    /// XTC never touched (making an existing generation irreproducible)
    /// or XTC would ask for a draw nobody made.
    pub fn xtc_can_fire(&self) -> bool {
        self.xtc_probability > 0.0 && self.xtc_threshold <= 0.5
    }

    // The two "may an argmax stand in for the chain" predicates --
    // [`Self::chain_keeps_the_argmax`] and
    // [`Self::greedy_equals_raw_argmax`] -- are in
    // [`super::greedy_equivalence`], with the exhaustive destructure of
    // THIS struct that a knob added below must satisfy before the crate
    // compiles. They used to be one function here, and it hand-listed
    // three of the chain's nine steps: GitHub issue #170.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every knob this struct defaults to is the value that makes its
    /// sampler do nothing, which is what lets llama.cpp's full default
    /// chain be ferrox's default chain without changing any existing
    /// run's output.
    ///
    /// The neutral values are llama.cpp's own (`common/common.h:228-250`):
    /// `typ_p 1.00`, `top_n_sigma -1.00`, `xtc_probability 0.00`,
    /// `dry_multiplier 0.0`. `xtc_threshold` defaults to upstream's
    /// 0.10, which is NOT neutral on its own -- the probability is what
    /// switches XTC off -- so it is pinned here rather than assumed.
    #[test]
    fn every_new_sampler_defaults_to_its_own_no_op() {
        let d = SamplingParams::default();
        assert_eq!(d.typical_p, 1.0, "1.0 disables typical-p");
        assert_eq!(d.top_n_sigma, -1.0, "<= 0 disables top-n-sigma");
        assert_eq!(d.xtc_probability, 0.0, "0.0 disables xtc");
        assert_eq!(d.xtc_threshold, 0.1, "llama.cpp's default threshold");
        assert!(!d.xtc_can_fire());
        assert!(!d.dry.is_enabled(), "dry_multiplier 0.0 disables dry");
    }

    /// Both halves of the XTC guard, because only one of them is
    /// obvious. A threshold above 0.5 disables XTC outright upstream: at
    /// most one candidate can hold more than half the mass, and XTC
    /// never removes the last candidate above the threshold.
    #[test]
    fn a_threshold_above_a_half_disables_xtc_as_surely_as_a_zero_probability() {
        let live = SamplingParams {
            xtc_probability: 0.5,
            xtc_threshold: 0.1,
            ..SamplingParams::default()
        };
        assert!(live.xtc_can_fire());
        assert!(!SamplingParams {
            xtc_threshold: 0.51,
            ..live.clone()
        }
        .xtc_can_fire());
        assert!(
            SamplingParams {
                xtc_threshold: 0.5,
                ..live.clone()
            }
            .xtc_can_fire(),
            "0.5 itself is still live"
        );
        assert!(!SamplingParams {
            xtc_probability: 0.0,
            ..live
        }
        .xtc_can_fire());
    }
}
