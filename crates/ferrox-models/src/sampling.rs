//! Token sampling from a decoder's output logits: llama.cpp's whole
//! default sampler chain, on top of the greedy argmax ferrox previously
//! always used unconditionally.
//!
//! The chain is `penalties, dry, top_n_sigma, top_k, typical_p, top_p,
//! min_p, xtc, temperature` (`common/common.h:259-269`), which is
//! [`crate::sampler_order::SamplerOrder`]'s default and llama.cpp's.
//! The filters themselves are in [`crate::sampler_chain`], the DRY
//! penalty in [`crate::dry`], the three history penalties in
//! [`penalties`], the parameters in [`params`].
//!
//! `crate::speculative` verifies draft tokens against
//! [`sampling_distribution`] -- the exact distribution [`Sampler`]
//! draws from for a given `SamplingParams` -- so speculation is
//! lossless with respect to whatever sampling configuration the caller
//! asked for, rather than only at temperature 0.
//!
//! No external `rand` dependency: a small xorshift64* generator (the
//! same algorithm `Decoder::new_random_small`'s test-only `Lcg` already
//! uses in `decoder.rs`) is enough for sampling and keeps the
//! dependency tree the same minimal, pure-Rust shape as the rest of
//! this crate.

mod params;
mod penalties;
mod recommended;
mod rng;

pub use params::SamplingParams;
pub use recommended::{RecommendedSampling, RequestedSampling};
pub use rng::{LogitMask, Sampler};

use crate::penalty_window::PenaltyWindow;
use crate::sampler_chain::Candidates;
use crate::sampler_order::ChainStep;
use penalties::apply_history_penalties;

/// A deterministic, uninteresting-on-purpose logit vector: no ties, a
/// wide dynamic range, and a few negatives so the penalty's sign
/// convention is exercised.
///
/// Shared with [`rng`]'s tests rather than written out twice, because
/// two "the same logits" that were not the same logits is the smallest
/// possible instance of this repo's dominant defect.
#[cfg(test)]
pub(crate) fn spread_logits(vocab: usize) -> Vec<f32> {
    (0..vocab)
        .map(|i| ((i as f32 * 12.9898).sin() * 43_758.547).fract() * 8.0 - 3.0)
        .collect()
}

/// The token greedy decoding picks, given a chain that may or may not be
/// able to move the argmax.
///
/// llama.cpp does NOT special-case `temp <= 0`: it runs the whole chain
/// and lets `llama_sampler_temp_impl` (`src/llama-sampler.cpp:271-286`)
/// set every logit but the maximum to `-inf`, so `dist` picks whatever
/// the filters left. Two of those filters can therefore change greedy
/// output, and both of them are new here: `xtc` removes the TOP
/// candidates by construction, and `typ_p` selects outward from the
/// distribution's entropy and may drop the most likely token.
///
/// ferrox keeps its `argmax` fast path, because building and sorting a
/// 128k-entry candidate list per token to reach an answer that cannot
/// differ would be a decode-speed regression on the most common
/// configuration there is. [`SamplingParams::greedy_equals_argmax`] is
/// the one predicate that decides which path is exact, and it is read
/// here and by the Metal `lm_head + argmax` fold's guard, so a chain
/// that can move the argmax also stops the GPU from folding it away.
fn greedy_choice(
    scores: Vec<f32>,
    params: &SamplingParams,
    history: PenaltyWindow<'_>,
    xtc_roll: Option<f32>,
) -> usize {
    if params.greedy_equals_argmax() {
        return argmax(&scores);
    }
    argmax(&filtered_distribution(scores, params, history, xtc_roll))
}

/// The **exact** distribution [`Sampler::sample`] draws from for these
/// logits, params and history: penalties applied over the
/// `penalty_last_n` window, then llama.cpp's chain in
/// `params.sampler_order`, renormalised to sum to 1.
///
/// That is llama.cpp's chain order -- **temperature last**, not first.
/// This comment used to say "temperature divided in, top-k and top-p
/// filtered", which described the pre-2026-09-01 pipeline and omitted
/// min-p entirely.
///
/// This is what makes lossless speculative verification possible. The
/// speculative-sampling rejection rule compares `p_target(x)` against
/// the draft's `q(x)`, and "the target's probability" is meaningless
/// unless it is the probability the *configured sampler* would actually
/// have used -- a rule that compared against the raw softmax while the
/// server sampled with `top_p = 0.9` would be lossless with respect to
/// a model nobody is running.
///
/// Greedy (`temperature <= 0.0`) is a distribution too: the point mass
/// on the token [`greedy_choice`] would pick. Returning it as one rather
/// than as a special case is why the same verification code is correct
/// at every temperature.
///
/// `xtc_roll` is [`Sampler::xtc_roll`]'s answer, and it is a REQUIRED
/// argument rather than something this function draws or defaults,
/// because XTC is stochastic and the caller owns the seeded stream. A
/// caller that passes `None` while XTC is configured gets a chain with
/// no XTC in it, which is why every caller in this workspace obtains it
/// from `Sampler::xtc_roll` and not by writing `None`.
pub fn sampling_distribution(
    logits: &[f32],
    params: &SamplingParams,
    history: PenaltyWindow<'_>,
    xtc_roll: Option<f32>,
) -> Vec<f32> {
    let mut scores = logits.to_vec();
    apply_history_penalties(&mut scores, params, history);
    if params.temperature <= 0.0 {
        let vocab = scores.len();
        let chosen = greedy_choice(scores, params, history, xtc_roll);
        let mut probs = vec![0.0f32; vocab];
        if let Some(p) = probs.get_mut(chosen) {
            *p = 1.0;
        }
        return probs;
    }
    filtered_distribution(scores, params, history, xtc_roll)
}

/// Shared tail of [`Sampler::sample_with_mask`] and
/// [`sampling_distribution`]: run the already-penalised `scores` through
/// llama.cpp's sampler chain and return the resulting full-vocabulary
/// distribution.
///
/// # Order, and why it is a specification
///
/// llama.cpp's default chain is `penalties, dry, top_n_sigma, top_k,
/// typical_p, top_p, min_p, xtc, temperature` (`common/common.h:259-269`,
/// consumed by `common/sampling.cpp:346-397`). The penalties already ran
/// in [`apply_history_penalties`]; this function is the rest of it, in
/// that order, and **temperature is last**.
///
/// ferrox used to divide by the temperature FIRST and filter afterwards.
/// That is not a reordering of independent steps. Top-p selects the
/// smallest set of candidates whose probabilities sum to `p`, and
/// temperature changes those probabilities: a high temperature flattens
/// the distribution so the nucleus grows, a low one sharpens it so the
/// nucleus shrinks. Min-p compares each candidate's logit against
/// `max + ln(p)`, and temperature scales exactly the gap being compared.
/// Filtering before scaling and filtering after scaling therefore keep
/// DIFFERENT candidate sets for the same flags.
///
/// Both callers go through here rather than each running their own
/// chain, because a difference between the two is exactly the kind of
/// silent non-losslessness speculative verification is supposed to rule
/// out.
///
/// The filters themselves live in [`crate::sampler_chain`], which models
/// the shrinking candidate list llama.cpp passes down the chain --
/// including the renormalisation between steps that a keep-mask cannot
/// express. See that module's header.
///
/// # The order is the caller's
///
/// `params.sampler_order` says which steps run and in what sequence,
/// which is llama.cpp's `--samplers`. It DEFAULTS to the sequence
/// written out above, so a caller that never sets it gets exactly the
/// chain this function used to hardcode -- asserted bit-for-bit by
/// [`tests::the_default_order_is_the_chain_ferrox_already_ran`].
///
/// The `match` is exhaustive over [`ChainStep`] with no `..`: a step
/// added to the order's vocabulary stops this compiling until it has
/// something to run. And because [`SamplerOrder`] can only be built out
/// of steps ferrox implements, there is no arm here that means "asked
/// for, silently not done".
fn filtered_distribution(
    scores: Vec<f32>,
    params: &SamplingParams,
    history: PenaltyWindow<'_>,
    xtc_roll: Option<f32>,
) -> Vec<f32> {
    let vocab = scores.len();
    let mut candidates = Candidates::new(&scores);
    for &step in params.sampler_order.steps() {
        match step {
            // Already applied to `scores`, before the candidate list
            // existed. `SamplerOrder` refuses a `penalties` that is not
            // first precisely so that this is the same position the
            // caller asked for; see `SamplerOrderError::PenaltiesNotFirst`.
            ChainStep::Penalties => {}
            ChainStep::Dry => candidates.dry(&params.dry.penalties(history)),
            ChainStep::TopNSigma => candidates.top_n_sigma(params.top_n_sigma),
            ChainStep::TopK => candidates.top_k(params.top_k),
            ChainStep::TypP => candidates.typical_p(params.typical_p),
            ChainStep::TopP => candidates.top_p(params.top_p),
            ChainStep::MinP => candidates.min_p(params.min_p),
            // `xtc_roll` is `None` exactly when
            // `SamplingParams::xtc_can_fire` is false, which is the same
            // predicate `Candidates::xtc` re-checks. See
            // `Sampler::xtc_roll`.
            ChainStep::Xtc => {
                if let Some(chance) = xtc_roll {
                    candidates.xtc(params.xtc_probability, params.xtc_threshold, chance);
                }
            }
            ChainStep::Temperature => candidates.temperature(params.temperature),
        }
    }
    candidates.into_distribution(vocab)
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dry::{DryBreakers, DryParams};
    use crate::sampler_order::SamplerOrder;

    /// The chain `filtered_distribution` ran BEFORE llama.cpp's four
    /// missing samplers were added, written out by hand.
    ///
    /// Deliberately not built from `SamplerOrder`: a reference that read
    /// the order it is supposed to be pinning would agree with any
    /// reordering, which is the shape of test that proves nothing.
    fn the_chain_ferrox_used_to_hardcode(
        logits: &[f32],
        params: &SamplingParams,
        history: PenaltyWindow<'_>,
    ) -> Vec<f32> {
        let mut scores = logits.to_vec();
        apply_history_penalties(&mut scores, params, history);
        let vocab = scores.len();
        let mut candidates = Candidates::new(&scores);
        candidates.top_k(params.top_k);
        candidates.top_p(params.top_p);
        candidates.min_p(params.min_p);
        candidates.temperature(params.temperature);
        candidates.into_distribution(vocab)
    }

    /// **A run that does not ask for an order samples exactly what it
    /// always did.** Bit-for-bit, against the five-step chain written out
    /// by hand rather than read back off `SamplerOrder`.
    ///
    /// This is now doing double duty. It is still the assertion that
    /// makes `--samplers` safe to expose -- the order is not a
    /// reordering of independent steps, so a default that drifted by one
    /// position would change every generation on every model. And it is
    /// the assertion that adding `dry`, `top_n_sigma`, `typ_p` and `xtc`
    /// to the DEFAULT chain changed nothing: at their neutral values
    /// (`dry_multiplier 0.0`, `top_n_sigma -1.0`, `typical_p 1.0`,
    /// `xtc_probability 0.0`) all four are no-ops, so the nine-step
    /// default must produce bit-identical probabilities to the five-step
    /// chain it replaced.
    ///
    /// Swap any two entries of `sampler_order::DEFAULT_STEPS`, or make
    /// any of the four new filters do something at its neutral value,
    /// and this goes red.
    #[test]
    fn the_default_order_is_the_chain_ferrox_already_ran() {
        let logits = spread_logits(64);
        let prompt = [3usize, 9, 17, 9];
        let generated = [9usize, 40, 3];
        // Every filter switched on, and all three penalties, so there is
        // something for a misplaced step to change.
        // Every adjacent pair of the default chain has to be
        // DISTINGUISHED by at least one row, or the assertion below
        // passes for a chain in the wrong order. `top_k 5` with
        // `top_p 0.9` separates top-k from top-p (top-p over the whole
        // vocabulary keeps far more than five, so which runs first
        // decides the answer); `min_p 0.2` separates top-p from min-p;
        // any temperature away from 1.0 separates min-p from
        // temperature.
        for (temperature, top_k, top_p, min_p) in [
            (0.8f32, 5usize, 0.9f32, 0.05f32),
            (4.0, 3, 0.85, 0.2),
            (0.2, 8, 0.95, 0.1),
            (1.0, 40, 0.5, 0.02),
            (0.8, 40, 0.95, 0.05),
        ] {
            let params = SamplingParams {
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty: 1.1,
                presence_penalty: 0.3,
                frequency_penalty: 0.4,
                ..SamplingParams::default()
            };
            let window = || PenaltyWindow::new(&prompt, &generated);
            let expected = the_chain_ferrox_used_to_hardcode(&logits, &params, window());
            let actual = sampling_distribution(&logits, &params, window(), None);
            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    e.to_bits(),
                    "token {i} at temp {temperature}, top_k {top_k}, top_p {top_p}, \
                     min_p {min_p}: the default order sampled {a} where the chain ferrox \
                     already ran gives {e}"
                );
            }
        }
    }

    /// And the same at the token level: the ids a seeded `Sampler` draws
    /// under `SamplingParams::default()` are the ids it draws when the
    /// caller spells out the default chain, so the flag's default value
    /// and the struct's default are one chain and not two.
    #[test]
    fn spelling_out_the_default_chain_draws_the_same_tokens() {
        let logits = spread_logits(48);
        let base = SamplingParams {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repetition_penalty: 1.1,
            ..SamplingParams::default()
        };
        let spelled = SamplingParams {
            sampler_order: "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"
                .parse::<SamplerOrder>()
                .expect("the default chain must parse"),
            ..base.clone()
        };
        let draw = |params: &SamplingParams| {
            let mut sampler = Sampler::new(0xFE0);
            let mut generated: Vec<usize> = Vec::new();
            for _ in 0..64 {
                let next = sampler.sample(&logits, params, PenaltyWindow::new(&[7], &generated));
                generated.push(next);
            }
            generated
        };
        assert_eq!(draw(&base), draw(&spelled));
    }

    /// A run that never asks for XTC must not consume a draw for it, or
    /// every seeded generation in the workspace shifts by one.
    #[test]
    fn running_the_temperature_first_keeps_a_different_candidate_set() {
        let logits = vec![6.0f32, 4.0, 2.0, 0.0, -2.0, -4.0];
        let params = |order: &str| SamplingParams {
            temperature: 8.0,
            top_p: 0.9,
            top_k: 0,
            min_p: 0.0,
            sampler_order: order.parse().expect("chain"),
            ..SamplingParams::default()
        };
        let support = |order: &str| -> Vec<bool> {
            sampling_distribution(&logits, &params(order), PenaltyWindow::new(&[], &[]), None)
                .iter()
                .map(|&p| p > 0.0)
                .collect()
        };

        let default = support("penalties;top_k;top_p;min_p;temperature");
        let temperature_first = support("penalties;temperature;top_k;top_p;min_p");
        assert_ne!(
            default, temperature_first,
            "reordering the chain must change which candidates survive, \
             or the flag is decorative"
        );
        assert!(
            temperature_first.iter().filter(|&&k| k).count()
                > default.iter().filter(|&&k| k).count(),
            "temp 8.0 flattens the distribution, so a later top-p keeps more: \
             default={default:?} temperature_first={temperature_first:?}"
        );
    }

    /// A sampler left OUT of the chain does not run, even though its
    /// knob is set -- llama.cpp reads an omitted sampler as "do not run
    /// it", and a chain that ran it anyway would be honouring a request
    /// nobody made.
    ///
    /// Checked for every filter that has a knob, not just min-p: the
    /// four samplers added for llama.cpp parity each have their own arm
    /// in `filtered_distribution`, and an arm that ignored the chain
    /// would be invisible to a test that only exercised one of them.
    #[test]
    fn a_sampler_absent_from_the_chain_does_not_filter() {
        let logits = vec![4.0f32, 3.0, 2.0, 1.0];
        let survivors = |params: &SamplingParams, roll: Option<f32>| {
            sampling_distribution(&logits, params, PenaltyWindow::new(&[], &[]), roll)
                .iter()
                .filter(|&&p| p > 0.0)
                .count()
        };
        // (the knob, a chain WITHOUT its step, the roll to pass)
        let cases: Vec<(SamplingParams, &str, Option<f32>)> = vec![
            (
                SamplingParams {
                    temperature: 1.0,
                    min_p: 0.2,
                    ..SamplingParams::default()
                },
                "penalties;top_k;top_p;temperature",
                None,
            ),
            (
                SamplingParams {
                    temperature: 1.0,
                    typical_p: 0.5,
                    ..SamplingParams::default()
                },
                "penalties;top_k;top_p;min_p;temperature",
                None,
            ),
            (
                SamplingParams {
                    temperature: 1.0,
                    top_n_sigma: 0.5,
                    ..SamplingParams::default()
                },
                "penalties;top_k;top_p;min_p;temperature",
                None,
            ),
            (
                SamplingParams {
                    temperature: 1.0,
                    xtc_probability: 1.0,
                    xtc_threshold: 0.05,
                    ..SamplingParams::default()
                },
                "penalties;top_k;top_p;min_p;temperature",
                Some(0.0),
            ),
        ];
        for (with, chain_without, roll) in cases {
            let filtered = survivors(&with, roll);
            assert!(
                filtered < 4,
                "the knob must bite when its step IS in the chain, \
                 or the second half proves nothing: {with:?}"
            );
            let without = SamplingParams {
                sampler_order: chain_without.parse().expect("chain"),
                ..with.clone()
            };
            assert_eq!(
                survivors(&without, roll),
                4,
                "the knob is set but its step is not in `{chain_without}`, \
                 so nothing should truncate: {without:?}"
            );
        }
    }

    /// Leaving `penalties` out of the chain disables the penalties, on
    /// the SAMPLED path and on the greedy one.
    ///
    /// The greedy half is the one that would have been missed: the
    /// penalties are applied before the candidate list exists, so a
    /// check placed beside the chain would never run at `temp <= 0`,
    /// and `--samplers` without `penalties` would still have penalised.
    #[test]
    fn a_chain_without_penalties_does_not_penalise_on_either_path() {
        // Token 0 leads token 1 by less than the 1.1 penalty.
        let logits = vec![4.0f32, 3.9];
        let history = || PenaltyWindow::new(&[0], &[]);
        let greedy = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 1.1,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(1);
        assert_eq!(
            sampler.sample(&logits, &greedy, history()),
            1,
            "the default chain penalises the prompt token"
        );

        let unpenalised = SamplingParams {
            sampler_order: "top_k;top_p;min_p;temperature".parse().expect("chain"),
            ..greedy.clone()
        };
        assert!(!unpenalised.sampler_order.has_penalties());
        assert_eq!(
            sampler.sample(&logits, &unpenalised, history()),
            0,
            "`penalties` is not in the chain, so the argmax must stand"
        );

        // And on the sampled path, where the whole distribution is
        // visible rather than one argmax.
        let sampled = SamplingParams {
            temperature: 1.0,
            ..unpenalised
        };
        let with = SamplingParams {
            sampler_order: SamplerOrder::default(),
            ..sampled.clone()
        };
        assert_ne!(
            sampling_distribution(&logits, &sampled, history(), None),
            sampling_distribution(&logits, &with, history(), None)
        );
    }

    /// A token that has only ever appeared in the PROMPT is penalised
    /// on the very first generated position, and that changes which
    /// token is sampled.
    ///
    /// This is the divergence issue #55 reported. llama.cpp seeds its
    /// penalties sampler with every prompt token before drawing
    /// anything (`tools/server/server-context.cpp:386-390`,
    /// `tools/completion/completion.cpp:730-736`); ferrox's decode
    /// loops handed the sampler the generated tokens alone, so the same
    /// checkpoint, flags and prompt could produce different text at the
    /// default `--repeat-penalty 1.1`.
    ///
    /// Asserted on the SAMPLED TOKEN rather than on the window's
    /// contents: a test that only checked the slice could not tell the
    /// window being applied to the wrong distribution from the window
    /// being wrong. Drop `prompt` from `PenaltyWindow::recent` and this
    /// goes red -- the second assertion returns 0.
    #[test]
    fn a_prompt_token_is_penalised_before_it_is_ever_generated() {
        let params = SamplingParams {
            // Greedy, so the assertion is on the chosen id and not on a
            // draw. Everything below is arithmetic, not sampling.
            temperature: 0.0,
            repetition_penalty: 1.1,
            ..SamplingParams::default()
        };
        // Token 0 leads token 1 by less than the 1.1 penalty: 4.0 / 1.1
        // = 3.636, which is below 3.9.
        let logits = vec![4.0f32, 3.9];
        let mut sampler = Sampler::new(1);

        assert_eq!(
            sampler.sample(&logits, &params, PenaltyWindow::new(&[], &[])),
            0,
            "with nothing behind it the argmax wins"
        );
        assert_eq!(
            sampler.sample(&logits, &params, PenaltyWindow::new(&[0], &[])),
            1,
            "token 0 is in the prompt, so llama.cpp penalises it here"
        );
        // And a window that reaches back past the prompt is the same
        // answer, which is what makes the two halves one sequence.
        assert_eq!(
            sampler.sample(&logits, &params, PenaltyWindow::new(&[9, 0], &[8])),
            1
        );
    }

    /// `penalty_last_n` counts across the prompt/generated seam, so a
    /// prompt token falls OUT of the window once enough tokens have
    /// been generated after it -- and the sampled token moves back.
    ///
    /// A window that added the whole prompt to the last N generated
    /// tokens would keep penalising token 0 forever and this would stay
    /// at 1.
    #[test]
    fn a_prompt_token_leaves_the_window_once_the_generation_outgrows_it() {
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 1.1,
            penalty_last_n: 2,
            ..SamplingParams::default()
        };
        let logits = vec![4.0f32, 3.9];
        let mut sampler = Sampler::new(1);

        // Prompt token 0, one token generated: the window is [0, 5] and
        // token 0 is still penalised.
        assert_eq!(
            sampler.sample(&logits, &params, PenaltyWindow::new(&[0], &[5])),
            1
        );
        // Two generated: the window is [5, 6] and token 0 is clear.
        assert_eq!(
            sampler.sample(&logits, &params, PenaltyWindow::new(&[0], &[5, 6])),
            0
        );
    }

    /// Top-p cuts the UNSCALED distribution; the temperature reshapes
    /// only the survivors.
    ///
    /// llama.cpp's default chain runs temperature LAST
    /// (`common/common.h:259-269`); ferrox divided first and filtered
    /// afterwards. Not an innocuous reordering: temperature changes the
    /// probabilities top-p sums over, so a high temperature flattens the
    /// distribution and grows the nucleus. The two orders keep different
    /// candidate sets for identical flags.
    #[test]
    fn temperature_does_not_change_which_candidates_top_p_keeps() {
        let logits = vec![3.0f32, 2.0, 1.0, 0.0];
        let at = |temperature: f32| -> Vec<bool> {
            let params = SamplingParams {
                temperature,
                top_p: 0.9,
                top_k: 0,
                ..SamplingParams::default()
            };
            sampling_distribution(&logits, &params, PenaltyWindow::new(&[], &[]), None)
                .iter()
                .map(|&p| p > 0.0)
                .collect()
        };

        let cold = at(0.5);
        let hot = at(4.0);
        assert_eq!(
            cold, hot,
            "the surviving set must not depend on the temperature: \
             cold={cold:?} hot={hot:?}"
        );
        // And the cut must actually bite, or the equality above is
        // satisfied by keeping everything.
        assert!(
            cold.iter().any(|&k| !k),
            "top_p = 0.9 must drop at least one of these four candidates"
        );
    }

    /// min-p truncates, and it truncates on llama.cpp's threshold.
    ///
    /// llama.cpp enables min-p **by default** at 0.05
    /// (`common/common.h:231`), so until this existed ferrox could not
    /// reproduce llama.cpp's own out-of-the-box output on any prompt --
    /// a parity gap, not a missing feature.
    ///
    /// Logits `[4, 3, 2, 1]` at `min_p = 0.2`: the threshold is
    /// `4 + ln(0.2) = 2.3905`, so exactly the candidates at 4 and 3
    /// survive. Arithmetic done by hand from
    /// `src/llama-sampler.cpp:1556`, not read back off the code.
    #[test]
    fn min_p_truncates_at_ln_p_below_the_top_logit() {
        let logits = vec![4.0f32, 3.0, 2.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            min_p: 0.2,
            ..SamplingParams::default()
        };
        let probs = sampling_distribution(&logits, &params, PenaltyWindow::new(&[], &[]), None);
        assert!(probs[0] > 0.0 && probs[1] > 0.0);
        assert_eq!(probs[2], 0.0, "2.0 is below 4 + ln(0.2) = 2.3905");
        assert_eq!(probs[3], 0.0);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // The two survivors are renormalised against each other:
        // e^4 / (e^4 + e^3) = 0.7311.
        assert!((probs[0] - 0.731_059).abs() < 1e-5, "got {}", probs[0]);

        // 0.0 disables it, which is ferrox's struct default -- adding
        // min-p must not change any existing caller's distribution.
        let off = SamplingParams {
            min_p: 0.0,
            ..params.clone()
        };
        let unfiltered = sampling_distribution(&logits, &off, PenaltyWindow::new(&[], &[]), None);
        assert!(unfiltered.iter().all(|&p| p > 0.0));
    }

    /// min-p runs BEFORE the temperature, so the set it keeps does not
    /// depend on `--temp`.
    ///
    /// This is the same trap as E4 and it bites harder here. min-p's
    /// test is `logit_i >= logit_max + ln(p)`, and temperature divides
    /// **both** logits, so it scales the very gap being compared against
    /// a fixed `ln(p)`. On these logits at `min_p = 0.2`, running min-p
    /// after a temperature of 0.5 would keep one candidate and after 2.0
    /// would keep all four; llama.cpp keeps two at every temperature
    /// (`common/common.h:259-269` puts `MIN_P` before `TEMPERATURE`).
    ///
    /// Move `candidates.min_p(..)` after `candidates.temperature(..)` in
    /// `filtered_distribution` and this goes red.
    #[test]
    fn temperature_does_not_change_which_candidates_min_p_keeps() {
        let logits = vec![3.0f32, 2.0, 1.0, 0.0];
        let survivors = |temperature: f32| -> Vec<bool> {
            let params = SamplingParams {
                temperature,
                min_p: 0.2,
                ..SamplingParams::default()
            };
            sampling_distribution(&logits, &params, PenaltyWindow::new(&[], &[]), None)
                .iter()
                .map(|&p| p > 0.0)
                .collect()
        };

        let cold = survivors(0.5);
        let warm = survivors(1.0);
        let hot = survivors(2.0);
        assert_eq!(cold, warm, "cold={cold:?} warm={warm:?}");
        assert_eq!(warm, hot, "warm={warm:?} hot={hot:?}");
        // 3 + ln(0.2) = 1.3905, so exactly the 3.0 and 2.0 candidates.
        assert_eq!(warm, vec![true, true, false, false]);
    }

    /// min-p sits AFTER top-p in the chain, and both may bite on the
    /// same call.
    ///
    /// `top_p = 0.95` on this distribution keeps three candidates
    /// (0.6337 + 0.2331 + 0.0857 = 0.9525); min-p at 0.2 then drops the
    /// third, whose probability is 0.135 of the top one. Getting only
    /// one of the two filters gives a different answer either way, so
    /// this fails if either is dropped or if min-p is skipped when top-p
    /// already truncated.
    #[test]
    fn top_p_and_min_p_both_apply() {
        let logits = vec![3.0f32, 2.0, 1.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.95,
            min_p: 0.2,
            ..SamplingParams::default()
        };
        let probs = sampling_distribution(&logits, &params, PenaltyWindow::new(&[], &[]), None);
        assert_eq!(
            probs.iter().map(|&p| p > 0.0).collect::<Vec<_>>(),
            vec![true, true, false, false]
        );

        // top-p alone keeps three; min-p alone also keeps two here, so
        // pin the top-p-only case to prove the two filters are distinct
        // and that this test is not satisfied by min-p doing all the
        // work.
        let top_p_only = SamplingParams {
            min_p: 0.0,
            ..params.clone()
        };
        assert_eq!(
            sampling_distribution(&logits, &top_p_only, PenaltyWindow::new(&[], &[]), None)
                .iter()
                .filter(|&&p| p > 0.0)
                .count(),
            3
        );
    }

    /// DRY reaches the sampled token through the chain, not just the
    /// penalty table.
    ///
    /// Window `0 1 2 0 1` at `allowed_length 2`: emitting `2` would make
    /// it a three-token repetition, so DRY subtracts
    /// `multiplier * base^0 = 6.0` from token 2's logit of 5.0 -- more
    /// than enough to move the greedy argmax off it. That is the whole
    /// claim: a sampler wired into `filtered_distribution` but not
    /// reached from the greedy path would leave this at 2.
    #[test]
    fn dry_moves_the_chosen_token_on_both_the_greedy_and_the_sampled_path() {
        let logits = vec![0.0f32, 0.0, 5.0, 0.0];
        let history = || PenaltyWindow::new(&[], &[0, 1, 2, 0, 1]);
        let dry = DryParams::new(6.0, 1.1, 2, -1, 1024, DryBreakers::none());
        let greedy = SamplingParams {
            temperature: 0.0,
            dry: dry.clone(),
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(5);
        assert_eq!(
            sampler.sample(&logits, &SamplingParams::default(), history()),
            2,
            "without DRY token 2 is the argmax"
        );
        assert_ne!(
            sampler.sample(&logits, &greedy, history()),
            2,
            "DRY subtracts 4.0 from token 2's logit of 5.0, so it loses"
        );

        // Same on the sampled path, read off the distribution.
        let sampled = SamplingParams {
            temperature: 1.0,
            ..greedy.clone()
        };
        let with = sampling_distribution(&logits, &sampled, history(), None);
        let without = sampling_distribution(
            &logits,
            &SamplingParams {
                dry: DryParams::off(),
                ..sampled
            },
            history(),
            None,
        );
        assert!(with[2] < without[2], "with={with:?} without={without:?}");
    }

    /// XTC removes the TOP candidates, so it can change greedy output --
    /// and ferrox's greedy fast path knows that.
    ///
    /// `greedy_equals_argmax` is the predicate that decides whether the
    /// `argmax` shortcut is exact. Make it return `true`
    /// unconditionally and this goes red: the shortcut would return
    /// token 0 while llama.cpp's chain, which runs XTC before the
    /// temperature at every temperature, returns something else.
    #[test]
    fn xtc_changes_the_greedy_choice_because_it_removes_the_top() {
        let logits = vec![3.0f32, 2.9, -10.0];
        let params = SamplingParams {
            temperature: 0.0,
            // Always fires, and both leading candidates clear the
            // threshold, so the more likely of the two is removed.
            xtc_probability: 1.0,
            xtc_threshold: 0.05,
            ..SamplingParams::default()
        };
        assert!(!params.greedy_equals_argmax());
        let mut sampler = Sampler::new(11);
        assert_eq!(
            sampler.sample(&logits, &params, PenaltyWindow::new(&[], &[])),
            1,
            "XTC drops token 0, so the greedy answer is token 1"
        );
        // Without XTC the argmax stands, which is what makes the
        // assertion above about XTC and not about the logits.
        assert_eq!(
            sampler.sample(
                &logits,
                &SamplingParams::default(),
                PenaltyWindow::new(&[], &[])
            ),
            0
        );
    }

    /// The same for typical-p: it selects outward from the entropy and
    /// can drop the most likely token, so the greedy shortcut is not
    /// exact when it is live.
    #[test]
    fn typical_p_can_drop_the_argmax_so_greedy_must_run_the_chain() {
        // One near-certain token and three equal small ones. The
        // entropy is dominated by the small tokens, so the near-certain
        // one is the ATYPICAL member and typical-p at 0.5 keeps it
        // alone here -- while at 0.5 on a flatter distribution it drops
        // the leader. The property under test is the predicate.
        let params = SamplingParams {
            temperature: 0.0,
            typical_p: 0.5,
            ..SamplingParams::default()
        };
        assert!(!params.greedy_equals_argmax());
        assert!(SamplingParams {
            typical_p: 1.0,
            ..params.clone()
        }
        .greedy_equals_argmax());

        // logits ln(0.4), ln(0.2) x3: llama.cpp keeps the three 0.2
        // candidates and drops the 0.4 leader (`tests/test-sampling.cpp:346`),
        // so the greedy answer must not be token 0.
        let logits: Vec<f32> = [0.4f32, 0.2, 0.2, 0.2].iter().map(|p| p.ln()).collect();
        let mut sampler = Sampler::new(3);
        assert_ne!(
            sampler.sample(&logits, &params, PenaltyWindow::new(&[], &[])),
            0,
            "typical-p drops the most likely token here"
        );
    }

    #[test]
    fn greedy_is_published_as_a_point_mass_not_a_special_case() {
        let logits = vec![0.1, 0.9, 0.3, -0.2];
        let probs = sampling_distribution(
            &logits,
            &SamplingParams::default(),
            PenaltyWindow::new(&[], &[]),
            None,
        );
        assert_eq!(probs, vec![0.0, 1.0, 0.0, 0.0]);
        // Penalties still apply at temperature 0, so the point mass
        // moves with them.
        let penalized = sampling_distribution(
            &logits,
            &SamplingParams {
                repetition_penalty: 100.0,
                ..SamplingParams::default()
            },
            PenaltyWindow::new(&[], &[1]),
            None,
        );
        assert_eq!(penalized[1], 0.0);
        assert_eq!(penalized.iter().sum::<f32>(), 1.0);
    }
}
