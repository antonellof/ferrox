//! When an argmax may stand in for the whole sampler chain, and over
//! WHICH logits it may stand in.
//!
//! # The two questions, which are not the same question
//!
//! There are two callers, and they hand an argmax two different vectors:
//!
//! * [`super::greedy_choice`] argmaxes scores the penalties have
//!   ALREADY been applied to ([`super::penalties::apply_history_penalties`]
//!   runs first, on the greedy path as well as the sampled one). It
//!   needs to know whether anything LEFT in the chain can move the
//!   argmax.
//! * A backend that folds `final_norm + lm_head + argmax` into its
//!   decode stack argmaxes the **raw** logits on the device and returns
//!   one token id. Nothing on the host ever sees the vocabulary, so the
//!   penalties never happen at all. It needs to know whether the WHOLE
//!   chain, penalties included, can move the argmax.
//!
//! Until GitHub issue #170 those were one predicate,
//! `SamplingParams::greedy_equals_argmax`, whose body answered the first
//! question and whose name and Metal callers asked the second. It tested
//! XTC, typical-p and DRY and did not test the penalties, so on Metal at
//! `--ngl 99 --temp 0` with this project's default `--repeat-penalty
//! 1.1` the fold silently dropped the repetition penalty and returned a
//! different token from the host path and from the CPU reference.
//!
//! Measured on an M2 Pro, `Llama-3.2-3B-Instruct-Q4_K_M --ngl 99
//! --temp 0 --no-cnv`, 64 tokens, md5 of the completion:
//!
//! | path | `--repeat-penalty` | md5 |
//! |---|---|---|
//! | Metal, fold on | 1.1 (default) | `85ef7c43…` |
//! | Metal, fold off | 1.1 (default) | `36c6ae0d…` |
//! | CPU reference | 1.1 (default) | `36c6ae0d…` |
//! | Metal, fold on | 1.0 | `85ef7c43…` |
//! | Metal, fold off | 1.0 | `85ef7c43…` |
//!
//! The folded answer at 1.1 is bit-identical to the answer at 1.0, which
//! is what "the penalty never ran" looks like, and the unfolded answer
//! agrees with the CPU reference exactly.
//!
//! # Why this is a table and not two `&&` chains
//!
//! Because the thing that went wrong is the repo's dominant defect
//! shape: a predicate hand-listing a SUBSET of the sampler, with nothing
//! tying the list to the sampler. So the classification is one
//! EXHAUSTIVE `match` over [`ChainStep`] and one EXHAUSTIVE destructure
//! of [`SamplingParams`] with no `..`. A step added to the chain, or a
//! knob added to the params, does not compile until somebody says what
//! it does to an argmax.
//!
//! The two predicates then differ by exactly one clause -- whether
//! [`ChainStep::Penalties`] is excused because the caller already ran it
//! -- which is the whole of issue #170 written down in one line.

use super::SamplingParams;
use crate::sampler_order::ChainStep;

/// What one chain step does to the argmax of the scores it is handed.
///
/// Three states rather than a `bool`, because "cannot change any score"
/// and "changes scores but never which one is largest" are different
/// facts, and only the first says the step is switched off. Reading them
/// as one would make the default chain look live and the live chain look
/// harmless, depending on which way the collapse went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StepEffect {
    /// Configured to its own no-op value: it cannot change any score.
    Inert,
    /// Live, and keeps the maximum by construction: it either filters
    /// candidates by a threshold measured FROM the maximum, or scales
    /// every score monotonically.
    KeepsTheMaximum,
    /// Live, and can change which token the argmax picks -- either by
    /// removing the maximum from the candidate list or by moving a
    /// logit past it.
    MovesTheArgmax,
}

/// What `step` would do to an argmax, given `params`.
///
/// Chain MEMBERSHIP is the caller's question, not this function's: the
/// callers walk `params.sampler_order.steps()`, and a step a caller left
/// out of `--samplers` is a step llama.cpp does not run.
pub(crate) fn step_effect(step: ChainStep, params: &SamplingParams) -> StepEffect {
    // EXHAUSTIVE, with NO `..`. A knob added to `SamplingParams` stops
    // this crate compiling here until someone states what it does to an
    // argmax. That is the enforcement issue #170 was missing: the old
    // predicate hand-listed three of the chain's nine steps and nothing
    // noticed that the penalties were not among them.
    //
    // A field bound to `_` is one whose verdict is written in the arm
    // that consumes it; binding it is what keeps the pattern exhaustive.
    let SamplingParams {
        temperature,
        top_p,
        min_p,
        top_k,
        typical_p,
        top_n_sigma,
        // Read together through `SamplingParams::xtc_can_fire`, which
        // exists precisely so the two conditions are not restated. See
        // the `Xtc` arm.
        xtc_probability: _,
        xtc_threshold: _,
        dry,
        repetition_penalty,
        penalty_last_n,
        presence_penalty,
        frequency_penalty,
        // Membership is the caller's loop; see this function's doc.
        sampler_order: _,
    } = params;

    match step {
        // The repetition / presence / frequency penalties move
        // individual logits, so they can move which logit is the
        // maximum. This is the arm the old predicate did not have.
        //
        // The switched-off conditions restate `apply_history_penalties`'s
        // own early returns, and
        // `tests::the_inert_verdict_matches_what_the_penalty_step_actually_does`
        // walks a matrix asserting the two agree, rather than trusting
        // this copy of them.
        ChainStep::Penalties => {
            let neutral =
                *repetition_penalty == 1.0 && *presence_penalty == 0.0 && *frequency_penalty == 0.0;
            if *penalty_last_n == 0 || neutral {
                StepEffect::Inert
            } else {
                StepEffect::MovesTheArgmax
            }
        }
        // DRY subtracts from the logits of tokens that would extend a
        // repetition, so like the penalties it can move the maximum.
        ChainStep::Dry => {
            if dry.is_enabled() {
                StepEffect::MovesTheArgmax
            } else {
                StepEffect::Inert
            }
        }
        // Keeps candidates within `n` standard deviations BELOW the
        // maximum, so the maximum is always one of them.
        ChainStep::TopNSigma => {
            if *top_n_sigma <= 0.0 {
                StepEffect::Inert
            } else {
                StepEffect::KeepsTheMaximum
            }
        }
        // Keeps the `k` most likely, which starts at the maximum.
        ChainStep::TopK => {
            if *top_k == 0 {
                StepEffect::Inert
            } else {
                StepEffect::KeepsTheMaximum
            }
        }
        // Selects OUTWARD from the distribution's entropy and can drop
        // the most likely token -- llama.cpp's own test case
        // `test_typical({0.4, 0.2, 0.2, 0.2}, {0.2, 0.2, 0.2}, 0.5)`
        // (`tests/test-sampling.cpp:346`) drops it.
        ChainStep::TypP => {
            if *typical_p >= 1.0 {
                StepEffect::Inert
            } else {
                StepEffect::MovesTheArgmax
            }
        }
        // Accumulates probability from the most likely downward and
        // keeps at least one candidate, so the maximum always survives.
        ChainStep::TopP => {
            if *top_p >= 1.0 {
                StepEffect::Inert
            } else {
                StepEffect::KeepsTheMaximum
            }
        }
        // A threshold expressed as a fraction OF the maximum, which the
        // maximum meets by construction.
        ChainStep::MinP => {
            if *min_p <= 0.0 {
                StepEffect::Inert
            } else {
                StepEffect::KeepsTheMaximum
            }
        }
        // Removes the TOP candidates, by construction.
        ChainStep::Xtc => {
            if params.xtc_can_fire() {
                StepEffect::MovesTheArgmax
            } else {
                StepEffect::Inert
            }
        }
        // A positive temperature divides every logit by the same
        // positive number, which is monotone. `temp <= 0` is
        // llama.cpp's greedy collapse (`src/llama-sampler.cpp:271-286`),
        // which sets every logit but the maximum to `-inf`. Neither can
        // change WHICH index is maximal, so temperature is never a
        // reason to refuse a fold -- and it is never `Inert` either,
        // since there is no value at which it stops touching scores.
        ChainStep::Temperature => {
            debug_assert!(!temperature.is_nan(), "a NaN temperature is not a chain");
            StepEffect::KeepsTheMaximum
        }
    }
}

impl SamplingParams {
    /// True when no step LEFT in the chain can move the argmax of the
    /// scores it is handed -- scores the penalties have ALREADY been
    /// applied to.
    ///
    /// This is [`super::greedy_choice`]'s question and only its
    /// question. It is called per token, so it is a walk over at most
    /// [`crate::sampler_order::SamplerName::ALL`]`.len()` steps with no
    /// allocation, not a candidate list.
    ///
    /// A device fold must ask [`Self::greedy_equals_raw_argmax`]
    /// instead: this one excuses the penalties because its caller
    /// already ran them, and a device that argmaxes raw logits has not.
    pub fn chain_keeps_the_argmax(&self) -> bool {
        self.sampler_order.steps().iter().all(|&step| {
            step == ChainStep::Penalties || step_effect(step, self) != StepEffect::MovesTheArgmax
        })
    }

    /// True when the argmax of the **raw** logits is the token the whole
    /// sampler chain would choose, penalties included.
    ///
    /// This is the question a backend folding `lm_head + argmax` into
    /// its decode stack has to ask, because the fold returns one token
    /// id and the host never sees a vocabulary: every step of the chain
    /// is skipped, not just the candidate-list filters.
    ///
    /// **Three readers**, because the alternative is this repo's
    /// dominant defect: `ferrox_cli::run`'s `needs_vocab_logits`,
    /// `ferrox_server::generate`'s, and through them the Metal fold's
    /// own guard in `ferrox_metal::greedy_fold`.
    ///
    /// Note what this costs: with the CLI's default `--repeat-penalty
    /// 1.1` the answer is `false`, so the Metal fold does not fire in
    /// the default configuration. That is deliberate -- see issue #170
    /// and this module's header for the numbers. `--repeat-penalty 1.0`
    /// or `--repeat-last-n 0` gets it back.
    pub fn greedy_equals_raw_argmax(&self) -> bool {
        self.sampler_order
            .steps()
            .iter()
            .all(|&step| step_effect(step, self) != StepEffect::MovesTheArgmax)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dry::{DryBreakers, DryParams};
    use crate::penalty_window::PenaltyWindow;
    use crate::sampler_order::SamplerOrder;
    use crate::sampling::Sampler;

    /// The CLI's defaults, which are the configuration the bug was live
    /// in. Built here rather than imported because `ferrox-models` does
    /// not depend on `ferrox-cli`. `ferrox_cli::run::tests::
    /// the_default_flags_forbid_the_metal_greedy_argmax_fold` asserts
    /// the real `default_value_t`s resolve to this.
    fn cli_defaults() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            top_p: 0.95,
            min_p: 0.05,
            top_k: 40,
            repetition_penalty: 1.1,
            penalty_last_n: 64,
            ..SamplingParams::default()
        }
    }

    /// GitHub issue #170, at the predicate: the default repetition
    /// penalty forbids a raw-logit argmax, and `1.0` permits it.
    ///
    /// This is what shipped broken. `--repeat-penalty` defaults to 1.1
    /// in this project (llama.cpp's is 1.0), the Metal fold gate read a
    /// predicate that did not test the penalties, and a greedy `--ngl
    /// 99` run returned a token the host sampler would not have chosen.
    ///
    /// Sabotage: drop the `ChainStep::Penalties` arm's
    /// `MovesTheArgmax` for `Inert`; the first assertion goes red, and
    /// so does every gate test in `ferrox-cli` and `ferrox-server`.
    #[test]
    fn the_default_repetition_penalty_forbids_a_raw_argmax_fold() {
        let defaults = cli_defaults();
        assert!(
            !defaults.greedy_equals_raw_argmax(),
            "a device argmax over raw logits skips the repetition penalty"
        );
        // ... and the chain BEHIND the penalties is still argmax-safe,
        // which is what makes this assertion about the penalties and not
        // about top-k or min-p.
        assert!(defaults.chain_keeps_the_argmax());

        // The proof that the penalties are the whole difference: the
        // same flags at `--repeat-penalty 1.0` fold again.
        assert!(SamplingParams {
            repetition_penalty: 1.0,
            ..defaults.clone()
        }
        .greedy_equals_raw_argmax());
        // And so does `--repeat-last-n 0`, llama.cpp's other off switch.
        assert!(SamplingParams {
            penalty_last_n: 0,
            ..defaults.clone()
        }
        .greedy_equals_raw_argmax());
        // The OpenAI-shaped penalties are the same statement.
        for moved in [
            SamplingParams {
                repetition_penalty: 1.0,
                presence_penalty: 0.5,
                ..defaults.clone()
            },
            SamplingParams {
                repetition_penalty: 1.0,
                frequency_penalty: 0.5,
                ..defaults.clone()
            },
        ] {
            assert!(
                !moved.greedy_equals_raw_argmax(),
                "{moved:?} moves logits before the argmax"
            );
        }
        // A chain that does not NAME `penalties` does not penalise, so
        // the fold is sound however the knobs are set.
        assert!(SamplingParams {
            sampler_order: SamplerOrder::from_names(["top_k", "top_p", "min_p", "temperature"])
                .expect("a chain without penalties"),
            ..defaults
        }
        .greedy_equals_raw_argmax());
    }

    /// The predicate's actual contract, exercised through the sampler
    /// rather than asserted about itself: when
    /// [`SamplingParams::greedy_equals_raw_argmax`] says yes, the argmax
    /// of the RAW logits is the token the chain chooses; when it says
    /// no, this particular case proves it was right to.
    ///
    /// The penalty case is the reproduction: token 0 leads by 0.2, the
    /// history contains token 0, and `1.1` divides it below token 1. A
    /// device fold returns 0; the sampler returns 1; the CPU reference
    /// returns 1.
    ///
    /// Sabotage: as above. With `Penalties => Inert` the predicate says
    /// the fold is sound and the `assert_ne!` below shows it is not.
    #[test]
    fn a_penalty_that_moves_the_argmax_is_one_the_fold_must_not_skip() {
        let logits = vec![4.0f32, 3.8, 1.0];
        let history = || PenaltyWindow::new(&[], &[0]);
        let params = SamplingParams {
            repetition_penalty: 1.1,
            ..cli_defaults()
        };

        // 4.0 / 1.1 = 3.636, which is below 3.8.
        let chosen = Sampler::new(7).sample(&logits, &params, history());
        assert_eq!(chosen, 1, "the penalty demotes the raw argmax");
        assert_ne!(
            chosen,
            super::super::argmax(&logits),
            "raw argmax and the chain's answer differ, so a fold is wrong here"
        );
        assert!(!params.greedy_equals_raw_argmax());

        // Neutralise the penalty and the two agree, which is the same
        // proof the `--repeat-penalty 1.0` row of the table in this
        // module's header carries.
        let neutral = SamplingParams {
            repetition_penalty: 1.0,
            ..params
        };
        assert!(neutral.greedy_equals_raw_argmax());
        assert_eq!(
            Sampler::new(7).sample(&logits, &neutral, history()),
            super::super::argmax(&logits)
        );
    }

    /// Params under which `step` is LIVE, one per step.
    ///
    /// EXHAUSTIVE, no `_` arm: a step added to [`ChainStep`] does not
    /// compile until someone supplies a configuration that switches it
    /// on, which is what stops the next sampler being forgotten the way
    /// the penalties were.
    fn live_exemplar(step: ChainStep) -> SamplingParams {
        let base = SamplingParams::default();
        match step {
            ChainStep::Penalties => SamplingParams {
                repetition_penalty: 1.1,
                ..base
            },
            ChainStep::Dry => SamplingParams {
                dry: DryParams::new(6.0, 1.1, 2, -1, 1024, DryBreakers::none()),
                ..base
            },
            ChainStep::TopNSigma => SamplingParams {
                top_n_sigma: 1.0,
                ..base
            },
            ChainStep::TopK => SamplingParams { top_k: 40, ..base },
            ChainStep::TypP => SamplingParams {
                typical_p: 0.5,
                ..base
            },
            ChainStep::TopP => SamplingParams { top_p: 0.9, ..base },
            ChainStep::MinP => SamplingParams {
                min_p: 0.05,
                ..base
            },
            ChainStep::Xtc => SamplingParams {
                xtc_probability: 1.0,
                xtc_threshold: 0.1,
                ..base
            },
            // Temperature has no off switch; see its arm in
            // `step_effect`.
            ChainStep::Temperature => SamplingParams {
                temperature: 0.8,
                ..base
            },
        }
    }

    /// Every step of the chain has a verdict, every step is `Inert` at
    /// the struct's own defaults, and every exemplar switches its step
    /// on.
    ///
    /// The three halves close the loop the old predicate left open. It
    /// hand-listed three steps of nine, so a step could be added to the
    /// chain and never appear in it; here `step_effect`'s `match` and
    /// `live_exemplar`'s must both name every step, and this walks
    /// `ChainStep::all()` -- which is derived from the name table -- to
    /// prove neither list is a private one.
    ///
    /// Sabotage: add `top_k: 40` to `SamplingParams::default`; the
    /// `Inert` half goes red. Or make the `TopK` arm of `step_effect`
    /// return `Inert` unconditionally; the exemplar half does.
    #[test]
    fn every_chain_step_is_classified_and_neutral_at_the_struct_defaults() {
        let neutral = SamplingParams::default();
        let steps = ChainStep::all();
        assert_eq!(steps.len(), 9, "the chain gained or lost a step");
        for step in steps {
            // Temperature is the one step with no neutral value: it
            // always scales, and always monotonically.
            let expected_at_rest = if step == ChainStep::Temperature {
                StepEffect::KeepsTheMaximum
            } else {
                StepEffect::Inert
            };
            assert_eq!(
                step_effect(step, &neutral),
                expected_at_rest,
                "{step:?} is not neutral at the struct's defaults"
            );
            assert_ne!(
                step_effect(step, &live_exemplar(step)),
                StepEffect::Inert,
                "{step:?}'s exemplar does not switch it on, so its \
                 classification is never exercised"
            );
        }
        // A neutral chain folds, which is the statement the whole table
        // exists to make safely.
        assert!(neutral.greedy_equals_raw_argmax());
        assert!(neutral.chain_keeps_the_argmax());
    }

    /// Every step classified `MovesTheArgmax` refuses the fold, and no
    /// step classified otherwise refuses it.
    ///
    /// This is the join between the table and the two predicates. A
    /// verdict nothing reads would be a gate that cannot fire.
    ///
    /// Sabotage: make `greedy_equals_raw_argmax` ignore
    /// `ChainStep::Penalties`, i.e. restore the old body; the
    /// `Penalties` row goes red.
    #[test]
    fn a_step_that_moves_the_argmax_is_a_step_that_refuses_the_fold() {
        for step in ChainStep::all() {
            let live = live_exemplar(step);
            let moves = step_effect(step, &live) == StepEffect::MovesTheArgmax;
            assert_eq!(
                !live.greedy_equals_raw_argmax(),
                moves,
                "{step:?}: the classification and the fold gate disagree"
            );
            // And the post-penalty predicate is the same statement with
            // exactly one step excused, which is the whole of #170.
            let expected_after_penalties = if step == ChainStep::Penalties {
                true
            } else {
                !moves
            };
            assert_eq!(
                live.chain_keeps_the_argmax(),
                expected_after_penalties,
                "{step:?}: the post-penalty predicate excuses the wrong step"
            );
        }
    }

    /// The `Inert` verdict for the penalties is the SAME condition
    /// `apply_history_penalties` short-circuits on.
    ///
    /// Two copies of "the penalties are switched off" is the defect this
    /// module is a fix for, one level down: if the classification said
    /// inert where the penalty step still moved a logit, the fold would
    /// be permitted over scores the host would have changed. Rather than
    /// derive one from the other -- the penalty step's early returns are
    /// entangled with its window scan -- this walks a matrix and asserts
    /// they agree, so a change to either goes red.
    ///
    /// Sabotage: drop `|| neutral` from the `Penalties` arm; the
    /// all-neutral rows go red.
    #[test]
    fn the_inert_verdict_matches_what_the_penalty_step_actually_does() {
        use crate::sampling::penalties::apply_history_penalties;

        let logits = vec![4.0f32, 3.8, -1.0];
        for &rep in &[1.0f32, 1.1] {
            for &pres in &[0.0f32, 0.5] {
                for &freq in &[0.0f32, 0.5] {
                    for &last_n in &[0usize, 64] {
                        let params = SamplingParams {
                            repetition_penalty: rep,
                            presence_penalty: pres,
                            frequency_penalty: freq,
                            penalty_last_n: last_n,
                            ..SamplingParams::default()
                        };
                        let mut scores = logits.clone();
                        apply_history_penalties(
                            &mut scores,
                            &params,
                            PenaltyWindow::new(&[], &[0, 1]),
                        );
                        let untouched = scores == logits;
                        let inert = step_effect(ChainStep::Penalties, &params) == StepEffect::Inert;
                        assert_eq!(
                            inert,
                            untouched,
                            "rep={rep} pres={pres} freq={freq} last_n={last_n}: the \
                             classification says inert={inert} and the penalty step \
                             {}",
                            if untouched {
                                "changed nothing"
                            } else {
                                "changed the scores"
                            }
                        );
                    }
                }
            }
        }
    }
}
