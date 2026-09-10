//! Whether a Metal decode step folds `final_norm + lm_head + argmax`
//! into its own command buffer and returns one token id, and WHERE that
//! decision is kept.
//!
//! # The decision
//!
//! At `temperature <= 0`, with nothing in the chain that needs to see a
//! vocabulary before a token is chosen, the argmax can be computed on
//! the device. The stack then downloads four bytes instead of
//! `vocab_size` floats. `ferrox-cli` and `ferrox-server` each decide
//! this per request and announce it with [`set_metal_greedy_argmax`];
//! `ferrox-models` reads it back with [`metal_greedy_argmax_active`] and
//! turns it into the `argmax_only` parameter the stack actually takes.
//!
//! # Why this is its own module (GitHub issue #166)
//!
//! Because that announcement is THREAD-LOCAL, and the decode step it
//! configures need not run on the announcing thread.
//!
//! Measured on an M2 Pro, `Llama-3.2-3B-Instruct-Q4_K_M`, `--ngl 99`,
//! `--temp 0 --no-cnv`, 32 tokens: driving the same build's forward pass
//! from a rayon worker instead of the process's main thread changes the
//! completion, deterministically, from the tenth token or so. The flag
//! is set on the main thread, read as `false` on the worker, and the
//! stack silently takes its other path.
//!
//! Bisected with two temporary kill switches, one per suspect:
//!
//! | fold | resident activation | thread | answer |
//! |---|---|---|---|
//! | on | on | main | A |
//! | on | on | worker | B |
//! | off | on | either | B |
//! | off | off | either | B |
//!
//! So the resident-activation hand-off (`crate::resident_act`) is
//! value-neutral and this flag is the whole of the difference.
//!
//! # Why the two paths did not agree, which was the deeper defect
//!
//! They should not differ at all. Both compute `lm_head` with the same
//! `MatvecLaunch` through the same `encode_matvec`, over an activation
//! that is bit-identical either way. What differed is what happens AFTER
//! the logits: the folded path takes a device argmax of the RAW logits,
//! and the unfolded path hands the vocabulary to the host sampler, which
//! applies the repetition penalties first.
//!
//! `--repeat-penalty` defaults to 1.1 in this project (llama.cpp's is
//! 1.0), so on a default `--temp 0` run the penalty is live and the fold
//! dropped it. The gate that permits the fold,
//! `SamplingParams::greedy_equals_argmax`, tested XTC, typical-p and DRY
//! and did NOT test the penalties: two structures that must agree about
//! when an argmax is the answer, with nothing enforcing it.
//!
//! Fixed in `ferrox-models` (GitHub issue #170), where that gate lives.
//! It is now two predicates rather than one -- `chain_keeps_the_argmax`
//! for a host that has already penalised, `greedy_equals_raw_argmax` for
//! a device that has not -- built from an exhaustive classification of
//! every chain step and every `SamplingParams` field. The consequence
//! for THIS module is worth stating plainly, because it moves Metal
//! decode numbers: **at the CLI's default `--repeat-penalty 1.1` the
//! fold no longer fires**. `--repeat-penalty 1.0` or `--repeat-last-n 0`
//! gets it back, and so would a fold that masked the penalty window on
//! the device. This module can still only say which flag selects which
//! path.
//!
//! # What is fixed here and what is not
//!
//! Fixed: the setting is now [`GreedyFold`], a value with three states
//! rather than a bare `bool`, so "no caller on this thread said
//! anything" is no longer spelled the same way as "this caller wants no
//! fold"; and it can be CAPTURED on the deciding thread and ADOPTED on
//! the thread that runs the step ([`greedy_fold_setting`],
//! [`adopt_greedy_fold`]).
//!
//! Not fixed, deliberately: this module does NOT resolve an unset thread
//! by looking at what other threads chose. It cannot. `ferrox-server`'s
//! non-greedy branch announces nothing at all, so a thread with no
//! setting is indistinguishable from a thread that wants no fold, and
//! adopting some other request's `On` would hand a sampling request one
//! precomputed id where it expects a vocabulary. Carrying the setting
//! explicitly is the only sound answer, and the carrying has to be done
//! by whoever moves the work.
//!
//! That is why `ferrox_core::par::on_workers` still declines to promote
//! a decode step under a GPU backend. Relaxing that gate is a separate,
//! measurable change, and it should pass [`greedy_fold_setting`] through
//! [`adopt_greedy_fold`] when it does.

use std::cell::Cell;

/// What the caller on some thread has said about the fold.
///
/// Three states and not a `bool`, because the two spellings of `false`
/// are not the same claim, and collapsing them is what makes the
/// setting impossible to carry safely.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GreedyFold {
    /// No caller on this thread has decided. Behaves as [`Self::Off`]
    /// and is never resolved from another thread's choice.
    #[default]
    Unset,
    /// This caller needs the vocabulary; the stack must not fold.
    Off,
    /// This caller is greedy and needs nothing but the chosen id.
    On,
}

impl GreedyFold {
    /// Whether a step configured this way folds.
    pub fn folds(self) -> bool {
        matches!(self, Self::On)
    }
}

thread_local! {
    /// Per-request setting, kept per thread so two concurrent requests
    /// sharing one `Arc<Decoder>` cannot overwrite each other's.
    ///
    /// Per thread is right for isolation and wrong for hand-off; see
    /// this module's header for the half that is not solved here.
    static SETTING: Cell<GreedyFold> = const { Cell::new(GreedyFold::Unset) };
}

/// Enable/disable the greedy GPU argmax fold for THIS THREAD's decode
/// steps.
///
/// `false` records [`GreedyFold::Off`], not "unset": a caller that has
/// finished a greedy request has expressed something, and the
/// distinction is what lets a carried setting be checked.
pub fn set_metal_greedy_argmax(on: bool) {
    SETTING.set(if on { GreedyFold::On } else { GreedyFold::Off });
}

/// True when this thread's decode steps should fold
/// `final_norm + lm_head + argmax` into the stack and return a
/// 1-element `[token_id as f32]` instead of hidden or full vocab logits.
pub fn metal_greedy_argmax_active() -> bool {
    greedy_fold_setting().folds()
}

/// This thread's setting, as a value that can be sent to another thread.
///
/// Capture it on the thread that decided, hand it to
/// [`adopt_greedy_fold`] on the thread that will run the step. Anything
/// that moves a Metal decode between threads has to do this, or the step
/// silently runs with a different setting from the one its caller chose.
pub fn greedy_fold_setting() -> GreedyFold {
    SETTING.get()
}

/// Installs `setting` on this thread until the returned guard drops.
///
/// The seam a work-stealing scheduler needs: `ferrox_core::par`'s
/// `on_workers` promotion is gated off GPU backends today precisely
/// because a promoted step lost this setting, and relaxing that gate
/// means capturing on the submitting thread and adopting here.
#[must_use = "the setting is restored when the guard drops"]
pub fn adopt_greedy_fold(setting: GreedyFold) -> GreedyFoldGuard {
    let previous = SETTING.replace(setting);
    GreedyFoldGuard { previous }
}

/// Restores the setting a thread had before [`adopt_greedy_fold`].
///
/// A worker is reused, so leaving a borrowed setting behind would make
/// the NEXT job on this thread inherit it, which is the same defect one
/// level down.
pub struct GreedyFoldGuard {
    previous: GreedyFold,
}

impl Drop for GreedyFoldGuard {
    fn drop(&mut self) {
        SETTING.set(self.previous);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GitHub issue #166's mechanism, in one test: the decision does not
    /// follow the work.
    ///
    /// This is a characterization test, not an aspiration. A Metal
    /// decode step driven on a thread that was never told what the
    /// caller wanted takes the other lm_head path, and on a default
    /// `--temp 0` run the two paths give different tokens. Anything that
    /// moves a decode step between threads must carry the setting.
    ///
    /// Sabotage: make `greedy_fold_setting` read a `static AtomicU8`
    /// instead of the thread-local, which is the tempting "fix" that
    /// would break two concurrent server requests at different
    /// temperatures.
    #[test]
    fn the_fold_decision_does_not_follow_the_work_to_another_thread() {
        set_metal_greedy_argmax(true);
        assert!(metal_greedy_argmax_active());

        let seen_elsewhere = std::thread::spawn(metal_greedy_argmax_active)
            .join()
            .expect("worker thread");
        assert!(
            !seen_elsewhere,
            "a thread that was never told sees the default, so a decode \
             step moved there silently changes lm_head path"
        );
    }

    /// And the fix for anything that DOES move the work: capture, carry,
    /// adopt.
    ///
    /// Sabotage: drop the `SETTING.replace` in `adopt_greedy_fold` for a
    /// plain read.
    #[test]
    fn a_captured_setting_reproduces_itself_on_the_thread_that_adopts_it() {
        for (announced, expected) in [(true, GreedyFold::On), (false, GreedyFold::Off)] {
            set_metal_greedy_argmax(announced);
            let carried = greedy_fold_setting();
            assert_eq!(carried, expected);

            let seen = std::thread::spawn(move || {
                let _adopted = adopt_greedy_fold(carried);
                (greedy_fold_setting(), metal_greedy_argmax_active())
            })
            .join()
            .expect("worker thread");
            assert_eq!(seen, (expected, announced));
        }
    }

    /// A worker is reused, so an adopted setting must not outlive the
    /// job that adopted it.
    ///
    /// Sabotage: make `GreedyFoldGuard::drop` a no-op.
    #[test]
    fn an_adopted_setting_is_gone_when_its_guard_drops() {
        set_metal_greedy_argmax(false);
        {
            let _adopted = adopt_greedy_fold(GreedyFold::On);
            assert!(metal_greedy_argmax_active());
        }
        assert_eq!(greedy_fold_setting(), GreedyFold::Off);
        assert!(!metal_greedy_argmax_active());
    }

    /// The two spellings of "does not fold" are different claims, and
    /// collapsing them is what makes a carried setting unsafe: an unset
    /// thread must never be resolved from another thread's choice.
    ///
    /// Sabotage: give `GreedyFold` a `Default` of `Off` and delete
    /// `Unset`; the first assertion goes red.
    #[test]
    fn unset_is_not_the_same_claim_as_off() {
        let fresh = std::thread::spawn(greedy_fold_setting)
            .join()
            .expect("worker thread");
        assert_eq!(
            fresh,
            GreedyFold::Unset,
            "a thread nobody configured must say so"
        );
        assert!(!fresh.folds(), "and it must still behave as no fold");
        assert_ne!(GreedyFold::Unset, GreedyFold::Off);
    }
}
