//! What "the graphs disagree" is allowed to mean — measured on THIS
//! checkpoint, not fitted once and reused.
//!
//! # The defect this replaces
//!
//! `frink parity` used to call a checkpoint WRONG above an absolute KL,
//! with a second, looser absolute KL for checkpoints llama.cpp dots
//! against Q8_K activations. That second constant was
//! `KL_REFERENCE_BUILD_SPREAD_KQUANT * 1.1` = 3.008e-2, derived in
//! [#108](https://github.com/antonellof/frink/issues/108) from the
//! largest reference-against-reference KL then known (2.735e-2, on
//! Qwen2.5-1.5B). Its own doc comment named what would invalidate it:
//! *"the spread is the largest of nine checkpoints, so a tenth could
//! exceed it"*.
//!
//! A tenth did. [#111](https://github.com/antonellof/frink/issues/111)
//! measured 3.514e-2 between two builds of llama.cpp on Qwen3-0.6B
//! `--pure` Q4_K_S — above the line that was supposed to mean "two
//! graphs disagree", produced by two binaries whose graph is IDENTICAL.
//! So `frink parity` read DRIFT on that file against Homebrew b7650
//! and WRONG against `.scratch/llama.cpp`, from the same frink, and
//! exited non-zero on the second.
//!
//! **Raising the constant does not fix that.** Setting it to the new
//! largest measurement puts the line at 3.865e-2 and the newer
//! reference reads 3.889e-2, still over. The next checkpoint is wider
//! again. The spread is not a constant of the engine at all: it is a
//! property of the pair of llama.cpp builds and of the checkpoint, and
//! across 21 checkpoints measured here it ranges over **five orders of
//! magnitude**, from exactly 0 to 3.514e-2. One number cannot separate
//! "frink is wrong" from "this is a checkpoint where two references
//! disagree a lot", and it fails in BOTH directions: 3.008e-2 was
//! simultaneously too tight for Qwen3-0.6B and 14x too loose for
//! Llama-3.2-1B Q4_K_M, whose two references agree to 2.1e-3.
//!
//! # The rule
//!
//! Measure the references' own disagreement on this checkpoint, and ask
//! whether frink is a bigger outlier than they are:
//!
//! > frink is WRONG only when it is further from EVERY reference than
//! > the references are from EACH OTHER — and never below the absolute
//! > [`KL_WRONG`] line, which is what a wrong graph costs regardless.
//!
//! `line = max(KL_WRONG, spread)`, tested against `min_i KL(ref_i ‖
//! frink)`. There is **no margin constant and no fitted number**: the
//! only constant left is [`KL_WRONG`], unchanged at 1e-2 since before
//! any of this. Three constants are deleted, none added.
//!
//! Why the NEAREST reference rather than the primary: the claim a WRONG
//! makes is "frink's forward pass computes something different from
//! llama.cpp's". If frink reproduces build A as closely as build B
//! reproduces A, then frink's graph agrees with A's as well as B's
//! does, and calling that a different graph would convict B too. That
//! is exactly the situation #102 and #111 describe.
//!
//! # The measurement, 21 checkpoints, frink out of the pairwise half
//!
//! Homebrew libllama b7650 against `.scratch/llama.cpp` (ggml 0.18.0),
//! CPU-only, the fixed parity prompt, 2026-09-04. `spread` is the
//! larger of the two KL directions between the references; `nearest` is
//! `min(KL(b7650‖frink), KL(scratch‖frink))`; `ratio` is
//! `nearest / line`.
//!
//! | checkpoint | spread | nearest | line | ratio |
//! |---|---|---|---|---|
//! | Llama-3.2-1B Q8_0 | **0.0** | 6.912e-4 | 1.000e-2 | 0.07 |
//! | tinyllama-1.1B Q8_0 | **0.0** | 2.445e-4 | 1.000e-2 | 0.02 |
//! | Llama-3.2-1B q6khead-q8body | 1.140e-13 | 2.313e-4 | 1.000e-2 | 0.02 |
//! | Llama-3.2-1B IQ4_XS | 2.981e-4 | 9.208e-4 | 1.000e-2 | 0.09 |
//! | olmoe-1b-7b Q4_0 | 4.570e-4 | 4.259e-4 | 1.000e-2 | 0.04 |
//! | Llama-3.2-3B Q4_K_M | 7.518e-4 | 6.463e-4 | 1.000e-2 | 0.06 |
//! | Llama-3.1-8B Q4_K_M | 7.579e-4 | 1.072e-3 | 1.000e-2 | 0.11 |
//! | Llama-3.2-1B pure-q4ks | 1.163e-3 | 1.126e-3 | 1.000e-2 | 0.11 |
//! | Llama-3.2-1B Q6_K | 1.280e-3 | 3.560e-3 | 1.000e-2 | 0.36 |
//! | Mistral-7B Q4_K_M | 1.436e-3 | 4.684e-4 | 1.000e-2 | 0.05 |
//! | Llama-3.2-1B q8head-q4ksbody | 1.857e-3 | 8.650e-4 | 1.000e-2 | 0.09 |
//! | Llama-3.2-1B Q4_K_M | 2.118e-3 | 1.077e-3 | 1.000e-2 | 0.11 |
//! | Llama-3.2-1B Q5_K_M | 2.174e-3 | 1.382e-3 | 1.000e-2 | 0.14 |
//! | Phi-4-mini Q4_K_M | 5.585e-3 | 3.800e-3 | 1.000e-2 | 0.38 |
//! | Qwen1.5-MoE-A2.7B Q4_K_M | 5.698e-3 | 4.826e-3 | 1.000e-2 | 0.48 |
//! | Yi-1.5-6B Q4_K_M | 7.318e-3 | 2.177e-3 | 1.000e-2 | 0.22 |
//! | DeepSeek-R1-Distill-1.5B Q4_K_M | 1.570e-2 | 9.157e-3 | 1.570e-2 | **0.58** |
//! | gemma-2-2b Q4_K_M | 1.674e-2 | 6.511e-3 | 1.674e-2 | 0.39 |
//! | Qwen2.5-1.5B Q4_K_M | 2.735e-2 | 7.679e-3 | 2.735e-2 | 0.28 |
//! | **Qwen3-0.6B q8head-q4ksbody** | 2.746e-2 | 1.297e-2 | 2.746e-2 | 0.47 |
//! | **Qwen3-0.6B pure-q4ks** | **3.514e-2** | 1.975e-2 | 3.514e-2 | 0.56 |
//!
//! Every row is inside its line, the worst at 58% of it. The two bold
//! rows are the ones that read WRONG under the old constant against the
//! newer reference (3.889e-2 and 3.519e-2 against 3.008e-2) — #111 —
//! and they are the only two rows this rule moves.
//!
//! The measurement also corrects a claim #108 made from a smaller
//! sample. "The reference's spread on Q8_0 is exactly zero" holds for
//! VINTAGE (b7650 and ggml 0.18.0 are bit-identical on both Q8_0
//! checkpoints) but not for COMPILER FLAGS: a third build of the same
//! source with FMA contraction disabled differs from b7650 by 4.11e-4
//! on tinyllama Q8_0, and Q4_0 (olmoe) already moves 4.57e-4 between
//! vintages. All of it is two orders under [`KL_WRONG`], which is why
//! that line is left exactly where it was.
//!
//! # What this costs, stated rather than hoped
//!
//! **A genuine frink bug inside a wide band is excused.** That is real
//! and it is the price. What makes it acceptable is that the band is
//! only wide where the reference is: 16 of the 21 rows above get the
//! 1e-2 floor, THREE TIMES TIGHTER than the 3.008e-2 they used to be
//! judged against. So per-checkpoint calibration raises sensitivity on
//! most checkpoints and lowers it only on the ones where a WRONG could
//! not have been believed anyway — on Qwen3-0.6B `--pure`, no verdict
//! of "the graphs disagree" is available from this instrument at any
//! threshold, because llama.cpp reproduces that difference against
//! llama.cpp.
//!
//! **Two builds are two points, not a distribution.** The spread is a
//! lower bound on how much implementation choice moves this checkpoint,
//! measured from the two implementations to hand. It is not an estimate
//! of dispersion and this module does not pretend otherwise; the
//! report prints the number and the pair it came from so a reader can
//! judge it. Adding a third reference widens the band (it is the max
//! over pairs), so the references passed should be builds you would
//! accept as correct.
//!
//! **It costs a second reference run per checkpoint**, and only when
//! one is configured. With ONE reference there is no spread to measure
//! and [`WrongLine::Uncalibrated`] is the honest answer for the
//! Q8_K-dotted class: no KL can be called "the graphs disagree" when
//! the instrument has not been shown able to tell. The top-1 half of
//! the verdict is unaffected — it rests on no threshold — so a
//! single-reference run still fails on a genuine argmax disagreement,
//! and Q8_0/Q4_0/IQ4_NL checkpoints still get [`KL_WRONG`], because
//! their measured spread is at or near zero.

use super::metrics;
use super::quant::DominantQuant;

/// Above this KL two distributions differ by more than accumulation
/// order — a different rope base, a missing bias, a skipped norm. The
/// "wrong graph" line, and the floor under every calibrated one.
///
/// Unchanged, and deliberately: the reference's own build-to-build
/// spread on the arithmetic this line governs is 0.0 (Q8_0, both
/// vintages) to 4.6e-4 (Q4_0), so there is nothing here to make room
/// for. It is also the floor under a calibrated line, because a pair of
/// references that happen to agree closely on one checkpoint is not
/// evidence that frink must agree with them to the same 1e-13.
pub(super) const KL_WRONG: f64 = 1e-2;

/// Two references, and how far apart they are on this checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Spread {
    /// The largest KL over ORDERED pairs. Ordered because KL is
    /// asymmetric and picking one direction would quietly report the
    /// smaller of two equally valid numbers.
    pub(super) kl: f64,
    /// Indices, into the caller's reference list, of the pair that
    /// produced it.
    pub(super) between: (usize, usize),
}

/// The line a KL has to cross before "the graphs disagree" is a claim
/// this instrument can make.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum WrongLine {
    /// Measured on this checkpoint: `max(KL_WRONG, spread)`.
    Calibrated { spread: Spread, line: f64 },
    /// [`KL_WRONG`], for a checkpoint whose arithmetic llama.cpp dots
    /// against Q8_0 activations. Legitimate without a second reference
    /// because the spread there is at or near zero.
    Absolute(f64),
    /// NO LINE. One reference, on a checkpoint whose K-quant activation
    /// path is exactly the thing two llama.cpp builds disagree about by
    /// up to 3.5e-2. Any constant here would fire on differences
    /// llama.cpp reproduces against itself (#111).
    Uncalibrated,
}

impl WrongLine {
    fn decide(spread: Option<Spread>, quant: &DominantQuant) -> Self {
        match spread {
            // `max` and not `spread` alone: see KL_WRONG's own comment.
            Some(s) => {
                let line = s.kl.max(KL_WRONG);
                Self::Calibrated { spread: s, line }
            }
            None if quant.q8k_dotted() => Self::Uncalibrated,
            None => Self::Absolute(KL_WRONG),
        }
    }

    /// The number, when there is one.
    pub(super) fn value(&self) -> Option<f64> {
        match self {
            Self::Calibrated { line, .. } => Some(*line),
            Self::Absolute(v) => Some(*v),
            Self::Uncalibrated => None,
        }
    }
}

/// Everything the WRONG rung of the verdict rests on.
///
/// Holds the KL to every reference as well as the line, so the report
/// and the verdict read ONE set of numbers. Recomputing `KL(primary ‖
/// frink)` in `compare` would be two evaluations of one quantity with
/// nothing tying them together, which is how this repo loses things.
#[derive(Debug, Clone)]
pub(super) struct Band {
    /// `KL(reference_i ‖ frink)`, in the order the references were
    /// given. Never empty: a run with no reference cannot get this far.
    to_frink: Vec<f64>,
    line: WrongLine,
}

impl Band {
    /// `references` must be non-empty and every vector the same length
    /// as `frink`; `run` drops any reference that disagrees about the
    /// vocabulary before getting here.
    pub(super) fn measure(references: &[&[f32]], frink: &[f32], quant: &DominantQuant) -> Self {
        let probs: Vec<Vec<f64>> = references.iter().map(|r| metrics::softmax(r)).collect();
        let fx = metrics::softmax(frink);
        let to_frink = probs.iter().map(|p| metrics::kl(p, &fx)).collect();

        // Largest KL over ordered pairs of DISTINCT references.
        let mut spread: Option<Spread> = None;
        for (i, a) in probs.iter().enumerate() {
            for (j, b) in probs.iter().enumerate() {
                if i == j {
                    continue;
                }
                let kl = metrics::kl(a, b);
                // A spread of exactly zero is not a band. It means the
                // two references are the same program for this
                // checkpoint — the same dumper passed twice, or two
                // builds that are bit-identical on it — and a pair that
                // cannot disagree bounds nothing. Calibrating on it
                // would collapse the line onto KL_WRONG for a K-quant,
                // which is the false-WRONG this module exists to stop.
                if kl > 0.0 && spread.as_ref().is_none_or(|s| kl > s.kl) {
                    spread = Some(Spread {
                        kl,
                        between: (i, j),
                    });
                }
            }
        }

        Self {
            line: WrongLine::decide(spread, quant),
            to_frink,
        }
    }

    /// `KL(reference_i ‖ frink)`.
    pub(super) fn kl_to_frink(&self, i: usize) -> f64 {
        self.to_frink[i]
    }

    /// How far frink is from the reference it agrees with best.
    pub(super) fn nearest(&self) -> f64 {
        self.to_frink.iter().copied().fold(f64::INFINITY, f64::min)
    }

    pub(super) fn line(&self) -> &WrongLine {
        &self.line
    }

    /// THE PREDICATE. Every rung of the verdict that could read WRONG
    /// goes through this one call, so the "same top-1" branch and the
    /// "top-1 flipped" branch cannot come to disagree about what the
    /// line is — they were two spellings of `kl < kl_wrong` before.
    ///
    /// `false` whenever there is no line: an instrument that has not
    /// been shown able to tell does not get to convict.
    pub(super) fn frink_is_outside(&self) -> bool {
        self.line.value().is_some_and(|line| self.nearest() > line)
    }
}

#[cfg(test)]
mod tests {
    use super::super::quant::tests::{Q8K_DOTTED, Q8_0_DOTTED};
    use super::*;

    fn kquant() -> DominantQuant {
        DominantQuant::weigh(Some("Q4K"), Some("Q4K"))
    }

    fn q8_0() -> DominantQuant {
        DominantQuant::weigh(Some("Q8_0"), Some("Q8_0"))
    }

    fn spread(kl: f64) -> Option<Spread> {
        Some(Spread {
            kl,
            between: (0, 1),
        })
    }

    /// EVERY measured row of the 2026-09-04 three-reference sweep, as
    /// `(checkpoint, spread, KL to b7650, KL to scratch, q8k-dotted)`.
    ///
    /// This is the evidence the rule is made of, in a form that fails
    /// the build when the rule stops agreeing with it. A doc-comment
    /// table records what was measured; this asserts the code still
    /// answers the same way about it.
    const SWEEP: &[(&str, f64, f64, f64, bool)] = &[
        ("Llama-3.2-1B-Q8_0", 0.0, 6.9121e-4, 6.9121e-4, false),
        ("tinyllama-1.1B-Q8_0", 0.0, 2.4448e-4, 2.4448e-4, false),
        (
            "Llama-3.2-1B-q6khead-q8body",
            1.1399e-13,
            2.3131e-4,
            2.3132e-4,
            true,
        ),
        (
            "Llama-3.2-1B-IQ4_XS",
            2.9812e-4,
            9.2083e-4,
            1.4186e-3,
            false,
        ),
        ("olmoe-1b-7b-Q4_0", 4.5703e-4, 4.8177e-4, 4.2587e-4, false),
        ("Llama-3.2-3B-Q4_K_M", 7.5180e-4, 9.1456e-4, 6.4631e-4, true),
        ("Llama-3.1-8B-Q4_K_M", 7.5787e-4, 1.0718e-3, 1.3068e-3, true),
        (
            "Llama-3.2-1B-pure-q4ks",
            1.1626e-3,
            1.1260e-3,
            1.3212e-3,
            true,
        ),
        ("Llama-3.2-1B-Q6_K", 1.2803e-3, 3.5602e-3, 6.9239e-3, true),
        ("Mistral-7B-Q4_K_M", 1.4363e-3, 4.6836e-4, 1.0008e-3, true),
        (
            "Llama-3.2-1B-q8head-q4ksbody",
            1.8567e-3,
            1.4173e-3,
            8.6501e-4,
            true,
        ),
        ("Llama-3.2-1B-Q4_K_M", 2.1176e-3, 1.8193e-3, 1.0773e-3, true),
        ("Llama-3.2-1B-Q5_K_M", 2.1742e-3, 3.4487e-3, 1.3816e-3, true),
        ("Phi-4-mini-Q4_K_M", 5.5853e-3, 1.1790e-2, 3.7999e-3, true),
        (
            "Qwen1.5-MoE-A2.7B-Q4_K_M",
            5.6984e-3,
            5.0306e-3,
            4.8256e-3,
            true,
        ),
        ("Yi-1.5-6B-Q4_K_M", 7.3179e-3, 2.1770e-3, 8.0876e-3, true),
        (
            "DeepSeek-R1-Distill-1.5B-Q4_K_M",
            1.5695e-2,
            1.9874e-2,
            9.1570e-3,
            true,
        ),
        ("gemma-2-2b-Q4_K_M", 1.6736e-2, 6.5107e-3, 1.5307e-2, true),
        ("Qwen2.5-1.5B-Q4_K_M", 2.7348e-2, 7.6786e-3, 2.6692e-2, true),
        (
            "Qwen3-0.6B-q8head-q4ksbody",
            2.7459e-2,
            1.2966e-2,
            3.5192e-2,
            true,
        ),
        (
            "Qwen3-0.6B-pure-q4ks",
            3.5141e-2,
            1.9748e-2,
            3.8885e-2,
            true,
        ),
    ];

    fn band_of(spread_kl: f64, kls: &[f64], quant: &DominantQuant) -> Band {
        Band {
            to_frink: kls.to_vec(),
            line: WrongLine::decide(
                (spread_kl > 0.0).then_some(Spread {
                    kl: spread_kl,
                    between: (0, 1),
                }),
                quant,
            ),
        }
    }

    /// NOT ONE of the 21 measured checkpoints is WRONG under this rule,
    /// with two references present.
    ///
    /// The old constant made two of them WRONG — the two Qwen3-0.6B
    /// rows, against `.scratch/llama.cpp` — which is #111. Everything
    /// else must land exactly where it already did, or the fix has a
    /// blast radius nobody measured.
    #[test]
    fn no_measured_checkpoint_is_wrong_when_the_references_are_calibrated() {
        for &(name, spread_kl, kl_brew, kl_scratch, q8k) in SWEEP {
            let quant = if q8k { kquant() } else { q8_0() };
            let band = band_of(spread_kl, &[kl_brew, kl_scratch], &quant);
            let line = band.line().value().expect("two references give a line");
            assert!(
                !band.frink_is_outside(),
                "{name}: nearest {:.4e} crosses the {line:.4e} line built from a {spread_kl:.4e} \
                 reference spread",
                band.nearest()
            );
            // And the headroom is not a rounding accident: the worst
            // row in the sweep sits at 58% of its line. A rule that
            // only just clears every row is a rule about to fire on the
            // next checkpoint.
            assert!(
                band.nearest() <= 0.6 * line,
                "{name}: {:.4e} is {:.2} of its {line:.4e} line — the measured worst is 0.58, \
                 so either the sweep moved or the rule did",
                band.nearest(),
                band.nearest() / line
            );
        }
    }

    /// The two rows #111 is about, and the arithmetic that settles them.
    ///
    /// Qwen3-0.6B `--pure` Q4_K_S: two builds of llama.cpp disagree with
    /// EACH OTHER by 3.514e-2, above the 3.008e-2 that used to mean "the
    /// graphs disagree". Against the newer build frink measured
    /// 3.889e-2 and read WRONG; against b7650 it measured 1.975e-2 and
    /// read DRIFT — same frink, same file. Raising the constant to the
    /// new largest spread would have put the line at 3.865e-2, which
    /// 3.889e-2 still crosses; this asserts the SHAPE fixes it and the
    /// bump would not have.
    #[test]
    fn the_checkpoint_that_broke_the_constant_is_inside_its_own_measured_band() {
        let band = band_of(3.5141e-2, &[1.9748e-2, 3.8885e-2], &kquant());
        assert_eq!(
            band.line().value(),
            Some(3.5141e-2),
            "the line is the spread itself, the floor being smaller"
        );
        assert!(!band.frink_is_outside());
        assert_eq!(band.nearest(), 1.9748e-2);

        // What the issue says would NOT have worked: the old shape with
        // the new largest measurement in it.
        let bumped_constant = 3.5141e-2 * 1.1;
        assert!(
            3.8885e-2 > bumped_constant,
            "raising the constant to the newest spread leaves the newer reference over the \
             line — the point of #111"
        );
    }

    /// ONE reference cannot render a Q8_K-dotted checkpoint WRONG.
    ///
    /// This is the honest degradation and it is a deliberate loss of a
    /// gate: with a single libllama there is no measurement of how much
    /// this checkpoint's K-quant activation path moves between builds,
    /// and every constant tried so far has been crossed by two builds
    /// with an identical graph. `Uncalibrated` carries NO number, so
    /// nothing downstream can quietly reintroduce one.
    #[test]
    fn one_reference_declines_to_convict_a_kquant_rather_than_guessing_a_line() {
        for kind in Q8K_DOTTED {
            let quant = DominantQuant::weigh(Some("Q8_0"), Some(kind));
            let band = band_of(0.0, &[9.9e-1], &quant);
            assert_eq!(*band.line(), WrongLine::Uncalibrated);
            assert_eq!(band.line().value(), None, "{kind} must carry no number");
            assert!(
                !band.frink_is_outside(),
                "{kind}: a KL of 0.99 is enormous, and with one reference this instrument still \
                 cannot say it is a wrong GRAPH — the top-1 half of the verdict is what covers it"
            );
        }
    }

    /// ONE reference still gates everything llama.cpp dots against Q8_0
    /// activations, at the unchanged 1e-2.
    ///
    /// The measured spread there is 0.0 across both vintages on Q8_0 and
    /// 4.6e-4 on Q4_0, so an absolute line is a real line for this
    /// class. Losing it too would have thrown away the half of the
    /// oracle that never broke.
    #[test]
    fn one_reference_still_gates_a_q8_0_dotted_checkpoint_at_the_unchanged_line() {
        for kind in Q8_0_DOTTED {
            let quant = DominantQuant::weigh(Some(kind), Some(kind));
            let band = band_of(0.0, &[1.1e-2], &quant);
            assert_eq!(*band.line(), WrongLine::Absolute(1e-2));
            assert!(
                band.frink_is_outside(),
                "{kind}: 1.1e-2 is over the 1e-2 line and nothing about this class is calibrated \
                 away"
            );
            assert!(!band_of(0.0, &[9.9e-3], &quant).frink_is_outside());
        }
    }

    /// TWO REFERENCES THAT AGREE EXACTLY ARE ONE REFERENCE.
    ///
    /// Passing `--dumper` twice with the same binary, or two builds that
    /// happen to be bit-identical on this checkpoint, measures a spread
    /// of 0. Treating that as a calibrated band would put the line at
    /// the 1e-2 floor for a K-quant and manufacture exactly the WRONG
    /// this module removes — the failure would look like a successful
    /// calibration, which is worse than no calibration.
    #[test]
    fn a_pair_of_references_that_cannot_disagree_does_not_calibrate_anything() {
        let identical = vec![0.5f32, 2.0, -1.0, 7.25];
        let frink = vec![0.5f32, 2.9, -1.0, 7.25];
        let band = Band::measure(&[&identical, &identical], &frink, &kquant());
        assert_eq!(*band.line(), WrongLine::Uncalibrated);
        assert!(!band.frink_is_outside());

        // Two references that DO differ, on the same frink, do
        // calibrate — otherwise this test would pass on a `measure`
        // that never calibrates at all.
        let other = vec![0.5f32, 2.4, -1.0, 7.25];
        let band = Band::measure(&[&identical, &other], &frink, &kquant());
        assert!(matches!(band.line(), WrongLine::Calibrated { .. }));
    }

    /// The spread walks ORDERED pairs, so the larger of the two KL
    /// directions is the one it reports.
    ///
    /// KL is asymmetric, so a loop over `i < j` would report a band
    /// smaller than the evidence supports on every pair where the two
    /// references have different entropies — and a band too small is a
    /// WRONG that should not have been printed. The references are
    /// deliberately ordered so that the LARGER direction is the pair
    /// `(1, 0)`, which an `i < j` loop never visits; asserting the pair
    /// as well as the value is what makes that visible, because the
    /// value alone would still be right for half of all orderings.
    #[test]
    fn the_spread_is_the_larger_of_the_two_kl_directions() {
        // A peaked distribution against a flat one: the two KL
        // directions differ by a factor of several.
        let peaked = vec![6.0f32, 0.0, 0.0, 0.0];
        let flat = vec![0.0f32, 0.0, 0.0, 0.0];
        let frink = vec![0.1f32, 0.0, 0.0, 0.0];

        let p = metrics::softmax(&peaked);
        let q = metrics::softmax(&flat);
        let (fwd, back) = (metrics::kl(&p, &q), metrics::kl(&q, &p));
        assert!(
            back > fwd,
            "the fixture must put the larger direction on the reversed pair: {fwd} vs {back}"
        );

        let band = Band::measure(&[&peaked, &flat], &frink, &kquant());
        match band.line() {
            WrongLine::Calibrated { spread, .. } => {
                assert_eq!(spread.kl, back);
                assert_ne!(spread.kl, fwd);
                assert_eq!(
                    spread.between,
                    (1, 0),
                    "the reported pair must be the ordered one that produced the number"
                );
            }
            other => panic!("expected a calibrated line, got {other:?}"),
        }
    }

    /// frink is judged by the reference it agrees with BEST, and the
    /// choice is what makes the verdict independent of which libllama
    /// happens to be installed.
    ///
    /// #102's whole complaint was a verdict that changed with the
    /// bottle. Ordering the same two references the other way round
    /// must not change the answer.
    #[test]
    fn the_verdict_does_not_depend_on_which_reference_was_passed_first() {
        for &(name, spread_kl, kl_brew, kl_scratch, q8k) in SWEEP {
            let quant = if q8k { kquant() } else { q8_0() };
            let forward = band_of(spread_kl, &[kl_brew, kl_scratch], &quant);
            let reversed = band_of(spread_kl, &[kl_scratch, kl_brew], &quant);
            assert_eq!(
                forward.frink_is_outside(),
                reversed.frink_is_outside(),
                "{name}: the verdict moved when the references swapped places"
            );
            assert_eq!(forward.nearest(), reversed.nearest());
            // The PRIMARY's KL still differs, which is why the report
            // prints every reference's number and not just one.
            assert_eq!(forward.kl_to_frink(0), reversed.kl_to_frink(1));
        }
    }

    /// A calibrated line is never below the absolute one.
    ///
    /// Without the floor, a pair of references that agree to 1e-13 —
    /// which two of the measured checkpoints do — would demand that
    /// frink agree to 1e-13 too, and every row would read WRONG.
    #[test]
    fn a_calibrated_line_never_drops_below_the_absolute_one() {
        for kl in [1e-13, 1e-6, 1e-3, 9.99e-3, KL_WRONG, 1e-1] {
            let line = WrongLine::decide(spread(kl), &kquant())
                .value()
                .expect("a spread gives a line");
            assert!(
                line >= KL_WRONG,
                "a {kl:e} reference spread produced a {line:e} line"
            );
            assert!(line >= kl);
        }
        // And above the floor the line IS the spread, with nothing
        // added: no margin, no fitted headroom.
        assert_eq!(
            WrongLine::decide(spread(4.2e-2), &kquant()).value(),
            Some(4.2e-2)
        );
    }
}
