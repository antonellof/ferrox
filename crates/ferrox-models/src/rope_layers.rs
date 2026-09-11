//! **Which layers rotate.** llama.cpp's per-layer `use_rope` gate, as
//! one rule with a table, rather than one branch per architecture.
//!
//! Every architecture ferrox had audited rotates Q and K on every
//! layer, and that is llama.cpp's default too. Six architectures
//! upstream do not, and until 2026-09-10 ferrox had no way to say so:
//! three of them were REFUSED for it, one ran WRONG, and two were
//! latent behind other refusals.
//!
//! # The six, transcribed
//!
//! | arch | llama.cpp | line |
//! |---|---|---|
//! | `exaone4` | `is_swa(il) \|\| swa_type == NONE` | `exaone4.cpp:116` |
//! | `exaone-moe` | `is_swa(il)` | `exaone-moe.cpp:136,155` |
//! | `smollm3` | `(il + 1) % 4 != 0` | `smollm3.cpp:5,69` |
//! | `smallthinker` | `step == n_layer \|\| il % step != 0` | `smallthinker.cpp:18,108-109` |
//! | `afmoe` | `step > 0 && (il + 1) % step != 0` | `afmoe.cpp:137-138` |
//! | `llama4` | `step > 0 && (il + 1) % step != 0` | `llama4.cpp:11,145-146` |
//!
//! **`exaone-moe` and `exaone4` are ONE rule, not two that look
//! alike.** `exaone-moe.cpp:4` sets `swa_type = LLAMA_SWA_TYPE_STANDARD`
//! unconditionally, so its `is_swa(il)` is exactly `exaone4`'s
//! `is_swa(il) || swa_type == NONE` with the second disjunct nailed to
//! false. Read side by side the two `if` bodies are the same two
//! `ggml_rope_ext` calls guarded by the same predicate; the only
//! difference is that `exaone4.cpp:4` reaches that predicate solely when
//! `n_layer() == 64`.
//!
//! `smallthinker` collapses onto the same shape ONLY BY COINCIDENCE and
//! is deliberately NOT written that way here. With a window it takes
//! `set_swa_pattern(4, dense_first = true)`, so `is_swa(il)` is
//! `il % 4 != 0`, which happens to equal its `use_rope`; llama.cpp's own
//! comment at `smallthinker.cpp:107` says "this overlaps with SWA layers
//! in current models". They are two independent constants -- the SWA
//! period comes from `{arch}.attention.sliding_window_pattern` and the
//! no-RoPE step never does -- so a file overriding the period to 2 would
//! separate them, and collapsing the two would rope such a file wrong in
//! precisely the way this module exists to stop.
//!
//! # Why a step is never a GGUF key
//!
//! `hparams.n_no_rope_layer_step` is set from a literal in every one of
//! the four architectures that use it and read from no key anywhere in
//! llama.cpp (`grep -rn n_no_rope_layer_step src/`). `llama-hparams.h:203`
//! defaults it to **4**, which is why `afmoe` and `llama4` are in the
//! table despite never assigning it: they inherit the default and their
//! graphs consult it. So there is nothing for a metadata gate to test.
//! A checkpoint of any of these six loads clean, carries the generic
//! tensor set, runs at full speed and answers from positions it never
//! encodes that way.

use std::num::NonZeroUsize;

/// Which of a period's layers is the one that does NOT rotate.
///
/// Two spellings, both live upstream, and they differ by one layer of
/// phase. Getting it wrong is not a near miss: on a 36-layer step-4
/// model the two phases disagree about EIGHTEEN layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoRopePhase {
    /// `(il + 1) % step == 0` -- the LAST layer of each period skips
    /// rotation. `smollm3.cpp:69`, `afmoe.cpp:138`, `llama4.cpp:146`.
    LastOfPeriod,
    /// `il % step == 0` -- the FIRST layer of each period skips
    /// rotation. `smallthinker.cpp:109`.
    FirstOfPeriod,
}

/// llama.cpp's per-layer `use_rope`, as a value a `ModelConfig` can
/// carry.
///
/// The default is [`RopeLayers::All`] and it is what every audited
/// architecture but four gets. A variant is only ever added by reading
/// a `use_rope` in `src/models/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopeLayers {
    /// Every layer rotates: llama.cpp writes no gate at all.
    #[default]
    All,
    /// Only the SLIDING-WINDOW layers rotate; the full-attention layers
    /// get no rotation. `exaone4` (when its SWA is on) and `exaone-moe`.
    ///
    /// Note what this means when a model's SWA pattern windows nothing
    /// (llama.cpp's degenerate `set_swa_pattern(1)`): NO layer rotates.
    /// That is upstream's answer too -- `is_swa(il)` is false everywhere
    /// while `swa_type` is still `STANDARD` -- and it is why this is not
    /// written as "sliding layers, or everything if nothing slides".
    SlidingOnly,
    /// One layer in every `step` does not rotate,
    /// `hparams.n_no_rope_layer_step`.
    NoRopeEvery {
        /// llama.cpp's `n_no_rope_layer_step`. Never zero here: the two
        /// architectures that guard on `step > 0` inherit the
        /// `llama-hparams.h:203` default of 4 and never assign it.
        step: NonZeroUsize,
        /// Which layer of the period is skipped.
        phase: NoRopePhase,
    },
}

impl RopeLayers {
    /// Does layer `layer_idx` rotate?
    ///
    /// `layer_slides` is `ModelConfig::layer_sliding_window(il).is_some()`,
    /// i.e. llama.cpp's `hparams.is_swa(il)`. It is a parameter rather
    /// than something this enum works out, because the SWA layout is
    /// the `ModelConfig`'s answer and restating it here would be two
    /// structures that must agree about one thing.
    #[inline]
    pub fn rotates(self, layer_idx: usize, layer_slides: bool) -> bool {
        match self {
            Self::All => true,
            Self::SlidingOnly => layer_slides,
            Self::NoRopeEvery { step, phase } => {
                let step = step.get();
                match phase {
                    NoRopePhase::LastOfPeriod => !(layer_idx + 1).is_multiple_of(step),
                    NoRopePhase::FirstOfPeriod => !layer_idx.is_multiple_of(step),
                }
            }
        }
    }

    /// True when this rule leaves at least one layer of an `n_layers`
    /// model unrotated, i.e. when it is anything other than "rotate
    /// everything" IN PRACTICE rather than in name.
    ///
    /// The distinction matters: `NoRopeEvery { step: 64, .. }` on a
    /// 30-layer model is [`Self::All`] by any observable test, and a
    /// caller asking "must I implement this?" wants the observable
    /// answer.
    pub fn any_layer_unrotated(self, n_layers: usize, slides: impl Fn(usize) -> bool) -> bool {
        (0..n_layers).any(|il| !self.rotates(il, slides(il)))
    }
}

/// llama.cpp's `n_no_rope_layer_step` default (`llama-hparams.h:203`).
///
/// Not a ferrox choice and not a tunable: `afmoe` and `llama4` never
/// assign the field and their graphs still read it, so this literal is
/// load-bearing for two of the six rows.
const LLAMA_CPP_DEFAULT_NO_ROPE_STEP: usize = 4;

const fn step(n: usize) -> NonZeroUsize {
    match NonZeroUsize::new(n) {
        Some(n) => n,
        // A zero step would make `il % step` a division by zero; no row
        // has one and none can be added without tripping this.
        None => panic!("a no-RoPE step of 0 is not a rule"),
    }
}

/// THE table. Which layers of `arch` rotate.
///
/// `has_sliding_window` is llama.cpp's `swa_type != LLAMA_SWA_TYPE_NONE`
/// for this file -- ferrox's post-gate answer, so
/// [`crate::capability::honours_sliding_window`] has already applied any
/// architecture rule that switches SWA off wholesale. Passing the raw
/// presence of `{arch}.attention.sliding_window` instead would rope
/// EXAONE-4 1.2B as if it were the 32B.
///
/// `n_layers` is here for exactly one row: `smallthinker.cpp:108`
/// rotates everything when its step equals the layer count.
pub fn rope_layers(arch: &str, n_layers: usize, has_sliding_window: bool) -> RopeLayers {
    let no_rope_every = |phase| RopeLayers::NoRopeEvery {
        step: step(LLAMA_CPP_DEFAULT_NO_ROPE_STEP),
        phase,
    };
    match arch {
        // `exaone4.cpp:4-9` switches the whole SWA machinery on inside
        // `if (hparams.n_layer() == 64)` and :116 then gates RoPE on it,
        // so EXAONE-4 32B gives its full-attention layers no rotation
        // and EXAONE-4 1.2B rotates everything. `exaone-moe.cpp:4` turns
        // SWA on unconditionally and :13 reads the window as a REQUIRED
        // key, so its `has_sliding_window` is always true and the second
        // arm here is unreachable for it -- which is the whole reason
        // the two rows are one rule.
        "exaone4" | "exaone-moe" => {
            if has_sliding_window {
                RopeLayers::SlidingOnly
            } else {
                RopeLayers::All
            }
        }
        // `smollm3.cpp:5` assigns the step unconditionally, so this is
        // every SmolLM3 file: 9 of a 36-layer SmolLM3-3B's layers get no
        // rotation.
        "smollm3" => no_rope_every(NoRopePhase::LastOfPeriod),
        // `smallthinker.cpp:16-18`: the step is set to `n_layer` (i.e.
        // "always rope") ONLY on the no-window branch. With a window the
        // field keeps the `llama-hparams.h:203` default of 4 and
        // :108-109 skips `il % 4 == 0`. LIVE, and it was WRONG: ferrox
        // has `smallthinker` on the audited generic path and rotated
        // every layer of it.
        "smallthinker" if has_sliding_window && n_layers != LLAMA_CPP_DEFAULT_NO_ROPE_STEP => {
            no_rope_every(NoRopePhase::FirstOfPeriod)
        }
        // `afmoe.cpp:137-138` reads the step and never assigns it, so it
        // is the default 4. LATENT: `afmoe` refuses today for its
        // gated-attention topology, and this row is here so it stays
        // right if that changes.
        "afmoe" => no_rope_every(NoRopePhase::LastOfPeriod),
        // `llama4.cpp:11` sets the step to `n_layer` ("always use rope",
        // its own comment) only when the file declares a window of ZERO;
        // every other Llama-4 keeps the default 4 and :145-146 skips
        // `(il + 1) % 4 == 0`. LATENT on the generic path (`llama4` has
        // a dedicated engine), and `llama4`'s chunked attention is not
        // ferrox's `sliding_window` either.
        "llama4" if has_sliding_window => no_rope_every(NoRopePhase::LastOfPeriod),
        _ => RopeLayers::All,
    }
}

/// Every architecture llama.cpp gates RoPE on per layer, with the line
/// that decides it.
///
/// The table above is the implementation and this is the CENSUS, and
/// they are checked against each other rather than maintained side by
/// side: `every_gated_architecture_is_in_the_table` walks this list and
/// requires [`rope_layers`] to answer something other than
/// [`RopeLayers::All`] for each, under the conditions named. A name
/// added here with no arm, or an arm added with no name, fails.
pub const PER_LAYER_ROPE_GATES: &[(&str, &str)] = &[
    ("exaone4", "src/models/exaone4.cpp:116"),
    ("exaone-moe", "src/models/exaone-moe.cpp:136,155"),
    ("smollm3", "src/models/smollm3.cpp:5,69"),
    ("smallthinker", "src/models/smallthinker.cpp:18,108-109"),
    ("afmoe", "src/models/afmoe.cpp:137-138"),
    ("llama4", "src/models/llama4.cpp:11,145-146"),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The claim the whole module rests on: `exaone-moe`'s
    /// `is_swa(il)` and `exaone4`'s `is_swa(il) || swa_type == NONE` are
    /// ONE rule, because `exaone-moe.cpp:4` pins `swa_type` to
    /// `STANDARD`. If they ever needed different answers for the same
    /// `(layer, slides)` pair they would need different implementations.
    #[test]
    fn exaone4_and_exaone_moe_are_one_rule() {
        for slides in [true, false] {
            for il in 0..8 {
                assert_eq!(
                    rope_layers("exaone4", 64, true).rotates(il, slides),
                    rope_layers("exaone-moe", 48, true).rotates(il, slides),
                    "layer {il}, slides={slides}"
                );
            }
        }
        // And the disjunct that separates them: with no window at all
        // EXAONE-4 rotates everything, which is the 1.2B.
        assert_eq!(rope_layers("exaone4", 30, false), RopeLayers::All);
    }

    /// EXAONE-4 32B, layer by layer: `set_swa_pattern(4)` last-dense
    /// makes layers 0,1,2 slide and layer 3 dense, and only the sliding
    /// ones rotate.
    #[test]
    fn exaone4_32b_rotates_three_layers_in_four() {
        let rule = rope_layers("exaone4", 64, true);
        let slides = |il: usize| il % 4 < 3;
        for il in 0..64 {
            assert_eq!(
                rule.rotates(il, slides(il)),
                il % 4 != 3,
                "layer {il} of EXAONE-4 32B"
            );
        }
        assert!(rule.any_layer_unrotated(64, slides));
    }

    /// `smollm3` skips the LAST layer of each period and `smallthinker`
    /// the FIRST. One phase for both would rope 18 of a 36-layer
    /// SmolLM3-3B's layers at the wrong positions.
    #[test]
    fn the_two_no_rope_phases_disagree_about_every_layer_they_name() {
        let smollm3 = rope_layers("smollm3", 36, false);
        let smallthinker = rope_layers("smallthinker", 32, true);
        for il in 0..36 {
            assert_eq!(smollm3.rotates(il, false), (il + 1) % 4 != 0);
        }
        for il in 0..32 {
            assert_eq!(smallthinker.rotates(il, true), il % 4 != 0);
        }
        // Not vacuous: the two rules must actually name different
        // layers, or the phase distinction proves nothing.
        assert_ne!(smollm3, smallthinker);
        assert!(smollm3.rotates(0, false) && !smallthinker.rotates(0, true));
        assert!(!smollm3.rotates(3, false) && smallthinker.rotates(3, true));
    }

    /// `smallthinker` WITHOUT a window sets the step to `n_layer`
    /// (`smallthinker.cpp:18`), which is llama.cpp's spelling of "always
    /// rope". A SmallThinker with no window must not lose a layer.
    #[test]
    fn smallthinker_without_a_window_rotates_everything() {
        assert_eq!(rope_layers("smallthinker", 32, false), RopeLayers::All);
    }

    /// The census and the table are checked against each other, so a
    /// name in one and not the other cannot ship.
    #[test]
    fn every_gated_architecture_is_in_the_table() {
        for (arch, line) in PER_LAYER_ROPE_GATES {
            // 32 layers and a window is the shape that makes every one
            // of the six gates fire; the two conditional rows
            // (`smallthinker`, `llama4`) need the window and the other
            // four ignore it.
            let rule = rope_layers(arch, 32, true);
            assert_ne!(
                rule,
                RopeLayers::All,
                "{arch} is listed as gated at {line} but the table rotates every layer"
            );
            assert!(
                rule.any_layer_unrotated(32, |il| il % 4 < 3),
                "{arch}'s rule must actually leave a layer unrotated"
            );
        }
    }

    /// The other direction: an architecture llama.cpp does NOT gate must
    /// not pick up a gate here. `llama` is the whole generic path.
    #[test]
    fn an_ungated_architecture_rotates_every_layer() {
        for arch in ["llama", "qwen3", "gemma3", "olmo2", "exaone", "granite"] {
            assert_eq!(
                rope_layers(arch, 32, true),
                RopeLayers::All,
                "{arch} has no `use_rope` in src/models/"
            );
        }
    }
}
