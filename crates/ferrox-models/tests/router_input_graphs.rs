//! SmallThinker, checked against llama.cpp itself: the MoE router that
//! reads the RAW LAYER INPUT.
//!
//! `smallthinker` was triaged NEW CODE on three things, the first of
//! them a shape no other generic-path graph has: `smallthinker.cpp:111`
//! computes the router logits from `inpL` -- the residual stream as it
//! ENTERS the layer, before `attn_norm`, before attention -- and
//! `:151-161` hands them to `build_moe_ffn` as a precomputed `probs`
//! with a NULL `ffn_gate_inp`. Every other MoE body in ferrox routed on
//! the normed FFN input, which is the `build_moe_ffn` default and what
//! the experts read. The reach was MEASURED before a line was written
//! (`ferrox_models::router_input` has the table): four of 140 graphs
//! pass `probs_in`, and this is the only one on the generic path whose
//! operand differs. The other two: `LLM_FFN_RELU` experts with a REAL
//! gate (`ggml_reglu_split`, `FfnActivation::Reglu` -- NOT `arcee`'s
//! ungated `relu(up)^2`), and `n_swa` PINNED to 4096 over whatever the
//! file declares (`:8`, `capability::swa_window_override`).
//!
//! # What each fixture is shaped to catch
//!
//! | fixture | layers | window | gating | what it isolates |
//! |---|---|---|---|---|
//! | `smallthinker` | 5 | declared 3, pinned 4096 | sigmoid | the router operand; NoPE on 0 and 4; the pin |
//! | `smallthinker_nowindow` | 4 | none | softmax | `:16-18`: every layer rotates; the converter's other gating arm |
//! | `smallthinker_swa2` | 5 | declared 3, pinned 4096, `sliding_window_pattern = 2`, `rope.freq_base_swa = 100` | sigmoid | the SWA period from the KEY while the NoPE step stays the literal 4 |
//!
//! FIVE layers, so that an unrotated layer (`il % 4 == 0`: 0 and 4)
//! sits on top of rotated ones and the phase is visible at both ends.
//! The declared window of 3 is narrower than the six-token prompt on
//! purpose: honouring it would mask three of five layers and move the
//! logits by far more than the tolerance, and llama.cpp's logits for
//! this file and for the same file declaring 4096 are BYTE-IDENTICAL
//! (`cmp` on the two reference dumps), which is the measurement that
//! the pin is upstream's behaviour rather than a reading of it.
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own
//! `smallthinker` graph over the fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`. Not by re-reading a
//! spec, and not by ferrox checking itself.
//!
//! Measured against that reference over `GRAPH_PROMPT`, all three
//! fixtures being F32 so no `vec_dot_type` question arises
//! (`report_kl_against_llama_cpp` prints this table):
//!
//! | fixture | KL(llama.cpp \|\| ferrox) | max abs logit delta |
//! |---|---|---|
//! | `smallthinker` | 1.13e-14 | 5.96e-07 |
//! | `smallthinker_nowindow` | 1.27e-14 | 4.17e-07 |
//! | `smallthinker_swa2` | 6.56e-15 | 2.98e-07 |
//!
//! Regenerating (both halves must be redone together if a fixture
//! changes):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_smallthinker_fixture.py \
//!     crates/ferrox-models/tests/fixtures/smallthinker_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_smallthinker_fixture.py \
//!     crates/ferrox-models/tests/fixtures/smallthinker_nowindow_tiny.gguf --no-window
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_smallthinker_fixture.py \
//!     crates/ferrox-models/tests/fixtures/smallthinker_swa2_tiny.gguf --swa-period 2
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, kl_vs_golden, load_graph_fixture, worst_vs,
    GRAPH_PROMPT,
};
use ferrox_models::capability::{swa_window_override, SwaWindowOverride};
use ferrox_models::rope_layers::{NoRopePhase, RopeLayers};
use ferrox_models::router_input::RouterInput;
use ferrox_models::FfnActivation;
use ferrox_moe::GatingFunction;
use std::num::NonZeroUsize;
use std::sync::atomic::Ordering;

const WINDOWED: &str = "smallthinker";
const NO_WINDOW: &str = "smallthinker_nowindow";
const SWA2: &str = "smallthinker_swa2";

const SMALLTHINKER_GOLDEN: [f32; 48] = [
    -0.11644731,
    -0.09004255,
    -0.8652805,
    -0.1401813,
    0.06755003,
    0.39814454,
    -0.7316002,
    -0.46614897,
    -0.4459425,
    0.26024663,
    0.05146058,
    0.35796142,
    0.17482181,
    0.56213284,
    -0.13135439,
    0.49121585,
    0.6398525,
    0.19739252,
    0.20245396,
    0.49869946,
    0.16869414,
    0.15422034,
    -0.44662634,
    -0.3033952,
    -0.49946374,
    -0.39468992,
    0.39495352,
    -0.68436354,
    0.32730043,
    0.71584314,
    0.46055603,
    -0.5821071,
    0.04401402,
    -0.07890791,
    -0.512202,
    -0.32212967,
    0.522609,
    -0.6786745,
    0.68634814,
    -0.12438017,
    -0.7085644,
    0.88270426,
    -0.6042815,
    -0.15557137,
    -0.24430412,
    0.10377431,
    -0.97128,
    -0.69121647,
];

const SMALLTHINKER_NOWINDOW_GOLDEN: [f32; 48] = [
    0.15811518,
    0.0045722574,
    -0.09486203,
    0.014217727,
    0.093866035,
    0.35228467,
    -0.069771536,
    0.29466307,
    -0.68409336,
    -0.2999894,
    -0.07432782,
    -0.26421875,
    -0.21538484,
    0.21146034,
    -0.3649695,
    0.17957366,
    -0.3356589,
    0.26104423,
    -0.4966172,
    -0.4099353,
    -0.18745187,
    0.20364304,
    0.22314933,
    -0.709954,
    -0.16323356,
    -0.31290513,
    -0.5261392,
    0.0014657602,
    0.1779711,
    -0.12045112,
    0.20205164,
    0.13557914,
    -0.34491152,
    0.24907143,
    0.23877358,
    -0.018763602,
    0.115765415,
    -0.16348249,
    0.21951774,
    -0.5326319,
    0.56108963,
    -0.33193195,
    0.057139575,
    -0.217567,
    -0.2453047,
    -0.7356065,
    -0.27280375,
    -0.10678294,
];

const SMALLTHINKER_SWA2_GOLDEN: [f32; 48] = [
    -0.13205203,
    -0.0557603,
    -0.91823936,
    -0.16909967,
    0.021973073,
    0.344267,
    -0.7689327,
    -0.44909763,
    -0.45687044,
    0.1885989,
    0.03978818,
    0.35996455,
    0.18049982,
    0.5335289,
    -0.107659906,
    0.52262235,
    0.6458509,
    0.24143234,
    0.11847356,
    0.50185907,
    0.07708104,
    0.15330389,
    -0.408247,
    -0.27529505,
    -0.55876666,
    -0.36203945,
    0.40324852,
    -0.65618324,
    0.29595768,
    0.7243937,
    0.45502502,
    -0.5877708,
    0.041427076,
    -0.07837877,
    -0.49255326,
    -0.3359354,
    0.49686372,
    -0.68598735,
    0.7717606,
    -0.13895339,
    -0.6594104,
    0.862266,
    -0.6058197,
    -0.10096485,
    -0.23063655,
    0.15476298,
    -0.8784477,
    -0.7100779,
];

#[test]
fn the_windowed_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(WINDOWED, &SMALLTHINKER_GOLDEN);
}

#[test]
fn the_no_window_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(NO_WINDOW, &SMALLTHINKER_NOWINDOW_GOLDEN);
}

#[test]
fn a_keyed_swa_period_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(SWA2, &SMALLTHINKER_SWA2_GOLDEN);
}

/// The table in the module doc.
#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (WINDOWED, &SMALLTHINKER_GOLDEN),
        (NO_WINDOW, &SMALLTHINKER_NOWINDOW_GOLDEN),
        (SWA2, &SMALLTHINKER_SWA2_GOLDEN),
    ] {
        let d = load_graph_fixture(name);
        let mut kv = graph_caches(&d);
        let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
        let kl = kl_vs_golden(&got, golden);
        let worst = worst_vs(&got, golden);
        println!("| `{name}` | {kl:.2e} | {worst:.2e} |");
        assert!(kl < 1e-8, "{name}: KL {kl}");
    }
}

/// What the loader resolved, value by value: the operand, the
/// activation, the pinned window, the gating, the NoPE phase.
#[test]
fn the_loader_resolves_the_row_the_way_llama_cpp_does() {
    let d = load_graph_fixture(WINDOWED);
    assert_eq!(d.config.router_input, RouterInput::RawLayerInput);
    assert_eq!(d.config.ffn_activation, FfnActivation::Reglu);
    assert!(!d.config.ffn_is_ungated(), "the gate is a real tensor");
    // smallthinker.cpp:8 over the file's 3.
    assert_eq!(d.config.sliding_window, Some(4096));
    assert_eq!(
        swa_window_override("smallthinker", 5),
        SwaWindowOverride::Pin(4096)
    );
    assert_eq!(d.config.moe.gating, GatingFunction::Sigmoid);
    assert!(
        d.config.moe.norm_topk_prob,
        "norm_w = true, a literal at :158"
    );
    assert_eq!(d.config.moe.expert_weights_scale, 1.0);
    // :11 `set_swa_pattern(4, true)`: `il % 4 != 0` slides.
    for il in 0..5 {
        assert_eq!(
            d.config.layer_sliding_window(il),
            (il % 4 != 0).then_some(4096),
            "layer {il}"
        );
    }
    // :108-109 with the default step of 4: `il % 4 == 0` unrotated.
    assert_eq!(
        d.config.rope_layers,
        RopeLayers::NoRopeEvery {
            step: NonZeroUsize::new(4).unwrap(),
            phase: NoRopePhase::FirstOfPeriod
        }
    );

    let n = load_graph_fixture(NO_WINDOW);
    assert_eq!(n.config.router_input, RouterInput::RawLayerInput);
    assert_eq!(n.config.sliding_window, None);
    assert_eq!(n.config.moe.gating, GatingFunction::Softmax);
    // :18 `n_no_rope_layer_step = n_layer`: everything rotates.
    assert_eq!(n.config.rope_layers, RopeLayers::All);

    let s = load_graph_fixture(SWA2);
    assert_eq!(s.config.sliding_window, Some(4096));
    for il in 0..5 {
        assert_eq!(
            s.config.layer_sliding_window(il),
            (il % 2 != 0).then_some(4096),
            "layer {il}: the period is the KEY's 2"
        );
    }
    // The NoPE step is NOT the key's period.
    assert_eq!(
        s.config.rope_layers,
        RopeLayers::NoRopeEvery {
            step: NonZeroUsize::new(4).unwrap(),
            phase: NoRopePhase::FirstOfPeriod
        }
    );
    assert_eq!(s.config.rope_theta_swa, Some(100.0));
}

/// The routing decision itself differs between the two operands on
/// this prompt -- measured on the expert activation counters, not
/// assumed from the weights being random. Then: routing on the normed
/// FFN input, which every other MoE body does, diverges from
/// llama.cpp.
#[test]
fn routing_on_the_normed_input_picks_different_experts_and_diverges_from_llama_cpp() {
    let raw = load_graph_fixture(WINDOWED);
    let mut kv = graph_caches(&raw);
    let got = raw.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    assert!(worst_vs(&got, &SMALLTHINKER_GOLDEN) < 1e-5, "the premise");
    let counts = |d: &ferrox_models::Decoder| -> Vec<Vec<u64>> {
        d.layers
            .iter()
            .map(|l| {
                l.moe
                    .activation_counts
                    .iter()
                    .map(|c| c.load(Ordering::Relaxed))
                    .collect()
            })
            .collect()
    };
    let raw_counts = counts(&raw);

    let mut normed = load_graph_fixture(WINDOWED);
    normed.config.router_input = RouterInput::NormedFfnInput;
    let mut kv = graph_caches(&normed);
    let got = normed.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    let normed_counts = counts(&normed);

    let layers_that_differ = raw_counts
        .iter()
        .zip(normed_counts.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        layers_that_differ > 0,
        "the two operands picked the SAME experts on every layer, so this fixture cannot \
         see the seam: raw {raw_counts:?}, normed {normed_counts:?}"
    );
    println!("layers whose top-2 differs between the two operands: {layers_that_differ} of 5");
    let worst = worst_vs(&got, &SMALLTHINKER_GOLDEN);
    assert!(
        worst > 1e-2,
        "routing on the normed input moved the output by only {worst}"
    );
}

/// The gate is real. Running the experts as `arcee`'s ungated
/// `relu(up)^2` -- which is what a loader that put both spellings on
/// one variant would do -- diverges; so does SwiGLU on the same pair.
#[test]
fn the_experts_are_gated_relu_and_the_other_two_spellings_diverge() {
    for (act, what) in [
        (
            FfnActivation::ReluSqr,
            "ungated relu(up)^2 on the gated pair",
        ),
        (FfnActivation::Swiglu, "SwiGLU"),
    ] {
        let mut d = load_graph_fixture(WINDOWED);
        d.config.ffn_activation = act;
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &SMALLTHINKER_GOLDEN,
        );
        assert!(worst > 1e-2, "{what} moved the output by only {worst}");
    }
}

/// The pin is load-bearing on this fixture: honouring the declared 3
/// masks three of five layers over a six-token prompt and diverges
/// from llama.cpp, which masked nothing.
#[test]
fn honouring_the_declared_window_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture(WINDOWED);
    assert_eq!(d.config.sliding_window, Some(4096), "the premise");
    d.config.sliding_window = Some(3);
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &SMALLTHINKER_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "honouring the file's window moved the output by only {worst}"
    );
}

/// The NoPE phase: rotating everything, or skipping the LAST layer of
/// each period (`smollm3`'s phase), each diverges.
#[test]
fn the_nope_phase_is_the_first_layer_of_each_period() {
    for (layers, what) in [
        (RopeLayers::All, "rotating every layer"),
        (
            RopeLayers::NoRopeEvery {
                step: NonZeroUsize::new(4).unwrap(),
                phase: NoRopePhase::LastOfPeriod,
            },
            "smollm3's phase",
        ),
    ] {
        let mut d = load_graph_fixture(WINDOWED);
        d.config.rope_layers = layers;
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            &SMALLTHINKER_GOLDEN,
        );
        assert!(worst > 1e-2, "{what} moved the output by only {worst}");
    }
    // And on the no-window file, NOT rotating layer 0 diverges: :18
    // sets the step to n_layer.
    let mut d = load_graph_fixture(NO_WINDOW);
    d.config.rope_layers = RopeLayers::NoRopeEvery {
        step: NonZeroUsize::new(4).unwrap(),
        phase: NoRopePhase::FirstOfPeriod,
    };
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &SMALLTHINKER_NOWINDOW_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "skipping layer 0's rotation on the no-window file moved the output by only {worst}"
    );
}

/// On the keyed-period file the SWA base is what makes the period
/// visible (the window never bites at 4096): the seeded period of 4
/// ropes layer 2 at the wrong base and layer 3 at the wrong one too.
#[test]
fn the_swa_period_comes_from_the_key_not_the_nope_step() {
    let mut d = load_graph_fixture(SWA2);
    d.config.swa_layers = ferrox_models::swa_layers::SwaLayers::Period {
        period: NonZeroUsize::new(4).unwrap(),
        dense_first: true,
    };
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &SMALLTHINKER_SWA2_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "the seeded period on the keyed file moved the output by only {worst}"
    );
}
