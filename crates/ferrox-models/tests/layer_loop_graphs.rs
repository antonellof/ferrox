//! Nanbeige, checked against llama.cpp itself: the same physical layers
//! run more than once.
//!
//! `nanbeige` was triaged NEW CODE on one thing no other graph has:
//! `src/models/nanbeige.cpp:6-12` read `num_loops` and
//! `skip_loop_final_norm`, `:19-31` set `n_layer_all = n_phys *
//! n_loops` and replicate the per-layer arrays, `:69-73` alias
//! `layers[i + j * n_phys] = layers[i]`, and `:167-175` norm the
//! residual with `output_norm` after every pass but the last unless the
//! flag skips it. `ferrox_models::layer_loops` is the seam: the weights
//! are shared and the KV is not, so `Decoder::layers` stays physical,
//! `ModelConfig::n_layers` is the logical count, `Decoder::layer_for`
//! is the one mapping, and the loop norm sits at the end of both FFN
//! bodies. The reach was MEASURED first: one graph of 140 reads either
//! key.
//!
//! # What each fixture is shaped to catch
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `nanbeige` | two physical layers, `num_loops = 2`: logical 0,1,0,1; the loop norm after logical 1 and NOT after logical 3 |
//! | `nanbeige_skipnorm` | the same four layers with `skip_loop_final_norm = true`: nothing between the passes |
//! | `nanbeige_loop1` | `num_loops = 1`: a plain two-layer Llama, the arm every non-looping export takes |
//!
//! `output_norm` is drawn away from one, so a loop norm skipped or
//! applied on the last pass moves the logits by far more than the
//! tolerance; the sabotage tests measure that rather than assume it.
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own
//! `nanbeige` graph over the fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp` (which prints `n_layer =
//! 4` for a two-block file with `num_loops = 2`). Not by re-reading a
//! spec, and not by ferrox checking itself.
//!
//! Measured against that reference over `GRAPH_PROMPT`, all three
//! fixtures being F32 (`report_kl_against_llama_cpp` prints this):
//!
//! | fixture | KL(llama.cpp \|\| ferrox) | max abs logit delta |
//! |---|---|---|
//! | `nanbeige` | see the test's output | |
//! | `nanbeige_skipnorm` | see the test's output | |
//! | `nanbeige_loop1` | see the test's output | |
//!
//! Regenerating (both halves must be redone together if a fixture
//! changes):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_nanbeige_fixture.py \
//!     crates/ferrox-models/tests/fixtures/nanbeige_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_nanbeige_fixture.py \
//!     crates/ferrox-models/tests/fixtures/nanbeige_skipnorm_tiny.gguf --skip-loop-norm
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_nanbeige_fixture.py \
//!     crates/ferrox-models/tests/fixtures/nanbeige_loop1_tiny.gguf --loops 1
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, kl_vs_golden, load_graph_fixture, worst_vs,
    GRAPH_PROMPT, GRAPH_TOL,
};
use ferrox_models::layer_loops::LayerLoops;
use ferrox_models::Decoder;

const LOOPED: &str = "nanbeige";
const SKIPNORM: &str = "nanbeige_skipnorm";
const LOOP1: &str = "nanbeige_loop1";

const NANBEIGE_GOLDEN: [f32; 48] = [
    1.1305385,
    -5.711855,
    0.8349031,
    -0.035932124,
    0.02799058,
    -2.0019739,
    2.732423,
    -2.4800794,
    0.13988751,
    -0.52112067,
    -2.8954809,
    -2.6291122,
    -0.9244696,
    -0.9872197,
    -0.17097169,
    2.1482935,
    -2.1112132,
    0.35528448,
    0.40608835,
    -2.5971627,
    -1.7644988,
    -0.118543744,
    3.217648,
    -3.7003458,
    2.5736485,
    -0.59041154,
    1.9001606,
    -1.9392523,
    -2.1275954,
    -0.115244925,
    -1.4734187,
    0.45592272,
    1.9284294,
    -1.8763123,
    -0.09153171,
    -0.5131397,
    -2.7508638,
    2.4147606,
    1.971565,
    1.6190343,
    -1.4798229,
    0.014007658,
    -2.8219712,
    -3.2271469,
    1.4229413,
    1.2155411,
    -1.0735766,
    -1.7230687,
];

const NANBEIGE_SKIPNORM_GOLDEN: [f32; 48] = [
    2.5658839,
    -2.7943225,
    0.28535634,
    1.3280082,
    -0.47646153,
    -0.76998156,
    3.0584044,
    -1.3222843,
    -1.2651266,
    1.8458974,
    -2.1073875,
    -0.9586396,
    -0.39816928,
    -0.41424656,
    0.36848706,
    4.3258176,
    -0.19926155,
    0.017288089,
    -0.6653415,
    0.43408036,
    -1.7050917,
    -1.265347,
    4.6364307,
    0.7094608,
    0.2864207,
    0.02199231,
    2.1767802,
    -2.6333094,
    0.42972475,
    0.5398185,
    -3.070771,
    -1.5311263,
    -2.116632,
    -0.07213342,
    2.7537897,
    -4.097416,
    -1.2504869,
    0.12223554,
    0.26501068,
    1.4385788,
    -2.302234,
    0.6442913,
    -2.9693336,
    -3.9248037,
    1.7215174,
    0.89901733,
    0.78444004,
    -0.67056036,
];

const NANBEIGE_LOOP1_GOLDEN: [f32; 48] = [
    3.7698843,
    -1.8963976,
    0.6540094,
    2.0661683,
    -1.4469407,
    1.096543,
    2.7851002,
    -0.3450266,
    -1.3187073,
    2.0070322,
    -0.063951254,
    -0.58663154,
    0.7075949,
    -0.8899846,
    -0.41187525,
    3.6233528,
    0.14984924,
    1.1246641,
    -0.28479582,
    1.9589229,
    -0.5773934,
    -3.0859282,
    3.9350853,
    1.2719893,
    -0.21432352,
    -1.4641414,
    1.2468581,
    -2.6249545,
    2.6324449,
    0.78799236,
    -3.2689028,
    -1.5629017,
    -0.20807657,
    1.0579505,
    3.4306884,
    -3.2170668,
    -1.7771192,
    -2.2918086,
    -0.93163383,
    1.219386,
    -1.3006366,
    1.6272469,
    -2.8724341,
    -1.7744331,
    4.024157,
    -1.380977,
    -0.3089743,
    -1.9155804,
];

#[test]
fn the_looped_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(LOOPED, &NANBEIGE_GOLDEN);
}

#[test]
fn the_skip_norm_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(SKIPNORM, &NANBEIGE_SKIPNORM_GOLDEN);
}

#[test]
fn the_single_pass_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(LOOP1, &NANBEIGE_LOOP1_GOLDEN);
}

/// The measurement the module doc quotes.
#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (LOOPED, &NANBEIGE_GOLDEN),
        (SKIPNORM, &NANBEIGE_SKIPNORM_GOLDEN),
        (LOOP1, &NANBEIGE_LOOP1_GOLDEN),
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

/// What the loader resolved: two physical layers, four logical, four
/// caches, one mapping; and the single-pass file resolves to no loop.
#[test]
fn the_loader_resolves_the_row_the_way_llama_cpp_does() {
    let d = load_graph_fixture(LOOPED);
    assert_eq!(
        d.config.layer_loops,
        Some(LayerLoops {
            n_phys: 2,
            n_loops: 2,
            skip_loop_final_norm: false,
        })
    );
    assert_eq!(d.config.n_layers, 4, "llama.cpp's n_layer for this file");
    assert_eq!(d.layers.len(), 2, "tensors for the physical layers only");
    assert_eq!(graph_caches(&d).len(), 4, "one KV cache per logical layer");
    // The same weights, twice each: layer_for is a mapping, not a copy.
    for l in 0..4 {
        assert!(
            std::ptr::eq(d.layer_for(l), &d.layers[l % 2]),
            "logical {l}"
        );
    }
    let skip = load_graph_fixture(SKIPNORM);
    assert!(skip.config.layer_loops.unwrap().skip_loop_final_norm);
    let one = load_graph_fixture(LOOP1);
    assert_eq!(
        one.config.layer_loops, None,
        "num_loops = 1 is a plain model"
    );
    assert_eq!(one.config.n_layers, 2);
}

/// Skipping the loop norm on the file that has it diverges from
/// llama.cpp; applying it on the file that skips it diverges too; and
/// running the physical layers ONCE (the mapping's absence) diverges
/// from both.
#[test]
fn the_loop_norm_and_the_second_pass_are_each_visible() {
    let mut d = load_graph_fixture(LOOPED);
    d.config.layer_loops = Some(LayerLoops {
        skip_loop_final_norm: true,
        ..d.config.layer_loops.unwrap()
    });
    assert!(
        decode_worst(&d, &NANBEIGE_GOLDEN) > 100.0 * GRAPH_TOL,
        "loop norm skipped"
    );

    let mut d = load_graph_fixture(SKIPNORM);
    d.config.layer_loops = Some(LayerLoops {
        skip_loop_final_norm: false,
        ..d.config.layer_loops.unwrap()
    });
    assert!(
        decode_worst(&d, &NANBEIGE_SKIPNORM_GOLDEN) > 100.0 * GRAPH_TOL,
        "loop norm applied where the file skips it"
    );

    // One pass over the same weights is the loop1 golden, not this one:
    // the goldens differ by construction.
    assert!(NANBEIGE_GOLDEN
        .iter()
        .zip(NANBEIGE_LOOP1_GOLDEN.iter())
        .any(|(a, b)| (a - b).abs() > 1e-3));
}

/// The batched body and the row body agree over twelve positions on the
/// looped file: the mapping and the loop norm are in both.
#[test]
fn the_prefill_body_agrees_with_the_row_body_on_a_looped_model() {
    let d = load_graph_fixture(LOOPED);
    let prompt: Vec<usize> = GRAPH_PROMPT
        .iter()
        .chain(GRAPH_PROMPT.iter())
        .copied()
        .collect();
    let mut kv = graph_caches(&d);
    let batched = d.forward_batch_last(&prompt, 0, &mut kv);
    let mut kv = graph_caches(&d);
    let mut rowwise = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        rowwise = d.forward_token(tok, pos, &mut kv);
    }
    let worst = worst_vs(&batched, &rowwise);
    assert!(worst < GRAPH_TOL, "batched vs row-wise differ by {worst}");
}

fn decode_worst(d: &Decoder, golden: &[f32]) -> f32 {
    let mut kv = graph_caches(d);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = d.forward_token(tok, pos, &mut kv);
    }
    worst_vs(&out, golden)
}
