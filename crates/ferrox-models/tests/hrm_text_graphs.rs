//! `hrm_text` (DFM Mimir 1B) against libllama: TWO residual streams and
//! two stacks replayed over twelve slots.
//!
//! `src/models/hrm-text.cpp` is the first decoder here that is not a
//! walk down one residual stream:
//!
//! ```text
//! zH = embeddings * embedding_scale        // :174
//! zL = hrm.z_l_init                        // :182, one [n_embd] row
//! for h in 0..h_cycles {                   // :183
//!     for l in 0..l_cycles { zL = stack(zH + zL) }
//!     zH = stack(zH + zL)
//! }
//! logits = output * zH                     // :198, NO final norm
//! ```
//!
//! and its stacks are ALIASES: `:47-87` creates `2 * layers_per_stack`
//! blocks (LOW at `[0, lps)`, HIGH at `[lps, 2*lps)`) while
//! `block_count` is the expanded slot count `lps * h * (l + 1)`, which
//! `:22-23` asserts. Each slot keeps its own KV.
//!
//! `ferrox_models::layer_loops::LayerLoops::Hrm` is the schedule --
//! which physical block a slot runs, where a stack starts, which
//! stream a finished stack writes -- and `ferrox_models::hrm` is the
//! state the four host bodies carry, one type with two methods rather
//! than four copies of "hold two vectors and add them here".
//!
//! The fixture is `lps = 2, h = 2, l = 2`: twelve slots over four
//! blocks, passes LOW LOW HIGH LOW LOW HIGH. Both counts are above one
//! on purpose -- at `l_cycles = 1` the passes alternate and a schedule
//! that had the aliasing backwards would still touch both stacks in
//! the right order.
//!
//! **Where the numbers come from.** `scripts/gptoss_reference_logits.cpp`
//! against a libllama built from `.scratch/llama.cpp` at `5b59b83`.
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_hrm_text_fixture.py \\
//!     crates/ferrox-models/tests/fixtures/hrm_text_tiny.gguf
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/hrm_text_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, kl_vs_golden, load_graph_fixture, worst_vs,
    GRAPH_PROMPT,
};
use ferrox_models::layer_loops::{HrmStream, LayerLoops, LoopNorm};
use ferrox_models::norm::NormOp;

const HRM: &str = "hrm_text";
const HRM_DEEP: &str = "hrm_text_deep";

/// The deep schedule's line, and where it comes from.
///
/// Every stack ends with a WEIGHTLESS RMS, so a renormalised residual
/// amplifies the ~1e-7 that two f32 reduction orders differ by, and the
/// amplification grows with the number of stacks. Measured across
/// three depths on arm64: two stacks exact at the suite default (3e-5),
/// three at 3.5e-5, six at 1.6e-4 -- and the six-stack file measured
/// 1.1e-3 on x86_64 CI, which is the same effect with another SIMD
/// width. So the deep fixture is THREE stacks, the shallowest schedule
/// that runs the LOW stack twice in a row, and the line is 5e-4: an
/// order over the arm64 number, comfortably over the x86 one at this
/// depth, and three orders under every sabotage below.
const HRM_DEEP_TOL: f32 = 5e-4;

/// llama.cpp's logits for `hrm_text_tiny.gguf` over [`GRAPH_PROMPT`].
const HRM_GOLDEN: [f32; 48] = [
    0.48859122,
    -0.027548946,
    -2.1163096,
    0.8374626,
    -0.40648997,
    -1.293237,
    -2.8178706,
    -0.8692963,
    -0.14350104,
    1.971564,
    0.14476222,
    1.0473267,
    -2.6039953,
    -1.0416603,
    -1.4675854,
    2.786407,
    1.3373334,
    2.3350945,
    -1.4018738,
    0.79700214,
    -0.3719061,
    -2.0813038,
    1.5884194,
    -3.7440796,
    1.1386358,
    -0.54637504,
    -2.2635274,
    -1.2015781,
    -1.4259753,
    0.04429221,
    -1.649365,
    0.33377907,
    -1.6249535,
    0.2575058,
    -0.26017588,
    0.59181,
    0.14685744,
    -0.6177069,
    3.3195543,
    0.04783091,
    0.563745,
    0.084713995,
    -0.9055257,
    -1.2128559,
    0.53929687,
    2.757055,
    0.09012246,
    1.6185268,
];

/// The same weights with TWO low passes per cycle.
const HRM_DEEP_GOLDEN: [f32; 48] = [
    0.90758383,
    0.22953302,
    -2.9290957,
    -0.91447467,
    0.3418988,
    -1.0681655,
    0.06694257,
    -1.3104312,
    -0.27177373,
    2.2454808,
    1.4815192,
    -0.39861965,
    -0.95136833,
    -0.24828023,
    -0.76831377,
    2.8056233,
    0.29154465,
    -0.73113346,
    -0.9889295,
    -0.19202757,
    -0.10745835,
    -2.554587,
    1.3898308,
    -2.4129229,
    -0.11541099,
    -0.4458959,
    -2.7572427,
    -1.4002583,
    2.796239,
    -1.2634301,
    0.12001756,
    -1.1158018,
    0.40241373,
    0.3288846,
    -1.8391914,
    -0.8395929,
    0.12101996,
    0.7995589,
    2.2497063,
    1.5525922,
    -0.7898507,
    1.5446601,
    -0.9307585,
    -3.3308845,
    -0.55149263,
    1.7406132,
    0.480614,
    1.1693362,
];

#[test]
fn hrm_text_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(HRM, &HRM_GOLDEN);
}

/// The number in the report, so it can be regenerated rather than
/// trusted. Run with `--nocapture` to see it.
/// The deep schedule: two LOW passes in a row, so the LOW stack runs
/// twice against two different cache slots.
#[test]
fn the_deep_schedule_matches_llama_cpp_on_all_three_paths() {
    common::assert_all_three_paths_match_within(HRM_DEEP, &HRM_DEEP_GOLDEN, HRM_DEEP_TOL);
}

#[test]
fn report_kl_against_llama_cpp() {
    let d = load_graph_fixture(HRM);
    let mut kv = graph_caches(&d);
    let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    let kl = kl_vs_golden(&got, &HRM_GOLDEN);
    let worst = worst_vs(&got, &HRM_GOLDEN);
    println!("| `hrm_text` | {kl:.2e} | {worst:.2e} |");
    assert!(kl < 1e-8, "hrm_text: KL {kl}");
}

/// The file holds FOUR blocks and the model runs TWELVE slots, each
/// slot with its own KV and the weights aliased by the schedule.
///
/// Structural, so a mapping that ran the wrong stack says which before
/// the numeric test says "diverged".
#[test]
fn twelve_slots_alias_four_blocks_in_the_order_hrm_text_cpp_runs_them() {
    let d = load_graph_fixture(HRM);
    assert_eq!(d.layers.len(), 4, "the file holds 2 * layers_per_stack");
    assert_eq!(d.config.n_layers, 8, "block_count is the slot count");
    assert_eq!(
        d.config.layer_loops,
        Some(LayerLoops::Hrm {
            lps: 2,
            h_cycles: 2,
            l_cycles: 1,
        })
    );
    let physical: Vec<usize> = (0..8)
        .map(|l| {
            d.layers
                .iter()
                .position(|w| std::ptr::eq(w, d.layer_for(l)))
                .expect("every slot aliases one of the blocks")
        })
        .collect();
    assert_eq!(physical, [0, 1, 2, 3, 0, 1, 2, 3], "LOW HIGH LOW HIGH");
    // The deep file is the other order: two LOW passes, then a HIGH.
    let deep = load_graph_fixture(HRM_DEEP);
    assert_eq!(deep.layers.len(), 4);
    assert_eq!(deep.config.n_layers, 6);
    let deep_physical: Vec<usize> = (0..6)
        .map(|l| {
            deep.layers
                .iter()
                .position(|w| std::ptr::eq(w, deep.layer_for(l)))
                .expect("every slot aliases one of the blocks")
        })
        .collect();
    assert_eq!(deep_physical, [0, 1, 0, 1, 2, 3], "LOW LOW HIGH");
    // Every slot has its own cache: the KV is per SLOT even though the
    // weights are per block.
    assert_eq!(d.config.new_kv_caches().len(), 8);
}

/// The stream schedule, and the norms that close each stack.
#[test]
fn every_stack_ends_with_a_weightless_norm_and_writes_one_stream() {
    let d = load_graph_fixture(HRM);
    let loops = d.config.layer_loops.expect("an HRM schedule");
    assert_eq!(
        (0..8)
            .filter(|&l| loops.loop_norm_after(l) == Some(LoopNorm::Weightless))
            .collect::<Vec<_>>(),
        [1, 3, 5, 7],
        "hrm-text.cpp:162 norms at the end of EVERY stack"
    );
    assert_eq!(
        (0..8)
            .filter_map(|l| loops.stream_after(l))
            .collect::<Vec<_>>(),
        [
            HrmStream::Low,
            HrmStream::High,
            HrmStream::Low,
            HrmStream::High,
        ]
    );
    let deep = load_graph_fixture(HRM_DEEP)
        .config
        .layer_loops
        .expect("an HRM schedule");
    assert_eq!(
        (0..12)
            .filter_map(|l| deep.stream_after(l))
            .collect::<Vec<_>>(),
        [
            HrmStream::Low,
            HrmStream::Low,
            HrmStream::High,
            HrmStream::Low,
            HrmStream::Low,
            HrmStream::High,
        ]
    );
    // There is no `output_norm` tensor to read: the last stack's own
    // norm is the final one (`norm_sites::NO_OUTPUT_NORM`).
    assert_eq!(d.final_norm, NormOp::None);
    // And every layer norms without a weight.
    assert!(d.hrm_z_l_init.is_some());
}
