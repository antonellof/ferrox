//! Llama 4 (`llama4`: Scout 17B-16E, Maverick 17B-128E), checked
//! against llama.cpp itself.
//!
//! `llama4.cpp` is a Llama graph with four things on top, each one
//! seam: the CHUNKED window (`:13-14`, `crate::chunked_swa`), the
//! attention temperature from literals on the layers that do NOT
//! rotate (`:15-17,175-176`, `attn_temperature::
//! LITERAL_ATTN_TEMPERATURE`), a weightless per-head RMS norm on Q and
//! K AFTER RoPE on the layers that do, for every expert count but 128
//! (`:43,182-188`, `crate::weightless_qk_norm`), and an interleave
//! step the TENSOR LOADER honours (`:64`, `moe_interleave::
//! INTERLEAVE_STEP_HONOURED_BY_LOADER`), with a shared expert at
//! `n_ff_exp` beside SIGMOID routing from a literal, `norm_w = false`
//! (`:86-89,228-241`).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `llama4` (16 experts, step 2: dense / MoE / dense / MoE; the QK norm; no window key) | see `report_kl_against_llama_cpp` | |
//! | `llama4_128e` (128 experts, no QK norm, a separate `output.weight`) | | |
//! | `llama4_long` (the default file at 8200 positions: the chunk boundary crossed, the temperature stepped) | | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_llama4_fixture.py \
//!     crates/frink-models/tests/fixtures/llama4_tiny.gguf [--128e --output | --noswa | --dense]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/llama4_tiny.gguf 3 7 11 19 23 5
//! REF_N_CTX=8256 /tmp/ref_logits crates/frink-models/tests/fixtures/llama4_tiny.gguf $(long_prompt)
//! ```
//!
//! Two more fixtures are REFUSAL evidence: `--noswa` (the converter's
//! `sliding_window 0` for an all-full-attention MobileLLM) makes
//! libllama abort at `llama-graph.cpp:159` (`GGML_ASSERT(
//! f_attn_temp_scale != 0.0f)`), and `--dense` (no experts) is refused
//! by its loader (`llama4.cpp:49-51`, "model cannot have zero
//! experts"). Both are refused by name here.

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    graph_fixture_path, kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::{BatchWindow, ModelConfig, RopeLayout};
use frink_models::Decoder;
use frink_moe::GatingFunction;

const L4: &str = "llama4";
const L4_128E: &str = "llama4_128e";

const L4_GOLDEN: [f32; 48] = [
    -1.5475562,
    0.77342916,
    1.2293538,
    -1.1329669,
    1.8428595,
    2.5622656,
    -0.93854386,
    0.5100415,
    -0.9415591,
    2.0834305,
    0.9729803,
    -3.8363652,
    2.1561897,
    0.3488644,
    0.69792974,
    2.2396395,
    0.77410734,
    1.0463085,
    0.21874388,
    -1.1137091,
    -1.3612422,
    1.1912389,
    -0.4672369,
    -0.9765758,
    1.029448,
    2.056127,
    0.45193216,
    2.0196543,
    1.7487644,
    -1.6167521,
    -0.35535514,
    -0.40712395,
    -0.43447474,
    -0.9924098,
    0.18440327,
    0.26770714,
    -0.5326547,
    0.17145872,
    0.9612616,
    0.6286746,
    0.25729966,
    -2.7964578,
    1.0113577,
    -0.84189683,
    -1.0806758,
    0.24338612,
    0.4874415,
    -0.23967063,
];

const L4_128E_GOLDEN: [f32; 48] = [
    0.5844998,
    -0.535391,
    0.75035745,
    -2.3311503,
    -0.36858213,
    -0.53630006,
    -0.26200026,
    0.04257828,
    -0.43551522,
    -0.7403396,
    1.3307201,
    0.052830927,
    -0.4171842,
    1.1698272,
    1.0279479,
    0.15975925,
    -1.0362741,
    -4.0258884,
    1.3929234,
    -0.2184426,
    -1.0330604,
    0.74741656,
    0.90167075,
    1.5435951,
    -1.1763881,
    2.0775855,
    1.7439289,
    -0.024317801,
    -0.5970681,
    0.89493954,
    0.47672153,
    2.0078604,
    -0.16099504,
    2.0555422,
    -1.5990355,
    -1.396646,
    -1.6255786,
    -0.9483558,
    1.2548604,
    1.271211,
    -1.258501,
    0.253671,
    0.6739031,
    0.70500726,
    -0.47921857,
    -0.57197535,
    0.27564216,
    -0.24839129,
];

fn decode(d: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(d);
    d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv)
}

/// Scout's shape: the QK norm on the rotating layers, the temperature
/// on the unrotated one, MoE on every second layer.
#[test]
fn llama4_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(L4, &L4_GOLDEN);
}

/// Maverick's shape: 128 experts switch the QK norm OFF (`llama4.cpp:
/// 43`); the file also carries a separate `output.weight`.
#[test]
fn llama4_128e_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(L4_128E, &L4_128E_GOLDEN);
    let d = load_graph_fixture(L4_128E);
    assert!(!d.config.weightless_qk_norm);
    assert_eq!(d.config.moe.n_experts, 128);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [(L4, &L4_GOLDEN), (L4_128E, &L4_128E_GOLDEN)] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built from a file that declares NO window key: the
/// literal chunk, the literal period, the literal temperature on the
/// one unrotated layer, the QK norm, the interleaved experts, sigmoid
/// gating with no renormalisation, the shared expert at `n_ff_exp`.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("llama4"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Norm
        })
    ));
    let d = load_graph_fixture(L4);
    let c = &d.config;
    assert_eq!(c.sliding_window, Some(8192), "llama4.cpp:14, a literal");
    assert!(c.swa_chunked);
    assert_eq!(c.rope_theta_swa, Some(c.rope_theta), "llama4.cpp:23");
    // 3 chunked, 1 full (`:19`); the full layer is the unrotated one
    // (`:145-146`, the default step of 4) and takes the temperature.
    for il in 0..3 {
        assert_eq!(c.layer_sliding_window(il), Some(8192), "blk.{il}");
        assert!(c.layer_rotates(il), "blk.{il}");
    }
    assert_eq!(c.layer_sliding_window(3), None);
    assert!(!c.layer_rotates(3));
    let t = c.attn_temperature.expect("llama4.cpp:15-17");
    assert_eq!((t.scale, t.floor_scale.get(), t.offset), (0.1, 8192, 1.0));
    assert!(t.unrotated_layers_only);
    assert!(c.weightless_qk_norm, "16 experts: use_kq_norm");
    assert!(d.layers.iter().all(|l| l.attn.q_norm.is_none()));
    // `:64`: step 2 makes layers 1 and 3 MoE and 0 and 2 dense.
    assert_eq!(c.moe_interleave_step, Some(2));
    assert!(c.layer_is_dense(0) && !c.layer_is_dense(1));
    assert!(c.layer_is_dense(2) && !c.layer_is_dense(3));
    assert_eq!(d.layers[0].moe.n_experts(), 1);
    assert!(d.layers[0].moe.shared_experts.is_empty());
    assert_eq!(d.layers[1].moe.n_experts(), 16);
    assert_eq!(d.layers[1].moe.shared_experts.len(), 1);
    assert_eq!(c.moe.gating, GatingFunction::Sigmoid, "llama4.cpp:230");
    assert!(!c.moe.norm_topk_prob, "llama4.cpp:228");
    assert_eq!(c.moe.n_shared_experts, 1);
    assert_eq!(c.moe.expert_ffn_dim, 16, "the routed and shared width");
    assert!(c.moe.routed_weight_before_ffn, "llama-graph.cpp:1947");
    // The dense layers load `feed_forward_length` from the tensors'
    // own shape (`llama4.cpp:91-93`, `n_ff` = 40 here).
    let dense_rows = d.layers[0].moe.with_expert(0, |ex| ex.gate.rows());
    assert_eq!(dense_rows, 40, "the dense width");
    let shexp = &d.layers[1].moe.shared_experts[0];
    assert_eq!(
        shexp.gate.rows(),
        16,
        "n_ff_shexp = n_ff_exp, llama4.cpp:86"
    );
}

/// Each of the four facts is visible in the logits: turning any one
/// off diverges from the golden, and restoring it lands back on it.
#[test]
fn every_llama4_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(L4);
    assert_decoder_matches_on_all_three_paths(&d, &L4_GOLDEN, GRAPH_TOL, "baseline");

    d.config.weightless_qk_norm = false;
    let worst = worst_vs(&decode(&d), &L4_GOLDEN);
    assert!(worst > 1e-2, "the QK norm not seen: {worst}");
    d.config.weightless_qk_norm = true;

    // The temperature on EVERY layer, as the key-driven readers apply
    // it: the golden is below the floor, so make the floor small.
    let t = d.config.attn_temperature.unwrap();
    d.config.attn_temperature = Some(frink_models::attn_temperature::AttnTemperature {
        floor_scale: std::num::NonZeroU32::new(2).unwrap(),
        unrotated_layers_only: false,
        ..t
    });
    let worst = worst_vs(&decode(&d), &L4_GOLDEN);
    assert!(worst > 1e-2, "the temperature gate not seen: {worst}");
    d.config.attn_temperature = Some(t);

    // Renormalised sigmoid weights, as `norm_w = true` would.
    d.config.moe.norm_topk_prob = true;
    let worst = worst_vs(&decode(&d), &L4_GOLDEN);
    assert!(worst > 1e-2, "norm_w not seen: {worst}");
    d.config.moe.norm_topk_prob = false;

    // The routing weight on the OUTPUT, as every other graph has it.
    d.config.moe.routed_weight_before_ffn = false;
    let worst = worst_vs(&decode(&d), &L4_GOLDEN);
    assert!(worst > 1e-2, "the weight site not seen: {worst}");
    d.config.moe.routed_weight_before_ffn = true;

    // Rotating the full-attention layer too.
    d.config.rope_layers = frink_models::rope_layers::RopeLayers::All;
    let worst = worst_vs(&decode(&d), &L4_GOLDEN);
    assert!(worst > 1e-2, "the no-RoPE layer not seen: {worst}");
    d.config.rope_layers = frink_models::rope_layers::rope_layers("llama4", 4, true, 0);
    assert_decoder_matches_on_all_three_paths(&d, &L4_GOLDEN, GRAPH_TOL, "restored");
}

/// The chunked mask, position by position: a query at `p` sees
/// `p % 8192 + 1` keys, and a batch inside the first chunk is full
/// causal to the blocked kernel.
#[test]
fn the_chunked_window_is_the_query_s_own_chunk() {
    let d = load_graph_fixture(L4);
    let c = &d.config;
    assert_eq!(c.layer_window_for_query(0, 0), Some(1));
    assert_eq!(c.layer_window_for_query(0, 8191), Some(8192));
    assert_eq!(c.layer_window_for_query(0, 8192), Some(1));
    assert_eq!(c.layer_window_for_query(0, 8200), Some(9));
    assert_eq!(c.layer_window_for_query(3, 8200), None, "the full layer");
    assert_eq!(c.batch_window(0, 0, 6), BatchWindow::Uniform(None));
    assert_eq!(c.batch_window(0, 8000, 192), BatchWindow::Uniform(None));
    assert_eq!(c.batch_window(0, 8000, 193), BatchWindow::PerQuery);
    assert_eq!(c.batch_window(3, 8000, 193), BatchWindow::Uniform(None));
    // A sliding model's answer, for contrast: the window, not the chunk.
    let mut sliding = c.clone();
    sliding.swa_chunked = false;
    assert_eq!(sliding.layer_window_for_query(0, 8200), Some(8192));
    assert_eq!(
        sliding.batch_window(0, 8000, 193),
        BatchWindow::Uniform(Some(8192))
    );
}

/// The two files libllama cannot run are refused by name: the
/// converter's `sliding_window 0` (llama-graph.cpp:159 aborts) and a
/// zero expert count (llama4.cpp:49-51 throws).
#[test]
fn the_two_shapes_llama_cpp_cannot_run_are_refused_by_name() {
    for (name, needle) in [
        ("llama4_noswa", "llama-graph.cpp:159"),
        ("llama4_dense", "cannot have zero"),
    ] {
        let file = frink_gguf::GgufFile::open(graph_fixture_path(name)).expect("opens");
        let err = ModelConfig::from_gguf(&file).expect_err(name);
        let msg = err.to_string();
        assert!(msg.contains(needle), "{name}: {msg}");
    }
}
const L4_LONG_GOLDEN: [f32; 48] = [
    -0.7119035,
    -0.3397321,
    -0.75749516,
    1.4765843,
    0.03815364,
    -1.2261777,
    -0.2391832,
    1.1882699,
    -1.8714976,
    1.2008265,
    0.48729414,
    0.35849994,
    -0.5644614,
    1.6515466,
    -0.48844355,
    -0.1269112,
    0.21342757,
    -0.70501274,
    0.021885358,
    -0.6487958,
    -0.35042533,
    -0.82658386,
    -1.183315,
    -1.428663,
    0.31103954,
    0.27568892,
    0.51128364,
    -1.1113261,
    0.016373962,
    0.25835213,
    0.2358028,
    0.44606423,
    2.4274902,
    0.6971911,
    2.1417193,
    1.7641784,
    -0.0035396703,
    -1.6380006,
    -2.2818022,
    0.81444186,
    -1.2722605,
    2.999013,
    1.395515,
    -0.45339054,
    0.621953,
    0.58589804,
    2.640603,
    2.2151496,
];

/// The prompt behind `L4_LONG_GOLDEN`: 8200 positions, so the last
/// eight sit in the SECOND chunk (`8192..8199`, seeing 1 to 8 keys on
/// the chunked layers and everything on the full one) and the
/// temperature has stepped once (`floor((8191 + 1) / 8192) = 1`).
fn long_prompt() -> Vec<usize> {
    (0..8200).map(|i| 3 + (i * 7919 + 13) % 45).collect()
}

/// The chunk boundary, measured: libllama's logits for the last of
/// 8200 positions, matched by the batched prefill body on a batch that
/// STRADDLES the boundary (`BatchWindow::PerQuery`, positions 8188 to
/// 8199) and by the row body decoding the last token
/// (`layer_window_for_query(8199)` is 8 keys). A SLIDING reading of
/// the same window -- what every other windowed model does, and what
/// `swa_chunked = false` spells -- sees 8192 keys at that position and
/// lands elsewhere.
///
/// The first 8188 positions go through the blocked kernel in one
/// batch (`Uniform(None)`: the whole batch is in the first chunk),
/// which is what keeps this test at a few seconds.
#[test]
fn the_chunk_boundary_matches_llama_cpp_on_the_prefill_and_row_bodies() {
    let d = load_graph_fixture(L4);
    let prompt = long_prompt();
    let (inside, straddle) = prompt.split_at(8188);
    assert_eq!(
        d.config.batch_window(0, 0, inside.len()),
        BatchWindow::Uniform(None)
    );
    assert_eq!(
        d.config.batch_window(0, inside.len(), straddle.len()),
        BatchWindow::PerQuery
    );

    let mut kv = graph_caches(&d);
    d.forward_batch_last(inside, 0, &mut kv);
    let at_boundary = kv.clone();

    let prefill = d.forward_batch_last(straddle, inside.len(), &mut kv);
    let worst = worst_vs(&prefill, &L4_LONG_GOLDEN);
    assert!(worst <= GRAPH_TOL, "prefill across the boundary: {worst}");

    let mut kv = at_boundary.clone();
    let (head, last) = straddle.split_at(straddle.len() - 1);
    d.forward_batch_last(head, inside.len(), &mut kv);
    let before_last = kv.clone();
    let decoded = d.forward_token(last[0], prompt.len() - 1, &mut kv);
    let worst = worst_vs(&decoded, &L4_LONG_GOLDEN);
    assert!(worst <= GRAPH_TOL, "decode at 8199: {worst}");

    let mut sliding = load_graph_fixture(L4);
    sliding.config.swa_chunked = false;
    let mut kv = before_last;
    let slid = sliding.forward_token(last[0], prompt.len() - 1, &mut kv);
    let worst = worst_vs(&slid, &L4_LONG_GOLDEN);
    assert!(
        worst > 1e-2,
        "a sliding window must not match the chunk: {worst}"
    );
}
