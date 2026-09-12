//! MiMo-V2, checked against llama.cpp itself: a V head width that
//! differs from the K head width.
//!
//! `mimo2` was triaged NEW CODE on one thing no other generic-path
//! graph has: `conversion/mimo.py:154` writes `attention.value_length`
//! from `v_head_dim` apart from the `attention.key_length` the base
//! converter writes from `head_dim` (`192` / `128` on MiMo-V2-Flash),
//! and `src/models/mimo2.cpp:47-48,132-140,152-154` size and view K
//! and V separately with `wo` at `n_embd_head_v * n_head` (`:52`).
//! Every KV cache, attention kernel and projection check in ferrox
//! took ONE head width, and the loader refused the file.
//! `ferrox_models::kv_head_dims` is the seam and has the census:
//! fourteen converters write `value_length`, three write it apart from
//! `key_length`, one on this engine. Its second, small half is
//! `attention.value_scale` (`:14-17,180-183`, `0.707` on every real
//! export): `ferrox_models::attn_value_scale`.
//!
//! # What the fixture carries, and why all of it
//!
//! Every other thing a real MiMo-V2 file needs had a seam already, and
//! the fixture carries each so the new width is measured in their
//! company rather than alone: the per-layer `head_count_kv` ARRAY
//! (`[2, 1, 2]`), the per-layer `sliding_window_pattern` array
//! (`[1, 0, 1]`) with a window of 3 (narrower than the six-token
//! prompt, so it bites) and `rope.freq_base_swa = 100`, `attn_sinks`
//! on every layer, a dense layer 0 and MoE layers 1-2 with sigmoid
//! gating, `exp_probs_b` and `expert_weights_scale = 2.5`, and partial
//! NEOX RoPE (`rope.dimension_count = 8` over a 12-wide K head).
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `mimo2` | the converter's FUSED `attn_qkv` (Q rows at 12, K rows at 12, V rows at 8), `value_scale = 0.707` |
//! | `mimo2_split` | the same weights as split `attn_q` / `attn_k` / `attn_v` (`:142-155`); libllama's logits are BYTE-IDENTICAL to the fused file's |
//! | `mimo2_novscale` | no `attention.value_scale` key (`:14-17`, no scale); its golden differs from the first |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own `mimo2`
//! graph over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//! Not by re-reading a spec, and not by ferrox checking itself.
//!
//! Measured against that reference over `GRAPH_PROMPT`, all three
//! fixtures being F32 (`report_kl_against_llama_cpp` prints this):
//!
//! | fixture | KL(llama.cpp \|\| ferrox) | max abs logit delta |
//! |---|---|---|
//! | `mimo2` | 5.42e-15 | 3.58e-07 |
//! | `mimo2_split` | 5.42e-15 | 3.58e-07 |
//! | `mimo2_novscale` | 3.49e-15 | 1.94e-07 |
//!
//! Building it found that `expert_weights_scale` was honoured for EVERY
//! architecture here while llama.cpp reads the key in twenty
//! per-architecture loaders and nowhere else: the fixture declares
//! `2.5`, `mimo2.cpp` never reads it, and libllama's golden is
//! unscaled. `EXPERT_WEIGHTS_SCALE_READERS` / `EXPERT_WEIGHTS_NORM_READERS`
//! in the loader are the readers, measured.
//!
//! Regenerating (both halves must be redone together if a fixture
//! changes):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_mimo2_fixture.py \
//!     crates/ferrox-models/tests/fixtures/mimo2_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_mimo2_fixture.py \
//!     crates/ferrox-models/tests/fixtures/mimo2_split_tiny.gguf --split
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_mimo2_fixture.py \
//!     crates/ferrox-models/tests/fixtures/mimo2_novscale_tiny.gguf --no-value-scale
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, kl_vs_golden,
    load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use ferrox_models::config::RopeLayout;
use ferrox_models::{Decoder, ModelConfig};
use ferrox_moe::GatingFunction;

const FUSED: &str = "mimo2";
const SPLIT: &str = "mimo2_split";
const NO_VSCALE: &str = "mimo2_novscale";

const MIMO2_GOLDEN: [f32; 48] = [
    -0.2922334,
    0.13215135,
    -0.010728233,
    0.280187,
    -0.2590774,
    0.46919924,
    -0.14346087,
    -0.4925399,
    -0.0689113,
    -0.57132673,
    0.16720971,
    0.2164537,
    -0.48464936,
    -0.33445397,
    0.1300167,
    0.26022017,
    -0.08508543,
    0.33771738,
    0.04474199,
    0.0901041,
    -0.37434575,
    0.2058759,
    0.01687321,
    -0.27022967,
    0.22835478,
    0.5829097,
    0.100595295,
    0.270191,
    -0.12631035,
    -0.15130028,
    0.03268848,
    0.37676102,
    0.19175436,
    0.33096537,
    0.16553192,
    0.08288535,
    0.078797646,
    0.16736573,
    0.19621119,
    0.3383339,
    -0.04659992,
    -0.40250474,
    -0.08039723,
    0.09582412,
    0.10921795,
    -0.07999155,
    -0.2329648,
    0.1259083,
];

const MIMO2_NOVSCALE_GOLDEN: [f32; 48] = [
    -0.29295748,
    0.12065003,
    -0.032091483,
    0.25790793,
    -0.27376854,
    0.45487624,
    -0.16501981,
    -0.5107799,
    -0.06877457,
    -0.58691585,
    0.17269471,
    0.21428943,
    -0.46408775,
    -0.3248454,
    0.14189157,
    0.24064967,
    -0.08579496,
    0.3336945,
    0.019117132,
    0.09144734,
    -0.36427397,
    0.19630176,
    0.019454047,
    -0.28041232,
    0.23677847,
    0.58755136,
    0.102202326,
    0.2604071,
    -0.1267527,
    -0.14026551,
    0.025088914,
    0.3628003,
    0.18189442,
    0.31380445,
    0.15929961,
    0.10085506,
    0.052927185,
    0.18979812,
    0.21441442,
    0.32742763,
    -0.070622504,
    -0.3921549,
    -0.07894744,
    0.07657934,
    0.0930012,
    -0.111996494,
    -0.20993978,
    0.11554047,
];

#[test]
fn the_fused_qkv_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(FUSED, &MIMO2_GOLDEN);
}

/// The split spelling loads through the other branch of `qkv_fused`
/// and lands on the same numbers -- libllama's are byte-identical for
/// the two files, so one golden serves both.
#[test]
fn the_split_qkv_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(SPLIT, &MIMO2_GOLDEN);
}

#[test]
fn the_no_value_scale_shape_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(NO_VSCALE, &MIMO2_NOVSCALE_GOLDEN);
}

/// The measurement the module doc quotes.
#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (FUSED, &MIMO2_GOLDEN),
        (SPLIT, &MIMO2_GOLDEN),
        (NO_VSCALE, &MIMO2_NOVSCALE_GOLDEN),
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

/// What the loader resolved, value by value.
#[test]
fn the_loader_resolves_the_row_the_way_llama_cpp_does() {
    let d = load_graph_fixture(FUSED);
    let c = &d.config;
    assert_eq!(c.head_dim, 12, "attention.key_length");
    assert_eq!(c.v_head_dim(), 8, "attention.value_length");
    assert!(c.kv_head_dims_split());
    assert_eq!(c.rope_dim, Some(8), "partial rotary over the K head");
    assert_eq!(c.rope_layout, RopeLayout::Neox);
    assert_eq!(c.attn_value_scale, Some(0.707));
    // mimo2.cpp:227 passes the SIGMOID literal; the key is never read.
    assert_eq!(c.moe.gating, GatingFunction::Sigmoid);
    assert!(c.moe.norm_topk_prob, "`norm_w = true` literal at :227");
    // The file declares `expert_weights_scale = 2.5` and `mimo2.cpp`
    // reads the key nowhere, so libllama ran the fixture UNSCALED (the
    // golden) and so must ferrox: `EXPERT_WEIGHTS_SCALE_READERS`. This
    // was the bisection's finding -- every other knob in the fixture
    // was right and this one key moved the logits by 1e-1.
    assert_eq!(c.moe.expert_weights_scale, 1.0);
    // The per-layer kv-head array and the window array, both from the
    // converter's arrays.
    let kv_heads: Vec<usize> = (0..3)
        .map(|il| c.layer_shape(il).attention.n_kv_heads())
        .collect();
    assert_eq!(kv_heads, [2, 1, 2]);
    let windows: Vec<Option<usize>> = (0..3).map(|il| c.layer_sliding_window(il)).collect();
    assert_eq!(windows, [Some(3), None, Some(3)]);
    assert_eq!(
        c.layer_rope_theta(0),
        Some(100.0),
        "rope.freq_base_swa on a sliding layer"
    );
    assert_eq!(c.layer_rope_theta(1), Some(10000.0));
    // Every layer routed, with sinks and exp_probs_b.
    for (il, layer) in d.layers.iter().enumerate() {
        assert!(layer.attn.sinks.is_some(), "layer {il} sinks");
        let n_kv = c.layer_shape(il).attention.n_kv_heads();
        // V and o_proj at the V width; K at the K width.
        assert_eq!(layer.attn.k_proj.rows(), n_kv * 12, "layer {il} K rows");
        assert_eq!(layer.attn.v_proj.rows(), n_kv * 8, "layer {il} V rows");
        assert_eq!(layer.attn.o_proj.cols(), 4 * 8, "layer {il} wo cols");
        assert!(layer.moe.exp_probs_bias.is_some(), "layer {il}");
    }
    // The caches the model builds for itself carry both widths.
    let caches = graph_caches(&d);
    assert_eq!(caches[0].head_dim, 12);
    assert_eq!(caches[0].v_head_dim, 8);
    assert_eq!(caches[1].n_kv_heads, 1);
    // And the no-scale file resolves to no scale, not to 1.0.
    assert_eq!(load_graph_fixture(NO_VSCALE).config.attn_value_scale, None);
}

/// Reading V at the K width -- what every kernel did before the seam --
/// is not a shape the loader can be talked into: the file's V rows are
/// `n_kv * 8` and the check names them.
#[test]
fn a_v_projection_sized_at_the_k_width_is_refused_naming_the_tensor() {
    let path = graph_fixture_path(SPLIT);
    let file = ferrox_gguf::GgufFile::open(&path).expect("fixture opens");
    let mut config = ModelConfig::from_gguf(&file).expect("header parses");
    // Lie about the V width: pretend it equals K's.
    config.v_head_dim = None;
    let msg = match Decoder::from_gguf(&path, config) {
        Ok(_) => panic!("a V width that contradicts the tensors loaded"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("blk.0.attn_v.weight"), "{msg}");
    assert!(msg.contains("v_head_dim 12"), "{msg}");
}

/// The same pair on an architecture whose graph asserts them equal is
/// refused at the header, naming the assert.
#[test]
fn a_split_width_on_an_architecture_that_asserts_them_equal_is_refused() {
    let err = ferrox_models::kv_head_dims::resolve_v_head_dim("qwen3moe", 192, Some(128))
        .expect_err("qwen3moe asserts n_embd_head_k == n_embd_head_v");
    let msg = err.to_string();
    assert!(msg.contains("split K/V head dims"), "{msg}");
    assert!(msg.contains("n_embd_head_v()"), "{msg}");
}

/// Dropping the value scale on the file that has it diverges from
/// llama.cpp; applying it on the file that has not diverges too. The
/// two goldens differ by construction, and this pins that ferrox
/// tracks the right one on each.
#[test]
fn the_value_scale_is_applied_where_the_file_declares_it_and_nowhere_else() {
    let mut d = load_graph_fixture(FUSED);
    d.config.attn_value_scale = None;
    assert!(decode_worst(&d, &MIMO2_GOLDEN) > 100.0 * GRAPH_TOL);
    let mut d = load_graph_fixture(NO_VSCALE);
    d.config.attn_value_scale = Some(0.707);
    assert!(decode_worst(&d, &MIMO2_NOVSCALE_GOLDEN) > 100.0 * GRAPH_TOL);
    // And the two goldens are not one: the key moved llama.cpp's own
    // logits.
    assert!(MIMO2_GOLDEN
        .iter()
        .zip(MIMO2_NOVSCALE_GOLDEN.iter())
        .any(|(a, b)| (a - b).abs() > 1e-3));
}

/// The batched prefill kernel at split widths and the row kernel agree
/// on a batch wide enough to take the batched arm on every layer,
/// including the sink-bearing per-query arm (every layer here has
/// sinks) -- twelve positions, past every threshold.
#[test]
fn the_prefill_body_agrees_with_the_row_body_at_split_widths() {
    let d = load_graph_fixture(FUSED);
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

/// `expert_weights_scale` is dead metadata for `mimo2` upstream
/// (`mimo2.cpp` never reads the key; the golden was produced with the
/// key in the file and is unscaled). Honouring it -- what ferrox did
/// for every architecture before the readers table -- diverges.
#[test]
fn honouring_the_expert_weights_scale_key_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture(FUSED);
    d.config.moe.expert_weights_scale = 2.5;
    let worst = decode_worst(&d, &MIMO2_GOLDEN);
    assert!(worst > 100.0 * GRAPH_TOL, "{worst}");
}

fn decode_worst(d: &Decoder, golden: &[f32]) -> f32 {
    let mut kv = graph_caches(d);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = d.forward_token(tok, pos, &mut kv);
    }
    worst_vs(&out, golden)
}
