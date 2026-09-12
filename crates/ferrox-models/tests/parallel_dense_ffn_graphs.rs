//! Arctic and Grok-2, checked against llama.cpp itself: a dense FFN
//! SUMMED with the routed experts on every layer, and -- for Arctic --
//! a routed branch that reads the layer INPUT under a second norm.
//!
//! `arctic` was triaged NEW CODE on `src/models/arctic.cpp:118-154`: a
//! dense SiLU FFN sized `{n_embd, n_embd}` (`:38-42`) on
//! `ffn_norm(ffn_inp)`, added to `ffn_inp`; the router AND the experts
//! on `ffn_norm_exps(inpSA)` (`:45,135-152`), the residual stream as
//! it ENTERS the layer under a SECOND per-layer weight; the two summed
//! (`:154`). Grok-2 (`grok.cpp:171-184`) is the other graph of 140 that
//! sums a dense FFN with its experts -- on the same `cur` the router
//! reads, GELU, the sum scaled by `sqrt(2)/2` -- and had been refused
//! by name from `grok_dense_ffn_tiny.gguf`. The two are one table,
//! `ferrox_models::parallel_dense_ffn` (presence, sum scale), served
//! through the shared-expert slot; Arctic's operand is
//! `RouterInput::NormedLayerInput`, one graph of 140.
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `arctic` | the converter's shape: dense triple + router + `ffn_norm_exps` + experts on every layer, no scale key |
//! | `arctic_wscale` | the same file declaring `expert_weights_scale = 2.5`, which `arctic.cpp:3-14` never reads; libllama's logits are BYTE-IDENTICAL to `arctic`'s (measured), so the golden is shared and the test pins that ferrox ignores the key too |
//! | `grok_dense_ffn` | Grok-2: `grok_tiny` plus the dense triple, the sum scaled by `sqrt(2)/2` before `ffn_post_norm` |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//! Grok is compared at [`GELU_TABLE_TOL`] for the reason
//! `tests/grok_graphs.rs` gives: llama.cpp's f16 GELU table is the
//! approximate side.
//!
//! | fixture | KL(llama.cpp \|\| ferrox) | max abs logit delta |
//! |---|---|---|
//! | `arctic` | 6.23e-14 | 7.75e-07 |
//! | `arctic_wscale` | same golden, same numbers | |
//! | `grok_dense_ffn` | 3.28e-10 | 4.96e-05 (GELU table) |
//!
//! One thing the Grok-2 golden cannot show: `grok.cpp:180` scales the
//! sum by `sqrt(2)/2` and `:186` RMS-norms the result, and an RMSNorm
//! is invariant under a positive scalar up to its epsilon, so the
//! factor moves libllama's logits by the eps term only (2.5e-4 here,
//! measured by dropping it). ferrox applies it where upstream does;
//! the test pins the measurement rather than pretending to a
//! sabotage.
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_arctic_fixture.py \
//!     crates/ferrox-models/tests/fixtures/arctic_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_arctic_fixture.py \
//!     crates/ferrox-models/tests/fixtures/arctic_wscale_tiny.gguf --weights-scale 2.5
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_grok_fixture.py \
//!     crates/ferrox-models/tests/fixtures/grok_dense_ffn_tiny.gguf --dense-ffn
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_all_three_paths_match_within,
    assert_decoder_matches_on_all_three_paths, graph_caches, kl_vs_golden, load_graph_fixture,
    worst_vs, GELU_TABLE_TOL, GRAPH_PROMPT, GRAPH_TOL,
};
use ferrox_gguf::TensorSource;
use ferrox_models::parallel_dense_ffn::{parallel_dense_ffn, DensePresence};
use ferrox_models::router_input::RouterInput;
use ferrox_models::Decoder;

const ARCTIC: &str = "arctic";
const ARCTIC_WSCALE: &str = "arctic_wscale";
const GROK2: &str = "grok_dense_ffn";

const ARCTIC_GOLDEN: [f32; 48] = [
    0.05027893,
    -0.14217487,
    0.05858913,
    0.1049784,
    0.066366896,
    -0.16455166,
    0.09970479,
    0.204927,
    -0.24035208,
    -0.13215064,
    -0.60382706,
    0.31372398,
    0.12064396,
    0.5836151,
    -0.5339172,
    0.27403653,
    0.38266587,
    -0.5910626,
    -0.5953618,
    0.5192094,
    -0.4455191,
    1.0098794,
    0.21578619,
    -0.23193917,
    0.2448336,
    -0.0852384,
    0.03282144,
    0.116781235,
    0.2811427,
    0.061673924,
    -0.40997168,
    0.033094466,
    0.0857943,
    0.121605314,
    0.408826,
    0.3581158,
    -0.13064215,
    0.14064175,
    0.20259959,
    0.3292458,
    0.3153615,
    0.3310253,
    0.100842476,
    0.82923156,
    0.15710491,
    -0.068920046,
    -0.52754986,
    0.27026743,
];

const GROK2_GOLDEN: [f32; 48] = [
    -0.02123012,
    -0.0008031098,
    0.030255696,
    -0.049563676,
    -0.026045168,
    -0.08789676,
    -0.02589143,
    -0.1334206,
    0.014215416,
    -0.017605985,
    0.13694334,
    -0.028282624,
    -0.11195763,
    -0.13209228,
    0.043226466,
    0.13278718,
    -0.029626718,
    0.0008411446,
    0.15519008,
    0.06174244,
    -0.029303163,
    -0.074429564,
    0.08645898,
    0.07315082,
    -0.0058240956,
    0.18907215,
    0.047012743,
    -0.027040038,
    -0.01837196,
    -0.008848941,
    -0.1237416,
    -0.034202468,
    -0.029883496,
    -0.093388245,
    0.03624947,
    -0.05482963,
    0.13346094,
    0.07847034,
    -0.04524482,
    -0.33681315,
    -0.13255954,
    0.09112677,
    0.038498543,
    -0.07358104,
    -0.09981882,
    0.17744401,
    -0.083169885,
    0.16222343,
];

#[test]
fn arctic_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(ARCTIC, &ARCTIC_GOLDEN);
}

/// `expert_weights_scale` is read by nothing in `arctic.cpp`; libllama's
/// logits for the declaring file are byte-identical to the base file's
/// (measured), so the golden is the same array and ferrox must not be
/// the one that honours it (`EXPERT_WEIGHTS_SCALE_READERS`).
#[test]
fn arctic_ignores_expert_weights_scale_as_llama_cpp_does() {
    let file = ferrox_gguf::GgufFile::open(common::graph_fixture_path(ARCTIC_WSCALE)).unwrap();
    assert_eq!(
        file.metadata_f32("arctic.expert_weights_scale"),
        Some(2.5),
        "the fixture must declare the key for this to prove anything"
    );
    let decoder = load_graph_fixture(ARCTIC_WSCALE);
    assert_eq!(decoder.config.moe.expert_weights_scale, 1.0);
    assert_all_three_paths_match(ARCTIC_WSCALE, &ARCTIC_GOLDEN);
}

#[test]
fn grok2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(GROK2, &GROK2_GOLDEN, GELU_TABLE_TOL);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [(ARCTIC, &ARCTIC_GOLDEN), (GROK2, &GROK2_GOLDEN)] {
        let decoder = load_graph_fixture(name);
        let out = decode(&decoder);
        println!(
            "{name}: KL(llama.cpp || ferrox) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

fn decode(decoder: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(decoder);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = decoder.forward_token(tok, pos, &mut kv);
    }
    out
}

/// What the loader built, read back: the dense triple in the
/// shared-expert slot on every layer, Arctic without a scale and with
/// its second norm, Grok-2 with `sqrt(2)/2` and without.
#[test]
fn the_loaded_layers_are_the_tables_rows() {
    let arctic_row = parallel_dense_ffn("arctic").unwrap();
    assert_eq!(arctic_row.presence, DensePresence::Required);
    assert_eq!(arctic_row.sum_scale, None);
    let grok_row = parallel_dense_ffn("grok").unwrap();
    assert_eq!(grok_row.presence, DensePresence::Optional);
    assert_eq!(grok_row.sum_scale, Some(std::f32::consts::FRAC_1_SQRT_2));

    let arctic = load_graph_fixture(ARCTIC);
    assert_eq!(arctic.config.router_input, RouterInput::NormedLayerInput);
    for layer in &arctic.layers {
        assert_eq!(
            layer.moe.shared_experts.len(),
            1,
            "the dense FFN, in the slot"
        );
        assert_eq!(
            layer.moe.shared_experts[0].up.rows(),
            32,
            "n_embd by n_embd, not n_ff"
        );
        assert_eq!(layer.moe.parallel_sum_scale, None);
        assert_eq!(layer.moe.exps_norm.as_ref().map(Vec::len), Some(32));
        assert_eq!(layer.moe.n_experts(), 4);
    }

    let grok2 = load_graph_fixture(GROK2);
    assert_eq!(grok2.config.router_input, RouterInput::NormedFfnInput);
    for layer in &grok2.layers {
        assert_eq!(layer.moe.shared_experts.len(), 1);
        assert_eq!(
            layer.moe.parallel_sum_scale,
            Some(std::f32::consts::FRAC_1_SQRT_2)
        );
        assert!(layer.moe.exps_norm.is_none());
    }
    // Grok-1: no triple, no slot, no scale -- the `else` branch.
    let grok1 = load_graph_fixture("grok");
    for layer in &grok1.layers {
        assert!(layer.moe.shared_experts.is_empty());
        assert_eq!(layer.moe.parallel_sum_scale, None);
    }
}

/// Each thing the two goldens check, sabotaged one at a time on the
/// loaded decoder, moves the logits by far more than the tolerance:
/// the dense branch itself, Arctic's second norm, Grok-2's scale.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut arctic = load_graph_fixture(ARCTIC);
    assert_decoder_matches_on_all_three_paths(&arctic, &ARCTIC_GOLDEN, GRAPH_TOL, "baseline");

    // The dense branch dropped: experts alone.
    let saved: Vec<_> = arctic
        .layers
        .iter_mut()
        .map(|l| std::mem::take(&mut l.moe.shared_experts))
        .collect();
    let worst = worst_vs(&decode(&arctic), &ARCTIC_GOLDEN);
    assert!(worst > 1e-2, "dense branch not seen: {worst}");
    for (l, s) in arctic.layers.iter_mut().zip(saved) {
        l.moe.shared_experts = s;
    }

    // The routed branch's norm flattened to one: the operand is still
    // the layer input, but under the wrong weight.
    let saved: Vec<_> = arctic
        .layers
        .iter_mut()
        .map(|l| l.moe.exps_norm.replace(vec![1.0; 32]))
        .collect();
    let worst = worst_vs(&decode(&arctic), &ARCTIC_GOLDEN);
    assert!(worst > 1e-2, "ffn_norm_exps not seen: {worst}");
    for (l, s) in arctic.layers.iter_mut().zip(saved) {
        l.moe.exps_norm = s;
    }

    // The routed branch fed the post-attention residual instead of the
    // layer input: the default operand, which is what a generic MoE
    // body would have computed.
    arctic.config.router_input = RouterInput::NormedFfnInput;
    let worst = worst_vs(&decode(&arctic), &ARCTIC_GOLDEN);
    assert!(worst > 1e-2, "branch operand not seen: {worst}");
    arctic.config.router_input = RouterInput::NormedLayerInput;
    assert_decoder_matches_on_all_three_paths(&arctic, &ARCTIC_GOLDEN, GRAPH_TOL, "restored");

    // Grok-2's sqrt(2)/2 dropped. This one is a MEASUREMENT, not a
    // sabotage: `grok.cpp:180-186` scales the sum and then RMS-norms it
    // (`ffn_post_norm`), and an RMSNorm is invariant under a positive
    // scalar up to its epsilon, so the factor is unobservable in
    // llama.cpp's own graph beyond eps. Dropping it here moves the
    // logits by 2.5e-4 on this fixture -- the eps term -- and a test
    // that demanded more would be demanding something the reference
    // cannot show. Pinned as "small and nonzero" so that a change that
    // made it LARGE (the scale applied after the norm, say) is caught.
    let mut grok2 = load_graph_fixture(GROK2);
    for l in grok2.layers.iter_mut() {
        l.moe.parallel_sum_scale = None;
    }
    let worst = worst_vs(&decode(&grok2), &GROK2_GOLDEN);
    assert!(
        worst > 0.0 && worst < 1e-3,
        "sqrt(2)/2 before the post-FFN norm is invisible beyond eps; measured {worst}"
    );
}
