//! Qwen3.5 (`qwen35`), checked against llama.cpp itself: the gated delta
//! net on the generic path (`crate::gdn`, `frink_core::gdn`,
//! `layer_shapes::AttnShape::Gdn`).
//!
//! `qwen35.cpp:126-152` runs every layer as `attn_norm` -> block ->
//! residual -> `post_attention_norm` -> SwiGLU -> residual, the block
//! being the delta net on the layers `attention.recurrent_layers` or
//! `(i + 1) % full_attention_interval != 0` name (`:17-24`) and gated
//! full attention on the rest (`:186-234`: the gate interleaved with
//! the query in `wq`, per-head QK norm, partial IMROPE over the
//! `rope.dimension_sections` -- NEOX band for band on text positions --
//! `sigmoid(gate) * attn` before `wo`). The delta net
//! (`:236-317`, `delta-net-base.cpp:289-365`) reads V head `h`'s keys
//! from K head `h % n_k_heads` (`llama-model.cpp:524-526`).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `qwen35` | see `report_kl_against_llama_cpp` | |
//! | `qwen35_array` | (`attention.recurrent_layers`; libllama byte-identical to `qwen35`) | |
//! | `qwen35_output` | (a separate `output.weight`) | |
//! | `qwen35moe` | (`qwen35moe`: `qwen2moe`'s FFN, softmax, a sigmoid-gated shared expert) | |
//! | `qwen3next` | (`qwen3next`: grouped V heads, fused `ssm_ba`, plain NEOX RoPE) | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_qwen35_fixture.py \
//!     crates/frink-models/tests/fixtures/qwen35_tiny.gguf [--array | --output]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/qwen35_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_all_three_paths_match_within,
    assert_decoder_matches_on_all_three_paths, graph_caches, kl_vs_golden, load_graph_fixture,
    worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath, QkNormStyle};
use frink_models::config::RopeLayout;
use frink_models::layer_shapes::AttnShape;
use frink_models::norm::NormOp;
use frink_models::Decoder;

const Q35: &str = "qwen35";
const Q35_ARRAY: &str = "qwen35_array";
const Q35_OUTPUT: &str = "qwen35_output";
const Q35MOE: &str = "qwen35moe";
const Q3NEXT: &str = "qwen3next";

/// The MoE row sits at 2e-5 max delta against libllama (KL 2.9e-11),
/// with the differences of either sign: the four routed SwiGLU experts
/// and the shared expert accumulate in a different order from ggml's
/// `mul_mat_id`. The `phimoe` / `orion` class; the dense rows stay at
/// `GRAPH_TOL`.
const Q35MOE_TOL: f32 = 5e-5;

const Q35_GOLDEN: [f32; 48] = [
    1.1752617,
    -0.10627127,
    0.33335793,
    0.6561886,
    -1.4498894,
    -0.19351739,
    -1.8791107,
    0.798954,
    1.6245198,
    -0.49334693,
    -1.5738757,
    0.45307195,
    -1.0303729,
    -1.7160652,
    -0.57821476,
    -0.52908224,
    1.0848951,
    0.109870985,
    -0.07334429,
    1.8102084,
    1.2377963,
    1.6395442,
    0.6890778,
    0.3326063,
    -0.10412532,
    0.7999805,
    -1.788507,
    -1.6497498,
    -1.8424459,
    -0.061935186,
    0.3562174,
    -0.8260913,
    -0.4012126,
    1.1811497,
    0.14384389,
    0.56542516,
    1.0739757,
    -0.89419425,
    -1.2602042,
    1.235939,
    1.5505302,
    -1.009949,
    2.3749595,
    -0.5552903,
    2.3903658,
    0.7304524,
    1.7889483,
    -0.71491814,
];

const Q35_OUTPUT_GOLDEN: [f32; 48] = [
    -2.210021,
    1.0246606,
    0.6213392,
    -3.0916357,
    -1.3098708,
    -0.23730385,
    0.7887325,
    0.8281358,
    -0.8026632,
    -0.85546494,
    -1.1705769,
    0.6187868,
    1.5126367,
    2.7802553,
    -2.7743077,
    -0.040545344,
    0.88707435,
    -0.1191566,
    -2.8073404,
    1.0185285,
    0.30770153,
    -0.4338447,
    0.8514995,
    -0.8660101,
    -2.231432,
    0.10326177,
    -2.2773354,
    -0.29860196,
    0.19373669,
    -3.8206549,
    1.8511194,
    -0.5398853,
    0.53501076,
    -3.0734997,
    0.6476717,
    0.39778078,
    1.7783022,
    0.067026764,
    0.38871494,
    -0.039477587,
    0.08354175,
    0.1955744,
    -0.56214714,
    -0.4750025,
    -0.57363975,
    -0.73621875,
    0.40045166,
    0.025140703,
];

const Q35MOE_GOLDEN: [f32; 48] = [
    0.22360724,
    -0.6776414,
    1.4371357,
    1.2821944,
    -1.5060322,
    -1.1283274,
    0.3802011,
    -0.56525075,
    1.2369745,
    -0.1832807,
    -1.6393037,
    3.4682975,
    -1.5699571,
    1.0261061,
    2.4505923,
    2.7590377,
    1.3118783,
    1.7313827,
    1.026756,
    1.170905,
    0.34071773,
    0.7808708,
    -0.69619524,
    1.2979344,
    1.1081562,
    -1.7758526,
    0.59420073,
    0.89650184,
    -0.336622,
    -0.6407294,
    1.7617202,
    -2.418144,
    0.1003468,
    -1.8583033,
    -1.1456127,
    0.50612867,
    0.059143424,
    1.0522888,
    -2.314548,
    0.1743792,
    -0.29793173,
    2.1630657,
    -0.3102206,
    -0.71420944,
    -0.21808052,
    0.2306974,
    0.037070632,
    0.87471986,
];

const Q3NEXT_GOLDEN: [f32; 48] = [
    -1.822459,
    2.3457584,
    1.3742877,
    -0.8353147,
    0.72063106,
    -0.26173767,
    -0.8956128,
    -1.3595253,
    -0.39872923,
    -0.41714746,
    -2.8663216,
    -2.0913703,
    -1.2430406,
    -1.6470684,
    0.74821794,
    -0.6727401,
    -1.1653898,
    0.19283068,
    0.8031446,
    -1.8707193,
    -0.872216,
    -0.03859979,
    1.4898968,
    1.2567704,
    -2.8511357,
    0.26563025,
    1.263978,
    -0.15689299,
    -3.245607,
    -1.1053009,
    0.6680958,
    -1.1433238,
    -0.5721313,
    2.0743752,
    0.18879429,
    -0.4341504,
    -1.9876447,
    0.18905528,
    0.65706193,
    -1.8302537,
    1.729474,
    0.5731117,
    0.49496412,
    1.674253,
    0.505594,
    1.0427579,
    -0.30545288,
    -2.0598164,
];

fn decode(decoder: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(decoder);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = decoder.forward_token(tok, pos, &mut kv);
    }
    out
}

#[test]
fn qwen35_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(Q35, &Q35_GOLDEN);
}

/// The same layout declared with `attention.recurrent_layers`, which
/// `qwen35.cpp:17` takes over the interval.
#[test]
fn the_recurrent_layers_array_matches_the_same_golden() {
    assert_all_three_paths_match(Q35_ARRAY, &Q35_GOLDEN);
}

#[test]
fn a_separate_output_weight_matches_llama_cpp() {
    assert_all_three_paths_match(Q35_OUTPUT, &Q35_OUTPUT_GOLDEN);
}

/// `qwen35moe`: the same layers with `qwen2moe`'s FFN
/// (`qwen35moe.cpp:496-538`): softmax over four experts, two used,
/// renormalised, plus a shared expert scaled by the sigmoid of its own
/// one-logit gate.
#[test]
fn qwen35moe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(Q35MOE, &Q35MOE_GOLDEN, Q35MOE_TOL);
    let d = load_graph_fixture(Q35MOE);
    assert!(matches!(
        resolve_architecture("qwen35moe"),
        Some(ArchPath::GenericGqa { .. })
    ));
    assert_eq!(d.config.moe.n_experts, 4);
    assert_eq!(d.config.moe.n_shared_experts, 1);
    assert!(d.config.moe.norm_topk_prob);
    assert!(
        d.layers[0].moe.shared_expert_gate.is_some(),
        "ffn_gate_inp_shexp"
    );
    assert_eq!(d.config.layer_shape(0).attention, AttnShape::Gdn);
}

/// `qwen3next`: `qwen35moe`'s layers with the V heads GROUPED over the
/// K heads (`qwen3next.cpp:521-539`), beta and alpha from one `ssm_ba`
/// projection (`:422-436`) and plain NEOX RoPE (`:282-291`). Tiling the
/// heads instead is a different graph: this golden is the grouped one.
#[test]
fn qwen3next_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(Q3NEXT, &Q3NEXT_GOLDEN, Q35MOE_TOL);
    let d = load_graph_fixture(Q3NEXT);
    assert!(matches!(
        resolve_architecture("qwen3next"),
        Some(ArchPath::GenericGqa { .. })
    ));
    let g = d.layers[0].attn.ssm.as_ref().unwrap().gdn().unwrap();
    assert_eq!(g.h.map, frink_core::gdn::HeadMap::Grouped);
    assert!(matches!(
        g.beta_alpha,
        frink_models::gdn::BetaAlpha::Fused { .. }
    ));
    assert_eq!(d.config.moe.n_experts, 4);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (Q35, &Q35_GOLDEN),
        (Q35_ARRAY, &Q35_GOLDEN),
        (Q35_OUTPUT, &Q35_OUTPUT_GOLDEN),
        (Q35MOE, &Q35MOE_GOLDEN),
        (Q3NEXT, &Q3NEXT_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: three delta-net layers and one gated
/// attention layer, its `wq` twice the query width, the pre-FFN norm
/// from `post_attention_norm`, per-head QK norm, partial rotation.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("qwen35"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(Q35);
    assert_eq!(d.config.qk_norm_style, QkNormStyle::PerHead);
    assert_eq!(d.config.rope_dim, Some(4));
    assert!(d.config.has_recurrent_layers());
    for il in 0..3 {
        assert_eq!(
            d.config.layer_shape(il).attention,
            AttnShape::Gdn,
            "blk.{il}"
        );
        let g = d.layers[il]
            .attn
            .ssm
            .as_ref()
            .unwrap()
            .gdn()
            .expect("delta net");
        assert_eq!(
            (g.h.d_conv, g.h.head_dim, g.h.n_k_heads, g.h.n_v_heads),
            (4, 8, 2, 4)
        );
        assert!(
            d.layers[il].moe.norm_weight != NormOp::None,
            "blk.{il} post_attention_norm"
        );
    }
    assert!(matches!(
        d.config.layer_shape(3).attention,
        AttnShape::Gqa {
            n_heads: 4,
            n_kv_heads: 2
        }
    ));
    let attn = &d.layers[3].attn;
    assert!(attn.q_gate_interleaved);
    assert_eq!(attn.q_proj.rows(), 2 * 4 * 8);
    assert_eq!(attn.q_norm.as_ref().map(Vec::len), Some(8));
    assert!(attn.ssm.is_none());
}

/// The paged backing carries the delta state, rows and state agreeing
/// with the contiguous one.
#[test]
fn paged_decode_matches_contiguous() {
    let d = load_graph_fixture(Q35);
    let store = std::sync::Arc::new(d.config.new_paged_kv(4, 8));
    let mut paged: Vec<frink_core::cache::PagedKvCache> = (0..d.config.n_layers)
        .map(|_| frink_core::cache::PagedKvCache::new())
        .collect();
    let mut contiguous = graph_caches(&d);
    let mut want = Vec::new();
    let mut got = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        want = d.forward_token(tok, pos, &mut contiguous);
        got = d
            .forward_token_paged(tok, pos, &mut paged, &store)
            .expect("8 blocks of 4 hold 6 positions");
    }
    assert_eq!(got, want);
    assert!(worst_vs(&got, &Q35_GOLDEN) < GRAPH_TOL);
    assert_eq!(paged[0].recurrent, contiguous[0].recurrent);
    assert_eq!(
        contiguous[3].rows(),
        GRAPH_PROMPT.len(),
        "the attention layer's rows"
    );
}

/// The delta state is visible: a decay of -30 forgets everything.
/// (The head map and the attention gate are pinned by the golden
/// itself: tiling the V heads the other way, or skipping the gate's
/// sigmoid, each moved the logits by more than 1e-2 when sabotaged.)
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(Q35);
    assert_decoder_matches_on_all_three_paths(&d, &Q35_GOLDEN, GRAPH_TOL, "baseline");
    let saved: Vec<f32> = d.layers[0]
        .attn
        .ssm
        .as_ref()
        .unwrap()
        .gdn()
        .unwrap()
        .a
        .clone();
    for a in d.layers[0]
        .attn
        .ssm
        .as_mut()
        .unwrap()
        .gdn_mut()
        .unwrap()
        .a
        .iter_mut()
    {
        *a = -30.0;
    }
    let worst = worst_vs(&decode(&d), &Q35_GOLDEN);
    assert!(worst > 1e-2, "the delta state not seen: {worst}");
    d.layers[0].attn.ssm.as_mut().unwrap().gdn_mut().unwrap().a = saved;
    assert_decoder_matches_on_all_three_paths(&d, &Q35_GOLDEN, GRAPH_TOL, "restored");
}
