//! Nemotron-H (`nemotron_h`), checked against llama.cpp itself: the
//! one-block-per-layer hybrid, on the Mamba-2 seam `granitehybrid`
//! opened (`crate::mamba2`, `layer_shapes::AttnShape::Mamba2`).
//!
//! `nemotron-h.cpp:143-158` runs every layer as ONE block under ONE
//! `attn_norm` with ONE residual add: Mamba-2 where `n_head_kv(i) == 0
//! && n_ff(i) == 0` (`:9-11`), attention where `n_ff(i) == 0` (no RoPE,
//! `:181-193`; optional `attn_output.bias`, `:75`), the ungated
//! ReLU-squared FFN otherwise (`:227-231`; optional biases, `:96-97`).
//! On the generic layer those are "a block with `ffn_dim 0`"
//! (`layer_shapes::BLOCK_WITHOUT_FFN_KEEPS_ITS_OUTPUT`: the output IS
//! added, unlike deci's) and "no block, an FFN" whose pre-norm is
//! `attn_norm` (`norm_sites::ONE_NORM_PER_LAYER`).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `nemotron_h` | see `report_kl_against_llama_cpp` | |
//! | `nemotron_h_biases` | (`attn_output.bias`, `ffn_up.bias`, `ffn_down.bias`) | |
//! | `nemotron_h_output` | (a separate `output.weight`) | |
//! | `nemotron_h_moe` | (`nemotron_h_moe`: the FFN layer a sigmoid MoE of ungated ReLU-squared experts with the router bias, plus an ungated shared expert) | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_nemotron_h_fixture.py \
//!     crates/frink-models/tests/fixtures/nemotron_h_tiny.gguf [--biases | --output]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/nemotron_h_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::{FfnActivation, RopeLayout};
use frink_models::layer_shapes::AttnShape;
use frink_models::norm::NormOp;
use frink_models::rope_layers::RopeLayers;
use frink_models::Decoder;
use frink_moe::GatingFunction;

const NH: &str = "nemotron_h";
const NH_BIASES: &str = "nemotron_h_biases";
const NH_OUTPUT: &str = "nemotron_h_output";
const NH_MOE: &str = "nemotron_h_moe";

const NH_GOLDEN: [f32; 48] = [
    0.49835235,
    -1.4659572,
    0.5571548,
    -1.1211274,
    -1.7389448,
    0.7609785,
    -2.770547,
    -0.1412692,
    -0.8707858,
    0.41768798,
    -0.0069743767,
    1.0850573,
    -0.082372606,
    -2.822941,
    0.8018663,
    -1.6038666,
    -0.35132936,
    -1.5974506,
    0.22312206,
    1.3650357,
    2.3566117,
    -0.2661721,
    1.3530788,
    0.68977743,
    1.8871508,
    0.6253891,
    -0.88075465,
    0.5687464,
    -3.1587484,
    2.3875623,
    -2.2404652,
    0.37548417,
    1.9546745,
    -0.90025973,
    -1.328382,
    0.2573392,
    2.4777703,
    0.41277438,
    -0.09976143,
    0.7568951,
    -0.6550994,
    0.99391955,
    -0.040353492,
    1.7089186,
    0.8152884,
    -1.2833884,
    1.4878078,
    0.49597543,
];

const NH_BIASES_GOLDEN: [f32; 48] = [
    1.1697109,
    -2.2336028,
    0.780195,
    -0.20783521,
    -3.6183677,
    -0.20179912,
    -0.031140111,
    -2.2032003,
    -0.6056086,
    -2.6798868,
    0.2811456,
    -1.6729908,
    0.21278548,
    -0.81446266,
    1.0800894,
    -1.3217437,
    -0.03867937,
    -0.8464547,
    -3.6989594,
    -0.42247808,
    1.495323,
    -0.9601151,
    -1.7219754,
    0.18881777,
    -0.72661465,
    1.5732236,
    -1.5485148,
    0.8623599,
    -3.5940664,
    0.123084635,
    -2.2586596,
    0.53104913,
    1.6465766,
    -3.6216521,
    1.8478383,
    1.7369587,
    1.0122374,
    -0.6400236,
    0.22576565,
    -0.42475307,
    0.118449725,
    0.9707836,
    0.42944506,
    0.08975468,
    -0.9211626,
    0.4513417,
    1.9414268,
    1.8472226,
];

const NH_OUTPUT_GOLDEN: [f32; 48] = [
    -0.08315659,
    -1.6332958,
    -0.6035919,
    1.1308731,
    -1.1232846,
    -0.9105842,
    -0.6359521,
    -0.6387201,
    0.8588952,
    -0.3319878,
    0.11025135,
    0.071847245,
    -1.5116369,
    -1.6171105,
    -0.5890583,
    -0.2039983,
    0.69057626,
    0.91685146,
    -1.1913584,
    -0.86281204,
    -0.7288308,
    -1.5927404,
    -1.7983351,
    4.6099043,
    0.08835098,
    0.15666425,
    -0.69174117,
    0.4165244,
    1.0937598,
    -3.5335526,
    0.3802758,
    -1.8192322,
    0.7732676,
    -0.5447902,
    -1.154995,
    0.43965477,
    -0.63115335,
    -2.7141044,
    0.9777248,
    -2.2609298,
    0.18054335,
    1.2854404,
    -0.7552384,
    3.065359,
    -0.83561087,
    1.2382677,
    0.10847542,
    -0.6347847,
];

const NH_MOE_GOLDEN: [f32; 48] = [
    -0.08564373,
    -1.6987907,
    1.8891659,
    -1.4673355,
    0.29491323,
    0.91232556,
    -0.8917374,
    0.025633007,
    -1.7253807,
    -0.01343681,
    0.707696,
    0.6290622,
    0.29452106,
    -3.4463603,
    -0.27748054,
    -0.4468344,
    -1.6725605,
    0.982375,
    -1.7984426,
    0.96634746,
    2.6736352,
    -0.88111913,
    -0.14342129,
    -1.6324458,
    0.91854924,
    1.2190024,
    -1.8341832,
    0.47902635,
    -1.6169864,
    1.8832711,
    -3.628996,
    1.2772206,
    0.22597688,
    -1.5320915,
    -0.7546907,
    0.52415156,
    2.734423,
    -0.40010202,
    1.4872544,
    0.3147399,
    -0.31410587,
    1.9143537,
    -0.88486254,
    0.8757988,
    0.6030685,
    -1.2249955,
    1.2363064,
    0.4540758,
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
fn nemotron_h_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(NH, &NH_GOLDEN);
}

/// The three optional biases (`:75,96-97`), each visible.
#[test]
fn the_optional_biases_match_llama_cpp_and_are_visible() {
    assert_all_three_paths_match(NH_BIASES, &NH_BIASES_GOLDEN);
    let mut d = load_graph_fixture(NH_BIASES);
    assert!(d.layers[1].attn.o_bias.is_some());
    assert!(d.layers[2].moe.dense_bias.is_some());
    d.layers[1].attn.o_bias = None;
    d.layers[2].moe.dense_bias = None;
    // Without them the file IS the plain fixture (same seed, same draws
    // up to the biases): the logits land on the plain golden's side of
    // the biased one, not on it.
    let worst = worst_vs(&decode(&d), &NH_BIASES_GOLDEN);
    assert!(worst > 1e-2, "biases not seen: {worst}");
}

#[test]
fn a_separate_output_weight_matches_llama_cpp() {
    assert_all_three_paths_match(NH_OUTPUT, &NH_OUTPUT_GOLDEN);
}

/// `nemotron_h_moe` (Nemotron-3 Nano 30B-A3B): the FFN layer is a
/// sigmoid MoE (`nemotron-h.cpp:206-231`, the gating function a literal,
/// `expert_weights_norm` / `_scale` from the file, the router bias
/// required) of UNGATED ReLU-squared experts, plus an ungated
/// ReLU-squared shared expert added to the routed sum. The fixture
/// declares `expert_weights_scale 2.5` and `expert_weights_norm true`,
/// both READ here (unlike `mimo2`'s), so the golden carries both.
#[test]
fn nemotron_h_moe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(NH_MOE, &NH_MOE_GOLDEN);
    let d = load_graph_fixture(NH_MOE);
    assert!(matches!(
        resolve_architecture("nemotron_h_moe"),
        Some(ArchPath::GenericGqa { .. })
    ));
    assert_eq!(d.config.moe.gating, GatingFunction::Sigmoid);
    assert!(d.config.moe.norm_topk_prob);
    assert_eq!(d.config.moe.expert_weights_scale, 2.5);
    assert_eq!(d.config.moe.n_experts, 4);
    assert_eq!(d.config.moe.n_shared_experts, 1);
    assert_eq!(d.config.ffn_activation, FfnActivation::ReluSqr);
    // The block layers carry no experts; the FFN layer routes with its
    // bias and its gate is the aliased `up` (no `ffn_gate_exps` in the
    // file).
    assert!(d.layers[0].moe.exp_probs_bias.is_none());
    assert!(d.layers[2].moe.exp_probs_bias.is_some());
    assert_eq!(d.layers[2].moe.router.rows(), 4);
    assert_eq!(d.config.layer_shape(2).attention, AttnShape::Absent);
    assert_eq!(d.config.layer_shape(2).ffn_dim, 16);
}

/// The routed sum's scale and the shared expert are each visible.
#[test]
fn the_moe_seams_are_visible_in_the_logits() {
    let mut d = load_graph_fixture(NH_MOE);
    assert_decoder_matches_on_all_three_paths(&d, &NH_MOE_GOLDEN, GRAPH_TOL, "baseline");
    d.config.moe.expert_weights_scale = 1.0;
    let worst = worst_vs(&decode(&d), &NH_MOE_GOLDEN);
    assert!(worst > 1e-2, "expert_weights_scale not seen: {worst}");
    d.config.moe.expert_weights_scale = 2.5;
    assert_decoder_matches_on_all_three_paths(&d, &NH_MOE_GOLDEN, GRAPH_TOL, "restored");
    let saved = std::mem::take(&mut d.layers[2].moe.shared_experts);
    let worst = worst_vs(&decode(&d), &NH_MOE_GOLDEN);
    assert!(worst > 1e-2, "the shared expert not seen: {worst}");
    d.layers[2].moe.shared_experts = saved;
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (NH, &NH_GOLDEN),
        (NH_BIASES, &NH_BIASES_GOLDEN),
        (NH_OUTPUT, &NH_OUTPUT_GOLDEN),
        (NH_MOE, &NH_MOE_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: three layer kinds from the two arrays, one
/// norm per layer (the FFN-only layer's is `attn_norm`), no rotation,
/// the ungated ReLU-squared FFN.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("nemotron_h"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(NH);
    assert_eq!(
        d.config.rope_layers,
        RopeLayers::Never,
        "no ggml_rope_ext in the graph"
    );
    assert_eq!(d.config.ffn_activation, FfnActivation::ReluSqr);
    let kinds: Vec<(AttnShape, usize)> = (0..4)
        .map(|il| {
            let s = d.config.layer_shape(il);
            (s.attention, s.ffn_dim)
        })
        .collect();
    assert_eq!(
        kinds,
        [
            (AttnShape::Mamba2, 0),
            (
                AttnShape::Gqa {
                    n_heads: 4,
                    n_kv_heads: 2
                },
                0
            ),
            (AttnShape::Absent, 40),
            (AttnShape::Mamba2, 0),
        ]
    );
    // The block layers norm at the attention slot and have no FFN norm;
    // the FFN-only layer norms at the FFN slot, with `attn_norm`.
    for il in [0, 1, 3] {
        assert!(
            d.layers[il].attn.norm_weight != NormOp::None,
            "blk.{il} attn_norm"
        );
        assert_eq!(
            d.layers[il].moe.norm_weight,
            NormOp::None,
            "blk.{il} has no FFN"
        );
    }
    assert_eq!(
        d.layers[2].attn.norm_weight,
        NormOp::None,
        "blk.2 has no block"
    );
    assert!(
        d.layers[2].moe.norm_weight != NormOp::None,
        "blk.2 norms with attn_norm"
    );
    assert!(d.layers[0].attn.ssm.is_some() && d.layers[3].attn.ssm.is_some());
}

/// Every layer's cache counts positions, whichever block it holds, and
/// the paged path agrees with the contiguous one.
#[test]
fn every_layer_counts_positions_and_paged_matches_contiguous() {
    let d = load_graph_fixture(NH);
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
    assert!(worst_vs(&got, &NH_GOLDEN) < GRAPH_TOL);
    for il in [0, 1, 3] {
        assert_eq!(contiguous[il].positions(), GRAPH_PROMPT.len(), "blk.{il}");
        assert_eq!(paged[il].seq_len(), GRAPH_PROMPT.len(), "blk.{il}");
    }
    assert!(contiguous[0].recurrent.is_some() && contiguous[1].recurrent.is_none());
}

/// The one residual add per layer is the block's output (nemotron-h.cpp:
/// 157), not deci's discarded branch: zeroing the attention layer's
/// `wo` moves the logits.
#[test]
fn a_block_without_an_ffn_still_reaches_the_residual() {
    let mut d = load_graph_fixture(NH);
    assert_decoder_matches_on_all_three_paths(&d, &NH_GOLDEN, GRAPH_TOL, "baseline");
    let zero = frink_core::weight_matrix::WeightMatrix::F32(frink_core::Tensor::new(
        vec![0.0; d.layers[1].attn.o_proj.rows() * d.layers[1].attn.o_proj.cols()],
        vec![
            d.layers[1].attn.o_proj.rows(),
            d.layers[1].attn.o_proj.cols(),
        ],
    ));
    let saved = std::mem::replace(&mut d.layers[1].attn.o_proj, zero);
    let worst = worst_vs(&decode(&d), &NH_GOLDEN);
    assert!(
        worst > 1e-2,
        "the attention layer's output not seen: {worst}"
    );
    d.layers[1].attn.o_proj = saved;
    assert_decoder_matches_on_all_three_paths(&d, &NH_GOLDEN, GRAPH_TOL, "restored");
}
