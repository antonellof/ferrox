//! MiniMax-M2, checked against llama.cpp itself: the row its refusal had
//! called "UNAUDITED, not unimplemented" for a week, on the fixture that
//! had evidenced the claim (`tests/minimax_refusal.rs`,
//! `scripts/make_minimax_fixture.py`).
//!
//! `minimax-m2.cpp` is plain GQA (`:26`), ONE RMSNorm over the whole Q
//! projection and one over K (`:30-31`, `attn_q_norm` is
//! `n_embd_head_k * n_head` wide: `QkNormStyle::WholeVector`), partial NEOX RoPE
//! (`:96-106`; the real model is `n_rot 64` of `head_dim 128`), and one
//! SiLU MoE on every layer with `exp_probs_b`, `norm_w = true` and the
//! gating function from `expert_gating_func` (`:131-141`; SIGMOID on
//! every real export, since the default aborts upstream). No dense
//! layer, no shared expert, no biases, and `expert_weights_scale` is
//! read by nothing in its hparams.
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `minimax_m2` | 3.44e-15 | 2.09e-07 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_minimax_fixture.py \
//!     crates/frink-models/tests/fixtures/minimax_m2_tiny.gguf
//! /tmp/ref_logits crates/frink-models/tests/fixtures/minimax_m2_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath, QkNormStyle};
use frink_models::config::RopeLayout;
use frink_models::Decoder;
use frink_moe::GatingFunction;

const MINIMAX_M2: &str = "minimax_m2";

const MINIMAX_M2_GOLDEN: [f32; 48] = [
    -0.18008175,
    0.030930072,
    0.35588336,
    -0.46009502,
    0.35002363,
    0.37202615,
    0.3365625,
    0.66242194,
    -0.6652639,
    -0.66702265,
    0.12213123,
    -0.1327475,
    -0.19054496,
    -0.06448652,
    -0.6863829,
    -0.49462888,
    0.09449274,
    0.6324295,
    1.002193,
    0.24385493,
    0.25452095,
    0.45146582,
    -0.03238845,
    0.083857715,
    -0.26413268,
    -0.20129281,
    0.11958325,
    0.21058841,
    0.08861752,
    0.630594,
    -0.51071906,
    0.4815063,
    0.37045747,
    0.04172594,
    0.086865224,
    -0.6867334,
    -0.45762473,
    -0.023611449,
    -0.3898184,
    -0.41879585,
    0.27729136,
    0.18444277,
    -0.89964426,
    -0.36254674,
    0.095838726,
    -0.4006339,
    0.5631254,
    0.011058535,
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
fn minimax_m2_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(MINIMAX_M2, &MINIMAX_M2_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    let out = decode(&load_graph_fixture(MINIMAX_M2));
    println!(
        "{MINIMAX_M2}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
        kl_vs_golden(&out, &MINIMAX_M2_GOLDEN),
        worst_vs(&out, &MINIMAX_M2_GOLDEN)
    );
}

/// What the loader built: the whole-vector QK norm, the half-width
/// NEOX rotary, sigmoid routing with the router bias on every layer.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("minimax-m2"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    let d = load_graph_fixture(MINIMAX_M2);
    assert_eq!(d.config.qk_norm_style, QkNormStyle::WholeVector);
    assert_eq!(d.config.moe.gating, GatingFunction::Sigmoid);
    assert!(d.config.moe.norm_topk_prob);
    assert_eq!(d.config.rope_dim, Some(d.config.head_dim / 2));
    for layer in &d.layers {
        assert_eq!(
            layer.moe.router.rows(),
            d.config.moe.n_experts,
            "every layer routes"
        );
        assert!(
            layer.moe.exp_probs_bias.is_some(),
            "exp_probs_b on every layer"
        );
        let q = layer.attn.q_norm.as_ref().expect("attn_q_norm");
        assert_eq!(
            q.len(),
            d.config.n_heads * d.config.head_dim,
            "whole-vector"
        );
    }
}

/// The router bias and the whole-vector QK norm are each visible.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(MINIMAX_M2);
    assert_decoder_matches_on_all_three_paths(&d, &MINIMAX_M2_GOLDEN, GRAPH_TOL, "baseline");

    let saved: Vec<_> = d
        .layers
        .iter_mut()
        .map(|l| l.moe.exp_probs_bias.take())
        .collect();
    let worst = worst_vs(&decode(&d), &MINIMAX_M2_GOLDEN);
    assert!(worst > 1e-2, "exp_probs_b not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.moe.exp_probs_bias = s;
    }

    let saved: Vec<_> = d
        .layers
        .iter_mut()
        .map(|l| (l.attn.q_norm.take(), l.attn.k_norm.take()))
        .collect();
    let worst = worst_vs(&decode(&d), &MINIMAX_M2_GOLDEN);
    assert!(worst > 1e-2, "the whole-vector QK norm not seen: {worst}");
    for (l, (q, k)) in d.layers.iter_mut().zip(saved) {
        l.attn.q_norm = q;
        l.attn.k_norm = k;
    }
    assert_decoder_matches_on_all_three_paths(&d, &MINIMAX_M2_GOLDEN, GRAPH_TOL, "restored");
}
