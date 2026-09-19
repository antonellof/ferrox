//! openPangu-Embedded, checked against llama.cpp itself: a row that had
//! been filed as an EMBEDDING model from its name.
//!
//! `pangu-embedded` is openPangu-Embedded-1B / 7B (Huawei), a decoder
//! LLM (`PanguEmbeddedForCausalLM`; `conversion/pangu.py` is a
//! `TextModel` with an `lm_head`; "Embedded" means edge devices). frink
//! had it under `DeferredEncoderEmbedding` in the catalog and in
//! `embedding_model::NOT_YET`, so a real file was refused as an encoder
//! this crate does not have. `pangu-embed.cpp` is `llama.cpp`'s graph
//! with ONE difference: a REQUIRED `attn_output.bias` (`:37`, flag 0;
//! `proj_bias::ATTN_OUT_BIAS_CREATORS`). NEOX RoPE
//! (llama-model.cpp:2675), `n_rot == n_embd_head` (`:59`), fused or
//! split QKV (`:35`), SwiGLU (`:118-123`), `output` tied when absent
//! (`:22-27`).
//!
//! # Where the numbers come from
//!
//! The `GOLDEN` arrays were produced by running llama.cpp's own graph
//! over each fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| frink) | max abs logit delta |
//! |---|---|---|
//! | `pangu_embedded` | see `report_kl_against_llama_cpp` | |
//! | `pangu_embedded_fused` | (fused `attn_qkv`; libllama byte-identical to the split file) | |
//! | `pangu_embedded_output` | (a separate `output.weight`) | |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_pangu_fixture.py \
//!     crates/frink-models/tests/fixtures/pangu_embedded_tiny.gguf [--fused-qkv | --output]
//! /tmp/ref_logits crates/frink-models/tests/fixtures/pangu_embedded_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_models::capability::{resolve_architecture, ArchPath};
use frink_models::config::RopeLayout;
use frink_models::Decoder;

const PANGU: &str = "pangu_embedded";
const PANGU_FUSED: &str = "pangu_embedded_fused";
const PANGU_OUTPUT: &str = "pangu_embedded_output";

const PANGU_GOLDEN: [f32; 48] = [
    0.36418787,
    -2.9886932,
    -0.6848891,
    -1.205361,
    0.07544863,
    -2.7622547,
    1.3927892,
    -2.9490218,
    -1.3029487,
    -0.5356415,
    -0.5718187,
    -0.34816056,
    -1.6766404,
    -3.0085373,
    1.0241663,
    0.37920883,
    1.331836,
    -1.5700009,
    -0.17715663,
    -1.3311366,
    1.3281704,
    -1.4164956,
    -1.2125974,
    0.32548308,
    -1.0929209,
    -2.0692935,
    -0.9779026,
    -0.10343247,
    1.6959383,
    -2.1802588,
    2.8027813,
    -0.30698818,
    -1.6554751,
    -2.9196153,
    -1.0220584,
    -0.5137609,
    0.9052211,
    -0.1777057,
    1.8490103,
    -0.026606921,
    2.5837765,
    2.440811,
    -0.8441128,
    -0.30698663,
    -0.70197374,
    2.077433,
    -2.4971662,
    -0.09410411,
];

const PANGU_OUTPUT_GOLDEN: [f32; 48] = [
    1.1323619,
    -0.9568927,
    0.89949393,
    1.029346,
    0.4979661,
    1.3224282,
    -1.5152202,
    -1.5602864,
    1.5065863,
    0.6214521,
    0.6626645,
    3.7042758,
    -0.58301985,
    -0.23464614,
    -0.67621064,
    0.34227937,
    1.5349947,
    -1.8877958,
    -1.036495,
    -1.6657357,
    -0.45708665,
    2.2566447,
    0.6440324,
    -1.0101992,
    -0.9288244,
    0.1181117,
    2.303063,
    -0.15895462,
    -1.3371644,
    0.12627214,
    -1.1192995,
    -1.9105798,
    -1.1504521,
    -0.443059,
    -1.6249064,
    -1.8980792,
    0.20056704,
    -1.190697,
    0.6522395,
    0.8981875,
    -0.5747896,
    0.40487498,
    -0.8579127,
    -0.28360268,
    0.5954941,
    0.2213046,
    1.1331675,
    -0.23485681,
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
fn pangu_embedded_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(PANGU, &PANGU_GOLDEN);
}

#[test]
fn the_fused_qkv_spelling_matches_the_same_golden() {
    assert_all_three_paths_match(PANGU_FUSED, &PANGU_GOLDEN);
}

#[test]
fn a_separate_output_weight_matches_llama_cpp() {
    assert_all_three_paths_match(PANGU_OUTPUT, &PANGU_OUTPUT_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (PANGU, &PANGU_GOLDEN),
        (PANGU_FUSED, &PANGU_GOLDEN),
        (PANGU_OUTPUT, &PANGU_OUTPUT_GOLDEN),
    ] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || frink) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// A decoder, not an embedding model: on the generic NEOX path, with the
/// required bias loaded on every layer.
#[test]
fn the_loaded_decoder_is_the_graph() {
    assert!(matches!(
        resolve_architecture("pangu-embedded"),
        Some(ArchPath::GenericGqa {
            rope: RopeLayout::Neox
        })
    ));
    assert!(!frink_models::embedding_model::is_embedding_arch(
        "pangu-embedded"
    ));
    let d = load_graph_fixture(PANGU);
    for (il, layer) in d.layers.iter().enumerate() {
        let b = layer.attn.o_bias.as_ref().expect("attn_output.bias");
        assert_eq!(b.len(), d.config.hidden_dim, "blk.{il}");
    }
}

/// The bias is the one thing the graph adds to `llama.cpp`'s, and it is
/// visible: dropped, the logits leave the golden.
#[test]
fn the_output_bias_is_visible_in_the_logits() {
    let mut d = load_graph_fixture(PANGU);
    assert_decoder_matches_on_all_three_paths(&d, &PANGU_GOLDEN, GRAPH_TOL, "baseline");
    let saved: Vec<_> = d.layers.iter_mut().map(|l| l.attn.o_bias.take()).collect();
    let worst = worst_vs(&decode(&d), &PANGU_GOLDEN);
    assert!(worst > 1e-2, "attn_output.bias not seen: {worst}");
    for (l, s) in d.layers.iter_mut().zip(saved) {
        l.attn.o_bias = s;
    }
    assert_decoder_matches_on_all_three_paths(&d, &PANGU_GOLDEN, GRAPH_TOL, "restored");
}
