//! Orion and Nemotron, checked against llama.cpp itself: the LayerNorm
//! WITH A BIAS at every norm site, the variant `capability::
//! WEIGHTED_LAYER_NORM` had named as having no caller.
//!
//! Eight architectures create `attn_norm.bias` / `ffn_norm.bias` /
//! `output_norm.bias` as REQUIRED and `build_norm(x, w, b, LLM_NORM,
//! il)` multiplies by `w` and then adds `b`. `tests/attn_bias.rs` had
//! them all refused with the bias named. Read one by one, two of them
//! need NOTHING ELSE of the generic decoder, and those two close here
//! on `NormOp::LayerNormBias` (`capability::BIASED_LAYER_NORM`):
//!
//! - `orion` (Orion-14B): a Llama with NEOX RoPE, no `rope.dimension_count`
//!   and no `rope.freq_base` in the file (`conversion/orion.py:13-37`).
//! - `nemotron` (Nemotron-4, Minitron): the ungated ReLU-squared FFN
//!   `arcee` already serves, partial NEOX RoPE, and three OPTIONAL
//!   biases (`nemotron.cpp:31,40-41`) the generic dense path has no slot
//!   for -- a file carrying them is refused as unread.
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `orion` | the converter's shape: six per-layer norm tensors and two output-norm tensors, NEOX at the defaults |
//! | `nemotron` | the same norm on the ReLU-squared graph with a half-width rotary and `rope.scaling.type = none` |
//! | `nemotron_biases` | the optional `attn_output.bias` / `ffn_up.bias` / `ffn_down.bias` present; libllama applies them (its logits move by 8.07), ferrox refuses the file rather than drop them |
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own graph
//! over the fixture through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| ferrox) | max abs logit delta |
//! |---|---|---|
//! | `orion` | 2.33e-11 | 1.64e-05 (see `ORION_TOL`) |
//! | `nemotron` | 5.13e-13 | 4.86e-06 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_orion_fixture.py \
//!     crates/ferrox-models/tests/fixtures/orion_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_nemotron_fixture.py \
//!     crates/ferrox-models/tests/fixtures/nemotron_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_nemotron_fixture.py \
//!     crates/ferrox-models/tests/fixtures/nemotron_biases_tiny.gguf --biases
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, assert_all_three_paths_match_within,
    assert_decoder_matches_on_all_three_paths, graph_caches, graph_fixture_path, kl_vs_golden,
    load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use ferrox_models::capability::{resolve_architecture, ArchPath, BIASED_LAYER_NORM};
use ferrox_models::config::RopeLayout;
use ferrox_models::norm::NormOp;
use ferrox_models::{Decoder, FfnActivation, ModelConfig};

const ORION: &str = "orion";
const NEMOTRON: &str = "nemotron";
const NEMOTRON_BIASES: &str = "nemotron_biases";

/// Orion sits at 1.6e-5 max |delta| (KL 2.3e-11), above the 1e-5 line
/// the suite holds every other fixture to and two orders under the next
/// tolerance the suite has. It is noise, and it was measured rather
/// than assumed: drawing every projection at unit scale still leaves
/// 1.2e-5 (KL 6e-12), where `nemotron` -- the same norm on a ReLU-squared
/// FFN -- is at 4.9e-6 and `dbrx` / `olmo`, the other two LayerNorm
/// rows, are at 1e-12 or better. What is different is a SwiGLU fed a
/// biased (non-zero-mean) LayerNorm output on a libllama built with
/// Accelerate (`vDSP_measqv`, `sgemm`), which is where f32 summation
/// order shows. The line here is 5e-5, and the sabotages below move the
/// logits by three orders more than it.
const ORION_TOL: f32 = 5e-5;

const ORION_GOLDEN: [f32; 48] = [
    -1.0600787,
    2.4058733,
    1.182935,
    2.4633207,
    1.2401185,
    -4.8361483,
    0.1367904,
    -1.257247,
    -2.4184408,
    0.3085447,
    0.39239192,
    3.3186865,
    2.6637363,
    0.6739763,
    -4.361998,
    2.0904145,
    1.5585349,
    -1.7527164,
    0.0007061064,
    -1.1774886,
    -1.5299392,
    -2.4000235,
    -0.23690084,
    -0.9865638,
    -6.2291765,
    2.3851466,
    -4.575435,
    -2.159268,
    -0.32663774,
    1.8916454,
    2.598425,
    -0.46904385,
    -0.6126561,
    -3.099367,
    -0.49888074,
    3.3013453,
    0.4253,
    -0.14596522,
    -0.2624246,
    -2.6646674,
    -1.899404,
    1.4912336,
    2.2139342,
    1.3733875,
    0.86803055,
    0.92236865,
    -2.8611984,
    0.6219342,
];

const NEMOTRON_GOLDEN: [f32; 48] = [
    0.021550179,
    1.8955879,
    -1.3320813,
    6.056007,
    3.3412948,
    0.9120406,
    -1.1567364,
    1.102097,
    -0.46142924,
    0.8820058,
    1.3761998,
    0.7271421,
    2.0539563,
    -0.36991945,
    0.05811578,
    0.059939086,
    -2.956265,
    -0.9914286,
    0.92990816,
    0.5021645,
    0.31386787,
    0.8486869,
    0.7654047,
    -1.5778378,
    -3.191719,
    0.84653544,
    2.0144532,
    2.4971068,
    -1.4107854,
    3.3293858,
    1.2479157,
    -3.9631557,
    1.7998682,
    2.6418035,
    -6.1240745,
    -1.378068,
    0.81102777,
    -0.9004905,
    0.13881844,
    -1.7073916,
    -0.56480485,
    -0.1232782,
    -0.6907536,
    2.6854124,
    1.5081784,
    0.19908667,
    1.4768641,
    -4.941042,
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
fn orion_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match_within(ORION, &ORION_GOLDEN, ORION_TOL);
}

#[test]
fn nemotron_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(NEMOTRON, &NEMOTRON_GOLDEN);
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [(ORION, &ORION_GOLDEN), (NEMOTRON, &NEMOTRON_GOLDEN)] {
        let out = decode(&load_graph_fixture(name));
        println!(
            "{name}: KL(llama.cpp || ferrox) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
}

/// What the loader built: every norm site is the biased LayerNorm,
/// with the bias read from the file; the two rows are NEOX; orion's
/// rotary defaults and nemotron's half-width rotary and ReLU-squared
/// FFN are what the graphs read.
#[test]
fn the_loaded_layers_are_the_two_graphs() {
    assert_eq!(BIASED_LAYER_NORM, &["orion", "nemotron"]);
    for name in [ORION, NEMOTRON] {
        assert!(matches!(
            resolve_architecture(name),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        ));
        let d = load_graph_fixture(name);
        for layer in &d.layers {
            for (site, op) in [
                ("attn", &layer.attn.norm_weight),
                ("ffn", &layer.moe.norm_weight),
            ] {
                match op {
                    NormOp::LayerNormBias { weight, bias } => {
                        assert_eq!(weight.len(), 32, "{name} {site}");
                        assert_eq!(bias.len(), 32, "{name} {site}");
                        assert!(
                            bias.iter().any(|b| b.abs() > 0.1),
                            "{name} {site}: the bias is the file's"
                        );
                    }
                    other => panic!("{name} {site}: {other:?} is not the biased LayerNorm"),
                }
            }
        }
        assert!(matches!(d.final_norm, NormOp::LayerNormBias { .. }));
    }
    let orion = load_graph_fixture(ORION);
    assert_eq!(
        orion.config.rope_dim, None,
        "no rope.dimension_count: the whole head rotates"
    );
    assert_eq!(
        orion.config.rope_theta, 10000.0,
        "no rope.freq_base: llama.cpp's default"
    );
    assert_eq!(
        orion.config.rms_norm_eps, 1e-5,
        "attention.layer_norm_epsilon"
    );
    let nemotron = load_graph_fixture(NEMOTRON);
    assert_eq!(nemotron.config.rope_dim, Some(4));
    assert_eq!(nemotron.config.ffn_activation, FfnActivation::ReluSqr);
}

/// Nemotron's optional projection biases: llama.cpp applies them (the
/// two goldens differ by 8.07), the generic dense path has no slot for
/// them, and the file is refused rather than run unbiased. The plain
/// file, which differs only by those three tensors per layer, loads.
#[test]
fn nemotrons_optional_projection_biases_are_refused_not_dropped() {
    let file =
        ferrox_gguf::GgufFile::open(graph_fixture_path(NEMOTRON_BIASES)).expect("fixture opens");
    let config = ModelConfig::from_gguf(&file).expect("the config reads");
    let err = Decoder::from_gguf(graph_fixture_path(NEMOTRON_BIASES), config)
        .err()
        .expect("a file with biases the generic path cannot apply is refused");
    let msg = err.to_string();
    assert!(
        msg.contains("ffn_up.bias")
            || msg.contains("attn_output.bias")
            || msg.contains("ffn_down.bias"),
        "the refusal names a bias: {msg}"
    );
}

/// Each half of the norm, sabotaged on the loaded decoder: the bias
/// zeroed and the weight flattened each move the logits past the
/// tolerance, on both rows.
#[test]
fn the_bias_and_the_weight_are_each_visible_in_the_logits() {
    for (name, golden, tol) in [
        (ORION, &ORION_GOLDEN, ORION_TOL),
        (NEMOTRON, &NEMOTRON_GOLDEN, GRAPH_TOL),
    ] {
        let mut d = load_graph_fixture(name);
        assert_decoder_matches_on_all_three_paths(&d, golden, tol, name);

        let weight_of = |op: &NormOp| -> Vec<f32> {
            let NormOp::LayerNormBias { weight, .. } = op else {
                unreachable!()
            };
            weight.clone()
        };

        let saved: Vec<NormOp> = d
            .layers
            .iter_mut()
            .map(|l| {
                let weight = weight_of(&l.attn.norm_weight);
                std::mem::replace(
                    &mut l.attn.norm_weight,
                    NormOp::LayerNormBias {
                        weight,
                        bias: vec![0.0; 32],
                    },
                )
            })
            .collect();
        let worst = worst_vs(&decode(&d), golden);
        assert!(worst > 1e-2, "{name}: attn_norm.bias not seen: {worst}");
        for (l, s) in d.layers.iter_mut().zip(saved) {
            l.attn.norm_weight = s;
        }

        // The weight without the bias: `NormOp::LayerNorm`, dbrx's form,
        // is the plausible wrong variant.
        let saved: Vec<NormOp> = d
            .layers
            .iter_mut()
            .map(|l| {
                let weight = weight_of(&l.moe.norm_weight);
                std::mem::replace(&mut l.moe.norm_weight, NormOp::LayerNorm(weight))
            })
            .collect();
        let worst = worst_vs(&decode(&d), golden);
        assert!(worst > 1e-2, "{name}: ffn_norm.bias not seen: {worst}");
        for (l, s) in d.layers.iter_mut().zip(saved) {
            l.moe.norm_weight = s;
        }

        // And RMSNorm with the weight -- what the generic path used to
        // compute for these files -- on the final norm.
        let weight = weight_of(&d.final_norm);
        let saved = std::mem::replace(&mut d.final_norm, NormOp::Rms(weight));
        let worst = worst_vs(&decode(&d), golden);
        assert!(
            worst > 1e-2,
            "{name}: output_norm as RMSNorm not seen: {worst}"
        );
        d.final_norm = saved;

        assert_decoder_matches_on_all_three_paths(&d, golden, tol, "restored");
    }
}
