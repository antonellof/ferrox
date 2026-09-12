//! Arcee (AFM), checked against llama.cpp itself: the UNGATED
//! ReLU-squared FFN.
//!
//! `arcee` was triaged NEW CODE on one fact, `arcee.cpp:39-40` creating
//! `ffn_up` and `ffn_down` and no `ffn_gate`, with `:123-128` calling
//! `build_ffn` with a NULL gate, `LLM_FFN_RELU_SQR` and `LLM_FFN_SEQ`:
//! `down(relu(up(x))^2)`. The verdict shared that constant with `plm`
//! and predicted the pair would close together; reading `plm.cpp`
//! against `arcee.cpp` before assuming so found MLA attention in it, so
//! only this row closed and `plm`'s verdict now says which half is done
//! (`capability::UNGATED_RELU_SQR`).
//!
//! **How the FFN is spelled.** `ExpertWeights` has three required
//! matrices and thirty construction sites; the ungated FFN is not given
//! a fourth shape. `FfnActivation::ReluSqr` maps to
//! `ferrox_moe::GluAct::ReluSqr` (`relu(up)^2`, reading `up` alone) and
//! the loader ALIASES the expert's gate to its up matrix, so every
//! gated path -- routed, placed, batched, slotted -- serves it with no
//! branch, and the dense hot paths (`run_expert`, the batched dense
//! FFN) skip the aliased matmul through `GluAct::ungated`. (It mapped
//! to `GluAct::Reglu`, `relu(gate) * up`, until SmallThinker's REAL
//! gate showed that one variant cannot mean both.) The suite
//! below pins both halves: the aliasing is visible on the loaded
//! weights, and un-aliasing the gate (a real gate, or SwiGLU on the
//! pair) diverges from the golden.
//!
//! **Where the numbers come from.** `ARCEE_GOLDEN` was produced by
//! running llama.cpp's own graph over the fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`.
//!
//! Regenerating:
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_arcee_fixture.py \
//!     crates/ferrox-models/tests/fixtures/arcee_tiny.gguf
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/arcee_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, kl_vs_golden,
    load_graph_fixture, worst_vs, GRAPH_PROMPT,
};
use ferrox_models::{FfnActivation, ModelConfig, RopeLayout};

const ARCEE: &str = "arcee";

const ARCEE_GOLDEN: [f32; 48] = [
    -1.0826676,
    0.53167236,
    -0.62626576,
    0.89818555,
    1.0244392,
    1.7364765,
    -0.72995603,
    -0.30296987,
    -0.067103535,
    0.011495039,
    -0.7972398,
    0.6270751,
    1.378322,
    0.6192732,
    1.203366,
    -0.33688697,
    -0.1442725,
    0.26987925,
    -0.23601888,
    0.743614,
    -0.54334843,
    -0.35515717,
    1.882766,
    0.070676856,
    1.6999218,
    1.0760334,
    0.9305699,
    -2.0933466,
    1.4192882,
    -1.6191919,
    0.15880257,
    1.354697,
    1.4674244,
    -0.78983176,
    -0.4017068,
    0.16163737,
    -3.0151858,
    0.5517989,
    -1.2759664,
    0.5420454,
    0.36993393,
    0.0010383157,
    -2.9409926,
    -0.13414197,
    -0.19217438,
    1.3035148,
    -1.5270181,
    0.32940996,
];

/// The row itself, on all three forward paths.
#[test]
fn arcee_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(ARCEE, &ARCEE_GOLDEN);
}

/// The number in the report, so it can be regenerated rather than
/// trusted. Run with `--nocapture` to see it.
#[test]
fn report_kl_against_llama_cpp() {
    let d = load_graph_fixture(ARCEE);
    let mut kv = graph_caches(&d);
    let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    let kl = kl_vs_golden(&got, &ARCEE_GOLDEN);
    let worst = worst_vs(&got, &ARCEE_GOLDEN);
    println!("| `arcee` | {kl:.2e} | {worst:.2e} |");
    assert!(kl < 1e-8, "KL {kl}");
}

/// The loaded shape: the activation is the ungated one, and on every
/// layer the gate IS the up matrix, row for row.
///
/// Structural rather than numeric, so that a loader that quietly went
/// back to demanding `ffn_gate` -- or aliased the wrong tensor -- says
/// which layer and why before the numeric test says "diverged".
#[test]
fn every_layer_s_gate_is_an_alias_of_its_up_matrix() {
    let d = load_graph_fixture(ARCEE);
    assert_eq!(d.config.ffn_activation, FfnActivation::ReluSqr);
    assert_eq!(d.layers.len(), 2);
    for (il, layer) in d.layers.iter().enumerate() {
        layer.moe.with_expert(0, |ex| {
            assert_eq!(ex.gate.rows(), ex.up.rows(), "blk.{il}");
            assert_eq!(ex.gate.cols(), ex.up.cols(), "blk.{il}");
            for r in 0..ex.up.rows() {
                assert_eq!(
                    ex.gate.dequant_row(r),
                    ex.up.dequant_row(r),
                    "blk.{il}: gate row {r} is not the up row; the alias is the whole \
                     implementation of the ungated FFN"
                );
            }
            assert_eq!(ex.up.rows(), 40, "blk.{il}: n_ff");
        });
    }
}

/// SwiGLU on the aliased pair -- what the loader would compute if the
/// activation were dropped and the alias kept -- diverges.
///
/// The sabotage that matters: "arcee has an `ffn_up` and an `ffn_down`
/// like everyone else, so load it and run the default". With the gate
/// aliased that runs `silu(up) * up`, which is fluent and wrong.
#[test]
fn running_swiglu_on_the_aliased_pair_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture(ARCEE);
    d.config.ffn_activation = FfnActivation::Swiglu;
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &ARCEE_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "SwiGLU on the aliased pair moved the output by only {worst}; the FFN \
         pre-activations must be too small for the two nonlinearities to differ"
    );
}

/// A file WITH an `ffn_gate` is refused by name, as llama.cpp refuses it.
///
/// `arcee.cpp` never asks for the tensor, so libllama fails on
/// `arcee_gated_tiny.gguf` with `done_getting_tensors: wrong number of
/// tensors; expected 21, got 19` (measured). ferrox's refusal is
/// reachable from the same file, and names the graph fact rather than
/// leaving a generic unread-tensor message.
#[test]
fn a_file_with_a_gate_is_refused_like_llama_cpp_refuses_it() {
    let path = graph_fixture_path("arcee_gated");
    let file = ferrox_gguf::GgufFile::open(&path).expect("fixture opens");
    let config = ModelConfig::from_gguf(&file).expect("the header is the audited one");
    let msg = match ferrox_models::Decoder::from_gguf(&path, config) {
        Ok(_) => panic!("a file with a gate must be refused"),
        Err(err) => format!("{err}"),
    };
    assert!(msg.contains("ffn_gate"), "{msg}");
    assert!(msg.contains("ungated"), "{msg}");
    assert!(msg.contains("arcee.cpp:123-128"), "{msg}");
}

/// The rest of the graph, pinned against the C and with the fixture
/// able to see the other answer where there is one.
#[test]
fn arcee_is_a_norm_rope_llama_with_an_untied_head_and_no_scalars() {
    let d = load_graph_fixture(ARCEE);
    assert_eq!(
        d.config.rope_layout,
        RopeLayout::Norm,
        "llama-model.cpp:2600"
    );
    assert_eq!(
        d.config.attention_scale, None,
        "arcee.cpp:64 with f_attention_scale unset"
    );
    assert_eq!(d.config.n_heads, 4);
    assert_eq!(d.config.n_kv_heads, 2, "the fixture exercises GQA");
    assert_eq!(
        d.config.head_dim, 6,
        "arcee.cpp:51-52: n_embd_head == n_rot"
    );
    assert!(d.config.layer_shapes.is_uniform());
    assert!(d.layers[0].attn.post_attn_norm.is_none());
    assert!(d.layers[0].attn.post_ffn_norm.is_none());
    assert_eq!(d.config.embedding_scale, None);
    assert_eq!(d.config.residual_scale, None);
    assert_eq!(d.config.logit_multiplier, None);

    let mut d = load_graph_fixture(ARCEE);
    d.config.rope_layout = RopeLayout::Neox;
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &ARCEE_GOLDEN,
    );
    assert!(
        worst > 1e-3,
        "rotating the NEOX pairs moved the output by only {worst}; the attention in this \
         fixture is too flat to see a positional bug"
    );
}
