//! PLM, checked against llama.cpp itself: DeepSeek-2 MLA attention on a
//! dense ReLU-squared model, on the MLA engine -- which makes this the
//! MLA engine's FIRST libllama-golden fixture.
//!
//! `plm` was triaged NEW CODE on the attention (`src/models/plm.cpp:
//! 84-166`: a direct `attn_q` split per head into nope / pe, a
//! compressed KV RMS-normed and re-expanded through `attn_kv_b`, ONE
//! shared roped key repeated onto every head, `kq_scale =
//! 1/sqrt(n_embd_head_k)`), which ferrox had only inside `MlaEngine`,
//! arch-gated to `deepseek2` / `mistral4` and REQUIRING a low-rank Q.
//! The three places PLM differs from DeepSeek-2 are one table now
//! (`ferrox_models::mla_arch`): a direct `attn_q` (`plm.cpp:32`;
//! `ferrox_models::mla_q_proj`, which the lite DeepSeek-V2 checkpoints
//! take too), an ungated `LLM_FFN_RELU_SQR` dense FFN (`:181-187`,
//! `GluAct::ReluSqr` with the gate aliased as for `arcee`), and a tied
//! lm_head the graph never reads an `output.weight` for (`:23-24`).
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `plm` | the converter's shape (`conversion/plm.py:14-19`): `key_length` = nope + rope, `value_length`, `rope.dimension_count` = rope, `kv_lora_rank` |
//! | `plm_decoy_output` | the same file with an `output.weight` the graph does not read; libllama REFUSES it (`done_getting_tensors: wrong number of tensors; expected 30, got 29`, measured), so ferrox must too rather than prefer the decoy |
//!
//! # Where the numbers come from
//!
//! `GOLDEN` was produced by running llama.cpp's own `plm` graph over
//! the fixture through `scripts/gptoss_reference_logits.cpp` linked
//! against a real `libllama` built from `.scratch/llama.cpp`.
//!
//! | fixture | KL(llama.cpp \|\| ferrox) | max abs logit delta |
//! |---|---|---|
//! | `plm` | 1.87e-13 | 1.67e-06 |
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_plm_fixture.py \
//!     crates/ferrox-models/tests/fixtures/plm_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_plm_fixture.py \
//!     crates/ferrox-models/tests/fixtures/plm_decoy_output_tiny.gguf --decoy-output
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/plm_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{assert_close, graph_fixture_path, kl_vs_golden, worst_vs, GRAPH_PROMPT, GRAPH_TOL};
use ferrox_models::engine::Engine;
use ferrox_models::mla::MlaQProj;
use ferrox_models::mla_arch::{mla_arch, MlaOutputHead, QProjRule};
use ferrox_models::{load_mla_engine_from_path, LoadError, MlaEngine, ServedEngine};
use ferrox_moe::GluAct;

const FIXTURE: &str = "plm";
const DECOY: &str = "plm_decoy_output";

const PLM_GOLDEN: [f32; 48] = [
    -1.2535992,
    -2.542297,
    -4.15273,
    -1.5252366,
    -0.35572457,
    2.627922,
    -5.146885,
    -1.498502,
    3.3820598,
    0.041666746,
    -1.6646056,
    -3.053184,
    -1.5522586,
    2.4025483,
    2.2013927,
    -0.3343612,
    0.45524335,
    1.0816802,
    0.30913138,
    -0.910275,
    -4.093313,
    1.0066636,
    0.073821425,
    0.77932185,
    -1.8273256,
    0.1780591,
    -4.2316604,
    2.1787424,
    1.4572556,
    2.224849,
    0.9564159,
    -4.1414394,
    0.025222957,
    -1.1887589,
    -1.1254009,
    3.938088,
    -0.18220347,
    -1.1765492,
    -3.400023,
    0.691692,
    -2.0464454,
    0.37650633,
    1.2984468,
    -0.8227435,
    4.003407,
    6.391534,
    1.3387175,
    -1.8556216,
];

fn load(name: &str) -> Result<MlaEngine, LoadError> {
    match load_mla_engine_from_path(std::path::Path::new(&graph_fixture_path(name)))? {
        ServedEngine::Mla(m) => Ok(m),
        _ => panic!("`{name}` is served by the MLA engine"),
    }
}

/// The MLA engine has one body, token by token; the last position's
/// logits over the six-token prompt.
fn decode(engine: &MlaEngine) -> Vec<f32> {
    let mut state = Engine::new_state(engine);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = engine.forward_token(tok, pos, &mut state);
    }
    out
}

#[test]
fn plm_matches_llama_cpp() {
    let engine = load(FIXTURE).expect("plm loads on the MLA engine");
    let got = decode(&engine);
    println!(
        "plm: KL(llama.cpp || ferrox) = {:.3e}, max |delta| = {:.3e}",
        kl_vs_golden(&got, &PLM_GOLDEN),
        worst_vs(&got, &PLM_GOLDEN)
    );
    assert_close(&got, &PLM_GOLDEN, GRAPH_TOL, "plm: MLA engine decode");
}

/// The three table decisions the fixture exercises, read back off the
/// loaded engine rather than off the table: a direct Q, an ungated
/// ReLU-squared dense FFN with the gate aliased, a tied lm_head.
#[test]
fn the_loaded_engine_is_the_plm_row() {
    let row = mla_arch("plm").expect("plm is an MLA-engine row");
    assert_eq!(row.q_proj, QProjRule::Direct);
    assert_eq!(row.output_head, MlaOutputHead::TiedOnly);
    assert!(matches!(row.dense_act, GluAct::ReluSqr));

    let engine = load(FIXTURE).unwrap();
    assert_eq!(engine.layers.len(), 3);
    assert_eq!(engine.mla_cfg.q_lora_rank, 0);
    assert_eq!(
        engine.mla_cfg.qk_nope_head_dim, 8,
        "key_length - rope.dimension_count"
    );
    assert_eq!(engine.mla_cfg.qk_rope_head_dim, 4, "rope.dimension_count");
    assert_eq!(engine.mla_cfg.v_head_dim, 8, "value_length");
    assert_eq!(engine.mla_cfg.kv_lora_rank, 12);
    for layer in &engine.layers {
        assert!(matches!(layer.attn.q, MlaQProj::Direct(_)));
        assert_eq!(layer.attn.q.q_rows(), 4 * 12);
        match &layer.ffn {
            ferrox_models::MlaLayerFfn::Dense(d) => {
                assert!(matches!(d.act, GluAct::ReluSqr));
                assert_eq!(d.weights.up.rows(), 48);
            }
            ferrox_models::MlaLayerFfn::Moe(_) => panic!("plm is dense"),
        }
    }
    // Tied: the lm_head IS the embedding, row for row.
    assert_eq!(engine.output_head.rows(), engine.embedding.rows());
    assert_eq!(
        engine.output_head.dequant_row(7),
        engine.embedding.dequant_row(7)
    );
}

/// libllama refuses the file (measured); a loader that silently
/// preferred the decoy would produce different logits from a file
/// llama.cpp cannot run at all.
#[test]
fn an_output_weight_on_plm_is_refused_as_llama_cpp_refuses_it() {
    let err = load(DECOY).err().expect("the decoy file is refused");
    let msg = err.to_string();
    assert!(
        msg.contains("output.weight") && msg.contains("llama-model-loader.cpp:1309-1313"),
        "the refusal names the tensor and the upstream check: {msg}"
    );
}

/// Each thing the golden checks, sabotaged one at a time, moves the
/// logits by far more than the tolerance: the ReLU-squared activation,
/// the compressed-KV norm, the RoPE base and the rotation itself. The
/// per-head nope/pe ORDER and the softmax scale have no knob to turn
/// without resizing the weights; they are what the golden match
/// itself evidences. A test that cannot see one of them would be
/// reporting agreement on a graph it does not check.
#[test]
fn each_of_the_rows_decisions_is_visible_in_the_logits() {
    let mut engine = load(FIXTURE).unwrap();
    let golden = decode(&engine);
    assert_close(&golden, &PLM_GOLDEN, GRAPH_TOL, "baseline");

    // The ungated activation swapped for SwiGLU over the aliased gate.
    for layer in engine.layers.iter_mut() {
        if let ferrox_models::MlaLayerFfn::Dense(d) = &mut layer.ffn {
            d.act = GluAct::Swiglu;
        }
    }
    assert!(
        worst_vs(&decode(&engine), &PLM_GOLDEN) > 1e-2,
        "activation not seen"
    );
    for layer in engine.layers.iter_mut() {
        if let ferrox_models::MlaLayerFfn::Dense(d) = &mut layer.ffn {
            d.act = GluAct::ReluSqr;
        }
    }

    // The compressed-KV norm weight flattened to one.
    let saved: Vec<Vec<f32>> = engine
        .layers
        .iter_mut()
        .map(|l| std::mem::replace(&mut l.attn.kv_a_layernorm, vec![1.0; 12]))
        .collect();
    assert!(
        worst_vs(&decode(&engine), &PLM_GOLDEN) > 1e-2,
        "kv_a_norm not seen"
    );
    for (l, w) in engine.layers.iter_mut().zip(saved) {
        l.attn.kv_a_layernorm = w;
    }

    // The file's `rope.freq_base` replaced by another.
    let rope = engine.mla_cfg.rope.expect("plm rotates its pe slices");
    engine.mla_cfg.rope = Some(ferrox_models::config::MlaRopeConfig { theta: 500.0 });
    assert!(
        worst_vs(&decode(&engine), &PLM_GOLDEN) > 1e-2,
        "rope theta not seen"
    );
    engine.mla_cfg.rope = Some(rope);

    // RoPE off entirely.
    let theta = engine.mla_cfg.rope.take();
    assert!(
        worst_vs(&decode(&engine), &PLM_GOLDEN) > 1e-2,
        "rope not seen"
    );
    engine.mla_cfg.rope = theta;

    assert_close(&decode(&engine), &PLM_GOLDEN, GRAPH_TOL, "restored");
}
