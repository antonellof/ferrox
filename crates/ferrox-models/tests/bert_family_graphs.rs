//! The BERT-family encoders against llama.cpp's own pooled embedding,
//! on synthetic fixtures: `nomic-bert` and `jina-bert-v3`.
//!
//! # What these rows are
//!
//! `src/models/bert.cpp`'s graph serves several architectures, and the
//! ones ferrox builds differ from `bert` in two lines of it and
//! nothing else. `bert_gguf_loader::ENCODER_ARCHS` is the table:
//!
//! | arch | rotation | FFN |
//! |---|---|---|
//! | `bert` | learned position table | ungated GELU, both biases |
//! | `nomic-bert` | NEOX RoPE on Q/K | gated SiLU, no biases |
//! | `jina-bert-v3` | NEOX RoPE on Q/K | ungated GELU, both biases |
//!
//! `nomic-bert` differs from `bert` in exactly two lines:
//!
//!   * **RoPE on Q and K** (`:126-133`, NEOX by
//!     `llama_model_rope_type`), where `bert` adds a learned position
//!     table to the embeddings instead (`:90`, gated on
//!     `arch == LLM_ARCH_BERT`). A rotating file carries NO
//!     `position_embd`: measured, libllama's load log never names it
//!     for this architecture, and `bert_gguf_loader` refuses a
//!     rotating file that carries one rather than ignoring it.
//!   * **A gated SiLU FFN** (`:195-201`, the final `else`): `ffn_up`
//!     and `ffn_gate` with no biases, where `bert`'s is an ungated
//!     GELU with both.
//!
//! `bert_encoder::BertFfn` and `BertHparams::rope_theta` are those two
//! facts, read from the architecture at load through
//! `bert_gguf_loader::ENCODER_ARCHS` -- not from tensor presence,
//! because a `bert` file that happens to carry a gate would then run
//! gated here and ungated in llama.cpp.
//!
//! # Why a synthetic fixture and a committed golden
//!
//! `tests/bert_llama_cpp_parity.rs` is the same comparison against a
//! real 36 MB checkpoint and is `#[ignore]`d for that reason. This one
//! runs everywhere: the fixture is 200 KB of F32 in the repo and the
//! golden is what `tools/llama_logits.c --embed` printed for it,
//! through llama.cpp's own MEAN pooling.
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_nomic_bert_fixture.py \\
//!     crates/ferrox-models/tests/fixtures/nomic_bert_tiny.gguf
//! ./target/llama_logits --embed \\
//!     crates/ferrox-models/tests/fixtures/nomic_bert_tiny.gguf "hello world"
//! ```

use ferrox_models::EmbeddingModel;

/// llama.cpp's MEAN-pooled embedding for `"hello world"` on
/// `nomic_bert_tiny.gguf`, un-normalized.
const NOMIC_GOLDEN: [f32; 32] = [
    0.32149905,
    0.094289646,
    0.15053454,
    -2.6003315,
    -0.019941427,
    0.049918007,
    -0.075354025,
    1.1375569,
    -0.49217373,
    2.3775523,
    -1.7570381,
    -1.7811859,
    -0.22807966,
    0.95391023,
    0.9663884,
    0.96289575,
    -1.3318931,
    -0.00059386156,
    -0.2390738,
    -0.17903815,
    -0.022602148,
    0.053266555,
    1.3737991,
    0.747592,
    0.09730208,
    -1.1276442,
    -0.6946099,
    0.31933406,
    0.4832091,
    0.8345471,
    2.83744,
    -0.011955578,
];

/// llama.cpp's MEAN-pooled embedding for `"hello world"` on
/// `jina_bert_v3_tiny.gguf`, un-normalized.
const JINA_V3_GOLDEN: [f32; 32] = [
    -0.67560554,
    0.19488943,
    0.47437486,
    -2.4178503,
    -0.30169535,
    -0.31104457,
    0.12225819,
    -1.6278875,
    -0.6008126,
    0.21150663,
    2.5187652,
    -1.4892101,
    -0.47848803,
    -0.28579736,
    0.9464601,
    1.0431744,
    -0.37948397,
    0.18413225,
    1.6074077,
    -2.496784,
    -0.44237632,
    1.1085048,
    1.0128899,
    0.43405753,
    -0.16862528,
    0.31937033,
    -1.0025164,
    -0.060127847,
    1.1103966,
    -0.7716639,
    -0.3838148,
    0.96059936,
];

fn fixture_named(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nomic_bert_tiny.gguf")
}

/// The whole row: RoPE on Q/K, the gated FFN, no position table, and
/// llama.cpp's pooling.
#[test]
fn nomic_bert_matches_llama_cpp_pooled_embedding() {
    let model = EmbeddingModel::from_gguf_path(fixture()).expect("load the nomic fixture");
    let ours = model.embed("hello world", false).expect("embed");
    assert_eq!(ours.len(), NOMIC_GOLDEN.len());
    let mut worst = 0.0f32;
    for (a, b) in ours.iter().zip(&NOMIC_GOLDEN) {
        worst = worst.max((a - b).abs());
    }
    // 5e-3, and the number comes from a bisection rather than from
    // widening until it passed. On this f32 fixture the two engines
    // agree EXACTLY (to six decimals, every element) when the
    // attention output is switched off -- `attn_output.weight = 0`,
    // which leaves the embedding path, both LayerNorms and the gated
    // FFN -- and differ by about 3e-4 per element when it is on. The
    // residual is therefore inside the attention block and not in
    // anything this row added; it survives setting `attn_q.weight = 0`
    // (a uniform softmax, so no Q.K product at all) and is unchanged
    // by rounding K and V to f16 the way a reference that casts them
    // would, so the two obvious explanations are measured and
    // eliminated rather than assumed. `tests/bert_llama_cpp_parity.rs`
    // sees the same magnitude on a real checkpoint and attributes it
    // to Q8_0 activation quantization; this fixture is f32, so that
    // explanation does not cover it and the question is recorded here
    // rather than answered.
    assert!(worst < 5e-3, "max |ferrox - llama.cpp| = {worst}");
}

/// `jina-bert-v3` is the third architecture on this graph, and it is
/// the OTHER combination: `nomic-bert`'s rotation with `bert`'s
/// ungated GELU FFN.
///
/// Its own tensor loader (`jina-bert-v3.cpp:25-43`) creates no
/// position table and no QK-norm tensors, so the two branches of the
/// shared graph that could have differed are settled by what the file
/// holds -- which is why the row cost one line of
/// `bert_gguf_loader::ENCODER_ARCHS` and a fixture.
#[test]
fn jina_bert_v3_matches_llama_cpp_pooled_embedding() {
    let model = EmbeddingModel::from_gguf_path(fixture_named("jina_bert_v3_tiny.gguf"))
        .expect("load the jina-bert-v3 fixture");
    let hp = model.hparams().expect("a BERT-family encoder");
    assert_eq!(hp.arch, "jina-bert-v3");
    assert_eq!(hp.rope_theta, Some(1000.0), "it rotates");
    assert_eq!(
        hp.ffn,
        ferrox_models::bert_encoder::BertFfn::GeluSeq,
        "bert.cpp:179-187 includes JINA_BERT_V3 in the ungated GELU arm"
    );
    let ours = model.embed("hello world", false).expect("embed");
    let mut worst = 0.0f32;
    for (a, b) in ours.iter().zip(&JINA_V3_GOLDEN) {
        worst = worst.max((a - b).abs());
    }
    // Wider than the gated row's line for a reason that is measured
    // elsewhere in this tree: this FFN is GELU, and llama.cpp computes
    // GELU from a 65536-entry f16 table
    // (`tests/proj_bias_graphs.rs` carries the same class at 1e-2).
    assert!(worst < 1e-2, "max |ferrox - llama.cpp| = {worst}");
}

/// The two facts reached the encoder, and they came from the
/// ARCHITECTURE rather than from which tensors happen to be present.
#[test]
fn the_rotation_and_the_gate_are_read_from_the_architecture() {
    let model = EmbeddingModel::from_gguf_path(fixture()).expect("load the nomic fixture");
    let hp = model.hparams().expect("a BERT-family encoder");
    assert_eq!(hp.arch, "nomic-bert");
    assert_eq!(hp.rope_theta, Some(1000.0), "bert.cpp:126-133 rotates");
    assert_eq!(hp.rope_dim, 8, "the whole head");
    assert_eq!(
        hp.ffn,
        ferrox_models::bert_encoder::BertFfn::SwigluPar,
        "bert.cpp:195-201 is the gated SiLU"
    );
}
