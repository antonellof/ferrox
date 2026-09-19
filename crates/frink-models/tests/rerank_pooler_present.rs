//! `load_rank_head` reads the pooler **when the file carries it**, and
//! runs `classifier(tanh(pooler(cls)))` when it does — issue #82.
//!
//! # Why this file exists
//!
//! `cls` in a GGUF is HuggingFace's `bert.pooler.dense`, and llama.cpp's
//! converter deletes it by name (`conversion/bert.py`,
//! `BertModel.filter_tensors`: "we are only using BERT for embeddings so
//! we don't need the pooling layer"). Every reranker GGUF anyone has
//! downloaded is therefore missing it, `ms-marco-MiniLM-L6-v2-Q8_0`
//! included — which is exactly why the `dense` branch of
//! [`frink_models::RankHead`] had **no end-to-end evidence at all**.
//! It was unit-tested against a hand-built `RankHead`, and the loader
//! path that has to find `cls.weight` in a file and orient it correctly
//! had never once run.
//!
//! That matters more than an untested branch usually does, because the
//! whole point of route 1 of #82 is the promise that "a converter which
//! keeps `pooler.dense` immediately works, with no second change to
//! frink". A promise about a code path nobody has executed is a guess.
//!
//! # Why the fixture is synthetic
//!
//! `crates/frink-models/tests/rerank_cross_encoder_ordering.rs` covers
//! the real checkpoint, and its
//! `the_head_reproduces_huggingface_when_the_gguf_carries_the_pooler`
//! covers the real *pooler* — but both need `models/`, so both are
//! `#[ignore]`d and CI never runs them. These tests run everywhere, on a GGUF they write
//! themselves, so the regression bar for "frink reads a pooler" does
//! not depend on a 25 MB download.
//!
//! Every expected number below is a closed-form expression over the
//! fixture's own constants rather than a recorded float: a value copied
//! out of a previous run agrees with whatever the code does today,
//! including the wrong thing.

use std::collections::BTreeMap;
use std::path::PathBuf;

use frink_gguf::{GgmlType, GgufValue, GgufWriter, ShardedGguf, TensorPlan};
use frink_models::load_rank_head;

/// The head's input width: the encoder's `n_embd`, and the number
/// `cls.weight` must take as its INPUT.
const N_EMBD: usize = 3;

/// The pooler, `[out=2, in=3]` in frink's row-major orientation. It is
/// deliberately NOT square: GGUF stores `ne[]` fastest-dimension-first,
/// so this tensor's on-disk shape is `[3, 2]` and a loader that forgot
/// to reverse it would build a `[2, 3]`-shaped matrix and be caught by
/// `load_rank_head`'s own `cols() != n_embd` check. A square fixture
/// would pass either way.
const POOLER_W: [f32; 6] = [0.5, -0.25, 1.0, -1.5, 0.75, 0.25];
const POOLER_B: [f32; 2] = [0.1, -0.2];

/// `cls.output` for the pooled head: `[out=1, in=2]`, stored with
/// `n_dims = 1` the way `cross-encoder/ms-marco-MiniLM-L6-v2` stores
/// its own — ggml's trailing dimensions are implicitly 1, and reading
/// that back as a 1-D vector instead of a `1 x n` matrix is the defect
/// that made the real checkpoint unloadable (see the ordering test's
/// module docs).
const CLASSIFIER_W_POOLED: [f32; 2] = [2.0, -3.0];
/// `cls.output` for the head with no pooler: `[out=1, in=3]`, because
/// there the classifier reads the CLS row directly.
const CLASSIFIER_W_DIRECT: [f32; 3] = [2.0, -3.0, 0.5];
const CLASSIFIER_B: [f32; 1] = [0.25];

/// The CLS row fed to the head. Signed and not symmetric, so a sign
/// error or a reversed row survives nothing.
const CLS_ROW: [f32; N_EMBD] = [1.0, -2.0, 0.5];

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Writes a `bert` GGUF carrying nothing but a classification head.
///
/// `load_rank_head` is given the architecture and the width as
/// parameters, so it needs no hparams and no encoder tensors — which is
/// the point: this exercises the head loader alone, and the encoder is
/// already covered against a real checkpoint elsewhere.
///
/// `tensors` is `(name, gguf_shape, data)`, and `gguf_shape` is written
/// verbatim, so a test can store a `1 x n` projection as `[n]` exactly
/// as the real converter does.
fn write_head_gguf(name: &str, tensors: &[(&str, Vec<u64>, Vec<f32>)]) -> PathBuf {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("bert".to_string()),
    );
    metadata.insert(
        "bert.classifier.output_labels".to_string(),
        GgufValue::Array(vec![GgufValue::String("LABEL_0".to_string())]),
    );

    let plan: Vec<TensorPlan> = tensors
        .iter()
        .map(|(n, shape, data)| TensorPlan {
            name: (*n).to_string(),
            shape: shape.clone(),
            dtype: GgmlType::F32,
            byte_len: data.len() * 4,
        })
        .collect();

    // `target/` rather than the system temp dir: a leftover file is
    // then swept by `cargo clean` and is never mistaken for a
    // checkpoint. The pid keeps two concurrent `cargo test` runs apart.
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    let path = dir.join(format!("{name}-{}.gguf", std::process::id()));
    let file = std::fs::File::create(&path).expect("create the fixture");
    let mut w = GgufWriter::create(file, &metadata, plan).expect("write the header");
    for (n, _, data) in tensors {
        w.write_tensor(n, &f32_bytes(data)).expect("write a tensor");
    }
    w.finish().expect("finish the fixture");
    path
}

/// `tanh(POOLER_W · CLS_ROW + POOLER_B)`, written out rather than
/// looped, so this is a second statement of the arithmetic and not a
/// second call into the code under test.
fn expected_pooled() -> [f32; 2] {
    let p0 = 0.5 * CLS_ROW[0] + -0.25 * CLS_ROW[1] + 1.0 * CLS_ROW[2] + POOLER_B[0];
    let p1 = -1.5 * CLS_ROW[0] + 0.75 * CLS_ROW[1] + 0.25 * CLS_ROW[2] + POOLER_B[1];
    [p0.tanh(), p1.tanh()]
}

/// **The point of the file.** A GGUF that carries `cls.weight` /
/// `cls.bias` loads them, and the score is
/// `classifier(tanh(pooler(cls)))` — the composition the checkpoint was
/// trained as.
///
/// The assertion is against the hand-written composition AND against
/// the same composition with the `tanh` removed, because the tanh is
/// the entire difference between the two score regimes of #82. Dropping
/// it still produces a plausible signed relevance logit -- on the real
/// checkpoint about fifty times bigger, which is exactly what a
/// thresholding client silently gets wrong.
#[test]
fn a_gguf_carrying_a_pooler_runs_classifier_tanh_pooler() {
    let path = write_head_gguf(
        "rerank-head-pooled",
        &[
            ("cls.weight", vec![3, 2], POOLER_W.to_vec()),
            ("cls.bias", vec![2], POOLER_B.to_vec()),
            // `n_dims = 1`, as the real converter writes a 1 x n head.
            ("cls.output.weight", vec![2], CLASSIFIER_W_POOLED.to_vec()),
            ("cls.output.bias", vec![1], CLASSIFIER_B.to_vec()),
        ],
    );
    let file = ShardedGguf::open(&path).expect("open the fixture");
    let head = load_rank_head(&file, "bert", N_EMBD, 1e-12)
        .expect("the head loads")
        .expect("the fixture carries a head");

    assert!(head.has_pooler(), "cls.weight is in the file");
    assert_eq!(head.graph(), "classifier(tanh(pooler(cls)))");
    assert_eq!(head.n_cls_out(), 1);
    assert_eq!(head.labels(), ["LABEL_0"]);

    let t = expected_pooled();
    let want = CLASSIFIER_W_POOLED[0] * t[0] + CLASSIFIER_W_POOLED[1] * t[1] + CLASSIFIER_B[0];
    let got = head.score(&CLS_ROW);
    assert!(
        (got - want).abs() < 1e-6,
        "head produced {got}, hand-computed classifier(tanh(pooler(cls))) is {want}"
    );

    // The same graph with the tanh deleted. It is a different number by
    // a wide margin, so "the tanh ran" is asserted and not assumed.
    let p0 = 0.5 * CLS_ROW[0] + -0.25 * CLS_ROW[1] + 1.0 * CLS_ROW[2] + POOLER_B[0];
    let p1 = -1.5 * CLS_ROW[0] + 0.75 * CLS_ROW[1] + 0.25 * CLS_ROW[2] + POOLER_B[1];
    let no_tanh = CLASSIFIER_W_POOLED[0] * p0 + CLASSIFIER_W_POOLED[1] * p1 + CLASSIFIER_B[0];
    assert!(
        (got - no_tanh).abs() > 1.0,
        "a head with no tanh would have scored {no_tanh}, and {got} is indistinguishable"
    );

    std::fs::remove_file(&path).ok();
}

/// **The regression bar for every reranker GGUF that exists today.**
/// A file with no `cls.weight` is scored EXACTLY as it was before #82's
/// route 1: `classifier(cls)`, with no invented identity pooler and no
/// zero-filled `cls`.
///
/// A fabricated pooler is the one outcome worse than the uncalibrated
/// score, because it would be a number no checkpoint ever produced,
/// reported under the model's name. This is the test that says frink
/// did not do that.
#[test]
fn a_gguf_with_no_pooler_still_projects_the_cls_row_directly() {
    let path = write_head_gguf(
        "rerank-head-direct",
        &[
            ("cls.output.weight", vec![3], CLASSIFIER_W_DIRECT.to_vec()),
            ("cls.output.bias", vec![1], CLASSIFIER_B.to_vec()),
        ],
    );
    let file = ShardedGguf::open(&path).expect("open the fixture");
    let head = load_rank_head(&file, "bert", N_EMBD, 1e-12)
        .expect("the head loads")
        .expect("the fixture carries a head");

    assert!(!head.has_pooler(), "cls.weight is NOT in the file");
    assert_eq!(head.graph(), "classifier(cls)");

    let want = CLASSIFIER_W_DIRECT[0] * CLS_ROW[0]
        + CLASSIFIER_W_DIRECT[1] * CLS_ROW[1]
        + CLASSIFIER_W_DIRECT[2] * CLS_ROW[2]
        + CLASSIFIER_B[0];
    let got = head.score(&CLS_ROW);
    assert!(
        (got - want).abs() < 1e-6,
        "head produced {got}, hand-computed classifier(cls) is {want}"
    );
    // Linear, exactly: a stray tanh anywhere would break this, and it
    // is the property `classifier(cls)` names.
    let doubled: Vec<f32> = CLS_ROW.iter().map(|v| v * 2.0).collect();
    let unbiased = |s: f32| s - CLASSIFIER_B[0];
    assert!(
        (unbiased(head.score(&doubled)) - 2.0 * unbiased(got)).abs() < 1e-6,
        "the head with no pooler is not linear, so something applied a non-linearity"
    );

    std::fs::remove_file(&path).ok();
}
