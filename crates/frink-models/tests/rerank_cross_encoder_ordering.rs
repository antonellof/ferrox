//! `/v1/rerank`'s model half, on a real cross-encoder checkpoint, with
//! the ORDER checked against an independent implementation.
//!
//! # Why this file exists
//!
//! The rerank route (PR #42) shipped assembled from pieces that were
//! each unit-tested and never once run together, because there was no
//! reranker checkpoint on the development machine (issue #43). "It
//! returns 200" is not evidence for a route whose entire contract is
//! that the ordering is right, and a wrong ordering is invisible: it
//! produces a ranking, just not the model's.
//!
//! Running it end to end found two defects that green CI could not:
//!
//! 1. **The checkpoint would not load at all.** `cls.output.weight` is
//!    a `1 x 384` projection, and GGUF stores it with `n_dims = 1`
//!    because ggml's trailing dimensions are implicitly 1. frink's
//!    `load_weight_matrix` demanded two, so every reranker died with
//!    `UnsupportedDtype("cls.output.weight (expected 2D, got shape
//!    [384])")`.
//! 2. **The ordering was wrong** — see
//!    [`scoring_both_halves_as_segment_zero_ranks_the_relevant_document_last`],
//!    which is the old behaviour, preserved as a test so the fix cannot
//!    silently revert. That is issue #44: the document half of the pair
//!    was embedded as "Sentence A".
//!
//! # Getting the checkpoint
//!
//! ```text
//! frink download sinjab/ms-marco-MiniLM-L6-v2-Q8_0-GGUF --local-dir models
//! cargo test -p frink-models --test rerank_cross_encoder_ordering -- --ignored --nocapture
//! ```
//!
//! 24 MB, `bert` / WordPiece, six layers, `n_embd = 384`, one output
//! labelled `LABEL_0`. `bge-reranker-*` cannot stand in for it: those
//! are XLM-R with a SentencePiece vocabulary and refuse at
//! `EmbedError::UnsupportedTokenizer`.
//!
//! **These tests FAIL rather than skip when the file is absent.** The
//! repo's other checkpoint tests print `SKIP` and return, which passes
//! in 0.00 s — and a `--ignored` run that silently passes is exactly
//! how a route ships unverified twice. `#[ignore]` already means "only
//! when asked"; asking and getting nothing is a failure.
//!
//! # Where the expected numbers come from
//!
//! `scripts/rerank_reference_ms_marco.py` — a NumPy transcription of
//! HuggingFace `BertForSequenceClassification` read straight from the
//! checkpoint's safetensors, with no `transformers` model classes and
//! no torch, so it is a genuinely second implementation rather than a
//! recording of frink's own output.
//!
//! # The one place frink does NOT match HuggingFace, and why (#82)
//!
//! llama.cpp's converter drops `bert.pooler.dense` ("we are only using
//! BERT for embeddings so we don't need the pooling layer",
//! `conversion/bert.py`), so THIS GGUF does not contain it and the head
//! runs as `classifier(cls_hidden)` instead of
//! `classifier(tanh(pooler(cls_hidden)))`. No engine can apply a tensor
//! that is not in the file. The measured effect is on the score SCALE
//! (about +-0.2 rather than about +-11), not on which document wins;
//! the reference script prints both columns so the difference stays
//! visible. Most of the assertions below are against the column frink
//! can actually reach, plus the *ordering* of the full HuggingFace
//! reference.
//!
//! What changed with #82's route 1 is the "cannot" in that paragraph:
//! it is a property of the FILE, not of frink. When a GGUF does carry
//! `cls`, frink runs it, and
//! [`the_head_reproduces_huggingface_when_the_gguf_carries_the_pooler`]
//! proves that against the `hf` column by splicing the checkpoint's own
//! pooler back in. frink still refuses to INVENT one — see
//! [`the_shipped_checkpoint_is_the_unpooled_regime_and_says_so`], which
//! is the regression bar for every reranker GGUF in circulation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use frink_gguf::{GgmlType, GgufValue, GgufWriter, ShardedGguf, TensorPlan};
use frink_models::{load_rank_head, pool, EmbeddingModel, PoolingType};

/// The Q8_0 conversion of `cross-encoder/ms-marco-MiniLM-L6-v2`.
///
/// Absence is a failure, not a skip: see the module docs.
fn checkpoint() -> PathBuf {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/ms-marco-MiniLM-L6-v2-Q8_0.gguf");
    assert!(
        path.exists(),
        "{} is missing. These tests are `#[ignore]`d precisely so that running them means \
         you want them to run, so this fails instead of passing in 0.00s.\n    \
         frink download sinjab/ms-marco-MiniLM-L6-v2-Q8_0-GGUF --local-dir models\n\
         (a git worktree has no models/ of its own — symlink the one in the main checkout)",
        path.display()
    );
    path
}

/// The query, and documents in an order that is NOT the answer.
///
/// Document 1 answers the question and documents 2 and 3 are about
/// something else entirely, so "the relevant one ranks first" is a
/// property of the model and not of the fixture. Document 4 is loosely
/// on-topic and document 0 is about the right city and the wrong thing,
/// which is what distinguishes a reranker from a keyword match.
const QUERY: &str = "How many people live in Berlin?";
const DOCUMENTS: [&str; 5] = [
    "Berlin is well known for its museums.",
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "The capital of France is Paris.",
    "Elephants are the largest land animals.",
    "Berlin is the capital and largest city of Germany by both area and population.",
];

/// `scripts/rerank_reference_ms_marco.py`, `frink` row: the NumPy
/// reference restricted to what the GGUF carries (no pooler, real
/// segment ids).
const REFERENCE_SCORES: [f32; 5] = [-0.027210, 0.073057, -0.172285, -0.245995, 0.010248];

/// `scripts/rerank_reference_ms_marco.py`, `hf` row: the ordering the
/// checkpoint's full HuggingFace head produces. frink must reproduce
/// this ORDER from the shipped GGUF even though that file cannot carry
/// the scores.
const REFERENCE_ORDER: [usize; 5] = [1, 4, 0, 2, 3];

/// `scripts/rerank_reference_ms_marco.py`, `hf` row: the SCORES of the
/// checkpoint's full head, `classifier(tanh(pooler(cls)))`.
///
/// About fifty times wider than [`REFERENCE_SCORES`], which is the
/// whole of issue #82: the ordering is the same and an absolute
/// threshold is not.
const HF_REFERENCE_SCORES: [f32; 5] = [-4.320078, 8.60714, -11.101217, -11.18758, 0.636921];

/// Q8_0 weights against an f64 reference. The largest deviation
/// measured across all four query sets in the script is 1.7e-3.
const TOLERANCE: f32 = 4e-3;

fn ranking(scores: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    order
}

/// The load itself, which is where this route died before anything
/// could be said about its ordering.
///
/// Three claims in one, because each was broken or untested:
///
/// * a `1 x n_embd` `cls.output.weight` stored with `n_dims = 1` loads;
/// * `load_rank_head` runs before `load_bert_encoder`, so
///   `assert_every_tensor_consumed` does not reject `cls.*` — the
///   ordering carried a load-bearing comment and no test;
/// * the head is the checkpoint's own, with its declared label.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf"]
fn a_real_reranker_checkpoint_loads_with_its_classification_head() {
    let model = EmbeddingModel::from_gguf_path(checkpoint()).expect("load the reranker");
    assert_eq!(model.architecture(), "bert");
    assert_eq!(model.n_embd(), 384);
    let head = model
        .rank_head()
        .expect("a reranker checkpoint with no head is not a reranker");
    assert_eq!(head.n_cls_out(), 1);
    assert_eq!(head.labels(), ["LABEL_0"]);
}

/// The input the head was trained on: `[CLS] query [SEP] document
/// [SEP]`, with the query half segment 0 and the document half segment
/// 1 — `tokenizer(query, document)`'s `token_type_ids`.
///
/// Checked against ids this checkpoint's own vocabulary produces, so a
/// tokenizer change that silently drops the boundary shows up here and
/// not as a slightly worse ranking.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf"]
fn the_pair_is_cls_query_sep_document_sep_with_the_document_on_segment_one() {
    let model = EmbeddingModel::from_gguf_path(checkpoint()).expect("load the reranker");
    let pair = model.rerank_input(QUERY, DOCUMENTS[0]).expect("pair input");

    // [CLS] how many people live in berlin ? [SEP] berlin is well known
    // for its museums . [SEP]
    assert_eq!(
        pair.tokens,
        vec![
            101, 2129, 2116, 2111, 2444, 1999, 4068, 1029, 102, 4068, 2003, 2092, 2124, 2005, 2049,
            9941, 1012, 102
        ]
    );
    assert_eq!(
        pair.segments,
        vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    );
    assert_eq!(pair.tokens.len(), pair.segments.len());
    // The boundary is the first [SEP] and it belongs to the query half.
    assert_eq!(pair.tokens[8], 102);
    assert_eq!(pair.segments[8], 0);
    assert_eq!(pair.segments[9], 1);
}

/// **The point of the whole file.** The document that answers the
/// question must rank FIRST, and the assertion is on the INDEX, not on
/// a score threshold: a score says nothing without the others to
/// compare it against, and it is the index a client joins its own array
/// back onto.
///
/// Also asserts the scores themselves against the NumPy reference. That
/// is what rules out "it ranks plausibly for the wrong reason": a
/// cosine similarity, or a head that ran on the wrong row, would still
/// put document 1 near the top on a fixture this easy.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf"]
fn the_relevant_document_ranks_first_and_the_scores_match_the_numpy_reference() {
    let model = EmbeddingModel::from_gguf_path(checkpoint()).expect("load the reranker");
    let scores: Vec<f32> = DOCUMENTS
        .iter()
        .map(|d| {
            let pair = model.rerank_input(QUERY, d).expect("pair input");
            model.rerank_score(&pair).expect("score")
        })
        .collect();

    let order = ranking(&scores);
    assert_eq!(
        order[0],
        1,
        "the document that answers the query ranked {} of {}: {scores:?}",
        order.iter().position(|&i| i == 1).unwrap() + 1,
        DOCUMENTS.len()
    );
    assert_eq!(
        order,
        REFERENCE_ORDER.to_vec(),
        "frink ordered {order:?}, HuggingFace {REFERENCE_ORDER:?}, from {scores:?}"
    );

    for (i, (got, want)) in scores.iter().zip(REFERENCE_SCORES).enumerate() {
        assert!(
            (got - want).abs() < TOLERANCE,
            "document {i}: frink {got}, NumPy reference {want}"
        );
    }
    // A relevance logit, not a similarity: it is signed and it is not
    // confined to -1..1, so nothing here could be a cosine standing in
    // for the head.
    assert!(scores.iter().any(|s| *s < 0.0) && scores.iter().any(|s| *s > 0.0));
}

/// The number `/v1/rerank` reports is the classification head's output
/// applied to the CLS row, and this rebuilds that composition from its
/// parts to prove it.
///
/// `rerank_score` is one call; this is `encode`, then CLS pooling, then
/// [`frink_models::RankHead::score`], assembled here. If they disagree,
/// the route is reporting something other than the head — which is the
/// failure the whole route was written to avoid, and the reason
/// `/v1/embeddings` refuses a RANK checkpoint rather than serving a
/// cosine.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf"]
fn the_reported_score_is_the_head_run_on_the_cls_row() {
    let model = EmbeddingModel::from_gguf_path(checkpoint()).expect("load the reranker");
    let head = model.rank_head().expect("head is present");

    let pair = model.rerank_input(QUERY, DOCUMENTS[1]).expect("pair input");
    let hidden = model.pair_hidden_states(&pair).expect("hidden states");
    assert_eq!(hidden.len(), pair.tokens.len() * model.n_embd());
    let cls = pool(&hidden, model.n_embd(), PoolingType::Cls).expect("cls row");
    let by_hand = head.score(&cls);

    let by_route = model.rerank_score(&pair).expect("score");
    assert_eq!(
        by_hand, by_route,
        "the score the model reports is not the head applied to the CLS row"
    );
    // The head is doing real work: 384 floats in, one out, and it is
    // not the CLS row's own first element passed through.
    assert_eq!(cls.len(), 384);
    assert_ne!(by_route, cls[0]);
}

/// **The old behaviour, kept as a test so the fix cannot revert.**
///
/// Before issue #44 was fixed, `encode` added row 0 of
/// `token_types.weight` at every position, matching llama.cpp — whose
/// `llama_batch` carries no segment ids, so its `bert.cpp` views
/// `type_embd` at offset 0 and says token types are hardcoded to zero.
/// #44 recorded that as a defensible shared deviation and asked for it
/// to be MEASURED before being changed. Measured, on this checkpoint,
/// it puts the document that answers the question DEAD LAST out of
/// five, and it does the same in two of the other three query sets in
/// `scripts/rerank_reference_ms_marco.py`.
///
/// So the deviation is not a scale difference, it is a different
/// ranking, and this asserts the size of what was fixed rather than
/// leaving "it changes the order" as a claim in a commit message.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf"]
fn scoring_both_halves_as_segment_zero_ranks_the_relevant_document_last() {
    let model = EmbeddingModel::from_gguf_path(checkpoint()).expect("load the reranker");
    let head = model.rank_head().expect("head is present");

    let segment_blind: Vec<f32> = DOCUMENTS
        .iter()
        .map(|d| {
            let mut pair = model.rerank_input(QUERY, d).expect("pair input");
            // The old graph, expressed as data rather than as a second
            // copy of the forward pass: every position on "Sentence A".
            pair.segments.fill(0);
            let hidden = model.pair_hidden_states(&pair).expect("hidden states");
            let cls = pool(&hidden, model.n_embd(), PoolingType::Cls).expect("cls row");
            head.score(&cls)
        })
        .collect();

    let order = ranking(&segment_blind);
    assert_eq!(
        *order.last().unwrap(),
        1,
        "the segment-blind graph no longer ranks the relevant document last ({order:?} from \
         {segment_blind:?}); if the reference changed, re-run \
         scripts/rerank_reference_ms_marco.py before relaxing this"
    );
    assert_ne!(order, REFERENCE_ORDER.to_vec());
}

/// `bert.pooler.dense.{weight,bias}` as dumped by
/// `scripts/rerank_reference_ms_marco.py --dump-pooler`: the eight-byte
/// magic, `[out, in]` as two little-endian `u32`, then `out * in`
/// float32 in row-major `[out][in]` order, then `out` float32 of bias.
///
/// Absence is a failure for the same reason the checkpoint's is — see
/// the module docs.
fn pooler_weights() -> (usize, usize, Vec<u8>, Vec<u8>) {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/ms-marco-MiniLM-L6-v2-pooler.bin");
    assert!(
        path.exists(),
        "{} is missing. It is the checkpoint's own pooler, which the GGUF does not carry \
         (issue #82), dumped from the safetensors by the same NumPy reference this file's \
         golden values come from:\n    \
         python3 scripts/rerank_reference_ms_marco.py --dump-pooler \
         models/ms-marco-MiniLM-L6-v2-pooler.bin",
        path.display()
    );
    let raw = std::fs::read(&path).expect("read the pooler dump");
    assert!(
        raw.len() >= 16 && &raw[..8] == b"FXPOOL01",
        "{} is not a --dump-pooler file (magic is {:?})",
        path.display(),
        &raw[..raw.len().min(8)]
    );
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
    let (out, inp) = (u32_at(8), u32_at(12));
    let w_end = 16 + out * inp * 4;
    let b_end = w_end + out * 4;
    assert_eq!(
        raw.len(),
        b_end,
        "{} is {} bytes but [{out}, {inp}] + [{out}] needs {b_end}",
        path.display(),
        raw.len()
    );
    (
        out,
        inp,
        raw[16..w_end].to_vec(),
        raw[w_end..b_end].to_vec(),
    )
}

/// Writes a GGUF carrying ONLY a classification head: this
/// checkpoint's own `cls.output` bytes, copied verbatim with their
/// dtype and shape, plus the `cls` the converter deleted.
///
/// A head-only file rather than a rewritten 25 MB checkpoint, because
/// the encoder half is already proven against the real weights by every
/// other test here — what has never run is the loader finding `cls` in
/// a file and orienting it. `load_rank_head` takes the architecture and
/// the width as parameters, so it needs no hparams and no encoder
/// tensors to do that.
fn write_pooled_head_gguf(
    file: &ShardedGguf,
    out: usize,
    inp: usize,
    w: &[u8],
    b: &[u8],
) -> PathBuf {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("bert".to_string()),
    );
    metadata.insert(
        "bert.classifier.output_labels".to_string(),
        GgufValue::Array(vec![GgufValue::String("LABEL_0".to_string())]),
    );

    // GGUF's `ne[]` is fastest-dimension-first, so a `[out, in]` matrix
    // is stored `[in, out]`. Getting this backwards is caught by
    // `load_rank_head`'s own width check, and the pooler is square, so
    // it is stated rather than relied upon.
    let mut plan = vec![
        TensorPlan {
            name: "cls.weight".to_string(),
            shape: vec![inp as u64, out as u64],
            dtype: GgmlType::F32,
            byte_len: w.len(),
        },
        TensorPlan {
            name: "cls.bias".to_string(),
            shape: vec![out as u64],
            dtype: GgmlType::F32,
            byte_len: b.len(),
        },
    ];
    let copied: Vec<(&str, Vec<u8>)> = ["cls.output.weight", "cls.output.bias"]
        .iter()
        .map(|name| {
            let info = file
                .find_tensor(name)
                .unwrap_or_else(|| panic!("the checkpoint carries {name}"));
            let bytes = file.tensor_bytes(name).expect("read the tensor").to_vec();
            plan.push(TensorPlan {
                name: (*name).to_string(),
                shape: info.shape.clone(),
                dtype: info.dtype,
                byte_len: bytes.len(),
            });
            (*name, bytes)
        })
        .collect();

    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    let path = dir.join(format!("ms-marco-head-pooled-{}.gguf", std::process::id()));
    let sink = std::fs::File::create(&path).expect("create the fixture");
    let mut writer = GgufWriter::create(sink, &metadata, plan).expect("write the header");
    writer.write_tensor("cls.weight", w).expect("write cls");
    writer.write_tensor("cls.bias", b).expect("write cls.bias");
    for (name, bytes) in &copied {
        writer.write_tensor(name, bytes).expect("copy cls.output");
    }
    writer.finish().expect("finish the fixture");
    path
}

/// **The regression bar for every reranker GGUF that exists today.**
///
/// The shipped checkpoint has no pooler, and #82's route 1 must not
/// have changed a single thing about how it scores: no invented
/// identity pooler, no zero-filled `cls`, the same numbers this file
/// has asserted since #44. A fabricated pooler would be worse than the
/// uncalibrated score, because it would be a number no checkpoint ever
/// produced, reported under the model's own name.
///
/// It also pins the other half of route 1: which regime a caller is in
/// has to be VISIBLE, because the two differ by about fifty times and
/// nothing errors in between. `graph()` is what `/v1/rerank` puts on
/// every response as `frink_score_head`.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf"]
fn the_shipped_checkpoint_is_the_unpooled_regime_and_says_so() {
    let model = EmbeddingModel::from_gguf_path(checkpoint()).expect("load the reranker");
    let head = model.rank_head().expect("head is present");
    assert!(
        !head.has_pooler(),
        "llama.cpp's converter deletes pooler.dense, so this file cannot carry a cls.weight; \
         if it now does, the GGUF changed and HF_REFERENCE_SCORES is the golden set"
    );
    assert_eq!(head.graph(), "classifier(cls)");

    let scores: Vec<f32> = DOCUMENTS
        .iter()
        .map(|d| {
            let pair = model.rerank_input(QUERY, d).expect("pair input");
            model.rerank_score(&pair).expect("score")
        })
        .collect();
    for (i, (got, want)) in scores.iter().zip(REFERENCE_SCORES).enumerate() {
        assert!(
            (got - want).abs() < TOLERANCE,
            "document {i}: frink {got}, NumPy reference {want} -- the unpooled path moved"
        );
    }
}

/// **What route 1 of #82 buys.** Splice the checkpoint's own
/// `bert.pooler.dense` back in, and frink reproduces the `hf` column
/// of `scripts/rerank_reference_ms_marco.py` -- the scores the
/// cross-encoder was trained to produce -- with no change to frink
/// beyond reading a tensor that is present.
///
/// This is the evidence for the claim route 1 exists to make: a
/// converter that keeps `pooler.dense` needs no second change here. It
/// was a claim about a code path that had never executed, because every
/// published reranker GGUF is missing the tensor.
///
/// The encoder is the real Q8_0 one and the reference is f64, so the
/// tolerance is looser than [`TOLERANCE`] in absolute terms and much
/// tighter relative to a +-11 range. The assertion that carries the
/// point is the one against [`REFERENCE_SCORES`]: the same hidden
/// states, through the pooled head, are about FIFTY times larger.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf and its --dump-pooler sidecar"]
fn the_head_reproduces_huggingface_when_the_gguf_carries_the_pooler() {
    let model = EmbeddingModel::from_gguf_path(checkpoint()).expect("load the reranker");
    let (out, inp, w, b) = pooler_weights();
    assert_eq!((out, inp), (model.n_embd(), model.n_embd()));

    let source = ShardedGguf::open(checkpoint()).expect("reopen the checkpoint");
    let path = write_pooled_head_gguf(&source, out, inp, &w, &b);
    let spliced = ShardedGguf::open(&path).expect("open the head fixture");
    let head = load_rank_head(&spliced, "bert", model.n_embd(), 1e-12)
        .expect("the head loads")
        .expect("the fixture carries a head");
    assert!(head.has_pooler());
    assert_eq!(head.graph(), "classifier(tanh(pooler(cls)))");

    let pooled: Vec<f32> = DOCUMENTS
        .iter()
        .map(|d| {
            let pair = model.rerank_input(QUERY, d).expect("pair input");
            let hidden = model.pair_hidden_states(&pair).expect("hidden states");
            let cls = pool(&hidden, model.n_embd(), PoolingType::Cls).expect("cls row");
            head.score(&cls)
        })
        .collect();
    std::fs::remove_file(&path).ok();

    // Q8_0 weights against an f64 reference, on a +-11 range: the
    // largest deviation measured across these five documents is 0.011,
    // i.e. 0.1% -- tighter in relative terms than TOLERANCE is on the
    // unpooled +-0.2 range.
    const POOLED_TOLERANCE: f32 = 0.02;
    for (i, (got, want)) in pooled.iter().zip(HF_REFERENCE_SCORES).enumerate() {
        assert!(
            (got - want).abs() < POOLED_TOLERANCE,
            "document {i}: frink with the pooler {got}, HuggingFace {want}"
        );
    }
    assert_eq!(ranking(&pooled), REFERENCE_ORDER.to_vec());

    // And the size of what #82 is about: the same encoder, the same
    // documents, one tensor's presence, fifty times the range.
    let widest = |v: &[f32]| v.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let unpooled = widest(&REFERENCE_SCORES);
    assert!(
        widest(&pooled) > 20.0 * unpooled,
        "the pooled head's range ({}) is not the calibration difference #82 describes \
         against the unpooled {unpooled}",
        widest(&pooled)
    );
}
