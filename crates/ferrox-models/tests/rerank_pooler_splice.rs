//! `ferrox splice-pooler` end to end: a GGUF goes in without its
//! pooler, a safetensors supplies one, and the file that comes out
//! scores as `classifier(tanh(pooler(cls)))` -- or nothing comes out,
//! because the pooler was not this checkpoint's (issue #82).
//!
//! # Two kinds of test here
//!
//! The synthetic ones write both input files themselves, so they run
//! everywhere and every expected number is a closed-form expression
//! over the fixture's own constants. They pin the CONTRACT: what is
//! written, what is refused, and that a refusal leaves no file behind.
//!
//! The `#[ignore]`d one is the evidence for the claim in `docs/API.md`:
//! on the real `ms-marco-MiniLM-L6-v2` Q8_0 GGUF, spliced from the real
//! HuggingFace safetensors, `/v1/rerank`'s scores match the NumPy
//! transcription of `BertForSequenceClassification`
//! (`scripts/rerank_reference_ms_marco.py`, `hf` row) on all four query
//! sets. It FAILS rather than skips when either file is missing, for
//! the reason `rerank_cross_encoder_ordering.rs` gives: an `--ignored`
//! run that silently passes is how a route ships unverified.
//!
//! ```text
//! ferrox download sinjab/ms-marco-MiniLM-L6-v2-Q8_0-GGUF --local-dir models
//! ferrox download cross-encoder/ms-marco-MiniLM-L6-v2 model.safetensors --local-dir models/ms-marco-MiniLM-L6-v2
//! cargo test -p ferrox-models --test rerank_pooler_splice -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ferrox_gguf::{GgmlType, GgufFile, GgufValue, GgufWriter, ShardedGguf, TensorPlan};
use ferrox_models::rerank_pooler::POOLER_SOURCE_KEY;
use ferrox_models::{load_rank_head, splice_pooler, EmbeddingModel, SpliceError};

const N_EMBD: usize = 3;

/// The classifier, as it sits in BOTH files: `[out=1, in=3]`.
const CLASSIFIER_W: [f32; 3] = [2.0, -3.0, 0.5];
const CLASSIFIER_B: [f32; 1] = [0.25];
/// The pooler only the safetensors has: `[out=3, in=3]`, row-major
/// `[out][in]`, and deliberately not symmetric, so a transposed copy
/// gives a different number below.
const POOLER_W: [f32; 9] = [0.5, -0.25, 1.0, -1.5, 0.75, 0.25, 0.1, 0.2, -0.3];
const POOLER_B: [f32; 3] = [0.1, -0.2, 0.05];
const CLS_ROW: [f32; N_EMBD] = [1.0, -2.0, 0.5];

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn fixture_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    dir
}

fn fixture_path(name: &str, ext: &str) -> PathBuf {
    fixture_dir().join(format!("{name}-{}.{ext}", std::process::id()))
}

/// A `bert` GGUF the way the converter leaves a reranker: hparams,
/// labels, `cls.output.*` with the `1 x n` weight stored as `[n]`, and
/// NO `cls`. `extra` lets a test add tensors on top.
fn write_unpooled_gguf(name: &str, extra: &[(&str, Vec<u64>, Vec<f32>)]) -> PathBuf {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("bert".to_string()),
    );
    metadata.insert(
        "bert.embedding_length".to_string(),
        GgufValue::U32(N_EMBD as u32),
    );
    metadata.insert(
        "bert.attention.layer_norm_epsilon".to_string(),
        GgufValue::F32(1e-12),
    );
    metadata.insert(
        "bert.classifier.output_labels".to_string(),
        GgufValue::Array(vec![GgufValue::String("LABEL_0".to_string())]),
    );
    let mut tensors: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
        (
            "cls.output.weight",
            vec![N_EMBD as u64],
            CLASSIFIER_W.to_vec(),
        ),
        ("cls.output.bias", vec![1], CLASSIFIER_B.to_vec()),
    ];
    tensors.extend(extra.iter().cloned());
    let plan: Vec<TensorPlan> = tensors
        .iter()
        .map(|(n, shape, data)| TensorPlan {
            name: (*n).to_string(),
            shape: shape.clone(),
            dtype: GgmlType::F32,
            byte_len: data.len() * 4,
        })
        .collect();
    let path = fixture_path(name, "gguf");
    let file = std::fs::File::create(&path).expect("create the fixture");
    let mut w = GgufWriter::create(file, &metadata, plan).expect("write the header");
    for (n, _, data) in &tensors {
        w.write_tensor(n, &f32_bytes(data)).expect("write a tensor");
    }
    w.finish().expect("finish the fixture");
    path
}

/// A safetensors file by hand: 8-byte little-endian header length, the
/// JSON header, then the tensors' bytes contiguous in header order.
/// Written here rather than through a library so the format the splice
/// reads is a second statement of it.
fn write_safetensors(name: &str, tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> PathBuf {
    let mut header = String::from("{");
    let mut data = Vec::new();
    for (i, (n, shape, values)) in tensors.iter().enumerate() {
        let start = data.len();
        data.extend(f32_bytes(values));
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "\"{n}\":{{\"dtype\":\"F32\",\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",
            data.len()
        ));
    }
    header.push('}');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend(data);
    let path = fixture_path(name, "safetensors");
    std::fs::write(&path, bytes).expect("write the safetensors");
    path
}

/// The checkpoint's own head, as HuggingFace saves it.
fn hf_tensors(classifier_w: &[f32]) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    vec![
        ("classifier.weight", vec![1, N_EMBD], classifier_w.to_vec()),
        ("classifier.bias", vec![1], CLASSIFIER_B.to_vec()),
        (
            "bert.pooler.dense.weight",
            vec![N_EMBD, N_EMBD],
            POOLER_W.to_vec(),
        ),
        ("bert.pooler.dense.bias", vec![N_EMBD], POOLER_B.to_vec()),
    ]
}

/// `classifier(tanh(POOLER_W · CLS_ROW + POOLER_B))`, written out.
fn expected_pooled_score() -> f32 {
    let p0 = 0.5 * CLS_ROW[0] + -0.25 * CLS_ROW[1] + 1.0 * CLS_ROW[2] + POOLER_B[0];
    let p1 = -1.5 * CLS_ROW[0] + 0.75 * CLS_ROW[1] + 0.25 * CLS_ROW[2] + POOLER_B[1];
    let p2 = 0.1 * CLS_ROW[0] + 0.2 * CLS_ROW[1] + -0.3 * CLS_ROW[2] + POOLER_B[2];
    CLASSIFIER_W[0] * p0.tanh()
        + CLASSIFIER_W[1] * p1.tanh()
        + CLASSIFIER_W[2] * p2.tanh()
        + CLASSIFIER_B[0]
}

/// The same composition with the pooler transposed, which a copy in
/// the wrong orientation would produce. Must differ from the above or
/// the fixture cannot see a transposition.
fn transposed_pooled_score() -> f32 {
    let p0 = 0.5 * CLS_ROW[0] + -1.5 * CLS_ROW[1] + 0.1 * CLS_ROW[2] + POOLER_B[0];
    let p1 = -0.25 * CLS_ROW[0] + 0.75 * CLS_ROW[1] + 0.2 * CLS_ROW[2] + POOLER_B[1];
    let p2 = 1.0 * CLS_ROW[0] + 0.25 * CLS_ROW[1] + -0.3 * CLS_ROW[2] + POOLER_B[2];
    CLASSIFIER_W[0] * p0.tanh()
        + CLASSIFIER_W[1] * p1.tanh()
        + CLASSIFIER_W[2] * p2.tanh()
        + CLASSIFIER_B[0]
}

fn cleanup(paths: &[&Path]) {
    for p in paths {
        std::fs::remove_file(p).ok();
    }
}

/// **The point of the file.** The spliced GGUF carries the pooler under
/// llama.cpp's names, `load_rank_head` finds it with no change, and the
/// score is the trained composition -- in the right orientation.
#[test]
fn a_spliced_gguf_scores_as_classifier_tanh_pooler_in_the_right_orientation() {
    let gguf = write_unpooled_gguf("splice-ok", &[]);
    let st = write_safetensors("splice-ok", &hf_tensors(&CLASSIFIER_W));
    let out = fixture_path("splice-ok-out", "gguf");

    let done = splice_pooler(&gguf, &st, &out).expect("the splice succeeds");
    assert_eq!(done.n_embd, N_EMBD);
    assert_eq!(done.n_out, 1);
    assert_eq!(done.head_dtype, GgmlType::F32);
    assert_eq!(done.classifier_max_abs_diff, 0.0, "F32 in, F32 out: exact");

    let written = GgufFile::open(&out).expect("the output is a GGUF");
    // Every input tensor, then the pooler, and the provenance key.
    let names: Vec<&str> = written.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "cls.output.weight",
            "cls.output.bias",
            "cls.weight",
            "cls.bias"
        ]
    );
    assert_eq!(
        written.find_tensor("cls.weight").unwrap().shape,
        vec![N_EMBD as u64, N_EMBD as u64]
    );
    assert_eq!(
        written.metadata_str(POOLER_SOURCE_KEY),
        st.file_name().map(|n| n.to_str().unwrap())
    );
    assert_eq!(written.metadata_u64("bert.embedding_length"), Some(3));

    let sharded = ShardedGguf::open(&out).expect("reopen");
    let head = load_rank_head(&sharded, "bert", N_EMBD, 1e-12)
        .expect("loads")
        .expect("has a head");
    assert!(head.has_pooler());
    assert_eq!(head.graph(), "classifier(tanh(pooler(cls)))");
    let got = head.score(&CLS_ROW);
    let want = expected_pooled_score();
    assert!(
        (transposed_pooled_score() - want).abs() > 0.1,
        "the fixture cannot see a transposed pooler"
    );
    assert!(
        (got - want).abs() < 1e-6,
        "spliced head produced {got}, hand-computed {want} (transposed would be {})",
        transposed_pooled_score()
    );
    cleanup(&[&gguf, &st, &out]);
}

/// **A pooler from another checkpoint is refused, by element, and
/// leaves nothing on disk.** The safetensors is the checkpoint's own
/// with one classifier element moved by 1% of the tensor's maximum:
/// well inside what two fine-tunes differ by and well outside what any
/// storage precision rounds by. A wrong pooler is worse than none --
/// its scores look calibrated -- so a partially written file would be
/// the one outcome this test exists to rule out.
#[test]
fn a_pooler_whose_classifier_is_not_the_ggufs_is_refused_and_no_file_is_written() {
    let gguf = write_unpooled_gguf("splice-mismatch", &[]);
    let mut other = CLASSIFIER_W;
    other[1] += 0.03; // absmax is 3.0; the bound is 3.0 / 128 = 0.0234
    let st = write_safetensors("splice-mismatch", &hf_tensors(&other));
    let out = fixture_path("splice-mismatch-out", "gguf");

    let err = splice_pooler(&gguf, &st, &out).expect_err("a foreign pooler is refused");
    match &err {
        SpliceError::Mismatch { mismatch, .. } => {
            assert_eq!(mismatch.tensor, "cls.output.weight");
            assert_eq!(mismatch.index, 1);
            assert_eq!(mismatch.gguf, CLASSIFIER_W[1]);
            assert_eq!(mismatch.reference, other[1]);
        }
        other => panic!("refused for the wrong reason: {other}"),
    }
    assert!(err.to_string().contains("does not belong to"), "{err}");
    assert!(
        !out.exists(),
        "a refused splice must not leave {out:?} behind"
    );
    cleanup(&[&gguf, &st]);
}

/// A classifier bias that disagrees is the same refusal: the bias is
/// part of the head, and a checkpoint re-trained only in its bias is
/// still another checkpoint.
#[test]
fn a_classifier_bias_that_disagrees_is_refused_too() {
    let gguf = write_unpooled_gguf("splice-bias", &[]);
    let mut tensors = hf_tensors(&CLASSIFIER_W);
    tensors[1].2 = vec![CLASSIFIER_B[0] + 0.01]; // absmax 0.25; bound 0.00195
    let st = write_safetensors("splice-bias", &tensors);
    let out = fixture_path("splice-bias-out", "gguf");
    let err = splice_pooler(&gguf, &st, &out).expect_err("refused");
    assert!(
        matches!(&err, SpliceError::Mismatch { mismatch, .. } if mismatch.tensor == "cls.output.bias"),
        "{err}"
    );
    assert!(!out.exists());
    cleanup(&[&gguf, &st]);
}

/// The refusals that need no arithmetic: a file that already has its
/// pooler, a safetensors that has none, a pooler of the wrong width,
/// and a GGUF of another architecture. Each names what it saw.
#[test]
fn the_shape_refusals_name_what_they_saw() {
    let gguf = write_unpooled_gguf("splice-shapes", &[]);
    let st = write_safetensors("splice-shapes", &hf_tensors(&CLASSIFIER_W));
    let out = fixture_path("splice-shapes-out", "gguf");
    splice_pooler(&gguf, &st, &out).expect("first splice");

    // Splicing over the output: it already carries cls.weight, and the
    // refusal says where that came from.
    let twice = fixture_path("splice-shapes-twice", "gguf");
    let err = splice_pooler(&out, &st, &twice).expect_err("already pooled");
    assert!(matches!(err, SpliceError::AlreadyPooled { .. }), "{err}");
    assert!(err.to_string().contains("spliced from"), "{err}");
    assert!(!twice.exists());

    // A safetensors with no pooler at all.
    let no_pooler = write_safetensors("splice-no-pooler", &hf_tensors(&CLASSIFIER_W)[..2]);
    let err = splice_pooler(&gguf, &no_pooler, &twice).expect_err("no pooler");
    assert!(
        matches!(&err, SpliceError::MissingSafetensor { tried, .. } if tried.contains(&"bert.pooler.dense.weight")),
        "{err}"
    );

    // A pooler of another width: its classifier matches, so this is
    // the width check and not the identity check.
    let mut wide = hf_tensors(&CLASSIFIER_W);
    wide[2] = ("bert.pooler.dense.weight", vec![2, N_EMBD], vec![0.0; 6]);
    let wide = write_safetensors("splice-wide", &wide);
    let err = splice_pooler(&gguf, &wide, &twice).expect_err("wrong width");
    assert!(
        matches!(&err, SpliceError::Shape { name, shape, .. } if name == "bert.pooler.dense.weight" && *shape == vec![2, N_EMBD]),
        "{err}"
    );

    // A GGUF that is not bert.
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("llama".to_string()),
    );
    let llama = fixture_path("splice-llama", "gguf");
    GgufWriter::create(
        std::fs::File::create(&llama).unwrap(),
        &metadata,
        vec![TensorPlan {
            name: "x".into(),
            shape: vec![1],
            dtype: GgmlType::F32,
            byte_len: 4,
        }],
    )
    .unwrap()
    .write_tensor("x", &[0; 4])
    .unwrap();
    let err = splice_pooler(&llama, &st, &twice).expect_err("not bert");
    assert!(
        matches!(&err, SpliceError::NotBert { arch, .. } if arch == "llama"),
        "{err}"
    );
    assert!(!twice.exists());

    cleanup(&[&gguf, &st, &out, &no_pooler, &wide, &llama]);
}

// ---------------------------------------------------------------------
// The real checkpoint.
// ---------------------------------------------------------------------

fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn real_gguf() -> PathBuf {
    let path = models_dir().join("ms-marco-MiniLM-L6-v2-Q8_0.gguf");
    assert!(
        path.exists(),
        "{} is missing; this test fails rather than skips (see the module docs).\n    \
         ferrox download sinjab/ms-marco-MiniLM-L6-v2-Q8_0-GGUF --local-dir models",
        path.display()
    );
    path
}

/// `cross-encoder/ms-marco-MiniLM-L6-v2/model.safetensors`, from
/// `models/ms-marco-MiniLM-L6-v2/` or, failing that, the HuggingFace
/// cache the NumPy reference script populates.
fn real_safetensors() -> PathBuf {
    let local = models_dir().join("ms-marco-MiniLM-L6-v2/model.safetensors");
    if local.exists() {
        return local;
    }
    let cache = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/huggingface")
        })
        .join("hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots");
    let snapshot = std::fs::read_dir(&cache)
        .ok()
        .and_then(|d| d.flatten().next())
        .map(|e| e.path().join("model.safetensors"))
        .filter(|p| p.exists());
    snapshot.unwrap_or_else(|| {
        panic!(
            "neither {} nor a cached snapshot under {} exists; this test fails rather than \
             skips.\n    ferrox download cross-encoder/ms-marco-MiniLM-L6-v2 model.safetensors \
             --local-dir models/ms-marco-MiniLM-L6-v2",
            local.display(),
            cache.display()
        )
    })
}

/// `scripts/rerank_reference_ms_marco.py`, `hf` rows: the four query
/// sets, the scores of `classifier(tanh(pooler(cls)))` from the
/// checkpoint's safetensors in f64 NumPy, and their order.
const HF_REFERENCE: [(&str, &[&str], &[f32]); 4] = [
    (
        "How many people live in Berlin?",
        &[
            "Berlin is well known for its museums.",
            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
            "The capital of France is Paris.",
            "Elephants are the largest land animals.",
            "Berlin is the capital and largest city of Germany by both area and population.",
        ],
        &[-4.320078, 8.60714, -11.101217, -11.18758, 0.636921],
    ),
    (
        "What is the boiling point of water?",
        &[
            "Water freezes at 0 degrees Celsius at sea level.",
            "At standard atmospheric pressure water boils at 100 degrees Celsius.",
            "The Pacific Ocean is the largest ocean on Earth.",
            "Coffee is usually brewed just below boiling.",
        ],
        &[-7.443397, 3.964292, -11.025803, -9.224071],
    ),
    (
        "Who wrote Romeo and Juliet?",
        &[
            "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
            "The play is set in Verona, Italy.",
            "Python is a programming language created by Guido van Rossum.",
            "Juliet is fourteen years old in the play.",
        ],
        &[10.916302, -8.157381, -8.715409, -3.193443],
    ),
    (
        "How do I install Rust on macOS?",
        &[
            "Run the rustup installer script from rustup.rs to install the Rust toolchain.",
            "Rust is a systems programming language focused on safety.",
            "macOS Ventura was released in 2022.",
            "Cargo is the Rust package manager.",
        ],
        &[3.531187, -3.242298, -9.813922, -5.040743],
    ),
];

/// Q8_0 encoder against an f64 reference, on a range of about +-11.
/// The largest deviation measured across all seventeen pairs is 0.051
/// (document 1 of the second set), i.e. under 0.5% of the range; the
/// unpooled file's tolerance in `rerank_cross_encoder_ordering.rs` is
/// 4e-3 on a +-0.25 range, which is 1.6%.
const POOLED_TOLERANCE: f32 = 0.1;

fn ranking(scores: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    order
}

/// **The claim `docs/API.md` makes.** The published GGUF, spliced from
/// the published safetensors, loaded through the same
/// `EmbeddingModel::from_gguf_path` the server uses, scores every pair
/// of all four reference query sets within [`POOLED_TOLERANCE`] of
/// HuggingFace -- on the +-11 range the checkpoint was trained to
/// produce, not the +-0.2 the converter's output gives.
#[test]
#[ignore = "needs models/ms-marco-MiniLM-L6-v2-Q8_0.gguf and the checkpoint's safetensors"]
fn the_spliced_real_checkpoint_matches_huggingface_on_all_four_query_sets() {
    let out = fixture_path("ms-marco-L6-pooled", "gguf");
    let done = splice_pooler(&real_gguf(), &real_safetensors(), &out).expect("the splice");
    assert_eq!(done.n_embd, 384);
    assert_eq!(
        done.head_dtype,
        GgmlType::F16,
        "the published file's cls.output.weight"
    );
    assert!(
        done.classifier_max_abs_diff <= done.classifier_allowed,
        "{done:?}"
    );

    let model = EmbeddingModel::from_gguf_path(&out).expect("load the pooled reranker");
    let head = model.rank_head().expect("head");
    assert!(head.has_pooler());
    assert_eq!(head.graph(), "classifier(tanh(pooler(cls)))");

    let mut worst = 0.0f32;
    for (query, documents, want) in HF_REFERENCE {
        let scores: Vec<f32> = documents
            .iter()
            .map(|d| {
                let pair = model.rerank_input(query, d).expect("pair");
                model.rerank_score(&pair).expect("score")
            })
            .collect();
        assert_eq!(
            ranking(&scores),
            ranking(want),
            "{query}: ferrox {scores:?}, HuggingFace {want:?}"
        );
        for (i, (got, want)) in scores.iter().zip(want).enumerate() {
            let diff = (got - want).abs();
            worst = worst.max(diff);
            assert!(
                diff < POOLED_TOLERANCE,
                "{query}, document {i}: ferrox {got}, HuggingFace {want}"
            );
        }
    }
    eprintln!("largest |ferrox - HuggingFace| across 17 pairs: {worst}");
    std::fs::remove_file(&out).ok();
}
