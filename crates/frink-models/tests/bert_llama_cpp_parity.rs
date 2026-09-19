//! The BERT encoder against llama.cpp's own embeddings, on the same
//! GGUF.
//!
//! # Why this file exists in this shape
//!
//! `frink parity` compares first-token logit distributions, and an
//! encoder has no first-token logit: it has `n_tokens × n_embd` hidden
//! states and one pooled vector. So the same idea is applied to the
//! thing this model actually produces — the CLS-pooled, L2-normalized
//! embedding — element by element against the number llama.cpp's own
//! library returns for the same text.
//!
//! # Running it
//!
//! ```text
//! frink download CompendiumLabs/bge-small-en-v1.5-gguf \
//!     bge-small-en-v1.5-q8_0.gguf --local-dir models
//! ./tools/build_llama_logits.sh         # link against .scratch/llama.cpp,
//!                                       # NOT the Homebrew bottle
//! cargo test -p frink-models --test bert_llama_cpp_parity -- --ignored --nocapture
//! ```
//!
//! Both tests are `#[ignore]`d and skip loudly when their inputs are
//! missing: neither the 36 MB checkpoint nor a libllama build is in the
//! repository. `THE REFERENCE'S VINTAGE IS PART OF THE RESULT` —
//! `tools/build_llama_logits.sh`'s header explains why a Homebrew
//! bottle is not an acceptable oracle here.
//!
//! # Why the threshold is a cosine and not equality
//!
//! Measured on this checkpoint, frink and llama.cpp agree to
//! `cos ≈ 0.9998`, not to the last bit. That residual is the
//! **reference's** arithmetic, not a graph difference, and it was
//! attributed rather than assumed: llama.cpp's `ggml_vec_dot_q8_0_q8_0`
//! quantizes the *activations* to 8 bits per 32-value block, and
//! turning frink's own equivalent on (`FRINK_CPU_INT_DOT=1`) moves
//! frink's answer away from its own f32 answer by the same magnitude
//! (max |Δ| 3.6e-2, `cos` 0.99969). Frink's default path dequantizes
//! the weights and dots in f32, so it is the more accurate of the two;
//! there is no arrangement of these two engines that agrees exactly.
//!
//! # What 0.9998 is worth: the sabotage table
//!
//! A cosine near one only means something if the mistakes this graph
//! could plausibly contain score much worse. Each row below was
//! produced by breaking exactly one thing in the loaded model and
//! re-running the comparison on "Hello world":
//!
//! ```text
//! baseline                            cos = 0.999822
//! token-type embedding not added      cos = 0.990527
//! input LayerNorm bias dropped        cos = 0.991448
//! Q/K/V biases dropped                cos = 0.952229
//! MEAN pooling instead of CLS         cos = 0.958935
//! FFN biases dropped                  cos = 0.836248
//! learned position embedding dropped  cos = 0.703805
//! the two per-layer LayerNorms swapped cos = 0.683257
//! ```
//!
//! The *smallest* of those errors — failing to add one 384-float
//! constant — is fifty times further from llama.cpp than the noise
//! floor. So the threshold below sits between the two, at 0.9995.

use std::path::{Path, PathBuf};
use std::process::Command;

use frink_models::embedding_model::EmbeddingModel;
use frink_models::pooling::PoolingType;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<crate>/ has two ancestors")
        .to_path_buf()
}

fn model_path() -> Option<PathBuf> {
    let p = repo_root().join("models/bge-small-en-v1.5-q8_0.gguf");
    p.exists().then_some(p)
}

/// The corpus. Short, mixed-script, and including the cases that
/// exercise the tokenizer's normalizer as well as the graph.
const CASES: &[&str] = &[
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "What is the capital of France?",
    "def main():\n    print(\"hi\")",
    "Représentant naïve café",
    "东京是日本的首都",
    "a",
    "Embeddings are dense vector representations of text used for retrieval.",
];

/// Loads the real checkpoint and embeds every case. No oracle: this is
/// the test that says the graph runs at all, that the checkpoint's own
/// `pooling_type` was read, and that the vectors are finite and
/// normalized.
#[test]
#[ignore = "needs models/bge-small-en-v1.5-q8_0.gguf"]
fn bge_small_loads_and_embeds() {
    let Some(path) = model_path() else {
        eprintln!("SKIP: models/bge-small-en-v1.5-q8_0.gguf not present");
        return;
    };
    let model = EmbeddingModel::from_gguf_path(&path).expect("load bge-small");
    assert_eq!(model.architecture(), "bert");
    assert_eq!(model.n_embd(), 384);
    assert_eq!(model.n_ctx_train(), 512);
    assert_eq!(
        model.pooling_type(),
        PoolingType::Cls,
        "bert.pooling_type = 2 is CLS; reading it wrong is the whole point of the key"
    );

    for case in CASES {
        let ids = model.token_ids(case);
        assert_eq!(ids.first(), Some(&101), "[CLS] must lead: {case:?}");
        assert_eq!(ids.last(), Some(&102), "[SEP] must trail: {case:?}");
        let v = model.embed(case, true).expect("embed");
        assert_eq!(v.len(), 384);
        assert!(
            v.iter().all(|x| x.is_finite()),
            "non-finite output: {case:?}"
        );
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "‖v‖ = {norm} for {case:?}");
    }

    // Semantics, not just shape: a paraphrase must sit closer than an
    // unrelated sentence. A graph with a transposed matrix or a dropped
    // residual still produces finite unit vectors; it does not produce
    // these.
    let cos = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let q = model.embed("How do I bake bread?", true).unwrap();
    let near = model
        .embed("What is a good recipe for baking bread?", true)
        .unwrap();
    let far = model
        .embed("The stock market fell three percent today.", true)
        .unwrap();
    let (s_near, s_far) = (cos(&q, &near), cos(&q, &far));
    assert!(
        s_near > s_far + 0.2,
        "paraphrase {s_near:.3} is not clearly closer than the unrelated sentence {s_far:.3}"
    );
}

/// The oracle: llama.cpp's own `llama_get_embeddings_seq` for the same
/// checkpoint, the same text, and the same pooling.
#[test]
#[ignore = "needs models/…gguf and target/llama_logits"]
fn frink_matches_llama_cpp_embeddings() {
    let Some(path) = model_path() else {
        eprintln!("SKIP: models/bge-small-en-v1.5-q8_0.gguf not present");
        return;
    };
    let tool = repo_root().join("target/llama_logits");
    if !tool.exists() {
        eprintln!("SKIP: target/llama_logits not built (./tools/build_llama_logits.sh)");
        return;
    }
    let model = EmbeddingModel::from_gguf_path(&path).expect("load bge-small");

    let mut worst = 0.0f32;
    let mut worst_case = "";
    for case in CASES {
        let out = Command::new(&tool)
            .arg("--embed")
            .arg(&path)
            .arg(case)
            .output()
            .expect("run llama_logits --embed");
        assert!(
            out.status.success(),
            "llama_logits --embed failed on {case:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let reference: Vec<f32> = String::from_utf8_lossy(&out.stdout)
            .split_ascii_whitespace()
            .map(|t| t.parse::<f32>().expect("float"))
            .collect();
        assert_eq!(reference.len(), model.n_embd(), "reference width, {case:?}");

        // The reference prints the ids it used. Frink's tokenizer is
        // already checked against this library elsewhere, but
        // `wrap_special` is not: this is where [CLS]/[SEP] is confirmed
        // to be added the same way, on the same text.
        let stderr = String::from_utf8_lossy(&out.stderr);
        let line = stderr
            .lines()
            .find(|l| l.starts_with("pooling_type "))
            .expect("llama_embed prints its ids");
        let reference_ids: Vec<u32> = line
            .rsplit("ids:")
            .next()
            .unwrap()
            .split_ascii_whitespace()
            .map(|t| t.parse().expect("id"))
            .collect();
        assert_eq!(
            model.token_ids(case),
            reference_ids,
            "token ids differ from llama.cpp for {case:?}"
        );

        // `--embed` prints the raw pooled vector, so compare raw.
        let ours = model.embed(case, false).expect("embed");
        let mut max_abs = 0.0f32;
        for (a, b) in ours.iter().zip(&reference) {
            max_abs = max_abs.max((a - b).abs());
        }
        let dot: f32 = ours.iter().zip(&reference).map(|(a, b)| a * b).sum();
        let na: f32 = ours.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = reference.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb);
        eprintln!("{case:?}: max|Δ| = {max_abs:.3e}, cos = {cos:.9}");
        assert!(
            cos > 0.999_5,
            "cosine {cos} against llama.cpp for {case:?} — the graph disagrees. \
             The noise floor on this checkpoint is 0.9998 and the mildest single-fault \
             sabotage measured 0.9905; see this file's header"
        );
        if max_abs > worst {
            worst = max_abs;
            worst_case = case;
        }
    }
    eprintln!("worst element-wise difference: {worst:.3e} on {worst_case:?}");
    assert!(
        worst < 6e-2,
        "worst element-wise difference {worst:.3e} on {worst_case:?} is larger than \
         quantized-kernel rounding explains"
    );
}
