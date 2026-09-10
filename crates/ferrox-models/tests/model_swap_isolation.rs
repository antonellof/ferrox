//! A checkpoint must answer the same whether or not another checkpoint
//! was loaded and dropped before it in the same process.
//!
//! `POST /admin/models/load` hot-swaps the served checkpoint, so the
//! second load in a process is what a Studio user gets when they pick a
//! different model from the selector. It came back wrong: on an M2 Pro
//! with `-ngl 99`, Llama-3.2-3B-Q4_K_M answered `'Tokyo'` on a fresh
//! server, `'Question\nI\n\nThe\n\n is\n\n'` after a swap through
//! Llama-3.2-1B-Q8_0, and the same corrupt string byte for byte on
//! every repeat (GitHub issue #180).
//!
//! The shape is process-global GPU state that outlives the `Decoder`
//! that filled it and is keyed by something a second model can
//! reproduce -- a host address plus a length. `paged_metal_parity`
//! already had to run one model per child process because of this, and
//! says so in its own docs.
//!
//! The check is a swapped run against a fresh run of the same
//! checkpoint, both as children of this process, so "fresh" means a
//! process that has loaded nothing else.
//!
//! Requires:
//! - `cargo test -p ferrox-models --features metal --test model_swap_isolation -- --ignored --nocapture`
//! - Apple Silicon + Metal (`FERROX_METAL=0` ablates it to the CPU path)
//! - the GGUFs under `models/` (`FERROX_TEST_MODELS_DIR` to point elsewhere)

#![cfg(feature = "metal")]

use std::path::{Path, PathBuf};

use ferrox_core::cache::KvCache;
use ferrox_gguf::ShardedGguf;
use ferrox_models::config::ModelConfig;
use ferrox_models::decoder::Decoder;
use ferrox_models::tokenizer::{GgufBpeTokenizer, GgufSpmTokenizer};

const PROMPT: &str = "The capital of France is";
/// Long enough that prefill runs the batched Metal kernels rather than
/// degenerating to a handful of single-token steps.
const PROMPT_TOKENS: usize = 64;
const MAX_NEW_TOKENS: usize = 24;

/// The pairs to run, as `(loaded first, then checked)`.
///
/// Both orders of the pair from the issue, because Q8_0 survived the
/// swap there and Q4_K_M did not: a fix that only helps one direction
/// is not a fix. The third pair is a different size class again, so a
/// pass is not an accident of two files that happen not to collide.
const PAIRS: &[(&str, &str)] = &[
    (
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    ),
    (
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
    ),
    (
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "Llama-3.2-1B-Instruct-Q6_K.gguf",
    ),
    // The MoE decode stack, which had a per-layer cache of its own
    // keyed on a bare pointer. This is the pair `paged_metal_parity`
    // names in its docs as the one that made it run a child process
    // per model.
    (
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "olmoe-1b-7b-0924-q4_0.gguf",
    ),
];

fn model_dir() -> PathBuf {
    // `FERROX_TEST_MODELS_DIR` because a git worktree has no `models/` of
    // its own, and this check is worth running from one.
    if let Ok(d) = std::env::var("FERROX_TEST_MODELS_DIR") {
        return PathBuf::from(d);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// Repeat until exactly `want` tokens, keeping at most one leading BOS.
fn stretch(mut tokens: Vec<usize>, want: usize, bos: Option<usize>) -> Vec<usize> {
    if tokens.len() >= want {
        tokens.truncate(want);
        return tokens;
    }
    let leading_bos = (bos.is_some() && tokens.first().copied() == bos).then(|| tokens[0]);
    let body: Vec<usize> = tokens[leading_bos.iter().count()..].to_vec();
    tokens.truncate(leading_bos.iter().count());
    while tokens.len() < want {
        let take = (want - tokens.len()).min(body.len());
        tokens.extend_from_slice(&body[..take]);
    }
    tokens
}

fn tokenize(file: &ShardedGguf) -> Vec<usize> {
    let raw: Vec<usize> = match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2" | "gemma4") => GgufBpeTokenizer::from_gguf(file)
            .expect("bpe tokenizer")
            .encode(PROMPT)
            .into_iter()
            .map(|i| i as usize)
            .collect(),
        Some("llama") => GgufSpmTokenizer::from_gguf(file)
            .expect("spm tokenizer")
            .encode(PROMPT)
            .into_iter()
            .map(|i| i as usize)
            .collect(),
        other => panic!("model-swap isolation does not cover tokenizer {other:?}"),
    };
    let bos = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);
    let mut tokens = raw;
    if ferrox_models::tokenizer::should_add_bos_token(file) {
        if let Some(b) = bos {
            if tokens.first() != Some(&b) {
                tokens.insert(0, b);
            }
        }
    }
    stretch(tokens, PROMPT_TOKENS, bos)
}

/// Load `path`, greedy-decode `MAX_NEW_TOKENS`, drop everything.
///
/// The drop is the point: it is what `/admin/models/load` does to the
/// outgoing model before the next one is mapped.
fn greedy_once(path: &Path) -> Vec<usize> {
    let file = ShardedGguf::open(path).expect("open gguf");
    let config = ModelConfig::from_gguf(&file).expect("model config");
    let eos = file
        .metadata_u64("tokenizer.ggml.eos_token_id")
        .map(|v| v as usize);
    let prompt = tokenize(&file);
    let decoder = Decoder::from_gguf(path, config).expect("decoder");

    let mut caches: Vec<KvCache> = (0..decoder.layers.len())
        .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
        .collect();
    let mut logits = decoder.forward_batch_last(&prompt, 0, &mut caches);
    let mut out = Vec::with_capacity(MAX_NEW_TOKENS);
    for pos in (prompt.len()..).take(MAX_NEW_TOKENS) {
        let next = argmax(&logits);
        out.push(next);
        if Some(next) == eos {
            break;
        }
        logits = decoder.forward_token(next, pos, &mut caches);
    }
    out
}

/// Env var carrying the `;`-separated checkpoint sequence a child runs.
const SEQUENCE_VAR: &str = "FERROX_TEST_SWAP_SEQUENCE";

/// Run `paths` in order in one process, returning each one's answer.
fn run_sequence_in_child(paths: &[PathBuf]) -> Vec<Vec<usize>> {
    let exe = std::env::current_exe().expect("test binary path");
    let joined = paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(";");
    let out = std::process::Command::new(&exe)
        .args([
            "--ignored",
            "--exact",
            "--nocapture",
            "a_checkpoint_answers_the_same_after_another_checkpoint_was_loaded_first",
        ])
        .env(SEQUENCE_VAR, &joined)
        .output()
        .expect("spawning the sequence child");
    let text =
        String::from_utf8_lossy(&out.stderr).into_owned() + &String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child running [{joined}] failed:\n{text}"
    );
    let answers: Vec<Vec<usize>> = text
        .lines()
        .filter_map(|l| l.strip_prefix("SWAP-ANSWER "))
        .map(|l| {
            l.split_whitespace()
                .map(|t| t.parse::<usize>().expect("token id"))
                .collect()
        })
        .collect();
    assert_eq!(
        answers.len(),
        paths.len(),
        "child ran {} of {} checkpoints:\n{text}",
        answers.len(),
        paths.len()
    );
    answers
}

/// Loading a checkpoint may not change what the NEXT checkpoint says.
///
/// A model whose answer depends on what the process loaded before it is
/// serving one model's weights through another model's cached buffers,
/// and it does it at full speed and full fluency.
#[test]
#[ignore = "needs Apple Metal GPU + the GGUFs under models/ (FERROX_TEST_MODELS_DIR to relocate)"]
fn a_checkpoint_answers_the_same_after_another_checkpoint_was_loaded_first() {
    // Honour a pre-set value so `FERROX_METAL=0 cargo test …` ablates to
    // the CPU path, which is how a GPU-cache cause is told apart from a
    // host-side one.
    for (k, v) in [("FERROX_METAL", "1"), ("FERROX_METAL_ATTN", "1")] {
        if std::env::var(k).is_err() {
            std::env::set_var(k, v);
        }
    }

    // Child mode: run the sequence this process was handed and report.
    if let Ok(seq) = std::env::var(SEQUENCE_VAR) {
        for path in seq.split(';') {
            let answer = greedy_once(Path::new(path));
            let ids = answer
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            println!("SWAP-ANSWER {ids}");
        }
        return;
    }

    if ferrox_metal::gpu::probe().is_none() && std::env::var("FERROX_METAL").as_deref() != Ok("0") {
        eprintln!("skip: no Metal GPU detected");
        return;
    }

    let mut ran = 0usize;
    let mut broken: Vec<String> = Vec::new();
    for (first, second) in PAIRS {
        let first_path = model_dir().join(first);
        let second_path = model_dir().join(second);
        if !first_path.exists() || !second_path.exists() {
            eprintln!("skip: {first} or {second} missing");
            continue;
        }
        let alone = run_sequence_in_child(std::slice::from_ref(&second_path));
        let swapped = run_sequence_in_child(&[first_path, second_path]);
        let want = &alone[0];
        let got = &swapped[1];
        eprintln!("{second} alone:        {want:?}");
        eprintln!("{second} after {first}: {got:?}");
        if want != got {
            broken.push(format!("{second} after {first}"));
        }
        ran += 1;
    }
    assert!(ran > 0, "no model pair available to check");
    assert!(
        broken.is_empty(),
        "a checkpoint answered differently because another was loaded first: {broken:?}"
    );
}
