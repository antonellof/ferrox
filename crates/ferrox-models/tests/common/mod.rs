//! Helpers shared by the integration tests.
//!
//! Each of these had grown two copies. That matters more in a test than
//! in library code: a tolerance or a comparison that drifts between two
//! copies makes one suite quietly weaker than the other, and a test
//! that has stopped checking what it claims to check looks exactly like
//! a test that passes.
//!
//! Rust compiles every file under `tests/` as its own crate, so a
//! helper only one suite uses is dead code in the others. `#[allow]` on
//! the module rather than on each item, since which suite uses what is
//! not a property worth maintaining.

#![allow(dead_code)]

use ferrox_core::cache::KvCache;
use ferrox_models::{Decoder, ModelConfig};
use std::path::{Path, PathBuf};

/// Fails with the worst absolute difference and the first few values of
/// each side.
///
/// The worst difference rather than the first: a reference mismatch is
/// almost never at index 0, and reporting the first divergence hides
/// how bad it gets. The sample of both sides is what turns "0.03 > 0.01"
/// into a diagnosis.
pub fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: logit count");
    let worst = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst <= tol,
        "{what}: max |ferrox - llama.cpp| = {worst} > {tol}\n  ferrox: {:?}\n  llama:  {:?}",
        &got[..8.min(got.len())],
        &want[..8.min(want.len())]
    );
}

/// Every `.gguf` under `dir`, recursively.
///
/// A missing or unreadable directory yields nothing rather than
/// failing: these suites scan an optional local model directory, and a
/// machine that has none should skip rather than error.
pub fn collect_gguf(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_gguf(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("gguf") {
            out.push(p);
        }
    }
}

// --- llama.cpp-golden graph fixtures --------------------------------
//
// The harness behind `one_match_arm_graphs.rs` and
// `fixture_away_graphs.rs`. Both suites admit architectures to
// `capability::AUDITED_GENERIC_GQA` on the same evidence -- a tiny
// synthetic GGUF whose golden values come from llama.cpp's own graph via
// libllama -- so they must compare on the same forward paths, at the
// same tolerance, from the same prompt. Two copies of that would be two
// standards, and the weaker one would be invisible.

/// The token ids every graph fixture is driven with. Explicit ids, never
/// tokenized text: the fixtures' vocabularies are placeholders.
pub const GRAPH_PROMPT: [usize; 6] = [3, 7, 11, 19, 23, 5];

/// Float32 accumulation order differs between the two engines (ggml
/// blocks its matmuls; ferrox does not), so this is a numeric-agreement
/// tolerance, not a bit-exactness claim. The measured worst case across
/// the fixtures is ~1e-6; every sabotage test moves the outputs by orders
/// of magnitude more.
pub const GRAPH_TOL: f32 = 1e-5;

/// The tolerance for a row whose FFN is **GeGLU**, where the reference
/// itself is the approximate side.
///
/// ggml defines `GGML_GELU_FP16` (`ggml/src/ggml-cpu/vec.h:46`), so
/// llama.cpp's CPU GELU is a 65536-entry **f16 lookup table**: the input
/// is rounded to f16 to index it and the stored value is f16 too
/// (`vec.h:1414-1425`). `GGML_SILU_FP16` is NOT defined, so SiLU is
/// exact SIMD -- which is exactly why every SwiGLU row in these suites
/// agrees to ~1e-6 and the one GeGLU row does not.
///
/// This is the same shape as the K-quant `vec_dot_type` divergence in
/// `docs/plans/llama-cpp-gap-inventory.md` §10: two defensible
/// implementations, one of them the reference's, and the gap is not a
/// ferrox bug. It was measured rather than assumed. An independent numpy
/// forward pass over the same fixture
/// (`scripts/gemma_fixture_numpy_ref.py`) reproduces llama.cpp's logits
/// to **1.19e-7** when its GELU goes through ggml's table and to
/// **3.93e-5** when its GELU is exact -- the second number being, to the
/// digit, what ferrox shows.
///
/// 2e-4 is therefore ~5x the measured gap and still orders of magnitude
/// under every sabotage in these suites, each of which moves the logits
/// by more than 1e-2.
pub const GELU_TABLE_TOL: f32 = 2e-4;

pub fn graph_fixture_path(name: &str) -> String {
    format!(
        "{}/tests/fixtures/{name}_tiny.gguf",
        env!("CARGO_MANIFEST_DIR")
    )
}

pub fn load_graph_fixture(name: &str) -> Decoder {
    let path = graph_fixture_path(name);
    let file = ferrox_gguf::GgufFile::open(&path).expect("fixture opens");
    let config = ModelConfig::from_gguf(&file).expect("fixture config parses");
    Decoder::from_gguf(&path, config).expect("fixture loads")
}

pub fn graph_caches(decoder: &Decoder) -> Vec<KvCache> {
    decoder
        .layers
        .iter()
        .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
        .collect()
}

/// Prefill, decode and continuous batching are three separate bodies in
/// this repo, and the reason this helper runs all three on every
/// architecture is that they have diverged before: five model features
/// went missing from one of them while the others kept working.
pub fn assert_all_three_paths_match(name: &str, golden: &[f32]) {
    assert_all_three_paths_match_within(name, golden, GRAPH_TOL);
}

/// The same three paths at a stated tolerance, for the rows where the
/// REFERENCE is the approximate side.
///
/// Parameterised rather than copied: two bodies would be two standards,
/// and the weaker one would be invisible. Only [`GELU_TABLE_TOL`] is
/// ever passed here, and only by the GeGLU rows.
pub fn assert_all_three_paths_match_within(name: &str, golden: &[f32], tol: f32) {
    let decoder = load_graph_fixture(name);

    let mut kv = graph_caches(&decoder);
    assert_close(
        &decoder.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        golden,
        tol,
        &format!("{name}: prefill (forward_batch_last)"),
    );

    let mut kv = graph_caches(&decoder);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = decoder.forward_token(tok, pos, &mut kv);
    }
    assert_close(&out, golden, tol, &format!("{name}: decode"));

    let mut kv = vec![graph_caches(&decoder)];
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        let batch = decoder.forward_multi_seq(&[tok], &[pos], &mut kv);
        out = batch.into_iter().next().unwrap();
    }
    assert_close(&out, golden, tol, &format!("{name}: multi-seq"));
}

/// The worst absolute difference from the golden values, for the
/// sabotage tests: a mutation that does not move this is a mutation the
/// suite cannot see.
pub fn worst_vs(got: &[f32], golden: &[f32]) -> f32 {
    got.iter()
        .zip(golden.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

/// `KL(llama.cpp || ferrox)` over the softmax of the two logit vectors,
/// in nats.
///
/// The max absolute logit difference is what
/// [`assert_all_three_paths_match`] gates on, because it is the sharper
/// instrument on a 48-wide synthetic vocabulary. KL is reported beside
/// it because it is the number `ferrox parity` speaks in and the one a
/// reader can compare against the K-quant drift in
/// `docs/plans/llama-cpp-gap-inventory.md` §10 -- a comparison that only
/// means anything if both sides are quoted in the same unit.
///
/// Here rather than in each suite: it had grown two copies
/// (`granite_family_graphs`, `no_rope_layer_graphs`), and a helper that
/// drifts between two suites makes one of them quietly report a
/// different number under the same name.
pub fn kl_vs_golden(got: &[f32], golden: &[f32]) -> f64 {
    let softmax = |v: &[f32]| -> Vec<f64> {
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exp: Vec<f64> = v.iter().map(|&x| (x as f64 - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        exp.into_iter().map(|e| e / sum).collect()
    };
    let p = softmax(golden);
    let q = softmax(got);
    p.iter()
        .zip(q.iter())
        .map(|(&p, &q)| if p > 0.0 { p * (p / q).ln() } else { 0.0 })
        .sum()
}

// ---------------------------------------------------------------------
// Real-checkpoint sweeps
//
// Shared by `paged_metal_parity` and `model_swap_isolation`, which both
// greedy-decode one prompt on a real GGUF and compare the token ids two
// runs produce. Each had grown its own copy of the tokenize/stretch
// pair, and a prompt or a BOS rule that drifted between them would have
// made one suite quietly measure something else.
// ---------------------------------------------------------------------

/// Root the real-GGUF suites scan.
///
/// `FERROX_TEST_MODELS_DIR` because a git worktree has no `models/` of
/// its own, and these checks are worth running from one.
pub fn model_dir() -> PathBuf {
    if let Ok(d) = std::env::var("FERROX_TEST_MODELS_DIR") {
        return PathBuf::from(d);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

/// The greedy pick, which is what makes two runs comparable at all.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// Repeat until exactly `want` tokens, keeping at most one leading BOS
/// -- the same stretch `ferrox verify --prompt-tokens` performs, down
/// to only treating the first token as BOS when it really is one. A
/// checkpoint without BOS otherwise loses its first word, which makes a
/// suite run a different prompt from the one `verify` reports on.
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

/// `prompt` under this checkpoint's own tokenizer, with its BOS policy
/// applied, stretched to exactly `want` tokens.
///
/// Panics rather than skipping for a tokenizer family it does not
/// cover: a suite that silently ran a different prompt would report a
/// pass it had not earned.
pub fn prompt_tokens(file: &ferrox_gguf::ShardedGguf, prompt: &str, want: usize) -> Vec<usize> {
    use ferrox_models::tokenizer::{GgufBpeTokenizer, GgufSpmTokenizer, SpecialTokens};
    let raw: Vec<usize> = match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2" | "gemma4") => GgufBpeTokenizer::from_gguf(file)
            .expect("bpe tokenizer")
            .encode(prompt, SpecialTokens::Parse)
            .into_iter()
            .map(|i| i as usize)
            .collect(),
        Some("llama") => GgufSpmTokenizer::from_gguf(file)
            .expect("spm tokenizer")
            .encode(prompt, SpecialTokens::Parse)
            .into_iter()
            .map(|i| i as usize)
            .collect(),
        other => panic!("this suite does not cover tokenizer {other:?}"),
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
    stretch(tokens, want, bos)
}
