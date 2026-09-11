//! Exact-checkpoint parity receipts: pin a real GGUF by SHA-256 and
//! assert critical next-token / greedy-decode properties against the
//! committed receipt JSON under `tests/receipts/`.
//!
//! Receipts are small (kilobytes); the multi-GB checkpoint stays on
//! disk. Tests are `#[ignore]`d — same convention as `kimi_real_data`
//! and CUDA/Metal hardware tests.
//!
//! Run:
//! ```text
//! export FERROX_TEST_RECEIPT_CHECKPOINT=/path/to/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
//! cargo test -p ferrox-models --test checkpoint_receipts -- --ignored --nocapture
//! ```

use std::fs;
use std::path::Path;

use ferrox_core::cache::KvCache;
use ferrox_gguf::ShardedGguf;
use ferrox_models::config::{ModelConfig, RopeLayout};
use ferrox_models::decoder::Decoder;
use ferrox_models::tokenizer::{GgufBpeTokenizer, SpecialTokens};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Receipt {
    id: String,
    checkpoint: Checkpoint,
    prompt: Prompt,
    checks: Vec<Check>,
}

#[derive(Debug, Deserialize)]
struct Checkpoint {
    filename: String,
    sha256: String,
    bytes: u64,
}

#[derive(Debug, Deserialize)]
struct Prompt {
    rendered: String,
    token_ids: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct Check {
    name: String,
    #[serde(default)]
    forbid_top1: Vec<usize>,
    #[serde(default)]
    require_in_top5: Vec<usize>,
    #[serde(default)]
    max_new_tokens: Option<usize>,
    #[serde(default)]
    decoded_substring: Option<String>,
    #[serde(default)]
    forbid_decoded_substring: Option<String>,
}

fn load_receipt(name: &str) -> Receipt {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/receipts")
        .join(name);
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
}

fn sha256_hex(path: &Path) -> String {
    let out = std::process::Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .expect("shasum must be available to verify receipt identity");
    assert!(out.status.success(), "shasum failed: {:?}", out.status);
    let line = String::from_utf8_lossy(&out.stdout);
    line.split_whitespace()
        .next()
        .expect("shasum output")
        .to_string()
}

fn top5(logits: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.into_iter().take(5).collect()
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
#[ignore = "needs FERROX_TEST_RECEIPT_CHECKPOINT pointing at the pinned Llama-3.1 Q4_K_M GGUF"]
fn llama31_8b_q4km_forced_continuation_receipt() {
    let receipt = load_receipt("llama31_8b_instruct_q4_k_m.json");
    let path = std::env::var("FERROX_TEST_RECEIPT_CHECKPOINT").unwrap_or_else(|_| {
        panic!(
            "set FERROX_TEST_RECEIPT_CHECKPOINT to the local path of {}",
            receipt.checkpoint.filename
        )
    });
    let path = Path::new(&path);
    assert!(
        path.exists(),
        "checkpoint missing at {path:?} (receipt {})",
        receipt.id
    );

    let meta = fs::metadata(path).expect("stat checkpoint");
    assert_eq!(
        meta.len(),
        receipt.checkpoint.bytes,
        "byte size mismatch for {} — refuse to run against a different file",
        receipt.id
    );
    let digest = sha256_hex(path);
    assert_eq!(
        digest, receipt.checkpoint.sha256,
        "SHA-256 mismatch for {} — refuse to run against a different file",
        receipt.id
    );

    let file = ShardedGguf::open(path).expect("open GGUF");
    let config = ModelConfig::from_gguf(&file).expect("from_gguf");
    assert_eq!(
        config.rope_layout,
        RopeLayout::Norm,
        "llama architecture must select Norm RoPE layout"
    );
    assert!(
        config.rope_freqs.is_some(),
        "Llama-3.1 must load rope_freqs.weight"
    );

    let tok = GgufBpeTokenizer::from_gguf(&file).expect("tokenizer");
    let decoder = Decoder::from_gguf(path, config.clone()).expect("load decoder");

    let mut tokens: Vec<usize> = tok
        .encode(&receipt.prompt.rendered, SpecialTokens::Parse)
        .into_iter()
        .map(|t| t as usize)
        .collect();
    let bos_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);
    if let Some(bos) = bos_id {
        if tokens.first() != Some(&bos) {
            tokens.insert(0, bos);
        }
    }
    assert_eq!(
        tokens, receipt.prompt.token_ids,
        "tokenized prompt must match the pinned receipt IDs byte-for-byte"
    );

    let mut caches: Vec<KvCache> = (0..config.n_layers)
        .map(|_| KvCache::new(config.n_kv_heads, config.head_dim))
        .collect();
    let logits_per_pos = decoder.forward_batch(&tokens, 0, &mut caches);
    let mut last_logits = logits_per_pos.last().expect("non-empty prompt").clone();
    let top = top5(&last_logits);
    let top1 = top[0];

    let mut generated: Vec<usize> = Vec::new();
    for check in &receipt.checks {
        if !check.forbid_top1.is_empty() || !check.require_in_top5.is_empty() {
            assert!(
                !check.forbid_top1.contains(&top1),
                "{}: top-1={top1} is forbidden (was the early-EOS bug)",
                check.name
            );
            for &need in &check.require_in_top5 {
                assert!(
                    top.contains(&need),
                    "{}: expected token {need} in top-5 {top:?}",
                    check.name
                );
            }
            eprintln!("{}: ok top5={top:?}", check.name);
        }
        if let (Some(max_new), Some(want)) = (check.max_new_tokens, &check.decoded_substring) {
            if generated.is_empty() {
                for step in 0..max_new {
                    let next = argmax(&last_logits);
                    generated.push(next);
                    let pos = tokens.len() + step;
                    last_logits = decoder.forward_token(next, pos, &mut caches);
                }
            }
            let ids: Vec<u32> = generated.iter().map(|&x| x as u32).collect();
            let text = tok.decode(&ids);
            assert!(
                text.contains(want),
                "{}: decoded {text:?} missing {want:?}",
                check.name
            );
            if let Some(bad) = &check.forbid_decoded_substring {
                assert!(
                    !text.contains(bad),
                    "{}: decoded {text:?} still contains forbidden {bad:?}",
                    check.name
                );
            }
            eprintln!("{}: ok text={text:?}", check.name);
        }
    }
}

#[test]
fn receipt_json_files_parse() {
    // Always-on: every committed receipt must stay valid JSON against
    // this schema, even when the multi-GB checkpoint is absent.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/receipts");
    let mut found = 0usize;
    for entry in fs::read_dir(&dir).expect("receipts dir") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let raw = fs::read_to_string(&path).unwrap();
        let _: Receipt = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("receipt {path:?} failed to parse: {e}"));
        found += 1;
    }
    assert!(
        found >= 1,
        "expected at least one receipt JSON under {dir:?}"
    );
}
