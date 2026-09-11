//! Greedy Metal regression for SmolLM2-135M Q8_0: prompt
//! `The capital of France is` must continue with `Paris`.
//!
//! Requires:
//! - `cargo test -p ferrox-models --features metal --test smollm2_metal_paris -- --ignored`
//! - Apple Silicon + Metal
//! - `models/hf_test/SmolLM2-135M-Instruct-Q8_0.gguf` (or `FERROX_TEST_SMOLLM2_GGUF`)
//!
//! Mirrors the llama.cpp parity check documented in `ferrox-moe` (Paris vs
//! wrong continuation when MoE/Metal routing regresses).

#![cfg(feature = "metal")]

use std::path::{Path, PathBuf};

use ferrox_core::cache::KvCache;
use ferrox_gguf::ShardedGguf;
use ferrox_models::config::ModelConfig;
use ferrox_models::decoder::Decoder;
use ferrox_models::tokenizer::{GgufBpeTokenizer, SpecialTokens};

const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 16;

fn smollm2_gguf_path() -> PathBuf {
    if let Ok(p) = std::env::var("FERROX_TEST_SMOLLM2_GGUF") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/hf_test/SmolLM2-135M-Instruct-Q8_0.gguf")
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
#[ignore = "needs Apple Metal GPU + models/hf_test/SmolLM2-135M-Instruct-Q8_0.gguf (or FERROX_TEST_SMOLLM2_GGUF)"]
fn smollm2_metal_greedy_paris_regression() {
    let path = smollm2_gguf_path();
    if !path.exists() {
        eprintln!("skip: SmolLM2 GGUF missing at {}", path.display());
        return;
    }

    std::env::set_var("FERROX_METAL", "1");

    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("skip: Metal regression requires macOS");
        return;
    }

    if ferrox_metal::gpu::probe().is_none() {
        eprintln!("skip: no Metal GPU detected");
        return;
    }

    let file = ShardedGguf::open(&path).expect("open GGUF");
    let config = ModelConfig::from_gguf(&file).expect("from_gguf");
    let tok = GgufBpeTokenizer::from_gguf(&file).expect("tokenizer");
    let decoder = Decoder::from_gguf(&path, config.clone()).expect("load decoder");

    let mut tokens: Vec<usize> = tok
        .encode(PROMPT, SpecialTokens::Parse)
        .into_iter()
        .map(|t| t as usize)
        .collect();

    let mut caches: Vec<KvCache> = config.new_kv_caches();

    let logits_per_pos = decoder.forward_batch(&tokens, 0, &mut caches);
    let mut last_logits = logits_per_pos.last().expect("non-empty prompt").clone();

    let mut generated = Vec::with_capacity(MAX_NEW_TOKENS);
    for step in 0..MAX_NEW_TOKENS {
        let next = argmax(&last_logits);
        generated.push(next);
        let pos = tokens.len() + step;
        last_logits = decoder.forward_token(next, pos, &mut caches);
    }

    tokens.extend(&generated);
    let ids: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();
    let text = tok.decode(&ids);
    eprintln!("decoded={text:?}");

    assert!(
        text.to_ascii_lowercase().contains("paris"),
        "expected Paris in greedy Metal continuation, got {text:?}"
    );
}
