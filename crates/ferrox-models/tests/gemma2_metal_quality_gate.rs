//! Quality gate: Gemma-2 Metal greedy must contain `Paris` on the capitals
//! prompt (CPU always; Metal when a GPU is present).
//!
//! Root cause of prior Metal `*` / BOS-loop garbage: Concurrent encoder +
//! sandwich post-norms in `launch_decode_dense_stack` — fixed via serial
//! encode + eager residuals for sandwich layers.
//!
//! ```text
//! cargo test -p ferrox-models --features metal --test gemma2_metal_quality_gate -- --ignored
//! ```

use std::path::{Path, PathBuf};

use ferrox_core::cache::KvCache;
use ferrox_gguf::ShardedGguf;
use ferrox_models::config::ModelConfig;
use ferrox_models::decoder::Decoder;
use ferrox_models::tokenizer::{should_add_bos_token, GgufSpmTokenizer, SpecialTokens};

const PROMPT: &str = "The capital of France is";
const MAX_NEW: usize = 24;

fn gguf_path() -> PathBuf {
    if let Ok(p) = std::env::var("FERROX_TEST_GEMMA2_GGUF") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma-2-2b-it-Q4_K_M.gguf")
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn greedy_continuation(decoder: &Decoder, tok: &GgufSpmTokenizer, file: &ShardedGguf) -> String {
    let mut tokens: Vec<usize> = tok
        .encode(PROMPT, SpecialTokens::Parse)
        .into_iter()
        .map(|t| t as usize)
        .collect();
    if should_add_bos_token(file) {
        if let Some(bos) = file.metadata_u64("tokenizer.ggml.bos_token_id") {
            let bos = bos as usize;
            if tokens.first().copied() != Some(bos) {
                tokens.insert(0, bos);
            }
        }
    }
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
    let mut logits = decoder.forward_batch_last(&tokens, 0, &mut caches);
    let mut text = String::new();
    for _ in 0..MAX_NEW {
        let next = argmax(&logits);
        tokens.push(next);
        text.push_str(&tok.decode(&[next as u32]));
        if next as u32
            == file
                .metadata_u64("tokenizer.ggml.eos_token_id")
                .unwrap_or(u64::MAX) as u32
        {
            break;
        }
        logits = decoder.forward_token(next, tokens.len() - 1, &mut caches);
    }
    text
}

#[test]
#[ignore = "needs models/gemma-2-2b-it-Q4_K_M.gguf"]
fn gemma2_cpu_greedy_contains_paris() {
    let path = gguf_path();
    if !path.exists() {
        eprintln!("skip: missing {}", path.display());
        return;
    }
    std::env::set_var("FERROX_METAL", "0");
    std::env::set_var("FERROX_METAL_ATTN", "0");
    let file = ShardedGguf::open(&path).expect("open");
    let config = ModelConfig::from_gguf(&file).expect("config");
    let tok = GgufSpmTokenizer::from_gguf(&file).expect("tokenizer");
    let decoder = Decoder::from_gguf(&path, config).expect("load");
    let text = greedy_continuation(&decoder, &tok, &file);
    eprintln!("gemma2 cpu: {text:?}");
    assert!(
        text.to_lowercase().contains("paris"),
        "CPU Gemma-2 must answer Paris; got {text:?}"
    );
}

#[test]
#[ignore = "needs Metal GPU + models/gemma-2-2b-it-Q4_K_M.gguf"]
fn gemma2_metal_greedy_contains_paris() {
    #[cfg(not(feature = "metal"))]
    eprintln!("skip: build with --features metal");
    #[cfg(feature = "metal")]
    {
        let path = gguf_path();
        if !path.exists() {
            eprintln!("skip: missing {}", path.display());
            return;
        }
        if ferrox_metal::gpu::probe().is_none() {
            eprintln!("skip: no Metal GPU");
            return;
        }
        std::env::set_var("FERROX_METAL", "1");
        std::env::set_var("FERROX_METAL_ATTN", "1");
        // Host lm_head + host argmax, so this gate measures the attention
        // and FFN stack rather than the GPU argmax fold.
        ferrox_models::set_metal_greedy_argmax(false);
        let file = ShardedGguf::open(&path).expect("open");
        let config = ModelConfig::from_gguf(&file).expect("config");
        let tok = GgufSpmTokenizer::from_gguf(&file).expect("tokenizer");
        let decoder = Decoder::from_gguf(&path, config).expect("load");
        let text = greedy_continuation(&decoder, &tok, &file);
        eprintln!("gemma2 metal: {text:?}");
        assert!(
            text.to_lowercase().contains("paris"),
            "Metal Gemma-2 must answer Paris; got {text:?}"
        );
    }
}
