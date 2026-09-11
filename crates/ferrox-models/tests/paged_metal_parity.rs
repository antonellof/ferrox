//! Paged KV must answer exactly what contiguous KV answers, on Metal.
//!
//! `ferrox verify` compares CPU against Metal over the CONTIGUOUS cache
//! only, so it cannot see a paged/GPU disagreement: neither of its two
//! children ever builds a `PagedKvStore`. This test closes that hole
//! from the other direction -- one process, one backend, the same
//! greedy continuation taken twice, once through `KvCache` and once
//! through `PagedKvCache`.
//!
//! It exists because the two together were wrong while each alone was
//! right. Metal prefill leaves K/V on the device and zero-fills the
//! host cache (`KvCache::advance_len`), which the contiguous decode
//! path knows and reads around; the paged prefill then copied those
//! zeros into the page store and decoded against a prompt the model
//! never saw.
//!
//! Requires:
//! - `cargo test -p ferrox-models --features metal --test paged_metal_parity -- --ignored --nocapture`
//! - Apple Silicon + Metal
//! - the GGUFs under `models/` (`FERROX_TEST_MODELS_DIR` to point elsewhere),
//!   or `FERROX_TEST_PAGED_PARITY_GGUF` for a single file

#![cfg(feature = "metal")]

use std::path::Path;

use ferrox_core::cache::{KvCache, PagedKvCache, SharedPagedKv};
use ferrox_gguf::ShardedGguf;
use ferrox_models::config::ModelConfig;
use ferrox_models::decoder::Decoder;

mod common;
use common::{argmax, model_dir, prompt_tokens};

const PROMPT: &str = "The capital of France is";
/// Long enough that prefill runs the batched Metal kernels rather than
/// degenerating to a handful of single-token steps.
const PROMPT_TOKENS: usize = 64;
const MAX_NEW_TOKENS: usize = 24;
/// Small on purpose: a block size that divides the prompt exactly would
/// hide every partial-block bug in the gather and the scatter.
const BLOCK_SIZE: usize = 16;

/// The three shapes the acceptance names: a dense model, an MoE model,
/// and a sliding-window model.
const MODELS: &[&str] = &[
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "olmoe-1b-7b-0924-q4_0.gguf",
    "gemma-2-2b-it-Q4_K_M.gguf",
];

fn greedy_contiguous(decoder: &Decoder, prompt: &[usize], eos: Option<usize>) -> Vec<usize> {
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
    let mut logits = decoder.forward_batch_last(prompt, 0, &mut caches);
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

fn greedy_paged(decoder: &Decoder, prompt: &[usize], eos: Option<usize>) -> Vec<usize> {
    let n_layers = decoder.layers.len();
    let positions = prompt.len() + MAX_NEW_TOKENS + BLOCK_SIZE;
    let stores = SharedPagedKv::new(
        n_layers,
        BLOCK_SIZE,
        positions.div_ceil(BLOCK_SIZE) + 2,
        decoder.config.n_kv_heads,
        decoder.config.head_dim,
    );
    let mut caches: Vec<PagedKvCache> = (0..n_layers).map(|_| PagedKvCache::new()).collect();
    let mut logits = decoder
        .forward_batch_last_paged(prompt, 0, &mut caches, &stores)
        .expect("store sized for the whole run");
    let mut out = Vec::with_capacity(MAX_NEW_TOKENS);
    for pos in (prompt.len()..).take(MAX_NEW_TOKENS) {
        let next = argmax(&logits);
        out.push(next);
        if Some(next) == eos {
            break;
        }
        logits = decoder
            .forward_token_paged(next, pos, &mut caches, &stores)
            .expect("store sized for the whole run");
    }
    out
}

/// Paging is a storage-layout choice. It may not change a single token,
/// on any backend, or a deployment's answers depend on whether a KV
/// pool happened to be configured.
#[test]
#[ignore = "needs Apple Metal GPU + the GGUFs under models/ (or FERROX_TEST_PAGED_PARITY_GGUF)"]
fn paged_kv_answers_exactly_what_contiguous_kv_answers_on_metal() {
    if ferrox_metal::gpu::probe().is_none() {
        eprintln!("skip: no Metal GPU detected");
        return;
    }
    // Honour a pre-set value so `FERROX_METAL=0 cargo test …` ablates to
    // the CPU pair, which is how the two halves of this bug were told
    // apart in the first place.
    for (k, v) in [("FERROX_METAL", "1"), ("FERROX_METAL_ATTN", "1")] {
        if std::env::var(k).is_err() {
            std::env::set_var(k, v);
        }
    }

    match std::env::var("FERROX_TEST_PAGED_PARITY_GGUF") {
        Ok(p) => check_one(Path::new(&p)),
        Err(_) => check_each_in_its_own_process(),
    }
}

/// One model per process, run as children of this one.
///
/// Not fussiness: two checkpoints loaded into a single process did NOT
/// come out the same as either alone on Metal. Loading Llama-3.2-1B and
/// then OLMoE in one run produced three different OLMoE continuations
/// across three runs, none of them the stable answer OLMoE gives on its
/// own, while paged and contiguous agreed with each other every time.
/// That was process-global Metal state surviving a dropped `Decoder`:
/// resident-buffer caches keyed on a host address the dropped model's
/// allocator had already handed to the next one (GitHub issue #180),
/// now fixed and held by `model_swap_isolation`. The isolation stays,
/// because what this suite measures should not depend on that fix
/// holding.
///
/// The re-invocation is the shape `ferrox verify` already uses for the
/// same reason: some state is per process, so isolate per process.
fn check_each_in_its_own_process() {
    let exe = std::env::current_exe().expect("test binary path");
    let mut ran = 0usize;
    let mut failed: Vec<String> = Vec::new();
    for name in MODELS {
        let path = model_dir().join(name);
        if !path.exists() {
            eprintln!("skip: {} missing", path.display());
            continue;
        }
        let out = std::process::Command::new(&exe)
            .args([
                "--ignored",
                "--exact",
                "--nocapture",
                "paged_kv_answers_exactly_what_contiguous_kv_answers_on_metal",
            ])
            .env("FERROX_TEST_PAGED_PARITY_GGUF", &path)
            .output()
            .expect("spawning the per-model child");
        eprint!("{}", String::from_utf8_lossy(&out.stderr));
        if !out.status.success() {
            failed.push(name.to_string());
        }
        ran += 1;
    }
    assert!(ran > 0, "no model available to check");
    assert!(failed.is_empty(), "paged KV changed the answer: {failed:?}");
}

fn check_one(path: &Path) {
    let file = ShardedGguf::open(path).expect("open gguf");
    let config = ModelConfig::from_gguf(&file).expect("model config");
    let eos = file
        .metadata_u64("tokenizer.ggml.eos_token_id")
        .map(|v| v as usize);
    let prompt = prompt_tokens(&file, PROMPT, PROMPT_TOKENS);
    let decoder = Decoder::from_gguf(path, config).expect("decoder");

    let want = greedy_contiguous(&decoder, &prompt, eos);
    let got = greedy_paged(&decoder, &prompt, eos);
    eprintln!(
        "{}: contiguous {want:?}\n{}: paged      {got:?}",
        path.display(),
        path.display()
    );
    assert_eq!(want, got, "{}: paged KV changed the answer", path.display());
}
