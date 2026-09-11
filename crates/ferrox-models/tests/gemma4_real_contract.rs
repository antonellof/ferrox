//! Gemma-4 hparam / tensor contract, checked against the real
//! `models/gemma-4-E2B-it-Q4_K_M.gguf` rather than a fixture.
//!
//! Why this file exists. Twice in this audit a dedicated loader was
//! found demanding a hyper-parameter the architecture is not supposed to
//! have — `glm4moe` sent to an MLA loader that wants `q_lora_rank`, and
//! `mla_gguf_loader` requiring `attention.qk_nope_head_dim`, a string
//! that is in neither `LLM_KV_NAMES` nor `gguf-py`. Both stayed
//! invisible because the loaders' own tests build fixtures that supply
//! the invented keys. `gemma4_gguf_loader`'s in-crate test has the same
//! shape, and it additionally pins `shared_kv_layers = 0` and ships no
//! `rope_freqs.weight`, so the two hardest parts of the Gemma-4 graph —
//! KV reuse and proportional RoPE on the full-attention layers — are
//! exercised by nothing. A real checkpoint cannot lie about its own key
//! set, which is why this test reads one.
//!
//! **There is no cross-engine evidence here and this file does not claim
//! any.** The Homebrew `libllama` on this machine predates `gemma4` and
//! refuses the checkpoint with "unknown model architecture: 'gemma4'",
//! so `scripts/gptoss_reference_logits.cpp` cannot produce reference
//! logits for it and `ferrox parity` skips it. What is checked here is
//! the contract — that every key ferrox requires exists, that every
//! value ferrox derives matches the derivation in
//! `.scratch/llama.cpp/src/models/gemma4.cpp`, and that the model then
//! decodes a fact correctly end to end. That is weaker than a pinned
//! logit comparison and must not be read as one. `gemma4` is
//! `DedicatedOnly` and stays off `AUDITED_GENERIC_GQA`.
//!
//! Run: `cargo test -p ferrox-models --test gemma4_real_contract -- --ignored`

use std::path::{Path, PathBuf};

use ferrox_gguf::{GgufFile, TensorSource};
use ferrox_models::engine::Engine;
use ferrox_models::gemma4_gguf_loader::{load_gemma4_engine, read_gemma4_hparams};
use ferrox_models::tokenizer::{should_add_bos_token, GgufBpeTokenizer, SpecialTokens};

const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 8;

fn gemma4_path() -> PathBuf {
    if let Ok(p) = std::env::var("FERROX_TEST_GEMMA4_GGUF") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/gemma-4-E2B-it-Q4_K_M.gguf")
}

fn open() -> Option<GgufFile> {
    let path = gemma4_path();
    if !path.exists() {
        eprintln!("skip: Gemma-4 GGUF missing at {}", path.display());
        return None;
    }
    Some(GgufFile::open(&path).expect("open GGUF"))
}

/// Every key `read_gemma4_hparams` treats as REQUIRED is actually in the
/// file.
///
/// This is the assertion that would have caught the deepseek2 bug: there,
/// two of the required keys are HF `config.json` field names that no
/// converter has ever written into a GGUF. Listing them against a real
/// checkpoint is the only check that cannot be satisfied by inventing
/// the key in a fixture.
#[test]
#[ignore = "needs models/gemma-4-E2B-it-Q4_K_M.gguf (or FERROX_TEST_GEMMA4_GGUF)"]
fn every_hparam_the_gemma4_loader_requires_exists_in_a_real_checkpoint() {
    let Some(file) = open() else { return };
    assert_eq!(file.metadata_str("general.architecture"), Some("gemma4"));
    for key in [
        // read_gemma4_hparams: meta_u64 / unconditional `?`.
        "gemma4.block_count",
        "gemma4.embedding_length",
        "gemma4.feed_forward_length",
        "gemma4.attention.head_count",
        "gemma4.attention.key_length",
        "gemma4.attention.sliding_window",
        "gemma4.attention.sliding_window_pattern",
    ] {
        assert!(
            file.metadata(key).is_some(),
            "`{key}` is required by read_gemma4_hparams but absent from a real \
             Gemma-4 checkpoint — that is the glm4moe/deepseek2 bug shape"
        );
    }
    // And the optional ones this particular checkpoint does carry, so
    // the defaults below are never silently in play.
    for key in [
        "gemma4.attention.head_count_kv",
        "gemma4.attention.key_length_swa",
        "gemma4.attention.shared_kv_layers",
        "gemma4.embedding_length_per_layer_input",
        "gemma4.final_logit_softcapping",
        "gemma4.attention.layer_norm_rms_epsilon",
        "gemma4.rope.freq_base",
        "gemma4.rope.freq_base_swa",
    ] {
        assert!(file.metadata(key).is_some(), "expected `{key}` in E2B");
    }
}

/// Every value ferrox derives matches llama.cpp's own derivation for
/// this file, with the line that defines it.
#[test]
#[ignore = "needs models/gemma-4-E2B-it-Q4_K_M.gguf (or FERROX_TEST_GEMMA4_GGUF)"]
fn gemma4_hparams_match_llama_cpps_derivation() {
    let Some(file) = open() else { return };
    let hp = read_gemma4_hparams(&file).expect("read_gemma4_hparams");

    assert_eq!(hp.n_layer, 35, "E2B is the 35-layer case (gemma4.cpp:25)");
    assert_eq!(hp.hidden_dim, 1536);
    assert_eq!(hp.n_heads, 8);
    assert_eq!(hp.n_kv_heads, 1);

    // gemma4.cpp:18-19 reads KEY_LENGTH_SWA / VALUE_LENGTH_SWA as
    // REQUIRED, and :39-41 asserts k == v on both pairs. The split is
    // real: full layers are 512-wide heads, SWA layers 256.
    assert_eq!(hp.head_dim_full, 512);
    assert_eq!(hp.head_dim_swa, 256);
    assert_eq!(
        file.metadata_u64("gemma4.attention.value_length"),
        Some(hp.head_dim_full as u64),
        "gemma4.cpp:38 requires n_embd_head_k == n_embd_head_v"
    );
    assert_eq!(
        file.metadata_u64("gemma4.attention.value_length_swa"),
        Some(hp.head_dim_swa as u64),
        "gemma4.cpp:40 requires the same of the SWA pair"
    );

    // gemma4.cpp:9 — n_layer_kv_from_start = n_layer_all - shared_kv.
    assert_eq!(
        file.metadata_u64("gemma4.attention.shared_kv_layers"),
        Some(20)
    );
    assert_eq!(hp.n_layer_kv_from_start, 35 - 20);
    assert!(hp.has_kv(14) && !hp.has_kv(15), "layers 15.. reuse KV");
    // llama-model.cpp:2312-2313 — the reused layer is the last KV layer,
    // minus one more for an SWA layer.
    for il in 15..35 {
        let want = hp.n_layer_kv_from_start - if hp.is_swa_layer(il) { 2 } else { 1 };
        assert_eq!(hp.kv_reuse_layer(il), want);
        assert!(
            hp.has_kv(hp.kv_reuse_layer(il)),
            "reuse target must have KV"
        );
    }

    // The alternating pattern is a per-layer bool ARRAY here, not a
    // period: gemma4.cpp:5 uses `get_key_or_arr(..., is_swa_impl,
    // n_layer)`, which is why `capability::default_swa_layout` has no
    // gemma4 entry to give.
    assert_eq!(hp.is_swa.len(), 35);
    assert!(
        hp.is_swa.iter().any(|&b| b) && hp.is_swa.iter().any(|&b| !b),
        "both kinds of layer must be present or the split head dims are dead"
    );

    // gemma4.cpp:11 — Gemma-4 sets f_attention_scale = 1.0f, i.e. NO
    // 1/sqrt(head_dim). Getting this wrong is a factor of 22.6 on every
    // full-attention score.
    assert_eq!(hp.attention_scale, 1.0);

    // Per-layer FFN widths: E2B widens from 6144 to 12288 partway down,
    // so a scalar `feed_forward_length` read would be wrong for 20 of
    // the 35 layers.
    assert_eq!(hp.ffn_dims.len(), 35);
    assert!(
        hp.ffn_dims
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1,
        "E2B's feed_forward_length really is a per-layer array"
    );

    assert_eq!(hp.embd_per_layer, 256);
    assert_eq!(hp.final_logit_softcap, Some(30.0));
    assert_eq!(hp.rope_theta, 1_000_000.0);
    assert_eq!(hp.rope_theta_swa, 10_000.0);
    assert_eq!(hp.sliding_window, 512);
}

/// The loader reads the two tensors this checkpoint has and the
/// in-crate synthetic test does not: `rope_freqs.weight` and the shared
/// KV layout.
#[test]
#[ignore = "needs models/gemma-4-E2B-it-Q4_K_M.gguf (or FERROX_TEST_GEMMA4_GGUF)"]
fn gemma4_loads_and_covers_what_the_synthetic_fixture_cannot() {
    let Some(file) = open() else { return };
    // gemma4.cpp:85-88: full-attention layers take proportional RoPE
    // from a shared `rope_freqs`; SWA layers pass nullptr.
    assert!(
        file.find_tensor("rope_freqs.weight").is_some(),
        "E2B carries rope_freqs; ignoring it mis-rotates every full-attn layer"
    );
    let engine = load_gemma4_engine(&file).expect("load_gemma4_engine on a real Gemma-4");

    let freqs = engine
        .weights
        .rope_freqs
        .as_ref()
        .expect("rope_freqs must reach the engine, not be dropped at load");
    assert_eq!(
        freqs.len(),
        engine.hp.head_dim_full / 2,
        "gemma4.cpp:86 sizes it {{n_embd_head/2}}"
    );
    assert!(
        freqs.iter().any(|&f| (f - 1.0).abs() > 1e-6),
        "an all-ones rope_freqs would make this test vacuous"
    );

    // `layer_output_scale` is TENSOR_NOT_REQUIRED in llama.cpp
    // (gemma4.cpp:82) but present in every block of this checkpoint.
    assert!(engine.weights.layers.iter().all(|l| l.out_scale.is_some()));

    // KV reuse is live: the sharing layers must hold no cache of their
    // own, or the engine is silently allocating 20 caches llama.cpp does
    // not have.
    let state = engine.new_state();
    let with_kv = (0..engine.hp.n_layer)
        .filter(|&il| engine.hp.has_kv(il))
        .count();
    assert_eq!(with_kv, 15);
    assert_eq!(state.kv.iter().filter(|c| c.is_some()).count(), 15);
}

/// End-to-end: the graph is right enough to answer a fact.
///
/// This is a behavioural check, not a parity check — see the module
/// docs. It is here because every individual assertion above can pass
/// while the graph is still assembled wrongly, and a Gemma-4 with the
/// attention scale, the per-layer embeddings or the KV reuse wrong does
/// not produce `Paris`, it produces noise.
#[test]
#[ignore = "needs models/gemma-4-E2B-it-Q4_K_M.gguf (or FERROX_TEST_GEMMA4_GGUF)"]
fn gemma4_greedy_decode_answers_paris() {
    let Some(file) = open() else { return };
    let tok = GgufBpeTokenizer::from_gguf(&file).expect("tokenizer");
    let engine = load_gemma4_engine(&file).expect("load");

    let mut tokens: Vec<usize> = Vec::new();
    if should_add_bos_token(&file) {
        tokens.push(file.metadata_u64("tokenizer.ggml.bos_token_id").unwrap() as usize);
    }
    tokens.extend(
        tok.encode(PROMPT, SpecialTokens::Parse)
            .into_iter()
            .map(|t| t as usize),
    );

    let mut state = engine.new_state();
    let mut logits = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        logits = engine.forward_token(t, pos, &mut state);
    }
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "non-finite logits out of the Gemma-4 graph"
    );

    let mut out = Vec::new();
    for _ in 0..MAX_NEW_TOKENS {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        out.push(next as u32);
        logits = engine.forward_token(next, tokens.len() + out.len() - 1, &mut state);
    }
    let text = tok.decode(&out);
    eprintln!("gemma4 continuation: {text:?}");
    assert!(
        text.contains("Paris"),
        "greedy continuation of {PROMPT:?} should name Paris, got {text:?}"
    );
}
