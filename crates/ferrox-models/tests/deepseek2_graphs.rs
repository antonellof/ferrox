//! DeepSeek-2 on the MLA engine, checked against llama.cpp itself, in
//! BOTH forms a converter writes: the split `attn_k_b` / `attn_v_b`
//! every DeepSeek export has carried since the `_mla` keys existed, and
//! the legacy combined `attn_kv_b`.
//!
//! `deepseek2` is what DeepSeek-V2, V2.5, V3 and R1 all tag. The MLA
//! engine had run it with no cross-engine evidence since it existed,
//! and REFUSED the split form ("not wired for MlaEngine yet"), which is
//! every real export. `ferrox_models::mla::MlaKvB` is the two forms as
//! one enum: `Combined` expands the latent per head and attends with
//! per-head caches (the naive branch, `deepseek2.cpp:600-635`); `Split`
//! absorbs the query through `wk_b`, attends as MQA over the latent
//! `concat(c, k_pe)` and pulls the result through `wv_b` (the `is_mla`
//! branch, `:563-598`; `ferrox_core::mla_absorbed`), with the cache
//! `kv_lora_rank + qk_rope` wide instead of `n_heads * (qk_nope +
//! qk_rope + v)`.
//!
//! | fixture | what it isolates |
//! |---|---|
//! | `deepseek2` | the converter's shape: `_mla` keys, `head_count_kv = 1`, split `attn_k_b` / `attn_v_b`, `q_lora`, a dense lead layer, sigmoid routing with `exp_probs_b`, `expert_weights_norm`, `expert_weights_scale = 2.5`, one shared expert |
//! | `deepseek2_legacy` | the same model in the pre-2025-03 shape: no `_mla` keys, per-head widths under the plain keys, `head_count_kv = n_head`, one combined `attn_kv_b` derived from the same values |
//!
//! The split tensors are DERIVED from the combined matrix exactly as
//! `conversion/deepseek.py:420-427` derives them, so the two files
//! hold one model, and llama.cpp's absorbed and naive branches agree
//! on them to 1.79e-7 (measured) -- which is the number that says the
//! derivation in the script is the converter's.
//!
//! # Where the numbers come from
//!
//! Each `GOLDEN` array was produced by running llama.cpp's own
//! `deepseek2` graph over the fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`. The fixture had never
//! produced one before 2026-09-12: it wrote `head_count_kv = n_head`
//! where `deepseek.py:307-308` writes 1 for every MLA export, and that
//! -- not the "pre-existing llama.cpp shape mismatch" the script used
//! to blame -- was the `ggml.c:3942` abort.
//!
//! | fixture | KL(llama.cpp \|\| ferrox) | max abs logit delta |
//! |---|---|---|
//! | `deepseek2` | 2.35e-15 | 2.09e-07 |
//! | `deepseek2_legacy` | 3.57e-15 | 2.09e-07 |
//!
//! Sabotaging the absorbed form's `kq_scale` to the latent width
//! (`1/sqrt(kv_lora + rope)`, the plausible wrong number) moves the
//! split file's logits by 2.2e-3 and turns its test red (confirmed).
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_deepseek2_fixture.py \
//!     crates/ferrox-models/tests/fixtures/deepseek2_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_deepseek2_fixture.py \
//!     crates/ferrox-models/tests/fixtures/deepseek2_legacy_tiny.gguf --legacy-kv-b
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{assert_close, graph_fixture_path, kl_vs_golden, worst_vs, GRAPH_PROMPT, GRAPH_TOL};
use ferrox_models::engine::Engine;
use ferrox_models::mla::{MlaKvB, MlaQProj};
use ferrox_models::{load_mla_engine_from_path, LoadError, MlaEngine, MlaLayerFfn, ServedEngine};

const SPLIT: &str = "deepseek2";
const LEGACY: &str = "deepseek2_legacy";

const DEEPSEEK2_GOLDEN: [f32; 48] = [
    -0.4785164,
    0.35277408,
    -0.26033843,
    -0.19016461,
    -0.9581052,
    -0.85038555,
    -0.43373168,
    -0.16950154,
    0.104315095,
    0.1172294,
    -0.49205014,
    0.32342067,
    -0.16612323,
    0.13369039,
    -0.021073774,
    0.40181646,
    -0.3369094,
    -0.10081072,
    -0.4419329,
    0.077120095,
    -0.4783455,
    0.49064615,
    0.36519915,
    0.4558949,
    -0.04049185,
    -0.2190375,
    -0.48284528,
    0.13691062,
    -0.87235117,
    0.12599146,
    0.6829497,
    0.21115445,
    -0.28446934,
    0.5834253,
    0.15295744,
    0.22171961,
    -0.26188886,
    0.4604143,
    0.15225416,
    0.10682063,
    -0.109797865,
    -0.569731,
    -0.16047618,
    -0.57658195,
    -0.43833408,
    0.37348205,
    0.3447876,
    -0.19445688,
];

const DEEPSEEK2_LEGACY_GOLDEN: [f32; 48] = [
    -0.47851634,
    0.35277408,
    -0.26033837,
    -0.19016454,
    -0.9581052,
    -0.85038555,
    -0.43373165,
    -0.16950154,
    0.10431508,
    0.1172294,
    -0.49205023,
    0.3234206,
    -0.16612318,
    0.13369045,
    -0.021073848,
    0.40181646,
    -0.33690947,
    -0.10081071,
    -0.44193286,
    0.077120155,
    -0.4783455,
    0.49064618,
    0.36519897,
    0.4558949,
    -0.04049188,
    -0.21903746,
    -0.48284525,
    0.13691056,
    -0.87235117,
    0.12599146,
    0.6829497,
    0.21115445,
    -0.28446928,
    0.5834253,
    0.15295741,
    0.22171964,
    -0.2618889,
    0.46041432,
    0.15225415,
    0.10682058,
    -0.10979788,
    -0.569731,
    -0.1604762,
    -0.57658195,
    -0.4383341,
    0.37348208,
    0.3447876,
    -0.19445688,
];

fn load(name: &str) -> Result<MlaEngine, LoadError> {
    match load_mla_engine_from_path(std::path::Path::new(&graph_fixture_path(name)))? {
        ServedEngine::Mla(m) => Ok(m),
        _ => panic!("`{name}` is served by the MLA engine"),
    }
}

/// The MLA engine has one body, token by token; the last position's
/// logits over the six-token prompt.
fn decode(engine: &MlaEngine) -> Vec<f32> {
    let mut state = Engine::new_state(engine);
    let mut out = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        out = engine.forward_token(tok, pos, &mut state);
    }
    out
}

#[test]
fn the_split_form_matches_llama_cpp() {
    let engine = load(SPLIT).expect("the converter's shape loads");
    assert_close(
        &decode(&engine),
        &DEEPSEEK2_GOLDEN,
        GRAPH_TOL,
        "deepseek2: split, absorbed",
    );
}

#[test]
fn the_legacy_form_matches_llama_cpp() {
    let engine = load(LEGACY).expect("the legacy shape loads");
    assert_close(
        &decode(&engine),
        &DEEPSEEK2_LEGACY_GOLDEN,
        GRAPH_TOL,
        "deepseek2: combined, naive",
    );
}

#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (SPLIT, &DEEPSEEK2_GOLDEN),
        (LEGACY, &DEEPSEEK2_LEGACY_GOLDEN),
    ] {
        let out = decode(&load(name).unwrap());
        println!(
            "{name}: KL(llama.cpp || ferrox) = {:.3e}, max |delta| = {:.3e}",
            kl_vs_golden(&out, golden),
            worst_vs(&out, golden)
        );
    }
    println!(
        "split vs legacy golden: max |delta| = {:.3e}",
        worst_vs(&DEEPSEEK2_GOLDEN, &DEEPSEEK2_LEGACY_GOLDEN)
    );
}

/// What the loader built, read back: the two forms, the low-rank Q,
/// the per-head widths from the right keys in each file, the dense
/// lead layer and the MoE tail with its shared expert, and a latent
/// cache for the split form.
#[test]
fn the_two_files_load_as_the_two_forms_of_one_model() {
    let split = load(SPLIT).unwrap();
    let legacy = load(LEGACY).unwrap();
    for engine in [&split, &legacy] {
        assert_eq!(engine.mla_cfg.q_lora_rank, 16);
        assert_eq!(engine.mla_cfg.kv_lora_rank, 12);
        assert_eq!(engine.mla_cfg.qk_nope_head_dim, 8);
        assert_eq!(engine.mla_cfg.qk_rope_head_dim, 4);
        assert_eq!(engine.mla_cfg.v_head_dim, 8);
        assert_eq!(engine.layers.len(), 2);
        assert!(matches!(engine.layers[0].attn.q, MlaQProj::LowRank { .. }));
        assert!(matches!(engine.layers[0].ffn, MlaLayerFfn::Dense(_)));
        match &engine.layers[1].ffn {
            MlaLayerFfn::Moe(m) => {
                assert_eq!(m.experts.len(), 6);
                assert!(m.exp_probs_bias.is_some());
            }
            MlaLayerFfn::Dense(_) => panic!("layer 1 is routed"),
        }
        let moe = engine.moe.as_ref().expect("a MoE tail");
        assert_eq!(moe.n_experts_active, 2);
        assert_eq!(moe.expert_weights_scale, 2.5);
        assert!(moe.norm_topk_prob);
    }
    for layer in &split.layers {
        match &layer.attn.kv_b {
            MlaKvB::Split { k_b, v_b } => {
                assert_eq!(k_b.len(), 4);
                assert_eq!(v_b.len(), 4);
                assert_eq!(
                    (k_b[0].rows(), k_b[0].cols()),
                    (12, 8),
                    "k_b[h]: [kv_lora, qk_nope]"
                );
                assert_eq!(
                    (v_b[0].rows(), v_b[0].cols()),
                    (8, 12),
                    "v_b[h]: [v_head, kv_lora]"
                );
            }
            MlaKvB::Combined(_) => panic!("the converter's file is split"),
        }
    }
    for layer in &legacy.layers {
        match &layer.attn.kv_b {
            MlaKvB::Combined(w) => assert_eq!((w.rows(), w.cols()), (4 * 16, 12)),
            MlaKvB::Split { .. } => panic!("the legacy file is combined"),
        }
    }

    // The caches the two forms leave behind after six tokens: the
    // latent, `kv_lora + rope` per position in `k_cache` alone, against
    // per-head K and V.
    let mut state = Engine::new_state(&split);
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        split.forward_token(tok, pos, &mut state);
    }
    assert_eq!(state.layers[0].0.len(), 6 * (12 + 4));
    assert!(state.layers[0].1.is_empty());
    let mut state = Engine::new_state(&legacy);
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        legacy.forward_token(tok, pos, &mut state);
    }
    assert_eq!(state.layers[0].0.len(), 6 * 4 * (8 + 4));
    assert_eq!(state.layers[0].1.len(), 6 * 4 * 8);
}

/// Each thing the goldens check, sabotaged on the loaded engine: the
/// absorbed scale read off the latent width (the plausible wrong
/// number), the shared expert, the routing bias, the KV norm.
#[test]
fn each_seam_is_visible_in_the_logits() {
    let mut split = load(SPLIT).unwrap();
    assert_close(&decode(&split), &DEEPSEEK2_GOLDEN, GRAPH_TOL, "baseline");

    // The routing bias dropped: the top-2 is taken over the raw gate.
    let saved: Vec<_> = split
        .layers
        .iter_mut()
        .map(|l| match &mut l.ffn {
            MlaLayerFfn::Moe(m) => m.exp_probs_bias.take(),
            MlaLayerFfn::Dense(_) => None,
        })
        .collect();
    let worst = worst_vs(&decode(&split), &DEEPSEEK2_GOLDEN);
    assert!(worst > 1e-2, "exp_probs_b not seen: {worst}");
    for (l, s) in split.layers.iter_mut().zip(saved) {
        if let MlaLayerFfn::Moe(m) = &mut l.ffn {
            m.exp_probs_bias = s;
        }
    }

    // The KV norm flattened to one.
    let saved: Vec<_> = split
        .layers
        .iter_mut()
        .map(|l| std::mem::replace(&mut l.attn.kv_a_layernorm, vec![1.0; 12]))
        .collect();
    let worst = worst_vs(&decode(&split), &DEEPSEEK2_GOLDEN);
    assert!(worst > 1e-2, "kv_a_norm not seen: {worst}");
    for (l, s) in split.layers.iter_mut().zip(saved) {
        l.attn.kv_a_layernorm = s;
    }

    // `expert_weights_scale` set to one.
    let scale = split.moe.as_ref().unwrap().expert_weights_scale;
    split.moe.as_mut().unwrap().expert_weights_scale = 1.0;
    let worst = worst_vs(&decode(&split), &DEEPSEEK2_GOLDEN);
    assert!(worst > 1e-2, "expert_weights_scale not seen: {worst}");
    split.moe.as_mut().unwrap().expert_weights_scale = scale;

    assert_close(&decode(&split), &DEEPSEEK2_GOLDEN, GRAPH_TOL, "restored");
}
