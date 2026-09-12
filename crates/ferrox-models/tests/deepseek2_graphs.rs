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
//! | `deepseek2_yarn_v2` | the split file plus YaRN as a DeepSeek-V2 export declares it: factor 4, `original_context_length = 16`, `yarn_log_multiplier = 0.1 * 0.707` (`ferrox_models::mla_yarn`) |
//! | `deepseek2_yarn_v3` | the same with `yarn_log_multiplier = 0.1 * 1.0`, DeepSeek-V3 / R1's value; its golden differs from `_v2`'s by 1.5e-3, which is `mscale^2` in `kq_scale` |
//! | `deepseek2_yarn_legacy` | the combined file plus V2's YaRN: the naive branch under the same rewrite; agrees with `_v2` to 1.78e-7 |
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
//! | `deepseek2_yarn_v2` | 2.98e-15 | 2.38e-07 |
//! | `deepseek2_yarn_v3` | 4.42e-15 | 2.38e-07 |
//! | `deepseek2_yarn_legacy` | 2.94e-15 | 2.38e-07 |
//!
//! YaRN moves the plain file's logits by 3.6e-3 (measured on the
//! goldens), so a YaRN dropped or half-applied is visible at the
//! tolerance; libllama's log line for both YaRN files reads `setting
//! new yarn_attn_factor = 1.0000`, which is `mla_yarn`'s step 2 coming
//! out at 1 for the real DeepSeek shapes, leaving the whole effect in
//! the frequency rewrite and `kq_scale`. Sabotaging the `/ 0.1` that
//! `deepseek2.cpp:37` applies to the key turns both YaRN tests red
//! (confirmed).
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
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_deepseek2_fixture.py \
//!     crates/ferrox-models/tests/fixtures/deepseek2_yarn_v2_tiny.gguf --yarn 0.707
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_deepseek2_fixture.py \
//!     crates/ferrox-models/tests/fixtures/deepseek2_yarn_v3_tiny.gguf --yarn 1.0
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_deepseek2_fixture.py \
//!     crates/ferrox-models/tests/fixtures/deepseek2_yarn_legacy_tiny.gguf --yarn 0.707 --legacy-kv-b
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/<name>_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{assert_close, graph_fixture_path, kl_vs_golden, worst_vs, GRAPH_PROMPT, GRAPH_TOL};
use ferrox_models::engine::Engine;
use ferrox_models::mla::{MlaKvB, MlaQProj};
use ferrox_models::{load_mla_engine_from_path, LoadError, MlaEngine, MlaLayerFfn, ServedEngine};

const SPLIT: &str = "deepseek2";
const LEGACY: &str = "deepseek2_legacy";
const YARN_V2: &str = "deepseek2_yarn_v2";
const YARN_V3: &str = "deepseek2_yarn_v3";
const YARN_LEGACY: &str = "deepseek2_yarn_legacy";

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

const DEEPSEEK2_YARN_V2_GOLDEN: [f32; 48] = [
    -0.47546825,
    0.35450304,
    -0.25883466,
    -0.18994278,
    -0.9571208,
    -0.8507633,
    -0.4327122,
    -0.17080505,
    0.102811925,
    0.116720825,
    -0.49224812,
    0.32395956,
    -0.1659309,
    0.13609141,
    -0.019698426,
    0.4019622,
    -0.334695,
    -0.102382824,
    -0.44328713,
    0.07782495,
    -0.48113766,
    0.4907025,
    0.36495674,
    0.45440108,
    -0.0433515,
    -0.21830899,
    -0.48403463,
    0.13596614,
    -0.87134326,
    0.12563528,
    0.6807467,
    0.20989059,
    -0.28153777,
    0.5853704,
    0.1542272,
    0.22318108,
    -0.26371175,
    0.46274626,
    0.1486539,
    0.10559261,
    -0.1085327,
    -0.571272,
    -0.16197905,
    -0.57890975,
    -0.43521696,
    0.3713826,
    0.34621453,
    -0.19711521,
];

const DEEPSEEK2_YARN_V3_GOLDEN: [f32; 48] = [
    -0.47413808,
    0.35526732,
    -0.2582222,
    -0.18980934,
    -0.95664954,
    -0.8509193,
    -0.43231177,
    -0.1713295,
    0.10211216,
    0.11649465,
    -0.49237254,
    0.32420722,
    -0.1657735,
    0.1371189,
    -0.019091293,
    0.4019858,
    -0.33366066,
    -0.103042334,
    -0.44388324,
    0.07815254,
    -0.48238492,
    0.49077725,
    0.36485127,
    0.45375234,
    -0.044579536,
    -0.21795247,
    -0.4845924,
    0.13551636,
    -0.8708396,
    0.12551789,
    0.6797794,
    0.20931362,
    -0.28019375,
    0.5862174,
    0.15487202,
    0.22378036,
    -0.26452285,
    0.4638173,
    0.1471195,
    0.10499786,
    -0.107970566,
    -0.5719474,
    -0.16266495,
    -0.57996434,
    -0.43385056,
    0.37049216,
    0.34687984,
    -0.19830683,
];

const DEEPSEEK2_YARN_LEGACY_GOLDEN: [f32; 48] = [
    -0.47546828,
    0.354503,
    -0.25883463,
    -0.18994278,
    -0.95712084,
    -0.8507633,
    -0.43271223,
    -0.17080505,
    0.10281198,
    0.11672078,
    -0.49224812,
    0.32395953,
    -0.16593093,
    0.13609144,
    -0.019698426,
    0.4019622,
    -0.33469498,
    -0.102382824,
    -0.44328725,
    0.07782501,
    -0.48113763,
    0.49070254,
    0.36495668,
    0.45440114,
    -0.04335141,
    -0.2183089,
    -0.4840346,
    0.135966,
    -0.87134314,
    0.1256353,
    0.6807467,
    0.20989063,
    -0.2815378,
    0.5853704,
    0.15422714,
    0.22318108,
    -0.2637118,
    0.46274644,
    0.14865392,
    0.10559255,
    -0.10853268,
    -0.5712719,
    -0.16197906,
    -0.5789098,
    -0.43521684,
    0.37138268,
    0.34621453,
    -0.19711521,
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
        (YARN_V2, &DEEPSEEK2_YARN_V2_GOLDEN),
        (YARN_V3, &DEEPSEEK2_YARN_V3_GOLDEN),
        (YARN_LEGACY, &DEEPSEEK2_YARN_LEGACY_GOLDEN),
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
    println!(
        "plain vs yarn_v2 golden: max |delta| = {:.3e}; yarn_v2 vs yarn_v3: {:.3e}",
        worst_vs(&DEEPSEEK2_GOLDEN, &DEEPSEEK2_YARN_V2_GOLDEN),
        worst_vs(&DEEPSEEK2_YARN_V2_GOLDEN, &DEEPSEEK2_YARN_V3_GOLDEN)
    );
}

/// YaRN as the two real DeepSeek generations declare it, on both
/// tensor forms, against libllama.
#[test]
fn yarn_matches_llama_cpp_in_both_generations_and_both_forms() {
    for (name, golden) in [
        (YARN_V2, &DEEPSEEK2_YARN_V2_GOLDEN),
        (YARN_V3, &DEEPSEEK2_YARN_V3_GOLDEN),
        (YARN_LEGACY, &DEEPSEEK2_YARN_LEGACY_GOLDEN),
    ] {
        let engine = load(name).unwrap_or_else(|e| panic!("{name} loads: {e}"));
        let yarn = engine.yarn.as_ref().expect("the file declares YaRN");
        assert_eq!(
            yarn.freq_factors.len(),
            2,
            "one divisor per band of qk_rope = 4"
        );
        assert!(
            yarn.freq_factors.iter().any(|&f| f != 1.0),
            "the ramp moved a band"
        );
        assert!(
            (yarn.pe_magnitude - 1.0).abs() < 1e-6,
            "libllama logs `yarn_attn_factor = 1.0000` for both shapes: {}",
            yarn.pe_magnitude
        );
        assert_close(&decode(&engine), golden, GRAPH_TOL, name);
    }
    // The two generations differ in `kq_scale` alone (`mscale^2`,
    // `mscale = 1 + 0.1 * L * ln 4`), and the numbers say so.
    let v2 = load(YARN_V2).unwrap().yarn.unwrap();
    let v3 = load(YARN_V3).unwrap().yarn.unwrap();
    assert_eq!(v2.freq_factors, v3.freq_factors);
    let l2 = 0.707f32;
    let expect2 = (1.0 + 0.1 * l2 * 4f32.ln()).powi(2) / 12f32.sqrt();
    let expect3 = (1.0 + 0.1 * 1.0 * 4f32.ln()).powi(2) / 12f32.sqrt();
    assert!(
        (v2.kq_scale - expect2).abs() < 1e-6,
        "{} vs {expect2}",
        v2.kq_scale
    );
    assert!(
        (v3.kq_scale - expect3).abs() < 1e-6,
        "{} vs {expect3}",
        v3.kq_scale
    );
}

/// Each YaRN piece sabotaged on the loaded engine: the divisors, the
/// scale, and both together (a plain file's numbers on a YaRN file),
/// each moving the logits past the tolerance.
#[test]
fn each_yarn_piece_is_visible_in_the_logits() {
    let mut engine = load(YARN_V2).unwrap();
    assert_close(
        &decode(&engine),
        &DEEPSEEK2_YARN_V2_GOLDEN,
        GRAPH_TOL,
        "baseline",
    );
    let yarn = engine.yarn.clone().unwrap();

    engine.yarn.as_mut().unwrap().freq_factors = vec![1.0; 2];
    let worst = worst_vs(&decode(&engine), &DEEPSEEK2_YARN_V2_GOLDEN);
    assert!(worst > 1e-4, "frequency rewrite not seen: {worst}");
    engine.yarn = Some(yarn.clone());

    engine.yarn.as_mut().unwrap().kq_scale = 1.0 / 12f32.sqrt();
    let worst = worst_vs(&decode(&engine), &DEEPSEEK2_YARN_V2_GOLDEN);
    assert!(worst > 1e-4, "kq_scale not seen: {worst}");
    engine.yarn = Some(yarn.clone());

    engine.yarn = None;
    let worst = worst_vs(&decode(&engine), &DEEPSEEK2_YARN_V2_GOLDEN);
    assert!(worst > 1e-3, "YaRN dropped entirely not seen: {worst}");
    engine.yarn = Some(yarn);
    assert_close(
        &decode(&engine),
        &DEEPSEEK2_YARN_V2_GOLDEN,
        GRAPH_TOL,
        "restored",
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
