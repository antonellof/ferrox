//! `mistral3`, checked against llama.cpp itself: the per-position
//! attention temperature, and the YaRN magnitude term that reading it
//! found missing.
//!
//! `mistral3` was triaged NEW CODE on `attention.temperature_scale`.
//! `src/models/mistral3.cpp:5` reads it, `:14-17` floors it on
//! `hparams.n_ctx_orig_yarn`, and `:153-156` multiplies Q -- after
//! RoPE, before `build_attn`, `kq_scale` untouched -- by the per-token
//! vector `llama-graph.cpp:163-167` computes as
//! `log(floor(pos / floor_scale) + 1) * temp_scale + 1`. ferrox had no
//! per-position Q scale and NO gate on the key, so a real Ministral-3
//! loaded and ran at the wrong temperature with no error. That is the
//! defect this suite closes; `crate::attn_temperature` is the seam and
//! carries the census (three graphs of 140 build the input; only this
//! one is on the generic path).
//!
//! **Two things the fixtures found on the way.**
//!
//! 1. The verdict described the graph as "leading-dense + MoE + shared
//!    expert". `mistral3.cpp:64-84` is EITHER dense OR MoE on every
//!    layer (no `leading_dense_block_count` is read), and its
//!    `_shexp` tensors are created only under an `n_ff_shexp` its
//!    hparams never set and are consumed by no line of its graph. A
//!    `mistral3` file is a `llama` file with three keys, which is what
//!    real Ministral-3 files are (`conversion/mistral3.py:18-29`).
//! 2. `mistral3.cpp:9` reads `rope.scaling.yarn_log_multiplier`, whose
//!    job is to adjust YaRN's MAGNITUDE term, and ferrox turned out not
//!    to apply that term at all: `rope_attn_factor` carried
//!    `rope.scaling.attn_factor` alone where llama.cpp multiplies it by
//!    `get_mscale(factor, 1) / get_mscale(factor, log_mul)`
//!    (`llama-context.cpp:196-231`, with ggml's `rope_yarn` term
//!    cancelled). Every YaRN checkpoint on the generic path was roped
//!    at the right frequencies and the wrong magnitude -- both q and k,
//!    so attention logits low by `(1 + 0.1 ln factor)^2`, 1.30x at
//!    factor 4. `crate::yarn_magnitude` is the fix; `mistral3_yarn` and
//!    `mistral3_yarn_logmul` are the evidence for its two arms.
//!
//! **What each golden pins**, all five from one weight set so they
//! differ only by the keys under test (libllama's own logits, measured
//! before a line of Rust was written):
//!
//! | file | keys | vs `mistral3` |
//! |---|---|---|
//! | `mistral3` | none | -- |
//! | `mistral3_temp` | scale 0.5, `original_context_length = 2` | max diff 0.144 |
//! | `mistral3_temp_ctx` | scale 0.5, NO original-context key, `context_length = 2` | byte-identical to `mistral3_temp` |
//! | `mistral3_yarn` | scale 0.5, YaRN factor 4 over 4096 | max diff 0.086 |
//! | `mistral3_yarn_logmul` | the same + `yarn_log_multiplier = 0.5` | 0.046 from `mistral3_yarn` |
//!
//! The floor of 2 makes the temperature step TWICE inside the six-token
//! prompt (1, 1, 1.35, 1.35, 1.55, 1.55), so the per-position term is
//! exercised rather than merely declared; `mistral3_temp_ctx` measures
//! that `llama-model.cpp:1164` really seeds `n_ctx_orig_yarn` from
//! `n_ctx_train` when the YaRN key is absent, which is why the loader
//! hands the resolver that fallback. The YaRN files' floor of 4096 is
//! never reached, so their temperature is 1 everywhere and what they
//! see is the magnitude term; libllama's own log line for the second
//! reads `setting new yarn_attn_factor = 1.0648 (mscale == 1.0,
//! mscale_all_dim = 0.5)`, which is `get_mscale(4, 1) / get_mscale(4,
//! 0.5)` to four places.
//!
//! **Where the numbers come from.** Each golden was produced by running
//! llama.cpp's own graph over its fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`.
//!
//! Regenerating:
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_mistral3_fixture.py \
//!     crates/ferrox-models/tests/fixtures/mistral3_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_mistral3_fixture.py \
//!     crates/ferrox-models/tests/fixtures/mistral3_temp_tiny.gguf --temp
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_mistral3_fixture.py \
//!     crates/ferrox-models/tests/fixtures/mistral3_temp_ctx_tiny.gguf --temp-ctx
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_mistral3_fixture.py \
//!     crates/ferrox-models/tests/fixtures/mistral3_yarn_tiny.gguf --yarn
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_mistral3_fixture.py \
//!     crates/ferrox-models/tests/fixtures/mistral3_yarn_logmul_tiny.gguf --yarn-logmul
//! /tmp/ref_logits <fixture> 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, kl_vs_golden,
    load_graph_fixture, worst_vs, GRAPH_PROMPT,
};
use ferrox_models::attn_temperature::AttnTemperature;
use ferrox_models::{ModelConfig, RopeLayout};
use std::num::NonZeroU32;

const PLAIN: &str = "mistral3";
const TEMP: &str = "mistral3_temp";
const TEMP_CTX: &str = "mistral3_temp_ctx";
const YARN: &str = "mistral3_yarn";
const YARN_LOGMUL: &str = "mistral3_yarn_logmul";

const MISTRAL3_GOLDEN: [f32; 48] = [
    0.035145313,
    0.3805903,
    -0.65960157,
    -0.34140337,
    0.22953402,
    -0.19089603,
    -0.13329972,
    0.19459614,
    -0.030176222,
    0.52173686,
    -0.007890761,
    0.10892065,
    -0.46260777,
    0.1557597,
    -0.17326099,
    -0.33615416,
    0.73886025,
    0.30648717,
    -0.21968625,
    -0.36555165,
    -0.40534353,
    0.6346734,
    -0.6015506,
    -1.2230585,
    1.370538,
    -0.7122428,
    0.6104216,
    -0.4784669,
    0.3364483,
    -0.5658176,
    0.106032655,
    -0.06901357,
    0.82991946,
    -0.4170397,
    -0.27967316,
    0.48048717,
    0.68565196,
    0.1715278,
    0.38064677,
    -0.029511213,
    -0.09762375,
    -0.49827886,
    0.49238428,
    -0.43755686,
    0.67487586,
    -0.20328495,
    -0.25475603,
    0.3884181,
];

const MISTRAL3_TEMP_GOLDEN: [f32; 48] = [
    0.059440397,
    0.31117457,
    -0.6807579,
    -0.27981204,
    0.22777972,
    -0.1783177,
    -0.059636652,
    0.27615458,
    -0.08586329,
    0.55205303,
    -0.00616847,
    0.025807261,
    -0.43711376,
    0.09964611,
    -0.16606903,
    -0.36289242,
    0.72440004,
    0.34341347,
    -0.27377403,
    -0.36233854,
    -0.3985669,
    0.5740186,
    -0.64586645,
    -1.1662205,
    1.4224979,
    -0.77239895,
    0.6514742,
    -0.41272596,
    0.24619204,
    -0.5864017,
    0.06020665,
    -0.052058406,
    0.8355124,
    -0.48661178,
    -0.29611072,
    0.52294767,
    0.6897682,
    0.19648476,
    0.39624256,
    -0.06434509,
    -0.1327593,
    -0.46647474,
    0.6363203,
    -0.485519,
    0.7198791,
    -0.1593136,
    -0.25098082,
    0.40674073,
];

const MISTRAL3_TEMP_CTX_GOLDEN: [f32; 48] = [
    0.059440397,
    0.31117457,
    -0.6807579,
    -0.27981204,
    0.22777972,
    -0.1783177,
    -0.059636652,
    0.27615458,
    -0.08586329,
    0.55205303,
    -0.00616847,
    0.025807261,
    -0.43711376,
    0.09964611,
    -0.16606903,
    -0.36289242,
    0.72440004,
    0.34341347,
    -0.27377403,
    -0.36233854,
    -0.3985669,
    0.5740186,
    -0.64586645,
    -1.1662205,
    1.4224979,
    -0.77239895,
    0.6514742,
    -0.41272596,
    0.24619204,
    -0.5864017,
    0.06020665,
    -0.052058406,
    0.8355124,
    -0.48661178,
    -0.29611072,
    0.52294767,
    0.6897682,
    0.19648476,
    0.39624256,
    -0.06434509,
    -0.1327593,
    -0.46647474,
    0.6363203,
    -0.485519,
    0.7198791,
    -0.1593136,
    -0.25098082,
    0.40674073,
];

const MISTRAL3_YARN_GOLDEN: [f32; 48] = [
    0.037416384,
    0.35439748,
    -0.6726677,
    -0.31677258,
    0.2236433,
    -0.19846518,
    -0.09891516,
    0.25768638,
    -0.069939315,
    0.554602,
    -0.0036208183,
    0.05876775,
    -0.45582983,
    0.1381414,
    -0.18475357,
    -0.36375272,
    0.7353936,
    0.3289106,
    -0.2441029,
    -0.36568967,
    -0.4141621,
    0.6121365,
    -0.60600704,
    -1.2037218,
    1.4138552,
    -0.7501655,
    0.65165126,
    -0.4362427,
    0.29051006,
    -0.58705086,
    0.08573136,
    -0.05877699,
    0.8447298,
    -0.45682847,
    -0.28596085,
    0.5169485,
    0.6918694,
    0.18773945,
    0.4012962,
    -0.05182457,
    -0.13709591,
    -0.48134974,
    0.5779678,
    -0.46520582,
    0.7002759,
    -0.19636647,
    -0.2537354,
    0.399202,
];

const MISTRAL3_YARN_LOGMUL_GOLDEN: [f32; 48] = [
    0.035736978,
    0.3708968,
    -0.6652639,
    -0.3311197,
    0.22623576,
    -0.19652769,
    -0.11678791,
    0.2240156,
    -0.047929883,
    0.53712225,
    -0.0056593195,
    0.08694801,
    -0.46026525,
    0.14989093,
    -0.17889298,
    -0.3483265,
    0.736112,
    0.31717932,
    -0.22973104,
    -0.3663324,
    -0.4103995,
    0.6259844,
    -0.60395956,
    -1.217087,
    1.3917558,
    -0.73011506,
    0.63053167,
    -0.46009445,
    0.31492513,
    -0.5756469,
    0.09824181,
    -0.064462006,
    0.8396392,
    -0.43517584,
    -0.2823056,
    0.4969523,
    0.690752,
    0.17812343,
    0.39091194,
    -0.03800696,
    -0.11639012,
    -0.49040884,
    0.5322593,
    -0.44991818,
    0.6870024,
    -0.20304333,
    -0.2543509,
    0.393121,
];

/// The plain row: a `mistral3` file with none of its three keys is a
/// `llama` file, and must already match.
#[test]
fn a_plain_mistral3_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(PLAIN, &MISTRAL3_GOLDEN);
}

/// The row itself: the temperature stepping twice inside the prompt.
#[test]
fn the_attention_temperature_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(TEMP, &MISTRAL3_TEMP_GOLDEN);
}

/// The floor from `context_length` when the YaRN key is absent.
#[test]
fn the_context_length_floor_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(TEMP_CTX, &MISTRAL3_TEMP_CTX_GOLDEN);
}

/// YaRN's magnitude term, with no log multiplier.
#[test]
fn yarn_with_its_magnitude_term_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(YARN, &MISTRAL3_YARN_GOLDEN);
}

/// YaRN's magnitude term divided by the log multiplier's.
#[test]
fn yarn_with_a_log_multiplier_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(YARN_LOGMUL, &MISTRAL3_YARN_LOGMUL_GOLDEN);
}

/// The numbers in the report, so they can be regenerated rather than
/// trusted. Run with `--nocapture` to see them.
#[test]
fn report_kl_against_llama_cpp() {
    for (name, golden) in [
        (PLAIN, &MISTRAL3_GOLDEN),
        (TEMP, &MISTRAL3_TEMP_GOLDEN),
        (TEMP_CTX, &MISTRAL3_TEMP_CTX_GOLDEN),
        (YARN, &MISTRAL3_YARN_GOLDEN),
        (YARN_LOGMUL, &MISTRAL3_YARN_LOGMUL_GOLDEN),
    ] {
        let d = load_graph_fixture(name);
        let mut kv = graph_caches(&d);
        let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
        let kl = kl_vs_golden(&got, golden);
        let worst = worst_vs(&got, golden);
        println!("| `{name}` | {kl:.2e} | {worst:.2e} |");
        assert!(kl < 1e-8, "{name}: KL {kl}");
    }
}

/// What the loader resolved for each file, pinned value by value: the
/// scale, the floor and WHERE the floor came from.
#[test]
fn the_loader_resolves_the_temperature_and_its_floor_the_way_llama_cpp_does() {
    let plain = load_graph_fixture(PLAIN);
    assert_eq!(
        plain.config.attn_temperature, None,
        "no key, no temperature"
    );
    assert_eq!(
        plain.config.rope_layout,
        RopeLayout::Norm,
        "LLM_ARCH_MISTRAL3 is NORM"
    );
    assert_eq!(
        plain.config.attention_scale, None,
        "mistral3.cpp:115 passes 1/sqrt(n_embd_head)"
    );

    let want = AttnTemperature {
        scale: 0.5,
        floor_scale: NonZeroU32::new(2).unwrap(),
        offset: 0.0,
    };
    let temp = load_graph_fixture(TEMP);
    assert_eq!(
        temp.config.attn_temperature,
        Some(want),
        "floor from the YaRN key"
    );
    let temp_ctx = load_graph_fixture(TEMP_CTX);
    assert_eq!(
        temp_ctx.config.attn_temperature,
        Some(want),
        "floor from context_length when the key is absent (llama-model.cpp:1164)"
    );
    assert_eq!(
        temp_ctx.config.rope_orig_ctx, None,
        "the premise: this file really has no original-context key"
    );

    // The YaRN files: floor 4096, never reached in six tokens.
    let yarn = load_graph_fixture(YARN);
    let t = yarn.config.attn_temperature.expect("declared");
    assert_eq!(t.floor_scale.get(), 4096);
    for pos in 0..GRAPH_PROMPT.len() {
        assert_eq!(t.scale_at(pos), 1.0, "position {pos} is below a 4096 floor");
    }
}

/// The temperature is applied, and visible: dropping it diverges from
/// libllama by the 0.14 the goldens differ by.
#[test]
fn dropping_the_temperature_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture(TEMP);
    assert!(d.config.attn_temperature.is_some(), "the premise");
    d.config.attn_temperature = None;
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &MISTRAL3_TEMP_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "dropping the temperature moved the output by only {worst}; the floor must not be \
         stepping inside the prompt"
    );
    // And without the key the same weights ARE the plain file: the two
    // goldens differ by exactly the temperature.
    let mut kv = graph_caches(&d);
    let plain = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &MISTRAL3_GOLDEN,
    );
    assert!(
        plain < 1e-5,
        "with the temperature dropped this is the plain file: {plain}"
    );
}

/// The per-POSITION half specifically: a temperature that is one
/// constant for the whole prompt (the floor pushed past it) is a
/// different answer from one that steps, and so is one that steps at
/// every position. A helper that scaled every row by the last
/// position's value, or by the first's, would fail here and pass
/// nowhere else.
#[test]
fn the_temperature_is_per_position_not_one_constant() {
    let mut d = load_graph_fixture(TEMP);
    let t = d.config.attn_temperature.expect("declared");
    // Same scale, floor beyond the prompt: every position at 1.0.
    d.config.attn_temperature = Some(AttnTemperature {
        floor_scale: NonZeroU32::new(64).unwrap(),
        ..t
    });
    let mut kv = graph_caches(&d);
    let flat = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    assert!(worst_vs(&flat, &MISTRAL3_TEMP_GOLDEN) > 1e-2);
    // Floor 1: every position on its own period, steeper than the
    // golden's floor of 2 -- also wrong, in the other direction.
    d.config.attn_temperature = Some(AttnTemperature {
        floor_scale: NonZeroU32::new(1).unwrap(),
        ..t
    });
    let mut kv = graph_caches(&d);
    let steep = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    assert!(worst_vs(&steep, &MISTRAL3_TEMP_GOLDEN) > 1e-2);
    assert!(
        worst_vs(&steep, &flat) > 1e-2,
        "the two wrong floors must differ from each other"
    );
}

/// YaRN's magnitude term is applied, and visible: the loader folded
/// `1 + 0.1 ln 4` into `rope_attn_factor`, and resetting that to the
/// key's bare value (1.0 here) diverges from libllama.
#[test]
fn dropping_yarns_magnitude_term_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture(YARN);
    let want = 1.0 + 0.1 * 4f32.ln();
    assert!(
        (d.config.rope_attn_factor - want).abs() < 1e-6,
        "rope_attn_factor {} should be get_mscale(4, 1) = {want}",
        d.config.rope_attn_factor
    );
    d.config.rope_attn_factor = 1.0;
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        &MISTRAL3_YARN_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "dropping the magnitude term moved the output by only {worst}"
    );
}

/// The log multiplier is read for THIS architecture and changes the
/// magnitude: the two YaRN files share every weight and every other
/// key, and their goldens differ by 0.046.
#[test]
fn the_yarn_log_multiplier_is_applied_and_ignoring_it_diverges_from_llama_cpp() {
    let mut d = load_graph_fixture(YARN_LOGMUL);
    let want = (1.0 + 0.1 * 4f32.ln()) / (1.0 + 0.05 * 4f32.ln());
    assert!(
        (d.config.rope_attn_factor - want).abs() < 1e-6,
        "rope_attn_factor {} should be get_mscale(4, 1) / get_mscale(4, 0.5) = {want}",
        d.config.rope_attn_factor
    );
    // Ignoring the multiplier is the other file's factor, and the
    // other file's golden -- which this file's golden is not.
    d.config.rope_attn_factor = 1.0 + 0.1 * 4f32.ln();
    let mut kv = graph_caches(&d);
    let got = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv);
    assert!(worst_vs(&got, &MISTRAL3_YARN_LOGMUL_GOLDEN) > 1e-2);
    assert!(
        worst_vs(&got, &MISTRAL3_YARN_GOLDEN) < 1e-5,
        "with the multiplier ignored this is the other YaRN file"
    );
}

/// A `mistral3` file whose temperature has no floor to divide by is
/// refused, as `mistral3.cpp:16-17` refuses it. The `temp_ctx` fixture
/// with its `context_length` overridden to zero, so the refusal is
/// reachable from a file shaped like the converter's minus one value.
#[test]
fn a_temperature_with_no_floor_is_refused_like_llama_cpp_refuses_it() {
    use ferrox_gguf::{GgufValue, TensorSource};
    struct Wrapped(ferrox_gguf::GgufFile, GgufValue);
    impl TensorSource for Wrapped {
        fn metadata(&self, key: &str) -> Option<&GgufValue> {
            if key == "mistral3.context_length" {
                Some(&self.1)
            } else {
                self.0.metadata(key)
            }
        }
        fn find_tensor(&self, name: &str) -> Option<&ferrox_gguf::TensorInfo> {
            self.0.find_tensor(name)
        }
        fn tensor_bytes(&self, name: &str) -> Result<&[u8], ferrox_gguf::GgufError> {
            self.0.tensor_bytes(name)
        }
        fn tensor_mapped_range(
            &self,
            name: &str,
        ) -> Result<
            (
                std::sync::Arc<ferrox_gguf::MmapHandle>,
                std::ops::Range<usize>,
            ),
            ferrox_gguf::GgufError,
        > {
            self.0.tensor_mapped_range(name)
        }
    }
    let file = ferrox_gguf::GgufFile::open(graph_fixture_path(TEMP_CTX)).expect("fixture opens");
    let wrapped = Wrapped(file, GgufValue::U32(0));
    let err = ModelConfig::from_gguf(&wrapped).expect_err("no floor");
    let msg = format!("{err}");
    assert!(
        msg.contains("mistral3.attention.temperature_scale"),
        "{msg}"
    );
    assert!(msg.contains("mistral3.cpp:16-17"), "{msg}");
    // The unwrapped file is the one that loads, or the gate above
    // would be refusing the architecture rather than the floor.
    let ok = ferrox_gguf::GgufFile::open(graph_fixture_path(TEMP_CTX)).expect("fixture opens");
    assert!(ModelConfig::from_gguf(&ok).is_ok());
}
