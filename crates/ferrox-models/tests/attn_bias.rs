//! The per-layer bias tensors llama.cpp's loaders REQUIRE, pinned
//! against what ferrox's generic decoder can actually apply.
//!
//! Third table in the `rope_layout.rs` family, and the same failure
//! mode: a dropped bias is invisible. `attn_output.bias` missing does
//! not make the model refuse or crash, it makes every attention output
//! off by a learned constant; a missing `attn_norm.bias` turns a
//! LayerNorm into an RMSNorm. Both load clean, run at full speed, and
//! answer fluently.
//!
//! `LLAMA_REQUIRED_BIASES` transcribes every
//! `create_tensor(tn(TENSOR, "bias", i), {...}, 0)` in
//! `src/models/*.cpp` — flag `0`, i.e. REQUIRED, as opposed to
//! `TENSOR_NOT_REQUIRED`. Required matters: it means llama.cpp refuses
//! to load a checkpoint of that architecture *without* the tensor, so
//! every real file has it. Optional biases are deliberately out of
//! scope; they need a tensor-presence gate at load time, not a registry
//! entry.
//!
//! What ferrox has a slot for is a short list:
//!
//! - `AttnWeights::{q_bias, k_bias, v_bias}` — the split
//!   `blk.N.attn_{q,k,v}.bias` spelling, AND the fused
//!   `blk.N.attn_qkv.bias`, which `qkv_fused` slices into the same
//!   three fields by the same spans that split the fused weight. The
//!   fused spelling was unread until 2026-09-10, which is why `qwen`
//!   and `chatglm` were refused; both are audited now.
//! - `GptOssLayer::{o_bias, router_bias}` — gpt-oss's
//!   `blk.N.attn_output.bias` and `blk.N.ffn_gate_inp.bias`, on the
//!   gpt-oss path only.
//!
//! Everything else in the table below is read by nothing.
//!
//! Regenerate by re-reading the reference; do not edit an entry to make
//! a failing test pass.

use ferrox_models::capability::{resolve_profile, ArchPath};

/// The QKV biases, in both spellings -- the only required biases the
/// generic decoder applies. Anything else in a row means that
/// architecture must not reach the generic path.
///
/// `attn_qkv.bias` joined this list with `qkv_fused`, and
/// `the_fused_qkv_bias_entry_is_backed_by_real_code` below is what
/// stops it from being a name in a table that no loader reads -- the
/// exact shape of the defect it was added to fix.
const GENERIC_DECODER_APPLIES: &[&str] =
    &["attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_qkv.bias"];

/// `(gguf arch, required bias tensors, citation)`.
///
/// Tensor names are the GGUF spellings from `LLM_TENSOR_NAMES`
/// (`src/llama-arch.cpp:383-420`), with the `blk.%d.` prefix dropped for
/// the per-layer ones.
const LLAMA_REQUIRED_BIASES: &[(&str, &[&str], &str)] = &[
    (
        "bloom",
        &[
            "output_norm.bias",
            "attn_norm.bias",
            "attn_qkv.bias",
            "attn_output.bias",
            "ffn_norm.bias",
            "ffn_down.bias",
            "ffn_up.bias",
        ],
        "src/models/bloom.cpp:30,42,45,48,51,54,57",
    ),
    (
        "codeshell",
        &[
            "output_norm.bias",
            "attn_norm.bias",
            "attn_output.bias",
            "ffn_norm.bias",
            "ffn_down.bias",
            "ffn_up.bias",
        ],
        "src/models/codeshell.cpp:24,31,36,39,42,45",
    ),
    (
        "falcon",
        &["output_norm.bias", "attn_norm.bias"],
        "src/models/falcon.cpp:21,33",
    ),
    (
        "gpt2",
        &[
            "output_norm.bias",
            "attn_norm.bias",
            "attn_qkv.bias",
            "attn_output.bias",
            "ffn_norm.bias",
            "ffn_down.bias",
            "ffn_up.bias",
        ],
        "src/models/gpt2.cpp:23,35,38,41,44,47,50",
    ),
    (
        "gptneox",
        &[
            "output_norm.bias",
            "attn_norm.bias",
            "attn_qkv.bias",
            "attn_output.bias",
            "ffn_norm.bias",
            "ffn_down.bias",
            "ffn_up.bias",
        ],
        "src/models/gptneox.cpp:57,64,67,70,73,76,79",
    ),
    (
        "jais",
        &[
            "output_norm.bias",
            "attn_norm.bias",
            "attn_qkv.bias",
            "attn_output.bias",
            "ffn_norm.bias",
            "ffn_down.bias",
            "ffn_gate.bias",
            "ffn_up.bias",
        ],
        "src/models/jais.cpp:22,29,32,35,38,41,44,47",
    ),
    (
        "jais2",
        &[
            "output_norm.bias",
            "attn_norm.bias",
            "attn_q.bias",
            "attn_k.bias",
            "attn_v.bias",
            "attn_output.bias",
            "ffn_norm.bias",
            "ffn_up.bias",
            "ffn_down.bias",
        ],
        "src/models/jais2.cpp:20,30,38,39,40,41,44,48,50",
    ),
    (
        "jina-bert-v2",
        &["attn_output.bias", "ffn_down.bias"],
        "src/models/jina-bert-v2.cpp:37,56",
    ),
    (
        "nemotron",
        &["output_norm.bias", "attn_norm.bias", "ffn_norm.bias"],
        "src/models/nemotron.cpp:19,26,35",
    ),
    (
        "gpt-oss",
        &["attn_output.bias", "ffn_gate_inp.bias"],
        "src/models/openai-moe.cpp:51,53",
    ),
    (
        "orion",
        &["output_norm.bias", "attn_norm.bias", "ffn_norm.bias"],
        "src/models/orion.cpp:18,25,31",
    ),
    (
        "pangu-embedded",
        &["attn_output.bias"],
        "src/models/pangu-embed.cpp:37",
    ),
    (
        "phi2",
        &[
            "output_norm.bias",
            "output.bias",
            "attn_norm.bias",
            "attn_output.bias",
            "ffn_down.bias",
            "ffn_up.bias",
        ],
        "src/models/phi2.cpp:20,22,28,33,36,39",
    ),
    (
        "phimoe",
        &[
            "output_norm.bias",
            "output.bias",
            "attn_norm.bias",
            "attn_output.bias",
            "ffn_norm.bias",
        ],
        "src/models/phimoe.cpp:21,23,29,33,36",
    ),
    ("qwen", &["attn_qkv.bias"], "src/models/qwen.cpp:28"),
    (
        "rwkv6",
        &["output_norm.bias", "attn_norm.bias"],
        "src/models/rwkv6.cpp:37,50",
    ),
    (
        "rwkv7",
        &["output_norm.bias", "attn_norm.bias"],
        "src/models/rwkv7.cpp:57,71",
    ),
    (
        "stablelm",
        &["output_norm.bias", "attn_norm.bias"],
        "src/models/stablelm.cpp:20,28",
    ),
    (
        "starcoder",
        &[
            "output_norm.bias",
            "attn_norm.bias",
            "attn_qkv.bias",
            "attn_output.bias",
            "ffn_norm.bias",
            "ffn_down.bias",
            "ffn_up.bias",
        ],
        "src/models/starcoder.cpp:24,37,40,43,46,49,52",
    ),
    (
        "starcoder2",
        &[
            "output_norm.bias",
            "attn_norm.bias",
            "attn_output.bias",
            "ffn_norm.bias",
            "ffn_down.bias",
            "ffn_up.bias",
        ],
        "src/models/starcoder2.cpp:23,35,41,44,50,51",
    ),
    (
        "wavtokenizer-dec",
        &["output_norm.bias", "output.bias"],
        "src/models/wavtokenizer-dec.cpp:107,111",
    ),
];

/// gpt-oss is the ONE architecture in the table admitted to the generic
/// path while requiring a bias outside `GENERIC_DECODER_APPLIES`,
/// because ferrox implements those two on the gpt-oss path.
const GPT_OSS_EXEMPTION: &str = "gpt-oss";

/// gpt-oss's exemption has to be backed by code, not by this constant.
///
/// If `GptOssLayer` ever loses either field the exemption below becomes
/// a lie, and the model quietly runs its attention output and its
/// router unbiased. Referencing both fields makes that a compile error.
/// `attn_qkv.bias`'s place in [`GENERIC_DECODER_APPLIES`] has to be
/// backed by a loader that reads it, not by this table saying so.
///
/// A name in a coverage list that no code honours is worse than no
/// entry: it stops `an_architecture_whose_required_bias_ferrox_drops...`
/// from refusing the row, so the architecture reaches the generic
/// decoder and runs unbiased -- which is exactly what happened to
/// `qwen` and `chatglm` for as long as the loader read only the split
/// spelling. The chatglm fixture carries the fused spelling and NOTHING
/// else, so all three biases here can only have come from splitting it.
#[test]
fn the_fused_qkv_bias_entry_is_backed_by_real_code() {
    assert!(GENERIC_DECODER_APPLIES.contains(&"attn_qkv.bias"));
    let path = format!(
        "{}/tests/fixtures/chatglm_tiny.gguf",
        env!("CARGO_MANIFEST_DIR")
    );
    let file = ferrox_gguf::GgufFile::open(&path).expect("fixture opens");
    assert!(
        file.find_tensor("blk.0.attn_qkv.bias").is_some()
            && file.find_tensor("blk.0.attn_q.bias").is_none(),
        "the fixture must carry ONLY the fused spelling, or this proves nothing"
    );
    let config = ferrox_models::ModelConfig::from_gguf(&file).expect("config parses");
    let d = ferrox_models::Decoder::from_gguf(&path, config).expect("fixture loads");
    for (i, layer) in d.layers.iter().enumerate() {
        assert!(
            layer.attn.q_bias.is_some()
                && layer.attn.k_bias.is_some()
                && layer.attn.v_bias.is_some(),
            "layer {i}: the fused attn_qkv.bias was not applied"
        );
    }
}

#[test]
fn the_gpt_oss_exemption_is_backed_by_real_fields() {
    let _o_bias: fn(&ferrox_models::decoder::GptOssLayer) -> &Vec<f32> = |l| &l.o_bias;
    let _router_bias: fn(&ferrox_models::decoder::GptOssLayer) -> &Vec<f32> = |l| &l.router_bias;
    assert!(matches!(
        resolve_profile(GPT_OSS_EXEMPTION).map(|p| p.path),
        Some(ArchPath::GenericGqa { .. })
    ));
}

/// An architecture llama.cpp REQUIRES a bias for that ferrox cannot
/// apply must not reach the generic decoder.
///
/// This is the assertion that found the bug. Nine architectures were on
/// `ArchPath::GenericGqa` while llama.cpp refuses to load them without
/// tensors ferrox never reads: `codeshell`, `jais2`, `nemotron`,
/// `orion`, `qwen`, `stablelm`, `starcoder`, `starcoder2` and `phimoe`.
/// Six of the nine require `attn_norm.bias` / `ffn_norm.bias` /
/// `output_norm.bias`, which is a LayerNorm; the generic decoder has
/// only `rms_norm(x, w, eps)`, no mean subtraction and no bias, so it
/// was computing a different normalisation at every layer of every one
/// of them.
///
/// EIGHT of the nine are still refused. `qwen` left, because its only
/// dropped bias was the fused `attn_qkv.bias` and `qkv_fused` applies
/// that now; it is audited against libllama in
/// `tests/one_match_arm_graphs.rs`. That is what this test is for -- a
/// row leaves it by the blocker being implemented, not by the row being
/// edited.
#[test]
fn an_architecture_whose_required_bias_ferrox_drops_is_not_on_the_generic_path() {
    let mut admitted = Vec::new();
    for &(arch, biases, cite) in LLAMA_REQUIRED_BIASES {
        if arch == GPT_OSS_EXEMPTION {
            continue;
        }
        let dropped: Vec<&str> = biases
            .iter()
            .copied()
            .filter(|b| !GENERIC_DECODER_APPLIES.contains(b))
            .collect();
        if dropped.is_empty() {
            continue;
        }
        let Some(p) = resolve_profile(arch) else {
            continue; // caught by the completeness test below
        };
        if matches!(p.path, ArchPath::GenericGqa { .. }) {
            admitted.push(format!("{arch} ({cite}) drops {dropped:?}"));
        }
    }
    assert!(
        admitted.is_empty(),
        "on the generic path while llama.cpp requires a bias ferrox never reads:\n  {}",
        admitted.join("\n  ")
    );
}

/// The refusals added for this reason must keep saying so.
///
/// A refusal whose reason drifts to something else is a refusal nobody
/// can act on, and it would let the arch back onto the generic path the
/// moment that other reason is fixed.
#[test]
fn the_bias_refusals_name_the_bias() {
    for arch in [
        "codeshell",
        "jais2",
        "nemotron",
        "orion",
        "stablelm",
        "starcoder",
        "starcoder2",
        "phimoe",
    ] {
        match resolve_profile(arch).map(|p| p.path) {
            Some(ArchPath::DedicatedOnly { reason }) => assert!(
                reason.contains("bias"),
                "{arch} is refused, but not for its biases: {reason}"
            ),
            other => panic!("{arch} must be refused for its required biases, got {other:?}"),
        }
    }
}

/// The transcription itself must not silently shrink, and every name in
/// it has to resolve or the test above compares nothing.
#[test]
fn the_bias_transcription_is_complete_enough_to_be_worth_pinning() {
    assert!(
        LLAMA_REQUIRED_BIASES.len() >= 21,
        "transcribed {} architectures with a required bias tensor; \
         `src/models/*.cpp` has 21",
        LLAMA_REQUIRED_BIASES.len()
    );
    let mut unknown = Vec::new();
    let mut empty = Vec::new();
    for &(arch, biases, _) in LLAMA_REQUIRED_BIASES {
        if resolve_profile(arch).is_none() {
            unknown.push(arch);
        }
        if biases.is_empty() {
            empty.push(arch);
        }
    }
    assert!(
        unknown.is_empty(),
        "in llama.cpp's inventory but not in ferrox's capability registry: {unknown:?}"
    );
    assert!(
        empty.is_empty(),
        "an empty bias list makes the row inert -- delete it or fill it: {empty:?}"
    );
    // A row consisting only of biases ferrox applies would make
    // `an_architecture_whose_required_bias_ferrox_drops...` skip it, so
    // at least one row must actually exercise the skip.
    assert!(
        LLAMA_REQUIRED_BIASES
            .iter()
            .any(|&(_, b, _)| b.iter().any(|x| GENERIC_DECODER_APPLIES.contains(x))),
        "no row exercises the supported-bias branch"
    );
}
