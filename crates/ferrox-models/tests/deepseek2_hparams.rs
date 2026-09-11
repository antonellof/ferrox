//! The `deepseek2` MLA hparam contract: does ferrox's loader ask for
//! keys a real checkpoint actually carries?
//!
//! `deepseek2` is what DeepSeek-V2, V2.5, V3 and R1 all tag, so it is
//! the architecture behind the largest open models people run. Nobody
//! had checked the contract, and it was wrong:
//! `read_deepseek2_hparams` required `{arch}.attention.qk_nope_head_dim`
//! and `.qk_rope_head_dim`, and **neither is a GGUF key**. Neither
//! string appears in `.scratch/llama.cpp/src/llama-arch.cpp`'s
//! `LLM_KV_NAMES` nor anywhere in `gguf-py`; they are HF `config.json`
//! field names. So every real DeepSeek checkpoint failed with "missing
//! hparam deepseek2.attention.qk_nope_head_dim" -- a true statement
//! about a key no converter has ever written.
//!
//! That is the same shape as the `glm4moe` defect
//! (`tests/glm4moe_refusal.rs`): a loader demanding a hyper-parameter
//! the architecture is not supposed to have, and the error naming the
//! demand rather than the mismatch.
//!
//! What a real file carries, and how llama.cpp derives the per-head
//! dims from it (`src/models/deepseek2.cpp:77-82`):
//!
//! ```text
//! qk_rope = rope.dimension_count
//! qk_nope = attention.key_length_mla - qk_rope
//! v_head  = attention.value_length_mla
//! ```
//!
//! `tests/fixtures/deepseek2_tiny.gguf` is the evidence. Every metadata
//! key in it is transcribed from llama.cpp's own converter
//! (`conversion/deepseek.py::DeepseekV2Model.set_gguf_parameters`), so
//! it carries exactly what a converted DeepSeek carries and nothing
//! ferrox invented.
//!
//! Regenerating it:
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_deepseek2_fixture.py \
//!     crates/ferrox-models/tests/fixtures/deepseek2_tiny.gguf
//! ```

use ferrox_models::mla_gguf_loader::read_deepseek2_hparams;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/deepseek2_tiny.gguf"
);

// From `scripts/make_deepseek2_fixture.py`.
const QK_NOPE: usize = 8;
const QK_ROPE: usize = 4;
const V_HEAD: usize = 8;
const KV_LORA: usize = 12;
const Q_LORA: usize = 16;

fn open() -> ferrox_gguf::GgufFile {
    ferrox_gguf::GgufFile::open(FIXTURE).expect("fixture opens")
}

/// The fixture is only evidence if it really lacks the HF-only keys --
/// otherwise this suite tests a strawman rather than the contract.
#[test]
fn a_real_shaped_deepseek2_checkpoint_carries_no_hf_head_dim_keys() {
    let file = open();
    assert_eq!(file.metadata_str("general.architecture"), Some("deepseek2"));
    for absent in [
        "deepseek2.attention.qk_nope_head_dim",
        "deepseek2.attention.qk_rope_head_dim",
    ] {
        assert!(
            file.metadata_u64(absent).is_none(),
            "{absent} is an HF config.json field, not a GGUF key -- if a converter \
             started writing it, re-derive this whole suite"
        );
    }
    for present in [
        "deepseek2.attention.key_length_mla",
        "deepseek2.attention.value_length_mla",
        "deepseek2.rope.dimension_count",
        "deepseek2.attention.q_lora_rank",
        "deepseek2.attention.kv_lora_rank",
    ] {
        assert!(
            file.metadata_u64(present).is_some(),
            "{present} is what llama.cpp's converter emits and the loader must read"
        );
    }
}

/// The per-head dims are DERIVED, and each has to come out right.
#[test]
fn the_per_head_dims_are_derived_the_way_llama_cpp_derives_them() {
    let hp = read_deepseek2_hparams(&open()).expect("a converter-shaped deepseek2 file loads");
    assert_eq!(
        hp.qk_rope_head_dim, QK_ROPE,
        "qk_rope = rope.dimension_count"
    );
    assert_eq!(
        hp.qk_nope_head_dim, QK_NOPE,
        "qk_nope = key_length_mla - rope.dimension_count"
    );
    assert_eq!(hp.v_head_dim, V_HEAD, "v_head = value_length_mla");
    assert_eq!(hp.kv_lora_rank, KV_LORA);
    assert_eq!(hp.q_lora_rank, Q_LORA);
}

/// The trap this replaced, and the reason the fix is not "read
/// `attention.key_length` instead".
///
/// For an MLA checkpoint `attention.key_length` and `.value_length` hold
/// the COMPRESSED MQA widths -- `kv_lora_rank + qk_rope` and
/// `kv_lora_rank` -- not per-head dims. Reading them as per-head dims
/// does not fail; it builds a differently shaped model, which is the
/// worse outcome.
#[test]
fn the_compressed_widths_are_not_the_per_head_dims() {
    let file = open();
    let key_length = file
        .metadata_u64("deepseek2.attention.key_length")
        .expect("converter writes it");
    let value_length = file
        .metadata_u64("deepseek2.attention.value_length")
        .expect("converter writes it");

    assert_eq!(key_length as usize, KV_LORA + QK_ROPE);
    assert_eq!(value_length as usize, KV_LORA);

    let hp = read_deepseek2_hparams(&file).expect("loads");
    assert_ne!(
        key_length as usize,
        hp.qk_nope_head_dim + hp.qk_rope_head_dim,
        "if these were equal the mix-up would be invisible and this test worthless"
    );
    assert_ne!(value_length as usize, hp.v_head_dim);
}

/// A `deepseek2` file declaring Mistral-Large-3's per-position
/// attention temperature is REFUSED by the MLA loader, naming the seam,
/// where it used to load and silently drop both keys.
///
/// `deepseek2.cpp:46-47` reads `attention.temperature_scale` and
/// `attention.temperature_length` ("used by mistral-large") and
/// `:595-598` / `:632-635` multiply Q by the per-position scale; the
/// MLA engine has no such multiply and no libllama-golden to check one
/// against, so the file stops. `deepseek2_temp_tiny.gguf` is the
/// fixture above plus the two keys `conversion/mistral.py:110,177`
/// write, and llama.cpp's loader reads both (measured).
#[test]
fn a_mistral_large_temperature_is_refused_by_name_rather_than_dropped() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/deepseek2_temp_tiny.gguf"
    );
    let file = ferrox_gguf::GgufFile::open(path).expect("fixture opens");
    // The premise: the file really carries the keys.
    assert_eq!(
        ferrox_gguf::TensorSource::metadata_f32(&file, "deepseek2.attention.temperature_scale"),
        Some(0.5)
    );
    let err = read_deepseek2_hparams(&file).expect_err("a per-position temperature is refused");
    let msg = format!("{err}");
    assert!(
        msg.contains("deepseek2.attention.temperature_scale"),
        "{msg}"
    );
    assert!(msg.contains("attn_temperature"), "{msg}");
    assert!(msg.contains("deepseek2.cpp:46-47"), "{msg}");
    // And the plain fixture, which differs only by those two keys,
    // still loads -- the refusal is on the key, not the architecture.
    read_deepseek2_hparams(&open()).expect("the plain file loads");
}
