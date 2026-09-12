//! Synthetic-GGUF tests for the MLA loader: the dense and MoE shapes,
//! the trunk-only MTP split, the lite direct-Q rule, the RoPE-scaling
//! refusal. The libllama-golden coverage is `tests/plm_graphs.rs`.

use super::*;
use crate::engine::Engine;
use byteorder::{LittleEndian, WriteBytesExt};
use ferrox_gguf::GgufFile;
use std::io::Write;

struct FixtureTensor {
    name: String,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.write_f32::<LittleEndian>(*x).unwrap();
    }
    b
}

fn f32_tensor(name: &str, shape: Vec<u64>, values: Vec<f32>) -> FixtureTensor {
    FixtureTensor {
        name: name.into(),
        shape,
        bytes: f32_bytes(&values),
    }
}

fn build_gguf(
    arch: &str,
    kv: &[(&str, u64)],
    fkv: &[(&str, f32)],
    tensors: &[FixtureTensor],
) -> Vec<u8> {
    build_gguf_full(arch, kv, fkv, &[], tensors)
}

fn build_gguf_full(
    arch: &str,
    kv: &[(&str, u64)],
    fkv: &[(&str, f32)],
    skv: &[(&str, &str)],
    tensors: &[FixtureTensor],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(ferrox_gguf::GGUF_MAGIC)
        .unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
    // general.architecture + uint + float + string kvs
    let kv_count = 1 + kv.len() + fkv.len() + skv.len();
    buf.write_u64::<LittleEndian>(kv_count as u64).unwrap();

    let write_string = |buf: &mut Vec<u8>, s: &str| {
        buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        buf.write_all(s.as_bytes()).unwrap();
    };
    write_string(&mut buf, "general.architecture");
    buf.write_u32::<LittleEndian>(8).unwrap();
    write_string(&mut buf, arch);
    for &(k, v) in kv {
        write_string(&mut buf, k);
        buf.write_u32::<LittleEndian>(10).unwrap(); // UINT64
        buf.write_u64::<LittleEndian>(v).unwrap();
    }
    for &(k, v) in fkv {
        write_string(&mut buf, k);
        buf.write_u32::<LittleEndian>(6).unwrap(); // FLOAT32
        buf.write_f32::<LittleEndian>(v).unwrap();
    }
    for &(k, v) in skv {
        write_string(&mut buf, k);
        buf.write_u32::<LittleEndian>(8).unwrap(); // STRING
        write_string(&mut buf, v);
    }

    let mut offset = 0u64;
    let mut offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        write_string(&mut buf, &t.name);
        buf.write_u32::<LittleEndian>(t.shape.len() as u32).unwrap();
        for &d in t.shape.iter().rev() {
            buf.write_u64::<LittleEndian>(d).unwrap();
        }
        buf.write_u32::<LittleEndian>(0).unwrap();
        offsets.push(offset);
        buf.write_u64::<LittleEndian>(offset).unwrap();
        offset += (t.bytes.len().div_ceil(32) * 32) as u64;
    }
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    let data_start = buf.len();
    for (t, &off) in tensors.iter().zip(offsets.iter()) {
        while buf.len() < data_start + off as usize {
            buf.push(0);
        }
        buf.extend_from_slice(&t.bytes);
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
    }
    buf
}

/// The two-layer dense fixture, with `n_mtp_blocks` NextN/MTP blocks
/// declared after it: `block_count` counts them and
/// `nextn_predict_layers` names them, and NO tensor is written for
/// them -- llama.cpp's "trunk-only" split (`deepseek2.cpp:64-66`),
/// which is also exactly the file that would fail to load if the
/// blocks were treated as layers.
fn synthetic_dense_deepseek2(n_mtp_blocks: u64) -> Vec<u8> {
    synthetic_dense_deepseek2_with(2, n_mtp_blocks, QForm::LowRank, None)
}

/// Which Q tensors a synthetic layer carries.
#[derive(Clone, Copy, PartialEq)]
enum QForm {
    /// `attn_q_a` + `attn_q_b`.
    LowRank,
    /// One `attn_q`.
    Direct,
    /// All three: a file no graph reads whole.
    Both,
}

fn synthetic_dense_deepseek2_with(
    n_layers: usize,
    n_mtp_blocks: u64,
    q_form: QForm,
    rope_scaling: Option<&str>,
) -> Vec<u8> {
    let h = 16usize;
    let n_heads = 2usize;
    let q_lora = 8usize;
    let kv_lora = 4usize;
    let qk_nope = 4usize;
    let qk_rope = 2usize;
    let v_dim = 4usize;
    let ffn = 32usize;
    let vocab = 8usize;
    let q_head = qk_nope + qk_rope;
    let arch = "deepseek2";

    let mut tensors = vec![
        f32_tensor(
            "token_embd.weight",
            vec![vocab as u64, h as u64],
            vec![0.01; h * vocab],
        ),
        f32_tensor("output_norm.weight", vec![h as u64], vec![1.0; h]),
        f32_tensor(
            "output.weight",
            vec![vocab as u64, h as u64],
            vec![0.02; h * vocab],
        ),
    ];
    for l in 0..n_layers {
        tensors.push(f32_tensor(
            &format!("blk.{l}.attn_norm.weight"),
            vec![h as u64],
            vec![1.0; h],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.ffn_norm.weight"),
            vec![h as u64],
            vec![1.0; h],
        ));
        if q_form != QForm::Direct {
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_q_a.weight"),
                vec![q_lora as u64, h as u64],
                vec![0.01; h * q_lora],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_q_b.weight"),
                vec![(n_heads * q_head) as u64, q_lora as u64],
                vec![0.01; q_lora * n_heads * q_head],
            ));
        }
        if q_form != QForm::LowRank {
            tensors.push(f32_tensor(
                &format!("blk.{l}.attn_q.weight"),
                vec![(n_heads * q_head) as u64, h as u64],
                vec![0.01; h * n_heads * q_head],
            ));
        }
        tensors.push(f32_tensor(
            &format!("blk.{l}.attn_kv_a_mqa.weight"),
            vec![(kv_lora + qk_rope) as u64, h as u64],
            vec![0.01; h * (kv_lora + qk_rope)],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.attn_kv_b.weight"),
            vec![(n_heads * (qk_nope + v_dim)) as u64, kv_lora as u64],
            vec![0.01; kv_lora * n_heads * (qk_nope + v_dim)],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.attn_output.weight"),
            vec![h as u64, (n_heads * v_dim) as u64],
            vec![0.01; n_heads * v_dim * h],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.ffn_gate.weight"),
            vec![ffn as u64, h as u64],
            vec![0.01; h * ffn],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.ffn_up.weight"),
            vec![ffn as u64, h as u64],
            vec![0.01; h * ffn],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.ffn_down.weight"),
            vec![h as u64, ffn as u64],
            vec![0.01; ffn * h],
        ));
    }

    let kv = [
        ("deepseek2.block_count", n_layers as u64 + n_mtp_blocks),
        ("deepseek2.nextn_predict_layers", n_mtp_blocks),
        ("deepseek2.embedding_length", h as u64),
        ("deepseek2.feed_forward_length", ffn as u64),
        ("deepseek2.attention.head_count", n_heads as u64),
        ("deepseek2.attention.q_lora_rank", q_lora as u64),
        ("deepseek2.attention.kv_lora_rank", kv_lora as u64),
        ("deepseek2.attention.qk_nope_head_dim", qk_nope as u64),
        ("deepseek2.attention.qk_rope_head_dim", qk_rope as u64),
        ("deepseek2.attention.v_head_dim", v_dim as u64),
        ("deepseek2.leading_dense_block_count", n_layers as u64),
        ("deepseek2.expert_count", 0u64),
    ];
    let fkv = [
        ("deepseek2.attention.layer_norm_rms_epsilon", 1e-5f32),
        ("deepseek2.rope.freq_base", 10000.0f32),
    ];
    let skv: Vec<(&str, &str)> = rope_scaling
        .map(|k| ("deepseek2.rope.scaling.type", k))
        .into_iter()
        .collect();
    build_gguf_full(arch, &kv, &fkv, &skv, &tensors)
}

fn write_temp(bytes: &[u8], tag: &str) -> std::path::PathBuf {
    let path =
        std::env::temp_dir().join(format!("ferrox_mla_gguf_{tag}_{}.gguf", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path
}

fn load_and_forward(bytes: &[u8], tag: &str) -> MlaEngine {
    let path = write_temp(bytes, tag);
    let file = GgufFile::open(&path).unwrap();
    let engine = load_mla_engine(&file).expect("load mla");
    let mut state = engine.new_state();
    let logits = engine.forward_token(0, 0, &mut state);
    assert_eq!(logits.len(), engine.vocab_size());
    assert!(logits.iter().all(|x| x.is_finite()));
    let _ = std::fs::remove_file(&path);
    engine
}

/// `deepseek2.cpp:8,11-13`: a 27-layer file is LITE, its
/// `q_lora_rank` key is never read, and Q is one direct `attn_q`.
/// The fixture CARRIES the key (`q_lora_rank = 8`), which is the
/// shape that tells "direct iff the key is absent" from upstream's
/// order; the loader used to demand `attn_q_a` here and fail every
/// DeepSeek-V2-Lite export.
#[test]
fn a_lite_deepseek2_projects_q_directly_whatever_its_key_says() {
    let engine = load_and_forward(
        &synthetic_dense_deepseek2_with(27, 0, QForm::Direct, None),
        "lite_direct",
    );
    assert_eq!(engine.layers.len(), 27);
    assert!(engine
        .layers
        .iter()
        .all(|l| matches!(l.attn.q, MlaQProj::Direct(_))));
    assert_eq!(engine.mla_cfg.q_lora_rank, 0);

    // The same 27 layers with the low-rank pair: upstream creates
    // `attn_q` for a lite file and fails on ITS absence; ferrox
    // names the stray pair instead of loading half a graph.
    let path = write_temp(
        &synthetic_dense_deepseek2_with(27, 0, QForm::LowRank, None),
        "lite_lowrank",
    );
    let err = load_mla_engine(&GgufFile::open(&path).unwrap())
        .err()
        .expect("refused");
    let _ = std::fs::remove_file(&path);
    assert!(
        err.to_string().contains("attn_q_a") && err.to_string().contains("projects Q directly"),
        "{err}"
    );

    // And a 2-layer (non-lite) file with only a direct `attn_q` is
    // refused for the low-rank tensors the key promises.
    let path = write_temp(
        &synthetic_dense_deepseek2_with(2, 0, QForm::Direct, None),
        "nonlite_direct",
    );
    let err = load_mla_engine(&GgufFile::open(&path).unwrap())
        .err()
        .expect("refused");
    let _ = std::fs::remove_file(&path);
    assert!(err.to_string().contains("attn_q_a"), "{err}");

    // Both forms in one file: refused whichever the rule picks.
    let path = write_temp(
        &synthetic_dense_deepseek2_with(27, 0, QForm::Both, None),
        "both",
    );
    let err = load_mla_engine(&GgufFile::open(&path).unwrap())
        .err()
        .expect("refused");
    let _ = std::fs::remove_file(&path);
    assert!(err.to_string().contains("never both"), "{err}");
}

/// `deepseek2.cpp:312-328`: a scaled file stops with the lines,
/// where it used to run at factor 1; `none` is not a scaling.
#[test]
fn a_rope_scaling_is_refused_by_name_and_none_is_not_one() {
    let refused = synthetic_dense_deepseek2_with(2, 0, QForm::LowRank, Some("yarn"));
    let path = write_temp(&refused, "yarn");
    let err = load_mla_engine(&GgufFile::open(&path).unwrap())
        .err()
        .expect("refused");
    let _ = std::fs::remove_file(&path);
    assert!(
        err.to_string().contains("rope.scaling.type = \"yarn\"")
            && err.to_string().contains("deepseek2.cpp:312-328"),
        "{err}"
    );
    let served = synthetic_dense_deepseek2_with(2, 0, QForm::LowRank, Some("none"));
    load_and_forward(&served, "scaling_none");
}

#[test]
fn load_synthetic_deepseek2_dense_and_forward() {
    let engine = load_and_forward(&synthetic_dense_deepseek2(0), "dense");
    assert_eq!(engine.layers.len(), 2);
    assert_eq!(engine.vocab_size(), 8);
}

/// A real DeepSeek-V3 / GLM-4.7-Flash export counts its MTP block
/// in `block_count` (`conversion/deepseek.py:457,498`). This loader
/// took `block_count` verbatim, so on such a file it would have
/// either run the block as a 62nd layer or, on a trunk-only split,
/// failed on its missing tensors. The trunk is what loads now, and
/// the block's absent tensors are not asked for.
#[test]
fn a_deepseek2_file_with_an_mtp_block_loads_only_its_trunk() {
    let engine = load_and_forward(&synthetic_dense_deepseek2(1), "mtp");
    assert_eq!(engine.layers.len(), 2, "block_count 3 minus one MTP block");
}

#[allow(clippy::too_many_arguments)] // test fixture: mirrors the MLA tensor shape set
fn push_mla_attn_tensors(
    tensors: &mut Vec<FixtureTensor>,
    l: usize,
    h: usize,
    n_heads: usize,
    q_lora: usize,
    kv_lora: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_dim: usize,
) {
    let q_head = qk_nope + qk_rope;
    tensors.push(f32_tensor(
        &format!("blk.{l}.attn_norm.weight"),
        vec![h as u64],
        vec![1.0; h],
    ));
    tensors.push(f32_tensor(
        &format!("blk.{l}.ffn_norm.weight"),
        vec![h as u64],
        vec![1.0; h],
    ));
    tensors.push(f32_tensor(
        &format!("blk.{l}.attn_q_a.weight"),
        vec![q_lora as u64, h as u64],
        vec![0.01; h * q_lora],
    ));
    tensors.push(f32_tensor(
        &format!("blk.{l}.attn_q_b.weight"),
        vec![(n_heads * q_head) as u64, q_lora as u64],
        vec![0.01; q_lora * n_heads * q_head],
    ));
    tensors.push(f32_tensor(
        &format!("blk.{l}.attn_kv_a_mqa.weight"),
        vec![(kv_lora + qk_rope) as u64, h as u64],
        vec![0.01; h * (kv_lora + qk_rope)],
    ));
    tensors.push(f32_tensor(
        &format!("blk.{l}.attn_kv_b.weight"),
        vec![(n_heads * (qk_nope + v_dim)) as u64, kv_lora as u64],
        vec![0.01; kv_lora * n_heads * (qk_nope + v_dim)],
    ));
    tensors.push(f32_tensor(
        &format!("blk.{l}.attn_output.weight"),
        vec![h as u64, (n_heads * v_dim) as u64],
        vec![0.01; n_heads * v_dim * h],
    ));
}

#[test]
fn load_synthetic_deepseek2_moe_after_dense_and_forward() {
    let h = 16usize;
    let n_heads = 2usize;
    let q_lora = 8usize;
    let kv_lora = 4usize;
    let qk_nope = 4usize;
    let qk_rope = 2usize;
    let v_dim = 4usize;
    let ffn = 32usize;
    let exp_ff = 16usize;
    let n_exp = 4usize;
    let vocab = 8usize;
    let arch = "deepseek2";

    let mut tensors = vec![
        f32_tensor(
            "token_embd.weight",
            vec![vocab as u64, h as u64],
            vec![0.01; h * vocab],
        ),
        f32_tensor("output_norm.weight", vec![h as u64], vec![1.0; h]),
        f32_tensor(
            "output.weight",
            vec![vocab as u64, h as u64],
            vec![0.02; h * vocab],
        ),
    ];
    // Layer 0: dense
    push_mla_attn_tensors(
        &mut tensors,
        0,
        h,
        n_heads,
        q_lora,
        kv_lora,
        qk_nope,
        qk_rope,
        v_dim,
    );
    tensors.push(f32_tensor(
        "blk.0.ffn_gate.weight",
        vec![ffn as u64, h as u64],
        vec![0.01; h * ffn],
    ));
    tensors.push(f32_tensor(
        "blk.0.ffn_up.weight",
        vec![ffn as u64, h as u64],
        vec![0.01; h * ffn],
    ));
    tensors.push(f32_tensor(
        "blk.0.ffn_down.weight",
        vec![h as u64, ffn as u64],
        vec![0.01; ffn * h],
    ));
    // Layer 1: MoE
    push_mla_attn_tensors(
        &mut tensors,
        1,
        h,
        n_heads,
        q_lora,
        kv_lora,
        qk_nope,
        qk_rope,
        v_dim,
    );
    tensors.push(f32_tensor(
        "blk.1.ffn_gate_inp.weight",
        vec![n_exp as u64, h as u64],
        vec![0.01; h * n_exp],
    ));
    // Packed expert tensors: logical [n_experts, out, in] → GGUF shape write uses rev
    // so pass shape as [n_experts, out, in] matching other fixtures' logical order.
    tensors.push(f32_tensor(
        "blk.1.ffn_gate_exps.weight",
        vec![n_exp as u64, exp_ff as u64, h as u64],
        vec![0.01; n_exp * exp_ff * h],
    ));
    tensors.push(f32_tensor(
        "blk.1.ffn_up_exps.weight",
        vec![n_exp as u64, exp_ff as u64, h as u64],
        vec![0.01; n_exp * exp_ff * h],
    ));
    tensors.push(f32_tensor(
        "blk.1.ffn_down_exps.weight",
        vec![n_exp as u64, h as u64, exp_ff as u64],
        vec![0.01; n_exp * h * exp_ff],
    ));
    tensors.push(f32_tensor(
        "blk.1.ffn_gate_shexp.weight",
        vec![exp_ff as u64, h as u64],
        vec![0.01; h * exp_ff],
    ));
    tensors.push(f32_tensor(
        "blk.1.ffn_up_shexp.weight",
        vec![exp_ff as u64, h as u64],
        vec![0.01; h * exp_ff],
    ));
    tensors.push(f32_tensor(
        "blk.1.ffn_down_shexp.weight",
        vec![h as u64, exp_ff as u64],
        vec![0.01; exp_ff * h],
    ));

    let kv = [
        ("deepseek2.block_count", 2u64),
        ("deepseek2.embedding_length", h as u64),
        ("deepseek2.feed_forward_length", ffn as u64),
        ("deepseek2.attention.head_count", n_heads as u64),
        ("deepseek2.attention.q_lora_rank", q_lora as u64),
        ("deepseek2.attention.kv_lora_rank", kv_lora as u64),
        ("deepseek2.attention.qk_nope_head_dim", qk_nope as u64),
        ("deepseek2.attention.qk_rope_head_dim", qk_rope as u64),
        ("deepseek2.attention.v_head_dim", v_dim as u64),
        ("deepseek2.leading_dense_block_count", 1u64),
        ("deepseek2.expert_count", n_exp as u64),
        ("deepseek2.expert_used_count", 2u64),
        ("deepseek2.expert_shared_count", 1u64),
        ("deepseek2.expert_feed_forward_length", exp_ff as u64),
        ("deepseek2.expert_gating_func", 1u64), // softmax
    ];
    let fkv = [
        ("deepseek2.attention.layer_norm_rms_epsilon", 1e-5f32),
        ("deepseek2.rope.freq_base", 10000.0f32),
        ("deepseek2.expert_weights_scale", 1.0f32),
    ];
    let bytes = build_gguf(arch, &kv, &fkv, &tensors);
    let path =
        std::env::temp_dir().join(format!("ferrox_mla_moe_gguf_{}.gguf", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let file = GgufFile::open(&path).unwrap();
    let engine = load_mla_engine(&file).expect("load mla moe");
    assert_eq!(engine.layers.len(), 2);
    assert!(matches!(
        engine.layers[0].ffn,
        crate::engine::MlaLayerFfn::Dense(_)
    ));
    assert!(matches!(
        engine.layers[1].ffn,
        crate::engine::MlaLayerFfn::Moe(_)
    ));
    assert!(engine.moe.is_some());
    let mut state = engine.new_state();
    let logits = engine.forward_token(0, 0, &mut state);
    assert_eq!(logits.len(), vocab);
    assert!(logits.iter().all(|x| x.is_finite()));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn moe_after_dense_fails_closed_without_expert_tensors() {
    let h = 16usize;
    let n_heads = 2usize;
    let q_lora = 8usize;
    let kv_lora = 4usize;
    let qk_nope = 4usize;
    let qk_rope = 2usize;
    let v_dim = 4usize;
    let ffn = 32usize;
    let vocab = 8usize;
    let arch = "deepseek2";

    let mut tensors = vec![
        f32_tensor(
            "token_embd.weight",
            vec![vocab as u64, h as u64],
            vec![0.01; h * vocab],
        ),
        f32_tensor("output_norm.weight", vec![h as u64], vec![1.0; h]),
        f32_tensor(
            "output.weight",
            vec![vocab as u64, h as u64],
            vec![0.02; h * vocab],
        ),
    ];
    for l in 0..2usize {
        push_mla_attn_tensors(
            &mut tensors,
            l,
            h,
            n_heads,
            q_lora,
            kv_lora,
            qk_nope,
            qk_rope,
            v_dim,
        );
        // Only dense FFN tensors — MoE layer 1 will fail closed.
        tensors.push(f32_tensor(
            &format!("blk.{l}.ffn_gate.weight"),
            vec![ffn as u64, h as u64],
            vec![0.01; h * ffn],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.ffn_up.weight"),
            vec![ffn as u64, h as u64],
            vec![0.01; h * ffn],
        ));
        tensors.push(f32_tensor(
            &format!("blk.{l}.ffn_down.weight"),
            vec![h as u64, ffn as u64],
            vec![0.01; ffn * h],
        ));
    }
    let kv = [
        ("deepseek2.block_count", 2u64),
        ("deepseek2.embedding_length", h as u64),
        ("deepseek2.feed_forward_length", ffn as u64),
        ("deepseek2.attention.head_count", n_heads as u64),
        ("deepseek2.attention.q_lora_rank", q_lora as u64),
        ("deepseek2.attention.kv_lora_rank", kv_lora as u64),
        ("deepseek2.attention.qk_nope_head_dim", qk_nope as u64),
        ("deepseek2.attention.qk_rope_head_dim", qk_rope as u64),
        ("deepseek2.attention.v_head_dim", v_dim as u64),
        ("deepseek2.leading_dense_block_count", 1u64),
        ("deepseek2.expert_count", 4u64),
        ("deepseek2.expert_used_count", 2u64),
    ];
    let fkv = [
        ("deepseek2.attention.layer_norm_rms_epsilon", 1e-5f32),
        ("deepseek2.rope.freq_base", 10000.0f32),
    ];
    let bytes = build_gguf(arch, &kv, &fkv, &tensors);
    let path = std::env::temp_dir().join(format!(
        "ferrox_mla_moe_missing_{}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, &bytes).unwrap();
    let file = GgufFile::open(&path).unwrap();
    let err = match load_mla_engine(&file) {
        Err(e) => e,
        Ok(_) => panic!("expected missing MoE tensors to fail closed"),
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("ffn_gate_inp") || msg.contains("TensorNotFound"),
        "unexpected error: {msg}"
    );
    let _ = std::fs::remove_file(&path);
}
