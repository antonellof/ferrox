//! MoE expert quant kinds per layer for Qwen1.5-MoE (GGUF tensor table).

use std::path::{Path, PathBuf};

use frink_gguf::ShardedGguf;

fn qwen2moe_gguf_path() -> PathBuf {
    if let Ok(p) = std::env::var("FRINK_TEST_QWEN2MOE_GGUF") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf")
}

fn tensor_kind(file: &ShardedGguf, name: &str) -> Option<String> {
    file.tensors()
        .find(|(_, t)| t.name == name)
        .map(|(_, t)| format!("{:?}", t.dtype))
}

#[test]
#[ignore = "needs Qwen1.5-MoE GGUF"]
fn qwen15_moe_expert_quant_kinds_from_gguf() {
    let path = qwen2moe_gguf_path();
    if !path.exists() {
        eprintln!("skip: {}", path.display());
        return;
    }
    let file = ShardedGguf::open(&path).expect("open");
    let n_layers = file
        .metadata_u64("qwen2moe.block_count")
        .or_else(|| file.metadata_u64("general.block_count"))
        .unwrap_or(24) as usize;
    for i in 0..n_layers {
        let gate = tensor_kind(&file, &format!("blk.{i}.ffn_gate_exps.weight"));
        let up = tensor_kind(&file, &format!("blk.{i}.ffn_up_exps.weight"));
        let down = tensor_kind(&file, &format!("blk.{i}.ffn_down_exps.weight"));
        eprintln!("layer {i}: gate={gate:?} up={up:?} down={down:?}");
        assert_eq!(
            gate, up,
            "layer {i}: gate/up dtype must match for Metal MoE"
        );
        assert!(down.is_some(), "layer {i}: missing down expert tensor");
    }
}
