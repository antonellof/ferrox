//! Qwen1.5-MoE packed-plane stride check — no full decoder load.

use std::path::{Path, PathBuf};

use frink_gguf::ShardedGguf;

fn qwen2moe_gguf_path() -> PathBuf {
    if let Ok(p) = std::env::var("FRINK_TEST_QWEN2MOE_GGUF") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf")
}

#[test]
#[ignore = "needs models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf"]
fn qwen15_moe_gate_and_up_expert_bytes_match() {
    let path = qwen2moe_gguf_path();
    if !path.exists() {
        eprintln!("skip: {}", path.display());
        return;
    }
    let file = ShardedGguf::open(&path).expect("open");
    let n_experts = 60usize;
    let gate = file
        .tensor_bytes("blk.0.ffn_gate_exps.weight")
        .expect("gate");
    let up = file.tensor_bytes("blk.0.ffn_up_exps.weight").expect("up");
    assert_eq!(gate.len() % n_experts, 0);
    assert_eq!(up.len() % n_experts, 0);
    let gate_stride = gate.len() / n_experts;
    let up_stride = up.len() / n_experts;
    assert_eq!(
        gate_stride, up_stride,
        "Metal MoE prefill requires equal gate/up expert stride; \
         gate={gate_stride} up={up_stride}"
    );
}
