//! Qwen1.5-MoE Metal prefill launch against real checkpoint weights.
#![cfg(feature = "metal")]

use std::path::{Path, PathBuf};

use frink_gguf::ShardedGguf;
use frink_models::config::ModelConfig;
use frink_models::decoder::Decoder;

fn qwen2moe_gguf_path() -> PathBuf {
    if let Ok(p) = std::env::var("FRINK_TEST_QWEN2MOE_GGUF") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf")
}

#[test]
#[ignore = "needs models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf and Metal"]
fn qwen15_moe_layer3_q5_0_down_prefill_launch_succeeds() {
    let path = qwen2moe_gguf_path();
    if !path.exists() {
        eprintln!("skip: {}", path.display());
        return;
    }
    let file = ShardedGguf::open(&path).expect("open");
    let config = ModelConfig::from_gguf(&file).expect("config");
    let decoder = Decoder::from_gguf(&path, config).expect("decoder");
    let layer = &decoder.layers[3];
    let planes = layer.moe.packed_q4.as_ref().expect("packed");
    let packed = planes.view();
    assert_eq!(packed.down_kind, "Q5_0");
    let hidden = packed.hidden_rows;
    let top_k = decoder.config.moe.n_experts_active;
    let n_tokens = 5usize;
    let n_experts = layer.moe.n_experts();
    let x_batch: Vec<f32> = (0..n_tokens * hidden)
        .map(|i| ((i as f32) * 0.0013).sin())
        .collect();
    let mut ids = Vec::with_capacity(n_tokens * top_k);
    let mut route = Vec::with_capacity(n_tokens * top_k);
    for t in 0..n_tokens {
        for k in 0..top_k {
            ids.push(((t + k) % n_experts) as i32);
            route.push(1.0 / top_k as f32);
        }
    }
    frink_metal::gpu::launch_moe_prefill_q4_0(&x_batch, n_tokens, &packed, &ids, &route, top_k)
        .expect("Q5_0 down prefill");
}

#[test]
#[ignore = "needs models/Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf and Metal"]
fn qwen15_moe_real_weights_prefill_launch_succeeds() {
    let path = qwen2moe_gguf_path();
    if !path.exists() {
        eprintln!("skip: {}", path.display());
        return;
    }
    let file = ShardedGguf::open(&path).expect("open");
    let config = ModelConfig::from_gguf(&file).expect("config");
    let decoder = Decoder::from_gguf(&path, config).expect("decoder");
    let layer = &decoder.layers[0];
    let planes = layer
        .moe
        .packed_q4
        .as_ref()
        .expect("layer 0 should have packed MoE planes");
    let packed = planes.view();
    eprintln!(
        "packed: gate_stride={} up_stride={} kinds={}/{}/{} ffn={} hidden={} experts={}",
        packed.gate_stride,
        packed.up_stride,
        packed.gate_kind,
        packed.up_kind,
        packed.down_kind,
        packed.ffn_rows,
        packed.hidden_rows,
        packed.n_experts,
    );
    assert_eq!(
        packed.gate_stride, packed.up_stride,
        "mul_mv_id gate∥up path requires equal stride"
    );

    let hidden = packed.hidden_rows;
    let top_k = decoder.config.moe.n_experts_active;
    let n_tokens = 5usize;
    let n_experts = layer.moe.n_experts();
    let x_batch: Vec<f32> = (0..n_tokens * hidden)
        .map(|i| ((i as f32) * 0.0013).sin())
        .collect();
    let mut ids = Vec::with_capacity(n_tokens * top_k);
    let mut route = Vec::with_capacity(n_tokens * top_k);
    for t in 0..n_tokens {
        for k in 0..top_k {
            ids.push(((t + k) % n_experts) as i32);
            route.push(1.0 / top_k as f32);
        }
    }

    match frink_metal::gpu::launch_moe_prefill_q4_0(
        &x_batch, n_tokens, &packed, &ids, &route, top_k,
    ) {
        Ok(out) => assert_eq!(out.len(), n_tokens * hidden),
        Err(e) => panic!("launch_moe_prefill_q4_0 failed: {e:?}"),
    }
}
