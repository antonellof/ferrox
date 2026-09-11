//! Serialized per-dispatch GPU cost of the decode stack's small kernels,
//! measured the way llama.cpp's `test-backend-ops perf` measures its own
//! under `GGML_METAL_CONCURRENCY_DISABLE=1`: N copies of one dispatch in
//! ONE encoder, a barrier between each, command-buffer GPU time over N.
//!
//! `crate::kernel_timing` attributes a real token's GPU time to kinds
//! but has to put every kind in its own encoder, so its small-kernel
//! rows carry an encoder boundary each. This is the number without it,
//! and the one to put beside a llama.cpp `us/run`. Hardware only.

use crate::attn::{encode_gqa_for_bench, encode_rope, MetalRope, MetalRopeLayout, RopeTarget};
use crate::elem::{encode_gelu_mul, encode_vec_add};
use crate::gpu::{
    compute_encoder_concurrent, memory_barrier_buffers, shared_metal, MetalError, SharedMetal,
};
use crate::norm::{encode_add_rms_norm, encode_rms_norm};
use crate::timing::{commit_wait_note, SubmitClock};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLResourceOptions,
};
use std::sync::Arc;

type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;

fn f32_buf(device: &ProtocolObject<dyn MTLDevice>, n: usize) -> Result<Buf, MetalError> {
    device
        .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)
}

fn f16_buf(device: &ProtocolObject<dyn MTLDevice>, n: usize) -> Result<Buf, MetalError> {
    device
        .newBufferWithLength_options(n * 2, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAllocFailed)
}

/// GPU microseconds per dispatch of `encode`, serialized `n` deep: the
/// MINIMUM over `REPEATS` command buffers, because this host is never
/// quiet and interference only ever adds time.
const REPEATS: usize = 7;

fn serialized_us(
    shared: &Arc<SharedMetal>,
    n: usize,
    mut encode: impl FnMut(
        &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    ) -> Result<(), MetalError>,
) -> Result<f64, MetalError> {
    let mut best = f64::INFINITY;
    for _ in 0..REPEATS {
        best = best.min(serialized_us_once(shared, n, &mut encode)?);
    }
    Ok(best)
}

fn serialized_us_once(
    shared: &Arc<SharedMetal>,
    n: usize,
    mut encode: impl FnMut(
        &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    ) -> Result<(), MetalError>,
) -> Result<f64, MetalError> {
    // Warm the pipeline outside the timed buffer.
    {
        let cmd_buf = shared
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandFailed)?;
        let encoder = compute_encoder_concurrent(&cmd_buf)?;
        encode(&encoder)?;
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }
    let clock = SubmitClock::start();
    let cmd_buf = shared
        .queue
        .commandBuffer()
        .ok_or(MetalError::CommandFailed)?;
    let encoder = compute_encoder_concurrent(&cmd_buf)?;
    for i in 0..n {
        if i > 0 {
            memory_barrier_buffers(&encoder);
        }
        encode(&encoder)?;
    }
    encoder.endEncoding();
    commit_wait_note(&cmd_buf, "kernel-bench", u64::MAX, clock);
    let gpu_s = cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime();
    Ok(gpu_s * 1e6 / n as f64)
}

/// Prints the serialized cost of each decode-stack small kernel at
/// Gemma-2-2B's shapes (hidden 2304, 8 q heads and 4 kv heads of 256,
/// ffn 9216) and Llama-3.2-3B's RoPE shape (24 + 8 heads of 128).
///
/// Run: `cargo test -p ferrox-metal --features metal -- --ignored
/// --nocapture serialized_small_kernel_costs`.
#[test]
#[ignore = "needs a real Metal GPU; prints a table"]
fn serialized_small_kernel_costs() {
    let shared = shared_metal().expect("Metal device");
    let device = &shared.device;
    const N: usize = 256;
    let hidden = 2304usize;
    let ffn = 9216usize;
    let (n_heads, n_kv, d) = (8u32, 4u32, 256u32);
    let kv_len = 96usize;

    let h = f32_buf(device, hidden).unwrap();
    let o = f32_buf(device, hidden).unwrap();
    let w = f32_buf(device, hidden).unwrap();
    let x = f32_buf(device, hidden).unwrap();
    let gate = f32_buf(device, ffn).unwrap();
    let up = f32_buf(device, ffn).unwrap();
    let act = f32_buf(device, ffn).unwrap();
    let q = f32_buf(device, (n_heads * d) as usize).unwrap();
    let k = f32_buf(device, (n_kv * d) as usize).unwrap();
    let kc = f16_buf(device, kv_len * (n_kv * d) as usize).unwrap();
    let vc = f16_buf(device, kv_len * (n_kv * d) as usize).unwrap();
    let attn = f32_buf(device, (n_heads * d) as usize).unwrap();
    let q3b = f32_buf(device, 24 * 128).unwrap();
    let k3b = f32_buf(device, 8 * 128).unwrap();

    let rope = MetalRope::new(MetalRopeLayout::Neox);
    let rope_norm = MetalRope::new(MetalRopeLayout::Norm);
    let mut rows: Vec<(&str, f64)> = Vec::new();
    let mut row = |name: &'static str, r: Result<f64, MetalError>| {
        rows.push((name, r.expect(name)));
    };

    row(
        "vec_add 2304",
        serialized_us(&shared, N, |e| {
            encode_vec_add(e, device, &h, &o, hidden as u32)
        }),
    );
    row(
        "rms_norm 2304",
        serialized_us(&shared, N, |e| {
            encode_rms_norm(e, device, &h, &w, &x, hidden as u32, 1e-6)
        }),
    );
    row(
        "add_rms_norm 2304",
        serialized_us(&shared, N, |e| {
            encode_add_rms_norm(e, device, &h, &o, &w, &x, hidden as u32, 1e-6)
        }),
    );
    row(
        "gelu_mul 9216",
        serialized_us(&shared, N, |e| {
            encode_gelu_mul(e, device, &gate, &up, &act, ffn as u32)
        }),
    );
    row(
        "rope neox 8+4 heads d256",
        serialized_us(&shared, N, |e| {
            encode_rope(
                e,
                device,
                rope,
                RopeTarget { vecs: &q, n_heads },
                Some(RopeTarget {
                    vecs: &k,
                    n_heads: n_kv,
                }),
                d,
                10000.0,
                50,
                None,
            )
        }),
    );
    row(
        "rope norm 24+8 heads d128",
        serialized_us(&shared, N, |e| {
            encode_rope(
                e,
                device,
                rope_norm,
                RopeTarget {
                    vecs: &q3b,
                    n_heads: 24,
                },
                Some(RopeTarget {
                    vecs: &k3b,
                    n_heads: 8,
                }),
                128,
                500000.0,
                50,
                None,
            )
        }),
    );
    row(
        "gqa d256 8/4 heads kv96 softcap",
        serialized_us(&shared, N, |e| {
            encode_gqa_for_bench(
                e,
                device,
                &q,
                &kc,
                &vc,
                &attn,
                n_heads,
                n_kv,
                d,
                kv_len as u32,
                0,
                Some(50.0),
            )
        }),
    );
    row(
        "gqa d256 8/4 heads kv96 no softcap",
        serialized_us(&shared, N, |e| {
            encode_gqa_for_bench(
                e,
                device,
                &q,
                &kc,
                &vc,
                &attn,
                n_heads,
                n_kv,
                d,
                kv_len as u32,
                0,
                None,
            )
        }),
    );

    eprintln!("serialized GPU us per dispatch, {N} deep, barrier between each, min of {REPEATS}:");
    for (name, us) in &rows {
        eprintln!("  {name:<36} {us:>8.2} us");
    }
}
