//! RMSNorm on Metal, in the shapes the decode and prefill stacks need:
//! plain, fused with the residual add that precedes it (ggml F=3), the
//! batched row forms of both, the batched forms that write f16 for the
//! prefill GEMM, and the per-head QK-norm.
//!
//! Split out of `elem.rs` along the seam the Gemma-2 decode work
//! touches: a sandwich-norm layer runs four of these per layer, and
//! the kernels' cost is what that work changes.

use crate::dispatch::dispatch_counted;
use crate::gpu::{ensure_pipeline, MetalError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLSize};
use std::ptr::NonNull;

const RMS_NORM_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    float partial = 0.0f;
    for (uint i = tid; i < n; i += tg) {
        float v = x[i];
        partial += v * v;
    }
    partial = simd_sum(partial);
    if (tiisg == 0u) {
        scratch[sgitg] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg + 31u) / 32u;
    if (tiisg < nsg) {
        total = scratch[tiisg];
    }
    total = simd_sum(total);
    float inv_rms = rsqrt(total / float(n) + eps);
    for (uint i = tid; i < n; i += tg) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// One threadgroup per row — avoids B separate dispatches on prefill.
kernel void rms_norm_f32_batch(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    device const float* xr = x + row * n;
    device float* orow = out + row * n;
    float partial = 0.0f;
    for (uint i = tid; i < n; i += tg) {
        float v = xr[i];
        partial += v * v;
    }
    partial = simd_sum(partial);
    if (tiisg == 0u) {
        scratch[sgitg] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg + 31u) / 32u;
    if (tiisg < nsg) {
        total = scratch[tiisg];
    }
    total = simd_sum(total);
    float inv_rms = rsqrt(total / float(n) + eps);
    for (uint i = tid; i < n; i += tg) {
        orow[i] = xr[i] * inv_rms * weight[i];
    }
}

// RMSNorm writing half (skip separate f32→f16 before mul_mm_sg_f16).
kernel void rms_norm_f32_to_f16_batch(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    device const float* xr = x + row * n;
    device half* orow = out + row * n;
    float partial = 0.0f;
    for (uint i = tid; i < n; i += tg) {
        float v = xr[i];
        partial += v * v;
    }
    partial = simd_sum(partial);
    if (tiisg == 0u) {
        scratch[sgitg] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg + 31u) / 32u;
    if (tiisg < nsg) {
        total = scratch[tiisg];
    }
    total = simd_sum(total);
    float inv_rms = rsqrt(total / float(n) + eps);
    for (uint i = tid; i < n; i += tg) {
        orow[i] = half(xr[i] * inv_rms * weight[i]);
    }
}
"#;

const ADD_RMS_NORM_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_rms_norm_f32(
    device float* h [[buffer(0)]],
    device const float* add [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    float partial = 0.0f;
    for (uint i = tid; i < n; i += tg) {
        float v = h[i] + add[i];
        h[i] = v;
        partial += v * v;
    }
    partial = simd_sum(partial);
    if (tiisg == 0u) {
        scratch[sgitg] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg + 31u) / 32u;
    if (tiisg < nsg) {
        total = scratch[tiisg];
    }
    total = simd_sum(total);
    float inv_rms = rsqrt(total / float(n) + eps);
    for (uint i = tid; i < n; i += tg) {
        out[i] = h[i] * inv_rms * weight[i];
    }
}

// One threadgroup per row (prefill B tokens).
kernel void add_rms_norm_f32_batch(
    device float* h [[buffer(0)]],
    device const float* add [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    device float* hr = h + row * n;
    device const float* ar = add + row * n;
    device float* orow = out + row * n;
    float partial = 0.0f;
    for (uint i = tid; i < n; i += tg) {
        float v = hr[i] + ar[i];
        hr[i] = v;
        partial += v * v;
    }
    partial = simd_sum(partial);
    if (tiisg == 0u) {
        scratch[sgitg] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg + 31u) / 32u;
    if (tiisg < nsg) {
        total = scratch[tiisg];
    }
    total = simd_sum(total);
    float inv_rms = rsqrt(total / float(n) + eps);
    for (uint i = tid; i < n; i += tg) {
        orow[i] = hr[i] * inv_rms * weight[i];
    }
}

// Same, writing half — folds the f32→f16 staging convert that used to sit
// between this norm and `mul_mm_sg_f16` into the norm's own store loop.
// Saves one dispatch and, more importantly, one barrier per layer.
kernel void add_rms_norm_f32_to_f16_batch(
    device float* h [[buffer(0)]],
    device const float* add [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    device float* hr = h + row * n;
    device const float* ar = add + row * n;
    device half* orow = out + row * n;
    float partial = 0.0f;
    for (uint i = tid; i < n; i += tg) {
        float v = hr[i] + ar[i];
        hr[i] = v;
        partial += v * v;
    }
    partial = simd_sum(partial);
    if (tiisg == 0u) {
        scratch[sgitg] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    const uint nsg = (tg + 31u) / 32u;
    if (tiisg < nsg) {
        total = scratch[tiisg];
    }
    total = simd_sum(total);
    float inv_rms = rsqrt(total / float(n) + eps);
    for (uint i = tid; i < n; i += tg) {
        orow[i] = half(hr[i] * inv_rms * weight[i]);
    }
}
"#;

const RMS_NORM_PER_HEAD_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_per_head_f32(
    device float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    device float* xh = x + head * head_dim;
    float partial = 0.0f;
    for (uint i = tiisg; i < head_dim; i += 32u) {
        float v = xh[i];
        partial += v * v;
    }
    float total = simd_sum(partial);
    float inv_rms = rsqrt(total / float(head_dim) + eps);
    for (uint i = tiisg; i < head_dim; i += 32u) {
        xh[i] = xh[i] * inv_rms * weight[i];
    }
}
"#;

pub(crate) fn encode_rms_norm(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    eps: f32,
) -> Result<(), MetalError> {
    encode_rms_norm_at(encoder, device, x, 0, weight, out, 0, n, eps)
}

/// [`encode_rms_norm`] with byte offsets into `x` / `out` (prefill batch rows).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rms_norm_at(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &ProtocolObject<dyn MTLBuffer>,
    x_off_bytes: usize,
    weight: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    out_off_bytes: usize,
    n: u32,
    eps: f32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, RMS_NORM_KERNEL_SRC, "rms_norm_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    let tg = 256u32;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x), x_off_bytes, 0);
        encoder.setBuffer_offset_atIndex(Some(weight), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out), out_off_bytes, 2);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut eps_f = eps;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut eps_f as *mut f32 as *mut _).unwrap(),
            4,
            4,
        );
        // One float per simdgroup (simd_sum path).
        encoder.setThreadgroupMemoryLength_atIndex(((tg as usize) / 32) * 4, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Batched RMSNorm: one threadgroup per row of length `n` (`batch` rows).
/// Contiguous layout `[batch][n]` in `x` / `out`. Replaces `batch` calls to
/// [`encode_rms_norm_at`] (dominant dispatch storm on tiny-model pp512).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rms_norm_batch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    batch: u32,
    eps: f32,
) -> Result<(), MetalError> {
    if batch == 0 {
        return Ok(());
    }
    if batch == 1 {
        return encode_rms_norm(encoder, device, x, weight, out, n, eps);
    }
    let pipe = ensure_pipeline(device, RMS_NORM_KERNEL_SRC, "rms_norm_f32_batch")?;
    encoder.setComputePipelineState(&pipe.0);
    let tg = 256u32;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(weight), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 2);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut eps_f = eps;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut eps_f as *mut f32 as *mut _).unwrap(),
            4,
            4,
        );
        encoder.setThreadgroupMemoryLength_atIndex(((tg as usize) / 32) * 4, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: batch as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Batched RMSNorm writing `half` rows (prefill → `mul_mm_sg_f16`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rms_norm_f32_to_f16_batch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    out_h: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    batch: u32,
    eps: f32,
) -> Result<(), MetalError> {
    if batch == 0 {
        return Ok(());
    }
    let pipe = ensure_pipeline(device, RMS_NORM_KERNEL_SRC, "rms_norm_f32_to_f16_batch")?;
    encoder.setComputePipelineState(&pipe.0);
    let tg = 256u32;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(weight), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out_h), 0, 2);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
        let mut eps_f = eps;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut eps_f as *mut f32 as *mut _).unwrap(),
            4,
            4,
        );
        encoder.setThreadgroupMemoryLength_atIndex(((tg as usize) / 32) * 4, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: batch as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// `h += add`, then `out = rms_norm(h) * weight`. One dispatch replaces
/// [`encode_vec_add`] + [`encode_rms_norm`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_add_rms_norm(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    h: &ProtocolObject<dyn MTLBuffer>,
    add: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    eps: f32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, ADD_RMS_NORM_KERNEL_SRC, "add_rms_norm_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    let tg = 256u32;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(h), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(add), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(weight), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 3);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        let mut eps_f = eps;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut eps_f as *mut f32 as *mut _).unwrap(),
            4,
            5,
        );
        // One float per simdgroup (tg/32).
        encoder.setThreadgroupMemoryLength_atIndex(((tg as usize) / 32) * 4, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Batched [`encode_add_rms_norm`]: `h[row] += add[row]`, then
/// `out[row] = rms_norm(h[row]) * weight` for `batch` contiguous rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_add_rms_norm_batch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    h: &ProtocolObject<dyn MTLBuffer>,
    add: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    batch: u32,
    eps: f32,
) -> Result<(), MetalError> {
    if batch == 0 {
        return Ok(());
    }
    if batch == 1 {
        return encode_add_rms_norm(encoder, device, h, add, weight, out, n, eps);
    }
    let pipe = ensure_pipeline(device, ADD_RMS_NORM_KERNEL_SRC, "add_rms_norm_f32_batch")?;
    encoder.setComputePipelineState(&pipe.0);
    let tg = 256u32;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(h), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(add), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(weight), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 3);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        let mut eps_f = eps;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut eps_f as *mut f32 as *mut _).unwrap(),
            4,
            5,
        );
        encoder.setThreadgroupMemoryLength_atIndex(((tg as usize) / 32) * 4, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: batch as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// [`encode_add_rms_norm_batch`] storing `half` — the fused residual add +
/// FFN RMSNorm + f32→f16 staging convert the dense prefill layer needs
/// before `mul_mm_sg_f16`. `h` is still updated in f32 (it is the residual
/// stream); only `out` is half. Saves one dispatch and one barrier/layer.
///
/// Unlike [`encode_add_rms_norm_batch`] there is no `batch == 1` fallback:
/// the one-threadgroup-per-row kernel is correct at any batch and the only
/// caller (the prefill stack) rejects `batch < 4` anyway.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_add_rms_norm_f32_to_f16_batch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    h: &ProtocolObject<dyn MTLBuffer>,
    add: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    batch: u32,
    eps: f32,
) -> Result<(), MetalError> {
    if batch == 0 {
        return Ok(());
    }
    let pipe = ensure_pipeline(
        device,
        ADD_RMS_NORM_KERNEL_SRC,
        "add_rms_norm_f32_to_f16_batch",
    )?;
    encoder.setComputePipelineState(&pipe.0);
    let tg = 256u32;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(h), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(add), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(weight), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 3);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        let mut eps_f = eps;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut eps_f as *mut f32 as *mut _).unwrap(),
            4,
            5,
        );
        encoder.setThreadgroupMemoryLength_atIndex(((tg as usize) / 32) * 4, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: batch as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// In-place per-head RMSNorm×γ over `n_heads * head_dim` values in `x`
/// (`batch == 1`). Tests and decode helpers may call this; prefill uses
/// [`encode_rms_norm_per_head_batch`].
#[allow(dead_code)]
pub(crate) fn encode_rms_norm_per_head(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(), MetalError> {
    encode_rms_norm_per_head_batch(encoder, device, x, weight, n_heads, head_dim, 1, eps)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rms_norm_per_head_batch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    head_dim: u32,
    batch: u32,
    eps: f32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(
        device,
        RMS_NORM_PER_HEAD_KERNEL_SRC,
        "rms_norm_per_head_f32",
    )?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(weight), 0, 1);
        let mut hd = head_dim;
        encoder.setBytes_length_atIndex(NonNull::new(&mut hd as *mut u32 as *mut _).unwrap(), 4, 2);
        let mut eps_f = eps;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut eps_f as *mut f32 as *mut _).unwrap(),
            4,
            3,
        );
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: (n_heads * batch) as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Compile every norm pipeline the prefill stack can reach, so the
/// first prefill command buffer does not pay for it.
pub(crate) fn warm_prefill_norm_pipelines(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
) -> Result<(), MetalError> {
    ensure_pipeline(device, RMS_NORM_KERNEL_SRC, "rms_norm_f32")?;
    ensure_pipeline(device, RMS_NORM_KERNEL_SRC, "rms_norm_f32_batch")?;
    ensure_pipeline(device, RMS_NORM_KERNEL_SRC, "rms_norm_f32_to_f16_batch")?;
    ensure_pipeline(device, ADD_RMS_NORM_KERNEL_SRC, "add_rms_norm_f32")?;
    ensure_pipeline(device, ADD_RMS_NORM_KERNEL_SRC, "add_rms_norm_f32_batch")?;
    ensure_pipeline(
        device,
        ADD_RMS_NORM_KERNEL_SRC,
        "add_rms_norm_f32_to_f16_batch",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elem::tests::{read_f32, upload};
    use crate::gpu::shared_metal;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rms_norm_per_head_matches_cpu() {
        let shared = shared_metal().expect("metal");
        let device = &shared.device;
        let n_heads = 5usize;
        let head_dim = 96usize; // deliberately not a multiple of 32 lanes' 64/128
        let eps = 1e-6f32;
        let x: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| (i as f32 * 0.037).sin() * 2.0)
            .collect();
        let w: Vec<f32> = (0..head_dim).map(|i| 0.8 + (i as f32) * 0.003).collect();

        let mut cpu = x.clone();
        for h in 0..n_heads {
            let s = &mut cpu[h * head_dim..(h + 1) * head_dim];
            let mean_sq = s.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for (v, ww) in s.iter_mut().zip(w.iter()) {
                *v = *v * inv * ww;
            }
        }

        let x_buf = upload(device, &x).unwrap();
        let w_buf = upload(device, &w).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_rms_norm_per_head(
            &enc,
            device,
            &x_buf,
            &w_buf,
            n_heads as u32,
            head_dim as u32,
            eps,
        )
        .unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let gpu = read_f32(&x_buf, x.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "per-head rms {i}: {a} vs {b}");
        }
    }
}
