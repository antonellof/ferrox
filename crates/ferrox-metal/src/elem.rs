//! Metal elementwise ops for decode-layer residency: RMSNorm, residual
//! add, and SiLU×up (SwiGLU pair). Used by the fused dense-layer path
//! so activations stay on-GPU between attention and FFN.

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

const VEC_ADD_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void vec_add_f32(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        a[i] += b[i];
    }
}

kernel void f32_to_f16(
    device const float* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        dst[i] = half(src[i]);
    }
}
"#;

/// Fused residual add + RMSNorm×γ (ggml F=3-style, Pre-LN shape):
/// `h[i] += add[i]`, then `out[i] = rms_norm(h) * weight[i]`.
/// Replaces a separate `vec_add` + `rms_norm` pair (~2× dispatches/layer).
/// Uses simd_sum reduction (matches ggml `kernel_rms_norm_mul_add_f32`).
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

/// In-place per-head RMSNorm×γ (Qwen3 / Gemma-3 QK-norm): each head of
/// `head_dim` elements is normalized independently with the shared
/// `weight[head_dim]`. One simdgroup per head (head_dim ≤ 256 → ≤ 8
/// elements/lane), matching ggml's RMS_NORM on a [head_dim, n_head] view.
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

const SILU_MUL_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + exp(-g))) * up[i];
    }
}

// Same activation, half store — folds the f32→f16 staging convert that
// used to sit between SwiGLU and the down `mul_mm_sg_f16`. The product is
// still formed in f32 and rounded once, exactly as the two-dispatch pair
// did, so this is bit-identical to silu_mul_f32 + f32_to_f16.
kernel void silu_mul_f32_to_f16(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        float g = gate[i];
        out[i] = half((g / (1.0f + exp(-g))) * up[i]);
    }
}
"#;

/// `y[i] += a * x[i]` — MoE weighted expert accumulate.
const AXPY_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void axpy_f32(
    device float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant float& a [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        y[i] += a * x[i];
    }
}
"#;

/// Gemma GeGLU pair: `gelu(gate) * up`, tanh approximation matching
/// `ferrox_core::matmul::gelu` (HF / llama.cpp convention).
const GELU_MUL_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gelu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        float g = gate[i];
        const float K = 0.7978845608028654f; // sqrt(2/pi)
        const float C = 0.044715f;
        float gelu = 0.5f * g * (1.0f + precise::tanh(K * (g + C * g * g * g)));
        out[i] = gelu * up[i];
    }
}

// Half store — see `silu_mul_f32_to_f16`.
kernel void gelu_mul_f32_to_f16(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < n) {
        float g = gate[i];
        const float K = 0.7978845608028654f; // sqrt(2/pi)
        const float C = 0.044715f;
        float gelu = 0.5f * g * (1.0f + precise::tanh(K * (g + C * g * g * g)));
        out[i] = half(gelu * up[i]);
    }
}
"#;

/// Parallel argmax over `n` floats (one threadgroup). Each thread scans a
/// strided slice, then a tree-reduce keeps the first index on ties (`>`).
/// Sequential single-thread scan of vocab (~128k) was measured to erase the
/// gain from keeping lm_head on-GPU (llama.cpp leaves logits on device and
/// samples without a host round-trip).
const ARGMAX_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void argmax_f32(
    device const float* x [[buffer(0)]],
    device uint* out_idx [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* sh_v [[threadgroup(0)]],
    threadgroup uint* sh_i [[threadgroup(1)]]
) {
    float bv = -INFINITY;
    uint bi = 0u;
    for (uint i = tid; i < n; i += tg) {
        float v = x[i];
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    sh_v[tid] = bv;
    sh_i[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float v2 = sh_v[tid + stride];
            uint i2 = sh_i[tid + stride];
            float v1 = sh_v[tid];
            uint i1 = sh_i[tid];
            if (v2 > v1 || (v2 == v1 && i2 < i1)) {
                sh_v[tid] = v2;
                sh_i[tid] = i2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        out_idx[0] = sh_i[0];
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

pub(crate) fn encode_vec_add(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    encode_vec_add_at(encoder, device, a, 0, b, n)
}

/// Contiguous `float` → `half` (dense-prefill GEMM src1 / llama `*_f16`).
pub(crate) fn encode_f32_to_f16(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, VEC_ADD_KERNEL_SRC, "f32_to_f16")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst), 0, 1);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
    }
    let tg = 256usize;
    let n_tg = (n as usize).div_ceil(tg);
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// `a[a_offset_bytes/4 ..] += b[0..n]` (in-place on `a`).
pub(crate) fn encode_vec_add_at(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    a: &ProtocolObject<dyn MTLBuffer>,
    a_offset_bytes: usize,
    b: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, VEC_ADD_KERNEL_SRC, "vec_add_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a), a_offset_bytes, 0);
        encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
    }
    let tg = 256usize;
    let n_tg = (n as usize).div_ceil(tg);
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
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

pub(crate) fn encode_silu_mul(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, SILU_MUL_KERNEL_SRC, "silu_mul_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(gate), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(up), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 2);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
    }
    let tg = 256usize;
    let n_tg = (n as usize).div_ceil(tg);
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// `y[i] += a * x[i]`.
pub(crate) fn encode_axpy(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    y: &ProtocolObject<dyn MTLBuffer>,
    x: &ProtocolObject<dyn MTLBuffer>,
    a: f32,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, AXPY_KERNEL_SRC, "axpy_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(y), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(x), 0, 1);
        let mut a_f = a;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut a_f as *mut f32 as *mut _).unwrap(),
            4,
            2,
        );
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
    }
    let tg = 256usize;
    let n_tg = (n as usize).div_ceil(tg);
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// `out = gelu(gate) * up` (Gemma GeGLU; tanh-approx gelu).
/// SwiGLU / GeGLU writing `half` — [`encode_silu_mul`] / [`encode_gelu_mul`]
/// with the following f32→f16 staging convert folded in. Bit-identical to
/// the two-dispatch pair (the product is still formed in f32 and rounded
/// once); saves one dispatch and one barrier per layer.
pub(crate) fn encode_act_mul_f32_to_f16(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    gelu: bool,
) -> Result<(), MetalError> {
    let (src, name) = if gelu {
        (GELU_MUL_KERNEL_SRC, "gelu_mul_f32_to_f16")
    } else {
        (SILU_MUL_KERNEL_SRC, "silu_mul_f32_to_f16")
    };
    let pipe = ensure_pipeline(device, src, name)?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(gate), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(up), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 2);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
    }
    let tg = 256usize;
    let n_tg = (n as usize).div_ceil(tg);
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(crate) fn encode_gelu_mul(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, GELU_MUL_KERNEL_SRC, "gelu_mul_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(gate), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(up), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 2);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            3,
        );
    }
    let tg = 256usize;
    let n_tg = (n as usize).div_ceil(tg);
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Encode argmax of `n` f32 values in `x` into a single `u32` at `out_idx[0]`.
pub(crate) fn encode_argmax(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &ProtocolObject<dyn MTLBuffer>,
    out_idx: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) -> Result<(), MetalError> {
    let pipe = ensure_pipeline(device, ARGMAX_KERNEL_SRC, "argmax_f32")?;
    encoder.setComputePipelineState(&pipe.0);
    // Power-of-two TG; 1024 covers vocab with ~128 iters/thread on Llama-3.
    let tg = 1024u32;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(x), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_idx), 0, 1);
        let mut n_u = n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            2,
        );
        encoder.setThreadgroupMemoryLength_atIndex((tg as usize) * 4, 0);
        encoder.setThreadgroupMemoryLength_atIndex((tg as usize) * 4, 1);
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

/// Front-load elementwise pipelines used by [`crate::attn::launch_prefill_dense_layer`].
pub(crate) fn warm_prefill_elem_pipelines(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    gelu_ffn: bool,
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
    ensure_pipeline(device, VEC_ADD_KERNEL_SRC, "vec_add_f32")?;
    ensure_pipeline(device, VEC_ADD_KERNEL_SRC, "f32_to_f16")?;
    if gelu_ffn {
        ensure_pipeline(device, GELU_MUL_KERNEL_SRC, "gelu_mul_f32")?;
        ensure_pipeline(device, GELU_MUL_KERNEL_SRC, "gelu_mul_f32_to_f16")?;
    } else {
        ensure_pipeline(device, SILU_MUL_KERNEL_SRC, "silu_mul_f32")?;
        ensure_pipeline(device, SILU_MUL_KERNEL_SRC, "silu_mul_f32_to_f16")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{shared_metal, MetalError};
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLResourceOptions};
    use std::ptr::NonNull;

    fn upload(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        data: &[f32],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let mut owned = data.to_vec();
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(owned.as_mut_ptr() as *mut _).unwrap(),
                owned.len() * 4,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocFailed)
    }

    fn alloc(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        n: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        device
            .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocFailed)
    }

    fn read_f32(buf: &ProtocolObject<dyn MTLBuffer>, n: usize) -> Vec<f32> {
        let ptr = buf.contents();
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, n).to_vec() }
    }

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn rms_norm_silu_add_match_cpu() {
        let shared = shared_metal().expect("metal");
        let device = &shared.device;
        let n = 128usize;
        let eps = 1e-5f32;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).sin()).collect();
        let w: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        let cpu_rms: Vec<f32> = x
            .iter()
            .zip(w.iter())
            .map(|(v, ww)| v * scale * ww)
            .collect();

        let x_buf = upload(device, &x).unwrap();
        let w_buf = upload(device, &w).unwrap();
        let out_buf = alloc(device, n).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_rms_norm(&enc, device, &x_buf, &w_buf, &out_buf, n as u32, eps).unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let gpu_rms = read_f32(&out_buf, n);
        for (i, (a, b)) in cpu_rms.iter().zip(gpu_rms.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "rms {i}: {a} vs {b}");
        }

        let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos()).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin()).collect();
        let cpu_silu: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let g_buf = upload(device, &gate).unwrap();
        let u_buf = upload(device, &up).unwrap();
        let s_buf = alloc(device, n).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_silu_mul(&enc, device, &g_buf, &u_buf, &s_buf, n as u32).unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let gpu_silu = read_f32(&s_buf, n);
        for (i, (a, b)) in cpu_silu.iter().zip(gpu_silu.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "silu {i}: {a} vs {b}");
        }

        let mut a = cpu_rms.clone();
        let b = cpu_silu.clone();
        for (aa, bb) in a.iter_mut().zip(b.iter()) {
            *aa += bb;
        }
        let a_buf = upload(device, &cpu_rms).unwrap();
        let b_buf = upload(device, &b).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_vec_add(&enc, device, &a_buf, &b_buf, n as u32).unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let gpu_add = read_f32(&a_buf, n);
        for (i, (aa, bb)) in a.iter().zip(gpu_add.iter()).enumerate() {
            let tol = 1e-4 * aa.abs().max(1.0);
            assert!((aa - bb).abs() <= tol, "add {i}: {aa} vs {bb}");
        }

        // Fused h+=add; out=rms_norm(h)*w
        let h0: Vec<f32> = (0..n).map(|i| (i as f32 * 0.09).sin()).collect();
        let addend: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).cos()).collect();
        let mut h_cpu = h0.clone();
        for (h, a) in h_cpu.iter_mut().zip(addend.iter()) {
            *h += *a;
        }
        let mean_sq2 = h_cpu.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let scale2 = 1.0 / (mean_sq2 + eps).sqrt();
        let cpu_fused: Vec<f32> = h_cpu
            .iter()
            .zip(w.iter())
            .map(|(v, ww)| v * scale2 * ww)
            .collect();
        let h_buf = upload(device, &h0).unwrap();
        let add_buf = upload(device, &addend).unwrap();
        let fused_out = alloc(device, n).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_add_rms_norm(
            &enc, device, &h_buf, &add_buf, &w_buf, &fused_out, n as u32, eps,
        )
        .unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let gpu_h = read_f32(&h_buf, n);
        let gpu_fused = read_f32(&fused_out, n);
        for (i, (a, b)) in h_cpu.iter().zip(gpu_h.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "fused h {i}: {a} vs {b}");
        }
        for (i, (a, b)) in cpu_fused.iter().zip(gpu_fused.iter()).enumerate() {
            let tol = 1e-4 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "fused out {i}: {a} vs {b}");
        }
    }

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

    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn argmax_matches_host_including_ties() {
        let shared = shared_metal().expect("metal");
        let device = &shared.device;
        // Unique max at index 7; then a tie at 2 and 9 (first wins).
        let mut x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.13).sin()).collect();
        x[7] = 100.0;
        let host = x
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(host, 7);

        let x_buf = upload(device, &x).unwrap();
        let idx_buf = alloc(device, 1).unwrap();
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_argmax(&enc, device, &x_buf, &idx_buf, x.len() as u32).unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let ptr = idx_buf.contents();
        let gpu = unsafe { *(ptr.as_ptr() as *const u32) as usize };
        assert_eq!(gpu, host);

        x[7] = 0.0;
        x[2] = 50.0;
        x[9] = 50.0;
        // Match Metal/`>` scan (first index wins). Rust `Iterator::max_by`
        // on Equal keeps the later element — do not use it for the oracle.
        let mut host_tie = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in x.iter().enumerate() {
            if v > best {
                best = v;
                host_tie = i;
            }
        }
        assert_eq!(host_tie, 2);
        let x_buf = upload(device, &x).unwrap();
        // Shared-mode upload must be visible to the host immediately.
        let uploaded = read_f32(&x_buf, x.len());
        assert_eq!(uploaded[2], 50.0);
        assert_eq!(uploaded[9], 50.0);
        let cmd = shared.queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        encode_argmax(&enc, device, &x_buf, &idx_buf, x.len() as u32).unwrap();
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let ptr = idx_buf.contents();
        let gpu = unsafe { *(ptr.as_ptr() as *const u32) as usize };
        assert_eq!(
            gpu, host_tie,
            "gpu={gpu} host={host_tie}; x[2]={} x[9]={}",
            x[2], x[9]
        );
    }
}
