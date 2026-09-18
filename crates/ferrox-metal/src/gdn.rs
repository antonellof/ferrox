//! The gated delta-net recurrence on the device.
//!
//! `ferrox_core::gdn::delta_step` is the definition, and this is the
//! same arithmetic per head: decay the state, predict `k` through it,
//! scale the error by beta, apply the rank-one update, read it out with
//! the scaled query. It is here because a Bonsai-2-27B decode token
//! spends 83% of its wall clock waiting on command buffers, and the
//! recurrence sitting on the HOST is what splits a recurrent layer into
//! three submissions: project on the GPU, come back for the state, go
//! out again for `ssm_out`.
//!
//! One threadgroup per head, `head_dim` threads. The state row a thread
//! owns is `S` floats it reads twice and writes once, so the whole step
//! is two passes over the state with a threadgroup reduction between
//! them, which is what the CPU version does per row with the loop
//! order swapped.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLSize,
};

use crate::gpu::{ensure_pipeline, MetalError};

/// How a value head picks its key head, the two orders llama.cpp uses
/// (`ferrox_core::gdn::HeadMap`). Passed as a flag rather than baked in
/// because the two families disagree and a wrong map is a wrong model
/// that still produces fluent text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadMapKind {
    /// `h % n_k`: Qwen3.5, whose converter reorders V heads for
    /// `ggml_repeat`.
    Tiled,
    /// `h / (n_v / n_k)`: Qwen3-Next.
    Grouped,
}

impl HeadMapKind {
    fn flag(self) -> u32 {
        match self {
            HeadMapKind::Tiled => 0,
            HeadMapKind::Grouped => 1,
        }
    }
}

/// The shapes one dispatch needs, mirroring `ferrox_core::gdn::DeltaDims`.
#[derive(Debug, Clone, Copy)]
pub struct DeltaShape {
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub head_dim: usize,
    pub map: HeadMapKind,
}

/// `head_dim` threads per head, capped at what a threadgroup holds. A
/// head wider than this would need a strided inner loop, which the
/// kernel deliberately does not have: it would be a second shape with
/// no caller (Bonsai and Qwen3-Next are both 128).
pub const MAX_HEAD_DIM: usize = 1024;

pub const GDN_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One threadgroup per value head, one thread per state ROW (the value
// index), each owning `S` contiguous state floats.
//
//   row_j = state[h][j][:]            (S floats, key-indexed)
//   pred_j = dot(row_j * decay, k)
//   d_j    = (v_j - pred_j) * beta
//   row_j += k * d_j
//   out_j  = dot(row_j, q) * scale
//
// which is `ferrox_core::gdn::delta_step` with the two passes kept and
// the per-row reductions done in registers.
kernel void gdn_delta_step(
    device float* state [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device const float* g [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint4& dims [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]
) {
    // dims: x = n_k_heads, y = n_v_heads, z = head_dim, w = map flag.
    const uint n_k = dims.x;
    const uint n_v = dims.y;
    const uint S = dims.z;
    const uint h = tgid;
    if (h >= n_v) {
        return;
    }
    const uint kh = (dims.w == 0u) ? (h % n_k) : (h / (n_v / n_k));

    threadgroup float qs[1024];
    threadgroup float ks[1024];
    for (uint i = tid; i < S; i += tcount) {
        qs[i] = q[kh * S + i];
        ks[i] = k[kh * S + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float decay = exp(g[h]);
    const float b = beta[h];
    const float scale = 1.0f / sqrt(float(S));
    device float* st = state + (size_t)h * S * S;

    for (uint j = tid; j < S; j += tcount) {
        device float* row = st + (size_t)j * S;
        // Pass one: decay in place and predict.
        float pred = 0.0f;
        for (uint i = 0; i < S; ++i) {
            const float r = row[i] * decay;
            row[i] = r;
            pred += r * ks[i];
        }
        const float d = (v[h * S + j] - pred) * b;
        // Pass two: the rank-one update, then the read-out.
        float o = 0.0f;
        for (uint i = 0; i < S; ++i) {
            const float r = row[i] + ks[i] * d;
            row[i] = r;
            o += r * qs[i];
        }
        out[h * S + j] = o * scale;
    }
}
"#;

/// The gated output norm that follows the recurrence: per head,
/// `rms_norm(o, weight) * silu(z)` (`qwen35.cpp:311-313`, llama.cpp's
/// `build_norm_gated`). One threadgroup per head, the sum of squares
/// reduced in threadgroup memory.
///
/// It lives beside the recurrence because the two are always adjacent
/// and the vector between them is the only reason the host would see
/// the step's output at all.
pub const GDN_GATED_NORM_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gdn_gated_norm(
    device const float* o [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint2& dims [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]
) {
    const uint S = dims.x;
    const uint h = tgid;
    threadgroup float partial[256];
    const uint base = h * S;
    float acc = 0.0f;
    for (uint i = tid; i < S; i += tcount) {
        const float v = o[base + i];
        acc += v * v;
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tcount >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / sqrt(partial[0] / float(S) + eps);
    for (uint i = tid; i < S; i += tcount) {
        const float normed = o[base + i] * inv * weight[i];
        const float zi = z[base + i];
        y[base + i] = normed * (zi / (1.0f + exp(-zi)));
    }
}
"#;

/// Encodes the gated norm over `n_v_heads` heads of `head_dim`.
#[allow(clippy::too_many_arguments)]
pub fn encode_gated_norm(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    o: &ProtocolObject<dyn MTLBuffer>,
    z: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
) -> Result<(), MetalError> {
    if head_dim == 0 || n_heads == 0 {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(device, GDN_GATED_NORM_SRC, "gdn_gated_norm")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        for (idx, buf) in [o, z, weight, y].into_iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buf), 0, idx);
        }
        let mut dims: [u32; 2] = [head_dim as u32, n_heads as u32];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 8, 4);
        let mut e = eps;
        encoder.setBytes_length_atIndex(NonNull::new(&mut e as *mut f32 as *mut _).unwrap(), 4, 5);
    }
    // A power of two so the reduction's halving loop covers the group,
    // and at most the 256 floats of scratch the kernel declares.
    let threads = head_dim.next_power_of_two().min(256);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_heads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Encodes one token's delta step. Every buffer is the caller's, and
/// `state` is read AND written, so a caller that keeps it across tokens
/// hands the same buffer back.
#[allow(clippy::too_many_arguments)]
pub fn encode_delta_step(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    shape: DeltaShape,
    state: &ProtocolObject<dyn MTLBuffer>,
    q: &ProtocolObject<dyn MTLBuffer>,
    k: &ProtocolObject<dyn MTLBuffer>,
    v: &ProtocolObject<dyn MTLBuffer>,
    g: &ProtocolObject<dyn MTLBuffer>,
    beta: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
) -> Result<(), MetalError> {
    if shape.head_dim == 0
        || shape.head_dim > MAX_HEAD_DIM
        || shape.n_k_heads == 0
        || shape.n_v_heads == 0
        || !shape.n_v_heads.is_multiple_of(shape.n_k_heads)
    {
        return Err(MetalError::CommandFailed);
    }
    let pipe = ensure_pipeline(device, GDN_KERNEL_SRC, "gdn_delta_step")?;
    encoder.setComputePipelineState(&pipe.0);
    unsafe {
        for (idx, buf) in [state, q, k, v, g, beta, out].into_iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buf), 0, idx);
        }
        let mut dims: [u32; 4] = [
            shape.n_k_heads as u32,
            shape.n_v_heads as u32,
            shape.head_dim as u32,
            shape.map.flag(),
        ];
        encoder.setBytes_length_atIndex(NonNull::new(dims.as_mut_ptr() as *mut _).unwrap(), 16, 7);
    }
    // One thread per state row, capped at the pipeline's own maximum so
    // a 1024-wide head still dispatches (the row loop strides).
    let threads = shape.head_dim.min(pipe.0.maxTotalThreadsPerThreadgroup());
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: shape.n_v_heads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// One token's delta step end to end: uploads the operands, runs the
/// kernel, reads the state and the output back.
///
/// The standalone entry, used by the test that pins this against
/// `ferrox_core::gdn::delta_step` and by a caller that has no device
/// buffers of its own yet. A caller inside a larger command buffer
/// wants [`encode_delta_step`] instead, which is the point of the
/// kernel: no round trip.
pub fn launch_delta_step(
    shape: DeltaShape,
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>, MetalError> {
    let shared = crate::gpu::shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;
    let upload = |xs: &[f32]| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let mut owned = xs.to_vec();
        // SAFETY: `owned` is live for this call and `newBufferWithBytes`
        // copies the bytes.
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(owned.as_mut_ptr() as *mut _).ok_or(MetalError::BufferAllocFailed)?,
                std::mem::size_of_val(owned.as_slice()),
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocFailed)
    };
    let state_buf = upload(state)?;
    let (q_buf, k_buf, v_buf) = (upload(q)?, upload(k)?, upload(v)?);
    let (g_buf, beta_buf) = (upload(g)?, upload(beta)?);
    let out_len = shape.n_v_heads * shape.head_dim;
    let out_buf = device
        .newBufferWithLength_options(
            out_len * 4,
            objc2_metal::MTLResourceOptions::StorageModeShared,
        )
        .ok_or(MetalError::BufferAllocFailed)?;

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_delta_step(
        &encoder, device, shape, &state_buf, &q_buf, &k_buf, &v_buf, &g_buf, &beta_buf, &out_buf,
    )?;
    encoder.endEncoding();
    let clock = crate::timing::SubmitClock::start();
    crate::timing::commit_wait_note(&cmd_buf, "gdn-delta", 64, clock);

    // SAFETY: both buffers are shared storage of exactly these lengths,
    // written by a kernel this call has waited for.
    unsafe {
        let st =
            std::slice::from_raw_parts(state_buf.contents().as_ptr() as *const f32, state.len());
        state.copy_from_slice(st);
        let o = std::slice::from_raw_parts(out_buf.contents().as_ptr() as *const f32, out_len);
        Ok(o.to_vec())
    }
}

/// One token's recurrent tail in ONE command buffer: the delta step,
/// the gated output norm, the rotation `ssm_out` was folded with, and
/// `ssm_out` itself.
///
/// This is the submission the host used to make three of, because the
/// recurrence sat between two matvecs. `state` is read and written in
/// place; `out_proj` is the matvec descriptor the caller would have
/// passed to `launch_matvec_fused_folded`.
#[allow(clippy::too_many_arguments)]
pub fn launch_gdn_tail(
    shape: DeltaShape,
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    z: &[f32],
    norm_weight: &[f32],
    eps: f32,
    out_proj: &crate::gpu::MatvecLaunch<'_>,
    fold: Option<&crate::hadamard::FoldPlan<'_>>,
) -> Result<Vec<f32>, MetalError> {
    let value_dim = shape.n_v_heads * shape.head_dim;
    if z.len() != value_dim || norm_weight.len() != shape.head_dim {
        return Err(MetalError::CommandFailed);
    }
    let shared = crate::gpu::shared_metal()?;
    let device = &shared.device;
    let queue = &shared.queue;
    let upload = |xs: &[f32]| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let mut owned = xs.to_vec();
        // SAFETY: `owned` outlives the call and the buffer copies it.
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(owned.as_mut_ptr() as *mut _).ok_or(MetalError::BufferAllocFailed)?,
                std::mem::size_of_val(owned.as_slice()),
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocFailed)
    };
    let scratch = |n: usize| -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        device
            .newBufferWithLength_options(n * 4, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocFailed)
    };

    let state_buf = upload(state)?;
    let (q_buf, k_buf, v_buf) = (upload(q)?, upload(k)?, upload(v)?);
    let (g_buf, beta_buf, z_buf) = (upload(g)?, upload(beta)?, upload(z)?);
    let norm_buf = upload(norm_weight)?;
    let o_buf = scratch(value_dim)?;
    let y_buf = scratch(value_dim)?;
    let out_buf = scratch(out_proj.rows)?;
    let weights_buf = crate::gpu::resident_weight_buffer(device, out_proj.weights)?;
    let signs_buf = match fold.and_then(|p| p.signs) {
        None => None,
        Some(signs) => {
            // SAFETY: a `&[f32]` viewed as its own bytes, for a read.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    signs.as_ptr() as *const u8,
                    std::mem::size_of_val(signs),
                )
            };
            Some(crate::gpu::resident_weight_buffer(device, bytes)?)
        }
    };

    let cmd_buf = queue.commandBuffer().ok_or(MetalError::CommandFailed)?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(MetalError::CommandFailed)?;
    encode_delta_step(
        &encoder, device, shape, &state_buf, &q_buf, &k_buf, &v_buf, &g_buf, &beta_buf, &o_buf,
    )?;
    encode_gated_norm(
        &encoder,
        device,
        shape.n_v_heads,
        shape.head_dim,
        eps,
        &o_buf,
        &z_buf,
        &norm_buf,
        &y_buf,
    )?;
    if let Some(plan) = fold {
        plan.check(value_dim)?;
        crate::hadamard::encode_fold(
            &encoder,
            device,
            &y_buf,
            value_dim,
            plan,
            signs_buf.as_ref().map(|b| &*b.buffer),
        )?;
    }
    crate::gpu::encode_matvec(&encoder, device, out_proj, &weights_buf, &y_buf, &out_buf)?;
    encoder.endEncoding();
    let clock = crate::timing::SubmitClock::start();
    crate::timing::commit_wait_note(&cmd_buf, "gdn-tail", 32, clock);

    // SAFETY: both are shared-storage buffers of exactly these lengths,
    // written by kernels this call has waited for.
    unsafe {
        let st =
            std::slice::from_raw_parts(state_buf.contents().as_ptr() as *const f32, state.len());
        state.copy_from_slice(st);
        let o =
            std::slice::from_raw_parts(out_buf.contents().as_ptr() as *const f32, out_proj.rows);
        Ok(o.to_vec())
    }
}
