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

/// One RMSNorm kernel for every row shape the stacks encode, templated
/// on whether a residual add precedes the norm (ggml F=3) and on the
/// output type (f32, or f16 straight into the prefill GEMM's staging).
/// Row `r` is threadgroup `r`, so the single-row decode call is the
/// batch call with one threadgroup.
///
/// WHY THE FLOAT4 LOOP IS THE WHOLE POINT. The previous kernels read
/// one float per thread per iteration with a runtime trip count, and
/// an Apple GPU issues those loads in order: every iteration paid a
/// full memory latency, ~0.4 us, so a 2304-wide norm at 256 threads
/// was 9 iterations x 2 loops x 0.4 us = 14.2 us serialized
/// (`crate::kernel_bench`) against llama.cpp's 3.2 us for the same
/// op. float4 loads and a threadgroup sized to the row make it one or
/// two iterations. The scalar path stays for a width that is not a
/// multiple of four, where a `float4*` into row `r` would not be
/// 16-byte aligned, and every caller's byte offset is `row * n * 4`,
/// so `n % 4 == 0` is the one alignment fact that decides both.
const RMS_NORM_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ADD: `h[row] += add[row]` first (h is then the residual stream and
// is written back); otherwise `h` is read only and `add` is unused.
// OUT4/OUT: float4/float, or half4/half for the prefill GEMM.
template <bool ADD, typename OUT4, typename OUT>
kernel void rms_norm_rows(
    device float* h [[buffer(0)]],
    device const float* add [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device OUT* out [[buffer(3)]],
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
    device OUT* orow = out + row * n;
    const bool vec4 = (n % 4u) == 0u;
    float partial = 0.0f;
    if (vec4) {
        device float4* h4 = (device float4*)hr;
        device const float4* a4 = (device const float4*)ar;
        const uint n4 = n / 4u;
        for (uint i = tid; i < n4; i += tg) {
            float4 v = h4[i];
            if (ADD) {
                v += a4[i];
                h4[i] = v;
            }
            partial += dot(v, v);
        }
    } else {
        for (uint i = tid; i < n; i += tg) {
            float v = hr[i];
            if (ADD) {
                v += ar[i];
                hr[i] = v;
            }
            partial += v * v;
        }
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
    if (vec4) {
        device const float4* h4 = (device const float4*)hr;
        device const float4* w4 = (device const float4*)weight;
        device OUT4* o4 = (device OUT4*)orow;
        const uint n4 = n / 4u;
        for (uint i = tid; i < n4; i += tg) {
            o4[i] = OUT4(h4[i] * inv_rms * w4[i]);
        }
    } else {
        for (uint i = tid; i < n; i += tg) {
            orow[i] = OUT(hr[i] * inv_rms * weight[i]);
        }
    }
}

typedef decltype(rms_norm_rows<false, float4, float>) rms_norm_f32_t;
typedef decltype(rms_norm_rows<false, half4, half>) rms_norm_f16_t;
template [[host_name("rms_norm_f32")]] kernel rms_norm_f32_t rms_norm_rows<false, float4, float>;
template [[host_name("add_rms_norm_f32")]] kernel rms_norm_f32_t rms_norm_rows<true, float4, float>;
template [[host_name("rms_norm_f32_to_f16")]] kernel rms_norm_f16_t rms_norm_rows<false, half4, half>;
template [[host_name("add_rms_norm_f32_to_f16")]] kernel rms_norm_f16_t rms_norm_rows<true, half4, half>;
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
    encode_norm_rows(
        encoder,
        device,
        NormRows {
            h: x,
            h_off_bytes: x_off_bytes,
            add: None,
            weight,
            out,
            out_off_bytes,
            out_half: false,
            n,
            rows: 1,
            eps,
        },
    )
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
    encode_norm_rows(
        encoder,
        device,
        NormRows {
            h: x,
            h_off_bytes: 0,
            add: None,
            weight,
            out,
            out_off_bytes: 0,
            out_half: false,
            n,
            rows: batch,
            eps,
        },
    )
}

/// Batched RMSNorm writing `half` rows (prefill → `mul_mm_sg_f16`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rms_norm_f32_to_f16_batch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    batch: u32,
    eps: f32,
) -> Result<(), MetalError> {
    encode_norm_rows(
        encoder,
        device,
        NormRows {
            h: x,
            h_off_bytes: 0,
            add: None,
            weight,
            out,
            out_off_bytes: 0,
            out_half: true,
            n,
            rows: batch,
            eps,
        },
    )
}

/// `h += add`, then `out = rms_norm(h) * weight`. One dispatch replaces
/// [`crate::elem::encode_vec_add`] + [`encode_rms_norm`].
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
    encode_add_rms_norm_batch(encoder, device, h, add, weight, out, n, 1, eps)
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
    encode_norm_rows(
        encoder,
        device,
        NormRows {
            h,
            h_off_bytes: 0,
            add: Some(add),
            weight,
            out,
            out_off_bytes: 0,
            out_half: false,
            n,
            rows: batch,
            eps,
        },
    )
}

/// [`encode_add_rms_norm_batch`] storing `half` — the fused residual add +
/// FFN RMSNorm + f32→f16 staging convert the dense prefill layer needs
/// before `mul_mm_sg_f16`. `h` is still updated in f32 (it is the residual
/// stream); only `out` is half. Saves one dispatch and one barrier/layer.
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
    encode_norm_rows(
        encoder,
        device,
        NormRows {
            h,
            h_off_bytes: 0,
            add: Some(add),
            weight,
            out,
            out_off_bytes: 0,
            out_half: true,
            n,
            rows: batch,
            eps,
        },
    )
}

/// One RMSNorm dispatch over `rows` contiguous rows of `n`, as the
/// eight public encoders above spell it.
struct NormRows<'a> {
    h: &'a ProtocolObject<dyn MTLBuffer>,
    h_off_bytes: usize,
    /// `Some`: `h += add` before the norm, and `h` is written back.
    add: Option<&'a ProtocolObject<dyn MTLBuffer>>,
    weight: &'a ProtocolObject<dyn MTLBuffer>,
    out: &'a ProtocolObject<dyn MTLBuffer>,
    out_off_bytes: usize,
    /// `out` is `half` rows rather than `f32`.
    out_half: bool,
    n: u32,
    rows: u32,
    eps: f32,
}

/// The kernel entry point for a (residual add, output type) pair. One
/// table, read by the encoder and by the prefill warm-up.
fn norm_kernel_name(add: bool, out_half: bool) -> &'static str {
    match (add, out_half) {
        (false, false) => "rms_norm_f32",
        (true, false) => "add_rms_norm_f32",
        (false, true) => "rms_norm_f32_to_f16",
        (true, true) => "add_rms_norm_f32_to_f16",
    }
}

/// Threads per row: enough that the float4 loop runs once or twice,
/// never fewer than a simdgroup, never more than Metal allows. `n` here
/// is the element count the loop strides over (float4s, or floats on
/// the scalar path).
fn norm_threadgroup(n: u32) -> u32 {
    let per_thread_units = if n.is_multiple_of(4) { n / 4 } else { n };
    per_thread_units
        .div_ceil(2)
        .next_power_of_two()
        .clamp(32, 1024)
}

fn encode_norm_rows(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    r: NormRows<'_>,
) -> Result<(), MetalError> {
    if r.rows == 0 {
        return Ok(());
    }
    let name = norm_kernel_name(r.add.is_some(), r.out_half);
    let pipe = ensure_pipeline(device, RMS_NORM_KERNEL_SRC, name)?;
    encoder.setComputePipelineState(&pipe.0);
    let tg = norm_threadgroup(r.n);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(r.h), r.h_off_bytes, 0);
        // Without a residual the kernel never reads index 1; `h` is a
        // valid buffer to leave bound there.
        encoder.setBuffer_offset_atIndex(Some(r.add.unwrap_or(r.h)), r.h_off_bytes, 1);
        encoder.setBuffer_offset_atIndex(Some(r.weight), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(r.out), r.out_off_bytes, 3);
        let mut n_u = r.n;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut n_u as *mut u32 as *mut _).unwrap(),
            4,
            4,
        );
        let mut eps_f = r.eps;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut eps_f as *mut f32 as *mut _).unwrap(),
            4,
            5,
        );
        // One float per simdgroup (simd_sum path).
        encoder.setThreadgroupMemoryLength_atIndex(((tg as usize) / 32) * 4, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: r.rows as usize,
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
    for add in [false, true] {
        for out_half in [false, true] {
            ensure_pipeline(device, RMS_NORM_KERNEL_SRC, norm_kernel_name(add, out_half))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elem::tests::{alloc, read_f32, upload};
    use crate::gpu::shared_metal;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

    fn read_f16_as_f32(buf: &ProtocolObject<dyn MTLBuffer>, n: usize) -> Vec<f32> {
        let ptr = buf.contents();
        let bits = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const u16, n) };
        bits.iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect()
    }

    /// One kernel serves eight encoders, and the width decides which
    /// of its two loops runs: float4 when `n % 4 == 0`, scalar
    /// otherwise, because a `float4*` into row `r` of a width that is
    /// not a multiple of four is not 16-byte aligned. Every (width,
    /// rows, residual, output type) the stacks reach is checked
    /// against a CPU reference here, including widths that take the
    /// scalar path, widths wide enough that the float4 loop runs more
    /// than once at the largest threadgroup, and a residual whose
    /// write-back into `h` is part of the contract.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn every_norm_shape_matches_the_cpu_reference() {
        let shared = shared_metal().expect("metal");
        let device = &shared.device;
        let eps = 1e-6f32;
        for &n in &[2304usize, 8192, 130, 37, 4] {
            for &rows in &[1usize, 3] {
                for add in [false, true] {
                    for half_out in [false, true] {
                        let h0: Vec<f32> = (0..n * rows)
                            .map(|i| ((i as f32) * 0.37).sin() * 3.0)
                            .collect();
                        let a: Vec<f32> =
                            (0..n * rows).map(|i| ((i as f32) * 0.11).cos()).collect();
                        let w: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
                        let mut h_cpu = h0.clone();
                        if add {
                            for (h, a) in h_cpu.iter_mut().zip(a.iter()) {
                                *h += *a;
                            }
                        }
                        let mut want = vec![0.0f32; n * rows];
                        for r in 0..rows {
                            let row = &h_cpu[r * n..(r + 1) * n];
                            let ms = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
                            let inv = 1.0 / (ms + eps).sqrt();
                            for i in 0..n {
                                want[r * n + i] = row[i] * inv * w[i];
                            }
                        }
                        let h_buf = upload(device, &h0).unwrap();
                        let a_buf = upload(device, &a).unwrap();
                        let w_buf = upload(device, &w).unwrap();
                        let out = alloc(device, n * rows).unwrap();
                        let cmd = shared.queue.commandBuffer().unwrap();
                        let enc = cmd.computeCommandEncoder().unwrap();
                        let r = NormRows {
                            h: &h_buf,
                            h_off_bytes: 0,
                            add: add.then_some(&*a_buf),
                            weight: &w_buf,
                            out: &out,
                            out_off_bytes: 0,
                            out_half: half_out,
                            n: n as u32,
                            rows: rows as u32,
                            eps,
                        };
                        encode_norm_rows(&enc, device, r).unwrap();
                        enc.endEncoding();
                        cmd.commit();
                        cmd.waitUntilCompleted();
                        let got = if half_out {
                            read_f16_as_f32(&out, n * rows)
                        } else {
                            read_f32(&out, n * rows)
                        };
                        let tol = if half_out { 2e-3 } else { 1e-5 };
                        for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
                            assert!(
                                (g - e).abs() <= tol * e.abs().max(1.0),
                                "n={n} rows={rows} add={add} half={half_out} elem {i}: got {g} want {e}"
                            );
                        }
                        let h_after = read_f32(&h_buf, n * rows);
                        for (i, (g, e)) in h_after.iter().zip(h_cpu.iter()).enumerate() {
                            assert!(
                                (g - e).abs() <= 1e-6 * e.abs().max(1.0),
                                "n={n} rows={rows} add={add}: residual stream elem {i}: got {g} want {e}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The threadgroup is sized so the float4 loop runs at most twice
    /// and never below a simdgroup or above Metal's limit.
    #[test]
    fn norm_threadgroup_is_a_simdgroup_multiple_that_runs_the_loop_at_most_twice() {
        assert_eq!(
            norm_threadgroup(2304),
            512,
            "576 float4s over 512 threads: 2 passes"
        );
        assert_eq!(
            norm_threadgroup(4096),
            512,
            "1024 float4s: exactly 2 passes"
        );
        assert_eq!(norm_threadgroup(8192), 1024, "capped at Metal's maximum");
        assert_eq!(norm_threadgroup(64), 32, "never below one simdgroup");
        assert_eq!(
            norm_threadgroup(130),
            128,
            "scalar path: 130 floats over 128"
        );
        for n in [4, 37, 130, 2304, 3072, 8192, 16384] {
            let tg = norm_threadgroup(n);
            assert!(
                tg.is_power_of_two() && (32..=1024).contains(&tg),
                "n={n}: tg={tg}"
            );
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
}
