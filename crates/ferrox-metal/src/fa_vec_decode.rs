//! The FA-vec decode attention kernels: one query row against an f16
//! KV cache, one threadgroup per head, NSG simdgroups each walking
//! every NSG-th tile of 32 keys with an online softmax, merged at the
//! end. Specialised per head width (64 / 96 / 128 / 256) because the
//! lane-to-slice mapping differs.
//!
//! Split out of `attn.rs` along the seam the Gemma-2 decode work
//! touches next: per token, these kernels cost a Gemma-2-2B layer
//! ~3x what llama.cpp's `flash_attn_ext_vec` does at the same shape.

use crate::dispatch::dispatch_counted;
use crate::gpu::{ensure_pipeline, MetalError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLSize};
use std::ptr::NonNull;

/// llama.cpp-style FA-vec decode for **head_dim=128**, f16 KV, NE=1, C=32.
/// One TG per head; NSG simdgroups each own every NSG-th KV tile, then
/// online-softmax merge. Replaces the old FA_VEC that recomputed V ×32.
const GQA_DECODE_FA_VEC_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_decode_fa_vec(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_start [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Specialized for D=128 (host only dispatches when head_dim==128).
    constexpr uint D = 128u;
    constexpr uint D4 = 32u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    // Per-SG floats: C scores + D output.
    constexpr uint SG_F = C + D;

    if (h >= n_heads || seq_len == 0u || head_dim != D) return;

    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + h * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    so4[tiisg] = float4(0.0f);
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    // Each SG walks KV tiles: ic0 = sgitg, sgitg+nsg, ...
    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = kv_start + ic0 * C;
        if (ic >= seq_len) break;
        uint chunk = min(C, seq_len - ic);

        // Q·K for all C positions: lane `ii` owns float4-slice `ii` of the head;
        // after simd_sum, every lane holds the full score for each cc.
        float scores[C];
        for (uint cc = 0; cc < C; cc++) {
            scores[cc] = -INFINITY;
        }
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float partial = dot(sq4[tiisg], float4(k4[tiisg]));
            float sc = simd_sum(partial) * scale;
            if (softcap > 0.0f) {
                sc = softcap * tanh(sc / softcap);
            }
            scores[cc] = sc;
        }

        // Online softmax over this tile (one score per lane).
        float s_lane = (tiisg < chunk) ? scores[tiisg] : -INFINITY;
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        so4[tiisg] *= ms;
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O += P · V  (lane owns float4-slice tiisg of the output)
        float4 lo = float4(0.0f);
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* v4 =
                (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            lo += float4(v4[tiisg]) * ss[cc];
        }
        so4[tiisg] += lo;
    }

    // Publish S,M for cross-SG reduce (reuse ss[0], ss[1]).
    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SG online-softmax merge.
    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        float inv = (ss0[0] == 0.0f) ? 0.0f : (1.0f / ss0[0]);
        device float4* out4 = (device float4*)(out + h * D);
        out4[tiisg] = so0[tiisg] * inv;
    }
}
"#;

/// llama.cpp-style FA-vec decode for **head_dim=64**, f16 KV, NE=2, C=32.
/// Same tile/merge structure as the d=128 kernel, but D4=16 float4
/// slices only cover half a simdgroup — so each warp processes **two**
/// KV positions per pass (half-warp `ty=0` gets even `cc`, `ty=1` odd),
/// with a 16-lane shuffle-xor dot reduce and a cross-half xor-16 merge
/// of the V accumulators. This keeps all 32 lanes busy where a naive
/// D4=16 port would idle half the warp (TinyLlama / Llama-3.2-1B are
/// d=64).
const GQA_DECODE_FA_VEC_D64_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_decode_fa_vec_d64(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_start [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Specialized for D=64 (host only dispatches when head_dim==64).
    constexpr uint D = 64u;
    constexpr uint D4 = 16u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    // Per-SG floats: C scores + D output.
    constexpr uint SG_F = C + D;

    if (h >= n_heads || seq_len == 0u || head_dim != D) return;

    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;
    const uint tx = tiisg % D4; // float4 slice of the head
    const uint ty = tiisg / D4; // 0/1: token parity within a warp pass

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + h * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    if (tiisg < D4) {
        so4[tiisg] = float4(0.0f);
    }
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    // Each SG walks KV tiles: ic0 = sgitg, sgitg+nsg, ...
    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = kv_start + ic0 * C;
        if (ic >= seq_len) break;
        uint chunk = min(C, seq_len - ic);

        // Q·K, two positions per warp pass: half-warp `ty` owns token
        // ic+cc; 16-lane xor reduce leaves the full dot in every lane
        // of that half; lane tx==0 publishes it.
        for (uint cc = ty; cc < chunk; cc += 2u) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float p = dot(sq4[tx], float4(k4[tx]));
            p += simd_shuffle_xor(p, 8u);
            p += simd_shuffle_xor(p, 4u);
            p += simd_shuffle_xor(p, 2u);
            p += simd_shuffle_xor(p, 1u);
            if (tx == 0u) {
                float sc = p * scale;
                if (softcap > 0.0f) {
                    sc = softcap * tanh(sc / softcap);
                }
                ss[cc] = sc;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax over this tile (one score per lane).
        float s_lane = (tiisg < chunk) ? ss[tiisg] : -INFINITY;
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        if (tiisg < D4) {
            so4[tiisg] *= ms;
        }
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O += P · V, two positions per warp pass; merge the two token
        // halves with an xor-16 shuffle before accumulating.
        float4 lo = float4(0.0f);
        for (uint cc = ty; cc < chunk; cc += 2u) {
            device const half4* v4 =
                (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            lo += float4(v4[tx]) * ss[cc];
        }
        lo += simd_shuffle_xor(lo, 16u);
        if (ty == 0u) {
            so4[tx] += lo;
        }
    }

    // Publish S,M for cross-SG reduce (reuse ss[0], ss[1]).
    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SG online-softmax merge.
    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            if (tiisg < D4) {
                so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u && tiisg < D4) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        float inv = (ss0[0] == 0.0f) ? 0.0f : (1.0f / ss0[0]);
        device float4* out4 = (device float4*)(out + h * D);
        out4[tiisg] = so0[tiisg] * inv;
    }
}
"#;

/// FA-vec decode for **head_dim=96** (Phi-3-mini). D4=24 float4 slices —
/// lanes `tiisg < 24` own Q/K/V float4 work; remaining warp lanes contribute
/// zeros to the simd_sum score reduce so the tile/merge structure stays
/// identical to the d=128 kernel.
const GQA_DECODE_FA_VEC_D96_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_decode_fa_vec_d96(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_start [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = 96u;
    constexpr uint D4 = 24u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    constexpr uint SG_F = C + D;

    if (h >= n_heads || seq_len == 0u || head_dim != D) return;

    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + h * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    if (tiisg < D4) {
        so4[tiisg] = float4(0.0f);
    }
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = kv_start + ic0 * C;
        if (ic >= seq_len) break;
        uint chunk = min(C, seq_len - ic);

        float scores[C];
        for (uint cc = 0; cc < C; cc++) {
            scores[cc] = -INFINITY;
        }
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float partial = (tiisg < D4) ? dot(sq4[tiisg], float4(k4[tiisg])) : 0.0f;
            float sc = simd_sum(partial) * scale;
            if (softcap > 0.0f) {
                sc = softcap * tanh(sc / softcap);
            }
            scores[cc] = sc;
        }

        float s_lane = (tiisg < chunk) ? scores[tiisg] : -INFINITY;
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        if (tiisg < D4) {
            so4[tiisg] *= ms;
        }
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo = float4(0.0f);
        if (tiisg < D4) {
            for (uint cc = 0; cc < chunk; cc++) {
                device const half4* v4 =
                    (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
                lo += float4(v4[tiisg]) * ss[cc];
            }
            so4[tiisg] += lo;
        }
    }

    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            if (tiisg < D4) {
                so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u && tiisg < D4) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        float inv = (ss0[0] == 0.0f) ? 0.0f : (1.0f / ss0[0]);
        device float4* out4 = (device float4*)(out + h * D);
        out4[tiisg] = so0[tiisg] * inv;
    }
}
"#;

/// FA-vec decode for **head_dim=256** (Gemma-3). D4=64 float4 slices —
/// each warp lane owns two slices (`tiisg` and `tiisg+32`) so the simd
/// reduce still covers the full head.
const GQA_DECODE_FA_VEC_D256_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_decode_fa_vec_d256(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n_heads [[buffer(4)]],
    constant uint& n_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_start [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    constexpr uint D = 256u;
    constexpr uint D4 = 64u;
    constexpr uint C = 32u;
    constexpr uint NW = 32u;
    constexpr uint SG_F = C + D;

    if (h >= n_heads || seq_len == 0u || head_dim != D) return;

    const uint tiisg = tid % NW;
    const uint sgitg = tid / NW;
    const uint nsg = tg / NW;

    threadgroup float4* sq4 = (threadgroup float4*)shared;
    threadgroup float* ss = shared + D + sgitg * SG_F;
    threadgroup float4* so4 = (threadgroup float4*)(ss + C);

    uint group_size = n_heads / max(n_kv_heads, 1u);
    uint kv_h = h / max(group_size, 1u);
    float scale = 1.0f / sqrt(float(D));

    device const float4* q4 = (device const float4*)(q + h * D);
    for (uint i = tid; i < D4; i += tg) {
        sq4[i] = q4[i];
    }
    so4[tiisg] = float4(0.0f);
    so4[tiisg + NW] = float4(0.0f);
    ss[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -INFINITY;

    for (uint ic0 = sgitg; ; ic0 += nsg) {
        uint ic = kv_start + ic0 * C;
        if (ic >= seq_len) break;
        uint chunk = min(C, seq_len - ic);

        float scores[C];
        for (uint cc = 0; cc < C; cc++) {
            scores[cc] = -INFINITY;
        }
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* k4 =
                (device const half4*)(k_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            float partial = 0.0f;
            for (uint i = tiisg; i < D4; i += NW) {
                partial += dot(sq4[i], float4(k4[i]));
            }
            float sc = simd_sum(partial) * scale;
            if (softcap > 0.0f) {
                sc = softcap * tanh(sc / softcap);
            }
            scores[cc] = sc;
        }

        float s_lane = (tiisg < chunk) ? scores[tiisg] : -INFINITY;
        float M2 = simd_max(max(M, s_lane));
        float ms = (M == -INFINITY) ? 0.0f : exp(M - M2);
        float vs = (s_lane == -INFINITY) ? 0.0f : exp(s_lane - M2);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        so4[tiisg] *= ms;
        so4[tiisg + NW] *= ms;
        M = M2;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo0 = float4(0.0f);
        float4 lo1 = float4(0.0f);
        for (uint cc = 0; cc < chunk; cc++) {
            device const half4* v4 =
                (device const half4*)(v_cache + ((ic + cc) * n_kv_heads + kv_h) * D);
            lo0 += float4(v4[tiisg]) * ss[cc];
            lo1 += float4(v4[tiisg + NW]) * ss[cc];
        }
        so4[tiisg] += lo0;
        so4[tiisg + NW] += lo1;
    }

    if (tiisg == 0u) {
        ss[0] = S;
        ss[1] = M;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = nsg >> 1; r > 0u; r >>= 1) {
        if (sgitg < r) {
            threadgroup float* ss0 = shared + D + sgitg * SG_F;
            threadgroup float* ss1 = shared + D + (sgitg + r) * SG_F;
            threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
            threadgroup float4* so1 = (threadgroup float4*)(ss1 + C);
            float S0 = ss0[0];
            float S1 = ss1[0];
            float M0 = ss0[1];
            float M1 = ss1[1];
            float Mn = max(M0, M1);
            float a0 = (M0 == -INFINITY) ? 0.0f : exp(M0 - Mn);
            float a1 = (M1 == -INFINITY) ? 0.0f : exp(M1 - Mn);
            if (tiisg == 0u) {
                ss0[0] = S0 * a0 + S1 * a1;
                ss0[1] = Mn;
            }
            so0[tiisg] = so0[tiisg] * a0 + so1[tiisg] * a1;
            so0[tiisg + NW] = so0[tiisg + NW] * a0 + so1[tiisg + NW] * a1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0u) {
        threadgroup float* ss0 = shared + D;
        threadgroup float4* so0 = (threadgroup float4*)(ss0 + C);
        float inv = (ss0[0] == 0.0f) ? 0.0f : (1.0f / ss0[0]);
        device float4* out4 = (device float4*)(out + h * D);
        out4[tiisg] = so0[tiisg] * inv;
        out4[tiisg + NW] = so0[tiisg + NW] * inv;
    }
}
"#;

/// TG size for FA-vec decode (d=64/96/128/256): NSG=8 × NW=32.
pub(crate) fn gqa_fa_vec_threadgroup_size(_head_dim: u32) -> u32 {
    256
}

/// Head dims the FA-vec decode kernels cover (dedicated specializations).
pub(crate) fn gqa_fa_vec_supported(head_dim: u32) -> bool {
    matches!(head_dim, 64 | 96 | 128 | 256)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gqa_fa_vec(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    q: &ProtocolObject<dyn MTLBuffer>,
    k: &ProtocolObject<dyn MTLBuffer>,
    v: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_start: u32,
    softcap: f32,
) -> Result<(), MetalError> {
    let pipe = match head_dim {
        256 => ensure_pipeline(
            device,
            GQA_DECODE_FA_VEC_D256_KERNEL_SRC,
            "gqa_decode_fa_vec_d256",
        )?,
        128 => ensure_pipeline(device, GQA_DECODE_FA_VEC_KERNEL_SRC, "gqa_decode_fa_vec")?,
        96 => ensure_pipeline(
            device,
            GQA_DECODE_FA_VEC_D96_KERNEL_SRC,
            "gqa_decode_fa_vec_d96",
        )?,
        64 => ensure_pipeline(
            device,
            GQA_DECODE_FA_VEC_D64_KERNEL_SRC,
            "gqa_decode_fa_vec_d64",
        )?,
        _ => return Err(MetalError::CommandFailed),
    };
    encoder.setComputePipelineState(&pipe.0);
    let tg = gqa_fa_vec_threadgroup_size(head_dim);
    let nsg = tg / 32;
    // Q[D] + NSG * (C=32 scores + D output)
    let tg_mem = ((head_dim + nsg * (32 + head_dim)) * 4) as usize;
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(q), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(k), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(v), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out), 0, 3);
        let mut nh = n_heads;
        encoder.setBytes_length_atIndex(NonNull::new(&mut nh as *mut u32 as *mut _).unwrap(), 4, 4);
        let mut nkv = n_kv_heads;
        encoder.setBytes_length_atIndex(
            NonNull::new(&mut nkv as *mut u32 as *mut _).unwrap(),
            4,
            5,
        );
        let mut hd = head_dim;
        encoder.setBytes_length_atIndex(NonNull::new(&mut hd as *mut u32 as *mut _).unwrap(), 4, 6);
        let mut sl = seq_len;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sl as *mut u32 as *mut _).unwrap(), 4, 7);
        let mut ks = kv_start;
        encoder.setBytes_length_atIndex(NonNull::new(&mut ks as *mut u32 as *mut _).unwrap(), 4, 8);
        let mut sc = softcap;
        encoder.setBytes_length_atIndex(NonNull::new(&mut sc as *mut f32 as *mut _).unwrap(), 4, 9);
        encoder.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
    }
    dispatch_counted(
        encoder,
        MTLSize {
            width: n_heads as usize,
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
