use super::{sdot, sdot_lane, vmmla_s32};
use crate::repack::common::*;
use crate::repack::q8_0x4::*;
use crate::{Q8Activations, Q8_0_BLOCK_ELEMS};
use half::f16;
use std::arch::aarch64::*;

/// NEON DotProd GEMV for `block_q8_0x4` (llama `ggml_gemv_q8_0_4x4_q8_0`).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q8_0x4_q8_0_neon_sdot(
    packed: &[u8],
    act: &Q8Activations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    for x in 0..n_row_groups {
        let mut acc = vdupq_n_f32(0.0);
        let group_off = x * nb * Q8_0X4_BLOCK_BYTES;
        for b in 0..nb {
            let blk = packed.as_ptr().add(group_off + b * Q8_0X4_BLOCK_BYTES);
            let qs = blk.add(8);
            // Four int8x16: first 64 qs bytes (k=0..3 × 4 rows × 4).
            let b0 = vld1q_s8(qs as *const i8);
            let b1 = vld1q_s8(qs.add(16) as *const i8);
            let b2 = vld1q_s8(qs.add(32) as *const i8);
            let b3 = vld1q_s8(qs.add(48) as *const i8);
            let b4 = vld1q_s8(qs.add(64) as *const i8);
            let b5 = vld1q_s8(qs.add(80) as *const i8);
            let b6 = vld1q_s8(qs.add(96) as *const i8);
            let b7 = vld1q_s8(qs.add(112) as *const i8);

            let a_ptr = act.q.as_ptr().add(b * Q8_0_BLOCK_ELEMS);
            let a0 = vld1q_s8(a_ptr);
            let a1 = vld1q_s8(a_ptr.add(16));

            let mut ret = vdupq_n_s32(0);
            ret = sdot_lane(ret, b0, a0, 0);
            ret = sdot_lane(ret, b1, a0, 1);
            ret = sdot_lane(ret, b2, a0, 2);
            ret = sdot_lane(ret, b3, a0, 3);
            ret = sdot_lane(ret, b4, a1, 0);
            ret = sdot_lane(ret, b5, a1, 1);
            ret = sdot_lane(ret, b6, a1, 2);
            ret = sdot_lane(ret, b7, a1, 3);

            // Four f16 weight scales at blk[0..8] — load as u16 then
            // convert (avoids 4× scalar half::f16 path per block).
            let d_bits = vld1_u16(blk as *const u16);
            let mut dw = [0f32; 4];
            dw[0] = f16::from_bits(vget_lane_u16(d_bits, 0)).to_f32();
            dw[1] = f16::from_bits(vget_lane_u16(d_bits, 1)).to_f32();
            dw[2] = f16::from_bits(vget_lane_u16(d_bits, 2)).to_f32();
            dw[3] = f16::from_bits(vget_lane_u16(d_bits, 3)).to_f32();
            let scale = vmulq_n_f32(vld1q_f32(dw.as_ptr()), act.d[b]);
            acc = vfmaq_f32(acc, vcvtq_f32_s32(ret), scale);
        }
        vst1q_f32(out.as_mut_ptr().add(x * Q8_0X4_NROWS), acc);
    }
}

/// NEON DotProd GEMM for one `block_q8_0x4` row-group against
/// several activations (llama `ggml_gemm_q8_0_4x4_q8_0` in shape).
///
/// The eight weight vectors of a block are loaded once and reused
/// across a tile of [`Q8_0X4_GEMM_NC`] activations, which is the
/// whole point of having a GEMM rather than a loop over the GEMV.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemm_q8_0x4_q8_0_neon_sdot(
    group: &[u8],
    acts: &[Q8Activations],
    n_cols: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    let n_acts = acts.len();
    let mut j0 = 0;
    while j0 < n_acts {
        let tile = Q8_0X4_GEMM_NC.min(n_acts - j0);
        let mut acc = [vdupq_n_f32(0.0); Q8_0X4_GEMM_NC];
        for b in 0..nb {
            let blk = group.as_ptr().add(b * Q8_0X4_BLOCK_BYTES);
            let qs = blk.add(8);
            let w = [
                vld1q_s8(qs as *const i8),
                vld1q_s8(qs.add(16) as *const i8),
                vld1q_s8(qs.add(32) as *const i8),
                vld1q_s8(qs.add(48) as *const i8),
                vld1q_s8(qs.add(64) as *const i8),
                vld1q_s8(qs.add(80) as *const i8),
                vld1q_s8(qs.add(96) as *const i8),
                vld1q_s8(qs.add(112) as *const i8),
            ];
            let d_bits = vld1_u16(blk as *const u16);
            let mut dw = [0f32; 4];
            dw[0] = f16::from_bits(vget_lane_u16(d_bits, 0)).to_f32();
            dw[1] = f16::from_bits(vget_lane_u16(d_bits, 1)).to_f32();
            dw[2] = f16::from_bits(vget_lane_u16(d_bits, 2)).to_f32();
            dw[3] = f16::from_bits(vget_lane_u16(d_bits, 3)).to_f32();
            let dw_v = vld1q_f32(dw.as_ptr());

            for t in 0..tile {
                let act = &acts[j0 + t];
                let a_ptr = act.q.as_ptr().add(b * Q8_0_BLOCK_ELEMS);
                let a0 = vld1q_s8(a_ptr);
                let a1 = vld1q_s8(a_ptr.add(16));
                let mut ret = vdupq_n_s32(0);
                ret = sdot_lane(ret, w[0], a0, 0);
                ret = sdot_lane(ret, w[1], a0, 1);
                ret = sdot_lane(ret, w[2], a0, 2);
                ret = sdot_lane(ret, w[3], a0, 3);
                ret = sdot_lane(ret, w[4], a1, 0);
                ret = sdot_lane(ret, w[5], a1, 1);
                ret = sdot_lane(ret, w[6], a1, 2);
                ret = sdot_lane(ret, w[7], a1, 3);
                let scale = vmulq_n_f32(dw_v, act.d[b]);
                acc[t] = vfmaq_f32(acc[t], vcvtq_f32_s32(ret), scale);
            }
        }
        for t in 0..tile {
            let mut lanes = [0f32; Q8_0X4_NROWS];
            vst1q_f32(lanes.as_mut_ptr(), acc[t]);
            for (r, v) in lanes.iter().enumerate() {
                out[r * n_acts + j0 + t] = *v;
            }
        }
        j0 += tile;
    }
}

/// NEON DotProd GEMV for interleave-8 packed Q8_0 weights (llama.cpp
/// `ggml_gemv_q8_0_4x8_q8_0` in `arch/arm/repack.cpp`). Each 8-byte
/// activation run is broadcast to both vector halves so one `sdot`
/// covers two interleaved rows.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q8_0x4_q8_0_neon_4x8(
    packed: &[u8],
    act: &Q8Activations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q8_0_BLOCK_ELEMS;

    for x in 0..n_row_groups {
        let mut acc = vdupq_n_f32(0.0);
        let group_off = x * nb * Q8_0X4_BLOCK_BYTES;

        for b in 0..nb {
            let blk = packed.as_ptr().add(group_off + b * Q8_0X4_BLOCK_BYTES);
            let qs = blk.add(8) as *const i8;
            let mut d_arr = [0f32; 4];
            for (j, slot) in d_arr.iter_mut().enumerate() {
                *slot = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
            }
            let b_d = vld1q_f32(d_arr.as_ptr());
            let a_base = act.q.as_ptr().add(b * Q8_0_BLOCK_ELEMS);

            let mut ret0 = vdupq_n_s32(0);
            let mut ret1 = vdupq_n_s32(0);
            for c in 0..4 {
                let a = vreinterpretq_s8_s64(vld1q_dup_s64(a_base.add(c * 8) as *const i64));
                ret0 = sdot(ret0, vld1q_s8(qs.add(c * 32)), a);
                ret1 = sdot(ret1, vld1q_s8(qs.add(c * 32 + 16)), a);
            }
            let ret = vpaddq_s32(ret0, ret1);

            acc = vfmaq_f32(acc, vcvtq_f32_s32(ret), vmulq_n_f32(b_d, act.d[b]));
        }

        vst1q_f32(out.as_mut_ptr().add(x * Q8_0X4_NROWS), acc);
    }
}

/// NEON i8mm **GEMM** for interleave-8 packed Q8_0 weights (llama.cpp
/// `ggml_gemm_q8_0_4x8_q8_0` in `arch/arm/repack.cpp`, NEON branch).
/// The activation quad arrives pre-interleaved as [`Q8ActsX4`], so
/// every `vmmlaq_s32` covers a 2×2 (activation × weight-row) tile.
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn gemm_q8_0x4_q8_0_neon_i8mm(
    packed: &[u8],
    tile: &Q8ActsX4,
    n_cols: usize,
    out: &mut [f32],
) {
    let na = tile.na;
    debug_assert!(na <= Q8K_ACTS_X4_NC);
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    debug_assert_eq!(tile.n_blocks, nb);

    // acc_f32[a] holds activation row a's 4 weight-row outputs.
    let mut acc_f32 = [vdupq_n_f32(0.0); Q8K_ACTS_X4_NC];

    for b in 0..nb {
        let blk = packed.as_ptr().add(b * Q8_0X4_BLOCK_BYTES);
        let qs = blk.add(8) as *const i8;
        let a_base = tile.qs.as_ptr().add(b * Q8_0_BLOCK_ELEMS * 4);

        let mut acc = [vdupq_n_s32(0); 4];
        for chunk in 0..4 {
            let a01 = vld1q_s8(a_base.add(chunk * 32));
            let a23 = vld1q_s8(a_base.add(chunk * 32 + 16));
            let b01 = vld1q_s8(qs.add(chunk * 32));
            let b23 = vld1q_s8(qs.add(chunk * 32 + 16));

            acc[0] = vmmla_s32(acc[0], a01, b01);
            acc[1] = vmmla_s32(acc[1], a01, b23);
            acc[2] = vmmla_s32(acc[2], a23, b01);
            acc[3] = vmmla_s32(acc[3], a23, b23);
        }

        // 2×2 tiles → per-activation-row vectors of 4 weight rows.
        let rows = [
            vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1])),
            vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1])),
            vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3])),
            vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3])),
        ];

        let mut d_arr = [0f32; 4];
        for (j, slot) in d_arr.iter_mut().enumerate() {
            *slot = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
        }
        let b_d = vld1q_f32(d_arr.as_ptr());

        for a in 0..na {
            acc_f32[a] = vfmaq_f32(
                acc_f32[a],
                vcvtq_f32_s32(rows[a]),
                vmulq_n_f32(b_d, *tile.d.as_ptr().add(b * 4 + a)),
            );
        }
    }

    for a in 0..na {
        let mut lanes = [0f32; Q8_0X4_NROWS];
        vst1q_f32(lanes.as_mut_ptr(), acc_f32[a]);
        for (r, v) in lanes.iter().enumerate() {
            out[r * na + a] = *v;
        }
    }
}
