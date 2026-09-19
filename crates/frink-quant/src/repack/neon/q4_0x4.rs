use super::{sdot, sdot_lane, vmmla_s32};
use crate::repack::common::*;
use crate::repack::q4_0x4::*;
use crate::{Q8Activations, Q4_0_BLOCK_ELEMS};
use half::f16;
use std::arch::aarch64::*;

/// NEON DotProd GEMV for `block_q4_0x4` (llama `ggml_gemv_q4_0_4x4_q8_0`).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q4_0x4_q8_0_neon_sdot(
    packed: &[u8],
    act: &Q8Activations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q4_0_BLOCK_ELEMS;
    let maskf0 = vdupq_n_u8(0xF0);
    for x in 0..n_row_groups {
        let mut acc = vdupq_n_f32(0.0);
        let group_off = x * nb * Q4_0X4_BLOCK_BYTES;
        for b in 0..nb {
            let blk = packed.as_ptr().add(group_off + b * Q4_0X4_BLOCK_BYTES);
            let qs = blk.add(8);
            let a_ptr = act.q.as_ptr().add(b * Q4_0_BLOCK_ELEMS);
            let a0 = vld1q_s8(a_ptr);
            let a1 = vld1q_s8(a_ptr.add(16));

            let mut ret = vdupq_n_s32(0);
            for wi in 0..4u32 {
                let w = vld1q_u8(qs.add(wi as usize * 16));
                let hi = vreinterpretq_s8_u8(vshlq_n_u8(w, 4));
                let lo = vreinterpretq_s8_u8(vandq_u8(w, maskf0));
                ret = sdot_lane(ret, hi, a0, wi);
                ret = sdot_lane(ret, lo, a1, wi);
            }

            let d_bits = vld1_u16(blk as *const u16);
            let mut dw = [0f32; 4];
            dw[0] = f16::from_bits(vget_lane_u16(d_bits, 0)).to_f32();
            dw[1] = f16::from_bits(vget_lane_u16(d_bits, 1)).to_f32();
            dw[2] = f16::from_bits(vget_lane_u16(d_bits, 2)).to_f32();
            dw[3] = f16::from_bits(vget_lane_u16(d_bits, 3)).to_f32();
            let scale = vmulq_n_f32(vld1q_f32(dw.as_ptr()), act.d[b]);
            acc = vfmaq_f32(acc, vcvtq_f32_s32(vshrq_n_s32(ret, 4)), scale);
        }
        vst1q_f32(out.as_mut_ptr().add(x * Q4_0X4_NROWS), acc);
    }
}

/// NEON DotProd GEMM for one `block_q4_0x4` row-group against several
/// activations (llama `ggml_gemm_q4_0_4x4_q8_0` in shape).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemm_q4_0x4_q8_0_neon_sdot(
    group: &[u8],
    acts: &[Q8Activations],
    n_cols: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q4_0_BLOCK_ELEMS;
    let n_acts = acts.len();
    let maskf0 = vdupq_n_u8(0xF0);
    let mut j0 = 0;
    while j0 < n_acts {
        let tile = Q4_0X4_GEMM_NC.min(n_acts - j0);
        let mut acc = [vdupq_n_f32(0.0); Q4_0X4_GEMM_NC];
        for b in 0..nb {
            let blk = group.as_ptr().add(b * Q4_0X4_BLOCK_BYTES);
            let qs = blk.add(8);
            let w = [
                vld1q_u8(qs),
                vld1q_u8(qs.add(16)),
                vld1q_u8(qs.add(32)),
                vld1q_u8(qs.add(48)),
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
                let a_ptr = act.q.as_ptr().add(b * Q4_0_BLOCK_ELEMS);
                let a0 = vld1q_s8(a_ptr);
                let a1 = vld1q_s8(a_ptr.add(16));
                let mut ret = vdupq_n_s32(0);
                for (wi, wchunk) in w.iter().enumerate() {
                    let hi = vreinterpretq_s8_u8(vshlq_n_u8(*wchunk, 4));
                    let lo = vreinterpretq_s8_u8(vandq_u8(*wchunk, maskf0));
                    ret = sdot_lane(ret, hi, a0, wi as u32);
                    ret = sdot_lane(ret, lo, a1, wi as u32);
                }
                let scale = vmulq_n_f32(dw_v, act.d[b]);
                acc[t] = vfmaq_f32(acc[t], vcvtq_f32_s32(vshrq_n_s32(ret, 4)), scale);
            }
        }
        for t in 0..tile {
            let mut lanes = [0f32; Q4_0X4_NROWS];
            vst1q_f32(lanes.as_mut_ptr(), acc[t]);
            for (r, v) in lanes.iter().enumerate() {
                out[r * n_acts + j0 + t] = *v;
            }
        }
        j0 += tile;
    }
}

/// NEON DotProd GEMV for interleave-8 packed Q4_0 weights (llama.cpp
/// `ggml_gemv_q4_0_4x8_q8_0` in `arch/arm/repack.cpp`). Nibbles are
/// consumed at 16× their value (`<< 4` for the low half, `& 0xf0` for
/// the high half — the pack's 0x88 XOR already folded in the -8), and
/// the fixed-point convert divides the 16 back out.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q4_0x4_q8_0_neon_4x8(
    packed: &[u8],
    act: &Q8Activations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q4_0_BLOCK_ELEMS;
    let m4b = vdupq_n_u8(0xf0);

    for x in 0..n_row_groups {
        let mut acc = vdupq_n_f32(0.0);
        let group_off = x * nb * Q4_0X4_BLOCK_BYTES;

        for b in 0..nb {
            let blk = packed.as_ptr().add(group_off + b * Q4_0X4_BLOCK_BYTES);
            let qs = blk.add(8);
            let mut d_arr = [0f32; 4];
            for (j, slot) in d_arr.iter_mut().enumerate() {
                *slot = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
            }
            let b_d = vld1q_f32(d_arr.as_ptr());
            let a_base = act.q.as_ptr().add(b * Q4_0_BLOCK_ELEMS);

            let b0 = vld1q_u8(qs);
            let b1 = vld1q_u8(qs.add(16));
            let b2 = vld1q_u8(qs.add(32));
            let b3 = vld1q_u8(qs.add(48));

            let mut a = [vdupq_n_s8(0); 4];
            for (c, slot) in a.iter_mut().enumerate() {
                *slot = vreinterpretq_s8_s64(vld1q_dup_s64(a_base.add(c * 8) as *const i64));
            }

            let mut ret0 = vdupq_n_s32(0);
            let mut ret1 = vdupq_n_s32(0);
            ret0 = sdot(ret0, vreinterpretq_s8_u8(vshlq_n_u8(b0, 4)), a[0]);
            ret1 = sdot(ret1, vreinterpretq_s8_u8(vshlq_n_u8(b1, 4)), a[0]);
            ret0 = sdot(ret0, vreinterpretq_s8_u8(vshlq_n_u8(b2, 4)), a[1]);
            ret1 = sdot(ret1, vreinterpretq_s8_u8(vshlq_n_u8(b3, 4)), a[1]);
            ret0 = sdot(ret0, vreinterpretq_s8_u8(vandq_u8(b0, m4b)), a[2]);
            ret1 = sdot(ret1, vreinterpretq_s8_u8(vandq_u8(b1, m4b)), a[2]);
            ret0 = sdot(ret0, vreinterpretq_s8_u8(vandq_u8(b2, m4b)), a[3]);
            ret1 = sdot(ret1, vreinterpretq_s8_u8(vandq_u8(b3, m4b)), a[3]);
            let ret = vpaddq_s32(ret0, ret1);

            acc = vfmaq_f32(acc, vcvtq_n_f32_s32::<4>(ret), vmulq_n_f32(b_d, act.d[b]));
        }

        vst1q_f32(out.as_mut_ptr().add(x * Q4_0X4_NROWS), acc);
    }
}

/// NEON i8mm **GEMM** for interleave-8 packed Q4_0 weights. llama.cpp
/// ships `ggml_gemm_q4_0_4x8_q8_0` (`arch/arm/repack.cpp`) as raw
/// inline asm; this is the same computation with intrinsics, following
/// the `4x8` GEMV's nibble handling and the Q8_0 GEMM's MMLA tiling.
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn gemm_q4_0x4_q8_0_neon_i8mm(
    packed: &[u8],
    tile: &Q8ActsX4,
    n_cols: usize,
    out: &mut [f32],
) {
    let na = tile.na;
    debug_assert!(na <= Q8K_ACTS_X4_NC);
    let nb = n_cols / Q4_0_BLOCK_ELEMS;
    debug_assert_eq!(tile.n_blocks, nb);
    let m4b = vdupq_n_u8(0xf0);

    let mut acc_f32 = [vdupq_n_f32(0.0); Q8K_ACTS_X4_NC];

    for b in 0..nb {
        let blk = packed.as_ptr().add(b * Q4_0X4_BLOCK_BYTES);
        let qs = blk.add(8);
        let a_base = tile.qs.as_ptr().add(b * Q4_0_BLOCK_ELEMS * 4);

        let bv = [
            vld1q_u8(qs),
            vld1q_u8(qs.add(16)),
            vld1q_u8(qs.add(32)),
            vld1q_u8(qs.add(48)),
        ];
        // Weight vectors per activation chunk: lo nibbles cover elems
        // 0..16 (chunks 0,1), hi nibbles elems 16..32 (chunks 2,3),
        // all at 16× their value until the fixed-point convert.
        let w = [
            [
                vreinterpretq_s8_u8(vshlq_n_u8(bv[0], 4)),
                vreinterpretq_s8_u8(vshlq_n_u8(bv[1], 4)),
            ],
            [
                vreinterpretq_s8_u8(vshlq_n_u8(bv[2], 4)),
                vreinterpretq_s8_u8(vshlq_n_u8(bv[3], 4)),
            ],
            [
                vreinterpretq_s8_u8(vandq_u8(bv[0], m4b)),
                vreinterpretq_s8_u8(vandq_u8(bv[1], m4b)),
            ],
            [
                vreinterpretq_s8_u8(vandq_u8(bv[2], m4b)),
                vreinterpretq_s8_u8(vandq_u8(bv[3], m4b)),
            ],
        ];

        let mut acc = [vdupq_n_s32(0); 4];
        for (chunk, w_pair) in w.iter().enumerate() {
            let a01 = vld1q_s8(a_base.add(chunk * 32));
            let a23 = vld1q_s8(a_base.add(chunk * 32 + 16));

            acc[0] = vmmla_s32(acc[0], a01, w_pair[0]);
            acc[1] = vmmla_s32(acc[1], a01, w_pair[1]);
            acc[2] = vmmla_s32(acc[2], a23, w_pair[0]);
            acc[3] = vmmla_s32(acc[3], a23, w_pair[1]);
        }

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
                vcvtq_n_f32_s32::<4>(rows[a]),
                vmulq_n_f32(b_d, *tile.d.as_ptr().add(b * 4 + a)),
            );
        }
    }

    for a in 0..na {
        let mut lanes = [0f32; Q4_0X4_NROWS];
        vst1q_f32(lanes.as_mut_ptr(), acc_f32[a]);
        for (r, v) in lanes.iter().enumerate() {
            out[r * na + a] = *v;
        }
    }
}
