use super::{sdot, vmmla_s32};
use crate::repack::common::*;
use crate::repack::q6_kx8::*;
use crate::{Q8KActivations, Q6_K_BLOCK_ELEMS};
use std::arch::aarch64::*;

/// NEON DotProd GEMV for interleave-8 packed Q6_K weights (llama.cpp
/// `ggml_gemv_q6_K_8x8_q8_K` in `arch/arm/repack.cpp`). The -32 offset
/// is folded into a bsums × scales bias (shifted left 5) instead of
/// being subtracted from every value.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q6_kx8_q8_k_neon_8x8(
    packed: &[u8],
    act: &Q8KActivations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q6_K_BLOCK_ELEMS;
    let m4b = vdupq_n_u8(0x0f);
    let mask_lo = vdupq_n_u8(0x03);
    let mask_hi = vdupq_n_u8(0x30);

    for x in 0..n_row_groups {
        let mut acc_f32 = [vdupq_n_f32(0.0), vdupq_n_f32(0.0)];
        let group_off = x * nb * Q6_KX8_BLOCK_BYTES;

        for b in 0..nb {
            let blk = packed.as_ptr().add(group_off + b * Q6_KX8_BLOCK_BYTES);
            let scales_base = blk.add(16) as *const i8;
            let ql_blk = blk.add(144);
            let qh_blk = blk.add(1168);

            let mut d_arr = [0f32; 8];
            for (j, slot) in d_arr.iter_mut().enumerate() {
                *slot = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
            }
            let q8_d = act.d[b];
            let sb_scale = [
                vmulq_n_f32(vld1q_f32(d_arr.as_ptr()), q8_d),
                vmulq_n_f32(vld1q_f32(d_arr.as_ptr().add(4)), q8_d),
            ];

            let mut acc = [vdup_n_s32(0); 4];

            // 16 groups of 8 i8 scales, widened once per block.
            let mut q6_scales = [0i16; 16 * 8];
            for i in 0..16 {
                let s16 = vmovl_s8(vld1_s8(scales_base.add(i * 8)));
                vst1q_s16(q6_scales.as_mut_ptr().add(i * 8), s16);
            }

            // Bias: bsums × scales × 32 replaces subtracting 32 from
            // every 6-bit value.
            let mut bias_lo = vdupq_n_s32(0);
            let mut bias_hi = vdupq_n_s32(0);
            for i in (0..16).step_by(4) {
                let bsums_vec = vld1_s16(act.bsums.as_ptr().add(b * 16 + i));
                let sc = q6_scales.as_ptr();
                bias_lo = vmlal_lane_s16::<0>(bias_lo, vld1_s16(sc.add(i * 8)), bsums_vec);
                bias_hi = vmlal_lane_s16::<0>(bias_hi, vld1_s16(sc.add(i * 8 + 4)), bsums_vec);
                bias_lo = vmlal_lane_s16::<1>(bias_lo, vld1_s16(sc.add((i + 1) * 8)), bsums_vec);
                bias_hi =
                    vmlal_lane_s16::<1>(bias_hi, vld1_s16(sc.add((i + 1) * 8 + 4)), bsums_vec);
                bias_lo = vmlal_lane_s16::<2>(bias_lo, vld1_s16(sc.add((i + 2) * 8)), bsums_vec);
                bias_hi =
                    vmlal_lane_s16::<2>(bias_hi, vld1_s16(sc.add((i + 2) * 8 + 4)), bsums_vec);
                bias_lo = vmlal_lane_s16::<3>(bias_lo, vld1_s16(sc.add((i + 3) * 8)), bsums_vec);
                bias_hi =
                    vmlal_lane_s16::<3>(bias_hi, vld1_s16(sc.add((i + 3) * 8 + 4)), bsums_vec);
            }
            bias_lo = vshlq_n_s32(bias_lo, 5);
            bias_hi = vshlq_n_s32(bias_hi, 5);

            for half in 0..2 {
                let ql_base = ql_blk.add(half * 512);
                let qh_base = qh_blk.add(half * 256);
                let q8_half = act.q.as_ptr().add(b * Q6_K_BLOCK_ELEMS + half * 128);

                for sb in 0..4 {
                    let q8_base_l = q8_half.add(sb * 16);
                    let q8_base_h = q8_base_l.add(64);
                    let mut q8_l = [vdupq_n_s8(0); 2];
                    let mut q8_h = [vdupq_n_s8(0); 2];
                    for i in 0..2 {
                        q8_l[i] =
                            vreinterpretq_s8_s64(vld1q_dup_s64(q8_base_l.add(i * 8) as *const i64));
                        q8_h[i] =
                            vreinterpretq_s8_s64(vld1q_dup_s64(q8_base_h.add(i * 8) as *const i64));
                    }

                    let ql_off = sb * (Q6_K_BLOCK_ELEMS / 2);
                    let qh_off = ql_off & 255; // wraps after 256 bytes
                    let mut q6_ql_0 = [vdupq_n_u8(0); 4];
                    let mut q6_ql_1 = [vdupq_n_u8(0); 4];
                    let mut q6_qh_0 = [vdupq_n_u8(0); 4];
                    let mut q6_qh_1 = [vdupq_n_u8(0); 4];
                    for k in 0..4 {
                        q6_ql_0[k] = vld1q_u8(ql_base.add(ql_off + 16 * k));
                        q6_ql_1[k] = vld1q_u8(ql_base.add(ql_off + 64 + 16 * k));
                        q6_qh_0[k] = vld1q_u8(qh_base.add(qh_off + 16 * k));
                        q6_qh_1[k] = vld1q_u8(qh_base.add(qh_off + 64 + 16 * k));
                    }
                    // High bits for sub-blocks 2 and 3 sit two bits up.
                    if sb > 1 {
                        for k in 0..4 {
                            q6_qh_0[k] = vshrq_n_u8(q6_qh_0[k], 2);
                            q6_qh_1[k] = vshrq_n_u8(q6_qh_1[k], 2);
                        }
                    }

                    for cp in 0..4 {
                        let hh_0 = vandq_u8(q6_qh_0[cp], mask_hi);
                        let hh_1 = vandq_u8(q6_qh_1[cp], mask_hi);

                        // q6 = low4 | high2<<4; no -32 here, the bias
                        // pass above already carries it.
                        let q6_l0 = vreinterpretq_s8_u8(vsliq_n_u8(
                            vandq_u8(q6_ql_0[cp], m4b),
                            vandq_u8(q6_qh_0[cp], mask_lo),
                            4,
                        ));
                        let q6_l1 = vreinterpretq_s8_u8(vsliq_n_u8(
                            vandq_u8(q6_ql_1[cp], m4b),
                            vandq_u8(q6_qh_1[cp], mask_lo),
                            4,
                        ));
                        let q6_h0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_ql_0[cp], 4), hh_0));
                        let q6_h1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_ql_1[cp], 4), hh_1));

                        let mut sb_acc_l = vdupq_n_s32(0);
                        sb_acc_l = sdot(sb_acc_l, q6_l0, q8_l[0]);
                        sb_acc_l = sdot(sb_acc_l, q6_l1, q8_l[1]);
                        let mut sb_acc_h = vdupq_n_s32(0);
                        sb_acc_h = sdot(sb_acc_h, q6_h0, q8_h[0]);
                        sb_acc_h = sdot(sb_acc_h, q6_h1, q8_h[1]);

                        let sum_l = vpadd_s32(vget_low_s32(sb_acc_l), vget_high_s32(sb_acc_l));
                        let sum_h = vpadd_s32(vget_low_s32(sb_acc_h), vget_high_s32(sb_acc_h));

                        let scale_idx_l = half * 8 + sb;
                        let scale_idx_h = half * 8 + sb + 4;
                        let scale_vec_l = vset_lane_s32::<1>(
                            i32::from(q6_scales[scale_idx_l * 8 + cp * 2 + 1]),
                            vdup_n_s32(i32::from(q6_scales[scale_idx_l * 8 + cp * 2])),
                        );
                        let scale_vec_h = vset_lane_s32::<1>(
                            i32::from(q6_scales[scale_idx_h * 8 + cp * 2 + 1]),
                            vdup_n_s32(i32::from(q6_scales[scale_idx_h * 8 + cp * 2])),
                        );

                        acc[cp] = vmla_s32(acc[cp], sum_l, scale_vec_l);
                        acc[cp] = vmla_s32(acc[cp], sum_h, scale_vec_h);
                    }
                }
            }

            acc[0] = vsub_s32(acc[0], vget_low_s32(bias_lo));
            acc[1] = vsub_s32(acc[1], vget_high_s32(bias_lo));
            acc[2] = vsub_s32(acc[2], vget_low_s32(bias_hi));
            acc[3] = vsub_s32(acc[3], vget_high_s32(bias_hi));

            let w_01 = vmul_f32(vcvt_f32_s32(acc[0]), vget_low_f32(sb_scale[0]));
            let w_23 = vmul_f32(vcvt_f32_s32(acc[1]), vget_high_f32(sb_scale[0]));
            let w_45 = vmul_f32(vcvt_f32_s32(acc[2]), vget_low_f32(sb_scale[1]));
            let w_67 = vmul_f32(vcvt_f32_s32(acc[3]), vget_high_f32(sb_scale[1]));

            acc_f32[0] = vaddq_f32(acc_f32[0], vcombine_f32(w_01, w_23));
            acc_f32[1] = vaddq_f32(acc_f32[1], vcombine_f32(w_45, w_67));
        }

        let base = x * Q6_KX8_NROWS;
        vst1q_f32(out.as_mut_ptr().add(base), acc_f32[0]);
        vst1q_f32(out.as_mut_ptr().add(base + 4), acc_f32[1]);
    }
}

/// NEON i8mm **GEMM** for interleave-8 packed Q6_K weights (llama.cpp
/// `ggml_gemm_q6_K_8x8_q8_K` in `arch/arm/repack.cpp`). Q6_K has no
/// mins: the -32 offset is folded into the i8 values before the MMLAs
/// (63 - 32 fits i8), so there is no bias pass at all.
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn gemm_q6_kx8_q8_k_neon_i8mm(
    packed: &[u8],
    tile: &Q8KActsX4,
    n_cols: usize,
    out: &mut [f32],
) {
    let na = tile.na;
    debug_assert!(na <= Q8K_ACTS_X4_NC);
    let nb = n_cols / Q6_K_BLOCK_ELEMS;
    debug_assert_eq!(tile.n_blocks, nb);
    let m4b = vdupq_n_u8(0x0f);
    let mask_lo = vdupq_n_u8(0x03);
    let mask_hi = vdupq_n_u8(0x30);
    let m32s = vdupq_n_s8(32);
    const Q8_K_BLOCKLEN: usize = 4;

    let mut acc_f32 = [vdupq_n_f32(0.0); Q8K_ACTS_X4_NC * 2];

    for b in 0..nb {
        let blk = packed.as_ptr().add(b * Q6_KX8_BLOCK_BYTES);
        let scales_base = blk.add(16) as *const i8;
        let ql_blk = blk.add(144);
        let qh_blk = blk.add(1168);
        let q8_blk = tile.qs.as_ptr().add(b * Q6_K_BLOCK_ELEMS * 4);

        let mut acc = [vdupq_n_s32(0); 8];

        // 16 groups of 8 i8 scales, widened once per block.
        let mut q6_scales = [0i16; 16 * 8];
        for i in 0..16 {
            let s16 = vmovl_s8(vld1_s8(scales_base.add(i * 8)));
            vst1q_s16(q6_scales.as_mut_ptr().add(i * 8), s16);
        }

        for half in 0..2 {
            let ql_base = ql_blk.add(half * 512);
            let qh_base = qh_blk.add(half * 256);

            for sb in 0..4 {
                let q8_base_l = q8_blk.add(half * 512 + sb * 64);
                let q8_base_h = q8_blk.add(half * 512 + 256 + sb * 64);

                let mut q8_l_01 = [vdupq_n_s8(0); 2];
                let mut q8_l_23 = [vdupq_n_s8(0); 2];
                let mut q8_h_01 = [vdupq_n_s8(0); 2];
                let mut q8_h_23 = [vdupq_n_s8(0); 2];
                for i in 0..2 {
                    q8_l_01[i] = vld1q_s8(q8_base_l.add(i * 32));
                    q8_l_23[i] = vld1q_s8(q8_base_l.add(i * 32 + 16));
                    q8_h_01[i] = vld1q_s8(q8_base_h.add(i * 32));
                    q8_h_23[i] = vld1q_s8(q8_base_h.add(i * 32 + 16));
                }

                let ql_off = sb * (Q6_K_BLOCK_ELEMS / 2);
                let qh_off = ql_off & 255; // wraps after 256 bytes
                let mut q6_ql_0 = [vdupq_n_u8(0); 4];
                let mut q6_ql_1 = [vdupq_n_u8(0); 4];
                let mut q6_qh_0 = [vdupq_n_u8(0); 4];
                let mut q6_qh_1 = [vdupq_n_u8(0); 4];
                for k in 0..4 {
                    q6_ql_0[k] = vld1q_u8(ql_base.add(ql_off + 16 * k));
                    q6_ql_1[k] = vld1q_u8(ql_base.add(ql_off + 64 + 16 * k));
                    q6_qh_0[k] = vld1q_u8(qh_base.add(qh_off + 16 * k));
                    q6_qh_1[k] = vld1q_u8(qh_base.add(qh_off + 64 + 16 * k));
                }
                // High bits for sub-blocks 2 and 3 sit two bits up.
                if sb > 1 {
                    for k in 0..4 {
                        q6_qh_0[k] = vshrq_n_u8(q6_qh_0[k], 2);
                        q6_qh_1[k] = vshrq_n_u8(q6_qh_1[k], 2);
                    }
                }

                for cp in 0..4 {
                    let hh_0 = vandq_u8(q6_qh_0[cp], mask_hi);
                    let hh_1 = vandq_u8(q6_qh_1[cp], mask_hi);

                    // q6 = (low4 | high2<<4) - 32
                    let q6_l0 = vsubq_s8(
                        vreinterpretq_s8_u8(vsliq_n_u8(
                            vandq_u8(q6_ql_0[cp], m4b),
                            vandq_u8(q6_qh_0[cp], mask_lo),
                            4,
                        )),
                        m32s,
                    );
                    let q6_l1 = vsubq_s8(
                        vreinterpretq_s8_u8(vsliq_n_u8(
                            vandq_u8(q6_ql_1[cp], m4b),
                            vandq_u8(q6_qh_1[cp], mask_lo),
                            4,
                        )),
                        m32s,
                    );
                    let q6_h0 = vsubq_s8(
                        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_ql_0[cp], 4), hh_0)),
                        m32s,
                    );
                    let q6_h1 = vsubq_s8(
                        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_ql_1[cp], 4), hh_1)),
                        m32s,
                    );

                    let mut sb_acc_0l = vmmla_s32(vdupq_n_s32(0), q6_l0, q8_l_01[0]);
                    sb_acc_0l = vmmla_s32(sb_acc_0l, q6_l1, q8_l_01[1]);
                    let mut sb_acc_0h = vmmla_s32(vdupq_n_s32(0), q6_h0, q8_h_01[0]);
                    sb_acc_0h = vmmla_s32(sb_acc_0h, q6_h1, q8_h_01[1]);
                    let mut sb_acc_1l = vmmla_s32(vdupq_n_s32(0), q6_l0, q8_l_23[0]);
                    sb_acc_1l = vmmla_s32(sb_acc_1l, q6_l1, q8_l_23[1]);
                    let mut sb_acc_1h = vmmla_s32(vdupq_n_s32(0), q6_h0, q8_h_23[0]);
                    sb_acc_1h = vmmla_s32(sb_acc_1h, q6_h1, q8_h_23[1]);

                    let scale_idx_l = half * 8 + sb;
                    let scale_idx_h = half * 8 + sb + 4;
                    let scale_l = vcombine_s32(
                        vdup_n_s32(i32::from(q6_scales[scale_idx_l * 8 + cp * 2])),
                        vdup_n_s32(i32::from(q6_scales[scale_idx_l * 8 + cp * 2 + 1])),
                    );
                    let scale_h = vcombine_s32(
                        vdup_n_s32(i32::from(q6_scales[scale_idx_h * 8 + cp * 2])),
                        vdup_n_s32(i32::from(q6_scales[scale_idx_h * 8 + cp * 2 + 1])),
                    );

                    acc[cp] = vmlaq_s32(acc[cp], sb_acc_0l, scale_l);
                    acc[cp] = vmlaq_s32(acc[cp], sb_acc_0h, scale_h);
                    acc[cp + 4] = vmlaq_s32(acc[cp + 4], sb_acc_1l, scale_l);
                    acc[cp + 4] = vmlaq_s32(acc[cp + 4], sb_acc_1h, scale_h);
                }
            }
        }

        for lane in acc.iter_mut() {
            let aux = vzip_s32(vget_low_s32(*lane), vget_high_s32(*lane));
            *lane = vcombine_s32(aux.0, aux.1);
        }
        let reorder_acc = [
            vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1])),
            vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3])),
            vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1])),
            vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3])),
            vcombine_s32(vget_low_s32(acc[4]), vget_low_s32(acc[5])),
            vcombine_s32(vget_low_s32(acc[6]), vget_low_s32(acc[7])),
            vcombine_s32(vget_high_s32(acc[4]), vget_high_s32(acc[5])),
            vcombine_s32(vget_high_s32(acc[6]), vget_high_s32(acc[7])),
        ];

        let mut d_arr = [0f32; 8];
        for (j, slot) in d_arr.iter_mut().enumerate() {
            *slot = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
        }

        for i in 0..na {
            for j in 0..2 {
                let q8_d = vdupq_n_f32(*tile.d.as_ptr().add(b * Q8_K_BLOCKLEN + i));
                let scale = vmulq_f32(vld1q_f32(d_arr.as_ptr().add(j * 4)), q8_d);
                let idx = 2 * i + j;
                acc_f32[idx] = vmlaq_f32(acc_f32[idx], vcvtq_f32_s32(reorder_acc[idx]), scale);
            }
        }
    }

    for a in 0..na {
        let mut row = [0f32; Q6_KX8_NROWS];
        vst1q_f32(row.as_mut_ptr(), acc_f32[2 * a]);
        vst1q_f32(row.as_mut_ptr().add(4), acc_f32[2 * a + 1]);
        for (r, v) in row.iter().enumerate() {
            out[r * na + a] = *v;
        }
    }
}
