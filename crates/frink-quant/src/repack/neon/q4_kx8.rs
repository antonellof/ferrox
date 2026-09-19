use super::{sdot, sdot_lane, vmmla_s32};
use crate::repack::common::*;
use crate::repack::q4_kx8::*;
use crate::{Q8KActivations, Q4_K_BLOCK_ELEMS};
use std::arch::aarch64::*;

/// NEON DotProd GEMV for interleave-4 packed weights (Apple Silicon path).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q4_kx8_q8_k_neon_sdot(
    packed: &[u8],
    act: &Q8KActivations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q4_K_BLOCK_ELEMS;
    let m4b = vdupq_n_u8(0x0f);

    for x in 0..n_row_groups {
        let mut acc_f32 = [vdupq_n_f32(0.0), vdupq_n_f32(0.0)];
        let group_off = x * nb * Q4_KX8_BLOCK_BYTES;

        for b in 0..nb {
            let blk = packed.as_ptr().add(group_off + b * Q4_KX8_BLOCK_BYTES);
            let mut d_arr = [0f32; 8];
            let mut dmin_arr = [0f32; 8];
            for j in 0..8 {
                d_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
                dmin_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(16 + j * 2), 2));
            }
            let q8_d = act.d[b];
            let sb_scale_0123 = vmulq_n_f32(vld1q_f32(d_arr.as_ptr()), q8_d);
            let sb_scale_4567 = vmulq_n_f32(vld1q_f32(d_arr.as_ptr().add(4)), q8_d);
            let sb_min_0123 = vmulq_n_f32(vld1q_f32(dmin_arr.as_ptr()), q8_d);
            let sb_min_4567 = vmulq_n_f32(vld1q_f32(dmin_arr.as_ptr().add(4)), q8_d);

            let mut bias_acc = [vdupq_n_s32(0), vdupq_n_s32(0)];
            let q8_base = act.q.as_ptr().add(b * Q4_K_BLOCK_ELEMS);
            let bsums_ptr = act.bsums.as_ptr().add(b * 16);
            // Pairwise-add 16 bsums → 8 (matching llama vpaddq_s16).
            let mut bsums_arr = [0i16; 8];
            for (i, slot) in bsums_arr.iter_mut().enumerate() {
                *slot = *bsums_ptr.add(2 * i) + *bsums_ptr.add(2 * i + 1);
            }

            let scales_base = blk.add(32);
            let qs_base = blk.add(128);

            for sb in 0..4 {
                let mut acc_lo = [vdupq_n_s32(0), vdupq_n_s32(0)];
                let mut acc_hi = [vdupq_n_s32(0), vdupq_n_s32(0)];

                let mut q4sb_mins = [vdupq_n_s16(0); 2];
                let mut q4sb_scales = [vdupq_n_s16(0); 2];
                for i in 0..2 {
                    let mut sc = [0u8; 8];
                    let mut mn = [0u8; 8];
                    let offset = sb * 24 + i * 12;
                    decode_scales_mins(
                        std::slice::from_raw_parts(scales_base.add(offset), 12),
                        &mut sc,
                        &mut mn,
                    );
                    let mut sc_i8 = [0i8; 8];
                    let mut mn_i8 = [0i8; 8];
                    for t in 0..8 {
                        sc_i8[t] = sc[t] as i8;
                        mn_i8[t] = mn[t] as i8;
                    }
                    q4sb_scales[i] = vmovl_s8(vld1_s8(sc_i8.as_ptr()));
                    q4sb_mins[i] = vmovl_s8(vld1_s8(mn_i8.as_ptr()));
                }

                let mut q8_qs = [vdupq_n_s8(0); 4];
                for (i, slot) in q8_qs.iter_mut().enumerate() {
                    *slot = vld1q_s8(q8_base.add(sb * 64 + i * 16));
                }

                for c in 0..2 {
                    let mut q4_cols = [vdupq_n_u8(0); 8];
                    for (i, slot) in q4_cols.iter_mut().enumerate() {
                        *slot = vld1q_u8(qs_base.add(sb * Q4_K_BLOCK_ELEMS + i * 32 + 16 * c));
                    }

                    acc_lo[c] = sdot_lane(
                        acc_lo[c],
                        vreinterpretq_s8_u8(vandq_u8(q4_cols[0], m4b)),
                        q8_qs[0],
                        0,
                    );
                    acc_lo[c] = sdot_lane(
                        acc_lo[c],
                        vreinterpretq_s8_u8(vandq_u8(q4_cols[1], m4b)),
                        q8_qs[0],
                        1,
                    );
                    acc_lo[c] = sdot_lane(
                        acc_lo[c],
                        vreinterpretq_s8_u8(vandq_u8(q4_cols[2], m4b)),
                        q8_qs[0],
                        2,
                    );
                    acc_lo[c] = sdot_lane(
                        acc_lo[c],
                        vreinterpretq_s8_u8(vandq_u8(q4_cols[3], m4b)),
                        q8_qs[0],
                        3,
                    );
                    acc_lo[c] = sdot_lane(
                        acc_lo[c],
                        vreinterpretq_s8_u8(vandq_u8(q4_cols[4], m4b)),
                        q8_qs[1],
                        0,
                    );
                    acc_lo[c] = sdot_lane(
                        acc_lo[c],
                        vreinterpretq_s8_u8(vandq_u8(q4_cols[5], m4b)),
                        q8_qs[1],
                        1,
                    );
                    acc_lo[c] = sdot_lane(
                        acc_lo[c],
                        vreinterpretq_s8_u8(vandq_u8(q4_cols[6], m4b)),
                        q8_qs[1],
                        2,
                    );
                    acc_lo[c] = sdot_lane(
                        acc_lo[c],
                        vreinterpretq_s8_u8(vandq_u8(q4_cols[7], m4b)),
                        q8_qs[1],
                        3,
                    );

                    acc_hi[c] = sdot_lane(
                        acc_hi[c],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[0], 4)),
                        q8_qs[2],
                        0,
                    );
                    acc_hi[c] = sdot_lane(
                        acc_hi[c],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[1], 4)),
                        q8_qs[2],
                        1,
                    );
                    acc_hi[c] = sdot_lane(
                        acc_hi[c],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[2], 4)),
                        q8_qs[2],
                        2,
                    );
                    acc_hi[c] = sdot_lane(
                        acc_hi[c],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[3], 4)),
                        q8_qs[2],
                        3,
                    );
                    acc_hi[c] = sdot_lane(
                        acc_hi[c],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[4], 4)),
                        q8_qs[3],
                        0,
                    );
                    acc_hi[c] = sdot_lane(
                        acc_hi[c],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[5], 4)),
                        q8_qs[3],
                        1,
                    );
                    acc_hi[c] = sdot_lane(
                        acc_hi[c],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[6], 4)),
                        q8_qs[3],
                        2,
                    );
                    acc_hi[c] = sdot_lane(
                        acc_hi[c],
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[7], 4)),
                        q8_qs[3],
                        3,
                    );
                }

                let sc_0123_lo = vget_low_s16(q4sb_scales[0]);
                let sc_0123_hi = vget_low_s16(q4sb_scales[1]);
                let sumf_0123 = vcvtq_f32_s32(vaddq_s32(
                    vmulq_s32(vmovl_s16(sc_0123_lo), acc_lo[0]),
                    vmulq_s32(vmovl_s16(sc_0123_hi), acc_hi[0]),
                ));
                acc_f32[0] = vfmaq_f32(acc_f32[0], sb_scale_0123, sumf_0123);

                let sc_4567_lo = vget_high_s16(q4sb_scales[0]);
                let sc_4567_hi = vget_high_s16(q4sb_scales[1]);
                let sumf_4567 = vcvtq_f32_s32(vaddq_s32(
                    vmulq_s32(vmovl_s16(sc_4567_lo), acc_lo[1]),
                    vmulq_s32(vmovl_s16(sc_4567_hi), acc_hi[1]),
                ));
                acc_f32[1] = vfmaq_f32(acc_f32[1], sb_scale_4567, sumf_4567);

                let bsums_vec_lo = vdup_n_s16(bsums_arr[2 * sb]);
                let bsums_vec_hi = vdup_n_s16(bsums_arr[2 * sb + 1]);
                bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_lo, vget_low_s16(q4sb_mins[0]));
                bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_hi, vget_low_s16(q4sb_mins[1]));
                bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_lo, vget_high_s16(q4sb_mins[0]));
                bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_hi, vget_high_s16(q4sb_mins[1]));
            }

            acc_f32[0] = vmlsq_f32(acc_f32[0], vcvtq_f32_s32(bias_acc[0]), sb_min_0123);
            acc_f32[1] = vmlsq_f32(acc_f32[1], vcvtq_f32_s32(bias_acc[1]), sb_min_4567);
        }

        let base = x * Q4_KX8_NROWS;
        vst1q_f32(out.as_mut_ptr().add(base), acc_f32[0]);
        vst1q_f32(out.as_mut_ptr().add(base + 4), acc_f32[1]);
    }
}

/// NEON DotProd GEMV for interleave-8 packed Q4_K weights (llama.cpp
/// `ggml_gemv_q4_K_8x8_q8_K` in `arch/arm/repack.cpp`). Each 8-byte q8
/// run is broadcast to both vector halves so one `sdot` covers two
/// interleaved columns at once.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemv_q4_kx8_q8_k_neon_8x8(
    packed: &[u8],
    act: &Q8KActivations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q4_K_BLOCK_ELEMS;
    let m4b = vdupq_n_u8(0x0f);

    for x in 0..n_row_groups {
        let mut acc_f32 = [vdupq_n_f32(0.0), vdupq_n_f32(0.0)];
        let group_off = x * nb * Q4_KX8_BLOCK_BYTES;

        for b in 0..nb {
            let blk = packed.as_ptr().add(group_off + b * Q4_KX8_BLOCK_BYTES);
            let mut d_arr = [0f32; 8];
            let mut dmin_arr = [0f32; 8];
            for j in 0..8 {
                d_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
                dmin_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(16 + j * 2), 2));
            }
            let q8_d = act.d[b];
            let sb_scale = [
                vmulq_n_f32(vld1q_f32(d_arr.as_ptr()), q8_d),
                vmulq_n_f32(vld1q_f32(d_arr.as_ptr().add(4)), q8_d),
            ];
            let sb_min = [
                vmulq_n_f32(vld1q_f32(dmin_arr.as_ptr()), q8_d),
                vmulq_n_f32(vld1q_f32(dmin_arr.as_ptr().add(4)), q8_d),
            ];

            let q8_base = act.q.as_ptr().add(b * Q4_K_BLOCK_ELEMS);
            let bsums_ptr = act.bsums.as_ptr().add(b * 16);
            let mut bsums_arr = [0i16; 8];
            for (i, slot) in bsums_arr.iter_mut().enumerate() {
                *slot = *bsums_ptr.add(2 * i) + *bsums_ptr.add(2 * i + 1);
            }

            let scales_base = blk.add(32);
            let qs_base = blk.add(128);

            let mut bias_acc = [vdupq_n_s32(0), vdupq_n_s32(0)];

            for sb in 0..4 {
                let mut acc_lo = [vdupq_n_s32(0); 4];
                let mut acc_hi = [vdupq_n_s32(0); 4];

                let mut q4sb_scales = [vdupq_n_s16(0); 2];
                let mut q4sb_mins = [vdupq_n_s16(0); 2];
                for i in 0..2 {
                    let mut sc = [0u8; 8];
                    let mut mn = [0u8; 8];
                    let offset = sb * 24 + i * 12;
                    decode_scales_mins(
                        std::slice::from_raw_parts(scales_base.add(offset), 12),
                        &mut sc,
                        &mut mn,
                    );
                    let mut sc_i8 = [0i8; 8];
                    let mut mn_i8 = [0i8; 8];
                    for t in 0..8 {
                        sc_i8[t] = sc[t] as i8;
                        mn_i8[t] = mn[t] as i8;
                    }
                    q4sb_scales[i] = vmovl_s8(vld1_s8(sc_i8.as_ptr()));
                    q4sb_mins[i] = vmovl_s8(vld1_s8(mn_i8.as_ptr()));
                }

                let q8_sb = q8_base.add(sb * 64);
                let mut q8_qs = [vdupq_n_s8(0); 8];
                for (i, slot) in q8_qs.iter_mut().enumerate() {
                    *slot = vreinterpretq_s8_s64(vld1q_dup_s64(q8_sb.add(i * 8) as *const i64));
                }

                for cp in 0..4 {
                    let q4_qs = [
                        vld1q_u8(qs_base.add(sb * Q4_K_BLOCK_ELEMS + 16 * cp)),
                        vld1q_u8(qs_base.add(sb * Q4_K_BLOCK_ELEMS + 16 * cp + 64)),
                        vld1q_u8(qs_base.add(sb * Q4_K_BLOCK_ELEMS + 16 * cp + 128)),
                        vld1q_u8(qs_base.add(sb * Q4_K_BLOCK_ELEMS + 16 * cp + 192)),
                    ];
                    for m in 0..4 {
                        let q4_lo = vreinterpretq_s8_u8(vandq_u8(q4_qs[m], m4b));
                        let q4_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_qs[m], 4));
                        acc_lo[cp] = sdot(acc_lo[cp], q4_lo, q8_qs[m]);
                        acc_hi[cp] = sdot(acc_hi[cp], q4_hi, q8_qs[m + 4]);
                    }
                }

                for i in 0..2 {
                    let p = i * 2;
                    let (scales_lo, scales_hi) = if i == 0 {
                        (vget_low_s16(q4sb_scales[0]), vget_low_s16(q4sb_scales[1]))
                    } else {
                        (vget_high_s16(q4sb_scales[0]), vget_high_s16(q4sb_scales[1]))
                    };
                    let sumf_0 = vcvtq_f32_s32(vmulq_s32(
                        vmovl_s16(scales_lo),
                        vpaddq_s32(acc_lo[p], acc_lo[p + 1]),
                    ));
                    acc_f32[i] = vfmaq_f32(acc_f32[i], sb_scale[i], sumf_0);
                    let sumf_1 = vcvtq_f32_s32(vmulq_s32(
                        vmovl_s16(scales_hi),
                        vpaddq_s32(acc_hi[p], acc_hi[p + 1]),
                    ));
                    acc_f32[i] = vfmaq_f32(acc_f32[i], sb_scale[i], sumf_1);
                }

                let bsums_vec_lo = vdup_n_s16(bsums_arr[2 * sb]);
                let bsums_vec_hi = vdup_n_s16(bsums_arr[2 * sb + 1]);
                bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_lo, vget_low_s16(q4sb_mins[0]));
                bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_hi, vget_low_s16(q4sb_mins[1]));
                bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_lo, vget_high_s16(q4sb_mins[0]));
                bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_hi, vget_high_s16(q4sb_mins[1]));
            }

            acc_f32[0] = vmlsq_f32(acc_f32[0], vcvtq_f32_s32(bias_acc[0]), sb_min[0]);
            acc_f32[1] = vmlsq_f32(acc_f32[1], vcvtq_f32_s32(bias_acc[1]), sb_min[1]);
        }

        let base = x * Q4_KX8_NROWS;
        vst1q_f32(out.as_mut_ptr().add(base), acc_f32[0]);
        vst1q_f32(out.as_mut_ptr().add(base + 4), acc_f32[1]);
    }
}

/// NEON DotProd **GEMM** for interleave-4 packed Q4_K weights: one
/// row-group (8 rows) against up to [`Q4_KX8_GEMM_NC`] activations.
///
/// Same arithmetic as [`gemv_q4_kx8_q8_k_neon_sdot`], reordered so
/// the weight-side unpack happens once per activation *tile* rather
/// than once per activation. Per 256-element super-block that hoists
/// 16 f16 scale conversions, 8 `decode_scales_mins` calls and 16
/// `q4_cols` loads out of the batch loop -- which is the whole point,
/// and the same reason llama.cpp ships `ggml_gemm_q4_K_8x4_q8_K`
/// beside its GEMV rather than looping the GEMV.
///
/// `out` is `[row][act]`: `out[r * na + j]`.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemm_q4_kx8_q8_k_neon_sdot(
    packed: &[u8],
    acts: &[Q8KActivations],
    n_cols: usize,
    out: &mut [f32],
) {
    let na = acts.len();
    debug_assert!(na <= Q4_KX8_GEMM_NC);
    let nb = n_cols / Q4_K_BLOCK_ELEMS;
    let m4b = vdupq_n_u8(0x0f);

    // [act][row-half]; row-half 0 is rows 0..3, 1 is rows 4..7.
    let mut acc_f32 = [[vdupq_n_f32(0.0); 2]; Q4_KX8_GEMM_NC];
    let mut bias_acc = [[vdupq_n_s32(0); 2]; Q4_KX8_GEMM_NC];

    for b in 0..nb {
        let blk = packed.as_ptr().add(b * Q4_KX8_BLOCK_BYTES);

        // --- weight-side, once per block (was once per activation) ---
        let mut d_arr = [0f32; 8];
        let mut dmin_arr = [0f32; 8];
        for j in 0..8 {
            d_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
            dmin_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(16 + j * 2), 2));
        }
        let d_lo = vld1q_f32(d_arr.as_ptr());
        let d_hi = vld1q_f32(d_arr.as_ptr().add(4));
        let dmin_lo = vld1q_f32(dmin_arr.as_ptr());
        let dmin_hi = vld1q_f32(dmin_arr.as_ptr().add(4));

        // Per-activation scaling of those, plus the pairwise-added
        // bsums this block needs (llama's vpaddq_s16).
        let mut sb_scale = [[vdupq_n_f32(0.0); 2]; Q4_KX8_GEMM_NC];
        let mut sb_min = [[vdupq_n_f32(0.0); 2]; Q4_KX8_GEMM_NC];
        let mut bsums_arr = [[0i16; 8]; Q4_KX8_GEMM_NC];
        for (a, act) in acts.iter().enumerate() {
            let q8_d = act.d[b];
            sb_scale[a] = [vmulq_n_f32(d_lo, q8_d), vmulq_n_f32(d_hi, q8_d)];
            sb_min[a] = [vmulq_n_f32(dmin_lo, q8_d), vmulq_n_f32(dmin_hi, q8_d)];
            let bsums_ptr = act.bsums.as_ptr().add(b * 16);
            for (i, slot) in bsums_arr[a].iter_mut().enumerate() {
                *slot = *bsums_ptr.add(2 * i) + *bsums_ptr.add(2 * i + 1);
            }
        }

        let scales_base = blk.add(32);
        let qs_base = blk.add(128);

        for sb in 0..4 {
            // 6-bit scale/min decode: once per block-quarter, not
            // once per (block-quarter, activation).
            let mut q4sb_mins = [vdupq_n_s16(0); 2];
            let mut q4sb_scales = [vdupq_n_s16(0); 2];
            for i in 0..2 {
                let mut sc = [0u8; 8];
                let mut mn = [0u8; 8];
                let offset = sb * 24 + i * 12;
                decode_scales_mins(
                    std::slice::from_raw_parts(scales_base.add(offset), 12),
                    &mut sc,
                    &mut mn,
                );
                let mut sc_i8 = [0i8; 8];
                let mut mn_i8 = [0i8; 8];
                for t in 0..8 {
                    sc_i8[t] = sc[t] as i8;
                    mn_i8[t] = mn[t] as i8;
                }
                q4sb_scales[i] = vmovl_s8(vld1_s8(sc_i8.as_ptr()));
                q4sb_mins[i] = vmovl_s8(vld1_s8(mn_i8.as_ptr()));
            }

            // `c` selects the row half, so each pass owns one output
            // quad and the accumulators can be consumed immediately
            // instead of all eight staying live.
            for c in 0..2 {
                let mut q4_cols = [vdupq_n_u8(0); 8];
                for (i, slot) in q4_cols.iter_mut().enumerate() {
                    *slot = vld1q_u8(qs_base.add(sb * Q4_K_BLOCK_ELEMS + i * 32 + 16 * c));
                }
                let (sc_lo, sc_hi) = if c == 0 {
                    (vget_low_s16(q4sb_scales[0]), vget_low_s16(q4sb_scales[1]))
                } else {
                    (vget_high_s16(q4sb_scales[0]), vget_high_s16(q4sb_scales[1]))
                };

                // Mask once per weight tile, not once per
                // activation, and keep the lane indices literal --
                // a runtime lane forces a real call per `sdot`
                // instead of the single instruction it should be.
                let lo0 = vreinterpretq_s8_u8(vandq_u8(q4_cols[0], m4b));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(q4_cols[1], m4b));
                let lo2 = vreinterpretq_s8_u8(vandq_u8(q4_cols[2], m4b));
                let lo3 = vreinterpretq_s8_u8(vandq_u8(q4_cols[3], m4b));
                let lo4 = vreinterpretq_s8_u8(vandq_u8(q4_cols[4], m4b));
                let lo5 = vreinterpretq_s8_u8(vandq_u8(q4_cols[5], m4b));
                let lo6 = vreinterpretq_s8_u8(vandq_u8(q4_cols[6], m4b));
                let lo7 = vreinterpretq_s8_u8(vandq_u8(q4_cols[7], m4b));
                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[0], 4));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[1], 4));
                let hi2 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[2], 4));
                let hi3 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[3], 4));
                let hi4 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[4], 4));
                let hi5 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[5], 4));
                let hi6 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[6], 4));
                let hi7 = vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[7], 4));
                let sc_lo_w = vmovl_s16(sc_lo);
                let sc_hi_w = vmovl_s16(sc_hi);

                for a in 0..na {
                    let q8_base = acts[a].q.as_ptr().add(b * Q4_K_BLOCK_ELEMS);
                    let y0 = vld1q_s8(q8_base.add(sb * 64));
                    let y1 = vld1q_s8(q8_base.add(sb * 64 + 16));
                    let y2 = vld1q_s8(q8_base.add(sb * 64 + 32));
                    let y3 = vld1q_s8(q8_base.add(sb * 64 + 48));
                    let mut acc_lo = vdupq_n_s32(0);
                    let mut acc_hi = vdupq_n_s32(0);
                    acc_lo = sdot_lane(acc_lo, lo0, y0, 0);
                    acc_lo = sdot_lane(acc_lo, lo1, y0, 1);
                    acc_lo = sdot_lane(acc_lo, lo2, y0, 2);
                    acc_lo = sdot_lane(acc_lo, lo3, y0, 3);
                    acc_lo = sdot_lane(acc_lo, lo4, y1, 0);
                    acc_lo = sdot_lane(acc_lo, lo5, y1, 1);
                    acc_lo = sdot_lane(acc_lo, lo6, y1, 2);
                    acc_lo = sdot_lane(acc_lo, lo7, y1, 3);
                    acc_hi = sdot_lane(acc_hi, hi0, y2, 0);
                    acc_hi = sdot_lane(acc_hi, hi1, y2, 1);
                    acc_hi = sdot_lane(acc_hi, hi2, y2, 2);
                    acc_hi = sdot_lane(acc_hi, hi3, y2, 3);
                    acc_hi = sdot_lane(acc_hi, hi4, y3, 0);
                    acc_hi = sdot_lane(acc_hi, hi5, y3, 1);
                    acc_hi = sdot_lane(acc_hi, hi6, y3, 2);
                    acc_hi = sdot_lane(acc_hi, hi7, y3, 3);
                    let sumf = vcvtq_f32_s32(vaddq_s32(
                        vmulq_s32(sc_lo_w, acc_lo),
                        vmulq_s32(sc_hi_w, acc_hi),
                    ));
                    acc_f32[a][c] = vfmaq_f32(acc_f32[a][c], sb_scale[a][c], sumf);
                }
            }

            for a in 0..na {
                let bs_lo = vdup_n_s16(bsums_arr[a][2 * sb]);
                let bs_hi = vdup_n_s16(bsums_arr[a][2 * sb + 1]);
                bias_acc[a][0] = vmlal_s16(bias_acc[a][0], bs_lo, vget_low_s16(q4sb_mins[0]));
                bias_acc[a][0] = vmlal_s16(bias_acc[a][0], bs_hi, vget_low_s16(q4sb_mins[1]));
                bias_acc[a][1] = vmlal_s16(bias_acc[a][1], bs_lo, vget_high_s16(q4sb_mins[0]));
                bias_acc[a][1] = vmlal_s16(bias_acc[a][1], bs_hi, vget_high_s16(q4sb_mins[1]));
            }
        }

        for a in 0..na {
            for c in 0..2 {
                acc_f32[a][c] =
                    vmlsq_f32(acc_f32[a][c], vcvtq_f32_s32(bias_acc[a][c]), sb_min[a][c]);
                bias_acc[a][c] = vdupq_n_s32(0);
            }
        }
    }

    for a in 0..na {
        let mut row = [0f32; Q4_KX8_NROWS];
        vst1q_f32(row.as_mut_ptr(), acc_f32[a][0]);
        vst1q_f32(row.as_mut_ptr().add(4), acc_f32[a][1]);
        for (r, v) in row.iter().enumerate() {
            out[r * na + a] = *v;
        }
    }
}

/// NEON i8mm **GEMM** for interleave-8 packed Q4_K weights (llama.cpp
/// `ggml_gemm_q4_K_8x8_q8_K` in `arch/arm/repack.cpp`). Uses `vmmlaq_s32`
/// on 2×8×8 tiles; the activation quad arrives pre-interleaved as
/// [`Q8KActsX4`] (llama's `block_q8_Kx4` in `wdata`), so nothing here is
/// repacked per row-group.
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn gemm_q4_kx8_q8_k_neon_i8mm(
    packed: &[u8],
    tile: &Q8KActsX4,
    n_cols: usize,
    out: &mut [f32],
) {
    let na = tile.na;
    debug_assert!(na <= Q4_KX8_GEMM_NC);
    let nb = n_cols / Q4_K_BLOCK_ELEMS;
    debug_assert_eq!(tile.n_blocks, nb);
    let m4b = vdupq_n_u8(0x0f);
    const Q8_K_BLOCKLEN: usize = 4;

    let mut acc_f32 = [vdupq_n_f32(0.0); Q4_KX8_GEMM_NC * 2];

    for b in 0..nb {
        let blk = packed.as_ptr().add(b * Q4_KX8_BLOCK_BYTES);
        let bsums_base = tile.bsums.as_ptr().add(b * Q8_K_BLOCKLEN * 8);

        let mut acc = [vdupq_n_s32(0); 8];
        let mut bias_acc = [vdupq_n_s32(0); 8];
        for i in 0..8 {
            acc[i] = vdupq_n_s32(0);
            bias_acc[i] = vdupq_n_s32(0);
        }

        let scales_base = blk.add(32);
        let qs_base = blk.add(128);
        let q8_base = tile.qs.as_ptr().add(b * Q4_K_BLOCK_ELEMS * 4);

        for sb in 0..4 {
            let mut q4sb_scales = [[0i8; 8]; 2];
            let mut q4sb_mins = [vdupq_n_s16(0); 2];
            for i in 0..2 {
                let mut sc = [0u8; 8];
                let mut mn = [0u8; 8];
                let offset = sb * 24 + i * 12;
                decode_scales_mins(
                    std::slice::from_raw_parts(scales_base.add(offset), 12),
                    &mut sc,
                    &mut mn,
                );
                let mut mn_i8 = [0i8; 8];
                for t in 0..8 {
                    q4sb_scales[i][t] = sc[t] as i8;
                    mn_i8[t] = mn[t] as i8;
                }
                q4sb_mins[i] = vmovl_s8(vld1_s8(mn_i8.as_ptr()));
            }

            let q8_sb = q8_base.add(sb * 256);
            let mut q8_qs_01 = [vdupq_n_s8(0); 8];
            let mut q8_qs_23 = [vdupq_n_s8(0); 8];
            for i in 0..8 {
                q8_qs_01[i] = vld1q_s8(q8_sb.add(i * 32));
                q8_qs_23[i] = vld1q_s8(q8_sb.add(i * 32 + 16));
            }
            let q8s = [q8_qs_01, q8_qs_23];

            for cp in 0..4 {
                let mut sb_acc = [vdupq_n_s32(0); 4];

                let q4_qs = [
                    vld1q_u8(qs_base.add(sb * Q4_K_BLOCK_ELEMS + 16 * cp)),
                    vld1q_u8(qs_base.add(sb * Q4_K_BLOCK_ELEMS + 16 * cp + 64)),
                    vld1q_u8(qs_base.add(sb * Q4_K_BLOCK_ELEMS + 16 * cp + 128)),
                    vld1q_u8(qs_base.add(sb * Q4_K_BLOCK_ELEMS + 16 * cp + 192)),
                ];
                let q4_nibbles = [
                    [
                        vreinterpretq_s8_u8(vandq_u8(q4_qs[0], m4b)),
                        vreinterpretq_s8_u8(vandq_u8(q4_qs[1], m4b)),
                        vreinterpretq_s8_u8(vandq_u8(q4_qs[2], m4b)),
                        vreinterpretq_s8_u8(vandq_u8(q4_qs[3], m4b)),
                    ],
                    [
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_qs[0], 4)),
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_qs[1], 4)),
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_qs[2], 4)),
                        vreinterpretq_s8_u8(vshrq_n_u8(q4_qs[3], 4)),
                    ],
                ];

                for rp in 0..2 {
                    for blk in 0..2 {
                        let q8 = &q8s[rp][4 * blk..4 * blk + 4];
                        let q4 = &q4_nibbles[blk];
                        let mut tile_acc = sb_acc[2 * rp + blk];
                        for qs_offset in 0..4 {
                            tile_acc = vmmla_s32(tile_acc, q4[qs_offset], q8[qs_offset]);
                        }
                        sb_acc[2 * rp + blk] = tile_acc;
                    }
                }

                let scale_offset = cp * 2;
                let block_scale_0 = vcombine_s32(
                    vdup_n_s32(i32::from(q4sb_scales[0][scale_offset])),
                    vdup_n_s32(i32::from(q4sb_scales[0][scale_offset + 1])),
                );
                let block_scale_1 = vcombine_s32(
                    vdup_n_s32(i32::from(q4sb_scales[1][scale_offset])),
                    vdup_n_s32(i32::from(q4sb_scales[1][scale_offset + 1])),
                );

                acc[cp] = vmlaq_s32(acc[cp], sb_acc[0], block_scale_0);
                acc[cp + 4] = vmlaq_s32(acc[cp + 4], sb_acc[2], block_scale_0);
                acc[cp] = vmlaq_s32(acc[cp], sb_acc[1], block_scale_1);
                acc[cp + 4] = vmlaq_s32(acc[cp + 4], sb_acc[3], block_scale_1);
            }

            for q8_row in 0..Q8_K_BLOCKLEN {
                let bs_lo = vdup_n_s16(*bsums_base.add(q8_row * 8 + 2 * sb));
                let bs_hi = vdup_n_s16(*bsums_base.add(q8_row * 8 + 2 * sb + 1));
                bias_acc[2 * q8_row] =
                    vmlal_s16(bias_acc[2 * q8_row], bs_lo, vget_low_s16(q4sb_mins[0]));
                bias_acc[2 * q8_row] =
                    vmlal_s16(bias_acc[2 * q8_row], bs_hi, vget_low_s16(q4sb_mins[1]));
                bias_acc[2 * q8_row + 1] =
                    vmlal_s16(bias_acc[2 * q8_row + 1], bs_lo, vget_high_s16(q4sb_mins[0]));
                bias_acc[2 * q8_row + 1] =
                    vmlal_s16(bias_acc[2 * q8_row + 1], bs_hi, vget_high_s16(q4sb_mins[1]));
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
        let mut dmin_arr = [0f32; 8];
        for j in 0..8 {
            d_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
            dmin_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(16 + j * 2), 2));
        }

        for i in 0..na {
            for j in 0..2 {
                let q8_d = vdupq_n_f32(*tile.d.as_ptr().add(b * Q8_K_BLOCKLEN + i));
                let dmins = vmulq_f32(vld1q_f32(dmin_arr.as_ptr().add(j * 4)), q8_d);
                let scale = vmulq_f32(vld1q_f32(d_arr.as_ptr().add(j * 4)), q8_d);
                let idx = 2 * i + j;
                acc_f32[idx] = vmlsq_f32(acc_f32[idx], vcvtq_f32_s32(bias_acc[idx]), dmins);
                acc_f32[idx] = vmlaq_f32(acc_f32[idx], vcvtq_f32_s32(reorder_acc[idx]), scale);
            }
        }
    }

    for a in 0..na {
        let mut row = [0f32; Q4_KX8_NROWS];
        vst1q_f32(row.as_mut_ptr(), acc_f32[2 * a]);
        vst1q_f32(row.as_mut_ptr().add(4), acc_f32[2 * a + 1]);
        for (r, v) in row.iter().enumerate() {
            out[r * na + a] = *v;
        }
    }
}
