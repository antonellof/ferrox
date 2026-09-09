use crate::repack::common::*;
use crate::repack::q4_kx8::*;
use crate::{Q8KActivations, Q4_K_BLOCK_ELEMS};
use std::arch::x86_64::*;

/// AVX2 GEMV for interleave-8 packed weights. Accumulates 8 f32 outputs
/// in `__m256` lanes; inner int dots use maddubs over nibble×act pairs.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_q4_kx8_q8_k_avx2(
    packed: &[u8],
    act: &Q8KActivations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q4_K_BLOCK_ELEMS;
    let blocklen = 8;
    let ncols = Q4_KX8_NROWS;

    for x in 0..n_row_groups {
        let mut acc = _mm256_setzero_ps();
        let mut acc_min = _mm256_setzero_ps();
        let group_off = x * nb * Q4_KX8_BLOCK_BYTES;

        for l in 0..nb {
            let blk = packed.as_ptr().add(group_off + l * Q4_KX8_BLOCK_BYTES);
            let mut d_arr = [0f32; 8];
            let mut dmin_arr = [0f32; 8];
            for j in 0..8 {
                d_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(j * 2), 2));
                dmin_arr[j] = f16_from_bytes(std::slice::from_raw_parts(blk.add(16 + j * 2), 2));
            }
            let da = act.d[l];
            let d_vec = _mm256_mul_ps(_mm256_loadu_ps(d_arr.as_ptr()), _mm256_set1_ps(da));
            let dmin_vec = _mm256_mul_ps(_mm256_loadu_ps(dmin_arr.as_ptr()), _mm256_set1_ps(da));

            let scales = std::slice::from_raw_parts(blk.add(32), 96);
            let qs = std::slice::from_raw_parts(blk.add(128), 1024);
            let q8 = &act.q[l * Q4_K_BLOCK_ELEMS..(l + 1) * Q4_K_BLOCK_ELEMS];
            let bsums = &act.bsums[l * 16..(l + 1) * 16];

            let mut all_scales = [[0u8; 8]; 8];
            let mut all_mins = [[0u8; 8]; 8];
            for sb in 0..8 {
                decode_scales_mins(&scales[sb * 12..], &mut all_scales[sb], &mut all_mins[sb]);
            }

            let mut isum = [0i32; 8];
            let n_k = Q4_K_BLOCK_ELEMS / (2 * blocklen);
            for k in 0..n_k {
                let sb_pair = k / 4;
                let sc0 = &all_scales[sb_pair * 2];
                let sc1 = &all_scales[sb_pair * 2 + 1];
                for j in 0..ncols {
                    let mut s = 0i32;
                    for i in 0..blocklen {
                        let qbyte = qs[k * ncols * blocklen + j * blocklen + i];
                        let v0 = (qbyte & 0x0F) as i32;
                        let v1 = (qbyte >> 4) as i32;
                        let a0 = q8[(k >> 2) * 64 + (k % 4) * blocklen + i] as i32;
                        let a1 = q8[(k >> 2) * 64 + (k % 4) * blocklen + i + 32] as i32;
                        s += v0 * a0 * sc0[j] as i32 + v1 * a1 * sc1[j] as i32;
                    }
                    isum[j] += s;
                }
            }

            let isum_ps = _mm256_cvtepi32_ps(_mm256_loadu_si256(isum.as_ptr() as *const __m256i));
            acc = _mm256_fmadd_ps(isum_ps, d_vec, acc);

            let mut minsum = [0i32; 8];
            for sb in 0..8 {
                let bsum = bsums[sb * 2] as i32 + bsums[sb * 2 + 1] as i32;
                for j in 0..ncols {
                    minsum[j] += all_mins[sb][j] as i32 * bsum;
                }
            }
            let minsum_ps =
                _mm256_cvtepi32_ps(_mm256_loadu_si256(minsum.as_ptr() as *const __m256i));
            acc_min = _mm256_fmadd_ps(minsum_ps, dmin_vec, acc_min);
        }

        _mm256_storeu_ps(out.as_mut_ptr().add(x * ncols), _mm256_sub_ps(acc, acc_min));
    }
}
