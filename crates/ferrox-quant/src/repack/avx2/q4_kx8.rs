use crate::repack::common::*;
use crate::repack::q4_kx8::*;
use crate::{Q8KActivations, Q4_K_BLOCK_ELEMS};
use std::arch::x86_64::*;

use super::{bcast8, load_f16x8, rows8_from_pairs, scale_lanes_u8};

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

/// AVX2 GEMM: one `block_q4_Kx8` row-group (8 rows) against a quad of up
/// to four Q8_K activations. The x86 twin of
/// [`crate::repack::neon::gemm_q4_kx8_q8_k_neon_i8mm`], and the scalar
/// twin it is held against is
/// [`crate::repack::q4_kx8::gemm_q4_kx8_acts_x4_scalar_8`].
///
/// # Lane mapping
///
/// llama.cpp's `ggml_gemm_q4_K_8x8_q8_K`
/// (`ggml/src/ggml-cpu/arch/x86/repack.cpp:2042`, AVX2 arm at
/// `repack.cpp:2065`) is the shape this mirrors, on the same
/// `block_q4_Kx8` bytes:
///
/// * the 4-bit quants enter `_mm256_maddubs_epi16` as the **unsigned**
///   operand and the int8 activation as the signed one, exactly as
///   llama.cpp's `rhs_mat_*` / `lhs_mat_*` pairing does;
/// * the 6-bit row scales are folded in with `_mm256_madd_epi16` on the
///   `i16` pair sums rather than after widening —
///   `iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0)`
///   at `repack.cpp:2757`;
/// * llama.cpp reaches the four-row activation quad with
///   `_mm256_shuffle_epi32` over a `block_q8_Kx4` load, where this
///   broadcasts one activation's 8-byte run with [`bcast8`] and walks
///   the quad's rows in the outer loop. Same arithmetic, one activation
///   per pass instead of four, which is what keeps the accumulator count
///   inside 16 YMM registers without AVX-512's 32.
///
/// # Overflow
///
/// The `i16` stage accumulates four `k` steps before the scale fold. A
/// pair sum is at most `2 * 15 * 127 = 3810` (both `quantize_activations_q8_k`
/// and ggml clamp to `+-127`), so four of them reach `15240`, inside
/// `i16`. The `i32` accumulator then takes at most
/// `256 * 15 * 127 * 63` per super-block, which is `30.7M`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_q4_kx8_q8_k_avx2(
    packed: &[u8],
    tile: &Q8KActsX4,
    n_cols: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q4_K_BLOCK_ELEMS;
    let na = tile.na;
    let m4b = _mm256_set1_epi8(0x0F);
    let mut acc = [_mm256_setzero_ps(); Q8K_ACTS_X4_NC];
    let mut acc_min = [_mm256_setzero_ps(); Q8K_ACTS_X4_NC];

    for l in 0..nb {
        let blk = packed.as_ptr().add(l * Q4_KX8_BLOCK_BYTES);
        let d_vec = load_f16x8(blk);
        let dmin_vec = load_f16x8(blk.add(16));
        let scales = std::slice::from_raw_parts(blk.add(32), 96);
        let qs = blk.add(128);
        let acts = tile.qs.as_ptr().add(l * Q4_K_BLOCK_ELEMS * 4);

        let mut all_scales = [[0u8; 8]; 8];
        let mut all_mins = [[0u8; 8]; 8];
        for sb in 0..8 {
            decode_scales_mins(&scales[sb * 12..], &mut all_scales[sb], &mut all_mins[sb]);
        }
        // [sub-block pair][nibble half][row half], hoisted out of the
        // activation loop: the weights do not depend on which activation
        // is being dotted, and that reuse is the whole point of a GEMM.
        let mut sc_v = [[[_mm256_setzero_si256(); 2]; 2]; 4];
        for (p, pair) in sc_v.iter_mut().enumerate() {
            for (nib, halves) in pair.iter_mut().enumerate() {
                let s = &all_scales[p * 2 + nib];
                halves[0] = scale_lanes_u8(s, 0);
                halves[1] = scale_lanes_u8(s, 4);
            }
        }
        // Mins, widened once per super-block for the same reason.
        let mut min_v = [_mm256_setzero_si256(); 8];
        for (sb, slot) in min_v.iter_mut().enumerate() {
            *slot = _mm256_cvtepu8_epi32(_mm_loadl_epi64(all_mins[sb].as_ptr() as *const __m128i));
        }

        for a in 0..na {
            let da = tile.d[l * 4 + a];
            let mut i32acc = [_mm256_setzero_si256(); 2];
            for (p, pair) in sc_v.iter().enumerate() {
                let mut i16acc = [[_mm256_setzero_si256(); 2]; 2];
                for kk in 0..4 {
                    let k = p * 4 + kk;
                    // Canonical element `e` lives at run `e / 8`, quad
                    // row `a`, lane `e % 8`; the low nibble's run is
                    // `p * 8 + kk` and the high nibble's is four later.
                    let ka = p * 8 + kk;
                    let a0 = bcast8(acts.add(ka * 32 + a * 8));
                    let a1 = bcast8(acts.add((ka + 4) * 32 + a * 8));
                    for half in 0..2 {
                        let w = _mm256_loadu_si256(qs.add(k * 64 + half * 32) as *const __m256i);
                        let v0 = _mm256_and_si256(w, m4b);
                        let v1 = _mm256_and_si256(_mm256_srli_epi16(w, 4), m4b);
                        i16acc[0][half] =
                            _mm256_add_epi16(i16acc[0][half], _mm256_maddubs_epi16(v0, a0));
                        i16acc[1][half] =
                            _mm256_add_epi16(i16acc[1][half], _mm256_maddubs_epi16(v1, a1));
                    }
                }
                for half in 0..2 {
                    i32acc[half] = _mm256_add_epi32(
                        i32acc[half],
                        _mm256_add_epi32(
                            _mm256_madd_epi16(i16acc[0][half], pair[0][half]),
                            _mm256_madd_epi16(i16acc[1][half], pair[1][half]),
                        ),
                    );
                }
            }
            let dav = _mm256_set1_ps(da);
            acc[a] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(rows8_from_pairs(i32acc[0], i32acc[1])),
                _mm256_mul_ps(d_vec, dav),
                acc[a],
            );

            let mut minsum = _mm256_setzero_si256();
            for (sb, m) in min_v.iter().enumerate() {
                let bsum = _mm256_set1_epi32(tile.bsums[(l * 4 + a) * 8 + sb] as i32);
                minsum = _mm256_add_epi32(minsum, _mm256_mullo_epi32(*m, bsum));
            }
            acc_min[a] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(minsum),
                _mm256_mul_ps(dmin_vec, dav),
                acc_min[a],
            );
        }
    }

    for a in 0..na {
        let mut v = [0f32; 8];
        _mm256_storeu_ps(v.as_mut_ptr(), _mm256_sub_ps(acc[a], acc_min[a]));
        for (j, got) in v.iter().enumerate() {
            out[j * na + a] = *got;
        }
    }
}
