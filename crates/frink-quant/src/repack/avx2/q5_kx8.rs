use crate::repack::common::*;
use crate::repack::q5_kx8::*;
use crate::Q5_K_BLOCK_ELEMS;
use std::arch::x86_64::*;

use super::{bcast8, load_f16x8, rows8_from_pairs, scale_lanes_u8};

/// AVX2 GEMM: one `block_q5_Kx8` row-group (8 rows) against a quad of up
/// to four Q8_K activations. The x86 twin of
/// [`crate::repack::neon::gemm_q5_kx8_q8_k_neon_i8mm`], held against the
/// scalar twin [`crate::repack::q5_kx8::gemm_q5_kx8_acts_x4_scalar_8`].
///
/// # Lane mapping
///
/// llama.cpp has no x86 Q5_K repack kernel; the shape mirrored here is
/// its Q4_K one, `ggml_gemm_q4_K_8x8_q8_K`
/// (`ggml/src/ggml-cpu/arch/x86/repack.cpp:2042`), which this layout is
/// the five-bit sibling of. A 32-byte load holds four rows' 8-byte runs,
/// `_mm256_maddubs_epi16` puts row `t` in `i16` lanes `4t..4t+3` with the
/// quant unsigned, and the 6-bit row scale is folded with
/// `_mm256_madd_epi16` on those pair sums (`repack.cpp:2757`).
///
/// The fifth bit lives in a separate `qh` plane interleaved the same way
/// as `qs`, so it is one more 32-byte load per `(k, row half)`, read at
/// bit `2 * (k / 4)` for the low nibble and one bit up for the high one.
/// That leaves the quant unsigned in `0..31`, so no bias term is needed.
///
/// # Overflow
///
/// The `i16` stage accumulates four `k` steps before the scale fold. A
/// pair sum is at most `2 * 31 * 127 = 7874`, so four reach `31496`,
/// inside `i16` — the tightest margin of the five kernels here, and the
/// reason `quantize_activations_q8_k`'s clamp to `+-127` is load-bearing
/// rather than incidental.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_q5_kx8_q8_k_avx2(
    packed: &[u8],
    tile: &Q8KActsX4,
    n_cols: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q5_K_BLOCK_ELEMS;
    let na = tile.na;
    let m4b = _mm256_set1_epi8(0x0F);
    let m1 = _mm256_set1_epi8(0x01);
    let m10 = _mm256_set1_epi8(0x10);
    let mut acc = [_mm256_setzero_ps(); Q8K_ACTS_X4_NC];
    let mut acc_min = [_mm256_setzero_ps(); Q8K_ACTS_X4_NC];

    for l in 0..nb {
        let blk = packed.as_ptr().add(l * Q5_KX8_BLOCK_BYTES);
        let d_vec = load_f16x8(blk);
        let dmin_vec = load_f16x8(blk.add(16));
        let scales = std::slice::from_raw_parts(blk.add(32), 96);
        let qh = blk.add(128);
        let qs = blk.add(384);
        let acts = tile.qs.as_ptr().add(l * Q5_K_BLOCK_ELEMS * 4);

        let mut all_scales = [[0u8; 8]; 8];
        let mut all_mins = [[0u8; 8]; 8];
        for sb in 0..8 {
            decode_scales_mins(&scales[sb * 12..], &mut all_scales[sb], &mut all_mins[sb]);
        }
        let mut sc_v = [[[_mm256_setzero_si256(); 2]; 2]; 4];
        for (p, pair) in sc_v.iter_mut().enumerate() {
            for (nib, halves) in pair.iter_mut().enumerate() {
                let s = &all_scales[p * 2 + nib];
                halves[0] = scale_lanes_u8(s, 0);
                halves[1] = scale_lanes_u8(s, 4);
            }
        }
        let mut min_v = [_mm256_setzero_si256(); 8];
        for (sb, slot) in min_v.iter_mut().enumerate() {
            *slot = _mm256_cvtepu8_epi32(_mm_loadl_epi64(all_mins[sb].as_ptr() as *const __m128i));
        }

        for a in 0..na {
            let da = tile.d[l * 4 + a];
            let mut i32acc = [_mm256_setzero_si256(); 2];
            for (p, pair) in sc_v.iter().enumerate() {
                let sh0 = _mm_cvtsi32_si128((p * 2) as i32);
                let sh1 = _mm_cvtsi32_si128((p * 2 + 1) as i32);
                let mut i16acc = [[_mm256_setzero_si256(); 2]; 2];
                for kk in 0..4 {
                    let k = p * 4 + kk;
                    let ka = p * 8 + kk;
                    let a0 = bcast8(acts.add(ka * 32 + a * 8));
                    let a1 = bcast8(acts.add((ka + 4) * 32 + a * 8));
                    for half in 0..2 {
                        let w = _mm256_loadu_si256(qs.add(k * 64 + half * 32) as *const __m256i);
                        // `qh` runs 32 elements to a chunk, so the chunk
                        // index is `k % 4` where `qs`'s is `k`.
                        let hv =
                            _mm256_loadu_si256(qh.add((k % 4) * 64 + half * 32) as *const __m256i);
                        let h0 = _mm256_and_si256(
                            _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(hv, sh0), m1), 4),
                            m10,
                        );
                        let h1 = _mm256_and_si256(
                            _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(hv, sh1), m1), 4),
                            m10,
                        );
                        let v0 = _mm256_or_si256(_mm256_and_si256(w, m4b), h0);
                        let v1 =
                            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(w, 4), m4b), h1);
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
