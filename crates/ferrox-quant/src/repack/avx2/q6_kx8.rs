use crate::repack::common::*;
use crate::repack::q6_kx8::*;
use crate::Q6_K_BLOCK_ELEMS;
use std::arch::x86_64::*;

use super::{bcast8, load_f16x8, rows8_from_pairs, scale_lanes_i8};

/// AVX2 GEMM: one `block_q6_Kx8` row-group (8 rows) against a quad of up
/// to four Q8_K activations. The x86 twin of
/// [`crate::repack::neon::gemm_q6_kx8_q8_k_neon_i8mm`], held against the
/// scalar twin [`crate::repack::q6_kx8::gemm_q6_kx8_acts_x4_scalar_8`].
///
/// # Lane mapping
///
/// llama.cpp has no x86 Q6_K repack kernel — `arch/x86/repack.cpp`
/// implements `8x8` GEMMs for Q4_0, Q4_K, Q2_K, IQ4_NL and MXFP4 only —
/// so the shape mirrored here is the one llama.cpp uses for the K-quants
/// it does cover, `ggml_gemm_q4_K_8x8_q8_K`
/// (`ggml/src/ggml-cpu/arch/x86/repack.cpp:2042`): a 32-byte load holds
/// four rows' 8-byte runs, `_mm256_maddubs_epi16` puts row `t` in `i16`
/// lanes `4t..4t+3` with the **quant** as the unsigned operand, and the
/// per-16 scale is folded with `_mm256_madd_epi16` on those pair sums —
/// `repack.cpp:2757`.
///
/// Q6_K's own two wrinkles:
///
/// * the 6-bit quant is `((qh >> shift) & 3) << 4 | (ql & 15)`, an
///   unsigned `0..63` whose value is `u - 32`. Keeping `u` unsigned for
///   `maddubs` and subtracting `32 * sum(a)` separately is what avoids
///   a sign-extend pass, the same identity the Q4_0 kernel uses;
/// * the `qh` chunk index is the SAME for the low and high halves of a
///   `k` step (both are `(k / 8) * 4 + (k % 4)`), so one `qh` load feeds
///   both, read at shifts `s` and `s + 4`.
///
/// # Overflow
///
/// After the bias subtraction an `i16` lane holds a two-element dot with
/// `|q| <= 32`, at most `2 * 32 * 127 = 8128`; the scale fold then
/// produces at most `2 * 8128 * 127` per `i32` lane and 32 of those per
/// super-block, which is `66M`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_q6_kx8_q8_k_avx2(
    packed: &[u8],
    tile: &Q8KActsX4,
    n_cols: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q6_K_BLOCK_ELEMS;
    let na = tile.na;
    let m4b = _mm256_set1_epi8(0x0F);
    let m3 = _mm256_set1_epi8(0x03);
    let m30 = _mm256_set1_epi8(0x30);
    let bias32 = _mm256_set1_epi8(32);
    let mut acc = [_mm256_setzero_ps(); Q8K_ACTS_X4_NC];

    for l in 0..nb {
        let blk = packed.as_ptr().add(l * Q6_KX8_BLOCK_BYTES);
        let d_vec = load_f16x8(blk);
        let scales = blk.add(16);
        let ql = blk.add(144);
        let qh = blk.add(1168);
        let acts = tile.qs.as_ptr().add(l * Q6_K_BLOCK_ELEMS * 4);

        // [k][low or high 64-element half][row half], hoisted: the
        // scales do not depend on which activation is being dotted.
        let mut sc_v = [[[_mm256_setzero_si256(); 2]; 2]; 16];
        for (k, per_k) in sc_v.iter_mut().enumerate() {
            let base_l = (k / 8) * 128 + (k % 8) * 8;
            let si_l = base_l / 16;
            for half in 0..2 {
                per_k[0][half] = scale_lanes_i8(scales, si_l * 8 + half * 4);
                per_k[1][half] = scale_lanes_i8(scales, (si_l + 4) * 8 + half * 4);
            }
        }

        for a in 0..na {
            let mut i32acc = [_mm256_setzero_si256(); 2];
            for (k, per_k) in sc_v.iter().enumerate() {
                let base_l = (k / 8) * 128 + (k % 8) * 8;
                // Canonical element `base + i` lives at run `base / 8`,
                // quad row `a`, lane `i`; the high half is 64 elements
                // (8 runs) later.
                let run_l = base_l / 8;
                let sh_l = _mm_cvtsi32_si128((((k % 8) / 4) * 2) as i32);
                let sh_h = _mm_cvtsi32_si128(((((k % 8) / 4) * 2) + 4) as i32);
                let qh_chunk = (k / 8) * 4 + (k % 4);
                let a_l = bcast8(acts.add(run_l * 32 + a * 8));
                let a_h = bcast8(acts.add((run_l + 8) * 32 + a * 8));
                for half in 0..2 {
                    let qlv = _mm256_loadu_si256(ql.add(k * 64 + half * 32) as *const __m256i);
                    let qhv =
                        _mm256_loadu_si256(qh.add(qh_chunk * 64 + half * 32) as *const __m256i);
                    let u_l = _mm256_or_si256(
                        _mm256_and_si256(qlv, m4b),
                        _mm256_and_si256(
                            _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qhv, sh_l), m3), 4),
                            m30,
                        ),
                    );
                    let u_h = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(qlv, 4), m4b),
                        _mm256_and_si256(
                            _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qhv, sh_h), m3), 4),
                            m30,
                        ),
                    );
                    let p_l = _mm256_sub_epi16(
                        _mm256_maddubs_epi16(u_l, a_l),
                        _mm256_maddubs_epi16(bias32, a_l),
                    );
                    let p_h = _mm256_sub_epi16(
                        _mm256_maddubs_epi16(u_h, a_h),
                        _mm256_maddubs_epi16(bias32, a_h),
                    );
                    i32acc[half] = _mm256_add_epi32(
                        i32acc[half],
                        _mm256_add_epi32(
                            _mm256_madd_epi16(p_l, per_k[0][half]),
                            _mm256_madd_epi16(p_h, per_k[1][half]),
                        ),
                    );
                }
            }
            acc[a] = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(rows8_from_pairs(i32acc[0], i32acc[1])),
                _mm256_mul_ps(d_vec, _mm256_set1_ps(tile.d[l * 4 + a])),
                acc[a],
            );
        }
    }

    for a in 0..na {
        let mut v = [0f32; 8];
        _mm256_storeu_ps(v.as_mut_ptr(), acc[a]);
        for (j, got) in v.iter().enumerate() {
            out[j * na + a] = *got;
        }
    }
}
