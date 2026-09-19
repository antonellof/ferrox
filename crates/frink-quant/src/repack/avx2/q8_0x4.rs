use crate::repack::common::*;
use crate::repack::q8_0x4::*;
use crate::Q8_0_BLOCK_ELEMS;
use std::arch::x86_64::*;

use super::{bcast8, load_f16x4, rows4_from_pairs};

/// AVX2 GEMM: one `block_q8_0x4` row-group (4 rows) against a quad of up
/// to four Q8_0 activations. The x86 twin of
/// [`crate::repack::neon::gemm_q8_0x4_q8_0_neon_i8mm`], held against the
/// scalar twin [`crate::repack::q8_0x4::gemm_q8_0x4_acts_x4_scalar_8`].
///
/// # Lane mapping
///
/// llama.cpp has **no** x86 `q8_0_4x8` kernel — `arch/x86/repack.cpp`
/// repacks Q8_0 nowhere and only implements the `8x8` shapes for Q4_0,
/// Q4_K, Q2_K, IQ4_NL and MXFP4 — so this is not a transcription. What it
/// does mirror is llama.cpp's signed-8-bit dot idiom for AVX2 without
/// VNNI, `mul_sum_i8_pairs_acc_int32x8` at
/// `ggml/src/ggml-cpu/arch/x86/repack.cpp:165`:
///
/// ```text
/// ax = _mm256_sign_epi8(x, x);   // |weight|, an unsigned operand
/// sy = _mm256_sign_epi8(y, x);   // activation, signed by the weight
/// madd(maddubs(ax, sy), ones)
/// ```
///
/// The interleaved layout puts one row's 8-byte run per 8 bytes, so one
/// 32-byte load covers all four rows and [`bcast8`] puts one
/// activation's run opposite them; `maddubs` then leaves lanes `4t..4t+3`
/// holding row `t`, and `madd` against ones folds those to `2t`, `2t+1`.
///
/// # Why `-128` cannot reach the sign trick
///
/// `_mm256_sign_epi8(y, x)` negates `y` where `x` is negative, and
/// negating `-128` wraps to itself. Both sides here are ggml-shaped int8:
/// [`crate::quantize_activations_q8`] clamps to `+-127` and
/// [`crate::prepare_q8_acts_x4`] `debug_assert`s that the quad it hands
/// over holds no `-128`, so the negated operand is always in range. The
/// magnitude operand may be `-128` and is fine: `|-128|` reads back as
/// unsigned `128`, and `128 * 127 * 2 = 32512` still fits `i16`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_q8_0x4_q8_0_avx2(
    packed: &[u8],
    tile: &Q8ActsX4,
    n_cols: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    let na = tile.na;
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm_setzero_ps(); Q8K_ACTS_X4_NC];

    for l in 0..nb {
        let blk = packed.as_ptr().add(l * Q8_0X4_BLOCK_BYTES);
        let d4 = load_f16x4(blk);
        let qs = blk.add(8);
        let acts = tile.qs.as_ptr().add(l * Q8_0_BLOCK_ELEMS * 4);

        for a in 0..na {
            let mut i32acc = _mm256_setzero_si256();
            for k in 0..(Q8_0_BLOCK_ELEMS / 8) {
                let w = _mm256_loadu_si256(qs.add(k * 32) as *const __m256i);
                // Canonical element `e = k * 8 + i` lives at run `k`,
                // quad row `a`, lane `i`.
                let av = bcast8(acts.add(k * 32 + a * 8));
                let ax = _mm256_sign_epi8(w, w);
                let sy = _mm256_sign_epi8(av, w);
                i32acc = _mm256_add_epi32(
                    i32acc,
                    _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), ones),
                );
            }
            let da = _mm_set1_ps(tile.d[l * 4 + a]);
            acc[a] = _mm_fmadd_ps(
                _mm_cvtepi32_ps(rows4_from_pairs(i32acc)),
                _mm_mul_ps(d4, da),
                acc[a],
            );
        }
    }

    for a in 0..na {
        let mut v = [0f32; 4];
        _mm_storeu_ps(v.as_mut_ptr(), acc[a]);
        for (j, got) in v.iter().enumerate() {
            out[j * na + a] = *got;
        }
    }
}
