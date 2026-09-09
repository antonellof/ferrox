use crate::repack::common::*;
use crate::repack::q4_0x4::*;
use crate::Q4_0_BLOCK_ELEMS;
use std::arch::x86_64::*;

use super::{bcast8, load_f16x4, rows4_from_pairs};

/// AVX2 GEMM: one `block_q4_0x4` row-group (4 rows) against a quad of up
/// to four Q8_0 activations. The x86 twin of
/// [`crate::repack::neon::gemm_q4_0x4_q8_0_neon_i8mm`], held against the
/// scalar twin [`crate::repack::q4_0x4::gemm_q4_0x4_acts_x4_scalar_8`].
///
/// # Lane mapping, and where it departs from llama.cpp
///
/// llama.cpp's x86 Q4_0 GEMM is `ggml_gemm_q4_0_8x8_q8_0`
/// (`ggml/src/ggml-cpu/arch/x86/repack.cpp:2026`), which packs **eight**
/// rows and sign-extends nibbles through a lookup
/// (`gemm_q4_b32_8x8_q8_0_lut_avx`, `repack.cpp:641`). ferrox packs Q4_0
/// four rows deep with the ARM convention, XOR-ing each nibble with `8`
/// at pack time (`make_block_q4_0x4`) so NEON can skip the `- 8` bias.
/// So the lane mapping is llama.cpp's — a 32-byte load holds four rows'
/// 8-byte runs, `maddubs` puts row `t` in `i16` lanes `4t..4t+3`, `madd`
/// folds to `2t`, `2t+1` — but the nibble decode is this layout's own:
///
/// XOR-ing the packed byte back with `0x88` recovers the raw `0..15`
/// quant `n`, whose value is `n - 8`. That gives the identity this kernel
/// runs on,
///
/// ```text
/// sum(v * a) = sum(n * a) - 8 * sum(a)
/// ```
///
/// which keeps the weight side **unsigned** for `maddubs` and needs no
/// `_mm256_sign_epi8` trick at all, so no `-128` hazard exists here.
/// The scalar twin's `>> 4` is exact for the same reason: it divides a
/// value that is an exact multiple of 16.
///
/// # Overflow
///
/// An `i16` lane holds `n0*a0 + n1*a1` over two elements each: at most
/// `4 * 15 * 127 = 7620` before the bias term and `4 * 8 * 127 = 4064`
/// after it, and the two `k` steps of a 32-element block sum to `8128`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_q4_0x4_q8_0_avx2(
    packed: &[u8],
    tile: &Q8ActsX4,
    n_cols: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q4_0_BLOCK_ELEMS;
    let na = tile.na;
    let m4b = _mm256_set1_epi8(0x0F);
    let unxor = _mm256_set1_epi8(0x88u8 as i8);
    let eight = _mm256_set1_epi8(8);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm_setzero_ps(); Q8K_ACTS_X4_NC];

    for l in 0..nb {
        let blk = packed.as_ptr().add(l * Q4_0X4_BLOCK_BYTES);
        let d4 = load_f16x4(blk);
        let qs = blk.add(8);
        let acts = tile.qs.as_ptr().add(l * Q4_0_BLOCK_ELEMS * 4);

        for a in 0..na {
            let mut i16acc = _mm256_setzero_si256();
            for k in 0..(Q4_0_BLOCK_ELEMS / 16) {
                let x = _mm256_loadu_si256(qs.add(k * 32) as *const __m256i);
                let orig = _mm256_xor_si256(x, unxor);
                let n0 = _mm256_and_si256(orig, m4b);
                let n1 = _mm256_and_si256(_mm256_srli_epi16(orig, 4), m4b);
                // Canonical elements `k*8 + i` and `k*8 + i + 16` live at
                // runs `k` and `k + 2` of the quad.
                let a0 = bcast8(acts.add(k * 32 + a * 8));
                let a1 = bcast8(acts.add((k + 2) * 32 + a * 8));
                let p =
                    _mm256_add_epi16(_mm256_maddubs_epi16(n0, a0), _mm256_maddubs_epi16(n1, a1));
                let bias = _mm256_add_epi16(
                    _mm256_maddubs_epi16(eight, a0),
                    _mm256_maddubs_epi16(eight, a1),
                );
                i16acc = _mm256_add_epi16(i16acc, _mm256_sub_epi16(p, bias));
            }
            let da = _mm_set1_ps(tile.d[l * 4 + a]);
            acc[a] = _mm_fmadd_ps(
                _mm_cvtepi32_ps(rows4_from_pairs(_mm256_madd_epi16(i16acc, ones))),
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
