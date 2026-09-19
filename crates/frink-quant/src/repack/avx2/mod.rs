//! x86_64 AVX2 kernels for the interleaved repack tier, one file per
//! kind family. The dispatchers in `super::*` call these as
//! `avx2::<fn>`, which the re-exports below keep true after the split.
//!
//! # The lane mapping, once
//!
//! Every packed layout here stores one row's `blocklen`-element run
//! contiguously, `ncols_interleaved` rows in a row, so at `blocklen = 8`
//! a 32-byte load holds four consecutive rows. That single fact decides
//! the whole register plan, and it is llama.cpp's plan too
//! (`ggml/src/ggml-cpu/arch/x86/repack.cpp`):
//!
//! 1. [`bcast8`] puts one activation's 8-byte run opposite all four.
//! 2. `_mm256_maddubs_epi16` folds each row's run into four `i16` pair
//!    sums: lanes `4t..4t+3` belong to row `t`. The weight side is the
//!    *unsigned* operand, which is why the K-quant kernels below dot the
//!    raw `0..2^b-1` quant and subtract the bias term separately rather
//!    than sign-extending first.
//! 3. `_mm256_madd_epi16` against the row's scale folds those into two
//!    `i32` per row (lanes `2t`, `2t+1`) — llama.cpp does the same fold
//!    at `repack.cpp:2757`, `iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0,
//!    scale_014589CD_0)`.
//! 4. [`rows4_from_pairs`] / [`rows8_from_pairs`] collapse those pairs
//!    into one `i32` per row, in row order.
//!
//! The `i16` stage is the one that can overflow, so each kernel states
//! its bound where it accumulates.

// Every range loop in this module is a coordinate — `a` is the quad row,
// `half` the 4-row half of a 32-byte load, `k` the 8-element run — and
// each body indexes two or three arrays plus a raw pointer with it.
// Rewriting those as zipped iterators moves the coordinate out of the
// code and into the zip order, which is where a lane bug would hide.
// `lib.rs` already carries this allow at one such site; the boundary
// here is the kernels, and it is deliberate rather than inherited.
#![allow(clippy::needless_range_loop)]

use half::f16;
use std::arch::x86_64::*;

mod q4_0x4;
mod q4_kx8;
mod q5_kx8;
mod q6_kx8;
mod q8_0x4;

pub(crate) use q4_0x4::*;
pub(crate) use q4_kx8::*;
pub(crate) use q5_kx8::*;
pub(crate) use q6_kx8::*;
pub(crate) use q8_0x4::*;

/// Broadcast the 8 bytes at `p` into all four 64-bit lanes: one
/// activation run opposite four interleaved weight rows.
///
/// # Safety
/// `p` must be readable for 8 bytes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn bcast8(p: *const i8) -> __m256i {
    _mm256_set1_epi64x(std::ptr::read_unaligned(p as *const i64))
}

/// Read `n` f16 at `p` into an f32 array, lane `j` = row `j`.
///
/// Scalar on purpose: F16C is a separate CPUID bit from AVX2 and this
/// runs once per super-block, not once per dot.
///
/// # Safety
/// `p` must be readable for `2 * n` bytes and `n <= 8`.
#[inline]
unsafe fn f16x_to_f32(p: *const u8, n: usize) -> [f32; 8] {
    let mut out = [0f32; 8];
    for (j, slot) in out.iter_mut().enumerate().take(n) {
        *slot = f16::from_le_bytes([*p.add(j * 2), *p.add(j * 2 + 1)]).to_f32();
    }
    out
}

/// Eight f16 at `p` as an `f32x8`, lane `j` = row `j`.
///
/// # Safety
/// `p` must be readable for 16 bytes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_f16x8(p: *const u8) -> __m256 {
    _mm256_loadu_ps(f16x_to_f32(p, 8).as_ptr())
}

/// Four f16 at `p` as an `f32x4`, lane `j` = row `j`.
///
/// # Safety
/// `p` must be readable for 8 bytes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_f16x4(p: *const u8) -> __m128 {
    _mm_loadu_ps(f16x_to_f32(p, 4).as_ptr())
}

/// Collapse an `i32x8` holding two lanes per row for four rows
/// (`2t`, `2t+1` are row `t`) into one `i32` per row, in row order.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn rows4_from_pairs(x: __m256i) -> __m128i {
    // hadd(x, x) = [r0, r1, r0, r1 | r2, r3, r2, r3] across the two
    // 128-bit halves, so the two low quadwords already are the answer.
    let h = _mm256_hadd_epi32(x, x);
    _mm_unpacklo_epi64(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1))
}

/// Collapse two `i32x8` pair accumulators — `lo` for rows 0..4, `hi` for
/// rows 4..8 — into one `i32x8`, lane `j` = row `j`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn rows8_from_pairs(lo: __m256i, hi: __m256i) -> __m256i {
    // hadd interleaves the two sources per 128-bit half:
    // [r0, r1, r4, r5 | r2, r3, r6, r7].
    let h = _mm256_hadd_epi32(lo, hi);
    _mm256_permutevar8x32_epi32(h, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7))
}

/// Widen eight per-row `u8` scales to the `i16` lane layout
/// `_mm256_maddubs_epi16` leaves behind: four copies of row `base + t` at
/// lanes `4t..4t+3`, for `t` in `0..4`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn scale_lanes_u8(scales: &[u8; 8], base: usize) -> __m256i {
    let mut w = [0i16; 16];
    for t in 0..4 {
        let v = scales[base + t] as i16;
        for u in 0..4 {
            w[t * 4 + u] = v;
        }
    }
    _mm256_loadu_si256(w.as_ptr() as *const __m256i)
}

/// [`scale_lanes_u8`] for the signed per-16 scales Q6_K carries.
///
/// # Safety
/// `p` must be readable for `base + 4` bytes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn scale_lanes_i8(p: *const u8, base: usize) -> __m256i {
    let mut w = [0i16; 16];
    for t in 0..4 {
        let v = *p.add(base + t) as i8 as i16;
        for u in 0..4 {
            w[t * 4 + u] = v;
        }
    }
    _mm256_loadu_si256(w.as_ptr() as *const __m256i)
}
