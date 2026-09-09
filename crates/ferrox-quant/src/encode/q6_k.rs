//! The Q6_K weight encoder: a transcription of llama.cpp b7650's
//! `quantize_row_q6_K_ref` (`ggml/src/ggml-quants.c:1692`).
//!
//! Q6_K is NOT built the way Q4_K and Q5_K are, and that is the whole
//! reason it is its own module rather than another [`QkFit`] row:
//!
//! * The fit is **symmetric**. There is no per-sub-block minimum and no
//!   `dmin`: [`super::fit::make_qx_quants`] fits `x ~= scale * (L - 32)`
//!   with the codes centred on 32, over **16**-element groups, not 32.
//!   So a Q6_K super-block has 16 groups where a Q4_K one has 8.
//! * The 16 group scales are quantized to **signed 8-bit** against a
//!   single `d`, not to 6 bits packed against a `d`/`dmin` pair. That
//!   is why `iscale` here is `-128/max_scale` and why the stored scales
//!   are `int8_t`: a group whose scale has the opposite sign to the
//!   super-block's extreme lands on the negative half of the range and
//!   is meant to.
//! * A super-block whose scales are all under `GROUP_MAX_EPS` is
//!   written as **210 zero bytes**, and llama.cpp's `memset` of the
//!   whole block is what makes that exact. An encoder that wrote a
//!   scale of 1.0 and zero codes would dequantize identically and
//!   differ in bytes, which is the failure the goldens exist to catch.
//!
//! The `ql`/`qh` split is the other thing worth stating, because a
//! plausible rewrite gets it wrong and still round-trips within the
//! error bound: the 256 codes are walked in **two** groups of 128, and
//! within a group the four quarters `l`, `l+32`, `l+64`, `l+96` share
//! one `qh` byte two bits each, while their low nibbles pair up as
//! `(q1, q3)` and `(q2, q4)` -- NOT `(q1, q2)`. `dequant_q6_k` in this
//! crate reads it back with the same pairing.

use half::f16;

use super::fit::{make_qx_quants, nearest_int, GROUP_MAX_EPS};
use crate::{Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS};

/// Elements per Q6_K group, and groups per super-block.
const GROUP_ELEMS: usize = 16;
const GROUPS: usize = Q6_K_BLOCK_ELEMS / GROUP_ELEMS; // 16

/// `nmax` for the symmetric fit: codes run `-32..=31`, stored `+32`.
const NMAX: i32 = 32;

/// `rmse_type` Q6_K passes to `make_qx_quants`: `1`, the `x^2` weight.
const RMSE_TYPE: i32 = 1;

/// Encodes one Q6_K super-block (exactly [`Q6_K_BLOCK_ELEMS`] values)
/// and appends its [`Q6_K_BLOCK_BYTES`] bytes to `out`.
pub fn encode_block_q6_k(block: &[f32; Q6_K_BLOCK_ELEMS], out: &mut Vec<u8>) {
    // `l` is carried from the fit into the recode below, and the recode
    // skips any group whose reconstructed `d` rounded to zero (`if (!d)
    // continue;` upstream). The codes then written are the ones the fit
    // left behind -- NOT zeros. Same rule, same reason, as the Q4_K/Q5_K
    // stage 3 in `fit.rs`.
    let mut l = [0i8; Q6_K_BLOCK_ELEMS];
    let mut scales = [0f32; GROUPS];

    let mut max_scale = 0f32;
    let mut max_abs_scale = 0f32;
    for ib in 0..GROUPS {
        let lo = GROUP_ELEMS * ib;
        let scale = make_qx_quants(
            &block[lo..lo + GROUP_ELEMS],
            &mut l[lo..lo + GROUP_ELEMS],
            NMAX,
            RMSE_TYPE,
            None,
        );
        scales[ib] = scale;
        let abs_scale = scale.abs();
        if abs_scale > max_abs_scale {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    if max_abs_scale < GROUP_MAX_EPS {
        // `memset(&y[i], 0, sizeof(block_q6_K))` followed by an
        // explicit `d = 0`. Both halves are the same bytes here, and a
        // scale of 1.0 with zero codes would dequantize identically --
        // which is exactly why this is a byte fact and not a value one.
        out.resize(out.len() + Q6_K_BLOCK_BYTES, 0);
        return;
    }

    let iscale = -128.0f32 / max_scale;
    let d = f16::from_f32(1.0 / iscale);
    let mut qscales = [0i8; GROUPS];
    for ib in 0..GROUPS {
        // Upstream is `y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]))`,
        // where the MIN happens in `int` and the result is then stored
        // in an `int8_t`. The clamp therefore comes BEFORE the
        // narrowing here, the opposite order from the Q4_K/Q5_K scale
        // packing next door, where C's `uint8_t ls = nearest_int(...)`
        // narrows first. Two nearly identical lines with opposite
        // orders is exactly why they are not shared: `as i8` reproduces
        // C's wrap for the sub-(-128) case that `MIN` does not catch.
        qscales[ib] = nearest_int(iscale * scales[ib]).min(127) as i8;
    }

    for j in 0..GROUPS {
        let dj = d.to_f32() * qscales[j] as f32;
        if dj == 0.0 {
            continue;
        }
        for ii in 0..GROUP_ELEMS {
            let idx = GROUP_ELEMS * j + ii;
            l[idx] = (nearest_int(block[idx] / dj).clamp(-32, 31) + 32) as i8;
        }
    }

    out.reserve(Q6_K_BLOCK_BYTES);
    let mut ql = [0u8; Q6_K_BLOCK_ELEMS / 2];
    let mut qh = [0u8; Q6_K_BLOCK_ELEMS / 4];
    for (half, j) in (0..Q6_K_BLOCK_ELEMS).step_by(128).enumerate() {
        let (qlo, qho) = (half * 64, half * 32);
        for i in 0..32 {
            let q1 = (l[j + i] & 0xF) as u8;
            let q2 = (l[j + i + 32] & 0xF) as u8;
            let q3 = (l[j + i + 64] & 0xF) as u8;
            let q4 = (l[j + i + 96] & 0xF) as u8;
            ql[qlo + i] = q1 | (q3 << 4);
            ql[qlo + i + 32] = q2 | (q4 << 4);
            qh[qho + i] = ((l[j + i] >> 4) as u8)
                | (((l[j + i + 32] >> 4) as u8) << 2)
                | (((l[j + i + 64] >> 4) as u8) << 4)
                | (((l[j + i + 96] >> 4) as u8) << 6);
        }
    }
    // The on-disk order is ql, qh, scales, d -- the scales sit BETWEEN
    // the code planes and the super-scale, unlike Q4_K/Q5_K where the
    // scales lead. `dequant_q6_k` reads the same order.
    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    out.extend_from_slice(&qscales.map(|v| v as u8));
    out.extend_from_slice(&d.to_le_bytes());
}

/// Encodes a whole row (or any slice whose length is a multiple of
/// [`Q6_K_BLOCK_ELEMS`]) into Q6_K super-blocks, appending to `out`.
///
/// Returns `None` when `src.len()` is not a multiple of the super-block
/// size. llama.cpp answers that case by silently *changing type* --
/// `convert_incompatible_tensor` rewrites a Q6_K tensor with an awkward
/// row length to Q8_0, and to F16 if that does not fit either. ferrox
/// HAS a Q8_0 encoder, but the fallback is llama.cpp's decision about a
/// tensor, not this function's about a row: `encode_row_*` is told
/// which format to write, so silently writing a different one here
/// would produce bytes the caller's plan sized for Q6_K. The refusal
/// belongs to the row and the fallback, if it is ever wanted, belongs
/// to the planner.
pub fn encode_row_q6_k(src: &[f32], out: &mut Vec<u8>) -> Option<()> {
    let (blocks, rest) = src.as_chunks::<Q6_K_BLOCK_ELEMS>();
    if !rest.is_empty() {
        return None;
    }
    out.reserve(blocks.len() * Q6_K_BLOCK_BYTES);
    for block in blocks {
        encode_block_q6_k(block, out);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant_q6_k;
    use crate::encode::testdata::k_quant_fixture;

    /// llama.cpp's own bytes for `k_quant_fixture`.
    ///
    /// Produced by the C harness described in the PR body: it links
    /// llama.cpp b7650's `libggml-base` and calls the exported
    /// `quantize_row_q6_K_ref` on the f32s
    /// `encode::testdata::dump_the_fixture_the_c_harness_reads` writes.
    /// The same harness also calls `ggml_quantize_chunk(GGML_TYPE_Q6_K,
    /// ...)` -- the entry point `llama-quantize` itself goes through --
    /// and asserts the two agree, so this is what the real tool writes
    /// and not merely what a reference function does.
    ///
    /// An encoder that is within Q6_K's error bound passes any
    /// tolerance test and still writes a different file. Only this
    /// catches that.
    ///
    /// One thing this golden does NOT catch, measured rather than
    /// assumed: substituting `f32::round` for `fit::nearest_int` leaves
    /// it green, while it turns the Q4_K and Q5_K goldens red. Q6_K's
    /// 16-element groups and `x^2` weight simply never land on an exact
    /// tie over this fixture. The property is covered --
    /// `fit::tests::nearest_int_rounds_ties_to_even_like_the_fpu` goes
    /// red, and there is only ONE `nearest_int` for all three encoders
    /// to share -- but claiming this golden covers it would be false.
    const LLAMA_CPP_Q6_K_GOLDEN: [u8; 4 * Q6_K_BLOCK_BYTES] = [
        0xa4, 0x58, 0xde, 0x50, 0x5e, 0x68, 0xb7, 0x71, 0xa6, 0x0d, 0xf7, 0x1d, 0x04, 0xd5, 0x2c,
        0x7b, 0x8e, 0xfb, 0xe1, 0x3b, 0xa1, 0x00, 0x96, 0x52, 0x08, 0x88, 0x39, 0x95, 0xa4, 0x16,
        0x1c, 0x72, 0x0d, 0x43, 0x9f, 0x22, 0x0d, 0xb7, 0x05, 0x2a, 0x19, 0xc3, 0x88, 0x2b, 0xe4,
        0x8f, 0x65, 0x00, 0xdf, 0xf6, 0xf6, 0xc2, 0x74, 0xc2, 0x53, 0x9c, 0x28, 0x2c, 0xee, 0x29,
        0x0d, 0x60, 0x54, 0x4c, 0xf2, 0xf7, 0x52, 0x57, 0x82, 0xa6, 0xbf, 0xc4, 0x43, 0x17, 0x91,
        0xae, 0x71, 0x95, 0xf8, 0x1e, 0xbf, 0x3f, 0xa8, 0x03, 0xf7, 0x98, 0x9f, 0xb0, 0xef, 0xb8,
        0xd0, 0xee, 0x1a, 0xd2, 0xdc, 0xbf, 0x55, 0x6f, 0x62, 0x09, 0xcd, 0x8f, 0x8a, 0x70, 0xf5,
        0xee, 0x39, 0x2c, 0xc0, 0x3e, 0xac, 0x0f, 0x17, 0xd7, 0xf4, 0x0e, 0xfc, 0x71, 0x6e, 0x69,
        0x72, 0x77, 0x1b, 0x7a, 0x10, 0x52, 0xc5, 0xcf, 0x6c, 0xfd, 0x28, 0x14, 0x2c, 0x2b, 0x7a,
        0x4a, 0x38, 0x29, 0x8d, 0x48, 0x37, 0x7d, 0x19, 0xe0, 0x48, 0xb8, 0xe7, 0xac, 0xac, 0x98,
        0xfa, 0xea, 0xba, 0x84, 0x65, 0x09, 0x30, 0xe3, 0x0a, 0xdf, 0x41, 0xfe, 0xb4, 0x17, 0x40,
        0x28, 0x81, 0x06, 0x95, 0xa9, 0x9d, 0x7c, 0x33, 0xbb, 0xb9, 0x42, 0x0a, 0x6f, 0x6e, 0x78,
        0x34, 0x43, 0xff, 0xe9, 0x24, 0xdd, 0x88, 0xf3, 0x84, 0x9e, 0x0f, 0x1b, 0x0a, 0xf8, 0xe8,
        0xe9, 0x33, 0x34, 0x7c, 0x80, 0x0a, 0x0a, 0xe7, 0xe7, 0x32, 0x34, 0x88, 0x8f, 0x00, 0x84,
        0x40, 0xa0, 0x80, 0x70, 0x50, 0x70, 0x60, 0x10, 0xd0, 0x80, 0x20, 0x00, 0xc0, 0xb0, 0xa0,
        0x30, 0x60, 0x80, 0x80, 0xa0, 0x10, 0x00, 0x30, 0x80, 0x20, 0x00, 0x00, 0xb0, 0x00, 0x00,
        0x30, 0x50, 0x80, 0x20, 0x10, 0xd0, 0x60, 0x50, 0x20, 0x60, 0xa0, 0x30, 0xf0, 0x60, 0x40,
        0x70, 0x10, 0x00, 0xd0, 0x80, 0x80, 0xe0, 0x10, 0xf0, 0xd0, 0xa0, 0x10, 0x30, 0xb0, 0xa0,
        0x10, 0xc0, 0xb0, 0x00, 0x45, 0x1b, 0x9b, 0x3c, 0xc7, 0x11, 0x61, 0xc1, 0x0a, 0x59, 0xc5,
        0x60, 0xaf, 0x2f, 0x9c, 0xdb, 0xa5, 0x49, 0x4d, 0xcf, 0x0b, 0xa6, 0x02, 0xb3, 0x95, 0xf0,
        0x15, 0x41, 0x11, 0xde, 0x03, 0xb6, 0x07, 0xc2, 0x72, 0x88, 0xa1, 0xe3, 0x57, 0x80, 0xfe,
        0xc0, 0x8c, 0x86, 0x30, 0xae, 0xe8, 0x45, 0xe5, 0x0f, 0x9e, 0x99, 0xd1, 0xfb, 0x33, 0x08,
        0x07, 0x69, 0x11, 0xc8, 0xed, 0x12, 0x4b, 0x7f, 0x50, 0x40, 0x10, 0x00, 0x10, 0x00, 0x10,
        0x00, 0x00, 0x40, 0x10, 0x50, 0x40, 0x40, 0x10, 0x40, 0x10, 0x40, 0x40, 0x00, 0x00, 0x00,
        0x10, 0x40, 0x00, 0x10, 0x40, 0x00, 0x10, 0x00, 0x00, 0x50, 0x06, 0x4f, 0x8e, 0xc1, 0xc3,
        0x7c, 0xad, 0x01, 0x5d, 0x2a, 0x75, 0x71, 0x24, 0xfd, 0x82, 0x5d, 0x26, 0xcb, 0xfd, 0x6e,
        0x92, 0xfa, 0x3b, 0x22, 0x17, 0xef, 0xae, 0x00, 0x8e, 0x42, 0xeb, 0xfc, 0x00, 0x00, 0x42,
        0x42, 0x1f, 0x1f, 0xe2, 0xe0, 0x0a, 0x0b, 0x1a, 0x1a, 0xc9, 0x36, 0x94, 0x80, 0xdd, 0x83,
        0xa0, 0x6f, 0xb6, 0x7d, 0x4e, 0xe1, 0x0e, 0xa0, 0x72, 0x8f, 0x00, 0x16, 0xd8, 0x90, 0xe6,
        0x63, 0x5a, 0x10, 0x5a, 0x88, 0xab, 0x7e, 0x60, 0x01, 0x1e, 0x40, 0x2a, 0xea, 0xcd, 0xe6,
        0x72, 0x7e, 0xbe, 0x51, 0x2a, 0x17, 0xe0, 0x0b, 0xe5, 0x3a, 0x74, 0xe1, 0xbf, 0x52, 0x06,
        0x67, 0xed, 0xba, 0x33, 0x21, 0x87, 0x14, 0xd6, 0xdb, 0xa1, 0x29, 0xb9, 0x15, 0x7e, 0x1a,
        0xd6, 0x84, 0xf5, 0x4b, 0x47, 0x34, 0xbb, 0x67, 0xa7, 0x0e, 0x98, 0x13, 0x21, 0x19, 0xdb,
        0x20, 0x24, 0x29, 0xb9, 0x93, 0x19, 0xe1, 0xec, 0x9f, 0xcc, 0x86, 0x01, 0xa1, 0x75, 0xa5,
        0x1a, 0x70, 0x89, 0xaa, 0xca, 0xf3, 0x20, 0x7b, 0x0a, 0xe3, 0x06, 0x53, 0x50, 0xb9, 0x17,
        0xd7, 0x0d, 0xb7, 0x19, 0xf5, 0xa6, 0x8a, 0x93, 0x55, 0xe8, 0x6c, 0x4d, 0xf9, 0x08, 0x6d,
        0xa7, 0x0a, 0xb1, 0x4c, 0x41, 0x77, 0x4b, 0x6c, 0xab, 0x75, 0x43, 0x74, 0x70, 0xf7, 0x6c,
        0x14, 0x68, 0xef, 0x07, 0x64, 0x0a, 0x0d, 0x4e, 0xf9, 0xb1, 0x00, 0xf5, 0x6a, 0xc9, 0xe7,
        0x81, 0xab, 0x39, 0x8d, 0x8d, 0x38, 0x89, 0x24, 0xd4, 0xe8, 0x88, 0x14, 0x76, 0x19, 0xaa,
        0x4c, 0xf0, 0x3d, 0xbe, 0x82, 0x9f, 0xf0, 0x1a, 0xdf, 0x9a, 0x0a, 0xd0, 0xe9, 0xa6, 0xe5,
        0x94, 0xf6, 0xf8, 0x7c, 0x71, 0x2f, 0xc1, 0x92, 0x0a, 0x30, 0x5a, 0xde, 0x0a, 0xf6, 0x00,
        0x00, 0xd0, 0xcd, 0x82, 0x80, 0xf6, 0x0a, 0xe7, 0x1a, 0xd0, 0x33, 0x7d, 0x7c, 0xfb, 0x83,
        0xdc, 0x24, 0x64, 0x60, 0xcf, 0xf7, 0x54, 0xce, 0x39, 0x9b, 0x0b, 0x52, 0x64, 0x30, 0x07,
        0x73, 0xad, 0x9b, 0x69, 0xca, 0x2f, 0x85, 0xea, 0x6f, 0x53, 0xfb, 0xd6, 0x60, 0x0d, 0x6b,
        0xd9, 0x14, 0xe3, 0x70, 0x0d, 0x7d, 0x51, 0x1f, 0xf7, 0xe0, 0xba, 0xb8, 0x1c, 0x94, 0x90,
        0x48, 0xef, 0xaa, 0xde, 0xe6, 0x18, 0x01, 0x36, 0xd3, 0x04, 0x08, 0x70, 0x51, 0x35, 0x30,
        0x29, 0x73, 0x4c, 0x61, 0xa5, 0x31, 0x7a, 0x60, 0x0e, 0x47, 0x50, 0x1d, 0x96, 0x27, 0x71,
        0xab, 0x3c, 0xe2, 0x80, 0xab, 0x14, 0xfa, 0xd9, 0x52, 0x63, 0x14, 0x88, 0xed, 0x66, 0xf7,
        0x32, 0x77, 0xce, 0xcd, 0xa1, 0xbc, 0x4b, 0x9e, 0xc0, 0x3d, 0x4b, 0x89, 0x55, 0x15, 0x47,
        0xc8, 0x4a, 0xf0, 0x7f, 0x8b, 0x16, 0xa9, 0xbb, 0xcb, 0x6b, 0x78, 0xdd, 0x88, 0x1e, 0xf1,
        0x36, 0x7a, 0x71, 0x42, 0x14, 0xa0, 0xe5, 0x91, 0xf6, 0xb0, 0x21, 0x08, 0x03, 0xd8, 0x13,
        0x3d, 0x45, 0x58, 0xcc, 0xaa, 0x30, 0x8b, 0x99, 0x18, 0x75, 0xa2, 0xc2, 0xb2, 0x77, 0xa8,
        0x0c, 0xfc, 0xe1, 0xc0, 0x15, 0x28, 0x0f, 0x92, 0x56, 0x3a, 0xb7, 0xda, 0x94, 0xb1, 0x2b,
        0xb0, 0xba, 0x0d, 0x2a, 0xf0, 0x6c, 0x71, 0x34, 0x1b, 0xf4, 0x8a, 0x0b, 0x0b, 0x24, 0x19,
        0x43, 0xa4, 0x25, 0xce, 0x45, 0xa2, 0xf0, 0xab, 0xd0, 0xae, 0xbf, 0x57, 0x08, 0xf6, 0xe9,
        0x18, 0x2a, 0xd0, 0x88, 0x7a, 0x09, 0x0a, 0xe9, 0x19, 0x32, 0x2f, 0x80, 0x7e, 0x1e, 0x04,
    ];

    /// The property that makes `ferrox quantize --type q6_k`'s output a
    /// file llama.cpp would have written, rather than one that merely
    /// decodes to similar numbers.
    #[test]
    fn q6_k_matches_llama_cpp_quantize_row_q6_k_ref() {
        let x = k_quant_fixture();
        let mut got = Vec::new();
        encode_row_q6_k(&x, &mut got).unwrap();
        assert_eq!(got.len(), LLAMA_CPP_Q6_K_GOLDEN.len());
        for (b, (g, w)) in got
            .as_chunks::<Q6_K_BLOCK_BYTES>()
            .0
            .iter()
            .zip(LLAMA_CPP_Q6_K_GOLDEN.as_chunks::<Q6_K_BLOCK_BYTES>().0)
            .enumerate()
        {
            assert_eq!(g, w, "super-block {b} disagrees with llama.cpp");
        }
    }

    /// A row that is not a whole number of super-blocks is refused, not
    /// padded, and not quietly written as the Q8_0 llama.cpp would fall
    /// back to. Padding would shift every following row on decode; a
    /// silent type change would write bytes the caller sized for Q6_K.
    #[test]
    fn a_row_that_is_not_a_whole_number_of_super_blocks_is_refused() {
        let mut out = Vec::new();
        assert!(encode_row_q6_k(&[0.5; Q6_K_BLOCK_ELEMS + 1], &mut out).is_none());
        assert!(encode_row_q6_k(&[0.5; 16], &mut out).is_none());
        assert!(encode_row_q6_k(&[], &mut out).is_some());
    }

    /// An all-zero super-block is 210 zero bytes, which is what
    /// llama.cpp's `memset` writes. A scale of 1.0 with zero codes
    /// dequantizes identically, so only a byte comparison catches it --
    /// which is why this is its own test and not a corollary of a value
    /// check.
    #[test]
    fn an_all_zero_super_block_is_all_zero_bytes_the_way_llama_cpp_writes_it() {
        let mut out = Vec::new();
        encode_row_q6_k(&[0.0; Q6_K_BLOCK_ELEMS], &mut out).unwrap();
        assert_eq!(out, vec![0u8; Q6_K_BLOCK_BYTES]);
    }

    /// Round trip through this crate's own reader, against an exact
    /// property rather than a tolerance: for every element, **no
    /// representable level is strictly closer** than the one the
    /// encoder chose.
    ///
    /// A tolerance would have to be invented, and an invented tolerance
    /// is what this whole issue exists to avoid. This is a fact
    /// instead: the recode rounds to the nearest of the 64 levels
    /// `d * sc * (k - 32)`, so a code split across the wrong `ql`/`qh`
    /// pair, a group scale read with the wrong sign, or an off-by-one
    /// in the 16-element stride all move some element off its nearest
    /// level and turn this red.
    ///
    /// (A group whose 8-bit scale rounded to zero has all 64 levels
    /// equal, so it passes trivially. The near-zero sub-blocks in the
    /// fixture are that case, on purpose.)
    #[test]
    fn every_element_lands_on_its_nearest_representable_level() {
        let x = k_quant_fixture();
        let mut bytes = Vec::new();
        encode_row_q6_k(&x, &mut bytes).unwrap();
        let back = dequant_q6_k(&bytes).unwrap();
        assert_eq!(back.len(), x.len());

        for (b, block) in bytes.as_chunks::<Q6_K_BLOCK_BYTES>().0.iter().enumerate() {
            let d = f16::from_le_bytes([block[208], block[209]]).to_f32();
            for g in 0..GROUPS {
                let sc = block[192 + g] as i8;
                let dg = d * sc as f32;
                for ii in 0..GROUP_ELEMS {
                    let idx = b * Q6_K_BLOCK_ELEMS + GROUP_ELEMS * g + ii;
                    let chosen = (x[idx] - back[idx]).abs();
                    for k in -32..=31i32 {
                        let level = dg * k as f32;
                        assert!(
                            (x[idx] - level).abs() >= chosen,
                            "block {b} group {g} element {ii}: {} is closer to {} than to the \
                             chosen {}",
                            x[idx],
                            level,
                            back[idx]
                        );
                    }
                }
            }
        }
    }

    /// The group scales really are signed. An encoder that stored them
    /// as `u8` would round-trip every group whose scale is positive and
    /// silently negate the rest, and on a symmetric weight
    /// distribution the error bound would not notice.
    #[test]
    fn the_group_scales_are_stored_signed() {
        let x = k_quant_fixture();
        let mut bytes = Vec::new();
        encode_row_q6_k(&x, &mut bytes).unwrap();
        let any_negative = bytes
            .as_chunks::<Q6_K_BLOCK_BYTES>()
            .0
            .iter()
            .flat_map(|b| b[192..208].iter())
            .any(|&v| (v as i8) < 0);
        assert!(
            any_negative,
            "no group scale is negative; this fixture no longer exercises the signed path"
        );
    }
}
