//! The Q5_K weight encoder: a transcription of llama.cpp b7650's
//! `quantize_row_q5_K_ref` (`ggml/src/ggml-quants.c:1467`).
//!
//! Q5_K is Q4_K's super-block fit with two things changed and nothing
//! else:
//!
//! * the codes run `0..=31` instead of `0..=15`, and the candidate
//!   grid handed to `make_qkx2_quants` is `(-0.5, 0.1, 15)` rather than
//!   `(-1.0, 0.1, 20)` -- both of which are [`Q5_K_FIT`], passed to the
//!   SAME [`super::fit::fit_qk_super_block`] Q4_K uses;
//! * the fifth bit of each code goes into a separate 32-byte `qh`
//!   plane, which is the only packing this module writes out itself.
//!
//! Two transcriptions of that shared fit is precisely the shape this
//! repo has paid the most for: the copies agree the day they are
//! written and drift the first time one of them is corrected. So the
//! difference between Q4_K and Q5_K is four numbers in a struct, and if
//! someone fixes the least-squares step it is fixed for both or for
//! neither.
//!
//! The `qh` bit assignment is worth stating because it is not the
//! obvious one: the 256 codes are walked in four groups of 64, and
//! within a group the first 32 codes' high bits go to bit `m1` of
//! `qh[j]` and the second 32's to bit `m2`, where `m1`/`m2` start at 1
//! and 2 and shift LEFT BY TWO per group. So `qh[j]` holds the high
//! bits of elements `j`, `j+32`, `j+64`, ... in bit pairs, not one
//! contiguous run. `dequant_q5_k` in this crate reads it back with the
//! same `u1 <<= 2` walk.

use super::fit::{fit_qk_super_block, QkFit};
use crate::{Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS};

/// Q5_K's half of the shared super-block fit: 5-bit codes, and the
/// `(-0.5, 0.1, 15)` candidate grid from `ggml-quants.c:1488`.
const Q5_K_FIT: QkFit = QkFit {
    nmax: 31,
    rmin: -0.5,
    rdelta: 0.1,
    nstep: 15,
};

/// Bytes of `qh` (one bit per element) in a Q5_K super-block.
const QH_BYTES: usize = Q5_K_BLOCK_ELEMS / 8;

/// Encodes one Q5_K super-block (exactly [`Q5_K_BLOCK_ELEMS`] values)
/// and appends its [`Q5_K_BLOCK_BYTES`] bytes to `out`.
pub fn encode_block_q5_k(block: &[f32; Q5_K_BLOCK_ELEMS], out: &mut Vec<u8>) {
    let fitted = fit_qk_super_block(block, Q5_K_FIT);

    let mut qh = [0u8; QH_BYTES];
    let mut ql = [0u8; Q5_K_BLOCK_ELEMS / 2];
    let (mut m1, mut m2) = (1u8, 2u8);
    for (g, n) in (0..Q5_K_BLOCK_ELEMS).step_by(64).enumerate() {
        for j in 0..32 {
            // `l1 -= 16` where upstream tests `> 15`: the fifth bit is
            // stripped into `qh` and the low four stay in `ql`. Writing
            // `l1 & 0xF` instead would be the same for a code in
            // `0..=31` and would silently keep a code above 31 -- which
            // `fit_qk_super_block` cannot produce, but only because it
            // clamps to `nmax`. Keeping the C's shape means the two
            // facts stay tied together.
            let mut l1 = fitted.l[n + j];
            if l1 > 15 {
                l1 -= 16;
                qh[j] |= m1;
            }
            let mut l2 = fitted.l[n + j + 32];
            if l2 > 15 {
                l2 -= 16;
                qh[j] |= m2;
            }
            ql[g * 32 + j] = l1 | (l2 << 4);
        }
        m1 <<= 2;
        m2 <<= 2;
    }

    out.reserve(Q5_K_BLOCK_BYTES);
    out.extend_from_slice(&fitted.d.to_le_bytes());
    out.extend_from_slice(&fitted.dmin.to_le_bytes());
    out.extend_from_slice(&fitted.packed);
    out.extend_from_slice(&qh);
    out.extend_from_slice(&ql);
}

/// Encodes a whole row (or any slice whose length is a multiple of
/// [`Q5_K_BLOCK_ELEMS`]) into Q5_K super-blocks, appending to `out`.
///
/// Returns `None` when `src.len()` is not a multiple of the super-block
/// size. llama.cpp answers that case by silently *changing type* --
/// `convert_incompatible_tensor` rewrites a Q5_K tensor with an awkward
/// row length to Q5_1, and to F16 if that does not fit either -- and
/// ferrox has neither encoder, so this refuses instead of padding.
/// Padding would write more elements than the tensor's shape declares
/// and every following row would decode shifted.
pub fn encode_row_q5_k(src: &[f32], out: &mut Vec<u8>) -> Option<()> {
    let (blocks, rest) = src.as_chunks::<Q5_K_BLOCK_ELEMS>();
    if !rest.is_empty() {
        return None;
    }
    out.reserve(blocks.len() * Q5_K_BLOCK_BYTES);
    for block in blocks {
        encode_block_q5_k(block, out);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::testdata::k_quant_fixture;
    use crate::{dequant_q5_k, q4_k_scale_min, Q4_K_SCALE_BYTES};
    use half::f16;

    /// llama.cpp's own bytes for [`k_quant_fixture`].
    ///
    /// Produced by the C harness described in the PR body: it links
    /// llama.cpp b7650's `libggml-base` and calls the exported
    /// `quantize_row_q5_K_ref` on the f32s
    /// `encode::testdata::dump_the_fixture_the_c_harness_reads` writes.
    /// The same harness also calls `ggml_quantize_chunk(GGML_TYPE_Q5_K,
    /// ...)` -- the entry point `llama-quantize` itself goes through --
    /// and asserts the two agree, so this is what the real tool writes
    /// and not merely what a reference function does.
    ///
    /// An encoder that is within Q5_K's error bound passes any
    /// tolerance test and still writes a different file. Only this
    /// catches that.
    const LLAMA_CPP_Q5_K_GOLDEN: [u8; 4 * Q5_K_BLOCK_BYTES] = [
        0x1f, 0x0c, 0x06, 0x1c, 0x04, 0x0c, 0x59, 0xff, 0x04, 0x0c, 0x58, 0xff, 0x55, 0xcc, 0x89,
        0xc3, 0x5b, 0xa3, 0x9b, 0x4d, 0xdb, 0x3a, 0xda, 0x4e, 0xdb, 0xbb, 0xf7, 0x3f, 0x08, 0xab,
        0xbf, 0x41, 0x66, 0xaa, 0xa9, 0x3a, 0x1a, 0x4e, 0xab, 0xbb, 0x1b, 0xfc, 0xf0, 0x86, 0xd0,
        0xe9, 0x67, 0x6f, 0xef, 0x94, 0x7a, 0x9f, 0xeb, 0x31, 0x2b, 0x4f, 0x4f, 0x11, 0xb5, 0x5b,
        0xa4, 0xe6, 0x22, 0x0c, 0x77, 0x25, 0xb8, 0x86, 0x91, 0x10, 0x12, 0x51, 0x43, 0xe4, 0xfc,
        0x4b, 0x72, 0x1a, 0x25, 0xc8, 0x7a, 0x65, 0xb9, 0xe4, 0xfd, 0xac, 0x72, 0x6b, 0xe2, 0x9f,
        0xc7, 0x6e, 0x87, 0x41, 0xc6, 0x8c, 0xeb, 0x70, 0xf8, 0x6d, 0x3a, 0x67, 0xa2, 0xcd, 0x07,
        0x1b, 0xfe, 0x1a, 0x02, 0xaf, 0x2f, 0xa4, 0x26, 0xfc, 0x8e, 0xc4, 0x6e, 0x7c, 0x50, 0x7e,
        0xa6, 0x74, 0xc7, 0xe9, 0x07, 0x75, 0x64, 0x79, 0x39, 0xb0, 0xac, 0x7e, 0xec, 0x04, 0xf0,
        0x47, 0x98, 0xb3, 0x5f, 0x41, 0x7a, 0x9f, 0xa2, 0x70, 0xc7, 0xf0, 0x65, 0x04, 0x0b, 0x5a,
        0x79, 0x49, 0xb5, 0xaf, 0x42, 0xa2, 0x74, 0x43, 0x80, 0x9e, 0x29, 0x1d, 0x2a, 0xa7, 0x90,
        0xdb, 0xe2, 0xe9, 0x58, 0xf1, 0x39, 0xf0, 0x3f, 0x51, 0x89, 0x82, 0x08, 0x0c, 0xcf, 0x1b,
        0x00, 0x10, 0x48, 0xc6, 0x00, 0x00, 0x40, 0xcf, 0x45, 0xcc, 0xaa, 0xff, 0x2a, 0x0e, 0x82,
        0xae, 0xa2, 0x56, 0xd2, 0x36, 0x06, 0x6e, 0x7a, 0x7a, 0x7e, 0xce, 0xa2, 0x0e, 0x22, 0xce,
        0x8e, 0x0e, 0xe6, 0x8e, 0x02, 0x2e, 0x66, 0x82, 0x8e, 0x76, 0xc2, 0x66, 0xc6, 0x9a, 0xf0,
        0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
        0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
        0xf0, 0xdb, 0x54, 0x17, 0x07, 0x8a, 0x67, 0x39, 0x7d, 0xc2, 0x76, 0x3d, 0xaf, 0x83, 0xc3,
        0x16, 0x4b, 0xea, 0xd6, 0xc6, 0x05, 0x0c, 0x2e, 0xec, 0xf6, 0x0c, 0x2f, 0xfe, 0xa4, 0x0e,
        0xde, 0xcb, 0x2a, 0x4a, 0x60, 0x67, 0xce, 0xf1, 0x6b, 0x33, 0xf4, 0x0f, 0x08, 0x12, 0xd4,
        0x85, 0x0d, 0xc7, 0x5f, 0x5a, 0x70, 0x0e, 0x34, 0xf7, 0xa9, 0xe3, 0xcb, 0x41, 0x34, 0x7a,
        0xcd, 0x0c, 0xf5, 0xa2, 0x0a, 0x21, 0xe0, 0x24, 0x91, 0xa5, 0xf8, 0x23, 0x66, 0xf7, 0x72,
        0xce, 0xcb, 0x45, 0xa9, 0x54, 0xbe, 0x7a, 0x7d, 0xc5, 0xca, 0x67, 0xf3, 0x17, 0x0a, 0x03,
        0xa8, 0x0f, 0x6d, 0x6f, 0x89, 0x90, 0xb2, 0x21, 0x0c, 0xe7, 0x1b, 0x05, 0x00, 0x58, 0xff,
        0x05, 0x00, 0x59, 0xff, 0x55, 0xdd, 0x88, 0xcc, 0x0c, 0x87, 0xd2, 0xa7, 0x77, 0xae, 0x45,
        0xe3, 0x75, 0x1c, 0x32, 0x47, 0xb0, 0x31, 0x30, 0xbd, 0x7e, 0x12, 0x2e, 0x35, 0x78, 0x2f,
        0x1a, 0x9d, 0xb4, 0x88, 0x78, 0x64, 0xc8, 0xb6, 0xca, 0x4c, 0x87, 0x60, 0xa5, 0x38, 0xe8,
        0x27, 0x58, 0x2f, 0xde, 0x70, 0x07, 0x6c, 0xcb, 0x47, 0x1d, 0xa5, 0xec, 0xf0, 0x4c, 0xd3,
        0xcd, 0x2e, 0xf8, 0xa8, 0xaf, 0x48, 0x0c, 0xa5, 0xce, 0x53, 0x51, 0x97, 0x55, 0xaa, 0x96,
        0x8b, 0xea, 0x7f, 0xe0, 0x1d, 0xb3, 0xe4, 0x51, 0xa1, 0x07, 0x35, 0xe7, 0xcb, 0x1b, 0x00,
        0xbb, 0x84, 0xd5, 0xe3, 0x43, 0x00, 0x59, 0x02, 0x31, 0x0f, 0x66, 0x47, 0xeb, 0x94, 0xf3,
        0xd2, 0xd5, 0x1b, 0x23, 0x87, 0x04, 0xb9, 0xa0, 0x44, 0xdc, 0x40, 0x41, 0x9b, 0x24, 0x41,
        0xda, 0xd6, 0x3a, 0x10, 0x19, 0x3d, 0xbe, 0x1e, 0xb5, 0x35, 0xe2, 0x9f, 0xfb, 0xba, 0xab,
        0x2e, 0xe2, 0xc9, 0x7c, 0x8a, 0xf4, 0x50, 0x5b, 0xa8, 0xf8, 0x90, 0xfe, 0x28, 0xf9, 0x09,
        0xad, 0xb4, 0x37, 0x58, 0x88, 0x4b, 0xd1, 0x03, 0x77, 0x42, 0x23, 0xfa, 0x2f, 0xd4, 0xdc,
        0xc2, 0x51, 0x40, 0x16, 0x0c, 0x06, 0x1c, 0x05, 0x0c, 0x57, 0xff, 0x05, 0x0c, 0x55, 0xff,
        0x55, 0xdc, 0x97, 0xef, 0x77, 0x16, 0x2e, 0x68, 0xdb, 0x60, 0x5b, 0x8c, 0xda, 0x68, 0xc0,
        0xe5, 0xee, 0xd1, 0x60, 0x18, 0x31, 0x38, 0x4c, 0x28, 0x10, 0xcb, 0x67, 0xbb, 0x09, 0xdd,
        0xc5, 0xf3, 0x86, 0xfc, 0xf4, 0x12, 0x65, 0xf4, 0x9b, 0x92, 0xfd, 0x86, 0xc8, 0x8f, 0x3d,
        0xc7, 0x27, 0xe1, 0xf5, 0xc6, 0x9c, 0xb4, 0xf1, 0x2a, 0x4b, 0x0b, 0xb0, 0x1e, 0xab, 0xc8,
        0x06, 0x0a, 0xa5, 0x0f, 0xd1, 0x1a, 0xdb, 0x0e, 0x1c, 0xc6, 0xf1, 0xc2, 0xd5, 0x8e, 0x89,
        0x9b, 0x21, 0x2b, 0x80, 0xb1, 0xb9, 0xe1, 0x97, 0xaa, 0xe1, 0x7a, 0x8c, 0x00, 0xa5, 0x6a,
        0x08, 0x83, 0xbc, 0xa7, 0x20, 0x2b, 0x1f, 0x33, 0xa0, 0x36, 0x39, 0xa0, 0x96, 0xa9, 0xcd,
        0xc5, 0xf0, 0x7e, 0xe3, 0xd4, 0x52, 0xfe, 0x16, 0xc8, 0x61, 0xd5, 0x7a, 0x7d, 0xe4, 0x59,
        0x7a, 0xd2, 0x0c, 0xa6, 0xcb, 0x63, 0x11, 0x2b, 0x37, 0x96, 0xc9, 0x9e, 0xef, 0x3a, 0x9c,
        0xed, 0xe1, 0xbc, 0xdd, 0xf0, 0xe6, 0x1b, 0x65, 0x0f, 0xcb, 0xc0, 0x7e, 0xa5, 0x51, 0x58,
        0x28, 0x3b, 0xe4, 0x42, 0x05, 0xf8, 0x94, 0x39, 0xbb, 0x15, 0x8f, 0x57, 0x6e, 0xcf,
    ];

    /// The property that makes `ferrox quantize --type q5_k_m`'s output
    /// a file llama.cpp would have written, rather than one that merely
    /// decodes to similar numbers.
    #[test]
    fn q5_k_matches_llama_cpp_quantize_row_q5_k_ref() {
        let x = k_quant_fixture();
        let mut got = Vec::new();
        encode_row_q5_k(&x, &mut got).unwrap();
        assert_eq!(got.len(), LLAMA_CPP_Q5_K_GOLDEN.len());
        for (b, (g, w)) in got
            .as_chunks::<Q5_K_BLOCK_BYTES>()
            .0
            .iter()
            .zip(LLAMA_CPP_Q5_K_GOLDEN.as_chunks::<Q5_K_BLOCK_BYTES>().0)
            .enumerate()
        {
            assert_eq!(g, w, "super-block {b} disagrees with llama.cpp");
        }
    }

    /// A row that is not a whole number of super-blocks is refused, not
    /// padded. llama.cpp answers this case by changing the tensor's
    /// TYPE (Q5_K -> Q5_1 -> F16); ferrox has neither encoder, and
    /// padding would shift every following row on decode.
    #[test]
    fn a_row_that_is_not_a_whole_number_of_super_blocks_is_refused() {
        let mut out = Vec::new();
        assert!(encode_row_q5_k(&[0.5; Q5_K_BLOCK_ELEMS + 1], &mut out).is_none());
        // 32 is a Q8_0 block and a Q5_K sub-block, and still not a
        // Q5_K row: the block size that matters here is 256.
        assert!(encode_row_q5_k(&[0.5; 32], &mut out).is_none());
        assert!(encode_row_q5_k(&[], &mut out).is_some());
    }

    /// Round trip through this crate's own reader, against an exact
    /// property rather than a tolerance: for every element, **no
    /// representable level is strictly closer** than the one the
    /// encoder chose.
    ///
    /// A tolerance would have to be invented, and an invented tolerance
    /// is what this whole issue exists to avoid. This is a fact
    /// instead: stage 3 rounds to the nearest of the 32 levels
    /// `d*sc*k - dmin*m`, so a fifth bit written into the wrong `qh`
    /// bit, a nibble packed into the wrong half-byte, or an off-by-one
    /// in the sub-block stride all move some element off its nearest
    /// level and turn this red. In particular it is the only test here
    /// that would catch `m1 <<= 1` in place of `m1 <<= 2`, which the
    /// golden also catches but which nothing else would explain.
    #[test]
    fn every_element_lands_on_its_nearest_representable_level() {
        let x = k_quant_fixture();
        let mut bytes = Vec::new();
        encode_row_q5_k(&x, &mut bytes).unwrap();
        let back = dequant_q5_k(&bytes).unwrap();
        assert_eq!(back.len(), x.len());

        for (b, block) in bytes.as_chunks::<Q5_K_BLOCK_BYTES>().0.iter().enumerate() {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let packed: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            for j in 0..8 {
                let (sc, m) = q4_k_scale_min(j, &packed);
                let (dj, dm) = (d * sc as f32, dmin * m as f32);
                for ii in 0..32 {
                    let idx = b * Q5_K_BLOCK_ELEMS + 32 * j + ii;
                    let chosen = (x[idx] - back[idx]).abs();
                    for k in 0..=31u8 {
                        let level = dj * k as f32 - dm;
                        assert!(
                            (x[idx] - level).abs() >= chosen,
                            "block {b} sub-block {j} element {ii}: {} is closer to {} than to the \
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

    /// The fifth bit is actually used. A Q5_K encoder that packed only
    /// the low nibble would still round-trip within Q4_K's error and
    /// would pass any tolerance test; what it would NOT do is set a
    /// single bit in `qh`.
    #[test]
    fn the_high_bit_plane_is_not_all_zero() {
        let x = k_quant_fixture();
        let mut bytes = Vec::new();
        encode_row_q5_k(&x, &mut bytes).unwrap();
        for (b, block) in bytes.as_chunks::<Q5_K_BLOCK_BYTES>().0.iter().enumerate() {
            let qh = &block[16..16 + QH_BYTES];
            // Super-block 2 of the fixture is the all-zero / constant
            // one, whose codes are legitimately all low.
            if b == 2 {
                continue;
            }
            assert!(
                qh.iter().any(|&v| v != 0),
                "super-block {b} has an empty qh"
            );
        }
    }
}
