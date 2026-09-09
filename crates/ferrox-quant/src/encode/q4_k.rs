//! The Q4_K weight encoder: a transcription of llama.cpp b7650's
//! `quantize_row_q4_K_ref` (`ggml/src/ggml-quants.c:1280`), not a
//! reimplementation of it.
//!
//! A K-quant is NOT min/max over a block. Q4_K's 256-element
//! super-block is fitted in three stages, all of which live in
//! [`super::fit`] because Q5_K's fit is the same three stages with four
//! numbers changed:
//!
//! 1. Each of the 8 sub-blocks of 32 gets an **iterative** affine fit
//!    (`make_qkx2_quants`): 21 candidate inverse scales are tried, each
//!    one re-solves a weighted least-squares for (scale, min) from the
//!    integer codes it produced, and the lowest weighted squared error
//!    wins. The weights are `sqrt(mean(x^2)) + |x|`, so a sub-block's
//!    large values pull the fit toward themselves.
//! 2. The 8 scales and 8 mins are themselves quantized to 6 bits
//!    against the super-block's `d`/`dmin` and packed into 12 bytes.
//! 3. The 4-bit codes are then recomputed **against the 6-bit-rounded**
//!    scale and min, not against the fit from stage 1 -- so stage 3
//!    sees a slightly different affine map than stage 1 did.
//!
//! A naive min/max encoder skips all three and produces a file that
//! loads and generates measurably worse text. That is the failure this
//! module exists to not ship, so the arithmetic in [`super::fit`] is
//! deliberately the same shape as the C, down to the operation order in
//! the least-squares accumulation.
//!
//! What is left HERE is only what is Q4_K's own: the candidate grid it
//! passes to the shared fit, and the nibble packing.

use super::fit::{fit_qk_super_block, make_qkx2_quants, QkFit, QK_SUB_ELEMS};
use crate::{Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS};

/// Q4_K's half of the shared super-block fit: 4-bit codes, and the
/// `(-1.0, 0.1, 20)` candidate grid from `ggml-quants.c:1301`.
const Q4_K_FIT: QkFit = QkFit {
    nmax: 15,
    rmin: -1.0,
    rdelta: 0.1,
    nstep: 20,
};

/// Runs one 32-element sub-block through exactly the path
/// [`encode_block_q4_k`] uses and returns its `(scale, min)`.
///
/// Tooling, not a code path: it exists so a single sub-block can be
/// compared against llama.cpp's own `make_qkx2_quants` on the same
/// input. Chasing a floating-point difference that affects 0.55% of
/// super-blocks by quantizing whole checkpoints is far too coarse a
/// loop, and `examples/q4k_probe.rs` is the other half of it.
#[doc(hidden)]
pub fn probe_sub_block(xs: &[f32]) -> (f32, f32) {
    assert_eq!(xs.len(), QK_SUB_ELEMS);
    let mut l = [0u8; QK_SUB_ELEMS];
    let mut laux = [0u8; QK_SUB_ELEMS];
    let mut weights = [0f32; QK_SUB_ELEMS];
    super::fit::qk_sub_block_weights(xs, &mut weights);
    make_qkx2_quants(
        xs,
        &weights,
        &mut l,
        &mut laux,
        Q4_K_FIT.nmax,
        Q4_K_FIT.rmin,
        Q4_K_FIT.rdelta,
        Q4_K_FIT.nstep,
        false,
    )
}

/// Encodes one Q4_K super-block (exactly [`Q4_K_BLOCK_ELEMS`] values)
/// and appends its [`Q4_K_BLOCK_BYTES`] bytes to `out`.
pub fn encode_block_q4_k(block: &[f32; Q4_K_BLOCK_ELEMS], out: &mut Vec<u8>) {
    let fitted = fit_qk_super_block(block, Q4_K_FIT);

    out.reserve(Q4_K_BLOCK_BYTES);
    out.extend_from_slice(&fitted.d.to_le_bytes());
    out.extend_from_slice(&fitted.dmin.to_le_bytes());
    out.extend_from_slice(&fitted.packed);
    // Two 32-element halves of every 64 elements share a byte: the low
    // nibble is the first half, the high nibble the second. The reader
    // in `dequant_q4_k` walks the same pairing.
    for j in (0..Q4_K_BLOCK_ELEMS).step_by(64) {
        for i in 0..32 {
            out.push(fitted.l[j + i] | (fitted.l[j + i + 32] << 4));
        }
    }
}

/// Encodes a whole row (or any slice whose length is a multiple of
/// [`Q4_K_BLOCK_ELEMS`]) into Q4_K super-blocks, appending to `out`.
///
/// Returns `None` when `src.len()` is not a multiple of the super-block
/// size. llama.cpp handles that case by silently *changing type* --
/// `convert_incompatible_tensor` rewrites a Q4_K tensor with an awkward
/// row length to Q5_0, and to F16 if that does not fit either -- and
/// ferrox has neither encoder, so this refuses instead of padding.
/// Padding would write more elements than the tensor's shape declares
/// and every following row would decode shifted.
pub fn encode_row_q4_k(src: &[f32], out: &mut Vec<u8>) -> Option<()> {
    let (blocks, rest) = src.as_chunks::<Q4_K_BLOCK_ELEMS>();
    if !rest.is_empty() {
        return None;
    }
    out.reserve(blocks.len() * Q4_K_BLOCK_BYTES);
    for block in blocks {
        encode_block_q4_k(block, out);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::fit::QK_SUBS;
    use crate::encode::testdata::k_quant_fixture;
    use crate::{dequant_q4_k, q4_k_scale_min, Q4_K_SCALE_BYTES};
    use half::f16;

    /// llama.cpp's own bytes for [`k_quant_fixture`].
    ///
    /// Produced by the C harness described in the PR body: it links
    /// llama.cpp b7650's `libggml-base` and calls the exported
    /// `quantize_row_q4_K_ref` on the f32s
    /// `encode::testdata::dump_the_fixture_the_c_harness_reads` writes.
    /// The same harness also calls `ggml_quantize_chunk(GGML_TYPE_Q4_K,
    /// ...)` --
    /// the entry point `llama-quantize` itself goes through -- and
    /// asserts the two agree, so this is what the real tool writes and
    /// not merely what a reference function does.
    const LLAMA_CPP_Q4_K_GOLDEN: [u8; 4 * Q4_K_BLOCK_BYTES] = [
        0x32, 0x10, 0x14, 0x1c, 0x05, 0x0c, 0x59, 0xff, 0x04, 0x0b, 0x58, 0xff, 0x55, 0xcc, 0x8a,
        0xb3, 0xed, 0xc8, 0xba, 0x4e, 0xeb, 0x91, 0x85, 0xa6, 0x9c, 0x87, 0xd8, 0xab, 0x42, 0xe9,
        0x87, 0x0b, 0xb3, 0x82, 0x59, 0xb2, 0xc0, 0x80, 0x87, 0xa7, 0x98, 0x62, 0x75, 0x94, 0x31,
        0x0a, 0x89, 0xda, 0xc5, 0x32, 0xd4, 0xfa, 0xf6, 0xd6, 0xc1, 0xbd, 0xf1, 0xc8, 0x6c, 0xbf,
        0xc4, 0xa0, 0xeb, 0x46, 0x7d, 0xb0, 0xf4, 0xb7, 0x95, 0xbc, 0xd1, 0xe6, 0x84, 0x8d, 0x77,
        0x1d, 0x01, 0xd7, 0x1f, 0xda, 0x1b, 0xf6, 0x4f, 0x62, 0x3f, 0xce, 0x28, 0x47, 0x5b, 0xba,
        0xeb, 0xfc, 0x04, 0xb3, 0xba, 0x44, 0x94, 0xe0, 0xd6, 0xbf, 0x7e, 0x02, 0xf0, 0xac, 0x4c,
        0xda, 0xbf, 0x21, 0x4d, 0xc7, 0xd1, 0xb0, 0x6b, 0xf0, 0xb2, 0x0a, 0x8d, 0x25, 0xbc, 0x2c,
        0xda, 0xd7, 0xa9, 0x51, 0x32, 0xa2, 0xc0, 0x5e, 0x1c, 0x86, 0x95, 0x53, 0x40, 0x7d, 0xf1,
        0xf5, 0x34, 0xf8, 0x9c, 0xf0, 0x9f, 0xa8, 0x4c, 0x48, 0x2d, 0x10, 0xeb, 0x1b, 0x00, 0x10,
        0x48, 0xc6, 0x00, 0x00, 0x40, 0xcf, 0x45, 0xcc, 0xaa, 0xff, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
        0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
        0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xe6, 0xba, 0x13,
        0x8b, 0x45, 0x3b, 0x24, 0x4e, 0x69, 0xbb, 0x96, 0xd7, 0xc9, 0xe9, 0x13, 0xad, 0x75, 0xeb,
        0xeb, 0x8a, 0x0e, 0x9e, 0x76, 0xfb, 0x0d, 0x17, 0xfe, 0x5a, 0x07, 0x7e, 0x6d, 0x95, 0xa5,
        0x30, 0x33, 0xe7, 0xf1, 0x3d, 0x29, 0xfa, 0x07, 0x84, 0x99, 0xea, 0xca, 0x06, 0xe3, 0x27,
        0xa5, 0x40, 0x07, 0x12, 0xf3, 0x55, 0x72, 0xe5, 0xa1, 0x12, 0x35, 0xee, 0x06, 0xf2, 0x51,
        0x0d, 0x11, 0x70, 0x92, 0xc1, 0xd3, 0x7c, 0x99, 0x33, 0x74, 0x49, 0x6e, 0x6d, 0x2a, 0xdc,
        0xa2, 0x57, 0x35, 0xbe, 0xd3, 0x65, 0xbb, 0xf1, 0x14, 0x05, 0x09, 0xd4, 0x87, 0x3e, 0xbf,
        0x4c, 0xc8, 0xd1, 0x30, 0x10, 0xdc, 0x1b, 0x05, 0x00, 0x58, 0xff, 0x05, 0x00, 0x58, 0xff,
        0x55, 0xdd, 0x89, 0xce, 0x44, 0xa8, 0xc2, 0x9c, 0xec, 0x84, 0x2c, 0x8f, 0x6f, 0x30, 0x74,
        0xae, 0x66, 0x2b, 0x16, 0x5b, 0xe6, 0xe0, 0x96, 0x69, 0x66, 0x8f, 0xe4, 0x5c, 0x57, 0x24,
        0x06, 0x52, 0x67, 0xa1, 0xa0, 0x43, 0xaa, 0x5d, 0x43, 0x4d, 0x7c, 0xbf, 0x78, 0x16, 0x59,
        0xf9, 0x30, 0x58, 0x03, 0x12, 0x73, 0xed, 0x8d, 0x00, 0xdd, 0x49, 0xe2, 0xf9, 0xa1, 0x88,
        0x2c, 0x80, 0x90, 0x0f, 0xb3, 0x2b, 0xf5, 0xc9, 0x72, 0x61, 0x6a, 0x85, 0x99, 0xc3, 0x02,
        0xd4, 0xd8, 0x2a, 0xee, 0x20, 0xa9, 0xcd, 0x9a, 0xa8, 0xed, 0x6b, 0x95, 0x98, 0x8c, 0x96,
        0x6f, 0x1f, 0xda, 0x13, 0xf9, 0xc7, 0x75, 0xdd, 0x55, 0x17, 0x71, 0xd4, 0xbd, 0xc5, 0x79,
        0xa0, 0x2d, 0xcb, 0x7b, 0x40, 0x76, 0x1b, 0xf4, 0x04, 0x56, 0xd2, 0x1b, 0x24, 0x44, 0x25,
        0x68, 0x01, 0x33, 0xa1, 0x92, 0xf5, 0x1f, 0x69, 0xed, 0xd1, 0xa8, 0x28, 0x3c, 0x10, 0x2d,
        0x1c, 0x05, 0x0c, 0x58, 0xff, 0x05, 0x0c, 0x53, 0xff, 0x55, 0xcc, 0x88, 0x9f, 0xba, 0xf2,
        0xc5, 0x51, 0xfe, 0x43, 0xec, 0x47, 0x96, 0x64, 0x14, 0x78, 0xf3, 0x6b, 0x46, 0x52, 0x79,
        0x15, 0x26, 0x05, 0x50, 0x9f, 0xdd, 0xec, 0x0b, 0x0d, 0x5a, 0x8f, 0xe1, 0x15, 0x76, 0x87,
        0x1c, 0x6a, 0xf7, 0xe1, 0xe2, 0x46, 0xc4, 0xcc, 0x90, 0x95, 0x40, 0x67, 0xdb, 0x70, 0x53,
        0xd4, 0x70, 0xb4, 0xcd, 0x80, 0x52, 0xb4, 0x0b, 0xc1, 0xd5, 0xda, 0x17, 0x15, 0x1e, 0x99,
        0x57, 0x22, 0x9c, 0x58, 0xc3, 0xc4, 0x5e, 0xd2, 0x78, 0x37, 0x69, 0xe2, 0x21, 0xf7, 0x83,
        0x5c, 0xa1, 0x6a, 0xbd, 0xbe, 0x72, 0xa4, 0x3d, 0x61, 0x76, 0xcb, 0x55, 0x2a, 0x01, 0x8d,
        0x14, 0xcb, 0xdc, 0x4f, 0x6f, 0x15, 0x46, 0x6e, 0xe8, 0x5d, 0x6d, 0xf0, 0xea, 0x0d, 0xaa,
        0x8f, 0xdd, 0xd7, 0x3e, 0x52, 0x20, 0x24, 0x1b, 0x15, 0x62, 0x98, 0x0a, 0xf4, 0x42, 0x9b,
        0xdc, 0x8a, 0xb7, 0xab, 0xae, 0x57,
    ];

    /// The property that makes `ferrox quantize --type q4_k_s --pure`'s
    /// output a file llama.cpp would have written, rather than one that
    /// merely decodes to similar numbers.
    ///
    /// An encoder that is within Q4_K's error bound passes any
    /// tolerance test and still writes a different file. Only this
    /// catches that.
    #[test]
    fn q4_k_matches_llama_cpp_quantize_row_q4_k_ref() {
        let x = k_quant_fixture();
        let mut got = Vec::new();
        encode_row_q4_k(&x, &mut got).unwrap();
        assert_eq!(got.len(), LLAMA_CPP_Q4_K_GOLDEN.len());
        for (b, (g, w)) in got
            .as_chunks::<Q4_K_BLOCK_BYTES>()
            .0
            .iter()
            .zip(LLAMA_CPP_Q4_K_GOLDEN.as_chunks::<Q4_K_BLOCK_BYTES>().0)
            .enumerate()
        {
            assert_eq!(g, w, "super-block {b} disagrees with llama.cpp");
        }
    }

    /// A row that is not a whole number of super-blocks is refused, not
    /// padded. llama.cpp answers this case by changing the tensor's
    /// TYPE (Q4_K -> Q5_0 -> F16); ferrox has neither encoder, and
    /// padding would shift every following row on decode.
    #[test]
    fn a_row_that_is_not_a_whole_number_of_super_blocks_is_refused() {
        let mut out = Vec::new();
        assert!(encode_row_q4_k(&[0.5; Q4_K_BLOCK_ELEMS + 1], &mut out).is_none());
        // 32 is a Q8_0 block and a Q4_K sub-block, and still not a
        // Q4_K row: the block size that matters here is 256.
        assert!(encode_row_q4_k(&[0.5; 32], &mut out).is_none());
        assert!(encode_row_q4_k(&[], &mut out).is_some());
    }

    /// Round trip through this crate's own reader, against an exact
    /// property rather than a tolerance: for every element, **no
    /// representable level is strictly closer** than the one the
    /// encoder chose.
    ///
    /// A tolerance would have to be invented, and an invented tolerance
    /// is what this whole issue exists to avoid. This is a fact instead:
    /// stage 3 rounds to the nearest of the 16 levels `d*sc*k -
    /// dmin*m`, so a nibble packed into the wrong half-byte, a scale
    /// unpacked from the wrong bits, or an off-by-one in the sub-block
    /// stride all move some element off its nearest level and turn this
    /// red. It says nothing about whether the *fit* is good -- that is
    /// what the golden above is for, and this is the weak half.
    ///
    /// (A sub-block whose 6-bit scale rounded to zero has all 16 levels
    /// equal, so it passes trivially. Sub-block 17 is that case, on
    /// purpose.)
    #[test]
    fn every_element_lands_on_its_nearest_representable_level() {
        let x = k_quant_fixture();
        let mut bytes = Vec::new();
        encode_row_q4_k(&x, &mut bytes).unwrap();
        let back = dequant_q4_k(&bytes).unwrap();
        assert_eq!(back.len(), x.len());

        for (b, block) in bytes.as_chunks::<Q4_K_BLOCK_BYTES>().0.iter().enumerate() {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
            let packed: [u8; Q4_K_SCALE_BYTES] = block[4..16].try_into().unwrap();
            for j in 0..QK_SUBS {
                let (sc, m) = q4_k_scale_min(j, &packed);
                let (dj, dm) = (d * sc as f32, dmin * m as f32);
                for ii in 0..QK_SUB_ELEMS {
                    let idx = b * Q4_K_BLOCK_ELEMS + QK_SUB_ELEMS * j + ii;
                    let chosen = (x[idx] - back[idx]).abs();
                    for k in 0..=15u8 {
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
}
