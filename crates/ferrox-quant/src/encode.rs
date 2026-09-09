//! Weight *encoders*: f32 in, GGUF block bytes out.
//!
//! The rest of this crate reads quantized blocks. This module is the
//! only place that writes them, and today it writes four formats: Q8_0
//! here, and Q4_K, Q5_K and Q6_K in [`q4_k`], [`q5_k`] and [`q6_k`].
//! The rest is not an oversight, it is the scope: llama.cpp's remaining
//! K-quant and IQ encoders each need their own transcription (and, for
//! the IQ tiers, a lattice search over a codebook), and a naive min/max
//! encoder wearing a K-quant's name produces a file that loads and
//! generates measurably worse text. `ferrox quantize` refuses every
//! target this module cannot encode, by name.
//!
//! The three K-quants share ONE transcription of the per-sub-block fit,
//! in [`fit`]. Q4_K and Q5_K differ by four numbers in a `QkFit`, not by
//! a second copy of `make_qkx2_quants`; Q6_K reaches the same module for
//! `nearest_int` and `make_qx_quants`. Two copies of a fit that must
//! agree is this repo's dominant bug shape, and a K-quant encoder is
//! about the worst place to have one: the copies would agree the day
//! they were written and diverge invisibly, since both would still
//! dequantize to plausible weights.
//!
//! Each format lands with a **byte-identical** golden against
//! llama.cpp's own encoder, never a tolerance: two encoders can agree
//! on dequantized values and still write different files.
//!
//! **Q8_0 here is byte-for-byte llama.cpp's `quantize_row_q8_0_ref`**,
//! not merely "close enough". The arithmetic below is deliberately the
//! same shape as the C, including the reciprocal multiply and the
//! `a > b ? a : b` maximum, because the file this writes is meant to be
//! indistinguishable from `llama-quantize --type Q8_0`'s. See
//! `q8_0_matches_llama_cpp_quantize_row_q8_0_ref` for the golden.

pub mod fit;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
#[cfg(test)]
mod testdata;

use half::f16;

use crate::{Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};

/// Encodes one Q8_0 block (exactly [`Q8_0_BLOCK_ELEMS`] values) and
/// appends its [`Q8_0_BLOCK_BYTES`] bytes to `out`.
///
/// Every arithmetic choice here mirrors `ggml-quants.c`:
///
/// * `amax` is folded with `a > b ? a : b`, not Rust's `f32::max`.
///   They differ on NaN -- `f32::max` returns the non-NaN operand,
///   ggml's macro propagates it -- and a checkpoint with a NaN weight
///   should produce llama.cpp's bytes, not politely different ones.
/// * The scale is applied as a multiply by `1/d`, not a divide by `d`.
///   `v * (1.0/d)` and `v / d` differ by an ulp for many inputs, and an
///   ulp either side of `.5` is a different `roundf` result, so a
///   divide here would disagree with llama.cpp on real weights.
/// * `d == 0` (an all-zero block) yields the reciprocal `0.0`, so the
///   stored scale is `+0.0` and every quant is 0. The obvious
///   alternative, storing a scale of 1.0, dequantizes identically and
///   is therefore invisible to every test that checks values -- and
///   produces a file that differs from llama.cpp's in bytes.
#[inline]
pub fn encode_block_q8_0(block: &[f32; Q8_0_BLOCK_ELEMS], out: &mut Vec<u8>) {
    let mut amax = 0f32;
    for &v in block.iter() {
        let av = v.abs();
        // Deliberately not `amax.max(av)`: see the doc comment.
        amax = if amax > av { amax } else { av };
    }
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    for &v in block.iter() {
        // `as i8` saturates in Rust where C's float->int8 conversion is
        // undefined out of range; |v * id| <= 127 + an ulp for every
        // finite input, so the two agree wherever the C is defined, and
        // this one has no UB where it is not.
        out.push(((v * id).round() as i8) as u8);
    }
}

/// Encodes a whole row (or any slice whose length is a multiple of
/// [`Q8_0_BLOCK_ELEMS`]) into Q8_0 blocks, appending to `out`.
///
/// Returns `None` when `src.len()` is not a multiple of the block size.
/// llama.cpp `assert`s the same condition and its Q8_0 path has no
/// fallback type, so a row that cannot be tiled is a refusal, never a
/// zero-padded block: padding changes the row length the reader
/// computes from the shape, and the file would decode shifted.
pub fn encode_row_q8_0(src: &[f32], out: &mut Vec<u8>) -> Option<()> {
    let (blocks, rest) = src.as_chunks::<Q8_0_BLOCK_ELEMS>();
    if !rest.is_empty() {
        return None;
    }
    out.reserve(blocks.len() * Q8_0_BLOCK_BYTES);
    for block in blocks {
        encode_block_q8_0(block, out);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant_q8_0;

    /// Deterministic pseudo-random f32s in roughly the range real
    /// weights occupy, from a 32-bit xorshift so the C harness that
    /// produced the golden below can generate the identical input.
    fn sample_input(n: usize) -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                // [-1, 1), 24 bits of mantissa.
                ((state >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    /// The golden: bytes produced by llama.cpp's own encoder for
    /// `sample_input(64)`.
    ///
    /// Generated by linking `.scratch/llama.cpp/build/bin/libggml-base`
    /// and calling the exported `quantize_row_q8_0_ref` on the same
    /// input this test builds; the same C harness also calls
    /// `ggml_quantize_chunk(GGML_TYPE_Q8_0, ...)` -- the entry point
    /// `llama-quantize` itself goes through -- and asserts the two
    /// agree, so this golden is what the real tool writes and not just
    /// what a reference function does.
    ///
    /// An encoder that is merely *within Q8_0's error bound* passes a
    /// tolerance test and still writes a different file; this is what
    /// catches that.
    const LLAMA_CPP_Q8_0_GOLDEN: [u8; 2 * Q8_0_BLOCK_BYTES] = [
        0xdc, 0x1f, 0x08, 0x93, 0xc7, 0x02, 0xf0, 0xa8, 0x0a, 0x46, 0x55, 0xb9, 0xe9, 0xcd, 0x7f,
        0xb4, 0x0d, 0x79, 0x4f, 0x71, 0x6a, 0xc0, 0xac, 0x6d, 0xa8, 0x51, 0x7a, 0x77, 0x2d, 0x42,
        0x6c, 0xcc, 0x8e, 0x7a, 0xe9, 0x1f, 0x42, 0x4d, 0x18, 0x19, 0x05, 0x50, 0x66, 0xfa, 0xe8,
        0x59, 0xf7, 0xc4, 0xac, 0x9c, 0xb4, 0xa8, 0xe9, 0x93, 0x17, 0x3f, 0xad, 0xef, 0x06, 0x4a,
        0xf8, 0x3f, 0xa3, 0xea, 0x7f, 0x30, 0x3e, 0x8d,
    ];

    /// The property that makes `ferrox quantize`'s output a file
    /// llama.cpp would have written, rather than one that merely
    /// decodes to similar numbers.
    #[test]
    fn q8_0_matches_llama_cpp_quantize_row_q8_0_ref() {
        let x = sample_input(2 * Q8_0_BLOCK_ELEMS);
        let mut got = Vec::new();
        encode_row_q8_0(&x, &mut got).unwrap();
        assert_eq!(
            got.as_slice(),
            &LLAMA_CPP_Q8_0_GOLDEN[..],
            "ferrox's Q8_0 encoder disagrees with llama.cpp's"
        );
    }

    /// One block, as f16 bit patterns, on which `v * (1/d)` and `v / d`
    /// round to DIFFERENT int8s -- exactly one of its 32 quants, 63
    /// against 64.
    ///
    /// It exists because the obvious golden does not catch the
    /// difference: over uniform f32 noise the two spellings agree for
    /// at least 8192 consecutive values. Over real F16 weights they
    /// disagree constantly -- f16's 11-bit mantissa lands on the `.5`
    /// boundary far more often than f32's 24-bit one -- and quantizing
    /// a 135M F16 checkpoint with the divide spelling gave a file whose
    /// every one of 211 quantized tensors differed from
    /// `llama-quantize`'s (9877 of token_embd's 30 MB, for instance).
    ///
    /// So this block is the small, checked-in stand-in for that whole
    /// experiment. Found by scanning the same xorshift stream rounded
    /// through f16 and scaled to where weights actually live.
    const TIE_BLOCK_F16_BITS: [u16; Q8_0_BLOCK_ELEMS] = [
        0xadf3, 0x247e, 0x28d0, 0x221a, 0xb017, 0x98ed, 0xa97d, 0x3010, 0xac34, 0x0c3e, 0x2cb8,
        0x2bf9, 0xa7e5, 0xb0b8, 0x3030, 0xb00a, 0xac33, 0xac39, 0xac46, 0x2b78, 0x3007, 0xa5fe,
        0x2feb, 0x30b2, 0x3033, 0xad15, 0xb046, 0x2cd7, 0xaff4, 0xaca6, 0x2c7c, 0xaf49,
    ];

    /// llama.cpp's bytes for [`TIE_BLOCK_F16_BITS`], from the same
    /// harness (and again cross-checked against `ggml_quantize_chunk`).
    const LLAMA_CPP_TIE_BLOCK_GOLDEN: [u8; Q8_0_BLOCK_BYTES] = [
        0xc2, 0x14, 0xb0, 0x0f, 0x20, 0x0a, 0x92, 0xfe, 0xdb, 0x6d, 0xc7, 0x00, 0x3f, 0x36, 0xe5,
        0x81, 0x71, 0x93, 0xc7, 0xc7, 0xc6, 0x32, 0x6c, 0xec, 0x6b, 0x7e, 0x71, 0xbc, 0x8d, 0x41,
        0x95, 0xc1, 0x3c, 0x9e,
    ];

    /// The reciprocal multiply is not a micro-optimisation, it is what
    /// llama.cpp does, and `v / d` rounds this block differently.
    #[test]
    fn the_scale_is_applied_as_llama_cpp_applies_it_not_as_a_division() {
        let x: Vec<f32> = TIE_BLOCK_F16_BITS
            .iter()
            .map(|b| f16::from_bits(*b).to_f32())
            .collect();
        let mut got = Vec::new();
        encode_row_q8_0(&x, &mut got).unwrap();
        assert_eq!(got.as_slice(), &LLAMA_CPP_TIE_BLOCK_GOLDEN[..]);

        // And the divide spelling really does differ here, so the test
        // above is asserting something rather than restating an
        // identity.
        let amax = x
            .iter()
            .fold(0f32, |a, &b| if a > b.abs() { a } else { b.abs() });
        let d = amax / 127.0;
        let divided: Vec<u8> = x.iter().map(|v| ((v / d).round() as i8) as u8).collect();
        assert_ne!(
            divided.as_slice(),
            &LLAMA_CPP_TIE_BLOCK_GOLDEN[2..],
            "this block no longer distinguishes the two spellings"
        );
    }

    /// An all-zero block stores a scale of +0.0, which is what
    /// llama.cpp stores. A scale of 1.0 dequantizes identically, so
    /// only a byte comparison catches it -- which is why this is its
    /// own test and not a corollary of a value check.
    #[test]
    fn an_all_zero_block_stores_a_zero_scale_the_way_llama_cpp_does() {
        let mut out = Vec::new();
        encode_row_q8_0(&[0.0; Q8_0_BLOCK_ELEMS], &mut out).unwrap();
        assert_eq!(out, vec![0u8; Q8_0_BLOCK_BYTES]);
    }

    /// A row whose length is not a whole number of blocks is refused,
    /// not padded. Padding would write more elements than the tensor's
    /// shape declares and every following row would decode shifted.
    #[test]
    fn a_row_that_is_not_a_whole_number_of_blocks_is_refused() {
        let mut out = Vec::new();
        assert!(encode_row_q8_0(&[0.5; Q8_0_BLOCK_ELEMS + 1], &mut out).is_none());
        assert!(encode_row_q8_0(&[0.5; 1], &mut out).is_none());
        assert!(encode_row_q8_0(&[], &mut out).is_some());
    }

    /// Round trip through this crate's own reader, bounded by Q8_0's
    /// own arithmetic rather than by a tolerance picked to make the
    /// test pass. One step is `d = amax/127`. Rounding to the nearest
    /// step costs at most `d/2`. The scale is then stored as an f16,
    /// whose half-ulp is `2^-11` relative, and that error is multiplied
    /// by the quant, `|q| <= 127`. So the bound is
    /// `d * (0.5 + 127 * 2^-11)`, about `0.562 d` -- and the observed
    /// worst case over this input sits just above `0.5 d`, which is why
    /// the f16 term is not optional.
    ///
    /// (A block whose `d` lands in f16's subnormal range would have a
    /// larger relative scale error; `sample_input` is nowhere near it.)
    #[test]
    fn dequantizing_what_this_encodes_lands_within_a_quantization_step() {
        let x = sample_input(8 * Q8_0_BLOCK_ELEMS);
        let mut bytes = Vec::new();
        encode_row_q8_0(&x, &mut bytes).unwrap();
        let back = dequant_q8_0(&bytes).unwrap();
        assert_eq!(back.len(), x.len());
        for (block_i, (chunk, got)) in x
            .chunks(Q8_0_BLOCK_ELEMS)
            .zip(back.chunks(Q8_0_BLOCK_ELEMS))
            .enumerate()
        {
            let amax = chunk.iter().fold(0f32, |a, &b| a.max(b.abs()));
            let step = amax / 127.0;
            let bound = step * (0.5 + 127.0 * 2f32.powi(-11));
            for (i, (&want, &have)) in chunk.iter().zip(got.iter()).enumerate() {
                assert!(
                    (want - have).abs() <= bound,
                    "block {block_i} element {i}: {want} -> {have}, error {} > {bound}",
                    (want - have).abs()
                );
            }
        }
    }
}
