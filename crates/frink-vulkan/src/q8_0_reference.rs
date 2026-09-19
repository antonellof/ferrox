//! The scalar twin of [`crate::q8_0_shader`], and the byte→word packing
//! both the twin and the real upload path share.
//!
//! # What this is for
//!
//! This repo's rule for a kernel it cannot run is the one every
//! `unsafe` SIMD arm here already follows and that `frink-cuda`'s
//! `mul_mm_ref` follows for CUDA: a scalar twin implementing *identical
//! arithmetic*, held against an independent reference.
//!
//! [`matvec_reference`] is not "a Q8_0 matvec that should agree". It is
//! a transcription of the shader: same word-indexed byte extraction,
//! same integer f16 decode, same `(d * q) * x` term shape, same
//! block-ascending then element-ascending accumulation order, same
//! `row < rows` guard. Read it beside `emit_main`; a line here with no
//! counterpart there is a bug in one of them.
//!
//! Two independent references check it, and neither shares any code
//! with it:
//!
//! - [`f16_to_f32`] against the `half` crate, over **all 65,536** f16
//!   bit patterns.
//! - the whole matvec against `frink_quant::dequant_q8_0` followed by
//!   a plain dot product -- a different unpacker, a different f16
//!   decoder, and the same numbers.
//!
//! What that does *not* establish is that the emitted SPIR-V says what
//! this file says; the two are hand-transcribed from each other. On a
//! machine with a Vulkan driver, `crate::device` closes that gap by
//! running the real shader and comparing. See the crate docs for
//! whether that has happened.

use crate::q8_0_shader::{BLOCK_BYTES, BLOCK_ELEMS};

/// Pack raw bytes into the `uint[]` a storage buffer holds, zero-padding
/// the tail.
///
/// A Q8_0 block is 34 bytes, so a row of an odd number of blocks is not
/// 4-byte aligned and the *last* word of the buffer is usually partial.
/// Both the twin and the device upload go through here so the padding
/// cannot differ between them.
pub fn pack_words(bytes: &[u8]) -> Vec<u32> {
    let mut words = Vec::with_capacity(bytes.len().div_ceil(4));
    for chunk in bytes.chunks(4) {
        let mut w = [0u8; 4];
        w[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(w));
    }
    words
}

/// `(w[k >> 2] >> ((k & 3) * 8)) & 0xff` -- the shader's byte read.
#[inline]
fn weight_byte(words: &[u32], k: usize) -> u32 {
    (words[k >> 2] >> ((k & 3) * 8)) & 0xff
}

/// The shader's integer f16 decode, in Rust.
///
/// No `f16` type, no `half`: this is the same `OpSelect` chain
/// [`crate::q8_0_shader::Kernel::decode_f16`] emits, so a mistake in
/// either shows up as a disagreement with `half` in the tests below.
pub fn f16_to_f32(h: u32) -> f32 {
    let sign = h >> 15;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let mant_hi = mant << 13;

    let normal = f32::from_bits(((exp + 112) << 23) | mant_hi);
    let subnormal = mant as f32 * f32::from_bits(0x3380_0000);
    let inf_or_nan = f32::from_bits(0x7f80_0000 | mant_hi);

    let magnitude = if exp == 0 {
        subnormal
    } else if exp == 31 {
        inf_or_nan
    } else {
        normal
    };
    if sign != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Host emulation of the Q8_0 matvec shader.
///
/// `weight_words` is [`pack_words`] applied to `rows * row_bytes` of
/// GGUF Q8_0 data; `x` holds `n_blocks_per_row * 32` activations.
/// Returns `rows` floats. `row_bytes` is passed rather than derived,
/// mirroring the push-constant block.
///
/// This is a correctness reference, not a fast path.
pub fn matvec_reference(
    weight_words: &[u32],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; rows];
    for (row, y) in out.iter_mut().enumerate() {
        // The shader's `row < rows` guard; every row inside the buffer
        // is in range by construction, and the guard exists for the
        // padding invocations a 64-wide workgroup dispatches.
        let row_base = row * row_bytes;
        let mut acc = 0f32;
        for b in 0..n_blocks_per_row {
            let off = row_base + b * BLOCK_BYTES;
            let lo = weight_byte(weight_words, off);
            let hi = weight_byte(weight_words, off + 1);
            let scale = f16_to_f32(lo | (hi << 8));
            let x_base = b * BLOCK_ELEMS;
            let q_base = off + 2;
            for j in 0..BLOCK_ELEMS {
                let q_byte = weight_byte(weight_words, q_base + j);
                // (b ^ 0x80) - 0x80, wrapping, is int8 sign extension.
                let biased = (q_byte ^ 128).wrapping_sub(128);
                let q = (biased as i32) as f32;
                acc += scale * q * x[x_base + j];
            }
        }
        *y = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic, dependency-free values in `[-1, 1)`.
    fn pseudo_random(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn build_rows(rows: usize, cols: usize, seed: u64) -> (Vec<u8>, Vec<Vec<f32>>) {
        let mut bytes = Vec::new();
        let mut dense = Vec::new();
        for r in 0..rows {
            let row = pseudo_random(seed + r as u64 * 7919, cols);
            bytes.extend(frink_quant::quantize_q8_0(&row));
            dense.push(row);
        }
        (bytes, dense)
    }

    #[test]
    fn f16_decode_matches_half_crate_on_every_bit_pattern() {
        let mut checked = 0u32;
        for bits in 0..=u16::MAX {
            let want = half::f16::from_bits(bits).to_f32();
            let got = f16_to_f32(bits as u32);
            if want.is_nan() {
                assert!(got.is_nan(), "0x{bits:04x}: expected NaN, got {got}");
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "0x{bits:04x}: {got} != {want}"
                );
            }
            checked += 1;
        }
        assert_eq!(checked, 65_536);
    }

    #[test]
    fn pack_words_zero_pads_a_partial_tail() {
        assert_eq!(pack_words(&[1, 2, 3, 4]), vec![0x0403_0201]);
        assert_eq!(pack_words(&[1, 2, 3]), vec![0x0003_0201]);
        assert_eq!(pack_words(&[]), Vec::<u32>::new());
        // A single Q8_0 block is 34 bytes -> 9 words, last one half full.
        assert_eq!(pack_words(&[0u8; BLOCK_BYTES]).len(), 9);
    }

    /// The twin against an independent unpacker: `frink_quant`'s
    /// `dequant_q8_0` plus a plain dot product. Both do the same
    /// multiplications in the same order, so this is exact rather than
    /// approximate; a tolerance here would hide a transcription error.
    #[test]
    fn reference_matches_frink_quant_dequant_then_dot() {
        for (rows, blocks) in [(1usize, 1usize), (5, 3), (64, 8), (7, 2)] {
            let cols = blocks * BLOCK_ELEMS;
            let (bytes, _) = build_rows(rows, cols, 0xfe11 + rows as u64);
            let row_bytes = blocks * BLOCK_BYTES;
            assert_eq!(bytes.len(), rows * row_bytes);
            let x = pseudo_random(0xa5a5, cols);
            let got = matvec_reference(&pack_words(&bytes), &x, rows, row_bytes, blocks);

            for (r, g) in got.iter().enumerate() {
                let dequantized =
                    frink_quant::dequant_q8_0(&bytes[r * row_bytes..(r + 1) * row_bytes]).unwrap();
                let want: f32 = dequantized
                    .iter()
                    .zip(&x)
                    .fold(0f32, |acc, (w, xv)| acc + w * xv);
                assert_eq!(
                    g.to_bits(),
                    want.to_bits(),
                    "rows={rows} blocks={blocks} row={r}: {g} != {want}"
                );
            }
        }
    }

    /// The alignment case the whole byte-extraction design exists for:
    /// with an odd block count `row_bytes` is 34 * odd, so every row
    /// after the first starts at a byte offset that is not a multiple
    /// of 4. Deleting the `(k & 3) * 8` shift makes this red while the
    /// even-block cases stay green.
    #[test]
    fn reference_is_correct_when_rows_are_not_word_aligned() {
        let blocks = 3;
        let cols = blocks * BLOCK_ELEMS;
        let row_bytes = blocks * BLOCK_BYTES;
        assert_eq!(row_bytes % 4, 2, "this test needs a misaligned row stride");
        let rows = 9;
        let (bytes, _) = build_rows(rows, cols, 0x0dd0);
        let x = pseudo_random(0x1234, cols);
        let got = matvec_reference(&pack_words(&bytes), &x, rows, row_bytes, blocks);
        for (r, g) in got.iter().enumerate() {
            let dequantized =
                frink_quant::dequant_q8_0(&bytes[r * row_bytes..(r + 1) * row_bytes]).unwrap();
            let want: f32 = dequantized
                .iter()
                .zip(&x)
                .fold(0f32, |acc, (w, xv)| acc + w * xv);
            assert_eq!(g.to_bits(), want.to_bits(), "row {r}");
        }
    }

    #[test]
    fn sign_extension_covers_the_whole_int8_range() {
        for v in 0..=255u32 {
            let got = ((v ^ 128).wrapping_sub(128) as i32) as f32;
            let want = (v as u8 as i8) as f32;
            assert_eq!(got, want, "byte {v}");
        }
    }
}
