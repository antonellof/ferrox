//! The 32-element legacy formats the CUDA `mul_mm` GEMM consumes:
//! Q8_0, Q4_0 and Q5_0. Each is a [`MulMmKind`] row plus the Rust twin
//! of its `dequant_src`, written statement for statement beside it.
//!
//! `nl` is 2 for every kind here: `il` selects the LOW half (elements
//! 0..16) or the HIGH half (16..32) of one block, not one of sixteen
//! sub-blocks the way the K-quants' `il` does.

use crate::mul_mm::{f16_to_f32, MulMmKind, SUB};

/// Q8_0: `half d`, then 32 `int8` quants. Transcribed from
/// `ferrox-metal`'s `Q8_0Dequant::get`.
pub const Q8_0: MulMmKind = MulMmKind {
    name: "Q8_0",
    module_name: "ferrox_mul_mm_q8_0",
    fn_name: "q8_0_mul_mm",
    block_bytes: 34,
    block_elems: 32,
    dequant_src: r#"
__device__ __forceinline__ void ferrox_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = ferrox_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const signed char* qs = (const signed char*)(xb + 2) + 16 * il;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        reg[i] = (float)qs[i] * d;
    }
}
"#,
    dequant_twin: dequant_sub_q8_0,
};

/// Scalar twin of [`Q8_0`]'s `dequant_src`. Read the two side by side:
/// the loop bounds, the pointer offset and the multiply order are the
/// same statements in two languages.
fn dequant_sub_q8_0(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_to_f32(u16::from(xb[0]) | (u16::from(xb[1]) << 8));
    let qs = &xb[2 + SUB * il..2 + SUB * il + SUB];
    for (r, q) in reg.iter_mut().zip(qs.iter()) {
        *r = f32::from(*q as i8) * d;
    }
}

/// Q4_0: `half d`, then 16 bytes holding 32 nibbles (low nibble of byte
/// `j` is element `j`, high nibble is element `j + 16`), each biased by
/// -8. Transcribed from `ferrox-metal`'s `Q4_0Dequant::get`, which
/// composes llama's `uint16` pair reads out of bytes because a GGUF
/// tensor row is only 2-byte aligned.
///
/// Note the bias: `d1 * q + (-8 * d)`, not `d * (q - 8)`. That is
/// llama's order and the twin mirrors it, so the two agree bit for bit
/// where fp32 rounding would otherwise separate them.
pub const Q4_0: MulMmKind = MulMmKind {
    name: "Q4_0",
    module_name: "ferrox_mul_mm_q4_0",
    fn_name: "q4_0_mul_mm",
    block_bytes: 18,
    block_elems: 32,
    dequant_src: r#"
__device__ __forceinline__ void ferrox_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = ferrox_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const unsigned char* qs = xb + 2;
    const float d1 = il ? d / 16.0f : d;
    const float d2 = d1 / 256.0f;
    const float md = -8.0f * d;
    const unsigned short mask0 = il ? 0x00F0 : 0x000F;
    const unsigned short mask1 = (unsigned short)(mask0 << 8);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned short w =
            (unsigned short)qs[2 * i] | ((unsigned short)qs[2 * i + 1] << 8);
        reg[2 * i + 0] = d1 * (float)(w & mask0) + md;
        reg[2 * i + 1] = d2 * (float)(w & mask1) + md;
    }
}
"#,
    dequant_twin: dequant_sub_q4_0,
};

/// Scalar twin of [`Q4_0`]'s `dequant_src`.
fn dequant_sub_q4_0(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_to_f32(u16::from(xb[0]) | (u16::from(xb[1]) << 8));
    let qs = &xb[2..2 + 16];
    let d1 = if il != 0 { d / 16.0 } else { d };
    let d2 = d1 / 256.0;
    let md = -8.0 * d;
    let mask0: u16 = if il != 0 { 0x00F0 } else { 0x000F };
    let mask1: u16 = mask0 << 8;
    for i in 0..8 {
        let w = u16::from(qs[2 * i]) | (u16::from(qs[2 * i + 1]) << 8);
        reg[2 * i] = d1 * f32::from(w & mask0) + md;
        reg[2 * i + 1] = d2 * f32::from(w & mask1) + md;
    }
}

/// Q5_0: `half d`, `uint32 qh`, then 16 bytes holding 32 nibbles. The
/// fifth bit of each quant lives in `qh`, and each value is biased by
/// -16. Transcribed from `ferrox-metal`'s `Q5_0Dequant::get`, which is
/// llama's `dequantize_q5_0`.
///
/// **`nl` is 2, not 16.** This is a 32-element legacy block like Q4_0,
/// so `il` selects the LOW half (elements 0..16) or the HIGH half
/// (elements 16..32) -- it does not index one of sixteen sub-blocks the
/// way the K-quants' `il` does. Every derived quantity below flips on
/// that one bit, and the four that do are easy to confuse:
///
/// - `mask` picks the low or high nibble of each `qs` byte,
/// - `x_mv` shifts the high nibble back down to 0..15,
/// - `gh_mv` selects `qh` bit `j` (low half) or bit `j + 16` (high),
/// - `gh_bk` puts that bit into position 4.
///
/// `gh_mv`/`gh_bk` are llama's two spellings of the same fifth bit:
/// `((qh >> j) << 4) & 0x10` for the low half and
/// `(qh >> (j + 12)) & 0x10` for the high one, which is `12`, not `16`,
/// precisely because the bit is left in place rather than shifted to 0.
pub const Q5_0: MulMmKind = MulMmKind {
    name: "Q5_0",
    module_name: "ferrox_mul_mm_q5_0",
    fn_name: "q5_0_mul_mm",
    block_bytes: 22,
    block_elems: 32,
    dequant_src: r#"
__device__ __forceinline__ void ferrox_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = ferrox_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const float md = -16.0f * d;
    const unsigned int qh = (unsigned int)xb[2]
        | ((unsigned int)xb[3] << 8)
        | ((unsigned int)xb[4] << 16)
        | ((unsigned int)xb[5] << 24);
    const unsigned char* qs = xb + 6;
    const unsigned short mask = il ? 0x00F0 : 0x000F;
    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ? 0 : 4;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned short w =
            (unsigned short)qs[2 * i] | ((unsigned short)qs[2 * i + 1] << 8);
        const unsigned char xh_0 =
            (unsigned char)(((qh >> (gh_mv + 2 * i)) << gh_bk) & 0x10u);
        const unsigned char xh_1 =
            (unsigned char)(((qh >> (gh_mv + 2 * i + 1)) << gh_bk) & 0x10u);
        const int x0 = (int)((((w) & mask) >> x_mv) | xh_0);
        const int x1 = (int)((((w >> 8) & mask) >> x_mv) | xh_1);
        reg[2 * i + 0] = d * (float)x0 + md;
        reg[2 * i + 1] = d * (float)x1 + md;
    }
}
"#,
    dequant_twin: dequant_sub_q5_0,
};

/// Scalar twin of [`Q5_0`]'s `dequant_src`.
///
/// Note the bias, as in [`Q4_0`]: `d * q + (-16 * d)`, not
/// `d * (q - 16)`. That is llama's order and the twin mirrors it.
fn dequant_sub_q5_0(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_to_f32(u16::from(xb[0]) | (u16::from(xb[1]) << 8));
    let md = -16.0 * d;
    let qh = u32::from(xb[2])
        | (u32::from(xb[3]) << 8)
        | (u32::from(xb[4]) << 16)
        | (u32::from(xb[5]) << 24);
    let qs = &xb[6..6 + 16];
    let mask: u16 = if il != 0 { 0x00F0 } else { 0x000F };
    let x_mv = if il != 0 { 4 } else { 0 };
    let gh_mv = if il != 0 { 12 } else { 0 };
    let gh_bk = if il != 0 { 0 } else { 4 };
    for i in 0..8 {
        let w = u16::from(qs[2 * i]) | (u16::from(qs[2 * i + 1]) << 8);
        let xh_0 = (((qh >> (gh_mv + 2 * i)) << gh_bk) & 0x10) as u8;
        let xh_1 = (((qh >> (gh_mv + 2 * i + 1)) << gh_bk) & 0x10) as u8;
        let x0 = i32::from((((w & mask) >> x_mv) as u8) | xh_0);
        let x1 = i32::from(((((w >> 8) & mask) >> x_mv) as u8) | xh_1);
        reg[2 * i] = d * x0 as f32 + md;
        reg[2 * i + 1] = d * x1 as f32 + md;
    }
}
