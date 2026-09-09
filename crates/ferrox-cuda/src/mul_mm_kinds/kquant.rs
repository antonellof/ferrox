//! The 256-element K-quant super-block formats the CUDA `mul_mm` GEMM
//! consumes: Q4_K, Q5_K and Q6_K. Each is a [`MulMmKind`] row plus the
//! Rust twin of its `dequant_src`, written statement for statement
//! beside it.
//!
//! `nl` is 16 for every kind here: `il` selects one of sixteen
//! 16-element sub-blocks of a super-block.

use crate::mul_mm::{f16_to_f32, MulMmKind, SUB};

/// The 6-bit scale/min unpack Q4_K and Q5_K share, as CUDA C.
///
/// llama's `get_scale_min_k4_just2`: eight pairs are packed into twelve
/// bytes, the low four plainly and the high four with their top two bits
/// borrowed from the low bytes. Emitted once and textually included by
/// both kinds, so the two cannot drift apart.
pub const K_SCALE_MIN_SRC: &str = r#"
__device__ __forceinline__ void ferrox_k_scale_min_just2(
    int j, int k, const unsigned char* q, unsigned char* out
) {
    if (j < 4) {
        out[0] = (unsigned char)(q[j + 0 + k] & 63);
        out[1] = (unsigned char)(q[j + 4 + k] & 63);
    } else {
        out[0] = (unsigned char)((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2));
        out[1] = (unsigned char)((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2));
    }
}
"#;

/// Scalar twin of `K_SCALE_MIN_SRC`.
fn k_scale_min_just2(j: usize, k: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j + k] & 63, q[j + 4 + k] & 63)
    } else {
        (
            (q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2),
            (q[j + 4 + k] >> 4) | ((q[j + k] & 0xc0) >> 2),
        )
    }
}

/// Q4_K: `half d`, `half dmin`, 12 scale bytes, 128 nibble-packed
/// quants. Transcribed from `ferrox-metal`'s `q4k_dequant_16`, which is
/// llama's `dequantize_q4_K`.
///
/// `il` selects one of sixteen 16-value sub-blocks. Note that `il` is
/// consumed three times in three different forms before the loop: `il /
/// 4` picks the 64-value group, `il & 1` the half within it, and only
/// then is `il` masked to `il & 3` for the scale lookup and the nibble
/// half. Getting that order wrong reads plausible values from the wrong
/// place, which is why the twin repeats it statement for statement.
pub const Q4_K: MulMmKind = MulMmKind {
    name: "Q4_K",
    module_name: "ferrox_mul_mm_q4_k",
    fn_name: "q4_k_mul_mm",
    block_bytes: 144,
    block_elems: 256,
    dequant_src: r#"
__device__ __forceinline__ void ferrox_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d_all = ferrox_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const float dmin = ferrox_f16_to_f32(
        (unsigned short)xb[2] | ((unsigned short)xb[3] << 8));
    const unsigned char* scales = xb + 4;
    const unsigned char* q = xb + 16 + (il / 4) * 32 + 16 * (il & 1);

    const int is = (il / 4) * 2;
    const int ilm = il & 3;
    unsigned char sc[2];
    ferrox_k_scale_min_just2(is, ilm / 2, scales, sc);
    const float d = ilm < 2 ? d_all : d_all / 16.0f;
    const float dl = d * (float)sc[0];
    const float ml = dmin * (float)sc[1];
    const unsigned char mask = ilm < 2 ? 0x0F : 0xF0;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        reg[i] = dl * (float)(q[i] & mask) - ml;
    }
}
"#,
    dequant_twin: dequant_sub_q4_k,
};

/// Scalar twin of [`Q4_K`]'s `dequant_src`.
fn dequant_sub_q4_k(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d_all = f16_to_f32(u16::from(xb[0]) | (u16::from(xb[1]) << 8));
    let dmin = f16_to_f32(u16::from(xb[2]) | (u16::from(xb[3]) << 8));
    let scales = &xb[4..16];
    let base = 16 + (il / 4) * 32 + 16 * (il & 1);
    let q = &xb[base..base + 16];

    let is = (il / 4) * 2;
    let ilm = il & 3;
    let (sc0, sc1) = k_scale_min_just2(is, ilm / 2, scales);
    let d = if ilm < 2 { d_all } else { d_all / 16.0 };
    let dl = d * f32::from(sc0);
    let ml = dmin * f32::from(sc1);
    let mask: u8 = if ilm < 2 { 0x0F } else { 0xF0 };
    for (r, qv) in reg.iter_mut().zip(q.iter()) {
        *r = dl * f32::from(qv & mask) - ml;
    }
}

/// Q5_K: `half d`, `half dmin`, 12 scale bytes, 32 high-bit bytes, 128
/// nibble-packed quants. The fifth bit of each quant lives in `qh`.
/// Transcribed from `ferrox-metal`'s `Q5KDequant`, which is llama's
/// `dequantize_q5_K`.
///
/// `ul` is built from the UNMASKED `il` (`1 << (il / 2)`, so bits 0..7),
/// while the scale lookup uses `il & 3`. Two different derivations of
/// the same input, in that order.
pub const Q5_K: MulMmKind = MulMmKind {
    name: "Q5_K",
    module_name: "ferrox_mul_mm_q5_k",
    fn_name: "q5_k_mul_mm",
    block_bytes: 176,
    block_elems: 256,
    dequant_src: r#"
__device__ __forceinline__ void ferrox_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d_all = ferrox_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const float dmin = ferrox_f16_to_f32(
        (unsigned short)xb[2] | ((unsigned short)xb[3] << 8));
    const unsigned char* scales = xb + 4;
    const unsigned char* q = xb + 48 + 32 * (il / 4) + 16 * (il & 1);
    const unsigned char* qh = xb + 16 + 16 * (il & 1);

    const int is = (il / 4) * 2;
    const unsigned char ul = (unsigned char)(1 << (il / 2));
    const int ilm = il & 3;
    unsigned char sc[2];
    ferrox_k_scale_min_just2(is, ilm / 2, scales, sc);
    const float d = ilm < 2 ? d_all : d_all / 16.0f;
    const float dl = d * (float)sc[0];
    const float ml = dmin * (float)sc[1];
    const unsigned char mask = ilm < 2 ? 0x0F : 0xF0;
    const float qh_val = ilm < 2 ? 16.0f : 256.0f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const float hi = (qh[i] & ul) ? qh_val : 0.0f;
        reg[i] = dl * ((float)(q[i] & mask) + hi) - ml;
    }
}
"#,
    dequant_twin: dequant_sub_q5_k,
};

/// Scalar twin of [`Q5_K`]'s `dequant_src`.
fn dequant_sub_q5_k(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d_all = f16_to_f32(u16::from(xb[0]) | (u16::from(xb[1]) << 8));
    let dmin = f16_to_f32(u16::from(xb[2]) | (u16::from(xb[3]) << 8));
    let scales = &xb[4..16];
    let qbase = 48 + 32 * (il / 4) + 16 * (il & 1);
    let q = &xb[qbase..qbase + 16];
    let hbase = 16 + 16 * (il & 1);
    let qh = &xb[hbase..hbase + 16];

    let is = (il / 4) * 2;
    let ul: u8 = 1u8 << (il / 2);
    let ilm = il & 3;
    let (sc0, sc1) = k_scale_min_just2(is, ilm / 2, scales);
    let d = if ilm < 2 { d_all } else { d_all / 16.0 };
    let dl = d * f32::from(sc0);
    let ml = dmin * f32::from(sc1);
    let mask: u8 = if ilm < 2 { 0x0F } else { 0xF0 };
    let qh_val: f32 = if ilm < 2 { 16.0 } else { 256.0 };
    for i in 0..SUB {
        let hi = if qh[i] & ul != 0 { qh_val } else { 0.0 };
        reg[i] = dl * (f32::from(q[i] & mask) + hi) - ml;
    }
}

/// Q6_K: `ql[128]`, `qh[64]`, `int8 scales[16]`, `half d` -- the scale
/// is at the END of the block, not the start. Transcribed from
/// `ferrox-metal`'s `Q6KDequant`, which is llama's `dequantize_q6_K`.
///
/// This one reconstructs llama's `uint16` pair reads out of individual
/// bytes on purpose: a GGUF tensor row is only 2-byte aligned, so a
/// wider load is not safe to assume. The four masks and three shifts
/// are llama's, and the four outputs per iteration come from the four
/// bytes of one `uint` in ascending order.
pub const Q6_K: MulMmKind = MulMmKind {
    name: "Q6_K",
    module_name: "ferrox_mul_mm_q6_k",
    fn_name: "q6_k_mul_mm",
    block_bytes: 210,
    block_elems: 256,
    dequant_src: r#"
__device__ __forceinline__ void ferrox_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d_all = ferrox_f16_to_f32(
        (unsigned short)xb[208] | ((unsigned short)xb[209] << 8));
    const unsigned char* ql8 = xb;
    const unsigned char* qh8 = xb + 128;
    const signed char* scales = (const signed char*)(xb + 192);

    const int ql_off = 64 * (il / 8) + 32 * ((il / 2) & 1) + 16 * (il & 1);
    const int qh_off = 32 * (il / 8) + 16 * (il & 1);
    const float sc = (float)scales[(il % 2) + 2 * (il / 2)];
    const int ilm = (il / 2) & 3;

    const unsigned int kmask1 = ilm > 1 ? (ilm > 2 ? 0xC0C0C0C0u : 0x30303030u)
                                        : (ilm > 0 ? 0x0C0C0C0Cu : 0x03030303u);
    const unsigned int kmask2 = ilm > 1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
    const float ml = d_all * sc * 32.0f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.0f;
    const float dl2 = dl0 / (256.0f * 256.0f);
    const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
    const int shr_h = ilm > 2 ? 2 : 0;
    const int shl_h = ilm > 1 ? 0 : (ilm > 0 ? 2 : 4);
    const int shr_l = ilm > 1 ? 4 : 0;

#pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned char* lp = ql8 + ql_off + 4 * i;
        const unsigned char* hp = qh8 + qh_off + 4 * i;
        const unsigned int low =
            (((unsigned int)lp[0] | ((unsigned int)lp[1] << 8))
             | (((unsigned int)lp[2] | ((unsigned int)lp[3] << 8)) << 16)) & kmask2;
        const unsigned int high =
            (((unsigned int)hp[0] | ((unsigned int)hp[1] << 8))
             | (((unsigned int)hp[2] | ((unsigned int)hp[3] << 8)) << 16)) & kmask1;
        const unsigned int q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[4 * i + 0] = dl0 * (float)(q & 0xFFu) - ml;
        reg[4 * i + 1] = dl1 * (float)(q & 0xFF00u) - ml;
        reg[4 * i + 2] = dl2 * (float)(q & 0xFF0000u) - ml;
        reg[4 * i + 3] = dl3 * (float)(q & 0xFF000000u) - ml;
    }
}
"#,
    dequant_twin: dequant_sub_q6_k,
};

/// Scalar twin of [`Q6_K`]'s `dequant_src`.
fn dequant_sub_q6_k(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d_all = f16_to_f32(u16::from(xb[208]) | (u16::from(xb[209]) << 8));
    let ql8 = xb;
    let qh8 = &xb[128..];
    let scales = &xb[192..208];

    let ql_off = 64 * (il / 8) + 32 * ((il / 2) & 1) + 16 * (il & 1);
    let qh_off = 32 * (il / 8) + 16 * (il & 1);
    let sc = f32::from(scales[(il % 2) + 2 * (il / 2)] as i8);
    let ilm = (il / 2) & 3;

    let kmask1: u32 = if ilm > 1 {
        if ilm > 2 {
            0xC0C0_C0C0
        } else {
            0x3030_3030
        }
    } else if ilm > 0 {
        0x0C0C_0C0C
    } else {
        0x0303_0303
    };
    let kmask2: u32 = if ilm > 1 { 0xF0F0_F0F0 } else { 0x0F0F_0F0F };
    let ml = d_all * sc * 32.0;
    let dl0 = d_all * sc;
    let dl1 = dl0 / 256.0;
    let dl2 = dl0 / (256.0 * 256.0);
    let dl3 = dl0 / (256.0 * 256.0 * 256.0);
    let shr_h = if ilm > 2 { 2 } else { 0 };
    let shl_h = if ilm > 1 {
        0
    } else if ilm > 0 {
        2
    } else {
        4
    };
    let shr_l = if ilm > 1 { 4 } else { 0 };

    for i in 0..4 {
        let lp = &ql8[ql_off + 4 * i..ql_off + 4 * i + 4];
        let hp = &qh8[qh_off + 4 * i..qh_off + 4 * i + 4];
        let low = ((u32::from(lp[0]) | (u32::from(lp[1]) << 8))
            | ((u32::from(lp[2]) | (u32::from(lp[3]) << 8)) << 16))
            & kmask2;
        let high = ((u32::from(hp[0]) | (u32::from(hp[1]) << 8))
            | ((u32::from(hp[2]) | (u32::from(hp[3]) << 8)) << 16))
            & kmask1;
        let q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[4 * i] = dl0 * (q & 0xFF) as f32 - ml;
        reg[4 * i + 1] = dl1 * (q & 0xFF00) as f32 - ml;
        reg[4 * i + 2] = dl2 * (q & 0x00FF_0000) as f32 - ml;
        reg[4 * i + 3] = dl3 * (q & 0xFF00_0000) as f32 - ml;
    }
}
