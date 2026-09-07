//! `mul_mm`: a batched quantized GEMM for CUDA -- the CUDA C source and
//! the per-quant-kind dispatch table.
//!
//! # UNRUN ON HARDWARE
//!
//! **No kernel in this module has ever executed on a GPU.** There is no
//! NVIDIA hardware in the environment it was written in, and this repo's
//! standing rule (`docs/plans/roadmap.md`, "CUDA stays at *must
//! compile*") is that a CUDA claim is worth nothing until someone
//! measures it. Nothing here may be described as a measured capability
//! in `docs/FEATURES.md` or `docs/MODELS.md` until
//! `cargo test -p ferrox-cuda --features cuda -- --ignored` has been run
//! on a real device and the result written down.
//!
//! What *is* established, and how:
//!
//! 1. **It is a port, not a design.** The arithmetic comes from
//!    `ferrox-metal`'s `mul_mm_sg_impl` (`crates/ferrox-metal/src/gpu.rs`),
//!    which is at parity with llama.cpp on Metal and has goldens. The
//!    per-kind unpack functions below are line-for-line transcriptions of
//!    that file's `Q8_0Dequant` / `Q4_0Dequant` / `Q5_0Dequant` /
//!    `Q5KDequant` / `Q6KDequant` functors, which are themselves
//!    llama's `dequantize_q8_0` / `dequantize_q4_0` /
//!    `dequantize_q5_0` / `dequantize_q5_K` / `dequantize_q6_K`.
//! 2. **It has a scalar twin.** [`crate::mul_mm_ref`] emulates this
//!    kernel on the CPU -- same tiling, same clamping, same index
//!    arithmetic, same accumulation order -- and its tests, which run in
//!    the default (no-GPU) build, hold it against an independent
//!    dequantize-then-GEMM built on `ferrox_quant`. A transcription
//!    error in the index math or the unpack shows up there.
//! 3. **Geometry cannot drift.** Every tile constant in the emitted CUDA
//!    is `#define`d from the Rust constants in this module, which are
//!    the same constants the twin uses. The kernel and its twin cannot
//!    disagree about a tile size; they can only disagree about the ~10
//!    lines of unpack expression, which is what (1) and (2) cover.
//!
//! 4. **The emitted C itself was executed.**
//!    `tools/mul_mm_host_check/run.sh` compiles each generated `.cu`
//!    against a keyword-and-barrier shim and runs it on the host CPU --
//!    one real thread per CUDA thread, a counting barrier for
//!    `__syncthreads()`, one block at a time so `__shared__` behaves --
//!    then compares it to the twin. Result on 2026-09-01, macOS/clang,
//!    both kinds then in the table, three shapes (exact tiles, partial
//!    on both axes, narrow batch): **zero mismatches, bit for bit**,
//!    including 1,458 positions where a degenerate f16 scale made both
//!    sides NaN together. Deleting one term of the CUDA-only index
//!    arithmetic makes it fail, so it is a check and not a formality.
//!
//!    Re-run on 2026-09-05 over all six kinds: **zero mismatches**
//!    again, 40,932 compared positions. That run was the first for the
//!    K-quants and for Q5_0 -- the tool took `n_cols` from a fixed list
//!    that is not a whole Q4_K super-block, so from the day the
//!    K-quants landed it panicked on the first one and checked
//!    NOTHING. A tool that cannot fire is worse than no tool; `n_cols`
//!    is now rounded up per kind. Sabotaging Q5_0's `qh` shift
//!    (`12` for `16`) makes it report 6,096 mismatches.
//!
//! What none of that covers, and what only hardware can settle: that
//! NVRTC accepts the source (clang and NVRTC are different front ends),
//! that the barrier placement survives a real warp scheduler, that the
//! launch configuration is valid on the target device, and what any of
//! it costs. A real GPU also contracts `acc += a * b` into an FMA, so
//! on-device results will be *close to* rather than equal to the twin's;
//! the hardware test compares with a relative tolerance for that reason.
//!
//! # Why this shape
//!
//! `ferrox-cuda` had no matrix-matrix product of any kind, so a batched
//! prefill decomposed into one matvec per position
//! (`crates/ferrox-core/src/weight_matrix.rs`, the CUDA batch arm). This
//! is the naive tiled `mul_mm` half of
//! `docs/plans/llama-cpp-gap-inventory.md` §2.7; the `dp4a` integer path
//! (llama's `mmq.cu`) is explicitly *not* attempted here.
//!
//! Unlike Metal's version this uses no matrix-fragment intrinsics
//! (`wmma`/`mma`): the K-loop is a plain fp32 FMA over shared-memory
//! tiles. That costs the constant factor and buys a kernel whose
//! arithmetic a CPU twin can reproduce exactly, which is the only kind
//! of correctness available without a device.

/// Rows of the weight matrix per threadblock tile.
pub const BM: usize = 64;
/// Tokens (batch entries) per threadblock tile.
pub const BN: usize = 64;
/// K-elements consumed per tile step. Must be a multiple of [`SUB`], and
/// every real quantized row length is a multiple of 32, so a K-loop that
/// steps 32 never straddles a partial block.
pub const BK: usize = 32;
/// Rows of the output micro-tile each thread owns.
pub const TM: usize = 4;
/// Columns of the output micro-tile each thread owns.
pub const TN: usize = 4;
/// Threads per block. `BM/TM * BN/TN` -- one thread per micro-tile.
pub const THREADS: usize = (BM / TM) * (BN / TN);
/// Elements produced by one call to the per-kind unpack function. This
/// is llama's (and `ferrox-metal`'s) sub-block granularity: `il` selects
/// which 16 consecutive elements of a super-block to decode.
pub const SUB: usize = 16;

// Compile-time geometry gates. These are the invariants the kernel's
// `tx`/`ty` decomposition and its A-tile loader assume; a retune that
// broke one would otherwise produce a kernel that launches and is
// wrong, so they fail the build rather than a test.
const _: () = assert!(THREADS == (BM / TM) * (BN / TN));
const _: () = assert!(BM.is_multiple_of(TM) && BN.is_multiple_of(TN));
const _: () = assert!(
    BK.is_multiple_of(SUB),
    "the K-tile must be whole sub-blocks"
);
const _: () = assert!(
    BM * (BK / SUB) <= THREADS,
    "the A-tile loader uses a prefix of the block"
);
const _: () = assert!(THREADS <= 1024, "CUDA caps a block at 1024 threads");
// The inner loop reads its micro-tile operands as `float4`, which is
// what keeps a warp's 32 lanes off eight shared-memory banks. That
// needs each thread's slice to start 16-byte aligned: the micro-tile
// widths must be multiples of four, and so must the shared rows they
// index into, or the fourth lane of a row starts mid-vector.
const _: () = assert!(
    TM.is_multiple_of(4) && TN.is_multiple_of(4),
    "the inner loop loads float4 from shared memory"
);
const _: () = assert!(
    BM.is_multiple_of(4) && BN.is_multiple_of(4),
    "a shared row must keep the next row 16-byte aligned"
);

/// One quantized weight format the GEMM can consume.
///
/// Adding a format is a row in [`KINDS`] plus a `dequant_src` snippet --
/// never a second copy of the GEMM body. That is the seam
/// `ferrox-metal` proved: its `mul_mm_sg_impl` is one templated body
/// with seven `Dequant` functors, written that way *because* the
/// previous copy-per-format generation is how
/// `gqa_prefill_fa_vec_d256` ended up handling half a head.
#[derive(Debug, Clone, Copy)]
pub struct MulMmKind {
    /// GGUF quant name, for error messages.
    pub name: &'static str,
    /// NVRTC module cache key. Must be unique per kind.
    pub module_name: &'static str,
    /// `__global__` entry point name inside that module.
    pub fn_name: &'static str,
    /// On-disk stride of one super-block.
    pub block_bytes: usize,
    /// Elements one super-block decodes to.
    pub block_elems: usize,
    /// CUDA C defining
    /// `void ferrox_dequant_sub(const unsigned char* xb, int il, float* reg)`,
    /// writing `SUB` floats: the elements at `[SUB*il, SUB*il + SUB)`
    /// of the super-block at `xb`, in ascending element order.
    pub dequant_src: &'static str,
    /// The scalar twin of `dequant_src`: the same arithmetic in Rust,
    /// on the host, in the same order. It sits in this struct rather
    /// than in a parallel table so a kind cannot be added without one --
    /// the untestable half and the testable half are the same row.
    pub dequant_twin: fn(xb: &[u8], il: usize, reg: &mut [f32; SUB]),
}

impl MulMmKind {
    /// 16-value sub-blocks per super-block -- llama's `nl` template
    /// argument (2 for the 32-element legacy formats, 16 for the
    /// 256-element K-quants).
    pub const fn nl(&self) -> usize {
        self.block_elems / SUB
    }
}

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

/// The 6-bit scale/min unpack Q4_K and Q5_K share, as CUDA C.
///
/// llama's `get_scale_min_k4_just2`: eight pairs are packed into twelve
/// bytes, the low four plainly and the high four with their top two bits
/// borrowed from the low bytes. Emitted once and textually included by
/// both kinds, so the two cannot drift apart.
const K_SCALE_MIN_SRC: &str = r#"
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

/// Scalar twin of the CUDA `ferrox_f16_to_f32` in `F16_SRC`: the same
/// bit surgery, including the `exp == 31` NaN/Inf arm. `ldexpf(m, e)`
/// with an exact power of two is a multiply, so this is bit-identical
/// rather than merely close.
///
/// Held against `half::f16` by a test, which is what makes it a twin of
/// something and not a second guess.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);
    let scale = if exp == 0 {
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 31 {
        if mant != 0 {
            f32::from_bits(0x7fc0_0000)
        } else {
            f32::from_bits(0x7f80_0000)
        }
    } else {
        ((mant | 0x400) as f32) * 2f32.powi(exp as i32 - 25)
    };
    if sign != 0 {
        -scale
    } else {
        scale
    }
}

/// The dispatch table. A caller looks up by GGUF quant name; a new
/// format is one row here.
pub const KINDS: &[MulMmKind] = &[Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K];

/// Looks up a kind by its GGUF quant name (`"Q4_0"`, `"Q8_0"`).
/// `None` means this GEMM does not implement that format -- the caller
/// must fall back and say so, never compute something else.
pub fn kind_by_name(name: &str) -> Option<&'static MulMmKind> {
    KINDS.iter().find(|k| k.name == name)
}

/// f16 -> f32 by explicit bit surgery, shared with the matvec kernels in
/// `gpu.rs`. NVRTC has `__half` available but only with the CUDA headers
/// on the include path, which this crate deliberately does not require.
const F16_SRC: &str = r#"
__device__ __forceinline__ float ferrox_f16_to_f32(unsigned short bits) {
    unsigned int sign = (bits >> 15) & 0x1u;
    unsigned int exp = (bits >> 10) & 0x1Fu;
    unsigned int mant = bits & 0x3FFu;
    float scale;
    if (exp == 0) {
        scale = ldexpf((float)mant, -24);
    } else if (exp == 31) {
        scale = mant ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
    } else {
        scale = ldexpf((float)(mant | 0x400), (int)exp - 25);
    }
    return sign ? -scale : scale;
}
"#;

/// The GEMM body, identical for every quant kind.
///
/// `src0` is the quantized weight matrix, `n_rows` rows of `row_bytes`
/// each. `src1` is `batch` activation rows of `n_cols` f32. `dst` is
/// written as `dst[token * n_rows + row]`, which is the layout
/// `WeightMatrix::apply_batch` already returns.
///
/// Every `FX_*` name is `#define`d by [`kernel_src`] from this module's
/// Rust constants, so the emitted kernel and [`crate::mul_mm_ref`]
/// cannot disagree about geometry.
///
/// The out-of-range row clamp is llama's trick, kept from
/// `mul_mm_sg_impl`: a lane whose row does not exist re-reads the last
/// valid row rather than branching, so the load loop stays uniform, and
/// its result is thrown away by the bounds check at the store. Reading
/// a real row also means the dequant never touches unmapped bytes.
const BODY_SRC: &str = r#"
extern "C" __global__ void FX_FN_NAME(
    const unsigned char* __restrict__ src0,
    const float* __restrict__ src1,
    float* __restrict__ dst,
    int n_rows,
    int n_cols,
    int batch,
    int row_bytes
) {
    __shared__ float sa[FX_BK][FX_BM];
    __shared__ float sb[FX_BK][FX_BN];

    const int r0 = blockIdx.y * FX_BM;
    const int r1 = blockIdx.x * FX_BN;
    const int tid = threadIdx.x;

    // Micro-tile owner: `tx` walks rows, `ty` walks tokens.
    const int tx = tid % (FX_BM / FX_TM);
    const int ty = tid / (FX_BM / FX_TM);

    float acc[FX_TN][FX_TM];
#pragma unroll
    for (int n = 0; n < FX_TN; n++) {
#pragma unroll
        for (int m = 0; m < FX_TM; m++) {
            acc[n][m] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < n_cols; k0 += FX_BK) {
        // Guards the previous iteration's reads of sa/sb.
        __syncthreads();

        // A-tile: one thread decodes one FX_SUB-element sub-block, so
        // FX_BM * (FX_BK / FX_SUB) threads cover the tile. Stored
        // k-major so the K-loop below reads one row of sa per step.
        if (tid < FX_BM * (FX_BK / FX_SUB)) {
            const int lr = tid / (FX_BK / FX_SUB);
            const int ils = tid % (FX_BK / FX_SUB);
            int row = r0 + lr;
            if (row >= n_rows) {
                row = n_rows - 1;
            }
            const unsigned char* rp =
                src0 + (size_t)row * (size_t)row_bytes;
            const int sub = (k0 / FX_SUB) + ils;
            float reg[FX_SUB];
            ferrox_dequant_sub(
                rp + (size_t)(sub / FX_NL) * (size_t)FX_BLOCK_BYTES,
                sub % FX_NL,
                reg);
#pragma unroll
            for (int i = 0; i < FX_SUB; i++) {
                sa[FX_SUB * ils + i][lr] = reg[i];
            }
        }

        // B-tile: consecutive threads read consecutive k of one token.
        // Tokens past the end are zero-filled rather than skipped, so
        // the K-loop needs no per-token predicate.
        for (int idx = tid; idx < FX_BK * FX_BN; idx += FX_THREADS) {
            const int j = idx / FX_BK;
            const int kk = idx % FX_BK;
            const int col = r1 + j;
            sb[kk][j] = (col < batch)
                ? src1[(size_t)col * (size_t)n_cols + (size_t)(k0 + kk)]
                : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < FX_BK; kk++) {
            float a[FX_TM];
            float b[FX_TN];
            // One 16-byte load per four operands, not four 4-byte ones.
            // A warp's 32 lanes take 16 distinct `tx`, so the scalar
            // form had them striding four floats apart across eight
            // banks -- a four-way conflict on the hottest load in the
            // kernel. As `float4` the same 16 lanes read 256 contiguous
            // bytes, which the hardware serves without conflict.
#pragma unroll
            for (int m = 0; m < FX_TM; m += 4) {
                const float4 v = *(const float4*)&sa[kk][tx * FX_TM + m];
                a[m + 0] = v.x;
                a[m + 1] = v.y;
                a[m + 2] = v.z;
                a[m + 3] = v.w;
            }
#pragma unroll
            for (int n = 0; n < FX_TN; n += 4) {
                const float4 v = *(const float4*)&sb[kk][ty * FX_TN + n];
                b[n + 0] = v.x;
                b[n + 1] = v.y;
                b[n + 2] = v.z;
                b[n + 3] = v.w;
            }
#pragma unroll
            for (int n = 0; n < FX_TN; n++) {
#pragma unroll
                for (int m = 0; m < FX_TM; m++) {
                    acc[n][m] += a[m] * b[n];
                }
            }
        }
    }

    for (int n = 0; n < FX_TN; n++) {
        const int col = r1 + ty * FX_TN + n;
        if (col >= batch) {
            continue;
        }
        for (int m = 0; m < FX_TM; m++) {
            const int row = r0 + tx * FX_TM + m;
            if (row < n_rows) {
                dst[(size_t)col * (size_t)n_rows + (size_t)row] = acc[n][m];
            }
        }
    }
}
"#;

/// Emits the complete CUDA C translation unit for one quant kind.
///
/// Deterministic and side-effect free, which is what lets the tests in
/// [`crate::mul_mm_ref`] assert against it without a device.
pub fn kernel_src(kind: &MulMmKind) -> String {
    let defines = format!(
        "#define FX_BM {}\n\
         #define FX_BN {}\n\
         #define FX_BK {}\n\
         #define FX_TM {}\n\
         #define FX_TN {}\n\
         #define FX_THREADS {}\n\
         #define FX_SUB {}\n\
         #define FX_NL {}\n\
         #define FX_BLOCK_BYTES {}\n",
        BM,
        BN,
        BK,
        TM,
        TN,
        THREADS,
        SUB,
        kind.nl(),
        kind.block_bytes,
    );
    let body = BODY_SRC.replace("FX_FN_NAME", kind.fn_name);
    // Q4_K and Q5_K share llama's 6-bit scale/min unpack. It is emitted
    // for every kind rather than conditionally: an unused `__device__`
    // helper costs nothing after NVRTC's dead-code pass, and a
    // per-kind include list is one more table that has to agree with
    // another one.
    format!(
        "{defines}{F16_SRC}{K_SCALE_MIN_SRC}{}{body}",
        kind.dequant_src
    )
}

/// Why a `mul_mm` dispatch was refused. Named rather than silent: a
/// shape this kernel cannot do must fall back to a path that can, and
/// the caller has to be able to say which.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MulMmUnsupported {
    /// `n_cols` is not a multiple of the K-tile. Every real GGUF row
    /// length is a multiple of 32, so this means a synthetic shape.
    ColsNotTileAligned { n_cols: usize, tile: usize },
    /// `n_cols` is not a whole number of super-blocks for this kind.
    ColsNotBlockAligned {
        n_cols: usize,
        block_elems: usize,
        kind: &'static str,
    },
    /// `row_bytes` does not match `n_cols` worth of super-blocks.
    RowBytesMismatch {
        row_bytes: usize,
        expected: usize,
        kind: &'static str,
    },
    /// The weight buffer is not `n_rows * row_bytes`.
    WeightsTooSmall { got: usize, want: usize },
    /// The activation buffer is not `batch * n_cols`.
    ActivationsTooSmall { got: usize, want: usize },
    /// A zero-sized dispatch. Not an error the caller must handle
    /// specially, but not something to launch a grid for either.
    EmptyShape,
}

impl std::fmt::Display for MulMmUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColsNotTileAligned { n_cols, tile } => {
                write!(f, "mul_mm: n_cols {n_cols} is not a multiple of the K-tile {tile}")
            }
            Self::ColsNotBlockAligned {
                n_cols,
                block_elems,
                kind,
            } => write!(
                f,
                "mul_mm: n_cols {n_cols} is not a whole number of {kind} blocks ({block_elems} elems)"
            ),
            Self::RowBytesMismatch {
                row_bytes,
                expected,
                kind,
            } => write!(
                f,
                "mul_mm: row_bytes {row_bytes} does not match {expected} for {kind} at this n_cols"
            ),
            Self::WeightsTooSmall { got, want } => {
                write!(f, "mul_mm: weight buffer is {got} bytes, needs {want}")
            }
            Self::ActivationsTooSmall { got, want } => {
                write!(f, "mul_mm: activation buffer is {got} floats, needs {want}")
            }
            Self::EmptyShape => write!(f, "mul_mm: empty shape"),
        }
    }
}

impl std::error::Error for MulMmUnsupported {}

/// The shape checks the kernel's index arithmetic assumes, in one place
/// so the launch path and the scalar twin cannot check different things.
pub fn validate_shape(
    kind: &MulMmKind,
    weights_len: usize,
    x_len: usize,
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    row_bytes: usize,
) -> Result<(), MulMmUnsupported> {
    if n_rows == 0 || n_cols == 0 || batch == 0 {
        return Err(MulMmUnsupported::EmptyShape);
    }
    if !n_cols.is_multiple_of(BK) {
        return Err(MulMmUnsupported::ColsNotTileAligned { n_cols, tile: BK });
    }
    if !n_cols.is_multiple_of(kind.block_elems) {
        return Err(MulMmUnsupported::ColsNotBlockAligned {
            n_cols,
            block_elems: kind.block_elems,
            kind: kind.name,
        });
    }
    let expected_row_bytes = (n_cols / kind.block_elems) * kind.block_bytes;
    if row_bytes != expected_row_bytes {
        return Err(MulMmUnsupported::RowBytesMismatch {
            row_bytes,
            expected: expected_row_bytes,
            kind: kind.name,
        });
    }
    let want_weights = n_rows * row_bytes;
    if weights_len < want_weights {
        return Err(MulMmUnsupported::WeightsTooSmall {
            got: weights_len,
            want: want_weights,
        });
    }
    let want_x = batch * n_cols;
    if x_len < want_x {
        return Err(MulMmUnsupported::ActivationsTooSmall {
            got: x_len,
            want: want_x,
        });
    }
    Ok(())
}

/// Whether a batched dispatch of this shape is worth a GEMM at all.
///
/// One token is a matvec, and `gpu.rs`'s matvec kernels are the arm that
/// has actually run on hardware; sending a single row through the tile
/// would waste every output column but one. The caller should
/// keep using `apply_gpu` below this threshold.
pub fn worth_a_gemm(batch: usize) -> bool {
    // `.max(2)` so this stays "never a single token" even if the tile
    // width is retuned downward.
    batch >= (BN / 4).max(2)
}

/// Grid dimensions for a dispatch, shared by the launch path and the
/// twin's block loop so they enumerate exactly the same tiles.
pub fn grid_dims(n_rows: usize, batch: usize) -> (usize, usize) {
    (batch.div_ceil(BN), n_rows.div_ceil(BM))
}

#[cfg(test)]
mod dequant_twin_tests {
    use super::*;

    /// Deterministic pseudo-random bytes: a block's contents only have
    /// to be varied and reproducible, not meaningful.
    fn bytes(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// Every twin, against the CPU dequant this project already holds
    /// against llama.cpp.
    ///
    /// This is the only oracle available without a GPU, and it is a real
    /// one: `ferrox_quant::dequant_*` decodes a whole super-block, so
    /// sub-block `il` of the kernel must equal elements `[16*il,
    /// 16*il+16)` of it. A transcription that mixes up llama's three
    /// different uses of `il` produces plausible numbers from the wrong
    /// offsets, and that is exactly what this catches.
    #[test]
    fn every_dequant_twin_matches_the_cpu_dequant() {
        struct Case {
            kind: &'static MulMmKind,
            dequant: fn(&[u8]) -> Result<Vec<f32>, ferrox_quant::QuantError>,
        }
        let cases = [
            Case {
                kind: &Q5_0,
                dequant: ferrox_quant::dequant_q5_0,
            },
            Case {
                kind: &Q4_K,
                dequant: ferrox_quant::dequant_q4_k,
            },
            Case {
                kind: &Q5_K,
                dequant: ferrox_quant::dequant_q5_k,
            },
            Case {
                kind: &Q6_K,
                dequant: ferrox_quant::dequant_q6_k,
            },
        ];

        for case in &cases {
            let k = case.kind;
            for seed in [1u32, 7, 12345] {
                let mut block = bytes(k.block_bytes, seed);
                // Random bytes make a random `half`, and a random half
                // is Inf or NaN often enough to be the usual outcome.
                // The scales get finite values so the comparison is
                // about the UNPACK: 0x2C00 is 0.0625, 0x2800 is
                // 0.03125. Everything else stays random -- for Q5_0
                // that deliberately includes the four `qh` bytes at
                // 2..6, which carry the fifth bit of all 32 quants and
                // are the half of the format a transcription gets
                // wrong.
                let (d_at, dmin_at) = match k.name {
                    "Q6_K" => (208, None),
                    // Q5_0 has no `dmin`; bytes 2..6 are `qh`, and
                    // pinning them would blind the fifth-bit check.
                    "Q5_0" => (0, None),
                    _ => (0, Some(2)),
                };
                block[d_at] = 0x00;
                block[d_at + 1] = 0x2C;
                if let Some(at) = dmin_at {
                    block[at] = 0x00;
                    block[at + 1] = 0x28;
                }
                let block = block;
                let want = (case.dequant)(&block).expect("cpu dequant");
                assert_eq!(want.len(), k.block_elems, "{} block size", k.name);

                for il in 0..k.nl() {
                    let mut reg = [0f32; SUB];
                    (k.dequant_twin)(&block, il, &mut reg);
                    for (j, got) in reg.iter().enumerate() {
                        let expect = want[SUB * il + j];
                        let tol = expect.abs().max(1.0) * 1e-5;
                        assert!(
                            (got - expect).abs() <= tol,
                            "{} seed {seed} sub-block {il} element {j}: \
                             kernel twin {got} vs cpu dequant {expect}",
                            k.name
                        );
                    }
                }
            }
        }
    }

    /// The block geometry each kind declares has to be the geometry the
    /// format actually has, or the GEMM walks the row with the wrong
    /// stride and every number after the first block is garbage.
    ///
    /// `nl` is part of that geometry and is NOT the same for every row:
    /// the K-quants are 256-element super-blocks (`nl` 16) and Q5_0 is a
    /// 32-element legacy block (`nl` 2). A Q5_0 row that inherited the
    /// K-quant assumption would ask the kernel for sub-blocks 2..16 of a
    /// block that has two, so the expectation is per row rather than one
    /// shared constant.
    #[test]
    fn declared_block_geometry_is_the_gguf_geometry() {
        for (k, bytes_, elems, nl) in [
            (
                &Q5_0,
                ferrox_quant::Q5_0_BLOCK_BYTES,
                ferrox_quant::Q5_0_BLOCK_ELEMS,
                2,
            ),
            (&Q4_K, 144, 256, 16),
            (&Q5_K, 176, 256, 16),
            (&Q6_K, 210, 256, 16),
        ] {
            assert_eq!(k.block_bytes, bytes_, "{} block_bytes", k.name);
            assert_eq!(k.block_elems, elems, "{} block_elems", k.name);
            assert_eq!(k.nl(), nl, "{} sub-blocks per super-block", k.name);
        }
    }
}
