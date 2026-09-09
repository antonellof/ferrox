//! The codebook matvec kernels: IQ4_NL, IQ4_XS and MXFP4.
//!
//! Same one-block-per-row, 256-thread, tree-reduction shape as
//! [`super::legacy`] and [`super::kquant`]; the only thing that differs
//! is the unpack, which here is a `__constant__` table lookup rather
//! than an affine transform. See
//! [`crate::mul_mm_kinds::codebook`](crate::mul_mm_kinds::codebook) for
//! the formats themselves and the llama.cpp citations.
//!
//! # The codebook is written out twice, and a test says so
//!
//! The GEMM emits its `__constant__` array from the `Codebook` row, so
//! there the device table and the host table are one slice. A matvec
//! kernel is a `&'static str` handed to NVRTC verbatim, so its sixteen
//! values are a literal. That is a second structure that must agree
//! with the first, which is this repo's dominant bug shape -- so
//! `every_embedded_codebook_is_the_mul_mm_codebook` parses the numbers
//! back out of each source below and holds them to
//! [`crate::mul_mm::Codebook::values`], bit for bit. Changing one
//! without the other fails the suite rather than decoding every tensor
//! slightly wrong.
//!
//! # UNVERIFIED ON HARDWARE
//!
//! No GPU has run any of these, and unlike the GEMM there is no host
//! harness for a matvec kernel: `tools/mul_mm_host_check` executes
//! `mul_mm`'s emitted C, not this. The checks that exist are the
//! codebook test below, the `#[ignore]`d hardware tests in `gpu.rs`,
//! and the fact that the unpack is the same arithmetic the `mul_mm`
//! twins are held to against `ferrox_quant`. Run
//! `cargo test -p ferrox-cuda --features cuda -- --ignored` on a device
//! before any doc calls these measured.

/// Fused IQ4_NL dequant+dot. 18-byte blocks: `half d`, then 16 bytes of
/// 4-bit codes, low nibble of byte `j` giving element `j` and the high
/// nibble element `j + 16`.
///
/// Mirrors `ferrox_quant::dot_iq4_nl_f32_scalar` except that the scale
/// is factored out of the inner loop and applied once per block, which
/// is what `Q4_0_MATVEC_KERNEL_SRC` already does and differs only by
/// fp32 rounding.
pub const IQ4_NL_MATVEC_KERNEL_SRC: &str = r#"
__constant__ float ferrox_kvalues_iq4nl[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
    1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f
};

extern "C" __global__ void iq4_nl_matvec(
    const unsigned char* weights, // [rows * row_bytes]
    const float* x,               // [cols]
    float* out,                   // [rows]
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;

    __shared__ float partial[256];
    float acc = 0.0f;

    for (int b = threadIdx.x; b < n_blocks_per_row; b += blockDim.x) {
        const unsigned char* block = row_ptr + (size_t)b * 18;
        unsigned short bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
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
        if (sign) scale = -scale;

        int base = b * 32;
        float block_acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned char byte = block[2 + i];
            block_acc += ferrox_kvalues_iq4nl[byte & 0x0F] * x[base + i];
            block_acc += ferrox_kvalues_iq4nl[byte >> 4] * x[base + i + 16];
        }
        acc += block_acc * scale;
    }

    partial[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[row] = partial[0];
    }
}
"#;

/// Fused IQ4_XS dequant+dot. 136-byte super-blocks of 256 elements:
/// `half d`, `uint16 scales_h`, `uint8 scales_l[4]`, then 128 bytes of
/// 4-bit codes as eight 32-element groups.
///
/// Each group's 6-bit scale is assembled from a nibble of `scales_l`
/// and a 2-bit field of `scales_h`, then biased by -32 -- the same
/// three derivations of `ib` the GEMM's `dequant_src` performs, and the
/// part of this format a transcription gets wrong. Mirrors
/// `ferrox_quant::dot_iq4_xs_f32_scalar`, with the per-group scale
/// factored out of the inner loop.
pub const IQ4_XS_MATVEC_KERNEL_SRC: &str = r#"
__constant__ float ferrox_kvalues_iq4nl[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
    1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f
};

extern "C" __global__ void iq4_xs_matvec(
    const unsigned char* weights, // [rows * row_bytes]
    const float* x,               // [cols]
    float* out,                   // [rows]
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;

    __shared__ float partial[256];
    float acc = 0.0f;

    for (int b = threadIdx.x; b < n_blocks_per_row; b += blockDim.x) {
        const unsigned char* block = row_ptr + (size_t)b * 136;
        unsigned short bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned int sign = (bits >> 15) & 0x1u;
        unsigned int exp = (bits >> 10) & 0x1Fu;
        unsigned int mant = bits & 0x3FFu;
        float d;
        if (exp == 0) {
            d = ldexpf((float)mant, -24);
        } else if (exp == 31) {
            d = mant ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
        } else {
            d = ldexpf((float)(mant | 0x400), (int)exp - 25);
        }
        if (sign) d = -d;

        unsigned int scales_h = (unsigned int)block[2] | ((unsigned int)block[3] << 8);
        const unsigned char* scales_l = block + 4;
        const unsigned char* qs = block + 8;
        int base = b * 256;

        for (int ib = 0; ib < 8; ib++) {
            unsigned int ls =
                ((unsigned int)(scales_l[ib / 2] >> (4 * (ib & 1))) & 0xFu)
                | (((scales_h >> (2 * ib)) & 3u) << 4);
            float dl = d * ((float)ls - 32.0f);
            const unsigned char* sub = qs + 16 * ib;
            int gbase = base + 32 * ib;
            float group_acc = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                unsigned char byte = sub[i];
                group_acc += ferrox_kvalues_iq4nl[byte & 0x0F] * x[gbase + i];
                group_acc += ferrox_kvalues_iq4nl[byte >> 4] * x[gbase + i + 16];
            }
            acc += group_acc * dl;
        }
    }

    partial[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[row] = partial[0];
    }
}
"#;

/// Fused MXFP4 dequant+dot, GGUF block form (ggml type tag 39).
/// 17-byte blocks: one E8M0 scale byte, then 16 bytes of 4-bit E2M1
/// codes packed the way IQ4_NL packs its block.
///
/// The codebook holds the REAL E2M1 values against the full
/// `2^(e-127)` scale, following `ferrox_quant`; ggml stores the values
/// doubled against a halved scale and the products are identical. Both
/// conventions must not be mixed, which is why the scale helper is
/// spelled out here rather than reused from an f16 kernel.
///
/// Mirrors `ferrox_quant::dot_mxfp4_gguf_f32_scalar`, with the scale
/// factored out of the inner loop.
pub const MXFP4_MATVEC_KERNEL_SRC: &str = r#"
__constant__ float ferrox_kvalues_mxfp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void mxfp4_matvec(
    const unsigned char* weights, // [rows * row_bytes]
    const float* x,               // [cols]
    float* out,                   // [rows]
    int rows,
    int row_bytes,
    int n_blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const unsigned char* row_ptr = weights + (size_t)row * row_bytes;

    __shared__ float partial[256];
    float acc = 0.0f;

    for (int b = threadIdx.x; b < n_blocks_per_row; b += blockDim.x) {
        const unsigned char* block = row_ptr + (size_t)b * 17;
        // An E8M0 byte IS an f32 exponent field (bias 127), so placing
        // it there is exact. `e == 0` means 2^-127, which the shift
        // alone would give as 0.0; `e == 255` is reserved for NaN by
        // the OCP spec and is not handled, matching ggml.
        unsigned char e = block[0];
        float scale = (e == 0) ? __int_as_float(0x00400000)
                               : __int_as_float((int)((unsigned int)e << 23));

        int base = b * 32;
        float block_acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned char byte = block[1 + i];
            block_acc += ferrox_kvalues_mxfp4[byte & 0x0F] * x[base + i];
            block_acc += ferrox_kvalues_mxfp4[byte >> 4] * x[base + i + 16];
        }
        acc += block_acc * scale;
    }

    partial[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[row] = partial[0];
    }
}
"#;
