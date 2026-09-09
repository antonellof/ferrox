//! The 32-element legacy matvec kernels: Q8_0, Q4_0 and Q5_0.
//!
//! Moved here verbatim from `gpu.rs`, which held 2,481 lines of device
//! plumbing and 800 lines of kernel text in one file. The plumbing and
//! the formats are different concepts; the formats are the half that
//! grows every time a quantization is added.
//!
//! Each source is a `__global__` entry point with the signature
//! [`crate::matvec_kinds::MatvecKind`] documents: one threadblock per
//! output row, 256 threads striding the row's blocks, a tree reduction
//! into `out[row]`.

/// CUDA C source for a fused Q8_0 dequant+dot kernel: one thread block
/// per output row, each thread handling a subset of the row's Q8_0
/// blocks, block-level reduction into the row's output element. This
/// mirrors `ferrox_quant::dot_q8_0_f32_scalar`'s math exactly (same
/// block layout: 2-byte f16 scale + 32 int8 values per 34-byte block).
///
/// Verified: compiled by NVRTC and executed on a real GPU (RTX 3060),
/// matching the CPU reference exactly -- see the module doc comment.
pub const Q8_0_MATVEC_KERNEL_SRC: &str = r#"
extern "C" __global__ void q8_0_matvec(
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
        const unsigned char* block = row_ptr + b * 34;
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
        for (int i = 0; i < 32; i++) {
            signed char q = (signed char)block[2 + i];
            block_acc += (float)q * x[base + i];
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

/// CUDA C source for a fused Q4_0 dequant+dot kernel, the same
/// one-block-per-row / block-level-reduction structure as
/// `Q8_0_MATVEC_KERNEL_SRC` above, but unpacking Q4_0's 18-byte blocks
/// (2-byte f16 scale + 16 bytes of packed 4-bit nibbles, low nibble =
/// element `i`, high nibble = element `i+16`, both biased by -8) to
/// mirror `ferrox_quant::dot_q4_0_f32_scalar`'s exact math.
///
/// Verified: compiled by NVRTC and executed on a real GPU (RTX 3060),
/// matching the CPU reference exactly -- see the module doc comment.
pub const Q4_0_MATVEC_KERNEL_SRC: &str = r#"
extern "C" __global__ void q4_0_matvec(
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
        const unsigned char* block = row_ptr + b * 18;
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
            int lo = (int)(byte & 0x0F) - 8;
            int hi = (int)((byte >> 4) & 0x0F) - 8;
            block_acc += (float)lo * x[base + i];
            block_acc += (float)hi * x[base + i + 16];
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

/// CUDA C source for a fused Q5_0 dequant+dot kernel, the same
/// one-block-per-lane / block-level-reduction structure as
/// `Q4_0_MATVEC_KERNEL_SRC` above, but unpacking Q5_0's 22-byte blocks:
/// 2-byte f16 scale, a 4-byte `qh` bitplane, then 16 bytes of packed
/// 4-bit nibbles. Element `j` takes the low nibble of `qs[j]` with bit
/// `j` of `qh` as its fifth bit; element `j + 16` takes the high nibble
/// with bit `j + 16`. Both are biased by -16. This mirrors
/// `ferrox_quant::dot_q5_0_f32_scalar`'s exact math, and is `ggml`'s
/// `dequantize_row_q5_0` reference form rather than the nibble-packed
/// `ushort` trick `Q4_0_MATVEC_KERNEL_SRC` uses -- the same choice
/// `ferrox-metal`'s `Q5_0_MATVEC_KERNEL_SRC` made and for the same
/// reason: the fifth bit is indexed differently in the two halves, so
/// folding it into the activation scaling needs two more shift chains
/// and is much easier to get subtly wrong.
///
/// **UNVERIFIED ON HARDWARE.** No GPU has run this. Unlike the GEMM in
/// `mul_mm.rs`, whose emitted C is executed on the host by
/// `tools/mul_mm_host_check/run.sh` and compared bit for bit against a
/// Rust twin, there is no host harness for the matvec kernels: the only
/// check on this text is `launch_q5_0_matvec_matches_cpu_reference`,
/// which is `#[ignore]`d and needs a device. Run it with
/// `cargo test -p ferrox-cuda --features cuda -- --ignored` before any
/// doc calls Q5_0 a measured CUDA capability.
pub const Q5_0_MATVEC_KERNEL_SRC: &str = r#"
extern "C" __global__ void q5_0_matvec(
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
        const unsigned char* block = row_ptr + (size_t)b * 22;
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

        const unsigned int qh = (unsigned int)block[2]
            | ((unsigned int)block[3] << 8)
            | ((unsigned int)block[4] << 16)
            | ((unsigned int)block[5] << 24);
        const unsigned char* qs = block + 6;

        int base = b * 32;
        float block_acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            // ggml `dequantize_row_q5_0`: the low half takes bit `j` of
            // qh shifted UP into position 4, the high half takes bit
            // `j + 16` shifted DOWN into it -- hence `j + 12`, not
            // `j + 16`, because the bit is left in place rather than
            // moved to position 0.
            unsigned int xh_0 = ((qh >> j) << 4) & 0x10u;
            unsigned int xh_1 = (qh >> (j + 12)) & 0x10u;
            int x0 = (int)(((unsigned int)qs[j] & 0x0Fu) | xh_0) - 16;
            int x1 = (int)(((unsigned int)qs[j] >> 4) | xh_1) - 16;
            block_acc += (float)x0 * x[base + j];
            block_acc += (float)x1 * x[base + j + 16];
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
