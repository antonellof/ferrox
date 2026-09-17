//! `mul_mm` on the tensor cores: the same per-kind dequantization as
//! [`crate::mul_mm`], a different inner product.
//!
//! The SIMT body in `mul_mm.rs` reached about 7 TFLOPS on an RTX 3090
//! (`q4_k_mul_mm`, 1.9 ms per Llama-3.2-3B projection at pp512), which
//! after the resident prefill stack (#259) was three quarters of the
//! GPU time of a prefill step. llama.cpp's `mmq` runs the same product
//! on the tensor cores as int8 `mma`. This body takes the middle road
//! that reuses every unpack function in the table unchanged: each tile
//! of weights is dequantized into shared memory as f16, the activation
//! tile is converted to f16 on the way in, and the product is
//! `mma.sync.m16n8k16` with f32 accumulation. f16 keeps eleven bits of
//! each operand where llama.cpp's `q8_1` activations keep eight, so
//! the result is closer to the f32 twin than upstream's is to its own.
//!
//! `mma.sync.m16n8k16.f16` needs `sm_80`, so the launch asks the
//! device for its compute capability and a pre-Ampere card keeps the
//! SIMT body; nothing here replaces it.
//!
//! Fragment layouts are the PTX ISA's for `m16n8k16` with `.f16`
//! operands and `.f32` accumulators, written out in `TC_BODY_SRC`
//! beside the loads that honour them: lane `l` has `g = l / 4` and
//! `t = l % 4`; A holds rows `g` and `g + 8` at k `2t, 2t+1` and
//! `2t+8, 2t+9`; B holds token `g` at the same k; C holds rows `g`,
//! `g + 8` at tokens `2t, 2t+1`. Both shared tiles are stored
//! k-contiguous with a row stride of 40 halves, which is what makes
//! the eight rows a fragment load touches land on distinct banks.

use crate::mul_mm::{kernel_src_with_body, MulMmKind};

/// Weight rows per block tile.
pub const TBM: usize = 128;
/// Tokens per block tile.
pub const TBN: usize = 128;
/// K elements per tile step: two `k16` MMA steps.
pub const TBK: usize = 32;
/// Warps along M and N; eight warps, 256 threads.
pub const WM: usize = 4;
pub const WN: usize = 2;
pub const TTHREADS: usize = 32 * WM * WN;
/// Padded shared row stride in halves. `TBK + 8`: 80 bytes, so the
/// eight rows a fragment touches start 20 words apart and cover eight
/// distinct bank groups.
pub const LDS: usize = TBK + 8;

const _: () = assert!(
    TBM / WM == 32 && TBN / WN == 64,
    "the warp tile is 32 rows x 64 tokens"
);
const _: () = assert!(
    TBM * (TBK / crate::mul_mm::SUB) == TTHREADS,
    "one sub-block per thread per step"
);
const _: () = assert!(
    TBN * 2 == TTHREADS,
    "two threads per token row of the B tile"
);
const _: () = assert!(TBK == 32, "the B loader converts sixteen floats per thread");
const _: () = assert!((LDS * 2).is_multiple_of(16), "shared rows stay 16-byte aligned");

/// The compute capability the body needs: `mma.sync.m16n8k16` with
/// f16 operands is `sm_80` and up.
pub const MIN_CC_MAJOR: i32 = 8;
/// The NVRTC target for the body.
pub const ARCH: &str = "compute_80";

/// The kernel body. `FX_*` geometry names are `#define`d by
/// [`tc_kernel_src`]; the dequant helpers come from the kind's row
/// exactly as for the SIMT body.
pub const TC_BODY_SRC: &str = r#"
__device__ __forceinline__ unsigned int ferrox_pack2h(float lo, float hi) {
    unsigned int r;
    asm("{ .reg .f16 l, h; cvt.rn.f16.f32 l, %1; cvt.rn.f16.f32 h, %2; mov.b32 %0, {l, h}; }"
        : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

__device__ __forceinline__ void ferrox_mma16816(float* c, const unsigned int* a, const unsigned int* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

extern "C" __global__ void FX_FN_NAME(
    const unsigned char* __restrict__ src0,
    const float* __restrict__ src1,
    float* __restrict__ dst,
    int n_rows,
    int n_cols,
    int batch,
    int row_bytes
) {
    __shared__ __align__(16) unsigned short sa[FX_TBM * FX_LDS];
    __shared__ __align__(16) unsigned short sb[FX_TBN * FX_LDS];

    const int r0 = blockIdx.y * FX_TBM;
    const int c0 = blockIdx.x * FX_TBN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp % FX_WM;
    const int wn = warp / FX_WM;
    const int g = lane >> 2;
    const int tg = lane & 3;

    float acc[FX_MF][FX_NF][4];
#pragma unroll
    for (int mf = 0; mf < FX_MF; mf++) {
#pragma unroll
        for (int nf = 0; nf < FX_NF; nf++) {
            acc[mf][nf][0] = 0.f; acc[mf][nf][1] = 0.f;
            acc[mf][nf][2] = 0.f; acc[mf][nf][3] = 0.f;
        }
    }

    for (int k0 = 0; k0 < n_cols; k0 += FX_TBK) {
        __syncthreads();

        // A tile: one FX_SUB-element sub-block per thread, dequantized
        // to f32 by the kind's own function and packed to f16 pairs.
        // Out-of-range rows re-read the last valid row (llama's clamp)
        // and are dropped at the store.
        {
            const int lr = tid >> 1;
            const int ils = tid & 1;
            int row = r0 + lr;
            if (row >= n_rows) row = n_rows - 1;
            const unsigned char* rp = src0 + (size_t)row * (size_t)row_bytes;
            const int sub = (k0 / FX_SUB) + ils;
            float reg[FX_SUB];
            ferrox_dequant_sub(
                rp + (size_t)(sub / FX_NL) * (size_t)FX_BLOCK_BYTES,
                sub % FX_NL,
                reg);
            unsigned int* dp = (unsigned int*)&sa[lr * FX_LDS + ils * FX_SUB];
#pragma unroll
            for (int i = 0; i < FX_SUB; i += 2) {
                dp[i / 2] = ferrox_pack2h(reg[i], reg[i + 1]);
            }
        }

        // B tile: two threads per token, sixteen k each, read as
        // float4 (n_cols is a multiple of 32 and the activation buffer
        // is a whole allocation, so the offset is 16-byte aligned) and
        // packed to f16 pairs. Tokens past the batch are zero.
        {
            const int j = tid >> 1;
            const int kh = (tid & 1) * 16;
            const int col = c0 + j;
            unsigned int* dp = (unsigned int*)&sb[j * FX_LDS + kh];
            if (col < batch) {
                const float* sp = src1 + (size_t)col * (size_t)n_cols + (size_t)(k0 + kh);
#pragma unroll
                for (int i = 0; i < 16; i += 4) {
                    const float4 v = *(const float4*)(sp + i);
                    dp[i / 2] = ferrox_pack2h(v.x, v.y);
                    dp[i / 2 + 1] = ferrox_pack2h(v.z, v.w);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) dp[i] = 0u;
            }
        }

        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < FX_TBK; ks += 16) {
            unsigned int af[FX_MF][4];
#pragma unroll
            for (int mf = 0; mf < FX_MF; mf++) {
                const int rbase = wm * FX_WARP_M + mf * 16;
                const unsigned short* p0 = &sa[(rbase + g) * FX_LDS + ks + tg * 2];
                const unsigned short* p1 = &sa[(rbase + g + 8) * FX_LDS + ks + tg * 2];
                af[mf][0] = *(const unsigned int*)p0;
                af[mf][1] = *(const unsigned int*)p1;
                af[mf][2] = *(const unsigned int*)(p0 + 8);
                af[mf][3] = *(const unsigned int*)(p1 + 8);
            }
#pragma unroll
            for (int nf = 0; nf < FX_NF; nf++) {
                const int nbase = wn * FX_WARP_N + nf * 8;
                const unsigned short* p = &sb[(nbase + g) * FX_LDS + ks + tg * 2];
                unsigned int bf[2];
                bf[0] = *(const unsigned int*)p;
                bf[1] = *(const unsigned int*)(p + 8);
#pragma unroll
                for (int mf = 0; mf < FX_MF; mf++) {
                    ferrox_mma16816(acc[mf][nf], af[mf], bf);
                }
            }
        }
    }

    // C fragments: rows g and g + 8, tokens 2t and 2t + 1.
#pragma unroll
    for (int mf = 0; mf < FX_MF; mf++) {
        const int row0 = r0 + wm * FX_WARP_M + mf * 16 + g;
        const int row1 = row0 + 8;
#pragma unroll
        for (int nf = 0; nf < FX_NF; nf++) {
            const int col0 = c0 + wn * FX_WARP_N + nf * 8 + tg * 2;
            const int col1 = col0 + 1;
            if (col0 < batch) {
                if (row0 < n_rows) dst[(size_t)col0 * (size_t)n_rows + (size_t)row0] = acc[mf][nf][0];
                if (row1 < n_rows) dst[(size_t)col0 * (size_t)n_rows + (size_t)row1] = acc[mf][nf][2];
            }
            if (col1 < batch) {
                if (row0 < n_rows) dst[(size_t)col1 * (size_t)n_rows + (size_t)row0] = acc[mf][nf][1];
                if (row1 < n_rows) dst[(size_t)col1 * (size_t)n_rows + (size_t)row1] = acc[mf][nf][3];
            }
        }
    }
}
"#;

/// The complete translation unit for one kind's tensor-core GEMM: the
/// SIMT unit's preamble (f16 decode, K-scale unpack, codebook, the
/// kind's `ferrox_dequant_sub`) with this body in place of the SIMT
/// one, under the names [`tc_names`] gives the kind.
pub fn tc_kernel_src(kind: &MulMmKind) -> String {
    let defines = format!(
        "#define FX_TBM {TBM}\n\
         #define FX_TBN {TBN}\n\
         #define FX_TBK {TBK}\n\
         #define FX_WM {WM}\n\
         #define FX_WN {WN}\n\
         #define FX_WARP_M {}\n\
         #define FX_WARP_N {}\n\
         #define FX_MF {}\n\
         #define FX_NF {}\n\
         #define FX_LDS {LDS}\n",
        TBM / WM,
        TBN / WN,
        TBM / WM / 16,
        TBN / WN / 8,
    );
    let (_, fn_name) = tc_names(kind);
    let body = TC_BODY_SRC.replace("FX_FN_NAME", fn_name);
    format!("{defines}{}", kernel_src_with_body(kind, &body))
}

/// The NVRTC module and function names for a kind's tensor-core
/// kernel: the SIMT names with a `_tc` suffix, leaked once so the
/// load-once cache can key on `&'static str` as it does for every
/// other kernel.
pub fn tc_names(kind: &MulMmKind) -> (&'static str, &'static str) {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static NAMES: OnceLock<Mutex<HashMap<&'static str, (&'static str, &'static str)>>> =
        OnceLock::new();
    let mut map = NAMES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    *map.entry(kind.name).or_insert_with(|| {
        (
            Box::leak(format!("{}_tc", kind.module_name).into_boxed_str()),
            Box::leak(format!("{}_tc", kind.fn_name).into_boxed_str()),
        )
    })
}

/// Grid for a launch: tokens along x, rows along y, as the SIMT body.
pub fn tc_grid_dims(n_rows: usize, batch: usize) -> (usize, usize) {
    (batch.div_ceil(TBN), n_rows.div_ceil(TBM))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mul_mm::KINDS;

    #[test]
    fn every_kind_emits_a_tensor_core_unit_that_names_its_dequant() {
        for kind in KINDS {
            let src = tc_kernel_src(kind);
            let (module, fn_name) = tc_names(kind);
            assert!(module.ends_with("_tc") && fn_name.ends_with("_tc"));
            assert!(
                src.contains(&format!("__global__ void {fn_name}(")),
                "{}",
                kind.name
            );
            assert!(src.contains("ferrox_dequant_sub("), "{}", kind.name);
            assert!(src.contains("mma.sync.aligned.m16n8k16"), "{}", kind.name);
            assert!(
                src.contains("#define FX_SUB 16\n"),
                "{}: the sub-block width the loader assumes",
                kind.name
            );
            // No stale SIMT geometry name survives into this unit's body.
            assert!(!src.contains("FX_TM"), "{}", kind.name);
        }
    }

    #[test]
    fn the_names_are_stable_across_calls() {
        let a = tc_names(&crate::mul_mm::Q8_0);
        let b = tc_names(&crate::mul_mm::Q8_0);
        assert!(std::ptr::eq(a.0, b.0) && std::ptr::eq(a.1, b.1));
    }
}
