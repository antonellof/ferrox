//! Pieces more than one interleaved kind family needs: the f16 read,
//! the 6-bit K-quant scale/min decode, the `×4` kernel choice, and the
//! pre-interleaved activation quads the batch GEMMs consume.

use crate::{Q8Activations, Q8KActivations, Q4_K_BLOCK_ELEMS, Q8_0_BLOCK_ELEMS};
use half::f16;

pub(crate) const KMASK1: u32 = 0x3f3f_3f3f;
pub(crate) const KMASK2: u32 = 0x0f0f_0f0f;
pub(crate) const KMASK3: u32 = 0x0303_0303;

#[inline]
pub(crate) fn f16_from_bytes(b: &[u8]) -> f32 {
    f16::from_le_bytes([b[0], b[1]]).to_f32()
}

/// Which `×4` GEMM kernel this host runs, resolved once instead of once
/// per call.
///
/// The `gemm_*_group_x4` entry points are called once per (row-group ×
/// activation-quad) pair, which on a `pp512` projection is 10^4 to 10^5
/// calls per GEMM. Each one used to re-run `is_aarch64_feature_detected!`,
/// whose relaxed atomic load LLVM cannot hoist out of the caller's loop.
/// Callers now probe once per matmul and pass the answer down through the
/// `_on` variants; [`gemm_q4_kx8_group_x4`] and its siblings stay as
/// probe-per-call wrappers so existing callers and tests are unchanged.
///
/// This is a dispatch decision only. Both arms compute the same values,
/// bit-identically, which is what the `*_x4_portable_is_bit_exact_vs_scalar_gemv`
/// tests assert. Forcing [`AccelX4::Portable`] on an i8mm host is therefore
/// a valid (slow) way to run, and the tests use it that way.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AccelX4 {
    /// ARM i8mm `SMMLA` kernels in [`neon`].
    NeonI8mm,
    /// The portable scalar reference.
    Portable,
}

impl AccelX4 {
    /// The fastest kernel available on this host.
    #[inline]
    pub fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("i8mm") {
                return AccelX4::NeonI8mm;
            }
        }
        AccelX4::Portable
    }
}

/// Decode one 12-byte packed scale/min group into 8 scales + 8 mins (u8).
#[inline]
pub(crate) fn decode_scales_mins(
    scales12: &[u8],
    scales_out: &mut [u8; 8],
    mins_out: &mut [u8; 8],
) {
    debug_assert!(scales12.len() >= 12);
    let mut utmp = [0u32; 4];
    utmp[0] = u32::from_le_bytes(scales12[0..4].try_into().unwrap());
    utmp[1] = u32::from_le_bytes(scales12[4..8].try_into().unwrap());
    utmp[2] = u32::from_le_bytes(scales12[8..12].try_into().unwrap());
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux_0 = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux_0;
    utmp[0] &= KMASK1;
    let bytes = unsafe { std::slice::from_raw_parts(utmp.as_ptr() as *const u8, 16) };
    scales_out.copy_from_slice(&bytes[0..8]);
    mins_out.copy_from_slice(&bytes[8..16]);
}

/// A quad of up to [`Q8K_ACTS_X4_NC`] Q8_0 activations, pre-interleaved
/// into the layout llama.cpp's `ggml_quantize_mat_q8_0_4x8` writes into
/// `block_q8_0x4` (`arch/arm/repack.cpp`): every 32-element block's qs in
/// 8-byte runs, plus the per-block per-row scales. Consumed by the i8mm
/// `4x8` GEMMs; prepared once per matmul, same hoist as [`Q8KActsX4`].
pub struct Q8ActsX4 {
    /// Real activations in the quad (≤ 4); rows `na..4` are zero padding.
    pub na: usize,
    /// Q8_0 blocks per activation (`n_cols / 32`).
    pub n_blocks: usize,
    /// Interleaved quants, `n_blocks * 128` long. Block `b`, 8-element run
    /// `c`, quad row `a`, lane `k` ↦
    /// `qs[b*128 + c*32 + a*8 + k] = acts[a].q[b*32 + c*8 + k]`.
    pub qs: Vec<i8>,
    /// Activation scales, `n_blocks * 4` long: `d[b*4 + a] = acts[a].d[b]`.
    pub d: Vec<f32>,
}

/// Interleave a quad of Q8_0 activations for the `4x8` i8mm GEMMs
/// (llama.cpp `ggml_quantize_mat_q8_0_4x8`, minus the quantization we
/// already did). Zero-pads when `acts.len() < 4`. Available on every
/// target so the portable GEMMs — and the tests pinning the NEON kernels
/// to them — run anywhere.
pub fn prepare_q8_acts_x4(acts: &[Q8Activations], n_cols: usize) -> Q8ActsX4 {
    assert!(acts.len() <= Q8K_ACTS_X4_NC);
    assert!(n_cols.is_multiple_of(Q8_0_BLOCK_ELEMS));
    let na = acts.len();
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    let mut qs = vec![0i8; nb * Q8_0_BLOCK_ELEMS * 4];
    let mut d = vec![0f32; nb * 4];
    for (a, act) in acts.iter().enumerate() {
        debug_assert_eq!(act.d.len(), nb);
        for b in 0..nb {
            let src = &act.q[b * Q8_0_BLOCK_ELEMS..(b + 1) * Q8_0_BLOCK_ELEMS];
            let dst = &mut qs[b * Q8_0_BLOCK_ELEMS * 4..(b + 1) * Q8_0_BLOCK_ELEMS * 4];
            for (c, run) in src.as_chunks::<8>().0.iter().enumerate() {
                dst[c * 32 + a * 8..c * 32 + a * 8 + 8].copy_from_slice(run);
            }
            d[b * 4 + a] = act.d[b];
        }
    }
    Q8ActsX4 {
        na,
        n_blocks: nb,
        qs,
        d,
    }
}

/// A quad of up to [`Q8K_ACTS_X4_NC`] Q8_K activations, pre-interleaved into
/// the layout llama.cpp's `ggml_quantize_mat_q8_K_4x8` writes into
/// `block_q8_Kx4` (`ggml-cpu/repack.cpp`): every super-block's qs, the folded
/// `bsums` pairs, and the per-block per-row scales.
///
/// The i8mm GEMM consumes activations in this shape. Interleaving them once
/// per matmul — instead of once per (row-group, block) inside the kernel —
/// is the point: the old in-kernel repack was a scalar pass over
/// `rows/8 · batch · cols` bytes with a div and a mod per element, roughly
/// 4× the instruction count of the `vmmlaq_s32` math it fed.
/// Activations per [`Q8KActsX4`] quad (llama.cpp's `q8_k_blocklen`).
pub const Q8K_ACTS_X4_NC: usize = 4;

pub struct Q8KActsX4 {
    /// Real activations in the quad (≤ 4); rows `na..4` are zero padding.
    pub na: usize,
    /// Q8_K super-blocks per activation (`n_cols / 256`).
    pub n_blocks: usize,
    /// Interleaved quants, `n_blocks * 1024` long. Block `b`, 8-element run
    /// `c`, quad row `a`, lane `k` ↦
    /// `qs[b*1024 + c*32 + a*8 + k] = acts[a].q[b*256 + c*8 + k]`.
    pub qs: Vec<i8>,
    /// Folded `bsums` pairs, `n_blocks * 4 * 8` long:
    /// `bsums[(b*4 + a)*8 + i] = acts[a].bsums[b*16 + 2i] + acts[a].bsums[b*16 + 2i + 1]`.
    pub bsums: Vec<i16>,
    /// Activation scales, `n_blocks * 4` long: `d[b*4 + a] = acts[a].d[b]`.
    pub d: Vec<f32>,
}

/// Interleave a quad of activations for [`gemm_q4_kx8_group_x4`]
/// (llama.cpp `ggml_quantize_mat_q8_K_4x8`, minus the quantization we
/// already did). Zero-pads when `acts.len() < 4`, matching what the kernel's
/// in-loop repack used to emit. Available on every target so the portable
/// GEMM below — and the tests pinning the NEON kernel to it — run anywhere.
pub fn prepare_q8_k_acts_x4(acts: &[Q8KActivations], n_cols: usize) -> Q8KActsX4 {
    assert!(acts.len() <= Q8K_ACTS_X4_NC);
    assert!(n_cols.is_multiple_of(Q4_K_BLOCK_ELEMS));
    let na = acts.len();
    let nb = n_cols / Q4_K_BLOCK_ELEMS;
    let mut qs = vec![0i8; nb * Q4_K_BLOCK_ELEMS * 4];
    let mut bsums = vec![0i16; nb * 4 * 8];
    let mut d = vec![0f32; nb * 4];
    for (a, act) in acts.iter().enumerate() {
        debug_assert_eq!(act.n_blocks(), nb);
        for b in 0..nb {
            let src = &act.q[b * Q4_K_BLOCK_ELEMS..(b + 1) * Q4_K_BLOCK_ELEMS];
            let dst = &mut qs[b * Q4_K_BLOCK_ELEMS * 4..(b + 1) * Q4_K_BLOCK_ELEMS * 4];
            for (c, run) in src.as_chunks::<8>().0.iter().enumerate() {
                dst[c * 32 + a * 8..c * 32 + a * 8 + 8].copy_from_slice(run);
            }
            let src_bs = &act.bsums[b * 16..(b + 1) * 16];
            let dst_bs = &mut bsums[(b * 4 + a) * 8..(b * 4 + a) * 8 + 8];
            for (slot, pair) in dst_bs.iter_mut().zip(src_bs.as_chunks::<2>().0) {
                *slot = pair[0] + pair[1];
            }
            d[b * 4 + a] = act.d[b];
        }
    }
    Q8KActsX4 {
        na,
        n_blocks: nb,
        qs,
        bsums,
        d,
    }
}
