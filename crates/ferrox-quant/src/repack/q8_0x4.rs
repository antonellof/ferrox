//! Q8_0 packed 4 rows deep into `block_q8_0x4` (llama.cpp
//! `make_block_q8_0x4`), with the GEMV and GEMM that read it.

#[cfg(target_arch = "x86_64")]
use super::avx2;
use super::common::*;
#[cfg(target_arch = "aarch64")]
use super::neon;
use crate::{Q8Activations, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};

// ---------------------------------------------------------------------------
// Q8_0 ×4 interleaved GEMV (llama.cpp `block_q8_0x4` / `ggml_gemv_q8_0_4x4`)
// ---------------------------------------------------------------------------

/// Bytes per interleaved `block_q8_0x4` (4 × f16 d + 128 qs).
pub const Q8_0X4_BLOCK_BYTES: usize = 136;
/// Number of Q8_0 rows packed into one interleaved block.
pub const Q8_0X4_NROWS: usize = 4;
/// qs interleave width for `ggml_gemv_q8_0_4x4_q8_0` (NEON SDOT). The
/// DotProd-only default; [`q8_0x4_interleave`] picks 8 on i8mm hosts.
pub const Q8_0X4_INTERLEAVE: usize = 4;

/// Preferred qs interleave width: 8 on ARM i8mm (`ggml_gemm_q8_0_4x8_q8_0`
/// via `ggml_repack_get_optimal_repack_type`), 4 on DotProd-only NEON and
/// everywhere else (the scalar fallback handles either).
#[inline]
pub fn q8_0x4_interleave() -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            return 8;
        }
    }
    Q8_0X4_INTERLEAVE
}

/// Pack four canonical Q8_0 blocks (same column-block) into one
/// `block_q8_0x4`. `interleave` is 4 (ARM 4x4) or 8 (4x8).
pub fn make_block_q8_0x4(
    rows: [&[u8]; Q8_0X4_NROWS],
    interleave: usize,
) -> [u8; Q8_0X4_BLOCK_BYTES] {
    debug_assert!(interleave == 4 || interleave == 8);
    for r in &rows {
        debug_assert_eq!(r.len(), Q8_0_BLOCK_BYTES);
    }
    let mut out = [0u8; Q8_0X4_BLOCK_BYTES];
    for (i, row) in rows.iter().enumerate() {
        out[i * 2] = row[0];
        out[i * 2 + 1] = row[1];
    }
    let end = (Q8_0_BLOCK_ELEMS * Q8_0X4_NROWS) / interleave;
    let qs_out = &mut out[8..];
    for i in 0..end {
        let src_id = i % Q8_0X4_NROWS;
        let src_offset = (i / Q8_0X4_NROWS) * interleave;
        let dst_offset = i * interleave;
        let src_qs = &rows[src_id][2..34];
        qs_out[dst_offset..dst_offset + interleave]
            .copy_from_slice(&src_qs[src_offset..src_offset + interleave]);
    }
    out
}

/// Repack a Q8_0 matrix into interleaved `block_q8_0x4` groups. Tail rows
/// (not divisible by 4) are omitted; caller dots them with [`crate::dot_q8_0_q8`].
pub fn pack_q8_0_matrix_x4(data: &[u8], rows: usize, cols: usize, interleave: usize) -> Vec<u8> {
    assert!(cols.is_multiple_of(Q8_0_BLOCK_ELEMS));
    let n_blocks = cols / Q8_0_BLOCK_ELEMS;
    let row_bytes = n_blocks * Q8_0_BLOCK_BYTES;
    assert_eq!(data.len(), rows * row_bytes);
    let n_groups = rows / Q8_0X4_NROWS;
    let mut out = Vec::with_capacity(n_groups * n_blocks * Q8_0X4_BLOCK_BYTES);
    for g in 0..n_groups {
        for b in 0..n_blocks {
            let mut row_refs: [&[u8]; Q8_0X4_NROWS] = [&[]; Q8_0X4_NROWS];
            for (r, slot) in row_refs.iter_mut().enumerate() {
                let base = (g * Q8_0X4_NROWS + r) * row_bytes + b * Q8_0_BLOCK_BYTES;
                *slot = &data[base..base + Q8_0_BLOCK_BYTES];
            }
            out.extend_from_slice(&make_block_q8_0x4(row_refs, interleave));
        }
    }
    out
}

/// Scalar GEMV (`ggml_gemv_q8_0_4x{4,8}_q8_0_generic`); `blocklen` is the
/// interleave the matrix was packed with.
pub(crate) fn gemv_q8_0x4_q8_0_scalar(
    packed: &[u8],
    act: &Q8Activations,
    n_cols: usize,
    n_row_groups: usize,
    blocklen: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    let ncols = Q8_0X4_NROWS;
    debug_assert_eq!(act.n_blocks(), nb);
    debug_assert_eq!(out.len(), n_row_groups * ncols);
    debug_assert_eq!(packed.len(), n_row_groups * nb * Q8_0X4_BLOCK_BYTES);

    for x in 0..n_row_groups {
        let mut sumf = [0f32; 4];
        let group_off = x * nb * Q8_0X4_BLOCK_BYTES;
        for l in 0..nb {
            let blk = &packed[group_off + l * Q8_0X4_BLOCK_BYTES..][..Q8_0X4_BLOCK_BYTES];
            let qs = &blk[8..];
            let da = act.d[l];
            let q8 = &act.q[l * Q8_0_BLOCK_ELEMS..(l + 1) * Q8_0_BLOCK_ELEMS];
            for k in 0..(Q8_0_BLOCK_ELEMS / blocklen) {
                for j in 0..ncols {
                    let mut sumi = 0i32;
                    for i in 0..blocklen {
                        let v0 = qs[k * ncols * blocklen + j * blocklen + i] as i8 as i32;
                        sumi += v0 * q8[k * blocklen + i] as i32;
                    }
                    sumf[j] += sumi as f32 * f16_from_bytes(&blk[j * 2..]) * da;
                }
            }
        }
        let base = x * ncols;
        out[base..base + ncols].copy_from_slice(&sumf);
    }
}

/// GEMV: interleaved Q8_0 weights × Q8 activation → `n_row_groups * 4` f32s.
/// `interleave` must match the packing (4: NEON SDOT `4x4`; 8: NEON `4x8`).
pub fn gemv_q8_0x4_q8_0(
    packed: &[u8],
    act: &Q8Activations,
    n_cols: usize,
    n_row_groups: usize,
    interleave: usize,
    out: &mut [f32],
) {
    assert!(n_cols.is_multiple_of(Q8_0_BLOCK_ELEMS));
    assert_eq!(out.len(), n_row_groups * Q8_0X4_NROWS);
    match interleave {
        4 => {
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    unsafe {
                        neon::gemv_q8_0x4_q8_0_neon_sdot(packed, act, n_cols, n_row_groups, out);
                    }
                    return;
                }
            }
            gemv_q8_0x4_q8_0_scalar(packed, act, n_cols, n_row_groups, 4, out);
        }
        8 => {
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    unsafe {
                        neon::gemv_q8_0x4_q8_0_neon_4x8(packed, act, n_cols, n_row_groups, out);
                    }
                    return;
                }
            }
            gemv_q8_0x4_q8_0_scalar(packed, act, n_cols, n_row_groups, 8, out);
        }
        _ => panic!("q8_0x4 interleave must be 4 or 8, got {interleave}"),
    }
}

/// How many activations one [`gemm_q8_0x4_group`] pass keeps in flight.
/// Four f32x4 accumulators plus the eight loaded weight vectors fit
/// comfortably in NEON's register file, so each weight load is amortized
/// over four activations instead of being repeated per activation.
pub const Q8_0X4_GEMM_NC: usize = 8;

/// GEMM counterpart of [`gemv_q8_0x4_group`]: one row-group (4 rows)
/// against `acts.len()` activations at once.
///
/// The difference that matters is register blocking over the *batch*
/// dimension. Calling the GEMV once per activation reloads the group's
/// eight `int8x16` weight vectors for every activation; this loads them
/// once per `Q8_0X4_GEMM_NC` activations and issues the dot products
/// back to back. That is the same reason llama.cpp ships
/// `ggml_gemm_q8_0_4x4_q8_0` next to `ggml_gemv_q8_0_4x4_q8_0` rather
/// than looping the GEMV.
///
/// `out` is `[row][act]`: `out[r * acts.len() + j]`, which is the layout
/// `WeightMatrix::apply_batch` accumulates into.
pub fn gemm_q8_0x4_group(
    packed: &[u8],
    group: usize,
    acts: &[Q8Activations],
    n_cols: usize,
    interleave: usize,
    out: &mut [f32],
) {
    assert_eq!(out.len(), Q8_0X4_NROWS * acts.len());
    assert!(n_cols.is_multiple_of(Q8_0_BLOCK_ELEMS));
    if acts.is_empty() {
        return;
    }
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    let off = group * nb * Q8_0X4_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q8_0X4_BLOCK_BYTES];

    #[cfg(target_arch = "aarch64")]
    {
        if interleave == 8 && std::arch::is_aarch64_feature_detected!("i8mm") {
            // Compatibility entry: interleaves the quads here, once per
            // call. Batch callers should prepare them once per matmul and
            // use [`gemm_q8_0x4_group_x4`] instead.
            for (t, chunk) in acts.chunks(Q8K_ACTS_X4_NC).enumerate() {
                let tile = prepare_q8_acts_x4(chunk, n_cols);
                let mut tmp = [0f32; Q8_0X4_NROWS * Q8K_ACTS_X4_NC];
                let n = chunk.len();
                unsafe {
                    neon::gemm_q8_0x4_q8_0_neon_i8mm(
                        slice,
                        &tile,
                        n_cols,
                        &mut tmp[..Q8_0X4_NROWS * n],
                    );
                }
                for r in 0..Q8_0X4_NROWS {
                    for j in 0..n {
                        out[r * acts.len() + t * Q8K_ACTS_X4_NC + j] = tmp[r * n + j];
                    }
                }
            }
            return;
        }
        if interleave == 4 && std::arch::is_aarch64_feature_detected!("dotprod") {
            unsafe {
                neon::gemm_q8_0x4_q8_0_neon_sdot(slice, acts, n_cols, out);
            }
            return;
        }
    }
    // Portable fallback: the GEMV, once per activation. Same results,
    // none of the reuse.
    let mut tmp = [0f32; Q8_0X4_NROWS];
    for (j, act) in acts.iter().enumerate() {
        gemv_q8_0x4_q8_0(slice, act, n_cols, 1, interleave, &mut tmp);
        for (r, v) in tmp.iter().enumerate() {
            out[r * acts.len() + j] = *v;
        }
    }
}

/// Whether [`gemm_q8_0x4_group_x4`] is the fast Q8_0 batch path on this
/// CPU: ARM i8mm with the interleave-8 layout (`ggml_gemm_q8_0_4x8_q8_0`).
#[inline]
pub fn q8_0x4_gemm_uses_acts_x4(interleave: usize) -> bool {
    interleaved_gemm_is_accelerated(interleave)
}

/// [`gemm_q8_0x4_group`] against a pre-interleaved activation quad;
/// interleave-8 packing only, quad prepared once per matmul by
/// [`prepare_q8_acts_x4`]. `out` is `[row][act]`: `out[r * tile.na + a]`.
pub fn gemm_q8_0x4_group_x4(
    packed: &[u8],
    group: usize,
    tile: &Q8ActsX4,
    n_cols: usize,
    interleave: usize,
    out: &mut [f32],
) {
    gemm_q8_0x4_group_x4_on(
        packed,
        group,
        tile,
        n_cols,
        interleave,
        AccelX4::detect(),
        out,
    );
}

/// [`gemm_q8_0x4_group_x4`] with the kernel choice already made; see
/// [`AccelX4`].
#[inline]
pub fn gemm_q8_0x4_group_x4_on(
    packed: &[u8],
    group: usize,
    tile: &Q8ActsX4,
    n_cols: usize,
    interleave: usize,
    accel: AccelX4,
    out: &mut [f32],
) {
    assert_eq!(
        interleave, 8,
        "the x4 GEMM only exists for interleave-8 packing"
    );
    assert_eq!(out.len(), Q8_0X4_NROWS * tile.na);
    assert!(n_cols.is_multiple_of(Q8_0_BLOCK_ELEMS));
    debug_assert_eq!(tile.n_blocks, n_cols / Q8_0_BLOCK_ELEMS);
    if tile.na == 0 {
        return;
    }
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    let off = group * nb * Q8_0X4_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q8_0X4_BLOCK_BYTES];

    #[cfg(target_arch = "aarch64")]
    if accel == AccelX4::NeonI8mm {
        unsafe {
            neon::gemm_q8_0x4_q8_0_neon_i8mm(slice, tile, n_cols, out);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if accel == AccelX4::Avx2 {
        unsafe {
            avx2::gemm_q8_0x4_q8_0_avx2(slice, tile, n_cols, out);
        }
        return;
    }
    let _ = accel;
    gemm_q8_0x4_acts_x4_scalar_8(slice, tile, n_cols, out);
}

/// Portable reference for the Q8_0 ×4 GEMM: the same math as
/// [`gemv_q8_0x4_q8_0_scalar`] at blocklen 8, per quad row, reading qs and
/// d out of the pre-interleaved [`Q8ActsX4`]. Bit-identical to running
/// that GEMV per activation, which is what the tests assert.
pub(crate) fn gemm_q8_0x4_acts_x4_scalar_8(
    packed: &[u8],
    tile: &Q8ActsX4,
    n_cols: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    let blocklen = 8;
    let ncols = Q8_0X4_NROWS;
    let na = tile.na;
    let mut sumf = [[0f32; Q8_0X4_NROWS]; Q8K_ACTS_X4_NC];
    for l in 0..nb {
        let blk = &packed[l * Q8_0X4_BLOCK_BYTES..][..Q8_0X4_BLOCK_BYTES];
        let qs = &blk[8..];
        let q8 = &tile.qs[l * Q8_0_BLOCK_ELEMS * 4..][..Q8_0_BLOCK_ELEMS * 4];
        for a in 0..na {
            let da = tile.d[l * 4 + a];
            for k in 0..(Q8_0_BLOCK_ELEMS / blocklen) {
                for j in 0..ncols {
                    let mut sumi = 0i32;
                    for i in 0..blocklen {
                        let v0 = qs[k * ncols * blocklen + j * blocklen + i] as i8 as i32;
                        // Canonical q8 element `e` lives at run `e/8`, row
                        // `a`, lane `e%8` of the interleaved block.
                        let e = k * blocklen + i;
                        sumi += v0 * q8[(e / 8) * 32 + a * 8 + (e % 8)] as i32;
                    }
                    sumf[a][j] += sumi as f32 * f16_from_bytes(&blk[j * 2..]) * da;
                }
            }
        }
    }
    for j in 0..ncols {
        for (a, row) in sumf.iter().take(na).enumerate() {
            out[j * na + a] = row[j];
        }
    }
}

/// One row-group (4 outputs) starting at `group` within a packed Q8_0x4 matrix.
#[inline]
pub fn gemv_q8_0x4_group(
    packed: &[u8],
    group: usize,
    act: &Q8Activations,
    n_cols: usize,
    interleave: usize,
    out4: &mut [f32],
) {
    debug_assert_eq!(out4.len(), Q8_0X4_NROWS);
    let nb = n_cols / Q8_0_BLOCK_ELEMS;
    let off = group * nb * Q8_0X4_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q8_0X4_BLOCK_BYTES];
    gemv_q8_0x4_q8_0(slice, act, n_cols, 1, interleave, out4);
}
