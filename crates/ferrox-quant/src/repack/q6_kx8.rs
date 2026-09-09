//! Q6_K packed 8 rows deep into `block_q6_Kx8`, with the GEMV and
//! GEMM that read it. Six bits split across a `ql` nibble plane and a
//! `qh` 2-bit plane, with signed per-16 scales and no mins.

#[cfg(target_arch = "x86_64")]
use super::avx2;
use super::common::*;
#[cfg(target_arch = "aarch64")]
use super::neon;
use crate::{Q8KActivations, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS};

// ---------------------------------------------------------------------------
// Q6_K ×8 interleaved GEMV/GEMM (llama.cpp `block_q6_Kx8`)
// ---------------------------------------------------------------------------

/// Bytes per interleaved `block_q6_Kx8` (8×f16 d + 128 scales + 1024 ql + 512 qh).
pub const Q6_KX8_BLOCK_BYTES: usize = 1680;
/// Number of Q6_K rows packed into one interleaved block.
pub const Q6_KX8_NROWS: usize = 8;

/// Preferred ql/qh interleave width: 8 on x86 AVX2 and ARM i8mm
/// (`ggml_gemm_q6_K_8x8_q8_K`), 4 on DotProd-only NEON.
#[inline]
pub fn q6_kx8_interleave() -> usize {
    if cfg!(target_arch = "x86_64") {
        return 8;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            return 8;
        }
    }
    4
}

/// Pack eight canonical Q6_K super-blocks into one `block_q6_Kx8`.
pub fn make_block_q6_kx8(
    rows: [&[u8]; Q6_KX8_NROWS],
    interleave: usize,
) -> [u8; Q6_KX8_BLOCK_BYTES] {
    debug_assert!(interleave == 4 || interleave == 8);
    for r in &rows {
        debug_assert_eq!(r.len(), Q6_K_BLOCK_BYTES);
    }
    let mut out = [0u8; Q6_KX8_BLOCK_BYTES];
    // d[8] @0, scales[128] @16, ql[1024] @144, qh[512] @1168
    for (i, row) in rows.iter().enumerate() {
        out[i * 2] = row[208];
        out[i * 2 + 1] = row[209];
    }
    let end_ls = (Q6_K_BLOCK_ELEMS * 4) / interleave;
    let ql_out = &mut out[144..1168];
    for i in 0..end_ls {
        let src_id = i % Q6_KX8_NROWS;
        let src_offset = (i / Q6_KX8_NROWS) * interleave;
        let dst_offset = i * interleave;
        let src_ql = &rows[src_id][0..128];
        ql_out[dst_offset..dst_offset + interleave]
            .copy_from_slice(&src_ql[src_offset..src_offset + interleave]);
    }
    let end_hs = end_ls / 2;
    let qh_out = &mut out[1168..];
    for i in 0..end_hs {
        let src_id = i % Q6_KX8_NROWS;
        let src_offset = (i / Q6_KX8_NROWS) * interleave;
        let dst_offset = i * interleave;
        let src_qh = &rows[src_id][128..192];
        qh_out[dst_offset..dst_offset + interleave]
            .copy_from_slice(&src_qh[src_offset..src_offset + interleave]);
    }
    let n_scales = Q6_K_BLOCK_ELEMS / 16;
    let scales_out = &mut out[16..144];
    for i in 0..Q6_KX8_NROWS {
        let src_sc = &rows[i][192..208];
        for j in 0..n_scales {
            scales_out[j * Q6_KX8_NROWS + i] = src_sc[j];
        }
    }
    out
}

pub fn pack_q6_k_matrix_x8(data: &[u8], rows: usize, cols: usize, interleave: usize) -> Vec<u8> {
    // Rows past the last full group of 8 are left canonical, same as the
    // other pack_*_matrix helpers; callers dot them row-by-row.
    assert!(cols.is_multiple_of(Q6_K_BLOCK_ELEMS));
    let row_bytes = (cols / Q6_K_BLOCK_ELEMS) * Q6_K_BLOCK_BYTES;
    assert_eq!(data.len(), rows * row_bytes);
    let n_blocks = cols / Q6_K_BLOCK_ELEMS;
    let n_groups = rows / Q6_KX8_NROWS;
    let mut out = Vec::with_capacity(n_groups * n_blocks * Q6_KX8_BLOCK_BYTES);
    for g in 0..n_groups {
        for b in 0..n_blocks {
            let mut row_refs: [&[u8]; Q6_KX8_NROWS] = [&[]; Q6_KX8_NROWS];
            for (r, slot) in row_refs.iter_mut().enumerate() {
                let base = (g * Q6_KX8_NROWS + r) * row_bytes + b * Q6_K_BLOCK_BYTES;
                *slot = &data[base..base + Q6_K_BLOCK_BYTES];
            }
            out.extend_from_slice(&make_block_q6_kx8(row_refs, interleave));
        }
    }
    out
}

pub(crate) fn gemv_q6_kx8_q8_k_scalar(
    packed: &[u8],
    act: &Q8KActivations,
    n_cols: usize,
    n_row_groups: usize,
    blocklen: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q6_K_BLOCK_ELEMS;
    let ncols = Q6_KX8_NROWS;
    let blocks_per_half = 64 / blocklen;
    debug_assert_eq!(act.n_blocks(), nb);
    debug_assert_eq!(out.len(), n_row_groups * ncols);
    for x in 0..n_row_groups {
        let mut sumf = [0f32; 8];
        let group_off = x * nb * Q6_KX8_BLOCK_BYTES;
        for l in 0..nb {
            let blk = &packed[group_off + l * Q6_KX8_BLOCK_BYTES..][..Q6_KX8_BLOCK_BYTES];
            let d = &blk[0..16];
            let scales = &blk[16..144];
            let ql = &blk[144..1168];
            let qh = &blk[1168..];
            let da = act.d[l];
            let q8 = &act.q[l * Q6_K_BLOCK_ELEMS..(l + 1) * Q6_K_BLOCK_ELEMS];
            for k in 0..(Q6_K_BLOCK_ELEMS / (2 * blocklen)) {
                let base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                let base_h = base_l + 64;
                let scale_idx_l = base_l / 16;
                let scale_idx_h = base_h / 16;
                let qh_shift_l = ((base_l % 128) / 32) * 2;
                let qh_shift_h = ((base_h % 128) / 32) * 2;
                let qh_half_l = (base_l / 128) * 32;
                let qh_half_h = (base_h / 128) * 32;
                for j in 0..ncols {
                    let scale_l = scales[scale_idx_l * ncols + j] as i8 as i32;
                    let scale_h = scales[scale_idx_h * ncols + j] as i8 as i32;
                    let mut sumi_l = 0i32;
                    let mut sumi_h = 0i32;
                    for i in 0..blocklen {
                        let ql_pos = k * ncols * blocklen + j * blocklen + i;
                        let l_4 = (ql[ql_pos] & 0x0F) as i32;
                        let hi_4 = ((ql[ql_pos] >> 4) & 0x0F) as i32;
                        let qh_idx_l = qh_half_l + ((base_l + i) % 32);
                        let qh_chunk_l = qh_idx_l / blocklen;
                        let qh_pos_l = qh_idx_l % blocklen;
                        let qh_offset_l = qh_chunk_l * (blocklen * ncols) + j * blocklen + qh_pos_l;
                        let hi_2_l = ((qh[qh_offset_l] >> qh_shift_l) & 0x3) as i32;
                        let qh_idx_h = qh_half_h + ((base_h + i) % 32);
                        let qh_chunk_h = qh_idx_h / blocklen;
                        let qh_pos_h = qh_idx_h % blocklen;
                        let qh_offset_h = qh_chunk_h * (blocklen * ncols) + j * blocklen + qh_pos_h;
                        let hi_2_h = ((qh[qh_offset_h] >> qh_shift_h) & 0x3) as i32;
                        let q_l = ((hi_2_l << 4) | l_4) - 32;
                        let q_h = ((hi_2_h << 4) | hi_4) - 32;
                        sumi_l += q_l * (q8[base_l + i] as i32);
                        sumi_h += q_h * (q8[base_h + i] as i32);
                    }
                    sumf[j] += (sumi_l * scale_l + sumi_h * scale_h) as f32
                        * f16_from_bytes(&d[j * 2..])
                        * da;
                }
            }
        }
        let base = x * ncols;
        out[base..base + ncols].copy_from_slice(&sumf);
    }
}

pub fn gemv_q6_kx8_q8_k(
    packed: &[u8],
    act: &Q8KActivations,
    n_cols: usize,
    n_row_groups: usize,
    interleave: usize,
    out: &mut [f32],
) {
    assert_eq!(out.len(), n_row_groups * Q6_KX8_NROWS);
    match interleave {
        4 => gemv_q6_kx8_q8_k_scalar(packed, act, n_cols, n_row_groups, 4, out),
        8 => {
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    unsafe {
                        neon::gemv_q6_kx8_q8_k_neon_8x8(packed, act, n_cols, n_row_groups, out);
                    }
                    return;
                }
            }
            gemv_q6_kx8_q8_k_scalar(packed, act, n_cols, n_row_groups, 8, out)
        }
        _ => panic!("q6_kx8 interleave must be 4 or 8, got {interleave}"),
    }
}

pub fn gemv_q6_kx8_group(
    packed: &[u8],
    group: usize,
    act: &Q8KActivations,
    n_cols: usize,
    interleave: usize,
    out8: &mut [f32],
) {
    debug_assert_eq!(out8.len(), Q6_KX8_NROWS);
    let nb = n_cols / Q6_K_BLOCK_ELEMS;
    let off = group * nb * Q6_KX8_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q6_KX8_BLOCK_BYTES];
    gemv_q6_kx8_q8_k(slice, act, n_cols, 1, interleave, out8);
}

pub const Q6_KX8_GEMM_NC: usize = 8;

/// Multi-act GEMM for one Q6_Kx8 row-group; weight decode amortized across acts.
pub fn gemm_q6_kx8_group(
    packed: &[u8],
    group: usize,
    acts: &[Q8KActivations],
    n_cols: usize,
    interleave: usize,
    out: &mut [f32],
) {
    assert_eq!(out.len(), Q6_KX8_NROWS * acts.len());
    assert!(n_cols.is_multiple_of(Q6_K_BLOCK_ELEMS));
    if acts.is_empty() {
        return;
    }
    let nb = n_cols / Q6_K_BLOCK_ELEMS;
    let off = group * nb * Q6_KX8_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q6_KX8_BLOCK_BYTES];
    #[cfg(target_arch = "aarch64")]
    {
        if interleave == 8
            && acts.len() <= Q8K_ACTS_X4_NC
            && std::arch::is_aarch64_feature_detected!("i8mm")
        {
            // Compatibility entry: interleaves the quad here, once per
            // call. Batch callers should prepare the quad once per matmul
            // and use [`gemm_q6_kx8_group_x4`] instead.
            let tile = prepare_q8_k_acts_x4(acts, n_cols);
            unsafe {
                neon::gemm_q6_kx8_q8_k_neon_i8mm(slice, &tile, n_cols, out);
            }
            return;
        }
    }
    let blocklen = interleave;
    assert!(blocklen == 4 || blocklen == 8);
    let na = acts.len();
    let ncols = Q6_KX8_NROWS;
    let blocks_per_half = 64 / blocklen;
    out.fill(0.0);
    for l in 0..nb {
        let blk = &slice[l * Q6_KX8_BLOCK_BYTES..][..Q6_KX8_BLOCK_BYTES];
        let d = &blk[0..16];
        let scales = &blk[16..144];
        let ql = &blk[144..1168];
        let qh = &blk[1168..];
        let mut d_f = [0f32; 8];
        for j in 0..8 {
            d_f[j] = f16_from_bytes(&d[j * 2..]);
        }
        for k in 0..(Q6_K_BLOCK_ELEMS / (2 * blocklen)) {
            let base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
            let base_h = base_l + 64;
            let scale_idx_l = base_l / 16;
            let scale_idx_h = base_h / 16;
            let qh_shift_l = ((base_l % 128) / 32) * 2;
            let qh_shift_h = ((base_h % 128) / 32) * 2;
            let qh_half_l = (base_l / 128) * 32;
            let qh_half_h = (base_h / 128) * 32;
            for j in 0..ncols {
                let scale_l = scales[scale_idx_l * ncols + j] as i8 as i32;
                let scale_h = scales[scale_idx_h * ncols + j] as i8 as i32;
                // Decode 8 weight quants for this (k,j) once.
                let mut q_l = [0i32; 8];
                let mut q_h = [0i32; 8];
                for i in 0..blocklen {
                    let ql_pos = k * ncols * blocklen + j * blocklen + i;
                    let l_4 = (ql[ql_pos] & 0x0F) as i32;
                    let hi_4 = ((ql[ql_pos] >> 4) & 0x0F) as i32;
                    let qh_idx_l = qh_half_l + ((base_l + i) % 32);
                    let qh_chunk_l = qh_idx_l / blocklen;
                    let qh_pos_l = qh_idx_l % blocklen;
                    let qh_offset_l = qh_chunk_l * (blocklen * ncols) + j * blocklen + qh_pos_l;
                    let hi_2_l = ((qh[qh_offset_l] >> qh_shift_l) & 0x3) as i32;
                    let qh_idx_h = qh_half_h + ((base_h + i) % 32);
                    let qh_chunk_h = qh_idx_h / blocklen;
                    let qh_pos_h = qh_idx_h % blocklen;
                    let qh_offset_h = qh_chunk_h * (blocklen * ncols) + j * blocklen + qh_pos_h;
                    let hi_2_h = ((qh[qh_offset_h] >> qh_shift_h) & 0x3) as i32;
                    q_l[i] = ((hi_2_l << 4) | l_4) - 32;
                    q_h[i] = ((hi_2_h << 4) | hi_4) - 32;
                }
                for (a, act) in acts.iter().enumerate() {
                    let da = act.d[l];
                    let q8 = &act.q[l * Q6_K_BLOCK_ELEMS..(l + 1) * Q6_K_BLOCK_ELEMS];
                    let mut sumi_l = 0i32;
                    let mut sumi_h = 0i32;
                    for i in 0..blocklen {
                        sumi_l += q_l[i] * (q8[base_l + i] as i32);
                        sumi_h += q_h[i] * (q8[base_h + i] as i32);
                    }
                    out[j * na + a] += (sumi_l * scale_l + sumi_h * scale_h) as f32 * d_f[j] * da;
                }
            }
        }
    }
}

/// Whether [`gemm_q6_kx8_group_x4`] is the fast Q6_K batch path on this CPU:
/// ARM i8mm with the interleave-8 layout (`ggml_gemm_q6_K_8x8_q8_K`). The
/// scalar Kx8 GEMM measured slower than the per-row NEON dot on ARM, so
/// batch callers should use the Kx8 layout only when this returns true.
#[inline]
pub fn q6_kx8_gemm_uses_acts_x4(interleave: usize) -> bool {
    interleaved_gemm_is_accelerated(interleave)
}

/// [`gemm_q6_kx8_group`] against a pre-interleaved activation quad; the
/// Q6_K counterpart of [`gemm_q4_kx8_group_x4`], with the same contract:
/// interleave-8 packing only, quad prepared once per matmul by
/// [`prepare_q8_k_acts_x4`] (up to 4 activations, not [`Q6_KX8_GEMM_NC`]),
/// `out[r * tile.na + a]`.
pub fn gemm_q6_kx8_group_x4(
    packed: &[u8],
    group: usize,
    tile: &Q8KActsX4,
    n_cols: usize,
    interleave: usize,
    out: &mut [f32],
) {
    gemm_q6_kx8_group_x4_on(
        packed,
        group,
        tile,
        n_cols,
        interleave,
        AccelX4::detect(),
        out,
    );
}

/// [`gemm_q6_kx8_group_x4`] with the kernel choice already made; see
/// [`AccelX4`].
#[inline]
pub fn gemm_q6_kx8_group_x4_on(
    packed: &[u8],
    group: usize,
    tile: &Q8KActsX4,
    n_cols: usize,
    interleave: usize,
    accel: AccelX4,
    out: &mut [f32],
) {
    assert_eq!(
        interleave, 8,
        "the x4 GEMM only exists for interleave-8 packing"
    );
    assert_eq!(out.len(), Q6_KX8_NROWS * tile.na);
    assert!(n_cols.is_multiple_of(Q6_K_BLOCK_ELEMS));
    debug_assert_eq!(tile.n_blocks, n_cols / Q6_K_BLOCK_ELEMS);
    if tile.na == 0 {
        return;
    }
    let nb = n_cols / Q6_K_BLOCK_ELEMS;
    let off = group * nb * Q6_KX8_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q6_KX8_BLOCK_BYTES];

    #[cfg(target_arch = "aarch64")]
    if accel == AccelX4::NeonI8mm {
        unsafe {
            neon::gemm_q6_kx8_q8_k_neon_i8mm(slice, tile, n_cols, out);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if accel == AccelX4::Avx2 {
        unsafe {
            avx2::gemm_q6_kx8_q8_k_avx2(slice, tile, n_cols, out);
        }
        return;
    }
    let _ = accel;
    gemm_q6_kx8_acts_x4_scalar_8(slice, tile, n_cols, out);
}

/// Portable reference for the Q6_K ×4 GEMM: the same math as
/// [`gemv_q6_kx8_q8_k_scalar`] at blocklen 8, per quad row, reading qs and
/// d out of the pre-interleaved [`Q8KActsX4`] (Q6_K has no mins, so the
/// folded bsums are unused). Bit-identical to running that GEMV per
/// activation, which is what the tests assert.
pub(crate) fn gemm_q6_kx8_acts_x4_scalar_8(
    packed: &[u8],
    tile: &Q8KActsX4,
    n_cols: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q6_K_BLOCK_ELEMS;
    let blocklen = 8;
    let ncols = Q6_KX8_NROWS;
    let blocks_per_half = 64 / blocklen;
    let na = tile.na;
    let mut sumf = [[0f32; Q6_KX8_NROWS]; Q8K_ACTS_X4_NC];
    for l in 0..nb {
        let blk = &packed[l * Q6_KX8_BLOCK_BYTES..][..Q6_KX8_BLOCK_BYTES];
        let d = &blk[0..16];
        let scales = &blk[16..144];
        let ql = &blk[144..1168];
        let qh = &blk[1168..];
        let q8 = &tile.qs[l * Q6_K_BLOCK_ELEMS * 4..][..Q6_K_BLOCK_ELEMS * 4];
        for a in 0..na {
            let da = tile.d[l * 4 + a];
            for k in 0..(Q6_K_BLOCK_ELEMS / (2 * blocklen)) {
                let base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                let base_h = base_l + 64;
                let scale_idx_l = base_l / 16;
                let scale_idx_h = base_h / 16;
                let qh_shift_l = ((base_l % 128) / 32) * 2;
                let qh_shift_h = ((base_h % 128) / 32) * 2;
                let qh_half_l = (base_l / 128) * 32;
                let qh_half_h = (base_h / 128) * 32;
                for j in 0..ncols {
                    let scale_l = scales[scale_idx_l * ncols + j] as i8 as i32;
                    let scale_h = scales[scale_idx_h * ncols + j] as i8 as i32;
                    let mut sumi_l = 0i32;
                    let mut sumi_h = 0i32;
                    for i in 0..blocklen {
                        let ql_pos = k * ncols * blocklen + j * blocklen + i;
                        let l_4 = (ql[ql_pos] & 0x0F) as i32;
                        let hi_4 = ((ql[ql_pos] >> 4) & 0x0F) as i32;
                        let qh_idx_l = qh_half_l + ((base_l + i) % 32);
                        let qh_chunk_l = qh_idx_l / blocklen;
                        let qh_pos_l = qh_idx_l % blocklen;
                        let qh_offset_l = qh_chunk_l * (blocklen * ncols) + j * blocklen + qh_pos_l;
                        let hi_2_l = ((qh[qh_offset_l] >> qh_shift_l) & 0x3) as i32;
                        let qh_idx_h = qh_half_h + ((base_h + i) % 32);
                        let qh_chunk_h = qh_idx_h / blocklen;
                        let qh_pos_h = qh_idx_h % blocklen;
                        let qh_offset_h = qh_chunk_h * (blocklen * ncols) + j * blocklen + qh_pos_h;
                        let hi_2_h = ((qh[qh_offset_h] >> qh_shift_h) & 0x3) as i32;
                        let q_l = ((hi_2_l << 4) | l_4) - 32;
                        let q_h = ((hi_2_h << 4) | hi_4) - 32;
                        // Canonical q8 element `e` lives at run `e/8`, row
                        // `a`, lane `e%8` of the interleaved block.
                        let e_l = base_l + i;
                        let e_h = base_h + i;
                        sumi_l += q_l * (q8[(e_l / 8) * 32 + a * 8 + (e_l % 8)] as i32);
                        sumi_h += q_h * (q8[(e_h / 8) * 32 + a * 8 + (e_h % 8)] as i32);
                    }
                    sumf[a][j] += (sumi_l * scale_l + sumi_h * scale_h) as f32
                        * f16_from_bytes(&d[j * 2..])
                        * da;
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
