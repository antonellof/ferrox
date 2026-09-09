//! Q5_K packed 8 rows deep into `block_q5_Kx8`, with the GEMV and
//! GEMM that read it. The fifth bit lives in a separate `qh` plane,
//! interleaved the same way as `qs`.

#[cfg(target_arch = "x86_64")]
use super::avx2;
use super::common::*;
#[cfg(target_arch = "aarch64")]
use super::neon;
use crate::{Q8KActivations, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS};

// ---------------------------------------------------------------------------
// Q5_K ×8 interleaved GEMV/GEMM (llama.cpp `block_q5_Kx8` / `ggml_gemm_q5_K_8x4`)
// ---------------------------------------------------------------------------

/// Bytes per interleaved `block_q5_Kx8` (8 × f16 d + 8 × f16 dmin + 96 scales +
/// 256 qh + 1024 qs).
pub const Q5_KX8_BLOCK_BYTES: usize = 1408;
/// Number of Q5_K rows packed into one interleaved block.
pub const Q5_KX8_NROWS: usize = 8;

/// Preferred qs/qh interleave width: 8 where a `×4` GEMM kernel exists
/// (x86 AVX2 and ARM i8mm), 4 on DotProd-only NEON (`8x4`). One answer
/// for every kind — see [`preferred_interleave`].
#[inline]
pub fn q5_kx8_interleave() -> usize {
    preferred_interleave()
}

/// Pack eight canonical Q5_K super-blocks (same column-block index) into
/// one `block_q5_Kx8`. `interleave` is 4 (ARM DotProd) or 8 (x86 / ARM i8mm).
pub fn make_block_q5_kx8(
    rows: [&[u8]; Q5_KX8_NROWS],
    interleave: usize,
) -> [u8; Q5_KX8_BLOCK_BYTES] {
    debug_assert!(interleave == 4 || interleave == 8);
    for r in &rows {
        debug_assert_eq!(r.len(), Q5_K_BLOCK_BYTES);
    }
    let mut out = [0u8; Q5_KX8_BLOCK_BYTES];
    // d[8] at 0, dmin[8] at 16, scales[96] at 32, qh[256] at 128, qs[1024] at 384.
    for (i, row) in rows.iter().enumerate() {
        out[i * 2] = row[0];
        out[i * 2 + 1] = row[1];
        out[16 + i * 2] = row[2];
        out[16 + i * 2 + 1] = row[3];
    }

    let end = (Q5_K_BLOCK_ELEMS * 4) / interleave;
    let qs_out = &mut out[384..];
    for i in 0..end {
        let src_id = i % Q5_KX8_NROWS;
        let src_offset = (i / Q5_KX8_NROWS) * interleave;
        let dst_offset = i * interleave;
        let src_qs = &rows[src_id][48..176];
        qs_out[dst_offset..dst_offset + interleave]
            .copy_from_slice(&src_qs[src_offset..src_offset + interleave]);
    }

    let qh_end = end / 4;
    let qh_out = &mut out[128..384];
    for i in 0..qh_end {
        let src_id = i % Q5_KX8_NROWS;
        let src_offset = (i / Q5_KX8_NROWS) * interleave;
        let dst_offset = i * interleave;
        let src_qh = &rows[src_id][16..48];
        qh_out[dst_offset..dst_offset + interleave]
            .copy_from_slice(&src_qh[src_offset..src_offset + interleave]);
    }

    // Scale/min rearrangement (same 6-bit packing as Q4_Kx8).
    let mut s = [0u8; 8];
    let mut m = [0u8; 8];
    let scales_out = &mut out[32..128];

    for i in 0..4 {
        for j in 0..8 {
            let sc = &rows[j][4..16];
            s[j] = sc[i] & 63;
            m[j] = sc[i + 4] & 63;
        }
        let base = i * 12;
        scales_out[base] = (s[0] & 63) + ((s[4] & 48) << 2);
        scales_out[base + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        scales_out[base + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        scales_out[base + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        scales_out[base + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        scales_out[base + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        scales_out[base + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        scales_out[base + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        scales_out[base + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        scales_out[base + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        scales_out[base + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        scales_out[base + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    for i in 0..4 {
        for j in 0..8 {
            let sc = &rows[j][4..16];
            s[j] = ((sc[i] & 192) >> 2) | (sc[i + 8] & 15);
            m[j] = ((sc[i + 4] & 192) >> 2) | ((sc[i + 8] & 240) >> 4);
        }
        let base = 48 + i * 12;
        scales_out[base] = (s[0] & 63) + ((s[4] & 48) << 2);
        scales_out[base + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
        scales_out[base + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
        scales_out[base + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
        scales_out[base + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
        scales_out[base + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
        scales_out[base + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
        scales_out[base + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
        scales_out[base + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
        scales_out[base + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
        scales_out[base + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        scales_out[base + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    out
}

/// Repack a full Q5_K matrix (row-major canonical blocks) into interleaved
/// `block_q5_Kx8` groups. Tail rows (not divisible by 8) are omitted.
pub fn pack_q5_k_matrix_x8(data: &[u8], rows: usize, cols: usize, interleave: usize) -> Vec<u8> {
    assert!(cols.is_multiple_of(Q5_K_BLOCK_ELEMS));
    let n_blocks = cols / Q5_K_BLOCK_ELEMS;
    let row_bytes = n_blocks * Q5_K_BLOCK_BYTES;
    assert_eq!(data.len(), rows * row_bytes);
    let n_groups = rows / Q5_KX8_NROWS;
    let mut out = Vec::with_capacity(n_groups * n_blocks * Q5_KX8_BLOCK_BYTES);
    for g in 0..n_groups {
        for b in 0..n_blocks {
            let mut row_refs: [&[u8]; Q5_KX8_NROWS] = [&[]; Q5_KX8_NROWS];
            for (r, slot) in row_refs.iter_mut().enumerate() {
                let base = (g * Q5_KX8_NROWS + r) * row_bytes + b * Q5_K_BLOCK_BYTES;
                *slot = &data[base..base + Q5_K_BLOCK_BYTES];
            }
            out.extend_from_slice(&make_block_q5_kx8(row_refs, interleave));
        }
    }
    out
}

/// Scalar GEMV for interleave=4 (`ggml_gemv_q5_K_8x4_q8_K_generic`).
pub(crate) fn gemv_q5_kx8_q8_k_scalar_4(
    packed: &[u8],
    act: &Q8KActivations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q5_K_BLOCK_ELEMS;
    let blocklen = 4;
    let ncols_interleaved = Q5_KX8_NROWS;
    debug_assert_eq!(act.n_blocks(), nb);
    debug_assert_eq!(out.len(), n_row_groups * ncols_interleaved);
    debug_assert_eq!(packed.len(), n_row_groups * nb * Q5_KX8_BLOCK_BYTES);

    for x in 0..n_row_groups {
        let mut sumf = [0f32; 8];
        let mut sum_minf = [0f32; 8];
        let group_off = x * nb * Q5_KX8_BLOCK_BYTES;
        for l in 0..nb {
            let blk = &packed[group_off + l * Q5_KX8_BLOCK_BYTES..][..Q5_KX8_BLOCK_BYTES];
            let d = &blk[0..16];
            let dmin = &blk[16..32];
            let scales = &blk[32..128];
            let qh = &blk[128..384];
            let qs = &blk[384..];
            let da = act.d[l];
            let q8 = &act.q[l * Q5_K_BLOCK_ELEMS..(l + 1) * Q5_K_BLOCK_ELEMS];
            let bsums = &act.bsums[l * 16..(l + 1) * 16];

            let mut all_scales = [[0u8; 8]; 8];
            let mut all_mins = [[0u8; 8]; 8];
            for sb in 0..8 {
                decode_scales_mins(&scales[sb * 12..], &mut all_scales[sb], &mut all_mins[sb]);
            }

            let n_k = Q5_K_BLOCK_ELEMS / (2 * blocklen); // 32
            for k in 0..n_k {
                let sb_pair = k / 8;
                let sc0 = &all_scales[sb_pair * 2];
                let sc1 = &all_scales[sb_pair * 2 + 1];
                let qh_shift = sb_pair * 2;
                for j in 0..ncols_interleaved {
                    let mut sumi = 0i32;
                    for i in 0..blocklen {
                        let b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;
                        let qh_idx = (k * blocklen + i) % 32;
                        let qh_chunk = qh_idx / blocklen;
                        let qh_pos = qh_idx % blocklen;
                        let b_qh_offset =
                            qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;
                        let qh_val = qh[b_qh_offset];
                        let h0 = (qh_val >> qh_shift) & 1;
                        let h1 = (qh_val >> (qh_shift + 1)) & 1;
                        let v0 = ((qs[b_qs_offset] & 0x0F) | (h0 << 4)) as i32;
                        let v1 = ((qs[b_qs_offset] >> 4) | (h1 << 4)) as i32;
                        let a0 = q8[(k / 8) * 64 + (k % 8) * blocklen + i] as i32;
                        let a1 = q8[(k / 8) * 64 + (k % 8) * blocklen + i + 32] as i32;
                        sumi += v0 * a0 * sc0[j] as i32 + v1 * a1 * sc1[j] as i32;
                    }
                    sumf[j] += sumi as f32 * f16_from_bytes(&d[j * 2..]) * da;
                }
            }
            for sb in 0..8 {
                let mins = &all_mins[sb];
                let bsum = bsums[sb * 2] as i32 + bsums[sb * 2 + 1] as i32;
                for j in 0..ncols_interleaved {
                    sum_minf[j] +=
                        mins[j] as f32 * bsum as f32 * f16_from_bytes(&dmin[j * 2..]) * da;
                }
            }
        }
        let base = x * ncols_interleaved;
        for j in 0..ncols_interleaved {
            out[base + j] = sumf[j] - sum_minf[j];
        }
    }
}

/// Scalar GEMV for interleave=8 (`ggml_gemv_q5_K_8x8_q8_K_generic`).
pub(crate) fn gemv_q5_kx8_q8_k_scalar_8(
    packed: &[u8],
    act: &Q8KActivations,
    n_cols: usize,
    n_row_groups: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q5_K_BLOCK_ELEMS;
    let blocklen = 8;
    let ncols_interleaved = Q5_KX8_NROWS;
    debug_assert_eq!(act.n_blocks(), nb);
    debug_assert_eq!(out.len(), n_row_groups * ncols_interleaved);

    for x in 0..n_row_groups {
        let mut sumf = [0f32; 8];
        let mut sum_minf = [0f32; 8];
        let group_off = x * nb * Q5_KX8_BLOCK_BYTES;
        for l in 0..nb {
            let blk = &packed[group_off + l * Q5_KX8_BLOCK_BYTES..][..Q5_KX8_BLOCK_BYTES];
            let d = &blk[0..16];
            let dmin = &blk[16..32];
            let scales = &blk[32..128];
            let qh = &blk[128..384];
            let qs = &blk[384..];
            let da = act.d[l];
            let q8 = &act.q[l * Q5_K_BLOCK_ELEMS..(l + 1) * Q5_K_BLOCK_ELEMS];
            let bsums = &act.bsums[l * 16..(l + 1) * 16];

            let mut all_scales = [[0u8; 8]; 8];
            let mut all_mins = [[0u8; 8]; 8];
            for sb in 0..8 {
                decode_scales_mins(&scales[sb * 12..], &mut all_scales[sb], &mut all_mins[sb]);
            }

            let n_k = Q5_K_BLOCK_ELEMS / (2 * blocklen); // 16
            for k in 0..n_k {
                let sb_pair = k / 4;
                let sc0 = &all_scales[sb_pair * 2];
                let sc1 = &all_scales[sb_pair * 2 + 1];
                let qh_shift = sb_pair * 2;
                for j in 0..ncols_interleaved {
                    let mut sumi = 0i32;
                    for i in 0..blocklen {
                        let b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;
                        let qh_idx = (k * blocklen + i) % 32;
                        let qh_chunk = qh_idx / blocklen;
                        let qh_pos = qh_idx % blocklen;
                        let b_qh_offset =
                            qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;
                        let qh_val = qh[b_qh_offset];
                        let h0 = (qh_val >> qh_shift) & 1;
                        let h1 = (qh_val >> (qh_shift + 1)) & 1;
                        let v0 = ((qs[b_qs_offset] & 0x0F) | (h0 << 4)) as i32;
                        let v1 = ((qs[b_qs_offset] >> 4) | (h1 << 4)) as i32;
                        let a0 = q8[(k >> 2) * 64 + (k % 4) * blocklen + i] as i32;
                        let a1 = q8[(k >> 2) * 64 + (k % 4) * blocklen + i + 32] as i32;
                        sumi += v0 * a0 * sc0[j] as i32 + v1 * a1 * sc1[j] as i32;
                    }
                    sumf[j] += sumi as f32 * f16_from_bytes(&d[j * 2..]) * da;
                }
            }
            for sb in 0..8 {
                let mins = &all_mins[sb];
                let bsum = bsums[sb * 2] as i32 + bsums[sb * 2 + 1] as i32;
                for j in 0..ncols_interleaved {
                    sum_minf[j] +=
                        mins[j] as f32 * bsum as f32 * f16_from_bytes(&dmin[j * 2..]) * da;
                }
            }
        }
        let base = x * ncols_interleaved;
        for j in 0..ncols_interleaved {
            out[base + j] = sumf[j] - sum_minf[j];
        }
    }
}

/// GEMV: interleaved Q5_K weights × Q8_K activation → `n_row_groups * 8` f32s.
pub fn gemv_q5_kx8_q8_k(
    packed: &[u8],
    act: &Q8KActivations,
    n_cols: usize,
    n_row_groups: usize,
    interleave: usize,
    out: &mut [f32],
) {
    assert!(n_cols.is_multiple_of(Q5_K_BLOCK_ELEMS));
    assert_eq!(out.len(), n_row_groups * Q5_KX8_NROWS);
    match interleave {
        4 => {
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    unsafe {
                        neon::gemv_q5_kx8_q8_k_neon_sdot(packed, act, n_cols, n_row_groups, out);
                    }
                    return;
                }
            }
            gemv_q5_kx8_q8_k_scalar_4(packed, act, n_cols, n_row_groups, out);
        }
        8 => {
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("dotprod") {
                    unsafe {
                        neon::gemv_q5_kx8_q8_k_neon_8x8(packed, act, n_cols, n_row_groups, out);
                    }
                    return;
                }
            }
            gemv_q5_kx8_q8_k_scalar_8(packed, act, n_cols, n_row_groups, out);
        }
        _ => panic!("q5_kx8 interleave must be 4 or 8, got {interleave}"),
    }
}

/// One row-group (8 outputs) starting at `group` within a packed Q5_K matrix.
#[inline]
pub fn gemv_q5_kx8_group(
    packed: &[u8],
    group: usize,
    act: &Q8KActivations,
    n_cols: usize,
    interleave: usize,
    out8: &mut [f32],
) {
    debug_assert_eq!(out8.len(), Q5_KX8_NROWS);
    let nb = n_cols / Q5_K_BLOCK_ELEMS;
    let off = group * nb * Q5_KX8_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q5_KX8_BLOCK_BYTES];
    gemv_q5_kx8_q8_k(slice, act, n_cols, 1, interleave, out8);
}

/// How many activations one [`gemm_q5_kx8_group`] pass keeps in flight.
pub const Q5_KX8_GEMM_NC: usize = 4;

/// GEMM counterpart of [`gemv_q5_kx8_group`]: one row-group (8 rows)
/// against `acts.len()` activations at once. `out` is `[row][act]`:
/// `out[r * acts.len() + j]`.
///
/// Weight-side decode (scales/mins/qh/qs addressing) is amortized across
/// the activation tile — same motivation as llama `ggml_gemm_q5_K_*`.
pub fn gemm_q5_kx8_group(
    packed: &[u8],
    group: usize,
    acts: &[Q8KActivations],
    n_cols: usize,
    interleave: usize,
    out: &mut [f32],
) {
    assert_eq!(out.len(), Q5_KX8_NROWS * acts.len());
    assert!(n_cols.is_multiple_of(Q5_K_BLOCK_ELEMS));
    if acts.is_empty() {
        return;
    }
    let nb = n_cols / Q5_K_BLOCK_ELEMS;
    let off = group * nb * Q5_KX8_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q5_KX8_BLOCK_BYTES];
    #[cfg(target_arch = "aarch64")]
    {
        if acts.len() <= Q5_KX8_GEMM_NC {
            if interleave == 8 && std::arch::is_aarch64_feature_detected!("i8mm") {
                // Compatibility entry: interleaves the quad here, once per
                // call. Batch callers should prepare the quad once per
                // matmul and use [`gemm_q5_kx8_group_x4`] instead.
                let tile = prepare_q8_k_acts_x4(acts, n_cols);
                unsafe {
                    neon::gemm_q5_kx8_q8_k_neon_i8mm(slice, &tile, n_cols, out);
                }
                return;
            }
            if interleave == 4 && std::arch::is_aarch64_feature_detected!("dotprod") {
                unsafe {
                    neon::gemm_q5_kx8_q8_k_neon_sdot(slice, acts, n_cols, out);
                }
                return;
            }
        }
    }
    match interleave {
        4 => gemm_q5_kx8_q8_k_scalar_4(slice, acts, n_cols, out),
        8 => gemm_q5_kx8_q8_k_scalar_8(slice, acts, n_cols, out),
        _ => panic!("q5_kx8 interleave must be 4 or 8, got {interleave}"),
    }
}

/// Whether [`gemm_q5_kx8_group_x4`] is the fast Q5_K batch path on this CPU:
/// ARM i8mm with the interleave-8 layout (`ggml_gemm_q5_K_8x8_q8_K`).
#[inline]
pub fn q5_kx8_gemm_uses_acts_x4(interleave: usize) -> bool {
    interleaved_gemm_is_accelerated(interleave)
}

/// [`gemm_q5_kx8_group`] against a pre-interleaved activation quad; the
/// Q5_K counterpart of [`gemm_q4_kx8_group_x4`], with the same contract:
/// interleave-8 packing only, quad prepared once per matmul by
/// [`prepare_q8_k_acts_x4`], `out[r * tile.na + a]`.
pub fn gemm_q5_kx8_group_x4(
    packed: &[u8],
    group: usize,
    tile: &Q8KActsX4,
    n_cols: usize,
    interleave: usize,
    out: &mut [f32],
) {
    gemm_q5_kx8_group_x4_on(
        packed,
        group,
        tile,
        n_cols,
        interleave,
        AccelX4::detect(),
        out,
    );
}

/// [`gemm_q5_kx8_group_x4`] with the kernel choice already made; see
/// [`AccelX4`].
#[inline]
pub fn gemm_q5_kx8_group_x4_on(
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
    assert_eq!(out.len(), Q5_KX8_NROWS * tile.na);
    assert!(n_cols.is_multiple_of(Q5_K_BLOCK_ELEMS));
    debug_assert_eq!(tile.n_blocks, n_cols / Q5_K_BLOCK_ELEMS);
    if tile.na == 0 {
        return;
    }
    let nb = n_cols / Q5_K_BLOCK_ELEMS;
    let off = group * nb * Q5_KX8_BLOCK_BYTES;
    let slice = &packed[off..off + nb * Q5_KX8_BLOCK_BYTES];

    #[cfg(target_arch = "aarch64")]
    if accel == AccelX4::NeonI8mm {
        unsafe {
            neon::gemm_q5_kx8_q8_k_neon_i8mm(slice, tile, n_cols, out);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if accel == AccelX4::Avx2 {
        unsafe {
            avx2::gemm_q5_kx8_q8_k_avx2(slice, tile, n_cols, out);
        }
        return;
    }
    let _ = accel;
    gemm_q5_kx8_acts_x4_scalar_8(slice, tile, n_cols, out);
}

/// Portable reference for the Q5_K ×4 GEMM: the same math as
/// [`gemv_q5_kx8_q8_k_scalar_8`], per quad row, reading qs / qh / folded
/// bsums / d out of the pre-interleaved [`Q8KActsX4`]. Bit-identical to
/// running that GEMV per activation, which is what the tests assert.
pub(crate) fn gemm_q5_kx8_acts_x4_scalar_8(
    packed: &[u8],
    tile: &Q8KActsX4,
    n_cols: usize,
    out: &mut [f32],
) {
    let nb = n_cols / Q5_K_BLOCK_ELEMS;
    let blocklen = 8;
    let ncols_interleaved = Q5_KX8_NROWS;
    let na = tile.na;
    let mut sumf = [[0f32; Q5_KX8_NROWS]; Q5_KX8_GEMM_NC];
    let mut sum_minf = [[0f32; Q5_KX8_NROWS]; Q5_KX8_GEMM_NC];
    for l in 0..nb {
        let blk = &packed[l * Q5_KX8_BLOCK_BYTES..][..Q5_KX8_BLOCK_BYTES];
        let d = &blk[0..16];
        let dmin = &blk[16..32];
        let scales = &blk[32..128];
        let qh = &blk[128..384];
        let qs = &blk[384..];
        let q8 = &tile.qs[l * Q5_K_BLOCK_ELEMS * 4..][..Q5_K_BLOCK_ELEMS * 4];

        let mut all_scales = [[0u8; 8]; 8];
        let mut all_mins = [[0u8; 8]; 8];
        for sb in 0..8 {
            decode_scales_mins(&scales[sb * 12..], &mut all_scales[sb], &mut all_mins[sb]);
        }

        for a in 0..na {
            let da = tile.d[l * 4 + a];
            let n_k = Q5_K_BLOCK_ELEMS / (2 * blocklen); // 16
            for k in 0..n_k {
                let sb_pair = k / 4;
                let sc0 = &all_scales[sb_pair * 2];
                let sc1 = &all_scales[sb_pair * 2 + 1];
                let qh_shift = sb_pair * 2;
                for j in 0..ncols_interleaved {
                    let mut sumi = 0i32;
                    for i in 0..blocklen {
                        let b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;
                        let qh_idx = (k * blocklen + i) % 32;
                        let qh_chunk = qh_idx / blocklen;
                        let qh_pos = qh_idx % blocklen;
                        let b_qh_offset =
                            qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;
                        let qh_val = qh[b_qh_offset];
                        let h0 = (qh_val >> qh_shift) & 1;
                        let h1 = (qh_val >> (qh_shift + 1)) & 1;
                        let v0 = ((qs[b_qs_offset] & 0x0F) | (h0 << 4)) as i32;
                        let v1 = ((qs[b_qs_offset] >> 4) | (h1 << 4)) as i32;
                        // Canonical q8 element `e` lives at run `e/8`, row
                        // `a`, lane `e%8` of the interleaved block.
                        let e0 = (k >> 2) * 64 + (k % 4) * blocklen + i;
                        let e1 = e0 + 32;
                        let a0 = q8[(e0 / 8) * 32 + a * 8 + (e0 % 8)] as i32;
                        let a1 = q8[(e1 / 8) * 32 + a * 8 + (e1 % 8)] as i32;
                        sumi += v0 * a0 * sc0[j] as i32 + v1 * a1 * sc1[j] as i32;
                    }
                    sumf[a][j] += sumi as f32 * f16_from_bytes(&d[j * 2..]) * da;
                }
            }
            for (sb, mins) in all_mins.iter().enumerate() {
                let bsum = tile.bsums[(l * 4 + a) * 8 + sb] as i32;
                for j in 0..ncols_interleaved {
                    sum_minf[a][j] +=
                        mins[j] as f32 * bsum as f32 * f16_from_bytes(&dmin[j * 2..]) * da;
                }
            }
        }
    }
    for j in 0..ncols_interleaved {
        for (a, row) in sumf.iter().take(na).enumerate() {
            out[j * na + a] = row[j] - sum_minf[a][j];
        }
    }
}

pub(crate) fn gemm_q5_kx8_q8_k_scalar_4(
    packed: &[u8],
    acts: &[Q8KActivations],
    n_cols: usize,
    out: &mut [f32],
) {
    let na = acts.len();
    let nb = n_cols / Q5_K_BLOCK_ELEMS;
    let blocklen = 4;
    let ncols = Q5_KX8_NROWS;
    out.fill(0.0);
    let mut sum_minf = vec![0f32; ncols * na];
    for l in 0..nb {
        let blk = &packed[l * Q5_KX8_BLOCK_BYTES..][..Q5_KX8_BLOCK_BYTES];
        let d = &blk[0..16];
        let dmin = &blk[16..32];
        let scales = &blk[32..128];
        let qh = &blk[128..384];
        let qs = &blk[384..];
        let mut all_scales = [[0u8; 8]; 8];
        let mut all_mins = [[0u8; 8]; 8];
        for sb in 0..8 {
            decode_scales_mins(&scales[sb * 12..], &mut all_scales[sb], &mut all_mins[sb]);
        }
        let n_k = Q5_K_BLOCK_ELEMS / (2 * blocklen);
        for (a, act) in acts.iter().enumerate() {
            let da = act.d[l];
            let q8 = &act.q[l * Q5_K_BLOCK_ELEMS..(l + 1) * Q5_K_BLOCK_ELEMS];
            let bsums = &act.bsums[l * 16..(l + 1) * 16];
            for k in 0..n_k {
                let sb_pair = k / 8;
                let sc0 = &all_scales[sb_pair * 2];
                let sc1 = &all_scales[sb_pair * 2 + 1];
                let qh_shift = sb_pair * 2;
                for j in 0..ncols {
                    let mut sumi = 0i32;
                    for i in 0..blocklen {
                        let b_qs_offset = k * ncols * blocklen + j * blocklen + i;
                        let qh_idx = (k * blocklen + i) % 32;
                        let qh_chunk = qh_idx / blocklen;
                        let qh_pos = qh_idx % blocklen;
                        let b_qh_offset = qh_chunk * (blocklen * ncols) + j * blocklen + qh_pos;
                        let qh_val = qh[b_qh_offset];
                        let h0 = (qh_val >> qh_shift) & 1;
                        let h1 = (qh_val >> (qh_shift + 1)) & 1;
                        let v0 = ((qs[b_qs_offset] & 0x0F) | (h0 << 4)) as i32;
                        let v1 = ((qs[b_qs_offset] >> 4) | (h1 << 4)) as i32;
                        let a0 = q8[(k / 8) * 64 + (k % 8) * blocklen + i] as i32;
                        let a1 = q8[(k / 8) * 64 + (k % 8) * blocklen + i + 32] as i32;
                        sumi += v0 * a0 * sc0[j] as i32 + v1 * a1 * sc1[j] as i32;
                    }
                    out[j * na + a] += sumi as f32 * f16_from_bytes(&d[j * 2..]) * da;
                }
            }
            for sb in 0..8 {
                let mins = &all_mins[sb];
                let bsum = bsums[sb * 2] as i32 + bsums[sb * 2 + 1] as i32;
                for j in 0..ncols {
                    sum_minf[j * na + a] +=
                        mins[j] as f32 * bsum as f32 * f16_from_bytes(&dmin[j * 2..]) * da;
                }
            }
        }
    }
    for i in 0..ncols * na {
        out[i] -= sum_minf[i];
    }
}

pub(crate) fn gemm_q5_kx8_q8_k_scalar_8(
    packed: &[u8],
    acts: &[Q8KActivations],
    n_cols: usize,
    out: &mut [f32],
) {
    let na = acts.len();
    let nb = n_cols / Q5_K_BLOCK_ELEMS;
    let blocklen = 8;
    let ncols = Q5_KX8_NROWS;
    out.fill(0.0);
    let mut sum_minf = vec![0f32; ncols * na];
    for l in 0..nb {
        let blk = &packed[l * Q5_KX8_BLOCK_BYTES..][..Q5_KX8_BLOCK_BYTES];
        let d = &blk[0..16];
        let dmin = &blk[16..32];
        let scales = &blk[32..128];
        let qh = &blk[128..384];
        let qs = &blk[384..];
        let mut all_scales = [[0u8; 8]; 8];
        let mut all_mins = [[0u8; 8]; 8];
        for sb in 0..8 {
            decode_scales_mins(&scales[sb * 12..], &mut all_scales[sb], &mut all_mins[sb]);
        }
        let n_k = Q5_K_BLOCK_ELEMS / (2 * blocklen);
        for (a, act) in acts.iter().enumerate() {
            let da = act.d[l];
            let q8 = &act.q[l * Q5_K_BLOCK_ELEMS..(l + 1) * Q5_K_BLOCK_ELEMS];
            let bsums = &act.bsums[l * 16..(l + 1) * 16];
            for k in 0..n_k {
                let sb_pair = k / 4;
                let sc0 = &all_scales[sb_pair * 2];
                let sc1 = &all_scales[sb_pair * 2 + 1];
                let qh_shift = sb_pair * 2;
                for j in 0..ncols {
                    let mut sumi = 0i32;
                    for i in 0..blocklen {
                        let b_qs_offset = k * ncols * blocklen + j * blocklen + i;
                        let qh_idx = (k * blocklen + i) % 32;
                        let qh_chunk = qh_idx / blocklen;
                        let qh_pos = qh_idx % blocklen;
                        let b_qh_offset = qh_chunk * (blocklen * ncols) + j * blocklen + qh_pos;
                        let qh_val = qh[b_qh_offset];
                        let h0 = (qh_val >> qh_shift) & 1;
                        let h1 = (qh_val >> (qh_shift + 1)) & 1;
                        let v0 = ((qs[b_qs_offset] & 0x0F) | (h0 << 4)) as i32;
                        let v1 = ((qs[b_qs_offset] >> 4) | (h1 << 4)) as i32;
                        let a0 = q8[(k >> 2) * 64 + (k % 4) * blocklen + i] as i32;
                        let a1 = q8[(k >> 2) * 64 + (k % 4) * blocklen + i + 32] as i32;
                        sumi += v0 * a0 * sc0[j] as i32 + v1 * a1 * sc1[j] as i32;
                    }
                    out[j * na + a] += sumi as f32 * f16_from_bytes(&d[j * 2..]) * da;
                }
            }
            for sb in 0..8 {
                let mins = &all_mins[sb];
                let bsum = bsums[sb * 2] as i32 + bsums[sb * 2 + 1] as i32;
                for j in 0..ncols {
                    sum_minf[j * na + a] +=
                        mins[j] as f32 * bsum as f32 * f16_from_bytes(&dmin[j * 2..]) * da;
                }
            }
        }
    }
    for i in 0..ncols * na {
        out[i] -= sum_minf[i];
    }
}
