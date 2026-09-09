use super::*;
use crate::repack::*;
use crate::{dot_q6_k_q8_scalar, quantize_activations_q8_k, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS};

#[test]
fn q6_kx8_gemm_x4_portable_is_bit_exact_vs_scalar_gemv() {
    let n_blocks = 2;
    let cols = n_blocks * Q6_K_BLOCK_ELEMS;
    let rows = Q6_KX8_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q6_k_row(n_blocks, (r * 3 + 2) as u8));
    }
    let packed = pack_q6_k_matrix_x8(&matrix, rows, cols, 8);

    for na in 1..=Q8K_ACTS_X4_NC {
        let acts = synth_q8_k_acts(na, cols);
        let tile = prepare_q8_k_acts_x4(&acts, cols);
        let mut got = vec![0f32; Q6_KX8_NROWS * na];
        gemm_q6_kx8_acts_x4_scalar_8(&packed, &tile, cols, &mut got);

        for (j, act) in acts.iter().enumerate() {
            let mut want = [0f32; Q6_KX8_NROWS];
            gemv_q6_kx8_q8_k_scalar(&packed, act, cols, 1, 8, &mut want);
            for r in 0..Q6_KX8_NROWS {
                assert_eq!(
                    got[r * na + j].to_bits(),
                    want[r].to_bits(),
                    "row {r} act {j} na {na}: x4 {} vs GEMV {}",
                    got[r * na + j],
                    want[r]
                );
            }
        }
    }
}

/// Q6_K twin of the test above.
#[test]
#[cfg(target_arch = "aarch64")]
fn q6_kx8_interleave8_neon_matches_references_when_available() {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        return;
    }
    let n_blocks = 2;
    let cols = n_blocks * Q6_K_BLOCK_ELEMS;
    let n_groups = 2;
    let rows = n_groups * Q6_KX8_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q6_k_row(n_blocks, (r * 9 + 4) as u8));
    }
    let packed = pack_q6_k_matrix_x8(&matrix, rows, cols, 8);

    let acts = synth_q8_k_acts(4, cols);
    for act in &acts {
        let mut got = vec![0f32; rows];
        gemv_q6_kx8_q8_k(&packed, act, cols, n_groups, 8, &mut got);
        let mut want = vec![0f32; rows];
        gemv_q6_kx8_q8_k_scalar(&packed, act, cols, n_groups, 8, &mut want);
        for r in 0..rows {
            let err = (got[r] - want[r]).abs();
            // Tolerance, not bit equality: the int8 dots are exact in
            // i32, but the two paths fold the per-sub-block f32
            // scales in a different order, so the f32 accumulation
            // rounds differently. Measured on an M2 Pro (the first
            // i8mm host this ever ran on): 63 of 64 outputs agree to
            // <= 6.3e-6 relative, one to 2.6e-5. 5e-5 keeps a margin
            // over that while still being orders of magnitude
            // tighter than any indexing error, which moves a result
            // by the size of the data. Cross-checked end to end:
            // greedy CPU generation with these kernels is
            // token-identical to `FERROX_CPU_INT_DOT=0`.
            assert!(
                err / want[r].abs().max(1.0) < 5e-5 || err < 1e-3,
                "gemv row {r}: NEON 8x8 {} vs scalar {}",
                got[r],
                want[r]
            );
        }
    }

    if !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    for na in 1..=Q8K_ACTS_X4_NC {
        let acts = synth_q8_k_acts(na, cols);
        let tile = prepare_q8_k_acts_x4(&acts, cols);
        for group in 0..n_groups {
            let mut x4_out = vec![0f32; Q6_KX8_NROWS * na];
            gemm_q6_kx8_group_x4(&packed, group, &tile, cols, 8, &mut x4_out);

            let mut group_out = vec![0f32; Q6_KX8_NROWS * na];
            gemm_q6_kx8_group(&packed, group, &acts, cols, 8, &mut group_out);
            assert_eq!(
                x4_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                group_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "x4 entry diverged from the compat entry, group {group} na {na}"
            );

            let nb = cols / Q6_K_BLOCK_ELEMS;
            let slice = &packed[group * nb * Q6_KX8_BLOCK_BYTES..][..nb * Q6_KX8_BLOCK_BYTES];
            let mut want = vec![0f32; Q6_KX8_NROWS * na];
            gemm_q6_kx8_acts_x4_scalar_8(slice, &tile, cols, &mut want);
            for (got, want) in x4_out.iter().zip(want.iter()) {
                let err = (got - want).abs();
                // Same 5e-5 as the GEMV check above, for the same
                // reason: i8mm folds the f32 scales in a different
                // order than the portable path, and the worst
                // deviation measured on an i8mm host is 2.6e-5.
                assert!(
                    err / want.abs().max(1.0) < 5e-5 || err < 1e-3,
                    "group {group} na {na}: i8mm GEMM {got} vs portable {want}"
                );
            }
        }
    }
}

#[test]
fn q6_kx8_pack_and_gemv_matches_scalar_row_dots() {
    let n_blocks = 3;
    let cols = n_blocks * Q6_K_BLOCK_ELEMS;
    let rows = 2 * Q6_KX8_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q6_k_row(n_blocks, (r * 5 + 1) as u8));
    }
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32) * 0.017 - 0.8).cos() * 1.8)
        .collect();
    let act = quantize_activations_q8_k(&x);
    let row_bytes = n_blocks * Q6_K_BLOCK_BYTES;
    let mut reference = vec![0f32; rows];
    for r in 0..rows {
        reference[r] = dot_q6_k_q8_scalar(&matrix[r * row_bytes..(r + 1) * row_bytes], &act);
    }
    for interleave in [4usize, 8] {
        let packed = pack_q6_k_matrix_x8(&matrix, rows, cols, interleave);
        let n_groups = rows / Q6_KX8_NROWS;
        let mut out = vec![0f32; rows];
        gemv_q6_kx8_q8_k(&packed, &act, cols, n_groups, interleave, &mut out);
        for r in 0..rows {
            let err = (out[r] - reference[r]).abs();
            let scale = reference[r].abs().max(1.0);
            assert!(
                err / scale < 1e-4 || err < 1e-3,
                "interleave={interleave} row {r}: got {} want {} err={err}",
                out[r],
                reference[r]
            );
        }
    }
}

#[test]
fn q6_kx8_gemm_matches_the_gemv_run_once_per_activation() {
    let n_blocks = 2;
    let cols = n_blocks * Q6_K_BLOCK_ELEMS;
    let rows = 2 * Q6_KX8_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q6_k_row(n_blocks, (r * 3 + 2) as u8));
    }
    let interleave = q6_kx8_interleave();
    let packed = pack_q6_k_matrix_x8(&matrix, rows, cols, interleave);
    let acts: Vec<_> = (0..Q6_KX8_GEMM_NC)
        .map(|j| {
            let x: Vec<f32> = (0..cols)
                .map(|i| (((i + j * 11) as f32) * 0.015 - 0.7).sin() * 2.0)
                .collect();
            quantize_activations_q8_k(&x)
        })
        .collect();
    let row_bytes = n_blocks * Q6_K_BLOCK_BYTES;
    for group in 0..rows / Q6_KX8_NROWS {
        for chunk in acts.chunks(Q6_KX8_GEMM_NC) {
            let mut gemm_out = vec![0f32; Q6_KX8_NROWS * chunk.len()];
            gemm_q6_kx8_group(&packed, group, chunk, cols, interleave, &mut gemm_out);
            for (j, act) in chunk.iter().enumerate() {
                let mut gemv_out = [0f32; Q6_KX8_NROWS];
                gemv_q6_kx8_group(&packed, group, act, cols, interleave, &mut gemv_out);
                for r in 0..Q6_KX8_NROWS {
                    let got = gemm_out[r * chunk.len() + j];
                    let want = gemv_out[r];
                    let err = (got - want).abs();
                    let scale = want.abs().max(1.0);
                    assert!(
                        err / scale < 1e-4 || err < 1e-3,
                        "group {group} row {r} act {j}: gemm {got} vs gemv {want}"
                    );
                    let row_idx = group * Q6_KX8_NROWS + r;
                    let row = &matrix[row_idx * row_bytes..(row_idx + 1) * row_bytes];
                    let dot = dot_q6_k_q8_scalar(row, act);
                    let err2 = (got - dot).abs();
                    let scale2 = dot.abs().max(1.0);
                    assert!(
                        err2 / scale2 < 1e-4 || err2 < 1e-3,
                        "group {group} row {r} act {j}: gemm {got} vs dot {dot}"
                    );
                }
            }
        }
    }
}
