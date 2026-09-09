use super::*;
use crate::repack::*;
use crate::{
    dot_q5_k_q8_scalar, quantize_activations_q8_k, Q8KActivations, Q5_K_BLOCK_BYTES,
    Q5_K_BLOCK_ELEMS,
};

#[test]
fn q5_kx8_gemm_x4_portable_is_bit_exact_vs_scalar_gemv() {
    let n_blocks = 3;
    let cols = n_blocks * Q5_K_BLOCK_ELEMS;
    let rows = Q5_KX8_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q5_k_row(n_blocks, (r * 5 + 3) as u8));
    }
    let packed = pack_q5_k_matrix_x8(&matrix, rows, cols, 8);

    for na in 1..=Q5_KX8_GEMM_NC {
        let acts = synth_q8_k_acts(na, cols);
        let tile = prepare_q8_k_acts_x4(&acts, cols);
        let mut got = vec![0f32; Q5_KX8_NROWS * na];
        gemm_q5_kx8_acts_x4_scalar_8(&packed, &tile, cols, &mut got);

        for (j, act) in acts.iter().enumerate() {
            let mut want = [0f32; Q5_KX8_NROWS];
            gemv_q5_kx8_q8_k_scalar_8(&packed, act, cols, 1, &mut want);
            for r in 0..Q5_KX8_NROWS {
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

/// The interleave-8 NEON paths (DotProd 8x8 GEMV, i8mm GEMM) against
/// the scalar interleave-8 references, plus the x4 entry against the
/// compat entry (bit-exact -- they share the kernel).
#[test]
#[cfg(target_arch = "aarch64")]
fn q5_kx8_interleave8_neon_matches_references_when_available() {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        return;
    }
    let n_blocks = 3;
    let cols = n_blocks * Q5_K_BLOCK_ELEMS;
    let n_groups = 2;
    let rows = n_groups * Q5_KX8_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q5_k_row(n_blocks, (r * 7 + 1) as u8));
    }
    let packed = pack_q5_k_matrix_x8(&matrix, rows, cols, 8);

    let acts = synth_q8_k_acts(4, cols);
    for act in &acts {
        let mut got = vec![0f32; rows];
        gemv_q5_kx8_q8_k(&packed, act, cols, n_groups, 8, &mut got);
        let mut want = vec![0f32; rows];
        gemv_q5_kx8_q8_k_scalar_8(&packed, act, cols, n_groups, &mut want);
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
    for na in 1..=Q5_KX8_GEMM_NC {
        let acts = synth_q8_k_acts(na, cols);
        let tile = prepare_q8_k_acts_x4(&acts, cols);
        for group in 0..n_groups {
            let mut x4_out = vec![0f32; Q5_KX8_NROWS * na];
            gemm_q5_kx8_group_x4(&packed, group, &tile, cols, 8, &mut x4_out);

            let mut group_out = vec![0f32; Q5_KX8_NROWS * na];
            gemm_q5_kx8_group(&packed, group, &acts, cols, 8, &mut group_out);
            assert_eq!(
                x4_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                group_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "x4 entry diverged from the compat entry, group {group} na {na}"
            );

            let nb = cols / Q5_K_BLOCK_ELEMS;
            let slice = &packed[group * nb * Q5_KX8_BLOCK_BYTES..][..nb * Q5_KX8_BLOCK_BYTES];
            let mut want = vec![0f32; Q5_KX8_NROWS * na];
            gemm_q5_kx8_acts_x4_scalar_8(slice, &tile, cols, &mut want);
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
fn q5_kx8_pack_and_gemv_matches_scalar_row_dots() {
    let n_blocks = 2;
    let cols = n_blocks * Q5_K_BLOCK_ELEMS;
    let rows = 16;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q5_k_row(n_blocks, r as u8));
    }
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32) * 0.019 - 1.8).sin() * 1.6)
        .collect();
    let act = quantize_activations_q8_k(&x);

    let row_bytes = n_blocks * Q5_K_BLOCK_BYTES;
    let mut reference = vec![0f32; rows];
    for r in 0..rows {
        reference[r] = dot_q5_k_q8_scalar(&matrix[r * row_bytes..(r + 1) * row_bytes], &act);
    }

    for &interleave in &[4usize, 8] {
        let packed = pack_q5_k_matrix_x8(&matrix, rows, cols, interleave);
        let n_groups = rows / Q5_KX8_NROWS;
        let mut out = vec![0f32; rows];
        gemv_q5_kx8_q8_k(&packed, &act, cols, n_groups, interleave, &mut out);
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
fn q5_kx8_gemm_matches_the_gemv_run_once_per_activation() {
    let n_blocks = 3;
    let cols = n_blocks * Q5_K_BLOCK_ELEMS;
    let rows = 2 * Q5_KX8_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q5_k_row(n_blocks, (r * 5 + 3) as u8));
    }
    let interleave = q5_kx8_interleave();
    let packed = pack_q5_k_matrix_x8(&matrix, rows, cols, interleave);

    let n_acts = 6;
    let acts: Vec<Q8KActivations> = (0..n_acts)
        .map(|j| {
            let x: Vec<f32> = (0..cols)
                .map(|i| (((i + j * 29) as f32) * 0.011 - 0.4).cos() * 2.3)
                .collect();
            quantize_activations_q8_k(&x)
        })
        .collect();

    let row_bytes = n_blocks * Q5_K_BLOCK_BYTES;
    for group in 0..rows / Q5_KX8_NROWS {
        for chunk in acts.chunks(Q5_KX8_GEMM_NC) {
            let mut gemm_out = vec![0f32; Q5_KX8_NROWS * chunk.len()];
            gemm_q5_kx8_group(&packed, group, chunk, cols, interleave, &mut gemm_out);

            for (j, act) in chunk.iter().enumerate() {
                let mut gemv_out = [0f32; Q5_KX8_NROWS];
                gemv_q5_kx8_group(&packed, group, act, cols, interleave, &mut gemv_out);
                for r in 0..Q5_KX8_NROWS {
                    let got = gemm_out[r * chunk.len() + j];
                    let want = gemv_out[r];
                    // Tolerance, not bit equality: on i8mm hosts the
                    // GEMM (i8mm, per-block f32 accumulation) and the
                    // GEMV (DotProd, per-sub-block) round differently.
                    // Measured on an M2 Pro: worst observed relative
                    // deviation 2.9e-5, everything else <= 4.8e-6.
                    let err = (got - want).abs();
                    let scale = want.abs().max(1.0);
                    assert!(
                        err / scale < 5e-5 || err < 1e-3,
                        "group {group} row {r} act {j}: Q5_K GEMM {got} vs GEMV {want}"
                    );
                }
            }
        }

        // Also check against per-row dot_q5_k_q8 for each activation.
        for (j, act) in acts.iter().enumerate() {
            for r in 0..Q5_KX8_NROWS {
                let row_idx = group * Q5_KX8_NROWS + r;
                let row = &matrix[row_idx * row_bytes..(row_idx + 1) * row_bytes];
                let want = dot_q5_k_q8_scalar(row, act);
                let mut gemv_out = [0f32; Q5_KX8_NROWS];
                gemv_q5_kx8_group(&packed, group, act, cols, interleave, &mut gemv_out);
                let err = (gemv_out[r] - want).abs();
                let scale = want.abs().max(1.0);
                assert!(
                    err / scale < 1e-4 || err < 1e-3,
                    "group {group} row {r} act {j}: packed gemv {} vs dot {want}",
                    gemv_out[r]
                );
            }
        }
    }
}

#[test]
fn q5_kx8_gemm_with_no_activations_is_a_no_op() {
    let n_blocks = 2;
    let cols = n_blocks * Q5_K_BLOCK_ELEMS;
    let mut matrix = Vec::new();
    for r in 0..Q5_KX8_NROWS {
        matrix.extend_from_slice(&synth_q5_k_row(n_blocks, r as u8));
    }
    let packed = pack_q5_k_matrix_x8(&matrix, Q5_KX8_NROWS, cols, 4);
    let mut out: Vec<f32> = Vec::new();
    gemm_q5_kx8_group(&packed, 0, &[], cols, 4, &mut out);
    assert!(out.is_empty());
}
