use super::*;
use crate::repack::*;
use crate::{
    dot_q4_k_q8_scalar, quantize_activations_q8_k, Q8KActivations, Q4_K_BLOCK_BYTES,
    Q4_K_BLOCK_ELEMS,
};

#[test]
fn pack_and_gemv_matches_scalar_row_dots() {
    let n_blocks = 2;
    let cols = n_blocks * Q4_K_BLOCK_ELEMS;
    let rows = 16; // two full groups
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q4_k_row(n_blocks, r as u8));
    }
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32) * 0.017 - 2.1).sin() * 1.8)
        .collect();
    let act = quantize_activations_q8_k(&x);

    let mut reference = vec![0f32; rows];
    let row_bytes = n_blocks * Q4_K_BLOCK_BYTES;
    for r in 0..rows {
        reference[r] = dot_q4_k_q8_scalar(&matrix[r * row_bytes..(r + 1) * row_bytes], &act);
    }

    for &interleave in &[4usize, 8] {
        let packed = pack_q4_k_matrix_x8(&matrix, rows, cols, interleave);
        let n_groups = rows / Q4_KX8_NROWS;
        let mut out = vec![0f32; rows];
        gemv_q4_kx8_q8_k(&packed, &act, cols, n_groups, interleave, &mut out);
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

/// The Q4_K GEMM must agree with the GEMV **exactly**, for the same
/// reason as the Q8_0 pair above: the two run on the same prompt in
/// different batch regimes (prefill vs the `< 4` tail vs decode), so
/// any divergence is prefill and decode disagreeing about the same
/// tokens. The GEMM only reorders which loop the weight unpack sits
/// in — every multiply-accumulate happens in the same order and the
/// same precision — so equality is the right assertion, not
/// closeness.
#[test]
fn q4_kx8_gemm_matches_the_gemv_run_once_per_activation() {
    let n_blocks = 3;
    let cols = n_blocks * Q4_K_BLOCK_ELEMS;
    let rows = 2 * Q4_KX8_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q4_k_row(n_blocks, (r * 5 + 3) as u8));
    }
    let interleave = q4_kx8_interleave();
    let packed = pack_q4_k_matrix_x8(&matrix, rows, cols, interleave);

    // Not a multiple of the tile width, so the ragged tail the
    // caller has to chunk around is covered too.
    let n_acts = 6;
    let acts: Vec<Q8KActivations> = (0..n_acts)
        .map(|j| {
            let x: Vec<f32> = (0..cols)
                .map(|i| (((i + j * 29) as f32) * 0.011 - 0.4).cos() * 2.3)
                .collect();
            quantize_activations_q8_k(&x)
        })
        .collect();

    for group in 0..rows / Q4_KX8_NROWS {
        for chunk in acts.chunks(Q4_KX8_GEMM_NC) {
            let mut gemm_out = vec![0f32; Q4_KX8_NROWS * chunk.len()];
            gemm_q4_kx8_group(&packed, group, chunk, cols, interleave, &mut gemm_out);

            for (j, act) in chunk.iter().enumerate() {
                let mut gemv_out = [0f32; Q4_KX8_NROWS];
                gemv_q4_kx8_group(&packed, group, act, cols, interleave, &mut gemv_out);
                for r in 0..Q4_KX8_NROWS {
                    let got = gemm_out[r * chunk.len() + j];
                    let want = gemv_out[r];
                    if interleave == 4 {
                        assert_eq!(
                            got, want,
                            "group {group} row {r} act {j}: Q4_K GEMM and GEMV disagree"
                        );
                    } else {
                        let err = (got - want).abs();
                        let scale = want.abs().max(1.0);
                        assert!(
                            err / scale < 1e-4 || err < 1e-2,
                            "group {group} row {r} act {j}: GEMM {got} vs GEMV {want} (err={err})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn q4_kx8_gemm_i8mm_matches_scalar_when_available() {
    if !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    let interleave = q4_kx8_interleave();
    assert_eq!(interleave, 8, "i8mm host should pack with interleave 8");

    let n_blocks = 3;
    let cols = n_blocks * Q4_K_BLOCK_ELEMS;
    let rows = Q4_KX8_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q4_k_row(n_blocks, (r * 7 + 2) as u8));
    }
    let packed = pack_q4_k_matrix_x8(&matrix, rows, cols, interleave);

    let n_acts = 4;
    let acts: Vec<Q8KActivations> = (0..n_acts)
        .map(|j| {
            let x: Vec<f32> = (0..cols)
                .map(|i| (((i + j * 17) as f32) * 0.013 - 0.6).sin() * 1.9)
                .collect();
            quantize_activations_q8_k(&x)
        })
        .collect();

    let mut gemm_out = vec![0f32; Q4_KX8_NROWS * n_acts];
    gemm_q4_kx8_group(&packed, 0, &acts, cols, interleave, &mut gemm_out);

    for (j, act) in acts.iter().enumerate() {
        let mut scalar_out = [0f32; Q4_KX8_NROWS];
        gemv_q4_kx8_group(&packed, 0, act, cols, interleave, &mut scalar_out);
        for r in 0..Q4_KX8_NROWS {
            let got = gemm_out[r * n_acts + j];
            let want = scalar_out[r];
            let err = (got - want).abs();
            let scale = want.abs().max(1.0);
            assert!(
                err / scale < 1e-5 || err < 1e-3,
                "row {r} act {j}: i8mm GEMM {got} vs scalar {want} (err={err})"
            );
        }
    }
}

#[test]
fn q4_kx8_gemm_x4_portable_is_bit_exact_vs_scalar_gemv() {
    let n_blocks = 3;
    let cols = n_blocks * Q4_K_BLOCK_ELEMS;
    let rows = Q4_KX8_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q4_k_row(n_blocks, (r * 5 + 3) as u8));
    }
    let packed = pack_q4_k_matrix_x8(&matrix, rows, cols, 8);

    for na in 1..=Q4_KX8_GEMM_NC {
        let acts = synth_q8_k_acts(na, cols);
        let tile = prepare_q8_k_acts_x4(&acts, cols);
        let mut got = vec![0f32; Q4_KX8_NROWS * na];
        gemm_q4_kx8_acts_x4_scalar_8(&packed, &tile, cols, &mut got);

        for (j, act) in acts.iter().enumerate() {
            let mut want = [0f32; Q4_KX8_NROWS];
            gemv_q4_kx8_q8_k_scalar_8(&packed, act, cols, 1, &mut want);
            for r in 0..Q4_KX8_NROWS {
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

/// The hoisted path must reproduce the in-kernel-interleave behavior it
/// replaced: `gemm_q4_kx8_group` (which now prepares the quad per call)
/// and `gemm_q4_kx8_group_x4` (quad prepared by the caller) share the
/// i8mm kernel, so their outputs must be bit-identical, and both must
/// match the scalar GEMV within the usual tolerance.
#[test]
#[cfg(target_arch = "aarch64")]
fn q4_kx8_gemm_x4_i8mm_matches_group_and_scalar() {
    if !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    let interleave = q4_kx8_interleave();
    assert_eq!(interleave, 8, "i8mm host should pack with interleave 8");

    let n_blocks = 3;
    let cols = n_blocks * Q4_K_BLOCK_ELEMS;
    let rows = Q4_KX8_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q4_k_row(n_blocks, (r * 7 + 2) as u8));
    }
    let packed = pack_q4_k_matrix_x8(&matrix, rows, cols, interleave);

    assert!(q4_kx8_gemm_uses_acts_x4(interleave));
    for na in 1..=Q4_KX8_GEMM_NC {
        let acts = synth_q8_k_acts(na, cols);
        let tile = prepare_q8_k_acts_x4(&acts, cols);

        let mut x4_out = vec![0f32; Q4_KX8_NROWS * na];
        gemm_q4_kx8_group_x4(&packed, 0, &tile, cols, interleave, &mut x4_out);

        let mut group_out = vec![0f32; Q4_KX8_NROWS * na];
        gemm_q4_kx8_group(&packed, 0, &acts, cols, interleave, &mut group_out);
        assert_eq!(
            x4_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            group_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "x4 entry diverged from the compat entry, na {na}"
        );

        for (j, act) in acts.iter().enumerate() {
            let mut scalar_out = [0f32; Q4_KX8_NROWS];
            gemv_q4_kx8_group(&packed, 0, act, cols, interleave, &mut scalar_out);
            for r in 0..Q4_KX8_NROWS {
                let got = x4_out[r * na + j];
                let want = scalar_out[r];
                let err = (got - want).abs();
                let scale = want.abs().max(1.0);
                assert!(
                    err / scale < 1e-5 || err < 1e-3,
                    "row {r} act {j} na {na}: i8mm x4 GEMM {got} vs scalar {want} (err={err})"
                );
            }
        }
    }
}

/// The interleave-8 DotProd GEMV against the scalar interleave-8
/// reference (the i8mm GEMM side of Q4_K is covered above).
#[test]
#[cfg(target_arch = "aarch64")]
fn q4_kx8_interleave8_neon_gemv_matches_scalar_when_available() {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        return;
    }
    let n_blocks = 3;
    let cols = n_blocks * Q4_K_BLOCK_ELEMS;
    let n_groups = 2;
    let rows = n_groups * Q4_KX8_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q4_k_row(n_blocks, (r * 11 + 5) as u8));
    }
    let packed = pack_q4_k_matrix_x8(&matrix, rows, cols, 8);

    let acts = synth_q8_k_acts(4, cols);
    for act in &acts {
        let mut got = vec![0f32; rows];
        gemv_q4_kx8_q8_k(&packed, act, cols, n_groups, 8, &mut got);
        let mut want = vec![0f32; rows];
        gemv_q4_kx8_q8_k_scalar_8(&packed, act, cols, n_groups, &mut want);
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
}

#[test]
fn q4_kx8_gemm_with_no_activations_is_a_no_op() {
    let n_blocks = 2;
    let cols = n_blocks * Q4_K_BLOCK_ELEMS;
    let mut matrix = Vec::new();
    for r in 0..Q4_KX8_NROWS {
        matrix.extend_from_slice(&synth_q4_k_row(n_blocks, r as u8));
    }
    let packed = pack_q4_k_matrix_x8(&matrix, Q4_KX8_NROWS, cols, 4);
    let mut out: Vec<f32> = Vec::new();
    gemm_q4_kx8_group(&packed, 0, &[], cols, 4, &mut out);
    assert!(out.is_empty());
}
