use super::*;
use crate::repack::*;
use crate::{
    dot_q8_0_q8_scalar, quantize_activations_q8, Q8Activations, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};

#[test]
fn q8_0x4_pack_and_gemv_matches_scalar_row_dots() {
    let n_blocks = 3;
    let cols = n_blocks * Q8_0_BLOCK_ELEMS;
    let rows = 12; // three full groups of 4
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q8_0_row(n_blocks, r as u8));
    }
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32) * 0.023 - 1.4).cos() * 2.2)
        .collect();
    let act = quantize_activations_q8(&x);

    let row_bytes = n_blocks * Q8_0_BLOCK_BYTES;
    let mut reference = vec![0f32; rows];
    for r in 0..rows {
        reference[r] = dot_q8_0_q8_scalar(&matrix[r * row_bytes..(r + 1) * row_bytes], &act);
    }

    for &interleave in &[4usize, 8] {
        let packed = pack_q8_0_matrix_x4(&matrix, rows, cols, interleave);
        let n_groups = rows / Q8_0X4_NROWS;
        let mut out = vec![0f32; rows];
        gemv_q8_0x4_q8_0(&packed, &act, cols, n_groups, interleave, &mut out);
        for r in 0..rows {
            let err = (out[r] - reference[r]).abs();
            let scale = reference[r].abs().max(1.0);
            assert!(
                err / scale < 1e-4 || err < 1e-3,
                "Q8_0x4 interleave={interleave} row {r}: got {} want {} err={err}",
                out[r],
                reference[r]
            );
        }
    }
}

/// The GEMM exists purely to reuse weight loads across activations,
/// so it must produce exactly what the per-activation GEMV produces
/// -- not merely something close. Any divergence would be a
/// batch-size-dependent numeric difference, i.e. prefill and decode
/// disagreeing about the same prompt.
#[test]
fn q8_0x4_gemm_matches_the_gemv_run_once_per_activation() {
    let n_blocks = 4;
    let cols = n_blocks * Q8_0_BLOCK_ELEMS;
    let rows = 8;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q8_0_row(n_blocks, (r * 3 + 1) as u8));
    }
    let packed = pack_q8_0_matrix_x4(&matrix, rows, cols, Q8_0X4_INTERLEAVE);

    // Deliberately not a multiple of the tile width, so the tail
    // path is covered too.
    let n_acts = 7;
    let acts: Vec<Q8Activations> = (0..n_acts)
        .map(|j| {
            let x: Vec<f32> = (0..cols)
                .map(|i| (((i + j * 13) as f32) * 0.017 - 0.9).sin() * 1.7)
                .collect();
            quantize_activations_q8(&x)
        })
        .collect();

    for group in 0..rows / Q8_0X4_NROWS {
        let mut gemm_out = vec![0f32; Q8_0X4_NROWS * n_acts];
        gemm_q8_0x4_group(
            &packed,
            group,
            &acts,
            cols,
            Q8_0X4_INTERLEAVE,
            &mut gemm_out,
        );

        for (j, act) in acts.iter().enumerate() {
            let mut gemv_out = [0f32; Q8_0X4_NROWS];
            gemv_q8_0x4_group(&packed, group, act, cols, Q8_0X4_INTERLEAVE, &mut gemv_out);
            for r in 0..Q8_0X4_NROWS {
                assert_eq!(
                    gemm_out[r * n_acts + j],
                    gemv_out[r],
                    "group {group} row {r} act {j}: GEMM and GEMV disagree"
                );
            }
        }
    }
}

#[test]
fn q8_0x4_gemm_x4_portable_is_bit_exact_vs_scalar_gemv() {
    let n_blocks = 3;
    let cols = n_blocks * Q8_0_BLOCK_ELEMS;
    let rows = Q8_0X4_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q8_0_row(n_blocks, (r * 7 + 3) as u8));
    }
    let packed = pack_q8_0_matrix_x4(&matrix, rows, cols, 8);

    for na in 1..=Q8K_ACTS_X4_NC {
        let acts = synth_q8_0_acts(na, cols);
        let tile = prepare_q8_acts_x4(&acts, cols);
        let mut got = vec![0f32; Q8_0X4_NROWS * na];
        gemm_q8_0x4_acts_x4_scalar_8(&packed, &tile, cols, &mut got);

        for (j, act) in acts.iter().enumerate() {
            let mut want = [0f32; Q8_0X4_NROWS];
            gemv_q8_0x4_q8_0_scalar(&packed, act, cols, 1, 8, &mut want);
            for r in 0..Q8_0X4_NROWS {
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

#[test]
fn q8_0x4_gemm_with_no_activations_is_a_no_op() {
    let n_blocks = 2;
    let cols = n_blocks * Q8_0_BLOCK_ELEMS;
    let mut matrix = Vec::new();
    for r in 0..Q8_0X4_NROWS {
        matrix.extend_from_slice(&synth_q8_0_row(n_blocks, r as u8));
    }
    let packed = pack_q8_0_matrix_x4(&matrix, Q8_0X4_NROWS, cols, Q8_0X4_INTERLEAVE);
    let mut out: Vec<f32> = Vec::new();
    gemm_q8_0x4_group(&packed, 0, &[], cols, Q8_0X4_INTERLEAVE, &mut out);
    assert!(out.is_empty());
}
