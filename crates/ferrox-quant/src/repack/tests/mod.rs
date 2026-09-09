//! Shared fixtures for the repack tier's tests, plus the assertions
//! that span more than one kind family. Per-family tests live beside
//! this file, one module per packed layout.

#[cfg(target_arch = "x86_64")]
mod avx2;
mod q4_0x4;
mod q4_kx8;
mod q5_kx8;
mod q6_kx8;
mod q8_0x4;

use crate::repack::*;
use crate::{
    quantize_activations_q8, quantize_activations_q8_k, Q8Activations, Q8KActivations,
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q5_K_BLOCK_BYTES,
    Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
use half::f16;

fn synth_q5_k_row(n_blocks: usize, seed: u8) -> Vec<u8> {
    let mut weights = Vec::with_capacity(n_blocks * Q5_K_BLOCK_BYTES);
    for b in 0..n_blocks {
        weights.extend_from_slice(
            &f16::from_f32(0.05 + (b as f32 + seed as f32) * 0.01).to_le_bytes(),
        );
        weights.extend_from_slice(
            &f16::from_f32(0.01 + (b as f32 + seed as f32) * 0.002).to_le_bytes(),
        );
        for i in 0..12u8 {
            // `wrapping_add`, not `+`: a randomized seed overflows u8
            // here and the panic is the fixture's, not a kernel's.
            weights.push(20u8.wrapping_add(i.wrapping_mul(3)).wrapping_add(seed));
        }
        for i in 0..32u8 {
            weights.push(i.wrapping_mul(11).wrapping_add(b as u8).wrapping_add(seed));
        }
        for i in 0..128u8 {
            weights.push(i.wrapping_mul(19).wrapping_add(b as u8).wrapping_add(seed));
        }
    }
    weights
}

fn synth_q6_k_row(n_blocks: usize, seed: u8) -> Vec<u8> {
    let mut weights = Vec::with_capacity(n_blocks * Q6_K_BLOCK_BYTES);
    for b in 0..n_blocks {
        for i in 0..128u8 {
            weights.push(i.wrapping_mul(17).wrapping_add(b as u8).wrapping_add(seed));
        }
        for i in 0..64u8 {
            weights.push(i.wrapping_mul(13).wrapping_add(seed).wrapping_add(b as u8));
        }
        for i in 0..16u8 {
            // signed scales in -32..31-ish
            weights.push((20i8).wrapping_add(i as i8).wrapping_add(seed as i8) as u8);
        }
        weights.extend_from_slice(
            &f16::from_f32(0.04 + (b as f32 + seed as f32) * 0.008).to_le_bytes(),
        );
    }
    weights
}

fn synth_q4_k_row(n_blocks: usize, seed: u8) -> Vec<u8> {
    let mut weights = Vec::with_capacity(n_blocks * Q4_K_BLOCK_BYTES);
    for b in 0..n_blocks {
        weights.extend_from_slice(
            &f16::from_f32(0.05 + (b as f32 + seed as f32) * 0.01).to_le_bytes(),
        );
        weights.extend_from_slice(
            &f16::from_f32(0.01 + (b as f32 + seed as f32) * 0.002).to_le_bytes(),
        );
        for i in 0..12u8 {
            // `wrapping_add`, not `+`: a randomized seed overflows u8
            // here and the panic is the fixture's, not a kernel's.
            weights.push(20u8.wrapping_add(i.wrapping_mul(3)).wrapping_add(seed));
        }
        for i in 0..128u8 {
            weights.push(i.wrapping_mul(17).wrapping_add(b as u8).wrapping_add(seed));
        }
    }
    weights
}

fn synth_q4_0_row(n_blocks: usize, seed: u8) -> Vec<u8> {
    let mut weights = Vec::with_capacity(n_blocks * Q4_0_BLOCK_BYTES);
    for b in 0..n_blocks {
        weights.extend_from_slice(
            &f16::from_f32(0.05 + (b as f32 + seed as f32) * 0.012).to_le_bytes(),
        );
        for i in 0..16u8 {
            weights.push(i.wrapping_mul(23).wrapping_add(b as u8).wrapping_add(seed));
        }
    }
    weights
}

fn synth_q8_0_row(n_blocks: usize, seed: u8) -> Vec<u8> {
    let mut weights = Vec::with_capacity(n_blocks * Q8_0_BLOCK_BYTES);
    for b in 0..n_blocks {
        weights.extend_from_slice(
            &f16::from_f32(0.04 + (b as f32 + seed as f32) * 0.008).to_le_bytes(),
        );
        for i in 0..32u8 {
            // signed i8 stored as u8 bytes
            let q = ((i as i8)
                .wrapping_mul(3)
                .wrapping_add(seed as i8)
                .wrapping_add(b as i8)) as u8;
            weights.push(q);
        }
    }
    weights
}

fn synth_q8_k_acts(n: usize, cols: usize) -> Vec<Q8KActivations> {
    (0..n)
        .map(|j| {
            let x: Vec<f32> = (0..cols)
                .map(|i| (((i + j * 17) as f32) * 0.013 - 0.6).sin() * 1.9)
                .collect();
            quantize_activations_q8_k(&x)
        })
        .collect()
}

/// The retired per-block interleave (`pack_q8_k_qs_x4_i8`), kept verbatim
/// as the reference `prepare_q8_k_acts_x4` must reproduce: llama.cpp's
/// `ggml_quantize_mat_q8_K_4x8` qs ordering with 8-byte runs.
fn reference_q8_kx4_block_qs(acts: &[Q8KActivations], block: usize) -> [i8; Q4_K_BLOCK_ELEMS * 4] {
    const BLCK: usize = 8;
    let na = acts.len();
    let mut out = [0i8; Q4_K_BLOCK_ELEMS * 4];
    for (j, slot) in out.iter_mut().enumerate() {
        let src_offset = (j / (4 * BLCK)) * BLCK + (j % BLCK);
        let src_id = (j % (4 * BLCK)) / BLCK;
        *slot = if src_id < na {
            acts[src_id].q[block * Q4_K_BLOCK_ELEMS + src_offset]
        } else {
            0
        };
    }
    out
}

#[test]
fn prepare_q8_k_acts_x4_matches_block_interleave_reference() {
    let n_blocks = 3;
    let cols = n_blocks * Q4_K_BLOCK_ELEMS;
    for na in 1..=Q4_KX8_GEMM_NC {
        let acts = synth_q8_k_acts(na, cols);
        let tile = prepare_q8_k_acts_x4(&acts, cols);
        assert_eq!(tile.na, na);
        assert_eq!(tile.n_blocks, n_blocks);
        for b in 0..n_blocks {
            let want_qs = reference_q8_kx4_block_qs(&acts, b);
            assert_eq!(
                &tile.qs[b * Q4_K_BLOCK_ELEMS * 4..][..Q4_K_BLOCK_ELEMS * 4],
                &want_qs[..],
                "qs mismatch, block {b} na {na}"
            );
            for a in 0..4 {
                let act = acts.get(a);
                for i in 0..8 {
                    let want = act.map_or(0, |act| {
                        act.bsums[b * 16 + 2 * i] + act.bsums[b * 16 + 2 * i + 1]
                    });
                    assert_eq!(
                        tile.bsums[(b * 4 + a) * 8 + i],
                        want,
                        "bsums mismatch, block {b} row {a} pair {i} na {na}"
                    );
                }
                let want_d = act.map_or(0.0, |act| act.d[b]);
                assert_eq!(tile.d[b * 4 + a], want_d, "d mismatch, block {b} row {a}");
            }
        }
    }
}

/// `AccelX4` only chooses a kernel; it must never change an answer.
///
/// Two claims, for all five `×4` GEMMs at once. First, passing the
/// host's own [`AccelX4::detect`] to the `_on` entry point is
/// bit-identical to letting the probing wrapper detect per call -- that
/// is what lets `apply_batch` hoist the probe out of a 10^5-iteration
/// loop. Second, [`AccelX4::Portable`] really does select the portable
/// kernel even on an i8mm host, so the scalar reference stays reachable
/// and the `_portable_is_bit_exact_vs_scalar_gemv` tests keep meaning
/// something on this machine.
#[test]
fn accel_x4_only_picks_a_kernel_it_never_changes_the_answer() {
    let na = 3;
    let here = AccelX4::detect();

    // Q4_K / Q5_K / Q6_K share the Q8_K activation quad.
    {
        let n_blocks = 3;
        let cols = n_blocks * Q4_K_BLOCK_ELEMS;
        let acts = synth_q8_k_acts(na, cols);
        let tile = prepare_q8_k_acts_x4(&acts, cols);

        let mut q4 = Vec::new();
        let mut q5 = Vec::new();
        let mut q6 = Vec::new();
        for r in 0..Q4_KX8_NROWS {
            q4.extend_from_slice(&synth_q4_k_row(n_blocks, (r * 7 + 2) as u8));
            q5.extend_from_slice(&synth_q5_k_row(n_blocks, (r * 3 + 5) as u8));
            q6.extend_from_slice(&synth_q6_k_row(n_blocks, (r * 11 + 1) as u8));
        }
        let q4 = pack_q4_k_matrix_x8(&q4, Q4_KX8_NROWS, cols, 8);
        let q5 = pack_q5_k_matrix_x8(&q5, Q5_KX8_NROWS, cols, 8);
        let q6 = pack_q6_k_matrix_x8(&q6, Q6_KX8_NROWS, cols, 8);

        let mut wrapper = vec![0f32; Q4_KX8_NROWS * na];
        let mut hoisted = vec![0f32; Q4_KX8_NROWS * na];
        let mut portable = vec![0f32; Q4_KX8_NROWS * na];
        let mut reference = vec![0f32; Q4_KX8_NROWS * na];

        gemm_q4_kx8_group_x4(&q4, 0, &tile, cols, 8, &mut wrapper);
        gemm_q4_kx8_group_x4_on(&q4, 0, &tile, cols, 8, here, &mut hoisted);
        gemm_q4_kx8_group_x4_on(&q4, 0, &tile, cols, 8, AccelX4::Portable, &mut portable);
        gemm_q4_kx8_acts_x4_scalar_8(&q4, &tile, cols, &mut reference);
        assert_bits_eq("q4_k hoisted", &hoisted, &wrapper);
        assert_bits_eq("q4_k portable", &portable, &reference);

        gemm_q5_kx8_group_x4(&q5, 0, &tile, cols, 8, &mut wrapper);
        gemm_q5_kx8_group_x4_on(&q5, 0, &tile, cols, 8, here, &mut hoisted);
        gemm_q5_kx8_group_x4_on(&q5, 0, &tile, cols, 8, AccelX4::Portable, &mut portable);
        gemm_q5_kx8_acts_x4_scalar_8(&q5, &tile, cols, &mut reference);
        assert_bits_eq("q5_k hoisted", &hoisted, &wrapper);
        assert_bits_eq("q5_k portable", &portable, &reference);

        gemm_q6_kx8_group_x4(&q6, 0, &tile, cols, 8, &mut wrapper);
        gemm_q6_kx8_group_x4_on(&q6, 0, &tile, cols, 8, here, &mut hoisted);
        gemm_q6_kx8_group_x4_on(&q6, 0, &tile, cols, 8, AccelX4::Portable, &mut portable);
        gemm_q6_kx8_acts_x4_scalar_8(&q6, &tile, cols, &mut reference);
        assert_bits_eq("q6_k hoisted", &hoisted, &wrapper);
        assert_bits_eq("q6_k portable", &portable, &reference);
    }

    // Q8_0 / Q4_0 share the Q8_0 activation quad.
    {
        let n_blocks = 4;
        let cols = n_blocks * Q8_0_BLOCK_ELEMS;
        let acts = synth_q8_0_acts(na, cols);
        let tile = prepare_q8_acts_x4(&acts, cols);

        let mut q8 = Vec::new();
        let mut q4 = Vec::new();
        for r in 0..Q8_0X4_NROWS {
            q8.extend_from_slice(&synth_q8_0_row(n_blocks, (r * 9 + 4) as u8));
            q4.extend_from_slice(&synth_q4_0_row(n_blocks, (r * 13 + 6) as u8));
        }
        let q8 = pack_q8_0_matrix_x4(&q8, Q8_0X4_NROWS, cols, 8);
        let q4 = pack_q4_0_matrix_x4(&q4, Q4_0X4_NROWS, cols, 8);

        let mut wrapper = vec![0f32; Q8_0X4_NROWS * na];
        let mut hoisted = vec![0f32; Q8_0X4_NROWS * na];
        let mut portable = vec![0f32; Q8_0X4_NROWS * na];
        let mut reference = vec![0f32; Q8_0X4_NROWS * na];

        gemm_q8_0x4_group_x4(&q8, 0, &tile, cols, 8, &mut wrapper);
        gemm_q8_0x4_group_x4_on(&q8, 0, &tile, cols, 8, here, &mut hoisted);
        gemm_q8_0x4_group_x4_on(&q8, 0, &tile, cols, 8, AccelX4::Portable, &mut portable);
        gemm_q8_0x4_acts_x4_scalar_8(&q8, &tile, cols, &mut reference);
        assert_bits_eq("q8_0 hoisted", &hoisted, &wrapper);
        assert_bits_eq("q8_0 portable", &portable, &reference);

        gemm_q4_0x4_group_x4(&q4, 0, &tile, cols, 8, &mut wrapper);
        gemm_q4_0x4_group_x4_on(&q4, 0, &tile, cols, 8, here, &mut hoisted);
        gemm_q4_0x4_group_x4_on(&q4, 0, &tile, cols, 8, AccelX4::Portable, &mut portable);
        gemm_q4_0x4_acts_x4_scalar_8(&q4, &tile, cols, &mut reference);
        assert_bits_eq("q4_0 hoisted", &hoisted, &wrapper);
        assert_bits_eq("q4_0 portable", &portable, &reference);
    }
}

fn assert_bits_eq(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{what}: {got:?} vs {want:?}"
    );
}

fn synth_q8_0_acts(n: usize, cols: usize) -> Vec<Q8Activations> {
    (0..n)
        .map(|j| {
            let x: Vec<f32> = (0..cols)
                .map(|i| (((i + j * 13) as f32) * 0.021 - 0.9).sin() * 1.7)
                .collect();
            quantize_activations_q8(&x)
        })
        .collect()
}

/// llama.cpp `ggml_quantize_mat_q8_0_4x8`'s qs ordering, as the
/// reference `prepare_q8_acts_x4` must reproduce.
#[test]
fn prepare_q8_acts_x4_matches_interleave_reference() {
    let n_blocks = 3;
    let cols = n_blocks * Q8_0_BLOCK_ELEMS;
    for na in 1..=Q8K_ACTS_X4_NC {
        let acts = synth_q8_0_acts(na, cols);
        let tile = prepare_q8_acts_x4(&acts, cols);
        assert_eq!(tile.na, na);
        assert_eq!(tile.n_blocks, n_blocks);
        for b in 0..n_blocks {
            for (j, got) in tile.qs[b * 128..(b + 1) * 128].iter().enumerate() {
                let src_offset = (j / 32) * 8 + (j % 8);
                let src_id = (j % 32) / 8;
                let want = if src_id < na {
                    acts[src_id].q[b * Q8_0_BLOCK_ELEMS + src_offset]
                } else {
                    0
                };
                assert_eq!(*got, want, "qs mismatch, block {b} pos {j} na {na}");
            }
            for a in 0..4 {
                let want_d = acts.get(a).map_or(0.0, |act| act.d[b]);
                assert_eq!(tile.d[b * 4 + a], want_d, "d mismatch, block {b} row {a}");
            }
        }
    }
}

/// The interleave-8 NEON paths (DotProd `4x8` GEMV, i8mm GEMM) against
/// the scalar interleave-8 references, plus the x4 entries against the
/// compat entries (bit-exact -- they share the kernels).
#[test]
#[cfg(target_arch = "aarch64")]
fn q8_0_q4_0_interleave8_neon_matches_references_when_available() {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        return;
    }
    let n_blocks = 3;
    let n_groups = 2;

    // Q8_0
    let cols = n_blocks * Q8_0_BLOCK_ELEMS;
    let rows = n_groups * Q8_0X4_NROWS;
    let mut matrix = Vec::new();
    for r in 0..rows {
        matrix.extend_from_slice(&synth_q8_0_row(n_blocks, (r * 3 + 2) as u8));
    }
    let packed = pack_q8_0_matrix_x4(&matrix, rows, cols, 8);
    let acts = synth_q8_0_acts(4, cols);
    for act in &acts {
        let mut got = vec![0f32; rows];
        gemv_q8_0x4_q8_0(&packed, act, cols, n_groups, 8, &mut got);
        let mut want = vec![0f32; rows];
        gemv_q8_0x4_q8_0_scalar(&packed, act, cols, n_groups, 8, &mut want);
        for r in 0..rows {
            let err = (got[r] - want[r]).abs();
            assert!(
                err / want[r].abs().max(1.0) < 1e-5 || err < 1e-3,
                "q8_0 gemv row {r}: NEON 4x8 {} vs scalar {}",
                got[r],
                want[r]
            );
        }
    }
    // Q4_0
    let q4_cols = n_blocks * Q4_0_BLOCK_ELEMS;
    let q4_rows = n_groups * Q4_0X4_NROWS;
    let mut q4_matrix = Vec::new();
    for r in 0..q4_rows {
        q4_matrix.extend_from_slice(&synth_q4_0_row(n_blocks, (r * 9 + 1) as u8));
    }
    let q4_packed = pack_q4_0_matrix_x4(&q4_matrix, q4_rows, q4_cols, 8);
    let q4_acts = synth_q8_0_acts(4, q4_cols);
    for act in &q4_acts {
        let mut got = vec![0f32; q4_rows];
        gemv_q4_0x4_q8_0(&q4_packed, act, q4_cols, n_groups, 8, &mut got);
        let mut want = vec![0f32; q4_rows];
        gemv_q4_0x4_q8_0_scalar(&q4_packed, act, q4_cols, n_groups, 8, &mut want);
        for r in 0..q4_rows {
            let err = (got[r] - want[r]).abs();
            assert!(
                err / want[r].abs().max(1.0) < 1e-5 || err < 1e-3,
                "q4_0 gemv row {r}: NEON 4x8 {} vs scalar {}",
                got[r],
                want[r]
            );
        }
    }

    if !std::arch::is_aarch64_feature_detected!("i8mm") {
        return;
    }
    for na in 1..=Q8K_ACTS_X4_NC {
        let acts = synth_q8_0_acts(na, cols);
        let tile = prepare_q8_acts_x4(&acts, cols);
        for group in 0..n_groups {
            let mut x4_out = vec![0f32; Q8_0X4_NROWS * na];
            gemm_q8_0x4_group_x4(&packed, group, &tile, cols, 8, &mut x4_out);

            let mut group_out = vec![0f32; Q8_0X4_NROWS * na];
            gemm_q8_0x4_group(&packed, group, &acts, cols, 8, &mut group_out);
            assert_eq!(
                x4_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                group_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "q8_0 x4 entry diverged from the compat entry, group {group} na {na}"
            );

            let slice =
                &packed[group * n_blocks * Q8_0X4_BLOCK_BYTES..][..n_blocks * Q8_0X4_BLOCK_BYTES];
            let mut want = vec![0f32; Q8_0X4_NROWS * na];
            gemm_q8_0x4_acts_x4_scalar_8(slice, &tile, cols, &mut want);
            for (got, want) in x4_out.iter().zip(want.iter()) {
                let err = (got - want).abs();
                assert!(
                    err / want.abs().max(1.0) < 1e-5 || err < 1e-3,
                    "q8_0 group {group} na {na}: i8mm GEMM {got} vs portable {want}"
                );
            }
        }

        let q4_acts = synth_q8_0_acts(na, q4_cols);
        let q4_tile = prepare_q8_acts_x4(&q4_acts, q4_cols);
        for group in 0..n_groups {
            let mut x4_out = vec![0f32; Q4_0X4_NROWS * na];
            gemm_q4_0x4_group_x4(&q4_packed, group, &q4_tile, q4_cols, 8, &mut x4_out);

            let mut group_out = vec![0f32; Q4_0X4_NROWS * na];
            gemm_q4_0x4_group(&q4_packed, group, &q4_acts, q4_cols, 8, &mut group_out);
            assert_eq!(
                x4_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                group_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "q4_0 x4 entry diverged from the compat entry, group {group} na {na}"
            );

            let slice = &q4_packed[group * n_blocks * Q4_0X4_BLOCK_BYTES..]
                [..n_blocks * Q4_0X4_BLOCK_BYTES];
            let mut want = vec![0f32; Q4_0X4_NROWS * na];
            gemm_q4_0x4_acts_x4_scalar_8(slice, &q4_tile, q4_cols, &mut want);
            for (got, want) in x4_out.iter().zip(want.iter()) {
                let err = (got - want).abs();
                assert!(
                    err / want.abs().max(1.0) < 1e-5 || err < 1e-3,
                    "q4_0 group {group} na {na}: i8mm GEMM {got} vs portable {want}"
                );
            }
        }
    }
}

#[test]
fn block_size_matches_ggml() {
    assert_eq!(Q4_KX8_BLOCK_BYTES, 16 + 16 + 96 + 1024);
    assert_eq!(Q5_KX8_BLOCK_BYTES, 16 + 16 + 96 + 256 + 1024);
    assert_eq!(Q6_KX8_BLOCK_BYTES, 16 + 128 + 1024 + 512);
    assert_eq!(Q8_0X4_BLOCK_BYTES, 4 * 2 + Q8_0_BLOCK_ELEMS * Q8_0X4_NROWS);
    assert_eq!(Q4_0X4_BLOCK_BYTES, 4 * 2 + Q4_0_BLOCK_ELEMS * 2);
}

/// Quantized activations never reach `-128`, which is the invariant the
/// AVX2 Q8_0 GEMM's `_mm256_sign_epi8` negation rests on: negating
/// `i8::MIN` wraps to itself, so one product per occurrence would come
/// out with the wrong sign and nothing would say so.
///
/// `prepare_q8_acts_x4` `debug_assert`s the quad it builds; this pins the
/// property at its source instead, so a quantizer that stopped clamping
/// fails here rather than in an architecture-specific kernel nobody runs
/// on the machine that changed it.
#[test]
fn quantized_activations_never_reach_the_value_that_negates_to_itself() {
    // Deliberately extreme and asymmetric: the clamp only shows up when
    // rounding would otherwise land past -127.
    let x: Vec<f32> = (0..512)
        .map(|i| ((i as f32) * 0.37).sin() * 1000.0 - 500.0)
        .collect();
    assert!(!quantize_activations_q8(&x).q.contains(&i8::MIN));
    assert!(!quantize_activations_q8_k(&x).q.contains(&i8::MIN));
}
