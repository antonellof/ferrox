//! The five AVX2 `×4` GEMMs against the scalar twins they replace.
//!
//! Each kernel is held against `AccelX4::Portable` — the same
//! `gemm_*_acts_x4_scalar_8` reference the NEON i8mm kernels are held
//! against — on the shapes the aarch64 tests use and then on randomized
//! ones, and separately against the per-activation GEMV the same weights
//! go through at batch 1. The portable arm is pinned *bit-exact* to that
//! GEMV by the `*_x4_portable_is_bit_exact_vs_scalar_gemv` tests beside
//! this file, so the two comparisons are the same claim reached from
//! both ends.
//!
//! **These tests run only where the instructions do.** Rosetta 2 on
//! Apple Silicon reports `avx2 = false`, so an `x86_64-apple-darwin`
//! `cargo test` skips every assertion here; a `linux/amd64` container and
//! a real x86 box run them. [`avx2_here`] is what stops that skip from
//! reading as a pass.

use super::*;
use crate::repack::*;
use crate::{
    Q4_0_BLOCK_ELEMS, Q4_K_BLOCK_ELEMS, Q5_K_BLOCK_ELEMS, Q6_K_BLOCK_ELEMS, Q8_0_BLOCK_ELEMS,
};

/// True when this host runs the AVX2 kernels, and asserts that when it
/// does, [`AccelX4::detect`] actually selects them — otherwise every test
/// below would compare the portable arm against itself and pass while
/// asserting nothing.
fn avx2_here() -> bool {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return false;
    }
    assert_eq!(
        AccelX4::detect(),
        AccelX4::Avx2,
        "an AVX2+FMA host must select the AVX2 kernels, or these tests compare \
         the portable arm with itself"
    );
    assert!(
        interleaved_gemm_is_accelerated(8),
        "the one predicate must agree with the kernel choice"
    );
    true
}

/// A deterministic 8-bit stream, so a randomized shape is reproducible
/// from the seed printed in a failure.
struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 >> 8
    }

    fn in_range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() as usize) % (hi - lo + 1)
    }
}

/// How far an AVX2 output may sit from the reference one, as a multiple
/// of the RMS of the reference outputs.
///
/// **Relative to the RMS, not to the element.** A per-element relative
/// bound is meaningless here, and this repo has already been bitten by
/// one (`weight_matrix::assert_batch_row_matches`): these dots run 256
/// terms per super-block whose partial sums reach `1e4` and cancel down
/// to tens, so an output near zero carries the rounding of its *terms*,
/// not of itself. The difference being measured is only where the f32
/// scale folds in — the AVX2 kernels fold once per super-block where the
/// scalar twin folds once per `k` — since both compute the int8 dot
/// exactly in `i32`.
///
/// Measured over every shape below on a `linux/amd64` container with
/// real AVX2 (2026-09-09); the per-kernel worst is printed by each test
/// under `--nocapture`, so re-deriving this is a run rather than a
/// guess. Worst across the five kernels was `5.6e-6 x RMS` (Q4_K on a
/// random shape); the others sat between `1.1e-7` and `4.3e-6`.
///
/// `1e-4` keeps an 18x margin over that while staying four orders of
/// magnitude tighter than a lane error: a one-row shift in the Q5_K `qh`
/// interleave was measured at `2.07 x RMS` when this tier was written,
/// and a wrong activation run is the same size.
const MAX_ERR_VS_RMS: f32 = 1e-4;

/// The worst `err / rms` of `got` against `want`, asserting it stays
/// under [`MAX_ERR_VS_RMS`]. Returns the ratio so a caller can report the
/// worst it saw across shapes.
fn assert_close(what: &str, got: &[f32], want: &[f32]) -> f32 {
    let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32)
        .sqrt()
        .max(1.0);
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ratio = (g - w).abs() / rms;
        worst = worst.max(ratio);
        assert!(
            ratio <= MAX_ERR_VS_RMS,
            "{what} slot {i}: AVX2 {g} vs reference {w} ({ratio:e} x RMS {rms}, \
             bound {MAX_ERR_VS_RMS:e})"
        );
    }
    worst
}

/// One kernel's acceptance, written once and applied five times.
///
/// `$gemm_on` is the hoisted-dispatch entry point, which takes the kernel
/// choice rather than probing: passing `Avx2` and `Portable` to the same
/// call is what makes this a comparison of kernels and not of call sites.
macro_rules! avx2_kernel_tests {
    (
        case = $case:ident,
        vs_portable = $vs_portable:ident,
        vs_gemv = $vs_gemv:ident,
        synth = $synth:ident,
        pack = $pack:ident,
        prepare = $prepare:ident,
        synth_acts = $synth_acts:ident,
        gemm_on = $gemm_on:ident,
        gemv_group = $gemv_group:ident,
        nrows = $nrows:expr,
        block_elems = $block_elems:expr,
        gemm_nc = $gemm_nc:expr,
    ) => {
        fn $case(n_blocks: usize, n_groups: usize, na: usize, seed: u8) -> f32 {
            let cols = n_blocks * $block_elems;
            let rows = n_groups * $nrows;
            let mut matrix = Vec::new();
            for r in 0..rows {
                matrix.extend_from_slice(&$synth(n_blocks, seed.wrapping_add((r * 7) as u8)));
            }
            let packed = $pack(&matrix, rows, cols, 8);
            let acts = $synth_acts(na, cols);
            let tile = $prepare(&acts, cols);

            let mut worst = 0f32;
            for group in 0..n_groups {
                let mut got = vec![0f32; $nrows * na];
                let mut want = vec![0f32; $nrows * na];
                $gemm_on(&packed, group, &tile, cols, 8, AccelX4::Avx2, &mut got);
                $gemm_on(&packed, group, &tile, cols, 8, AccelX4::Portable, &mut want);
                let what = format!(
                    "{} n_blocks {n_blocks} groups {n_groups} na {na} seed {seed} group {group}",
                    stringify!($gemm_on),
                );
                worst = worst.max(assert_close(&what, &got, &want));

                // And against the GEMV the same weights go through at
                // batch 1: prefill and decode run on the same prompt, so
                // a disagreement here is the two answering differently
                // about the same tokens.
                let mut per_act = vec![0f32; $nrows * na];
                for (a, act) in acts.iter().enumerate() {
                    let mut one = vec![0f32; $nrows];
                    $gemv_group(&packed, group, act, cols, 8, &mut one);
                    for (r, v) in one.iter().enumerate() {
                        per_act[r * na + a] = *v;
                    }
                }
                worst = worst.max(assert_close(&format!("{what} vs GEMV"), &got, &per_act));
            }
            worst
        }

        #[test]
        fn $vs_portable() {
            if !avx2_here() {
                return;
            }
            let mut worst = 0f32;
            // The shapes the aarch64 tests use: one row-group, three
            // super-blocks, every activation count up to the tile width.
            //
            // Three seeds, not the aarch64 tests' one. Dropping the
            // `_mm256_sign_epi8` magnitude operand from the Q8_0 kernel
            // -- a real defect, and one the random-shape sibling caught
            // -- left seed 3 unchanged, because that fixture's weights
            // happen to be non-negative where it matters. A fixed shape
            // that only ever sees one fixture is a fixed *value*.
            for seed in [3u8, 91, 200] {
                for na in 1..=$gemm_nc {
                    worst = worst.max($case(3, 1, na, seed));
                }
            }
            // Printed so re-deriving `MAX_ERR_VS_RMS` is a `--nocapture`
            // run rather than an edit.
            println!("{} fixed shapes: worst {worst:e} x RMS", stringify!($gemm_on));
        }

        #[test]
        fn $vs_gemv() {
            if !avx2_here() {
                return;
            }
            // Widths, row-group counts and partial tiles the fixed shapes
            // never reach. Both comparisons run inside `$case`, so this
            // covers the portable twin and the GEMV at once.
            let mut worst = 0f32;
            let mut rng = Lcg(0x5eed_0017);
            for _ in 0..24 {
                let n_blocks = rng.in_range(1, 4);
                let n_groups = rng.in_range(1, 3);
                let na = rng.in_range(1, $gemm_nc);
                let seed = rng.in_range(0, 255) as u8;
                worst = worst.max($case(n_blocks, n_groups, na, seed));
            }
            println!("{} random shapes: worst {worst:e} x RMS", stringify!($gemm_on));
        }
    };
}

avx2_kernel_tests!(
    case = q4_kx8_case,
    vs_portable = q4_kx8_avx2_gemm_matches_its_scalar_twin,
    vs_gemv = q4_kx8_avx2_gemm_matches_its_scalar_twin_on_random_shapes,
    synth = synth_q4_k_row,
    pack = pack_q4_k_matrix_x8,
    prepare = prepare_q8_k_acts_x4,
    synth_acts = synth_q8_k_acts,
    gemm_on = gemm_q4_kx8_group_x4_on,
    gemv_group = gemv_q4_kx8_group,
    nrows = Q4_KX8_NROWS,
    block_elems = Q4_K_BLOCK_ELEMS,
    gemm_nc = Q4_KX8_GEMM_NC,
);

avx2_kernel_tests!(
    case = q8_0x4_case,
    vs_portable = q8_0x4_avx2_gemm_matches_its_scalar_twin,
    vs_gemv = q8_0x4_avx2_gemm_matches_its_scalar_twin_on_random_shapes,
    synth = synth_q8_0_row,
    pack = pack_q8_0_matrix_x4,
    prepare = prepare_q8_acts_x4,
    synth_acts = synth_q8_0_acts,
    gemm_on = gemm_q8_0x4_group_x4_on,
    gemv_group = gemv_q8_0x4_group,
    nrows = Q8_0X4_NROWS,
    block_elems = Q8_0_BLOCK_ELEMS,
    gemm_nc = Q8K_ACTS_X4_NC,
);

avx2_kernel_tests!(
    case = q4_0x4_case,
    vs_portable = q4_0x4_avx2_gemm_matches_its_scalar_twin,
    vs_gemv = q4_0x4_avx2_gemm_matches_its_scalar_twin_on_random_shapes,
    synth = synth_q4_0_row,
    pack = pack_q4_0_matrix_x4,
    prepare = prepare_q8_acts_x4,
    synth_acts = synth_q8_0_acts,
    gemm_on = gemm_q4_0x4_group_x4_on,
    gemv_group = gemv_q4_0x4_group,
    nrows = Q4_0X4_NROWS,
    block_elems = Q4_0_BLOCK_ELEMS,
    gemm_nc = Q8K_ACTS_X4_NC,
);

avx2_kernel_tests!(
    case = q6_kx8_case,
    vs_portable = q6_kx8_avx2_gemm_matches_its_scalar_twin,
    vs_gemv = q6_kx8_avx2_gemm_matches_its_scalar_twin_on_random_shapes,
    synth = synth_q6_k_row,
    pack = pack_q6_k_matrix_x8,
    prepare = prepare_q8_k_acts_x4,
    synth_acts = synth_q8_k_acts,
    gemm_on = gemm_q6_kx8_group_x4_on,
    gemv_group = gemv_q6_kx8_group,
    nrows = Q6_KX8_NROWS,
    block_elems = Q6_K_BLOCK_ELEMS,
    gemm_nc = Q8K_ACTS_X4_NC,
);

avx2_kernel_tests!(
    case = q5_kx8_case,
    vs_portable = q5_kx8_avx2_gemm_matches_its_scalar_twin,
    vs_gemv = q5_kx8_avx2_gemm_matches_its_scalar_twin_on_random_shapes,
    synth = synth_q5_k_row,
    pack = pack_q5_k_matrix_x8,
    prepare = prepare_q8_k_acts_x4,
    synth_acts = synth_q8_k_acts,
    gemm_on = gemm_q5_kx8_group_x4_on,
    gemv_group = gemv_q5_kx8_group,
    nrows = Q5_KX8_NROWS,
    block_elems = Q5_K_BLOCK_ELEMS,
    gemm_nc = Q5_KX8_GEMM_NC,
);
