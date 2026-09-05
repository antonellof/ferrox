//! Scalar twin of the `dp4a` matvec kernels.
//!
//! The CUDA C in `gpu.rs` cannot run here, so it is transcribed a second
//! time in Rust and held against a dequantize-then-dot reference built
//! from `ferrox_quant`. That ladder caught a real transcription error in
//! the K-quant GEMM (llama's `il` is used three different ways in one
//! function), and the `dp4a` port has the same hazard: the 32 weight
//! bytes of a Q4_K sub-block pair carry the FIRST 32 activations in
//! their low nibbles and the NEXT 32 in their high nibbles, so a reader
//! who pairs byte `l` with activation `l` for both halves gets
//! plausible numbers from the wrong place.

/// One block of 32 activations quantized to int8 with a single scale,
/// the shape llama.cpp calls `q8_1`.
#[derive(Debug, Clone, Default)]
pub struct Q8Activations {
    pub qs: Vec<i8>,
    pub ds: Vec<f32>,
}

/// Scalar twin of `QUANTIZE_Q8_1_KERNEL_SRC`.
///
/// `rintf`, not truncation: a truncating cast biases every value toward
/// zero, and on a 4096-wide activation that bias does not cancel.
pub fn quantize_q8_1(x: &[f32]) -> Q8Activations {
    assert!(x.len().is_multiple_of(32), "q8_1 blocks are 32 wide");
    let mut qs = vec![0i8; x.len()];
    let mut ds = vec![0f32; x.len() / 32];
    for (b, chunk) in x.as_chunks::<32>().0.iter().enumerate() {
        let amax = chunk.iter().fold(0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        ds[b] = d;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (i, v) in chunk.iter().enumerate() {
            qs[b * 32 + i] = (v * inv).round() as i8;
        }
    }
    Q8Activations { qs, ds }
}

/// Four int8 lanes multiplied and accumulated, the CPU's `__dp4a`.
fn dp4a(a: [i8; 4], b: [i8; 4], c: i32) -> i32 {
    c + a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| i32::from(*x) * i32::from(*y))
        .sum::<i32>()
}

fn q4_k_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    crate::mul_mm::f16_to_f32(bits)
}

/// Scalar twin of `Q4_K_MATVEC_DP4A_KERNEL_SRC`, for one output row.
pub fn q4_k_matvec_dp4a_row(row: &[u8], acts: &Q8Activations, n_blocks: usize) -> f32 {
    let mut acc = 0f32;
    for blk in 0..n_blocks {
        let block = &row[blk * 144..(blk + 1) * 144];
        let d = f16_to_f32(u16::from(block[0]) | (u16::from(block[1]) << 8));
        let dmin = f16_to_f32(u16::from(block[2]) | (u16::from(block[3]) << 8));
        let scales = &block[4..16];
        let qs = &block[16..144];
        let xblk = blk * 8;

        let (mut is, mut q_off, mut sub) = (0usize, 0usize, 0usize);
        for _oi in 0..4 {
            let (sc1, m1) = q4_k_scale_min(is, scales);
            let (sc2, m2) = q4_k_scale_min(is + 1, scales);
            let wp = &qs[q_off..q_off + 32];
            let lo_at = (xblk + sub) * 32;
            let xlo = &acts.qs[lo_at..lo_at + 32];
            let xhi = &acts.qs[lo_at + 32..lo_at + 64];

            let (mut dot_lo, mut sum_lo, mut dot_hi, mut sum_hi) = (0i32, 0i32, 0i32, 0i32);
            for k in 0..8 {
                let w = &wp[4 * k..4 * k + 4];
                let a_lo: [i8; 4] = xlo[4 * k..4 * k + 4].try_into().expect("4");
                let a_hi: [i8; 4] = xhi[4 * k..4 * k + 4].try_into().expect("4");
                let w_lo: [i8; 4] = std::array::from_fn(|i| (w[i] & 0x0F) as i8);
                let w_hi: [i8; 4] = std::array::from_fn(|i| (w[i] >> 4) as i8);
                dot_lo = dp4a(w_lo, a_lo, dot_lo);
                sum_lo = dp4a([1; 4], a_lo, sum_lo);
                dot_hi = dp4a(w_hi, a_hi, dot_hi);
                sum_hi = dp4a([1; 4], a_hi, sum_hi);
            }

            let dxlo = acts.ds[xblk + sub];
            let dxhi = acts.ds[xblk + sub + 1];
            acc += d * f32::from(sc1) * dxlo * dot_lo as f32
                - dmin * f32::from(m1) * dxlo * sum_lo as f32;
            acc += d * f32::from(sc2) * dxhi * dot_hi as f32
                - dmin * f32::from(m2) * dxhi * sum_hi as f32;

            q_off += 32;
            sub += 2;
            is += 2;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// A Q4_K row with finite scales, so the comparison is about the
    /// unpack rather than about NaN.
    fn q4_k_row(n_blocks: usize, seed: u32) -> Vec<u8> {
        let mut row = bytes(n_blocks * 144, seed);
        for blk in 0..n_blocks {
            let b = blk * 144;
            row[b] = 0x00;
            row[b + 1] = 0x2C; // d = 0.0625
            row[b + 2] = 0x00;
            row[b + 3] = 0x28; // dmin = 0.03125
        }
        row
    }

    /// The ALGEBRA, with quantization error removed from the
    /// comparison.
    ///
    /// The reference dots the dequantized weights against the
    /// DEQUANTIZED activations (`xd * xq`), i.e. exactly the values the
    /// integer path is summing. Any difference is then an algebra
    /// error, not a rounding one, so the tolerance can be tight enough
    /// to catch a dropped term.
    ///
    /// It exists because the looser test below did not: zeroing one of
    /// the two min terms left every test green, which made it a test
    /// that could not fail.
    #[test]
    fn the_dp4a_row_matches_the_same_sum_computed_exactly() {
        for (n_blocks, seed) in [(1usize, 3u32), (4, 11), (7, 29)] {
            let row = q4_k_row(n_blocks, seed);
            let n = n_blocks * 256;
            // Deliberately all-positive: with mixed signs the min term
            // partly cancels against itself and a dropped one hides.
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 101) as f32 / 25.0) + 0.5)
                .collect();

            let weights = ferrox_quant::dequant_q4_k(&row).expect("dequant");
            let acts = quantize_q8_1(&x);
            // The activation the integer path actually sees.
            let x_seen: Vec<f64> = (0..n)
                .map(|i| f64::from(acts.qs[i]) * f64::from(acts.ds[i / 32]))
                .collect();
            let exact: f64 = weights
                .iter()
                .zip(x_seen.iter())
                .map(|(w, v)| f64::from(*w) * v)
                .sum();

            let got = q4_k_matvec_dp4a_row(&row, &acts, n_blocks) as f64;
            let scale = exact.abs().max(1.0);
            let err = (got - exact).abs() / scale;
            assert!(
                err < 1e-5,
                "n_blocks {n_blocks} seed {seed}: dp4a {got} vs exact {exact} (rel {err:e})"
            );
        }
    }

    /// The whole point: the integer path must agree with dequantize
    /// then dot, within the error q8_1 quantization actually costs.
    #[test]
    fn the_dp4a_row_matches_dequantize_then_dot() {
        for (n_blocks, seed) in [(1usize, 3u32), (4, 11), (7, 29)] {
            let row = q4_k_row(n_blocks, seed);
            let n = n_blocks * 256;
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0)
                .collect();

            let weights = ferrox_quant::dequant_q4_k(&row).expect("dequant");
            let reference: f32 = weights.iter().zip(x.iter()).map(|(w, v)| w * v).sum();

            let acts = quantize_q8_1(&x);
            let got = q4_k_matvec_dp4a_row(&row, &acts, n_blocks);

            // q8_1 keeps ~7 bits of the activation, so the tolerance is
            // set by that, not by f32 rounding. Scaled by the magnitude
            // of the terms rather than the result, because the sum
            // cancels.
            let scale: f32 = weights
                .iter()
                .zip(x.iter())
                .map(|(w, v)| (w * v).abs())
                .sum::<f32>()
                .max(1.0);
            let err = (got - reference).abs() / scale;
            assert!(
                err < 5e-3,
                "n_blocks {n_blocks} seed {seed}: dp4a {got} vs dequant-dot {reference} \
                 (err {err:e} of term scale {scale})"
            );
        }
    }

    /// The hazard this port has: the 32 weight bytes carry the first 32
    /// activations in their LOW nibbles and the next 32 in their HIGH
    /// nibbles. Pairing byte `l` with activation `l` for both halves
    /// reads plausible numbers from the wrong place.
    #[test]
    fn the_high_nibble_half_reads_the_second_activation_block() {
        let row = q4_k_row(1, 5);
        let mut x = vec![0f32; 256];
        // Only the SECOND 32-element sub-block is non-zero.
        for (i, v) in x.iter_mut().enumerate().take(64).skip(32) {
            *v = ((i % 7) as f32) + 1.0;
        }
        let acts = quantize_q8_1(&x);
        let got = q4_k_matvec_dp4a_row(&row, &acts, 1);

        let weights = ferrox_quant::dequant_q4_k(&row).expect("dequant");
        let reference: f32 = weights[32..64]
            .iter()
            .zip(x[32..64].iter())
            .map(|(w, v)| w * v)
            .sum();
        let scale = reference.abs().max(1.0);
        assert!(
            (got - reference).abs() / scale < 5e-3,
            "high-nibble half read the wrong activations: {got} vs {reference}"
        );
    }

    /// A zero activation block must not divide by zero.
    #[test]
    fn a_zero_activation_block_quantizes_to_zero_not_nan() {
        let q = quantize_q8_1(&vec![0f32; 64]);
        assert!(q.ds.iter().all(|d| *d == 0.0));
        assert!(q.qs.iter().all(|v| *v == 0));
    }
}
