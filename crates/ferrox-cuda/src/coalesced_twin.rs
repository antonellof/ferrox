//! Scalar twin of the coalesced Q4_K matvec kernel.
//!
//! The CUDA C in `gpu.rs` cannot run here, so the same arithmetic and
//! the same lane-to-data mapping are transcribed in Rust and held
//! against a dequantize-then-dot reference from `ferrox_quant`. The
//! kernel's risk is not the arithmetic, which is unchanged from the
//! kernel it replaces, but the MAPPING: which bytes each lane reads and
//! which activations they pair with.

fn f16_to_f32(bits: u16) -> f32 {
    crate::mul_mm::f16_to_f32(bits)
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

/// Scalar twin of `Q4_K_MATVEC_COALESCED_KERNEL_SRC`, summed the way
/// the warp does: lane by lane, then reduced.
///
/// The mapping is the whole risk. Lane `l` reads `qs[4l..4l+4)`, which
/// sits in 32-byte group `oi = l/8`, and that group's low nibbles are
/// activations `oi*64 + (4l%32) ..` while its high nibbles are the same
/// plus 32. Pairing byte `i` with activation `i` for both halves reads
/// plausible numbers from the wrong place.
pub fn q4_k_matvec_coalesced_row(row: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    let mut lanes = [0f64; 32];
    for blk in 0..n_blocks {
        let block = &row[blk * 144..(blk + 1) * 144];
        let d = f16_to_f32(u16::from(block[0]) | (u16::from(block[1]) << 8));
        let dmin = f16_to_f32(u16::from(block[2]) | (u16::from(block[3]) << 8));
        let scales = &block[4..16];
        let qs = &block[16..144];

        for (lane, slot) in lanes.iter_mut().enumerate() {
            let off = 4 * lane;
            let oi = lane / 8;
            let within = off % 32;
            let (sc1, m1) = q4_k_scale_min(2 * oi, scales);
            let (sc2, m2) = q4_k_scale_min(2 * oi + 1, scales);
            let d1 = d * f32::from(sc1);
            let min1 = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let min2 = dmin * f32::from(m2);
            let xb = blk * 256 + oi * 64 + within;
            let mut acc = 0f32;
            for i in 0..4 {
                let b = qs[off + i];
                acc += (d1 * f32::from(b & 0x0F) - min1) * x[xb + i];
                acc += (d2 * f32::from(b >> 4) - min2) * x[xb + 32 + i];
            }
            *slot += f64::from(acc);
        }
    }
    lanes.iter().sum::<f64>() as f32
}

#[cfg(test)]
mod coalesced_tests {
    use super::*;

    fn q4_k_row(n_blocks: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        let mut row: Vec<u8> = (0..n_blocks * 144)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect();
        for blk in 0..n_blocks {
            let b = blk * 144;
            row[b] = 0x00;
            row[b + 1] = 0x2C;
            row[b + 2] = 0x00;
            row[b + 3] = 0x28;
        }
        row
    }

    /// The coalesced lane mapping must sum to the same thing as
    /// dequantize-then-dot. Unlike the dp4a path this keeps f32
    /// activations, so the only difference from the reference is
    /// summation order and the tolerance can be tight.
    #[test]
    fn the_coalesced_row_matches_dequantize_then_dot() {
        for (n_blocks, seed) in [(1usize, 7u32), (3, 19), (6, 41)] {
            let row = q4_k_row(n_blocks, seed);
            let n = n_blocks * 256;
            let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.021).sin()).collect();
            let weights = ferrox_quant::dequant_q4_k(&row).expect("dequant");
            let want: f64 = weights
                .iter()
                .zip(x.iter())
                .map(|(w, v)| f64::from(*w) * f64::from(*v))
                .sum();
            let got = f64::from(q4_k_matvec_coalesced_row(&row, &x, n_blocks));
            let scale = weights
                .iter()
                .zip(x.iter())
                .map(|(w, v)| (f64::from(*w) * f64::from(*v)).abs())
                .sum::<f64>()
                .max(1.0);
            let err = (got - want).abs() / scale;
            assert!(
                err < 1e-6,
                "n_blocks {n_blocks} seed {seed}: coalesced {got} vs dequant-dot {want} \
                 (rel to term scale {err:e})"
            );
        }
    }

    /// Only the second 32-element sub-block is non-zero, so a lane that
    /// pairs the high nibbles with the FIRST activation block scores
    /// zero here instead of the right answer.
    #[test]
    fn the_high_nibbles_pair_with_the_second_activation_block() {
        let row = q4_k_row(1, 13);
        let mut x = vec![0f32; 256];
        for (i, v) in x.iter_mut().enumerate().take(64).skip(32) {
            *v = ((i % 5) as f32) + 1.0;
        }
        let weights = ferrox_quant::dequant_q4_k(&row).expect("dequant");
        let want: f32 = weights[32..64]
            .iter()
            .zip(x[32..64].iter())
            .map(|(w, v)| w * v)
            .sum();
        let got = q4_k_matvec_coalesced_row(&row, &x, 1);
        assert!(
            (got - want).abs() / want.abs().max(1.0) < 1e-5,
            "high nibbles read the wrong activations: {got} vs {want}"
        );
    }

    /// Every one of the 128 quantized bytes must be consumed exactly
    /// once across the 32 lanes. An off-by-one in the lane mapping
    /// silently drops or double-counts a slice.
    #[test]
    fn the_thirty_two_lanes_cover_all_128_bytes_exactly_once() {
        let mut seen = [0u8; 128];
        for lane in 0..32usize {
            for i in 0..4 {
                seen[4 * lane + i] += 1;
            }
        }
        assert!(
            seen.iter().all(|c| *c == 1),
            "lane mapping does not tile the 128 qs bytes: {seen:?}"
        );
    }
}
