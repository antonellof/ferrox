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

/// Scalar twin of `Q6_K_MATVEC_COALESCED_KERNEL_SRC`, lane by lane.
pub fn q6_k_matvec_coalesced_row(row: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    let mut lanes = [0f64; 32];
    for blk in 0..n_blocks {
        let block = &row[blk * 210..(blk + 1) * 210];
        let d = f16_to_f32(u16::from(block[208]) | (u16::from(block[209]) << 8));
        let x_base = blk * 256;
        for (lane, slot) in lanes.iter_mut().enumerate() {
            let is = lane / 16;
            let mut acc = 0f32;
            for half in 0..2 {
                let ql = &block[half * 64..];
                let qh = &block[128 + half * 32..];
                let sc = &block[192 + half * 8..192 + half * 8 + 8];
                let xh = x_base + half * 128;
                let q1 = i32::from((ql[lane] & 0x0F) | ((qh[lane] & 0x03) << 4)) - 32;
                let q2 = i32::from((ql[lane + 32] & 0x0F) | (((qh[lane] >> 2) & 0x03) << 4)) - 32;
                let q3 = i32::from((ql[lane] >> 4) | (((qh[lane] >> 4) & 0x03) << 4)) - 32;
                let q4 = i32::from((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4)) - 32;
                acc += d * f32::from(sc[is] as i8) * q1 as f32 * x[xh + lane];
                acc += d * f32::from(sc[is + 2] as i8) * q2 as f32 * x[xh + lane + 32];
                acc += d * f32::from(sc[is + 4] as i8) * q3 as f32 * x[xh + lane + 64];
                acc += d * f32::from(sc[is + 6] as i8) * q4 as f32 * x[xh + lane + 96];
            }
            *slot += f64::from(acc);
        }
    }
    lanes.iter().sum::<f64>() as f32
}

#[cfg(test)]
mod q6_k_coalesced_tests {
    use super::*;

    fn q6_k_row(n_blocks: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        let mut row: Vec<u8> = (0..n_blocks * 210)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect();
        for blk in 0..n_blocks {
            let b = blk * 210;
            row[b + 208] = 0x00;
            row[b + 209] = 0x2C; // finite d
        }
        row
    }

    #[test]
    fn the_coalesced_q6_k_row_matches_dequantize_then_dot() {
        for (n_blocks, seed) in [(1usize, 5u32), (3, 23), (5, 47)] {
            let row = q6_k_row(n_blocks, seed);
            let n = n_blocks * 256;
            let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).cos()).collect();
            let weights = ferrox_quant::dequant_q6_k(&row).expect("dequant");
            let want: f64 = weights
                .iter()
                .zip(x.iter())
                .map(|(w, v)| f64::from(*w) * f64::from(*v))
                .sum();
            let got = f64::from(q6_k_matvec_coalesced_row(&row, &x, n_blocks));
            let scale = weights
                .iter()
                .zip(x.iter())
                .map(|(w, v)| (f64::from(*w) * f64::from(*v)).abs())
                .sum::<f64>()
                .max(1.0);
            assert!(
                (got - want).abs() / scale < 1e-6,
                "n_blocks {n_blocks} seed {seed}: coalesced {got} vs dequant-dot {want}"
            );
        }
    }

    /// The four sub-scales are 2 apart, not adjacent. Getting that
    /// wrong scales three of the four outputs by the wrong factor.
    #[test]
    fn the_four_scale_indices_are_two_apart() {
        let row = q6_k_row(1, 9);
        let x: Vec<f32> = (0..256).map(|i| ((i % 11) as f32) - 5.0).collect();
        let weights = ferrox_quant::dequant_q6_k(&row).expect("dequant");
        let want: f64 = weights
            .iter()
            .zip(x.iter())
            .map(|(w, v)| f64::from(*w) * f64::from(*v))
            .sum();
        let got = f64::from(q6_k_matvec_coalesced_row(&row, &x, 1));
        assert!(
            (got - want).abs() / want.abs().max(1.0) < 1e-5,
            "{got} vs {want}"
        );
    }
}

/// Scalar twin of `Q5_K_MATVEC_COALESCED_KERNEL_SRC`.
///
/// Q5_K shares Q4_K's six-bit scale/min packing; what it adds is a
/// fifth bit per weight, held in `qh` one BIT-PLANE per 32-element
/// group. Lane `l` owns element `l` of each group, so it reads
/// `qh[l]` once per block and picks bit `2*oi` for the low nibble and
/// bit `2*oi + 1` for the high one. Reading the wrong bit-plane
/// perturbs weights by exactly 16 and is easy to miss by eye.
pub fn q5_k_matvec_coalesced_row(row: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    let mut lanes = [0f64; 32];
    for blk in 0..n_blocks {
        let block = &row[blk * 176..(blk + 1) * 176];
        let d = f16_to_f32(u16::from(block[0]) | (u16::from(block[1]) << 8));
        let dmin = f16_to_f32(u16::from(block[2]) | (u16::from(block[3]) << 8));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let x_base = blk * 256;

        for (lane, slot) in lanes.iter_mut().enumerate() {
            let mut acc = 0f32;
            for oi in 0..4usize {
                let (sc1, m1) = q4_k_scale_min(2 * oi, scales);
                let (sc2, m2) = q4_k_scale_min(2 * oi + 1, scales);
                let d1 = d * f32::from(sc1);
                let min1 = dmin * f32::from(m1);
                let d2 = d * f32::from(sc2);
                let min2 = dmin * f32::from(m2);
                let ql = qs[oi * 32 + lane];
                let u1 = 1u8 << (2 * oi);
                let u2 = 2u8 << (2 * oi);
                let xb = x_base + oi * 64;
                let hi1 = if qh[lane] & u1 != 0 { 16u8 } else { 0 };
                let hi2 = if qh[lane] & u2 != 0 { 16u8 } else { 0 };
                acc += (d1 * f32::from((ql & 0x0F) + hi1) - min1) * x[xb + lane];
                acc += (d2 * f32::from((ql >> 4) + hi2) - min2) * x[xb + 32 + lane];
            }
            *slot += f64::from(acc);
        }
    }
    lanes.iter().sum::<f64>() as f32
}

/// Scalar twin of `Q8_0_MATVEC_COALESCED_KERNEL_SRC`.
///
/// The simplest of the family: a Q8_0 block is exactly one warp wide,
/// so lane `l` takes byte `l` of every block and the whole warp's load
/// is 32 contiguous bytes. The scale is per block, so every lane reads
/// the same two header bytes and the hardware broadcasts them.
pub fn q8_0_matvec_coalesced_row(row: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    let mut lanes = [0f64; 32];
    for blk in 0..n_blocks {
        let block = &row[blk * 34..(blk + 1) * 34];
        let d = f16_to_f32(u16::from(block[0]) | (u16::from(block[1]) << 8));
        let base = blk * 32;
        for (lane, slot) in lanes.iter_mut().enumerate() {
            let q = block[2 + lane] as i8;
            *slot += f64::from(d * f32::from(q) * x[base + lane]);
        }
    }
    lanes.iter().sum::<f64>() as f32
}

#[cfg(test)]
mod q5_k_and_q8_0_coalesced_tests {
    use super::*;

    fn pseudo(len: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    fn q5_k_row(n_blocks: usize, seed: u32) -> Vec<u8> {
        let mut row = pseudo(n_blocks * 176, seed);
        for blk in 0..n_blocks {
            let b = blk * 176;
            row[b] = 0x00;
            row[b + 1] = 0x2C; // finite d
            row[b + 2] = 0x00;
            row[b + 3] = 0x28; // finite dmin
        }
        row
    }

    fn q8_0_row(n_blocks: usize, seed: u32) -> Vec<u8> {
        let mut row = pseudo(n_blocks * 34, seed);
        for blk in 0..n_blocks {
            let b = blk * 34;
            row[b] = 0x00;
            row[b + 1] = 0x2C; // finite d
        }
        row
    }

    fn close(got: f64, want: f64, weights: &[f32], x: &[f32], tol: f64, what: &str) {
        let scale = weights
            .iter()
            .zip(x.iter())
            .map(|(w, v)| (f64::from(*w) * f64::from(*v)).abs())
            .sum::<f64>()
            .max(1.0);
        assert!(
            (got - want).abs() / scale < tol,
            "{what}: coalesced {got} vs dequant-dot {want}"
        );
    }

    #[test]
    fn the_coalesced_q5_k_row_matches_dequantize_then_dot() {
        for (n_blocks, seed) in [(1usize, 7u32), (3, 31), (5, 53)] {
            let row = q5_k_row(n_blocks, seed);
            let x: Vec<f32> = (0..n_blocks * 256)
                .map(|i| ((i as f32) * 0.023).sin())
                .collect();
            let weights = ferrox_quant::dequant_q5_k(&row).expect("dequant");
            let want: f64 = weights
                .iter()
                .zip(x.iter())
                .map(|(w, v)| f64::from(*w) * f64::from(*v))
                .sum();
            let got = f64::from(q5_k_matvec_coalesced_row(&row, &x, n_blocks));
            close(
                got,
                want,
                &weights,
                &x,
                1e-6,
                &format!("q5_k {n_blocks}/{seed}"),
            );
        }
    }

    /// The fifth bit lives in a per-group BIT-PLANE, so the low nibble
    /// of group `oi` takes bit `2*oi` and the high nibble bit
    /// `2*oi + 1`. Both wrong readings shift weights by 16.
    #[test]
    fn the_fifth_bit_comes_from_the_right_bit_plane() {
        let row = q5_k_row(1, 13);
        let x: Vec<f32> = (0..256).map(|i| ((i % 9) as f32) - 4.0).collect();
        let weights = ferrox_quant::dequant_q5_k(&row).expect("dequant");
        let want: f64 = weights
            .iter()
            .zip(x.iter())
            .map(|(w, v)| f64::from(*w) * f64::from(*v))
            .sum();
        let got = f64::from(q5_k_matvec_coalesced_row(&row, &x, 1));
        close(got, want, &weights, &x, 1e-5, "q5_k bit-plane");
    }

    #[test]
    fn the_coalesced_q8_0_row_matches_dequantize_then_dot() {
        for (n_blocks, seed) in [(1usize, 3u32), (7, 29), (16, 61)] {
            let row = q8_0_row(n_blocks, seed);
            let x: Vec<f32> = (0..n_blocks * 32)
                .map(|i| ((i as f32) * 0.031).cos())
                .collect();
            let weights = ferrox_quant::dequant_q8_0(&row).expect("dequant");
            let want: f64 = weights
                .iter()
                .zip(x.iter())
                .map(|(w, v)| f64::from(*w) * f64::from(*v))
                .sum();
            let got = f64::from(q8_0_matvec_coalesced_row(&row, &x, n_blocks));
            close(
                got,
                want,
                &weights,
                &x,
                1e-6,
                &format!("q8_0 {n_blocks}/{seed}"),
            );
        }
    }

    /// Q8_0's quants are SIGNED. Reading them unsigned agrees with
    /// itself and with nothing else.
    #[test]
    fn the_q8_0_quants_are_read_signed() {
        let mut row = q8_0_row(1, 1);
        for i in 0..32 {
            row[2 + i] = 0xF0; // -16 signed, 240 unsigned
        }
        let x = vec![1.0f32; 32];
        let got = q8_0_matvec_coalesced_row(&row, &x, 1);
        assert!(
            got < 0.0,
            "signed quants must give a negative sum, got {got}"
        );
    }
}
