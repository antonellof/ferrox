//! Walsh–Hadamard transform helpers for TurboQuant-style KV compression.
//!
//! Host FWHT + group quant; Metal `FRINK_CTK=turbo4` stores the 4-bit
//! groups (WHT optional on host upload). turbo8 aliases ggml Q8_0 on Metal.
//! Algorithm follows the public TurboQuant line (randomized Hadamard +
//! per-group scale).
//!
//! In-place FWHT on `x` of length `n = 2^k`. Output is unnormalized
//! (each butterfly is `a+b`, `a-b`); callers that need orthonormal
//! Hadamard divide by `sqrt(n)`.

/// In-place Fast Walsh–Hadamard Transform. `x.len()` must be a power of two.
pub fn fwht_inplace(x: &mut [f32]) {
    let n = x.len();
    assert!(
        n.is_power_of_two() && n > 0,
        "fwht length must be 2^k, got {n}"
    );
    let mut h = 1usize;
    while h < n {
        let step = h * 2;
        for i in (0..n).step_by(step) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h = step;
    }
}

/// Orthonormal FWHT: `fwht_inplace` then scale by `1/sqrt(n)`.
pub fn fwht_orthonormal_inplace(x: &mut [f32]) {
    let n = x.len();
    fwht_inplace(x);
    let inv = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Per-group absmax quantize after optional WHT (turbo4-style sketch).
///
/// Packs each group of `group` floats into `group/2` bytes (two 4-bit
/// codes per byte) plus one f32 scale. Returns `(packed, scales)`.
pub fn quantize_turbo4_groups(x: &[f32], group: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(group >= 2 && group.is_power_of_two());
    assert_eq!(x.len() % group, 0);
    let n_groups = x.len() / group;
    let mut packed = Vec::with_capacity(n_groups * (group / 2));
    let mut scales = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let chunk = &x[g * group..(g + 1) * group];
        let mut amax = 0.0f32;
        for &v in chunk {
            amax = amax.max(v.abs());
        }
        let scale = if amax > 0.0 { amax / 7.0 } else { 1.0 };
        scales.push(scale);
        let inv = 1.0 / scale;
        for pair in chunk.as_chunks::<2>().0 {
            let q0 = (pair[0] * inv).round().clamp(-8.0, 7.0) as i8;
            let q1 = (pair[1] * inv).round().clamp(-8.0, 7.0) as i8;
            let n0 = (q0 as u8) & 0x0f;
            let n1 = (q1 as u8) & 0x0f;
            packed.push(n0 | (n1 << 4));
        }
    }
    (packed, scales)
}

/// Inverse of [`quantize_turbo4_groups`] (ignores WHT; dequant only).
pub fn dequantize_turbo4_groups(packed: &[u8], scales: &[f32], group: usize) -> Vec<f32> {
    assert!(group >= 2 && group.is_power_of_two());
    let n_groups = scales.len();
    assert_eq!(packed.len(), n_groups * (group / 2));
    let mut out = Vec::with_capacity(n_groups * group);
    for (g, &scale) in scales.iter().enumerate() {
        let bytes = &packed[g * (group / 2)..(g + 1) * (group / 2)];
        for &b in bytes {
            let q0 = ((b & 0x0f) as i8) << 4 >> 4; // sign-extend 4-bit
            let q1 = ((b >> 4) as i8) << 4 >> 4;
            out.push(q0 as f32 * scale);
            out.push(q1 as f32 * scale);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_involutory_up_to_scale() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.25, -0.5];
        let orig = x.clone();
        fwht_inplace(&mut x);
        fwht_inplace(&mut x);
        // Unnormalized FWHT twice → n * identity.
        let n = orig.len() as f32;
        for (a, b) in orig.iter().zip(x.iter()) {
            assert!((a * n - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn orthonormal_fwht_is_involution() {
        let mut x = vec![0.1, -0.2, 0.3, -0.4];
        let orig = x.clone();
        fwht_orthonormal_inplace(&mut x);
        fwht_orthonormal_inplace(&mut x);
        for (a, b) in orig.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn turbo4_roundtrip_reasonable() {
        let mut x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3).sin()).collect();
        fwht_orthonormal_inplace(&mut x);
        let (packed, scales) = quantize_turbo4_groups(&x, 8);
        let mut y = dequantize_turbo4_groups(&packed, &scales, 8);
        fwht_orthonormal_inplace(&mut y);
        // After inverse WHT, compare to pre-WHT... we mutated x in place.
        // Rebuild original for error check.
        let orig: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut err = 0.0f32;
        for (a, b) in orig.iter().zip(y.iter()) {
            err += (a - b).abs();
        }
        err /= orig.len() as f32;
        assert!(err < 0.15, "mean abs err {err}");
    }
}
