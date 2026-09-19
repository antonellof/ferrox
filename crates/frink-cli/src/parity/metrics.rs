//! The distribution metrics every parity number is made of.
//!
//! They live in their own module because two callers need them and only
//! one of them used to exist. `compare` scores frink against the
//! reference; [`super::calibration`] scores the REFERENCES AGAINST EACH
//! OTHER, which is the comparison that says whether a difference of a
//! given size means anything at all. If each had its own softmax and its
//! own KL the two halves of one verdict would be computed by two
//! spellings of the same arithmetic — the shape this repo keeps
//! shipping.

/// Probabilities from logits, in f64.
///
/// Shift-invariant by construction (the max is subtracted), which is
/// why an additive constant between two engines is not a disagreement.
pub(super) fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut out: Vec<f64> = logits.iter().map(|&v| (v as f64 - max).exp()).collect();
    let sum: f64 = out.iter().sum();
    if sum > 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

/// KL(p || q) in nats.
///
/// Terms where `p` is zero contribute nothing by the `0 ln 0 = 0`
/// convention; clamping `q` away from zero avoids an infinity produced
/// by a single underflowed float rather than by a real disagreement.
pub(super) fn kl(p: &[f64], q: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for (&pi, &qi) in p.iter().zip(q) {
        if pi > 0.0 {
            acc += pi * (pi.max(f64::MIN_POSITIVE) / qi.max(f64::MIN_POSITIVE)).ln();
        }
    }
    acc
}

/// Total variation distance and the largest single-token probability
/// difference, in one pass.
pub(super) fn total_variation(p: &[f64], q: &[f64]) -> (f64, f64) {
    let mut tv = 0.0f64;
    let mut max_delta = 0.0f64;
    for (&pi, &qi) in p.iter().zip(q) {
        let d = (pi - qi).abs();
        tv += d;
        if d > max_delta {
            max_delta = d;
        }
    }
    (0.5 * tv, max_delta)
}

/// Indices sorted by descending probability.
///
/// Ties break by index on both sides, so a tie can never by itself make
/// two engines' orderings look different.
pub(super) fn order_desc(p: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
    idx
}

/// Distance between two f32 in representable values.
///
/// `(a - b).abs()` cannot answer "is this a tie": near 3.0 an absolute
/// gap of 1e-6 is about four ulps, near 3e-8 the same gap is the whole
/// number. The count of representable values between two floats is the
/// only scale-free way to say how close a float comparison was, and a
/// tie-flip verdict is exactly that claim.
///
/// Works by mapping f32 to a monotone integer key: for non-negative
/// values the bit pattern already increases with the value, and for
/// negative values it increases as the value decreases, so the
/// magnitude bits are negated.
pub(super) fn ulps_between(a: f32, b: f32) -> i64 {
    fn key(x: f32) -> i64 {
        let bits = i64::from(x.to_bits() & 0x7fff_ffff);
        if x.is_sign_negative() {
            -bits
        } else {
            bits
        }
    }
    (key(a) - key(b)).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Adjacent floats are one ulp apart wherever they sit on the line,
    /// and the count is scale-free.
    ///
    /// This is the measurement a TIE-FLIP verdict rests on. An absolute
    /// difference cannot make the claim: `3.0` and its neighbour differ
    /// by 2.4e-7 while `3e-8` and its neighbour differ by 1e-15, and
    /// both are one ulp. A tie-flip reported in absolute units is
    /// therefore unfalsifiable, which is what #103 pointed at.
    #[test]
    fn ulps_between_counts_representable_values_not_absolute_distance() {
        for anchor in [3.0f32, 3e-8, -3.0, 1.0, 65_536.0] {
            let next = f32::from_bits(if anchor.is_sign_negative() {
                anchor.to_bits() - 1
            } else {
                anchor.to_bits() + 1
            });
            assert_eq!(
                ulps_between(anchor, next),
                1,
                "{anchor} and its neighbour {next} must be one ulp apart"
            );
        }
        // Symmetric, and zero for equal values.
        assert_eq!(ulps_between(1.5, 1.5), 0);
        assert_eq!(ulps_between(2.0, 1.0), ulps_between(1.0, 2.0));
        // Crossing zero must not wrap: -0.0 and +0.0 are the same
        // value, and a naive bit subtraction would call them 2^31
        // apart, which would turn every sign change into "not a tie".
        assert_eq!(ulps_between(-0.0, 0.0), 0);
        // The two representable values either side of zero are two
        // steps apart, not 2^32: a naive bit subtraction makes any sign
        // change look maximally far, which would turn every flipped
        // near-tie into "not a tie".
        let tiny = f32::from_bits(1);
        assert_eq!(ulps_between(-tiny, tiny), 2);
        // A gap that is genuinely large stays large: 1.0 and 2.0 are a
        // whole binade apart, which is 2^23 representable values.
        assert_eq!(ulps_between(1.0, 2.0), 1 << 23);
    }

    /// KL is ZERO for identical distributions and ASYMMETRIC otherwise.
    ///
    /// Both halves matter to the caller: [`super::super::calibration`]
    /// takes the reference spread as the largest KL over ORDERED pairs
    /// precisely because `kl(a, b)` and `kl(b, a)` are different
    /// numbers, and a spread that quietly picked one direction would be
    /// a smaller band than the evidence supports.
    #[test]
    fn kl_is_zero_for_identical_distributions_and_asymmetric_otherwise() {
        let p = softmax(&[0.1f32, 5.0, -2.0, 3.3]);
        assert_eq!(kl(&p, &p), 0.0);

        let q = softmax(&[0.1f32, 5.0, -2.0, 1.0]);
        assert!(kl(&p, &q) > 0.0 && kl(&q, &p) > 0.0);
        assert_ne!(
            kl(&p, &q),
            kl(&q, &p),
            "if these ever agree the spread's `max over ordered pairs` is pointless"
        );

        // A token the reference gives EXACTLY zero mass contributes
        // nothing, by the `0 ln 0 = 0` convention. Over a 150k
        // vocabulary a logit far enough below the max underflows to a
        // true zero, so this is the ordinary case and not a corner: an
        // unguarded `p * ln(p/q)` evaluates `0 * -inf` there and turns
        // the whole KL into NaN, which every comparison downstream
        // would then read as "not greater than the line".
        let with_a_dead_token = softmax(&[0.0f32, -800.0]);
        assert_eq!(
            with_a_dead_token[1], 0.0,
            "the fixture must actually contain an exact zero"
        );
        let uniform = softmax(&[0.0f32, 0.0]);
        let d = kl(&with_a_dead_token, &uniform);
        assert!(d.is_finite(), "a zero-mass token made the KL {d}");
        assert!((d - std::f64::consts::LN_2).abs() < 1e-12, "{d}");
    }

    /// An additive constant on the logits is not a disagreement, and it
    /// stays not one at a magnitude where `exp` alone would overflow.
    ///
    /// Shift invariance is free algebraically; what is NOT free is
    /// subtracting the max first. `exp(800)` is `inf` and `inf/inf` is
    /// `NaN`, so a softmax without that subtraction turns a shifted
    /// copy of a distribution into a vector of NaN and every downstream
    /// KL into a number no comparison can be read off. The shift here
    /// is large on purpose.
    #[test]
    fn softmax_is_shift_invariant_even_where_exp_would_overflow() {
        let base = [0.5f32, 1.5, -3.0];
        let a = softmax(&base);
        let shifted: Vec<f32> = base.iter().map(|v| v + 800.0).collect();
        let b = softmax(&shifted);
        for (x, y) in a.iter().zip(&b) {
            assert!(x.is_finite() && y.is_finite(), "{x} vs {y}");
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
        assert!((a.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((b.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
