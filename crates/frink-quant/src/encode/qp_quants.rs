//! llama.cpp's `make_qp_quants` (`ggml/src/ggml-quants.c:899` at
//! b7650): the NON-NEGATIVE symmetric fit `x[i] ~= scale * L[i]`,
//! `L[i]` in `0..=nmax`, with a per-element importance weight and a
//! greedy per-code refinement after the grid search.
//!
//! The importance-matrix Q4_K and Q5_K encoders
//! (`quantize_row_q4_K_impl` at `:1376`, `quantize_row_q5_K_impl` at
//! `:1581`) use it for stage 2 of the super-block fit, quantizing the
//! eight sub-block scales and the eight sub-block mins to 6 bits
//! against a single `d`/`dmin`. The plain encoders' stage 2 is
//! `63/max`, one line, and that is the whole reason the imatrix
//! encoders are a different function upstream rather than a `weights`
//! argument on the same one. Q2_K and the IQ tiers reach this function
//! too, so it takes upstream's `n`, `nmax` and weights rather than
//! being specialised to eight scales and 63.
//!
//! # Every `mul_add` here is load-bearing
//!
//! Same rule as [`super::fit`]: the C is compiled with contraction, so
//! each `acc += a*b*c` is one fused multiply-add of the rounded `a*b`
//! with `c`, and `x - scale*l` is one fused negate-multiply-add. The
//! refinement loop compares `slx*slx*suml2 > sumlx*sumlx*sl2`, which
//! is a near-tie often enough that one rounding decides whether a code
//! moves, and a moved code changes the returned scale and with it the
//! bytes of the whole super-block.
//!
//! # `L` is `uint8_t` and the fit reads it back
//!
//! The codes are stored into `uint8_t L[]` from an `int` that can be
//! NEGATIVE: `x` here is a list of sub-block scales, and a sub-block's
//! least-squares scale is negative for a fraction of a percent of real
//! super-blocks (see the note in [`super::fit::fit_qk_super_block`]).
//! C's conversion wraps `-3` to `253`, and the refinement loop then
//! reads `L[i]` back as 253 in `w*x[i]*L[i]`. Rust's `as u8` on an
//! `i32` is the same wrap, and every read of `L[i]` below goes through
//! the `u8`, so the arithmetic sees what the C sees. Clamping the code
//! to `0..=nmax` before storing it would be tidier and would write a
//! different file.

use super::fit::{nearest_int, GROUP_MAX_EPS};

/// Fit `x[i] ~= scale * L[i]` with `L[i]` in `0..=nmax`, minimising the
/// `quant_weights`-weighted squared error, and return `scale`.
///
/// `x` is non-negative in every upstream call except the negative
/// sub-block-scale case described in the module doc, which is handled
/// by reproducing the C's `uint8_t` wrap rather than by clamping.
pub(crate) fn make_qp_quants(x: &[f32], l: &mut [u8], nmax: i32, quant_weights: &[f32]) -> f32 {
    let n = x.len();
    debug_assert_eq!(l.len(), n);
    debug_assert_eq!(quant_weights.len(), n);

    // `MAX(max, x[i])` starting from 0: a negative entry never raises
    // it, and NaN compares false and is skipped, both as in the C.
    let mut max = 0f32;
    for &v in x {
        if v > max {
            max = v;
        }
    }
    if max < GROUP_MAX_EPS {
        l[..n].fill(0);
        return 0.0;
    }

    let mut iscale = nmax as f32 / max;
    for i in 0..n {
        // Stored without a clamp, so a negative `x[i]` wraps.
        l[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0f32;
    for i in 0..n {
        // `x[i] - scale*L[i]`, contracted: one rounding.
        let diff = (-scale).mul_add(l[i] as f32, x[i]);
        let w = quant_weights[i];
        best_mse = (w * diff).mul_add(diff, best_mse);
    }
    for is in -4..=4i32 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0f32;
        for i in 0..n {
            // The candidate code is clamped in `int` and used as an
            // `int` here -- a negative one stays negative in the error
            // term -- unlike the stored `L[i]` above.
            let li = nearest_int(iscale_is * x[i]).min(nmax);
            let diff = (-scale_is).mul_add(li as f32, x[i]);
            let w = quant_weights[i];
            mse = (w * diff).mul_add(diff, mse);
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }

    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let li = nearest_int(iscale * x[i]).min(nmax);
        l[i] = li as u8;
        let w = quant_weights[i];
        // The accumulators use the `int` code; `L[i]` holds its wrap.
        sumlx = (w * x[i]).mul_add(li as f32, sumlx);
        suml2 = (w * li as f32).mul_add(li as f32, suml2);
    }
    for _itry in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = quant_weights[i];
            let li = l[i] as f32;
            // `sumlx - w*x[i]*L[i]` and `suml2 - w*L[i]*L[i]`, each one
            // fused negate-multiply-add of a rounded product.
            let mut slx = (-(w * x[i])).mul_add(li, sumlx);
            let mut sl2 = (-(w * li)).mul_add(li, suml2);
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int(x[i] * sl2 / slx).min(nmax);
                if new_l != l[i] as i32 {
                    slx = (w * x[i]).mul_add(new_l as f32, slx);
                    sl2 = (w * new_l as f32).mul_add(new_l as f32, sl2);
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An all-zero (or all-negative) input is the `GROUP_MAX_EPS` early
    /// return: zero codes and a scale of exactly 0, which the caller
    /// stores as `d = 0` and which stage 3 then reads as "skip".
    #[test]
    fn an_input_with_no_positive_entry_yields_zero_codes_and_zero_scale() {
        let mut l = [9u8; 8];
        assert_eq!(make_qp_quants(&[0.0; 8], &mut l, 63, &[1.0; 8]), 0.0);
        assert_eq!(l, [0u8; 8]);
        let mut l = [9u8; 8];
        assert_eq!(make_qp_quants(&[-0.5; 8], &mut l, 63, &[1.0; 8]), 0.0);
        assert_eq!(l, [0u8; 8]);
    }

    /// The weights actually steer the fit. Two calls over the same
    /// scales with the weight moved from one entry to another must
    /// choose different codes or a different scale; if they did not,
    /// `sw` in the imatrix encoders would be decoration.
    #[test]
    fn the_importance_weights_change_the_fit() {
        let x = [
            0.031f32, 0.0155, 0.0071, 0.0203, 0.0119, 0.0298, 0.0043, 0.0176,
        ];
        let mut w_a = [1.0f32; 8];
        w_a[2] = 40.0;
        let mut w_b = [1.0f32; 8];
        w_b[0] = 40.0;
        let mut l_a = [0u8; 8];
        let mut l_b = [0u8; 8];
        let s_a = make_qp_quants(&x, &mut l_a, 63, &w_a);
        let s_b = make_qp_quants(&x, &mut l_b, 63, &w_b);
        assert!(
            s_a != s_b || l_a != l_b,
            "moving the weight changed nothing: {s_a} {l_a:?} vs {s_b} {l_b:?}"
        );
    }

    /// A negative entry beside positive ones is stored WRAPPED, as the
    /// C's `uint8_t` conversion does, not clamped to 0. The Q4_K
    /// imatrix encoder packs that byte straight into the file, so the
    /// wrap is a byte fact, not a curiosity.
    #[test]
    fn a_negative_entry_is_stored_as_its_uint8_wrap_not_clamped() {
        let x = [0.02f32, 0.01, -0.001, 0.015, 0.012, 0.018, 0.005, 0.009];
        let mut l = [0u8; 8];
        make_qp_quants(&x, &mut l, 63, &[1.0; 8]);
        assert!(l[2] > 63, "expected a wrapped negative code, got {}", l[2]);
        assert!(l.iter().enumerate().all(|(i, &c)| i == 2 || c <= 63));
    }
}
