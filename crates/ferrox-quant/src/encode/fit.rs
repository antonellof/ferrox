//! The per-sub-block fitting helpers every K-quant encoder is built
//! from, transcribed from llama.cpp b7650's `ggml/src/ggml-quants.c`:
//!
//! * [`nearest_int`] (`ggml-quants.c:444`), the rounding every encoder
//!   shares;
//! * [`make_qkx2_quants`] (`ggml-quants.c:622`), the affine
//!   `scale * L - min` fit Q4_K and Q5_K use per 32 weights;
//! * [`make_qx_quants`] (`ggml-quants.c:451`), the symmetric `scale * L`
//!   fit Q6_K uses per 16 weights;
//! * [`fit_qk_super_block`], the three-stage Q4_K/Q5_K super-block flow
//!   (`quantize_row_q4_K_ref` at `ggml-quants.c:1280`,
//!   `quantize_row_q5_K_ref` at `ggml-quants.c:1467`) that differs
//!   between the two formats by exactly four numbers.
//!
//! One module, because the alternative is what this repo keeps paying
//! for: a copy of `make_qkx2_quants` in each of `q4_k.rs` and `q5_k.rs`
//! that agree today and drift the first time one of them is corrected.
//! Q5_K is the same super-block fit as Q4_K with `nmax = 31` and a
//! different candidate grid, so it is a CALL into the same code, not a
//! second transcription with the constants changed.
//!
//! Deviation from upstream, shown not to change a byte by the goldens
//! in `q4_k`, `q5_k` and `q6_k`: `nearest_int`'s
//! `assert(fabsf(fval) <= 4194303.f)` is not reproduced. It is compiled
//! out of the release `libggml` that `llama-quantize` actually links,
//! so asserting here would make ferrox stop where llama.cpp proceeds --
//! a refusal that fires on input llama.cpp handles is not coverage, it
//! is a different tool.
//!
//! # Every `mul_add` here is load-bearing. Do not "simplify" one.
//!
//! `sumlx += w*x[i]*l` in the C is **one fused multiply-add**, not a
//! multiply followed by an add: the compiler that builds `libggml`
//! contracts it, so the intermediate product is never rounded to f32.
//! Rust does not contract, so every such site is spelled `mul_add`
//! explicitly. Writing `sumlx += w * x[i] * l as f32` instead is one
//! rounding more, and that rounding is not cosmetic: these fits choose
//! between candidate scales with `sumlx*sumlx > best*suml2`, a
//! comparison that is a near-tie often enough that ONE ulp flips which
//! candidate wins and rewrites the whole super-block.
//!
//! Measured on an F16 Llama-3.2-1B, against the installed
//! `llama-quantize` b7650: with these `mul_add`s, ferrox writes
//! byte-identical files -- 0 of 3244032 Q4_K super-blocks differ, 0 of
//! 3244032 Q5_K, 0 of 4827136 Q6_K. Remove them and it is 1.15%, 0.15%
//! and 1.39% respectively. That is the entire difference between "a
//! file llama.cpp would have written" and "a file that decodes to
//! similar numbers".
//!
//! Nine of the thirteen sites have a real-weight super-block in
//! `testdata::REAL_WEIGHT_BLOCKS` that turns a golden red when that one
//! `mul_add` is removed; the fixture doc names the four that do not and
//! why. Synthetic noise pins NONE of them, which is how they were
//! nearly shipped wrong.

use half::f16;

use crate::Q4_K_SCALE_BYTES;

/// Elements per Q4_K/Q5_K sub-block, and sub-blocks per super-block.
pub(crate) const QK_SUB_ELEMS: usize = 32;
pub(crate) const QK_SUBS: usize = 8;

/// `GROUP_MAX_EPS` (`ggml-quants.c:16`): below this, a group is "all
/// zero" to [`make_qx_quants`].
pub(crate) const GROUP_MAX_EPS: f32 = 1e-15;

/// ggml's `nearest_int`: add 1.5 * 2^23 so the mantissa's low bits hold
/// the rounded integer, then read them back out.
///
/// This is **round-half-to-even**, because it is the FPU's own rounding
/// mode that does the work. `f32::round` is round-half-away-from-zero
/// and disagrees on every exact tie -- and ties are not rare here: the
/// candidate inverse scales in [`make_qkx2_quants`] walk a 0.1-wide
/// grid, so `iscale * (x - min)` lands on `.5` constantly.
#[inline]
pub(crate) fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0f32;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// llama.cpp's `make_qkx2_quants`: fit `x[i] ~= scale * L[i] - the_min`
/// with `L[i]` in `0..=nmax`, minimising the `weights`-weighted error.
///
/// Returns `(scale, the_min)` and fills `l`. `laux` is scratch, passed
/// in rather than allocated because the C does the same and this runs
/// once per 32 weights of the checkpoint.
///
/// The signature is upstream's, `use_mad` and all: Q2_K passes `true`
/// with `n = 16`, Q4_K passes `nmax = 15`, Q5_K passes `nmax = 31`.
/// Keeping the parameters means the next K-quant is a call, not a copy
/// of this function with two constants changed -- which is how this
/// repo has lost a model feature eight times.
#[allow(clippy::too_many_arguments)]
pub(crate) fn make_qkx2_quants(
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    laux: &mut [u8],
    nmax: i32,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> (f32, f32) {
    let n = x.len();
    debug_assert_eq!(weights.len(), n);
    debug_assert_eq!(l.len(), n);
    debug_assert!(laux.len() >= n);

    // Deliberately not `min.min(x[i])` / `max.max(x[i])`. They differ
    // from the C comparisons only when `x[0]` is NaN -- Rust's
    // `f32::min` returns the non-NaN operand, `x[i] < NaN` is false and
    // keeps the NaN -- so no fixture can tell them apart on a real
    // checkpoint. The C's shape is kept anyway, because a checkpoint
    // with a NaN weight should produce llama.cpp's bytes rather than
    // politely different ones. Same choice, same reason, as the `amax`
    // fold in the Q8_0 encoder next door.
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x = w.mul_add(x[i], sum_x);
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l[..n].fill(0);
        return (0.0, -min);
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = li as u8;
        let diff = scale.mul_add(l[i] as f32, min) - x[i];
        let diff = if use_mad { diff.abs() } else { diff * diff };
        best_error = weights[i].mul_add(diff, best_error);
    }
    if nstep < 1 {
        return (scale, -min);
    }

    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = li as u8;
            let w = weights[i];
            sum_l = w.mul_add(li as f32, sum_l);
            sum_l2 = (w * li as f32).mul_add(li as f32, sum_l2);
            sum_xl = (w * li as f32).mul_add(x[i], sum_xl);
        }
        let det = sum_w.mul_add(sum_l2, -(sum_l * sum_l));
        if det > 0.0 {
            let mut this_scale = sum_w.mul_add(sum_xl, -(sum_x * sum_l)) / det;
            let mut this_min = sum_l2.mul_add(sum_x, -(sum_l * sum_xl)) / det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let diff = this_scale.mul_add(laux[i] as f32, this_min) - x[i];
                let diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error = weights[i].mul_add(diff, cur_error);
            }
            if cur_error < best_error {
                l[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// The importance weight [`make_qx_quants`] gives element `i`.
///
/// Spelled ONCE and called from all three loops that need it, because
/// upstream spells the same ternary chain out three times and two of
/// the three are inside the candidate search. Three copies of one
/// weight rule is the shape this repo names first, and here it is
/// upstream's own.
#[inline]
fn qx_weight(x: &[f32], qw: Option<&[f32]>, rmse_type: i32, i: usize) -> f32 {
    match qw {
        Some(qw) => qw[i],
        None => match rmse_type {
            1 => x[i] * x[i],
            2 => 1.0,
            3 => x[i].abs(),
            _ => x[i].abs().sqrt(),
        },
    }
}

/// llama.cpp's `make_qx_quants` (`ggml-quants.c:451`): fit
/// `x[i] ~= scale * (L[i] - nmax)` with `L[i]` in `0..2*nmax`, that is a
/// SYMMETRIC fit with no min, which is what Q6_K's 16-element
/// sub-blocks use (`nmax = 32`, so the codes are `-32..=31` plus 32).
///
/// The signature is upstream's. `rmse_type` selects the error weight
/// (`1` is `x^2`, which Q6_K uses; `0` skips the search entirely, which
/// Q3_K uses; a negative value returns early with a blended scale) and
/// `qw` is the importance-matrix weight, `None` for the plain encoders.
/// Q6_K only ever calls this one way, and the other arms are kept
/// because Q3_K and the imatrix variants reach the same function with
/// different arguments -- a second copy with the arms removed is how
/// two encoders come to disagree about one fit. Only the `rmse_type=1,
/// qw=None` path is covered by a golden, and saying so is better than
/// implying otherwise.
pub(crate) fn make_qx_quants(
    x: &[f32],
    l: &mut [i8],
    nmax: i32,
    rmse_type: i32,
    qw: Option<&[f32]>,
) -> f32 {
    let n = x.len();
    debug_assert_eq!(l.len(), n);
    let mut max = 0f32;
    let mut amax = 0f32;
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l[..n].fill(0);
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let li = nearest_int(iscale * x[i]);
            l[i] = (nmax + li.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    // Upstream flips the sign of `rmse_type` in place and then keeps
    // using it to pick the weight, so the weight for a negative
    // `rmse_type` is the POSITIVE one's. Shadowing reproduces that
    // without a second variable that could be read in the wrong order.
    let (rmse_type, return_early) = if rmse_type < 0 {
        (-rmse_type, true)
    } else {
        (rmse_type, false)
    };
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let li = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l[i] = (li + nmax) as i8;
        let w = qx_weight(x, qw, rmse_type, i);
        sumlx = (w * x[i]).mul_add(li as f32, sumlx);
        suml2 = (w * li as f32).mul_add(li as f32, suml2);
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 {
            0.5 * (scale + 1.0 / iscale)
        } else {
            1.0 / iscale
        };
    }
    let mut best = scale * sumlx;
    for is in -9..=9i32 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let li = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            let w = qx_weight(x, qw, rmse_type, i);
            sumlx = (w * x[i]).mul_add(li as f32, sumlx);
            suml2 = (w * li as f32).mul_add(li as f32, suml2);
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let li = nearest_int(iscale * x[i]);
                l[i] = (nmax + li.clamp(-nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// The `sqrt(mean(x^2)) + |x|` weights Q4_K and Q5_K hand to
/// [`make_qkx2_quants`] for one 32-element sub-block.
pub(crate) fn qk_sub_block_weights(xs: &[f32], weights: &mut [f32; QK_SUB_ELEMS]) {
    let mut sum_x2 = 0f32;
    for &v in xs {
        sum_x2 += v * v;
    }
    let av_x = (sum_x2 / QK_SUB_ELEMS as f32).sqrt();
    for (w, &v) in weights.iter_mut().zip(xs) {
        *w = av_x + v.abs();
    }
}

/// The fitted super-block a Q4_K or Q5_K encoder packs: the two f16
/// super-scales, the 12 bytes of 6-bit sub-block scales and mins, and
/// the per-element codes.
pub(crate) struct QkSuperBlock {
    pub d: f16,
    pub dmin: f16,
    pub packed: [u8; Q4_K_SCALE_BYTES],
    pub l: [u8; QK_SUBS * QK_SUB_ELEMS],
}

/// The candidate grid and code range that distinguish one
/// `make_qkx2_quants`-based super-block format from another. Q4_K and
/// Q5_K differ by these four numbers and NOTHING else, which is why
/// they share [`fit_qk_super_block`] instead of having a transcription
/// each.
#[derive(Clone, Copy)]
pub(crate) struct QkFit {
    /// Largest code: 15 for Q4_K, 31 for Q5_K.
    pub nmax: i32,
    /// `rmin`, `rdelta`, `nstep` for `make_qkx2_quants`:
    /// `(-1.0, 0.1, 20)` for Q4_K, `(-0.5, 0.1, 15)` for Q5_K.
    pub rmin: f32,
    pub rdelta: f32,
    pub nstep: i32,
}

/// The three-stage super-block fit `quantize_row_q4_K_ref`
/// (`ggml-quants.c:1280`) and `quantize_row_q5_K_ref`
/// (`ggml-quants.c:1467`) share, parameterised by [`QkFit`].
///
/// 1. Each of the 8 sub-blocks of 32 gets an **iterative** affine fit:
///    the candidate inverse scales are tried, each one re-solves a
///    weighted least-squares for (scale, min) from the integer codes it
///    produced, and the lowest weighted squared error wins.
/// 2. The 8 scales and 8 mins are themselves quantized to 6 bits
///    against the super-block's `d`/`dmin` and packed into 12 bytes.
/// 3. The codes are then recomputed **against the 6-bit-rounded** scale
///    and min, not against the fit from stage 1.
///
/// `l` is deliberately carried from stage 1 into stage 3. Stage 3 skips
/// any sub-block whose reconstructed `d` rounded to zero (`if (!d)
/// continue;` upstream), and the codes then written are the ones stage
/// 1 left behind -- NOT zeros. Clearing `l` per sub-block reads as
/// tidier and writes a different file.
pub(crate) fn fit_qk_super_block(
    block: &[f32; QK_SUBS * QK_SUB_ELEMS],
    fit: QkFit,
) -> QkSuperBlock {
    let mut l = [0u8; QK_SUBS * QK_SUB_ELEMS];
    let mut laux = [0u8; QK_SUB_ELEMS];
    let mut weights = [0f32; QK_SUB_ELEMS];
    let mut mins = [0f32; QK_SUBS];
    let mut scales = [0f32; QK_SUBS];

    let mut max_scale = 0f32; // deducting the min keeps scales positive
    let mut max_min = 0f32;
    for j in 0..QK_SUBS {
        let lo = QK_SUB_ELEMS * j;
        let xs = &block[lo..lo + QK_SUB_ELEMS];
        qk_sub_block_weights(xs, &mut weights);
        let (scale, min) = make_qkx2_quants(
            xs,
            &weights,
            &mut l[lo..lo + QK_SUB_ELEMS],
            &mut laux,
            fit.nmax,
            fit.rmin,
            fit.rdelta,
            fit.nstep,
            false,
        );
        scales[j] = scale;
        mins[j] = min;
        if scale > max_scale {
            max_scale = scale;
        }
        if min > max_min {
            max_min = min;
        }
    }

    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut packed = [0u8; Q4_K_SCALE_BYTES];
    for j in 0..QK_SUBS {
        // Upstream's `MIN(63, ls)`. It cannot fire on THIS path:
        // `inv_scale` is `63/max_scale` and `max_scale` is the largest
        // of `scales`, so the product is at most 63 plus an ulp and
        // rounds to 63. It is kept because it is what the C says and
        // because the imatrix variants of these encoders
        // (`quantize_row_q4_K_impl` / `quantize_row_q5_K_impl`) reach
        // the same packing from `make_qp_quants`, where the bound is
        // not automatic -- but no fixture here can turn its removal
        // red, and saying so is better than implying the goldens cover
        // it.
        // The cast comes BEFORE the clamp, because upstream's does:
        //
        //     uint8_t ls = nearest_int(inv_scale*scales[j]);
        //     ls = MIN(63, ls);
        //
        // `nearest_int` returns `int`, and storing it in a `uint8_t`
        // truncates to eight bits FIRST. Clamping to 63 and casting
        // afterwards is the same for every value in `0..=255` and
        // different for a negative one: C wraps -1 to 255 and then
        // clamps to 63, this order clamps -1 to -1 and casts to 255.
        //
        // A negative reaches here when a sub-block's least-squares fit
        // returns a negative scale while some other sub-block's is
        // positive, so `inv_scale` is positive and the product is not.
        // Upstream's comment says scales are always positive "as we are
        // deducting the min", which is the assumption this arithmetic
        // quietly does not rely on. Rare, and it was 0.55% of the
        // super-blocks in a real Qwen3-0.6B tensor.
        let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
        let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
        if j < 4 {
            packed[j] = ls;
            packed[j + 4] = lm;
        } else {
            packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            packed[j - 4] |= (ls >> 4) << 6;
            packed[j] |= (lm >> 4) << 6;
        }
    }
    let d = f16::from_f32(max_scale / 63.0);
    let dmin = f16::from_f32(max_min / 63.0);

    // Stage 3 unpacks the 6-bit scale/min with the same
    // `q4_k_scale_min` the READER uses, rather than a second copy of
    // `get_scale_min_k4`. One function means the encoder and the
    // decoder cannot disagree about what was packed. (Q5_K shares
    // Q4_K's packing, which is why one unpacker serves both.)
    for j in 0..QK_SUBS {
        let (sc, m) = crate::q4_k_scale_min(j, &packed);
        let dj = d.to_f32() * sc as f32;
        if dj == 0.0 {
            continue;
        }
        let dm = dmin.to_f32() * m as f32;
        for ii in 0..QK_SUB_ELEMS {
            let idx = QK_SUB_ELEMS * j + ii;
            l[idx] = nearest_int((block[idx] + dm) / dj).clamp(0, fit.nmax) as u8;
        }
    }

    QkSuperBlock { d, dmin, packed, l }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `nearest_int` is round-half-to-even, not `f32::round`. The two
    /// agree everywhere except exact ties, and ties are where the
    /// candidate-grid search lands constantly.
    #[test]
    fn nearest_int_rounds_ties_to_even_like_the_fpu() {
        assert_eq!(nearest_int(0.5), 0);
        assert_eq!(nearest_int(1.5), 2);
        assert_eq!(nearest_int(2.5), 2);
        assert_eq!(nearest_int(-0.5), 0);
        assert_eq!(nearest_int(-1.5), -2);
        assert_eq!(nearest_int(3.7), 4);
        assert_eq!(nearest_int(-3.7), -4);
        // And where they agree, they agree.
        assert_eq!(nearest_int(3.2), 3.2f32.round() as i32);
    }

    /// A group under `GROUP_MAX_EPS` is all-zero to `make_qx_quants`:
    /// codes 0 (NOT `nmax`, which is the code a zero value gets
    /// everywhere else) and a scale of exactly 0. Q6_K's super-block
    /// reads that scale to decide whether to write an all-zero block,
    /// so the two conventions meeting here is load-bearing.
    #[test]
    fn make_qx_quants_reports_an_all_zero_group_with_zero_codes_and_zero_scale() {
        let mut l = [7i8; 16];
        let scale = make_qx_quants(&[0.0; 16], &mut l, 32, 1, None);
        assert_eq!(scale, 0.0);
        assert_eq!(l, [0i8; 16]);
    }

    /// The symmetric fit puts the largest-magnitude element at the
    /// NEGATIVE end of the code range: `iscale = -nmax/max`, so the
    /// element equal to `max` maps to `-nmax`, and a positive `max`
    /// yields a negative scale. Getting the sign convention backwards
    /// dequantizes to the negated tensor, which no error bound catches
    /// on a symmetric distribution.
    #[test]
    fn make_qx_quants_maps_the_extreme_element_to_the_negative_end() {
        let x: [f32; 16] = [
            1.0, -0.5, 0.25, 0.0, 0.125, -0.75, 0.5, -0.25, 0.0625, -0.0625, 0.3, -0.3, 0.9, -0.9,
            0.7, -0.1,
        ];
        let mut l = [0i8; 16];
        let scale = make_qx_quants(&x, &mut l, 32, 1, None);
        assert!(scale < 0.0, "scale {scale}");
        // x[0] = 1.0 is the extreme: code -32, stored as -32 + 32 = 0.
        assert_eq!(l[0], 0);
        // And its mirror image lands on the far side of 32.
        assert!(l[13] > 32, "l[13] = {}", l[13]);
    }

    /// The `x^2` weight (`rmse_type = 1`, the one Q6_K uses) is not the
    /// uniform weight. A group with one large element and fifteen tiny
    /// ones fits the large element under `rmse_type = 1` and splits the
    /// difference under `rmse_type = 2`, so an encoder that passed the
    /// wrong `rmse_type` would still produce plausible output.
    #[test]
    fn the_rmse_type_actually_selects_a_different_weight() {
        let mut x = [0.001f32; 16];
        x[0] = 1.0;
        x[7] = -0.4;
        let mut l1 = [0i8; 16];
        let mut l2 = [0i8; 16];
        let s1 = make_qx_quants(&x, &mut l1, 32, 1, None);
        let s2 = make_qx_quants(&x, &mut l2, 32, 2, None);
        assert_ne!(
            s1, s2,
            "rmse_type 1 and 2 produced the same scale; this input no \
             longer distinguishes the weights"
        );
    }

    /// The two Q4_K/Q5_K grids are not interchangeable. If they were,
    /// `QkFit` would be decoration and one encoder could quietly use
    /// the other's constants.
    #[test]
    fn the_q4_k_and_q5_k_candidate_grids_fit_the_same_data_differently() {
        let mut block = [0f32; QK_SUBS * QK_SUB_ELEMS];
        let mut state: u32 = 0x2f6b_1c05;
        for v in block.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *v = f16::from_f32(((state >> 8) as f32 / 8_388_608.0 - 1.0) * 0.07).to_f32();
        }
        let q4 = fit_qk_super_block(
            &block,
            QkFit {
                nmax: 15,
                rmin: -1.0,
                rdelta: 0.1,
                nstep: 20,
            },
        );
        let q5 = fit_qk_super_block(
            &block,
            QkFit {
                nmax: 31,
                rmin: -0.5,
                rdelta: 0.1,
                nstep: 15,
            },
        );
        assert_ne!(q4.d, q5.d);
        assert_ne!(q4.packed, q5.packed);
        // Q5_K's codes use the top half of the range, Q4_K's cannot.
        assert!(q5.l.iter().any(|&c| c > 15));
        assert!(q4.l.iter().all(|&c| c <= 15));
    }
}
