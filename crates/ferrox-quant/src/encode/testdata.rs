//! The one f32 fixture every K-quant golden in this crate is built
//! from, and the developer tool that hands it to the C harness.
//!
//! ONE fixture, three formats. The alternative -- a `sample_input` next
//! to each encoder -- is two structures that must agree with nothing
//! enforcing it: the day one copy's seed is nudged, its golden is
//! regenerated against different input and the comment claiming all
//! three encoders see the same data becomes false without a test going
//! red. It also means one C harness run produces all three goldens from
//! one dump, so they cannot be generated from different bytes.
//!
//! The input lives here and ONLY here. A C harness that re-derived the
//! same values from a copy of the generator would be the same bug shape
//! one level up, and it would silently compare two different inputs the
//! day one copy drifted.

use half::f16;

use super::fit::{QK_SUBS, QK_SUB_ELEMS};
use crate::Q4_K_BLOCK_ELEMS;

/// Four 256-element super-blocks of deterministic, **f16-shaped**
/// input, built so that every branch of the references a plausible
/// rewrite would get wrong is exercised at least once.
///
/// f16-shaped is not decoration. Step 1 of this work learned it the
/// expensive way: its Q8_0 golden was documented as catching
/// `v * (1/d)` versus `v / d` and did not, because over uniform f32
/// noise the two spellings agree for 8192 consecutive values. f16's
/// 11-bit mantissa lands on rounding boundaries constantly, and real
/// weights are f16, so the fixture is f16.
///
/// The 32-element sub-block roster, by index (32 sub-blocks of 32
/// values). The indices are Q4_K/Q5_K's; Q6_K reads the same buffer in
/// 16-element groups, so each entry below covers two of its groups:
///
/// * 8 -- all zero: `max == min`, the `make_qkx2_quants` early return
///   that fills the codes with 0 and reports a scale of 0; and, for
///   Q6_K, `amax < GROUP_MAX_EPS` in `make_qx_quants`.
/// * 9 -- constant non-zero: `max == min` again, but with a min that is
///   clamped to 0 because it is positive.
/// * 10 -- all positive: exercises `if (min > 0) min = 0`.
/// * 11 -- all negative: `max` is negative and `min` is not clamped --
///   and for Q6_K, `max` negative makes `iscale` positive, which is the
///   sign case a symmetric fit gets wrong quietly.
/// * 17 -- four orders of magnitude smaller than its super-block's
///   neighbours, so its 6-bit scale rounds to **zero** and stage 3
///   skips it. The codes written for it are the ones stage 1 left in
///   `l`; an encoder that clears `l` per sub-block writes 32 different
///   bytes here and nowhere else.
/// * everything else -- weight-like noise at one of four gains, so
///   sub-blocks within a super-block disagree about scale and the
///   6-bit scale quantization actually has to do something.
///
/// **The seed is not decorative either.** Two of the references'
/// decisions -- `nearest_int`'s round-half-to-even and the
/// `this_min > 0` clamp inside the least-squares step -- only show up
/// on some data, and the first seed tried exercised neither: the whole
/// Q4_K golden stayed green with `f32::round` substituted for
/// `nearest_int`. This one was picked by encoding 3999 candidate
/// fixtures twice, once with each spelling of every decision in the
/// reference, and keeping a seed where all of them differ. 255 of the
/// 3999 qualify, so this is a fixture chosen to be able to fail, not a
/// seed fitted to one assertion.
pub(crate) fn k_quant_fixture() -> Vec<f32> {
    const GAINS: [f32; 4] = [0.02, 0.05, 0.1, 0.25];
    let mut state: u32 = 0xb54c_da26;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // [-1, 1)
        ((state >> 8) as f32 / 8_388_608.0) - 1.0
    };
    let mut out = Vec::with_capacity(4 * Q4_K_BLOCK_ELEMS);
    for sub in 0..4 * QK_SUBS {
        for _ in 0..QK_SUB_ELEMS {
            let v = next();
            let shaped = match sub {
                8 => 0.0,
                9 => 0.125,
                10 => v.abs() * 0.05 + 0.01,
                11 => -(v.abs() * 0.05 + 0.01),
                17 => v * 1e-4,
                _ => v * GAINS[sub % GAINS.len()],
            };
            out.push(f16::from_f32(shaped).to_f32());
        }
    }
    out
}

/// Regenerates every K-quant golden in this crate. Ignored, because it
/// needs a llama.cpp checkout: it writes [`k_quant_fixture`] as raw
/// little-endian f32 to `$FERROX_K_QUANT_FIXTURE_OUT`, which the C
/// harness described in the PR body then feeds to llama.cpp's own
/// `quantize_row_q4_K_ref`, `quantize_row_q5_K_ref` and
/// `quantize_row_q6_K_ref`.
#[test]
#[ignore = "developer tool: dumps the fixture the C golden harness reads"]
fn dump_the_fixture_the_c_harness_reads() {
    let path = std::env::var("FERROX_K_QUANT_FIXTURE_OUT")
        .expect("set FERROX_K_QUANT_FIXTURE_OUT to the path to write");
    let mut bytes = Vec::new();
    for v in k_quant_fixture() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}
