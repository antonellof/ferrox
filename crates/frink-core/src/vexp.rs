//! The `ggml_v_expf` polynomial, shared.
//!
//! `attention` (softmax) and `matmul` (GELU/SiLU) each had a private
//! copy of these nine constants. They were byte-identical, and nine
//! duplicated magic numbers in two hot kernels is a drift hazard: a
//! correction applied to one copy and not the other is invisible until
//! two paths disagree about a number.
//!
//! What is NOT shared is the clamp, and that is deliberate. A softmax
//! argument is always `<= 0`, so `attention` clamps below only. A GELU
//! or SiLU argument is unbounded and appears as a DENOMINATOR, so a
//! saturated exponential there divides a numerator with no bound of its
//! own: `matmul` must select zero above the clamp rather than merely
//! clamp. Each file keeps its own clamp constant next to the code that
//! relies on it.
//!
//! Source: `ggml/src/ggml-cpu/vec.h`, ARM optimized-routines `expf`.

/// `0x1.8p23`: adding this to `x*log2(e)` rounds it to an integer and
/// parks that integer in the mantissa's low bits.
pub const EXP_SHIFT: f32 = 12582912.0;
/// `log2(e)`, `0x1.715476p+0`.
pub const EXP_LOG2E: f32 = std::f32::consts::LOG2_E;
/// High half of `ln 2`, `0x1.62e4p-1`, chosen with trailing zero bits
/// so `n * ln2_hi` is exact in `f32`.
pub const EXP_LN2_HI: f32 = 0.693_145_75;
/// Low half of `ln 2`, `0x1.7f7d1cp-20`.
pub const EXP_LN2_LO: f32 = 1.428_606_8e-6;
/// Minimax coefficients for `e^b - 1` on `[-ln2/2, ln2/2]`:
/// `0x1.ffffecp-1`, `0x1.fffdb6p-2`, `0x1.555e66p-3`, `0x1.573e2ep-5`,
/// `0x1.0e4020p-7`.
pub const EXP_C0: f32 = 0.999_999_4;
pub const EXP_C1: f32 = 0.499_991_27;
pub const EXP_C2: f32 = 0.166_683_96;
pub const EXP_C3: f32 = 0.041_899_767;
pub const EXP_C4: f32 = 0.008_247_39;
