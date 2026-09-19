//! **ALiBi** -- the per-head linear position bias llama.cpp adds inside
//! `ggml_soft_max_ext` when `hparams.f_max_alibi_bias > 0`.
//!
//! # What it is
//!
//! `ggml_soft_max_ext(kq, kq_mask, scale, max_bias)` computes
//! `softmax(scale * kq + slope_h * mask)`, where the mask a model with
//! `use_alibi` fills is `-|p_key - p_query|` on the visible entries and
//! `-inf` elsewhere (`llama-kv-cache.cpp:1673-1676`), and the slope is
//! `ggml-cpu/ops.cpp:5489-5508`:
//!
//! ```text
//! n_head_log2 = 2^floor(log2(n_head))
//! m0 = 2^(-max_bias / n_head_log2)
//! m1 = 2^(-(max_bias / 2) / n_head_log2)
//! slope_h = h < n_head_log2 ? m0^(h + 1) : m1^(2 (h - n_head_log2) + 1)
//! ```
//!
//! So a query at `p_q` sees key `p_k <= p_q` with `slope_h * (p_k -
//! p_q)` added to its scaled score, after any softcap (the softcapped
//! graphs scale and `tanh` BEFORE `ggml_soft_max_ext`; none of them
//! uses ALiBi, but the order is the graph's). [`slopes`] is the
//! formula; the kernels in `attention` take the result as an optional
//! per-head slice and rotate nothing for such a model.
//!
//! The graphs that set `f_max_alibi_bias` are the caller's business
//! (`frink_models::alibi`); this module is the arithmetic.

/// One slope per query head, `n_heads` long, for a `max_bias` that is
/// positive. Returns `None` for a non-positive `max_bias`, which is
/// llama.cpp's "no ALiBi" (`slope = 1.0` on a mask that is then
/// `0 / -inf`, i.e. no bias at all).
pub fn slopes(n_heads: usize, max_bias: f32) -> Option<Vec<f32>> {
    // `max_bias <= 0.0` is false for NaN, and NaN is "no ALiBi" too.
    if max_bias.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || n_heads == 0 {
        return None;
    }
    let n_head_log2 = 1usize << (usize::BITS - 1 - n_heads.leading_zeros());
    let m0 = 2f32.powf(-max_bias / n_head_log2 as f32);
    let m1 = 2f32.powf(-(max_bias / 2.0) / n_head_log2 as f32);
    Some(
        (0..n_heads)
            .map(|h| {
                if h < n_head_log2 {
                    m0.powi(h as i32 + 1)
                } else {
                    m1.powi(2 * (h - n_head_log2) as i32 + 1)
                }
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The textbook case: a power-of-two head count and `max_bias 8`
    /// gives `2^(-8 (h+1) / n_head)`, i.e. `1/2, 1/4, ..., 1/256` for
    /// eight heads.
    #[test]
    fn a_power_of_two_head_count_is_the_geometric_sequence() {
        let s = slopes(8, 8.0).unwrap();
        let want = [
            0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
        ];
        for (g, w) in s.iter().zip(want) {
            assert!((g - w).abs() < 1e-7, "{g} vs {w}");
        }
    }

    /// The non-power-of-two rule llama.cpp copies from the paper: the
    /// first `n_head_log2` heads take `m0`, the rest interleave `m1` at
    /// odd powers. Twelve heads (GPT-2-small's count, MPT-7B is 32).
    #[test]
    fn a_non_power_of_two_head_count_takes_the_second_base_for_the_tail() {
        let s = slopes(12, 8.0).unwrap();
        // n_head_log2 = 8: m0 = 2^-1, m1 = 2^-0.5.
        let m1 = 2f32.powf(-0.5);
        for (h, got) in s.iter().enumerate() {
            let want = if h < 8 {
                0.5f32.powi(h as i32 + 1)
            } else {
                m1.powi(2 * (h as i32 - 8) + 1)
            };
            assert!((got - want).abs() < 1e-7, "head {h}: {got} vs {want}");
        }
    }

    #[test]
    fn a_zero_max_bias_is_no_alibi() {
        assert!(slopes(8, 0.0).is_none());
        assert!(slopes(8, -1.0).is_none());
        assert!(slopes(0, 8.0).is_none());
    }
}
