//! MiniMax-01's lightning attention: a linear attention with a
//! per-head exponential decay, as a token-at-a-time recurrence.
//!
//! # What llama.cpp computes
//!
//! `src/models/minimax-01.cpp:315-388` runs it over a whole ubatch at
//! once, with three decay inputs the graph fills on the host
//! (`:105-171`). For head `h` with slope `s_h`, layer scale `c_il`, and
//! positions taken RELATIVE to the lowest one in the chunk:
//!
//! ```text
//! q_decay[i]     = exp(-c s_h (i + 1))
//! k_decay[i]     = exp(-c s_h (n - i - 1))
//! diag[j][i]     = exp(-c s_h (j - i))      for j >= i, else 0
//!
//! out[j]  = (q[j] * q_decay[j]) @ KV      + sum_{i<=j} (q[j].k[i]) diag[j][i] v[i]
//! KV_new  = KV * exp(-c s_h n)            + sum_i (k[i] * k_decay[i])^T v[i]
//! ```
//!
//! where `KV` is `[head_dim, head_dim]` per head.
//!
//! # Why this implements it one token at a time
//!
//! The chunked form and the per-token recurrence are the SAME
//! function, and the equality is worth writing out because it is what
//! lets a decode step and a prefill share one body. Take two tokens in
//! one chunk:
//!
//! ```text
//! chunked: out_1 = q_1 e^{-2cs} KV + (q_1.k_0) e^{-cs} v_0 + (q_1.k_1) v_1
//!          KV'   = KV e^{-2cs} + k_0^T v_0 e^{-cs} + k_1^T v_1
//!
//! per tok: KV_1  = KV e^{-cs} + k_0^T v_0
//!          out_1 = q_1 e^{-cs} KV_1 + (q_1.k_1) v_1
//!                = q_1 e^{-2cs} KV + e^{-cs} (q_1.k_0) v_0 + (q_1.k_1) v_1
//!          KV_2  = KV_1 e^{-cs} + k_1^T v_1
//!                = KV e^{-2cs} + e^{-cs} k_0^T v_0 + k_1^T v_1
//! ```
//!
//! -- term for term. So frink keeps ONE body, [`lightning_step`], and
//! a prefill is that body in a loop. The cost is fp: llama.cpp's chunk
//! sums the intra-chunk term as a matmul while this accumulates it
//! through the state, so the two agree to reduction order and not to
//! the bit.

/// MiniMax-01's per-head decay slopes (`minimax-01.cpp:93-103`).
///
/// `start = ratio = 2^(-2^(-(log2(n_head) - 3)))` and head `h` gets
/// `start * ratio^h`, which is ALiBi's geometric ladder with a
/// different base -- the same shape `frink_core::alibi` computes for
/// the additive bias, and deliberately not shared with it: that one is
/// llama.cpp's `n_head_log2` construction with two ratios for a
/// non-power-of-two head count, and this one is a single geometric
/// series whatever the count.
pub fn slopes(n_head: usize) -> Vec<f32> {
    debug_assert!(n_head > 0);
    let start = 2f32.powf(-(2f32.powf(-((n_head as f32).log2() - 3.0))));
    (0..n_head).map(|h| start * start.powi(h as i32)).collect()
}

/// The layer's scale on every slope (`minimax-01.cpp:288`):
/// `1 - il / (n_layer - 1) + 1e-5`, so the first layer decays at the
/// full slope and the last at almost none.
///
/// `n_layer` is the LOGICAL layer count llama.cpp's graph loops over,
/// and a one-layer model would divide by zero there -- upstream would
/// too, so this refuses to guess and asserts instead of inventing a
/// scale nothing measures.
pub fn slope_scale(il: usize, n_layer: usize) -> f32 {
    assert!(
        n_layer > 1,
        "minimax-01.cpp:288 divides by n_layer - 1; a one-layer model has no scale"
    );
    1.0 - il as f32 / (n_layer - 1) as f32 + 1e-5
}

/// One token through one head's lightning attention, updating the
/// state.
///
/// `q`, `k`, `v` are this head's `head_dim` values; `state` is its
/// `head_dim * head_dim` KV, row-major with the KEY dimension outer --
/// the layout `ggml_mul_mat(kv_old, q)` reads, i.e. `state[a * d + b]`
/// accumulates `k[a] * v[b]`.
///
/// Returns this head's output.
pub fn lightning_step(q: &[f32], k: &[f32], v: &[f32], state: &mut [f32], decay: f32) -> Vec<f32> {
    let d = q.len();
    debug_assert_eq!(k.len(), d);
    debug_assert_eq!(v.len(), d);
    debug_assert_eq!(state.len(), d * d);

    // The inter-chunk term: the decayed query against the state BEFORE
    // this token is folded in.
    let mut out = vec![0.0f32; d];
    for (a, &qa) in q.iter().enumerate() {
        let qa = qa * decay;
        if qa == 0.0 {
            continue;
        }
        let row = &state[a * d..(a + 1) * d];
        for (o, &s) in out.iter_mut().zip(row) {
            *o += qa * s;
        }
    }

    // The intra-chunk term at chunk size one: this token's own
    // `(q . k) v`, with a diagonal decay of `exp(0) = 1`.
    let qk: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
    for (o, &vb) in out.iter_mut().zip(v) {
        *o += qk * vb;
    }

    // And the state carries forward: decayed, plus this token's outer
    // product at a `k_decay` of `exp(0) = 1`.
    for (a, &ka) in k.iter().enumerate() {
        let row = &mut state[a * d..(a + 1) * d];
        for (s, &vb) in row.iter_mut().zip(v) {
            *s = *s * decay + ka * vb;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The slopes are the geometric ladder `minimax-01.cpp:96-102`
    /// builds, and they decrease.
    #[test]
    fn the_slopes_are_the_geometric_ladder() {
        let s = slopes(8);
        assert_eq!(s.len(), 8);
        let start = 2f32.powf(-(2f32.powf(-((8f32).log2() - 3.0))));
        assert!((s[0] - start).abs() < 1e-7);
        assert!((s[1] - start * start).abs() < 1e-7);
        for w in s.windows(2) {
            assert!(w[1] < w[0], "the ladder decreases");
        }
    }

    /// Layer 0 keeps the full slope and the last layer almost none.
    #[test]
    fn the_layer_scale_walks_from_one_to_almost_zero() {
        assert!((slope_scale(0, 4) - 1.000_01).abs() < 1e-6);
        assert!((slope_scale(3, 4) - 1e-5).abs() < 1e-6);
    }

    /// The per-token recurrence equals the chunked form llama.cpp
    /// computes, term for term, over two tokens.
    ///
    /// Written out rather than trusted: the chunked side here is the
    /// formula from `minimax-01.cpp:315-388` with `n = 2`, and if the
    /// two ever disagree it is this file that is wrong.
    #[test]
    fn the_recurrence_equals_the_chunked_form_over_two_tokens() {
        let d = 3;
        let decay = 0.7f32;
        let q0 = [0.3, -0.5, 0.9];
        let q1 = [-0.2, 0.4, 0.1];
        let k0 = [0.6, 0.1, -0.3];
        let k1 = [0.2, -0.7, 0.5];
        let v0 = [1.0, -2.0, 0.5];
        let v1 = [-0.4, 0.8, 1.2];
        let kv0: Vec<f32> = (0..d * d).map(|i| 0.1 * (i as f32) - 0.3).collect();

        let mut state = kv0.clone();
        let out0 = lightning_step(&q0, &k0, &v0, &mut state, decay);
        let out1 = lightning_step(&q1, &k1, &v1, &mut state, decay);

        // The chunked form, spelled out.
        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        let apply = |q: &[f32], scale: f32, kv: &[f32]| -> Vec<f32> {
            (0..d)
                .map(|b| (0..d).map(|a| q[a] * scale * kv[a * d + b]).sum())
                .collect()
        };
        // out_0 = q_0 e^{-cs} KV + (q_0 . k_0) v_0
        let want0: Vec<f32> = apply(&q0, decay, &kv0)
            .iter()
            .zip(&v0)
            .map(|(x, v)| x + dot(&q0, &k0) * v)
            .collect();
        // out_1 = q_1 e^{-2cs} KV + e^{-cs}(q_1 . k_0) v_0 + (q_1 . k_1) v_1
        let want1: Vec<f32> = apply(&q1, decay * decay, &kv0)
            .iter()
            .enumerate()
            .map(|(b, x)| x + decay * dot(&q1, &k0) * v0[b] + dot(&q1, &k1) * v1[b])
            .collect();
        for (got, want) in out0.iter().zip(&want0) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        for (got, want) in out1.iter().zip(&want1) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        // KV'' = KV e^{-2cs} + e^{-cs} k_0^T v_0 + k_1^T v_1
        for a in 0..d {
            for b in 0..d {
                let want = kv0[a * d + b] * decay * decay + decay * k0[a] * v0[b] + k1[a] * v1[b];
                let got = state[a * d + b];
                assert!(
                    (got - want).abs() < 1e-6,
                    "state[{a}][{b}]: {got} vs {want}"
                );
            }
        }
    }
}
