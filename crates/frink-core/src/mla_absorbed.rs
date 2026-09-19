//! **MLA WITH THE ABSORPTION OPTIMIZATION** -- attention over the
//! COMPRESSED latent, the form llama.cpp runs for every MLA checkpoint
//! whose `attn_kv_b` was split into `attn_k_b` / `attn_v_b` at
//! conversion (`is_mla()`, every DeepSeek-V2/V3/R1 export since
//! 2025-03).
//!
//! # What it is
//!
//! `src/models/deepseek2.cpp:563-598`: instead of expanding the
//! compressed KV `c` (`kv_lora_rank` wide, RMS-normed) into per-head
//! `k_nope` / `v` through `kv_b`, the QUERY is pushed the other way --
//! `q_nope_absorbed[h] = wk_b[h] · q_nope[h]` (`kv_lora_rank` wide) --
//! and attention runs as MQA over ONE shared key per position,
//! `concat(c, k_pe)`, and one shared value, `c`:
//!
//! ```text
//! score[h][t] = (q_abs[h] · c[t] + q_pe[h] · k_pe[t]) * kq_scale
//! out_lat[h]  = sum_t softmax(score[h])[t] * c[t]        // kv_lora_rank wide
//! v[h]        = wv_b[h] · out_lat[h]                     // v_head_dim wide, the caller's
//! ```
//!
//! `kq_scale` stays `1/sqrt(n_embd_head_k_mla)` = `1/sqrt(qk_nope +
//! qk_rope)` (`deepseek2.cpp:312-319`), the width of the UNabsorbed
//! head, not of the latent -- the two forms compute the same logits,
//! and a scale read off the latent width would not.
//!
//! # Why a separate kernel
//!
//! `causal_mla_attention` takes per-head K and V caches
//! (`[seq, n_heads, dim]`); the absorbed form has ONE key row per
//! position shared by every head and its value is a prefix of its key.
//! The naive kernel could be fed a repeated cache, at `n_heads` times
//! the memory the whole optimization exists to save.

/// Attention over a latent cache for one query token.
///
/// `q` is `[n_heads, kv_lora_rank + rope_dim]`, each head's absorbed
/// query followed by its roped `q_pe`. `latent_cache` is `[seq_len,
/// kv_lora_rank + rope_dim]`, each position's normed compressed KV
/// followed by its roped `k_pe` (llama.cpp's `Kcur = concat(kv_cmpr,
/// k_pe)` order, `deepseek2.cpp:582`). `scale` is the caller's
/// `kq_scale`. Returns `[n_heads, kv_lora_rank]`, the weighted sum of
/// the compressed KV, BEFORE `wv_b`.
pub fn causal_mla_absorbed_attention(
    q: &[f32],
    latent_cache: &[f32],
    n_heads: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Vec<f32> {
    let width = kv_lora_rank + rope_dim;
    assert_eq!(q.len(), n_heads * width);
    assert_eq!(latent_cache.len(), seq_len * width);
    let mut out = vec![0f32; n_heads * kv_lora_rank];
    let mut scores = vec![0f32; seq_len];
    for h in 0..n_heads {
        let q_h = &q[h * width..(h + 1) * width];
        for (t, score) in scores.iter_mut().enumerate() {
            let k_t = &latent_cache[t * width..(t + 1) * width];
            let mut dot = 0f32;
            for d in 0..width {
                dot += q_h[d] * k_t[d];
            }
            *score = dot * scale;
        }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        if sum > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }
        let out_h = &mut out[h * kv_lora_rank..(h + 1) * kv_lora_rank];
        for (t, &w) in scores.iter().enumerate() {
            let c_t = &latent_cache[t * width..t * width + kv_lora_rank];
            for d in 0..kv_lora_rank {
                out_h[d] += w * c_t[d];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::causal_mla_attention;

    /// The absorbed form and the naive form are the same attention.
    /// Build a random `kv_b` = `[n_heads * (nope + v), kv_lora]`, run
    /// the naive kernel over the EXPANDED per-head K/V, and the
    /// absorbed kernel over the latent with the query pushed through
    /// `k_b` and the output pulled through `v_b`; the two agree to
    /// float noise, with the naive head width as the scale for both.
    #[test]
    fn absorbed_equals_naive_for_the_same_kv_b() {
        let (n_heads, nope, rope, v_dim, kv_lora, seq) =
            (3usize, 4usize, 2usize, 5usize, 6usize, 7usize);
        let mut s = 0x9E37u32;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) - 0.5
        };
        // kv_b[h]: rows nope+v, cols kv_lora.
        let kv_b: Vec<Vec<f32>> = (0..n_heads)
            .map(|_| (0..(nope + v_dim) * kv_lora).map(|_| rnd()).collect())
            .collect();
        let latent: Vec<f32> = (0..seq * (kv_lora + rope)).map(|_| rnd()).collect();
        // The query in the naive layout: [n_heads, nope + rope].
        let q_naive: Vec<f32> = (0..n_heads * (nope + rope)).map(|_| rnd()).collect();

        // Naive: expand every position through kv_b.
        let mut k_cache = vec![0f32; seq * n_heads * (nope + rope)];
        let mut v_cache = vec![0f32; seq * n_heads * v_dim];
        for t in 0..seq {
            let c = &latent[t * (kv_lora + rope)..t * (kv_lora + rope) + kv_lora];
            let k_pe = &latent[t * (kv_lora + rope) + kv_lora..(t + 1) * (kv_lora + rope)];
            for h in 0..n_heads {
                let w = &kv_b[h];
                let k_h = &mut k_cache
                    [(t * n_heads + h) * (nope + rope)..(t * n_heads + h + 1) * (nope + rope)];
                for i in 0..nope {
                    k_h[i] = (0..kv_lora).map(|j| w[i * kv_lora + j] * c[j]).sum();
                }
                k_h[nope..].copy_from_slice(k_pe);
                let v_h = &mut v_cache[(t * n_heads + h) * v_dim..(t * n_heads + h + 1) * v_dim];
                for i in 0..v_dim {
                    v_h[i] = (0..kv_lora)
                        .map(|j| w[(nope + i) * kv_lora + j] * c[j])
                        .sum();
                }
            }
        }
        let naive = causal_mla_attention(
            &q_naive,
            &k_cache,
            &v_cache,
            n_heads,
            nope + rope,
            v_dim,
            seq,
        );

        // Absorbed: push q_nope through k_b (the nope rows of kv_b,
        // transposed), attend over the latent, pull through v_b.
        let width = kv_lora + rope;
        let mut q_abs = vec![0f32; n_heads * width];
        for h in 0..n_heads {
            let w = &kv_b[h];
            let q_h = &q_naive[h * (nope + rope)..(h + 1) * (nope + rope)];
            for j in 0..kv_lora {
                q_abs[h * width + j] = (0..nope).map(|i| w[i * kv_lora + j] * q_h[i]).sum();
            }
            q_abs[h * width + kv_lora..(h + 1) * width].copy_from_slice(&q_h[nope..]);
        }
        let scale = 1.0 / ((nope + rope) as f32).sqrt();
        let out_lat =
            causal_mla_absorbed_attention(&q_abs, &latent, n_heads, kv_lora, rope, seq, scale);
        let mut absorbed = vec![0f32; n_heads * v_dim];
        for h in 0..n_heads {
            let w = &kv_b[h];
            let lat = &out_lat[h * kv_lora..(h + 1) * kv_lora];
            for i in 0..v_dim {
                absorbed[h * v_dim + i] = (0..kv_lora)
                    .map(|j| w[(nope + i) * kv_lora + j] * lat[j])
                    .sum();
            }
        }
        for (a, b) in naive.iter().zip(absorbed.iter()) {
            assert!((a - b).abs() < 1e-5, "{naive:?} vs {absorbed:?}");
        }
        // The scale matters: the latent width would be the wrong one.
        let wrong = causal_mla_absorbed_attention(
            &q_abs,
            &latent,
            n_heads,
            kv_lora,
            rope,
            seq,
            1.0 / (width as f32).sqrt(),
        );
        assert!(wrong
            .iter()
            .zip(out_lat.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4));
    }
}
