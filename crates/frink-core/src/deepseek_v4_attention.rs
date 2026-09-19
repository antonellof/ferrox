//! DeepSeek V4's CSA (Compressed Sparse Attention) / HCA (Heavily
//! Compressed Attention) attention assembly: combining a small raw local
//! window with a set of already-compressed entries (see
//! [`crate::csa_hca_compress`] for how those entries are produced).
//!
//! Transcribed from the real, merged reference implementation
//! (llama.cpp PR #24162, `src/models/deepseek4.cpp`,
//! `build_csa_lid_attention`/`build_hca_attention`): both real functions
//! fetch a raw K/V window (`mctx_raw->get_k`, a real sliding-window
//! cache, never filtered further) and a compressed K/V set (`get_csa()`/
//! `get_hca()`), concatenate them along the sequence axis
//! (`k_all = ggml_concat(ctx0, raw_k, csa_k or hca_k, 2)`), build one
//! combined attention mask, and run one ordinary masked-attention call
//! (`build_attn_mha`) over the concatenation -- **not** two separate
//! attention passes blended afterward. The two mechanisms differ only in
//! which compressed entries are visible: HCA's mask (`inp_hca.kq_mask`)
//! is used as-is (every compressed entry visible, i.e. dense), while
//! CSA's mask is built by [`build_top_k_mask`] restricting it to the
//! indexer's top-`indexer_top_k` selection first
//! (`build_top_k_mask(inp_csa.kq_mask, top_k, ...)`, where `top_k` itself
//! comes from `build_lid_top_k`, DeepSeek V4's own dedicated lightning
//! indexer).
//!
//! Both of frink-core's existing MLA attention primitives turn out to
//! be *directly* reusable for this, with no CSA/HCA-specific attention
//! math needed: since the real mechanism is exactly "concatenate raw and
//! compressed K/V into one sequence, then run one masked attention pass
//! over the concatenation," building that one combined K/V cache and
//! delegating to [`crate::attention::causal_mla_attention_sinks`] (HCA: dense,
//! every raw and compressed position visible) or
//! [`crate::attention::causal_mla_attention_sparse_sinks`] (CSA: raw positions
//! always visible, compressed positions restricted to the indexer's
//! top-k) reproduces the real op sequence exactly. The lightning-indexer
//! selection itself is likewise DeepSeek-V3.2/GLM-5.2's existing
//! [`crate::attention::lightning_indexer_topk`] unchanged -- the real
//! source's own comment on this indexer family ("it is a variant of DSA
//! ... operates on the same principle of the lightning indexer") holds
//! exactly at the primitive level; the only real difference is that its
//! `indexer_keys` here are themselves *compressed* representations (see
//! [`crate::csa_hca_compress`]) rather than raw per-token keys, which is
//! a difference in what the caller passes in, not in the indexer math.
//!
//! Both functions here take the raw window and compressed set as
//! separate, already-computed K/V buffers -- assembling those buffers
//! (raw SWA window eviction, incremental compressed-entry bookkeeping)
//! is real, cited scope this module does not attempt; see
//! `crate::csa_hca_compress`'s module doc comment for the same caveat.

use crate::attention::{
    causal_mla_attention_sinks, causal_mla_attention_sparse_sinks, lightning_indexer_topk,
};

/// Concatenates a raw window and a compressed set of K or V buffers
/// (each `[n_positions, n_heads, head_dim]` flattened row-major) into one
/// combined `[n_raw + n_compressed, n_heads, head_dim]` buffer, matching
/// the real `ggml_concat(ctx0, raw_k, csa_k_or_hca_k, 2)` (concatenation
/// along the sequence axis).
fn concat_raw_and_compressed(raw: &[f32], compressed: &[f32]) -> Vec<f32> {
    let mut combined = Vec::with_capacity(raw.len() + compressed.len());
    combined.extend_from_slice(raw);
    combined.extend_from_slice(compressed);
    combined
}

/// DeepSeek V4's HCA attention (`build_hca_attention`): dense attention
/// over the concatenation of a raw local window and *every* compressed
/// (128-token-block) entry -- no top-k filtering of the compressed set,
/// unlike CSA. `raw_k`/`raw_v` are `[n_raw, n_heads, *_head_dim]`;
/// `compressed_k`/`compressed_v` are `[n_compressed, n_heads,
/// *_head_dim]` (each already produced by
/// `crate::csa_hca_compress::compress_block`, one entry per compressed
/// block). Returns `[n_heads, v_head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn hca_attention(
    q: &[f32],
    raw_k: &[f32],
    raw_v: &[f32],
    n_raw: usize,
    compressed_k: &[f32],
    compressed_v: &[f32],
    n_compressed: usize,
    n_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    sinks: Option<&[f32]>,
) -> Vec<f32> {
    let k_all = concat_raw_and_compressed(raw_k, compressed_k);
    let v_all = concat_raw_and_compressed(raw_v, compressed_v);
    causal_mla_attention_sinks(
        q,
        &k_all,
        &v_all,
        n_heads,
        qk_head_dim,
        v_head_dim,
        n_raw + n_compressed,
        sinks,
    )
}

/// DeepSeek V4's CSA attention (`build_csa_lid_attention`): attention
/// over the concatenation of a raw local window (always fully visible)
/// and only the top-`top_k` compressed (every-4-token-block) entries as
/// scored by DeepSeek V4's dedicated lightning indexer
/// (`build_lid_top_k`, the same [`lightning_indexer_topk`] primitive
/// GLM-5.2/DeepSeek-V3.2's DSA uses, scored here against compressed
/// indexer keys instead of raw ones -- see the module doc comment).
///
/// `indexer_q`/`indexer_weights` score this query against
/// `indexer_keys` (one entry per compressed position, i.e. length
/// `n_compressed`, itself compressed via the indexer's own dedicated
/// `indexer_comp_*` projections per the real source -- a separate
/// compressor from CSA's own `attn_comp_*`, not the same tensors).
#[allow(clippy::too_many_arguments)]
pub fn csa_attention(
    q: &[f32],
    raw_k: &[f32],
    raw_v: &[f32],
    n_raw: usize,
    compressed_k: &[f32],
    compressed_v: &[f32],
    n_compressed: usize,
    indexer_q: &[Vec<f32>],
    indexer_keys: &[Vec<f32>],
    indexer_weights: &[f32],
    top_k: usize,
    n_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    sinks: Option<&[f32]>,
) -> Vec<f32> {
    assert_eq!(indexer_keys.len(), n_compressed);

    let selected = lightning_indexer_topk(indexer_q, indexer_keys, indexer_weights, top_k);

    let k_all = concat_raw_and_compressed(raw_k, compressed_k);
    let v_all = concat_raw_and_compressed(raw_v, compressed_v);

    let mut visible: Vec<usize> = (0..n_raw).collect();
    visible.extend(selected.iter().map(|&i| n_raw + i));

    causal_mla_attention_sparse_sinks(
        q,
        &k_all,
        &v_all,
        n_heads,
        qk_head_dim,
        v_head_dim,
        n_raw + n_compressed,
        &visible,
        sinks,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::causal_mla_attention;

    #[test]
    fn hca_attention_with_single_raw_and_single_compressed_position_matches_full_causal_over_both()
    {
        let n_heads = 1;
        let qk_head_dim = 2;
        let v_head_dim = 2;
        let q = vec![1.0, 0.0];
        let raw_k = vec![1.0, 0.0];
        let raw_v = vec![9.0, -3.0];
        let compressed_k = vec![0.5, 0.5];
        let compressed_v = vec![100.0, 200.0];

        let out = hca_attention(
            &q,
            &raw_k,
            &raw_v,
            1,
            &compressed_k,
            &compressed_v,
            1,
            n_heads,
            qk_head_dim,
            v_head_dim,
            None,
        );

        // Cross-check against causal_mla_attention on the manually
        // concatenated buffer -- HCA attention must be nothing more than
        // dense attention over the raw+compressed concatenation.
        let k_all = vec![1.0, 0.0, 0.5, 0.5];
        let v_all = vec![9.0, -3.0, 100.0, 200.0];
        let expected =
            causal_mla_attention(&q, &k_all, &v_all, n_heads, qk_head_dim, v_head_dim, 2);
        assert_eq!(out.len(), expected.len());
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "a={a} b={b}");
        }
    }

    #[test]
    fn hca_attention_gives_nonzero_weight_to_compressed_positions() {
        // A raw window of one position and a wildly-different-valued
        // compressed position with an equally strong key match: since
        // HCA is dense (no top-k filtering), the compressed position's V
        // must visibly pull the output away from the raw-only answer.
        let n_heads = 1;
        let qk_head_dim = 2;
        let v_head_dim = 1;
        let q = vec![1.0, 0.0];
        let raw_k = vec![1.0, 0.0];
        let raw_v = vec![5.0];
        let compressed_k = vec![1.0, 0.0]; // identical key match to raw
        let compressed_v = vec![999.0];

        let out = hca_attention(
            &q,
            &raw_k,
            &raw_v,
            1,
            &compressed_k,
            &compressed_v,
            1,
            n_heads,
            qk_head_dim,
            v_head_dim,
            None,
        );
        // Identical keys -> softmax splits 50/50 -> average of 5 and 999.
        assert!((out[0] - 502.0).abs() < 1e-3, "out[0]={}", out[0]);
    }

    #[test]
    fn csa_attention_with_top_k_covering_every_compressed_entry_matches_hca_attention() {
        // Selecting every compressed entry via top_k must reduce CSA
        // attention to exactly the same dense computation as HCA
        // attention -- the two mechanisms share identical attention math,
        // differing only in whether the compressed set is filtered.
        let n_heads = 1;
        let qk_head_dim = 2;
        let v_head_dim = 1;
        let q = vec![0.3, 0.7];
        let raw_k = vec![0.2, 0.4, 0.1, 0.9];
        let raw_v = vec![1.0, 2.0];
        let n_raw = 2;
        let compressed_k = vec![0.5, 0.1, 0.05, 0.6, 0.3, 0.3];
        let compressed_v = vec![10.0, 20.0, 30.0];
        let n_compressed = 3;

        let indexer_q = vec![vec![1.0, 0.0]];
        let indexer_keys = vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]];
        let indexer_weights = vec![1.0];

        let csa_out = csa_attention(
            &q,
            &raw_k,
            &raw_v,
            n_raw,
            &compressed_k,
            &compressed_v,
            n_compressed,
            &indexer_q,
            &indexer_keys,
            &indexer_weights,
            n_compressed, // top_k covers everything
            n_heads,
            qk_head_dim,
            v_head_dim,
            None,
        );
        let hca_out = hca_attention(
            &q,
            &raw_k,
            &raw_v,
            n_raw,
            &compressed_k,
            &compressed_v,
            n_compressed,
            n_heads,
            qk_head_dim,
            v_head_dim,
            None,
        );
        assert_eq!(csa_out.len(), hca_out.len());
        for (a, b) in csa_out.iter().zip(hca_out.iter()) {
            assert!((a - b).abs() < 1e-6, "a={a} b={b}");
        }
    }

    #[test]
    fn csa_attention_ignores_compressed_entries_the_indexer_does_not_select() {
        // Two compressed entries with wildly different V; the indexer
        // strongly favors entry 0 over entry 1 (orthogonal key, relu'd
        // to zero score), and top_k=1 keeps only entry 0. Entry 1's
        // extreme V must have zero influence on the output.
        let n_heads = 1;
        let qk_head_dim = 2;
        let v_head_dim = 1;
        let q = vec![1.0, 0.0];
        let raw_k = vec![1.0, 0.0];
        let raw_v = vec![5.0];
        let compressed_k = vec![1.0, 0.0, 1.0, 0.0]; // both compressed keys identical to q
        let compressed_v = vec![7.0, 99999.0];

        let indexer_q = vec![vec![1.0, 0.0]];
        // Entry 0 aligned with indexer query (score>0); entry 1 orthogonal
        // (dot=0, relu'd score 0) -- must lose at top_k=1.
        let indexer_keys = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let indexer_weights = vec![1.0];

        let out = csa_attention(
            &q,
            &raw_k,
            &raw_v,
            1,
            &compressed_k,
            &compressed_v,
            2,
            &indexer_q,
            &indexer_keys,
            &indexer_weights,
            1,
            n_heads,
            qk_head_dim,
            v_head_dim,
            None,
        );
        // Raw and the one selected compressed entry have identical keys
        // (both [1,0]), so softmax splits 50/50 between V=5 and V=7.
        assert!((out[0] - 6.0).abs() < 1e-3, "out[0]={}", out[0]);
    }
}
