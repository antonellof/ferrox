//! DeepSeek V4's CSA (Compressed Sparse Attention) / HCA (Heavily
//! Compressed Attention) shared block-compression primitive: pooling a
//! block of raw per-token key/value-role vectors down to one compressed
//! entry, before RMSNorm and RoPE-on-the-rope-slice.
//!
//! Transcribed directly from the real, merged reference implementation
//! (llama.cpp PR #24162, `src/models/deepseek4.cpp`):
//! `build_hca_compressed_kv_from_state` (HCA, non-overlapping 128-token
//! blocks) and `build_overlap_compressed_kv_from_state` (CSA,
//! overlapping every-4-token blocks -- also reused verbatim, per the real
//! source, for DeepSeek V4's dedicated lightning-indexer compressor,
//! which only ever runs in `ratio == 4` mode). Both real functions share
//! exactly the same pooling+norm+RoPE math once a caller has assembled
//! the block of raw per-token vectors that feed one compressed output;
//! only how that block of raw positions is constructed differs between
//! CSA and HCA (see "What's real but NOT implemented here" below).
//!
//! ## The real pooling math (read line-by-line from both functions)
//!
//! Given a block of `n` raw per-token vectors, each `n_embd_head` wide
//! (`kv_block[p][c]` for raw position `p` in the block, channel `c`) and
//! a matching `score_block[p][c]` (the real code's `score_state`, itself
//! `attn_comp_wgate`'s projection plus a learned per-position additive
//! positional embedding -- `attn_comp_ape`/`indexer_comp_ape`, added by
//! the caller before this function; see [`channel_gated_pool`]'s doc
//! comment), the per-channel weighted pool is:
//!
//! ```text
//! comp[c] = sum_p( kv_block[p][c] * softmax_over_p(score_block[.][c])[p] )
//! ```
//!
//! This is a real, unusual detail worth calling out explicitly: it is
//! **not** a single shared attention distribution pooling every channel
//! together (the way standard attention pooling works) -- each embedding
//! channel gets its *own* softmax gate over the block's raw positions,
//! computed from that channel's own score values. This falls directly
//! out of ggml's real op sequence in both functions (`ggml_permute` swaps
//! the channel/position axes so `ggml_soft_max`'s implicit "softmax over
//! `ne[0]`" operates over positions independently per channel, then
//! `ggml_mul` + `ggml_sum_rows` folds positions away channel-by-channel)
//! -- not an approximation or a design choice made here.
//!
//! After pooling, both real functions RMSNorm the result (`attn_comp_norm`
//! / `indexer_comp_norm`, a learned per-channel scale, no bias) and apply
//! RoPE to *only* the trailing `n_embd_head_rope`-wide slice (the leading
//! "nope" channels are left untouched), at the compressed block's own
//! position (not the raw per-token position) and using a dedicated
//! frequency base (`hparams.dsv4_compress_rope_base`, distinct from the
//! main attention's `freq_base`). DeepSeek V4 (`LLM_ARCH_DEEPSEEK4`) is
//! added to the exact same `llama_model_rope_type` switch-case group as
//! `LLM_ARCH_DEEPSEEK2`/`DEEPSEEK32`/`CHATGLM`/`GRANITE` (all
//! `LLAMA_ROPE_TYPE_NEOX`), so this module uses
//! [`crate::attention::apply_rope`] (split-half NEOX), not
//! [`crate::attention::apply_rope_interleaved`].
//!
//! ## What's real but NOT implemented here
//!
//! The real reference is a stateful, incremental KV-cache mechanism
//! (`llama-kv-cache-dsv4.cpp`, ~3400 lines, not read in full for this
//! pass) that carries partially-filled blocks across forward calls via
//! `state_read_idxs`/`state_write_idxs`/`state_persist_*` index tensors,
//! and pads out-of-range reads near the start of the sequence with a real
//! zero-KV/`-inf`-score phantom row (`dsv4_append_zero_row`) rather than
//! requiring the caller to assemble exactly-sized blocks -- callers of
//! this module must apply the same convention by hand (e.g. an `-inf`
//! score entry contributes exactly zero weight after softmax, same
//! effect as the real padding row) when a compression window runs off
//! the start of the sequence.
//!
//! It also confirms a real, subtle detail this module's API surfaces but
//! does not automate: CSA's (and the indexer's) raw per-token projection
//! is **twice as wide** as HCA's (`coff = ratio == 4 ? 2 : 1` in
//! `load_arch_tensors`) because each raw token is projected once for its
//! role as the *tail* of the block ending at it, and once for its role
//! as the *head* of the next (overlapping) block -- two different
//! learned projections of the same token, not one projection reused
//! twice (`build_overlap_compressed_kv_from_state`'s real
//! `GGML_ASSERT(kv_state->ne[0] == 2*n_embd_head)` plus its `kv_prev`/
//! `kv_cur` split at `ggml_row_size(.., n_embd_head)` offset). Assembling
//! the exact overlapping-window index sequences (and the incremental
//! persist/pad state machine) is real, cited scope left for whoever
//! wires this into a real decoder; this module is the pooling primitive
//! both CSA and HCA (and the indexer) share once that block is
//! assembled.

use crate::attention::apply_rope;
use crate::matmul::rms_norm;

/// The real per-channel softmax-gated pool shared by
/// `build_hca_compressed_kv_from_state` and
/// `build_overlap_compressed_kv_from_state`:
/// `comp[c] = sum_p( kv_block[p][c] * softmax_over_p(score_block[.][c])[p] )`.
///
/// Both slices must be `[block_len][n_embd_head]` (one row per raw
/// position), already restricted to exactly the raw positions that feed
/// this one compressed output (HCA: `ratio` consecutive raw positions;
/// CSA/indexer: the `2*ratio`-wide concatenation of the previous and
/// current half-blocks' role-specific projections -- see the module doc
/// comment's "twice as wide" note). Returns a vector of length
/// `n_embd_head`.
pub fn channel_gated_pool(kv_block: &[Vec<f32>], score_block: &[Vec<f32>]) -> Vec<f32> {
    assert_eq!(
        kv_block.len(),
        score_block.len(),
        "kv_block and score_block must have the same number of raw positions"
    );
    assert!(
        !kv_block.is_empty(),
        "a compression block must have at least one raw position"
    );
    let n_embd_head = kv_block[0].len();
    for (kv, score) in kv_block.iter().zip(score_block.iter()) {
        assert_eq!(kv.len(), n_embd_head);
        assert_eq!(score.len(), n_embd_head);
    }

    let mut out = vec![0f32; n_embd_head];
    for c in 0..n_embd_head {
        let max = score_block
            .iter()
            .map(|s| s[c])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut weights: Vec<f32> = score_block.iter().map(|s| (s[c] - max).exp()).collect();
        let sum: f32 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= sum;
        }
        out[c] = kv_block
            .iter()
            .zip(weights.iter())
            .map(|(kv, &w)| kv[c] * w)
            .sum();
    }
    out
}

/// Wraps [`channel_gated_pool`] with the real post-processing every
/// compressed entry gets before being written into the compressed K
/// cache: RMSNorm (learned `norm_weight`, the real `attn_comp_norm` /
/// `indexer_comp_norm` tensor), then RoPE applied *only* to the trailing
/// `n_embd_head_rope`-wide slice (the leading `n_embd_head -
/// n_embd_head_rope` "nope" channels are left untouched) at
/// `block_position` using `compress_rope_theta` (the real
/// `dsv4_compress_rope_base`, a dedicated frequency base distinct from
/// the model's main RoPE base) -- see the module doc comment for the
/// exact real function names and line-level detail this was transcribed
/// from.
pub fn compress_block(
    kv_block: &[Vec<f32>],
    score_block: &[Vec<f32>],
    norm_weight: &[f32],
    rms_eps: f32,
    n_embd_head_rope: usize,
    block_position: usize,
    compress_rope_theta: f32,
) -> Vec<f32> {
    let pooled = channel_gated_pool(kv_block, score_block);
    let n_embd_head = pooled.len();
    assert!(
        n_embd_head_rope <= n_embd_head,
        "rope slice cannot exceed the compressed vector's width"
    );
    assert_eq!(norm_weight.len(), n_embd_head);

    let mut normed = rms_norm(&pooled, norm_weight, rms_eps);

    let rope_start = n_embd_head - n_embd_head_rope;
    apply_rope(
        &mut normed[rope_start..],
        block_position,
        compress_rope_theta,
    );
    normed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_gated_pool_with_one_hot_scores_selects_that_positions_value() {
        // Position 1 has an overwhelmingly large score on every channel,
        // so softmax collapses to (approximately) one-hot on it -- the
        // pooled output must match position 1's raw kv vector, regardless
        // of what the other positions' kv values are.
        let kv_block = vec![vec![1.0, 2.0], vec![9.0, -3.0], vec![100.0, 200.0]];
        let score_block = vec![vec![0.0, 0.0], vec![50.0, 50.0], vec![0.0, 0.0]];
        let out = channel_gated_pool(&kv_block, &score_block);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 9.0).abs() < 1e-3, "out[0]={}", out[0]);
        assert!((out[1] - (-3.0)).abs() < 1e-3, "out[1]={}", out[1]);
    }

    #[test]
    fn channel_gated_pool_gates_each_channel_independently() {
        // Channel 0's scores favor position 0; channel 1's scores favor
        // position 1 -- a single shared (non-per-channel) softmax could
        // never produce this, since it would have to pick one winning
        // position for both channels at once.
        let kv_block = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
        let score_block = vec![vec![50.0, -50.0], vec![-50.0, 50.0]];
        let out = channel_gated_pool(&kv_block, &score_block);
        assert!((out[0] - 10.0).abs() < 1e-3, "out[0]={}", out[0]);
        assert!((out[1] - 40.0).abs() < 1e-3, "out[1]={}", out[1]);
    }

    #[test]
    fn channel_gated_pool_with_uniform_scores_is_a_plain_average() {
        let kv_block = vec![vec![2.0, 4.0], vec![6.0, 8.0], vec![10.0, 12.0]];
        let score_block = vec![vec![0.0, 0.0]; 3];
        let out = channel_gated_pool(&kv_block, &score_block);
        assert!((out[0] - 6.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 8.0).abs() < 1e-5, "out[1]={}", out[1]);
    }

    #[test]
    fn channel_gated_pool_single_position_block_is_identity() {
        let kv_block = vec![vec![1.5, -2.5, 3.5]];
        let score_block = vec![vec![7.0, -3.0, 0.0]];
        let out = channel_gated_pool(&kv_block, &score_block);
        assert_eq!(out, vec![1.5, -2.5, 3.5]);
    }

    #[test]
    fn compress_block_leaves_nope_slice_untouched_by_rope() {
        // With block_position=0, apply_rope is documented as an exact
        // identity (see attention.rs's rope_at_position_zero_is_identity
        // test), so the whole vector should equal the plain RMSNorm of
        // the pooled block when position is 0 -- this specifically pins
        // that the nope/rope split offset is being sliced correctly
        // (a swapped or off-by-one split would still pass at position 0
        // only by accident, so the follow-up test checks a nonzero
        // position too).
        let kv_block = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let score_block = vec![vec![0.0, 0.0, 0.0, 0.0]];
        let norm_weight = vec![1.0, 1.0, 1.0, 1.0];
        let out = compress_block(&kv_block, &score_block, &norm_weight, 1e-5, 2, 0, 10000.0);
        let plain = rms_norm(&[1.0, 2.0, 3.0, 4.0], &norm_weight, 1e-5);
        for (a, b) in out.iter().zip(plain.iter()) {
            assert!((a - b).abs() < 1e-5, "a={a} b={b}");
        }
    }

    #[test]
    fn compress_block_rotates_only_the_rope_slice_at_nonzero_position() {
        let kv_block = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let score_block = vec![vec![0.0, 0.0, 0.0, 0.0]];
        let norm_weight = vec![1.0, 1.0, 1.0, 1.0];
        let out = compress_block(&kv_block, &score_block, &norm_weight, 1e-5, 2, 5, 10000.0);
        let plain = rms_norm(&[1.0, 2.0, 3.0, 4.0], &norm_weight, 1e-5);
        // nope slice (first 2 channels) must be untouched by RoPE.
        assert!((out[0] - plain[0]).abs() < 1e-5);
        assert!((out[1] - plain[1]).abs() < 1e-5);
        // rope slice (last 2 channels) must have been rotated, so it
        // must differ from the un-roped RMSNorm output.
        let differs = (out[2] - plain[2]).abs() > 1e-4 || (out[3] - plain[3]).abs() > 1e-4;
        assert!(differs, "rope slice must be rotated at a nonzero position");
    }

    #[test]
    fn compress_block_rope_preserves_rope_slice_norm() {
        // RoPE is a rotation, so it must preserve the rope slice's norm
        // -- a real property check, not just a "did it change" check.
        let kv_block = vec![vec![0.3, -0.7, 1.1, -1.3, 0.2, 0.5]];
        let score_block = vec![vec![0.0; 6]];
        let norm_weight = vec![1.0; 6];
        let out = compress_block(&kv_block, &score_block, &norm_weight, 1e-5, 4, 3, 10000.0);
        let plain = rms_norm(&[0.3, -0.7, 1.1, -1.3, 0.2, 0.5], &norm_weight, 1e-5);
        let rope_norm_before: f32 = plain[2..].iter().map(|v| v * v).sum::<f32>().sqrt();
        let rope_norm_after: f32 = out[2..].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((rope_norm_before - rope_norm_after).abs() < 1e-4);
    }
}
