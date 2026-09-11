//! The attention half of one decoder layer, written once.
//!
//! Everything from the QKV projection to `post_attn_norm` used to be
//! spelled out longhand in `forward_token`'s CPU arm and again in
//! `forward_token_paged`, with a third copy of just the push-and-attend
//! step in `forward_multi_seq_kv`. That is how five model features went
//! missing from the paged path one at a time, and how a sixth
//! (`attention_scale`) reached only two of the four host bodies: a copy
//! diverges from its original and nothing notices.
//!
//! So the decorations live here, in one body, and the ONE thing that
//! genuinely differs between the callers -- where this row's K and V are
//! written and read -- is a parameter, [`KvStep`]. That is the same
//! answer [`super::MultiSeqKv`] already reached for the batched path and
//! the same one llama.cpp reached by overloading `build_attn` on its
//! memory-input type.

use ferrox_core::attention::{causal_gqa_attention_softcap, causal_gqa_attention_windowed_softcap};
use ferrox_core::cache::{KvCache, PagedKvCache, SharedPagedKv};
use ferrox_core::matmul::rms_norm;

use super::{Decoder, GptOssLayer, LayerWeights};

/// Where one row's K/V is written, and what that implies for the kernel
/// that reads it back.
///
/// A named variant per backing rather than a `paged: bool`, for the
/// reason `MultiSeqKv`'s doc comment gives: a tenth caller can silently
/// forget a flag, and cannot silently forget to name a variant.
pub(crate) enum KvStep<'a> {
    /// Single-sequence contiguous decode (`forward_token`).
    ///
    /// The only variant allowed to reach the CUDA resident per-layer KV
    /// in [`Decoder::gqa_attention`]: that buffer holds ONE sequence's
    /// history, seeded by `forward_token` at `pos == 0`.
    Decode(&'a mut KvCache),
    /// One sequence of a multi-sequence batch, contiguous
    /// (`forward_multi_seq`).
    ///
    /// Identical math to [`KvStep::Decode`] minus the CUDA resident
    /// hook. Taking that hook here would answer sequence `b` out of
    /// sequence 0's history, silently -- the resident buffer is never
    /// populated by the batched path.
    Batched(&'a mut KvCache),
    /// Block-table-indexed KV, shared across sequences
    /// (`forward_token_paged`, `forward_multi_seq_kv`'s paged arm).
    Paged {
        cache: &'a mut PagedKvCache,
        stores: &'a SharedPagedKv,
    },
}

impl Decoder {
    /// One layer's attention block for ONE row: QKV projection, the
    /// three QKV biases, the two QK norms, RoPE's `mscale`, per-head
    /// RoPE, `attention_scale`, the KV push and attend, `o_proj`,
    /// gpt-oss's `o_bias`, and `post_attn_norm`.
    ///
    /// Takes `normed` rather than computing it: `forward_token`'s Metal
    /// arm needs the normed vector before it knows whether the block
    /// will run on the host at all.
    ///
    /// Returns the attention branch's contribution to the residual --
    /// the caller adds it.
    pub(crate) fn attn_block(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        normed: &[f32],
        pos: usize,
        kv: KvStep<'_>,
    ) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;

        let (mut q, mut k, mut v) = {
            #[cfg(any(feature = "cuda", feature = "metal"))]
            {
                if let Some(mut outs) = ferrox_core::WeightMatrix::apply_gpu_multi(
                    &[&layer.attn.q_proj, &layer.attn.k_proj, &layer.attn.v_proj],
                    normed,
                ) {
                    let v = outs.pop().unwrap();
                    let k = outs.pop().unwrap();
                    let q = outs.pop().unwrap();
                    (q, k, v)
                } else {
                    ferrox_core::weight_matrix::WeightMatrix::apply_three(
                        &layer.attn.q_proj,
                        &layer.attn.k_proj,
                        &layer.attn.v_proj,
                        normed,
                    )
                }
            }
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            {
                ferrox_core::weight_matrix::WeightMatrix::apply_three(
                    &layer.attn.q_proj,
                    &layer.attn.k_proj,
                    &layer.attn.v_proj,
                    normed,
                )
            }
        };

        // Whole rows here: one token's Q and K. See
        // `Decoder::qk_norm_after_rope` for why the norm has two homes.
        let (q_width, kv_width) = (q.len(), k.len());
        self.apply_qkv_bias_and_clamp(layer, &mut q, &mut k, &mut v, q_width, kv_width);
        self.apply_qk_norms_pre_rope(layer, &mut q, &mut k, q_width, kv_width);
        self.apply_rope_attn_factor(&mut q, &mut k, layer_idx);

        for h in 0..n_heads {
            self.apply_rope_head_layer(&mut q[h * head_dim..(h + 1) * head_dim], pos, layer_idx);
        }
        for h in 0..n_kv_heads {
            self.apply_rope_head_layer(&mut k[h * head_dim..(h + 1) * head_dim], pos, layer_idx);
        }
        self.apply_qk_norms_post_rope(layer, &mut q, &mut k, q_width, kv_width);
        self.apply_attention_scale(&mut q);

        let oai = self.gpt_oss.as_ref().map(|g| &g.layers[layer_idx]);
        let attn_out = self.push_and_attend_row(kv, layer_idx, &k, &v, &q, oai);

        let mut projected = layer.attn.o_proj.apply(&attn_out);
        if let Some(oai) = oai {
            for (x, b) in projected.iter_mut().zip(oai.o_bias.iter()) {
                *x += b;
            }
        }
        if let Some(post) = &layer.attn.post_attn_norm {
            projected = rms_norm(&projected, post, self.config.rms_norm_eps);
        }
        projected
    }

    /// Appends one row's K/V to whichever backing `kv` names, then
    /// attends over everything that sequence holds.
    ///
    /// The ONLY place the backing shows through, which is the whole
    /// point: paging changes where rows live and nothing else, so an arm
    /// one backing reproduced and the other did not would be a model
    /// that answers differently depending on whether a KV pool happened
    /// to be configured. `causal_gqa_attention_paged_sinks` covers all
    /// three contiguous arms in one entry point and is bit-identical to
    /// each by construction.
    pub(crate) fn push_and_attend_row(
        &self,
        kv: KvStep<'_>,
        layer_idx: usize,
        k: &[f32],
        v: &[f32],
        q: &[f32],
        oai: Option<&GptOssLayer>,
    ) -> Vec<f32> {
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let window = self.config.layer_sliding_window(layer_idx);
        // Derived from the variant rather than passed as a flag; see
        // `KvStep::Batched`.
        let cuda_resident_layer = match &kv {
            KvStep::Decode(_) => Some(layer_idx),
            KvStep::Batched(_) | KvStep::Paged { .. } => None,
        };
        match kv {
            KvStep::Decode(cache) | KvStep::Batched(cache) => {
                cache
                    .push(k, v)
                    .expect("unbounded/planned KvCache growth is infallible");
                let out = if let Some(oai) = oai {
                    ferrox_core::causal_gqa_attention_sinks(
                        q,
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        cache.rows(),
                        window,
                        &oai.attn_sinks,
                    )
                } else {
                    match (window, cuda_resident_layer) {
                        (Some(window), _) => causal_gqa_attention_windowed_softcap(
                            q,
                            &cache.k,
                            &cache.v,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            cache.rows(),
                            window,
                            self.config.attn_logit_softcap,
                        ),
                        (None, Some(l)) => self.gqa_attention(
                            l,
                            q,
                            &cache.k,
                            &cache.v,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            cache.rows(),
                        ),
                        (None, None) => causal_gqa_attention_softcap(
                            q,
                            &cache.k,
                            &cache.v,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            cache.rows(),
                            self.config.attn_logit_softcap,
                        ),
                    }
                };
                // AFTER the read, never inside `push`: the rows this
                // drops are rows every kernel above has finished with
                // (#61). A no-op unless `FERROX_KV_WINDOW` is on and
                // this layer is windowed -- and note that it is the same
                // `window` the kernels just used, taken from the same
                // `ModelConfig`, because keeping fewer rows than the
                // kernel reads would answer out of a truncated history.
                self.evict_layer_kv(layer_idx, cache);
                out
            }
            KvStep::Paged { cache, stores } => {
                // Write guard for the push alone, then a read guard for
                // the attention: the rule `SharedPagedKv` documents.
                // Holding the write guard across attention would
                // serialise the expensive half and give back a global
                // lock.
                {
                    let mut store = stores.write(layer_idx);
                    cache
                        .push(&mut store, k, v)
                        .expect("every caller reserves this row's pages before the stack runs");
                }
                let store = stores.read(layer_idx);
                ferrox_core::causal_gqa_attention_paged_sinks(
                    q,
                    &store,
                    cache.block_table(),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    cache.seq_len(),
                    window,
                    oai.map(|o| o.attn_sinks.as_slice()),
                    // The sink arm carries no softcap, matching the
                    // contiguous dispatch above.
                    if oai.is_some() {
                        None
                    } else {
                        self.config.attn_logit_softcap
                    },
                )
            }
        }
    }
}
