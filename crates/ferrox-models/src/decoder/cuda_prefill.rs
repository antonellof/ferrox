//! The CUDA resident prefill layer, as the batched host body reaches
//! it (#259, `docs/plans/cpu-cuda-parity.md` step 2b).
//!
//! One dense layer per call: the hidden batch goes up once, everything
//! between the seven matmuls is a kernel, and the new hidden batch and
//! the batch's K/V rows come back -- against the 111 MB and seven
//! synchronous round trips per Llama-3.2-3B layer the host body spends
//! at pp512. The host `KvCache` stays authoritative: the rows come back
//! and are pushed here, and a prefix already in the cache is uploaded
//! per layer for the attention kernel. Which layers this may take is
//! `fused_view`'s decision, shared with the Metal stack; what this file
//! adds is the one question only CUDA can answer (does the GEMM serve
//! every projection) and the mapping onto the launch's types.

use ferrox_core::cache::KvCache;
use ferrox_cuda::prefill::{
    launch_prefill_dense_layer, AttnExtrasCuda, LayerRopeCuda, MulMmWeights, PrefillDenseLayerCuda,
    PrefillParams, RopeLayoutCuda,
};

use super::fused_view::FusedAttnExtras;
use super::{Decoder, LayerWeights};

impl Decoder {
    /// Runs layer `l` over `hidden_batch` on the device and pushes the
    /// batch's K/V rows into `cache`, or answers `None` -- BEFORE
    /// touching the device or the cache -- for a layer or a state the
    /// launch does not serve, so the host body runs it instead. A
    /// launch that fails after admission is reported once and also
    /// answers `None`, with the cache untouched.
    pub(crate) fn try_cuda_prefill_dense_layer(
        &self,
        l: usize,
        layer: &LayerWeights,
        hidden_batch: &[f32],
        start_pos: usize,
        batch_size: usize,
        cache: &mut KvCache,
    ) -> Option<Vec<f32>> {
        if batch_size < 4
            || !ferrox_core::weight_matrix::cuda_dense_enabled()
            || !Self::fused_prefill_dense_layer_eligible(layer, &self.config, self.lora_attached())
            || !self.layer_supports_fused_attn(layer)
        {
            return None;
        }
        // The activation the fused body runs is SwiGLU; `Some(true)` is
        // GELU, which this launch has no kernel for yet.
        if self
            .config
            .model_ffn_act()
            .and_then(ferrox_moe::GluAct::fused_kernel_gelu_flag)
            != Some(false)
        {
            return None;
        }
        // POSITIONS and ROWS both: the prefix the kernel attends over
        // is the cache's rows, and they are only the positions when
        // nothing has been evicted. A windowed cache that has evicted
        // takes the host body, which reads the window it kept.
        if cache.positions() != start_pos || cache.rows() != start_pos {
            return None;
        }
        let head_dim = self.config.head_dim;
        let (n_heads, n_kv_heads) = (self.config.n_heads, self.config.n_kv_heads);
        let kv_width = n_kv_heads * head_dim;
        if cache.n_kv_heads != n_kv_heads
            || cache.head_dim != head_dim
            || cache.v_head_dim != head_dim
        {
            return None;
        }

        let FusedAttnExtras {
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
        } = Self::fused_attn_extras(layer)?;
        let rope = self.config.layer_rope(l).map(|r| LayerRopeCuda {
            theta: r.theta,
            freq_factors: r.freq_factors,
            rot_dim: r.rot_dim.unwrap_or(head_dim),
            layout: match self.config.rope_layout {
                crate::config::RopeLayout::Norm => RopeLayoutCuda::Norm,
                crate::config::RopeLayout::Neox => RopeLayoutCuda::Neox,
            },
            // `apply_rope_attn_factor` on the host: the rotated channels
            // of Q and K, on a layer that rotates.
            mscale: self.config.rope_attn_factor,
        });
        let attn_norm_w = layer.attn.norm_weight.rms_weights()?;
        let ffn_norm_w = layer.moe.norm_weight.rms_weights()?;
        fn view(m: &ferrox_core::WeightMatrix) -> Option<MulMmWeights<'_>> {
            m.cuda_mul_mm_view()
        }
        let (q, k, v, o) = (
            view(&layer.attn.q_proj)?,
            view(&layer.attn.k_proj)?,
            view(&layer.attn.v_proj)?,
            view(&layer.attn.o_proj)?,
        );

        let out = layer.moe.with_expert(0, |ex| {
            let layer_cuda = PrefillDenseLayerCuda {
                attn_norm_w,
                ffn_norm_w,
                q,
                k,
                v,
                o,
                gate: view(&ex.gate)?,
                up: view(&ex.up)?,
                down: view(&ex.down)?,
                post_attn_norm: layer.attn.post_attn_norm.as_deref(),
                post_ffn_norm: layer.attn.post_ffn_norm.as_deref(),
                extras: AttnExtrasCuda {
                    q_bias,
                    k_bias,
                    v_bias,
                    q_norm,
                    k_norm,
                },
                rope,
            };
            let params = PrefillParams {
                n_heads,
                n_kv_heads,
                head_dim,
                rms_eps: self.config.rms_norm_eps,
                // The host body's scale (`causal_gqa_attention_row`);
                // `attention_scale` is fenced off by
                // `layer_supports_fused_attn`.
                attn_scale: 1.0 / (head_dim as f32).sqrt(),
                attn_softcap: self.config.attn_logit_softcap,
                window: self.config.layer_sliding_window(l),
                prefix_k: &cache.k[..start_pos * kv_width],
                prefix_v: &cache.v[..start_pos * kv_width],
                start_pos,
            };
            match launch_prefill_dense_layer(hidden_batch, &layer_cuda, &params, batch_size) {
                Ok(out) => Some(out),
                Err(e) => {
                    warn_once(&format!(
                        "ferrox: CUDA prefill layer declined, host body runs it: {e}"
                    ));
                    None
                }
            }
        })?;
        layer.moe.record_activations(&[0]);
        for b in 0..batch_size {
            cache
                .push(
                    &out.k_rows[b * kv_width..(b + 1) * kv_width],
                    &out.v_rows[b * kv_width..(b + 1) * kv_width],
                )
                .expect("unbounded/planned KvCache growth is infallible");
        }
        Some(out.hidden)
    }
}

/// One line per distinct message per process: a launch that declines
/// every layer of every prefill would otherwise say so 28 times a step.
fn warn_once(msg: &str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
    if guard
        .get_or_insert_with(HashSet::new)
        .insert(msg.to_string())
    {
        eprintln!("{msg}");
    }
}
