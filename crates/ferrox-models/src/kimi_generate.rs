//! Ties `kimi_tokenizer::KimiTokenizer`, `kimi_decoder::kimi_forward_token`,
//! and `sampling::Sampler` into a real text-in/text-out generation loop
//! for Kimi K3 -- the piece that turns "a decoder that can run one
//! forward pass given weights and a token id" into something a CLI or
//! server can actually use for a prompt. Mirrors the shape of
//! `ferrox-server::generate`'s loop (encode prompt, decode incrementally,
//! sample each step, stop at max tokens or EOS, decode the output ids)
//! but against `kimi_decoder`'s separate per-token state/forward-pass
//! API rather than `ferrox_core::cache::KvCache`.

use crate::config::{KdaConfig, MlaConfig};
use crate::kimi_decoder::{
    kimi_forward_token, KimiDecodeState, KimiDecoderConfig, KimiDecoderWeights,
};
use crate::kimi_tokenizer::KimiTokenizer;
use crate::penalty_window::PenaltyWindow;
use crate::sampling::{Sampler, SamplingParams};
use crate::tokenizer::SpecialTokens;

/// Encodes `prompt`, runs it through the decoder to prime `KimiDecodeState`,
/// then samples up to `max_new_tokens` further tokens (greedy if
/// `sampling.temperature <= 0.0`, matching `Sampler::sample`'s
/// convention), stopping early if `eos_id` is produced. Returns the
/// newly generated text (not including the prompt) and the raw
/// generated token ids, since a caller may want both (e.g. to report
/// `finish_reason`/token counts the way `ferrox-server` does for its
/// GGUF path).
#[allow(clippy::too_many_arguments)]
pub fn kimi_generate(
    weights: &KimiDecoderWeights,
    cfg: &KimiDecoderConfig,
    mla_cfg: &MlaConfig,
    kda_cfg: &KdaConfig,
    tokenizer: &KimiTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    sampling: &SamplingParams,
    eos_id: Option<u32>,
    seed: u64,
) -> (String, Vec<u32>) {
    ferrox_core::par::on_workers(move || {
        generate_on_workers(
            weights,
            cfg,
            mla_cfg,
            kda_cfg,
            tokenizer,
            prompt,
            max_new_tokens,
            sampling,
            eos_id,
            seed,
        )
    })
}

/// [`kimi_generate`]'s body, running on a rayon worker.
///
/// The promotion is placed around the WHOLE loop rather than around each
/// `kimi_forward_token`, because this is the outermost public entry
/// point of this path: `ferrox run-kimi` calls `kimi_generate` and
/// nothing else. One `rayon::scope` for the generation makes every
/// parallel region inside it -- every layer of every token, plus the
/// sampler's -- take rayon's in-worker path, where wrapping each forward
/// call instead would still pay one cold entry per token and would put a
/// second spelling of the rule in this file.
///
/// This path does not reach [`crate::engine::Engine`]: it is handed
/// borrowed weights and configs rather than an owned [`crate::KimiEngine`],
/// so `Engine::forward_token`'s promotion cannot cover it.
#[allow(clippy::too_many_arguments)]
fn generate_on_workers(
    weights: &KimiDecoderWeights,
    cfg: &KimiDecoderConfig,
    mla_cfg: &MlaConfig,
    kda_cfg: &KdaConfig,
    tokenizer: &KimiTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    sampling: &SamplingParams,
    eos_id: Option<u32>,
    seed: u64,
) -> (String, Vec<u32>) {
    // `AsText`: the prompt is user text, and `tokenization_kimi.py`
    // encodes that with `disallowed_special=()`. There is no template
    // in front of it here whose markers would need parsing.
    let prompt_ids = tokenizer.encode(prompt, SpecialTokens::AsText);
    let mut state = KimiDecodeState::new(weights, kda_cfg);
    let mut sampler = Sampler::new(seed);
    // The two halves of the penalty window, kept apart because that is
    // what `PenaltyWindow` is built from. The prompt half is what
    // llama.cpp seeds its sampler with before the first draw.
    let mut prompt_history: Vec<usize> = Vec::with_capacity(prompt_ids.len());
    let mut generated_history: Vec<usize> = Vec::with_capacity(max_new_tokens);

    let mut logits = vec![0.0f32; weights.output_head.rows()];
    for &tok in &prompt_ids {
        logits = kimi_forward_token(weights, cfg, mla_cfg, kda_cfg, tok as usize, &mut state);
        prompt_history.push(tok as usize);
    }

    let mut generated_ids = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let next = sampler.sample(
            &logits,
            sampling,
            PenaltyWindow::new(&prompt_history, &generated_history),
        ) as u32;
        if Some(next) == eos_id {
            break;
        }
        generated_ids.push(next);
        generated_history.push(next as usize);
        logits = kimi_forward_token(weights, cfg, mla_cfg, kda_cfg, next as usize, &mut state);
    }

    let text = tokenizer.decode(&generated_ids);
    (text, generated_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kimi_decoder::{KimiDecoderLayerWeights, KimiLayerAttention, KimiLayerFfn};
    use crate::kimi_tokenizer::parse_tiktoken_vocab;
    use crate::latent_moe::KimiMoeConfig;
    use ferrox_core::tensor::Tensor;
    use ferrox_core::weight_matrix::WeightMatrix;
    use std::collections::HashMap;

    /// A minimal but real tiktoken-format vocab: every single byte plus
    /// a handful of multi-byte merges, enough to round-trip short ASCII
    /// text without needing the real 163584-entry Kimi K3 vocab.
    fn tiny_vocab_text() -> String {
        use base64::Engine;
        let mut lines = Vec::new();
        for b in 0u32..256 {
            let bytes = [b as u8];
            let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
            lines.push(format!("{b64} {b}"));
        }
        lines.join("\n")
    }

    /// The smallest thing `kimi_generate` will actually run: a
    /// one-layer dense+KDA stack over the tiny byte vocab, with every
    /// config it needs.
    ///
    /// Extracted rather than written out twice. Two tests drive this
    /// loop for different reasons -- one checks the token sequence it
    /// produces, one counts how many times it enters the worker pool --
    /// and a second copy of a ninety-line builder is exactly where the
    /// two would drift into exercising different graphs while both
    /// still passed.
    #[allow(clippy::type_complexity)]
    fn tiny_kda_stack() -> (
        KimiDecoderWeights,
        KimiDecoderConfig,
        MlaConfig,
        KdaConfig,
        KimiTokenizer,
        usize,
    ) {
        let hidden_dim = 8;
        let kda_num_heads = 2;
        let kda_head_dim = 3;
        let dense_intermediate = 5;
        // Exactly the tiny vocab's size (every one of the 256 single
        // bytes) -- whichever index argmax picks (including tie-break
        // behavior, which favors the last index among equal logits),
        // it must be a real vocab entry that decodes back to a byte.
        let vocab_size = 256;

        let ranks = parse_tiktoken_vocab(&tiny_vocab_text()).expect("must parse synthetic vocab");
        let tokenizer = KimiTokenizer::new(ranks, HashMap::new()).expect("regex must compile");

        // One synthetic dense+KDA layer -- enough to exercise the full
        // generation loop without needing MoE/MLA machinery too (both
        // are already covered end to end by
        // kimi_loader::load_kimi_checkpoint's own test).
        let proj = kda_num_heads * kda_head_dim;
        let zeros_matrix = |rows: usize, cols: usize| {
            WeightMatrix::F32(Tensor::new(vec![0.01f32; rows * cols], vec![rows, cols]))
        };
        let attn = crate::kda::KdaAttnWeights {
            q_proj: zeros_matrix(proj, hidden_dim),
            k_proj: zeros_matrix(proj, hidden_dim),
            v_proj: zeros_matrix(proj, hidden_dim),
            q_conv_weight: vec![0.1; proj * 4],
            k_conv_weight: vec![0.1; proj * 4],
            v_conv_weight: vec![0.1; proj * 4],
            a_log: vec![0.5; kda_num_heads],
            f_a_proj: zeros_matrix(kda_head_dim, hidden_dim),
            f_b_proj: zeros_matrix(proj, kda_head_dim),
            dt_bias: vec![0.1; proj],
            b_proj: zeros_matrix(kda_num_heads, hidden_dim),
            g_proj: zeros_matrix(proj, hidden_dim),
            o_norm_weight: vec![1.0; kda_head_dim],
            o_proj: zeros_matrix(hidden_dim, proj),
        };
        let ffn = crate::kimi_decoder::DenseMlpWeights {
            gate_proj: zeros_matrix(dense_intermediate, hidden_dim),
            up_proj: zeros_matrix(dense_intermediate, hidden_dim),
            down_proj: zeros_matrix(hidden_dim, dense_intermediate),
        };
        let layer = KimiDecoderLayerWeights {
            input_layernorm_weight: vec![1.0; hidden_dim],
            attn: KimiLayerAttention::Kda(Box::new(attn)),
            post_attention_layernorm_weight: vec![1.0; hidden_dim],
            ffn: KimiLayerFfn::Dense(Box::new(ffn)),
            self_attention_res_norm_weight: vec![1.0; hidden_dim],
            self_attention_res_proj_weight: vec![0.01; hidden_dim],
            mlp_res_norm_weight: vec![1.0; hidden_dim],
            mlp_res_proj_weight: vec![0.01; hidden_dim],
        };

        let weights = KimiDecoderWeights {
            embedding: Tensor::new(
                vec![0.02f32; vocab_size * hidden_dim],
                vec![vocab_size, hidden_dim],
            ),
            layers: vec![layer],
            output_attn_res_norm_weight: vec![1.0; hidden_dim],
            output_attn_res_proj_weight: vec![0.01; hidden_dim],
            final_norm_weight: vec![1.0; hidden_dim],
            output_head: zeros_matrix(vocab_size, hidden_dim),
        };

        let decoder_cfg = KimiDecoderConfig {
            attn_res_block_size: 12,
            rms_norm_eps: 1e-5,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            moe: KimiMoeConfig {
                n_experts_active: 1,
                moe_renormalize: true,
                routed_scaling_factor: 1.0,
                situ_beta: 4.0,
                situ_linear_beta: 25.0,
                rms_norm_eps: 1e-5,
            },
        };
        let mla_cfg = MlaConfig {
            num_heads: 1,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 2,
            v_head_dim: 2,
            use_output_gate: true,
            rope: None,
        };
        let kda_cfg = KdaConfig {
            num_heads: kda_num_heads,
            head_dim: kda_head_dim,
            short_conv_kernel_size: 4,
            gate_lower_bound: -5.0,
            use_full_rank_gate: true,
        };

        (
            weights,
            decoder_cfg,
            mla_cfg,
            kda_cfg,
            tokenizer,
            vocab_size,
        )
    }

    #[test]
    fn kimi_generate_produces_a_finite_length_bounded_token_sequence() {
        let (weights, decoder_cfg, mla_cfg, kda_cfg, tokenizer, vocab_size) = tiny_kda_stack();

        let (text, ids) = kimi_generate(
            &weights,
            &decoder_cfg,
            &mla_cfg,
            &kda_cfg,
            &tokenizer,
            "hi",
            5,
            &SamplingParams::default(),
            None,
            42,
        );

        assert!(
            ids.len() <= 5,
            "must never generate more than max_new_tokens: got {}",
            ids.len()
        );
        assert!(ids.iter().all(|&id| (id as usize) < vocab_size));
        // This synthetic vocab maps raw byte values 0..255 directly to
        // single-token entries, which aren't all individually valid
        // UTF-8 (e.g. continuation bytes) -- decode()'s lossy handling
        // may expand length via replacement characters, so this just
        // confirms decode() ran without panicking, not a byte-count
        // equivalence.
        let _ = text;
    }

    /// A whole `kimi_generate` run enters the CPU worker pool ONCE.
    ///
    /// This path never reaches [`crate::engine::Engine`] -- it is handed
    /// borrowed weights rather than an owned `KimiEngine` -- so the
    /// promotion every other engine gets from `engine/entry.rs` cannot
    /// cover it, and before this it paid rayon's cold submission arm
    /// once per parallel region of every token. `ferrox run-kimi` is
    /// the caller.
    ///
    /// Measured against the unpromoted body, because a count of one
    /// means nothing on its own: a loop that opened no regions would
    /// read the same.
    ///
    /// Sabotage: drop the `par::on_workers` from `kimi_generate` and
    /// the two counts become equal.
    #[test]
    fn a_whole_kimi_generation_enters_the_pool_once() {
        let (weights, decoder_cfg, mla_cfg, kda_cfg, tokenizer, _) = tiny_kda_stack();
        if !crate::engine::on_workers_promotes_here() {
            return;
        }
        // `generate_on_workers` is the same loop with the wrapper NOT
        // applied, so this compares the shipped entry point against its
        // own body rather than against a hand-wrapped stand-in.
        let count = |f: &dyn Fn()| {
            let before = ferrox_core::par::cold_regions();
            f();
            ferrox_core::par::cold_regions() - before
        };
        let raw = count(&|| {
            generate_on_workers(
                &weights,
                &decoder_cfg,
                &mla_cfg,
                &kda_cfg,
                &tokenizer,
                "hi",
                5,
                &SamplingParams::default(),
                None,
                42,
            );
        });
        let promoted = count(&|| {
            kimi_generate(
                &weights,
                &decoder_cfg,
                &mla_cfg,
                &kda_cfg,
                &tokenizer,
                "hi",
                5,
                &SamplingParams::default(),
                None,
                42,
            );
        });
        assert_eq!(
            promoted, 1,
            "a whole generation must enter the pool once, not once per region"
        );
        assert!(
            raw > promoted,
            "the unpromoted loop must open a region per parallel section \
             ({raw} vs {promoted}); equal counts mean this is not measuring \
             the promotion at all"
        );
    }

    #[test]
    fn kimi_generate_stops_early_on_eos() {
        let hidden_dim = 4;
        let vocab_size = 8;
        let ranks: HashMap<Vec<u8>, u32> = (0u32..8).map(|b| (vec![b as u8], b)).collect();
        let tokenizer = KimiTokenizer::new(ranks, HashMap::new()).expect("regex must compile");

        let zeros_matrix = |rows: usize, cols: usize| {
            WeightMatrix::F32(Tensor::new(vec![0.0f32; rows * cols], vec![rows, cols]))
        };
        const WINNER_ID: u32 = 3;
        // Every output_head row is zero except WINNER_ID's, which is a
        // large positive constant -- since embedding is nonzero and
        // residual connections carry it through even with every other
        // weight zeroed out, this makes greedy argmax deterministically
        // pick WINNER_ID regardless of any tie-breaking rule (avoids
        // relying on the all-zero-logits case, where `max_by` breaks
        // ties toward the *last* index, not token 0).
        let output_head_data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| {
                if i / hidden_dim == WINNER_ID as usize {
                    5.0
                } else {
                    0.0
                }
            })
            .collect();
        let proj = 2;
        let attn = crate::kda::KdaAttnWeights {
            q_proj: zeros_matrix(proj, hidden_dim),
            k_proj: zeros_matrix(proj, hidden_dim),
            v_proj: zeros_matrix(proj, hidden_dim),
            q_conv_weight: vec![0.0; proj * 4],
            k_conv_weight: vec![0.0; proj * 4],
            v_conv_weight: vec![0.0; proj * 4],
            a_log: vec![0.5; 1],
            f_a_proj: zeros_matrix(2, hidden_dim),
            f_b_proj: zeros_matrix(proj, 2),
            dt_bias: vec![0.0; proj],
            b_proj: zeros_matrix(1, hidden_dim),
            g_proj: zeros_matrix(proj, hidden_dim),
            o_norm_weight: vec![1.0; 2],
            o_proj: zeros_matrix(hidden_dim, proj),
        };
        let ffn = crate::kimi_decoder::DenseMlpWeights {
            gate_proj: zeros_matrix(4, hidden_dim),
            up_proj: zeros_matrix(4, hidden_dim),
            down_proj: zeros_matrix(hidden_dim, 4),
        };
        let layer = KimiDecoderLayerWeights {
            input_layernorm_weight: vec![1.0; hidden_dim],
            attn: KimiLayerAttention::Kda(Box::new(attn)),
            post_attention_layernorm_weight: vec![1.0; hidden_dim],
            ffn: KimiLayerFfn::Dense(Box::new(ffn)),
            self_attention_res_norm_weight: vec![1.0; hidden_dim],
            self_attention_res_proj_weight: vec![0.0; hidden_dim],
            mlp_res_norm_weight: vec![1.0; hidden_dim],
            mlp_res_proj_weight: vec![0.0; hidden_dim],
        };
        let weights = KimiDecoderWeights {
            embedding: Tensor::new(
                vec![0.1f32; vocab_size * hidden_dim],
                vec![vocab_size, hidden_dim],
            ),
            layers: vec![layer],
            output_attn_res_norm_weight: vec![1.0; hidden_dim],
            output_attn_res_proj_weight: vec![0.0; hidden_dim],
            final_norm_weight: vec![1.0; hidden_dim],
            output_head: WeightMatrix::F32(Tensor::new(
                output_head_data,
                vec![vocab_size, hidden_dim],
            )),
        };
        let decoder_cfg = KimiDecoderConfig {
            attn_res_block_size: 12,
            rms_norm_eps: 1e-5,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            moe: KimiMoeConfig {
                n_experts_active: 1,
                moe_renormalize: true,
                routed_scaling_factor: 1.0,
                situ_beta: 4.0,
                situ_linear_beta: 25.0,
                rms_norm_eps: 1e-5,
            },
        };
        let mla_cfg = MlaConfig {
            num_heads: 1,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 2,
            v_head_dim: 2,
            use_output_gate: true,
            rope: None,
        };
        let kda_cfg = KdaConfig {
            num_heads: 1,
            head_dim: 2,
            short_conv_kernel_size: 4,
            gate_lower_bound: -5.0,
            use_full_rank_gate: true,
        };

        let (_, ids) = kimi_generate(
            &weights,
            &decoder_cfg,
            &mla_cfg,
            &kda_cfg,
            &tokenizer,
            "a",
            10,
            &SamplingParams::default(),
            Some(WINNER_ID),
            1,
        );
        assert!(
            ids.is_empty(),
            "eos_id={WINNER_ID} must be hit on the very first sampled token (its logit deterministically wins), got {ids:?}"
        );
    }
}
