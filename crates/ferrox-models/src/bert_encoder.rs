//! BERT: the encoder graph, transcribed from llama.cpp
//! `src/models/bert.cpp` (`llama_model_bert::graph::graph`).
//!
//! Loading lives next door in [`crate::bert_gguf_loader`]; pooling in
//! [`crate::pooling`]; the reason this is not an
//! [`crate::engine::Engine`] in [`crate::encoder`].
//!
//! # The graph, and the five places it is not a decoder
//!
//! ```text
//! h[i] = tok_embd[t[i]] + type_embd[seg[i]] + pos_embd[i] (1) (2)
//! h    = LayerNorm(h, token_embd_norm)                        (3)
//! for each layer:
//!     q,k,v = Wq h + bq,  Wk h + bk,  Wv h + bv
//!     a     = softmax(q·kᵀ / √head_dim) v                  (4)
//!     x     = LayerNorm(Wo a + bo + h,  attn_output_norm)      (3)
//!     f     = W_down · GELU(W_up x + b_up) + b_down        (5)
//!     h     = LayerNorm(f + x, layer_output_norm)              (3)
//! result = h                                              (6)
//! ```
//!
//! 1. **Learned position embeddings, added.** Not RoPE. `pos_embd` is a
//!    real `[n_ctx_train, n_embd]` table and position `i` is a row
//!    lookup. A learned table cannot be extrapolated, which is why
//!    [`crate::encoder::EncodeError::TooLong`] is an error and not a
//!    warning.
//! 2. **A token-type embedding, per position.** A single-sequence
//!    embedding pass is all "Sentence A" and uses row 0, which is what
//!    upstream hardcodes — `ggml_view_1d(ctx0, model.type_embd, n_embd,
//!    0)`, with the comment that token types are hardcoded to zero
//!    because `llama_batch` carries no segment ids. A cross-encoder
//!    PAIR is not that case: HuggingFace's `tokenizer(query, document)`
//!    emits `0…0 1…1` and `BertModel` adds row 1 to every position
//!    after the first `[SEP]`. Ferrox adds the row the caller names,
//!    which is row 0 for every embedding request and 0/1 for a rerank
//!    pair. Matching upstream here instead was measured, on
//!    `cross-encoder/ms-marco-MiniLM-L6-v2` against a NumPy transcription
//!    of `BertForSequenceClassification`, to put the RELEVANT document
//!    LAST in three of four rankings — see
//!    `tests/rerank_cross_encoder_ordering.rs`.
//! 3. **LayerNorm, not RMSNorm, at three sites per layer plus one on
//!    the input.** Mean-subtracting, and every one of them carries a
//!    `bias` tensor as well as a `weight`. Substituting RMSNorm here
//!    loads fine and produces a plausible-looking vector that is wrong.
//! 4. **No causal mask.** Row 0 attends to the last token. This is the
//!    single property that makes the whole model an encoder, and
//!    `attention_is_bidirectional_not_causal` below is the test that
//!    would go red if a mask ever appeared.
//! 5. **A plain GELU MLP, not a gated one.** Two matrices, not three,
//!    and both carry biases. `LLM_FFN_GELU, LLM_FFN_SEQ` upstream.
//! 6. **No output head and no logits.** The hidden states *are* the
//!    result (`res->t_embd`); this checkpoint has no `output.weight` at
//!    all.
//!
//! # What this module does not do
//!
//! Only `arch == "bert"`, and only its dense, non-RoPE, separate-QKV
//! shape. `nomic-bert` (RoPE + gated FFN), `jina-bert-v2` (GEGLU + a
//! second attention norm), `nomic-bert-moe` (expert layers) and
//! `modern-bert` all share `bert.cpp` upstream and are all refused by
//! name in the loader instead of being run through this graph.

use ferrox_core::matmul::{gelu, layer_norm};
use ferrox_core::weight_matrix::WeightMatrix;

use crate::encoder::{EncodeError, PairSequence, TextEncoder};
use crate::pooling::PoolingType;

/// `bert.*` metadata, after the loader has checked it.
#[derive(Debug, Clone)]
pub struct BertHparams {
    pub arch: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// Height of the learned position table.
    pub n_ctx_train: usize,
    pub n_token_types: usize,
    pub layer_norm_eps: f32,
    pub pooling: PoolingType,
    /// `[CLS]` / `[SEP]`, from `tokenizer.ggml.bos_token_id` and
    /// `tokenizer.ggml.seperator_token_id` (upstream's spelling of the
    /// key, typo included). See [`BertEncoder::wrap_special`].
    pub cls_id: u32,
    pub sep_id: u32,
}

impl BertHparams {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

/// One transformer block's weights. Biases that llama.cpp marks
/// `TENSOR_NOT_REQUIRED` are `Option`, so a checkpoint without them is
/// run without them rather than with a silently fabricated zero vector.
pub struct BertLayer {
    pub wq: WeightMatrix,
    pub bq: Option<Vec<f32>>,
    pub wk: WeightMatrix,
    pub bk: Option<Vec<f32>>,
    pub wv: WeightMatrix,
    pub bv: Option<Vec<f32>>,
    pub wo: WeightMatrix,
    pub bo: Option<Vec<f32>>,
    /// `attn_output_norm`, applied after the attention residual.
    pub attn_out_norm_w: Vec<f32>,
    pub attn_out_norm_b: Vec<f32>,
    pub ffn_up: WeightMatrix,
    pub ffn_up_b: Option<Vec<f32>>,
    pub ffn_down: WeightMatrix,
    pub ffn_down_b: Option<Vec<f32>>,
    /// `layer_output_norm`, applied after the FFN residual.
    pub layer_out_norm_w: Vec<f32>,
    pub layer_out_norm_b: Vec<f32>,
}

pub struct BertEncoder {
    pub hp: BertHparams,
    pub tok_embd: WeightMatrix,
    /// `token_types.weight`, **every** row: `[n_token_types, n_embd]`.
    /// Row 0 is "Sentence A" and row 1 "Sentence B". `None` when the
    /// checkpoint carries no table at all, which upstream allows
    /// (`TENSOR_NOT_REQUIRED`) and which means no segment embedding is
    /// added anywhere. Loading only row 0 — what this held before — is
    /// what made a rerank pair score both halves as Sentence A.
    pub type_embd: Option<Vec<Vec<f32>>>,
    pub pos_embd: WeightMatrix,
    pub tok_norm_w: Vec<f32>,
    pub tok_norm_b: Vec<f32>,
    pub layers: Vec<BertLayer>,
}

/// Adds `bias` to every `width`-wide row of `rows`, when there is one.
fn add_bias_rows(rows: &mut [f32], width: usize, bias: Option<&Vec<f32>>) {
    let Some(b) = bias else { return };
    debug_assert_eq!(b.len(), width);
    for row in rows.chunks_exact_mut(width) {
        for (x, bv) in row.iter_mut().zip(b.iter()) {
            *x += bv;
        }
    }
}

/// LayerNorm applied independently to each `width`-wide row, in place.
fn layer_norm_rows(rows: &mut [f32], width: usize, weight: &[f32], bias: &[f32], eps: f32) {
    for row in rows.chunks_exact_mut(width) {
        let normed = layer_norm(row, weight, bias, eps);
        row.copy_from_slice(&normed);
    }
}

/// In-place softmax over one score row, max-shifted.
fn softmax_row(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    let inv = 1.0 / sum;
    for s in scores.iter_mut() {
        *s *= inv;
    }
}

/// Full bidirectional multi-head attention over `n` positions.
///
/// `q` is `[n][n_head * head_dim]`; `k` and `v` are
/// `[n][n_head_kv * head_dim]`. **Every query row attends to every key
/// row** — there is no mask argument here on purpose, so a causal mask
/// cannot be added by accident.
fn bidirectional_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
) -> Vec<f32> {
    let q_width = n_head * head_dim;
    let kv_width = n_head_kv * head_dim;
    let heads_per_kv = n_head / n_head_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n * q_width];
    let mut scores = vec![0.0f32; n];
    for h in 0..n_head {
        let kv_h = h / heads_per_kv;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;
        for i in 0..n {
            let qi = &q[i * q_width + q_off..i * q_width + q_off + head_dim];
            for (j, s) in scores.iter_mut().enumerate() {
                let kj = &k[j * kv_width + kv_off..j * kv_width + kv_off + head_dim];
                *s = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            softmax_row(&mut scores);
            let dst = &mut out[i * q_width + q_off..i * q_width + q_off + head_dim];
            for (j, &p) in scores.iter().enumerate() {
                let vj = &v[j * kv_width + kv_off..j * kv_width + kv_off + head_dim];
                for (o, &vv) in dst.iter_mut().zip(vj) {
                    *o += p * vv;
                }
            }
        }
    }
    out
}

impl BertEncoder {
    pub fn vocab_size(&self) -> usize {
        self.tok_embd.rows()
    }
}

impl TextEncoder for BertEncoder {
    fn n_embd(&self) -> usize {
        self.hp.n_embd
    }

    fn n_ctx_train(&self) -> usize {
        self.hp.n_ctx_train
    }

    fn pooling_type(&self) -> PoolingType {
        self.hp.pooling
    }

    /// `[CLS] … [SEP]`, which is what llama.cpp's WPM branch adds when
    /// `add_special` is set: it pushes `special_bos_id` before the
    /// pieces and `special_sep_id` after, unconditionally — the
    /// `add_bos`/`add_eos` flags are not consulted on that path
    /// (`llama-vocab.cpp`, `case LLAMA_VOCAB_TYPE_WPM`).
    fn wrap_special(&self, pieces: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(pieces.len() + 2);
        out.push(self.hp.cls_id);
        out.extend_from_slice(pieces);
        out.push(self.hp.sep_id);
        out
    }

    /// The height of `token_types.weight`, or 1 when the checkpoint
    /// carries no table (nothing is added at any position, which is
    /// what a one-row table would do anyway).
    fn n_segments(&self) -> usize {
        self.type_embd.as_ref().map(Vec::len).unwrap_or(1)
    }

    /// `[CLS] a [SEP] b [SEP]` with segments `0…0 1…1` — what
    /// HuggingFace's `tokenizer(query, document)` builds for a BERT
    /// cross-encoder, which is the input these checkpoints were
    /// trained on.
    ///
    /// The boundary is defined once, here, and both vectors are cut on
    /// it: the first `[SEP]` closes segment 0 (HF counts it as part of
    /// the first half) and everything after it is segment 1. Returning
    /// the ids alone and letting the graph assume a segment is how the
    /// document half came to be scored as "Sentence A".
    fn wrap_special_pair(&self, a: &[u32], b: &[u32]) -> Option<PairSequence> {
        let mut tokens = Vec::with_capacity(a.len() + b.len() + 3);
        tokens.push(self.hp.cls_id);
        tokens.extend_from_slice(a);
        tokens.push(self.hp.sep_id);
        let first_half = tokens.len();
        tokens.extend_from_slice(b);
        tokens.push(self.hp.sep_id);
        let mut segments = vec![0u32; tokens.len()];
        for s in segments[first_half..].iter_mut() {
            *s = 1;
        }
        Some(PairSequence { tokens, segments })
    }

    fn encode_on_worker(
        &self,
        tokens: &[u32],
        segments: Option<&[u32]>,
    ) -> Result<Vec<f32>, EncodeError> {
        let n = tokens.len();
        if n == 0 {
            return Err(EncodeError::EmptySequence);
        }
        if let Some(seg) = segments {
            if seg.len() != n {
                return Err(EncodeError::RaggedSegments {
                    tokens: n,
                    segments: seg.len(),
                });
            }
        }
        if n > self.hp.n_ctx_train {
            return Err(EncodeError::TooLong {
                got: n,
                max: self.hp.n_ctx_train,
                arch: self.hp.arch.clone(),
            });
        }
        let d = self.hp.n_embd;
        let vocab_size = self.vocab_size();

        // (1)(2) token + type + position, then the input LayerNorm.
        let mut h = vec![0.0f32; n * d];
        for (i, &t) in tokens.iter().enumerate() {
            if t as usize >= vocab_size {
                return Err(EncodeError::TokenOutOfRange { id: t, vocab_size });
            }
            let tok = self.tok_embd.dequant_row(t as usize);
            let pos = self.pos_embd.dequant_row(i);
            let row = &mut h[i * d..(i + 1) * d];
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = tok[j] + pos[j];
            }
            if let Some(table) = &self.type_embd {
                let seg = segments.map(|s| s[i]).unwrap_or(0);
                let ty = table
                    .get(seg as usize)
                    .ok_or(EncodeError::SegmentOutOfRange {
                        id: seg,
                        pos: i,
                        n_segments: table.len(),
                    })?;
                for (slot, tv) in row.iter_mut().zip(ty.iter()) {
                    *slot += tv;
                }
            }
        }
        layer_norm_rows(
            &mut h,
            d,
            &self.tok_norm_w,
            &self.tok_norm_b,
            self.hp.layer_norm_eps,
        );

        let head_dim = self.hp.head_dim();
        for layer in &self.layers {
            let mut q = layer.wq.apply_batch(&h, n);
            let mut k = layer.wk.apply_batch(&h, n);
            let mut v = layer.wv.apply_batch(&h, n);
            add_bias_rows(&mut q, self.hp.n_head * head_dim, layer.bq.as_ref());
            add_bias_rows(&mut k, self.hp.n_head_kv * head_dim, layer.bk.as_ref());
            add_bias_rows(&mut v, self.hp.n_head_kv * head_dim, layer.bv.as_ref());

            let attn =
                bidirectional_attention(&q, &k, &v, n, self.hp.n_head, self.hp.n_head_kv, head_dim);

            let mut x = layer.wo.apply_batch(&attn, n);
            add_bias_rows(&mut x, d, layer.bo.as_ref());
            // Residual over the *layer input*, then attn_output_norm.
            for (xv, hv) in x.iter_mut().zip(h.iter()) {
                *xv += hv;
            }
            layer_norm_rows(
                &mut x,
                d,
                &layer.attn_out_norm_w,
                &layer.attn_out_norm_b,
                self.hp.layer_norm_eps,
            );

            // (5) plain GELU MLP; the FFN residual is over `x`, i.e.
            // over the post-norm value, not over the layer input.
            let mut up = layer.ffn_up.apply_batch(&x, n);
            add_bias_rows(&mut up, self.hp.n_ff, layer.ffn_up_b.as_ref());
            for a in up.iter_mut() {
                *a = gelu(*a);
            }
            let mut down = layer.ffn_down.apply_batch(&up, n);
            add_bias_rows(&mut down, d, layer.ffn_down_b.as_ref());
            for (dv, xv) in down.iter_mut().zip(x.iter()) {
                *dv += xv;
            }
            layer_norm_rows(
                &mut down,
                d,
                &layer.layer_out_norm_w,
                &layer.layer_out_norm_b,
                self.hp.layer_norm_eps,
            );
            h = down;
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_core::tensor::Tensor;

    /// Deterministic pseudo-random weights: a small LCG, so the fixture
    /// is reproducible without pulling in a dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
        fn matrix(&mut self, rows: usize, cols: usize) -> WeightMatrix {
            WeightMatrix::F32(Tensor::new(self.vec(rows * cols), vec![rows, cols]))
        }
    }

    const D: usize = 8;
    const FF: usize = 16;
    const HEADS: usize = 2;
    const VOCAB: usize = 20;
    const CTX: usize = 12;
    const EPS: f32 = 1e-12;

    fn fixture(n_layer: usize) -> BertEncoder {
        let mut r = Lcg(0x5EED);
        let tok_embd = r.matrix(VOCAB, D);
        let pos_embd = r.matrix(CTX, D);
        // Two rows, like every real BERT: "Sentence A" and "Sentence B".
        let type_embd = Some(vec![r.vec(D), r.vec(D)]);
        let tok_norm_w = r.vec(D);
        let tok_norm_b = r.vec(D);
        let layers = (0..n_layer)
            .map(|_| BertLayer {
                wq: r.matrix(D, D),
                bq: Some(r.vec(D)),
                wk: r.matrix(D, D),
                bk: Some(r.vec(D)),
                wv: r.matrix(D, D),
                bv: Some(r.vec(D)),
                wo: r.matrix(D, D),
                bo: Some(r.vec(D)),
                attn_out_norm_w: r.vec(D),
                attn_out_norm_b: r.vec(D),
                ffn_up: r.matrix(FF, D),
                ffn_up_b: Some(r.vec(FF)),
                ffn_down: r.matrix(D, FF),
                ffn_down_b: Some(r.vec(D)),
                layer_out_norm_w: r.vec(D),
                layer_out_norm_b: r.vec(D),
            })
            .collect();
        BertEncoder {
            hp: BertHparams {
                arch: "bert".into(),
                n_layer,
                n_embd: D,
                n_ff: FF,
                n_head: HEADS,
                n_head_kv: HEADS,
                n_ctx_train: CTX,
                n_token_types: 2,
                layer_norm_eps: EPS,
                pooling: PoolingType::Cls,
                cls_id: 1,
                sep_id: 2,
            },
            tok_embd,
            type_embd,
            pos_embd,
            tok_norm_w,
            tok_norm_b,
            layers,
        }
    }

    /// An f64 transcription of the graph in the module docs, written
    /// the slowest possible way: no `apply_batch`, no shared buffers,
    /// one scalar loop per matrix element. It exists to disagree with
    /// [`BertEncoder::encode`] if the fast path transposes a matrix,
    /// drops a bias, norms the wrong residual, reuses a buffer it
    /// should not, or reads the wrong row of the token-type table.
    fn reference_forward(m: &BertEncoder, tokens: &[u32], segments: &[u32]) -> Vec<f64> {
        let d = m.hp.n_embd;
        let n = tokens.len();
        let hd = m.hp.head_dim();

        let dense = |w: &WeightMatrix| -> Vec<Vec<f64>> {
            (0..w.rows())
                .map(|r| w.dequant_row(r).iter().map(|&v| v as f64).collect())
                .collect()
        };
        let matvec = |w: &Vec<Vec<f64>>, x: &[f64]| -> Vec<f64> {
            w.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let ln = |x: &[f64], wt: &[f32], b: &[f32]| -> Vec<f64> {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
            let inv = 1.0 / (var + m.hp.layer_norm_eps as f64).sqrt();
            x.iter()
                .zip(wt)
                .zip(b)
                .map(|((v, w), bb)| (v - mean) * inv * (*w as f64) + (*bb as f64))
                .collect()
        };

        let mut h: Vec<Vec<f64>> = tokens
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                let tok = m.tok_embd.dequant_row(t as usize);
                let pos = m.pos_embd.dequant_row(i);
                let ty = m
                    .type_embd
                    .as_ref()
                    .map(|t| t[segments[i] as usize].clone())
                    .unwrap_or_else(|| vec![0.0; d]);
                let row: Vec<f64> = (0..d)
                    .map(|j| tok[j] as f64 + pos[j] as f64 + ty[j] as f64)
                    .collect();
                ln(&row, &m.tok_norm_w, &m.tok_norm_b)
            })
            .collect();

        for layer in &m.layers {
            let (wq, wk, wv, wo) = (
                dense(&layer.wq),
                dense(&layer.wk),
                dense(&layer.wv),
                dense(&layer.wo),
            );
            let (wu, wd) = (dense(&layer.ffn_up), dense(&layer.ffn_down));
            let bias = |v: &mut Vec<f64>, b: &Option<Vec<f32>>| {
                if let Some(b) = b {
                    for (x, bb) in v.iter_mut().zip(b) {
                        *x += *bb as f64;
                    }
                }
            };
            let mut q = Vec::new();
            let mut k = Vec::new();
            let mut v = Vec::new();
            for row in &h {
                let mut a = matvec(&wq, row);
                bias(&mut a, &layer.bq);
                q.push(a);
                let mut a = matvec(&wk, row);
                bias(&mut a, &layer.bk);
                k.push(a);
                let mut a = matvec(&wv, row);
                bias(&mut a, &layer.bv);
                v.push(a);
            }
            let mut attn = vec![vec![0.0f64; d]; n];
            for head in 0..m.hp.n_head {
                let off = head * hd;
                for i in 0..n {
                    let raw: Vec<f64> = (0..n)
                        .map(|j| {
                            (0..hd).map(|c| q[i][off + c] * k[j][off + c]).sum::<f64>()
                                / (hd as f64).sqrt()
                        })
                        .collect();
                    let mx = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let ex: Vec<f64> = raw.iter().map(|s| (s - mx).exp()).collect();
                    let sum: f64 = ex.iter().sum();
                    for j in 0..n {
                        let p = ex[j] / sum;
                        for c in 0..hd {
                            attn[i][off + c] += p * v[j][off + c];
                        }
                    }
                }
            }
            let mut next = Vec::new();
            for i in 0..n {
                let mut o = matvec(&wo, &attn[i]);
                bias(&mut o, &layer.bo);
                for (x, hv) in o.iter_mut().zip(&h[i]) {
                    *x += hv;
                }
                let x = ln(&o, &layer.attn_out_norm_w, &layer.attn_out_norm_b);
                let mut up = matvec(&wu, &x);
                bias(&mut up, &layer.ffn_up_b);
                let act: Vec<f64> = up
                    .iter()
                    .map(|&u| {
                        const K: f64 = 0.797_884_560_802_865_4;
                        const C: f64 = 0.044_715;
                        0.5 * u * (1.0 + (K * (u + C * u * u * u)).tanh())
                    })
                    .collect();
                let mut down = matvec(&wd, &act);
                bias(&mut down, &layer.ffn_down_b);
                for (dv, xv) in down.iter_mut().zip(&x) {
                    *dv += xv;
                }
                next.push(ln(&down, &layer.layer_out_norm_w, &layer.layer_out_norm_b));
            }
            h = next;
        }
        h.into_iter().flatten().collect()
    }

    #[test]
    fn matches_an_independent_f64_transcription_of_the_graph() {
        let m = fixture(3);
        let tokens = [1u32, 7, 13, 4, 9, 2];
        let got = m.encode_tokens(&tokens).unwrap();
        let want = reference_forward(&m, &tokens, &[0; 6]);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (*g as f64 - w).abs() < 2e-4,
                "element {i}: {g} vs reference {w}"
            );
        }
    }

    /// The same transcription, driven with a real `0 0 0 1 1 1` split.
    /// The point is not that segments *do something* — it is that the
    /// fast path reads the SAME row the reference does at every
    /// position, so an off-by-one on the boundary or a table indexed
    /// with the token id would show up here.
    #[test]
    fn the_segment_id_selects_the_token_type_row_at_every_position() {
        let m = fixture(3);
        let tokens = [1u32, 7, 13, 4, 9, 2];
        let segments = [0u32, 0, 0, 1, 1, 1];
        let got = m.encode(&tokens, Some(&segments)).unwrap();
        let want = reference_forward(&m, &tokens, &segments);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (*g as f64 - w).abs() < 2e-4,
                "element {i}: {g} vs reference {w}"
            );
        }
        // And it is genuinely a different graph from the all-zeros one,
        // which is the whole of issue #44: scoring the second half as
        // "Sentence A" is not a rounding difference.
        let all_zero = m.encode_tokens(&tokens).unwrap();
        let moved: f32 = all_zero
            .iter()
            .zip(&got)
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>();
        assert!(moved > 1e-3, "segment 1 changed nothing ({moved})");
    }

    /// A segment id with no row, and a segment list that is not one per
    /// token, are refusals rather than a panic or a silently wrong row.
    #[test]
    fn a_segment_id_off_the_table_and_a_ragged_segment_list_are_refused() {
        let m = fixture(1);
        assert!(matches!(
            m.encode(&[1, 7, 2], Some(&[0, 2, 0])),
            Err(EncodeError::SegmentOutOfRange { id: 2, pos: 1, .. })
        ));
        assert!(matches!(
            m.encode(&[1, 7, 2], Some(&[0, 0])),
            Err(EncodeError::RaggedSegments {
                tokens: 3,
                segments: 2
            })
        ));
    }

    /// The property that makes this an encoder. Row 0's output must
    /// change when the *last* token changes; under a causal mask it
    /// could not, because position 0 would attend only to itself.
    #[test]
    fn attention_is_bidirectional_not_causal() {
        let m = fixture(2);
        let a = m.encode_tokens(&[5u32, 6, 7, 8]).unwrap();
        let b = m.encode_tokens(&[5u32, 6, 7, 19]).unwrap();
        let moved: f32 = a[..D].iter().zip(&b[..D]).map(|(x, y)| (x - y).abs()).sum();
        assert!(
            moved > 1e-3,
            "row 0 barely moved ({moved}) when the last token changed — \
             attention is behaving causally"
        );
    }

    /// Position is a learned table lookup, so the same token at a
    /// different index must land somewhere else.
    #[test]
    fn position_embeddings_make_the_same_token_differ_by_index() {
        let m = fixture(1);
        let out = m.encode_tokens(&[11u32, 11]).unwrap();
        let delta: f32 = out[..D]
            .iter()
            .zip(&out[D..2 * D])
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(
            delta > 1e-3,
            "identical tokens gave identical rows: {delta}"
        );
    }

    /// The graph ends on a LayerNorm: with unit weight and zero bias
    /// each output row is mean-zero and unit-variance. An RMSNorm in
    /// that slot would leave the mean wherever it was.
    #[test]
    fn the_last_op_is_a_mean_subtracting_layer_norm() {
        let mut m = fixture(2);
        let last = m.layers.last_mut().unwrap();
        last.layer_out_norm_w = vec![1.0; D];
        last.layer_out_norm_b = vec![0.0; D];
        let out = m.encode_tokens(&[3u32, 4, 5]).unwrap();
        for row in out.as_chunks::<D>().0 {
            let mean: f32 = row.iter().sum::<f32>() / D as f32;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / D as f32;
            assert!(mean.abs() < 1e-4, "row mean {mean} is not zero");
            assert!((var - 1.0).abs() < 1e-3, "row variance {var} is not one");
        }
    }

    #[test]
    fn refuses_an_empty_sequence_and_one_past_the_position_table() {
        let m = fixture(1);
        assert!(matches!(
            m.encode_tokens(&[]),
            Err(EncodeError::EmptySequence)
        ));
        let long: Vec<u32> = (0..CTX as u32 + 1).map(|i| i % VOCAB as u32).collect();
        let err = m.encode_tokens(&long).unwrap_err();
        assert!(
            matches!(err, EncodeError::TooLong { got, max, .. } if got == CTX + 1 && max == CTX)
        );
        assert!(matches!(
            m.encode_tokens(&[VOCAB as u32]),
            Err(EncodeError::TokenOutOfRange { .. })
        ));
    }

    #[test]
    fn wrap_special_brackets_the_pieces_with_cls_and_sep() {
        let m = fixture(1);
        assert_eq!(m.wrap_special(&[7, 8]), vec![1, 7, 8, 2]);
        assert_eq!(m.wrap_special(&[]), vec![1, 2]);
    }

    /// The cross-encoder input is `[CLS] a [SEP] b [SEP]` with segments
    /// `0 0 0 0 1 1` — the boundary between the two halves is the whole
    /// reason a reranker scores differently from an embedding model.
    /// Concatenating without it, dropping the trailing `[SEP]`, or
    /// leaving every segment at 0, produces a perfectly plausible
    /// ranking that is not the model's, so both vectors are asserted
    /// exactly rather than by length.
    ///
    /// The first `[SEP]` belongs to segment 0, which is what
    /// HuggingFace's `tokenizer(query, document)` emits: an off-by-one
    /// there is a one-position difference that no shape check catches.
    #[test]
    fn the_pair_form_separates_the_two_halves_and_labels_each_one() {
        let m = fixture(1);
        let pair = m.wrap_special_pair(&[7, 8], &[9]).unwrap();
        assert_eq!(pair.tokens, vec![1, 7, 8, 2, 9, 2]);
        assert_eq!(pair.segments, vec![0, 0, 0, 0, 1, 1]);
        // An empty half is still a half: the boundary stays.
        let empty = m.wrap_special_pair(&[], &[]).unwrap();
        assert_eq!(empty.tokens, vec![1, 2, 2]);
        assert_eq!(empty.segments, vec![0, 0, 1]);
        // And it is NOT the single-sequence form of the two texts run
        // together, which is what a defaulted implementation would give.
        assert_ne!(pair.tokens, m.wrap_special(&[7, 8, 9]));
    }

    /// A checkpoint with no "Sentence B" row cannot express a pair, and
    /// [`crate::EmbeddingModel`] refuses one at load. This is the value
    /// that refusal reads.
    #[test]
    fn n_segments_is_the_height_of_the_token_type_table() {
        let mut m = fixture(1);
        assert_eq!(m.n_segments(), 2);
        m.type_embd = Some(vec![vec![0.0; D]]);
        assert_eq!(m.n_segments(), 1);
        m.type_embd = None;
        assert_eq!(m.n_segments(), 1);
    }

    /// `embed_tokens` must return the CLS row of the hidden states this
    /// checkpoint's `pooling_type` names, not the mean and not the last.
    #[test]
    fn embed_tokens_pools_the_way_the_hparams_say() {
        let m = fixture(2);
        let tokens = [1u32, 9, 4, 2];
        let hidden = m.encode_tokens(&tokens).unwrap();
        assert_eq!(m.embed_tokens(&tokens).unwrap(), hidden[..D].to_vec());
    }
}
