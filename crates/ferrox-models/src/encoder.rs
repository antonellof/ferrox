//! The seam for encoder-only (embedding) models, which is deliberately
//! **not** [`crate::engine::Engine`].
//!
//! # Why a second trait and not a variant of the first
//!
//! [`crate::engine::Engine`] is `forward_token(token_id, pos, &mut
//! State) -> Vec<f32>`: one token in, one logit vector out, carrying
//! per-layer state forward. Every part of that signature is an
//! autoregression assumption, and a BERT encoder violates all of them:
//!
//! * **There is no state to carry.** Attention is bidirectional, so
//!   token 0's output depends on token 7. Nothing can be computed until
//!   the whole sequence is present, and nothing computed for one
//!   sequence is reusable for the next. A KV cache is not merely
//!   unnecessary here, it is meaningless — there is no "next token" for
//!   a cached key to be attended to by.
//! * **There are no logits.** The result is `n_tokens × n_embd` hidden
//!   states (llama.cpp's `res->t_embd`); this checkpoint has no output
//!   head at all, and `token_embd.weight` is not tied to one.
//! * **`pos` is not a cursor, it is an index into a learned table.**
//!   BERT adds `position_embd.weight[i]` to the token embedding rather
//!   than rotating Q/K, which is why the sequence length is hard-capped
//!   by `n_ctx_train` instead of merely degrading past it.
//!
//! Forcing that through `Engine` would mean a `State` that is a
//! pretend-cache, a `forward_token` that can only be called with the
//! last position after a hidden batch call, and a `vocab_size` that has
//! no meaning. The Kimi comment on `Engine` already records the cost of
//! bending a trait around a model it does not fit; this is the same
//! judgement, made before rather than after.
//!
//! So: [`TextEncoder`] is sequence-in, matrix-out, stateless. What the
//! two seams *do* share is pooling ([`crate::pooling`]), which is why
//! that lives in its own module and not in either of them.

use crate::pooling::{pool, PoolingError, PoolingType};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EncodeError {
    #[error("an encoder needs at least one token; got an empty sequence")]
    EmptySequence,
    #[error(
        "sequence of {got} tokens exceeds the {max} learned position embeddings this \
         checkpoint carries ({arch}.context_length). A learned position table cannot be \
         extrapolated the way RoPE can, so this is a hard limit, not a quality cliff — \
         truncate the input or use a longer-context embedding model"
    )]
    TooLong {
        got: usize,
        max: usize,
        arch: String,
    },
    #[error("token id {id} is outside this checkpoint's {vocab_size}-entry vocabulary")]
    TokenOutOfRange { id: u32, vocab_size: usize },
    #[error(
        "segment id {id} at position {pos} is outside this checkpoint's {n_segments}-row \
         token-type table"
    )]
    SegmentOutOfRange {
        id: u32,
        pos: usize,
        n_segments: usize,
    },
    #[error(
        "{tokens} token(s) were given {segments} segment id(s); every position needs exactly \
         one, or the graph would add a segment embedding to the wrong row"
    )]
    RaggedSegments { tokens: usize, segments: usize },
    #[error(transparent)]
    Pooling(#[from] PoolingError),
}

/// One two-segment encoder input: the token ids and, for each of them,
/// which half of the pair it belongs to.
///
/// The two vectors are built together and travel together on purpose.
/// They are the repo's dominant bug shape waiting to happen — two
/// structures that must agree, here about a length and about where the
/// boundary is — and the single place that can get them right is
/// [`TextEncoder::wrap_special_pair`], which inserts the `[SEP]` that
/// the boundary is defined by. Handing a caller the ids alone (what
/// this seam used to do) meant the segment ids did not exist at all
/// and every position was scored as "Sentence A".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairSequence {
    /// `[CLS] a [SEP] b [SEP]` for BERT.
    pub tokens: Vec<u32>,
    /// `0` for the `[CLS] a [SEP]` half, `1` for the `b [SEP]` half —
    /// HuggingFace's `token_type_ids`. Always the same length as
    /// [`Self::tokens`].
    pub segments: Vec<u32>,
}

/// A model that turns a whole token sequence into hidden states in one
/// pass, with no carried state and no logits.
///
/// `Sync` because [`TextEncoder::encode`] hands `&self` to a rayon
/// worker for the duration of the pass; see its doc comment.
pub trait TextEncoder: Sync {
    /// Width of one hidden-state row, and of the pooled embedding.
    fn n_embd(&self) -> usize;

    /// Longest sequence this checkpoint can represent. For a learned
    /// position table this is the table's height, and exceeding it is
    /// an error rather than a degradation.
    fn n_ctx_train(&self) -> usize;

    /// What the checkpoint's own `{arch}.pooling_type` said.
    fn pooling_type(&self) -> PoolingType;

    /// Wraps a tokenizer's pieces in whatever the *model* requires
    /// around them — for BERT, `[CLS] … [SEP]`.
    ///
    /// This is on the encoder rather than the tokenizer because ferrox's
    /// tokenizers deliberately encode text only (they are checked
    /// token-for-token against `llama_tokenize(..., add_special =
    /// false, ...)`), while llama.cpp keeps `add_special` in the vocab
    /// and applies it here. The default adds nothing, so an encoder that
    /// genuinely needs no wrapper does not have to say so.
    fn wrap_special(&self, pieces: &[u32]) -> Vec<u32> {
        pieces.to_vec()
    }

    /// How many rows the checkpoint's token-type ("segment") table
    /// carries, i.e. the number of distinct segment ids
    /// [`Self::encode`] will accept.
    ///
    /// `1` — the default — means the encoder can only represent
    /// "Sentence A", which is enough for an embedding pass and is not
    /// enough for a cross-encoder pair. Read at load time by
    /// [`crate::EmbeddingModel`], which refuses a reranker checkpoint
    /// that cannot express segment 1 rather than silently scoring the
    /// document half as segment 0.
    fn n_segments(&self) -> usize {
        1
    }

    /// The two-segment input a **cross-encoder** scores: one sequence
    /// holding a query and a document with the model's own boundary
    /// between them, and the segment id of every position. For BERT
    /// that is `[CLS] a [SEP] b [SEP]` with segments `0…0 1…1`, which is
    /// exactly what HuggingFace's `tokenizer(query, document)` emits.
    ///
    /// `None` — the default — means this encoder has no two-segment
    /// form, and a caller that needs one must refuse. Deliberately NOT
    /// defaulted to `wrap_special(a ++ b)`: a cross-encoder was trained
    /// with a separator between the halves, and one that never sees it
    /// still returns a plausible float. That is the "computes something
    /// else" failure, and it is invisible — a rerank with no boundary
    /// produces an ordering, just not the model's.
    fn wrap_special_pair(&self, _a: &[u32], _b: &[u32]) -> Option<PairSequence> {
        None
    }

    /// `n_tokens × n_embd` hidden states, in row order.
    ///
    /// `segments` is the per-position segment id, or `None` for "all
    /// zeros" — the single-sequence case. There is deliberately ONE
    /// graph with a parameter rather than a segment-aware copy of a
    /// segment-blind one: this repo has lost a model feature to every
    /// copied forward pass it has ever had, and the pair path is
    /// exercised far less often than the embedding path, so a copy is
    /// exactly where a fix would fail to land.
    ///
    /// This is the body an encoder writes; callers want
    /// [`Self::encode`], which is the same computation with the CPU
    /// worker pool entered once for the whole pass.
    fn encode_on_worker(
        &self,
        tokens: &[u32],
        segments: Option<&[u32]>,
    ) -> Result<Vec<f32>, EncodeError>;

    /// [`Self::encode_on_worker`], with the CPU worker pool entered once
    /// for the whole pass.
    ///
    /// Same rule, and the same reason, as
    /// [`crate::engine::Engine::forward_token`]: a forward pass through
    /// a stack of quantized projections opens a parallel region per
    /// matmul, and every one of them costs a pthread park and wake when
    /// the driving thread is not a rayon worker. An encoder pass is one
    /// pass over the whole sequence rather than one per token, so the
    /// saving is smaller than a decode loop's -- but it is the same
    /// saving, and an encoder that had to remember to ask for it would
    /// be one more place for the rule to be forgotten.
    ///
    /// Do not override. See
    /// [`ferrox_core::par::on_workers`] for the three cases it declines
    /// to promote.
    fn encode(&self, tokens: &[u32], segments: Option<&[u32]>) -> Result<Vec<f32>, EncodeError> {
        ferrox_core::par::on_workers(move || self.encode_on_worker(tokens, segments))
    }

    /// [`Self::encode`] for a single sequence: every position is
    /// segment 0.
    fn encode_tokens(&self, tokens: &[u32]) -> Result<Vec<f32>, EncodeError> {
        self.encode(tokens, None)
    }

    /// [`Self::encode_tokens`] followed by the checkpoint's own pooling.
    /// Not L2-normalized — see [`crate::pooling::pool`].
    fn embed_tokens(&self, tokens: &[u32]) -> Result<Vec<f32>, EncodeError> {
        let hidden = self.encode_tokens(tokens)?;
        Ok(pool(&hidden, self.n_embd(), self.pooling_type())?)
    }
}
