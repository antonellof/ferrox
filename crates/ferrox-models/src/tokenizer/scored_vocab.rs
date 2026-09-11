//! The score-carrying vocabulary shared by the two SentencePiece
//! tokenizers.
//!
//! `tokenizer.ggml.tokens` and `tokenizer.ggml.scores` are two arrays
//! that must agree about one thing -- how many entries the vocabulary
//! has -- and nothing used to relate them. Each tokenizer collected
//! whatever the file happened to hold and then spelled the score lookup
//! for itself: [`super::GgufSpmTokenizer`] guarded it with
//! `.get(id).unwrap_or(0.0)`, [`super::GgufUnigramTokenizer`] indexed
//! `scores[id]` raw. A GGUF carrying 100 tokens and a 3-element scores
//! array therefore LOADED CLEANLY, was listed as a loaded model, and
//! then panicked with an index-out-of-bounds inside the generation task
//! on the first prompt whose Viterbi pass matched a piece with id >= 3
//! -- once per request, forever (issue #34). The same missing check let
//! an *empty* scores array through, whose `min_score` is `+INFINITY`,
//! so Unigram's unknown-token transition (`min_score - 10.0`) won every
//! comparison it took part in.
//!
//! Both spellings are gone. The two arrays are validated against each
//! other ONCE, here, at load: a vocabulary whose scores do not cover it
//! is a checkpoint this engine cannot run, so it is refused with both
//! lengths named rather than accepted and paid for per request.
//!
//! There is deliberately no `score(id)` method to be the next copy of
//! that lookup -- see [`ScoredVocab::lookup`].

use std::collections::HashMap;

use super::TokenizerLoadError;

/// A GGUF vocabulary and its per-token scores, held together with the
/// invariant that makes a score lookup infallible: `scores.len() ==
/// id_to_token.len()`, and `id_to_token` is not empty.
///
/// Both invariants are established in [`ScoredVocab::from_gguf`], the
/// only constructor, and the fields are private, so no other code can
/// build one that violates them.
pub(crate) struct ScoredVocab {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// One score per entry of `id_to_token`, same order, same length.
    scores: Vec<f32>,
}

impl ScoredVocab {
    /// Reads `tokenizer.ggml.tokens` + `tokenizer.ggml.scores` and
    /// checks them against each other.
    ///
    /// An absent scores key is not an error: llama.cpp's own loader
    /// treats it as optional and scores every token `0.0`, and a
    /// merge-table vocabulary converted with `tokenizer.ggml.model =
    /// "llama"` really does ship without one. A scores key that is
    /// PRESENT and does not cover the vocabulary is a different thing
    /// entirely -- the file disagrees with itself about how big its own
    /// vocabulary is, and there is no way to tell which half is right.
    pub(crate) fn from_gguf(
        file: &impl ferrox_gguf::TensorSource,
    ) -> Result<Self, TokenizerLoadError> {
        let tokens_value = file
            .metadata("tokenizer.ggml.tokens")
            .ok_or(TokenizerLoadError::MissingTokens)?;
        let id_to_token: Vec<String> = match tokens_value {
            ferrox_gguf::GgufValue::Array(items) => items
                .iter()
                .map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Option<Vec<_>>>()
                .ok_or(TokenizerLoadError::TokensNotStringArray)?,
            _ => return Err(TokenizerLoadError::TokensNotStringArray),
        };
        if id_to_token.is_empty() {
            return Err(TokenizerLoadError::EmptyVocabulary);
        }

        let scores: Vec<f32> = match file.metadata("tokenizer.ggml.scores") {
            Some(ferrox_gguf::GgufValue::Array(items)) => {
                items.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect()
            }
            _ => vec![0.0; id_to_token.len()],
        };
        if scores.len() != id_to_token.len() {
            return Err(TokenizerLoadError::ScoresVocabLengthMismatch {
                tokens: id_to_token.len(),
                scores: scores.len(),
            });
        }

        let token_to_id: HashMap<String, u32> = id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        Ok(ScoredVocab {
            id_to_token,
            token_to_id,
            scores,
        })
    }

    /// Number of vocabulary entries, which is also the number of
    /// scores.
    pub(crate) fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// The token texts in id order, for the callers that walk the
    /// vocabulary itself (`SpecialTokenTable::from_gguf` zips it against
    /// `tokenizer.ggml.token_type`; Unigram measures its longest
    /// piece).
    pub(crate) fn tokens(&self) -> &[String] {
        &self.id_to_token
    }

    /// The id of an exact vocabulary entry, for the callers that need
    /// the id and not the score.
    pub(crate) fn id_of(&self, piece: &str) -> Option<u32> {
        self.token_to_id.get(piece).copied()
    }

    /// The text of `id`, or `None` for an id this vocabulary did not
    /// issue. Decode inputs are whatever a caller passes -- including a
    /// sampled id from a model whose output head is wider than its
    /// vocabulary -- so this one stays fallible on purpose.
    pub(crate) fn token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }

    /// The id of an exact vocabulary entry AND its score, together.
    ///
    /// This is the only way to reach a score, and it is why there is no
    /// `score(id)`: an id-keyed score lookup is exactly what the two
    /// tokenizers used to spell separately, one guarded and one not.
    /// Handing the score back with the id it belongs to leaves no
    /// id-to-score step for a caller to write a second time, and the
    /// index here cannot go out of bounds -- every value in
    /// `token_to_id` is an index into `id_to_token`, and `from_gguf`
    /// refuses a file whose `scores` is not that same length.
    pub(crate) fn lookup(&self, piece: &str) -> Option<(u32, f32)> {
        let id = self.id_of(piece)?;
        Some((id, self.scores[id as usize]))
    }

    /// The lowest score in the vocabulary. Unigram derives its
    /// unknown-token penalty from this, so it matters that the fold is
    /// over a non-empty set: `from_gguf` refuses an empty vocabulary,
    /// which is the case that used to make this `+INFINITY`.
    pub(crate) fn min_score(&self) -> f32 {
        self.scores.iter().copied().fold(f32::INFINITY, f32::min)
    }
}

/// A `TensorSource` that carries metadata and no tensors, so a
/// tokenizer's loading rules can be tested against a vocabulary shaped
/// exactly like the one in a bug report without writing a GGUF file.
#[cfg(test)]
pub(crate) struct MetadataOnlyGguf {
    meta: HashMap<String, ferrox_gguf::GgufValue>,
}

#[cfg(test)]
impl MetadataOnlyGguf {
    pub(crate) fn new() -> Self {
        MetadataOnlyGguf {
            meta: HashMap::new(),
        }
    }

    pub(crate) fn with(mut self, key: &str, value: ferrox_gguf::GgufValue) -> Self {
        self.meta.insert(key.to_string(), value);
        self
    }

    pub(crate) fn with_tokens(self, tokens: &[&str]) -> Self {
        let items = tokens
            .iter()
            .map(|t| ferrox_gguf::GgufValue::String((*t).to_string()))
            .collect();
        self.with(
            "tokenizer.ggml.tokens",
            ferrox_gguf::GgufValue::Array(items),
        )
    }

    pub(crate) fn with_scores(self, scores: &[f32]) -> Self {
        let items = scores
            .iter()
            .map(|s| ferrox_gguf::GgufValue::F32(*s))
            .collect();
        self.with(
            "tokenizer.ggml.scores",
            ferrox_gguf::GgufValue::Array(items),
        )
    }
}

#[cfg(test)]
impl ferrox_gguf::TensorSource for MetadataOnlyGguf {
    fn metadata(&self, key: &str) -> Option<&ferrox_gguf::GgufValue> {
        self.meta.get(key)
    }
    fn find_tensor(&self, _name: &str) -> Option<&ferrox_gguf::TensorInfo> {
        None
    }
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], ferrox_gguf::GgufError> {
        Err(ferrox_gguf::GgufError::TensorNotFound(name.to_string()))
    }
    fn tensor_mapped_range(
        &self,
        name: &str,
    ) -> Result<
        (
            std::sync::Arc<ferrox_gguf::MmapHandle>,
            std::ops::Range<usize>,
        ),
        ferrox_gguf::GgufError,
    > {
        Err(ferrox_gguf::GgufError::TensorNotFound(name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape of issue #34's file: a vocabulary of 100 tokens and a
    /// scores array covering three of them.
    fn short_scores_file() -> MetadataOnlyGguf {
        let tokens: Vec<String> = (0..100).map(|i| format!("piece{i}")).collect();
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        MetadataOnlyGguf::new()
            .with_tokens(&refs)
            .with_scores(&[-1.0, -2.0, -3.0])
    }

    /// What shipped broken: this file loaded, and every later score
    /// lookup for an id >= 3 was an out-of-bounds index. The refusal
    /// has to name both lengths, because "the scores are wrong" is not
    /// something an operator can act on and "100 tokens, 3 scores" is.
    #[test]
    fn a_scores_array_shorter_than_the_vocabulary_is_refused_with_both_lengths_named() {
        let err = ScoredVocab::from_gguf(&short_scores_file())
            .err()
            .expect("a 100-token vocabulary with 3 scores must not load");
        assert!(
            matches!(
                err,
                TokenizerLoadError::ScoresVocabLengthMismatch {
                    tokens: 100,
                    scores: 3
                }
            ),
            "err={err:?}"
        );
        let text = err.to_string();
        assert!(text.contains("100") && text.contains('3'), "text={text}");
    }

    /// A scores array LONGER than the vocabulary is refused by the same
    /// check. It cannot panic, but the file still disagrees with itself
    /// about its own vocabulary size, and guessing which half is right
    /// is how a checkpoint gets tokenized with someone else's scores.
    #[test]
    fn a_scores_array_longer_than_the_vocabulary_is_refused_too() {
        let file = MetadataOnlyGguf::new()
            .with_tokens(&["a", "b"])
            .with_scores(&[-1.0, -2.0, -3.0]);
        assert!(matches!(
            ScoredVocab::from_gguf(&file),
            Err(TokenizerLoadError::ScoresVocabLengthMismatch {
                tokens: 2,
                scores: 3
            })
        ));
    }

    /// An absent scores key stays legal, because llama.cpp treats it as
    /// optional and real converted vocabularies omit it. It must give
    /// one zero per token, not zero scores.
    #[test]
    fn an_absent_scores_key_scores_every_token_zero_rather_than_refusing() {
        let file = MetadataOnlyGguf::new().with_tokens(&["a", "b", "c"]);
        let vocab = ScoredVocab::from_gguf(&file).expect("a scoreless vocabulary is legal");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.lookup("c"), Some((2, 0.0)));
        assert_eq!(vocab.min_score(), 0.0);
    }

    /// The degenerate half of the same missing check: with an empty
    /// vocabulary the scores fold has nothing to fold, `min_score` is
    /// `+INFINITY`, and Unigram's `min_score - 10.0` unknown-token
    /// penalty beats every real transition it is compared against.
    #[test]
    fn an_empty_vocabulary_is_refused_so_min_score_is_always_a_real_score() {
        let file = MetadataOnlyGguf::new().with_tokens(&[]).with_scores(&[]);
        assert!(matches!(
            ScoredVocab::from_gguf(&file),
            Err(TokenizerLoadError::EmptyVocabulary)
        ));
    }

    #[test]
    fn every_id_a_lookup_hands_back_indexes_its_own_score() {
        let file = MetadataOnlyGguf::new()
            .with_tokens(&["a", "bb", "ccc"])
            .with_scores(&[-1.5, -2.5, -3.5]);
        let vocab = ScoredVocab::from_gguf(&file).expect("lengths agree");
        assert_eq!(vocab.lookup("a"), Some((0, -1.5)));
        assert_eq!(vocab.lookup("bb"), Some((1, -2.5)));
        assert_eq!(vocab.lookup("ccc"), Some((2, -3.5)));
        assert_eq!(vocab.lookup("dddd"), None);
        assert_eq!(vocab.min_score(), -3.5);
        assert_eq!(vocab.token(2), Some("ccc"));
        assert_eq!(vocab.token(3), None);
        assert_eq!(vocab.id_of("bb"), Some(1));
    }
}
