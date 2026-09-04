//! How many of a completion's tokens were spent thinking.
//!
//! `usage.completion_tokens_details.reasoning_tokens` is OpenAI's own
//! field and the only part of a reasoning model's accounting that is in
//! their spec (`reasoning_content`, which this server also speaks, is
//! DeepSeek's convention). It is how a caller prices or budgets a
//! thinking model, and `/v1/responses` reported a hardcoded `0` for it
//! -- which says "this model did not think" rather than "nobody
//! counted", the exact confusion `ferrox_api::usage`'s header rules out
//! for timings.
//!
//! # Why this counts tokens instead of measuring text
//!
//! `usage`'s doc comment states the rule: counts come from the exact
//! token ids the generation loop processed, because re-tokenizing
//! decoded text is not guaranteed to round-trip to the same count. So
//! the reasoning text is NOT re-encoded. The generated ids are replayed
//! through the same parser the response body is split with, and the
//! tokens are counted.
//!
//! # Why an offset, and not "which push emitted content"
//!
//! [`ReasoningParser`] buffers: a token may emit nothing while the
//! parser holds text back against a partial marker, and a later token
//! releases all of it at once. So the push that FIRST emits content is
//! not the token content started at -- it is one or more tokens late,
//! by however much was held. Counting that way was tried and is off by
//! exactly the length of every buffered run.
//!
//! What buffering cannot move is where the answer begins in the raw
//! output. In every format this server supports the chain of thought is
//! a **prefix**: `<think>` blocks, DeepSeek's DSML variant, Gemma's
//! channels, MiniMax-M3's namespaced pair and the gpt-oss harmony
//! `analysis` channel all put reasoning first and never return to it.
//! So the content the parser produces is a SUFFIX of the raw text, its
//! start offset is `raw.len() - content.len()`, and the count is the
//! number of tokens needed to reach that offset.
//!
//! A token that straddles the boundary (its piece closes the marker and
//! opens the answer) counts as reasoning: the thinking ended inside it,
//! and the alternative is to charge the answer for a token it did not
//! begin.

use crate::policy::parser::reasoning::{ReasoningFormat, ReasoningParser};

/// Counts the tokens spent reasoning, replaying `ids` through the split.
///
/// `decode` turns one id into its piece, which is the same detokenizer
/// the generation loop used. Returns `None` when the served checkpoint
/// has no reasoning format at all, so a model that does not think
/// reports the field as absent rather than as a zero it did not earn.
pub(crate) fn count(
    format: Option<ReasoningFormat>,
    prompt_opened_reasoning: bool,
    ids: &[usize],
    mut decode: impl FnMut(&[usize]) -> String,
) -> Option<usize> {
    let format = format?;
    let pieces: Vec<String> = ids
        .iter()
        .map(|id| decode(std::slice::from_ref(id)))
        .collect();
    let raw: String = pieces.concat();

    // Untrimmed on purpose: `parse_complete` trims both sides, and a
    // trimmed length cannot be subtracted from a raw one to get an
    // offset into that raw text.
    let mut parser = ReasoningParser::new(format, prompt_opened_reasoning, true);
    let head = parser.push(&raw);
    let tail = parser.flush();
    let content_len = head.content.len() + tail.content.len();

    // Where the answer starts. Saturating because a format that
    // rewrites rather than merely strips (harmony channels) can emit
    // more content than the raw text held, and a wrapped subtraction
    // here would charge the whole generation to reasoning.
    let boundary = raw.len().saturating_sub(content_len);

    let mut consumed = 0usize;
    for (index, piece) in pieces.iter().enumerate() {
        if consumed >= boundary {
            return Some(index);
        }
        consumed += piece.len();
    }
    Some(pieces.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pieces stand in for a tokenizer: each "id" indexes this table,
    /// so a test can place a marker on its own token or bury it inside
    /// one.
    fn decoder(pieces: &'static [&'static str]) -> impl FnMut(&[usize]) -> String {
        move |ids: &[usize]| ids.iter().map(|i| pieces[*i]).collect()
    }

    fn count_pieces(pieces: &'static [&'static str]) -> Option<usize> {
        let ids: Vec<usize> = (0..pieces.len()).collect();
        count(
            Some(ReasoningFormat::Think),
            false,
            &ids,
            decoder(pieces),
        )
    }

    #[test]
    fn thinking_is_charged_to_reasoning_and_the_answer_is_not() {
        // <think> a b </think> answer  ->  the four tokens through the
        // closing marker are reasoning, "answer" is not.
        let n = count_pieces(&["<think>", "a", "b", "</think>", "answer"]);
        assert_eq!(n, Some(4));
    }

    /// The regression the prefix rule exists for: a parser that buffers
    /// against a partial marker emits nothing for several tokens, and
    /// counting only the pushes that produced reasoning text would
    /// score those as answer tokens.
    #[test]
    fn a_buffered_run_is_still_reasoning() {
        let n = count_pieces(&["<think>", "a", "<", "/th", "ink>", "answer"]);
        assert_eq!(n, Some(5), "the split marker's tokens are reasoning");
    }

    #[test]
    fn a_model_that_never_thinks_spends_no_reasoning_tokens() {
        assert_eq!(count_pieces(&["hello", " world"]), Some(0));
    }

    /// A checkpoint whose family emits no reasoning must report the
    /// field as ABSENT. Zero is a claim about the model; `None` is the
    /// truth about this build.
    #[test]
    fn no_format_means_no_number_rather_than_zero() {
        let ids = [0usize, 1];
        assert_eq!(count(None, false, &ids, decoder(&["a", "b"])), None);
    }

    /// A generation that is still inside its thinking when it runs out
    /// of budget spent ALL of its tokens reasoning -- this is the case
    /// that rendered as an empty message in #118.
    #[test]
    fn an_answer_that_never_stopped_thinking_is_all_reasoning() {
        let n = count_pieces(&["<think>", "a", "b", "c"]);
        assert_eq!(n, Some(4));
    }
}
