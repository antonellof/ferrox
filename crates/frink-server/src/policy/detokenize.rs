//! Deciding what streamed text is safe to send now.
//!
//! A stop string may straddle a token boundary. Streaming `<` when
//! `<|end|>` is a stop string means retracting it later, which SSE
//! cannot do -- so any trailing run that is still a proper prefix of a
//! stop string is withheld until a later token decides it.
//!
//! Ported 1:1 from FreeToken's
//! `python/freetoken/tokenizer/detokenize.py` (Apache-2.0); see
//! `docs/THIRD_PARTY_NOTICES.md`.

/// Length in bytes of the longest trailing run of `text` that is a
/// *proper* prefix of some stop string.
///
/// Those bytes are withheld until a later token resolves whether the
/// stop completes, so a partial stop is never streamed and then needs
/// retracting. A run equal to a whole stop string is not held back --
/// that is a match, and the caller ends the generation instead.
///
/// Byte comparison rather than character comparison is deliberate: the
/// answer is a split point, and callers pass it through
/// [`floor_char_boundary`], so a run that lands mid-character withholds
/// slightly more rather than slicing a `str` invalidly.
///
/// This is the single implementation of the rule in the workspace --
/// `frink-server`'s `stop::StopMatcher`, which owns the same promise
/// for the non-batched decode path, delegates here rather than keeping
/// its own copy.
pub fn stop_prefix_holdback(text: &str, stop_strs: &[String]) -> usize {
    let bytes = text.as_bytes();
    let longest = stop_strs
        .iter()
        .map(|s| s.len().saturating_sub(1))
        .max()
        .unwrap_or(0);
    let max_k = longest.min(bytes.len());
    (1..=max_k)
        .rev()
        .find(|&k| {
            let tail = &bytes[bytes.len() - k..];
            stop_strs
                .iter()
                .any(|s| s.len() > k && s.as_bytes().starts_with(tail))
        })
        .unwrap_or(0)
}

/// The largest index `<= at` that is a char boundary of `text`.
///
/// `str::floor_char_boundary` is still unstable, and every split point
/// derived from a byte-length rule above needs one.
pub fn floor_char_boundary(text: &str, at: usize) -> usize {
    let mut idx = at.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holdback_covers_only_proper_prefixes() {
        let stops = vec!["<|end|>".to_string(), "STOP".to_string()];
        assert_eq!(stop_prefix_holdback("hello <|en", &stops), 4);
        assert_eq!(stop_prefix_holdback("hello ST", &stops), 2);
        assert_eq!(stop_prefix_holdback("hello", &stops), 0);
        // A whole stop string is a match, not a partial one: the caller
        // ends the generation rather than withholding it forever.
        assert_eq!(stop_prefix_holdback("hello STOP", &stops), 0);
    }

    /// The holdback is a BYTE rule, so a caller must floor it to a char
    /// boundary before slicing -- withholding slightly more rather than
    /// panicking mid-character.
    #[test]
    fn a_split_point_is_floored_to_a_char_boundary() {
        let text = "hi \u{e9}";
        assert_eq!(floor_char_boundary(text, text.len()), text.len());
        assert_eq!(floor_char_boundary(text, 4), 3);
        assert_eq!(floor_char_boundary(text, 0), 0);
        assert_eq!(floor_char_boundary(text, 99), text.len());
    }
}
