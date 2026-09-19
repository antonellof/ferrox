//! Holding the tail of a character that has not finished arriving.
//!
//! The generation loop decodes ONE token at a time, and every
//! tokenizer's `decode` ends in `String::from_utf8_lossy`. A character
//! whose UTF-8 encoding straddles a token boundary is therefore two
//! separately-invalid fragments, each of which becomes U+FFFD before
//! any caller sees a byte. A DeepSeek answer ending in an emoji printed
//! as `Hello! How can I assist you today? \u{fffd}\u{fffd}` (#124), and
//! the same thing happens to CJK text and to accented Latin coming
//! through byte-fallback tokens.
//!
//! `GgufSpmTokenizer::decode_bytes` already carries a comment about a
//! character split across several `<0xXX>` tokens WITHIN one call. This
//! is the same defect one level up: between calls.
//!
//! # Incomplete is not the same as invalid
//!
//! Nothing here guesses. [`std::str::from_utf8`] distinguishes the two
//! cases exactly, and the distinction is the whole design:
//!
//! - `error_len() == None` -- the input ended mid-character. The tail is
//!   a valid PREFIX, so it is buffered and the next token completes it.
//! - `error_len() == Some(n)` -- the bytes cannot begin a character at
//!   all. That is genuinely broken output, so it becomes U+FFFD here,
//!   where the replacement is a true statement about the model rather
//!   than an artefact of where a token boundary fell.
//!
//! [`Utf8Stream::flush`] exists because a generation can END mid
//! character (a truncated answer, a cancelled stream). What is still
//! held then is never going to be completed, so it is emitted as U+FFFD
//! rather than silently dropped: losing the bytes would make a
//! truncated answer look like a shorter finished one.

/// Accumulates decoded bytes and yields only whole characters.
#[derive(Debug, Default)]
pub(crate) struct Utf8Stream {
    /// Bytes of a character whose remaining bytes have not arrived.
    /// Never longer than 3: a UTF-8 sequence is at most 4 bytes, and a
    /// 4th byte always completes one.
    pending: Vec<u8>,
}

impl Utf8Stream {
    /// Feeds one token's raw bytes, returning the text now complete.
    ///
    /// Returns an empty string when the token added nothing but the
    /// middle of a character -- the caller emits nothing and asks again
    /// with the next token.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    out.push_str(text);
                    self.pending.clear();
                    return out;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    // SAFETY-equivalent: `valid_up_to` is exactly the
                    // length `from_utf8`already validated, so this cannot panic
                    // and cannot split a character.
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).expect("validated"));
                    match error.error_len() {
                        // Ended mid-character: keep the tail, wait.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                        // Genuinely not UTF-8: say so, and carry on with
                        // whatever follows it.
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }

    /// Whatever is still held when the generation ends.
    ///
    /// A character that was never completed cannot be recovered, so it
    /// surfaces as U+FFFD. Dropping it instead would turn a truncated
    /// answer into a shorter one that looks whole.
    pub(crate) fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        self.pending.clear();
        char::REPLACEMENT_CHARACTER.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bug in #124, as a test: a 4-byte emoji arriving as two
    /// tokens must come out as the emoji.
    #[test]
    fn a_character_split_across_two_tokens_survives() {
        let smiley = "😊".as_bytes();
        let (first, second) = smiley.split_at(2);
        let mut stream = Utf8Stream::default();
        assert_eq!(stream.push(first), "", "half a character is not text yet");
        assert_eq!(stream.push(second), "😊");
        assert_eq!(stream.flush(), "", "nothing left over");
    }

    /// Byte-fallback tokenizers emit one byte per token, so a
    /// three-byte CJK character arrives in three pieces.
    #[test]
    fn one_byte_at_a_time_still_assembles() {
        let mut stream = Utf8Stream::default();
        let mut out = String::new();
        for byte in "世界".as_bytes() {
            out.push_str(&stream.push(&[*byte]));
        }
        assert_eq!(out, "世界");
    }

    /// Text and a split character in the same token stream: the ASCII
    /// must not be held back waiting for the character behind it.
    #[test]
    fn ascii_is_emitted_immediately_and_not_held() {
        let mut stream = Utf8Stream::default();
        assert_eq!(stream.push(b"today? "), "today? ");
        assert_eq!(stream.push(&"😊".as_bytes()[..1]), "");
        assert_eq!(stream.push(&"😊".as_bytes()[1..]), "😊");
    }

    /// Bytes that cannot start a character are NOT buffered forever:
    /// they are reported as the replacement they are, and decoding
    /// continues with what follows.
    #[test]
    fn genuinely_invalid_bytes_become_one_replacement_and_do_not_stall() {
        let mut stream = Utf8Stream::default();
        assert_eq!(stream.push(&[0xff, b'o', b'k']), "\u{fffd}ok");
    }

    /// A generation cut off mid-character says so, rather than dropping
    /// the bytes and looking complete.
    #[test]
    fn a_truncated_character_surfaces_at_flush() {
        let mut stream = Utf8Stream::default();
        assert_eq!(stream.push(&"😊".as_bytes()[..3]), "");
        assert_eq!(stream.flush(), "\u{fffd}");
        assert_eq!(stream.flush(), "", "flush does not repeat itself");
    }
}
