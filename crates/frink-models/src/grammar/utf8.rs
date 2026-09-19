//! UTF-8 decoding for the grammar engine, transcribed from the two
//! `decode_utf8` overloads in llama.cpp's `src/llama-grammar.cpp`.
//!
//! Both are deliberately *not* `str::chars()`. Two reasons:
//!
//! 1. A token piece can end mid-codepoint. Multi-byte characters are split
//!    across tokens by every BPE vocabulary, so the decoder has to carry a
//!    [`PartialUtf8`] between pieces and the constraint has to be able to
//!    ask "could some continuation of these bits still satisfy this
//!    character class?" ([`super::machine`]'s `match_partial_char`).
//! 2. The lead-byte length tables here are llama.cpp's, including their
//!    quirks: the parser's table maps a continuation byte to length 1 and
//!    decodes it as a 7-bit character, while the piece decoder maps the
//!    same byte to length 0 and reports an invalid sequence. Replacing
//!    either with a correct UTF-8 decoder would change which grammars
//!    parse and which tokens are rejected.

/// Bits of a UTF-8 sequence decoded so far, carried between token pieces.
///
/// `llama_partial_utf8`. `n_remain` is the number of continuation bytes
/// still expected: `0` means "nothing pending", `-1` means "the byte
/// stream was not valid UTF-8".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PartialUtf8 {
    /// Bit value so far, unshifted.
    pub value: u32,
    /// Continuation bytes still expected; `-1` marks an invalid sequence.
    pub n_remain: i32,
}

impl PartialUtf8 {
    pub const fn new(value: u32, n_remain: i32) -> Self {
        Self { value, n_remain }
    }
}

/// llama.cpp's lead-byte length table for [`decode_char`], indexed by the
/// top four bits. Note indices 8..=11 (continuation bytes) map to 1, not 0:
/// the parser assumes it is handed valid UTF-8 and only guards overrun.
const CHAR_LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];

/// llama.cpp's lead-byte length table for [`decode_piece`]. Here indices
/// 8..=11 map to 0, which becomes `n_remain = -1` and aborts the decode:
/// a token piece is untrusted input and a stray continuation byte is a
/// real error.
const PIECE_LOOKUP: [i32; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4];

/// Byte at `i`, or `0` past the end.
///
/// llama.cpp walks NUL-terminated `const char *`, so every one of its loop
/// guards is `*pos != 0`. Returning `0` past the end reproduces that
/// exactly, including for an embedded NUL byte, which upstream also treats
/// as end of input.
#[inline]
pub(crate) fn byte_at(src: &[u8], i: usize) -> u8 {
    if i < src.len() {
        src[i]
    } else {
        0
    }
}

/// Decode one code point starting at `pos`, returning it and the offset
/// just past it.
///
/// `static std::pair<uint32_t, const char *> decode_utf8(const char * src)`.
/// Assumes valid UTF-8 but does not run off the end of the buffer.
pub(crate) fn decode_char(src: &[u8], pos: usize) -> (u32, usize) {
    let first_byte = byte_at(src, pos);
    let highbits = (first_byte >> 4) as usize;
    let len = CHAR_LOOKUP[highbits];
    // `(1 << (8 - len)) - 1` keeps one more bit than the strict UTF-8 mask,
    // but that bit is always 0 in a well-formed lead byte of this length.
    let mask = (1u32 << (8 - len)) - 1;
    let mut value = first_byte as u32 & mask;
    let end = pos + len; // may overrun the buffer; the guard below stops us
    let mut p = pos + 1;
    while p < end && byte_at(src, p) != 0 {
        value = (value << 6) + (byte_at(src, p) & 0x3F) as u32;
        p += 1;
    }
    (value, p)
}

/// Decode a token piece into code points, continuing an earlier partial
/// sequence and reporting whatever partial sequence is left over.
///
/// `static std::pair<std::vector<uint32_t>, llama_partial_utf8>
/// decode_utf8(const std::string &, llama_partial_utf8)`.
///
/// The returned vector is **always terminated by a `0`**, which callers
/// rely on: `reject_candidates_for_stack` uses `*code_points == 0` to mean
/// "this token's complete code points are exhausted".
pub(crate) fn decode_piece(src: &[u8], partial_start: PartialUtf8) -> (Vec<u32>, PartialUtf8) {
    let mut pos = 0usize;
    // Common English pieces have as many code points as bytes; `+1` for the
    // terminating 0.
    let mut code_points: Vec<u32> = Vec::with_capacity(src.len() + 1);

    let mut value = partial_start.value;
    let mut n_remain = partial_start.n_remain;

    // Continue the previous decode, if applicable.
    while byte_at(src, pos) != 0 && n_remain > 0 {
        let next_byte = byte_at(src, pos);
        if (next_byte >> 6) != 2 {
            // Not a continuation byte: invalid sequence, abort.
            code_points.push(0);
            return (code_points, PartialUtf8::new(0, -1));
        }
        value = (value << 6) + (next_byte & 0x3F) as u32;
        pos += 1;
        n_remain -= 1;
    }

    if partial_start.n_remain > 0 && n_remain == 0 {
        code_points.push(value);
    }

    // Decode subsequent sequences, the last of which may be incomplete.
    while byte_at(src, pos) != 0 {
        let first_byte = byte_at(src, pos);
        let highbits = (first_byte >> 4) as usize;
        n_remain = PIECE_LOOKUP[highbits] - 1;

        if n_remain < 0 {
            // Invalid sequence, abort. Upstream drops the code points it
            // had already decoded here; keeping them would let a prefix of
            // a malformed piece advance the parse.
            code_points.clear();
            code_points.push(0);
            return (code_points, PartialUtf8::new(0, n_remain));
        }

        let mask = (1u32 << (7 - n_remain)) - 1;
        value = first_byte as u32 & mask;

        pos += 1;
        while byte_at(src, pos) != 0 && n_remain > 0 {
            value = (value << 6) + (byte_at(src, pos) & 0x3F) as u32;
            pos += 1;
            n_remain -= 1;
        }
        if n_remain == 0 {
            code_points.push(value);
        }
    }
    code_points.push(0);

    (code_points, PartialUtf8::new(value, n_remain))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_piece_decodes_to_its_bytes_plus_a_terminator() {
        let (v, p) = decode_piece(b"abc", PartialUtf8::default());
        assert_eq!(v, vec![0x61, 0x62, 0x63, 0]);
        assert_eq!(p.n_remain, 0);
    }

    #[test]
    fn empty_piece_is_just_the_terminator() {
        let (v, p) = decode_piece(b"", PartialUtf8::default());
        assert_eq!(v, vec![0]);
        assert_eq!(p, PartialUtf8::default());
    }

    #[test]
    fn multibyte_piece_decodes_whole() {
        // "é" U+00E9 (2 bytes), "€" U+20AC (3), "𝄞" U+1D11E (4).
        let (v, p) = decode_piece("é€𝄞".as_bytes(), PartialUtf8::default());
        assert_eq!(v, vec![0xE9, 0x20AC, 0x1D11E, 0]);
        assert_eq!(p.n_remain, 0);
    }

    #[test]
    fn a_codepoint_split_across_two_pieces_is_carried_and_completed() {
        // "€" is E2 82 AC. Split it after the first byte, the way a BPE
        // vocabulary splits a multi-byte character across two tokens.
        let euro = "€".as_bytes();
        let (v1, p1) = decode_piece(&euro[..1], PartialUtf8::default());
        assert_eq!(v1, vec![0], "no complete code point yet");
        assert_eq!(p1.n_remain, 2, "two continuation bytes still expected");

        let (v2, p2) = decode_piece(&euro[1..2], p1);
        assert_eq!(v2, vec![0], "still incomplete after one continuation byte");
        assert_eq!(p2.n_remain, 1);

        let (v3, p3) = decode_piece(&euro[2..], p2);
        assert_eq!(v3, vec![0x20AC, 0], "completes on the last byte");
        assert_eq!(p3.n_remain, 0);
    }

    #[test]
    fn a_split_codepoint_followed_by_more_text_decodes_both() {
        let bytes = "€!".as_bytes();
        let (_, p1) = decode_piece(&bytes[..1], PartialUtf8::default());
        let (v, p) = decode_piece(&bytes[1..], p1);
        assert_eq!(v, vec![0x20AC, b'!' as u32, 0]);
        assert_eq!(p.n_remain, 0);
    }

    #[test]
    fn a_lead_byte_where_a_continuation_was_due_is_an_invalid_sequence() {
        let p1 = PartialUtf8::new(0x02, 1); // mid "é"
        let (v, p) = decode_piece(b"A", p1);
        assert_eq!(v, vec![0]);
        assert_eq!(p, PartialUtf8::new(0, -1), "n_remain = -1 marks invalid");
    }

    #[test]
    fn a_stray_continuation_byte_clears_everything_already_decoded() {
        // Upstream calls code_points.clear() here: the leading "ab" is
        // thrown away, not returned. A decoder that kept it would let a
        // malformed piece advance the parse by two characters.
        let (v, p) = decode_piece(&[b'a', b'b', 0x80], PartialUtf8::default());
        assert_eq!(v, vec![0]);
        assert_eq!(p.n_remain, -1);
    }

    #[test]
    fn a_truncated_lead_byte_leaves_a_partial_not_an_error() {
        let (v, p) = decode_piece(&[0xE2], PartialUtf8::default());
        assert_eq!(v, vec![0]);
        assert_eq!(p, PartialUtf8::new(0x02, 2));
    }

    #[test]
    fn an_embedded_nul_ends_the_piece_as_it_does_in_c() {
        let (v, _) = decode_piece(b"ab\0cd", PartialUtf8::default());
        assert_eq!(v, vec![b'a' as u32, b'b' as u32, 0]);
    }

    #[test]
    fn decode_char_reads_one_codepoint_and_reports_its_width() {
        assert_eq!(decode_char(b"a", 0), (0x61, 1));
        assert_eq!(decode_char("é".as_bytes(), 0), (0xE9, 2));
        assert_eq!(decode_char("€".as_bytes(), 0), (0x20AC, 3));
        assert_eq!(decode_char("𝄞".as_bytes(), 0), (0x1D11E, 4));
        // Offset into the middle of a string.
        assert_eq!(decode_char("aé".as_bytes(), 1), (0xE9, 3));
    }

    #[test]
    fn decode_char_does_not_run_past_a_truncated_sequence() {
        // Lead byte claims 3 bytes, only 1 is present. llama.cpp's loop
        // guard is `pos < end && *pos`; both must hold.
        let (value, end) = decode_char(&[0xE2], 0);
        assert_eq!(end, 1, "stopped at the buffer end, not at pos+3");
        assert_eq!(value, 0x02);
    }

    #[test]
    fn the_two_lookup_tables_disagree_on_continuation_bytes_and_that_is_deliberate() {
        // Parser table: 0x80 >> 4 == 8 -> length 1, decoded as a character.
        assert_eq!(CHAR_LOOKUP[8], 1);
        assert_eq!(decode_char(&[0x80], 0), (0x00, 1));
        // Piece table: same byte -> length 0 -> n_remain -1 -> invalid.
        assert_eq!(PIECE_LOOKUP[8], 0);
    }
}
