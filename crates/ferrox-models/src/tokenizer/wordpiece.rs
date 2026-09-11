//! WordPiece, for GGUF files whose `tokenizer.ggml.model` is `bert`.
//!
//! A transcription of llama.cpp's `LLAMA_VOCAB_TYPE_WPM` path:
//! `llm_tokenizer_wpm_session::tokenize` and its `preprocess`, in
//! `.scratch/llama.cpp/src/llama-vocab.cpp`. Its own module rather than
//! a fourth section of [`super`], which is already past two thousand
//! lines, and because everything here is reviewed *against that file*.
//!
//! # What a WordPiece GGUF actually holds
//!
//! Not what the HuggingFace `vocab.txt` holds. llama.cpp's converter
//! (`conversion/bert.py`) rewrites every vocabulary entry on the way
//! into the GGUF, and the tokenizer below only makes sense against the
//! rewritten form:
//!
//! * a continuation piece `##ing` is stored as `ing`, with the `##`
//!   stripped;
//! * a word-initial piece `hello` is stored as `▁hello` (U+2581);
//! * a `CONTROL` entry such as `[CLS]` is stored unchanged.
//!
//! So the `##` prefix does not appear anywhere in this file. Its job,
//! marking "this piece may only continue a word", is done by the
//! *absence* of the `▁` that only the first lookup of each word can
//! match. That is why the loop below prepends `▁` to the word once and
//! then walks straight through: position 0 can only match a
//! word-initial piece, and every later position can only match a
//! continuation.
//!
//! # The algorithm, in the order it runs
//!
//! 1. **Normalize and split into words** ([`preprocess`]). This is
//!    BertNormalizer plus BertPreTokenizer, not the BPE pre-tokenizer
//!    regex next door in [`super::pretokenize`]: NFD-fold and drop
//!    accents, drop controls, lowercase, break on whitespace, and give
//!    every punctuation, ASCII symbol and CJK character a word of its
//!    own.
//! 2. **Greedy longest-match-first** per word, from the left, over the
//!    `▁`-prefixed word.
//! 3. **All-or-nothing fallback.** If any position in a word has no
//!    match at any length, every piece already emitted for that word is
//!    discarded and the word becomes a single unknown token. WordPiece
//!    does not fall back per character, and it does not fall back to
//!    bytes: a word is either fully covered or it is `[UNK]`.
//!
//! # Bytes, not characters
//!
//! The match loop indexes **bytes**, because llama.cpp's does
//! (`word1.substr(i, j - i)` over a `std::string`) and because a
//! vocabulary piece is free to hold one byte of a multi-byte character.
//! Cutting on `char` boundaries instead would silently skip candidate
//! lengths and split CJK differently. The lookup table is therefore
//! keyed by byte string; a slice that is not valid UTF-8 simply matches
//! nothing, exactly as it matches nothing upstream.

use super::unicode;
use super::{SpecialTokenTable, SpecialTokens, TextOrSpecial, TokenizerLoadError};
use std::collections::HashMap;

/// The phantom space llama.cpp's converter puts in front of every
/// word-initial piece, and that [`GgufWordPieceTokenizer::encode`] puts
/// in front of every word before matching.
const PHANTOM_SPACE: &str = "\u{2581}";

/// llama.cpp's `bert` defaults for `special_unk_id`. Applied when the
/// GGUF carries no `tokenizer.ggml.unknown_token_id`, which is what
/// upstream does: it seeds the id from the `tokenizer_model == "bert"`
/// arm and only then lets metadata override it.
const DEFAULT_UNK_ID: u32 = 100;

/// The `BertNormalizer` switches, and the defaults llama.cpp applies
/// when a checkpoint does not carry them.
///
/// Both default to **true**, and `strip_accents` defaults to whatever
/// `lowercase` resolved to rather than to a constant. That chain is
/// upstream's, verbatim: `normalizer_opts.lowercase` is read first,
/// `strip_accents` is then seeded from it, and only then is
/// `tokenizer.ggml.normalizer.strip_accents` allowed to override. The
/// GGUFs people actually have predate both keys, so the defaults are
/// the live path, not the fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizerOptions {
    pub lowercase: bool,
    pub strip_accents: bool,
}

impl NormalizerOptions {
    fn from_gguf(file: &impl ferrox_gguf::TensorSource) -> Self {
        let lowercase = file
            .metadata_bool("tokenizer.ggml.normalizer.lowercase")
            .unwrap_or(true);
        let strip_accents = file
            .metadata_bool("tokenizer.ggml.normalizer.strip_accents")
            .unwrap_or(lowercase);
        NormalizerOptions {
            lowercase,
            strip_accents,
        }
    }
}

/// A real WordPiece tokenizer built from a GGUF file's own
/// `tokenizer.ggml.tokens` metadata array.
///
/// See the module docs for the algorithm and for why the vocabulary
/// looks the way it does. Checked against llama.cpp on the same GGUF by
/// `crates/ferrox-models/tests/wordpiece_parity.rs`.
pub struct GgufWordPieceTokenizer {
    /// Keyed by the piece's **bytes**. See the module docs.
    token_to_id: HashMap<Vec<u8>, u32>,
    id_to_token: Vec<String>,
    /// The longest piece in the vocabulary, in bytes. Upstream's
    /// `max_token_len`, and the reason the match loop is linear rather
    /// than quadratic in the length of the word.
    max_token_len: usize,
    unk_id: u32,
    normalizer: NormalizerOptions,
    /// The vocabulary's special entries, carved out of raw text before
    /// normalization runs. For a BERT vocabulary this is `[PAD]`,
    /// `[UNK]`, `[CLS]`, `[SEP]` and `[MASK]`, which is exactly
    /// llama.cpp's `cache_special_tokens` for the same file.
    special_tokens: SpecialTokenTable,
}

impl GgufWordPieceTokenizer {
    pub fn from_gguf(file: &impl ferrox_gguf::TensorSource) -> Result<Self, TokenizerLoadError> {
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

        // First id wins on a duplicate piece, matching upstream's
        // `token_to_id[word] = i` only ever being written for a word it
        // has not seen (a GGUF with a repeated piece is malformed, but
        // it must not decide the answer by hash order).
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(id_to_token.len());
        for (i, text) in id_to_token.iter().enumerate() {
            token_to_id
                .entry(text.as_bytes().to_vec())
                .or_insert(i as u32);
        }
        let max_token_len = id_to_token.iter().map(|t| t.len()).max().unwrap_or(0);

        let unk_id = file
            .metadata("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(DEFAULT_UNK_ID);

        let special_tokens = SpecialTokenTable::from_gguf(file, &id_to_token);

        Ok(GgufWordPieceTokenizer {
            token_to_id,
            id_to_token,
            max_token_len,
            unk_id,
            normalizer: NormalizerOptions::from_gguf(file),
            special_tokens,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn normalizer(&self) -> NormalizerOptions {
        self.normalizer
    }

    /// Encodes `text`, with no `[CLS]`/`[SEP]` added.
    ///
    /// Upstream's `add_special` wraps the result in BOS and SEP; that
    /// decision lives with [`super::should_add_bos_token`] and
    /// [`super::prepend_bos`] for every tokenizer in this crate, and
    /// baking it in here would double it for callers that already do it.
    /// `specials` is llama.cpp's `parse_special`: whether a literal
    /// `[SEP]` in the text is the separator or five characters.
    pub fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<u32> {
        let mut out = Vec::new();
        for seg in self.special_tokens.split(text, specials) {
            match seg {
                TextOrSpecial::Special(id) => out.push(id),
                TextOrSpecial::Text(t) => self.encode_normal_run(t, &mut out),
            }
        }
        out
    }

    fn encode_normal_run(&self, text: &str, out: &mut Vec<u32>) {
        for word in preprocess(text, self.normalizer) {
            if word.is_empty() {
                continue;
            }
            let word1 = format!("{PHANTOM_SPACE}{word}");
            let bytes = word1.as_bytes();
            let n = bytes.len();
            let start = out.len();

            let mut i = 0usize;
            while i < n {
                // Longest match first, capped at the longest piece the
                // vocabulary holds. The `+ 1` is upstream's and is
                // deliberately kept: it lets the first probe be one byte
                // longer than any piece, which can never match and costs
                // one lookup.
                let mut j = n.min(i + self.max_token_len + 1);
                let mut matched = false;
                while j > i {
                    if let Some(&id) = self.token_to_id.get(&bytes[i..j]) {
                        out.push(id);
                        i = j;
                        matched = true;
                        break;
                    }
                    j -= 1;
                }
                if !matched {
                    // All or nothing: discard the pieces already emitted
                    // for THIS word and stop. The `[UNK]` below covers
                    // the whole word.
                    out.truncate(start);
                    break;
                }
            }

            if out.len() == start {
                out.push(self.unk_id);
            }
        }
    }

    /// The text a token id stands for.
    ///
    /// Reverses the converter's phantom space, so `▁hello` decodes to
    /// `" hello"` and the continuation `ing` decodes to `"ing"`.
    /// Concatenating a sequence therefore reproduces the normalized
    /// text with a leading space, which is what llama.cpp's
    /// `llama_unescape_whitespace` produces before its `clean_spaces`
    /// pass trims the first one.
    ///
    /// A WordPiece round trip is lossy no matter what this does: the
    /// normalizer lowercased, stripped accents and dropped controls
    /// before any of these ids existed. Round-tripping is therefore NOT
    /// evidence that this tokenizer is right, which is why the tests
    /// that matter compare ids against llama.cpp instead.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(id as usize) {
                out.push_str(&token.replace(PHANTOM_SPACE, " "));
            }
        }
        out
    }
}

/// The codepoint ranges llama.cpp counts as Chinese, and therefore
/// splits into single-character words.
///
/// Verbatim from `llm_tokenizer_wpm_session::is_chinese_char`, including
/// the range that upstream's own comment flags as wrong (`0x2B920`
/// should be `0x2B820`; the HuggingFace Rust implementation has the same
/// value, so matching it is the point) and including the two ranges it
/// leaves commented out. Widening this set would split CJK punctuation
/// differently from the reference.
fn is_chinese_char(c: char) -> bool {
    let cpt = c as u32;
    (0x04E00..=0x09FFF).contains(&cpt)
        || (0x03400..=0x04DBF).contains(&cpt)
        || (0x20000..=0x2A6DF).contains(&cpt)
        || (0x2A700..=0x2B73F).contains(&cpt)
        || (0x2B740..=0x2B81F).contains(&cpt)
        || (0x2B920..=0x2CEAF).contains(&cpt)
        || (0x0F900..=0x0FAFF).contains(&cpt)
        || (0x2F800..=0x2FA1F).contains(&cpt)
}

/// BertNormalizer + BertPreTokenizer: normalize `text` and cut it into
/// words.
///
/// A transcription of `llm_tokenizer_wpm_session::preprocess`. The order
/// of the tests is load-bearing and is upstream's:
///
/// 1. whitespace ends the current word and is otherwise dropped;
/// 2. NUL, U+FFFD and `\p{C}` are dropped, so a soft hyphen or a
///    zero-width space joins the characters on either side of it into
///    one word rather than breaking it;
/// 3. an accent mark is dropped when `strip_accents`;
/// 4. the character is lowercased when `lowercase`;
/// 5. punctuation, an **ASCII** symbol and a CJK character each become a
///    word of their own;
/// 6. anything else extends the current word.
///
/// Two details that a plausible re-derivation gets wrong. The accent
/// fold in step 3 runs over the NFD-folded text, so `é` has already
/// become `e` and there is no mark left to drop; the test exists for
/// marks that were standalone in the input, like the `e` + U+0301 in the
/// parity corpus. And step 5 tests `is_symbol` only below U+007F, so
/// `+` starts a new word and `€` does not.
///
/// Returns owned `String`s because steps 3 and 4 mean a word is not a
/// slice of the input.
fn preprocess(text: &str, opts: NormalizerOptions) -> Vec<String> {
    let mut words: Vec<String> = vec![String::new()];

    for raw in text.chars() {
        let c = if opts.strip_accents {
            unicode::nfd_base(raw)
        } else {
            raw
        };

        if unicode::is_whitespace(c) {
            if !words.last().is_some_and(String::is_empty) {
                words.push(String::new());
            }
            continue;
        }

        let flags = unicode::flags(c);
        debug_assert!(
            !flags.is_separator(),
            "every \\p{{Z}} codepoint is also White_Space, so the check above \
             should have consumed it: {c:?}"
        );

        if c == '\0' || c == '\u{fffd}' || flags.is_control() {
            continue;
        }
        if opts.strip_accents && flags.is_accent_mark() {
            continue;
        }

        let c = if opts.lowercase {
            unicode::to_lower(c)
        } else {
            c
        };

        if flags.is_punctuation() || ((c as u32) < 0x7F && flags.is_symbol()) || is_chinese_char(c)
        {
            if !words.last().is_some_and(String::is_empty) {
                words.push(String::new());
            }
            // A word of exactly this character, then a fresh word for
            // whatever follows.
            words.last_mut().expect("just pushed or non-empty").push(c);
            words.push(String::new());
        } else {
            words.last_mut().expect("seeded with one word").push(c);
        }
    }

    if words.last().is_some_and(String::is_empty) {
        words.pop();
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::SpecialKind;

    const BOTH: NormalizerOptions = NormalizerOptions {
        lowercase: true,
        strip_accents: true,
    };
    const NEITHER: NormalizerOptions = NormalizerOptions {
        lowercase: false,
        strip_accents: false,
    };

    fn words(text: &str, opts: NormalizerOptions) -> Vec<String> {
        preprocess(text, opts)
    }

    #[test]
    fn whitespace_splits_words_and_is_dropped() {
        assert_eq!(words("hello world", BOTH), ["hello", "world"]);
        assert_eq!(words("  hello   world  ", BOTH), ["hello", "world"]);
        assert_eq!(words("", BOTH), Vec::<String>::new());
        assert_eq!(words("   ", BOTH), Vec::<String>::new());
    }

    /// The parity corpus's `unicode-space` case, which is the reason
    /// this cannot be `char::is_ascii_whitespace`: NBSP and the
    /// ideographic space break words, and ZWSP does not because it is
    /// `Cf` and gets dropped instead.
    #[test]
    fn unicode_spaces_break_words_but_zero_width_ones_vanish() {
        assert_eq!(
            words("a\u{a0}b\u{3000}c\u{2009}d\u{200b}e", BOTH),
            ["a", "b", "c", "de"]
        );
    }

    #[test]
    fn punctuation_and_ascii_symbols_become_single_character_words() {
        assert_eq!(words("don't.", BOTH), ["don", "'", "t", "."]);
        assert_eq!(words("a+b", BOTH), ["a", "+", "b"]);
        assert_eq!(words("1+2=3", BOTH), ["1", "+", "2", "=", "3"]);
    }

    /// `is_symbol` is consulted only below U+007F. A non-ASCII symbol
    /// therefore stays welded to its neighbours, which is the opposite
    /// of what "split on symbols" would do.
    #[test]
    fn non_ascii_symbols_do_not_split() {
        assert_eq!(words("a€b", BOTH), ["a€b"]);
        assert_eq!(words("a$b", BOTH), ["a", "$", "b"]);
    }

    #[test]
    fn cjk_characters_each_get_their_own_word() {
        assert_eq!(words("日本語", BOTH), ["日", "本", "語"]);
        assert_eq!(words("a日b", BOTH), ["a", "日", "b"]);
    }

    /// Hangul is not in the Chinese ranges, so it is never split into
    /// per-character words. What happens to it instead is worse and is
    /// the reference's behaviour, not a defect here: every Hangul
    /// syllable has an NFD decomposition into jamo, so the accent fold
    /// replaces it with its LEADING jamo and the rest of the syllable is
    /// gone. `서울` becomes `ᄉᄋ`, one word, and a `[UNK]` in practice.
    ///
    /// Asserted rather than left implicit because it is exactly the kind
    /// of behaviour a future "obvious fix" would break parity by
    /// improving. The parity corpus's `cjk` case carries `서울` and the
    /// oracle agrees with this.
    #[test]
    fn hangul_is_folded_to_its_leading_jamo_by_the_accent_pass() {
        assert_eq!(words("서울", BOTH), ["\u{1109}\u{110b}"]);
        assert_eq!(words("서울", NEITHER), ["서울"], "only the fold does this");
    }

    #[test]
    fn controls_are_dropped_without_breaking_the_word() {
        assert_eq!(words("a\u{1b}b", BOTH), ["ab"], "ESC is Cc");
        assert_eq!(words("a\u{ad}b", BOTH), ["ab"], "soft hyphen is Cf");
        assert_eq!(words("a\u{fffd}b", BOTH), ["ab"], "the replacement char");
        assert_eq!(words("a\0b", BOTH), ["ab"], "NUL");
    }

    #[test]
    fn lowercase_and_accent_stripping_follow_their_flags() {
        assert_eq!(words("Café", BOTH), ["cafe"]);
        assert_eq!(words("Café", NEITHER), ["Café"]);
        assert_eq!(
            words(
                "Café",
                NormalizerOptions {
                    lowercase: true,
                    strip_accents: false
                }
            ),
            ["café"]
        );
        assert_eq!(
            words(
                "Café",
                NormalizerOptions {
                    lowercase: false,
                    strip_accents: true
                }
            ),
            ["Cafe"]
        );
    }

    /// The corpus writes `café` precomposed and `e\u{301}clair`
    /// decomposed on purpose. Both must land on the same bytes, and they
    /// do so by two different routes: the NFD fold for the first, the
    /// `is_accent_mark` drop for the second.
    #[test]
    fn precomposed_and_decomposed_accents_normalize_alike() {
        assert_eq!(words("café", BOTH), words("cafe\u{301}", BOTH));
        assert_eq!(words("cafe\u{301}", BOTH), ["cafe"]);
    }

    /// A vocabulary in the form llama.cpp's converter writes: `▁` on
    /// word-initial pieces, bare continuations, `[...]` specials.
    fn toy() -> GgufWordPieceTokenizer {
        let pieces = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", // 0..=3
            "▁un", "▁hello", "▁.", "▁world", // 4..=7
            "aff", "able", "ing",    // 8..=10
            "▁unaff", // 11, a word-initial piece that EXTENDS ▁un
        ];
        let id_to_token: Vec<String> = pieces.iter().map(|s| s.to_string()).collect();
        let mut token_to_id = HashMap::new();
        for (i, t) in id_to_token.iter().enumerate() {
            token_to_id.entry(t.as_bytes().to_vec()).or_insert(i as u32);
        }
        let max_token_len = id_to_token.iter().map(|t| t.len()).max().unwrap();
        GgufWordPieceTokenizer {
            token_to_id,
            id_to_token,
            max_token_len,
            unk_id: 1,
            normalizer: BOTH,
            special_tokens: SpecialTokenTable::from_entries([
                ("[CLS]", 2, SpecialKind::Control),
                ("[SEP]", 3, SpecialKind::Control),
            ]),
        }
    }

    #[test]
    fn a_word_is_covered_word_initial_piece_first_then_continuations() {
        let t = toy();
        // ▁un + able, which only works because `able` carries no phantom
        // space and so cannot match at position 0.
        assert_eq!(t.encode("unable", SpecialTokens::AsText), [4, 9]);
        assert_eq!(t.encode("hello", SpecialTokens::AsText), [5]);
    }

    /// Longest match first, not merely *a* match.
    ///
    /// `unaffable` has two full covers in this vocabulary: `▁unaff` +
    /// `able`, and `▁un` + `aff` + `able`. Both reach the end of the
    /// word, so the all-or-nothing rule does not choose between them and
    /// a shortest-first loop would be just as "correct" while producing
    /// different ids for every real checkpoint. Only the probe order
    /// decides, which is why it is asserted on its own.
    #[test]
    fn the_longest_match_wins_where_a_shorter_one_would_also_cover() {
        let t = toy();
        assert_eq!(
            t.encode("unaffable", SpecialTokens::AsText),
            [11, 9],
            "▁unaff + able"
        );
        assert_ne!(
            t.encode("unaffable", SpecialTokens::AsText),
            [4, 8, 9],
            "▁un + aff + able is the shortest-first answer"
        );
    }

    /// The continuation pieces must not be reachable at the start of a
    /// word. `aff` alone has no `▁aff` entry, so it is unknown even
    /// though its bytes are in the vocabulary. This is the `##` rule,
    /// expressed the way a GGUF expresses it.
    #[test]
    fn a_continuation_piece_cannot_start_a_word() {
        let t = toy();
        assert_eq!(
            t.encode("aff", SpecialTokens::AsText),
            [1],
            "no ▁aff, so [UNK]"
        );
    }

    /// The whole point of the all-or-nothing rule. `worlds` matches
    /// `▁world` at position 0 and then has nothing for the trailing `s`,
    /// so the `▁world` already emitted is thrown away and the word
    /// becomes ONE unknown, not `▁world` plus an unknown.
    #[test]
    fn a_partly_covered_word_discards_its_pieces_and_becomes_one_unknown() {
        let t = toy();
        assert_eq!(
            t.encode("worlds", SpecialTokens::AsText),
            [1],
            "a partial cover must be discarded, not kept"
        );
        // The same word without the stray byte does cover, which is what
        // makes the assertion above about the fallback rule rather than
        // about the vocabulary being too small.
        assert_eq!(t.encode("world", SpecialTokens::AsText), [7]);
    }

    #[test]
    fn each_word_falls_back_independently() {
        let t = toy();
        assert_eq!(
            t.encode("hello zzz world", SpecialTokens::AsText),
            [5, 1, 7]
        );
    }

    #[test]
    fn punctuation_is_its_own_word_and_matches_its_own_piece() {
        let t = toy();
        assert_eq!(
            t.encode("hello.", SpecialTokens::AsText),
            [5, 6],
            "▁hello then ▁."
        );
    }

    #[test]
    fn special_tokens_are_carved_out_before_normalization() {
        let t = toy();
        // Without the carve-out the brackets would each be their own
        // punctuation word and `[CLS]` would come back as four unknowns.
        assert_eq!(t.encode("[CLS]hello[SEP]", SpecialTokens::Parse), [2, 5, 3]);
    }

    /// llama.cpp's default: a `[SEP]` written in the text is text. The
    /// toy vocabulary has no `[`, so the marker shatters to unknowns
    /// around `hello`, which is what upstream does with
    /// `parse_special = false`.
    #[test]
    fn as_text_leaves_a_written_marker_to_the_normal_pass() {
        let t = toy();
        let ids = t.encode("[CLS]hello[SEP]", SpecialTokens::AsText);
        assert!(!ids.contains(&2) && !ids.contains(&3), "got {ids:?}");
        assert!(ids.contains(&5), "hello survives: {ids:?}");
    }

    #[test]
    fn decode_reverses_the_phantom_space() {
        let t = toy();
        assert_eq!(t.decode(&[4, 8, 9]), " unaffable");
        assert_eq!(t.decode(&[5, 7]), " hello world");
        assert_eq!(t.decode(&[2]), "[CLS]");
    }

    /// `max_token_len` caps the probe length, so a vocabulary whose
    /// longest piece is short must still tokenize correctly rather than
    /// missing longer matches that do not exist.
    #[test]
    fn the_probe_cap_is_the_longest_piece_in_bytes() {
        let t = toy();
        assert_eq!(t.max_token_len, "▁hello".len(), "6 bytes, not 6 chars");
    }
}
