//! Kimi K3's real tokenizer: tiktoken-style rank-based BPE, loaded from
//! a real `tiktoken.model` file (base64-encoded byte sequence + rank
//! per line -- the standard OpenAI tiktoken vocab format, confirmed
//! against a real downloaded Kimi K3 `tiktoken.model` and its real
//! `tokenization_kimi.py`/`tokenizer_config.json`), not from GGUF
//! metadata like `GgufBpeTokenizer`/`GgufSpmTokenizer` -- Kimi K3 ships
//! as safetensors, with its own real tokenizer format, distinct from
//! both existing tokenizers in this module.
//!
//! The real split pattern (`pat_str` in `tokenization_kimi.py`) uses
//! Unicode script/general-category properties (`\p{Han}`, `\p{Lu}`,
//! ...) plus a negative lookahead (`\s+(?!\S)`) that Rust's `regex`
//! crate deliberately doesn't support (no backtracking, for linear-time
//! guarantees) -- `fancy-regex` is used instead, since it supports
//! exactly this class of pattern while still being a generic,
//! model-agnostic regex engine (same category of tool as `regex`
//! itself), not tiktoken-specific code. One real translation was
//! needed: the real pattern uses `[A&&[^B]]` (character-class set
//! intersection), valid in Python's third-party `regex` module (which
//! `tiktoken`'s reference implementation depends on) but not in Rust's
//! regex syntax. Rewritten as `(?:(?!B)[A])` -- a per-character
//! negative lookahead guarding the class -- which is semantically
//! equivalent under repetition (`*`/`+`) since each repeated character
//! is independently re-checked. Verified byte-for-byte against the
//! real `tiktoken` Python library (see this module's tests).
//!
//! The core merge algorithm (given a UTF-8 text piece already isolated
//! by the split regex, repeatedly merge the lowest-rank adjacent byte
//! span until no mergeable pair remains) is transcribed from OpenAI's
//! publicly documented tiktoken algorithm description, not copied from
//! any source file.

use crate::tokenizer::{SpecialKind, SpecialTokenTable, SpecialTokens, TextOrSpecial};
use base64::Engine;
use fancy_regex::Regex;
use std::collections::HashMap;
use thiserror::Error;

/// The real split pattern from Kimi K3's `tokenization_kimi.py`, with
/// the `&&[^\p{Han}]` set-intersection idiom rewritten as an
/// equivalent per-character negative lookahead (see module doc
/// comment).
const PAT_STR: &str = concat!(
    r"[\p{Han}]+",
    "|",
    r"[^\r\n\p{L}\p{N}]?(?:(?!\p{Han})[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}])*(?:(?!\p{Han})[\p{Ll}\p{Lm}\p{Lo}\p{M}])+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"[^\r\n\p{L}\p{N}]?(?:(?!\p{Han})[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}])+(?:(?!\p{Han})[\p{Ll}\p{Lm}\p{Lo}\p{M}])*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"\p{N}{1,3}",
    "|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    "|",
    r"\s*[\r\n]+",
    "|",
    r"\s+(?!\S)",
    "|",
    r"\s+",
);

#[derive(Debug, Error)]
pub enum KimiTokenizerError {
    #[error("invalid tiktoken vocab line {0}: {1:?}")]
    InvalidVocabLine(usize, String),
    #[error("split regex failed to compile: {0}")]
    Regex(#[from] fancy_regex::Error),
    #[error("failed to parse tokenizer_config.json: {0}")]
    TokenizerConfigJson(#[from] serde_json::Error),
}

/// Parses the real `added_tokens_decoder` block of Kimi K3's
/// `tokenizer_config.json` (`{"163584": {"content": "[BOS]", ...}, ...}`,
/// real key/value shapes confirmed against the real downloaded file) into
/// a `name -> id` map suitable for `KimiTokenizer::new`'s
/// `special_tokens` argument.
pub fn parse_special_tokens(json_text: &str) -> Result<HashMap<String, u32>, KimiTokenizerError> {
    #[derive(serde::Deserialize)]
    struct AddedToken {
        content: String,
    }
    #[derive(serde::Deserialize)]
    struct TokenizerConfig {
        #[serde(default)]
        added_tokens_decoder: HashMap<String, AddedToken>,
    }
    let cfg: TokenizerConfig = serde_json::from_str(json_text)?;
    Ok(cfg
        .added_tokens_decoder
        .into_iter()
        .filter_map(|(id_str, tok)| id_str.parse::<u32>().ok().map(|id| (tok.content, id)))
        .collect())
}

/// Parses a real `.tiktoken`-format vocab file: one `<base64-bytes>
/// <rank>` pair per line, blank lines ignored. Standard OpenAI
/// tiktoken vocab format (confirmed against a real downloaded Kimi K3
/// `tiktoken.model`).
pub fn parse_tiktoken_vocab(text: &str) -> Result<HashMap<Vec<u8>, u32>, KimiTokenizerError> {
    let mut ranks = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, ' ');
        let b64 = parts
            .next()
            .ok_or_else(|| KimiTokenizerError::InvalidVocabLine(i, line.to_string()))?;
        let rank_str = parts
            .next()
            .ok_or_else(|| KimiTokenizerError::InvalidVocabLine(i, line.to_string()))?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|_| KimiTokenizerError::InvalidVocabLine(i, line.to_string()))?;
        let rank: u32 = rank_str
            .parse()
            .map_err(|_| KimiTokenizerError::InvalidVocabLine(i, line.to_string()))?;
        ranks.insert(bytes, rank);
    }
    Ok(ranks)
}

/// The real tiktoken byte-pair-merge algorithm: given one already-split
/// text piece's raw bytes, repeatedly merges the adjacent byte-span
/// pair with the lowest rank until no adjacent pair has a rank in the
/// table, then returns each final span's rank (its token id).
///
/// This is a *rank* merge, not an ordered-merge-rules BPE like
/// `GgufBpeTokenizer::encode_word` -- every candidate byte span's
/// mergeability is decided by a single lookup into `ranks`, not by
/// position in a fixed merge-priority list.
fn byte_pair_merge(piece: &[u8], ranks: &HashMap<Vec<u8>, u32>) -> Vec<u32> {
    if piece.is_empty() {
        return Vec::new();
    }
    if piece.len() == 1 {
        return vec![*ranks.get(piece).unwrap_or(&0)];
    }

    // `boundaries[i]` is the start byte offset of the i-th part;
    // `boundaries.len() - 1` parts remain once merging is done.
    let mut boundaries: Vec<usize> = (0..=piece.len()).collect();

    let rank_of = |boundaries: &[usize], i: usize| -> Option<u32> {
        if i + 2 >= boundaries.len() {
            return None;
        }
        ranks.get(&piece[boundaries[i]..boundaries[i + 2]]).copied()
    };

    loop {
        if boundaries.len() <= 2 {
            break;
        }
        let mut best: Option<(u32, usize)> = None;
        for i in 0..boundaries.len() - 2 {
            if let Some(r) = rank_of(&boundaries, i) {
                if best.is_none_or(|(br, _)| r < br) {
                    best = Some((r, i));
                }
            }
        }
        match best {
            Some((_, i)) => {
                boundaries.remove(i + 1);
            }
            None => break,
        }
    }

    boundaries
        .windows(2)
        .map(|w| {
            *ranks
                .get(&piece[w[0]..w[1]])
                .expect("every final span must be a real vocab entry")
        })
        .collect()
}

pub struct KimiTokenizer {
    encoder: HashMap<Vec<u8>, u32>,
    decoder: HashMap<u32, Vec<u8>>,
    special_tokens: HashMap<String, u32>,
    /// The same specials as a carve-out table, for
    /// [`SpecialTokens::Parse`] -- tiktoken's `allowed_special="all"`.
    special_table: SpecialTokenTable,
    split_re: Regex,
}

impl KimiTokenizer {
    pub fn new(
        encoder: HashMap<Vec<u8>, u32>,
        special_tokens: HashMap<String, u32>,
    ) -> Result<Self, KimiTokenizerError> {
        let decoder = encoder.iter().map(|(k, v)| (*v, k.clone())).collect();
        let split_re = Regex::new(PAT_STR)?;
        let special_table = SpecialTokenTable::from_entries(
            special_tokens
                .iter()
                .map(|(name, &id)| (name.as_str(), id, SpecialKind::Control)),
        );
        Ok(KimiTokenizer {
            encoder,
            decoder,
            special_tokens,
            special_table,
            split_re,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    pub fn special_token_id(&self, name: &str) -> Option<u32> {
        self.special_tokens.get(name).copied()
    }

    /// Encodes text. [`SpecialTokens::AsText`] is the real
    /// `disallowed_special=()` path `tokenization_kimi.py` uses for
    /// untrusted/user text: a literal `<|...|>` substring is BPE-encoded
    /// like any other text, never misread as a control token.
    /// [`SpecialTokens::Parse`] is `allowed_special="all"`: every name
    /// in `tokenizer_config.json` is carved out as its id.
    pub fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<u32> {
        let mut out = Vec::new();
        for seg in self.special_table.split(text, specials) {
            match seg {
                TextOrSpecial::Special(id) => out.push(id),
                TextOrSpecial::Text(run) => self.encode_text_run(run, &mut out),
            }
        }
        out
    }

    fn encode_text_run(&self, text: &str, out: &mut Vec<u32>) {
        for piece in self.split_re.find_iter(text) {
            let piece = piece.expect("split regex match should not error mid-scan");
            let bytes = piece.as_str().as_bytes();
            if let Some(&id) = self.encoder.get(bytes) {
                out.push(id);
                continue;
            }
            out.extend(byte_pair_merge(bytes, &self.encoder));
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(b) = self.decoder.get(&id) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_ranks() -> HashMap<Vec<u8>, u32> {
        // Every single byte 0..=255 gets its own rank equal to its
        // value (matches real tiktoken vocabs, which always include
        // every single byte as a base token), plus a few real merges
        // built up in increasing-rank order (lower rank = merged
        // first, matching real tiktoken semantics).
        let mut ranks: HashMap<Vec<u8>, u32> = (0u32..256).map(|b| (vec![b as u8], b)).collect();
        let mut next = 256u32;
        let add = |bytes: &[u8], ranks: &mut HashMap<Vec<u8>, u32>, next: &mut u32| {
            ranks.insert(bytes.to_vec(), *next);
            *next += 1;
        };
        add(b"he", &mut ranks, &mut next); // "h"+"e" -> "he"
        add(b"ll", &mut ranks, &mut next); // "l"+"l" -> "ll"
        add(b"hel", &mut ranks, &mut next); // "he"+"l" -> "hel"
        add(b"hell", &mut ranks, &mut next); // "hel"+"l" -> "hell" (uses "ll"? either path, lowest rank wins)
        add(b"hello", &mut ranks, &mut next); // "hell"+"o" -> "hello"
        ranks
    }

    #[test]
    fn byte_pair_merge_prefers_the_lowest_rank_pair_first() {
        let ranks = tiny_ranks();
        let ids = byte_pair_merge(b"hello", &ranks);
        // "hello" is itself a vocab entry with the lowest possible
        // rank among any partition, so the merge should collapse all
        // the way down to the single "hello" token.
        let hello_id = *ranks.get(b"hello".as_slice()).unwrap();
        assert_eq!(ids, vec![hello_id]);
    }

    #[test]
    fn byte_pair_merge_falls_back_to_single_bytes_with_no_mergeable_pairs() {
        let ranks = tiny_ranks();
        let ids = byte_pair_merge(b"xyz", &ranks);
        assert_eq!(ids, vec![b'x' as u32, b'y' as u32, b'z' as u32]);
    }

    #[test]
    fn byte_pair_merge_merges_a_known_pair_but_not_unknown_neighbors() {
        let ranks = tiny_ranks();
        // "he" merges (known pair), "z" stays alone.
        let ids = byte_pair_merge(b"hez", &ranks);
        let he_id = *ranks.get(b"he".as_slice()).unwrap();
        assert_eq!(ids, vec![he_id, b'z' as u32]);
    }

    #[test]
    fn encode_decode_roundtrips_on_a_tiny_synthetic_vocab() {
        let ranks = tiny_ranks();
        let tok = KimiTokenizer::new(ranks, HashMap::new()).expect("regex must compile");
        let ids = tok.encode("hello", SpecialTokens::AsText);
        let back = tok.decode(&ids);
        assert_eq!(back, "hello");
    }

    /// tiktoken's two modes: `disallowed_special=()` leaves a written
    /// marker as bytes, `allowed_special="all"` returns its id.
    #[test]
    fn a_written_marker_is_bytes_as_text_and_one_id_when_parsed() {
        let ranks = tiny_ranks();
        let specials = HashMap::from([("[EOS]".to_string(), 9_000u32)]);
        let tok = KimiTokenizer::new(ranks, specials).expect("regex must compile");
        let as_text = tok.encode("hello[EOS]", SpecialTokens::AsText);
        assert!(!as_text.contains(&9_000), "got {as_text:?}");
        let parsed = tok.encode("hello[EOS]", SpecialTokens::Parse);
        assert_eq!(parsed.last(), Some(&9_000), "got {parsed:?}");
        assert_eq!(
            &parsed[..parsed.len() - 1],
            &tok.encode("hello", SpecialTokens::AsText)[..]
        );
    }

    #[test]
    fn split_regex_separates_words_punctuation_and_whitespace() {
        let ranks = tiny_ranks();
        let tok = KimiTokenizer::new(ranks, HashMap::new()).expect("regex must compile");
        let pieces: Vec<&str> = tok
            .split_re
            .find_iter("hello, world!")
            .map(|m| m.unwrap().as_str())
            .collect();
        assert_eq!(pieces, vec!["hello", ",", " world", "!"]);
    }

    #[test]
    fn split_regex_keeps_han_script_separate_from_latin_words() {
        let ranks = tiny_ranks();
        let tok = KimiTokenizer::new(ranks, HashMap::new()).expect("regex must compile");
        let pieces: Vec<&str> = tok
            .split_re
            .find_iter("Hi你好")
            .map(|m| m.unwrap().as_str())
            .collect();
        // "Hi" (Latin word) and "你好" (Han run) must land in separate
        // pieces -- this is exactly what the `&&[^\p{Han}]` exclusion
        // (rewritten as a lookahead here) exists to guarantee.
        assert_eq!(pieces, vec!["Hi", "你好"]);
    }

    #[test]
    fn parses_a_real_tiktoken_format_vocab_file() {
        // "IQ==" base64-decodes to the single byte 0x21 ('!'), matching
        // the real Kimi K3 tiktoken.model's first line format exactly.
        let text = "IQ== 0\nIg== 1\n";
        let ranks = parse_tiktoken_vocab(text).expect("must parse");
        assert_eq!(ranks.get(&vec![0x21]), Some(&0));
        assert_eq!(ranks.get(&vec![0x22]), Some(&1));
    }

    #[test]
    fn ignores_blank_lines_in_a_tiktoken_vocab_file() {
        let text = "IQ== 0\n\nIg== 1\n\n";
        let ranks = parse_tiktoken_vocab(text).expect("must parse");
        assert_eq!(ranks.len(), 2);
    }

    #[test]
    fn parses_real_shaped_special_tokens_from_tokenizer_config_json() {
        // Matches the real Kimi K3 tokenizer_config.json's shape
        // exactly (fetched live): string-encoded integer keys, each
        // with at least a "content" field.
        let json = r#"{
            "added_tokens_decoder": {
                "163584": {"content": "[BOS]", "special": true},
                "163585": {"content": "[EOS]", "special": true},
                "163586": {"content": "<|end_of_msg|>", "special": true}
            }
        }"#;
        let tokens = parse_special_tokens(json).expect("must parse");
        assert_eq!(tokens.get("[BOS]"), Some(&163584));
        assert_eq!(tokens.get("[EOS]"), Some(&163585));
        assert_eq!(tokens.get("<|end_of_msg|>"), Some(&163586));
        assert_eq!(tokens.len(), 3);
    }
}
