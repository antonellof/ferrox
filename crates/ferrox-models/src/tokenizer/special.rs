//! Special tokens: which vocabulary entries are one, and whether a
//! literal marker in the input is parsed as the token it names.
//!
//! Shared by every GGUF tokenizer in this crate (BPE, SPM, Unigram,
//! WordPiece), so the four cannot come to disagree about what a special
//! token is. Transcribed from llama.cpp's `src/llama-vocab.cpp`, and
//! every rule below cites the line it came from.
//!
//! # The two questions
//!
//! **Which entries are special?** llama.cpp's `cache_special_tokens`
//! (`llama-vocab.cpp:2948-2952`): every token whose attribute is
//! `CONTROL`, `USER_DEFINED` or `UNKNOWN`. The attribute starts as the
//! file's `tokenizer.ggml.token_type` (`:2447-2458`) and is then
//! adjusted by name: every text in [`EOG_TOKEN_TEXTS`] is promoted to
//! `CONTROL` whatever the file said (`:2800-2832`, "control-looking
//! token ... its type will be overridden"), and two family-specific
//! demotions follow (`:2880-2937`).
//!
//! That is the WHOLE rule. This module used to add one of its own -- any
//! vocabulary entry shaped like `<...>` was treated as special -- and
//! that made Qwen2.5's `<s>`, which its vocabulary carries as an ordinary
//! NORMAL entry, tokenize as one id where llama.cpp gives three (`<`,
//! `s`, `>`), under BOTH `parse_special` settings. A document that
//! mentions `<s>` in prose was off by one token per mention.
//!
//! **Is a marker in the text parsed?** llama.cpp's `parse_special`
//! (`llama.h`, `llama_tokenize`: "Allow tokenizing special and/or
//! control tokens which otherwise are not exposed and treated as
//! plaintext"). Its `tokenizer_st_partition` (`:3163-3175`) skips
//! `CONTROL` and `UNKNOWN` entries when it is false and still carves
//! out `USER_DEFINED` ones, because HuggingFace's tokenizers do. The
//! library default is `false` (`common/common.h:1015`), and this crate
//! used to behave as if it were always `true`: prose that mentioned
//! `<|im_end|>` inside backticks became the end-of-turn token. Every
//! `encode` now takes a [`SpecialTokens`] so a caller says which it
//! wants, and the callers are matched to llama.cpp's one by one -- the
//! table is in the PR that introduced this module.
//!
//! Not ported: the `LSTRIP`/`RSTRIP` attributes llama.cpp sets by model
//! name for jina-v2, phi-3 and modern-bert (`:3020-3047`), which strip
//! whitespace next to a special. No local checkpoint carries them, so
//! there is no fixture to hold an implementation to.

use super::EOG_TOKEN_TEXTS;

/// Whether a literal special-token marker in the input (`<s>`,
/// `<|im_end|>`, `[SEP]`) is carved out as the token it names or
/// tokenized as the characters it is written with.
///
/// llama.cpp's `parse_special`. Callers choose per site, because the
/// right answer differs: a rendered chat prompt needs its template's own
/// markers parsed, while a stop string, a DRY sequence breaker or a
/// rerank document is plain text whatever it happens to mention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialTokens {
    /// `parse_special = false`, llama.cpp's library default. `CONTROL`
    /// and `UNKNOWN` entries are ordinary text; `USER_DEFINED` ones are
    /// still carved out (`llama-vocab.cpp:3169-3174`).
    AsText,
    /// `parse_special = true`: every special entry is matched as an
    /// atomic substring before normal tokenization runs.
    Parse,
}

/// The attribute that makes an entry special, from llama.cpp's
/// `llama_token_attr` (`include/llama.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpecialKind {
    /// `LLAMA_TOKEN_ATTR_CONTROL`: parsed only under [`SpecialTokens::Parse`].
    Control,
    /// `LLAMA_TOKEN_ATTR_USER_DEFINED`: parsed under both settings.
    UserDefined,
    /// `LLAMA_TOKEN_ATTR_UNKNOWN`: parsed only under [`SpecialTokens::Parse`].
    Unknown,
}

impl SpecialKind {
    /// `tokenizer_st_partition`'s gate (`llama-vocab.cpp:3169`): with
    /// `parse_special == false`, control and unknown tokens are skipped.
    fn is_parsed(self, mode: SpecialTokens) -> bool {
        match mode {
            SpecialTokens::Parse => true,
            SpecialTokens::AsText => self == SpecialKind::UserDefined,
        }
    }
}

/// GGUF's `tokenizer.ggml.token_type` per-token integer tag, matching
/// llama.cpp's `llama_token_type` enum (`include/llama.h`) and its
/// GGUF-loading switch (`llama-vocab.cpp:2447-2458`). A plain sequential
/// enum on disk (`1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED,
/// 5=UNUSED, 6=BYTE`), which is a different and simpler representation
/// than llama.cpp's internal bit-flag `llama_token_attr`.
const GGML_TOKEN_TYPE_UNKNOWN: i64 = 2;
const GGML_TOKEN_TYPE_CONTROL: i64 = 3;
const GGML_TOKEN_TYPE_USER_DEFINED: i64 = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SpecialToken {
    pub text: String,
    pub id: u32,
    pub kind: SpecialKind,
}

/// One chunk of [`SpecialTokenTable::split`]'s output: either a raw
/// text run to tokenize normally, or an already-resolved special id.
pub(crate) enum TextOrSpecial<'a> {
    Text(&'a str),
    Special(u32),
}

/// The special entries of one vocabulary, in the order llama.cpp
/// partitions on them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SpecialTokenTable {
    /// Longest text first (`llama-vocab.cpp:2954-2958`); ties keep id
    /// order, which llama.cpp's unstable sort leaves unspecified.
    entries: Vec<SpecialToken>,
}

impl SpecialTokenTable {
    /// Reads `tokenizer.ggml.token_type` and applies llama.cpp's by-name
    /// adjustments. `id_to_token` is the vocabulary in id order.
    pub fn from_gguf(file: &impl ferrox_gguf::TensorSource, id_to_token: &[String]) -> Self {
        let mut kinds: Vec<Option<SpecialKind>> = vec![None; id_to_token.len()];

        if let Some(ferrox_gguf::GgufValue::Array(items)) =
            file.metadata("tokenizer.ggml.token_type")
        {
            for (kind, v) in kinds.iter_mut().zip(items) {
                let ty = match v {
                    ferrox_gguf::GgufValue::I32(t) => *t as i64,
                    ferrox_gguf::GgufValue::U32(t) => *t as i64,
                    _ => continue,
                };
                *kind = match ty {
                    GGML_TOKEN_TYPE_CONTROL => Some(SpecialKind::Control),
                    GGML_TOKEN_TYPE_USER_DEFINED => Some(SpecialKind::UserDefined),
                    GGML_TOKEN_TYPE_UNKNOWN => Some(SpecialKind::Unknown),
                    _ => None,
                };
            }
        }

        // `llama-vocab.cpp:2800-2832`: every end-of-generation text is
        // CONTROL whatever the file said. Yi-1.5-6B-Chat ships
        // `<|im_end|>` as NORMAL; this is what makes it one token there
        // (and `<|im_start|>`, which is not on the list, stays
        // shattered -- on llama.cpp too).
        for (id, text) in id_to_token.iter().enumerate() {
            if EOG_TOKEN_TEXTS.contains(&text.as_str()) {
                kinds[id] = Some(SpecialKind::Control);
            }
        }

        // `llama-vocab.cpp:2880-2912`: gpt-oss / solar-open render
        // `<|end|>` as USER_DEFINED so it is parsed even as plain text.
        let has = |t: &str| id_to_token.iter().any(|x| x == t);
        if has("<|end|>")
            && ((has("<|return|>") && has("<|call|>")) || (has("<|calls|>") && has("<|flush|>")))
        {
            for (id, text) in id_to_token.iter().enumerate() {
                if text == "<|end|>" {
                    kinds[id] = Some(SpecialKind::UserDefined);
                }
            }
        }
        // `llama-vocab.cpp:2914-2937`: gemma-4 / paddleocr carry `</s>`
        // as an ordinary word once `<|tool_response>` is present.
        if has("<|tool_response>") && has("</s>") {
            for (id, text) in id_to_token.iter().enumerate() {
                if text == "</s>" {
                    kinds[id] = None;
                }
            }
        }

        Self::from_entries(
            kinds
                .into_iter()
                .enumerate()
                .filter_map(|(id, kind)| Some((id_to_token[id].as_str(), id as u32, kind?))),
        )
    }

    /// A table from `(text, id, kind)` triples: for a vocabulary that
    /// carries its specials outside GGUF metadata (Kimi's
    /// `tokenizer_config.json`), and for tests. The GGUF constructor
    /// above ends here too, so there is one ordering rule.
    pub fn from_entries<'a>(
        entries: impl IntoIterator<Item = (&'a str, u32, SpecialKind)>,
    ) -> Self {
        let mut entries: Vec<SpecialToken> = entries
            .into_iter()
            // An empty special would match everywhere and nowhere.
            .filter(|(text, _, _)| !text.is_empty())
            .map(|(text, id, kind)| SpecialToken {
                text: text.to_string(),
                id,
                kind,
            })
            .collect();
        // Stable, so equal lengths keep id order.
        entries.sort_by_key(|e| std::cmp::Reverse(e.text.len()));
        SpecialTokenTable { entries }
    }

    /// Splits `text` around every literal occurrence of every special
    /// entry `mode` lets through, leaving the text runs between them
    /// for the caller's normal tokenization pass.
    ///
    /// A port of `tokenizer_st_partition` (`llama-vocab.cpp:3163-3268`):
    /// specials are taken longest first, and each one splits every raw
    /// fragment left by the ones before it. Order matters when two
    /// specials overlap, and this is llama.cpp's order.
    pub fn split<'a>(&self, text: &'a str, mode: SpecialTokens) -> Vec<TextOrSpecial<'a>> {
        let mut fragments = vec![TextOrSpecial::Text(text)];
        for special in self.entries.iter().filter(|s| s.kind.is_parsed(mode)) {
            let mut next = Vec::with_capacity(fragments.len());
            for fragment in fragments {
                match fragment {
                    TextOrSpecial::Special(id) => next.push(TextOrSpecial::Special(id)),
                    TextOrSpecial::Text(run) => {
                        let mut rest = run;
                        while let Some(at) = rest.find(special.text.as_str()) {
                            if at > 0 {
                                next.push(TextOrSpecial::Text(&rest[..at]));
                            }
                            next.push(TextOrSpecial::Special(special.id));
                            rest = &rest[at + special.text.len()..];
                        }
                        if !rest.is_empty() {
                            next.push(TextOrSpecial::Text(rest));
                        }
                    }
                }
            }
            fragments = next;
        }
        fragments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_gguf::{GgufError, GgufValue, TensorInfo, TensorSource};

    fn text_of<'a>(seg: &TextOrSpecial<'a>) -> Option<&'a str> {
        match seg {
            TextOrSpecial::Text(t) => Some(t),
            TextOrSpecial::Special(_) => None,
        }
    }

    fn table(specials: &[(&str, u32)]) -> SpecialTokenTable {
        SpecialTokenTable::from_entries(
            specials
                .iter()
                .map(|&(t, id)| (t, id, SpecialKind::Control)),
        )
    }

    #[test]
    fn an_empty_table_returns_the_whole_text_unsplit() {
        let segs = table(&[]).split("hello world", SpecialTokens::Parse);
        assert_eq!(segs.len(), 1);
        assert_eq!(text_of(&segs[0]), Some("hello world"));
    }

    #[test]
    fn splits_around_a_single_special_token_in_the_middle() {
        let segs = table(&[("<|sep|>", 99)]).split("before<|sep|>after", SpecialTokens::Parse);
        assert_eq!(segs.len(), 3);
        assert_eq!(text_of(&segs[0]), Some("before"));
        assert!(matches!(segs[1], TextOrSpecial::Special(99)));
        assert_eq!(text_of(&segs[2]), Some("after"));
    }

    #[test]
    fn multiple_occurrences_and_multiple_distinct_specials_all_split() {
        let segs = table(&[("<a>", 1), ("<b>", 2)]).split("<a>x<b>y<a>", SpecialTokens::Parse);
        let ids: Vec<u32> = segs
            .iter()
            .filter_map(|s| match s {
                TextOrSpecial::Special(id) => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec![1, 2, 1]);
        let texts: Vec<&str> = segs.iter().filter_map(text_of).collect();
        assert_eq!(texts, vec!["x", "y"]);
    }

    /// llama.cpp partitions longest-first, so a marker that is a prefix
    /// of a longer one never steals the longer one's match.
    #[test]
    fn the_longest_special_is_carved_out_before_a_prefix_of_it() {
        let segs = table(&[("<s>", 1), ("<s>x", 2)]).split("<s>x", SpecialTokens::Parse);
        assert_eq!(segs.len(), 1);
        assert!(matches!(segs[0], TextOrSpecial::Special(2)));
    }

    #[test]
    fn no_match_at_all_returns_the_whole_text_as_one_segment() {
        let segs = table(&[("<|zzz|>", 5)]).split("nothing here", SpecialTokens::Parse);
        assert_eq!(segs.len(), 1);
        assert_eq!(text_of(&segs[0]), Some("nothing here"));
    }

    /// The gate this module exists for. `tokenizer_st_partition` skips
    /// CONTROL and UNKNOWN entries when `parse_special` is false and
    /// still carves out USER_DEFINED ones (`llama-vocab.cpp:3169-3174`).
    #[test]
    fn as_text_leaves_control_and_unknown_markers_as_prose_but_still_parses_user_defined() {
        let t = SpecialTokenTable::from_entries(vec![
            ("<|im_end|>", 7, SpecialKind::Control),
            ("<unk>", 0, SpecialKind::Unknown),
            ("<|user|>", 9, SpecialKind::UserDefined),
        ]);
        let segs = t.split("a<|im_end|>b<unk>c<|user|>d", SpecialTokens::AsText);
        let texts: Vec<&str> = segs.iter().filter_map(text_of).collect();
        assert_eq!(texts, vec!["a<|im_end|>b<unk>c", "d"]);
        assert!(matches!(segs[1], TextOrSpecial::Special(9)));

        let segs = t.split("a<|im_end|>b<unk>c<|user|>d", SpecialTokens::Parse);
        let ids: Vec<u32> = segs
            .iter()
            .filter_map(|s| match s {
                TextOrSpecial::Special(id) => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec![7, 0, 9]);
    }

    struct MetaOnly(std::collections::HashMap<String, GgufValue>);

    impl TensorSource for MetaOnly {
        fn metadata(&self, key: &str) -> Option<&GgufValue> {
            self.0.get(key)
        }
        fn find_tensor(&self, _name: &str) -> Option<&TensorInfo> {
            None
        }
        fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
            Err(GgufError::TensorNotFound(name.to_string()))
        }
        fn tensor_mapped_range(
            &self,
            name: &str,
        ) -> Result<
            (
                std::sync::Arc<ferrox_gguf::MmapHandle>,
                std::ops::Range<usize>,
            ),
            GgufError,
        > {
            Err(GgufError::TensorNotFound(name.to_string()))
        }
    }

    fn vocab(tokens: &[(&str, i32)]) -> (MetaOnly, Vec<String>) {
        let mut m = std::collections::HashMap::new();
        m.insert(
            "tokenizer.ggml.token_type".to_string(),
            GgufValue::Array(tokens.iter().map(|&(_, ty)| GgufValue::I32(ty)).collect()),
        );
        let id_to_token = tokens.iter().map(|&(t, _)| t.to_string()).collect();
        (MetaOnly(m), id_to_token)
    }

    /// Qwen2.5-1.5B's vocabulary carries `<s>` (id 128245) as NORMAL,
    /// and llama.cpp tokenizes it as `<`, `s`, `>` under both settings.
    /// A shape-based promotion made it one token here; that is the
    /// off-by-one `ferrox imatrix` measured on `docs/CLI.md`.
    #[test]
    fn a_normal_typed_entry_shaped_like_a_marker_is_not_special() {
        let (file, ids) = vocab(&[("<", 1), ("s", 1), (">", 1), ("<s>", 1), ("<|im_end|>", 3)]);
        let t = SpecialTokenTable::from_gguf(&file, &ids);
        let segs = t.split("<s><|im_end|>", SpecialTokens::Parse);
        let texts: Vec<&str> = segs.iter().filter_map(text_of).collect();
        assert_eq!(texts, vec!["<s>"]);
        assert!(matches!(segs[1], TextOrSpecial::Special(4)));
    }

    /// `llama-vocab.cpp:2800-2832`: an end-of-generation text is CONTROL
    /// whatever the file says. Yi-1.5-6B-Chat is the checkpoint that
    /// needs it -- `<|im_end|>` is NORMAL in its file.
    #[test]
    fn an_end_of_generation_text_is_control_even_when_the_file_says_normal() {
        let (file, ids) = vocab(&[("<|im_start|>", 1), ("<|im_end|>", 1)]);
        let t = SpecialTokenTable::from_gguf(&file, &ids);
        assert_eq!(
            t.entries,
            vec![SpecialToken {
                text: "<|im_end|>".to_string(),
                id: 1,
                kind: SpecialKind::Control
            }]
        );
    }

    #[test]
    fn user_defined_and_unknown_types_are_special_and_normal_byte_and_unused_are_not() {
        let (file, ids) = vocab(&[
            ("<unk>", 2),
            ("<ctl>", 3),
            ("<usr>", 4),
            ("<unused>", 5),
            ("<0x00>", 6),
            ("word", 1),
        ]);
        let t = SpecialTokenTable::from_gguf(&file, &ids);
        let kinds: Vec<(u32, SpecialKind)> = t.entries.iter().map(|e| (e.id, e.kind)).collect();
        assert_eq!(
            kinds,
            vec![
                (0, SpecialKind::Unknown),
                (1, SpecialKind::Control),
                (2, SpecialKind::UserDefined)
            ]
        );
    }

    /// `llama-vocab.cpp:2914-2937`: gemma-4 ships `</s>` beside
    /// `<|tool_response>`, and there it is an ordinary word.
    #[test]
    fn gemma4_style_end_of_sentence_is_demoted_beside_tool_response() {
        let (file, ids) = vocab(&[("</s>", 3), ("<|tool_response>", 3)]);
        let t = SpecialTokenTable::from_gguf(&file, &ids);
        assert_eq!(t.entries.iter().map(|e| e.id).collect::<Vec<_>>(), vec![1]);
    }

    /// `llama-vocab.cpp:2880-2912`: gpt-oss's `<|end|>` becomes
    /// USER_DEFINED, so it is parsed even as plain text.
    #[test]
    fn harmony_end_is_user_defined_when_return_and_call_are_present() {
        let (file, ids) = vocab(&[("<|end|>", 3), ("<|return|>", 3), ("<|call|>", 3)]);
        let t = SpecialTokenTable::from_gguf(&file, &ids);
        let end = t.entries.iter().find(|e| e.text == "<|end|>").unwrap();
        assert_eq!(end.kind, SpecialKind::UserDefined);
        let segs = t.split("x<|end|>y", SpecialTokens::AsText);
        assert!(matches!(segs[1], TextOrSpecial::Special(0)));
    }

    #[test]
    fn a_file_without_token_types_has_only_the_by_name_specials() {
        let file = MetaOnly(std::collections::HashMap::new());
        let ids: Vec<String> = ["a", "<|eot_id|>", "b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let t = SpecialTokenTable::from_gguf(&file, &ids);
        assert_eq!(t.entries.iter().map(|e| e.id).collect::<Vec<_>>(), vec![1]);
    }
}
