//! WordPiece against llama.cpp's own answer, on the same GGUF.
//!
//! # Why this file and not a round trip
//!
//! A WordPiece round trip cannot fail usefully: the normalizer
//! lowercases, folds accents and drops control characters before any id
//! exists, so `decode(encode(x)) != x` for most inputs and the test
//! would have to be written loosely enough to prove nothing. That is the
//! exact shape of the hole the BPE pre-tokenizer defects hid in for the
//! life of this project (see `ferrox parity`'s tokenizer module). So the
//! assertion here is against llama.cpp's ids, token for token.
//!
//! # The two tests, and why there are two
//!
//! * [`ferrox_matches_the_recorded_llama_cpp_tokenization`] needs only
//!   the GGUF. It is the evidence: 36 cases and 511 tokens of llama.cpp
//!   output, frozen below.
//! * [`the_recorded_tokenization_is_still_what_llama_cpp_produces`] needs
//!   llama.cpp as well, and re-derives the table. It is what stops the
//!   frozen numbers from becoming a record of ferrox agreeing with
//!   itself.
//!
//! Both are `#[ignore]`d and both skip loudly when their inputs are
//! missing, because neither the 36 MB checkpoint nor a libllama build is
//! in the repository.
//!
//! # Reproducing the recorded table
//!
//! ```text
//! ferrox download CompendiumLabs/bge-small-en-v1.5-gguf \
//!     bge-small-en-v1.5-q8_0.gguf --local-dir models
//! ./tools/build_llama_logits.sh
//! cargo test -p ferrox-models --test wordpiece_llama_cpp_parity -- --ignored
//! ```
//!
//! The recorded ids came from llama.cpp at
//! `1269cb1ff1598751f846241be90083ae9ad036fb`, which is the revision in
//! `.scratch/llama.cpp` that [`ferrox_models::GgufWordPieceTokenizer`]
//! was transcribed from.
//!
//! **A libllama older than that revision implements a different
//! normalizer**, and the second test detects that and skips rather than
//! reporting a ferrox defect. Upstream's WordPiece normalizer grew
//! `BertNormalizer`'s `strip_accents` switch, which drops a *standalone*
//! combining mark (`e` + U+0301); before that it kept the mark, welded
//! it into the word and produced `[UNK]`. HuggingFace's own
//! `BertNormalizer` drops it, and so does the revision above, so ferrox
//! follows it. Homebrew's `llama.cpp 7650` is on the older side of that
//! change, and `strings libllama.dylib | grep normalizer` finding
//! nothing is the other way to tell.
//!
//! # What the corpus is
//!
//! The first nineteen cases are `ferrox parity`'s own corpus, verbatim,
//! so this checkpoint is held to the same inputs as the nine BPE and SPM
//! checkpoints that oracle covers. They are duplicated rather than
//! shared because the corpus lives in a private module of `ferrox-cli`,
//! which depends on this crate and cannot be depended on back.
//!
//! The remaining seventeen are WordPiece's own hazards, which a corpus
//! written against BPE pre-tokenizer regexes does not reach: words that
//! must split into continuation pieces, words that must come back as one
//! `[UNK]` rather than several, `[CLS]`-style specials and text that only
//! looks like one, and inputs that normalize away to nothing.

use ferrox_gguf::ShardedGguf;
use ferrox_models::tokenizer::SpecialTokens;
use ferrox_models::GgufWordPieceTokenizer;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Golden {
    name: &'static str,
    text: &'static str,
    /// llama.cpp's ids for `text`, with `add_special = false` and
    /// `parse_special = true` (the `bracket-specials` case is `[CLS]`
    /// as id 101, which only the parsed setting gives).
    llama: &'static [u32],
}

/// llama.cpp's answer for the whole corpus. See the module docs for how
/// it was produced and how to reproduce it.
#[rustfmt::skip]
const GOLDEN: &[Golden] = &[
    Golden {
        name: "digit-runs",
        text: "Build 1234567 of 89 took 100000 ms on port 8080 with seed 42.",
        llama: &[
            3857, 13138, 19961, 2575, 2581, 1997, 6486, 2165, 6694, 8889,
            5796, 2006, 3417, 3770, 17914, 2007, 6534, 4413, 1012,
        ],
    },
    Golden {
        name: "digit-boundaries",
        text: "1 12 123 1234 12345 123456 1234567 12345678 123456789 1234567890",
        llama: &[
            1015, 2260, 13138, 13138, 2549, 13138, 19961, 13138, 19961, 2575,
            13138, 19961, 2575, 2581, 13138, 19961, 2575, 2581, 2620, 13138,
            19961, 2575, 2581, 2620, 2683, 13138, 19961, 2575, 2581, 2620,
            21057,
        ],
    },
    Golden {
        name: "double-space",
        text: "one  two   three    four     five",
        llama: &[
            2028, 2048, 2093, 2176, 2274,
        ],
    },
    Golden {
        name: "indented-code",
        text: "def f(x):\n    if x > 10:\n        return x * 2\n    return 0\n",
        llama: &[
            13366, 1042, 1006, 1060, 1007, 1024, 2065, 1060, 1028, 2184,
            1024, 2709, 1060, 1008, 1016, 2709, 1014,
        ],
    },
    Golden {
        name: "blank-lines",
        text: "para one\n\npara two\n\n\n    indented after two blanks\n",
        llama: &[
            11498, 2028, 11498, 2048, 27427, 14088, 2044, 2048, 8744, 2015,
        ],
    },
    Golden {
        name: "trailing-space",
        text: "wrapped up   ",
        llama: &[
            5058, 2039,
        ],
    },
    Golden {
        name: "tabs",
        text: "\tif x:\n\t\treturn 1\n\t \treturn 2\n",
        llama: &[
            2065, 1060, 1024, 2709, 1015, 2709, 1016,
        ],
    },
    Golden {
        name: "contractions-upper",
        text: "DON'T SAY IT'S HERS. I'LL GO. THEY'VE WON. WE'RE READY. HE'D KNOWN.",
        llama: &[
            2123, 1005, 1056, 2360, 2009, 1005, 1055, 5106, 1012, 1045, 1005,
            2222, 2175, 1012, 2027, 1005, 2310, 2180, 1012, 2057, 1005, 2128,
            3201, 1012, 2002, 1005, 1040, 2124, 1012,
        ],
    },
    Golden {
        name: "contractions-mixed",
        text: "It's Bob's, isn't it? You'd've said so. O'Neill's team we're sure They'Ve gone.",
        llama: &[
            2009, 1005, 1055, 3960, 1005, 1055, 1010, 3475, 1005, 1056, 2009,
            1029, 2017, 1005, 1040, 1005, 2310, 2056, 2061, 1012, 1051, 1005,
            11511, 1005, 1055, 2136, 2057, 1005, 2128, 2469, 2027, 1005,
            2310, 2908, 1012,
        ],
    },
    Golden {
        name: "cjk",
        text: "日本語のテキスト、中文字符和한국어が混在。城市名：東京、北京、서울。",
        llama: &[
            1864, 1876, 1950, 1671, 30239, 30227, 30233, 30240, 1635, 1746,
            1861, 100, 100, 1796, 1469, 29991, 29999, 30177, 100, 100, 1636,
            1804, 100, 1795, 1993, 1879, 1755, 1635, 1781, 1755, 1635, 1461,
            29999, 1636,
        ],
    },
    Golden {
        name: "emoji",
        text: "ship it 🚀 done ✅ 👨\u{200d}👩\u{200d}👧\u{200d}👦 🇯🇵 café e\u{301}clair",
        llama: &[
            2911, 2009, 100, 2589, 100, 100, 100, 7668, 14925, 19771, 2099,
        ],
    },
    Golden {
        name: "punct-runs",
        text: "-- --- **bold** `code` <<x>> !!!??? ... ->|<- ;;;\n\n",
        llama: &[
            1011, 1011, 1011, 1011, 1011, 1008, 1008, 7782, 1008, 1008, 1036,
            3642, 1036, 1026, 1026, 1060, 1028, 1028, 999, 999, 999, 1029,
            1029, 1029, 1012, 1012, 1012, 1011, 1028, 1064, 1026, 1011, 1025,
            1025, 1025,
        ],
    },
    Golden {
        name: "alnum-mixtures",
        text: "Qwen2.5-1.5B v0.13.3 IPv4 192.168.0.1:8443 0xDEADBEEF sha256:9f86d0",
        llama: &[
            1053, 12449, 2475, 1012, 1019, 1011, 1015, 1012, 1019, 2497,
            1058, 2692, 1012, 2410, 1012, 1017, 12997, 2615, 2549, 17613,
            1012, 16923, 1012, 1014, 1012, 1015, 1024, 6391, 23777, 1014,
            2595, 3207, 4215, 11306, 2546, 21146, 17788, 2575, 1024, 1023,
            2546, 20842, 2094, 2692,
        ],
    },
    Golden {
        name: "crlf",
        text: "line one\r\nline two\r\n\r\n    line three\r\n",
        llama: &[
            2240, 2028, 2240, 2048, 2240, 2093,
        ],
    },
    Golden {
        name: "leading-whitespace",
        text: "   leading spaces, then text.",
        llama: &[
            2877, 7258, 1010, 2059, 3793, 1012,
        ],
    },
    Golden {
        name: "unicode-space",
        text: "a\u{a0}b\u{3000}c\u{2009}d\u{200b}e",
        llama: &[
            1037, 1038, 1039, 2139,
        ],
    },
    Golden {
        name: "control-bytes",
        text: "a\u{1b}[31mred\u{1b}[0m b\u{c}c\u{7}d",
        llama: &[
            1037, 1031, 2861, 2213, 5596, 1031, 1014, 2213, 1038, 3729,
        ],
    },
    Golden {
        name: "long-words",
        text: "supercalifragilisticexpialidocious antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis",
        llama: &[
            3565, 9289, 10128, 29181, 24411, 4588, 10288, 19312, 21273,
            10085, 6313, 3424, 10521, 4355, 7875, 13602, 3672, 12199, 2964,
            1052, 2638, 2819, 17175, 11314, 6444, 2594, 7352, 26461, 27572,
            11261, 6767, 15472, 6761, 8663, 10735, 2483,
        ],
    },
    Golden {
        name: "single-space",
        text: " ",
        llama: &[],
    },
    Golden {
        name: "wordpiece-continuations",
        text: "unaffable tokenization embeddings biophysicist",
        llama: &[
            14477, 20961, 3468, 19204, 3989, 7861, 8270, 4667, 2015, 16012,
            21281, 19570, 2923,
        ],
    },
    Golden {
        name: "unknown-word",
        text: "zzzzqqqq xyzzyplugh",
        llama: &[
            1062, 13213, 2480, 4160, 4160, 4160, 4160, 1060, 2100, 28753,
            24759, 8953,
        ],
    },
    Golden {
        name: "mixed-case",
        text: "The Quick BROWN Fox JuMpEd",
        llama: &[
            1996, 4248, 2829, 4419, 5598,
        ],
    },
    Golden {
        name: "accents",
        text: "café naïve Zürich cafe\u{301} résumé",
        llama: &[
            7668, 15743, 10204, 7668, 13746,
        ],
    },
    Golden {
        name: "bracket-specials",
        text: "before [CLS] middle [SEP] after [MASK] end [UNK] [PAD]",
        llama: &[
            2077, 101, 2690, 102, 2044, 103, 2203, 100, 0,
        ],
    },
    Golden {
        name: "bracket-lookalike",
        text: "[NOTSPECIAL] [unused0] [ cls ]",
        llama: &[
            1031, 2025, 13102, 8586, 4818, 1033, 1031, 15171, 2692, 1033,
            1031, 18856, 2015, 1033,
        ],
    },
    Golden {
        name: "cjk-mixed",
        text: "東京tokyo日本",
        llama: &[
            1879, 1755, 5522, 1864, 1876,
        ],
    },
    Golden {
        name: "symbols",
        text: "a+b=c $5 100% x^2 ~y |z| €10 £5",
        llama: &[
            1037, 1009, 1038, 1027, 1039, 1002, 1019, 2531, 1003, 1060, 1034,
            1016, 1066, 1061, 1064, 1062, 1064, 1574, 10790, 27813,
        ],
    },
    Golden {
        name: "emails-urls",
        text: "test@example.com https://a.b/c?d=e#f",
        llama: &[
            3231, 1030, 2742, 1012, 4012, 16770, 1024, 1013, 1013, 1037,
            1012, 1038, 1013, 1039, 1029, 1040, 1027, 1041, 1001, 1042,
        ],
    },
    Golden {
        name: "numbers",
        text: "3.14159 1,000,000 -42 0x1F 2024-01-31",
        llama: &[
            1017, 1012, 15471, 28154, 1015, 1010, 2199, 1010, 2199, 1011,
            4413, 1014, 2595, 2487, 2546, 16798, 2549, 1011, 5890, 1011,
            2861,
        ],
    },
    Golden {
        name: "hangul",
        text: "서울 한국어",
        llama: &[
            1461, 29999, 1469, 29991, 29999,
        ],
    },
    Golden {
        name: "zero-width",
        text: "a\u{200b}b\u{feff}c\u{ad}d",
        llama: &[
            5925, 2094,
        ],
    },
    Golden {
        name: "empty-after-normalize",
        text: "\u{200b}\u{feff}",
        llama: &[],
    },
    Golden {
        name: "long-unknown-word",
        text: "supercalifragilisticexpialidociouszzzz",
        llama: &[
            3565, 9289, 10128, 29181, 24411, 4588, 10288, 19312, 21273,
            10085, 6313, 13213, 13213,
        ],
    },
    Golden {
        name: "repeated-punct",
        text: "!!! ??? ... --- ,,, ;;;",
        llama: &[
            999, 999, 999, 1029, 1029, 1029, 1012, 1012, 1012, 1011, 1011,
            1011, 1010, 1010, 1010, 1025, 1025, 1025,
        ],
    },
    Golden {
        name: "single-char-words",
        text: "a b c I A",
        llama: &[
            1037, 1038, 1039, 1045, 1037,
        ],
    },
    Golden {
        name: "partial-cover",
        text: "hello\u{1f680} tokenizing\u{1f680} world",
        llama: &[
            100, 100, 2088,
        ],
    },
];

fn model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/bge-small-en-v1.5-q8_0.gguf")
}

/// The reference dumper, overridable with `FERROX_LLAMA_LOGITS`.
///
/// The override exists because the interesting question here is which
/// llama.cpp is answering: `./tools/build_llama_logits.sh` links whatever
/// libllama is installed, and pointing this at a build made from the
/// revision in `.scratch/llama.cpp` is how the recorded table was
/// produced in the first place.
fn dumper_path() -> PathBuf {
    match std::env::var_os("FERROX_LLAMA_LOGITS") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/llama_logits"),
    }
}

fn tokenizer() -> Option<GgufWordPieceTokenizer> {
    let path = model_path();
    if !path.exists() {
        eprintln!("skip: missing {path:?}");
        return None;
    }
    let file = ShardedGguf::open(&path).expect("open the bge GGUF");
    assert_eq!(
        file.metadata_str("tokenizer.ggml.model"),
        Some("bert"),
        "this test is about the WordPiece path; a file that is not `bert` \
         would exercise something else and still pass"
    );
    assert!(
        ferrox_models::tokenizer::should_add_bos_token(&file),
        "llama.cpp's WPM arm sets add_bos = true, so ferrox must too"
    );
    Some(GgufWordPieceTokenizer::from_gguf(&file).expect("build the tokenizer"))
}

/// Where two id sequences first disagree, rendered with the pieces on
/// both sides so a failure can be debugged from the output alone.
fn report(
    tok: &GgufWordPieceTokenizer,
    name: &str,
    text: &str,
    llama: &[u32],
    ferrox: &[u32],
) -> String {
    let at = llama
        .iter()
        .zip(ferrox)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| llama.len().min(ferrox.len()));
    let window = |ids: &[u32]| -> String {
        let lo = at.saturating_sub(3);
        let hi = (at + 4).min(ids.len());
        ids[lo.min(ids.len())..hi]
            .iter()
            .map(|&id| format!("{id}:{:?}", tok.decode(&[id])))
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!(
        "[{name}] diverges at token {at} of {} (llama) / {} (ferrox)\n  \
         text   {text:?}\n  llama  {}\n  ferrox {}",
        llama.len(),
        ferrox.len(),
        window(llama),
        window(ferrox),
    )
}

/// The evidence. Ferrox's ids, against llama.cpp's, on every case.
#[test]
#[ignore = "needs models/bge-small-en-v1.5-q8_0.gguf"]
fn ferrox_matches_the_recorded_llama_cpp_tokenization() {
    let Some(tok) = tokenizer() else { return };
    assert_eq!(
        tok.vocab_size(),
        30522,
        "llama.cpp reports the same n_vocab"
    );

    let mut failures = Vec::new();
    let mut tokens = 0usize;
    for case in GOLDEN {
        let ferrox = tok.encode(case.text, SpecialTokens::Parse);
        tokens += case.llama.len();
        if ferrox != case.llama {
            failures.push(report(&tok, case.name, case.text, case.llama, &ferrox));
        }
    }

    // Printed on success too: a corpus that tokenized to nothing would
    // also "agree", and this is the line that says work was done.
    assert_eq!(tokens, 511, "the recorded corpus is 511 llama.cpp tokens");
    assert_eq!(GOLDEN.len(), 36, "the recorded corpus is 36 cases");
    assert!(
        failures.is_empty(),
        "{} of {} cases diverge from llama.cpp:\n\n{}",
        failures.len(),
        GOLDEN.len(),
        failures.join("\n\n")
    );
}

/// The corpus is the coverage claim, so it is asserted rather than
/// trusted. Trimming it to the cases that pass would leave the two tests
/// above still green while proving less.
#[test]
fn the_corpus_covers_what_wordpiece_can_get_wrong() {
    let text = |name: &str| -> &'static str {
        GOLDEN
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("corpus case '{name}' was removed"))
            .text
    };

    // A word that must split into a word-initial piece plus bare
    // continuations. This is the `##` rule, and it is the one thing a
    // greedy longest-match loop over an unprepared vocabulary would get
    // wrong on every single word.
    assert!(text("wordpiece-continuations").contains("unaffable"));
    // A word no sequence of pieces covers, which must come back as ONE
    // unknown rather than a partial cover plus an unknown.
    assert!(text("unknown-word").contains("zzzzqqqq"));
    assert!(
        text("long-unknown-word").len() > 30,
        "a long uncoverable word"
    );
    // A word whose PREFIX is coverable and whose tail is not. This is
    // the only input that separates the all-or-nothing rule from a
    // per-position fallback, and without it a build that kept the
    // already-emitted pieces matched the whole rest of this corpus.
    assert!(
        text("partial-cover").starts_with("hello\u{1f680}"),
        "a coverable prefix welded to an uncoverable tail"
    );
    // The normalizer's four switches.
    assert!(text("mixed-case").contains("BROWN"), "lowercasing");
    assert!(text("accents").contains('é'), "a precomposed accent");
    assert!(
        text("accents").contains("cafe\u{301}"),
        "a STANDALONE combining mark, which folds by a different route \
         than the precomposed one"
    );
    assert!(text("zero-width").contains('\u{200b}'), "a Cf control");
    assert!(
        text("empty-after-normalize")
            .chars()
            .all(|c| c == '\u{200b}' || c == '\u{feff}'),
        "an input that normalizes away to no words at all"
    );
    // Special tokens, and text that only looks like one.
    assert!(text("bracket-specials").contains("[CLS]"));
    assert!(text("bracket-lookalike").contains("[NOTSPECIAL]"));
    // The single-character-word rules: punctuation, ASCII symbols, CJK.
    assert!(text("repeated-punct").contains("!!!"));
    assert!(
        text("symbols").contains('€'),
        "a non-ASCII symbol, which does NOT split"
    );
    assert!(text("symbols").contains('+'), "an ASCII symbol, which does");
    assert!(text("cjk-mixed").contains('東'));
    assert!(text("hangul").contains('서'));

    let mut names: Vec<&str> = GOLDEN.iter().map(|c| c.name).collect();
    names.sort_unstable();
    let n = names.len();
    names.dedup();
    assert_eq!(names.len(), n, "corpus case names must be unique");
    // Two cases tokenize to nothing, and both do so on purpose: an
    // input that is one space, and one that is nothing but zero-width
    // controls. Naming them here means a THIRD case going empty is a
    // failure rather than a quiet loss of coverage.
    let empty: Vec<&str> = GOLDEN
        .iter()
        .filter(|c| c.llama.is_empty())
        .map(|c| c.name)
        .collect();
    assert_eq!(
        empty,
        ["single-space", "empty-after-normalize"],
        "only the two deliberately-empty cases may tokenize to nothing"
    );
}

/// Two texts prepended to the batch, to ask the reference a question
/// about ITSELF before its answers are used as evidence.
///
/// `"e"` and `"e"` plus a standalone U+0301 differ only by a combining
/// mark. A reference that implements `BertNormalizer`'s `strip_accents`
/// drops the mark and gives both the same id; one that predates it welds
/// the mark into the word and gives `[UNK]` for the second. That is a
/// capability question with a yes/no answer, not a list of the cases
/// that happen to disagree, and it is asked without naming any corpus
/// case.
const PROBE: [&str; 2] = ["e", "e\u{301}"];

/// Re-derives the table above from llama.cpp itself.
///
/// Without this the frozen ids would decay into a record of ferrox
/// agreeing with ferrox.
///
/// A libllama older than the recorded revision implements a different
/// normalizer and therefore answers a different question, so this SKIPS
/// on one rather than reporting either a pass or a failure. That is the
/// same call `ferrox parity` makes when llama.cpp cannot load a
/// checkpoint: a reference with no answer is not evidence in either
/// direction. The frozen table above is still checked in full by the
/// test at the top, which needs no libllama at all.
#[test]
#[ignore = "needs models/bge-small-en-v1.5-q8_0.gguf and target/llama_logits"]
fn the_recorded_tokenization_is_still_what_llama_cpp_produces() {
    let model = model_path();
    let dumper = dumper_path();
    if !model.exists() {
        eprintln!("skip: missing {model:?}");
        return;
    }
    if !dumper.exists() {
        eprintln!("skip: missing {dumper:?}, build it with ./tools/build_llama_logits.sh");
        return;
    }
    let Some(tok) = tokenizer() else { return };

    let texts: Vec<&str> = PROBE
        .iter()
        .copied()
        .chain(GOLDEN.iter().map(|c| c.text))
        .collect();
    let (flags, n_vocab, answers) = reference_tokenize(&dumper, &model, &texts);
    assert_eq!(
        n_vocab,
        tok.vocab_size(),
        "the two id spaces must be the same"
    );
    assert!(flags & 1 != 0, "llama.cpp's WPM arm sets add_bos");
    assert_eq!(answers.len(), texts.len());

    if answers[0] != answers[1] {
        eprintln!(
            "skip: this libllama predates BertNormalizer's strip_accents switch \
             ({:?} tokenizes as {:?} and {:?} as {:?}), so it implements a \
             different normalizer than the revision the recorded table came \
             from. Rebuild ./tools/build_llama_logits.sh against a newer \
             llama.cpp to run this.",
            PROBE[0], answers[0], PROBE[1], answers[1]
        );
        return;
    }

    let mut failures = Vec::new();
    for (case, live) in GOLDEN.iter().zip(&answers[PROBE.len()..]) {
        if live != case.llama {
            failures.push(report(&tok, case.name, case.text, live, case.llama));
        }
    }
    assert!(
        failures.is_empty(),
        "the recorded table is not what this libllama produces, and the \
         accent probe says the two implement the same normalizer, so this is \
         a real change. \"llama\" is the live answer, \"ferrox\" is the \
         recorded one.\n\n{}",
        failures.join("\n\n")
    );
}

/// Runs the dumper's `--tokenize` mode over every text in one
/// invocation, and returns `(flags, n_vocab, ids per text)`.
fn reference_tokenize(dumper: &Path, model: &Path, texts: &[&str]) -> (u32, usize, Vec<Vec<u32>>) {
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let in_path = dir.join(format!("ferrox-wpm-cases-{pid}.bin"));
    let out_path = dir.join(format!("ferrox-wpm-toks-{pid}.bin"));

    // The FXTK case file `tools/llama_logits.c` reads: length-prefixed,
    // because the corpus holds newlines and control bytes on purpose.
    let mut blob = Vec::new();
    blob.extend_from_slice(b"FXTK");
    blob.extend_from_slice(&(texts.len() as u32).to_le_bytes());
    for text in texts {
        blob.extend_from_slice(&(text.len() as u32).to_le_bytes());
        blob.extend_from_slice(text.as_bytes());
    }
    std::fs::write(&in_path, blob).expect("write the case file");

    let out = Command::new(dumper)
        .arg("--tokenize")
        .arg(model)
        .arg(&in_path)
        .arg(&out_path)
        .output()
        .expect("run the reference tokenizer");
    let _ = std::fs::remove_file(&in_path);
    assert!(
        out.status.success(),
        "reference tokenizer failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&out_path).expect("read the reference output");
    let _ = std::fs::remove_file(&out_path);
    parse_fxtk(&bytes)
}

/// The FXTK result format `tools/llama_logits.c` writes:
/// `"FXTK" | u32 version=2 | u32 flags | u32 n_vocab | u32 n_cases |
/// repeat( run(parse_special=false) run(parse_special=true) )`, each
/// run `u32 n_ids | n_ids * u32`. Only the parsed run is kept, because
/// that is the setting the recorded table was made under.
fn parse_fxtk(bytes: &[u8]) -> (u32, usize, Vec<Vec<u32>>) {
    assert_eq!(&bytes[..4], b"FXTK", "not an FXTK result file");
    let mut at = 4usize;
    let mut next = || {
        let v = u32::from_le_bytes(bytes[at..at + 4].try_into().expect("truncated FXTK"));
        at += 4;
        v
    };
    assert_eq!(
        next(),
        2,
        "this test reads FXTK v2; rebuild the dumper with ./tools/build_llama_logits.sh"
    );
    let flags = next();
    let n_vocab = next() as usize;
    let n_cases = next() as usize;
    let cases = (0..n_cases)
        .map(|_| {
            let n_as_text = next() as usize;
            for _ in 0..n_as_text {
                next();
            }
            let n = next() as usize;
            (0..n).map(|_| next()).collect()
        })
        .collect();
    (flags, n_vocab, cases)
}
