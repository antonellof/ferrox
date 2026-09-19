//! Tests for the stack machine and candidate rejection.
//!
//! The grammars are the ones llama.cpp ships in `grammars/`, reused from
//! [`super::parser_tests`], plus small hand-written ones for the corners.
//!
//! The strongest test here is
//! [`the_two_acceptance_paths_agree`]: `reject_candidates` (a shared-prefix
//! walk over all candidates at once) and `Grammar::accept_token` (a
//! per-stack replay) are independent implementations of the same
//! predicate, and are checked against each other over every grammar and a
//! spread of pieces. The one piece they read differently is the empty
//! string, and that is upstream's behaviour too --- see
//! [`the_empty_piece_is_the_one_case_the_two_paths_read_differently`].

use super::candidates::{accepts_token, reject_candidates, Candidate};
use super::error::GrammarError;
use super::grammars::{ARITHMETIC_GBNF, JAPANESE_GBNF, JSON_GBNF, LIST_GBNF, SHIPPED_GRAMMARS};
use super::machine::Grammar;
use super::utf8::PartialUtf8;

fn grammar(src: &str) -> Grammar {
    Grammar::from_str_with_root(src, "root")
        .unwrap_or_else(|e| panic!("failed to build a machine for the grammar: {e}"))
}

/// Feed a whole string through `accept_str`, reporting where it died.
fn feed(g: &mut Grammar, text: &str) -> Result<(), GrammarError> {
    g.accept_str(text)
}

/// Does this grammar accept exactly this complete string?
fn accepts_whole(src: &str, text: &str) -> bool {
    let mut g = grammar(src);
    match feed(&mut g, text) {
        Ok(()) => g.allows_eog(),
        Err(_) => false,
    }
}

// -- the thing json_mode.rs cannot do --

#[test]
fn json_grammar_tracks_brace_depth() {
    assert!(accepts_whole(JSON_GBNF, r#"{"a": 1}"#));
    assert!(accepts_whole(JSON_GBNF, r#"{"a": {"b": [1, 2, null]}}"#));
    assert!(accepts_whole(JSON_GBNF, "{}"));

    // Every one of these passes `json_mode.rs`'s character filter: each
    // is built only from JSON-safe characters. Only a machine that knows
    // what was opened can refuse them.
    assert!(
        !accepts_whole(JSON_GBNF, r#"{"a": 1}}"#),
        "extra close brace"
    );
    assert!(!accepts_whole(JSON_GBNF, r#"{"a": 1"#), "unclosed object");
    assert!(
        !accepts_whole(JSON_GBNF, r#"{"a": [1, 2}"#),
        "bracket closed by a brace"
    );
    assert!(!accepts_whole(JSON_GBNF, r#"{"a" 1}"#), "missing colon");
    assert!(!accepts_whole(JSON_GBNF, r#"{"a": ,}"#), "missing value");
    assert!(!accepts_whole(JSON_GBNF, r#"{a: 1}"#), "unquoted key");
}

#[test]
fn json_grammar_refuses_a_close_brace_as_the_very_first_character() {
    let mut g = grammar(JSON_GBNF);
    let err = feed(&mut g, "}").unwrap_err();
    assert!(matches!(err, GrammarError::NoViableStack { .. }), "{err:?}");
    assert!(g.is_dead());
}

#[test]
fn a_complete_json_object_allows_end_of_generation_and_an_incomplete_one_does_not() {
    let mut g = grammar(JSON_GBNF);
    feed(&mut g, "{").unwrap();
    assert!(!g.allows_eog(), "an open object is not a finished parse");
    assert!(g.accept_eog().is_err());

    feed(&mut g, "}").unwrap();
    assert!(g.allows_eog(), "a closed object is");
    assert!(g.accept_eog().is_ok());
}

// -- the other shipped grammars --

#[test]
fn arithmetic_grammar_accepts_its_own_examples() {
    assert!(accepts_whole(ARITHMETIC_GBNF, "1+2=3\n"));
    assert!(accepts_whole(ARITHMETIC_GBNF, "x = (a+b)\n"));
    assert!(
        !accepts_whole(ARITHMETIC_GBNF, "1+2=3"),
        "no trailing newline"
    );
    assert!(!accepts_whole(ARITHMETIC_GBNF, "1+2\n"), "no '='");
    assert!(
        !accepts_whole(ARITHMETIC_GBNF, "1+2=(3\n"),
        "unclosed paren"
    );
    // `root ::= expr "=" ws term "\n"`, so the right-hand side is a single
    // TERM, not an expr: `(a+b) * 2` is two terms and does not fit.
    assert!(!accepts_whole(ARITHMETIC_GBNF, "x = (a+b) * 2\n"));
}

#[test]
fn list_grammar_needs_the_bullet_and_the_newline() {
    assert!(accepts_whole(LIST_GBNF, "- one\n- two\n"));
    assert!(!accepts_whole(LIST_GBNF, "one\n"), "no bullet");
    assert!(!accepts_whole(LIST_GBNF, "- one"), "no newline");
    // `[^\r\n\x0b\x0c\x85  ]` -- a negated class with several
    // members. Every member must be excluded, not just the first.
    assert!(!accepts_whole(LIST_GBNF, "- a\rb\n"), "\\r is excluded");
    assert!(
        !accepts_whole(LIST_GBNF, "- a\u{0b}b\n"),
        "\\x0b is excluded"
    );
    assert!(
        !accepts_whole(LIST_GBNF, "- a\u{2028}b\n"),
        "U+2028 is excluded"
    );
    assert!(accepts_whole(LIST_GBNF, "- a\u{2027}b\n"), "U+2027 is not");
}

#[test]
fn every_shipped_grammar_builds_a_machine_with_at_least_one_stack() {
    for (name, src, _) in SHIPPED_GRAMMARS {
        let g = Grammar::from_str_with_root(src, "root").unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!g.stacks().is_empty(), "{name} starts with no viable stack");
    }
}

// -- character classes --

#[test]
fn a_negated_class_excludes_every_member_not_just_the_first() {
    // The negation lives on the first element of the class alone. A
    // machine that tested each element against its own type would read
    // `[^ab]` as "not a, or not b", which every character satisfies.
    assert!(!accepts_whole("root ::= [^ab]", "a"));
    assert!(!accepts_whole("root ::= [^ab]", "b"));
    assert!(accepts_whole("root ::= [^ab]", "c"));
}

#[test]
fn a_negated_range_excludes_the_whole_range() {
    assert!(!accepts_whole("root ::= [^1-3]", "1"));
    assert!(!accepts_whole("root ::= [^1-3]", "2"));
    assert!(!accepts_whole("root ::= [^1-3]", "3"));
    assert!(accepts_whole("root ::= [^1-3]", "4"));
    assert!(accepts_whole("root ::= [^1-3]", "0"));
}

#[test]
fn a_class_mixing_singles_and_ranges_accepts_all_of_them() {
    for c in ["b", "d", "x", "y", "z"] {
        assert!(accepts_whole("root ::= [bdx-z]", c), "{c} should match");
    }
    for c in ["a", "c", "w"] {
        assert!(
            !accepts_whole("root ::= [bdx-z]", c),
            "{c} should not match"
        );
    }
}

#[test]
fn dot_matches_any_character_including_a_multibyte_one() {
    assert!(accepts_whole("root ::= .", "a"));
    assert!(accepts_whole("root ::= .", "\n"));
    assert!(accepts_whole("root ::= .", "€"));
    assert!(!accepts_whole("root ::= .", "ab"), "exactly one");
}

#[test]
fn multibyte_ranges_bound_correctly() {
    // hiragana ::= [ぁ-ゟ] from japanese.gbnf. U+3041..U+309F.
    assert!(accepts_whole(JAPANESE_GBNF, "ひらがな"));
    assert!(accepts_whole(JAPANESE_GBNF, "カタカナ"));
    assert!(accepts_whole(JAPANESE_GBNF, "漢字"));
    assert!(!accepts_whole(JAPANESE_GBNF, "€"), "outside every class");
}

// -- partial UTF-8, the reason a token can end mid-codepoint --

#[test]
fn a_token_that_ends_mid_codepoint_stays_viable() {
    // "ひ" is E3 81 B2. Split it the way a BPE vocabulary would. Neither
    // half is valid UTF-8 on its own, which is why the API takes bytes.
    let hira = "ひ".as_bytes();
    let mut g = grammar(JAPANESE_GBNF);
    g.accept_bytes(&hira[..1])
        .expect("a partial codepoint must survive");
    assert_eq!(g.partial_utf8().n_remain, 2);
    assert!(!g.is_dead());
    g.accept_bytes(&hira[1..]).expect("and complete");
    assert_eq!(g.partial_utf8().n_remain, 0);
    assert!(g.allows_eog());
}

#[test]
fn a_partial_codepoint_that_cannot_complete_into_the_class_is_rejected() {
    // `root ::= [a-z]` can never be satisfied by any multi-byte
    // character, so a token holding the lead byte of one is impossible.
    let g = grammar("root ::= [a-z]+");
    let euro_lead = &"€".as_bytes()[..1];
    assert!(
        !accepts_token(&g, 1, euro_lead).unwrap(),
        "no continuation of a 3-byte lead lands in [a-z]"
    );
    // Whereas under a grammar whose class covers that range, it is viable.
    let g2 = grammar("root ::= [\\u2000-\\u2100]+");
    assert!(accepts_token(&g2, 1, euro_lead).unwrap());
}

#[test]
fn an_overlong_two_byte_partial_is_refused() {
    // `n_remain == 1 && value < 2` is upstream's overlong guard: a 7-bit
    // character encoded in two bytes. Nothing may complete it.
    let g = grammar("root ::= .");
    // 0xC0 leads an overlong encoding, which is never valid UTF-8.
    assert!(!accepts_token(&g, 1, &[0xC0u8]).unwrap());
}

#[test]
fn a_hand_built_partial_with_an_impossible_length_is_refused_not_a_panic() {
    // `n_remain > 3` cannot come out of `decode_piece`, but `PartialUtf8`
    // is constructible. Upstream would shift by more than 31 bits.
    use super::element::RulePos;
    use super::machine::match_partial_char;
    let g = grammar("root ::= [a-z]");
    let verdict = match_partial_char(g.rules(), RulePos::new(0, 0), PartialUtf8::new(1, 9));
    assert_eq!(verdict, Ok(false));
}

// -- token elements --

#[test]
fn a_token_element_matches_on_id_not_on_text() {
    let mut g = grammar("root ::= <[1000]> \"a\"");
    // The piece is irrelevant; the id is what matters.
    g.accept_token(1000, b"anything at all")
        .expect("id matches");
    g.accept_str("a").expect("then the literal");
    assert!(g.allows_eog());

    let mut g2 = grammar("root ::= <[1000]> \"a\"");
    let err = g2.accept_token(1001, b"anything at all").unwrap_err();
    assert!(matches!(err, GrammarError::NoViableStack { .. }), "{err:?}");
}

#[test]
fn an_inverted_token_element_matches_everything_but_its_id() {
    let mut g = grammar("root ::= !<[1001]>");
    g.accept_token(7, b"x").expect("7 is not 1001");
    assert!(g.allows_eog());

    let mut g2 = grammar("root ::= !<[1001]>");
    assert!(g2.accept_token(1001, b"x").is_err());
}

#[test]
fn accept_str_cannot_satisfy_a_token_element() {
    // A stack resting on a token element consumes a token, never a
    // character, so plain text kills it.
    let mut g = grammar("root ::= <[1000]>");
    assert!(g.accept_str("a").is_err());
}

#[test]
fn candidate_rejection_over_a_token_element_uses_the_id() {
    let g = grammar("root ::= <[1000]> \"a\"");
    let cands = [
        Candidate::new(0, 1000, b"whatever"),
        Candidate::new(1, 1001, b"whatever"),
        Candidate::new(2, 1000, b"a"),
    ];
    let mut rejected = reject_candidates(&g, &cands).unwrap();
    rejected.sort_unstable();
    assert_eq!(rejected, vec![1]);
}

// -- left recursion --

#[test]
fn direct_left_recursion_is_refused() {
    let err = Grammar::from_str_with_root("root ::= root \"a\" | \"a\"", "root").unwrap_err();
    assert!(matches!(err, GrammarError::LeftRecursion { .. }), "{err:?}");
}

#[test]
fn indirect_left_recursion_is_refused() {
    let err = Grammar::from_str_with_root("root ::= a \"x\"\na ::= root \"y\" | \"y\"", "root")
        .unwrap_err();
    assert!(matches!(err, GrammarError::LeftRecursion { .. }), "{err:?}");
}

#[test]
fn left_recursion_through_a_nullable_prefix_is_refused() {
    // `opt` may be empty, so `root` reaches itself without consuming.
    // This is the case the `rules_may_be_empty` pass exists for; a
    // detector that only looked at the first element would miss it.
    let err =
        Grammar::from_str_with_root("root ::= opt root \"a\" | \"a\"\nopt ::= \"b\" |", "root")
            .unwrap_err();
    assert!(matches!(err, GrammarError::LeftRecursion { .. }), "{err:?}");
}

#[test]
fn right_recursion_is_fine() {
    assert!(accepts_whole("root ::= \"a\" root | \"a\"", "aaaa"));
}

#[test]
fn a_missing_root_is_named() {
    let err = Grammar::from_str_with_root("start ::= \"a\"", "root").unwrap_err();
    match err {
        GrammarError::MissingRoot { name } => assert_eq!(name, "root"),
        other => panic!("expected MissingRoot, got {other:?}"),
    }
}

// -- candidate rejection --

#[test]
fn rejection_masks_exactly_the_impossible_tokens() {
    let g = grammar(JSON_GBNF);
    // At the start, json.gbnf can only open an object.
    let pieces = ["{", "}", "[", "\"", "1", " ", "{\"", "null"];
    let cands: Vec<Candidate> = pieces
        .iter()
        .enumerate()
        .map(|(i, p)| Candidate::new(i, i as u32, p.as_bytes()))
        .collect();
    let mut rejected = reject_candidates(&g, &cands).unwrap();
    rejected.sort_unstable();
    // Only "{" and "{\"" are possible openings.
    assert_eq!(rejected, vec![1, 2, 3, 4, 5, 7]);
}

#[test]
fn rejection_is_empty_when_every_candidate_is_possible() {
    let g = grammar("root ::= [a-c]+");
    let cands = [
        Candidate::new(0, 0, b"a"),
        Candidate::new(1, 1, b"b"),
        Candidate::new(2, 2, b"abc"),
    ];
    assert!(reject_candidates(&g, &cands).unwrap().is_empty());
}

#[test]
fn a_multi_character_token_is_rejected_if_any_of_its_characters_is() {
    // The whole point of walking the piece: "ab" is fine, "ax" is not,
    // even though "a" alone is.
    let g = grammar("root ::= \"ab\" \"cd\"");
    assert!(accepts_token(&g, 0, b"ab").unwrap());
    assert!(accepts_token(&g, 0, b"abc").unwrap());
    assert!(accepts_token(&g, 0, b"abcd").unwrap());
    assert!(!accepts_token(&g, 0, b"ax").unwrap());
    assert!(!accepts_token(&g, 0, b"abce").unwrap());
    assert!(!accepts_token(&g, 0, b"abcde").unwrap(), "one past the end");
}

#[test]
fn a_satisfied_grammar_rejects_every_token_that_adds_text() {
    let mut g = grammar("root ::= \"a\"");
    g.accept_str("a").unwrap();
    assert!(g.allows_eog());
    let cands = [Candidate::new(0, 0, b"b"), Candidate::new(1, 1, b"")];
    let mut rejected = reject_candidates(&g, &cands).unwrap();
    rejected.sort_unstable();
    // The empty piece adds nothing, so the finished stack keeps it. The
    // sampler hook is what masks empty pieces; the grammar does not.
    assert_eq!(rejected, vec![0]);
}

#[test]
fn a_dead_grammar_rejects_everything() {
    let mut g = grammar("root ::= \"a\"");
    let _ = g.accept_str("z");
    assert!(g.is_dead());
    let cands = [Candidate::new(0, 0, b"a"), Candidate::new(1, 1, b"b")];
    let mut rejected = reject_candidates(&g, &cands).unwrap();
    rejected.sort_unstable();
    assert_eq!(rejected, vec![0, 1]);
}

#[test]
fn nested_alternation_keeps_more_than_one_viable_stack() {
    // `advance_stack` must expand a rule reference into every alternate.
    // With only the first alternate expanded, "b" would be impossible.
    let g = grammar("root ::= choice \"!\"\nchoice ::= \"a\" | \"b\" | \"c\"");
    assert!(g.stacks().len() >= 3, "got {:?}", g.stacks());
    for c in ["a", "b", "c"] {
        assert!(accepts_whole(
            "root ::= choice \"!\"\nchoice ::= \"a\" | \"b\" | \"c\"",
            &format!("{c}!")
        ));
    }
    assert!(!accepts_token(&g, 0, b"d").unwrap());
}

#[test]
fn the_empty_piece_is_the_one_case_the_two_paths_read_differently() {
    // On a fully satisfied grammar `reject_candidates` keeps the empty
    // piece (it adds no code points, so the finished stack survives it)
    // while `accept_token` drops every empty stack and then finds nothing
    // left. **Upstream does exactly the same**: `accept_token`'s loop
    // `continue`s on an empty stack, and `reject_candidates_for_stack`'s
    // empty-stack branch rejects only tokens that add something.
    //
    // It is harmless because llama.cpp's sampler masks an empty piece
    // unconditionally, before the grammar sees it
    // (`llama_grammar_apply_impl`: `piece.empty() || piece[0] == 0`). That
    // rule belongs to the sampler hook, which is why it is not in
    // `candidates.rs`. This test exists so the asymmetry is recorded
    // rather than discovered again.
    let mut g = grammar("root ::= \"a\"");
    g.accept_str("a").unwrap();
    assert!(g.stacks().iter().all(|s| s.is_empty()), "fully satisfied");

    assert!(
        accepts_token(&g, 0, b"").unwrap(),
        "the batch path keeps the empty piece"
    );
    assert!(
        g.clone().accept_token(0, b"").is_err(),
        "the replay path drops the finished stack and dies"
    );
}

/// The cross-check. `reject_candidates` and `accept_token` share the
/// element types and nothing else: one walks all candidates together
/// against the stack set, the other replays one piece through a clone of
/// the machine. They must agree on every piece.
#[test]
fn the_two_acceptance_paths_agree() {
    let pieces = [
        "a", "b", "z", "{", "}", "[", "]", "\"", ":", ",", " ", "\n", "0", "1", "9", "-", "+", "e",
        "true", "null", "{\"a", "\": ", "ab", "abc", "- ", "1. ", "€", "ひ", "\t",
    ];
    let mut checked = 0usize;
    for (name, src, _) in SHIPPED_GRAMMARS {
        // Walk the grammar forward through a few accepted pieces so the
        // comparison covers mid-parse states, not only the start state.
        let mut g =
            Grammar::from_str_with_root(src, "root").unwrap_or_else(|e| panic!("{name}: {e}"));
        for step in 0..4 {
            for piece in pieces {
                let batch = accepts_token(&g, 0, piece.as_bytes())
                    .unwrap_or_else(|e| panic!("{name} step {step} piece {piece:?}: {e}"));
                let mut replay = g.clone();
                let replayed = replay.accept_token(0, piece.as_bytes()).is_ok();
                assert_eq!(
                    batch, replayed,
                    "{name} step {step}: reject_candidates and accept_token disagree on {piece:?}"
                );
                checked += 1;
            }
            // Advance by the first piece the grammar will take.
            let Some(next) = pieces
                .iter()
                .find(|p| accepts_token(&g, 0, p.as_bytes()).unwrap_or(false))
            else {
                break;
            };
            g.accept_token(0, next.as_bytes())
                .expect("just checked it is viable");
        }
    }
    assert!(checked > 500, "only {checked} comparisons ran");
}
