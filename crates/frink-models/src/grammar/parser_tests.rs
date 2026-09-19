//! Golden rule tables transcribed from llama.cpp's
//! `tests/test-grammar-parser.cpp`.
//!
//! Upstream pins the compiled *table*, not the accepted language, because
//! the synthesized rule ids are observable. Each case here carries both
//! halves of upstream's expectation: the symbol-name-to-id map (in sorted
//! order, since upstream iterates a `std::map`) and the element sequence
//! of every rule.
//!
//! If one of these fails after a change to `parser.rs`, the change altered
//! the compiled form of a grammar that llama.cpp compiles differently. It
//! is not enough that the language still matches.

use super::element::{GrammarElement, GreType};
use super::error::GrammarError;
use super::grammars::SHIPPED_GRAMMARS;
use super::parser::{parse, parse_with_vocab, GrammarVocab, MAX_REPETITION_THRESHOLD};

// -- element constructors, mirroring the LLAMA_GRETYPE_* spellings --

fn ch(c: char) -> GrammarElement {
    GrammarElement::new(GreType::Char, c as u32)
}
fn cnot(c: char) -> GrammarElement {
    GrammarElement::new(GreType::CharNot, c as u32)
}
fn calt(c: char) -> GrammarElement {
    GrammarElement::new(GreType::CharAlt, c as u32)
}
fn crng(c: char) -> GrammarElement {
    GrammarElement::new(GreType::CharRngUpper, c as u32)
}
fn any() -> GrammarElement {
    GrammarElement::new(GreType::CharAny, 0)
}
fn rr(id: u32) -> GrammarElement {
    GrammarElement::new(GreType::RuleRef, id)
}
fn alt() -> GrammarElement {
    GrammarElement::new(GreType::Alt, 0)
}
fn end() -> GrammarElement {
    GrammarElement::new(GreType::End, 0)
}
fn tok(id: u32) -> GrammarElement {
    GrammarElement::new(GreType::Token, id)
}
fn tnot(id: u32) -> GrammarElement {
    GrammarElement::new(GreType::TokenNot, id)
}

/// `verify_parsing`: assert the symbol table and every rule.
fn verify(src: &str, symbols: &[(&str, u32)], rules: &[&[GrammarElement]]) {
    let g = parse(src).unwrap_or_else(|e| panic!("failed to parse {src:?}: {e}"));
    let got: Vec<(&str, u32)> = g.symbol_ids.iter().map(|(k, &v)| (k.as_str(), v)).collect();
    assert_eq!(got, symbols, "symbol table mismatch for {src:?}");
    assert_eq!(
        g.rules.len(),
        rules.len(),
        "rule count mismatch for {src:?}: {:#?}",
        g.rules
    );
    for (i, (actual, expected)) in g.rules.iter().zip(rules.iter()).enumerate() {
        assert_eq!(
            actual.as_slice(),
            *expected,
            "rule {i} ({}) mismatch for {src:?}",
            g.symbol_name(i as u32).unwrap_or("?")
        );
    }
}

/// `verify_failure`: the grammar must be refused, and must say why.
fn verify_failure(src: &str) -> GrammarError {
    match parse(src) {
        Ok(g) => panic!("expected {src:?} to be refused, got {:#?}", g.rules),
        Err(e) => {
            assert!(e.to_string().len() > 20, "thin message: {e}");
            e
        }
    }
}

// -- upstream's verify_failure cases --

#[test]
fn unclosed_repetition_brace_is_refused() {
    let e = verify_failure("\n        root ::= \"a\"{,}\"\n    ");
    assert!(
        matches!(&e, GrammarError::Syntax { expected, .. } if expected.contains("int")),
        "{e:?}"
    );
}

#[test]
fn repetition_with_no_min_is_refused() {
    let e = verify_failure("\n        root ::= \"a\"{,10}\"\n    ");
    assert!(
        matches!(&e, GrammarError::Syntax { expected, .. } if expected.contains("int")),
        "{e:?}"
    );
}

#[test]
fn nested_repetitions_that_multiply_out_are_refused() {
    // Upstream's second verify_failure case. 99^k rules is the explosion
    // MAX_REPETITION_THRESHOLD exists to stop.
    let e = verify_failure(
        "\n        root ::= (((((([^x]*){0,99}){0,99}){0,99}){0,99}){0,99}){0,99}\n    ",
    );
    match e {
        GrammarError::RepetitionTooLarge {
            requested, limit, ..
        } => {
            assert!(requested >= limit);
            assert_eq!(limit, MAX_REPETITION_THRESHOLD);
        }
        other => panic!("expected a repetition-size refusal, got {other:?}"),
    }
}

// -- upstream's verify_parsing cases, in order --

#[test]
fn single_literal_char() {
    verify(
        "\n        root  ::= \"a\"\n    ",
        &[("root", 0)],
        &[&[ch('a'), end()]],
    );
}

#[test]
fn alternation_of_literal_and_two_char_classes() {
    // `[bdx-z]`: only the first element carries the class type, the rest
    // are CHAR_ALT, and `x-z` becomes CHAR_ALT 'x' + CHAR_RNG_UPPER 'z'.
    // `[^1-3]` is CHAR_NOT '1' + CHAR_RNG_UPPER '3' -- the negation lives
    // on the first element alone.
    verify(
        "\n        root  ::= \"a\" | [bdx-z] | [^1-3]\n    ",
        &[("root", 0)],
        &[&[
            ch('a'),
            alt(),
            ch('b'),
            calt('d'),
            calt('x'),
            crng('z'),
            alt(),
            cnot('1'),
            crng('3'),
            end(),
        ]],
    );
}

#[test]
fn plus_on_a_rule_reference() {
    verify(
        "\n        root  ::= a+\n        a     ::= \"a\"\n    ",
        &[("a", 1), ("root", 0), ("root_2", 2)],
        &[
            &[rr(1), rr(2), end()],
            &[ch('a'), end()],
            &[rr(1), rr(2), alt(), end()],
        ],
    );
}

#[test]
fn plus_on_a_literal() {
    verify(
        "\n        root  ::= \"a\"+\n    ",
        &[("root", 0), ("root_1", 1)],
        &[&[ch('a'), rr(1), end()], &[ch('a'), rr(1), alt(), end()]],
    );
}

#[test]
fn question_on_a_rule_reference() {
    verify(
        "\n        root  ::= a?\n        a     ::= \"a\"\n    ",
        &[("a", 1), ("root", 0), ("root_2", 2)],
        &[&[rr(2), end()], &[ch('a'), end()], &[rr(1), alt(), end()]],
    );
}

#[test]
fn question_on_a_literal() {
    verify(
        "\n        root  ::= \"a\"?\n    ",
        &[("root", 0), ("root_1", 1)],
        &[&[rr(1), end()], &[ch('a'), alt(), end()]],
    );
}

#[test]
fn star_on_a_rule_reference() {
    verify(
        "\n        root  ::= a*\n        a     ::= \"a\"\n    ",
        &[("a", 1), ("root", 0), ("root_2", 2)],
        &[
            &[rr(2), end()],
            &[ch('a'), end()],
            &[rr(1), rr(2), alt(), end()],
        ],
    );
}

#[test]
fn star_on_a_literal() {
    verify(
        "\n        root  ::= \"a\"*\n    ",
        &[("root", 0), ("root_1", 1)],
        &[&[rr(1), end()], &[ch('a'), rr(1), alt(), end()]],
    );
}

#[test]
fn exact_repetition_is_unrolled_with_no_extra_rule() {
    verify(
        "\n        root  ::= \"a\"{2}\n    ",
        &[("root", 0)],
        &[&[ch('a'), ch('a'), end()]],
    );
}

#[test]
fn open_ended_repetition_unrolls_the_minimum_then_recurses() {
    verify(
        "\n        root  ::= \"a\"{2,}\n    ",
        &[("root", 0), ("root_1", 1)],
        &[
            &[ch('a'), ch('a'), rr(1), end()],
            &[ch('a'), rr(1), alt(), end()],
        ],
    );
}

#[test]
fn space_inside_the_repetition_brace_is_allowed() {
    verify(
        "\n        root  ::= \"a\"{ 4}\n    ",
        &[("root", 0)],
        &[&[ch('a'), ch('a'), ch('a'), ch('a'), end()]],
    );
}

#[test]
fn bounded_repetition_chains_one_rule_per_optional_copy() {
    // `{2,4}`: two unrolled copies, then a chain root_2 -> root_1 giving
    // exactly two more optional copies. Note the ids: root_1 is generated
    // first but referenced by root_2, so the chain runs backwards.
    verify(
        "\n        root  ::= \"a\"{2,4}\n    ",
        &[("root", 0), ("root_1", 1), ("root_2", 2)],
        &[
            &[ch('a'), ch('a'), rr(2), end()],
            &[ch('a'), alt(), end()],
            &[ch('a'), rr(1), alt(), end()],
        ],
    );
}

#[test]
fn arithmetic_without_whitespace_rule() {
    verify(
        "\n        root  ::= (expr \"=\" term \"\\n\")+\n        expr  ::= term ([-+*/] term)*\n        term  ::= [0-9]+\n    ",
        &[
            ("expr", 2),
            ("expr_5", 5),
            ("expr_6", 6),
            ("root", 0),
            ("root_1", 1),
            ("root_4", 4),
            ("term", 3),
            ("term_7", 7),
        ],
        &[
            &[rr(1), rr(4), end()],
            &[rr(2), ch('='), rr(3), ch('\n'), end()],
            &[rr(3), rr(6), end()],
            &[ch('0'), crng('9'), rr(7), end()],
            &[rr(1), rr(4), alt(), end()],
            &[ch('-'), calt('+'), calt('*'), calt('/'), rr(3), end()],
            &[rr(5), rr(6), alt(), end()],
            &[ch('0'), crng('9'), rr(7), alt(), end()],
        ],
    );
}

#[test]
fn the_shipped_arithmetic_grammar() {
    // `grammars/arithmetic.gbnf`, upstream's largest parser-test case.
    verify(
        "\n        root  ::= (expr \"=\" ws term \"\\n\")+\n        expr  ::= term ([-+*/] term)*\n        term  ::= ident | num | \"(\" ws expr \")\" ws\n        ident ::= [a-z] [a-z0-9_]* ws\n        num   ::= [0-9]+ ws\n        ws    ::= [ \\t\\n]*\n    ",
        &[
            ("expr", 2),
            ("expr_6", 6),
            ("expr_7", 7),
            ("ident", 8),
            ("ident_10", 10),
            ("num", 9),
            ("num_11", 11),
            ("root", 0),
            ("root_1", 1),
            ("root_5", 5),
            ("term", 4),
            ("ws", 3),
            ("ws_12", 12),
        ],
        &[
            &[rr(1), rr(5), end()],
            &[rr(2), ch('='), rr(3), rr(4), ch('\n'), end()],
            &[rr(4), rr(7), end()],
            &[rr(12), end()],
            &[
                rr(8),
                alt(),
                rr(9),
                alt(),
                ch('('),
                rr(3),
                rr(2),
                ch(')'),
                rr(3),
                end(),
            ],
            &[rr(1), rr(5), alt(), end()],
            &[ch('-'), calt('+'), calt('*'), calt('/'), rr(4), end()],
            &[rr(6), rr(7), alt(), end()],
            &[ch('a'), crng('z'), rr(10), rr(3), end()],
            &[ch('0'), crng('9'), rr(11), rr(3), end()],
            &[
                ch('a'),
                crng('z'),
                calt('0'),
                crng('9'),
                calt('_'),
                rr(10),
                alt(),
                end(),
            ],
            &[ch('0'), crng('9'), rr(11), alt(), end()],
            &[ch(' '), calt('\t'), calt('\n'), rr(12), alt(), end()],
        ],
    );
}

#[test]
fn token_and_inverted_token_elements() {
    // Upstream's last case: `<[1000]>` is "<think>", `<[1001]>` is
    // "</think>". The `<[id]>` form needs no vocabulary.
    verify(
        "\n        root  ::= <[1000]> !<[1001]> <[1001]>\n    ",
        &[("root", 0)],
        &[&[tok(1000), tnot(1001), tok(1001), end()]],
    );
}

// -- cases upstream's parser test does not cover, checked against the
//    behaviour of `llama-grammar.cpp` by reading it --

#[test]
fn dot_compiles_to_char_any() {
    verify("root ::= .", &[("root", 0)], &[&[any(), end()]]);
}

#[test]
fn escapes_decode_to_code_points() {
    verify(
        r#"root ::= "\t\r\n\\\"" [\x41\u00e9\U0001D11E]"#,
        &[("root", 0)],
        &[&[
            ch('\t'),
            ch('\r'),
            ch('\n'),
            ch('\\'),
            ch('"'),
            ch('A'),
            calt('é'),
            calt('𝄞'),
            end(),
        ]],
    );
}

#[test]
fn bracket_escapes_inside_a_class_are_literal_brackets() {
    verify(
        r"root ::= [\[\]]",
        &[("root", 0)],
        &[&[ch('['), calt(']'), end()]],
    );
}

#[test]
fn a_trailing_dash_in_a_class_is_a_literal_dash() {
    // `pos[0] == '-' && pos[1] != ']'` -- the second half is what makes
    // `[a-]` two characters rather than an unterminated range.
    verify(
        "root ::= [a-]",
        &[("root", 0)],
        &[&[ch('a'), calt('-'), end()]],
    );
}

#[test]
fn multibyte_class_bounds_survive() {
    // From `grammars/japanese.gbnf`.
    verify(
        "root ::= [ぁ-ゟ]",
        &[("root", 0)],
        &[&[ch('ぁ'), crng('ゟ'), end()]],
    );
}

#[test]
fn comments_are_skipped_and_end_a_rule_body() {
    verify(
        "# leading comment\nroot ::= \"a\" # trailing comment\n",
        &[("root", 0)],
        &[&[ch('a'), end()]],
    );
}

#[test]
fn an_empty_alternate_is_a_legal_epsilon_branch() {
    // `ws ::= | " "` from `grammars/json.gbnf`: the first alternate is
    // empty, which is how that grammar makes whitespace optional.
    verify(
        "root ::= | \" \"",
        &[("root", 0)],
        &[&[alt(), ch(' '), end()]],
    );
}

#[test]
fn a_referenced_but_undefined_rule_is_named_in_the_error() {
    let e = verify_failure("root ::= missing");
    match e {
        GrammarError::UndefinedRule { name, .. } => assert_eq!(name, "missing"),
        other => panic!("expected UndefinedRule, got {other:?}"),
    }
}

#[test]
fn a_missing_definition_arrow_is_refused() {
    let e = verify_failure("root \"a\"");
    assert!(
        matches!(&e, GrammarError::Syntax { expected, .. } if expected.contains("::=")),
        "{e:?}"
    );
}

#[test]
fn an_unclosed_group_is_refused() {
    let e = verify_failure("root ::= (\"a\"");
    assert!(
        matches!(&e, GrammarError::Syntax { expected, .. } if expected.contains(')')),
        "{e:?}"
    );
}

#[test]
fn an_unclosed_literal_is_refused() {
    let e = verify_failure("root ::= \"a");
    assert!(
        matches!(&e, GrammarError::Syntax { expected, .. } if expected.contains("end of input")),
        "{e:?}"
    );
}

#[test]
fn a_repetition_with_nothing_before_it_is_refused() {
    let e = verify_failure("root ::= *");
    assert!(
        matches!(&e, GrammarError::Syntax { expected, .. } if expected.contains("preceding item")),
        "{e:?}"
    );
}

#[test]
fn an_unknown_escape_is_refused() {
    let e = verify_failure(r#"root ::= "\q""#);
    assert!(
        matches!(&e, GrammarError::Syntax { expected, .. } if expected.contains("escape")),
        "{e:?}"
    );
}

#[test]
fn a_short_hex_escape_is_refused() {
    let e = verify_failure(r#"root ::= "\x4""#);
    assert!(
        matches!(&e, GrammarError::Syntax { expected, .. } if expected.contains("hex")),
        "{e:?}"
    );
}

#[test]
fn a_single_repetition_over_the_threshold_is_refused() {
    let e = verify_failure("root ::= \"a\"{3000}");
    assert!(
        matches!(e, GrammarError::RepetitionTooLarge { .. }),
        "{e:?}"
    );
}

#[test]
fn a_reversed_repetition_range_does_not_hang() {
    // `{4,2}` underflows `max_times - min_times` upstream and loops ~2^64
    // times. We saturate to zero optional copies, so it reads as `{4}`.
    // This is a deliberate divergence, recorded here so it cannot be
    // changed silently.
    verify(
        "root ::= \"a\"{4,2}",
        &[("root", 0)],
        &[&[ch('a'), ch('a'), ch('a'), ch('a'), end()]],
    );
}

#[test]
fn a_named_token_without_a_vocabulary_names_what_is_missing() {
    let e = verify_failure("root ::= <think>");
    match e {
        GrammarError::TokenNeedsVocabulary { token, .. } => assert_eq!(token, "<think>"),
        other => panic!("expected TokenNeedsVocabulary, got {other:?}"),
    }
}

struct FakeVocab;

impl GrammarVocab for FakeVocab {
    fn tokenize_special(&self, text: &str) -> Vec<u32> {
        match text {
            "<think>" => vec![1000],
            "<two>" => vec![1, 2],
            _ => vec![],
        }
    }
}

#[test]
fn a_named_token_resolves_through_a_vocabulary() {
    let g = parse_with_vocab("root ::= <think>", Some(&FakeVocab)).expect("parses");
    assert_eq!(g.rules[0], vec![tok(1000), end()]);
}

#[test]
fn a_named_token_that_is_not_one_token_is_refused() {
    let e = parse_with_vocab("root ::= <two>", Some(&FakeVocab)).unwrap_err();
    match e {
        GrammarError::TokenNotSingle { token, n_tokens } => {
            assert_eq!(token, "<two>");
            assert_eq!(n_tokens, 2);
        }
        other => panic!("expected TokenNotSingle, got {other:?}"),
    }
}

#[test]
fn every_shipped_grammar_parses() {
    // The eight grammars in llama.cpp's `grammars/` directory. Each is
    // checked for the root rule and for a plausible rule count, so that a
    // parser that silently produced an empty table would fail.
    for (name, src, min_rules) in SHIPPED_GRAMMARS {
        let g = parse(src).unwrap_or_else(|e| panic!("{name} failed to parse: {e}"));
        assert!(
            g.symbol_id("root").is_some(),
            "{name} has no root rule: {:?}",
            g.symbol_ids.keys().collect::<Vec<_>>()
        );
        assert!(
            g.rules.len() >= *min_rules,
            "{name} produced {} rules, expected at least {min_rules}",
            g.rules.len()
        );
        for (i, rule) in g.rules.iter().enumerate() {
            assert!(!rule.is_empty(), "{name} rule {i} is empty");
            assert_eq!(
                rule.last().map(|e| e.gtype),
                Some(GreType::End),
                "{name} rule {i} does not end with END"
            );
        }
    }
}
