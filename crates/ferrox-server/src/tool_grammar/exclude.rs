//! "Every string that does not contain any of these literals", as GBNF.
//!
//! The XML-ish wire formats end an argument's value at a closing tag,
//! so the value rule is *everything up to* `</parameter>` (or whatever
//! that family calls it), and a grammar that FORCES a call has to be
//! able to say exactly that. `[^<]*` says something else: it forbids
//! every `<`, and a coding agent's arguments are whole files.
//!
//! This is the complement of the literals' Aho-Corasick automaton, one
//! right-recursive rule per state -- llama.cpp's
//! `gbnf_excluding_grammar` (`common/peg-parser.cpp`, added by
//! ggml-org/llama.cpp#24839). Every state accepts, and the transition
//! that would COMPLETE a literal is the one alternative that is never
//! written, so no literal can ever be matched.
//!
//! # Why a SET rather than one literal
//!
//! Most formats stop reading a value at exactly one string, and for
//! those the automaton degenerates to KMP. Gemma 4 is the one that does
//! not: its string values are wrapped in `<|"|>`, and the reader toggles
//! quoting at every one of those *and* ends the whole call at the first
//! `<tool_call|>` without regard for quoting, so a value that may
//! contain either is a value this server would read back wrong. Two
//! separate KMP exclusions cannot be intersected in GBNF; one automaton
//! over both patterns is the same construction and answers it.
//!
//! Right recursion is deliberate and is not a stack leak: a rule
//! reference in final position is not pushed as a continuation
//! (`machine::advance_stack`, transcribed from
//! `llama_grammar_advance_stack`), so a value of any length costs one
//! stack entry.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use ferrox_models::grammar::json_schema::GrammarBuilder;

use crate::ApiError;

/// Emit the rules for text that cannot contain any of `forbidden`, and
/// return the name of the rule to reference.
///
/// `prefix` must be free in `builder`: these rules reference each other
/// by name, and `GrammarBuilder::add_rule` renames a name already bound
/// to a different body, which would silently point a state at somebody
/// else's rule. Call this BEFORE adding any schema, while nothing but
/// the builtins is bound. The check below is what makes a caller that
/// gets that order wrong a refusal rather than a wrong grammar.
pub(super) fn text_excluding(
    builder: &mut GrammarBuilder,
    prefix: &str,
    forbidden: &[&str],
) -> Result<String, ApiError> {
    if forbidden.is_empty() || forbidden.iter().any(|literal| literal.is_empty()) {
        // Every string contains the empty string, and a set with no
        // literal in it forbids nothing this caller meant to forbid, so
        // the honest language here is the empty one -- which GBNF cannot
        // spell. The empty string is the only under-approximation, and
        // it is safe: it never lets through text the reader would not
        // read back. Every format `wire::shape` gives a value rule to
        // names at least one non-empty literal, so this is a guard on
        // the call rather than a case a request can reach.
        return Ok(builder.add_rule(prefix, r#""""#));
    }

    let automaton = Automaton::over(forbidden);
    let name_of = |state: usize| {
        if state == 0 {
            prefix.to_string()
        } else {
            format!("{prefix}-{state}")
        }
    };

    for state in 0..automaton.nodes.len() {
        if automaton.matched[state] {
            // A state only reached by completing a literal. The
            // complement never enters it, so it gets no rule -- and
            // nothing references one, because every transition into a
            // matched state is the alternative that is never written.
            continue;
        }
        // Chars whose transition leads somewhere other than the start
        // state, grouped by where; plus every char that has any
        // explicit transition at all, so the rest can be swept up by
        // one negated class.
        let mut buckets: BTreeMap<usize, Vec<char>> = BTreeMap::new();
        let mut specific: Vec<char> = Vec::new();
        for &c in &automaton.alphabet {
            let next = automaton.step(state, c);
            if automaton.matched[next] {
                // Completing a literal. Listed as "explicit" so the
                // catch-all cannot match it, and given no alternative
                // of its own: that is the whole exclusion.
                specific.push(c);
            } else if next != 0 {
                buckets.entry(next).or_default().push(c);
                specific.push(c);
            }
        }

        // The empty first alternative: every state of the complement
        // accepts, because a string that has not completed a literal
        // does not contain one.
        let mut alternatives = vec![String::new()];
        for (next, group) in &buckets {
            alternatives.push(format!("{} {}", char_class(group, false), name_of(*next)));
        }
        alternatives.push(format!("{} {}", char_class(&specific, true), name_of(0)));

        let name = name_of(state);
        let got = builder.add_rule(&name, &alternatives.join(" | "));
        if got != name {
            return Err(super::internal(format!(
                "tool-call grammar: the rule {name:?} that holds text excluding {forbidden:?} was \
                 renamed to {got:?}, so its own states would reference the wrong rule"
            )));
        }
    }
    Ok(name_of(0))
}

/// The Aho-Corasick automaton of a set of literals, over the characters
/// those literals are spelled with.
///
/// A state is one node of the trie of the literals: the longest suffix
/// of the input read so far that is a prefix of some literal.
struct Automaton {
    /// The trie's explicit edges, one map per node.
    nodes: Vec<BTreeMap<char, usize>>,
    /// The longest proper suffix of this node's string that is also a
    /// node.
    fail: Vec<usize>,
    /// Whether reaching this node means a literal has been matched --
    /// either because the node IS a literal, or because one ends inside
    /// the string that reaches it.
    matched: Vec<bool>,
    /// Every character any literal is spelled with. A character outside
    /// it can end no literal's prefix, so it always leads back to the
    /// start state and is swept up by the catch-all class.
    alphabet: BTreeSet<char>,
}

impl Automaton {
    fn over(literals: &[&str]) -> Self {
        let mut automaton = Automaton {
            nodes: vec![BTreeMap::new()],
            fail: vec![0],
            matched: vec![false],
            alphabet: BTreeSet::new(),
        };
        for literal in literals {
            let mut node = 0usize;
            for c in literal.chars() {
                automaton.alphabet.insert(c);
                node = match automaton.nodes[node].get(&c) {
                    Some(&next) => next,
                    None => {
                        automaton.nodes.push(BTreeMap::new());
                        automaton.fail.push(0);
                        automaton.matched.push(false);
                        let next = automaton.nodes.len() - 1;
                        automaton.nodes[node].insert(c, next);
                        next
                    }
                };
            }
            automaton.matched[node] = true;
        }

        // Breadth-first, so a node's failure link is computed after the
        // shorter node it points at. `matched` propagates BOTH ways a
        // literal can be present in the string that reaches a node: as
        // a suffix (the failure link) and as a prefix (the parent).
        let mut queue: VecDeque<usize> = automaton.nodes[0].values().copied().collect();
        while let Some(node) = queue.pop_front() {
            automaton.matched[node] =
                automaton.matched[node] || automaton.matched[automaton.fail[node]];
            for (&c, &child) in &automaton.nodes[node].clone() {
                automaton.fail[child] = automaton.step(automaton.fail[node], c);
                automaton.matched[child] = automaton.matched[child] || automaton.matched[node];
                queue.push_back(child);
            }
        }
        automaton
    }

    /// The state reached from `state` on `c`: the longest suffix of the
    /// input read so far that is a prefix of some literal.
    fn step(&self, state: usize, c: char) -> usize {
        let mut state = state;
        loop {
            if let Some(&next) = self.nodes[state].get(&c) {
                return next;
            }
            if state == 0 {
                return 0;
            }
            state = self.fail[state];
        }
    }
}

/// A GBNF character class over `chars`, negated or not.
///
/// `-` is spelled `\x2D` rather than `\-` for the reason
/// `json_schema::primitives::escape_in_range` gives: llama.cpp's table
/// lists `\-` but its GBNF *parser* has no such escape, and this repo's
/// parser transcribes the parser.
fn char_class(chars: &[char], negated: bool) -> String {
    let mut out = String::from("[");
    if negated {
        out.push('^');
    }
    for &c in chars {
        match c {
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '"' => out.push_str("\\\""),
            '-' => out.push_str("\\x2D"),
            ']' => out.push_str("\\]"),
            '[' => out.push_str("\\["),
            '\\' => out.push_str("\\\\"),
            '^' => out.push_str("\\x5E"),
            c => out.push(c),
        }
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_models::grammar::Grammar;

    /// Compile `root ::= <text excluding forbidden> "END"` and report
    /// whether `text` parses -- the shape the value rules use, so the
    /// exclusion is tested where it has to be exact: right before the
    /// literal it excludes.
    fn accepts(forbidden: &str, text: &str) -> bool {
        accepts_any(&[forbidden], text)
    }

    /// The same, for a set of literals: the text must contain none of
    /// them.
    fn accepts_any(forbidden: &[&str], text: &str) -> bool {
        let mut builder = GrammarBuilder::new();
        let body = text_excluding(&mut builder, "not", forbidden).expect("a rule");
        builder.add_rule("root", &format!("{body} \"END\""));
        let grammar = Grammar::from_str_with_root(&builder.finish().expect("grammar"), "root")
            .expect("compiles");
        let mut g = grammar.clone();
        let whole = format!("{text}END");
        if g.accept_token(0, whole.as_bytes()).is_err() {
            return false;
        }
        g.allows_eog()
    }

    /// The headline: a value may hold anything at all except the tag
    /// that ends it.
    #[test]
    fn only_the_forbidden_literal_is_refused() {
        assert!(accepts("</parameter>", "plain text"));
        assert!(accepts("</parameter>", "<html><body>a < b</body></html>"));
        assert!(accepts(
            "</parameter>",
            "</param> </parameters> <parameter>"
        ));
        assert!(accepts("</parameter>", ""));
        assert!(!accepts("</parameter>", "before</parameter>after"));
        assert!(!accepts("</parameter>", "</parameter>"));
    }

    /// The exclusion has to survive a partial match that restarts, which
    /// is the whole reason it is an automaton and not an alternation of
    /// "a prefix then a mismatching character".
    #[test]
    fn a_restarted_partial_match_still_completes_the_literal() {
        // "aab" contains "ab" from index 1. The naive
        // `([^a] | "a" [^b])*` construction accepts it.
        assert!(!accepts("ab", "aab"));
        assert!(accepts("ab", "aa"));
        assert!(!accepts("aa", "baaa"));
        assert!(accepts("aa", "aba"));
        // A literal with a border, where the failure function matters.
        assert!(!accepts("aba", "xxababa"));
        assert!(accepts("aba", "xxabb"));
    }

    /// The multi-byte tags are the ones a byte-wise automaton would get
    /// wrong: DeepSeek spells its tags with U+FF5C.
    #[test]
    fn a_multi_byte_literal_is_excluded_by_codepoint() {
        assert!(accepts("</｜DSML｜parameter>", "値 with a ｜ in it"));
        assert!(accepts("</｜DSML｜parameter>", "</｜DSML｜invoke>"));
        assert!(!accepts("</｜DSML｜parameter>", "x</｜DSML｜parameter>y"));
    }

    /// Two literals at once, which is the case a pair of KMP exclusions
    /// cannot express: gemma 4's string values may contain neither the
    /// quote that ends them nor the marker that ends the whole call.
    #[test]
    fn a_set_of_literals_excludes_every_member() {
        const QUOTE: &str = "<|\"|>";
        const BLOCK_CLOSE: &str = "<tool_call|>";
        let gemma = [QUOTE, BLOCK_CLOSE];

        assert!(accepts_any(&gemma, "plain text"));
        // Long proper prefixes of both, and the other family's markers.
        assert!(accepts_any(&gemma, "<| <|\" <tool_call| </tool_call>"));
        assert!(!accepts_any(&gemma, "before<|\"|>after"));
        assert!(!accepts_any(&gemma, "before<tool_call|>after"));
        // Each member is excluded even when the other is the one that
        // nearly matched first: a KMP automaton for either alone accepts
        // the string that ends in the other.
        assert!(!accepts_any(&gemma, "<tool_call<|\"|>"));
        assert!(!accepts_any(&gemma, "<|\"<tool_call|>"));
    }

    /// A literal that is a suffix of another shares states, so the
    /// automaton's failure links -- not just its trie -- have to carry
    /// `matched` for the shorter one.
    #[test]
    fn a_literal_that_is_a_suffix_of_another_is_still_excluded() {
        assert!(!accepts_any(&["abc", "bc"], "xbcx"));
        assert!(!accepts_any(&["abc", "bc"], "xabcx"));
        assert!(accepts_any(&["abc", "bc"], "xacx"));
        // And a literal that CONTAINS another: the longer one is
        // unreachable, and the shorter one still bites.
        assert!(!accepts_any(&["bc", "abcd"], "xbcx"));
        assert!(accepts_any(&["bc", "abcd"], "xabd"));
    }

    /// A literal reached only by passing THROUGH another one can never
    /// be matched, so its states must not be written down at all.
    ///
    /// This is not about the language -- an unreferenced rule accepts
    /// nothing extra. It is about the grammar staying the size of the
    /// automaton it means: the tail of `"abcde"` past the `"bc"` inside
    /// it is a state nothing can enter, and a value rule is emitted once
    /// per argument of every offered tool.
    #[test]
    fn a_state_behind_a_matched_one_gets_no_rule() {
        let mut builder = GrammarBuilder::new();
        let body = text_excluding(&mut builder, "not", &["bc", "abcde"]).expect("a rule");
        builder.add_rule("root", &body);
        let text = builder.finish().expect("grammar");

        let mut defined: Vec<&str> = Vec::new();
        for line in text.lines() {
            if let Some((name, _)) = line.split_once("::=") {
                let name = name.trim();
                if name.starts_with("not") {
                    defined.push(name);
                }
            }
        }
        assert!(
            defined.len() > 1,
            "the automaton should have states of its own: {text}"
        );
        for name in &defined {
            if *name == "not" {
                continue;
            }
            let referenced = text
                .lines()
                .filter(|line| !line.trim_start().starts_with(&format!("{name} ")))
                .any(|line| {
                    line.split_once("::=")
                        .is_some_and(|(_, body)| body.split_whitespace().any(|word| word == *name))
                });
            assert!(
                referenced,
                "{name} is a state nothing can enter, so it should not have been written: {text}"
            );
        }
    }

    /// Every character that has to be escaped to reach a GBNF class
    /// alive. A literal holding one of these used to be the way this
    /// emitted a grammar that does not parse.
    #[test]
    fn a_literal_of_class_metacharacters_still_compiles() {
        for forbidden in ["]-^", "[\\]", "\"a\"", "\n\t"] {
            assert!(
                accepts(forbidden, "harmless"),
                "{forbidden:?} should compile and accept text without it"
            );
            assert!(
                !accepts(forbidden, forbidden),
                "{forbidden:?} should exclude itself"
            );
        }
    }
}
