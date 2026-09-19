//! `pattern`: llama.cpp's regex-to-GBNF compiler, and the inputs it
//! mishandles that [`super::super::pattern`] refuses instead.
//!
//! The first six expectations are upstream's, byte for byte. The rest
//! are this port's own, and each names the upstream behaviour it
//! replaces.

use super::{check, refuse};
use crate::grammar::json_schema::SchemaError;

#[test]
fn patterns_from_upstream() {
    let space = r##"space ::= | " " | "\n"{1,2} [ \t]{0,20}"##;
    for (pattern, root) in [
        (
            r##"^abc?d*efg+(hij)?kl$"##,
            r##"root ::= "\"" ("ab" "c"? "d"* "ef" "g"+ ("hij")? "kl") "\"""##,
        ),
        (
            r##"^\\[\\]\\{\\}\\(\\)\\|\\+\\*\\?$"##,
            r##"root ::= "\"" ("[]{}()|+*?") "\"""##,
        ),
        (r##"^\"$"##, r##"root ::= "\"" ("\"") "\"""##),
        (
            r##"^A|B|C|D$"##,
            r##"root ::= "\"" ("A" | "B" | "C" | "D") "\"""##,
        ),
        (
            r##"^(?:foo|bar)baz$"##,
            r##"root ::= "\"" (("foo" | "bar") "baz") "\"""##,
        ),
        (
            r##"^(?:(?:ab)+c)?d$"##,
            r##"root ::= "\"" ((("ab")+ "c")? "d") "\"""##,
        ),
    ] {
        check(
            &format!(r##"{{"type": "string", "pattern": "{pattern}"}}"##),
            &format!("{root}\n{space}"),
        );
    }
}

#[test]
fn pattern_with_dot_and_sub_rules() {
    // The `{m,n}` over a non-literal hoists it into `root-1`, numbered
    // from 1, and every later `[0-9]{...}` reuses that same rule.
    check(
        r##"{
            "type": "string",
            "pattern": "^(\\([0-9]{1,3}\\))?[0-9]{3}-[0-9]{4} a{3,5}nd...$"
        }"##,
        r##"
        dot ::= [^\x0A\x0D]
        root ::= "\"" (("(" root-1{1,3} ")")? root-1{3,3} "-" root-1{4,4} " " "a"{3,5} "nd" dot dot dot) "\""
        root-1 ::= [0-9]
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

#[test]
fn shorthand_classes_are_translated_not_passed_through() {
    // llama.cpp copies `\d` into the grammar, where its own GBNF parser
    // rejects it as an unknown escape. ECMA-262 defines these exactly.
    check(
        r##"{"type": "string", "pattern": "^\\d+-\\w{2}$"}"##,
        r##"
        root ::= "\"" ([0-9]+ "-" root-1{2,2}) "\""
        root-1 ::= [0-9A-Za-z_]
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
    // Inside a class the members splice in rather than nesting brackets.
    check(
        r##"{"type": "string", "pattern": "^[\\dA-F]+$"}"##,
        r##"
        root ::= "\"" ([0-9A-F]+) "\""
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
    // A `-` inside a class must not become a range, and GBNF has no `\-`.
    check(
        r##"{"type": "string", "pattern": "^[a\\-z]$"}"##,
        r##"
        root ::= "\"" ([a\x2Dz]) "\""
        space ::= | " " | "\n"{1,2} [ \t]{0,20}
        "##,
    );
}

#[test]
fn patterns_upstream_mishandles_are_refused() {
    for (pattern, needle) in [
        // Unanchored: upstream errors here too.
        (r##"[a-z]+"##, "start with '^'"),
        // Upstream warns, drops the group, and keeps going.
        (r##"^(?=foo)bar$"##, "lookahead"),
        // Upstream emits `"\s"`, which its own GBNF parser rejects.
        (r##"^\\s+$"##, "Unicode whitespace"),
        (r##"^\\bword$"##, "zero-width"),
        (r##"^(a)\\1$"##, "backreferences"),
        // Upstream loops forever on a stray closer.
        (r##"^a]b$"##, "no opening '['"),
        (r##"^a}b$"##, "no opening '{'"),
        // Upstream reads `seq.back()` on an empty vector.
        (r##"^*a$"##, "nothing before it to repeat"),
        // Upstream returns early and silently discards the `b`.
        (r##"^a)b$"##, "unbalanced parentheses"),
        (r##"^(ab$"##, "unbalanced parentheses"),
        (r##"^[ab$"##, "unbalanced square brackets"),
        (r##"^a{2$"##, "unbalanced curly brackets"),
        (r##"^a\\$"##, "dangling"),
    ] {
        let schema = format!(r##"{{"type": "string", "pattern": "{pattern}"}}"##);
        match refuse(&schema) {
            SchemaError::UnsupportedPattern { why, .. } => assert!(
                why.contains(needle),
                "pattern {pattern:?}: expected {needle:?} in {why:?}"
            ),
            other => panic!("pattern {pattern:?}: expected a pattern refusal, got {other}"),
        }
    }
}
