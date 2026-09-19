//! llama.cpp's `PRIMITIVE_RULES` and `STRING_FORMAT_RULES` tables, the
//! literal escaping around them, and `build_repetition`.
//!
//! The GBNF text is transcribed byte for byte from
//! `common/json-schema-to-grammar.cpp`; the C++ source hides it behind
//! two layers of escaping, so each constant below is a Rust *raw* string
//! holding what the C++ literal decodes to.

/// One entry of a builtin table: its GBNF body, and the other builtins its
/// body refers to by name.
pub(super) struct BuiltinRule {
    pub content: &'static str,
    pub deps: &'static [&'static str],
}

/// `SPACE_RULE`. The leading `|` is an empty first alternative: whitespace
/// is optional everywhere it appears.
pub(super) const SPACE_RULE: &str = r##"| " " | "\n"{1,2} [ \t]{0,20}"##;

/// `PRIMITIVE_RULES`.
pub(super) const PRIMITIVE_RULES: &[(&str, BuiltinRule)] = &[
    (
        "boolean",
        BuiltinRule {
            content: r##"("true" | "false")"##,
            deps: &[],
        },
    ),
    (
        "decimal-part",
        BuiltinRule {
            content: r##"[0-9]{1,16}"##,
            deps: &[],
        },
    ),
    (
        "integral-part",
        BuiltinRule {
            content: r##"[0] | [1-9] [0-9]{0,15}"##,
            deps: &[],
        },
    ),
    (
        "number",
        BuiltinRule {
            content: r##"("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?"##,
            deps: &["integral-part", "decimal-part"],
        },
    ),
    (
        "integer",
        BuiltinRule {
            content: r##"("-"? integral-part)"##,
            deps: &["integral-part"],
        },
    ),
    (
        "value",
        BuiltinRule {
            content: r##"object | array | string | number | boolean | null"##,
            deps: &["object", "array", "string", "number", "boolean", "null"],
        },
    ),
    (
        "object",
        BuiltinRule {
            content: r##""{" space ( string ":" space value ("," space string ":" space value)* )? space "}""##,
            deps: &["string", "value"],
        },
    ),
    (
        "array",
        BuiltinRule {
            content: r##""[" space ( value ("," space value)* )? space "]""##,
            deps: &["value"],
        },
    ),
    (
        "uuid",
        BuiltinRule {
            content: r##""\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"""##,
            deps: &[],
        },
    ),
    (
        "char",
        BuiltinRule {
            content: r##"[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})"##,
            deps: &[],
        },
    ),
    (
        "string",
        BuiltinRule {
            content: r##""\"" char* "\"""##,
            deps: &["char"],
        },
    ),
    (
        "null",
        BuiltinRule {
            content: r##""null""##,
            deps: &[],
        },
    ),
];

/// `STRING_FORMAT_RULES`.
pub(super) const STRING_FORMAT_RULES: &[(&str, BuiltinRule)] = &[
    (
        "date",
        BuiltinRule {
            content: r##"[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )"##,
            deps: &[],
        },
    ),
    (
        "time",
        BuiltinRule {
            content: r##"([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )"##,
            deps: &[],
        },
    ),
    (
        "date-time",
        BuiltinRule {
            content: r##"date "T" time"##,
            deps: &["date", "time"],
        },
    ),
    (
        "date-string",
        BuiltinRule {
            content: r##""\"" date "\"""##,
            deps: &["date"],
        },
    ),
    (
        "time-string",
        BuiltinRule {
            content: r##""\"" time "\"""##,
            deps: &["time"],
        },
    ),
    (
        "date-time-string",
        BuiltinRule {
            content: r##""\"" date-time "\"""##,
            deps: &["date-time"],
        },
    ),
];

/// A builtin by name, primitives first then string formats -- the lookup
/// order `_add_primitive` uses for a dependency.
pub(super) fn builtin(name: &str) -> Option<&'static BuiltinRule> {
    PRIMITIVE_RULES
        .iter()
        .chain(STRING_FORMAT_RULES.iter())
        .find(|(n, _)| *n == name)
        .map(|(_, r)| r)
}

/// A primitive by name. `visit`'s final fallback accepts only these as a
/// `"type"`, which is what makes `{"type": "kaboom"}` an error.
pub(super) fn primitive(name: &str) -> Option<&'static BuiltinRule> {
    PRIMITIVE_RULES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, r)| r)
}

/// `is_reserved_name`: `root` plus every builtin name. A schema-derived
/// rule that would collide with one gets a `-` suffix instead.
pub(super) fn is_reserved_name(name: &str) -> bool {
    name == "root" || builtin(name).is_some()
}

/// `GRAMMAR_LITERAL_ESCAPES` as applied by `format_literal`: the four
/// characters `GRAMMAR_LITERAL_ESCAPE_RE` matches inside a `"…"` literal.
pub(super) fn format_literal(literal: &str) -> String {
    let mut out = String::with_capacity(literal.len() + 2);
    out.push('"');
    for c in literal.chars() {
        match c {
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `GRAMMAR_RANGE_LITERAL_ESCAPES`: escaping for a character that appears
/// inside a `[…]` class.
///
/// Upstream's `_not_strings` writes property-name bytes into a class with
/// no escaping at all, so a property named `a-b` or `a]b` yields a grammar
/// that does not parse. This module escapes them, which is the one place
/// its output can differ from llama.cpp's for a schema llama.cpp accepts.
///
/// `-` is spelled `\x2D` rather than `\-`: llama.cpp's *table* lists `\-`,
/// but its GBNF *parser* -- and so this repo's [`crate::grammar::parser`],
/// which transcribes it -- has no `\-` escape and rejects it as an unknown
/// one. A bare `-` would start a range. The codepoint form is the only
/// spelling both halves agree on.
pub(super) fn escape_in_range(c: char, out: &mut String) {
    match c {
        '\r' => out.push_str("\\r"),
        '\n' => out.push_str("\\n"),
        '"' => out.push_str("\\\""),
        '-' => out.push_str("\\x2D"),
        ']' => out.push_str("\\]"),
        '[' => out.push_str("\\["),
        '\\' => out.push_str("\\\\"),
        c => out.push(c),
    }
}

/// `build_repetition`. `max_items` of `None` is upstream's `INT_MAX`.
pub(super) fn build_repetition(
    item_rule: &str,
    min_items: u64,
    max_items: Option<u64>,
    separator_rule: &str,
) -> String {
    if max_items == Some(0) {
        return String::new();
    }
    if min_items == 0 && max_items == Some(1) {
        return format!("{item_rule}?");
    }

    if separator_rule.is_empty() {
        if min_items == 1 && max_items.is_none() {
            return format!("{item_rule}+");
        }
        if min_items == 0 && max_items.is_none() {
            return format!("{item_rule}*");
        }
        let max = max_items.map(|m| m.to_string()).unwrap_or_default();
        return format!("{item_rule}{{{min_items},{max}}}");
    }

    let inner = build_repetition(
        &format!("({separator_rule} {item_rule})"),
        min_items.saturating_sub(1),
        // `has_max ? max_items - 1 : max_items` upstream. `max_items` is
        // at least 1 here: 0 returned above.
        max_items.map(|m| m - 1),
        "",
    );
    let result = format!("{item_rule} {inner}");
    if min_items == 0 {
        format!("({result})?")
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_matches_upstream_shapes() {
        assert_eq!(build_repetition("a", 0, Some(0), ""), "");
        assert_eq!(build_repetition("a", 0, Some(1), ""), "a?");
        assert_eq!(build_repetition("a", 1, None, ""), "a+");
        assert_eq!(build_repetition("a", 0, None, ""), "a*");
        assert_eq!(build_repetition("a", 3, None, ""), "a{3,}");
        assert_eq!(build_repetition("a", 1, Some(4), ""), "a{1,4}");
        // The separator forms, from the `minItems` / `maxItems` goldens.
        assert_eq!(
            build_repetition("boolean", 0, None, "\",\" space"),
            "(boolean (\",\" space boolean)*)?"
        );
        assert_eq!(
            build_repetition("boolean", 2, None, "\",\" space"),
            "boolean (\",\" space boolean)+"
        );
        assert_eq!(
            build_repetition("boolean", 0, Some(2), "\",\" space"),
            "(boolean (\",\" space boolean)?)?"
        );
        assert_eq!(
            build_repetition("item", 3, Some(5), "\",\" space"),
            "item (\",\" space item){2,4}"
        );
    }

    #[test]
    fn builtin_lookup_finds_both_tables() {
        assert!(primitive("string").is_some());
        assert!(primitive("date-string").is_none());
        assert!(builtin("date-string").is_some());
        assert!(is_reserved_name("root"));
        assert!(is_reserved_name("number"));
        assert!(!is_reserved_name("space"));
    }

    #[test]
    fn literal_escaping() {
        assert_eq!(format_literal(" \r \n \" \\ "), r##"" \r \n \" \\ ""##);
    }
}
