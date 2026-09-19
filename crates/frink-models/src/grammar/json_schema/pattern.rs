//! `_visit_pattern`: an ECMA-262 regular expression compiled to GBNF.
//!
//! This is llama.cpp's regex-to-grammar compiler, transcribed. It is not a
//! general regex engine: it walks the pattern once, emitting GBNF as it
//! goes, and it handles exactly what upstream handles -- literals, `.`,
//! character classes, alternation, groups (capturing and `(?:`), the three
//! quantifiers and `{m,n}`. Anchors are required (`^…$`) and stripped.
//!
//! Two upstream behaviours cannot be carried over, because llama.cpp's
//! regex compiler and llama.cpp's *own* GBNF parser disagree about what
//! text is valid:
//!
//! - An escape the GBNF parser does not know (`\d`, `\s`, `\/`, a dangling
//!   `\`) is copied straight through by `_visit_pattern`, producing a
//!   grammar that fails to parse. `\d` / `\D` / `\w` / `\W` are translated
//!   here to the classes ECMA-262 defines them as; everything else is a
//!   typed refusal ([`SchemaError::UnsupportedPattern`]).
//! - Lookahead and lookbehind are a `_warnings.push_back` upstream, after
//!   which the whole group is *skipped* -- the constraint silently
//!   disappears from the grammar. That is the widening this module exists
//!   to prevent, so it is refused.
//!
//! Three upstream hangs and one silent truncation are refusals here too: a
//! stray `]` or `}` spins `_visit_pattern`'s loop forever, `*` with nothing
//! before it reads `seq.back()` on an empty vector, and a `)` at the top
//! level returns early and discards the rest of the pattern.

use super::converter::Converter;
use super::error::SchemaError;
use super::primitives::build_repetition;
use std::collections::BTreeMap;

/// `get_dot` with `dotall` off, which is the only setting this port has.
const DOT_RULE: &str = r##"[^\x0A\x0D]"##;

/// `NON_LITERAL_SET`.
fn is_non_literal(c: char) -> bool {
    matches!(
        c,
        '|' | '.' | '(' | ')' | '[' | ']' | '{' | '}' | '*' | '+' | '?'
    )
}

/// `ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS`: a regex escape whose meaning
/// is just the character, which a GBNF literal spells bare.
fn is_bare_in_gbnf(c: char) -> bool {
    matches!(
        c,
        '^' | '$' | '.' | '[' | ']' | '(' | ')' | '|' | '{' | '}' | '*' | '+' | '?'
    )
}

/// An escape this repo's GBNF parser understands, spelled the same way.
/// Mirrors the arms of [`crate::grammar::parser`]'s `parse_char`.
fn is_gbnf_escape(c: char) -> bool {
    matches!(c, '\\' | '"' | 'n' | 'r' | 't' | 'x' | 'u' | 'U')
}

/// The ECMA-262 definitions of the two shorthand classes that have an
/// exact, finite spelling. `\s` is deliberately absent: ECMA-262 defines it
/// over Unicode whitespace including U+1680 and U+2000..U+200A, and the
/// ASCII subset a grammar could write would *reject* documents the pattern
/// accepts.
fn shorthand_class(c: char) -> Option<&'static str> {
    match c {
        'd' => Some("[0-9]"),
        'D' => Some("[^0-9]"),
        'w' => Some("[0-9A-Za-z_]"),
        'W' => Some("[^0-9A-Za-z_]"),
        _ => None,
    }
}

/// The members of a shorthand class, for splicing into a larger `[…]`.
fn shorthand_members(c: char) -> Option<&'static str> {
    match c {
        'd' => Some("0-9"),
        'w' => Some("0-9A-Za-z_"),
        _ => None,
    }
}

/// `literal_or_rule`: a fragment that is either literal text (to be
/// wrapped in GBNF quotes) or GBNF source already.
#[derive(Clone)]
struct Piece {
    text: String,
    literal: bool,
}

impl Piece {
    fn rule(text: impl Into<String>) -> Self {
        Piece {
            text: text.into(),
            literal: false,
        }
    }

    fn literal(text: impl Into<String>) -> Self {
        Piece {
            text: text.into(),
            literal: true,
        }
    }

    /// `to_rule`.
    fn to_rule(&self) -> String {
        if self.literal {
            format!("\"{}\"", self.text)
        } else {
            self.text.clone()
        }
    }
}

/// `join_seq`: concatenate, merging runs of adjacent literals into one
/// GBNF string so `a` `b` `c` becomes `"abc"` rather than `"a" "b" "c"`.
fn join_seq(seq: &[Piece]) -> Piece {
    let mut merged: Vec<Piece> = Vec::with_capacity(seq.len());
    let mut literal = String::new();
    for item in seq {
        if item.literal {
            literal.push_str(&item.text);
        } else {
            if !literal.is_empty() {
                merged.push(Piece::literal(std::mem::take(&mut literal)));
            }
            merged.push(item.clone());
        }
    }
    if !literal.is_empty() {
        merged.push(Piece::literal(literal));
    }
    Piece::rule(
        merged
            .iter()
            .map(Piece::to_rule)
            .collect::<Vec<_>>()
            .join(" "),
    )
}

/// The single-pass walk over a pattern.
pub(super) struct PatternCompiler<'a> {
    conv: &'a mut Converter,
    /// `char`s, not bytes as upstream indexes: a multibyte literal cannot
    /// then be split across two `seq` entries.
    chars: Vec<char>,
    pos: usize,
    /// The rule name, which sub-rules are numbered from.
    name: String,
    /// The pattern as written, for error messages.
    source: String,
    /// `sub_rule_ids`: a `{m,n}` over a non-literal hoists it into its own
    /// rule, and the same fragment reuses the same rule.
    sub_rule_ids: BTreeMap<String, String>,
}

impl<'a> PatternCompiler<'a> {
    /// `_visit_pattern`. Returns the name of the rule it defined.
    pub(super) fn compile(
        conv: &'a mut Converter,
        pattern: &str,
        name: &str,
    ) -> Result<String, SchemaError> {
        let chars: Vec<char> = pattern.chars().collect();
        if chars.first() != Some(&'^') || chars.last() != Some(&'$') || chars.len() < 2 {
            return Err(SchemaError::UnsupportedPattern {
                pattern: pattern.to_string(),
                why: "a pattern must start with '^' and end with '$'; llama.cpp anchors every \
                      pattern it compiles, and an unanchored one would match a substring"
                    .to_string(),
            });
        }
        let mut compiler = PatternCompiler {
            conv,
            chars: chars[1..chars.len() - 1].to_vec(),
            pos: 0,
            name: name.to_string(),
            source: pattern.to_string(),
            sub_rule_ids: BTreeMap::new(),
        };
        let body = compiler.transform(true)?.to_rule();
        let rule = format!("\"\\\"\" ({body}) \"\\\"\"");
        Ok(compiler.conv.add_rule(name, &rule))
    }

    fn refuse(&self, why: impl Into<String>) -> SchemaError {
        SchemaError::UnsupportedPattern {
            pattern: self.source.clone(),
            why: why.into(),
        }
    }

    fn at(&self, i: usize) -> Option<char> {
        self.chars.get(i).copied()
    }

    /// `transform`. `top_level` marks the outermost call, where a `)` has
    /// no group to close.
    fn transform(&mut self, top_level: bool) -> Result<Piece, SchemaError> {
        let start = self.pos;
        let mut seq: Vec<Piece> = Vec::new();

        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            match c {
                '.' => {
                    let dot = self.conv.add_rule("dot", DOT_RULE);
                    seq.push(Piece::rule(dot));
                    self.pos += 1;
                }
                '(' => {
                    self.pos += 1;
                    if self.at(self.pos) == Some('?') {
                        if self.at(self.pos + 1) == Some(':') {
                            self.pos += 2;
                        } else {
                            return Err(self.refuse(
                                "lookahead and lookbehind groups ((?=, (?!, (?<=, (?<!) have no \
                                 GBNF form; llama.cpp warns and then drops the group entirely, \
                                 which would let the grammar accept what the pattern rejects",
                            ));
                        }
                    }
                    let inner = self.transform(false)?;
                    seq.push(Piece::rule(format!("({})", inner.to_rule())));
                }
                ')' => {
                    self.pos += 1;
                    // Upstream's check that the `(` this closes was really
                    // opened by the caller. At the top level there is no
                    // such `(`, and upstream returns anyway, discarding
                    // the rest of the pattern.
                    let opened_group = !top_level
                        && start > 0
                        && (self.chars[start - 1] == '('
                            || (start >= 2
                                && self.chars[start - 2] == '?'
                                && self.chars[start - 1] == ':'));
                    if !opened_group {
                        return Err(self.refuse("unbalanced parentheses"));
                    }
                    return Ok(join_seq(&seq));
                }
                '[' => {
                    let class = self.char_class()?;
                    seq.push(Piece::rule(class));
                }
                '|' => {
                    seq.push(Piece::rule("|"));
                    self.pos += 1;
                }
                '*' | '+' | '?' => {
                    let last = seq
                        .last_mut()
                        .ok_or_else(|| SchemaError::UnsupportedPattern {
                            pattern: self.source.clone(),
                            why: format!("'{c}' has nothing before it to repeat"),
                        })?;
                    *last = Piece::rule(format!("{}{c}", last.to_rule()));
                    self.pos += 1;
                }
                '{' => self.repetition(&mut seq)?,
                // A shorthand class ends any literal run, so it is matched
                // before the literal branch rather than inside it.
                '\\' if self.at(self.pos + 1).and_then(shorthand_class).is_some() => {
                    let next = self.chars[self.pos + 1];
                    let class = shorthand_class(next).unwrap_or(DOT_RULE);
                    seq.push(Piece::rule(class));
                    self.pos += 2;
                }
                _ => {
                    let before = self.pos;
                    if let Some(literal) = self.literal_run()? {
                        seq.push(literal);
                    }
                    if self.pos == before {
                        // Upstream spins here forever: `]` and `}` are in
                        // NON_LITERAL_SET but no branch above consumes one.
                        return Err(self.refuse(format!(
                            "'{c}' has no opening '{}'",
                            match c {
                                ']' => '[',
                                _ => '{',
                            }
                        )));
                    }
                }
            }
        }
        if !top_level {
            return Err(self.refuse("unbalanced parentheses"));
        }
        Ok(join_seq(&seq))
    }

    /// The default branch: a run of characters that stand for themselves.
    /// The run stops one character early when the next character carries a
    /// quantifier, so the quantifier binds to a single character rather
    /// than to the whole run.
    fn literal_run(&mut self) -> Result<Option<Piece>, SchemaError> {
        let mut literal = String::new();
        let len = self.chars.len();
        while self.pos < len {
            let c = self.chars[self.pos];
            if c == '\\' && self.pos + 1 < len {
                let next = self.chars[self.pos + 1];
                if is_bare_in_gbnf(next) {
                    literal.push(next);
                    self.pos += 2;
                } else if shorthand_class(next).is_some() {
                    break;
                } else if is_gbnf_escape(next) {
                    literal.push('\\');
                    literal.push(next);
                    self.pos += 2;
                } else {
                    return Err(self.refuse(format!(
                        "the escape \"\\{next}\" has no GBNF spelling{}",
                        match next {
                            's' | 'S' =>
                                "; ECMA-262 defines it over Unicode whitespace, and the ASCII \
                                 subset a grammar could write would reject strings the pattern \
                                 accepts",
                            'b' | 'B' =>
                                "; word boundaries are zero-width and a grammar has no \
                                          way to express one",
                            '0'..='9' => "; backreferences need memory a grammar does not have",
                            _ => "",
                        }
                    )));
                }
            } else if c == '"' {
                literal.push_str("\\\"");
                self.pos += 1;
            } else if c == '\\' {
                // A trailing lone backslash. Upstream copies it into a GBNF
                // literal, where it opens an escape that never closes.
                return Err(self.refuse("the pattern ends in a dangling '\\'"));
            } else if !is_non_literal(c)
                && (self.pos == len - 1
                    || literal.is_empty()
                    || self.chars[self.pos + 1] == '.'
                    || !is_non_literal(self.chars[self.pos + 1]))
            {
                literal.push(c);
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(if literal.is_empty() {
            None
        } else {
            Some(Piece::literal(literal))
        })
    }

    /// A `[…]` class. Upstream copies the bracket text through verbatim;
    /// this re-spells each escape in the form this repo's GBNF parser
    /// accepts, and refuses the ones with no such form.
    fn char_class(&mut self) -> Result<String, SchemaError> {
        let len = self.chars.len();
        let mut out = String::from("[");
        self.pos += 1;
        while self.pos < len && self.chars[self.pos] != ']' {
            let c = self.chars[self.pos];
            if c != '\\' {
                out.push(c);
                self.pos += 1;
                continue;
            }
            let next = self
                .at(self.pos + 1)
                .ok_or_else(|| self.refuse("unbalanced square brackets"))?;
            match next {
                // A shorthand splices its members in: `[\dA-F]` is `[0-9A-F]`.
                _ if shorthand_members(next).is_some() => {
                    out.push_str(shorthand_members(next).unwrap_or(""));
                }
                // GBNF spells these the same way a regex does.
                '\\' | '"' | '[' | ']' | 'n' | 'r' | 't' | 'x' | 'u' | 'U' => {
                    out.push('\\');
                    out.push(next);
                }
                // `-` would open a range if written bare, and GBNF has no
                // `\-`; the codepoint form is unambiguous.
                '-' => out.push_str("\\x2D"),
                // Regex escapes for characters a GBNF class takes literally.
                '^' | '$' | '.' | '(' | ')' | '|' | '{' | '}' | '*' | '+' | '?' | '/' => {
                    out.push(next)
                }
                _ => {
                    return Err(self.refuse(format!(
                        "the escape \"\\{next}\" has no GBNF spelling inside a [] class"
                    )))
                }
            }
            self.pos += 2;
        }
        if self.pos >= len {
            return Err(self.refuse("unbalanced square brackets"));
        }
        out.push(']');
        self.pos += 1;
        Ok(out)
    }

    /// A `{m,n}` quantifier applied to whatever precedes it.
    fn repetition(&mut self, seq: &mut [Piece]) -> Result<(), SchemaError> {
        let len = self.chars.len();
        self.pos += 1;
        let mut text = String::new();
        while self.pos < len && self.chars[self.pos] != '}' {
            text.push(self.chars[self.pos]);
            self.pos += 1;
        }
        if self.pos >= len {
            return Err(self.refuse("unbalanced curly brackets"));
        }
        self.pos += 1;

        let parse = |part: &str| -> Result<u64, SchemaError> {
            part.trim()
                .parse::<u64>()
                .map_err(|_| self.refuse(format!("{part:?} in {{}} is not a non-negative integer")))
        };
        let parts: Vec<&str> = text.split(',').collect();
        let (min, max) = match parts.as_slice() {
            [only] => {
                let n = parse(only)?;
                (n, Some(n))
            }
            [lo, hi] => (
                if lo.is_empty() { 0 } else { parse(lo)? },
                if hi.is_empty() {
                    None
                } else {
                    Some(parse(hi)?)
                },
            ),
            _ => return Err(self.refuse("wrong number of values in curly brackets")),
        };

        let last = seq
            .last()
            .ok_or_else(|| self.refuse("'{' has nothing before it to repeat"))?
            .clone();
        let sub = if last.literal {
            format!("\"{}\"", last.text)
        } else if let Some(id) = self.sub_rule_ids.get(&last.text) {
            id.clone()
        } else {
            // Upstream numbers from 1, because the map entry it is about
            // to fill is default-constructed before `size()` is read.
            let index = self.sub_rule_ids.len() + 1;
            let name = format!("{}-{index}", self.name);
            let id = self.conv.add_rule(&name, &last.text);
            self.sub_rule_ids.insert(last.text.clone(), id.clone());
            id
        };

        if let Some(slot) = seq.last_mut() {
            *slot = Piece::rule(build_repetition(&sub, min, max, ""));
        }
        Ok(())
    }
}
