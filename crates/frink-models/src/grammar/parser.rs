//! The GBNF parser, transcribed from `llama_grammar_parser` in llama.cpp's
//! `src/llama-grammar.cpp`.
//!
//! It compiles grammar text into a flat rule table. Every construct that
//! is not a literal character, a character class, a rule reference or a
//! token is rewritten into extra synthesized rules at parse time, so the
//! stack machine in [`super::machine`] only ever sees five element kinds.
//!
//! The rewrites are upstream's, quoted from the comment in
//! `parse_sequence`:
//!
//! ```text
//! S{m,n} --> S S S (m times) S'(n-m)
//!            S'(x)   ::= S S'(x-1) |
//!            S'(1)   ::= S |
//! S{m,}  --> S S S (m times) S'
//!            S'      ::= S S' |
//! S*     --> S{0,}   -->  S'  ::= S S' |
//! S+     --> S{1,}   -->  S S'    with S' ::= S S' |
//! S?     --> S{0,1}  -->  S'  ::= S |
//! ```
//!
//! Getting these exactly right matters more than it looks: the synthesized
//! rule *ids* are observable, because `S'` is named `<rule>_<id>` and the
//! id is the symbol count at the time it is generated. Two parsers that
//! accept the same language can still build different rule tables, and
//! llama.cpp's own `tests/test-grammar-parser.cpp` pins the tables, not
//! the language. Those pinned tables are transcribed in
//! [`super::parser_tests`].

use std::collections::BTreeMap;

use super::element::{GrammarElement, GrammarRule, GreType};
use super::error::GrammarError;
use super::utf8::{byte_at, decode_char};

/// `MAX_REPETITION_THRESHOLD`: the ceiling on both a single repetition
/// count and on the running product of nested repetitions.
pub const MAX_REPETITION_THRESHOLD: u64 = 2000;

/// Resolves the `<name>` form of a grammar token element to a token id.
///
/// The `<[42]>` form needs no vocabulary. `<name>` does, and llama.cpp
/// requires that the text (angle brackets included) tokenizes to exactly
/// one token, with special tokens enabled.
pub trait GrammarVocab {
    /// Tokenize `text`, which includes its surrounding `<` and `>`, with
    /// special-token parsing on and no BOS.
    fn tokenize_special(&self, text: &str) -> Vec<u32>;
}

/// A parsed grammar: the rule table plus the symbol names that produced it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedGrammar {
    /// `rules[id]` is the definition of symbol `id`, terminated by
    /// [`GreType::End`].
    pub rules: Vec<GrammarRule>,
    /// Symbol name to rule id. Ordered, mirroring upstream's `std::map`,
    /// so error messages and diagnostics are reproducible.
    pub symbol_ids: BTreeMap<String, u32>,
}

impl ParsedGrammar {
    /// The rule id for a symbol name, if the grammar defines or references
    /// it.
    pub fn symbol_id(&self, name: &str) -> Option<u32> {
        self.symbol_ids.get(name).copied()
    }

    /// The name of a rule id, for diagnostics.
    pub fn symbol_name(&self, rule_id: u32) -> Option<&str> {
        self.symbol_ids
            .iter()
            .find(|(_, &v)| v == rule_id)
            .map(|(k, _)| k.as_str())
    }
}

/// Parse GBNF text with no vocabulary. The `<name>` token form is refused
/// by name; `<[id]>` works.
pub fn parse(src: &str) -> Result<ParsedGrammar, GrammarError> {
    parse_with_vocab(src, None)
}

/// Parse GBNF text, resolving `<name>` token elements through `vocab`.
pub fn parse_with_vocab(
    src: &str,
    vocab: Option<&dyn GrammarVocab>,
) -> Result<ParsedGrammar, GrammarError> {
    let mut p = Parser {
        src: src.as_bytes(),
        pos: 0,
        vocab,
        out: ParsedGrammar::default(),
    };
    p.parse_all()?;
    Ok(p.out)
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    vocab: Option<&'a dyn GrammarVocab>,
    out: ParsedGrammar,
}

impl<'a> Parser<'a> {
    #[inline]
    fn at(&self, i: usize) -> u8 {
        byte_at(self.src, i)
    }

    #[inline]
    fn cur(&self) -> u8 {
        self.at(self.pos)
    }

    fn err(&self, expected: impl Into<String>) -> GrammarError {
        GrammarError::syntax(expected, self.src, self.pos)
    }

    fn err_at(&self, expected: impl Into<String>, offset: usize) -> GrammarError {
        GrammarError::syntax(expected, self.src, offset)
    }

    // -- character classification, `is_digit_char` / `is_word_char` --

    fn is_digit_char(c: u8) -> bool {
        c.is_ascii_digit()
    }

    fn is_word_char(c: u8) -> bool {
        c.is_ascii_alphabetic() || c == b'-' || Self::is_digit_char(c)
    }

    // -- lexical helpers --

    /// `parse_space`. Skips spaces, tabs and `#` comments; newlines too
    /// when `newline_ok`. A comment runs to the end of the line but does
    /// not eat the newline, so a comment inside a rule body terminates the
    /// rule exactly as a bare newline would.
    fn parse_space(&mut self, newline_ok: bool) {
        loop {
            let c = self.cur();
            if c == b' ' || c == b'\t' {
                self.pos += 1;
            } else if c == b'#' {
                while self.cur() != 0 && self.cur() != b'\r' && self.cur() != b'\n' {
                    self.pos += 1;
                }
            } else if newline_ok && (c == b'\r' || c == b'\n') {
                self.pos += 1;
            } else {
                return;
            }
        }
    }

    /// `parse_name`. Returns the end offset of the name starting at
    /// `self.pos`; does not advance.
    fn parse_name(&self) -> Result<usize, GrammarError> {
        let mut end = self.pos;
        while Self::is_word_char(self.at(end)) {
            end += 1;
        }
        if end == self.pos {
            return Err(self.err("expecting name"));
        }
        Ok(end)
    }

    /// `parse_int`. Returns the end offset; does not advance.
    fn parse_int_end(&self) -> Result<usize, GrammarError> {
        let mut end = self.pos;
        while Self::is_digit_char(self.at(end)) {
            end += 1;
        }
        if end == self.pos {
            return Err(self.err("expecting integer"));
        }
        Ok(end)
    }

    /// `parse_int` plus the `std::stoull` upstream applies to the result.
    fn parse_u64(&mut self) -> Result<u64, GrammarError> {
        let end = self.parse_int_end()?;
        let text = std::str::from_utf8(&self.src[self.pos..end])
            .map_err(|_| self.err("expecting integer"))?;
        let value: u64 = text
            .parse()
            .map_err(|_| self.err_at("integer is too large", self.pos))?;
        self.pos = end;
        Ok(value)
    }

    /// `parse_hex`. Consumes exactly `size` hex digits.
    fn parse_hex(&mut self, size: usize) -> Result<u32, GrammarError> {
        let start = self.pos;
        let end = start + size;
        let mut value: u32 = 0;
        let mut p = start;
        while p < end && self.at(p) != 0 {
            let c = self.at(p);
            let digit = match c {
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                b'0'..=b'9' => c - b'0',
                _ => break,
            };
            value = (value << 4) + digit as u32;
            p += 1;
        }
        if p != end {
            self.pos = p;
            return Err(self.err_at(format!("expecting {size} hex chars"), start));
        }
        self.pos = p;
        Ok(value)
    }

    /// `parse_char`. One literal character, escape sequences included.
    fn parse_char(&mut self) -> Result<u32, GrammarError> {
        if self.cur() == b'\\' {
            let start = self.pos;
            let next = self.at(self.pos + 1);
            return match next {
                b'x' => {
                    self.pos += 2;
                    self.parse_hex(2)
                }
                b'u' => {
                    self.pos += 2;
                    self.parse_hex(4)
                }
                b'U' => {
                    self.pos += 2;
                    self.parse_hex(8)
                }
                b't' => {
                    self.pos += 2;
                    Ok(u32::from(b'\t'))
                }
                b'r' => {
                    self.pos += 2;
                    Ok(u32::from(b'\r'))
                }
                b'n' => {
                    self.pos += 2;
                    Ok(u32::from(b'\n'))
                }
                b'\\' | b'"' | b'[' | b']' => {
                    self.pos += 2;
                    Ok(u32::from(next))
                }
                _ => Err(self.err_at("unknown escape", start)),
            };
        }
        if self.cur() != 0 {
            let (value, next) = decode_char(self.src, self.pos);
            self.pos = next;
            return Ok(value);
        }
        Err(self.err("unexpected end of input"))
    }

    /// `parse_token`. Either `<[id]>` or `<name>`.
    fn parse_token(&mut self) -> Result<u32, GrammarError> {
        let start = self.pos;
        if self.cur() != b'<' {
            return Err(self.err("expecting '<'"));
        }
        self.pos += 1;

        if self.cur() == b'[' {
            self.pos += 1;
            let id = self.parse_u64()?;
            let id = u32::try_from(id).map_err(|_| self.err_at("token id is too large", start))?;
            if self.cur() != b']' {
                return Err(self.err("expecting ']'"));
            }
            self.pos += 1;
            if self.cur() != b'>' {
                return Err(self.err("expecting '>'"));
            }
            self.pos += 1;
            return Ok(id);
        }

        while self.cur() != 0 && self.cur() != b'>' {
            self.pos += 1;
        }
        if self.cur() != b'>' {
            return Err(self.err("expecting '>'"));
        }
        self.pos += 1;

        let text = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| self.err_at("token name is not valid UTF-8", start))?
            .to_string();

        let Some(vocab) = self.vocab else {
            return Err(GrammarError::TokenNeedsVocabulary {
                token: text,
                offset: start,
            });
        };
        let ids = vocab.tokenize_special(&text);
        if ids.len() != 1 {
            return Err(GrammarError::TokenNotSingle {
                token: text,
                n_tokens: ids.len(),
            });
        }
        Ok(ids[0])
    }

    // -- symbol table --

    /// `get_symbol_id`: intern a name, reusing the id if it is already
    /// interned. The id is the symbol count *before* insertion.
    fn get_symbol_id(&mut self, name: &str) -> u32 {
        let next_id = self.out.symbol_ids.len() as u32;
        *self
            .out
            .symbol_ids
            .entry(name.to_string())
            .or_insert(next_id)
    }

    /// `generate_symbol_id`: a fresh `<base>_<id>` symbol, always new.
    fn generate_symbol_id(&mut self, base_name: &str) -> u32 {
        let next_id = self.out.symbol_ids.len() as u32;
        self.out
            .symbol_ids
            .insert(format!("{base_name}_{next_id}"), next_id);
        next_id
    }

    /// `add_rule`, growing the table with empty rules as needed. An empty
    /// rule left behind at the end is an undefined symbol.
    fn add_rule(&mut self, rule_id: u32, rule: GrammarRule) {
        let idx = rule_id as usize;
        if self.out.rules.len() <= idx {
            self.out.rules.resize(idx + 1, GrammarRule::new());
        }
        self.out.rules[idx] = rule;
    }

    // -- the grammar of the grammar --

    /// `parse_alternates`.
    fn parse_alternates(
        &mut self,
        rule_name: &str,
        rule_id: u32,
        is_nested: bool,
    ) -> Result<(), GrammarError> {
        let mut rule = GrammarRule::new();
        self.parse_sequence(rule_name, &mut rule, is_nested)?;
        while self.cur() == b'|' {
            rule.push(GrammarElement::new(GreType::Alt, 0));
            self.pos += 1;
            self.parse_space(true);
            self.parse_sequence(rule_name, &mut rule, is_nested)?;
        }
        rule.push(GrammarElement::new(GreType::End, 0));
        self.add_rule(rule_id, rule);
        Ok(())
    }

    /// The `handle_repetitions` lambda of `parse_sequence`, hoisted to a
    /// method. `last_sym_start` is read, never written, upstream too.
    fn handle_repetitions(
        &mut self,
        rule: &mut GrammarRule,
        rule_name: &str,
        last_sym_start: usize,
        n_prev_rules: &mut u64,
        min_times: u64,
        max_times: Option<u64>,
    ) -> Result<(), GrammarError> {
        let no_max = max_times.is_none();
        if last_sym_start == rule.len() {
            return Err(self.err("expecting preceding item to */+/?/{"));
        }

        let prev_rule: GrammarRule = rule[last_sym_start..].to_vec();

        // Total rules this repetition will generate, before nesting.
        let mut total_rules: u64 = 1;
        match max_times {
            Some(max) if max > 0 => total_rules = max,
            _ => {
                if min_times > 0 {
                    total_rules = min_times;
                }
            }
        }

        let product = n_prev_rules.saturating_mul(total_rules);
        if product >= MAX_REPETITION_THRESHOLD {
            return Err(GrammarError::RepetitionTooLarge {
                requested: product,
                limit: MAX_REPETITION_THRESHOLD,
                offset: self.pos,
            });
        }

        if min_times == 0 {
            rule.truncate(last_sym_start);
        } else {
            for _ in 1..min_times {
                rule.extend_from_slice(&prev_rule);
            }
        }

        let mut last_rec_rule_id: u32 = 0;
        // `max_times - min_times` in upstream, which wraps on `{4,2}` and
        // then loops ~2^64 times. Saturating gives zero optional copies,
        // so `{4,2}` reads as `{4}` instead of hanging.
        let n_opt = match max_times {
            None => 1,
            Some(max) => max.saturating_sub(min_times),
        };

        let mut rec_rule = prev_rule.clone();
        for i in 0..n_opt {
            rec_rule.truncate(prev_rule.len());
            let rec_rule_id = self.generate_symbol_id(rule_name);
            if i > 0 || no_max {
                rec_rule.push(GrammarElement::new(
                    GreType::RuleRef,
                    if no_max {
                        rec_rule_id
                    } else {
                        last_rec_rule_id
                    },
                ));
            }
            rec_rule.push(GrammarElement::new(GreType::Alt, 0));
            rec_rule.push(GrammarElement::new(GreType::End, 0));
            self.add_rule(rec_rule_id, rec_rule.clone());
            last_rec_rule_id = rec_rule_id;
        }
        if n_opt > 0 {
            rule.push(GrammarElement::new(GreType::RuleRef, last_rec_rule_id));
        }
        // Upstream asserts `n_prev_rules >= 1` here; it holds because
        // `total_rules` is at least 1 and the product was bounds-checked
        // above, so this is a plain assignment.
        *n_prev_rules = product;
        Ok(())
    }

    /// `parse_sequence`.
    fn parse_sequence(
        &mut self,
        rule_name: &str,
        rule: &mut GrammarRule,
        is_nested: bool,
    ) -> Result<(), GrammarError> {
        let mut last_sym_start = rule.len();
        let mut n_prev_rules: u64 = 1;

        while self.cur() != 0 {
            match self.cur() {
                b'"' => {
                    // Literal string.
                    self.pos += 1;
                    last_sym_start = rule.len();
                    n_prev_rules = 1;
                    while self.cur() != b'"' {
                        if self.cur() == 0 {
                            return Err(self.err("unexpected end of input"));
                        }
                        let value = self.parse_char()?;
                        rule.push(GrammarElement::new(GreType::Char, value));
                    }
                    self.pos += 1;
                    self.parse_space(is_nested);
                }
                b'[' => {
                    // Character class, possibly negated, possibly ranged.
                    self.pos += 1;
                    let mut start_type = GreType::Char;
                    if self.cur() == b'^' {
                        self.pos += 1;
                        start_type = GreType::CharNot;
                    }
                    last_sym_start = rule.len();
                    n_prev_rules = 1;
                    while self.cur() != b']' {
                        if self.cur() == 0 {
                            return Err(self.err("unexpected end of input"));
                        }
                        let value = self.parse_char()?;
                        // Only the FIRST element of the class carries the
                        // negation; every later one is CHAR_ALT. That is
                        // what makes `[^ab]` one negated set rather than
                        // two.
                        let gtype = if last_sym_start < rule.len() {
                            GreType::CharAlt
                        } else {
                            start_type
                        };
                        rule.push(GrammarElement::new(gtype, value));
                        if self.at(self.pos) == b'-' && self.at(self.pos + 1) != b']' {
                            if self.at(self.pos + 1) == 0 {
                                return Err(self.err("unexpected end of input"));
                            }
                            self.pos += 1;
                            let endchar = self.parse_char()?;
                            rule.push(GrammarElement::new(GreType::CharRngUpper, endchar));
                        }
                    }
                    self.pos += 1;
                    self.parse_space(is_nested);
                }
                b'<' | b'!' => {
                    // Token, or inverted token.
                    let mut gtype = GreType::Token;
                    if self.cur() == b'!' {
                        gtype = GreType::TokenNot;
                        self.pos += 1;
                    }
                    let token_id = self.parse_token()?;
                    last_sym_start = rule.len();
                    n_prev_rules = 1;
                    rule.push(GrammarElement::new(gtype, token_id));
                    self.parse_space(is_nested);
                }
                c if Self::is_word_char(c) => {
                    // Rule reference.
                    let name_end = self.parse_name()?;
                    let name = std::str::from_utf8(&self.src[self.pos..name_end])
                        .map_err(|_| self.err("rule name is not valid UTF-8"))?
                        .to_string();
                    let ref_rule_id = self.get_symbol_id(&name);
                    self.pos = name_end;
                    self.parse_space(is_nested);
                    last_sym_start = rule.len();
                    n_prev_rules = 1;
                    rule.push(GrammarElement::new(GreType::RuleRef, ref_rule_id));
                }
                b'(' => {
                    // Grouping: parse nested alternates into a synthesized
                    // rule and refer to it.
                    self.pos += 1;
                    self.parse_space(true);
                    let n_rules_before = self.out.symbol_ids.len() as u64;
                    let sub_rule_id = self.generate_symbol_id(rule_name);
                    self.parse_alternates(rule_name, sub_rule_id, true)?;
                    n_prev_rules = (self.out.symbol_ids.len() as u64 - n_rules_before).max(1);
                    last_sym_start = rule.len();
                    rule.push(GrammarElement::new(GreType::RuleRef, sub_rule_id));
                    if self.cur() != b')' {
                        return Err(self.err("expecting ')'"));
                    }
                    self.pos += 1;
                    self.parse_space(is_nested);
                }
                b'.' => {
                    last_sym_start = rule.len();
                    n_prev_rules = 1;
                    rule.push(GrammarElement::new(GreType::CharAny, 0));
                    self.pos += 1;
                    self.parse_space(is_nested);
                }
                b'*' => {
                    self.pos += 1;
                    self.parse_space(is_nested);
                    self.handle_repetitions(
                        rule,
                        rule_name,
                        last_sym_start,
                        &mut n_prev_rules,
                        0,
                        None,
                    )?;
                }
                b'+' => {
                    self.pos += 1;
                    self.parse_space(is_nested);
                    self.handle_repetitions(
                        rule,
                        rule_name,
                        last_sym_start,
                        &mut n_prev_rules,
                        1,
                        None,
                    )?;
                }
                b'?' => {
                    self.pos += 1;
                    self.parse_space(is_nested);
                    self.handle_repetitions(
                        rule,
                        rule_name,
                        last_sym_start,
                        &mut n_prev_rules,
                        0,
                        Some(1),
                    )?;
                }
                b'{' => {
                    self.pos += 1;
                    self.parse_space(is_nested);

                    if !Self::is_digit_char(self.cur()) {
                        return Err(self.err("expecting an int"));
                    }
                    let min_times = self.parse_u64()?;
                    self.parse_space(is_nested);

                    let mut max_times: Option<u64> = None;

                    if self.cur() == b'}' {
                        max_times = Some(min_times);
                        self.pos += 1;
                        self.parse_space(is_nested);
                    } else if self.cur() == b',' {
                        self.pos += 1;
                        self.parse_space(is_nested);

                        if Self::is_digit_char(self.cur()) {
                            max_times = Some(self.parse_u64()?);
                            self.parse_space(is_nested);
                        }

                        if self.cur() != b'}' {
                            return Err(self.err("expecting '}'"));
                        }
                        self.pos += 1;
                        self.parse_space(is_nested);
                    } else {
                        return Err(self.err("expecting ','"));
                    }
                    if min_times > MAX_REPETITION_THRESHOLD
                        || max_times.is_some_and(|m| m > MAX_REPETITION_THRESHOLD)
                    {
                        return Err(GrammarError::RepetitionTooLarge {
                            requested: max_times.unwrap_or(min_times),
                            limit: MAX_REPETITION_THRESHOLD,
                            offset: self.pos,
                        });
                    }
                    self.handle_repetitions(
                        rule,
                        rule_name,
                        last_sym_start,
                        &mut n_prev_rules,
                        min_times,
                        max_times,
                    )?;
                }
                _ => break,
            }
        }
        Ok(())
    }

    /// `parse_rule`: one `name ::= alternates` line.
    fn parse_rule(&mut self) -> Result<(), GrammarError> {
        let name_end = self.parse_name()?;
        let name = std::str::from_utf8(&self.src[self.pos..name_end])
            .map_err(|_| self.err("rule name is not valid UTF-8"))?
            .to_string();
        self.pos = name_end;
        self.parse_space(false);
        let rule_id = self.get_symbol_id(&name);

        if !(self.cur() == b':' && self.at(self.pos + 1) == b':' && self.at(self.pos + 2) == b'=') {
            return Err(self.err("expecting ::="));
        }
        self.pos += 3;
        self.parse_space(true);

        self.parse_alternates(&name, rule_id, false)?;

        if self.cur() == b'\r' {
            self.pos += if self.at(self.pos + 1) == b'\n' { 2 } else { 1 };
        } else if self.cur() == b'\n' {
            self.pos += 1;
        } else if self.cur() != 0 {
            return Err(self.err("expecting newline or end"));
        }
        self.parse_space(true);
        Ok(())
    }

    /// `parse`, plus the validation pass that follows it.
    fn parse_all(&mut self) -> Result<(), GrammarError> {
        self.parse_space(true);
        while self.cur() != 0 {
            self.parse_rule()?;
        }

        // Every symbol that was referenced must also have been defined.
        // A rule left empty by `add_rule`'s resize is a reference with no
        // `::=`; a RULE_REF past the end of the table is the same thing
        // when the missing symbol has the highest id.
        for id in 0..self.out.rules.len() {
            if self.out.rules[id].is_empty() {
                return Err(self.undefined(id as u32));
            }
        }
        for rule in &self.out.rules {
            for elem in rule {
                if elem.gtype == GreType::RuleRef {
                    let idx = elem.value as usize;
                    if idx >= self.out.rules.len() || self.out.rules[idx].is_empty() {
                        return Err(self.undefined(elem.value));
                    }
                }
            }
        }
        Ok(())
    }

    fn undefined(&self, rule_id: u32) -> GrammarError {
        GrammarError::UndefinedRule {
            name: self
                .out
                .symbol_name(rule_id)
                .unwrap_or("<unnamed>")
                .to_string(),
            rule_id,
        }
    }
}
