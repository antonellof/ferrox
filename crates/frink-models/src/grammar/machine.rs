//! The pushdown stack machine, transcribed from the second half of
//! llama.cpp's `src/llama-grammar.cpp`.
//!
//! A [`Grammar`] holds the *set* of parse stacks that are still viable.
//! Every stack rests on an element that consumes something -- a character
//! class or a token -- so "which characters may come next" is read
//! straight off the stack tops, and a character that no stack accepts
//! kills the parse.
//!
//! Two invariants earn their keep and are easy to get wrong:
//!
//! - `advance_stack` runs a rule reference out to *every* alternate before
//!   settling, so one input stack becomes N output stacks. A grammar with
//!   nested alternation has more viable stacks than it has rules.
//! - An empty stack means "the grammar is satisfied". It is not an error
//!   and it is not dropped; it is the only thing that lets end-of-
//!   generation be accepted (`allows_eog`).

use std::collections::HashSet;

use super::element::{GrammarElement, GrammarRule, GrammarStack, GreType, RulePos};
use super::error::GrammarError;
use super::lazy::{LazyState, LazyTriggers, TriggerStep};
use super::parser::{parse_with_vocab, GrammarVocab, ParsedGrammar};
use super::utf8::{decode_piece, PartialUtf8};

/// A compiled grammar plus the live set of viable parse stacks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar {
    rules: Vec<GrammarRule>,
    stacks: Vec<GrammarStack>,
    partial_utf8: PartialUtf8,
    /// Set by [`Grammar::into_lazy`]. `None` is llama.cpp's `lazy = false`:
    /// the grammar applies from the first token.
    lazy: Option<LazyState>,
}

impl Grammar {
    /// Parse GBNF text and start from `root_name`.
    ///
    /// `llama_grammar_init_impl(vocab, grammar_str, grammar_root, ...)`,
    /// minus the lazy-trigger machinery.
    pub fn from_str_with_root(src: &str, root_name: &str) -> Result<Self, GrammarError> {
        Self::from_str_with_vocab(src, root_name, None)
    }

    /// As [`Self::from_str_with_root`], resolving `<name>` token elements
    /// through a vocabulary.
    pub fn from_str_with_vocab(
        src: &str,
        root_name: &str,
        vocab: Option<&dyn GrammarVocab>,
    ) -> Result<Self, GrammarError> {
        let parsed = parse_with_vocab(src, vocab)?;
        Self::from_parsed(&parsed, root_name)
    }

    /// Start a machine over an already-parsed grammar.
    pub fn from_parsed(parsed: &ParsedGrammar, root_name: &str) -> Result<Self, GrammarError> {
        let start = parsed
            .symbol_id(root_name)
            .ok_or_else(|| GrammarError::MissingRoot {
                name: root_name.to_string(),
            })?;
        Self::from_rules(parsed.rules.clone(), start, |id| {
            parsed.symbol_name(id).map(str::to_string)
        })
    }

    /// Build from a raw rule table.
    ///
    /// `name_of` supplies a symbol name for diagnostics; pass `|_| None` if
    /// there is no symbol table.
    pub fn from_rules(
        rules: Vec<GrammarRule>,
        start_rule_index: u32,
        name_of: impl Fn(u32) -> Option<String>,
    ) -> Result<Self, GrammarError> {
        let n_rules = rules.len();

        // Every rule must be terminated, or `advance_stack` walks off the
        // end of one. llama.cpp guarantees this by construction and does
        // not check; a hand-built table can violate it.
        for (i, rule) in rules.iter().enumerate() {
            if rule.last().map(|e| e.gtype) != Some(GreType::End) {
                return Err(GrammarError::UndefinedRule {
                    name: name_of(i as u32).unwrap_or_else(|| "<unnamed>".into()),
                    rule_id: i as u32,
                });
            }
        }

        // Every rule reference must resolve.
        for rule in rules.iter() {
            for elem in rule {
                if elem.gtype == GreType::RuleRef {
                    let idx = elem.value as usize;
                    if idx >= n_rules || rules[idx].is_empty() {
                        return Err(GrammarError::UndefinedRule {
                            name: name_of(elem.value).unwrap_or_else(|| "<unnamed>".into()),
                            rule_id: elem.value,
                        });
                    }
                }
            }
        }

        if start_rule_index as usize >= n_rules {
            return Err(GrammarError::MissingRoot {
                name: name_of(start_rule_index).unwrap_or_else(|| "root".into()),
            });
        }

        detect_left_recursion_all(&rules, &name_of)?;

        // Loop over the alternates of the start rule to build the initial
        // stacks.
        let mut stacks: Vec<GrammarStack> = Vec::new();
        let mut pos = RulePos::new(start_rule_index, 0);
        loop {
            let mut stack = GrammarStack::new();
            if !elem(&rules, pos).is_end_of_sequence() {
                stack.push(pos);
            }
            advance_stack(&rules, &stack, &mut stacks)?;
            while !elem(&rules, pos).is_end_of_sequence() {
                pos = pos.next();
            }
            if elem(&rules, pos).gtype == GreType::Alt {
                pos = pos.next();
            } else {
                break;
            }
        }

        Ok(Grammar {
            rules,
            stacks,
            partial_utf8: PartialUtf8::default(),
            lazy: None,
        })
    }

    /// Make this grammar LAZY: it constrains nothing until one of
    /// `triggers` matches the output.
    ///
    /// `llama_grammar_init_impl`'s `lazy` / `trigger_patterns` /
    /// `trigger_tokens` arguments. See [`super::lazy`] for what the
    /// triggers match against and what happens to the text before one.
    ///
    /// Refuses an empty trigger set: upstream allows it, and the result is
    /// a grammar that can never switch on -- an unconstrained generation
    /// that looks constrained from the outside.
    pub fn into_lazy(mut self, triggers: LazyTriggers) -> Result<Self, GrammarError> {
        if triggers.is_empty() {
            return Err(GrammarError::LazyWithoutTriggers);
        }
        self.lazy = Some(LazyState::new(triggers));
        Ok(self)
    }

    /// Whether this grammar waits for a trigger before it constrains.
    pub fn is_lazy(&self) -> bool {
        self.lazy.is_some()
    }

    /// Whether this grammar is lazy and has NOT yet been triggered, i.e.
    /// constrains nothing right now.
    ///
    /// `llama_grammar::awaiting_trigger`, which is the first thing both
    /// `llama_grammar_apply_impl` and `llama_grammar_accept_impl` test.
    pub fn is_awaiting_trigger(&self) -> bool {
        self.lazy.as_ref().is_some_and(LazyState::awaiting)
    }

    /// The output accumulated while awaiting a trigger. Empty once one has
    /// fired, and for a grammar that is not lazy.
    pub fn trigger_buffer(&self) -> &[u8] {
        self.lazy.as_ref().map_or(&[], LazyState::buffer)
    }

    /// The compiled rule table.
    pub fn rules(&self) -> &[GrammarRule] {
        &self.rules
    }

    /// The stacks still viable after everything accepted so far.
    pub fn stacks(&self) -> &[GrammarStack] {
        &self.stacks
    }

    /// The partial UTF-8 sequence carried over from the last piece.
    pub fn partial_utf8(&self) -> PartialUtf8 {
        self.partial_utf8
    }

    /// True when at least one viable parse is complete, so an
    /// end-of-generation token is allowed.
    ///
    /// `llama_grammar_apply_impl`'s `allow_eog`. An empty stack is a
    /// finished parse.
    ///
    /// A lazy grammar that has not triggered allows it unconditionally:
    /// upstream's `awaiting_trigger` early-return sits *above* both the
    /// `allow_eog` mask and the abort in `llama_grammar_accept_impl`, so
    /// an untriggered grammar has no opinion about ending. It has not been
    /// applied; a generation that never calls a tool must be able to stop.
    ///
    /// Unless its trigger is MANDATORY, which is this repo's own addition
    /// and the one place it departs from upstream here: see
    /// [`LazyTriggers::mandatory`].
    pub fn allows_eog(&self) -> bool {
        if self.is_awaiting_trigger() {
            return !self.trigger_is_mandatory();
        }
        self.stacks.iter().any(|s| s.is_empty())
    }

    /// Whether this grammar's trigger must fire before the generation may
    /// end. False for every grammar that is not lazy, and for every lazy
    /// grammar whose triggers were not marked
    /// [`mandatory`](LazyTriggers::mandatory).
    pub fn trigger_is_mandatory(&self) -> bool {
        self.lazy.as_ref().is_some_and(LazyState::is_mandatory)
    }

    /// True when no parse is viable at all. Reaching this means a token was
    /// accepted that should have been masked out.
    pub fn is_dead(&self) -> bool {
        self.stacks.is_empty()
    }

    /// Advance every stack over one code point.
    ///
    /// `llama_grammar_accept`. Stacks that cannot take the character are
    /// dropped; a stack resting on a token element is dropped too, since a
    /// token element consumes a whole token, never a character.
    pub fn accept_codepoint(&mut self, chr: u32) -> Result<(), GrammarError> {
        let mut next: Vec<GrammarStack> = Vec::with_capacity(self.stacks.len());
        for stack in &self.stacks {
            accept_chr(&self.rules, stack, chr, &mut next)?;
        }
        self.stacks = next;
        Ok(())
    }

    /// Accept a piece of generated text, carrying any partial UTF-8
    /// sequence across the call.
    ///
    /// `llama_grammar_accept_str`. Errors if nothing survives.
    pub fn accept_str(&mut self, piece: &str) -> Result<(), GrammarError> {
        self.accept_bytes(piece.as_bytes())
    }

    /// As [`Self::accept_str`], for a piece that is not valid UTF-8 on its
    /// own.
    ///
    /// This is the real signature: a BPE token piece is bytes, and a piece
    /// holding one byte of a multi-byte character is not a `str` at all.
    /// llama.cpp passes `std::string`, which has the same freedom.
    pub fn accept_bytes(&mut self, piece: &[u8]) -> Result<(), GrammarError> {
        let (code_points, partial) = decode_piece(piece, self.partial_utf8);
        // The vector is 0-terminated; the terminator is not a code point.
        for &cp in &code_points[..code_points.len() - 1] {
            self.accept_codepoint(cp)?;
        }
        self.partial_utf8 = partial;
        if self.stacks.is_empty() {
            return Err(GrammarError::NoViableStack {
                piece: String::from_utf8_lossy(piece).into_owned(),
            });
        }
        Ok(())
    }

    /// Accept a sampled token, given its decoded piece.
    ///
    /// `llama_grammar_accept_token`. This is not `accept_str` plus a token
    /// id: a stack resting on a `Token` / `TokenNot` element matches on the
    /// **id** and ignores the piece entirely, which is how a grammar can
    /// require a specific special token whose text is unreachable through
    /// its characters.
    ///
    /// While a lazy grammar is awaiting its trigger this does NOT advance
    /// the parse: the token goes to the trigger buffer instead, and the
    /// grammar is fed only once a trigger fires, and only from where it
    /// says. That dispatch lives here, on the one accept path, rather than
    /// in a lazy-aware twin of it.
    pub fn accept_token(&mut self, token: u32, piece: &[u8]) -> Result<(), GrammarError> {
        if self.is_awaiting_trigger() {
            let step = match self.lazy.as_mut() {
                Some(lazy) => lazy.observe(token, piece)?,
                None => return Err(GrammarError::Internal("lazy state vanished mid-accept")),
            };
            let replay = match step {
                TriggerStep::Awaiting => return Ok(()),
                TriggerStep::Fired(replay) => replay,
            };
            for (tok, piece) in replay {
                self.accept_token_now(tok, &piece)?;
            }
            return Ok(());
        }
        self.accept_token_now(token, piece)
    }

    /// `llama_grammar_accept_token`: the acceptance itself, with no
    /// trigger check. The replay above is upstream's direct call to it.
    fn accept_token_now(&mut self, token: u32, piece: &[u8]) -> Result<(), GrammarError> {
        let (code_points, partial) = decode_piece(piece, self.partial_utf8);
        let chars = &code_points[..code_points.len() - 1];

        let mut stacks_new: Vec<GrammarStack> = Vec::with_capacity(self.stacks.len());

        for stack in &self.stacks {
            // A completed parse cannot consume another token; only an
            // end-of-generation token, handled by `accept_eog`.
            let Some(&top) = stack.last() else {
                continue;
            };
            let top_elem = elem(&self.rules, top);

            if matches!(top_elem.gtype, GreType::Token | GreType::TokenNot) {
                if match_token(top_elem, token) {
                    let mut new_stack = stack[..stack.len() - 1].to_vec();
                    if !elem(&self.rules, top.next()).is_end_of_sequence() {
                        new_stack.push(top.next());
                    }
                    advance_stack(&self.rules, &new_stack, &mut stacks_new)?;
                }
                continue;
            }

            let mut current: Vec<GrammarStack> = vec![stack.clone()];
            for &cp in chars {
                let mut next: Vec<GrammarStack> = Vec::new();
                for cur in &current {
                    accept_chr(&self.rules, cur, cp, &mut next)?;
                }
                current = next;
                if current.is_empty() {
                    break;
                }
            }
            for surviving in current {
                if !stacks_new.contains(&surviving) {
                    stacks_new.push(surviving);
                }
            }
        }

        self.stacks = stacks_new;
        self.partial_utf8 = partial;

        if self.stacks.is_empty() {
            return Err(GrammarError::NoViableStack {
                piece: String::from_utf8_lossy(piece).into_owned(),
            });
        }
        Ok(())
    }

    /// Accept an end-of-generation token.
    ///
    /// The EOG branch of `llama_grammar_accept_impl`, which aborts if no
    /// stack is empty. Here it is a refusal: EOG at a point where the
    /// grammar is unsatisfied means the mask let it through.
    ///
    /// Upstream's EOG branch sits *below* the `awaiting_trigger` check, so
    /// an untriggered lazy grammar never reaches it: an EOG token is
    /// buffered like any other. A caller that has the token's piece --
    /// [`crate::grammar_sampler::GrammarSampler`] does -- must therefore
    /// send it to [`Self::accept_token`] while [`Self::is_awaiting_trigger`],
    /// not here.
    pub fn accept_eog(&mut self) -> Result<(), GrammarError> {
        if self.allows_eog() {
            Ok(())
        } else {
            Err(GrammarError::NoViableStack {
                piece: "<eog>".to_string(),
            })
        }
    }
}

/// `rules[pos.rule][pos.index]`.
///
/// Out of range is unreachable for a validated table: every rule ends with
/// `End`, every walk stops there, and `from_rules` checks it. Returning
/// `End` rather than panicking keeps a malformed hand-built table from
/// taking the process down.
#[inline]
pub(crate) fn elem(rules: &[GrammarRule], pos: RulePos) -> GrammarElement {
    rules
        .get(pos.rule as usize)
        .and_then(|r| r.get(pos.index as usize))
        .copied()
        .unwrap_or(GrammarElement::new(GreType::End, 0))
}

/// `llama_grammar_match_char`: does `chr` satisfy the character class at
/// `pos`? Returns the verdict and the position just past the class.
///
/// The negation lives on the *first* element of the class only, so this
/// walks the whole `CharAlt` chain accumulating `found`, and inverts once
/// at the end. Testing each element against its own type instead would
/// make `[^ab]` mean "not a, or not b", which is every character.
pub(crate) fn match_char(
    rules: &[GrammarRule],
    mut pos: RulePos,
    chr: u32,
) -> Result<(bool, RulePos), GrammarError> {
    let first = elem(rules, pos);
    let is_positive_char = matches!(first.gtype, GreType::Char | GreType::CharAny);
    if !is_positive_char && first.gtype != GreType::CharNot {
        return Err(GrammarError::Internal(
            "match_char called on an element that is not a character class",
        ));
    }

    let mut found = false;
    loop {
        let cur = elem(rules, pos);
        let nxt = elem(rules, pos.next());
        if nxt.gtype == GreType::CharRngUpper {
            // Inclusive range, e.g. [a-z].
            found = found || (cur.value <= chr && chr <= nxt.value);
            pos = pos.advance(2);
        } else if cur.gtype == GreType::CharAny {
            found = true;
            pos = pos.next();
        } else {
            // Exact match, e.g. [a] or "a".
            found = found || cur.value == chr;
            pos = pos.next();
        }
        if elem(rules, pos).gtype != GreType::CharAlt {
            break;
        }
    }

    Ok((found == is_positive_char, pos))
}

/// `llama_grammar_match_partial_char`: could *some* continuation of this
/// partial UTF-8 sequence satisfy the class at `pos`?
///
/// This is what keeps a token that ends mid-codepoint viable. Without it,
/// every multi-byte character a BPE vocabulary splits across two tokens
/// would be unreachable under any grammar.
pub(crate) fn match_partial_char(
    rules: &[GrammarRule],
    mut pos: RulePos,
    partial_utf8: PartialUtf8,
) -> Result<bool, GrammarError> {
    let first = elem(rules, pos);
    let is_positive_char = matches!(first.gtype, GreType::Char | GreType::CharAny);
    if !is_positive_char && first.gtype != GreType::CharNot {
        return Err(GrammarError::Internal(
            "match_partial_char called on an element that is not a character class",
        ));
    }

    let partial_value = partial_utf8.value;
    let n_remain = partial_utf8.n_remain;

    // Invalid sequence, or a 7-bit character split across two bytes
    // (overlong): no continuation can be legal UTF-8.
    if n_remain < 0 || (n_remain == 1 && partial_value < 2) {
        return Ok(false);
    }
    // A UTF-8 sequence never has more than 3 continuation bytes, so this
    // is unreachable from `decode_piece`. It is a guard against a
    // hand-built `PartialUtf8`, where upstream's `1 << (n_remain * 6)`
    // would be undefined behaviour and Rust's would panic.
    if n_remain > 3 {
        return Ok(false);
    }

    // The range of code points this partial sequence could complete to.
    let shift = (n_remain * 6) as u32;
    let mut low = partial_value << shift;
    let high = low | ((1u32 << shift) - 1);

    if low == 0 {
        if n_remain == 2 {
            low = 1 << 11;
        } else if n_remain == 3 {
            low = 1 << 16;
        }
    }

    loop {
        let cur = elem(rules, pos);
        let nxt = elem(rules, pos.next());
        if nxt.gtype == GreType::CharRngUpper {
            if cur.value <= high && low <= nxt.value {
                return Ok(is_positive_char);
            }
            pos = pos.advance(2);
        } else if cur.gtype == GreType::CharAny {
            // Upstream returns an unconditional `true` here, not
            // `is_positive_char`. `.` is never negated, so they agree.
            return Ok(true);
        } else {
            if low <= cur.value && cur.value <= high {
                return Ok(is_positive_char);
            }
            pos = pos.next();
        }
        if elem(rules, pos).gtype != GreType::CharAlt {
            break;
        }
    }

    Ok(!is_positive_char)
}

/// `llama_grammar_match_token`.
pub(crate) fn match_token(pos_elem: GrammarElement, token: u32) -> bool {
    match pos_elem.gtype {
        GreType::Token => pos_elem.value == token,
        GreType::TokenNot => pos_elem.value != token,
        _ => false,
    }
}

/// `llama_grammar_advance_stack`: expand rule references until every
/// resulting stack rests on something that consumes input.
///
/// Appends to `new_stacks`, skipping duplicates, exactly as upstream does.
pub(crate) fn advance_stack(
    rules: &[GrammarRule],
    stack: &GrammarStack,
    new_stacks: &mut Vec<GrammarStack>,
) -> Result<(), GrammarError> {
    let mut todo: Vec<GrammarStack> = vec![stack.clone()];
    let mut seen: HashSet<GrammarStack> = HashSet::new();

    while let Some(curr_stack) = todo.pop() {
        if !seen.insert(curr_stack.clone()) {
            continue;
        }

        let Some(&pos) = curr_stack.last() else {
            // An empty stack is a completed parse, and is kept.
            if !new_stacks.contains(&curr_stack) {
                new_stacks.push(curr_stack);
            }
            continue;
        };

        let pos_elem = elem(rules, pos);
        match pos_elem.gtype {
            GreType::RuleRef => {
                let rule_id = pos_elem.value;
                let mut subpos = RulePos::new(rule_id, 0);
                loop {
                    // The stack without its top, plus the continuation
                    // after this reference, plus the alternate's start.
                    let mut next_stack = curr_stack[..curr_stack.len() - 1].to_vec();
                    if !elem(rules, pos.next()).is_end_of_sequence() {
                        next_stack.push(pos.next());
                    }
                    if !elem(rules, subpos).is_end_of_sequence() {
                        next_stack.push(subpos);
                    }
                    todo.push(next_stack);

                    while !elem(rules, subpos).is_end_of_sequence() {
                        subpos = subpos.next();
                    }
                    if elem(rules, subpos).gtype == GreType::Alt {
                        subpos = subpos.next();
                    } else {
                        break;
                    }
                }
            }
            t if t.is_stack_terminal() => {
                if !new_stacks.contains(&curr_stack) {
                    new_stacks.push(curr_stack);
                }
            }
            _ => {
                // End / Alt / CharAlt / CharRngUpper. Upstream aborts the
                // process; a stack is never left on one of these.
                return Err(GrammarError::Internal(
                    "parse stack came to rest on END, ALT, CHAR_ALT or CHAR_RNG_UPPER",
                ));
            }
        }
    }
    Ok(())
}

/// `llama_grammar_accept_chr`: advance one stack over one code point.
pub(crate) fn accept_chr(
    rules: &[GrammarRule],
    stack: &GrammarStack,
    chr: u32,
    new_stacks: &mut Vec<GrammarStack>,
) -> Result<(), GrammarError> {
    let Some(&pos) = stack.last() else {
        return Ok(());
    };

    let pos_elem = elem(rules, pos);
    // A token element consumes a token, not a character; such a stack
    // simply does not advance here.
    if matches!(pos_elem.gtype, GreType::Token | GreType::TokenNot) {
        return Ok(());
    }

    let (matched, after) = match_char(rules, pos, chr)?;
    if matched {
        let mut new_stack = stack[..stack.len() - 1].to_vec();
        if !elem(rules, after).is_end_of_sequence() {
            new_stack.push(after);
        }
        advance_stack(rules, &new_stack, new_stacks)?;
    }
    Ok(())
}

// -- left recursion --

/// `llama_grammar_detect_left_recursion`, run over every rule.
fn detect_left_recursion_all(
    rules: &[GrammarRule],
    name_of: &impl Fn(u32) -> Option<String>,
) -> Result<(), GrammarError> {
    let n = rules.len();
    let mut visited = vec![false; n];
    let mut in_progress = vec![false; n];
    let mut may_be_empty = vec![false; n];
    for i in 0..n {
        if visited[i] {
            continue;
        }
        if detect_left_recursion(rules, i, &mut visited, &mut in_progress, &mut may_be_empty) {
            return Err(GrammarError::LeftRecursion {
                rule_id: i as u32,
                name: name_of(i as u32),
            });
        }
    }
    Ok(())
}

fn detect_left_recursion(
    rules: &[GrammarRule],
    rule_index: usize,
    visited: &mut [bool],
    in_progress: &mut [bool],
    may_be_empty: &mut [bool],
) -> bool {
    if in_progress[rule_index] {
        return true;
    }
    in_progress[rule_index] = true;

    let rule = &rules[rule_index];

    // First: can this rule produce the empty string? An alternate whose
    // very first element ends the sequence is empty.
    let mut at_rule_start = true;
    for e in rule {
        if e.is_end_of_sequence() {
            if at_rule_start {
                may_be_empty[rule_index] = true;
                break;
            }
            at_rule_start = true;
        } else {
            at_rule_start = false;
        }
    }

    // Second: recurse into leftmost non-terminals, and into the next one
    // along for as long as the previous one may be empty.
    let mut recurse_into_nonterminal = true;
    for e in rule {
        if e.gtype == GreType::RuleRef && recurse_into_nonterminal {
            let target = e.value as usize;
            if target >= rules.len() {
                continue;
            }
            if detect_left_recursion(rules, target, visited, in_progress, may_be_empty) {
                return true;
            }
            if !may_be_empty[target] {
                recurse_into_nonterminal = false;
            }
        } else {
            // A new alternate starts fresh; anything else has consumed
            // input, so nothing after it is leftmost any more.
            recurse_into_nonterminal = e.is_end_of_sequence();
        }
    }

    in_progress[rule_index] = false;
    visited[rule_index] = true;
    false
}
