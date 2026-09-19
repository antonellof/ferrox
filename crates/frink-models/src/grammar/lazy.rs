//! Lazy grammars: the trigger half of llama.cpp's `llama_grammar`.
//!
//! A port of the `lazy` / `awaiting_trigger` / `trigger_buffer` /
//! `trigger_buffer_positions` / `trigger_tokens` / `trigger_patterns`
//! fields of `src/llama-grammar.h`, of
//! `llama_grammar_trigger_pattern::find`, and of the `awaiting_trigger`
//! branch of `llama_grammar_accept_impl`. The trigger-kind mapping in
//! [`LazyTriggers`] is `common/sampling.cpp`'s
//! `COMMON_GRAMMAR_TRIGGER_TYPE_*` switch.
//!
//! # What a lazy grammar is
//!
//! An ordinary grammar constrains from the first token. That is wrong for
//! a tool call: the model is allowed to say "let me look that up" before
//! it emits one, and a grammar applied from token zero forbids the prose.
//! A lazy grammar therefore starts **not constraining at all** -- the mask
//! is a no-op, every token is legal, and the grammar is not advanced --
//! and switches on when a trigger matches. This is why it is a separate
//! mechanism rather than a flag on the sampler: the grammar's state before
//! the trigger is not "at the start of the parse", it is "not applied".
//!
//! # The three things that are easy to get wrong
//!
//! - **What the trigger matches against.** Not the last token's piece: the
//!   ACCUMULATED text of every token seen since generation began. A
//!   trigger word like `<tool_call>` is several tokens, and no single
//!   piece contains it.
//! - **What happens to the text before the trigger.** It is NOT fed to the
//!   grammar. The grammar is fed the buffer from the match start onward,
//!   by replaying the buffered *tokens* whose byte spans overlap that
//!   point -- so a token that straddles the match start is replayed with
//!   its piece truncated to the overlapping part, and keeps its token id.
//! - **How a trigger token differs from a trigger pattern.** A trigger
//!   token matches one token **id**, exactly, and when it fires the whole
//!   buffer is DISCARDED and the grammar is fed only that token. A trigger
//!   pattern matches text, and when it fires the grammar is fed the
//!   buffered text from the match start on. So `<tool_call>` as a single
//!   special token and `<tool_call>` as a pattern seed the grammar
//!   identically only because the pattern's match starts where the token
//!   does.
//!
//! # Where the match starts
//!
//! `llama_grammar_trigger_pattern::find` returns the position of the first
//! capture group that matched something non-empty, and the position of the
//! whole match when there is none. That is the mechanism behind upstream's
//! gpt-oss triggers: `<\|start\|>assistant(\s+to)` fires on the whole
//! phrase but hands the grammar only `\s+to` onward.
//!
//! # Deviations from upstream, and why
//!
//! - Upstream matches with `std::regex` over raw `std::string` bytes.
//!   [`fancy_regex`] -- already this crate's engine, and the one that can
//!   compile upstream's `>>>(?!all)` -- matches over `&str`, so the buffer
//!   is matched as its longest valid UTF-8 prefix. A token piece that ends
//!   mid-codepoint contributes no characters until the next piece
//!   completes it, which is the same rule the grammar machine already
//!   applies to partial UTF-8; the one observable difference is a
//!   `$`-anchored pattern, which sees the end of the buffer one token
//!   earlier than upstream would when the buffer's tail is a half
//!   character.
//! - A trigger pattern this engine cannot compile is
//!   [`GrammarError::TriggerPatternInvalid`], not a silently inert
//!   grammar.
//! - A lazy grammar with no triggers at all can never fire, which makes it
//!   an unconstrained generation wearing a grammar. Upstream permits it;
//!   [`crate::grammar::Grammar::into_lazy`] refuses it.
//!
//! # Not ported
//!
//! Upstream keeps `trigger_tokens` on NON-lazy grammars too, "to force
//! printing of special trigger tokens". That is a detokenizer decision
//! about what reaches the client, not a grammar decision about what may be
//! sampled, and it lives in llama.cpp's server rather than in the grammar.
//! Nothing here needs it and it is not represented.

use std::fmt;

use fancy_regex::Regex;

use super::error::GrammarError;

/// A regular expression that switches a lazy grammar on.
///
/// `llama_grammar_trigger_pattern`. Holds two compiled forms because
/// upstream's `find` uses two: `std::regex_match` (the whole buffer must
/// match) is tried first for a pattern written `^...$`, and
/// `std::regex_search` (match anywhere) is the fallback for every pattern.
/// The two can disagree about *which* alternative matches, and therefore
/// about where the first capture group starts.
#[derive(Clone)]
pub struct TriggerPattern {
    pattern: String,
    search: Regex,
    full: Option<Regex>,
}

impl TriggerPattern {
    /// Compile `pattern`.
    pub fn new(pattern: &str) -> Result<Self, GrammarError> {
        let search = compile(pattern)?;
        // `^...$` is upstream's signal to try a whole-buffer match first.
        // `\A(?:...)\z` is that, and the non-capturing wrapper leaves
        // every capture group's number -- and so `start_of_match` -- alone.
        let full = if pattern.starts_with('^') && pattern.ends_with('$') {
            Some(compile(&format!(r"\A(?:{pattern})\z"))?)
        } else {
            None
        };
        Ok(Self {
            pattern: pattern.to_string(),
            search,
            full,
        })
    }

    /// The pattern as written.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Where the grammar should start reading `input`, or `None` if this
    /// pattern does not match it yet.
    ///
    /// `llama_grammar_trigger_pattern::find`, whose `npos` is this `None`.
    pub fn find(&self, input: &str) -> Result<Option<usize>, GrammarError> {
        if let Some(full) = &self.full {
            if let Some(caps) = self.captures(full, input)? {
                return Ok(Some(start_of_match(&caps)));
            }
        }
        match self.captures(&self.search, input)? {
            Some(caps) => Ok(Some(start_of_match(&caps))),
            None => Ok(None),
        }
    }

    fn captures<'t>(
        &self,
        re: &Regex,
        input: &'t str,
    ) -> Result<Option<fancy_regex::Captures<'t>>, GrammarError> {
        re.captures(input)
            .map_err(|e| GrammarError::TriggerPatternFailed {
                pattern: self.pattern.clone(),
                reason: e.to_string(),
            })
    }
}

impl fmt::Debug for TriggerPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TriggerPattern")
            .field(&self.pattern)
            .finish()
    }
}

/// Two patterns are the same trigger when they are the same source. The
/// compiled forms are a function of it.
impl PartialEq for TriggerPattern {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
    }
}

impl Eq for TriggerPattern {}

fn compile(pattern: &str) -> Result<Regex, GrammarError> {
    Regex::new(pattern).map_err(|e| GrammarError::TriggerPatternInvalid {
        pattern: pattern.to_string(),
        reason: e.to_string(),
    })
}

/// `find_start_pos`: the first capture group that matched something
/// non-empty, else the whole match.
///
/// A group that participated but matched the empty string is skipped --
/// upstream's test is `match.length(i) > 0`, not "did it participate".
fn start_of_match(caps: &fancy_regex::Captures<'_>) -> usize {
    for i in 1..caps.len() {
        if let Some(m) = caps.get(i) {
            if m.end() > m.start() {
                return m.start();
            }
        }
    }
    caps.get(0).map(|m| m.start()).unwrap_or(0)
}

/// The set of things that switch a lazy grammar on.
///
/// The constructors are `common/sampling.cpp`'s mapping from the four
/// `COMMON_GRAMMAR_TRIGGER_TYPE_*` kinds onto the two the grammar itself
/// knows: a word is an escaped pattern, a full pattern is an anchored one,
/// and only tokens stay tokens.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LazyTriggers {
    tokens: Vec<u32>,
    patterns: Vec<TriggerPattern>,
    mandatory: bool,
}

impl LazyTriggers {
    pub fn new() -> Self {
        Self::default()
    }

    /// The generation may NOT end before a trigger has fired.
    ///
    /// **This is not upstream.** llama.cpp's lazy grammars are always
    /// optional: `awaiting_trigger` masks nothing, end-of-generation
    /// included, so a model that never triggers simply produces
    /// unconstrained text. Upstream enforces OpenAI's `tool_choice:
    /// "required"` a different way -- with an EAGER grammar
    /// (`grammar_lazy = false` for `COMMON_CHAT_TOOL_CHOICE_REQUIRED` in
    /// `common/chat.cpp`) whose root IS a tool call, so the very first
    /// token is already inside one.
    ///
    /// That trade does not survive this server: several checkpoint
    /// families open a reasoning block in the PROMPT
    /// (`frink_server::policy::parser::reasoning`'s `always_open`), so
    /// the model's first token is inside `<think>`, and a grammar that
    /// forces a tool call there produces a call that this server's own
    /// reasoning parser then reads as thinking. Marking the trigger
    /// mandatory keeps the prefix free -- thinking, prose, whatever the
    /// checkpoint does -- while making the *turn* unable to end until a
    /// call has begun, after which the grammar forces it to be complete
    /// and schema-valid.
    ///
    /// The cost is stated where it is wired: a model that never begins a
    /// call runs to `max_tokens` instead of stopping. For a caller who
    /// said `required`, that is a visible failure rather than an answer
    /// that quietly ignores what they asked for.
    pub fn mandatory(mut self) -> Self {
        self.mandatory = true;
        self
    }

    /// Whether the generation may end before a trigger fires.
    pub fn is_mandatory(&self) -> bool {
        self.mandatory
    }

    /// `COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN`: this token id, exactly.
    pub fn with_token(mut self, id: u32) -> Self {
        self.tokens.push(id);
        self
    }

    /// `COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN`: a regex, matched anywhere in
    /// the accumulated output.
    pub fn with_pattern(mut self, pattern: &str) -> Result<Self, GrammarError> {
        self.patterns.push(TriggerPattern::new(pattern)?);
        Ok(self)
    }

    /// `COMMON_GRAMMAR_TRIGGER_TYPE_WORD`: a literal string, matched
    /// anywhere in the accumulated output. `regex_escape(word)`.
    pub fn with_word(mut self, word: &str) -> Result<Self, GrammarError> {
        self.patterns
            .push(TriggerPattern::new(&fancy_regex::escape(word))?);
        Ok(self)
    }

    /// `COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL`: a regex that must match
    /// the whole accumulated output. Anchored the way upstream anchors it,
    /// including its empty case.
    pub fn with_full_pattern(mut self, pattern: &str) -> Result<Self, GrammarError> {
        let anchored = if pattern.is_empty() {
            "^$".to_string()
        } else {
            let head = if pattern.starts_with('^') { "" } else { "^" };
            let tail = if pattern.ends_with('$') { "" } else { "$" };
            format!("{head}{pattern}{tail}")
        };
        self.patterns.push(TriggerPattern::new(&anchored)?);
        Ok(self)
    }

    /// True when nothing here could ever fire.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty() && self.patterns.is_empty()
    }

    /// The trigger token ids.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The trigger patterns.
    pub fn patterns(&self) -> &[TriggerPattern] {
        &self.patterns
    }
}

/// What one observed token did to a not-yet-triggered grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerStep {
    /// Nothing matched. The token was buffered and the grammar is
    /// untouched -- it is not advanced by unconstrained output.
    Awaiting,
    /// A trigger fired. These `(token, piece)` pairs are what the grammar
    /// must now be fed, in order, as if they had been sampled under it.
    /// A piece here can be a *fragment* of the token's real piece: the
    /// token that straddles the match start is truncated to the part at or
    /// after it.
    Fired(Vec<(u32, Vec<u8>)>),
}

/// The lazy state carried by a grammar that is waiting for a trigger.
///
/// Held by [`crate::grammar::Grammar`], which is what makes the trigger
/// check unskippable: there is one `accept_token` and it consults this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LazyState {
    triggers: LazyTriggers,
    /// `awaiting_trigger`. Starts true; once false, never true again --
    /// a grammar does not un-trigger.
    awaiting: bool,
    /// `trigger_buffer`. Bytes, not `String`: a BPE piece can end
    /// mid-codepoint and the byte spans below must stay exact.
    buffer: Vec<u8>,
    /// `trigger_buffer_positions`: `(token, start, end)` in `buffer`.
    spans: Vec<(u32, usize, usize)>,
}

impl LazyState {
    /// A grammar that has not triggered yet.
    pub fn new(triggers: LazyTriggers) -> Self {
        Self {
            triggers,
            awaiting: true,
            buffer: Vec::new(),
            spans: Vec::new(),
        }
    }

    /// Whether the grammar is still unapplied.
    pub fn awaiting(&self) -> bool {
        self.awaiting
    }

    /// The output accumulated since generation began, and not yet handed
    /// to the grammar. Empty once a trigger has fired.
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// The triggers being waited on.
    pub fn triggers(&self) -> &LazyTriggers {
        &self.triggers
    }

    /// Whether the generation may not end before a trigger fires. See
    /// [`LazyTriggers::mandatory`], which is not upstream.
    pub fn is_mandatory(&self) -> bool {
        self.triggers.mandatory
    }

    /// Offer one sampled token to the triggers.
    ///
    /// The `awaiting_trigger` branch of `llama_grammar_accept_impl`. The
    /// caller must not have applied the grammar to this token, and must
    /// feed the grammar exactly what [`TriggerStep::Fired`] carries.
    pub fn observe(&mut self, token: u32, piece: &[u8]) -> Result<TriggerStep, GrammarError> {
        debug_assert!(self.awaiting, "observe on a grammar that already fired");

        // A trigger TOKEN throws the buffer away: the prose before it is
        // not part of the constrained output, and the grammar starts at
        // the token itself.
        if self.triggers.tokens.contains(&token) {
            self.fire();
            return Ok(TriggerStep::Fired(vec![(token, piece.to_vec())]));
        }

        let start = self.buffer.len();
        self.buffer.extend_from_slice(piece);
        self.spans.push((token, start, self.buffer.len()));

        // Patterns match the whole accumulated buffer, this token
        // included -- the trigger is a property of the output, not of the
        // token that completed it.
        let Some(at) = self.find_trigger()? else {
            return Ok(TriggerStep::Awaiting);
        };

        // Replay every token whose span reaches past the match start,
        // truncating the one that straddles it.
        let mut replay = Vec::new();
        for &(tok, tok_start, tok_end) in &self.spans {
            if tok_end <= at {
                continue;
            }
            let from = tok_start.max(at);
            replay.push((tok, self.buffer[from..tok_end].to_vec()));
        }
        self.fire();
        Ok(TriggerStep::Fired(replay))
    }

    /// The first pattern that matches, and where it says to start.
    fn find_trigger(&self) -> Result<Option<usize>, GrammarError> {
        if self.triggers.patterns.is_empty() {
            return Ok(None);
        }
        // The tail of the buffer can be half a character; it is not text
        // yet, and the token that completes it will bring it back here.
        let text = match std::str::from_utf8(&self.buffer) {
            Ok(s) => s,
            Err(e) => {
                let valid = e.valid_up_to();
                // SAFETY-adjacent: `valid_up_to` is by definition the
                // length of a valid prefix, so this cannot fail.
                std::str::from_utf8(&self.buffer[..valid]).unwrap_or("")
            }
        };
        for pattern in &self.triggers.patterns {
            if let Some(at) = pattern.find(text)? {
                return Ok(Some(at));
            }
        }
        Ok(None)
    }

    fn fire(&mut self) {
        self.awaiting = false;
        self.buffer.clear();
        self.spans.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triggers_with(pattern: &str) -> LazyState {
        LazyState::new(LazyTriggers::new().with_pattern(pattern).unwrap())
    }

    /// The headline rule: a trigger is matched against the ACCUMULATED
    /// output. `<tool_call>` arrives as three pieces and no piece
    /// contains it.
    #[test]
    fn a_pattern_matches_across_token_boundaries() {
        let mut s = triggers_with("<tool_call>");
        assert_eq!(s.observe(1, b"<tool").unwrap(), TriggerStep::Awaiting);
        assert_eq!(s.observe(2, b"_ca").unwrap(), TriggerStep::Awaiting);
        let TriggerStep::Fired(replay) = s.observe(3, b"ll>").unwrap() else {
            panic!("the buffer now holds the whole trigger word");
        };
        assert_eq!(
            replay,
            vec![
                (1, b"<tool".to_vec()),
                (2, b"_ca".to_vec()),
                (3, b"ll>".to_vec())
            ]
        );
        assert!(!s.awaiting());
        assert!(s.buffer().is_empty());
    }

    /// Text before the match start is dropped, and the token that
    /// straddles the start is replayed as a FRAGMENT, keeping its id.
    #[test]
    fn text_before_the_trigger_is_dropped_and_the_straddling_token_is_truncated() {
        let mut s = triggers_with("<tool_call>");
        assert_eq!(
            s.observe(7, b"sure, let me look").unwrap(),
            TriggerStep::Awaiting
        );
        let TriggerStep::Fired(replay) = s.observe(8, b" up<tool_call>").unwrap() else {
            panic!("trigger is complete");
        };
        assert_eq!(
            replay,
            vec![(8, b"<tool_call>".to_vec())],
            "the prose token must not reach the grammar, and token 8 must lose its \" up\""
        );
    }

    /// A trigger TOKEN is an id match, and it discards the buffer rather
    /// than replaying it -- the opposite of a pattern.
    #[test]
    fn a_trigger_token_matches_an_id_and_discards_the_prose() {
        let mut s = LazyState::new(LazyTriggers::new().with_token(42));
        assert_eq!(s.observe(1, b"thinking...").unwrap(), TriggerStep::Awaiting);
        let step = s.observe(42, b"<tool_call>").unwrap();
        assert_eq!(
            step,
            TriggerStep::Fired(vec![(42, b"<tool_call>".to_vec())]),
            "only the trigger token itself is fed to the grammar"
        );
    }

    /// A token whose PIECE spells a trigger word is not a trigger token:
    /// the id is what is compared.
    #[test]
    fn a_trigger_token_does_not_match_by_piece() {
        let mut s = LazyState::new(LazyTriggers::new().with_token(42));
        assert_eq!(s.observe(9, b"<tool_call>").unwrap(), TriggerStep::Awaiting);
    }

    /// `find_start_pos`: the grammar starts at the first non-empty
    /// capture group, not at the start of the match. This is upstream's
    /// gpt-oss trigger shape.
    #[test]
    fn the_grammar_starts_at_the_first_non_empty_capture_group() {
        let mut s = triggers_with(r"<\|start\|>assistant(\s+to)");
        let TriggerStep::Fired(replay) = s.observe(1, b"<|start|>assistant to").unwrap() else {
            panic!("trigger matches");
        };
        assert_eq!(
            replay,
            vec![(1, b" to".to_vec())],
            "the grammar must be fed from the capture group, not the match"
        );
    }

    /// An empty capture group is skipped, and with no non-empty group at
    /// all the whole match's start is used.
    #[test]
    fn an_empty_capture_group_is_skipped() {
        let p = TriggerPattern::new(r"ab(x?)(c)").unwrap();
        assert_eq!(
            p.find("zzabc").unwrap(),
            Some(4),
            "group 1 matched empty, so group 2 decides"
        );
        let p = TriggerPattern::new(r"ab(?:c)").unwrap();
        assert_eq!(
            p.find("zzabc").unwrap(),
            Some(2),
            "no group: the match start"
        );
    }

    /// A `^...$` pattern must match the WHOLE accumulated output.
    #[test]
    fn an_anchored_pattern_matches_only_the_whole_buffer() {
        let p = TriggerPattern::new(r"^\s+to$").unwrap();
        assert_eq!(p.find("  to").unwrap(), Some(0));
        assert_eq!(
            p.find("  to ").unwrap(),
            None,
            "a trailing space means the buffer is no longer the whole match"
        );
        assert_eq!(p.find("x  to").unwrap(), None);
    }

    /// A word trigger is a literal: its regex metacharacters are escaped.
    #[test]
    fn a_word_trigger_is_matched_literally() {
        let t = LazyTriggers::new().with_word("[TOOL_CALLS]").unwrap();
        let p = &t.patterns()[0];
        assert_eq!(p.find("say [TOOL_CALLS] now").unwrap(), Some(4));
        assert_eq!(
            p.find("say TOOL_CALLS now").unwrap(),
            None,
            "unescaped, the brackets would be a character class"
        );
    }

    /// A full pattern is anchored on both ends, once.
    #[test]
    fn a_full_pattern_is_anchored_at_both_ends() {
        let t = LazyTriggers::new().with_full_pattern("to").unwrap();
        assert_eq!(t.patterns()[0].pattern(), "^to$");
        let t = LazyTriggers::new().with_full_pattern("^to").unwrap();
        assert_eq!(t.patterns()[0].pattern(), "^to$");
        let t = LazyTriggers::new().with_full_pattern("").unwrap();
        assert_eq!(t.patterns()[0].pattern(), "^$");
    }

    /// A piece that ends mid-codepoint contributes nothing until it is
    /// completed, and then the completed character can trigger.
    #[test]
    fn a_partial_codepoint_does_not_break_matching() {
        let mut s = triggers_with("é!");
        // "é" is \xc3\xa9, split across two pieces.
        assert_eq!(s.observe(1, b"\xc3").unwrap(), TriggerStep::Awaiting);
        let TriggerStep::Fired(replay) = s.observe(2, b"\xa9!").unwrap() else {
            panic!("the character is complete now");
        };
        assert_eq!(
            replay,
            vec![(1, b"\xc3".to_vec()), (2, b"\xa9!".to_vec())],
            "the half-character token is replayed too: it is inside the match"
        );
    }

    /// A pattern this engine cannot compile is a refusal naming it, not a
    /// grammar that quietly never fires.
    #[test]
    fn an_uncompilable_pattern_is_refused() {
        let err = LazyTriggers::new().with_pattern("(unclosed").unwrap_err();
        assert!(
            matches!(err, GrammarError::TriggerPatternInvalid { .. }),
            "{err}"
        );
    }

    /// Upstream's functionary trigger uses a negative lookahead, which is
    /// why this compiles patterns with `fancy_regex`.
    #[test]
    fn a_lookahead_pattern_compiles_and_matches() {
        let p = TriggerPattern::new(r">>>(?!all)").unwrap();
        assert_eq!(p.find(">>>get_weather").unwrap(), Some(0));
        assert_eq!(p.find(">>>all").unwrap(), None);
    }
}

/// The half of the port that lives on [`Grammar`]: what a lazy grammar
/// does to acceptance, to end-of-generation, and to the candidate walk.
#[cfg(test)]
mod grammar_tests {
    use super::*;
    use crate::grammar::candidates::{reject_candidates, Candidate};
    use crate::grammar::machine::Grammar;

    fn lazy_grammar(src: &str, triggers: LazyTriggers) -> Grammar {
        Grammar::from_str_with_root(src, "root")
            .expect("grammar parses")
            .into_lazy(triggers)
            .expect("triggers are not empty")
    }

    /// The whole point: prose the grammar forbids passes untouched while
    /// awaiting, and the same grammar without the trigger dies on it.
    #[test]
    fn prose_the_grammar_forbids_is_accepted_while_awaiting() {
        let src = r#"root ::= "<t>" "{}""#;
        let mut lazy = lazy_grammar(src, LazyTriggers::new().with_word("<t>").unwrap());
        lazy.accept_token(1, b"sure!")
            .expect("an untriggered grammar accepts anything");
        assert!(lazy.is_awaiting_trigger());
        assert_eq!(lazy.trigger_buffer(), b"sure!");

        let mut eager = Grammar::from_str_with_root(src, "root").unwrap();
        eager
            .accept_token(1, b"sure!")
            .expect_err("without a trigger the same prose kills the parse");
    }

    /// After the trigger the grammar is live and mid-parse: it has been
    /// fed the trigger text and wants what follows it.
    #[test]
    fn the_replay_leaves_the_grammar_where_the_trigger_text_put_it() {
        let mut g = lazy_grammar(
            r#"root ::= "<t>" "{}""#,
            LazyTriggers::new().with_word("<t>").unwrap(),
        );
        g.accept_token(1, b"hmm <").unwrap();
        g.accept_token(2, b"t>").unwrap();
        assert!(!g.is_awaiting_trigger(), "the trigger word is complete");
        assert!(g.trigger_buffer().is_empty());
        assert!(
            g.accept_token(3, b"{").is_ok(),
            "the grammar should be past \"<t>\" and expecting \"{{\""
        );
        g.accept_token(4, b"}").unwrap();
        assert!(g.allows_eog(), "the parse is complete");
    }

    /// A grammar the replay cannot satisfy is a refusal, not a dead
    /// machine that silently rejects everything afterwards.
    #[test]
    fn a_replay_the_grammar_rejects_is_an_error() {
        let mut g = lazy_grammar(
            r#"root ::= "{}""#,
            LazyTriggers::new().with_word("<t>").unwrap(),
        );
        let err = g
            .accept_token(1, b"<t>")
            .expect_err("this grammar cannot consume its own trigger word");
        assert!(matches!(err, GrammarError::NoViableStack { .. }), "{err}");
    }

    /// An untriggered grammar has no opinion about ending: a generation
    /// that never calls a tool must be able to stop.
    #[test]
    fn an_untriggered_grammar_allows_end_of_generation() {
        let mut g = lazy_grammar(
            r#"root ::= "<t>" "{}""#,
            LazyTriggers::new().with_word("<t>").unwrap(),
        );
        assert!(g.allows_eog());
        assert!(g.accept_eog().is_ok());

        g.accept_token(1, b"<t>").unwrap();
        assert!(
            !g.allows_eog(),
            "once triggered it is an ordinary unsatisfied grammar"
        );
    }

    /// Asking an untriggered grammar what it forbids is refused. Answering
    /// it from the stacks would forbid the prose the trigger exists for.
    #[test]
    fn the_candidate_walk_refuses_an_untriggered_grammar() {
        let g = lazy_grammar(
            r#"root ::= "<t>" "{}""#,
            LazyTriggers::new().with_word("<t>").unwrap(),
        );
        let cands = [Candidate::new(0, 0, b"zzz")];
        let err = reject_candidates(&g, &cands).expect_err("no answer to give yet");
        assert!(matches!(err, GrammarError::AwaitingTrigger), "{err}");
    }

    /// A lazy grammar with no triggers can never fire, so it is refused
    /// rather than accepted as a grammar that constrains nothing.
    #[test]
    fn a_lazy_grammar_with_no_triggers_is_refused() {
        let err = Grammar::from_str_with_root(r#"root ::= "a""#, "root")
            .unwrap()
            .into_lazy(LazyTriggers::new())
            .expect_err("nothing could ever switch this on");
        assert!(matches!(err, GrammarError::LazyWithoutTriggers), "{err}");
    }

    /// A grammar that is not lazy is unchanged by any of this.
    #[test]
    fn an_eager_grammar_is_not_awaiting_anything() {
        let g = Grammar::from_str_with_root(r#"root ::= "a""#, "root").unwrap();
        assert!(!g.is_lazy());
        assert!(!g.is_awaiting_trigger());
        assert!(g.trigger_buffer().is_empty());
    }
}
