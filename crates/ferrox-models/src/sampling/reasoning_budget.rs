//! llama.cpp's reasoning budget, ported from `common/reasoning-budget.cpp`.
//!
//! A token budget for the chain of thought, enforced IN THE SAMPLER
//! rather than in the prompt: the machine watches the accepted tokens
//! for the family's opener, counts the tokens that follow it, and once
//! the budget is spent it forces the closer one token per step by
//! setting every other logit to `-inf`
//! (`common/reasoning-budget.cpp:166-186`). The answer then gets what
//! is left of `max_tokens`, which is the whole point: a caller who asked
//! for a 2,000-token thought gets one, and still gets an answer.
//!
//! The counting rule, from `common_reasoning_budget_accept`
//! (`common/reasoning-budget.cpp:74-164`):
//!
//! - `Idle` until a start sequence ends at the accepted token; then
//!   `remaining = budget` (`:78-91`). A budget of `0` goes straight to
//!   `Forcing` (`:85-89`), which is llama.cpp's "0 for immediate end"
//!   (`common/arg.cpp:3609`).
//! - `Counting`: an end sequence ending at this token is a NATURAL
//!   close and the machine is `Done` (`:96-101`); otherwise the token
//!   costs one (`:118`) -- the closer is never counted, and the token
//!   that spends the last unit is still emitted, because the machine
//!   only forces from the NEXT step on.
//! - When `remaining` hits zero the machine forces, unless the last
//!   piece left a UTF-8 sequence open, in which case it waits for the
//!   token that completes it (`:119-130`, `WaitingUtf8`) so the forced
//!   closer never lands inside a character.
//! - `Forcing`: each accepted token advances through the forced
//!   sequence; the last one makes the machine `Done` (`:134-145`).
//! - `Done` re-arms on a new start sequence, because some models open
//!   several blocks per response (`:146-162`).
//!
//! `-1` is llama.cpp's "unrestricted" and builds NO machine at all
//! (`common/sampling.cpp:316-322`: the sampler is skipped unless
//! `reasoning_budget_tokens >= 0`); `>= 0` builds one with that budget.
//! The prompt's own opener counts: the tokens the template placed after
//! it are fed through [`ReasoningBudget::accept`] before the first draw,
//! which is what `prefill_tokens` does upstream
//! (`common/sampling.cpp:283-297,324-327`).
//!
//! This module is the machine and nothing else. Which token sequences a
//! checkpoint's opener and closer are, and whether the prompt already
//! opened the block, are the caller's to resolve; the machine takes a
//! [`ReasoningBudgetPlan`] that has already answered both.

use std::collections::VecDeque;

/// Everything a [`ReasoningBudget`] needs, resolved once per generation
/// by whoever holds the tokenizer and the rendered prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningBudgetPlan {
    /// Token sequences any of which opens the block
    /// (`start_seqs` upstream; one per family here).
    pub start_seqs: Vec<Vec<usize>>,
    /// Token sequences any of which closes it naturally. The closer
    /// itself, plus whatever else ends a thought for this family -- a
    /// tool-call opener, for one (`common/chat.cpp:1135,2142`).
    pub end_seqs: Vec<Vec<usize>>,
    /// The sequence forced when the budget runs out: the closer,
    /// optionally preceded by a message (`common.h:291`).
    pub forced: Vec<usize>,
    /// Tokens of thought allowed after the opener. `0` forces the closer
    /// the moment the block opens.
    pub budget: u32,
    /// Tokens the PROMPT already put at and after the opener, fed through
    /// `accept` at construction so a template that opens the block itself
    /// starts the count where llama.cpp starts it.
    pub prefill: Vec<usize>,
}

/// Where the machine is; the enum at `common/reasoning-budget.h:10-16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetState {
    /// Waiting for a start sequence.
    Idle,
    /// Inside the block, counting down.
    Counting,
    /// Budget spent, waiting for the token that completes a character.
    WaitingUtf8,
    /// Forcing the closer, one token per step.
    Forcing,
    /// Passthrough until a new start sequence.
    Done,
}

/// llama.cpp's `token_matcher`: reports the longest of `seqs` that ends
/// at the token just fed, and forgets everything on a match
/// (`common/reasoning-budget.cpp:14-50`).
///
/// Upstream is an Aho-Corasick automaton whose state resets to the root
/// after every match. Over a window no longer than the longest
/// sequence, a suffix test is the same function: a sequence ends at
/// this token exactly when it is a suffix of the tokens fed so far,
/// and clearing the window after a match is the reset.
#[derive(Debug, Clone)]
struct SeqMatcher {
    seqs: Vec<Vec<usize>>,
    window: VecDeque<usize>,
    longest: usize,
}

impl SeqMatcher {
    /// Empty sequences and duplicates are dropped, as upstream's
    /// `collect` drops them (`:21-29`): an empty sequence would match
    /// at every token.
    fn new(seqs: &[Vec<usize>]) -> Self {
        let mut kept: Vec<Vec<usize>> = Vec::new();
        for seq in seqs {
            if !seq.is_empty() && !kept.contains(seq) {
                kept.push(seq.clone());
            }
        }
        let longest = kept.iter().map(Vec::len).max().unwrap_or(0);
        Self {
            seqs: kept,
            window: VecDeque::with_capacity(longest),
            longest,
        }
    }

    /// Feed one token; the index of the longest sequence ending here.
    fn advance(&mut self, token: usize) -> Option<usize> {
        if self.longest == 0 {
            return None;
        }
        if self.window.len() == self.longest {
            self.window.pop_front();
        }
        self.window.push_back(token);
        let hit = self
            .seqs
            .iter()
            .enumerate()
            .filter(|(_, seq)| {
                seq.len() <= self.window.len()
                    && self
                        .window
                        .iter()
                        .skip(self.window.len() - seq.len())
                        .eq(seq.iter())
            })
            .max_by_key(|(_, seq)| seq.len())
            .map(|(i, _)| i);
        if hit.is_some() {
            self.reset();
        }
        hit
    }

    fn reset(&mut self) {
        self.window.clear();
    }
}

/// The budget machine for one generation.
#[derive(Debug, Clone)]
pub struct ReasoningBudget {
    start: SeqMatcher,
    end: SeqMatcher,
    forced: Vec<usize>,
    budget: u32,
    remaining: u32,
    state: BudgetState,
    force_pos: usize,
}

impl ReasoningBudget {
    /// A machine at `Idle`, with the plan's prefill already accepted.
    pub fn new(plan: &ReasoningBudgetPlan) -> Self {
        let mut machine = Self {
            start: SeqMatcher::new(&plan.start_seqs),
            end: SeqMatcher::new(&plan.end_seqs),
            forced: plan.forced.clone(),
            budget: plan.budget,
            remaining: plan.budget,
            state: BudgetState::Idle,
            force_pos: 0,
        };
        for &token in &plan.prefill {
            // Prompt text is whole characters, so every prefill piece is
            // complete as far as the wait-for-UTF-8 rule is concerned.
            machine.accept(token, true);
        }
        machine
    }

    pub fn state(&self) -> BudgetState {
        self.state
    }

    /// The token the next draw MUST produce, when the machine is forcing.
    /// `None` is passthrough: the logits are not touched
    /// (`common/reasoning-budget.cpp:169-176`).
    pub fn forced_token(&self) -> Option<usize> {
        match self.state {
            BudgetState::Forcing => self.forced.get(self.force_pos).copied(),
            _ => None,
        }
    }

    /// `common_reasoning_budget_apply` (`:166-186`): every logit but the
    /// forced token's to `-inf`. A no-op outside `Forcing`.
    pub fn mask(&self, logits: &mut [f32]) {
        let Some(forced) = self.forced_token() else {
            return;
        };
        for (id, logit) in logits.iter_mut().enumerate() {
            if id != forced {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Enter the block, `common_reasoning_budget_accept`'s two
    /// activation arms (`:80-90` and `:149-161`).
    fn arm(&mut self) {
        self.state = BudgetState::Counting;
        self.remaining = self.budget;
        self.end.reset();
        if self.remaining == 0 {
            self.state = BudgetState::Forcing;
            self.force_pos = 0;
        }
    }

    /// Start forcing the closer from its first token (`:121-123`).
    fn force(&mut self) {
        self.state = BudgetState::Forcing;
        self.force_pos = 0;
        self.end.reset();
    }

    /// `common_reasoning_budget_accept` (`:74-164`): the token that was
    /// just sampled, and whether its piece left a UTF-8 sequence open.
    pub fn accept(&mut self, token: usize, piece_complete: bool) {
        match self.state {
            BudgetState::Idle | BudgetState::Done => {
                if self.start.advance(token).is_some() {
                    self.arm();
                }
            }
            BudgetState::Counting | BudgetState::WaitingUtf8 => {
                if self.end.advance(token).is_some() {
                    self.state = BudgetState::Done;
                    return;
                }
                if self.state == BudgetState::WaitingUtf8 {
                    if piece_complete {
                        self.force();
                    }
                    return;
                }
                self.remaining = self.remaining.saturating_sub(1);
                if self.remaining == 0 {
                    if piece_complete {
                        self.force();
                    } else {
                        self.state = BudgetState::WaitingUtf8;
                        self.end.reset();
                    }
                }
            }
            BudgetState::Forcing => {
                // Upstream (`:134-145`) advances on ANY token here,
                // trusting `apply` to have left only the forced one. A
                // generated token is always that one; a PREFILL token
                // need not be -- a template that opens `<think>\n`
                // feeds its newline through this arm at budget 0, and
                // upstream then counts the newline as the closer and
                // never forces it. Checked here instead, so the closer
                // is forced whatever the prompt put after the opener.
                self.end.advance(token);
                if self.forced.get(self.force_pos) == Some(&token) {
                    self.force_pos += 1;
                } else {
                    self.force_pos = 0;
                }
                if self.force_pos >= self.forced.len() {
                    self.state = BudgetState::Done;
                }
            }
        }
    }
}

/// Whether a per-token piece rendered LOSSILY (`String::from_utf8_lossy`
/// of the token's bytes) is a complete UTF-8 sequence.
///
/// `common_utf8_is_complete` (`common/unicode.cpp:72-84`) reads the raw
/// bytes: incomplete when the last lead byte declares more continuation
/// bytes than follow it, or when every trailing byte is a continuation
/// byte. Both decode caller-side to a trailing U+FFFD and nothing else
/// does, so on a lossy piece the two tests are the same test. The
/// decode loops hand the sampler text, not bytes, so this is the form
/// the sampler can ask.
pub fn lossy_piece_is_complete(piece: &str) -> bool {
    !piece.ends_with('\u{FFFD}')
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPEN: usize = 100;
    const CLOSE: usize = 101;
    const TOOL_A: usize = 102;
    const TOOL_B: usize = 103;

    fn plan(budget: u32, prefill: &[usize]) -> ReasoningBudgetPlan {
        ReasoningBudgetPlan {
            start_seqs: vec![vec![OPEN]],
            end_seqs: vec![vec![CLOSE], vec![TOOL_A, TOOL_B]],
            forced: vec![CLOSE],
            budget,
            prefill: prefill.to_vec(),
        }
    }

    fn logits(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    /// Run the machine over `tokens` as generated, whole-character
    /// pieces, returning the state after each.
    fn trace(m: &mut ReasoningBudget, tokens: &[usize]) -> Vec<BudgetState> {
        tokens
            .iter()
            .map(|&t| {
                m.accept(t, true);
                m.state()
            })
            .collect()
    }

    /// The counting rule itself: the opener is free, N tokens follow,
    /// and the (N+1)th draw is the forced closer. `reasoning-budget.cpp:
    /// 93-133`: the token that spends the last unit is still accepted
    /// as the model's own; forcing starts on the step after it.
    #[test]
    fn n_tokens_of_thought_pass_and_the_next_draw_is_the_closer() {
        let mut m = ReasoningBudget::new(&plan(3, &[]));
        assert_eq!(m.state(), BudgetState::Idle);
        assert_eq!(m.forced_token(), None);
        assert_eq!(
            trace(&mut m, &[OPEN, 1, 2]),
            [
                BudgetState::Counting,
                BudgetState::Counting,
                BudgetState::Counting
            ]
        );
        assert_eq!(m.forced_token(), None, "two of three spent: passthrough");
        m.accept(3, true);
        assert_eq!(m.state(), BudgetState::Forcing);
        assert_eq!(m.forced_token(), Some(CLOSE));
        let mut l = logits(200);
        m.mask(&mut l);
        assert!(l[CLOSE].is_finite());
        assert!(
            l.iter()
                .enumerate()
                .all(|(i, v)| i == CLOSE || *v == f32::NEG_INFINITY),
            "every logit but the closer's is -inf while forcing"
        );
        m.accept(CLOSE, true);
        assert_eq!(m.state(), BudgetState::Done);
        assert_eq!(m.forced_token(), None, "the answer is unconstrained");
        let mut l = logits(200);
        m.mask(&mut l);
        assert_eq!(l, logits(200), "mask is a no-op once done");
    }

    /// `common/arg.cpp:3609`: "0 for immediate end". The opener is the
    /// last free token; the very next draw is the closer
    /// (`reasoning-budget.cpp:85-89`).
    #[test]
    fn budget_zero_forces_the_closer_right_after_the_opener() {
        let mut m = ReasoningBudget::new(&plan(0, &[]));
        m.accept(OPEN, true);
        assert_eq!(m.state(), BudgetState::Forcing);
        assert_eq!(m.forced_token(), Some(CLOSE));
    }

    /// A model that closes the block itself before the budget is spent
    /// is left alone: no forcing, and the closer was not charged
    /// (`reasoning-budget.cpp:96-101`).
    #[test]
    fn a_natural_close_ends_counting_without_forcing() {
        let mut m = ReasoningBudget::new(&plan(10, &[]));
        trace(&mut m, &[OPEN, 1, 2, CLOSE]);
        assert_eq!(m.state(), BudgetState::Done);
        assert_eq!(m.forced_token(), None);
    }

    /// A multi-token end sequence (a tool-call opener) closes the
    /// thought too, and only when the whole sequence has arrived.
    #[test]
    fn a_multi_token_end_sequence_closes_only_when_complete() {
        let mut m = ReasoningBudget::new(&plan(10, &[]));
        trace(&mut m, &[OPEN, TOOL_A]);
        assert_eq!(m.state(), BudgetState::Counting);
        m.accept(TOOL_B, true);
        assert_eq!(m.state(), BudgetState::Done);
    }

    /// The prompt's own opener starts the count, and what the template
    /// put after it is charged: `<think>\n` at budget 2 leaves ONE
    /// token for the model, exactly as upstream's prefill feed does
    /// (`common/sampling.cpp:324-327`).
    #[test]
    fn a_prompt_that_opened_the_block_starts_counting_before_the_first_draw() {
        const NEWLINE: usize = 7;
        let mut m = ReasoningBudget::new(&plan(2, &[OPEN, NEWLINE]));
        assert_eq!(m.state(), BudgetState::Counting);
        m.accept(1, true);
        assert_eq!(m.state(), BudgetState::Forcing);
    }

    /// The one place this port reads the accepted token where upstream
    /// does not (`reasoning-budget.cpp:134-145`): a prefill token
    /// arriving while forcing is not the closer, so it must not count
    /// as the closer. Upstream would go `Done` on the template's
    /// newline and never force anything at budget 0.
    #[test]
    fn a_prefill_token_after_the_opener_at_budget_zero_does_not_stand_in_for_the_closer() {
        const NEWLINE: usize = 7;
        let m = ReasoningBudget::new(&plan(0, &[OPEN, NEWLINE]));
        assert_eq!(m.state(), BudgetState::Forcing);
        assert_eq!(m.forced_token(), Some(CLOSE));
    }

    /// A forced sequence longer than one token (a budget message before
    /// the closer) is walked one draw at a time.
    #[test]
    fn a_multi_token_forced_sequence_is_forced_in_order() {
        let mut p = plan(0, &[]);
        p.forced = vec![50, 51, CLOSE];
        let mut m = ReasoningBudget::new(&p);
        m.accept(OPEN, true);
        for &expect in &[50, 51, CLOSE] {
            assert_eq!(m.forced_token(), Some(expect));
            m.accept(expect, true);
        }
        assert_eq!(m.state(), BudgetState::Done);
    }

    /// The budget spent on a token that left a character half-written
    /// waits for the token that finishes it, then forces
    /// (`reasoning-budget.cpp:104-131`).
    #[test]
    fn a_split_character_delays_forcing_until_it_is_whole() {
        let mut m = ReasoningBudget::new(&plan(1, &[]));
        m.accept(OPEN, true);
        m.accept(1, false);
        assert_eq!(m.state(), BudgetState::WaitingUtf8);
        assert_eq!(
            m.forced_token(),
            None,
            "the closer must not split a character"
        );
        m.accept(2, false);
        assert_eq!(m.state(), BudgetState::WaitingUtf8);
        m.accept(3, true);
        assert_eq!(m.state(), BudgetState::Forcing);
    }

    /// A second block in the same response gets a fresh budget
    /// (`reasoning-budget.cpp:146-162`).
    #[test]
    fn done_re_arms_on_a_new_opener() {
        let mut m = ReasoningBudget::new(&plan(1, &[]));
        trace(&mut m, &[OPEN, CLOSE, 5, 6, OPEN]);
        assert_eq!(m.state(), BudgetState::Counting);
        m.accept(9, true);
        assert_eq!(m.state(), BudgetState::Forcing);
    }

    /// A multi-token opener arms only when it has fully arrived, and a
    /// token that breaks the sequence does not count as it.
    #[test]
    fn a_multi_token_opener_must_arrive_whole() {
        let mut p = plan(5, &[]);
        p.start_seqs = vec![vec![OPEN, 1, 2]];
        let mut m = ReasoningBudget::new(&p);
        trace(&mut m, &[OPEN, 1, 9, 1, 2]);
        assert_eq!(m.state(), BudgetState::Idle, "OPEN 1 9 broke the sequence");
        trace(&mut m, &[OPEN, 1, 2]);
        assert_eq!(m.state(), BudgetState::Counting);
    }

    /// An empty sequence would match every token and arm on the first
    /// draw; upstream's `collect` drops them (`:21-29`) and so does this.
    #[test]
    fn empty_sequences_never_match() {
        let mut p = plan(5, &[]);
        p.start_seqs = vec![vec![]];
        let mut m = ReasoningBudget::new(&p);
        trace(&mut m, &[1, 2, 3]);
        assert_eq!(m.state(), BudgetState::Idle);
    }

    /// The lossy-piece test against real lossy decodes: a multibyte
    /// character split across tokens renders as U+FFFD until the last
    /// byte arrives, and a whole one never does. Matches
    /// `common_utf8_is_complete` on the bytes those pieces came from.
    #[test]
    fn a_split_multibyte_piece_reads_incomplete_until_its_last_byte() {
        let euro = "€".as_bytes(); // E2 82 AC
        let first = String::from_utf8_lossy(&euro[..1]);
        let middle = String::from_utf8_lossy(&euro[1..2]);
        let whole = String::from_utf8_lossy(euro);
        assert!(!lossy_piece_is_complete(&first));
        assert!(!lossy_piece_is_complete(&middle));
        assert!(lossy_piece_is_complete(&whole));
        assert!(lossy_piece_is_complete("plain ascii"));
        assert!(lossy_piece_is_complete(""));
    }
}
