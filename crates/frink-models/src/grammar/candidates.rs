//! Which candidate tokens no viable parse stack accepts.
//!
//! `llama_grammar_reject_candidates` and
//! `llama_grammar_reject_candidates_for_stack`. This is the piece a
//! sampler hook sits on: it answers "of these tokens, which are
//! impossible?" without touching logits, a sampler, or a vocabulary.
//!
//! The algorithm is a shared-prefix walk, not a per-token replay. All
//! candidates are advanced one code point together, the stack set is
//! advanced once, and the survivors recurse. A token whose first character
//! is impossible costs one comparison; a token that fully matches costs
//! one pass per character. Replaying the whole machine per token would be
//! correct and roughly `vocab_size` times slower.
//!
//! Two things upstream does in `llama_grammar_apply_impl` are **not** here,
//! because they need a vocabulary rather than a grammar, and belong to the
//! sampler hook that does not exist yet:
//!
//! - an end-of-generation token is masked unless
//!   [`Grammar::allows_eog`](super::Grammar::allows_eog);
//! - a token whose piece is empty, or starts with a NUL byte, is masked
//!   unconditionally.
//!
//! Everything else about a candidate is decided here.

use super::element::{GrammarRule, GrammarStack, GreType};
use super::error::GrammarError;
use super::machine::{advance_stack, elem, match_char, match_partial_char, match_token, Grammar};
use super::utf8::{decode_piece, PartialUtf8};

/// One token offered to the grammar.
///
/// `index` is the caller's own index -- a position in a logit array, say --
/// and is what [`reject_candidates`] hands back. The grammar never
/// interprets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate<'a> {
    pub index: usize,
    pub id: u32,
    /// The token's decoded piece, as bytes. Not `&str`: a BPE piece can
    /// hold a fragment of a multi-byte character and still be viable.
    pub piece: &'a [u8],
}

impl<'a> Candidate<'a> {
    pub fn new(index: usize, id: u32, piece: &'a [u8]) -> Self {
        Self { index, id, piece }
    }
}

/// A token piece decoded against the grammar's carried partial sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedPiece {
    /// Complete code points, always terminated by a `0`.
    pub code_points: Vec<u32>,
    /// Whatever incomplete sequence the piece ends with.
    pub partial_utf8: PartialUtf8,
}

/// Decode a piece the way the grammar would, continuing from the partial
/// UTF-8 sequence the grammar currently carries.
pub fn decode_for(grammar: &Grammar, piece: &[u8]) -> DecodedPiece {
    let (code_points, partial_utf8) = decode_piece(piece, grammar.partial_utf8());
    DecodedPiece {
        code_points,
        partial_utf8,
    }
}

/// The internal candidate: an index into the decoded arena plus a cursor
/// along it. `llama_grammar_candidate` keeps a raw `const uint32_t *` here
/// and walks it forwards on descent and backwards on return; `slot`/`off`
/// is the same cursor, and `off` can go back down the same way.
#[derive(Debug, Clone, Copy)]
struct Cand {
    index: usize,
    id: u32,
    slot: usize,
    off: usize,
    partial_utf8: PartialUtf8,
}

/// Return the `index` of every candidate that no viable stack accepts.
///
/// The result order follows the algorithm's traversal, not the input
/// order; callers mask by index, so it does not matter. If the grammar has
/// no viable stack left at all, every candidate is rejected.
pub fn reject_candidates(
    grammar: &Grammar,
    candidates: &[Candidate<'_>],
) -> Result<Vec<usize>, GrammarError> {
    if grammar.is_awaiting_trigger() {
        // A lazy grammar that has not fired constrains nothing, and its
        // stacks are still at the start of the parse -- so answering this
        // question from them would forbid the very prose the trigger
        // exists to allow. Upstream's `apply` returns before it ever asks;
        // a caller that gets here has skipped that check.
        return Err(GrammarError::AwaitingTrigger);
    }
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    if grammar.stacks().is_empty() {
        // A dead grammar accepts nothing. Upstream asserts this cannot
        // happen; it can only be reached by accepting a token the mask
        // should have removed, so say so rather than abort.
        return Ok(candidates.iter().map(|c| c.index).collect());
    }

    // Decode every piece once, against the grammar's carried partial.
    let mut arena: Vec<Vec<u32>> = Vec::with_capacity(candidates.len());
    let mut cands: Vec<Cand> = Vec::with_capacity(candidates.len());
    for c in candidates {
        let (code_points, partial_utf8) = decode_piece(c.piece, grammar.partial_utf8());
        arena.push(code_points);
        cands.push(Cand {
            index: c.index,
            id: c.id,
            slot: arena.len() - 1,
            off: 0,
            partial_utf8,
        });
    }

    let rejects = reject_over_stacks(grammar.rules(), grammar.stacks(), &arena, &cands)?;
    Ok(rejects.into_iter().map(|c| c.index).collect())
}

/// Convenience for a single token: does any viable stack accept it?
pub fn accepts_token(grammar: &Grammar, id: u32, piece: &[u8]) -> Result<bool, GrammarError> {
    let c = [Candidate::new(0, id, piece)];
    Ok(reject_candidates(grammar, &c)?.is_empty())
}

/// `llama_grammar_reject_candidates`: a candidate is rejected only if
/// *every* stack rejects it, so the reject set is intersected across
/// stacks by feeding each stack the previous stack's rejects.
fn reject_over_stacks(
    rules: &[GrammarRule],
    stacks: &[GrammarStack],
    arena: &[Vec<u32>],
    candidates: &[Cand],
) -> Result<Vec<Cand>, GrammarError> {
    if stacks.is_empty() {
        return Err(GrammarError::Internal(
            "reject_over_stacks called with no stacks",
        ));
    }
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let mut rejects = reject_for_stack(rules, &stacks[0], arena, candidates)?;
    for stack in &stacks[1..] {
        if rejects.is_empty() {
            break;
        }
        rejects = reject_for_stack(rules, stack, arena, &rejects)?;
    }
    Ok(rejects)
}

/// `llama_grammar_reject_candidates_for_stack`.
fn reject_for_stack(
    rules: &[GrammarRule],
    stack: &GrammarStack,
    arena: &[Vec<u32>],
    candidates: &[Cand],
) -> Result<Vec<Cand>, GrammarError> {
    let mut rejects: Vec<Cand> = Vec::with_capacity(candidates.len());

    let Some(&stack_pos) = stack.last() else {
        // The grammar is satisfied. Only a token that contributes nothing
        // more survives: no complete code points and no partial sequence.
        for tok in candidates {
            if arena[tok.slot][tok.off] != 0 || tok.partial_utf8.n_remain != 0 {
                rejects.push(*tok);
            }
        }
        return Ok(rejects);
    };

    let stack_elem = elem(rules, stack_pos);

    // A stack resting on a token element decides on the token id alone.
    if matches!(stack_elem.gtype, GreType::Token | GreType::TokenNot) {
        for tok in candidates {
            if arena[tok.slot][tok.off] == 0 {
                // The character rules consumed this token's code points;
                // reject only if it ended mid-codepoint.
                if tok.partial_utf8.n_remain != 0 {
                    rejects.push(*tok);
                }
            } else if !match_token(stack_elem, tok.id) {
                rejects.push(*tok);
            }
        }
        return Ok(rejects);
    }

    let mut next_candidates: Vec<Cand> = Vec::with_capacity(candidates.len());

    for tok in candidates {
        if arena[tok.slot][tok.off] == 0 {
            // Out of complete code points. Reject only if the trailing
            // partial sequence could not possibly satisfy this position.
            if tok.partial_utf8.n_remain != 0
                && !match_partial_char(rules, stack_pos, tok.partial_utf8)?
            {
                rejects.push(*tok);
            }
        } else if match_char(rules, stack_pos, arena[tok.slot][tok.off])?.0 {
            let mut advanced = *tok;
            advanced.off += 1;
            next_candidates.push(advanced);
        } else {
            rejects.push(*tok);
        }
    }

    // Advance the stack past this character class once, for everyone.
    let (_, stack_pos_after) = match_char(rules, stack_pos, 0)?;
    let mut stack_after = stack[..stack.len() - 1].to_vec();
    if !elem(rules, stack_pos_after).is_end_of_sequence() {
        stack_after.push(stack_pos_after);
    }
    let mut next_stacks: Vec<GrammarStack> = Vec::new();
    advance_stack(rules, &stack_after, &mut next_stacks)?;

    let next_rejects = reject_over_stacks(rules, &next_stacks, arena, &next_candidates)?;
    for tok in next_rejects {
        let mut back = tok;
        back.off -= 1;
        rejects.push(back);
    }

    Ok(rejects)
}
