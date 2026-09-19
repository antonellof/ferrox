//! The sampler hook a [`Grammar`] hangs on: a live grammar plus the
//! vocabulary it constrains.
//!
//! [`crate::grammar`] deliberately knows nothing about a vocabulary --
//! [`reject_candidates`] answers "which of these pieces is impossible?"
//! and stops there. Two of the rules in llama.cpp's
//! `llama_grammar_apply_impl` are therefore missing from it, because they
//! are questions about the *vocabulary* rather than the grammar, and this
//! is where they live:
//!
//! - an end-of-generation token is masked unless [`Grammar::allows_eog`];
//! - a token whose piece is empty, or starts with a NUL byte, is masked
//!   unconditionally -- it would advance the grammar by nothing and the
//!   decode loop by one token, which is how a constrained generation
//!   spins to `max_tokens` emitting nothing.
//!
//! # Why the two halves are one type
//!
//! Masking before the sample and accepting after it are not two features.
//! A loop that masks and forgets to accept produces text that satisfies
//! the grammar's FIRST token over and over; a loop that accepts and
//! forgets to mask produces unconstrained text and then dies on the first
//! token that does not parse. Both halves are private to
//! [`GrammarSampler`] -- [`GrammarSampler::mask_logits`] and
//! [`GrammarSampler::accept`] -- so a caller that holds one holds the
//! other, and the server keeps that pairing in exactly one function
//! (`frink_server::sample_step::sample_next`).
//!
//! # The vocabulary is snapshotted once
//!
//! Detokenizing every vocabulary entry costs a real amount, and it costs
//! the same on every token step because the vocabulary does not change.
//! [`GrammarSampler::new`] takes the snapshot once per request; the
//! per-step cost is then the shared-prefix walk in
//! [`reject_candidates`] and nothing else.

use crate::grammar::{reject_candidates, Candidate, Grammar, GrammarError};

/// A constrained-sampling failure. Kept separate from [`GrammarError`],
/// which is about a grammar, because every variant here is about the
/// grammar's fit to a *vocabulary* or to a caller's logits.
#[derive(Debug, thiserror::Error)]
pub enum ConstraintError {
    /// The logits handed to the mask are not vocabulary-shaped.
    ///
    /// The live cause is a backend that folded `lm_head + argmax` onto
    /// the device and returned a one-element vector holding a token id
    /// (`frink_server::generate::greedy_gpu_fold_allowed`). Masking that
    /// would zero a token id rather than a logit, so it is refused rather
    /// than performed on the wrong thing.
    #[error(
        "grammar-constrained sampling needs one logit per vocabulary entry, \
         but was handed {got} for a vocabulary of {expected}"
    )]
    VocabMismatch { got: usize, expected: usize },

    /// The grammar forbids every token in the vocabulary, and the parse
    /// is NOT complete.
    ///
    /// Not a bug in the grammar engine: it is a grammar this vocabulary
    /// cannot spell (a rule requiring a character no token piece
    /// contains), and the only alternative to refusing is to sample from
    /// an all-`-inf` distribution, i.e. to emit an arbitrary token and
    /// call it constrained output.
    ///
    /// The complete case is [`MaskOutcome::Complete`] and is not an
    /// error: nothing left to say, having said everything, is an answer.
    #[error(
        "grammar allows no token in this vocabulary after {accepted} accepted token(s), \
         and the parse is incomplete; the grammar requires something this tokenizer \
         cannot spell"
    )]
    NoAllowedToken { accepted: usize },

    /// A token id outside the snapshotted vocabulary was accepted.
    #[error(
        "token id {token} is outside the vocabulary of {vocab_size} the grammar was built over"
    )]
    TokenOutOfVocab { token: usize, vocab_size: usize },

    /// The grammar itself refused.
    #[error(transparent)]
    Grammar(#[from] GrammarError),
}

/// What a mask left behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskOutcome {
    /// At least one token survived; sample normally.
    Allowed,
    /// Nothing survived, and the parse is COMPLETE.
    ///
    /// Reached when a satisfied grammar can be continued by nothing and
    /// the vocabulary has no end-of-generation token to end on -- which
    /// is otherwise the ordinary way a constrained generation stops,
    /// since a satisfied grammar leaves EOG unmasked and it is the only
    /// thing left to sample. The generation is finished, and the token
    /// that comes back out of an all-`-inf` distribution is not a
    /// choice and must be discarded.
    Complete,
}

/// A grammar being applied to one generation, over one vocabulary.
pub struct GrammarSampler {
    grammar: Grammar,
    /// Token pieces as BYTES, by token id. Not `String`: a BPE piece can
    /// hold a fragment of a multi-byte character, and the grammar's
    /// decoder carries that fragment to the next piece rather than
    /// rejecting it.
    pieces: Vec<Vec<u8>>,
    /// End-of-generation flags, by token id.
    eog: Vec<bool>,
    /// Tokens accepted so far; reported when a grammar dead-ends, since
    /// "where" is the first thing anyone debugging one asks.
    accepted: usize,
}

impl GrammarSampler {
    /// Snapshot `n_vocab` token pieces and their end-of-generation flags,
    /// and start applying `grammar` over them.
    pub fn new(
        grammar: Grammar,
        n_vocab: usize,
        piece_of: impl Fn(usize) -> Vec<u8>,
        is_eog: impl Fn(usize) -> bool,
    ) -> Self {
        let mut pieces = Vec::with_capacity(n_vocab);
        let mut eog = Vec::with_capacity(n_vocab);
        for id in 0..n_vocab {
            pieces.push(piece_of(id));
            eog.push(is_eog(id));
        }
        Self {
            grammar,
            pieces,
            eog,
            accepted: 0,
        }
    }

    /// The vocabulary size this was built over.
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// The live grammar, for a caller that wants to ask it something.
    pub fn grammar(&self) -> &Grammar {
        &self.grammar
    }

    /// Set every logit the grammar forbids to `-inf`, in place.
    ///
    /// `llama_grammar_apply_impl`. A logit already at `-inf` is left
    /// alone and never offered to the grammar: it is forbidden whatever
    /// the grammar thinks, and the walk in [`reject_candidates`] is
    /// linear in the candidates it is given.
    pub fn mask_logits(&self, logits: &mut [f32]) -> Result<MaskOutcome, ConstraintError> {
        if logits.len() != self.pieces.len() {
            return Err(ConstraintError::VocabMismatch {
                got: logits.len(),
                expected: self.pieces.len(),
            });
        }

        // A lazy grammar that has not triggered masks NOTHING -- not even
        // the end-of-generation and empty-piece tokens the two vocabulary
        // rules above would otherwise take out. `llama_grammar_apply_impl`
        // returns before all of it. The shape check stays above this: the
        // trigger can fire on any token, so the caller must be handing
        // over real logits from the first one.
        //
        // The one exception is this repo's own
        // [`LazyTriggers::mandatory`](crate::grammar::LazyTriggers::mandatory),
        // which forbids ENDING before the trigger fires and nothing else.
        // Everything the model might say on the way there is still free.
        if self.grammar.is_awaiting_trigger() {
            if self.grammar.allows_eog() {
                return Ok(MaskOutcome::Allowed);
            }
            return self.mask_eog_only(logits);
        }

        let allow_eog = self.grammar.allows_eog();
        let mut candidates: Vec<Candidate<'_>> = Vec::with_capacity(logits.len());
        // Tokens left allowed that the grammar was never asked about:
        // the end-of-generation tokens, when the grammar is satisfied.
        let mut allowed_eog = 0usize;

        for (id, piece) in self.pieces.iter().enumerate() {
            if logits[id] == f32::NEG_INFINITY {
                continue;
            }
            if self.eog[id] {
                if allow_eog {
                    allowed_eog += 1;
                } else {
                    logits[id] = f32::NEG_INFINITY;
                }
            } else if piece.is_empty() || piece[0] == 0 {
                logits[id] = f32::NEG_INFINITY;
            } else {
                candidates.push(Candidate::new(id, id as u32, piece));
            }
        }

        let rejected = reject_candidates(&self.grammar, &candidates)?;
        for index in &rejected {
            logits[*index] = f32::NEG_INFINITY;
        }

        if candidates.len() - rejected.len() + allowed_eog == 0 {
            if allow_eog {
                return Ok(MaskOutcome::Complete);
            }
            return Err(ConstraintError::NoAllowedToken {
                accepted: self.accepted,
            });
        }
        Ok(MaskOutcome::Allowed)
    }

    /// Take out every end-of-generation token and leave the rest alone.
    ///
    /// The whole mask for an untriggered MANDATORY lazy grammar: the turn
    /// may not end, and nothing else is decided yet. A vocabulary with
    /// nothing left but its end-of-generation tokens is refused rather
    /// than sampled, for the same reason the ordinary path refuses one.
    fn mask_eog_only(&self, logits: &mut [f32]) -> Result<MaskOutcome, ConstraintError> {
        let mut survivors = 0usize;
        for (id, logit) in logits.iter_mut().enumerate() {
            if *logit == f32::NEG_INFINITY {
                continue;
            }
            if self.eog[id] {
                *logit = f32::NEG_INFINITY;
            } else {
                survivors += 1;
            }
        }
        if survivors == 0 {
            return Err(ConstraintError::NoAllowedToken {
                accepted: self.accepted,
            });
        }
        Ok(MaskOutcome::Allowed)
    }

    /// Advance the grammar over a sampled token.
    ///
    /// `llama_grammar_accept_impl`. An end-of-generation token does not
    /// consume characters -- it asserts the parse is finished -- so it
    /// goes to [`Grammar::accept_eog`], which refuses if it is not.
    ///
    /// The order of the two tests is upstream's and matters: the trigger
    /// check comes FIRST, so while a lazy grammar is awaiting, even an
    /// end-of-generation token is buffered rather than asserted against a
    /// parse that has not started.
    pub fn accept(&mut self, token: usize) -> Result<(), ConstraintError> {
        let piece = self
            .pieces
            .get(token)
            .ok_or(ConstraintError::TokenOutOfVocab {
                token,
                vocab_size: self.pieces.len(),
            })?;
        if self.grammar.is_awaiting_trigger() {
            self.grammar.accept_token(token as u32, piece)?;
        } else if self.eog[token] {
            self.grammar.accept_eog()?;
        } else {
            self.grammar.accept_token(token as u32, piece)?;
        }
        self.accepted += 1;
        Ok(())
    }

    /// Whether a parse is complete, so generation may end here.
    ///
    /// True throughout an untriggered lazy grammar: it has not been
    /// applied, so it has no say in when generation ends.
    pub fn allows_eog(&self) -> bool {
        self.grammar.allows_eog()
    }

    /// Whether this is a lazy grammar that has not switched on yet.
    ///
    /// A caller deciding whether it needs full vocabulary logits must NOT
    /// read this as "unconstrained": the trigger can fire on any token, so
    /// the grammar needs a real logit vector from the first one. It is for
    /// reporting and for tests.
    pub fn is_awaiting_trigger(&self) -> bool {
        self.grammar.is_awaiting_trigger()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::LazyTriggers;

    /// A toy vocabulary: ids are indices into this table.
    ///
    /// Id 4 is empty and id 5 leads with a NUL, which are the two pieces
    /// the grammar is never allowed to see. Id 6 is end-of-generation.
    const PIECES: &[&[u8]] = &[
        b"a",        // 0
        b"b",        // 1
        b"c",        // 2
        b"ab",       // 3
        b"",         // 4
        b"\0stop",   // 5
        b"</s>",     // 6 (EOG)
        b"\xf0\x9f", // 7: the first two bytes of a 4-byte emoji
    ];
    const EOG_ID: usize = 6;

    fn sampler(src: &str) -> GrammarSampler {
        let grammar = Grammar::from_str_with_root(src, "root").expect("grammar parses");
        GrammarSampler::new(
            grammar,
            PIECES.len(),
            |id| PIECES[id].to_vec(),
            |id| id == EOG_ID,
        )
    }

    /// Every logit starts allowed, so a masked one is the grammar's doing.
    fn flat_logits() -> Vec<f32> {
        vec![0.0; PIECES.len()]
    }

    fn allowed(logits: &[f32]) -> Vec<usize> {
        logits
            .iter()
            .enumerate()
            .filter(|(_, l)| **l != f32::NEG_INFINITY)
            .map(|(i, _)| i)
            .collect()
    }

    /// The core of the hook: only pieces the grammar can consume survive.
    /// `root ::= "ab"` admits "a" and "ab" at the start, and nothing else.
    #[test]
    fn mask_leaves_only_pieces_the_grammar_admits() {
        let s = sampler(r#"root ::= "ab""#);
        let mut logits = flat_logits();
        s.mask_logits(&mut logits).expect("grammar has a move");
        assert_eq!(allowed(&logits), vec![0, 3]);
    }

    /// The first of the two vocabulary-side rules. `root ::= "a"*` admits
    /// the empty string, so a zero-width piece is one the *grammar* would
    /// happily accept -- it is masked because a vocabulary rule says so,
    /// not because the grammar rejected it.
    #[test]
    fn an_empty_or_nul_leading_piece_is_masked_even_when_the_grammar_would_take_it() {
        let s = sampler(r#"root ::= "a"*"#);
        let mut logits = flat_logits();
        s.mask_logits(&mut logits).expect("grammar has a move");
        assert!(
            !allowed(&logits).contains(&4),
            "an empty piece advances the grammar by nothing and the loop by a token"
        );
        assert!(
            !allowed(&logits).contains(&5),
            "a NUL-leading piece is masked unconditionally"
        );
    }

    /// The second. `root ::= "a"` is unsatisfied before "a" is accepted,
    /// so the end-of-generation token must not be sampleable; once the
    /// parse completes it must be.
    #[test]
    fn eog_is_masked_until_the_grammar_is_satisfied() {
        let mut s = sampler(r#"root ::= "a""#);
        let mut logits = flat_logits();
        s.mask_logits(&mut logits).expect("grammar has a move");
        assert!(
            !allowed(&logits).contains(&EOG_ID),
            "generation could end before the grammar was satisfied"
        );

        s.accept(0).expect("\"a\" is what the grammar asked for");
        assert!(s.allows_eog());
        let mut logits = flat_logits();
        s.mask_logits(&mut logits).expect("eog is still a move");
        assert!(allowed(&logits).contains(&EOG_ID));
    }

    /// Accepting moves the machine: after "a", `root ::= "ab"` wants "b".
    #[test]
    fn accepting_a_token_advances_what_is_allowed_next() {
        let mut s = sampler(r#"root ::= "ab""#);
        s.accept(0).unwrap();
        let mut logits = flat_logits();
        s.mask_logits(&mut logits).expect("grammar has a move");
        assert_eq!(allowed(&logits), vec![1]);
    }

    /// A token the mask forbade, accepted anyway, is a refusal rather
    /// than a silently dead grammar that then rejects everything.
    #[test]
    fn accepting_a_token_the_grammar_forbids_is_an_error() {
        let mut s = sampler(r#"root ::= "ab""#);
        let err = s.accept(2).expect_err("\"c\" is not in this grammar");
        assert!(matches!(err, ConstraintError::Grammar(_)), "{err}");
    }

    /// Ending on EOG when the parse is unfinished is refused too: the
    /// mask should have made it unsampleable, so reaching here means the
    /// two halves disagreed.
    #[test]
    fn accepting_eog_before_the_grammar_is_satisfied_is_an_error() {
        let mut s = sampler(r#"root ::= "ab""#);
        let err = s.accept(EOG_ID).expect_err("the parse is not finished");
        assert!(matches!(err, ConstraintError::Grammar(_)), "{err}");
    }

    /// A satisfied grammar with nothing left to say, in a vocabulary
    /// with no end-of-generation token to say it with, is a COMPLETE
    /// answer and not a failure. (With an EOG token -- the ordinary
    /// case -- EOG survives the mask and this is never reached, which
    /// the test above already shows.)
    #[test]
    fn a_satisfied_grammar_with_no_continuation_is_complete_not_an_error() {
        let grammar = Grammar::from_str_with_root(r#"root ::= "a""#, "root").unwrap();
        let mut s = GrammarSampler::new(grammar, PIECES.len(), |id| PIECES[id].to_vec(), |_| false);
        assert_eq!(
            s.mask_logits(&mut flat_logits()).unwrap(),
            MaskOutcome::Allowed
        );
        s.accept(0).unwrap();
        assert_eq!(
            s.mask_logits(&mut flat_logits()).unwrap(),
            MaskOutcome::Complete,
            "a finished parse reported as a failure"
        );
    }

    /// A grammar this vocabulary cannot spell must STOP, not sample from
    /// an all-`-inf` distribution and call the result constrained.
    #[test]
    fn a_grammar_no_token_can_satisfy_is_refused_rather_than_sampled() {
        let s = sampler(r#"root ::= "zzz""#);
        let mut logits = flat_logits();
        let err = s
            .mask_logits(&mut logits)
            .expect_err("no piece in this vocabulary starts with z");
        assert!(
            matches!(err, ConstraintError::NoAllowedToken { .. }),
            "{err}"
        );
    }

    /// The folded-lm_head case: one number that is a token id, not a
    /// vocabulary. Masking it would zero the id.
    #[test]
    fn logits_that_are_not_vocabulary_shaped_are_refused() {
        let s = sampler(r#"root ::= "a""#);
        let mut folded = vec![3.0f32];
        let err = s.mask_logits(&mut folded).expect_err("not a vocabulary");
        assert!(
            matches!(
                err,
                ConstraintError::VocabMismatch {
                    got: 1,
                    expected: 8
                }
            ),
            "{err}"
        );
        assert_eq!(folded[0], 3.0, "the token id was overwritten");
    }

    /// A piece that ends mid-codepoint stays viable: the grammar carries
    /// the partial sequence to the next piece. Rejecting it here is the
    /// bug llama.cpp's `partial_utf8` exists to avoid, and it would make
    /// every emoji unreachable under any grammar with a `.`-like class.
    #[test]
    fn a_piece_ending_mid_codepoint_is_not_rejected() {
        // U+1F600 is \xf0\x9f\x98\x80; piece 7 is its first two bytes.
        let s = sampler(r#"root ::= [\U0001F600-\U0001F64F]"#);
        let mut logits = flat_logits();
        s.mask_logits(&mut logits).expect("grammar has a move");
        assert!(
            allowed(&logits).contains(&7),
            "a partial UTF-8 piece was rejected before its continuation could arrive"
        );
    }

    /// A vocabulary for the lazy tests: the trigger word `<tool_call>` is
    /// two pieces, so no single token spells it.
    const LAZY_PIECES: &[&[u8]] = &[
        b"sure",        // 0: prose
        b", one sec ",  // 1: prose, and the token that straddles the trigger
        b"<tool",       // 2
        b"_call>",      // 3
        b"{",           // 4
        b"}",           // 5
        b"</s>",        // 6 (EOG)
        b"never valid", // 7: forbidden by the grammar at every point
    ];
    const LAZY_EOG: usize = 6;
    /// The grammar begins with the trigger word, because a WORD trigger
    /// feeds the matched text to the grammar.
    const LAZY_GRAMMAR: &str = r#"root ::= "<tool_call>" "{" "}""#;

    fn lazy_sampler(triggers: LazyTriggers) -> GrammarSampler {
        let grammar = Grammar::from_str_with_root(LAZY_GRAMMAR, "root")
            .expect("grammar parses")
            .into_lazy(triggers)
            .expect("triggers are not empty");
        GrammarSampler::new(
            grammar,
            LAZY_PIECES.len(),
            |id| LAZY_PIECES[id].to_vec(),
            |id| id == LAZY_EOG,
        )
    }

    fn lazy_logits() -> Vec<f32> {
        vec![0.0; LAZY_PIECES.len()]
    }

    /// The case lazy grammars exist for, end to end: free prose, then a
    /// trigger spanning two tokens, then constrained output.
    ///
    /// The vacuity check is the first assertion: while awaiting, the mask
    /// leaves token 7 sampleable, and the last assertion shows the
    /// triggered grammar forbids it. Without that pair the test would pass
    /// on a mask that does nothing at all.
    #[test]
    fn free_text_then_a_trigger_then_constrained_output() {
        let mut s = lazy_sampler(LazyTriggers::new().with_word("<tool_call>").unwrap());

        let mut logits = lazy_logits();
        assert_eq!(s.mask_logits(&mut logits).unwrap(), MaskOutcome::Allowed);
        assert_eq!(
            allowed(&logits),
            (0..LAZY_PIECES.len()).collect::<Vec<_>>(),
            "an untriggered lazy grammar must mask nothing at all"
        );

        // Prose the grammar could never accept.
        s.accept(0).expect("prose is free");
        s.accept(1).expect("prose is free");
        assert!(s.is_awaiting_trigger());

        // The trigger word arrives across two tokens; neither piece holds
        // it, so only the accumulated buffer can match.
        s.accept(2)
            .expect("still prose as far as the grammar knows");
        assert!(
            s.is_awaiting_trigger(),
            "\"<tool\" alone is not the trigger word"
        );
        s.accept(3).expect("the trigger completes here");
        assert!(!s.is_awaiting_trigger(), "the trigger did not fire");

        // Now constrained: the replay put the grammar past "<tool_call>".
        let mut logits = lazy_logits();
        s.mask_logits(&mut logits).expect("grammar has a move");
        assert_eq!(
            allowed(&logits),
            vec![4],
            "after the trigger only \"{{\" continues the grammar"
        );

        s.accept(4).unwrap();
        s.accept(5).unwrap();
        let mut logits = lazy_logits();
        s.mask_logits(&mut logits).expect("eog is a move");
        assert_eq!(allowed(&logits), vec![LAZY_EOG], "the tool call is done");
    }

    /// The vacuity check's other half: the same grammar WITHOUT the
    /// trigger forbids the prose from the first token, which is why lazy
    /// is a separate mechanism and not a flag.
    #[test]
    fn the_same_grammar_eagerly_forbids_the_prose_the_lazy_one_allowed() {
        let grammar = Grammar::from_str_with_root(LAZY_GRAMMAR, "root").unwrap();
        let mut s = GrammarSampler::new(
            grammar,
            LAZY_PIECES.len(),
            |id| LAZY_PIECES[id].to_vec(),
            |id| id == LAZY_EOG,
        );
        let mut logits = lazy_logits();
        s.mask_logits(&mut logits).expect("grammar has a move");
        assert_eq!(
            allowed(&logits),
            vec![2],
            "eagerly, only the start of the trigger word is sampleable"
        );
        s.accept(0).expect_err("\"sure\" is not \"<tool_call>\"");
    }

    /// A trigger TOKEN fires on an id and throws the prose away: the
    /// grammar is fed the trigger token alone.
    #[test]
    fn a_trigger_token_seeds_the_grammar_with_itself_only() {
        // Token 2's piece is "<tool", so this grammar starts where that
        // token leaves off.
        let grammar = Grammar::from_str_with_root(r#"root ::= "<tool" "{}""#, "root")
            .unwrap()
            .into_lazy(LazyTriggers::new().with_token(2))
            .unwrap();
        let mut s = GrammarSampler::new(
            grammar,
            LAZY_PIECES.len(),
            |id| LAZY_PIECES[id].to_vec(),
            |id| id == LAZY_EOG,
        );
        s.accept(0).expect("prose is free");
        s.accept(2)
            .expect("the trigger token fires and is replayed");
        assert!(!s.is_awaiting_trigger());
        let mut logits = lazy_logits();
        s.mask_logits(&mut logits).expect("grammar has a move");
        assert_eq!(
            allowed(&logits),
            vec![4],
            "the prose must not have been fed to the grammar"
        );
    }

    /// Generation may end while a lazy grammar waits: a turn with no tool
    /// call is a legal turn.
    #[test]
    fn end_of_generation_is_free_while_awaiting_a_trigger() {
        let mut s = lazy_sampler(LazyTriggers::new().with_word("<tool_call>").unwrap());
        let mut logits = lazy_logits();
        s.mask_logits(&mut logits).unwrap();
        assert!(allowed(&logits).contains(&LAZY_EOG));
        assert!(s.allows_eog());
        s.accept(LAZY_EOG)
            .expect("an untriggered grammar cannot object to stopping");
    }

    /// A MANDATORY trigger forbids exactly one thing before it fires:
    /// ending the turn. Everything the model might say on the way to a
    /// tool call is still free, which is the difference from an eager
    /// grammar and the reason this exists.
    #[test]
    fn a_mandatory_trigger_forbids_ending_the_turn_before_it_fires() {
        let mut s = lazy_sampler(
            LazyTriggers::new()
                .with_word("<tool_call>")
                .unwrap()
                .mandatory(),
        );
        assert!(!s.allows_eog(), "the turn must not be endable yet");

        let mut logits = lazy_logits();
        s.mask_logits(&mut logits).expect("prose is still free");
        assert_eq!(
            allowed(&logits),
            vec![0, 1, 2, 3, 4, 5, 7],
            "only the end-of-generation token may be taken away"
        );

        // Prose, then the trigger.
        s.accept(0).unwrap();
        s.accept(2).unwrap();
        s.accept(3).unwrap();
        assert!(!s.is_awaiting_trigger());
        assert!(
            !s.allows_eog(),
            "the grammar is now live and unsatisfied, so still no ending"
        );

        s.accept(4).unwrap();
        s.accept(5).unwrap();
        let mut logits = lazy_logits();
        s.mask_logits(&mut logits).unwrap();
        assert_eq!(
            allowed(&logits),
            vec![LAZY_EOG],
            "with the call complete the turn may finally end"
        );
    }

    /// The same triggers WITHOUT `mandatory` leave the ending free. The
    /// pair is what shows the flag is what did it.
    #[test]
    fn an_optional_trigger_leaves_the_ending_free() {
        let s = lazy_sampler(LazyTriggers::new().with_word("<tool_call>").unwrap());
        assert!(s.allows_eog());
        let mut logits = lazy_logits();
        s.mask_logits(&mut logits).unwrap();
        assert!(allowed(&logits).contains(&LAZY_EOG));
    }

    /// The mask never *unblocks* anything: a logit already forbidden --
    /// by `logit_bias: -inf`, or by JSON mode -- stays forbidden even
    /// where the grammar is happy with it.
    #[test]
    fn an_already_masked_logit_is_left_masked() {
        let s = sampler(r#"root ::= "ab""#);
        let mut logits = flat_logits();
        logits[0] = f32::NEG_INFINITY;
        s.mask_logits(&mut logits).expect("\"ab\" is still a move");
        assert_eq!(allowed(&logits), vec![3]);
    }
}
