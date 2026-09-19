//! The one place that decides which tokens the repetition, presence
//! and frequency penalties look back over.
//!
//! # What llama.cpp does, and where
//!
//! llama.cpp's penalties sampler is stateful: it keeps a ring buffer of
//! the last `penalty_last_n` tokens it has ACCEPTED, plus a count map
//! over that buffer, and `apply` walks the candidate list looking each
//! candidate up in the map (`src/llama-sampler.cpp:2698-2759`). Nothing
//! about that buffer knows whether a token was generated or read out of
//! the prompt -- only that the sampler was told about it.
//!
//! Both front ends tell it about the prompt.
//!
//! - `llama-server` seeds the sampler with every prompt token before
//!   the first token is drawn
//!   (`tools/server/server-context.cpp:375-397`, the loop at 386-390:
//!   `for (int i = 0; i < prompt.tokens.size(); i++) { ...
//!   common_sampler_accept(smpl.get(), id, false); }`).
//! - `llama-cli` does the same as it consumes the prompt, with the
//!   reason written on the line above
//!   (`tools/completion/completion.cpp:730-736`: *"push the prompt in
//!   the sampling context in order to apply repetition penalties
//!   later"*, `common_sampler_accept(smpl, embd_inp[n_consumed],
//!   /* accept_grammar= */ false)`).
//!
//! `common_sampler_accept` pushes into the chain unconditionally
//! (`common/sampling.cpp:472-504`), so a prompt token lands in the
//! penalties ring buffer exactly like a generated one.
//!
//! So llama.cpp's window is the last `penalty_last_n` tokens of
//! `prompt ++ generated`, and frink matches that. **This changes
//! output** relative to frink before this module existed, on every run
//! at the default `--repeat-penalty 1.1`: a token that occurs in the
//! prompt is now penalised on its first generated occurrence.
//!
//! # Why it is a type and not a slice
//!
//! Because it was a slice, and five call sites each chose their own.
//! `frink run`'s decode loops passed the generated tokens; the server's
//! two decode loops passed the generated tokens (still do -- issue #73,
//! the prompt ids do not reach that seam); `speculative` passed
//! the prompt as well and then grew a `penalty_history_start` knob to
//! paper over the disagreement; `draft_model` cloned the whole history
//! per block; `kimi_generate` passed prompt and generated and was the
//! only one that matched llama.cpp. Five sites, four answers, nothing
//! enforcing agreement -- this repo's dominant bug shape.
//!
//! A [`PenaltyWindow`] is built from BOTH halves and there is no
//! constructor that takes one slice, so a caller cannot produce a window
//! without saying what its prompt is. A caller that genuinely has none
//! writes `&[]` and that is visible in the diff.

/// The tokens the penalties may see: a prompt and the tokens generated
/// after it, in that order.
///
/// Borrowed rather than owned because this is built once per sampled
/// token on every decode loop in the workspace; an owning window would
/// clone the whole sequence per token.
///
/// The two halves are kept separate rather than concatenated because a
/// decode loop already holds them separately, and concatenating would
/// mean an allocation per token for a value only ever read back as "the
/// last N of the two".
#[derive(Debug, Clone, Copy)]
pub struct PenaltyWindow<'a> {
    prompt: &'a [usize],
    generated: &'a [usize],
}

impl<'a> PenaltyWindow<'a> {
    /// The window over `prompt` followed by `generated`.
    ///
    /// `prompt` is the tokens the model was fed before generation
    /// started, and it belongs in the window: see the module docs for
    /// the llama.cpp lines that put it there.
    pub fn new(prompt: &'a [usize], generated: &'a [usize]) -> Self {
        PenaltyWindow { prompt, generated }
    }

    /// Total tokens in the sequence, before `penalty_last_n` truncates
    /// it.
    pub fn len(&self) -> usize {
        self.prompt.len() + self.generated.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The most recent `last_n` tokens of `prompt ++ generated`, oldest
    /// first.
    ///
    /// This is llama.cpp's ring buffer expressed as a view: the buffer
    /// holds at most `penalty_last_n` entries and the oldest is dropped
    /// on every accept (`src/llama-sampler.cpp:2707-2716`), so its
    /// contents are exactly the tail of the accepted sequence.
    ///
    /// Order does not matter to any caller -- only the multiset does --
    /// but it is the natural one anyway, and a test reads it.
    pub fn recent(&self, last_n: usize) -> impl Iterator<Item = usize> + '_ {
        let start = self.len().saturating_sub(last_n);
        // Split the single cut point across the two halves. `start` is
        // at most `len()`, so both indices are in range and neither
        // subtraction can wrap.
        let from_prompt = start.min(self.prompt.len());
        let from_generated = start.saturating_sub(self.prompt.len());
        self.prompt[from_prompt..]
            .iter()
            .chain(self.generated[from_generated..].iter())
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window is the tail of `prompt ++ generated`, so it slides
    /// across the seam between them rather than restarting at it.
    ///
    /// A window implemented as "the last N of `generated`, plus all of
    /// `prompt`" would keep token 0 here, and a window implemented as
    /// "the last N of `generated`" would keep neither prompt token.
    /// llama.cpp's ring buffer keeps exactly the last N accepted tokens
    /// whichever half they came from.
    #[test]
    fn the_window_is_the_tail_of_the_prompt_and_the_generation_together() {
        let window = PenaltyWindow::new(&[0, 1, 2], &[3, 4]);
        assert_eq!(window.len(), 5);
        assert_eq!(window.recent(3).collect::<Vec<_>>(), vec![2, 3, 4]);
        // The cut can land inside the prompt, inside the generation, or
        // exactly on the seam.
        assert_eq!(window.recent(4).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(window.recent(2).collect::<Vec<_>>(), vec![3, 4]);
        // Wider than the sequence is the whole sequence, not a panic.
        assert_eq!(window.recent(1000).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
        assert_eq!(window.recent(0).count(), 0);
    }

    /// Nothing generated yet is still a non-empty window, which is the
    /// whole point: the first sampled token is already penalised
    /// against the prompt.
    #[test]
    fn a_prompt_alone_is_a_window() {
        let window = PenaltyWindow::new(&[7, 7, 8], &[]);
        assert!(!window.is_empty());
        assert_eq!(window.recent(64).collect::<Vec<_>>(), vec![7, 7, 8]);
        assert_eq!(window.recent(2).collect::<Vec<_>>(), vec![7, 8]);
    }

    /// And an empty prompt is not a special case.
    #[test]
    fn an_empty_prompt_leaves_the_generated_tail() {
        let window = PenaltyWindow::new(&[], &[1, 2, 3]);
        assert_eq!(window.recent(2).collect::<Vec<_>>(), vec![2, 3]);
        assert!(PenaltyWindow::new(&[], &[]).is_empty());
    }
}
