//! The three history penalties -- repetition, presence, frequency --
//! applied to the whole vocabulary before the candidate list exists.
//!
//! Split out of `sampling.rs`: this is `llama_sampler_penalties`, one
//! step of the chain, and it is the only step that runs on the GREEDY
//! path as well as the sampled one, which is why it lives beside the
//! window rather than beside the candidate-list filters.

use super::SamplingParams;
use crate::penalty_window::PenaltyWindow;

/// Penalise tokens that already appear in `history`, once each.
///
/// `history` is a [`PenaltyWindow`], so "already appear" includes the
/// PROMPT. That is llama.cpp's rule and the module docs of
/// [`crate::penalty_window`] carry the upstream lines; before it, every
/// caller in this workspace picked its own slice and four of the five
/// picked differently.
///
/// ONCE EACH is the whole subtlety, and ferrox used to get it wrong.
/// llama.cpp walks the CANDIDATE list and looks each candidate up in a
/// count map (`llama-sampler.cpp:2735-2756`), so a token repeated `n`
/// times is divided by `penalty_repeat` exactly once. ferrox walked the
/// HISTORY, so the same token was divided `n` times and the effective
/// penalty was `penalty^n`.
///
/// That was live on every `ferrox run`: `--repeat-penalty` defaults to
/// 1.1, so a token seen five times was penalised 1.61x rather than
/// 1.1x, and the divergence grew with the length of the output.
///
/// The sign convention is llama.cpp's and its comment explains it:
/// dividing alone would make tokens with NEGATIVE logits more likely,
/// so negatives are multiplied instead.
///
/// A chain that does not name `penalties` does not penalise. llama.cpp
/// reads an omitted sampler as "do not run it", and this function is the
/// one place the penalties happen -- on the greedy path as well as the
/// sampled one -- so the check belongs here rather than beside the
/// candidate list, where the greedy path would never see it.
pub(crate) fn apply_history_penalties(
    scores: &mut [f32],
    params: &SamplingParams,
    history: PenaltyWindow<'_>,
) {
    if !params.sampler_order.has_penalties() {
        return;
    }
    if params.repetition_penalty == 1.0
        && params.presence_penalty == 0.0
        && params.frequency_penalty == 0.0
    {
        return;
    }
    // Only the last `penalty_last_n`, as llama.cpp's ring buffer does.
    if params.penalty_last_n == 0 {
        return;
    }
    let mut counts = std::collections::HashMap::<usize, usize>::new();
    for tok in history.recent(params.penalty_last_n) {
        *counts.entry(tok).or_insert(0) += 1;
    }
    for (tok, count) in counts {
        let Some(s) = scores.get_mut(tok) else {
            continue;
        };
        if params.repetition_penalty != 1.0 {
            *s = if *s > 0.0 {
                *s / params.repetition_penalty
            } else {
                *s * params.repetition_penalty
            };
        }
        *s -= params.frequency_penalty * count as f32;
        *s -= params.presence_penalty;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The repetition penalty is applied ONCE per token, however many
    /// times that token appears in the history.
    ///
    /// ferrox walked the history and divided once per OCCURRENCE, so the
    /// effective penalty was `penalty^n`. llama.cpp walks the candidates
    /// and looks each up in a count map, so it is `penalty` flat
    /// (`llama-sampler.cpp:2735-2756`).
    ///
    /// Live on every `ferrox run`: `--repeat-penalty` defaults to 1.1,
    /// so a token seen five times was penalised 1.61x, and the
    /// divergence grew with the length of the output. Twenty-four
    /// sampling tests passed with the bug in place, which is why this
    /// one exists.
    #[test]
    fn the_repetition_penalty_does_not_compound_with_repeats() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 2.0,
            ..SamplingParams::default()
        };
        let logits = vec![4.0f32, 1.0, 1.0];

        // Token 0 appears five times. Penalised once, its score is 2.0;
        // compounded it would be 4 / 2^5 = 0.125.
        let mut scores = logits.clone();
        apply_history_penalties(
            &mut scores,
            &params,
            PenaltyWindow::new(&[], &[0, 0, 0, 0, 0]),
        );
        assert!(
            (scores[0] - 2.0).abs() < 1e-6,
            "expected one division (2.0), got {} -- {} would be 2^5",
            scores[0],
            4.0f32 / 32.0
        );

        // And once really is once: one occurrence and five occurrences
        // must land on the same score, or the count still leaks in.
        let mut once = logits.clone();
        apply_history_penalties(&mut once, &params, PenaltyWindow::new(&[], &[0]));
        assert_eq!(once[0].to_bits(), scores[0].to_bits());

        // A NEGATIVE logit is multiplied rather than divided, or the
        // penalty would make it more likely -- llama.cpp's own comment.
        let mut negative = vec![-4.0f32];
        apply_history_penalties(&mut negative, &params, PenaltyWindow::new(&[], &[0, 0, 0]));
        assert!((negative[0] + 8.0).abs() < 1e-6, "got {}", negative[0]);
    }

    /// The penalties look at the last `penalty_last_n` tokens, not the
    /// whole history.
    ///
    /// llama.cpp keeps a ring buffer of `penalty_last_n` (default 64,
    /// `common/common.h:238`); ferrox scanned everything generated so
    /// far. On a long generation that is a steadily growing set of
    /// penalised tokens against llama.cpp's fixed 64 -- the divergence
    /// grows with output length, which is when a repetition penalty
    /// matters most.
    #[test]
    fn the_penalties_only_see_the_last_n_tokens() {
        let params = SamplingParams {
            repetition_penalty: 2.0,
            penalty_last_n: 2,
            ..SamplingParams::default()
        };
        let mut scores = vec![8.0f32, 8.0, 8.0];
        // Token 0 fell out of the window; tokens 1 and 2 are in it.
        apply_history_penalties(&mut scores, &params, PenaltyWindow::new(&[], &[0, 1, 2]));
        assert_eq!(
            scores[0].to_bits(),
            8.0f32.to_bits(),
            "token 0 is outside the window"
        );
        assert!((scores[1] - 4.0).abs() < 1e-6, "got {}", scores[1]);
        assert!((scores[2] - 4.0).abs() < 1e-6, "got {}", scores[2]);

        // `0` disables the penalties outright, as llama.cpp documents.
        let off = SamplingParams {
            penalty_last_n: 0,
            ..params.clone()
        };
        let mut untouched = vec![8.0f32; 3];
        apply_history_penalties(&mut untouched, &off, PenaltyWindow::new(&[], &[0, 1, 2]));
        assert_eq!(untouched, vec![8.0f32; 3]);

        // A window longer than the history is not an overflow.
        let wide = SamplingParams {
            penalty_last_n: 1000,
            ..params
        };
        let mut short = vec![8.0f32];
        apply_history_penalties(&mut short, &wide, PenaltyWindow::new(&[], &[0]));
        assert!((short[0] - 4.0).abs() < 1e-6);
    }

    /// Frequency penalty still scales with the count, while the
    /// repetition penalty does not.
    ///
    /// Both live in the same loop, so a fix that made the repetition
    /// penalty flat by dropping the counts would break this one.
    #[test]
    fn the_frequency_penalty_still_counts_repeats() {
        let params = SamplingParams {
            frequency_penalty: 0.5,
            presence_penalty: 0.25,
            ..SamplingParams::default()
        };
        let mut scores = vec![10.0f32];
        apply_history_penalties(&mut scores, &params, PenaltyWindow::new(&[], &[0, 0, 0, 0]));
        // 10 - 0.5*4 - 0.25 = 7.75
        assert!((scores[0] - 7.75).abs() < 1e-6, "got {}", scores[0]);
    }
}
