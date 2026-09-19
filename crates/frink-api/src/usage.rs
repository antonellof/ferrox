//! OpenAI-convention token accounting plus llama.cpp-style timings.
//!
//! Counted from the exact token ids the generation loop processed
//! (prompt after BOS insertion, and every generated id), not
//! re-tokenized after the fact -- re-tokenizing decoded text is not
//! guaranteed to round-trip to the same count.
//!
//! Why the server reports timings at all, when a client can hold a
//! stopwatch: the client's stopwatch measures the network, the proxy's
//! buffer and its own event loop. More importantly it cannot separate
//! **prefill from decode**, and a UI that divides total tokens by total
//! wall time reports a 50 tok/s model as 5 tok/s whenever the prompt is
//! long. Every downstream number built on that is then wrong in the
//! same direction. So the phases are reported separately and the client
//! is never asked to infer one from the other.
//!
//! Every timing is optional: a cached response, a batched decode, or an
//! engine path that does not time itself must be able to answer
//! honestly rather than emit a plausible zero.

use serde::{Deserialize, Serialize};

/// OpenAI's `usage.completion_tokens_details`.
///
/// Only the one field is carried: the others in OpenAI's object
/// (`audio_tokens`, `accepted_prediction_tokens`) describe features
/// this server does not implement, and inventing zeroes for them would
/// be the same lie this type exists to stop telling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// OpenAI's nested completion breakdown. Absent unless the
    /// reasoning split actually ran, because a zero here is a claim
    /// about the model and not about this server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// Prefill throughput (prompt tokens / prefill seconds), when timed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_per_second: Option<f64>,
    /// Decode throughput (completion tokens / decode seconds), when timed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_per_second: Option<f64>,
    /// Wall time spent processing the prompt, in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration_ms: Option<f64>,
    /// Wall time spent in the decode loop, in milliseconds. Kept
    /// separate from `prompt_eval_duration_ms` on purpose (see the
    /// module docs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_duration_ms: Option<f64>,
    /// Time to first token: from the start of prefill to the moment the
    /// first token was produced. `None` when no token was produced at
    /// all (an immediate EOS), because a zero there would read as an
    /// instantaneous response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_ms: Option<f64>,
    /// Prompt tokens served from the KV prefix cache instead of being
    /// recomputed. `Some(0)` means "the cache was consulted and missed";
    /// `None` means "no prefix cache is configured" -- a distinction the
    /// UI needs to decide whether to show the row at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<usize>,
    /// Completion tokens per verification step when speculative
    /// decoding ran: the published *acceptance length*. `None` means
    /// speculation did not run, which is not the same as an acceptance
    /// length of 1.0 (speculation ran and never helped).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acceptance_length: Option<f64>,
    /// Draft tokens the target actually evaluated. Positions after a
    /// rejection are not counted, so the ratio below tracks the
    /// drafter's accuracy rather than its block size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_tokens: Option<usize>,
    /// Draft tokens accepted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_draft_tokens: Option<usize>,
    /// Accept rate at each position within the draft block, each
    /// conditional on that position having been reached.
    ///
    /// Reported alongside the mean and not folded into it: a drafter
    /// that is right at position 0 and useless by position 7 has the
    /// same mean as one that is uniformly mediocre, and the two want
    /// opposite block sizes. Suffix decay is only visible per position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_accept_rate_per_position: Option<Vec<f64>>,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens_details: None,
            prompt_per_second: None,
            predicted_per_second: None,
            prompt_eval_duration_ms: None,
            generation_duration_ms: None,
            time_to_first_token_ms: None,
            cached_tokens: None,
            acceptance_length: None,
            draft_tokens: None,
            accepted_draft_tokens: None,
            draft_accept_rate_per_position: None,
        }
    }

    /// Records the two phase durations, in seconds, and the rates they
    /// imply. A zero-length phase leaves the rate unset rather than
    /// dividing by zero into infinity.
    pub fn with_timings(mut self, prompt_secs: f64, predicted_secs: f64) -> Self {
        self.prompt_eval_duration_ms = Some(prompt_secs * 1000.0);
        self.generation_duration_ms = Some(predicted_secs * 1000.0);
        if prompt_secs > 0.0 && self.prompt_tokens > 0 {
            self.prompt_per_second = Some(self.prompt_tokens as f64 / prompt_secs);
        }
        if predicted_secs > 0.0 && self.completion_tokens > 0 {
            self.predicted_per_second = Some(self.completion_tokens as f64 / predicted_secs);
        }
        self
    }

    /// Time-to-first-token, in seconds, measured from the start of
    /// prefill. Ignored when no token was generated.
    pub fn with_ttft(mut self, secs: f64) -> Self {
        if self.completion_tokens > 0 {
            self.time_to_first_token_ms = Some(secs * 1000.0);
        }
        self
    }

    /// How many of the completion's tokens were spent reasoning.
    ///
    /// OpenAI's own field, and the only part of a reasoning model's
    /// accounting that IS in their spec -- `reasoning_content` is a
    /// DeepSeek convention this server also speaks, but the token
    /// count is standard, and it is how a caller prices or budgets a
    /// thinking model.
    ///
    /// `None`, not zero, when this build cannot know: a checkpoint
    /// whose family emits no reasoning at all, and any path that did
    /// not run the split. `/v1/responses` used to report a hardcoded
    /// `0` here, which reads as "this model did not think" rather than
    /// "nobody counted" -- the exact confusion this module's header
    /// rules out for timings.
    pub fn with_reasoning_tokens(mut self, reasoning: usize) -> Self {
        self.completion_tokens_details = Some(CompletionTokensDetails {
            reasoning_tokens: reasoning,
        });
        self
    }

    /// Prompt tokens that came from the prefix cache. Call this only
    /// when a prefix cache actually exists (see `cached_tokens`).
    pub fn with_cached_tokens(mut self, cached: usize) -> Self {
        self.cached_tokens = Some(cached);
        self
    }

    /// Records what speculative decoding actually achieved for this
    /// request. Call this only when speculation ran: leaving the fields
    /// unset is how a non-speculative request says so, and a zero would
    /// read as "speculation ran and failed".
    ///
    /// `accepted` and `drafted` are token counts, `per_position` the
    /// accept rate at each position inside the draft block. A zero
    /// `verification_steps` leaves `acceptance_length` unset rather
    /// than dividing by zero.
    pub fn with_speculation(
        mut self,
        verification_steps: usize,
        accepted: usize,
        drafted: usize,
        per_position: Vec<f64>,
    ) -> Self {
        if verification_steps > 0 {
            self.acceptance_length =
                Some(self.completion_tokens as f64 / verification_steps as f64);
        }
        self.accepted_draft_tokens = Some(accepted);
        self.draft_tokens = Some(drafted);
        self.draft_accept_rate_per_position = Some(per_position);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn totals_are_the_sum_of_the_two_phases() {
        let usage = Usage::new(7, 3);
        assert_eq!(usage.total_tokens, 10);
    }

    #[test]
    fn phase_durations_stay_separate() {
        // 100 prompt tokens in 1s, 10 generated in 1s. A client that
        // conflated the phases would report 110/2 = 55 tok/s for both.
        let usage = Usage::new(100, 10).with_timings(1.0, 1.0);
        assert_eq!(usage.prompt_per_second, Some(100.0));
        assert_eq!(usage.predicted_per_second, Some(10.0));
        assert_eq!(usage.prompt_eval_duration_ms, Some(1000.0));
        assert_eq!(usage.generation_duration_ms, Some(1000.0));
    }

    #[test]
    fn zero_length_phases_do_not_become_infinite_rates() {
        let usage = Usage::new(5, 5).with_timings(0.0, 0.0);
        assert_eq!(usage.prompt_per_second, None);
        assert_eq!(usage.predicted_per_second, None);
        assert_eq!(usage.prompt_eval_duration_ms, Some(0.0));
    }

    #[test]
    fn ttft_is_unset_when_nothing_was_generated() {
        let usage = Usage::new(5, 0).with_ttft(0.25);
        assert_eq!(usage.time_to_first_token_ms, None);
        assert_eq!(
            Usage::new(5, 1).with_ttft(0.25).time_to_first_token_ms,
            Some(250.0)
        );
    }

    #[test]
    fn untimed_usage_serializes_to_the_plain_openai_shape() {
        // Older clients must not start seeing null-valued extras.
        let json = serde_json::to_string(&Usage::new(2, 3)).unwrap();
        assert_eq!(
            json,
            "{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5}"
        );
    }

    #[test]
    fn a_non_speculative_request_reports_no_acceptance_length_at_all() {
        // `None` and `1.0` mean different things: "speculation did not
        // run" and "speculation ran and never helped". A UI that saw a
        // zero or a one on every plain request would report a
        // speculative decoder that does not exist.
        let plain = Usage::new(10, 5);
        assert_eq!(plain.acceptance_length, None);
        assert_eq!(plain.draft_tokens, None);
        let json = serde_json::to_value(&plain).unwrap();
        assert!(json.get("acceptance_length").is_none());
        assert!(json.get("draft_accept_rate_per_position").is_none());
    }

    #[test]
    fn acceptance_length_is_completion_tokens_per_verification_step() {
        // 12 tokens out of 5 verification steps is an acceptance length
        // of 2.4 -- the published metric. Dividing by forward passes
        // including prefill, or by drafted tokens, gives a different
        // and incomparable number.
        let usage = Usage::new(20, 12).with_speculation(5, 7, 10, vec![0.9, 0.6, 0.2]);
        assert_eq!(usage.acceptance_length, Some(2.4));
        assert_eq!(usage.accepted_draft_tokens, Some(7));
        assert_eq!(usage.draft_tokens, Some(10));
        assert_eq!(
            usage.draft_accept_rate_per_position,
            Some(vec![0.9, 0.6, 0.2])
        );
    }

    #[test]
    fn speculation_that_verified_nothing_reports_no_length_rather_than_infinity() {
        let usage = Usage::new(20, 0).with_speculation(0, 0, 0, Vec::new());
        assert_eq!(usage.acceptance_length, None);
        // The counters are still reported: speculation was configured,
        // it just never got to verify anything.
        assert_eq!(usage.draft_tokens, Some(0));
    }

    #[test]
    fn a_prefix_cache_miss_is_distinguishable_from_no_prefix_cache() {
        assert_eq!(Usage::new(2, 3).cached_tokens, None);
        assert_eq!(
            Usage::new(2, 3).with_cached_tokens(0).cached_tokens,
            Some(0)
        );
    }
}
