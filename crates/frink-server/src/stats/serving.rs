//! Live serving counters: what is admitted right now, and what became
//! of everything that is not.

use std::collections::BTreeSet;

use super::rate::RateWindow;

/// Live serving counters: what is in flight, what completed, and at what
/// rate.
///
/// The `aborting` set is the part worth reading twice. A request that
/// has been sent an abort stays *active* until its terminal reply
/// actually arrives, and then completes the count as an abort rather
/// than as a completion. Dropping it from the in-flight set at dispatch
/// would let a shutdown declare itself drained while a request was still
/// producing tokens, and counting it as completed would make a
/// timed-out shutdown look like a clean one.
#[derive(Debug, Clone)]
pub struct ServingStats {
    inflight: BTreeSet<u64>,
    aborting: BTreeSet<u64>,
    completed: u64,
    aborted: u64,
    prompt_tokens_total: u64,
    completion_tokens_total: u64,
    decode: RateWindow,
    prefill: RateWindow,
}

impl Default for ServingStats {
    fn default() -> Self {
        Self::new(5_000)
    }
}

impl ServingStats {
    pub fn new(window_ms: u64) -> Self {
        ServingStats {
            inflight: BTreeSet::new(),
            aborting: BTreeSet::new(),
            completed: 0,
            aborted: 0,
            prompt_tokens_total: 0,
            completion_tokens_total: 0,
            decode: RateWindow::new(window_ms),
            prefill: RateWindow::new(window_ms),
        }
    }

    /// One request admitted. Re-admitting an id clears any abort left
    /// over from a previous life of that slot.
    pub fn on_admitted(&mut self, uid: u64) {
        self.inflight.insert(uid);
        self.aborting.remove(&uid);
    }

    /// An abort was dispatched. The request stays active until its
    /// terminal reply arrives.
    pub fn on_abort(&mut self, uid: u64) {
        if self.inflight.contains(&uid) {
            self.aborting.insert(uid);
        }
    }

    pub fn on_prefill(&mut self, at_ms: u64, tokens: u64) {
        self.prompt_tokens_total += tokens;
        self.prefill.record(at_ms, tokens);
    }

    pub fn on_decode(&mut self, at_ms: u64, tokens: u64) {
        self.completion_tokens_total += tokens;
        self.decode.record(at_ms, tokens);
    }

    /// A terminal reply arrived for `uid`.
    pub fn on_finished(&mut self, uid: u64) {
        if !self.inflight.remove(&uid) {
            return;
        }
        if self.aborting.remove(&uid) {
            self.aborted += 1;
        } else {
            self.completed += 1;
        }
    }

    pub fn active(&self) -> usize {
        self.inflight.len()
    }

    /// A stable snapshot of everything still admitted, for a shutdown
    /// that has to abort each one by name.
    pub fn inflight_uids(&self) -> Vec<u64> {
        self.inflight.iter().copied().collect()
    }

    pub fn completed(&self) -> u64 {
        self.completed
    }

    pub fn aborted(&self) -> u64 {
        self.aborted
    }

    pub fn prompt_tokens_total(&self) -> u64 {
        self.prompt_tokens_total
    }

    pub fn completion_tokens_total(&self) -> u64 {
        self.completion_tokens_total
    }

    pub fn decode_tokens_per_second(&mut self, now_ms: u64) -> f64 {
        self.decode.tokens_per_second(now_ms)
    }

    pub fn prefill_tokens_per_second(&mut self, now_ms: u64) -> f64 {
        self.prefill.tokens_per_second(now_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_is_counted_once_it_reaches_a_terminal_reply() {
        let mut stats = ServingStats::new(1_000);
        stats.on_admitted(1);
        stats.on_admitted(2);
        assert_eq!(stats.active(), 2);
        assert_eq!(stats.inflight_uids(), vec![1, 2]);
        stats.on_finished(1);
        assert_eq!(stats.completed(), 1);
        assert_eq!(stats.active(), 1);
    }

    /// The abort rule, which is what a shutdown's correctness rests on:
    /// dispatching an abort does not make a request inactive, and when
    /// it does end it is not a completion.
    #[test]
    fn an_aborted_request_stays_active_until_its_terminal_reply() {
        let mut stats = ServingStats::new(1_000);
        stats.on_admitted(7);
        stats.on_abort(7);
        assert_eq!(stats.active(), 1, "an abort is dispatched, not applied");
        assert_eq!(stats.completed(), 0);

        // Tokens racing the abort still count toward lifetime totals.
        stats.on_decode(10, 3);
        stats.on_finished(7);
        assert_eq!(stats.active(), 0);
        assert_eq!(stats.completed(), 0);
        assert_eq!(stats.aborted(), 1);
        assert_eq!(stats.completion_tokens_total(), 3);
    }

    #[test]
    fn a_terminal_reply_for_an_unknown_request_changes_nothing() {
        let mut stats = ServingStats::new(1_000);
        stats.on_admitted(1);
        stats.on_finished(1);
        stats.on_finished(1);
        assert_eq!(stats.completed(), 1);
        assert_eq!(stats.active(), 0);
    }
}
