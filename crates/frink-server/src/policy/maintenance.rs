//! The gate every request, rebuild, and shutdown passes through.
//!
//! One small state machine, and every interesting rule in it is a
//! fail-closed one.
//!
//! # Why a gate at all
//!
//! Three operations mutate a serving engine out from under its
//! requests: loading a model, re-splitting the caches, and stopping.
//! Each one has a window in which admitting a request would be wrong,
//! and each one can *fail in a way that leaves the engine in an unknown
//! state*. A boolean `is_serving` handles the first half and not the
//! second, which is how an engine ends up accepting work while a rebuild
//! is halfway through moving its KV pool.
//!
//! # The three fail-closed rules
//!
//! **A rebuild that times out does not reopen the gate.** The scheduler
//! may still be mid-rebuild -- a slow CUDA-graph recapture, a large
//! copy -- and nothing cancelled it. Reopening would admit requests into
//! a cache that is being resized under them. The state stays
//! [`Rebuilding`](MaintenanceState::Rebuilding) until a real reply
//! arrives and says which way it went.
//!
//! **A rebuild that never dispatched *does* reopen it.** This is the
//! exception that proves the rule, and it is exact: if the request never
//! reached the scheduler, the engine is untouched. Latching maintenance
//! on a transient enqueue error would leave the gate closed forever with
//! no reply ever coming to clear it.
//!
//! **A drain that times out never reopens admission either.** A server
//! that announced it was stopping, then accepted more work, then sealed
//! totals that do not include it, has produced accounting that is simply
//! false. The caller is expected to preserve the process and retry --
//! killing an engine that has not crossed the terminal accounting
//! barrier can lose a late sampled token.
//!
//! And one idempotency rule: a successful seal is cached for the life of
//! the process, so a supervisor that retries after losing the response
//! receives exactly the same totals rather than a second, smaller
//! snapshot.
//!
//! This module holds no clock and does no waiting. The caller owns the
//! timeouts; what is here is only what those timeouts *mean*.
//!
//! Ported from FreeToken's `server/accounting.py` and the maintenance
//! gate in `server/api_server.py` (Apache-2.0); see
//! `docs/THIRD_PARTY_NOTICES.md`.

/// What the engine is doing, and therefore what it will accept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaintenanceState {
    /// Weights are still being read. Nothing can be served yet.
    Loading,
    /// The only state that admits requests.
    Serving,
    /// A cache re-split is in flight.
    Rebuilding,
    /// Admission is closed and in-flight work is draining.
    Stopping,
    /// Latched. Something failed in a way that leaves the engine
    /// unusable, and only a restart clears it.
    Failed,
}

impl MaintenanceState {
    pub fn as_str(self) -> &'static str {
        match self {
            MaintenanceState::Loading => "loading",
            MaintenanceState::Serving => "serving",
            MaintenanceState::Rebuilding => "rebuilding",
            MaintenanceState::Stopping => "stopping",
            MaintenanceState::Failed => "failed",
        }
    }
}

/// Why a request was not admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionClosed {
    pub state: MaintenanceState,
}

impl std::fmt::Display for AdmissionClosed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "server unavailable: engine is {}", self.state.as_str())
    }
}

/// Why a rebuild was refused before it started.
///
/// The distinction between the three is the HTTP status a caller should
/// return, and it is not cosmetic: `Busy` is retryable in a moment,
/// `NotReady` is retryable when loading finishes, and `Latched` is not
/// retryable at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildRefused {
    /// Still loading; there is no cache to rebuild yet.
    NotReady,
    /// Another rebuild, or a stop, is already in flight.
    Busy(MaintenanceState),
    /// The engine is latched in maintenance; a restart is required.
    Latched,
}

impl std::fmt::Display for RebuildRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RebuildRefused::NotReady => {
                write!(f, "model is still loading; cannot rebuild cache yet")
            }
            RebuildRefused::Busy(MaintenanceState::Stopping) => {
                write!(f, "engine stop is in progress")
            }
            RebuildRefused::Busy(_) => write!(f, "a cache rebuild is already in progress"),
            RebuildRefused::Latched => {
                write!(f, "server latched in maintenance; restart required")
            }
        }
    }
}

/// Why a stop could not be prepared, or could not be sealed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopRefused {
    /// A rebuild owns the engine. Stopping through one would seal
    /// totals for a cache geometry that is being replaced.
    RebuildInProgress,
    /// The drain did not finish and there is nothing left to abort by
    /// name -- an active count with no identities behind it can neither
    /// be aborted nor proven terminal.
    UnidentifiedInflight(usize),
    /// The abort barrier expired with requests still active. Admission
    /// stays closed and nothing is sealed.
    AbortBarrierTimedOut(usize),
}

impl std::fmt::Display for StopRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopRefused::RebuildInProgress => {
                write!(
                    f,
                    "cache rebuild is in progress; retry stop after it finishes"
                )
            }
            StopRefused::UnidentifiedInflight(n) => write!(
                f,
                "accounting drain timed out with {n} unidentified request(s)"
            ),
            StopRefused::AbortBarrierTimedOut(n) => write!(
                f,
                "accounting abort barrier timed out with {n} request(s) still active"
            ),
        }
    }
}

/// The final snapshot a stopping engine hands its supervisor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedAccounting {
    pub model_id: Option<String>,
    pub prompt_tokens_total: u64,
    pub completion_tokens_total: u64,
    pub uptime_seconds: u64,
    /// Always `true` in a sealed snapshot -- it is what "sealed" means.
    /// Carried explicitly because the *refusal* shape carries
    /// `drain_complete: false`, and a supervisor reads the same field
    /// on both.
    pub drain_complete: bool,
}

/// The engine's maintenance gate.
#[derive(Debug, Clone)]
pub struct MaintenanceGate {
    state: MaintenanceState,
    sealed: Option<SealedAccounting>,
}

impl Default for MaintenanceGate {
    fn default() -> Self {
        Self::new()
    }
}

impl MaintenanceGate {
    /// A new engine is [`Loading`](MaintenanceState::Loading), never
    /// `Serving`: the gate must be closed before the weights are there,
    /// not opened optimistically and closed on the first failure.
    pub fn new() -> Self {
        MaintenanceGate {
            state: MaintenanceState::Loading,
            sealed: None,
        }
    }

    /// A gate for an engine whose weights are already resident.
    ///
    /// Only for a host that finished its load *before* it built the
    /// gate: the server constructs its state after the startup
    /// checkpoint is activated, so starting that gate in `Loading`
    /// would refuse every request forever, since nothing would ever
    /// call [`finish_loading`](Self::finish_loading). A host that loads
    /// after building the gate must use [`new`](Self::new) instead --
    /// an optimistically-open gate admits requests into weights that
    /// are not there yet.
    pub fn serving() -> Self {
        MaintenanceGate {
            state: MaintenanceState::Serving,
            sealed: None,
        }
    }

    pub fn state(&self) -> MaintenanceState {
        self.state
    }

    /// May a request be admitted right now?
    ///
    /// `Serving` and nothing else. Every other state is either "not yet"
    /// or "not any more", and both must reject rather than queue.
    pub fn check_admission(&self) -> Result<(), AdmissionClosed> {
        if self.state == MaintenanceState::Serving {
            Ok(())
        } else {
            Err(AdmissionClosed { state: self.state })
        }
    }

    /// Loading finished. `ok == false` latches the engine.
    pub fn finish_loading(&mut self, ok: bool) {
        self.state = if ok {
            MaintenanceState::Serving
        } else {
            MaintenanceState::Failed
        };
    }

    /// Latch the engine. Only a restart clears this.
    pub fn latch_failed(&mut self) {
        self.state = MaintenanceState::Failed;
    }

    /// Take the gate for a cache rebuild.
    ///
    /// Refuses from every state but `Serving`, and the refusal says
    /// which, because a caller turns that into three different HTTP
    /// statuses.
    pub fn begin_rebuild(&mut self) -> Result<(), RebuildRefused> {
        match self.state {
            MaintenanceState::Serving => {
                self.state = MaintenanceState::Rebuilding;
                Ok(())
            }
            MaintenanceState::Loading => Err(RebuildRefused::NotReady),
            MaintenanceState::Failed => Err(RebuildRefused::Latched),
            other => Err(RebuildRefused::Busy(other)),
        }
    }

    /// The rebuild request never reached the engine, so the engine is
    /// untouched: reopen.
    ///
    /// The *only* path that reopens a rebuilding gate. A transient
    /// enqueue error must not latch maintenance forever with no reply
    /// ever arriving to clear it.
    pub fn rebuild_never_dispatched(&mut self) {
        if self.state == MaintenanceState::Rebuilding {
            self.state = MaintenanceState::Serving;
        }
    }

    /// A real reply arrived for the rebuild.
    pub fn finish_rebuild(&mut self, ok: bool) {
        if self.state == MaintenanceState::Rebuilding {
            self.state = if ok {
                MaintenanceState::Serving
            } else {
                MaintenanceState::Failed
            };
        }
    }

    /// The rebuild's own timeout expired.
    ///
    /// Deliberately does nothing. The scheduler may still be mid-rebuild
    /// and nothing cancelled it, so the gate stays closed until a real
    /// reply says which way it went. Written as a named no-op rather
    /// than as an omission, because "we chose not to reopen here" is the
    /// rule and an absent call looks like a missing one.
    pub fn rebuild_timed_out(&self) {}

    /// Close admission for a shutdown.
    ///
    /// Idempotent. Refuses only from `Rebuilding`: sealing totals for a
    /// cache geometry that is being replaced would record a
    /// configuration the engine never actually served under.
    pub fn begin_stop(&mut self) -> Result<(), StopRefused> {
        if self.state == MaintenanceState::Rebuilding {
            return Err(StopRefused::RebuildInProgress);
        }
        self.state = MaintenanceState::Stopping;
        Ok(())
    }

    /// The snapshot already sealed by an earlier successful stop, if
    /// there was one.
    ///
    /// A supervisor whose response was lost retries and must receive
    /// exactly these numbers, not a second snapshot taken later.
    pub fn sealed(&self) -> Option<&SealedAccounting> {
        self.sealed.as_ref()
    }

    /// Seal the totals, once the drain has actually finished.
    ///
    /// `active` is what the caller's stats say is still in flight, and
    /// it is checked rather than trusted: sealing with work still live
    /// is exactly the race the barrier exists to prevent. A refusal
    /// leaves the gate closed and unsealed -- never reopened -- so the
    /// caller preserves the process and retries.
    pub fn seal(
        &mut self,
        active: usize,
        had_identities: bool,
        snapshot: impl FnOnce() -> SealedAccounting,
    ) -> Result<SealedAccounting, StopRefused> {
        if let Some(sealed) = &self.sealed {
            return Ok(sealed.clone());
        }
        if active > 0 {
            return Err(if had_identities {
                StopRefused::AbortBarrierTimedOut(active)
            } else {
                StopRefused::UnidentifiedInflight(active)
            });
        }
        let sealed = snapshot();
        self.sealed = Some(sealed.clone());
        Ok(sealed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> SealedAccounting {
        SealedAccounting {
            model_id: Some("m".to_string()),
            prompt_tokens_total: 10,
            completion_tokens_total: 20,
            uptime_seconds: 30,
            drain_complete: true,
        }
    }

    fn serving() -> MaintenanceGate {
        let mut gate = MaintenanceGate::new();
        gate.finish_loading(true);
        gate
    }

    #[test]
    fn an_engine_starts_closed_and_opens_only_when_loading_succeeds() {
        let mut gate = MaintenanceGate::new();
        assert!(gate.check_admission().is_err());
        gate.finish_loading(true);
        assert!(gate.check_admission().is_ok());

        let mut gate = MaintenanceGate::new();
        gate.finish_loading(false);
        assert_eq!(gate.state(), MaintenanceState::Failed);
        assert!(gate.check_admission().is_err());
    }

    #[test]
    fn only_serving_admits_and_the_refusal_names_the_state() {
        let mut gate = serving();
        gate.begin_rebuild().expect("rebuild starts");
        let err = gate.check_admission().expect_err("rebuilding must refuse");
        assert_eq!(err.to_string(), "server unavailable: engine is rebuilding");
    }

    /// The fail-closed rule that costs the most to get wrong: nothing
    /// cancelled the rebuild, so nothing may assume it finished.
    #[test]
    fn a_rebuild_timeout_leaves_the_gate_closed() {
        let mut gate = serving();
        gate.begin_rebuild().expect("rebuild starts");
        gate.rebuild_timed_out();
        assert_eq!(gate.state(), MaintenanceState::Rebuilding);
        assert!(gate.check_admission().is_err());

        // Only a real reply moves it.
        gate.finish_rebuild(true);
        assert!(gate.check_admission().is_ok());
    }

    /// And its exact exception: a request that never reached the engine
    /// left the engine untouched, so latching on it would close the gate
    /// forever with no reply ever coming to clear it.
    #[test]
    fn a_rebuild_that_never_dispatched_reopens_the_gate() {
        let mut gate = serving();
        gate.begin_rebuild().expect("rebuild starts");
        gate.rebuild_never_dispatched();
        assert_eq!(gate.state(), MaintenanceState::Serving);
        assert!(gate.check_admission().is_ok());
    }

    #[test]
    fn a_refused_rebuild_says_which_kind_of_no_it_is() {
        let mut loading = MaintenanceGate::new();
        assert_eq!(loading.begin_rebuild(), Err(RebuildRefused::NotReady));

        let mut latched = serving();
        latched.latch_failed();
        assert_eq!(latched.begin_rebuild(), Err(RebuildRefused::Latched));

        let mut busy = serving();
        busy.begin_rebuild().expect("first");
        assert_eq!(
            busy.begin_rebuild(),
            Err(RebuildRefused::Busy(MaintenanceState::Rebuilding))
        );

        let mut stopping = serving();
        stopping.begin_stop().expect("stop starts");
        assert_eq!(
            stopping.begin_rebuild(),
            Err(RebuildRefused::Busy(MaintenanceState::Stopping))
        );
    }

    #[test]
    fn a_stop_cannot_start_through_a_rebuild() {
        let mut gate = serving();
        gate.begin_rebuild().expect("rebuild starts");
        assert_eq!(gate.begin_stop(), Err(StopRefused::RebuildInProgress));
        assert_eq!(gate.state(), MaintenanceState::Rebuilding);
    }

    #[test]
    fn a_drained_stop_seals() {
        let mut gate = serving();
        gate.begin_stop().expect("stop starts");
        assert_eq!(gate.seal(0, false, snapshot), Ok(snapshot()));
        assert_eq!(gate.sealed(), Some(&snapshot()));
    }

    /// The rule the whole barrier exists for. A request that will not
    /// terminate must not be able to talk the server back into serving,
    /// and must not be sealed around.
    #[test]
    fn an_abort_barrier_timeout_seals_nothing_and_reopens_nothing() {
        let mut gate = serving();
        gate.begin_stop().expect("stop starts");
        assert_eq!(
            gate.seal(2, true, snapshot),
            Err(StopRefused::AbortBarrierTimedOut(2))
        );
        assert_eq!(gate.state(), MaintenanceState::Stopping);
        assert!(gate.check_admission().is_err());
        assert_eq!(gate.sealed(), None);

        // And once the stragglers really do finish, the retry seals.
        assert_eq!(gate.seal(0, true, snapshot), Ok(snapshot()));
    }

    /// An active count with no identities behind it can be neither
    /// aborted nor proven terminal, which is a different failure from a
    /// straggler that simply took too long.
    #[test]
    fn an_active_count_with_no_identities_is_its_own_refusal() {
        let mut gate = serving();
        gate.begin_stop().expect("stop starts");
        assert_eq!(
            gate.seal(1, false, snapshot),
            Err(StopRefused::UnidentifiedInflight(1))
        );
    }

    /// A supervisor that lost the response retries and must get the same
    /// numbers, not a second snapshot taken later.
    #[test]
    fn sealing_is_idempotent_for_the_life_of_the_process() {
        let mut gate = serving();
        gate.begin_stop().expect("stop starts");
        let first = gate.seal(0, false, snapshot).expect("seals");
        let second = gate
            .seal(0, false, || SealedAccounting {
                completion_tokens_total: 99_999,
                ..snapshot()
            })
            .expect("seals again");
        assert_eq!(first, second, "a retry must not re-measure");
    }
}
