//! Backend detection behind `GET /health`.
//!
//! The endpoint answers in one of three states, and the third one is the
//! point. A UI that renders its *guess* while backends are still being
//! probed greys out the GPU controls in a way that is pixel-identical to
//! a measured "your machine cannot do this" -- and the user reads the
//! guess as the verdict. So detection is a state the client can see and
//! hold on (`detecting`), not a gap it has to paper over.
//!
//! Two rules make that safe to rely on:
//!
//! 1. **The handler never blocks.** Detection runs once in the
//!    background; `/health` reads whatever is known at that instant and
//!    returns immediately. A probe that hangs must not take the health
//!    endpoint down with it -- that is precisely when a supervisor is
//!    about to ask whether the process is alive.
//! 2. **Hard [`DETECTION_BUDGET`], then answer provisionally.** After
//!    the budget elapses the state stays `detecting`, but the capability
//!    list is filled in with a CPU-only view whose reason code is
//!    `detection_timed_out`. The client can render something without
//!    being told it is a measurement.
//!
//! Every capability carries a machine `reason` and a human `detail`, so
//! the UI greys a control and shows the sentence rather than re-deriving
//! an explanation the server already knows. "No Metal device" and "this
//! binary has no Metal kernels" look the same to a boolean and lead to
//! completely different advice.

use std::sync::Mutex;
use std::time::Duration;

use frink_api::health::{capability, reason, Capability, HealthState};

/// How long detection gets before `/health` starts answering
/// provisionally. The desktop shell probes with a short timeout and does
/// not retry, so an unbounded probe reads to it as a dead backend.
pub const DETECTION_BUDGET: Duration = Duration::from_secs(1);

pub struct HealthSnapshot {
    pub state: HealthState,
    pub capabilities: Vec<Capability>,
}

#[derive(Default)]
struct Inner {
    /// Measured backend capabilities; `None` until the probe finishes.
    measured: Option<Vec<Capability>>,
    /// Set when the probe overran [`DETECTION_BUDGET`]. The probe is
    /// still running: a late answer replaces this.
    timed_out: bool,
}

#[derive(Default)]
pub struct Detection {
    inner: Mutex<Inner>,
}

impl Detection {
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts the probe and returns immediately. Runs on the blocking
    /// pool because device enumeration is a synchronous FFI call.
    pub fn spawn() -> std::sync::Arc<Self> {
        let detection = std::sync::Arc::new(Detection::new());
        let handle_target = std::sync::Arc::clone(&detection);
        tokio::spawn(async move {
            let mut probe = tokio::task::spawn_blocking(probe_backends);
            // `&mut JoinHandle` so the timeout does not consume the
            // task: a probe that overran the budget is still worth
            // waiting for, it just no longer gets to hold up an answer.
            match tokio::time::timeout(DETECTION_BUDGET, &mut probe).await {
                Ok(Ok(caps)) => handle_target.complete(caps),
                Ok(Err(e)) => {
                    tracing::warn!("backend detection panicked: {e}");
                    handle_target.mark_timed_out();
                }
                Err(_) => {
                    tracing::warn!(
                        "backend detection exceeded its {:?} budget; /health answers \
                         provisionally until it lands",
                        DETECTION_BUDGET
                    );
                    handle_target.mark_timed_out();
                    if let Ok(caps) = probe.await {
                        handle_target.complete(caps);
                    }
                }
            }
        });
        detection
    }

    /// A `Detection` that is already finished, for tests and for any
    /// path that has no runtime to spawn on.
    #[cfg(test)]
    pub fn ready(capabilities: Vec<Capability>) -> Self {
        let detection = Detection::new();
        detection.complete(capabilities);
        detection
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        // A poisoned lock here would mean a panic while swapping a
        // capability list -- the data is still consistent, and refusing
        // to answer /health because of it would be strictly worse.
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn complete(&self, capabilities: Vec<Capability>) {
        let mut inner = self.lock();
        inner.measured = Some(capabilities);
        inner.timed_out = false;
    }

    fn mark_timed_out(&self) {
        self.lock().timed_out = true;
    }

    /// What `/health` should say right now. Never blocks.
    pub fn snapshot(&self) -> HealthSnapshot {
        let inner = self.lock();
        match (&inner.measured, inner.timed_out) {
            (Some(caps), _) => HealthSnapshot {
                state: HealthState::Ready,
                capabilities: caps.clone(),
            },
            // Budget blown, still no answer: say what is certain (CPU
            // kernels always work) and mark the rest as unmeasured.
            (None, true) => HealthSnapshot {
                state: HealthState::Detecting,
                capabilities: vec![
                    cpu_capability(),
                    timed_out_capability(capability::METAL),
                    timed_out_capability(capability::CUDA),
                ],
            },
            // Still inside the budget: the only honest answer is the
            // one that carries no verdict about the GPU at all.
            (None, false) => HealthSnapshot {
                state: HealthState::Detecting,
                capabilities: vec![cpu_capability()],
            },
        }
    }
}

fn timed_out_capability(id: &str) -> Capability {
    Capability::unavailable(
        id,
        reason::DETECTION_TIMED_OUT,
        "Still probing this backend; treat as unknown, not unsupported.",
    )
}

fn cpu_capability() -> Capability {
    Capability::available(
        capability::CPU,
        format!(
            "Quantized CPU kernels on {} performance core(s) (override with FRINK_CPU_THREADS).",
            frink_core::threads::perf_core_count()
        ),
    )
}

/// Synchronous device enumeration. Called once, off the reactor.
pub fn probe_backends() -> Vec<Capability> {
    vec![cpu_capability(), metal_capability(), cuda_capability()]
}

#[cfg(feature = "metal")]
fn metal_capability() -> Capability {
    let profile = frink_metal::MetalProfile::detect();
    let Some(name) = profile.device_name.filter(|_| profile.available) else {
        return Capability::unavailable(
            capability::METAL,
            reason::METAL_UNAVAILABLE,
            "This build has Metal kernels but no Metal device was found.",
        );
    };
    // A present device that the operator turned off is not the same
    // answer as an absent one: the fix is a flag, not new hardware.
    if !frink_core::metal_dense_enabled() {
        return Capability::unavailable(
            capability::METAL,
            reason::DISABLED,
            format!("{name} is present but Metal offload is off (FRINK_METAL=0 / --device cpu)."),
        );
    }
    Capability::available(capability::METAL, format!("Metal kernels on {name}."))
}

#[cfg(not(feature = "metal"))]
fn metal_capability() -> Capability {
    // Probe even without the feature: "you have an M-series GPU but this
    // binary cannot use it" is a build problem the user can fix, and
    // reporting it as "no GPU" hides that.
    let profile = frink_metal::MetalProfile::detect();
    let detail = match profile.device_name.filter(|_| profile.available) {
        Some(name) => format!(
            "{name} is present but this binary was built without --features metal; \
             rebuild to use it."
        ),
        None => "This binary was built without --features metal.".to_string(),
    };
    Capability::unavailable(capability::METAL, reason::METAL_NOT_BUILT, detail)
}

#[cfg(feature = "cuda")]
fn cuda_capability() -> Capability {
    let profile = frink_cuda::HardwareProfile::detect();
    if !profile.cuda_available {
        return Capability::unavailable(
            capability::CUDA,
            reason::CUDA_UNAVAILABLE,
            "This build has CUDA kernels but no CUDA device was found.",
        );
    }
    let name = profile
        .cuda_device_name
        .unwrap_or_else(|| "unknown device".to_string());
    Capability::available(
        capability::CUDA,
        format!(
            "CUDA kernels on {name} ({} device(s)); routed experts need \
             FRINK_GPU_VRAM_BUDGET_BYTES.",
            profile.cuda_device_count
        ),
    )
}

#[cfg(not(feature = "cuda"))]
fn cuda_capability() -> Capability {
    let profile = frink_cuda::HardwareProfile::detect();
    let detail = if profile.cuda_available {
        let name = profile
            .cuda_device_name
            .unwrap_or_else(|| "unknown device".to_string());
        format!(
            "{name} is present but this binary was built without --features cuda; \
             rebuild to use it."
        )
    } else {
        "This binary was built without --features cuda.".to_string()
    };
    Capability::unavailable(capability::CUDA, reason::CUDA_NOT_BUILT, detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn before_the_budget_no_gpu_verdict_is_offered_at_all() {
        let detection = Detection::new();
        let snap = detection.snapshot();
        assert_eq!(snap.state, HealthState::Detecting);
        // The client must not be able to read "no Metal" out of a
        // response that simply has not looked yet.
        assert!(snap
            .capabilities
            .iter()
            .all(|c| c.id != capability::METAL && c.id != capability::CUDA));
        assert!(snap
            .capabilities
            .iter()
            .any(|c| c.id == capability::CPU && c.available));
    }

    #[test]
    fn after_the_budget_the_answer_is_marked_provisional_not_unsupported() {
        let detection = Detection::new();
        detection.mark_timed_out();
        let snap = detection.snapshot();
        assert_eq!(snap.state, HealthState::Detecting);
        let metal = snap
            .capabilities
            .iter()
            .find(|c| c.id == capability::METAL)
            .expect("provisional answer names the backend");
        assert!(!metal.available);
        assert_eq!(metal.reason, reason::DETECTION_TIMED_OUT);
    }

    #[test]
    fn a_late_probe_result_replaces_the_provisional_answer() {
        let detection = Detection::new();
        detection.mark_timed_out();
        detection.complete(probe_backends());
        let snap = detection.snapshot();
        assert_eq!(snap.state, HealthState::Ready);
        assert!(snap
            .capabilities
            .iter()
            .all(|c| c.reason != reason::DETECTION_TIMED_OUT));
    }

    #[test]
    fn every_probed_capability_carries_a_reason_and_a_sentence() {
        for cap in probe_backends() {
            assert!(!cap.reason.is_empty(), "{cap:?}");
            assert!(
                cap.detail.ends_with('.'),
                "{cap:?} detail should read as a sentence"
            );
            if !cap.available {
                assert_ne!(cap.reason, reason::AVAILABLE, "{cap:?}");
            }
        }
    }

    #[test]
    fn an_unbuilt_backend_says_so_rather_than_reporting_missing_hardware() {
        // This test binary is built without --features metal/cuda in
        // CI, and the distinction is the whole reason `reason` exists.
        let caps = probe_backends();
        let metal = caps.iter().find(|c| c.id == capability::METAL).unwrap();
        let cuda = caps.iter().find(|c| c.id == capability::CUDA).unwrap();
        #[cfg(not(feature = "metal"))]
        assert_eq!(metal.reason, reason::METAL_NOT_BUILT);
        #[cfg(not(feature = "cuda"))]
        assert_eq!(cuda.reason, reason::CUDA_NOT_BUILT);
        #[cfg(feature = "metal")]
        assert_ne!(metal.reason, reason::METAL_NOT_BUILT);
        #[cfg(feature = "cuda")]
        assert_ne!(cuda.reason, reason::CUDA_NOT_BUILT);
        let _ = (metal, cuda);
    }

    #[tokio::test]
    async fn a_spawned_probe_reaches_ready_without_blocking_the_first_answer() {
        let detection = Detection::spawn();
        // Answering immediately is the requirement; whether the probe
        // has landed by now is deliberately not asserted.
        assert!(matches!(
            detection.snapshot().state,
            HealthState::Detecting | HealthState::Ready
        ));
        for _ in 0..50 {
            if detection.snapshot().state == HealthState::Ready {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("detection never completed");
    }
}
