//! `GET /health` as a capability handshake rather than a boolean.
//!
//! Three states, not two. `detecting` exists because a UI that renders
//! its *guess* while backends are still being probed paints a
//! greyed-out GPU control that is pixel-identical to a measured
//! "unsupported" -- and the user reads the guess as a verdict. With a
//! third state the client can hold: show a spinner, not a conclusion.
//!
//! Every capability carries **both** a stable machine `reason` and a
//! human `detail` sentence. The client greys the control and puts
//! `detail` in the tooltip; it never re-derives the explanation from the
//! flag, because the server is the only side that knows whether Metal is
//! missing (no device) or merely not compiled in (`--features metal`),
//! and those two produce completely different advice.

use serde::{Deserialize, Serialize};

/// Stable machine-readable reason codes. Clients may switch on these;
/// the human `detail` string beside them is free-form and may be
/// reworded at any time.
pub mod reason {
    /// The capability is present and usable.
    pub const AVAILABLE: &str = "available";
    /// Built with Metal support, but no usable Metal device was found.
    pub const METAL_UNAVAILABLE: &str = "metal_unavailable";
    /// This binary was compiled without `--features metal`.
    pub const METAL_NOT_BUILT: &str = "metal_not_built";
    /// Built with CUDA support, but no CUDA device was found.
    pub const CUDA_UNAVAILABLE: &str = "cuda_unavailable";
    /// This binary was compiled without `--features cuda`.
    pub const CUDA_NOT_BUILT: &str = "cuda_not_built";
    /// No GPU backend is usable; work runs on CPU kernels.
    pub const CPU_ONLY: &str = "cpu_only";
    /// Backend probing is still in flight; nothing beside this reason
    /// is a measurement yet.
    pub const DETECTING: &str = "detecting";
    /// Backend probing did not finish inside its budget, so the answer
    /// beside this reason is provisional and may improve.
    pub const DETECTION_TIMED_OUT: &str = "detection_timed_out";
    /// Supported by this build and this host, but switched off by
    /// configuration -- the fix is a flag, not new hardware.
    pub const DISABLED: &str = "disabled";
    /// No model is loaded, so generation endpoints will fail.
    pub const MODEL_NOT_LOADED: &str = "model_not_loaded";
}

/// Well-known capability ids, so a client can look one up rather than
/// pattern-matching on position in the list.
pub mod capability {
    pub const CPU: &str = "cpu";
    pub const METAL: &str = "metal";
    pub const CUDA: &str = "cuda";
    /// Whether the loaded weights are real, or the synthetic
    /// random-weight demo model that frink falls back to.
    pub const REAL_WEIGHTS: &str = "real_weights";
    /// Continuous batching (`FRINK_CONTINUOUS_BATCHING=1`).
    pub const CONTINUOUS_BATCHING: &str = "continuous_batching";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthState {
    /// Backends are still being probed. Anything capability-shaped in
    /// this response is provisional; hold the UI rather than rendering
    /// a verdict.
    Detecting,
    /// Model loaded, backends probed, ready to serve.
    Ready,
    /// The port is bound (this response arrived) but the server cannot
    /// serve generation. `reason`/`detail` say why.
    Unavailable,
}

impl HealthState {
    /// The HTTP status a server should answer with. `detecting` is a
    /// 200: the process is alive and the client is expected to poll,
    /// and a 503 there would trip generic "backend down" logic in
    /// proxies and supervisors. Only `unavailable` is a 503.
    pub fn http_status(self) -> u16 {
        match self {
            HealthState::Detecting | HealthState::Ready => 200,
            HealthState::Unavailable => 503,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    /// See [`capability`] for the well-known ids.
    pub id: String,
    pub available: bool,
    /// One of [`reason`]. Stable; safe to switch on.
    pub reason: String,
    /// A sentence to show the user. Free-form; never parse it.
    pub detail: String,
}

impl Capability {
    pub fn available(id: &str, detail: impl Into<String>) -> Self {
        Capability {
            id: id.to_string(),
            available: true,
            reason: reason::AVAILABLE.to_string(),
            detail: detail.into(),
        }
    }

    pub fn unavailable(id: &str, reason: &str, detail: impl Into<String>) -> Self {
        Capability {
            id: id.to_string(),
            available: false,
            reason: reason.to_string(),
            detail: detail.into(),
        }
    }
}

/// Summary of what is loaded, so the client does not need a second
/// round-trip to `/v1/models` just to label the connection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSummary {
    pub id: String,
    pub tokenizer: String,
    /// True when the weights are the synthetic random-weight demo, not
    /// a real checkpoint. Output from such a model is noise; a UI that
    /// does not say so invites a bug report about "quality".
    pub synthetic_weights: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthResponse {
    pub state: HealthState,
    /// Machine code for a non-`ready` state (see [`reason`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Human sentence for a non-`ready` state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelSummary>,
    pub capabilities: Vec<Capability>,
    pub version: String,
    pub pid: u32,
    pub uptime_seconds: f64,
    /// Server wall clock at the moment of the answer. The browser's own
    /// clock is not trusted for anything the server timestamps -- a
    /// skewed client would otherwise render negative durations.
    pub server_time_unix_ms: u64,
    /// Seconds since this process last finished a request, when it has
    /// served one. A GPU saturated by a long decode can starve the
    /// health handler; a client that sees recent request activity has
    /// positive evidence of liveness and must not declare the backend
    /// dead on a single slow poll.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_request_age_seconds: Option<f64>,
}

impl HealthResponse {
    /// Look up a capability by id (see [`capability`]).
    pub fn capability(&self, id: &str) -> Option<&Capability> {
        self.capabilities.iter().find(|c| c.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detecting_is_not_an_error_status() {
        assert_eq!(HealthState::Detecting.http_status(), 200);
        assert_eq!(HealthState::Ready.http_status(), 200);
        assert_eq!(HealthState::Unavailable.http_status(), 503);
    }

    #[test]
    fn states_serialize_as_snake_case_strings() {
        let json = serde_json::to_string(&HealthState::Detecting).unwrap();
        assert_eq!(json, "\"detecting\"");
    }

    #[test]
    fn unavailable_capability_keeps_both_a_code_and_a_sentence() {
        let cap = Capability::unavailable(
            capability::CUDA,
            reason::CUDA_NOT_BUILT,
            "This build has no CUDA kernels; rebuild with --features cuda.",
        );
        // The UI must be able to switch on `reason` and show `detail`
        // without ever inferring one from the other.
        assert!(!cap.available);
        assert_eq!(cap.reason, "cuda_not_built");
        assert!(cap.detail.contains("--features cuda"));
    }

    #[test]
    fn optional_fields_are_omitted_rather_than_null() {
        let health = HealthResponse {
            state: HealthState::Ready,
            reason: None,
            detail: None,
            model: None,
            capabilities: vec![Capability::available(capability::CPU, "CPU kernels")],
            version: "0.5.0".into(),
            pid: 1,
            uptime_seconds: 1.0,
            server_time_unix_ms: 0,
            last_request_age_seconds: None,
        };
        let json = serde_json::to_string(&health).unwrap();
        assert!(!json.contains("null"), "{json}");
        let back: HealthResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back, health);
        assert!(back.capability(capability::CPU).is_some());
    }
}
