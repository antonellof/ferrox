//! Wire shapes for `POST /v1/cancel`.
//!
//! One request id in, one honest verdict out. The verdict is
//! deliberately not a bare `ok: true`: a client that cancels the moment
//! the last token arrives, or retries a cancel after the first one
//! worked, needs to be able to tell "I stopped a live generation" from
//! "there was nothing left to stop". Both are fine outcomes; only one
//! of them saved any work, and a UI that cannot distinguish them will
//! claim it stopped something it did not.

use serde::{Deserialize, Serialize};

/// The body of `POST /v1/cancel`.
///
/// `request_id` is the value the server states as `request_id` on the
/// first chunk of a streamed completion (and as `id` on every chunk),
/// so a client never has to invent or correlate one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancelGenerationRequest {
    pub request_id: String,
}

/// The answer to `POST /v1/cancel`.
///
/// Sent with `200` when a live generation was signalled and with `404`
/// when the id names nothing that is running -- already finished, never
/// issued, or served by a path that does not register for cancellation.
/// The body says the same thing in both cases so a client that only
/// reads JSON is not left guessing at the status line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancelGenerationResponse {
    pub request_id: String,
    /// Whether a generation was actually signalled to stop.
    pub cancelled: bool,
    /// A human sentence for the state above, on the same principle as
    /// the capability reasons in [`crate::health`]: the UI shows the
    /// server's explanation rather than re-deriving one.
    pub detail: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_verdict_survives_a_round_trip_and_keeps_both_states() {
        for cancelled in [true, false] {
            let response = CancelGenerationResponse {
                request_id: "chatcmpl-1".to_string(),
                cancelled,
                detail: "…".to_string(),
            };
            let json = serde_json::to_string(&response).unwrap();
            assert_eq!(
                serde_json::from_str::<CancelGenerationResponse>(&json).unwrap(),
                response
            );
            // The field must never be elided: a missing `cancelled`
            // would default to `false` in some clients and `true` in
            // the reader's head.
            assert!(json.contains("\"cancelled\""), "{json}");
        }
    }

    #[test]
    fn the_request_names_only_the_id_the_server_already_stated() {
        let parsed: CancelGenerationRequest =
            serde_json::from_str(r#"{"request_id":"chatcmpl-abc"}"#).unwrap();
        assert_eq!(parsed.request_id, "chatcmpl-abc");
    }
}
