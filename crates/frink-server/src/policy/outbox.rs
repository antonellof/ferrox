//! The accounting receipt, and the one ordering that keeps it honest.
//!
//! [`crate::policy::maintenance::MaintenanceGate`] is the barrier half of a
//! stop: it closes admission, refuses to seal while work is in flight,
//! and hands back a [`SealedAccounting`] exactly once. This module is
//! the other half -- turning that snapshot into a receipt that cannot
//! be lost and cannot be counted twice.
//!
//! # The ordering IS the rule
//!
//! 1. prepare-stop (the gate; done elsewhere),
//! 2. the receipt **durable**,
//! 3. and only then, signal.
//!
//! Reversed, a crash between the signal and the write loses a whole
//! engine generation's token totals -- the engine is gone, and the only
//! record of what it served went with it. [`finish_stop`] enforces the
//! order by construction: a failed persist never reaches the signal,
//! and there is no argument order that lets a caller do it the other
//! way round.
//!
//! # Why the id is derived and not generated
//!
//! A supervisor whose response was lost retries. With a fresh id per
//! attempt, the retry writes a SECOND receipt for one engine
//! generation, which downstream is a second billing event for work
//! that happened once. [`receipt_id`] is a pure function of a stable
//! identity, so a retry produces the same id, addresses the same
//! document, and persisting is idempotent.
//!
//! Ported from FreeToken's `daemon/accounting.py`; see
//! `docs/THIRD_PARTY_NOTICES.md`.

use sha2::{Digest, Sha256};

use crate::policy::maintenance::SealedAccounting;

/// How much a receipt claims to know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptStatus {
    /// The drain finished: every request that was in flight either
    /// completed or was accounted for, so the totals are the whole
    /// generation's.
    Complete,
    /// The drain did NOT finish. The totals are a lower bound, and the
    /// receipt says so rather than being accepted as final -- a
    /// downstream that bills a degraded receipt as complete undercounts
    /// silently, which is the failure mode nobody notices.
    Degraded,
}

impl ReceiptStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ReceiptStatus::Complete => "complete",
            ReceiptStatus::Degraded => "degraded",
        }
    }
}

/// What one engine generation served, addressed by a stable id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub id: String,
    pub model_id: Option<String>,
    pub prompt_tokens_total: u64,
    pub completion_tokens_total: u64,
    pub uptime_seconds: u64,
    pub status: ReceiptStatus,
}

impl Receipt {
    /// A receipt for a sealed generation.
    ///
    /// `identity` is whatever names THIS engine generation stably
    /// across retries -- an instance id, a boot id, a lease. It is the
    /// caller's to choose because only the caller knows what survives a
    /// restart in its deployment; what this module guarantees is that
    /// the same identity yields the same receipt id.
    ///
    /// A snapshot whose drain did not complete is DEMOTED here rather
    /// than at the point of use. Demotion at the boundary means every
    /// consumer sees the qualification; leaving it to consumers means
    /// the first one that forgets bills a partial total as a whole one.
    pub fn from_sealed(identity: &str, sealed: &SealedAccounting) -> Self {
        Receipt {
            id: receipt_id(identity),
            model_id: sealed.model_id.clone(),
            prompt_tokens_total: sealed.prompt_tokens_total,
            completion_tokens_total: sealed.completion_tokens_total,
            uptime_seconds: sealed.uptime_seconds,
            status: if sealed.drain_complete {
                ReceiptStatus::Complete
            } else {
                ReceiptStatus::Degraded
            },
        }
    }
}

/// The namespace every frink receipt id is derived under, so an id
/// from this engine cannot collide with one derived elsewhere from the
/// same identity string.
const RECEIPT_NAMESPACE: &str = "frink.accounting.receipt.v1";

/// A stable, UUID-shaped id for one engine generation.
///
/// Deterministic by construction: the same identity always yields the
/// same id, which is what makes persisting idempotent and a retried
/// stop reuse its document instead of creating a second billing event.
///
/// **This is a UUIDv8, not a v5.** RFC 9562 reserves version 8 for
/// custom, implementation-defined derivations, which is exactly what
/// this is: SHA-256 over the namespace and the identity, truncated to
/// 128 bits, with the version and variant bits set. A v5 would be
/// SHA-1-based, and frink does not depend on SHA-1; emitting a v8 and
/// saying so beats stamping `5` on a digest that is not one, which
/// would mislead anyone who later tried to re-derive the id.
pub fn receipt_id(identity: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(RECEIPT_NAMESPACE.as_bytes());
    // A separator, so ("ab", "c") and ("a", "bc") cannot collide. The
    // namespace is fixed today and this costs nothing, but an id scheme
    // that is only unambiguous by luck is one refactor from not being.
    hasher.update([0u8]);
    hasher.update(identity.as_bytes());
    let digest = hasher.finalize();

    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    // Version 8 in the high nibble of byte 6, RFC 4122 variant in the
    // top two bits of byte 8.
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Why a stop did not complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopFailure {
    /// The receipt could not be made durable. **Nothing was signalled**,
    /// which is the whole point: the engine is still there, and a
    /// supervisor that retries will seal the same totals under the same
    /// id.
    NotPersisted(String),
    /// The receipt IS durable and the signal failed. Materially
    /// different from the above and reported separately: the record
    /// survives, so a retry is safe and cannot double-count, and an
    /// operator must not be sent to look for a lost receipt that is on
    /// disk.
    NotSignalled {
        receipt: Box<Receipt>,
        error: String,
    },
}

impl std::fmt::Display for StopFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopFailure::NotPersisted(e) => {
                write!(
                    f,
                    "accounting receipt was not persisted, nothing signalled: {e}"
                )
            }
            StopFailure::NotSignalled { receipt, error } => write!(
                f,
                "accounting receipt {} is durable but the stop was not signalled: {error}",
                receipt.id
            ),
        }
    }
}

impl std::error::Error for StopFailure {}

/// Persist, then signal -- and never the other way round.
///
/// The argument order is the contract. A caller cannot signal first,
/// because `signal` is only reached when `persist` returned `Ok`, and
/// there is no path through this function that skips the write.
///
/// `persist` must be idempotent by [`Receipt::id`]: a retried stop
/// derives the same id and addresses the same document, so a second
/// call for the same generation is expected and must succeed rather
/// than conflict.
pub fn finish_stop(
    receipt: Receipt,
    persist: impl FnOnce(&Receipt) -> Result<(), String>,
    signal: impl FnOnce(&Receipt) -> Result<(), String>,
) -> Result<Receipt, StopFailure> {
    persist(&receipt).map_err(StopFailure::NotPersisted)?;
    signal(&receipt).map_err(|error| StopFailure::NotSignalled {
        receipt: Box::new(receipt.clone()),
        error,
    })?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sealed(drain_complete: bool) -> SealedAccounting {
        SealedAccounting {
            model_id: Some("glm-5.2".to_string()),
            prompt_tokens_total: 1_000,
            completion_tokens_total: 250,
            uptime_seconds: 3_600,
            drain_complete,
        }
    }

    /// A supervisor whose response was lost retries. With a fresh id
    /// per attempt the retry writes a SECOND receipt for one engine
    /// generation, which downstream is a second billing event for work
    /// that happened once.
    #[test]
    fn the_same_generation_always_derives_the_same_receipt_id() {
        assert_eq!(receipt_id("instance-7"), receipt_id("instance-7"));
        assert_ne!(receipt_id("instance-7"), receipt_id("instance-8"));

        let id = receipt_id("instance-7");
        assert_eq!(id.len(), 36, "UUID-shaped: {id}");
        let groups: Vec<usize> = id.split('-').map(str::len).collect();
        assert_eq!(groups, vec![8, 4, 4, 4, 12]);
        assert!(
            id.chars().all(|c| c == '-' || c.is_ascii_hexdigit()),
            "{id}"
        );
    }

    /// Version 8 and not 5, because this is SHA-256 and a v5 is
    /// SHA-1-based. Stamping a `5` on a digest that is not one would
    /// mislead anyone who later tried to re-derive the id.
    #[test]
    fn a_receipt_id_declares_the_version_it_really_is() {
        let id = receipt_id("anything");
        let version = id.split('-').nth(2).unwrap().chars().next().unwrap();
        assert_eq!(version, '8', "RFC 9562 custom version: {id}");

        let variant = id.split('-').nth(3).unwrap().chars().next().unwrap();
        assert!("89ab".contains(variant), "RFC 4122 variant bits: {id}");
    }

    /// A partial total accepted as a whole one undercounts silently,
    /// which is the failure nobody notices. Demotion happens at the
    /// boundary so every consumer sees the qualification, rather than
    /// being left to consumers of which the first to forget bills it
    /// wrong.
    #[test]
    fn a_drain_that_did_not_finish_is_demoted_rather_than_accepted() {
        assert_eq!(
            Receipt::from_sealed("i", &sealed(true)).status,
            ReceiptStatus::Complete
        );
        let degraded = Receipt::from_sealed("i", &sealed(false));
        assert_eq!(degraded.status, ReceiptStatus::Degraded);
        assert_eq!(
            degraded.completion_tokens_total, 250,
            "the totals are still reported -- as a lower bound, which is \
             what `degraded` says"
        );
    }

    /// The ordering the module exists for. A crash between the signal
    /// and the write loses a whole engine generation's totals: the
    /// engine is gone and the only record of what it served went with
    /// it.
    #[test]
    fn a_receipt_that_could_not_be_persisted_signals_nothing() {
        let mut signalled = false;
        let result = finish_stop(
            Receipt::from_sealed("i", &sealed(true)),
            |_| Err("disk full".to_string()),
            |_| {
                signalled = true;
                Ok(())
            },
        );
        assert_eq!(
            result,
            Err(StopFailure::NotPersisted("disk full".to_string()))
        );
        assert!(
            !signalled,
            "the signal must not be reachable past a failed write"
        );
    }

    /// The other failure, and it is materially different: the record
    /// survives, so a retry is safe and cannot double-count, and an
    /// operator must not be sent looking for a lost receipt that is
    /// sitting on disk.
    #[test]
    fn a_persisted_receipt_that_could_not_be_signalled_says_it_is_durable() {
        let receipt = Receipt::from_sealed("i", &sealed(true));
        let err = finish_stop(receipt.clone(), |_| Ok(()), |_| Err("no pipe".to_string()))
            .expect_err("the signal failed");
        assert!(err.to_string().contains("durable"), "{err}");
        match &err {
            StopFailure::NotSignalled { receipt: r, .. } => assert_eq!(**r, receipt),
            other => panic!("expected NotSignalled, got {other:?}"),
        }
    }

    #[test]
    fn a_stop_that_persisted_and_signalled_returns_the_receipt() {
        let order = std::cell::RefCell::new(Vec::new());
        let receipt = finish_stop(
            Receipt::from_sealed("i", &sealed(true)),
            |_| {
                order.borrow_mut().push("persist");
                Ok(())
            },
            |_| {
                order.borrow_mut().push("signal");
                Ok(())
            },
        )
        .expect("both steps succeed");
        assert_eq!(
            order.into_inner(),
            vec!["persist", "signal"],
            "durable first, always"
        );
        assert_eq!(receipt.model_id.as_deref(), Some("glm-5.2"));
    }
}
