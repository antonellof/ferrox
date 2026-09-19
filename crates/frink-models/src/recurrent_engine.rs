//! Recurrent engine stub (Mamba / RWKV).
//!
//! Fail-closed at load via [`crate::capability`] until a real state-machine
//! serve path lands. This module defines the engine shape so future work
//! has a single place to grow without touching GQA kernels.

use thiserror::Error;

#[derive(Debug, Error)]
#[error("recurrent engine not implemented for architecture {arch}")]
pub struct RecurrentUnavailable {
    pub arch: String,
}

/// Placeholder for Mamba/RWKV serve. Calling any method returns
/// [`RecurrentUnavailable`].
pub struct RecurrentEngine {
    pub arch: String,
}

impl RecurrentEngine {
    pub fn reject(arch: &str) -> Result<(), RecurrentUnavailable> {
        Err(RecurrentUnavailable {
            arch: arch.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_is_fail_closed() {
        assert!(RecurrentEngine::reject("mamba2").is_err());
    }
}
