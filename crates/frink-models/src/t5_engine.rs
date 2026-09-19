//! T5 encoder-decoder engine stub.
//!
//! Fail-closed at load. Will own the encoder pass + cross-attention
//! decode loop when implemented.

use thiserror::Error;

#[derive(Debug, Error)]
#[error("T5 encoder-decoder engine not implemented for architecture {arch}")]
pub struct T5Unavailable {
    pub arch: String,
}

pub struct T5Engine {
    pub arch: String,
}

impl T5Engine {
    pub fn reject(arch: &str) -> Result<(), T5Unavailable> {
        Err(T5Unavailable {
            arch: arch.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_is_fail_closed() {
        assert!(T5Engine::reject("t5").is_err());
    }
}
