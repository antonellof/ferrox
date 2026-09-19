//! The refusal for the hybrid rows no generic-path seam serves yet.
//!
//! Every hybrid graph llama.cpp has -- LFM2's short convolution, the
//! Mamba-1 and Mamba-2 blocks, Qwen3.5's gated delta net -- is served on
//! the generic decoder as a block where attention would be
//! (`crate::layer_shapes::AttnShape`, `crate::ssm_block`). What is left
//! refuses through [`HybridEngine::reject`] from
//! [`crate::engine_factory`], and each such row's catalog reason names
//! what it needs: `qwen3next`'s grouped head map and legacy fused
//! `ssm_in` / `ssm_ba`, `qwen35moe`'s MoE half, `plamo2`'s own Mamba-1
//! spelling. There is no separate hybrid engine and there is not going
//! to be one: the state rides on the layer's KV cache
//! (`frink_core::recurrent_state`).

use thiserror::Error;

#[derive(Debug, Error)]
#[error("hybrid engine not implemented for architecture {arch}")]
pub struct HybridUnavailable {
    pub arch: String,
}

pub struct HybridEngine;

impl HybridEngine {
    /// Fail-closed entry used by the engine factory for the hybrid rows
    /// still off the generic path.
    pub fn reject(arch: &str) -> Result<(), HybridUnavailable> {
        Err(HybridUnavailable {
            arch: arch.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_is_fail_closed() {
        assert!(HybridEngine::reject("plamo2").is_err());
    }
}
