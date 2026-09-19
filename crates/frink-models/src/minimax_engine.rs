//! MiniMax M2 / M3 refusal, and why the reason it used to give was the
//! wrong one.
//!
//! This module used to say MiniMax "needs a loader, 256-expert sigmoid
//! MoE routing and MTP draft heads, none of which exist yet". Checked
//! against llama.cpp, two of those three clauses are false and the third
//! is only true of one of the two architectures:
//!
//! - **MTP does not exist in either model.** Neither
//!   `.scratch/llama.cpp/src/models/minimax-m2.cpp` nor `minimax-m3.cpp`
//!   creates a single `nextn.*` tensor, and `gguf-py`'s
//!   `MODEL_ARCH.MINIMAXM2` / `.MINIMAXM3` tensor lists contain no
//!   `NEXTN_*` entry, so no converter can emit MTP weights for these
//!   files. `minimax-m3.cpp:9` states it: "MTP is not in released model
//!   weights." A refusal naming a tensor family the checkpoint cannot
//!   contain is unreachable by construction — glm4moe's `q_lora_rank`
//!   defect in a different costume.
//! - **Sigmoid MoE routing already exists here.** `loader.rs` reads
//!   `{arch}.expert_gating_func` into `GatingFunction::Sigmoid`, loads
//!   `blk.N.exp_probs_b.bias`, and reads `expert_weights_scale` and
//!   `expert_weights_norm`. `frink_moe::route_top_k_sigmoid` is the
//!   routing DeepSeek-V3 and GLM-4-MoE use. "256 experts" is an hparam,
//!   not a ceiling.
//! - **Block-sparse attention is M3's, not MiniMax's.** `minimax-m2.cpp`
//!   builds ordinary dense GQA attention (:112). The ported selection in
//!   [`frink_core::block_sparse`] is relevant only to M3's MSA, and is
//!   the smallest piece of it.
//!
//! What is actually true, per architecture, is in [`crate::capability`]
//! — and that is deliberately the ONLY copy. The live refusal a user
//! hits is `LoadError::DedicatedArchitectureRequired`, built from
//! `ArchPath::DedicatedOnly { reason }` in `loader.rs`; this module's
//! [`MinimaxEngine::reject`] used to carry a second, longer, *different*
//! reason that nothing ever printed. Two copies of one explanation is
//! how they came to disagree, so `reject` now reads the catalog's string
//! rather than repeating it.
//!
//! In short: `minimax-m2` was UNAUDITED and is audited since 2026-09-14
//! (`tests/minimax_m2_graphs.rs`, a libllama golden on the fixture that
//! had evidenced the claim), while `minimax-m3` is genuinely
//! UNIMPLEMENTED (the MSA indexer and its own KV cache). This module
//! refuses `minimax-m3` and nothing else.

use thiserror::Error;

#[derive(Debug, Error)]
#[error("MiniMax dedicated engine not implemented: {reason}")]
pub struct MinimaxUnavailable {
    pub reason: &'static str,
}

pub struct MinimaxEngine {
    pub arch: String,
}

impl MinimaxEngine {
    /// Refuses, with the SAME reason the loader gives for `arch`.
    ///
    /// Deliberately delegating: the catalog owns the per-architecture
    /// reason, so this cannot drift away from what a user actually sees.
    pub fn reject(arch: &str) -> Result<(), MinimaxUnavailable> {
        let reason = match crate::capability::resolve_architecture(arch) {
            Some(crate::capability::ArchPath::DedicatedOnly { reason }) => reason,
            // Any other path for a name routed here is itself the bug:
            // say so rather than inventing an architecture reason.
            _ => {
                return Err(MinimaxUnavailable {
                    reason: "not a MiniMax architecture this shim refuses: `minimax-m3` is the \
                             one the capability catalog marks DedicatedOnly (`minimax-m2` is on \
                             the generic path)",
                });
            }
        };
        Err(MinimaxUnavailable { reason })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_is_fail_closed() {
        assert!(MinimaxEngine::reject("minimax-m3").is_err());
        // `minimax-m2` is generic now; the shim names that as the reason
        // rather than inventing an architecture one.
        let err = MinimaxEngine::reject("minimax-m2").unwrap_err();
        assert!(err.reason.contains("not a MiniMax architecture"));
    }

    /// The defect this module was fixed for: the engine's reason and the
    /// loader's reason were two different strings, and only the loader's
    /// was ever shown.
    #[test]
    fn reject_repeats_the_catalog_reason_verbatim() {
        let Some(crate::capability::ArchPath::DedicatedOnly { reason }) =
            crate::capability::resolve_architecture("minimax-m3")
        else {
            panic!("minimax-m3 must be DedicatedOnly");
        };
        let err = MinimaxEngine::reject("minimax-m3").unwrap_err();
        assert_eq!(
            err.reason, reason,
            "minimax-m3: the engine must not carry a second, different reason"
        );
    }

    /// Neither reason may claim MTP again: no MiniMax GGUF can carry
    /// `nextn.*` tensors, because `gguf-py`'s MINIMAXM2/MINIMAXM3 tensor
    /// lists have no NEXTN entry and neither `minimax-m*.cpp` creates
    /// one.
    #[test]
    fn no_minimax_reason_blames_mtp() {
        let err = MinimaxEngine::reject("minimax-m3").unwrap_err();
        let lower = err.reason.to_ascii_lowercase();
        assert!(
            !lower.contains("mtp") && !lower.contains("nextn"),
            "minimax-m3 may not be refused for MTP it cannot have: {}",
            err.reason
        );
    }

    #[test]
    fn a_non_minimax_name_is_named_as_such() {
        let err = MinimaxEngine::reject("llama").unwrap_err();
        assert!(err.reason.contains("not a MiniMax architecture"));
    }
}
