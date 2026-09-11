//! Load-time execution / memory plans (llama.cpp graph-params analogue).
//!
//! Selected once per model (and cached by batch geometry for decode vs
//! prefill). Hot-path forward never re-derives architecture strings or
//! fused-op availability.

use crate::capability::{DecoderFamily, MemoryKind, QkNormStyle};
use crate::config::{FfnActivation, RopeLayout};

/// Backend fused-op availability discovered at load / first probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FusedOpCaps {
    pub metal_flash_attn: bool,
    pub metal_swiglu: bool,
    pub metal_matvec: bool,
    /// CUDA is deferred — kept for ABI stability, always false for now.
    pub cuda_gqa: bool,
}

/// Memory layout chosen once from the architecture profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPlan {
    pub kind: MemoryKind,
    pub swa_layers: crate::swa_layers::SwaLayers,
    pub sliding_window: Option<usize>,
}

/// Per-model execution plan: everything the forward path needs that is
/// constant across tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionPlan {
    pub family: DecoderFamily,
    pub rope: RopeLayout,
    pub qk_norm: QkNormStyle,
    pub ffn_activation: FfnActivation,
    pub memory: MemoryPlan,
    pub fused: FusedOpCaps,
    pub embedding_scale: Option<f32>,
    pub attention_scale: Option<f32>,
    pub rope_theta_swa: Option<f32>,
    pub attn_logit_softcap: Option<f32>,
    pub final_logit_softcap: Option<f32>,
}

/// Cache key for decode/prefill plan reuse (batch geometry only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanGeometry {
    pub n_tokens: usize,
    pub n_seqs: usize,
    pub flash_attn: bool,
}

impl ExecutionPlan {
    /// Build from a resolved [`crate::config::ModelConfig`] + profile.
    pub fn from_config(
        config: &crate::config::ModelConfig,
        family: DecoderFamily,
        memory_kind: MemoryKind,
        fused: FusedOpCaps,
    ) -> Self {
        Self {
            family,
            rope: config.rope_layout,
            qk_norm: config.qk_norm_style,
            ffn_activation: config.ffn_activation,
            memory: MemoryPlan {
                kind: memory_kind,
                swa_layers: config.swa_layers.clone(),
                sliding_window: config.sliding_window,
            },
            fused,
            embedding_scale: config.embedding_scale,
            attention_scale: config.attention_scale,
            rope_theta_swa: config.rope_theta_swa,
            attn_logit_softcap: config.attn_logit_softcap,
            final_logit_softcap: config.final_logit_softcap,
        }
    }

    /// Probe Metal fused-op availability without changing the Llama
    /// default path. Returns conservative caps when Metal is off.
    pub fn probe_metal_caps() -> FusedOpCaps {
        #[cfg(feature = "metal")]
        {
            let metal_on = ferrox_core::metal_dense_enabled();
            FusedOpCaps {
                metal_flash_attn: metal_on && ferrox_metal::attn::metal_attn_enabled(),
                metal_swiglu: metal_on,
                metal_matvec: metal_on,
                cuda_gqa: false,
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            FusedOpCaps::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_dense_fixture;

    #[test]
    fn plan_from_tiny_config_defaults() {
        let cfg = test_dense_fixture();
        let plan = ExecutionPlan::from_config(
            &cfg,
            DecoderFamily::StandardGqa,
            MemoryKind::KvGqa,
            FusedOpCaps::default(),
        );
        assert_eq!(plan.family, DecoderFamily::StandardGqa);
        assert_eq!(plan.qk_norm, QkNormStyle::WholeVector);
        assert!(!plan.fused.cuda_gqa);
    }
}
