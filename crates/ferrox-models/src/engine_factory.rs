//! Load-time engine selection (llama.cpp `llama_model_*` factory analogue).
//!
//! GGUF text-generation architectures that the generic [`crate::decoder::Decoder`]
//! can run are routed there. Dedicated stacks (MLA / DSA / recurrent /
//! T5 encoder-decoder) are selected here and must not accumulate
//! `if arch` branches inside matmul / attention kernels.

use crate::capability::{resolve_profile, ArchPath, DecoderFamily};
use crate::decoder::Decoder;
use crate::engine::{Engine, Glm52Engine, KimiEngine, MlaEngine};
use crate::gemma4_engine::Gemma4Engine;
use crate::gemma4_gguf_loader;
use crate::glm52_gguf_loader;
use crate::loader::LoadError;
use crate::mla_gguf_loader;
use thiserror::Error;

/// Why a GGUF cannot be served by the currently compiled engine set.
#[derive(Debug, Error)]
pub enum EngineSelectError {
    #[error("architecture {0:?} is outside the text-generation scope ({1})")]
    OutOfScope(String, &'static str),
    #[error("architecture {0:?} requires a dedicated engine that is not wired for serve yet: {1}")]
    DedicatedUnavailable(String, &'static str),
    #[error("unknown or unsupported architecture {0:?}")]
    Unknown(String),
}

/// Result of load-time engine selection for a GGUF architecture string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedEngineKind {
    /// Standard / Phi / Gemma / Qwen3 GQA path via [`Decoder`].
    GenericDecoder,
    /// Kimi / GLM / DeepSeek dedicated stacks (separate loaders).
    DedicatedStack,
    /// MLA memory engines (DeepSeek2 / Mistral4) — fail-closed for generic.
    Mla,
    /// Gemma-4 dedicated text engine (per-layer emb + shared KV + SWA split).
    Gemma4,
    /// Recurrent / hybrid SSM families — fail-closed until wired.
    RecurrentHybrid,
    /// T5 encoder-decoder — fail-closed until wired.
    EncoderDecoder,
}

/// Resolve which engine kind a GGUF `general.architecture` should use.
pub fn select_engine_kind(arch: &str) -> Result<SelectedEngineKind, EngineSelectError> {
    let profile =
        resolve_profile(arch).ok_or_else(|| EngineSelectError::Unknown(arch.to_string()))?;
    match profile.path {
        ArchPath::GenericGqa { .. } | ArchPath::TestFixture { .. } => {
            Ok(SelectedEngineKind::GenericDecoder)
        }
        ArchPath::Deferred { reason } => {
            Err(EngineSelectError::OutOfScope(arch.to_string(), reason))
        }
        ArchPath::DedicatedOnly { reason } => match profile.family {
            DecoderFamily::Dedicated => Ok(SelectedEngineKind::DedicatedStack),
            DecoderFamily::Mla => Ok(SelectedEngineKind::Mla),
            DecoderFamily::GemmaFamily if is_gemma4_arch(arch) => Ok(SelectedEngineKind::Gemma4),
            DecoderFamily::Hybrid | DecoderFamily::Recurrent => {
                Ok(SelectedEngineKind::RecurrentHybrid)
            }
            DecoderFamily::EncoderDecoder => Ok(SelectedEngineKind::EncoderDecoder),
            _ => Err(EngineSelectError::DedicatedUnavailable(
                arch.to_string(),
                reason,
            )),
        },
    }
}

/// Fail-closed check used by `ferrox-server` before constructing a
/// [`Decoder`] for architectures that need another engine.
pub fn ensure_generic_decoder(arch: &str) -> Result<(), EngineSelectError> {
    match select_engine_kind(arch)? {
        SelectedEngineKind::GenericDecoder => Ok(()),
        SelectedEngineKind::DedicatedStack => Err(EngineSelectError::DedicatedUnavailable(
            arch.to_string(),
            if is_glm52_arch(arch) {
                "use load_glm52_engine_from_path / ServedEngine::Glm52 — not generic Decoder"
            } else {
                "use the dedicated Kimi/GLM/DeepSeek loader, not generic Decoder"
            },
        )),
        SelectedEngineKind::Mla => Err(EngineSelectError::DedicatedUnavailable(
            arch.to_string(),
            "use load_mla_engine_from_path / ServedEngine::Mla — not generic Decoder",
        )),
        SelectedEngineKind::Gemma4 => Err(EngineSelectError::DedicatedUnavailable(
            arch.to_string(),
            "use load_gemma4_engine_from_path / ServedEngine::Gemma4 — not generic Decoder",
        )),
        SelectedEngineKind::RecurrentHybrid => {
            let _ = crate::recurrent_engine::RecurrentEngine::reject(arch);
            let _ = crate::hybrid_engine::HybridEngine::reject(arch);
            Err(EngineSelectError::DedicatedUnavailable(
                arch.to_string(),
                "recurrent/hybrid SSM engine stub present — not yet on the serve path",
            ))
        }
        SelectedEngineKind::EncoderDecoder => {
            let _ = crate::t5_engine::T5Engine::reject(arch);
            Err(EngineSelectError::DedicatedUnavailable(
                arch.to_string(),
                "T5 encoder-decoder engine stub present — not yet on the serve path",
            ))
        }
    }
}

/// Type-erased serve handle: today ordinary GGUFs use [`Decoder`];
/// Kimi/GLM/MLA use dedicated engines once loaders succeed.
pub enum ServedEngine {
    Decoder(Box<Decoder>),
    Kimi(KimiEngine),
    Glm52(Glm52Engine),
    Mla(MlaEngine),
    // Boxed: the Gemma-4 engine is ~200 bytes larger than any other
    // variant, so inlining it would pad every `ServedEngine` to its size.
    Gemma4(Box<Gemma4Engine>),
}

impl ServedEngine {
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Decoder(d) => Engine::vocab_size(d.as_ref()),
            Self::Kimi(k) => Engine::vocab_size(k),
            Self::Glm52(g) => Engine::vocab_size(g),
            Self::Mla(m) => Engine::vocab_size(m),
            Self::Gemma4(g) => Engine::vocab_size(g.as_ref()),
        }
    }
}

/// Open a DeepSeek-2 / Mistral-4 GGUF and build [`ServedEngine::Mla`].
pub fn load_mla_engine_from_path(path: &std::path::Path) -> Result<ServedEngine, LoadError> {
    let file = ferrox_gguf::ShardedGguf::open(path)?;
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown");
    match select_engine_kind(arch) {
        Ok(SelectedEngineKind::Mla) => {}
        Ok(other) => {
            return Err(LoadError::DedicatedArchitectureRequired(
                arch.to_string(),
                match other {
                    SelectedEngineKind::GenericDecoder => "generic decoder arch, not MLA",
                    SelectedEngineKind::DedicatedStack => "dedicated non-MLA stack",
                    SelectedEngineKind::RecurrentHybrid => "hybrid/recurrent, not MLA",
                    SelectedEngineKind::EncoderDecoder => "encoder-decoder, not MLA",
                    SelectedEngineKind::Gemma4 => "gemma4 dedicated, not MLA",
                    SelectedEngineKind::Mla => unreachable!(),
                },
            ));
        }
        Err(EngineSelectError::Unknown(a)) => {
            return Err(LoadError::UnsupportedArchitecture(a));
        }
        Err(EngineSelectError::OutOfScope(a, r)) => {
            return Err(LoadError::UnsupportedFeature(a, r.to_string()));
        }
        Err(EngineSelectError::DedicatedUnavailable(a, r)) => {
            return Err(LoadError::DedicatedArchitectureRequired(a, r));
        }
    }
    Ok(ServedEngine::Mla(mla_gguf_loader::load_mla_engine(&file)?))
}

/// Neither `glm4moe` nor `glm4` is here: GLM-4.5 / 4.5-Air / 4.6 and
/// GLM-4-0414 are plain GQA and run on the generic decoder
/// (`tests/glm4moe_graphs.rs`, `tests/glm4_graphs.rs`); both used to be
/// sent to this loader for MLA keys their graphs never read.
fn is_glm52_arch(arch: &str) -> bool {
    arch == "glm-dsa"
}

fn is_gemma4_arch(arch: &str) -> bool {
    crate::gemma4_engine::GEMMA4_ARCHES.contains(&arch)
}

/// Open a GLM-5.2 / GLM4-family GGUF and build [`ServedEngine::Glm52`].
pub fn load_glm52_engine_from_path(path: &std::path::Path) -> Result<ServedEngine, LoadError> {
    let file = ferrox_gguf::ShardedGguf::open(path)?;
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown");
    if !is_glm52_arch(arch) {
        return Err(LoadError::DedicatedArchitectureRequired(
            arch.to_string(),
            "not a GLM-5.2 architecture (expected glm-dsa)",
        ));
    }
    match select_engine_kind(arch) {
        Ok(SelectedEngineKind::DedicatedStack) => {}
        Ok(other) => {
            return Err(LoadError::DedicatedArchitectureRequired(
                arch.to_string(),
                match other {
                    SelectedEngineKind::GenericDecoder => "generic decoder arch, not GLM DSA",
                    SelectedEngineKind::Mla => "MLA arch, not GLM DSA",
                    SelectedEngineKind::RecurrentHybrid => "hybrid/recurrent, not GLM DSA",
                    SelectedEngineKind::EncoderDecoder => "encoder-decoder, not GLM DSA",
                    SelectedEngineKind::Gemma4 => "gemma4 dedicated, not GLM DSA",
                    SelectedEngineKind::DedicatedStack => unreachable!(),
                },
            ));
        }
        Err(EngineSelectError::Unknown(a)) => {
            return Err(LoadError::UnsupportedArchitecture(a));
        }
        Err(EngineSelectError::OutOfScope(a, r)) => {
            return Err(LoadError::UnsupportedFeature(a, r.to_string()));
        }
        Err(EngineSelectError::DedicatedUnavailable(a, r)) => {
            return Err(LoadError::DedicatedArchitectureRequired(a, r));
        }
    }
    Ok(ServedEngine::Glm52(glm52_gguf_loader::load_glm52_engine(
        &file,
    )?))
}

/// Open a Gemma-4 GGUF and build [`ServedEngine::Gemma4`].
pub fn load_gemma4_engine_from_path(path: &std::path::Path) -> Result<ServedEngine, LoadError> {
    let file = ferrox_gguf::ShardedGguf::open(path)?;
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown");
    if !is_gemma4_arch(arch) {
        return Err(LoadError::DedicatedArchitectureRequired(
            arch.to_string(),
            "not a gemma4 / gemma4-assistant architecture",
        ));
    }
    match select_engine_kind(arch) {
        Ok(SelectedEngineKind::Gemma4) => {}
        Ok(other) => {
            return Err(LoadError::DedicatedArchitectureRequired(
                arch.to_string(),
                match other {
                    SelectedEngineKind::GenericDecoder => "generic decoder arch, not Gemma4",
                    SelectedEngineKind::Mla => "MLA arch, not Gemma4",
                    SelectedEngineKind::DedicatedStack => "dedicated non-Gemma4 stack",
                    SelectedEngineKind::RecurrentHybrid => "hybrid/recurrent, not Gemma4",
                    SelectedEngineKind::EncoderDecoder => "encoder-decoder, not Gemma4",
                    SelectedEngineKind::Gemma4 => unreachable!(),
                },
            ));
        }
        Err(EngineSelectError::Unknown(a)) => {
            return Err(LoadError::UnsupportedArchitecture(a));
        }
        Err(EngineSelectError::OutOfScope(a, r)) => {
            return Err(LoadError::UnsupportedFeature(a, r.to_string()));
        }
        Err(EngineSelectError::DedicatedUnavailable(a, r)) => {
            return Err(LoadError::DedicatedArchitectureRequired(a, r));
        }
    }
    Ok(ServedEngine::Gemma4(Box::new(
        gemma4_gguf_loader::load_gemma4_engine(&file)?,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_and_qwen3_select_generic_decoder() {
        assert_eq!(
            select_engine_kind("llama").unwrap(),
            SelectedEngineKind::GenericDecoder
        );
        assert_eq!(
            select_engine_kind("qwen3").unwrap(),
            SelectedEngineKind::GenericDecoder
        );
        assert_eq!(
            select_engine_kind("gemma3").unwrap(),
            SelectedEngineKind::GenericDecoder
        );
        assert_eq!(
            select_engine_kind("phi3").unwrap(),
            SelectedEngineKind::GenericDecoder
        );
        // `mixtral` was HERE and is not a generic-decoder row any
        // more: no converter writes that string, libllama refuses it,
        // and every real Mixtral checkpoint declares `llama` (which is
        // the first case in this test). See `capability::NO_UPSTREAM_ARCH`.
        assert_eq!(
            select_engine_kind("olmoe").unwrap(),
            SelectedEngineKind::GenericDecoder
        );
    }

    #[test]
    fn mamba_is_recurrent_fail_closed_for_generic() {
        assert!(matches!(
            select_engine_kind("mamba2").unwrap(),
            SelectedEngineKind::RecurrentHybrid
        ));
        assert!(ensure_generic_decoder("mamba2").is_err());
    }

    #[test]
    fn deepseek2_is_mla_not_generic() {
        assert_eq!(
            select_engine_kind("deepseek2").unwrap(),
            SelectedEngineKind::Mla
        );
        assert!(ensure_generic_decoder("deepseek2").is_err());
    }

    #[test]
    fn glm4_is_generic_and_glm_dsa_is_the_dedicated_stack() {
        assert_eq!(
            select_engine_kind("glm4").unwrap(),
            SelectedEngineKind::GenericDecoder
        );
        assert!(ensure_generic_decoder("glm4").is_ok());
        assert_eq!(
            select_engine_kind("glm-dsa").unwrap(),
            SelectedEngineKind::DedicatedStack
        );
    }

    #[test]
    fn gemma4_selects_dedicated_engine() {
        assert_eq!(
            select_engine_kind("gemma4").unwrap(),
            SelectedEngineKind::Gemma4
        );
        assert_eq!(
            select_engine_kind("gemma4-assistant").unwrap(),
            SelectedEngineKind::Gemma4
        );
        assert!(ensure_generic_decoder("gemma4").is_err());
    }
}
