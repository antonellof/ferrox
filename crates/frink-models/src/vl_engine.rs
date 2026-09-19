//! Multimodal / VL serve stub (Qwen2-VL / CogVLM / Gemma4-VL / …).
//!
//! Vision primitives for Kimi MoonViT live in [`crate::vision`]; GGUF VL
//! architectures remain `DeferredMultimodal` in the capability registry
//! until a projection + chat-template path is wired.
//!
//! Expected pairing (P7): main text GGUF + companion `mmproj*.gguf` beside
//! it — discovered via [`crate::mmproj::find_mmproj_beside`]. A future
//! [`VlProjectorPair`] will hold both paths; generation stays fail-closed
//! until projector weights + image tokenization land.

use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
#[error("multimodal/VL engine not implemented for architecture {arch}")]
pub struct VlUnavailable {
    pub arch: String,
}

/// Main checkpoint + mmproj companion (not loaded yet).
#[derive(Debug, Clone)]
pub struct VlProjectorPair {
    pub arch: String,
    pub main_gguf: PathBuf,
    pub mmproj_gguf: PathBuf,
}

impl VlProjectorPair {
    /// Document expected pairing when mmproj is found beside a main GGUF.
    pub fn from_paths(arch: &str, main_gguf: &Path, mmproj_gguf: PathBuf) -> Self {
        Self {
            arch: arch.to_string(),
            main_gguf: main_gguf.to_path_buf(),
            mmproj_gguf,
        }
    }
}

pub struct VlEngine {
    pub arch: String,
    /// When set, records the mmproj path we expect to wire later.
    pub projector: Option<VlProjectorPair>,
}

impl VlEngine {
    pub fn reject(arch: &str) -> Result<(), VlUnavailable> {
        Err(VlUnavailable {
            arch: arch.to_string(),
        })
    }

    /// Fail-closed generate entry — projector pairing may be recorded for logs.
    pub fn reject_with_mmproj(arch: &str, pair: VlProjectorPair) -> Result<(), VlUnavailable> {
        let _ = pair;
        Self::reject(arch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_is_fail_closed() {
        assert!(VlEngine::reject("qwen2vl").is_err());
    }

    #[test]
    fn projector_pair_records_paths() {
        let main = Path::new("/tmp/model.gguf");
        let mm = PathBuf::from("/tmp/mmproj-f16.gguf");
        let pair = VlProjectorPair::from_paths("qwen2vl", main, mm.clone());
        assert_eq!(pair.mmproj_gguf, mm);
        assert!(VlEngine::reject_with_mmproj("qwen2vl", pair).is_err());
    }
}
