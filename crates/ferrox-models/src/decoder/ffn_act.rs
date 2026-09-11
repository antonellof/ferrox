//! The one place `ModelConfig::ffn_activation` becomes the gated
//! activation the expert FFN actually runs.
//!
//! It is a separate module for one reason: this mapping used to be
//! written out only in `run_ffn_block`'s DENSE arm, and every routed
//! path hardcoded SwiGLU. That is not a drift between copies, it is a
//! decision that was made in one place and never made in the others, and
//! the only thing keeping it from producing wrong logits was that
//! `loader.rs` hands out [`FfnActivation::Gelu`] for
//! `DecoderFamily::GemmaFamily` alone and every GemmaFamily row on
//! `ArchPath::GenericGqa` is dense.
//!
//! With one conversion and a [`GluAct`] argument the routed paths cannot
//! refuse to pass, there is nothing left to forget: adding a third
//! activation is a non-exhaustive `match` here and a compile error at
//! every call site, not a silent SwiGLU.

use crate::config::FfnActivation;
use ferrox_moe::GluAct;

impl From<FfnActivation> for GluAct {
    fn from(a: FfnActivation) -> Self {
        match a {
            // `SwigluFused` is the same activation as `Swiglu`; it only
            // says gate and up arrive as one on-disk tensor (Phi), which
            // the loader has already split by the time a `WeightMatrix`
            // exists.
            FfnActivation::Swiglu | FfnActivation::SwigluFused => GluAct::Swiglu,
            FfnActivation::Gelu => GluAct::Geglu,
            // Ungated on disk, gated in the enum: the loader aliases
            // gate to up. See `FfnActivation::ReluSqr`.
            FfnActivation::ReluSqr => GluAct::Reglu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mapping itself, pinned. If a later edit makes `Gelu` map to
    /// `Swiglu` -- which is exactly what every routed path did before
    /// this existed -- this goes red without needing a model.
    #[test]
    fn gelu_maps_to_geglu_and_never_to_swiglu() {
        assert_eq!(GluAct::from(FfnActivation::Gelu), GluAct::Geglu);
        assert_eq!(GluAct::from(FfnActivation::Swiglu), GluAct::Swiglu);
        assert_eq!(GluAct::from(FfnActivation::SwigluFused), GluAct::Swiglu);
        assert_eq!(GluAct::from(FfnActivation::ReluSqr), GluAct::Reglu);
    }
}
