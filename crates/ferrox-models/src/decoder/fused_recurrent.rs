//! A recurrent decoder layer as ONE Metal submission, or nothing.
//!
//! [`crate::fused_layer`] says what the layer's WEIGHTS have to be.
//! This says what the MODEL has to be, and then runs it. The two halves
//! are separate because they fail for different reasons and a caller
//! has to answer both at one point, which is
//! [`Decoder::fused_recurrent_layer`].
//!
//! Why it exists at all is arithmetic, not a hunch: a Bonsai decode
//! token submits 192 command buffers and pays 0.15 ms of OS wake-up for
//! each, 35 ms of a 137 ms token, against GPU work that is already
//! faster than the reference's whole token. Three submissions a layer
//! become one here.

use super::{Decoder, LayerWeights};
use crate::fused_layer::LayerFfnParts;
use crate::layer_shapes::AttnShape;
use crate::ssm_block::SsmBlock;
use ferrox_core::recurrent_state::RecurrentState;

impl Decoder {
    /// Layer `l` end to end in one submission, or `None` when either
    /// half refuses and the host bodies run it.
    ///
    /// `normed` is `attn_norm(hidden)`, `hidden` the residual stream as
    /// it entered the layer. The returned vector is the stream as it
    /// leaves: the caller assigns it and moves to the next layer
    /// without an attention block or an FFN block of its own.
    #[cfg(feature = "metal")]
    pub(crate) fn fused_recurrent_layer(
        &self,
        l: usize,
        layer: &LayerWeights,
        normed: &[f32],
        hidden: &[f32],
        slot: &mut Option<RecurrentState>,
    ) -> Option<Vec<f32>> {
        // The MODEL's half. Each of these is applied by a host body and
        // by no kernel in the fused launch, so a model that has one
        // takes the host path -- not a launch that silently drops it.
        let shape = self.config.layer_shape(l);
        if !matches!(shape.attention, AttnShape::Gdn)
            // An FFN-free block keeps or discards its output by
            // architecture (`crate::layer_shapes`); neither is this.
            || shape.ffn_dim == 0
            // Granite's multipliers, applied to both branch outputs.
            || self.config.residual_scale.is_some()
            // Talkie's normed embedding added into every layer output.
            || self.config.skip_stream
            // gpt-oss's side table is a different FFN entirely.
            || self.gpt_oss.is_some()
            // Anything but SwiGLU has no kernel in this launch.
            || !self.config.layer_ffn_acts(l).all_swiglu()
        {
            return None;
        }
        let SsmBlock::Gdn(gdn) = layer.attn.ssm.as_ref()? else {
            return None;
        };
        let ffn = LayerFfnParts::for_layer(layer, self.config.rms_norm_eps, true)?;
        let state = slot.get_or_insert_with(|| gdn.zero_state());
        // The layer's own `attn_norm`, so the fused launch can compute
        // `normed` on device and the host copy above goes unused.
        let attn_norm = layer.attn.norm_weight.rms_weights()?;
        let out = gdn.fused_layer(
            attn_norm,
            normed,
            state,
            self.config.rms_norm_eps,
            &ffn,
            hidden,
        )?;
        // The host body records expert 0 for a dense layer every token,
        // so this does too, and only once the layer has actually run
        // here rather than fallen back.
        layer.moe.record_activations_dense();
        Some(out)
    }
}
