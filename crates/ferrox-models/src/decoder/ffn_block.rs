//! The FFN half of one decoder layer for ONE row, written once.
//!
//! `forward_token`'s CPU arm, its Metal-attention arm and
//! `forward_token_paged` each spelled out the same six lines -- norm,
//! gpt-oss or generic FFN, `post_ffn_norm`, residual add -- and the
//! per-layer seam (`crate::layer_shapes`) needed a seventh fact in all
//! three: an FFN-free layer (`deci.cpp:147-149`) runs none of it. Three
//! copies of a branch is how features go missing from one path; this is
//! the one body, and the callers pass the one thing that differs.

use ferrox_core::matmul::rms_norm;

use super::{Decoder, GptOssLayer, LayerWeights};
use crate::scalar_multipliers::residual_add;

impl Decoder {
    /// Runs layer `layer_idx`'s FFN on `hidden` and adds it back, or
    /// does nothing for a layer whose shape has no FFN.
    ///
    /// `hidden` is the post-attention residual; on return it is the
    /// layer's output.
    pub(crate) fn ffn_block_row(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        hidden: &mut [f32],
        oai: Option<&GptOssLayer>,
        plan: Option<&ferrox_moe::PlacementPlan>,
    ) {
        if self.config.layer_shape(layer_idx).ffn_dim == 0 {
            return;
        }
        let hidden_dim = self.config.hidden_dim;
        let normed2 = layer
            .moe
            .norm_weight
            .apply(hidden, self.config.rms_norm_eps);
        let mut ffn_out = match oai {
            Some(oai) => Self::gpt_oss_ffn(layer, oai, &normed2, &self.config, hidden_dim),
            None => Self::run_ffn_block(layer_idx, layer, &normed2, &self.config, hidden_dim, plan),
        };
        if let Some(post) = &layer.attn.post_ffn_norm {
            ffn_out = rms_norm(&ffn_out, post, self.config.rms_norm_eps);
        }
        residual_add(hidden, &ffn_out, self.config.residual_scale);
    }
}
