//! Semantic tensor roles mirroring llama.cpp `llm_tensor` / `LLM_TN`.
//!
//! Architecture loaders resolve roles to concrete GGUF names once at
//! load time. Hot-path kernels never see string names.

/// Logical weight / activation tensor identity inside a decoder block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    TokenEmbd,
    Output,
    OutputNorm,
    AttnNorm,
    AttnQ,
    AttnK,
    AttnV,
    AttnQkv,
    AttnOut,
    AttnQNorm,
    AttnKNorm,
    AttnPostNorm,
    FfnNorm,
    FfnGate,
    FfnUp,
    FfnDown,
    FfnPostNorm,
    FfnGateInp,
    FfnGateExps,
    FfnUpExps,
    FfnDownExps,
    RopeFreqs,
}

impl TensorRole {
    /// Default GGUF tensor name for this role (layer-scoped when `layer`
    /// is `Some`). Matches llama.cpp `LLM_TENSOR_NAMES` conventions.
    pub fn gguf_name(self, layer: Option<usize>) -> String {
        match (self, layer) {
            (Self::TokenEmbd, _) => "token_embd.weight".into(),
            (Self::Output, _) => "output.weight".into(),
            (Self::OutputNorm, _) => "output_norm.weight".into(),
            (Self::RopeFreqs, _) => "rope_freqs.weight".into(),
            (Self::AttnNorm, Some(i)) => format!("blk.{i}.attn_norm.weight"),
            (Self::AttnQ, Some(i)) => format!("blk.{i}.attn_q.weight"),
            (Self::AttnK, Some(i)) => format!("blk.{i}.attn_k.weight"),
            (Self::AttnV, Some(i)) => format!("blk.{i}.attn_v.weight"),
            (Self::AttnQkv, Some(i)) => format!("blk.{i}.attn_qkv.weight"),
            (Self::AttnOut, Some(i)) => format!("blk.{i}.attn_output.weight"),
            (Self::AttnQNorm, Some(i)) => format!("blk.{i}.attn_q_norm.weight"),
            (Self::AttnKNorm, Some(i)) => format!("blk.{i}.attn_k_norm.weight"),
            (Self::AttnPostNorm, Some(i)) => format!("blk.{i}.post_attention_norm.weight"),
            (Self::FfnNorm, Some(i)) => format!("blk.{i}.ffn_norm.weight"),
            (Self::FfnGate, Some(i)) => format!("blk.{i}.ffn_gate.weight"),
            (Self::FfnUp, Some(i)) => format!("blk.{i}.ffn_up.weight"),
            (Self::FfnDown, Some(i)) => format!("blk.{i}.ffn_down.weight"),
            (Self::FfnPostNorm, Some(i)) => format!("blk.{i}.post_ffw_norm.weight"),
            (Self::FfnGateInp, Some(i)) => format!("blk.{i}.ffn_gate_inp.weight"),
            (Self::FfnGateExps, Some(i)) => format!("blk.{i}.ffn_gate_exps.weight"),
            (Self::FfnUpExps, Some(i)) => format!("blk.{i}.ffn_up_exps.weight"),
            (Self::FfnDownExps, Some(i)) => format!("blk.{i}.ffn_down_exps.weight"),
            (role, None) => panic!("{role:?} requires a layer index"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_scoped_names_match_llama_convention() {
        assert_eq!(TensorRole::AttnQ.gguf_name(Some(3)), "blk.3.attn_q.weight");
        assert_eq!(
            TensorRole::AttnQkv.gguf_name(Some(0)),
            "blk.0.attn_qkv.weight"
        );
        assert_eq!(TensorRole::TokenEmbd.gguf_name(None), "token_embd.weight");
    }
}
