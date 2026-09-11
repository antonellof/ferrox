//! MoE layers that ALSO run a dense FFN and sum the two: the Grok-2
//! shape, refused by name.
//!
//! `src/models/grok.cpp:66-68` creates `ffn_gate` / `ffn_up` /
//! `ffn_down` as `TENSOR_NOT_REQUIRED` beside the routed experts, and
//! `:171-184` does this when they are present:
//!
//! ```text
//! if (model.layers[il].ffn_up) {
//!     ffn_out = build_ffn(cur, ffn_up, ffn_gate, ffn_down, LLM_FFN_GELU, LLM_FFN_PAR, il);
//!     cur = ggml_scale(ctx0, ggml_add(ctx0, ffn_out, moe_out), std::sqrt(2) / 2);
//! } else {
//!     cur = moe_out;
//! }
//! ```
//!
//! A dense GELU FFN over the SAME normed input as the router, added to
//! the routed output, and the sum scaled by `sqrt(2)/2` before the
//! post-FFN norm. That is Grok-2 (`conversion/grok.py` maps its
//! `model.layers.N.mlp.*` onto the dense names); Grok-1 has no dense
//! FFN and takes the `else` branch.
//!
//! ferrox has no slot for it. The nearest thing is the shared expert
//! (`MoeWeights::shared_experts`), which is also always-on and added to
//! the routed sum -- but a shared expert loads from the `*_shexp`
//! names, is SiLU on every architecture that has one, and nothing
//! scales the sum afterwards. Reading Grok-2's dense FFN into that slot
//! would drop the `sqrt(2)/2` and swap the activation, and answer
//! fluently. So a `grok` file carrying a dense `ffn_up` on a routed
//! layer STOPS here, and the row is admitted for Grok-1.
//!
//! The gate is reachable: `scripts/make_grok_fixture.py --dense-ffn`
//! writes exactly this file, and `tests/grok_graphs.rs` drives the
//! refusal from it.

use ferrox_gguf::TensorSource;

/// Architectures whose graph sums a dense FFN with the routed experts
/// when the dense tensors are present.
pub const PARALLEL_DENSE_FFN_ARCHITECTURES: &[&str] = &["grok"];

/// The refusal reason when `arch` would run a parallel dense FFN on
/// its routed layers and the file carries one, or `None` when the file
/// may be run.
///
/// Checks layer 0 only: `grok.cpp:54-79` creates the same optional set
/// for every layer, and a file with a dense FFN on some layers and not
/// others is not one any converter writes.
pub fn parallel_dense_refusal(arch: &str, file: &impl TensorSource) -> Option<String> {
    if !PARALLEL_DENSE_FFN_ARCHITECTURES.contains(&arch) {
        return None;
    }
    file.find_tensor("blk.0.ffn_up.weight")?;
    Some(format!(
        "`blk.N.ffn_up.weight` beside routed experts: src/models/grok.cpp:171-184 runs a \
         dense GELU FFN in parallel with the MoE on every layer and scales their sum by \
         sqrt(2)/2, which is the Grok-2 shape. ferrox has no slot for a dense FFN that is \
         summed with the experts -- the shared-expert slot has the wrong activation, the \
         wrong tensor names and no scale on the sum -- so it stops rather than loading the \
         file into the nearest thing. Grok-1, which has no dense FFN, is unaffected; \
         `{arch}` is admitted for it"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_gguf::{GgufError, GgufValue, TensorInfo};

    /// The smallest `TensorSource` that can say whether one tensor
    /// exists.
    struct Names(Vec<TensorInfo>);

    impl Names {
        fn of(names: &[&str]) -> Self {
            Self(
                names
                    .iter()
                    .map(|n| TensorInfo {
                        name: (*n).to_string(),
                        shape: Vec::new(),
                        dtype: ferrox_gguf::GgmlType::F32,
                        offset: 0,
                    })
                    .collect(),
            )
        }
    }

    impl TensorSource for Names {
        fn metadata(&self, _key: &str) -> Option<&GgufValue> {
            None
        }
        fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
            self.0.iter().find(|t| t.name == name)
        }
        fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
            // Never reached: the gate asks only whether a tensor exists.
            Err(GgufError::TensorNotFound(name.to_string()))
        }
        fn tensor_mapped_range(
            &self,
            name: &str,
        ) -> Result<
            (
                std::sync::Arc<ferrox_gguf::MmapHandle>,
                std::ops::Range<usize>,
            ),
            GgufError,
        > {
            Err(GgufError::TensorNotFound(name.to_string()))
        }
    }

    /// The gate fires on the Grok-2 tensor set and not on Grok-1's.
    #[test]
    fn a_dense_ffn_up_beside_experts_refuses_grok_and_its_absence_does_not() {
        let grok2 = Names::of(&["blk.0.ffn_up_exps.weight", "blk.0.ffn_up.weight"]);
        let msg = parallel_dense_refusal("grok", &grok2).expect("refused");
        assert!(msg.contains("grok.cpp:171-184"), "{msg}");
        assert!(msg.contains("sqrt(2)/2"), "{msg}");

        let grok1 = Names::of(&["blk.0.ffn_up_exps.weight"]);
        assert_eq!(parallel_dense_refusal("grok", &grok1), None);
    }

    /// Every other architecture with a dense `ffn_up` is a dense model
    /// or a leading-dense MoE, and neither is this gate's business.
    #[test]
    fn a_dense_ffn_on_any_other_architecture_is_not_this_gates_business() {
        let dense = Names::of(&["blk.0.ffn_up.weight"]);
        for arch in ["llama", "deepseek", "dbrx", "qwen3moe"] {
            assert_eq!(parallel_dense_refusal(arch, &dense), None, "{arch}");
        }
    }
}
