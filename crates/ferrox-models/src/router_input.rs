//! **WHICH TENSOR THE MoE ROUTER READS** -- the operand of
//! `ffn_gate_inp`, as one value a `ModelConfig` carries and one table
//! that says which architecture reads what.
//!
//! # What it is
//!
//! llama.cpp's `build_moe_ffn` (`llama-graph.cpp:1914-1948`) computes
//! the router logits itself, `logits = gate_inp · cur`, from the SAME
//! `cur` the experts then read -- the normed FFN input -- UNLESS the
//! caller hands it a precomputed `probs_in`, in which case `gate_inp`
//! is unused and the caller decided the operand. Every ferrox MoE body
//! computed `router · normed2`, which is the default and right for
//! every graph that takes it.
//!
//! # Who passes `probs_in` -- MEASURED, not read off one file
//!
//! Every `build_moe_ffn(` call in all 140 `src/models/*.cpp` was parsed
//! for its `gate_inp` and `probs_in` arguments (2026-09-11). Fifty-nine
//! call sites; four pass a precomputed `probs_in`:
//!
//! | arch | router operand | why precomputed | engine here | line |
//! |---|---|---|---|---|
//! | `smallthinker` | `inpL` -- the RAW LAYER INPUT, before `attn_norm`, before attention | the operand is different | generic GQA | `smallthinker.cpp:111,151-161` |
//! | `grovemoe` | `cur` -- the normed FFN input, the default | shared between TWO `build_moe_ffn` calls (the expert bank and the chunk-expert bank) | generic GQA, refused for the second bank | `grovemoe.cpp:133,137-148,153-164` |
//! | `gemma4` | `rms_norm(attn_out) * (1/sqrt(n_embd)) * ffn_gate_inp_s` -- the attention output, its own norm, a scale tensor | the operand is different | its own engine (`gemma4_engine`) | `gemma4.cpp:289-294` |
//! | `nemotron-h` | `cur` -- the FFN input BEFORE the latent down-projection the experts read | the experts read `inp_latent`, the router does not | hybrid recurrent engine | `nemotron-h.cpp:210-232` |
//!
//! Two more route on something other than a variable named `cur` and
//! are the default anyway: `llama4.cpp:221` passes `ffn_inp_normed`
//! (the normed FFN input) and `cohere2moe.cpp:234,389` pass `ffn_inp`
//! (the parallel-residual topology's one normed input, which its
//! experts read too). Fifty-three sites pass `nullptr` or the 13-arg
//! overload and route on `cur`.
//!
//! So `smallthinker` is the ONLY generic-path graph whose router
//! operand is not what the experts read, and [`RouterInput`] has two
//! variants rather than four: `gemma4`'s and `nemotron-h`'s shapes
//! live on engines that do not read this field, and a variant with no
//! caller is the OLMo lesson (`capability::WEIGHTED_LAYER_NORM`).
//! `grovemoe` shares the mechanism (a precomputed `probs`) and NOT the
//! cause; that is why the table is keyed by what the router reads and
//! not by whether `probs_in` is non-null.
//!
//! # What `inpL` is, exactly
//!
//! `smallthinker.cpp:86` sets `inpL = build_inp_embd(...)` and `:172`
//! sets `inpL = cur` at the bottom of every layer, so at layer `il` it
//! is the residual stream as it ENTERS the layer: the scaled embedding
//! row at layer 0, the previous layer's output after both residual
//! adds otherwise. `:111` reads it BEFORE `:115` norms it for
//! attention, so the router sees no norm at all. ferrox captures it at
//! the same point (`Decoder::router_operand`, called where the row's
//! `attn_norm` is applied) and computes the logits there, in the same
//! order llama.cpp does, so the operand cannot be the post-attention
//! residual by mistake.
//!
//! Everything downstream of the logits is the ordinary
//! `build_moe_ffn` (`:151-161`): `expert_gating_func` from the file
//! (`conversion/smallthinker.py:27-30` writes SOFTMAX or SIGMOID),
//! `norm_w = true` as a literal, `expert_weights_scale` unset
//! (skipped at 0), no `exp_probs_b`, no shared expert, no groups.
//! `Decoder::route_for_layer` already implements all of that.
//!
//! # Where it is served, and where it refuses
//!
//! The CPU row body and both batched host bodies take the operand
//! from `Decoder::router_operand`, ONE function, and hand it to the
//! ONE FFN body per shape (`decoder/ffn_block.rs`). Every Metal path
//! that runs the router on the GPU reads `normed2` and nothing else,
//! so `Decoder::gpu_router_matches_host_routing` -- the predicate all
//! of them already share -- answers false for [`RouterInput::
//! RawLayerInput`], and those launches fall back to the host bodies
//! rather than routing on the wrong tensor.

/// The operand of the MoE router's matmul.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouterInput {
    /// `gate_inp · ffn_norm(ffn_inp)` -- the normed FFN input, the
    /// same vector the experts read. `build_moe_ffn`'s own
    /// computation, and every generic-path graph but one.
    #[default]
    NormedFfnInput,
    /// `gate_inp · inpL` -- the residual stream as it enters the
    /// layer, unnormed, before attention (`smallthinker.cpp:111`). The
    /// experts still read the normed FFN input.
    RawLayerInput,
}

/// Which operand each architecture's router reads. The table behind
/// the census above, restricted to the generic path; the two rows on
/// other engines are documented there and not here, because nothing
/// on those engines asks this question.
pub const ROUTER_INPUT_TABLE: &[(&str, RouterInput, &str)] = &[(
    "smallthinker",
    RouterInput::RawLayerInput,
    "src/models/smallthinker.cpp:111,151-161",
)];

/// The router operand for an architecture: the table's entry, or the
/// default for every architecture the table does not name.
pub fn router_input(arch: &str) -> RouterInput {
    ROUTER_INPUT_TABLE
        .iter()
        .find(|(name, _, _)| *name == arch)
        .map(|(_, input, _)| *input)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one row, and the default everywhere else -- including the
    /// three other graphs that pass a precomputed `probs_in`, none of
    /// which is this shape (`grovemoe`) or on this engine (`gemma4`,
    /// `nemotron-h`).
    #[test]
    fn only_smallthinker_routes_on_the_raw_layer_input() {
        assert_eq!(router_input("smallthinker"), RouterInput::RawLayerInput);
        for arch in [
            "llama",
            "qwen3moe",
            "olmoe",
            "deepseek",
            "grovemoe",
            "gemma4",
            "nemotron-h",
            "llama4",
            "cohere2moe",
        ] {
            assert_eq!(router_input(arch), RouterInput::NormedFfnInput, "{arch}");
        }
        assert_eq!(RouterInput::default(), RouterInput::NormedFfnInput);
    }

    /// Every table row is an architecture the generic loader can
    /// reach, so the seam it names is a seam something asks.
    #[test]
    fn every_table_row_is_on_the_generic_path() {
        for (arch, _, line) in ROUTER_INPUT_TABLE {
            let profile = crate::capability::resolve_profile(arch)
                .unwrap_or_else(|| panic!("`{arch}` ({line}) is not a registered architecture"));
            assert!(
                matches!(profile.path, crate::capability::ArchPath::GenericGqa { .. }),
                "`{arch}` ({line}) is {:?}, and only the generic decoder reads this table",
                profile.path
            );
            assert!(
                crate::capability::AUDITED_GENERIC_GQA.contains(arch),
                "`{arch}` is served here and must be audited, or the seam is unevidenced"
            );
        }
    }
}
