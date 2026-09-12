//! **THE MLA ENGINE'S PER-ARCHITECTURE DECISIONS** -- one table for the
//! three places `deepseek2`, `mistral4` and `plm` differ in what
//! `mla_gguf_loader` builds, so the loader reads a row instead of
//! restating an `if arch ==` chain at each site.
//!
//! # What differs, with the lines
//!
//! | | `deepseek2` / `mistral4` | `plm` |
//! |---|---|---|
//! | Q projection | `attn_q_a` / `attn_q_b` when `q_lora_rank > 0`, direct `attn_q` for the LITE layer counts (`deepseek2.cpp:8,11-13,104-115`) | direct `attn_q`, no key (`plm.cpp:32`) |
//! | dense FFN | gated SwiGLU (`deepseek2.cpp:129-131`, `build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR)`) | ungated `LLM_FFN_RELU_SQR` over `ffn_up` / `ffn_down` (`plm.cpp:39-40,181-187`) |
//! | lm_head | `output`, else `tok_embd` (`deepseek2.cpp:92-96`) | `tok_embd` DUPLICATED, no `output` read (`plm.cpp:23-24`) |
//!
//! `mistral4` has no loader of its own (`mistral4.cpp` is one
//! `build_arch_graph` line; its hparams and tensors are `deepseek2`'s),
//! so it is `deepseek2`'s row under a second name.
//!
//! What does NOT differ and so is not a column: `attn_kv_a_mqa` /
//! `attn_kv_a_norm` / `attn_kv_b` / `attn_output` and the attention
//! graph between them (`plm.cpp:84-166` is `deepseek2.cpp`'s naive
//! branch line for line), NORM RoPE on the `pe` slices
//! (`llama-model.cpp:2588-2592` lists both), `kq_scale =
//! 1/sqrt(n_embd_head_k)` with no YaRN term (`plm.cpp:50`;
//! `deepseek2.cpp:312-319` has one, which is why a scaled `deepseek2`
//! is refused in the loader).
//!
//! # The lite rule
//!
//! `deepseek2.cpp:8` decides `is_lite` from the LAYER COUNT -- 27, 26,
//! or 48 with a 128256-token vocabulary ("DeepSeek-V2-Lite,
//! GigaChat3-10B-A1.8B, Kanana-2-30B-A3B") -- and `:11-13` read
//! `attention.q_lora_rank` ONLY when it is false, so a lite file's key
//! is dead metadata whatever it says and `n_lora_q` stays 0, which
//! `:104-115` turn into a direct `attn_q`. Copied here in that order,
//! because the alternative reading -- "direct iff the key is absent or
//! zero" -- agrees with upstream on every converter-written file and
//! disagrees on a hand-written lite file carrying the key, which is
//! exactly the file a test would use to tell the two apart.

use ferrox_moe::GluAct;

/// How the row projects Q; the type it produces is
/// `crate::mla_q_proj::MlaQProj`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QProjRule {
    /// `attention.q_lora_rank` is REQUIRED and low-rank, unless the
    /// layer count is a lite one, in which case the key is not read
    /// and Q is direct (`deepseek2.cpp:8,11-13`).
    LoraUnlessLite,
    /// Always direct; the key is never read (`plm.cpp:32`).
    Direct,
}

/// What the row does with `output.weight`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlaOutputHead {
    /// `output.weight` if present, else `token_embd.weight`
    /// (`deepseek2.cpp:92-96`, `TENSOR_NOT_REQUIRED` then `DUPLICATED`).
    OutputOrTied,
    /// `token_embd.weight` always; an `output.weight` in the file is
    /// one llama.cpp never creates and its loader REFUSES the file for
    /// (`llama-model-loader.cpp:1309-1313`, "wrong number of tensors";
    /// measured on `tests/fixtures/plm_decoy_output_tiny.gguf`).
    TiedOnly,
}

/// One architecture the MLA engine serves.
#[derive(Debug, Clone, Copy)]
pub struct MlaArch {
    pub name: &'static str,
    pub q_proj: QProjRule,
    /// The dense layers' activation. The ungated variants alias the
    /// gate to `ffn_up` at load, as the generic loader does for
    /// `arcee`, so `ferrox_moe::run_expert` computes them with no
    /// second matmul (`GluAct::ungated`).
    pub dense_act: GluAct,
    pub output_head: MlaOutputHead,
    /// Where the row's three decisions are read from.
    pub lines: &'static str,
}

/// Every architecture `mla_gguf_loader` accepts, with its decisions.
pub const MLA_ENGINE_ARCHS: &[MlaArch] = &[
    MlaArch {
        name: "deepseek2",
        q_proj: QProjRule::LoraUnlessLite,
        dense_act: GluAct::Swiglu,
        output_head: MlaOutputHead::OutputOrTied,
        lines: "src/models/deepseek2.cpp:8,11-13,92-96,104-115,129-131",
    },
    MlaArch {
        name: "mistral4",
        q_proj: QProjRule::LoraUnlessLite,
        dense_act: GluAct::Swiglu,
        output_head: MlaOutputHead::OutputOrTied,
        lines: "src/models/mistral4.cpp (deepseek2's loader and graph)",
    },
    MlaArch {
        name: "plm",
        q_proj: QProjRule::Direct,
        dense_act: GluAct::ReluSqr,
        output_head: MlaOutputHead::TiedOnly,
        lines: "src/models/plm.cpp:23-24,32,39-40,181-187",
    },
];

/// The row for an architecture, or `None` for one this engine does not
/// serve.
pub fn mla_arch(arch: &str) -> Option<&'static MlaArch> {
    MLA_ENGINE_ARCHS.iter().find(|a| a.name == arch)
}

/// `deepseek2.cpp:8`: the lite checkpoints, by trunk layer count and
/// vocabulary.
pub fn is_lite(n_layer: usize, n_vocab: usize) -> bool {
    n_layer == 27 || n_layer == 26 || (n_layer == 48 && n_vocab == 128256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lite_rule_is_deepseek2_cpp_line_8() {
        assert!(is_lite(27, 102400));
        assert!(is_lite(26, 1));
        assert!(is_lite(48, 128256));
        assert!(
            !is_lite(48, 129280),
            "DeepSeek-V3 is 61 layers; a 48-layer file with V3's vocab is not lite"
        );
        assert!(!is_lite(60, 102400), "DeepSeek-V2 236B");
        assert!(!is_lite(3, 48), "the fixtures");
    }

    #[test]
    fn every_row_is_a_dedicated_mla_profile() {
        for row in MLA_ENGINE_ARCHS {
            let profile = crate::capability::resolve_profile(row.name).unwrap_or_else(|| {
                panic!(
                    "`{}` ({}) is not a registered architecture",
                    row.name, row.lines
                )
            });
            assert_eq!(
                profile.family,
                crate::capability::DecoderFamily::Mla,
                "{}",
                row.name
            );
            assert!(
                matches!(
                    profile.path,
                    crate::capability::ArchPath::DedicatedOnly { .. }
                ),
                "{}: the MLA engine is reached through DedicatedOnly, not the generic path",
                row.name
            );
        }
        assert!(mla_arch("deepseek32").is_none(), "DSA is its own stack");
        assert!(mla_arch("llama").is_none());
    }

    /// `mistral4` IS `deepseek2` upstream; a column that drifted between
    /// the two rows would describe a graph that does not exist.
    #[test]
    fn mistral4_is_deepseek2s_row_under_a_second_name() {
        let ds = mla_arch("deepseek2").unwrap();
        let m4 = mla_arch("mistral4").unwrap();
        assert_eq!(ds.q_proj, m4.q_proj);
        assert_eq!(ds.output_head, m4.output_head);
        assert!(matches!(
            (ds.dense_act, m4.dense_act),
            (GluAct::Swiglu, GluAct::Swiglu)
        ));
    }
}
