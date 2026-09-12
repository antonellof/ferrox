//! **THE PARALLEL RESIDUAL** -- `x + attn(norm(x)) + ffn(norm(x))`, the
//! layer shape the generic decoder does not have, and which of llama.cpp's
//! graphs build it.
//!
//! # What it is
//!
//! The generic layer is sequential: `h = x + attn(norm1(x))`, then
//! `h + ffn(norm2(h))`. The parallel shape feeds the FFN the LAYER INPUT
//! -- normed -- rather than the attention output, and sums the three
//! terms once. Two spellings exist upstream:
//!
//! - **One shared norm** (`SharedNorm`): the FFN reads the SAME normed
//!   tensor attention read. `stablelm.cpp:135-137` (`cur = inpSA` when
//!   `ffn_norm` is ABSENT), `phi2.cpp:67,108,116-117`,
//!   `falcon.cpp:124-135` (Falcon-7B, no `attn_norm_2`),
//!   `command-r.cpp:68,106-119`, `cohere2.cpp:120-134`,
//!   `cohere2moe.cpp:222-266`, `plamo.cpp:59-64,97-98,111-112` (`cur =
//!   sa_inp`, over an RMSNorm).
//! - **Two norms** (`TwoNorms`): `x + attn(ln1(x)) + ffn(ln2(x))`,
//!   `gptneox.cpp:143-166` (`use_parallel_residual`, read at `:5`) and
//!   `falcon.cpp:79-85` (Falcon-40B, `attn_norm_2`).
//!
//! # Reach -- MEASURED
//!
//! Over all 140 `src/models/*.cpp` (2026-09-12): `grep -l "par_res\|
//! parallel residual"` is `gptneox.cpp` and `stablelm.cpp`; a scan for
//! TWO consecutive `cur = ggml_add(ctx0, cur, ...)` lines -- the
//! three-term sum spelled out -- is `cohere2`, `cohere2moe` (twice, the
//! trunk and its MTP block), `command-r`, `falcon`, `phi2` and `plamo`,
//! with `gptneox` and `stablelm` separating their two adds by a `cb`
//! line. `gemma4.cpp:260` names an `attn_out` that is ALREADY `cur +
//! inpL`, so it is sequential and not in the table. Eight graphs, two
//! spellings, and `stablelm` is the one where BOTH shapes sit behind one
//! architecture string, decided by tensor presence. (`plamo` was missed
//! by a first grep that looked for `attn_out` by name; its attention
//! output is `sa_out`. The two-adds scan is the measurement.)
//!
//! # What this module does today
//!
//! Refuses, by name, from a fixture llama.cpp runs
//! (`tests/fixtures/stablelm_parallel_tiny.gguf`, libllama's logits
//! differ from the sequential file's by 8.85, measured). The
//! `stablelm` row is decided per layer, as `stablelm.cpp:129` decides
//! it: a layer with no `ffn_norm.weight` is a parallel layer. The
//! `use_parallel_residual` key the converter writes (`conversion/
//! stablelm.py`) is read by NOTHING in `stablelm.cpp` and is dead
//! metadata there -- libllama's logits with and without it are
//! byte-identical (measured, `tests/stablelm_graphs.rs`) -- so it is
//! not consulted here either. The seam that SERVES the shape is the
//! next PR on the plan (`docs/plans/README.md`, `b2-close-the-68`): the
//! two batched host bodies and the row body each add the FFN output to
//! the ATTENTION output's residual, and the parallel shape needs them
//! to add it to the layer input instead, in one place.

use ferrox_gguf::TensorSource;

/// Which normed input the FFN reads in a parallel layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelNorm {
    /// The FFN reads the tensor attention read (`attn_norm(x)`).
    SharedNorm,
    /// The FFN reads its own norm of the layer input (`ffn_norm(x)`).
    TwoNorms,
}

/// How a graph decides that a layer is parallel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelWhen {
    /// Every layer, unconditionally.
    Always,
    /// `blk.N.ffn_norm.weight` is absent (`stablelm.cpp:129`).
    FfnNormAbsent,
    /// `{arch}.use_parallel_residual` is true (`gptneox.cpp:5,143`).
    ParallelResidualKey,
    /// `blk.N.attn_norm_2.weight` is present (`falcon.cpp:79`); absent
    /// is the shared-norm shape.
    AttnNorm2Present,
}

/// One graph that builds the parallel residual.
#[derive(Debug, Clone, Copy)]
pub struct ParallelResidual {
    pub arch: &'static str,
    pub norm: ParallelNorm,
    pub when: ParallelWhen,
    pub lines: &'static str,
}

/// The eight graphs, with the lines. Only the `stablelm` row is on the
/// generic path today; the others are recorded so the seam that serves
/// the shape is sized from the table and not from one graph.
pub const PARALLEL_RESIDUAL_GRAPHS: &[ParallelResidual] = &[
    ParallelResidual {
        arch: "stablelm",
        norm: ParallelNorm::SharedNorm,
        when: ParallelWhen::FfnNormAbsent,
        lines: "src/models/stablelm.cpp:38-39,129-138,147",
    },
    ParallelResidual {
        arch: "gptneox",
        norm: ParallelNorm::TwoNorms,
        when: ParallelWhen::ParallelResidualKey,
        lines: "src/models/gptneox.cpp:5,143-166",
    },
    ParallelResidual {
        arch: "phi2",
        norm: ParallelNorm::SharedNorm,
        when: ParallelWhen::Always,
        lines: "src/models/phi2.cpp:67,108,116-117",
    },
    ParallelResidual {
        arch: "falcon",
        norm: ParallelNorm::TwoNorms,
        when: ParallelWhen::AttnNorm2Present,
        lines: "src/models/falcon.cpp:35-36,79-85,124-135",
    },
    ParallelResidual {
        arch: "command-r",
        norm: ParallelNorm::SharedNorm,
        when: ParallelWhen::Always,
        lines: "src/models/command-r.cpp:68,106-119",
    },
    ParallelResidual {
        arch: "cohere2",
        norm: ParallelNorm::SharedNorm,
        when: ParallelWhen::Always,
        lines: "src/models/cohere2.cpp:120-134",
    },
    ParallelResidual {
        arch: "cohere2moe",
        norm: ParallelNorm::SharedNorm,
        when: ParallelWhen::Always,
        lines: "src/models/cohere2moe.cpp:222-266",
    },
    ParallelResidual {
        arch: "plamo",
        norm: ParallelNorm::SharedNorm,
        when: ParallelWhen::Always,
        lines: "src/models/plamo.cpp:59-64,97-98,111-112",
    },
];

/// The row for an architecture, or `None` for a sequential graph.
pub fn parallel_residual(arch: &str) -> Option<&'static ParallelResidual> {
    PARALLEL_RESIDUAL_GRAPHS.iter().find(|row| row.arch == arch)
}

/// Whether layer `l` of `arch` in `file` is a parallel layer, by the
/// row's rule. `false` for a graph not in the table.
pub fn layer_is_parallel(file: &impl TensorSource, arch: &str, l: usize) -> bool {
    let Some(row) = parallel_residual(arch) else {
        return false;
    };
    match row.when {
        ParallelWhen::Always => true,
        ParallelWhen::FfnNormAbsent => file
            .find_tensor(&format!("blk.{l}.ffn_norm.weight"))
            .is_none(),
        ParallelWhen::ParallelResidualKey => file
            .metadata_bool(&format!("{arch}.use_parallel_residual"))
            .unwrap_or(false),
        ParallelWhen::AttnNorm2Present => file
            .find_tensor(&format!("blk.{l}.attn_norm_2.weight"))
            .is_some(),
    }
}

/// The refusal reason for a file whose trunk has a parallel layer, or
/// `None` when every layer is sequential. Names the first parallel
/// layer and the rule that decided it.
pub fn parallel_residual_refusal(
    file: &impl TensorSource,
    arch: &str,
    n_layers: usize,
) -> Option<String> {
    let row = parallel_residual(arch)?;
    let l = (0..n_layers).find(|&l| layer_is_parallel(file, arch, l))?;
    let decided_by = match row.when {
        ParallelWhen::Always => "every layer of this graph".to_string(),
        ParallelWhen::FfnNormAbsent => format!("`blk.{l}.ffn_norm.weight` is absent"),
        ParallelWhen::ParallelResidualKey => {
            format!("`{arch}.use_parallel_residual` is true")
        }
        ParallelWhen::AttnNorm2Present => format!("`blk.{l}.attn_norm_2.weight` is present"),
    };
    let reads = match row.norm {
        ParallelNorm::SharedNorm => "the normed input attention read",
        ParallelNorm::TwoNorms => "its own norm of the layer input",
    };
    Some(format!(
        "layer {l} is a PARALLEL residual, `x + attn(norm(x)) + ffn(norm(x))`: {decided_by}, \
         so llama.cpp feeds the FFN {reads} and sums the three terms once ({}). The generic \
         decoder adds the FFN output to the attention output's residual on every layer, which \
         is a different graph; libllama's logits for this shape differ from the sequential \
         file's by 8.85 (measured, tests/fixtures/stablelm_parallel_tiny.gguf), so it stops \
         rather than run the sequential one (`ferrox_models::parallel_residual`)",
        row.lines
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row names a real architecture string llama.cpp has, and the
    /// generic-path row is the only one the loader can reach; the rest
    /// are refused or deferred before any tensor is read, which is what
    /// makes their `when` a recorded fact rather than a live rule.
    #[test]
    fn every_row_is_a_registered_architecture_and_only_stablelm_is_generic() {
        for row in PARALLEL_RESIDUAL_GRAPHS {
            assert!(
                crate::capability::resolve_profile(row.arch).is_some(),
                "`{}` ({}) is not a registered architecture",
                row.arch,
                row.lines
            );
            let generic = matches!(
                crate::capability::resolve_architecture(row.arch),
                Some(crate::capability::ArchPath::GenericGqa { .. })
            );
            assert_eq!(
                generic,
                row.arch == "stablelm",
                "`{}`: a second generic-path row means the seam must serve it, not refuse",
                row.arch
            );
        }
    }

    /// The rows are distinct, so a graph cannot be given two rules.
    #[test]
    fn no_architecture_has_two_rows() {
        let mut names: Vec<&str> = PARALLEL_RESIDUAL_GRAPHS.iter().map(|r| r.arch).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), PARALLEL_RESIDUAL_GRAPHS.len());
        assert_eq!(PARALLEL_RESIDUAL_GRAPHS.len(), 8, "the measured reach");
    }

    /// A graph not in the table is sequential on every layer, whatever
    /// its tensors say.
    #[test]
    fn a_sequential_graph_is_never_parallel() {
        let file = crate::test_source::StubSource::with_tensors(&[]);
        assert!(!layer_is_parallel(&file, "llama", 0));
        assert!(parallel_residual_refusal(&file, "llama", 4).is_none());
    }

    /// The `stablelm` rule: parallel exactly when the layer's
    /// `ffn_norm.weight` is missing, per layer, and the refusal names the
    /// first such layer.
    #[test]
    fn stablelm_is_decided_by_ffn_norm_presence_per_layer() {
        use crate::test_source::StubSource;
        let sequential =
            StubSource::with_tensors(&["blk.0.ffn_norm.weight", "blk.1.ffn_norm.weight"]);
        assert!(!layer_is_parallel(&sequential, "stablelm", 0));
        assert!(!layer_is_parallel(&sequential, "stablelm", 1));
        assert!(parallel_residual_refusal(&sequential, "stablelm", 2).is_none());

        let mixed = StubSource::with_tensors(&["blk.0.ffn_norm.weight"]);
        assert!(!layer_is_parallel(&mixed, "stablelm", 0));
        assert!(layer_is_parallel(&mixed, "stablelm", 1));
        let reason = parallel_residual_refusal(&mixed, "stablelm", 2).expect("refused");
        assert!(
            reason.contains("layer 1 is a PARALLEL residual"),
            "{reason}"
        );
        assert!(
            reason.contains("`blk.1.ffn_norm.weight` is absent"),
            "{reason}"
        );
        assert!(
            reason.contains("stablelm.cpp:38-39,129-138,147"),
            "{reason}"
        );

        // The trunk length bounds the scan: a parallel block past it is
        // not this loader's layer.
        assert!(parallel_residual_refusal(&mixed, "stablelm", 1).is_none());
    }

    /// The other three rules, on the rows that carry them, so the
    /// table's `when` column is exercised and not only recorded.
    #[test]
    fn the_key_the_second_norm_and_the_unconditional_rules() {
        use crate::test_source::StubSource;
        use ferrox_gguf::GgufValue;
        let neox_seq = StubSource::with_tensors(&["blk.0.ffn_norm.weight"])
            .with_key("gptneox.use_parallel_residual", GgufValue::Bool(false));
        assert!(!layer_is_parallel(&neox_seq, "gptneox", 0));
        let neox_par = StubSource::with_tensors(&["blk.0.ffn_norm.weight"])
            .with_key("gptneox.use_parallel_residual", GgufValue::Bool(true));
        assert!(layer_is_parallel(&neox_par, "gptneox", 0));
        assert!(parallel_residual_refusal(&neox_par, "gptneox", 1)
            .expect("refused")
            .contains("`gptneox.use_parallel_residual` is true"));

        let falcon_7b = StubSource::with_tensors(&["blk.0.attn_norm.weight"]);
        assert!(!layer_is_parallel(&falcon_7b, "falcon", 0));
        let falcon_40b = StubSource::with_tensors(&["blk.0.attn_norm_2.weight"]);
        assert!(layer_is_parallel(&falcon_40b, "falcon", 0));

        let phi2 = StubSource::with_tensors(&["blk.0.ffn_norm.weight"]);
        assert!(layer_is_parallel(&phi2, "phi2", 0));
    }
}
