//! **THE SAME PHYSICAL LAYERS RUN MORE THAN ONCE** -- Nanbeige's
//! `num_loops`, as one value a `ModelConfig` carries and one rule for
//! which logical layer runs which weights and where the loop norm sits.
//!
//! # What it is
//!
//! `src/models/nanbeige.cpp:6-12` read `{arch}.num_loops` (default 1)
//! and `{arch}.skip_loop_final_norm` (default false). With `n_loops >
//! 1`, `:19-31` set `n_layer_all = n_layer_phys * n_loops` and COPY the
//! per-layer head / kv-head / ff / swa arrays from physical layer `i`
//! to every logical slot `i + j * n_phys`; `:47-66` create tensors for
//! the `n_phys` physical layers only and `:69-73` alias
//! `layers[i + j * n_phys] = layers[i]`. So the graph (`:94-176`) walks
//! `n_layer_all` logical layers, each with its OWN KV cache and its own
//! row in every per-layer table, over `n_phys` sets of weights. After
//! the last logical layer of every pass but the final one (`:167-175`:
//! `(il + 1) % n_phys == 0 && (il + 1) < n_layer`), the running
//! residual is normed with `output_norm` -- the lm_head's norm, the
//! same tensor -- unless `skip_loop_final_norm`. Everything inside a
//! pass is plain Llama.
//!
//! # Reach -- MEASURED
//!
//! `grep -l 'n_loops\|n_layer_phys' src/models/*.cpp` over all 140
//! graphs (2026-09-12) is `nanbeige.cpp`; `LLM_KV_NUM_LOOPS` and
//! `LLM_KV_SKIP_LOOP_FINAL_NORM` are read nowhere else. So
//! [`LOOP_READERS`] has one row and the keys are dead metadata on every
//! other architecture, as upstream (the `yarn_log_multiplier` rule,
//! `crate::yarn_magnitude`).
//!
//! # What ferrox does with it
//!
//! The weights are shared and the KV is not, and the seam says exactly
//! that rather than copying weights: `Decoder::layers` stays the
//! PHYSICAL vector the loader filled, `ModelConfig::n_layers` is the
//! LOGICAL count (so every KV cache, per-layer table and budget is
//! sized per logical layer, as `n_layer_all` sizes them upstream), and
//! `Decoder::layer_for(l)` is the ONE mapping from a logical index to
//! its weights, `l % n_phys`. The three host bodies iterate logical
//! indices and ask it; `LayerLoops::loop_norm_after(l)` says where the
//! loop norm goes and `Decoder::final_norm` is the tensor that goes
//! there. The per-layer shape arrays are replicated at load, as
//! `:24-26` replicate them.
//!
//! Every fused Metal launch indexes `layers[l]` and `metal_kvs[l]` with
//! one `l`, so `metal_can_serve_model` refuses a looped model; the host
//! bodies serve it. A LoRA adapter attaches to physical layers by
//! tensor name and so reaches every logical layer that shares them,
//! which is what `build_lora_mm` on an aliased `layers[il]` does too.

use crate::LoadError;
use ferrox_gguf::TensorSource;

/// Architectures whose `load_arch_hparams` reads `num_loops`, with the
/// line.
pub const LOOP_READERS: &[(&str, &str)] = &[("nanbeige", "src/models/nanbeige.cpp:6-31,167-175")];

/// A model whose logical layers are `n_loops` passes over `n_phys`
/// physical ones. Only ever constructed with `n_loops >= 2`: a file
/// declaring 1 (or nothing) is a plain model and carries `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerLoops {
    /// Physical layers: the blocks the file holds tensors for.
    pub n_phys: usize,
    /// Passes over them; `>= 2`.
    pub n_loops: usize,
    /// `{arch}.skip_loop_final_norm`: no norm between passes.
    pub skip_loop_final_norm: bool,
}

impl LayerLoops {
    /// `n_phys * n_loops`: llama.cpp's `n_layer_all`, the count every
    /// per-layer thing is sized by.
    pub fn logical_layers(&self) -> usize {
        self.n_phys * self.n_loops
    }

    /// The physical layer logical layer `l` runs.
    pub fn physical(&self, l: usize) -> usize {
        l % self.n_phys
    }

    /// Whether `output_norm` is applied to the residual AFTER logical
    /// layer `l` (`nanbeige.cpp:167-175`): the last layer of every pass
    /// but the final one, unless the file skips it.
    pub fn loop_norm_after(&self, l: usize) -> bool {
        !self.skip_loop_final_norm
            && (l + 1).is_multiple_of(self.n_phys)
            && l + 1 < self.logical_layers()
    }
}

/// Reads the two keys for a file: `Some` only for an architecture whose
/// graph loops AND a `num_loops` above 1.
pub fn read_layer_loops(
    file: &impl TensorSource,
    arch: &str,
    n_phys: usize,
) -> Result<Option<LayerLoops>, LoadError> {
    if !LOOP_READERS.iter().any(|(name, _)| *name == arch) {
        return Ok(None);
    }
    let key = |k: &str| format!("{arch}.{k}");
    let n_loops = file.metadata_u64(&key("num_loops")).unwrap_or(1) as usize;
    if n_loops == 0 {
        // `GGML_ASSERT(n_loops_u >= 1)`, nanbeige.cpp:8.
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            "num_loops is 0; llama.cpp asserts it is at least 1".to_string(),
        ));
    }
    if n_loops == 1 {
        return Ok(None);
    }
    let skip_loop_final_norm = file
        .metadata_bool(&key("skip_loop_final_norm"))
        .unwrap_or(false);
    Ok(Some(LayerLoops {
        n_phys,
        n_loops,
        skip_loop_final_norm,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two physical layers run twice: logical 0,1,2,3 are physical
    /// 0,1,0,1; the loop norm sits after logical 1 and NOT after
    /// logical 3, which the final norm follows.
    #[test]
    fn the_schedule_and_the_loop_norm_follow_nanbeige_cpp() {
        let loops = LayerLoops {
            n_phys: 2,
            n_loops: 2,
            skip_loop_final_norm: false,
        };
        assert_eq!(loops.logical_layers(), 4);
        assert_eq!(
            (0..4).map(|l| loops.physical(l)).collect::<Vec<_>>(),
            [0, 1, 0, 1]
        );
        assert_eq!(
            (0..4).map(|l| loops.loop_norm_after(l)).collect::<Vec<_>>(),
            [false, true, false, false]
        );
        let skipped = LayerLoops {
            skip_loop_final_norm: true,
            ..loops
        };
        assert!((0..4).all(|l| !skipped.loop_norm_after(l)));
        // Three passes over three layers: norms after 2 and 5, not 8.
        let three = LayerLoops {
            n_phys: 3,
            n_loops: 3,
            skip_loop_final_norm: false,
        };
        assert_eq!(
            (0..9)
                .filter(|&l| three.loop_norm_after(l))
                .collect::<Vec<_>>(),
            [2, 5]
        );
    }

    #[test]
    fn every_reader_is_an_audited_generic_row() {
        for (arch, line) in LOOP_READERS {
            let profile = crate::capability::resolve_profile(arch)
                .unwrap_or_else(|| panic!("`{arch}` ({line}) is not a registered architecture"));
            assert!(matches!(
                profile.path,
                crate::capability::ArchPath::GenericGqa { .. }
            ));
            assert!(crate::capability::AUDITED_GENERIC_GQA.contains(arch));
        }
    }
}
