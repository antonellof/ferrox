//! WHICH ACTIVATION EACH LAYER RUNS, with WHICH PARAMETERS -- the
//! per-layer half of `ModelConfig::ffn_activation`.
//!
//! llama.cpp has two graphs whose FFN activation takes scalars that
//! vary by layer, and both read them the same way: `get_key_or_arr` at
//! `n_layer()` length, an array exactly that long or one scalar
//! broadcast to every layer (`llama-model-loader.cpp:455-478`).
//!
//! * `apertus.cpp:6-9` reads FOUR arrays, `xielu.alpha_n`,
//!   `xielu.alpha_p`, `xielu.beta`, `xielu.eps` (no architecture
//!   prefix -- `llama-arch.cpp:370-373`), all REQUIRED, and `:132-138`
//!   builds `ggml_xielu(up, alpha_n[il], alpha_p[il], beta[il],
//!   eps[il])` for layer `il`. The activation is xIELU, UNGATED.
//! * `step35.cpp:28-29` reads TWO optional arrays,
//!   `{arch}.swiglu_clamp_exp` and `{arch}.swiglu_clamp_shexp`, and
//!   the generic `build_moe_ffn` / `build_ffn` (`llama-graph.cpp:2146-
//!   2164`, `:1751-1768`) clamp SwiGLU by layer `il`'s entry when it
//!   is above `1e-6`. The activation is SwiGLU with one scalar; the
//!   routed experts read one array and the shared experts AND the
//!   leading dense layers read the other, because `build_ffn` is both.
//!
//! So the two are ONE plumbing question -- "layer `il` runs its FFN
//! activation with these scalars" -- and TWO activation bodies. This
//! module is the plumbing: [`XieluLayers`] holds one parameter set per
//! trunk layer, [`read_xielu_layers`] reads the four keys the way
//! `get_key_or_arr` does, and [`ModelConfig::layer_ffn_act`] is the
//! ONE accessor every FFN body asks. The uniform case (every
//! architecture but these) is the special case where every layer
//! answers the same parameter-free [`GluAct`].
//!
//! The two bodies are `ferrox_moe::GluAct::Xielu` (caller: `apertus`)
//! and `ferrox_moe::GluAct::SwigluClamped` (caller: `step35`), and the
//! one thing the second needed of the plumbing that the first did not
//! is the SITE: llama.cpp's `build_moe_ffn` reads one array and its
//! `build_ffn` the other, so [`ModelConfig::layer_ffn_acts`] answers a
//! [`LayerFfnActs`] pair -- `routed` for the top-k experts, `dense` for
//! the dense layers and the shared experts -- and every FFN body names
//! the field it runs. For every other activation the two fields are
//! the same value.

use std::sync::Arc;

use ferrox_gguf::{GgufValue, TensorSource};
use ferrox_moe::{GluAct, XieluParams};

use crate::config::{FfnActivation, ModelConfig};
use crate::loader::LoadError;

/// One xIELU parameter set per TRUNK layer, already folded the way
/// `ggml_xielu` folds them ([`XieluParams::from_gguf`]).
///
/// `Arc` because `ModelConfig` is cloned per request in the server and
/// `FfnActivation` is compared in `ExecutionPlan`; a shared slice is
/// both a cheap clone and a value comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct XieluLayers(Arc<[XieluParams]>);

impl XieluLayers {
    /// One entry per trunk layer, in layer order.
    pub fn new(layers: Vec<XieluParams>) -> Self {
        Self(layers.into())
    }

    /// Layer `il`'s parameters.
    ///
    /// Indexing panics on a layer the table does not have, and that is
    /// the right failure: the loader sized the table from the same
    /// `n_layers` every layer loop runs over, so an out-of-range `il`
    /// here is a decoder bug, not a file the user handed in.
    pub fn layer(&self, il: usize) -> XieluParams {
        self.0[il]
    }

    /// How many layers the table covers.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Never, for a loaded model; here so `len` has its clippy twin.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// The four `xielu.*` keys, as `llama-arch.cpp:370-373` spells them:
/// NO `{arch}.` prefix, unlike every other per-architecture
/// hyper-parameter.
pub const XIELU_KEYS: [&str; 4] = ["xielu.alpha_n", "xielu.alpha_p", "xielu.beta", "xielu.eps"];

/// Reads one of the four keys the way `apertus.cpp:6-9` does through
/// `get_key_or_arr(key, arr, n_layer())`: an array of EXACTLY
/// `n_layers` floats, or one scalar broadcast to every layer, and an
/// error when the key is absent (`required = true` is the default).
fn read_f32_per_layer(
    file: &impl TensorSource,
    key: &str,
    n_layers: usize,
) -> Result<Vec<f32>, LoadError> {
    let Some(value) = file.metadata(key) else {
        return Err(LoadError::MissingHparam(key.to_string()));
    };
    match value {
        GgufValue::Array(items) => {
            if items.len() != n_layers {
                return Err(LoadError::UnsupportedFeature(
                    key.to_string(),
                    format!(
                        "array of {} entries for {n_layers} layers; llama.cpp refuses this too \
                         (`key has wrong array length`, llama-model-loader.cpp:464-465)",
                        items.len()
                    ),
                ));
            }
            let mut out = Vec::with_capacity(n_layers);
            for (il, item) in items.iter().enumerate() {
                out.push(item.as_f32().ok_or_else(|| {
                    LoadError::UnsupportedFeature(
                        key.to_string(),
                        format!("entry {il} is not a float: {item:?}"),
                    )
                })?);
            }
            Ok(out)
        }
        scalar => scalar
            .as_f32()
            .map(|v| vec![v; n_layers])
            .ok_or_else(|| LoadError::MissingHparam(key.to_string())),
    }
}

/// The xIELU table for a file, read exactly as `apertus.cpp:6-9` reads
/// it, folded exactly as `ggml_xielu` folds it.
///
/// `n_layers` is the TRUNK count. `apertus` reads no
/// `nextn_predict_layers`, so `crate::mtp_blocks` refuses a nonzero one
/// for it and the trunk is `block_count`; passing the trunk rather than
/// `block_count` keeps that true if a NextN reader ever adopts xIELU.
pub fn read_xielu_layers(
    file: &impl TensorSource,
    n_layers: usize,
) -> Result<XieluLayers, LoadError> {
    let [alpha_n, alpha_p, beta, eps] = XIELU_KEYS;
    let alpha_n = read_f32_per_layer(file, alpha_n, n_layers)?;
    let alpha_p = read_f32_per_layer(file, alpha_p, n_layers)?;
    let beta = read_f32_per_layer(file, beta, n_layers)?;
    let eps = read_f32_per_layer(file, eps, n_layers)?;
    Ok(XieluLayers::new(
        (0..n_layers)
            .map(|il| XieluParams::from_gguf(alpha_n[il], alpha_p[il], beta[il], eps[il]))
            .collect(),
    ))
}

/// One layer's SwiGLU clamps, as `hparams.swiglu_clamp_exp[il]` /
/// `swiglu_clamp_shexp[il]`: `0.0` (or anything at or below `1e-6`,
/// llama-graph.cpp:1753) is no clamp on that site.
///
/// Both arrays are OPTIONAL upstream (`step35.cpp:28-29` pass
/// `required = false`; the arrays are zero-filled at
/// `llama-model.cpp:1146-1147`), so a file carrying neither runs
/// plain SwiGLU and the loader picks `FfnActivation::Swiglu` for it.
#[derive(Debug, Clone, PartialEq)]
pub struct SwigluClamps {
    /// `{arch}.swiglu_clamp_exp`, read by `build_moe_ffn` for the
    /// ROUTED experts (llama-graph.cpp:2146).
    routed: Arc<[f32]>,
    /// `{arch}.swiglu_clamp_shexp`, read by `build_ffn` for the SHARED
    /// experts AND the leading dense layers (llama-graph.cpp:1751),
    /// because `build_ffn` is both.
    dense: Arc<[f32]>,
}

/// llama-graph.cpp:1753 / :2148: `constexpr float eps = 1e-6f; if
/// (limit > eps)`.
const CLAMP_EPS: f32 = 1e-6;

impl SwigluClamps {
    /// One entry per trunk layer in each array, in layer order.
    pub fn new(routed: Vec<f32>, dense: Vec<f32>) -> Self {
        assert_eq!(routed.len(), dense.len(), "one entry per layer in both");
        Self {
            routed: routed.into(),
            dense: dense.into(),
        }
    }

    /// The activation layer `il`'s routed experts run.
    pub fn routed(&self, il: usize) -> GluAct {
        Self::act(self.routed[il])
    }

    /// The activation layer `il`'s dense FFN or shared experts run.
    pub fn dense(&self, il: usize) -> GluAct {
        Self::act(self.dense[il])
    }

    fn act(limit: f32) -> GluAct {
        if limit > CLAMP_EPS {
            GluAct::SwigluClamped { limit }
        } else {
            GluAct::Swiglu
        }
    }

    /// How many layers the tables cover.
    pub fn len(&self) -> usize {
        self.routed.len()
    }

    /// Never, for a loaded model; here so `len` has its clippy twin.
    pub fn is_empty(&self) -> bool {
        self.routed.is_empty()
    }
}

/// Architectures on the generic path whose graph READS the two clamp
/// arrays: `grep -l LLM_KV_SWIGLU_CLAMP src/models/*.cpp` is
/// `step35.cpp`, `deepseek4.cpp` and `dflash.cpp`, and the last two are
/// on ferrox's own DeepSeek-4 engine. For every other architecture the
/// arrays stay zero-filled upstream whatever the file says, so the
/// keys are dead metadata there and ferrox ignores them the same way.
pub const SWIGLU_CLAMP_READERS: &[&str] = &["step35"];

/// Does this architecture's graph read `swiglu_clamp_exp` / `_shexp`?
pub fn reads_swiglu_clamps(arch: &str) -> bool {
    SWIGLU_CLAMP_READERS.contains(&arch)
}

/// The two clamp arrays for a file, read as `step35.cpp:28-29` reads
/// them: `get_key_or_arr` at `n_layer()` length with `required =
/// false`, so an absent key is all zeros. `Ok(None)` when the file
/// carries NEITHER key, which is plain SwiGLU with nothing per layer
/// to carry.
///
/// `n_layers` is the TRUNK count: `step35.cpp:28-29` run AFTER `:32`
/// has read `nextn_predict_layers`, so `n_layer()` is the trunk there
/// -- but the converter writes both arrays at `block_count` length
/// (`step3.py:207-220`, padded with `0.0` for the MTP blocks), and
/// llama.cpp's `get_key_or_arr` refuses a length other than the one
/// asked for. `block_count` is what a real export carries, so that is
/// the length accepted here, and only the trunk's entries are kept.
pub fn read_swiglu_clamps(
    file: &impl TensorSource,
    arch: &str,
    trunk: &crate::mtp_blocks::TrunkLayers,
) -> Result<Option<SwigluClamps>, LoadError> {
    let key = |k: &str| format!("{arch}.{k}");
    let (exp_key, shexp_key) = (key("swiglu_clamp_exp"), key("swiglu_clamp_shexp"));
    if file.metadata(&exp_key).is_none() && file.metadata(&shexp_key).is_none() {
        return Ok(None);
    }
    let read = |k: &str| -> Result<Vec<f32>, LoadError> {
        if file.metadata(k).is_none() {
            return Ok(vec![0.0; trunk.n_layers]);
        }
        let mut v = read_f32_per_layer(file, k, trunk.block_count)?;
        v.truncate(trunk.n_layers);
        Ok(v)
    };
    Ok(Some(SwigluClamps::new(read(&exp_key)?, read(&shexp_key)?)))
}

/// Architectures whose FFN activation is xIELU: the graphs that call
/// `ggml_xielu`, measured by `grep -l ggml_xielu src/models/*.cpp` --
/// `apertus.cpp` alone at this checkout.
pub const XIELU_ARCHITECTURES: &[&str] = &["apertus"];

/// Does this architecture's FFN run xIELU? See [`XIELU_ARCHITECTURES`].
pub fn uses_xielu(arch: &str) -> bool {
    XIELU_ARCHITECTURES.contains(&arch)
}

/// One layer's FFN activations, by SITE: llama.cpp builds the routed
/// experts with `build_moe_ffn` and everything else -- the dense
/// layers' FFN and the shared experts -- with `build_ffn`, and the two
/// read different clamp arrays (`llama-graph.cpp:2146` vs `:1751`).
///
/// A struct rather than a second accessor argument so that a body
/// cannot ask for "the activation" without saying which; for every
/// activation but the clamped one the two fields are equal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerFfnActs {
    /// What the top-k routed experts run (`build_moe_ffn`).
    pub routed: GluAct,
    /// What a dense layer's FFN and the shared experts run
    /// (`build_ffn`).
    pub dense: GluAct,
}

impl LayerFfnActs {
    fn same(act: GluAct) -> Self {
        Self {
            routed: act,
            dense: act,
        }
    }

    /// True when both sites run plain SwiGLU -- the per-layer question
    /// the fused Metal stacks ask, since their kernels spell nothing
    /// else.
    pub fn all_swiglu(self) -> bool {
        self.routed.is_swiglu() && self.dense.is_swiglu()
    }
}

impl ModelConfig {
    /// Layer `il`'s FFN activations, with their parameters, by site.
    /// THE accessor: every FFN body -- routed, shared, dense, batched,
    /// slotted -- reads its activation here and nowhere else.
    ///
    /// For every architecture but the parameterised ones this is the
    /// same answer for every `il` and both sites, which is what
    /// `ffn_activation` used to be converted to directly; that
    /// conversion no longer exists, because it could not be written for
    /// a variant that needs the layer.
    pub fn layer_ffn_acts(&self, il: usize) -> LayerFfnActs {
        match &self.ffn_activation {
            // `SwigluFused` is the same activation as `Swiglu`; it only
            // says gate and up arrive as one on-disk tensor (Phi), which
            // the loader has already split by the time a `WeightMatrix`
            // exists.
            FfnActivation::Swiglu | FfnActivation::SwigluFused => {
                LayerFfnActs::same(GluAct::Swiglu)
            }
            FfnActivation::Gelu => LayerFfnActs::same(GluAct::Geglu),
            // Ungated on disk, gated in the enum: the loader aliases
            // gate to up. See `FfnActivation::ReluSqr`.
            FfnActivation::ReluSqr => LayerFfnActs::same(GluAct::Reglu),
            FfnActivation::Xielu(layers) => LayerFfnActs::same(GluAct::Xielu(layers.layer(il))),
            FfnActivation::SwigluClamped(clamps) => LayerFfnActs {
                routed: clamps.routed(il),
                dense: clamps.dense(il),
            },
        }
    }

    /// The ONE activation every layer of this model runs, or `None`
    /// when it varies by layer -- the whole-model question the fused
    /// Metal stacks and their eligibility checks ask, since each takes
    /// one activation uniform for a whole run of layers.
    ///
    /// `None` is a refusal at every such site. It is not derived by
    /// comparing `layer_ffn_act` across layers, because a
    /// parameterised activation is per layer BY TYPE: a two-layer
    /// xIELU model whose two parameter sets happen to be equal is still
    /// not something a kernel with no xIELU in it can serve.
    pub fn model_ffn_act(&self) -> Option<GluAct> {
        match &self.ffn_activation {
            FfnActivation::Xielu(_) | FfnActivation::SwigluClamped(_) => None,
            FfnActivation::Swiglu
            | FfnActivation::SwigluFused
            | FfnActivation::Gelu
            | FfnActivation::ReluSqr => Some(self.layer_ffn_acts(0).dense),
        }
    }

    /// Does this model's FFN have no gate matrix on disk?
    ///
    /// The two ungated activations share the loader's aliasing
    /// (`load_dense_expert`), so the question is asked once here rather
    /// than as `== ReluSqr` at the site, where the second variant would
    /// have been forgotten.
    pub fn ffn_is_ungated(&self) -> bool {
        match &self.ffn_activation {
            FfnActivation::ReluSqr | FfnActivation::Xielu(_) => true,
            FfnActivation::Swiglu
            | FfnActivation::SwigluFused
            | FfnActivation::SwigluClamped(_)
            | FfnActivation::Gelu => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrox_moe::GluAct;

    #[derive(Clone, Default)]
    struct Meta(Vec<(String, GgufValue)>);
    impl Meta {
        fn insert(&mut self, key: &str, value: GgufValue) {
            self.remove(key);
            self.0.push((key.to_string(), value));
        }
        fn remove(&mut self, key: &str) {
            self.0.retain(|(k, _)| k != key);
        }
    }
    impl TensorSource for Meta {
        fn metadata(&self, key: &str) -> Option<&GgufValue> {
            self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        }
        fn find_tensor(&self, _: &str) -> Option<&ferrox_gguf::TensorInfo> {
            None
        }
        fn tensor_bytes(&self, name: &str) -> Result<&[u8], ferrox_gguf::GgufError> {
            Err(ferrox_gguf::GgufError::TensorNotFound(name.to_string()))
        }
        fn tensor_mapped_range(
            &self,
            name: &str,
        ) -> Result<(Arc<ferrox_gguf::MmapHandle>, std::ops::Range<usize>), ferrox_gguf::GgufError>
        {
            Err(ferrox_gguf::GgufError::TensorNotFound(name.to_string()))
        }
    }

    fn base_config() -> ModelConfig {
        let mut cfg = crate::config::glm_5_2();
        cfg.n_layers = 2;
        cfg
    }

    fn xielu_config(layers: Vec<XieluParams>) -> ModelConfig {
        let mut cfg = base_config();
        cfg.n_layers = layers.len();
        cfg.ffn_activation = FfnActivation::Xielu(XieluLayers::new(layers));
        cfg
    }

    /// The accessor indexes by layer. If it read `[0]` for every layer
    /// -- which is what a scalar `ffn_activation` conversion amounted
    /// to -- layer 1 would answer layer 0's parameters here.
    #[test]
    fn layer_ffn_act_answers_each_layer_s_own_parameters() {
        let p0 = XieluParams::from_gguf(0.8, 0.8, 0.5, -1e-6);
        let p1 = XieluParams::from_gguf(0.2, 1.5, 0.75, -0.3);
        let cfg = xielu_config(vec![p0, p1]);
        assert_eq!(cfg.layer_ffn_acts(0), LayerFfnActs::same(GluAct::Xielu(p0)));
        assert_eq!(cfg.layer_ffn_acts(1), LayerFfnActs::same(GluAct::Xielu(p1)));
        assert_ne!(p0, p1, "the test needs two different parameter sets");
        assert!(cfg.ffn_is_ungated());
        assert_eq!(
            cfg.model_ffn_act(),
            None,
            "a parameterised activation has no whole-model answer, even with equal parameters"
        );
        // Equal parameters on every layer are STILL per layer by type.
        assert_eq!(xielu_config(vec![p0, p0]).model_ffn_act(), None);
    }

    /// The uniform kinds answer the same thing on every layer, and the
    /// whole-model accessor agrees with the per-layer one.
    #[test]
    fn uniform_activations_answer_the_same_on_every_layer() {
        for (kind, want, ungated) in [
            (FfnActivation::Swiglu, GluAct::Swiglu, false),
            (FfnActivation::SwigluFused, GluAct::Swiglu, false),
            (FfnActivation::Gelu, GluAct::Geglu, false),
            (FfnActivation::ReluSqr, GluAct::Reglu, true),
        ] {
            let mut cfg = base_config();
            cfg.ffn_activation = kind.clone();
            for il in 0..cfg.n_layers {
                assert_eq!(
                    cfg.layer_ffn_acts(il),
                    LayerFfnActs::same(want),
                    "{kind:?} layer {il}"
                );
                assert!(cfg.layer_ffn_acts(il).all_swiglu() == (want == GluAct::Swiglu));
            }
            assert_eq!(cfg.model_ffn_act(), Some(want), "{kind:?}");
            assert_eq!(cfg.ffn_is_ungated(), ungated, "{kind:?}");
        }
    }

    /// `get_key_or_arr`'s three answers: an array of the right length
    /// is taken per layer, a scalar is broadcast, and an array of the
    /// wrong length or a missing key is an error naming the key.
    #[test]
    fn the_four_keys_are_read_as_llama_cpp_reads_them() {
        let arr = |v: &[f32]| GgufValue::Array(v.iter().map(|&x| GgufValue::F32(x)).collect());
        let mut md = Meta::default();
        md.insert("xielu.alpha_n", arr(&[0.8, 0.2]));
        md.insert("xielu.alpha_p", arr(&[0.8, 1.5]));
        md.insert("xielu.beta", GgufValue::F32(0.5)); // scalar, broadcast
        md.insert("xielu.eps", arr(&[-1e-6, -0.3]));
        let layers = read_xielu_layers(&md, 2).expect("reads");
        assert_eq!(layers.len(), 2);
        assert_eq!(
            layers.layer(0),
            XieluParams::from_gguf(0.8, 0.8, 0.5, -1e-6)
        );
        assert_eq!(layers.layer(1), XieluParams::from_gguf(0.2, 1.5, 0.5, -0.3));

        let mut short = md.clone();
        short.insert("xielu.eps", arr(&[-1e-6]));
        let err = read_xielu_layers(&short, 2).expect_err("wrong length refuses");
        let msg = err.to_string();
        assert!(
            msg.contains("xielu.eps") && msg.contains("1 entries"),
            "{msg}"
        );

        let mut missing = md.clone();
        missing.remove("xielu.alpha_p");
        let err = read_xielu_layers(&missing, 2).expect_err("a missing key refuses");
        assert!(err.to_string().contains("xielu.alpha_p"), "{err}");
    }

    /// The clamp tables: each site reads its own array, a zero entry is
    /// plain SwiGLU on that site alone, and the whole-model answer is
    /// `None` even when every entry is zero, because the variant is per
    /// layer by type.
    #[test]
    fn the_clamp_arrays_are_read_per_site_and_zero_means_plain_swiglu() {
        let mut cfg = base_config();
        cfg.n_layers = 3;
        cfg.ffn_activation = FfnActivation::SwigluClamped(SwigluClamps::new(
            vec![0.0, 1.5, 0.0],
            vec![2.0, 0.0, 1e-7],
        ));
        assert_eq!(
            cfg.layer_ffn_acts(0),
            LayerFfnActs {
                routed: GluAct::Swiglu,
                dense: GluAct::SwigluClamped { limit: 2.0 },
            }
        );
        assert_eq!(
            cfg.layer_ffn_acts(1),
            LayerFfnActs {
                routed: GluAct::SwigluClamped { limit: 1.5 },
                dense: GluAct::Swiglu,
            }
        );
        // At or below llama.cpp's 1e-6 is no clamp.
        assert_eq!(cfg.layer_ffn_acts(2), LayerFfnActs::same(GluAct::Swiglu));
        assert!(cfg.layer_ffn_acts(2).all_swiglu() && !cfg.layer_ffn_acts(0).all_swiglu());
        assert_eq!(cfg.model_ffn_act(), None);
        assert!(!cfg.ffn_is_ungated());
    }

    /// The two clamp keys are read as `get_key_or_arr(..., false)` reads
    /// them: absent is zeros, an array is taken at `block_count` length
    /// and truncated to the trunk, a scalar is broadcast, and a file
    /// with neither key has no table at all.
    #[test]
    fn the_clamp_keys_are_read_as_llama_cpp_reads_them() {
        let arr = |v: &[f32]| GgufValue::Array(v.iter().map(|&x| GgufValue::F32(x)).collect());
        let trunk = crate::mtp_blocks::TrunkLayers {
            block_count: 3,
            n_layers: 2,
            n_mtp_blocks: 1,
        };
        let mut md = Meta::default();
        assert_eq!(
            read_swiglu_clamps(&md, "step35", &trunk).expect("reads"),
            None
        );
        md.insert("step35.swiglu_clamp_exp", arr(&[0.0, 7.0, 0.0]));
        let clamps = read_swiglu_clamps(&md, "step35", &trunk)
            .expect("reads")
            .expect("one key is a table");
        assert_eq!(clamps.len(), 2, "trunk entries only");
        assert_eq!(clamps.routed(1), GluAct::SwigluClamped { limit: 7.0 });
        assert_eq!(clamps.dense(1), GluAct::Swiglu, "the absent key is zeros");
        md.insert("step35.swiglu_clamp_shexp", GgufValue::F32(16.0));
        let clamps = read_swiglu_clamps(&md, "step35", &trunk)
            .expect("reads")
            .expect("table");
        assert_eq!(clamps.dense(0), GluAct::SwigluClamped { limit: 16.0 });
        assert_eq!(clamps.dense(1), GluAct::SwigluClamped { limit: 16.0 });
        md.insert("step35.swiglu_clamp_exp", arr(&[0.0, 7.0]));
        let err = read_swiglu_clamps(&md, "step35", &trunk).expect_err("wrong length refuses");
        assert!(err.to_string().contains("swiglu_clamp_exp"), "{err}");
        assert!(reads_swiglu_clamps("step35"));
        for arch in ["llama", "apertus", "laguna", "deepseek2", "gpt-oss"] {
            assert!(!reads_swiglu_clamps(arch), "{arch}");
        }
    }

    /// The table is the measured list of graphs that call
    /// `ggml_xielu`, and nothing else reads as xIELU.
    #[test]
    fn only_the_graphs_that_call_ggml_xielu_use_it() {
        assert!(uses_xielu("apertus"));
        for arch in ["llama", "arcee", "plm", "step35", "gemma3"] {
            assert!(!uses_xielu(arch), "{arch}");
        }
        assert_eq!(XIELU_KEYS[0], "xielu.alpha_n", "no architecture prefix");
    }
}
