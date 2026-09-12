//! Which tensor each of a decoder layer's norm sites is stored under,
//! and with which function, per architecture -- resolved ONCE.
//!
//! A generic layer has up to four norm sites, plus the final one before
//! the lm_head:
//!
//! ```text
//! ffn_inp = x       + post_attn( attn( pre_attn(x) ) )
//! out     = ffn_inp + post_ffn ( ffn ( pre_ffn(ffn_inp) ) )
//! logits  = lm_head( output( out ) )
//! ```
//!
//! llama.cpp names each site's tensor per architecture, in that
//! architecture's `load_arch_tensors`, and reuses ONE tensor name for
//! DIFFERENT sites: `blk.N.attn_output_norm` (`LLM_TENSOR_ATTN_OUT_NORM`,
//! `llama-arch.cpp:423`) is `dbrx`'s pre-FFN norm (`dbrx.cpp:34,110-113`)
//! and `grok`'s post-attention norm (`grok.cpp:62,143-146`), and
//! `blk.N.post_attention_norm` (`LLM_TENSOR_ATTN_POST_NORM`) is Gemma-2's
//! post-attention norm and `gpt-oss`'s pre-FFN one. So which SITE a
//! tensor feeds is decided by the architecture string, never by which
//! tensors happen to be present: reading one file's tensor with
//! another's meaning moves a whole norm to the other side of a residual
//! add, silently.
//!
//! **Why one table.** `loader.rs` used to decide all of this with a
//! chain of `if` branches restated at the pre-attention, pre-FFN and
//! final sites -- three copies of one decision, and each new
//! architecture added a branch to each. Two of those branches also
//! answered "which FUNCTION" (RMS, LayerNorm, none) alongside "which
//! TENSOR", so the norm function was decided three times too. Here the
//! function is [`NormFunction`], read once, and each site is a
//! [`StoredNorm`] naming the tensor spellings it accepts and whether the
//! file must have one. The loader reads the table; it does not restate
//! it.
//!
//! The lists below are the whole content of the table, and each one
//! cites the graph it was read from. `loader.rs`'s
//! `the_norm_slot_and_function_lists_cannot_contradict` pins that no
//! architecture is on two lists that would read one tensor two ways,
//! and `every_architecture_keyed_behaviour_table_names_a_real_generic_row`
//! that every name is a row this loader can reach.

use crate::loader::load_f32_vec;
use crate::norm::{norm_function, NormFunction, NormOp, NormParam};
use crate::LoadError;
use ferrox_gguf::{GgufError, TensorSource};

/// Architectures that store their **pre-FFN** norm under the tensor name
/// `blk.N.post_attention_norm.weight` and carry no `blk.N.ffn_norm`.
///
/// Gemma writes the same tensor name for a genuinely different norm: it
/// is applied to the attention output *inside* the attention residual,
/// and Gemma also carries `ffn_norm`. Reading one file's tensor with the
/// other's meaning silently moves a whole RMSNorm to the wrong side of a
/// residual add, so the meaning is decided by architecture, not by which
/// tensors happen to be present.
///
/// - `gpt-oss`: `openai-moe.cpp` norms `ffn_inp` with `attn_post_norm`.
/// - `glm4moe`: `src/models/glm4-moe.cpp:75` creates `attn_post_norm`
///   and no `ffn_norm`, and `:215` norms `ffn_inp` with it -- the slot
///   its refusal had named for a year, and the one line that admits
///   GLM-4.5 / 4.5-Air / 4.6.
/// - `seed_oss`: `src/models/seed-oss.cpp:36-37` creates `attn_norm` and
///   `attn_post_norm` and **no** `ffn_norm`, and `:113-115` norms
///   `ffn_inp` -- the post-attention residual -- with `attn_post_norm`.
///
/// This is deliberately NOT `arch == "gpt-oss"`, which is what it used
/// to be. That one flag also gated gpt-oss's five extra per-layer
/// tensors (sinks, biases, the SwiGLU clamp), and widening it would have
/// handed `seed_oss` attention sinks it does not have. Two facts, two
/// predicates.
pub const PRE_FFN_NORM_IS_POST_ATTENTION_NORM: &[&str] = &["gpt-oss", "seed_oss", "glm4moe"];

/// Architectures that store their **pre-FFN** norm under
/// `blk.N.attn_output_norm.weight` (`LLM_TENSOR_ATTN_OUT_NORM`) and carry
/// no `blk.N.ffn_norm`.
///
/// `dbrx`: `dbrx.cpp:34` creates `attn_out_norm` and no `ffn_norm`, and
/// `:110-113` norms `ffn_inp` -- the post-attention residual -- with it,
/// which is the same slot `gpt-oss` keeps under the other name. Its
/// norm FUNCTION is a weighted LayerNorm
/// (`capability::WEIGHTED_LAYER_NORM`); that is [`NormFunction`]'s
/// business, not this list's.
pub const PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM: &[&str] = &["dbrx"];

/// Architectures whose **post-attention** norm (Gemma-2's
/// `post_attention_norm` slot) is stored as `blk.N.attn_output_norm` and
/// whose **post-FFN** norm is `blk.N.layer_output_norm`
/// (`LLM_TENSOR_LAYER_OUT_NORM`, `llama-arch.cpp:421`) with
/// `post_ffw_norm` as the fallback spelling -- both REQUIRED.
///
/// `grok`: `grok.cpp:62` creates `attn_out_norm` as required and
/// `:143-146` applies it to the attention output BEFORE the residual
/// add (`ffn_inp = add(cur, inpSA)` at :148), which is the post-norm
/// slot and not `dbrx`'s pre-FFN one -- `grok` has its own `ffn_norm`
/// at :64. `:75-78` create `ffn_post_norm` from `LAYER_OUT_NORM` when
/// present (`conversion/grok.py`'s Grok-1 mapping, `rms_norm_3`) and
/// from `FFN_POST_NORM` otherwise (Grok-2's `post_moe_norm`), required
/// either way, and `:185-188` apply it before the FFN residual add.
pub const POST_NORMS_UNDER_GROK_NAMES: &[&str] = &["grok"];

/// A norm site whose weight the file stores.
///
/// `names` are base names tried in order; each is looked up as
/// `blk.N.{name}.weight` first and `blk.N.{name}` second (or
/// `{name}.weight` / `{name}` for the final norm). The first spelling
/// present wins.
///
/// **Both spellings, because llama.cpp's own trees disagree.** `LLM_TN`
/// appends `.weight` only when it is given a suffix
/// (`src/llama-arch.cpp:898-910`). Every architecture that creates the
/// post-norms passes one -- `tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i)`
/// in gemma2, gemma3, glm4, exaone4, afmoe and the rest -- EXCEPT
/// `plamo3`, which uses the two-argument overload
/// (`src/models/plamo3.cpp:52,55`) and therefore asks for
/// `blk.N.post_attention_norm` with no suffix at all. The converter
/// agrees with it by a second accident: `gguf-py/gguf/tensor_mapping.py`
/// gives the PLaMo entries as `...post_mixer_norm.weight` -- keys that
/// already END in `.weight` -- and `TensorNameMap.get_type_and_name`
/// tries an exact match first, so the mapped name is emitted with
/// nothing appended. ferrox read only `.weight`, which is why `plamo3`
/// could not have loaded a real checkpoint: fail-closed rather than
/// wrong, but not "a fixture away", which is what its triage verdict
/// said. Both spellings are accepted rather than one chosen per
/// architecture, because the choice is a property of the file and
/// nothing in the metadata declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredNorm {
    pub names: &'static [&'static str],
    /// `create_tensor(..., 0)` in llama.cpp: the file must have one.
    /// `false` is `TENSOR_NOT_REQUIRED`.
    pub required: bool,
}

impl StoredNorm {
    const fn required(names: &'static [&'static str]) -> Self {
        Self {
            names,
            required: true,
        }
    }

    const fn optional(names: &'static [&'static str]) -> Self {
        Self {
            names,
            required: false,
        }
    }

    /// Every full tensor name this site accepts for its weight, in
    /// lookup order.
    fn candidates(&self, layer: Option<usize>) -> Vec<String> {
        self.candidates_for(layer, NormParam::Weight)
    }

    /// The names for one part. The bare name (no suffix) is a weight
    /// spelling some exporters use and is never a bias.
    fn candidates_for(&self, layer: Option<usize>, part: NormParam) -> Vec<String> {
        self.names
            .iter()
            .flat_map(|base| {
                let full = match layer {
                    Some(l) => format!("blk.{l}.{base}"),
                    None => (*base).to_string(),
                };
                match part {
                    NormParam::Weight => vec![format!("{full}.weight"), full],
                    NormParam::Bias => vec![format!("{full}.bias")],
                }
            })
            .collect()
    }

    /// The weight, or `None` for an optional site the file omits.
    pub fn load(
        &self,
        file: &impl TensorSource,
        layer: Option<usize>,
    ) -> Result<Option<Vec<f32>>, LoadError> {
        let candidates = self.candidates(layer);
        if let Some(name) = candidates.iter().find(|n| file.find_tensor(n).is_some()) {
            return load_f32_vec(file, name).map(Some);
        }
        if self.required {
            return Err(LoadError::Gguf(GgufError::TensorNotFound(
                candidates.join(" | "),
            )));
        }
        Ok(None)
    }

    /// One REQUIRED part of a REQUIRED site: the weight, or the bias of
    /// a `LayerNormBias` architecture, which `orion.cpp:18,25,31` and
    /// `nemotron.cpp:19,26,35` create with `create_tensor(..., 0)`.
    fn load_required_part(
        &self,
        file: &impl TensorSource,
        layer: Option<usize>,
        part: NormParam,
    ) -> Result<Vec<f32>, LoadError> {
        debug_assert!(self.required, "load_required_part on an optional site");
        let candidates = self.candidates_for(layer, part);
        match candidates.iter().find(|n| file.find_tensor(n).is_some()) {
            Some(name) => load_f32_vec(file, name),
            None => Err(LoadError::Gguf(GgufError::TensorNotFound(
                candidates.join(" | "),
            ))),
        }
    }
}

/// Where every norm of one architecture's layer lives.
///
/// The three pre-norm sites are `Option<StoredNorm>`: `None` is the
/// post-norm-only topology (`capability::POST_NORM_ONLY_ARCHITECTURES`),
/// where the site has no tensor AND no norm and
/// [`NormSites::load_pre_norm`] answers [`NormOp::None`]. A
/// non-parametric architecture (`olmo`) keeps its `Some`: the
/// [`NormFunction`] decides the tensor is never read, and that decision
/// belongs in one place rather than in a second `None` here.
///
/// The two post-norm sites are RMSNorm-only slots, as
/// `AttnWeights::post_attn_norm` / `post_ffn_norm` are: no admitted
/// architecture post-norms with anything else. `None` means the
/// architecture has no such site, which for `gpt-oss` / `seed_oss` is
/// how the tensor that WOULD be read there is kept for the pre-FFN
/// slot instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormSites {
    pub function: NormFunction,
    pub attn: Option<StoredNorm>,
    pub ffn: Option<StoredNorm>,
    pub post_attn: Option<StoredNorm>,
    pub post_ffn: Option<StoredNorm>,
    pub output: StoredNorm,
}

impl NormSites {
    /// The table row for `arch`.
    pub fn for_arch(arch: &str) -> Self {
        let function = norm_function(arch);
        let mut sites = Self {
            function,
            attn: Some(StoredNorm::required(&["attn_norm"])),
            ffn: Some(StoredNorm::required(&["ffn_norm"])),
            post_attn: Some(StoredNorm::optional(&["post_attention_norm"])),
            post_ffn: Some(StoredNorm::optional(&["post_ffw_norm"])),
            output: StoredNorm::required(&["output_norm"]),
        };
        if crate::capability::is_post_norm_only(arch) {
            sites.attn = None;
            sites.ffn = None;
        }
        if PRE_FFN_NORM_IS_POST_ATTENTION_NORM.contains(&arch) {
            sites.ffn = Some(StoredNorm::required(&["post_attention_norm"]));
            sites.post_attn = None;
        }
        if PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM.contains(&arch) {
            sites.ffn = Some(StoredNorm::required(&["attn_output_norm"]));
        }
        if POST_NORMS_UNDER_GROK_NAMES.contains(&arch) {
            sites.post_attn = Some(StoredNorm::required(&["attn_output_norm"]));
            sites.post_ffn = Some(StoredNorm::required(&[
                "layer_output_norm",
                "post_ffw_norm",
            ]));
        }
        sites
    }

    /// A pre-norm site (attention, FFN, or with `layer == None` the
    /// final norm), as the [`NormOp`] the decoder applies there.
    pub fn load_pre_norm(
        &self,
        site: Option<StoredNorm>,
        file: &impl TensorSource,
        layer: Option<usize>,
    ) -> Result<NormOp, LoadError> {
        match site {
            None => Ok(NormOp::None),
            Some(stored) => self
                .function
                .resolve(|part| stored.load_required_part(file, layer, part)),
        }
    }

    /// A post-norm site: an RMSNorm weight, or `None` when this
    /// architecture has no such site or the file omits an optional one.
    pub fn load_post_norm(
        site: Option<StoredNorm>,
        file: &impl TensorSource,
        layer: usize,
    ) -> Result<Option<Vec<f32>>, LoadError> {
        match site {
            None => Ok(None),
            Some(stored) => stored.load(file, Some(layer)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default row is the plain pre-norm layer with both post-norms
    /// optional, which is what every architecture not named in a list
    /// gets.
    #[test]
    fn the_default_row_is_the_plain_pre_norm_layer() {
        let s = NormSites::for_arch("llama");
        assert_eq!(s.function, NormFunction::Rms);
        assert_eq!(s.attn, Some(StoredNorm::required(&["attn_norm"])));
        assert_eq!(s.ffn, Some(StoredNorm::required(&["ffn_norm"])));
        assert_eq!(
            s.post_attn,
            Some(StoredNorm::optional(&["post_attention_norm"]))
        );
        assert_eq!(s.post_ffn, Some(StoredNorm::optional(&["post_ffw_norm"])));
        assert_eq!(s.output, StoredNorm::required(&["output_norm"]));
    }

    /// `attn_output_norm` feeds a DIFFERENT site on the two rows that
    /// store it, and neither row inherits the other's reading.
    ///
    /// This is the test that separates `grok` from `dbrx`: the same
    /// tensor name, one on each side of the attention residual add.
    #[test]
    fn attn_output_norm_is_dbrxs_pre_ffn_norm_and_groks_post_attention_norm() {
        let dbrx = NormSites::for_arch("dbrx");
        assert_eq!(dbrx.function, NormFunction::LayerNorm);
        assert_eq!(
            dbrx.ffn,
            Some(StoredNorm::required(&["attn_output_norm"])),
            "dbrx.cpp:110-113 norms ffn_inp with attn_out_norm"
        );
        assert_eq!(
            dbrx.post_attn,
            Some(StoredNorm::optional(&["post_attention_norm"])),
            "no post-attention norm of its own"
        );

        let grok = NormSites::for_arch("grok");
        assert_eq!(grok.function, NormFunction::Rms);
        assert_eq!(grok.ffn, Some(StoredNorm::required(&["ffn_norm"])));
        assert_eq!(
            grok.post_attn,
            Some(StoredNorm::required(&["attn_output_norm"])),
            "grok.cpp:143-146 norms the attention output before the residual add"
        );
        assert_eq!(
            grok.post_ffn,
            Some(StoredNorm::required(&[
                "layer_output_norm",
                "post_ffw_norm"
            ])),
            "grok.cpp:75-78: LAYER_OUT_NORM first, FFN_POST_NORM as the fallback"
        );
    }

    /// The post-norm-only rows have no pre-norm tensors and no pre-norm
    /// at all; the parameterless row keeps its sites and never reads
    /// them.
    #[test]
    fn the_two_no_tensor_shapes_are_kept_apart() {
        let olmo2 = NormSites::for_arch("olmo2");
        assert_eq!(olmo2.attn, None);
        assert_eq!(olmo2.ffn, None);
        assert_eq!(olmo2.function, NormFunction::Rms);

        let olmo = NormSites::for_arch("olmo");
        assert_eq!(olmo.function, NormFunction::LayerNormNoParams);
        assert!(
            olmo.attn.is_some(),
            "the site exists; the function skips the read"
        );
    }

    /// `gpt-oss` keeps `post_attention_norm` for the pre-FFN slot and
    /// therefore has NO post-attention site: the tensor cannot be read
    /// twice.
    #[test]
    fn the_gpt_oss_slot_reads_the_tensor_once() {
        let s = NormSites::for_arch("gpt-oss");
        assert_eq!(s.ffn, Some(StoredNorm::required(&["post_attention_norm"])));
        assert_eq!(s.post_attn, None);
    }

    /// Both spellings of every candidate, suffixed first, in order.
    #[test]
    fn candidates_try_the_suffixed_spelling_first_for_each_name() {
        let s = StoredNorm::required(&["layer_output_norm", "post_ffw_norm"]);
        assert_eq!(
            s.candidates(Some(3)),
            vec![
                "blk.3.layer_output_norm.weight",
                "blk.3.layer_output_norm",
                "blk.3.post_ffw_norm.weight",
                "blk.3.post_ffw_norm",
            ]
        );
        assert_eq!(
            StoredNorm::required(&["output_norm"]).candidates(None),
            vec!["output_norm.weight", "output_norm"]
        );
    }
}
