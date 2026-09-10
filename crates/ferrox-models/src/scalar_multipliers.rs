//! The four scalar multipliers a checkpoint can declare in METADATA,
//! and which architectures apply which of them.
//!
//! `{arch}.logit_scale`, `{arch}.residual_scale`,
//! `{arch}.embedding_scale` and `{arch}.attention.scale`
//! (`llama-arch.cpp`: `LLM_KV_LOGIT_SCALE`, `LLM_KV_RESIDUAL_SCALE`,
//! `LLM_KV_EMBEDDING_SCALE`, `LLM_KV_ATTENTION_SCALE`) are the blind
//! spot [`crate::loader::assert_every_tensor_consumed`] cannot cover:
//! they are hyper-parameters, not weights, so a checkpoint carrying one
//! leaves no unread tensor. Before this module ferrox REFUSED any file
//! declaring one, by name, because the alternative was loading it and
//! computing a differently-scaled graph than it was trained as.
//!
//! **One implementation, parameterised by architecture.** llama.cpp
//! spreads these over four unrelated places -- the shared
//! `build_inp_embd` for the embedding scale (llama-graph.cpp:2337-2342),
//! `kq_scale` for the attention scale, a `ggml_scale` before each
//! residual add, and one more after the lm_head -- and each
//! architecture picks a subset. ferrox resolves the subset ONCE here,
//! into plain `Option<f32>` fields on [`crate::ModelConfig`] that the
//! decoder reads as data. `granite` and `granitemoe` differ in the FFN
//! and not in the scaling, and `granite-moe` is a ferrox-only alias for
//! `granitemoe`, so all three share one [`MultiplierSupport`] constant
//! and cannot drift apart.
//!
//! **What each architecture reads, against the C.**
//!
//! | arch | llama.cpp | embedding | residual | logit | attention |
//! |---|---|---|---|---|---|
//! | `granite` | `granite.cpp:5-10` | yes | yes | divide | yes |
//! | `granitemoe` | `granite-moe.cpp:3-10` | yes | yes | divide | yes |
//!
//! **Gemma is not in that table, and that is the interesting part.** It
//! scales its embeddings by `sqrt(n_embd)` and, at 27B, overrides its
//! attention scale -- but it reads NEITHER KEY: `gemma3.cpp:31` and
//! `gemma2.cpp:27` assign `f_attention_scale` from the model type, and
//! the embedding scale is arithmetic in the graph. The whole family used
//! to be exempted from the refusal list wholesale, on the strength of
//! implementing two of the four, which meant a hand-written
//! `gemma3.residual_scale` would have loaded and been ignored. It is
//! refused now, along with the two keys Gemma's own scales are NOT read
//! from, because a file declaring one describes something llama.cpp does
//! not do either. The Gemma scales come from
//! `capability::attention_scale_override` and `loader.rs`'s family
//! branch, which is where an arch-computed value belongs.
//!
//! Two rows are deliberately NOT here, and both are one table entry
//! away rather than a second implementation:
//!
//! * **MiniCPM** runs `llama_model_granite::graph` verbatim
//!   (`models.h:1594-1601`) -- the same graph object, not a similar one.
//!   What it adds is DEFAULTS: `minicpm.cpp:5-7` hardcodes
//!   `f_embedding_scale = 12.0`, `f_residual_scale = 1.4/sqrt(n_layer)`
//!   and `f_logit_scale = 256/n_embd` BEFORE letting the file override
//!   them, so an older MiniCPM export carrying none of the three keys is
//!   still scaled by all three. That is a fallback hook this module does
//!   not have yet, and adding it without a MiniCPM fixture would be dead
//!   code, so MiniCPM stays a `DedicatedOnly` refusal.
//! * **Command-R / Cohere2** apply `f_logit_scale` as a MULTIPLY rather
//!   than a divide (`command-r.cpp:136-138`), which is
//!   [`LogitScaleUse`]'s missing third variant. But their blocker is not
//!   the multiplier: `command-r.cpp:66-119` feeds both branches the same
//!   normed input and sums `inpL + attn_out + ffn_out` once, over
//!   LayerNorm rather than RMSNorm. The multiplier work does not bring
//!   them closer.
//!
//! **`{arch}.attention.scale` does not live in this module's output.**
//! It resolves into the `ModelConfig::attention_scale` slot Gemma-27B
//! already uses, because that slot's contract -- "pre-scale Q and pass
//! 1.0 to the kernel" -- is exactly what llama.cpp's `kq_scale` needs
//! and having two fields for one number is the shape this repo keeps
//! paying for.

/// How an architecture's graph turns `{arch}.logit_scale` into a
/// multiplier on the lm_head's output.
///
/// The direction is a per-architecture fact with no key, so it is
/// resolved HERE and the decoder only ever multiplies. A third variant
/// (`AsIs`, Command-R's `ggml_scale(cur, f_logit_scale)`) is named in
/// the module header and deliberately absent until a row needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogitScaleUse {
    /// The graph never scales its logits.
    #[default]
    NotApplied,
    /// Granite / MiniCPM: `ggml_scale(cur, 1.0f / f_logit_scale)`
    /// (`granite.cpp:180`). The key is REQUIRED for both Granite rows.
    Reciprocal,
}

/// Which of the four multipliers this architecture's reference graph
/// applies -- and therefore which ferrox implements for it.
///
/// The same value drives BOTH halves: what the loader reads and applies,
/// and what [`crate::capability::unsupported_scaling_keys`] still
/// refuses. Deriving the refusal list from this struct is the point --
/// a hand-written second list is how ferrox once refused a key it
/// implemented and implemented a key it refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MultiplierSupport {
    /// `{arch}.embedding_scale` multiplies every token embedding row.
    pub embedding: bool,
    /// `{arch}.residual_scale` multiplies EVERY branch output before it
    /// rejoins the residual stream.
    pub residual: bool,
    /// What the graph does with `{arch}.logit_scale`.
    pub logit: LogitScaleUse,
    /// `{arch}.attention.scale` replaces the kernels' `1/sqrt(head_dim)`.
    pub attention: bool,
}

impl MultiplierSupport {
    /// Nothing declared and nothing applied: the generic decoder's own
    /// graph.
    pub const NONE: Self = Self {
        embedding: false,
        residual: false,
        logit: LogitScaleUse::NotApplied,
        attention: false,
    };

    /// `granite`, `granitemoe` and the `granite-moe` alias. One
    /// constant, so the three rows cannot disagree about the scaling
    /// they share.
    pub const GRANITE: Self = Self {
        embedding: true,
        residual: true,
        logit: LogitScaleUse::Reciprocal,
        attention: true,
    };
}

/// The GGUF architectures whose graph applies one or more of the four
/// multipliers, outside the Gemma family (which [`multiplier_support`]
/// keys off [`DecoderFamily::GemmaFamily`] instead of naming five
/// strings that would then have to be kept in step with the catalog).
///
/// `granite-moe` has no llama.cpp spelling -- `llama-arch.cpp:101` is
/// `granitemoe` -- and exists only because ferrox's catalog carries the
/// hyphenated alias. It is here so a file declaring it cannot get
/// different arithmetic from the row it is an alias FOR.
const MULTIPLIER_ARCHITECTURES: &[(&str, MultiplierSupport)] = &[
    ("granite", MultiplierSupport::GRANITE),
    ("granitemoe", MultiplierSupport::GRANITE),
    ("granite-moe", MultiplierSupport::GRANITE),
];

/// Which multipliers ferrox applies for `arch`.
///
/// This is about the KEYS, not about whether the architecture scales
/// anything. Gemma is the case that makes the distinction load-bearing:
/// it scales its embeddings and, at 27B, its attention scores, but it
/// reads neither key -- `gemma3.cpp:31` and `gemma2.cpp:27` ASSIGN
/// `f_attention_scale` from the model type, and the embedding scale is
/// `sqrt(n_embd)` computed in the graph. So a Gemma file declaring
/// `gemma3.embedding_scale` describes something llama.cpp does not do,
/// and ferrox refuses it here rather than honouring a number its own
/// reference ignores. The Gemma scales themselves come from
/// `capability::attention_scale_override` and `loader.rs`'s family
/// branch, which is where an arch-computed value belongs.
pub fn multiplier_support(arch: &str) -> MultiplierSupport {
    MULTIPLIER_ARCHITECTURES
        .iter()
        .find(|(n, _)| *n == arch)
        .map_or(MultiplierSupport::NONE, |(_, s)| *s)
}

/// The raw values a file declares, before llama.cpp's per-key sentinels
/// are applied.
///
/// Destructured exhaustively by [`resolve`] with no `..`, so a fifth
/// multiplier cannot be added here and silently ignored there.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DeclaredMultipliers {
    pub logit: Option<f32>,
    pub residual: Option<f32>,
    pub embedding: Option<f32>,
    pub attention: Option<f32>,
}

/// The resolved multipliers, in the form [`crate::ModelConfig`] carries
/// them: `None` means "this graph does not do that".
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ResolvedMultipliers {
    /// [`crate::ModelConfig::embedding_scale`].
    pub embedding_scale: Option<f32>,
    /// [`crate::ModelConfig::residual_scale`].
    pub residual_scale: Option<f32>,
    /// [`crate::ModelConfig::logit_multiplier`], already inverted where
    /// the architecture divides.
    pub logit_multiplier: Option<f32>,
    /// [`crate::ModelConfig::attention_scale`].
    pub attention_scale: Option<f32>,
}

/// Why a declared multiplier cannot be honoured.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiplierError {
    /// The architecture reads `{arch}.logit_scale` as REQUIRED
    /// (`granite.cpp:7`, `granite-moe.cpp:5`) and the file has no such
    /// key. llama.cpp throws on this file too.
    MissingRequiredLogitScale,
    /// A `logit_scale` of zero would divide by zero, and a negative one
    /// would REORDER the vocabulary -- which matters beyond the logits
    /// themselves, because a Metal decode stack is allowed to fold the
    /// lm_head and return an argmax id only while every post-head
    /// transform is monotone increasing.
    NonPositiveLogitScale(f32),
}

impl MultiplierError {
    /// The sentence the loader puts in its error, naming the key.
    pub fn message(&self, arch: &str) -> String {
        match self {
            MultiplierError::MissingRequiredLogitScale => format!(
                "`{arch}.logit_scale` is REQUIRED for this architecture (src/models/granite.cpp:7 \
                 reads it with no default) and the file does not declare it; llama.cpp refuses \
                 the same file"
            ),
            MultiplierError::NonPositiveLogitScale(v) => format!(
                "`{arch}.logit_scale` = {v}: the graph divides every logit by it \
                 (src/models/granite.cpp:180), so zero is a division by zero and a negative \
                 value reorders the vocabulary"
            ),
        }
    }
}

/// The no-op sentinel for `embedding_scale`, `residual_scale` and the
/// already-inverted `logit_scale`.
///
/// Two values are inert for these three, for two different reasons.
/// `1.0` is the arithmetic identity. `0.0` is llama.cpp's own "off":
/// `llama-graph.cpp:2337` tests `f_embedding_scale != 0.0f` and
/// `granite.cpp:235` tests `if (hparams.f_residual_scale)`, so a file
/// writing zero there means "do not scale" rather than "multiply
/// everything by zero", and reading it literally would blank the whole
/// residual stream.
///
/// **This is deliberately NOT applied to `attention.scale`**, and the
/// difference is the point. `f_attention_scale` uses `0.0` as its
/// "unset, use `1/sqrt(n_embd_head)`" sentinel (`granite.cpp:225`) and
/// `1.0` as a perfectly ordinary override -- llama.cpp passes it
/// straight to `build_attn` as `kq_scale`. Folding the two keys' rules
/// into one predicate silently dropped a declared `attention.scale` of
/// 1.0 while this module was being written, which is what
/// `each_keys_own_no_op_value_is_what_switches_it_off` is for.
fn scale_or_none(v: Option<f32>) -> Option<f32> {
    v.filter(|&v| v != 0.0 && v != 1.0)
}

/// Turn what the file declared into what the decoder applies.
///
/// `head_dim` is only used to drop an `attention.scale` that restates
/// the kernels' own `1/sqrt(head_dim)`: `ModelConfig::attention_scale`
/// means "pre-scale Q and pass 1.0 to the kernel", so restating the
/// default would be arithmetically identical but would fence the layer
/// off every fused Metal launch for nothing.
pub fn resolve(
    support: MultiplierSupport,
    declared: DeclaredMultipliers,
    head_dim: usize,
) -> Result<ResolvedMultipliers, MultiplierError> {
    let DeclaredMultipliers {
        logit,
        residual,
        embedding,
        attention,
    } = declared;

    let logit_multiplier = match support.logit {
        LogitScaleUse::NotApplied => None,
        LogitScaleUse::Reciprocal => {
            let v = logit.ok_or(MultiplierError::MissingRequiredLogitScale)?;
            if v <= 0.0 {
                return Err(MultiplierError::NonPositiveLogitScale(v));
            }
            // `1.0` inverts to `1.0`, which `scale_or_none` then drops:
            // a file declaring the identity gets the graph ferrox
            // already computes, with no needless multiply per token.
            scale_or_none(Some(1.0 / v))
        }
    };

    // `0.0` is this key's ONLY sentinel (`granite.cpp:225`); 1.0 is a
    // real override. See `scale_or_none`, which must not be used here.
    let attention_scale = if support.attention {
        attention.filter(|&v| v != 0.0).filter(|&v| {
            let kernel = 1.0 / (head_dim as f32).sqrt();
            (v - kernel).abs() > f32::EPSILON * kernel.max(1.0)
        })
    } else {
        None
    };

    Ok(ResolvedMultipliers {
        embedding_scale: support
            .embedding
            .then(|| scale_or_none(embedding))
            .flatten(),
        residual_scale: support.residual.then(|| scale_or_none(residual)).flatten(),
        logit_multiplier,
        attention_scale,
    })
}

/// `hidden += scale.unwrap_or(1.0) * branch`, the ONE residual add in
/// the generic decoder.
///
/// Every `hidden[i] += branch[i]` in `decoder.rs` goes through here, and
/// that is the whole point of the function existing. `residual_scale`
/// multiplies BOTH branch outputs of EVERY layer (`granite.cpp:235-238`,
/// `:288-292`), and `decoder.rs` spells the residual add out eighteen
/// times across prefill, decode, paged decode and continuous batching.
/// Eighteen hand-written adds that must all agree about one scalar is
/// precisely the shape that has cost this repo eight model features, so
/// the scalar is a parameter of a shared function rather than a rule
/// eighteen call sites are trusted to remember.
#[inline]
pub fn residual_add(hidden: &mut [f32], branch: &[f32], scale: Option<f32>) {
    debug_assert_eq!(hidden.len(), branch.len());
    match scale {
        None => {
            for (h, b) in hidden.iter_mut().zip(branch.iter()) {
                *h += *b;
            }
        }
        Some(s) => {
            for (h, b) in hidden.iter_mut().zip(branch.iter()) {
                *h += s * *b;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three Granite rows share one support constant, so they cannot
    /// be given different arithmetic by an edit to one of them.
    ///
    /// `granite-moe` is the row this matters most for: no llama.cpp GGUF
    /// spells it that way, so nothing outside ferrox would ever notice
    /// it drifting.
    #[test]
    fn the_three_granite_rows_have_identical_multiplier_support() {
        let dense = multiplier_support("granite");
        assert_eq!(dense, MultiplierSupport::GRANITE);
        for alias in ["granitemoe", "granite-moe"] {
            assert_eq!(
                multiplier_support(alias),
                dense,
                "`{alias}` must scale exactly like `granite`"
            );
        }
    }

    /// An architecture nobody read must apply NOTHING, so that adding a
    /// row to the catalog cannot silently start scaling it.
    #[test]
    fn an_architecture_that_was_not_read_applies_no_multipliers() {
        for arch in ["llama", "qwen3", "deepseek", "not-an-architecture"] {
            assert_eq!(
                multiplier_support(arch),
                MultiplierSupport::NONE,
                "`{arch}` must not scale"
            );
        }
    }

    /// Gemma reads NONE of the four keys, even though it scales two of
    /// the things they name.
    ///
    /// The distinction this pins is between "the architecture scales
    /// this" and "the architecture reads this key". Gemma's embedding
    /// scale is `sqrt(n_embd)` computed in the graph and its 27B
    /// attention scale is assigned from the model type
    /// (`gemma3.cpp:31`, `gemma2.cpp:27`), so a file declaring either
    /// key describes something llama.cpp does not do.
    ///
    /// The whole family used to be exempted from the refusal list
    /// wholesale, which meant a hand-written `gemma3.residual_scale`
    /// would have loaded and been silently ignored -- exactly the
    /// blind spot that list exists to close.
    #[test]
    fn the_gemma_family_reads_none_of_the_four_keys() {
        for arch in ["gemma", "gemma2", "gemma3"] {
            assert_eq!(
                multiplier_support(arch),
                MultiplierSupport::NONE,
                "`{arch}` computes its scales; it does not read them"
            );
        }
    }

    /// Granite DIVIDES by `logit_scale`; the config carries the already
    /// inverted multiplier so the decoder only ever multiplies.
    ///
    /// Getting the direction backwards is invisible in a smoke test --
    /// the logits are still finite, still ordered the same way, and only
    /// the temperature of the distribution moves.
    #[test]
    fn granites_logit_scale_is_inverted_at_load_time() {
        let got = resolve(
            MultiplierSupport::GRANITE,
            DeclaredMultipliers {
                logit: Some(8.0),
                ..Default::default()
            },
            64,
        )
        .expect("8.0 resolves");
        assert_eq!(got.logit_multiplier, Some(0.125));
    }

    /// The REQUIRED half of `logit_scale`, and the reason it is an error
    /// rather than a default of 1.0: llama.cpp cannot load such a file
    /// either, so silently running it would mean ferrox answering where
    /// its own reference refuses.
    #[test]
    fn a_granite_file_with_no_logit_scale_is_refused_rather_than_defaulted() {
        assert_eq!(
            resolve(
                MultiplierSupport::GRANITE,
                DeclaredMultipliers::default(),
                64
            ),
            Err(MultiplierError::MissingRequiredLogitScale)
        );
        assert!(
            MultiplierError::MissingRequiredLogitScale
                .message("granite")
                .contains("granite.logit_scale"),
            "the message must name the key"
        );
    }

    /// Zero divides by zero; a negative value reorders the vocabulary
    /// and would break the Metal decode stack's right to fold the
    /// lm_head into an argmax.
    #[test]
    fn a_non_positive_logit_scale_is_refused() {
        for bad in [0.0f32, -2.0] {
            assert_eq!(
                resolve(
                    MultiplierSupport::GRANITE,
                    DeclaredMultipliers {
                        logit: Some(bad),
                        ..Default::default()
                    },
                    64
                ),
                Err(MultiplierError::NonPositiveLogitScale(bad)),
                "logit_scale {bad} must be refused"
            );
        }
    }

    /// The two sentinels are different values and each key is judged
    /// against its own.
    ///
    /// A single "1.0 means off" rule would leave `attention.scale = 0.0`
    /// looking like a real override and pre-scale every Q by zero; a
    /// single "0.0 means off" rule would leave `residual_scale = 1.0`
    /// costing a multiply per element per branch per layer forever.
    #[test]
    fn each_keys_own_no_op_value_is_what_switches_it_off() {
        let got = resolve(
            MultiplierSupport::GRANITE,
            DeclaredMultipliers {
                logit: Some(1.0),
                residual: Some(1.0),
                embedding: Some(1.0),
                attention: Some(0.0),
            },
            64,
        )
        .expect("all no-ops resolve");
        assert_eq!(got, ResolvedMultipliers::default(), "{got:?}");

        // ... and the OTHER key's sentinel is not treated as a no-op.
        let got = resolve(
            MultiplierSupport::GRANITE,
            DeclaredMultipliers {
                logit: Some(2.0),
                residual: Some(0.0),
                embedding: Some(0.0),
                attention: Some(1.0),
            },
            64,
        )
        .expect("resolves");
        assert_eq!(
            got.residual_scale, None,
            "llama.cpp's `if (f_residual_scale)` guard makes 0.0 mean off"
        );
        assert_eq!(got.embedding_scale, None, "llama-graph.cpp:2337 likewise");
        assert_eq!(
            got.attention_scale,
            Some(1.0),
            "1.0 is a real attention-scale override, not its sentinel"
        );
    }

    /// An `attention.scale` that restates `1/sqrt(head_dim)` resolves to
    /// `None`.
    ///
    /// Arithmetically it makes no difference; operationally it does.
    /// `Some` here fences the whole model off every fused Metal
    /// attention launch (`Decoder::layer_supports_metal_attn`), so
    /// restating the default would cost a real checkpoint the GPU path
    /// for nothing.
    #[test]
    fn an_attention_scale_equal_to_the_kernels_own_resolves_to_none() {
        let head_dim = 64;
        let kernel = 1.0 / (head_dim as f32).sqrt();
        let got = resolve(
            MultiplierSupport::GRANITE,
            DeclaredMultipliers {
                logit: Some(2.0),
                attention: Some(kernel),
                ..Default::default()
            },
            head_dim,
        )
        .expect("resolves");
        assert_eq!(got.attention_scale, None);

        // A value that really differs survives.
        let got = resolve(
            MultiplierSupport::GRANITE,
            DeclaredMultipliers {
                logit: Some(2.0),
                attention: Some(0.015_625),
                ..Default::default()
            },
            head_dim,
        )
        .expect("resolves");
        assert_eq!(got.attention_scale, Some(0.015_625));
    }

    /// An architecture that does not apply a multiplier ignores the
    /// value even when the file declares it.
    ///
    /// It cannot reach here in practice -- the loader refuses such a
    /// file first -- but the two halves have to agree about which keys
    /// are live, and this is the half that says so in code.
    #[test]
    fn support_gates_the_value_rather_than_the_value_gating_itself() {
        let got = resolve(
            MultiplierSupport::NONE,
            DeclaredMultipliers {
                logit: Some(8.0),
                residual: Some(0.22),
                embedding: Some(12.0),
                attention: Some(0.015_625),
            },
            64,
        )
        .expect("an unsupported logit_scale is not even read");
        assert_eq!(got, ResolvedMultipliers::default());
    }

    /// The residual add, both arms, against arithmetic written out
    /// separately.
    #[test]
    fn the_residual_add_scales_the_branch_and_not_the_stream() {
        let mut hidden = vec![1.0f32, 2.0, 3.0];
        residual_add(&mut hidden, &[10.0, 20.0, 30.0], Some(0.5));
        assert_eq!(hidden, vec![6.0, 12.0, 18.0]);

        let mut hidden = vec![1.0f32, 2.0, 3.0];
        residual_add(&mut hidden, &[10.0, 20.0, 30.0], None);
        assert_eq!(
            hidden,
            vec![11.0, 22.0, 33.0],
            "no scale must be exactly the unscaled add, not a multiply by 1.0"
        );
    }
}
