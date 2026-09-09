//! Two decisions `ferrox quantize` has to make, and the refusals that
//! follow from them:
//!
//! 1. **Which target may be written at all.** ferrox READS every quant
//!    kind the engine runs and can WRITE four: Q8_0, Q4_K, Q5_K and
//!    Q6_K. A `quantize` subcommand whose name implies llama.cpp's
//!    range while it can emit four formats is the half-support this
//!    repo refuses, so a target ferrox cannot encode is refused BY
//!    NAME, with what it CAN write spelled out.
//!
//!    The `_S`/`_M` names are MIXES, not block formats: `Q4_K_M`
//!    promotes `output.weight` to Q6_K and some layers' `attn_v` and
//!    `ffn_down` to Q6_K. Those mixes used to be refused unless
//!    `--pure` said to write the uniform file instead, because ferrox
//!    had no Q5_K or Q6_K encoder. It has both now, so the mixes are
//!    written for real; [`super::recipe`] is the transcription, and
//!    `--pure` survives as what it is upstream -- an option to skip
//!    the mix, not a workaround for a missing encoder.
//! 2. **Which tensors get quantized.** llama.cpp keeps a specific set
//!    at source precision, and the set is not obvious: it is not "the
//!    small ones", it is a list of tensors whose values are used as
//!    something other than matrix rows (norms, router gates, position
//!    tables, conv kernels). Getting it wrong produces a file that
//!    loads and is subtly wrong, so the list is transcribed from
//!    `llama.cpp/src/llama-quant.cpp`'s `tensor_allows_quantization`
//!    with its line order intact rather than reinvented.

use ferrox_gguf::GgmlType;

/// The quantization targets `ferrox quantize` can actually encode.
///
/// Four encoders, six names: `Q4_K_S` and `Q4_K_M` start from the same
/// Q4_K blocks (as `Q5_K_S` and `Q5_K_M` do from Q5_K) and differ in
/// the MIX they apply on top -- see [`super::recipe`] -- and in the
/// `general.file_type` they record.
///
/// It is an enum rather than a string so that adding the next one means
/// adding the match arms the compiler asks for -- in [`Target::name`],
/// [`Target::ggml_type`], [`Target::llama_ftype`],
/// [`Target::fallback_note`] and `quantize::encode_row` -- instead of a
/// name landing in the CLI's help text
/// ahead of a kernel, which is how a format gets "supported" in a table
/// and nowhere else.
// The variants are spelled the way `llama-quantize --type` spells
// them, upper-case and underscored. `Q4KS` would be the camel-case
// name and would be one more place where ferrox's word for a quant and
// the ecosystem's differ.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Q8_0,
    Q4_K_S,
    Q4_K_M,
    Q5_K_S,
    Q5_K_M,
    Q6_K,
}

impl Target {
    /// Every target this build can write. The refusal message below is
    /// generated from this, so it cannot fall out of date with the
    /// encoder the way a hand-written "we support: Q8_0" string would.
    pub const ALL: &'static [Target] = &[
        Target::Q8_0,
        Target::Q4_K_S,
        Target::Q4_K_M,
        Target::Q5_K_S,
        Target::Q5_K_M,
        Target::Q6_K,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Target::Q8_0 => "Q8_0",
            Target::Q4_K_S => "Q4_K_S",
            Target::Q4_K_M => "Q4_K_M",
            Target::Q5_K_S => "Q5_K_S",
            Target::Q5_K_M => "Q5_K_M",
            Target::Q6_K => "Q6_K",
        }
    }

    /// The block format a tensor gets unless the mix promotes it.
    pub fn ggml_type(self) -> GgmlType {
        match self {
            Target::Q8_0 => GgmlType::Q8_0,
            Target::Q4_K_S | Target::Q4_K_M => GgmlType::Q4K,
            Target::Q5_K_S | Target::Q5_K_M => GgmlType::Q5K,
            Target::Q6_K => GgmlType::Q6K,
        }
    }

    /// llama.cpp's `general.file_type` (`LLAMA_FTYPE_MOSTLY_*`) value,
    /// so a file ferrox writes reports its mix the way every other tool
    /// in the ecosystem reads it.
    pub fn llama_ftype(self) -> u32 {
        match self {
            Target::Q8_0 => 7,    // LLAMA_FTYPE_MOSTLY_Q8_0
            Target::Q4_K_S => 14, // LLAMA_FTYPE_MOSTLY_Q4_K_S
            Target::Q4_K_M => 15, // LLAMA_FTYPE_MOSTLY_Q4_K_M
            Target::Q5_K_S => 16, // LLAMA_FTYPE_MOSTLY_Q5_K_S
            Target::Q5_K_M => 17, // LLAMA_FTYPE_MOSTLY_Q5_K_M
            Target::Q6_K => 18,   // LLAMA_FTYPE_MOSTLY_Q6_K
        }
    }

    /// What llama.cpp does with a tensor whose row length is not a
    /// multiple of this target's block size -- which is the difference
    /// between "llama.cpp stops here too" and "llama.cpp quietly writes
    /// a different type". Carried per target, because one sentence
    /// covering both was true of only Q8_0.
    pub fn fallback_note(self) -> &'static str {
        match self {
            // `tensor_type_fallback` has no Q8_0 arm: it throws.
            Target::Q8_0 => "llama.cpp has no fallback type for Q8_0 either -- it stops here too.",
            // `convert_incompatible_tensor`: Q4_K -> Q5_0, and then
            // -> F16 if the row is not a multiple of 32 either.
            Target::Q4_K_S | Target::Q4_K_M => {
                "llama.cpp answers this by changing the tensor's TYPE (Q4_K -> Q5_0, or F16 if the \
                 row is not a multiple of 32 either); ferrox can write neither, so it stops rather \
                 than write a file whose name says Q4_K."
            }
            // Q5_K -> Q5_1 -> F16.
            Target::Q5_K_S | Target::Q5_K_M => {
                "llama.cpp answers this by changing the tensor's TYPE (Q5_K -> Q5_1, or F16 if the \
                 row is not a multiple of 32 either); ferrox can write neither, so it stops rather \
                 than write a file whose name says Q5_K."
            }
            // Q6_K -> Q8_0 -> F16. ferrox HAS a Q8_0 encoder, and
            // still stops: the fallback also bumps llama.cpp's
            // `n_fallback`, which changes how a `--tensor-type`
            // override is applied, and a quantizer that silently wrote
            // a different type for one tensor than the plan it printed
            // is the failure this whole subcommand is shaped around.
            // Following the fallback is a change to the PLANNER, with
            // its own receipt, not something the encoder does quietly.
            Target::Q6_K => {
                "llama.cpp answers this by changing the tensor's TYPE (Q6_K -> Q8_0, or F16 if the \
                 row is not a multiple of 32 either); ferrox stops rather than write a tensor \
                 whose type disagrees with the plan it printed."
            }
        }
    }
}

/// Every name `llama-quantize` accepts, lowercased, paired with the
/// ferrox encoder that writes it -- `None` where there is none.
///
/// Transcribed from `tools/quantize/quantize.cpp`'s `QUANT_OPTIONS`,
/// including its aliasing: `q4_k` is listed there as "alias for Q4_K_M"
/// and carries `LLAMA_FTYPE_MOSTLY_Q4_K_M`, so it maps to the same
/// ferrox target and records the same `general.file_type`.
///
/// ONE table, not two. The shape here used to be a list of names plus a
/// separate `Target::ALL` lookup, and the two agreeing was nobody's
/// job; a name spelled differently in the two places refuses a target
/// this build can write, or accepts one it cannot.
const LLAMA_CPP_TARGETS: &[(&str, Option<Target>)] = &[
    ("q1_0", None),
    ("q2_0", None),
    ("q4_0", None),
    ("q4_1", None),
    ("mxfp4_moe", None),
    ("q5_0", None),
    ("q5_1", None),
    ("iq2_xxs", None),
    ("iq2_xs", None),
    ("iq2_s", None),
    ("iq2_m", None),
    ("iq1_s", None),
    ("iq1_m", None),
    ("tq1_0", None),
    ("tq2_0", None),
    ("q2_k", None),
    ("q2_k_s", None),
    ("iq3_xxs", None),
    ("iq3_s", None),
    ("iq3_m", None),
    ("q3_k", None),
    ("iq3_xs", None),
    ("q3_k_s", None),
    ("q3_k_m", None),
    ("q3_k_l", None),
    ("iq4_nl", None),
    ("iq4_xs", None),
    ("q4_k", Some(Target::Q4_K_M)),
    ("q4_k_s", Some(Target::Q4_K_S)),
    ("q4_k_m", Some(Target::Q4_K_M)),
    ("q5_k", Some(Target::Q5_K_M)),
    ("q5_k_s", Some(Target::Q5_K_S)),
    ("q5_k_m", Some(Target::Q5_K_M)),
    ("q6_k", Some(Target::Q6_K)),
    ("q8_0", Some(Target::Q8_0)),
    ("f16", None),
    ("bf16", None),
    ("f32", None),
    ("copy", None),
];

/// Why a requested target was refused. Two arms, because two very
/// different things go wrong and the user needs to know which: a gap in
/// ferrox, and a typo.
///
/// A third arm used to live here, `MixNeedsPure`: ferrox could encode
/// Q4_K blocks but not the Q5_K and Q6_K its MIX promotes some tensors
/// to, so the mix names were admitted only under `--pure`. Both
/// encoders exist now and [`super::recipe`] applies the mix, so the
/// refusal is DELETED rather than kept as a condition nothing can trip
/// -- a gate that cannot fire reads as coverage.
#[derive(Debug, PartialEq, Eq)]
pub enum TargetRefusal {
    /// A real llama.cpp target that ferrox has no encoder for.
    NotWritableYet(String),
    /// Not a quantization type at all.
    Unknown(String),
}

impl std::fmt::Display for TargetRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let writable = writable_targets();
        match self {
            TargetRefusal::NotWritableYet(name) => write!(
                f,
                "ferrox cannot WRITE {name} yet. It can read {name} and run it; it has no encoder \
                 for it.\n\
                 `ferrox quantize` writes: {writable}.\n\
                 The remaining K-quant and IQ encoders are each an iterative per-super-block \
                 scale/min fit (and, for the IQ tiers, a lattice search over a codebook). A \
                 min/max approximation of one produces a file that loads and generates measurably \
                 worse text, so ferrox stops here instead of writing it. Use llama.cpp's \
                 `llama-quantize --type {name}` for now; ferrox reads what it produces."
            ),
            TargetRefusal::Unknown(name) => write!(
                f,
                "'{name}' is not a quantization type. `ferrox quantize` writes: {writable}."
            ),
        }
    }
}

/// What `ferrox quantize` can write, in the form a user has to type it.
/// Generated from [`Target::ALL`], so no message can promise a target
/// the encoder dispatch does not have.
///
/// It is the one string every refusal quotes, which is why it is
/// derived and not restated: when Q4_K landed, a restated version would
/// still have advertised Q8_0 alone with nothing going red.
pub fn writable_targets() -> String {
    Target::ALL
        .iter()
        .map(|t| t.name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Parses a `--type` argument, case-insensitively, the way
/// `llama-quantize` does, and admits it only if this build can actually
/// produce that file.
///
/// Every reason a target can be refused lives in this one function, so
/// a second call site cannot admit a target by forgetting one of them.
pub fn parse_target(raw: &str) -> Result<Target, TargetRefusal> {
    let lower = raw.to_ascii_lowercase();
    let Some((_, encoder)) = LLAMA_CPP_TARGETS.iter().find(|(n, _)| *n == lower) else {
        return Err(TargetRefusal::Unknown(raw.to_string()));
    };
    let Some(target) = encoder else {
        return Err(TargetRefusal::NotWritableYet(raw.to_string()));
    };
    Ok(*target)
}

/// What happens to one tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Disposition {
    /// Re-encode to this type. It is the MIX's choice for this tensor,
    /// not the target's block format: under `Q4_K_M` a `ffn_down` on
    /// one layer carries `Q6K` here and its neighbour `Q4K`.
    Quantize(GgmlType),
    /// Copy the source bytes through unchanged, for the stated reason.
    Copy(&'static str),
}

/// Substring rules that keep a tensor at source precision, transcribed
/// in order from `tensor_allows_quantization` in llama.cpp's
/// `src/llama-quant.cpp`, each with the reason llama.cpp gives.
///
/// A single table, matched by one loop, rather than a chain of
/// hand-written `if`s: a chain is where one condition quietly stops
/// being checked. The reason string is not decoration -- `ferrox
/// quantize` prints it per tensor, so the file's composition is
/// auditable without reading this source.
const KEEP_AT_SOURCE_PRECISION: &[(&str, &str)] = &[
    ("_norm.weight", "norm"),
    ("ffn_gate_inp.weight", "expert gating"),
    (
        "ffn_gate_tid2eid.weight",
        "token-id -> expert-id routing table",
    ),
    ("altup", "tiny"),
    ("laurel", "tiny"),
    ("per_layer_model_proj", "small"),
    ("position_embd.weight", "positional embedding"),
    ("token_types.weight", "token types"),
    ("ssm_conv1d", "conv1d kernel"),
    ("shortconv.conv.weight", "conv kernel"),
    ("indexer.k_proj.weight", "tiny"),
    ("indexer.q_proj.weight", "tiny"),
    ("time_mix_first.weight", "RWKV small 2D"),
    ("time_mix_w0.weight", "RWKV small 2D"),
    ("time_mix_w1.weight", "RWKV small 2D"),
    ("time_mix_w2.weight", "RWKV small 2D"),
    ("time_mix_v0.weight", "RWKV small 2D"),
    ("time_mix_v1.weight", "RWKV small 2D"),
    ("time_mix_v2.weight", "RWKV small 2D"),
    ("time_mix_a0.weight", "RWKV small 2D"),
    ("time_mix_a1.weight", "RWKV small 2D"),
    ("time_mix_a2.weight", "RWKV small 2D"),
    ("time_mix_g1.weight", "RWKV small 2D"),
    ("time_mix_g2.weight", "RWKV small 2D"),
    ("time_mix_decay_w1.weight", "RWKV small 2D"),
    ("time_mix_decay_w2.weight", "RWKV small 2D"),
    ("time_mix_lerp_fused.weight", "RWKV small 2D"),
    ("attn_rel_b.weight", "relative position bias"),
    (".position_embd", "positional embedding"),
    ("sam.pos_embd", "multimodal"),
    ("sam.neck.", "multimodal"),
    ("sam.net_", "multimodal"),
    (".rel_pos", "multimodal"),
    (".patch_embd", "multimodal"),
    (".patch_merger", "multimodal"),
    ("a.rvq.codebook", "audio codebook"),
    ("mm.a.code_embd", "audio codebook"),
];

/// ggml's `ggml_n_dims`: the number of dimensions after trailing 1s are
/// dropped, floored at 1. GGUF stores the declared dims, so a `[4096,
/// 1]` tensor is 1-D to llama.cpp and must be to ferrox too, or the two
/// tools disagree about which tensors are quantized.
fn ggml_n_dims(shape: &[u64]) -> usize {
    for i in (1..shape.len()).rev() {
        if shape[i] > 1 {
            return i + 1;
        }
    }
    1
}

/// llama.cpp's `tensor_allows_quantization`: `None` if the tensor is
/// eligible, `Some(reason)` if it stays at source precision.
///
/// Split out from [`disposition`] because it is the predicate that
/// decides whether [`super::recipe::Recipe::tensor_type`] is called at
/// all, and those counters are positional: calling it for one extra
/// tensor shifts every later layer's promotion by one. Two spellings
/// of "is this tensor eligible" -- one here and one at the recipe's
/// call site -- is the shape this repo pays for, so there is one.
///
/// `--pure` does NOT skip this list. A norm quantized to Q4_K is a
/// broken model whichever mix asked for it.
pub fn allows_quantization(name: &str, shape: &[u64]) -> Option<&'static str> {
    if ggml_n_dims(shape) < 2 {
        return Some("1-D");
    }
    if !name.ends_with("weight") {
        return Some("not a weight");
    }
    for (needle, reason) in KEEP_AT_SOURCE_PRECISION {
        if name.contains(needle) {
            return Some(reason);
        }
    }
    None
}

/// What `ferrox quantize` does with one tensor, given the type the mix
/// already chose for it.
///
/// `chosen` comes from [`super::recipe::Recipe::tensor_type`] (or from
/// the target's block format under `--pure`), never from `target`
/// directly. That separation is the whole point: llama.cpp decides the
/// type first and only then asks whether re-encoding is a no-op
/// (`quantize = cur_type != new_type`), and folding the two together is
/// what let an earlier version of this file believe every mix was
/// uniform.
pub fn disposition(dtype: GgmlType, chosen: GgmlType) -> Disposition {
    if dtype == chosen {
        return Disposition::Copy("already the target type");
    }
    Disposition::Quantize(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal the whole subcommand is scoped around. A user who
    /// types a quant ferrox has no encoder for must be told so -- not
    /// handed a plausible-looking file.
    #[test]
    fn a_target_with_no_encoder_is_refused_by_name_and_says_what_can_be_written() {
        // Q6_K used to be the example here and is now writable, which
        // is the point of this change. Q3_K_M is the next mix up the
        // same family with no encoder.
        let err = parse_target("q3_k_m").unwrap_err();
        assert_eq!(err, TargetRefusal::NotWritableYet("q3_k_m".into()));
        let msg = err.to_string();
        assert!(msg.contains("cannot WRITE q3_k_m"), "{msg}");
        // What it CAN write, generated from `Target::ALL`.
        assert!(msg.contains("writes: Q8_0"), "{msg}");
        // And it must say ferrox STOPS, not that it did something
        // approximate: a message that leaves the door open is how a
        // user ends up with a file that loads and is worse.
        assert!(msg.contains("stops here instead of writing it"), "{msg}");
    }

    /// The K-quant MIX names are admitted outright now. They used to
    /// need `--pure`, because ferrox could encode Q4_K blocks and not
    /// the Q5_K / Q6_K the mix promotes some tensors to. Both encoders
    /// exist, so this asserts the refusal is GONE rather than that it
    /// still fires with a nicer message -- a stale refusal is a target
    /// a user cannot reach for no reason.
    #[test]
    fn the_k_quant_mixes_no_longer_need_pure() {
        for name in [
            "q4_k", "q4_k_s", "q4_k_m", "q5_k", "q5_k_s", "q5_k_m", "q6_k",
        ] {
            assert!(parse_target(name).is_ok(), "{name}");
        }
    }

    /// `q4_k` is an alias for `q4_k_m` in `llama-quantize`, and an
    /// alias that resolved to a different `general.file_type` would
    /// make the same command produce two different files depending on
    /// which tool ran it.
    #[test]
    fn q4_k_is_the_same_target_as_q4_k_m_the_way_llama_quantize_aliases_it() {
        assert_eq!(parse_target("q4_k").unwrap(), Target::Q4_K_M);
        assert_eq!(parse_target("q4_k_m").unwrap(), Target::Q4_K_M);
        assert_eq!(parse_target("q4_k_s").unwrap(), Target::Q4_K_S);
        // Same blocks, different declared mix. Both halves matter: the
        // first is why one encoder serves both, the second is why they
        // are two variants and not one.
        assert_eq!(
            Target::Q4_K_S.ggml_type(),
            Target::Q4_K_M.ggml_type(),
            "one encoder serves both"
        );
        assert_ne!(
            Target::Q4_K_S.llama_ftype(),
            Target::Q4_K_M.llama_ftype(),
            "and they must not report the same mix"
        );
    }

    /// Every llama.cpp target ferrox has no encoder for refuses, and
    /// none of them refuses as "unknown" -- a gap and a typo are
    /// different problems and get different messages.
    #[test]
    fn every_llama_cpp_target_ferrox_cannot_write_refuses_as_a_gap_not_a_typo() {
        for (name, encoder) in LLAMA_CPP_TARGETS {
            let parsed = parse_target(name);
            match encoder {
                Some(t) => assert_eq!(parsed.as_ref().ok(), Some(t), "{name} should be writable"),
                None => assert_eq!(
                    parsed.unwrap_err(),
                    TargetRefusal::NotWritableYet((*name).to_string()),
                    "{name}"
                ),
            }
        }
    }

    /// The table and the enum must agree in BOTH directions. One
    /// direction is the interesting one: a `Target` missing from the
    /// table is a target `ferrox quantize --type` can never reach, so
    /// it would sit in the help text as a target nobody can select.
    #[test]
    fn the_target_enum_and_the_llama_cpp_name_table_cover_each_other() {
        for t in Target::ALL {
            assert!(
                LLAMA_CPP_TARGETS
                    .iter()
                    .any(|(n, e)| *n == t.name().to_ascii_lowercase() && *e == Some(*t)),
                "{} is not reachable from the llama-quantize name table",
                t.name()
            );
        }
        for (name, encoder) in LLAMA_CPP_TARGETS {
            if let Some(t) = encoder {
                assert!(
                    Target::ALL.contains(t),
                    "{name} maps to a target that is not in Target::ALL"
                );
            }
        }
    }

    /// The one-line summary a user sees in every refusal and in
    /// `--help` is derived, not restated. If it were restated, adding
    /// Q4_K without editing it would have advertised Q8_0 only, and
    /// nothing would have gone red.
    #[test]
    fn the_writable_summary_names_every_target() {
        let s = writable_targets();
        for t in Target::ALL {
            assert!(s.contains(t.name()), "{s} is missing {}", t.name());
        }
    }

    #[test]
    fn a_name_that_is_not_a_quant_at_all_says_so() {
        let err = parse_target("q4_k_ultra").unwrap_err();
        assert_eq!(err, TargetRefusal::Unknown("q4_k_ultra".into()));
        assert!(err.to_string().contains("is not a quantization type"));
    }

    #[test]
    fn target_names_are_case_insensitive_like_llama_quantize() {
        assert_eq!(parse_target("q8_0").unwrap(), Target::Q8_0);
        assert_eq!(parse_target("Q8_0").unwrap(), Target::Q8_0);
        assert_eq!(parse_target("Q4_K_M").unwrap(), Target::Q4_K_M);
        assert_eq!(
            parse_target("Q8_o").unwrap_err(),
            TargetRefusal::Unknown("Q8_o".into())
        );
    }

    /// `ggml_n_dims` drops trailing 1s. A `[n, 1]` tensor is 1-D to
    /// llama.cpp; treating it as 2-D would quantize a tensor llama.cpp
    /// leaves alone and the two files would differ.
    #[test]
    fn trailing_unit_dimensions_do_not_make_a_tensor_two_dimensional() {
        assert_eq!(ggml_n_dims(&[4096]), 1);
        assert_eq!(ggml_n_dims(&[4096, 1]), 1);
        assert_eq!(ggml_n_dims(&[4096, 1, 1]), 1);
        assert_eq!(ggml_n_dims(&[4096, 11008]), 2);
        assert_eq!(ggml_n_dims(&[4096, 1, 8]), 3);
    }

    /// llama.cpp's keep-list, which `--pure` does not skip. This is the
    /// predicate that also decides whether the mix's counters advance,
    /// so a name landing on the wrong side of it shifts every later
    /// layer's promotion as well as changing one tensor's type.
    #[test]
    fn the_tensors_llama_cpp_keeps_at_source_precision_are_kept() {
        let two_d = [4096u64, 4096];
        let cases: &[(&str, bool)] = &[
            ("blk.0.attn_q.weight", true),
            ("blk.0.ffn_down.weight", true),
            ("token_embd.weight", true),
            ("output.weight", true),
            ("blk.0.attn_norm.weight", false),
            ("output_norm.weight", false),
            ("blk.0.ffn_gate_inp.weight", false),
            ("blk.0.ssm_conv1d.weight", false),
            ("position_embd.weight", false),
            ("token_types.weight", false),
            ("blk.0.altup_proj.weight", false),
            ("blk.0.attn_q.bias", false),
            ("v.patch_embd.weight", false),
        ];
        for (name, want_quantized) in cases {
            let got = allows_quantization(name, &two_d);
            assert_eq!(got.is_none(), *want_quantized, "{name} -> {got:?}");
        }
        // A 1-D tensor is kept whatever its name.
        assert_eq!(
            allows_quantization("blk.0.attn_q.weight", &[4096]),
            Some("1-D")
        );
    }

    /// Requantizing is not what this subcommand is for, but a tensor
    /// that is already the target type must not be re-encoded either:
    /// llama.cpp skips it (`quantize = cur_type != new_type`) and
    /// re-encoding it would need a decoder in the middle.
    #[test]
    fn a_tensor_already_in_the_target_type_is_copied_not_re_encoded() {
        assert_eq!(
            disposition(GgmlType::Q8_0, GgmlType::Q8_0),
            Disposition::Copy("already the target type")
        );
        assert_eq!(
            disposition(GgmlType::F16, GgmlType::Q6K),
            Disposition::Quantize(GgmlType::Q6K)
        );
    }
}
