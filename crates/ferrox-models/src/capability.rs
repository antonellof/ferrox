//! Explicit architecture capability registry for the generic GGUF path.
//!
//! Mirrors the pinned llama.cpp `llm_arch` / `LLM_ARCH_NAMES` inventory
//! (`.scratch/llama.cpp/src/llama-arch.{h,cpp}`) with Ferrox-side
//! classification into decoder families, memory kinds, and scope.
//! Unknown strings and detected-but-unimplemented features fail closed
//! (`LoadError`) instead of silently defaulting into fluent-but-wrong
//! logits.
//!
//! Architecture names are registry keys only. Hot-path kernels never
//! branch on them; load-time resolution produces an [`ArchProfile`]
//! whose fields the decoder reads as plain data.

use crate::config::RopeLayout;

/// How far this architecture is in Ferrox's delivery scope (plan:
/// text-generation parity; encoder/multimodal/diffusion/audio deferred).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchScope {
    /// Autoregressive / encoder-decoder text generation -- in scope.
    TextGeneration,
    /// Encoder / embedding / pooling models -- deferred.
    DeferredEncoderEmbedding,
    /// Vision / multimodal projector paths -- deferred.
    DeferredMultimodal,
    /// Diffusion / masked-LM samplers -- deferred.
    DeferredDiffusion,
    /// Audio tokenizers / codecs -- deferred.
    DeferredAudio,
    /// Enum present in llama.cpp but not a real serve target here.
    EnumOnly,
}

/// Shared execution family (maps many GGUF strings onto one engine path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderFamily {
    /// Standard GQA (+ optional MoE) with whole-vector optional QK-norm.
    StandardGqa,
    /// Qwen3-style: explicit head_dim + per-head Q/K RMSNorm before RoPE.
    Qwen3Family,
    /// Gemma-family: embedding scale, post-norms, softcap, SWA pattern, GeGLU.
    GemmaFamily,
    /// Phi-family: fused QKV and/or fused gate+up SwiGLU.
    PhiFamily,
    /// DeepSeek-2 / Mistral4 MLA (not generic GQA).
    Mla,
    /// Attn + SSM / delta-net hybrids.
    Hybrid,
    /// Pure recurrent (Mamba / RWKV) -- no KV cache.
    Recurrent,
    /// T5-style encoder-decoder.
    EncoderDecoder,
    /// Dedicated Ferrox stacks (GLM DSA, DeepSeek V4, Kimi).
    Dedicated,
    /// In-repo synthetic fixtures.
    TestFixture,
}

/// Memory / KV backend selected once at load (llama.cpp `create_memory`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    KvGqa,
    KvIswa,
    KvMla,
    KvDsa,
    KvDsv4,
    Recurrent,
    Hybrid,
    None,
}

/// How `attn_q_norm` / `attn_k_norm` weights are applied (when present).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QkNormStyle {
    /// OLMoE: one RMSNorm over the full Q/K projection width.
    #[default]
    WholeVector,
    /// Qwen3 / Gemma3: RMSNorm per head with weight length `head_dim`.
    PerHead,
}

/// How much work admitting one UNAUDITED architecture to the generic
/// path would actually be.
///
/// Every architecture on the generic path that is not in
/// [`AUDITED_GENERIC_GQA`] refuses with
/// `LoadError::UnauditedArchitecture`, and that message used to say the
/// same thing for all 47 of them. It hid a real difference:
/// `bailingmoe2` needs a test fixture and nothing else, `deepseek` needs
/// one name added to one list, and `olmo2` needs a decoder that can skip
/// the two pre-norms it does not have. A user reading "nobody has
/// checked this" cannot tell a one-line fix from a new attention
/// implementation.
///
/// **A verdict here is a reading of BOTH trees, never a guess.** Every
/// non-[`TriageClass::Unknown`] verdict names the `src/models/*.cpp`
/// line that decides it and the ferrox file that would change.
/// `Unknown` is a legitimate answer and says what would settle it. The
/// precedent this rule exists for: four architectures in this very file
/// once refused while naming a blocker that was not the real one --
/// `glm4moe` was told it lacked an MLA hyper-parameter it must not have,
/// and `minimax-m2` was blamed on MTP weights no converter can emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriageClass {
    /// Ferrox already implements everything this architecture needs.
    /// What is missing is EVIDENCE: a fixture, or a parity run against
    /// llama.cpp on a real checkpoint.
    FixtureAway,
    /// One small, nameable piece is missing: an activation, a norm slot,
    /// a routing flag, an ordering. Nameable is the bar -- if the blocker
    /// cannot be written as a sentence naming the thing, it is not this
    /// class.
    OneMatchArm,
    /// A different attention or residual structure: a norm the decoder
    /// unconditionally applies and this model does not have, a scaled
    /// residual, ALiBi, MLA, block-sparse, recurrent, hybrid.
    NewCode,
    /// Not decidable from reading the two trees. The blocker says what
    /// would settle it.
    Unknown,
}

impl TriageClass {
    /// Short slug used in the refusal message.
    pub fn label(self) -> &'static str {
        match self {
            TriageClass::FixtureAway => "FIXTURE-AWAY",
            TriageClass::OneMatchArm => "ONE MATCH ARM",
            TriageClass::NewCode => "NEW CODE",
            TriageClass::Unknown => "UNKNOWN",
        }
    }

    /// One sentence saying what the class means, so the message stands
    /// alone without this doc comment.
    pub fn headline(self) -> &'static str {
        match self {
            TriageClass::FixtureAway => {
                "ferrox already implements everything this architecture needs; what is \
                 missing is EVIDENCE, not capability"
            }
            TriageClass::OneMatchArm => {
                "one small, named piece is missing -- an activation, a norm slot, a \
                 routing flag or an ordering"
            }
            TriageClass::NewCode => {
                "a different attention or residual structure than the generic decoder \
                 computes; this is not a fixture away"
            }
            TriageClass::Unknown => {
                "reading both trees did not settle this one; the note below says what \
                 would"
            }
        }
    }
}

/// One architecture's triage verdict, carried on its own catalog row.
///
/// Deliberately NOT a second table keyed by architecture name. This repo
/// has fixed three separate bugs caused by two structures disagreeing
/// about the same architecture, so the verdict lives on the
/// [`ArchProfile`] the loader already resolves, and
/// `every_unaudited_generic_architecture_is_triaged_or_listed_as_pending`
/// pins that no generic row can exist without one or the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnauditedTriage {
    pub class: TriageClass,
    /// What is missing, with the llama.cpp `src/models/*.cpp` line that
    /// decides it and the ferrox file that would change.
    pub blocker: &'static str,
}

/// Unaudited generic-path architectures nobody has read against
/// llama.cpp's graph yet.
///
/// This is a TO-DO, not cover. A name here means the refusal honestly
/// says "not triaged" rather than inventing a class; a name leaves this
/// list only by gaining an [`UnauditedTriage`] on its catalog row, and
/// the two tests below make it impossible for a name to be on both or on
/// neither.
pub const TRIAGE_PENDING: &[&str] = &[
    // Norm-RoPE group.
    // NEOX-RoPE group.
];

/// This architecture's triage verdict, or `None` when it has not been
/// triaged (see [`TRIAGE_PENDING`]) or does not need one.
pub fn unaudited_triage(arch: &str) -> Option<UnauditedTriage> {
    resolve_profile(arch).and_then(|p| p.triage)
}

/// The triage half of the `UnauditedArchitecture` refusal, rendered for
/// the user.
///
/// Appended to the generic "nobody has verified this" sentence so the
/// message says which of the three classes the architecture is in and
/// what specifically is missing, rather than the same paragraph for all
/// 47.
pub fn unaudited_refusal_detail(arch: &str) -> String {
    match unaudited_triage(arch) {
        Some(t) => format!(
            "TRIAGE ({}): {}. {}.",
            t.class.label(),
            t.class.headline(),
            t.blocker
        ),
        None => format!(
            "TRIAGE: not done for `{arch}` yet -- nobody has read llama.cpp's \
             src/models/*.cpp for it against the generic decoder, so this refusal names \
             no blocker and you should not read it as one. Triaging the remaining \
             architectures is docs/plans/llama-cpp-gap-inventory.md section 8, item 6."
        ),
    }
}

/// Architectures on the shared generic-GQA path that somebody has
/// actually PROVEN, and the evidence for each.
///
/// The generic path is a guess: it assumes an architecture is plain GQA
/// because nothing said otherwise. That guess has already been wrong
/// five times. `gpt2`, `mpt`, `refact`, `bloom` and `jais` all sat here
/// computing ALiBi or learned absolute position embeddings as though
/// they were NEOX RoPE, and every downstream guard missed them: two
/// hardcode their ALiBi slope with no GGUF key, one leaves no unread
/// tensor, and the RoPE pin excluded their group by construction.
///
/// So membership here is not "we think this works", it is "there is a
/// benchmark row, a pinned logit comparison against llama.cpp, or a
/// fixture". Everything else on the generic path is UNAUDITED and says
/// so at load time rather than running and hoping.
///
/// Adding a name here without evidence defeats the entire point.
pub const AUDITED_GENERIC_GQA: &[&str] = &[
    // Bench rows in benchmarks/suite.json, measured against llama.cpp
    // on the same host and file.
    "llama",    // TinyLlama, Mistral, Mixtral, SmolLM2, Llama-3.x all tag llama
    "qwen2",    // Qwen2.5-0.5B
    "qwen2moe", // Qwen1.5-MoE-A2.7B
    "qwen3",    // Qwen3-0.6B
    "olmoe",    // OLMoE-1B-7B
    "gemma2",   // Gemma-2-2B
    "gemma3",   // Gemma-3-1B
    "phi3",     // Phi-4-mini tags phi3
    // Pinned against real libllama logits in tests/.
    "gpt-oss",
    "dots1",
    // tests/qwen3moe_graph.rs: a synthetic 2-layer fixture
    // (scripts/make_qwen3moe_fixture.py) compared against llama.cpp's
    // own qwen3moe graph via libllama, on all three forward paths.
    // Carries per-head QK norm before RoPE, head_dim * n_head != n_embd,
    // GQA, NEOX RoPE, softmax gating with renormalised top-k, and
    // n_ff != n_ff_exp.
    "qwen3moe",
    // tests/one_match_arm_graphs.rs: five architectures that were
    // triaged ONE MATCH ARM, each admitted with the same evidence
    // qwen3moe has -- a synthetic fixture whose golden logits come from
    // llama.cpp's own graph via libllama, checked on all three forward
    // paths. The arm each one needed is named beside it; every fixture
    // is built so that getting that arm wrong moves the logits by orders
    // of magnitude more than the comparison tolerance.
    //
    // `deepseek` (V1, not the MLA deepseek2): top-k weights are NOT
    // renormalised (deepseek.cpp:145-155 passes norm_w=false and no
    // converter writes expert_weights_norm), so the fixture carries no
    // such key and the answer has to come from
    // NO_TOPK_RENORMALIZE_ARCHITECTURES.
    "deepseek",
    // `bailingmoe`: llama.cpp reads leading_dense_block_count and never
    // branches on it (bailingmoe.cpp:5 vs :39-54). The fixture sets the
    // key to 1 and ships NO dense FFN on layer 0.
    "bailingmoe",
    // `seed_oss`: the pre-FFN norm is stored as post_attention_norm and
    // there is no ffn_norm (seed-oss.cpp:36-37,113-115) -- gpt-oss's
    // slot, now a named list rather than an `arch == "gpt-oss"` flag.
    "seed_oss",
    // `maincoder` and `hunyuan-moe`: per-head QK norm applied AFTER RoPE
    // (maincoder.cpp:78-95, hunyuan-moe.cpp:93-118). Both fixtures use
    // QK-norm weights centred near 1.5 so the ordering is visible.
    "maincoder",
    "hunyuan-moe",
    // `hunyuan-dense`: the same post-RoPE QK-norm order (it has no graph
    // of its own -- models.h:1830-1834 derives it from
    // llama_model_hunyuan_vl) PLUS the NTK-alpha RoPE base rescale at
    // hunyuan-vl.cpp:8-12, which is now `rope_ntk_alpha`. Its fixture
    // carries `hunyuan-dense.rope.scaling.alpha` explicitly, because the
    // HUNYUAN_DENSE converter does that arithmetic in Python and writes
    // the already-scaled base (conversion/hunyuan.py:254-281) -- the
    // `add_rope_scaling_alpha` at :356 is HunyuanVLTextModel, i.e. the
    // separate `hunyuan-vl` row. The triage verdict cited that line for
    // this architecture and was wrong about it.
    "hunyuan-dense",
    // `ernie4_5-moe`: the MoE sibling of the audited `ernie4_5`. Its
    // interleave step is a REFUSAL rather than an implementation, and
    // that is the finding, not a shortcut: llama.cpp's tensor loader
    // (ernie4-5.cpp:49) creates expert tensors for every layer past the
    // leading-dense prefix with NO step in the condition, while its
    // graph (ernie4-5-moe.cpp:64) takes the dense branch when
    // `(il + 1) % step != 0`, so a checkpoint whose interleave really
    // interleaves cannot be loaded by llama.cpp at all -- measured, on a
    // two-step fixture, as `check_tensor_dims: tensor
    // 'blk.2.ffn_gate_inp.weight' not found`. Both published ERNIE-4.5
    // MoE checkpoints carry a step of 1, which is what the golden
    // fixture pins; `moe_interleave` refuses anything else by name.
    "ernie4_5-moe",
    // tests/fixture_away_graphs.rs: architectures that were triaged
    // FIXTURE-AWAY -- ferrox already built their graph, and only the
    // evidence was missing. Same standard as the rows above: a synthetic
    // fixture from `scripts/make_<arch>_fixture.py` whose golden values
    // come from llama.cpp's own graph via libllama, compared on prefill,
    // decode and continuous batching, with a sabotage test per row
    // proving the fixture can SEE the fact its architecture turns on.
    //
    // Each was checked, against the C, on the six things this repo has
    // lost at least once: RoPE variant, SWA pattern and phase,
    // `attention_scale`, the two post-norm slots, and QK-norm ordering.
    //
    // `internlm2` (internlm2.cpp:3-11,25-33,59-122): plain llama, NORM
    // RoPE, `1/sqrt(head_dim)` scale, no post-norms, no QK-norm, no SWA.
    // Its fixture carries the OPTIONAL q/k/v projection biases real
    // InternLM2 exports ship.
    "internlm2",
    // `xverse` (xverse.cpp:3-12,14-35,59-121): the same, with no biases.
    "xverse",
    // `gemma` (gemma.cpp:3-11,13-34,41-138): Gemma-1, the oldest row of
    // the family and the last one that was not evidenced. Its three
    // Gemma-specific pieces were already implemented for `GemmaFamily`
    // and the fixture is what proves each of them: the sqrt(n_embd)
    // embedding scale (:49), GeGLU rather than SwiGLU (:112,
    // LLM_FFN_GELU) and a `1/sqrt(head_dim)` attention scale that
    // llama.cpp reaches by scaling Q at :86 and passing kq_scale = 1.0f
    // at :91, which is what leaving `attention_scale` as None already
    // produces. Its lm_head is TIED with no fallback (:20), so the
    // fixture ships no `output.weight` and the embedding scale is not
    // cancelled downstream. Gemma-1 declares no softcap and no sliding
    // window, so the Gemma-2/3 machinery must resolve to inert, and
    // `tests/fixture_away_graphs.rs` asserts that rather than assuming
    // it.
    "gemma",
    // `ernie4_5` DENSE (ernie4-5.cpp:36-69,95-149): NORM RoPE, head_dim
    // decoupled from n_embd/n_head. `ernie4_5-moe` is a different row
    // with its own fixture, above.
    "ernie4_5",
    // `baichuan` (baichuan.cpp:5-14,17-40,64-137): the 7B ONLY. The 13B
    // is a different model under the same string and is refused by name
    // on `block_count == 40` in loader.rs before this list is consulted,
    // because llama.cpp picks ALiBi-and-no-RoPE off the layer count with
    // no GGUF key to declare it. The fixture therefore has 32 layers: a
    // 2-layer one would be LLM_TYPE_UNKNOWN and get no RoPE at all.
    "baichuan",
    // `exaone` (exaone.cpp:3-10,12-40,65-121): EXAONE 3.x, NEOX RoPE,
    // tied lm_head. NOT `exaone4` (no pre-norms) and NOT `exaone-moe`
    // (no RoPE on the full-attention layers); both stay refusing.
    "exaone",
    // `plamo3` (plamo3.cpp:3-60,91-193): the sandwich-norm row, and the
    // only one here with a sliding window. Its verdict was FIXTURE-AWAY
    // and was WRONG by one tensor name: plamo3 is the sole architecture
    // upstream that creates ATTN_POST_NORM / FFN_POST_NORM through the
    // two-argument `tn` overload (:52,55), so it asks for
    // `blk.N.post_attention_norm` and `blk.N.post_ffw_norm` with NO
    // `.weight`, and gguf-py emits exactly those names for it. ferrox
    // read only the suffixed spelling; `load_norm_vec_either_spelling`
    // in loader.rs now reads both, and says why.
    //
    // Its SWA is a real pattern with a real phase -- period from
    // `attention.sliding_window_pattern`, `dense_first = false` from
    // `set_swa_pattern`'s default -- and the fixture sets a window
    // narrower than the prompt so the mask actually bites.
    "plamo3",
    // tests/granite_family_graphs.rs: the Granite family, which was
    // triaged NEW CODE on four SCALAR MULTIPLIERS the generic decoder
    // did not apply -- `logit_scale`, `residual_scale`,
    // `embedding_scale` and `attention.scale`. They are hparams rather
    // than tensors, so `assert_every_tensor_consumed` cannot see them
    // and a Granite checkpoint would otherwise have loaded and answered
    // at the wrong scale. `crate::scalar_multipliers` implements all
    // four ONCE, parameterised by architecture, and
    // `capability::unsupported_scaling_keys` is now DERIVED from that
    // same table rather than restated beside it.
    //
    // `granite` (granite.cpp:5-10,180,225,235-238,288-292) is the dense
    // row. `granitemoe` has no graph of its own -- `models.h:1583-1591`
    // is `using graph = llama_model_granite::graph` -- so the two differ
    // in the FFN and in nothing else, and its fixture carries the MoE
    // branch, an UNGATED shared expert, and expert tensors sized from
    // `n_ff` rather than `n_ff_exp`.
    //
    // `granite-moe` is a ferrox-only alias: `llama-arch.cpp:101` spells
    // the architecture `granitemoe` and no GGUF anywhere says
    // `granite-moe`, so there is no libllama golden for it and there
    // never can be. Its evidence is a SECOND fixture, byte-identical
    // except for the architecture string and key prefixes, asserted
    // against `granitemoe`'s libllama golden -- which is the only thing
    // that can keep an alias nothing outside ferrox would ever exercise
    // from drifting away from the row it aliases.
    //
    // The `rope_finetuned` half of the verdict landed as a REFUSAL
    // (`crate::rope_finetuned`), not an implementation: granite.cpp:33-35
    // reads `{arch}.rope.scaling.finetuned` as a switch for RoPE itself,
    // and a file declaring it false runs UNROTATED in llama.cpp, which
    // ferrox has no way to express.
    "granite",
    "granitemoe",
    "granite-moe",
    // `bailingmoe2` (bailingmoe2.cpp:23-87,111-198): Ling-2.0. The one
    // MoE row in this batch, so the two MoE facts do arise and both are
    // asserted: SIGMOID gating, read from the file's REQUIRED
    // `expert_gating_func` (:11) against ferrox's softmax default, and
    // `expert_weights_norm` (:10), also read from the file. Its shared
    // expert is `n_ff_shexp * n_expert_shared` wide (:58), not
    // `n_ff_shexp`. Per-head QK norm BEFORE RoPE (:123-135), fused
    // attn_qkv, leading dense layers that llama.cpp really does branch
    // on (:57) -- unlike `bailingmoe`, which reads the same key and
    // ignores it.
    "bailingmoe2",
    // tests/post_norm_only_graphs.rs: the POST-NORM-ONLY family, two
    // architectures and ONE implementation (`crate::norm::NormOp`).
    // Neither has an `attn_norm` or an `ffn_norm` tensor; both read the
    // raw residual at both sublayers and norm each branch's output
    // before its residual add. Same evidence standard as the rows
    // above: a synthetic fixture per row whose golden logits come from
    // llama.cpp's own graph via libllama, on all three forward paths.
    //
    // `olmo2` (olmo2.cpp:45-52,92,160-165,169,177-182): WHOLE-VECTOR
    // QK-norm -- :45-46 sizes the norms `{n_embd}` and
    // `{n_head_kv * n_embd_head}` and :106-112 applies them to the 2-D
    // projections before `ggml_reshape_3d`. An `olmo2` file carrying
    // BOTH a sliding window and a rope scaling is Olmo-3, ropes its two
    // kinds of layer differently (:120-146), and is refused by name in
    // loader.rs.
    "olmo2",
    // `exaone4` (exaone4.cpp:60-67,118,152-169): the same graph with
    // PER-HEAD QK-norm instead -- :61-62 sizes them `{n_embd_head_k}`
    // and :127-128 applies them to what `build_qkv` already reshaped.
    // NOT the audited `exaone` row, which is EXAONE 3.x and a plain
    // pre-norm llama.
    //
    // BOTH SIZES run. EXAONE-4 32B (`block_count == 64`) used to be
    // refused by name: llama.cpp turns SWA on off the layer count
    // (:4-9) and then ropes only the sliding layers (:116), so its
    // full-attention layers get no rotation. `crate::rope_layers`
    // implements that rule and `capability::swa_disabled_by_arch`
    // carries the layer-count gate it depends on, with a 64-layer
    // libllama-golden fixture in `tests/no_rope_layer_graphs.rs`.
    "exaone4",
    // tests/no_rope_layer_graphs.rs: the PER-LAYER-RoPE group, three
    // rows on one rule (`crate::rope_layers`). llama.cpp gates rotation
    // per layer in six architectures and ferrox had no way to say so,
    // which cost `smollm3` and `exaone-moe` an outright refusal and
    // EXAONE-4 32B a refusal by name.
    //
    // `exaone-moe` (exaone-moe.cpp:136,155-161): `is_swa(il)` around
    // both `ggml_rope_ext` calls, with `swa_type` pinned to STANDARD at
    // :4 -- which is `exaone4.cpp:116` with the second disjunct nailed
    // false, i.e. the same rule and not a similar one. Its MoE half
    // (:72-93) is machinery ferrox already had and the fixture carries
    // all of it: leading dense, `exp_probs_b`, a shared expert sized by
    // `expert_shared_feed_forward_length`, sigmoid gating from
    // metadata, `expert_weights_scale`/`_norm`.
    "exaone-moe",
    // `smollm3` (smollm3.cpp:5,69): `(il + 1) % 4 != 0`, nine layers of
    // a 36-layer SmolLM3-3B unrotated, from a literal with no GGUF key.
    // The graph is otherwise the plain pre-norm llama one, so this is
    // the row where the rule is the ONLY thing -- which is why it was
    // in the "No RoPE at all" refusal group beside the ALiBi
    // architectures until the rule existed.
    "smollm3",
    // tests/one_match_arm_graphs.rs: the FUSED-`attn_qkv.bias` pair.
    // `create_tensor_qkv` (llama-model.cpp:2886-2900) creates the bias
    // beside a fused `wqkv`, and `build_qkv` (llama-graph.cpp:1605-1609)
    // adds it to the fused projection before splitting. ferrox split the
    // fused WEIGHT and then looked for the bias only under the split
    // `attn_q.bias` names, so it was dropped and all three projections
    // ran unbiased. Both halves now come out of one decision in
    // `qkv_fused`, sliced by the same spans.
    //
    // `chatglm` (chatglm.cpp:25-52,58-161) was the LAST ONE MATCH ARM
    // row anywhere in this file. It also pins the two things that made
    // it look fixture-away and are not: PARTIAL RoPE
    // (conversion/chatglm.py:151 writes `rope_dimension_count` as
    // `head_dim * 0.5`) and the fused gate+up SwiGLU
    // (chatglm.cpp:48,128-133), which is phi3's call shape.
    "chatglm",
    // `qwen` is Qwen-1 (`QWenLMHeadModel`), not qwen2/qwen3. Its
    // `attn_qkv.bias` is REQUIRED (qwen.cpp:28, flag `0`), which is
    // stronger than chatglm's optional one, and it needed a SECOND arm
    // the chatglm verdict did not name: qwen.cpp:33-35 sizes every FFN
    // matrix at `n_ff / 2`, because Qwen-1's `intermediate_size` counts
    // gate and up together. See `FFN_LENGTH_COUNTS_GATE_AND_UP` in
    // loader.rs.
    "qwen",
    // tests/minicpm_graphs.rs: MiniCPM, which was never an unaudited
    // row -- it was refused BY NAME, because the thing it does is
    // invisible in the file. `models.h:1594-1601` is
    // `using graph = llama_model_granite::graph`, so it is the Granite
    // graph object verbatim; what `minicpm.cpp:5-7` adds is DEFAULTS,
    // assigning an embedding multiplier of 12.0, a residual multiplier
    // of `1.4/sqrt(n_layer)` and a logit multiplier of `256/n_embd`
    // before `:12-14` lets the file override them. A MiniCPM export
    // carrying none of the three keys is still scaled by all three, so
    // `unsupported_scaling_keys` -- a key-PRESENCE gate -- can see
    // nothing to refuse. `scalar_multipliers::MultiplierDefaults` is
    // that hook, and the fixture that evidences it declares NO key at
    // all, which is the only fixture shape that can tell the hook from
    // its absence. A second fixture declares all three and pins that
    // the file still wins.
    //
    // It is Granite's arithmetic minus one column: `minicpm.cpp:3-24`
    // never reads `{arch}.attention.scale`, so that key stays refused
    // for this row by the derived list.
    "minicpm",
    // tests/olmo_graphs.rs: OLMo-1, the THIRD norm shape and the reason
    // `crate::norm::NormOp` has three variants rather than two. It is
    // pre-norm like `llama` -- `olmo.cpp:65-67` before attention,
    // :104-106 before the FFN -- so it is NOT the post-norm-only
    // topology `olmo2` and `exaone4` share. What differs is the norm
    // FUNCTION: all three sites are
    // `build_norm(x, NULL, NULL, LLM_NORM, il)`, a non-parametric
    // LayerNorm, and `olmo.cpp:15-36` creates no norm tensor at all --
    // no `attn_norm`, no `ffn_norm`, no `output_norm`.
    //
    // Its lm_head is TIED with a fallback (:21-25) and its RoPE is NORM
    // (llama-model.cpp:2585). Its CLAMP -- `olmo.cpp:5` reads
    // `{arch}.attention.clamp_kqv`, `llama-graph.cpp:1611-1652` applies
    // it to Q, K and V inside `build_qkv`, and `conversion/olmo.py:23-25`
    // really writes it for OLMo-7B-Twin-2T and OLMo-1.7-7B -- was a
    // refusal by name and is implemented now (`crate::clamp_kqv`), with
    // the clamped fixture matched against libllama rather than refused:
    // `dbrx` below needed the same clamp as a REQUIRED key.
    "olmo",
    // tests/dbrx_graphs.rs: DBRX, NEW CODE on three blockers that each
    // extended a seam landed the day before. `dbrx.cpp:69-71`, `:110-112`
    // and `:140-142` norm with `LLM_NORM` and a weight but no bias, which
    // is `crate::norm::NormOp::LayerNorm` -- the variant the OLMo-1 work
    // deliberately left unwritten until a row called it; `dbrx.cpp:5`
    // reads `attention.clamp_kqv` as REQUIRED, which `crate::clamp_kqv`
    // applies after the bias through the ONE helper every host body
    // shares (`decoder/qkv_bias.rs`); and `dbrx.cpp:34,110-113` keep the
    // pre-FFN norm under `blk.N.attn_output_norm`, which
    // `crate::norm_sites` reads into the same slot `gpt-oss` keeps under
    // `post_attention_norm`. Fused `attn_qkv` (:31), NEOX RoPE
    // (llama-model.cpp:2617), SiLU MoE with softmax gating and top-k
    // renormalisation (:115-125), untied lm_head (:24).
    "dbrx",
    // tests/grok_graphs.rs: Grok-1, NEW CODE on the MiniCPM shape --
    // `grok.cpp:5-12` seeds SEVEN hyper-parameters before `:14-27` let
    // the file override them, so a file declaring none is still scaled
    // by all of them and a key-presence gate sees nothing.
    // `scalar_multipliers::MultiplierDefaults::Grok` is that hook:
    // `embedding_scale` (78.38), `logit_scale` as a MULTIPLY (:211, the
    // `LogitScaleUse::AsIs` variant the module had named as absent),
    // `attention.output_scale` (0.0884, a fifth key that resolves into
    // the `attention_scale` slot) and the attention softcap default of
    // 30. The attention itself is `kq_scale = 1.0f` (:137) with the
    // real scale inside the tanh (llama-graph.cpp:2572-2582), which is
    // exactly "pre-scale Q, then softcap"; `router_logit_softcapping`
    // and `attention.temperature_length` are read at :20,:23 and applied
    // NOWHERE in the graph (measured: no other reference in `src/`), so
    // ferrox ignores them the same way. `blk.N.attn_output_norm` is the
    // POST-attention norm here (:143-146, before the residual add at
    // :148) and `layer_output_norm` / `post_ffw_norm` the post-FFN one
    // (:75-78, :185-188): a `crate::norm_sites` row. GELU MoE with
    // softmax gating (:158-168), NEOX RoPE (llama-model.cpp:2616),
    // tied-with-fallback lm_head (:46-51). Grok-2's parallel dense FFN
    // (`:171-184`, `sqrt(2)/2` on the sum) is refused BY NAME in
    // `loader.rs`, so this row is admitted for Grok-1.
    "grok",
    // tests/ungated_ffn_graphs.rs: Arcee AFM, NEW CODE on ONE fact --
    // the FFN has no gate. `arcee.cpp:39-40` creates `ffn_up` and
    // `ffn_down` only, and `:123-128` is `build_ffn` with a NULL gate,
    // `LLM_FFN_RELU_SQR` and `LLM_FFN_SEQ`: `down(relu(up(x))^2)`.
    // ferrox spells that as `FfnActivation::ReluSqr`, which the loader
    // serves by ALIASING the expert's gate to its up matrix and
    // `ferrox_moe::GluAct::Reglu` (`relu(gate) * up`), so the gated
    // struct and every gated path are untouched and `relu(up)^2` is
    // what they compute; the dense hot paths skip the aliased matmul.
    // No fused device kernel spells it, so `fused_kernel_gelu_flag`
    // returns `None` and `metal_can_serve_model` keeps the model off
    // the stacks -- which replaced six `gelu = !is_swiglu()` sites
    // that would have run a third activation as GELU. Everything else
    // is `llama` (:6 says so): NORM RoPE (llama-model.cpp:2600),
    // optional `output.weight` with a tied fallback (:20-25),
    // `n_embd_head == n_rot` asserted (:51-52). The same FFN is in
    // `plm`, `nemotron`, `jais2` and `nemotron-h`, each of which refuses
    // for something else; see `UNGATED_RELU_SQR`.
    "arcee",
    // tests/per_layer_shape_graphs.rs: the PER-LAYER-SHAPE pair, two
    // rows on one seam (`crate::layer_shapes`). llama.cpp reads
    // `head_count`, `head_count_kv` and `feed_forward_length` as
    // scalar-or-array for every architecture and hands most graphs
    // layer 0; these two index the arrays in both their tensor loader
    // and their graph, and ferrox carried all three as scalars. The
    // scan that sized the seam is recorded in
    // `layer_shapes::PER_LAYER_SHAPE_ARCHS`.
    //
    // `deci` (deci.cpp:30-34 loader, :103-105 graph): all three per
    // layer AND a three-way branch on them -- `n_head == 0` passes the
    // residual through with no norm (:107-109), `n_head_kv == 0` runs
    // `attn_norm` then `wo` alone (:115-118), `n_ff == 0` skips the FFN
    // (:147-149). `AttnShape::{Gqa, Linear, Absent}` and
    // `LayerShape::ffn_dim` are those, and the fixture has one layer of
    // each kind. A second fixture is the DeciLM-7B shape
    // (conversion/deci.py:114-118: `head_count_kv` alone as an array).
    // The FFN-free layer WITH attention is refused: `:147-149`
    // `continue`s before the residual add, and scaling that layer's
    // attention weights by 3 leaves libllama's logits byte-identical
    // (measured), so the branch is dead in the reference graph and
    // ferrox will not pin it. NORM RoPE (llama-model.cpp:2576).
    "deci",
    // `openelm` (openelm.cpp:26-28 loader, :67-69 graph): all three per
    // layer, one fused `wqkv` per layer sized `(2*n_head_kv(i) +
    // n_head(i)) * n_embd_head_k` (:34) -- `qkv_fused::FusedQkvRows::of`
    // takes the layer now -- per-head QK-norm before RoPE (:82-102),
    // NEOX RoPE (llama-model.cpp:2650), a tied lm_head with no fallback
    // (:22). The fixture's three layers share no KV width and no FFN
    // width. Its converter writes the arrays (conversion/openelm.py:
    // 57-59), which `layer_shapes::read_u64_per_layer` reads where
    // `GgufValue::as_u64` used to die on them.
    "openelm",
    // tests/gated_attention_graphs.rs: the GATED-ATTENTION pair, two
    // rows on one seam (`crate::attn_gate`). `afmoe.cpp:73,154,183-185`,
    // `laguna.cpp:110-124,211,246-257` and `step35.cpp:96,268-284` each
    // project a gate from the SAME normed input Q/K/V read and multiply
    // the attention output by it BEFORE `wo`; they differ in the
    // activation (sigmoid / softplus), in the width (per element / per
    // head / decided by the tensor's shape) and in whether the tensor
    // may be absent. Read side by side before being called one cause:
    // the three graphs, six create sites measured over all 140.
    //
    // `afmoe` (afmoe.cpp:73,120,154,183-185): sigmoid, per element,
    // REQUIRED. The other afmoe-only fact is `:120`, `sqrt(n_embd)` on
    // the embeddings from arithmetic -- the only non-Gemma graph that
    // does it (`embeddings_scaled_by_sqrt_n_embd`). Everything else it
    // needs it already had, and the fixture carries all of it: dual
    // norms on both blocks, per-head QK norm before RoPE, leading
    // dense, `exp_probs_b`, one shared expert, sigmoid gating with NO
    // key (`:29-30`), the NoPE layer from `crate::rope_layers`, and a
    // window with its own `rope.freq_base_swa`. NEOX RoPE
    // (llama-model.cpp:2676-2677).
    "afmoe",
    // `laguna` (laguna.cpp:110-124,211,246-257): SOFTPLUS, per head OR
    // per element -- `:112-123` reads the width off the stored tensor
    // and aborts on any other -- REQUIRED. Two fixtures, one per width:
    // the M.1 shape (no window, per element, uniform heads) and the
    // XS.2 shape (window, period 4 dense-first, per head, and
    // `head_count` as a per-layer ARRAY, which `crate::layer_shapes`
    // carries). One thing stays refused by name in `loader.rs`, from a
    // fixture that has it: a window together with a RoPE scaling
    // (`:48,184-192` run the sliding layers with YaRN off, the Olmo-3
    // rule). `rope.dimension_count_swa` (`:50`) differing from
    // `rope.dimension_count` -- a second rotary width -- was the other
    // and is SERVED since `step35` closed on the same two-valued width
    // (`ModelConfig::rope_dim_swa`, `crate::swa_geometry`); the
    // XS.2-shaped fixture that carries it matches libllama. NEOX RoPE
    // (llama-model.cpp:2676-2677).
    "laguna",
    // `mellum` (mellum.cpp:12-17,45-68,108-197): the per-layer
    // sliding-window ARRAY, honoured -- the scalar overload of
    // `get_key_or_arr` first, the array overload on its `false`, and
    // `conversion/mellum.py:28` always writes the array. The one
    // generic-path graph that honours it, so the fixture's array
    // [T, T, F, T] deliberately disagrees with the seeded period-4
    // [T, T, T, F] on two layers and the golden is the file's layout,
    // not the seed's. Everything else is machinery it already had: NEOX
    // RoPE (llama-model.cpp:2682), per-head QK norm before RoPE
    // (`:50-51,120-124`), softmax top-k renormalised (`:186`), the
    // expert width from its own key (`:5`). A window together with a
    // RoPE scaling -- `:128-142`, the Olmo-3 rule, and what every real
    // Mellum2 export declares -- stays refused by name
    // (`crate::swa_geometry`).
    "mellum",
    // tests/per_layer_activation_graphs.rs: `apertus` (apertus.cpp:6-9,
    // 45-46, 93-96, 129-142), the first architecture whose FFN
    // activation takes PARAMETERS THAT VARY BY LAYER -- xIELU with four
    // `n_layer`-long arrays (`xielu.alpha_n`, `.alpha_p`, `.beta`,
    // `.eps`, no architecture prefix) that `ggml_xielu` folds through
    // a softplus at graph build. `crate::act_layers` reads them exactly
    // as `get_key_or_arr` does (an array at `n_layer` length or a
    // scalar broadcast; a second fixture carries the scalar form and
    // libllama honours the broadcast), `ferrox_moe::GluAct::Xielu`
    // carries one layer's four, and `ModelConfig::layer_ffn_act(il)`
    // replaced the model-wide `GluAct::from(ffn_activation)` at every
    // FFN body, so no site can take the activation without saying
    // which layer's. The FFN is UNGATED like `arcee`'s and takes the
    // same gate-to-up alias; per-head RMS QK-norm before RoPE; NEOX.
    // Its optional `attn_q_norm.bias` / `attn_k_norm.bias` are created
    // and never read upstream (`crate::unread_tensors`, measured). No
    // fused Metal kernel spells xIELU, so every Metal launch refuses
    // it through `ModelConfig::model_ffn_act`.
    "apertus",
    "step35",
    // tests/attn_temperature_graphs.rs: `mistral3` (mistral3.cpp:5,
    // 14-17, 153-156), every Ministral-3 export. Its one blocker was
    // the PER-POSITION ATTENTION TEMPERATURE, `attention.temperature_scale`,
    // which llama-graph.cpp:163-167 turns into `log(floor(pos /
    // floor_scale) + 1) * scale + 1` per token and the graph multiplies
    // into Q after RoPE; `crate::attn_temperature` is the seam, with
    // the census (three graphs of 140 build the input, this the only
    // generic-path one) and the floor resolved as `llama-model.cpp:
    // 1164-1165` resolves it -- `context_length` first, the YaRN key
    // over it -- which the second fixture measures. Two corrections
    // to its verdict: the graph is either dense or MoE on every layer
    // with NO leading-dense split and NO shared expert (`:64-84` create
    // `_shexp` only under an `n_ff_shexp` its hparams never set, and
    // no graph line reads them); and `rope.scaling.yarn_log_multiplier`
    // (`:9`) adjusts a YaRN MAGNITUDE term ferrox turned out not to
    // apply at all -- `crate::yarn_magnitude`, evidenced on two more
    // fixtures with the factor at 4. NORM RoPE, `1/sqrt(head_dim)`.
    "mistral3",
];

/// Is this architecture's use of the shared generic path backed by
/// evidence?
pub fn is_audited_generic(arch: &str) -> bool {
    AUDITED_GENERIC_GQA.contains(&arch)
}

/// Architectures whose layers have **no pre-attention norm and no
/// pre-FFN norm at all**: the post-norm-only residual topology.
///
/// ```text
/// ffn_inp = x       + post_attn_norm(attn(x))
/// out     = ffn_inp + post_ffn_norm(ffn(ffn_inp))
/// ```
///
/// Not a family resemblance -- the two graphs were read side by side
/// and are the same statement for statement. `src/models/olmo2.cpp`
/// creates only `attn_q_norm`, `attn_k_norm`, `attn_post_norm` and
/// `ffn_post_norm` per layer (:45-52) and reads the raw residual at
/// both sublayers (`cur = inpL` at :92, `build_ffn(ffn_inp, ...)` at
/// :169), norming each branch's OUTPUT before its residual add
/// (:160-165, :177-182). `src/models/exaone4.cpp` is the same list
/// (:60-67) and the same four lines (:118, :159, :152-155, :166-169).
///
/// Both are refused unless [`AUDITED_GENERIC_GQA`] names them, and
/// `crate::norm::NormOp` is the one implementation they share.
/// Adding a third name here means having read a third `*.cpp`: this
/// list decides whether `loader.rs` demands `blk.N.attn_norm.weight`
/// from a file, so a wrong entry is a load that fails or a norm that
/// silently disappears.
pub const POST_NORM_ONLY_ARCHITECTURES: &[&str] = &["olmo2", "exaone4"];

/// Does this architecture read the raw residual at both sublayers?
/// See [`POST_NORM_ONLY_ARCHITECTURES`].
pub fn is_post_norm_only(arch: &str) -> bool {
    POST_NORM_ONLY_ARCHITECTURES.contains(&arch)
}

/// Architectures that normalise with a **non-parametric LayerNorm** --
/// subtract the mean, divide by the standard deviation, no learned
/// weight and no bias -- at every norm site.
///
/// `olmo` (OLMo-1), and llama.cpp has no second one. `olmo.cpp:27-35`
/// creates Q/K/V, `attn_output` and gate/up/down and NOT ONE norm
/// tensor, and its graph is `build_norm(x, NULL, NULL, LLM_NORM, il)`
/// at :65-67 (pre-attention), :104-106 (pre-FFN) and :128-130 (final).
///
/// It is pre-norm like `llama`, so this is orthogonal to
/// [`POST_NORM_ONLY_ARCHITECTURES`]: the difference is the norm
/// FUNCTION, not the residual wiring, and a name cannot be on both
/// lists (`loader.rs`'s
/// `the_norm_slot_and_function_lists_cannot_contradict`).
///
/// **This list will not grow, and that is a measured claim rather than
/// an expectation.** Every `build_norm` call in all of llama.cpp's
/// `src/models/*.cpp` graphs was scanned for a null weight argument:
/// three calls pass one to `LLM_NORM`, and all three are `olmo.cpp`.
/// `talkie.cpp` passes a null weight to `LLM_NORM_RMS` at five sites,
/// which is a non-parametric RMSNorm -- a different function, and a row
/// this list does not serve.
///
/// The LayerNorm *function* with a learned weight is a different list,
/// [`WEIGHTED_LAYER_NORM`], and it exists now because `dbrx` gave it a
/// caller.
pub const NON_PARAMETRIC_LAYER_NORM: &[&str] = &["olmo"];

/// Does this architecture normalise without any learned parameters?
/// See [`NON_PARAMETRIC_LAYER_NORM`].
pub fn uses_non_parametric_layer_norm(arch: &str) -> bool {
    NON_PARAMETRIC_LAYER_NORM.contains(&arch)
}

/// Architectures that normalise with a **LayerNorm with a learned
/// weight and no bias** -- `build_norm(x, w, NULL, LLM_NORM, il)` -- at
/// every norm site.
///
/// `dbrx`: `src/models/dbrx.cpp:4` reads `LLM_KV_ATTENTION_LAYERNORM_EPS`
/// (not the RMS one) and the graph passes `LLM_NORM` with a weight and
/// a null bias at all three sites -- `:69-71` pre-attention, `:110-112`
/// pre-FFN (on `attn_out_norm`, its pre-FFN tensor; see
/// `crate::norm_sites`) and `:140-142` final. `dbrx.cpp:29,34,23`
/// create the three weights and no bias tensor at all.
///
/// The variant is `crate::norm::NormOp::LayerNorm`. It was deliberately
/// not written alongside the parameterless one, because every row that
/// needed it refused for more than the norm; `dbrx` needed two more
/// things and both were one implementation each (`crate::clamp_kqv`,
/// `crate::norm_sites`), which is what made it worth landing.
///
/// **What this list does NOT close**, so nobody adds a name on the
/// strength of "it is LayerNorm too": the `nemotron` / `orion` /
/// `stablelm` / `codeshell` / `jais2` / `starcoder` / `starcoder2` /
/// `phimoe` group all create `*_norm.bias` as REQUIRED and `build_norm`
/// adds it after the multiply. That is a `LayerNorm(w, b)` variant this
/// enum does not have, and each of those rows refuses for something
/// else on top -- ALiBi, learned positions, a parallel residual --
/// which is why the bias form still has no caller.
pub const WEIGHTED_LAYER_NORM: &[&str] = &["dbrx"];

/// Does this architecture normalise with a weighted LayerNorm?
/// See [`WEIGHTED_LAYER_NORM`].
pub fn uses_weighted_layer_norm(arch: &str) -> bool {
    WEIGHTED_LAYER_NORM.contains(&arch)
}

/// How the generic `Decoder` / `ModelConfig::from_gguf` path treats a
/// GGUF architecture string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchPath {
    /// Standard GQA (+ optional MoE) decoder; RoPE layout is known.
    GenericGqa { rope: RopeLayout },
    /// In-repo test fixtures (`ferroxtest*`) -- not a real model family.
    TestFixture { rope: RopeLayout },
    /// Real architecture, but must not be loaded through the generic
    /// GQA decoder (wrong attention / residual math).
    DedicatedOnly { reason: &'static str },
    /// In the llama.cpp inventory but out of Ferrox scope for now.
    Deferred { reason: &'static str },
}

/// Load-time resolved profile for one GGUF `general.architecture` string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchProfile {
    pub gguf_name: &'static str,
    pub scope: ArchScope,
    pub family: DecoderFamily,
    pub memory: MemoryKind,
    pub rope: RopeLayout,
    pub path: ArchPath,
    /// Default QK-norm style when norm tensors are present; loader may
    /// refine from tensor length.
    pub qk_norm: QkNormStyle,
    /// For an UNAUDITED [`ArchPath::GenericGqa`] row: how far it is from
    /// running, read against llama.cpp's own graph. `None` on audited
    /// rows (which run) and on rows still in [`TRIAGE_PENDING`].
    pub triage: Option<UnauditedTriage>,
}

impl ArchProfile {
    /// Attach a triage verdict to a catalog row. Private on purpose:
    /// verdicts are data of the catalog, not something a caller supplies.
    fn triaged(mut self, class: TriageClass, blocker: &'static str) -> Self {
        self.triage = Some(UnauditedTriage { class, blocker });
        self
    }
}

fn prof(
    name: &'static str,
    scope: ArchScope,
    fam: DecoderFamily,
    mem: MemoryKind,
    rope: RopeLayout,
    path: ArchPath,
    qk: QkNormStyle,
) -> ArchProfile {
    ArchProfile {
        gguf_name: name,
        scope,
        family: fam,
        memory: mem,
        rope,
        path,
        qk_norm: qk,
        triage: None,
    }
}

fn gqa_norm(name: &'static str) -> ArchProfile {
    prof(
        name,
        ArchScope::TextGeneration,
        DecoderFamily::StandardGqa,
        MemoryKind::KvGqa,
        RopeLayout::Norm,
        ArchPath::GenericGqa {
            rope: RopeLayout::Norm,
        },
        QkNormStyle::WholeVector,
    )
}

fn gqa_neox(name: &'static str) -> ArchProfile {
    prof(
        name,
        ArchScope::TextGeneration,
        DecoderFamily::StandardGqa,
        MemoryKind::KvGqa,
        RopeLayout::Neox,
        ArchPath::GenericGqa {
            rope: RopeLayout::Neox,
        },
        QkNormStyle::WholeVector,
    )
}

fn dedicated(name: &'static str, reason: &'static str) -> ArchProfile {
    prof(
        name,
        ArchScope::TextGeneration,
        DecoderFamily::Dedicated,
        MemoryKind::KvGqa,
        RopeLayout::Norm,
        ArchPath::DedicatedOnly { reason },
        QkNormStyle::WholeVector,
    )
}

fn deferred_scope(name: &'static str, scope: ArchScope, reason: &'static str) -> ArchProfile {
    prof(
        name,
        scope,
        DecoderFamily::StandardGqa,
        MemoryKind::None,
        RopeLayout::Neox,
        ArchPath::Deferred { reason },
        QkNormStyle::WholeVector,
    )
}

/// Triaged rows of the generic **Norm**-RoPE group, with the llama.cpp
/// line that decides each verdict. Consumed by
/// [`architecture_catalog`]; a name here must not also appear in the
/// untriaged list above it or in [`TRIAGE_PENDING`], which
/// `catalog_has_unique_names` and
/// `every_unaudited_generic_architecture_is_triaged_or_listed_as_pending`
/// between them enforce.
const NORM_ROPE_TRIAGED: &[(&str, TriageClass, &str)] = &[
    // `ernie4_5-moe` was HERE, ONE MATCH ARM on
    // `{arch}.interleave_moe_layer_step`. Building its fixture found the
    // arm is not implementable against a reference: llama.cpp's tensor
    // loader (ernie4-5.cpp:49) and its graph (ernie4-5-moe.cpp:64)
    // disagree about which layers are MoE, and only the graph has the
    // step, so a checkpoint whose interleave interleaves cannot be
    // loaded by llama.cpp at all. The arm landed as a REFUSAL
    // (`crate::moe_interleave`) and the step every real checkpoint
    // carries is audited (`tests/one_match_arm_graphs.rs`), so the row
    // is in AUDITED_GENERIC_GQA and carries no verdict.
    //
    // `granite`, `granitemoe` and the `granite-moe` alias were HERE,
    // NEW CODE on the four scalar multipliers. All three are audited
    // now: `crate::scalar_multipliers` implements the multipliers ONCE,
    // parameterised by architecture, and `tests/granite_family_graphs.rs`
    // is the libllama-golden evidence. The `rope_finetuned` half of that
    // verdict landed as a REFUSAL rather than an implementation
    // (`crate::rope_finetuned`), because llama.cpp runs such a file with
    // no rotation at all and ferrox cannot express that.
    //
    // `chatglm` was HERE, ONE MATCH ARM on the fused `attn_qkv.bias`.
    // The arm landed (`crate::qkv_fused`, which now resolves the
    // projections and their biases from ONE decision about which
    // spelling the file uses) and is evidenced against libllama in
    // `tests/one_match_arm_graphs.rs`, so the row is in
    // AUDITED_GENERIC_GQA and carries no verdict.
    //
    // Its verdict said implementing the arm "closes chatglm and qwen
    // together". It did not, and that is the finding: `qwen` needs the
    // same bias AND a second, unrelated arm the verdict did not name --
    // `qwen.cpp:33-35` sizes every FFN matrix `n_ff/2`, because
    // Qwen-1's `intermediate_size` counts gate and up together. `qwen`
    // stays refused, with that added to its reason.
    // `deci` was HERE, NEW CODE on PER-LAYER SHAPES, and is audited
    // now with `openelm` on one seam (`crate::layer_shapes`,
    // `tests/per_layer_shape_graphs.rs`). Its three layer kinds are
    // `AttnShape::{Gqa, Linear, Absent}` plus `ffn_dim == 0`; the one
    // combination llama.cpp's graph handles by discarding a computed
    // branch (`deci.cpp:147-149` before `:150-153`) is refused by name
    // from a fixture that has it, with the drop MEASURED rather than
    // read.
    // `olmo` was HERE, NEW CODE on the non-parametric LayerNorm, and is
    // audited now: `crate::norm::NormOp::LayerNormNoParams` implements
    // the function and `tests/olmo_graphs.rs` carries the fixture. Its
    // verdict called the clamp "an optional key nothing here applies";
    // that half stayed a REFUSAL rather than an implementation, because
    // `llama-graph.cpp:1611-1652` really does clamp Q, K and V and
    // `conversion/olmo.py:23-25` really does write the key. See
    // `crate::clamp_kqv`.
    (
        "arctic",
        TriageClass::NewCode,
        "a PARALLEL dense+MoE layer whose MoE branch reads the pre-attention residual. \
         src/models/arctic.cpp:124-132 runs a dense SiLU FFN on `ffn_norm(ffn_inp)` and adds \
         it back to ffn_inp, then :136-141 norms `inpSA` -- the layer INPUT, before \
         attention -- through a second per-layer norm `ffn_norm_exps` (:45) and runs the MoE \
         on that, and :154 sums the two. The generic decoder computes one FFN on the \
         post-attention residual, so this is a different graph, not a wider one. The dense \
         half is also sized `{n_embd, n_embd}` (:40-42) rather than n_ff. Same shape as \
         `smallthinker`'s router: a branch fed from the raw layer input",
    ),
    // `mistral3` was HERE, NEW CODE on the PER-POSITION ATTENTION
    // TEMPERATURE, and is audited now: `crate::attn_temperature` is the
    // seam and `tests/attn_temperature_graphs.rs` carries five
    // fixtures. Its verdict's "leading-dense + MoE + shared expert" was
    // wrong on two counts (see the AUDITED entry), and its
    // `yarn_log_multiplier` half found that YaRN's magnitude term was
    // missing for every architecture (`crate::yarn_magnitude`). The
    // reach was measured before a line was written: `llama4` seeds the
    // same three constants from literals and gates the multiply on
    // its no-RoPE layers (`llama4.cpp:15-17,175-176`), and `deepseek2`
    // / `mistral4` read the same key with `attention.temperature_length`
    // as the floor (`deepseek2.cpp:46-49`) -- the MLA loader REFUSES
    // that by name now, where it used to drop it, because that engine
    // has no golden to check an implementation against.
    (
        "nanbeige",
        TriageClass::NewCode,
        "nanbeige RUNS THE SAME PHYSICAL LAYERS MORE THAN ONCE. \
         src/models/nanbeige.cpp:13-31 sets `n_layer_all = n_layer_phys * n_loops` and \
         rewrites the per-layer head/ff/swa arrays so the graph walks n_layer_all steps over \
         n_layer_phys sets of weights, and :167 applies `output_norm` to the running \
         residual inside the loop at the end of each pass. ferrox's decoder walks its layer \
         vector exactly once and has no concept of a loop count. The per-layer arrays \
         themselves (`crate::layer_shapes`) are not the blocker: :24-26 copies them, it \
         does not vary them, and it is the copy that has no home. Everything inside one \
         pass (:52-63, :106-155) is plain llama, which is what makes this deceptive: the \
         tensor set alone looks generic",
    ),
    // `arcee` was HERE, NEW CODE on `UNGATED_RELU_SQR`, and is audited
    // now: the FFN is `FfnActivation::ReluSqr` and
    // `tests/ungated_ffn_graphs.rs` carries the fixture.
    ("plm", TriageClass::NewCode, UNGATED_RELU_SQR),
];

/// Shared by the three ferrox-only alias rows `mistral`, `mixtral` and
/// `yi`: the reason none of them is an architecture at all.
///
/// These were UNKNOWN, and the open question was "what would settle
/// it?" -- a real GGUF whose `general.architecture` is literally one of
/// the three. **The investigation that settled it (2026-09-10) did not
/// find one, and found the reason no such file exists.** Three
/// measurements, not readings:
///
///   1. `grep '"mistral' src/llama-arch.cpp` returns `mistral3` and
///      `mistral4` and nothing else; `mixtral` and `yi` return nothing.
///      Neither is in gguf-py's `MODEL_ARCH_NAMES` either.
///   2. A GGUF written with `general.architecture = "mistral"` (and
///      `mixtral`, and `yi`) is REFUSED by libllama with
///      `llama_model_load: error loading model: unknown model
///      architecture: 'mistral'`. So no golden reference for these rows
///      can ever exist, at the evidence standard every audited row in
///      this file meets.
///   3. The two real checkpoints in `models/` --
///      `Mistral-7B-Instruct-v0.2-Q4_K_M.gguf` and
///      `Yi-1.5-6B-Chat-Q4_K_M.gguf` -- both declare
///      `general.architecture = llama`, which is audited and runs.
///
/// So the rows are refused as strings rather than triaged as
/// architectures, and the refusal says the actionable thing: your file
/// is spelled `llama`. Leaving them on the generic path would have kept
/// a live hazard: the catalog gave all three NEOX RoPE while `llama`,
/// the graph they really are, is in `llama_model_rope_type`'s NORM
/// group (llama-model.cpp, the `case LLM_ARCH_LLAMA:` arm), so a file
/// spelling `mistral` would have been rotated on the wrong pairs of
/// every Q/K head -- the exact defect behind the Llama-3.1-8B
/// wrong-logits bug -- and `rope_layout_matches_llama_cpp` cannot see
/// it, because its lookup miss on a ferrox-only name is a `continue`.
///
/// `phi4` is the same shape and is deliberately NOT changed here: it is
/// still `GenericGqa` + UNKNOWN, because unlike these three it names a
/// concrete, checkable hypothesis (phi3's fused-QKV graph) that a real
/// file would confirm or refute. These three name none.
const NO_UPSTREAM_ARCH: &str =
    "this is not a GGUF architecture. `mistral`, `mixtral` and `yi` appear in neither      LLM_ARCH_NAMES (src/llama-arch.cpp lists `mistral3` and `mistral4` and nothing else      under that prefix) nor gguf-py's MODEL_ARCH_NAMES, and libllama REFUSES a file      declaring one of them: `unknown model architecture: 'mistral'` -- measured, on a      synthetic llama-shaped file written under each of the three strings. Every real      Mistral, Mixtral and Yi checkpoint converts to `llama` instead, which ferrox audits      and runs: the two in this repo's own models/ directory      (Mistral-7B-Instruct-v0.2-Q4_K_M.gguf, Yi-1.5-6B-Chat-Q4_K_M.gguf) both declare      `general.architecture = llama`. IF YOUR FILE REALLY SPELLS THIS, it came from a      converter neither engine has read, so its RoPE variant, its norm placement and its      FFN shape are all undetermined and ferrox will not guess -- re-convert it with      llama.cpp's convert_hf_to_gguf.py and it will load as `llama`. These rows used to sit      on the generic path with NEOX RoPE, while `llama` is in llama_model_rope_type's NORM      group, so such a file would have been rotated on the wrong pairs of every Q/K head";

/// Was shared by `arcee` and `plm`: an ungated ReLU-squared MLP. `arcee`
/// closed on it (2026-09-11, `FfnActivation::ReluSqr`,
/// `tests/ungated_ffn_graphs.rs`); `plm` did not, and this is the
/// verdict that says why.
///
/// The original verdict called the two "one cause" from the FFN alone,
/// and it was wrong by an attention block: reading `plm.cpp` against
/// `arcee.cpp` before assuming they were identical (the `diff` is 150
/// lines) shows `plm.cpp:16-19,32-36` creating `attn_kv_a_mqa`,
/// `attn_kv_a_norm` and `attn_kv_b` sized by `kv_lora_rank` and
/// `n_rot`, and `:84-166` running DeepSeek-2's MLA attention -- Q split
/// into `q_nope`/`q_pe`, a compressed KV normed and re-expanded, a
/// shared roped key concatenated onto every head. ferrox has that
/// attention only in the dedicated `MlaEngine`. Five graphs use the
/// FFN (`arcee`, `plm`, `nemotron`, `jais2`, `nemotron-h`, measured by
/// grepping `LLM_FFN_RELU_SQR` over `src/models/`); only `arcee` needed
/// nothing else.
const UNGATED_RELU_SQR: &str =
    "DeepSeek-2 MLA attention on a dense model, and NOT the ungated ReLU-squared FFN the \
     verdict used to name -- that half is IMPLEMENTED (`FfnActivation::ReluSqr`, audited on \
     `arcee`, whose `arcee.cpp:123-128` is the same `build_ffn(up, NULL gate, down, \
     LLM_FFN_RELU_SQR, LLM_FFN_SEQ)` call as plm.cpp:181-187). What remains is the \
     attention: src/models/plm.cpp:16-19 sizes `attn_kv_a_mqa` {n_embd, kv_lora_rank + \
     n_rot}, `attn_kv_a_norm` {kv_lora_rank} and `attn_kv_b` {kv_lora_rank, n_head * \
     (qk_nope + v)} from `attention.kv_lora_rank` (:5), and :84-166 splits Q into nope/pe \
     views, RMS-norms the compressed KV, re-expands it, ropes ONE shared key and repeats it \
     across the heads -- `deepseek2.cpp`'s graph with no q_lora and no MoE. ferrox runs \
     that attention only inside the dedicated `MlaEngine` (`mla_gguf_loader.rs`, arch-gated \
     to `deepseek2` / `mistral4`), which has no dense ReLU-squared FFN and no libllama-golden \
     evidence of its own; the generic decoder has no MLA at all";

/// Triaged rows of the generic **NEOX**-RoPE group. Same rules as
/// [`NORM_ROPE_TRIAGED`].
const NEOX_ROPE_TRIAGED: &[(&str, TriageClass, &str)] = &[
    // `mellum` was HERE, NEW CODE on two things. The first -- its
    // sliding layers roped with the model's YaRN switched OFF
    // (`mellum.cpp:128-142`), the Olmo-3 rule -- is a REFUSAL BY NAME
    // in `crate::swa_geometry` for a file declaring both a window and
    // a RoPE scaling, which every real Mellum2 export does. The second
    // -- the per-layer sliding-window ARRAY that `:12-17` honour and
    // `conversion/mellum.py:28` always writes -- is `crate::swa_layers`
    // now, and `mellum` is the ONE generic-path architecture whose
    // graph honours the array, so it is the row that evidences that
    // branch against libllama (`tests/window_array_graphs.rs`). A
    // Mellum without a scaling runs; a Mellum2 stops on the first
    // thing, by name.
    (
        "talkie",
        TriageClass::NewCode,
        "talkie has NO norm weights and a learned per-layer skip connection. In \
         src/models/talkie.cpp every \
         normalisation is `build_norm(x, nullptr, nullptr, LLM_NORM_RMS, ...)` -- \
         non-parametric RMSNorm, no weight tensor -- at :50 (on the embeddings, before layer \
         0), :68, :90, :110 and :137; the only norm weight in the file is `attn_q_norm`, and \
         it is shaped {1, n_head} (:26), one SCALAR PER HEAD rather than a head_dim vector, \
         which is neither of ferrox's two QkNormStyle variants. Each layer then adds \
         `inp_skip * out_scale` (:123-126) with a per-layer learned scalar `out_scale` \
         (:32), a second residual stream the generic decoder has no slot for, and :5 reads \
         {arch}.logit_scale as REQUIRED",
    ),
    (
        "mimo2",
        TriageClass::NewCode,
        "a V head width that differs from the K head width, in every real export. \
         conversion/mimo.py:154 writes `attention.value_length` from `v_head_dim` separately \
         from `attention.key_length`, and MiMo-V2-Flash's config.json is `head_dim: 192, \
         v_head_dim: 128` (`swa_v_head_dim: 128` too), so src/models/mimo2.cpp:47-48 sizes K \
         and V per layer from two widths and :132-140,152-154 view Q/K at `n_embd_head_k` \
         and V at `n_embd_head_v`; ferrox's KV cache, attention kernels and every fused Metal \
         launch take ONE head width, and the loader refuses a file whose two differ \
         (`split K/V head dims`). Second, small: :14-17 and :180-183 scale the attention \
         output by `{arch}.attention.value_scale` after `wo` when the config sets \
         `attention_value_scale` (mimo.py:163-165; MiMo-V2-Flash sets 0.707), a key ferrox \
         neither reads nor gates. NO LONGER a blocker: the NEXTN blocks -- every export \
         appends three inside block_count (mimo.py:22,27,167; :19-20, :31-37,51-52 create \
         them TENSOR_SKIP), and `crate::mtp_blocks` skips them as llama.cpp does. NO \
         LONGER a blocker: the per-layer sliding-window ARRAY -- mimo.py:148-153 writes \
         `hybrid_layer_pattern` at block_count length, :12 `get_key_or_arr(..., \
         is_swa_impl, n_layer())` BEFORE :19 reads nextn so the length is n_layer_all, and \
         `crate::swa_layers` honours the array for this architecture and broadcasts a \
         scalar as a bool rather than a period. NO LONGER a blocker: the attention sinks \
         -- :58 creates `attn_sinks` `{n_head}` and :177 passes it into the SAME \
         `build_attn_mha` as `openai-moe.cpp:115`, and `AttnWeights::sinks` loads by \
         tensor presence. NO LONGER a blocker: the per-layer shapes -- :47-49 and \
         :111-112 read n_head / n_head_kv PER LAYER, which `crate::layer_shapes` carries. \
         The dense-or-MoE-per-layer choice at :200-223 \
         and the fused-or-split QKV at :127-155 ferrox already has",
    ),
    // `afmoe` was HERE, NEW CODE on the gated attention (`afmoe.cpp:73`)
    // and the `sqrt(n_embd)` embedding scale (`:120`). Both are
    // implemented -- `crate::attn_gate` and
    // `embeddings_scaled_by_sqrt_n_embd` -- and the row is audited on a
    // libllama-golden fixture (`tests/gated_attention_graphs.rs`). Its
    // sigmoid default for `expert_gating_func` (`:29-30`) had been in
    // `SIGMOID_GATING_ARCHITECTURES` since 2026-09-01; the fixture
    // declares no gating key so that default is what it measures.
    // `apertus` was HERE, NEW CODE on xIELU with four PER-LAYER
    // parameter arrays (`apertus.cpp:6-9,132-138`). The arrays are
    // `crate::act_layers` (read as `get_key_or_arr` reads them, an
    // array at `n_layer` length or a scalar broadcast), the activation
    // is `ferrox_moe::GluAct::Xielu` carrying that layer's four, and
    // `ModelConfig::layer_ffn_act(il)` is the one accessor every FFN
    // body asks -- the row is audited on a libllama-golden fixture
    // (`tests/per_layer_activation_graphs.rs`). The verdict's third
    // sentence was wrong: `:50,52` CREATE `attn_q_norm.bias` /
    // `attn_k_norm.bias` and `:93,96` pass `NULL` as the bias, so they
    // are never read; measured (libllama's logits byte-identical with
    // and without them) and recorded in `crate::unread_tensors`.
    (
        "grovemoe",
        TriageClass::NewCode,
        "a SECOND bank of experts, not just a scale. src/models/grovemoe.cpp:57-59 creates \
         `ffn_gate_chexps` / `ffn_down_chexps` / `ffn_up_chexps` -- `n_expert / \
         n_group_experts` \"chunk\" experts with their own width n_ff_chexp -- and the graph \
         runs build_moe_ffn TWICE (:137 over the ordinary experts, :153 over the chunk \
         experts) before :167 adds `scale(moe_out, expert_group_scale)` to the residual. The \
         inventory recorded only the post-sum group scale and called this small; the second \
         expert bank with its own routing is the larger half and ferrox's MoE layer holds \
         one bank. Both n_group_experts and expert_group_scale are REQUIRED keys (:6-7). \
         QK-norm is before RoPE (:100-109), which is the one thing that would otherwise have \
         been a blocker",
    ),
    // `hunyuan-dense` was HERE, ONE MATCH ARM on the NTK-alpha RoPE
    // base rescale. The arm landed (`crate::rope_ntk_alpha`), the
    // post-RoPE QK-norm half was already implemented, and both are
    // evidenced against libllama in `tests/one_match_arm_graphs.rs`, so
    // the row is audited and carries no verdict. Its verdict cited
    // `conversion/hunyuan.py:356` as the line that writes
    // `{arch}.rope.scaling.alpha` for this architecture; that line is in
    // HunyuanVLTextModel, whose model_arch is HUNYUAN_VL. The
    // HUNYUAN_DENSE converter (:254-281) does the same arithmetic in
    // Python and writes the already-scaled base instead.
    // `laguna` was HERE, NEW CODE on the gated attention (`laguna.cpp:124`,
    // softplus, per head or per element) and a second rotary width
    // (`:50`). The gate is `crate::attn_gate` and the row is audited on
    // two libllama-golden fixtures, one per width
    // (`tests/gated_attention_graphs.rs`). The second rotary width is a
    // REFUSAL by name in `loader.rs` for one day -- a file whose
    // `rope.dimension_count_swa` differs from `rope.dimension_count` --
    // and is served now (`ModelConfig::rope_dim_swa`, with `step35`);
    // a window with a RoPE scaling (`:48,184-192`, the Olmo-3 rule,
    // `swa_layers_unscaled_rope`) stays refused, from a fixture that
    // has it. Real Laguna-M.1 has neither; real Laguna-XS.2 has both
    // and stops at the scaling.
    // `step35` was HERE, NEW CODE on its per-layer SwiGLU clamp arrays
    // (`step35.cpp:28-29`, applied by llama.cpp's generic
    // `build_moe_ffn` / `build_ffn` at `llama-graph.cpp:2146-2164` /
    // `:1751-1768`) and its half-width rotary on the full layers
    // (`:9`). The clamp is the second body on the per-layer activation
    // seam `apertus` opened -- `crate::act_layers::SwigluClamps`, read
    // by SITE, `ferrox_moe::GluAct::SwigluClamped` -- and the width is
    // `ModelConfig::rope_dim_swa` (`crate::swa_geometry`, the same
    // two-valued `n_rot(il)` that lifted Laguna-XS.2's refusal). Three
    // libllama-golden fixtures (`tests/clamped_swiglu_graphs.rs`):
    // clamped, unclamped, and with a NextN block. Everything else the
    // verdict had crossed off is carried by them rather than assumed.
    // `mistral`, `mixtral` and `yi` were HERE, UNKNOWN on
    // NO_UPSTREAM_ARCH. The question that verdict asked -- "is there a
    // real GGUF spelling one of these?" -- was answered NO, with a
    // measurement: libllama refuses all three strings outright. They
    // are refused as strings now, not triaged as architectures. See
    // NO_UPSTREAM_ARCH.
    // `grok` and `dbrx` were HERE, both NEW CODE, and both closed on
    // seams that had landed the day before. `grok`'s verdict named
    // five hardcoded defaults (`grok.cpp:5-12` -- there are seven at
    // this checkout), a `kq_scale = 1.0f` attention with the real
    // scale folded into a tanh softcap, and `blk.N.attn_output_norm`
    // as an unread tensor name: the defaults are a
    // `scalar_multipliers::MultiplierDefaults` variant like MiniCPM's,
    // the attention is `attention_scale` plus the existing softcap, and
    // the tensor name is a `crate::norm_sites` row. Its Grok-2 shape --
    // a dense GELU FFN summed with the MoE at sqrt(2)/2 (`:171-184`)
    // -- is refused BY NAME in `loader.rs`, so the row is admitted for
    // Grok-1. `dbrx`'s verdict named the weighted LayerNorm, the
    // REQUIRED `attention.clamp_kqv`, and `attn_output_norm` as its
    // pre-FFN norm: `crate::norm::NormOp::LayerNorm`, `crate::clamp_kqv`
    // (which closed `olmo`'s clip_qkv sub-refusal with it) and the
    // same `norm_sites` table. See `AUDITED_GENERIC_GQA`.
    (
        "smallthinker",
        TriageClass::NewCode,
        "the MoE router reads a DIFFERENT tensor. src/models/smallthinker.cpp:111 computes \
         the router logits from the raw layer input `inpL`, before the attention block, and \
         passes them into build_moe_ffn as a precomputed `probs` with a NULL ffn_gate_inp \
         (:151-161); every other MoE architecture routes on the normed FFN input, which is \
         what ferrox computes. One more that alone would disqualify it: `LLM_FFN_RELU` \
         experts (:158), and FfnActivation has no ReLU variant. :8 also pins n_swa to 4096 \
         over whatever the file declares. NO LONGER a blocker: the NoPE layers. \
         llama-hparams.h:203 defaults n_no_rope_layer_step to 4 and the SWA branch (:6-15) \
         never overwrites it, so :108-109's `use_rope = n_no_rope_layer_step == n_layer || \
         il % n_no_rope_layer_step != 0` leaves layers 0, 4, 8 ... unrotated -- the FIRST \
         layer of each period, which is the OTHER phase from smollm3's -- and \
         `crate::rope_layers` carries smallthinker as exactly that, with the no-window \
         branch (:18, step = n_layer, i.e. rotate everything) as well. \
         `default_swa_layout` and `swa_rope_base_follows_model` already carry smallthinker \
         correctly; they are not the blocker",
    ),
    (
        "bitnet",
        TriageClass::NewCode,
        "two norms INSIDE the blocks, in slots ferrox does not have. \
         src/models/bitnet.cpp:24,36 require `attn_sub_norm` and `ffn_sub_norm`, and the \
         graph applies attn_sub_norm to the attention output BEFORE the output projection \
         (:101-106 -- not after it, where ferrox's post_attn_norm sits) and ffn_sub_norm \
         between the gate*up product and `ffn_down` (:135-140), inside the FFN. It also \
         carries a per-tensor `scale` for every projection (:27-43, applied via \
         build_lora_mm) and creates no `output` tensor at all, taking the LM head from \
         `tok_embd` unconditionally (:164). ferrox refuses it by name today via the \
         unread-tensor gate (`blk.N.attn_sub_norm`, llama-arch.cpp:510-511), which is the \
         right outcome and not a small fix",
    ),
    // `openelm` was HERE, NEW CODE on PER-LAYER SHAPES, and is audited
    // now with `deci` (`crate::layer_shapes`,
    // `tests/per_layer_shape_graphs.rs`). The misleading missing-hparam
    // message its verdict named is gone with it:
    // `layer_shapes::read_u64_per_layer` reads the arrays
    // `conversion/openelm.py:57-59` writes.
];

/// Full inventory keyed by GGUF `general.architecture` string.
/// Kept in sync with `.scratch/llama.cpp/src/llama-arch.cpp` `LLM_ARCH_NAMES`.
pub fn architecture_catalog() -> &'static [ArchProfile] {
    use std::sync::OnceLock;
    use ArchScope::*;
    use DecoderFamily::*;
    use MemoryKind::*;
    use QkNormStyle::*;
    use RopeLayout::*;

    static CAT: OnceLock<Vec<ArchProfile>> = OnceLock::new();
    CAT.get_or_init(|| {
        let mut v = Vec::with_capacity(160);
        // --- Verified / standard GQA (Norm RoPE) ---
        //
        // `llama` is the only untriaged name left in this group: it is
        // audited, so it runs and needs no verdict. Every other
        // Norm-RoPE row moved into `NORM_ROPE_TRIAGED` below when it was
        // read against llama.cpp's graph.
        v.push(gqa_norm("llama"));
        // Audited too, each by a libllama-golden fixture -- see
        // `AUDITED_GENERIC_GQA` for the arm each one needed and
        // `tests/one_match_arm_graphs.rs` for the evidence.
        for n in ["bailingmoe", "deepseek", "maincoder"] {
            v.push(gqa_norm(n));
        }
        // Were FIXTURE-AWAY in this group and now have the fixture:
        // `tests/fixture_away_graphs.rs`, same evidence standard.
        for n in ["baichuan", "ernie4_5", "internlm2", "xverse"] {
            v.push(gqa_norm(n));
        }
        // `ernie4_5-moe` was ONE MATCH ARM in `NORM_ROPE_TRIAGED` and is
        // audited now: the step every real checkpoint carries has a
        // libllama-golden fixture (`tests/one_match_arm_graphs.rs`) and
        // any other step is refused by name (`crate::moe_interleave`).
        v.push(gqa_norm("ernie4_5-moe"));
        // `chatglm` was the LAST ONE MATCH ARM row anywhere in this
        // file. Its arm -- the fused `attn_qkv.bias` -- landed in
        // `crate::qkv_fused` and has a libllama-golden fixture
        // (`tests/one_match_arm_graphs.rs`), so the class is empty now.
        v.push(gqa_norm("chatglm"));
        // The Granite family. All three were NEW CODE in
        // `NORM_ROPE_TRIAGED` on the four scalar multipliers, which
        // `crate::scalar_multipliers` now implements once for all of
        // them (`tests/granite_family_graphs.rs`). `granite-moe` is a
        // ferrox-only alias -- `llama-arch.cpp:101` spells it
        // `granitemoe` -- and is here rather than anywhere else so it
        // cannot be given a different path from the row it aliases.
        for n in ["granite", "granitemoe", "granite-moe"] {
            v.push(gqa_norm(n));
        }
        // OLMo-1 was NEW CODE in `NORM_ROPE_TRIAGED` on its
        // non-parametric LayerNorm, which `crate::norm::NormOp` now
        // implements (`tests/olmo_graphs.rs`). NORM RoPE:
        // `llama_model_rope_type` puts LLM_ARCH_OLMO in the
        // consecutive-pairs group (llama-model.cpp:2585), which is also
        // why `conversion/olmo.py:33-36` permutes q_proj and k_proj the
        // way `LlamaModel` does. Its `olmo.attention.clamp_kqv` was
        // refused by name and is applied now (`crate::clamp_kqv`),
        // since `dbrx` needed the same clamp.
        v.push(gqa_norm("olmo"));
        // `smollm3` was refused OUTRIGHT, in the "No RoPE at all" group
        // below, and it was the only row there whose graph is the plain
        // pre-norm llama one. What it needed was a way to say WHICH
        // LAYERS ROTATE: `smollm3.cpp:5,69` skip `(il + 1) % 4 == 0`,
        // nine layers of a 36-layer SmolLM3-3B, with no GGUF key.
        // `crate::rope_layers` says it now, once, for the six
        // architectures llama.cpp gates per layer, and
        // `tests/no_rope_layer_graphs.rs` is the libllama-golden
        // evidence. NORM RoPE: llama-model.cpp puts LLM_ARCH_SMOLLM3 in
        // the consecutive-pairs group (:2600).
        v.push(gqa_norm("smollm3"));
        // `arcee` was NEW CODE in `NORM_ROPE_TRIAGED` on the ungated
        // ReLU-squared FFN, which `FfnActivation::ReluSqr` implements
        // (`tests/ungated_ffn_graphs.rs`). NORM RoPE: LLM_ARCH_ARCEE is
        // in the consecutive-pairs group (llama-model.cpp:2600).
        v.push(gqa_norm("arcee"));
        // `deci` was NEW CODE in `NORM_ROPE_TRIAGED` on per-layer
        // shapes, which `crate::layer_shapes` implements
        // (`tests/per_layer_shape_graphs.rs`). NORM RoPE: LLM_ARCH_DECI
        // is in the consecutive-pairs group (llama-model.cpp:2576).
        v.push(gqa_norm("deci"));
        // `mistral3` was NEW CODE in `NORM_ROPE_TRIAGED` on the
        // per-position attention temperature, which
        // `crate::attn_temperature` implements
        // (`tests/attn_temperature_graphs.rs`). NORM RoPE:
        // LLM_ARCH_MISTRAL3 is in the consecutive-pairs group
        // (llama-model.cpp:2604), which `tests/rope_layout.rs` pins.
        v.push(gqa_norm("mistral3"));
        // Same generic Norm-RoPE path, but READ against llama.cpp's own
        // graph -- see [`TriageClass`]. Each row below refuses with its
        // class and its blocker instead of the generic
        // "nobody has checked this" paragraph.
        for (n, class, blocker) in NORM_ROPE_TRIAGED {
            v.push(gqa_norm(n).triaged(*class, blocker));
        }
        for n in [
            "olmoe", "qwen2", "qwen2moe",
            // llama-model.cpp `llama_model_rope_type`: LLM_ARCH_OPENAI_MOE
            // falls in the `return LLAMA_ROPE_TYPE_NEOX` group, and a live
            // load of a gpt-oss GGUF prints `rope type = 2` (= NEOX).
            // ferrox had it on the interleaved (NORM) list, which rotates
            // the wrong pairs of every Q/K head.
            "gpt-oss",
            // Same audit, run over every arch at once against
            // `llama_model_rope_type`'s NEOX group
            // (llama-model.cpp:2613-2683). These 24 were on ferrox's
            // interleaved (NORM) list and reach the generic GQA decoder,
            // so every one of them rotated the wrong pairs of every Q/K
            // head and answered fluently and wrongly. Pinned by
            // `rope_layout_matches_llama_cpp` below; dots1 additionally
            // checked end-to-end against llama.cpp's own logits in
            // `tests/moe_routing_bias.rs`.
            "dots1",
            // Audited by libllama-golden fixtures in
            // `tests/one_match_arm_graphs.rs`: `hunyuan-moe` needed the
            // post-RoPE QK-norm order, `seed_oss` the gpt-oss pre-FFN
            // norm slot.
            "hunyuan-moe",
            "seed_oss",
            // `hunyuan-dense` was ONE MATCH ARM in `NEOX_ROPE_TRIAGED`
            // and is audited now: the NTK-alpha RoPE base rescale
            // (`crate::rope_ntk_alpha`) plus the post-RoPE QK-norm order
            // it shares with `hunyuan-moe`, both against libllama's own
            // logits.
            "hunyuan-dense",
            // Were FIXTURE-AWAY and now have the fixture
            // (`tests/fixture_away_graphs.rs`). EXAONE 3.x only:
            // `exaone4` and `exaone-moe` are different graphs and stay
            // in `NEOX_ROPE_TRIAGED` below. `bailingmoe2` is Ling-2.0
            // and is unrelated to the NORM-RoPE `bailingmoe` row above.
            "exaone",
            "bailingmoe2",
            "plamo3",
            // Were NEW CODE in `NEOX_ROPE_TRIAGED` and are audited now.
            // One residual topology, `crate::norm`, shared by both:
            // no pre-attention norm and no pre-FFN norm, each branch's
            // OUTPUT normed before its residual add. The evidence is
            // `tests/post_norm_only_graphs.rs`, one libllama-golden
            // fixture each. `olmo2` with a sliding window AND a RoPE
            // scaling (Olmo-3) and `exaone4` with 64 layers (the 32B)
            // are refused by name in `loader.rs` and are NOT covered by
            // these two rows.
            "olmo2",
            "exaone4",
            // Was NEW CODE in `NEOX_ROPE_TRIAGED` on ONE blocker: its
            // GLOBAL layers get no RoPE (`exaone-moe.cpp:136,155-161`).
            // That is the SAME RULE as `exaone4`'s -- :4 pins
            // `swa_type` to STANDARD, which makes `exaone4.cpp:116`'s
            // second disjunct false and the two predicates identical --
            // so both rows take one implementation,
            // `crate::rope_layers`, with a libllama-golden fixture each
            // in `tests/no_rope_layer_graphs.rs`. Everything else it
            // needed (leading dense, `exp_probs_b`, shared expert,
            // sigmoid gating from metadata, a per-head QK-norm) ferrox
            // already had, and its fixture carries all of it rather
            // than asserting so. The two things a REAL export carries
            // on top -- K-EXAONE's one NextN block inside `block_count`
            // (`exaone.py:132,146`) and the window pattern as a bool
            // ARRAY (`:84`) that `exaone-moe.cpp:7` never reads -- are
            // `crate::mtp_blocks` and `crate::swa_layers`, with a
            // fixture carrying both (`tests/window_array_graphs.rs`).
            "exaone-moe",
            // Was a `DedicatedOnly` bias refusal, not an unaudited row:
            // its only dropped bias was the FUSED `attn_qkv.bias`, which
            // `crate::qkv_fused` applies now. Qwen-1 only; qwen2 and
            // later store the split spelling and were already audited.
            "qwen",
            // Were NEW CODE in `NEOX_ROPE_TRIAGED` and are audited now,
            // each on seams that landed the day before: `dbrx` on the
            // weighted LayerNorm (`crate::norm`), the QKV clamp
            // (`crate::clamp_kqv`) and the `attn_output_norm` slot
            // (`crate::norm_sites`); `grok` on the defaults hook
            // (`scalar_multipliers::MultiplierDefaults::Grok`), the
            // scale-inside-softcap attention and the same `norm_sites`
            // table. NEOX RoPE: llama-model.cpp:2616-2617 put both in
            // the `n_rot/2`-offset group. `tests/dbrx_graphs.rs`,
            // `tests/grok_graphs.rs`.
            "dbrx",
            "grok",
            // Was NEW CODE in `NEOX_ROPE_TRIAGED` on per-layer shapes,
            // audited now with `deci` on `crate::layer_shapes`
            // (`tests/per_layer_shape_graphs.rs`). NEOX RoPE:
            // llama-model.cpp:2650.
            "openelm",
            // Were NEW CODE in `NEOX_ROPE_TRIAGED` on the gated
            // attention, audited now on `crate::attn_gate`
            // (`tests/gated_attention_graphs.rs`). NEOX RoPE:
            // llama-model.cpp:2676-2677.
            "afmoe",
            "laguna",
            // Was NEW CODE in `NEOX_ROPE_TRIAGED` on the sliding-window
            // ARRAY (`mellum.cpp:12-17`), audited now on
            // `crate::swa_layers` (`tests/window_array_graphs.rs`); its
            // window-with-YaRN half stays refused by name. NEOX RoPE:
            // llama-model.cpp:2682.
            "mellum",
            // Was NEW CODE in `NEOX_ROPE_TRIAGED` on xIELU's per-layer
            // parameter arrays, audited now on `crate::act_layers`
            // (`tests/per_layer_activation_graphs.rs`). NEOX RoPE:
            // llama-model.cpp:2671.
            "apertus",
            "step35",
        ] {
            v.push(gqa_neox(n));
        }
        // Triaged NEOX-RoPE rows; see `NORM_ROPE_TRIAGED` above.
        for (n, class, blocker) in NEOX_ROPE_TRIAGED {
            v.push(gqa_neox(n).triaged(*class, blocker));
        }
        // --- No RoPE at all: refused, not rotated ------------------
        //
        // `llama_model_rope_type` opens with a `LLAMA_ROPE_TYPE_NONE`
        // group, and these five sat on ferrox's NEOX list instead. The
        // generic decoder rotates every Q/K head of every layer, so each
        // of them loaded, ran at full speed, and answered fluently from
        // positions the checkpoint never encodes that way, the same
        // silent failure the 24-arch RoPE audit found, one level worse,
        // because here the right answer is *no rotation*.
        //
        // Worse still for a metadata gate: `bloom` and `refact` hardcode
        // `f_max_alibi_bias = 8.0f` in `load_arch_hparams` and carry no
        // key at all, so `unsupported_feature_keys` could never have seen
        // them. Only the registry can. `tests/rope_layout.rs`'s
        // `LLAMA_NO_ROPE` pins the group so a later edit cannot quietly
        // put one back on a rotating path.
        for (n, reason) in [
            (
                "gpt2",
                "learned absolute position embeddings (`position_embd.weight`, \
                 src/models/gpt2.cpp:19,74) and no RoPE; the generic decoder has no \
                 slot for them and rotates instead",
            ),
            (
                "mpt",
                "ALiBi attention bias (src/models/mpt.cpp:6), plus an optional \
                 learned `position_embd` and an optional QKV clamp; the generic \
                 decoder implements none of the three and applies RoPE instead",
            ),
            (
                "refact",
                "ALiBi attention bias, hardcoded `f_max_alibi_bias = 8.0f` with no \
                 GGUF key to detect it (src/models/refact.cpp:12); the generic \
                 decoder applies RoPE instead",
            ),
            (
                "bloom",
                "ALiBi attention bias, hardcoded `f_max_alibi_bias = 8.0f` with no \
                 GGUF key (src/models/bloom.cpp:18), plus a `token_embd_norm` the \
                 generic decoder never applies; RoPE is applied instead",
            ),
            (
                "jais",
                "ALiBi attention bias (src/models/jais.cpp:5); the generic decoder \
                 applies RoPE instead",
            ),
        ] {
            v.push(prof(
                n,
                TextGeneration,
                StandardGqa,
                KvGqa,
                // No layout is right here. `Norm` is the struct's least
                // surprising filler and nothing reads it: the load
                // refuses in `ModelConfig::from_gguf` before any graph
                // asks. `rope_layout_matches_llama_cpp` skips
                // non-generic paths for exactly this reason.
                Norm,
                ArchPath::DedicatedOnly { reason },
                WholeVector,
            ));
        }
        // --- Required bias tensors the generic decoder has no slot for
        //
        // Found by transcribing every `create_tensor(tn(..., "bias"), ...)`
        // llama.cpp's per-architecture loaders create with flag `0`
        // (REQUIRED, as opposed to `TENSOR_NOT_REQUIRED`). Required means
        // every real checkpoint of that architecture carries it, so this
        // is not a "some files might" gate.
        //
        // `AttnWeights` carries exactly three of them -- `attn_q.bias`,
        // `attn_k.bias`, `attn_v.bias` -- and `GptOssWeights` carries
        // gpt-oss's `attn_output.bias` and `ffn_gate_inp.bias`. Nothing
        // else has anywhere to go:
        //
        // - `attn_output.bias`, `ffn_{up,down,gate}.bias` and the
        //   `output.bias` on the LM head are read by no loader path, so
        //   they are simply dropped: the projection runs unbiased.
        // - `attn_norm.bias` / `ffn_norm.bias` / `output_norm.bias` are
        //   the marker of a real LayerNorm. The generic decoder only has
        //   `rms_norm(x, w, eps)` -- no mean subtraction and no bias --
        //   so it computes a different normalisation at every layer.
        // - `attn_qkv.bias` is the *fused* spelling, and it IS applied
        //   now: `qkv_fused::load_fused_or_split_qkv` resolves the
        //   projections and their biases from one decision about which
        //   spelling the file uses, and slices the fused bias by the
        //   same spans as the fused weight. Until 2026-09-10 it did not,
        //   and that is what refused `qwen` and `chatglm`; both are
        //   audited now. `tests/attn_bias.rs`'s
        //   `GENERIC_DECODER_APPLIES` carries the name, so a row whose
        //   ONLY missing bias is this one no longer counts as dropped.
        //   `starcoder` and `bloom` still require five more each.
        //
        // Every one of these loads clean and answers fluently, which is
        // why they are refused here rather than left to a tensor gate.
        // Pinned by `tests/attn_bias.rs`.
        for (n, rope, reason) in [
            (
                "codeshell",
                Neox,
                "required bias tensors with no slot in the generic decoder: \
                 `attn_output.bias`, `ffn_down.bias`, `ffn_up.bias` \
                 (src/models/codeshell.cpp:36,42,45), plus the LayerNorm biases \
                 `output_norm.bias`, `attn_norm.bias`, `ffn_norm.bias` (:24,31,39) \
                 -- the generic decoder is RMSNorm-only and drops all six",
            ),
            (
                "jais2",
                Neox,
                "required bias tensors with no slot in the generic decoder: \
                 `attn_output.bias`, `ffn_up.bias`, `ffn_down.bias` \
                 (src/models/jais2.cpp:41,48,50), plus the LayerNorm biases \
                 `output_norm.bias`, `attn_norm.bias`, `ffn_norm.bias` (:20,30,44). \
                 Only its Q/K/V biases (:38-40) would have been applied",
            ),
            (
                "starcoder",
                Norm,
                "required bias tensors with no slot in the generic decoder. NOT the \
                 *fused* `attn_qkv.bias` (src/models/starcoder.cpp:40) any more -- \
                 `qkv_fused` applies that one now -- but `attn_output.bias`, \
                 `ffn_down.bias`, `ffn_up.bias` (:43,49,52) and the LayerNorm \
                 biases `output_norm.bias`, `attn_norm.bias`, `ffn_norm.bias` \
                 (:24,37,46). It also adds a learned `position_embd` to the \
                 embeddings (:75) that the generic decoder has no slot for",
            ),
            (
                "starcoder2",
                Neox,
                "required bias tensors with no slot in the generic decoder: \
                 `attn_output.bias`, `ffn_down.bias`, `ffn_up.bias` \
                 (src/models/starcoder2.cpp:41,50,51), plus the LayerNorm biases \
                 `output_norm.bias`, `attn_norm.bias`, `ffn_norm.bias` (:23,35,44)",
            ),
            (
                "phimoe",
                Neox,
                "required bias tensors with no slot in the generic decoder: \
                 `attn_output.bias` and an `output.bias` on the LM head \
                 (src/models/phimoe.cpp:33,23), plus the LayerNorm biases \
                 `output_norm.bias`, `attn_norm.bias`, `ffn_norm.bias` (:21,29,36). \
                 `phi3` stays generic: it requires none of them",
            ),
            (
                "nemotron",
                Neox,
                "required LayerNorm biases `output_norm.bias`, `attn_norm.bias`, \
                 `ffn_norm.bias` (src/models/nemotron.cpp:19,26,35). llama.cpp \
                 normalises with `build_norm(..., LLM_NORM, ...)` and a bias; the \
                 generic decoder applies RMSNorm with weight only, which is a \
                 different function of the same tensors at every layer",
            ),
            (
                "orion",
                Neox,
                "required LayerNorm biases `output_norm.bias`, `attn_norm.bias`, \
                 `ffn_norm.bias` (src/models/orion.cpp:18,25,31); the generic \
                 decoder is RMSNorm-only and drops all three",
            ),
            (
                "stablelm",
                Neox,
                "required LayerNorm biases `output_norm.bias` and `attn_norm.bias` \
                 (src/models/stablelm.cpp:20,28); the generic decoder is \
                 RMSNorm-only and drops both",
            ),
        ] {
            v.push(prof(
                n,
                TextGeneration,
                StandardGqa,
                KvGqa,
                // Unlike the no-RoPE group above, the layout here is
                // real and `rope_layout_matches_llama_cpp` still checks
                // it: refusing for a bias is not a licence to forget
                // what these rotate as.
                rope,
                ArchPath::DedicatedOnly { reason },
                WholeVector,
            ));
        }
        v.push(prof(
            "qwen3",
            TextGeneration,
            Qwen3Family,
            KvGqa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            PerHead,
        ));
        v.push(prof(
            "qwen3moe",
            TextGeneration,
            Qwen3Family,
            KvGqa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            PerHead,
        ));
        // `gemma` was FIXTURE-AWAY here until it got its fixture
        // (`tests/fixture_away_graphs.rs`); it is audited now and
        // carries no verdict at all.
        v.push(prof(
            "gemma",
            TextGeneration,
            GemmaFamily,
            KvGqa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            PerHead,
        ));
        v.push(prof(
            "gemma2",
            TextGeneration,
            GemmaFamily,
            KvIswa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            PerHead,
        ));
        v.push(prof(
            "gemma3",
            TextGeneration,
            GemmaFamily,
            KvIswa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            PerHead,
        ));
        // Gemma-4 text GGUFs (E2B): per-layer embeddings, shared-KV
        // layers, and split SWA/full head dims -- dedicated
        // [`crate::gemma4_engine::Gemma4Engine`] (not GenericGqa).
        for n in ["gemma4", "gemma4-assistant"] {
            v.push(prof(
                n,
                TextGeneration,
                GemmaFamily,
                KvIswa,
                Neox,
                ArchPath::DedicatedOnly {
                    reason: "use load_gemma4_engine_from_path / ServedEngine::Gemma4",
                },
                PerHead,
            ));
        }
        // Refused, not implemented: the generic decoder computes
        // `x + attn(norm(x))` then `y + ffn(norm(y))`, and every arch
        // here computes something else that no tensor and (for MiniCPM)
        // no metadata key makes visible. See
        // `unsupported_scaling_keys` for the metadata-visible half of
        // the same class.
        const PARALLEL_RESIDUAL: &str =
            "parallel attention+FFN residual -- llama.cpp feeds both branches the *same* \
             normed input and sums `inpL + attn_out + ffn_out` once; the generic decoder \
             computes the sequential form, which is a different graph";
        for (n, rope, fam) in [
            // src/models/cohere2.cpp:120-134, cohere2moe.cpp:222-266,
            // command-r.cpp:106-119. All three also carry a
            // `logit_scale` the generic decoder does not apply.
            ("command-r", Norm, StandardGqa),
            ("cohere2", Norm, StandardGqa),
            ("cohere2moe", Norm, StandardGqa),
            // src/models/falcon.cpp:121-135 (and an `attn_norm_2` the
            // generic decoder has no slot for).
            ("falcon", Neox, StandardGqa),
            // src/models/gptneox.cpp:147-195 -- parallel or sequential
            // per `use_par_res`, and the generic decoder implements
            // neither branch of that choice.
            ("gptneox", Neox, StandardGqa),
            // src/models/phi2.cpp:116-117, plamo.cpp:97-112.
            ("phi2", Neox, PhiFamily),
            ("plamo", Neox, StandardGqa),
        ] {
            v.push(prof(
                n,
                TextGeneration,
                fam,
                KvGqa,
                rope,
                ArchPath::DedicatedOnly {
                    reason: PARALLEL_RESIDUAL,
                },
                WholeVector,
            ));
        }
        // MiniCPM was the case `unsupported_scaling_keys` cannot catch:
        // `src/models/minicpm.cpp:5-7` *hardcodes* an embedding
        // multiplier of 12.0, a residual multiplier of
        // `1.4/sqrt(n_layer)` and a logit multiplier of `256/n_embd`,
        // and only then (`:12-14`) lets the GGUF override them. An older
        // MiniCPM export carrying none of the three keys is still scaled
        // by all three, so a key-presence gate sees nothing.
        //
        // It is generic now, on the same evidence every other row here
        // has: `scalar_multipliers::MultiplierDefaults` applies the
        // three, and `tests/minicpm_graphs.rs` drives a fixture that
        // declares NONE of them against llama.cpp's own logits. Its RoPE
        // is NORM (`llama_model_rope_type`, llama-model.cpp:2580, the
        // consecutive-pairs group), and it is deliberately NOT in
        // `rope_finetuned::ROPE_GATED_ON_FINETUNED`: it runs Granite's
        // graph, whose RoPE is gated on `hparams.rope_finetuned`, but
        // `minicpm.cpp:17` pins that true with no key read at all, so
        // the switch Granite exposes is unreachable here.
        v.push(gqa_norm("minicpm"));
        v.push(prof(
            "phi3",
            TextGeneration,
            PhiFamily,
            KvGqa,
            Neox,
            ArchPath::GenericGqa { rope: Neox },
            WholeVector,
        ));
        // Phi-4 GGUFs share the phi3 fused-QKV / fused gate+up graph
        // (PhiFamily). Many community checkpoints still tag `phi3`; admit
        // `phi4` the same way so either string can load. Receipts / head-dim
        // FA-vec coverage remain P6 evidence work -- not a speed claim.
        v.push(
            prof(
                "phi4",
                TextGeneration,
                PhiFamily,
                KvGqa,
                Neox,
                ArchPath::GenericGqa { rope: Neox },
                WholeVector,
            )
            .triaged(
                TriageClass::Unknown,
                "there is no llama.cpp graph to diff against. `phi4` is NOT in LLM_ARCH_NAMES \
                 -- src/llama-arch.cpp:44 lists \"phi3\" and there is no phi4 entry -- so this \
                 row is a ferrox-only alias and no llama.cpp-produced GGUF can carry the \
                 string. ferrox admits it as PhiFamily/NEOX, i.e. phi3's fused-QKV and fused \
                 gate+up graph, on the assumption that a file spelling it means the same \
                 thing. WHAT WOULD SETTLE IT: a real GGUF whose general.architecture is \
                 literally `phi4`. If its blk.0 carries attn_qkv.weight it is phi3's graph \
                 and this row is fixture-away behind an already-audited phi3; if it carries \
                 split attn_q/attn_k/attn_v it is a Llama-shaped graph and belongs on a \
                 different row",
            ),
        );
        // Llama 4: MoE + interleaved / non-generic attention graph -- not
        // safe to admit as GenericGqa (was wrongly listed with plain llama).
        v.push(prof(
            "llama4",
            TextGeneration,
            Dedicated,
            KvGqa,
            Norm,
            ArchPath::DedicatedOnly {
                reason: "llama4 MoE + chunked attention -- see llama4_engine.rs tensor list. \
                         Its attention temperature (llama4.cpp:15-17: scale 0.1, floor 8192, \
                         offset 1.0, from LITERALS, applied at :175-176 to the no-RoPE layers \
                         only) is `crate::attn_temperature` plus a per-layer gate on \
                         `ModelConfig::layer_rotates`; what it still needs is the chunked \
                         SWA (`LLAMA_SWA_TYPE_CHUNKED`, :13) and the interleaved MoE",
            },
            WholeVector,
        ));
        // MiniMax M2 and M3 are two DIFFERENT architectures and were
        // wrong to share one reason. Both used to refuse with "256-expert
        // sigmoid MoE + MTP"; neither clause is true.
        //
        // MTP: `minimax-m2.cpp` and `minimax-m3.cpp` create no `nextn.*`
        // tensor at all, and `gguf-py/gguf/constants.py`'s
        // `MODEL_ARCH.MINIMAXM2` / `.MINIMAXM3` tensor lists contain no
        // `NEXTN_*` entry -- so no converter can even emit MTP weights for
        // these files. `minimax-m3.cpp:9` says it outright: "MTP is not
        // in released model weights."
        //
        // Sigmoid MoE: ferrox HAS it. `loader.rs` reads
        // `{arch}.expert_gating_func` into `GatingFunction::Sigmoid`,
        // loads `blk.N.exp_probs_b.bias`, and reads
        // `expert_weights_scale` / `expert_weights_norm`. Expert count is
        // an hparam, not a ceiling.
        //
        // llama-arch.cpp puts both in the NEOX RoPE group.
        v.push(prof(
            "minimax-m2",
            TextGeneration,
            Dedicated,
            KvGqa,
            Neox,
            ArchPath::DedicatedOnly {
                // `minimax-m2.cpp` is plain GQA: `create_tensor_qkv` at
                // :26, whole-vector Q/K norm at :30-31 (`attn_q_norm` is
                // `n_embd_head_k * n_head` wide, NOT per-head), partial
                // NEOX RoPE at :96-106 (:51 notes head_dim=128 but
                // n_rot=64), and one SiLU MoE with `exp_probs_b`,
                // `expert_weights_scale` and norm_w=true at :131-141.
                // ferrox implements every one of those on the generic
                // path. What is missing is EVIDENCE, not capability.
                reason: "minimax-m2 is UNAUDITED, not unimplemented: llama.cpp's minimax-m2.cpp \
                         builds plain GQA + whole-vector QK-norm + partial NEOX RoPE (n_rot=64 < \
                         head_dim=128) + a SiLU sigmoid MoE with exp_probs_b, all of which the \
                         generic path already has. Admitting it needs a fixture or a parity run \
                         against llama.cpp, not new code",
            },
            // `attn_q_norm` is `{n_embd_head_k * n_head}` wide
            // (minimax-m2.cpp:30) -- one RMSNorm over the whole Q
            // projection, OLMoE's style, not Qwen3's per-head.
            WholeVector,
        ));
        v.push(prof(
            "minimax-m3",
            TextGeneration,
            Dedicated,
            KvGqa,
            Neox,
            ArchPath::DedicatedOnly {
                reason: "minimax-m3 needs MiniMax Sparse Attention: a per-layer indexer \
                         (index_q_proj/index_k_proj/index_q_norm/index_k_norm, minimax-m3.cpp:76-82) \
                         driving its own MSA KV cache (llama-kv-cache-msa.h) with position<->cell \
                         maps, plus SWIGLU_OAI experts and shared experts. ferrox has only the \
                         block-selection rule (ferrox_core::block_sparse), none of the rest",
            },
            // minimax-m3.cpp:53-55 -- `{n_embd_head_k}`, with llama.cpp's
            // own comment "per-head QK-norm: a single head_dim vector
            // applied to every head". M2 and M3 DIFFER here, which is why
            // the shared entry was wrong for M3.
            PerHead,
        ));
        // MiniCPM3 is MLA, not generic GQA, and the catalog said
        // otherwise: it claimed `StandardGqa`/`KvGqa`, which is false
        // about the model rather than merely unaudited.
        // `src/models/minicpm3.cpp:5-6` requires `q_lora_rank` and
        // `kv_lora_rank`, and `:41-46` creates
        // `attn_q_a`/`attn_q_b`/`attn_kv_a_mqa`/`attn_kv_b` -- the
        // DeepSeek-2 tensor set. There is no `attn_q.weight` in any
        // MiniCPM3 checkpoint, so the generic path could never have
        // loaded one whatever the audit said.
        //
        // Reclassified 2026-09-01 by the unaudited-refusal triage. This
        // is a MESSAGE-QUALITY fix, not a correctness one: the old
        // failure was already a clean missing-tensor error. It stops the
        // user being told "unaudited" for something that is not merely
        // unaudited.
        v.push(prof(
            "minicpm3",
            TextGeneration,
            Mla,
            KvMla,
            Neox,
            ArchPath::DedicatedOnly {
                reason: "MiniCPM3 is an MLA model (src/models/minicpm3.cpp:5-6,41-46 -- \
                         q_lora_rank/kv_lora_rank and the attn_q_a/attn_q_b/attn_kv_a_mqa/\
                         attn_kv_b tensor set), so it needs the MLA engine and not the \
                         generic GQA decoder. It ALSO hardcodes MiniCPM's multipliers with \
                         no GGUF key to read them from -- scale_embd = 12.0, \
                         scale_depth = 1.4, n_embd_base = 256 at :65-67, applied at :81 -- \
                         which is the same blind spot `minicpm` is refused for",
            },
            WholeVector,
        ));
        v.push(prof(
            "deepseek2",
            TextGeneration,
            Mla,
            KvMla,
            Norm,
            ArchPath::DedicatedOnly {
                reason: "DeepSeek-2 MLA needs the MLA engine, not generic GQA",
            },
            WholeVector,
        ));
        v.push(prof(
            "deepseek32",
            TextGeneration,
            Mla,
            KvDsa,
            Norm,
            ArchPath::DedicatedOnly {
                reason: "DeepSeek-3.2 DSA/MLA needs the dedicated sparse/MLA stack",
            },
            WholeVector,
        ));
        v.push(prof(
            "mistral4",
            TextGeneration,
            Mla,
            KvMla,
            Norm,
            ArchPath::DedicatedOnly {
                reason: "mistral4 reuses DeepSeek-2 MLA loader/graph in llama.cpp",
            },
            WholeVector,
        ));
        // The three ferrox-only alias rows. Refused as STRINGS, not
        // triaged as architectures: libllama refuses all three outright
        // and every real checkpoint of all three declares `llama`. See
        // `NO_UPSTREAM_ARCH` for the three measurements. Note the
        // `dedicated` helper gives them NORM RoPE, which is at least the
        // layout of the graph they claim to be; they had NEOX while
        // sitting on the generic path.
        for n in ["mistral", "mixtral", "yi"] {
            v.push(dedicated(n, NO_UPSTREAM_ARCH));
        }
        v.push(dedicated(
            "glm-dsa",
            "use ferrox_models::glm52_decoder / glm52_gguf_loader (DSA), not the generic GQA Decoder",
        ));
        v.push(dedicated(
            "glm4",
            "use ferrox_models::glm52_decoder / glm52_gguf_loader, not the generic GQA Decoder",
        ));
        // GLM-4.5 / GLM-4.5-Air / GLM-4.6 tag `glm4moe`, and the reason
        // here used to point at `glm52_gguf_loader` the way `glm-dsa`
        // does. It cannot load one: `read_glm52_hparams` requires
        // `{arch}.attention.q_lora_rank`, `.kv_lora_rank`,
        // `.qk_nope_head_dim` and `.qk_rope_head_dim`, and glm4moe is
        // NOT an MLA model -- `src/models/glm4-moe.cpp`'s
        // `load_arch_hparams` never reads any of the four and its
        // `load_arch_tensors` calls `create_tensor_qkv` (plain Q/K/V)
        // with no `attn_kv_a_mqa` / `attn_kv_b` / `attn_q_a` /
        // `attn_q_b` anywhere. So `ferrox run` on a real GLM-4.5-Air
        // answered "missing hparam glm4moe.attention.q_lora_rank" for a
        // model that has no MLA at all.
        //
        // What it actually is: plain GQA + DeepSeek-V3-shaped sigmoid
        // MoE (`exp_probs_b`, shared expert, leading dense,
        // `expert_weights_scale`), all of which the generic decoder
        // already computes and `dots1` already pins. The one thing that
        // does not fit is the norm slot, and it is a real divergence
        // rather than a missing key -- see the reason string. Pinned by
        // `tests/glm4moe_refusal.rs` against a synthetic checkpoint
        // llama.cpp itself loads and decodes.
        v.push(dedicated(
            "glm4moe",
            "GLM-4.5-MoE stores its pre-FFN norm as `blk.N.post_attention_norm.weight` and \
             carries NO `blk.N.ffn_norm.weight` (src/models/glm4-moe.cpp:75, applied to \
             `ffn_inp` at :215 -- i.e. AFTER the attention residual). The generic decoder \
             requires `ffn_norm` and puts `post_attention_norm` in Gemma's other slot, on the \
             attention branch BEFORE the residual add, so it would both fail to find its \
             tensors and compute a different graph. This is gpt-oss's norm slot exactly, and \
             `loader.rs` already implements it behind an `is_gpt_oss` flag; widening that flag \
             is what admits glm4moe. It is NOT MLA -- do not send it to glm52_gguf_loader, \
             which asks for a `q_lora_rank` no glm4moe checkpoint carries",
        ));
        v.push(dedicated(
            "deepseek4",
            "DeepSeek V4 needs CSA/HCA + mHC assembly; generic GQA Decoder is not valid",
        ));
        v.push(dedicated(
            "kimi-linear",
            "use ferrox_models::kimi_decoder / kimi_loader, not the generic GQA Decoder",
        ));
        v.push(dedicated(
            "kimi_k3",
            "use ferrox_models::kimi_decoder / kimi_loader, not the generic GQA Decoder",
        ));
        for (n, rope) in [
            ("jamba", Neox),
            ("falcon-h1", Neox),
            ("plamo2", Neox),
            ("granitehybrid", Norm),
            ("granite-hybrid", Norm),
            ("lfm2", Neox),
            ("lfm2moe", Neox),
            ("nemotron_h", Neox),
            ("nemotron_h_moe", Neox),
            ("qwen3next", Neox),
            ("qwen35", Neox),
            ("qwen35moe", Neox),
        ] {
            let qk = if n.starts_with("qwen3") {
                PerHead
            } else {
                WholeVector
            };
            v.push(prof(
                n,
                TextGeneration,
                DecoderFamily::Hybrid,
                MemoryKind::Hybrid,
                rope,
                ArchPath::DedicatedOnly {
                    reason: "hybrid attn+SSM/delta-net engine not yet on the serve path",
                },
                qk,
            ));
        }
        for n in ["mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7"] {
            v.push(prof(
                n,
                TextGeneration,
                DecoderFamily::Recurrent,
                MemoryKind::Recurrent,
                Neox,
                ArchPath::DedicatedOnly {
                    reason: "recurrent engine not yet on the serve path",
                },
                WholeVector,
            ));
        }
        v.push(prof(
            "t5",
            TextGeneration,
            EncoderDecoder,
            None,
            Neox,
            ArchPath::DedicatedOnly {
                reason: "T5 encoder-decoder engine not yet on the serve path",
            },
            WholeVector,
        ));
        for (n, scope, reason) in [
            (
                "t5encoder",
                DeferredEncoderEmbedding,
                "encoder-only; deferred from text-generation parity",
            ),
            // Deferred from the *decoder* path, and that is still
            // right: a `bert` GGUF has no output head, so
            // `ensure_generic_decoder` must keep refusing it. It is no
            // longer deferred outright -- it loads and embeds through
            // `bert_gguf_loader` / `bert_encoder`, checked against
            // llama.cpp by `tests/bert_llama_cpp_parity.rs`.
            (
                "bert",
                DeferredEncoderEmbedding,
                "encoder; no output head, so never a decoder -- served by \
                 ferrox_models::EmbeddingModel on /v1/embeddings",
            ),
            (
                "modern-bert",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "nomic-bert",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "nomic-bert-moe",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "neo-bert",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "jina-bert-v2",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "jina-bert-v3",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "eurobert",
                DeferredEncoderEmbedding,
                "encoder/embedding; deferred",
            ),
            (
                "llama-embed",
                DeferredEncoderEmbedding,
                "embedding variant; deferred",
            ),
            (
                "gemma-embedding",
                DeferredEncoderEmbedding,
                "embedding variant; deferred",
            ),
            (
                "pangu-embedded",
                DeferredEncoderEmbedding,
                "embedding variant; deferred",
            ),
            ("yi-vl", DeferredMultimodal, "Yi vision-language; deferred"),
            ("qwen2vl", DeferredMultimodal, "vision-language; deferred"),
            ("qwen3vl", DeferredMultimodal, "vision-language; deferred"),
            ("qwen3vlmoe", DeferredMultimodal, "vision-language; deferred"),
            ("cogvlm", DeferredMultimodal, "vision-language; deferred"),
            ("chameleon", DeferredMultimodal, "multimodal; deferred"),
            ("hunyuan_vl", DeferredMultimodal, "vision-language; deferred"),
            ("paddleocr", DeferredMultimodal, "OCR multimodal; deferred"),
            ("hy_v3", DeferredMultimodal, "multimodal; deferred"),
            ("deepseek2-ocr", DeferredMultimodal, "OCR multimodal; deferred"),
            ("dream", DeferredDiffusion, "diffusion LM; deferred"),
            ("llada", DeferredDiffusion, "diffusion LM; deferred"),
            ("llada-moe", DeferredDiffusion, "diffusion LM; deferred"),
            ("rnd1", DeferredDiffusion, "diffusion LM; deferred"),
            (
                "wavtokenizer-dec",
                DeferredAudio,
                "audio tokenizer; deferred",
            ),
            (
                "eagle3",
                EnumOnly,
                "speculative draft head; not a standalone decoder target",
            ),
            (
                "dflash",
                EnumOnly,
                "speculative draft head; not a standalone decoder target",
            ),
            ("clip", EnumOnly, "quantize dummy only"),
            ("gptj", EnumOnly, "enum-only in llama.cpp factory gap"),
            ("(unknown)", EnumOnly, "llama.cpp unknown sentinel"),
        ] {
            v.push(deferred_scope(n, scope, reason));
        }
        v.push(prof(
            "gemma3n",
            TextGeneration,
            GemmaFamily,
            KvIswa,
            Neox,
            ArchPath::DedicatedOnly {
                reason: "gemma3n AltUp/Laurel tensors not implemented in the generic decoder",
            },
            PerHead,
        ));
        for n in ["ferroxtest", "ferroxtestmoe", "ferroxtestmixed"] {
            v.push(prof(
                n,
                TextGeneration,
                TestFixture,
                KvGqa,
                Neox,
                ArchPath::TestFixture { rope: Neox },
                WholeVector,
            ));
        }
        v
    })
    .as_slice()
}

/// Resolve a GGUF `general.architecture` value to its profile.
pub fn resolve_profile(arch: &str) -> Option<&'static ArchProfile> {
    architecture_catalog().iter().find(|p| p.gguf_name == arch)
}

/// Resolve a GGUF `general.architecture` value. `None` means the string
/// is not in the registry -- callers must fail closed rather than guess.
pub fn resolve_architecture(arch: &str) -> Option<ArchPath> {
    resolve_profile(arch).map(|p| p.path)
}

/// llama.cpp's hardcoded alternating sliding-window layout for one
/// architecture: the period, *and* which end of each period is the
/// full-attention layer.
///
/// `llama_hparams::set_swa_pattern` (`src/llama-hparams.cpp:8-22`) has
/// two phases, and they are not interchangeable:
///
/// - `dense_first = false`: `is_swa[il] = il % p < (p - 1)` -- the
///   **last** layer of every period is full attention.
/// - `dense_first = true`:  `is_swa[il] = il % p != 0` -- the **first**
///   layer of every period is full attention.
///
/// For a 32-layer period-4 model the two disagree on 16 of the 32
/// layers. Storing only the period would therefore not be a partial
/// transcription, it would be a wrong one for the four architectures
/// llama.cpp passes `dense_first = true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwaPattern {
    /// llama.cpp's `swa_period` seed literal.
    pub period: usize,
    /// llama.cpp's `dense_first` argument to `set_swa_pattern`.
    pub dense_first: bool,
}

/// Every architecture for which llama.cpp seeds a sliding-window period
/// *before* letting `{arch}.attention.sliding_window_pattern` override
/// it, transcribed from `src/models/*.cpp`.
///
/// The period is not in the file for these families -- llama.cpp
/// hardcodes it per architecture and only lets the metadata key override
/// it (`ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN,
/// swa_period, false)` after seeding `swa_period` with the literal
/// below). A missing key therefore does **not** mean "every layer is
/// windowed", which is what ferrox assumed: `layer_sliding_window`
/// returns the window for all layers when `swa_pattern` is `None`, so a
/// gpt-oss or cohere2 checkpoint ran its full-attention layers through a
/// 128-token window and answered from a truncated history.
///
/// Two llama.cpp spellings are deliberately absent, because neither is
/// a per-arch *period*:
///
/// - `set_swa_pattern(0)` (`deepseek4.cpp:68`, `dflash.cpp:54`) makes
///   **every** layer sliding, which is what ferrox already does for a
///   declared window with no pattern.
/// - `set_swa_pattern(1)` (`phi3.cpp:23`) makes **no** layer sliding,
///   and phi3 zeroes `n_swa` and sets `swa_type = NONE` on the same
///   branch, so there is no window left to place.
///
/// Architectures that only ever read a per-layer *array*
/// (`get_key_or_arr(..., hparams.is_swa_impl, n_layer)`: `gemma4`,
/// `gemma4-assistant`, `step35`, `mimo2`, `dflash`) seed no scalar and
/// so have no default to pin.
///
/// Pinned by `tests/swa_pattern.rs`.
/// Architectures where llama.cpp DISABLES sliding-window attention even
/// though the checkpoint declares a window.
///
/// `src/models/phi3.cpp:12-24`: if `attention.sliding_window` is present
/// and non-zero, llama.cpp warns, then sets `n_swa = 0`,
/// `swa_type = LLAMA_SWA_TYPE_NONE` and `set_swa_pattern(1)` -- i.e. NO
/// layer slides. Its own comment says the conversion scripts populate
/// the key wrongly and links the PR that turned it off.
///
/// ferrox read the key and, having no per-architecture period for
/// `phi3`, windowed EVERY layer. So a Phi-3 or Phi-4 model attended over
/// a truncated history on every layer where llama.cpp attends over the
/// whole context. `phi3` is in [`AUDITED_GENERIC_GQA`], and
/// `models/Phi-4-mini-instruct-Q4_K_M.gguf` really does declare
/// `phi3.attention.sliding_window = 262144` -- so this was live on a
/// model in the benchmark suite, not hypothetical.
///
/// This is deliberately a REFUSAL TO HONOUR the key rather than a
/// transcribed period: llama.cpp is not choosing a different window
/// here, it is declining to use the one in the file.
///
/// # The second cause, and why it shares this predicate
///
/// `src/models/exaone4.cpp:4-14` wraps the ENTIRE SWA setup --
/// `swa_type`, `n_swa`, `set_swa_pattern`, both SWA RoPE fields -- in
/// `if (hparams.n_layer() == 64)`, and only then reads
/// `LLM_KV_ATTENTION_SLIDING_WINDOW` at :16 into an `hparams.n_swa` no
/// layer consults. So EXAONE-4 1.2B (30 layers) attends over the whole
/// context on every layer no matter what its file declares, and
/// EXAONE-4 32B (64) does not.
///
/// It is the same QUESTION as `phi3`'s -- "does this file get a window
/// at all" -- so it is the same predicate rather than a second one
/// beside it. `crate::rope_layers::rope_layers` takes this function's
/// answer, not the raw presence of the key, and getting that wrong
/// would rope the 1.2B as if it were the 32B: `exaone4.cpp:116` gates
/// rotation on `is_swa(il)`, so a spurious window would silently stop
/// three layers in four from rotating.
pub fn swa_disabled_by_arch(arch: &str, n_layers: usize) -> bool {
    match arch {
        "phi3" => true,
        // exaone4.cpp:4. NOT `>= 64` and not a range: llama.cpp tests
        // equality, so a hypothetical 63- or 65-layer EXAONE-4 gets no
        // window there either.
        "exaone4" => n_layers != 64,
        _ => false,
    }
}

/// Architectures whose FFN gate uses GELU rather than SiLU, i.e. GeGLU
/// rather than SwiGLU.
///
/// llama.cpp picks this PER ARCHITECTURE -- it is the `LLM_FFN_GELU` vs
/// `LLM_FFN_SILU` argument each `src/models/*.cpp` passes to `build_ffn`
/// / `build_moe_ffn` -- and ferrox picked it per FAMILY, which is not
/// the same partition. `grok` is the case that proves it:
/// `src/models/grok.cpp:165` passes `LLM_FFN_GELU` to `build_moe_ffn`,
/// but `grok` is `DecoderFamily::StandardGqa`, so ferrox handed it
/// SwiGLU and would have computed a different FFN on every layer.
///
/// It was latent while `grok` refused as unaudited, and it is LIVE
/// now: `tests/grok_graphs.rs` compares the GELU experts against
/// libllama, at the GeGLU tolerance that llama.cpp's f16 GELU table
/// forces on every GeGLU row.
///
/// The other `LLM_FFN_GELU` users upstream -- `bert`, `bloom`,
/// `codeshell`, `falcon`, `gpt2`, `gptneox`, `mpt`, `phi2`, `starcoder`,
/// `starcoder2`, `t5`, `wavtokenizer-dec` -- are all `Deferred` or
/// `DedicatedOnly` here, so none reaches the generic path and none is
/// listed. The Gemma lineage is GELU too and stays on the family rule,
/// because every Gemma row IS `GemmaFamily`.
pub fn uses_geglu(arch: &str) -> bool {
    matches!(arch, "grok")
}

/// Architectures whose FFN is the UNGATED ReLU-squared MLP:
/// `build_ffn(up, NULL gate, down, LLM_FFN_RELU_SQR, LLM_FFN_SEQ)`,
/// i.e. `down(relu(up(x))^2)` (`arcee.cpp:123-128`).
///
/// Five graphs pass `LLM_FFN_RELU_SQR` upstream -- measured, by
/// grepping `src/models/*.cpp`: `arcee`, `plm`, `nemotron`, `jais2`,
/// `nemotron-h`. Only `arcee` reaches the generic path with nothing
/// else in the way: `plm` is MLA attention (`UNGATED_RELU_SQR`),
/// `nemotron` and `jais2` are in the biased-LayerNorm group
/// (`WEIGHTED_LAYER_NORM`), `nemotron-h` is a hybrid recurrent model.
/// They are listed so that closing one of them finds its FFN already
/// implemented and named here, and so that `plm`'s verdict can point at
/// the half that is done.
pub fn uses_relu_sqr(arch: &str) -> bool {
    matches!(arch, "arcee" | "plm" | "nemotron" | "jais2" | "nemotron-h")
}

pub fn default_swa_layout(arch: &str) -> Option<SwaPattern> {
    let last_dense = |period| {
        Some(SwaPattern {
            period,
            dense_first: false,
        })
    };
    let dense_first = |period| {
        Some(SwaPattern {
            period,
            dense_first: true,
        })
    };
    match arch {
        // src/models/openai-moe.cpp:9
        "gpt-oss" => last_dense(2),
        // src/models/gemma2.cpp:6
        "gemma2" => last_dense(2),
        // src/models/gemma3.cpp:7
        "gemma3" => last_dense(6),
        // src/models/gemma3n.cpp:4 says 5, NOT 6. This was transcribed
        // as 6 alongside gemma3 and is simply wrong. Inert only because
        // `gemma3n` refuses for other reasons today.
        "gemma3n" => last_dense(5),
        // src/models/gemma-embedding.cpp:5. Deferred (embedding scope),
        // so latent rather than live.
        "gemma-embedding" => last_dense(6),
        // src/models/cohere2.cpp:5, exaone4.cpp:7, olmo2.cpp:9
        "cohere2" | "exaone4" | "olmo2" => last_dense(4),
        // Added after an audit found this table covered 6 architectures
        // where llama.cpp hardcodes a period for 17. A MISSING entry is
        // not neutral: with no period, every layer gets windowed, so a
        // model whose full-attention layers should see the whole context
        // sees only a window instead. That is a different model, and it
        // fails silently.
        //
        // src/models/mellum.cpp:11
        "mellum" => last_dense(4),
        // src/models/exaone-moe.cpp:6. SWA is unconditional there
        // with n_swa = 128, so without this every layer ran with a
        // 128-token history.
        "exaone-moe" => last_dense(4),
        // src/models/afmoe.cpp:17. LIVE: `afmoe` is audited, and its
        // fixture's window is narrower than the prompt
        // (`tests/gated_attention_graphs.rs`).
        "afmoe" => last_dense(4),
        // src/models/plamo3.cpp:9. LIVE: `plamo3` is audited, and its
        // fixture drives a period of 2 from the file with a window
        // narrower than the prompt, so both the period override and
        // this phase are exercised end to end against libllama.
        "plamo3" => last_dense(8),
        // src/models/llama4.cpp:19 ("pattern: 3 chunked - 1 full").
        // `llama4` is `DedicatedOnly` today, so latent.
        "llama4" => last_dense(4),
        // --- dense_first = true -----------------------------------
        //
        // These four put the full-attention layer at `il % p == 0`, not
        // at `il % p == p - 1`. `ModelConfig::layer_sliding_window`
        // implements BOTH phases and carries this flag as
        // `swa_dense_first`; it used to implement only the first, which
        // is why `smallthinker` and `laguna` windowed every layer.
        //
        // src/models/smallthinker.cpp:9-11. Latent: `smallthinker` is
        // triaged NEW CODE on its raw-input router and ReLU experts, so
        // it refuses before this row is consulted. This used to say
        // LIVE, and was wrong: the triage row predates the comment.
        "smallthinker" => dense_first(4),
        // src/models/laguna.cpp:39-41 (its own comment: "XS.2: FULL at
        // il%4==0"). LIVE: `laguna` is on the generic GQA path.
        "laguna" => dense_first(4),
        // src/models/cohere2moe.cpp:31-33. `DedicatedOnly` today
        // (parallel attention+FFN residual), so latent.
        "cohere2moe" => dense_first(4),
        // src/models/modern-bert.cpp:8-10. Deferred (encoder scope), so
        // latent.
        "modern-bert" => dense_first(3),
        _ => None,
    }
}

/// True when this architecture's SWA layers use the model's own RoPE
/// base rather than llama.cpp's `rope_freq_base_train_swa` default of
/// `10000`.
///
/// `llama_hparams` defaults that field to `10000.0f`
/// (`src/llama-hparams.h:127`) and the Gemma-3 lineage relies on the
/// default; the architectures listed here instead open with
/// `hparams.rope_freq_base_train_swa = hparams.rope_freq_base_train;`
/// before letting `rope.freq_base_swa` override it. ferrox applied the
/// Gemma default to everything, which rotates a gpt-oss SWA layer at
/// theta 10000 instead of its real 150000.
pub fn swa_rope_base_follows_model(arch: &str) -> bool {
    matches!(
        arch,
        "afmoe"
            | "cohere2"
            | "cohere2moe"
            | "dflash"
            | "exaone-moe"
            | "exaone4"
            | "gemma2"
            | "laguna"
            | "llama4"
            | "mellum"
            | "olmo2"
            | "gpt-oss"
            | "smallthinker"
    )
}

/// True when this architecture's SWA layers inherit the model's TRAINED
/// RoPE position scale rather than llama.cpp's
/// `rope_freq_scale_train_swa` default of `1.0`.
///
/// The sibling of [`swa_rope_base_follows_model`], and deliberately NOT
/// derived from it: llama.cpp defaults both fields
/// (`src/llama-hparams.h:127,129`) and each architecture assigns them
/// independently, so the two lists differ. `olmo2.cpp:13-14` and
/// `laguna.cpp:47-48` seed the BASE from the model and then pin the
/// SCALE to `1.0` -- laguna's own comment is "SWA uses plain RoPE (no
/// YaRN scaling); do NOT inherit full layers 1/factor". Collapsing the
/// two tables into one would rope those two architectures wrong in
/// exactly the way this function exists to stop.
///
/// The default matters more than the list. `gemma3.cpp:11` reads only
/// `LLM_KV_ROPE_FREQ_BASE_SWA` and never touches
/// `rope_freq_scale_train_swa`, so Gemma-3's sliding layers rope at
/// scale `1.0` while its full-attention layers use the trained scale --
/// and the converter agrees, writing `rope.scaling.factor` from
/// `rope_parameters["full_attention"]` alone (`conversion/base.py:1222`,
/// whose own comment is "TODO: Handle sliding_attention similarly when
/// models start implementing it").
///
/// Every name here is a `hparams.rope_freq_scale_train_swa =
/// hparams.rope_freq_scale_train;` in `src/models/`, at the line given.
pub fn swa_rope_scale_follows_model(arch: &str) -> bool {
    matches!(
        arch,
        "afmoe"          // afmoe.cpp:22
            | "cohere2"     // cohere2.cpp:10
            | "cohere2moe"  // cohere2moe.cpp:39
            | "dflash"      // dflash.cpp:59, :71
            | "exaone-moe"  // exaone-moe.cpp:10
            | "exaone4"     // exaone4.cpp:12
            | "gemma2"      // gemma2.cpp:11
            | "llama4"      // llama4.cpp:24
            | "mellum"      // mellum.cpp:20
            | "gpt-oss"     // openai-moe.cpp:14
            | "smallthinker" // smallthinker.cpp:14
    )
}

/// True when this architecture's graph multiplies every token
/// embedding by `sqrt(n_embd)` as ARITHMETIC, reading no key for it.
///
/// Measured over all 140 `src/models/*.cpp` for
/// `ggml_scale(ctx0, inpL, sqrtf(...n_embd...))`: every Gemma graph
/// (`gemma.cpp:49`, `gemma2.cpp:70`, `gemma3.cpp:93`, `gemma3n.cpp:104`,
/// `gemma4.cpp:155`, `gemma-embedding.cpp:85`) and exactly ONE other,
/// `afmoe.cpp:120` ("MuP scaling"). The Gemma side was a `family`
/// match in `loader.rs`; `afmoe` is not a Gemma and does the same
/// thing, so the fact is a table here rather than a second `if`
/// beside the first.
///
/// This is about the ARITHMETIC, not the key. A file for one of these
/// declaring `{arch}.embedding_scale` describes something its graph
/// does not do, and `scalar_multipliers::multiplier_support` --
/// which lists none of them -- refuses the key before this is asked.
pub fn embeddings_scaled_by_sqrt_n_embd(arch: &str, family: DecoderFamily) -> bool {
    matches!(family, DecoderFamily::GemmaFamily) || arch == "afmoe"
}

/// llama.cpp's `hparams.f_attention_scale`, but only when it DIFFERS
/// from the `1/sqrt(head_dim)` every ferrox attention kernel already
/// applies. `None` means "the kernels' own scale is already right", so
/// a caller stores it straight into `ModelConfig::attention_scale`.
///
/// Only the Gemma-2 and Gemma-3 27B checkpoints answer `Some`:
///
/// ```cpp
/// // src/models/gemma3.cpp:30-33 (src/models/gemma2.cpp:26-29 identical in shape)
/// hparams.f_attention_scale = type == LLM_TYPE_27B
///     ? 1.0f / std::sqrt(float(hparams.n_embd / hparams.n_head(0)))
///     : 1.0f / std::sqrt(float(hparams.n_embd_head_k()));
/// ```
///
/// and llama.cpp applies it as an explicit `ggml_scale` on Q followed by
/// `build_attn(..., 1.0f)` (`gemma3.cpp:154`, `gemma2.cpp:110`), which is
/// what [`crate::config::ModelConfig::attention_scale`] means here.
///
/// **The selector is the LAYER COUNT, not a comparison of the two
/// widths.** `LLM_TYPE_27B` comes from `switch (hparams.n_layer())`
/// (`gemma3.cpp:20-28` `case 62`, `gemma2.cpp:19-23` `case 46`), and
/// deriving it instead from `n_embd / n_head != head_dim` would be
/// wrong for EVERY other Gemma size -- all of them have
/// `n_embd / n_head != head_dim` too, and all of them take llama.cpp's
/// `1/sqrt(n_embd_head_k)` branch. See
/// `gemma_27b_is_the_only_size_that_overrides_the_kernel_scale`.
///
/// `hidden_dim / n_heads` is integer division on purpose: llama.cpp
/// divides two `uint32_t` and only then converts to float.
pub fn attention_scale_override(
    arch: &str,
    n_layers: usize,
    hidden_dim: usize,
    n_heads: usize,
    head_dim: usize,
) -> Option<f32> {
    // `case 62` / `case 46` in the `switch (hparams.n_layer())` that
    // picks `LLM_TYPE_27B`. Every other Gemma architecture
    // (`gemma-embedding`, `gemma3n`, `gemma4`) sets `f_attention_scale`
    // unconditionally and has no 27B branch at all.
    let is_27b = match arch {
        "gemma2" => n_layers == 46,
        "gemma3" => n_layers == 62,
        _ => false,
    };
    if !is_27b || n_heads == 0 || head_dim == 0 {
        return None;
    }
    let scale = 1.0 / ((hidden_dim / n_heads) as f32).sqrt();
    let kernel_scale = 1.0 / (head_dim as f32).sqrt();
    (scale != kernel_scale).then_some(scale)
}

/// Architectures outside the Gemma family whose graph applies the two
/// logit softcaps ferrox implements -- `attn_logit_softcapping` on the
/// attention scores and `final_logit_softcapping` after the lm_head.
///
/// `grok`: `llama-graph.cpp:2572-2582` applies
/// `30 * tanh(kq * f_attn_out_scale / 30)` before the softmax, which is
/// ferrox's `attn_logit_softcap` over a Q pre-scaled by
/// `ModelConfig::attention_scale`; `grok.cpp:214-218` applies the final
/// softcap when the file declares one (default 0, off). The converter
/// (`conversion/grok.py:34`) writes `attn_logit_softcapping` for EVERY
/// Grok export, so without this list no real Grok file could load.
///
/// The Gemma family is not here because it is exempted as a family
/// below; a name here is one whose graph was read for both softcaps.
pub const LOGIT_SOFTCAP_ARCHITECTURES: &[&str] = &["grok"];

/// Metadata keys that, when present with a nonzero value, require math
/// ferrox's generic decoder does not implement *unless* the architecture
/// profile opts into those features (Gemma family), or the architecture
/// is named in [`LOGIT_SOFTCAP_ARCHITECTURES`] for the softcaps.
pub fn unsupported_feature_keys(arch: &str) -> Vec<(String, &'static str)> {
    let profile = resolve_profile(arch);
    // Gemma family implements softcap + SWA pattern; others still refuse.
    if matches!(profile.map(|p| p.family), Some(DecoderFamily::GemmaFamily)) {
        return Vec::new();
    }
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let mut out = Vec::new();
    if !LOGIT_SOFTCAP_ARCHITECTURES.contains(&arch) {
        out.push((
            key("attention.logit_softcapping"),
            "attention logit soft-capping (Gemma 2+); not implemented in the generic decoder",
        ));
        // The spelling llama.cpp's converters ACTUALLY write
        // (`llama-arch.cpp:213` is `%s.attn_logit_softcapping`). The
        // line above is a spelling no converter emits, so this gate has
        // never fired for any non-Gemma architecture -- while
        // `loader.rs` reads BOTH spellings and applies the value.
        //
        // A checkpoint declaring an attention softcap was therefore not
        // refused; it ran with the generic formula. For `grok` that is a
        // wrong answer rather than an approximation: `grok.cpp` folds
        // the real attention scale INTO the softcap and passes
        // `kq_scale = 1.0f`, which the generic path does not do.
        //
        // A gate that cannot fire is not a gate, and it looked exactly
        // like one.
        out.push((
            key("attn_logit_softcapping"),
            "attention logit soft-capping (Gemma 2+); not implemented in the generic decoder",
        ));
        out.push((
            key("final_logit_softcapping"),
            "final logit soft-capping (Gemma 2+); not implemented in the generic decoder",
        ));
    }
    // `{arch}.nextn_predict_layers` WAS refused here, for every
    // architecture, with the reason that ferrox's `n_layers` IS
    // `block_count` and it would run the MTP head as decoder layers.
    // `crate::mtp_blocks::trunk_layers` subtracts the blocks now for
    // exactly the seventeen graphs whose `load_arch_hparams` reads the
    // key (`NEXTN_READERS`, measured) and still refuses a nonzero value
    // on any other -- where llama.cpp itself would run every block and
    // then fail on the unread `nextn.*` tensors. One place decides both
    // halves, so the reader table and the refusal cannot drift apart.
    // `{arch}.attention.sliding_window_pattern` WAS refused here,
    // with the reason "not implemented in the generic decoder".
    // That reason was false, and had been for some time: the
    // alternating pattern lives in `ModelConfig::layer_sliding_window`,
    // which implements BOTH phases and which `gpt-oss` -- a
    // `StandardGqa` row, not a Gemma one -- has relied on since it
    // was audited against libllama.
    //
    // What the gate really did was make the loader's own read of
    // that key (`swa_pattern`) unreachable for every non-Gemma
    // architecture: llama.cpp lets the file override the
    // architecture's hardcoded period, ferrox refused any file that
    // tried. `plamo3` is the case that proves it -- its converter
    // writes the key verbatim (`conversion/plamo.py:178`) -- and
    // `tests/fixture_away_graphs.rs` now drives a period of 2 out of
    // a plamo3 fixture and compares against llama.cpp's own graph on
    // all three forward paths, with the phase and the window
    // sabotaged separately.
    //
    // The real gap the key could hide was NOT the pattern: it was
    // that llama.cpp accepts the value as a scalar OR an n_layer-long
    // ARRAY (`ml.get_key_or_arr`), and ferrox carried one scalar
    // period. The array is `crate::swa_layers` now, read the way each
    // graph reads it -- ignored, honoured, or broadcast -- so neither
    // shape is refused here or anywhere else.
    out
}

/// Scalar multipliers a checkpoint can declare in **metadata** that the
/// generic decoder does not apply, with the value that means "no-op".
///
/// These are the blind spot left by
/// [`crate::loader::assert_every_tensor_consumed`]: that gate catches a
/// missing *tensor*, but Granite / MiniCPM / Command-R style multipliers
/// are hparams, not weights, so a checkpoint carrying them loads
/// cleanly, runs at full speed, and computes a graph scaled differently
/// from the one the checkpoint was trained as. Nothing says so.
///
/// llama.cpp key names (`llama-arch.cpp`):
/// `%s.logit_scale` (`LLM_KV_LOGIT_SCALE`), `%s.residual_scale`,
/// `%s.embedding_scale`, `%s.attention.scale`. Granite reads all four
/// (`src/models/granite.cpp::load_arch_hparams`); MiniCPM and
/// Command-R/Cohere2 read the subset they use.
///
/// **This list is DERIVED, never restated.** Which of the four an
/// architecture applies lives in
/// [`crate::scalar_multipliers::multiplier_support`], and this function
/// is exactly its complement: a key appears here if and only if that
/// table says the graph does not apply it. Two hand-written lists is the
/// shape that once let this repo refuse a key it implemented and
/// implement a key it refused, and the Gemma family used to be exempted
/// from ALL FOUR of these wholesale on the strength of implementing two,
/// so a hand-written `gemma3.residual_scale` would have loaded and been
/// ignored.
///
/// `residual_scale` is the one that reaches furthest: it multiplies the
/// attention and FFN branch outputs before every residual add, so on an
/// architecture that does not implement it a declared value would have
/// to be dropped by every CPU decode/prefill/multi-seq path *and* by the
/// fused Metal kernels that fold the residual in.
///
/// The no-op value differs by key: the three `*_scale` multipliers are
/// `1.0`, while llama.cpp's `f_attention_scale` uses `0.0` as its
/// "unset, use 1/sqrt(head_dim)" sentinel.
pub fn unsupported_scaling_keys(arch: &str) -> Vec<(String, &'static str, f32)> {
    use crate::scalar_multipliers::{AttentionScaleKey, LogitScaleUse};
    let support = crate::scalar_multipliers::multiplier_support(arch);
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let mut out = Vec::new();
    if support.logit == LogitScaleUse::NotApplied {
        out.push((
            key("logit_scale"),
            "logit multiplier (Granite / Command-R `logits_scaling`); not applied by the generic decoder",
            1.0,
        ));
    }
    if !support.residual {
        out.push((
            key("residual_scale"),
            "residual multiplier (Granite `residual_multiplier`); not applied by the generic decoder",
            1.0,
        ));
    }
    if !support.embedding {
        out.push((
            key("embedding_scale"),
            "embedding multiplier (Granite / MiniCPM `embedding_multiplier`); the generic decoder only scales embeddings for the Gemma and Granite families",
            1.0,
        ));
    }
    // Two spellings of one slot, and an architecture reads at most one
    // of them: the OTHER stays refused. `grok` reads `output_scale` and
    // never `scale`; Granite the reverse; everyone else neither.
    if support.attention != AttentionScaleKey::Scale {
        out.push((
            key("attention.scale"),
            "explicit attention score scale (Granite `attention_multiplier`); the generic decoder always uses 1/sqrt(head_dim)",
            0.0,
        ));
    }
    if support.attention != AttentionScaleKey::OutputScale {
        // Applied as-is by the one graph that reads it (`grok.cpp`, no
        // sentinel), so there is no value that means "off" -- the
        // no-op here is the kernels' own scale expressed as a key, which
        // no converter writes for a non-Grok architecture. A file
        // declaring ANY other value is refused.
        out.push((
            key("attention.output_scale"),
            "attention output scale (Grok `attn_output_multiplier`, applied inside its tanh softcap); the generic decoder always uses 1/sqrt(head_dim)",
            0.0,
        ));
    }
    out
}

/// Markdown coverage table for docs / CI drift checks.
pub fn coverage_report_markdown() -> String {
    let mut lines = vec![
        "# Architecture coverage manifest".to_string(),
        String::new(),
        "Generated from `ferrox_models::capability::architecture_catalog`.".to_string(),
        "Source of truth for names: pinned llama.cpp `LLM_ARCH_NAMES`.".to_string(),
        String::new(),
        "| GGUF arch | Scope | Family | Memory | Path |".to_string(),
        "|---|---|---|---|---|".to_string(),
    ];
    for p in architecture_catalog() {
        let path = match p.path {
            ArchPath::GenericGqa { .. } => "generic-gqa",
            ArchPath::TestFixture { .. } => "test-fixture",
            ArchPath::DedicatedOnly { .. } => "dedicated",
            ArchPath::Deferred { .. } => "deferred",
        };
        lines.push(format!(
            "| `{}` | {:?} | {:?} | {:?} | {} |",
            p.gguf_name, p.scope, p.family, p.memory, path
        ));
    }
    lines.push(String::new());
    lines.join("\n")
}

#[cfg(test)]
mod audit_tests {
    use super::*;

    /// Every audited name must actually be on the generic path.
    ///
    /// A name here that resolves to a dedicated engine, or to nothing,
    /// is a stale entry claiming evidence for a path it does not use.
    #[test]
    fn every_audited_name_is_actually_on_the_generic_path() {
        for name in AUDITED_GENERIC_GQA {
            let profile = resolve_profile(name)
                .unwrap_or_else(|| panic!("audited arch `{name}` is not in the catalog"));
            assert!(
                matches!(profile.path, ArchPath::GenericGqa { .. }),
                "`{name}` is listed as an audited GENERIC-path arch but resolves to {:?}",
                profile.path
            );
        }
    }

    /// The five architectures that were caught computing the wrong
    /// thing must never appear here.
    ///
    /// They are refused outright now, but this pins the intent: the
    /// audited list is evidence of correctness, and these are the
    /// counter-examples that motivated it.
    #[test]
    fn the_architectures_that_were_wrong_are_not_claimed_as_audited() {
        for name in ["gpt2", "mpt", "refact", "bloom", "jais"] {
            assert!(
                !is_audited_generic(name),
                "`{name}` was found computing ALiBi or learned position embeddings as \
                 though it were RoPE; it cannot be on the audited list"
            );
        }
    }

    /// Every unaudited generic-path architecture either carries a
    /// triage verdict or is named on [`TRIAGE_PENDING`] -- never both,
    /// never neither.
    ///
    /// This is the anti-drift gate. Adding a new architecture to the
    /// generic catalog without either reading it against llama.cpp or
    /// admitting on the pending list that nobody has, fails here.
    #[test]
    fn every_unaudited_generic_architecture_is_triaged_or_listed_as_pending() {
        for p in architecture_catalog() {
            if !matches!(p.path, ArchPath::GenericGqa { .. }) || is_audited_generic(p.gguf_name) {
                continue;
            }
            let pending = TRIAGE_PENDING.contains(&p.gguf_name);
            match (p.triage, pending) {
                (Some(_), false) | (None, true) => {}
                (Some(t), true) => panic!(
                    "`{}` carries a {:?} verdict AND is still on TRIAGE_PENDING; remove it \
                     from the pending list",
                    p.gguf_name, t.class
                ),
                (None, false) => panic!(
                    "`{}` is on the generic path, is not audited, has no triage verdict and \
                     is not on TRIAGE_PENDING. Read \
                     .scratch/llama.cpp/src/models/ for it, or say so on the pending list",
                    p.gguf_name
                ),
            }
        }
    }

    /// A name on [`TRIAGE_PENDING`] that is not an unaudited generic row
    /// is a stale to-do: it would keep claiming work that no longer
    /// exists, or point at an architecture the loader never asks about.
    #[test]
    fn nothing_on_the_pending_list_is_stale() {
        for name in TRIAGE_PENDING {
            let p = resolve_profile(name)
                .unwrap_or_else(|| panic!("TRIAGE_PENDING names `{name}`, not in the catalog"));
            assert!(
                matches!(p.path, ArchPath::GenericGqa { .. }),
                "`{name}` is on TRIAGE_PENDING but resolves to {:?}, which never reaches the \
                 unaudited refusal",
                p.path
            );
            assert!(
                !is_audited_generic(name),
                "`{name}` is audited and runs; it does not need a triage verdict"
            );
        }
        // The list is empty because the triage finished, not because it
        // was never populated. If a future architecture lands on the
        // generic path with no verdict, it belongs here and
        // `every_unaudited_generic_architecture_is_triaged_or_listed_as_pending`
        // will say so; until then, empty is the completed state.
        assert!(
            TRIAGE_PENDING.is_empty(),
            "TRIAGE_PENDING regrew to {:?}; that is fine, but say so in docs/MODELS.md too",
            TRIAGE_PENDING
        );
    }

    /// An audited architecture runs. A triage verdict on one would be a
    /// refusal class attached to something that never refuses.
    #[test]
    fn an_audited_architecture_carries_no_triage_verdict() {
        for name in AUDITED_GENERIC_GQA {
            assert!(
                unaudited_triage(name).is_none(),
                "`{name}` is audited and runs, so it must not carry a triage verdict"
            );
        }
    }

    /// A verdict has to say something. An empty blocker, or one that
    /// cites no llama.cpp source line, is the failure mode this whole
    /// item exists to prevent: a refusal that names a blocker nobody
    /// checked.
    #[test]
    fn every_triage_verdict_cites_the_llama_cpp_line_that_decides_it() {
        let mut seen = 0;
        for p in architecture_catalog() {
            let Some(t) = p.triage else { continue };
            seen += 1;
            assert!(
                t.blocker.len() > 80,
                "`{}`'s blocker is too short to name anything: {:?}",
                p.gguf_name,
                t.blocker
            );
            let cites_llama_cpp =
                t.blocker.contains("src/models/") || t.blocker.contains("src/llama-arch.cpp");
            assert!(
                cites_llama_cpp,
                "`{}`'s blocker cites no llama.cpp source: {}",
                p.gguf_name, t.blocker
            );
            if t.class == TriageClass::Unknown {
                assert!(
                    t.blocker.contains("WOULD SETTLE IT"),
                    "`{}` is UNKNOWN but does not say what would settle it",
                    p.gguf_name
                );
            }
        }
        assert!(
            seen == 9,
            "every unaudited generic architecture is triaged; found {seen}. \
             It was 47 until the triage found `minicpm3` was an MLA model on the \
             generic-GQA row and it moved to DedicatedOnly, 46 until five ONE MATCH ARM \
             rows -- deepseek, bailingmoe, seed_oss, maincoder, hunyuan-moe -- were admitted \
             with libllama-golden fixtures, 41 until seven FIXTURE-AWAY rows -- \
             internlm2, xverse, ernie4_5, baichuan, exaone, bailingmoe2, plamo3 -- got \
             theirs (tests/fixture_away_graphs.rs), 34 until `gemma`, `hunyuan-dense` \
             and `ernie4_5-moe` got theirs, 31 until `olmo2` and `exaone4` -- the \
             POST-NORM-ONLY pair, ONE topology and one implementation \
             (`crate::norm`) -- got theirs (tests/post_norm_only_graphs.rs), 29 until \
             `chatglm` -- the LAST ONE MATCH ARM row -- got its fused-QKV-bias arm and \
             its fixture, 28 until `mistral`, `mixtral` and `yi` turned out not to be \
             architectures at all (libllama refuses all three strings) and moved to \
             DedicatedOnly, and 25 until the three Granite rows -- granite, granitemoe \
             and the granite-moe alias -- closed together on ONE implementation of their \
             four scalar multipliers (tests/granite_family_graphs.rs), and 22 until \
             `olmo` closed on the non-parametric LayerNorm (`crate::norm`, \
             tests/olmo_graphs.rs). `olmo` is the FIRST NEW CODE row to close on its own, \
             and it says something the other closures do not: its cause is not shared. \
             Every `build_norm` call in llama.cpp's 140 graphs was scanned for a null \
             weight and all three hits are `olmo.cpp`, so this variant was never going to \
             take a second row with it -- see `NON_PARAMETRIC_LAYER_NORM`. `gemma` was the \
             last fixture-away row and `chatglm` the last one-match-arm row, so BOTH \
             classes are empty, and 21 until `exaone-moe` closed on the per-layer RoPE \
             gate (`crate::rope_layers`, tests/no_rope_layer_graphs.rs) -- which is ONE \
             cause behind three refusals, and the count moved by one only because the \
             other two were not in it: EXAONE-4 32B was refused BY NAME in loader.rs \
             and `smollm3` sat in the \"No RoPE at all\" DedicatedOnly group, so both \
             raise the audited number without lowering this one, and 20 until `grok` \
             and `dbrx` closed together on seams that had landed the day before -- the \
             defaults hook and the norm-site table for `grok`, the LayerNorm variant, \
             the QKV clamp and the same table for `dbrx` (tests/grok_graphs.rs, \
             tests/dbrx_graphs.rs) -- with the clamp also closing `olmo`'s clip_qkv \
             refusal by name, and 18 until `arcee` closed on the ungated ReLU-squared FFN \
             (`FfnActivation::ReluSqr`, tests/ungated_ffn_graphs.rs) -- ALONE, because the \
             constant it shared with `plm` had named the FFN and missed `plm`'s MLA \
             attention -- and `deci` and `openelm` closed together on the per-layer shape \
             seam (`crate::layer_shapes`, tests/per_layer_shape_graphs.rs), which the scan \
             that sized it says reaches `laguna`, `mimo2` and `step35` too, each of which \
             still needed something else, and 15 until `afmoe` and `laguna` closed together \
             on the gated attention (`crate::attn_gate`, tests/gated_attention_graphs.rs) \
             -- one op with two free parameters behind three verdicts, read side by side \
             before being called one cause; `step35` keeps its clamp arrays and window \
             array and says the gate is done, and `mimo2`'s sinks moved off the gpt-oss \
             name onto the tensor without closing it, and 13 until `mellum` closed on the \
             per-layer sliding-window ARRAY (`crate::swa_layers`, \
             tests/window_array_graphs.rs) -- the seam three verdicts named, and `mellum` \
             is the one generic-path graph that HONOURS the array; the same seam lifted the \
             over-refusal of every real EXAONE-4 32B / EXAONE-MoE / Olmo-3 export, whose \
             array llama.cpp IGNORES (measured: libllama's logits do not move when it is \
             inverted), and `crate::mtp_blocks` landed beside it and skips the NextN \
             blocks `mimo2` and `step35` named, so both lead with what is left, and 12 \
             until `apertus` and `step35` closed together on the per-layer ACTIVATION \
             PARAMETER seam (`crate::act_layers`, tests/per_layer_activation_graphs.rs, \
             tests/clamped_swiglu_graphs.rs) -- one plumbing question, `layer il runs its \
             FFN activation with these scalars`, and two bodies, xIELU and the clamped \
             SwiGLU, read side by side before being called one cause; `step35`'s \
             half-width rotary landed on `crate::swa_geometry` as a two-valued width and \
             lifted Laguna-XS.2's `rope.dimension_count_swa` refusal by name with it, and \
             10 until `mistral3` closed on the per-position attention temperature \
             (`crate::attn_temperature`, tests/attn_temperature_graphs.rs) -- the reach \
             measured first: three graphs of 140 build the input, `llama4` from literals \
             on its own engine and `deepseek2` / `mistral4` on the MLA engine, which \
             REFUSES the key by name now where it dropped it; and its `yarn_log_multiplier` \
             half found YaRN's magnitude term missing for EVERY architecture \
             (`crate::yarn_magnitude`). \
             What is left is 8 NEW CODE and one UNKNOWN (`phi4`). The NEW CODE rows that have \
             closed are `olmo2`, `exaone4`, the three Granite rows, `exaone-moe`, `grok`, \
             `dbrx`, `arcee`, `deci`, `openelm`, `afmoe`, `laguna`, `mellum`, `apertus`, \
             `step35` and `mistral3`, and each closure but `olmo`'s, `arcee`'s, `mellum`'s \
             and `mistral3`'s took more than one row at a time because each found ONE cause \
             behind several refusals; `mellum`'s cause IS shared and moved three verdicts, \
             but only one of them was closable by it, and `mistral3`'s is shared with two \
             rows on other engines"
        );
    }

    /// The class reaches the message. Two architectures in different
    /// classes must not read the same, which is the defect being fixed.
    #[test]
    fn the_refusal_detail_distinguishes_the_classes() {
        // TWO of the four classes have no rows left. `gemma` was the
        // last FIXTURE-AWAY row and `chatglm` the last ONE MATCH ARM
        // one, and both are audited now, so neither renders a detail at
        // all -- `every_triage_verdict_cites_the_llama_cpp_line...`
        // pins the count that says so. The two live classes are sampled
        // from the catalog; the two empty ones are sampled from
        // `headline()` below, because a class with no rows still has to
        // render distinctly the day something lands in it again.
        //
        // `smallthinker`, which used to be `dbrx`, which used to be
        // `olmo`: the sample keeps moving because the rows keep
        // closing. `olmo`'s non-parametric LayerNorm, then `dbrx`'s
        // weighted one plus its clamp and its `attn_output_norm` slot,
        // are all implemented now. `smallthinker` routes on the raw
        // layer input and runs ReLU experts, and nothing landed so far
        // reaches either.
        let new_code = unaudited_refusal_detail("smallthinker");
        // `phi4` is the only UNKNOWN row left: `mistral`, `mixtral` and
        // `yi` used to be the other three and are refused as strings
        // now (see `NO_UPSTREAM_ARCH`).
        let unknown = unaudited_refusal_detail("phi4");
        // TRIAGE_PENDING is empty now that all 47 are read, so the
        // untriaged branch is exercised through a name the catalog does
        // not carry. The branch has to keep working: it is what a NEW
        // architecture added to the catalog would render until somebody
        // reads it.
        let untriaged = unaudited_refusal_detail("an-arch-nobody-has-read");
        assert!(new_code.contains("NEW CODE"), "{new_code}");
        assert!(unknown.contains("UNKNOWN"), "{unknown}");
        assert!(
            untriaged.contains("not done for `an-arch-nobody-has-read` yet"),
            "{untriaged}"
        );
        for a in [&new_code, &unknown, &untriaged] {
            for b in [&new_code, &unknown, &untriaged] {
                if !std::ptr::eq(a, b) {
                    assert_ne!(a, b, "two refusal details are identical");
                }
            }
        }
        // The blocker itself, not only the class label, has to be in the
        // message -- a class with no specifics is the old refusal with a
        // new adjective.
        assert!(new_code.contains("smallthinker.cpp:111"), "{new_code}");
        assert!(unknown.contains("LLM_ARCH_NAMES"), "{unknown}");
        // The two empty classes still have to be distinguishable.
        let labels = [
            TriageClass::FixtureAway,
            TriageClass::OneMatchArm,
            TriageClass::NewCode,
            TriageClass::Unknown,
        ];
        for (i, a) in labels.iter().enumerate() {
            for b in &labels[i + 1..] {
                assert_ne!(a.label(), b.label());
                assert_ne!(a.headline(), b.headline());
            }
        }
    }

    /// An architecture nobody has checked is not audited, which is the
    /// whole point of the inversion.
    #[test]
    fn an_unchecked_architecture_is_not_audited() {
        assert!(!is_audited_generic("smallthinker"));
        assert!(!is_audited_generic("talkie"));
        assert!(!is_audited_generic("an-arch-that-does-not-exist"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_mainstream_families_resolve() {
        assert_eq!(
            resolve_architecture("llama"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Norm
            })
        );
        assert_eq!(
            resolve_architecture("qwen2moe"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        );
        // `mistral`, `mixtral` and `yi` are NOT here any more. They are
        // resolved, but refused: no converter writes those strings and
        // libllama refuses them outright, so they are alias rows that
        // exist to say "your file is spelled `llama`", not families
        // that load. Pinned by
        // `the_alias_rows_are_refused_as_strings_no_converter_writes`.
        for alias in ["mistral", "mixtral", "yi"] {
            assert!(
                matches!(
                    resolve_architecture(alias),
                    Some(ArchPath::DedicatedOnly { .. })
                ),
                "`{alias}` must be refused, not routed to the generic decoder"
            );
        }
        assert_eq!(
            resolve_architecture("phi3"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        );
        assert_eq!(
            resolve_architecture("phi4"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        );
        assert_eq!(
            resolve_profile("phi4").map(|p| p.family),
            Some(DecoderFamily::PhiFamily)
        );
        assert_eq!(
            resolve_architecture("gemma3"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Neox
            })
        );
        for arch in ["gemma4", "gemma4-assistant"] {
            assert!(
                matches!(
                    resolve_architecture(arch),
                    Some(ArchPath::DedicatedOnly { .. })
                ),
                "{arch} uses dedicated Gemma4 engine"
            );
            assert_eq!(
                resolve_profile(arch).map(|p| p.family),
                Some(DecoderFamily::GemmaFamily)
            );
        }
        assert!(matches!(
            resolve_architecture("gemma3n"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
        assert_eq!(
            resolve_architecture("deepseek"),
            Some(ArchPath::GenericGqa {
                rope: RopeLayout::Norm
            })
        );
        assert_eq!(
            resolve_profile("qwen3").map(|p| p.qk_norm),
            Some(QkNormStyle::PerHead)
        );
    }

    #[test]
    fn deepseek2_is_dedicated_mla_not_generic() {
        assert!(matches!(
            resolve_architecture("deepseek2"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
    }

    #[test]
    fn unknown_architecture_is_none() {
        assert_eq!(resolve_architecture("totally-unknown-arch"), None);
        // t5 is registered as dedicated encoder-decoder stub
        assert!(matches!(
            resolve_architecture("t5"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
    }

    #[test]
    fn dedicated_paths_are_not_generic() {
        assert!(matches!(
            resolve_architecture("glm-dsa"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
        assert!(matches!(
            resolve_architecture("deepseek4"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
        for arch in ["minimax-m2", "minimax-m3"] {
            assert!(
                matches!(
                    resolve_architecture(arch),
                    Some(ArchPath::DedicatedOnly { .. })
                ),
                "{arch} must fail closed, not silent generic GQA"
            );
        }
        match resolve_architecture("llama4") {
            Some(ArchPath::DedicatedOnly { reason }) => {
                assert!(
                    reason.contains("llama4_engine.rs") && reason.contains("attn_temperature"),
                    "llama4's reason names its engine and the temperature seam: {reason}"
                );
            }
            other => panic!("llama4 must fail closed, not silent generic GQA: {other:?}"),
        }
        assert!(matches!(
            resolve_architecture("glm4"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
        assert!(matches!(
            resolve_architecture("glm4moe"),
            Some(ArchPath::DedicatedOnly { .. })
        ));
    }

    #[test]
    fn test_fixtures_remain_loadable() {
        for arch in ["ferroxtest", "ferroxtestmoe", "ferroxtestmixed"] {
            assert!(matches!(
                resolve_architecture(arch),
                Some(ArchPath::TestFixture { .. })
            ));
        }
    }

    #[test]
    fn catalog_has_unique_names() {
        let mut seen = std::collections::HashSet::new();
        for p in architecture_catalog() {
            assert!(
                seen.insert(p.gguf_name),
                "duplicate arch name {}",
                p.gguf_name
            );
        }
    }

    #[test]
    fn gemma_family_does_not_fail_closed_on_softcap_keys() {
        assert!(unsupported_feature_keys("gemma3").is_empty());
        assert!(!unsupported_feature_keys("llama").is_empty());
    }

    /// `grok` applies both softcaps, so neither key refuses it -- while
    /// every OTHER gate in that list still does, and every name on the
    /// softcap list is an audited row.
    ///
    /// The first half is what lets a real Grok file load at all:
    /// `conversion/grok.py:34` writes `attn_logit_softcapping` for every
    /// export. The second half is what keeps the exemption from
    /// widening into "softcaps are fine everywhere": `llama` must still
    /// refuse them, and the NextN gate must still reach `grok`.
    #[test]
    fn grok_is_exempt_from_the_softcap_keys_and_nothing_else() {
        let keys: Vec<String> = unsupported_feature_keys("grok")
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        for softcap in [
            "grok.attention.logit_softcapping",
            "grok.attn_logit_softcapping",
            "grok.final_logit_softcapping",
        ] {
            assert!(
                !keys.iter().any(|k| k == softcap),
                "{softcap} must not refuse grok"
            );
        }
        // `nextn_predict_layers` used to be the "non-softcap gate still
        // applies" witness here. It is `crate::mtp_blocks` now, keyed by
        // which graphs read it, and `grok` is not one: its refusal
        // there is `a_non_reader_with_a_nonzero_count_is_refused_and_zero_is_not`.
        assert!(
            !keys.iter().any(|k| k.ends_with("nextn_predict_layers")),
            "nextn_predict_layers is decided by mtp_blocks::trunk_layers, not here: {keys:?}"
        );
        let llama: Vec<String> = unsupported_feature_keys("llama")
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert!(llama.iter().any(|k| k == "llama.attn_logit_softcapping"));
        for arch in LOGIT_SOFTCAP_ARCHITECTURES {
            assert!(
                is_audited_generic(arch),
                "`{arch}` is on LOGIT_SOFTCAP_ARCHITECTURES without a fixture proving both \
                 softcaps"
            );
        }
    }

    /// The derived scaling refusals for `grok`: the two keys its graph
    /// does not read stay refused, the three it reads do not, and the
    /// OTHER attention spelling is refused for Granite.
    ///
    /// This is the half of `AttentionScaleKey` that a hand-written list
    /// could have got wrong silently: `attention.output_scale` had no
    /// refusal at all before `grok`, so a Granite file declaring it
    /// would have loaded and been ignored.
    #[test]
    fn the_scaling_refusals_for_grok_are_derived_from_its_attention_key() {
        let refused = |arch: &str| -> Vec<String> {
            unsupported_scaling_keys(arch)
                .into_iter()
                .map(|(k, _, _)| k)
                .collect()
        };
        let grok = refused("grok");
        assert_eq!(
            grok,
            vec![
                "grok.residual_scale".to_string(),
                "grok.attention.scale".to_string()
            ],
            "{grok:?}"
        );
        let granite = refused("granite");
        assert_eq!(granite, vec!["granite.attention.output_scale".to_string()]);
        let llama = refused("llama");
        assert!(llama.contains(&"llama.attention.output_scale".to_string()));
        assert!(llama.contains(&"llama.attention.scale".to_string()));
        assert_eq!(llama.len(), 5, "{llama:?}");
    }

    /// Parallel attention+FFN residual is not a tensor and not a
    /// metadata key either, so neither the tensor-consumption gate nor
    /// `unsupported_scaling_keys` can see the difference: these
    /// architectures must not be admitted to the generic decoder at all.
    ///
    /// `minicpm` used to be on this list and is NOT a residual-topology
    /// row -- it runs Granite's graph verbatim
    /// (`models.h:1594-1601`). It was here because its three hardcoded
    /// multipliers are invisible to a key-presence gate the same way a
    /// parallel residual is, which made the list's name wrong about one
    /// of its own members. `scalar_multipliers::MultiplierDefaults`
    /// applies them now and `tests/minicpm_graphs.rs` is the evidence.
    #[test]
    fn architectures_with_a_different_residual_topology_are_refused() {
        for arch in [
            "command-r",
            "cohere2",
            "cohere2moe",
            "falcon",
            "gptneox",
            "phi2",
            "plamo",
        ] {
            match resolve_architecture(arch) {
                Some(ArchPath::DedicatedOnly { reason }) => {
                    assert!(!reason.is_empty(), "{arch} must say why");
                }
                other => panic!("{arch} must be refused, got {other:?}"),
            }
        }
        // The sequential-residual siblings stay on the generic path --
        // this is a named list, not a family-wide ban.
        //
        // `phimoe`, `starcoder2` and `nemotron` used to be checked here
        // too. They left the generic path for an unrelated reason (the
        // required bias tensors pinned by `tests/attn_bias.rs`), so
        // asserting them generic would now assert the wrong thing; what
        // still has to hold is that neither they nor the archs below are
        // refused for a *residual* reason they do not have.
        for arch in ["phi3", "plamo3", "qwen2", "llama"] {
            assert!(
                matches!(
                    resolve_architecture(arch),
                    Some(ArchPath::GenericGqa { .. })
                ),
                "{arch} must stay generic"
            );
        }
        for arch in ["phimoe", "starcoder2", "nemotron"] {
            match resolve_architecture(arch) {
                Some(ArchPath::DedicatedOnly { reason }) => assert!(
                    reason.contains("bias"),
                    "{arch} is refused for the wrong reason: {reason}"
                ),
                other => panic!("{arch} must be refused for its biases, got {other:?}"),
            }
        }
    }

    /// Every architecture appears exactly once, so a refusal added next
    /// to an existing entry cannot be shadowed by whichever the lookup
    /// happens to find first.
    #[test]
    fn no_architecture_is_listed_twice() {
        let mut seen = std::collections::HashSet::new();
        for p in architecture_catalog() {
            assert!(seen.insert(p.gguf_name), "{} listed twice", p.gguf_name);
        }
    }

    /// Every key this gate refuses must be a key a converter actually
    /// writes, or the gate cannot fire.
    ///
    /// `unsupported_feature_keys` listed `{arch}.attention.logit_softcapping`.
    /// llama.cpp writes `{arch}.attn_logit_softcapping`
    /// (`llama-arch.cpp:213`), and no converter emits the first
    /// spelling -- so that arm never matched anything, for any non-Gemma
    /// architecture, ever. Meanwhile `loader.rs` reads BOTH spellings,
    /// so the value was read and applied with the generic formula
    /// instead of being refused. For `grok` that is a wrong answer:
    /// `grok.cpp` folds the real attention scale into the softcap and
    /// passes `kq_scale = 1.0f`.
    ///
    /// A gate that cannot fire is worse than a missing gate, because it
    /// reads as coverage.
    #[test]
    fn every_refused_key_is_one_a_converter_actually_writes() {
        let keys: Vec<String> = unsupported_feature_keys("llama")
            .into_iter()
            .map(|(k, _)| k)
            .collect();

        // Transcribed from `llama-arch.cpp`'s LLM_KV_NAMES.
        // `llama.attention.sliding_window_pattern` was on this list and
        // is deliberately off it: the alternating pattern IS
        // implemented (`ModelConfig::layer_sliding_window`, both
        // phases), so refusing it was a gate with a false reason that
        // also made the loader's own read of the key unreachable. See
        // the comment where it used to be. The array-valued case is
        // `crate::swa_layers` (`tests/window_array_graphs.rs`).
        for real in [
            "llama.attn_logit_softcapping",
            "llama.final_logit_softcapping",
        ] {
            assert!(
                keys.iter().any(|k| k == real),
                "{real} is a key llama.cpp writes and this gate must refuse it; \
                 gate currently holds {keys:?}"
            );
        }

        // Gemma implements all three, so it must still be exempt --
        // otherwise "fix the spelling" would have turned into "refuse
        // every Gemma checkpoint".
        assert!(
            unsupported_feature_keys("gemma2").is_empty(),
            "the Gemma family implements softcap and the SWA pattern"
        );
        // And the pattern key must not come back: a file carrying it
        // gets its period READ, which is what llama.cpp does.
        assert!(
            !keys.iter().any(|k| k.ends_with("sliding_window_pattern")),
            "the SWA pattern is implemented; refusing it makes the loader's read of the \
             key dead code: {keys:?}"
        );
    }

    /// llama.cpp picks Gemma's `f_attention_scale` on the LAYER COUNT
    /// (`gemma3.cpp:20-33`, `gemma2.cpp:19-29`), and every published
    /// Gemma size -- not just 27B -- has `n_embd / n_head != head_dim`.
    /// An override derived from "the two widths disagree" would fire on
    /// all eight rows below and mis-scale six of them, which is why this
    /// walks the real sizes rather than asserting the 27B number alone.
    ///
    /// Shipped broken: `loader.rs` hardcoded `attention_scale = None`
    /// beside a comment naming the 27B exception, so Gemma-2-27B scored
    /// `sqrt(144/128)` and Gemma-3-27B `sqrt(168/128)` too large on
    /// every layer -- a sharper softmax than the trained one, with no
    /// error.
    #[test]
    fn gemma_27b_is_the_only_size_that_overrides_the_kernel_scale() {
        /// One published Gemma size, as its GGUF header declares it.
        struct Size {
            arch: &'static str,
            n_layers: usize,
            n_embd: usize,
            n_head: usize,
            /// `attention.key_length`, llama.cpp's `n_embd_head_k()`.
            head_dim: usize,
            /// The denominator llama.cpp's 27B branch produces, or
            /// `None` where it takes the `1/sqrt(n_embd_head_k)` branch.
            want_denom: Option<f32>,
        }
        let size = |arch, n_layers, n_embd, n_head, head_dim, want_denom| Size {
            arch,
            n_layers,
            n_embd,
            n_head,
            head_dim,
            want_denom,
        };
        let sizes = [
            size("gemma2", 26, 2304, 8, 256, None),         // Gemma-2-2B
            size("gemma2", 42, 3584, 16, 256, None),        // Gemma-2-9B
            size("gemma2", 46, 4608, 32, 128, Some(144.0)), // Gemma-2-27B
            size("gemma3", 18, 640, 4, 256, None),          // Gemma-3-270M
            size("gemma3", 26, 1152, 4, 256, None),         // Gemma-3-1B
            size("gemma3", 34, 2560, 8, 256, None),         // Gemma-3-4B
            size("gemma3", 48, 3840, 16, 256, None),        // Gemma-3-12B
            size("gemma3", 62, 5376, 32, 128, Some(168.0)), // Gemma-3-27B
        ];
        for &Size {
            arch,
            n_layers,
            n_embd,
            n_head,
            head_dim,
            want_denom,
        } in &sizes
        {
            // The premise of the whole test: no Gemma size has
            // `n_embd / n_head == head_dim`, so "the widths disagree"
            // cannot be the selector.
            assert_ne!(
                n_embd / n_head,
                head_dim,
                "{arch}/{n_layers}L: if this ever holds, re-read the derivation"
            );
            let got = attention_scale_override(arch, n_layers, n_embd, n_head, head_dim);
            match want_denom {
                None => assert_eq!(
                    got, None,
                    "{arch}/{n_layers}L takes llama.cpp's 1/sqrt(n_embd_head_k) branch, \
                     which the attention kernels already apply"
                ),
                Some(denom) => {
                    let want = 1.0 / denom.sqrt();
                    let got = got.unwrap_or_else(|| {
                        panic!("{arch}/{n_layers}L is llama.cpp's LLM_TYPE_27B; scale must be set")
                    });
                    assert!(
                        (got - want).abs() < 1e-7,
                        "{arch}/{n_layers}L: want 1/sqrt({denom}) = {want}, got {got}"
                    );
                    // The direction of the correction: the kernels' own
                    // scale is the LARGER one, so the override shrinks
                    // the scores rather than growing them.
                    let kernel = 1.0f32 / (head_dim as f32).sqrt();
                    assert!(
                        kernel > got,
                        "{arch}/{n_layers}L: kernel scale {kernel} must exceed {got}"
                    );
                }
            }
        }
        // `gemma-embedding`, `gemma3n` and `gemma4` set
        // `f_attention_scale` unconditionally in llama.cpp and have no
        // `LLM_TYPE_27B` branch; nothing outside gemma2/gemma3 reaches
        // this at all.
        for arch in ["gemma-embedding", "gemma3n", "gemma4", "llama", "qwen3"] {
            assert_eq!(
                attention_scale_override(arch, 62, 5376, 32, 128),
                None,
                "{arch} has no LLM_TYPE_27B branch in llama.cpp"
            );
        }
    }
}
