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
    // architectures and ONE implementation (`crate::pre_norm::PreNorm`).
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
    // EXAONE-4 32B is `block_count == 64`, where llama.cpp turns SWA on
    // off the layer count and then gives the full-attention layers no
    // RoPE at all (:4-9, :116); that is refused by name in loader.rs,
    // the `baichuan` precedent. NOT the audited `exaone` row, which is
    // EXAONE 3.x and a plain pre-norm llama.
    "exaone4",
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
/// `crate::pre_norm::PreNorm` is the one implementation they share.
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
    ("granite", TriageClass::NewCode, GRANITE_MULTIPLIERS),
    ("granitemoe", TriageClass::NewCode, GRANITE_MULTIPLIERS),
    // ferrox-only alias row; no llama.cpp GGUF spells it this way, but
    // it must not carry a different verdict from `granitemoe`.
    ("granite-moe", TriageClass::NewCode, GRANITE_MULTIPLIERS),
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
    (
        "deci",
        TriageClass::NewCode,
        "DeciLM / Llama-3.1-Nemotron layers are not all the same shape. \
         src/models/deci.cpp:30-34 reads n_head(i), n_head_kv(i) and n_ff(i) PER LAYER, and \
         the graph branches on them three ways: `n_head == 0` is an attention-free layer \
         that passes the residual straight through (:107-109), `n_head_kv == 0` is a \
         \"linear attention\" layer that applies only `wo` with no Q/K/V and no RoPE \
         (:115-118), and `n_ff == 0` skips the FFN and the residual add entirely with a \
         `continue` (:147-149). ferrox's ModelConfig carries n_heads, n_kv_heads and \
         expert_ffn_dim as SCALARS and its decoder runs the same block on every layer, so \
         there is nowhere to put any of the three. Same class as `openelm`, one step worse",
    ),
    (
        "olmo",
        TriageClass::NewCode,
        "OLMo-1 has NO norm weights at all. src/models/olmo.cpp:27-35 creates Q/K/V, \
         attn_output and gate/up/down and not one norm tensor, and the graph calls \
         `build_norm(x, NULL, NULL, LLM_NORM, il)` at all three sites (:65-67, :104-106, \
         :128-130) -- non-parametric LayerNorm: subtract the mean, divide by the standard \
         deviation, no learned weight and no bias. ferrox has only `rms_norm(x, w, eps)` \
         and requires `blk.N.attn_norm.weight`, so it is both a different function and a \
         missing tensor. It also reads an optional {arch}.attention.clamp_kqv (:5) that \
         nothing here applies. Note this is OLMo-1, and it is NOT the post-norm-only \
         topology `olmo2` and `exaone4` share: OLMo-1 norms BEFORE both sublayers \
         (:65-67 before attention, :104-106 before the FFN), so it is pre-norm like \
         llama and the difference is the norm FUNCTION, not the residual shape. Three \
         shapes, not two, and `crate::pre_norm` is no help here",
    ),
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
    (
        "mistral3",
        TriageClass::NewCode,
        "per-position attention temperature tuning. src/models/mistral3.cpp:5,14-17 reads \
         {arch}.attention.temperature_scale and seeds n_attn_temp_floor_scale from \
         n_ctx_orig_yarn, and :109-111 builds a per-position Q scale that llama-graph.cpp \
         computes as `log(floor(pos / floor_scale) + 1) * temp_scale + 1` (:159-167). ferrox \
         has no per-position attention scale at all and no gate on that key, so a checkpoint \
         carrying it would load and silently drop it -- the class of defect \
         `unsupported_scaling_keys` exists for, on a key that list does not have. :9 also \
         reads rope.scaling.yarn_log_multiplier, and loader.rs:588's own comment records \
         that ferrox implements only YaRN's magnitude term. The rest (:46-83, :120-210) is \
         leading-dense + MoE + shared expert on a sequential residual, which ferrox has",
    ),
    (
        "nanbeige",
        TriageClass::NewCode,
        "nanbeige RUNS THE SAME PHYSICAL LAYERS MORE THAN ONCE. \
         src/models/nanbeige.cpp:13-31 sets `n_layer_all = n_layer_phys * n_loops` and \
         rewrites the per-layer head/ff/swa arrays so the graph walks n_layer_all steps over \
         n_layer_phys sets of weights, and :167 applies `output_norm` to the running \
         residual inside the loop at the end of each pass. ferrox's decoder walks its layer \
         vector exactly once and has no concept of a loop count. Everything inside one pass \
         (:52-63, :106-155) is plain llama, which is what makes this deceptive: the tensor \
         set alone looks generic",
    ),
    ("arcee", TriageClass::NewCode, UNGATED_RELU_SQR),
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

/// Shared by `arcee` and `plm`: an ungated ReLU-squared MLP.
///
/// Found by the activation audit the `deepseek` renormalisation bug
/// prompted, not by reading these two files on purpose. Both were on the
/// generic path with nothing recording that their FFN is neither SwiGLU
/// nor GeGLU.
const UNGATED_RELU_SQR: &str =
    "an UNGATED ReLU-squared MLP, which is a different FFN shape and not only a different \
     activation. src/models/arcee.cpp:39-40 and plm.cpp:39-40 create only `ffn_up` and \
     `ffn_down` and no `ffn_gate` at all, and arcee.cpp:123-128 calls build_ffn with a NULL \
     gate, `LLM_FFN_RELU_SQR` and `LLM_FFN_SEQ` -- i.e. `down(relu(up(x))^2)`, two matrices \
     in sequence. ferrox's `ExpertWeights` has three required matrices and \
     `FfnActivation` has only the gated Swiglu / SwigluFused / Gelu variants \
     (config.rs:302-312), so there is no shape for this and no activation for it either. It \
     fails closed rather than computing SwiGLU: `load_dense_expert` (loader.rs:1112-1136) \
     finds no `ffn_gate`, falls to the Phi-3 fused path, and rejects an `ffn_up` that is \
     `n_ff` rows rather than `2 * n_ff`";

/// Shared by `granite`, `granitemoe` and the `granite-moe` alias: one
/// blocker, one string, so the three rows cannot drift apart.
const GRANITE_MULTIPLIERS: &str =
    "Granite's four multipliers. src/models/granite.cpp:7 reads {arch}.logit_scale as \
     REQUIRED (granite-moe.cpp:5 too) and :8-10 reads residual_scale / embedding_scale / \
     attention.scale; the graph divides the final logits by f_logit_scale (:188) and scales \
     BOTH branch outputs by f_residual_scale before every residual add (:241-242, :301-302). \
     The generic decoder applies none of them, and residual_scale in particular touches \
     every CPU and Metal residual path. In practice a real Granite checkpoint never reaches \
     THIS message: capability::unsupported_scaling_keys already refuses it by name at \
     loader.rs:191, which runs before the unaudited gate. Separately, granite.cpp:206 gates \
     RoPE on `hparams.rope_finetuned`, so a Granite export with rope.finetuned=false gets NO \
     rotation at all -- the ALiBi class of divergence, with no ferrox expression";

/// Triaged rows of the generic **NEOX**-RoPE group. Same rules as
/// [`NORM_ROPE_TRIAGED`].
const NEOX_ROPE_TRIAGED: &[(&str, TriageClass, &str)] = &[
    (
        "mellum",
        TriageClass::NewCode,
        "two per-layer RoPE variants in one model. src/models/mellum.cpp:128-142 runs the \
         SWA layers' RoPE with YaRN switched off -- freq_scale = 1.0, ext_factor = 0.0, \
         attn_factor = 1.0 -- while the full-attention layers use the model's own YaRN \
         (:143-154). ferrox carries one YaRN configuration for the whole model (it has \
         `rope_theta_swa` for the BASE only) and cannot express a per-layer ext_factor. \
         Second, smaller hazard on the same architecture: :12-17 accepts the sliding-window \
         pattern as a scalar OR as a per-layer ARRAY, and ferrox reads it only as a scalar \
         (`GgufValue::as_u64` returns None for an array), so an array-valued file falls back \
         to `default_swa_layout`'s period of 4 with nothing saying it substituted its own \
         layout for the file's. The tensor set and residual (:45-68, :169-197) are generic",
    ),
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
        "attention sinks on a non-gpt-oss architecture, plus per-layer shapes. \
         src/models/mimo2.cpp:58 creates `attn_sinks` per layer; ferrox implements sinks \
         only inside the gpt-oss path and, per docs/MODELS.md, on CPU only. :47-49 reads \
         n_head and the KV widths PER LAYER, :16 and :181 scale the attention output by \
         {arch}.attention.value_scale (a key ferrox neither reads nor gates), :6-12 makes \
         SWA unconditional with a per-layer is_swa ARRAY rather than a period, and :19,:76-82 \
         add NEXTN/MTP layers with a `layer_out_norm`. Any one of the first three would \
         disqualify it; the dense-or-MoE-per-layer choice at :63-72 is the only part ferrox \
         already has",
    ),
    (
        "afmoe",
        TriageClass::NewCode,
        "gated attention plus NoPE layers. src/models/afmoe.cpp:73 creates `wqkv_gate` \
         (LLM_TENSOR_ATTN_GATE), a learned gate applied to the attention output that the \
         generic decoder has no slot for, and :137-138 skips RoPE where \
         `(il + 1) % n_no_rope_layer_step == 0`, the smollm3 class with no GGUF key. It also \
         scales the embeddings by sqrt(n_embd) at :120, which ferrox does only for the Gemma \
         family. THIRD, and the quiet one: :8 reads expert_gating_func as OPTIONAL and \
         :29-30 defaults it to SIGMOID when absent, while ferrox's fallback \
         (loader.rs:375, SIGMOID_GATING_ARCHITECTURES) defaults to softmax for any \
         architecture not on its list -- so a checkpoint omitting the key would be routed \
         through the wrong scoring function. That last one is the `deepseek` shape and would \
         need fixing even if the rest were free",
    ),
    (
        "apertus",
        TriageClass::NewCode,
        "xIELU, with four PER-LAYER parameter arrays. src/models/apertus.cpp:6-9 reads \
         xielu_alpha_n, xielu_alpha_p, xielu_beta and xielu_eps as n_layer-long arrays and \
         :132-135 indexes them per layer; `FfnActivation` (config.rs:302-312) has three \
         variants and no way to carry a per-layer parameter at all. The FFN is also UNGATED \
         -- :45-46 creates only ffn_down and ffn_up, no ffn_gate -- so it is the same \
         two-matrix shape as `arcee` and `plm` on top of the activation. It further requires \
         optional attn_q_norm/attn_k_norm BIASES (:50,:52), and ferrox's norms take a weight \
         only",
    ),
    (
        "exaone-moe",
        TriageClass::NewCode,
        "the GLOBAL layers get no RoPE. src/models/exaone-moe.cpp:155-161 wraps both \
         ggml_rope_ext calls in `if (is_local_layer)`, where is_local_layer is \
         `hparams.is_swa(il)` (:136) -- so on the full-attention layers of every period Q \
         and K are never rotated. ferrox rotates every layer, and there is no GGUF key that \
         says otherwise: the SWA pattern implies it. Checked and CLEAN on the other axis: \
         :5 seeds n_swa = 128 but :13 reads {arch}.attention.sliding_window as REQUIRED, so \
         the window is always the file's own value and ferrox reads the same number, and \
         `default_swa_layout` already carries exaone-moe as period 4. The MoE half (:72-93) \
         -- leading dense, exp_probs_b, shared expert, gating from metadata -- ferrox has",
    ),
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
    (
        "laguna",
        TriageClass::NewCode,
        "per-layer head counts AND a second rotary width. conversion/laguna.py:79 calls \
         `add_head_count(per_layer_heads)` with a LIST, so the array is really in the file, \
         and src/models/laguna.cpp:87-88 and :176-177 read n_head(i) / n_head_kv(i) per \
         layer while ferrox carries both as scalars. :50 then reads \
         LLM_KV_ROPE_DIMENSION_COUNT_SWA into `n_rot_swa`, so the sliding-window layers \
         rotate a DIFFERENT number of dimensions than the full-attention layers (its own \
         comment at :43-45: full layers YaRN over 64 dims, SWA layers plain RoPE over 128); \
         ferrox has one rotary_dim. It also creates `wqkv_gate` (:124), the gated-attention \
         tensor afmoe has, and :55-56 defaults expert_gating_func to SIGMOID when the key is \
         absent where ferrox would default to softmax. `default_swa_layout` already has \
         laguna as dense_first period 4, which is correct and is not the blocker",
    ),
    (
        "step35",
        TriageClass::NewCode,
        "a per-LAYER rotary width. src/models/step35.cpp:65-70 takes `n_rot_max` as the max \
         of `hparams.n_rot(i)` over all layers -- because n_rot varies by layer -- and :9 \
         first halves n_rot_full; ferrox has one rotary_dim for the model. On top of that: \
         per-layer SwiGLU clamp arrays for the routed and shared experts (:28-29, \
         LLM_KV_SWIGLU_CLAMP_EXP / _SHEXP), where ferrox's only clamp is the gpt-oss scalar; \
         a `wqkv_gate` (:96); a per-layer is_swa ARRAY rather than a period (:26), which \
         ferrox reads only as a scalar; NEXTN/MTP layers with trunk-only and MTP-only load \
         modes (:32-49); and expert_gating_func defaulting to SIGMOID when absent (:19-20) \
         where ferrox defaults to softmax. The inventory guessed this was \"probably \
         parameterisable from the gpt-oss clamp\" -- the clamp is, the per-layer n_rot is \
         not",
    ),
    // `mistral`, `mixtral` and `yi` were HERE, UNKNOWN on
    // NO_UPSTREAM_ARCH. The question that verdict asked -- "is there a
    // real GGUF spelling one of these?" -- was answered NO, with a
    // measurement: libllama refuses all three strings outright. They
    // are refused as strings now, not triaged as architectures. See
    // NO_UPSTREAM_ARCH.
    (
        "grok",
        TriageClass::NewCode,
        "grok-1 hardcodes five constants BEFORE letting an optional key override them \
         (src/models/grok.cpp:5-21): logit_scale = 0.5773502691896257 (1/sqrt(3)), \
         embedding_scale = 78.38367176906169, attn_out_scale = 0.08838834764831845 \
         (1/sqrt(128)), and attn / router logit softcapping both 30.0. A GGUF omitting every \
         key is still scaled by all five, so a key-presence gate such as \
         `unsupported_scaling_keys` cannot see them -- the same blind spot `minicpm` is \
         refused for. On top of that the graph is not the generic one: attention runs with \
         kq_scale = 1.0f (:137) and folds the real scale into a tanh softcap instead \
         (llama-graph.cpp:2579-2581), every layer computes BOTH a dense GELU FFN and a GELU \
         MoE and sums them scaled by sqrt(2)/2 (:171-184), and `blk.N.attn_output_norm` \
         (:62, LLM_TENSOR_ATTN_OUT_NORM = \"blk.%d.attn_output_norm\", llama-arch.cpp:423) is \
         a tensor name ferrox never reads. Router logit softcapping has no ferrox concept at \
         all. `uses_geglu` already covers grok's GELU, which is necessary and nowhere near \
         sufficient",
    ),
    (
        "dbrx",
        TriageClass::NewCode,
        "LayerNorm, not RMSNorm. src/models/dbrx.cpp:4 reads LLM_KV_ATTENTION_LAYERNORM_EPS \
         (not the RMS one) and the graph normalises with `LLM_NORM` at all three sites -- \
         :69-71 pre-attention, :110-112 pre-FFN, :140-142 final -- which subtracts the mean; \
         ferrox has only `rms_norm(x, w, eps)`, a different function of the same tensors on \
         every layer. Note this is NOT caught by the required-bias refusal group: dbrx \
         creates no norm bias tensors at all, so the marker that group keys on is absent \
         while the normalisation is still LayerNorm. It also requires \
         {arch}.attention.clamp_kqv (:5, REQUIRED) and carries no `ffn_norm` -- \
         `attn_out_norm` (:34) IS the pre-FFN norm (:110-113), the gpt-oss slot again but \
         under the unread name `blk.%d.attn_output_norm`",
    ),
    (
        "smallthinker",
        TriageClass::NewCode,
        "the MoE router reads a DIFFERENT tensor. src/models/smallthinker.cpp:111 computes \
         the router logits from the raw layer input `inpL`, before the attention block, and \
         passes them into build_moe_ffn as a precomputed `probs` with a NULL ffn_gate_inp \
         (:151-161); every other MoE architecture routes on the normed FFN input, which is \
         what ferrox computes. Two more, either of which alone would disqualify it: (1) NoPE \
         layers with no GGUF key -- llama-hparams.h:203 defaults n_no_rope_layer_step to 4 \
         and the SWA branch (:6-15) never overwrites it, so :108-109's \
         `use_rope = n_no_rope_layer_step == n_layer || il % n_no_rope_layer_step != 0` \
         leaves layers 0, 4, 8 ... unrotated, the `smollm3` class exactly, which ferrox \
         refuses outright; (2) `LLM_FFN_RELU` experts (:158), and FfnActivation has no ReLU \
         variant. :8 also pins n_swa to 4096 over whatever the file declares. \
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
    (
        "openelm",
        TriageClass::NewCode,
        "per-LAYER head counts and FFN width. src/models/openelm.cpp:26-28 reads \
         `hparams.n_head(i)`, `n_head_kv(i)` and `n_ff(i)` per layer and sizes the fused \
         `wqkv` as `n_embd x (2*n_head_kv(i) + n_head(i)) * n_embd_head_k` (:34), and the \
         graph re-derives those widths for every layer (:67-69). ferrox's ModelConfig \
         carries n_heads, n_kv_heads and expert_ffn_dim as SCALARS, and \
         `qkv_fused::load_fused_or_split_qkv` splits a fused QKV at offsets computed from \
         those scalars, \
         so there is nowhere to put this. It fails closed, but NOT with this message: \
         conversion/openelm.py:57-59 writes head_count, head_count_kv and \
         feed_forward_length as ARRAYS, and `GgufValue::as_u64` returns None for an array \
         (ferrox-gguf/src/lib.rs:83-93), so the load dies on a missing-hparam error for keys \
         the file does carry, before the unaudited gate is reached. That misleading message \
         is the `glm4moe` shape and should be fixed alongside",
    ),
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
            // One residual topology, `crate::pre_norm`, shared by both:
            // no pre-attention norm and no pre-FFN norm, each branch's
            // OUTPUT normed before its residual add. The evidence is
            // `tests/post_norm_only_graphs.rs`, one libllama-golden
            // fixture each. `olmo2` with a sliding window AND a RoPE
            // scaling (Olmo-3) and `exaone4` with 64 layers (the 32B)
            // are refused by name in `loader.rs` and are NOT covered by
            // these two rows.
            "olmo2",
            "exaone4",
            // Was a `DedicatedOnly` bias refusal, not an unaudited row:
            // its only dropped bias was the FUSED `attn_qkv.bias`, which
            // `crate::qkv_fused` applies now. Qwen-1 only; qwen2 and
            // later store the split spelling and were already audited.
            "qwen",
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
                "smollm3",
                "a NoPE layer pattern: llama.cpp hardcodes \
                 `hparams.n_no_rope_layer_step = 4` (src/models/smollm3.cpp:5) and \
                 skips RoPE where `(il + 1) % 4 == 0` (:69), so 9 of a 36-layer \
                 SmolLM3-3B's layers get NO rotation at all. There is NO GGUF key \
                 for it, so no metadata gate could see it: the tensor set matches \
                 the generic llama set exactly and the file loads clean. The \
                 generic decoder rotates every layer, which is a different model. \
                 Same shape as the ALiBi group below, and found the same way",
            ),
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
        // MiniCPM is the case `unsupported_scaling_keys` cannot catch:
        // `src/models/minicpm.cpp:4-14` *hardcodes* an embedding
        // multiplier of 12.0, a residual multiplier of
        // `1.4/sqrt(n_layer)` and a logit multiplier of `256/n_embd`,
        // and only then lets the GGUF override them. An older MiniCPM
        // export carrying none of the three keys is still scaled by all
        // three, so a key-presence gate sees nothing and the generic
        // decoder computes an unscaled graph.
        v.push(prof(
            "minicpm",
            TextGeneration,
            StandardGqa,
            KvGqa,
            Norm,
            ArchPath::DedicatedOnly {
                reason: "unconditional embedding/residual/logit multipliers that llama.cpp \
                         applies even when the GGUF omits every key; not applied by the \
                         generic decoder",
            },
            WholeVector,
        ));
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
                reason: "llama4 MoE + non-GQA attn -- see llama4_engine.rs tensor list",
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
pub fn swa_disabled_by_arch(arch: &str) -> bool {
    matches!(arch, "phi3")
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
/// Latent only because `grok` is not in [`AUDITED_GENERIC_GQA`] and so
/// refuses today. It would have become wrong the moment somebody
/// audited it, which is the worst possible time to find out.
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
        // src/models/afmoe.cpp:17. `afmoe` refuses for other reasons
        // today, so this one is latent rather than live, and pinned
        // here so it stays right if that changes.
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
        // src/models/smallthinker.cpp:9-11. LIVE: `smallthinker` is on
        // the generic GQA path.
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

/// Metadata keys that, when present with a nonzero value, require math
/// ferrox's generic decoder does not implement *unless* the architecture
/// profile opts into those features (Gemma family).
pub fn unsupported_feature_keys(arch: &str) -> Vec<(String, &'static str)> {
    let profile = resolve_profile(arch);
    // Gemma family implements softcap + SWA pattern; others still refuse.
    if matches!(profile.map(|p| p.family), Some(DecoderFamily::GemmaFamily)) {
        return Vec::new();
    }
    let key = |suffix: &str| format!("{arch}.{suffix}");
    vec![
        (
            key("attention.logit_softcapping"),
            "attention logit soft-capping (Gemma 2+); not implemented in the generic decoder",
        ),
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
        (
            key("attn_logit_softcapping"),
            "attention logit soft-capping (Gemma 2+); not implemented in the generic decoder",
        ),
        (
            key("final_logit_softcapping"),
            "final logit soft-capping (Gemma 2+); not implemented in the generic decoder",
        ),
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
        // The real gap the key can hide is NOT the pattern: it is that
        // llama.cpp accepts the value as a scalar OR an n_layer-long
        // ARRAY (`ml.get_key_or_arr`), and ferrox carries one scalar
        // period. `loader.rs` refuses an array-valued pattern by name,
        // where the value can actually be inspected, instead of
        // refusing every file that has the key at all.
    ]
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
/// **This is a refusal, not an implementation.** `residual_scale` in
/// particular multiplies the attention and FFN branch outputs before
/// every residual add, which in ferrox means every CPU decode/prefill/
/// multi-seq path *and* the fused Metal kernels that fold the residual
/// in -- landing it half-way would be exactly the silent divergence this
/// list exists to stop. Until the math is there, a checkpoint that
/// declares one of these is refused by name.
///
/// The no-op value differs by key: the three `*_scale` multipliers are
/// `1.0`, while llama.cpp's `f_attention_scale` uses `0.0` as its
/// "unset, use 1/sqrt(head_dim)" sentinel.
pub fn unsupported_scaling_keys(arch: &str) -> Vec<(String, &'static str, f32)> {
    let profile = resolve_profile(arch);
    // Gemma implements its own embedding scale (`loader.rs`
    // `embedding_scale`) and its own attention scale
    // (`attention_scale_override`, including the 27B branch llama.cpp
    // takes at `gemma3.cpp:30-33` / `gemma2.cpp:26-29`). That second
    // half was an unimplemented claim until the 27B fix; the exemption
    // is only honest while `attention_scale_override` covers it, which
    // `gemma_27b_is_the_only_size_that_overrides_the_kernel_scale`
    // pins.
    if matches!(profile.map(|p| p.family), Some(DecoderFamily::GemmaFamily)) {
        return Vec::new();
    }
    let key = |suffix: &str| format!("{arch}.{suffix}");
    vec![
        (
            key("logit_scale"),
            "logit multiplier (Granite / Command-R `logits_scaling`); not applied by the generic decoder",
            1.0,
        ),
        (
            key("residual_scale"),
            "residual multiplier (Granite `residual_multiplier`); not applied by the generic decoder",
            1.0,
        ),
        (
            key("embedding_scale"),
            "embedding multiplier (Granite / MiniCPM `embedding_multiplier`); the generic decoder only scales embeddings for the Gemma family",
            1.0,
        ),
        (
            key("attention.scale"),
            "explicit attention score scale (Granite `attention_multiplier`); the generic decoder always uses 1/sqrt(head_dim)",
            0.0,
        ),
    ]
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
            seen == 25,
            "every unaudited generic architecture is triaged; found {seen}. \
             It was 47 until the triage found `minicpm3` was an MLA model on the \
             generic-GQA row and it moved to DedicatedOnly, 46 until five ONE MATCH ARM \
             rows -- deepseek, bailingmoe, seed_oss, maincoder, hunyuan-moe -- were admitted \
             with libllama-golden fixtures, 41 until seven FIXTURE-AWAY rows -- \
             internlm2, xverse, ernie4_5, baichuan, exaone, bailingmoe2, plamo3 -- got \
             theirs (tests/fixture_away_graphs.rs), 34 until `gemma`, `hunyuan-dense` \
             and `ernie4_5-moe` got theirs, 31 until `olmo2` and `exaone4` -- the \
             POST-NORM-ONLY pair, ONE topology and one implementation \
             (`crate::pre_norm`) -- got theirs (tests/post_norm_only_graphs.rs), 29 until \
             `chatglm` -- the LAST ONE MATCH ARM row -- got its fused-QKV-bias arm and \
             its fixture, and 28 until `mistral`, `mixtral` and `yi` turned out not to be \
             architectures at all (libllama refuses all three strings) and moved to \
             DedicatedOnly. `gemma` was the last fixture-away row and `chatglm` the last \
             one-match-arm row, so BOTH classes are empty: what is left is 24 NEW CODE \
             and one UNKNOWN (`phi4`). `olmo2` and `exaone4` were the first NEW CODE rows \
             to close"
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
        // `olmo`, not `olmo2`: OLMo-2 is audited now (its topology is
        // `crate::pre_norm`), and OLMo-1 is a THIRD residual shape --
        // non-parametric LayerNorm BEFORE both sublayers -- not the
        // post-norm-only one, so it stays NEW CODE and is the sample.
        let new_code = unaudited_refusal_detail("olmo");
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
        assert!(new_code.contains("olmo.cpp:27-35"), "{new_code}");
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
        assert!(!is_audited_generic("mellum"));
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
        assert!(
            matches!(
                resolve_architecture("llama4"),
                Some(ArchPath::DedicatedOnly {
                    reason: "llama4 MoE + non-GQA attn -- see llama4_engine.rs tensor list"
                })
            ),
            "llama4 must fail closed, not silent generic GQA"
        );
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

    /// Parallel attention+FFN residual is not a tensor and, for MiniCPM,
    /// not even a metadata key -- llama.cpp hardcodes MiniCPM's three
    /// multipliers. Neither the tensor-consumption gate nor
    /// `unsupported_scaling_keys` can see the difference, so these
    /// architectures must not be admitted to the generic decoder at all.
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
            "minicpm",
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
        // the comment where it used to be. The array-valued case, which
        // ferrox genuinely cannot express, is refused in `loader.rs`
        // where the value can be inspected.
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
