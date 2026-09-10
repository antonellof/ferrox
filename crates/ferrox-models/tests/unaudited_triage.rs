//! Triage of the `LoadError::UnauditedArchitecture` refusals.
//!
//! ferrox's generic GQA path is opt-in: an architecture that is not in
//! `capability::AUDITED_GENERIC_GQA` refuses rather than running on the
//! guess that it is plain GQA. That closed the "loads and computes
//! something else" class -- `gpt2`, `mpt`, `refact`, `bloom` and `jais`
//! all did exactly that -- but it left **47** architectures refusing with
//! one identical paragraph whose only content is "nobody has checked
//! this".
//!
//! That paragraph is useless for the decision a user actually has, which
//! is whether their model is one fixture away or needs an attention
//! implementation. `docs/plans/llama-cpp-gap-inventory.md` §1.3 shows
//! the 47 split at least three ways, and this suite pins the split for
//! the architectures that have been read on **both** sides:
//!
//! - **fixture-away** -- everything is implemented; evidence is missing.
//! - **one match arm** -- one small, nameable piece: an activation, a
//!   norm slot, a routing flag, an ordering.
//! - **new code** -- a different attention or residual structure.
//! - **UNKNOWN** -- reading did not settle it; the verdict says what
//!   would.
//!
//! **Why this suite exists rather than only the unit tests in
//! `capability.rs`:** the failure mode here is not a compile error, it
//! is a *confident wrong verdict*. This repo has now found four
//! architectures whose refusal named something that was not the real
//! blocker -- `glm4moe` was told it lacked an MLA hyper-parameter it must
//! not have, and `minimax-m2` was blamed on MTP weights no converter can
//! emit. Each assertion below therefore pins a specific claim about a
//! specific llama.cpp line, so that changing the verdict without
//! changing the reading fails.
//!
//! Every claim pinned here was read in
//! `.scratch/llama.cpp/src/models/*.cpp` against ferrox's generic
//! decoder, and the citation is in the verdict string itself. One test
//! is the exception and reads FILES rather than source --
//! `real_mistral_and_yi_checkpoints_declare_llama`, `#[ignore]`d
//! because it needs the checkpoints in `models/`. It is the measurement
//! that closed the last three UNKNOWN rows.

mod common;
use common::collect_gguf;
use ferrox_models::capability::{
    architecture_catalog, is_audited_generic, unaudited_refusal_detail, unaudited_triage, ArchPath,
    TriageClass, TRIAGE_PENDING,
};

/// Every architecture that reaches the unaudited refusal gets a
/// non-empty detail line -- triaged or not.
///
/// A blank detail would be the old refusal wearing a new field.
#[test]
fn every_unaudited_architecture_renders_a_detail_line() {
    let mut n = 0;
    for p in architecture_catalog() {
        if !matches!(p.path, ArchPath::GenericGqa { .. }) || is_audited_generic(p.gguf_name) {
            continue;
        }
        n += 1;
        let detail = unaudited_refusal_detail(p.gguf_name);
        assert!(
            detail.starts_with("TRIAGE"),
            "`{}` renders {detail:?}",
            p.gguf_name
        );
        assert!(detail.len() > 100, "`{}` renders {detail:?}", p.gguf_name);
    }
    assert_eq!(
        n, 21,
        "the unaudited count moved. It was 47 until the triage itself found `minicpm3` was \
         an MLA model sitting on the generic-GQA row and it was reclassified to \
         DedicatedOnly, 46 until `deepseek`, `bailingmoe`, `seed_oss`, `maincoder` and \
         `hunyuan-moe` were admitted with libllama-golden fixtures, 41 until \
         `internlm2`, `xverse`, `ernie4_5`, `baichuan`, `exaone`, `bailingmoe2` and \
         `plamo3` were \
         admitted with theirs (`tests/fixture_away_graphs.rs`), 34 until `gemma`, \
         `hunyuan-dense` and `ernie4_5-moe` were admitted with theirs, 31 until \
         `olmo2` and `exaone4` were -- the first two NEW CODE rows to close, and they \
         closed TOGETHER because they are one topology with one implementation \
         (`ferrox_models::norm`, `tests/post_norm_only_graphs.rs`) -- 29 until \
         `chatglm` was admitted with the fused-QKV-bias arm and its fixture, 28 until \
         `mistral`, `mixtral` and `yi` were found not to be architectures at all and \
         moved to DedicatedOnly, and 25 until `granite`, `granitemoe` and the \
         `granite-moe` alias closed together on ONE implementation of their four scalar \
         multipliers (`tests/granite_family_graphs.rs`), and 22 until `olmo` closed on \
         the non-parametric LayerNorm (`ferrox_models::norm`, `tests/olmo_graphs.rs`) -- \
         the FIRST NEW CODE row to close alone, and it closed alone because its cause \
         really is unshared: every `build_norm` call in llama.cpp's 140 graphs was \
         scanned for a null weight and all three hits are `olmo.cpp` -- rows closing is \
         the count going DOWN \
         for the best reason. Either an architecture was audited or reclassified (good -- \
         update the count and the docs) or one was added (check it was triaged)"
    );
}

/// Batch 1: the architectures people actually download, with the class
/// each was placed in and the llama.cpp fact that decides it.
///
/// The class is asserted together with a substring of the blocker on
/// purpose. Asserting the class alone would let somebody flip a verdict
/// and leave the (now-contradictory) reasoning in place, which is
/// exactly how `glm4moe` came to refuse for a reason it did not have.
#[test]
fn batch_one_verdicts_are_pinned_to_what_was_read() {
    let cases: &[(&str, TriageClass, &str)] = &[
        // --- fixture-away: implemented, unevidenced ------------------
        //
        // THIS CLASS IS NOW EMPTY. `gemma` was the last row in it and
        // got its fixture (`tests/fixture_away_graphs.rs`), so every
        // architecture still refusing needs code, not evidence. That is
        // the honest headline and
        // `every_unaudited_row_is_triaged_and_the_distribution_is_pinned`
        // is what holds it.
        //
        // `internlm2`, `exaone` and `ernie4_5` were HERE, and so were
        // `xverse` and `baichuan` in batch three. All five got their
        // fixture (`tests/fixture_away_graphs.rs`), so they are audited
        // now and carry no verdict at all. So did `bailingmoe2`, the one
        // MoE row of the six --
        // `every_verdict_is_attached_to_a_row_that_actually_refuses_as_unaudited`
        // is what stops a stale verdict outliving its refusal. Closing a
        // FIXTURE-AWAY row is the cheapest kind of progress there is and
        // the count going down here is what it looks like.
        //
        // --- one match arm: small and nameable -----------------------
        //
        // `seed_oss`, `deepseek` and `hunyuan-moe` were HERE. All three
        // arms landed, with libllama-golden fixtures
        // (`tests/one_match_arm_graphs.rs`), so they are audited now and
        // carry no verdict at all —
        // `every_verdict_is_attached_to_a_row_that_actually_refuses_as_unaudited`
        // is what stops a stale verdict outliving its refusal.
        //
        // `ernie4_5-moe` was HERE, ONE MATCH ARM on
        // `interleave_moe_layer_step`. The arm landed as a REFUSAL
        // rather than an implementation -- llama.cpp's own tensor loader
        // (ernie4-5.cpp:49) has no step in it, so an interleaved
        // checkpoint cannot be loaded by llama.cpp either -- and the
        // step every real checkpoint carries is audited against libllama
        // (`tests/one_match_arm_graphs.rs`), so the row carries no
        // verdict at all.
        //
        // --- new code: a different graph -----------------------------
        //
        // `olmo2` and `exaone4` used to head this group -- no attn_norm
        // and no ffn_norm at all, Q/K/V off the raw residual. That
        // blocker was real and it is now IMPLEMENTED, once, for both
        // (`ferrox_models::norm`), so neither carries a verdict any
        // more, and `the_post_norm_group_is_three_topologies_and_only_two_of_them_closed` below
        // is now about which of the three shapes each of the group is.
        //
        // `granite`, `granitemoe` and the `granite-moe` alias were HERE
        // too, NEW CODE on the four scalar multipliers
        // (granite.cpp:5-10,180,225,235-238,288-292). All three closed
        // together, on ONE implementation of the multipliers
        // (`ferrox_models::scalar_multipliers`) rather than three, and
        // all three have libllama-golden fixtures
        // (`tests/granite_family_graphs.rs`) -- the alias by reading
        // `granitemoe`'s golden out of a file that differs only in its
        // architecture string, because no llama.cpp GGUF spells it that
        // way and none ever will. They carry no verdict now;
        // `every_verdict_is_attached_to_a_row_that_actually_refuses_as_unaudited`
        // is what stops a stale one outliving its refusal.
        //
        // BOTH closures took more than one row at a time, and for the
        // same reason: each found ONE cause behind several refusals.
        // That is what the NEW CODE column moving looks like.
        //
        // The `rope_finetuned` half of the Granite verdict did NOT
        // become an implementation. granite.cpp:33-35 reads
        // `{arch}.rope.scaling.finetuned` as a switch for RoPE itself,
        // so a file declaring it false runs unrotated in llama.cpp and
        // ferrox refuses it by name (`ferrox_models::rope_finetuned`).
        // --- unknown: say so, and say what would settle it -----------
        //
        // `phi4` is not a llama.cpp architecture at all, so there is no
        // graph to diff against.
        ("phi4", TriageClass::Unknown, "WHAT WOULD SETTLE IT"),
    ];

    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
}

/// The "post-norm group" was never one group, and it is THREE norm
/// topologies, not two.
///
/// `docs/plans/llama-cpp-gap-inventory.md` §1.3 grouped `olmo2`,
/// `seed_oss` and `exaone4` together as "likely a fixture away, if the
/// loader wires the post-norm slots for non-Gemma families". The wiring
/// question had a yes answer, and it was the wrong question: the three
/// are three different residual shapes and were never one class.
///
/// * `seed_oss` HAS `attn_norm` and uses `attn_post_norm` as its
///   pre-FFN norm (`seed-oss.cpp:36-37,113-115`). Closed 2026-09-02.
/// * `olmo2` and `exaone4` have NEITHER pre-norm and read the raw
///   residual at both sublayers (`olmo2.cpp:45-52,92,169`,
///   `exaone4.cpp:60-67,118,159`). That is one topology across the two,
///   `ferrox_models::norm` is the one implementation, and
///   `tests/post_norm_only_graphs.rs` is the evidence for both.
/// * `olmo` (OLMo-1) is the third: it norms BEFORE both sublayers, so
///   it is pre-norm like llama, and what it lacks is the norm FUNCTION
///   -- non-parametric LayerNorm, all three `build_norm` calls with a
///   NULL weight (`olmo.cpp:65-67,104-106,128-130`). It still refuses,
///   and `norm` is no help to it.
///
/// Pinning the split stops the grouping being restored from the prose,
/// and it now also pins that closing two of the three did NOT sweep the
/// third along with them.
#[test]
fn the_post_norm_group_is_three_topologies_and_only_two_of_them_closed() {
    // seed_oss: has a pre-attention norm; its post_attention_norm IS
    // the pre-FFN norm.
    assert!(
        is_audited_generic("seed_oss") && unaudited_triage("seed_oss").is_none(),
        "seed_oss has attn_norm and olmo2 does not; they were never one class"
    );

    // olmo2 / exaone4: one topology, one implementation, both audited.
    for name in ["olmo2", "exaone4"] {
        assert!(
            is_audited_generic(name),
            "`{name}` closed with a libllama-golden fixture"
        );
        assert!(
            unaudited_triage(name).is_none(),
            "`{name}` is audited and must carry no verdict"
        );
        assert!(
            ferrox_models::capability::is_post_norm_only(name),
            "`{name}` is the post-norm-only topology"
        );
    }
    assert_eq!(
        ferrox_models::capability::POST_NORM_ONLY_ARCHITECTURES,
        &["olmo2", "exaone4"],
        "a third name here is a third llama.cpp graph somebody read"
    );

    // olmo (OLMo-1) is the THIRD topology and it closed too, on a
    // different variant of the same enum -- pre-norm, with a
    // non-parametric LayerNorm at all three sites
    // (`tests/olmo_graphs.rs`). What still has to hold is that it is
    // NOT on the post-norm-only list: a decoder that read OLMo-1 that
    // way would drop both its norms and answer fluently.
    assert!(is_audited_generic("olmo"));
    assert!(
        unaudited_triage("olmo").is_none(),
        "`olmo` is audited and must carry no verdict"
    );
    assert!(
        !ferrox_models::capability::is_post_norm_only("olmo"),
        "OLMo-1 norms before both sublayers; it is not post-norm-only"
    );
    assert!(
        ferrox_models::capability::uses_non_parametric_layer_norm("olmo"),
        "OLMo-1's norm has no parameters, which is the whole row"
    );
    assert_eq!(
        ferrox_models::capability::NON_PARAMETRIC_LAYER_NORM,
        &["olmo"],
        "a second name here would be a second llama.cpp graph with a null-weight \
         `LLM_NORM`, and the scan over all 140 found none"
    );
}

/// `ernie4_5-moe` is audited, and the sigmoid correction moved from its
/// verdict into a golden test rather than being dropped.
///
/// The inventory (§1.3) lists it beside `bailingmoe2` as "sigmoid-routed
/// MoE with `ffn_exp_probs_b` router bias". `bailingmoe2` reads its
/// gating function from metadata (`bailingmoe2.cpp:11`) so it can be
/// either; `ernie4-5-moe.cpp:90` **hardcodes**
/// `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`. That correction used to live
/// in the refusal string, which is gone now, so this pins the join: the
/// row really did move to the audited side, and the claim it used to
/// make in words is now made in logits by
/// `routing_ernie_moe_through_sigmoid_instead_of_softmax_diverges_from_llama_cpp`.
#[test]
fn ernie_moe_is_audited_and_carries_no_stale_sigmoid_verdict() {
    assert!(
        is_audited_generic("ernie4_5-moe"),
        "ernie4_5-moe was admitted with a libllama-golden fixture at step 1"
    );
    assert!(
        unaudited_triage("ernie4_5-moe").is_none(),
        "an audited row must carry no verdict; the interleave step it used to describe is \
         now a named refusal in `moe_interleave` for any step above 1"
    );
}

/// Every triaged architecture really does reach the unaudited refusal.
///
/// A verdict on a row that refuses for a *different*, named reason would
/// be dead text the user never sees -- the same shape as the `glm4moe`
/// bug, where the reason shown and the reason true were two different
/// strings.
#[test]
fn every_verdict_is_attached_to_a_row_that_actually_refuses_as_unaudited() {
    for p in architecture_catalog() {
        let Some(t) = p.triage else { continue };
        assert!(
            matches!(p.path, ArchPath::GenericGqa { .. }),
            "`{}` carries a {:?} verdict but resolves to {:?}, which refuses elsewhere",
            p.gguf_name,
            t.class,
            p.path
        );
        assert!(
            !is_audited_generic(p.gguf_name),
            "`{}` is audited and runs; a triage verdict there is never rendered",
            p.gguf_name
        );
    }
}

/// The pending list is a to-do with a shrinking count, not a parking
/// bay.
///
/// If this number goes UP without the total moving, an architecture lost
/// its verdict.
#[test]
fn the_remaining_work_is_counted() {
    assert_eq!(
        TRIAGE_PENDING.len(),
        0,
        "all 47 unaudited architectures are triaged; a name reappearing here means a new \
         architecture reached the generic path without being read"
    );
    let triaged = architecture_catalog()
        .iter()
        .filter(|p| p.triage.is_some())
        .count();
    assert_eq!(triaged + TRIAGE_PENDING.len(), 21);
}

/// `minicpm3` is refused as an MLA model, not as an unaudited one.
///
/// The triage found the catalog claimed `StandardGqa`/`KvGqa` for a
/// model whose every checkpoint carries `attn_q_a`/`attn_kv_a_mqa` and
/// no `attn_q.weight` (`src/models/minicpm3.cpp:41-46`), so the generic
/// path could never have loaded one. Being told "unaudited" for that is
/// telling the user the wrong thing about their model.
///
/// A message-quality fix rather than a correctness one -- the old
/// failure was already a clean missing-tensor error -- which is why the
/// reason has to name BOTH blockers, the MLA tensor set and the
/// hardcoded MiniCPM multipliers.
#[test]
fn minicpm3_is_refused_as_mla_not_as_unaudited() {
    assert!(
        unaudited_triage("minicpm3").is_none(),
        "minicpm3 left the unaudited generic set"
    );
    match ferrox_models::capability::resolve_architecture("minicpm3") {
        Some(ArchPath::DedicatedOnly { reason }) => {
            assert!(reason.contains("MLA"), "{reason}");
            assert!(
                reason.contains("scale_depth"),
                "the multipliers are the second blocker and must be named: {reason}"
            );
        }
        other => panic!("minicpm3 must be DedicatedOnly, got {other:?}"),
    }
}

/// The untriaged message still works, and still claims no class.
///
/// `TRIAGE_PENDING` is empty now that all 47 are read, so this exercises
/// the branch through a name the catalog does not carry. It is what a
/// NEW architecture added to the generic path would render until
/// somebody reads it, and it must not imply a class: "unaudited" and
/// "untriaged" are different claims, and reading a class into the
/// untriaged message is how a guess becomes a citation.
#[test]
fn the_untriaged_message_claims_no_class() {
    let d = unaudited_refusal_detail("an-architecture-nobody-has-read-yet");
    assert!(d.contains("not done for"), "{d}");
    for label in [
        TriageClass::FixtureAway.label(),
        TriageClass::OneMatchArm.label(),
        TriageClass::NewCode.label(),
    ] {
        assert!(
            !d.contains(label),
            "the untriaged message implies {label}: {d}"
        );
    }
}

/// Batch 2: the next six by download volume, plus the two the
/// activation audit turned up on the way.
///
/// Every one came out `NewCode`, which is itself the finding. Batch 1
/// mixed five fixture-away rows in; past the first dozen the unaudited
/// set is genuinely harder, and the refusal now says so per
/// architecture instead of implying a uniform distance.
#[test]
fn batch_two_verdicts_are_pinned_to_what_was_read() {
    let cases: &[(&str, TriageClass, &str)] = &[
        ("grok", TriageClass::NewCode, "grok.cpp:5-21"),
        ("dbrx", TriageClass::NewCode, "LayerNorm, not RMSNorm"),
        (
            "smallthinker",
            TriageClass::NewCode,
            "router reads a DIFFERENT tensor",
        ),
        ("bitnet", TriageClass::NewCode, "attn_sub_norm"),
        // `minicpm3` was HERE, and the triage that produced this list
        // is what removed it: reading `minicpm3.cpp:5-6,41-46` showed an
        // MLA tensor set on a row the catalog called `StandardGqa`, so
        // it moved to `DedicatedOnly` rather than staying an unaudited
        // generic architecture. Its refusal is now asserted by
        // `minicpm3_is_refused_as_mla_not_as_unaudited` below.
        ("openelm", TriageClass::NewCode, "per-LAYER head counts"),
        ("arcee", TriageClass::NewCode, "UNGATED ReLU-squared"),
        ("plm", TriageClass::NewCode, "UNGATED ReLU-squared"),
    ];
    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
}

/// `smallthinker`'s blocker leads with the routing input, not with the
/// activation.
///
/// It has three separate blockers and they are not equally severe. The
/// ReLU experts are one match arm on their own; the router reading
/// `inpL` instead of the normed FFN input, and the unkeyed NoPE layers,
/// are both "computes something else" and neither leaves a tensor or a
/// metadata key behind. A verdict that named only the activation would
/// read as one match arm and be wrong by two.
#[test]
fn smallthinker_names_the_routing_input_and_the_nope_layers() {
    let t = unaudited_triage("smallthinker").expect("verdict");
    assert_eq!(t.class, TriageClass::NewCode);
    for claim in ["inpL", "n_no_rope_layer_step", "smollm3", "LLM_FFN_RELU"] {
        assert!(
            t.blocker.contains(claim),
            "smallthinker's verdict drops {claim:?}: {}",
            t.blocker
        );
    }
    let routing = t.blocker.find("inpL").expect("inpL");
    let relu = t.blocker.find("LLM_FFN_RELU").expect("relu");
    assert!(
        routing < relu,
        "the severe blocker must lead: {}",
        t.blocker
    );
}

/// `dbrx` is refused for its normalisation, and the verdict says why the
/// existing bias-tensor refusal group does NOT cover it.
///
/// The group keys on required `*_norm.bias` tensors as the marker of a
/// real LayerNorm. `dbrx` creates none of them and is still LayerNorm,
/// because llama.cpp's `LLM_NORM` subtracts the mean with or without a
/// bias. Somebody reading only that group's comment would conclude dbrx
/// is fine.
#[test]
fn dbrx_says_why_the_bias_group_does_not_catch_it() {
    let t = unaudited_triage("dbrx").expect("verdict");
    assert!(
        t.blocker.contains("creates no norm bias tensors"),
        "dbrx's verdict must say the bias marker is absent: {}",
        t.blocker
    );
}

/// Where the refusal a user actually sees is NOT this one, the verdict
/// says so rather than letting the reader assume the triage line is what
/// they got.
///
/// `openelm` dies on a missing-hparam error for keys its file does
/// carry, before the unaudited gate. A verdict that stayed silent about
/// that would send someone looking for a message they will never see.
///
/// `granite` was the other case here: it died on
/// `capability::unsupported_scaling_keys`, which refused the very
/// multipliers its verdict named. Both halves are gone -- the
/// multipliers are implemented and the row is audited -- so the case
/// left this list rather than being kept as a stale example. One row is
/// enough to pin the rule; nothing is left that has to be true about
/// `granite` for it to hold.
#[test]
fn verdicts_disclose_when_an_earlier_refusal_fires_first() {
    let (arch, marker) = ("openelm", "before the unaudited gate is reached");
    let t = unaudited_triage(arch).expect("verdict");
    assert!(
        t.blocker.contains(marker),
        "`{arch}` must disclose that an earlier refusal fires first: {}",
        t.blocker
    );
}

/// Batch 3: the alias rows and the plain long-tail.
///
/// This is the batch that was expected to be cheap, and half of it was.
/// `xverse` and `baichuan` really were llama-shaped, are audited now
/// (`tests/fixture_away_graphs.rs`) and so carry no verdict any more.
/// `deci` and `olmo` are not llama-shaped, and neither is the alias
/// trio, for a reason that has nothing to do with their graphs.
#[test]
fn batch_three_verdicts_are_pinned_to_what_was_read() {
    let cases: &[(&str, TriageClass, &str)] = &[
        // `chatglm` was FIXTURE-AWAY here, then ONE MATCH ARM once
        // somebody tried to build its fixture and read the converter.
        // The arm -- the fused `attn_qkv.bias` -- landed in
        // `qkv_fused`, so the row is audited and carries no verdict at
        // all. It was the LAST one-match-arm row anywhere; see
        // `tests/one_match_arm_graphs.rs`.
        //
        // `mistral`, `mixtral` and `yi` were HERE too, UNKNOWN on "what
        // would settle it: a real GGUF spelling one of these". The
        // answer came back NO -- libllama refuses all three strings --
        // so they are refused as strings now, not triaged as
        // architectures. See
        // `the_alias_rows_are_refused_as_strings_no_converter_writes`.
        ("deci", TriageClass::NewCode, "PER LAYER"),
        // `olmo` was HERE, NEW CODE on "NO norm weights at all". It is
        // audited now (`tests/olmo_graphs.rs`) and carries no verdict;
        // `the_post_norm_group_is_three_topologies...` above is where
        // the claim about it lives.
    ];
    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
}

/// The three alias rows are refused as STRINGS NOBODY WRITES, not
/// triaged as architectures.
///
/// This closes the UNKNOWN their old verdict opened. That verdict asked
/// for "a real GGUF whose general.architecture is literally one of
/// these three", and the answer is that no such file can be produced:
///
///   * `mistral`, `mixtral` and `yi` are in neither `LLM_ARCH_NAMES`
///     (`src/llama-arch.cpp` carries `mistral3` and `mistral4` and
///     nothing else under that prefix) nor gguf-py's
///     `MODEL_ARCH_NAMES`.
///   * libllama REFUSES a file declaring any of the three:
///     `llama_model_load: error loading model: unknown model
///     architecture: 'mistral'` -- measured on a synthetic llama-shaped
///     file written under each string, which is also why no golden
///     reference for these rows can ever exist.
///   * The two real checkpoints in `models/` both declare `llama`; see
///     `real_mistral_and_yi_checkpoints_declare_llama` below.
///
/// Leaving them on the generic path was a live hazard, and the reason
/// this had to move rather than merely be re-worded: they carried NEOX
/// while `llama` -- the graph they claim to be -- is in
/// `llama_model_rope_type`'s NORM group, and
/// `rope_layout_matches_llama_cpp` cannot see it, because a name absent
/// from llama.cpp's table is a `continue` there.
#[test]
fn the_alias_rows_are_refused_as_strings_no_converter_writes() {
    for arch in ["mistral", "mixtral", "yi"] {
        assert!(
            unaudited_triage(arch).is_none(),
            "`{arch}` still carries a triage verdict; it is not an unaudited architecture, \
             it is a string no converter writes"
        );
        match ferrox_models::capability::resolve_architecture(arch) {
            Some(ArchPath::DedicatedOnly { reason }) => {
                // The refusal has to carry the actionable half -- "your
                // file is spelled `llama`" -- and the measurement that
                // decided it. A refusal that only says no sends the user
                // back to the same question.
                for claim in [
                    "LLM_ARCH_NAMES",
                    "unknown model architecture",
                    "general.architecture = llama",
                    "convert_hf_to_gguf.py",
                ] {
                    assert!(
                        reason.contains(claim),
                        "`{arch}`'s refusal drops {claim:?}: {reason}"
                    );
                }
            }
            other => panic!("`{arch}` must be refused as an alias, got {other:?}"),
        }
    }
    // And the string they redirect to must actually run, or the advice
    // is wrong.
    assert!(is_audited_generic("llama"));
}

/// The measurement behind the row above, re-runnable on this machine.
///
/// `#[ignore]`d because it needs the real checkpoints in `models/`.
/// Every Mistral, Mixtral and Yi GGUF converts to `llama`, and this is
/// what says so from FILES rather than from a reading of
/// `conversion/*.py`. If a checkpoint ever turns up declaring one of
/// the three, this fails and the alias rows need re-reading with that
/// file in hand -- which is exactly the evidence their old UNKNOWN
/// verdict asked for and nobody could supply.
///
///     FERROX_TEST_MODELS_DIR=$PWD/models \
///       cargo test -p ferrox-models --test unaudited_triage -- --ignored
///
/// The env var is not optional in practice: `cargo test` runs with the
/// PACKAGE directory as its cwd, so the bare `models` default resolves
/// under `crates/ferrox-models/` and finds nothing. Same convention as
/// `tests/chat_template_real_gguf.rs`. Run once on 2026-09-10 over the
/// development host's 20 checkpoints: none declares one of the three
/// strings, and both Mistral/Yi files declare `llama`.
#[test]
#[ignore = "needs the real GGUF checkpoints in models/"]
fn real_mistral_and_yi_checkpoints_declare_llama() {
    let root = std::env::var("FERROX_TEST_MODELS_DIR").unwrap_or_else(|_| "models".to_string());
    let mut files = Vec::new();
    collect_gguf(std::path::Path::new(&root), &mut files);
    assert!(!files.is_empty(), "no GGUFs under {root}");

    let mut checked = 0;
    for path in &files {
        let Ok(file) = ferrox_gguf::GgufFile::open(path.to_str().unwrap()) else {
            continue;
        };
        let arch = ferrox_gguf::TensorSource::metadata_str(&file, "general.architecture")
            .unwrap_or_default()
            .to_string();
        assert!(
            !["mistral", "mixtral", "yi"].contains(&arch.as_str()),
            "{}: declares `{arch}`, which no converter was thought to write -- re-read \
             the alias rows with this file in hand",
            path.display()
        );
        let name = ferrox_gguf::TensorSource::metadata_str(&file, "general.name")
            .unwrap_or_default()
            .to_lowercase();
        if name.contains("mistral") || name.contains("mixtral") || name.contains("yi-") {
            checked += 1;
            assert_eq!(
                arch,
                "llama",
                "{}: a Mistral/Mixtral/Yi checkpoint that is not `llama`",
                path.display()
            );
        }
    }
    assert!(
        checked > 0,
        "no Mistral/Mixtral/Yi checkpoint under {root}, so this proved nothing"
    );
}

/// `baichuan` is one architecture string covering two different models,
/// and admitting it admitted only ONE of them.
///
/// This test used to read the triage verdict, which said in words which
/// model the refusal was about. The verdict is gone -- `baichuan` is
/// audited now -- so the same fact has to be held somewhere, and it is
/// held in two places that must agree: `loader.rs` refuses
/// `block_count == 40` by name before the audited list is ever
/// consulted (`baichuan_13b_is_refused_because_it_uses_alibi_and_the_7b_is_not`),
/// and the fixture behind the admission has 32 layers, not 2, because
/// `src/models/baichuan.cpp:5-14` reads the variant off the layer count
/// and any other value gets NO RoPE
/// (`the_baichuan_fixture_has_the_32_layers_that_select_the_rotating_variant`).
///
/// What this asserts is the join: that the name really did move to the
/// audited side, so a reader who finds the 13B refusal knows it is not
/// the whole story.
#[test]
fn baichuan_is_audited_as_the_7b_and_carries_no_stale_verdict() {
    assert!(
        is_audited_generic("baichuan"),
        "baichuan-7B was admitted with a libllama-golden fixture"
    );
    assert!(
        unaudited_triage("baichuan").is_none(),
        "an audited row must carry no verdict; the refusal it used to describe now lives \
         in loader.rs's block_count == 40 check, which is about the 13B alone"
    );
}

/// Batch 4 and batch 5: the remaining long tail.
#[test]
fn batches_four_and_five_verdicts_are_pinned_to_what_was_read() {
    let cases: &[(&str, TriageClass, &str)] = &[
        // Batch 4. `maincoder` and `bailingmoe` were here and are now
        // audited; see `tests/one_match_arm_graphs.rs`.
        ("arctic", TriageClass::NewCode, "PARALLEL dense+MoE"),
        (
            "mistral3",
            TriageClass::NewCode,
            "attention temperature tuning",
        ),
        (
            "nanbeige",
            TriageClass::NewCode,
            "RUNS THE SAME PHYSICAL LAYERS MORE THAN ONCE",
        ),
        (
            "mellum",
            TriageClass::NewCode,
            "two per-layer RoPE variants",
        ),
        ("talkie", TriageClass::NewCode, "NO norm weights"),
        ("mimo2", TriageClass::NewCode, "attention sinks"),
        // Batch 5. `plamo3` was here, FIXTURE-AWAY. Building its
        // fixture found the verdict was wrong by one tensor name -- it
        // is the only architecture upstream that spells its two
        // post-norms without a `.weight` suffix -- so the arm landed in
        // `loader.rs` and it is audited now
        // (`tests/fixture_away_graphs.rs`).
        ("afmoe", TriageClass::NewCode, "gated attention"),
        ("apertus", TriageClass::NewCode, "xIELU"),
        (
            "exaone-moe",
            TriageClass::NewCode,
            "GLOBAL layers get no RoPE",
        ),
        ("grovemoe", TriageClass::NewCode, "SECOND bank of experts"),
        // `hunyuan-dense` was HERE, ONE MATCH ARM on the NTK-alpha RoPE
        // base rescale. The arm landed (`rope_ntk_alpha`) and is
        // evidenced against libllama, so the row is audited and carries
        // no verdict --
        // `the_qk_norm_ordering_arm_is_no_longer_anybody_s_leading_blocker`
        // below is what pins that.
        ("laguna", TriageClass::NewCode, "second rotary width"),
        ("step35", TriageClass::NewCode, "per-LAYER rotary width"),
    ];
    for (arch, class, evidence) in cases {
        let t = unaudited_triage(arch).unwrap_or_else(|| panic!("`{arch}` carries no verdict"));
        assert_eq!(t.class, *class, "`{arch}` changed class");
        assert!(
            t.blocker.contains(evidence),
            "`{arch}` is still {class:?} but no longer says {evidence:?}: {}",
            t.blocker
        );
    }
}

/// The QK-norm ordering arm was wanted by three architectures, and all
/// three run on it now.
///
/// The cross-row check was the argument for adding a shared flag rather
/// than special-casing one architecture, and this is what it bought:
/// `hunyuan-dense` needed only to be added to
/// `QK_NORM_AFTER_ROPE_ARCHITECTURES`, leaving one real arm (the
/// NTK-alpha base rescale) rather than two. The test now runs the other
/// way round -- no row may still refuse for an ordering that is
/// implemented, which is the `glm4moe` defect exactly: the reason shown
/// and the reason true being two different strings.
#[test]
fn the_qk_norm_ordering_arm_is_no_longer_anybody_s_leading_blocker() {
    for arch in ["hunyuan-moe", "maincoder", "hunyuan-dense"] {
        assert!(
            is_audited_generic(arch) && unaudited_triage(arch).is_none(),
            "`{arch}` got the ordering arm and evidence; it must not still be refused"
        );
    }
    for p in architecture_catalog() {
        let Some(t) = p.triage else { continue };
        assert!(
            !t.blocker.contains("QK norm AFTER") && !t.blocker.contains("QK-norm AFTER"),
            "`{}` still leads with an ordering ferrox implements: {}",
            p.gguf_name,
            t.blocker
        );
    }
}

/// `exaone-moe`'s hardcoded `n_swa = 128` was checked and is NOT a
/// divergence, and the verdict records that.
///
/// A cross-cutting sweep of every `hparams.n_swa =` in llama.cpp turned
/// this up as a candidate for the `deepseek` shape: a per-architecture
/// default with no key to correct it. It is not one, because
/// `exaone-moe.cpp:13` reads the window as a REQUIRED key. A clean
/// result is still a result, and recording it stops the next person
/// re-running the same sweep and re-raising the same false alarm.
#[test]
fn exaone_moe_records_the_swa_sweep_as_clean() {
    let t = unaudited_triage("exaone-moe").expect("verdict");
    assert!(
        t.blocker.contains("CLEAN"),
        "exaone-moe's verdict must record the checked-and-clean axis: {}",
        t.blocker
    );
    assert!(t.blocker.contains("REQUIRED"), "{}", t.blocker);
}

/// Every one of the 47 now carries a verdict, and the four classes are
/// all represented.
///
/// The distribution is the headline: `NewCode` dominates. That is the
/// honest answer to "how far is ferrox from llama.cpp on models", and it
/// is the number this whole item existed to produce.
#[test]
fn every_unaudited_row_is_triaged_and_the_distribution_is_pinned() {
    let mut fixture = 0;
    let mut arm = 0;
    let mut new_code = 0;
    let mut unknown = 0;
    for p in architecture_catalog() {
        if !matches!(p.path, ArchPath::GenericGqa { .. }) || is_audited_generic(p.gguf_name) {
            continue;
        }
        match p.triage.expect("every unaudited row is triaged").class {
            TriageClass::FixtureAway => fixture += 1,
            TriageClass::OneMatchArm => arm += 1,
            TriageClass::NewCode => new_code += 1,
            TriageClass::Unknown => unknown += 1,
        }
    }
    assert_eq!(
        (fixture, arm, new_code, unknown),
        (0, 0, 20, 1),
        "the triage distribution moved; if a verdict changed on evidence that is correct, \
         update this and docs/MODELS.md together. TWO classes are ZERO now: `gemma` was \
         the last FIXTURE-AWAY row and `chatglm` the last ONE MATCH ARM one, so nothing \
         refusing today is one fixture or one arm away and EVERY remaining row is NEW \
         CODE. That column went 26 to 24 when `olmo2` and `exaone4` closed together -- \
         one topology, one implementation -- 24 to 21 when `granite`, `granitemoe` \
         and the `granite-moe` alias closed on ONE implementation of their four scalar \
         multipliers, and 21 to 20 when `olmo` closed on the non-parametric LayerNorm. \
         The first two closures took several rows at once because each found ONE cause \
         behind several refusals; `olmo` is the first that did not, and the reason is \
         recorded rather than hoped over -- every `build_norm` call in llama.cpp's 140 \
         graphs was scanned for a null weight and all three hits are `olmo.cpp`, so \
         there was no second row to take. The \
         single UNKNOWN left is `phi4`; `mistral`, `mixtral` and `yi` were the other \
         three and turned out not to be architectures at all"
    );
    assert_eq!(fixture + arm + new_code + unknown, 21);
}
