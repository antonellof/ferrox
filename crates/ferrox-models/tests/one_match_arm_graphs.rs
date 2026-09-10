//! The ONE-MATCH-ARM architectures, checked against llama.cpp itself.
//!
//! `capability.rs` triaged 46 unaudited architectures into four classes.
//! ONE MATCH ARM means one small, nameable piece missing -- an
//! activation, a norm slot, a routing flag, an ordering. Seven are
//! closed here. Each needed a different arm, and each fixture is built
//! so that getting THAT arm wrong is a large, obvious divergence rather
//! than a rounding difference:
//!
//! | arch | the arm | what would break |
//! |---|---|---|
//! | `deepseek` | top-k weights are not renormalised | every routed token |
//! | `bailingmoe` | `leading_dense_block_count` is inert | layer 0 fails to load |
//! | `seed_oss` | pre-FFN norm lives in `post_attention_norm` | a whole RMSNorm on the wrong side of a residual |
//! | `maincoder` | QK norm AFTER RoPE | every layer's attention scores |
//! | `hunyuan-moe` | QK norm AFTER RoPE | every layer's attention scores |
//! | `hunyuan-dense` | NTK-alpha RoPE base rescale (and the same QK-norm order) | every position, on every layer |
//! | `ernie4_5-moe` | `interleave_moe_layer_step`, landed as a REFUSAL | see below |
//!
//! **One of the seven arms could not be implemented, and that is the
//! finding rather than a shortfall.** `ernie4_5-moe`'s interleave step
//! is real in llama.cpp's GRAPH (`ernie4-5-moe.cpp:64`) and absent from
//! llama.cpp's own TENSOR LOADER (`ernie4-5.cpp:49`), so the two agree
//! only where the step changes nothing and a genuinely interleaved
//! checkpoint cannot be loaded by llama.cpp at all -- measured, on the
//! two-step fixture this suite ships. The step every real checkpoint
//! carries is audited here; anything else is refused by name and the
//! refusal says why.
//!
//! **Where the numbers come from.** Each `GOLDEN` array below was
//! produced by running **llama.cpp's own graph** for that architecture
//! over the same fixture file, through `scripts/gptoss_reference_logits.cpp`
//! linked against a real `libllama`. Not by re-reading a spec, and not
//! by ferrox checking itself. That is the bar `AUDITED_GENERIC_GQA`
//! sets, and it is why these five moved off the refusal list rather
//! than merely having their verdicts reworded.
//!
//! Every fixture is a 2- or 3-layer synthetic checkpoint from
//! `scripts/make_<arch>_fixture.py`, fixed seed, byte-stable. The
//! scripts' docstrings carry the per-architecture reading of
//! `.scratch/llama.cpp/src/models/*.cpp` that each shape choice comes
//! from.
//!
//! Regenerating (both halves must be redone together if a fixture
//! changes):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_seed_oss_fixture.py \
//!     crates/ferrox-models/tests/fixtures/seed_oss_tiny.gguf
//! clang++ -std=c++17 -O2 scripts/gptoss_reference_logits.cpp \
//!     -I$LLAMA/include -I$LLAMA/ggml/include -L$BUILD/bin -lllama \
//!     -Wl,-rpath,$BUILD/bin -o /tmp/ref_logits
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/seed_oss_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches as caches, graph_fixture_path as fixture,
    load_graph_fixture as load, worst_vs, GRAPH_PROMPT as PROMPT,
};
use ferrox_gguf::TensorSource;
use ferrox_models::capability::QkNormStyle;
use ferrox_models::{Decoder, ModelConfig};

// --- deepseek (V1) -------------------------------------------------
//
// The arm: `src/models/deepseek.cpp:145-155` passes `norm_w = false` to
// `build_moe_ffn`, and `conversion/deepseek.py`'s `DeepseekModel` never
// writes `{arch}.expert_weights_norm` (only `DeepseekV2Model` does, at
// :354), so no real `deepseek` GGUF carries the key. ferrox has to get
// the answer from `NO_TOPK_RENORMALIZE_ARCHITECTURES`, and the fixture
// has no such key for exactly that reason.

const DEEPSEEK_GOLDEN: [f32; 48] = [
    0.048226774,
    0.2819079,
    -0.19458595,
    0.051145732,
    0.5227392,
    0.23523334,
    0.18219762,
    -0.5215794,
    0.100488976,
    -0.40474808,
    0.028783947,
    0.29588076,
    -0.18639557,
    -0.04201878,
    0.096413225,
    -0.13163471,
    -0.024807326,
    0.3165786,
    0.024499238,
    -0.035591282,
    0.008794859,
    -0.24559931,
    0.21297875,
    -0.2290324,
    -0.40852088,
    -0.17818648,
    -0.29124618,
    0.73868686,
    0.09183925,
    -0.05146555,
    -0.13013509,
    -0.12608205,
    -0.019688487,
    0.016693514,
    0.026398227,
    0.28899318,
    -0.39421725,
    0.036387824,
    0.15263395,
    0.0761507,
    -0.15402263,
    -0.064116284,
    0.15764138,
    -0.19088429,
    0.3309419,
    0.21663469,
    -0.5842512,
    0.23919708,
];

#[test]
fn deepseek_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("deepseek", &DEEPSEEK_GOLDEN);
}

/// What the loader decided, so a regression in a piece the logits alone
/// would not name still names itself.
#[test]
fn the_loader_reads_deepseeks_routing_and_its_leading_dense_layers() {
    let d = load("deepseek");
    // The arm. `false` here is NOT read from the file -- the fixture
    // carries no `expert_weights_norm` -- so this pins the
    // architecture-name fallback, which is the whole mechanism.
    assert!(
        !d.config.moe.norm_topk_prob,
        "deepseek must not renormalise the selected experts' weights"
    );
    // Leading dense IS honoured here, unlike `bailingmoe` below.
    assert!(d.config.layer_is_dense(0));
    assert!(!d.config.layer_is_dense(1));
    assert_eq!(d.config.moe.n_shared_experts, 2);
    assert_eq!(d.config.moe.expert_ffn_dim, 12);
    assert_eq!(d.config.rope_layout, ferrox_models::RopeLayout::Norm);
}

/// Renormalising the top-k weights is a visible error, not a rounding
/// one.
///
/// Without this the golden comparison would pass just as well with the
/// flag inverted for a fixture whose two selected experts happened to
/// have near-equal weights, and the arm would be untested.
#[test]
fn renormalising_deepseeks_top_k_weights_diverges_from_llama_cpp() {
    let path = fixture("deepseek");
    let file = ferrox_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.moe.norm_topk_prob = true;
    let d = Decoder::from_gguf(&path, config).expect("loads");
    let mut kv = caches(&d);
    let worst = worst_vs(&d.forward_batch_last(&PROMPT, 0, &mut kv), &DEEPSEEK_GOLDEN);
    assert!(
        worst > 1e-3,
        "renormalising changed the logits by only {worst}; the fixture cannot see this arm"
    );
}

// --- bailingmoe ----------------------------------------------------
//
// The arm: `src/models/bailingmoe.cpp:5` reads
// `LLM_KV_LEADING_DENSE_BLOCK_COUNT` and then nothing branches on it --
// :39-54 creates the expert and shared-expert tensors unconditionally
// for every layer, and the graph (:119-152) has no dense path. The
// fixture sets the key to 1 and ships no dense FFN on layer 0, so a
// decoder that honours the key dies on a missing tensor.

const BAILINGMOE_GOLDEN: [f32; 48] = [
    0.28660098,
    0.03384116,
    -0.24319153,
    0.089819975,
    -0.3759065,
    0.30608493,
    0.16745271,
    0.23978557,
    -0.04340898,
    0.20000786,
    0.082406804,
    0.014814936,
    -0.19221006,
    0.23547158,
    -0.029625641,
    -0.31056488,
    -0.37496945,
    0.0300856,
    -0.21707577,
    -0.012679767,
    -0.027634194,
    -0.36596966,
    0.040408455,
    0.18680914,
    -0.008012157,
    -0.09278015,
    0.1771502,
    0.5291099,
    -0.022168158,
    -0.08451706,
    -0.060037654,
    -0.12324926,
    0.34276655,
    0.5576784,
    -0.19961314,
    0.023494173,
    0.16179755,
    -0.1449597,
    -0.11124265,
    0.18491973,
    -0.123816594,
    0.031925693,
    -0.6112474,
    0.08633105,
    0.23084326,
    -0.12105487,
    -0.11914092,
    0.23888548,
];

#[test]
fn bailingmoe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("bailingmoe", &BAILINGMOE_GOLDEN);
}

#[test]
fn bailingmoe_ignores_the_leading_dense_key_its_file_carries() {
    let path = fixture("bailingmoe");
    let file = ferrox_gguf::GgufFile::open(&path).expect("opens");
    // The key really is in the file -- otherwise this test proves
    // nothing about ignoring it.
    assert_eq!(
        file.metadata_u64("bailingmoe.leading_dense_block_count"),
        Some(1),
        "the fixture must carry the key it is ignoring"
    );
    let d = load("bailingmoe");
    assert!(
        !d.config.layer_is_dense(0),
        "layer 0 must be MoE despite leading_dense_block_count = 1"
    );
    // `expert_weights_norm` IS written by conversion/bailingmoe.py:31,
    // and the fixture sets it false -- the opposite of ferrox's
    // architecture-name default -- so this pins that the file wins.
    assert!(!d.config.moe.norm_topk_prob);
    assert_eq!(d.config.moe.expert_weights_scale, 2.5);
    assert_eq!(d.config.rope_layout, ferrox_models::RopeLayout::Norm);
}

// --- seed_oss ------------------------------------------------------
//
// The arm: `src/models/seed-oss.cpp:36-37` creates `attn_norm` and
// `attn_post_norm` and no `ffn_norm`, and :113-115 norms `ffn_inp` --
// the post-attention residual -- with `attn_post_norm`. That is
// gpt-oss's slot, and it used to be reachable only through an
// `arch == "gpt-oss"` flag that also gated gpt-oss's attention sinks.

const SEED_OSS_GOLDEN: [f32; 48] = [
    0.4555686,
    0.08081946,
    -0.8877717,
    -0.008781537,
    0.016691634,
    -0.14418256,
    0.1414922,
    0.593925,
    0.28484803,
    -0.011401828,
    0.1034261,
    0.2122338,
    -0.26438826,
    -0.50978255,
    0.041276924,
    0.12567355,
    0.17167503,
    -0.3315112,
    -0.26611036,
    -0.0082461815,
    0.3068763,
    0.11299762,
    -0.4340123,
    -0.24792485,
    0.18786612,
    0.1469672,
    0.0567582,
    -0.2559087,
    0.3925688,
    0.43699247,
    -0.62407005,
    -0.16498223,
    0.049159832,
    0.4050939,
    0.041347966,
    -0.49512297,
    0.23467577,
    0.22438808,
    -0.14997265,
    0.5364686,
    0.4941529,
    -0.17830561,
    -0.08436005,
    0.09402853,
    1.8876046e-05,
    0.8314706,
    0.34555963,
    -0.27246982,
];

#[test]
fn seed_oss_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("seed_oss", &SEED_OSS_GOLDEN);
}

/// The norm slot was widened without widening gpt-oss's extra tensors
/// with it.
///
/// One flag standing for two facts is this repo's dominant bug shape.
/// If `seed_oss` had been admitted by widening `is_gpt_oss`, it would
/// have been handed attention sinks it does not have -- and the
/// attention-sink path also refuses Metal, so the mistake would have
/// been a silent performance cliff as well as wrong math.
#[test]
fn seed_oss_takes_the_norm_slot_without_taking_gpt_osss_other_tensors() {
    let d = load("seed_oss");
    assert!(
        d.gpt_oss.is_none(),
        "seed_oss has no attention sinks, no router bias and no SwiGLU clamp"
    );
    for (il, layer) in d.layers.iter().enumerate() {
        // `post_attention_norm` was consumed as the PRE-FFN norm, so
        // nothing is left in the Gemma post-attention slot.
        assert!(
            layer.attn.post_attn_norm.is_none(),
            "blk.{il}: post_attention_norm must be the pre-FFN norm here"
        );
        assert!(
            layer
                .moe
                .norm_weight
                .rms_weights()
                .is_some_and(|w| w.iter().any(|x| *x != 0.0)),
            "blk.{il}: the pre-FFN norm must have been loaded from somewhere"
        );
    }
    // head_dim comes from `seed_oss.attention.key_length`; n_embd/n_head
    // would be 6.
    assert_eq!(d.config.head_dim, 8);
    assert_eq!(d.config.hidden_dim, 24);
    assert_eq!(d.config.rope_layout, ferrox_models::RopeLayout::Neox);
}

// --- maincoder and hunyuan-moe: QK norm after RoPE ------------------
//
// The arm: both rotate Q and K and only then norm them
// (`maincoder.cpp:78-95`, `hunyuan-moe.cpp:93-118`), where every
// previously-audited architecture norms first
// (`qwen3moe.cpp:99,108`). Both fixtures carry QK-norm weights centred
// near 1.5 rather than near 1.0, so the two orders are far apart.

const MAINCODER_GOLDEN: [f32; 48] = [
    -0.11138739,
    0.11824765,
    0.13920553,
    -0.30956966,
    0.13018379,
    0.10043175,
    0.3886379,
    -0.19205174,
    0.30031204,
    0.10036778,
    -0.10180198,
    0.09747762,
    -0.28050625,
    -0.04379492,
    -0.2667072,
    0.2032988,
    0.054047327,
    0.11231842,
    -0.32600948,
    -0.05464149,
    -0.13469744,
    0.0149400495,
    -0.20797788,
    -0.123298734,
    0.09001486,
    0.29425055,
    0.012470618,
    -0.61496115,
    0.38163024,
    0.034936484,
    -0.5258511,
    -0.106218845,
    0.012823265,
    0.079234354,
    0.11675485,
    -0.15945944,
    0.23372354,
    -0.053463854,
    0.36375925,
    -0.21172805,
    0.010290567,
    0.08845574,
    -0.13449258,
    -0.42340145,
    0.36418775,
    0.06322843,
    0.3355647,
    0.44385982,
];

const HUNYUAN_MOE_GOLDEN: [f32; 48] = [
    0.35964164,
    -0.03193312,
    0.08860654,
    0.07882613,
    -0.09411318,
    -0.25228012,
    0.10206525,
    0.0019554244,
    0.22584079,
    -0.18803002,
    0.25445765,
    0.24844477,
    -0.18049283,
    -0.004734317,
    0.07832132,
    0.04426696,
    -0.24596754,
    -0.012545568,
    -0.10501391,
    -0.33010307,
    0.11569242,
    0.02127038,
    -0.1883502,
    -0.014134302,
    0.25798446,
    -0.28575167,
    -0.39197293,
    -0.19562551,
    0.11669691,
    -0.18544069,
    0.41355348,
    0.084669836,
    -0.25159535,
    -0.14645867,
    -0.09086223,
    0.23901,
    0.1571095,
    -0.12348597,
    -0.036701947,
    -0.28670818,
    0.08595218,
    -0.08441264,
    -0.04356384,
    0.16307725,
    -0.07740431,
    -0.023180101,
    0.23391853,
    -0.14467773,
];

#[test]
fn maincoder_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("maincoder", &MAINCODER_GOLDEN);
}

#[test]
fn hunyuan_moe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("hunyuan_moe", &HUNYUAN_MOE_GOLDEN);
}

#[test]
fn both_post_rope_architectures_resolve_per_head_qk_norm_and_the_ordering_flag() {
    for (name, hidden, head_dim) in [("maincoder", 24, 8), ("hunyuan_moe", 24, 8)] {
        let d = load(name);
        assert!(
            d.qk_norm_after_rope,
            "{name} norms Q and K after RoPE and the loader must say so"
        );
        assert_eq!(d.config.qk_norm_style, QkNormStyle::PerHead, "{name}");
        assert_eq!(d.config.hidden_dim, hidden, "{name}");
        assert_eq!(d.config.head_dim, head_dim, "{name}");
        for (il, layer) in d.layers.iter().enumerate() {
            assert_eq!(
                layer.attn.q_norm.as_ref().map(Vec::len),
                Some(head_dim),
                "{name} blk.{il}: per-head Q norm"
            );
            assert_eq!(
                layer.attn.k_norm.as_ref().map(Vec::len),
                Some(head_dim),
                "{name} blk.{il}: per-head K norm"
            );
        }
    }
}

/// Norming on the wrong side of RoPE is a large divergence, on both
/// architectures.
///
/// This is the sabotage that makes the two golden comparisons above
/// mean something. Without it, a fixture whose QK-norm weights happened
/// to be near 1.0 would agree with llama.cpp under either order and the
/// arm would be untested.
#[test]
fn norming_before_rope_instead_of_after_diverges_from_llama_cpp() {
    for (name, golden) in [
        ("maincoder", &MAINCODER_GOLDEN),
        ("hunyuan_moe", &HUNYUAN_MOE_GOLDEN),
    ] {
        let mut d = load(name);
        assert!(d.qk_norm_after_rope);
        d.qk_norm_after_rope = false;
        let mut kv = caches(&d);
        let worst = worst_vs(&d.forward_batch_last(&PROMPT, 0, &mut kv), golden);
        assert!(
            worst > 1e-2,
            "{name}: swapping the QK-norm order moved the logits by only {worst}; \
             the fixture cannot see this arm"
        );
    }
}

// --- hunyuan-dense: the NTK-alpha RoPE base rescale -----------------
//
// `hunyuan-dense` has no graph of its own: `src/models/models.h:1830-1834`
// derives `llama_model_hunyuan_dense` from `llama_model_hunyuan_vl` and
// reuses its hparams, its tensors and its graph, so the file to read is
// `src/models/hunyuan-vl.cpp`. It had two blockers and both are closed
// here.
//
// The arm: `:8-12` rescales the trained RoPE base by
// `alpha^(head_dim / (head_dim - 2))` when `{arch}.rope.scaling.alpha`
// is positive. That key is read for EVERY architecture at
// `llama-model.cpp:1186` and applied by exactly two graphs, which is why
// ferrox's `rope_ntk_alpha` is a named list and not a generic rule.
//
// The second half, per-head QK norm applied AFTER RoPE (`:56-66` rope,
// then `:73-81` norm), was already implemented as
// `Decoder::qk_norm_after_rope` for `maincoder` and `hunyuan-moe`; this
// row only had to be added to the list.
//
// WHAT THE VERDICT GOT WRONG, and the fixture is what found it: the
// triage cited `conversion/hunyuan.py:356` as the converter line that
// writes `{arch}.rope.scaling.alpha` for this architecture. That line is
// in `HunyuanVLTextModel`, whose `model_arch` is `HUNYUAN_VL` -- a
// different GGUF architecture string and a different, still-refusing
// row. The `HUNYUAN_DENSE` converter is `HunYuanModel` at :254-281, and
// it does the same arithmetic in PYTHON (`scaled_base = base * (alpha **
// (dim / (dim - 2)))`, :270) and writes the already-scaled value through
// `add_rope_freq_base`, with no alpha key at all. So on a converted file
// llama.cpp's :8-12 is a no-op. The fixture writes the key explicitly so
// that the arm ferrox implements is the arm the reference runs, and
// libllama prints `freq_base_train = 92100.8` for a file whose
// `rope.freq_base` is 500.

const HUNYUAN_DENSE_GOLDEN: [f32; 48] = [
    0.22020058,
    0.63418895,
    0.20907053,
    0.3790403,
    -0.11444236,
    0.057364255,
    -0.34231323,
    -0.072572984,
    -0.5012044,
    0.17969774,
    -0.45948866,
    0.09718554,
    0.3629007,
    -0.20113428,
    0.27587852,
    -0.35394314,
    0.1426361,
    -0.12607718,
    0.058810126,
    -0.47764817,
    0.17408267,
    0.19744661,
    0.21505915,
    0.19182259,
    -0.09484324,
    0.012222506,
    0.12370877,
    0.061753318,
    0.2311838,
    -0.23603764,
    -0.29653496,
    -0.43558064,
    0.8097959,
    -0.19447437,
    0.30741596,
    0.0015857695,
    -0.18803233,
    0.09101723,
    0.59113497,
    -0.22490542,
    -0.2511651,
    -0.1257552,
    -0.12590414,
    -0.119154006,
    -0.033712514,
    0.104955494,
    0.08142837,
    0.37710968,
];

#[test]
fn hunyuan_dense_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("hunyuan_dense", &HUNYUAN_DENSE_GOLDEN);
}

/// The rescale really ran, and it produced llama.cpp's own number.
///
/// `92100.8` is what libllama prints as `freq_base_train` for this
/// fixture, whose file says `rope.freq_base = 500` and
/// `rope.scaling.alpha = 50`. Asserting the resolved base rather than
/// only the logits means a regression names itself instead of arriving
/// as forty-eight wrong floats.
#[test]
fn hunyuan_dense_rotates_at_the_ntk_alpha_rescaled_base() {
    let path = fixture("hunyuan_dense");
    let file = ferrox_gguf::GgufFile::open(&path).expect("opens");
    assert_eq!(
        file.metadata_f32("hunyuan-dense.rope.freq_base"),
        Some(500.0),
        "the fixture must carry the UNSCALED base, or it pins nothing about the rescale"
    );
    assert_eq!(
        file.metadata_f32("hunyuan-dense.rope.scaling.alpha"),
        Some(50.0),
        "and it must carry the alpha, which no converter writes for this architecture"
    );
    let d = load("hunyuan_dense");
    let want = 500.0f32 * 50.0f32.powf(8.0 / 6.0);
    assert!(
        (d.config.rope_theta - want).abs() < 1e-1,
        "rope_theta resolved to {}, want {want} (libllama prints freq_base_train = 92100.8)",
        d.config.rope_theta
    );
    // The other half of the row, and the reason it was one arm rather
    // than two.
    assert!(d.qk_norm_after_rope, "hunyuan-vl.cpp:56-66 then :73-81");
    assert_eq!(d.config.qk_norm_style, QkNormStyle::PerHead);
    assert_eq!(d.config.rope_layout, ferrox_models::RopeLayout::Neox);
    assert!(
        d.config.attention_scale.is_none(),
        "hunyuan-vl.cpp:22 passes a literal 1/sqrt(n_embd_head), which the kernels apply"
    );
}

/// Skipping the rescale is a large divergence.
///
/// Without this the golden comparison would hold just as well against a
/// decoder that ignored the key, because RoPE at 500 and RoPE at 92100
/// differ only where the positions matter. It is why the fixture's
/// unscaled base is small and its Q/K are drawn wide.
#[test]
fn rotating_hunyuan_dense_at_the_unscaled_base_diverges_from_llama_cpp() {
    let path = fixture("hunyuan_dense");
    let file = ferrox_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.rope_theta = 500.0;
    let d = Decoder::from_gguf(&path, config).expect("loads");
    let mut kv = caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&PROMPT, 0, &mut kv),
        &HUNYUAN_DENSE_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "ignoring the NTK-alpha rescale moved the logits by only {worst}; the fixture \
         cannot see hunyuan-vl.cpp:8-12"
    );
}

/// And the ordering half is visible on this row too, not only on the
/// other two.
#[test]
fn norming_hunyuan_dense_before_rope_instead_of_after_diverges_from_llama_cpp() {
    let mut d = load("hunyuan_dense");
    assert!(d.qk_norm_after_rope);
    d.qk_norm_after_rope = false;
    let mut kv = caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&PROMPT, 0, &mut kv),
        &HUNYUAN_DENSE_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "swapping the QK-norm order moved the logits by only {worst}"
    );
}

// --- ernie4_5-moe: the interleave step, and why it is a refusal -----
//
// The arm was meant to be `{arch}.interleave_moe_layer_step`, so that
// `ModelConfig::layer_is_dense` matched `src/models/ernie4-5-moe.cpp:64`:
//
//     il >= n_layer_dense_lead && (il + 1) % n_moe_layer_step == 0
//
// Building the fixture is what changed the answer. llama.cpp's TENSOR
// LOADER has no step in it: `src/models/ernie4-5.cpp:49` creates
// `ffn_gate_inp`, `ffn_down_exps` and `ffn_up_exps` as REQUIRED for
// every layer at or past `n_layer_dense_lead`, and creates the dense
// `ffn_gate`/`ffn_up`/`ffn_down` for none of them. The loader and the
// graph therefore agree only where the modulo changes nothing, and a
// checkpoint whose interleave really interleaves cannot be loaded by
// llama.cpp at all. Measured, not reasoned: the two-step fixture beside
// this one makes libllama print
//
//     check_tensor_dims: tensor 'blk.2.ffn_gate_inp.weight' not found
//
// Both published ERNIE-4.5 MoE checkpoints (21B-A3B, 300B-A47B) carry a
// step of 1 -- `conversion/ernie.py:88` writes `moe_layer_interval`
// straight from the HF config -- at which point the rule collapses to
// the leading-dense prefix ferrox already implements. That is the file
// the golden values below come from; anything else is refused by name
// in `moe_interleave`.
//
// The rest of the row was read too: SOFTMAX routing, HARDCODED at :90 --
// this architecture is NOT sigmoid-routed, whatever the gap inventory
// said -- with `norm_w = true` (:88) and an optional `exp_probs_b`
// selection bias (ernie4-5.cpp:53). The fixture carries neither
// `expert_gating_func` nor `expert_weights_norm`, so both have to come
// out of ferrox's architecture-name defaults.

const ERNIE4_5_MOE_GOLDEN: [f32; 48] = [
    -0.08447625,
    -0.12077317,
    0.34613043,
    0.03877828,
    0.037789084,
    -0.18150453,
    -0.016461063,
    0.123786785,
    0.018926805,
    0.010408605,
    -5.1606377e-4,
    0.20932025,
    -0.09644128,
    -0.09167313,
    -0.047137674,
    0.32158756,
    0.42049405,
    -0.10163629,
    0.24967228,
    -0.17491005,
    -0.23651567,
    0.061103027,
    -0.025830623,
    0.30918807,
    -0.18258056,
    -0.32253858,
    -0.04022187,
    -0.07561384,
    -0.22288801,
    0.2532389,
    0.3253422,
    0.1982599,
    0.1800408,
    -0.0014230456,
    -0.13664317,
    -0.09072188,
    -0.19590256,
    -0.22582309,
    -0.062144432,
    -0.087381646,
    -0.33621082,
    0.20399752,
    0.18209141,
    -0.08127112,
    0.20059398,
    -0.23522364,
    -1.6790349e-5,
    0.016684819,
];

#[test]
fn ernie4_5_moe_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match("ernie4_5_moe", &ERNIE4_5_MOE_GOLDEN);
}

/// What the loader decided about the row's own facts.
///
/// The interleave step is in the file at 1, the leading-dense prefix is
/// honoured (unlike `bailingmoe`, which reads the same kind of key and
/// ignores it), and the routing resolved to llama.cpp's hardcoded pair.
#[test]
fn the_loader_reads_ernie_moes_step_its_dense_prefix_and_its_routing() {
    let path = fixture("ernie4_5_moe");
    let file = ferrox_gguf::GgufFile::open(&path).expect("opens");
    assert_eq!(
        file.metadata_u64("ernie4_5-moe.interleave_moe_layer_step"),
        Some(1),
        "the fixture must carry the REQUIRED key (ernie4-5.cpp:11)"
    );
    // Neither routing key is in the file, so both of these come from
    // ferrox's architecture-name defaults and this is what pins them.
    assert!(file
        .metadata_u64("ernie4_5-moe.expert_gating_func")
        .is_none());
    assert!(file
        .metadata_bool("ernie4_5-moe.expert_weights_norm")
        .is_none());

    let d = load("ernie4_5_moe");
    assert!(d.config.layer_is_dense(0), "leading_dense_block_count = 1");
    assert!(!d.config.layer_is_dense(1));
    assert!(!d.config.layer_is_dense(2));
    assert_eq!(
        d.config.moe.gating,
        ferrox_moe::GatingFunction::Softmax,
        "ernie4-5-moe.cpp:90 hardcodes LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX"
    );
    assert!(
        d.config.moe.norm_topk_prob,
        "ernie4-5-moe.cpp:88 passes norm_w = true"
    );
    assert_eq!(d.config.moe.n_experts, 4);
    assert_eq!(d.config.moe.n_experts_active, 2);
    assert_eq!(d.config.moe.n_shared_experts, 1);
    assert_eq!(d.config.rope_layout, ferrox_models::RopeLayout::Norm);
    // head_dim comes from `attention.key_length`; n_embd/n_head is 6.
    assert_eq!(d.config.head_dim, 8);
}

/// Routing this row through sigmoid instead of softmax is a large
/// divergence.
///
/// The gap inventory listed `ernie4_5-moe` beside `bailingmoe2` as
/// "sigmoid-routed MoE with `ffn_exp_probs_b` router bias", and only the
/// second half is true. Without this test the golden comparison would
/// rest on a default nothing had checked.
#[test]
fn routing_ernie_moe_through_sigmoid_instead_of_softmax_diverges_from_llama_cpp() {
    let path = fixture("ernie4_5_moe");
    let file = ferrox_gguf::GgufFile::open(&path).expect("opens");
    let mut config = ModelConfig::from_gguf(&file).expect("parses");
    config.moe.gating = ferrox_moe::GatingFunction::Sigmoid;
    let d = Decoder::from_gguf(&path, config).expect("loads");
    let mut kv = caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&PROMPT, 0, &mut kv),
        &ERNIE4_5_MOE_GOLDEN,
    );
    assert!(
        worst > 1e-2,
        "sigmoid routing moved the logits by only {worst}; the fixture cannot see the \
         gating function"
    );
}

/// The refusal fires, on a file that exists.
///
/// This is the half that stops `moe_interleave` from being a gate that
/// cannot fire. The second fixture is written exactly the way
/// `conversion/ernie.py` would write a two-step checkpoint -- dense
/// `blk.2.ffn_gate.weight`, no expert tensors on that layer -- and
/// libllama refuses it with `check_tensor_dims: tensor
/// 'blk.2.ffn_gate_inp.weight' not found`, because its loader creates
/// the expert tensors for every layer past the dense prefix regardless
/// of the step. ferrox refuses it earlier and says why.
#[test]
fn a_two_step_ernie_moe_checkpoint_is_refused_by_name() {
    let path = fixture("ernie4_5_moe_step2");
    let file = ferrox_gguf::GgufFile::open(&path).expect("opens");
    assert_eq!(
        file.metadata_u64("ernie4_5-moe.interleave_moe_layer_step"),
        Some(2),
        "the refusal fixture must actually declare the step it is refused for"
    );
    // And it really is shaped like a converted two-step checkpoint: the
    // interleaved dense layer stores dense FFN tensors.
    assert!(file.find_tensor("blk.2.ffn_gate.weight").is_some());
    assert!(file.find_tensor("blk.2.ffn_gate_inp.weight").is_none());

    let err = ModelConfig::from_gguf(&file).expect_err("must refuse");
    let msg = err.to_string();
    assert!(msg.contains("interleave_moe_layer_step"), "{msg}");
    assert!(msg.contains("ernie4-5.cpp:49"), "{msg}");
    assert!(msg.contains("cannot be loaded by llama.cpp"), "{msg}");
}
