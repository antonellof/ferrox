//! MiniCPM, checked against llama.cpp itself.
//!
//! MiniCPM was never an UNAUDITED row. It was refused **by name**, and
//! the reason it had to be is the reason this suite has two fixtures.
//!
//! `src/models/minicpm.cpp:5-7` assigns three scalar multipliers before
//! it reads anything at all:
//!
//! ```text
//! hparams.f_embedding_scale = 12.0f;
//! hparams.f_residual_scale  = 1.4f / sqrtf(float(hparams.n_layer()));
//! hparams.f_logit_scale     = hparams.n_embd ? (256.0f / float(hparams.n_embd)) : 1.0f;
//! ```
//!
//! and only then (`:12-14`) lets the file override them, each with
//! `required = false`. So a MiniCPM export that declares NONE of the
//! three keys is still scaled by all three, and
//! `capability::unsupported_scaling_keys` -- which asks whether a key is
//! PRESENT -- has nothing to look at. That is a blind spot no gate over
//! the metadata can close, which is why the row was refused on its
//! architecture string until the arithmetic itself was implemented.
//!
//! **The graph is Granite's, verbatim.** `models.h:1594-1601` is
//! `using graph = llama_model_granite::graph` -- the same graph object,
//! not a similar one -- so the four places the multipliers land are the
//! four `tests/granite_family_graphs.rs` already names:
//! `llama-graph.cpp:2337-2342` for the embedding scale,
//! `granite.cpp:235-238` and `:288-292` for the residual scale on BOTH
//! branch outputs, and `granite.cpp:180` for `1.0f / f_logit_scale`.
//! `crate::scalar_multipliers` therefore gains a DEFAULTS field on the
//! table it already had, not a second implementation.
//!
//! **Two differences from Granite, both pinned below.**
//!
//! 1. MiniCPM never reads `{arch}.attention.scale` (`minicpm.cpp:3-24`
//!    contains no `LLM_KV_ATTENTION_SCALE`), so `f_attention_scale`
//!    keeps its `0.0f` and `granite.cpp:225` falls back to
//!    `1/sqrt(n_embd_head)`. Measured, not assumed: a third fixture
//!    declares `minicpm.attention.scale = 0.9` and llama.cpp's logits
//!    for it are byte-identical to the file without it. ferrox refuses
//!    that file rather than honouring a number its own reference
//!    discards.
//! 2. Granite's graph gates RoPE entirely on `hparams.rope_finetuned`
//!    (`granite.cpp:206`) and `granite.cpp:33-35` lets a file switch it
//!    off. `minicpm.cpp:17` pins it `true` with no key read at all, so
//!    that switch is unreachable here and `minicpm` is deliberately
//!    absent from `rope_finetuned::ROPE_GATED_ON_FINETUNED`.
//!
//! **Why two goldens.** The defaults have to be applied BEFORE the file
//! and not after. A hook that ran the other way round would agree with
//! llama.cpp on every file that omits the keys -- the fixture that
//! proves the hook exists -- and disagree on every file that carries
//! them, which is the subset newer MiniCPM exports are in. One fixture
//! cannot see that; two can, and
//! `applying_the_defaults_over_the_file_instead_of_under_it_diverges`
//! is the test that does.
//!
//! **Where the numbers come from.** Each `GOLDEN` array was produced by
//! running llama.cpp's own graph over the fixture through
//! `scripts/gptoss_reference_logits.cpp` linked against a real
//! `libllama` built from `.scratch/llama.cpp`. Not by re-reading a spec,
//! and not by ferrox checking itself.
//!
//! Regenerating (all three files and both goldens together):
//!
//! ```text
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_minicpm_fixture.py \
//!     crates/ferrox-models/tests/fixtures/minicpm_tiny.gguf
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_minicpm_fixture.py \
//!     crates/ferrox-models/tests/fixtures/minicpm_declared_tiny.gguf --declared
//! PYTHONPATH=$LLAMA/gguf-py python3 scripts/make_minicpm_fixture.py \
//!     crates/ferrox-models/tests/fixtures/minicpm_attention_scale_tiny.gguf \
//!     --attention-scale
//! clang++ -std=c++17 -O2 scripts/gptoss_reference_logits.cpp \
//!     -I$LLAMA/include -I$LLAMA/ggml/include -L$BUILD/bin -lllama \
//!     -Wl,-rpath,$BUILD/bin -o /tmp/ref_logits
//! /tmp/ref_logits crates/ferrox-models/tests/fixtures/minicpm_tiny.gguf 3 7 11 19 23 5
//! ```

mod common;
use common::{
    assert_all_three_paths_match, graph_caches, graph_fixture_path, load_graph_fixture, worst_vs,
    GRAPH_PROMPT,
};
use ferrox_models::{ModelConfig, RopeLayout};

/// The fixture that declares nothing. Its whole job is to be an
/// ordinary-looking file that llama.cpp still scales three ways.
const DEFAULTS: &str = "minicpm";

/// The same weights with all three keys written. Same seed, same tensor
/// draw order, so the two goldens differ because of the METADATA.
const DECLARED: &str = "minicpm_declared";

/// `minicpm.cpp:5-7`, restated in Rust.
///
/// Restating the reference's own formula is the point rather than the
/// hazard: this is the llama.cpp SIDE of the comparison, and writing
/// three literals instead would let a fixture regenerated at a different
/// shape leave these assertions checking the old model's numbers while
/// still passing.
///
/// Returns `(embedding, residual, logit)`. The fixture is EIGHT layers,
/// not the two every other file in this directory uses, and that is
/// deliberate: `1.4/sqrt(2)` is 0.99, within 1% of the identity, and a
/// two-layer fixture could not tell the residual default from no
/// residual scaling at all -- measured at 3.3e-4, against a comparison
/// tolerance of 1e-5. At eight layers it is 0.495 and halves every
/// branch output.
fn llama_cpp_defaults(n_layer: usize, n_embd: usize) -> (f32, f32, f32) {
    (12.0, 1.4 / (n_layer as f32).sqrt(), 256.0 / n_embd as f32)
}

/// What `--declared` writes, chosen to be nowhere near any of the three
/// above so a hook applied in the wrong order is obvious rather than
/// marginal.
const DECLARED_EMBEDDING_SCALE: f32 = 2.0;
const DECLARED_RESIDUAL_SCALE: f32 = 0.6;
const DECLARED_LOGIT_SCALE: f32 = 2.5;

// --- minicpm, declaring no scaling key at all -----------------------

const MINICPM_DEFAULTS_GOLDEN: [f32; 48] = [
    -0.008597257,
    -0.011095079,
    -0.026547605,
    -0.025521975,
    -0.03098327,
    0.013243265,
    -0.0023162654,
    0.006206151,
    0.0142753115,
    0.010215266,
    0.013999276,
    -0.0036124494,
    -0.00055763754,
    -0.07160489,
    -0.025898434,
    -0.03411136,
    0.015863288,
    -0.030759301,
    0.035909362,
    0.049672533,
    0.0075411154,
    0.015415633,
    -0.051887345,
    0.00047458056,
    0.029006887,
    0.008306274,
    -0.0019820798,
    0.0010504243,
    -0.0853497,
    -0.01804373,
    -0.013589482,
    0.024263829,
    -0.046636067,
    0.011425146,
    -0.020435521,
    0.05519549,
    -0.035814814,
    0.037738454,
    0.018463945,
    0.0057971966,
    0.021860935,
    0.08652196,
    -0.032314952,
    -0.010404468,
    -0.022446092,
    0.039129704,
    -0.04060948,
    -0.07748856,
];

// --- the same weights with all three keys declared ------------------

const MINICPM_DECLARED_GOLDEN: [f32; 48] = [
    -0.15453912,
    -0.19590382,
    -0.019465739,
    -0.0013245508,
    -0.11342161,
    0.052824784,
    0.06937712,
    -0.085987605,
    -0.033899862,
    -0.015564865,
    -0.1752615,
    0.048371807,
    0.034458317,
    -0.068987414,
    -0.12057119,
    -0.18926334,
    0.054208964,
    -0.20525026,
    0.20817463,
    0.40593767,
    0.072739355,
    0.17150024,
    -0.22188793,
    0.15136957,
    0.103799574,
    0.0071456432,
    -0.08653117,
    0.035089742,
    -0.3068919,
    0.05307231,
    0.062167622,
    0.16122876,
    -0.13801384,
    -0.1486874,
    -0.11996346,
    0.07449855,
    0.09576403,
    0.18009911,
    0.2924678,
    0.105862916,
    0.07801372,
    0.29716626,
    -0.17477413,
    -0.049518377,
    0.20069416,
    0.1271135,
    -0.0460695,
    -0.123234324,
];

fn golden(name: &str) -> &'static [f32] {
    match name {
        DEFAULTS => &MINICPM_DEFAULTS_GOLDEN,
        DECLARED => &MINICPM_DECLARED_GOLDEN,
        other => panic!("no golden logits for `{other}`"),
    }
}

/// The row itself: a MiniCPM file that declares nothing, run on all
/// three forward paths against llama.cpp's own logits.
#[test]
fn minicpm_declaring_no_key_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(DEFAULTS, &MINICPM_DEFAULTS_GOLDEN);
}

/// The same weights with all three keys written, which is what a newer
/// MiniCPM export looks like.
#[test]
fn minicpm_declaring_all_three_keys_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(DECLARED, &MINICPM_DECLARED_GOLDEN);
}

/// The defaults are computed from the MODEL SHAPE, and the file wins
/// where it speaks.
///
/// Structural rather than numeric, and it is the assertion that says
/// what the hook is FOR: `residual_scale` is not a constant, it is
/// `1.4/sqrt(n_layer)`, and `logit_scale` is `256/n_embd`. A hook that
/// hardcoded one model's values would pass every logit comparison in
/// this file and be wrong for every real checkpoint, all of which have
/// more than two layers.
#[test]
fn the_defaults_are_derived_from_the_model_shape_and_the_file_overrides_them() {
    let d = load_graph_fixture(DEFAULTS);
    assert_eq!(d.config.n_layers, 8);
    assert_eq!(d.config.hidden_dim, 24);
    let (embedding, residual, logit) = llama_cpp_defaults(8, 24);
    assert_eq!(d.config.embedding_scale, Some(embedding));
    assert_eq!(d.config.residual_scale, Some(residual));
    assert!(
        (residual - 1.0).abs() > 0.4,
        "a residual default within a rounding error of 1.0 would make the sabotage below \
         unable to see it; got {residual}"
    );
    // The config carries the multiplier, already inverted, because
    // `granite.cpp:180` DIVIDES. Getting that direction backwards is
    // invisible in a smoke test: the logits stay finite and stay in the
    // same order, and only the temperature of the distribution moves.
    assert_eq!(
        d.config.logit_multiplier,
        Some(1.0 / logit),
        "the logit scale is applied as a reciprocal"
    );
    // MiniCPM does not read `attention.scale`, so the kernels' own
    // `1/sqrt(head_dim)` must survive.
    assert_eq!(d.config.attention_scale, None);

    let d = load_graph_fixture(DECLARED);
    assert_eq!(d.config.embedding_scale, Some(DECLARED_EMBEDDING_SCALE));
    assert_eq!(d.config.residual_scale, Some(DECLARED_RESIDUAL_SCALE));
    assert_eq!(d.config.logit_multiplier, Some(1.0 / DECLARED_LOGIT_SCALE));
    assert_eq!(d.config.attention_scale, None);
}

/// Dropping any one of the three defaults diverges.
///
/// This is the sabotage that makes the first fixture worth its runtime.
/// The file declares no key, so a ferrox that ignored `MultiplierDefaults`
/// entirely would load it, run it at full speed and answer -- which is
/// exactly what it did before this row landed. Each multiplier is turned
/// off on its own, so the suite cannot pass on two of three.
#[test]
fn dropping_any_one_of_the_three_defaults_diverges_from_llama_cpp() {
    for which in ["embedding", "residual", "logit"] {
        let mut d = load_graph_fixture(DEFAULTS);
        match which {
            "embedding" => d.config.embedding_scale = None,
            "residual" => d.config.residual_scale = None,
            _ => d.config.logit_multiplier = None,
        }
        let mut kv = graph_caches(&d);
        let worst = worst_vs(
            &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
            golden(DEFAULTS),
        );
        assert!(
            worst > 1e-2,
            "dropping the default {which} multiplier moved the output by only {worst}; \
             this fixture cannot tell MiniCPM's defaults from their absence"
        );
    }
}

/// Applying the defaults OVER the file rather than UNDER it diverges.
///
/// llama.cpp's order is assignment, then the optional key read
/// (`minicpm.cpp:5-7` then `:12-14`). A hook that ran the other way
/// round -- defaults last, so they win -- would agree with llama.cpp on
/// the first fixture and disagree on every real newer export. Nothing
/// about the code makes the order obvious, so it is measured here on the
/// file that can see it.
#[test]
fn applying_the_defaults_over_the_file_instead_of_under_it_diverges() {
    let (embedding, residual, logit) = llama_cpp_defaults(8, 24);
    let mut d = load_graph_fixture(DECLARED);
    d.config.embedding_scale = Some(embedding);
    d.config.residual_scale = Some(residual);
    d.config.logit_multiplier = Some(1.0 / logit);
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        golden(DECLARED),
    );
    assert!(
        worst > 1e-2,
        "letting the defaults win over the file moved the output by only {worst}; the \
         `--declared` fixture cannot see the merge order"
    );
}

/// MiniCPM's RoPE is the consecutive-pairs variant, and the fixture can
/// see the other one.
///
/// `LLM_ARCH_MINICPM` is in `llama_model_rope_type`'s NORM group
/// (llama-model.cpp:2580). Nothing in a GGUF says which variant an
/// architecture uses, so this is a fact ferrox carries in a table, and a
/// table entry that is wrong rotates the wrong pairs of every Q and K
/// head of every layer and answers fluently.
#[test]
fn minicpm_ropes_consecutive_pairs_and_the_fixture_can_see_the_other_variant() {
    let d = load_graph_fixture(DEFAULTS);
    assert_eq!(d.config.rope_layout, RopeLayout::Norm);

    let mut d = load_graph_fixture(DEFAULTS);
    d.config.rope_layout = RopeLayout::Neox;
    let mut kv = graph_caches(&d);
    let worst = worst_vs(
        &d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv),
        golden(DEFAULTS),
    );
    assert!(
        worst > 1e-3,
        "rotating the NEOX pairs moved the output by only {worst}; the attention in this \
         fixture is too flat to see a positional bug"
    );
}

/// A `minicpm` file declaring `attention.scale` is REFUSED, and the
/// refusal is reachable.
///
/// llama.cpp loads such a file and ignores the key -- measured, not
/// assumed: `minicpm_attention_scale_tiny.gguf` declares 0.9 and
/// libllama's logits for it are identical to
/// `MINICPM_DEFAULTS_GOLDEN`, because `minicpm.cpp:3-24` never assigns
/// `f_attention_scale` from anything. ferrox stops instead of honouring
/// a number its own reference discards, and the file here is what proves
/// that gate can fire rather than reading as coverage. The suite would
/// be weaker without it: this repo has shipped a refusal keyed on a GGUF
/// spelling nothing writes.
#[test]
fn a_minicpm_file_declaring_an_attention_scale_is_refused() {
    let path = graph_fixture_path("minicpm_attention_scale");
    let file = ferrox_gguf::GgufFile::open(&path).expect("fixture opens");
    let err = ModelConfig::from_gguf(&file)
        .expect_err("minicpm does not read attention.scale; a file declaring it must stop");
    let msg = format!("{err}");
    assert!(
        msg.contains("minicpm.attention.scale"),
        "the refusal must name the key it refuses: {msg}"
    );
}

/// The derived refusal list covers exactly the one key MiniCPM does not
/// read.
///
/// `unsupported_scaling_keys` is the COMPLEMENT of
/// `scalar_multipliers::multiplier_support`, not a second hand-written
/// list, and this is the assertion that says so for this row: three keys
/// implemented means three keys gone from the list, and the fourth still
/// on it. Two hand-maintained lists is how this repo once refused a key
/// it implemented and implemented a key it refused.
#[test]
fn the_derived_refusal_list_holds_only_the_key_minicpm_does_not_read() {
    let keys: Vec<String> = ferrox_models::capability::unsupported_scaling_keys("minicpm")
        .into_iter()
        .map(|(k, _, _)| k)
        .collect();
    assert_eq!(
        keys,
        vec!["minicpm.attention.scale".to_string()],
        "{keys:?}"
    );
}
