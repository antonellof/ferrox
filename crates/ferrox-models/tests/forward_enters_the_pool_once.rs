//! A forward pass enters the CPU worker pool ONCE, whatever its shape.
//!
//! # What shipped, and what this catches
//!
//! `ferrox_core::par`'s helpers funnel into `rayon::join` and the
//! `par_iter` bridges, and both of those cost very different things
//! depending on who calls them. From a rayon worker the caller runs one
//! half of the split itself and waits on a spin latch. From any other
//! thread the job is injected and the caller blocks on a pthread mutex
//! and condvar, doing no arithmetic at all while it sleeps.
//!
//! Every forward pass used to be driven from a thread rayon did not own,
//! so it paid the second kind once per parallel region: roughly five per
//! layer, so ~150 per token on a 30-layer model. Sampled on an M2 Pro
//! over SmolLM2-135M Q8_0 `tg128`, the driving thread spent 74% of the
//! token parked in `__psynch_cvwait` under rayon's `LockLatch`, while
//! the matvec kernel accounted for about a tenth of that window across
//! every thread in the process. The wait was mostly the round trip, not
//! the work.
//!
//! `decoder::entry` fixes that by wrapping each entry point in
//! `par::on_workers`, and this suite is the guard. It is an OPERATION
//! COUNT, not a throughput: `par::cold_regions` is a per-thread counter
//! of regions opened from outside the pool, so the assertion needs no
//! quiet host and no stopwatch, and it fails by a factor of a hundred
//! rather than by a few percent if the wrapper is removed.
//!
//! # Why every entry point and not just `forward_token`
//!
//! Because the failure mode is a NEW entry point, or an old one quietly
//! bypassing the wrapper. `decoder/entry.rs` carries the structural half
//! of that check (nothing outside it may declare `pub fn forward_*`);
//! this is the behavioural half, and it names each function so a
//! regression says which one.

use ferrox_core::cache::KvCache;
use ferrox_core::par;
use ferrox_models::config::ModelConfig;
use ferrox_models::decoder::Decoder;

/// A committed 2-layer synthetic checkpoint. Any of them would do: the
/// claim is about the number of pool entries, which does not depend on
/// the weights.
const FIXTURE: &str = "tests/fixtures/ferrox_real_test.gguf";

fn load() -> Decoder {
    let file = ferrox_gguf::GgufFile::open(FIXTURE).expect("fixture opens");
    let config = ModelConfig::from_gguf(&file).expect("fixture config parses");
    Decoder::from_gguf(FIXTURE, config).expect("fixture loads")
}

fn caches(decoder: &Decoder) -> Vec<KvCache> {
    decoder
        .layers
        .iter()
        .map(|_| KvCache::new(decoder.config.n_kv_heads, decoder.config.head_dim))
        .collect()
}

/// The cases `par::on_workers` declines to promote, and so the cases
/// where there is nothing here to assert.
///
/// `FERROX_CPU_POOL=spin` deliberately keeps the rayon pool unbuilt, a
/// GPU backend keeps the step on its own thread because the Metal stack
/// carries thread-local state across it, and a caller already on a
/// worker needs no promotion. Skip rather than assert something the
/// configuration makes untrue.
///
/// **Probed, not restated.** This used to spell out "not pinned to spin,
/// and the active backend is CPU" -- a second copy of `par::policy`'s
/// own rule, including its list of accepted env-var spellings, with
/// nothing making the two agree. `par::policy::pinned` is `pub(crate)`
/// to `ferrox-core`, so the copy was the only way to ask. It is not: an
/// EMPTY step through `on_workers` costs exactly one cold region when it
/// promotes and zero when it declines, which asks the real predicate and
/// cannot drift from it.
fn the_promotion_applies_here() -> bool {
    let before = par::cold_regions();
    par::on_workers(|| {});
    par::cold_regions() - before == 1
}

/// Runs `f` and reports how many parallel regions it opened from
/// outside the pool.
fn cold_entries(f: impl FnOnce()) -> u64 {
    let before = par::cold_regions();
    f();
    par::cold_regions() - before
}

/// The headline: one decode step, one entry into the pool.
///
/// Sabotage: drop the `par::on_workers` from `Decoder::forward_token`
/// in `decoder/entry.rs` and this reads dozens instead of one.
#[test]
fn one_decode_step_enters_the_pool_exactly_once() {
    if !the_promotion_applies_here() {
        return;
    }
    let decoder = load();
    let mut kv = caches(&decoder);
    // Warm up outside the measurement: the first call through a weight
    // matrix builds its repack cache, which opens regions of its own.
    let _ = decoder.forward_token(1, 0, &mut kv);

    let entries = cold_entries(|| {
        let _ = decoder.forward_token(2, 1, &mut kv);
    });
    assert_eq!(
        entries, 1,
        "a decode step must enter the pool once, not once per matvec"
    );
}

/// Every public forward entry point, one at a time, so a regression
/// names the one that broke.
///
/// Prefill and multi-sequence decode are wrapped for the same reason
/// single-token decode is; they are only cheaper to get wrong because
/// their regions are larger.
#[test]
fn every_forward_entry_point_enters_the_pool_exactly_once() {
    if !the_promotion_applies_here() {
        return;
    }
    let decoder = load();
    let prompt: Vec<usize> = vec![1, 2, 3, 4];

    // One untimed pass so no measurement below pays for a cold repack
    // cache or a first-touch allocation.
    let mut warm = caches(&decoder);
    let _ = decoder.forward_batch(&prompt, 0, &mut warm);
    let _ = decoder.forward_token(1, prompt.len(), &mut warm);

    let mut kv = caches(&decoder);
    assert_eq!(
        cold_entries(|| {
            let _ = decoder.forward_batch(&prompt, 0, &mut kv);
        }),
        1,
        "forward_batch"
    );

    let mut kv = caches(&decoder);
    assert_eq!(
        cold_entries(|| {
            let _ = decoder.forward_batch_with_hidden(&prompt, 0, &mut kv);
        }),
        1,
        "forward_batch_with_hidden"
    );

    let mut kv = caches(&decoder);
    assert_eq!(
        cold_entries(|| {
            let _ = decoder.forward_batch_last(&prompt, 0, &mut kv);
        }),
        1,
        "forward_batch_last"
    );

    let mut kv = caches(&decoder);
    assert_eq!(
        cold_entries(|| {
            let _ = decoder.forward_batch_last_host_kv(&prompt, 0, &mut kv);
        }),
        1,
        "forward_batch_last_host_kv"
    );

    let mut kv = caches(&decoder);
    assert_eq!(
        cold_entries(|| {
            let _ = decoder.forward_hidden_batch(&prompt, 0, &mut kv);
        }),
        1,
        "forward_hidden_batch"
    );

    let mut seqs = vec![caches(&decoder), caches(&decoder)];
    assert_eq!(
        cold_entries(|| {
            let _ = decoder.forward_multi_seq(&[1, 2], &[0, 0], &mut seqs);
        }),
        1,
        "forward_multi_seq"
    );
}

/// Without the wrapper a forward pass opens a region per matvec, so the
/// count scales with the model. This is the same measurement taken
/// against an unwrapped baseline: driving the private body directly is
/// not possible from a test, so instead assert the property that makes
/// the count meaningful at all -- that it does NOT grow when the same
/// step is taken twice in a row.
///
/// A wrapper that fired only on the first call, or a cache that opened
/// regions of its own on every token, would both show up here.
#[test]
fn the_pool_entry_count_does_not_grow_with_the_number_of_steps() {
    if !the_promotion_applies_here() {
        return;
    }
    let decoder = load();
    let mut kv = caches(&decoder);
    let _ = decoder.forward_token(1, 0, &mut kv);

    let one = cold_entries(|| {
        let _ = decoder.forward_token(2, 1, &mut kv);
    });
    let four = cold_entries(|| {
        for step in 2..6 {
            let _ = decoder.forward_token(2, step, &mut kv);
        }
    });
    assert_eq!(one, 1);
    assert_eq!(
        four, 4,
        "four steps must cost four entries, not four times N"
    );
}

/// The same property for a DEDICATED engine, on a real checkpoint.
///
/// #167 promoted `Decoder` alone, so every engine behind
/// `engine::Engine` -- Gemma-4, Kimi K3, GLM-5.2, DeepSeek-V4 Pro, the
/// MLA stack -- still opened a parallel region per matvec. Gemma-4 is
/// the one of them a real GGUF exists for on this machine, so it is the
/// one that can be asserted against weights rather than against a
/// synthetic stand-in; `engine/entry.rs` carries the checkpoint-free
/// half for the rest, which all share this one wrapper.
///
/// Sabotage: drop the `par::on_workers` from `Engine::forward_token` in
/// `engine/entry.rs` and this reads dozens instead of one.
#[test]
#[ignore = "needs models/gemma-4-E2B-it-Q4_K_M.gguf (or FERROX_TEST_GEMMA4_GGUF)"]
fn a_gemma4_decode_step_enters_the_pool_exactly_once() {
    use ferrox_models::engine::Engine;

    if !the_promotion_applies_here() {
        return;
    }
    let path = std::env::var("FERROX_TEST_GEMMA4_GGUF").unwrap_or_else(|_| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/gemma-4-E2B-it-Q4_K_M.gguf")
            .to_string_lossy()
            .into_owned()
    });
    if !std::path::Path::new(&path).exists() {
        eprintln!("skip: Gemma-4 GGUF missing at {path}");
        return;
    }
    let file = ferrox_gguf::GgufFile::open(&path).expect("open GGUF");
    let engine = ferrox_models::gemma4_gguf_loader::load_gemma4_engine(&file).expect("load");
    let mut state = Engine::new_state(&engine);

    // Warm up outside the measurement, for the same reason the generic
    // cases above do: the first pass through a weight matrix builds its
    // repack cache, which opens regions of its own.
    let _ = engine.forward_token(1, 0, &mut state);

    let entries = cold_entries(|| {
        let _ = engine.forward_token(2, 1, &mut state);
    });
    assert_eq!(
        entries, 1,
        "a Gemma-4 decode step must enter the pool once, not once per matvec"
    );
}

/// The encoder seam pays the same rule.
///
/// `TextEncoder::encode` is a forward pass through the same quantized
/// projections, just over a whole sequence at once instead of one token,
/// so it opened the same per-matmul cold region and nothing wrapped it.
/// One pass per request rather than one per token makes it a smaller
/// win, not a different rule.
///
/// Sabotage: drop the `par::on_workers` from `TextEncoder::encode` in
/// `encoder.rs` and this reads dozens instead of one.
#[test]
#[ignore = "needs models/bge-small-en-v1.5-q8_0.gguf (or FERROX_TEST_BERT_GGUF)"]
fn one_encoder_pass_enters_the_pool_exactly_once() {
    use ferrox_models::TextEncoder;

    if !the_promotion_applies_here() {
        return;
    }
    let path = std::env::var("FERROX_TEST_BERT_GGUF").unwrap_or_else(|_| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/bge-small-en-v1.5-q8_0.gguf")
            .to_string_lossy()
            .into_owned()
    });
    if !std::path::Path::new(&path).exists() {
        eprintln!("skip: BERT GGUF missing at {path}");
        return;
    }
    let encoder = ferrox_models::load_bert_encoder_from_path(&path).expect("load");
    let tokens: Vec<u32> = encoder.wrap_special(&[100, 200, 300, 400]);

    let _ = encoder.encode(&tokens, None).expect("warm-up pass");

    let entries = cold_entries(|| {
        let _ = encoder.encode(&tokens, None).expect("measured pass");
    });
    assert_eq!(
        entries, 1,
        "an encoder pass must enter the pool once, not once per matmul"
    );
}
