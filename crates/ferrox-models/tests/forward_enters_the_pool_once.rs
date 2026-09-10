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

/// The two cases `par::on_workers` declines to promote, and so the two
/// cases where there is nothing here to assert.
///
/// `FERROX_CPU_POOL=spin` deliberately keeps the rayon pool unbuilt, and
/// a GPU backend keeps the step on its own thread because the Metal
/// stack carries thread-local state across it. Skip rather than assert
/// something the configuration makes untrue.
fn the_promotion_applies_here() -> bool {
    let pinned_to_spin = matches!(
        std::env::var("FERROX_CPU_POOL")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("spin" | "persistent" | "1" | "on" | "true")
    );
    !pinned_to_spin
        && ferrox_core::weight_matrix::active_backend()
            == ferrox_core::kernel_registry::Backend::Cpu
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
