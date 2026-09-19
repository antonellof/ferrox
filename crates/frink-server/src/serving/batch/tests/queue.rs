use super::*;

#[test]
fn queue_gate_admits_up_to_its_cap_and_frees_slots_on_release() {
    let gate = QueueGate::new(2);
    assert!(gate.try_reserve().is_ok());
    assert!(gate.try_reserve().is_ok());
    assert_eq!(gate.depth(), 2);
    assert_eq!(gate.try_reserve(), Err(2), "the refusal reports the depth");
    assert_eq!(gate.rejected(), 1);
    gate.release();
    assert_eq!(gate.depth(), 1);
    assert!(
        gate.try_reserve().is_ok(),
        "a released slot must be reusable"
    );
    assert_eq!(gate.depth(), 2);
}

/// The cap is only a cap if it holds under the exact condition it
/// exists for: many clients submitting at once. A check-then-act
/// gate ("read the depth, then increment") lets every thread that
/// read a value below the cap through, which is how a retry storm
/// gets past a limit that looks correct when read in isolation.
///
/// Repeated rounds because a lost race is probabilistic: one round
/// can get lucky, sixty-four rounds of thirty-two racing threads do
/// not.
#[test]
fn queue_gate_never_exceeds_its_cap_under_concurrent_submitters() {
    const THREADS: usize = 32;
    const CAP: usize = 4;
    for round in 0..64 {
        let gate = Arc::new(QueueGate::new(CAP));
        let barrier = Arc::new(Barrier::new(THREADS));
        let admitted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let gate = Arc::clone(&gate);
                let barrier = Arc::clone(&barrier);
                let admitted = Arc::clone(&admitted);
                thread::spawn(move || {
                    barrier.wait();
                    if gate.try_reserve().is_ok() {
                        admitted.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            admitted.load(Ordering::Relaxed),
            CAP,
            "round {round}: exactly the cap may be admitted"
        );
        assert_eq!(gate.depth(), CAP, "round {round}: depth matches admissions");
        assert_eq!(gate.rejected(), (THREADS - CAP) as u64);
    }
}

/// End to end: a full queue is refused with a typed error naming
/// the depth and the cap, and the refusal costs nothing -- no
/// prompt is queued, no reply channel is parked.
#[test]
fn a_full_queue_refuses_new_jobs_with_queue_full() {
    let decoder = tiny_decoder();
    // cap 0 is degenerate on purpose: it makes "the queue is full"
    // deterministic in a test, where a real cap would be drained by
    // the worker before a second submission could ever see it.
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            max_queue: 0,
            ..BatcherConfig::default()
        },
    );
    let err = batcher
        .generate(vec![1, 2, 3], greedy_params(4, 1), StopTokens::default())
        .expect_err("a full queue must refuse");
    assert!(
        matches!(err, DecodeError::QueueFull { queued: 0, cap: 0 }),
        "expected QueueFull, got {err:?}"
    );
    assert_eq!(err.retry_after_secs(), Some(1), "a queue drains; say so");
    let stats = batcher.stats();
    assert_eq!(stats.queue_rejected, 1);
    assert_eq!(stats.queue_depth, 0, "a refused job holds nothing");
}

/// The gate must not leak slots: a job that is accepted, queued,
/// dequeued and served leaves the queue empty again.
#[test]
fn queue_depth_returns_to_zero_after_a_served_request() {
    let decoder = tiny_decoder();
    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            max_queue: 1,
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );
    for _ in 0..3 {
        batcher
            .generate(vec![1, 2, 3], greedy_params(2, 1), StopTokens::default())
            .expect("a cap of 1 still serves requests one after another");
    }
    assert_eq!(batcher.stats().queue_depth, 0);
    assert_eq!(batcher.stats().queue_rejected, 0);
}
