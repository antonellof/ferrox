//! Which CPU scheduler one operation's parallel regions run on, and the
//! work quantity that decides it.
//!
//! `FERROX_CPU_POOL=spin` is **+123% at 3B and +87% at 8B** on aarch64
//! and takes decode past llama.cpp; it is **-37% at 135M** on the same
//! host (#27, and the aarch64 table in
//! `docs/plans/cpu-cuda-parity.md`). A default that is right at 3B is
//! wrong at 135M and vice versa, so the switch cannot flip and cannot
//! stay: it needs a rule, and the rule has to be about the operation
//! rather than the process. That is step 3 of
//! `docs/plans/cpu-cuda-parity.md`.
//!
//! [`backend`] is that rule, and it is the ONLY place either scheduler
//! is chosen. Every helper in [`crate::par`] calls it, so no call site
//! can grow its own opinion: `weight_matrix` had four spellings of one
//! GPU-router eligibility test and they drifted four ways, which is the
//! failure this module exists to make impossible.
//!
//! # What deciding per operation costs, and has not been measured
//!
//! Deciding per operation means a single token can use both schedulers:
//! the FFN projections over the crossover go to the pool while
//! attention, the norms and the narrow projections fork. Both sets of
//! workers are then alive at once, and the pool workers spin for
//! `FERROX_CPU_POOL_SPIN_US` before parking. Whether that costs
//! anything is exactly the question a sweep on a quiet host answers and
//! reading this file does not. `FERROX_CPU_POOL=rayon` is the revert,
//! and it restores the previous behaviour exactly rather than
//! approximately.

use std::cell::Cell;

use super::Backend;

/// Multiply-accumulates below which one operation's parallel regions are
/// worth less than the persistent pool's wake-up, so they fork with
/// rayon instead.
///
/// **This value is BRACKETED by measurement, not measured.** What is
/// measured (#27, `docs/CONFIG.md`, quiet rented hosts, 2026-09-04) is
/// per MODEL, not per operation:
///
/// | host | 135M | 3B | 8B |
/// |---|---|---|---|
/// | 20-core Cortex-A725 (aarch64) | **-37%** | +123% | +87% |
/// | 10-core Xeon E5-2630 v4 (x86) | +49% | +23% | +15% |
///
/// Turning that into a per-operation number is arithmetic over the
/// shapes, not a sweep, and nobody has run the sweep:
///
/// | model | widest decode matvec | narrowest |
/// |---|---|---|
/// | SmolLM2-135M (576 / 1536) | 576 x 1536 = 0.88M | 576 x 576 = 0.33M |
/// | Llama-3.2-3B (3072 / 8192) | 3072 x 8192 = 25.2M | 3072 x 1024 = 3.1M |
/// | Llama-3.1-8B (4096 / 14336) | 4096 x 14336 = 58.7M | 4096 x 1024 = 4.2M |
///
/// So every operation in the model where the pool LOST is under 0.9M and
/// every operation in the models where it WON is over 3.1M. `1 << 21`
/// (2.1M) is the middle of that bracket in log space. The true crossover
/// is somewhere in `0.9M ..= 3.1M` and this constant is a guess inside
/// it; the exit criterion in `docs/plans/cpu-cuda-parity.md` asks for a
/// sweep on aarch64 AND x86, and that sweep is still owed.
///
/// One constant, both architectures. Above it the pool won on both;
/// below it the pool lost on aarch64 and WON on x86 (+49% at 135M), so
/// forking below the crossover leaves a measured x86 win unclaimed.
/// That is deliberate: an architecture-conditional default is what
/// `FERROX_CPU_INT_DOT` was, and it cost x86 between 4x and 8.8x of
/// decode before anyone noticed
/// (`weight_matrix::int_dot_is_a_win_here`). Claiming the x86 small-
/// model win needs the sweep, not a second `cfg!`.
///
/// `FERROX_CPU_POOL` stays as the A/B override precisely because of
/// all that: `rayon` restores the pre-rule behaviour exactly and `spin`
/// forces the pool at every size, so bracketing the constant is one
/// environment variable rather than two builds.
pub const SPIN_MIN_OP_MACS: usize = 1 << 21;

/// What `FERROX_CPU_POOL` pins, if anything. Read once, cached.
///
/// - `spin` / `persistent` / `1` / `on` / `true` — the pool, at every size
/// - `rayon` / `0` / `off` / `false` — fork-join, at every size
/// - unset / anything else — no pin: [`backend`] decides per operation
pub(crate) fn pinned() -> Option<Backend> {
    use std::sync::OnceLock;
    static PINNED: OnceLock<Option<Backend>> = OnceLock::new();
    *PINNED.get_or_init(|| {
        match std::env::var("FERROX_CPU_POOL")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("spin" | "persistent" | "1" | "on" | "true") => Some(Backend::Spin),
            Some("rayon" | "0" | "off" | "false") => Some(Backend::Rayon),
            _ => None,
        }
    })
}

thread_local! {
    /// `(rows, macs per row)` of the operation whose parallel regions the
    /// calling thread is about to open, published by [`with_op_work`].
    ///
    /// One value, two readers: [`op_macs`] multiplies it for the pool
    /// rule and [`macs_per_row`] takes half of it for rayon's task floor.
    /// They are derived from the same publish rather than restated
    /// beside each other, so a call site cannot tell the rule one size
    /// and the floor another.
    ///
    /// Read only on the thread that published it, before a region opens,
    /// so it never has to propagate into a worker.
    static OP_WORK: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

/// Publish the shape of the operation `f` performs: `rows` output rows,
/// each dotting `macs_per_row` elements.
///
/// Restores the previous value, so nesting is safe.
pub fn with_op_work<R>(rows: usize, macs_per_row: usize, f: impl FnOnce() -> R) -> R {
    let prev = OP_WORK.with(|c| c.replace((rows, macs_per_row)));
    let out = f();
    OP_WORK.with(|c| c.set(prev));
    out
}

/// Total multiply-accumulates of the published operation; `0` when
/// nothing published one.
fn op_macs() -> usize {
    OP_WORK.with(|c| {
        let (rows, per_row) = c.get();
        rows.saturating_mul(per_row)
    })
}

/// Multiply-accumulates per output row of the published operation; `0`
/// when nothing published one. Read by
/// [`crate::weight_matrix::WeightMatrix::min_rows_per_task`].
pub fn macs_per_row() -> usize {
    OP_WORK.with(|c| c.get().1)
}

/// **The one predicate.** Which scheduler the parallel regions of the
/// operation now being set up run on.
///
/// An unpublished operation (`op_macs() == 0`) reads as small and forks
/// with rayon. That is deliberate: attention, sampling and the norm
/// kernels do not publish a matrix shape, and their regions are the ones
/// the -37% at 135M is made of.
pub fn backend() -> Backend {
    if let Some(pinned) = pinned() {
        return pinned;
    }
    if op_macs() >= SPIN_MIN_OP_MACS {
        Backend::Spin
    } else {
        Backend::Rayon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule is a function of the published work and nothing else, so
    /// it can be asserted without a process whose env var is set -- but
    /// only when nothing has pinned it, which is the CI case.
    fn unpinned() -> bool {
        pinned().is_none()
    }

    /// The two ends of the measured bracket land on different
    /// schedulers. This is the whole claim of #27 step 3: one operation
    /// the size of a 135M matvec and one the size of a 3B matvec must
    /// not get the same answer.
    ///
    /// Sabotage: return `Backend::Rayon` unconditionally from `backend`
    /// (or drop the `op_macs()` term) and the 3B row goes red.
    #[test]
    fn a_135m_matvec_and_a_3b_matvec_choose_different_schedulers() {
        if !unpinned() {
            return;
        }
        // SmolLM2-135M's widest decode matvec, 576 x 1536.
        with_op_work(1536, 576, || {
            assert_eq!(
                backend(),
                Backend::Rayon,
                "the pool is -37% at this size and must not be chosen"
            );
        });
        // Llama-3.2-3B's narrowest, 3072 x 1024.
        with_op_work(1024, 3072, || {
            assert_eq!(
                backend(),
                Backend::Spin,
                "the pool is +123% at this size and must be chosen"
            );
        });
    }

    /// An operation that published nothing is treated as small. Attention
    /// and sampling reach the helpers this way, and they are regions the
    /// pool measured WORSE on.
    #[test]
    fn work_that_was_never_published_forks_with_rayon() {
        if !unpinned() {
            return;
        }
        assert_eq!(op_macs(), 0);
        assert_eq!(backend(), Backend::Rayon);
    }

    /// The published shape has one source: the rule and rayon's task
    /// floor read the same cell. If they could be published separately,
    /// a call site could tell one a size the other never sees, which is
    /// this repo's dominant defect shape.
    #[test]
    fn the_rule_and_the_task_floor_read_one_published_shape() {
        with_op_work(64, 4096, || {
            assert_eq!(macs_per_row(), 4096);
            assert_eq!(op_macs(), 64 * 4096);
        });
        // And it is restored, so nesting cannot leak one op's size into
        // the next.
        assert_eq!(op_macs(), 0);
        assert_eq!(macs_per_row(), 0);
    }

    /// Nesting restores the outer shape rather than clearing it.
    #[test]
    fn a_nested_publish_restores_the_shape_it_replaced() {
        with_op_work(8, 8, || {
            with_op_work(4096, 4096, || {
                assert_eq!(op_macs(), 4096 * 4096);
            });
            assert_eq!(op_macs(), 64);
        });
    }

    /// A row count times a row width cannot be allowed to wrap into a
    /// small number and quietly pick the wrong scheduler.
    #[test]
    fn an_absurd_shape_saturates_instead_of_wrapping() {
        with_op_work(usize::MAX, usize::MAX, || {
            assert_eq!(op_macs(), usize::MAX);
        });
    }
}
