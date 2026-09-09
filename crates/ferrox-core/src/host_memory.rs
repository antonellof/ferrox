//! How much memory the host actually has free, and whether a model fits.
//!
//! Lives here rather than in a binary because both `ferrox-cli` and
//! `ferrox-server` need the same answer, and two copies of a
//! platform probe drift.

/// Bytes a new allocation can reasonably expect to get.
///
/// On macOS this counts free plus INACTIVE pages, because inactive
/// pages are reclaimable and counting only free ones understates what
/// is available by many gigabytes on a machine that has been up for a
/// while. On Linux it is `MemAvailable`, which the kernel computes for
/// exactly this question. `None` when the platform will not say, which
/// callers must treat as "unknown", never as zero.
pub fn available_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("vm_stat").output().ok()?;
        if !out.status.success() {
            return None;
        }
        parse_vm_stat(&String::from_utf8_lossy(&out.stdout))
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        text.lines()
            .find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            .map(|kb| kb * 1024)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// Free plus inactive pages from `vm_stat` output, in bytes.
pub fn parse_vm_stat(text: &str) -> Option<u64> {
    let page = text
        .lines()
        .next()?
        .split("page size of ")
        .nth(1)?
        .split(' ')
        .next()?
        .parse::<u64>()
        .ok()?;
    let field = |name: &str| -> Option<u64> {
        text.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|v| v.trim().trim_end_matches('.').parse::<u64>().ok())
    };
    Some((field("Pages free")? + field("Pages inactive")?) * page)
}

/// What to do about a model whose weights may not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitPlan {
    /// Load everything resident. The fast path, and the default.
    Resident,
    /// Stream experts, keeping at most this many bytes of them cached.
    Stream { cache_bytes: u64 },
}

/// Decide whether streaming is needed, given what the model weighs and
/// what the host has.
///
/// The bar is deliberately not "does it fit exactly". A model that
/// fills memory to the last byte will thrash the page cache and leave
/// nothing for the KV cache, so `headroom_bytes` is subtracted first.
///
/// `available` of `None` means the platform would not say. That
/// resolves to `Resident`, NOT to streaming: guessing that a machine is
/// short on memory would silently put every user on the slow path on
/// any platform without a probe.
pub fn plan_for(
    weight_bytes: u64,
    available: Option<u64>,
    headroom_bytes: u64,
    min_cache_bytes: u64,
) -> FitPlan {
    let Some(available) = available else {
        return FitPlan::Resident;
    };
    let usable = available.saturating_sub(headroom_bytes);
    if weight_bytes <= usable {
        return FitPlan::Resident;
    }
    // It does not fit. Spend what is usable on the expert cache, but
    // never less than the floor: a cache too small to hold one decode
    // step's experts turns every acquire into a fresh read, which is
    // correct but pathologically slow.
    FitPlan::Stream {
        cache_bytes: usable.max(min_cache_bytes),
    }
}

/// Bytes to leave for everything that is not weights: the KV cache,
/// activations, and the pages of the checkpoint not yet touched.
///
/// Named here rather than at the call site because two consumers now
/// subtract it -- [`plan_for`] and [`derived_copy_budget`] -- and two
/// spellings of one headroom is this repo's dominant defect shape. Both
/// still take it as a parameter so a test can vary it.
pub const FIT_HEADROOM_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// One in this many usable bytes may be spent on retained *derived*
/// copies of weights that are already resident.
///
/// A share, not a byte count: the budget itself is derived from a live
/// probe of the host, so it shrinks on a small machine instead of being
/// a number somebody picked once and nobody revisited.
///
/// A quarter, because a repacked matrix is a second copy of weights the
/// process has already mapped: spending more of what is left than the
/// weights themselves could plausibly want is how a cache turns into
/// the reason a model stops fitting.
const DERIVED_COPY_SHARE: u64 = 4;

/// Bytes a process-wide cache of DERIVED copies of already-resident
/// weights may retain -- today, `weight_matrix::repack_cache`, which
/// holds the interleaved (`repacked`) form of matrices the GEMV kernels
/// read.
///
/// Three properties, and each one is why this function exists rather
/// than a constant next to that cache:
///
/// 1. **It is derived, not restated.** The byte figure comes from
///    [`available_bytes`] at the moment the cache first needs one.
/// 2. **It subtracts the expert budget instead of competing with it.**
///    `expert_store` is the SINGLE holder of the expert byte budget,
///    because on unified memory two budgets are the same RAM counted
///    twice. `committed_bytes`
///    ([`crate::expert_store::committed_expert_bytes`]) comes out of
///    the pool BEFORE the share is taken, so the two numbers are one
///    subtraction rather than two opinions.
/// 3. **It can be zero.** A host under `headroom + committed` yields
///    zero, and a zero budget means nothing is retained -- the
///    behaviour before the repack cache existed, which is the
///    memory-constrained case being expressible rather than merely
///    smaller.
///
/// `available` of `None` means the platform would not say, and that
/// resolves to **zero**, the opposite of [`plan_for`]'s answer to the
/// same unknown. The asymmetry is deliberate: guessing wrong there puts
/// a user on a slow path, guessing wrong here retains unbounded copies
/// of a checkpoint. Slow and correct beats unbounded.
pub fn derived_copy_budget(
    available: Option<u64>,
    headroom_bytes: u64,
    committed_bytes: u64,
) -> u64 {
    let Some(available) = available else {
        return 0;
    };
    available
        .saturating_sub(headroom_bytes)
        .saturating_sub(committed_bytes)
        / DERIVED_COPY_SHARE
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn a_model_that_fits_stays_resident() {
        assert_eq!(
            plan_for(8 * GB, Some(32 * GB), 4 * GB, GB),
            FitPlan::Resident
        );
    }

    /// Filling memory to the last byte leaves nothing for the KV cache,
    /// so the headroom is subtracted before the comparison.
    #[test]
    fn headroom_is_subtracted_before_deciding() {
        assert_eq!(
            plan_for(30 * GB, Some(32 * GB), 4 * GB, GB),
            FitPlan::Stream {
                cache_bytes: 28 * GB
            },
            "30 GiB of weights into 32 GiB with 4 GiB reserved does not fit"
        );
    }

    #[test]
    fn a_model_far_larger_than_ram_streams() {
        assert!(matches!(
            plan_for(155 * GB, Some(32 * GB), 4 * GB, GB),
            FitPlan::Stream { .. }
        ));
    }

    /// An unknown probe must not be read as "no memory". Guessing that
    /// would put every user on a platform without a probe onto the slow
    /// path silently.
    #[test]
    fn an_unknown_amount_of_memory_never_forces_streaming() {
        assert_eq!(plan_for(155 * GB, None, 4 * GB, GB), FitPlan::Resident);
    }

    /// A cache too small to hold one decode step's experts makes every
    /// acquire a fresh read: correct, and pathologically slow.
    #[test]
    fn the_cache_never_falls_below_the_floor() {
        assert_eq!(
            plan_for(100 * GB, Some(5 * GB), 4 * GB, 2 * GB),
            FitPlan::Stream {
                cache_bytes: 2 * GB
            },
            "1 GiB usable must be raised to the 2 GiB floor"
        );
    }

    /// The expert budget comes out of the pool BEFORE the derived-copy
    /// share is taken. `expert_store` is the single holder of the expert
    /// byte budget, and on unified memory a repacked byte and an expert
    /// byte are the same RAM: if this subtraction is dropped, the two
    /// budgets are one pool counted twice.
    ///
    /// Sabotage: remove the `committed_bytes` subtraction and the second
    /// assertion goes red.
    #[test]
    fn the_expert_budget_is_subtracted_before_the_derived_copy_share() {
        // 32 GiB, 4 GiB headroom, nothing committed: a quarter of 28.
        assert_eq!(
            derived_copy_budget(Some(32 * GB), 4 * GB, 0),
            7 * GB,
            "a quarter of what is usable"
        );
        // The same host with a 20 GiB expert cache has 8 GiB usable.
        assert_eq!(
            derived_copy_budget(Some(32 * GB), 4 * GB, 20 * GB),
            2 * GB,
            "the expert budget must come out of the pool first"
        );
    }

    /// Zero is reachable, and it has to be: it is how the memory-
    /// constrained case is expressed rather than merely made smaller.
    /// A zero budget retains nothing, which is the behaviour before the
    /// repack cache existed.
    #[test]
    fn a_host_with_no_room_left_gets_a_zero_budget() {
        assert_eq!(derived_copy_budget(Some(4 * GB), 4 * GB, 0), 0, "no slack");
        assert_eq!(
            derived_copy_budget(Some(32 * GB), 4 * GB, 100 * GB),
            0,
            "committed more than exists: saturates to zero, never wraps"
        );
    }

    /// The opposite of [`plan_for`]'s answer to the same unknown, on
    /// purpose: guessing wrong there is slow, guessing wrong here is
    /// unbounded.
    #[test]
    fn an_unknown_amount_of_memory_retains_nothing() {
        assert_eq!(derived_copy_budget(None, 4 * GB, 0), 0);
    }

    #[test]
    fn vm_stat_counts_reclaimable_pages_not_just_free_ones() {
        let sample = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                      Pages free:                          100000.\n\
                      Pages active:                        900000.\n\
                      Pages inactive:                       50000.\n";
        // (100000 + 50000) * 16384
        assert_eq!(parse_vm_stat(sample), Some(150_000 * 16_384));
    }
}
