//! What "the engine is using N GB" means, and how often it may be
//! asked.
//!
//! Frink prices a memory BUDGET before the weights load
//! (`frink_models::device_budget`) and then reads no live figure at
//! all, so an operator has no way to ask whether the engine is really
//! using what was planned -- which is the one question a budget is
//! sized to answer.
//!
//! # Why PSS and not RSS
//!
//! RSS counts every resident page of a mapping, in every process that
//! maps it. PSS -- proportional set size -- divides each shared page by
//! the number of processes sharing it, so summing PSS over a process
//! group counts a shared page ONCE. For an engine whose workers share
//! an mmap'd checkpoint, summing RSS reports several multiples of the
//! truth: two workers sharing a 40 GiB mapping report 80 GiB of a
//! machine that has 64.
//!
//! The two are not interchangeable and must never be silently swapped,
//! which is why [`Footprint`] carries [`FootprintKind`] beside the
//! bytes. A caller comparing this month's PSS with last month's RSS is
//! comparing two different quantities and will read the difference as a
//! leak.
//!
//! # Why it is cached
//!
//! Reading `/proc/<pid>/smaps_rollup` walks the whole VMA list, which
//! on a process mapping tens of gigabytes takes long enough to matter.
//! Every concurrent poller triggering its own walk turns a status
//! endpoint into a load generator, so [`ProbeCache`] collapses them:
//! one probe per TTL, and callers that arrive during a probe wait for
//! its answer rather than starting another.
//!
//! Ported from FreeToken's `daemon/metrics.py`; see
//! `docs/THIRD_PARTY_NOTICES.md`.

/// Which quantity a [`Footprint`] is.
///
/// Reported rather than assumed, because the two answer different
/// questions and a caller that treats them as one will misread the
/// difference between them as a change in the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FootprintKind {
    /// Proportional set size: shared pages divided by their sharers.
    /// The figure to sum across a process group.
    Pss,
    /// Resident set size. The fallback on a kernel without
    /// `smaps_rollup`, and an OVERCOUNT the moment anything is shared.
    Rss,
}

impl FootprintKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FootprintKind::Pss => "pss",
            FootprintKind::Rss => "rss",
        }
    }
}

/// One memory reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footprint {
    pub bytes: u64,
    pub kind: FootprintKind,
}

/// The `Pss:` total from a `/proc/<pid>/smaps_rollup`.
///
/// `smaps_rollup` carries exactly one `Pss:` line -- the kernel has
/// already summed the per-mapping values -- so this takes the first and
/// does not accumulate. Handed the per-mapping `smaps` instead, summing
/// would be right and taking the first would be wrong, which is why the
/// two files are not used interchangeably here.
///
/// Values are in kB, as every `/proc` memory line is. `None` when the
/// field is absent (a kernel older than 4.14) or unparseable, never a
/// zero: an engine using no memory is not a thing that happens, so a
/// zero would be a broken read presented as a fact.
pub fn parse_smaps_rollup_pss(text: &str) -> Option<u64> {
    parse_proc_kb_field(text, "Pss:")
}

/// The `VmRSS:` total from a `/proc/<pid>/status`, for kernels with no
/// `smaps_rollup`.
pub fn parse_status_rss(text: &str) -> Option<u64> {
    parse_proc_kb_field(text, "VmRSS:")
}

/// `<name> <number> kB` -> bytes.
///
/// The unit is checked rather than assumed. Every `/proc` memory field
/// is in kB today, and a field that ever said `mB` and was read as kB
/// would be off by a thousand and look entirely plausible.
fn parse_proc_kb_field(text: &str, name: &str) -> Option<u64> {
    for line in text.lines() {
        let Some(rest) = line.strip_prefix(name) else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let value: u64 = parts.next()?.parse().ok()?;
        if parts.next()? != "kB" {
            return None;
        }
        return Some(value * 1024);
    }
    None
}

/// A value re-probed at most once per TTL, with concurrent callers
/// collapsed onto one probe.
///
/// The collapsing is what the type is for. A probe that takes a second
/// and a status endpoint polled by four dashboards produces four
/// concurrent VMA walks, each slower for the others being there; the
/// endpoint then measures its own instrumentation. Holding the lock
/// ACROSS the probe is therefore deliberate, not an oversight: a caller
/// arriving mid-probe blocks and then finds the fresh value, which is
/// exactly one probe for all of them.
///
/// Time is a parameter, so a test states the clock instead of sleeping.
#[derive(Debug)]
pub struct ProbeCache<T> {
    ttl_ms: u64,
    latest: Option<(T, u64)>,
}

impl<T: Clone> ProbeCache<T> {
    pub fn new(ttl_ms: u64) -> Self {
        ProbeCache {
            ttl_ms,
            latest: None,
        }
    }

    /// The cached value if it is younger than the TTL, otherwise the
    /// result of `probe`.
    ///
    /// A `probe` that answers `None` is NOT cached: a failed read is a
    /// failure to learn anything, and caching it would keep answering
    /// "unknown" for a whole TTL after the condition cleared. A stale
    /// value is not served in its place either -- the caller asked what
    /// is true now, and the honest answer to a failed probe is that
    /// nothing is known.
    pub fn get_or_probe(&mut self, now_ms: u64, probe: impl FnOnce() -> Option<T>) -> Option<T> {
        if let Some((value, taken_at)) = &self.latest {
            if now_ms.saturating_sub(*taken_at) < self.ttl_ms {
                return Some(value.clone());
            }
        }
        let fresh = probe()?;
        self.latest = Some((fresh.clone(), now_ms));
        Some(fresh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROLLUP: &str = "\
55d0c0000000-7ffd0f7fe000 ---p 00000000 00:00 0 [rollup]
Rss:             1048576 kB
Pss:              524288 kB
Pss_Dirty:        131072 kB
Shared_Clean:     786432 kB
";

    #[test]
    fn a_rollup_reports_its_single_already_summed_pss_line() {
        assert_eq!(parse_smaps_rollup_pss(ROLLUP), Some(524_288 * 1024));
    }

    /// A zero would be a broken read presented as a fact, and "the
    /// engine is using 0 GB" is not a thing that happens. Absent stays
    /// absent.
    #[test]
    fn a_missing_or_malformed_field_is_absent_rather_than_zero() {
        assert_eq!(parse_smaps_rollup_pss("Rss: 100 kB\n"), None);
        assert_eq!(parse_smaps_rollup_pss(""), None);
        assert_eq!(parse_smaps_rollup_pss("Pss: notanumber kB\n"), None);
        // The unit is checked rather than assumed: a field read as kB
        // when it said something else is off by a factor and looks
        // entirely plausible.
        assert_eq!(parse_smaps_rollup_pss("Pss: 100 mB\n"), None);
        assert_eq!(parse_smaps_rollup_pss("Pss: 100\n"), None);
    }

    #[test]
    fn vmrss_is_the_fallback_on_a_kernel_without_a_rollup() {
        let status = "Name:\tfrink-server\nVmPeak:\t 900 kB\nVmRSS:\t  4096 kB\n";
        assert_eq!(parse_status_rss(status), Some(4096 * 1024));
    }

    /// The reason the type exists. Four dashboards polling a status
    /// endpoint must not produce four concurrent VMA walks, each slower
    /// for the others being there -- the endpoint would be measuring
    /// its own instrumentation.
    #[test]
    fn concurrent_callers_within_the_ttl_share_one_probe() {
        let mut cache = ProbeCache::new(1000);
        let mut probes = 0;

        for now in [0u64, 100, 999] {
            let value = cache.get_or_probe(now, || {
                probes += 1;
                Some(42u64)
            });
            assert_eq!(value, Some(42));
        }
        assert_eq!(probes, 1, "one probe covers the whole TTL");

        let value = cache.get_or_probe(1000, || {
            probes += 1;
            Some(43)
        });
        assert_eq!(value, Some(43), "past the TTL it re-probes");
        assert_eq!(probes, 2);
    }

    /// A failed probe is a failure to LEARN anything, so it is neither
    /// cached nor papered over with the previous value. Caching it
    /// would keep answering "unknown" for a whole TTL after the
    /// condition cleared; serving the stale value would answer a
    /// question about now with a fact about then.
    #[test]
    fn a_failed_probe_is_neither_cached_nor_answered_with_a_stale_value() {
        let mut cache = ProbeCache::new(100);
        assert_eq!(cache.get_or_probe(0, || Some(7u64)), Some(7));

        assert_eq!(
            cache.get_or_probe(100, || None),
            None,
            "not the stale 7: the caller asked what is true now"
        );

        // And the failure did not become the cached answer.
        let mut probes = 0;
        assert_eq!(
            cache.get_or_probe(101, || {
                probes += 1;
                Some(8)
            }),
            Some(8)
        );
        assert_eq!(
            probes, 1,
            "the next caller probes rather than being told None"
        );
    }
}
