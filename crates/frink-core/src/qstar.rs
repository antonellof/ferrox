//! The `q*` policy: bandwidth-adaptive CPU/GPU expert execution.
//!
//! On a consumer machine an MoE model's experts do not fit in VRAM. A
//! decode step routes to a handful of them, some already resident in
//! the GPU expert cache and some not. The misses have to come from
//! somewhere, and there are exactly two places they can come from:
//!
//! - **fetch** them over PCIe into the cache and multiply on the GPU;
//! - **compute** them on the CPU, straight out of host RAM.
//!
//! FreeToken's observation is that these two run *concurrently* on
//! different hardware, so the step costs `max(fetch_time,
//! cpu_time)`, not their sum -- and the split that minimizes that
//! maximum depends on the machine. A desktop with a x16 link and slow
//! DDR4 should fetch nearly everything; a laptop with a x4 link and
//! fast DDR5 should compute nearly everything on the CPU. Neither
//! choice is right in general, which is why the split is a *measured*
//! parameter rather than a constant.
//!
//! # The fraction
//!
//! With PCIe bandwidth `p` and CPU-MoE bandwidth `c` over the same
//! bytes, perfect overlap wants `fetched : cpu_computed = p : (c - p)`,
//! i.e. fetch a `p/c` fraction of each step's misses. When the two are
//! measured *while contending with each other* -- the honest
//! measurement, since that is how they actually run -- the equivalent
//! expression is `p_ov / (p_ov + c_ov)`. That is the preferred form;
//! the ratio of standalone numbers is the fallback.
//!
//! # Why fixed point
//!
//! The fraction is carried as Q16 ([`FRACTION_ONE`]) and every split is
//! integer arithmetic. A GPU kernel and a CPU reference implementation
//! have to agree on the split *exactly* -- they are two halves of one
//! step, and a one-expert disagreement means an expert computed twice
//! or not at all. Floating point does not guarantee that across two
//! compilers; fixed point does.
//!
//! Ported 1:1 from FreeToken's `moe/bench_profile.py` and the split in
//! `moe/offload_kernels.py` (Apache-2.0); see
//! `docs/THIRD_PARTY_NOTICES.md`.

use serde::{Deserialize, Serialize};

/// Q16: `FRACTION_ONE` means "fetch everything".
pub const FRACTION_ONE: u64 = 1 << 16;

/// Which MoE execution backend a machine should serve with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MoeBackend {
    /// Stream every missing expert over PCIe; the GPU does all the
    /// multiplying.
    Offload,
    /// Fetch some misses, compute the rest on the CPU, overlapped.
    Hybrid,
    /// Ship activations to the CPU and compute every expert there.
    Cpu,
    /// Experts are resident in VRAM; no cache, no streaming.
    Fused,
}

/// The default `cpu_bw > threshold * pcie_bw` factor for recommending
/// hybrid over offload.
pub const DEFAULT_RECOMMEND_THRESHOLD: f64 = 2.0;

/// Which backend a machine's measured bandwidths call for.
///
/// Offload is the always-safe answer, so hybrid has to *earn* the
/// recommendation by a real margin: a CPU only marginally faster than
/// the link buys nothing once the CPU is also running the rest of the
/// model, and paying for a CPU-MoE path that never wins is worse than
/// not having one.
pub fn recommend_backend(cpu_bw_gbs: f64, pcie_bw_gbs: f64, threshold: f64) -> MoeBackend {
    if cpu_bw_gbs > threshold * pcie_bw_gbs {
        MoeBackend::Hybrid
    } else {
        MoeBackend::Offload
    }
}

/// The fetch fraction implied by two *standalone* bandwidths.
///
/// Assumes each side gets the whole machine, which is not what happens
/// when they run together -- prefer
/// [`fetch_fraction_from_overlap`] when the contended pair was
/// measured.
pub fn fetch_fraction_from_bandwidths(cpu_gbs: f64, pcie_gbs: f64) -> Option<f64> {
    if cpu_gbs <= 0.0 || pcie_gbs <= 0.0 {
        return None;
    }
    Some((pcie_gbs / cpu_gbs).min(1.0))
}

/// The fetch fraction implied by the *contended* pair: both sides
/// measured while the other was running.
pub fn fetch_fraction_from_overlap(cpu_overlap_gbs: f64, pcie_overlap_gbs: f64) -> Option<f64> {
    if cpu_overlap_gbs <= 0.0 || pcie_overlap_gbs <= 0.0 {
        return None;
    }
    Some((pcie_overlap_gbs / (pcie_overlap_gbs + cpu_overlap_gbs)).min(1.0))
}

/// How one step's misses are split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QStarSplit {
    /// Misses to fetch over the link and multiply on the GPU.
    pub fetch: usize,
    /// Misses to compute on the CPU from host RAM.
    pub cpu: usize,
}

/// The split policy for one served model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QStarPolicy {
    /// Q16 fetch fraction. Zero means "no fraction configured": the
    /// fixed cap decides instead.
    fraction_q16: u64,
    /// The per-layer, per-step ceiling on fetches, used when no
    /// fraction is configured.
    max_fetch: usize,
}

impl QStarPolicy {
    /// A fixed cap of `max_fetch` experts fetched per layer per step.
    ///
    /// This is what an unbenchmarked machine gets, with `max_fetch ==
    /// 1`: enough to keep the cache warming without betting the step on
    /// a link whose speed nobody measured. `max_fetch == 0` never
    /// fetches (pure CPU); a very large cap is effectively pure
    /// offload.
    pub fn fixed_cap(max_fetch: usize) -> Self {
        QStarPolicy {
            fraction_q16: 0,
            max_fetch,
        }
    }

    /// A measured fetch fraction. Replaces the cap entirely.
    pub fn from_fraction(fraction: f64) -> Self {
        let scaled = (fraction * FRACTION_ONE as f64).round();
        let clamped = scaled.clamp(0.0, FRACTION_ONE as f64) as u64;
        QStarPolicy {
            fraction_q16: clamped,
            max_fetch: usize::MAX,
        }
    }

    pub fn fraction_q16(&self) -> u64 {
        self.fraction_q16
    }

    /// The configured fraction as a float, for reporting.
    pub fn fraction(&self) -> Option<f64> {
        if self.fraction_q16 == 0 {
            None
        } else {
            Some(self.fraction_q16 as f64 / FRACTION_ONE as f64)
        }
    }

    /// Split `missing` cache misses between the link and the CPU.
    pub fn split(&self, missing: usize) -> QStarSplit {
        let fetch = if self.fraction_q16 > 0 {
            balanced_fetch(missing, self.fraction_q16).min(missing)
        } else {
            self.max_fetch.min(missing)
        };
        QStarSplit {
            fetch,
            cpu: missing - fetch,
        }
    }
}

/// How many of `missing` misses to fetch, for a Q16 fraction `f`.
///
/// The fetch side takes time proportional to `F * (1 - f)` and the CPU
/// side to `(M - F) * f`, in units where both bandwidths are folded
/// into `f`. Since they overlap, the step costs the **larger** of the
/// two, so the answer is the `F` that minimizes that maximum.
///
/// The exact value `f * M` is generally not an integer, and rounding it
/// the obvious way is wrong: `ceil` over-fetches. With `M = 3` and `f =
/// 0.415` the ideal is 1.24, and fetching 2 makes the link side ~1.6x
/// slower than balance while the CPU sits idle -- so the rule tries
/// both neighbours and keeps whichever has the smaller maximum, ties to
/// the lower.
pub fn balanced_fetch(missing: usize, fraction_q16: u64) -> usize {
    if fraction_q16 == 0 || missing == 0 {
        return 0;
    }
    let m = missing as i128;
    let f = fraction_q16 as i128;
    let q = FRACTION_ONE as i128;
    let cost = |fetched: i128| -> i128 { (fetched * (q - f)).max((m - fetched) * f) };
    let lo = (m * f) >> 16;
    let best = if cost(lo) <= cost(lo + 1) { lo } else { lo + 1 };
    best.clamp(0, m) as usize
}

/// One measured (format, hardware) pair from a bandwidth benchmark.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KernelBandwidths {
    /// CPU-MoE bandwidth, measured alone.
    pub cpu_moe_gbs: Option<f64>,
    /// PCIe expert-gather bandwidth, measured alone.
    pub pcie_gather_gbs: Option<f64>,
    /// CPU-MoE bandwidth measured while the gather was running.
    pub cpu_moe_overlap_gbs: Option<f64>,
    /// Gather bandwidth measured while CPU-MoE was running.
    pub pcie_gather_overlap_gbs: Option<f64>,
    pub recommended: Option<MoeBackend>,
}

impl KernelBandwidths {
    /// The fetch fraction this entry implies, contended pair first.
    pub fn fetch_fraction(&self) -> Option<f64> {
        if let (Some(cpu), Some(pcie)) = (self.cpu_moe_overlap_gbs, self.pcie_gather_overlap_gbs) {
            if let Some(fraction) = fetch_fraction_from_overlap(cpu, pcie) {
                return Some(fraction);
            }
        }
        if let (Some(cpu), Some(pcie)) = (self.cpu_moe_gbs, self.pcie_gather_gbs) {
            return fetch_fraction_from_bandwidths(cpu, pcie);
        }
        None
    }
}

/// Which GPU a profile was measured on.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProfileGpu {
    pub index: Option<u32>,
    pub name: Option<String>,
    pub uuid: Option<String>,
}

/// A measured bandwidth profile for one machine.
///
/// These numbers are hardware facts, so a profile is keyed to the card
/// it was taken on. A profile whose GPU name disagrees with the card in
/// front of you is not "close enough": applying another machine's split
/// is worse than having no split at all, so the lookups below refuse it
/// rather than approximate.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BandwidthProfile {
    pub version: Option<u32>,
    pub threshold: Option<f64>,
    pub gpu: ProfileGpu,
    /// Per-format verdicts, the authoritative source.
    pub dtypes: std::collections::BTreeMap<String, MoeBackend>,
    /// Per-format measured bandwidths.
    pub dtype_kernels: std::collections::BTreeMap<String, KernelBandwidths>,
    /// Per-model detail, consulted when the per-format entry is absent.
    pub workloads: std::collections::BTreeMap<String, Workload>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Workload {
    pub kernels: std::collections::BTreeMap<String, KernelBandwidths>,
}

impl BandwidthProfile {
    /// Whether this profile describes the card in front of us.
    ///
    /// A profile with no recorded GPU name is accepted (an older
    /// benchmark), because refusing it would silently discard a
    /// measurement the user did take. A profile with a *different*
    /// recorded name is refused.
    pub fn matches_gpu(&self, gpu_name: Option<&str>) -> bool {
        match (self.gpu.name.as_deref(), gpu_name) {
            (Some(recorded), Some(actual)) => recorded == actual,
            _ => true,
        }
    }

    /// The backend this profile recommends for `format`.
    ///
    /// The per-format verdict wins. Failing that, the per-model entries
    /// for the same format are aggregated **conservatively**: hybrid
    /// only if every model that was measured picked hybrid, because one
    /// model that does not benefit is evidence the machine is on the
    /// wrong side of the line.
    pub fn backend_for(&self, format: &str) -> Option<MoeBackend> {
        if let Some(verdict) = self.dtypes.get(format) {
            return Some(*verdict);
        }
        let picks: Vec<MoeBackend> = self
            .workloads
            .values()
            .filter_map(|w| w.kernels.get(format))
            .filter_map(|k| k.recommended)
            .collect();
        if picks.is_empty() {
            return None;
        }
        Some(if picks.iter().all(|p| *p == MoeBackend::Hybrid) {
            MoeBackend::Hybrid
        } else {
            MoeBackend::Offload
        })
    }

    /// The fetch fraction this profile implies for `format`.
    pub fn fetch_fraction_for(&self, format: &str) -> Option<f64> {
        if let Some(fraction) = self
            .dtype_kernels
            .get(format)
            .and_then(KernelBandwidths::fetch_fraction)
        {
            return Some(fraction);
        }
        self.workloads
            .values()
            .filter_map(|w| w.kernels.get(format))
            .find_map(KernelBandwidths::fetch_fraction)
    }

    /// The policy to serve `format` with, given this profile.
    ///
    /// A profile that measured nothing usable for this format yields
    /// the unbenchmarked default: a fixed cap of one fetch per layer
    /// per step.
    pub fn policy_for(&self, format: &str) -> QStarPolicy {
        match self.fetch_fraction_for(format) {
            Some(fraction) => QStarPolicy::from_fraction(fraction),
            None => QStarPolicy::fixed_cap(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_has_to_beat_the_link_by_a_real_margin() {
        assert_eq!(
            recommend_backend(100.0, 40.0, DEFAULT_RECOMMEND_THRESHOLD),
            MoeBackend::Hybrid
        );
        // Only 1.5x: not worth a CPU-MoE path.
        assert_eq!(
            recommend_backend(60.0, 40.0, DEFAULT_RECOMMEND_THRESHOLD),
            MoeBackend::Offload
        );
    }

    #[test]
    fn the_fraction_comes_from_the_two_bandwidths() {
        assert_eq!(fetch_fraction_from_bandwidths(100.0, 40.0), Some(0.4));
        assert_eq!(fetch_fraction_from_overlap(90.0, 30.0), Some(0.25));
        // A link faster than host RAM should fetch everything, never
        // more than everything.
        assert_eq!(fetch_fraction_from_bandwidths(10.0, 40.0), Some(1.0));
        assert_eq!(fetch_fraction_from_bandwidths(0.0, 40.0), None);
    }

    /// The regression the rule exists for: rounding up over-fetches and
    /// leaves the CPU idle while the link is the bottleneck.
    #[test]
    fn the_split_minimizes_the_slower_side_rather_than_rounding() {
        let policy = QStarPolicy::from_fraction(0.415);
        assert_eq!(policy.split(3).fetch, 1, "1.24 ideal -> 1, not ceil 2");
        assert_eq!(policy.split(4).fetch, 2, "1.66 ideal -> 2");
    }

    #[test]
    fn the_split_tracks_the_fraction_within_one_expert() {
        for fraction in [0.1, 0.415, 0.454, 0.7, 1.0] {
            let policy = QStarPolicy::from_fraction(fraction);
            for missing in 0..=64usize {
                let split = policy.split(missing);
                assert_eq!(split.fetch + split.cpu, missing, "every miss is assigned");
                let ideal = fraction * missing as f64;
                assert!(
                    (split.fetch as f64 - ideal).abs() <= 1.0,
                    "fraction={fraction} missing={missing} fetch={}",
                    split.fetch
                );
            }
        }
    }

    #[test]
    fn a_full_fraction_fetches_everything_and_a_zero_cap_fetches_nothing() {
        assert_eq!(QStarPolicy::from_fraction(1.0).split(9).fetch, 9);
        assert_eq!(QStarPolicy::fixed_cap(0).split(9).fetch, 0);
        assert_eq!(QStarPolicy::fixed_cap(0).split(9).cpu, 9);
    }

    /// The unbenchmarked default: warm the cache one expert at a time,
    /// send the rest to the CPU.
    #[test]
    fn the_fixed_cap_bounds_fetches_per_step() {
        let policy = QStarPolicy::fixed_cap(1);
        assert_eq!(policy.split(8), QStarSplit { fetch: 1, cpu: 7 });
        assert_eq!(policy.split(0), QStarSplit { fetch: 0, cpu: 0 });
        assert_eq!(policy.fraction(), None);
    }

    fn profile() -> BandwidthProfile {
        let json = serde_json::json!({
            "version": 4,
            "gpu": {"name": "NVIDIA GeForce RTX 4090", "uuid": "GPU-abc"},
            "dtypes": {"nvfp4": "hybrid"},
            "dtype_kernels": {
                "nvfp4": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                          "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
                "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0}
            },
            "workloads": {
                "qwen": {"kernels": {"mxfp4_triton": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0,
                                                      "recommended": "hybrid"}}}
            }
        });
        serde_json::from_value(json).expect("profile parses")
    }

    /// The contended pair is the honest measurement, so it wins over
    /// the standalone ratio for the same format.
    #[test]
    fn the_overlapped_measurement_wins_over_the_standalone_ratio() {
        let profile = profile();
        assert_eq!(profile.fetch_fraction_for("nvfp4"), Some(0.25));
        assert_eq!(profile.fetch_fraction_for("bf16"), Some(0.4));
    }

    #[test]
    fn a_per_model_entry_fills_in_for_a_missing_per_format_one() {
        let profile = profile();
        assert_eq!(profile.fetch_fraction_for("mxfp4_triton"), Some(0.625));
        assert_eq!(
            profile.backend_for("mxfp4_triton"),
            Some(MoeBackend::Hybrid)
        );
        assert_eq!(profile.backend_for("nvfp4"), Some(MoeBackend::Hybrid));
        assert_eq!(profile.backend_for("q4_0"), None);
    }

    /// A profile from another card is refused rather than approximated:
    /// these are hardware numbers, and the wrong ones are worse than
    /// none.
    #[test]
    fn a_profile_from_another_card_is_refused() {
        let profile = profile();
        assert!(profile.matches_gpu(Some("NVIDIA GeForce RTX 4090")));
        assert!(!profile.matches_gpu(Some("NVIDIA GeForce RTX 3060 Ti")));
        assert!(
            profile.matches_gpu(None),
            "an unnamed card is not a mismatch"
        );
    }

    #[test]
    fn an_unmeasured_format_falls_back_to_the_one_fetch_default() {
        let profile = profile();
        assert_eq!(profile.policy_for("q4_0"), QStarPolicy::fixed_cap(1));
        assert_eq!(
            profile.policy_for("nvfp4"),
            QStarPolicy::from_fraction(0.25)
        );
    }
}
