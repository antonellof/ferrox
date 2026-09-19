//! Where a machine's measured bandwidth profile lives, and when it may
//! be trusted.
//!
//! [`crate::qstar`] can turn a [`BandwidthProfile`] into a split
//! policy, but only if something hands it one. This module is that
//! something: it decides which file on disk *is* this machine's
//! profile, reads it, and refuses it when it describes a different
//! machine. Without it every deployment falls back to
//! [`QStarPolicy::fixed_cap`]'s one-fetch-per-step default, which is
//! safe and slow -- the whole point of benchmarking a host is that the
//! result is then found again on the next run.
//!
//! # One file per GPU
//!
//! The profile is stored per **GPU UUID**, at
//! `$XDG_CACHE_HOME/frink/benchbw/<uuid>.json`, not once per box.
//! Bandwidth is a property of a *slot*, not of a machine: two identical
//! cards in the same chassis routinely sit behind different links (x16
//! off the CPU vs. x4 off the chipset), and the `q*` fraction that
//! balances one of them starves the other. Machines with a single card
//! and older benchmarks use the legacy `benchbw.json` next to it.
//!
//! # Lookup order, and the one place it stops
//!
//! An explicit path wins, then [`PROFILE_PATH_ENV`], then this card's
//! `benchbw/<uuid>.json`, then the legacy `benchbw.json`.
//!
//! A candidate that is simply *absent* is skipped -- that is the whole
//! reason the legacy file is in the list. A candidate that **exists but
//! does not parse** is not skipped: the lookup returns [`None`] on the
//! spot and the caller keeps its unbenchmarked default. Falling through
//! would mean a truncated or half-written profile for the card in slot
//! 1 silently promotes slot 0's numbers to describe slot 1, and a
//! wrong fetch fraction is worse than no fetch fraction: it does not
//! degrade to "a bit slower", it puts every decode step's misses on the
//! wrong side of a link that cannot carry them. Corruption is a reason
//! to stop, not a reason to guess.
//!
//! # Naming
//!
//! The environment variable is `FRINK_BENCHBW_PATH` and the cache
//! directory is `frink/` (matching `frink-core`'s `registry_dir`);
//! FreeToken spells the same two `FREETOKEN_BENCHBW_PATH` and
//! `freetoken/`. The on-disk layout is otherwise identical --
//! `benchbw/<uuid>.json` plus the legacy `benchbw.json`, same JSON
//! document -- so a profile written by either tool is readable by the
//! other once it is in the right directory.
//!
//! Ported 1:1 from FreeToken's `moe/bench_profile.py` (Apache-2.0); see
//! `docs/THIRD_PARTY_NOTICES.md`.

use std::path::{Path, PathBuf};

use crate::qstar::{BandwidthProfile, MoeBackend, QStarPolicy};

/// Overrides the whole lookup with one path. Empty means unset.
pub const PROFILE_PATH_ENV: &str = "FRINK_BENCHBW_PATH";

/// The per-GPU profile directory, under the cache directory.
pub const PROFILE_SUBDIR: &str = "benchbw";

/// The single-file profile that predates the per-GPU layout.
pub const LEGACY_PROFILE_FILE: &str = "benchbw.json";

/// The bench format key an engine quant name is measured under.
///
/// The benchmark keys its numbers by expert *format*, not by model,
/// because the CPU-MoE-vs-PCIe-gather ratio the choice rides on is
/// dominated by `(format, hardware)` -- so a profile taken on one
/// workload transfers to any model with the same expert format on the
/// same card. Most engine names are already the bench name; `mxfp4` is
/// benched under its kernel's name, `mxfp4_triton`.
///
/// Anything not in the table is passed through unchanged, which is
/// deliberate rather than lossy: an unmapped name finds no entry in the
/// profile, the lookup yields [`None`], and the caller keeps its safe
/// offload default. Only the offload-family formats that have a CPU MoE
/// weight path can ever resolve to hybrid, and those are exactly the
/// ones listed here.
pub fn bench_format(quant_format: &str) -> &str {
    match quant_format {
        "nvfp4" => "nvfp4",
        "ds_fp4" => "ds_fp4",
        "mxfp4" => "mxfp4_triton",
        "bf16" => "bf16",
        "fp8_block" => "fp8_block",
        other => other,
    }
}

/// `$XDG_CACHE_HOME/frink`, else `$HOME/.cache/frink`, else a
/// temporary directory.
///
/// The last fallback keeps a host with neither variable set (a bare
/// service account, a container) from resolving profiles relative to
/// the process's working directory; it matches `frink-core`'s
/// `registry_dir`. A profile written there does not survive a reboot,
/// which is the honest outcome when the machine has nowhere durable to
/// put one.
pub fn cache_dir() -> PathBuf {
    std::env::var("XDG_CACHE_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .filter(|s| !s.is_empty())
                .map(|h| PathBuf::from(h).join(".cache"))
        })
        .unwrap_or_else(std::env::temp_dir)
        .join("frink")
}

/// The path [`PROFILE_PATH_ENV`] names, if it names one.
///
/// An empty value counts as unset, so `FRINK_BENCHBW_PATH=` in a unit
/// file or a `docker run -e` with no value means "use the normal
/// lookup" rather than "look for a file called nothing".
pub fn env_profile_path() -> Option<PathBuf> {
    std::env::var(PROFILE_PATH_ENV)
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

/// `<cache_dir>/benchbw/<uuid>.json`, or the legacy
/// `<cache_dir>/benchbw.json` when the card has no UUID.
///
/// Takes the cache directory rather than finding it, so the layout can
/// be exercised without a home directory.
pub fn default_profile_path_in(cache_dir: &Path, gpu_uuid: Option<&str>) -> PathBuf {
    match gpu_uuid.filter(|u| !u.is_empty()) {
        Some(uuid) => cache_dir.join(PROFILE_SUBDIR).join(format!("{uuid}.json")),
        None => cache_dir.join(LEGACY_PROFILE_FILE),
    }
}

/// [`default_profile_path_in`] under the real [`cache_dir`].
pub fn default_profile_path(gpu_uuid: Option<&str>) -> PathBuf {
    default_profile_path_in(&cache_dir(), gpu_uuid)
}

// ---- writing a profile ---------------------------------------------

/// A measured (format, card) pair, before it becomes a profile entry.
///
/// All four bandwidths are `Option` for the same reason the profile's
/// are: a run that could only measure one side has not measured the
/// thing the fraction is made of, and a half-entry that reads as
/// complete is worse than none.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Measured {
    pub cpu_moe_gbs: Option<f64>,
    pub pcie_gather_gbs: Option<f64>,
    pub cpu_moe_overlap_gbs: Option<f64>,
    pub pcie_gather_overlap_gbs: Option<f64>,
}

/// Why a measurement cannot become a profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotMeasurable {
    /// One side was measured and the other was not.
    ///
    /// The fraction is a RATIO of the two, so one number alone implies
    /// nothing about the split -- and a profile carrying it would be
    /// consulted by `policy_for` as though it did.
    OnlyOneSide,
    /// A bandwidth came back at or below zero, which is a failed
    /// measurement rather than an infinitely slow link.
    NotPositive,
}

impl std::fmt::Display for NotMeasurable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotMeasurable::OnlyOneSide => write!(
                f,
                "only one side was measured; the fetch fraction is a ratio of \
                 the two, so one number alone says nothing about the split"
            ),
            NotMeasurable::NotPositive => {
                write!(
                    f,
                    "a bandwidth came back at or below zero, which is a failed measurement"
                )
            }
        }
    }
}

impl std::error::Error for NotMeasurable {}

/// Turns a measurement into the entry a profile stores, deriving the
/// verdict the reader consults.
///
/// `threshold` is the same one [`crate::qstar::recommend_backend`]
/// applies, and the CONTENDED pair is preferred wherever it exists --
/// standalone numbers assume each side owns the machine, and neither
/// does once they run together, which is the whole reason the contended
/// pair is measured at all.
pub fn entry_from(
    measured: &Measured,
    threshold: f64,
) -> Result<crate::qstar::KernelBandwidths, NotMeasurable> {
    let positive = |v: Option<f64>| -> Result<Option<f64>, NotMeasurable> {
        match v {
            Some(x) if x > 0.0 && x.is_finite() => Ok(Some(x)),
            Some(_) => Err(NotMeasurable::NotPositive),
            None => Ok(None),
        }
    };
    let cpu = positive(measured.cpu_moe_gbs)?;
    let pcie = positive(measured.pcie_gather_gbs)?;
    let cpu_ov = positive(measured.cpu_moe_overlap_gbs)?;
    let pcie_ov = positive(measured.pcie_gather_overlap_gbs)?;

    if cpu.is_some() != pcie.is_some() || cpu_ov.is_some() != pcie_ov.is_some() {
        return Err(NotMeasurable::OnlyOneSide);
    }
    let (Some(cpu), Some(pcie)) = (cpu, pcie) else {
        return Err(NotMeasurable::OnlyOneSide);
    };

    // The contended pair when it exists, exactly as `fetch_fraction`
    // prefers it, so the verdict and the fraction cannot disagree about
    // which numbers they came from.
    let (verdict_cpu, verdict_pcie) = match (cpu_ov, pcie_ov) {
        (Some(c), Some(p)) => (c, p),
        _ => (cpu, pcie),
    };
    Ok(crate::qstar::KernelBandwidths {
        cpu_moe_gbs: Some(cpu),
        pcie_gather_gbs: Some(pcie),
        cpu_moe_overlap_gbs: cpu_ov,
        pcie_gather_overlap_gbs: pcie_ov,
        recommended: Some(crate::qstar::recommend_backend(
            verdict_cpu,
            verdict_pcie,
            threshold,
        )),
    })
}

/// Writes `profile` where [`read_profile`] will find it.
///
/// Through a temporary file and a rename, because a plain write is not
/// atomic and a reader that catches a half-written profile gets serde's
/// parse error -- which `read_profile` turns into NO profile, silently
/// discarding a measurement the user did take.
pub fn write_profile(path: &Path, profile: &BandwidthProfile) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_vec_pretty(profile)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let tmp = path.with_extension("json.partial");
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, path)
}

/// The newest `benchbw/*.json` under `cache_dir`, else the legacy
/// `benchbw.json`, else [`None`].
///
/// This is for reporting and for tools that want *a* profile without
/// knowing which card they are asking about -- never for the serving
/// path, which must go through [`usable_profile`] so that another
/// card's numbers are refused rather than merely being the most recent.
///
/// Ties on modification time are broken by the greater file name, so
/// the answer does not depend on directory order.
pub fn latest_profile_path_in(cache_dir: &Path) -> Option<PathBuf> {
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(cache_dir.join(PROFILE_SUBDIR)) {
        for entry in entries.flatten() {
            if !entry.file_name().to_string_lossy().ends_with(".json") {
                continue;
            }
            let Ok(mtime) = entry.metadata().and_then(|m| m.modified()) else {
                continue;
            };
            found.push((mtime, entry.path()));
        }
    }
    if let Some((_, path)) = found.into_iter().max() {
        return Some(path);
    }
    let legacy = default_profile_path_in(cache_dir, None);
    legacy.is_file().then_some(legacy)
}

/// [`latest_profile_path_in`] under the real [`cache_dir`].
pub fn latest_profile_path() -> Option<PathBuf> {
    latest_profile_path_in(&cache_dir())
}

/// The profile document at `path`, or [`None`] when it is absent,
/// unreadable, or not a profile.
///
/// This collapses "no file" and "bad file" into one answer, which is
/// fine for a caller that only wants the document. The lookup itself
/// must tell the two apart -- see the module docs -- so it does not use
/// this.
pub fn read_profile(path: &Path) -> Option<BandwidthProfile> {
    match read_candidate(path) {
        Candidate::Profile(profile) => Some(*profile),
        _ => None,
    }
}

/// What one candidate path turned out to be.
enum Candidate {
    /// No such file. Try the next candidate.
    Missing,
    /// The file is there but is not a profile. Stop.
    Corrupt,
    /// Boxed: a profile is several maps, and the other two variants are
    /// empty.
    Profile(Box<BandwidthProfile>),
}

/// Read one candidate, keeping "absent" and "present but broken"
/// distinct.
///
/// Only [`std::io::ErrorKind::NotFound`] counts as absent. A permission
/// error, a directory in the file's place, or a half-written file are
/// all `Corrupt`: the profile *is* claimed by this path, we just cannot
/// have it, and that is precisely the case where borrowing another
/// card's file would be wrong.
fn read_candidate(path: &Path) -> Candidate {
    let body = match std::fs::read_to_string(path) {
        Ok(body) => body,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Candidate::Missing,
        Err(_) => return Candidate::Corrupt,
    };
    match serde_json::from_str::<BandwidthProfile>(&body) {
        Ok(profile) => Candidate::Profile(Box::new(profile)),
        Err(_) => Candidate::Corrupt,
    }
}

/// The candidate paths, in the order they are tried.
///
/// An explicit path is the *only* candidate: someone who named a file
/// wants that file, and quietly serving a different one because theirs
/// was missing would hide the typo.
fn candidate_paths(cache_dir: &Path, path: Option<&Path>, gpu_uuid: Option<&str>) -> Vec<PathBuf> {
    if let Some(explicit) = path {
        return vec![explicit.to_path_buf()];
    }
    let mut candidates = Vec::with_capacity(2);
    if gpu_uuid.is_some_and(|u| !u.is_empty()) {
        candidates.push(default_profile_path_in(cache_dir, gpu_uuid));
    }
    candidates.push(default_profile_path_in(cache_dir, None));
    candidates
}

/// The profile this machine may actually be served with, or [`None`].
///
/// [`None`] means one of three things, all of which the caller answers
/// the same way -- keep the unbenchmarked default:
///
/// - no candidate file exists (nobody has benched this host);
/// - a candidate exists and does not parse (see the module docs: the
///   lookup stops there instead of falling through to another card's
///   file);
/// - the profile parsed but records a *different* GPU name, so its
///   numbers describe hardware that is not in front of us.
///
/// Takes the cache directory rather than finding it, and takes the
/// explicit path already resolved, so the whole rule is testable
/// without a home directory, a GPU, or an environment variable.
pub fn usable_profile_in(
    cache_dir: &Path,
    gpu_name: Option<&str>,
    path: Option<&Path>,
    gpu_uuid: Option<&str>,
) -> Option<BandwidthProfile> {
    let mut found: Option<BandwidthProfile> = None;
    for candidate in candidate_paths(cache_dir, path, gpu_uuid) {
        match read_candidate(&candidate) {
            Candidate::Profile(profile) => {
                found = Some(*profile);
                break;
            }
            // Present and broken: do not borrow the next candidate's
            // numbers for this card.
            Candidate::Corrupt => return None,
            Candidate::Missing => continue,
        }
    }
    let profile = found?;
    profile.matches_gpu(gpu_name).then_some(profile)
}

/// [`usable_profile_in`] under the real [`cache_dir`], with
/// [`PROFILE_PATH_ENV`] standing in for an absent `path`.
pub fn usable_profile(
    gpu_name: Option<&str>,
    path: Option<&Path>,
    gpu_uuid: Option<&str>,
) -> Option<BandwidthProfile> {
    let from_env = path.is_none().then(env_profile_path).flatten();
    let explicit = path.or(from_env.as_deref());
    usable_profile_in(&cache_dir(), gpu_name, explicit, gpu_uuid)
}

/// The bench-recommended offload-family backend for `quant_format` on
/// this card, or [`None`].
///
/// [`None`] means "no usable profile, or nothing measured for this
/// format" -- not "offload". The caller keeps its own default, which is
/// offload today; the distinction matters because a caller that had
/// been told `Hybrid` by configuration should not be silently
/// downgraded by a missing file.
pub fn load_backend_recommendation(
    quant_format: &str,
    gpu_name: Option<&str>,
    path: Option<&Path>,
    gpu_uuid: Option<&str>,
) -> Option<MoeBackend> {
    usable_profile(gpu_name, path, gpu_uuid)?.backend_for(bench_format(quant_format))
}

/// The benched hybrid fetch fraction for `quant_format`, or [`None`].
///
/// The fraction itself -- contended pair first, standalone ratio as the
/// fallback, clamped to `[0, 1]` -- is [`BandwidthProfile::fetch_fraction_for`];
/// this only finds the file it comes from.
pub fn load_hybrid_fetch_fraction(
    quant_format: &str,
    gpu_name: Option<&str>,
    path: Option<&Path>,
    gpu_uuid: Option<&str>,
) -> Option<f64> {
    usable_profile(gpu_name, path, gpu_uuid)?.fetch_fraction_for(bench_format(quant_format))
}

/// The split policy to serve `quant_format` with on this card.
///
/// Total, unlike the two loaders above: every path that ends in "we do
/// not know" ends in the unbenchmarked default, a fixed cap of one
/// fetch per layer per step.
pub fn load_policy(
    quant_format: &str,
    gpu_name: Option<&str>,
    path: Option<&Path>,
    gpu_uuid: Option<&str>,
) -> QStarPolicy {
    match usable_profile(gpu_name, path, gpu_uuid) {
        Some(profile) => profile.policy_for(bench_format(quant_format)),
        None => QStarPolicy::fixed_cap(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, SystemTime};

    /// A unique directory under the system temp directory, removed when
    /// the test ends. Nothing here may read the developer's home
    /// directory or a real GPU.
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "frink-edge-bench-profile-{}-{tag}-{n}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("temp dir is creatable");
            TempDir { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn write(path: &Path, body: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("parent is creatable");
        }
        std::fs::write(path, body).expect("file is writable");
    }

    fn set_mtime(path: &Path, epoch_secs: u64) {
        let file = std::fs::File::options()
            .write(true)
            .open(path)
            .expect("file is openable");
        file.set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(epoch_secs))
            .expect("mtime is settable");
    }

    /// Overlapped 30/(30+90) = 0.25 for `nvfp4`, standalone 50/80 =
    /// 0.625 for `mxfp4_triton`.
    const CARD_PROFILE: &str = r#"{
        "version": 4,
        "gpu": {"name": "NVIDIA GeForce RTX 4090", "uuid": "GPU-slot0"},
        "dtypes": {"nvfp4": "hybrid", "mxfp4_triton": "hybrid"},
        "dtype_kernels": {
            "nvfp4": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                      "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
            "mxfp4_triton": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0}
        }
    }"#;

    /// Deliberately different numbers from `CARD_PROFILE`: 20/(20+80) =
    /// 0.2 for `nvfp4`, and `offload` where the card profile says
    /// hybrid. Any test that accidentally reads this file instead of
    /// the intended one sees it in the assertion.
    const LEGACY_PROFILE: &str = r#"{
        "version": 4,
        "gpu": {"name": "NVIDIA GeForce RTX 4090", "uuid": "GPU-other"},
        "dtypes": {"nvfp4": "offload"},
        "dtype_kernels": {
            "nvfp4": {"cpu_moe_overlap_gbs": 80.0, "pcie_gather_overlap_gbs": 20.0}
        }
    }"#;

    #[test]
    fn the_quant_name_maps_onto_the_bench_format_key() {
        assert_eq!(bench_format("mxfp4"), "mxfp4_triton");
        assert_eq!(bench_format("nvfp4"), "nvfp4");
        assert_eq!(bench_format("ds_fp4"), "ds_fp4");
        assert_eq!(bench_format("bf16"), "bf16");
        assert_eq!(bench_format("fp8_block"), "fp8_block");
    }

    /// An unmapped quant is passed through, finds no entry, and so
    /// leaves the caller on its safe offload default rather than
    /// inheriting some other format's numbers.
    #[test]
    fn an_unmapped_quant_name_is_passed_through_and_finds_no_entry() {
        assert_eq!(bench_format("q4_k_m"), "q4_k_m");
        let dir = TempDir::new("unmapped");
        let path = dir.path().join("profile.json");
        write(&path, CARD_PROFILE);
        assert_eq!(
            load_hybrid_fetch_fraction("q4_k_m", None, Some(&path), None),
            None
        );
        assert_eq!(
            load_backend_recommendation("q4_k_m", None, Some(&path), None),
            None
        );
        assert_eq!(
            load_policy("q4_k_m", None, Some(&path), None),
            QStarPolicy::fixed_cap(1)
        );
    }

    /// Bandwidth is a property of a slot, so the file is keyed by GPU
    /// UUID; only a card with no UUID lands on the legacy file.
    #[test]
    fn the_profile_path_is_one_file_per_gpu_uuid() {
        let root = Path::new("/cache/frink");
        assert_eq!(
            default_profile_path_in(root, Some("GPU-slot0")),
            Path::new("/cache/frink/benchbw/GPU-slot0.json")
        );
        assert_eq!(
            default_profile_path_in(root, Some("GPU-slot1")),
            Path::new("/cache/frink/benchbw/GPU-slot1.json")
        );
        assert_eq!(
            default_profile_path_in(root, None),
            Path::new("/cache/frink/benchbw.json")
        );
        assert_eq!(
            default_profile_path_in(root, Some("")),
            Path::new("/cache/frink/benchbw.json"),
            "an empty uuid is no uuid"
        );
    }

    #[test]
    fn the_newest_per_gpu_profile_is_the_latest_one() {
        let dir = TempDir::new("latest");
        let older = dir.path().join(PROFILE_SUBDIR).join("GPU-slot0.json");
        let newer = dir.path().join(PROFILE_SUBDIR).join("GPU-slot1.json");
        write(&older, CARD_PROFILE);
        write(&newer, CARD_PROFILE);
        write(&dir.path().join(LEGACY_PROFILE_FILE), LEGACY_PROFILE);
        set_mtime(&older, 1_700_000_000);
        set_mtime(&newer, 1_700_000_100);
        assert_eq!(latest_profile_path_in(dir.path()), Some(newer.clone()));
        // Freshness is by mtime, not by name.
        set_mtime(&older, 1_700_000_200);
        assert_eq!(latest_profile_path_in(dir.path()), Some(older));
    }

    #[test]
    fn the_legacy_file_answers_when_there_is_no_per_gpu_profile() {
        let dir = TempDir::new("legacy-latest");
        assert_eq!(
            latest_profile_path_in(dir.path()),
            None,
            "an unbenched host has no profile at all"
        );
        let legacy = dir.path().join(LEGACY_PROFILE_FILE);
        write(&legacy, LEGACY_PROFILE);
        assert_eq!(latest_profile_path_in(dir.path()), Some(legacy));
    }

    /// **The rule this module exists for.** The per-card file is there
    /// but unreadable, and a perfectly good legacy file sits next to
    /// it. The lookup must return `None`, not the legacy profile: this
    /// test fails if the code falls through to the legacy file, because
    /// the legacy numbers (0.2, `offload`) are visibly different from
    /// the card's (0.25, `hybrid`) and the assertions name `None`.
    #[test]
    fn a_corrupt_profile_for_this_card_is_not_replaced_by_the_legacy_file() {
        let dir = TempDir::new("corrupt");
        write(
            &dir.path().join(PROFILE_SUBDIR).join("GPU-slot0.json"),
            "{\"dtypes\": {\"nvfp4\": \"hyb",
        );
        write(&dir.path().join(LEGACY_PROFILE_FILE), LEGACY_PROFILE);
        assert!(
            usable_profile_in(dir.path(), None, None, Some("GPU-slot0")).is_none(),
            "a half-written profile for this card must not borrow the legacy file"
        );
        // The legacy file really is usable on its own, so the `None`
        // above is the corrupt-file rule and not an unreadable fixture.
        assert!(usable_profile_in(dir.path(), None, None, None).is_some());
    }

    /// A file that parses as JSON but is not a profile document is
    /// corrupt too, not empty.
    #[test]
    fn a_json_value_that_is_not_a_profile_counts_as_corrupt() {
        let dir = TempDir::new("not-a-document");
        write(
            &dir.path().join(PROFILE_SUBDIR).join("GPU-slot0.json"),
            "[1, 2, 3]",
        );
        write(&dir.path().join(LEGACY_PROFILE_FILE), LEGACY_PROFILE);
        assert!(usable_profile_in(dir.path(), None, None, Some("GPU-slot0")).is_none());
    }

    /// A card that has never been benched is not an error: the legacy
    /// file is exactly the fallback an absent candidate is meant to
    /// reach.
    #[test]
    fn a_missing_per_gpu_profile_falls_through_to_the_legacy_file() {
        let dir = TempDir::new("fallthrough");
        write(&dir.path().join(LEGACY_PROFILE_FILE), LEGACY_PROFILE);
        let profile = usable_profile_in(dir.path(), None, None, Some("GPU-slot0"))
            .expect("the legacy file answers for an unbenched card");
        assert_eq!(profile.fetch_fraction_for("nvfp4"), Some(0.2));
    }

    /// The per-card file wins over the legacy file when both are there.
    #[test]
    fn the_per_gpu_profile_wins_over_the_legacy_file() {
        let dir = TempDir::new("per-gpu-wins");
        write(
            &dir.path().join(PROFILE_SUBDIR).join("GPU-slot0.json"),
            CARD_PROFILE,
        );
        write(&dir.path().join(LEGACY_PROFILE_FILE), LEGACY_PROFILE);
        let profile = usable_profile_in(dir.path(), None, None, Some("GPU-slot0"))
            .expect("the card's own profile is usable");
        assert_eq!(profile.fetch_fraction_for("nvfp4"), Some(0.25));
        assert_eq!(profile.backend_for("nvfp4"), Some(MoeBackend::Hybrid));
    }

    /// Someone who named a file wants that file. A missing or broken
    /// explicit path is not quietly replaced by a cached profile, which
    /// would hide the typo behind plausible numbers.
    #[test]
    fn an_explicit_path_is_the_only_candidate_considered() {
        let dir = TempDir::new("explicit");
        write(
            &dir.path().join(PROFILE_SUBDIR).join("GPU-slot0.json"),
            CARD_PROFILE,
        );
        write(&dir.path().join(LEGACY_PROFILE_FILE), LEGACY_PROFILE);

        let absent = dir.path().join("typo.json");
        assert!(usable_profile_in(dir.path(), None, Some(&absent), Some("GPU-slot0")).is_none());

        let broken = dir.path().join("broken.json");
        write(&broken, "not json at all");
        assert!(usable_profile_in(dir.path(), None, Some(&broken), Some("GPU-slot0")).is_none());
    }

    /// Bandwidths are hardware facts. A profile recorded on another
    /// card is refused rather than approximated, even though it parses.
    #[test]
    fn a_profile_measured_on_another_card_is_ignored() {
        let dir = TempDir::new("other-card");
        let path = dir.path().join("profile.json");
        write(&path, CARD_PROFILE);
        assert!(usable_profile_in(
            dir.path(),
            Some("NVIDIA GeForce RTX 4090"),
            Some(&path),
            None
        )
        .is_some());
        assert!(
            usable_profile_in(
                dir.path(),
                Some("NVIDIA GeForce RTX 3060 Ti"),
                Some(&path),
                None
            )
            .is_none(),
            "another card's bandwidths are worse than no bandwidths"
        );
    }

    #[test]
    fn the_loaders_resolve_the_quant_name_before_the_lookup() {
        let dir = TempDir::new("loaders");
        let path = dir.path().join("profile.json");
        write(&path, CARD_PROFILE);
        assert_eq!(
            load_hybrid_fetch_fraction("mxfp4", None, Some(&path), None),
            Some(0.625),
            "mxfp4 is benched under mxfp4_triton"
        );
        assert_eq!(
            load_backend_recommendation("mxfp4", None, Some(&path), None),
            Some(MoeBackend::Hybrid)
        );
        assert_eq!(
            load_hybrid_fetch_fraction("nvfp4", None, Some(&path), None),
            Some(0.25)
        );
        assert_eq!(
            load_policy("nvfp4", None, Some(&path), None),
            QStarPolicy::from_fraction(0.25)
        );
    }

    /// No profile is not a verdict: the loaders say `None` and the
    /// caller keeps its own default, which `load_policy` spells out as
    /// the one-fetch cap.
    #[test]
    fn the_loaders_return_none_without_a_usable_profile() {
        let dir = TempDir::new("no-profile");
        let absent = dir.path().join("nothing.json");
        assert_eq!(
            load_backend_recommendation("nvfp4", None, Some(&absent), None),
            None
        );
        assert_eq!(
            load_hybrid_fetch_fraction("nvfp4", None, Some(&absent), None),
            None
        );
        assert_eq!(
            load_policy("nvfp4", None, Some(&absent), None),
            QStarPolicy::fixed_cap(1)
        );
    }

    #[test]
    fn reading_a_profile_yields_the_document_or_nothing() {
        let dir = TempDir::new("read");
        let path = dir.path().join("profile.json");
        write(&path, CARD_PROFILE);
        let profile = read_profile(&path).expect("the fixture parses");
        assert_eq!(profile.gpu.uuid.as_deref(), Some("GPU-slot0"));
        assert_eq!(read_profile(&dir.path().join("absent.json")), None);
        assert_eq!(read_profile(dir.path()), None, "a directory is not a file");
    }

    /// The cache directory follows `frink`'s own layout, and never
    /// resolves relative to the working directory.
    #[test]
    fn the_cache_directory_is_absolute_and_ends_in_frink() {
        let dir = cache_dir();
        assert!(dir.is_absolute(), "{dir:?}");
        assert_eq!(dir.file_name().and_then(|n| n.to_str()), Some("frink"));
    }

    /// The fraction is a RATIO, so one side alone implies nothing about
    /// the split -- and a profile carrying it would be consulted by
    /// `policy_for` as though it did.
    #[test]
    fn one_side_measured_is_not_a_measurement() {
        assert_eq!(
            entry_from(
                &Measured {
                    cpu_moe_gbs: Some(50.0),
                    ..Measured::default()
                },
                1.0
            ),
            Err(NotMeasurable::OnlyOneSide)
        );
        assert_eq!(
            entry_from(
                &Measured {
                    pcie_gather_gbs: Some(20.0),
                    ..Measured::default()
                },
                1.0
            ),
            Err(NotMeasurable::OnlyOneSide)
        );
        // Half a contended pair is the same failure: it is the pair
        // that means something, not either half.
        assert_eq!(
            entry_from(
                &Measured {
                    cpu_moe_gbs: Some(50.0),
                    pcie_gather_gbs: Some(20.0),
                    cpu_moe_overlap_gbs: Some(30.0),
                    pcie_gather_overlap_gbs: None,
                },
                1.0
            ),
            Err(NotMeasurable::OnlyOneSide)
        );
    }

    /// A zero or negative bandwidth is a FAILED measurement, not an
    /// infinitely slow link, and must not be written as though the
    /// benchmark had succeeded.
    #[test]
    fn a_bandwidth_at_or_below_zero_is_a_failed_measurement() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                entry_from(
                    &Measured {
                        cpu_moe_gbs: Some(bad),
                        pcie_gather_gbs: Some(20.0),
                        ..Measured::default()
                    },
                    1.0
                ),
                Err(NotMeasurable::NotPositive),
                "{bad} must not become a profile entry"
            );
        }
    }

    /// The verdict comes from the CONTENDED pair wherever it exists, so
    /// it cannot disagree with the fraction about which numbers it came
    /// from. Standalone numbers assume each side owns the machine, and
    /// neither does once they run together -- which is the entire
    /// reason the contended pair is measured.
    #[test]
    fn the_verdict_and_the_fraction_read_the_same_numbers() {
        // Standalone says hybrid (cpu far outruns pcie); contended says
        // offload, because under contention the CPU side collapses.
        let measured = Measured {
            cpu_moe_gbs: Some(100.0),
            pcie_gather_gbs: Some(20.0),
            cpu_moe_overlap_gbs: Some(10.0),
            pcie_gather_overlap_gbs: Some(19.0),
        };
        let entry = entry_from(&measured, 1.0).expect("both sides measured");
        assert_eq!(
            entry.recommended,
            Some(crate::qstar::MoeBackend::Offload),
            "the contended pair is what the machine actually does"
        );
        // And the fraction the reader derives comes from the same pair.
        let fraction = entry.fetch_fraction().expect("a pair exists");
        let from_pair = crate::qstar::fetch_fraction_from_overlap(10.0, 19.0).unwrap();
        assert!((fraction - from_pair).abs() < 1e-9);

        // With no contended pair, the standalone numbers are all there
        // is, and both halves fall back together.
        let standalone = entry_from(
            &Measured {
                cpu_moe_gbs: Some(100.0),
                pcie_gather_gbs: Some(20.0),
                ..Measured::default()
            },
            1.0,
        )
        .unwrap();
        assert_eq!(
            standalone.recommended,
            Some(crate::qstar::MoeBackend::Hybrid)
        );
    }

    /// A reader that catches a half-written profile gets serde's parse
    /// error, which `read_profile` turns into NO profile -- silently
    /// discarding a measurement the user did take. So the write is
    /// atomic, and nothing partial is ever left behind.
    #[test]
    fn a_profile_is_written_atomically_and_reads_back() {
        let dir = std::env::temp_dir().join(format!(
            "frink-benchbw-write-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let path = default_profile_path_in(&dir, Some("GPU-abc"));

        let mut profile = BandwidthProfile {
            threshold: Some(1.0),
            ..BandwidthProfile::default()
        };
        profile.gpu.name = Some("NVIDIA GeForce RTX 4090".to_string());
        profile.gpu.uuid = Some("GPU-abc".to_string());
        profile.dtype_kernels.insert(
            "q4_k".to_string(),
            entry_from(
                &Measured {
                    cpu_moe_gbs: Some(80.0),
                    pcie_gather_gbs: Some(20.0),
                    ..Measured::default()
                },
                1.0,
            )
            .unwrap(),
        );

        write_profile(&path, &profile).expect("writes");
        let read = read_profile(&path).expect("reads back");
        assert_eq!(read.gpu.uuid.as_deref(), Some("GPU-abc"));
        assert_eq!(
            read.dtype_kernels["q4_k"].recommended,
            Some(crate::qstar::MoeBackend::Hybrid)
        );

        assert!(
            std::fs::read_dir(path.parent().unwrap())
                .unwrap()
                .all(|e| !e
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".partial")),
            "nothing partial may survive a completed write"
        );

        // The card it was taken on is what it is keyed to: another
        // card's split is worse than no split.
        assert!(read.matches_gpu(Some("NVIDIA GeForce RTX 4090")));
        assert!(!read.matches_gpu(Some("NVIDIA GeForce RTX 3060 Ti")));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
