//! How many bytes the selected backend will let this process hold --
//! the right-hand side of [`crate::kv_budget`]'s inequality.
//!
//! One rule throughout: **ask the device, do not model the operating
//! system**. Each backend has exactly one query and it is the vendor's
//! own answer:
//!
//! | backend | query | crate |
//! |---|---|---|
//! | Metal | `MTLDevice.recommendedMaxWorkingSetSize` | `frink_metal::MetalProfile` |
//! | CUDA | `cuMemGetInfo` free bytes | `frink_cuda::HardwareProfile` |
//! | CPU | total physical RAM, minus a reserve | `frink_cuda::HardwareProfile` |
//!
//! The serving plan explicitly rules out the alternative -- process
//! `phys_footprint` sampling, wired-memory limits, jetsam avoidance, a
//! `free + inactive + active * ratio` dynamic ceiling. None of it is
//! here and none of it should be added: a conservative, explainable
//! number beats a clever one, and every one of those mechanisms is an
//! Apple-specific workaround for an allocator frink does not have.
//!
//! # What this number is not
//!
//! It is a **ceiling to plan against, not a reservation**. Nothing here
//! allocates, nothing holds the memory, and every source is a snapshot:
//! another process can take the VRAM a moment later, and macOS can
//! shrink a recommended working set under pressure.
//!
//! It is also **approximate for frink specifically**, for a reason
//! that has nothing to do with the probe: frink mmaps its quantized
//! weights. Their pages are owned by the kernel's page cache, not by
//! frink, so a check that charges the full checkpoint against this
//! budget is charging an upper bound. A model that overruns the budget
//! may still run, page-faulting; a model that fits may still be evicted
//! by pressure from elsewhere on the machine. Every path that prints
//! this number says so.

/// Which pool a budget was drawn from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetBackend {
    Cpu,
    Metal,
    Cuda,
}

impl BudgetBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            BudgetBackend::Cpu => "cpu",
            BudgetBackend::Metal => "metal",
            BudgetBackend::Cuda => "cuda",
        }
    }
}

impl std::fmt::Display for BudgetBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Lets a CLI flag take a backend name without this crate depending on
/// clap: clap derives a value parser from `FromStr`.
impl std::str::FromStr for BudgetBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" | "host" => Ok(BudgetBackend::Cpu),
            "metal" => Ok(BudgetBackend::Metal),
            "cuda" => Ok(BudgetBackend::Cuda),
            other => Err(format!("unknown backend `{other}` (cpu, metal, cuda)")),
        }
    }
}

/// Fraction of a *host RAM* budget held back for the OS and everything
/// else running on the machine. Deliberately blunt: the alternative is
/// modelling the OS, which the plan rules out.
pub const CPU_RESERVE_FRACTION: f64 = 0.2;

/// Fraction of a *device* budget (Metal working set, free VRAM) held
/// back for driver allocations, command buffers and the activation
/// scratch this module does not itemise.
pub const DEVICE_RESERVE_FRACTION: f64 = 0.1;

/// Overrides the probe entirely (`FRINK_DEVICE_BUDGET_BYTES`). The
/// escape hatch for a host whose real ceiling is something frink
/// cannot see -- a container memory limit, a shared GPU, an operator
/// who simply knows better.
pub const BUDGET_ENV: &str = "FRINK_DEVICE_BUDGET_BYTES";

/// A probed byte ceiling plus the sentence explaining where it came
/// from. The sentence is not decoration: a budget a user cannot trace
/// back to a query is a budget they will disable.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBudget {
    pub backend: BudgetBackend,
    /// What the query returned, before the reserve.
    pub total_bytes: u64,
    /// `total_bytes` minus the reserve: what a plan may actually spend.
    pub usable_bytes: u64,
    /// Held-back fraction, as applied.
    pub reserve_fraction: f64,
    /// Human sentence naming the query, e.g.
    /// "Metal recommendedMaxWorkingSetSize".
    pub source: String,
    /// True whenever the checkpoint is mmap'd -- i.e. always, for GGUF
    /// -- because resident weight bytes are then the kernel's business,
    /// not frink's. Kept as a field rather than a constant so nothing
    /// downstream can print the budget without deciding what to say
    /// about it.
    pub approximate: bool,
}

impl DeviceBudget {
    /// Applies `reserve` to `total` and records the source.
    pub fn new(backend: BudgetBackend, total_bytes: u64, reserve: f64, source: String) -> Self {
        let reserve = reserve.clamp(0.0, 1.0);
        DeviceBudget {
            backend,
            total_bytes,
            usable_bytes: (total_bytes as f64 * (1.0 - reserve)) as u64,
            reserve_fraction: reserve,
            source,
            approximate: true,
        }
    }

    /// Probes the backend the process is configured to use, honouring
    /// `FRINK_DEVICE_BUDGET_BYTES` first.
    ///
    /// `backend` is the caller's already-resolved choice (the CLI's
    /// `--device`, the server's `FRINK_METAL`/`FRINK_CUDA`), not a
    /// second guess at it -- this module decides how much memory a
    /// backend has, never which backend runs.
    pub fn detect(backend: BudgetBackend) -> Self {
        if let Some(bytes) = env_override() {
            return DeviceBudget {
                backend,
                total_bytes: bytes,
                usable_bytes: bytes,
                reserve_fraction: 0.0,
                source: format!("{BUDGET_ENV} override (no reserve applied)"),
                approximate: true,
            };
        }
        match backend {
            BudgetBackend::Metal => metal_budget(),
            BudgetBackend::Cuda => cuda_budget(),
            BudgetBackend::Cpu => host_ram_budget(),
        }
    }

    /// True when nothing could be probed. Callers must treat this as
    /// "do not enforce" rather than "reject everything": refusing to
    /// load because a probe failed would be worse than not checking.
    pub fn is_unknown(&self) -> bool {
        self.total_bytes == 0
    }

    /// Where `usable_bytes` came from, spelled out.
    ///
    /// `source` describes `total_bytes`, so printing it next to
    /// `usable_bytes` -- as the CLI's over-budget warning did -- reads
    /// as "27487790694 bytes is the total physical host RAM" when the
    /// machine has 32 GiB and 20% is held back. Same two-values-one-
    /// label shape as the rest of this repo's bugs.
    pub fn usable_provenance(&self) -> String {
        if self.is_unknown() {
            return self.source.clone();
        }
        format!(
            "{:.0}% of {} {}, {:.0}% held back",
            (1.0 - self.reserve_fraction) * 100.0,
            self.total_bytes,
            self.source,
            self.reserve_fraction * 100.0,
        )
    }

    /// The caveat sentence every printer of this number owes the user.
    pub fn caveat(&self) -> &'static str {
        "approximate: frink mmaps quantized weights, so how much of them stays resident is \
         the kernel's page cache to decide; this charges the whole checkpoint, which is an \
         upper bound, and the budget itself is a snapshot, not a reservation"
    }
}

impl std::fmt::Display for DeviceBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_unknown() {
            return write!(
                f,
                "{} budget: unknown ({}); no ceiling enforced",
                self.backend, self.source
            );
        }
        write!(
            f,
            "{} budget: {} usable of {} total ({:.0}% held back) via {}",
            self.backend,
            human(self.usable_bytes),
            human(self.total_bytes),
            self.reserve_fraction * 100.0,
            self.source
        )
    }
}

fn env_override() -> Option<u64> {
    std::env::var(BUDGET_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
}

/// `MTLDevice.recommendedMaxWorkingSetSize`. Without `--features metal`
/// there is no device to ask and no Metal execution either, so this
/// falls back to host RAM and says so.
fn metal_budget() -> DeviceBudget {
    let profile = frink_metal::MetalProfile::detect();
    if profile.available && profile.recommended_working_set_bytes > 0 {
        return DeviceBudget::new(
            BudgetBackend::Metal,
            profile.recommended_working_set_bytes,
            DEVICE_RESERVE_FRACTION,
            format!(
                "Metal recommendedMaxWorkingSetSize on {}",
                profile.device_name.as_deref().unwrap_or("unnamed device")
            ),
        );
    }
    let mut fallback = host_ram_budget();
    fallback.backend = BudgetBackend::Metal;
    fallback.source = format!(
        "no Metal device query available; fell back to {}",
        fallback.source
    );
    fallback
}

/// `cuMemGetInfo`'s free half, not the card's total: another process
/// may already hold most of it. Compiles without `--features cuda`,
/// where `HardwareProfile` honestly reports no device and this falls
/// back to host RAM.
fn cuda_budget() -> DeviceBudget {
    let profile = frink_cuda::HardwareProfile::detect();
    if profile.cuda_available && profile.cuda_vram_free_bytes > 0 {
        return DeviceBudget::new(
            BudgetBackend::Cuda,
            profile.cuda_vram_free_bytes,
            DEVICE_RESERVE_FRACTION,
            format!(
                "cuMemGetInfo free VRAM on {} ({} total)",
                profile.cuda_device_name.as_deref().unwrap_or("device 0"),
                human(profile.cuda_vram_total_bytes)
            ),
        );
    }
    let mut fallback = host_ram_budget();
    fallback.backend = BudgetBackend::Cuda;
    fallback.source = format!(
        "no CUDA device query available; fell back to {}",
        fallback.source
    );
    fallback
}

/// Total physical RAM minus [`CPU_RESERVE_FRACTION`]. Reported as `0`
/// on a host whose RAM cannot be read (see
/// `frink_cuda::HardwareProfile`), which
/// [`DeviceBudget::is_unknown`] turns into "do not enforce".
fn host_ram_budget() -> DeviceBudget {
    let total = frink_cuda::HardwareProfile::detect().host_ram_total_bytes;
    if total == 0 {
        return DeviceBudget {
            backend: BudgetBackend::Cpu,
            total_bytes: 0,
            usable_bytes: 0,
            reserve_fraction: 0.0,
            source: "host RAM could not be probed on this platform".to_string(),
            approximate: true,
        };
    }
    DeviceBudget::new(
        BudgetBackend::Cpu,
        total,
        CPU_RESERVE_FRACTION,
        "total physical host RAM".to_string(),
    )
}

pub(crate) fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    format!("{v:.2} {}", UNITS[u])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_is_applied_and_reported() {
        let b = DeviceBudget::new(BudgetBackend::Cpu, 1000, 0.2, "test".into());
        assert_eq!(b.total_bytes, 1000);
        assert_eq!(b.usable_bytes, 800);
        assert_eq!(b.reserve_fraction, 0.2);
        assert!(!b.is_unknown());
        // Always approximate: frink mmaps its weights.
        assert!(b.approximate);
    }

    #[test]
    fn a_nonsense_reserve_is_clamped_rather_than_producing_a_negative_budget() {
        let over = DeviceBudget::new(BudgetBackend::Cpu, 1000, 5.0, "test".into());
        assert_eq!(over.usable_bytes, 0);
        let under = DeviceBudget::new(BudgetBackend::Cpu, 1000, -1.0, "test".into());
        assert_eq!(under.usable_bytes, 1000);
    }

    #[test]
    fn zero_total_reads_as_unknown_not_as_a_zero_ceiling() {
        let b = DeviceBudget::new(BudgetBackend::Cpu, 0, 0.2, "nothing to probe".into());
        assert!(b.is_unknown());
        assert!(b.to_string().contains("no ceiling enforced"), "{b}");
    }

    /// Runs in both worlds, like the backend probes themselves: on a
    /// host that can report RAM the budget must be plausible and
    /// smaller than the total; on one that cannot it must be unknown.
    #[test]
    fn cpu_budget_is_either_unknown_or_a_plausible_fraction_of_real_ram() {
        let b = DeviceBudget::detect(BudgetBackend::Cpu);
        assert_eq!(b.backend, BudgetBackend::Cpu);
        if b.is_unknown() {
            assert_eq!(b.usable_bytes, 0);
        } else {
            assert!(b.total_bytes > 128 * 1024 * 1024);
            assert!(b.usable_bytes < b.total_bytes);
            assert!(b.usable_bytes > b.total_bytes / 2);
            assert!(b.to_string().contains("host RAM"), "{b}");
        }
    }

    /// Without `--features metal`/`cuda` these must still resolve (to
    /// the host-RAM fallback) rather than failing to compile or
    /// panicking -- the whole point of the honest-zero probe structs.
    #[test]
    fn accelerator_budgets_fall_back_to_host_ram_when_no_device_answers() {
        for backend in [BudgetBackend::Metal, BudgetBackend::Cuda] {
            let b = DeviceBudget::detect(backend);
            assert_eq!(b.backend, backend);
            if b.source.contains("fell back") {
                assert!(b.source.contains("host RAM"), "{b}");
            }
        }
    }

    #[test]
    fn human_bytes_are_readable_at_every_scale() {
        assert_eq!(human(0), "0.00 B");
        assert_eq!(human(1024), "1.00 KiB");
        assert_eq!(human(3 * 1024 * 1024 * 1024), "3.00 GiB");
    }
}
