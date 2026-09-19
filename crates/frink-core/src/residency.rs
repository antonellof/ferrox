//! What a layer's expert bank actually settled at in host memory, and
//! how much of host memory this machine will let us page-lock at all.
//!
//! [`crate::placement`] decides *which* layers give up their GPU expert
//! path when the banks do not fit the page-locking budget. It does not
//! say what happens to their bytes, and it takes the budget as a
//! parameter nothing produces. This module is both halves: the label a
//! bank carries once it has been filled, and the budget the label
//! selection runs against.
//!
//! # Three classes, one that can feed the GPU
//!
//! - [`HostResidency::Pinned`] -- page-locked *and* device-addressable
//!   (`cudaHostRegister`). Only this class can be DMA'd from, so only
//!   this class can serve the GPU expert cache.
//! - [`HostResidency::Locked`] -- resident and unswappable (`mlock`)
//!   but with no device address. The CPU executor reads it at full
//!   speed; the GPU cannot see it at all.
//! - [`HostResidency::Pageable`] -- an ordinary mapping. Correct, and
//!   free of any quota, but the kernel may swap it out from under a
//!   decode step.
//!
//! The two non-pinned classes exist for hosts that cap the CUDA pin
//! quota. Locking costs no pin quota, so a layer that has been handed
//! to the CPU executor anyway should be locked rather than pinned:
//! that is the whole point of the split.
//!
//! # Failure is recorded, not assumed
//!
//! A lock can fail -- `RLIMIT_MEMLOCK` is small by default and the
//! quota is per process, so the first refusal means every later,
//! larger, request fails too. When it does, the bank is *still there*,
//! merely pageable, and everything downstream still works. So a failed
//! lock is not an error: [`ResidencyPlan`] records the class the bank
//! **achieved** and echoes the achieved labels back, and one pageable
//! bank downgrades its whole layer (a layer is many banks; the layer is
//! only as resident as its worst one). The alternative -- assuming the
//! request was honored -- labels a swappable layer `Locked`, and the
//! swap-in then lands in the middle of a decode step as an
//! unexplainable latency spike.
//!
//! # The three invariants a non-pinned layer imposes
//!
//! [`BankResidency::new`] refuses a configuration that violates any of
//! them, at attach time, before a byte has moved:
//!
//! 1. **A non-pinned layer must already be a CPU layer.** It has no
//!    device address, so the only executor that can read it is the CPU
//!    one. If the CPU-layer set were decided after the labels, the
//!    layer's first decode step would index a device pointer that was
//!    never registered.
//! 2. **Prefill overlap is refused, not degraded.** The overlap path
//!    DMAs a layer's experts from a registered bank while the previous
//!    layer computes. A locked bank cannot serve that copy at all, so
//!    the answer is to turn the overlap off *up front* (the caller
//!    does, on the same signal) rather than to discover it per layer.
//! 3. **An unpinned layer accepts only the whole-layer materialize.**
//!    That path writes expert `e` to slot `e` -- `position == expert
//!    id` -- which is the one mapping that needs no device alias for
//!    the source rows. The LRU's slot remapping ([`crate::expert_cache`])
//!    picks arbitrary victim slots, and honoring it would require
//!    addressing individual host rows from the device. So a slot-remap
//!    copy staged against an unpinned layer is an error, not something
//!    to fix up silently with a different slot.
//!
//! # The budget the labels are chosen against
//!
//! [`resolve_pin_budget`] answers "how many bytes may this process
//! page-lock". On plain Linux the answer is [`None`]: nothing caps
//! pinning, every layer stays on the GPU path, and
//! [`crate::placement::auto_cpu_layers`] hands out no CPU layers. On
//! WSL, CUDA runs over WDDM and pinning is capped near half of RAM
//! **shared across every process on the machine**, so the budget is a
//! deliberately conservative 40% of physical RAM. Without this,
//! `auto_cpu_layers` has no budget on the one platform where the cap
//! actually bites, and the load dies inside the page-lock call *after*
//! the whole checkpoint has been read off disk.
//!
//! Every rule takes the host facts as parameters -- the kernel release
//! string and the physical-RAM figure -- so all of it is testable on
//! any host. [`host_pin_budget_bytes`] is the thin wrapper that reads
//! those two facts and applies them.
//!
//! Ported 1:1 from FreeToken's `moe/host_banks.py` (`HostResidency`,
//! `_ResidencyPlan`, `_settle`, `pin_banks`), the `set_bank_sources` /
//! `copy_missing` invariants in `moe/offload_cache.py`, and
//! `engine/engine.py`'s `_pin_budget_bytes` (Apache-2.0); see
//! `docs/THIRD_PARTY_NOTICES.md`.

use std::collections::BTreeSet;

/// Residency class of one layer's expert bank.
///
/// The string forms are the wire values FreeToken uses
/// (`"pinned"` / `"locked"` / `"pageable"`), so a label written by
/// either side reads the same.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HostResidency {
    /// Page-locked and device-addressable: the only class the GPU
    /// expert path can DMA from.
    #[default]
    Pinned,
    /// Resident (unswappable) but CPU-only: no device address.
    Locked,
    /// An ordinary mapping; may be swapped out.
    Pageable,
}

impl HostResidency {
    pub fn as_str(self) -> &'static str {
        match self {
            HostResidency::Pinned => "pinned",
            HostResidency::Locked => "locked",
            HostResidency::Pageable => "pageable",
        }
    }

    /// The inverse of [`as_str`](Self::as_str). Unknown text is
    /// [`None`] rather than a default: a label nobody wrote is not the
    /// same as a label that says "pinned", and guessing `Pinned` would
    /// promise a device address that does not exist.
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "pinned" => Some(HostResidency::Pinned),
            "locked" => Some(HostResidency::Locked),
            "pageable" => Some(HostResidency::Pageable),
            _ => None,
        }
    }

    /// Whether the GPU can read this bank directly. The single
    /// question every consumer of a label actually asks.
    pub fn is_device_addressable(self) -> bool {
        matches!(self, HostResidency::Pinned)
    }
}

impl std::fmt::Display for HostResidency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The per-layer labels a split-residency load asks for: the CPU
/// layers locked, everything else pinned.
///
/// This is the bridge from [`crate::placement::auto_cpu_layers`] --
/// which says *which* layers leave the GPU path -- to the loader, which
/// needs to know what to do with their bytes. Locked rather than
/// pageable because a CPU layer is read on every step that routes to
/// it: it must not be swappable, it just does not need a pin.
pub fn requested_labels(num_layers: usize, cpu_layers: &BTreeSet<u32>) -> Vec<HostResidency> {
    (0..num_layers)
        .map(|layer| {
            if cpu_layers.contains(&(layer as u32)) {
                HostResidency::Locked
            } else {
                HostResidency::Pinned
            }
        })
        .collect()
}

/// What the loader must do to a bank it has just finished filling.
///
/// Always *after* the fill: the banks are lazy anonymous mappings, so
/// page-locking an empty one faults and zero-fills every page, and the
/// read then overwrites all of it -- a whole redundant pass over
/// hundreds of gigabytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettleAction {
    /// Page-lock and register for device access (`cudaHostRegister`).
    PageLockForDevice,
    /// Page-lock only (`mlock`): resident, no device address, no pin
    /// quota spent.
    LockResident,
    /// Nothing to do -- a pageable bank is the mapping as allocated.
    LeavePageable,
}

/// The residency labels a bank load asked for, and the ones it got.
///
/// The loader walks layers, calls [`settle_action`](Self::settle_action)
/// for each, does the syscall itself, and reports back with
/// [`record`](Self::record) / [`record_lock`](Self::record_lock).
/// Nothing here touches memory: this side owns the bookkeeping, the
/// caller owns the syscall. [`achieved_labels`](Self::achieved_labels)
/// is what then feeds [`BankResidency::new`] -- the *achieved* labels,
/// never the requested ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyPlan {
    requested: Vec<HostResidency>,
    achieved: Vec<Option<HostResidency>>,
    applied: bool,
    lock_quota_exhausted: bool,
}

impl ResidencyPlan {
    pub fn new(requested: Vec<HostResidency>) -> Self {
        let achieved = vec![None; requested.len()];
        Self {
            requested,
            achieved,
            applied: false,
            lock_quota_exhausted: false,
        }
    }

    /// The no-split plan: every layer pinned, which is what a host
    /// with no pin cap always does.
    pub fn all_pinned(num_layers: usize) -> Self {
        Self::new(vec![HostResidency::Pinned; num_layers])
    }

    pub fn num_layers(&self) -> usize {
        self.requested.len()
    }

    /// Whether any layer was asked to be something other than pinned.
    /// A plan with none of them is indistinguishable from no plan.
    pub fn has_unpinned(&self) -> bool {
        self.requested.iter().any(|r| !r.is_device_addressable())
    }

    /// Whether any settle point consulted this plan.
    ///
    /// A loader that pins everything by construction never asks, and
    /// its banks are all pinned no matter what was requested. That is
    /// still a *working* load -- CPU-layer decode reads a pinned bank
    /// perfectly well -- it just saved no pin quota, so the honest
    /// report is "not applied", not the labels that were wished for.
    pub fn applied(&self) -> bool {
        self.applied
    }

    /// Whether a lock has already failed.
    ///
    /// The OS lock ceiling is a per-process quota, so once one request
    /// is over it every later (larger cumulative) request is too.
    /// Sticky, therefore, rather than retried per bank: retrying buys
    /// nothing and turns one failure into one failed syscall per bank
    /// for the rest of the load.
    pub fn lock_quota_exhausted(&self) -> bool {
        self.lock_quota_exhausted
    }

    /// The class layer `layer_id` was asked for.
    ///
    /// # Panics
    /// If `layer_id` is not a layer of this plan.
    pub fn requested(&self, layer_id: usize) -> HostResidency {
        self.requested[layer_id]
    }

    /// What to do with layer `layer_id`'s freshly filled bank, marking
    /// the plan applied.
    ///
    /// Returns [`SettleAction::LeavePageable`] for a `Locked` layer
    /// once the lock quota is known to be exhausted -- the syscall
    /// would fail anyway, and the bank ends up pageable either way.
    ///
    /// # Panics
    /// If `layer_id` is not a layer of this plan.
    pub fn settle_action(&mut self, layer_id: usize) -> SettleAction {
        self.applied = true;
        match self.requested[layer_id] {
            HostResidency::Pinned => SettleAction::PageLockForDevice,
            HostResidency::Locked if self.lock_quota_exhausted => SettleAction::LeavePageable,
            HostResidency::Locked => SettleAction::LockResident,
            HostResidency::Pageable => SettleAction::LeavePageable,
        }
    }

    /// Record what one of layer `layer_id`'s banks actually settled at.
    ///
    /// A layer is several banks and is only as resident as its worst
    /// one, so once a layer has recorded [`HostResidency::Pageable`] no
    /// later, better, report can raise it again.
    ///
    /// # Panics
    /// If `layer_id` is not a layer of this plan.
    pub fn record(&mut self, layer_id: usize, achieved: HostResidency) {
        if self.achieved[layer_id] != Some(HostResidency::Pageable) {
            self.achieved[layer_id] = Some(achieved);
        }
    }

    /// Record the outcome of one lock attempt: `locked` false means the
    /// bank is pageable *and* the quota is spent for the rest of the
    /// load.
    ///
    /// # Panics
    /// If `layer_id` is not a layer of this plan.
    pub fn record_lock(&mut self, layer_id: usize, locked: bool) {
        if !locked {
            self.lock_quota_exhausted = true;
        }
        self.record(
            layer_id,
            if locked {
                HostResidency::Locked
            } else {
                HostResidency::Pageable
            },
        );
    }

    /// What layer `layer_id` settled at, or [`None`] if no bank of it
    /// reported.
    ///
    /// # Panics
    /// If `layer_id` is not a layer of this plan.
    pub fn achieved(&self, layer_id: usize) -> Option<HostResidency> {
        self.achieved[layer_id]
    }

    /// The labels to hand [`BankResidency::new`]: what was achieved
    /// where anything reported, what was requested elsewhere.
    ///
    /// A pinned layer never reports, because a failed *pin* is a hard
    /// error the loader raises rather than a downgrade -- there is no
    /// second way to serve a GPU layer.
    pub fn achieved_labels(&self) -> Vec<HostResidency> {
        self.requested
            .iter()
            .zip(&self.achieved)
            .map(|(requested, achieved)| achieved.unwrap_or(*requested))
            .collect()
    }

    /// The layers that did not get what they asked for, for the one
    /// warning a human needs to see: they still decode on the CPU
    /// executor, but they may now swap under memory pressure.
    pub fn downgraded(&self) -> Vec<u32> {
        self.achieved_labels()
            .iter()
            .zip(&self.requested)
            .enumerate()
            .filter(|(_, (achieved, requested))| achieved != requested)
            .map(|(layer, _)| layer as u32)
            .collect()
    }
}

/// A residency configuration that cannot be served, refused before the
/// banks are attached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidencyError {
    /// The label list does not describe this model.
    LabelCountMismatch { labels: usize, num_layers: usize },
    /// Layers with no device address that nothing routed to the CPU.
    UnpinnedLayerNotOnCpu { layers: Vec<u32> },
    /// Prefill overlap asked for alongside a layer that cannot feed it.
    PrefillOverlapWithUnpinned { layers: Vec<u32> },
    /// A slot-remapping copy staged against a layer with no device
    /// alias for its host rows.
    SlotRemapOnUnpinnedLayer { layer: u32 },
}

impl std::fmt::Display for ResidencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResidencyError::LabelCountMismatch { labels, num_layers } => write!(
                f,
                "{labels} residency labels for a model of {num_layers} MoE layers"
            ),
            ResidencyError::UnpinnedLayerNotOnCpu { layers } => write!(
                f,
                "layers {layers:?} are not page-locked for the device and are not CPU layers: \
                 a layer without a device address can only decode on the CPU executor, so the \
                 CPU-layer set must be decided before the banks are attached"
            ),
            ResidencyError::PrefillOverlapWithUnpinned { layers } => write!(
                f,
                "prefill overlap DMAs from registered banks; it must be disabled when any layer \
                 is locked or pageable (layers {layers:?})"
            ),
            ResidencyError::SlotRemapOnUnpinnedLayer { layer } => write!(
                f,
                "layer {layer} is not page-locked for the device: its only copy is the \
                 whole-layer materialize (position == expert id); an LRU slot remap cannot be \
                 honored without a device alias for the host rows"
            ),
        }
    }
}

impl std::error::Error for ResidencyError {}

/// How a staged copy for one layer may be carried out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyRoute {
    /// The normal path: gather the missing expert rows from the
    /// registered bank into whichever slots the LRU picked.
    DeviceIndexed,
    /// A synchronous pageable copy of the whole layer into slots
    /// `[0, num_experts)`, `position == expert id`. Never captured in
    /// a CUDA graph: prefill is not captured, and decode never reaches
    /// this branch because an unpinned layer routes to the CPU
    /// executor.
    WholeLayerPageable,
}

/// The validated per-layer residency of an attached set of expert
/// banks.
///
/// Constructing one is the check; holding one is the proof the three
/// invariants in the module docs hold for this configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankResidency {
    labels: Vec<HostResidency>,
    unpinned: BTreeSet<u32>,
}

impl BankResidency {
    /// Attach `labels` (the **achieved** ones) to a model of
    /// `num_layers` MoE layers.
    ///
    /// `cpu_layers` is the set already routed to the CPU executor and
    /// `prefill_overlap` is whether the overlap path is still enabled.
    /// Both are inputs, not outputs: this refuses a bad combination
    /// rather than repairing it, because repairing it here would mean
    /// silently moving a layer to another executor after the cache
    /// geometry was already sized against the old answer.
    pub fn new(
        labels: &[HostResidency],
        num_layers: usize,
        cpu_layers: &BTreeSet<u32>,
        prefill_overlap: bool,
    ) -> Result<Self, ResidencyError> {
        if labels.len() != num_layers {
            return Err(ResidencyError::LabelCountMismatch {
                labels: labels.len(),
                num_layers,
            });
        }
        let unpinned: BTreeSet<u32> = labels
            .iter()
            .enumerate()
            .filter(|(_, label)| !label.is_device_addressable())
            .map(|(layer, _)| layer as u32)
            .collect();
        if !unpinned.is_empty() {
            let stranded: Vec<u32> = unpinned.difference(cpu_layers).copied().collect();
            if !stranded.is_empty() {
                return Err(ResidencyError::UnpinnedLayerNotOnCpu { layers: stranded });
            }
            if prefill_overlap {
                return Err(ResidencyError::PrefillOverlapWithUnpinned {
                    layers: unpinned.iter().copied().collect(),
                });
            }
        }
        Ok(Self {
            labels: labels.to_vec(),
            unpinned,
        })
    }

    /// The default when no plan was in force: every layer pinned, no
    /// invariant to check, prefill overlap free to stay on.
    pub fn all_pinned(num_layers: usize) -> Self {
        Self {
            labels: vec![HostResidency::Pinned; num_layers],
            unpinned: BTreeSet::new(),
        }
    }

    pub fn num_layers(&self) -> usize {
        self.labels.len()
    }

    pub fn labels(&self) -> &[HostResidency] {
        &self.labels
    }

    /// # Panics
    /// If `layer_id` is not a layer of this model.
    pub fn label(&self, layer_id: u32) -> HostResidency {
        self.labels[layer_id as usize]
    }

    /// Layers with no device address. The copy plan skips their rows
    /// entirely.
    pub fn unpinned_layers(&self) -> &BTreeSet<u32> {
        &self.unpinned
    }

    pub fn is_unpinned(&self, layer_id: u32) -> bool {
        self.unpinned.contains(&layer_id)
    }

    /// Whether any layer is locked or pageable -- the same signal that
    /// must have turned prefill overlap off.
    pub fn has_unpinned(&self) -> bool {
        !self.unpinned.is_empty()
    }

    /// How a copy staged for `layer_id` must be carried out.
    ///
    /// `whole_layer` is whether the caller staged the whole-layer
    /// materialize (every expert, slot `e` for expert `e`) rather than
    /// an LRU `ensure`. An unpinned layer accepts only the former; a
    /// pinned layer takes the indexed device path either way, since a
    /// registered bank can serve both.
    pub fn copy_route(
        &self,
        layer_id: u32,
        whole_layer: bool,
    ) -> Result<CopyRoute, ResidencyError> {
        if !self.is_unpinned(layer_id) {
            return Ok(CopyRoute::DeviceIndexed);
        }
        if whole_layer {
            Ok(CopyRoute::WholeLayerPageable)
        } else {
            Err(ResidencyError::SlotRemapOnUnpinnedLayer { layer: layer_id })
        }
    }
}

/// Overrides the pin budget on any host, in gibibytes. Empty means
/// unset.
///
/// Only the name differs from the reference, which spells it
/// `FREETOKEN_PIN_BUDGET_GB`; the value and every rule around it are
/// the same.
pub const PIN_BUDGET_ENV: &str = "FRINK_PIN_BUDGET_GB";

/// The tag WSL puts in its kernel release string. Matched
/// case-insensitively, as a substring: the surrounding version text
/// differs per WSL build and per distribution kernel.
pub const WSL_KERNEL_TAG: &str = "microsoft";

/// Fraction of physical RAM a WSL host may page-lock.
///
/// WDDM-backed CUDA caps pinning near *half* of RAM, and that ceiling
/// is shared across every process on the machine -- so 40% leaves room
/// for whatever else on the host has pinned memory. Taking the full
/// half would make the budget correct only on an otherwise idle
/// machine, and wrong exactly when the machine is busy.
pub const WSL_PIN_FRACTION: f64 = 0.4;

const GIB: f64 = (1u64 << 30) as f64;

/// A [`PIN_BUDGET_ENV`] value that is not a number of gibibytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinBudgetEnvError {
    pub value: String,
}

impl std::fmt::Display for PinBudgetEnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{PIN_BUDGET_ENV}={:?} is not a number of GiB",
            self.value
        )
    }
}

impl std::error::Error for PinBudgetEnvError {}

/// Read a [`PIN_BUDGET_ENV`] value: [`None`] for unset (or empty),
/// otherwise the budget in bytes.
///
/// A negative figure clamps to zero, which is a real answer -- "pin
/// nothing" -- and is how a deployment forces every MoE layer onto the
/// CPU executor. Text that is not a number at all is refused instead of
/// ignored: ignoring it would silently uncap the host the variable was
/// set to cap, and the failure then happens after the whole checkpoint
/// has been read.
pub fn parse_pin_budget_gb(value: &str) -> Result<Option<u64>, PinBudgetEnvError> {
    let text = value.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let gb: f64 = text.parse().map_err(|_| PinBudgetEnvError {
        value: value.to_string(),
    })?;
    if !gb.is_finite() {
        return Err(PinBudgetEnvError {
            value: value.to_string(),
        });
    }
    Ok(Some((gb * GIB).max(0.0) as u64))
}

/// Whether this host caps how much memory a process may page-lock for
/// the device.
///
/// The one platform that does is WSL, told by the `microsoft` tag its
/// kernel release carries. A release string we could not read at all
/// is empty, which reads as "not WSL" -- the same answer the reference
/// gives on a platform with no `uname`.
pub fn is_pin_capped_host(kernel_release: &str) -> bool {
    kernel_release.to_ascii_lowercase().contains(WSL_KERNEL_TAG)
}

/// Bytes this process may safely page-lock, from the host facts alone.
///
/// [`None`] means *uncapped*, not unknown: on plain Linux nothing caps
/// pinning, so every layer stays on the GPU path and
/// [`crate::placement::auto_cpu_layers`] hands out nothing.
///
/// On a capped host with `phys_ram_bytes == 0` -- the figure could not
/// be read -- the answer is `Some(0)`, "pin nothing". That is the
/// conservative direction: it costs throughput (every layer decodes on
/// the CPU) where returning [`None`] would claim an uncapped host and
/// die in the page-lock call after the whole checkpoint has been read.
/// [`PIN_BUDGET_ENV`] is the way out of it.
pub fn pin_budget_bytes(kernel_release: &str, phys_ram_bytes: u64) -> Option<u64> {
    if !is_pin_capped_host(kernel_release) {
        return None;
    }
    Some((phys_ram_bytes as f64 * WSL_PIN_FRACTION) as u64)
}

/// The pin budget with the environment override applied.
///
/// The override wins **anywhere**, including on a host this would
/// otherwise call uncapped: it is how a machine with an out-of-band pin
/// consumer (another process holding registered memory, a hypervisor)
/// tells the engine what is actually left.
pub fn resolve_pin_budget(
    kernel_release: &str,
    phys_ram_bytes: u64,
    env_value: Option<&str>,
) -> Result<Option<u64>, PinBudgetEnvError> {
    if let Some(value) = env_value {
        if let Some(bytes) = parse_pin_budget_gb(value)? {
            return Ok(Some(bytes));
        }
    }
    Ok(pin_budget_bytes(kernel_release, phys_ram_bytes))
}

/// This host's kernel release, or [`None`] where there is nothing to
/// read.
///
/// `/proc/sys/kernel/osrelease` is the file form of `uname -r`; the
/// reference calls `os.uname()`. A host without it is not WSL, which
/// is the answer either way.
pub fn host_kernel_release() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .ok()
        .map(|release| release.trim().to_string())
        .filter(|release| !release.is_empty())
}

/// This host's physical RAM, or [`None`] where it cannot be read.
///
/// `MemTotal` from `/proc/meminfo`, which is the file form of
/// `sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE)`; reading it keeps
/// this crate free of a libc dependency.
pub fn host_phys_ram_bytes() -> Option<u64> {
    parse_mem_total_bytes(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

fn parse_mem_total_bytes(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|line| line.starts_with("MemTotal:"))?;
    let mut fields = line.split_whitespace().skip(1);
    let value: u64 = fields.next()?.parse().ok()?;
    let scale = match fields.next() {
        None => 1,
        Some("kB") | Some("KB") | Some("kb") => 1024,
        Some(_) => return None,
    };
    value.checked_mul(scale)
}

/// This host's pin budget: [`PIN_BUDGET_ENV`], else the platform rule
/// applied to what `/proc` reports.
///
/// The convenience wrapper around [`resolve_pin_budget`] -- everything
/// it decides is in that function, so a caller that already knows the
/// host facts (a test, a remote sizing pass) should call that instead.
pub fn host_pin_budget_bytes() -> Result<Option<u64>, PinBudgetEnvError> {
    let env_value = std::env::var(PIN_BUDGET_ENV).ok();
    resolve_pin_budget(
        &host_kernel_release().unwrap_or_default(),
        host_phys_ram_bytes().unwrap_or(0),
        env_value.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::auto_cpu_layers;

    fn set(items: &[u32]) -> BTreeSet<u32> {
        items.iter().copied().collect()
    }

    #[test]
    fn a_label_survives_a_round_trip_through_its_wire_form() {
        for label in [
            HostResidency::Pinned,
            HostResidency::Locked,
            HostResidency::Pageable,
        ] {
            assert_eq!(HostResidency::from_label(label.as_str()), Some(label));
        }
        assert_eq!(HostResidency::from_label("registered"), None);
        assert!(HostResidency::Pinned.is_device_addressable());
        assert!(!HostResidency::Locked.is_device_addressable());
        assert!(!HostResidency::Pageable.is_device_addressable());
    }

    #[test]
    fn the_cpu_layers_are_the_ones_asked_to_lock() {
        assert_eq!(
            requested_labels(4, &set(&[0, 3])),
            vec![
                HostResidency::Locked,
                HostResidency::Pinned,
                HostResidency::Pinned,
                HostResidency::Locked,
            ]
        );
        assert_eq!(
            requested_labels(3, &BTreeSet::new()),
            vec![HostResidency::Pinned; 3]
        );
    }

    #[test]
    fn a_plan_settles_each_layer_at_the_class_it_asked_for() {
        let mut plan = ResidencyPlan::new(requested_labels(3, &set(&[1])));
        assert!(!plan.applied());
        assert_eq!(plan.settle_action(0), SettleAction::PageLockForDevice);
        assert_eq!(plan.settle_action(1), SettleAction::LockResident);
        assert!(plan.applied());
        assert!(plan.has_unpinned());
        assert!(!ResidencyPlan::all_pinned(3).has_unpinned());
    }

    /// The naive version assumes the lock it asked for succeeded and
    /// labels the layer `Locked`. The bank is pageable, and the label
    /// promises residency the kernel never granted -- so the swap-in
    /// lands mid-decode with nothing to explain it.
    #[test]
    fn a_failed_lock_is_recorded_as_pageable_rather_than_assumed_locked() {
        let mut plan = ResidencyPlan::new(requested_labels(3, &set(&[0, 2])));
        assert_eq!(plan.settle_action(0), SettleAction::LockResident);
        plan.record_lock(0, false);
        assert_eq!(plan.settle_action(2), SettleAction::LeavePageable);
        plan.record(2, HostResidency::Pageable);
        assert_eq!(plan.achieved(0), Some(HostResidency::Pageable));
        assert_eq!(
            plan.achieved_labels(),
            vec![
                HostResidency::Pageable,
                HostResidency::Pinned,
                HostResidency::Pageable,
            ],
            "the quota is spent for good, so layer 2's lock never even ran"
        );
        assert_eq!(plan.downgraded(), vec![0, 2]);
    }

    /// A pinned layer never reports, because a failed *pin* is a hard
    /// error rather than a downgrade: its requested label is what it
    /// achieved.
    #[test]
    fn an_unreported_layer_echoes_back_what_it_asked_for() {
        let mut plan = ResidencyPlan::new(requested_labels(2, &set(&[1])));
        plan.settle_action(1);
        plan.record_lock(1, true);
        assert_eq!(
            plan.achieved_labels(),
            vec![HostResidency::Pinned, HostResidency::Locked]
        );
        assert!(plan.downgraded().is_empty());
    }

    /// A layer is several banks and is only as resident as its worst
    /// one: taking the last report would let a later bank's success
    /// erase an earlier bank's failure.
    #[test]
    fn one_pageable_bank_downgrades_the_whole_layer() {
        let mut plan = ResidencyPlan::all_pinned(1);
        plan.record(0, HostResidency::Pageable);
        plan.record(0, HostResidency::Locked);
        plan.record(0, HostResidency::Pinned);
        assert_eq!(plan.achieved(0), Some(HostResidency::Pageable));
    }

    /// The lock ceiling is a per-process quota, so once one request is
    /// over it every larger cumulative request is too: retrying per
    /// bank buys nothing and costs a failing syscall per bank for the
    /// rest of the load.
    #[test]
    fn a_spent_lock_quota_leaves_every_later_layer_pageable() {
        let mut plan = ResidencyPlan::new(vec![HostResidency::Locked; 3]);
        assert_eq!(plan.settle_action(0), SettleAction::LockResident);
        plan.record_lock(0, false);
        assert!(plan.lock_quota_exhausted());
        assert_eq!(plan.settle_action(1), SettleAction::LeavePageable);
        assert_eq!(plan.settle_action(2), SettleAction::LeavePageable);
    }

    #[test]
    fn labels_that_do_not_describe_the_model_are_refused() {
        assert_eq!(
            BankResidency::new(&[HostResidency::Pinned; 3], 4, &BTreeSet::new(), false),
            Err(ResidencyError::LabelCountMismatch {
                labels: 3,
                num_layers: 4
            })
        );
    }

    /// The naive version attaches the banks and lets the layer's first
    /// decode step index a device pointer that was never registered.
    /// The relationship is checked here, at attach time, before a byte
    /// moves.
    #[test]
    fn a_non_pinned_layer_that_is_not_a_cpu_layer_is_refused() {
        let labels = requested_labels(4, &set(&[0, 3]));
        assert_eq!(
            BankResidency::new(&labels, 4, &set(&[0]), false),
            Err(ResidencyError::UnpinnedLayerNotOnCpu { layers: vec![3] })
        );
        // Both routed to the CPU: fine.
        let attached = BankResidency::new(&labels, 4, &set(&[0, 3]), false).unwrap();
        assert_eq!(attached.unpinned_layers(), &set(&[0, 3]));
        // A CPU layer whose bank did get pinned is not a violation --
        // it reads a registered bank perfectly well, it just saved no
        // pin quota.
        assert!(
            BankResidency::new(&[HostResidency::Pinned; 4], 4, &set(&[0, 3]), true).is_ok(),
            "an all-pinned load keeps prefill overlap even with CPU layers"
        );
    }

    /// The naive version leaves the overlap on and lets it degrade per
    /// layer. It cannot: the overlap path DMAs from a registered bank,
    /// and a locked bank has no device address to DMA from, so the
    /// configuration is refused and the caller turns the overlap off
    /// on the same signal.
    #[test]
    fn prefill_overlap_with_any_unpinned_layer_is_refused() {
        let labels = requested_labels(4, &set(&[2]));
        assert_eq!(
            BankResidency::new(&labels, 4, &set(&[2]), true),
            Err(ResidencyError::PrefillOverlapWithUnpinned { layers: vec![2] })
        );
        assert!(BankResidency::new(&labels, 4, &set(&[2]), false).is_ok());
    }

    /// The naive version honors the LRU's chosen victim slot for an
    /// unpinned layer. It cannot: only the whole-layer materialize's
    /// `position == expert id` mapping works without a device alias for
    /// the host rows, so a slot remap is an error rather than something
    /// to quietly copy somewhere else.
    #[test]
    fn an_unpinned_layer_accepts_only_the_whole_layer_materialize() {
        let labels = requested_labels(3, &set(&[1]));
        let banks = BankResidency::new(&labels, 3, &set(&[1]), false).unwrap();
        assert_eq!(banks.copy_route(1, true), Ok(CopyRoute::WholeLayerPageable));
        assert_eq!(
            banks.copy_route(1, false),
            Err(ResidencyError::SlotRemapOnUnpinnedLayer { layer: 1 })
        );
    }

    /// A registered bank serves both shapes of copy, so a pinned layer
    /// takes the indexed device path even for a whole-layer stage.
    #[test]
    fn a_pinned_layer_takes_the_indexed_device_copy_either_way() {
        let banks = BankResidency::all_pinned(3);
        assert!(!banks.has_unpinned());
        assert_eq!(banks.copy_route(0, true), Ok(CopyRoute::DeviceIndexed));
        assert_eq!(banks.copy_route(0, false), Ok(CopyRoute::DeviceIndexed));
        assert_eq!(banks.label(0), HostResidency::Pinned);
    }

    /// End to end: a lock fails, the achieved labels say so, and the
    /// attach validates against those -- not against what was asked
    /// for. The downgraded layer is still a CPU layer, so it still
    /// attaches; it is merely pageable now.
    #[test]
    fn the_achieved_labels_are_what_the_banks_attach_with() {
        let cpu_layers = set(&[0, 5]);
        let mut plan = ResidencyPlan::new(requested_labels(6, &cpu_layers));
        assert_eq!(plan.settle_action(0), SettleAction::LockResident);
        plan.record_lock(0, false);
        assert_eq!(plan.settle_action(5), SettleAction::LeavePageable);
        plan.record(5, HostResidency::Pageable);
        let labels = plan.achieved_labels();
        assert_eq!(labels[0], HostResidency::Pageable);
        assert_eq!(labels[5], HostResidency::Pageable);
        let banks = BankResidency::new(&labels, 6, &cpu_layers, false).unwrap();
        assert_eq!(banks.unpinned_layers(), &cpu_layers);
        assert_eq!(banks.label(0), HostResidency::Pageable);
    }

    /// The naive version reports a WSL host as uncapped, which is the
    /// one platform where the cap bites: `auto_cpu_layers` then hands
    /// out no CPU layers and the load dies inside the page-lock call
    /// after the whole checkpoint has been read.
    #[test]
    fn a_wsl_host_is_capped_at_forty_percent_of_ram() {
        let release = "5.15.153.1-microsoft-standard-WSL2";
        assert!(is_pin_capped_host(release));
        assert_eq!(
            pin_budget_bytes(release, 64 << 30),
            Some((64 << 30) * 2 / 5)
        );
        // Case-insensitive: distribution kernels spell the tag either way.
        assert!(is_pin_capped_host("5.10.16.3-Microsoft-standard-WSL2"));
    }

    /// And the consequence the budget exists for: with no budget every
    /// layer stays pinned on the GPU path; with the WSL budget the
    /// over-cap model gives layers to the CPU instead.
    #[test]
    fn the_wsl_budget_is_what_moves_layers_off_the_gpu_path() {
        let banks = 48u64 << 30;
        let budget = pin_budget_bytes("5.15.153.1-microsoft-standard-WSL2", 64 << 30);
        assert!(
            auto_cpu_layers(48, banks, None).is_empty(),
            "an uncapped host keeps everything pinned"
        );
        assert!(!auto_cpu_layers(48, banks, budget).is_empty());
    }

    #[test]
    fn plain_linux_reports_no_pin_cap() {
        assert!(!is_pin_capped_host("6.8.0-45-generic"));
        assert_eq!(pin_budget_bytes("6.8.0-45-generic", 64 << 30), None);
        assert_eq!(pin_budget_bytes("", 64 << 30), None, "no readable release");
        assert_eq!(
            resolve_pin_budget("6.8.0-45-generic", 64 << 30, None),
            Ok(None)
        );
    }

    /// A capped host whose RAM figure could not be read must not come
    /// back uncapped: budgeting nothing costs throughput, claiming no
    /// cap costs the whole load.
    #[test]
    fn an_unreadable_ram_figure_on_a_capped_host_budgets_nothing() {
        assert_eq!(pin_budget_bytes("microsoft-standard-WSL2", 0), Some(0));
        assert_eq!(auto_cpu_layers(8, 1 << 30, Some(0)).len(), 8);
    }

    #[test]
    fn the_environment_variable_overrides_on_any_host() {
        assert_eq!(
            resolve_pin_budget("6.8.0-45-generic", 64 << 30, Some("8")),
            Ok(Some(8 << 30)),
            "an uncapped host can still be told a budget"
        );
        assert_eq!(
            resolve_pin_budget("microsoft-standard-WSL2", 64 << 30, Some("1.5")),
            Ok(Some(1024 * 1024 * 1024 * 3 / 2)),
            "and a capped host's computed budget is replaced, not clamped"
        );
        assert_eq!(
            resolve_pin_budget("6.8.0-45-generic", 64 << 30, Some("-1")),
            Ok(Some(0)),
            "a negative budget means pin nothing"
        );
    }

    #[test]
    fn an_empty_pin_budget_variable_counts_as_unset() {
        assert_eq!(parse_pin_budget_gb(""), Ok(None));
        assert_eq!(parse_pin_budget_gb("   "), Ok(None));
        assert_eq!(
            resolve_pin_budget("microsoft-standard-WSL2", 64 << 30, Some("")),
            Ok(Some((64 << 30) * 2 / 5)),
            "an empty value in a unit file means 'use the normal rule'"
        );
    }

    /// The naive version ignores a value it cannot read, which
    /// silently uncaps the very host the variable was set to cap.
    #[test]
    fn an_unparsable_pin_budget_is_refused_rather_than_ignored() {
        for value in ["eight", "8GiB", "inf", "NaN"] {
            assert_eq!(
                parse_pin_budget_gb(value),
                Err(PinBudgetEnvError {
                    value: value.to_string()
                }),
                "{value:?} must not read as 'no cap'"
            );
        }
        assert!(resolve_pin_budget("6.8.0-45-generic", 64 << 30, Some("eight")).is_err());
    }

    #[test]
    fn mem_total_is_read_in_kilobytes() {
        let meminfo = "MemTotal:       65809172 kB\nMemFree:         1234 kB\n";
        assert_eq!(parse_mem_total_bytes(meminfo), Some(65809172 * 1024));
        assert_eq!(parse_mem_total_bytes("MemFree: 1234 kB\n"), None);
        assert_eq!(parse_mem_total_bytes("MemTotal: notanumber kB"), None);
        assert_eq!(parse_mem_total_bytes("MemTotal: 12 MB"), None);
    }
}
