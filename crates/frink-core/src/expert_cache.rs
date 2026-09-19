//! The global expert slot cache: which experts are resident on the GPU
//! right now, and what one decode step has to move.
//!
//! # One id space, one pool
//!
//! Experts are addressed by a **flat id**, `layer * num_experts +
//! expert`, and all layers compete for one pool of `cache_size` slots.
//! That is the "global" in global LRU, and it is the point: expert
//! activation is not uniform across layers, so a per-layer cache of
//! `cache_size / num_layers` slots wastes residency on layers whose
//! routing is flat and starves the layers where a few experts take most
//! of the traffic.
//!
//! # What a step produces
//!
//! [`ExpertCache::ensure`] takes the experts a layer routed to and
//! returns, for each, the slot it will be read from -- plus a
//! [`CopyPlan`]: the (slot, host row) pairs the caller must copy before
//! the multiply. Nothing here moves bytes; the plan is the whole
//! output.
//!
//! # Two rules that must not drift
//!
//! - **Victims are chosen by `(usage, slot)` ascending**, and a slot
//!   touched *by this very step* is not a victim. Without the second
//!   rule a step that misses more experts than it hits can evict an
//!   expert it is about to read.
//! - **Duplicate routes collapse.** A batch routing twice to the same
//!   expert counts one miss, issues one copy, and shares one slot.
//!
//! The hybrid entry point adds the [`crate::qstar`] split on top: only
//! the first `fetch` misses get slots, and the rest come back as
//! [`None`], meaning "compute this one on the CPU". Every route is
//! assigned exactly once, to exactly one device.
//!
//! # Reading the cache back
//!
//! [`ExpertCache::stats`] answers "is the cache big enough?" for the
//! model as a whole. [`ExpertCache::layer_stats`] answers it per MoE
//! layer, which is the only form that can tell *one layer thrashing*
//! from *every layer uniformly a little over budget*: both show the
//! same global miss rate and they have opposite fixes -- rebalance
//! versus grow. [`ExpertCache::routing_skew`] goes one level below the
//! cache and reports what the routing distribution itself allows; a
//! layer whose `oracle_hit_at_slots` is already low cannot be helped by
//! any cache size, because no policy could do better on that traffic.
//!
//! # The prefill double buffers
//!
//! A prefill chunk walks the layers in order, so it can stage layer
//! `L + 1`'s experts while layer `L` computes. The two staging buffers
//! **borrow** slots `[0, 2 * num_experts)` of this very cache and
//! rotate by `layer % 2`. Borrowed is not owned: the bytes in those
//! slots are rewritten every other layer within a chunk, so a slot at
//! or below `2 * num_experts` can never be read as residency, and the
//! buffers' contents are never registered in the residency map.
//! [`ExpertCache::prefetch_prefill_layer`] is the whole portable half
//! of that -- which expert rows are already on the device, which have
//! to cross the link, and what the borrowed slots do to the LRU. No
//! streams, no events and no copies live here.
//!
//! Ported 1:1 from FreeToken's `moe/offload_cache.py` and
//! `moe/offload_kernels.py`, whose LRU is the `lru_ensure` kernel from
//! `flashlib` (both Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

use crate::qstar::QStarPolicy;

/// A cached expert's address in the flat id space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertId {
    pub layer: u32,
    pub expert: u32,
}

/// The copies a step needs before it can multiply.
///
/// `dst_slots[i]` is a cache slot; `src_rows[i]` is a **layer-local**
/// expert row, so it indexes this layer's own host bank rather than a
/// flat table. Keeping it layer-local is what lets each layer's weights
/// live in their own allocation, which in turn is what lets residency
/// (pinned / locked / pageable) differ per layer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CopyPlan {
    pub dst_slots: Vec<u32>,
    pub src_rows: Vec<u32>,
}

impl CopyPlan {
    pub fn len(&self) -> usize {
        self.dst_slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dst_slots.is_empty()
    }
}

/// What one `ensure` decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnsurePlan {
    /// Per routed expert, in the caller's order: the slot to read it
    /// from, or `None` for "compute this one on the CPU" (hybrid only).
    pub slots: Vec<Option<u32>>,
    pub copy: CopyPlan,
    /// Distinct experts this step routed to.
    pub active: usize,
    /// Distinct experts that were not resident -- *before* any fetch
    /// cap was applied.
    pub missing: usize,
    /// Distinct experts actually being fetched over the link.
    pub fetched: usize,
}

/// Which miss gets the scarce fetch slots when the split caps them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FetchOrder {
    /// Most-recently-active expert first. An expert that keeps being
    /// routed to but keeps losing the cap ranks higher every step until
    /// it wins one, so a hot expert converges into residency instead of
    /// being computed on the CPU forever.
    #[default]
    ByRecency,
    /// Lowest expert id first. Deterministic and cheap; kept because it
    /// is the reference order the two implementations were first
    /// cross-checked against.
    LowestId,
}

/// Running counters, for `/metrics` and for deciding whether the cache
/// is sized right.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertCacheStats {
    /// `ensure` calls.
    pub calls: u64,
    /// Distinct experts routed to, summed over calls.
    pub active: u64,
    /// Distinct misses, summed over calls, before any cap.
    pub missing: u64,
    /// Misses actually fetched, summed over calls.
    pub fetched: u64,
}

impl ExpertCacheStats {
    /// Fraction of routed experts that were not resident. The number
    /// that says whether the cache is big enough.
    pub fn miss_rate(&self) -> f64 {
        if self.active == 0 {
            return 0.0;
        }
        self.missing as f64 / self.active as f64
    }

    /// Fraction of misses served over the link rather than by the CPU.
    /// The number that says what the `q*` split actually did.
    pub fn fetch_rate(&self) -> f64 {
        if self.missing == 0 {
            return 0.0;
        }
        self.fetched as f64 / self.missing as f64
    }

    pub fn active_per_call(&self) -> f64 {
        if self.calls == 0 {
            return 0.0;
        }
        self.active as f64 / self.calls as f64
    }

    /// Misses per call, before the cap. Quoted beside
    /// [`active_per_call`](Self::active_per_call) because the pair is
    /// what sizes a link budget: `missing_per_call * bytes_per_expert`
    /// is what one step wants to move.
    pub fn missing_per_call(&self) -> f64 {
        if self.calls == 0 {
            return 0.0;
        }
        self.missing as f64 / self.calls as f64
    }

    /// Misses per call that actually crossed the link. Under the `q*`
    /// split this is the one the link sees; `missing_per_call` minus
    /// this is what the CPU absorbed.
    pub fn fetched_per_call(&self) -> f64 {
        if self.calls == 0 {
            return 0.0;
        }
        self.fetched as f64 / self.calls as f64
    }
}

/// One MoE layer's routing concentration over the observed decode
/// histogram.
///
/// Every field is derived from that layer's histogram row alone, so a
/// layer that was never routed to has no entry at all rather than a row
/// of zeros -- a zero working set and a zero oracle hit rate would read
/// as "this layer is unservable" when the truth is "no data".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerRoutingSkew {
    pub layer: u32,
    /// Total routes observed for this layer, counting duplicates.
    pub routed: u64,
    /// Distinct experts that were routed to at least once.
    pub working_set: usize,
    /// How many of the hottest experts it takes to cover 90% of this
    /// layer's routing mass.
    pub experts_for_90pct: usize,
    /// Routing entropy over `ln(num_experts)`. `1.0` is a perfectly
    /// flat router, `0.0` a layer that always picks the same expert.
    pub norm_entropy: f64,
    /// The top-`oracle_slots` share of this layer's routing mass: the
    /// best hit rate *any* per-layer policy could reach on the observed
    /// distribution, with no LRU/LFU dynamics in it at all.
    ///
    /// This is the number that separates the two diagnoses. A high
    /// oracle hit rate next to a low realized one
    /// ([`ExpertCacheStats::miss_rate`]) means the policy is losing
    /// residency it could have kept; a low oracle hit rate means the
    /// layer's routing is flat and no cache size will help it.
    pub oracle_hit_at_slots: f64,
}

/// The routing-skew report: per layer, plus the means over the layers
/// that were actually routed to.
///
/// The means deliberately exclude un-routed layers, matching the
/// per-layer list. Averaging zeros for layers that produced no traffic
/// would drag every aggregate toward zero in proportion to how much of
/// the model the window happened to touch, which makes two windows of
/// the same model incomparable.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingSkewReport {
    /// `cache_size / num_layers`, unrounded -- the fair share a
    /// per-layer cache would get.
    pub slots_per_layer: f64,
    /// The fair share as a usable slot count, `max(1, round(share))`.
    /// The oracle is quoted at this many slots.
    pub oracle_slots: usize,
    pub working_set_mean: f64,
    pub working_set_max: usize,
    pub experts_for_90pct: f64,
    pub oracle_hit_at_slots: f64,
    pub norm_entropy: f64,
    /// One entry per layer that was routed to, in layer order.
    pub per_layer: Vec<LayerRoutingSkew>,
}

/// The share of routing mass [`LayerRoutingSkew::experts_for_90pct`]
/// covers.
const COVERAGE_FRACTION: f64 = 0.9;

/// Per-expert row size below which a host bank ships its whole layer as
/// one transfer entry.
///
/// A batched host-to-device copy silently degrades to a *synchronous*
/// transfer when one batch mixes large entries with sub-256 KiB ones.
/// The bytes still move at full link rate, but the host stalls for the
/// whole transfer, which un-hides the GEMM that the prefetch was
/// supposed to overlap with. Banks whose per-expert row is smaller than
/// this ship as a single whole-layer entry -- their entire layer is
/// tiny -- so every per-run entry a batch sees stays above the floor.
pub const SMALL_BANK_FEAT_BYTES: u64 = 256 * 1024;

/// Marks a slot as unevictable for this step.
const UNEVICTABLE: i64 = i64::MAX;

/// A contiguous run of expert ids, `[start, start + len)`.
///
/// Misses are coalesced into runs because one transfer entry per run is
/// one descriptor and one large copy, where one entry per expert is
/// `num_experts` descriptors of one row each -- the same bytes at a
/// fraction of the achievable rate, and enough small entries to trip
/// the [`SMALL_BANK_FEAT_BYTES`] floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissRun {
    pub start: u32,
    pub len: u32,
}

/// Device-to-device rows: what the prefill buffer can take from the
/// cache instead of from the host.
///
/// `dst_slots[i]` is an absolute cache slot inside the buffer's borrowed
/// range; `src_slots[i]` is the absolute cache slot the expert is
/// already resident in. Both are slots -- unlike [`CopyPlan`], nothing
/// here indexes a host bank.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GatherPlan {
    pub dst_slots: Vec<u32>,
    pub src_slots: Vec<u32>,
}

impl GatherPlan {
    pub fn len(&self) -> usize {
        self.dst_slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dst_slots.is_empty()
    }
}

/// One host-to-device transfer entry: `rows` consecutive expert rows of
/// one bank.
///
/// `dst_slot` is the absolute cache slot the run lands on (inside the
/// buffer's borrowed range) and `src_row` is the **layer-local** expert
/// row it comes from, the same convention as [`CopyPlan::src_rows`], so
/// each layer's bank can live in its own allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankEntry {
    /// Index into the `bank_feat_bytes` the caller passed, in that
    /// order.
    pub bank: usize,
    pub dst_slot: u32,
    pub src_row: u32,
    pub rows: u32,
    pub bytes: u64,
    /// Set when this entry is the whole layer because the bank is
    /// small ([`SMALL_BANK_FEAT_BYTES`]) rather than because the whole
    /// layer missed.
    pub whole_layer: bool,
}

/// What staging one layer into a prefill double buffer costs.
///
/// The two row sets are disjoint by construction -- an expert is either
/// gathered device-side or shipped from the host, never both -- which is
/// what lets the gather and the host transfer run without ordering
/// against each other.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillPlan {
    pub layer: u32,
    /// `layer % 2`.
    pub buffer_id: u32,
    /// The buffer already held this layer, so nothing moves. Every
    /// other field is empty.
    pub already_loaded: bool,
    /// Resident rows, cache -> buffer, device-side.
    pub gather: GatherPlan,
    /// Non-resident rows, coalesced.
    pub miss_runs: Vec<MissRun>,
    /// The host transfer batch, grouped by bank in the caller's order.
    pub entries: Vec<BankEntry>,
    /// Banks the gather serves -- those at or above
    /// [`SMALL_BANK_FEAT_BYTES`]. A small bank's rows are covered by its
    /// whole-layer entry instead, so gathering them too would move the
    /// same bytes twice.
    pub gather_banks: Vec<usize>,
}

/// The GPU expert cache's residency map.
#[derive(Debug)]
pub struct ExpertCache {
    num_layers: usize,
    num_experts: usize,
    cache_size: usize,
    /// Flat id -> slot, or -1.
    slot_for_id: Vec<i64>,
    /// Slot -> flat id, or -1 for empty.
    id_of_slot: Vec<i64>,
    /// Slot -> the step that last touched it. The LRU clock.
    usage: Vec<i64>,
    step: i64,
    /// Flat id -> the step this expert was last *routed to*, resident
    /// or not. Distinct from `usage`, which only tracks residency.
    recency: Vec<i64>,
    fetch_order: FetchOrder,
    stats: ExpertCacheStats,
    /// The same counters as `stats`, attributed to the layer that
    /// produced them. Indexed by MoE layer id.
    layer_stats: Vec<ExpertCacheStats>,
    collect_routing: bool,
    /// Flat id -> how many times it was routed to, duplicates counted.
    routing_freq: Vec<u64>,
    /// Which layer each prefill buffer currently stages, if any.
    prefill_layer: [Option<u32>; 2],
    /// Whether the consumer is done reading each buffer.
    prefill_released: [bool; 2],
    /// `slot_for_id` as it stood when the chunk opened. See
    /// [`ExpertCache::begin_prefill`].
    prefill_snapshot: Option<Vec<i64>>,
    prefill_hit_rows: u64,
    prefill_total_rows: u64,
}

impl ExpertCache {
    /// A cold cache of `cache_size` slots shared by every layer.
    ///
    /// `cache_size` must hold at least one whole layer: a prefill
    /// materializes a layer's experts into slots `0..num_experts`, and
    /// a cache that cannot hold one layer cannot serve a prefill at
    /// all.
    pub fn new(num_layers: usize, num_experts: usize, cache_size: usize) -> Self {
        assert!(
            num_layers > 0 && num_experts > 0,
            "a MoE model has layers and experts"
        );
        assert!(
            cache_size >= num_experts,
            "cache_size {cache_size} cannot hold one layer of {num_experts} experts"
        );
        let total_ids = num_layers * num_experts;
        ExpertCache {
            num_layers,
            num_experts,
            cache_size,
            slot_for_id: vec![-1; total_ids],
            id_of_slot: vec![-1; cache_size],
            usage: vec![0; cache_size],
            step: 0,
            recency: vec![-1; total_ids],
            fetch_order: FetchOrder::default(),
            stats: ExpertCacheStats::default(),
            layer_stats: vec![ExpertCacheStats::default(); num_layers],
            collect_routing: false,
            routing_freq: vec![0; total_ids],
            prefill_layer: [None, None],
            prefill_released: [true, true],
            prefill_snapshot: None,
            prefill_hit_rows: 0,
            prefill_total_rows: 0,
        }
    }

    pub fn with_fetch_order(mut self, order: FetchOrder) -> Self {
        self.fetch_order = order;
        self
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn cache_size(&self) -> usize {
        self.cache_size
    }

    pub fn stats(&self) -> ExpertCacheStats {
        self.stats
    }

    /// The same counters as [`stats`](Self::stats), attributed to one
    /// MoE layer over the current `reset_stats`-delimited window.
    ///
    /// A single global miss rate cannot distinguish one layer thrashing
    /// from every layer being uniformly a little over budget: the two
    /// average out to the same number and want opposite fixes -- move
    /// slots between layers, versus give the pool more slots. This is
    /// the breakdown that tells them apart. Sums over every layer equal
    /// [`stats`](Self::stats) exactly.
    pub fn layer_stats(&self, layer: u32) -> ExpertCacheStats {
        self.layer_stats[layer as usize]
    }

    /// Every layer's counters at once, indexed by MoE layer id.
    pub fn per_layer_stats(&self) -> &[ExpertCacheStats] {
        &self.layer_stats
    }

    /// Close the current stats window and open a new one.
    ///
    /// The routing histogram is deliberately *not* cleared here: it
    /// estimates a distribution, and its value grows with the number of
    /// steps behind it, so a `/metrics` scrape that resets the rate
    /// counters must not also destroy it. Use
    /// [`reset_routing`](Self::reset_routing) when the distribution
    /// itself is what went stale.
    pub fn reset_stats(&mut self) {
        self.stats = ExpertCacheStats::default();
        for layer in &mut self.layer_stats {
            *layer = ExpertCacheStats::default();
        }
        self.prefill_hit_rows = 0;
        self.prefill_total_rows = 0;
    }

    /// Total routed experts across the model -- the denominator the
    /// cache's residency rate is quoted against.
    pub fn total_experts(&self) -> usize {
        self.num_layers * self.num_experts
    }

    fn flat(&self, layer: u32, expert: u32) -> usize {
        let layer = layer as usize;
        let expert = expert as usize;
        debug_assert!(layer < self.num_layers && expert < self.num_experts);
        layer * self.num_experts + expert
    }

    /// Which slot holds `(layer, expert)`, if any. For tests and
    /// reporting; the hot path uses [`ensure`](Self::ensure).
    pub fn slot_of(&self, layer: u32, expert: u32) -> Option<u32> {
        let slot = self.slot_for_id[self.flat(layer, expert)];
        (slot >= 0).then_some(slot as u32)
    }

    /// Which expert occupies `slot`, if any.
    pub fn resident_in(&self, slot: u32) -> Option<ExpertId> {
        let id = self.id_of_slot[slot as usize];
        (id >= 0).then(|| ExpertId {
            layer: (id as usize / self.num_experts) as u32,
            expert: (id as usize % self.num_experts) as u32,
        })
    }

    /// Slots currently holding an expert.
    pub fn resident_slots(&self) -> usize {
        self.id_of_slot.iter().filter(|id| **id >= 0).count()
    }

    /// Drop one slot's residency, returning what it used to hold.
    ///
    /// The map here is a record of copies the caller was *asked* to
    /// make, and a copy can fail after the map has recorded it -- see
    /// [`crate::expert_slots::SlotFault::Device`]. Without this the
    /// next step reads that slot as a hit and multiplies whatever the
    /// failed copy left in it; with it, the expert simply misses again
    /// and is re-fetched.
    ///
    /// The slot becomes the *first* eviction candidate rather than
    /// merely an empty one: its LRU stamp goes to the beginning of
    /// time, so a pool under pressure spends it before evicting an
    /// expert that is really there. Idempotent, and `None` for a slot
    /// that already held nothing.
    pub fn forget_slot(&mut self, slot: u32) -> Option<ExpertId> {
        let id = *self.id_of_slot.get(slot as usize)?;
        if id < 0 {
            return None;
        }
        self.id_of_slot[slot as usize] = -1;
        self.slot_for_id[id as usize] = -1;
        self.usage[slot as usize] = i64::MIN;
        Some(ExpertId {
            layer: (id as usize / self.num_experts) as u32,
            expert: (id as usize % self.num_experts) as u32,
        })
    }

    /// Forget everything. A rebuild is a cold start, so the counters go
    /// too -- carrying them across would skew every rate that is quoted
    /// per call.
    pub fn reset(&mut self) {
        self.slot_for_id.fill(-1);
        self.id_of_slot.fill(-1);
        self.usage.fill(0);
        self.recency.fill(-1);
        self.step = 0;
        self.reset_stats();
        self.reset_routing();
        self.forget_prefill_buffers();
    }

    /// Resize the pool, keeping the model geometry.
    ///
    /// Everything resident is dropped: slot ids are positions in an
    /// allocation that no longer exists, so keeping the map would point
    /// at other experts' bytes. Refusing an impossible target *before*
    /// touching anything is deliberate -- the caller can then keep
    /// serving from the cache it already has.
    pub fn rebuild(&mut self, cache_size: usize) -> Result<(), RebuildRejected> {
        if cache_size < self.num_experts {
            return Err(RebuildRejected {
                requested: cache_size,
                minimum: self.num_experts,
            });
        }
        self.cache_size = cache_size;
        self.id_of_slot = vec![-1; cache_size];
        self.usage = vec![0; cache_size];
        self.slot_for_id.fill(-1);
        self.recency.fill(-1);
        self.step = 0;
        self.reset_stats();
        // A rebuild changes the geometry the oracle is quoted against
        // (`cache_size / num_layers`), so a histogram carried across it
        // would be reported at slot counts it was never observed under.
        self.reset_routing();
        self.forget_prefill_buffers();
        Ok(())
    }

    /// Make every expert this layer routed to resident, evicting by LRU.
    ///
    /// Every route gets a slot: this is the pure-offload path, where
    /// the CPU computes nothing.
    pub fn ensure(&mut self, layer: u32, expert_ids: &[u32]) -> EnsurePlan {
        self.ensure_split(layer, expert_ids, None)
    }

    /// The bandwidth-adaptive path: fetch what the `q*` split says to
    /// fetch, and hand the rest back for the CPU.
    ///
    /// A route that comes back `None` is the caller's to compute from
    /// host RAM. Together with the `Some` routes it covers every
    /// routed expert exactly once -- which is what makes it safe to
    /// simply add the two partial results.
    pub fn ensure_hybrid(
        &mut self,
        layer: u32,
        expert_ids: &[u32],
        policy: &QStarPolicy,
    ) -> EnsurePlan {
        self.ensure_split(layer, expert_ids, Some(policy))
    }

    fn ensure_split(
        &mut self,
        layer: u32,
        expert_ids: &[u32],
        policy: Option<&QStarPolicy>,
    ) -> EnsurePlan {
        self.step += 1;
        let step = self.step;
        let base = self.flat(layer, 0);

        // Routing is observed here, before anything collapses or caps
        // it: the histogram must describe what the router asked for,
        // not what the cache decided to do about it. Duplicates count,
        // because a batch that sends two tokens to one expert is two
        // units of that expert's mass -- deduplicating would flatten
        // exactly the skew the report exists to measure.
        if self.collect_routing {
            for expert in expert_ids {
                self.routing_freq[base + *expert as usize] += 1;
            }
        }

        // Distinct routes, first occurrence wins. A batch that routes
        // twice to one expert must not fetch it twice or, worse, evict
        // its own first copy.
        let mut distinct: Vec<u32> = Vec::with_capacity(expert_ids.len());
        for id in expert_ids {
            if !distinct.contains(id) {
                distinct.push(*id);
            }
        }

        let mut missing: Vec<u32> = Vec::new();
        for expert in &distinct {
            let slot = self.slot_for_id[base + *expert as usize];
            if slot >= 0 {
                // A hit refreshes the slot, which also makes it
                // unevictable for the rest of this step.
                self.usage[slot as usize] = step;
            } else {
                missing.push(*expert);
            }
        }

        match (policy, self.fetch_order) {
            // The capped path prefers experts that were routed to most
            // recently, so a hot expert that keeps losing the cap wins
            // one eventually.
            (Some(_), FetchOrder::ByRecency) => {
                missing.sort_by_key(|e| (-self.recency[base + *e as usize], *e));
            }
            _ => missing.sort_unstable(),
        }

        let num_missing = missing.len();
        let num_fetch = match policy {
            Some(policy) => policy.split(num_missing).fetch,
            None => num_missing,
        };

        // Slots this step already committed to -- hits above, and each
        // victim as it is claimed -- are off the table.
        let mut evictable: Vec<i64> = self
            .usage
            .iter()
            .map(|u| if *u == step { UNEVICTABLE } else { *u })
            .collect();
        // Under a cap, a slot holding an expert this step routes to is
        // also off the table even if the route is being sent to the
        // CPU: evicting it would throw away residency the very next
        // step wants.
        if policy.is_some() {
            for (evict, id) in evictable.iter_mut().zip(self.id_of_slot.iter()) {
                if *id < 0 {
                    continue;
                }
                let owner = *id - base as i64;
                if (0..self.num_experts as i64).contains(&owner)
                    && distinct.contains(&(owner as u32))
                {
                    *evict = UNEVICTABLE;
                }
            }
        }

        let mut copy = CopyPlan::default();
        for expert in missing.iter().take(num_fetch) {
            let victim = argmin_slot(&evictable);
            let old = self.id_of_slot[victim];
            if old >= 0 {
                self.slot_for_id[old as usize] = -1;
            }
            let id = base + *expert as usize;
            self.id_of_slot[victim] = id as i64;
            self.slot_for_id[id] = victim as i64;
            self.usage[victim] = step;
            evictable[victim] = UNEVICTABLE;
            copy.dst_slots.push(victim as u32);
            // Layer-local, so it indexes this layer's own host bank.
            copy.src_rows.push(*expert);
        }

        let slots: Vec<Option<u32>> = expert_ids
            .iter()
            .map(|expert| {
                let slot = self.slot_for_id[base + *expert as usize];
                (slot >= 0).then_some(slot as u32)
            })
            .collect();

        if policy.is_some() {
            for expert in &distinct {
                self.recency[base + *expert as usize] = step;
            }
        }

        self.stats.calls += 1;
        self.stats.active += distinct.len() as u64;
        self.stats.missing += num_missing as u64;
        self.stats.fetched += num_fetch as u64;

        // The same four counters again, keeping the layer this time.
        let per_layer = &mut self.layer_stats[layer as usize];
        per_layer.calls += 1;
        per_layer.active += distinct.len() as u64;
        per_layer.missing += num_missing as u64;
        per_layer.fetched += num_fetch as u64;

        EnsurePlan {
            slots,
            copy,
            active: distinct.len(),
            missing: num_missing,
            fetched: num_fetch,
        }
    }

    /// Put a whole layer's experts in slots `0..num_experts`, in expert
    /// order, for a prefill.
    ///
    /// A prefill touches every expert, so streaming them one miss at a
    /// time is pointless -- the layer is copied whole, and **position
    /// equals expert id**, which lets routing ids index the buffer
    /// directly with no slot lookup at all.
    ///
    /// The bookkeeping still has to be exact: any *other* layer's
    /// expert that was living in one of those slots loses its
    /// residency here, or the next decode step would hit on a slot that
    /// now holds someone else's bytes.
    pub fn materialize_layer(&mut self, layer: u32) -> CopyPlan {
        let base = self.flat(layer, 0);
        let owns = |id: i64| id >= base as i64 && id < (base + self.num_experts) as i64;
        // Snapshot before mutating: the checks below all read the
        // pre-existing occupant.
        let previous: Vec<i64> = self.id_of_slot.clone();

        for (slot, old) in previous.iter().enumerate() {
            if owns(*old) {
                // This layer's own stale placements go; they are about
                // to be re-established at position == expert id.
                self.id_of_slot[slot] = -1;
                self.usage[slot] = 0;
            }
        }
        for old in previous.iter().take(self.num_experts) {
            if *old >= 0 && !owns(*old) {
                self.slot_for_id[*old as usize] = -1;
            }
        }

        self.step += 1;
        let mut copy = CopyPlan::default();
        for expert in 0..self.num_experts {
            self.id_of_slot[expert] = (base + expert) as i64;
            self.slot_for_id[base + expert] = expert as i64;
            self.usage[expert] = self.step;
            copy.dst_slots.push(expert as u32);
            copy.src_rows.push(expert as u32);
        }
        copy
    }

    // ---- routing skew -------------------------------------------------

    /// Start (or stop) accumulating the decode routing histogram.
    ///
    /// Off by default: the counters are cheap but they are only
    /// meaningful over a long, stationary window, and a histogram that
    /// silently spans a model swap or a rebuild is worse than none.
    pub fn set_collect_routing(&mut self, on: bool) {
        self.collect_routing = on;
    }

    pub fn collects_routing(&self) -> bool {
        self.collect_routing
    }

    /// One layer's raw histogram row, indexed by expert id.
    pub fn routing_histogram(&self, layer: u32) -> &[u64] {
        let base = self.flat(layer, 0);
        &self.routing_freq[base..base + self.num_experts]
    }

    /// Throw the observed routing distribution away.
    pub fn reset_routing(&mut self) {
        self.routing_freq.fill(0);
    }

    /// Per-layer routing concentration over the observed histogram, or
    /// [`None`] if no routing has been observed at all.
    ///
    /// This describes the *traffic*, not the cache: nothing here
    /// depends on which policy ran or on what it happened to keep. That
    /// is the point. The realized miss rate says how a policy did;
    /// [`LayerRoutingSkew::oracle_hit_at_slots`] says how well the best
    /// possible policy could have done on the same traffic with the
    /// same fair share of slots. A layer where the two are close is not
    /// a cache-sizing problem however bad it looks, because the routing
    /// is flat and the slots are not where the hit rate went.
    pub fn routing_skew(&self) -> Option<RoutingSkewReport> {
        let experts = self.num_experts;
        let slots_per_layer = self.cache_size as f64 / self.num_layers as f64;
        // At least one slot: a per-layer share that rounds to zero
        // still gets to hold one expert, and an oracle over zero slots
        // would report 0.0 for every layer regardless of its routing.
        let oracle_slots = (slots_per_layer.round() as usize).max(1);
        // A one-expert layer has no skew to normalize against; ln(1) is
        // zero and the ratio would be a NaN that poisons the mean.
        let ln_experts = (experts as f64).ln();

        let mut per_layer: Vec<LayerRoutingSkew> = Vec::new();
        for layer in 0..self.num_layers {
            let freq = &self.routing_freq[layer * experts..(layer + 1) * experts];
            let routed: u64 = freq.iter().sum();
            if routed == 0 {
                continue;
            }
            let total = routed as f64;
            let working_set = freq.iter().filter(|f| **f > 0).count();

            let mut descending: Vec<u64> = freq.to_vec();
            descending.sort_unstable_by(|a, b| b.cmp(a));

            let head: u64 = descending.iter().take(oracle_slots).sum();
            let oracle_hit_at_slots = head as f64 / total;

            // The cdf is non-decreasing, so "how many entries are below
            // the coverage line" is a prefix length; +1 is the entry
            // that crosses it.
            let mut cumulative = 0u64;
            let mut below = 0usize;
            for count in &descending {
                cumulative += *count;
                if cumulative as f64 / total >= COVERAGE_FRACTION {
                    break;
                }
                below += 1;
            }

            let mut entropy = 0.0f64;
            for count in freq {
                if *count == 0 {
                    continue;
                }
                let p = *count as f64 / total;
                entropy -= p * p.max(1e-12).ln();
            }
            let norm_entropy = if ln_experts > 0.0 {
                entropy / ln_experts
            } else {
                0.0
            };

            per_layer.push(LayerRoutingSkew {
                layer: layer as u32,
                routed,
                working_set,
                experts_for_90pct: below + 1,
                norm_entropy,
                oracle_hit_at_slots,
            });
        }

        if per_layer.is_empty() {
            return None;
        }
        let layers = per_layer.len() as f64;
        Some(RoutingSkewReport {
            slots_per_layer,
            oracle_slots,
            working_set_mean: per_layer.iter().map(|l| l.working_set as f64).sum::<f64>() / layers,
            working_set_max: per_layer.iter().map(|l| l.working_set).max().unwrap_or(0),
            experts_for_90pct: per_layer
                .iter()
                .map(|l| l.experts_for_90pct as f64)
                .sum::<f64>()
                / layers,
            oracle_hit_at_slots: per_layer.iter().map(|l| l.oracle_hit_at_slots).sum::<f64>()
                / layers,
            norm_entropy: per_layer.iter().map(|l| l.norm_entropy).sum::<f64>() / layers,
            per_layer,
        })
    }

    // ---- prefill double buffers ---------------------------------------

    /// The slot range the two prefill buffers borrow: `[0, 2 *
    /// num_experts)`.
    pub fn prefill_buffer_slots(&self) -> usize {
        2 * self.num_experts
    }

    /// Whether this pool is large enough to lend the buffers their
    /// slots at all.
    ///
    /// A cache of exactly `2 * num_experts` fits the buffers but leaves
    /// no hit region above them, so every prefill row is a miss by the
    /// classification rule below. That is correct, merely slow; below
    /// `2 * num_experts` the buffers do not fit and overlap must be off.
    pub fn prefill_overlap_fits(&self) -> bool {
        self.cache_size >= self.prefill_buffer_slots()
    }

    /// Which layer a buffer currently stages, if any.
    pub fn prefill_buffer_layer(&self, buffer_id: u32) -> Option<u32> {
        self.prefill_layer[buffer_id as usize]
    }

    /// Expert rows served from the cache since the last
    /// [`reset_stats`](Self::reset_stats).
    pub fn prefill_hit_rows(&self) -> u64 {
        self.prefill_hit_rows
    }

    /// All expert rows staged into the buffers over the same window.
    /// The ratio against [`prefill_hit_rows`](Self::prefill_hit_rows)
    /// is how much of a prefill never touched the link.
    pub fn prefill_rows(&self) -> u64 {
        self.prefill_total_rows
    }

    /// Open a prefill chunk: both buffers empty, and take the residency
    /// snapshot the chunk classifies against.
    ///
    /// The snapshot is what makes the classification stable for the
    /// whole chunk. Hits are decided once, against the map as it stood
    /// before any of this chunk's staging ran, and the only writer
    /// inside the chunk -- buffer invalidation -- only ever rewrites
    /// slots that are already below the buffer threshold and therefore
    /// misses under both the live map and the snapshot. So the two can
    /// never disagree about a row, and a caller may issue the gathers
    /// and the host transfers in any order.
    pub fn begin_prefill(&mut self) {
        assert!(
            self.prefill_overlap_fits(),
            "cache of {} slots cannot lend {} to the prefill buffers",
            self.cache_size,
            self.prefill_buffer_slots()
        );
        self.prefill_layer = [None, None];
        self.prefill_released = [true, true];
        self.prefill_snapshot = Some(self.slot_for_id.clone());
    }

    /// Stage one layer into its buffer and say what that costs.
    ///
    /// `bank_feat_bytes` is each host bank's per-expert row size, in
    /// the caller's bank order; [`BankEntry::bank`] indexes it.
    ///
    /// Three rules decide the answer, and each of them is a correctness
    /// rule rather than a tuning choice:
    ///
    /// - **A row is resident only if its slot is at or above
    ///   `2 * num_experts`.** Anything below that -- including `-1` --
    ///   is a miss *by definition*, because the buffers own those slots
    ///   and rewrite them every other layer within the chunk. Widening
    ///   the test to "has a slot at all" makes a prefill gather from
    ///   slots the other buffer is concurrently overwriting, and it
    ///   loads some other expert's bytes without any error.
    /// - **Invalidation clears the residency map *and* zeroes usage**
    ///   for the buffer's slots. See
    ///   [`prefill_buffer_slots`](Self::prefill_buffer_slots).
    /// - **Misses coalesce into contiguous expert runs**, and a bank
    ///   under [`SMALL_BANK_FEAT_BYTES`] ships its whole layer as one
    ///   entry even when nothing missed.
    ///
    /// Nothing staged here is registered as resident: the buffer's
    /// bytes are volatile within the chunk, so recording them would
    /// hand the next decode step a hit on a slot that is about to be
    /// overwritten.
    pub fn prefetch_prefill_layer(&mut self, layer: u32, bank_feat_bytes: &[u64]) -> PrefillPlan {
        assert!(
            (layer as usize) < self.num_layers,
            "layer {layer} is outside a model of {} layers",
            self.num_layers
        );
        let experts = self.num_experts;
        let buffer_id = (layer as usize) % 2;
        let buffer_base = buffer_id * experts;
        let gather_banks: Vec<usize> = bank_feat_bytes
            .iter()
            .enumerate()
            .filter(|(_, feat)| **feat >= SMALL_BANK_FEAT_BYTES)
            .map(|(bank, _)| bank)
            .collect();

        // Re-staging the layer a buffer already holds is a no-op, which
        // is what lets a caller prefetch ahead and then ask for the
        // same layer when it reaches it.
        if self.prefill_layer[buffer_id] == Some(layer) {
            return PrefillPlan {
                layer,
                buffer_id: buffer_id as u32,
                already_loaded: true,
                gather: GatherPlan::default(),
                miss_runs: Vec::new(),
                entries: Vec::new(),
                gather_banks,
            };
        }
        if let Some(held) = self.prefill_layer[buffer_id] {
            assert!(
                self.prefill_released[buffer_id],
                "prefill buffer {buffer_id} still holds layer {held}; staging layer \
                 {layer} into it would overwrite bytes a running GEMM is reading"
            );
        }

        let snapshot = self
            .prefill_snapshot
            .as_ref()
            .expect("begin_prefill must open the chunk before a layer is staged");
        let base = layer as usize * experts;
        // The threshold, not `0`: the buffers own everything below it.
        let threshold = self.prefill_buffer_slots() as i64;

        let mut gather = GatherPlan::default();
        let mut missing: Vec<u32> = Vec::new();
        for expert in 0..experts {
            let slot = snapshot[base + expert];
            if slot >= threshold {
                gather.dst_slots.push((buffer_base + expert) as u32);
                gather.src_slots.push(slot as u32);
            } else {
                missing.push(expert as u32);
            }
        }

        self.prefill_hit_rows += gather.len() as u64;
        self.prefill_total_rows += experts as u64;
        self.invalidate_prefill_buffer(buffer_id);

        let miss_runs = coalesce_runs(&missing);
        let mut entries: Vec<BankEntry> = Vec::new();
        for (bank, feat) in bank_feat_bytes.iter().enumerate() {
            if *feat < SMALL_BANK_FEAT_BYTES {
                // Whole layer as one entry, even with zero misses: it
                // keeps every entry in the batch above the driver's
                // async floor, and it covers the rows the gather skips
                // for this bank.
                entries.push(BankEntry {
                    bank,
                    dst_slot: buffer_base as u32,
                    src_row: 0,
                    rows: experts as u32,
                    bytes: experts as u64 * feat,
                    whole_layer: true,
                });
                continue;
            }
            for run in &miss_runs {
                entries.push(BankEntry {
                    bank,
                    dst_slot: buffer_base as u32 + run.start,
                    src_row: run.start,
                    rows: run.len,
                    bytes: run.len as u64 * feat,
                    whole_layer: false,
                });
            }
        }

        self.prefill_layer[buffer_id] = Some(layer);
        self.prefill_released[buffer_id] = false;

        PrefillPlan {
            layer,
            buffer_id: buffer_id as u32,
            already_loaded: false,
            gather,
            miss_runs,
            entries,
            gather_banks,
        }
    }

    /// Mark a layer's buffer done, freeing it for the next layer of the
    /// same parity. A layer that is not the one staged is ignored.
    pub fn release_prefill_layer(&mut self, layer: u32) {
        let buffer_id = (layer as usize) % 2;
        if self.prefill_layer[buffer_id] == Some(layer) {
            self.prefill_released[buffer_id] = true;
        }
    }

    /// Drop whatever a buffer held.
    ///
    /// Two things must happen and skipping either one is a live bug.
    /// Clearing `slot_for_id` for the old occupants is what stops a
    /// later decode step from hitting on bytes the buffer is about to
    /// overwrite. Zeroing `usage` is what makes these slots the *oldest*
    /// in the pool, so the `argmin(usage)` victim search in
    /// [`ensure`](Self::ensure) takes them before anything else: leave
    /// the old usage in place and the very next miss evicts a real
    /// resident from somewhere else in the cache while these free slots
    /// sit unused.
    fn invalidate_prefill_buffer(&mut self, buffer_id: usize) {
        let start = buffer_id * self.num_experts;
        for slot in start..start + self.num_experts {
            let old = self.id_of_slot[slot];
            if old >= 0 {
                self.slot_for_id[old as usize] = -1;
            }
            self.id_of_slot[slot] = -1;
            self.usage[slot] = 0;
        }
    }

    /// Forget the buffers without touching residency, for the paths
    /// that just wiped residency anyway.
    fn forget_prefill_buffers(&mut self) {
        self.prefill_layer = [None, None];
        self.prefill_released = [true, true];
        self.prefill_snapshot = None;
    }
}

/// A slot resize the cache refused, having changed nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RebuildRejected {
    pub requested: usize,
    pub minimum: usize,
}

impl std::fmt::Display for RebuildRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "expert cache of {} slots cannot hold one layer of {} experts",
            self.requested, self.minimum
        )
    }
}

impl std::error::Error for RebuildRejected {}

/// Ascending expert ids collapsed into contiguous runs.
///
/// One entry per run rather than one per expert: a batched transfer
/// pays a fixed cost per entry, and a layer whose misses happen to be
/// adjacent -- the common case, since routing tends to cluster -- would
/// otherwise ship `num_experts` single-row entries instead of a handful
/// of large ones.
fn coalesce_runs(experts: &[u32]) -> Vec<MissRun> {
    let mut runs: Vec<MissRun> = Vec::new();
    for expert in experts {
        match runs.last_mut() {
            Some(run) if run.start + run.len == *expert => run.len += 1,
            _ => runs.push(MissRun {
                start: *expert,
                len: 1,
            }),
        }
    }
    runs
}

/// The lowest-`usage` slot, ties to the lowest slot index.
///
/// The tie-break is not cosmetic: a GPU kernel and a CPU reference have
/// to pick the same victim, or the two halves of a hybrid step disagree
/// about which expert is where.
fn argmin_slot(usage: &[i64]) -> usize {
    let mut best = 0usize;
    let mut best_usage = usage[0];
    for (slot, value) in usage.iter().enumerate().skip(1) {
        if *value < best_usage {
            best = slot;
            best_usage = *value;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> ExpertCache {
        ExpertCache::new(4, 8, 16)
    }

    #[test]
    fn a_cold_miss_is_fetched_and_becomes_resident() {
        let mut cache = cache();
        let plan = cache.ensure(0, &[3, 5]);
        assert_eq!(plan.missing, 2);
        assert_eq!(plan.fetched, 2);
        assert_eq!(plan.copy.src_rows, vec![3, 5], "layer-local rows");
        assert_eq!(plan.slots.len(), 2);
        assert!(plan.slots.iter().all(Option::is_some));

        // The second visit is free.
        let again = cache.ensure(0, &[3, 5]);
        assert_eq!(again.missing, 0);
        assert!(again.copy.is_empty());
        assert_eq!(again.slots, plan.slots);
    }

    /// Layers share one pool, so the same expert *index* in two layers
    /// is two different residents.
    #[test]
    fn layers_share_one_pool_through_a_flat_id_space() {
        let mut cache = cache();
        cache.ensure(0, &[3]);
        let plan = cache.ensure(1, &[3]);
        assert_eq!(plan.missing, 1, "layer 1's expert 3 is a different expert");
        assert_ne!(cache.slot_of(0, 3), cache.slot_of(1, 3));
        assert_eq!(cache.resident_slots(), 2);
    }

    #[test]
    fn duplicate_routes_collapse_into_one_copy_and_one_slot() {
        let mut cache = cache();
        let plan = cache.ensure(0, &[7, 7, 7, 2]);
        assert_eq!(plan.active, 2);
        assert_eq!(plan.copy.len(), 2, "one copy per distinct expert");
        assert_eq!(plan.slots[0], plan.slots[1]);
        assert_eq!(plan.slots[1], plan.slots[2]);
        assert_ne!(plan.slots[0], plan.slots[3]);
    }

    /// The rule that keeps a step from sabotaging itself: an expert
    /// this step just hit cannot be the victim of this step's misses.
    #[test]
    fn a_step_never_evicts_an_expert_it_is_about_to_read() {
        // Two layers competing for exactly one layer's worth of slots.
        let mut cache = ExpertCache::new(2, 4, 4);
        cache.ensure(0, &[0, 1, 2, 3]);
        cache.ensure(1, &[0]);
        let held = cache.slot_of(1, 0).unwrap();

        // A route to a resident expert plus a miss: the miss must take
        // some *other* slot.
        let plan = cache.ensure(1, &[0, 1]);
        assert_eq!(plan.missing, 1);
        assert_eq!(cache.slot_of(1, 0), Some(held), "the hit kept its slot");
        assert_ne!(plan.copy.dst_slots[0], held);
        assert_eq!(plan.slots[0], Some(held));
    }

    #[test]
    fn eviction_takes_the_least_recently_used_slot() {
        let mut cache = ExpertCache::new(2, 4, 4);
        for expert in 0..4u32 {
            cache.ensure(0, &[expert]); // one step each, ascending usage
        }
        cache.ensure(0, &[0]); // refresh 0, making 1 the oldest
        let oldest = cache.slot_of(0, 1).unwrap();

        let plan = cache.ensure(1, &[0]);
        assert_eq!(plan.copy.dst_slots, vec![oldest]);
        assert_eq!(cache.slot_of(0, 1), None, "the oldest went");
        assert!(cache.slot_of(0, 0).is_some(), "the refreshed one stayed");
    }

    /// A large cold batch must give every distinct expert its own slot
    /// -- no two routes may share one.
    #[test]
    fn a_large_cold_batch_assigns_unique_slots() {
        let mut cache = ExpertCache::new(4, 64, 256);
        let ids: Vec<u32> = (0..64).collect();
        let plan = cache.ensure(2, &ids);
        assert_eq!(plan.missing, 64);
        assert_eq!(plan.copy.len(), 64);

        let mut slots: Vec<u32> = plan.slots.iter().map(|s| s.unwrap()).collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), 64, "no slot serves two experts");
        assert_eq!(
            plan.copy.src_rows, ids,
            "misses are ranked by ascending expert id"
        );
    }

    /// Every routed expert is assigned exactly once, to exactly one
    /// device -- the invariant that makes adding the GPU and CPU
    /// partial results correct.
    #[test]
    fn the_hybrid_split_assigns_every_route_to_exactly_one_device() {
        let mut cache = ExpertCache::new(1, 32, 40);
        let policy = QStarPolicy::from_fraction(0.415);
        let routes: Vec<u32> = (0..8).collect();

        let plan = cache.ensure_hybrid(0, &routes, &policy);
        assert_eq!(plan.missing, 8);
        assert_eq!(plan.fetched, 3, "0.415 * 8 = 3.32 -> 3");
        assert_eq!(plan.copy.len(), 3);
        let on_gpu = plan.slots.iter().filter(|s| s.is_some()).count();
        assert_eq!(on_gpu, 3);
        assert_eq!(plan.slots.iter().filter(|s| s.is_none()).count(), 5);
    }

    /// A hot expert that keeps losing the cap must eventually win one,
    /// or it is computed on the CPU forever while the cache holds cold
    /// experts.
    #[test]
    fn a_recurring_miss_climbs_the_fetch_order() {
        let mut cache = ExpertCache::new(1, 16, 16);
        let policy = QStarPolicy::fixed_cap(1);
        // Expert 9 is routed every step alongside a rotating cast.
        for round in 0..6u32 {
            cache.ensure_hybrid(0, &[9, round], &policy);
        }
        assert!(
            cache.slot_of(0, 9).is_some(),
            "the expert routed every step became resident"
        );
    }

    #[test]
    fn the_fixed_cap_is_the_unbenchmarked_default() {
        let mut cache = ExpertCache::new(1, 32, 40);
        let plan = cache.ensure_hybrid(0, &[0, 1, 2, 3, 4, 5, 6, 7], &QStarPolicy::fixed_cap(1));
        assert_eq!(plan.missing, 8);
        assert_eq!(plan.fetched, 1);
        assert_eq!(plan.copy.len(), 1);
    }

    /// A prefill copies a whole layer and lays it out so that position
    /// equals expert id -- and must not leave another layer thinking it
    /// still owns one of those slots.
    #[test]
    fn materializing_a_layer_reclaims_other_layers_slots_cleanly() {
        let mut cache = ExpertCache::new(2, 4, 4);
        cache.ensure(0, &[0, 1, 2, 3]);
        assert_eq!(cache.resident_slots(), 4);

        let copy = cache.materialize_layer(1);
        assert_eq!(copy.dst_slots, vec![0, 1, 2, 3]);
        assert_eq!(copy.src_rows, vec![0, 1, 2, 3], "position == expert id");
        for expert in 0..4u32 {
            assert_eq!(cache.slot_of(1, expert), Some(expert));
            assert_eq!(
                cache.slot_of(0, expert),
                None,
                "layer 0 must not hit on layer 1's bytes"
            );
        }

        // And a later decode on layer 0 misses and reloads, rather than
        // reading whatever is in the slot.
        let plan = cache.ensure(0, &[3]);
        assert_eq!(plan.missing, 1);
    }

    #[test]
    fn a_second_materialize_of_the_same_layer_is_idempotent() {
        let mut cache = ExpertCache::new(2, 4, 6);
        cache.materialize_layer(1);
        cache.materialize_layer(1);
        for expert in 0..4u32 {
            assert_eq!(cache.slot_of(1, expert), Some(expert));
        }
        assert_eq!(cache.resident_slots(), 4);
    }

    #[test]
    fn stats_answer_whether_the_cache_is_big_enough() {
        let mut cache = ExpertCache::new(1, 2, 2);
        cache.ensure(0, &[0, 1]); // 2 misses
        cache.ensure(0, &[0, 1]); // 2 hits
        let stats = cache.stats();
        assert_eq!(stats.calls, 2);
        assert_eq!(stats.active, 4);
        assert_eq!(stats.missing, 2);
        assert_eq!(stats.miss_rate(), 0.5);
        assert_eq!(stats.fetch_rate(), 1.0);
        assert_eq!(stats.active_per_call(), 2.0);
    }

    #[test]
    fn a_rebuild_that_cannot_hold_a_layer_changes_nothing() {
        let mut cache = ExpertCache::new(2, 8, 16);
        cache.ensure(0, &[1]);
        let err = cache.rebuild(4).unwrap_err();
        assert_eq!(err.minimum, 8);
        assert_eq!(cache.cache_size(), 16, "still serving from the old cache");
        assert!(cache.slot_of(0, 1).is_some());

        cache.rebuild(8).expect("one layer fits");
        assert_eq!(cache.cache_size(), 8);
        assert_eq!(
            cache.resident_slots(),
            0,
            "slot ids no longer mean anything"
        );
    }

    #[test]
    #[should_panic(expected = "cannot hold one layer")]
    fn a_cache_too_small_for_one_layer_is_rejected_at_construction() {
        ExpertCache::new(2, 8, 4);
    }

    /// The residency map must stay coherent under sustained pressure:
    /// no slot claimed by two experts, no expert claiming a slot that
    /// does not point back.
    #[test]
    fn the_residency_map_stays_bijective_under_pressure() {
        let mut cache = ExpertCache::new(4, 16, 20);
        let policy = QStarPolicy::from_fraction(0.5);
        for step in 0..200u32 {
            let layer = step % 4;
            let routes: Vec<u32> = (0..6).map(|i| (step * 7 + i * 3) % 16).collect();
            let plan = if step % 2 == 0 {
                cache.ensure(layer, &routes)
            } else {
                cache.ensure_hybrid(layer, &routes, &policy)
            };
            assert_eq!(plan.slots.len(), routes.len());

            for slot in 0..cache.cache_size() as u32 {
                if let Some(id) = cache.resident_in(slot) {
                    assert_eq!(
                        cache.slot_of(id.layer, id.expert),
                        Some(slot),
                        "slot {slot} and its occupant disagree"
                    );
                }
            }
            assert!(cache.resident_slots() <= cache.cache_size());
        }
    }

    // ---- per-layer stats ----------------------------------------------

    /// The rule: the counters keep the layer that produced them.
    ///
    /// This test FAILS if per-layer stats are derived the naive way --
    /// by taking the global window and apportioning it across layers.
    /// Both caches below end the window with an identical global
    /// picture (8 active, 4 missing, a 0.5 miss rate), so any
    /// apportioning reports the same 0.5 for every layer of both. The
    /// two situations are in fact opposites: the first has one layer
    /// thrashing at 1.0 beside one at 0.0 and wants slots moved between
    /// layers, the second is uniformly half-missing and wants a bigger
    /// pool.
    #[test]
    fn per_layer_stats_tell_one_thrashing_layer_from_a_uniformly_tight_cache() {
        let mut thrashing = ExpertCache::new(2, 4, 8);
        thrashing.ensure(1, &[0, 1]); // warm layer 1 outside the window
        thrashing.reset_stats();
        thrashing.ensure(0, &[0, 1]); // 2 active, 2 missing
        thrashing.ensure(0, &[2, 3]); // 2 active, 2 missing
        thrashing.ensure(1, &[0, 1]); // 2 active, 0 missing
        thrashing.ensure(1, &[0, 1]); // 2 active, 0 missing

        let mut uniform = ExpertCache::new(2, 4, 8);
        uniform.ensure(0, &[0]);
        uniform.ensure(1, &[0]);
        uniform.reset_stats();
        uniform.ensure(0, &[0, 1]); // hit + miss
        uniform.ensure(0, &[0, 2]); // hit + miss
        uniform.ensure(1, &[0, 1]);
        uniform.ensure(1, &[0, 2]);

        // Indistinguishable globally.
        assert_eq!(thrashing.stats().active, uniform.stats().active);
        assert_eq!(thrashing.stats().missing, uniform.stats().missing);
        assert_eq!(thrashing.stats().miss_rate(), 0.5);
        assert_eq!(uniform.stats().miss_rate(), 0.5);

        // Opposite per layer.
        assert_eq!(thrashing.layer_stats(0).miss_rate(), 1.0);
        assert_eq!(thrashing.layer_stats(1).miss_rate(), 0.0);
        assert_eq!(uniform.layer_stats(0).miss_rate(), 0.5);
        assert_eq!(uniform.layer_stats(1).miss_rate(), 0.5);
    }

    #[test]
    fn per_layer_stats_sum_to_the_global_stats() {
        let mut cache = ExpertCache::new(3, 8, 16);
        let policy = QStarPolicy::from_fraction(0.5);
        for step in 0..30u32 {
            let layer = step % 3;
            let routes: Vec<u32> = (0..4).map(|i| (step * 5 + i * 3) % 8).collect();
            if step % 2 == 0 {
                cache.ensure(layer, &routes);
            } else {
                cache.ensure_hybrid(layer, &routes, &policy);
            }
        }
        let global = cache.stats();
        let summed =
            cache
                .per_layer_stats()
                .iter()
                .fold(ExpertCacheStats::default(), |mut acc, layer| {
                    acc.calls += layer.calls;
                    acc.active += layer.active;
                    acc.missing += layer.missing;
                    acc.fetched += layer.fetched;
                    acc
                });
        assert_eq!(summed, global);
        assert_eq!(cache.per_layer_stats().len(), 3);
        assert_eq!(cache.layer_stats(1).calls, 10);
    }

    #[test]
    fn resetting_stats_opens_a_new_per_layer_window() {
        let mut cache = ExpertCache::new(2, 4, 8);
        cache.ensure(0, &[0, 1]);
        assert_eq!(cache.layer_stats(0).missing, 2);
        cache.reset_stats();
        assert_eq!(cache.layer_stats(0), ExpertCacheStats::default());
        assert_eq!(cache.stats(), ExpertCacheStats::default());

        // The window is about counters, not residency: the experts are
        // still there, so the new window sees hits.
        cache.ensure(0, &[0, 1]);
        assert_eq!(cache.layer_stats(0).missing, 0);
        assert_eq!(cache.layer_stats(0).active, 2);
        assert_eq!(cache.layer_stats(0).active_per_call(), 2.0);
        assert_eq!(cache.layer_stats(0).missing_per_call(), 0.0);
    }

    // ---- routing skew --------------------------------------------------

    /// The rule: the histogram records what the router asked for, so
    /// duplicate routes count as separate mass.
    ///
    /// This test FAILS if the histogram is fed the deduplicated route
    /// list `ensure` already computes for its miss accounting. Skew is
    /// precisely the difference between "which experts were touched"
    /// and "how much traffic each one took"; deduplicating erases the
    /// second and reports every busy layer as uniform.
    #[test]
    fn the_routing_histogram_counts_every_route_not_every_distinct_expert() {
        let mut cache = ExpertCache::new(2, 4, 8);
        cache.set_collect_routing(true);
        cache.ensure(0, &[3, 3, 3, 1]);
        assert_eq!(cache.routing_histogram(0), &[0, 1, 0, 3]);
        assert_eq!(cache.routing_histogram(1), &[0, 0, 0, 0]);
    }

    /// The rule: `oracle_hit_at_slots` is the top-`C` share of the
    /// observed mass, `C = max(1, round(cache_size / num_layers))`.
    ///
    /// This test FAILS if the oracle is derived the naive way, from the
    /// working set alone -- `C / working_set`, "how much of what this
    /// layer touches fits". Both layers below touch all 8 experts, so
    /// that formula reports 0.5 for each. The real answer is 0.96 for
    /// the skewed layer and 0.52 for the flat one, and that gap is the
    /// whole diagnosis: four slots serve the first layer almost
    /// perfectly and cannot help the second at any size.
    #[test]
    fn oracle_hit_at_slots_separates_a_skewed_layer_from_a_flat_one() {
        let mut cache = ExpertCache::new(2, 8, 8);
        cache.set_collect_routing(true);

        let routes = |counts: [usize; 8]| -> Vec<u32> {
            let mut out = Vec::new();
            for (expert, count) in counts.iter().enumerate() {
                for _ in 0..*count {
                    out.push(expert as u32);
                }
            }
            out
        };
        // 96 of 100 routes land on four experts.
        cache.ensure(0, &routes([24, 24, 24, 24, 1, 1, 1, 1]));
        // Near-flat: the best four experts hold 52 of 100.
        cache.ensure(1, &routes([13, 13, 13, 13, 12, 12, 12, 12]));

        let report = cache.routing_skew().expect("routing was observed");
        assert_eq!(report.slots_per_layer, 4.0);
        assert_eq!(report.oracle_slots, 4);
        assert_eq!(report.per_layer.len(), 2);

        let skewed = report.per_layer[0];
        let flat = report.per_layer[1];
        assert_eq!(skewed.routed, 100);
        assert_eq!(flat.routed, 100);
        assert_eq!(
            skewed.working_set, flat.working_set,
            "both layers touch every expert, so the working set cannot tell them apart"
        );
        assert!((skewed.oracle_hit_at_slots - 0.96).abs() < 1e-12);
        assert!((flat.oracle_hit_at_slots - 0.52).abs() < 1e-12);
        assert!(skewed.norm_entropy < flat.norm_entropy);
        assert!(skewed.experts_for_90pct < flat.experts_for_90pct);
    }

    #[test]
    fn routing_skew_reports_the_working_set_coverage_and_entropy_per_layer() {
        let mut cache = ExpertCache::new(2, 8, 8);
        cache.set_collect_routing(true);
        cache.ensure(0, &[0, 1, 2, 3, 4, 5, 6, 7]); // perfectly flat
        cache.ensure(1, &[2, 2, 2, 2]); // one expert, always

        let report = cache.routing_skew().expect("routing was observed");
        let flat = report.per_layer[0];
        assert_eq!(flat.working_set, 8);
        assert_eq!(flat.experts_for_90pct, 8, "7 experts reach 0.875, not 0.9");
        assert!((flat.norm_entropy - 1.0).abs() < 1e-12);
        assert!((flat.oracle_hit_at_slots - 0.5).abs() < 1e-12);

        let single = report.per_layer[1];
        assert_eq!(single.working_set, 1);
        assert_eq!(single.experts_for_90pct, 1);
        assert_eq!(single.norm_entropy, 0.0);
        assert_eq!(single.oracle_hit_at_slots, 1.0);

        assert_eq!(report.working_set_max, 8);
        assert_eq!(report.working_set_mean, 4.5);
        assert_eq!(report.experts_for_90pct, 4.5);
        assert!((report.oracle_hit_at_slots - 0.75).abs() < 1e-12);
        assert!((report.norm_entropy - 0.5).abs() < 1e-12);
    }

    /// A layer nothing routed to has no row, and an unobserved model
    /// has no report -- zeros here would read as "flat routing, no
    /// cache will help", which is the opposite of "no data".
    #[test]
    fn routing_skew_is_absent_until_routing_is_observed() {
        let mut cache = ExpertCache::new(4, 8, 16);
        cache.ensure(0, &[1, 2]);
        assert!(
            cache.routing_skew().is_none(),
            "collection is opt-in, so nothing was recorded"
        );

        cache.set_collect_routing(true);
        cache.ensure(2, &[1, 2]);
        let report = cache.routing_skew().expect("layer 2 was observed");
        assert_eq!(report.per_layer.len(), 1);
        assert_eq!(report.per_layer[0].layer, 2);

        cache.reset_routing();
        assert!(cache.routing_skew().is_none());
    }

    // ---- prefill double buffers ----------------------------------------

    const BIG_BANK: u64 = 512 * 1024;
    const SMALL_BANK: u64 = 4 * 1024;

    /// Four layers of four experts filling all sixteen slots in layer
    /// order, so layer `L` owns slots `[4L, 4L + 4)`. The buffers
    /// borrow slots `[0, 8)`, which puts layers 0 and 1 inside the
    /// buffer range and layers 2 and 3 above it.
    fn cache_with_one_layer_per_quarter() -> ExpertCache {
        let mut cache = ExpertCache::new(4, 4, 16);
        for layer in 0..4u32 {
            cache.ensure(layer, &[0, 1, 2, 3]);
        }
        for layer in 0..4u32 {
            for expert in 0..4u32 {
                assert_eq!(cache.slot_of(layer, expert), Some(layer * 4 + expert));
            }
        }
        cache
    }

    /// The rule: a row is resident only if its slot is at or above
    /// `2 * num_experts`.
    ///
    /// This test FAILS if hits are classified the naive way, `slot >=
    /// 0` -- "the residency map has a slot for it, so it is on the
    /// device". Layer 1's experts live in slots 4..8, which is exactly
    /// the range buffer 1 is about to overwrite; the naive test calls
    /// all four hits and emits a gather that reads those slots while
    /// the staging copy writes them, so the prefill silently computes
    /// on whichever bytes won.
    #[test]
    fn a_slot_inside_the_prefill_buffers_is_a_miss_however_resident_it_looks() {
        let mut cache = cache_with_one_layer_per_quarter();
        cache.begin_prefill();

        // Layer 1 -> buffer 1, whose slots are 4..8 -- where layer 1's
        // own experts are resident.
        assert_eq!(cache.slot_of(1, 0), Some(4));
        let plan = cache.prefetch_prefill_layer(1, &[BIG_BANK]);
        assert_eq!(plan.buffer_id, 1);
        assert!(
            plan.gather.is_empty(),
            "slots below 2 * num_experts are volatile, not resident"
        );
        assert_eq!(plan.miss_runs, vec![MissRun { start: 0, len: 4 }]);
        assert_eq!(cache.prefill_hit_rows(), 0);
        assert_eq!(cache.prefill_rows(), 4);
    }

    #[test]
    fn a_resident_expert_above_the_buffer_slots_is_gathered_device_side() {
        let mut cache = cache_with_one_layer_per_quarter();
        cache.begin_prefill();

        // Layer 2 lives in slots 8..12, clear of the buffers.
        let plan = cache.prefetch_prefill_layer(2, &[BIG_BANK]);
        assert_eq!(plan.buffer_id, 0);
        assert_eq!(plan.gather.dst_slots, vec![0, 1, 2, 3]);
        assert_eq!(plan.gather.src_slots, vec![8, 9, 10, 11]);
        assert!(plan.miss_runs.is_empty());
        assert!(
            plan.entries.is_empty(),
            "a big bank with no misses ships nothing"
        );
        assert_eq!(cache.prefill_hit_rows(), 4);
        assert_eq!(cache.prefill_rows(), 4);
    }

    /// The rule: invalidation zeroes `usage` on the buffer's slots as
    /// well as clearing the residency map.
    ///
    /// This test FAILS if invalidation is done the naive way -- drop
    /// the occupants and leave the LRU clock alone. Slots 0..4 are
    /// freshly touched here, so their stale usage outranks layer 2's;
    /// the next decode miss then evicts a live resident from slot 8
    /// while four empty slots sit unused, and layer 2 pays a fetch it
    /// did not have to.
    #[test]
    fn invalidating_a_prefill_buffer_makes_its_slots_the_first_victims() {
        let mut cache = cache_with_one_layer_per_quarter();
        // Re-touch the low half so the buffer slots are the *newest*
        // in the pool, not the oldest.
        cache.ensure(0, &[0, 1, 2, 3]);
        cache.ensure(1, &[0, 1, 2, 3]);

        cache.begin_prefill();
        cache.prefetch_prefill_layer(0, &[BIG_BANK]); // buffer 0 == slots 0..4
        for expert in 0..4u32 {
            assert_eq!(
                cache.slot_of(0, expert),
                None,
                "the buffer's old occupants lost their residency"
            );
        }

        // A cold decode miss must land in the invalidated buffer.
        let plan = cache.ensure(0, &[0]);
        assert_eq!(plan.missing, 1);
        assert_eq!(plan.copy.dst_slots, vec![0]);
        assert_eq!(
            cache.slot_of(2, 0),
            Some(8),
            "no real resident was evicted while free slots existed"
        );
    }

    /// The rule: misses coalesce into contiguous expert runs.
    ///
    /// This test FAILS if each missing expert ships as its own entry.
    /// Experts 0,1,2 and 5,6 miss here; the run form is two entries of
    /// three and two rows, the naive form five single-row entries --
    /// five times the descriptors for the same bytes, at a fraction of
    /// the achievable rate.
    #[test]
    fn prefill_misses_ship_as_contiguous_expert_runs() {
        let mut cache = ExpertCache::new(4, 8, 32);
        cache.ensure(0, &[0, 1, 2, 3, 4, 5, 6, 7]); // slots 0..8
        cache.ensure(2, &[0, 1, 2, 3, 4, 5, 6, 7]); // slots 8..16
        cache.ensure(1, &[3, 4, 7]); // slots 16, 17, 18 -- above the buffers
        cache.begin_prefill();

        let plan = cache.prefetch_prefill_layer(1, &[BIG_BANK]);
        assert_eq!(plan.buffer_id, 1);
        assert_eq!(plan.gather.dst_slots, vec![11, 12, 15]);
        assert_eq!(plan.gather.src_slots, vec![16, 17, 18]);
        assert_eq!(
            plan.miss_runs,
            vec![MissRun { start: 0, len: 3 }, MissRun { start: 5, len: 2 }]
        );
        assert_eq!(
            plan.entries,
            vec![
                BankEntry {
                    bank: 0,
                    dst_slot: 8,
                    src_row: 0,
                    rows: 3,
                    bytes: 3 * BIG_BANK,
                    whole_layer: false,
                },
                BankEntry {
                    bank: 0,
                    dst_slot: 13,
                    src_row: 5,
                    rows: 2,
                    bytes: 2 * BIG_BANK,
                    whole_layer: false,
                },
            ]
        );
    }

    /// The rule: a bank whose per-expert row is under
    /// `SMALL_BANK_FEAT_BYTES` ships its whole layer as one entry, even
    /// with zero misses.
    ///
    /// This test FAILS if entries are emitted the naive way, only for
    /// experts that actually missed. Every row of layer 1 is resident
    /// here, so the naive plan ships nothing at all -- but the gather
    /// deliberately skips small banks, so their rows would never arrive
    /// and the layer would compute on stale buffer bytes. The
    /// whole-layer entry is also what keeps a tiny bank from dropping a
    /// sub-256 KiB entry into a batch and turning the whole transfer
    /// synchronous.
    #[test]
    fn a_small_bank_ships_its_whole_layer_even_with_no_misses() {
        let mut cache = ExpertCache::new(4, 8, 32);
        cache.ensure(0, &[0, 1, 2, 3, 4, 5, 6, 7]); // slots 0..8
        cache.ensure(2, &[0, 1, 2, 3, 4, 5, 6, 7]); // slots 8..16
        cache.ensure(1, &[0, 1, 2, 3, 4, 5, 6, 7]); // slots 16..24
        cache.begin_prefill();

        let plan = cache.prefetch_prefill_layer(1, &[BIG_BANK, SMALL_BANK]);
        assert_eq!(plan.gather.len(), 8, "every row is resident");
        assert!(plan.miss_runs.is_empty());
        assert_eq!(plan.gather_banks, vec![0], "the small bank is not gathered");
        assert_eq!(
            plan.entries,
            vec![BankEntry {
                bank: 1,
                dst_slot: 8,
                src_row: 0,
                rows: 8,
                bytes: 8 * SMALL_BANK,
                whole_layer: true,
            }]
        );
    }

    #[test]
    fn the_two_prefill_buffers_rotate_by_layer_parity() {
        let mut cache = ExpertCache::new(4, 4, 16);
        cache.begin_prefill();
        assert_eq!(cache.prefill_buffer_layer(0), None);

        assert_eq!(cache.prefetch_prefill_layer(0, &[BIG_BANK]).buffer_id, 0);
        assert_eq!(cache.prefetch_prefill_layer(1, &[BIG_BANK]).buffer_id, 1);
        assert_eq!(cache.prefill_buffer_layer(0), Some(0));
        assert_eq!(cache.prefill_buffer_layer(1), Some(1));

        // Re-staging what a buffer already holds moves nothing.
        let again = cache.prefetch_prefill_layer(1, &[BIG_BANK]);
        assert!(again.already_loaded);
        assert!(again.entries.is_empty());
        assert_eq!(cache.prefill_rows(), 8, "the no-op staged no rows");

        cache.release_prefill_layer(0);
        assert_eq!(cache.prefetch_prefill_layer(2, &[BIG_BANK]).buffer_id, 0);
        assert_eq!(cache.prefill_buffer_layer(0), Some(2));
    }

    /// Reusing a buffer the consumer has not finished with would
    /// overwrite bytes a running multiply is reading, which is a silent
    /// wrong answer rather than a failure -- so it is refused.
    #[test]
    #[should_panic(expected = "still holds layer 0")]
    fn reusing_a_prefill_buffer_before_it_is_released_is_refused() {
        let mut cache = ExpertCache::new(4, 4, 16);
        cache.begin_prefill();
        cache.prefetch_prefill_layer(0, &[BIG_BANK]);
        cache.prefetch_prefill_layer(2, &[BIG_BANK]);
    }

    #[test]
    #[should_panic(expected = "begin_prefill")]
    fn staging_a_layer_without_opening_the_chunk_is_refused() {
        let mut cache = ExpertCache::new(4, 4, 16);
        cache.prefetch_prefill_layer(0, &[BIG_BANK]);
    }

    /// A chunk classifies against the snapshot taken when it opened.
    /// Buffer invalidation is the only writer inside a chunk and it
    /// only ever touches slots below the threshold, which are misses
    /// under both maps -- so the snapshot and the live map agree on
    /// every row, and the caller may order the gather and the host
    /// transfer however it likes.
    #[test]
    fn the_chunk_snapshot_and_the_live_map_agree_on_every_row() {
        let mut cache = ExpertCache::new(4, 4, 16);
        for layer in 0..4u32 {
            cache.ensure(layer, &[0, 1, 2, 3]);
        }
        cache.begin_prefill();
        for layer in [2u32, 3, 2] {
            let live: Vec<Option<u32>> = (0..4)
                .map(|expert| cache.slot_of(layer, expert))
                .map(|slot| slot.filter(|s| *s >= 8))
                .collect();
            let plan = cache.prefetch_prefill_layer(layer, &[BIG_BANK]);
            cache.release_prefill_layer(layer);
            let hits: Vec<u32> = plan.gather.src_slots.clone();
            let live_hits: Vec<u32> = live.into_iter().flatten().collect();
            if !plan.already_loaded {
                assert_eq!(hits, live_hits, "layer {layer} classified differently");
            }
        }
    }

    #[test]
    fn a_pool_that_cannot_lend_the_buffers_their_slots_says_so() {
        let cache = ExpertCache::new(2, 8, 12);
        assert!(!cache.prefill_overlap_fits());
        assert_eq!(cache.prefill_buffer_slots(), 16);

        let fits = ExpertCache::new(2, 8, 16);
        assert!(fits.prefill_overlap_fits());
    }

    #[test]
    #[should_panic(expected = "cannot lend")]
    fn opening_a_chunk_on_a_pool_too_small_for_the_buffers_is_refused() {
        let mut cache = ExpertCache::new(2, 8, 12);
        cache.begin_prefill();
    }
}
