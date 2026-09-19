//! Re-splitting the pools of a server that is already running.
//!
//! The arithmetic that decides the split in the first place is
//! [`frink_core::expert_budget`], beside the expert cache whose slots
//! it counts. This module is the other half: validating a *change* to
//! that split against what the engine is doing right now.
//!
//! # Why a rebuild validates before it frees
//!
//! Re-splitting the pools means dropping allocations and taking new
//! ones. If the new size does not fit, the old pool is already gone and
//! the server is down -- for a request that was never going to work.
//! So every size check happens up front, on arithmetic alone
//! ([`validate_rebuild`]), and a rejection leaves the engine serving
//! exactly what it was serving before.
//!
//! Ported 1:1 from FreeToken's `engine/cache_budget.py` (Apache-2.0);
//! see `docs/THIRD_PARTY_NOTICES.md`.

pub(crate) use frink_core::expert_budget::{BudgetTooSmall, PoolSizes};

use frink_core::expert_budget::required_bytes;

/// How many tokens a window request keeps live past the window itself,
/// so a decode step that slides the window does not immediately need
/// state it just freed.
pub const SWA_RETAIN_GAP: usize = 16;

/// How often the window is slid, in decode steps. Sliding every step
/// would cost a pool operation per token; sliding rarely means the pool
/// must hold that many extra tokens per request. The pool floor below
/// pays for exactly this.
pub const DEFAULT_SWA_EVICTION_INTERVAL: usize = 128;

/// A live re-split request. `None` means "leave this pool alone".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RebuildRequest {
    pub moe_cache_slots: Option<u64>,
    pub kv_pages: Option<u64>,
    pub mamba_slots: Option<u64>,
    pub swa_pages: Option<u64>,
}

impl RebuildRequest {
    pub fn is_empty(&self) -> bool {
        self.moe_cache_slots.is_none()
            && self.kv_pages.is_none()
            && self.mamba_slots.is_none()
            && self.swa_pages.is_none()
    }

    /// Whether this request invalidates the prefix cache.
    ///
    /// Any KV-side resize does: a page index, a recurrent slot id and a
    /// window slot id are all positions in an allocation that is about
    /// to stop existing, and a cached prefix pointing into the old one
    /// would hand back another request's state. A MoE-only resize does
    /// not -- the expert cache holds no per-request state.
    pub fn invalidates_prefix_cache(&self) -> bool {
        self.kv_pages.is_some() || self.mamba_slots.is_some() || self.swa_pages.is_some()
    }
}

/// Why a rebuild was refused. The engine is untouched in every case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RebuildRejected {
    /// The server is not idle. A rebuild frees pools that in-flight
    /// requests hold indices into.
    Busy,
    /// The model has no such pool.
    NoSuchPool(&'static str),
    /// The target is below what the pool must hold to function.
    BelowFloor {
        pool: &'static str,
        requested: u64,
        floor: u64,
    },
    /// The split does not fit in the budget.
    TooLarge(BudgetTooSmall),
}

impl std::fmt::Display for RebuildRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RebuildRejected::Busy => {
                write!(f, "the engine is not idle; retry when it drains")
            }
            RebuildRejected::NoSuchPool(pool) => {
                write!(f, "this model has no {pool} pool")
            }
            RebuildRejected::BelowFloor {
                pool,
                requested,
                floor,
            } => write!(
                f,
                "{pool}={requested} is below the working-set floor of {floor}"
            ),
            RebuildRejected::TooLarge(inner) => write!(f, "{inner}"),
        }
    }
}

impl std::error::Error for RebuildRejected {}

/// What the served model actually has, and what each pool must hold to
/// work at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolFloors {
    /// Minimum expert-cache slots: one whole layer.
    pub moe_slots: u64,
    /// Minimum KV pages.
    pub kv_pages: u64,
    /// `None` when the model has no recurrent-state pool.
    pub mamba_slots: Option<u64>,
    /// `None` when the model has no window pool.
    pub swa_pages: Option<u64>,
}

/// Check a rebuild against the floors and the budget, changing nothing.
///
/// `bytes_per_expert` and `bytes_per_page` price the two resizable
/// pools; `fixed_bytes` is everything the target geometry costs that
/// does *not* scale with them -- the window pool at its target size, the
/// recurrent pool at its target size.
///
/// `fixed_bytes` comes off the budget rather than onto the requirement.
/// That is the same arithmetic either way, but it matches the reference
/// (`net_cache_budget_bytes(.., fixed_cache_size + extra_fixed_bytes)`)
/// and it makes the `budget_bytes` carried in a rejection mean the one
/// useful thing: what was actually left for the two resizable pools.
///
/// Passing `0` for a **window model** prices its window pool at nothing.
/// The fit-check then waves through a full pool that leaves no room for
/// the window, and the rebuild dies of an allocation failure *after* it
/// has freed the old cache -- the one failure this whole path exists to
/// prevent. Price a window model with [`hybrid_swa_kv_cost`] at the
/// rebuild's *target* window and pass both of its terms:
/// [`KvCost::cache_per_page`] as `bytes_per_page`,
/// [`KvCost::fixed_cache_size`] as `fixed_bytes`.
#[allow(clippy::too_many_arguments)]
pub fn validate_rebuild(
    request: &RebuildRequest,
    current: &PoolSizes,
    floors: &PoolFloors,
    idle: bool,
    budget_bytes: i64,
    bytes_per_expert: u64,
    bytes_per_page: u64,
    fixed_bytes: u64,
) -> Result<PoolSizes, RebuildRejected> {
    if !idle {
        return Err(RebuildRejected::Busy);
    }
    let below = |pool, requested: u64, floor: u64| RebuildRejected::BelowFloor {
        pool,
        requested,
        floor,
    };
    if let Some(slots) = request.moe_cache_slots {
        if slots < floors.moe_slots {
            return Err(below("moe", slots, floors.moe_slots));
        }
    }
    if let Some(pages) = request.kv_pages {
        if pages < floors.kv_pages.max(2) {
            return Err(below("kv", pages, floors.kv_pages.max(2)));
        }
    }
    if let Some(slots) = request.mamba_slots {
        match floors.mamba_slots {
            None => return Err(RebuildRejected::NoSuchPool("recurrent-state")),
            Some(floor) if slots < floor => return Err(below("mamba", slots, floor)),
            Some(_) => {}
        }
    }
    if let Some(pages) = request.swa_pages {
        match floors.swa_pages {
            None => return Err(RebuildRejected::NoSuchPool("window")),
            Some(floor) if pages < floor => return Err(below("swa", pages, floor)),
            Some(_) => {}
        }
    }

    let target = PoolSizes {
        moe_cache_slots: request.moe_cache_slots.unwrap_or(current.moe_cache_slots),
        kv_pages: request.kv_pages.unwrap_or(current.kv_pages),
        prefill_overlap: current.prefill_overlap,
    };
    let needed = required_bytes(
        target.moe_cache_slots,
        target.kv_pages,
        bytes_per_expert,
        bytes_per_page,
    );
    // Signed, and saturating: an over-committed deployment gets a
    // negative budget that every requirement exceeds, never a
    // wrapped-around enormous one that every requirement fits inside.
    let budget_bytes = budget_bytes.saturating_sub(fixed_bytes as i64);
    if needed > budget_bytes {
        return Err(RebuildRejected::TooLarge(BudgetTooSmall {
            needed_bytes: needed,
            budget_bytes,
            sizes: target,
        }));
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    #[test]
    fn a_rebuild_is_refused_while_the_engine_is_busy() {
        let current = PoolSizes {
            moe_cache_slots: 64,
            kv_pages: 1024,
            prefill_overlap: true,
        };
        let floors = PoolFloors {
            moe_slots: 8,
            kv_pages: 16,
            ..PoolFloors::default()
        };
        let request = RebuildRequest {
            moe_cache_slots: Some(128),
            ..RebuildRequest::default()
        };
        assert_eq!(
            validate_rebuild(&request, &current, &floors, false, i64::MAX, MIB, MIB, 0),
            Err(RebuildRejected::Busy)
        );
    }

    #[test]
    fn a_rebuild_below_a_floor_or_over_the_budget_changes_nothing() {
        let current = PoolSizes {
            moe_cache_slots: 64,
            kv_pages: 1024,
            prefill_overlap: true,
        };
        let floors = PoolFloors {
            moe_slots: 8,
            kv_pages: 16,
            mamba_slots: None,
            swa_pages: Some(32),
        };

        assert!(matches!(
            validate_rebuild(
                &RebuildRequest {
                    moe_cache_slots: Some(4),
                    ..RebuildRequest::default()
                },
                &current,
                &floors,
                true,
                i64::MAX,
                MIB,
                MIB,
                0
            ),
            Err(RebuildRejected::BelowFloor { pool: "moe", .. })
        ));

        // A pool this model does not have is named, not silently
        // accepted.
        assert_eq!(
            validate_rebuild(
                &RebuildRequest {
                    mamba_slots: Some(64),
                    ..RebuildRequest::default()
                },
                &current,
                &floors,
                true,
                i64::MAX,
                MIB,
                MIB,
                0
            ),
            Err(RebuildRejected::NoSuchPool("recurrent-state"))
        );

        assert!(matches!(
            validate_rebuild(
                &RebuildRequest {
                    kv_pages: Some(1 << 40),
                    ..RebuildRequest::default()
                },
                &current,
                &floors,
                true,
                (2 * GIB) as i64,
                MIB,
                MIB,
                0
            ),
            Err(RebuildRejected::TooLarge(_))
        ));
    }

    #[test]
    fn a_rebuild_leaves_untouched_pools_at_their_current_size() {
        let current = PoolSizes {
            moe_cache_slots: 64,
            kv_pages: 1024,
            prefill_overlap: true,
        };
        let floors = PoolFloors {
            moe_slots: 8,
            kv_pages: 16,
            ..PoolFloors::default()
        };
        let target = validate_rebuild(
            &RebuildRequest {
                moe_cache_slots: Some(128),
                ..RebuildRequest::default()
            },
            &current,
            &floors,
            true,
            (4 * GIB) as i64,
            MIB,
            MIB,
            0,
        )
        .expect("fits");
        assert_eq!(target.moe_cache_slots, 128);
        assert_eq!(target.kv_pages, 1024);
    }

    /// Only a KV-side resize invalidates cached prefixes; the expert
    /// cache holds no per-request state.
    #[test]
    fn only_a_kv_side_resize_invalidates_the_prefix_cache() {
        assert!(!RebuildRequest {
            moe_cache_slots: Some(128),
            ..RebuildRequest::default()
        }
        .invalidates_prefix_cache());
        for request in [
            RebuildRequest {
                kv_pages: Some(1),
                ..RebuildRequest::default()
            },
            RebuildRequest {
                mamba_slots: Some(1),
                ..RebuildRequest::default()
            },
            RebuildRequest {
                swa_pages: Some(1),
                ..RebuildRequest::default()
            },
        ] {
            assert!(request.invalidates_prefix_cache());
        }
        assert!(RebuildRequest::default().is_empty());
    }
}
