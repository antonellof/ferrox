//! The integer KV-block ledger admission is decided on.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use frink_models::Ceiling;

use crate::budget::ContextCeiling;
use crate::generate::DecodeError;

/// The integer KV-block ledger admission is decided on.
///
/// Blocks are *positions*, not bytes: the scheduler knows its own KV
/// layout, so `blocks_free` is an exact statement about capacity rather
/// than an estimate that needs a safety margin. All mutation happens on
/// the worker thread; the atomics exist so `/metrics` can read the
/// gauge without taking a lock, not to make reservation concurrent.
pub(super) struct BlockBudget {
    pub(super) block_size: usize,
    /// The per-request context ceiling, *shared* with the private
    /// `generate` path (see `crate::budget`). Held as an `Arc` rather
    /// than as a plain limit + shape so the two decode paths cannot
    /// disagree about where the ceiling is or what a position costs --
    /// there is one object, one limit, one counter.
    pub(super) ceiling: Arc<ContextCeiling>,
    /// `None` means no block budget is configured -- every request is
    /// admissible as far as this ledger is concerned.
    pub(super) total: Option<usize>,
    pub(super) free: AtomicUsize,
    /// Requests refused because they exceed the whole server's KV
    /// budget. Split from `QueueGate::rejected` on purpose: one says
    /// "come back later", the other says "this will never work", and an
    /// operator sent to the wrong one of those tunes the wrong knob.
    pub(super) rejected_too_large: AtomicU64,
}

impl BlockBudget {
    pub(super) fn new(
        block_size: usize,
        total: Option<usize>,
        ceiling: Arc<ContextCeiling>,
    ) -> Self {
        assert!(block_size > 0, "kv block size must be positive");
        BlockBudget {
            block_size,
            ceiling,
            total,
            free: AtomicUsize::new(total.unwrap_or(0)),
            rejected_too_large: AtomicU64::new(0),
        }
    }

    /// Prices `positions` of context in real KV bytes, sliding-window
    /// cap included.
    pub(super) fn bytes_for(&self, positions: usize) -> u64 {
        self.ceiling.bytes_for(positions)
    }

    /// The typed refusal for a request of `requested` tokens that will
    /// hold `held` of them at once, or `None` when no immovable ceiling
    /// binds.
    ///
    /// Order matters: the context ceiling is checked first because it
    /// is the request's own size, and telling a client "the machine is
    /// too small" when the real answer is "your prompt is too long"
    /// sends it to a knob it does not have.
    ///
    /// The two arguments differ only for a sliding-window request on a
    /// paged store, where the tail is given back as the window moves.
    /// They are on either side of that split deliberately: a window
    /// bounds the MEMORY, so it belongs to the KV-budget check, and it
    /// does not shorten what the client asked for, so the context
    /// ceiling must not see it. A deployment that caps context at 8k
    /// means 8k on a window model too.
    pub(super) fn immovable_refusal(&self, requested: usize, held: usize) -> Option<DecodeError> {
        if let Some(err) = self.ceiling.refusal(requested) {
            return Some(err);
        }
        let total = self.total?;
        let positions = held;
        let blocks = self.blocks_for(positions);
        if blocks <= total {
            return None;
        }
        self.rejected_too_large.fetch_add(1, Ordering::Relaxed);
        let limit_positions = total * self.block_size;
        Some(DecodeError::KvBudgetExceeded {
            binding: Ceiling::DeviceMemory.code(),
            estimated_bytes: self.bytes_for(positions),
            limit_bytes: self.bytes_for(limit_positions),
            positions,
            positions_limit: limit_positions,
            detail: format!(
                "request needs {blocks} KV blocks ({positions} token positions at {} per \
                 block) but this server's whole KV budget is {total} blocks; an idle server \
                 would refuse it identically",
                self.block_size
            ),
        })
    }

    /// Blocks a sequence of `positions` tokens occupies. At least one:
    /// even a single-token request holds a block.
    pub(super) fn blocks_for(&self, positions: usize) -> usize {
        positions.div_ceil(self.block_size).max(1)
    }

    /// Takes `blocks` if they are there. Worker thread only.
    pub(super) fn try_reserve(&self, blocks: usize) -> bool {
        if self.total.is_none() {
            return true;
        }
        let free = self.free.load(Ordering::Relaxed);
        if blocks > free {
            return false;
        }
        self.free.store(free - blocks, Ordering::Relaxed);
        true
    }

    /// Gives `blocks` back when a request ends, however it ends. Worker
    /// thread only.
    pub(super) fn release(&self, blocks: usize) {
        let Some(total) = self.total else {
            return;
        };
        let free = self.free.load(Ordering::Relaxed);
        debug_assert!(
            free + blocks <= total,
            "released more blocks than were ever reserved"
        );
        self.free
            .store((free + blocks).min(total), Ordering::Relaxed);
    }

    pub(super) fn free(&self) -> usize {
        self.total
            .map(|_| self.free.load(Ordering::Relaxed))
            .unwrap_or(0)
    }
}
