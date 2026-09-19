//! Admission and chunked prefill: which requests run this step, and how
//! much of each prompt.
//!
//! Everything here is arithmetic over capacities the caller measures.
//! Nothing allocates, nothing runs a forward pass. That is deliberate:
//! the failure this module exists to prevent -- admitting a request the
//! machine cannot finish -- is an arithmetic failure, and it is much
//! easier to be sure about arithmetic that can be run without a GPU.
//!
//! # This is not the batcher, and it is not meant to become it
//!
//! `frink-server`'s [`super::batch`] also admits requests FIFO, and
//! it is tempting to read the two as one policy written twice. They are
//! not, and the difference is the memory model each admits against.
//!
//! The batcher serves the BLOCK model frink ships today: a request
//! reserves `ceil((prompt + max_tokens) / block_size)` blocks for its
//! lifetime. It says so itself -- "frink serves no windowed or
//! recurrent model through this batcher" -- and it contains no
//! sliding-window or recurrent-slot logic at all.
//!
//! This module admits against the WINDOW and RECURRENT models: chunks
//! that must end on a page boundary, a request reclaiming its own
//! slid-out window, and a recurrent slot per request. Those are exactly
//! the cases the batcher excludes.
//!
//! The genuine overlap is "FIFO, bounded by a token budget", which is a
//! `min()` and a subtraction. Routing the live batcher through this
//! module to share that would put a never-run code path on the serving
//! hot path to deduplicate two lines. So: the batcher is authoritative
//! for the block model, this is authoritative for the window and
//! recurrent models, and neither should grow the other's rules.
//!
//! # Prefill first, and no skipping ahead
//!
//! A step runs prefill if any prompt can be admitted, otherwise decode.
//! Admission is strict FIFO with **head-of-line blocking**: the first
//! request that does not fit stops the pass, even if a smaller one
//! behind it would have fit.
//!
//! That is a choice, not an oversight. Skipping ahead is a fairness
//! decision made by memory pressure rather than by policy, and its
//! victim is always the largest request -- the one already waiting
//! longest, which can then be starved indefinitely by a stream of small
//! ones. Blocking makes the queue's order mean what it says.
//!
//! # What a chunk reserves
//!
//! Admitting a prompt reserves the KV for its **whole remaining
//! prompt plus its whole output budget**, not for the chunk being run.
//! A chunked prompt whose later chunks have nowhere to go is worse than
//! one that never started: it holds a request slot and a locked prefix
//! while making no progress. So the reservation is taken once, at
//! admission, against the worst case.
//!
//! Continuations then bypass the gate entirely -- the memory was
//! already reserved, and re-checking could refuse a request that is
//! half-computed.
//!
//! Ported 1:1 from FreeToken's `scheduler/prefill.py`,
//! `scheduler/decode.py`, `scheduler/table.py`, `scheduler/status.py`
//! and the pool-occupancy helpers of `scheduler/scheduler.py`
//! (Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

use crate::policy::radix::{align_ceil, align_down};

/// Request slots: the fixed set of rows the engine's page table has.
///
/// The count is the real concurrency limit, independent of memory --
/// a row is a fixed-width allocation made once at startup.
#[derive(Debug)]
pub struct SlotTable {
    free: Vec<u32>,
    capacity: usize,
}

impl SlotTable {
    pub fn new(capacity: usize) -> Self {
        SlotTable {
            free: (0..capacity as u32).collect(),
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// Take a slot. Last-in-first-out, so a burst of requests reuses
    /// the same few rows and leaves the rest of the page table cold.
    pub fn allocate(&mut self) -> Option<u32> {
        self.free.pop()
    }

    pub fn free(&mut self, slot: u32) {
        debug_assert!(!self.free.contains(&slot), "slot {slot} was freed twice");
        self.free.push(slot);
    }

    /// Resize. Only legal while idle: a live request holds indices into
    /// the page table this is about to replace.
    pub fn rebuild(&mut self, capacity: usize) {
        assert_eq!(
            self.free.len(),
            self.capacity,
            "a slot table may only be rebuilt while idle"
        );
        self.capacity = capacity;
        self.free = (0..capacity as u32).collect();
    }
}

/// One request that is decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodingRequest {
    pub uid: u64,
    /// Tokens it may still produce.
    pub remaining: usize,
}

/// The set of requests currently decoding.
///
/// Every runnable request decodes every step -- there is no admission
/// decision here, because concurrency is already bounded by the slot
/// table. Batches come out sorted by uid so a replayed trace produces
/// byte-identical batches.
#[derive(Debug, Default)]
pub struct DecodeSet {
    running: Vec<DecodingRequest>,
}

impl DecodeSet {
    pub fn new() -> Self {
        DecodeSet::default()
    }

    pub fn len(&self) -> usize {
        self.running.len()
    }

    pub fn is_empty(&self) -> bool {
        self.running.is_empty()
    }

    pub fn runnable(&self) -> bool {
        !self.running.is_empty()
    }

    /// Add newly-prefilled requests and drop the finished ones.
    pub fn admit(&mut self, requests: impl IntoIterator<Item = DecodingRequest>) {
        for request in requests {
            if let Some(slot) = self.running.iter_mut().find(|r| r.uid == request.uid) {
                *slot = request;
            } else {
                self.running.push(request);
            }
        }
        self.running.retain(|r| r.remaining > 0);
    }

    pub fn remove(&mut self, uid: u64) -> Option<DecodingRequest> {
        let index = self.running.iter().position(|r| r.uid == uid)?;
        Some(self.running.remove(index))
    }

    /// KV the running set will still consume before it drains.
    ///
    /// The `page_size - 1` per request is not slack: a request's next
    /// token may land in a fresh page, and a page is allocated whole. A
    /// budget that ignores it admits a prompt that then cannot get its
    /// last page.
    pub fn inflight_tokens(&self, page_size: usize) -> usize {
        let reserved = (page_size - 1) * self.running.len();
        self.running.iter().map(|r| r.remaining).sum::<usize>() + reserved
    }

    /// This step's decode batch, in uid order.
    pub fn next_batch(&self) -> Vec<DecodingRequest> {
        let mut batch = self.running.clone();
        batch.sort_by_key(|r| r.uid);
        batch
    }
}

/// A prompt waiting to be prefilled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingRequest {
    pub uid: u64,
    pub prompt_len: usize,
    /// The output budget, reserved at admission along with the prompt.
    pub output_len: usize,
    /// Set once the request has been admitted and is being prefilled in
    /// chunks.
    pub chunk: Option<ChunkState>,
}

impl PendingRequest {
    pub fn new(uid: u64, prompt_len: usize, output_len: usize) -> Self {
        PendingRequest {
            uid,
            prompt_len,
            output_len,
            chunk: None,
        }
    }

    pub fn is_continuation(&self) -> bool {
        self.chunk.is_some()
    }
}

/// Where a chunked prefill has got to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkState {
    /// Tokens of this prompt already computed.
    pub computed_len: usize,
    /// The request slot it holds.
    pub slot: u32,
    /// How far the request has let its own window state slide out.
    pub swa_evicted_len: usize,
    /// The prefix length the tree owns, which the request may never
    /// free.
    pub locked_prefix_len: usize,
}

/// The capacities an admission decision reads, all measured by the
/// caller.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Capacity {
    /// Free request slots.
    pub request_slots: usize,
    /// KV tokens available: free plus evictable.
    pub kv_tokens: usize,
    /// Window-pool tokens available, for window models.
    pub swa_tokens: usize,
    /// Free recurrent-state slots, for hybrid models.
    pub recurrent_slots: usize,
}

/// The model's shape, as far as admission cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub page_size: usize,
    /// `None` for a model with no window pool.
    pub sliding_window: Option<usize>,
    /// Whether the model has a recurrent-state pool that admission must
    /// reserve slots in.
    pub recurrent: bool,
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            page_size: 1,
            sliding_window: None,
            recurrent: false,
        }
    }
}

/// A recurrent model needs three slots per admitted request: the live
/// state, and two the snapshot ping-pongs between.
const RECURRENT_SLOTS_PER_REQUEST: usize = 3;

/// Why a prompt was not admitted this pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotAdmitted {
    /// No free request slot.
    NoRequestSlot,
    /// The KV it would reserve does not fit alongside what is already
    /// running.
    KvBudget,
    /// The window pool cannot seat it.
    WindowBudget,
    /// The recurrent pool cannot seat it.
    RecurrentBudget,
    /// Its next chunk cannot reach a page boundary, so there is nothing
    /// useful to run this pass.
    NoWholePage,
    /// The pass has no token budget left.
    BudgetSpent,
}

/// A chunk this pass will run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmittedChunk {
    pub uid: u64,
    pub slot: u32,
    /// Prompt tokens already computed before this chunk.
    pub computed_len: usize,
    /// Tokens this chunk computes.
    pub chunk_len: usize,
    /// Whether more chunks follow.
    pub chunked: bool,
    /// Set on a prompt's **first** chunk only: what to report as its
    /// prompt size and prefix hit. Reporting it per chunk would count
    /// one prompt several times.
    pub admission: Option<PromptAdmission>,
}

/// What a prompt cost, reported once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptAdmission {
    pub uid: u64,
    /// The complete prompt, not this chunk.
    pub prompt_tokens: usize,
    /// How much of it the prefix cache already had.
    pub cached_tokens: usize,
}

/// One prefill pass's running state.
///
/// The reservations accumulate across the pass, so the second prompt
/// admitted is checked against a budget the first one has already
/// spent.
#[derive(Debug)]
pub struct PrefillPass {
    token_budget: usize,
    reserved_tokens: usize,
    reserved_swa: usize,
    geometry: Geometry,
}

impl PrefillPass {
    /// Start a pass with `token_budget` tokens of prefill to spend, on
    /// top of what the decoding set will still consume.
    pub fn new(token_budget: usize, inflight_tokens: usize, geometry: Geometry) -> Self {
        PrefillPass {
            token_budget,
            reserved_tokens: inflight_tokens,
            reserved_swa: 0,
            geometry,
        }
    }

    pub fn token_budget(&self) -> usize {
        self.token_budget
    }

    pub fn reserved_tokens(&self) -> usize {
        self.reserved_tokens
    }

    /// Can this prompt be seated at all?
    ///
    /// Checked before anything is locked or allocated, and in this
    /// order, because each answer is cheaper than the next and a
    /// refusal must leave no trace.
    pub fn check_admission(
        &self,
        request: &PendingRequest,
        cached_len: usize,
        capacity: &Capacity,
    ) -> Result<(), NotAdmitted> {
        if capacity.request_slots == 0 {
            return Err(NotAdmitted::NoRequestSlot);
        }
        let extend_len = request.prompt_len.saturating_sub(cached_len);
        // The whole remaining prompt AND the whole output budget: a
        // request that runs out of KV mid-generation has to be dropped
        // after doing all the work.
        let estimated = extend_len + request.output_len;
        if estimated + self.reserved_tokens > capacity.kv_tokens {
            return Err(NotAdmitted::KvBudget);
        }
        if self.geometry.recurrent && capacity.recurrent_slots < RECURRENT_SLOTS_PER_REQUEST {
            return Err(NotAdmitted::RecurrentBudget);
        }
        if let Some(window) = self.geometry.sliding_window {
            // One whole page beyond the window's first token: enough to
            // start, since the window slides as the request decodes.
            let need = align_ceil(extend_len.max(1).min(window) + 1, self.geometry.page_size);
            if capacity.swa_tokens.saturating_sub(self.reserved_swa) < need {
                return Err(NotAdmitted::WindowBudget);
            }
        }
        Ok(())
    }

    /// Size the chunk this pass will run for `request`, charging the
    /// pass's budgets for it.
    ///
    /// Call after [`check_admission`](Self::check_admission) for a new
    /// prompt, or directly for a continuation.
    pub fn take_chunk(
        &mut self,
        request: &PendingRequest,
        cached_len: usize,
        slot: u32,
        capacity: &Capacity,
    ) -> Result<AdmittedChunk, NotAdmitted> {
        if self.token_budget == 0 {
            return Err(NotAdmitted::BudgetSpent);
        }
        let computed_len = match request.chunk {
            Some(chunk) => chunk.computed_len,
            None => cached_len,
        };
        let remaining = request.prompt_len.saturating_sub(computed_len);
        let mut chunk_len = self.token_budget.min(remaining);

        if let Some(window) = self.geometry.sliding_window {
            chunk_len =
                self.window_bounded_chunk(request, computed_len, chunk_len, window, capacity)?;
        }
        if chunk_len == 0 {
            return Err(NotAdmitted::NoWholePage);
        }

        let chunked = chunk_len < remaining;
        self.token_budget -= chunk_len;
        // Charged once, on the first chunk: the whole remaining prompt
        // and the whole output budget.
        if request.chunk.is_none() {
            self.reserved_tokens += remaining + request.output_len;
        }

        Ok(AdmittedChunk {
            uid: request.uid,
            slot,
            computed_len,
            chunk_len,
            chunked,
            admission: request.chunk.is_none().then_some(PromptAdmission {
                uid: request.uid,
                prompt_tokens: request.prompt_len,
                cached_tokens: cached_len,
            }),
        })
    }

    /// A window model's chunk is bounded by how much window state the
    /// pool can seat, and a continuation must end on a page boundary.
    ///
    /// The page rule is what makes the window pool's accounting
    /// tractable: window slots are charged per whole page, so a chunk
    /// that stops mid-page charges a page for tokens the next chunk
    /// then charges again.
    fn window_bounded_chunk(
        &mut self,
        request: &PendingRequest,
        computed_len: usize,
        chunk_len: usize,
        window: usize,
        capacity: &Capacity,
    ) -> Result<usize, NotAdmitted> {
        let page = self.geometry.page_size;
        let chunk = request.chunk.unwrap_or(ChunkState {
            computed_len,
            slot: 0,
            swa_evicted_len: 0,
            locked_prefix_len: 0,
        });
        // What this request can free for itself as its own window slides
        // past what it has already computed. That memory is not in
        // `capacity`, but it is genuinely available to this request.
        let newly_evicted = align_down(computed_len.saturating_sub(window + page), page);
        let already_gone = chunk.swa_evicted_len.max(chunk.locked_prefix_len);
        let self_reclaim = newly_evicted.saturating_sub(already_gone);

        let budget = (capacity.swa_tokens + self_reclaim).saturating_sub(self.reserved_swa);
        let max_end = (computed_len.div_ceil(page) + budget / page) * page;
        let mut bounded = chunk_len.min(max_end.saturating_sub(computed_len));

        let remaining = request.prompt_len.saturating_sub(computed_len);
        if bounded > 0 && bounded < remaining {
            let aligned = align_down(computed_len + bounded, page).saturating_sub(computed_len);
            if aligned == 0 {
                // Not even one whole page fits this pass. Running a
                // partial page would charge the pool for it twice.
                return Err(NotAdmitted::NoWholePage);
            }
            bounded = aligned;
        }
        self.reserved_swa +=
            ((computed_len + bounded).div_ceil(page) - computed_len.div_ceil(page)) * page;
        Ok(bounded)
    }
}

/// Why a generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model ended its own turn, or produced a stop string.
    Stop,
    /// The output budget ran out.
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// Decide why a step ended, if it did.
///
/// The precedence matters: a model that emits its end-of-turn token *as*
/// its last budgeted token finished on its own terms, and reporting
/// `length` there would tell a client to ask for a continuation of a
/// complete answer.
pub fn finish_reason(
    hit_eos: bool,
    matched_stop: bool,
    budget_exhausted: bool,
) -> Option<FinishReason> {
    if hit_eos || matched_stop {
        Some(FinishReason::Stop)
    } else if budget_exhausted {
        Some(FinishReason::Length)
    } else {
        None
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn geometry() -> Geometry {
        Geometry {
            page_size: 16,
            ..Geometry::default()
        }
    }

    pub(crate) fn roomy() -> Capacity {
        Capacity {
            request_slots: 4,
            kv_tokens: 100_000,
            swa_tokens: 100_000,
            recurrent_slots: 64,
        }
    }

    #[test]
    fn slots_are_handed_out_last_in_first_out() {
        let mut table = SlotTable::new(3);
        assert_eq!(table.available(), 3);
        assert_eq!(table.allocate(), Some(2));
        assert_eq!(table.allocate(), Some(1));
        table.free(2);
        assert_eq!(table.allocate(), Some(2), "the most recent slot comes back");
        assert_eq!(table.allocate(), Some(0));
        assert_eq!(table.allocate(), None);
    }

    #[test]
    #[should_panic(expected = "only be rebuilt while idle")]
    fn a_slot_table_with_a_live_request_may_not_be_rebuilt() {
        let mut table = SlotTable::new(3);
        table.allocate();
        table.rebuild(8);
    }

    /// The page reservation is not slack: the next token may open a
    /// fresh page, and pages are allocated whole.
    #[test]
    fn inflight_tokens_reserve_a_page_per_running_request() {
        let mut decode = DecodeSet::new();
        decode.admit([
            DecodingRequest {
                uid: 1,
                remaining: 100,
            },
            DecodingRequest {
                uid: 2,
                remaining: 50,
            },
        ]);
        assert_eq!(decode.inflight_tokens(16), 150 + 2 * 15);
        assert_eq!(decode.inflight_tokens(1), 150, "no page, no reservation");
    }

    #[test]
    fn a_finished_request_leaves_the_decode_set() {
        let mut decode = DecodeSet::new();
        decode.admit([DecodingRequest {
            uid: 1,
            remaining: 1,
        }]);
        assert!(decode.runnable());
        decode.admit([DecodingRequest {
            uid: 1,
            remaining: 0,
        }]);
        assert!(!decode.runnable());
        assert!(decode.next_batch().is_empty());
    }

    #[test]
    fn decode_batches_come_out_in_uid_order() {
        let mut decode = DecodeSet::new();
        decode.admit([
            DecodingRequest {
                uid: 7,
                remaining: 4,
            },
            DecodingRequest {
                uid: 2,
                remaining: 4,
            },
        ]);
        let uids: Vec<u64> = decode.next_batch().iter().map(|r| r.uid).collect();
        assert_eq!(uids, vec![2, 7]);
    }

    /// The reservation covers the whole output budget, so a request
    /// never gets half-generated and then dropped.
    #[test]
    fn admission_reserves_the_whole_prompt_and_output() {
        let mut pass = PrefillPass::new(8192, 0, geometry());
        let request = PendingRequest::new(1, 1000, 500);
        let capacity = Capacity {
            request_slots: 1,
            kv_tokens: 1400, // 1000 + 500 = 1500 > 1400
            ..roomy()
        };
        assert_eq!(
            pass.check_admission(&request, 0, &capacity),
            Err(NotAdmitted::KvBudget)
        );

        let capacity = Capacity {
            kv_tokens: 1500,
            ..capacity
        };
        assert!(pass.check_admission(&request, 0, &capacity).is_ok());
        let chunk = pass.take_chunk(&request, 0, 0, &capacity).unwrap();
        assert_eq!(pass.reserved_tokens(), 1500);
        assert_eq!(chunk.chunk_len, 1000, "the budget covers the whole prompt");
        assert!(!chunk.chunked);
    }

    /// A prefix hit shrinks what must be computed *and* what must be
    /// reserved.
    #[test]
    fn a_prefix_hit_shrinks_the_reservation() {
        let mut pass = PrefillPass::new(8192, 0, geometry());
        let request = PendingRequest::new(1, 1000, 500);
        let capacity = Capacity {
            kv_tokens: 800,
            ..roomy()
        };
        assert_eq!(
            pass.check_admission(&request, 0, &capacity),
            Err(NotAdmitted::KvBudget)
        );
        assert!(pass.check_admission(&request, 800, &capacity).is_ok());

        let chunk = pass.take_chunk(&request, 800, 0, &capacity).unwrap();
        assert_eq!(chunk.chunk_len, 200);
        assert_eq!(chunk.computed_len, 800);
        assert_eq!(pass.reserved_tokens(), 700);
    }

    #[test]
    fn no_free_slot_refuses_before_anything_else_is_computed() {
        let pass = PrefillPass::new(8192, 0, geometry());
        let capacity = Capacity {
            request_slots: 0,
            ..roomy()
        };
        assert_eq!(
            pass.check_admission(&PendingRequest::new(1, 8, 8), 0, &capacity),
            Err(NotAdmitted::NoRequestSlot)
        );
    }

    #[test]
    fn a_recurrent_model_seats_three_slots_per_request() {
        let geometry = Geometry {
            recurrent: true,
            ..geometry()
        };
        let pass = PrefillPass::new(8192, 0, geometry);
        let request = PendingRequest::new(1, 64, 8);
        assert!(pass
            .check_admission(
                &request,
                0,
                &Capacity {
                    recurrent_slots: 3,
                    ..roomy()
                }
            )
            .is_ok());
        assert_eq!(
            pass.check_admission(
                &request,
                0,
                &Capacity {
                    recurrent_slots: 2,
                    ..roomy()
                }
            ),
            Err(NotAdmitted::RecurrentBudget)
        );
    }

    #[test]
    fn a_window_model_needs_a_seat_in_the_window_pool() {
        let geometry = Geometry {
            sliding_window: Some(512),
            ..geometry()
        };
        let pass = PrefillPass::new(8192, 0, geometry);
        let request = PendingRequest::new(1, 2000, 100);
        assert_eq!(
            pass.check_admission(
                &request,
                0,
                &Capacity {
                    swa_tokens: 16,
                    ..roomy()
                }
            ),
            Err(NotAdmitted::WindowBudget)
        );
        assert!(pass
            .check_admission(
                &request,
                0,
                &Capacity {
                    swa_tokens: 528,
                    ..roomy()
                }
            )
            .is_ok());
    }

    /// A long prompt is cut into chunks, and the reservation is taken
    /// once rather than per chunk.
    #[test]
    fn a_long_prompt_is_chunked_and_charged_once() {
        let mut pass = PrefillPass::new(256, 0, geometry());
        let request = PendingRequest::new(1, 1000, 100);
        let capacity = roomy();

        let first = pass.take_chunk(&request, 0, 3, &capacity).unwrap();
        assert_eq!(first.chunk_len, 256);
        assert!(first.chunked);
        assert_eq!(
            first.admission,
            Some(PromptAdmission {
                uid: 1,
                prompt_tokens: 1000,
                cached_tokens: 0
            }),
            "the whole prompt is reported, not the chunk"
        );
        assert_eq!(pass.reserved_tokens(), 1100);
        assert_eq!(pass.token_budget(), 0);

        // The continuation, next pass.
        let mut pass = PrefillPass::new(256, 1100, geometry());
        let continued = PendingRequest {
            chunk: Some(ChunkState {
                computed_len: 256,
                slot: 3,
                swa_evicted_len: 0,
                locked_prefix_len: 0,
            }),
            ..request
        };
        let second = pass.take_chunk(&continued, 0, 3, &capacity).unwrap();
        assert_eq!(second.computed_len, 256);
        assert_eq!(second.chunk_len, 256);
        assert_eq!(second.admission, None, "reported once, on the first chunk");
        assert_eq!(
            pass.reserved_tokens(),
            1100,
            "a continuation reserves nothing new"
        );
    }

    /// A window model's continuation must stop on a page boundary, or
    /// the window pool is charged for the same page twice.
    #[test]
    fn a_window_continuation_stops_on_a_page_boundary() {
        let geometry = Geometry {
            page_size: 128,
            sliding_window: Some(512),
            ..Geometry::default()
        };
        // A budget that is not a multiple of the page size.
        let mut pass = PrefillPass::new(724, 0, geometry);
        let request = PendingRequest::new(1, 2000, 100);
        let chunk = pass.take_chunk(&request, 0, 0, &roomy()).unwrap();
        assert_eq!(chunk.chunk_len, 640, "5 whole pages, not 724");
        assert!(chunk.chunked);
    }

    /// ... but the *final* chunk may be ragged: there is nothing after
    /// it to charge twice.
    #[test]
    fn the_last_chunk_may_end_mid_page() {
        let geometry = Geometry {
            page_size: 128,
            sliding_window: Some(512),
            ..Geometry::default()
        };
        let mut pass = PrefillPass::new(8192, 0, geometry);
        let request = PendingRequest::new(1, 700, 100);
        let chunk = pass.take_chunk(&request, 0, 0, &roomy()).unwrap();
        assert_eq!(chunk.chunk_len, 700);
        assert!(!chunk.chunked);
    }

    /// A pass with too little window budget to reach one page has
    /// nothing useful to run, and says so rather than running a partial
    /// page.
    #[test]
    fn a_chunk_that_cannot_reach_a_page_boundary_is_refused() {
        let geometry = Geometry {
            page_size: 128,
            sliding_window: Some(512),
            ..Geometry::default()
        };
        let mut pass = PrefillPass::new(8192, 0, geometry);
        let request = PendingRequest::new(1, 2000, 100);
        let capacity = Capacity {
            swa_tokens: 64, // less than one page
            ..roomy()
        };
        assert_eq!(
            pass.take_chunk(&request, 0, 0, &capacity),
            Err(NotAdmitted::NoWholePage)
        );
    }

    /// A long-running window request reclaims its own window as it
    /// slides, and that memory counts toward its next chunk.
    #[test]
    fn a_sliding_window_request_reclaims_its_own_state() {
        let geometry = Geometry {
            page_size: 128,
            sliding_window: Some(512),
            ..Geometry::default()
        };
        let capacity = Capacity {
            swa_tokens: 256,
            ..roomy()
        };
        let deep = PendingRequest {
            chunk: Some(ChunkState {
                computed_len: 4096,
                slot: 0,
                swa_evicted_len: 0,
                locked_prefix_len: 0,
            }),
            ..PendingRequest::new(1, 8192, 100)
        };
        let mut pass = PrefillPass::new(8192, 0, geometry);
        let chunk = pass.take_chunk(&deep, 0, 0, &capacity).unwrap();
        assert!(
            chunk.chunk_len > 256,
            "the request's own slid-out window paid for more than the pool had ({})",
            chunk.chunk_len
        );

        // A request that has already given that state up gets only what
        // the pool has.
        let already = PendingRequest {
            chunk: Some(ChunkState {
                computed_len: 4096,
                slot: 0,
                swa_evicted_len: 4096,
                locked_prefix_len: 0,
            }),
            ..deep
        };
        let mut pass = PrefillPass::new(8192, 0, geometry);
        let chunk = pass.take_chunk(&already, 0, 0, &capacity).unwrap();
        assert_eq!(chunk.chunk_len, 256);
    }

    /// The pass's reservations accumulate: the second prompt is checked
    /// against a budget the first already spent.
    #[test]
    fn one_pass_charges_each_admission_against_the_last() {
        let capacity = Capacity {
            kv_tokens: 1000,
            ..roomy()
        };
        let mut pass = PrefillPass::new(8192, 0, geometry());
        let first = PendingRequest::new(1, 400, 200);
        assert!(pass.check_admission(&first, 0, &capacity).is_ok());
        pass.take_chunk(&first, 0, 0, &capacity).unwrap();

        let second = PendingRequest::new(2, 400, 200);
        assert_eq!(
            pass.check_admission(&second, 0, &capacity),
            Err(NotAdmitted::KvBudget),
            "600 already reserved leaves 400 for a request that needs 600"
        );
    }

    /// A model that ends its turn on its last budgeted token finished
    /// on its own terms.
    #[test]
    fn stopping_beats_running_out_of_budget() {
        assert_eq!(finish_reason(true, false, true), Some(FinishReason::Stop));
        assert_eq!(finish_reason(false, true, true), Some(FinishReason::Stop));
        assert_eq!(
            finish_reason(false, false, true),
            Some(FinishReason::Length)
        );
        assert_eq!(finish_reason(false, false, false), None);
        assert_eq!(FinishReason::Stop.as_str(), "stop");
    }
}
