//! llama.cpp-style Concurrent hazard tracking for Metal encode passes.
//!
//! Mirrors `ggml_mem_ranges` (`ggml-metal-common.cpp`): two ops may overlap
//! under `MTLDispatchTypeConcurrent` when they don't write memory another
//! op in the set reads or writes. SRC∩SRC is allowed; SRC∩DST and DST∩*
//! require a barrier + range reset. The encode loop is llama's
//! `ggml_metal_op_encode`:
//!
//! ```text
//! if (!concurrency_check(node)) { memory_barrier(); mem_ranges_reset(); }
//! ...encode the dispatch...
//! concurrency_add(node);   // srcs ∪ dst
//! ```
//!
//! Ferrox tracks whole MTLBuffer identities (llama uses byte ranges on the
//! alloc — equivalent for our non-view scratch buffers, each of which is
//! its own `MTLBuffer` and is always touched whole).
//!
//! Barriers use [`memory_barrier_resource_list`] on the pending set (not
//! scope-Buffers as llama does) so weight-buffer traffic from prior
//! matvecs does not stall the next activation-only dispatch — measured
//! ~2× GPU-idle on OLMoE when every conflict used scope-Buffers.
//!
//! Set `FERROX_METAL_BARRIER_LOG=1` to log the running barriers-per-op
//! ratio, which is the direct measure of how much a graph change bought:
//! 1.00 means the pass is fully serialised, lower means dispatches are
//! overlapping.

use crate::dispatch::{metal_encode_stats, note_begin_op};
use crate::gpu::memory_barrier_resource_list;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLResource};
use std::ptr::NonNull;

/// True when barrier logging is requested (cached; read once).
fn barrier_log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FERROX_METAL_BARRIER_LOG").is_some())
}

#[derive(Default)]
pub(crate) struct MemRanges {
    /// Buffer identities read by a pending op.
    srcs: Vec<usize>,
    /// Buffer identities written by a pending op.
    dsts: Vec<usize>,
    /// The same buffers as `srcs ∪ dsts`, in the form the barrier API
    /// wants. Written only by [`MemRanges::push_src`] /
    /// [`MemRanges::push_dst`], which push here and to the key list in
    /// the same call, so the resource list cannot fall out of step with
    /// the ranges it is supposed to order.
    ///
    /// Kept across barriers (cleared, not dropped) because a decode token
    /// takes ~160 of them and this used to allocate twice per barrier on
    /// the per-token path.
    res: Vec<NonNull<ProtocolObject<dyn MTLResource>>>,
}

#[inline]
fn buf_key(b: &ProtocolObject<dyn MTLBuffer>) -> usize {
    b as *const ProtocolObject<dyn MTLBuffer> as usize
}

impl MemRanges {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn reset(&mut self) {
        self.srcs.clear();
        self.dsts.clear();
        self.res.clear();
    }

    /// The hazard rule, stated once on opaque identities so it can be
    /// tested without a GPU: an op conflicts with the pending set when it
    /// READS something pending writes, or WRITES something pending reads
    /// or writes. Two reads of the same buffer do not conflict.
    fn keys_conflict(
        &self,
        srcs: impl Iterator<Item = usize>,
        dsts: impl Iterator<Item = usize>,
    ) -> bool {
        for k in srcs {
            if self.dsts.contains(&k) {
                return true;
            }
        }
        for k in dsts {
            if self.srcs.contains(&k) || self.dsts.contains(&k) {
                return true;
            }
        }
        false
    }

    fn push_res(&mut self, b: &ProtocolObject<dyn MTLBuffer>) {
        let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(b);
        let p = NonNull::from(r);
        if !self.res.contains(&p) {
            self.res.push(p);
        }
    }

    fn push_src(&mut self, b: &ProtocolObject<dyn MTLBuffer>) {
        let k = buf_key(b);
        if !self.srcs.contains(&k) {
            self.srcs.push(k);
        }
        self.push_res(b);
    }

    fn push_dst(&mut self, b: &ProtocolObject<dyn MTLBuffer>) {
        let k = buf_key(b);
        if !self.dsts.contains(&k) {
            self.dsts.push(k);
        }
        self.push_res(b);
    }

    /// Return false if `srcs`/`dsts` conflict with the pending Concurrent set.
    pub(crate) fn check(
        &self,
        srcs: &[&ProtocolObject<dyn MTLBuffer>],
        dsts: &[&ProtocolObject<dyn MTLBuffer>],
    ) -> bool {
        !self.keys_conflict(
            srcs.iter().map(|b| buf_key(b)),
            dsts.iter().map(|b| buf_key(b)),
        )
    }

    pub(crate) fn add(
        &mut self,
        srcs: &[&ProtocolObject<dyn MTLBuffer>],
        dsts: &[&ProtocolObject<dyn MTLBuffer>],
    ) {
        for s in srcs {
            self.push_src(s);
        }
        for d in dsts {
            self.push_dst(d);
        }
    }

    /// llama `concurrency_check` + optional `concurrency_reset` (barrier).
    ///
    /// A barrier is emitted ONLY on a real hazard, never per op: an op
    /// whose reads and writes are disjoint from everything pending is
    /// encoded straight into the concurrent pass. `FERROX_METAL_GPU_TIMING`
    /// reports the resulting barriers-per-op ratio, which on a dense decode
    /// token is ~0.99 -- the chain really is serial -- while the four
    /// intra-group pairs (Q∥K∥V, gate∥up) overlap for free.
    pub(crate) fn begin_op(
        &mut self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        srcs: &[&ProtocolObject<dyn MTLBuffer>],
        dsts: &[&ProtocolObject<dyn MTLBuffer>],
    ) {
        note_begin_op();
        if !self.check(srcs, dsts) {
            // Counts itself: see `crate::dispatch`.
            memory_barrier_resource_list(encoder, &mut self.res);
            self.reset();
            if barrier_log_enabled() {
                let s = metal_encode_stats();
                if s.barriers.is_multiple_of(4096) {
                    eprintln!(
                        "ferrox: metal barriers={} begin_ops={} (~{:.2} bar/op)",
                        s.barriers,
                        s.begin_ops,
                        s.barriers as f64 / s.begin_ops.max(1) as f64
                    );
                }
            }
        }
    }

    pub(crate) fn end_op(
        &mut self,
        srcs: &[&ProtocolObject<dyn MTLBuffer>],
        dsts: &[&ProtocolObject<dyn MTLBuffer>],
    ) {
        self.add(srcs, dsts);
    }
}

#[cfg(test)]
mod hazard_tests {
    use super::MemRanges;

    /// Build a tracker whose pending set is the given identities. The
    /// hazard rule is pure -- it compares buffer identities, never touches
    /// the GPU -- so it is tested here with plain integers rather than
    /// behind `#[ignore]` on hardware.
    fn pending(srcs: &[usize], dsts: &[usize]) -> MemRanges {
        MemRanges {
            srcs: srcs.to_vec(),
            dsts: dsts.to_vec(),
            res: Vec::new(),
        }
    }

    /// A barrier per encoded op would read as correctness and cost the
    /// whole point of the concurrent encoder: on a Llama-3.2-1B decode
    /// token that is ~240 `memoryBarrierWithResources` calls instead of
    /// ~160, and command encoding is 26-29% of Metal decode wall time
    /// (GitHub issue #149). Disjoint work must stay barrier-free.
    #[test]
    fn a_write_then_a_read_of_a_different_buffer_needs_no_barrier() {
        // Pending: op wrote buffer 2, reading buffer 1.
        let m = pending(&[1], &[2]);
        // A later op reading 3 and writing 4 touches neither.
        assert!(
            !m.keys_conflict([3].into_iter(), [4].into_iter()),
            "disjoint ranges must not force a barrier"
        );
        // Two READS of the same buffer are also free: SRC∩SRC is allowed.
        assert!(
            !m.keys_conflict([1].into_iter(), [4].into_iter()),
            "two ops reading the same buffer must not force a barrier"
        );
    }

    /// And the other half: an overlap really is ordered, exactly once.
    #[test]
    fn a_read_after_a_pending_write_needs_exactly_one_barrier() {
        let m = pending(&[1], &[2]);
        // Read-after-write on buffer 2.
        assert!(m.keys_conflict([2].into_iter(), [4].into_iter()));
        // Write-after-write on buffer 2.
        assert!(m.keys_conflict([3].into_iter(), [2].into_iter()));
        // Write-after-read on buffer 1.
        assert!(m.keys_conflict([3].into_iter(), [1].into_iter()));

        // One conflicting op reports ONE conflict however many buffers it
        // names, so the encode loop emits one barrier and resets, rather
        // than one barrier per overlapping buffer.
        assert!(m.keys_conflict([2, 2, 2].into_iter(), [1, 2].into_iter()));
    }

    /// `reset` after a barrier must clear the resource list too, or the
    /// next barrier orders buffers no pending op touches -- which is a
    /// silently wider barrier, not a wrong answer, and so would never
    /// show up as a failure anywhere else.
    #[test]
    fn a_barrier_reset_clears_the_resource_list_with_the_ranges() {
        let mut m = pending(&[1], &[2]);
        m.res.push(std::ptr::NonNull::dangling());
        m.reset();
        assert!(m.srcs.is_empty());
        assert!(m.dsts.is_empty());
        assert!(
            m.res.is_empty(),
            "the resource list is the pending set in another form; \
             clearing one without the other is how they drift"
        );
    }
}

#[cfg(test)]
mod declaration_tests {
    /// Every dispatch encoded into the decode stack must declare what it
    /// reads and writes, or the concurrent encoder is free to run it
    /// against data still in flight.
    ///
    /// This is the failure that used to be papered over: Gemma's post-norm
    /// layers diverged under concurrent dispatch, and the fix disabled
    /// concurrency for that whole class of model rather than finding the
    /// undeclared op. Concurrency is only safe by CONSTRUCTION -- every op
    /// declared -- and nothing was checking that construction held.
    ///
    /// An op declares itself one of two ways: the call site wraps it in
    /// `begin_op`/`end_op`, or the helper takes `&mut mrs` and tracks its
    /// own hazards (`encode_gqa_with_kv` does, because with a quantized KV
    /// cache it writes an f16 dequant scratch no caller can name).
    #[test]
    fn every_encode_in_the_decode_stack_declares_its_hazards() {
        let src = include_str!("decode_dense.rs");
        let start = src
            .find("pub fn launch_decode_dense_stack(")
            .expect("decode stack function");
        // Stop at the next top-level item.
        let body = &src[start..];
        let end = body[1..]
            .find("\npub fn ")
            .map(|i| i + 1)
            .unwrap_or(body.len());
        let body = &body[..end];

        let mut open = false;
        let mut undeclared = Vec::new();
        for (i, line) in body.lines().enumerate() {
            let t = line.trim();
            if t.starts_with("mrs.begin_op(") {
                open = true;
            }
            if t.starts_with("mrs.end_op(") {
                open = false;
            }
            let is_encode = t.starts_with("encode_") && t.contains('(');
            if is_encode && !open {
                // A self-tracking helper takes the tracker itself. The call
                // spans several lines, so look at the next few.
                let window: String = body.lines().skip(i).take(6).collect::<Vec<_>>().join(" ");
                // The tracker is handed over either as `&mut mrs` or, where
                // the caller already holds `&mut MemRanges`, as bare `mrs`.
                let handed_over = window.contains("&mut mrs") || window.contains(" mrs,");
                if !handed_over {
                    undeclared.push(t.to_string());
                }
            }
        }
        assert!(
            undeclared.is_empty(),
            "these dispatches declare no reads/writes and are not self-tracking, \
             so the concurrent encoder may run them against in-flight data: {undeclared:#?}"
        );
    }
}
