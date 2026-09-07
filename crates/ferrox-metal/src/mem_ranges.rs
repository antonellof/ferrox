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
//! Barriers use [`memory_barrier_resources`] on the pending set (not
//! scope-Buffers as llama does) so weight-buffer traffic from prior
//! matvecs does not stall the next activation-only dispatch — measured
//! ~2× GPU-idle on OLMoE when every conflict used scope-Buffers.
//!
//! Set `FERROX_METAL_BARRIER_LOG=1` to log the running barriers-per-op
//! ratio, which is the direct measure of how much a graph change bought:
//! 1.00 means the pass is fully serialised, lower means dispatches are
//! overlapping.

use crate::gpu::memory_barrier_resources;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};
use std::sync::atomic::{AtomicU64, Ordering};

/// Process-wide barrier count across every tracked encode pass.
static BARRIER_COUNT: AtomicU64 = AtomicU64::new(0);
static BEGIN_OP_COUNT: AtomicU64 = AtomicU64::new(0);

/// True when barrier logging is requested (cached; read once).
fn barrier_log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FERROX_METAL_BARRIER_LOG").is_some())
}

/// Snapshot `(barriers, begin_ops)` since last reset (test / debug).
pub fn metal_barrier_stats() -> (u64, u64) {
    (
        BARRIER_COUNT.load(Ordering::Relaxed),
        BEGIN_OP_COUNT.load(Ordering::Relaxed),
    )
}

pub fn metal_barrier_stats_reset() {
    BARRIER_COUNT.store(0, Ordering::Relaxed);
    BEGIN_OP_COUNT.store(0, Ordering::Relaxed);
}

#[derive(Default)]
pub(crate) struct MemRanges {
    /// Pending Concurrent-set buffers (srcs ∪ dsts), for resource barriers.
    bufs: Vec<*const ProtocolObject<dyn MTLBuffer>>,
    srcs: Vec<usize>,
    dsts: Vec<usize>,
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
        self.bufs.clear();
        self.srcs.clear();
        self.dsts.clear();
    }

    /// Return false if `srcs`/`dsts` conflict with the pending Concurrent set.
    pub(crate) fn check(
        &self,
        srcs: &[&ProtocolObject<dyn MTLBuffer>],
        dsts: &[&ProtocolObject<dyn MTLBuffer>],
    ) -> bool {
        for s in srcs {
            let k = buf_key(s);
            if self.dsts.contains(&k) {
                return false;
            }
        }
        for d in dsts {
            let k = buf_key(d);
            if self.srcs.contains(&k) || self.dsts.contains(&k) {
                return false;
            }
        }
        true
    }

    pub(crate) fn add(
        &mut self,
        srcs: &[&ProtocolObject<dyn MTLBuffer>],
        dsts: &[&ProtocolObject<dyn MTLBuffer>],
    ) {
        for s in srcs {
            let k = buf_key(s);
            if !self.srcs.contains(&k) {
                self.srcs.push(k);
            }
            let p: *const ProtocolObject<dyn MTLBuffer> = *s;
            if !self.bufs.contains(&p) {
                self.bufs.push(p);
            }
        }
        for d in dsts {
            let k = buf_key(d);
            if !self.dsts.contains(&k) {
                self.dsts.push(k);
            }
            let p: *const ProtocolObject<dyn MTLBuffer> = *d;
            if !self.bufs.contains(&p) {
                self.bufs.push(p);
            }
        }
    }

    /// llama `concurrency_check` + optional `concurrency_reset` (barrier).
    pub(crate) fn begin_op(
        &mut self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        srcs: &[&ProtocolObject<dyn MTLBuffer>],
        dsts: &[&ProtocolObject<dyn MTLBuffer>],
    ) {
        BEGIN_OP_COUNT.fetch_add(1, Ordering::Relaxed);
        if !self.check(srcs, dsts) {
            // SAFETY: pointers were taken from live encoder-bound scratch /
            // weight buffers that outlive this encode pass.
            let refs: Vec<&ProtocolObject<dyn MTLBuffer>> =
                self.bufs.iter().map(|&p| unsafe { &*p }).collect();
            memory_barrier_resources(encoder, &refs);
            self.reset();
            let n = BARRIER_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
            if barrier_log_enabled() && n.is_multiple_of(4096) {
                let begins = BEGIN_OP_COUNT.load(Ordering::Relaxed);
                eprintln!(
                    "ferrox: metal barriers={n} begin_ops={begins} (~{:.2} bar/op)",
                    n as f64 / begins.max(1) as f64
                );
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
        let src = include_str!("attn.rs");
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
