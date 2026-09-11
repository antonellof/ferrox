//! The one place this crate encodes a compute dispatch, and the counters
//! that measure what a decode token costs the host.
//!
//! GitHub issue #149 measured that 26-29% of Metal decode wall time was
//! not GPU time, and read that as host-side command encoding:
//! `dispatchThreadgroups`, `emitComputeProgramVariantAndArguments`,
//! `memoryBarrierWithResources`. Counting was the first step, and the
//! count retired that reading in two stages: PR #156 removed 13% of the
//! dispatches for 2.3% of the host time, and `crate::timing` then
//! clocked the encode phase directly at 0.15 ms per token on
//! Llama-3.2-1B and 0.4 ms on Gemma-2-2B, about 2% of wall. Most of
//! the 26% was a second, untimed command buffer executing on the GPU.
//! The counters stay because a claim about encode work still needs
//! them to be checked.
//!
//! So every `dispatchThreadgroups_threadsPerThreadgroup` in the crate
//! goes through [`dispatch_counted`], and every barrier through
//! [`note_barrier`]. Two counters that must agree with what is actually
//! encoded is the repo's dominant bug shape, so nothing is allowed to
//! bypass them: [`no_encode_site_bypasses_the_counter`] scans the crate's
//! own source and fails if a raw call reappears.
//!
//! Read the counters with [`metal_encode_stats`], reset them with
//! [`metal_encode_stats_reset`]. `FERROX_METAL_GPU_TIMING=1` prints them
//! per token beside the GPU-versus-wall time.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
use std::sync::atomic::{AtomicU64, Ordering};

static DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);
static BARRIER_COUNT: AtomicU64 = AtomicU64::new(0);
static BEGIN_OP_COUNT: AtomicU64 = AtomicU64::new(0);

/// What one encode pass cost the host, in the two units that matter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeStats {
    /// Compute dispatches encoded.
    pub dispatches: u64,
    /// `memoryBarrierWithResources` calls emitted.
    pub barriers: u64,
    /// Hazard checks performed, i.e. ops offered to the tracker. The
    /// `barriers / begin_ops` ratio is 1.00 for a fully serialised pass
    /// and lower when dispatches overlap.
    pub begin_ops: u64,
}

/// Snapshot the process-wide counters.
pub fn metal_encode_stats() -> EncodeStats {
    EncodeStats {
        dispatches: DISPATCH_COUNT.load(Ordering::Relaxed),
        barriers: BARRIER_COUNT.load(Ordering::Relaxed),
        begin_ops: BEGIN_OP_COUNT.load(Ordering::Relaxed),
    }
}

pub fn metal_encode_stats_reset() {
    DISPATCH_COUNT.store(0, Ordering::Relaxed);
    BARRIER_COUNT.store(0, Ordering::Relaxed);
    BEGIN_OP_COUNT.store(0, Ordering::Relaxed);
}

/// Encode one compute dispatch, counting it.
///
/// The counting is the point: see the module docs. Every call site in
/// the crate uses this instead of the raw Metal method.
#[inline]
pub(crate) fn dispatch_counted(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    threadgroups: MTLSize,
    threads_per_threadgroup: MTLSize,
) {
    DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
    enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
}

/// Record one emitted barrier, returning the new running total.
///
/// Called from the two barrier primitives in [`crate::gpu`], which are
/// the only places the crate touches Metal's barrier API, so this counts
/// barriers from the hazard tracker AND the hand-placed ones in the MoE
/// encode paths.
#[inline]
pub(crate) fn note_barrier() -> u64 {
    BARRIER_COUNT.fetch_add(1, Ordering::Relaxed) + 1
}

/// Record one hazard check offered to the tracker (barrier or not).
#[inline]
pub(crate) fn note_begin_op() {
    BEGIN_OP_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    /// A dispatch that does not go through [`super::dispatch_counted`] is
    /// invisible to the per-token accounting, and the accounting is the
    /// only evidence that a fusion removed work. A counter beside the
    /// thing it counts, with nothing forcing them to agree, is this
    /// repo's dominant defect shape, so make the disagreement a red test
    /// rather than a wrong number in a PR body.
    #[test]
    fn no_encode_site_bypasses_the_counter() {
        // Every `.rs` in this crate, read at compile time so a new file
        // that is not listed here also fails to compile this test.
        let sources: &[(&str, &str)] = &[
            ("attn.rs", include_str!("attn.rs")),
            ("capability.rs", include_str!("capability.rs")),
            ("decode_dense.rs", include_str!("decode_dense.rs")),
            ("dispatch.rs", include_str!("dispatch.rs")),
            ("elem.rs", include_str!("elem.rs")),
            ("embd.rs", include_str!("embd.rs")),
            ("gpu.rs", include_str!("gpu.rs")),
            ("kernel_bench.rs", include_str!("kernel_bench.rs")),
            ("kernel_timing.rs", include_str!("kernel_timing.rs")),
            ("lib.rs", include_str!("lib.rs")),
            ("mem_ranges.rs", include_str!("mem_ranges.rs")),
            ("moe_ids.rs", include_str!("moe_ids.rs")),
            ("rope.rs", include_str!("rope.rs")),
            ("timing.rs", include_str!("timing.rs")),
        ];
        // This file holds the wrapper (the one legitimate caller of the
        // raw dispatch method) and the patterns this test searches for,
        // so it is not itself a subject.
        let subjects = || sources.iter().filter(|(name, _)| *name != "dispatch.rs");

        let mut raw = Vec::new();
        for (name, src) in subjects() {
            for (i, line) in src.lines().enumerate() {
                if line.contains(".dispatchThreadgroups_threadsPerThreadgroup(") {
                    raw.push(format!("{name}:{}: {}", i + 1, line.trim()));
                }
            }
        }
        assert!(
            raw.is_empty(),
            "these dispatches bypass dispatch_counted, so the per-token \
             dispatch count under-reports what the host actually encodes: {raw:#?}"
        );

        // And the same for barriers. `memory_barrier_buffers` /
        // `memory_barrier_resources` in gpu.rs are the crate's only
        // callers of Metal's barrier API, and they count what they emit.
        let mut raw_barriers = Vec::new();
        for (name, src) in subjects() {
            for (i, line) in src.lines().enumerate() {
                if !line.contains(".memoryBarrierWith") {
                    continue;
                }
                let inside_the_primitive = *name == "gpu.rs"
                    && (line.contains("encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers)")
                        || line.contains("encoder.memoryBarrierWithResources_count("));
                if !inside_the_primitive {
                    raw_barriers.push(format!("{name}:{}: {}", i + 1, line.trim()));
                }
            }
        }
        assert!(
            raw_barriers.is_empty(),
            "these barriers bypass the counting primitives in gpu.rs and are \
             therefore invisible to the per-token barrier count: {raw_barriers:#?}"
        );
    }
}
