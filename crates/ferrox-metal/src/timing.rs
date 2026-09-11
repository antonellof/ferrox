//! Where a Metal submission's time goes, measured rather than inferred.
//!
//! Two instruments, both opt-in by environment variable and both
//! process-wide:
//!
//! - `FERROX_METAL_MM_TIMING`: prefill GEMM wall-clock totals
//!   (setup / GPU wait / readback), see [`mm_timing_add`].
//! - `FERROX_METAL_GPU_TIMING`: per-tag GPU-clock time of each command
//!   buffer beside the encode work that bought it, see
//!   [`gpu_timing_note`].
//!
//! Moved out of `gpu.rs` unchanged along the seam GitHub issue #149
//! touches.

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;
use std::sync::atomic::{AtomicU64, Ordering as AtomOrd};
static MM_SETUP_US: AtomicU64 = AtomicU64::new(0);
static MM_GPU_US: AtomicU64 = AtomicU64::new(0);
static MM_READ_US: AtomicU64 = AtomicU64::new(0);
static MM_CALLS: AtomicU64 = AtomicU64::new(0);

/// Accumulate into the `FERROX_METAL_MM_TIMING` counters (prefill evidence).
pub(crate) fn mm_timing_add(setup: u128, gpu: u128, read: u128) {
    MM_SETUP_US.fetch_add(setup as u64, AtomOrd::Relaxed);
    MM_GPU_US.fetch_add(gpu as u64, AtomOrd::Relaxed);
    MM_READ_US.fetch_add(read as u64, AtomOrd::Relaxed);
    let n = MM_CALLS.fetch_add(1, AtomOrd::Relaxed) + 1;
    // First call + every 224: stack prefill is often one timed launch.
    if n == 1 || n.is_multiple_of(224) {
        eprintln!(
            "ferrox: mul_mm {n} calls -- setup {:.1} ms, gpu {:.1} ms, readback {:.1} ms",
            MM_SETUP_US.load(AtomOrd::Relaxed) as f64 / 1000.0,
            MM_GPU_US.load(AtomOrd::Relaxed) as f64 / 1000.0,
            MM_READ_US.load(AtomOrd::Relaxed) as f64 / 1000.0,
        );
    }
}

/// GPU-clock accumulators for `FERROX_METAL_GPU_TIMING`, keyed by tag.
///
/// Wall-clock (`FERROX_METAL_MM_TIMING`) measures how long the host waited,
/// which on a loaded host is mostly scheduler noise. `MTLCommandBuffer`'s
/// `GPUStartTime`/`GPUEndTime` measure the command buffer's own occupancy
/// and stay usable while the machine is busy, so A/B evidence in
/// `docs/plans/llama-cpp-parity-push.md` is taken from these.
static GPU_TIMING: std::sync::Mutex<Vec<TimingSlot>> = std::sync::Mutex::new(Vec::new());

/// One tag's running GPU-time total, plus what the host encoded to
/// produce it.
///
/// The encode counters in [`crate::dispatch`] are process-wide, so a
/// per-submission figure has to come from the DELTA between consecutive
/// submissions of the same tag rather than from a total divided by a
/// count -- otherwise a prefill's dispatches land in the decode average.
/// GitHub issue #149 is about the host cost of encoding, so this number
/// belongs next to the GPU time it bought, not in a separate readout
/// somebody has to line up by hand.
struct TimingSlot {
    tag: &'static str,
    n: u64,
    acc_ns: u64,
    /// Counter snapshot at the previous submission under this tag.
    prev: crate::dispatch::EncodeStats,
    /// Encode work attributed to this tag, summed over `n` submissions.
    acc_dispatches: u64,
    acc_barriers: u64,
    acc_begin_ops: u64,
}

/// True when `FERROX_METAL_GPU_TIMING` is set (cached; read once).
pub(crate) fn gpu_timing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FERROX_METAL_GPU_TIMING").is_some())
}

/// Accumulate one command buffer's GPU-side duration under `tag` and log a
/// running mean every `every` submissions. No-op unless timing is enabled.
pub(crate) fn gpu_timing_note(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    tag: &'static str,
    every: u64,
) {
    if !gpu_timing_enabled() {
        return;
    }
    let dt_ns = ((cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e9).max(0.0) as u64;
    let now = crate::dispatch::metal_encode_stats();
    let Ok(mut slots) = GPU_TIMING.lock() else {
        return;
    };
    let slot = match slots.iter_mut().find(|s| s.tag == tag) {
        Some(s) => s,
        None => {
            slots.push(TimingSlot {
                tag,
                n: 0,
                acc_ns: 0,
                prev: now,
                acc_dispatches: 0,
                acc_barriers: 0,
                acc_begin_ops: 0,
            });
            slots.last_mut().expect("just pushed")
        }
    };
    slot.n += 1;
    slot.acc_ns += dt_ns;
    slot.acc_dispatches += now.dispatches.saturating_sub(slot.prev.dispatches);
    slot.acc_barriers += now.barriers.saturating_sub(slot.prev.barriers);
    slot.acc_begin_ops += now.begin_ops.saturating_sub(slot.prev.begin_ops);
    slot.prev = now;
    let n = slot.n;
    if n.is_multiple_of(every.max(1)) {
        let per = |acc: u64| acc as f64 / n as f64;
        eprintln!(
            "ferrox: metal gpu[{tag}] {:.3} ms avg over {n} (last {:.3} ms); \
             encode/submission: {:.1} dispatches, {:.1} barriers, \
             {:.1} hazard checks ({:.2} bar/op)",
            (slot.acc_ns as f64 / n as f64) / 1e6,
            dt_ns as f64 / 1e6,
            per(slot.acc_dispatches),
            per(slot.acc_barriers),
            per(slot.acc_begin_ops),
            slot.acc_barriers as f64 / slot.acc_begin_ops.max(1) as f64,
        );
    }
}
