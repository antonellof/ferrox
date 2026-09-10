//! The one hand-off that lets a decode step skip re-uploading its own
//! activation, and the identity that decides whether the hand-off is
//! valid.
//!
//! # What the hand-off is
//!
//! When [`crate::decode_dense::launch_decode_dense_stack`] runs
//! `final_norm` but no `lm_head` (the caller wants hidden state, not
//! logits), the normalized hidden already sits in the decode scratch's
//! `x` buffer on the GPU. The stack downloads it and returns it, and the
//! caller's very next act is to hand that same vector straight back to
//! a matvec. Uploading it again would allocate a fresh `MTLBuffer` and
//! memcpy `hidden_dim` floats for data the device already has, so the
//! stack PUBLISHES the fact instead and the next matvec REUSES the
//! buffer.
//!
//! # Why it needed rewriting (GitHub issue #166)
//!
//! The published fact used to be a thread-local raw
//! `*const MTLBuffer` plus a LENGTH, matched by comparing that length
//! against the incoming activation's. Both halves of that were wrong:
//!
//! - **The pointer aimed into shared, mutable state with no lock.**
//!   `DECODE_SCRATCH` is a process-wide `Mutex`, and the publication
//!   escaped it: the pointer outlived the guard. Two concurrent decodes
//!   in one process (`ferrox-server` serves each request on its own
//!   `spawn_blocking` thread over one `Arc<Decoder>`) meant thread A
//!   could publish `&scratch.x`, thread B's dense stack could overwrite
//!   `scratch.x`, and thread A's `lm_head` would then read B's
//!   activation. Same model, so the lengths always agreed and nothing
//!   would have noticed.
//! - **Length is not an identity.** Every consumer of a published
//!   activation ([`crate::gpu::launch_matvec_fused`],
//!   `launch_dense_ffn_swiglu`, `launch_moe_topk_swiglu`) is routinely
//!   handed a `hidden_dim`-long activation that is NOT the published
//!   one. Nothing but call-order discipline in another crate stood
//!   between that and a silently wrong answer.
//!
//! # The identity now
//!
//! A publication records the HOST ADDRESS and length of the exact
//! `Vec<f32>` the dense stack returned, and it lives INSIDE
//! `DecodeScratch`, under the mutex that owns the buffer it describes.
//! That gives three properties the length comparison did not:
//!
//! 1. Only the exact slice the stack returned can match. A different
//!    activation of the same length has a different address and misses.
//! 2. [`crate::attn::borrow_decode_scratch`] drops any publication when
//!    it hands out the guard, so anything that may WRITE `scratch.x`
//!    invalidates the claim that `scratch.x` holds a known activation.
//! 3. A reuse holds the scratch guard for as long as it uses the buffer,
//!    so no other thread can be writing `scratch.x` underneath it.
//!
//! The lock is taken with `try_lock`: a busy scratch is a miss and an
//! ordinary upload, never a wait and never a deadlock. That also makes
//! the optimisation self-disabling under concurrency, which is the
//! correct direction to fail in.
//!
//! What this does NOT promise: a publication whose `Vec` is freed and
//! whose address is then reused by another same-length allocation
//! before the next scratch borrow would still match. That window is
//! within one decode step, and closing it would cost a content compare
//! as expensive as the upload being avoided. It is stated rather than
//! hidden.

use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::attn::DecodeScratch;
use crate::gpu::MetalError;

/// A claim that the decode scratch's `x` buffer holds the activation
/// whose host copy is `len` floats at `host_addr`.
///
/// Stored in [`DecodeScratch`] rather than beside it, so the claim and
/// the buffer it describes are behind one lock and cannot drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResidentPublication {
    host_addr: usize,
    len: usize,
}

impl ResidentPublication {
    /// The publication a dense stack makes for the vector it is about
    /// to return.
    pub(crate) fn of(host: &[f32]) -> Self {
        Self {
            host_addr: host.as_ptr() as usize,
            len: host.len(),
        }
    }

    /// Whether `x` is the very slice this publication was made for.
    ///
    /// The whole point of the rewrite: a caller cannot get this right by
    /// accident the way it could get a length right by accident.
    pub(crate) fn describes(&self, x: &[f32]) -> bool {
        self.host_addr == x.as_ptr() as usize && self.len == x.len()
    }
}

/// How many times a matvec reused a published activation instead of
/// uploading it.
///
/// Read by the hardware tests: an optimisation whose fast path never
/// fires reads as coverage while doing nothing, which is the same
/// defect shape as a gate that cannot fire.
static REUSES: AtomicU64 = AtomicU64::new(0);

/// Reuses of a resident activation since process start.
pub fn resident_activation_reuses() -> u64 {
    REUSES.load(Ordering::Relaxed)
}

/// Publishes `host` as the contents of the decode scratch's `x` buffer.
///
/// `scratch` is the caller's live guard, which is what makes this safe:
/// the publication is written under the same lock that protects the
/// buffer, by the code that just filled it.
pub(crate) fn publish(scratch: &mut DecodeScratch, host: &[f32]) {
    scratch.resident = Some(ResidentPublication::of(host));
}

/// Drops any published activation.
///
/// Best effort: a busy scratch is left alone, because a publication is
/// already invalidated by the next borrow of the scratch and can only
/// be consumed by the exact slice it was made for. Blocking here would
/// mean waiting on another thread's whole decode step.
///
/// `ferrox-models` calls this at the boundaries of a forward pass, where
/// it is belt-and-braces rather than the thing correctness rests on.
pub fn clear_resident_activation() {
    if let Some(mut guard) = crate::attn::try_lock_decode_scratch() {
        if let Some(scratch) = guard.as_mut() {
            scratch.resident = None;
        }
    }
}

/// The activation buffer a matvec binds, however it was obtained.
///
/// Derefs to the buffer, so call sites read exactly as they did when
/// this was a bare `Retained`. When the buffer is the decode scratch's
/// own `x`, the scratch guard rides along and is released only when the
/// matvec is done with it.
pub(crate) struct ActivationBuffer {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Held to keep another thread's dense stack out of `scratch.x`
    /// while this buffer is bound. `None` for a freshly uploaded copy,
    /// which nothing else can reach.
    _scratch: Option<std::sync::MutexGuard<'static, Option<DecodeScratch>>>,
}

impl Deref for ActivationBuffer {
    type Target = ProtocolObject<dyn MTLBuffer>;

    fn deref(&self) -> &Self::Target {
        &self.buf
    }
}

/// The single way a matvec gets its activation onto the GPU: reuse the
/// published buffer when `x` IS the published activation, upload
/// otherwise.
///
/// This was three copies of the same fourteen lines in `gpu.rs`, one per
/// consumer, which is the shape that loses a fix in two of three places.
pub(crate) fn upload_or_reuse(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    x: &[f32],
) -> Result<ActivationBuffer, MetalError> {
    if let Some(mut guard) = crate::attn::try_lock_decode_scratch() {
        let hit = guard
            .as_ref()
            .and_then(|s| s.resident)
            .is_some_and(|p| p.describes(x));
        if hit {
            let buf = {
                let scratch = guard.as_mut().expect("hit implies a live scratch");
                // Consumed: a publication answers exactly one matvec.
                scratch.resident = None;
                scratch.x.clone()
            };
            REUSES.fetch_add(1, Ordering::Relaxed);
            return Ok(ActivationBuffer {
                buf,
                _scratch: Some(guard),
            });
        }
    }
    let mut x_owned = x.to_vec();
    // SAFETY: `x_owned` is a live, non-empty-capacity `Vec<f32>` of
    // exactly `x_owned.len() * 4` bytes, and `newBufferWithBytes` COPIES
    // those bytes into the new buffer, so the allocation may be dropped
    // at the end of this call.
    let buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(x_owned.as_mut_ptr() as *mut _).ok_or(MetalError::BufferAllocFailed)?,
            std::mem::size_of_val(x_owned.as_slice()),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAllocFailed)?;
    Ok(ActivationBuffer {
        buf,
        _scratch: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bug this module was rewritten for: the old publication
    /// matched on LENGTH, so any same-length activation consumed it.
    ///
    /// Sabotage: make `describes` compare `self.len == x.len()` only,
    /// which is verbatim what `take_resident_activation_if_matches`
    /// did before issue #166.
    #[test]
    fn a_same_length_activation_is_not_the_published_one() {
        let published = vec![1.0f32, 2.0, 3.0, 4.0];
        let other = vec![9.0f32, 9.0, 9.0, 9.0];
        assert_eq!(published.len(), other.len(), "the trap needs equal lengths");
        let p = ResidentPublication::of(&published);
        assert!(p.describes(&published), "the published slice must match");
        assert!(
            !p.describes(&other),
            "a different activation of the same length must NOT be \
             mistaken for the published one"
        );
    }

    /// A sub-slice starting at the same address is a different vector
    /// and must miss too: the buffer holds `len` floats and binding it
    /// for a shorter matvec would feed the kernel the wrong columns.
    #[test]
    fn a_prefix_of_the_published_activation_is_not_the_published_one() {
        let published = vec![1.0f32, 2.0, 3.0, 4.0];
        let p = ResidentPublication::of(&published);
        assert!(!p.describes(&published[..2]));
    }

    const COLS: usize = 64;
    const ROWS: usize = 8;

    /// The decode scratch, the publication in it and the reuse counter
    /// are all process-wide, and `cargo test` runs tests in parallel.
    /// Two of these at once read each other's publications and each
    /// other's counter deltas, which is a flake, not a finding.
    static SERIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
        SERIALIZE.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `rows x COLS` of f32 weights, row `r` scaled by `r + 1`, so a
    /// wrong activation cannot accidentally give the right answer.
    fn f32_weights() -> Vec<u8> {
        let mut w = Vec::with_capacity(ROWS * COLS * 4);
        for r in 0..ROWS {
            for c in 0..COLS {
                let v = (r + 1) as f32 * (1.0 + c as f32 * 0.01);
                w.extend_from_slice(&v.to_le_bytes());
            }
        }
        w
    }

    fn f32_matvec(weights: &[u8], x: &[f32]) -> Vec<f32> {
        let launch = crate::gpu::MatvecLaunch {
            kernel_src: crate::gpu::F32_MATVEC_KERNEL_SRC,
            fn_name: "f32_matvec",
            block_bytes: 4,
            block_elems: 1,
            weights,
            rows: ROWS,
            row_bytes: COLS * 4,
            rows_per_tg: 1,
        };
        crate::gpu::launch_matvec_fused(x, std::slice::from_ref(&launch))
            .expect("matvec launch")
            .pop()
            .expect("one launch, one output")
    }

    /// Puts `act` into the decode scratch's `x` buffer and publishes it,
    /// exactly as `launch_decode_dense_stack` does when `final_norm` ran
    /// without `lm_head`.
    fn publish_into_scratch(act: &[f32]) {
        let shared = crate::gpu::shared_metal().expect("Metal device");
        let mut guard = crate::attn::borrow_decode_scratch(
            &shared.device,
            crate::attn::ScratchCaps {
                hidden: COLS,
                max_q: COLS,
                max_kv: COLS,
                attn: COLS,
                max_gate: COLS,
                logits: 0,
            },
        )
        .expect("scratch");
        let scratch = guard.as_mut().expect("scratch just ensured");
        crate::attn::copy_f32_into(&scratch.x, act);
        publish(scratch, act);
    }

    /// The reuse must be a pure optimisation: the same activation gives
    /// the same bits whether it was uploaded or found already resident.
    ///
    /// The reuse COUNTER is asserted as well, because a fast path that
    /// silently stopped firing would leave this test green while proving
    /// nothing -- the same shape as a refusal whose condition is
    /// unreachable.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn reusing_a_published_activation_is_bit_identical_to_uploading_it() {
        let _serialized = one_at_a_time();
        let weights = f32_weights();
        let act: Vec<f32> = (0..COLS).map(|i| (i as f32 * 0.37).sin()).collect();

        clear_resident_activation();
        let uploaded = f32_matvec(&weights, &act);

        publish_into_scratch(&act);
        let before = resident_activation_reuses();
        let reused = f32_matvec(&weights, &act);
        assert_eq!(
            resident_activation_reuses(),
            before + 1,
            "the resident fast path never fired, so this test proves nothing"
        );

        let a: Vec<u32> = uploaded.iter().map(|v| v.to_bits()).collect();
        let b: Vec<u32> = reused.iter().map(|v| v.to_bits()).collect();
        assert_eq!(a, b, "reuse must not change a single bit of the result");
    }

    /// GitHub issue #166's aliasing half: the old publication matched on
    /// LENGTH, and every consumer of one is routinely handed a
    /// `hidden_dim`-long activation that is not the published one.
    ///
    /// `published` is all zeros and `other` is all ones, so consuming
    /// the wrong buffer is not a rounding difference: it is an all-zero
    /// answer where the row sums belong.
    ///
    /// Sabotage: make `ResidentPublication::describes` compare
    /// `self.len == x.len()` only, which is verbatim the rule
    /// `take_resident_activation_if_matches` used before this change.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn a_same_length_activation_never_consumes_a_published_one_on_the_gpu() {
        let _serialized = one_at_a_time();
        let weights = f32_weights();
        let published = vec![0.0f32; COLS];
        let other = vec![1.0f32; COLS];

        clear_resident_activation();
        let want = f32_matvec(&weights, &other);
        assert!(
            want.iter().any(|v| v.abs() > 1.0),
            "the reference answer must be far from the zero one"
        );

        publish_into_scratch(&published);
        let got = f32_matvec(&weights, &other);

        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "a same-length activation consumed the published one: the \
             matvec answered for the resident vector, not for its own"
        );
    }

    /// Anything that may WRITE `scratch.x` must drop the claim about
    /// what `scratch.x` holds, and taking the guard IS that permission.
    ///
    /// Without this, a publication would outlive the next decode step
    /// that overwrote the buffer, which is precisely how a second
    /// concurrent request used to be able to answer the first one's
    /// `lm_head` with its own activation.
    ///
    /// Sabotage: delete the `scratch.resident = None` that
    /// `crate::attn::borrow_decode_scratch` does before it hands out the
    /// guard.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn borrowing_the_scratch_invalidates_a_publication() {
        let _serialized = one_at_a_time();
        let weights = f32_weights();
        let act: Vec<f32> = (0..COLS).map(|i| (i as f32 * 0.23).sin()).collect();

        publish_into_scratch(&act);
        clear_resident_activation();
        let before = resident_activation_reuses();
        let _ = f32_matvec(&weights, &act);
        assert_eq!(
            resident_activation_reuses(),
            before,
            "an explicitly cleared publication was still consumed"
        );

        // What the NEXT decode step does first: borrow the scratch.
        publish_into_scratch(&act);
        let shared = crate::gpu::shared_metal().expect("Metal device");
        drop(
            crate::attn::borrow_decode_scratch(
                &shared.device,
                crate::attn::ScratchCaps {
                    hidden: COLS,
                    max_q: COLS,
                    max_kv: COLS,
                    attn: COLS,
                    max_gate: COLS,
                    logits: 0,
                },
            )
            .expect("scratch"),
        );
        let before = resident_activation_reuses();
        let _ = f32_matvec(&weights, &act);
        assert_eq!(
            resident_activation_reuses(),
            before,
            "a publication survived a borrow of the buffer it describes"
        );
    }

    /// Issue #166 proper: the same matvec, driven from the calling
    /// thread and from another thread, must give the same bits.
    ///
    /// The publication is process-wide now rather than thread-local, so
    /// the hand-off survives the step moving threads instead of being
    /// silently skipped on one side and taken on the other.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn the_same_matvec_gives_the_same_bits_on_any_thread() {
        let _serialized = one_at_a_time();
        let weights = f32_weights();
        let act: Vec<f32> = (0..COLS).map(|i| (i as f32 * 0.11).cos()).collect();

        clear_resident_activation();
        let here = f32_matvec(&weights, &act);
        let there = std::thread::scope(|s| {
            s.spawn(|| f32_matvec(&weights, &act))
                .join()
                .expect("worker thread")
        });
        assert_eq!(
            here.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            there.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the Metal matvec path is not thread-agnostic"
        );

        // And the hand-off itself crosses the thread boundary: published
        // here, consumed there.
        publish_into_scratch(&act);
        let before = resident_activation_reuses();
        let on_worker = std::thread::scope(|s| {
            s.spawn(|| f32_matvec(&weights, &act))
                .join()
                .expect("worker thread")
        });
        assert_eq!(
            resident_activation_reuses(),
            before + 1,
            "a publication made on one thread must still be visible on \
             another; a thread-local one was not"
        );
        assert_eq!(
            on_worker.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            here.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "consuming the publication on another thread changed the answer"
        );
    }
}
