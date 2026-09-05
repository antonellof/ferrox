//! One decode step's activations, kept on the device between kernels.
//!
//! # The problem this exists for
//!
//! `launch_matvec` returns `Vec<f32>`, so it ends in `dtoh_sync_copy`,
//! so every matmul on the CUDA path uploads its activation, allocates
//! an output, launches, **synchronises**, and downloads. llama.cpp's
//! `ggml_backend_cuda_graph_compute` enqueues an entire token's graph on
//! one stream with zero `cudaStreamSynchronize` in its node loop and
//! synchronises once, at the buffer-API boundary where the caller reads
//! logits. ferrox pays roughly one sync per matmul instead. That ratio
//! -- not any kernel -- is what #133 measured as 36% GPU utilization.
//!
//! An [`ActChain`] is the seam that lets a sequence of kernels run
//! without the host in between: one [`ActChain::upload`], a chain of
//! device-resident operations, one [`ActChain::download`].
//!
//! # What it can and cannot reach today
//!
//! Counted on the dense CUDA decode path, one decoder layer performs
//! **five** device-to-host synchronisations:
//!
//! | # | site | why it syncs |
//! |---|---|---|
//! | 1-3 | `apply_gpu_multi` for Q, K, V | one `dtoh_sync_copy` per output |
//! | 4 | `o_proj` via `launch_matvec` | returns `Vec<f32>` |
//! | 5 | `launch_dense_ffn_swiglu` | returns `Vec<f32>` |
//!
//! plus one for `lm_head` per token. Chaining collapses 1-3 into a
//! single sync ([`ActChain::download_all`]), which is what this module
//! delivers. It **cannot** collapse 4 or 5, and that is a property of
//! the decoder rather than of this seam: between the QKV projection and
//! `o_proj` sit the QKV biases, the QK norms, RoPE and the attention
//! reduction, all on the host; between `o_proj` and the FFN sit
//! `post_attn_norm`, the residual add and the FFN norm, also on the
//! host. A device-resident activation that is immediately consumed by
//! host arithmetic has to come back.
//!
//! So the honest ceiling of chaining alone is 5 syncs per layer down to
//! 3. Reaching llama.cpp's one-per-token needs the residual stream
//! itself to be device-resident for the whole token -- every
//! intermediate allocated in a device buffer, the way
//! `ggml_backend_cuda_buffer_type` does it -- which is a decoder change,
//! not a backend change. `docs/plans/cpu-cuda-parity.md` step 2 is where
//! that belongs.
//!
//! # Identity
//!
//! See [`crate::chain_id`]. Residency here is never inferred from a
//! length, a pointer or a thread-local: a [`DeviceAct`] carries the
//! [`ChainId`] of the chain that produced it, and the device pointer is
//! structurally unreachable except through a check against that id.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};

use crate::chain_id::{check_same_chain, ChainId};
use crate::gpu::{
    enqueue_matvec, fused_add_rmsnorm_device, shared_device, silu_mul_device, CudaError,
    MatvecLaunch, ResidentCudaWeights,
};

/// `DeviceAct` lives in its own module so that its `slice` field is
/// private to it and not merely private to this file.
///
/// That is the difference between an invariant and a convention. Rust's
/// privacy is module-scoped, so a `DeviceAct` declared next to
/// [`ActChain`] would let every one of `ActChain`'s methods reach
/// `act.slice` directly and skip the identity check -- and "N call
/// sites that must all remember to check one thing" is the exact defect
/// shape `CLAUDE.md` says this repo keeps paying for. Behind this
/// module boundary there is precisely one accessor,
/// [`DeviceAct::slice_for`], it demands a [`ChainId`], and a chain that
/// forgets to check cannot compile.
mod guarded {
    use super::{check_same_chain, ChainId, CudaSlice};
    use crate::chain_id::ForeignActivation;
    use cudarc::driver::DeviceSlice;

    /// A device-resident activation vector: a `CudaSlice<f32>`, its
    /// logical length, and the identity of the chain that produced it.
    ///
    /// Not `Clone` and not `Copy`: it owns its device allocation, and
    /// two handles to one buffer is the aliasing this design refuses.
    pub struct DeviceAct {
        slice: CudaSlice<f32>,
        len: usize,
        chain: ChainId,
    }

    impl DeviceAct {
        /// Stamps a freshly produced device buffer with its chain. Only
        /// [`super::ActChain`] can call this, so every activation in
        /// existence was stamped by the chain that produced it.
        ///
        /// `len` is DERIVED from the allocation rather than passed in.
        /// A caller-supplied length would be a second structure that has
        /// to agree with the buffer -- `matvec` would pass
        /// `launch.rows`, `upload` would pass `x.len()`, and the day one
        /// of those stopped matching what was allocated, an unchecked
        /// `memcpy` of that length would read past the buffer. There is
        /// one length here, and it is the buffer's.
        pub(super) fn stamped(slice: CudaSlice<f32>, chain: ChainId) -> Self {
            let len = slice.len();
            DeviceAct { slice, len, chain }
        }

        /// The ONLY route from a `DeviceAct` to its device pointer, and
        /// it is a refusal point. See this module's doc comment for why
        /// it is here rather than beside `ActChain`.
        pub(super) fn slice_for(
            &self,
            chain: ChainId,
        ) -> Result<&CudaSlice<f32>, ForeignActivation> {
            check_same_chain(chain, self.chain)?;
            Ok(&self.slice)
        }

        /// Logical element count.
        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// Which chain produced this activation. Diagnostics only --
        /// the enforcement is [`DeviceAct::slice_for`], which cannot be
        /// bypassed by reading this.
        pub fn chain(&self) -> ChainId {
            self.chain
        }
    }

    /// Hand-written rather than derived: the device pointer is not
    /// something a log line should print, and the two things worth
    /// seeing when a refusal fires are the length and the chain.
    impl std::fmt::Debug for DeviceAct {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("DeviceAct")
                .field("len", &self.len)
                .field("chain", &self.chain)
                .finish()
        }
    }
}

pub use guarded::DeviceAct;

/// One device-resident activation chain: a run of kernels that hand
/// their outputs to each other without the host seeing an intermediate.
///
/// A chain is opened per logical step (one fused FFN, one QKV
/// projection), used, and dropped. It is deliberately not `Clone` and
/// not `Send`: a chain is a local sequence, and the moment one could be
/// stashed somewhere and picked up later is the moment its activations
/// start outliving the step that produced them.
pub struct ActChain {
    id: ChainId,
    dev: Arc<CudaDevice>,
    /// Resident-weight handles for every kernel this chain enqueued.
    ///
    /// A kernel reads its weight buffer asynchronously, so the `Arc`
    /// must outlive the sync at the end of the chain, not the call that
    /// launched it. `resident_cuda_weights` never evicts today, so this
    /// is belt and braces -- but the day it does evict, dropping the
    /// handle at the end of `matvec` would free a buffer a queued
    /// kernel is still reading, and the symptom would be wrong numbers
    /// rather than a crash.
    held_weights: RefCell<Vec<Arc<ResidentCudaWeights>>>,
}

impl ActChain {
    /// Opens a chain on the process-wide shared device.
    ///
    /// Every activation this chain produces is stamped with a
    /// process-unique [`ChainId`], and this chain refuses every
    /// activation it did not stamp.
    pub fn open() -> Result<ActChain, CudaError> {
        Ok(ActChain {
            id: ChainId::mint(),
            dev: shared_device()?,
            held_weights: RefCell::new(Vec::new()),
        })
    }

    /// This chain's identity. Diagnostics and tests.
    pub fn id(&self) -> ChainId {
        self.id
    }

    /// Uploads a host activation once (one HtoD, no sync).
    pub fn upload(&self, x: &[f32]) -> Result<DeviceAct, CudaError> {
        let slice = self
            .dev
            .htod_copy(x.to_vec())
            .map_err(|e| CudaError::Launch(format!("act upload: {e:?}")))?;
        Ok(DeviceAct::stamped(slice, self.id))
    }

    /// Enqueues one matvec whose activation is already on the device,
    /// producing another device-resident activation. No HtoD of `x`, no
    /// DtoH of the result, no host sync.
    pub fn matvec(&self, launch: &MatvecLaunch<'_>, x: &DeviceAct) -> Result<DeviceAct, CudaError> {
        let d_x = x.slice_for(self.id)?;
        let (d_out, weights) = enqueue_matvec(&self.dev, launch, d_x)?;
        self.held_weights.borrow_mut().push(weights);
        Ok(DeviceAct::stamped(d_out, self.id))
    }

    /// Enqueues the elementwise `silu(gate) * up` fuse. Both inputs must
    /// be this chain's and the same length.
    pub fn silu_mul(&self, gate: &DeviceAct, up: &DeviceAct) -> Result<DeviceAct, CudaError> {
        let n = same_len("silu_mul", gate.len(), up.len())?;
        let d_gate = gate.slice_for(self.id)?;
        let d_up = up.slice_for(self.id)?;
        let d_out = silu_mul_device(&self.dev, d_gate, d_up, n)?;
        Ok(DeviceAct::stamped(d_out, self.id))
    }

    /// Enqueues `rms_norm(x + residual, weight, eps)` as one kernel.
    pub fn add_rms_norm(
        &self,
        x: &DeviceAct,
        residual: &DeviceAct,
        weight: &DeviceAct,
        eps: f32,
    ) -> Result<DeviceAct, CudaError> {
        let n = same_len("add_rms_norm", x.len(), residual.len())?;
        let n = same_len("add_rms_norm", n, weight.len())?;
        let d_x = x.slice_for(self.id)?;
        let d_residual = residual.slice_for(self.id)?;
        let d_weight = weight.slice_for(self.id)?;
        let d_out = fused_add_rmsnorm_device(&self.dev, d_x, d_residual, d_weight, n, eps)?;
        Ok(DeviceAct::stamped(d_out, self.id))
    }

    /// Reads one activation back. One DtoH and one stream sync.
    ///
    /// Deliberately built on `cudarc`'s safe `dtoh_sync_copy` rather
    /// than on [`ActChain::download_all`]: it is the scalar twin the
    /// raw-FFI batched path is checked against, and a hardware test
    /// asserts the two agree element for element.
    pub fn download(&self, act: &DeviceAct) -> Result<Vec<f32>, CudaError> {
        let slice = act.slice_for(self.id)?;
        self.dev
            .dtoh_sync_copy(slice)
            .map_err(|e| CudaError::Launch(format!("act download: {e:?}")))
    }

    /// Reads several activations back with **one** stream sync for all
    /// of them, in the order given.
    ///
    /// `cudarc` 0.11.9's `dtoh_sync_copy` calls `synchronize()` at the
    /// end of every copy, so downloading N outputs through it costs N
    /// `cuStreamSynchronize` calls. Q/K/V is that N=3, once per layer,
    /// on every layer of every token. This enqueues the N copies back to
    /// back on the device's own stream and synchronises once, which is
    /// the discipline `ggml-cuda` applies to a whole graph.
    ///
    /// **What that is and is not worth.** The destination `Vec`s are
    /// ordinary pageable host memory, and CUDA's own contract is that a
    /// device-to-pageable-host `cuMemcpyDtoHAsync` returns only once the
    /// copy has completed. So this does not overlap the copies; what it
    /// removes is N-1 redundant `cuStreamSynchronize` driver round trips
    /// per call. Pinned staging buffers would be needed to overlap them,
    /// and `cudarc` 0.11.9 exposes no pinned allocator. Claim the driver
    /// calls, not the bandwidth.
    pub fn download_all(&self, acts: &[DeviceAct]) -> Result<Vec<Vec<f32>>, CudaError> {
        if acts.is_empty() {
            return Ok(Vec::new());
        }
        // Check EVERY activation before copying any of them: a chain
        // that refuses must not have already written into half the
        // caller's buffers.
        let slices = acts
            .iter()
            .map(|a| a.slice_for(self.id))
            .collect::<Result<Vec<_>, _>>()?;

        self.dev
            .bind_to_thread()
            .map_err(|e| CudaError::Launch(format!("bind for batched download: {e:?}")))?;

        let mut outs: Vec<Vec<f32>> = acts.iter().map(|a| vec![0.0f32; a.len()]).collect();
        for (slice, dst) in slices.iter().zip(outs.iter_mut()) {
            // SAFETY:
            // 1. `T` is `f32`, the type every `CudaSlice<f32>` in this
            //    module was allocated with (`upload`, `enqueue_matvec`,
            //    `silu_mul_device`, `fused_add_rmsnorm_device` all
            //    allocate `f32`).
            // 2. The allocation is live: `slice` borrows a `DeviceAct`
            //    borrowed from `acts` for the whole of this call, and
            //    `CudaSlice` frees only on drop.
            // 3. `dst` is exactly `slice.len()` elements long. `outs`
            //    is built from the same `acts` in the same order, and
            //    `DeviceAct::len` is not a second, restated length: it
            //    is READ OFF the allocation in `DeviceAct::stamped`, so
            //    it cannot drift from it.
            // 4. The stream is this device's own stream, the one the
            //    buffers were allocated on and every kernel above was
            //    enqueued on, so the copies are ordered after the work
            //    that produced them.
            // 5. `dst` is not read before the `synchronize()` below --
            //    nothing between here and there touches `outs`, and
            //    `outs` is not returned until after it.
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async(
                    dst.as_mut_slice(),
                    *slice.device_ptr(),
                    *self.dev.cu_stream(),
                )
            }
            .map_err(|e| CudaError::Launch(format!("batched act download: {e:?}")))?;
        }

        self.dev
            .synchronize()
            .map_err(|e| CudaError::Launch(format!("batched download sync: {e:?}")))?;
        Ok(outs)
    }
}

/// Two activations an elementwise kernel requires to be the same
/// length. Returns the length, or names the mismatch.
///
/// A `Result` rather than an `assert_eq!` because a shape this backend
/// cannot run must STOP with the reason named, and `CudaError`'s caller
/// already knows how to fall back to a path that can compute it.
fn same_len(op: &'static str, a: usize, b: usize) -> Result<usize, CudaError> {
    if a == b {
        Ok(a)
    } else {
        Err(CudaError::Unsupported(format!(
            "{op}: activations must be the same length, got {a} and {b}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs without a GPU: `same_len` is the shape guard the elementwise
    /// chain operations refuse on, and it must REFUSE rather than
    /// silently run the kernel over the shorter of the two, which is
    /// what an `assert_eq!` in release with `-C debug-assertions=off`
    /// would... not do, but which a `min(a, b)` would.
    #[test]
    fn same_len_accepts_a_match_and_names_a_mismatch() {
        assert_eq!(
            same_len("silu_mul", 4096, 4096).expect("equal lengths"),
            4096
        );

        let err = same_len("silu_mul", 4096, 2048)
            .expect_err("a length mismatch must be refused, not silently truncated");
        assert!(
            matches!(err, CudaError::Unsupported(_)),
            "a shape this backend cannot run must be Unsupported, got {err}"
        );
        let rendered = err.to_string();
        for needle in ["silu_mul", "4096", "2048"] {
            assert!(
                rendered.contains(needle),
                "refusal must name {needle}: {rendered}"
            );
        }
    }

    /// Zero is not a special case that gets waved through: two empty
    /// activations agree, an empty and a non-empty one do not.
    #[test]
    fn same_len_does_not_wave_through_zero() {
        assert_eq!(same_len("add_rms_norm", 0, 0).expect("both empty"), 0);
        assert!(same_len("add_rms_norm", 0, 8).is_err());
        assert!(same_len("add_rms_norm", 8, 0).is_err());
    }
}
