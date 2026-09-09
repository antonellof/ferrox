//! The device side of `mul_mm`: NVRTC compile, upload, launch, download.
//!
//! # UNRUN ON HARDWARE
//!
//! Nothing in this module has executed on a GPU. It is written against
//! `cudarc` 0.11.9's API the same way `gpu.rs`'s matvec launchers are,
//! and it reuses their proven plumbing verbatim -- the process-wide
//! device (`shared_device`), the load-once NVRTC cache
//! (`ensure_module_loaded_lazy`) and the pointer-keyed resident weight
//! cache (`resident_cuda_weights`) -- rather than growing a second copy
//! of any of it. What is new here is one launch configuration and one
//! output allocation.
//!
//! The arithmetic this launches is checked on the host by
//! [`crate::mul_mm_ref`]. The launch itself is checked by
//! [`tests::launch_mul_mm_matches_the_scalar_twin`], which is
//! `#[ignore]`d because it needs a device. Do not un-ignore it here; run
//! it on real hardware and write down what happened.

use crate::gpu::{ensure_module_loaded_lazy, resident_cuda_weights, shared_device, CudaError};
use crate::mul_mm::{grid_dims, kernel_src, validate_shape, MulMmKind, THREADS};

/// Batched quantized GEMM: `dst[token][row] = sum_k W[row][k] * x[token][k]`.
///
/// * `weights` -- `n_rows` quantized rows of `row_bytes` each, in
///   `kind`'s format, exactly as the GGUF mmap holds them. Cached on the
///   device by host pointer, so a repeated call for the same tensor does
///   not re-upload it.
/// * `x_batch` -- `batch` activation rows of `n_cols` f32, row-major,
///   the layout `WeightMatrix::apply_batch` already has.
/// * returns `batch * n_rows` f32 as `out[token * n_rows + row]`, again
///   the layout `apply_batch` already returns.
///
/// Returns [`CudaError::Unsupported`] for a shape this kernel does not
/// implement, so the caller can fall back and disclose it rather than
/// computing something else.
pub fn launch_mul_mm(
    kind: &MulMmKind,
    weights: &[u8],
    x_batch: &[f32],
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, CudaError> {
    use cudarc::driver::LaunchAsync;

    validate_shape(
        kind,
        weights.len(),
        x_batch.len(),
        n_rows,
        n_cols,
        batch,
        row_bytes,
    )
    .map_err(|e| CudaError::Unsupported(e.to_string()))?;

    let dev = shared_device()?;
    ensure_module_loaded_lazy(&dev, kind.module_name, kind.fn_name, || kernel_src(kind))?;
    let func = dev
        .get_func(kind.module_name, kind.fn_name)
        .ok_or_else(|| {
            CudaError::KernelCompile(format!(
                "function '{}' not found after load_ptx",
                kind.fn_name
            ))
        })?;

    // `_weights` must outlive the launch: the kernel reads that buffer
    // asynchronously and only the DtoH below synchronizes.
    let d_weights = resident_cuda_weights(&dev, weights)?;
    let d_x = dev
        .htod_copy(x_batch[..batch * n_cols].to_vec())
        .map_err(|e| CudaError::Launch(format!("mul_mm activation upload: {e:?}")))?;
    let mut d_out = dev
        .alloc_zeros::<f32>(batch * n_rows)
        .map_err(|e| CudaError::Launch(format!("mul_mm output alloc: {e:?}")))?;

    let (grid_x, grid_y) = grid_dims(n_rows, batch);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_x as u32, grid_y as u32, 1),
        block_dim: (THREADS as u32, 1, 1),
        // The tiles are declared `__shared__` inside the kernel, so the
        // dynamic shared-memory request is zero. Asking for more here
        // would be added to the static allocation, not replace it.
        shared_mem_bytes: 0,
    };

    // SAFETY: `func` was compiled from `kernel_src(kind)`, whose
    // parameter list is (const uchar*, const float*, float*, int, int,
    // int, int) and is matched positionally by the tuple below. Each
    // buffer is at least the size the kernel indexes: `validate_shape`
    // has established `weights.len() >= n_rows * row_bytes` and
    // `x_batch.len() >= batch * n_cols`, `d_out` is allocated at exactly
    // `batch * n_rows`, and the kernel bounds-checks every store against
    // `n_rows`/`batch`. The grid covers `ceil(batch/BN) x
    // ceil(n_rows/BM)` tiles, so no thread addresses a row beyond
    // `n_rows - 1` (out-of-range rows are clamped inside the kernel).
    // `d_weights` is held alive across the launch and the DtoH.
    unsafe {
        func.launch(
            cfg,
            (
                &d_weights.slice,
                &d_x,
                &mut d_out,
                n_rows as i32,
                n_cols as i32,
                batch as i32,
                row_bytes as i32,
            ),
        )
        .map_err(|e| CudaError::Launch(format!("kernel {}: {e:?}", kind.fn_name)))?;
    }

    let out = dev
        .dtoh_sync_copy(&d_out)
        .map_err(|e| CudaError::Launch(format!("mul_mm output download: {e:?}")))?;
    drop(d_weights);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mul_mm::{BM, Q8_0};
    use crate::mul_mm_ref::mul_mm_reference;

    #[test]
    fn a_shape_the_kernel_cannot_do_is_refused_without_touching_the_device() {
        // n_cols is not a whole K-tile. This must come back as a named
        // refusal, not a launch attempt -- on a host with no CUDA
        // library at all, reaching `shared_device()` would be a
        // different error entirely.
        let err = launch_mul_mm(&Q8_0, &[0; 68], &[0.0; 48], 2, 48, 1, 34).unwrap_err();
        match err {
            CudaError::Unsupported(msg) => {
                assert!(msg.contains("K-tile"), "unhelpful refusal: {msg}");
            }
            other => panic!("expected a named refusal, got {other:?}"),
        }
    }

    /// The one test that would close the gap this module leaves open.
    ///
    /// It compares the kernel against [`mul_mm_reference`] -- the same
    /// scalar twin the host-side tests already hold to `ferrox_quant`
    /// -- on an exact-tile shape, a partial tile on both axes, and a
    /// narrow batch, for EVERY kind in [`KINDS`].
    ///
    /// It used to name three kinds and pick their fixtures out of an
    /// index-keyed `match`, so the kinds it did not name (the K-quants
    /// then, the codebook formats now) were covered only by
    /// `tools/mul_mm_host_check/run.sh` -- which executes the emitted C
    /// on a host CPU and therefore cannot see anything NVRTC or a warp
    /// scheduler would. The loop is over the table now, and the
    /// fixtures come from [`crate::mul_mm_ref::fixtures`], so a kind
    /// added to `KINDS` is on this list the moment it exists.
    ///
    /// Run it on a machine with a real device:
    ///   cargo test -p ferrox-cuda --features cuda -- --ignored
    #[test]
    #[ignore = "requires real CUDA hardware -- NEVER RUN: this kernel has never executed on a GPU. Run with --ignored on a CUDA-capable machine and record the result before any doc claims CUDA mul_mm works"]
    fn launch_mul_mm_matches_the_scalar_twin() {
        use crate::mul_mm::KINDS;

        for kind in KINDS {
            for (n_rows, cols, batch) in [(BM * 2, 128usize, 32), (BM + 7, 96, 37), (33, 64, 3)] {
                // `validate_shape` refuses a column count that is not a
                // whole super-block, so round up per kind rather than
                // reusing one list of literals across formats whose
                // block sizes differ by 8x.
                let n_cols = cols.next_multiple_of(kind.block_elems);
                let row_bytes = (n_cols / kind.block_elems) * kind.block_bytes;
                let weights = crate::mul_mm_ref::fixtures::weights(kind, n_rows, n_cols, 4242);
                let x: Vec<f32> = (0..batch * n_cols)
                    .map(|i| ((i as f32) * 0.019).cos())
                    .collect();

                let want = mul_mm_reference(kind, &weights, &x, n_rows, n_cols, batch, row_bytes)
                    .expect("the twin must accept every shape the kernel accepts");
                let got = launch_mul_mm(kind, &weights, &x, n_rows, n_cols, batch, row_bytes)
                    .expect("kernel launch must succeed on real CUDA hardware");

                assert_eq!(got.len(), want.len());
                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    let where_ = format!("{} {n_rows}x{n_cols}x{batch} element {i}", kind.name);
                    // NaN==NaN is agreement, not failure: a fixture byte
                    // pattern that decodes as a NaN f16 scale produces a
                    // NaN identically on both sides, and IEEE754 makes
                    // every comparison against it false. This is the
                    // rule a real RTX 3060 run forced on `gpu.rs`'s
                    // `assert_close_relative`; inheriting it here rather
                    // than rediscovering it on rented hardware.
                    if w.is_nan() {
                        assert!(g.is_nan(), "{where_}: twin is NaN but GPU={g} is not");
                        continue;
                    }
                    // Relative, and deliberately not exact. The host
                    // execution check (`tools/mul_mm_host_check`) IS
                    // bit-exact against this same twin, but only because
                    // it disables FP contraction; a real GPU contracts
                    // `acc += a * b` into an FMA, so the accumulator
                    // legitimately drifts from the twin's over `n_cols`
                    // steps. A failure here at 1e-4 is a real bug, not
                    // rounding.
                    let scale = w.abs().max(1.0);
                    assert!((g - w).abs() <= 1e-4 * scale, "{where_}: GPU={g} twin={w}");
                }
            }
        }
    }
}
