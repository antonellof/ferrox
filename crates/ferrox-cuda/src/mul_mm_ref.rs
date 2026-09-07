//! The scalar twin of the CUDA `mul_mm` kernel: the same GEMM, executed
//! on the host, one emulated threadblock at a time.
//!
//! # Why this exists
//!
//! There is no NVIDIA GPU in the environment this kernel was written in,
//! so the kernel itself cannot be run. This repo's answer to an
//! unrunnable arm is the same one every `unsafe` SIMD path here already
//! gets: a scalar twin implementing identical arithmetic, checked
//! against an independent reference on data the host can compute.
//!
//! [`mul_mm_reference`] is not "a matmul that should give the same
//! answer". It is a **transcription of the kernel**: same tile
//! enumeration, same thread-to-micro-tile mapping, same out-of-range row
//! clamp, same zero-fill of absent tokens, same k-ascending accumulation
//! order, same shared-memory layout. Barriers are modelled by running
//! every thread's load phase before any thread's compute phase, which is
//! exactly what `__syncthreads()` guarantees. Read it beside `BODY_SRC`
//! in [`crate::mul_mm`]; a line that does not correspond to a line there
//! is a bug in one of them.
//!
//! That buys three things a device would otherwise have to: that the
//! index arithmetic addresses the elements it means to, that the
//! per-kind unpack decodes the format, and that partial tiles neither
//! write out of bounds nor drop rows.
//!
//! It does **not**, by itself, establish that the *emitted CUDA C* says
//! the same thing this file says -- the two are hand-transcribed from
//! each other. `tools/mul_mm_host_check/run.sh` closes that gap by
//! compiling the real kernel text and running it against this twin on
//! the host (bit-exact as of 2026-09-01). What remains for hardware:
//! that NVRTC accepts the source, that the barriers survive a real warp
//! scheduler, that the launch config is valid, and what it costs. Those
//! stay unproven until someone runs
//! `cargo test -p ferrox-cuda --features cuda -- --ignored`.

use crate::mul_mm::{
    grid_dims, validate_shape, MulMmKind, MulMmUnsupported, BK, BM, BN, SUB, THREADS, TM, TN,
};

/// Host emulation of the CUDA `mul_mm` kernel.
///
/// Returns `batch * n_rows` floats laid out as `out[token * n_rows +
/// row]`, the layout `WeightMatrix::apply_batch` already produces.
///
/// This is a correctness reference, not a fast path: it is the kernel's
/// arithmetic, thread by thread, and is roughly as slow as that sounds.
pub fn mul_mm_reference(
    kind: &MulMmKind,
    weights: &[u8],
    x_batch: &[f32],
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, MulMmUnsupported> {
    validate_shape(
        kind,
        weights.len(),
        x_batch.len(),
        n_rows,
        n_cols,
        batch,
        row_bytes,
    )?;

    let mut dst = vec![0f32; batch * n_rows];
    let (grid_x, grid_y) = grid_dims(n_rows, batch);
    for by in 0..grid_y {
        for bx in 0..grid_x {
            emulate_block(
                kind, weights, x_batch, &mut dst, n_rows, n_cols, batch, row_bytes, bx, by,
            );
        }
    }
    Ok(dst)
}

/// One threadblock of the kernel: `blockIdx = (bx, by)`, `THREADS`
/// threads, `sa`/`sb` standing in for the two `__shared__` tiles.
#[allow(clippy::too_many_arguments)] // The kernel's own parameter list plus its block index; bundling it would only move the same values behind a name that says less.
fn emulate_block(
    kind: &MulMmKind,
    src0: &[u8],
    src1: &[f32],
    dst: &mut [f32],
    n_rows: usize,
    n_cols: usize,
    batch: usize,
    row_bytes: usize,
    bx: usize,
    by: usize,
) {
    let nl = kind.nl();
    let r0 = by * BM;
    let r1 = bx * BN;

    // `__shared__ float sa[BK][BM]` / `sb[BK][BN]`.
    let mut sa = vec![0f32; BK * BM];
    let mut sb = vec![0f32; BK * BN];
    // Per-thread `float acc[TN][TM]`, all registers of the block at once.
    let mut acc = vec![0f32; THREADS * TN * TM];

    let mut k0 = 0usize;
    while k0 < n_cols {
        // --- load phase (every thread, then the barrier) ---
        for tid in 0..THREADS {
            if tid < BM * (BK / SUB) {
                let lr = tid / (BK / SUB);
                let ils = tid % (BK / SUB);
                let mut row = r0 + lr;
                if row >= n_rows {
                    row = n_rows - 1;
                }
                let rp = &src0[row * row_bytes..(row + 1) * row_bytes];
                let sub = (k0 / SUB) + ils;
                let xb = &rp[(sub / nl) * kind.block_bytes..];
                let mut reg = [0f32; SUB];
                (kind.dequant_twin)(xb, sub % nl, &mut reg);
                for (i, v) in reg.iter().enumerate() {
                    sa[(SUB * ils + i) * BM + lr] = *v;
                }
            }

            let mut idx = tid;
            while idx < BK * BN {
                let j = idx / BK;
                let kk = idx % BK;
                let col = r1 + j;
                sb[kk * BN + j] = if col < batch {
                    src1[col * n_cols + k0 + kk]
                } else {
                    0.0
                };
                idx += THREADS;
            }
        }

        // --- compute phase (after `__syncthreads()`) ---
        for tid in 0..THREADS {
            let tx = tid % (BM / TM);
            let ty = tid / (BM / TM);
            let acc = &mut acc[tid * TN * TM..(tid + 1) * TN * TM];
            for kk in 0..BK {
                let mut a = [0f32; TM];
                let mut b = [0f32; TN];
                for (m, slot) in a.iter_mut().enumerate() {
                    *slot = sa[kk * BM + tx * TM + m];
                }
                for (n, slot) in b.iter_mut().enumerate() {
                    *slot = sb[kk * BN + ty * TN + n];
                }
                for n in 0..TN {
                    for m in 0..TM {
                        acc[n * TM + m] += a[m] * b[n];
                    }
                }
            }
        }

        k0 += BK;
    }

    // --- store ---
    for tid in 0..THREADS {
        let tx = tid % (BM / TM);
        let ty = tid / (BM / TM);
        let acc = &acc[tid * TN * TM..(tid + 1) * TN * TM];
        for n in 0..TN {
            let col = r1 + ty * TN + n;
            if col >= batch {
                continue;
            }
            for m in 0..TM {
                let row = r0 + tx * TM + m;
                if row < n_rows {
                    dst[col * n_rows + row] = acc[n * TM + m];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mul_mm::{f16_to_f32, kernel_src, kind_by_name, KINDS, Q4_0, Q8_0};

    /// Deterministic pseudo-random bytes, the same generator shape
    /// `gpu.rs`'s K-quant fixtures already use.
    fn pseudo_bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect()
    }

    /// Q8_0 weights produced by `ferrox_quant`'s real quantizer, so the
    /// bytes are a format a loader would actually hand the kernel.
    fn q8_0_matrix(n_rows: usize, n_cols: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for r in 0..n_rows {
            let row: Vec<f32> = (0..n_cols)
                .map(|i| (((r * n_cols + i) as f32) * 0.037).sin())
                .collect();
            out.extend(ferrox_quant::quantize_q8_0(&row));
        }
        out
    }

    /// Q4_0 has no quantizer in `ferrox_quant` (the format is load-only
    /// here), so build valid block bytes directly: an f16 scale that is
    /// exactly representable, then 16 nibble pairs.
    fn q4_0_matrix(n_rows: usize, n_cols: usize) -> Vec<u8> {
        let blocks = n_cols / 32;
        let mut out = Vec::new();
        for r in 0..n_rows {
            for b in 0..blocks {
                let scale = half::f16::from_f32(0.05 + ((r * blocks + b) % 13) as f32 * 0.01);
                out.extend_from_slice(&scale.to_le_bytes());
                let nibbles = pseudo_bytes((r * blocks + b) as u32 + 7, 16);
                out.extend_from_slice(&nibbles);
            }
        }
        out
    }

    fn activations(batch: usize, n_cols: usize) -> Vec<f32> {
        (0..batch * n_cols)
            .map(|i| ((i as f32) * 0.019).cos())
            .collect()
    }

    /// An independent GEMM: dequantize with `ferrox_quant` (a different
    /// implementation, written for a different purpose, cross-validated
    /// against NumPy) and do a plain dot product. This is the thing the
    /// twin is checked against; if the twin's tiling or its unpack is
    /// wrong, these disagree.
    fn independent_gemm(
        dequant_row: impl Fn(&[u8]) -> Vec<f32>,
        weights: &[u8],
        x: &[f32],
        n_rows: usize,
        n_cols: usize,
        batch: usize,
        row_bytes: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; batch * n_rows];
        for r in 0..n_rows {
            let w = dequant_row(&weights[r * row_bytes..(r + 1) * row_bytes]);
            assert_eq!(w.len(), n_cols);
            for t in 0..batch {
                let xr = &x[t * n_cols..(t + 1) * n_cols];
                let mut acc = 0f32;
                for k in 0..n_cols {
                    acc += w[k] * xr[k];
                }
                out[t * n_rows + r] = acc;
            }
        }
        out
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let scale = w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol * scale,
                "{what}: element {i}: twin={g} reference={w}"
            );
        }
    }

    /// The f16 helper the kernels use is hand-written bit surgery, not a
    /// library call. Hold it against `half`, denormals and specials
    /// included.
    #[test]
    fn f16_twin_matches_half_crate_over_every_bit_pattern() {
        for bits in 0u32..=0xFFFF {
            let bits = bits as u16;
            let want = half::f16::from_bits(bits).to_f32();
            let got = f16_to_f32(bits);
            if want.is_nan() {
                assert!(got.is_nan(), "bits {bits:#06x}: want NaN, got {got}");
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "bits {bits:#06x}: got {got}, want {want}"
                );
            }
        }
    }

    /// The single most likely porting error: `il` selecting the wrong 16
    /// elements, or the right 16 in the wrong order. Decode a super-block
    /// sub-block by sub-block and require it to equal `ferrox_quant`'s
    /// whole-block dequantization, element for element.
    #[test]
    fn q8_0_sub_block_twin_reconstructs_the_block_in_order() {
        let row: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.31).sin()).collect();
        let bytes = ferrox_quant::quantize_q8_0(&row);
        let want = ferrox_quant::dequant_q8_0(&bytes).unwrap();
        for (blk, chunk) in bytes.chunks(Q8_0.block_bytes).enumerate() {
            for il in 0..Q8_0.nl() {
                let mut reg = [0f32; SUB];
                (Q8_0.dequant_twin)(chunk, il, &mut reg);
                for (i, got) in reg.iter().enumerate() {
                    let want = want[blk * Q8_0.block_elems + il * SUB + i];
                    assert_eq!(*got, want, "block {blk} il {il} elem {i}");
                }
            }
        }
    }

    #[test]
    fn q4_0_sub_block_twin_reconstructs_the_block_in_order() {
        let bytes = q4_0_matrix(1, 64);
        let want = ferrox_quant::dequant_q4_0(&bytes).unwrap();
        for (blk, chunk) in bytes.chunks(Q4_0.block_bytes).enumerate() {
            for il in 0..Q4_0.nl() {
                let mut reg = [0f32; SUB];
                (Q4_0.dequant_twin)(chunk, il, &mut reg);
                for (i, got) in reg.iter().enumerate() {
                    let want = want[blk * Q4_0.block_elems + il * SUB + i];
                    // llama computes `d*q + (-8*d)`, ferrox_quant
                    // computes `(q-8)*d`; the two differ by at most one
                    // fp32 rounding of the same value.
                    assert!(
                        (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                        "block {blk} il {il} elem {i}: twin={got} reference={want}"
                    );
                }
            }
        }
    }

    /// Exact tiling: rows and batch both land on tile boundaries, so no
    /// clamp or zero-fill is exercised. If this fails, the K-loop or the
    /// micro-tile mapping is wrong.
    #[test]
    fn q8_0_gemm_twin_matches_independent_reference_on_exact_tiles() {
        let (n_rows, n_cols, batch) = (BM * 2, 128, BN);
        let row_bytes = (n_cols / 32) * Q8_0.block_bytes;
        let weights = q8_0_matrix(n_rows, n_cols);
        let x = activations(batch, n_cols);
        let got = mul_mm_reference(&Q8_0, &weights, &x, n_rows, n_cols, batch, row_bytes).unwrap();
        let want = independent_gemm(
            |r| ferrox_quant::dequant_q8_0(r).unwrap(),
            &weights,
            &x,
            n_rows,
            n_cols,
            batch,
            row_bytes,
        );
        assert_close(&got, &want, 1e-5, "q8_0 exact tiles");
    }

    /// Partial tiles on both axes: `n_rows` is not a multiple of `BM`
    /// (exercises the out-of-range row clamp) and `batch` is not a
    /// multiple of `BN` (exercises the zero-filled B-tile columns and the
    /// bounds-checked store). This is where a kernel written without a
    /// device usually writes past the end.
    #[test]
    fn q8_0_gemm_twin_matches_independent_reference_on_partial_tiles() {
        let (n_rows, n_cols, batch) = (BM + 7, 96, BN + 5);
        let row_bytes = (n_cols / 32) * Q8_0.block_bytes;
        let weights = q8_0_matrix(n_rows, n_cols);
        let x = activations(batch, n_cols);
        let got = mul_mm_reference(&Q8_0, &weights, &x, n_rows, n_cols, batch, row_bytes).unwrap();
        let want = independent_gemm(
            |r| ferrox_quant::dequant_q8_0(r).unwrap(),
            &weights,
            &x,
            n_rows,
            n_cols,
            batch,
            row_bytes,
        );
        assert_close(&got, &want, 1e-5, "q8_0 partial tiles");
    }

    /// batch = 1 is the decode shape. A GEMM that is only right for wide
    /// batches would still be wrong for every token after the prompt.
    #[test]
    fn q8_0_gemm_twin_matches_independent_reference_at_batch_one() {
        let (n_rows, n_cols, batch) = (37, 64, 1);
        let row_bytes = (n_cols / 32) * Q8_0.block_bytes;
        let weights = q8_0_matrix(n_rows, n_cols);
        let x = activations(batch, n_cols);
        let got = mul_mm_reference(&Q8_0, &weights, &x, n_rows, n_cols, batch, row_bytes).unwrap();
        let want = independent_gemm(
            |r| ferrox_quant::dequant_q8_0(r).unwrap(),
            &weights,
            &x,
            n_rows,
            n_cols,
            batch,
            row_bytes,
        );
        assert_close(&got, &want, 1e-5, "q8_0 batch 1");
    }

    /// The second kind through the same GEMM body. This is the test that
    /// says the seam is a seam: nothing but the table row changed.
    #[test]
    fn q4_0_gemm_twin_matches_independent_reference_on_partial_tiles() {
        let (n_rows, n_cols, batch) = (BM + 3, 96, BN + 9);
        let row_bytes = (n_cols / 32) * Q4_0.block_bytes;
        let weights = q4_0_matrix(n_rows, n_cols);
        let x = activations(batch, n_cols);
        let got = mul_mm_reference(&Q4_0, &weights, &x, n_rows, n_cols, batch, row_bytes).unwrap();
        let want = independent_gemm(
            |r| ferrox_quant::dequant_q4_0(r).unwrap(),
            &weights,
            &x,
            n_rows,
            n_cols,
            batch,
            row_bytes,
        );
        assert_close(&got, &want, 1e-5, "q4_0 partial tiles");
    }

    /// Sabotage check for the tests above: perturbing one weight byte
    /// must move the twin's output. A GEMM test that passes on data it
    /// never reads is not a test.
    #[test]
    fn twin_output_depends_on_every_part_of_the_weight_matrix() {
        let (n_rows, n_cols, batch) = (BM + 7, 96, 3);
        let row_bytes = (n_cols / 32) * Q8_0.block_bytes;
        let base = q8_0_matrix(n_rows, n_cols);
        let x = activations(batch, n_cols);
        let want = mul_mm_reference(&Q8_0, &base, &x, n_rows, n_cols, batch, row_bytes).unwrap();
        // Last row, last block, last quant: the corner a tiling bug is
        // most likely to skip.
        let mut poked = base.clone();
        let last = poked.len() - 1;
        poked[last] = poked[last].wrapping_add(64);
        let got = mul_mm_reference(&Q8_0, &poked, &x, n_rows, n_cols, batch, row_bytes).unwrap();
        assert!(
            got.iter().zip(want.iter()).any(|(a, b)| a != b),
            "poking the last weight byte changed nothing -- the twin is not reading it"
        );
    }

    #[test]
    fn shape_validation_names_what_it_refuses() {
        // n_cols not a whole number of blocks.
        let err = mul_mm_reference(&Q8_0, &[], &[], 4, 48, 1, 51).unwrap_err();
        assert!(
            matches!(err, MulMmUnsupported::ColsNotTileAligned { .. }),
            "got {err:?}"
        );
        // row_bytes inconsistent with n_cols.
        let err = mul_mm_reference(&Q8_0, &[0; 999], &[0.0; 64], 4, 64, 1, 33).unwrap_err();
        assert!(
            matches!(err, MulMmUnsupported::RowBytesMismatch { .. }),
            "got {err:?}"
        );
        // Weight buffer short of n_rows * row_bytes.
        let err = mul_mm_reference(&Q8_0, &[0; 68], &[0.0; 64], 4, 64, 1, 68).unwrap_err();
        assert!(
            matches!(err, MulMmUnsupported::WeightsTooSmall { .. }),
            "got {err:?}"
        );
        // Activation buffer short of batch * n_cols.
        let err = mul_mm_reference(&Q8_0, &[0; 272], &[0.0; 64], 4, 64, 2, 68).unwrap_err();
        assert!(
            matches!(err, MulMmUnsupported::ActivationsTooSmall { .. }),
            "got {err:?}"
        );
        assert!(mul_mm_reference(&Q8_0, &[], &[], 0, 64, 1, 68).is_err());
    }

    /// Every row of the table has to describe a real GGUF format, and
    /// the geometry the kernel is `#define`d with has to be that
    /// format's actual geometry. A wrong `block_bytes` here would stride
    /// the whole weight matrix incorrectly on a device and produce
    /// plausible garbage.
    #[test]
    fn kind_table_geometry_matches_ferrox_quant() {
        assert_eq!(Q8_0.block_bytes, ferrox_quant::Q8_0_BLOCK_BYTES);
        assert_eq!(Q8_0.block_elems, ferrox_quant::Q8_0_BLOCK_ELEMS);
        assert_eq!(Q4_0.block_bytes, ferrox_quant::Q4_0_BLOCK_BYTES);
        assert_eq!(Q4_0.block_elems, ferrox_quant::Q4_0_BLOCK_ELEMS);
        use crate::mul_mm::{Q4_K, Q5_0, Q5_K, Q6_K};
        assert_eq!(Q5_0.block_bytes, ferrox_quant::Q5_0_BLOCK_BYTES);
        assert_eq!(Q5_0.block_elems, ferrox_quant::Q5_0_BLOCK_ELEMS);
        assert_eq!(Q4_K.block_bytes, ferrox_quant::Q4_K_BLOCK_BYTES);
        assert_eq!(Q4_K.block_elems, ferrox_quant::Q4_K_BLOCK_ELEMS);
        assert_eq!(Q5_K.block_bytes, ferrox_quant::Q5_K_BLOCK_BYTES);
        assert_eq!(Q5_K.block_elems, ferrox_quant::Q5_K_BLOCK_ELEMS);
        assert_eq!(Q6_K.block_bytes, ferrox_quant::Q6_K_BLOCK_BYTES);
        assert_eq!(Q6_K.block_elems, ferrox_quant::Q6_K_BLOCK_ELEMS);
        for k in KINDS {
            assert_eq!(
                k.block_elems,
                k.nl() * SUB,
                "{}: nl() must partition the super-block into {SUB}-element sub-blocks",
                k.name
            );
            assert!(
                BK.is_multiple_of(SUB),
                "the K-tile must be a whole number of sub-blocks"
            );
            assert_eq!(kind_by_name(k.name).map(|f| f.name), Some(k.name));
        }
        // Q4_K, Q5_K and Q6_K resolve as of 2026-09-04, Q5_0 as of
        // 2026-09-05. IQ4_XS does
        // not, and the distinction is not an oversight: it is a
        // codebook lookup rather than an affine dequant, so it does
        // not fit `dequant_src`'s shape and needs its own row. A kind
        // that resolves without a kernel would compute silently wrong
        // numbers, which is the failure this line stands against.
        assert!(
            kind_by_name("IQ4_NL").is_none() && kind_by_name("IQ4_XS").is_none(),
            "a kind with no mul_mm must not resolve"
        );
    }

    /// The emitted translation unit has to define the entry point the
    /// launch path asks NVRTC for, and has to carry the geometry the
    /// twin assumed. This cannot prove the C compiles; it does catch a
    /// table row whose `fn_name` no longer matches its source.
    #[test]
    fn emitted_source_defines_the_entry_point_and_the_geometry() {
        for k in KINDS {
            let src = kernel_src(k);
            assert!(
                src.contains(&format!("__global__ void {}(", k.fn_name)),
                "{}: emitted source does not define {}",
                k.name,
                k.fn_name
            );
            assert!(
                src.contains("void ferrox_dequant_sub("),
                "{}: no unpack function",
                k.name
            );
            assert!(
                src.contains("float ferrox_f16_to_f32("),
                "{}: no f16 helper",
                k.name
            );
            assert!(
                src.contains(&format!("#define FX_BLOCK_BYTES {}\n", k.block_bytes)),
                "{}: block geometry not defined from the Rust constant",
                k.name
            );
            assert!(
                src.contains(&format!("#define FX_NL {}\n", k.nl())),
                "{}: sub-block count not defined from the Rust constant",
                k.name
            );
            // The twin's whole claim is that it walks the same tiles as
            // the kernel, and it reads BM/BN/BK/TM/TN from Rust while
            // the kernel reads them from these `#define`s. Two
            // structures that have to agree about one thing: pin them.
            // Without this, retuning a constant in Rust and leaving a
            // literal in the emitter gives a kernel that launches, and
            // a twin that agrees with itself about the wrong shape.
            for (name, value) in [
                ("FX_BM", BM),
                ("FX_BN", BN),
                ("FX_BK", BK),
                ("FX_TM", TM),
                ("FX_TN", TN),
                ("FX_THREADS", THREADS),
                ("FX_SUB", SUB),
            ] {
                assert!(
                    src.contains(&format!("#define {name} {value}\n")),
                    "{}: {name} is not emitted as the Rust constant {value}",
                    k.name
                );
            }
            // The `float4` loads are what keep a warp off eight banks.
            // A rewrite that quietly went back to scalar reads would be
            // correct and several times slower, which is the kind of
            // regression a numeric test cannot see.
            assert!(
                src.contains("const float4 v = *(const float4*)&sa[kk]")
                    && src.contains("const float4 v = *(const float4*)&sb[kk]"),
                "{}: the inner loop no longer loads its operands as float4",
                k.name
            );
            assert!(
                src.contains(&format!("#define FX_THREADS {THREADS}\n")),
                "{}: thread count not defined from the Rust constant",
                k.name
            );
            assert!(
                !src.contains("FX_FN_NAME"),
                "{}: unsubstituted name",
                k.name
            );
        }
        // Two kinds must not collide in the process-wide module cache.
        assert_ne!(Q8_0.module_name, Q4_0.module_name);
        assert_ne!(Q8_0.fn_name, Q4_0.fn_name);
    }

    /// A batch of one is a matvec, and the matvec kernels are the arm
    /// that has actually run on hardware. The GEMM must not claim it.
    ///
    /// The threshold is DERIVED from the tile width, so this asserts the
    /// property rather than the number. It used to spell `8`, which was
    /// `BN / 4` at the time; retuning `BN` to 64 moved the threshold and
    /// turned a passing test red for no defect. A test that restates a
    /// derived constant is one more structure that has to agree with
    /// another one.
    #[test]
    fn single_token_dispatches_stay_on_the_matvec_path() {
        use crate::mul_mm::worth_a_gemm;
        assert!(!worth_a_gemm(0));
        assert!(!worth_a_gemm(1), "one token is a matvec at any tile width");

        // Monotone, so there is one threshold rather than a range of
        // shapes that flip back and forth.
        let threshold = (1..=4 * BN)
            .find(|b| worth_a_gemm(*b))
            .expect("some batch is worth a GEMM");
        assert!(threshold >= 2, "never a single token");
        assert!(
            threshold <= BN,
            "a full tile of tokens must be worth a GEMM, threshold {threshold} > BN {BN}"
        );
        for b in 1..4 * BN {
            assert_eq!(
                worth_a_gemm(b),
                b >= threshold,
                "batch {b} disagrees with threshold {threshold}"
            );
        }
    }

    /// A partial tile on both axes must still be *covered* by the grid:
    /// one row or one token past a tile boundary needs a second tile, not
    /// a silently dropped output. (The tile geometry itself is pinned by
    /// the `const _: () = assert!(..)` gates in `mul_mm`, which fail the
    /// build rather than a test.)
    #[test]
    fn the_grid_covers_a_partial_tile_on_both_axes() {
        assert_eq!(grid_dims(BM, BN), (1, 1));
        assert_eq!(grid_dims(BM + 1, BN + 1), (2, 2));
        assert_eq!(grid_dims(BM * 3, BN * 2 + 1), (3, 3));
    }
}
