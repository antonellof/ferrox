//! Metal launch descriptions built from a [`WeightMatrix`].
//!
//! One module because two callers need it -- the decoder's four
//! eligibility checks and its per-layer launches, and
//! [`crate::gdn`]'s fused recurrent branch -- and a second copy of a
//! table mapping a quant kind to a kernel is exactly the disagreement
//! this repo keeps paying for. A kind added to
//! `ferrox_metal::gpu::matvec_launch_meta` and not here shows up as a
//! matrix that silently falls back to the host, not as a wrong answer,
//! which is why the mapping is asked for by NAME rather than matched
//! twice.

use ferrox_core::WeightMatrix;

pub(crate) fn matvec<'a>(m: &'a WeightMatrix) -> Option<ferrox_metal::gpu::MatvecLaunch<'a>> {
    match m {
        WeightMatrix::F32(t) => {
            let rows = t.shape[0];
            let cols = t.shape[1];
            let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
                ferrox_metal::gpu::matvec_launch_meta("F32")?;
            // SAFETY: f32 ↔ little-endian byte view for Metal upload/alias.
            let bytes = unsafe {
                std::slice::from_raw_parts(t.data.as_ptr() as *const u8, t.data.len() * 4)
            };
            Some(ferrox_metal::gpu::MatvecLaunch {
                kernel_src: src,
                fn_name,
                block_bytes,
                block_elems,
                weights: bytes,
                rows,
                row_bytes: cols * 4,
                rows_per_tg,
            })
        }
        WeightMatrix::Quantized {
            data,
            rows,
            cols: _,
            kind,
        } => {
            // The backend's own table decides, asked by the name
            // `QuantKind` gives. Matching kinds to kernels a SECOND
            // time is what left `Q5_0` and `PTQ1_0` unreachable from
            // every fused Metal path in this crate while
            // `ferrox_metal::gpu::MATVEC_KINDS` served both.
            let kind_name = kind.metal_kind_name()?;
            let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
                ferrox_metal::gpu::matvec_launch_meta(kind_name)?;
            // A zero-row matrix has no rows to stride over, so
            // there is no meaningful row size; `checked_div`
            // says that once instead of splitting it across a
            // guard and a bare division.
            let row_bytes = data.as_slice().len().checked_div(*rows).unwrap_or(0);
            Some(ferrox_metal::gpu::MatvecLaunch {
                kernel_src: src,
                fn_name,
                block_bytes,
                block_elems,
                weights: data.as_slice(),
                rows: *rows,
                row_bytes,
                rows_per_tg,
            })
        }
        // No fused kernel adds a LoRA delta, and the safetensors
        // MXFP4 pair has no Metal matvec. Spelled out rather than
        // `_` so a fifth storage has to answer here.
        // A folded matrix's launch would read the untransformed
        // activation; `apply` transforms and then runs the base.
        WeightMatrix::Mxfp4 { .. } | WeightMatrix::Adapted { .. } | WeightMatrix::Folded { .. } => {
            None
        }
    }
}
