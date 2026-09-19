//! Emits everything `tools/mul_mm_host_check/run.sh` needs to execute
//! the *real* `mul_mm` CUDA C on a host CPU and compare it against the
//! scalar twin: one `.cu` per quant kind, a fixture per shape, the
//! twin's expected output, and a manifest tying them together.
//!
//! This is not a substitute for a GPU. It cannot see a race a real warp
//! scheduler would expose, and it cannot check the launch configuration.
//! It does check that the emitted translation unit compiles, that its
//! index arithmetic addresses what the twin addresses, and that its
//! unpack decodes what the twin decodes -- against the kernel text
//! itself rather than against a Rust paraphrase of it.
//!
//! Run it through `tools/mul_mm_host_check/run.sh`, not directly.

use frink_cuda::mul_mm::{kernel_src, KINDS};
use frink_cuda::mul_mm_ref::mul_mm_reference;

/// Exact tiles, partial on both axes, and a narrow batch.
const SHAPES: &[(usize, usize, usize)] = &[(128, 128, 32), (71, 96, 37), (33, 64, 3)];

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: mul_mm_host_check <output-dir>");
    let mut manifest = String::new();
    for k in KINDS {
        std::fs::write(format!("{dir}/{}.cu", k.name), kernel_src(k)).unwrap();
        for &(n_rows, n_cols, batch) in SHAPES {
            // `n_cols` is rounded UP to a whole super-block for this
            // kind rather than taken literally. `validate_shape`
            // refuses anything else, and it was right to: a 128-column
            // Q4_K row is not a Q4_K row. Without this the whole tool
            // panicked on the first K-quant in `KINDS` and checked
            // nothing at all, which is what it did between the
            // K-quants landing and 2026-09-05. The 32-element kinds
            // (Q8_0, Q4_0, Q5_0) keep the exact shapes they had.
            let n_cols = n_cols.next_multiple_of(k.block_elems);
            let row_bytes = (n_cols / k.block_elems) * k.block_bytes;
            // Deliberately arbitrary bytes, not a well-formed
            // quantization: it exercises degenerate f16 scales (NaN/Inf)
            // the same way `gpu.rs`'s K-quant fixtures do, and the
            // comparison requires the two sides to agree on those too.
            let mut state = 12345u32;
            let weights: Vec<u8> = (0..n_rows * row_bytes)
                .map(|_| {
                    state = state.wrapping_mul(1103515245).wrapping_add(12345);
                    (state >> 16) as u8
                })
                .collect();
            let x: Vec<f32> = (0..batch * n_cols)
                .map(|i| ((i as f32) * 0.019).cos())
                .collect();
            let want = mul_mm_reference(k, &weights, &x, n_rows, n_cols, batch, row_bytes).unwrap();

            let tag = format!("{}_{n_rows}_{n_cols}_{batch}", k.name);
            std::fs::write(format!("{dir}/{tag}.w"), &weights).unwrap();
            std::fs::write(
                format!("{dir}/{tag}.x"),
                x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            )
            .unwrap();
            std::fs::write(
                format!("{dir}/{tag}.want"),
                want.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
            .unwrap();
            manifest.push_str(&format!(
                "{} {} {tag} {n_rows} {n_cols} {batch} {row_bytes}\n",
                k.name, k.fn_name
            ));
        }
    }
    std::fs::write(format!("{dir}/manifest.txt"), manifest).unwrap();
}
