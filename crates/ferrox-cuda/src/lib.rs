//! ferrox-cuda: hardware capability detection (always compiled, always
//! tested) plus an optional, feature-gated CUDA execution path.
//!
//! Build without any GPU support (the default): `cargo build -p ferrox-cuda`.
//! Build with the CUDA scaffolding included: `cargo build -p ferrox-cuda --features cuda`.
//!
//! The `cuda` feature compiles cleanly in this development sandbox
//! (which has neither a CUDA toolkit nor a GPU) because `cudarc` is
//! configured for dynamic loading -- the driver and NVRTC libraries
//! are `dlopen`'d at runtime, not linked at build time. That means
//! "this crate compiles with `--features cuda`" is a true, checked
//! fact. It does **not** mean the CUDA kernels in `gpu.rs` have ever
//! executed successfully; see that module's docs for exactly what has
//! and has not been verified.

pub mod capability;

/// The `mul_mm` kernel source and its per-quant-kind dispatch table.
/// Always compiled: it is CUDA C *text* plus a scalar twin, neither of
/// which needs `cudarc`, so the default `cargo test -p ferrox-cuda` run
/// on a GPU-less host still exercises the arithmetic the kernel encodes.
/// Only the launch path ([`mul_mm_launch`]) is feature-gated.
pub mod mul_mm;
pub mod mul_mm_ref;

#[cfg(feature = "cuda")]
pub mod attn;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod graph;
#[cfg(feature = "cuda")]
pub mod mul_mm_launch;

pub use capability::{HardwareProfile, SimdCaps};
