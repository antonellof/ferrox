//! ferrox-metal: Apple Silicon GPU capability detection (always
//! compiled, always tested) plus an optional, feature-gated Metal
//! compute execution path.
//!
//! Build without any GPU support (the default): `cargo build -p ferrox-metal`.
//! Build with the Metal scaffolding included (macOS only):
//! `cargo build -p ferrox-metal --features metal`.
//!
//! Unlike `ferrox-cuda`'s CUDA path (which had to be verified against
//! rented hardware since this development machine has no NVIDIA GPU),
//! the `metal` feature's kernels have been run directly on the real
//! Apple Silicon GPU this project is developed on -- see `gpu.rs`'s
//! module docs for exactly what's been verified and against what.

pub mod capability;

#[cfg(feature = "metal")]
pub mod gpu;

#[cfg(feature = "metal")]
pub mod attn;

#[cfg(feature = "metal")]
pub mod decode_dense;

#[cfg(feature = "metal")]
pub mod resident_act;

#[cfg(feature = "metal")]
mod resident_cache;

#[cfg(feature = "metal")]
pub mod greedy_fold;

#[cfg(feature = "metal")]
mod dispatch;

#[cfg(feature = "metal")]
mod mem_ranges;

#[cfg(feature = "metal")]
pub(crate) mod timing;

#[cfg(feature = "metal")]
pub(crate) mod kernel_timing;

#[cfg(all(test, feature = "metal"))]
mod kernel_bench;

#[cfg(feature = "metal")]
mod moe_ids;

#[cfg(feature = "metal")]
pub use dispatch::{metal_encode_stats, metal_encode_stats_reset, EncodeStats};

#[cfg(feature = "metal")]
pub mod elem;

#[cfg(feature = "metal")]
pub mod embd;

pub use capability::MetalProfile;
