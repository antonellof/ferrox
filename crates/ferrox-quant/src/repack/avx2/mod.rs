//! x86_64 AVX2 kernels for the interleaved repack tier, one file per
//! kind family. The dispatchers in `super::*` call these as
//! `avx2::<fn>`, which the re-exports below keep true after the split.

mod q4_kx8;

pub(crate) use q4_kx8::*;
