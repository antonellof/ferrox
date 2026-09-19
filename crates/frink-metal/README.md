# frink-metal

Apple Metal kernels for Frink.

Capability detection always builds. Real Metal compute sits behind
`--features metal` (macOS, Apple Silicon). Quantized GEMM, attention,
and resident mmap-backed weight buffers.
