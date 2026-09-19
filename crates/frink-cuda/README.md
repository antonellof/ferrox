# frink-cuda

CUDA/NVRTC kernels for Frink.

Capability detection always builds. Real CUDA dispatch sits behind
`--features cuda`. Loading is dynamic, through `cudarc`, so no toolkit
is needed to compile. Run the hardware tests only on a machine with a
GPU.

`mul_mm`, the batched quantized GEMM, is **unrun on hardware** and not
wired into any model path yet. It ships with a scalar twin
(`mul_mm_ref`) that the default `cargo test -p frink-cuda` exercises,
and with `tools/mul_mm_host_check/run.sh`, which compiles the emitted
CUDA C against a barrier shim and runs it on this CPU to check it
against that twin. Neither is a measurement. Nothing in `docs/` may
claim a CUDA GEMM capability until
`cargo test -p frink-cuda --features cuda -- --ignored` has run on a
real device.
