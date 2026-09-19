# frink-inference

Facade crate for [Frink](https://github.com/antonellof/frink), a
pure-Rust GGUF / MoE inference engine: mmap loaders, quantized CPU +
Apple Metal + CUDA kernels, and an OpenAI-compatible server.

It contains no logic. It re-exports the workspace under one name, so a
dependent writes one line instead of six, and so the project is
findable on crates.io, since the name `frink` belongs to an unrelated
crate.

```toml
[dependencies]
frink-inference = "0.13"
```

```rust
use frink_inference::gguf::ShardedGguf;

let file = ShardedGguf::open("model.gguf")?;
```

## The binaries are elsewhere

```bash
cargo install frink-cli      # installs the `frink` binary
cargo install frink-server   # OpenAI-compatible HTTP server
```

They are not shipped from this crate on purpose: two crates installing
a binary called `frink` would fight over the same path in
`~/.cargo/bin`.

## Features

| Feature | Effect |
|---|---|
| `metal` | Apple Metal kernels. Apple Silicon only. |
| `cuda` | CUDA/NVRTC kernels. Needs a CUDA toolkit at build time. |
| `api` | Re-export `frink-api` (route constants + wire DTOs). |

Neither GPU feature is on by default. `metal` does not build off Apple
Silicon, and `cuda` needs a toolkit most machines do not have.

The bar CUDA is held to is "must compile". There is no pinned benchmark
host for it and no published timings, so treat a Windows or Linux
install as CPU-only in practice. See
[`docs/FEATURES.md`](https://github.com/antonellof/frink/blob/main/docs/FEATURES.md).

## The rest of the workspace

| Crate | What it is |
|---|---|
| [`frink-gguf`](https://crates.io/crates/frink-gguf) | GGUF mmap reader, sharded checkpoints |
| [`frink-quant`](https://crates.io/crates/frink-quant) | Block layouts, fused dequant+dot |
| [`frink-safetensors`](https://crates.io/crates/frink-safetensors) | SafeTensors mmap reader |
| [`frink-core`](https://crates.io/crates/frink-core) | Tensor ops, RoPE, GQA, KV cache |
| [`frink-moe`](https://crates.io/crates/frink-moe) | Expert routing and dispatch |
| [`frink-models`](https://crates.io/crates/frink-models) | Loaders and decoder stacks |
| [`frink-edge`](https://crates.io/crates/frink-edge) | Serving policy: prefix caches, schedulers, output parsers |
| [`frink-api`](https://crates.io/crates/frink-api) | Route constants + wire DTOs |
| [`frink-metal`](https://crates.io/crates/frink-metal) | Apple Metal kernels |
| [`frink-cuda`](https://crates.io/crates/frink-cuda) | CUDA/NVRTC kernels |

Speed claims live in
[`benchmarks/RESULTS.md`](https://github.com/antonellof/frink/blob/main/benchmarks/RESULTS.md),
measured against llama.cpp on the same host and the same GGUF. That
table is generated from the raw timing files each run writes, and it
says so wherever nothing has been measured.

Apache-2.0.
