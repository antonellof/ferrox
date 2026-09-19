# frink-vulkan

The Vulkan **beachhead**, not a backend. One question, answered end to
end: *can frink reach a Vulkan device from Rust, upload a quantized
weight verbatim, run one compute shader, and read back a correct
answer?*

It can. See [`docs/plans/vulkan-beachhead-verdict.md`](../../docs/plans/vulkan-beachhead-verdict.md)
for the verdict, the costs, and what it does **not** establish.

```bash
cargo test -p frink-vulkan                      # SPIR-V half, no GPU needed
cargo run  -p frink-vulkan --features vulkan --example probe
cargo test -p frink-vulkan --features vulkan    # runs the shader on a real device
```

The GPU test is not `#[ignore]`d: it skips loudly when there is no
device, so it is free on a driverless machine and automatic everywhere
else.

## Why a separate crate

`CLAUDE.md` says "no new crate for something one crate uses", and this
crate has **no** caller today — so the rule deserves an answer rather
than a precedent.

- `frink-metal` and `frink-cuda` are both separate crates for the same
  reason: the backend's third-party dependency must be absent from the
  default build, and the backend's kernels must not enlarge the crate
  that dispatches to them. `ash` is optional here exactly as `cudarc` is
  there.
- The beachhead must not touch `frink-core`. `weight_matrix.rs` is the
  file `backend-seam-refactor` has to rewrite, and adding a third
  backend to it *before* that refactor is the copy-a-code-path-to-vary-it
  mistake `CLAUDE.md` names by name.
- Gates stay cheap. `cargo test -p frink-vulkan` is under a second; the
  same tests inside `frink-core` would ride behind that crate's whole
  suite.

If the verdict had been NO-GO, the honest disposition would be to delete
this crate rather than leave it as dead code. It is GO, and the crate is
the evidence, so it stays — with `publish = false` until it has a
caller.

## Getting a Vulkan device on macOS

There is no Vulkan on a stock Mac. Three bottled Homebrew formulae are
enough, and after them the loader finds the ICD with no environment
variable:

```bash
brew install molten-vk vulkan-loader glslang
```

`glslang` is not used to build anything — it supplies `spirv-val`, which
the shader tests use as an external authority on whether the emitted
module is legal SPIR-V. When it is absent that test says so and passes.

If `libvulkan` is installed somewhere the dynamic linker does not look,
set `FRINK_VULKAN_LOADER` to its full path.

## Layout

| File | What |
|---|---|
| `spirv.rs` | a minimal SPIR-V word emitter; no build script, no shader compiler |
| `q8_0_shader.rs` | the Q8_0 matvec compute shader, emitted from Rust |
| `q8_0_reference.rs` | its scalar twin, checked against `frink_quant` and `half` |
| `device.rs` | `ash` instance/device acquisition, incl. MoltenVK portability |
| `dispatch.rs` | buffers, descriptors, pipeline, dispatch, readback |

The first three compile and are tested **unconditionally**, with no
Vulkan driver present.
