---
name: frink on AMD Strix Halo (Ryzen AI Max+)
overview: "GOAL: frink-server runs correctly and defensibly fast on AMD Ryzen AI Max (Strix Halo, gfx1151 / Radeon 8060S / 16C Zen 5 / up to 128 GB unified LPDDR5X). RECOMMENDED PATH: CPU-first on x86_64 Linux (measure, then AVX-512), and Vulkan — NOT HIPIFY of frink-cuda — as the eventual GPU backend. Two repo facts decide this and both are cited below: (1) frink has ZERO AVX-512 kernels while Strix Halo is the rare mobile-class part with a FULL 512-bit Zen 5 datapath, so the cheapest large win needs no new backend at all; (2) frink-cuda has no batched GEMM path whatsoever (`apply_gpu_batch` is `#[cfg(feature = \"metal\")]`-only, weight_matrix.rs:2884) and copies every weight to device memory (gpu.rs:754-756), so hipifying it lands a decode-only, double-footprint backend on a unified-memory box — precisely the two things Strix Halo punishes. The zero-copy UMA residency design frink needs ALREADY EXISTS, in frink-metal (`register_weight_mmap` / `BytesNoCopy`, gpu.rs:5477-5513), not in frink-cuda. HONESTY NOTE: nobody has run frink on this hardware, or on ANY x86_64 host in the published ledger (benchmarks/RESULTS.md:3 — Host B is an M2 Pro). Every performance claim in this document about frink on Strix Halo is a prediction, not a measurement, and is marked as such."
todos:
  - id: x86-first-measurement
    content: "BLOCKING EVERYTHING: there is no frink measurement on any x86_64 host. Build `x86_64-unknown-linux-gnu` (release.yml:32-35 already ships this target) on a Strix Halo box and run `bench --suite --fit-host --skip-missing --compare`. Until this exists every other item here is speculation. Acceptance: a Strix Halo CPU column in benchmarks/RESULTS.md, both engines measured in the same session per the parity plan's measurement contract, `uptime` recorded and below 2.0"
    status: pending
  - id: thread-default-x86
    content: "`default_worker_threads()` reads `hw.perflevel0.physicalcpu` on macOS and otherwise falls through to `available_parallelism()` (threads.rs:79-86). On 16C/32T Strix Halo that returns 32 SMT threads where llama.cpp defaults to 16 physical cores. The parity plan already measured CPU decode as a fork-join SCALING problem, not a throughput one (llama-cpp-parity-push.md, `cpu-decode-scaling`), so a 2x-oversubscribed pool is the worst possible starting point. Acceptance: physical-core detection on Linux (/sys/devices/system/cpu/*/topology/thread_siblings_list or /proc/cpuinfo core id), plus an A/B at 8/16/32 threads on one dense and one MoE model"
    status: pending
  - id: strict-kernels-on-x86
    content: "Run the x86 baseline under FRINK_STRICT_KERNELS=1 (kernel_registry.rs:630-638, seal_or_error kernel_registry.rs:708-714) BEFORE publishing any number. On x86 every aarch64 NEON/i8mm kernel compiles out (the parity plan already lost days to exactly this: 'Phase 1 landed eight kernel changes on x86, where every aarch64-gated kernel is compiled out'). The registry is the tool that turns that into a load-time error instead of a slow benchmark. Acceptance: the strict run either passes or produces an explicit list of quant kinds with no x86 kernel, and that list is what ranks avx512-int-dot"
    status: pending
  - id: avx512-int-dot
    content: "LARGEST LEVER WITH NO NEW BACKEND. `SimdCaps` detects avx512f and the code says so in its own comment: 'avx512f is reported here as detected-but-unused -- no AVX-512 kernel exists yet' (frink-cuda/src/capability.rs:56-58). frink-quant has 57 aarch64 target_feature sites (neon / neon,dotprod / neon,i8mm) against 30 x86 ones (avx2 / avx2,fma / fma) and zero avx512. Strix Halo has FULL 512-bit Zen 5 FPUs, unlike Strix Point's double-pumped 256-bit (chipsandcheese, URL in body). Port the Q8_0/Q4_0/Q4_K int-dot GEMV+GEMM to AVX-512 + AVX512-VNNI (`_mm512_dpbusd_epi32`), mirroring the NEON dotprod/i8mm structure. Acceptance: bit-exact against the existing scalar/AVX2 kernels on the checked-in fixtures, plus a measured pp512/tg128 A/B on the same host in the same session"
    status: pending
  - id: uma-residency-semantics
    content: "`FRINK_GPU_VRAM_BUDGET_BYTES` ('Cap GPU-resident MoE experts', CONFIG.md) and `FRINK_EXPERT_CACHE_BYTES` encode a discrete-GPU model: a small fast pool worth protecting from a large slow one. On Strix Halo there is ONE pool. Decide and document what these mean on UMA (proposal: on a UMA backend the expert-streaming path is disabled by default and the budget knob is a no-op that warns, rather than silently throttling residency for a boundary that does not exist). Acceptance: `frink inspect-plan` on a 128 GB Strix Halo reports a plan that does not double-count host and device bytes for the same weight"
    status: pending
  - id: gtt-carveout-doc
    content: "Document the host-side prerequisite honestly in docs/CONFIG.md or a new docs/STRIX_HALO.md: the iGPU's usable memory is set by BIOS UMA reservation plus the Linux GTT/TTM limit, not by the 128 GB figure on the box. AMD's own guidance is to keep the BIOS reservation minimal and raise TTM instead (rocm.docs.amd.com strixhalo page). This is a documentation item, not code, and it is a prerequisite for any GPU-backend measurement being reproducible. Acceptance: a reader can reproduce a stated frink number from the doc alone"
    status: pending
  - id: backend-seam-refactor
    content: "PREREQUISITE FOR ANY THIRD BACKEND. There is no backend trait. `kernel_registry::Backend` (kernel_registry.rs:74-78) is observability-only by explicit design ('Observe only. Nothing in this module may change a dispatch decision', kernel_registry.rs:60-64). Dispatch is literal per-backend cfg blocks with per-backend fn-pointer aliases of DIFFERENT ARITY (`CudaMatvecLaunchFn` weight_matrix.rs:385-387 vs `MetalMatvecLaunchFn` weight_matrix.rs:391-393), two non-identical kind tables in one function (weight_matrix.rs:2528-2547 CUDA, 2551-2560 Metal), and 75 backend cfg attributes in decoder.rs alone. Adding a third backend as-is means a third copy of all of it. Do the seam first: one launch-fn signature, one kind-capability table per backend behind a common shape, and the `let gpu = ...; #[cfg(feature = \"x\")] let gpu = gpu || ...;` shadowing idiom (weight_matrix.rs:986-994, decoder.rs:2844-2848) replaced. Acceptance: adding a stub backend touches <= 5 files"
    status: pending
  - id: vulkan-beachhead
    content: "GO/NO-GO GATE, do not skip to a full backend. Smallest honest Vulkan slice: `ash` + one SPIR-V compute shader for Q4_K matvec, host buffers imported zero-copy from the GGUF mmap via VK_EXT_external_memory_host, wired only into `apply_gpu`. Measure decode tok/s on one model against the CPU baseline from x86-first-measurement. Acceptance: a NUMBER, plus a written verdict. If the beachhead does not beat tuned-AVX-512 CPU decode by a margin larger than host spread, the full Vulkan backend is NOT justified and this plan is re-ranked rather than continued"
    status: pending
  - id: vulkan-decode-path
    content: "Gated on vulkan-beachhead passing. Q8_0/Q4_0/Q4_K/Q5_K/Q6_K matvec + fused QKV + fused SwiGLU FFN + GQA decode attention, i.e. functional parity with what frink-cuda has today (5 of 21 QuantKind variants, weight_matrix.rs:2528-2547). Use VK_KHR_shader_integer_dot_product where available. Acceptance: `frink verify` greedy-id parity against the CPU reference at 40/128/300-token prompts, on at least 3 models, per the parity plan's length-aware verify discipline"
    status: pending
  - id: vulkan-prefill-gemm
    content: "The item that decides whether this is worth doing at all. Batched prefill GEMM + prefill attention. NOTE THE ASYMMETRY: on Strix Halo the published HIP-vs-Vulkan split is HIP wins prefill / Vulkan wins decode (URLs in body), so a Vulkan-only backend inherits the WEAKER half on the axis frink is already worst at. Budget for subgroup-level tiling, not naive one-thread-per-output-cell. Acceptance: pp512 on the iGPU beats pp512 on the tuned CPU path by more than host spread"
    status: pending
  - id: ci-x86-and-vulkan
    content: "CI today builds cuda twice and metal once (ci.yml:32-67) and runs GPU tests never. Add: (a) an x86_64 Linux job that actually EXERCISES the x86 SIMD paths rather than only compiling them; (b) a `--features vulkan` compile-only job mirroring cuda-scaffolding + cuda-feature-chain the moment a vulkan feature exists (shader compilation to SPIR-V is a build-time step and CAN be validated with no GPU — this is strictly more testable than CUDA, where NVRTC compiles at runtime, gpu.rs:596); (c) FRINK_STRICT_KERNELS=1 in the test job, which is currently set nowhere in CI. Acceptance: a missing x86 kernel or a broken shader fails a PR"
    status: pending
  - id: bench-suite-on-128gb
    content: "`--fit-host` on a 128 GB box admits models that have never run under frink at all (Mixtral is skipped by --fit-host on the M2 Pro per RESULTS.md:113). Expect this to surface COVERAGE bugs, not speed gaps — and the parity plan's `coverage-fail-closed` item says ~50 archs currently load and emit wrong logits instead of refusing. Run the suite with FRINK_STRICT_KERNELS=1 and treat every new admission as a correctness question first. Acceptance: every newly-admitted model either produces verified-correct output or is refused at load"
    status: pending
  - id: hip-revisit-gate
    content: "The HIPIFY path is REJECTED below, not deleted. Reopen it if and only if ALL THREE hold: (1) vulkan-prefill-gemm lands and iGPU prefill is still behind llama.cpp's HIP backend by more than 1.3x on the same host; (2) frink has by then grown a real batched-GEMM GPU path so hipifying frink-cuda would not just clone its decode-only shape; (3) ROCm's gfx1151 support has stopped being version-fragile (see the rocWMMA contradiction in the body — two 2026 sources give OPPOSITE build flags). Acceptance: this todo is closed by a written re-decision citing measurements, never by preference"
    status: pending
  - id: docs-and-features-honesty
    content: "docs/FEATURES.md says 'Backends: CPU, Apple Metal, and CUDA' and README/ROADMAP make no claim about AMD. Do not add a Strix Halo row to FEATURES.md, MODELS.md or RESULTS.md until x86-first-measurement has produced same-session numbers. The repo's own worst failure mode is publishing a claim ahead of a measurement (llama-cpp-parity-push.md: 'Work that cannot be measured on the host that wrote it is not landed; it is staged'). Acceptance: no AMD claim ships without a receipt in benchmarks/receipts/engine/"
    status: pending
isProject: false
---

# frink on AMD Strix Halo (Ryzen AI Max+)

> Plan for running `frink-server` on AMD Ryzen AI Max / Max+ systems
> ("Strix Halo", gfx1151). Written **2026-08-14** from a read-only audit
> of `frink-cuda`, `frink-metal`, `frink-core`'s dispatch seams and
> the CI workflows, plus published third-party measurements of
> llama.cpp on this chip. Every repo claim carries a `file:line`; every
> hardware claim carries a URL.
>
> **This plan was written without access to the hardware.** No frink
> number for Strix Halo exists, and no frink number for *any* x86_64
> host exists in the published ledger — `benchmarks/RESULTS.md:3` names
> Host B (Apple M2 Pro) as the only host. Read every performance
> statement below as either (a) a cited third-party llama.cpp
> measurement, or (b) an explicitly-labelled prediction. There are no
> frink Strix Halo numbers to quote and none are invented here.

## The target, from primary sources

| Property | Value | Source |
|---|---|---|
| CPU | 16 Zen 5 cores, up to 5.1 GHz, two 8-core CCDs | [AMD](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html), [Chips and Cheese](https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix) |
| FPU width | **Full 512-bit**, same as desktop Zen 5 — *not* Strix Point's double-pumped 256-bit | [Chips and Cheese](https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix) |
| iGPU | Radeon 8060S, 40 CUs, RDNA 3.5, 32 MB Infinity Cache, `gfx1151` | [Chips and Cheese](https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix) |
| Memory | 256-bit LPDDR5X-8000, up to 128 GB, **256 GB/s shared** | [Chips and Cheese](https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix) |
| CPU-side bandwidth | ~64 GB/s read per CCD; **>175 GB/s across both CCDs** | [Chips and Cheese](https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix) |
| NPU | XDNA, up to 50 TOPS | [AMD](https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-395-processor-breakthrough-ai-.html) |

Two of those rows carry the whole plan.

**The 512-bit FPU row is why CPU-first is not a consolation prize.**
Strix Halo is the unusual mobile-class part where AVX-512 is real
silicon rather than two 256-bit passes, and frink has no AVX-512
kernel at all.

**The bandwidth rows are why CPU-first is not the destination either.**
Decode is bandwidth-bound — the parity plan states this outright
("Decode is one activation against the whole weight matrix, so it is
bandwidth- and latency-bound, not FLOP-bound"). The CPU cores reach
>175 GB/s aggregate; the iGPU addresses the full 256 GB/s and has 32 MB
of Infinity Cache the CPU cannot touch. So there is a **hard ceiling on
CPU-only decode on this chip, at roughly 68% of the iGPU's**, and no
amount of AVX-512 moves it. AVX-512 buys prefill (FLOP-bound) and buys
decode only up to that wall.

### The NPU is out of scope and should stay out

The 50-TOPS XDNA NPU is not addressable from any path frink has or is
likely to grow: it needs the Ryzen AI / XDNA driver stack and a
quantization format that is not GGUF. No item below touches it. Say so
in the docs rather than letting the number on the spec sheet imply a
capability.

## What llama.cpp actually does there, and what it costs

This is the honest comparison set. All numbers are third-party, on
Radeon 8060S / gfx1151, **not** reproduced here.

**ROCm 7.2.2 / Ubuntu 26.04, llama.cpp b8966, Q4_K_M, 96 GB VGM**
([nabe2030/hip-vs-vulkan-evo-x2](https://github.com/nabe2030/hip-vs-vulkan-evo-x2)):

| Model | pp16384 Vulkan | pp16384 HIP | tg128 Vulkan | tg128 HIP |
|---|---|---|---|---|
| Qwen 3.5-35B-A3B | 758 | **1122** (+48%) | **62.19** (+18%) | 52.73 |
| Gemma 4 31B-it | 161 | **232** (+44%) | **11.25** (+8%) | 10.44 |
| Qwen 3.5-122B-A10B | 366 | **415** (+13%) | **23.10** (+8%) | 21.38 |

**Independent replication of the same split**
([soothill.io, 2026-08-03](https://www.soothill.io/blog/2026/08/03/llamacpp-vulkan-vs-rocm-strix-halo/)),
Qwen3-Coder-30B-A3B Q4_K_S: Vulkan 1115.30 pp512 / 97.73 tg128 vs ROCm
1344.65 pp512 / 73.65 tg128 — ROCm +20.6% prefill, Vulkan +24.6%
decode.

The split reproduces across two independent testers, different
llama.cpp builds and different models: **HIP wins prefill, Vulkan wins
decode.** That is the single most important fact for choosing a path,
and it cuts against the recommendation below, which is why it is stated
before the recommendation rather than after.

### ROCm on gfx1151 is real but version-fragile

- gfx1151 was **not** on ROCm 7.0's supported-GPU list; official
  support arrived in later 7.x
  ([Phoronix](https://www.phoronix.com/review/amd-rocm-7-strix-halo),
  [ROCm issue #5339](https://github.com/ROCm/ROCm/issues/5339)).
- ROCm's own optimization page gates gfx1151 stability on **Linux
  6.18.4+** (or Ubuntu 24.04 HWE 6.17.0-19.19+)
  ([ROCm docs](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)).
- **Two 2026 sources give opposite build flags.** The Ubuntu 26.04 /
  ROCm 7.2.2 writeup says `GGML_HIP_ROCWMMA_FATTN=OFF` is "essential on
  gfx1151" and that enabling it costs ~60% of pp16384
  ([nabe2030](https://github.com/nabe2030/hip-vs-vulkan-evo-x2)); the
  llama.cpp known-good-stack discussion says `GGML_HIP_ROCWMMA_FATTN=ON`
  ([discussion #20856](https://github.com/ggml-org/llama.cpp/discussions/20856)).
  Both are recent, both are from people with the hardware. **This
  contradiction is unresolved and this plan does not resolve it.** It is
  evidence about the *stability of the platform*, not about who is
  right.
- The same discussion's author later **moved off HIP to Vulkan/RADV**
  and reported +28% single-stream decode, noting the HIP-specific
  `NO_VMM` / `-dio` workarounds "stop mattering entirely on Vulkan"
  ([discussion #20856](https://github.com/ggml-org/llama.cpp/discussions/20856)).

Vulkan's counterweight: RADV ships in-box on any modern Linux distro
with zero ROCm install, and gfx1151 supports `VK_KHR_cooperative_matrix`
(v1) though **not** `NV_cooperative_matrix2`.

## Recommendation

**CPU-first on x86_64 Linux (measure, then AVX-512), then Vulkan — not
HIPIFY — for the GPU backend.**

One line: *the cheapest large win on this chip needs no backend at all
(frink has zero AVX-512 kernels on a full-512-bit CPU), and the
existing CUDA crate is the wrong template to clone onto a unified-memory
box, because frink-metal — not frink-cuda — is where frink's UMA
design already lives.*

The evidence, in the order it decided things.

### 1. Hipifying `frink-cuda` clones a decode-only backend

`frink-cuda` is 5 source files, ~2,570 Rust lines and **464 lines of
CUDA C** across 8 `__global__` kernels
(`crates/frink-cuda/src/gpu.rs`, `attn.rs`). For contrast,
`frink-metal/src/{gpu.rs,attn.rs}` is 9,163 + 9,205 lines. CUDA is
roughly 5% of the Metal backend, and the ROADMAP says so plainly: CUDA
kernels "build and run on real hardware but have had no tuning pass"
(`docs/ROADMAP.md:41-42`).

Mechanically, HIPIFY would do well on the C:

- **Zero** occurrences of `mma.sync`, `wmma`, cuBLAS, cuDNN, cutlass,
  `__dp4a`, `__ldg`, inline PTX, cooperative groups, textures, or
  `cudaMallocManaged`. F16 decode is hand-rolled integer bit
  manipulation (`gpu.rs:151-163`) rather than `__half2float` —
  accidentally excellent for portability.
- 7 of the 8 kernels are `__shared__` + `__syncthreads()` block
  reductions at `blockDim.x = 256` (`gpu.rs:694-698`), warp-width
  agnostic and correct on wave64.
- Only `gqa_decode` (`attn.rs:22-79`, 56 lines) carries real wave32
  assumptions: a 32-bit full-warp mask `0xffffffffu` (`attn.rs:48`),
  `__shfl_down_sync`/`__shfl_sync` reductions (`attn.rs:55,57`), a
  `float acc[8]` register array silently sized as `256/32`
  (`attn.rs:42`), and `block_dim: (32,1,1)` hardcoded at both launch
  sites (`attn.rs:272,345`).

So the kernels are cheap. **The kernels are not the problem.**

**Problem A — the host layer is 100% `cudarc`, which has no HIP twin.**
`cudarc` is pinned `=0.11.9` (`frink-cuda/Cargo.toml:32-37`) and
appears in every signature: `compile_ptx` (`gpu.rs:596`), `load_ptx`
(`gpu.rs:598`), `htod_copy`, `alloc_zeros`, `dtoh_sync_copy`,
`LaunchConfig`, `CudaSlice<T>`, `Arc<CudaDevice>`, plus raw driver-API
FFI for `cuGraph*` because cudarc exposes no safe wrapper
(`graph.rs:3-6, 70-144`). HIPIFY does nothing for any of it. Rust HIP
bindings exist (`cubecl-hip-sys`, `hip-sys`, `rocm-rs`,
`oxicuda-rocm`), and hipRTC maps well onto frink's NVRTC-string design
— but this is a **rewrite of ~1,200 lines of host glue**, not a
translation.

**Problem B — and this is the decisive one — `frink-cuda` has no
batched GEMM at all.** `apply_gpu_batch` is
`#[cfg(feature = "metal")]`-only (`weight_matrix.rs:2884`); the CUDA
prefill path is a per-position matvec loop
(`weight_matrix.rs:1755-1780`); `decoder.rs` has a `try_metal_prefill_dense_stack`
(`decoder.rs:901-903`) with no CUDA analogue anywhere. CUDA covers **5
of 21 `QuantKind` variants** (`weight_matrix.rs:2528-2547`) — Metal
covers one more, and neither covers the IQ tiers that just landed.

Put those together against the measured Strix Halo split: HIP's
advantage on this chip is **prefill**, by 20-48%. Hipifying
`frink-cuda` produces a backend that **cannot do batched prefill at
all**. You would take on a ROCm dependency, a kernel-version floor, an
unresolved rocWMMA contradiction and a second GPU maintenance burden —
to acquire the half of the split HIP is worse at.

**Problem C — `frink-cuda`'s residency model is wrong for UMA, and
`frink-metal`'s is right.** CUDA uploads every weight with
`dev.htod_copy(weights.to_vec())` (`gpu.rs:754-756`), cached by
`(host_ptr, len)` and **never evicted** (`gpu.rs:729-763`, noted at
`gpu.rs:1152`). On a discrete GPU that is the entire point. On Strix
Halo, where GPU memory *is* system memory, it **doubles the resident
footprint of every weight and burns 256 GB/s of shared bandwidth to
move bytes to where they already are** — and it destroys frink's
stated load path, "GGUF mmap → keep quantized → fused dequant+dot"
(`CLAUDE.md`).

Metal already solved exactly this. `register_weight_mmap`
(`frink-metal/src/gpu.rs:5477-5513`) wraps an entire GGUF mmap in one
`newBufferWithBytesNoCopy` buffer with page-aligned length and a
keepalive `Arc<Mmap>` (`ResidentMmapFile`, `gpu.rs:5453-5468`), so
tensor slices alias the file at offsets instead of copying, degrading
gracefully to a copy if alignment fails (`gpu.rs:5443-5451`). The
Vulkan equivalent is `VK_EXT_external_memory_host` importing the same
host pointer.

**So: frink's unified-memory backend design already exists, and it is
the Metal crate.** A Strix Halo backend should be structured after
`frink-metal`, and the crate it should *not* be a copy of is
`frink-cuda`.

### 2. Why Vulkan, stated with its costs

For:

- No ROCm install, no DKMS, no kernel floor beyond a normal distro
  RADV. The reproducibility burden on users drops to near zero.
- Portable well beyond this chip (other RDNA parts, Intel Arc, ARM
  iGPUs), so the cost amortizes instead of being Strix-Halo-specific.
- The published decode advantage on this exact chip (+18-25% over HIP
  across two independent testers) sits on the axis the parity plan
  calls frink's weakest — CPU decode is "the only axis with nothing at
  parity."
- Shader compilation to SPIR-V is a **build-time** step, so a
  compile-only CI job genuinely validates the kernels. This is
  *strictly better* than CUDA, where NVRTC compiles at runtime
  (`gpu.rs:596`) and CI's two cuda jobs (`ci.yml:32-53`) therefore
  never touch a kernel body.

Against, stated plainly:

- **It is the largest single item in this plan by an order of
  magnitude.** Functional parity with Metal is ~18k lines. Parity with
  today's *CUDA* backend is much less, but today's CUDA backend is not
  a useful target (§1, Problem B).
- **Vulkan inherits the weaker half of the measured split.** A
  Vulkan-only frink backend is choosing the prefill-loser on a chip
  where frink is already prefill-behind. `vulkan-prefill-gemm` is
  where this plan is most likely to fail, and it is ranked and
  acceptance-gated accordingly.
- gfx1151 has `VK_KHR_cooperative_matrix` v1 but not coopmat2, and
  **lacks MFMA entirely**
  ([llm-tracker](https://llm-tracker.info/_TOORG/Strix-Halo)) — so the
  simdgroup-MMA playbook that fixed Metal prefill does not transfer
  unchanged.
- Nobody in this repo has written Vulkan. There is no `ash`,
  `vulkano`, `wgpu` or SPIR-V anywhere in the workspace.

That last set is why `vulkan-beachhead` is a **go/no-go gate with a
written verdict**, not a phase-one deliverable.

### 3. Why CPU-first is first, and where it stops

Not because it is easy — because **nobody knows the number.**
`benchmarks/RESULTS.md:3` names one host, an M2 Pro. Every frink
x86_64 claim in the repo is untested at the performance level. The
parity plan already burned days on exactly this failure: "Phase 1 (PRs
#2-#8) landed eight kernel changes on x86, where every aarch64-gated
kernel is compiled out, so none of them could be measured and none of
them were."

On x86 today frink runs on AVX2+FMA. `SimdCaps` detects `avx512f`
(`frink-cuda/src/capability.rs:34`) and the comment beside it is
already the finding:

> `avx512f` is reported here as detected-but-unused -- no AVX-512
> kernel exists yet
> — `crates/frink-cuda/src/capability.rs:56-58`

The `target_feature` census in `frink-quant` confirms the asymmetry:
57 aarch64 sites (26 `neon`, 24 `neon,dotprod`, 7 `neon,i8mm`) against
30 x86 sites (17 `avx2,fma`, 9 `avx2`, 4 `fma`), and **zero** avx512.
Every kernel the parity push spent Phase 1 building — interleaved
Q4_Kx8, i8mm SMMLA GEMM, int-dot GEMV — exists only for NEON.

And a first-day defect that costs nothing to fix:
`default_worker_threads()` reads `hw.perflevel0.physicalcpu` on macOS
and otherwise falls through to `available_parallelism()`
(`frink-core/src/threads.rs:79-86`). On 16C/32T Strix Halo that is 32
SMT threads where llama.cpp uses 16 physical cores — on an engine whose
measured CPU-decode deficit is *fork-join scaling*, not per-thread
throughput.

Where CPU-first stops: the >175 GB/s CCD-aggregate ceiling above. State
that limit in the docs at the same time the CPU numbers are published,
so nobody reads "frink runs on Strix Halo" as "frink uses Strix
Halo."

## Memory and MoE residency on unified memory

This is where Strix Halo differs most from every host frink has run
on, and where the existing knobs are wrong-shaped rather than
mis-tuned.

**The pool is one pool, and it is configurable at boot, not at
runtime.** Usable iGPU memory is BIOS UMA reservation plus the Linux
GTT/TTM limit. AMD's guidance is to keep the BIOS reservation minimal
(e.g. 0.5 GB) and raise the shared TTM limit instead, via
`/sys/module/ttm/parameters/pages_limit` or the `amd-ttm` helper
([ROCm docs](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)).
Field reports use `amdgpu.gttsize` / `ttm.pages_limit` boot parameters
to reach ~96-110 GB
([nabe2030](https://github.com/nabe2030/hip-vs-vulkan-evo-x2) ran at 96
GB VGM). **`amdgpu.gttsize` is deprecated in favour of
`ttm.pages_limit`** — verify against the running kernel rather than
copying a blog. None of this is frink's code, all of it is frink's
reproducibility problem, hence `gtt-carveout-doc`.

**Copy-based residency is a bug on this box, not a cost.** Restating
§1 Problem C in memory terms: `resident_cuda_weights` doubles every
weight's footprint (`frink-cuda/src/gpu.rs:754-763`). On a 96 GB
carve-out that halves the largest model that fits, which is the exact
thing `docs/ROADMAP.md:30-33` ("Run bigger models on the same
hardware") exists to prevent. Any Strix Halo backend must import the
mmap, following `frink-metal`'s `register_weight_mmap` pattern
(`gpu.rs:5477-5513`).

**MoE expert placement loses its premise.** `ExpertPlacement::{Cpu,
GpuDevice(u32)}` (`frink-moe/src/lib.rs:20-23`),
`FRINK_GPU_VRAM_BUDGET_BYTES` ("Cap GPU-resident MoE experts", `docs/CONFIG.md`),
`FRINK_EXPERT_CACHE_BYTES` and `FRINK_SSD_STREAMING` all encode
"small fast pool, large slow pool, decide what crosses." On Strix Halo
there is no crossing. Two consequences:

- The ROADMAP's headline MoE lever — "hot experts resident on GPU, cold
  ones streamed or run on CPU" (`docs/ROADMAP.md:37-40`) — is
  **structurally unnecessary here**, and that is *good news*: a 96 GB
  carve-out fits MoE checkpoints that need placement heuristics
  everywhere else. The published llama.cpp numbers on this chip are
  dominated by large MoE (Qwen 3.5-122B-A10B at 23 tok/s decode) for
  exactly this reason.
- But hybrid CPU/GPU MoE gains a *different* premise worth measuring
  later: the CPU and iGPU draw on one pool and one bandwidth budget, so
  running some experts on CPU while others run on the iGPU competes for
  bandwidth rather than adding a second source. **Unverified.** Do not
  design for it before measuring.

`uma-residency-semantics` covers making the knobs honest. The
conservative choice is a warning no-op, not a silent reinterpretation.

## What can be developed and tested without the hardware

Genuinely doable on the M2 Pro or in CI:

- **All of `backend-seam-refactor`.** It is a pure refactor of existing
  cfg branches, validated by `cargo test --workspace` plus the existing
  Metal and CUDA compile chains. It needs no AMD anything.
- **All AVX-512 kernel *correctness*.** frink's whole test culture
  supports this: fixtures and golden values cross-validated against
  independent NumPy references (`CLAUDE.md`), and the IQ-tier work was
  validated by linking llama.cpp's `ggml-quants.c` and asserting
  bit-exactness. AVX-512 kernels can be written and proven bit-exact
  against the existing scalar/AVX2 paths on any AVX-512 host —
  including a cloud VM, or under SDE. **What cannot be done off-host:
  any performance claim.** Zen 5's 512-bit datapath is the entire
  premise; an Intel or Zen 4 result does not transfer.
- **SPIR-V shader compilation and validation.** `glslangValidator` /
  `spirv-val` run with no GPU. A Vulkan job can check every shader
  compiles and its descriptor layout matches the Rust side.
- **Vulkan host-layer plumbing against a software driver.** lavapipe
  (Mesa's llvmpipe Vulkan ICD) runs compute shaders on CPU. It is
  correct and unusably slow — perfect for a correctness harness,
  useless for a number.
- **The kernel-registry arm, `frink caps`, `--list-devices`, config
  parsing, docs.** All CPU-side.
- **Every UMA design decision**, because the Metal crate already has a
  working zero-copy mmap import to read
  (`frink-metal/src/gpu.rs:5453-5513`).

Absolutely requires the hardware:

- **Every number.** No exceptions. Prefill/decode tok/s, thread-count
  A/Bs, the AVX-512 speedup, the Vulkan beachhead verdict, the
  GTT-limit ceiling.
- **The `vulkan-beachhead` go/no-go**, which is a measurement by
  definition.
- **Anything that depends on gfx1151 driver behaviour**: RADV vs AMDVLK,
  `VK_KHR_cooperative_matrix` v1 actually being fast, whether
  `VK_EXT_external_memory_host` import works against a `memmap2` mapping
  on this driver, subgroup size (RDNA is wave32-native but wave64-capable
  and the driver chooses).
- **GTT/TTM carve-out behaviour**, including whether a 96 GB import
  actually succeeds.

**Consequence for the todo ordering:** `x86-first-measurement` blocks
essentially everything, and there is no honest way around it. Get one
Strix Halo box — a mini-PC, or a rented one — before spending effort on
items 4 and later.

## CI implications

CI today (`.github/workflows/ci.yml`) is 5 jobs: fmt, `build + test`
(ubuntu, no GPU features), `cuda-scaffolding` (`ci.yml:32-39`),
`cuda-feature-chain` (`ci.yml:41-53`), `metal-feature-chain` on
macos-latest (`ci.yml:55-67`), and workspace clippy. **No GPU test ever
runs, and `FRINK_STRICT_KERNELS` is set nowhere in CI.**

Three specific changes:

1. **The x86 test job is weaker than it looks.** `cargo test --workspace`
   on ubuntu-latest is x86_64, so it *does* exercise the AVX2 paths —
   but GitHub's standard runners do not reliably expose AVX-512, so an
   AVX-512 kernel guarded by `is_x86_feature_detected!` would compile
   in CI and **never execute there**. That is the same shape as the
   Phase 1 aarch64 failure. Mitigations, in preference order: a
   larger-runner or self-hosted AVX-512 job; Intel SDE emulation for
   correctness only; or, at minimum, a test that **fails loudly** when
   the AVX-512 path was not taken, so a green CI cannot be mistaken for
   coverage.
2. **`FRINK_STRICT_KERNELS=1` belongs in the test job now**, ahead of
   any AMD work. `docs/CONFIG.md` already says to "set this in CI and in
   benchmark harnesses so a number cannot be published for a backend it
   was not taken on." It currently is not.
3. **A `vulkan` feature gets two jobs mirroring the CUDA pair** the
   moment it exists — and unlike CUDA, the compile-only job is
   *meaningful*, because SPIR-V is produced at build time. Add
   `spirv-val` to it. Optionally a lavapipe job that actually runs the
   correctness harness with no GPU; that is a capability the CUDA and
   Metal backends have never had.

`release.yml:24-35` builds exactly two targets:
`aarch64-apple-darwin --features metal` and
`x86_64-unknown-linux-gnu --features ""`. **The second one already
ships a Strix-Halo-compatible binary today** — CPU-only, AVX2, no
AVX-512, oversubscribed thread pool. That binary is the actual starting
line, and `x86-first-measurement` is measuring it.

## What is NOT known

Stated explicitly, in the repo's tradition of naming wrong prior
diagnoses rather than quietly correcting them.

1. **No frink measurement exists on any x86_64 host**, let alone this
   one. Everything in this plan about frink's CPU speed on Strix Halo
   is extrapolated from an M2 Pro ledger and a source-level reading of
   which kernels compile on which arch. It could be wrong in either
   direction.
2. **The AVX-512 win is unquantified.** The *existence* of the gap is
   proven from source (`capability.rs:56-58` plus the target_feature
   census). Its size is not. Predicting "N%" would be inventing a
   number. Zen 5's AVX-512 also has frequency behaviour that a
   throughput model ignores.
3. **The rocWMMA contradiction is unresolved.** Two recent sources with
   the hardware give opposite `GGML_HIP_ROCWMMA_FATTN` settings, one
   claiming ~60% pp16384 loss from the setting the other recommends.
   This plan cites both and resolves neither.
4. **Whether `VK_EXT_external_memory_host` import works against a
   `memmap2` mapping on RADV/gfx1151 is unverified.** The entire
   zero-copy residency argument rests on it. If it fails, the Vulkan
   path inherits `frink-cuda`'s copy problem and the recommendation
   weakens materially. **Test this first inside `vulkan-beachhead`,
   before writing a single kernel.**
5. **The prefill half of the Vulkan path is the plan's biggest
   unknown.** Two independent sources say HIP wins prefill on this chip
   by 20-48%, gfx1151 has no MFMA, and coopmat2 is unavailable. If
   `vulkan-prefill-gemm` cannot beat the tuned CPU path, this plan's GPU
   half has failed and `hip-revisit-gate` is the honest response.
6. **`frink bench --compare` requires `llama-bench` on the host.** No
   one has confirmed the compare harness works against a Vulkan or HIP
   llama.cpp build, or that the backend labels line up. Check before
   promising a comparison table.
7. **Whether Strix Halo's 128 GB actually surfaces coverage bugs is a
   guess.** `--fit-host` will admit models never run under frink, and
   the parity plan's `coverage-fail-closed` item says ~50 archs
   currently load and emit wrong logits rather than refusing. It is
   *likely* that a 128 GB host trips this. It is not established.
8. **Idle power, thermals and sustained-vs-burst behaviour are entirely
   unmeasured.** The parity plan's measurement contract exists because
   host load moved published rows by 25-45% on an M2 Pro. A 120 W-class
   APU in a mini-PC chassis may be worse. Establish this host's spread
   before quoting any gap tighter than it.
9. **No claim is made about `frink-server`'s HTTP/serving layer on this
   platform.** It is portable Rust and should be fine. "Should be fine"
   is not a measurement either.

## Defaults for this workstream

- **Measurement contract is inherited unchanged** from
  `llama-cpp-parity-push.md`: both engines in the same session, never in
  parallel, `uptime` below ~2.0, `--suite` + `--render` as the unit of
  truth, no number published from a partial or loaded run.
- **`FRINK_STRICT_KERNELS=1` on every Strix Halo measurement.** On a
  host where most of frink's fast kernels compile out, a silent CPU
  fallback is the single most likely way to publish a wrong number.
- **No AMD row in `FEATURES.md` / `MODELS.md` / `RESULTS.md` before a
  receipt exists** in `benchmarks/receipts/engine/`.
- **`frink-metal` is the structural reference for a UMA backend, not
  `frink-cuda`.** When the two crates disagree on a design question,
  Metal is right for this hardware.
- **The go/no-go gates are real.** `vulkan-beachhead` and
  `vulkan-prefill-gemm` each carry a numeric acceptance criterion and a
  written verdict. A failed gate re-ranks this plan; it does not get
  argued past.
