# Results vs llama.cpp

**Gap** = `llama.cpp / ferrox`, same host, same GGUF, same backend.
**Below 1.0 means ferrox is faster.** 🟢 better · ⚪ within ~5% · 🔴 slower.

The summary and detail tables below are **generated** from
[`receipts/engine/`](receipts/engine/) by `ferrox bench --render`. Do
not hand-edit them. Rows are never compared across machines: a gap only
means something against the host it was measured on.

## Measured elsewhere, no receipt

Two hosts were benchmarked with `bench -m --compare` rather than
`--suite`, so they wrote no receipt and are absent from the generated
tables. They are the two most interesting results in this file.

**aarch64 CPU** (rented 20-core Cortex-A725, idle). ferrox is **ahead**
here, and the decode column depends on one switch:

| Model | Test | ferrox | ferrox `spin` | llama.cpp | Gap |
|---|---|---|---|---|---|
| Llama-3.2-3B Q4_K_M | pp512 | **132.53** | | 46.40 | 🟢 **0.35×** |
| Llama-3.2-3B Q4_K_M | tg128 | 10.38 | **23.14** | 17.86 | 🟢 **0.77×** |
| Llama-3.1-8B Q4_K_M | pp512 | **61.17** | | 19.15 | 🟢 **0.31×** |
| Llama-3.1-8B Q4_K_M | tg128 | 6.64 | **12.41** | 9.06 | 🟢 **0.73×** |
| SmolLM2-135M Q8_0 | tg128 | 14.73 | 9.16 | 120.56 | 🔴 **8.2×** |

`FERROX_CPU_POOL=spin` turns decode from a loss into a win at 3B and
8B, and into a bigger loss at 135M. That is why it is still opt-in
([#27](https://github.com/antonellof/ferrox/issues/27)).

**CUDA on Ampere** (RTX 3060). The prefill gap is far worse than on the
Pascal rows in the generated table:

| Model | Test | ferrox | llama.cpp | Gap |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | pp512 | 185.33 | 10277.19 | 🔴 **55.5×** |
| Llama-3.2-1B Q4_K_M | tg128 | 24.50 | 282.17 | 🔴 **11.5×** |
| Llama-3.2-3B Q4_K_M | pp512 | 75.58 | 4276.84 | 🔴 **56.6×** |
| Llama-3.2-3B Q4_K_M | tg128 | 10.38 | 127.03 | 🔴 **12.2×** |

llama.cpp is 2.4× faster on Ampere than on Pascal; ferrox is not faster
at all. The prefill gap is not a constant factor, it widens with GPU
generation.

**CUDA decode, before and after coalescing the matvec kernels**
(RTX 3080, `tg128`, `--n-gpu-layers 99`, runs interleaved
`main, branch, main, branch`; PRs
[#144](https://github.com/antonellof/ferrox/pull/144) and
[#145](https://github.com/antonellof/ferrox/pull/145)). The old kernels
gave one thread a whole super-block, so 32 lanes read addresses one
block apart and each load instruction spread across as many cache lines
as it had lanes. A warp now takes the super-block and each lane one
contiguous slice.

| Model | Before | After | Change | GB/s after | % of 760 GB/s |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B Q5_K_M | 34.74 | **43.17** | 🟢 **+24.3%** | 39.3 | 5.2% |
| Llama-3.2-3B Q4_K_M | 20.74 | **23.24** | 🟢 **+12.1%** | 46.9 | 6.2% |
| Llama-3.2-1B Q8_0 | 51.73 | **56.98** | 🟢 **+10.1%** | 75.3 | 9.9% |

Output is byte-identical to the CPU reference on all three. The last
column is the point: llama.cpp reaches about 60% of card bandwidth, so
the access pattern was a real cost and was not the main one. The next
lever is occupancy, and the fused FFN and attention kernels have not
been touched at all.

## Open

| Issue | Gap | What is known |
|---|---|---|
| [#133](https://github.com/antonellof/ferrox/issues/133) | CUDA prefill, up to 56× | widens with GPU generation; GPU ~50% idle during prefill |
| [#133](https://github.com/antonellof/ferrox/issues/133) | CUDA decode, ~10× | memory-bound, not host- or arithmetic-bound: 5–10% of card bandwidth against llama.cpp's ~60%. Coalescing the matvecs bought 10–24%; the FFN and attention kernels are still uncoalesced |
| [#127](https://github.com/antonellof/ferrox/issues/127) | x86 CPU prefill, ~10× | uninvestigated; the decode half was a wrong default, now fixed |
| [#27](https://github.com/antonellof/ferrox/issues/27) | CPU decode default | `spin` wins at 3B/8B, loses at 135M, so it needs a size rule not a flag |
| [#128](https://github.com/antonellof/ferrox/issues/128) | ~60 ms fixed per-token cost | flat in thread count and model size; dominates small models on every backend |

## Method

Both engines pick their own thread count: llama.cpp defaults to
performance cores and loses 2× to 4× above them, so forcing a shared
count makes the comparison worse, not fairer. A warmup rep is
discarded; host load is recorded at both ends of every run.

Four traps, each of which put a wrong number in this file before:

- **A load average cannot see one busy core.** The `--max-load` guard
  passed for a day while a daemon held 97% of a core. Check `ps` too.
- **One instantaneous sample is not a measurement.** "CUDA decode at
  36% GPU utilization" came from an `nvidia-smi` taken after the run.
  The real figure was 86% to 93%, and the error cost a day and a PR.
- **A label is not a backend.** 13 CPU rows were published whose own
  receipts recorded `backend_active: "Metal"`. Deleted rather than
  corrected; a receipt whose label disagrees with what ran is now
  refused at write time.
- **A gap column cannot show a missing kernel.** A CUDA K-quant prefill
  read 4.88 tok/s and looked slow. There was no GEMM at all, and the
  fallback still answered correctly.

Do not compare this file to a pre-0.13 version: those receipts had no
warmup, so their prefill numbers include cold mmap page faults.

<!-- BEGIN ENGINE TABLE (generated by `ferrox bench --render`) -->

## Engine (`ferrox bench` vs `llama-bench`)

Measured on **3 hosts**, one section each. Rows are never compared across machines.

### Summary

| Host | Backend | Prefill gap | Decode gap |
|---|---|---|---|
| AMD Ryzen 9 7945HX with Radeon Graphics (16c) Linux 6.17.0-23-generic | CPU | 🔴 **6.26×** to 🔴 **10.14×** | 🔴 **1.06×** to 🔴 **1.92×** |
| Apple M2 Pro (10c/6p) macOS 26.6.1 | METAL | ⚪ **1.00×** to 🔴 **1.10×** | 🟢 **0.64×** to 🔴 **1.23×** |
| Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz (10c) Linux 5.15.0-160-generic | CUDA | 🔴 **10.75×** to 🔴 **17.41×** | 🔴 **9.22×** to 🔴 **19.15×** |

### AMD Ryzen 9 7945HX with Radeon Graphics (16c) Linux 6.17.0-23-generic

#### CPU

| Model | Test | ferrox tok/s | llama.cpp tok/s | Gap |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct Q4_K_M | pp512 | **89.61** | **908.72** | 🔴 **10.14×** |
| Llama-3.2-3B-Instruct Q4_K_M | pp512 | **31.54** | **316.49** | 🔴 **10.04×** |
| Meta-Llama-3.1-8B-Instruct Q4_K_M | pp512 | **13.29** | **132.01** | 🔴 **9.93×** |
| Gemma-2-2B-IT Q4_K_M | pp512 | **42.72** | **420.94** | 🔴 **9.85×** |
| Llama-3.2-1B-Instruct Q6_K | pp512 | **64.50** | **513.30** | 🔴 **7.96×** |
| Qwen3-0.6B Q8_0 | pp512 | **204.07** | **1413.34** | 🔴 **6.93×** |
| Qwen2.5-0.5B-Instruct Q8_0 | pp512 | **272.33** | **1860.01** | 🔴 **6.83×** |
| SmolLM2-135M-Instruct Q8_0 | pp512 | **574.98** | **3891.58** | 🔴 **6.77×** |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | pp512 | **112.39** | **734.28** | 🔴 **6.53×** |
| Gemma-3-1B-IT Q8_0 | pp512 | **146.65** | **917.50** | 🔴 **6.26×** |
| SmolLM2-135M-Instruct Q8_0 | tg128 | **206.83** | **397.71** | 🔴 **1.92×** |
| Llama-3.2-1B-Instruct Q6_K | tg128 | **37.28** | **53.84** | 🔴 **1.44×** |
| Llama-3.2-1B-Instruct Q4_K_M | tg128 | **46.53** | **67.10** | 🔴 **1.44×** |
| Llama-3.2-3B-Instruct Q4_K_M | tg128 | **18.91** | **27.02** | 🔴 **1.43×** |
| Meta-Llama-3.1-8B-Instruct Q4_K_M | tg128 | **9.06** | **12.07** | 🔴 **1.33×** |
| Gemma-2-2B-IT Q4_K_M | tg128 | **22.23** | **29.47** | 🔴 **1.33×** |
| Qwen3-0.6B Q8_0 | tg128 | **63.64** | **82.39** | 🔴 **1.29×** |
| Qwen2.5-0.5B-Instruct Q8_0 | tg128 | **85.86** | **99.03** | 🔴 **1.15×** |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | tg128 | **43.56** | **49.98** | 🔴 **1.15×** |
| Gemma-3-1B-IT Q8_0 | tg128 | **46.47** | **49.46** | 🔴 **1.06×** |

### Apple M2 Pro (10c/6p) macOS 26.6.1

#### Metal

| Model | Test | ferrox tok/s | llama.cpp tok/s | Gap |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct Q8_0 | pp512 | **4469.08** | **4909.41** | 🔴 **1.10×** |
| OLMoE-1B-7B-0924 Q4_0 | pp512 | **1411.98** | **1550.30** | 🔴 **1.10×** |
| Llama-3.2-1B-Instruct Q6_K | pp512 | **1699.31** | **1841.96** | 🔴 **1.08×** |
| Qwen3-0.6B Q8_0 | pp512 | **3312.80** | **3509.22** | 🔴 **1.06×** |
| Gemma-2-2B-IT Q4_K_M | pp512 | **864.51** | **914.85** | 🔴 **1.06×** |
| Llama-3.2-1B-Instruct Q4_K_M | pp512 | **1801.36** | **1884.33** | ⚪ **1.05×** |
| Gemma-3-1B-IT Q8_0 | pp512 | **2655.14** | **2776.49** | ⚪ **1.05×** |
| Llama-3.2-1B-Instruct IQ4_XS | pp512 | **1833.72** | **1903.21** | ⚪ **1.04×** |
| Llama-3.2-1B-Instruct Q5_K_M | pp512 | **1644.07** | **1696.82** | ⚪ **1.03×** |
| Llama-3.2-3B-Instruct Q4_K_M | pp512 | **641.64** | **660.11** | ⚪ **1.03×** |
| Phi-4-mini-Instruct Q4_K_M | pp512 | **549.10** | **561.12** | ⚪ **1.02×** |
| Meta-Llama-3.1-8B-Instruct Q4_K_M | pp512 | **266.49** | **271.79** | ⚪ **1.02×** |
| Mistral-7B-Instruct-v0.2 Q4_K_M | pp512 | **271.28** | **275.03** | ⚪ **1.01×** |
| SmolLM2-135M-Instruct Q8_0 | pp512 | **12086.82** | **12101.23** | ⚪ **1.00×** |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | pp512 | **2026.76** | **2024.56** | ⚪ **1.00×** |
| Gemma-4-E2B-IT Q4_K_M | pp512 | **14.27** | — | — |
| Gemma-2-2B-IT Q4_K_M | tg128 | **55.25** | **67.82** | 🔴 **1.23×** |
| OLMoE-1B-7B-0924 Q4_0 | tg128 | **156.02** | **167.82** | 🔴 **1.08×** |
| Llama-3.2-3B-Instruct Q4_K_M | tg128 | **62.41** | **64.13** | ⚪ **1.03×** |
| Mistral-7B-Instruct-v0.2 Q4_K_M | tg128 | **32.29** | **32.11** | ⚪ **0.99×** |
| Llama-3.2-1B-Instruct Q4_K_M | tg128 | **148.63** | **147.72** | ⚪ **0.99×** |
| Meta-Llama-3.1-8B-Instruct Q4_K_M | tg128 | **30.34** | **30.05** | ⚪ **0.99×** |
| Llama-3.2-1B-Instruct Q6_K | tg128 | **132.19** | **130.02** | ⚪ **0.98×** |
| Phi-4-mini-Instruct Q4_K_M | tg128 | **51.14** | **49.94** | ⚪ **0.98×** |
| Llama-3.2-1B-Instruct IQ4_XS | tg128 | **155.53** | **147.00** | 🟢 **0.95×** |
| Llama-3.2-1B-Instruct Q5_K_M | tg128 | **128.60** | **116.45** | 🟢 **0.91×** |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | tg128 | **125.23** | **109.44** | 🟢 **0.87×** |
| Gemma-3-1B-IT Q8_0 | tg128 | **94.81** | **82.68** | 🟢 **0.87×** |
| Qwen3-0.6B Q8_0 | tg128 | **159.58** | **114.57** | 🟢 **0.72×** |
| SmolLM2-135M-Instruct Q8_0 | tg128 | **315.42** | **217.11** | 🟢 **0.69×** |
| Qwen2.5-0.5B-Instruct Q8_0 | tg128 | **201.64** | **129.19** | 🟢 **0.64×** |
| Gemma-4-E2B-IT Q4_K_M | tg128 | **15.91** | — | — |

### Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz (10c) Linux 5.15.0-160-generic

#### CUDA

| Model | Test | ferrox tok/s | llama.cpp tok/s | Gap |
|---|---|---|---|---|
| SmolLM2-135M-Instruct Q8_0 | pp512 | **996.30** | **17346.46** | 🔴 **17.41×** |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | pp512 | **293.56** | **4404.46** | 🔴 **15.00×** |
| Qwen2.5-0.5B-Instruct Q8_0 | pp512 | **621.77** | **9131.45** | 🔴 **14.69×** |
| Qwen3-0.6B Q8_0 | pp512 | **431.80** | **6064.31** | 🔴 **14.04×** |
| Gemma-3-1B-IT Q8_0 | pp512 | **409.39** | **5090.90** | 🔴 **12.44×** |
| Llama-3.2-1B-Instruct Q5_K_M | pp512 | **346.56** | **4122.03** | 🔴 **11.89×** |
| Gemma-2-2B-IT Q4_K_M | pp512 | **181.60** | **2107.69** | 🔴 **11.61×** |
| Llama-3.2-1B-Instruct Q4_K_M | pp512 | **384.82** | **4318.25** | 🔴 **11.22×** |
| Llama-3.2-3B-Instruct Q4_K_M | pp512 | **144.59** | **1576.87** | 🔴 **10.91×** |
| Llama-3.2-1B-Instruct Q6_K | pp512 | **370.12** | **3979.69** | 🔴 **10.75×** |
| Gemma-2-2B-IT Q4_K_M | tg128 | **4.18** | **80.09** | 🔴 **19.15×** |
| Llama-3.2-3B-Instruct Q4_K_M | tg128 | **4.18** | **71.57** | 🔴 **17.13×** |
| Llama-3.2-1B-Instruct Q5_K_M | tg128 | **9.55** | **161.01** | 🔴 **16.87×** |
| Llama-3.2-1B-Instruct Q4_K_M | tg128 | **11.80** | **178.68** | 🔴 **15.14×** |
| Llama-3.2-1B-Instruct Q6_K | tg128 | **11.06** | **146.39** | 🔴 **13.24×** |
| Qwen2.5-0.5B-Instruct Q8_0 | tg128 | **17.76** | **227.78** | 🔴 **12.83×** |
| Qwen3-0.6B Q8_0 | tg128 | **15.60** | **177.34** | 🔴 **11.37×** |
| Gemma-3-1B-IT Q8_0 | tg128 | **10.16** | **112.26** | 🔴 **11.05×** |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | tg128 | **14.47** | **153.37** | 🔴 **10.60×** |
| SmolLM2-135M-Instruct Q8_0 | tg128 | **37.80** | **348.41** | 🔴 **9.22×** |

<!-- END ENGINE TABLE -->

## Notes

- **Gemma-4-E2B** uses `Gemma4Engine`, whose `pp*` is a sequential
  `forward_token` until batched prefill lands. Homebrew `llama-bench`
  has no `gemma4` arch, so its column is blank.
- **Mixtral** is skipped by `--fit-host` on the Apple host.
- **Metal regressions to keep off:** legacy GQA NSG=4, sequential
  GREEDY argmax, float4 elem, early Multi-CB. `FERROX_METAL_FA_VEC=0`
  costs ~25.5 pred.
- Run-to-run spread is ~20% on the Apple host; a claim tighter than
  that needs interleaved A/B.
