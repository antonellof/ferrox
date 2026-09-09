# Results vs llama.cpp

**Gap** = `llama.cpp / ferrox`, same host, same GGUF, same backend.
**Below 1.0 means ferrox is faster.** 🟢 better · ⚪ within ~5% · 🔴 slower.

**One generated section is behind the code.** It is left as measured
rather than adjusted, because a hand-edited receipt is not a
measurement; re-running the suite replaces them.

| Section | Predates | Understated by |
|---|---|---|
| CUDA | [#148](https://github.com/antonellof/ferrox/pull/148) | prefill, ~20–26% |

**Metal was re-measured on 2026-09-09** on ferrox 0.17.1, so it is
current: it now includes #150, #156 and the merges around them, and the
16 stale 0.13.3 receipts it replaces were deleted rather than kept
beside it, because two sections for one machine is not two hosts.
**Mistral-7B-Instruct Q4_K_M dropped out of the table** in that
re-measurement and is not a regression: `--fit-host` refuses it at ~10
GiB needed against 11.2 GiB free, since a run from swap is not a
measurement of the engine.

The summary and detail tables below are **generated** from
[`receipts/engine/`](receipts/engine/) by `ferrox bench --render`. Do
not hand-edit them. Rows are never compared across machines: a gap only
means something against the host it was measured on. A GPU row also names
the card it ran on, which the current Metal rows carry.

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
8B, and into a bigger loss at 135M. That is why it was opt-in.
[#155](https://github.com/antonellof/ferrox/pull/155) replaced the flag
with a work-size rule so the scheduler is chosen per operation, and
`FERROX_CPU_POOL` now only overrides it for A/B. **These rows predate
that change and have not been re-measured**, so they still describe the
old flag ([#27](https://github.com/antonellof/ferrox/issues/27)).

**CUDA now has receipts** and is in the generated table below, on an
RTX 3060, so it is no longer described here. Decode reads 2.2× to 5.0×
and prefill 22.6× to 33.8×. The one thing receipts cannot show is a
before/after against the same engine, so that stays:

**What coalescing the matvecs bought** (RTX 3070, runs interleaved
`main, branch, main, branch`; PRs
[#144](https://github.com/antonellof/ferrox/pull/144),
[#145](https://github.com/antonellof/ferrox/pull/145),
[#146](https://github.com/antonellof/ferrox/pull/146)). The old kernels
gave one thread a whole super-block, so 32 lanes read addresses one
block apart and every load instruction spread across as many cache
lines as it had lanes. A warp now takes the super-block and each lane
one contiguous slice.

| Model | Before | After | Change | GB/s after | % of 448 GB/s |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B Q5_K_M | 32.85 | **100.90** | 🟢 **+207%** | 92.0 | 20.5% |
| Llama-3.2-3B Q4_K_M | 18.38 | **48.50** | 🟢 **+164%** | 97.9 | 21.9% |
| Llama-3.2-1B Q8_0 | 45.75 | **59.81** | 🟢 **+31%** | 79.0 | 17.6% |

Output stays byte-identical to the CPU reference on all three. Most of
this is #146, and the reason is worth keeping: #144 and #145 routed
only `launch_matvec`, while the fused FFN — gate, up and down, about
83% of the weight bytes a 3B decode step reads — enqueues directly and
kept the old kernels. The benchmarks said "coalesced" and the FFN was
not. Q8_0 gains least because its old kernel already read 32 contiguous
bytes per thread.

Achieved bandwidth is the number to watch, not tok/s: 17% to 22% of the
card, against llama.cpp's ~60%. The access pattern was a real cost and
was not the last one.

**Metal, what concurrent encode bought** (M2 Pro, interleaved
`main, branch, main, branch`, `MTLCommandBuffer` GPU-clock, which is
immune to host load; [#150](https://github.com/antonellof/ferrox/pull/150)).
Gemma-class models were forced onto a serial encoder by a correctness
fix that outlived its cause, so they ran with no dispatch overlap.

| Model | Before | After | Change |
|---|---:|---:|---:|
| Gemma-3-1B Q8_0 | 8.24 | **7.15** ms/tok | 🟢 **−13.2%** |
| Gemma-2-2B Q4_K_M | 13.09 | **12.00** ms/tok | 🟢 **−8.3%** |

Output is byte-identical to `main`. That predicted Gemma-2-2B decode
would move from 1.23× to about 1.12×, and the 2026-09-09 re-measurement
confirms it at **1.11×**, still the worst Metal row.

**Metal, where the rest of the gap is** (quiet host, GPU-clock and wall
from one process; [#149](https://github.com/antonellof/ferrox/issues/149)).

| Model | wall ms/tok | GPU ms/tok | host | gap | gap if host were 0 |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 6.53 | **4.82** | 26% | 0.97× | **0.72×** |
| Gemma-2-2B Q4_K_M | 16.46 | **11.79** | 28% | 1.12× | **0.88×** |

ferrox's Metal kernels already finish faster than llama.cpp's whole
token. The remaining gap is host-side, and injecting dispatches to
measure the slope directly says it is **not** op count: a dispatch costs
**0.61 µs** of host time and **5.16 µs** of GPU time, so all ~515 of
Gemma-2's dispatches are 7% of its host cost. The host lever is
pipelining encode against execution; fusion is a GPU-side lever worth
~22% of GPU time.

## Open

| Issue | Gap | What is known |
|---|---|---|
| [#133](https://github.com/antonellof/ferrox/issues/133) | CUDA prefill, 22× to 34× | ~4× is tensor cores (`mul_mm` has none), ~5× is undiagnosed kernel efficiency. #148 bought 20–26% and ruled out dequant redundancy and occupancy |
| [#133](https://github.com/antonellof/ferrox/issues/133) | CUDA decode, 2.2× to 5.0× | memory-bound: 17–22% of card bandwidth against llama.cpp's ~60%. Coalescing closed 9–19× to 2–5×. What limits the rest is not diagnosed — the access pattern was a real cost and was not the last one |
| [#149](https://github.com/antonellof/ferrox/issues/149) | Metal decode, 1.11× worst row | kernels already beat llama.cpp's whole token, and the cost is host-side. [#156](https://github.com/antonellof/ferrox/pull/156) removed 13% of dispatches and 9% of barriers for **2.3%** of host time, so the count is not the lever and the hypothesis that it was is retired. Barriers were already hazard-driven. What is left is per-dispatch argument binding: ~2400 encoder calls per token against 418 dispatches and barriers |
| [#127](https://github.com/antonellof/ferrox/issues/127) | x86 CPU prefill, 6.3× to 10.1× | was a missing kernel tier. [#159](https://github.com/antonellof/ferrox/pull/159) added AVX2 GEMMs for all five interleaved kinds and a per-workload dispatch rule, verified by execution on real AVX2 but **not yet benchmarked**, so this gap number still describes the code before it |
| [#27](https://github.com/antonellof/ferrox/issues/27) | CPU decode default | the size rule landed in [#155](https://github.com/antonellof/ferrox/pull/155); the crossover constant is bracketed by the published numbers, not swept, and no before/after on a quiet host has been run. `MIN_TASK_MACS` is still there, which the issue asks to delete |
| [#128](https://github.com/antonellof/ferrox/issues/128) | CPU fixed per-token cost | **82–87% of the main thread is `__psynch_cvwait`**, parked on rayon's condvar, at 1 and 6 threads alike. ferrox dispatches a parallel region per matmul; llama.cpp's threads all run the graph. Neither pool fixes it |

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
- **A ratio across two models is not a marginal cost.** Dividing Metal
  host time by dispatch count across two models gave ~12 µs per
  dispatch. Injecting dispatches and measuring the slope gave **0.61
  µs** — wrong by 20×, and it pointed a whole plan at the wrong lever.
- **Check the effect clears the noise before believing a null.** A
  fusion removing 5% of dispatches was worth ~1.4% of wall against a
  ~1.1% noise floor. "No difference" measured nothing either way.

Do not compare this file to a pre-0.13 version: those receipts had no
warmup, so their prefill numbers include cold mmap page faults.

<!-- BEGIN ENGINE TABLE (generated by `ferrox bench --render`) -->

## Engine (`ferrox bench` vs `llama-bench`)

Measured on **3 hosts**, one section each. Rows are never compared across machines.

### Summary

| Host | Backend | Prefill gap | Decode gap |
|---|---|---|---|
| AMD Ryzen 9 7945HX with Radeon Graphics (16c) Linux 6.17.0-23-generic | CPU | 🔴 **6.26×** to 🔴 **10.14×** | 🔴 **1.06×** to 🔴 **1.92×** |
| Apple M2 Pro (10c/6p) macOS 26.6.2 + Apple M2 Pro | METAL | ⚪ **1.01×** to 🔴 **1.10×** | 🟢 **0.64×** to 🔴 **1.11×** |
| Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz (10c) Linux 5.15.0-186-generic + NVIDIA GeForce RTX 3060 | CUDA | 🔴 **22.55×** to 🔴 **33.79×** | 🔴 **2.18×** to 🔴 **5.04×** |

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

### Apple M2 Pro (10c/6p) macOS 26.6.2 + Apple M2 Pro

#### Metal

| Model | Test | ferrox tok/s | llama.cpp tok/s | Gap |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct Q8_0 | pp512 | **4485.46** | **4925.36** | 🔴 **1.10×** |
| OLMoE-1B-7B-0924 Q4_0 | pp512 | **1424.39** | **1552.45** | 🔴 **1.09×** |
| Llama-3.2-1B-Instruct Q6_K | pp512 | **1711.86** | **1847.15** | 🔴 **1.08×** |
| Gemma-2-2B-IT Q4_K_M | pp512 | **876.12** | **917.02** | ⚪ **1.05×** |
| Llama-3.2-1B-Instruct Q4_K_M | pp512 | **1814.86** | **1889.92** | ⚪ **1.04×** |
| Gemma-3-1B-IT Q8_0 | pp512 | **2685.48** | **2785.07** | ⚪ **1.04×** |
| Llama-3.2-1B-Instruct IQ4_XS | pp512 | **1852.11** | **1907.91** | ⚪ **1.03×** |
| Meta-Llama-3.1-8B-Instruct Q4_K_M | pp512 | **271.89** | **279.43** | ⚪ **1.03×** |
| Llama-3.2-3B-Instruct Q4_K_M | pp512 | **646.88** | **662.51** | ⚪ **1.02×** |
| Llama-3.2-1B-Instruct Q5_K_M | pp512 | **1658.64** | **1694.99** | ⚪ **1.02×** |
| Phi-4-mini-Instruct Q4_K_M | pp512 | **550.75** | **561.29** | ⚪ **1.02×** |
| Qwen3-0.6B Q8_0 | pp512 | **3449.90** | **3511.35** | ⚪ **1.02×** |
| SmolLM2-135M-Instruct Q8_0 | pp512 | **12004.03** | **12184.12** | ⚪ **1.02×** |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | pp512 | **2017.64** | **2035.53** | ⚪ **1.01×** |
| Gemma-4-E2B-IT Q4_K_M | pp512 | **14.27** | — | — |
| Gemma-2-2B-IT Q4_K_M | tg128 | **61.59** | **68.16** | 🔴 **1.11×** |
| Llama-3.2-3B-Instruct Q4_K_M | tg128 | **62.65** | **64.35** | ⚪ **1.03×** |
| Meta-Llama-3.1-8B-Instruct Q4_K_M | tg128 | **30.65** | **30.12** | ⚪ **0.98×** |
| Llama-3.2-1B-Instruct Q4_K_M | tg128 | **151.46** | **148.73** | ⚪ **0.98×** |
| Phi-4-mini-Instruct Q4_K_M | tg128 | **51.22** | **50.00** | ⚪ **0.98×** |
| Llama-3.2-1B-Instruct Q6_K | tg128 | **135.97** | **131.70** | ⚪ **0.97×** |
| OLMoE-1B-7B-0924 Q4_0 | tg128 | **160.09** | **153.39** | ⚪ **0.96×** |
| Llama-3.2-1B-Instruct IQ4_XS | tg128 | **156.69** | **146.64** | 🟢 **0.94×** |
| Llama-3.2-1B-Instruct Q5_K_M | tg128 | **129.30** | **116.60** | 🟢 **0.90×** |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | tg128 | **127.65** | **109.49** | 🟢 **0.86×** |
| Gemma-3-1B-IT Q8_0 | tg128 | **103.33** | **83.06** | 🟢 **0.80×** |
| Qwen3-0.6B Q8_0 | tg128 | **151.76** | **115.73** | 🟢 **0.76×** |
| Qwen2.5-0.5B-Instruct Q8_0 | tg128 | **202.05** | **130.70** | 🟢 **0.65×** |
| SmolLM2-135M-Instruct Q8_0 | tg128 | **317.40** | **202.84** | 🟢 **0.64×** |
| Gemma-4-E2B-IT Q4_K_M | tg128 | **16.23** | — | — |

### Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz (10c) Linux 5.15.0-186-generic + NVIDIA GeForce RTX 3060

#### CUDA

| Model | Test | ferrox tok/s | llama.cpp tok/s | Gap |
|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | pp512 | **302.19** | **10212.32** | 🔴 **33.79×** |
| Qwen3-0.6B Q8_0 | pp512 | **438.82** | **14049.28** | 🔴 **32.02×** |
| Qwen2.5-0.5B-Instruct Q8_0 | pp512 | **651.16** | **20442.80** | 🔴 **31.39×** |
| SmolLM2-135M-Instruct Q8_0 | pp512 | **995.37** | **28311.27** | 🔴 **28.44×** |
| Llama-3.2-1B-Instruct Q5_K_M | pp512 | **376.64** | **10203.41** | 🔴 **27.09×** |
| Llama-3.2-1B-Instruct Q4_K_M | pp512 | **416.29** | **10571.65** | 🔴 **25.39×** |
| Llama-3.2-3B-Instruct Q4_K_M | pp512 | **161.53** | **4060.99** | 🔴 **25.14×** |
| Gemma-2-2B-IT Q4_K_M | pp512 | **205.72** | **5140.69** | 🔴 **24.99×** |
| Gemma-3-1B-IT Q8_0 | pp512 | **451.52** | **10803.23** | 🔴 **23.93×** |
| Llama-3.2-1B-Instruct Q6_K | pp512 | **425.71** | **9598.55** | 🔴 **22.55×** |
| TinyLlama-1.1B-Chat-v1.0 Q8_0 | tg128 | **47.87** | **241.17** | 🔴 **5.04×** |
| Qwen3-0.6B Q8_0 | tg128 | **64.56** | **310.11** | 🔴 **4.80×** |
| SmolLM2-135M-Instruct Q8_0 | tg128 | **141.55** | **670.59** | 🔴 **4.74×** |
| Qwen2.5-0.5B-Instruct Q8_0 | tg128 | **89.38** | **387.65** | 🔴 **4.34×** |
| Gemma-3-1B-IT Q8_0 | tg128 | **51.98** | **176.24** | 🔴 **3.39×** |
| Llama-3.2-1B-Instruct Q5_K_M | tg128 | **88.43** | **257.49** | 🔴 **2.91×** |
| Gemma-2-2B-IT Q4_K_M | tg128 | **43.19** | **124.58** | 🔴 **2.88×** |
| Llama-3.2-3B-Instruct Q4_K_M | tg128 | **42.72** | **120.12** | 🔴 **2.81×** |
| Llama-3.2-1B-Instruct Q4_K_M | tg128 | **100.78** | **278.65** | 🔴 **2.76×** |
| Llama-3.2-1B-Instruct Q6_K | tg128 | **100.16** | **217.90** | 🔴 **2.18×** |

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
