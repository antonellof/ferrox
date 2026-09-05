# Results vs llama.cpp

Host B = Apple M2 Pro (6 performance + 4 efficiency cores). Thread counts
are not forced. Each engine picks its own default.

Suite: [`suite.json`](suite.json). Runner: `ferrox bench --suite` vs
`llama-bench`. Receipts: [`receipts/engine/`](receipts/engine/).
**This file is generated** by `ferrox bench --render`. Do not hand-edit
the table below.

**Gap** = `llama / ferrox` (&lt;1 ferrox faster; &gt;1 ferrox slower).

**North star:** ≥ llama.cpp same host/GGUF/backend.

**Metal is at or past parity almost everywhere.** Every dense `pp512`
row lands between 0.98× and 1.10×, and 8 of the 12 `tg128` rows are
faster than llama.cpp, the widest being Qwen2.5-0.5B at 0.62×. The
remaining Metal gap is small enough that run-to-run spread explains
most of it.

**x86 CPU has rows, and they exist because a default was wrong.**
`FERROX_CPU_INT_DOT` defaults on, and the interleaved integer kernels it
selects were written for aarch64 (i8mm, interleave-8 NEON, the Q8_K
repack). On x86 it chose a scalar integer loop and simultaneously
bypassed the AVX2 f32 dot that does exist, costing between **4x and
8.8x of decode** on an idle Zen 4. Llama-3.2-1B Q4_K_M `tg64` went
10.08 to 46.62 tok/s against llama.cpp's 66.24 once the default became
architecture-aware, which is 6.8x off to 1.4x off (#127). The rows
below are taken with the fix.

Prefill on x86 is still 6x to 10x and is the honest remaining gap
there.

**The Apple CPU rows were withdrawn on 2026-09-04, not improved away.** All
13 of them recorded `backend_active: "Metal"` inside their own
receipts while being published under a CPU heading: `--n-gpu-layers 0`
did not force CPU, and nothing compared the label to what ran
([#126](https://github.com/antonellof/ferrox/issues/126), fixed). The
old "1.41× to 5.06×" range described Metal runs. Honest CPU rows need
a quiet host, and this project's laptop currently has a system daemon
holding a core.

**CUDA has rows for the first time.** The code carried the comment
"UNRUN ON HARDWARE" until 2026-09-04; the first run found that
`Cuda::gemm_supported` covered only `Q8_0` and `Q4_0`, so a K-quant
prefill decomposed into one matvec launch per position. Llama-3.2-3B
Q4_K_M managed **4.88 tok/s** of `pp512` against llama.cpp's 1586.80, a
**325×** gap, on the most common quantization in circulation.

The K-quant GEMM landed the same day and the rows below are taken with
it, on the same GPU:

| model | pp512 before | pp512 after | gap before | gap after |
|---|---|---|---|---|
| Gemma-2-2B Q4_K_M | 5.73 | **181.60** | 369× | **11.61×** |
| Llama-3.2-3B Q4_K_M | 4.88 | **144.59** | 325× | **10.91×** |
| Llama-3.2-1B Q4_K_M | 15.24 | **384.82** | 284× | **11.22×** |
| Llama-3.2-1B Q6_K | 13.96 | **368.66** | 280× | **10.84×** |

Correctness was checked before speed: `ferrox verify --backend cuda
--prompt-tokens 512` is token-identical to the CPU reference for Q4_K,
Q5_K and Q6_K.

**The CUDA gap gets WORSE on newer hardware.** Re-measured on an
RTX 3060 (Ampere, compute 8.6), ferrox and llama.cpp both built with
CUDA on the same box, quiet host:

| model | test | ferrox | llama.cpp | gap |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | pp512 | 185.33 | 10277.19 | **55.5×** |
| Llama-3.2-1B Q4_K_M | tg128 | 24.50 | 282.17 | 11.5× |
| Llama-3.2-3B Q4_K_M | pp512 | 75.58 | 4276.84 | **56.6×** |
| Llama-3.2-3B Q4_K_M | tg128 | 10.38 | 127.03 | 12.2× |

Against the Pascal rows above, prefill goes from ~11× to ~56×.
**llama.cpp is 2.4× faster on Ampere than on Pascal (4318 to 10277
tok/s on the 1B); ferrox does not scale at all.** Decode is roughly
unchanged. So the remaining prefill gap is not a constant factor: it
widens with GPU generation, which means the kernel leaves newer
hardware unused.

Thread count is not the explanation (`-t 4` against ferrox's chosen
`-t 1` gives 232.73 against 186.83, still 44× off), and the GPU is
about half idle during prefill (sampled 0%, 57%, 50%), a different
signature from decode's ~90%.

These four rows are **prose, not receipts.** They come from
`ferrox bench -m --compare` rather than `--suite`, so nothing was
written to `receipts/engine/` and they are absent from the generated
table below. Stated here rather than omitted, and marked rather than
mixed in.

**What remains is one number, not a list.** Every CUDA row is now
between 10.8× and 17.4× on prefill and 9.2× and 17.1× on decode,
across every kind. A uniform band is a systemic per-token cost rather
than a set of missing kernels, which is a different investigation
([#131](https://github.com/antonellof/ferrox/issues/131), and
`docs/plans/cpu-cuda-parity.md` step 2).

**A second host, measured 2026-09-04.** One rented 20-core Cortex-A725
(aarch64, ARMv9.2 with `i8mm`), idle, ferrox and llama.cpp built and run
on the same box, same GGUFs, CPU only. It says something this table
cannot:

| model | test | ferrox (default) | ferrox `spin` | llama.cpp | best gap |
|---|---|---|---|---|---|
| SmolLM2-135M Q8_0 | pp512 | 648.84 | | 894.21 | 1.38× |
| SmolLM2-135M Q8_0 | tg128 | 14.73 | 9.16 | 120.56 | **8.2×** |
| Llama-3.2-3B Q4_K_M | pp512 | 132.53 | | 46.40 | **0.35×** |
| Llama-3.2-3B Q4_K_M | tg128 | 10.38 | **23.14** | 17.86 | **0.77×** |
| Llama-3.1-8B Q4_K_M | pp512 | 61.17 | | 19.15 | **0.31×** |
| Llama-3.1-8B Q4_K_M | tg128 | 6.64 | **12.41** | 9.06 | **0.73×** |

Three things follow, and none of them are visible in the M2 Pro table
above:

1. **Prefill on server aarch64 is not a gap, it is a lead.** 0.35× and
   0.31× mean ferrox is roughly 3x FASTER than llama.cpp at 3B and 8B.
2. **Decode is only red with the default thread pool.** With
   `FERROX_CPU_POOL=spin` ferrox beats llama.cpp at both 3B and 8B.
   The switch is worth more than every other CPU item on the roadmap
   combined on this hardware.
3. **135M is a different problem.** ferrox holds 13 to 15 tok/s at 4, 8
   and 19 threads while llama.cpp does 190 to 204. It is flat in thread
   count and unmoved by either pool, so it is a fixed per-token cost of
   roughly 60 ms, not a scheduling loss
   ([#128](https://github.com/antonellof/ferrox/issues/128)).

**Read every CPU row with two caveats, both found on 2026-09-04.**

1. **This table is one laptop.** Every row is Host B, an Apple M2 Pro.
   There is no x86 row, and a spot check on a rented 10-core Xeon put
   Llama-3.2-3B Q4_K_M at **1.03 tok/s** `tg128` , an order of
   magnitude below what this table's aarch64 rows would lead you to
   expect. The `1.41×–5.06×` range above is an **aarch64** range;
   ferrox's x86 gap is unmeasured and looks far worse
   ([#127](https://github.com/antonellof/ferrox/issues/127)).
2. **The mislabelled CPU rows are gone.** They are not corrected,
   they are deleted: a receipt that says `cpu` and records `Metal`
   cannot be repaired after the fact. #126 is fixed, so a future
   receipt whose label disagrees with the backend that ran is refused
   at write time rather than published.

**And the host has to be genuinely quiet, not just idle-looking.**
Every measurement in this session was taken while `suggestd` held ~97%
of one core for over a day. The `--max-load` guard passed throughout,
because a single pegged core on a 6-core box keeps the load average
under 2.0. A guard that reads load cannot see one busy core, so check
`ps` as well.

**Gap colors (GitHub-safe):** 🟢 ferrox better; ⚪ near-parity (within ~5%);
🔴 ferrox meaningfully slower.

Keep off (regressions): legacy GQA NSG=4, sequential GREEDY argmax,
float4 elem, early Multi-CB. `FERROX_METAL_FA_VEC=0` → ~25.5 pred.

<!-- BEGIN ENGINE TABLE (generated by `ferrox bench --render`) -->

## Engine (`ferrox bench` vs `llama-bench`)

Measured on **3 hosts**, one section each. Rows are never compared across machines.

No HTTP, no chat template, no tokenizer, no sampler. This is the engine
alone. `pp512` is batched prefill, `tg128` is decode. **Neither engine's
thread count is forced**: each picks its own default, because llama.cpp
defaults to performance cores and loses 2–4× when pushed above them, so
pinning both to the same count does not make the comparison fairer.

**Gap** = `llama / ferrox` (<1 ferrox faster). Rows are grouped by
backend (Metal → CUDA → CPU), then test (`pp` then `tg`), then **worst
gap first**. Regenerate with `ferrox bench --suite` / `--render`.

**Largest engine prefill gaps (pp\*, gap > 1.05×):**

- `SmolLM2-135M-Instruct Q8_0` / cuda / pp512: 🔴 **17.41×**
- `TinyLlama-1.1B-Chat-v1.0 Q8_0` / cuda / pp512: 🔴 **15.00×**
- `Qwen2.5-0.5B-Instruct Q8_0` / cuda / pp512: 🔴 **14.69×**
- `Qwen3-0.6B Q8_0` / cuda / pp512: 🔴 **14.04×**
- `Gemma-3-1B-IT Q8_0` / cuda / pp512: 🔴 **12.44×**
- `Llama-3.2-1B-Instruct Q5_K_M` / cuda / pp512: 🔴 **11.89×**
- `Gemma-2-2B-IT Q4_K_M` / cuda / pp512: 🔴 **11.61×**
- `Llama-3.2-1B-Instruct Q4_K_M` / cuda / pp512: 🔴 **11.22×**

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

## Open

1. **Dense Metal prefill is closed.** Every dense `pp512` row is
   between 0.98× and 1.10×, and TinyLlama is ahead at 0.98×. The
   simdgroup-MMA flash attention at d=64 and d=128 did this.
2. **Metal MoE prefill closed too.** OLMoE `pp512` is 1.09× and its
   `tg128` is 1.00×. An earlier ledger put it at 2.62×, measured before
   warmup existed, so the two numbers are not describing the same
   experiment.
3. **Metal decode is ahead of llama.cpp on 8 of 12 rows**, from 0.93×
   down to 0.62×. The rest are within 3% of parity. No red rows left.
4. **CPU is the entire remaining gap.** All 16 comparable CPU rows are
   red: prefill 1.41× to 5.06×, decode 1.68× to 3.55×, nothing at
   parity. The measured cause is fork-join *scaling* rather than
   per-thread throughput, since ferrox beats llama at `-t 1` on
   Mistral-7B, and llama runs a persistent spin-barrier pool. See
   `docs/plans/llama-cpp-parity-push.md`.
5. **Do not compare this table to any earlier one.** Every receipt here
   is 0.12.0, measured in one session, with a warmup rep and the host
   load recorded at both ends of each run. Receipts before 0.12.0 had
   no warmup, so their prefill numbers include cold mmap page faults:
   llama.cpp's own SmolLM2 CPU `pp512` reads 1957 in the old ledger and
   12196 here, on the same binary and the same file. The methodology
   changed, not the engine.
6. CUDA has no in-tree receipt and is skipped on darwin via
   `--fit-host`.
7. Gemma-4-E2B: `ferrox bench` uses `Gemma4Engine` (sequential
   `forward_token` for pp* until batched prefill lands). SPM `gemma4`
   BPE and `<|turn>` chat wrap landed. Homebrew `llama-bench` still
   lacks the `gemma4` arch, so the llama column is blank.
8. DS4 / GLM / MLA MoE real-checkpoint e2e when feasible. Mixtral skipped by `--fit-host` on Host B.
9. Run-to-run spread on this host is ~20%; claims tighter than that need interleaved A/B (still sequential per engine).
