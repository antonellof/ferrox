# Benchmark history and method

The live comparison is [`RESULTS.md`](RESULTS.md), which is generated
from receipts and holds nothing else. This file holds what a generator
cannot: measurements taken without a receipt, before/after studies
against ferrox itself, and the traps that have put a wrong number in
this directory.

## Sections that predate the code

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

**The 8.2x on the 135M row is known to be stale, and is left because
nothing has re-measured that host.** Two fixes landed after it. #155
removed a per-matvec repack that fired only on Q8_0 and Q4_0, which is
what that row is, and was 89% to 90% of decode work.
[#167](https://github.com/antonellof/ferrox/pull/167) then collapsed
about 150 cold rayon entries per token into one, worth +29% at 135M as
an interleaved within-process ratio. On an M2 Pro after both, 135M reads
roughly **1.9x** rather than 8.2x. That figure is a different machine
from the A725 in this table and is not a replacement for it: this row
needs a quiet Cortex-A725 to be re-stated honestly, and until then the
number above should be read as an upper bound on a gap that is known to
have shrunk.

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

**2026-09-11: the table above is wrong, and the reason is worth more
than the correction.** "GPU ms/tok" was the GPU time of ONE command
buffer per token. A sampled decode token has TWO: the dense stack, and
the lm_head in `launch_matvec_fused`, which had no timing tag. Its GPU
time was therefore booked as host time. `FERROX_METAL_GPU_TIMING=1` now
clocks every submission's encode phase, GPU phase and submit latency,
and tags the lm_head. Per token, window means from one process (GPU
columns load-immune; the host was not quiet, so the wall-derived ones
are upper bounds):

| Model | stack GPU | lm_head GPU | encode (both) | submit latency (both) | CPU between |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B Q4_K_M | 4.97-5.08 | **1.17-1.28** | 0.15-0.24 | 0.41-0.46 | ~0.06 |
| Gemma-2-2B Q4_K_M | 12.03-12.44 | **2.78-2.91** | 0.42-0.45 | 0.81-0.88 | ~0.89 |

So the "host" share was mostly the lm_head running on the GPU, and the
part that IS the host divides three ways. **Encoding**, all ~2400
argument-binding calls and ~420 dispatches and barriers, is **0.15-0.24 ms
on Llama-3.2-1B and 0.42-0.45 ms on Gemma-2-2B, 2-3% of wall** -- the
third independent measurement to say so, after the 0.61 µs injection
slope (515 × 0.61 = 0.31 ms) and #156's removal of 48 ops for 0.04 ms.
Argument packing and encode/execute pipelining are each bounded by that
number, so neither was built. **Submit latency**, commit-to-GPU-start
plus completion-to-wakeup, is ~0.2 ms per command buffer and there are
two per token; folding the lm_head into the stack's buffer for sampled
decode (the greedy fold already does) is the one host lever left, worth
at most ~0.25 ms, 4% on the 1B. **CPU work between buffers** is Gemma's
`final_logit_softcap`: a scalar `tanh` over 256k logits, 0.65 ms, more
than that model's whole encode phase.

And the GPU column changes the ranking: Gemma-2-2B's GPU time alone,
12.03 + 2.78 = 14.8 ms, already exceeds llama.cpp's 14.74 ms token. The
worst Metal row is a kernel gap after all, not a host one.

## Open

| Issue | Gap | What is known |
|---|---|---|
| [#133](https://github.com/antonellof/ferrox/issues/133) | CUDA prefill, 22× to 34× | ~4× is tensor cores (`mul_mm` has none), ~5× is undiagnosed kernel efficiency. #148 bought 20–26% and ruled out dequant redundancy and occupancy |
| [#133](https://github.com/antonellof/ferrox/issues/133) | CUDA decode, 2.2× to 5.0× | memory-bound: 17–22% of card bandwidth against llama.cpp's ~60%. Coalescing closed 9–19× to 2–5×. What limits the rest is not diagnosed — the access pattern was a real cost and was not the last one |
| [#149](https://github.com/antonellof/ferrox/issues/149) | Metal decode, 1.11× worst row | the "26% host" was an accounting error: the lm_head runs in a second, untimed command buffer, and its GPU time was booked as host. Encoding, argument binding included, is ~2% of wall (three measurements agree), so packing and pipelining are retired unbuilt. What is left on the host is ~0.2 ms of submit latency per command buffer (two per sampled token) and Gemma's 0.65 ms CPU softcap. Gemma-2's GPU time alone already exceeds llama.cpp's token, so the row is a kernel gap |
| [#127](https://github.com/antonellof/ferrox/issues/127) | x86 CPU prefill, 6.3× to 10.1× | was a missing kernel tier. [#159](https://github.com/antonellof/ferrox/pull/159) added AVX2 GEMMs for all five interleaved kinds and a per-workload dispatch rule, verified by execution on real AVX2 but **not yet benchmarked**, so this gap number still describes the code before it |
| [#27](https://github.com/antonellof/ferrox/issues/27) | CPU decode default | the size rule landed in [#155](https://github.com/antonellof/ferrox/pull/155); the crossover constant is bracketed by the published numbers, not swept, and no before/after on a quiet host has been run. `MIN_TASK_MACS` is still there, which the issue asks to delete |
| [#128](https://github.com/antonellof/ferrox/issues/128) | CPU decode dispatch, **closed** | The condvar wait was real and the cause was rayon's two-armed `join`: from a non-worker thread it injects and blocks on a mutex, ~150 times per token. [#167](https://github.com/antonellof/ferrox/pull/167) runs a whole forward in one `rayon::scope`. Note the trap: #128 had computed scheduling at 6.7% of a token and ruled it out, against a **stale denominator** taken before #155 removed the repack that inflated the token to 17 ms. At ~5 ms the same fixed cost is a much larger share |

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

- **`ferrox bench` does not use the Metal greedy argmax fold, so the
  fold's eligibility does not move these rows.** Worth stating because
  [#172](https://github.com/antonellof/ferrox/pull/172) stopped the fold
  firing in `ferrox run`'s default configuration, which looks like it
  should have changed the `tg` numbers and did not. The bench path asks
  the engine for full logits and argmaxes them on the host
  (`bench_guard::greedy_pick`), because a row that cannot show which
  token it produced cannot show it computed anything. The fold is opt-in
  through `set_metal_greedy_argmax`, and its thread-local setting
  defaults to unset. It has **two** non-test callers, `ferrox-cli`'s
  `run.rs` and `ferrox-server`'s `generate.rs`, and neither is in the
  bench path. An earlier version of this note said only `ferrox run`
  calls it, which was wrong: the conclusion survives because what matters
  is that no caller is on the bench path, not how many callers exist.
  The two also differ in what the fix changed for them, which is worth
  knowing: `ferrox run` defaults `--repeat-penalty` to 1.1 so its greedy
  Metal path stops folding, while the server defaults
  `repetition_penalty` to 1.0 so its greedy requests still fold unless a
  client sends a penalty.

Do not compare this file to a pre-0.13 version: those receipts had no
warmup, so their prefill numbers include cold mmap page faults.

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
