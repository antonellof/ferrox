# CPU and CUDA parity with llama.cpp

Metal is done. This plan is about the two backends that are not, and it
is written from measurements taken on 2026-09-04 on rented, dedicated
hosts rather than from the ledger, because the ledger could not see any
of it: it had no x86 row, no CUDA row, and its CPU rows were Metal runs
mislabelled (#126).

Read `north-star.md` first. This plan is subordinate to it: same
models, same command shapes, same or better performance, on the
hardware people actually own. Most of them own x86 with an NVIDIA card.

## What parity means here, precisely

Three claims, in order. A step that improves one while breaking a
higher one is a regression, not progress.

1. **It answers the same.** Token-identical greedy output, or a KL
   within the reference spread. `ferrox parity` is the oracle.
2. **It runs at all.** The kind has a kernel on that backend. Falling
   back to the host is not running: it is a different program with the
   same output.
3. **It is not slower.** `gap = llama / ferrox` at or below 1.0 on the
   same host, same GGUF, same backend, quiet host.

Claim 2 is where CUDA fails today, and it is invisible in a gap column
because a fallback still produces numbers.

## Measured state, 2026-09-04

### CUDA (GTX 1080, CUDA 12.4, llama.cpp built with CUDA on the same box)

First execution of this code path on a GPU. `Cuda::gemm_supported`
carried the comment "UNRUN ON HARDWARE" until this run.

| model | pp512 gap | tg128 gap |
|---|---|---|
| gemma-2-2b Q4_K_M | **369x** | 19.3x |
| Llama-3.2-3B Q4_K_M | **325x** | 17.5x |
| Llama-3.2-1B Q4_K_M | 284x | 15.1x |
| Llama-3.2-1B Q6_K | 280x | 13.0x |
| SmolLM2-135M Q8_0 | 17.2x | 9.2x |
| Qwen3-0.6B Q8_0 | 15.4x | 11.3x |
| TinyLlama-1.1B Q8_0 | 15.2x | 10.7x |
| Qwen2.5-0.5B Q8_0 | 14.9x | 12.8x |
| gemma-3-1b Q8_0 | 12.4x | 11.0x |

The K-quant prefill rows are a **missing kernel**, not slowness: there
was no `mul_mm` for them, so a 512-token prefill issued 512 matvec
launches. Fixed in this branch; **unverified on hardware**.

### CPU, aarch64 (20-core Cortex-A725, i8mm, idle)

| model | test | ferrox default | ferrox `spin` | llama.cpp |
|---|---|---|---|---|
| 3B Q4_K_M | pp512 | **132.53** | | 46.40 |
| 3B Q4_K_M | tg128 | 10.38 | **23.14** | 17.86 |
| 8B Q4_K_M | pp512 | **61.17** | | 19.15 |
| 8B Q4_K_M | tg128 | 6.64 | **12.41** | 9.06 |

**Prefill is already a 3x lead.** Decode is a loss with the default
thread pool and a win with the persistent one.

### CPU, x86 (10-core Xeon E5-2630 v4, idle)

Llama-3.2-3B Q4_K_M: **1.03 tok/s** tg128. Roughly an order of
magnitude off. Not diagnosed (#127).

### Every backend, small models

SmolLM2-135M decode is 13 to 15 tok/s at 4, 8 and 19 aarch64 threads
while llama.cpp does 190 to 204. Flat in thread count, unmoved by
either pool: a fixed cost of about 60 ms per token (#128).

## Coverage: which kinds have a kernel

ferrox has 21 `QuantKind`s. llama.cpp implements all of them on both
CPU and CUDA. ferrox does not, and this table is the parity gap that a
tok/s column cannot show.

| kind | CPU fast path | CUDA matvec | CUDA GEMM | Metal |
|---|---|---|---|---|
| Q8_0 | yes | yes | yes | yes |
| Q4_0 | yes | yes | yes | yes |
| Q4_K | yes | yes | **new** | yes |
| Q5_K | yes | yes | **new** | yes |
| Q6_K | yes | yes | **new** | yes |
| Q5_0 | no | **new** | **new** | yes |
| IQ4_NL | yes | **new, unverified** | **new, unverified** | **no** |
| IQ4_XS | yes | **new, unverified** | **new, unverified** | matvec+GEMM |
| Q2_K | yes | **new, unverified** | **new, unverified** | **no** |
| Q3_K | yes | **new, unverified** | **new, unverified** | **no** |
| Q4_1, Q5_1, Q8_1 | no | **no** | **no** | **no** |
| IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | IQ1_M only | **no** | **no** | **no** |
| MXFP4 | yes | **new, unverified** | **new, unverified** | **no** |

"no" means the tensor is decoded on the host and the GPU is idle for
that matmul. It still answers correctly, which is why this never
surfaced as a bug.

"new, unverified" means the kernel exists, its arithmetic is checked
against `ferrox_quant` by a Rust twin, and for the GEMM the emitted
CUDA C is executed on a host CPU and compared to that twin bit for bit
(`crates/ferrox-cuda/tools/mul_mm_host_check/run.sh`, 75,042 positions
over eleven kinds and three shapes each, zero mismatches on 2026-09-09).
**No GPU has run it.** `cargo test -p ferrox-cuda --features cuda --
--ignored` on a real device is the exit criterion, plus a bench row.

## The four kinds of gap, and why the distinction matters

Ordering work by tok/s alone puts the 369x row first and gets the
priorities wrong. These are different problems with different fixes.

1. **No kernel.** The backend cannot run the format and silently uses
   another one. Fix: write the kernel. Measurable before and after,
   and the before number is meaningless as a performance signal.
2. **Wrong dispatch.** A kernel exists and is not selected, or is
   selected for the wrong shape. Fix: the predicate. Costs nothing to
   run and is usually a large win. `FERROX_CPU_POOL` is this.
3. **Fixed per-token overhead.** Independent of weights, threads and
   backend. Fix: find the constant. Worth more than any kernel at small
   sizes and worth nothing at large ones.
4. **Genuinely slower arithmetic.** The kernel is right and loses.
   Fix: profile. This is the only class that needs new performance
   work, and it is the class ferrox has the least of.

Today's evidence says ferrox's remaining gap is mostly 1, 2 and 3.
That is good news and it should change the order of work.

## Ordered plan

Each step names its exit criterion. "Measured" always means: quiet
host, `ferrox bench` guard passing, `ps` checked for a single busy
core, receipt committed.

### 1. Verify the K-quant CUDA GEMM on hardware  [blocks everything else on CUDA]

Written and held against `ferrox_quant::dequant_q*_k` sub-block by
sub-block, but the CUDA C is a second transcription of the same
arithmetic and no GPU has run it. Rent one box, run
`ferrox parity` for correctness and `ferrox bench --suite --backend
cuda` for the gap.

**Exit:** Q4_K/Q5_K/Q6_K prefill within the same order of magnitude as
llama.cpp, and `parity` agreeing on tokens. If the kernel is wrong,
this is where it is caught, before any of it is published.

### 2. Close the CUDA decode gap (9x to 17x)

Present on every kind including Q8_0, so it is not the missing GEMM.
Three candidates were tested on hardware on 2026-09-04 and **two are
now ruled out**:

- **The GQA reduction is not it.** `FERROX_CUDA_GQA=1` is correct
  (`verify` is token-identical) and **42% SLOWER**: 6.85 tok/s against
  11.88 on Llama-3.2-1B Q4_K_M. It also never compiled before that day
  (NVRTC has no `INFINITY`), so the flag had never run at all.
- **CUDA graphs are not it, yet.** `FERROX_CUDA_GRAPH=1` measures
  11.80 against 11.84 off, exactly as its own doc predicts: nothing
  enqueues into a captured stream, so it is groundwork.
- **The GPU is NOT idle, and the claim that it was is retracted.** An
  earlier version of this plan said `nvidia-smi` reported 36%
  utilization during decode, and concluded the cost was host-side. That
  number came from a single instantaneous sample taken AFTER a bench
  run had finished, so it caught an idle moment. Sampled five times
  DURING a `tg256` decode on a GTX 1080, `main` sits at **86% to 93%**.
  Decode is **kernel-bound**, and the host-side theory this plan was
  built on for a day was wrong.

  The correction cost a PR (#136, measured 22% SLOWER on hardware:
  8.24 against 10.55 tok/s). The arithmetic that should have caught it
  was available before the code was written: at 36% utilization,
  removing every host round-trip buys at most 1/0.36 = 2.8x against a
  9x to 17x gap. A lever that cannot reach the target is the wrong
  lever even if the diagnosis is right, and here the diagnosis was also
  wrong.

**And prefill is a second, larger problem that widens with hardware.**
Re-measured on an RTX 3060 (Ampere): prefill is **55x to 57x** off
llama.cpp, against ~11x on the GTX 1080. llama.cpp is 2.4x faster on
Ampere than on Pascal; ferrox is not faster at all. Decode is roughly
unchanged at 11.5x to 12.2x. The GPU sits about half idle during
prefill (0%, 57%, 50%) where decode runs it at ~90%, so the two have
different signatures and are probably different bugs. Thread count is
not it: `-t 4` against ferrox's chosen `-t 1` is worth 25% and leaves
44x.

Treat the 50% as a lead needing repeated sampling, not a conclusion.
One instantaneous utilization sample already cost this plan a day and a
PR.

**What llama.cpp's CUDA backend does**, read out of
`.scratch/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`. Kept because it
is true and useful, with the caveat that it is NOT the explanation for
this gap:

- `ggml_backend_cuda_graph_compute` runs a whole token's graph, and its
  node loop contains **zero** `cudaStreamSynchronize` or
  `cudaDeviceSynchronize` calls. Every sync in the file is at the graph
  boundary, in `tensor_set` / `tensor_get` / `cpy_tensor`.
- Every tensor, including every intermediate activation, is allocated
  in a device buffer up front. Residency is a tensor's default state,
  not an optimisation applied to a pair of calls, so the host never
  sees an intermediate.

That is a real architectural difference and it settles the identity
question a residency scheme would face: a tensor's device buffer IS its
identity, for its lifetime, which is stronger than any length or epoch
comparison. But ferrox's decode already runs the GPU at ~90%, so it is
not waiting on the host, and closing this difference would not close
the gap. Recorded so the next reader does not re-derive it and reach
the conclusion this plan already retracted.

**The cause is the kernels.** ferrox's CUDA matvec uses one 256-thread
block per row with a shared-memory tree reduction; ggml-cuda uses
warp-level `dp4a` with no shared-memory round trip. At ~90%
utilization and 9x to 17x off, that is where essentially all of the
difference is. Compare one kernel against its ggml-cuda equivalent for
the same kind and shape, and close the arithmetic.

**A dead API that should still go.** 
`ferrox-cuda/src/gpu.rs` defines `DeviceAct` / `upload_act` /
`matvec_into` / `download_act`, whose doc says they exist "so a
matvec's output can be fed straight into the next matvec without a
DtoH/HtoD round-trip (the exact per-call upload/download overhead that
made CUDA decode bandwidth-starved)". **All four have zero uses outside
that file.** The decode path takes `launch_matvec`, which returns
`Vec<f32>` and therefore ends in `dtoh_sync_copy`: every matmul
uploads, allocates, launches, synchronises and downloads, on the order
of a hundred times per token.

Five DtoH points exist per dense decode layer and chaining can reach
only three of them: norms, RoPE and the attention reduction sit between
the matmuls on the host, so there is no consecutive-matmul pair left
for `matvec_into` to serve. Measured, and the reason the narrow fix
could never have worked even had the diagnosis been right.

**Exit:** tok/s against llama.cpp on the same GPU and model, with
utilization already high on both sides. Not a utilization target.

**The hazard to design around first, and it was real.** Metal's
equivalent used to match on LENGTH alone, which this note called safe
"only because exactly one site sets it and it is cleared aggressively".
It was not safe. There is one publisher but THREE consumers, each
routinely handed a `hidden_dim`-long activation that is not the
published one, and the publication was a thread-local raw pointer into
`DECODE_SCRATCH` -- a process-wide `Mutex` the pointer escaped, so two
concurrent `ferrox-server` requests could have one answer the other's
`lm_head` with its own activation, lengths agreeing by construction.
Fixed in `ferrox-metal/src/resident_act.rs` (issue #166): the
publication lives inside the thing the mutex protects, records the host
address and length of the exact vector the stack returned, is dropped by
any borrow of the buffer it describes, and holds the guard while the
buffer is bound. Read that module before writing the CUDA twin.
Whatever carries residency needs an identity the caller cannot get
wrong, not a length comparison.

### 3. Decide the CPU pool by work size, not by environment variable

`FERROX_CPU_POOL=spin` is +123% at 3B and +87% at 8B on aarch64 and
takes decode PAST llama.cpp. It is -37% at 135M. So the default cannot
flip and cannot stay: it needs a rule.

`MIN_TASK_MACS` already computes the quantity the rule needs. #27
proposes deleting it; the measurements say keep it and use it to select
the pool per operation.

**Exit:** one predicate, shared by every call site, with the crossover
measured on both aarch64 and x86 rather than guessed. `spin` stops
being a user-visible knob.

### 4. Find the 60 ms (#128)  [NAMED 2026-09-10: rayon's cold submit]

**The premise of this step was wrong twice, and both corrections are
recorded because each one cost work.**

The first framing, "a fixed ~60 ms per token, flat in model size", was
corrected by #155: the cost was proportional to gate and up projection
BYTES and fired only on Q8_0 and Q4_0, because the int-dot matvec
repacked the whole weight matrix on every call. That was 89% to 90% of
decode and it is fixed.

The second framing, in a comment on #128, measured 5.48 us per
fork-join region, multiplied by ~210 regions per token, got 6.7% of the
token, and concluded scheduling could not be what remained. **The
arithmetic was right and the denominator was stale.** It was taken
against a 17.23 ms token, i.e. WITH the repack bug still inflating the
work. Once #155 removed that work the token fell to about 5 ms and the
same fixed dispatch became a much larger share of it.

#### What it actually is

`rayon::join` and the `par_iter` bridges both funnel into
`Registry::in_worker`, which has two arms with very different costs.
From a rayon worker: run one half inline, post the other for stealing,
wait on a `SpinLatch`, no syscall. From any other thread: inject the
job and block on a `LockLatch`, which is a pthread mutex and condvar.
Every forward pass was driven from a thread rayon did not own, so it
paid the second arm once per region, roughly five per layer.

Sampled with `sample` on an M2 Pro over SmolLM2-135M Q8_0 `tg128`,
CPU-only, `-t 6`:

| | share of the driving thread's wall time |
|---|---|
| `__psynch_cvwait` under rayon's `LockLatch` | **74%** |
| attention (`causal_gqa_attention_softcap`) | 10% |
| the one matvec that ran inline (`o_proj`) | 5% |

Across all twelve threads in that process, the NEON `q8_0x4` matvec
kernel held 6.6% of the samples and `__psynch_cvwait` 63.7%. During the
74% the driving thread spent asleep, the six workers it was waiting for
held about an eighth as many kernel samples between them: most of the
wait was the round trip, not the work.

#### The fix, and what it measured

`ferrox_core::par::on_workers` wraps a whole forward pass in one
`rayon::scope`, so the step runs on a worker and every nested region
takes the hot arm. `~150` cold entries per token become **one**.
`decoder/entry.rs` is the one place every public `Decoder::forward_*`
does this, and `par::cold_regions` is a per-thread operation counter so
the property is a test rather than a stopwatch.

Interleaved `main, branch, main, branch` on the M2 Pro, CPU only, which
is NOT a benchmark host, so these are ratios and not ledger rows:

| model | decode | prefill |
|---|---|---|
| SmolLM2-135M Q8_0 | **+29%** (4 of 4 rounds) | flat |
| Llama-3.2-3B Q4_K_M | **+9%** (4 of 4 rounds) | flat |
| Llama-3.1-8B Q4_K_M | +3%, swap-bound on this host | not run |

Against `llama-bench` on the same host and file, best-of-3 at `tg64`,
the gap moved from about 1.9x to about 1.5x. The host was too noisy for
that number to be worth more than its order of magnitude; the
main-versus-branch ratio is the reliable half.

Metal and CUDA are deliberately NOT promoted. Moving the step off the
main thread changes Metal's output (`Llama-3.2-3B Q4_K_M --ngl 99`,
greedy, diverges around the tenth token, deterministically on both
sides), because the Metal stack carries thread-local state across a
step. `on_workers` asks `weight_matrix::active_backend` first, and with
that gate the Metal answer is byte-identical to `main` over three runs.

**Exit:** 135M decode within 2x of llama.cpp on a QUIET host. Still
owed: the numbers above are from a laptop, so the ledger row in
`benchmarks/RESULTS.md` has not moved and must be re-measured on the
rented aarch64 box it was taken on.

### 5. x86 CPU  [DONE for decode, 2026-09-04]

The answer was a **default**, not a missing kernel.
`FERROX_CPU_INT_DOT` defaults on, its interleaved integer kernels are
aarch64-only, and on x86 it selected a scalar loop while bypassing the
AVX2 f32 dot that does exist. Cost: 4x to 8.8x of decode. Fixed
architecture-aware; Llama-3.2-1B Q4_K_M went from 6.8x off llama.cpp to
**1.4x**.

Note the shape of the error, because it recurs: the AVX2 arms were
present and correct, and were being SKIPPED. Two of the three
hypotheses in this issue (no x86 SIMD, slow x86 SIMD) were wrong, and
the code read as if they were right.

**What is left on x86:**

- **Prefill is still 6x to 10x.** That is now the biggest CPU gap in
  the ledger and has had no investigation at all.
- **No x86 int8 path exists.** Zen 4 advertises `avx512_vnni` and
  nothing here uses it. That is the natural successor to this fix, and
  `FERROX_CPU_INT_DOT=1` stays available precisely so such a port can
  measure itself against the f32 path.

### 6. Kernel coverage, by what people actually run

Not alphabetically, and not all 21. In order of how often a checkpoint
in the wild uses it:

- **CUDA Q5_0, landed 2026-09-05, UNVERIFIED ON HARDWARE.** Matvec and
  GEMM together, because `a_cuda_kind_with_a_matvec_also_has_a_gemm`
  forbids half of it. The GEMM's emitted C is executed on the host by
  `crates/ferrox-cuda/tools/mul_mm_host_check/run.sh` and matches the
  Rust twin bit for bit; the matvec's C has no such harness and no GPU
  has run either. `cargo test -p ferrox-cuda --features cuda --
  --ignored` is the exit criterion, plus a Q5_0 bench row.
- **CUDA IQ4_NL and IQ4_XS, landed 2026-09-09, UNVERIFIED ON
  HARDWARE.** Matvec and GEMM together. They are codebook formats, so
  `MulMmKind` grew a `codebook: Option<Codebook>` field and
  `kernel_src` emits it as a `__constant__ float[16]` ahead of
  `dequant_src`; the `dequant_twin` seam is unchanged, because its
  contract was always "given the block and `il`, write 16 floats in
  ascending element order" and a table lookup satisfies it exactly as
  an affine transform does. The GEMM's `Codebook` row is ONE slice: the
  emitter formats it into the CUDA and the Rust twin indexes it, so
  there is nothing to drift. The matvec kernels are `&'static str` and
  carry the sixteen values as a literal, which is a second structure --
  `every_embedded_codebook_is_the_mul_mm_codebook` parses them back out
  of the kernel text and holds them to the `Codebook`, bit for bit.
- **MXFP4 on GPU, landed 2026-09-09, UNVERIFIED ON HARDWARE.** gpt-oss
  ships it and no GPU backend had it at all, so every expert decoded on
  the host with the device idle. Matvec and GEMM, the same codebook
  seam, with the E2M1 table and an E8M0 scale helper. 17-byte blocks:
  the only odd stride in the table, so nothing in either kernel may
  assume a block pointer is aligned to anything.
- **CUDA Q2_K and Q3_K, landed 2026-09-09, UNVERIFIED ON HARDWARE.**
  Common in small-memory builds, and they were absent on all three GPU
  backends. Matvec and GEMM, both affine. Q3_K is the fiddly one: its
  third quant bit is a bit plane in `hmask` and it is INVERTED (a set
  bit means bias 0, a clear one bias 4), and its six-bit scales use a
  four-arm packing that is not Q4_K's. Sabotaging that inversion in the
  emitted CUDA alone makes the host check report 3,968 mismatches out
  of 4,096, so the check sees it. Metal still has neither.
- The IQ1/IQ2/IQ3 family last: rare, and each is a separate codebook.

**Exit per kind:** a kernel, a scalar twin, a `parity` run, and a bench
row. A kind without a bench row is unmeasured, not done.

### 7. Make the ledger able to hold the answer

Partly landed: the renderer now groups by host, receipts carry a host
slug, the suite lists `cuda` on 13 entries instead of 1, and a receipt
whose label disagrees with the backend that ran is refused at write
time.

Still owed: an x86 CPU row, an aarch64 CPU row from a quiet host, and a
CUDA row taken after step 1. `RESULTS.md` currently has no CPU rows at
all, which is honest and temporary.

## Traps, each one paid for already

- **A gap column cannot see a missing kernel.** The 369x row looked
  like a performance problem and was an absent GEMM. Check coverage
  before profiling.
- **A load average cannot see one busy core.** Every CPU number in this
  session was first taken while a daemon held 97% of a core, and the
  `--max-load 2.0` guard passed throughout.
- **A backend label is not the backend.** All 13 published CPU receipts
  recorded `backend_active: "Metal"`. Two fields in one file, with
  nothing comparing them.
- **A fallback is not a failure, which is what makes it dangerous.** A
  kind with no kernel still answers correctly, so nothing goes red.
- **One host is not a platform.** The pool wins on x86 at every size
  and loses at 135M on aarch64. A single-machine ledger would have
  published either as universal.
- **Do not benchmark on a laptop with a UI.** Rent a box. The whole
  investigation behind this plan cost under a dollar.
- **One instantaneous sample is not a measurement.** The claim that
  CUDA decode ran at 36% GPU utilization came from a single
  `nvidia-smi` taken after a bench had finished. It sent a day of work
  at a host-side cost that does not exist: the real figure, sampled
  during the run, is 86% to 93%. Sample repeatedly, and sample WHILE
  the thing is running.
- **Check whether the lever can reach the target before pulling it.**
  At the (wrong) 36% figure, removing every host round-trip was worth
  at most 2.8x against a 9x to 17x gap. That arithmetic was in the
  agent's PR body before the code was written, and reading past it cost
  a merged-nothing PR.

## Status

| Step | Issue | State |
|---|---|---|
| 1 CUDA K-quant GEMM verified | #131 | **done**: verify token-identical, 325x to 10.9x |
| 2 CUDA decode | #133 | GQA and graphs ruled out; GPU at 36% util, host-bound |
| 3 CPU pool rule | #27 | measured, needs the predicate |
| 4 fixed per-token cost | #128 | not started |
| 5 x86 decode | #127 | **done**: default was wrong, 6.8x to 1.4x |
| 5b x86 prefill | | not started, now the largest CPU gap (6x to 10x) |
| 6 kernel coverage | | Q4_K/Q5_K/Q6_K landed on CUDA, Q5_0 2026-09-05, Q2_K/Q3_K/IQ4_NL/IQ4_XS/MXFP4 2026-09-09 (all unverified on hardware); 10 kinds still host-only |
| 7 ledger | #126 | **done**: three hosts, and a committed-receipt check |
