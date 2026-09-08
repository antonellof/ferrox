# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Every crate in the workspace shares one version and is published
together, so a version number describes the whole engine, not a
single crate. `ferrox` is pre-1.0: minor versions may change
behaviour, and a refusal that becomes a supported path counts as a
feature rather than a break.

Entries name what changed and, where it matters, what was wrong
before. A fix that closed a silent-wrong-answer class says so — those
are the ones worth reading twice.

## [0.17.1] - 2026-09-04 

### Fixed

- **A character split across two tokens was destroyed.** The decode
  loop resolved UTF-8 one token at a time, so any character whose
  encoding spanned a token boundary became two U+FFFD before a caller
  saw the bytes. A DeepSeek answer ending in an emoji rendered as
  `today? \u{fffd}\u{fffd}`; the same applies to CJK text and every
  byte-fallback token. Both the streamed and the buffered paths, and
  each batched row now buffers its own tail (#124).

### Documented

- `FERROX_CPU_POOL` is measured rather than "unmeasured": on a quiet
  20-core aarch64 host the persistent pool is **+123% at 3B and +87% at
  8B**, which takes decode past llama.cpp (23.14 vs 17.86, 12.41 vs
  9.06). It stays opt-in because at 135M it is 37% slower,
  reproducibly and on quiet hosts (#27).
- `benchmarks/RESULTS.md` gains a second host and three caveats the
  single-laptop table could not show: prefill on server aarch64 is
  ~3x FASTER than llama.cpp; the published `1.41x to 5.06x` CPU range
  is aarch64-only and x86 looks far worse (#127); and the `cpu` rows
  may include Metal, because `--n-gpu-layers 0` does not force CPU
  (#126).
- `benchmarks/README.md` records the discipline lesson that cost this
  session a day of numbers: **a load average cannot see one busy
  core.**

### Added

- The batch GEMM tests now run through the int-dot kernels they gate,
  under both `ForceIntDot` settings, plus sub-tile shapes that bypass
  the repack path entirely.

## [0.17.0] - 2026-09-04

### Added

- **Seven more architectures run with evidence**, each with a
  libllama-golden fixture: `internlm2`, `xverse`, `ernie4_5`,
  `baichuan`, `exaone`, `bailingmoe2` (Ling-2.0) and `plamo3`. The
  audited count moves 16 to 23; unaudited refusals 41 to 34.
- `ferrox perplexity` — the quality axis nothing in the repo measured.
- `ferrox quantize` writes **Q4_K_S and Q4_K_M**, with a sub-block
  probe for llama.cpp encoder parity. Byte-identity is documented as
  the wrong bar for a K-quant; perplexity is the bar it is held to.
- A caller-supplied **sampler order** is honoured, and the samplers
  ferrox lacks are refused by name rather than ignored.
- **Sliding-window KV eviction** (`FERROX_KV_WINDOW`, off by default):
  a windowed layer's CPU cache drops rows behind its window. Gemma-3-4B
  at 32k context falls from 9.13 GiB to 1.69 GiB.
- `ferrox parity` gained a repeatable `--dumper` and `--dump-logits`,
  so a verdict can be taken against more than one reference build.
- The rerank route runs the GGUF's **pooler** when the file carries one,
  and reports which scale a score is on.

### Fixed

- **The parity oracle's WRONG line was a property of the reference
  build, not of ferrox.** It is now measured per checkpoint as
  `max(KL_WRONG, spread)` against the *nearest* reference. Three tuned
  constants were deleted and none added; no threshold moved. With one
  reference, a Q8_K-dotted checkpoint gets no WRONG line at all rather
  than a guessed one.
- Phi-4 applied LongRoPE context **after** the decode load, so the
  parity run measured a model configured differently from the one that
  answered.
- Metal Q5_0 MoE prefill on Qwen1.5-MoE mixed quant planes.
- `chatglm` was triaged FIXTURE-AWAY; it needs the fused QKV bias, and
  the triage now says so.
- The KV budget takes residency from the store instead of restating the
  rule — the repo's dominant bug shape, removed at one more site.
- **A batched row reported token counts and no rates at all.**
  Continuous batching built its `Usage` without timings, so every rate
  and duration came back null — and batching is the default on Metal,
  so Ferrox Studio's tok/s and duration columns were blank for every
  answer a Mac produced (#116).
- **Ferrox Studio dropped `reasoning_content`.** The server streams a
  reasoning model's thinking correctly; the client read only `content`,
  so an R1 distill looked like a dead stream and an answer that spent
  its whole budget thinking rendered as an empty message under a stat
  line reporting 512 decoded tokens. Thinking is now shown collapsed
  above the answer, and still never replayed as context (#118).

## [0.16.0] - 2026-09-02

### Added

- `-hf user/repo:QUANT`, llama.cpp's one-command model fetch, and `-d`
  to load a draft model so speculative decoding runs on a real
  checkpoint pair.
- `ferrox quantize` writes Q8_0 (byte-identical to llama.cpp, 272/272
  tensors) and refuses every other target by name.
- A forced tool call in eight of eleven wire formats.
- A persistent CPU worker pool behind `FERROX_CPU_POOL`.
- The llama.cpp server flags a copied command line actually carries.

### Fixed

- **`/v1/rerank` shipped broken**: it could not load a reranker at all,
  and ranked the answering document last.
- **Sampling penalties never saw the prompt.** Five call sites gave four
  different answers about the penalty window; one type decides it now,
  and the HTTP API penalises what llama-server penalises.
- `max_tokens` from an HTTP body could reach
  `Vec::with_capacity(usize::MAX)` from a single unauthenticated POST.
- A partial answer had a spelling that reached the response cache.
- The cache key is built from an **exhaustive destructure** of
  `GenerationParams`, so a new field cannot be silently dropped.
- Gemma-3 sliding-window layers roped unscaled; Gemma 27B takes
  llama.cpp's attention scale; the KV budget priced a window cap no
  store implements.
- A short `tokenizer.ggml.scores` array loaded and then panicked once
  per request.

## [0.15.3] - 2026-09-02

- Five one-match-arm architectures now run, and a dead gate is no longer
  mistaken for coverage.
- The batched prefill re-ran the whole prompt on top of the prefix it
  had just adopted.
- Host K/V stays authoritative for Metal continuous-batching prefill.

## [0.15.2] - 2026-09-02

- A second GGUF can be the drafter, which is what makes speculation
  worth running.
- Incremental token streaming under continuous batching.
- The reranker head gets a route; `response_format: json_schema` is
  served rather than refused.

## [0.15.1] - 2026-09-02

- **Release plumbing.** `ferrox-vulkan` must be publishable, because
  `ferrox-core` depends on it — this is what left 0.15.0 half-published.
  The dry run's "blocked by ordering" detector matched only one of the
  two ways Cargo says it, which is the same defect twice.
- The startup banner promised a KV dtype the run does not keep.

## [0.15.0] - 2026-09-02

### Added

- **BGE embeddings end to end**, checked against llama.cpp with a
  calibrated threshold; an encoder-only checkpoint can be the loaded
  model; the reranker classification head, ahead of its route.
- **Vulkan is a third backend**, and the registry list is generated
  rather than hand-kept.

### Fixed

- GGUF bounds allocations sized by untrusted header counts, and bounds
  array length and nesting depth (#25).
- The repack cache served a dead mapping's bytes, because a bool cannot
  say "still alive".

## [0.14.0] - 2026-09-01

### Added

- **Grammar-constrained decoding**: a GBNF engine ported from llama.cpp
  (parser and stack machine), JSON Schema to GBNF with everything
  unported refused by name, lazy grammars, `--grammar`,
  `--grammar-file`, `--json-schema`, and `tool_choice` required/named.
- **WordPiece tokenizer**, byte-exact against llama.cpp on a real BGE
  checkpoint — and putting it in the oracle showed the reference was the
  thing that was wrong.
- llama.cpp's native `POST /completion`, with four copies of the decode
  setup collapsed to one.
- A batched quantized CUDA GEMM, reached from batched prefill, closing
  the last silent CPU fallback.
- All 47 unaudited architectures **triaged**: the refusal now says which
  of three things is missing.

### Fixed

- `logit_bias` and JSON mode were both silently dropped, in four places
  between them; `/v1/completions` dropped four sampler fields.
- The attention-softcap refusal gated on a GGUF key no converter writes,
  so it could never fire — a gate that cannot fire reads as coverage.
- Metal Q5_0 decode ran on the CPU while its prefill ran on the GPU.
- The cross-target gate could not see Linux, which is what broke two
  releases.

## [0.13.3] - 2026-08-28

Chat-template and tokenizer correctness: Yi leaked its turn marker as
text, R1 distills lost their reasoning, a base model answered correctly
and then talked to itself for 512 tokens, one hardcoded regex was
pre-tokenizing every BPE checkpoint, and olmo's pre-tokenizer ends where
gpt2's does not. **The generic architecture path became opt-in**, because
a guess that loads is worse than a refusal.

## [0.13.0] - 2026-08-28

- **Four checkpoints loaded clean and computed the wrong thing.**
- The Metal MoE decode stack ignored four features its dense twin
  implements; Metal paged prefill kept the KV and handed back zeros, so
  two caches read a prompt the model never saw.
- The radix cache never gave a page back, so the pool drained until
  admission refused.
- `ferrox download`, so fetching a model needs no Python.
- CPU: swiglu/geglu spent a libm call per element; the i8mm feature
  probe ran 131k times per GEMM.

## [0.12.0] - 2026-08-27

Serving and bench hardening: two routes never matched, the paged-KV
guard missed the common way to ask for a GPU, the decode guard refused
every GPU run, and a suite that measured nothing could still republish
the ledger.

## [0.11.0] - 2026-08-24

- `ferrox serve` behind an optional, default-off `serve` feature.
- A compile-time assertion that backend features reach the server.
- 0.11.1 fixed the publish order and built the shipped binary with
  `serve`.

## [0.10.0] - 2026-08-24

- **Speculative decoding**: lossless verification, a `Drafter` trait,
  warm-cache resume, and acceptance metrics on `usage` and
  `/admin/stats`.
- Opt-in **resumable SSE streams** with a replay buffer and a JSON
  polling fallback, consumed by the UI.
- The bench asserts the engine *answered* the same, not just that it was
  asked the same.

## [0.9.0] - 2026-08-21

- **24 architectures rotated the wrong RoPE pairs**, plus MoE routing
  bias — the largest single correctness fix in the project.
- Stop sequences in two layers shared by both decode paths; stopping on
  the whole EOG set rather than `eos_token_id` alone.
- Batched requests admitted on an integer KV block budget and cancellable
  at a step boundary.
- Ferrox Studio rebuilt on React, Tailwind and assistant-ui.

## [0.8.0] - 2026-08-20

- **Published to crates.io** for the first time.
- A disk tier for KV prefix-cache blocks, read asynchronously and ahead
  of the request.
- The GGUF's own Jinja chat template is evaluated instead of sniffed.
- Two-tier cancellation for streamed generations.

## [0.7.0] - 2026-08-20

- `ferrox parity` — first-token distribution against llama.cpp. The
  oracle this project is now held to.
- The gpt-oss CPU graph, checked against llama.cpp.
- Ferrox Studio, three-state `/health`, `/admin`, request ids and
  per-phase usage timings, resumable chunked prefill.
- Exact pre-load KV budget arithmetic and a real per-backend device
  memory budget.

## [0.6.0] - 2026-08-18

- IQ2_XS / IQ2_S / IQ3_S / IQ1_M decode and mmap-resident load.
- Refuse checkpoints whose tensors this build never reads.
- Swappable active model and the `/admin` control surface.
- MoE layers run inside the fused Metal prefill stack.

## [0.5.0] - 2026-08-13

- F16 tensor loading; `ferrox verify --prompt` reaches prefill kernels;
  the `clippy -D warnings` gate restored on both feature sets.

## [0.4.0] - 2026-08-11

CPU quantization throughput: i8mm SMMLA tiers for Q8_0/Q4_0,
interleave-8 NEON kernels for Q4_K/Q5_K/Q6_K, NEON DotProd GEMV/GEMM for
Q6_K, one activation-quant pass shared across q/k/v and gate/up. 0.4.1
added simdgroup-MMA flash attention at d=128 and d=64, `ferrox verify`,
and a sealed kernel-lookup registry so a missing kernel is loud.

## [0.3.0] - 2026-08-10

Metal prefill rewritten around llama.cpp's `mul_mm`: a real simdgroup
GEMM for Q4_K extended to every quant kind, FA-vec prefill at d=64/96,
a batched dense FFN (4x on Metal pp512), and pooled scratch buffers.
`ferrox bench -m`, a `llama-bench` work-alike, landed here. Two silent
fallbacks were closed: batched prefill never touched the GPU, and
`--features cuda` never enabled CUDA in `ferrox-core`.

## [0.2.0] - 2026-08-06

Q8_0 KV cache in the Metal backend, a CUDA backend, and the first
benchmark ledger.

## [0.1.0] - 2026-08-05

First tag. GGUF mmap loader, quantized CPU kernels, Metal backend,
`ferrox` CLI and `ferrox-server`.

[0.17.1]: https://github.com/antonellof/ferrox/compare/v0.17.0...v0.17.1
[0.17.0]: https://github.com/antonellof/ferrox/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/antonellof/ferrox/compare/v0.15.3...v0.16.0
[0.15.3]: https://github.com/antonellof/ferrox/compare/v0.15.2...v0.15.3
[0.15.2]: https://github.com/antonellof/ferrox/compare/v0.15.1...v0.15.2
[0.15.1]: https://github.com/antonellof/ferrox/compare/v0.15.0...v0.15.1
[0.15.0]: https://github.com/antonellof/ferrox/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/antonellof/ferrox/compare/v0.13.3...v0.14.0
[0.13.3]: https://github.com/antonellof/ferrox/compare/v0.13.0...v0.13.3
[0.13.0]: https://github.com/antonellof/ferrox/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/antonellof/ferrox/compare/v0.11.1...v0.12.0
[0.11.0]: https://github.com/antonellof/ferrox/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/antonellof/ferrox/compare/v0.9.1...v0.10.0
[0.9.0]: https://github.com/antonellof/ferrox/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/antonellof/ferrox/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/antonellof/ferrox/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/antonellof/ferrox/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/antonellof/ferrox/compare/v0.4.1...v0.5.0
[0.4.0]: https://github.com/antonellof/ferrox/compare/v0.3.1...v0.4.0
[0.3.0]: https://github.com/antonellof/ferrox/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/antonellof/ferrox/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/antonellof/ferrox/releases/tag/v0.1.0
