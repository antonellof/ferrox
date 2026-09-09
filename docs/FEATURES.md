# Features

Ferrox is a pure-Rust GGUF inference engine for dense and MoE models.
Weights stay quantized when the file is mmapped, and dequantization
happens inside the matvec. Backends: CPU, Apple Metal, and CUDA.

## Models

Measured against llama.cpp on the same host and the same GGUF with
`ferrox bench`. Gap = `llama / ferrox`, so anything under 1 means Ferrox
is faster.

- **Dense GQA**: TinyLlama, Llama 3.2, Mistral-7B, SmolLM2,
  Qwen2.5/Qwen3, Gemma-3/4. Metal decode leads on several small models
  (SmolLM2 **~0.67×**, Qwen2.5-0.5B **~0.70×**, Qwen3-0.6B **~0.71×**,
  Gemma-3-1B **~0.88×**), and Llama-3.2-3B sits at **~0.96×**, ahead of
  llama.cpp rather than behind it. Mistral-7B and Llama-3.2-1B Q4_K_M
  land on 1.00×. Dense Metal prefill is closed: every dense `pp512` row
  is 1.02–1.08×. **CPU decode is still behind everywhere (~1.17–2.44×),
  and CPU prefill is behind on 6 of 8 rows.**
- **Phi-4**: CPU **and** Metal. Partial rotary (`n_rot < head_dim`) and
  LongRoPE `attn_factor` ride the Metal RoPE kernels as `rot_dim` and
  `mscale` uniforms, matching ggml's `rope_yarn` (the magnitude scale
  reaches the rotated channels only). The published Metal speed rows
  predate that fix and need measuring again before you quote them.
- **MoE**: OLMoE-1B-7B, still behind on Metal decode (~1.41×). Metal
  Concurrent plus fused encode groups, `MemRanges`, `mul_mv_id` and
  prefill `mul_mm_id`. On CPU: int-dot and interleaved Q4_K.
- **MLA**: dense-lead and MoE-after-dense `deepseek2` / `mistral4`.
- **Gemma-4**: dedicated engine (per-layer embeddings, shared KV,
  SWA/full), an SPM-style `gemma4` BPE tokenizer, and the `<|turn>` chat
  wrap. Dense only: the MoE router is ported and tested
  (`ferrox_moe::route_gemma4_moe`) but the loader still expects
  `ffn_gate.weight`, so a MoE Gemma-4 GGUF does not load yet.
- **MiniMax**, and the two architectures are not one thing.
  `minimax-m2` builds ordinary dense GQA with whole-vector Q/K norm,
  partial NEOX RoPE and a sigmoid MoE with router bias, every one of
  which the generic path implements: it is **unaudited, not
  unimplemented**, and what it needs is a fixture. `minimax-m3` is
  genuinely unimplemented, and the blocker is MiniMax Sparse Attention:
  a per-layer indexer driving its own KV cache with position-to-cell
  maps, plus `SWIGLU_OAI` and shared experts. The block-sparse block
  selection (`ferrox_core::block_sparse`) is the smallest piece of that
  and is the only piece ported. Neither is blocked on MTP draft heads,
  which no MiniMax GGUF can carry: `gguf-py`'s tensor lists for both
  have no `NEXTN_*` entry, so the writer physically cannot emit one.
- **Also loadable**: yi, qwen2moe / qwen3moe (MiroThinker GGUFs, for
  example), Gemma-2, Phi-3, Llama-3.1, and GLM4 when the tensors are
  there. None of these are in the published suite.
- **MoE routing bias** (`exp_probs_b`, DeepSeek-V3's aux-loss-free
  selection bias) plus `expert_weights_scale` and
  `expert_weights_norm`, on the generic path. That one tensor is what
  used to stop dots1, ernie4_5-moe, bailingmoe2, exaone-moe,
  hunyuan-moe and afmoe from loading at all. It is checked against
  llama.cpp's own dots1 implementation reading the same synthetic
  checkpoint, and has not been validated on a published one.
- **Granite, MiniCPM and Command-R scalar multipliers stop the load.**
  `logit_scale`, `residual_scale`, `embedding_scale` and
  `attention.scale` are hyperparameters, not tensors, so the check for
  unread tensors never sees them. A checkpoint that declares one of them
  with a value that changes the maths stops with an error naming the
  key. `minicpm` stops outright, because llama.cpp applies its three
  multipliers even when the file carries no key at all.
- **Parallel-residual architectures do not load either**: `command-r`,
  `cohere2`, `cohere2moe`, `falcon`, `gptneox`, `phi2`, `plamo`. They
  sum `inpL + attn_out + ffn_out` once instead of taking two sequential
  residuals. That is a different graph, and the tensor list looks
  identical either way, so these are listed by name.

Full matrix: [`MODELS.md`](MODELS.md) ·
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) ·
[`benchmarks/suite.json`](../benchmarks/suite.json).

## Backends

| Backend | Capabilities |
|---|---|
| **CPU** | Dense and MoE. int8×int8 matvec on by default (`FERROX_CPU_INT_DOT=0` opts out), interleaved Q4_Kx8 / Q8_0x4 GEMV, Q8_0x4 batch GEMM for prefill, Q5/Q6 int-dot, pool sized to performance cores |
| **Metal** | FA-vec attention (decode d=64/96/128/256, prefill d=128/256), concurrent FFN/QKV encode, MoE Concurrent with fused groups, `MemRanges`, `mul_mm_id` prefill, quantized KV (`q8_0` / `turbo8` / `fp8` / `turbo4`) |
| **CUDA** | Matvec, resident weights, FFN fuse (`--features cuda`), plus a batched `Q8_0`/`Q4_0` GEMM that **has never executed on a GPU** |
| **Vulkan** | `Q8_0` matvec only, no GEMM (`--features vulkan`). A beachhead, not a backend: see below |

**Vulkan is one kernel, and calling it a backend would be generous.**
`--features vulkan` gives a `Q8_0` matvec and nothing else. It reports
no GEMM for any kind, so a prefill genuinely lands on the host, and it
claims no other quantization. It did run on real hardware (an M2 Pro
through MoltenVK) against a scalar twin, which is what earned it a place
in the dispatch table at all, but there is no measured number for it and
zero-copy residency from mmap is unproven. It exists so that AMD and
Intel have a path at all, and because the seam it needed is the seam a
real backend needs. `docs/plans/vulkan-beachhead-verdict.md` has the
sizing: a full Vulkan backend is 15 to 25k lines.

**The CUDA GEMM is unrun, and that is not a formality.** It is wired
into a wide prefill and it decides nothing about performance here,
because nobody has put it on a card. What it does have is a
thread-by-thread scalar twin held against `ferrox-quant`'s independent
dequantize-then-GEMM, and a host harness that compiles and *executes*
the emitted CUDA C against a barrier shim
(`crates/ferrox-cuda/tools/mul_mm_host_check/run.sh`) with zero
mismatches on both kinds. Its hardware test is `#[ignore]`d with "NEVER
RUN" as the reason, and NVRTC is not clang, so "it compiles here" is not
"NVRTC accepts it". Below the width threshold a single token stays on
the matvec kernels, which are the arm that *has* run on a GPU.

**CUDA compiles and runs, and nobody has benchmarked it.** Every number
in this repo was taken on CPU or Apple Metal. GPU acceleration on
Windows and Linux means CUDA, and the bar CUDA is held to is "must
compile". There is no pinned benchmark host for it and no published
timings. Treat a Windows or Linux install as **CPU-only in practice**
until that changes. `/health` reports the same thing per capability,
with a reason string, instead of quietly greying a control out.

## CLI

llama.cpp-style completion flags (`-m`, `-p`, `-n`, `-ngl`, `--ctk`, …),
plus `ferrox chat`, `ferrox pull` (Hugging Face Hub), `inspect`, `archs`,
and `presets`. See [`CLI.md`](CLI.md).

Constrained decoding is on the CLI too: `--grammar`, `--grammar-file`
and `-j` / `--json-schema`, the same spellings llama.cpp uses, reaching
the same stack machine the HTTP `grammar` field does. `--ctk` selects a
KV dtype on Metal only; the CPU and CUDA KV cache is the host `Vec<f32>`
and the startup banner says so when the flag is being ignored.

`ferrox perplexity` is the quality axis: corpus evaluation using
llama.cpp's method, agreeing with `llama-perplexity` to within a fifth
of one standard error on five checkpoints. Where the two differ, the gap
is monotone in the quant and has the sign the documented `vec_dot_type`
difference predicts. `ferrox quantize` writes `Q8_0` byte-identically to
`llama_model_quantize()`, plus `Q4_K_S` and `Q4_K_M` with `--pure`, and
refuses every other target by name. Q4_K is not byte-identical and
cannot be, because the C reference is compiled with FP contraction; it
matches on perplexity instead. See
[`CLI.md`](CLI.md).

The sampler flags carry llama.cpp's own defaults on `--temp` (0.8),
`--top-k` (40), `--top-p` (0.95), `--min-p` (0.05) and `--repeat-last-n`
(64). **One default still differs on purpose**: `--repeat-penalty` is
1.1 here and 1.0 (off) in llama.cpp
(`common/common.h:239`), so a run left entirely to defaults is not
token-identical. `-e`/`--escape` is on by default as it is there, and a
*partial* `-ngl N` is refused rather than silently offloading every
layer.

`ferrox bench -m model.gguf` works like `llama-bench`: the same `pp512`
and `tg128` workloads, reported as a median with a population stddev.
Add `--compare` to run `llama-bench` alongside it and print the gap.
`--suite` drives every entry in
[`benchmarks/suite.json`](../benchmarks/suite.json) and regenerates
[`RESULTS.md`](../benchmarks/RESULTS.md). See
[`benchmarks/README.md`](../benchmarks/README.md).

## Server

Two ways to start the same server. `ferrox serve` is a subcommand of the
main binary behind an optional `serve` feature, off by default for
`cargo install` because it pulls in 98 crates a completion-only user
does not need. `ferrox-server` is that same server as its own
executable, and both parse identical arguments through identical code.
The prebuilt release binary is built with `serve`, so the downloaded
`ferrox` does both.

OpenAI-compatible HTTP API:

- Chat completions (SSE, optionally resumable: `id:` + `retry:` +
  `Last-Event-ID` replay and a JSON polling fallback), completions,
  tokenize / detokenize
- `POST /v1/cancel` stops a running generation by request id, which is
  the stop path a resumable stream needs, since closing its socket no
  longer ends it
- Embeddings. A real encoder checkpoint (BGE / E5 / GTE class, anything
  whose `tokenizer.ggml.model` is `bert`) can be the loaded model:
  `FERROX_MODEL_PATH=bge-small-en-v1.5-q8_0.gguf` serves `/v1/embeddings`
  and the six generating routes answer **501 naming the model**, not a
  missing tensor. Pooling comes from the checkpoint's own
  `pooling_type` (NONE / MEAN / CLS / LAST; RANK refuses, it is a
  classification head rather than a pooling rule, and a rank-head
  checkpoint belongs on `/v1/rerank` instead). A decoder GGUF still
  pools its hidden states (mean/last) as before, and
  `FERROX_EMBEDDING_MODEL_PATH` runs an encoder side-by-side with a
  generative model in one process
- Reranking. A `bert` checkpoint carrying a rank head (`cls`,
  `cls.output`, `cls.norm`, `classifier.output_labels`) is served by
  `POST /v1/rerank`, scoring `[CLS] query [SEP] document [SEP]` through
  the head itself rather than through the cosine of two embeddings. Such
  a checkpoint could not load at all before: `assert_every_tensor_
  consumed` rejected the `cls.*` tensors nobody read. Verified end to
  end against `ms-marco-MiniLM-L6-v2`: the order matches a HuggingFace
  reference, which needed the document to be segment 1 rather than
  llama.cpp's all-zero token types
- Anthropic Messages: `POST /v1/messages` streaming and buffered
  (thinking and tool blocks, protocol-native `ping` keepalive) plus
  `POST /v1/messages/count_tokens`
- `POST /v1/responses`, the surface `codex` speaks, streaming and
  buffered. This server keeps no responses, so the two lookups by
  response id answer 404
- Sampling matched to llama.cpp's own chain, all nine steps of it in
  upstream's order: `penalties`, `dry`, `top_n_sigma`, `top_k`,
  `typ_p`, `top_p`, `min_p`, `xtc`, `temperature`
  (`common/common.h:259-269`). Temperature runs **last**, after the
  truncation filters, and the repetition penalty is applied once per
  candidate rather than once per occurrence. DRY's sequence breakers are
  tokenised against the loaded model's own vocabulary; a checkpoint with
  none refuses DRY rather than running it breaker-less. `mirostat` and
  `infill` are the two upstream samplers still refused, each by name.
  Every route reads the same knobs through one `SamplingKnobs::resolve`
- Grammar-constrained decoding, in every spelling: llama.cpp's own
  `grammar` field, OpenAI's `response_format: json_schema`, llama.cpp's
  bare `json_schema` field on `/completion`, and a forced `tool_choice`.
  A schema is compiled to GBNF first, so all of them end at one stack
  machine that masks every token which cannot continue a valid string,
  on chat and completions and all three decode paths. Two constraints in
  one request are refused rather than ranked. `response_format:
  json_object` is still the best-effort character mask, and composes
- **Parallel serving (Metal).** Multiple concurrent streaming clients
  share one batched decode worker (llama.cpp slots model). Continuous
  batching is on by default when compatible; streaming emits tokens
  incrementally under CB (0.15.2). Metal CB prefill keeps host K/V
  authoritative for batched decode (0.15.3). Host B receipt on
  Llama-3.2-3B Q4_K_M: 16/16 OK at concurrency 8, ~24 aggregate tok/s,
  ~118 ms mean TTFT sequential. CLI: `-cb`, `-np` / `--parallel N`,
  `--no-cont-batching` for the serialized private path. See
  [`plans/metal-parallel-concurrency.md`](plans/metal-parallel-concurrency.md)
- Chunked prefill (same scheduler as continuous batching)
- Paged KV: shared page storage many requests read through a block
  table, with a radix tree over reference-counted page groups so
  conversations off one system prompt share its KV rather than each
  holding a copy. Off unless `FERROX_PAGED_KV_BLOCKS` is set. It used to
  be refused on a GPU backend, where it returned fluent wrong tokens; the
  cause was a Metal prefill leaving K/V on the device and filling the
  host cache with placeholders that the paged prefill then copied into
  the page store, and that refusal is lifted. See
  [`CONFIG.md`](CONFIG.md)
- On a model whose layers *all* slide by the same window, that
  window slides during decode, so a request holds its prompt and a
  window rather than its whole context , and admission prices it
  that way, so a store too small for the whole context still serves
  it. A tool call anchors the slide at the position the next agentic
  turn will rejoin at, and the anchor is dropped once the cursor
  drifts a window past it. An alternating-SWA model (gpt-oss,
  Gemma-3) does not slide: a page group holds one block in every
  layer, and the full-attention layers still read position 0
- `ferrox serve-bench`: concurrency, TTFT, TPOT and queueing numbers
  for a live server, with the methodology (positional split, pooled
  nearest-rank percentiles, whole-run throughput) tested socket-free.
  Host B receipts for Metal CB at 0.15.3:
  [`benchmarks/receipts/serving/`](../benchmarks/receipts/serving/)
- Live serving telemetry (`GET /v1/stats`, `GET /v1/requests`) and an
  elastic KV/expert split that can be reported and re-sized without a
  restart (`GET /v1/cache/status`, `POST /v1/cache/rebuild`). A request
  that arrives mid-rebuild is turned away with an error rather than
  parked in a queue behind it
- `reasoning_content`: a reasoning model's chain of thought is split
  out of `content`, streamed as it arrives rather than at the end
- Tool calls in eleven wire formats, not one: the format the served
  checkpoint's family emits, then the prompt-engineered one, and every
  call in a response rather than the first. Five of the eleven stream
  their arguments as deltas
- Prompts rendered by *evaluating* the checkpoint's own
  `tokenizer.chat_template`, with `chat_template_kwargs` and
  `reasoning_effort` passed through (the effort quantized onto what that
  checkpoint's template really grades)

## Edge-native MoE serving: what is real here

FreeToken describes an edge-native MoE serving engine, and ferrox ports
its host-side policy (Apache-2.0, see
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)). That policy now lives in
the crates that use it rather than in a crate of its own: the expert
residency stack in `ferrox-core` beside `expert_store`, and the serving
policy in `ferrox-server::policy`.

This table is what ferrox actually does against that description,
checked against the code rather than asserted. The gap is the roadmap.

| Capability | In ferrox today |
|---|---|
| Bandwidth-adaptive CPU/GPU co-execution (`q*`) | **Partial.** `qstar::BandwidthProfile` is in `ferrox-core` and used by `ferrox bench-bw`, a measurement tool. The serving path does not consult it. |
| Full-layer double-buffered prefill streaming | **Built, not wired.** Kept for the out-of-core work, which names it. |
| Global LRU expert caching | **Yes, and now singular.** `expert_store` is wired into both decode paths and proven bit-identical to resident at a 1-byte budget. The second, competing cache and its separate byte budget were folded in beside it. |
| Graph-compatible execution | **No.** Execution is eager. `ExecutionPlan` is built and read by nothing. |
| FTW fast weight format | **No.** GGUF only. |
| Semantic anchor checkpoints for KV | **Yes.** `anchor::decode_slide` and `WindowPolicy` are wired into `generate.rs` and the batch scheduler. |
| Agentic context edits without recompute | **Partial, and currently leaking.** The radix prefix cache shares pages and reports `cached_tokens`, but `RadixCache::evict` has no caller, so the page pool shrinks until admission refuses. |
| Elastic VRAM re-allocation without restart | **Partial.** `POST /v1/cache/rebuild` re-splits KV pool geometry at runtime. Moving bytes between an expert cache and KV is not implemented. |
| MXFP4 / BF16 | **Yes**, executable. MXFP4 is CPU-only. |
| NVFP4 / FP8 | **No.** Neither is parsed. |
| DeepSeek-V4-Flash, GLM-5.2, Kimi K3 | **Loaders and primitives only.** Nothing has run end to end on a real checkpoint. |
| OpenAI + Anthropic compatible APIs | **Yes**, both, plus Responses. Tool calls parsed in eleven wire formats. |
| NVIDIA RTX 30/40/50 | **Compiles, never measured.** CUDA has no in-tree benchmark receipt and no GPU in CI. |

Two honest notes. Ferrox runs on Apple Metal, which that description
does not cover, and Metal is where it is fastest: every dense `pp512`
row is 0.98x to 1.10x against llama.cpp and 8 of 12 `tg128` rows are
faster. And the single largest gap is not on this table: running a model
that does not fit in memory works as policy and not as execution.

## Serving policy

A Rust port of the host-side decision logic in
[FreeToken](https://github.com/FlashML-org/FreeToken)
([arXiv:2608.16157](https://arxiv.org/abs/2608.16157)): the parts of an
edge-native MoE engine that *decide* rather than compute. Tensor-free
and testable without a GPU. Each module takes measured numbers and
returns a decision.

It lives in the crates that use it: the serving half in
`ferrox-server::policy`, the MoE expert-residency half in `ferrox-core`
beside `expert_store`, which is the single holder of the expert byte
budget.

### Driving something today

| Module | Decides | Where it runs |
|---|---|---|
| `parser` | where reasoning ends and the answer begins, and which tool was called in which format | `/v1/chat/completions`, `/v1/messages`, `/v1/responses`, streaming and buffered |
| `detokenize` | what text is safe to stream after one more token | the stop-string withhold rule, which `ferrox-server`'s `StopMatcher` delegates to so there is one implementation |
| `radix` | which prefix of a new prompt is already computed, page-keyed and node-sharing | the paged-KV serving path, where it shares KV pages between prompts by reference count |
| `anchor` | how far a window may slide, and where a tool call pins it so the next agentic turn rejoins rather than recomputes | the paged-KV serving path, on both the private generate loop and the continuous batcher |
| `scheduler` | admission, chunked-prefill sizing, and what a chunk reserves | the continuous batcher's status and pool accounting |
| `effort` | which reasoning-effort dialect a checkpoint speaks | probed once per checkpoint at load, then applied to every request's `chat_template_kwargs`, and advertised on `/v1/models` |
| `serving_stats` | what a server may honestly claim about its own throughput and latency | `/v1/stats`, `/v1/requests`, `/admin/stats` |
| `maintenance` | whether a request, a cache rebuild or a stop may proceed right now | `POST /v1/cache/rebuild` and `POST /v1/admin/prepare-stop` |
| `pool` | how VRAM splits between the expert cache and KV, and how it is re-split live | the target geometry `POST /v1/cache/rebuild` validates against |
| `rebuild` · `outbox` · `footprint` | whether a re-split rolls back, what a stop receipt is worth, what this process really occupies | the same two admin endpoints |
| `deepseek_v4_budget` | per-layer KV tier sizing, and which compressor each layer runs (none / CSA / HCA) | the DeepSeek-V4 decoder |
| `bench_profile` · `bench_client` | when a measured bandwidth profile may be trusted, and what a serving benchmark may report | `ferrox bench-bw` and `ferrox serve-bench` |

### Complete, tested, and waiting for a consumer

`qstar` (the `q*` bandwidth split), `expert_cache`, `expert_slots`,
`expert_budget`, `placement` and `residency`, all in `ferrox-core`.
Each is covered by unit tests and none of them is on a serving path.
Do not read a benchmark as evidence for any of them.
[`plans/out-of-core-moe.md`](plans/out-of-core-moe.md) is what they are
waiting on: running a model larger than memory, which is the single
largest thing they would buy.

`expert_slots` sits closest to real memory: it executes the expert
cache's copy plans against a bounded slot pool, and a warm decode step
copies zero bytes on a host pool. `ferrox-core`'s `CudaExpertPool`
implements its `SlotDevice` trait under `--features cuda`, and that pool
is compile-verified with its hardware test left `#[ignore]`d, so on a
real card the property is written down and not yet measured. A host
`SlotDevice` (`HostSlotMemory`) also exists; a Metal one does not, and
that is the concrete gap.

Inside `ferrox-server::policy`, the modules carrying an unwired half
name the roadmap item that would close it, at their declaration in
`policy/mod.rs`. `grep -n "allow(dead_code)" crates/ferrox-server/src/policy/mod.rs`
is the list of what still owes a caller.

Ferrox Studio, the web UI in [`ui/`](../ui), is a separate app that
talks to this API over HTTP. `ferrox-server` does not serve it, and
`GET /` on it is a 404.

See [`API.md`](API.md) and [`AGENTS_COOKBOOK.md`](AGENTS_COOKBOOK.md).
