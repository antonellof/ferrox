# Models

What Ferrox runs, and how it compares to llama.cpp on the same host.
Speed table: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)
(`ferrox bench` vs `llama-bench`). Suite list:
[`benchmarks/suite.json`](../benchmarks/suite.json). Architecture list:
`ferrox archs` →
[`manifests/architecture_manifest.md`](manifests/architecture_manifest.md).

**Gap** = `llama / ferrox`. Values below 1.0 mean Ferrox is faster.

Suite policy: keep the **current** generation per family. Llama-3.2, not
3.1. Gemma-3/4, not Gemma-2. Phi-4, not Phi-3. Older GGUFs still load
when the architecture is supported, they are simply not measured in the
published table. To measure a new model, add a suite entry and put the
GGUF under `models/`.

## Recommended starters

| Model | Notes |
|---|---|
| SmolLM2-135M-Instruct Q8_0 | Tiny. Metal ahead of llama, CPU well behind |
| TinyLlama-1.1B-Chat Q8_0 | Smallest verified smoke |
| Phi-4-mini-Instruct Q4_K_M | Metal works again. The RoPE kernels now carry `n_rot` (96 of head_dim 128) and LongRoPE's `attn_factor`, and `verify --backend metal` returns identical CPU and Metal token ids with prefill covered. The Metal rows in `benchmarks/RESULTS.md` predate that fix and were taken on the wrong graph. **Do not quote them until Phi-4 is measured again.** |
| Llama-3.2-3B-Instruct Q4_K_M | Metal flagship in the suite |
| Gemma-4-E2B-IT Q4_K_M | Dedicated engine + `gemma4` BPE |

```bash
./target/release/ferrox -m /path/to/model.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

./target/release/ferrox-server -m /path/to/model.gguf \
  --host 127.0.0.1 --port 8383

./target/release/ferrox chat --url http://127.0.0.1:8383
```

## Verified (Host B)

Gap = `llama / ferrox` from `ferrox bench` vs `llama-bench` (tg128 unless
noted). **Bold** = ferrox faster. Neither engine's thread count is forced.

| Model | Metal decode | CPU decode |
|---|---|---|
| SmolLM2-135M Q8_0 | **0.67×** | 2.44× |
| Qwen2.5-0.5B Q8_0 | **0.70×** | 1.66× |
| Qwen3-0.6B Q8_0 | **0.71×** | 1.63× |
| Gemma-3-1B-IT Q8_0 | **0.88×** | 1.31× |
| Llama-3.2-1B IQ4_XS | **0.94×** |, |
| Llama-3.2-1B Q4_K_M | 1.00× |, |
| TinyLlama-1.1B Q8_0 | **0.85×** | 1.49× |
| Llama-3.2-3B Q4_K_M | **0.96×** |, |
| Phi-4-mini Q4_K_M |, (owed, see above) | 1.22× |
| Mistral-7B-v0.2 Q4_K_M | 1.00× | 1.17× |
| OLMoE-1B-7B Q4_0 | 1.41× | 1.50× |

These numbers drift as runs are refreshed.
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) is generated
straight from the raw timing files, so trust it over this hand-written
summary.

Prefill is **closed on Metal for dense models** (every dense `pp512` row
is 1.02–1.08×). What is left on `pp512` is CPU across the board, plus
OLMoE (1.11×) and Gemma-3-1B (1.18×) on Metal.

## Other support

| Model / family | Status |
|---|---|
| Yi (text) | Works (GenericGqa, Neox RoPE), not in suite yet |
| MiroThinker | Works via `qwen3moe` |
| Qwen2-MoE / Qwen1.5-MoE | Loads. Not in the current suite (OLMoE is the MoE entry) |
| Mixtral | In the suite, skipped on 32 GiB Host B (`--fit-host`) |
| MLA (`deepseek2` / `mistral4`) | Dense-lead + MoE-after-dense via `MlaEngine` |
| GLM4 / glm4moe | Loads via the GLM-5.2 path when the tensors are there. Never measured in the suite |
| Gemma-4-E2B | Dedicated `Gemma4Engine` + SPM-style `gemma4` BPE tokenizer + `<|turn>` chat wrap. GGUF: `models/gemma-4-E2B-it-Q4_K_M.gguf` (`unsloth/gemma-4-E2B-it-GGUF`). Suite id `gemma4_e2b_q4km`, Homebrew llama may still lack `gemma4` arch. |
| gpt-oss | **CPU only.** Attention sinks, alternating sliding-window attention, biased router and the `swiglu_oai` clamp, checked against llama.cpp's own reference logits. Metal stops with an error, because no Metal kernel implements attention sinks. The paged-KV decode path runs it: all three attention arms are bit-identical to their contiguous twins |
| Llama 4 | **Will not load**, with the reason stated: `llama4 MoE + non-GQA attn` |
| MiniMax | **Will not load**, and the two are refused for different reasons. `minimax-m2` is *unaudited, not unimplemented*: plain GQA + whole-vector QK norm + partial NEOX RoPE + a sigmoid MoE with `exp_probs_b`, all of which the generic path has, so it needs a fixture rather than code. `minimax-m3` needs MiniMax Sparse Attention (a per-layer indexer driving its own MSA KV cache), of which ferrox has only the block-selection rule |
| Hybrid GDN / Qwen3.5 | Scaffold only |
| Kimi K3 / GLM-5.2 / DeepSeek V4 | Loaders and primitives only. Nothing has been run end to end on a real checkpoint |
| Vision | Finds an mmproj file and warns about it. An `image_url` in a request returns an error |
| MTP / speculative | `--mtp` errors by design. `ferrox speculative` is prompt-lookup only (an n-gram match over the history, no draft model) and runs on **synthetic random weights**, so the hit rate it prints is not representative of a real drafter. Plan for a real one: [`docs/plans/on-hold/dflash-speculative-decoding.md`](plans/on-hold/dflash-speculative-decoding.md) |
| Embeddings | `/v1/embeddings` for GGUF Decoder (mean/last pool) |

## When a model will not load

Some checkpoints stop with an error instead of running. That is
deliberate. The alternative is worse: a model whose graph Ferrox only
partly implements will load, run fast, and return fluent text computed
by the wrong maths, and nothing in the output tells you. An error you
can read beats output you cannot trust.

The error always names the reason. Six things cause it:

1. **Ferrox does not know the architecture.** It is not in the
   capability registry.

2. **Ferrox knows it and has not implemented it.** `llama4`,
   `minimax-m2` and `minimax-m3` stop with the missing feature named.
   So do architectures whose residual wiring differs from the
   `x + attn(norm(x))` then `y + ffn(norm(y))` shape the generic decoder
   computes: `command-r`, `cohere2`, `cohere2moe`, `falcon`, `gptneox`,
   `phi2` and `plamo` feed both branches the same normed input and sum
   once. `minicpm` scales its embeddings, residuals and logits by
   multipliers llama.cpp applies even when the GGUF omits every key.
   None of that shows up in a tensor or, for MiniCPM, in metadata, so
   these are listed by name rather than detected.

3. **The file contains weights Ferrox never reads.** The GGUF reader
   records every tensor name a loader looks up and stops if any are left
   over: *"checkpoint carries N tensor(s) this build never reads, so its
   graph is not the one this build computes."* Unread weights mean the
   file describes a model Ferrox is not computing. This catches missing
   graph features automatically rather than one at a time, which is how
   `attn_sinks` and `exp_probs_b` were both found. Tensors for parts
   Ferrox does not claim to run (`mm.`, `v.`, `mmproj.`, `resampler.`,
   `audio.`) are ignored.

4. **The file declares a scale factor Ferrox does not apply.** These are
   hyperparameters rather than weights, so the check above cannot see
   them, and a Granite, MiniCPM or Command-R checkpoint would otherwise
   load cleanly while computing a differently-scaled graph than it was
   trained as. `{arch}.logit_scale`, `{arch}.residual_scale`,
   `{arch}.embedding_scale` and `{arch}.attention.scale` stop the load
   unless they hold a value that changes nothing. Implementing
   `residual_scale` properly means touching every CPU residual add plus
   the fused Metal kernels that fold the residual in, and getting half
   of that right gives you a model that loads, runs, and returns wrong
   answers with nothing in the output to say so. That is the outcome
   this check exists to prevent.

5. **The architecture encodes position some other way than RoPE.** The
   generic decoder rotates every Q and K head of every layer. `gpt2`
   uses a learned absolute position table instead; `mpt`, `refact`,
   `bloom` and `jais` use ALiBi. All five stop with the reason named.
   This is the least visible failure of the five: `bloom` and `refact`
   hardcode their ALiBi slope in llama.cpp's own loader and carry no
   GGUF key at all, and `mpt` leaves no unread tensor behind, so neither
   check 3 nor check 4 could ever see them. `baichuan` is the same
   problem conditionally: the 7B rotates, the 13B uses ALiBi, and
   llama.cpp tells them apart by layer count alone, so a 40-layer
   Baichuan is refused and a 32-layer one is not.

6. **Nobody has ever verified this architecture against llama.cpp.**
   The shared generic-GQA decoder is a *guess*: it assumes plain GQA
   because nothing said otherwise, and that guess was already wrong for
   the five architectures in cause 5. So the generic path is opt-in.
   An architecture reaches it only if there is a benchmark row, a pinned
   logit comparison against real `libllama`, or a fixture; 11 do today
   (`llama`, `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `olmoe`,
**Gemma-2-27B, Gemma-3-4B/12B/27B: corrected 2026-09-02.** Those four
sizes were quietly wrong until then, in two ways that both produce
fluent text. The 27B checkpoints took `1/sqrt(head_dim)` as their
attention scale where llama.cpp takes `1/sqrt(n_embd/n_head)` for that
size alone, selected by layer count; and Gemma-3 4B and up applied
linear RoPE scaling to the sliding-window layers llama.cpp ropes
unscaled, because `rope_theta` is per layer while `rope_freqs` was
global. Gemma-3-1B is the one size with no `rope_scaling`, and it was
the audited fixture, which is why neither was visible.

The evidence is llama.cpp's source and header-driven loader tests, NOT a
logit comparison: no checkpoint of those sizes exists on the development
host. A `ferrox parity` run against one is what would settle it.

A cost came with it and has since been paid back. Gemma-3 4B and up
briefly left the fused Metal dense stacks, because those took one
`freq_factors` slice for a whole run of layers while these models need
one per layer: correct, and slower. The stacks now carry a per-layer
`LayerRope` holding the base and its divisors together, so supplying one
without the other does not compile, and Gemma-3 4B+ is back on the fused
path (#63).

**That last part is proven on synthetic layers, not on a checkpoint.**
No Gemma-3 4B/12B/27B GGUF exists on the development host, and
Gemma-3-1B would prove nothing because it declares no rope scaling. The
Metal tests build a two-layer stack carrying Gemma-3's two answers
(base 1e6 divided by 8, and base 1e4 divided by 1) and assert the fused
and per-layer launches agree bit for bit, then re-run the old bug on
purpose so neither can pass vacuously. `ferrox layer-divergence` on a
real gemma-3-4b, plus `ferrox parity` at Q8_0, is what would settle it.

No published number changes:
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) carries Gemma-3-1B
only, and 1B is neither 27B nor rope-scaled, so it is untouched by all
of this. The speed recovery from returning to the fused path is
unmeasured, because measuring it needs a quiet host.

   `gemma`, `gemma2`, `gemma3`, `phi3`, `gpt-oss`, `dots1`). The other
   **29** stop with `UnauditedArchitecture`.
   `FERROX_ALLOW_UNAUDITED_ARCH=1` runs one anyway; compare the output
   against llama.cpp yourself before you trust it.

### What "unaudited" costs you, per architecture

"Unaudited" is not one thing. One of the 29 is one named match arm away
and the rest need an attention implementation or a reading nobody has
done, so the refusal says which, with the
`llama.cpp/src/models/*.cpp` line that decides it:

| Class | Means |
|---|---|
| `FIXTURE-AWAY` | Ferrox already computes this graph. What is missing is evidence. |
| `ONE MATCH ARM` | One small, named piece: an activation, a norm slot, a routing flag, an ordering. |
| `NEW CODE` | A different attention or residual structure. Not close. |
| `UNKNOWN` | Reading both trees did not settle it. The message says what would. |

All 29 have now been read on both sides (`ferrox_models::capability`,
pinned by `crates/ferrox-models/tests/unaudited_triage.rs`). The
distribution is the headline answer to "how far is Ferrox from llama.cpp
on models":

| Class | Count |
|---|---|
| fixture-away | 0 |
| one match arm | 1 |
| new code | 24 |
| unknown | 4 |

**Fixture-away is empty.** Every row that only needed evidence has it
now, so everything still refusing needs code or a reading. That is a
better answer than the count alone: the cheap wins are spent.

It was 47 until the triage itself removed one. Reading
`src/models/minicpm3.cpp:5-6,41-46` showed `minicpm3` requires
`q_lora_rank`/`kv_lora_rank` and the DeepSeek-2
`attn_q_a`/`attn_q_b`/`attn_kv_a_mqa`/`attn_kv_b` tensor set: it is an
MLA model that was never on the generic path, so it now refuses by name
(naming both the MLA tensor set and MiniCPM's three hardcoded
multipliers) rather than as unaudited. The count going down for the
right reason.

**Fixture-away (0).** The class started at 9 and is empty.
`internlm2`, `exaone`, `ernie4_5`, `bailingmoe2`, `xverse`, `baichuan`
(the 7B; the 13B uses ALiBi and is refused by layer count) and `plamo3`
were admitted with libllama-golden fixtures on 2026-09-03, `gemma`
followed, and `chatglm` left the class the other way: an attempt to
build its fixture found the fused `attn_qkv.bias` that every real
ChatGLM2/3 export carries and ferrox drops, so it is ONE MATCH ARM now.

**One match arm (1).** `chatglm`, above: the fused `attn_qkv.bias`,
which is the same arm `qwen` and `starcoder` are refused by name for.
The other six all closed. `seed_oss` and the gpt-oss norm slot;
`deepseek` and top-k renormalisation; `bailingmoe` and a
`leading_dense_block_count` llama.cpp reads but never uses;
`hunyuan-moe`, `maincoder` and `hunyuan-dense`, which all wanted the
same flag (QK norm applied *after* RoPE rather than before, plus, for
`hunyuan-dense` alone, the NTK-alpha RoPE base rescale); and
`ernie4_5-moe`, whose interleaved MoE layers landed as a REFUSAL rather
than an implementation, because llama.cpp's own tensor loader
(`ernie4-5.cpp:49`) has no interleave step in it while its graph
(`ernie4-5-moe.cpp:64`) does, so an interleaved checkpoint cannot be
loaded by llama.cpp either. The step every published ERNIE-4.5 MoE
checkpoint carries is 1, and that is what Ferrox runs and pins against
libllama.

**New code (24).** A different attention or residual structure. The
recurring shapes, rather than 24 separate stories:

The column moved for the first time on 2026-09-10, 26 to 24. `olmo2`
and `exaone4` were the POST-NORM-ONLY pair -- no `attn_norm` and no
`ffn_norm` at all, both sublayers reading the raw residual, each
branch's output normed before its residual add -- and they closed
together because reading `olmo2.cpp:45-52,92,160-182` against
`exaone4.cpp:60-67,118,152-169` showed one graph, not two. One
implementation (`ferrox_models::pre_norm`), one fixture each
(`tests/post_norm_only_graphs.rs`). Two sub-cases stay refused by name:
an `olmo2` with BOTH a sliding window and a RoPE scaling (Olmo-3) ropes
its sliding and full layers differently, and EXAONE-4 32B
(`block_count == 64`) gives its full-attention layers no RoPE at all --
both decided by llama.cpp with no GGUF key, the `baichuan` shape.

| Shape | Architectures |
|---|---|
| Per-layer head counts, FFN width or rotary width | `openelm`, `deci`, `laguna`, `step35`, `mimo2` |
| A norm the generic decoder always applies and the model does not have (or a norm it does not have a slot for) | `olmo` (and `olmo2` / `exaone4`, now CLOSED), `talkie`, `bitnet`, `dbrx` |
| LayerNorm rather than RMSNorm | `dbrx`, `olmo` |
| Unkeyed NoPE layers, RoPE skipped on some layers with no GGUF key | `smallthinker`, `afmoe`, `exaone-moe` |
| A branch fed from the raw layer input rather than the post-attention residual | `smallthinker` (its MoE router), `arctic` (its MoE branch) |
| Hardcoded scales applied even when the GGUF carries no key | `grok`, `granite`, `granitemoe`, `granite-moe`, `mistral3` |
| An ungated or non-SwiGLU FFN | `arcee`, `plm`, `apertus` |
| Something structurally new | `nanbeige` (runs the same layers more than once), `grovemoe` (a second expert bank), `mellum` (two per-layer RoPE variants), `mistral3` (per-position attention temperature) |

**Unknown (4).** Reading both trees did not settle it, and each says
what would. `phi4`, `mistral`, `mixtral` and `yi` are all names that do
not exist in llama.cpp's `LLM_ARCH_NAMES`, so there is no reference
graph to diff against. For the three alias rows this is not academic:
Ferrox gives them NEOX RoPE, while `llama`, the string real Mistral,
Mixtral and Yi checkpoints actually ship under, is in llama.cpp's NORM
group. A file spelling `mistral` would be rotated on the wrong pairs of
every Q/K head. Latent only because the row refuses.

`FERROX_ALLOW_UNKNOWN_TENSORS=1` loads the checkpoint anyway and accepts
whatever comes out. Use it while you debug, not to get past the error
and carry on.

## Quantization support

Parsed and executable on CPU: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`,
`Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`,
`IQ4_NL`, `IQ4_XS`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ3_S`, `MXFP4`.

"Executable" is not one speed. What a format actually gets, read off
the kernel tables (`ferrox_quant`'s dispatch functions, and
`metal_matvec_kind_name` / `metal_mul_mm_kind_supported` /
`cuda_matvec_kind_supported` / `cuda_mul_mm_kind_supported` in
`ferrox-core`'s `weight_matrix.rs`):

| Tier | Formats | CPU SIMD | GPU |
|---|---|---|---|
| Full | `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K` | AVX2 + NEON, plus the int-dot path (`FERROX_CPU_INT_DOT=1`) | Metal matvec + simdgroup GEMM, CUDA matvec |
| Metal only | `IQ4_XS`, `Q5_0` | AVX2 + NEON | Metal matvec + simdgroup GEMM; no CUDA kernel of either kind |
| CPU-vectorized | `Q4_1`, `Q5_1`, `Q8_1`, `Q2_K`, `Q3_K`, `IQ4_NL`, safetensors two-buffer `MXFP4` | AVX2 + NEON | none |
| AVX2 only | `IQ1_S`, `IQ2_XXS`, `IQ3_XXS` | AVX2; **scalar on ARM** | none |
| Scalar only | `IQ2_XS`, `IQ2_S`, `IQ3_S`, `IQ1_M`, GGUF-block `MXFP4` | none | none |

`Q5_0` moved up on 2026-09-01. It had a Metal simdgroup GEMM and no
matvec, so its prefill ran on the GPU and every decode step fell back to
the CPU, silently. The matvec now exists
(`Q5_0_MATVEC_KERNEL_SRC`, `ferrox-metal/src/gpu.rs:439`) and
`metal_matvec_kind_name` / `metal_mul_mm_kind_supported` name the same
seven kinds. It is **correct by construction and unmeasured**: there is
no `Q5_0` checkpoint in `benchmarks/suite.json`, so no row in
`RESULTS.md` covers it.

Metal's MoE indexed GEMM (`mul_mm_id`) is narrower still: `Q4_0`,
`Q8_0` and `Q4_K` only.

The GPU column deliberately says **CUDA matvec** and not CUDA GEMM.
`cuda_mul_mm_kind_supported` does hold `Q8_0` and `Q4_0`, and that
kernel has never executed on a GPU, so it is not a tier this table can
promise anything about. See Backends below.

Three caveats that matter in practice:

- **The IQ tiers split, and the split matters if you are choosing a
  quant.** The bottom two rows load and produce correct output, and they
  are slow. That was deliberate. They were added for coverage, because
  before them those tags could not be decoded at all, which ruled out 5
  of the 16 published Unsloth `UD-*` variants. A vectorized path was
  left out rather than written without a golden vector that could tell
  it apart from the scalar one.
- **On an Apple machine the "AVX2 only" row is the scalar row.**
  `IQ1_S`, `IQ2_XXS` and `IQ3_XXS` have x86 kernels and no NEON ones, so
  on ARM they run at the same speed as the scalar tier below them.
- **`I32`, `TQ1_0`, `TQ2_0`, `NVFP4`, `Q1_0` and `Q2_0` are recognized
  and sized, but nothing executes them.** They parse, `ferrox inspect`
  reports their real footprint, and a checkpoint that needs one stops
  with an error naming the format rather than being quietly skipped or
  silently mis-measured. Ternary (`TQ*`) and the two newest `Q*_0`
  formats are a real gap, not a claim of support.

`IQ2_XS`, `IQ2_S`, `IQ3_S` and `IQ1_M` were validated bit-exact against
llama.cpp's own `dequantize_row_*` by linking `ggml-quants.c`, not by
re-reading the spec. They have not been validated end to end on a
published `UD-*` checkpoint.

## Backends

| Backend | What it covers |
|---|---|
| CPU | Dense and MoE. `FERROX_CPU_INT_DOT=1` (Q4_Kx8 / Q8_0x4 / Q5·Q6 int-dot) on suite runs |
| Metal | Dense, MoE, FA-vec, fused MoE encode groups, `mul_mm_id` prefill, quantized KV |
| CUDA | Matvec + resident weights + FFN fuse. A batched `Q8_0`/`Q4_0` GEMM exists and has never run on a GPU (see below) |

Every number on this page was taken on CPU or Apple Metal. CUDA compiles
and runs, has no pinned benchmark host, and has no published timings, so
treat a Windows or Linux install as CPU-only in practice.

A batched quantized GEMM for CUDA (`Q8_0` and `Q4_0` only) is in the
tree and reachable from a wide prefill, and it has **never executed on a
GPU**. Its evidence is a thread-by-thread scalar twin plus a host
harness that compiles and runs the emitted CUDA against a barrier shim
(`crates/ferrox-cuda/tools/mul_mm_host_check/run.sh`); the hardware test
is `#[ignore]`d with "NEVER RUN" as its reason. That is not a
performance claim, and no row in `RESULTS.md` rests on it.

Paged KV used to be refused on Metal and CUDA, because the paged
attention path there returned fluent wrong tokens. That refusal is
**lifted**: a Metal prefill left K/V on the device and filled the host
cache with placeholders that the paged prefill then copied into the page
store, and the prefill now downloads the real rows for the caller that
reads them. Pinned on hardware by `cargo test -p ferrox-models --features
metal --test paged_metal_parity -- --ignored`, which gets identical
greedy ids from the paged and contiguous caches on a dense, an MoE and a
sliding-window model. CUDA carries no equivalent hardware run. See
[`CONFIG.md`](CONFIG.md).

Capabilities overview: [`FEATURES.md`](FEATURES.md).
Planned work: [`ROADMAP.md`](ROADMAP.md).
