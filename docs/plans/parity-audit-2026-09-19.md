# Parity audit, 2026-09-19: llama.cpp AND vLLM

What this is: a re-measurement of the two engines ferrox is read
against, done on the day 0.25.0 shipped, so the next work is chosen
from numbers rather than from the last audit's memory.

Method, and the reason for it: every count below came from a command,
and the command is printed beside it. `docs/plans/llama-cpp-gap-
inventory.md` (2026-09-01) was written the same way and one of its own
rows turned out to be wrong; a count nobody can re-run is a claim, not
a measurement.

## 0. The one-line answer

Against the llama.cpp this repo PINS, ferrox serves **every text
generation architecture that has a graph**. Against llama.cpp's
`master` as of today it serves 116 of 130, because the pin is six
weeks and **792 commits** stale and fourteen architectures landed in
that window. Against vLLM the architecture question is the wrong one
-- the overlap is high and the naming spaces differ -- and the gap is
in SERVING features, where one of them is already half-built in this
tree.

## 1. llama.cpp

### 1.1 The pin is stale, and that is now the headline

```
$ cd .scratch/llama.cpp && git log -1 --format=%ci      # 2026-08-04
$ git rev-list --count HEAD..origin/master              # 792
$ git ls-tree -r --name-only origin/master src/models | wc -l   # 156
```

The pinned tree has 140 graphs and `master` has 155 plus `clip.cpp`.
Fifteen new files, fourteen of them new architectures:

```
$ git show origin/master:src/llama-arch.cpp | grep -o 'LLM_ARCH_[A-Z0-9_]*,\s*"[^"]*"' \
    | sed 's/.*"\(.*\)"/\1/' | sort > /tmp/up.txt      # 153 names
$ git show HEAD:src/llama-arch.cpp | ... > /tmp/pin.txt  # 139 names
$ comm -13 /tmp/pin.txt /tmp/up.txt
```

| arch | graph | lines | what it is, and what ferrox would need |
|---|---|---|---|
| `granite_swa` | `granite-swa.cpp` | 319 | Granite's four multipliers (served) on an iSWA pattern array (served) with an optional per-layer `expert_used_count` ARRAY. Closest to a fixture-away row of the fourteen |
| `graniteswitch` | `granite-switch.cpp` | 427 | the same multipliers plus an `adapter_ids` argument threaded through the layer: a per-token expert-adapter selection with no counterpart here |
| `maple` | `maple.cpp` | 150 | iSWA with `rope.freq_base_swa` and the `swiglu_clamp_exp` array (both served, `step35`), MoE with a per-layer `expert_ff_length` array. Second-closest |
| `muse-glimmer` | `muse-glimmer.cpp` | 203 | window + `logit_scale` + final logit softcap (all served) with an attention GATE (`attn_gate`, served since `afmoe`) |
| `spark2_5` | `spark2-5.cpp` | 146 | plain layers with an attention gate |
| `hrm_text` | `hrm-text.cpp` | 213 | two transformer stacks alternating over one token stream under H/L cycle counts. `crate::layer_loops` (nanbeige) is the same IDEA -- weights replayed, KV logical -- with a different schedule |
| `dots3note` | `dots3note.cpp` | 476 | DSA indexer + absorbed MLA (the `glm-dsa` engine's shape) with step35's head-wise output gate |
| `hy_v4` | `hy-v4.cpp` | 601 | independent hyper-connections: several residual streams reduced and redistributed per layer, plus a DSA cache |
| `minimax-01` | `minimax-01.cpp` | 484 | lightning attention as a RECURRENT layer under `attention.recurrent_layers` / `full_attention_interval` -- the hybrid seam `qwen35` built, with a different block |
| `bailingmoe3` | `bailingmoe3.cpp` | 540 | MLA + KDA (Kimi delta attention) hybrid with a conv kernel and a safe-gate flag |
| `kimi-k3` | `kimi-k3.cpp` | 618 | KDA + MLA hybrid, cross-layer residual attention, latent MoE, "situ" activation, an MLA output gate |
| `qwen4exp` | `qwen4exp.cpp` | 1297 | the largest new graph: hybrid memory index, delta-net, MoE |
| `pockettts` | `pockettts.cpp` | 146 | TTS. Out of scope until an audio scope exists |
| `qwen3tts` | `qwen3tts.cpp` | 3 | TTS shim |

Twelve of the fourteen are text generation. Two of those twelve
(`granite_swa`, `maple`) read only keys and ops ferrox already serves,
which is the cheapest class this repo has -- and the lesson from
`minimax-m2` is that a row in that class costs a fixture and an hour,
so it should not sit in a table for a week.

Three (`minimax-01`, `bailingmoe3`, `kimi-k3`) are hybrid recurrent
rows on the seam `granitehybrid` / `qwen35` built, which is the seam
that has closed nine rows in two weeks.

### 1.2 Against the pin, the text-generation gap is zero

```
$ ./target/release/ferrox archs | awk -F'|' 'NF>4{print $6}' | sort | uniq -c
   95 generic-gqa   21 dedicated   31 deferred   3 test-fixture
$ ... | awk -F'|' 'NF>4{print $3}' | sort | uniq -c
  119 TextGeneration  11 DeferredEncoderEmbedding  10 DeferredMultimodal
    5 EnumOnly         4 DeferredDiffusion          1 DeferredAudio
```

Every deferred row is an encoder/embedding, a multimodal, a diffusion
or an audio model, or an `EnumOnly` name llama.cpp itself has no graph
for (`gptj`, `eagle3`, `dflash`, `clip`, `(unknown)`). **No text
generation architecture of the pinned llama.cpp is refused.**

So the four remaining llama.cpp scopes are, in the order their user
population justifies:

1. **encoder / embedding (11)**: `bert`, `nomic-bert`, `nomic-bert-moe`,
   `jina-bert-v2`, `jina-bert-v3`, `modern-bert`, `neo-bert`,
   `eurobert`, `gemma-embedding`, `llama-embed`, `t5encoder`. ferrox
   already serves `/v1/embeddings` and `/v1/rerank` -- from a decoder.
   These are the models people actually embed with, and `bert` alone
   is most of that population.
2. **multimodal (10)**, which needs an image encoder and a projector
   before any of the ten matters.
3. **diffusion text (4)** and **audio (1)**.

### 1.3 What this means for the pin

Updating the pin is not a chore here, it is the measurement: every
capability table in `ferrox-models` is derived from a census over
`src/models/*.cpp`, and a census over a six-week-old tree can be
wrong in the direction that matters (a graph that started reading a
key). The pin bump and the census re-run belong in one PR, before any
of the fourteen rows.

## 2. vLLM

### 2.1 The model count is not the comparison

```
$ curl .../vllm/model_executor/models/registry.py
_TEXT_GENERATION_MODELS 137   _EMBEDDING_MODELS 37   _MULTIMODAL_MODELS ~225 entries
```

Those are HF `*ForCausalLM` class names, several of which map to one
GGUF architecture string (`LlamaForCausalLM`, `MistralForCausalLM`,
`YiForCausalLM` are all `llama` in GGUF, which this repo learned the
hard way on 2026-09-10). Counting them against ferrox's 116 would be
comparing two different things. The honest statement is that the
text-generation OVERLAP is close to complete, and vLLM's advantage is
in three scopes ferrox defers: multimodal, pooling/embedding models,
and encoder-decoder.

### 2.2 Feature parity, measured against this tree

vLLM's own `docs/features/README.md` matrix rows, each checked against
ferrox by grep rather than by memory:

| vLLM feature | ferrox | evidence |
|---|---|---|
| chunked prefill (CP) | **yes** | `ferrox-server/src/generate.rs` |
| automatic prefix caching (APC) | **yes** | `policy/radix`, over paged KV |
| LoRA, per request | **yes** | `ferrox-server/src/lora.rs`, with a reader/writer gate llama.cpp does not have |
| speculative decoding (SD) | **in the engine, NOT in the server** | `ferrox_models::speculative` + `draft_model`; the only caller is `ferrox-cli/src/main.rs:1468` |
| structured outputs | **yes** | `grammar_request.rs`, `json_mode.rs`, `tool_grammar/`, `ferrox_models::grammar` |
| tool calling | **yes** | and 0.25.0 fixed the format being chosen by the served NAME |
| reasoning outputs | **yes** | `reasoning_tokens.rs`, `reasoning_budget.rs` |
| pooling / embeddings | **partial** | `/v1/embeddings`, `/v1/rerank` from a decoder; no BERT-family encoder |
| logprobs / top_logprobs | **yes** | `responses.rs`, `openai_extra.rs` |
| prompt logprobs | **no** | no match anywhere |
| `n` > 1 / best-of / beam search | **no** | no match anywhere |
| prompt embeds as input | **no** | no match anywhere |
| encoder-decoder | **no** | `t5` and friends are deferred |
| multimodal | **no** | 10 deferred rows |
| CUDA graph capture | **no** | and on Metal the equivalent -- one encoded graph per token -- is exactly what 0.25.0's submission collapsing approximates by hand |
| tensor / pipeline parallel | **no** | single process, single device |
| disaggregated prefill, KV connectors, KV offload | **no** | `docs/plans/out-of-core-moe.md` is the nearest thing and is groundwork |
| sleep mode | **no** | |
| per-request metrics | **yes** | `stats/` |
| quantized KV cache | **yes** | `--cache-type-k` / `-v` |

### 2.3 The finding worth acting on

**Speculative decoding is built and unreachable from the server**, and
it is this repo's dominant bug shape wearing a different hat: two
structures that must agree, with nothing enforcing it. The evidence is
not an opinion --

```
$ grep -rn 'speculat' crates/ferrox-server/src --include=*.rs -l
crates/ferrox-server/src/stats/requests.rs
$ grep -rn 'with_speculation' crates/ferrox-server/src | grep -v test
(nothing)
```

-- the server has an acceptance-rate metric, a test that the metric
reaches the admin ring, and NO producer for it. A metrics column that
no code path can fill reads as coverage, which is the same defect
class as a gate that cannot fire.

`ferrox_models::speculative` is lossless by construction (the
Leviathan / Chen rejection rule, with `accept_or_resample` pinned by
tests), `PromptLookupSpeculator` needs no second checkpoint and no
GPU, and `draft_model.rs` already exists for the `--model-draft` case.
So the whole of vLLM's `n_gram` and `draft_model` spec-decode arms are
one wiring job away over the API, and the `mtp` arm is one `Drafter`
impl away for the seventeen architectures whose MTP blocks ferrox
already SKIPS by name (`crate::mtp_blocks::NEXTN_READERS`).

## 3. Ranked, against the north star

The north star is "the Rust alternative to llama.cpp: same models,
same command shapes, same or better performance". vLLM is the second
reading, not the first, so a vLLM-only feature ranks below a llama.cpp
gap of the same size.

1. **Update the llama.cpp pin and re-run every census.** Everything
   below is measured against a tree that is 792 commits old, and two
   of this repo's tables are derived from a grep over it.
2. **Speculative decoding in the server.** Built, tested, lossless,
   unreachable; the metric for it already exists. Both engines have
   it; only ferrox has it and cannot serve it.
3. **`granite_swa` and `maple`**, the two new llama.cpp rows that need
   no new op. A fixture each.
4. **`bert` and the encoder/embedding family.** Eleven llama.cpp rows
   and vLLM's whole pooling scope in one seam, and ferrox already has
   the two routes that would serve them.
5. **`minimax-01`**, the new hybrid recurrent row, on the seam that has
   closed nine rows in two weeks.
6. The MLA/DSA and hyper-connection rows (`dots3note`, `hy_v4`,
   `bailingmoe3`, `kimi-k3`, `qwen4exp`), each of which needs a block
   that does not exist here yet.
7. `n` > 1 / best-of / prompt logprobs -- small, API-shaped, and
   nobody has asked.

Multimodal is deliberately below all of these: it is an image encoder,
a projector and a preprocessing pipeline, and it would be the largest
single thing in the tree.
