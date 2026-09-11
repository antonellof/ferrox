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
   once. None of that shows up in a tensor, so these are listed by name
   rather than detected. `minicpm` was on this list and no longer is:
   it never had a different residual, only three multipliers llama.cpp
   applies whether or not the GGUF declares them, and those are
   implemented now (see 4 below).

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
   them, and a checkpoint declaring one would otherwise load cleanly
   while computing a differently-scaled graph than it was trained as.
   `{arch}.logit_scale`, `{arch}.residual_scale`,
   `{arch}.embedding_scale` and `{arch}.attention.scale` stop the load
   unless they hold a value that changes nothing.

   **Granite is the exception now, and the refusal list is derived from
   the same table that says so.** `granite`, `granitemoe` and the
   `granite-moe` alias apply all four, checked against llama.cpp's own
   logits (`crates/ferrox-models/tests/granite_family_graphs.rs`), and
   `ferrox_models::scalar_multipliers` implements them once for all
   three rather than once per architecture. `residual_scale` was the
   expensive half: it multiplies both branch outputs of every layer,
   which meant collapsing eighteen hand-written residual adds in
   `decoder.rs` onto one function that takes the scalar as a parameter,
   and fencing the fused Metal launches -- which fold the residual in on
   device with no uniform for a multiplier -- off any model that
   declares one. Getting half of that right gives you a model that
   loads, runs, and returns wrong answers with nothing in the output to
   say so, which is what the refusal existed to prevent and what the
   fixtures now check.

   Two Granite cases still stop, and both are the same class of
   metadata-only fact. `{arch}.logit_scale` is REQUIRED for Granite
   (`granite.cpp:7` reads it with no default), so a file omitting it is
   refused rather than defaulted to 1.0 -- llama.cpp cannot load such a
   file either. And `{arch}.rope.scaling.finetuned = false` is refused
   outright, because `granite.cpp:33-35` reads that key as a switch for
   **RoPE itself**: llama.cpp then builds no positions and runs the
   checkpoint unrotated, which the generic decoder has no way to
   express. No Granite converter writes it
   (`conversion/granite.py:253` is `GraniteHybridModel`, a different
   architecture string), so a real export never sees that message.

   **MiniCPM runs now**, and it is the case a key-presence gate could
   never have caught. llama.cpp assigns its three multipliers
   (`minicpm.cpp:5-7`: 12.0, `1.4/sqrt(n_layer)` and `256/n_embd`) and
   only THEN lets the file override them (`:12-14`), so an export
   declaring nothing is still scaled three ways and the metadata looks
   ordinary -- which is why the row was refused on its architecture
   string. Its graph is `llama_model_granite::graph` verbatim
   (`models.h:1594-1601`), so the fix was a DEFAULTS field on the same
   table, and the fixture that evidences it declares no scaling key at
   all: a fixture that declared them would pass with or without the
   hook. A second fixture declares all three and pins that the FILE
   still wins, which one fixture cannot see
   (`crates/ferrox-models/tests/minicpm_graphs.rs`). MiniCPM reads no
   `attention.scale` (`minicpm.cpp:3-24` has no such key), so that one
   is still refused for this row, by the derived list -- measured, too:
   libllama's logits for a file declaring it are byte-identical to the
   file without it.

   `command-r` and `cohere2` are further off: their `logit_scale` is a
   multiply rather than a divide, but their real blocker is a parallel
   residual over LayerNorm.

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
   logit comparison against real `libllama`, or a fixture; **39** do
   today (`llama`, `qwen`, `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`,
   `olmoe`, `olmo2`, `chatglm`, `deepseek`, `bailingmoe`, `bailingmoe2`,
   `seed_oss`, `maincoder`, `hunyuan-moe`, `hunyuan-dense`, `ernie4_5`,
   `ernie4_5-moe`, `internlm2`, `xverse`, `baichuan`, `exaone`,
   `exaone4`, `exaone-moe`, `smollm3`, `plamo3`, `granite`, `granitemoe`,
   `granite-moe`, `minicpm`, `olmo`, `dbrx`, `grok`,
   `gemma`, `gemma2`, `gemma3`, `phi3`, `gpt-oss`, `dots1`). The other
   **18** stop with `UnauditedArchitecture`.
   `FERROX_ALLOW_UNAUDITED_ARCH=1` runs one anyway; compare the output
   against llama.cpp yourself before you trust it.

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


### What "unaudited" costs you, per architecture

"Unaudited" is not one thing. None of the 18 is a fixture or a single
match arm away any more: they need an attention implementation or a
reading nobody has done, and the refusal says which, with the
`llama.cpp/src/models/*.cpp` line that decides it:

| Class | Means |
|---|---|
| `FIXTURE-AWAY` | Ferrox already computes this graph. What is missing is evidence. |
| `ONE MATCH ARM` | One small, named piece: an activation, a norm slot, a routing flag, an ordering. |
| `NEW CODE` | A different attention or residual structure. Not close. |
| `UNKNOWN` | Reading both trees did not settle it. The message says what would. |

All 18 have now been read on both sides (`ferrox_models::capability`,
pinned by `crates/ferrox-models/tests/unaudited_triage.rs`). The
distribution is the headline answer to "how far is Ferrox from llama.cpp
on models":

| Class | Count |
|---|---|
| fixture-away | 0 |
| one match arm | 0 |
| new code | 17 |
| unknown | 1 |

**Both cheap classes are empty.** `gemma` was the last fixture-away row
and `chatglm` the last one-match-arm row; nothing still refusing is one
fixture or one arm away. That is a better answer than the count alone:
the cheap wins are spent, and what is left is 17 rows needing a
different graph plus one name nobody can get a file for.

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
ChatGLM2/3 export carries and ferrox dropped, so it became ONE MATCH
ARM before closing as that.

**One match arm (0).** `chatglm` was the last row here and closed on
2026-09-10. Its arm was the fused `attn_qkv.bias`: llama.cpp's
`create_tensor_qkv` puts the bias where the weight is, and ferrox split
the fused weight while reading the bias only under the split
`attn_q.bias` names, so ChatGLM2/3's `add_qkv_bias: true` was dropped
and all three projections ran unbiased. Both halves now come out of one
decision in `qkv_fused`, sliced by the same spans.

**`qwen` came with it, and needed a second arm nobody had named.** The
`chatglm` verdict predicted the bias would close both rows. The bias
really is shared -- `qwen.cpp:28` marks it REQUIRED, stronger than
chatglm's optional one -- but building the fixture found that
`qwen.cpp:33-35` also sizes every FFN matrix at `n_ff / 2`, because
Qwen-1's `intermediate_size` counts gate and up together. That costs no
logits (ferrox loads the dense FFN by tensor name and uses each
matrix's own shape) and still made `expert_ffn_dim` twice the real
width, which is what every memory estimate prices the FFN from. Both
rows are audited against libllama now, and a test compares the declared
FFN width against the matrices that load, over every dense fixture.

The other seven one-match-arm rows had closed earlier. `seed_oss` and
the gpt-oss norm slot;
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

**New code (17).** A different attention or residual structure. The
recurring shapes, rather than 17 separate stories:

The column moved for the first time on 2026-09-10, three times: 26 to
24, 24 to 21, then 21 to 20, and on 2026-09-11 twice more, 20 to 19 and
19 to 17. The first two took several rows at once, and for the same
reason -- each found ONE cause behind several refusals. The fourth did
too, and the count hides it: the per-layer RoPE gate closed THREE
refusals and only one of them (`exaone-moe`) was ever in this column.
The fifth is a different lesson, below: `grok` and `dbrx` each closed
by extending a seam that had landed the day before, and the clamp one
of them needed closed a refusal-by-name on a third row.

`olmo2` and `exaone4` were the POST-NORM-ONLY pair -- no `attn_norm` and
no `ffn_norm` at all, both sublayers reading the raw residual, each
branch's output normed before its residual add -- and they closed
together because reading `olmo2.cpp:45-52,92,160-182` against
`exaone4.cpp:60-67,118,152-169` showed one graph, not two. One
implementation (`ferrox_models::norm`), one fixture each
(`tests/post_norm_only_graphs.rs`). One sub-case stays refused by name:
an `olmo2` with BOTH a sliding window and a RoPE scaling (Olmo-3) ropes
its sliding and full layers differently, decided by llama.cpp with no
GGUF key, the `baichuan` shape. EXAONE-4 32B was the other, and it is
CLOSED -- see the next paragraph.

`exaone-moe`, `smollm3` and EXAONE-4 32B were the PER-LAYER-RoPE trio,
and the claim that they are one cause was checked before it was
assumed. `exaone4.cpp:116` is `use_rope = is_swa(il) || swa_type ==
NONE`; `exaone-moe.cpp:136,155-161` is `is_swa(il)` around the same two
`ggml_rope_ext` calls, and `exaone-moe.cpp:4` pins `swa_type` to
`STANDARD`, which nails the second disjunct false -- identical, not
similar. `smollm3.cpp:5,69` is a different variant of the same enum,
`(il + 1) % 4 != 0`, with no window involved. All six architectures
llama.cpp gates this way (`smallthinker`, `afmoe` and `llama4` are the
other three, each still refused for something else) sit in ONE table,
`ferrox_models::rope_layers`, and `ModelConfig::layer_rope` answers
`None` for an unrotated layer -- an `Option` around the base and the
divisors rather than a `bool` beside them, so no rotation site can take
the pair without answering the third question. Every site was checked:
the CPU head loop, the YaRN `attn_factor` (an argument to
`ggml_rope_ext`, so it goes with it), the four per-layer Metal launches
(now one `LayerRope` argument instead of a loose base/divisor pair),
and both fused Metal stacks, whose RoPE dispatch had been written in
unconditionally the way OLMo-1's final norm had. One libllama-golden
fixture per row (`tests/no_rope_layer_graphs.rs`): KL 2.05e-12 on the
64-layer EXAONE-4 32B file, 1.43e-14 on `exaone-moe`, 5.29e-15 on
`smollm3`. Building them found that EXAONE-4 1.2B must IGNORE a window
its file declares (`exaone4.cpp:4-14` reaches `set_swa_pattern` only at
64 layers), which `capability::swa_disabled_by_arch` now carries beside
the `phi3` case; that `nextn_predict_layers` was unrefused everywhere
(MTP blocks are inside `block_count` and llama.cpp skips them), which
`unsupported_feature_keys` now gates on the value the converters
actually write; and that `default_swa_layout`'s comment calling
`smallthinker` LIVE was wrong -- it has refused on its router since it
was triaged.

`granite`, `granitemoe` and the `granite-moe` alias were the SCALAR
MULTIPLIER trio, and the same story again: `granite-moe.cpp` has no
graph of its own (`models.h:1583-1591` is
`using graph = llama_model_granite::graph`), so the two upstream rows
differ in the FFN and in nothing else, and the third is a ferrox-only
alias for the second. One implementation
(`ferrox_models::scalar_multipliers`), one libllama-golden fixture each
(`tests/granite_family_graphs.rs`). Half the verdict stayed a refusal --
see cause 4 above for `rope.scaling.finetuned`.

`olmo` (OLMo-1) closed ALONE, and that is the interesting part. It is a
THIRD norm shape: pre-norm like llama, but `olmo.cpp:65-67,104-106,128-130`
normalise with `build_norm(x, NULL, NULL, LLM_NORM, il)` -- a
non-parametric LayerNorm -- and `olmo.cpp:15-36` creates no norm tensor
of any kind, not even an `output_norm`. Before writing it, the question
"what else shares this cause" was answered by measurement rather than
hope: every `build_norm` call in all 140 of llama.cpp's
`src/models/*.cpp` graphs was scanned for a null weight argument, and
all three hits are `olmo.cpp`. `openelm`, `bitnet`, `arcee`, `mellum`,
`nanbeige` and `deci` were the candidates and none of them qualifies.
The LayerNorm *function* is shared -- `dbrx` and the bias group below --
but at the time none of those was one variant away, so a
weighted-LayerNorm variant would have had no caller and was deliberately
not written. Half of OLMo-1's verdict stayed a refusal, and it is the
half that read like an aside: `olmo.cpp:5` reads
`{arch}.attention.clamp_kqv`, `llama-graph.cpp:1611-1652` clamps Q, K
and V by it inside `build_qkv`, and `conversion/olmo.py:23-25` writes it
for every checkpoint whose HF config has a `clip_qkv` -- OLMo-7B-Twin-2T
and OLMo-1.7-7B do, at 8.0; the original OLMo-7B does not. A second
fixture measures that llama.cpp answers differently with it, so it is
not a no-op that could be ignored. Both halves of that paragraph turned
out to be one day old.

`dbrx` and `grok` closed on 2026-09-11, each by extending a seam that
had landed the day before, and that is the whole reason they were cheap
enough to take. `dbrx`'s three blockers were the weighted LayerNorm --
the variant the `olmo` work had refused to write without a caller, and
`dbrx` is the caller (`NormOp::LayerNorm`, `dbrx.cpp:69-71,110-112,
140-142`) -- a REQUIRED `attention.clamp_kqv` (`dbrx.cpp:5`), and its
pre-FFN norm stored as `blk.N.attn_output_norm` (`:34,110-113`). The
clamp was the expensive one and the one worth the most: the three host
bodies each applied the QKV bias in their own hand-written loop, which
is exactly why the OLMo clamp had been refused rather than implemented
(a clamp added to some copies and not the others is this repo's
dominant bug shape), so the three loops collapsed onto one helper
(`decoder/qkv_bias.rs`) and the clamp is a line in it, with the fused
Metal launches fenced off through the same predicate as
`residual_scale`. The clamped OLMo fixture, which used to evidence a
refusal, now matches libllama on all three paths (KL 1e-11 class), so
OLMo-7B-Twin-2T and OLMo-1.7-7B run. KL on the DBRX fixture: 3.4e-12,
max |delta| 6.1e-6. A DBRX file without the clamp key is refused, as
libllama refuses it (`key not found in model: dbrx.attention.clamp_kqv`,
measured).

`grok` was the MiniCPM shape and the verdict said so: `grok.cpp:5-12`
seeds SEVEN hyper-parameters before `:14-27` let the file override them,
so a Grok-1 export declaring none is still scaled by all of them.
`MultiplierDefaults::Grok` is the hook, on the same table as MiniCPM's,
and two of the seven needed a column the table did not have:
`logit_scale` is a MULTIPLY (`grok.cpp:211`, `LogitScaleUse::AsIs`, the
variant the module had named as absent), and the attention scale comes
from `{arch}.attention.output_scale`, applied INSIDE the tanh softcap
with `kq_scale = 1.0f` (`:137`, `llama-graph.cpp:2572-2582`) -- which is
arithmetically "pre-scale Q, then softcap", i.e. the `attention_scale`
slot plus the softcap Gemma-2 already uses, so no new attention code.
The other two keys Grok reads, `router_logit_softcapping` and
`attention.temperature_length`, are applied NOWHERE in llama.cpp's
graph (no other reference under `src/`, measured), so ferrox neither
applies nor refuses them. `attn_output_norm` is Grok's POST-attention
norm -- the same tensor name `dbrx` stores its pre-FFN norm under --
which is why `ferrox_models::norm_sites` exists: one table for which
tensor feeds which site, replacing the `if` chain the loader restated
at every site. Two fixtures, as MiniCPM needed: one declaring NO key
(the only shape that can tell the hook from its absence) and one
declaring every key at a value far from its default (pinning that the
file wins; a hook merged the wrong way round agrees with llama.cpp on
exactly the files that prove it exists). KL 4.7e-10 and 1.6e-10, max
|delta| 6.3e-5 and 4.0e-5 -- at the GeGLU tolerance, and measured to be
entirely llama.cpp's f16 GELU table: with ferrox's GELU made to emulate
the table both files agree to 1.0e-7 / 1.5e-7. Grok-2's parallel dense
FFN (`grok.cpp:171-184`, summed with the experts at `sqrt(2)/2`) is
refused by name from a fixture that has it, so the row is admitted for
Grok-1.

| Shape | Architectures |
|---|---|
| Per-layer head counts, FFN width or rotary width | `openelm`, `deci`, `laguna`, `step35`, `mimo2` |
| A norm the generic decoder always applies and the model does not have (or a norm it does not have a slot for) | `talkie`, `bitnet` (`olmo`, `olmo2`, `exaone4` and `dbrx` were here and are CLOSED) |
| LayerNorm rather than RMSNorm | CLOSED for the weightless (`olmo`) and weighted (`dbrx`) forms; the bias group below still refuses for more than the norm |
| Unkeyed NoPE layers, RoPE skipped on some layers with no GGUF key | CLOSED for all six (`ferrox_models::rope_layers`): `exaone-moe`, `smollm3` and EXAONE-4 32B run on it; `smallthinker` and `afmoe` still refuse for the rows below and their verdicts say so |
| A branch fed from the raw layer input rather than the post-attention residual | `smallthinker` (its MoE router), `arctic` (its MoE branch) |
| Hardcoded scales applied even when the GGUF carries no key | `mistral3` (`grok` was here and is CLOSED on the MiniCPM defaults hook) |
| An ungated or non-SwiGLU FFN | `arcee`, `plm`, `apertus` |
| Something structurally new | `nanbeige` (runs the same layers more than once), `grovemoe` (a second expert bank), `mellum` (two per-layer RoPE variants), `mistral3` (per-position attention temperature) |

**Unknown (1).** `phi4` is the only row left here. It is not in
llama.cpp's `LLM_ARCH_NAMES` -- `src/llama-arch.cpp` carries `phi3` and
no phi4 entry -- so there is no reference graph to diff against, and
Ferrox admits it as phi3's fused-QKV / fused gate+up graph on the
assumption that a file spelling it means the same thing. It refuses
until a real file settles that, and its message says which tensor in
`blk.0` would decide it.

**`mistral`, `mixtral` and `yi` were the other three, and were resolved
on 2026-09-10 by finding they are not architectures.** The old verdict
asked for "a real GGUF whose `general.architecture` is literally one of
these three". No such file can be produced: none of the three is in
llama.cpp's `LLM_ARCH_NAMES` or in gguf-py's `MODEL_ARCH_NAMES`
(`mistral3` and `mistral4` are the only strings under that prefix), and
libllama REFUSES a file declaring one -- `unknown model architecture:
'mistral'`, measured on a synthetic llama-shaped file written under
each string. The two real checkpoints on the development host,
`Mistral-7B-Instruct-v0.2-Q4_K_M.gguf` and `Yi-1.5-6B-Chat-Q4_K_M.gguf`,
both declare `general.architecture = llama`, which is audited and runs.

So all three are refused as *strings*, not triaged as architectures,
and the refusal says the actionable thing: re-convert with
`convert_hf_to_gguf.py` and your file will load as `llama`. Moving them
also closed a live hazard. They sat on the generic path with NEOX RoPE
while `llama` -- the graph they claim to be -- is in llama.cpp's NORM
group, so a file spelling `mistral` would have been rotated on the
wrong pairs of every Q/K head, and the only test that compares RoPE
layouts could not see it, because a name absent from llama.cpp's table
is a `continue` there. That skip now has to be declared by name.

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
