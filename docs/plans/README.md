# Plans

Two files hold the plan:

- **[`north-star.md`](north-star.md)** is the goal and the ranking rule.
  Be the Rust alternative to llama.cpp: same models, same command
  shapes, same or better performance, on the hardware people actually
  own.
- **[`roadmap.md`](roadmap.md)** is every open item, merged by theme.

Three items are large enough to carry their own design document:

- **[`speculative-decoding.md`](speculative-decoding.md)**, the one
  decode item that raises throughput without buying hardware. Decode
  reads every weight per token, so bandwidth divided by model bytes is a
  hard ceiling; a draft model changes what is read per token rather than
  how fast. The lossless half already ships and is tested at 200k
  samples. What is missing is a drafter worth having.
- **[`model-layer-reorg.md`](model-layer-reorg.md)**, splitting the
  decoder so architectures scale. It was 6438 lines when that document
  was written and is 6702 today, the first time it has shrunk.
- **[`out-of-core-moe.md`](out-of-core-moe.md)**, running a 155 GB model
  on a 32 GB machine.

How work lands is written down too:

- **[`contribution-workflow.md`](contribution-workflow.md)**, the rule
  that a completed feature is a branch and a pull request, a defect is a
  GitHub issue, and the two are never the same artifact. It also carries
  the parallel-agent rules, whose first failure is two branches editing
  one file.

One item has a written **verdict** rather than a design:

- **[`vulkan-beachhead-verdict.md`](vulkan-beachhead-verdict.md)**, the
  `d-hardware-reach` GO/NO-GO. GO: a Q8_0 matvec ran as a hand-emitted
  SPIR-V shader on a real device and matched its scalar twin. It also
  carries the survey of the backend seam a third backend would need,
  which is `backend-seam-refactor`'s to-do list.

Parity inventory and deltas against llama.cpp:

- **[`llama-cpp-gap-inventory.md`](llama-cpp-gap-inventory.md)** — evidence-backed differential (not a plan)
- **[`llama-cpp-full-parity-audit-2026-09-02.md`](llama-cpp-full-parity-audit-2026-09-02.md)** — file map + sweep + priority plan
- **[`llama-cpp-parity-update-2026-09-03.md`](llama-cpp-parity-update-2026-09-03.md)** — post-merge delta (Qwen MoE Metal, Phi-4 LongRoPE, sweep)
- **[`cpu-cuda-parity.md`](cpu-cuda-parity.md)** — the two backends that
  are not at parity, ordered by what was measured on rented hosts on
  2026-09-04 rather than by tok/s. Carries the kernel-coverage matrix,
  because a gap column cannot show a format the backend never runs

Everything else is history: [`archive/`](archive/) holds the five plans
whose items were merged into the roadmap, [`on-hold/`](on-hold/) holds
work ranked below the goal with the condition that brings it back, and
[`done/`](done/) holds plans whose todos are all completed.

## Where the project stands

Re-audited 2026-09-12, by what happens when a real checkpoint loads
rather than by whether the architecture name is known:

| Outcome | Count |
|---|---|
| Runs, **with evidence** | **53** (`capability::AUDITED_GENERIC_GQA`) |
| Loads on a dedicated engine | 4 engines (`Mla`, `Glm52`, `Kimi`, `Gemma4`); `Mla` has cross-engine evidence since 2026-09-12 (`plm`, `tests/plm_graphs.rs`), the other three none |
| Refuses as **unaudited**, now triaged | 3 |
| Off the generic path: refuses by name, or reaches one of those 4 engines | 90 (58 `dedicated` + 32 `deferred` in the manifest) |
| **Loads and is WRONG** | **closed** |

Counts reproduce from
[`../manifests/architecture_manifest.md`](../manifests/architecture_manifest.md),
regenerated with `ferrox archs --write`: 150 rows, 56 generic-gqa (53 of
them audited), 59 dedicated, 32 deferred, 3 test fixtures.

The "loads and is WRONG" class is closed because the generic path is
opt-in: an architecture not on the audited list stops rather than
guessing. The five strings that used to compute ALiBi or learned
position embeddings as though they were NEOX RoPE (`gpt2`, `mpt`,
`refact`, `bloom`, `jais`) are `DedicatedOnly` refusals, pinned by a
test that they can never be re-listed as audited.

The 3 unaudited refusals split 0 fixture-away / 0 one-match-arm /
2 new-code / 1 unknown, each naming the `llama.cpp/src/models/*.cpp`
line that decides it. **Both cheap classes are empty**: nothing still
refusing is one fixture or one arm away, so every row left needs a
different graph. Five one-match-arm rows closed on 2026-09-02
(`seed_oss`, `maincoder`, `bailingmoe`, `deepseek`, `hunyuan-moe`),
seven fixture-away rows on 2026-09-03, `gemma`, `hunyuan-dense` and
`ernie4_5-moe` after them, and on 2026-09-10 `olmo2` and `exaone4`
(below) plus `chatglm` -- the last one-match-arm row -- and `qwen`,
which the same arm turned out to close only halfway, the three
Granite rows and `olmo` (both below), and on 2026-09-11 `exaone-moe`
(below), then `grok` and `dbrx` on seams landed the day before, then
`arcee` on the ungated ReLU-squared FFN and `deci` and `openelm`
together on the per-layer shape seam (`ferrox_models::layer_shapes`,
sized by a scan of all 140 graphs before it was built), then `afmoe`
and `laguna` together on the gated attention
(`ferrox_models::attn_gate`, one op with two free parameters behind
three verdicts, read side by side first), then `mellum` on the
per-layer window array, `apertus` and `step35` together on the
per-layer activation parameters, and `mistral3` on the per-position
attention temperature (`ferrox_models::attn_temperature`, whose
reach -- three graphs of 140 -- was measured first and came back with
one generic-path row), and on 2026-09-12 `smallthinker` on the MoE
router operand (`ferrox_models::router_input`: fifty-nine
`build_moe_ffn` call sites parsed first, four pass a precomputed
`probs_in`, one on this engine routes on something other than the
normed FFN input; its "one match arm" ReLU experts turned out to need
a variant, because the one that existed served `arcee` by aliasing a
gate SmallThinker really has), and `bitnet` on the two norms INSIDE
the blocks (`ferrox_models::sub_norms`: one graph of 140 creates
either tensor, so the seam is a `bool`; its optional per-projection
`.scale` tensors, which llama.cpp applies for every architecture, are
refused by name in `ferrox_models::weight_scales`), and `mimo2` on the
split K/V head width (`ferrox_models::kv_head_dims`: one generic-path
converter writes the two widths apart, the KV cache, the one row
kernel the three contiguous arms collapsed onto, the batched prefill
kernel and every check took the V width, and the bisection to its
last 2e-3 of KL found `expert_weights_scale` honoured for every
architecture where llama.cpp reads it in twenty loaders), and
`nanbeige` on the layer loop (`ferrox_models::layer_loops`: the weights
are shared and the KV is not, so the seam is a logical-to-physical
mapping and a loop norm, not a copy of the weights), and `talkie` on
four seams at once (`NormOp::RmsNoParams`, `QkNormStyle::PerHeadScalar`,
`ferrox_models::skip_stream`, and the two `.scale` companions
`ferrox_models::weight_scales` now serves), each one graph of 140,
and `plm` on the MLA engine (`ferrox_models::mla_arch`,
`ferrox_models::mla_q_proj`: its attention was already there, and the
three ways it differs from DeepSeek-2 -- a direct `attn_q`, an ungated
ReLU-squared dense FFN, a tied lm_head -- are one table; the direct-Q
column also lifts the refusal of every lite DeepSeek-V2 export, and
the fixture is that engine's FIRST libllama golden). Each with a
libllama-golden fixture. `minicpm` closed on
2026-09-10 and is not in that arithmetic: it was refused BY NAME rather
than as unaudited, so it raises the audited count without lowering the
refusing one; `smollm3` and EXAONE-4 32B closed with `exaone-moe` on
2026-09-11 and are the same case, one a DedicatedOnly refusal and the
other a refusal by name.

The three UNKNOWN rows `mistral`, `mixtral` and `yi` closed the same
day by turning out not to be architectures: libllama refuses all three
strings outright and every real checkpoint of all three declares
`llama`, so they are refused as spellings now rather than triaged as
graphs. `phi4` is the one UNKNOWN left.

**The NEW CODE column moved for the first time on 2026-09-10**, three
times: 26 to 24, 24 to 21, then 21 to 20, and on 2026-09-11 seven times
more, 20 to 19, 19 to 17, 17 to 14, 14 to 12, 12 to 11, 11 to 9 and 9
to 8. The first two took several rows
at once for the same reason -- each found ONE cause behind several
refusals. The fourth did too and the count hides it: the per-layer RoPE
gate closed three refusals, and only `exaone-moe` was in this column.
The sixth is the per-layer shape seam, whose reach was measured across
all 140 graphs before it was built (`layer_shapes::PER_LAYER_SHAPE_ARCHS`
is the record): it closed `deci` and `openelm` and narrowed `laguna`,
`mimo2` and `step35` to what else each needs. The seventh took the
seam's leftovers: `afmoe`, `laguna` and `step35` had been narrowed to
the same last word, `wqkv_gate`, and reading the three graphs side by
side found one op with two free parameters (`ferrox_models::attn_gate`),
so two closed and the third says the gate is done. `mimo2`'s sinks
became a tensor-presence fact on the same day without closing it:
every real export carries MTP blocks and a per-layer window array;
it closed on 2026-09-12 on its split K/V head width.
The tenth, `mistral3`, is what a reach measurement looks like when it
comes back with one: the other two graphs that build the temperature
input are on other engines (`llama4` from literals, `deepseek2` /
`mistral4` from the same key, which the MLA loader refuses by name
now where it dropped it), and the verdict's second half -- one GGUF
key, `yarn_log_multiplier` -- found YaRN's magnitude term missing for
every architecture on the generic path (`ferrox_models::yarn_magnitude`).

`olmo` is the one that did not, and it is worth reading for the way the
question was settled rather than for the row. "What else shares this
cause" was answered by MEASUREMENT before any code was written: every
`build_norm` call in all 140 of llama.cpp's `src/models/*.cpp` graphs
was scanned for a null weight argument, and all three hits are
`olmo.cpp`. `openelm`, `bitnet`, `arcee`, `mellum`, `nanbeige` and
`deci` were the rows the search was aimed at and not one of them norms
without parameters -- they are checked-and-recorded now instead of
still-plausible, which is most of the value. The LayerNorm *function*
IS shared, by `dbrx` and the `nemotron` / `orion` / `stablelm` /
`codeshell` / `jais2` / `starcoder` / `starcoder2` / `phimoe` bias
group, and none of them is one variant away because each refuses for
more than the norm; `capability::NON_PARAMETRIC_LAYER_NORM` carries the
whole finding.

`olmo2` and `exaone4` closed TOGETHER, because they are one residual
topology and not two. Neither has an `attn_norm` or an `ffn_norm`
tensor; both read the raw residual at each sublayer and norm each
branch's output before its residual add (`olmo2.cpp:45-52,92,160-182`,
`exaone4.cpp:60-67,118,152-169`, line for line the same graph).
`ferrox_models::norm` is the one implementation and
`tests/post_norm_only_graphs.rs` the evidence, a libllama-golden fixture
each. One sub-case stays refused BY NAME rather than being swept in: an
`olmo2` carrying both a sliding window and a RoPE scaling (Olmo-3) ropes
its two kinds of layer differently. EXAONE-4 32B (`block_count == 64`),
whose full-attention layers get no RoPE at all, was the other and is
closed (next paragraph but one).
`olmo` (OLMo-1) is a THIRD shape -- pre-norm with a non-parametric
LayerNorm at all three sites, `olmo.cpp:65-67,104-106,128-130` -- and
closed as a third variant of the same enum
(`tests/olmo_graphs.rs`). `Decoder::final_norm` became a `NormOp` with
it: OLMo-1's final norm has no weights either, and the fused Metal
stacks that fold `final_norm + lm_head + argmax` had
`Some(&self.final_norm)` written into them unconditionally. Half its
verdict stayed a refusal, and the half that looked like an aside:
`olmo.cpp:5` reads `{arch}.attention.clamp_kqv`,
`llama-graph.cpp:1611-1652` clamps Q, K and V by it, and
`conversion/olmo.py:23-25` writes it for every checkpoint whose HF
config carries a `clip_qkv` -- OLMo-7B-Twin-2T and OLMo-1.7-7B do, the
original OLMo-7B does not. A second fixture measures that llama.cpp's
own logits move when the key is present, so it is not a no-op that
could be ignored.

`exaone-moe`, `smollm3` and EXAONE-4 32B closed TOGETHER on 2026-09-11,
on the per-layer RoPE gate, and the pairing was CHECKED before it was
assumed: `exaone4.cpp:116` is `use_rope = is_swa(il) || swa_type ==
NONE`, `exaone-moe.cpp:136,155-161` is `is_swa(il)` around the same two
`ggml_rope_ext` calls, and `exaone-moe.cpp:4` pins `swa_type` to
`STANDARD`, which nails the second disjunct false. Identical, not
similar. `smollm3.cpp:5,69` is a different variant of the same enum
(`(il + 1) % 4 != 0`, no window). `ferrox_models::rope_layers` is one
table for all six architectures llama.cpp gates this way, with
`smallthinker`, `afmoe` and `llama4` in it; the first two closed later
on other seams and `llama4` is still refused for other things. The durable part is the type: `ModelConfig::layer_rope` returns
`Option<(base, divisors)>`, so a rotation site cannot take the pair
without answering whether to rotate, and the Metal stacks take an
`Option<LayerRope>` per layer -- their RoPE dispatch had been written in
unconditionally, the OLMo-1 final-norm shape again -- while the four
per-layer Metal launches lost their loose base/divisor parameter pair
for one `LayerRope`. `tests/no_rope_layer_graphs.rs`: KL 2.05e-12 on a
64-layer EXAONE-4 fixture (64 because `exaone4.cpp:4` tests equality),
1.43e-14 on `exaone-moe`, 5.29e-15 on `smollm3`. Found on the way:
EXAONE-4 1.2B must IGNORE a window its file declares
(`exaone4.cpp:4-14` reaches `set_swa_pattern` only at 64 layers), now
in `capability::swa_disabled_by_arch` beside `phi3`; and
`nextn_predict_layers` -- MTP blocks INSIDE `block_count`, which
llama.cpp skips -- was refused nowhere, so a real EXAONE-MoE export
with an MTP head would have run it as two extra decoder layers. (Since
2026-09-11 the blocks are skipped as llama.cpp skips them,
`ferrox_models::mtp_blocks`, and the per-layer window array that the
same exports carry is `ferrox_models::swa_layers`.)

`minicpm` was never an unaudited row: it was refused BY NAME, because
`minicpm.cpp:5-7` assigns an embedding multiplier of 12.0, a residual
multiplier of `1.4/sqrt(n_layer)` and a logit multiplier of `256/n_embd`
BEFORE `:12-14` lets the file override them, so a key-PRESENCE gate sees
nothing in a file that is still scaled three ways. It runs Granite's
graph verbatim (`models.h:1594-1601`), so the fix was a DEFAULTS field
on the table `scalar_multipliers` already had, and the fixture that
evidences it declares no scaling key at all -- the only fixture shape
that can tell the hook from its absence. A second one declares all three
and pins the merge ORDER, which one fixture cannot see.

`granite`, `granitemoe` and the `granite-moe` alias closed together too,
on ONE implementation of the four scalar multipliers they share
(`ferrox_models::scalar_multipliers`, `tests/granite_family_graphs.rs`).
`granite-moe.cpp` has no graph of its own -- `models.h:1583-1591` is
`using graph = llama_model_granite::graph` -- so the two upstream rows
differ in the FFN and in nothing else, and the third is a ferrox-only
alias for the second. Deriving
`capability::unsupported_scaling_keys` from that same table instead of
restating it beside it found a live gap on the way past: the Gemma
family was exempted from all four keys while reading none of them, so a
hand-written `gemma3.residual_scale` would have loaded and been ignored.
Half the Granite verdict stayed a refusal: llama.cpp reads
`{arch}.rope.scaling.finetuned` as a switch for RoPE itself, and a file
declaring it false runs unrotated, which ferrox cannot express.

| | llama.cpp | ferrox |
|---|---|---|
| Per-architecture graphs | 140 hand-written | 150 catalog rows, **37 proven** |
| Metal `pp512` | baseline | 0.98x-1.10x, at parity |
| Metal `tg128` | baseline | **8 of 12 rows faster** |
| CPU, all rows | baseline | **1.41x-5.06x slower** |
| GPU backends | CUDA, Metal, Vulkan, SYCL, HIP | CUDA, Metal |

Do not read the architecture catalog as a support matrix.

## The rules that keep being re-learned

**A plan's own status field is a claim, not evidence.** Verify against
the code. A merged PR once marked `paged-decode-path` complete while it
returned wrong tokens on Metal.

**One agent owns a file.** Two branches editing the same file produce a
merge nobody can review, and this project has already had one branch
silently revert three others.

**No agent runs benchmarks.** Measurement needs a quiet host, and a
loaded run reads 25-45% low.

**Refusing is not a defect.** llama.cpp will often run something
approximately; this project stops and names what is missing. A refusal
is a gap in coverage, not a bug.
