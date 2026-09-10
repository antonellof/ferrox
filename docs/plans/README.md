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

Re-audited 2026-09-01, by what happens when a real checkpoint loads
rather than by whether the architecture name is known:

| Outcome | Count |
|---|---|
| Runs, **with evidence** | **28** (`capability::AUDITED_GENERIC_GQA`) |
| Loads on a dedicated engine, no cross-engine evidence | 4 engines (`Mla`, `Glm52`, `Kimi`, `Gemma4`) |
| Refuses as **unaudited**, now triaged | 29 |
| Off the generic path: refuses by name, or reaches one of those 4 engines | 90 (58 `dedicated` + 32 `deferred` in the manifest) |
| **Loads and is WRONG** | **closed** |

Counts reproduce from
[`../manifests/architecture_manifest.md`](../manifests/architecture_manifest.md),
regenerated with `ferrox archs --write`: 150 rows, 57 generic-gqa (28 of
them audited), 58 dedicated, 32 deferred, 3 test fixtures.

The "loads and is WRONG" class is closed because the generic path is
opt-in: an architecture not on the audited list stops rather than
guessing. The five strings that used to compute ALiBi or learned
position embeddings as though they were NEOX RoPE (`gpt2`, `mpt`,
`refact`, `bloom`, `jais`) are `DedicatedOnly` refusals, pinned by a
test that they can never be re-listed as audited.

The 29 unaudited refusals split 0 fixture-away / 1 one-match-arm /
24 new-code / 4 unknown, each naming the `llama.cpp/src/models/*.cpp`
line that decides it. **Fixture-away is empty**: every row that only
needed evidence has it, so what is left needs code or a reading. Five
one-match-arm rows closed on 2026-09-02 (`seed_oss`, `maincoder`,
`bailingmoe`, `deepseek`, `hunyuan-moe`), seven fixture-away rows on
2026-09-03, and `gemma`, `hunyuan-dense` and `ernie4_5-moe` after them,
each with a libllama-golden fixture.

**The NEW CODE column moved for the first time on 2026-09-10**, from 26
to 24: `olmo2` and `exaone4` closed TOGETHER, because they are one
residual topology and not two. Neither has an `attn_norm` or an
`ffn_norm` tensor; both read the raw residual at each sublayer and norm
each branch's output before its residual add
(`olmo2.cpp:45-52,92,160-182`, `exaone4.cpp:60-67,118,152-169`, line for
line the same graph). `ferrox_models::pre_norm` is the one
implementation and `tests/post_norm_only_graphs.rs` the evidence, a
libllama-golden fixture each. Two sub-cases stay refused BY NAME rather
than being swept in: an `olmo2` carrying both a sliding window and a
RoPE scaling (Olmo-3) ropes its two kinds of layer differently, and
EXAONE-4 32B (`block_count == 64`) gives its full-attention layers no
RoPE at all. `olmo` (OLMo-1) is a THIRD shape -- pre-norm with a
non-parametric LayerNorm -- and still needs code.

| | llama.cpp | ferrox |
|---|---|---|
| Per-architecture graphs | 140 hand-written | 150 catalog rows, **28 proven** |
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
