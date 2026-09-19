---
name: completion push, finishing the seven open plans
overview: "GOAL: close the 56 todos still open across the seven plans in docs/plans/, using parallel agents where the work is genuinely independent and strict ordering where it is not. This plan owns no engineering of its own. It is the schedule: which items may run at the same time, which must not, which are blocked on something nobody in this repo controls, and what each wave must prove before the next starts. THE CONSTRAINT THAT SHAPES EVERYTHING: agents collide on files, not on topics. Four branches editing crates/frink-models/src/decoder.rs is a merge nobody can review, and two of them silently reverting a third is how this project already lost a day. So waves are cut by FILE OWNERSHIP first and by dependency second. HONEST COUNT: of the 56, seven cannot be finished on any machine this project has access to, and they are named in `blocked-on-hardware-or-downloads` rather than left to fail in a wave."
todos:
  - id: wave-0-unblock
    content: "DONE 2026-08-27, branch `wf/wave0-paged-gpu`. The cause was not the page indirection and not a missing Metal attention arm: a Metal prefill leaves K/V on the device and zero-fills the host `KvCache` via `advance_len`, which the CONTIGUOUS decode path knows and reads around, and `forward_batch_last_paged` then copied those zeros into the page store -- so decode attended over a prompt the model never saw. The prefill now downloads the real rows for the one caller that reads them. Five MORE model features had also been lost by the paged decode loop being a copy, and three of them (Gemma's `attention_scale` and the two sandwich norms) were wrong on CPU as well as GPU. The refusal in frink-server/src/lib.rs is out in the same commit, along with the `gpu_offload_resolved` helper that existed only to serve it. Verified on hardware by a new `paged_metal_parity` test -- greedy ids identical between paged and contiguous KV on a dense model, OLMoE and Gemma-2, five consecutive Metal runs -- and end to end through the server on Llama-3.2-3B, where paged and contiguous return the same text. THE SAME CAUSE WAS BREAKING `FRINK_PREFIX_CACHE_ENTRIES` ON METAL, with no refusal in front of it: the snapshot it stores IS the host rows, so it stored zeros and the next request restoring them answered nonsense at full speed. Reproduced and fixed in the same commit, because a live wrong answer with no guard is worse than the one that had a guard. Details and the two things that are NOT fixed are in freetoken-parity's `paged-decode-path`. Wave 2 is unblocked."
    status: completed
  - id: wave-1-independent
    content: "WAVE 1, five agents in parallel, no file overlap between them, none blocked on anything. (a) CPU KERNELS, owns crates/frink-core/src/{weight_matrix,matmul}.rs: cpu-act-interleave-hoist, cpu-kill-transpose, cpu-i8mm-q5k-q6k, cpu-i8mm-q8_0-q4_0, cpu-actquant-flat, in that order because each rewrites the same GEMM entry points. Check cpu-gemma3-prefill's note first: it records that gemm_q8_0x4_q8_0_neon_i8mm is ALREADY the i8mm 4x8 kernel, which may make half of cpu-i8mm-q8_0-q4_0 stale. (b) CPU ATTENTION, owns crates/frink-core/src/attention.rs: cpu-prefill-attn-block and cpu-gemma3-prefill. (c) UI, owns ui/: ui-frontend-chat, which needs a real conversation API on the server first, so it owns crates/frink-server/src/conversations.rs as a NEW file and touches lib.rs only to mount the route. (d) TOOLING, owns crates/frink-cli/: tooling-quant-sensitivity, tooling-layer-divergence, hygiene-cross-target-gate. (e) COVERAGE, owns crates/frink-models/src/{chat_template,loader}.rs: coverage-jinja-templates (the evaluator exists and NOTHING CALLS IT, that is the whole item) and coverage-cheap-archs"
    status: pending
  - id: wave-2-paged-and-serving
    content: "WAVE 2, three agents, starts only after wave-0-unblock lands. (a) RADIX, owns crates/frink-edge/ and the batcher: wire-radix-prefix-cache, radix-on-the-batcher, window-slide-during-decode. These three are one feature and must not be split across agents. (b) SCHEDULER, owns crates/frink-server/src/batch_scheduler.rs: sched-time-debt. Read its todo first, it carries the design (chunk DURATION is the quantum because GPUs cannot preempt a running kernel). (c) MOE EXECUTION, owns crates/frink-moe/ and the CPU expert path: concurrent-cpu-moe-executor, double-buffered-prefill. persistent-gpu-expert-cache is NOT in this wave; it needs CUDA hardware, see blocked-on-hardware-or-downloads"
    status: pending
  - id: wave-3-architectures
    content: "WAVE 3, two agents, after wave 1's coverage agent lands because all three touch the loader and the decoder. (a) gemma4-moe-router and minimax-m3-block-sparse, one agent, sequential, owns crates/frink-models/src/decoder.rs. Both are real graph features and both must fail closed rather than compute a different model if only half-implemented, which is this repo's standing rule and the reason 24 architectures were found rotating the wrong RoPE pairs. (b) child-log-stream, owns crates/frink-server/src/, independent of the above and only in this wave because it is small and nothing else needs the slot"
    status: pending
  - id: wave-4-measure-and-close
    content: "WAVE 4, sequential and NOT parallel, because every item is a measurement and measurements do not share a host. Runs last, on a quiet machine, with no agents building: suite-validate-every-change, quality-gates, close-all-red-rows, bench-bw. `frink bench` now refuses a busy host, a hot host, and a host without the free memory to hold the model, so a wave-4 run that reports numbers is a run worth publishing. close-all-red-rows is the definition of done for llama-cpp-parity-push and can only be judged after wave 1's CPU work lands and is measured. If the CPU rows have not moved, the honest outcome is to say so and reopen the items, not to close the plan"
    status: pending
  - id: blocked-on-hardware-or-downloads
    content: "SEVEN ITEMS CANNOT BE FINISHED HERE, stated up front rather than discovered at wave 4. (1) real-moe-checkpoints: nobody has downloaded a published UD-* or dots1-family checkpoint, so the IQ tiers and the MoE routing bias are validated bit-exact against llama.cpp's own dequantization and NOT end to end on a real file. Needs a download decision, not an agent. (2) persistent-gpu-expert-cache: needs a CUDA host; the CUDA pool is compile-verified only and its hardware test is #[ignore]d. (3) dflash-checkpoint-reality: a GO/NO-GO that needs a published drafter checkpoint to exist in a loadable format. Everything else in dflash-speculative-decoding (drafter-forward, block-diffusion, kv-injection, path-selector, two-tap-conv, metal, server-integration, bench-harness) is downstream of that answer and MUST NOT be started before it, because frink does not train and a drafter without weights proposes noise, which is slower than no drafter at all. (4-7) amd-strix-halo's vulkan-beachhead, vulkan-decode-path, vulkan-prefill-gemm and bench-suite-on-128gb need a Strix Halo box. NOTE the rest of amd-strix-halo does NOT: x86-first-measurement, thread-default-x86, strict-kernels-on-x86, avx512-int-dot, uma-residency-semantics, gtt-carveout-doc, backend-seam-refactor, ci-x86-and-vulkan, hip-revisit-gate and docs-and-features-honesty are all doable on any x86 host or on this one, and hygiene-cross-target-gate already proved the cross-target build works locally"
    status: pending
  - id: wave-5-x86-track
    content: "WAVE 5, one agent, independent of waves 1-4 and safe to run alongside any of them because it owns no file they touch. The x86 half of amd-strix-halo: x86-first-measurement, thread-default-x86, strict-kernels-on-x86, avx512-int-dot, backend-seam-refactor, ci-x86-and-vulkan, plus the three documentation items (uma-residency-semantics, gtt-carveout-doc, docs-and-features-honesty) and hip-revisit-gate. The Vulkan and 128 GB items stay blocked. This wave is where the plan's own honesty clause gets tested: docs-and-features-honesty exists to stop the project claiming a backend it has never measured, which is the same rule that keeps CUDA at 'must compile' in docs/FEATURES.md"
    status: pending
  - id: found-in-wave-1-not-yet-fixed
    content: "THREE BUGS THE WAVE 1 AGENTS FOUND WHILE DOING SOMETHING ELSE, recorded here so they are not lost with the agent transcripts. None is fixed. (1) MODEL SWAP IS WRONG ON METAL, and the server exposes it through `/admin/models/load`. Two checkpoints loaded into one process do not answer the same as either alone: Llama-3.2-1B then OLMoE gave three different OLMoE continuations across three runs, none of them the stable answer OLMoE gives alone, while paged and contiguous agreed with each other every time. `crates/frink-models/tests/paged_metal_parity.rs` runs one model per child process because of it. NARROWED, not proven: `resident_weight_buffer` in frink-metal/src/gpu.rs:5564 keys the cache on `(host pointer, length)` and SKIPS the fingerprint check whenever `mmap_backed` is true. The comment argues an mmap-backed range cannot be reused because the cached entry holds an `Arc<ResidentMmapFile>`, which is true until the budget path fires: it drops the whole cache, releases those Arcs, and a later mapping can then land on the same address. FRINK_METAL_WEIGHT_CACHE_BYTES defaults to usize::MAX so that path is rare, which fits a bug that is nondeterministic rather than constant. Whoever takes this must reproduce it before believing the mechanism. (2) THE METAL MoE PATH NEVER RECORDS `activation_counts`. OLMoE shows 128 expert selections per layer on CPU and 0 on Metal, so `placement_plan`'s observed hotness is always empty there. It degrades rather than corrupts, since `has_observations` is false and placement falls back to size order, which is why this is low priority and not silent damage. (3) `run-kimi` FRAMES NOTHING. crates/frink-cli/src/main.rs:1263 reads tokenizer_config.json for special tokens only and hands the prompt to `kimi_generate` unframed, while the server's Kimi loader reads `chat_template` from that same file. The two Kimi paths genuinely disagree about what a prompt looks like"
    status: pending
  - id: merge-discipline
    content: "NOT A FEATURE, the rule every wave follows, written down because ignoring it has already cost this project real work. (1) One agent owns a file. If two waves would touch the same file they go in different waves, and the plan is wrong if they do not. (2) Every agent branches from current origin/main and pushes; nobody merges to main, the maintainer does, one branch at a time, running the full gates between each. (3) A merge that conflicts on a decode send site or a KV write is RECONCILED, not resolved: the SSE orphan deadline and the resumable emitter both wanted the same five lines and taking either side alone would have shipped a stream the timeout kills 30 seconds after a client disconnects. (4) A branch's own plan-frontmatter edit is a claim, not evidence. Verify against the code before believing a status, because the radix-cache commit marked paged-decode-path complete while it returned wrong tokens on Metal. (5) No agent runs benchmarks. Wave 4 owns every number"
    status: pending
isProject: false
---

# Completion push

> The seven plans in `docs/plans/` have 56 open todos between them. This
> file is the schedule for closing them: what runs together, what waits,
> and what cannot be done here at all.
>
> It contains no engineering. Each item points at the plan that owns the
> work, and the todo text in that plan is where the measurements and the
> design arguments live.

## Where the 56 are

| Plan | Open | Shape of what is left |
|---|---|---|
| `llama-cpp-parity-push` | 16 | Mostly one thing: CPU performance. Seven kernel items, three tooling, two coverage, three that can only close last. |
| `amd-strix-halo` | 14 | Four need the hardware. Ten do not. |
| `freetoken-parity` | 13 | The paged/radix path, MoE execution, two architectures, one download. |
| `dflash-speculative-decoding` | 9 | All nine sit behind one go/no-go. |
| `frink-ui` | 2 | Server-side conversations, and the Tauri shell. |
| `serving-and-tiered-kv` | 1 | Time-debt scheduling. |
| `one-binary-serve` | 1 | The cross-target CI gate. |

## Why the waves are cut this way

Agents collide on **files**, not on topics. Two branches that both edit
`crates/frink-models/src/decoder.rs` produce a merge nobody can review,
and this project has already had one branch silently revert three others
because its copy of a file predated them.

So the rule is one owner per file per wave. Where that forces work to
wait, it waits. Where it does not, five agents run at once.

The second cut is dependency. Wave 0 exists alone because the paged path
returns wrong tokens on GPU today, and three items in wave 2 build
directly on it. Wiring a radix cache onto a decode path that computes
the wrong answer would produce a faster wrong answer.

## What wave 4 is for

Every measurement item is in the last wave, run sequentially on a quiet
machine with nothing else building. That is not fussiness. `frink bench`
refuses a host whose 1-minute load is above 2.0, a host the OS is
thermally throttling, and a host without the free memory to hold the
model, because a paged or loaded run reports a real-looking number for
work the disk or the scheduler did.

`close-all-red-rows` is the definition of done for the parity plan. If
wave 1's CPU work does not move the CPU rows, the honest outcome is to
reopen those items rather than close the plan.

## The seven that cannot be finished here

Named in `blocked-on-hardware-or-downloads` above so no wave burns an
agent on them:

- **`real-moe-checkpoints`** needs somebody to download a published
  `UD-*` or dots1-family checkpoint. The IQ tiers and the MoE routing
  bias are bit-exact against llama.cpp's own dequantization, and have
  never been run end to end on a real published file.
- **`persistent-gpu-expert-cache`** needs a CUDA host.
- **`dflash-checkpoint-reality`** needs a published drafter checkpoint to
  exist in a format frink can load. The other eight dFlash items are
  downstream of that answer and must not start before it.
- **Four Strix Halo items** need a Strix Halo box. The other ten do not,
  and they are wave 5.

## Order

```
wave 0  ────────────────► paged GPU correctness           (1 agent)
                              │
wave 1  ══════════════════════╪═══════════════════════════ (5 agents, parallel)
        cpu kernels │ cpu attention │ ui │ tooling │ coverage
                              │
wave 2  ──────────────────────┴───────────────────────────  (3 agents)
        radix+batcher │ time-debt │ moe execution
                              │
wave 3  ──────────────────────┴───────────────────────────  (2 agents)
        gemma4 + minimax │ child log stream
                              │
wave 4  ──────────────────────┴───────────────────────────  (sequential, quiet host)
        suite validate │ quality gates │ close red rows │ bench-bw

wave 5  ═══════════════════════════════════════════════════ (1 agent, any time)
        x86 half of strix halo
```

Wave 5 runs whenever there is a slot. It owns no file the others touch.
