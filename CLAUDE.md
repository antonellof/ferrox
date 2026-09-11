# CLAUDE.md

Guidance for agents working in this repo.

## What this is

Pure-Rust GGUF / MoE inference engine: mmap loaders, quantized CPU +
Metal + CUDA kernels, OpenAI-compatible `ferrox-server`.

**The goal is to be the Rust alternative to llama.cpp**: same models,
same command shapes, same or better performance, on the hardware people
actually own. `docs/plans/north-star.md` is the ranking every other plan
is read through, and `docs/plans/README.md` is the index.

Honest position, re-audited 2026-09-11. **37** architectures run with
evidence (`capability::AUDITED_GENERIC_GQA`), 4 more have dedicated
engines, and everything else REFUSES. The "loads and is WRONG" class is
closed: the generic path is opt-in, so an unaudited architecture stops
instead of guessing.

The 20 unaudited refusals are now TRIAGED, and the refusal says which of
three things is missing: **0 are a fixture away, 0 are one match arm
away**, 19 need new code, 1 is unknown with the question stated. Five
one-match-arm rows closed on 2026-09-02, seven fixture-away rows on
2026-09-03, `gemma`, `hunyuan-dense` and `ernie4_5-moe` on 2026-09-09,
and `olmo2`, `exaone4`, `chatglm`, `qwen`, the three Granite rows and
`olmo` on 2026-09-10, and `exaone-moe` on 2026-09-11, each with a
libllama-golden fixture, which is what moved 46 to 41 to 34 to 31 to 29
to 28 to 25 to 22 to 21 to 20; the step from 28 to 25 was moving the
three alias rows off the generic path rather than a closure. `minicpm`
moved too and is not in that count: it was refused BY NAME, never as
unaudited, so it raises the audited number without lowering the
refusing one. `smollm3` and EXAONE-4 32B closed with `exaone-moe` and
are the same case, one a DedicatedOnly refusal and the other a refusal
by name.
BOTH cheap classes being EMPTY is the honest headline: nothing still
refusing is one fixture or one arm away, so every row that is left
needs a different graph.

**On 2026-09-10 the NEW CODE column moved for the first time**, three
times: 26 to 24, 24 to 21, then 21 to 20, and on 2026-09-11 a fourth
time, 20 to 19. The first two took several rows at once for the same
reason, and it is the lesson: each found ONE cause behind several
refusals. The fourth did too and the column hides it: the per-layer
RoPE gate closed THREE refusals and only `exaone-moe` was in the column.

`olmo` is the exception that says what the rule is really made of. It
closed ALONE, and before writing a line of code the question "what else
shares this cause" was answered by MEASUREMENT rather than by hope:
every `build_norm` call in all 140 of llama.cpp's `src/models/*.cpp`
graphs was scanned for a null weight argument, and all three hits are
`olmo.cpp`. So there was no second row to take, and knowing that in
advance is worth as much as a shared cause would have been -- the six
rows the search was aimed at (`openelm`, `bitnet`, `arcee`, `mellum`,
`nanbeige`, `deci`) are now checked-and-recorded rather than
still-plausible. What IS shared is the LayerNorm *function* with a
learned weight: `dbrx` plus the `nemotron` / `orion` / `stablelm` /
`codeshell` / `jais2` / `starcoder` / `starcoder2` / `phimoe` bias
group. None of them is one variant away, because each refuses for more
than the norm, so that variant was deliberately NOT written --
`capability::NON_PARAMETRIC_LAYER_NORM` records the whole finding where
the next person will look.

`olmo2` and `exaone4` closed TOGETHER, because they are ONE residual
topology: no `attn_norm` and no `ffn_norm` tensor, both sublayers
reading the raw residual, each branch's output normed before its
residual add. Reading `olmo2.cpp:45-52,92,160-182` beside
`exaone4.cpp:60-67,118,152-169` gives the same graph line for line, so
they got one implementation (`ferrox-models/src/norm.rs`) and a
fixture each. One sub-case stays refused by name rather than swept in --
an `olmo2` with both a window and a RoPE scaling; EXAONE-4 32B, whose
full-attention layers get no RoPE at all, was the other and closed on
2026-09-11 (below). `olmo` (OLMo-1) is a
THIRD shape, pre-norm with a non-parametric LayerNorm, and closed on
2026-09-10 as a third variant of the same enum; the type is
`ferrox-models/src/norm.rs`'s `NormOp` now rather than `PreNorm`,
because `Decoder::final_norm` is one too -- OLMo-1's final norm has no
weights either, and the fused Metal stacks had `Some(&self.final_norm)`
written into them unconditionally. Its `attention.clamp_kqv` stayed a
REFUSAL: `llama-graph.cpp:1611-1652` clamps Q, K and V by it,
`conversion/olmo.py:23-25` really writes it for OLMo-7B-Twin-2T and
OLMo-1.7-7B, and a second fixture measures that llama.cpp's own logits
move when it is present, so the row is admitted for the checkpoints it
covers rather than all of them.

`granite`, `granitemoe` and the `granite-moe` alias closed together for
the same kind of reason: they differ in the FFN, not in the four scalar
multipliers llama.cpp applies to them, and `granite-moe.cpp` has no
graph of its own at all (`models.h:1583-1591` is
`using graph = llama_model_granite::graph`).
`ferrox-models/src/scalar_multipliers.rs` implements the multipliers
once, parameterised, and `capability::unsupported_scaling_keys` -- which
still refuses those keys everywhere else -- is now DERIVED from that
same table instead of restated beside it. That derivation found a live
gap on the way past: the whole Gemma family was exempted from all four
keys while reading NONE of them, so a hand-written
`gemma3.residual_scale` would have loaded and been ignored. The
expensive half was `residual_scale`: it multiplies both branch outputs
of every layer, so `decoder.rs`'s EIGHTEEN hand-written residual adds
collapsed onto one function taking the scalar as a parameter, and the
four Metal eligibility checks gained one shared predicate rather than
four spellings of it. MiniCPM ran the same llama.cpp graph
object -- `models.h:1594-1601` is
`using graph = llama_model_granite::graph` -- and closed on 2026-09-10
on the defaults hook that was predicted: `minicpm.cpp:5-7` assigns 12.0,
`1.4/sqrt(n_layer)` and `256/n_embd` BEFORE `:12-14` lets the file
override them, so a MiniCPM export declaring nothing is still scaled by
all three and a key-PRESENCE gate has nothing to see. That is why it was
refused by name rather than detected. `MultiplierDefaults` is a FIELD of
the same table, so a default for a key the graph does not apply is not
expressible, and the fixture that evidences it declares NO key at all --
the only fixture shape that can tell the hook from its absence. A second
one declares all three and pins that the file still wins, because a hook
merged the wrong way round would agree with llama.cpp on exactly the
files that prove it exists. Command-R is still not close, because its
blocker is a parallel residual over LayerNorm rather than the
multiplier.

`exaone-moe`, `smollm3` and EXAONE-4 32B closed together on 2026-09-11
on the PER-LAYER RoPE gate, and the claim that they are one cause was
read before it was assumed: `exaone4.cpp:116` is
`use_rope = is_swa(il) || swa_type == NONE`, `exaone-moe.cpp:136,155-161`
is `is_swa(il)` around the same two `ggml_rope_ext` calls, and
`exaone-moe.cpp:4` pins `swa_type` to STANDARD, which makes the second
disjunct false -- identical, not similar. `smollm3.cpp:5,69` is another
variant of the same enum, `(il + 1) % 4 != 0`. llama.cpp gates rotation
this way in SIX architectures, always from a literal and never from a
GGUF key, and `ferrox-models/src/rope_layers.rs` is one table for all
six; `smallthinker`, `afmoe` and `llama4` are in it and still refuse
for other things, and their verdicts now say so. The durable part is
the type, not the arms: `ModelConfig::layer_rope` returns
`Option<(base, divisors)>`, so no rotation site can take the pair
without answering whether to rotate -- the CPU head loop, the YaRN
`attn_factor` (an argument to `ggml_rope_ext`, so it goes with it), the
four per-layer Metal launches (which lost a loose base/divisor
parameter pair for one `LayerRope`) and both fused Metal stacks, whose
RoPE dispatch had been written in unconditionally the way OLMo-1's
final norm had. A 64-layer fixture evidences the 32B because
`exaone4.cpp:4` tests equality, and building the three found two more
things: EXAONE-4 1.2B must IGNORE a window its file declares
(`exaone4.cpp:4-14` reaches `set_swa_pattern` only at 64 layers, so a
window key below that is dead metadata -- `capability::
swa_disabled_by_arch` carries it beside `phi3`), and
`nextn_predict_layers` was refused NOWHERE: MTP blocks are inside
`block_count` and llama.cpp skips them, so a real EXAONE-MoE export
with an MTP head would have run its speculative head as two more
decoder layers. It is gated on the value now, because the converter
writes the key as `0` for the sizes that have no head.

Building those fixtures keeps finding defects worth more than the
admissions. The last three UNKNOWN rows -- `mistral`, `mixtral`, `yi` --
closed on 2026-09-10 by turning out NOT TO BE ARCHITECTURES: libllama
refuses all three strings (`unknown model architecture: 'mistral'`,
measured), every real checkpoint of all three declares `llama`, and the
rows had been sitting on the generic path with NEOX RoPE while `llama`
is NORM -- a wrong-pairs rotation that `rope_layout_matches_llama_cpp`
could not see, because a name absent from llama.cpp's table is a
`continue` there. `chatglm`'s arm was predicted to close `qwen` too and
closed it only halfway: the fused `attn_qkv.bias` really is shared, but
`qwen.cpp:33-35` also halves every FFN matrix, which cost no logits and
made `expert_ffn_dim` twice the real width. `plamo3` could never have
loaded a real checkpoint, because it is the only architecture upstream
whose post-norms use the two-argument `LLM_TN` overload and ferrox asked for the wrong spelling;
a gate refused every file carrying `attention.sliding_window_pattern` as
unimplemented while the feature was already implemented, which made the
loader's own read of that key unreachable; llama.cpp's own
`ernie4_5-moe` tensor loader (`ernie4-5.cpp:49`) has no interleave step
in it while its graph (`ernie4-5-moe.cpp:64`) does, so no interleaved
ERNIE-4.5 MoE checkpoint can be loaded by llama.cpp at all and that arm
had to land as a refusal; and `hunyuan-dense`'s verdict cited a
converter line (`conversion/hunyuan.py:356`) that belongs to
`hunyuan-vl`, not to it; and no Granite converter writes
`{arch}.rope.scaling.finetuned`, so the RoPE-off switch llama.cpp reads
out of that key (`granite.cpp:33-35`) can only be reached by a
hand-written file -- which is why it landed as a refusal with a fixture
that actually carries the key rather than as a gate nobody could fire.
`unaudited_triage` carries the verdict and the
llama.cpp line that decides it. llama.cpp hand-writes 140
per-architecture graphs; `decoder.rs` is 6752 lines and that is why the
counts differ.

Do not read the architecture catalog as a support matrix. `ferrox
parity` is the oracle: its tokenizer half matches llama.cpp on every
local checkpoint libllama can load, and its logit half MATCHES on
Q8_0/IQ4_NL while DRIFTING on K-quants — for a known reason that is not
a ferrox bug (`docs/plans/llama-cpp-gap-inventory.md` §10).

Capabilities: `docs/FEATURES.md`. Models & speed ledger: `docs/MODELS.md`,
`benchmarks/RESULTS.md`. Planned: `docs/ROADMAP.md`.

| Doc | Role |
|---|---|
| `docs/FEATURES.md` | capabilities overview |
| `docs/CLI.md` | `ferrox` flags + `ferrox chat` |
| `docs/MODELS.md` | what runs / what doesn’t |
| `docs/API.md` | OpenAI compatibility matrix |
| `docs/AGENTS_COOKBOOK.md` | point IDEs at `ferrox-server` |
| `docs/CONFIG.md` | env vars |
| `benchmarks/RESULTS.md` | tok/s vs llama.cpp (Gap = llama/ferrox); `ferrox bench` ledger |
| `benchmarks/README.md` | how `ferrox bench` / `llama-bench` is measured |
| `docs/ROADMAP.md` | planned work |
| `docs/plans/README.md` | **the plan index and priority order** |
| `docs/plans/north-star.md` | the goal, and how plans are ranked against it |

## Commands

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all

cargo build --workspace --features cuda
cargo test -p ferrox-cuda --features cuda -- --ignored   # needs GPU

cargo build -p ferrox-cli -p ferrox-server --features metal
cargo test -p ferrox-metal --features metal -- --ignored   # needs Metal

# Completion (also: ferrox run -m …)
./target/debug/ferrox -m model.gguf -p "Hi" -n 64 --temp 0 --no-cnv
./target/debug/ferrox -m model.gguf -p "Hi" -n 64 --ngl 99   # Metal

./target/debug/ferrox presets | archs | caps | inspect <gguf> | inspect-plan <gguf>
./target/debug/ferrox smoke <preset> | run-kimi <dir>
./target/debug/ferrox chat --url http://127.0.0.1:8383   # needs ferrox-server

FERROX_MODEL_PATH=model.gguf FERROX_ADDR=127.0.0.1:8383 ./target/debug/ferrox-server

# Bench vs llama-bench (no HTTP). Models: benchmarks/suite.json
./target/release/ferrox bench -m model.gguf -p 512 -n 128 --compare
./target/release/ferrox bench --suite --fit-host --skip-missing
./target/release/ferrox bench --suite --id llama32_3b_q4km --backend metal
./target/release/ferrox bench --render
```

Fixtures and golden values were generated and cross-validated with
independent NumPy references.

Tests are mostly `#[cfg(test)]` next to the code. Integration:
`crates/ferrox-models/tests/gguf_roundtrip.rs`. Never un-ignore CUDA /
Metal hardware tests without a real GPU.

## How to write code here

**Keep files small, modules narrow, and the binary light.** This is not
style preference, it is the repo's most expensive lesson.

Re-measured 2026-09-10, against the 2026-09-03 numbers:

| File | Was | Now | |
|---|---|---|---|
| `ferrox-server/src/lib.rs` | 9628 | **9775** | grew |
| `ferrox-metal/src/gpu.rs` | 8935 | **9018** | grew |
| `ferrox-metal/src/attn.rs` | 9331 | **8715** | shrank |
| `ferrox-quant/src/lib.rs` | 8239 | **8242** | flat |
| `ferrox-models/src/decoder.rs` | 6752 | **6780** | grew |

**One of five shrank, and the rule still lost on balance.** `attn.rs`
gave up 616 lines only because a change was made to it and the split
came first, which is the rule working as written. `lib.rs` gained 147
adding four sampler steps to three routes, and `decoder.rs` crept up
again. Nothing shrinks on its own.

What did work is not in that table, because the files left it:
`ferrox-quant/src/repack.rs` was 6446 lines and is now a directory of
ten; `ferrox-models/src/sampling.rs` went 1569 to 894 with four
submodules beside it; `ferrox-cuda/src/mul_mm.rs` is 865 with its kinds
in two directories. Each of those splits happened because somebody was
about to add to the file and split it first. That is the whole
mechanism, and it is the only one that has ever worked here.

Those files are why llama.cpp has 140 architectures and ferrox has 37
proven. Adding a model means editing a 6750-line file, so nobody adds
one. The same decode layer used to be written out about ELEVEN times
across `decoder.rs` and `attn.rs`, which has already lost EIGHT model
features one at a time, each silently:
`attention_scale`, `post_attn_norm`, `post_ffn_norm`, gpt-oss `o_bias`,
`gpt_oss_ffn`, the four the Metal MoE decode stack ignored, and the
four-way drift of the GPU-router eligibility check, where the prefill
sites tested three conditions, the fused decode two and the whole-stack
decode NONE. A copy diverges from its original and nothing notices.

**THIS IS THE DOMINANT BUG SHAPE IN THIS REPO, and 2026-09-01 found a
dozen more instances of it in one day** — not all of them copied code.
The general form is TWO STRUCTURES THAT MUST AGREE ABOUT ONE THING,
WITH NOTHING ENFORCING IT: two spellings of a GGUF key (`unsupported_
feature_keys` gated on one no converter writes, so it never fired); two
copies of a default (the response cache restated every `unwrap_or`
independently of the sampler); four hand-written `SamplingParams`
literals in one file; three tables that had to agree about a Metal
kernel's threadgroup geometry (a correct kernel returned zeros for half
its rows); a wire struct and a sampler that silently disagreed about
which fields exist (SIX parameters accepted and ignored).

2026-09-04/05 added three more, all found by measuring rather than
reading: a benchmark receipt carrying `backend: "cpu"` beside
`backend_active: "Metal"`, in the same file, with nothing comparing the
two, for **13 of 13** published CPU rows; `KINDS` and the host-check
tool's fixed shape list, where adding the K-quants made the tool panic
on the first one and check NOTHING for a day while still exiting green;
and `FERROX_CPU_INT_DOT`, a default that was correct on the
architecture its kernels were written for and cost the other one 4x to
8.8x of decode.

The durable fixes are never the individual patches. They are the places
where disagreement now fails to COMPILE or turns a test red: an
exhaustive destructure with no `..`, one predicate the four call sites
share, a derived table instead of a restated one, and a test asserting
every refused key is one a converter actually writes.

Rules that follow from that:

- **A new file beats a new section.** If a change would push a file past
  roughly 1000 lines, split it first, then make the change.
- **One concept per module.** A module named after a noun that holds
  three unrelated things is three modules.
- **Never copy a code path to vary it.** Parameterise the original. The
  precedent that works: `forward_multi_seq` takes a `MultiSeqKv`
  parameter rather than having a paged twin. The precedent that failed:
  `forward_token_paged` was a copy, and lost five features -- it is now
  collapsed onto one `attn_block` taking a `KvStep`.
- **Dead code is a liability, not an asset.** Delete on sight unless it
  serves a named roadmap theme; if it does, wire it or say where it is
  going. `ferrox-edge` was 5,400 uncalled lines and is now dissolved.
- **A gate that cannot fire is worse than no gate**, because it reads as
  coverage. Check that a refusal's condition is reachable at all: this
  repo shipped one keyed on a GGUF spelling nothing writes.
- **No new crate for something one crate uses.** `ferrox-edge` became a
  crate instead of an integration and half of it was never called.

Rust specifics this repo holds to:

- `cargo clippy --workspace --all-targets -- -D warnings` is a gate, not
  advice. Also run it `--release`: `debug_assert!` type-checks its
  argument in release, and a `#[cfg(debug_assertions)]` method called
  from one broke every release build while all of CI stayed green.
- Prefer borrowing to cloning on any path that runs per token. Hoist
  feature probes out of loops: `is_aarch64_feature_detected!` ran 131k
  times in one Mistral-7B projection before it was hoisted.
- `unsafe` needs a `// SAFETY:` comment stating the invariant, and a
  scalar twin it is checked against. Every SIMD arm here has one.
- Return `Result` and name what is missing. A model this engine only
  partly implements must STOP, never compute something else. A refusal
  is coverage, not a defect.
- Tests live in `#[cfg(test)]` beside the code. A test that cannot fail
  is not a test: sabotage it once and confirm it goes red.
- **Confirm the sabotage LANDED.** A mutation that did not apply is
  indistinguishable from a test that holds. One sabotage here passed
  and proved nothing because `cargo fmt` had wrapped the target line
  across three lines, so the patch never matched. Grep the file for the
  mutated text before believing a green run, and if a test survives the
  first sabotage attempt, suspect the sabotage before believing the
  test.

## Measuring, and not fooling yourself

Every performance claim in this repo has to survive these. They are
here because each one was broken in a single week, and each break cost
either a wrong number in `benchmarks/RESULTS.md` or a merged-nothing
PR.

- **Rent a box; do not measure on this laptop.** `suggestd` holds ~97%
  of a core on it indefinitely and respawns hot when killed. CPU and
  CUDA rows come from vast.ai; the M2 Pro is for Metal, where it is the
  only hardware that can run the backend. Pick offers where
  `cpu_cores_effective == cpu_cores`, so no co-tenant shares the CPU.
  Four instances cost $0.49. Destroy them the moment the receipts are
  copied back, and confirm with `vastai show instances`.
- **A load average cannot see one busy core.** `ferrox bench`'s
  `--max-load 2.0` guard passed for an entire day while one of six
  cores was pegged, because one core does not move a six-core average
  enough to trip it. The guard is necessary and not sufficient: check
  `ps -eo pcpu,comm | sort -rn | head` too, and treat any process above
  ~90% as disqualifying.
- **One instantaneous sample is not a measurement.** "CUDA decode runs
  at 36% GPU utilization" came from a single `nvidia-smi` taken AFTER a
  bench had finished, so it caught an idle moment. Sampled five times
  DURING the run, the real figure was 86-93%. That one number sent a
  day of work at a host-side cost that did not exist and produced a PR
  measured 22% SLOWER. Sample repeatedly, and sample while the thing
  runs.
- **Verify the flag did what it says, from the artifact.** `ferrox
  bench --n-gpu-layers 0` is documented as forcing CPU and did not:
  the backend is decided once per process and cached, so the flag
  arrived too late. Read `backend_active` in the receipt, not the flag
  you passed. A receipt whose label disagrees with the backend that ran
  is now refused at write time and asserted over the committed set,
  because this was found only after publishing 13 wrong rows.
- **Check that the lever can reach the target before pulling it.** At
  the (wrong) 36% figure, removing EVERY host round-trip was worth at
  most 1/0.36 = 2.8x against a 9x-17x gap. That arithmetic was written
  down before the code was, and reading past it cost the PR. If the
  best case does not close the gap, the diagnosis is incomplete
  whatever else is true.
- **Interleave A/B when comparing two builds**, and report the raw
  sequence. `main, branch, main, branch` catches drift that two
  sequential runs hide.
- **A gap column cannot show a missing kernel.** A CUDA K-quant prefill
  read 4.88 tok/s against llama.cpp's 1586.80 and looked like a
  performance problem; there was no GEMM at all, and the fallback still
  answered correctly. Check coverage before profiling.
- **Check what the kernel is actually limited BY, before optimising
  anything in it.** Convert the measurement into the resource: for a
  decode matvec, `weights_bytes * tok/s` is achieved memory bandwidth,
  and that number is one line of arithmetic. ferrox reaches **5.3%** of
  an RTX 3060's 360 GB/s where llama.cpp reaches **60.4%**, so decode
  is limited by memory-request concurrency. A port of llama.cpp's
  `dp4a` inner loop was written, verified correct on hardware, and
  measured **under 1% faster**, because four MACs per instruction buys
  nothing in a kernel that is waiting on loads. The source diff between
  two kernels tells you what is different; it does not tell you which
  difference is the limit.

## Working with agents

- **One worktree per agent.** Two agents in the same checkout collided
  here: one detected another's half-finished refactor, backed its own
  work out, and redid it in isolation. Use `isolation: "worktree"`.
- **Do not redo an agent's task while it runs.** A narrower, better
  version of a deletion was in flight while the same deletion was
  attempted by hand; the hand version removed a live hardware test with
  it.
- **Agents do not tag, publish, force-push, rent hardware, or
  benchmark.** They implement and open a PR; verification on real
  hardware is the parent session's, and a kernel merges only after
  `cargo test -p ferrox-cuda --features cuda -- --ignored` has run on a
  GPU.

## Architecture

```
ferrox-gguf + ferrox-quant
        → ferrox-core (WeightMatrix, RoPE, GQA, KV; optional cuda/metal)
        → ferrox-moe
        → ferrox-models (loader, Decoder, Kimi/GLM/DS4 stacks)
        → ferrox-cli / ferrox-server

ferrox-api  (routes + wire DTOs, serde-only) → ferrox-server + clients
```

**The FreeToken port** (Apache-2.0; see `docs/THIRD_PARTY_NOTICES.md`,
which is a licence obligation and must stay accurate) is a Rust port of
FreeToken's host-side decision logic. It used to be a crate of its own,
`ferrox-edge`; it now lives in the crates that use it, and the parts
nothing would ever use are deleted.

- **`ferrox-core`** holds the MoE expert-residency half, beside
  `expert_store`: `expert_cache`, `expert_slots` (the `SlotDevice`
  seam), `expert_pool` (its CUDA implementation), `expert_budget`,
  `qstar`, `bench_profile`, `residency`, `placement`. `expert_store` is
  the SINGLE holder of the expert byte budget -- on unified memory two
  budgets are the same RAM counted twice.
- **`ferrox-server::policy`** holds the serving half: the two parsers,
  the radix prefix cache, anchor/window slide, scheduler, serving stats,
  maintenance, pool, rebuild, outbox, footprint, effort probing.

Wired today: the parsers, the stop-string withhold rule, the radix
prefix cache over paged KV, effort probing, stats, maintenance, outbox,
footprint, and the scheduler's status reporting. STILL GROUNDWORK, and
this is the gap that matters most: the whole `ferrox-core` expert
residency stack holds the policy for running a model larger than memory
(`docs/plans/out-of-core-moe.md`) and nothing executes it except a
compile-only CUDA pool whose hardware test is `#[ignore]`d.

Anything in `policy` with an unwired half names the roadmap item that
would close it, at its declaration in `policy/mod.rs`. That
`allow(dead_code)` list is meant to be read as a to-do, not as cover.

Load path: GGUF mmap → keep quantized → fused dequant+dot →
RMSNorm → GQA(+RoPE) → MoE/dense FFN. Serving: `FERROX_MODEL_PATH`
GGUF or Kimi dir; generation on `spawn_blocking`.

Presets `glm_5_2` / `deepseek_v4_pro` / `kimi_k3` are sketches,
not proof of real-checkpoint support. `test_*_fixture` presets match
Python test GGUFs only.
