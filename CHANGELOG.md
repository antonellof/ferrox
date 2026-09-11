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

## [Unreleased]

## [0.21.0] - 2026-09-11

### Fixed

- **The Metal benchmark ledger booked the lm_head's GPU time as host
  time.** #149's "host = wall minus GPU" used the GPU time of ONE
  command buffer per token; a sampled decode token has two, and the
  lm_head's had no timing tag, so 1.2 ms (Llama-3.2-1B) to 2.8 ms
  (Gemma-2-2B) of GPU work read as a 26-29% host share that did not
  exist. `FERROX_METAL_GPU_TIMING=1` clocks encode, GPU and submit
  latency for every submission from one clock, and a submission cannot
  be timed with a phase left out. Measured encode is 2-3% of wall, so
  the argument-packing lever the issue named was retired unbuilt;
  the correction moved the worst Metal row from "host" to "kernel".
- **`/v1/rerank` scores from the published `ms-marco-MiniLM-L6-v2` GGUF
  were fifty times too small.** llama.cpp's converter drops
  `pooler.dense.*` for every BERT, so the file scores `classifier(cls)`
  where HuggingFace scores `classifier(tanh(pooler(cls)))`; orderings
  were right and the scale was not, which is why a threshold copied from
  another engine never fired. `ferrox splice-pooler -m in.gguf
  --safetensors model.safetensors -o out.gguf` writes the pooler back
  under llama.cpp's own tensor names (llama-server loads the result and
  applies it too), tied to the checkpoint by the classifier both files
  hold rather than by a name the published file gets wrong. Seventeen
  pairs within 0.051 of HuggingFace afterwards, orderings identical.
- **The GLM, MLA and hybrid dedicated loaders would have run a
  checkpoint's MTP block as one more decoder layer.** `glm4moe`,
  `glm-dsa`, `glm4`, `deepseek2`, `qwen3next`, `qwen35` and `qwen35moe`
  all subtract `nextn_predict_layers` from `block_count` in llama.cpp,
  and their converters append the block INSIDE `block_count`
  (`conversion/glm.py:99`, `deepseek.py:457`); the three loaders read
  `block_count` verbatim and no gate stood in front of them, so a real
  GLM-4.5 or DeepSeek-V3 export would have loaded, run its NextN block
  as a 47th or 62nd layer, and left the `nextn.*` tensors silently
  unread. All four dedicated loaders take their layer count from
  `ferrox_models::mtp_blocks::trunk_layers` now; the MLA loader has a
  trunk-only fixture that fails without it. Not observed on a
  checkpoint: found by reading the seventeen graphs that read the key
  against the loaders that own them.
- **A reasoning model that ran out of `max_tokens` inside its thought
  showed thinking and then nothing.** Studio sent `max_tokens: 512` on
  every request, DeepSeek-R1-Distill spends about 900 tokens thinking
  on an ordinary question, the server correctly returned `finish_reason:
  "length"` with empty content, and the UI rendered that as silence.
  The default is now no cap (the context is the limit, llama.cpp's
  `n_predict: -1`), a saved 512 from the old default is migrated away,
  a `length` finish is shown as a cut-off with the token count, and a
  **Continue** button carries on from where it stopped.
- **The conversation store dropped a reasoning model's chain of
  thought.** An answer that was entirely `reasoning_content` persisted
  as `content: ""` and reloaded as an empty turn. The store keeps
  `reasoning_content` beside `content`; records written before the
  field read back unchanged.
- **A leading-dense MoE file that omits `expert_shared_count` loaded
  with its shared experts unread.** The inference probed `blk.0` for a
  `_shexp` tensor, and layer 0 of such a model is dense. `laguna.cpp:20`
  assigns the count before reading a key its converter never writes,
  so a real Laguna export would have run without its REQUIRED shared
  expert on every MoE layer. The probe is the first MoE layer now.
- `afmoe` scales its embeddings by `sqrt(n_embd)` from arithmetic, the
  only non-Gemma graph that does (measured over all 140); the Gemma
  family match in the loader is a table with two rows now.

### Added

- **Every one of the 16 comparable Metal `tg128` rows is now faster
  than llama.cpp** (gap 0.60x-0.96x; prefill 0.99x-1.09x), re-measured
  on a quiet host with `ferrox 0.20.0` receipts. Gemma-2-2B decode went
  from 1.11x to 0.94x on three kernel rewrites none of which is
  Gemma-specific: RoPE (one thread per rotary pair, one templated
  NORM/NEOX kernel where there were four sources; 61.6 to 3.4 us),
  RMSNorm (float4 loads, at most two loop trips; 14.2 to 4.7 us),
  FA-vec decode at d=128/256 (compile-time tile loops so K and V loads
  overlap; 38.9 to 17.6 us with the softcap), and the final logit
  softcap moved off the host into the lm_head's command buffer as an
  epilogue (0.65 ms of scalar `tanh` per Gemma-2 token, gone).
  `FERROX_METAL_KERNEL_TIMING=1` attributes a token's GPU time per
  dispatch kind, which is how the three were found: the matvecs, 80% of
  the stack, were already at parity. `ferrox verify` token-identical and
  `ferrox parity` unchanged on four models. Closes #149.
- **The Studio Thinking block times the thought.** A live `Thinking for
  12 seconds` while the model reasons, collapsing to `Thought for 15
  seconds` (`1 min 20 s`, `1 h 5 min`) once the answer begins, one click
  away; a thought cut off with no answer stays open under the Continue
  banner and a continuation keeps counting. The time is the wall-clock
  between the first `reasoning_content` delta and the first `content`
  delta, persisted as `reasoning_ms` beside `reasoning_content`; older
  records load and show their thought with no time.
- **`grok` and `dbrx` are audited**, 37 to 39, each by extending a seam
  from the day before by one column. Grok-1 on the MiniCPM defaults
  hook: `grok.cpp:5-12` seeds seven hyper-parameters before the file
  may override them, `logit_scale` is a multiply there and
  `attention.output_scale` is "pre-scale Q, then softcap", so it
  resolves into slots that existed; two of the seven are read by
  llama.cpp and applied nowhere (measured), and ferrox neither applies
  nor refuses them. DBRX on `NormOp::LayerNorm` (weight, no bias, the
  variant OLMo-1 deliberately left unwritten until a caller arrived),
  a REQUIRED `attention.clamp_kqv` (the three QKV-bias loops collapsed
  onto `decoder/qkv_bias.rs` first, so the clamp is one line rather
  than three), and `norm_sites.rs`, one table for which tensor feeds
  which norm site, because `blk.N.attn_output_norm` is DBRX's pre-FFN
  norm and Grok's post-attention norm. The clamp closed OLMo-1's
  `clip_qkv` checkpoints with it. KL 4.7e-10 / 1.6e-10 (Grok, at the
  GeGLU f16-table tolerance, measured), 3.4e-12 (DBRX). Grok-2's
  parallel dense FFN stays refused by name from a fixture that has it.
- **`arcee`, `deci` and `openelm` are audited**, 39 to 42. `arcee` is
  the ungated ReLU-squared FFN, spelled as `GluAct::Reglu` with the gate
  aliased to the up matrix rather than a fourth expert shape; on the
  way, six Metal launch sites that derived `gelu = !is_swiglu()` (a
  third activation would have run as GELU on all of them) now refuse on
  `None` from one function. `deci` and `openelm` are one seam,
  `layer_shapes.rs`: llama.cpp reads `head_count`, `head_count_kv` and
  `feed_forward_length` as scalar-or-array for every architecture and
  ferrox carried scalars; all 140 graphs were scanned for which honour
  a per-layer value before a line was written, and the table records
  each. `AttnShape::{Gqa, Linear, Absent}` is an enum so an
  attention-less layer cannot be a zero count a loop accepts;
  `ModelConfig::new_kv_caches` replaced ninety hand-written
  `KvCache::new(config.n_kv_heads, ..)` sites and `KvCache::push`
  asserts the width. KL 2.27e-14, 1.44e-13, 7.29e-13, 1.28e-13. `plm`
  did NOT close with `arcee`: its verdict had been read from one file,
  and the diff is 150 lines of MLA attention.
- **`apertus` and `step35` are audited, on ONE seam with TWO
  activation bodies.** An FFN activation whose parameters vary by
  layer had no home: `FfnActivation` was a unit enum and every FFN
  body converted the model-wide value to a `GluAct` without knowing
  which layer it ran. `apertus.cpp:6-9` reads xIELU's `xielu.alpha_n` /
  `.alpha_p` / `.beta` / `.eps` as `n_layer`-long arrays (or a scalar
  broadcast) and `:132-138` hands layer `il`'s four to `ggml_xielu`;
  `step35.cpp:28-29` reads `swiglu_clamp_exp` / `_shexp` the same way
  and llama.cpp's generic `build_moe_ffn` / `build_ffn` clamp SwiGLU by
  layer `il`'s entry, the routed experts from one array and the shared
  experts AND the dense layers from the other. `ferrox_models::
  act_layers` reads both families as `get_key_or_arr` does,
  `FfnActivation::Xielu` / `::SwigluClamped` carry their tables,
  `ferrox_moe::GluAct` gained the two bodies (and lost `gate_fn`, a
  gate-only signature xIELU cannot fit), and `ModelConfig::
  layer_ffn_acts(il)` answers a `routed` / `dense` pair at every FFN
  body; no fused Metal kernel spells either, and both refuse through
  the predicate the launches share. Step-3.5's other blocker, a rotary
  width halved on the full layers with no key (`step35.cpp:9`), is
  `ModelConfig::rope_dim_swa` -- the two-valued `n_rot(il)` llama.cpp
  already had -- handed out per layer by `layer_rope`, which also
  SERVES a `rope.dimension_count_swa` differing from the full width
  (Laguna-XS.2's shape) where it used to refuse it; the two `_swa`
  head-width keys stay refused. Five libllama-golden fixtures: apertus
  KL 4.91e-14 (arrays) and 5.06e-14 (the scalar spelling llama.cpp
  broadcasts); step35 KL 1.59e-13 (clamped), 2.59e-13 (neither key),
  1.59e-13 (a NextN block inside `block_count`); the Laguna-XS.2
  rotary-width fixture 7.29e-14. Building them found `apertus.cpp:93,96`
  pass `NULL` for the QK-norm biases `:50,52` create, so a file
  carrying them is served with them ignored, as libllama serves it
  (measured byte-identical; `ferrox_models::unread_tensors`). 47
  architectures audited, 10 refuse, 9 of them NEW CODE.
- **`mellum` is audited, and every real EXAONE-4 32B, EXAONE-MoE and
  Olmo-3 export loads.** `{arch}.attention.sliding_window_pattern` as a
  per-layer bool ARRAY was refused for every architecture; llama.cpp
  reads it through `get_key_or_arr`, which for fifteen graphs
  (`exaone4`, `exaone-moe`, `olmo2`, ...) IGNORES the array and keeps
  the seeded period, for `mimo2` / `step35` / `gemma4` honours it and
  broadcasts a scalar as a bool, and for `mellum` / `cohere2moe` tries
  the scalar then the array. `ferrox_models::swa_layers` carries the
  three modes as one table and one enum (`All`, `Period`, `PerLayer`)
  behind `ModelConfig::layer_sliding_window(il)`, which every backend
  already asked per layer. Fixtures with the EXAONE array agreeing with
  and INVERTED against the period measure that libllama's logits do not
  move (KL 1.43e-14 both), and a Mellum whose array disagrees with the
  seed on two layers is honoured (KL 1.02e-14); a Mellum with a window
  AND a RoPE scaling, which every real Mellum2 is, stays refused by
  name. 45 architectures audited, 12 refuse, 11 of them NEW CODE.
- **NextN / MTP blocks are skipped, as llama.cpp skips them.**
  `nextn_predict_layers` was refused for every architecture on any
  nonzero value. `ferrox_models::mtp_blocks` subtracts the blocks from
  `block_count` for the seventeen graphs that read the key (measured),
  keeps refusing it elsewhere, marks the skipped tensors deliberately
  unread so the consumption gate still sees a missing term, and hands
  `block_count` rather than the trunk to the two things llama.cpp
  decides before reading the key (`exaone4.cpp:4`'s 64-layer gate, the
  per-layer shape array lengths). K-EXAONE's shape, one block after the
  trunk, has a fixture (KL 1.09e-14). `mimo2` and `step35` verdicts
  now lead with what is left: a V head width differing from K's, and
  per-layer SwiGLU clamp arrays with a half-width rotary.
- `continue_final_message` on `/v1/chat/completions`, llama.cpp's field
  and value set (`true`, `"reasoning_content"`, `"content"`): the
  trailing assistant message renders as a turn still being written, its
  thought re-opened in the family's own markers, so the model continues
  rather than starting over. **Default on, as llama.cpp's server**: a
  trailing assistant message is continued unless the request says
  `false` or the server was started with `--no-prefill-assistant`
  (llama.cpp's flag). `/v1/messages` and `/v1/responses` render through
  the same function, so the three routes hold one default rather than
  three literals. The channel-grammar families and a content
  continuation for an always-open family are 501 by name.
- `reasoning_budget_tokens` / `thinking_budget_tokens`, llama.cpp's
  sampler-level thinking budget (`common/reasoning-budget.cpp`), on
  `/v1/chat/completions`, `/v1/responses` and, as `thinking.budget_tokens`,
  `/v1/messages`; `--reasoning-budget N` is the server default. After N
  tokens of thought the closing tag is forced one token per step, so the
  answer still arrives with `finish_reason: "stop"`; `0` closes the
  block the moment it opens; `-1` is unrestricted. Measured
  token-for-token against llama.cpp master on DeepSeek-R1-Distill at
  temperature 0. Studio's sampling panel carries the field. Replaces the
  501 that refused the field by name.
  rather than starting over. Default off; the channel-grammar families
  and a content continuation for an always-open family are 501 by name.
- `reasoning_budget_tokens` / `thinking_budget_tokens` are refused by
  name (501) except `-1`, rather than silently dropped. llama.cpp
  enforces the budget in its sampler; ferrox has no such sampler yet.
- **`afmoe` and `laguna` run with evidence**, 42 to 44, on one seam:
  the learned attention output gate (`attn_gate.rs`). llama.cpp's three
  gating graphs were read side by side and are one op with two free
  parameters, the activation (sigmoid / softplus) and the width (per
  channel / per head, read off the tensor), so the type has two axes
  and a table. Three libllama-golden fixtures, KL 7.03e-13, 1.51e-13,
  9.57e-14. `step35`, the third graph, keeps its clamp arrays and window
  array and says the gate is done.
- Attention sinks are a tensor-presence fact (`AttnWeights::sinks`)
  rather than a gpt-oss name check: four llama.cpp graphs pass the
  tensor into the one `build_attn_mha`. gpt-oss still requires it and
  the fused Metal launches still refuse a layer that has one, by the
  tensor. `mimo2` does not close on it, because every real export
  carries MTP blocks and a per-layer window array.
- Every fused Metal attention launch takes its view of a layer's
  weights from ONE exhaustive destructure of `AttnWeights`, so a field
  added there does not compile until the Metal side says whether the
  kernels serve it.
- A sliding-window geometry the full-attention layers do not share --
  `rope.dimension_count_swa`, `attention.key_length_swa`,
  `attention.value_length_swa` -- is refused by name for every
  architecture (`swa_geometry.rs`); it loaded and was ignored before.
  The Olmo-3 "window plus scaling" refusal is one table for `olmo2`,
  `mellum` and `laguna` rather than one `if`.

## [0.20.0] - 2026-09-11

### Fixed

- **The tokenizer disagreed with llama.cpp on text that mentions special
  tokens** (#198). Silent-wrong-answer class. Two bugs, not one. A
  heuristic added for one checkpoint's `<|im_end|>` promoted ANY
  angle-bracket-shaped vocabulary entry to special, sweeping in `<s>`,
  `</s>`, `<unk>` and `<?>`; in Qwen2.5's file `<s>` is a plain token
  and llama.cpp never treats it as special under any setting. Removed.
  And llama.cpp's common tokenize path defaults to NOT parsing special
  tokens in text, while ferrox behaved as if it always did. Every
  `encode` now takes an explicit `SpecialTokens::{AsText, Parse}`, so a
  caller cannot avoid choosing, and each caller sits where llama.cpp's
  own source puts it, cited line by line. The oracle was strengthened
  first: the golden dumper writes every case under both settings, two
  new cases contain markers as prose, and they were confirmed RED on
  main before the fix. After it, all 20 local checkpoints match a
  current libllama across 21 cases and both settings.

- **Metal decode was thread-affine, and the pool gate is now relaxed
  for it** (#184, closing #166). A thread-local configuration flag was
  read on whichever thread ran the step. It is carried across the pool
  explicitly now: `Carry` destructures itself exhaustively with no `..`,
  so a setting added later and not carried fails to compile. CUDA and
  Vulkan still decline, stated as a verdict with the reason.

- **Evicting behind a sliding window leaked pool blocks** (#189). The
  shrink made `push` compare rows against a capacity that had moved, so
  a pool-backed cache drew a fresh block every `slack + 1` tokens forever
  and exhausted a pool where `push` is documented infallible. Reachable
  with `FERROX_KV_POOL_BLOCKS` and `FERROX_KV_WINDOW` together; the
  existing pool test never reached the shrink.

- **Gemma 4 tool calls fell through to the Llama 3 parser** (#190).
  Format inference tested the literal string `gemma4` while the
  checkpoint calls itself `Gemma-4-E2B-It`, so the arm had NEVER fired.
  Widened, with the earlier Gemmas asserted still not to match.

- **The Gemma family was exempted from four scaling keys it does not
  read** (#186), so a hand-written `gemma3.residual_scale` would have
  loaded and been ignored. Found by deriving the refusal list from the
  implementation table instead of restating it beside it.

- **`nextn_predict_layers` was refused nowhere** (#193). MTP blocks sit
  inside `block_count` and llama.cpp skips them, so a real EXAONE-MoE
  export with an MTP head would have run its speculative head as two
  more decoder layers. Gated on the value, since the converter writes
  `0` for sizes with no head.

- **Swapping the model through `POST /admin/models/load` could leave
  generation fluent and wrong** (#180). Silent-wrong-answer class, so
  read it twice. Two Metal caches map `(host pointer, host length)` to
  an uploaded `MTLBuffer`, and an address is not an identity: the
  outgoing model's allocations are freed with it and the incoming
  model's land on the same addresses. `resident_weight_buffer` knew
  that and checked; `resident_f32_buffer`, seventy lines below it in the
  same file, did not, and what it caches are the RMSNorm gammas a
  `Decoder` owns. Measured, not assumed: instrumenting every hit to
  compare it against the host bytes reported **49 stale hits in a
  24-token decode** of Llama-3.2-1B-Q6_K loaded after
  Llama-3.2-1B-Q4_K_M. The MoE stack had a third cache in front of both,
  keyed on a bare pointer with no length at all. Every resident cache
  now goes through one lookup that will not serve an entry unless the
  entry can still prove it holds the caller's bytes, and the Studio
  model selector is that endpoint. The repack cache named as the likely
  cause in the issue was NOT it: the same swap sequence under
  `FERROX_METAL=0` answers identically on every pair.

### Added

- **Ten more architectures run with evidence**, 26 to 37, each with a
  libllama-golden fixture, and the two cheap triage classes are now
  EMPTY: nothing still refusing is one fixture or one match arm away.
  - `olmo2` and `exaone4`, one post-norm residual topology (#183)
  - `chatglm` and `qwen`, sharing the fused `attn_qkv.bias` arm; `qwen`
    also needed its FFN width halved, which the verdict never named
    (#185)
  - `granite`, `granitemoe` and `granite-moe`, on one implementation of
    the four scalar multipliers, with 18 hand-written residual adds in
    `decoder.rs` collapsed onto one function (#186)
  - `minicpm`, on the defaults hook that Granite made a field of the
    same table (#191)
  - `olmo`, on a non-parametric LayerNorm; its `clamp_kqv` stays a
    refusal because llama.cpp's logits move when the key is present
    (#191)
  - `exaone-moe`, `smollm3` and EXAONE-4 32B, on one per-layer RoPE
    gate that llama.cpp applies in six architectures from literals and
    never from a key (#193)
- **Three rows turned out not to be architectures**: `mistral`,
  `mixtral` and `yi`. libllama refuses all three strings and every real
  checkpoint declares `llama`. They now refuse by name and say to
  re-convert (#185).
- **Forced `tool_choice` on 10 of 11 wire formats**, up from 8. Two
  recorded refusals were wrong: MiniMax-M3's ambiguity is real when
  reading and absent when writing from a schema, and Gemma 4 needed a
  new shape rather than a new branch. Muse-Glimmer still refuses for a
  stated reason (#190, closing #29).
- **`ferrox imatrix`**, a port of `llama-imatrix`, and `ferrox quantize
  --imatrix`. Importance-weighted quantization is byte-identical to
  `llama-quantize --imatrix` on 311 of 311 tensors across five targets.
  The matrix file matches llama.cpp's in format, names and shapes, with
  values agreeing to the forward pass's precision (#195).
- **`ferrox batched-bench`**, a port of `llama-batched-bench`, on the
  same guards as `ferrox bench` rather than beside them; the host
  preflight and receipt envelope were extracted so both tools call one
  function. Four llama.cpp flags refuse by name through an exhaustive
  destructure (#197).
- **Server slot save and restore**, with an identity check llama.cpp's
  own slot file lacks: a digest over the checkpoint's sorted metadata,
  tensor directory and a sample of every tensor, plus layer geometry.
  Restore under a different quantisation refuses naming the checkpoint;
  under a different model, naming the model (#192).
- **`-b` and `-ub` batch flags** on the server, which exposed one number
  spelled as two independent environment variables at two readers; both
  names now live in one array both readers index. `-np` was already
  wired and the audit was stale, so `/metrics` now reports the scheduler
  configuration the worker was spawned with (#192).
- **Windowed models are priced by their window** (#189, on #61). The
  store already evicted behind `FERROX_KV_WINDOW`, including per-layer
  for alternating models; what was missing was that nothing spent the
  saving. `KvBudget` now carries a residency, so `--ctx-size auto`, the
  pre-load check and `inspect-plan` see it. Gemma-2-2B at 32k context:
  6.50 GiB to 4.06 GiB.
- **`/v1/tokenize` honours `parse_special`**, which was a 501 by name
  (#198).

### Changed

- **Ferrox Studio has a mark instead of two letters, and a palette with
  no brand hue** (#187). The logo is alpha-iron's body-centred cubic
  cell seen down its body diagonal, one outline, three spokes, one
  node; it reads at 16 pixels. The neutral ramp is zero-chroma and the
  only coloured tokens are the semantic states; a test fails if a
  non-semantic token gains colour or the two theme blocks drift. The
  send button is a circle, and the send glyph is one whose horizontal
  centre is verifiable.

### Documented

- `docs/MODELS.md`'s list of audited architectures had been severed in
  half by an inserted section and was three rows stale; it is rejoined
  and now verified against `AUDITED_GENERIC_GQA` in both directions
  (#188).
- Studio screenshots in the README and `ui/README.md`, captured from a
  real server, in the dark theme (#194, #196).
- `ferrox quantize --help` claimed only Q8_0 and Q4_K write; a test now
  walks every target against the long help (#195).

## [0.19.1] - 2026-09-10

Ferrox Studio only. No crate in the workspace changed, so an engine
built from 0.19.0 behaves identically.

### Changed

- **One model selector instead of three surfaces claiming the model.**
  The chat header switcher and a `Load` button on every row of the
  Models table posted the *same* request for the same server-wide
  effect: the server holds one checkpoint and `/v1/chat/completions` has
  no per-request override, so these were not different scopes. Models is
  now facts plus the one verb a picker cannot express, `Unload`, and the
  header keeps the picker. The rule comes from how this class of UI
  splits generally: a management screen installs, removes and reports,
  while a picker beside the conversation selects.
- **The status control at the bottom of the sidebar reads as status.**
  It was rendering the loaded model id under a chevron, which made a
  health indicator look like a fourth model selector. It now shows the
  server state and version, with the model id, capabilities and
  last-request age one click into its popover.
- **The base URL is a new chat, and a conversation has its own URL.**
  Entry previously ended in an unconditional "reopen the newest
  conversation", on every path and after any elapsed time. `/ui/chat` is
  now empty and `/ui/chat/<id>` is that conversation, with the id
  stamped in by the first message.
- **Returning after 30 minutes away starts fresh** and offers the
  previous conversation back. "Away" is measured only while the document
  is visible and is stored per tab, so a reload of an old tab reads as
  away while a deep link into a new tab is simply honoured. It never
  fires over a running generation or a half-typed message.

  Worth recording that the research contradicted this one: no product
  surveyed implements a staleness rule, and the advice was not to invent
  one. It is here because it was asked for, which is why it is narrow,
  announced, and undoable rather than silent.

### Fixed

- The staleness rule **could never fire**: the visibility heartbeat
  stamped the tab awake from its mount effect while the entry check read
  that stamp from inside a promise, so the evidence was always already
  overwritten.
- Correcting the URL **re-loaded the conversation the rule had just
  declined**, because a route correction is a state update and the
  effect ran again on the old id, printing its banner over a resurrected
  transcript.

### Known issue

`POST /admin/models/load` can leave generation producing garbage until
the server is restarted, and the Studio model selector is that endpoint
([#180](https://github.com/antonellof/ferrox/issues/180)). Present in
0.19.0 and not fixed here. Deterministic, and a fresh start on the same
checkpoint is correct, so restarting the server clears it.

## [0.19.0] - 2026-09-10

### Fixed

- **Metal greedy decoding returned the wrong token in the default
  configuration.** This is the silent-wrong-answer kind, so read it
  twice. Metal has two paths for the final `lm_head` step: a fused GPU
  fold that argmaxes on device, and a host path that samples on the CPU.
  **The fold argmaxes the RAW logits; the host path applies the
  repetition penalties first**, and `--repeat-penalty` defaults to 1.1
  here rather than llama.cpp's 1.0. So the fast path answered a question
  nobody asked. Proven by checksum: at 1.1 the folded completion was
  bit-identical to the completion at 1.0, which is what "the penalty
  never ran" looks like, while the unfolded completion matched the CPU
  reference exactly.

  The cause was **one predicate answering two questions**.
  `greedy_equals_argmax` was called both after the penalties had been
  applied, where excluding them is correct, and by Metal before anything
  had been applied, where it is not. It is now split in two, each
  derived from one exhaustive match over the sampler chain and one
  exhaustive destructure of the parameters with no `..`, and the old
  name is deleted so every call site had to choose. A sampler field
  added later fails to compile until it is classified.

  The fold is refused when a penalty is live rather than the penalties
  being reimplemented in a Metal kernel, deliberately: a second
  implementation that must agree with the host about sign convention,
  the once-per-candidate rule and the window is this repo's dominant
  defect shape, and putting it inside the fix for that shape is how it
  recurs (#170, #172).

- **Two same-length activations could alias on Metal, across requests.**
  The residency cache matched on length alone. `cpu-cuda-parity.md`
  recorded that as safe because "exactly one site sets it": one site
  sets it and **three** consume it, each routinely handed a same-length
  activation that is not the published one. The published value was a
  raw buffer pointer that escaped the mutex protecting it, so two
  concurrent `ferrox-server` requests could have one answer the other's
  `lm_head`. Live, not latent. Publication now lives inside the guarded
  scratch, keyed on the host address and length of the exact buffer
  returned, and drops on any borrow (#171).

- **Metal decode was thread-affine and did not say so.** Running a
  forward on a different thread changed its output. The stated cause was
  the resident activation cache and that was wrong: disabling the reuse
  changes nothing, because the dense stack downloads with an exact copy.
  The cause was `GREEDY_ARGMAX`, a thread-local *configuration* flag set
  on the main thread and read on whichever thread ran the step, so a
  worker read the default and silently took the other `lm_head` path. It
  is now a three-state type with capture and adopt, since the two
  spellings of "false" are not the same claim (#166, #171).

### Changed

- **CPU decode enters the thread pool once per forward instead of about
  150 times.** Rayon's `join` has two arms: from a rayon worker it runs
  inline on a spin latch with no syscall, and from any other thread it
  injects the job and blocks on a mutex and condvar. Every forward was
  driven from a thread rayon did not own, at roughly five regions per
  layer. A profile put **74% of the token** in `__psynch_cvwait` on the
  driving thread, against 6.6% of samples in the actual matvec kernel.
  Interleaved within-process ratios: **135M +29%, 3B +9%, 8B +3%**,
  prefill flat (#128, #167).

  Note what this corrects. #128 had computed scheduling at 6.7% of a
  token and ruled it out. The arithmetic was right and **the denominator
  was stale**, taken against a 17 ms token before the repack fix in
  0.18.0 shrank it to about 5 ms. The ruled-out cause was the real one.

- **Every engine gets that, not just the generic decoder.** The five
  dedicated engines share the `Engine` trait, so the wrapper is written
  once as that trait's provided body and an engine supplies only its
  inner worker. Cold regions per decode step: Gemma-4 **100 to 1**, the
  BERT encoder **72 to 1**, Kimi and GLM-5.2 30 to 1, DeepSeek-V4 16 to
  1. A structural test refuses any engine that overrides the promoted
  entry point, so the seam cannot be bypassed silently (#169).

### Documented

- `benchmarks/RESULTS.md` is now the generated table and nothing else,
  291 lines to 81, with each model's prefill and decode on **one row**
  rather than ten rows apart. Nothing was re-benchmarked: `--render`
  reads the committed receipts, and the sorted multiset of every gap
  cell is identical before and after. The prose moved to
  `benchmarks/HISTORY.md` rather than being deleted, because the
  aarch64 rows and the before/after studies are measurements a generator
  cannot reproduce (#175).
- The 8.2x SmolLM2-135M row is marked **stale** rather than edited. It
  predates both the 0.18.0 repack fix and #167; on an M2 Pro it now
  reads about 1.9x. That is a different machine, so it is recorded as
  evidence the gap shrank rather than as a replacement number, and the
  row still needs a quiet Cortex-A725 (#168).
- `ferrox bench` does not use the greedy fold, so #172 cannot move the
  published Metal rows. Written down because it looks like it should
  (#173), along with a correction: the fold has **two** callers, not
  one, and the conclusion rests on neither being on the bench path
  (#174).
- **An inverted quant claim is corrected.** `body_quant`'s doc used
  `Llama-3.2-1B-Instruct-IQ4_XS.gguf` to teach that a filename is not a
  quantization, and said the file holds 96 `IQ4_NL` tensors and no
  IQ4_XS. The count was right and the type was backwards: it holds 96
  IQ4_XS, 16 Q5_K, one Q6_K and zero IQ4_NL. That inversion crosses the
  line that decides a verdict, because ggml declares `vec_dot_type =
  Q8_K` for IQ4_XS and `Q8_0` for IQ4_NL, so the comment described a
  file whose DRIFT would be unexplained while the real file's DRIFT is
  the expected case. The tool's verdict was always right; only the
  explanation was wrong. A new test pins the two look-alike neighbours
  to their ggml facts, because the existing one walked whatever the
  lists happened to contain and stayed green under the inversion (#176).
- The file-size table in `CLAUDE.md` was re-measured. One of five files
  shrank, so the rule still lost on balance, and the real wins left the
  table entirely: `repack.rs` 6446 lines to a directory of ten,
  `sampling.rs` 1569 to 894, `mul_mm.rs` to 865. Every one of those
  splits happened because somebody was about to add to the file and
  split it first (#165).

## [0.18.0] - 2026-09-09

### Added

- **The four missing samplers**, so the chain is llama.cpp's full nine
  steps in upstream's order: `dry`, `xtc`, `typ_p` and `top_n_sigma`
  join `penalties`, `top_k`, `top_p`, `min_p` and `temperature`. Golden
  values were read out of `libllama` rather than reasoned about. Every
  new step is a no-op at its neutral value, so a default run is
  unchanged. `mirostat` and `infill` remain refused by name, and the
  reason mirostat is refused is written down: upstream *replaces* the
  chain with it and it carries per-sequence state ferrox has nowhere to
  put (#160).
- **`ferrox gguf-split`**, a port of `llama-gguf-split`: split by tensor
  count or by size, merge, `--dry-run`, the same shard names and the
  same `split.*` metadata keys. Cross-checked against the real tool,
  which produced 6 of 6 shards of identical size, and each tool merges
  the other's output (#154).
- **Three more architectures run with evidence**: `gemma`,
  `hunyuan-dense` and `ernie4_5-moe` at step 1, each with a
  libllama-golden fixture. Audited 23 to 26, unaudited refusals 34 to
  31, and **the fixture-away class is now empty**: every row that needed
  only evidence has it, so everything left needs code (#161).
- **`ferrox quantize` writes Q5_K_M and Q6_K**, byte-identically (#162).
- **CUDA gains Q2_K, Q3_K, IQ4_NL, IQ4_XS and MXFP4**, each with both a
  matvec and a GEMM, since landing half of a kind is forbidden. Verified
  on the host across 11 kinds, 33 shapes and 75,042 positions with zero
  mismatches. **None has run on a GPU** (#157).
- **AVX2 GEMMs for all five interleaved repack kinds on x86**, with one
  per-workload dispatch rule shared by ten call sites. Verified by
  execution on real AVX2 silicon, not emulation, and **not yet
  benchmarked** (#159).

### Fixed

- **Q4_K quantization was not byte-identical, and the documented reason
  it "could never be" was wrong.** llama.cpp's `sumlx += w*x[i]*l` is
  contracted by its compiler into a single fused multiply-add; Rust does
  not contract, so the strict transcription that shipped was the defect.
  One unit in the last place flips a comparison and rewrites a whole
  super-block, which is why 1.15% of super-blocks differed rather than a
  rounding-sized fraction. Spelling the fusion as `mul_add` takes Q4_K,
  Q5_K and Q6_K to **zero differing super-blocks across all 147 tensors**
  of a real model (#162).
- **The int-dot matvec repacked every weight matrix on every call.** It
  passed a hand-written "uncacheable" identity where every other matvec
  passed a real cache key, so the interleaved layout was rebuilt and
  copied per token. It was 89% to 90% of decode work on Q8_0 and Q4_0
  models. This also corrects the premise of #128: the cost is
  proportional to gate and up projection bytes rather than fixed, and
  fires only on those two formats (#155).
- **`FERROX_CUDA=0` did not mean CPU.** The matvec launcher never
  honoured the disable flag, on the strength of a comment claiming CUDA
  needed no guard because launchers return an error with no device.
  That is true of Metal and false of CUDA, where the binding panics, so
  any quantized matvec on a CUDA build without a driver aborted the
  process (#157).
- **The CUDA host-check harness had silently stopped compiling** when
  the `float4` inner loop landed, so it verified nothing while still
  exiting green-adjacent. It now iterates the kind table rather than a
  hand-kept list (#157).
- **`ferrox gguf-split` was unreachable**: the CLI module existed but
  was never registered, so the subcommand would have fallen through to
  an implicit `ferrox run` and started generating text (#154).
- Three pre-existing test races on a process-global override, which
  passed only because the two halves of the int-dot tier used to move
  together (#159).

### Changed

- **The CPU scheduler is chosen by work size**, not by
  `FERROX_CPU_POOL`. One predicate decides per operation and the
  environment variable is now an A/B override. The crossover constant is
  **bracketed by the published measurements rather than swept**, and no
  quiet-host before-and-after has been run, so this is not yet a
  performance claim (#155).
- **The repack cache has a derived byte budget with eviction.** The fix
  above retains the packed copy, which cost +527 MB at 1.1B on Q8_0 and
  would scale per expert on a mixture-of-experts model. The budget is
  available memory minus a shared headroom constant minus committed
  expert bytes, then a quarter share, so it spends the same pool as
  `expert_store` rather than opening a second one. Zero disables the
  cache and reproduces the previous behaviour exactly (#158).
- **Metal encodes less per token**: 13% fewer dispatches and 9% fewer
  barriers, by fusing RoPE for Q and K, folding the K and V cache append
  into one grid, and deleting a barrier that guarded zero work on models
  without QKV bias or QK-norm (#156).

### Measured

- **The Metal suite was re-measured on a quiet M2 Pro** and the stale
  0.13.3 rows retired. Prefill spans **1.01× to 1.10×** and decode
  **0.64× to 1.11×**, with **12 of 14 comparable decode rows faster than
  llama.cpp**. MoE decode on OLMoE moved from ~1.41× to **0.96×**.
  Gemma-2-2B at 1.11× is the worst row, confirming the 1.12× that the
  concurrent-encode work predicted. Mistral-7B leaves the table because
  `--fit-host` refuses it at ~10 GiB needed against 11.2 GiB free, and a
  run from swap measures the swap (#163).

### Retired hypotheses

Both of these were the stated cause of an open issue, and both are now
disproven by measurement rather than argument.

- **Metal host cost is not dispatch count.** Removing 11.5% of encode
  operations bought **2.3%** of host time, and barriers were already
  hazard-driven rather than per-operation. What remains is per-dispatch
  argument binding: roughly 2400 encoder calls per token against 418
  dispatches and barriers combined (#149, #156).
- **The CPU per-token cost is not fixed.** See the repack fix above
  (#128, #155).

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

[0.20.0]: https://github.com/antonellof/ferrox/compare/v0.19.1...v0.20.0
[0.19.1]: https://github.com/antonellof/ferrox/compare/v0.19.0...v0.19.1
[0.19.0]: https://github.com/antonellof/ferrox/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/antonellof/ferrox/compare/v0.17.1...v0.18.0
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
