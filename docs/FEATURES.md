# Features

Ferrox is a pure-Rust GGUF inference engine for dense and MoE models.
Weights stay quantized when the file is mmapped, and dequantization
happens inside the matvec. Backends: CPU, Apple Metal, and CUDA.

## Models

Measured against llama.cpp on the same host and the same GGUF with
`ferrox bench`. Gap = `llama / ferrox`, so anything under 1 means Ferrox
is faster.

- **Dense GQA**: TinyLlama, Llama 3.2, SmolLM2, Qwen2.5/Qwen3,
  Gemma-2/3/4, Phi-4-mini, Mistral-7B. Re-measured on 2026-09-11 on a
  quiet M2 Pro after the Metal kernel work in #208: **every one of the
  16 comparable Metal `tg128` rows is faster than llama.cpp**, from
  SmolLM2-135M and Qwen2.5-0.5B at **0.60×** through Qwen3-0.6B
  **0.61×**, Gemma-3-1B **0.68×**, TinyLlama **0.83×**, Phi-4-mini
  **0.88×**, up to OLMoE at **0.96×** as the closest row. Gemma-2-2B,
  the worst row two days earlier at 1.11×, reads **0.94×**. Dense Metal
  prefill spans **0.99× to 1.09×**. **CPU is a different story and is
  behind on every row**: prefill 6.3× to 10.1× and decode 1.06× to
  1.92× on x86, though the AVX2 GEMM tier that closes the prefill half
  landed unmeasured.
- **Phi-4**: CPU **and** Metal. Partial rotary (`n_rot < head_dim`) and
  LongRoPE `attn_factor` ride the Metal RoPE kernels as `rot_dim` and
  `mscale` uniforms, matching ggml's `rope_yarn` (the magnitude scale
  reaches the rotated channels only). Now measured with that fix in
  place: `pp512` 1.02×, `tg128` 0.98×.
- **MoE**: OLMoE-1B-7B, and Metal decode is no longer the weak spot it
  was. It read ~1.41× when this line was first written and reads
  **0.96×** on the 2026-09-09 suite, so MoE decode is now faster than
  llama.cpp rather than half again slower. Metal Concurrent plus fused
  encode groups, `MemRanges`, `mul_mv_id` and prefill `mul_mm_id`. On
  CPU: int-dot and interleaved Q4_K.
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
- **OLMo-2 and EXAONE-4**, audited against libllama on 2026-09-10 as
  ONE residual topology rather than two: neither has an `attn_norm` or
  an `ffn_norm` tensor, both sublayers read the raw residual, and each
  branch's output is normed before its residual add
  (`ferrox_models::norm`). One sub-case stays refused by name -- an
  `olmo2` carrying both a sliding window and a RoPE scaling (Olmo-3).
  EXAONE-4 32B was the other and runs since 2026-09-11 (next item).
- **Per-layer RoPE: EXAONE-4 32B, EXAONE-MoE and SmolLM3**, audited
  against libllama on 2026-09-11 as ONE rule. llama.cpp gates rotation
  per layer in six architectures and ferrox could not say so, which
  cost `exaone-moe` and `smollm3` an outright refusal and EXAONE-4 32B
  a refusal by name. `exaone4.cpp:116` and `exaone-moe.cpp:136,155-161`
  are the same predicate (`exaone-moe.cpp:4` pins `swa_type` to
  STANDARD, which makes `exaone4`'s `|| swa_type == NONE` vacuous);
  `smollm3.cpp:5,69` is `(il + 1) % 4 != 0`. `ferrox_models::rope_layers`
  holds the table for all six, `ModelConfig::layer_rope` answers `None`
  for an unrotated layer so no rotation site can take the base and the
  divisors without answering the third question, and both fused Metal
  stacks take an `Option<LayerRope>` per layer. A 64-layer fixture is
  what evidences the 32B, because `exaone4.cpp:4` tests equality.
  `smallthinker`, `afmoe` and `llama4` are in the same table; the first
  two closed later on other seams, and `llama4` still refuses for other
  things. Found on the way: EXAONE-4 1.2B must ignore a
  window its file declares, and `nextn_predict_layers` (MTP blocks
  inside `block_count`) was refused nowhere and was then refused
  everywhere; it is SKIPPED now, as llama.cpp skips it, for the
  seventeen graphs that read the key (`ferrox_models::mtp_blocks`), on
  the generic path and all four dedicated loaders, and still refused by
  name elsewhere.
- **The per-layer sliding-window ARRAY.** `attention.sliding_window_
  pattern` is read upstream with `get_key_or_arr`, three ways: ignored
  where a graph reads the scalar overload (`exaone4`, `exaone-moe`,
  `olmo2`, twelve more), honoured where it reads the array overload
  (`mimo2`, `step35`, `gemma4`; a scalar there is a broadcast bool, not
  a period), and scalar-then-array for `mellum` / `cohere2moe`.
  `ferrox_models::swa_layers` is one table and one enum behind
  `ModelConfig::layer_sliding_window(il)`. Every real EXAONE-4 32B,
  EXAONE-MoE and Olmo-3 export carries the array and was refused over a
  value llama.cpp never reads; `mellum` is audited on it, with its
  window-plus-YaRN case (every real Mellum2) refused by name.
- **The LayerNorm with a bias, and with it Orion-14B (`orion`) and
  Nemotron-4 / Minitron (`nemotron`).** `NormOp::LayerNormBias` is
  `build_norm(x, w, b, LLM_NORM)`, the variant the eight-row
  "LayerNorm-with-bias group" shares; these two rows need nothing else
  (Orion is a Llama, Nemotron the ReLU-squared FFN `arcee` serves),
  audited against libllama at KL 2.3e-11 and 5.1e-13. Nemotron's
  optional `attn_output.bias` / `ffn_up.bias` / `ffn_down.bias` are
  refused as unread; the other six rows say what else they need.
- **GLM-4-0414 / GLM-Z1 / GLM-OCR (`glm4`) on the generic path**,
  audited against libllama at KL 9.7e-15 with no code change: the row
  had been sent to the GLM-5.2 MLA loader for keys its graph never
  reads. `ferrox_models::mrope` decides what a vision export's text
  tower (`rope.dimension_sections`) means per architecture: served as
  NEOX for `glm4moe` (llama.cpp's M-RoPE on text positions, measured
  byte-identical), refused for `glm4` (converter-permuted weights, 0.72
  apart).
- **GLM-4.5 / 4.5-Air / 4.6 (`glm4moe`) on the generic path**, audited
  against libllama on the 355B shape (per-head Q/K norms) and the Air
  shape, KL 1.5e-15 / 2.4e-15. The pre-FFN norm stored as
  `post_attention_norm` is one row in
  `norm_sites::PRE_FFN_NORM_IS_POST_ATTENTION_NORM`; the sigmoid MoE with
  `exp_probs_b`, `expert_weights_norm` and `expert_weights_scale` read
  from the file, the shared expert and the NextN blocks were already
  served. A GLM-4.5V text tower's `rope.dimension_sections` rotates NEOX,
  which is what llama.cpp's M-RoPE computes on text positions (measured
  byte-identical).
- **MLA with the absorption optimization, and DeepSeek-2 checked
  against libllama in both tensor forms.** `ferrox_models::mla::MlaKvB`:
  the combined `attn_kv_b` (legacy exports, `plm`) and the split
  `attn_k_b` / `attn_v_b` (every DeepSeek export since the `_mla` keys),
  the latter attending over a latent cache `kv_lora_rank + qk_rope` wide
  (`ferrox_core::mla_absorbed`). KL 2.35e-15 and 3.57e-15
  (`tests/deepseek2_graphs.rs`). YaRN as DeepSeek-V2 and V3 declare it
  is `ferrox_models::mla_yarn`, three more goldens at 1e-15.
- **A dense FFN summed with the routed experts, and a routed branch
  fed from the layer input**, and with them Arctic (`arctic`) and
  Grok-2. `ferrox_models::parallel_dense_ffn` is the table of the two
  graphs that sum a dense FFN with their experts (presence, scale on
  the sum), served through the shared-expert slot under the dense
  names; `RouterInput::NormedLayerInput` is Arctic's router and experts
  reading `ffn_norm_exps` of the layer INPUT while the dense half
  reads the post-attention residual. Checked against libllama, KL
  6.23e-14 (arctic) and 3.28e-10 (Grok-2, GELU table). Every fused
  Metal MoE launch refuses both.
- **PLM on the MLA engine, and the engine's first cross-engine
  evidence.** `plm` (PLM-1.8B) is DeepSeek-2's MLA attention on a dense
  model; `ferrox_models::mla_arch` is the table of the three ways it
  differs (a direct `attn_q` -- `ferrox_models::mla_q_proj`, which the
  lite DeepSeek-V2 layer counts take too, where the loader had refused
  them for a key llama.cpp never reads; an ungated ReLU-squared dense
  FFN carried on `MlaDenseFfn`; a tied lm_head whose decoy
  `output.weight` is refused as libllama refuses it). Checked against
  libllama, KL 1.87e-13. The same pass made the engine REFUSE any
  `rope.scaling.type` but `none`, because it has neither the frequency
  rewrite nor the YaRN mscale in `kq_scale` that every real DeepSeek-V2
  / V3 export needs.
- **Weightless RMS norms, a per-head scalar Q gain, an embedding skip
  stream and two projection gains**, and with them Talkie (`talkie`).
  `NormOp::RmsNoParams` for a file with no norm tensor at all;
  `QkNormStyle::PerHeadScalar` for an `attn_q_norm` of one scalar per
  head with a weightless K norm; `ferrox_models::skip_stream` for the
  normed embedding added into every layer's output times
  `layer_output_scale`; and `attn_output.scale` / `ffn_down.scale`, the
  two per-tensor companions its converter writes, applied as
  `build_lora_mm` applies them (`ferrox_models::weight_scales` serves
  those two and still refuses the rest). The fused Metal launches
  refuse the model.
- **The same physical layers run more than once**, and with it Nanbeige
  (`nanbeige`). `num_loops` makes the logical layer count `n_phys *
  n_loops`, each logical layer with its own KV cache over shared
  weights, and `output_norm` between the passes unless
  `skip_loop_final_norm`. `ferrox_models::layer_loops`: `Decoder::layers`
  stays physical, `n_layers` is logical, one mapping serves the host
  bodies, and the loop norm sits at the end of both FFN bodies. The
  fused Metal launches refuse a looped model.
- **A V head width that differs from the K head width**, and with it
  MiMo-V2 (`mimo2`: `head_dim: 192, v_head_dim: 128` on every export).
  `ferrox_models::kv_head_dims` admits the pair for the one generic-path
  architecture whose converter writes them apart and keeps refusing it,
  naming llama.cpp's assert, for everyone else. `KvCache` and
  `PagedKvStore` size V by its own width, the single-query and batched
  prefill kernels accumulate over it, the projection check and the
  fused-QKV cut read it; every fused Metal launch, the CUDA resident
  hook, the slot file and the KV block file refuse a split model, so
  MiMo-V2 runs on the host paths. Its `attention.value_scale` is
  `ferrox_models::attn_value_scale`, one reader of 140. Building it
  found `expert_weights_scale` / `expert_weights_norm` honoured for
  every architecture where llama.cpp reads them in twenty loaders; the
  loader's `EXPERT_WEIGHTS_*_READERS` tables are the measurement.
- **The two norms inside the blocks**, and with it BitNet (`bitnet`).
  `bitnet.cpp:24,36` require `attn_sub_norm` on the attention output
  BEFORE `wo` and `ffn_sub_norm` on `silu(gate) * up` BEFORE `down`,
  two sites the generic decoder's four norm slots did not have; one
  graph of 140 creates either tensor (measured), so
  `ModelConfig::block_sub_norms` is a `bool` the loader and the Metal
  predicate both read (`ferrox_models::sub_norms`). Applied in the one
  attention tail and the one dense FFN row body; every fused Metal
  launch refuses the model. Per-projection `.scale` / `.input_scale`
  companions, which llama.cpp multiplies in for every architecture and
  the NVFP4 and older BitNet converters write, are refused by name
  (`ferrox_models::weight_scales`) rather than run at the wrong
  magnitude. A real BitNet-b1.58 still needs `TQ1_0` / `TQ2_0`
  kernels; a Q8_0 or F16 re-export runs.
- **The MoE router operand**, and with it every SmallThinker
  (`smallthinker`). `smallthinker.cpp:111` routes on `inpL`, the
  residual stream as it enters the layer, before `attn_norm` and before
  attention; every other MoE graph on this engine routes on the normed
  FFN input the experts read. `ferrox_models::router_input` is the
  table (fifty-nine `build_moe_ffn` call sites parsed, four pass a
  precomputed `probs_in`, one on this engine differs in the operand),
  `Decoder::router_operand` the one place the operand is captured, and
  the GPU router paths refuse the row through the predicate they
  already shared. Its gated ReLU experts are `FfnActivation::Reglu`,
  split from `arcee`'s ungated `ReluSqr` because the aliasing that
  served `arcee` would have dropped SmallThinker's real gate; its
  `n_swa` is pinned to 4096 as `smallthinker.cpp:8` pins it.
- **The per-position attention temperature**, and with it every
  Ministral-3 (`mistral3`). `attention.temperature_scale` is Llama-4's
  "attention temperature tuning" as a GGUF key: llama.cpp multiplies Q
  after RoPE by `log(floor(pos / floor) + 1) * scale + 1` per token,
  and the floor is `n_ctx_orig_yarn` -- `context_length` unless the
  YaRN key overrides it. `ferrox_models::attn_temperature` is one value
  behind `ModelConfig::attn_temperature`, applied on the three host
  bodies and fenced off the fused Metal launches; three graphs of 140
  build the input (measured), and the two on other engines are
  recorded, with the MLA loader refusing Mistral-Large-3's key by name
  where it used to drop it.
- **YaRN's magnitude term**, for every architecture. `rope_attn_factor`
  now carries `rope.scaling.attn_factor` times llama.cpp's
  `get_mscale(factor, 1) / get_mscale(factor, yarn_log_multiplier)`
  (`ferrox_models::yarn_magnitude`), which it did not before: a YaRN
  checkpoint was roped at the right frequencies and attended with
  logits low by `(1 + 0.1 ln factor)^2`. Same field, so the CPU helper
  and the Metal `mscale` uniform both carry it.
- **OLMo-1**, the third norm shape and a third variant of that same
  enum. It is pre-norm like llama, but `olmo.cpp:65-67,104-106,128-130`
  normalise with a null weight and a null bias -- a non-parametric
  LayerNorm, mean subtracted and standard deviation divided out -- and
  the file carries no norm tensor of any kind, not even an
  `output_norm`. `Decoder::final_norm` became a `NormOp` with it,
  because the fused Metal stacks that fold `final_norm + lm_head +
  argmax` had `Some(&self.final_norm)` written into them
  unconditionally. **A file declaring a positive
  `olmo.attention.clamp_kqv` still stops**: llama.cpp clamps Q, K and V
  by it (`llama-graph.cpp:1611-1652`) and ferrox clamps no projection
  anywhere, so OLMo-7B loads and OLMo-1.7-7B, whose `clip_qkv` is 8.0,
  does not.
- **MiniCPM**, which was refused by NAME rather than as unaudited,
  because what it does is invisible in the file: `minicpm.cpp:5-7`
  assigns an embedding multiplier of 12.0, a residual multiplier of
  `1.4/sqrt(n_layer)` and a logit multiplier of `256/n_embd` and only
  then lets the GGUF override them, so an export declaring nothing is
  still scaled three ways. It runs Granite's graph verbatim
  (`models.h:1594-1601`), so this is a DEFAULTS field on the table
  below rather than a second implementation. It reads no
  `attention.scale`, and a file declaring one is still refused.
- **ChatGLM and Qwen-1**, both audited against libllama on 2026-09-10
  and both closed by the same arm: the *fused* `blk.N.attn_qkv.bias`.
  ferrox split a fused `attn_qkv.weight` and then looked for the bias
  only under the split `attn_q.bias` names, so ChatGLM2/3's
  `add_qkv_bias: true` and Qwen-1's required bias were dropped and every
  Q, K and V projection ran unbiased. `chatglm` also exercises PARTIAL
  RoPE (its converter writes `rope_dimension_count` as half a head) and
  a fused gate+up SwiGLU; `qwen` needed a second arm, since its
  `feed_forward_length` counts gate and up together and every FFN matrix
  is half as wide as the key says.
- **Also loadable**: Yi-1.5, qwen2moe / qwen3moe (MiroThinker GGUFs, for
  example), Gemma-2, Phi-3, Llama-3.1, and GLM4 when the tensors are
  there. None of these are in the published suite. Note Yi loads as
  `llama`, which is what its GGUF declares -- the `yi` architecture
  string does not exist in llama.cpp and ferrox refuses it by name,
  saying so.
- **MoE routing bias** (`exp_probs_b`, DeepSeek-V3's aux-loss-free
  selection bias) plus `expert_weights_scale` and
  `expert_weights_norm`, on the generic path. That one tensor is what
  used to stop dots1, ernie4_5-moe, bailingmoe2, exaone-moe,
  hunyuan-moe and afmoe from loading at all. It is checked against
  llama.cpp's own dots1 implementation reading the same synthetic
  checkpoint, and has not been validated on a published one.
- **Granite's four scalar multipliers**: `logit_scale`,
  `residual_scale`, `embedding_scale` and `attention.scale`, on
  `granite`, `granitemoe` and the `granite-moe` alias, checked against
  llama.cpp's own logits on synthetic fixtures. They are
  hyperparameters, not tensors, so the check for unread tensors never
  sees them: before this, a Granite checkpoint would have loaded and
  answered at a scale it was never trained at. One implementation
  (`ferrox_models::scalar_multipliers`), parameterised by architecture,
  serves all three rows.
- **Every other architecture's scalar multipliers still stop the load**,
  from a list derived from that same table rather than restated beside
  it. A checkpoint that declares one with a value that changes the maths
  stops with an error naming the key. A Granite file declaring
  `rope.scaling.finetuned = false` stops too, because llama.cpp then
  runs it with no rotation at all and there is no way to express that
  here.
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
| **CUDA** | Matvec, resident weights, FFN fuse (`--features cuda`), plus batched GEMMs for `Q8_0`, `Q4_0`, `Q5_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q2_K`, `Q3_K`, `IQ4_NL`, `IQ4_XS` and `MXFP4` that **have never executed on a GPU** |
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
mismatches across **11 kinds, 33 shapes and 75,042 compared positions**.
Its hardware test is `#[ignore]`d, and NVRTC is not clang, so "it
compiles here" is not "NVRTC accepts it". Below the width threshold a
single token stays on the matvec kernels, which are the arm that *has*
run on a GPU. That harness is worth no more than its own health: it had
silently stopped compiling once already, which is why it now runs over
every kind in the table rather than a hand-kept list.

**CUDA is benchmarked and far behind.** It has receipts now, on an RTX
3060: prefill 22.6× to 33.8× and decode 2.2× to 5.0× against llama.cpp
on the same box. So a Windows or Linux install runs, and answers
correctly, and should not be chosen for speed yet. The newest kernels
in the table above are a further step back from that: they are verified
on the host and have never run on a card at all. `/health` reports the same thing per capability,
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

`--lora` / `--lora-scaled` load a LoRA adapter GGUF
(`convert_lora_to_gguf.py`'s format) on the CLI and the server, applied
inside every projection it names as llama.cpp's `build_lora_mm` applies
it and checked against libllama with the same adapter (KL under 5e-13 on
the fixture, 5.2e-4 on Llama-3.2-1B Q8_0 with a rank-8 adapter). The
server lists and rescales them through `GET`/`POST /lora-adapters` and
the per-request `lora` field. Routed-expert adapters, activated LoRAs
and the dedicated engines refuse by name; on Metal an adapted model runs
on the per-matrix path rather than the fused stacks. See
[`CLI.md`](CLI.md#lora-adapters) and [`API.md`](API.md#lora-adapters).

`ferrox perplexity` is the quality axis: corpus evaluation using
llama.cpp's method, agreeing with `llama-perplexity` to within a fifth
of one standard error on five checkpoints. Where the two differ, the gap
is monotone in the quant and has the sign the documented `vec_dot_type`
difference predicts. `ferrox quantize` writes **`Q8_0`, `Q4_K_S`,
`Q4_K_M`, `Q5_K_S`, `Q5_K_M` and `Q6_K` byte-identically** to
`llama_model_quantize()`, with or without an importance matrix
(`--imatrix`, 311 of 311 tensors identical to `llama-quantize
--imatrix` on a BF16 Qwen3-0.6B for all five targets tried), and
refuses every other target by name. `ferrox imatrix` is
`llama-imatrix`: same file format in both directions, so either tool's
matrix feeds either quantizer.
**The claim that Q4_K could never be byte-identical was wrong**, and it
was wrong for an instructive reason: llama.cpp's `sumlx += w*x[i]*l`
is contracted by its compiler into a single fused multiply-add, and
Rust does not contract, so a strict transcription of the C was the
defect. One unit in the last place flips a comparison and rewrites a
whole super-block. With the fusion spelled out as `mul_add`, Q4_K went
from 1.15% of super-blocks differing to zero, across all 147 tensors of
a real model. See [`CLI.md`](CLI.md).

The sampler flags carry llama.cpp's own defaults on `--temp` (0.8),
`--top-k` (40), `--top-p` (0.95), `--min-p` (0.05) and `--repeat-last-n`
(64). **One default still differs on purpose**: `--repeat-penalty` is
1.1 here and 1.0 (off) in llama.cpp
(`common/common.h:239`), so a run left entirely to defaults is not
token-identical. `-e`/`--escape` is on by default as it is there, and a
*partial* `-ngl N` is refused rather than silently offloading every
layer.

That non-default default has a **Metal decode cost**, and it is
deliberate. At `--temp 0` the Metal stack can fold
`final_norm + lm_head + argmax` into its own command buffer and hand
back one token id instead of `vocab_size` floats. A device argmax over
raw logits cannot apply the penalties, so the fold is now refused
whenever any of `--repeat-penalty`, `--presence-penalty` or
`--frequency-penalty` is live -- which, at `--repeat-penalty 1.1`, is
every plain greedy run. It used to fire anyway and return a token the
host sampler would not have chosen (GitHub issue #170).
`--repeat-penalty 1.0` or `--repeat-last-n 0` gets the fold back and
is also what makes a run token-identical to llama.cpp's defaults.

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
  llama.cpp's all-zero token types. The SCALE matches too, once the
  pooler is back: llama.cpp's converter deletes `bert.pooler.dense`
  from every BERT reranker, so the converter's output scores on about
  plus or minus 0.2 where the checkpoint was trained to produce about
  plus or minus 11 (#82). `ferrox splice-pooler` writes a GGUF that
  carries the pooler, tied to the checkpoint by the classifier both
  files hold rather than by a name; on the spliced file every score
  is within 0.051 of HuggingFace across four query sets. A file
  without the pooler still serves, with `ferrox_score_head:
  classifier(cls)` on every response, so a client can tell which range
  it is reading
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
- Chunked prefill (same scheduler as continuous batching). `-b` /
  `-ub` set the chunk on both decode paths from one number
- **Slot save/restore** (llama.cpp's `POST /slots/{id}?action=save|restore`):
  a prompt prefix's KV written to `--slot-save-path` and restored into
  the prefix cache after a restart, so a long system prompt is prefilled
  once per file rather than once per process. The file carries a
  checkpoint fingerprint, and a restore under another model or another
  quantisation is refused by name. See [`API.md`](API.md#slot-save-and-restore)
- Paged KV: shared page storage many requests read through a block
  table, with a radix tree over reference-counted page groups so
  conversations off one system prompt share its KV rather than each
  holding a copy. Off unless `FERROX_PAGED_KV_BLOCKS` is set. It used to
  be refused on a GPU backend, where it returned fluent wrong tokens; the
  cause was a Metal prefill leaving K/V on the device and filling the
  host cache with placeholders that the paged prefill then copied into
  the page store, and that refusal is lifted. See
  [`CONFIG.md`](CONFIG.md)
- **On the paged store**, a model whose layers *all* slide by the same
  window slides during decode, so a request holds its prompt and a
  window rather than its whole context , and admission prices it
  that way, so a store too small for the whole context still serves
  it. A tool call anchors the slide at the position the next agentic
  turn will rejoin at, and the anchor is dropped once the cursor
  drifts a window past it. An alternating-SWA model (gpt-oss,
  Gemma-3) does not slide *there*: a page group holds one block in
  every layer, and the full-attention layers still read position 0
- **On the contiguous host store**, eviction is PER LAYER, so the
  alternating models do get it: each layer's `KvCache` carries its own
  window from `attention.sliding_window` /
  `attention.sliding_window_pattern` and drops the rows behind it, while
  the full-attention layers keep everything. Off unless
  `FERROX_KV_WINDOW` is set, and it turns itself off under Metal
  attention, on a draft model, and beside the prefix cache. Output is
  token-identical with it on or off, asserted on logits as well as token
  ids against gemma-2-2b-it-Q4_K_M. `--ctx-size auto` and the pre-load
  admission check are priced against the same per-layer residency the
  stores evict with, so the saving is context a user is actually offered
  rather than memory nothing spends. See [`CONFIG.md`](CONFIG.md)
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
does not cover, and Metal is where it is fastest: every `pp512` row is
0.99x to 1.09x against llama.cpp and **every one of the 16** comparable
`tg128` rows is faster. And the single largest gap is not on this table:
running a model that does not fit in memory works as policy and not as
execution.

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
