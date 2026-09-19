# xInfer read against Frink, 2026-09-19

`guoqingbao/xinfer` (MIT, v0.14.2) is the other pure-Rust inference
engine with a serious feature list, so it is worth the same treatment
the llama.cpp pin gets: measure first, then decide what is worth
taking. The clone is `.scratch/xinfer` at `5665c22`, and its kernels
live in a second repo, `.scratch/attention.rs` at `2e0db42`, which is
where every claim in its README about speed or KV compression is
actually implemented.

## The measurement first

Both engines built from source on the same M2 Pro, both with Metal,
same GGUF files, same prompt, interleaved A/B, the box otherwise
quiet. Decode tokens per second, `-n 128`, greedy:

| Model (Q4_K_M) | frink | xinfer | reps |
|---|---:|---:|---|
| Llama-3.2-3B-Instruct | **67.1** | 45.8 | 66.90 / 67.23 / 67.06 vs 46.23 / 45.27 / 45.93 |
| Meta-Llama-3.1-8B-Instruct | **32.0** | 26.0 | 32.06 / 31.92 vs 25.91 / 26.00 |
| Phi-4-mini-instruct | **53.9** | refuses to load | 53.82 / 54.07 |

Prefill on the 3B, one prompt of about 540 tokens: frink 510 to 573
tok/s against xinfer's 249 to 338.

Phi-4-mini is not a slow row, it is a missing one: xinfer stops with
``Unable to read "..." as a GGUF file: Token `<?>` out of vocabulary``.

Against the committed Metal ledger (`benchmarks/RESULTS.md`), which has
frink at 68.5 tg128 and llama.cpp at 64.5 on that same 3B file, the
three-way on this hardware is frink 68.5, llama.cpp 64.5, xinfer 45.8.

So the performance half of "reach parity with xinfer" is already done
on Metal, and it was done by the work that closed the llama.cpp gap,
not by anything xinfer knows. Its published numbers are CUDA numbers
from rented 5090s and Hopper parts, where its speed comes from cutlass,
FlashAttention and FlashInfer through `attention-rs`, and from a forked
candle. frink's CUDA gap against llama.cpp is already measured and
already the target (`benchmarks/RESULTS.md`, CUDA tables); adding
xinfer to that comparison would not change the diagnosis, so it is not
worth a rented box yet.

One structural difference worth recording because it cost an hour:
xinfer compiles its `.metal` sources to `.air` at BUILD time, so it
does not build on a stock macOS without
`xcodebuild -downloadComponent MetalToolchain` (688 MB). frink compiles
Metal source at runtime and builds on the machine as shipped.

## What xInfer has that Frink does not

Read from its source, not its README.

1. **Multimodal.** Qwen3-VL, Gemma4, Mistral3-VL, Llama4 vision, as
   real graphs. frink has `vl_engine.rs` as a stub and every VL
   architecture sits in `capability` as `DeferredMultimodal`. This is
   also a llama.cpp parity row (`mtmd`), so it ranks above everything
   else on this list.
2. **Quantized safetensors.** FP8 blockwise, GPTQ, AWQ, MXFP4, NVFP4,
   each with its own kernel file in `attention-rs`. `frink-safetensors`
   reads F16 / BF16 / F32 / I8. llama.cpp does not do these either, so
   this is xinfer parity and not llama.cpp parity.
3. **ISQ**, quantizing a BF16 checkpoint to Q4K and friends while
   loading. frink quantizes offline (`frink quantize`), byte-identical
   to `llama-quantize`.
4. **TurboQuant's rotation.** See below. This is the one row on the
   list that is cheap, and the only one where frink ships the *name*
   without the algorithm.
5. **MTP / speculative decoding in the server.** frink has the
   drafter (`frink_models::speculative`), the CLI flag
   (`--model-draft`) and the accept-rate statistics, and no server
   route; `docs/plans/server-speculative-decoding.md` is the plan.
   llama.cpp's server has it, so this is a llama.cpp parity row too.
6. **Multi-GPU tensor parallel and multi-node**, and **PD
   disaggregation**. Already on the roadmap under serving.

And the other direction, so the list is honest: frink serves 98
audited architectures against xinfer's nineteen model files, every one
of them with a libllama golden; frink's HTTP surface is 24 routes
against xinfer's six, including `/v1/responses`, `/v1/messages`,
`/v1/rerank` and the admin set; and frink has the offline tools
(`quantize`, `imatrix`, `gguf-split`, `perplexity`, `parity`) that make
a GGUF engine usable without llama.cpp beside it.

## TurboQuant, which frink half has

`--ctk turbo4` exists here and is NOT TurboQuant. frink packs each 32
elements into an f16 scale plus 16 nibble bytes, per-group absmax, no
rotation, K and V alike (`frink_quant::pack_turbo4_kv_blocks`, and
`kv_append_turbo4` in `frink-metal/src/attn.rs`). What
`attention.rs`'s `flash_tq4_store` does is: per (token, head), flip the
sign of each channel by a deterministic hash, run a normalized
Walsh-Hadamard transform over the head vector, THEN quantize to 4 bits
with one absmax per head; V is quantized with no rotation; and Q is put
through the same sign flip and WHT at read time, which leaves `q . k`
unchanged because the transform is orthogonal and the sign diagonal is
its own inverse. The rotation is the whole point: it spreads the
outlier channels that make 4-bit KV lossy.

The cost of not having it, and what the rotation is actually worth,
took three metrics to answer, and the first two were wrong in a way
worth recording.

**A free-running greedy generation cannot measure a KV cache.** The
first attempt compared 200 greedy tokens against the f16 answer and
reported the character at which they part. Greedy decoding is chaotic:
one flipped token makes everything after it unrelated, so across six
prompts the same build scored anywhere from 0 to 134 characters and
the four schemes below could not be told apart. The number moves,
it just does not measure the thing.

**`frink perplexity` cannot measure it either**, which is worth
knowing because it looks like the obvious tool: its number is
byte-identical for `f16`, `q8_0` and `turbo4`, because that path never
touches the Metal KV store. A metric that does not move when the thing
under test changes is not evidence that the thing does not matter.

What does measure it is the next-token decision at a long context, with
no generation after it to amplify: sixty prefixes of the corpus, 2,000
to 20,172 characters, greedy, and the share where the store's answer is
the f16 store's answer. Llama-3.2-3B-Instruct Q4_K_M, head_dim 128:

| K scale granularity | K rotation | agreement with f16 |
|---|---|---:|
| per head | none | 41/60 |
| per head | WHT (xInfer's TurboQuant) | **52/60** |
| per 32-element group (frink's wire) | none (frink before this) | 45/60 |
| per 32-element group | **WHT** | **48/60** |

Read down the table and the rotation wins in both wires, by eleven
windows at a per-head scale and by three at frink's. Read across it and
the reading that looked obvious before the last cell arrived was wrong:
frink's finer scales are better than a per-head scale WITHOUT the
rotation, 45 against 41, and worse WITH it, 48 against 52. The first
three rows were measured before the fourth, and they supported a tidy
story about the two mechanisms being alternative ways to pay for the
same outlier; the fourth does not fit that story, and it is the row
that matters because it is what xInfer actually ships.

What landed is therefore the rotation on frink's existing wire: 45 to
48, no format change, and `turbo4` means what its name says. What is
NOT settled is the wire itself. A difference of four windows in sixty
is not resolvable at this sample size (the standard error on a share
near 0.8 is about three windows), so "52 beats 48" is a direction, not
a result, and turning it into one means either many more windows or a
paired per-window comparison that this harness does not keep. It is
recorded here rather than acted on, with the number, so the next person
to open the wire finds the question already asked.

The rotation is not free. At a 5,110-token context on the M2 Pro,
turbo4 decode goes from 37.8 to 35.8 tok/s with it, about 5%, against
f16's 51.8; prefill is unchanged at 476 against 480. Most of that 31%
gap to f16 predates this change and is the dequant-to-f16 scratch the
turbo4 read path goes through, not the rotation.

What none of these numbers is: a claim about model quality. Agreement
with an f16 store is agreement with frink's own f16 answer, not with a
reference implementation, and 48/60 still means a fifth of long-context
next tokens differ. `turbo4` buys context length, and it is not free at
any of these settings.

## Ranking

1. TurboQuant rotation on the `turbo4` path. Cheap, measured, named by
   the roadmap, and it retires a claim frink currently makes without
   the algorithm behind it.
2. Server-side speculative decoding. llama.cpp parity, plan written.
3. Multimodal. llama.cpp parity, and the largest thing on either list.
4. Quantized safetensors and ISQ. xInfer parity only, and both sit
   behind something neither of them is: a GENERIC safetensors loader.
   `grep -rln safetensors crates/frink-models/src` is Kimi K3's
   dedicated stack and one BERT pooler, and `safetensors_f32` widens
   exactly three float dtypes; there is no `config.json` to
   `ModelConfig` path for an arbitrary Hugging Face checkpoint. That
   prerequisite is most of the work and it duplicates the GGUF
   loader's architecture mapping against HF naming, which is why it
   ranks below the two llama.cpp rows rather than beside them.
   llama.cpp does not read HF checkpoints either.

## Defect found on the way

`FRINK_CTK` is documented in `docs/CONFIG.md` as "Same as `--ctk`", and
through `frink run` it cannot work: `run.rs:1222` does
`set_var("FRINK_CTK", args.ctk.trim())` unconditionally, so the flag's
default of `f16` overwrites whatever the environment said before any
Metal code reads it. Two spellings of one setting with nothing
enforcing that they agree, which is this repo's dominant bug shape;
filed separately.
