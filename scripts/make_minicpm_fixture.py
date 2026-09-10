#!/usr/bin/env python3
"""Generate the tiny synthetic `minicpm` GGUFs used by ferrox's MiniCPM
coverage test.

MiniCPM was never an UNAUDITED row. It was refused BY NAME, and the
reason is the reason this script writes two files:

`.scratch/llama.cpp/src/models/minicpm.cpp:5-7` assigns three scalar
multipliers before it reads anything:

```c
hparams.f_embedding_scale = 12.0f;
hparams.f_residual_scale  = 1.4f / sqrtf(float(hparams.n_layer()));
hparams.f_logit_scale     = hparams.n_embd ? (256.0f / float(hparams.n_embd)) : 1.0f;
```

and only THEN (`:12-14`) lets the file override them, each with
`required = false`. So a MiniCPM export that declares none of the three
keys is still scaled by all three, and a gate that looks for the keys
sees an ordinary file. That is why ferrox refused the architecture
string rather than detecting the feature, and it is why the DEFAULT
fixture this script writes declares **no scaling key at all**: a fixture
that declared them would pass with or without the hook and would prove
nothing.

`--declared` writes the other half, and it is not a nicety. The
defaults have to be applied BEFORE the file, not after: a hook that ran
the other way round would agree with llama.cpp on every file that omits
the keys and disagree on every file that carries them, which is the
subset real newer MiniCPM exports are in.

The rest of the row is Granite's, verbatim. `models.h:1594-1601` is
`using graph = llama_model_granite::graph` -- the same graph OBJECT, not
a similar one -- so everything `scripts/make_granite_fixture.py` says
about the graph applies here:

  * the embedding scale in the shared `build_inp_embd`
    (llama-graph.cpp:2337-2342, guarded `!= 0.0f`);
  * `f_residual_scale` on BOTH branch outputs, before each residual add
    (`granite.cpp:235-238`, `:288-292`);
  * `1.0f / f_logit_scale` on the final logits (`granite.cpp:180`);
  * RoPE is **NORM** (`LLM_ARCH_MINICPM` in `llama_model_rope_type`'s
    consecutive-pairs group, llama-model.cpp:2580).

Two differences from Granite, both load-bearing:

  * **No `attention.scale`.** `minicpm.cpp:3-24` contains no
    `LLM_KV_ATTENTION_SCALE`, so `hparams.f_attention_scale` keeps its
    `0.0f` and `granite.cpp:225` falls back to `1/sqrt(n_embd_head)`.
    Neither file here writes that key, and ferrox still refuses it for
    this architecture.
  * **The RoPE switch is unreachable.** Granite's graph gates RoPE
    entirely on `hparams.rope_finetuned` (`granite.cpp:206`), and
    `granite.cpp:33-35` lets a file set it false. `minicpm.cpp:17` pins
    it `true` with no key read at all, so a `minicpm` file carrying
    `minicpm.rope.scaling.finetuned = false` still rotates. MiniCPM is
    therefore deliberately absent from
    `rope_finetuned::ROPE_GATED_ON_FINETUNED`.

**Why the projections are drawn wider than the norms.** The embedding
scale of 12.0 is not ours to choose -- it is llama.cpp's default -- and
a pre-norm layer's branch output does not grow with the residual it
reads, because the RMSNorm divides that magnitude straight back out. At
the magnitudes the other fixtures in this repo use, a 12x residual would
leave the final RMSNorm handing the lm_head a vector that is ~92%
scaled embedding, and every sabotage that is not about the multipliers
-- the RoPE variant, the attention scale, the FFN -- would move the
logits by less than the comparison tolerance. `attn_output` and
`ffn_down` are drawn 6x wider so the branches stay visible against the
stream they join.

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_minicpm_fixture.py OUT.gguf
    PYTHONPATH=... python3 scripts/make_minicpm_fixture.py OUT.gguf --declared
    PYTHONPATH=... python3 scripts/make_minicpm_fixture.py OUT.gguf --attention-scale

The golden logits that go with them are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "minicpm"

# EIGHT layers, not the two every other fixture in this directory uses.
# `1.4/sqrt(n_layer)` is 0.99 at two layers -- within 1% of the identity
# -- so a two-layer file cannot tell MiniCPM's residual default from no
# residual scaling at all: measured at 3.3e-4, against a comparison
# tolerance of 1e-5 and sabotage margins elsewhere in this repo of 1e-2.
# At eight layers the default is 0.495 and halves every branch output,
# which is a fact a fixture can see.
N_LAYER = 8
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# Granite's graph asserts n_embd_head_k == n_embd_head_v == n_rot
# (granite.cpp:118), and MiniCPM runs that graph.
HEAD_DIM = N_EMBD // N_HEAD
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5

# What llama.cpp computes for THIS file when it declares nothing. Not
# written anywhere; here so the test can name the same numbers.
#   embedding = 12.0
#   residual  = 1.4 / sqrt(8)  = 0.49497475
#   logit     = 256 / 24       = 10.666667
#
# The `--declared` values, chosen to be nowhere near any of those, so a
# hook applied in the wrong order is visible rather than marginal.
DECLARED_LOGIT_SCALE = 2.5
DECLARED_RESIDUAL_SCALE = 0.6
DECLARED_EMBEDDING_SCALE = 2.0


def main(out_path: str, declared: bool, attention_scale: bool) -> None:
    # One seed for both variants, and the KV block above writes no
    # tensors, so the two files differ in their METADATA and in nothing
    # else. That is what lets the `--declared` golden be attributed to
    # the keys rather than to the weights.
    rng = np.random.default_rng(0x11CB4)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-minicpm-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    # minicpm.cpp:9 reads this one as REQUIRED.
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # THE POINT OF THE DEFAULT FILE IS THAT THIS BLOCK IS SKIPPED.
    #
    # No `minicpm.logit_scale`, no `minicpm.residual_scale`, no
    # `minicpm.embedding_scale` -- and llama.cpp still divides every
    # logit by 256/n_embd, still multiplies every branch output by
    # 1.4/sqrt(n_layer), and still multiplies every embedding row by 12.
    # `minicpm.attention.scale` is never written by either variant:
    # MiniCPM does not read that key, and ferrox refuses a file that
    # declares it.
    if declared:
        w.add_logit_scale(DECLARED_LOGIT_SCALE)
        w.add_residual_scale(DECLARED_RESIDUAL_SCALE)
        w.add_embedding_scale(DECLARED_EMBEDDING_SCALE)

    # `--attention-scale` writes the ONE key of the four that MiniCPM
    # does not read, to prove ferrox's refusal of it is reachable rather
    # than decorative. llama.cpp loads such a file and ignores the key
    # (`hparams.f_attention_scale` is never assigned from it, so
    # `granite.cpp:225` still uses `1/sqrt(n_embd_head)`); ferrox stops,
    # because honouring a number its own reference discards is the same
    # class of wrong as dropping one it applies. No golden goes with
    # this file -- it is never expected to load.
    if attention_scale:
        w.add_attention_scale(0.9)

    # Minimal SPM-flavoured vocab: llama.cpp needs tokens/scores/types to
    # build a vocab at all, but the fixture is always driven by explicit
    # token ids, never by tokenizing text.
    tokens = ["<unk>", "<s>", "</s>"] + [f"tok{i}" for i in range(3, N_VOCAB)]
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * N_VOCAB)
    types = [gguf.TokenType.CONTROL if i < 3 else gguf.TokenType.NORMAL for i in range(N_VOCAB)]
    w.add_token_types([int(t) for t in types])
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)

    # ne = [n_embd, n_vocab] -> numpy [n_vocab, n_embd].
    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        # Q and K wider, for the same reason every fixture in this repo
        # draws them wider: at one magnitude the attention scores over a
        # six-token prompt sit within a fraction of each other, softmax
        # comes out nearly uniform, and the RoPE-variant sabotage has
        # nothing to move.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        # NOTE: no `attn_output.bias`. minicpm.cpp:49 makes it optional
        # and ferrox would DROP it, so a fixture carrying one would be
        # refused by the unread-tensor gate rather than compared.
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q) * 6.0)

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        # SwiGLU: minicpm.cpp:62-64 creates gate/up/down and Granite's
        # graph runs LLM_FFN_SILU + LLM_FFN_PAR on them.
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 6.0)

    # minicpm.cpp:32 creates this one as REQUIRED, unlike `olmo`.
    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    # Untied lm_head (minicpm.cpp:33-38 accepts either). Untied keeps the
    # embedding scale and the lm_head as two separate facts, so a bug in
    # one cannot be cancelled by the other.
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "minicpm-fixture.gguf",
        declared="--declared" in sys.argv[1:],
        attention_scale="--attention-scale" in sys.argv[1:],
    )
