#!/usr/bin/env python3
"""Generate the tiny synthetic `olmo` GGUFs used by ferrox's OLMo-1
coverage test.

`olmo` is AI2's OLMo-1, and it is NOT `olmo2`. It sat on ferrox's
generic GQA path refusing as UNAUDITED, triaged NEW CODE, and the
blocker is the norm FUNCTION rather than the residual wiring:

`.scratch/llama.cpp/src/models/olmo.cpp:15-36` creates Q/K/V,
`attn_output` and gate/up/down and **not one norm tensor** -- no
`attn_norm`, no `ffn_norm`, no `output_norm` -- and the graph normalises
at all three sites with a null weight AND a null bias:

    cur = build_norm(inpL, NULL, NULL, LLM_NORM, il);   # :65-67
    cur = build_norm(ffn_inp, NULL, NULL, LLM_NORM, il) # :104-106
    cur = build_norm(cur, NULL, NULL, LLM_NORM, -1)     # :128-130

`LLM_NORM` is `ggml_norm`: subtract the mean, divide by the standard
deviation over the BIASED variance
(`ggml/src/ggml-cpu/ops.cpp:3716-3745`). With both weight and bias null
`build_norm` does nothing further. So OLMo-1 is a pre-norm layer like
`llama` -- :65-67 before attention, :104-106 before the FFN -- with a
different norm function and no parameters at all. That is a THIRD shape
beside `llama`'s RMSNorm and the post-norm-only topology `olmo2` and
`exaone4` share, and `crate::norm::NormOp::LayerNormNoParams` is it.

What this fixture pins beyond that, each against the C:

  * **No norm tensors anywhere.** The file ships none, so a decoder that
    demands `blk.N.attn_norm.weight` or `output_norm.weight` cannot load
    it, and a decoder that finds one somewhere is reading a file this
    architecture does not produce.
  * **A TIED lm_head.** :21-25 creates `output` as `TENSOR_NOT_REQUIRED`
    and falls back to `token_embd`. This file ships no `output.weight`,
    which is what `conversion/olmo.py` produces for OLMo-7B.
  * **NORM RoPE**, the consecutive-pairs variant (`LLM_ARCH_OLMO` is in
    `llama_model_rope_type`'s NORM group, llama-model.cpp:2585). That is
    also why `conversion/olmo.py:33-36` permutes `q_proj` and `k_proj`
    exactly as `LlamaModel` does.
  * **`1/sqrtf(float(n_embd_head))`** passed literally at :94, so
    `ModelConfig::attention_scale` must stay `None`.
  * **`LLM_FFN_SILU, LLM_FFN_PAR`** (:109-114): plain SwiGLU with a
    separate `ffn_gate`.
  * **GQA**: :30 sizes K and V `n_embd_gqa`, and this file uses 2 KV
    heads against 4 Q heads so the grouping is exercised.
  * `n_embd_head == n_rot` is asserted at :46-47, so head_dim is
    `n_embd / n_head` and not free.
  * The epsilon comes from `LLM_KV_ATTENTION_LAYERNORM_EPS` (:4), NOT
    the RMS one, and `conversion/olmo.py:22` writes it as
    `olmo.attention.layer_norm_epsilon`. ferrox reads either spelling
    into one field, so this file writes the LayerNorm spelling and
    nothing else.

**`--clamp` writes the one thing ferrox refuses.** `olmo.cpp:5` reads an
optional `{arch}.attention.clamp_kqv` and `llama-graph.cpp:1611-1652`
clamps Q, K and V by it inside `build_qkv`. ferrox clamps no projection
on any path and stops instead of running unclamped -- see
`crate::clamp_kqv`. The gate is REACHABLE from real checkpoints, not
only from a hand-written file: `conversion/olmo.py:23-25` writes the key
whenever the HF config carries a `clip_qkv`, which OLMo-7B-Twin-2T and
OLMo-1.7-7B do (8.0) and the original OLMo-7B does not (null). This
variant is what drives that refusal from a real file rather than
asserting it exists.

**Why the projections are drawn wider.** Same reason every fixture here
does it: at one magnitude the attention scores over a six-token prompt
sit within a fraction of each other, softmax comes out nearly uniform,
and the RoPE-variant sabotage has nothing to move.

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_olmo_fixture.py OUT.gguf
    PYTHONPATH=... python3 scripts/make_olmo_fixture.py OUT.gguf --clamp

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "olmo"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# olmo.cpp:44-47 asserts n_embd_head == n_embd_head_k() == n_rot, and
# :31 sizes `wo` {n_embd, n_embd}, so this is not free.
HEAD_DIM = N_EMBD // N_HEAD  # 6
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
# `LLM_KV_ATTENTION_LAYERNORM_EPS`, not the RMS one. conversion/olmo.py:22
# hardcodes 1e-5.
LAYER_NORM_EPS = 1e-5

# The value `--clamp` declares. Real OLMo-1.7-7B ships clip_qkv = 8.0.
CLAMP_KQV = 8.0


def main(out_path: str, clamp: bool) -> None:
    rng = np.random.default_rng(0x0117A)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-olmo-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    # THE LAYERNORM SPELLING. `add_layer_norm_rms_eps` would be the wrong
    # key for this architecture: olmo.cpp:4 reads
    # LLM_KV_ATTENTION_LAYERNORM_EPS.
    w.add_layer_norm_eps(LAYER_NORM_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    if clamp:
        w.add_clamp_kqv(CLAMP_KQV)

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

    # ne = [n_embd, n_vocab] -> numpy [n_vocab, n_embd]. This is also the
    # lm_head: olmo.cpp:21-25 ties them when `output` is absent, and this
    # file has no `output.weight`.
    #
    # Drawn with a NON-ZERO MEAN on purpose. A LayerNorm subtracts the
    # mean of the vector it norms and an RMSNorm does not, so a fixture
    # whose hidden states are centred by construction would make the two
    # functions agree and would not be able to see the difference it
    # exists to pin.
    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD) + 0.5)

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM

    for il in range(N_LAYER):
        p = f"blk.{il}."

        # NO `attn_norm` and NO `ffn_norm`, and unlike olmo2 no
        # `post_attention_norm` or `post_ffw_norm` either: OLMo-1 has no
        # norm tensor of any kind. That absence is the whole row.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        # Biased so the residual stream keeps a non-zero mean for the
        # next layer's LayerNorm to remove.
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD) + 0.3)
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) + 0.3)

    # NO `output_norm.weight` and NO `output.weight`.

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "olmo-fixture.gguf",
        clamp="--clamp" in sys.argv[1:],
    )
