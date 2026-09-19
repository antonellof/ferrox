#!/usr/bin/env python3
"""Generate the tiny synthetic `exaone` GGUF used by frink's EXAONE 3.x
coverage test.

`exaone` is LG AI Research's EXAONE-3.0 / 3.5. It is NOT `exaone4`, which
is a different residual topology (no pre-attention and no pre-FFN norm)
and stays refusing, nor `exaone-moe`, whose full-attention layers get no
RoPE at all. This row sat on frink's generic GQA path refusing as
UNAUDITED, triaged FIXTURE-AWAY.

`.scratch/llama.cpp/src/models/exaone.cpp`:

  * `load_arch_hparams` (:3-10) reads NOTHING but the RMS epsilon.
  * `load_arch_tensors` (:12-40) creates `attn_norm`, split Q/K/V via
    `create_tensor_qkv` sized from `n_embd_head_k * n_head`, an
    `attn_output` of `{n_embd_head_k * n_head, n_embd}`, `ffn_norm` and
    `ffn_gate`/`ffn_up`/`ffn_down`.
  * The graph (:65-121) is the sequential residual with
    `LLM_FFN_SILU, LLM_FFN_PAR` SwiGLU (:106-110) and a
    `1/sqrt(n_embd_head)` attention scale (:93).
  * RoPE is **NEOX** -- pairs offset by n_rot/2 (`LLM_ARCH_EXAONE` in
    `llama_model_rope_type`'s NEOX group, llama-model.cpp:2655). Note
    the contrast with `internlm2`/`xverse`/`baichuan`/`ernie4_5`, which
    are NORM; the test sabotages this one in the opposite direction.

What the fixture pins:

  * **Tied output.** `output` is `TENSOR_NOT_REQUIRED` (:19) and falls
    back to `token_embd` (:22-24), so this file ships no `output.weight`
    and the lm_head has to come from the embedding matrix.
  * `head_dim != n_embd / n_head` (8, not 6), which :31-32 allows.
  * GQA: `n_head_kv = 2 < n_head = 4`.

DELIBERATELY ABSENT: the optional global `rope_freqs.weight` (:35,
`LLM_TENSOR_ROPE_FREQS` maps to the un-suffixed name `rope_freqs`,
llama-arch.cpp:386). It is llama3-style RoPE frequency scaling, which
EXAONE 3.x does not ship, and frink loads it on a shared path already
exercised by the Llama-3.x parity runs.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_exaone_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "exaone"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# Deliberately NOT n_embd / n_head (which would be 6).
HEAD_DIM = 8
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0xE7A04E)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-exaone-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

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

    # ne = [n_embd, n_vocab] -> numpy [n_vocab, n_embd]
    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        # Q and K are drawn WIDER than everything else on purpose. At
        # the magnitude the rest of the file uses, the attention scores
        # over a six-token prompt sit within a fraction of each other,
        # softmax comes out nearly uniform, and the layer stops caring
        # where the tokens are -- which leaves the RoPE-variant sabotage
        # test in `tests/fixture_away_graphs.rs` with a margin of ~1e-3
        # rather than the ~1e-1 it should have. Real checkpoints have
        # peaked attention; a fixture that does not cannot see a
        # positional bug.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    # NOTE: no `output.weight`. exaone.cpp:19-24 falls back to the token
    # embedding, and this file is the tied case.

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "exaone-fixture.gguf")
