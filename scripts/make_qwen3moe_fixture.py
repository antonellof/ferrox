#!/usr/bin/env python3
"""Generate the tiny synthetic `qwen3moe` GGUF used by frink's Qwen3-MoE
coverage test.

`qwen3moe` is what Qwen3-30B-A3B / Qwen3-235B-A22B tag, which makes it
one of the most-run MoE architectures on consumer hardware, and it sat on
frink's *generic* GQA path with no evidence behind it. This fixture is
the evidence.

It is deliberately small (2 layers, 32-wide, 6 experts) and carries every
structural feature of `src/models/qwen3moe.cpp` that the generic decoder
could get wrong:

  * `blk.N.attn_{q,k}_norm.weight` of length `head_dim`, applied
    **per head** and **before** RoPE (qwen3moe.cpp:99,108). A whole-vector
    QK-norm, or one applied after RoPE, is a different model.
  * `head_dim * n_head != n_embd`. Qwen3 declares `key_length` /
    `value_length` explicitly (real 30B-A3B: n_embd 2048, n_head 32,
    head_dim 128 -> a 4096-wide Q), so `attn_output.weight` is
    `{n_embd_head_k * n_head, n_embd}` and not square. A decoder that
    derives `head_dim = n_embd / n_head` gets 8 here instead of 16.
  * GQA: `n_head_kv < n_head`.
  * NEOX RoPE (`llama_model_rope_type`'s NEOX group), not interleaved.
  * Softmax gating with **renormalised** top-k weights: qwen3moe.cpp:141
    passes `norm_w = true` and
    `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` as literals. Note it reads
    *none* of `expert_weights_norm` / `expert_gating_func` /
    `expert_weights_scale` from the file, unlike dots1.
  * No shared expert and no `exp_probs_b` bias -- the MoE output is added
    straight back to the residual.
  * `n_ff != n_ff_exp`. The expert FFNs must be sized from
    `expert_feed_forward_length`, never from `feed_forward_length`.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_qwen3moe_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "qwen3moe"

N_LAYER = 2
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
# Deliberately NOT n_embd / n_head (which would be 8): Qwen3 carries an
# explicit head_dim and the generic decoder must read it.
HEAD_DIM = 16
N_EXPERT = 6
N_EXPERT_USED = 2
# Deliberately different from N_FF_EXP: the experts are sized from
# `expert_feed_forward_length`.
N_FF = 40
N_FF_EXP = 16
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x93003)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-qwen3moe-fixture")
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
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_feed_forward_length(N_FF_EXP)
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

        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        # Per-head RMSNorm weights: one `head_dim`-long vector shared by
        # every head, applied before RoPE.
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM))
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM))

        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))

        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        # gate/up: ne = [n_embd, n_ff_exp, n_expert]
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        # down: ne = [n_ff_exp, n_embd, n_expert]
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen3moe-fixture.gguf")
