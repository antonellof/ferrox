#!/usr/bin/env python3
"""Generate the tiny synthetic `seed_oss` GGUF used by frink's Seed-OSS
coverage test.

`seed_oss` is ByteDance's Seed-OSS-36B. It sat on frink's generic GQA
path refusing as UNAUDITED, triaged ONE MATCH ARM: it stores its
**pre-FFN** norm as `blk.N.post_attention_norm.weight` and carries no
`blk.N.ffn_norm.weight` at all, which is gpt-oss's slot and not Gemma's
post-attention slot. This fixture is the evidence that the widened slot
computes llama.cpp's graph.

It is deliberately small (2 layers, 24-wide, dense) and carries every
structural feature of `.scratch/llama.cpp/src/models/seed-oss.cpp` that
the generic decoder could get wrong:

  * `attn_norm` before attention and `attn_post_norm` applied to
    `ffn_inp` -- i.e. AFTER the attention residual (seed-oss.cpp:36-37
    creates both and nothing else; :113-115 norms `ffn_inp` with
    `attn_post_norm`). A decoder that treated it as Gemma's
    post-attention norm would apply it inside the attention residual and
    then have no pre-FFN norm to apply.
  * NEOX RoPE (`llama_model_rope_type`'s NEOX group,
    llama-model.cpp:2669), not the interleaved NORM pairing.
  * `head_dim * n_head != n_embd`: seed-oss.cpp:15-17 sizes Q/K/V from
    `n_embd_head_k` explicitly, so `attn_output.weight` is
    `{n_head * head_dim, n_embd}` and not square. A decoder deriving
    `head_dim = n_embd / n_head` gets 6 here instead of 8.
  * GQA: `n_head_kv = 2 < n_head = 4`.
  * Dense SiLU SwiGLU on a sequential residual (:117-127), no MoE, no
    QK-norm, no attention sinks, no softcap.
  * `1/sqrt(head_dim)` attention scale (:66, no `f_attention_scale` key).

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_seed_oss_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "seed_oss"

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
    rng = np.random.default_rng(0x5EED05)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-seed-oss-fixture")
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
        # seed_oss's PRE-FFN norm. There is deliberately no
        # `ffn_norm.weight` in this file: that absence is half of what
        # the test is pinning.
        w.add_tensor(p + "post_attention_norm.weight", rnd(N_EMBD))

        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "seed-oss-fixture.gguf")
