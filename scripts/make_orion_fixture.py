#!/usr/bin/env python3
"""Generate the tiny synthetic `orion` GGUF used by ferrox's biased-LayerNorm
coverage test.

`orion` (Orion-14B) refused as a `dedicated` row because its every norm
is a LayerNorm WITH A BIAS -- `build_norm(x, w, b, LLM_NORM, il)` at the
pre-attention, pre-FFN and final sites (`src/models/orion.cpp:63-66,
104-107,127-130`), the weights and biases all REQUIRED (`:17-18,24-25,
30-31`). Everything else is a Llama: `create_tensor_qkv` with no biases
present, `wo`, a SwiGLU `ffn_gate` / `ffn_up` / `ffn_down`, NEOX RoPE
(llama-model.cpp's NEOX group), `kq_scale = 1/sqrt(head_dim)`, an
untied `output`.

Keys, as `conversion/orion.py:13-37` writes them: `context_length`,
`embedding_length`, `block_count`, `feed_forward_length`, `head_count`,
`head_count_kv`, and `attention.layer_norm_epsilon` -- the LayerNorm
key, written from the config's `rms_norm_eps` ("config provides rms
norm but it is actually layer norm", `:35-36`). NO `rope.dimension_count`
and NO `rope.freq_base`: llama.cpp defaults them to `head_dim` and
10000 (`llama-model.cpp`), and so must a reader.

`NormOp::LayerNormBias` is the seam. The biases are drawn AWAY from
zero and the weights away from one, so a bias dropped or a weight
skipped moves the logits by far more than the tolerance; the
projections are drawn at half the scale the other fixtures use,
because with the norm's gain on top of a x4 draw the SwiGLU's
activations were large enough that ggml's vectorised `expf` and
ferrox's `exp` disagreed at 9e-5 in the logits (KL 5.5e-10) -- noise,
and above the 1e-5 line the suite holds every fixture to.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_orion_fixture.py OUT.gguf

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "orion"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
LN_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x0410)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w() -> np.ndarray:
        return (1.5 + rng.standard_normal(N_EMBD) * 0.5).astype(np.float32)

    def norm_b() -> np.ndarray:
        return (0.5 + rng.standard_normal(N_EMBD) * 0.5).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-orion-fixture")
    # conversion/orion.py:27-37, and nothing about RoPE.
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_eps(LN_EPS)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

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

    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM
    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", norm_w())
        w.add_tensor(p + "attn_norm.bias", norm_b())
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_norm.weight", norm_w())
        w.add_tensor(p + "ffn_norm.bias", norm_b())
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 2.0)

    w.add_tensor("output_norm.weight", norm_w())
    w.add_tensor("output_norm.bias", norm_b())
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="orion-fixture.gguf")
    args = ap.parse_args()
    main(args.out)
