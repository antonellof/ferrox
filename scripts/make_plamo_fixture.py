#!/usr/bin/env python3
"""Generate the tiny synthetic `plamo` GGUF used by ferrox's
parallel-residual coverage test.

`plamo` (PLaMo-13B) refused as a `dedicated` row for its parallel
residual: `plamo.cpp:64` keeps the attention input `sa_inp = attn_norm(x)`,
`:97-98` hands THAT vector to the FFN (`cur = sa_inp`), and `:111-112`
sums `ffn_out + sa_out + inpL` once. One RMSNorm per layer, no `ffn_norm`
tensor at all (`:23-30`). That is `crate::parallel_residual`'s
`SharedNorm` arm on every layer (`ParallelWhen::Always`).

The rest is a Llama, as `conversion/plamo.py` writes it: split Q/K/V
with no biases, `head_count 40` / `head_count_kv 5` (GQA 8:1; the
fixture keeps the ratio at 4:1 on four heads), `attention.layer_norm_
rms_epsilon`, `context_length 4096` (hard-coded in the converter), NO
`rope.dimension_count` (`:41` asserts `n_embd_head == n_rot`, the whole
head rotates), no `rope.freq_base` (10000), NEOX RoPE (llama-model.cpp:
2639), SwiGLU, `output` REQUIRED.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_plamo_fixture.py OUT.gguf

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "plamo"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 1
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
RMS_EPS = 1e-6


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x9AB0)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w(n: int) -> np.ndarray:
        return (1.0 + rng.standard_normal(n) * 0.3).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-plamo-fixture")
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_block_count(N_LAYER)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(RMS_EPS)
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
        w.add_tensor(p + "attn_norm.weight", norm_w(N_EMBD))
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 2.0)

    w.add_tensor("output_norm.weight", norm_w(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="plamo-fixture.gguf")
    args = ap.parse_args()
    main(args.out)
