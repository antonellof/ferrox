#!/usr/bin/env python3
"""Generate the tiny synthetic `nemotron` GGUFs used by ferrox's
biased-LayerNorm coverage test.

`nemotron` (Nemotron-4, Minitron) refused as a `dedicated` row for its
REQUIRED LayerNorm biases: `build_norm(x, w, b, LLM_NORM, il)` at the
pre-attention, pre-FFN and final sites (`src/models/nemotron.cpp:71-74,
111-114,136-139`), the six per-layer tensors and both output-norm
tensors created with `create_tensor(..., 0)` (`:18-19,25-26,33-34`).
The rest the generic decoder already served: `create_tensor_qkv` with
no biases present, the UNGATED ReLU-squared FFN over `ffn_up` /
`ffn_down` (`:37-38,118-123`, `arcee`'s), partial NEOX RoPE
(`conversion/nemotron.py:170-174` writes `rope.dimension_count = int(
partial_rotary_factor * n_embd) // n_head`; llama-model.cpp's NEOX
group), `kq_scale = 1/sqrt(head_dim)`, an untied `output`.

Keys, as `conversion/nemotron.py:162-181` writes them:
`attention.layer_norm_epsilon` (the LayerNorm key), `vocab_size`,
`rope.dimension_count`, and `rope.scaling.type = none` (or `linear`
with a factor, which the generic path serves).

`--biases` adds the OPTIONAL `attn_output.bias`, `ffn_up.bias` and
`ffn_down.bias` (`:31,40-41`, `TENSOR_NOT_REQUIRED`), which llama.cpp
adds after `wo`, `ffn_up` and `ffn_down`. The generic dense path has no
slot for them and leaves them UNREAD, which `assert_every_tensor_consumed`
refuses; the variant pins that refusal from a file libllama runs.

`NormOp::LayerNormBias` is the seam. The biases are drawn AWAY from
zero and the weights away from one, so a bias dropped or a weight
skipped moves the logits by far more than the tolerance.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_nemotron_fixture.py OUT.gguf [--biases]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "nemotron"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
ROPE_DIM = 4  # partial_rotary_factor 0.5
N_FF = 48
N_VOCAB = 48
CTX = 64
LN_EPS = 1e-5
ROPE_BASE = 10000.0


def main(out_path: str, biases: bool) -> None:
    rng = np.random.default_rng(0x4E30)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w() -> np.ndarray:
        return (1.5 + rng.standard_normal(N_EMBD) * 0.5).astype(np.float32)

    def norm_b() -> np.ndarray:
        return (0.5 + rng.standard_normal(N_EMBD) * 0.5).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-nemotron-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_rope_freq_base(ROPE_BASE)
    # conversion/nemotron.py:165-179.
    w.add_vocab_size(N_VOCAB)
    w.add_layer_norm_eps(LN_EPS)
    w.add_rope_dimension_count(ROPE_DIM)
    w.add_rope_scaling_type(gguf.RopeScalingType.NONE)
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
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_norm.weight", norm_w())
        w.add_tensor(p + "ffn_norm.bias", norm_b())
        # :37-38: up and down, no gate.
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 4.0)
        if biases:
            w.add_tensor(p + "attn_output.bias", rnd(N_EMBD))
            w.add_tensor(p + "ffn_up.bias", rnd(N_FF))
            w.add_tensor(p + "ffn_down.bias", rnd(N_EMBD))

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
    ap.add_argument("out", nargs="?", default="nemotron-fixture.gguf")
    ap.add_argument("--biases", action="store_true")
    args = ap.parse_args()
    main(args.out, args.biases)
