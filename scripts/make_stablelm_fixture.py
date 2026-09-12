#!/usr/bin/env python3
"""Generate the tiny synthetic `stablelm` GGUFs used by ferrox's StableLM
coverage test.

`stablelm` (StableLM-2-1.6B / 12B, StableLM-3B-4E1T) refused as a
`dedicated` row for its REQUIRED LayerNorm biases (`stablelm.cpp:20,28`,
plus the optional `ffn_norm.bias`, `:39`). That norm is
`NormOp::LayerNormBias` now, and the graph has THREE shapes behind one
architecture string, decided by TENSOR PRESENCE and never by a key:

  * `ffn_norm` PRESENT (`:38-39`, `TENSOR_NOT_REQUIRED`): the ordinary
    sequential layer, `x + attn(ln1(x))` then `+ ffn(ln2(...))`
    (`:129-133`). StableLM-2-1.6B, StableLM-3B-4E1T. SERVED.
  * `ffn_norm` ABSENT: the PARALLEL residual, `cur = inpSA` at `:135-137`
    -- the FFN reads the SAME normed input attention read, and the layer
    output is `inpL + attn + ffn`. StableLM-2-12B. REFUSED by name
    (`crate::parallel_residual`), from a fixture llama.cpp runs.
  * `attn_q_norm` / `attn_k_norm` PRESENT (`:34-35`, `{n_embd_head_k,
    n_head}`, DISTINCT weights per head, applied as `LLM_NORM` -- a
    LayerNorm, not RMS -- `:84-95`, before RoPE). StableLM-2-12B again
    (it has both). REFUSED by name (`crate::qk_layer_norm`): ferrox's
    per-head QK norm is one RMS weight shared by every head.

`use_parallel_residual` is WRITTEN by the converter (`conversion/
stablelm.py:35`) and READ BY NOTHING in `stablelm.cpp`: the graph
decides by `ffn_norm`. `--par-key` writes the key `true` on the
sequential file; libllama's logits are byte-identical to the file
without it (measured), and the test pins that ferrox ignores it too.

Shapes, one script:

  * (default) sequential, no QK norm: the served shape.
  * `--parallel`: no `ffn_norm` tensors.
  * `--qk-norm`: per-head `attn_q_norm` / `attn_k_norm`.
  * `--par-key`: the dead key on the sequential file.

Keys, as the converter writes them (`:27-36`): `context_length`,
`embedding_length`, `block_count`, `feed_forward_length`,
`rope.dimension_count = partial_rotary_factor * head_dim`
(StableLM-2's 0.25), `head_count`, `head_count_kv`, `use_parallel_
residual`, `attention.layer_norm_epsilon`. Q/K/V biases through
`create_tensor_qkv` (StableLM-2 has them, `use_qkv_bias`), no `wo`
bias, SwiGLU, NEOX RoPE (llama-model.cpp:2624), `output` REQUIRED.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_stablelm_fixture.py OUT.gguf [--parallel] [--qk-norm] [--par-key]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "stablelm"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
ROPE_DIM = 2  # partial_rotary_factor 0.25
N_FF = 48
N_VOCAB = 48
CTX = 64
LN_EPS = 1e-5
ROPE_BASE = 10000.0


def main(out_path: str, parallel: bool, qk_norm: bool, par_key: bool) -> None:
    rng = np.random.default_rng(0x57AB)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w(*shape: int) -> np.ndarray:
        return (1.5 + rng.standard_normal(shape) * 0.5).astype(np.float32)

    def norm_b() -> np.ndarray:
        return (0.5 + rng.standard_normal(N_EMBD) * 0.5).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-stablelm-fixture")
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_rope_dimension_count(ROPE_DIM)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    # :35 -- always written; read by nothing in the graph.
    w.add_parallel_residual(par_key or parallel)
    w.add_layer_norm_eps(LN_EPS)
    w.add_rope_freq_base(ROPE_BASE)
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
        w.add_tensor(p + "attn_norm.bias", norm_b())
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_q.bias", rnd(n_embd_q))
        w.add_tensor(p + "attn_k.bias", rnd(n_embd_kv))
        w.add_tensor(p + "attn_v.bias", rnd(n_embd_kv))
        if qk_norm:
            # :34-35 ne = [n_embd_head_k, n_head] -> numpy (n_head, head_dim):
            # one LayerNorm weight PER HEAD.
            w.add_tensor(p + "attn_q_norm.weight", norm_w(N_HEAD, HEAD_DIM))
            w.add_tensor(p + "attn_k_norm.weight", norm_w(N_HEAD_KV, HEAD_DIM))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        if not parallel:
            w.add_tensor(p + "ffn_norm.weight", norm_w(N_EMBD))
            w.add_tensor(p + "ffn_norm.bias", norm_b())
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 2.0)

    w.add_tensor("output_norm.weight", norm_w(N_EMBD))
    w.add_tensor("output_norm.bias", norm_b())
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="stablelm-fixture.gguf")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--qk-norm", action="store_true")
    ap.add_argument("--par-key", action="store_true")
    args = ap.parse_args()
    main(args.out, args.parallel, args.qk_norm, args.par_key)
