#!/usr/bin/env python3
"""Generate the tiny synthetic `phi2` GGUFs used by frink's Phi-2
coverage test.

`phi2` (Phi-2, Phi-1.5) refused as a `dedicated` row for its parallel
residual; once that was served (`crate::parallel_residual`) what it still
needed was ONE slot: `output.bias` on the LM head (`phi2.cpp:22`,
REQUIRED; `:136` adds it after `build_lora_mm`). `phi2.cpp` is:

  * `:19-20,27-28` the biased LayerNorm (`NormOp::LayerNormBias`).
  * `:30` `create_tensor_qkv`: split or fused Q/K/V with OPTIONAL biases
    (Phi-2 has them; `conversion/phi.py` maps HF's `q_proj` / `k_proj` /
    `v_proj` or the older fused `Wqkv` onto the matching names).
  * `:33,36,39` REQUIRED `attn_output.bias`, `ffn_down.bias`,
    `ffn_up.bias` (`crate::proj_bias`); `:108-114` the ungated GELU FFN.
  * `:67,108,116-117` the shared-norm PARALLEL residual: the FFN reads
    `attn_norm(x)` and `cur + ffn_output + inpL` are summed once.
  * `:21-22` `output.weight` REQUIRED and `output.bias` REQUIRED.
  * NEOX RoPE over `rope.dimension_count = partial_rotary_factor *
    head_dim` (Phi-2's 0.4 of 80 = 32); `head_count_kv = head_count`.

Keys as `conversion/phi.py:20-35` writes them: `context_length`,
`embedding_length`, `feed_forward_length = 4 * hidden`, `block_count`,
`head_count`, `head_count_kv`, `attention.layer_norm_epsilon`,
`rope.dimension_count`, `tokenizer.ggml.add_bos_token = false`; no
`rope.freq_base` (10000).

Shapes:

  * (default) split Q/K/V with biases, as the current converter writes.
  * `--fused`: one `attn_qkv.weight` / `.bias` (older exports).

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_phi2_fixture.py OUT.gguf [--fused]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "phi2"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
HEAD_DIM = 8
ROPE_DIM = 4  # partial_rotary_factor 0.5 of 8 (Phi-2: 0.4 of 80 = 32)
N_FF = 4 * N_EMBD
N_VOCAB = 48
CTX = 64
LN_EPS = 1e-5


def main(out_path: str, fused: bool) -> None:
    rng = np.random.default_rng(0x9412)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w(n: int) -> np.ndarray:
        return (1.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    def norm_b(n: int) -> np.ndarray:
        return (0.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-phi2-fixture")
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_block_count(N_LAYER)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD)
    w.add_layer_norm_eps(LN_EPS)
    w.add_rope_dimension_count(ROPE_DIM)
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
    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", norm_w(N_EMBD))
        w.add_tensor(p + "attn_norm.bias", norm_b(N_EMBD))
        q, k, v = rnd(n_embd_q, N_EMBD) * 2.0, rnd(n_embd_q, N_EMBD) * 2.0, rnd(n_embd_q, N_EMBD)
        qb, kb, vb = rnd(n_embd_q), rnd(n_embd_q), rnd(n_embd_q)
        if fused:
            w.add_tensor(p + "attn_qkv.weight", np.concatenate([q, k, v], axis=0))
            w.add_tensor(p + "attn_qkv.bias", np.concatenate([qb, kb, vb]))
        else:
            w.add_tensor(p + "attn_q.weight", q)
            w.add_tensor(p + "attn_k.weight", k)
            w.add_tensor(p + "attn_v.weight", v)
            w.add_tensor(p + "attn_q.bias", qb)
            w.add_tensor(p + "attn_k.bias", kb)
            w.add_tensor(p + "attn_v.bias", vb)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "attn_output.bias", rnd(N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.bias", rnd(N_FF))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 2.0)
        w.add_tensor(p + "ffn_down.bias", rnd(N_EMBD))

    w.add_tensor("output_norm.weight", norm_w(N_EMBD))
    w.add_tensor("output_norm.bias", norm_b(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))
    # :22 -- REQUIRED, and drawn wide enough that dropping it moves the
    # argmax, not only the logits.
    w.add_tensor("output.bias", rnd(N_VOCAB) * 4.0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="phi2-fixture.gguf")
    ap.add_argument("--fused", action="store_true")
    args = ap.parse_args()
    main(args.out, args.fused)
