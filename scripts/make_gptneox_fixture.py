#!/usr/bin/env python3
"""Generate the tiny synthetic `gptneox` GGUFs used by ferrox's
parallel-residual coverage test.

`gptneox` (Pythia, GPT-NeoX-20B, Dolly-v2) refused as a `dedicated` row
for its parallel residual: `gptneox.cpp:143-166` computes
`x + attn(ln1(x)) + ffn(ln2(x))` when `use_parallel_residual` is true
(`:5`), the FFN reading its OWN LayerNorm of the LAYER INPUT rather than
of the post-attention residual, and `:167-195` the ordinary sequential
form when it is false. That is `crate::parallel_residual`'s `TwoNorms`
arm, decided by the key (`ParallelWhen::ParallelResidualKey`), and this
script writes both values of it.

The rest of the graph, as `conversion/gptneox.py` writes it: the biased
LayerNorm at every site (`:57-58,63-64,72-73`, `NormOp::LayerNormBias`),
a FUSED `attn_qkv.weight` / `.bias` in Q-then-K-then-V row order (the
converter de-interleaves HF's per-head `[q,k,v]` blocks, `gptneox.py:
27-50`; `qkv_fused` splits it), REQUIRED `attn_output.bias`,
`ffn_up.bias`, `ffn_down.bias` (`:67,76,79`, `crate::proj_bias`), the
ungated GELU FFN (`LLM_FFN_GELU, LLM_FFN_SEQ`, `:155-161`), NEOX RoPE
over `rope.dimension_count = rotary_pct * head_dim` (Pythia's 0.25), NO
`head_count_kv` key (multi-head; llama.cpp defaults it to `head_count`),
no `rope.freq_base` key (10000), `output` REQUIRED.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_gptneox_fixture.py OUT.gguf [--sequential]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "gptneox"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
HEAD_DIM = 8
ROPE_DIM = 2  # rotary_pct 0.25
N_FF = 64
N_VOCAB = 48
CTX = 64
LN_EPS = 1e-5


def main(out_path: str, sequential: bool) -> None:
    rng = np.random.default_rng(0x6E0E)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w(n: int) -> np.ndarray:
        return (1.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    def norm_b(n: int) -> np.ndarray:
        return (0.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-gptneox-fixture")
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_rope_dimension_count(ROPE_DIM)
    w.add_head_count(N_HEAD)
    w.add_parallel_residual(not sequential)
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
    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", norm_w(N_EMBD))
        w.add_tensor(p + "attn_norm.bias", norm_b(N_EMBD))
        # Q rows, then K rows, then V rows (`create_tensor_qkv` sizes it
        # `{n_embd, n_embd + 2*n_embd_gqa}`; `build_qkv` views it in that
        # order).
        q = rnd(n_embd_q, N_EMBD) * 2.0
        k = rnd(n_embd_q, N_EMBD) * 2.0
        v = rnd(n_embd_q, N_EMBD)
        w.add_tensor(p + "attn_qkv.weight", np.concatenate([q, k, v], axis=0))
        w.add_tensor(p + "attn_qkv.bias", rnd(3 * n_embd_q))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "attn_output.bias", rnd(N_EMBD))
        w.add_tensor(p + "ffn_norm.weight", norm_w(N_EMBD))
        w.add_tensor(p + "ffn_norm.bias", norm_b(N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.bias", rnd(N_FF))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 2.0)
        w.add_tensor(p + "ffn_down.bias", rnd(N_EMBD))

    w.add_tensor("output_norm.weight", norm_w(N_EMBD))
    w.add_tensor("output_norm.bias", norm_b(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="gptneox-fixture.gguf")
    ap.add_argument("--sequential", action="store_true")
    args = ap.parse_args()
    main(args.out, args.sequential)
