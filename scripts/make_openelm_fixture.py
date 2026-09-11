#!/usr/bin/env python3
"""Generate the tiny synthetic `openelm` GGUF used by ferrox's per-layer
shape coverage test.

`openelm` (Apple OpenELM) was triaged NEW CODE on PER-LAYER SHAPES:
`.scratch/llama.cpp/src/models/openelm.cpp:26-28` reads
`hparams.n_head(i)`, `n_head_kv(i)` and `n_ff(i)` for every layer, :34
sizes that layer's fused `wqkv` as
`{n_embd, (2*n_head_kv(i) + n_head(i)) * n_embd_head_k}`, and the graph
re-derives the three widths at :67-69. `conversion/openelm.py:57-59`
writes all three keys as ARRAYS, which is why the row did not even reach
the unaudited gate before: `GgufValue::as_u64` returns `None` for an
array and the load died on a missing-hparam error for keys the file
carries.

What this fixture pins, each against the C:

  * **Three layers, three shapes.** `head_count = [4, 2, 3]`,
    `head_count_kv = [2, 1, 3]`, `feed_forward_length = [40, 24, 32]`,
    so no two layers share a KV width or an FFN width and a cache or a
    tensor sized from any one layer's counts is wrong for the others.
    Layer 2 is MHA (3 over 3) beside two GQA layers.
  * **One fused `attn_qkv` per layer**, Q rows then K rows then V rows
    (:73-80 view it at offsets 0, `n_head`, `n_head + n_head_kv`).
  * **PER-HEAD QK-norm BEFORE RoPE** (:82-90 then :92-102), sized
    `{n_embd_head_k}` (:36-37), drawn away from 1 so the order shows.
  * **NEOX RoPE** (`LLM_ARCH_OPENELM` is in `llama_model_rope_type`'s
    NEOX group, llama-model.cpp:2650) over the whole head
    (`rope_dimension_count = head_dim`, conversion/openelm.py:65).
  * **A TIED lm_head with no fallback**: :22 creates `output` from
    `token_embd` unconditionally, so this file has no `output.weight`.
  * **`1/sqrt(n_embd_head)`** passed literally at :117, so
    `attention_scale` stays `None`; gated SiLU FFN (:132-137); RMS
    epsilon `1e-6` (conversion/openelm.py:64).

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_openelm_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "openelm"

N_EMBD = 24
HEAD_DIM = 6
N_HEAD = [4, 2, 3]
N_HEAD_KV = [2, 1, 3]
N_FF = [40, 24, 32]
N_LAYER = len(N_HEAD)
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-6


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x0E1E1)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-openelm-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    # THE ARRAYS. gguf-py writes a list as a GGUF array, exactly as
    # conversion/openelm.py:57-59 does.
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
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

    # Also the lm_head: openelm.cpp:22 ties them with no fallback.
    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    for il in range(N_LAYER):
        p = f"blk.{il}."
        nh, nkv, nff = N_HEAD[il], N_HEAD_KV[il], N_FF[il]
        n_qkv = (2 * nkv + nh) * HEAD_DIM

        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        # Q rows, then K rows, then V rows; wider Q/K so the six-token
        # attention is not flat.
        q = rnd(nh * HEAD_DIM, N_EMBD) * 4.0
        k = rnd(nkv * HEAD_DIM, N_EMBD) * 4.0
        v = rnd(nkv * HEAD_DIM, N_EMBD)
        wqkv = np.concatenate([q, k, v], axis=0)
        assert wqkv.shape == (n_qkv, N_EMBD)
        w.add_tensor(p + "attn_qkv.weight", wqkv)
        # Per-head norms, centred away from 1 so a norm applied after
        # RoPE (or not at all) is visible.
        w.add_tensor(p + "attn_q_norm.weight", (1.5 + rnd(HEAD_DIM)).astype(np.float32))
        w.add_tensor(p + "attn_k_norm.weight", (1.5 + rnd(HEAD_DIM)).astype(np.float32))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, nh * HEAD_DIM))

        w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "ffn_gate.weight", rnd(nff, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(nff, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, nff))

    w.add_tensor("output_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
    # NO `output.weight`.

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "openelm-fixture.gguf")
