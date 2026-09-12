#!/usr/bin/env python3
"""Generate the tiny synthetic `plm` GGUFs used by ferrox's MLA-engine
coverage test.

`plm` refused as UNAUDITED, triaged NEW CODE: DeepSeek-2 MLA attention
on a dense model (`src/models/plm.cpp`). What the graph is:

  * `:29-34` create a DIRECT `attn_q` {n_embd, n_head * n_embd_head_k}
    (no `q_lora_rank`, no `attn_q_a` / `attn_q_b`), `attn_kv_a_mqa`
    {n_embd, kv_lora_rank + n_rot}, `attn_kv_a_norm` {kv_lora_rank},
    `attn_kv_b` {kv_lora_rank, n_head * (qk_nope + v)} and `attn_output`
    {n_head * v, n_embd}.
  * `:84-166` split Q per head into `q_nope` / `q_pe` (nope FIRST),
    RMS-norm the compressed KV, re-expand it through `attn_kv_b` into
    per-head `k_nope` and `v`, rope `q_pe` and the ONE shared `k_pe`
    (NORM layout, llama-model.cpp:2592) and repeat `k_pe` onto every
    head. `kq_scale = 1/sqrt(n_embd_head_k)` (`:50`).
  * `:181-187`: an UNGATED `LLM_FFN_RELU_SQR` FFN over `ffn_up` /
    `ffn_down` alone (the `arcee` shape).
  * `:23-24`: `output` is `tok_embd` DUPLICATED -- the graph never reads
    an `output.weight`, and llama.cpp's loader refuses a file that
    carries one (`llama-model-loader.cpp:1309-1313`, "wrong number of
    tensors").

Keys, as `conversion/plm.py:14-19` writes them: `attention.key_length`
= qk_nope + qk_rope (NOT `key_length_mla`), `attention.value_length` =
v_head_dim, `rope.dimension_count` = qk_rope, `attention.kv_lora_rank`.

`--decoy-output` adds an `output.weight` the graph does not read, so
the test can pin that ferrox refuses the file as libllama does rather
than silently preferring the decoy.

Weights are pseudo-random from a fixed seed so the file is byte-stable.
`attn_kv_a_norm` is drawn away from one so that a dropped norm is
visible; `attn_q` is drawn wide so that the head split (nope first,
rope last) matters to the logits.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_plm_fixture.py OUT.gguf [--decoy-output]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "plm"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
QK_NOPE = 8
QK_ROPE = 4
V_HEAD = 8
KV_LORA = 12
N_FF = 48
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-6


def main(out_path: str, decoy_output: bool) -> None:
    rng = np.random.default_rng(0x9137)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def away_from_one(*shape: int) -> np.ndarray:
        return (1.5 + rng.standard_normal(shape) * 0.5).astype(np.float32)

    qk_head = QK_NOPE + QK_ROPE

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-plm-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    # PLM's K/V are expanded per query head; the converter writes
    # num_key_value_heads, which equals num_attention_heads.
    w.add_head_count_kv(N_HEAD)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    # conversion/plm.py:14-19.
    w.add_vocab_size(N_VOCAB)
    w.add_kv_lora_rank(KV_LORA)
    w.add_key_length(qk_head)
    w.add_value_length(V_HEAD)
    w.add_rope_dimension_count(QK_ROPE)
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

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))
        # :32 {n_embd, n_embd_head_k * n_head}: per head, nope then rope.
        w.add_tensor(p + "attn_q.weight", rnd(N_HEAD * qk_head, N_EMBD) * 4.0)
        # :33 {n_embd, kv_lora_rank + n_rot}: compressed KV, then k_pe.
        w.add_tensor(p + "attn_kv_a_mqa.weight", rnd(KV_LORA + QK_ROPE, N_EMBD) * 4.0)
        # :34 {kv_lora_rank}, away from one.
        w.add_tensor(p + "attn_kv_a_norm.weight", away_from_one(KV_LORA))
        # :35 {kv_lora_rank, n_head * (qk_nope + v)}: per head, k_nope then v.
        w.add_tensor(p + "attn_kv_b.weight", rnd(N_HEAD * (QK_NOPE + V_HEAD), KV_LORA) * 2.0)
        # :36 {n_head * v, n_embd}.
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * V_HEAD) * 4.0)
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        # :39-40: up and down, no gate.
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 4.0)

    w.add_tensor("output_norm.weight", away_from_one(N_EMBD))
    if decoy_output:
        # :23-24 never read it; llama-model-loader.cpp:1309-1313 refuses
        # the file for it.
        w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="plm-fixture.gguf")
    ap.add_argument("--decoy-output", action="store_true")
    args = ap.parse_args()
    main(args.out, args.decoy_output)
