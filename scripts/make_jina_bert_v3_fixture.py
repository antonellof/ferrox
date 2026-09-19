#!/usr/bin/env python3
"""Generate the tiny synthetic `jina-bert-v3` GGUF.

`nomic-bert` (nomic-embed-text-v1 / v1.5) shares llama.cpp's `bert.cpp`
graph and differs from `bert` in exactly two places:

  * **RoPE on Q and K** (`src/models/bert.cpp:126-133`), where `bert`
    adds a learned position table to the embeddings instead (`:90`,
    gated on `arch == LLM_ARCH_BERT`). A nomic file still CARRIES a
    `position_embd` tensor -- `:32` creates it REQUIRED for every
    architecture on this graph -- and the graph never reads it.
  * **A gated SiLU FFN** (`:195-201`, the final `else`): `ffn_up` and
    `ffn_gate` with no biases, where `bert`'s is an ungated GELU with
    both biases.

Everything else is `bert`'s: the token-type row, the embedding
LayerNorm, bidirectional attention with no mask, the post-attention and
post-FFN LayerNorms with biases, and CLS pooling.

The fixture carries NO `position_embd`: measured, libllama does not
ask a `nomic-bert` file for one (`create_tensor` never names it in the
load log) and refuses a file that has one as carrying an unread
tensor. So "does the loader add a position table" is answered by the
file's shape here and by `crate::bert_encoder`'s own refusal
elsewhere.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \
        python3 scripts/make_nomic_bert_fixture.py OUT.gguf

The golden values that go with it come from llama.cpp itself
(`tools/llama_logits.c --embed`), not from this script.
"""

import sys

import numpy as np

import gguf

ARCH = "jina-bert-v3"

N_EMBD = 32
N_HEAD = 4
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 72
CTX = 64
LAYER_NORM_EPS = 1e-12
ROPE_BASE = 1000.0
N_LAYER = 3


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x7E01C)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-jina-bert-v3-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD)
    w.add_layer_norm_eps(LAYER_NORM_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_token_type_count(2)
    w.add_pooling_type(gguf.PoolingType.MEAN)
    w.add_causal_attention(False)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # A WordPiece vocabulary, as every BERT-family export carries.
    # Real words, because the parity harness feeds TEXT: a WordPiece
    # vocabulary of `tok7` pieces cannot tokenize anything and
    # libllama aborts inside its own tokenizer rather than refusing.
    words = ["hello", "world", "the", "quick", "brown", "fox", "jumps", "over"]
    pieces = [f"##{c}" for c in "abcdefghijklmnopqrstuvwxyz"]
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words + pieces
    tokens += [chr(ord("a") + i) for i in range(26)]
    tokens += [f"tok{i}" for i in range(len(tokens), N_VOCAB)]
    tokens = tokens[:N_VOCAB]
    w.add_tokenizer_model("bert")
    w.add_token_list(tokens)
    w.add_token_types(
        [int(gguf.TokenType.CONTROL)] * 5
        + [int(gguf.TokenType.NORMAL)] * (N_VOCAB - 5)
    )
    w.add_bos_token_id(2)
    w.add_eos_token_id(3)
    w.add_unk_token_id(1)
    w.add_pad_token_id(0)
    w.add_mask_token_id(4)
    # `llama-vocab.cpp:3542` asserts a SEP for a WordPiece vocabulary.
    w.add_sep_token_id(3)
    w.add_add_bos_token(True)
    w.add_add_eos_token(True)

    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))
    w.add_tensor("token_types.weight", rnd(2, N_EMBD))
    w.add_tensor("token_embd_norm.weight", rnd(N_EMBD) + 1.0)
    w.add_tensor("token_embd_norm.bias", rnd(N_EMBD))

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_q.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_q.bias", rnd(N_HEAD * HEAD_DIM))
        w.add_tensor(p + "attn_k.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_k.bias", rnd(N_HEAD * HEAD_DIM))
        w.add_tensor(p + "attn_v.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_v.bias", rnd(N_HEAD * HEAD_DIM))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
        w.add_tensor(p + "attn_output_norm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "attn_output_norm.bias", rnd(N_EMBD))
        # bert's ungated GELU FFN, with both biases.
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.bias", rnd(N_FF))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
        w.add_tensor(p + "ffn_down.bias", rnd(N_EMBD))
        w.add_tensor(p + "layer_output_norm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "layer_output_norm.bias", rnd(N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nomic-bert-fixture.gguf")
