#!/usr/bin/env python3
"""Generate the tiny synthetic `chatglm` GGUF used by ferrox's ChatGLM
coverage test.

THE ARM this fixture exists for is the **fused `attn_qkv.bias`**.
`src/models/chatglm.cpp:42` calls `create_tensor_qkv`, which prefers a
fused `attn_qkv.weight` and, when it finds one, creates `attn_qkv.bias`
beside it (llama-model.cpp:2890-2892); `build_qkv` then adds that bias
to the fused projection BEFORE splitting it into Q, K and V
(llama-graph.cpp:1605-1609). ferrox's `load_qkv_projections` split the
fused WEIGHT and read bias only under the split `attn_q.bias` /
`attn_k.bias` / `attn_v.bias` names, so on a real ChatGLM2/3 checkpoint
-- which sets `add_qkv_bias: true` and stores
`encoder.layers.{bid}.self_attention.query_key_value` as
`blk.N.attn_qkv` (gguf-py `tensor_mapping.py`) -- all three projections
ran unbiased.

The bias here is drawn at a magnitude comparable to the projections'
own output, so dropping it is a divergence in the first decimal place
rather than a rounding difference.

The rest of the graph, read off `src/models/chatglm.cpp`:

  * `load_arch_hparams` (:3-22) reads only the RMS epsilon. Nothing
    else is architecture-specific, which is what made this row look
    fixture-away before the converter was read.
  * `load_arch_tensors` (:38-51): `attn_norm`, the fused QKV, `wo`
    sized `{n_embd, n_embd}`, `ffn_norm`, a FUSED `ffn_up` of
    `{n_embd, n_ff * 2}` and `ffn_down`. No `ffn_gate`, no post-norms,
    no QK-norm, no biases other than the QKV one.
  * The graph (:75-145) is a plain sequential residual with
    `1/sqrt(n_embd_head)` (:108) and `LLM_FFN_SWIGLU, LLM_FFN_SEQ`
    (:133) -- the fused gate+up SwiGLU ferrox already implements for
    phi3, gate first half, up second half (`ggml_swiglu`, non-swapped:
    `ggml/src/ggml-cpu/ops.cpp:3225-3229`).
  * RoPE is **NORM** (`LLM_ARCH_CHATGLM` sits in
    `llama_model_rope_type`'s NORM group, llama-model.cpp:2593).

PARTIAL ROPE, and why it is not incidental. `chatglm.cpp:59-61` asserts
only that the K and V head widths agree -- NOT that `n_embd_head ==
n_rot` -- and `conversion/chatglm.py:151` writes
`rope_dimension_count` as `head_dim * partial_rotary_factor` with the
factor defaulting to **0.5**. So every real ChatGLM file rotates half a
head and passes the other half through untouched. This fixture sets
`head_dim = 8` and `rope_dimension_count = 4` for exactly that reason:
a fixture that rotated a whole head would let `rope_dim` be ignored
without the comparison noticing.

`{arch}.rope.freq_base` is deliberately SMALL (500 rather than the
converter's 10000 * rope_ratio) so that six positions span radians
rather than milliradians on both surviving bands. At 10000 the second
band turns by 0.05 rad over the whole prompt and a positional sabotage
would be invisible.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_chatglm_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "chatglm"

N_LAYER = 2
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
# chatglm.cpp:44 sizes `wo` as {n_embd, n_embd}, so n_head * head_dim
# must be n_embd; the file declares no key_length and llama.cpp derives
# head_dim = n_embd / n_head = 8.
HEAD_DIM = N_EMBD // N_HEAD
# THE PARTIAL ROTARY WIDTH. See the module docstring: half a head, which
# is what conversion/chatglm.py:151 writes.
ROPE_DIM = HEAD_DIM // 2
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 500.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x0C61)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-chatglm-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(ROPE_DIM)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # Minimal SPM-flavoured vocab: llama.cpp needs tokens/scores/types to
    # build a vocab at all, but the fixture is always driven by explicit
    # token ids, never by tokenizing text. Real ChatGLM files carry a
    # BPE vocab (conversion/chatglm.py:121-122); nothing in either graph
    # reads the tokenizer, so the cheaper one is used here.
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
    n_embd_qkv = n_embd_q + 2 * n_embd_kv

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        # FUSED QKV, in llama.cpp's order: Q rows, then K rows, then V
        # rows (llama-graph.cpp:1615-1622 views them at offsets 0,
        # n_embd_q and n_embd_q + n_embd_kv).
        #
        # Q and K are drawn WIDER than everything else on purpose. At
        # the magnitude the rest of the file uses, the attention scores
        # over a six-token prompt sit within a fraction of each other,
        # softmax comes out nearly uniform, and the layer stops caring
        # where the tokens are -- fatal for a fixture whose sabotages
        # include the rotary width.
        qkv = np.concatenate(
            [
                rnd(n_embd_q, N_EMBD) * 4.0,
                rnd(n_embd_kv, N_EMBD) * 4.0,
                rnd(n_embd_kv, N_EMBD),
            ]
        ).astype(np.float32)
        w.add_tensor(p + "attn_qkv.weight", qkv)

        # THE ARM. Centred at 1.0 rather than 0.0: a zero-mean bias of
        # the same magnitude as the projection noise would move the
        # logits, but a shifted one moves every head's Q and K in the
        # same direction and therefore changes the attention pattern as
        # well as the values, which is what makes "bias dropped" a
        # first-decimal-place divergence.
        w.add_tensor(
            p + "attn_qkv.bias", (rnd(n_embd_qkv) + 1.0).astype(np.float32)
        )

        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        # FUSED gate+up, {n_embd, n_ff * 2} (chatglm.cpp:48). First half
        # is the SwiGLU gate, second half the up projection.
        w.add_tensor(p + "ffn_up.weight", rnd(2 * N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    # `output` is TENSOR_NOT_REQUIRED with a tok_embd fallback
    # (chatglm.cpp:32-36); the fixture ships its own so the lm_head is
    # not tied to the embedding table.
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "chatglm-fixture.gguf")
