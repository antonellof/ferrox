#!/usr/bin/env python3
"""Generate the tiny synthetic `glm4` GGUFs used by ferrox's GLM-4-0414
coverage test.

`glm4` is what GLM-4-0414 (9B, 32B), GLM-Z1 and GLM-OCR tag. ferrox
dispatched it to the GLM-5.2 MLA loader, which asks for
`attention.q_lora_rank` and three more MLA keys `src/models/glm4.cpp`
never reads (`:3-9` read `layer_norm_rms_epsilon`, the optional
`rope.dimension_sections` and `nextn_predict_layers`, nothing else), so
a real GLM-4-9B-0414 download failed with "missing hparam
glm4.attention.q_lora_rank" -- the `glm4moe` defect a second time. What
the graph is (`glm4.cpp:97-176`):

  * plain GQA with Q/K/V biases (`create_tensor_qkv`, `:42`), NORM RoPE
    over the first half of each head (`partial_rotary_factor = 0.5`,
    `conversion/glm.py:21,50`; llama-model.cpp:2699).
  * Gemma-2's two post norms in Gemma-2's slots: `post_attention_norm`
    on the attention branch BEFORE its residual add (`:144-148`) and
    `post_ffw_norm` on the FFN branch before its (`:166-169`), with the
    ordinary `attn_norm` and `ffn_norm` still present (`:41,48`).
  * a FUSED SwiGLU: one `ffn_up` of `{n_embd, 2 * n_ff}` and no
    `ffn_gate` (`:50`), through `LLM_FFN_SWIGLU` = `ggml_swiglu`, which
    is `silu(first half) * second half` (`:158-163`) -- Phi-3's fused
    form, which the generic loader already splits.
  * NextN blocks inside `block_count` for GLM-OCR (`:8,54-64`), skipped
    as `crate::mtp_blocks` skips them.
  * a tied lm_head when `output` is absent (`:26-29`).

`--mrope` writes `rope.dimension_sections`, what a GLM-4.1V text tower
carries (`conversion/glm.py:26-27`); llama.cpp then rotates with
`LLAMA_ROPE_TYPE_MROPE` (llama-model.cpp:2699) over weights the
converter PERMUTED to NEOX order (`glm.py:53-73,78-85`), so the file's
correct rotation is NEOX and the base file's is NORM. ferrox's rope
layout is decided per architecture, not per file, so that file is
REFUSED by name; libllama runs it (measured), which is what makes the
refusal a refusal and not an excuse.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_glm4_fixture.py OUT.gguf [--mrope]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "glm4"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 16
ROPE_DIM = HEAD_DIM // 2
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1.5625e-07


def main(out_path: str, mrope: bool) -> None:
    rng = np.random.default_rng(0x6714)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-glm4-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    # glm.py:50: head_dim * partial_rotary_factor.
    w.add_rope_dimension_count(ROPE_DIM)
    if mrope:
        w.add_rope_dimension_sections([2, 1, 1, 0])
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
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_q.bias", rnd(n_embd_q))
        w.add_tensor(p + "attn_k.bias", rnd(n_embd_kv))
        w.add_tensor(p + "attn_v.bias", rnd(n_embd_kv))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        # :46 and :52: Gemma-2's two post norms, drawn away from one so
        # a norm applied on the wrong side of a residual is visible.
        w.add_tensor(p + "post_attention_norm.weight", (1.5 + rng.standard_normal(N_EMBD) * 0.5).astype(np.float32))
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        # :50 {n_embd, n_ff * 2}: gate half first, up half second.
        w.add_tensor(p + "ffn_up.weight", rnd(2 * N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 4.0)
        w.add_tensor(p + "post_ffw_norm.weight", (1.5 + rng.standard_normal(N_EMBD) * 0.5).astype(np.float32))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="glm4-fixture.gguf")
    ap.add_argument("--mrope", action="store_true")
    args = ap.parse_args()
    main(args.out, args.mrope)
