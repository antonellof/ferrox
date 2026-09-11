#!/usr/bin/env python3
"""Generate the tiny synthetic `deci` GGUFs used by ferrox's per-layer
shape coverage test.

`deci` (DeciLM, Llama-3.1-Nemotron-51B/253B) was triaged NEW CODE on
PER-LAYER SHAPES, one step worse than `openelm`: not only do
`.scratch/llama.cpp/src/models/deci.cpp:30-34` (loader) and :103-105
(graph) read `n_head(i)`, `n_head_kv(i)` and `n_ff(i)` per layer, the
graph branches on them three ways:

    if (n_head == 0)                 cur = inpL;              // :107-109, no norm
    if (n_head > 0 && n_head_kv == 0) cur = wo * norm(inpL);  // :115-118, "linear attention"
    else if (n_head > 0)             cur = attn(norm(inpL));  // :119-137
    if (n_ff == 0) continue;                                  // :147-149
    ffn_inp = n_head > 0 ? cur + inpSA : cur;                 // :150-153
    cur = ffn(norm(ffn_inp)) + ffn_inp;                       // :155-172

`conversion/deci.py:92-105` writes the three arrays for the Nemotron
block_configs shape, with `0` for a no-op attention or FFN, and
:114-118 writes ONLY `head_count_kv` as an array for DeciLM-7B, whose
layers differ in their GQA ratio alone.

Three files:

  * default -- the NEMOTRON shape, four layers, one of each kind:
      blk.0  head_count 4, kv 2, ff 40   plain GQA + FFN
      blk.1  head_count 0, kv 0, ff 0    the identity (no tensors at all)
      blk.2  head_count 4, kv 0, ff 24   wo-only attention + FFN
      blk.3  head_count 0, kv 0, ff 32   no attention, FFN only
    Layer 2 has ONLY `attn_norm` and a `{n_embd, n_embd}` `attn_output`
    (:36-40); layer 3 has ONLY the FFN tensors; layer 1 has none.

    The identity layer is deliberately NOT last. Measured: with it as
    the final layer libllama ABORTS in `llm_graph_input_out_ids::
    set_input` (`GGML_ASSERT(buffer)`, ggml-backend.cpp:194), because
    deci.cpp:147-149's `continue` skips past the `inp_out_ids`
    `get_rows` at :141-144 whose result only `ffn_inp` would have
    carried forward, so the input tensor never joins the graph and is
    never allocated. A real export whose last block is a no-op cannot
    be run by llama.cpp at all.
  * `--kv-only` -- the DECILM-7B shape: `head_count 4` and
    `feed_forward_length 40` as scalars, `head_count_kv = [2, 1, 4]`.
    Same tensor set as `llama`, different K/V widths per layer.
  * `--attn-ffnfree` -- the combination ferrox REFUSES: a MIDDLE layer
    with attention (`head_count 4, kv 2`) and `feed_forward_length 0`
    (middle for the reason above).
    deci.cpp:147-149 `continue`s before the residual add at :150-153,
    so the attention output computed at :115-137 is DISCARDED and the
    layer is the identity in llama.cpp's graph, which is almost
    certainly not the model's. libllama loads the file; ferrox names
    the line and stops (`layer_shapes::LayerShapes::resolve`).

Common to all: NORM RoPE (`LLM_ARCH_DECI` is in `llama_model_rope_type`'s
NORM group, llama-model.cpp:2576), `n_rot == n_embd_head_k` enforced at
llama-model.cpp:1204, `attention.key_length` written because a layer
whose `head_count` is 0 cannot derive `head_dim` (conversion/deci.py:
108-109 writes it), `1/sqrt(n_embd_head)` unless `f_attention_scale`
(:96-97), gated SiLU FFN (:161-166), optional `output.weight` with a
tied fallback (:20-25).

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_deci_fixture.py OUT.gguf [--kv-only | --attn-ffnfree]

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "deci"

N_EMBD = 24
HEAD_DIM = 6
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5

SHAPES = {
    # (head_count, head_count_kv, feed_forward_length) per layer
    "nemotron": [(4, 2, 40), (0, 0, 0), (4, 0, 24), (0, 0, 32)],
    "kv-only": [(4, 2, 40), (4, 1, 40), (4, 4, 40)],
    "attn-ffnfree": [(4, 2, 40), (4, 2, 0), (4, 2, 40)],
}


def main(out_path: str, variant: str) -> None:
    shapes = SHAPES[variant]
    n_layer = len(shapes)
    rng = np.random.default_rng(0xDEC1)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name(f"ferrox-deci-{variant}-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    heads = [s[0] for s in shapes]
    kvs = [s[1] for s in shapes]
    ffs = [s[2] for s in shapes]
    if variant == "kv-only":
        # conversion/deci.py:114-118: scalars from the llama base class,
        # then head_count_kv overwritten with the per-layer list.
        w.add_head_count(heads[0])
        w.add_feed_forward_length(ffs[0])
        w.add_head_count_kv(kvs)
    else:
        w.add_head_count_kv(kvs)
        w.add_head_count(heads)
        w.add_feed_forward_length(ffs)
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

    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    for il, (nh, nkv, nff) in enumerate(shapes):
        p = f"blk.{il}."
        if nh > 0 and nkv == 0:
            # deci.cpp:36-40: norm and a square wo, nothing else.
            w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
            w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_EMBD))
        elif nkv > 0:
            w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
            w.add_tensor(p + "attn_q.weight", rnd(nh * HEAD_DIM, N_EMBD) * 4.0)
            w.add_tensor(p + "attn_k.weight", rnd(nkv * HEAD_DIM, N_EMBD) * 4.0)
            w.add_tensor(p + "attn_v.weight", rnd(nkv * HEAD_DIM, N_EMBD))
            w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, nh * HEAD_DIM))
        # nh == 0: no attention tensors at all (:36-45 create none).
        if nff > 0:
            w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
            w.add_tensor(p + "ffn_gate.weight", rnd(nff, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(nff, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, nff))
        # nff == 0: :52-54 and :63-67 create nothing.

    w.add_tensor("output_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} ({variant}: {shapes})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    variant = "nemotron"
    if "--kv-only" in sys.argv[1:]:
        variant = "kv-only"
    if "--attn-ffnfree" in sys.argv[1:]:
        variant = "attn-ffnfree"
    main(args[0] if args else "deci-fixture.gguf", variant)
