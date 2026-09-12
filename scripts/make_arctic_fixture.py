#!/usr/bin/env python3
"""Generate the tiny synthetic `arctic` GGUFs used by ferrox's
parallel-dense-FFN / MoE-branch-input coverage test.

`arctic` refused as UNAUDITED, triaged NEW CODE, on a PARALLEL dense +
MoE layer whose MoE branch reads the layer INPUT
(`src/models/arctic.cpp`):

  * `:38-42` create `ffn_norm` and a dense `ffn_gate` / `ffn_up` /
    `ffn_down` sized `{n_embd, n_embd}` -- NOT `n_ff` -- on EVERY layer,
    beside `:44-48`'s router, a SECOND per-layer norm `ffn_norm_exps`
    and the routed experts.
  * `:118-132`: `ffn_out = build_ffn(ffn_norm(ffn_inp), ...SILU, PAR) +
    ffn_inp`, the ordinary dense FFN on the post-attention residual.
  * `:135-152`: `build_moe_ffn(ffn_norm_exps(inpSA), ...)` -- the router
    AND the experts read a norm of `inpSA`, the residual stream as it
    ENTERS the layer, before attention. `norm_w = true` (a literal),
    softmax, `hparams.expert_weights_scale` which `load_arch_hparams`
    (`:3-14`) never reads, so it is 0 and `build_moe_ffn` skips the
    scale (`llama-graph.cpp:2070`).
  * `:154` sums the two: `cur = moe_out + ffn_out`.

NORM RoPE (llama-model.cpp:2588), `rope.dimension_count = head_dim`
(`conversion/arctic.py:110`, asserted at `:53`), untied `output`
optional (`:22-27`).

`ferrox_models::parallel_dense_ffn` (the dense FFN summed with the
experts; `grok`'s Grok-2 shape is the other row) and
`RouterInput::NormedLayerInput` (the branch operand; one graph of 140
creates `FFN_NORM_EXPS`) are the seams.

`--weights-scale S` writes `{arch}.expert_weights_scale = S`. llama.cpp
does not read it for `arctic` (measured: the golden is byte-identical
to the base file's), so the variant pins that ferrox ignores it too
(`EXPERT_WEIGHTS_SCALE_READERS`).

Weights are pseudo-random from a fixed seed so the file is byte-stable.
`ffn_norm_exps` is drawn AWAY from `ffn_norm` and from one, and the
attention is drawn wide, so a MoE branch fed the post-attention residual
(or the wrong norm) moves the logits by far more than the tolerance.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_arctic_fixture.py OUT.gguf [--weights-scale S]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "arctic"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_EXPERT = 4
N_EXPERT_USED = 2
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str, weights_scale: float | None) -> None:
    rng = np.random.default_rng(0xA2C7)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def away_from_one(*shape: int) -> np.ndarray:
        return (1.5 + rng.standard_normal(shape) * 0.5).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-arctic-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    # conversion/arctic.py:109-110.
    w.add_vocab_size(N_VOCAB)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    if weights_scale is not None:
        w.add_expert_weights_scale(weights_scale)
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
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q) * 4.0)
        # :38-42: the dense half, `{n_embd, n_embd}`, on every layer.
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_EMBD, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_EMBD, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_EMBD) * 4.0)
        # :44-48: the routed half. The router is drawn wide so top-2 of
        # four is a real decision; `ffn_norm_exps` away from one and
        # from `ffn_norm`.
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD) * 4.0)
        w.add_tensor(p + "ffn_norm_exps.weight", away_from_one(N_EMBD))
        # gate/up: ne = [n_embd, n_ff, n_expert]; down: [n_ff, n_embd, n_expert].
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF) * 4.0)

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="arctic-fixture.gguf")
    ap.add_argument("--weights-scale", type=float, default=None)
    args = ap.parse_args()
    main(args.out, args.weights_scale)
