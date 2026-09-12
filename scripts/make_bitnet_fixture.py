#!/usr/bin/env python3
"""Generate the tiny synthetic `bitnet` GGUFs used by ferrox's
sub-norm coverage test.

`bitnet` refused as UNAUDITED, triaged NEW CODE, on two norms INSIDE
the blocks, in slots the generic decoder did not have
(`src/models/bitnet.cpp`):

  * **`attn_sub_norm`** (:24, `{n_embd}`, REQUIRED) is an RMSNorm on
    the attention output -- the concatenated heads, after the
    softmax-weighted V sum -- applied BEFORE `wo` (:101-106). Not
    Gemma's `post_attention_norm`, which sits AFTER `wo`; the two are
    on opposite sides of a matmul.
  * **`ffn_sub_norm`** (:36, `{n_ff}`, REQUIRED) is an RMSNorm on the
    `silu(gate) * up` product, applied BEFORE `ffn_down` (:135-140),
    inside the FFN. `build_ffn` is called with a NULL down projection
    (:127-132) and the graph applies `ffn_down` itself afterwards.

`grep -l 'attn_sub_norm\\|ffn_sub_norm' src/models/*.cpp` over all
140 graphs is `bitnet.cpp` alone (measured). `crates/ferrox-models/
src/sub_norms.rs` is the seam.

The rest of the graph is plain Llama: pre-norm RMS, NEOX-layout RoPE
(`llama-model.cpp:2625` puts `LLM_ARCH_BITNET` in the
`LLAMA_ROPE_TYPE_NEOX` group), `1/sqrt(head_dim)`, SwiGLU. Two more
things the file shape pins:

  * **No `output.weight`.** `:14-17` creates `tok_embd` and
    `output_norm` and NO `output` tensor; `:164` computes the lm_head
    from `tok_embd` unconditionally. So the fixture has no
    `output.weight`, which is also what `conversion/bitnet.py` produces
    for the tied-embedding checkpoints (BitNet-b1.58-2B-4T).
  * **`rope.scaling.type = linear`, `factor = 1.0`.** `conversion/
    bitnet.py:19-20` writes both for every export; a factor of 1 is a
    no-op that the loader must accept rather than refuse.

`--with-scales` writes the OPTIONAL per-tensor `blk.N.<proj>.scale`
tensors (`{1}`, `TENSOR_NOT_REQUIRED`, `:27-43`) that `build_lora_mm`
multiplies the matmul result by (`llama-graph.cpp:1492-1494`). The
CURRENT converter writes ternary weights already divided by the scale
and emits no such tensor (`conversion/bitnet.py:23-32`); older BitNet
exports carry them, and llama.cpp still honours them. ferrox does NOT
apply them, and the test on this shape pins that it REFUSES the file
(the tensors are unread) rather than running it at the wrong scale.

Weights are pseudo-random from a fixed seed so the file is byte-stable.
The sub-norm weights are drawn AWAY from one (mean 1.5, spread 0.5),
so that a norm that is skipped, or applied with all-ones weights, moves
the logits by far more than the tolerance; the test measures that
rather than assuming it.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_bitnet_fixture.py OUT.gguf [--with-scales]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "bitnet"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str, with_scales: bool) -> None:
    rng = np.random.default_rng(0xB17)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w(n: int) -> np.ndarray:
        # Away from one on purpose: see the module doc.
        return (1.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-bitnet-fixture")
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
    w.add_rope_dimension_count(HEAD_DIM)
    # conversion/bitnet.py:19-20, on every export.
    w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
    w.add_rope_scaling_factor(1.0)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # Minimal SPM-flavoured vocab: llama.cpp needs tokens/scores/types to
    # build a vocab at all, but the fixture is always driven by explicit
    # token ids, never by tokenizing text.
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

    # ne = [n_embd, n_vocab] -> numpy [n_vocab, n_embd]. Also the lm_head
    # (bitnet.cpp:164); there is no output.weight.
    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))
        # bitnet.cpp:24: `{n_embd}`, on the attention output before wo.
        w.add_tensor(p + "attn_sub_norm.weight", norm_w(N_EMBD))

        # Drawn wider than the rest of the file so the softmax over a
        # six-token prompt is not near-uniform.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q) * 4.0)

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        # bitnet.cpp:36: `{n_ff}`, on silu(gate) * up before ffn_down.
        w.add_tensor(p + "ffn_sub_norm.weight", norm_w(N_FF))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        # Wide, so the FFN branch is a real share of the residual and a
        # sabotage of its inner norm has to clear the tolerance by
        # orders of magnitude.
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 4.0)

        if with_scales:
            # bitnet.cpp:27-43, `{1}` each, optional; build_lora_mm
            # multiplies the projection's output by it. Far from one so
            # that a file this shape run WITHOUT them is visibly wrong.
            for proj in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
                w.add_tensor(p + proj + ".scale", np.array([2.0], dtype=np.float32))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="bitnet-fixture.gguf")
    ap.add_argument("--with-scales", action="store_true")
    args = ap.parse_args()
    main(args.out, args.with_scales)
