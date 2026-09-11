#!/usr/bin/env python3
"""Generate the tiny synthetic `arcee` GGUF used by ferrox's ungated
ReLU-squared coverage test.

`arcee` (Arcee AFM-4.5B) was triaged NEW CODE on one fact shared with
`plm`, `nemotron`, `jais2` and `nemotron-h`: its FFN has NO gate.
`.scratch/llama.cpp/src/models/arcee.cpp:39-40` creates only `ffn_up`
and `ffn_down`, and :123-128 is

    cur = build_ffn(cur,
            model.layers[il].ffn_up,   NULL, NULL,
            NULL,                      NULL, NULL,
            model.layers[il].ffn_down, NULL, NULL,
            NULL,
            LLM_FFN_RELU_SQR, LLM_FFN_SEQ, il);

i.e. `down(relu(up(x))^2)`, two matrices in sequence. Everything else in
the file is `llama` (:6 says so): pre-norm RMSNorm at both sites,
split Q/K/V, NORM RoPE (`LLM_ARCH_ARCEE` is in `llama_model_rope_type`'s
NORM group), `1/sqrt(n_embd_head)` unless `f_attention_scale` is set
(:64), an OPTIONAL `output.weight` that falls back to `token_embd`
(:20-25), and `n_embd_head == n_rot` asserted at :51-52.

What this fixture pins:

  * **No `ffn_gate` tensor on any layer.** ferrox's `FfnActivation::
    ReluSqr` aliases the expert's gate to its up matrix so that
    `relu(gate) * up` is `relu(up)^2`; a decoder that demanded a gate
    could not load this file, and one that ran SwiGLU on the aliased
    pair would diverge (the suite sabotages exactly that).
  * **`ffn_up` pre-activations that cross zero**, drawn at unit
    magnitude with no offset, so `relu` clips roughly half of them and
    the square is visibly not the identity.
  * **NORM RoPE**, GQA (4 heads over 2), an explicit `output.weight`.

`--gated` writes the same file WITH an `ffn_gate` on every layer. No
converter produces that for this architecture (conversion/arcee.py maps
no gate projection), so it is a hand-written file, and it exists to
prove the refusal in `loader::load_dense_expert` can fire: an ungated
architecture's file carrying a gate is a graph llama.cpp does not
compute either -- `check_tensor_dims` never asks for the tensor and it
is left unread.

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_arcee_fixture.py OUT.gguf [--gated]

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "arcee"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = N_EMBD // N_HEAD  # 6; arcee.cpp:51-52 asserts n_embd_head == n_rot
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str, gated: bool) -> None:
    rng = np.random.default_rng(0xA5CEE)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-arcee-fixture")
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
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        # Wider projections so the six-token attention is not flat and
        # the RoPE-variant sabotage has something to move.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if gated:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        # Unit magnitude and zero mean: about half the pre-activations
        # are negative, which is what makes relu visible.
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD) * 4.0)
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "arcee-fixture.gguf", gated="--gated" in sys.argv[1:])
