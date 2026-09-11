#!/usr/bin/env python3
"""Generate the tiny synthetic `apertus` GGUF used by ferrox's per-layer
activation-parameter coverage test.

`apertus` (Swiss AI Apertus-8B / 70B) was triaged NEW CODE on one fact:
its FFN activation is xIELU with FOUR PER-LAYER PARAMETER ARRAYS.
`.scratch/llama.cpp/src/models/apertus.cpp:6-9` reads

    ml.get_key_or_arr(LLM_KV_XIELU_ALPHA_N, hparams.xielu_alpha_n, n_layer);
    ml.get_key_or_arr(LLM_KV_XIELU_ALPHA_P, hparams.xielu_alpha_p, n_layer);
    ml.get_key_or_arr(LLM_KV_XIELU_BETA,    hparams.xielu_beta,    n_layer);
    ml.get_key_or_arr(LLM_KV_XIELU_EPS,     hparams.xielu_eps,     n_layer);

(all four REQUIRED, an array at exactly `n_layer` length or a scalar
broadcast to every layer) and :132-138 applies layer `il`'s four to
that layer's `ffn_up` output:

    activated = ggml_xielu(ctx0, up, alpha_n[il], alpha_p[il], beta[il], eps[il]);

`ggml_xielu` (ggml.c:2837-2856) folds the softplus at graph build --
`alpha_n' = beta + softplus(alpha_n)`, `alpha_p' = softplus(alpha_p)`
-- and the CPU op (ggml-cpu/unary-ops.cpp:55-62) is then

    x > 0:  alpha_p' * x * x + beta * x
    x <= 0: (expm1(min(x, eps)) - x) * alpha_n' + beta * x

The keys carry NO architecture prefix: `xielu.alpha_n`, not
`apertus.xielu.alpha_n` (llama-arch.cpp:370-373, gguf-py
constants.py:403-406). The converter (conversion/llama.py:424-457)
collects each layer's `act_fn.{alpha_n,alpha_p,beta,eps}` scalar
TENSOR from the checkpoint and writes the four arrays, so a real
export always carries arrays, one entry per layer.

The FFN is UNGATED (`:45-46` creates `ffn_down` and `ffn_up` only) and
the rest of the graph is: pre-norm RMSNorm at both sites, split Q/K/V,
per-head RMS QK-norm BEFORE RoPE (`:93-96`), NEOX RoPE
(llama-model.cpp:2671), `n_embd_head == n_rot` asserted (`:64`),
`1/sqrt(head_dim)` unless `f_attention_scale` (`:74-75`), an OPTIONAL
`attn_output.bias` (`:42`), and a REQUIRED `output.weight` (`:24`).

What this fixture pins:

  * **Four arrays that DIFFER between the two layers**, in every
    parameter, so that indexing them by the wrong layer -- or reading
    one scalar for all -- moves the logits. Layer 1's `eps` is a large
    negative number rather than the real checkpoints' `-1e-6`, because
    `min(x, eps)` at `-1e-6` is invisible below the comparison
    tolerance and an `eps` the test cannot see is an `eps` the test
    cannot pin.
  * **`ffn_up` pre-activations that cross zero**, drawn at unit
    magnitude with no offset, so both branches of xIELU run on about
    half the channels each.
  * **No `ffn_gate` tensor on any layer.**
  * **NEOX RoPE, per-head QK-norm before RoPE, GQA (4 heads over 2),
    an explicit `output.weight`.**

`--scalar` writes the same weights with each of the four keys as ONE
scalar instead of an array. `get_key_or_arr` broadcasts it to every
layer (llama-model-loader.cpp:469-478), so libllama must run this file
with layer 0's parameters on both layers; the test pins that ferrox
does the same and that the two files' logits differ.

`--qk-norm-bias` adds `attn_q_norm.bias` and `attn_k_norm.bias` on
every layer. apertus.cpp:50,52 CREATE both (`TENSOR_NOT_REQUIRED`) and
:93,96 pass `NULL` as the bias to `build_norm`, so they are loaded and
never read; the test measures that libllama's logits do not move and
pins that ferrox leaves them deliberately unread rather than applying
or refusing them.

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_apertus_fixture.py OUT.gguf [--scalar | --qk-norm-bias]

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "apertus"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = N_EMBD // N_HEAD  # 6; apertus.cpp:64 asserts n_embd_head == n_rot
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5

# Per layer: (alpha_n, alpha_p, beta, eps), BEFORE the softplus ggml
# applies. Layer 0 is the real checkpoints' initialisation
# (alpha_p_init = alpha_n_init = 0.8, beta = 0.5, eps = -1e-6); layer 1
# differs in all four so a wrong index is visible.
XIELU = [
    (0.8, 0.8, 0.5, -1e-6),
    (0.2, 1.5, 0.75, -0.3),
]


def main(out_path: str, variant: str) -> None:
    rng = np.random.default_rng(0xA9E27)
    # A SEPARATE stream for the optional bias tensors, so every weight
    # the base file has is byte-identical in the `--qk-norm-bias` one
    # and the only difference between the two files is the biases.
    bias_rng = np.random.default_rng(0xB1A5)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name(f"ferrox-apertus-fixture{variant.replace('--', '-') if variant else ''}")
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

    alpha_n = [p[0] for p in XIELU]
    alpha_p = [p[1] for p in XIELU]
    beta = [p[2] for p in XIELU]
    eps = [p[3] for p in XIELU]
    if variant == "--scalar":
        # One scalar per key; llama.cpp broadcasts it to every layer.
        w.add_float32(gguf.Keys.xIELU.ALPHA_N, alpha_n[0])
        w.add_float32(gguf.Keys.xIELU.ALPHA_P, alpha_p[0])
        w.add_float32(gguf.Keys.xIELU.BETA, beta[0])
        w.add_float32(gguf.Keys.xIELU.EPS, eps[0])
    else:
        w.add_xielu_alpha_n(alpha_n)
        w.add_xielu_alpha_p(alpha_p)
        w.add_xielu_beta(beta)
        w.add_xielu_eps(eps)

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
        # Wider projections so the six-token attention is not flat.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        # QK-norm weights centred away from 1 so the norm is visible.
        w.add_tensor(p + "attn_q_norm.weight", (1.5 + rnd(HEAD_DIM)).astype(np.float32))
        w.add_tensor(p + "attn_k_norm.weight", (1.5 + rnd(HEAD_DIM)).astype(np.float32))
        if variant == "--qk-norm-bias":
            # Created at apertus.cpp:50,52 and never passed to build_norm.
            w.add_tensor(p + "attn_q_norm.bias", bias_rng.standard_normal(HEAD_DIM).astype(np.float32))
            w.add_tensor(p + "attn_k_norm.bias", bias_rng.standard_normal(HEAD_DIM).astype(np.float32))
        w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        # Unit magnitude and zero mean: about half the pre-activations
        # are negative, so both xIELU branches run.
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
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    main(args[0] if args else "apertus-fixture.gguf", flags[0] if flags else "")
