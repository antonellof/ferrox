#!/usr/bin/env python3
"""Generate the tiny synthetic `granitehybrid` GGUFs used by frink's
Granite-4.0 coverage test (`crates/frink-models/tests/granite_hybrid_graphs.rs`).

`granitehybrid` is IBM Granite 4.0 (H-Micro 3B dense, H-Tiny 7B-A1B and
H-Small 32B-A9B MoE): `granite.cpp`'s four scalar multipliers and
optional biases, with a MAMBA-2 block where attention would be on the
layers whose `head_count_kv` is 0 (`granite-hybrid.cpp:17-19`,
`conversion/granite.py:238-241` writes the array). The graph is one
residual topology for both kinds (`:128-142`): `attn_norm`, the block,
the residual add (scaled by `residual_scale`), `ffn_norm`, the FFN.

The Mamba-2 block (`mamba-base.cpp:149-288`, `crate::mamba2`) reads
`ssm.conv_kernel`, `ssm.inner_size` (asserted `2 * n_embd`, `:44`),
`ssm.state_size`, `ssm.time_step_rank` (the head count) and
`ssm.group_count`, and the tensors `:60-69` create: `ssm_in`
`{n_embd, 2 d_inner + 2 n_group d_state + n_head}`, `ssm_conv1d`
`{d_conv, d_inner + 2 n_group d_state}` with a bias that is optional
to the loader and not to the graph (see `--no-conv-bias`),
`ssm_dt.bias` `{n_head}`, `ssm_a` and `ssm_d` `{1, n_head}` (no
"weight" suffix), `ssm_norm` `{d_inner / n_group, n_group}`, `ssm_out`
`{d_inner, n_embd}`.

RoPE: `granite-hybrid.cpp:14-17` reads `rope.scaling.finetuned` as the
switch it is for `granite`, and `conversion/granite.py:253-256` writes
it FALSE for every export with a Mamba layer, so a real Granite-4.0-H
file rotates NOTHING (NoPE attention). The default fixture is that
shape; `--rope` writes the key true (Bamba's shape) and rotates NORM.

Layers (four): mamba, attention, mamba, mamba -- a conv-first layout
as every real export has, one attention layer with `head_count 4,
head_count_kv 2`.

Variants:
  * default        dense SwiGLU FFN, `rope.scaling.finetuned = false`
  * `--rope`       the key true: the attention layer rotates (NORM)
  * `--moe`        Granite-4.0-H-Tiny's FFN: 4 experts, 2 used, softmax,
                   plus the shared expert (`:87-97`, `n_ff_shexp`)
  * `--no-conv-bias` omits `ssm_conv1d.bias`. `granite-hybrid.cpp:63`
                   creates it `TENSOR_NOT_REQUIRED` and
                   `mamba-base.cpp:222` then `ggml_add`s the NULL:
                   libllama SEGFAULTS on this file (measured), so frink
                   REQUIRES the tensor and the variant is not a fixture

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_granite_hybrid_fixture.py OUT.gguf [--rope] [--moe] [--no-conv-bias]

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "granitehybrid"

N_EMBD = 24
N_HEAD = 4
N_KV = 2
HEAD_DIM = N_EMBD // N_HEAD
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5

# Mamba-2 (`granite-hybrid.cpp:44` asserts d_inner == 2 n_embd).
D_CONV = 4
D_INNER = 2 * N_EMBD
D_STATE = 8
SSM_HEADS = 4
N_GROUP = 2
CONV_W = D_INNER + 2 * N_GROUP * D_STATE
D_IN_PROJ = 2 * D_INNER + 2 * N_GROUP * D_STATE + SSM_HEADS

# MoE (`--moe`).
N_EXPERT = 4
N_EXPERT_USED = 2
N_FF_EXP = 16
N_FF_SHEXP = 12

# The four multipliers, at magnitudes where every feature stays visible
# (see scripts/make_granite_fixture.py for why not the published ones).
LOGIT_SCALE = 2.5
RESIDUAL_SCALE = 0.6
EMBEDDING_SCALE = 2.0
ATTENTION_SCALE = 0.9

# head_count_kv per layer; 0 is a Mamba-2 layer (granite-hybrid.cpp:18).
KV_PER_LAYER = [0, N_KV, 0, 0]


def main(out_path: str, rope: bool, moe: bool, conv_bias: bool) -> None:
    n_layer = len(KV_PER_LAYER)
    rng = np.random.default_rng(0x6247E4)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-granitehybrid-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF_EXP if moe else N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(KV_PER_LAYER)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_logit_scale(LOGIT_SCALE)
    w.add_residual_scale(RESIDUAL_SCALE)
    w.add_embedding_scale(EMBEDDING_SCALE)
    w.add_attention_scale(ATTENTION_SCALE)
    # conversion/granite.py:253-256: false for every export with a
    # Mamba layer.
    w.add_rope_scaling_finetuned(rope)
    w.add_ssm_conv_kernel(D_CONV)
    w.add_ssm_inner_size(D_INNER)
    w.add_ssm_state_size(D_STATE)
    w.add_ssm_time_step_rank(SSM_HEADS)
    w.add_ssm_group_count(N_GROUP)
    if moe:
        w.add_expert_count(N_EXPERT)
        w.add_expert_used_count(N_EXPERT_USED)
        w.add_expert_shared_feed_forward_length(N_FF_SHEXP)
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

    for il, nkv in enumerate(KV_PER_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if nkv == 0:
            # granite-hybrid.cpp:60-69. Shapes are ggml order reversed.
            w.add_tensor(p + "ssm_in.weight", rnd(D_IN_PROJ, N_EMBD) * 2.0)
            w.add_tensor(p + "ssm_conv1d.weight", rnd(CONV_W, D_CONV) * 2.0)
            if conv_bias:
                w.add_tensor(p + "ssm_conv1d.bias", rnd(CONV_W))
            w.add_tensor(p + "ssm_dt.bias", rnd(SSM_HEADS) * 2.0)
            # Stored negative, as the converter writes -exp(A_log).
            w.add_tensor(
                p + "ssm_a",
                (-(0.5 + 2.0 * rng.random(SSM_HEADS))).astype(np.float32).reshape(SSM_HEADS, 1),
            )
            w.add_tensor(p + "ssm_d", (1.0 + rnd(SSM_HEADS)).astype(np.float32).reshape(SSM_HEADS, 1))
            w.add_tensor(
                p + "ssm_norm.weight",
                (1.0 + rnd(N_GROUP, D_INNER // N_GROUP)).astype(np.float32),
            )
            w.add_tensor(p + "ssm_out.weight", rnd(N_EMBD, D_INNER))
        else:
            w.add_tensor(p + "attn_q.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD) * 4.0)
            w.add_tensor(p + "attn_k.weight", rnd(nkv * HEAD_DIM, N_EMBD) * 4.0)
            w.add_tensor(p + "attn_v.weight", rnd(nkv * HEAD_DIM, N_EMBD))
            w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
        w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if moe:
            w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
            w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
            w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
            w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))
            w.add_tensor(p + "ffn_gate_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
            w.add_tensor(p + "ffn_up_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
            w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, N_FF_SHEXP))
        else:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} (kv per layer {KV_PER_LAYER}, rope={rope}, moe={moe}, conv_bias={conv_bias})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "granitehybrid-fixture.gguf",
        "--rope" in sys.argv[1:],
        "--moe" in sys.argv[1:],
        "--no-conv-bias" not in sys.argv[1:],
    )
