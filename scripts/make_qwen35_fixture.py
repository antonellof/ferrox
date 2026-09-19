#!/usr/bin/env python3
"""Generate the tiny synthetic `qwen35` GGUFs used by frink's Qwen3.5
coverage test (`crates/frink-models/tests/qwen35_graphs.rs`).

`qwen35` is Qwen3.5 dense (0.8B / 2B / 4B / 9B / 27B): a gated delta-net
(GDN) block on three layers of four and gated full attention on the
fourth (`qwen35.cpp:17-24`: `attention.recurrent_layers` array, else
`(i + 1) % full_attention_interval != 0` is recurrent), every layer
`attn_norm` -> block -> residual -> `attn_post_norm` -> SwiGLU ->
residual (`:126-152`).

The GDN block (`:236-317`, `delta-net-base.cpp`), per token and V head:
`qkv = silu(conv(attn_qkv(x)))` split `[key_dim, key_dim, value_dim]`,
q and k l2-normed per head (`:296-297`), `beta = sigmoid(ssm_beta(x))`,
`g = softplus(ssm_alpha(x) + ssm_dt) * ssm_a`, then the delta rule
(`delta-net-base.cpp:289-365`) with V head `h` reading K head
`h % n_k_heads` (TILED, `llama-model.cpp:524-526`; the converter
reorders V heads for it, `conversion/qwen.py:_LinearAttentionVReorderBase`),
`rms_norm(o, ssm_norm) * silu(z)` per head, `ssm_out`. Dims:
`head_k_dim = head_v_dim = ssm.state_size`, `n_k_heads = ssm.group_count`,
`n_v_heads = ssm.time_step_rank`, `ssm.inner_size = n_v_heads * head_v_dim`.

The attention layer (`:186-234`): `attn_q` is `2 * n_head * head_dim`
wide, each head's `[q, gate]` interleaved (`:191-199`), per-head RMS QK
norm, `ggml_rope_multi` with `rope.dimension_sections` (IMROPE; on text
positions NEOX band for band) over `rope.dimension_count` bands,
`sigmoid(gate) * attn` before `wo` (`:229-231`).

Layers (four): GDN, GDN, GDN, attention (`full_attention_interval 4`).

Variants:
  * default          `full_attention_interval 4`
  * `--array`        the same layout declared with `attention.recurrent_layers`
  * `--output`       a separate `output.weight`
  * `--moe`          the `qwen35moe` architecture (Qwen3.5-35B-A3B and up):
                     the same layers with `qwen2moe`'s FFN on every one
                     (`qwen35moe.cpp:98-107,496-538`): a softmax MoE with
                     `norm_w = true` and a shared expert whose output is
                     scaled by `sigmoid(ffn_gate_inp_shexp . x)`
  * `--next`         the `qwen3next` architecture (Qwen3-Next-80B-A3B):
                     `--moe`'s layers with three differences: V heads
                     read K heads GROUPED (`h / (n_v / n_k)`,
                     `qwen3next.cpp:521-539`, `llama-model.cpp:525`), beta
                     and alpha come from ONE `ssm_ba` projection laid out
                     `[k_group][beta * ratio, alpha * ratio]`
                     (`:422-436`), and RoPE is plain NEOX with no sections
                     (`:282-291`; `llama-model.cpp:2678`)

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_qwen35_fixture.py OUT.gguf [--array] [--output] [--moe | --next]

Weights are pseudo-random from a fixed seed so the files are byte-stable.
The golden logits that go with them are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "qwen35"
MOE_ARCH = "qwen35moe"
NEXT_ARCH = "qwen3next"

N_EMBD = 32
N_HEAD = 4
N_KV = 2
HEAD_DIM = 8
ROPE_DIM = 4  # partial_rotary_factor 0.5 of head_dim 8
N_FF = 48
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5

# GDN.
D_CONV = 4
HEAD_KV = 8  # ssm.state_size: head_k_dim == head_v_dim
N_K_HEADS = 2  # ssm.group_count
N_V_HEADS = 4  # ssm.time_step_rank
KEY_DIM = HEAD_KV * N_K_HEADS
VALUE_DIM = HEAD_KV * N_V_HEADS
CONV_DIM = 2 * KEY_DIM + VALUE_DIM

N_LAYER = 4
RECURRENT = [True, True, True, False]

# MoE (`--moe`).
N_EXPERT = 4
N_EXPERT_USED = 2
N_FF_EXP = 16
N_FF_SHEXP = 12


def main(out_path: str, as_array: bool, separate_output: bool, moe: bool, nxt: bool) -> None:
    rng = np.random.default_rng(0x9335)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    arch = NEXT_ARCH if nxt else MOE_ARCH if moe else ARCH
    moe = moe or nxt
    w = gguf.GGUFWriter(out_path, arch)
    w.add_name(f"frink-{arch}-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF_EXP if moe else N_FF)
    if moe:
        # conversion/qwen.py (Qwen2MoeModel): the expert keys.
        w.add_expert_count(N_EXPERT)
        w.add_expert_used_count(N_EXPERT_USED)
        w.add_expert_feed_forward_length(N_FF_EXP)
        w.add_expert_shared_feed_forward_length(N_FF_SHEXP)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(ROPE_DIM)
    if not nxt:
        # conversion/qwen.py: the four M-RoPE sections, summing to
        # rope_dim / 2 (text, height, width, time). Qwen3-Next has none.
        w.add_rope_dimension_sections([1, 1, 0, 0])
    w.add_ssm_conv_kernel(D_CONV)
    w.add_ssm_state_size(HEAD_KV)
    w.add_ssm_group_count(N_K_HEADS)
    w.add_ssm_time_step_rank(N_V_HEADS)
    w.add_ssm_inner_size(VALUE_DIM)
    if as_array:
        w.add_array(f"{arch}.attention.recurrent_layers", RECURRENT)
    else:
        w.add_full_attention_interval(4)
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
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "post_attention_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if RECURRENT[il]:
            # qwen35.cpp:66-74.
            w.add_tensor(p + "attn_qkv.weight", rnd(CONV_DIM, N_EMBD) * 2.0)
            w.add_tensor(p + "attn_gate.weight", rnd(VALUE_DIM, N_EMBD) * 2.0)
            w.add_tensor(p + "ssm_conv1d.weight", rnd(CONV_DIM, D_CONV) * 2.0)
            w.add_tensor(p + "ssm_dt.bias", rnd(N_V_HEADS) * 2.0)
            w.add_tensor(p + "ssm_a", (-(0.5 + 2.0 * rng.random(N_V_HEADS))).astype(np.float32))
            if nxt:
                # qwen3next.cpp:96,422-436: one projection, per K group
                # `ratio` betas then `ratio` alphas.
                w.add_tensor(p + "ssm_ba.weight", rnd(2 * N_V_HEADS, N_EMBD) * 2.0)
            else:
                w.add_tensor(p + "ssm_beta.weight", rnd(N_V_HEADS, N_EMBD) * 2.0)
                w.add_tensor(p + "ssm_alpha.weight", rnd(N_V_HEADS, N_EMBD) * 2.0)
            w.add_tensor(p + "ssm_norm.weight", (1.0 + rnd(HEAD_KV)).astype(np.float32))
            w.add_tensor(p + "ssm_out.weight", rnd(N_EMBD, VALUE_DIM))
        else:
            # qwen35.cpp:59-64: q and its gate interleaved per head.
            w.add_tensor(p + "attn_q.weight", rnd(2 * N_HEAD * HEAD_DIM, N_EMBD) * 3.0)
            w.add_tensor(p + "attn_k.weight", rnd(N_KV * HEAD_DIM, N_EMBD) * 3.0)
            w.add_tensor(p + "attn_v.weight", rnd(N_KV * HEAD_DIM, N_EMBD))
            w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
            w.add_tensor(p + "attn_q_norm.weight", (1.0 + rnd(HEAD_DIM)).astype(np.float32))
            w.add_tensor(p + "attn_k_norm.weight", (1.0 + rnd(HEAD_DIM)).astype(np.float32))
        if moe:
            # qwen35moe.cpp:98-107: the router, the experts, the shared
            # expert with its own one-logit gate.
            w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD) * 4.0)
            w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
            w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
            w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))
            w.add_tensor(p + "ffn_gate_inp_shexp.weight", rnd(N_EMBD) * 2.0)
            w.add_tensor(p + "ffn_gate_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
            w.add_tensor(p + "ffn_up_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
            w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, N_FF_SHEXP))
        else:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
    if separate_output:
        w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} ({arch}, array={as_array}, output={separate_output})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "qwen35-fixture.gguf",
        "--array" in sys.argv[1:],
        "--output" in sys.argv[1:],
        "--moe" in sys.argv[1:],
        "--next" in sys.argv[1:],
    )
