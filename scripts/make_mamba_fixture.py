#!/usr/bin/env python3
"""Generate the tiny synthetic `jamba`, `mamba` and `mamba2` GGUFs used by
frink's Mamba-1 coverage test (`crates/frink-models/tests/mamba_graphs.rs`).

Three architectures, one block body each:

  * `jamba` (AI21 Jamba-v0.1 / 1.5): attention or a MAMBA-1 block
    (`build_mamba_layer`, `mamba-base.cpp:4-148`) where `head_count_kv`
    is 0 (`jamba.cpp:8-10`), then an FFN that is dense or MoE PER LAYER
    by whether `blk.N.ffn_gate_inp.weight` exists (`:89-101,152`): a
    softmax MoE with `norm_w = false` (`:164`) and no shared expert.
    Its Mamba-1 block REQUIRES the RMS norms on dt, B and C
    (`:49,52-53`). The attention has NO RoPE (`:98`, "No RoPE :)").
    `d_inner == 2 n_embd` is asserted (`:25`).
  * `mamba` (Mamba-130M to 2.8B, FalconMamba-7B): every layer the
    Mamba-1 block with no FFN at all (`mamba.cpp:73-88`; the converter
    writes `head_count 0`, `feed_forward_length 0`, `conversion/mamba.py`).
    `--dt-b-c-rms` writes `ssm.dt_b_c_rms = true` (FalconMamba,
    `mamba.cpp:7`): a WEIGHTLESS RMSNorm on dt, B and C (`mamba-base.cpp:
    84-88` with NULL weights).
  * `mamba2` (Mamba-Codestral-7B): every layer the Mamba-2 block
    (`mamba2.cpp:52-63`, `ssm_norm` REQUIRED), no FFN.

Layers (four) for `jamba`: mamba + MoE, attention + dense, mamba + dense,
mamba + MoE (`head_count_kv [0, 2, 0, 0]`, the router on layers 0 and
3). Three for the pure models.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_mamba_fixture.py OUT.gguf --arch {jamba,mamba,mamba2} [--dt-b-c-rms] [--output]

Weights are pseudo-random from a fixed seed so the files are byte-stable.
The golden logits that go with them are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

N_EMBD = 24
N_HEAD = 4
N_KV = 2
HEAD_DIM = N_EMBD // N_HEAD
N_FF = 40
N_VOCAB = 48
CTX = 64
RMS_EPS = 1e-5

# Mamba-1 (`jamba.cpp:25`: d_inner == 2 n_embd).
D_CONV = 4
D_INNER = 2 * N_EMBD
D_STATE = 8
DT_RANK = 3

# Mamba-2 (pure `mamba2`).
SSM_HEADS = 4
N_GROUP = 2
CONV_W2 = D_INNER + 2 * N_GROUP * D_STATE
D_IN_PROJ2 = 2 * D_INNER + 2 * N_GROUP * D_STATE + SSM_HEADS

# Jamba's MoE.
N_EXPERT = 4
N_EXPERT_USED = 2

JAMBA_KV = [0, N_KV, 0, 0]
JAMBA_MOE = [True, False, False, True]


def main(out_path: str, arch: str, dt_b_c_rms: bool, separate_output: bool) -> None:
    rng = np.random.default_rng({"jamba": 0x3A4B, "mamba": 0x4A4B, "mamba2": 0x4A4C}[arch])

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    pure = arch != "jamba"
    n_layer = 3 if pure else len(JAMBA_KV)
    w = gguf.GGUFWriter(out_path, arch)
    w.add_name(f"frink-{arch}-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    if pure:
        # conversion/mamba.py:155-156: "unused, but seemingly required".
        w.add_feed_forward_length(0)
        w.add_head_count(0)
    else:
        w.add_feed_forward_length(N_FF)
        w.add_head_count(N_HEAD)
        w.add_head_count_kv(JAMBA_KV)
        w.add_key_length(HEAD_DIM)
        w.add_value_length(HEAD_DIM)
        w.add_expert_count(N_EXPERT)
        w.add_expert_used_count(N_EXPERT_USED)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_ssm_conv_kernel(D_CONV)
    w.add_ssm_inner_size(D_INNER)
    w.add_ssm_state_size(D_STATE)
    if arch == "mamba2":
        w.add_ssm_time_step_rank(SSM_HEADS)
        w.add_ssm_group_count(N_GROUP)
    else:
        w.add_ssm_time_step_rank(DT_RANK)
    if dt_b_c_rms:
        w.add_ssm_dt_b_c_rms(True)
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

    def mamba1(p: str, norms: bool) -> None:
        # mamba.cpp:55-65 / jamba.cpp:44-56. Shapes are ggml order reversed.
        w.add_tensor(p + "ssm_in.weight", rnd(2 * D_INNER, N_EMBD) * 2.0)
        w.add_tensor(p + "ssm_conv1d.weight", rnd(D_INNER, D_CONV) * 2.0)
        w.add_tensor(p + "ssm_conv1d.bias", rnd(D_INNER))
        w.add_tensor(p + "ssm_x.weight", rnd(DT_RANK + 2 * D_STATE, D_INNER) * 2.0)
        if norms:
            w.add_tensor(p + "ssm_dt_norm.weight", (1.0 + rnd(DT_RANK)).astype(np.float32))
        w.add_tensor(p + "ssm_dt.weight", rnd(D_INNER, DT_RANK) * 2.0)
        w.add_tensor(p + "ssm_dt.bias", rnd(D_INNER) * 2.0)
        if norms:
            w.add_tensor(p + "ssm_b_norm.weight", (1.0 + rnd(D_STATE)).astype(np.float32))
            w.add_tensor(p + "ssm_c_norm.weight", (1.0 + rnd(D_STATE)).astype(np.float32))
        # {d_state, d_inner} in ggml: numpy (d_inner, d_state), stored negative.
        w.add_tensor(p + "ssm_a", (-(0.5 + 2.0 * rng.random((D_INNER, D_STATE)))).astype(np.float32))
        w.add_tensor(p + "ssm_d", (1.0 + rnd(D_INNER)).astype(np.float32))
        w.add_tensor(p + "ssm_out.weight", rnd(N_EMBD, D_INNER))

    def mamba2(p: str) -> None:
        w.add_tensor(p + "ssm_in.weight", rnd(D_IN_PROJ2, N_EMBD) * 2.0)
        w.add_tensor(p + "ssm_conv1d.weight", rnd(CONV_W2, D_CONV) * 2.0)
        w.add_tensor(p + "ssm_conv1d.bias", rnd(CONV_W2))
        w.add_tensor(p + "ssm_dt.bias", rnd(SSM_HEADS) * 2.0)
        w.add_tensor(p + "ssm_a", (-(0.5 + 2.0 * rng.random(SSM_HEADS))).astype(np.float32).reshape(SSM_HEADS, 1))
        w.add_tensor(p + "ssm_d", (1.0 + rnd(SSM_HEADS)).astype(np.float32).reshape(SSM_HEADS, 1))
        w.add_tensor(p + "ssm_norm.weight", (1.0 + rnd(N_GROUP, D_INNER // N_GROUP)).astype(np.float32))
        w.add_tensor(p + "ssm_out.weight", rnd(N_EMBD, D_INNER))

    for il in range(n_layer):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if arch == "mamba":
            mamba1(p, norms=False)
            continue
        if arch == "mamba2":
            mamba2(p)
            continue
        if JAMBA_KV[il] == 0:
            mamba1(p, norms=True)
        else:
            w.add_tensor(p + "attn_q.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD) * 4.0)
            w.add_tensor(p + "attn_k.weight", rnd(N_KV * HEAD_DIM, N_EMBD) * 4.0)
            w.add_tensor(p + "attn_v.weight", rnd(N_KV * HEAD_DIM, N_EMBD))
            w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
        w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if JAMBA_MOE[il]:
            w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD) * 4.0)
            w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF))
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
    print(f"wrote {out_path} ({arch}, dt_b_c_rms={dt_b_c_rms}, output={separate_output})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]
    arch = argv[argv.index("--arch") + 1] if "--arch" in argv else "jamba"
    if arch in args:
        args.remove(arch)
    main(
        args[0] if args else f"{arch}-fixture.gguf",
        arch,
        "--dt-b-c-rms" in argv,
        "--output" in argv,
    )
