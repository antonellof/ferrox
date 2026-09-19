#!/usr/bin/env python3
"""Generate the tiny synthetic `falcon-h1` GGUFs used by frink's
Falcon-H1 coverage test (`crates/frink-models/tests/falcon_h1_graphs.rs`).

`falcon-h1` (Falcon-H1 0.5B / 1.5B / 3B / 7B / 34B) runs attention AND
a Mamba-2 block on EVERY layer, in parallel, on the same normed input,
and sums them before the residual (`falcon-h1.cpp:130-167`):

    cur      = attn_norm(inpL)                                    # :137
    attn_out = attn(cur)              (rotated NEOX, :141-154)
    ssm_out  = mamba2(attn_norm(inpL))                            # :156-158
    inpSA    = inpL + attn_out + ssm_out                          # :160-161
    cur      = inpSA + ffn(ffn_norm(inpSA))                       # :167-174

`:12` marks every layer recurrent; `head_count` / `head_count_kv` are
scalars (`conversion/falcon_h1.py:108-109`). `wo_b` is created
optional (`:76`) and NOT passed to `build_attn` (`:154` passes NULL):
unread upstream. `ssm_norm` is optional (`:70`). `ffn_norm` is looked
up with the two-argument `LLM_TN` overload (`:80`), i.e. as
`blk.N.ffn_norm` with NO `.weight` suffix -- the fixture writes that
spelling, which is what libllama loads. The converter folds every
Falcon-H1 multiplier into the weights (`conversion/falcon_h1.py:
59-95`), so the graph has none.

Variants:
  * default          split QKV, `ssm_norm` present, tied output
  * `--no-ssm-norm`  without `ssm_norm` (`:70`, optional)
  * `--output`       a separate `output.weight`

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_falcon_h1_fixture.py OUT.gguf [--no-ssm-norm] [--output]

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "falcon-h1"

N_EMBD = 24
N_HEAD = 4
N_KV = 2
HEAD_DIM = N_EMBD // N_HEAD
N_FF = 40
N_VOCAB = 48
N_LAYER = 3
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5

D_CONV = 4
D_INNER = 48
D_STATE = 8
SSM_HEADS = 4
N_GROUP = 2
CONV_W = D_INNER + 2 * N_GROUP * D_STATE
D_IN_PROJ = 2 * D_INNER + 2 * N_GROUP * D_STATE + SSM_HEADS


def main(out_path: str, ssm_norm: bool, separate_output: bool, ffn_norm_suffix: bool) -> None:
    rng = np.random.default_rng(0xFA1C)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-falcon-h1-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_ssm_conv_kernel(D_CONV)
    w.add_ssm_inner_size(D_INNER)
    w.add_ssm_state_size(D_STATE)
    w.add_ssm_time_step_rank(SSM_HEADS)
    w.add_ssm_group_count(N_GROUP)
    w.add_vocab_size(N_VOCAB)
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
        # :55-71: the Mamba-2 block on every layer.
        w.add_tensor(p + "ssm_in.weight", rnd(D_IN_PROJ, N_EMBD) * 2.0)
        w.add_tensor(p + "ssm_conv1d.weight", rnd(CONV_W, D_CONV) * 2.0)
        w.add_tensor(p + "ssm_conv1d.bias", rnd(CONV_W))
        w.add_tensor(p + "ssm_dt.bias", rnd(SSM_HEADS) * 2.0)
        w.add_tensor(
            p + "ssm_a",
            (-(0.5 + 2.0 * rng.random(SSM_HEADS))).astype(np.float32).reshape(SSM_HEADS, 1),
        )
        w.add_tensor(p + "ssm_d", (1.0 + rnd(SSM_HEADS)).astype(np.float32).reshape(SSM_HEADS, 1))
        if ssm_norm:
            w.add_tensor(
                p + "ssm_norm.weight",
                (1.0 + rnd(N_GROUP, D_INNER // N_GROUP)).astype(np.float32),
            )
        w.add_tensor(p + "ssm_out.weight", rnd(N_EMBD, D_INNER))
        # :72-77: attention on every layer.
        w.add_tensor(p + "attn_q.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(N_KV * HEAD_DIM, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(N_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
        # :80: the two-argument overload, no suffix.
        w.add_tensor(
            p + ("ffn_norm.weight" if ffn_norm_suffix else "ffn_norm"),
            (1.0 + rnd(N_EMBD)).astype(np.float32),
        )
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
    print(
        f"wrote {out_path} (ssm_norm={ssm_norm}, output={separate_output}, "
        f"ffn_norm_suffix={ffn_norm_suffix})"
    )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "falcon-h1-fixture.gguf",
        "--no-ssm-norm" not in sys.argv[1:],
        "--output" in sys.argv[1:],
        "--ffn-norm-suffix" in sys.argv[1:],
    )
