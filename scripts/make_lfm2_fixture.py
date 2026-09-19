#!/usr/bin/env python3
"""Generate the tiny synthetic `lfm2` GGUF used by frink's LFM2 coverage
test (`crates/frink-models/tests/lfm2_graphs.rs`).

`lfm2` (LFM2-350M / 700M / 1.2B / 2.6B, LFM2-VL's text tower) is the
first HYBRID row on the generic path. `.scratch/llama.cpp/src/models/
lfm2.cpp:9-11` marks layer `il` recurrent when `n_head_kv(il) == 0`
(`conversion/lfm2.py:37-40` writes `head_count_kv` as an array with a
`0` on every `conv` layer), and the graph at :192-208 is ONE residual
topology for both kinds:

    cur = attn_norm(inpL)                                    # :196
    cur = is_recr(il) ? shortconv(cur) : attn(cur)           # :197-198
    cur = inpL + cur                                         # :203
    cur = cur + ffn(ffn_norm(cur))                           # :204-207

The short convolution (:139-189) is
`in_proj` `{n_embd, 3 n_embd}` split into `b, c, x` (:151-160),
`bx = b * x` (:162), a causal depthwise conv of width `l_cache` over
`bx` with the previous `l_cache - 1` inputs as the state (:164-186,
`ggml_ssm_conv`), `y = c * conv_out` (:187), then `out_proj` (:188).
The attention layers are GQA with a PER-HEAD RMS QK norm
(`{n_embd_head_k}`, :74-75, :117-119), NEOX RoPE (llama-model.cpp:2666),
`wo` `{n_embd, n_embd}` (:79). The final norm is stored as
`token_embd_norm.weight` (`LLM_TENSOR_OUTPUT_NORM_LFM2`,
llama-arch.cpp:384, "fix for wrong tensor name") and applied at :212 as
the OUTPUT norm; there is no norm on the embeddings. `output.weight` is
optional with a tied fallback (:37-41). Dense gated SiLU FFN on every
layer (:100-108; `n_layer_dense_lead = n_layer`, :13).

Layers (four, alternating so both kinds appear twice and a conv layer
is first, as in every real export):

    blk.0  conv    (head_count_kv 0)
    blk.1  attn    (head_count 4, head_count_kv 2)
    blk.2  conv
    blk.3  attn

Variants:
  * default          split `attn_q/k/v`, tied output, `l_cache 3`
  * `--moe`          the `lfm2moe` architecture (LFM2-8B-A1B): the same
                     graph (`models.h:1899`), `leading_dense_block_count
                     1` (`lfm2moe.cpp:8,38`), a sigmoid MoE on the rest
                     with `exp_probs_b` REQUIRED (`:42-47`), `norm_w =
                     true` (lfm2.cpp:118); `expert_weights_scale` is
                     read by nothing in its hparams
  * `--fused-qkv`    the converter's fused `attn_qkv.weight`
                     (`create_tensor_qkv`, :78)
  * `--output`       a separate `output.weight`
  * `--window`       `attention.sliding_window 4`, which lfm2.cpp:24-29
                     honours on the ATTENTION layers only; frink refuses
                     this file by name (`crate::shortconv`), libllama
                     runs it

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_lfm2_fixture.py OUT.gguf [--fused-qkv] [--output] [--window] [--moe]

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "lfm2"
MOE_ARCH = "lfm2moe"

N_EMBD = 24
N_HEAD = 4
N_KV = 2
HEAD_DIM = N_EMBD // N_HEAD
N_FF = 40
N_VOCAB = 48
CTX = 64
L_CACHE = 3
ROPE_BASE = 10000.0
RMS_EPS = 1e-5
N_EXPERT = 4
N_EXPERT_USED = 2
N_FF_EXP = 16
N_DENSE_LEAD = 1

# head_count_kv per layer; 0 is a conv layer (lfm2.cpp:10).
KV_PER_LAYER = [0, N_KV, 0, N_KV]


def main(
    out_path: str, fused_qkv: bool, separate_output: bool, window: bool, moe: bool
) -> None:
    n_layer = len(KV_PER_LAYER)
    rng = np.random.default_rng(0x1F02)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    arch = MOE_ARCH if moe else ARCH
    w = gguf.GGUFWriter(out_path, arch)
    w.add_name(f"frink-{arch}-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(KV_PER_LAYER)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_shortconv_l_cache(L_CACHE)
    w.add_vocab_size(N_VOCAB)
    if moe:
        # conversion/lfm2.py:107-109.
        w.add_expert_count(N_EXPERT)
        w.add_expert_used_count(N_EXPERT_USED)
        w.add_expert_feed_forward_length(N_FF_EXP)
        w.add_leading_dense_block_count(N_DENSE_LEAD)
        w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        # lfm2moe.cpp reads no LLM_KV_EXPERT_WEIGHTS_SCALE: dead metadata,
        # libllama's golden is unscaled.
        w.add_expert_weights_scale(2.5)
    if window:
        w.add_sliding_window(4)
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
        # "for operator_norm" (:70): both kinds have it.
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if nkv == 0:
            # :80-82. The conv kernel is {l_cache, n_embd} in ggml order,
            # i.e. numpy (n_embd, l_cache): one l_cache-tap filter per
            # channel (conversion/lfm2.py:62-64 squeezes the HF
            # (n_embd, 1, l_cache) to 2-D).
            w.add_tensor(p + "shortconv.conv.weight", rnd(N_EMBD, L_CACHE) * 2.0)
            w.add_tensor(p + "shortconv.in_proj.weight", rnd(3 * N_EMBD, N_EMBD))
            w.add_tensor(p + "shortconv.out_proj.weight", rnd(N_EMBD, N_EMBD))
        else:
            q = rnd(N_HEAD * HEAD_DIM, N_EMBD) * 4.0
            k = rnd(nkv * HEAD_DIM, N_EMBD) * 4.0
            v = rnd(nkv * HEAD_DIM, N_EMBD)
            if fused_qkv:
                w.add_tensor(p + "attn_qkv.weight", np.concatenate([q, k, v], axis=0))
            else:
                w.add_tensor(p + "attn_q.weight", q)
                w.add_tensor(p + "attn_k.weight", k)
                w.add_tensor(p + "attn_v.weight", v)
            # Per head (:74-75): drawn away from one so a whole-vector
            # reading of the same bytes is a different graph.
            w.add_tensor(p + "attn_q_norm.weight", (1.0 + rnd(HEAD_DIM)).astype(np.float32))
            w.add_tensor(p + "attn_k_norm.weight", (1.0 + rnd(HEAD_DIM)).astype(np.float32))
            w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
        w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if moe and il >= N_DENSE_LEAD:
            # lfm2moe.cpp:42-47: routed experts with the REQUIRED router bias.
            w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
            w.add_tensor(
                p + "exp_probs_b.bias",
                (rng.standard_normal(N_EXPERT) * 0.6).astype(np.float32),
            )
            w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
            w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
            w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))
        else:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    # The OUTPUT norm, under the embedding-norm name (llama-arch.cpp:384).
    w.add_tensor("token_embd_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
    if separate_output:
        w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {out_path} ({arch}, kv per layer {KV_PER_LAYER}, fused_qkv={fused_qkv}, "
        f"output={separate_output}, window={window})"
    )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "lfm2-fixture.gguf",
        "--fused-qkv" in sys.argv[1:],
        "--output" in sys.argv[1:],
        "--window" in sys.argv[1:],
        "--moe" in sys.argv[1:],
    )
