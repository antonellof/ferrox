#!/usr/bin/env python3
"""Generate the tiny synthetic `phimoe` GGUFs used by frink's Phi-3.5-MoE
coverage test.

`phimoe` (Phi-3.5-MoE-instruct) refused as a `dedicated` row for its
REQUIRED biases. Its graph is `phi3`'s (`models.h:626-634`, `using graph
= llama_model_phi3::graph`), on tensors of its own (`phimoe.cpp:12-46`):

  * `:28-29,35-36,20-21` norm weights AND biases, REQUIRED, handed to
    `LLM_NORM_RMS` by `phi3.cpp:99-102,137-139,174-177`: an RMSNorm
    with a bias added after the multiply, `NormOp::RmsBias`, one graph
    of 140 on the generic path (`capability::BIASED_RMS_NORM`).
  * `:31` `create_tensor_qkv` (Q/K/V biases OPTIONAL; PhiMoE has them),
    `:33` `attn_output.bias` REQUIRED, `:23` `output.bias` REQUIRED
    (`crate::proj_bias`), `:22` `output.weight` REQUIRED.
  * `:38-41` routed experts, `phi3.cpp:153-163`: softmax over the
    router, top-k, `norm_w = true` (renormalised), `expert_weights_scale`
    that `phimoe.cpp` never reads (0, skipped), no shared expert.
  * `:43-44` LongRoPE: `rope_factors_long` / `rope_factors_short`
    `{n_embd_head/2}`, picked by context against
    `rope.scaling.original_context_length`, with `rope.scaling.attn_factor`
    from the converter (`conversion/phi.py:193-201`); NEOX RoPE
    (llama-model.cpp:2638).
  * `phimoe.cpp:3-10` read NO window key, so `swa_type` stays NONE and
    the `attention.sliding_window` every export writes is dead
    metadata (`capability::swa_window_override`, the `phi3` answer).

Keys as `Phi3MiniModel.set_gguf_parameters` writes them (`phi.py:146-
171`): `context_length`, `rope.scaling.original_context_length`,
`embedding_length`, `feed_forward_length`, `block_count`, `head_count`,
`head_count_kv`, `attention.layer_norm_rms_epsilon`,
`rope.dimension_count`, `rope.freq_base`, `attention.sliding_window`,
`rope.scaling.attn_factor`; `expert_count` / `expert_used_count`
(`:346-347`). No `rope.scaling.type` key.

Shapes:

  * (default) 3 layers, 4 experts, top-2, `context_length 64` over
    `original 32` (so `rope_factors_long` is the pair in use and the
    attn factor is `sqrt(1 + ln 2 / ln 32) = 1.0955`), a window of 8
    the graph ignores.
  * `--no-longrope`: no factor tensors, no attn factor, context = orig.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_phimoe_fixture.py OUT.gguf [--no-longrope]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse
import math

import numpy as np

import gguf

ARCH = "phimoe"

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
ORIG_CTX = 32
RMS_EPS = 1e-5
ROPE_BASE = 10_000.0
WINDOW = 8


def main(out_path: str, no_longrope: bool) -> None:
    rng = np.random.default_rng(0x9E0E)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w(n: int) -> np.ndarray:
        return (1.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    def norm_b(n: int) -> np.ndarray:
        return (0.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    ctx = ORIG_CTX if no_longrope else CTX
    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-phimoe-fixture")
    w.add_context_length(ctx)
    w.add_rope_scaling_orig_ctx_len(ORIG_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_block_count(N_LAYER)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_sliding_window(WINDOW)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_count(N_EXPERT)
    if not no_longrope:
        scale = CTX / ORIG_CTX
        w.add_rope_scaling_attn_factors(math.sqrt(1 + math.log(scale) / math.log(ORIG_CTX)))
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
    if not no_longrope:
        # `generate_extra_tensors` writes ONE pair for the model
        # (`phi.py:203-204`); `phimoe.cpp:43-44` load it duplicated per
        # layer.
        w.add_tensor(
            "rope_factors_long.weight",
            (1.0 + rng.random(HEAD_DIM // 2) * 3.0).astype(np.float32),
        )
        w.add_tensor(
            "rope_factors_short.weight",
            (1.0 + rng.random(HEAD_DIM // 2) * 0.5).astype(np.float32),
        )

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM
    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", norm_w(N_EMBD))
        w.add_tensor(p + "attn_norm.bias", norm_b(N_EMBD))
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_q.bias", rnd(n_embd_q))
        w.add_tensor(p + "attn_k.bias", rnd(n_embd_kv))
        w.add_tensor(p + "attn_v.bias", rnd(n_embd_kv))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "attn_output.bias", rnd(N_EMBD))
        w.add_tensor(p + "ffn_norm.weight", norm_w(N_EMBD))
        w.add_tensor(p + "ffn_norm.bias", norm_b(N_EMBD))
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD) * 3.0)
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF) * 2.0)

    w.add_tensor("output_norm.weight", norm_w(N_EMBD))
    w.add_tensor("output_norm.bias", norm_b(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))
    w.add_tensor("output.bias", rnd(N_VOCAB) * 4.0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="phimoe-fixture.gguf")
    ap.add_argument("--no-longrope", action="store_true")
    args = ap.parse_args()
    main(args.out, args.no_longrope)
