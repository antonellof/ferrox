#!/usr/bin/env python3
"""Generate the tiny synthetic `mistral3` GGUFs used by ferrox's
per-position attention-temperature coverage test.

`mistral3` is what every Ministral-3 (3B/8B/14B) export tags
(`conversion/mistral3.py:20`). It refused as UNAUDITED, triaged NEW
CODE, for `attention.temperature_scale`: `src/models/mistral3.cpp:5`
reads it, `:14-17` floors it on `hparams.n_ctx_orig_yarn`, and
`:153-156` multiplies Q by the per-position vector
`llama-graph.cpp:163-167` computes,

    log(floor(pos / floor_scale) + 1) * temp_scale + 1

AFTER RoPE and BEFORE `build_attn`, with `kq_scale` untouched. Until
2026-09-11 ferrox had no per-position Q scale and no gate on the key.

The verdict also described this graph as "leading-dense + MoE + shared
expert", and reading `mistral3.cpp` in full corrects that: `:64-84`
create EITHER the dense FFN (`n_expert == 0`) OR the expert bank for
EVERY layer -- there is no `leading_dense_block_count` read and no
per-layer split -- and the `_shexp` tensors at `:80-84` are created
only when `hparams.n_ff_shexp > 0`, which `load_arch_hparams` never
sets, and are consumed by NO line of the graph (`grep shexp
mistral3.cpp` hits the loader only). A `mistral3` file is a `llama`
file with three extra keys, which is what real Ministral-3 files are
(`Ministral3Model(LlamaModel)`), and that is the shape here.

One weight set, five files, so that the goldens differ ONLY by the
keys under test:

  (none)          plain dense: NORM RoPE, GQA, SwiGLU, no scaling key.
  --temp          `attention.temperature_scale = 0.5` and
                  `rope.scaling.original_context_length = 2`, so the
                  floor steps TWICE inside the six-token prompt
                  (positions 0-1 scale 1, 2-3 scale 1 + 0.5 ln 2,
                  4-5 scale 1 + 0.5 ln 3). NO scaling type: the
                  original-context key is read unconditionally
                  (`llama-model.cpp:1165`), YaRN or not.
  --temp-ctx      the same scale with NO original-context key and
                  `context_length = 2`, because `llama-model.cpp:1164`
                  seeds `n_ctx_orig_yarn` from `n_ctx_train` first.
                  A loader that took the floor from the YaRN key alone
                  would refuse (or worse, not scale) this file.
  --yarn          YaRN factor 4 over an original context of 4096 with
                  the same temperature key (whose floor is never
                  reached in six tokens, so this file exercises YaRN's
                  MAGNITUDE term `1 + 0.1 ln 4` on q and k, which
                  ferrox did not apply before this fixture existed).
  --yarn-logmul   `--yarn` plus `rope.scaling.yarn_log_multiplier = 0.5`
                  (`mistral3.cpp:9`), which turns the magnitude into
                  `(1 + 0.1 ln 4) / (1 + 0.05 ln 4)`. Real Ministral-3
                  files write this key (`conversion/mistral3.py:28`).

Weights are pseudo-random from a fixed seed so every file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_mistral3_fixture.py OUT.gguf [VARIANT]

The golden logits that go with each file are produced by llama.cpp
itself (see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "mistral3"

N_LAYER = 2
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
# mistral3.cpp:96-97 asserts n_embd_head_v == n_embd_head_k == n_rot.
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5

# Far from 1 in effect: 1 + 0.5 ln 3 = 1.55 by position 4.
TEMP_SCALE = 0.5
# A floor the six-token prompt crosses twice.
TEMP_FLOOR = 2

YARN_FACTOR = 4.0
YARN_ORIG_CTX = 4096
YARN_LOG_MUL = 0.5

VARIANTS = ("--temp", "--temp-ctx", "--yarn", "--yarn-logmul")


def main(out_path: str, variant: str | None) -> None:
    if variant is not None and variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; one of {VARIANTS}")
    rng = np.random.default_rng(0x3157A13)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-mistral3-fixture")
    w.add_block_count(N_LAYER)
    # `--temp-ctx` floors on the TRAINED context, so the file has to
    # say a short one; llama.cpp's reference tool runs at n_ctx = 128
    # regardless, and ferrox sizes nothing from this key.
    w.add_context_length(TEMP_FLOOR if variant == "--temp-ctx" else CTX)
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

    if variant is not None:
        # `conversion/mistral.py:110` / `mistral3.py:29`, verbatim.
        w.add_attn_temperature_scale(TEMP_SCALE)
    if variant == "--temp":
        w.add_rope_scaling_orig_ctx_len(TEMP_FLOOR)
    if variant in ("--yarn", "--yarn-logmul"):
        # `conversion/mistral.py:97-102`: type, factor, betas, original
        # context. The betas are llama.cpp's own defaults so that the
        # ramp is decided by the factor and the context alone.
        w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        w.add_rope_scaling_factor(YARN_FACTOR)
        w.add_rope_scaling_yarn_beta_fast(32.0)
        w.add_rope_scaling_yarn_beta_slow(1.0)
        w.add_rope_scaling_orig_ctx_len(YARN_ORIG_CTX)
    if variant == "--yarn-logmul":
        w.add_rope_scaling_yarn_log_mul(YARN_LOG_MUL)

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
        # Wide Q/K so attention is peaked: a temperature on Q is a
        # change to the softmax's sharpness, and a near-uniform softmax
        # cannot see it.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} ({variant or 'plain'})")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "mistral3-fixture.gguf",
        sys.argv[2] if len(sys.argv) > 2 else None,
    )
