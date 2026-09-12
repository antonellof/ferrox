#!/usr/bin/env python3
"""Generate the tiny synthetic `command-r` GGUFs used by ferrox's Command-R
coverage test.

`command-r` (Command-R 35B, Aya-23) refused as a `dedicated` row for its
parallel residual. Read line by line, `command-r.cpp` is:

  * `:68` `build_norm(inpL, attn_norm, NULL, LLM_NORM, il)`: a LayerNorm
    with a learned weight and NO bias, `dbrx`'s variant
    (`capability::WEIGHTED_LAYER_NORM`); `:127` the same on `output_norm`.
  * `:106-119` the shared-norm PARALLEL residual: the FFN reads the
    vector attention read (`ffn_inp = cur` at `:70` is `attn_norm(x)`),
    `cur + inpL + attn_out` summed once (`crate::parallel_residual`).
  * `:4,137-138` `logit_scale`, OPTIONAL, MULTIPLIED onto the logits when
    nonzero (`crate::scalar_multipliers`, `LogitScaleUse::AsIsOptional`).
  * `:21` a TIED lm_head (`output` is `TENSOR_DUPLICATED` from
    `token_embd`); no `output.weight` in the file.
  * `:28-31` at `n_layer >= 64` ONLY (Command-R+): `attn_q_norm` /
    `attn_k_norm` of `{n_embd_head_k, n_head}`, REQUIRED, applied as a
    per-head LayerNorm with a distinct weight per head (`:80,87`).
    `crate::qk_layer_norm` refuses that by name.
  * NORM RoPE (llama-model.cpp:2582), SwiGLU, no biases anywhere.

Keys as `conversion/command_r.py` writes them through `TextModel.
set_gguf_parameters`: `context_length`, `embedding_length`,
`block_count`, `feed_forward_length`, `head_count`, `head_count_kv`,
`attention.layer_norm_epsilon`, `rope.freq_base`, plus `logit_scale`
and `rope.scaling.type = none`.

Shapes:

  * (default) 3 layers, no QK norm: Command-R 35B / Aya-23. SERVED.
  * `--plus`: 64 layers (`n_embd 16`, 2 heads), `attn_q_norm` `{8, 2}` /
    `attn_k_norm` `{8, 1}` on every layer, as llama.cpp REQUIRES at that
    depth. Command-R+. REFUSED by name from a file libllama runs.
  * `--no-logit-scale`: the default shape without the key, to pin that
    absent means no scale (`:137` `if (f_logit_scale)`).

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_command_r_fixture.py OUT.gguf [--plus] [--no-logit-scale]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "command-r"

N_VOCAB = 48
CTX = 64
LN_EPS = 1e-5
ROPE_BASE = 8_000_000.0  # Command-R's rope_theta
LOGIT_SCALE = 0.0625  # Command-R 35B's config.json


def main(out_path: str, plus: bool, no_logit_scale: bool) -> None:
    rng = np.random.default_rng(0xC0DE)
    if plus:
        n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff = 64, 16, 2, 1, 8, 32
    else:
        n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff = 3, 32, 4, 2, 8, 48

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w(*shape: int) -> np.ndarray:
        return (1.0 + rng.standard_normal(shape) * 0.3).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-command-r-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head_kv)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_layer_norm_eps(LN_EPS)
    if not no_logit_scale:
        w.add_logit_scale(LOGIT_SCALE)
    w.add_rope_scaling_type(gguf.RopeScalingType.NONE)
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

    # Tied: `output` is read from this tensor (`:21`).
    w.add_tensor("token_embd.weight", rnd(N_VOCAB, n_embd))

    n_embd_q = n_head * head_dim
    n_embd_kv = n_head_kv * head_dim
    for il in range(n_layer):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", norm_w(n_embd))
        if plus:
            # ne = [n_embd_head_k, n_head] -> numpy (n_head, head_dim).
            w.add_tensor(p + "attn_q_norm.weight", norm_w(n_head, head_dim))
            w.add_tensor(p + "attn_k_norm.weight", norm_w(n_head_kv, head_dim))
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, n_embd) * 2.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, n_embd) * 2.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, n_embd))
        w.add_tensor(p + "attn_output.weight", rnd(n_embd, n_embd_q))
        w.add_tensor(p + "ffn_gate.weight", rnd(n_ff, n_embd))
        w.add_tensor(p + "ffn_up.weight", rnd(n_ff, n_embd))
        w.add_tensor(p + "ffn_down.weight", rnd(n_embd, n_ff) * 2.0)

    w.add_tensor("output_norm.weight", norm_w(n_embd))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="command-r-fixture.gguf")
    ap.add_argument("--plus", action="store_true")
    ap.add_argument("--no-logit-scale", action="store_true")
    args = ap.parse_args()
    main(args.out, args.plus, args.no_logit_scale)
