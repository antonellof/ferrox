#!/usr/bin/env python3
"""Generate the tiny synthetic `cohere2` GGUFs used by frink's Command-R7B
coverage test.

`cohere2` (Command-R7B, Command-A) refused as a `dedicated` row for its
parallel residual. `cohere2.cpp` is `command-r.cpp` plus a window:

  * `:78` `build_norm(inpL, attn_norm, NULL, LLM_NORM)`: the weighted
    LayerNorm without a bias (`capability::WEIGHTED_LAYER_NORM`).
  * `:120-134` the shared-norm PARALLEL residual (`crate::parallel_
    residual`); `:30-31` a TIED lm_head (`TENSOR_DUPLICATED`).
  * `:14,153-154` `logit_scale`, REQUIRED here (`get_key` with no
    `false`) and MULTIPLIED onto the logits (`LogitScaleUse::AsIs`).
  * `:4-7` `swa_type = STANDARD` with `set_swa_pattern(4)` seeded and
    the scalar `attention.sliding_window_pattern` overriding it; `:13`
    `attention.sliding_window` REQUIRED; `:9-12` the sliding layers'
    rope base and scale follow the model's unless `rope.freq_base_swa`
    says otherwise.
  * `:72,90-99` ONLY the sliding layers are rotated: `if (is_swa)`
    around `ggml_rope_ext`, with llama3 `rope_freqs.weight` factors when
    present. That is `crate::rope_layers`'s `SlidingOnly`, the
    `exaone-moe` rule, on a NORM layout (llama-model.cpp:2583).
  * split Q/K/V with no biases (the converter DROPS the all-zero bias
    tensors, `conversion/command_r.py:34-41`), SwiGLU.

Keys as `Cohere2Model.set_gguf_parameters` writes them: the base set,
`logit_scale`, `attention.sliding_window`, `vocab_size`,
`rope.dimension_count = rotary_pct * head_dim`, `rope.scaling.type =
none`, `rope.freq_base`, `attention.layer_norm_epsilon`. Command-R7B:
window 4096, pattern 4 (not written; the seed), `rotary_pct 1.0`,
`logit_scale 0.25`, `rope_theta 50000`.

Shapes:

  * (default) 4 layers, window 3 over a 6-token prompt (so the window
    is visible in the golden), pattern from the seed (layers 0-2 slide
    and rotate, layer 3 is full and unrotated), `logit_scale 0.25`.
  * `--pattern-key`: the same with `sliding_window_pattern = 2` written
    (layers 0 and 2 slide), to pin that the scalar key is honoured.
  * `--no-window`: the key `:13` REQUIRES left out. libllama refuses the
    file (`key not found in model: cohere2.attention.sliding_window`,
    measured); frink refuses it by name too.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_cohere2_fixture.py OUT.gguf [--pattern-key] [--no-window]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "cohere2"

N_LAYER = 4
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
ROPE_DIM = 8  # rotary_pct 1.0
N_FF = 48
N_VOCAB = 48
CTX = 64
LN_EPS = 1e-5
ROPE_BASE = 50_000.0
LOGIT_SCALE = 0.25
WINDOW = 3


def main(out_path: str, pattern_key: bool, no_window: bool) -> None:
    rng = np.random.default_rng(0xC0E2)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w(n: int) -> np.ndarray:
        return (1.0 + rng.standard_normal(n) * 0.3).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-cohere2-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_layer_norm_eps(LN_EPS)
    w.add_logit_scale(LOGIT_SCALE)
    if not no_window:
        w.add_sliding_window(WINDOW)
    if pattern_key:
        w.add_sliding_window_pattern(2)
    w.add_vocab_size(N_VOCAB)
    w.add_rope_dimension_count(ROPE_DIM)
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

    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM
    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", norm_w(N_EMBD))
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 2.0)

    w.add_tensor("output_norm.weight", norm_w(N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="cohere2-fixture.gguf")
    ap.add_argument("--pattern-key", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()
    main(args.out, args.pattern_key, args.no_window)
