#!/usr/bin/env python3
"""Generate the tiny synthetic `talkie` GGUFs used by ferrox's
weightless-norm / skip-stream coverage test.

`talkie` refused as UNAUDITED, triaged NEW CODE, on four things
(`src/models/talkie.cpp`), none of which any other generic-path graph
has and each measured over all 140 graphs:

  * **No norm weights.** Every `build_norm` in the graph is
    `build_norm(x, nullptr, nullptr, LLM_NORM_RMS, il)`: on the
    embeddings before layer 0 (`:50`), `attn_norm` (`:68`), the K
    norm (`:90`), `ffn_norm` (`:110`), the final norm (`:137`). The
    file carries no `attn_norm` / `ffn_norm` / `output_norm` tensor.
  * **A per-head SCALAR Q gain.** `attn_q_norm` is `{1, n_head}`
    (`:26`): RMS over each head's `head_dim`, then one learned scalar
    per head -- neither a `head_dim` vector nor a whole-vector weight.
    Applied AFTER RoPE (`:82-91`, "reference applies qknorm after
    rope"), and K gets the weightless per-head RMS on the same side.
  * **A learned skip stream.** `embd_skip` is the normed embedding
    (`:52`), and every layer adds `embd_skip * out_scale[il]` after its
    FFN residual (`:123-126`), `out_scale` being
    `blk.N.layer_output_scale.weight` `{1}` (`:32`).
  * **Per-tensor gains on `wo` and `ffn_down`.** `conversion/
    talkie.py:26-31` writes `blk.N.attn_output.scale` (from
    `attn_gain.a_g`) and `blk.N.ffn_down.scale` (from `mlp_gain.a_g`),
    the `{1}` companions `build_lora_mm` multiplies the projection's
    output by (`wo_s` at `:94`, `ffn_down_s` at `:117`).

Plus `{arch}.logit_scale`, REQUIRED (`:5`), multiplied onto the logits
(`:141`, the `grok` shape), and NEOX RoPE. The converter also absorbs
an "inverse RoPE" sign flip into the Q/K weights (`talkie.py:33-43`),
which changes the file and not the graph, so a fixture need not
imitate it.

`crates/ferrox-models/src/skip_stream.rs` and the `NormOp::RmsNoParams`
/ `QkNormStyle::PerHeadScalar` variants are the seams. `--no-gains`
omits the two `.scale` tensors (a hand-written shape the converter
never produces) so that the test can pin that the gains are read and
applied rather than silently dropped.

Weights are pseudo-random from a fixed seed so the file is byte-stable.
`out_scale`, the Q gains and the two projection gains are drawn AWAY
from one (or zero, for `out_scale`), so that any of them dropped moves
the logits by far more than the tolerance.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_talkie_fixture.py OUT.gguf [--no-gains]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "talkie"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
# conversion/talkie.py:19: F.rms_norm's default, torch.finfo(float32).eps.
RMS_EPS = 1.1920929e-07
LOGIT_SCALE = 0.75


def main(out_path: str, gains: bool) -> None:
    rng = np.random.default_rng(0x7A1C)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def away_from_one(*shape: int) -> np.ndarray:
        return (1.5 + rng.standard_normal(shape) * 0.5).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-talkie-fixture")
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
    # talkie.cpp:5, REQUIRED; talkie.py:32-33 writes it from lm_head_gain.
    w.add_logit_scale(LOGIT_SCALE)
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
        # No attn_norm / ffn_norm: talkie.cpp:68,110 norm without weights.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q) * 4.0)
        # :26 `{1, n_head}`: ne = [1, n_head] -> numpy [n_head, 1]. One
        # scalar per head, away from one.
        w.add_tensor(p + "attn_q_norm.weight", away_from_one(N_HEAD, 1))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 4.0)
        # :32 `{1}`: the skip stream's per-layer scalar, away from zero.
        w.add_tensor(p + "layer_output_scale.weight", np.array([0.6 + 0.2 * il], dtype=np.float32))
        if gains:
            # talkie.py:26-31: the `{1}` companions of wo and ffn_down.
            w.add_tensor(p + "attn_output.scale", np.array([1.7 - 0.2 * il], dtype=np.float32))
            w.add_tensor(p + "ffn_down.scale", np.array([0.5 + 0.3 * il], dtype=np.float32))

    # No output_norm (:137 norms without weights); `output` is REQUIRED (:16).
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="talkie-fixture.gguf")
    ap.add_argument("--no-gains", action="store_true")
    args = ap.parse_args()
    main(args.out, not args.no_gains)
