#!/usr/bin/env python3
"""Generate the tiny synthetic `step35` GGUF used by ferrox's per-layer
SwiGLU-clamp and two-valued rotary-width coverage test.

`step35` (StepFun Step-3.5-Flash, 196B-A11B) was triaged NEW CODE on two
facts after every other blocker its verdict named had closed on an
earlier seam:

  1. **Per-layer SwiGLU clamp arrays.** `src/models/step35.cpp:28-29`
     reads `{arch}.swiglu_clamp_exp` and `{arch}.swiglu_clamp_shexp`
     as `n_layer`-long arrays (optional; `get_key_or_arr`, so a scalar
     is broadcast), and llama.cpp's GENERIC FFN builders apply layer
     `il`'s entry when it is above `1e-6`:

         up  = clamp(up, -limit, limit)
         act = min(silu(gate), limit)
         out = act * up

     `build_moe_ffn` (llama-graph.cpp:2146-2164) reads `_exp` for the
     ROUTED experts; `build_ffn` (:1751-1768) reads `_shexp` for the
     SHARED experts and for the leading DENSE layers, because
     `build_ffn` is both. `conversion/step3.py:207-220` writes both at
     `block_count` length from `swiglu_limits` / `swiglu_limits_shared`
     (`None` entries and MTP layers as 0.0, i.e. no clamp).

  2. **A half-width rotary on the FULL-attention layers.**
     `step35.cpp:9` is `n_rot_full = n_rot_full / 2`, run AFTER
     llama-model.cpp:1222 seeded `n_rot_swa` from the unhalved value,
     so `n_rot(il)` (llama-hparams.cpp:85-91, `is_swa(il) ? n_rot_swa :
     n_rot_full`) is the whole head on sliding layers and half of it on
     full ones. No GGUF key says so; the converter only ASSERTS the
     checkpoint's `partial_rotary_factors` are 1.0 / 0.5 (step3.py:170).

Everything else in the graph is a seam that already landed and is
carried by this fixture rather than assumed:

  * per-layer head counts (`head_count` / `head_count_kv` as arrays,
    `:76-78,208-209`; `crate::layer_shapes`),
  * the sliding-window BOOL ARRAY (`:26`; `crate::swa_layers`) with a
    window narrower than the prompt and its own RoPE base (`:23-24`),
  * the head-wise sigmoid attention gate (`:96,268-284`;
    `crate::attn_gate`), optional per layer,
  * per-head RMS QK-norm before RoPE (`:81-82,236-243`), optional,
  * SIGMOID routing by DEFAULT (`:19-21`, no key written),
    `exp_probs_b` (`:111`), `expert_weights_scale` / `_norm` (`:15-16`),
  * a shared expert on every MoE layer (`:114-116,329-338`), a dense
    leading layer decided by TENSOR PRESENCE (`:304`), not by the
    `leading_dense_block_count` key the converter also writes,
  * `1/sqrt(head_dim)` attention scale (`:262`), NEOX RoPE
    (llama-model.cpp:2680), a REQUIRED `output.weight` (`:61`).

The clamp values are chosen so the clamp BITES: `ffn_up` / `ffn_gate`
pre-activations are drawn wide enough that a limit of a few units
clips a visible fraction of them, and each array has a 0.0 entry (no
clamp on that layer) beside nonzero ones, so a reader that broadcast
one value or read the wrong array is visible.

`--no-clamp` writes the same weights with neither clamp key. `--mtp`
appends one NextN block inside `block_count` with
`nextn_predict_layers = 1` (step3.py:222-223), which llama.cpp skips
(`crate::mtp_blocks`); its clamp entries are 0.0 as the converter pads
them.

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_step35_fixture.py OUT.gguf [--no-clamp | --mtp]

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "step35"

N_LAYER = 4
N_EMBD = 24
HEAD_DIM = 8
# Full layers 0 and 2 (fewer heads, as the real checkpoint's
# attention_other_setting gives the sliding layers their own counts);
# sliding layers 1 and 3.
SWA_PATTERN = [False, True, False, True]
HEADS = [2, 4, 2, 4]
KV_HEADS = [1, 2, 1, 2]
N_FF = 40  # dense layer 0
N_FF_EXP = 16
N_FF_SHEXP = 24
N_EXPERT = 4
N_EXPERT_USED = 2
N_VOCAB = 48
CTX = 64
SWA_WINDOW = 3
ROPE_BASE = 10000.0
ROPE_BASE_SWA = 5000.0
RMS_EPS = 1e-5
EXPERT_WEIGHTS_SCALE = 2.5
# Routed-expert clamps: layer 0 is dense so its entry is unread by
# build_moe_ffn; layer 2 unclamped; 1 and 3 clamped at different limits.
CLAMP_EXP = [0.0, 1.5, 0.0, 2.5]
# Shared-expert AND dense clamps: layer 0's applies to the dense FFN,
# layer 3 unclamped.
CLAMP_SHEXP = [2.0, 3.0, 1.0, 0.0]


def main(out_path: str, variant: str) -> None:
    mtp = variant == "--mtp"
    n_layer_all = N_LAYER + (1 if mtp else 0)
    rng = np.random.default_rng(0x57E935)
    # A separate stream for the MTP block so the trunk's weights are
    # byte-identical with and without it.
    mtp_rng = np.random.default_rng(0x37)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name(f"ferrox-step35-fixture{variant.replace('--', '-') if variant else ''}")
    w.add_block_count(n_layer_all)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    swa = SWA_PATTERN + ([False] if mtp else [])
    heads = HEADS + ([2] if mtp else [])
    kv_heads = KV_HEADS + ([1] if mtp else [])
    w.add_head_count(heads)
    w.add_head_count_kv(kv_heads)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_freq_base_swa(ROPE_BASE_SWA)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_sliding_window(SWA_WINDOW)
    w.add_sliding_window_pattern(swa)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_expert_shared_feed_forward_length(N_FF_SHEXP)
    w.add_expert_weights_scale(EXPERT_WEIGHTS_SCALE)
    w.add_expert_weights_norm(True)
    # step3.py:196-197 writes these; step35.cpp reads neither.
    w.add_leading_dense_block_count(1)
    w.add_moe_every_n_layers(1)
    if variant != "--no-clamp":
        w.add_swiglu_clamp_exp(CLAMP_EXP + ([0.0] if mtp else []))
        w.add_swiglu_clamp_shexp(CLAMP_SHEXP + ([0.0] if mtp else []))
    if mtp:
        w.add_nextn_predict_layers(1)
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

    def block(il: int, r, dense: bool) -> None:
        p = f"blk.{il}."
        n_head_il = heads[il]
        n_embd_q = n_head_il * HEAD_DIM
        n_embd_kv = kv_heads[il] * HEAD_DIM

        def rr(*shape: int) -> np.ndarray:
            return (r.standard_normal(shape) * 0.25).astype(np.float32)

        w.add_tensor(p + "attn_norm.weight", rr(N_EMBD) + 1.0)
        w.add_tensor(p + "attn_q.weight", rr(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rr(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rr(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_q_norm.weight", rr(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rr(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_output.weight", rr(N_EMBD, n_embd_q))
        # Per head: ne = {n_embd, n_head_il} (step35.cpp:96).
        w.add_tensor(p + "attn_gate.weight", rr(n_head_il, N_EMBD) * 6.0)
        w.add_tensor(p + "ffn_norm.weight", rr(N_EMBD) + 1.0)
        # Wide FFN pre-activations so a clamp of a few units clips a
        # visible share of them.
        if dense:
            w.add_tensor(p + "ffn_gate.weight", rr(N_FF, N_EMBD) * 4.0)
            w.add_tensor(p + "ffn_up.weight", rr(N_FF, N_EMBD) * 4.0)
            w.add_tensor(p + "ffn_down.weight", rr(N_EMBD, N_FF))
            return
        w.add_tensor(p + "ffn_gate_inp.weight", rr(N_EXPERT, N_EMBD))
        w.add_tensor(
            p + "exp_probs_b.bias",
            (r.standard_normal(N_EXPERT) * 0.6).astype(np.float32),
        )
        w.add_tensor(p + "ffn_gate_exps.weight", rr(N_EXPERT, N_FF_EXP, N_EMBD) * 4.0)
        w.add_tensor(p + "ffn_up_exps.weight", rr(N_EXPERT, N_FF_EXP, N_EMBD) * 4.0)
        w.add_tensor(p + "ffn_down_exps.weight", rr(N_EXPERT, N_EMBD, N_FF_EXP))
        w.add_tensor(p + "ffn_gate_shexp.weight", rr(N_FF_SHEXP, N_EMBD) * 4.0)
        w.add_tensor(p + "ffn_up_shexp.weight", rr(N_FF_SHEXP, N_EMBD) * 4.0)
        w.add_tensor(p + "ffn_down_shexp.weight", rr(N_EMBD, N_FF_SHEXP))

    for il in range(N_LAYER):
        block(il, rng, dense=(il == 0))

    if mtp:
        il = N_LAYER
        block(il, mtp_rng, dense=False)
        p = f"blk.{il}."

        def mr(*shape: int) -> np.ndarray:
            return (mtp_rng.standard_normal(shape) * 0.25).astype(np.float32)

        w.add_tensor(p + "nextn.eh_proj.weight", mr(N_EMBD, 2 * N_EMBD))
        w.add_tensor(p + "nextn.enorm.weight", mr(N_EMBD) + 1.0)
        w.add_tensor(p + "nextn.hnorm.weight", mr(N_EMBD) + 1.0)
        w.add_tensor(p + "nextn.shared_head_norm.weight", mr(N_EMBD) + 1.0)
        w.add_tensor(p + "nextn.shared_head_head.weight", mr(N_VOCAB, N_EMBD))

    w.add_tensor("output_norm.weight", rnd(N_EMBD) + 1.0)
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    main(args[0] if args else "step35-fixture.gguf", flags[0] if flags else "")
