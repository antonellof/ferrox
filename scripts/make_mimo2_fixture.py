#!/usr/bin/env python3
"""Generate the tiny synthetic `mimo2` GGUFs used by ferrox's split
K/V head-width coverage test.

`mimo2` (MiMo-V2-Flash, every real export) refused as UNAUDITED,
triaged NEW CODE, on ONE thing no other generic-path graph has and one
small thing beside it (`src/models/mimo2.cpp`):

  * **A V head width that differs from the K head width.**
    `conversion/mimo.py:154` writes `attention.value_length` from
    `v_head_dim` separately from the `attention.key_length` the base
    converter writes from `head_dim`, and MiMo-V2-Flash's config is
    `head_dim: 192, v_head_dim: 128`. `:47-48` size K and V per layer
    from the two widths, `:132-140,152-154` view Q/K at `n_embd_head_k`
    and V at `n_embd_head_v`, and `wo` is `{n_embd_head_v * n_head,
    n_embd}` (`:52`). Every KV cache, attention kernel and projection
    check in ferrox took ONE head width; the loader refused the file.
    `crates/ferrox-models/src/kv_head_dims.rs` is the seam.
  * **`attention.value_scale`** (`:14-17`): when the key is present
    and not 1.0, `:180-183` multiply the attention output AFTER `wo` by
    it. `mimo.py:163-165` writes it from `attention_value_scale`;
    MiMo-V2-Flash sets 0.707.

Everything else the row needs had a seam already: the per-layer
`head_count_kv` ARRAY (`crate::layer_shapes`), the per-layer
`sliding_window_pattern` ARRAY with `rope.freq_base_swa`
(`crate::swa_layers`), attention sinks by tensor (`AttnWeights::sinks`),
NextN blocks inside `block_count` (`crate::mtp_blocks`), sigmoid gating
with `exp_probs_b` and `expert_weights_scale`, and partial NEOX RoPE
(`rope.dimension_count` from `head_dim * partial_rotary_factor`,
mimo.py:158-159). This fixture carries ALL of them, so that the new
width is measured in the company it keeps on a real file rather than
alone. Every layer is MoE, as on MiMo-V2-Flash (its config has no
`first_k_dense_replace` and `mimo.py` writes no
`leading_dense_block_count`); `mimo2.cpp:60-62`'s dense branch is
reached by tensor presence and no real export reaches it.

Shape (three layers, so that a full-attention layer sits between two
sliding ones):

  * `key_length = 12`, `value_length = 8`, `rope.dimension_count = 8`
    (partial rotary over the 12-wide K head), `head_count = 4`,
    `head_count_kv = [2, 1, 2]` (the converter's per-layer array:
    `swa_num_key_value_heads` on sliding layers).
  * `sliding_window_pattern = [1, 0, 1]`, `sliding_window = 3` (narrower
    than the six-token prompt, so the mask bites on layers 0 and 2),
    `rope.freq_base_swa = 100` against a base of 10000.
  * MoE on every layer: 6 experts, 2 used, sigmoid gating (the
    `:227` literal; the key agrees), `exp_probs_b`, and an
    `expert_weights_scale = 2.5` key that `mimo2.cpp` READS NOWHERE:
    llama.cpp's golden is unscaled, which is the measurement that the
    key is dead metadata for this architecture (the loader's
    `EXPERT_WEIGHTS_SCALE_READERS`).
  * `attn_sinks` on every layer, `attention.value_scale = 0.707`.
  * (default) a FUSED `attn_qkv.weight` per layer, rows `n_head * 12 +
    n_kv * 12 + n_kv * 8` -- the shape the converter emits from HF's
    `qkv_proj` (`:127-140`). `--split` writes `attn_q` / `attn_k` /
    `attn_v` instead (`:142-155`, the other branch).
  * `--no-value-scale` omits the key (`:14-17`: no scale).

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_mimo2_fixture.py OUT.gguf [--split] [--no-value-scale]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "mimo2"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = [2, 1, 2]
SWA_PATTERN = [True, False, True]
HEAD_DIM_K = 12
HEAD_DIM_V = 8
ROPE_DIM = 8
N_FF_EXP = 16
N_EXPERT = 6
N_EXPERT_USED = 2
N_VOCAB = 48
CTX = 64
SWA_WINDOW = 3
ROPE_BASE = 10000.0
ROPE_BASE_SWA = 100.0
RMS_EPS = 1e-5
VALUE_SCALE = 0.707
EXPERT_WEIGHTS_SCALE = 2.5


def main(out_path: str, split: bool, value_scale: bool) -> None:
    rng = np.random.default_rng(0x3130)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-mimo2-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    # mimo.py writes feed_forward_length from `intermediate_size`; the
    # graph reads n_ff only for the dense branch no real export takes.
    w.add_feed_forward_length(N_FF_EXP)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM_K)
    w.add_value_length(HEAD_DIM_V)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_freq_base_swa(ROPE_BASE_SWA)
    w.add_rope_dimension_count(ROPE_DIM)
    w.add_sliding_window(SWA_WINDOW)
    w.add_sliding_window_pattern(SWA_PATTERN)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_weights_scale(EXPERT_WEIGHTS_SCALE)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_nextn_predict_layers(0)
    if value_scale:
        w.add_attn_value_scale(VALUE_SCALE)
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
        n_kv = N_HEAD_KV[il]
        n_q_rows = N_HEAD * HEAD_DIM_K
        n_k_rows = n_kv * HEAD_DIM_K
        n_v_rows = n_kv * HEAD_DIM_V
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))
        # Wide, so a six-token softmax is not near-uniform and the window
        # and the rotation are both visible.
        q = rnd(n_q_rows, N_EMBD) * 4.0
        k = rnd(n_k_rows, N_EMBD) * 4.0
        v = rnd(n_v_rows, N_EMBD) * 4.0
        if split:
            w.add_tensor(p + "attn_q.weight", q)
            w.add_tensor(p + "attn_k.weight", k)
            w.add_tensor(p + "attn_v.weight", v)
        else:
            # mimo2.cpp:127-140: Q rows, then K rows at the K width, then
            # V rows at the V width, one matrix.
            w.add_tensor(p + "attn_qkv.weight", np.concatenate([q, k, v], axis=0))
        # :52: wo reads n_head * n_embd_head_v.
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM_V) * 4.0)
        # :55, {n_head}; one logit per query head into every softmax.
        w.add_tensor(p + "attn_sinks.weight", rnd(N_HEAD) * 4.0)

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        # :65-70 MoE branch, on every layer.
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD) * 4.0)
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP) * 4.0)
        w.add_tensor(p + "exp_probs_b.bias", rnd(N_EXPERT) * 2.0)

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="mimo2-fixture.gguf")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--no-value-scale", action="store_true")
    args = ap.parse_args()
    main(args.out, args.split, not args.no_value_scale)
