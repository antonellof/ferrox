#!/usr/bin/env python3
"""Generate the tiny synthetic `deepseek2` GGUFs used by ferrox's MLA
hparam-contract test and the MLA engine's libllama-golden test.

`deepseek2` is what DeepSeek-V2, V2.5, V3 and R1 all tag, so it is the
architecture behind the largest open models people run. ferrox routes it
to `mla_gguf_loader` / `MlaEngine`, and this fixture exists to check the
one thing nobody had checked: that the loader asks for keys a real
checkpoint actually carries.

Every metadata key and every tensor name below is exactly what
llama.cpp's own converter emits, transcribed from
`.scratch/llama.cpp/conversion/deepseek.py::DeepseekV2Model.set_gguf_parameters`
(:324-356) and `modify_tensors` (:422-427), and read back by
`src/models/deepseek2.cpp`. In particular:

  * `attention.key_length` is `kv_lora_rank + qk_rope_head_dim`, and
    `attention.value_length` is `kv_lora_rank` (deepseek.py:333-334).
    Those are the *compressed* MQA widths, not per-head K/V dims.
  * the per-head dims live in `attention.key_length_mla`
    (`qk_nope + qk_rope`) and `attention.value_length_mla` (`v_head_dim`)
    (deepseek.py:334-335, llama-arch.cpp:253).
  * `rope.dimension_count` is `qk_rope_head_dim` (deepseek.py:356), and
    llama.cpp derives `qk_nope = key_length_mla - rope.dimension_count`
    (deepseek2.cpp:80-81). There is **no** `qk_nope_head_dim` or
    `qk_rope_head_dim` GGUF key: neither string appears in
    `llama-arch.cpp`'s `LLM_KV_NAMES` nor anywhere in `gguf-py`. They are
    HF `config.json` field names.
  * because `key_length_mla` / `value_length_mla` are present,
    `llama_hparams::is_mla()` is true and the checkpoint carries the
    **split** `blk.N.attn_k_b` / `blk.N.attn_v_b` (deepseek2.cpp:120-122),
    not the legacy combined `attn_kv_b`. The converter splits them at
    conversion time (deepseek.py:426-427).
  * `attention.head_count_kv` is **1**: `deepseek.py:307-308` sets
    `num_key_value_heads = 1` for every MLA export ("deepseek2 using MLA
    converts into MQA"), because the cache holds ONE latent per position.
    This script wrote `N_HEAD` until 2026-09-12, and that -- not a
    llama.cpp defect -- was the `ggml.c:3942` shape abort that kept the
    file from ever producing a golden.

`--legacy-kv-b` writes the OTHER form llama.cpp reads (`deepseek2.cpp:
118-123`, `is_mla()` false): no `_mla` keys, `attention.key_length` =
qk_nope + qk_rope and `attention.value_length` = v_head_dim as per-head
widths, `head_count_kv = N_HEAD`, and ONE combined `attn_kv_b` per layer
`{kv_lora_rank, n_head * (qk_nope + v)}` -- the pre-2025-03 converter's
output, and `plm`'s shape. The split `attn_k_b` / `attn_v_b` of the
default file are DERIVED from that same combined matrix exactly as
`deepseek.py:420-427` derives them (the k half transposed), so the two
files hold the same model and llama.cpp's absorbed and naive branches
should agree on them to float noise; that agreement is one of the things
the golden test measures.

No RoPE scaling is declared by default, which keeps llama.cpp's YaRN
`mscale` correction at 1.0 (`attn_factor_org * ...` with `freq_scale =
1` makes every `logf(1/freq_scale)` term vanish, deepseek2.cpp:444-448).

`--yarn MSCALE_ALL_DIM` writes what `conversion/base.py:1231-1235` and
`conversion/deepseek.py:363-368` write for a real DeepSeek: `rope.
scaling.type = yarn`, `factor = 4`, `original_context_length = 16` (the
served context is 64), and `yarn_log_multiplier = 0.1 * MSCALE_ALL_DIM`
-- `0.707` is DeepSeek-V2 / V2-Lite (`config.json`: `mscale ==
mscale_all_dim == 0.707`), `1.0` is DeepSeek-V3 / R1. No beta keys, as
the converters write none for these configs. What llama.cpp does with
them is `crates/ferrox-models/src/mla_yarn.rs`; this fixture is how it
is checked, because `deepseek2.cpp:34-37` divide the key by 0.1 and
`llama-context.cpp:202-215` special-case `LLM_ARCH_DEEPSEEK2`, and a
reading of either can be wrong in a way only libllama's logits show.

`--temperature` writes the same file with `attention.temperature_scale
= 0.5` and `attention.temperature_length = 2`, the two keys
`deepseek2.cpp:46-47` read for Mistral-Large-3's per-position attention
temperature (`conversion/mistral.py:110,177`). llama.cpp's loader
accepts it and reads both keys (measured, `llama_model_loader: - kv
27/28`) and runs the graph; ferrox's MLA engine has no per-position Q
scale and REFUSES the file by name (`crate::attn_temperature`), where it
used to load and drop the key, so no golden is checked in for it.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_deepseek2_fixture.py OUT.gguf [--temperature] [--legacy-kv-b] [--yarn MSCALE_ALL_DIM]

The golden values that go with the default and `--legacy-kv-b` files
are produced by llama.cpp itself (`scripts/gptoss_reference_logits.cpp`
against a real `libllama`), not by this script; see
`crates/ferrox-models/tests/deepseek2_graphs.rs`.
"""

import sys

import numpy as np

import gguf

ARCH = "deepseek2"

N_LAYER = 2
N_DENSE_LEAD = 1
N_EMBD = 32
N_HEAD = 4
Q_LORA_RANK = 16
KV_LORA_RANK = 12
QK_NOPE_HEAD_DIM = 8
QK_ROPE_HEAD_DIM = 4
V_HEAD_DIM = 8
N_EXPERT = 6
N_EXPERT_USED = 2
N_EXPERT_SHARED = 1
N_FF = 40
N_FF_EXP = 16
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-6
EXPERT_WEIGHTS_SCALE = 2.5

# llama.cpp: n_embd_head_k_mla = qk_nope + qk_rope.
K_MLA = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM


# The Mistral-Large-3 keys, for the `--temperature` variant. A floor
# of 2 steps twice inside llama.cpp's six-token reference prompt.
TEMP_SCALE = 0.5
TEMP_LENGTH = 2
YARN_FACTOR = 4.0
YARN_ORIG_CTX = 16


def main(
    out_path: str,
    temperature: bool = False,
    legacy_kv_b: bool = False,
    yarn_mscale_all_dim: float | None = None,
) -> None:
    rng = np.random.default_rng(0xD5002)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-deepseek2-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    # deepseek.py:307-308 for an MLA export; the pre-MLA converter wrote
    # num_key_value_heads, which DeepSeek's configs set equal to n_head.
    w.add_head_count_kv(N_HEAD if legacy_kv_b else 1)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    if yarn_mscale_all_dim is not None:
        # conversion/base.py:1231-1235 and deepseek.py:363-368.
        w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        w.add_rope_scaling_factor(YARN_FACTOR)
        w.add_rope_scaling_orig_ctx_len(YARN_ORIG_CTX)
        w.add_rope_scaling_yarn_log_mul(0.1 * yarn_mscale_all_dim)
    # deepseek.py:356 -- the ROPE half of the head, not the whole head.
    w.add_rope_dimension_count(QK_ROPE_HEAD_DIM)
    w.add_vocab_size(N_VOCAB)
    w.add_leading_dense_block_count(N_DENSE_LEAD)
    w.add_q_lora_rank(Q_LORA_RANK)
    w.add_kv_lora_rank(KV_LORA_RANK)
    if legacy_kv_b:
        # The pre-MLA converter: per-head widths under the plain keys,
        # no `_mla` keys, so `is_mla()` is false (llama-hparams.cpp:244).
        w.add_key_length(K_MLA)
        w.add_value_length(V_HEAD_DIM)
    else:
        # deepseek.py:333-334: the COMPRESSED widths.
        w.add_key_length(KV_LORA_RANK + QK_ROPE_HEAD_DIM)
        w.add_value_length(KV_LORA_RANK)
        # deepseek.py:334-335: the per-head widths.
        w.add_key_length_mla(K_MLA)
        w.add_value_length_mla(V_HEAD_DIM)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_shared_count(N_EXPERT_SHARED)
    w.add_expert_weights_scale(EXPERT_WEIGHTS_SCALE)
    w.add_expert_weights_norm(True)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    if temperature:
        # conversion/mistral.py:110 and :177, verbatim.
        w.add_attn_temperature_scale(TEMP_SCALE)
        w.add_attn_temperature_length(TEMP_LENGTH)

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
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        # Q down/up projection through the LoRA rank.
        w.add_tensor(p + "attn_q_a.weight", rnd(Q_LORA_RANK, N_EMBD))
        w.add_tensor(p + "attn_q_a_norm.weight", rnd(Q_LORA_RANK))
        w.add_tensor(p + "attn_q_b.weight", rnd(N_HEAD * K_MLA, Q_LORA_RANK))

        # Compressed KV + the RoPE'd shared key head.
        w.add_tensor(
            p + "attn_kv_a_mqa.weight",
            rnd(KV_LORA_RANK + QK_ROPE_HEAD_DIM, N_EMBD),
        )
        w.add_tensor(p + "attn_kv_a_norm.weight", rnd(KV_LORA_RANK))

        # ONE combined decompression, drawn once so both file forms hold
        # the same model: ne = [kv_lora, n_head * (qk_nope + v)] ->
        # numpy (n_head * (qk_nope + v), kv_lora), head-major.
        kv_b = rnd(N_HEAD * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK)
        if legacy_kv_b:
            w.add_tensor(p + "attn_kv_b.weight", kv_b)
        else:
            # deepseek.py:420-427: per head, the k half TRANSPOSED
            # (`ne = [qk_nope, kv_lora, n_head]`, numpy (n_head, kv_lora,
            # qk_nope)) and the v half as is (`ne = [kv_lora, v_head,
            # n_head]`, numpy (n_head, v_head, kv_lora)).
            per_head = kv_b.reshape(N_HEAD, QK_NOPE_HEAD_DIM + V_HEAD_DIM, KV_LORA_RANK)
            k_b = np.ascontiguousarray(per_head[:, :QK_NOPE_HEAD_DIM, :].transpose(0, 2, 1))
            v_b = np.ascontiguousarray(per_head[:, QK_NOPE_HEAD_DIM:, :])
            w.add_tensor(p + "attn_k_b.weight", k_b)
            w.add_tensor(p + "attn_v_b.weight", v_b)

        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * V_HEAD_DIM))
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))

        if il < N_DENSE_LEAD:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
            continue

        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        w.add_tensor(
            p + "exp_probs_b.bias",
            (rng.standard_normal(N_EXPERT) * 0.6).astype(np.float32),
        )
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))

        n_ff_sh = N_FF_EXP * N_EXPERT_SHARED
        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(n_ff_sh, N_EMBD))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(n_ff_sh, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, n_ff_sh))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    yarn = None
    if "--yarn" in args:
        yarn = float(args[args.index("--yarn") + 1])
    main(
        args[0] if args and not args[0].startswith("--") else "deepseek2-fixture.gguf",
        temperature="--temperature" in args,
        legacy_kv_b="--legacy-kv-b" in args,
        yarn_mscale_all_dim=yarn,
    )
