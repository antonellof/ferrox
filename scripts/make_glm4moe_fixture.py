#!/usr/bin/env python3
"""Generate the tiny synthetic `glm4moe` GGUFs used by ferrox's GLM-4.5-MoE
coverage test.

`glm4moe` is what GLM-4.5, GLM-4.5-Air and GLM-4.6 tag. ferrox refused
it until 2026-09-12, and the refusal had two lives: first "use
`ferrox_models::glm52_decoder` / `glm52_gguf_loader`" -- and that loader
cannot read a `glm4moe` file at all: `read_glm52_hparams` requires
`{arch}.attention.q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim` and
`qk_rope_head_dim`, and **GLM-4.5 is not an MLA model**. This fixture is
the proof: a file llama.cpp itself loads and runs as `glm4moe`, carrying
none of those four keys, because `src/models/glm4-moe.cpp`'s
`load_arch_hparams` never asks for them and its `load_arch_tensors` calls
`create_tensor_qkv` (plain Q/K/V) rather than creating any `attn_kv_a_mqa`
/ `attn_kv_b` / `attn_q_a` / `attn_q_b`.

The fixture is deliberately small (2 layers: one leading dense, one MoE;
32-wide, 6 experts) and carries the structure of `glm4-moe.cpp` that
decides where it can and cannot run:

  * `blk.N.post_attention_norm.weight` and **no** `blk.N.ffn_norm.weight`
    (glm4-moe.cpp:75). llama.cpp norms `ffn_inp` -- the post-residual
    sum -- with it (:215), i.e. it is the *pre-FFN* norm, gpt-oss's slot
    and not Gemma's. That was the second life of the refusal, and it is
    one row in `norm_sites::PRE_FFN_NORM_IS_POST_ATTENTION_NORM` now.
  * per-head Q/K RMSNorm of length `head_dim` (:69,71, optional there:
    the 355B variant has them, GLM-4.5-Air does not). `--no-qk-norm`
    writes the Air shape.
  * required Q/K/V biases via `create_tensor_qkv`, and no output bias.
  * a leading dense block, a shared expert, `exp_probs_b.bias`, sigmoid
    gating and `expert_weights_scale` -- the DeepSeek-V3-shaped routing
    ferrox already validates on `dots1`.
  * partial RoPE: `rope.dimension_count` is half `head_dim`, GLM's
    `partial_rotary_factor = 0.5`.

`--mrope` writes `rope.dimension_sections = [2, 1, 1, 0]`, what a
GLM-4.5V text tower carries (`conversion/glm.py` writes the sections for
the multimodal exports); `glm4-moe.cpp:6,145,188` then rotate with
`ggml_rope_multi` in `LLAMA_ROPE_TYPE_MROPE` (llama-model.cpp:2700),
which ferrox does not implement and REFUSES by name. libllama runs the
file with text positions (measured), so the variant pins a refusal, not
a golden.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_glm4moe_fixture.py OUT.gguf [--no-qk-norm] [--mrope]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "glm4moe"

N_LAYER = 2
N_DENSE_LEAD = 1
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 16
# GLM's partial_rotary_factor = 0.5: only the first half of each head is
# rotated.
ROPE_DIM = HEAD_DIM // 2
N_EXPERT = 6
N_EXPERT_USED = 2
N_EXPERT_SHARED = 1
N_FF = 40
N_FF_EXP = 16
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5
EXPERT_WEIGHTS_SCALE = 2.5


def main(out_path: str, qk_norm: bool, mrope: bool) -> None:
    rng = np.random.default_rng(0x64114)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-glm4moe-fixture")
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
    w.add_rope_dimension_count(ROPE_DIM)
    if mrope:
        # A GLM-4.5V text tower: three sections over the ROPE_DIM/2
        # bands (2 + 1 + 1 = 4 = ROPE_DIM / 2), the fourth unused.
        w.add_rope_dimension_sections([2, 1, 1, 0])
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_shared_count(N_EXPERT_SHARED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_leading_dense_block_count(N_DENSE_LEAD)
    w.add_expert_weights_scale(EXPERT_WEIGHTS_SCALE)
    w.add_expert_weights_norm(True)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    # Deliberately absent, and that absence is the point of the fixture:
    # `{arch}.attention.q_lora_rank`, `.kv_lora_rank`,
    # `.qk_nope_head_dim`, `.qk_rope_head_dim`. GLM-4.5 is plain GQA.

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

        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_q.bias", rnd(n_embd_q))
        w.add_tensor(p + "attn_k.bias", rnd(n_embd_kv))
        w.add_tensor(p + "attn_v.bias", rnd(n_embd_kv))
        if qk_norm:
            w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM))
            w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM))

        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        # The pre-FFN norm, spelled `post_attention_norm`. There is no
        # `ffn_norm` in a glm4moe checkpoint.
        w.add_tensor(p + "post_attention_norm.weight", rnd(N_EMBD))

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
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="glm4moe-fixture.gguf")
    ap.add_argument("--no-qk-norm", action="store_true")
    ap.add_argument("--mrope", action="store_true")
    args = ap.parse_args()
    main(args.out, not args.no_qk_norm, args.mrope)
