#!/usr/bin/env python3
"""Generate the tiny synthetic `hunyuan-moe` GGUF used by frink's
Hunyuan-A13B coverage test.

`hunyuan-moe` sat on frink's generic GQA path refusing as UNAUDITED,
triaged ONE MATCH ARM for one reason, and this fixture is what makes the
arm checkable: it norms Q and K **after** RoPE.
`src/models/hunyuan-moe.cpp:93` and `:104` call `ggml_rope_ext`, and only
then `:110` and `:115` call `build_norm` on K and Q. Every architecture
frink had audited before this norms first, so the fixture's QK-norm
weights are deliberately far from 1.0: swapping the two operations moves
the logits by orders of magnitude more than the comparison tolerance.

The rest of what it pins against `hunyuan-moe.cpp`:

  * NEOX RoPE (`llama_model_rope_type`'s NEOX group,
    llama-model.cpp:2661).
  * PER-HEAD QK norm, `{n_embd_head_k}` long (:35-36).
  * `head_dim * n_head != n_embd`: Q is
    `{n_embd, n_embd_head_k * n_head}` and `attn_output` is
    `{n_embd_head_k * n_head, n_embd}` (:33-34).
  * GQA: `n_head_kv = 2 < n_head = 4`.
  * A shared expert on EVERY layer -- there is no leading-dense branch in
    this file -- run on the same normed FFN input as the router and added
    to the routed output before the residual (:137-163). Its width comes
    from `{arch}.expert_shared_feed_forward_length` (:5, :28) and is
    deliberately different from the routed expert width here.
  * SOFTMAX gating with `norm_topk_prob = true`, both literals at
    :146-149.
  * `1/sqrt(head_dim)` attention scale, a literal at :74.

One llama.cpp quirk this file has to respect: `load_arch_hparams` reads
`expert_feed_forward_length` as a REQUIRED key (:5) and then
`load_arch_tensors` sizes the routed experts from `n_ff`
(`feed_forward_length`) anyway (:42-44). So a real `hunyuan-moe`
checkpoint has the two equal, and this fixture does too -- a file where
they differ does not load in llama.cpp either.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_hunyuan_moe_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "hunyuan-moe"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# Deliberately NOT n_embd / n_head (which would be 6).
HEAD_DIM = 8
N_EXPERT = 6
N_EXPERT_USED = 2
N_EXPERT_SHARED = 1
# Equal on purpose: hunyuan-moe.cpp sizes the routed experts from `n_ff`
# while requiring the `expert_feed_forward_length` key. See the docstring.
N_FF = 12
N_FF_EXP = 12
# The shared expert is a different width, from its own key.
N_FF_SHEXP = 20
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x40C0DE)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-hunyuan-moe-fixture")
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
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_expert_shared_feed_forward_length(N_FF_SHEXP)
    w.add_expert_shared_count(N_EXPERT_SHARED)
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
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        # Per-head RMSNorm weights, applied AFTER RoPE. Centred well away
        # from 1.0 so that norming on the wrong side of the rotation is a
        # large, obvious divergence rather than a rounding difference.
        w.add_tensor(p + "attn_q_norm.weight", (rnd(HEAD_DIM) + 1.5).astype(np.float32))
        w.add_tensor(p + "attn_k_norm.weight", (rnd(HEAD_DIM) + 1.5).astype(np.float32))

        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))

        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF))

        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, N_FF_SHEXP))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hunyuan-moe-fixture.gguf")
