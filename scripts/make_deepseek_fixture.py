#!/usr/bin/env python3
"""Generate the tiny synthetic `deepseek` (V1) GGUF used by frink's
DeepSeek-V1 coverage test.

Not to be confused with `make_deepseek2_fixture.py`: `deepseek` is
DeepSeek-MoE-16B / DeepSeek-Coder-V1, a plain GQA MoE, while `deepseek2`
is the MLA family and runs on a dedicated stack.

`deepseek` sat on frink's generic GQA path refusing as UNAUDITED,
triaged ONE MATCH ARM for its routing renormalisation, and this fixture
is what makes the arm checkable:

  * `src/models/deepseek.cpp:145-155` passes `norm_w = false` to
    `build_moe_ffn` as a literal -- the selected experts' softmax weights
    are NOT renormalised -- and `conversion/deepseek.py`'s
    `DeepseekModel` (:124-217) never writes
    `{arch}.expert_weights_norm`; only `DeepseekV2Model` does (:354).
    **So this file deliberately carries no `expert_weights_norm` key**,
    exactly like a real DeepSeek-V1 GGUF, and frink has to get the
    answer from `NO_TOPK_RENORMALIZE_ARCHITECTURES`. Adding the key here
    would make the test pass without exercising the arm at all.
  * Leading dense layers ARE honoured on this architecture
    (deepseek.cpp:43 -- unlike `bailingmoe`, where the same key is
    inert), so the file has 3 layers with
    `leading_dense_block_count = 1`: layer 0 ships `ffn_gate/up/down`
    and no experts, layers 1 and 2 ship experts and no dense FFN.
  * A shared expert on the MoE layers (:63-65), sized
    `n_ff_exp * n_expert_shared`, added to the routed output before the
    residual (:160-168).
  * SOFTMAX gating, hardcoded at :154.
  * NORM RoPE -- consecutive-pair rotation
    (`llama_model_rope_type`'s NORM group, llama-model.cpp:2587).
  * `n_ff != n_ff_exp` and `n_head_kv < n_head`. Q/K/V are sized from
    `n_embd` and `n_embd_gqa` here (deepseek.cpp:38), so unlike the
    `seed_oss` and `bailingmoe` fixtures this architecture REQUIRES
    `n_head * head_dim == n_embd`; `head_dim` is still written
    explicitly, and `rope_dimension_count == head_dim` because
    deepseek.cpp:77 asserts `n_embd_head == n_rot`.
  * `expert_weights_scale = 1.0`, which is what the converter writes
    unconditionally (:144) and which `build_moe_ffn` treats as "no
    scaling".

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_deepseek_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "deepseek"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
# Must be n_embd / n_head on this architecture: deepseek.cpp:38 sizes Q
# as {n_embd, n_embd}, not {n_embd, n_head * head_dim}.
HEAD_DIM = 8
N_EXPERT = 6
N_EXPERT_USED = 2
N_EXPERT_SHARED = 2
# Deliberately different from N_FF_EXP.
N_FF = 40
N_FF_EXP = 12
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5
# Honoured on this architecture: layer 0 is dense, layers 1-2 are MoE.
LEADING_DENSE = 1


def main(out_path: str) -> None:
    rng = np.random.default_rng(0xDEE951)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-deepseek-fixture")
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
    w.add_expert_shared_count(N_EXPERT_SHARED)
    w.add_expert_weights_scale(1.0)
    w.add_leading_dense_block_count(LEADING_DENSE)
    # NO `add_expert_weights_norm` -- see the module docstring. A real
    # DeepSeek-V1 GGUF has no such key either.
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # Minimal SPM-flavoured vocab: llama.cpp needs tokens/scores/types to
    # build a vocab at all, but the fixture is always driven by explicit
    # token ids, never by tokenizing text.
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

    # ne = [n_embd, n_vocab] -> numpy [n_vocab, n_embd]
    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM
    n_ff_shexp = N_FF_EXP * N_EXPERT_SHARED

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))

        if il < LEADING_DENSE:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
            continue

        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        # gate/up: ne = [n_embd, n_ff_exp, n_expert]
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        # down: ne = [n_ff_exp, n_embd, n_expert]
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))

        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(n_ff_shexp, N_EMBD))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(n_ff_shexp, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, n_ff_shexp))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "deepseek-fixture.gguf")
