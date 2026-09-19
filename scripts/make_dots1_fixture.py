#!/usr/bin/env python3
"""Generate the tiny synthetic `dots1` GGUF used by frink's MoE
routing-bias coverage test.

`dots1` is the smallest architecture on llama.cpp's *generic* MoE path
that carries `blk.N.exp_probs_b.bias` -- the DeepSeek-V3 aux-loss-free
selection bias. The same tensor appears on ernie4_5-moe, bailingmoe2,
exaone-moe, hunyuan-moe and afmoe, so one validated implementation covers
all of them; dots1 is picked here only because its graph is otherwise
plain GQA + shared expert, which frink already implements.

The fixture is deliberately small (2 layers, 32-wide, 6 experts) and
carries the features the bias interacts with:

  * `blk.1.exp_probs_b.bias`       -- selection-only routing bias
    (llama.cpp's `LLM_TENSOR_FFN_EXP_PROBS_B` is spelled `blk.%d.exp_probs_b`
    on disk, *not* `ffn_exp_probs_b` -- see llama-arch.cpp:416 and
    gguf-py/gguf/constants.py:1240)
  * a leading dense layer          -- `leading_dense_block_count = 1`
  * a shared expert                -- added to the routed output
  * `expert_weights_scale` != 1    -- combine weights are scaled after
                                      renormalisation
  * sigmoid gating                 -- `expert_gating_func = 2`

The bias values are deliberately large enough to *reorder* the top-k
against the unbiased scores; a fixture where the bias never changes the
selection would pass with the bias ignored.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_dots1_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "dots1"

N_LAYER = 2
N_EMBD = 32
N_HEAD = 4
HEAD_DIM = 8
# dots1 builds q, k and v all `n_embd_head_k * n_head` wide
# (`create_tensor_qkv` in src/models/dots1.cpp), i.e. plain MHA.
N_HEAD_KV = N_HEAD
N_EXPERT = 6
N_EXPERT_USED = 2
N_EXPERT_SHARED = 1
N_FF = 24
N_FF_EXP = 16
N_VOCAB = 48
N_DENSE_LEAD = 1
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5
EXPERT_WEIGHTS_SCALE = 2.5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0xD0751)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-dots1-fixture")
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
    w.add_expert_shared_count(N_EXPERT_SHARED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_leading_dense_block_count(N_DENSE_LEAD)
    w.add_expert_weights_scale(EXPERT_WEIGHTS_SCALE)
    w.add_expert_weights_norm(True)
    # LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
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

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM))
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM))

        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))

        if il < N_DENSE_LEAD:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
            continue

        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        # Large enough to reorder the top-k: sigmoid scores live in
        # (0, 1), so a bias spread of ~1 changes which experts win.
        w.add_tensor(
            p + "exp_probs_b.bias",
            (rng.standard_normal(N_EXPERT) * 0.6).astype(np.float32),
        )

        # gate/up: ne = [n_embd, n_ff_exp, n_expert]
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        # down: ne = [n_ff_exp, n_embd, n_expert]
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
    main(sys.argv[1] if len(sys.argv) > 1 else "dots1-fixture.gguf")
