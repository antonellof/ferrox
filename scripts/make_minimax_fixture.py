#!/usr/bin/env python3
"""Generate the tiny synthetic `minimax-m2` GGUF behind frink's MiniMax
refusal test.

MiniMax-M2 used to be refused here for "256-expert sigmoid MoE + MTP".
Two of those clauses are false, and this fixture is what makes that
checkable without a 230B download.

**There is no MTP.** `.scratch/llama.cpp/src/models/minimax-m2.cpp`
creates no `nextn.*` tensor; `gguf-py/gguf/constants.py`'s
`MODEL_ARCH.MINIMAXM2` tensor list has no `NEXTN_*` entry, so the writer
cannot emit one; and `conversion/minimax.py` never mentions MTP (its M3
vision subclass says outright that "text / mtp / sparse-index are
dropped"). A refusal blaming MTP names something no MiniMax GGUF can
contain -- glm4moe's `q_lora_rank` defect in a different costume.

**The routing is nothing new.** `minimax-m2.cpp:131-141` is one
`build_moe_ffn` with `LLM_FFN_SILU`, `norm_w=true`, `exp_probs_b` and
whatever `expert_gating_func` says. frink reads all of that already.

So what this file pins is the shape of a *valid* minimax-m2 checkpoint,
transcribed from `minimax-m2.cpp` and `conversion/minimax.py`:

  * **Whole-vector Q/K norm.** `attn_q_norm` is `n_head * head_dim` wide
    and `attn_k_norm` is `n_head_kv * head_dim` (:30-31) -- OLMoE's
    style. NOT the `head_dim`-wide per-head norm Qwen3 and GLM-4.5 use.
    M3 differs from M2 here (`minimax-m3.cpp:53-55` is per-head), which
    is why the two cannot share one capability row.
  * **Partial NEOX RoPE.** `rope.dimension_count` is half `head_dim`;
    :51 records the real ratio, "head_dim = 128, n_rot = 64".
  * **No leading dense block and no shared expert.** Every layer is MoE
    (:23-40), unlike glm4moe/deepseek2/M3.
  * **No attention bias.** `create_tensor_qkv` marks q/k/v bias
    NOT_REQUIRED and MiniMax-M2 ships none.
  * **`expert_gating_func` is present and SIGMOID.** It has to be:
    `load_arch_hparams` reads it as optional, but the default
    `LLAMA_EXPERT_GATING_FUNC_TYPE_NONE` hits `GGML_ABORT` in
    `build_moe_ffn` (`llama-graph.cpp:2019`), so any file llama.cpp can
    actually run carries the key. `conversion/base.py:1291` writes it
    from HF's `scoring_func`.
  * **No `expert_weights_scale` and no `expert_weights_norm` key.**
    `conversion/minimax.py` writes both only for M3 (:70-71), and
    `minimax-m2.cpp` reads neither -- it hardcodes `norm_w=true` and
    leaves the scale at llama.cpp's 0.0f, which `llama-graph.cpp:2070`
    treats as "do not scale". frink's defaults land on the same
    behaviour, and `tests/minimax_refusal.rs` pins that they do.
  * **`n_ff == n_ff_exp`.** `minimax-m2.cpp` shapes its expert tensors
    with `n_ff`, not `n_ff_exp`, and the converter sets both from
    `intermediate_size`, so they are equal in every real M2 file.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_minimax_fixture.py OUT.gguf

No golden logits are checked in: frink has no minimax graph to compare
against, and admitting `minimax-m2` to `AUDITED_GENERIC_GQA` needs a
logit comparison against llama.cpp, not this file. See
`crates/frink-models/tests/minimax_refusal.rs`.
"""

import sys

import numpy as np

import gguf

ARCH = "minimax-m2"

N_LAYER = 2
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 16
# minimax-m2.cpp:51 -- "head_dim = 128, n_rot = 64". Half, as here.
ROPE_DIM = HEAD_DIM // 2
N_EXPERT = 6
N_EXPERT_USED = 2
# Equal on purpose: the graph shapes experts with n_ff and the converter
# sets both from `intermediate_size`.
N_FF = 16
N_FF_EXP = 16
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x6D2A2)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-minimax-m2-fixture")
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
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    # Deliberately absent, and the absences are the point:
    #   * every `nextn.*` tensor and `{arch}.nextn_predict_layers`
    #     -- MiniMax has no MTP in GGUF at all;
    #   * `{arch}.expert_weights_scale` / `.expert_weights_norm`
    #     -- M3-only in conversion/minimax.py;
    #   * `{arch}.leading_dense_block_count` / `.expert_shared_count`
    #     -- M2 has neither;
    #   * every MLA key -- M2 is plain GQA.

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
        # Whole-vector, not per-head: minimax-m2.cpp:30-31.
        w.add_tensor(p + "attn_q_norm.weight", rnd(n_embd_q))
        w.add_tensor(p + "attn_k_norm.weight", rnd(n_embd_kv))

        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))

        # Every layer is MoE. No leading dense, no shared expert.
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        w.add_tensor(
            p + "exp_probs_b.bias",
            (rng.standard_normal(N_EXPERT) * 0.6).astype(np.float32),
        )
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "minimax-m2-fixture.gguf")
