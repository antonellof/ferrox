#!/usr/bin/env python3
"""Generate the tiny synthetic `bailingmoe2` GGUF used by frink's
Ling-2.0 / Ring coverage test.

`bailingmoe2` is what inclusionAI's Ling-2.0 models tag. It is NOT
`bailingmoe`, which is a separate row, NORM-RoPE, and was admitted
earlier for a different reason entirely (llama.cpp reads its
`leading_dense_block_count` and then never branches on it). This one sat
on frink's generic GQA path refusing as UNAUDITED, triaged FIXTURE-AWAY:
every piece it needs is implemented, and only the evidence was missing.

Everything this fixture pins against
`.scratch/llama.cpp/src/models/bailingmoe2.cpp`:

  * A **fused** `attn_qkv` (:49), sized `{n_embd, n_embd + 2*n_embd_gqa}`
    -- so `n_head * head_dim` must equal `n_embd` here -- which frink
    splits in `load_qkv_projections` by the same
    `n_embd_head * n_head` / `n_embd_head * n_head_kv` arithmetic
    `llm_graph_context::build_qkv` uses (llama-graph.cpp:1598-1622).
  * Per-head `attn_q_norm` / `attn_k_norm` of width `n_embd_head_k`
    (:52-53) applied **BEFORE** RoPE (:123-135). This is the ordering
    that `maincoder` and `hunyuan-moe` get the other way round, so the
    fixture's norm weights are centred near 1.5 rather than near 1.0 and
    the two orders are far apart.
  * NEOX RoPE (`LLM_ARCH_BAILINGMOE2` in `llama_model_rope_type`'s NEOX
    group, llama-model.cpp:2659), not the interleaved NORM pairing.
  * `leading_dense_block_count = 1` that is **honoured**: :57 really does
    branch on it, unlike `bailingmoe`. Layer 0 therefore ships a dense
    `ffn_gate`/`ffn_up`/`ffn_down` and no experts, and layers 1-2 ship
    experts and no dense FFN. A decoder that ignored the key would die on
    a missing tensor either way round.
  * **SIGMOID** gating, read from metadata (:11, REQUIRED). frink's
    architecture-name fallback (`SIGMOID_GATING_ARCHITECTURES`) defaults
    to softmax and does NOT list `bailingmoe2`, so the file's own key is
    the only thing that can get this right -- which is exactly why the
    key is in this file and why the test asserts what it resolved to.
  * `expert_weights_norm = true` and `expert_weights_scale = 2.5`, both
    read from metadata (:9-10).
  * `blk.N.exp_probs_b.bias` (:61), the DeepSeek-V3-lineage
    selection-only routing bias. Spelled `exp_probs_b`, not
    `ffn_exp_probs_b` (llama-arch.cpp's `LLM_TENSOR_FFN_EXP_PROBS_B`).
  * A shared expert on every MoE layer (:67-69), added to the routed
    output before the residual (:186). Its width is
    `n_ff_shexp * n_expert_shared` (:58) -- NOT `n_ff_shexp` -- and the
    fixture sets the two independently (8 and 2, giving 16) so a reader
    that dropped the multiply gets a shape error.
  * `n_ff != n_ff_exp` and `n_head_kv < n_head`.
  * A sequential residual (:149, :191) and `1/sqrt(n_embd_head)`
    attention scale (:141).

DELIBERATELY ABSENT: `nextn_predict_layers` and the NEXTN/MTP tensors
(:77-85). frink refuses a checkpoint carrying them by name through the
unread-tensor gate, which is correct and is not what this fixture is
evidence about.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_bailingmoe2_fixture.py OUT.gguf

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "bailingmoe2"

N_LAYER = 3
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# Forced: bailingmoe2.cpp:49 sizes the fused QKV as
# {n_embd, n_embd + 2*n_embd_gqa}, so the Q slice is n_embd wide.
HEAD_DIM = 6
N_EXPERT = 6
N_EXPERT_USED = 2
N_EXPERT_SHARED = 2
# Deliberately different from N_FF_EXP, and from N_FF_SHEXP.
N_FF = 40
N_FF_EXP = 12
N_FF_SHEXP = 8
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5
# Honoured, unlike `bailingmoe`'s: layer 0 below is dense.
LEADING_DENSE = 1
EXPERT_WEIGHTS_SCALE = 2.5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0xBA1112)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic values in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-bailingmoe2-fixture")
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
    w.add_expert_shared_feed_forward_length(N_FF_SHEXP)
    w.add_leading_dense_block_count(LEADING_DENSE)
    w.add_expert_weights_scale(EXPERT_WEIGHTS_SCALE)
    w.add_expert_weights_norm(True)
    # REQUIRED by bailingmoe2.cpp:11, and the only thing that can tell
    # frink not to use its softmax default.
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
    n_ff_shexp = N_FF_SHEXP * N_EXPERT_SHARED

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        # Fused QKV, in llama.cpp's Q-then-K-then-V row order. Drawn
        # WIDER than the rest of the file: at one magnitude the attention
        # scores over a six-token prompt sit within a fraction of each
        # other, softmax comes out nearly uniform, and the layer stops
        # caring where its tokens are -- which would leave the
        # QK-norm-order sabotage test unable to see the flip.
        w.add_tensor(p + "attn_qkv.weight", rnd(n_embd_q + 2 * n_embd_kv, N_EMBD) * 4.0)
        # Per-head, width n_embd_head_k, centred near 1.5 rather than
        # near 1.0 so that norming after RoPE instead of before is a
        # large divergence rather than a rounding one.
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))

        if il < LEADING_DENSE:
            # Dense layer: no router, no experts, no shared expert.
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
            continue

        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        # Selection-only bias. Spelled `exp_probs_b`, not
        # `ffn_exp_probs_b`.
        w.add_tensor(p + "exp_probs_b.bias", rnd(N_EXPERT))
        # gate/up: ne = [n_embd, n_ff_exp, n_expert]
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        # down: ne = [n_ff_exp, n_embd, n_expert]
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))

        # n_ff_shexp * n_expert_shared, per bailingmoe2.cpp:58.
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
    main(sys.argv[1] if len(sys.argv) > 1 else "bailingmoe2-fixture.gguf")
