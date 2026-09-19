#!/usr/bin/env python3
"""Generate the tiny synthetic `bailingmoe` GGUF used by frink's
Ling/BailingMoE coverage test.

`bailingmoe` is what inclusionAI's Ling models tag. It sat on frink's
generic GQA path refusing as UNAUDITED, triaged ONE MATCH ARM for one
reason: llama.cpp READS `{arch}.leading_dense_block_count` and then never
branches on it. `src/models/bailingmoe.cpp:5` reads the key; :39-54
creates `ffn_gate_inp`, the expert tensors and the shared-expert tensors
unconditionally for every layer, with no `if (i < n_layer_dense_lead)`
anywhere in the file, and the graph (:119-152) has no dense path either.
`conversion/bailingmoe.py:27` writes `first_k_dense_replace` verbatim, so
real checkpoints carry a nonzero value that means nothing.

**This fixture therefore sets `leading_dense_block_count = 1` and ships
NO dense FFN tensors on layer 0.** That is the whole point: a decoder
that honours the key looks for `blk.0.ffn_gate.weight` and fails on a
missing tensor. Regenerating the file with the key removed would make
the test pass for the wrong reason.

Everything else it pins against `bailingmoe.cpp`:

  * NORM RoPE -- consecutive-pair rotation
    (`llama_model_rope_type`'s NORM group, llama-model.cpp:2598), not
    NEOX. Getting this wrong is the Llama-3.1-8B wrong-logits bug.
  * A shared expert on every layer, added to the routed output before
    the residual (:139-151), sized `n_ff_exp * n_expert_shared`.
  * SOFTMAX gating, hardcoded at :133; no converter writes
    `expert_gating_func` for this architecture, so frink's softmax
    default has to be the right one.
  * `expert_weights_norm` read from METADATA (:9, and
    `conversion/bailingmoe.py:31` writes it). This file sets it
    **false**, which is the opposite of frink's architecture-name
    default, so a decoder ignoring the key renormalises the top-k
    weights and diverges.
  * `expert_weights_scale` read from metadata (:8), set here to a value
    that is neither 0 nor 1.
  * `n_ff != n_ff_exp`, `n_head_kv < n_head`, and an explicit
    `head_dim != n_embd / n_head` with `rope_dimension_count == head_dim`
    (which is what `conversion/base.py:1302` and
    `conversion/bailingmoe.py:25` together guarantee for a real file --
    llama.cpp scales attention by `1/sqrt(n_rot)` here, not by
    `1/sqrt(n_embd_head_k)`, and the two agree only because of that).
  * `output.weight` is REQUIRED for this architecture (:28, flag 0), not
    tied to the embedding.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_bailingmoe_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "bailingmoe"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# Deliberately NOT n_embd / n_head (which would be 6). `n_rot` is set to
# the same value below, which is what the converter guarantees.
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
# Nonzero, and deliberately a lie: every layer in this file is MoE.
LEADING_DENSE = 1
EXPERT_WEIGHTS_SCALE = 2.5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0xBA111)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-bailingmoe-fixture")
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
    w.add_expert_weights_scale(EXPERT_WEIGHTS_SCALE)
    # False, against frink's architecture-name default of true.
    w.add_expert_weights_norm(False)
    # The inert key. Every layer below is MoE anyway.
    w.add_leading_dense_block_count(LEADING_DENSE)
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

        # NOTE: no `ffn_gate/up/down` on ANY layer, including layer 0,
        # even though `leading_dense_block_count` is 1.
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
    main(sys.argv[1] if len(sys.argv) > 1 else "bailingmoe-fixture.gguf")
