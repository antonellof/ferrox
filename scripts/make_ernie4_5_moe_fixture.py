#!/usr/bin/env python3
"""Generate the tiny synthetic `ernie4_5-moe` GGUFs used by ferrox's
ERNIE-4.5 MoE coverage test.

Two files, because this row has two things to prove:

    python3 scripts/make_ernie4_5_moe_fixture.py OUT.gguf
    python3 scripts/make_ernie4_5_moe_fixture.py OUT.gguf --step 2

The default (`--step 1`) is the fixture the golden logits come from: the
value both published ERNIE-4.5 MoE checkpoints carry, and the only value
llama.cpp can load. The `--step 2` file is the one ferrox REFUSES, and it
exists so that the refusal is demonstrably reachable rather than
decorative.

`.scratch/llama.cpp/src/models/ernie4-5.cpp` (loader) and
`ernie4-5-moe.cpp` (graph):

  * `load_arch_hparams` (ernie4-5.cpp:3-21) reads the RMS epsilon and,
    for the MoE arch only, `LLM_KV_EXPERT_FEED_FORWARD_LENGTH`,
    `LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH` (optional),
    `LLM_KV_INTERLEAVE_MOE_LAYER_STEP` (REQUIRED) and
    `LLM_KV_LEADING_DENSE_BLOCK_COUNT` (optional).
  * `load_arch_tensors` (:36-69) creates `attn_norm`, split Q/K/V sized
    from `n_embd_head_k * n_head`, `attn_output`, `ffn_norm`, and then
    branches on `i >= n_layer_dense_lead` ALONE (:49) -- expert tensors
    past the prefix, dense `ffn_gate`/`ffn_up`/`ffn_down` before it.
  * The graph (ernie4-5-moe.cpp:27-118) is the sequential residual with
    `kq_scale = 1/sqrt(n_embd_head)` (:52), and its FFN branch is
    chosen by `il >= n_layer_dense_lead && (il + 1) % n_moe_layer_step == 0`
    (:64) -- the modulo the loader does not have.
  * Routing (:81-91) is `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`,
    HARDCODED, with `norm_w = true` and an optional `ffn_exp_probs_b`
    selection bias. It is NOT sigmoid-routed, whatever
    `llama-cpp-gap-inventory.md` used to say.
  * RoPE is **NORM** (`LLM_ARCH_ERNIE4_5_MOE` in
    `llama_model_rope_type`'s NORM group, llama-model.cpp:2603).

THE FINDING, and why `--step 2` is a refusal fixture rather than a
second golden one: the loader's condition (:49) has no modulo and the
graph's (:64) does, so for any step above 1 llama.cpp creates expert
tensors for a layer whose graph then reads dense ones. The two agree
only where the modulo changes nothing. A converter-produced checkpoint
puts `blk.N.ffn_gate.weight` on the interleaved dense layers, which is
exactly what the `--step 2` file below does, and llama.cpp fails to load
it on the missing `blk.N.ffn_down_exps.weight`. See
`crates/ferrox-models/src/moe_interleave.rs`.

DELIBERATELY ABSENT from both files: `{arch}.expert_gating_func` and
`{arch}.expert_weights_norm`. llama.cpp hardcodes both for this
architecture, so ferrox has to reach SOFTMAX and renormalised top-k
through its architecture-name defaults, and a file carrying the keys
would test the file rather than the defaults. Also absent: the OPTIONAL
`attn_output.bias` (ernie4-5.cpp:45), which ferrox refuses by name
outside the gpt-oss path rather than dropping.

Weights are pseudo-random from a fixed seed so the files are
byte-stable.

The golden logits that go with the step-1 file are produced by llama.cpp
itself (see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "ernie4_5-moe"

N_LAYER = 3
# Layer 0 is dense, layers 1 and 2 are MoE. llama.cpp's loader gives
# layers 1 and 2 expert tensors on this alone (ernie4-5.cpp:49).
N_DENSE_LEAD = 1
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# Deliberately NOT n_embd / n_head (which would be 6). ernie4-5.cpp:41-42
# sizes Q and `wo` from n_embd_head_k * n_head. The graph does assert
# n_embd_head_k == n_embd_head_v == n_rot (ernie4-5-moe.cpp:11-12), so
# all three are 8 here.
HEAD_DIM = 8
N_FF = 40
N_FF_EXP = 12
# The shared expert is `{n_embd, n_ff_shexp}` (ernie4-5.cpp:60-62), sized
# straight from the key with no multiply by the shared count -- unlike
# `bailingmoe2`. Held equal to N_FF_EXP so a reader cannot mistake which
# of the two readings this fixture would catch; the shared-width question
# belongs to that other row.
N_FF_SHEXP = 12
N_EXPERT = 4
N_EXPERT_USED = 2
N_EXPERT_SHARED = 1
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str, step: int) -> None:
    rng = np.random.default_rng(0xE45E)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name(f"ferrox-ernie4-5-moe-fixture-step{step}")
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
    w.add_expert_shared_feed_forward_length(N_FF_SHEXP)
    w.add_leading_dense_block_count(N_DENSE_LEAD)
    # The key this whole row is about (ernie4-5.cpp:11, REQUIRED).
    w.add_interleave_moe_layer_step(step)
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

        # Q and K are drawn WIDER than everything else on purpose. At the
        # magnitude the rest of the file uses, the attention scores over
        # a six-token prompt sit within a fraction of each other, softmax
        # comes out nearly uniform, and the layer stops caring where the
        # tokens are -- which leaves the RoPE-variant sabotage test with
        # a margin of ~1e-3 rather than the ~1e-1 it should have.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        # NOTE: no `attn_output.bias`. See the module docstring.
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))

        # THE INTERLEAVE, written the way a converter would write it:
        # `conversion/ernie.py` passes the HF tensor names straight
        # through, so a layer the model builds dense ships
        # `blk.N.ffn_gate.weight` and no expert tensors. For step 1 this
        # is exactly llama.cpp's own loader condition (ernie4-5.cpp:49);
        # for step 2 layer 2 lands here and llama.cpp -- which creates
        # expert tensors for every layer past the prefix regardless of
        # the step -- cannot load the file at all.
        is_moe = il >= N_DENSE_LEAD and (il + 1) % step == 0
        if not is_moe:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
            continue

        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        # Large enough to reorder the top-k. These are SOFTMAX
        # probabilities in (0, 1) that sum to one over four experts, so a
        # bias spread of ~0.6 really does change which two experts win --
        # which is what makes the selection-only placement of the bias
        # (llama-graph.cpp:1983-1988) visible in the logits.
        w.add_tensor(
            p + "exp_probs_b.bias",
            (rng.standard_normal(N_EXPERT) * 0.6).astype(np.float32),
        )

        # gate/up: ne = [n_embd, n_ff_exp, n_expert]
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        # down: ne = [n_ff_exp, n_embd, n_expert]
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))

        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, N_FF_SHEXP))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    # `output` is TENSOR_NOT_REQUIRED with a tok_embd fallback
    # (ernie4-5.cpp:30-34); the fixture ships its own so the lm_head is
    # not the embedding table read twice.
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} (interleave_moe_layer_step = {step})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--step"]
    out = args[0] if args else "ernie4-5-moe-fixture.gguf"
    st = int(args[1]) if len(args) > 1 else 1
    main(out, st)
