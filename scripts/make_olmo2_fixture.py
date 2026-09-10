#!/usr/bin/env python3
"""Generate the tiny synthetic `olmo2` GGUF used by ferrox's OLMo-2
coverage test.

`olmo2` is AI2's OLMo-2. It sat on ferrox's generic GQA path refusing as
UNAUDITED, triaged NEW CODE, and the blocker was the residual topology:
`src/models/olmo2.cpp` creates **no `attn_norm` and no `ffn_norm` at
all** (:43-52 is the complete per-layer tensor list) and its graph reads
the raw residual at both sublayers -- `cur = inpL` before Q/K/V (:92) and
`build_ffn(ffn_inp, ...)` on the un-normed post-attention residual
(:169). The two norms it does have are applied to each branch's OUTPUT
before the residual add (:160-165, :177-182), which is exactly where
ferrox already applies `post_attn_norm` / `post_ffn_norm`.

So the layer is:

    ffn_inp = x       + post_attn_norm(attn(x))
    out     = ffn_inp + post_ffn_norm(ffn(ffn_inp))

`exaone4` is the same shape (`src/models/exaone4.cpp:60-67,118,152-169`)
and shares one implementation with this row; see
`crates/ferrox-models/src/norm.rs`.

What this fixture pins beyond that topology, each against the C:

  * **Whole-vector QK-norm.** :45-46 sizes `attn_q_norm` `{n_embd}` and
    `attn_k_norm` `{n_head_kv * n_embd_head}`, and :106-112 applies both
    to the 2-D projections BEFORE `ggml_reshape_3d` (:114-116), so the
    RMS is taken over the whole Q (and whole K) vector, not per head.
    That is ferrox's `QkNormStyle::WholeVector`, and it is the OPPOSITE
    of `exaone4` next door, whose norms are `{n_embd_head_k}` and land
    after `build_qkv` has already reshaped. The two rows share a
    topology and not a QK-norm style, which is why both fixtures exist.
  * **Before RoPE**, not after (:106-112 precede :124-146) -- the
    `maincoder` / `hunyuan-moe` ordering hazard, on the other side.
  * **NEOX RoPE** (`LLM_ARCH_OLMO2` is in `llama_model_rope_type`'s NEOX
    group, llama-model.cpp).
  * `1/sqrtf(float(n_embd_head))` attention scale, passed literally at
    :154, so `ModelConfig::attention_scale` must stay `None`.
  * `LLM_FFN_SILU, LLM_FFN_PAR` (:174): plain SwiGLU with a separate
    `ffn_gate`.
  * An UNTIED lm_head: :38 creates `output` as REQUIRED.

**NO SLIDING WINDOW, on purpose.** :6-18 only sets
`swa_type = LLAMA_SWA_TYPE_STANDARD` when `attention.sliding_window` is
present and non-zero, and the SWA branch of the graph (:120-134) ropes
with YaRN switched off -- `freq_scale = 1`, `ext_factor = 0`,
`attn_factor = 1` -- while the full-attention layers use the model's own
YaRN. ferrox carries one `attn_factor` for the whole model and cannot
express a per-layer one, so an OLMo-3-style `olmo2` checkpoint that has
BOTH a window and YaRN is refused by name in `loader.rs` rather than
roped at the wrong magnitude on half its layers. Writing the window into
this fixture would put that refusal in front of the fixture and test
nothing.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_olmo2_fixture.py OUT.gguf

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "olmo2"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# olmo2.cpp:32 derives `n_embd_head` as `n_embd / n_head` and :68-69
# asserts it equals both `n_embd_head_k()` and `n_rot`, and :44 sizes
# `wo` `{n_embd, n_embd}`. So head_dim is NOT free here the way it is
# for `ernie4_5` or `plamo3`: it must be n_embd / n_head.
HEAD_DIM = N_EMBD // N_HEAD  # 6
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x0102A2)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic values in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-olmo2-fixture")
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

        # NO `attn_norm` and NO `ffn_norm`. That absence is the whole
        # point of the row: a decoder that requires either one cannot
        # load this file, and a decoder that applies either one computes
        # a different model.
        #
        # Projections drawn WIDER than the rest of the file for the
        # reason plamo3's fixture gives: at one magnitude the attention
        # scores over a six-token prompt sit within a fraction of each
        # other and softmax comes out nearly uniform, which leaves the
        # RoPE-variant sabotage unable to see its own change.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        # WHOLE-VECTOR: n_embd wide for Q, n_head_kv * head_dim for K.
        # Centred near 1.5 rather than 1.0 so that applying them per
        # head, or on the wrong side of RoPE, is far from the truth.
        w.add_tensor(p + "attn_q_norm.weight", rnd(n_embd_q) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rnd(n_embd_kv) + 1.5)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        # The suffixed spelling: olmo2.cpp:47,52 use the THREE-argument
        # `tn` overload, so `LLM_TN` appends `.weight`
        # (llama-arch.cpp:898-910). `plamo3` is the one architecture
        # upstream that does not, and it is a different row.
        w.add_tensor(p + "post_attention_norm.weight", rnd(N_EMBD) + 1.0)

        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
        w.add_tensor(p + "post_ffw_norm.weight", rnd(N_EMBD) + 1.0)

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "olmo2-fixture.gguf")
