#!/usr/bin/env python3
"""Generate the tiny synthetic `exaone4` GGUF used by ferrox's EXAONE-4
coverage test.

`exaone4` is LG AI Research's EXAONE 4.0, and it is NOT the audited
`exaone` row (EXAONE 3.x, `src/models/exaone.cpp`, a plain pre-norm
llama). It refused as UNAUDITED, triaged NEW CODE, for the same blocker
as `olmo2`: `src/models/exaone4.cpp:60-67` is the complete per-layer norm
list and it creates `attn_post_norm`, `attn_q_norm`, `attn_k_norm` and
`ffn_post_norm` and **no `attn_norm` and no `ffn_norm`**. The graph reads
the raw residual at both sublayers -- `cur = inpL` before `build_qkv`
(:118, :124) and `build_ffn(ffn_inp, ...)` on the un-normed
post-attention residual (:159) -- and applies its two norms to each
branch's OUTPUT before the residual add (:152-155, :166-169).

    ffn_inp = x       + attn_post_norm(attn(x))
    out     = ffn_inp + ffn_post_norm(ffn(ffn_inp))

Identical to `olmo2.cpp:92,160-165,169,177-182`. The two rows share ONE
implementation in ferrox (`crates/ferrox-models/src/norm.rs`), and
this fixture and `olmo2_tiny.gguf` are what prove they are the same
graph rather than two that look alike.

What differs from `olmo2`, and why this row needs its own fixture:

  * **PER-HEAD QK-norm.** :61-62 sizes `attn_q_norm` and `attn_k_norm`
    `{n_embd_head_k}`, and :127-128 applies them to what `build_qkv` has
    ALREADY reshaped to 3-D (llama-graph.cpp:1656-1658), so the RMS is
    taken per head. `olmo2` norms the 2-D projection over its whole
    width. Same topology, opposite QK-norm style -- and ferrox picks
    between the two off the weight LENGTH, so this fixture is what
    checks the length it derives from a real per-head file.
  * Applied BEFORE RoPE (:127-128 precede :132-138).
  * `1.0f / sqrtf(float(n_embd_head))` at :145, so
    `ModelConfig::attention_scale` stays `None`.
  * `LLM_FFN_SILU, LLM_FFN_PAR` (:163): plain SwiGLU with `ffn_gate`.
  * NEOX RoPE (`LLM_ARCH_EXAONE4` is in `llama_model_rope_type`'s NEOX
    group, llama-model.cpp).

**THIRTY LAYERS ARE NOT AN ACCIDENT, and neither is "not 64".**
:4-14 switches the whole SWA machinery on off the LAYER COUNT with no
GGUF key involved: `if (hparams.n_layer() == 64)` sets
`swa_type = LLAMA_SWA_TYPE_STANDARD`, and :116 then makes RoPE
conditional -- `use_rope = hparams.is_swa(il) || swa_type == NONE`, i.e.
in a 64-layer EXAONE-4 the FULL-ATTENTION layers get **no rotation at
all**. That is the same NoPE class as `exaone-moe` and `smollm3`, ferrox
has no expression for it, and no metadata gate could see it. So the
32B is refused by name on `block_count == 64` in `loader.rs` -- the
`baichuan` precedent, where llama.cpp likewise picks a different graph
off the layer count -- and this file is deliberately on the other side
of that line. A 64-layer fixture would be refused before it was read.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_exaone4_fixture.py OUT.gguf

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "exaone4"

# NOT 64. See the module docstring: 64 is EXAONE-4 32B, where llama.cpp
# switches SWA on and stops rotating the full-attention layers.
N_LAYER = 2
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
# exaone4.cpp:54 sizes `wo` `{n_embd, n_embd}` and :92 asserts
# `n_embd_head_k == n_rot`, so head_dim * n_head must be n_embd.
HEAD_DIM = N_EMBD // N_HEAD  # 8
N_FF = 48
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x0E4A04)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic values in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-exaone4-fixture")
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

        # NO `attn_norm` and NO `ffn_norm`. See the module docstring.
        # Projections drawn wider than the rest of the file so the
        # softmax over a six-token prompt is not near-uniform; a
        # near-uniform attention cannot see a RoPE sabotage.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        # PER-HEAD: head_dim wide, unlike olmo2's whole-vector norms.
        # Centred near 1.5 so applying them whole-vector, or after RoPE,
        # is far from the truth.
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

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
    main(sys.argv[1] if len(sys.argv) > 1 else "exaone4-fixture.gguf")
