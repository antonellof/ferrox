#!/usr/bin/env python3
"""Generate the tiny synthetic `maincoder` GGUF used by frink's
MainCoder coverage test.

`maincoder` sat on frink's generic GQA path refusing as UNAUDITED,
triaged ONE MATCH ARM for one reason, and this fixture is what makes the
arm checkable: it norms Q and K **after** RoPE, not before.
`src/models/maincoder.cpp:78-90` calls `ggml_rope_ext` on Q and K and
only then `build_norm(Qcur, attn_q_norm, ...)` at :92 and the K norm at
:95. Every architecture frink had audited before this norms first, so
the fixture's QK-norm weights are deliberately far from 1.0 -- swapping
the two operations moves the logits by orders of magnitude more than the
comparison tolerance.

The rest of what it pins against `maincoder.cpp`:

  * NORM RoPE -- consecutive-pair rotation
    (`llama_model_rope_type`'s NORM group, llama-model.cpp:2608), not
    NEOX.
  * PER-HEAD QK norm: `attn_q_norm` / `attn_k_norm` are `{n_embd_head_k}`
    long (:33-34), not `n_head * head_dim`, so frink's loader has to
    resolve `QkNormStyle::PerHead` from the weight length.
  * `head_dim * n_head != n_embd`: Q is `{n_embd, n_embd_head_k * n_head}`
    and `attn_output` is `{n_embd_head_k * n_head, n_embd}` (:30-31), so
    neither is square. `rope_dimension_count == head_dim` because
    maincoder.cpp:51 asserts `n_embd_head == n_rot`.
  * GQA: `n_head_kv = 2 < n_head = 4`.
  * `1/sqrt(head_dim)` attention scale, written as a literal at :104 --
    this architecture reads no `f_attention_scale` key at all.
  * Dense SiLU SwiGLU on a sequential residual (:110,119-127), with
    `attn_norm` and `ffn_norm` both present and no post-norms.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_maincoder_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "maincoder"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# Deliberately NOT n_embd / n_head (which would be 6).
HEAD_DIM = 8
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x3A17C0)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-maincoder-fixture")
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
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "maincoder-fixture.gguf")
