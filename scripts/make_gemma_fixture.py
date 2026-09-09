#!/usr/bin/env python3
"""Generate the tiny synthetic `gemma` GGUF used by ferrox's Gemma-1
coverage test.

`gemma` is Gemma-1 (2B / 7B), the oldest row of the family. It sat on
ferrox's generic GQA path refusing as UNAUDITED, triaged FIXTURE-AWAY:
the three Gemma-specific pieces were all implemented for `GemmaFamily`
and only the evidence was missing.

`.scratch/llama.cpp/src/models/gemma.cpp` in full:

  * `load_arch_hparams` (:3-11) reads only the RMS epsilon and picks
    `LLM_TYPE_2B` / `LLM_TYPE_7B` off the layer count. Nothing branches
    on the type, so a 2-layer fixture is legitimate here -- unlike
    `baichuan`, where the layer count selects ALiBi.
  * `load_arch_tensors` (:13-34) creates `attn_norm`, split Q/K/V via
    `create_tensor_qkv`, an `attn_output` sized `{n_embd_head_k *
    n_head, n_embd}`, `ffn_norm` and `ffn_gate`/`ffn_up`/`ffn_down`.
    No biases, no QK-norm, no post-norms.
  * The lm_head is TIED: :20 duplicates `token_embd.weight` into
    `output` unconditionally, with no `TENSOR_NOT_REQUIRED` fallback.
    This fixture therefore ships NO `output.weight` -- one would be an
    unread tensor and llama.cpp would refuse the file outright.
  * The graph (:41-138) is the sequential residual, and the three
    Gemma-specific pieces are:
      - :49 `ggml_scale(inpL, sqrtf(n_embd))`, the embedding scale;
      - :112 `LLM_FFN_GELU, LLM_FFN_PAR`, i.e. GeGLU rather than SwiGLU;
      - :86 scales Q by `1/sqrtf(n_embd_head)` and :91 then passes
        `kq_scale = 1.0f`, which is exactly the `1/sqrt(head_dim)`
        ferrox's attention kernels apply when `attention_scale` is None.
  * RoPE is **NEOX** (`LLM_ARCH_GEMMA` in `llama_model_rope_type`'s NEOX
    group, llama-model.cpp:2642).
  * Gemma-1 declares no softcap and no sliding window, so the Gemma-2/3
    machinery is inert. The fixture carries neither key, and
    `fixture_away_graphs.rs` asserts that rather than assuming it.

Because `create_tensor_qkv` is sized from `n_embd_head_k * n_head`
(:27) rather than from `n_embd`, head_dim may disagree with
`n_embd / n_head`, and this fixture makes it disagree (8 against 6) --
real Gemma-1-2B does the same (n_embd 2048, n_head 8, head_dim 256).

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_gemma_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "gemma"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# Deliberately NOT n_embd / n_head (which would be 6). gemma.cpp:27-28
# sizes Q and `wo` from n_embd_head_k * n_head, so the two may disagree,
# and every real Gemma-1 checkpoint makes them disagree.
HEAD_DIM = 8
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x6E33A)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-gemma-fixture")
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

    # NOTE: no `{arch}.attention.sliding_window`, no softcap keys.
    # Gemma-1 reads none of them (gemma.cpp:3-11 is the whole hparams
    # body), so the Gemma-2/3 machinery must resolve to inert here.

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

    # ne = [n_embd, n_vocab] -> numpy [n_vocab, n_embd].
    #
    # This is also the lm_head: gemma.cpp:20 duplicates it into `output`
    # unconditionally, so there is deliberately no `output.weight` below.
    # Because the head is tied, the sqrt(n_embd) embedding scale at :49
    # is NOT cancelled by anything downstream -- it changes the logits.
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
        # a margin of ~1e-3 rather than the ~1e-1 it should have. Real
        # checkpoints have peaked attention; a fixture that does not
        # cannot see a positional bug.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        # GeGLU, not SwiGLU (gemma.cpp:112, LLM_FFN_GELU + LLM_FFN_PAR).
        # The gate is drawn wide so the two activations are far apart
        # over the range this fixture visits: gelu and silu agree to
        # within a few percent near zero and a fixture that only ever
        # saw small pre-activations could not tell them apart.
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD) * 4.0)
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    # NOTE: no `output.weight`. See the module docstring.

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gemma-fixture.gguf")
