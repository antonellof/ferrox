#!/usr/bin/env python3
"""Generate the tiny synthetic `baichuan` GGUF used by frink's
Baichuan-7B coverage test.

`baichuan` is ONE architecture string covering TWO different models, and
that is the whole reason this fixture has 32 layers rather than 2.

`.scratch/llama.cpp/src/models/baichuan.cpp:5-14` picks the variant from
the LAYER COUNT and nothing else -- 32 layers is `LLM_TYPE_7B`, 40 layers
is `LLM_TYPE_13B` and sets `f_max_alibi_bias = 8.0f`, with its own
comment "TODO: become GGUF KV parameter". The graph then branches on the
same enum: :58 builds positions only for 7B, and :77-95 applies
`ggml_rope_ext` only for 7B, so **a 13B or an unrecognised layer count
gets no rotation at all**. A 2-layer fixture would be `LLM_TYPE_UNKNOWN`,
would fall into the no-RoPE arm at :91, and would therefore be evidence
about a graph no real checkpoint runs. 32 layers is the smallest honest
size.

frink refuses the 13B by name at `loader.rs`'s `block_count == 40`
check, pinned by
`baichuan_13b_is_refused_because_it_uses_alibi_and_the_7b_is_not`, so the
unaudited refusal only ever reached a 32-layer file. For that file the
row was triaged FIXTURE-AWAY, and this is the evidence.

The rest of `baichuan.cpp` is plain llama:

  * `load_arch_hparams` (:3-15) reads the RMS epsilon and the ALiBi bias.
  * `load_arch_tensors` (:17-40) creates `attn_norm`, split Q/K/V via
    `create_tensor_qkv`, `attn_output` at `{n_embd, n_embd}`, `ffn_norm`
    and `ffn_gate`/`ffn_up`/`ffn_down`.
  * The graph (:64-137) is the sequential residual with
    `LLM_FFN_SILU, LLM_FFN_PAR` SwiGLU (:121-126) and a
    `1/sqrt(n_embd_head)` attention scale (:103).
  * RoPE is **NORM** (`LLM_ARCH_BAICHUAN` in `llama_model_rope_type`'s
    NORM group, llama-model.cpp:2577).

Dimensions are smaller than the other fixtures here because there are
sixteen times as many layers.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_baichuan_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "baichuan"

# 32, not 2: see the module docstring. baichuan.cpp:5-14 reads the
# variant off this number and a 2-layer file would get no RoPE.
N_LAYER = 32
N_EMBD = 16
N_HEAD = 4
N_HEAD_KV = 2
# baichuan.cpp:31 passes n_embd as the Q width and :32 sizes wo as
# {n_embd, n_embd}, so this is forced to n_embd / n_head.
HEAD_DIM = 4
N_FF = 24
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0xBA1C4)

    def rnd(*shape: int) -> np.ndarray:
        # Smaller than the two-layer fixtures': thirty-two residual adds
        # compound, and this keeps the logits in a range where a float32
        # comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.15).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-baichuan-fixture")
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

        # Q and K are drawn WIDER than everything else on purpose. At
        # the magnitude the rest of the file uses, the attention scores
        # over a six-token prompt sit within a fraction of each other,
        # softmax comes out nearly uniform, and the layer stops caring
        # where the tokens are -- which leaves the RoPE-variant sabotage
        # test in `tests/fixture_away_graphs.rs` with a margin of ~1e-3
        # rather than the ~1e-1 it should have. Real checkpoints have
        # peaked attention; a fixture that does not cannot see a
        # positional bug.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
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
    main(sys.argv[1] if len(sys.argv) > 1 else "baichuan-fixture.gguf")
