#!/usr/bin/env python3
"""Generate the tiny synthetic `ernie4_5` GGUF used by frink's ERNIE 4.5
coverage test.

`ernie4_5` is Baidu's dense ERNIE-4.5 (the MoE sibling tags
`ernie4_5-moe` and is a different, still-refusing row: its layers
interleave dense and MoE on a step frink does not read). It sat on
frink's generic GQA path refusing as UNAUDITED, triaged FIXTURE-AWAY.

`.scratch/llama.cpp/src/models/ernie4-5.cpp`, dense branch:

  * `load_arch_hparams` (:3-21) reads the RMS epsilon and, only for the
    MoE arch, the expert keys. Its one extra read is
    `LLM_KV_ROPE_DIMENSION_SECTIONS` as OPTIONAL (:5), for PaddleOCR's
    M-RoPE; a dense text checkpoint carries no such key and this fixture
    carries none either.
  * `load_arch_tensors` (:36-69) creates `attn_norm`, split Q/K/V via
    `create_tensor_qkv` sized from `n_embd_head_k * n_head`, an
    `attn_output` of `{n_embd_head_k * n_head, n_embd}`, `ffn_norm` and
    `ffn_gate`/`ffn_up`/`ffn_down`.
  * The graph (:95-149) is the sequential residual, `LLM_FFN_SILU,
    LLM_FFN_PAR` SwiGLU (:135-139), `kq_scale = 1/sqrt(n_embd_head)`
    (:120).
  * RoPE is **NORM** (`LLM_ARCH_ERNIE4_5` in `llama_model_rope_type`'s
    NORM group, llama-model.cpp:2602). The test sabotages exactly that.

DELIBERATELY ABSENT: the OPTIONAL `attn_output.bias` at :45. frink has
no slot for an output-projection bias outside the gpt-oss path, and a
checkpoint carrying one is refused BY NAME by
`assert_every_tensor_consumed` rather than run unbiased -- which is the
correct behaviour and is why this fixture must not carry one. Shipping it
here would either fail the load or, worse, pass while the bias was
dropped.

Because `create_tensor_qkv` is sized from `n_embd_head_k * n_head` rather
than `n_embd`, this architecture CAN carry a head_dim that disagrees with
`n_embd / n_head`, and this fixture does: 8 rather than 6. A decoder
deriving head_dim from `n_embd / n_head` gets a differently-shaped Q.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_ernie4_5_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "ernie4_5"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# Deliberately NOT n_embd / n_head (which would be 6). ernie4-5.cpp:41-42
# sizes Q and `wo` from n_embd_head_k * n_head, so the two may disagree.
HEAD_DIM = 8
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0xE4E45)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-ernie4-5-fixture")
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
        # NOTE: no `attn_output.bias`. See the module docstring.
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
    main(sys.argv[1] if len(sys.argv) > 1 else "ernie4-5-fixture.gguf")
