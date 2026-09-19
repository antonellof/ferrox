#!/usr/bin/env python3
"""Generate the tiny synthetic `internlm2` GGUF used by frink's
InternLM2 coverage test.

`internlm2` is Shanghai AI Lab's InternLM2-7B/20B. It sat on frink's
generic GQA path refusing as UNAUDITED, triaged FIXTURE-AWAY: the graph
is plain llama, so the only thing missing was evidence.

`.scratch/llama.cpp/src/models/internlm2.cpp` in full:

  * `load_arch_hparams` (:3-11) reads NOTHING but the RMS epsilon.
  * `load_arch_tensors` (:13-35) creates `attn_norm`, split Q/K/V via
    `create_tensor_qkv`, `attn_output`, `ffn_norm` and `ffn_gate`/
    `ffn_up`/`ffn_down`. No QK-norm, no post-norms, no sinks.
  * The graph (:59-122) is the sequential residual with
    `LLM_FFN_SILU, LLM_FFN_PAR` SwiGLU (:107-112) and a
    `1/sqrt(n_embd_head)` attention scale (:92).
  * RoPE is **NORM** (`LLM_ARCH_INTERNLM2` in `llama_model_rope_type`'s
    NORM group, llama-model.cpp:2579). That is the one fact a reader
    could plausibly get wrong, so the test sabotages exactly it.

What the fixture carries beyond the minimum:

  * The OPTIONAL per-projection biases `create_tensor_qkv` allows
    (llama-model.cpp:2897-2899, `TENSOR_NOT_REQUIRED`), which real
    InternLM2 checkpoints ship because the HF model sets `bias=True` on
    `wqkv`. frink loads them generically (`loader.rs`'s
    `q_bias`/`k_bias`/`v_bias`), and they are non-zero here so a decoder
    that dropped them would be visible.
  * GQA: `n_head_kv = 2 < n_head = 4`.
  * `output.weight` REQUIRED (:20, flag 0), not tied to the embedding.

The graph asserts `n_embd_head == n_rot` (:45) and `create_tensor_qkv` is
called with `n_embd` as the Q width (:27) with `wo` at `{n_embd, n_embd}`
(:29), so `n_head * head_dim == n_embd` here.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_internlm2_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "internlm2"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# internlm2.cpp:27 passes n_embd as the Q width and :29 sizes wo as
# {n_embd, n_embd}, so this is forced to n_embd / n_head.
HEAD_DIM = 6
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x1273422)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-internlm2-fixture")
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
        # The optional QKV biases. Not required by llama.cpp, but real
        # InternLM2 exports carry them, and a decoder that ignored them
        # would pass a fixture that omitted them.
        w.add_tensor(p + "attn_q.bias", rnd(n_embd_q))
        w.add_tensor(p + "attn_k.bias", rnd(n_embd_kv))
        w.add_tensor(p + "attn_v.bias", rnd(n_embd_kv))
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
    main(sys.argv[1] if len(sys.argv) > 1 else "internlm2-fixture.gguf")
