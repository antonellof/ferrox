#!/usr/bin/env python3
"""Generate the tiny synthetic `pangu-embedded` GGUF used by frink's
openPangu-Embedded coverage test
(`crates/frink-models/tests/pangu_embedded_graphs.rs`).

`pangu-embedded` is openPangu-Embedded-1B / 7B (Huawei), a DECODER LLM
(`PanguEmbeddedForCausalLM`; "Embedded" as in edge devices, not an
embedding model). frink had it filed under "embedding variant;
deferred" and in the embedding loader's not-yet list, which is what
this fixture corrects. `.scratch/llama.cpp/src/models/pangu-embed.cpp`
is `llama.cpp`'s graph with ONE difference: a REQUIRED
`attn_output.bias` (`:37`, flag 0). NEOX RoPE (llama-model.cpp:2675),
`n_rot == n_embd_head` asserted (`:59`), `create_tensor_qkv` (`:35`,
fused or split), SwiGLU (`:118-123`), `output` tied when absent
(`:22-27`), an optional llama-3 `rope_freqs` (`:46`).
`conversion/pangu.py:31-39` writes `rope.dimension_count` and, when the
config has no `head_dim`, `key_length` / `value_length`.

Variants:
  * default          split `attn_q/k/v`, tied output
  * `--fused-qkv`    the fused `attn_qkv.weight`
  * `--output`       a separate `output.weight`

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_pangu_fixture.py OUT.gguf [--fused-qkv] [--output]

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "pangu-embedded"

N_EMBD = 24
N_HEAD = 4
N_KV = 2
HEAD_DIM = N_EMBD // N_HEAD
N_FF = 40
N_VOCAB = 48
N_LAYER = 3
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str, fused_qkv: bool, separate_output: bool) -> None:
    rng = np.random.default_rng(0x9A6C)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-pangu-embedded-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_vocab_size(N_VOCAB)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

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

    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        q = rnd(N_HEAD * HEAD_DIM, N_EMBD) * 4.0
        k = rnd(N_KV * HEAD_DIM, N_EMBD) * 4.0
        v = rnd(N_KV * HEAD_DIM, N_EMBD)
        if fused_qkv:
            w.add_tensor(p + "attn_qkv.weight", np.concatenate([q, k, v], axis=0))
        else:
            w.add_tensor(p + "attn_q.weight", q)
            w.add_tensor(p + "attn_k.weight", k)
            w.add_tensor(p + "attn_v.weight", v)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
        # pangu-embed.cpp:37, REQUIRED. Drawn large so dropping it is
        # not a rounding error.
        w.add_tensor(p + "attn_output.bias", rnd(N_EMBD) * 4.0)
        w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
    if separate_output:
        w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} (fused_qkv={fused_qkv}, output={separate_output})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "pangu-embedded-fixture.gguf",
        "--fused-qkv" in sys.argv[1:],
        "--output" in sys.argv[1:],
    )
