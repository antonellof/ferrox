#!/usr/bin/env python3
"""Generate the tiny synthetic ALiBi GGUFs used by frink's ALiBi coverage
test: `refact`, `bloom`, `mpt`, `jais`, and `baichuan` at 40 layers.

Five graphs, one bias, three ways of arriving at it (`crate::alibi`):

  * `refact.cpp:12` and `bloom.cpp:18` hardcode `f_max_alibi_bias = 8.0`
    ("TODO: become GGUF KV parameter"); no key in the file.
  * `baichuan.cpp:11-14` hardcodes it ONLY when `n_layer == 40` (the
    13B); the 7B rotates. A 40-layer fixture at `n_embd 16` is what it
    takes to reach that arm.
  * `mpt.cpp:6` and `jais.cpp:5` read `attention.max_alibi_bias`,
    optional, default 0 (no position at all). The converters write it
    (`conversion/mpt.py:38`, `conversion/jais.py:104`).

None of the five graphs calls `ggml_rope`; `llama-model.cpp:1240-1242`
sets `use_alibi` from the bias and the KV mask then carries
`-|p_key - p_query|` (`llama-kv-cache.cpp:1673-1676`), which
`ggml_soft_max_ext` multiplies by the per-head slope (`ggml-cpu/ops.cpp:
5489-5508`). The rest of each graph:

  * `refact`: RMSNorm, split Q/K/V, SwiGLU, multi-query (`head_count_kv
    1`, `conversion/refact.py:41`), `output` optional.
  * `bloom`: the biased LayerNorm on the token embeddings
    (`token_embd_norm`, `:25-26,77-80`) and every site, a fused
    `attn_qkv` with bias, `attn_output` / `ffn_up` / `ffn_down` biases
    REQUIRED, the ungated GELU, `head_count_kv = head_count`.
  * `mpt`: the LayerNorm WITHOUT biases (all optional, `:34,43`; MPT
    has none), a fused `attn_qkv` with no bias, no projection biases,
    the ungated GELU, `attention.clamp_kqv` (`--clamp`), an optional
    `position_embd` (`--pos-embd`, some MPT fine-tunes).
  * `jais`: the biased LayerNorm, a fused `attn_qkv` with bias,
    `attn_output` / `ffn_gate` / `ffn_up` / `ffn_down` biases REQUIRED
    (`:35,41,44,47`), SwiGLU (`LLM_FFN_SILU, LLM_FFN_PAR`), `output`
    REQUIRED; the converter folds its embedding and width scales into
    the tensors.
  * `baichuan` (13B shape): the audited 7B's graph at 40 layers.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_alibi_fixture.py OUT.gguf --arch {refact,bloom,mpt,jais,baichuan}
            [--clamp] [--pos-embd] [--no-alibi-key]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

N_VOCAB = 48
CTX = 64


def main(out_path: str, arch: str, clamp: bool, pos_embd: bool, no_alibi_key: bool) -> None:
    seeds = {"refact": 0xA11B1, "bloom": 0xA11B2, "mpt": 0xA11B3, "jais": 0xA11B4, "baichuan": 0xA11B5}
    rng = np.random.default_rng(seeds[arch])
    if arch == "baichuan":
        n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff = 40, 16, 2, 2, 8, 32
    else:
        n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff = 3, 32, 4, 4, 8, 64
    if arch == "refact":
        n_head_kv = 1
    if arch == "mpt":
        n_head_kv = 2

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w(n: int) -> np.ndarray:
        return (1.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    def norm_b(n: int) -> np.ndarray:
        return (0.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    w = gguf.GGUFWriter(out_path, arch)
    w.add_name(f"frink-{arch}-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    if arch != "jais":
        w.add_head_count_kv(n_head_kv)
    if arch in ("refact", "baichuan"):
        w.add_layer_norm_rms_eps(1e-6)
    else:
        w.add_layer_norm_eps(1e-5)
    if arch == "baichuan":
        w.add_rope_freq_base(10000.0)
    if arch in ("mpt", "jais") and not no_alibi_key:
        w.add_max_alibi_bias(8.0)
    if arch == "mpt" and clamp:
        w.add_clamp_kqv(4.0)
    if arch == "jais":
        w.add_rope_dimension_count(head_dim)  # jais.py:21 writes it; nothing reads it
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

    w.add_tensor("token_embd.weight", rnd(N_VOCAB, n_embd))
    if arch == "bloom":
        w.add_tensor("token_embd_norm.weight", norm_w(n_embd))
        w.add_tensor("token_embd_norm.bias", norm_b(n_embd))
    if arch == "mpt" and pos_embd:
        w.add_tensor("position_embd.weight", rnd(CTX, n_embd))

    biased_ln = arch in ("bloom", "jais")
    rms = arch in ("refact", "baichuan")
    n_embd_q = n_head * head_dim
    n_embd_kv = n_head_kv * head_dim
    for il in range(n_layer):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", norm_w(n_embd) if not rms else (1.0 + rng.standard_normal(n_embd) * 0.3).astype(np.float32))
        if biased_ln:
            w.add_tensor(p + "attn_norm.bias", norm_b(n_embd))
        q = rnd(n_embd_q, n_embd) * 2.0
        k = rnd(n_embd_kv, n_embd) * 2.0
        v = rnd(n_embd_kv, n_embd)
        if arch in ("refact", "baichuan"):
            w.add_tensor(p + "attn_q.weight", q)
            w.add_tensor(p + "attn_k.weight", k)
            w.add_tensor(p + "attn_v.weight", v)
        else:
            w.add_tensor(p + "attn_qkv.weight", np.concatenate([q, k, v], axis=0))
            if arch != "mpt":
                w.add_tensor(p + "attn_qkv.bias", rnd(n_embd_q + 2 * n_embd_kv))
        w.add_tensor(p + "attn_output.weight", rnd(n_embd, n_embd_q))
        if arch in ("bloom", "jais"):
            w.add_tensor(p + "attn_output.bias", rnd(n_embd))
        w.add_tensor(p + "ffn_norm.weight", norm_w(n_embd) if not rms else (1.0 + rng.standard_normal(n_embd) * 0.3).astype(np.float32))
        if biased_ln:
            w.add_tensor(p + "ffn_norm.bias", norm_b(n_embd))
        if arch in ("refact", "baichuan", "jais"):
            w.add_tensor(p + "ffn_gate.weight", rnd(n_ff, n_embd))
            if arch == "jais":
                w.add_tensor(p + "ffn_gate.bias", rnd(n_ff))
        w.add_tensor(p + "ffn_up.weight", rnd(n_ff, n_embd))
        if arch in ("bloom", "jais"):
            w.add_tensor(p + "ffn_up.bias", rnd(n_ff))
        w.add_tensor(p + "ffn_down.weight", rnd(n_embd, n_ff) * 2.0)
        if arch in ("bloom", "jais"):
            w.add_tensor(p + "ffn_down.bias", rnd(n_embd))

    w.add_tensor("output_norm.weight", norm_w(n_embd) if not rms else (1.0 + rng.standard_normal(n_embd) * 0.3).astype(np.float32))
    if biased_ln:
        w.add_tensor("output_norm.bias", norm_b(n_embd))
    w.add_tensor("output.weight", rnd(N_VOCAB, n_embd))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="alibi-fixture.gguf")
    ap.add_argument("--arch", required=True, choices=["refact", "bloom", "mpt", "jais", "baichuan"])
    ap.add_argument("--clamp", action="store_true")
    ap.add_argument("--pos-embd", action="store_true")
    ap.add_argument("--no-alibi-key", action="store_true")
    args = ap.parse_args()
    main(args.out, args.arch, args.clamp, args.pos_embd, args.no_alibi_key)
