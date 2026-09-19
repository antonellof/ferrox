#!/usr/bin/env python3
"""Generate the tiny synthetic `plamo2` GGUFs used by frink's PLaMo-2
coverage test (`crates/frink-models/tests/plamo2_graphs.rs`).

`plamo2` is PLaMo-2 (1B / 2B / 8B, Preferred Networks): a hybrid whose
layers with `head_count_kv 0` run PLaMo-2's own state-space block
(`plamo2.cpp:218-343`, `crate::plamo2_ssm`) and whose others run
attention with a fused `attn_qkv`, a per-head QK RMSNorm with a
DISTINCT weight row per head (`:92-93,163,166`) and NEOX RoPE; every
layer has a post-attention norm, a pre-FFN norm, a Phi-3 fused
`ffn_up` (`{n_embd, 2 n_ff}`, SwiGLU) and a post-FFN norm
(`:98-102,158-183`). `conversion/plamo.py:72-95` writes the head-count
arrays from `mamba_step` (every other layer, the last an attention
layer); `:106-111` write `ssm.time_step_rank` as the SSM HEAD count,
`ssm.inner_size` as `heads * head_dim`, `ssm.group_count` 0.

The SSM block's widths (`:36-39`): `d_conv`, `d_state`, `n_heads`,
`d_inner = n_heads * head_dim`, and `dt_dim = max(64, n_embd / 16)`, a
LITERAL of the graph. Tensors (`:68-79`): `ssm_in` `{n_embd, 2 d_inner}`
with each head's `[z | x]` interleaved, `ssm_conv1d` `{d_conv, d_inner}`
(no bias), `ssm_x` `{d_inner, dt_dim + 2 d_state}` in the order B, C,
dt, `ssm_dt` `{dt_dim, n_heads}` with a bias `{n_heads}`, `ssm_a` and
`ssm_d` `{n_heads}`, `ssm_out`, and the three norms `ssm_dt_norm`
`{dt_dim}`, `ssm_b_norm` / `ssm_c_norm` `{d_state}`.

NAMES. The norms and the post-norms are created with the two-argument
`tn(..., i)` overload -- `blk.N.ssm_dt_norm`, `blk.N.ssm_b_norm`,
`blk.N.ssm_c_norm`, `blk.N.post_attention_norm`, `blk.N.post_ffw_norm`,
NO `.weight` -- and the converter emits exactly that because its
mapping entries for them end in `.weight` and match the HF name
exactly (`gguf-py/gguf/tensor_mapping.py:368,434,838,854,860`). This
script writes what a real file has. `--suffixed-norms` writes the
`.weight` spelling instead, which llama.cpp REFUSES (it asks for the
bare name): it exists to show that, not as a golden.

Layers (four): ssm, attention, ssm, attention (`mamba_step 2` on four
layers: `i % 2 != 1`), attention with `head_count 4, head_count_kv 2`.

ROPE, and the two head-count spellings. The HF model rotates q and k
over `qk_dim` after the QK norm (`modeling_plamo.py`, `_rotary_pos_emb`),
and `plamo2.cpp:239,245` call `ggml_rope_ext` for it. But
`llama-model.cpp:1189-1201` seed `n_rot` from `n_head()` -- layer 0's
head count -- and set it to 0 when that is 0, and the current
converter writes `head_count` as an ARRAY with 0 on every SSM layer
(`conversion/plamo.py:87-88,95`), layer 0 included. So libllama runs
every current export with `n_rot = 0` and rotates NOTHING (measured:
`print_info: n_rot = 0`; the logits move by 1.3 against the same
weights rotated). `rope.dimension_count` does not restore it -- that
read is inside the same `n_head() > 0` branch. frink rotates, as the
model does. The default fixture is the converter's spelling (both
arrays); `--scalar-heads` writes `head_count 4` as a scalar with the
KV array alone marking the SSM layers, which is an equally valid
llama.cpp file (`plamo2.cpp:19` reads only `n_head_kv(i)`) whose
`n_rot` seeds to `head_dim`. That file's libllama golden is the
rotated graph on the real layer order, and it is what frink must
match on BOTH files.

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_plamo2_fixture.py OUT.gguf [--scalar-heads] [--suffixed-norms]

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "plamo2"

N_EMBD = 32
N_HEAD = 4
N_KV = 2
HEAD_DIM = 8
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-6

# The SSM block (`plamo2.cpp:36-39`).
D_CONV = 4
D_STATE = 8
SSM_HEADS = 4
SSM_HEAD_DIM = 6
D_INNER = SSM_HEADS * SSM_HEAD_DIM
DT_DIM = max(64, N_EMBD // 16)

# head_count_kv per layer; 0 is an SSM layer (`plamo2.cpp:19`).
KV_PER_LAYER = [0, N_KV, 0, N_KV]
HEADS_PER_LAYER = [0 if k == 0 else N_HEAD for k in KV_PER_LAYER]


def main(out_path: str, suffixed_norms: bool, scalar_heads: bool) -> None:
    kv_per_layer = KV_PER_LAYER
    heads_per_layer = N_HEAD if scalar_heads else [0 if k == 0 else N_HEAD for k in kv_per_layer]
    n_layer = len(kv_per_layer)
    rng = np.random.default_rng(0x9A7A02)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_name(base: str) -> str:
        return base + ".weight" if suffixed_norms else base

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-plamo2-fixture")
    w.add_vocab_size(N_VOCAB)
    w.add_head_count(heads_per_layer)
    w.add_head_count_kv(kv_per_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_block_count(n_layer)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_ssm_state_size(D_STATE)
    w.add_ssm_conv_kernel(D_CONV)
    w.add_ssm_time_step_rank(SSM_HEADS)
    w.add_ssm_inner_size(D_INNER)
    w.add_ssm_group_count(0)
    w.add_feed_forward_length(N_FF)
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

    for il, nkv in enumerate(kv_per_layer):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if nkv == 0:
            # plamo2.cpp:68-79. Shapes are ggml order reversed.
            w.add_tensor(p + "ssm_in.weight", rnd(2 * D_INNER, N_EMBD) * 2.0)
            w.add_tensor(p + "ssm_conv1d.weight", rnd(D_INNER, D_CONV) * 2.0)
            w.add_tensor(p + "ssm_x.weight", rnd(DT_DIM + 2 * D_STATE, D_INNER) * 2.0)
            w.add_tensor(p + "ssm_dt.weight", rnd(SSM_HEADS, DT_DIM))
            w.add_tensor(p + "ssm_dt.bias", rnd(SSM_HEADS) * 2.0)
            # Stored negative, as the converter writes -exp(A_log).
            w.add_tensor(p + "ssm_a", (-(0.5 + 2.0 * rng.random(SSM_HEADS))).astype(np.float32))
            w.add_tensor(p + "ssm_d", (1.0 + rnd(SSM_HEADS)).astype(np.float32))
            w.add_tensor(p + "ssm_out.weight", rnd(N_EMBD, D_INNER))
            # Weights drawn AWAY from one, so a skipped norm shows.
            w.add_tensor(p + norm_name("ssm_dt_norm"), (1.0 + 2.0 * rnd(DT_DIM)).astype(np.float32))
            w.add_tensor(p + norm_name("ssm_b_norm"), (1.0 + 2.0 * rnd(D_STATE)).astype(np.float32))
            w.add_tensor(p + norm_name("ssm_c_norm"), (1.0 + 2.0 * rnd(D_STATE)).astype(np.float32))
        else:
            q_dim, k_dim, v_dim = N_HEAD * HEAD_DIM, nkv * HEAD_DIM, nkv * HEAD_DIM
            w.add_tensor(p + "attn_qkv.weight", rnd(q_dim + k_dim + v_dim, N_EMBD) * 3.0)
            # {qk_dim, n_head}: one row per head, drawn away from one.
            w.add_tensor(p + "attn_q_norm.weight", (1.0 + 2.0 * rnd(N_HEAD, HEAD_DIM)).astype(np.float32))
            w.add_tensor(p + "attn_k_norm.weight", (1.0 + 2.0 * rnd(nkv, HEAD_DIM)).astype(np.float32))
            w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
        w.add_tensor(p + norm_name("post_attention_norm"), (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        # Phi-3 fused up: {n_embd, 2 n_ff}, gate first (plamo2.cpp:101,177).
        w.add_tensor(p + "ffn_up.weight", rnd(2 * N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
        w.add_tensor(p + norm_name("post_ffw_norm"), (1.0 + rnd(N_EMBD)).astype(np.float32))

    w.add_tensor("output_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} (kv per layer {kv_per_layer}, heads {heads_per_layer}, suffixed_norms={suffixed_norms})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "plamo2-fixture.gguf",
        "--suffixed-norms" in sys.argv[1:],
        "--scalar-heads" in sys.argv[1:],
    )
