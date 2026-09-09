#!/usr/bin/env python3
"""Generate the tiny synthetic `hunyuan-dense` GGUF used by ferrox's
HunYuan dense coverage test.

`hunyuan-dense` has no graph of its own: `src/models/models.h:1830-1834`
derives `llama_model_hunyuan_dense` from `llama_model_hunyuan_vl` and
reuses its `load_arch_hparams`, its `load_arch_tensors` and its `graph`,
so the file to read is `src/models/hunyuan-vl.cpp`.

  * `load_arch_hparams` (:3-21) reads the RMS epsilon, the optional
    `LLM_KV_ROPE_DIMENSION_SECTIONS`, and then applies THE ARM (:8-12):

        if (hparams.rope_scaling_alpha > 0.0f) {
            const int dim = hparams.n_embd_head_k();
            hparams.rope_freq_base_train = hparams.rope_freq_base_train
                * powf(hparams.rope_scaling_alpha, (float)dim / (float)(dim - 2));
        }

    `{arch}.rope.scaling.alpha` itself is read generically for every
    architecture at llama-model.cpp:1186; only this graph and
    `hunyuan-vl` apply it.
  * `load_arch_tensors` (:23-54) creates `attn_norm`, split Q/K/V,
    `attn_output`, per-head `attn_q_norm`/`attn_k_norm` of width
    `n_embd_head_k`, `ffn_norm` and `ffn_gate`/`ffn_up`/`ffn_down`. No
    biases, no post-norms, no window, no experts.
  * The graph (:60-175) ropes Q and K (:56-66 of the else branch) and
    THEN norms them (:73-81) -- the post-RoPE QK-norm order ferrox
    already implements as `Decoder::qk_norm_after_rope` for
    `hunyuan-moe` and `maincoder`. `kq_scale = 1/sqrt(n_embd_head)`
    (:22), SiLU SwiGLU (:100-105), sequential residual.
  * RoPE is **NEOX** (`LLM_ARCH_HUNYUAN_DENSE` in
    `llama_model_rope_type`'s NEOX group, llama-model.cpp:2664).

WHAT A REAL CHECKPOINT CARRIES, and why the fixture is not one:
`conversion/hunyuan.py:254-281` is `HunYuanModel`, the `HUNYUAN_DENSE`
converter, and it does the NTK-alpha arithmetic in PYTHON --
`scaled_base = base * (alpha ** (dim / (dim - 2)))` at :270 -- writing
the already-scaled value through `add_rope_freq_base` and no alpha key
at all. The `add_rope_scaling_alpha` call at :356 belongs to
`HunyuanVLTextModel`, whose `model_arch` is `HUNYUAN_VL`: a different
GGUF architecture string and a different ferrox row. So on a converted
file llama.cpp's :8-12 is a no-op, and this fixture writes the key
explicitly precisely so that the arm ferrox implements is the arm the
reference runs.

`{arch}.rope.freq_base` is deliberately SMALL (500) rather than the
usual 10000. With alpha 50 and an 8-wide head the rescale takes the base
to 500 * 50^(8/6) ~= 92100, and the difference between rotating at 500
and rotating at 92100 is radians per position rather than rounding --
a fixture whose base barely moved could not see the arm it exists for.

DELIBERATELY ABSENT: `{arch}.rope.dimension_sections`. `use_mrope()`
(llama-hparams.cpp:284-286) is true only when the first two sections are
positive, and no converter writes the key under the `hunyuan-dense`
prefix, so the M-RoPE branch at :43-54 is unreachable for this row and
ferrox is not asked to have M-RoPE.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_hunyuan_dense_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "hunyuan-dense"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# hunyuan-vl.cpp:63-64 asserts n_embd_head_k == n_embd_head_v == n_rot,
# so all three are 8. It is still not n_embd / n_head, which is 6.
HEAD_DIM = 8
N_FF = 40
N_VOCAB = 48
CTX = 64
# Small on purpose -- see the module docstring. The scaled base is
# 500 * 50^(8/6) ~= 92100.
ROPE_BASE = 500.0
ROPE_SCALING_ALPHA = 50.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x40D3)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-hunyuan-dense-fixture")
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
    # THE ARM. hunyuan-vl.cpp:8-12 rescales the trained base by
    # alpha^(head_dim / (head_dim - 2)) when this is positive.
    w.add_rope_scaling_alpha(ROPE_SCALING_ALPHA)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # NOTE: no `{arch}.rope.dimension_sections`. See the module
    # docstring: `use_mrope()` must stay false so the plain
    # `ggml_rope_ext` branch is the one under test.

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

        # Q and K are drawn WIDER than everything else on purpose. At the
        # magnitude the rest of the file uses, the attention scores over
        # a six-token prompt sit within a fraction of each other, softmax
        # comes out nearly uniform, and the layer stops caring where the
        # tokens are -- which is fatal here, because both sabotages this
        # fixture has to survive (the RoPE base and the RoPE variant) are
        # positional.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        # Per-head QK norm, width n_embd_head_k (hunyuan-vl.cpp:44-45),
        # applied AFTER RoPE. Centred near 1.5 rather than near 1.0 for
        # the same reason `maincoder` and `hunyuan-moe` are: at weights
        # near 1 an RMSNorm is nearly the identity and the two orderings
        # agree, so the ordering sabotage would prove nothing.
        w.add_tensor(p + "attn_q_norm.weight", (rnd(HEAD_DIM) + 1.5).astype(np.float32))
        w.add_tensor(p + "attn_k_norm.weight", (rnd(HEAD_DIM) + 1.5).astype(np.float32))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    # `output` is TENSOR_NOT_REQUIRED with a tok_embd fallback
    # (hunyuan-vl.cpp:30-34); the fixture ships its own.
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hunyuan-dense-fixture.gguf")
