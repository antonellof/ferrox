#!/usr/bin/env python3
"""Generate the tiny synthetic `dbrx` GGUF used by ferrox's DBRX coverage
test.

`dbrx` sat on ferrox's generic GQA path refusing as UNAUDITED, triaged
NEW CODE on three blockers, and each turned out to be one implementation
that another row also needed:

  * **LayerNorm with a learned weight and no bias.** `dbrx.cpp:4` reads
    `LLM_KV_ATTENTION_LAYERNORM_EPS` (not the RMS one) and the graph
    normalises with `LLM_NORM` at all three sites -- :69-71
    pre-attention, :110-112 pre-FFN, :140-142 final -- with a weight and
    a null bias. `:29`, `:34` and `:23` create the three weights and no
    bias tensor at all. That is `crate::norm::NormOp::LayerNorm`, the
    variant the OLMo-1 work deliberately left unwritten until a row
    called it.
  * **A REQUIRED QKV clamp.** `dbrx.cpp:5` reads
    `{arch}.attention.clamp_kqv` with no default, and
    `llama-graph.cpp:1611-1652` clamps the fused projection to
    `[-c, c]` after the bias and before RoPE. `conversion/dbrx.py:28`
    writes `attn_config.clip_qkv` unconditionally, and every DBRX
    checkpoint sets it to 8. `crate::clamp_kqv` implements it, which
    also closed the clip_qkv sub-refusal on `olmo`.
  * **The pre-FFN norm under another name.** `dbrx.cpp:34` creates
    `attn_out_norm` (`blk.N.attn_output_norm`) and NO `ffn_norm`, and
    `:110-113` norms `ffn_inp` -- the post-attention residual -- with it.
    `crate::norm_sites` reads it into the pre-FFN slot, the same slot
    `gpt-oss` keeps under `post_attention_norm`.

What else this fixture pins, each against the C:

  * **A FUSED `attn_qkv.weight`** (:31, `{n_embd, n_embd + 2*n_embd_gqa}`)
    with no bias -- the `create_tensor` is for the weight alone -- read by
    `crate::qkv_fused`. GQA: 2 KV heads against 4 Q heads.
  * **NEOX RoPE**: `LLM_ARCH_DBRX` is in `llama_model_rope_type`'s
    `n_rot/2`-offset group (llama-model.cpp:2617). No
    `rope.dimension_count` key, as the converter writes none, so `n_rot`
    is `n_embd / n_head` and :50-51 assert it equals the head width.
  * **`1.0f/sqrtf(float(n_embd_head))`** passed literally at :97, so
    `ModelConfig::attention_scale` stays `None`.
  * **MoE only**: :16-18 throw on zero experts, every layer routes
    (:115-125) with `LLM_FFN_SILU`, softmax gating and `norm_w = true`
    (top-k weights renormalised), and there is no dense FFN and no
    shared expert.
  * **An UNTIED lm_head**: :24 creates `output` as required.

**Why the clamp bites.** A clamp that no projection ever reaches is a
no-op, and a fixture that cannot see whether the clamp ran cannot
evidence it. The norm weights are drawn at unit scale rather than the
0.25 every other tensor uses, so the LayerNorm hands the fused
projection a unit-variance vector; with Q and K drawn 4x wide the
projection's standard deviation is about 5 and a real fraction of its
elements cross 8.0. `tests/dbrx_graphs.rs` measures that the clamp
moves llama.cpp's own logits rather than assuming it.

**`--no-clamp` writes the file llama.cpp refuses.** `dbrx.cpp:5` reads
the clamp with no default, so a DBRX file without the key fails to load
there, and ferrox refuses it by name rather than defaulting it to "no
clamp" and running a graph the reference cannot. No golden goes with
that file; it exists so the refusal is driven from a file shaped exactly
like the ones the converter writes, minus one key.

Weights are pseudo-random from a fixed seed so the files are byte-stable
and differ in that one key alone.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_dbrx_fixture.py OUT.gguf
    PYTHONPATH=... python3 scripts/make_dbrx_fixture.py OUT.gguf --no-clamp

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "dbrx"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# dbrx.cpp:50-51 asserts n_embd_head == n_embd_head_k() == n_rot, and
# the converter writes no key_length / rope.dimension_count, so all
# three are n_embd / n_head.
HEAD_DIM = N_EMBD // N_HEAD  # 6
N_FF = 40
N_EXPERT = 4
N_EXPERT_USED = 2
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
# `LLM_KV_ATTENTION_LAYERNORM_EPS`, not the RMS one. conversion/dbrx.py:33
# hardcodes 1e-5.
LAYER_NORM_EPS = 1e-5
# Every DBRX checkpoint ships `attn_config.clip_qkv = 8`.
CLAMP_KQV = 8.0


def main(out_path: str, no_clamp: bool) -> None:
    rng = np.random.default_rng(0xDB2C)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-dbrx-fixture")
    # The same key set conversion/dbrx.py:16-36 writes, and nothing else.
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_rope_freq_base(ROPE_BASE)
    # REQUIRED (dbrx.cpp:5). A file without it is refused by llama.cpp
    # and by ferrox alike; `--no-clamp` is that file.
    if not no_clamp:
        w.add_clamp_kqv(CLAMP_KQV)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    # THE LAYERNORM SPELLING. `add_layer_norm_rms_eps` would be the wrong
    # key for this architecture: dbrx.cpp:4 reads
    # LLM_KV_ATTENTION_LAYERNORM_EPS.
    w.add_layer_norm_eps(LAYER_NORM_EPS)
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

    # Drawn with a NON-ZERO MEAN on purpose. A LayerNorm subtracts the
    # mean of the vector it norms and an RMSNorm does not, so a fixture
    # whose hidden states are centred by construction would make the two
    # functions agree and would not be able to see the difference it
    # exists to pin.
    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD) + 0.5)

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM

    for il in range(N_LAYER):
        p = f"blk.{il}."
        # Unit-scale norm weights (see the module docstring): the
        # LayerNorm's output then has unit variance, which is what lets
        # the clamp below bite.
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD) * 4.0)

        # ONE fused tensor, Q rows then K rows then V rows
        # (llama-model.cpp:2889 sizes it, llama-graph.cpp:1615-1622
        # views it). Q and K wide so the clamp has something to clamp
        # and the attention scores over six tokens are not near-uniform;
        # V at the ordinary scale.
        qkv = np.concatenate(
            [
                rnd(n_embd_q, N_EMBD) * 4.0,
                rnd(n_embd_kv, N_EMBD) * 4.0,
                rnd(n_embd_kv, N_EMBD),
            ],
            axis=0,
        )
        w.add_tensor(p + "attn_qkv.weight", qkv)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        # The PRE-FFN norm, under dbrx's name for it. No `ffn_norm`.
        w.add_tensor(p + "attn_output_norm.weight", rnd(N_EMBD) * 4.0)

        # Router logits drawn wide enough that the top-2 of four is a
        # real decision: a near-uniform router would make the whole
        # routed half of this fixture untestable.
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD) * 4.0)
        # gate/up: ne = [n_embd, n_ff, n_expert]
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD) + 0.3)
        # down: ne = [n_ff, n_embd, n_expert]. Biased so the residual
        # stream keeps a non-zero mean for the next LayerNorm to remove.
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF) + 0.3)

    w.add_tensor("output_norm.weight", rnd(N_EMBD) * 4.0)
    # Untied: dbrx.cpp:24 creates `output` as required.
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "dbrx-fixture.gguf",
        no_clamp="--no-clamp" in sys.argv[1:],
    )
