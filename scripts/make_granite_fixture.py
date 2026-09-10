#!/usr/bin/env python3
"""Generate the tiny synthetic `granite` GGUF used by ferrox's Granite
coverage test.

`granite` is IBM's Granite 3.x dense decoder. It sat on ferrox's generic
GQA path refusing, triaged NEW CODE, for one reason: the four scalar
MULTIPLIERS `src/models/granite.cpp` applies and the generic decoder did
not.

`.scratch/llama.cpp/src/models/granite.cpp`, the parts this fixture is
built to drive:

  * `load_arch_hparams` (:5-10) reads the RMS epsilon,
    `LLM_KV_LOGIT_SCALE` as **REQUIRED**, and `residual_scale`,
    `embedding_scale` and `attention.scale` as optional. Nothing else in
    it is load-bearing for a 2-layer file: :36-42 only picks
    `LLM_TYPE_3B` off a layer count of 32 or 40 and nothing branches on
    the type, so a 2-layer fixture is legitimate here -- unlike
    `baichuan`, where the layer count selects ALiBi.
  * `load_arch_tensors` (:48-107) creates `attn_norm`, split Q/K/V via
    `create_tensor_qkv`, `attn_output`, an OPTIONAL `attn_output.bias`,
    `ffn_norm` and `ffn_gate`/`ffn_up`/`ffn_down` (`n_expert == 0`).
    `output` is `TENSOR_NOT_REQUIRED` with a tie-to-`token_embd`
    fallback; this fixture ships its own `output.weight`, which is the
    untied shape real Granite-3.0-8B has.
  * The graph applies the four multipliers in four different places:
      - the embedding scale in the SHARED `build_inp_embd`
        (llama-graph.cpp:2337-2342, `f_embedding_scale != 0.0f`);
      - `f_attention_scale` as `kq_scale`, with a `0.0f` sentinel
        meaning "use `1/sqrt(n_embd_head)`" (:225);
      - `f_residual_scale` on BOTH branch outputs, immediately before
        each residual add (:235-238 and :288-292);
      - `1.0f / f_logit_scale` on the final logits (:180).
  * RoPE is **NORM** (`LLM_ARCH_GRANITE` in `llama_model_rope_type`'s
    consecutive-pairs group, llama-model.cpp:2594) and the graph asserts
    `n_embd_head == n_rot` (:118), so head_dim, key_length, value_length
    and `rope.dimension_count` are all the same number here. The Granite
    converter POPS `head_dim` from the HF config
    (`conversion/granite.py:33-35`), so a real Granite file never
    decouples it from `n_embd / n_head` either.
  * `granite.cpp:206` gates RoPE ENTIRELY on `hparams.rope_finetuned`,
    which defaults to TRUE (:33-35). **No Granite converter writes
    `granite.rope.scaling.finetuned`**: the only call to
    `add_rope_scaling_finetuned` in `conversion/granite.py` is at :253,
    inside `GraniteHybridModel`, whose `model_arch` is
    `GRANITE_HYBRID`. So every converter-produced `granite` and
    `granitemoe` file rotates. ferrox refuses a file that declares the
    key FALSE rather than rotating anyway, and
    `tests/granite_family_graphs.rs` proves that refusal is reachable by
    writing the key.

**Why the multiplier values here are not the published ones.**
Granite-3.0-2b ships `embedding_multiplier: 12.0` and
`residual_multiplier: 0.22`. Those two together mean the residual stream
is ~55x the size of anything a layer adds to it, which is fine over 40
trained layers and useless in a 2-layer synthetic file: the final
RMSNorm would hand the lm_head a vector that is ~99% scaled embedding,
and every OTHER sabotage in the suite -- the RoPE variant, the attention
scale, the FFN -- would move the logits by less than the comparison
tolerance. The values below keep the same STRUCTURE (an embedding scale
above one, a residual scale below one, an attention scale well away from
`1/sqrt(head_dim)`, a logit scale above one) at magnitudes where every
feature of the graph is still visible in the output.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_granite_fixture.py OUT.gguf

    # the two reachability variants, for the refusals that stayed
    # refusals
    PYTHONPATH=... python3 scripts/make_granite_fixture.py OUT.gguf --no-rope
    PYTHONPATH=... python3 scripts/make_granite_fixture.py OUT.gguf --no-logit-scale

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "granite"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# granite.cpp:118 asserts n_embd_head_k == n_embd_head_v == n_rot, and
# the converter pops head_dim, so this IS n_embd / n_head.
HEAD_DIM = N_EMBD // N_HEAD
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5

# The four multipliers. See the module docstring for why these are not
# Granite-3.0-2b's published values.
LOGIT_SCALE = 2.5  # logits are DIVIDED by this (granite.cpp:180)
RESIDUAL_SCALE = 0.6  # both branch outputs, before the add
EMBEDDING_SCALE = 2.0  # the token embedding row
ATTENTION_SCALE = 0.9  # kq_scale; 1/sqrt(6) = 0.4082 is what it replaces


def main(out_path: str, rope_finetuned: bool, logit_scale: bool) -> None:
    rng = np.random.default_rng(0x6247E)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-granite-fixture")
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

    # The four keys this whole row is about.
    #
    # `--no-logit-scale` omits the one llama.cpp reads as REQUIRED
    # (granite.cpp:7, no default), producing a file llama.cpp itself
    # cannot load -- `key not found in model: granite.logit_scale`. It
    # exists so ferrox's matching refusal is driven by a real file
    # rather than asserted to exist.
    if logit_scale:
        w.add_logit_scale(LOGIT_SCALE)
    w.add_residual_scale(RESIDUAL_SCALE)
    w.add_embedding_scale(EMBEDDING_SCALE)
    w.add_attention_scale(ATTENTION_SCALE)

    # Only written for the `--no-rope` variant, which exists to prove
    # ferrox's refusal is reachable. llama.cpp would happily run this
    # file with NO rotation at all (granite.cpp:206); ferrox has no way
    # to express "this architecture, unrotated", so it stops.
    if not rope_finetuned:
        w.add_rope_scaling_finetuned(False)

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
        # cannot see a positional bug. Doubly so here, where
        # ATTENTION_SCALE replaces 1/sqrt(head_dim) with a value of its
        # own and the fixture has to stay peaked under BOTH.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        # NOTE: no `attn_output.bias`. granite.cpp:70 makes it optional
        # and no published Granite checkpoint carries one.
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        # SwiGLU (granite.cpp:257-263, LLM_FFN_SILU + LLM_FFN_PAR).
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    # Untied lm_head. granite.cpp:56-62 accepts either; an untied head
    # keeps `embedding_scale` and the lm_head as two separate facts.
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "granite-fixture.gguf",
        rope_finetuned="--no-rope" not in sys.argv[1:],
        logit_scale="--no-logit-scale" not in sys.argv[1:],
    )
