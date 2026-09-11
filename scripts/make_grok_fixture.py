#!/usr/bin/env python3
"""Generate the tiny synthetic `grok` GGUFs used by ferrox's Grok-1
coverage test.

`grok` sat on ferrox's generic GQA path refusing as UNAUDITED, triaged
NEW CODE, and its verdict named the MiniCPM shape: defaults assigned
before the file can override them, so a key-presence gate has nothing
to see. `.scratch/llama.cpp/src/models/grok.cpp:5-12` seeds SEVEN
hyper-parameters ("defaults for old GGUFs") and only then (`:14-27`)
reads each key with `required = false`:

    hparams.yarn_beta_fast           = 8.0f;
    hparams.f_logit_scale            = 0.5773502691896257f;   # 1/sqrt(3)
    hparams.f_embedding_scale        = 78.38367176906169f;    # sqrt(6144)
    hparams.f_attn_out_scale         = 0.08838834764831845f;  # 1/sqrt(128)
    hparams.f_attn_logit_softcapping = 30.0f;
    hparams.f_router_logit_softcapping = 30.0f;
    hparams.f_final_logit_softcapping = 0.0f;                 # off

That is why the DEFAULT fixture this script writes declares **none of
those keys**: a fixture that declared them would pass with or without
the defaults hook and would prove nothing. `--declared` writes the
other half, every key present with a value nowhere near its default,
and it is not a nicety -- a hook merged the wrong way round agrees with
llama.cpp on exactly the files that prove it exists, and every fresh
export carries the keys (`conversion/grok.py:34-57` writes all of them).

How each seed reaches the graph, against the C, and what ferrox does:

  * **embedding_scale** -- the shared `build_inp_embd`
    (llama-graph.cpp:2337-2342). `scalar_multipliers`, as for Granite.
  * **logit_scale** -- `ggml_scale(cur, f_logit_scale)` at :211, a
    MULTIPLY where Granite divides. `LogitScaleUse::AsIs`.
  * **attn_out_scale** -- NOT `kq_scale`. :137 passes `kq_scale = 1.0f`
    and llama-graph.cpp:2572-2582 computes
    `kq = 30 * tanh(kq * f_attn_out_scale / 30)` before the softmax,
    inside the non-flash-attention branch that llama-context.cpp:3544
    forces for Grok. Arithmetically that is "pre-scale Q by
    f_attn_out_scale, then softcap at 30", so the key resolves into
    `ModelConfig::attention_scale` beside the existing softcap. The key
    is `{arch}.attention.output_scale` (:18), not `attention.scale`.
  * **attn_logit_softcapping** -- the 30 in that formula.
    `MultiplierDefaults::attn_logit_softcap` is the default.
  * **final_logit_softcapping** -- :214-218, applied only when nonzero.
    The default file leaves it off; `--declared` switches it on.
  * **router_logit_softcapping** and **attention.temperature_length**
    -- read at :20 and :23 and applied NOWHERE: no other reference to
    either field exists under `src/` (measured with grep). `--declared`
    writes both so the test can pin that ferrox neither applies nor
    refuses them.
  * **yarn_beta_fast** -- only reachable through YaRN rope scaling,
    which Grok-1 does not declare; carried in the same defaults variant
    and pinned by a unit test rather than by this fixture.

The rest of the graph, each against the C:

  * `attn_output_norm` (:62) is applied to the ATTENTION OUTPUT before
    the residual add (:143-148): Gemma-2's post-attention slot, and NOT
    dbrx's pre-FFN slot under the same name. `ffn_norm` (:64) is a
    separate pre-FFN norm. `layer_output_norm` (:75, the Grok-1 name)
    is the post-FFN norm applied at :185-190. All RMSNorm (:110, :145,
    :154, :187, :203).
  * **GELU MoE** with softmax gating and top-k renormalisation
    (:158-168, `LLM_FFN_GELU, true`). No dense FFN: Grok-1 takes the
    `else` at :182-184. The experts are sized from `n_ff` because
    `expert_feed_forward_length` is absent (:53).
  * **NEOX RoPE** (`LLM_ARCH_GROK` in the `n_rot/2`-offset group,
    llama-model.cpp:2616), full head width (:90 asserts
    `n_embd_head == n_rot`).
  * An lm_head that is UNTIED here. :46-51 accept either and fall back
    to `token_embd` when `output` is absent; this file ships
    `output.weight` because `token_embd` is drawn 1/78.4 as wide (see
    below) and a tied head would shrink every logit by the same factor,
    leaving the sabotage margins under the comparison tolerance. The
    fallback itself is generic loader code that `olmo`'s fixture
    exercises.

**Magnitudes.** The default embedding multiplier is 78.4 and is not
ours to choose, so `token_embd` is drawn 1/78.4 as wide as every other
tensor: after the multiplier the residual stream sits at the same scale
the other fixtures use, and the branches stay visible against it. Q and
K are drawn 32x wide so that the attention scores reach the softcap --
at ordinary magnitudes `30 * tanh(s / 30)` is `s` to six digits and the
softcap default would be invisible to the test that exists to pin it.

`--dense-ffn` writes the Grok-2 shape: dense `ffn_gate/up/down` beside
the experts, which :171-184 sums with the MoE and scales by sqrt(2)/2.
ferrox refuses it by name (`crate::parallel_dense_ffn`); no golden goes
with that file.

Weights are pseudo-random from a fixed seed so the files are byte-stable
and the three variants differ only in metadata (or, for `--dense-ffn`,
in three extra tensors per layer).

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_grok_fixture.py OUT.gguf
    PYTHONPATH=... python3 scripts/make_grok_fixture.py OUT.gguf --declared
    PYTHONPATH=... python3 scripts/make_grok_fixture.py OUT.gguf --dense-ffn

The golden logits that go with them are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "grok"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# grok.cpp:89-90 asserts n_embd_head == n_embd_head_k() == n_rot.
HEAD_DIM = N_EMBD // N_HEAD  # 6
N_FF = 40
N_EXPERT = 4
N_EXPERT_USED = 2
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
# grok.cpp:14 reads the RMS spelling as REQUIRED.
RMS_EPS = 1e-5

# What llama.cpp applies to THIS file when it declares nothing. Not
# written anywhere; here so the test can name the same numbers.
DEFAULT_EMBEDDING_SCALE = 78.38367176906169
#   logit          = 0.5773502691896257  (multiplied)
#   attn_out_scale = 0.08838834764831845 (inside the softcap)
#   attn softcap   = 30.0
#   final softcap  = 0.0 (off)

# The `--declared` values, chosen to be nowhere near any default, so a
# hook applied in the wrong order is visible rather than marginal.
DECLARED_LOGIT_SCALE = 0.9
DECLARED_EMBEDDING_SCALE = 40.0
DECLARED_ATTN_OUTPUT_SCALE = 0.2
DECLARED_ATTN_SOFTCAP = 10.0
DECLARED_FINAL_SOFTCAP = 5.0
# Read and never applied by llama.cpp; written to prove ferrox neither
# applies nor refuses them.
DECLARED_ROUTER_SOFTCAP = 30.0
DECLARED_ATTN_TEMPERATURE_LENGTH = 4096


def main(out_path: str, declared: bool, dense_ffn: bool) -> None:
    rng = np.random.default_rng(0x6120)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-grok-fixture")
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
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # THE POINT OF THE DEFAULT FILE IS THAT THIS BLOCK IS SKIPPED.
    #
    # No `grok.logit_scale`, no `grok.embedding_scale`, no
    # `grok.attention.output_scale`, no `grok.attn_logit_softcapping` --
    # and llama.cpp still multiplies every logit by 1/sqrt(3), every
    # embedding row by 78.4, every attention score by 1/sqrt(128), and
    # still softcaps the scores at 30.
    if declared:
        w.add_logit_scale(DECLARED_LOGIT_SCALE)
        w.add_embedding_scale(DECLARED_EMBEDDING_SCALE)
        w.add_attn_output_scale(DECLARED_ATTN_OUTPUT_SCALE)
        w.add_attn_logit_softcapping(DECLARED_ATTN_SOFTCAP)
        w.add_final_logit_softcapping(DECLARED_FINAL_SOFTCAP)
        w.add_router_logit_softcapping(DECLARED_ROUTER_SOFTCAP)
        w.add_attn_temperature_length(DECLARED_ATTN_TEMPERATURE_LENGTH)

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

    # ne = [n_embd, n_vocab] -> numpy [n_vocab, n_embd]. Drawn narrow:
    # see "Magnitudes" in the module docstring.
    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD) / DEFAULT_EMBEDDING_SCALE)

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        # Q and K wide enough for the scores to reach the softcap; see
        # the module docstring.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 32.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 32.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        # The POST-attention norm, under Grok's name for it (:62, :143).
        w.add_tensor(p + "attn_output_norm.weight", rnd(N_EMBD))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))

        # Router logits drawn wide enough that the top-2 of four is a
        # real decision: a near-uniform router would make the whole
        # routed half of this fixture untestable.
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD) * 4.0)
        # gate/up: ne = [n_embd, n_ff, n_expert]; down: [n_ff, n_embd, n_expert].
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF))

        if dense_ffn:
            # The Grok-2 shape (:66-68 create them optional, :171-184
            # sum them with the experts). Refused by ferrox.
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

        # The POST-FFN norm, under the Grok-1 name (:75). :77 falls back
        # to `post_ffw_norm` for Grok-2 exports.
        w.add_tensor(p + "layer_output_norm.weight", rnd(N_EMBD))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    # Untied, at the ordinary scale; see the module docstring.
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD) * 0.5)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "grok-fixture.gguf",
        declared="--declared" in sys.argv[1:],
        dense_ffn="--dense-ffn" in sys.argv[1:],
    )
