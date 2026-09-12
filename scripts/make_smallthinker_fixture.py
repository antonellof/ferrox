#!/usr/bin/env python3
"""Generate the tiny synthetic `smallthinker` GGUFs used by ferrox's
router-input coverage test.

`smallthinker` refused as UNAUDITED, triaged NEW CODE, on three things,
and the first of them is the one no other generic-path graph has:

  * **The MoE router reads the RAW LAYER INPUT.** `src/models/
    smallthinker.cpp:111` is `probs = build_lora_mm(ffn_gate_inp, inpL)`
    -- `inpL` being the residual stream as it ENTERS the layer, before
    `attn_norm` (:115), before attention, before the FFN norm -- and
    :151-161 hands those logits to `build_moe_ffn` as `probs_in` with a
    NULL `ffn_gate_inp`. Every other MoE graph on the generic path
    routes on the normed FFN input, which is what ferrox computed.
    `crates/ferrox-models/src/router_input.rs` is the seam and has the
    census: four of 140 graphs pass a precomputed `probs_in`, and this
    is the only one whose operand is the layer input.
  * **`LLM_FFN_RELU` experts** (:158), the GATED form: `build_moe_ffn`
    takes `ggml_reglu_split(gate, up)` for it (llama-graph.cpp:2195-
    2197), i.e. `relu(gate) * up` with a real gate tensor. NOT `arcee`'s
    ungated `relu(up)^2`, which is `LLM_FFN_RELU_SQR` and a different
    op. `FfnActivation::Reglu`.
  * **`n_swa` is PINNED to 4096** (:8) whenever the file declares ANY
    nonzero window (:4-6); the declared value is read and discarded.
    `capability::swa_window_override`.

And the one it had already: the NoPE layers. `llama-hparams.h:203`
defaults `n_no_rope_layer_step` to 4, no key ever changes it, and
:108-109 leaves `il % 4 == 0` unrotated -- layers 0 and 4 here, the
FIRST of each period, the other phase from `smollm3`'s. Without a
window :18 sets the step to `n_layer`, which rotates everything.
`crate::rope_layers` carries both.

**FIVE LAYERS ARE NOT AN ACCIDENT.** With a NoPE period of 4, layers
0 and 4 are unrotated and 1-3 rotate; a four-layer file would have one
unrotated layer at the very bottom, where a sabotage of the phase is
cheapest to miss. Five puts an unrotated layer on top of a rotated
one, and it also puts a full-attention layer (`set_swa_pattern(4,
dense_first=true)`, :11: `il % 4 != 0` slides) at each end with three
sliding ones between.

Three shapes, one script:

  * (default) `attention.sliding_window = 3`, narrower than the
    six-token prompt: if the value were honoured, three of five layers
    would mask, and the logits would move by far more than any
    tolerance. llama.cpp pins 4096 over it and masks nothing; the
    golden values for this file are BYTE-IDENTICAL to those for the
    same file declaring 4096 (measured, see the test), which is the
    evidence that the pin is upstream's behaviour and not a reading of
    it. SIGMOID gating, which is what `conversion/smallthinker.py:27-30`
    writes for a config without `moe_primary_router_apply_softmax`.
  * `--no-window`: the key absent, so :16-18 take the no-SWA branch and
    every layer rotates. SOFTMAX gating, the converter's other arm.
    Four layers, because with the step at `n_layer` there is no phase
    to see.
  * `--swa-period 2`: `attention.sliding_window_pattern = 2` and
    `rope.freq_base_swa = 100` against a base of 10000. :10 reads the
    period from the key while the NoPE step stays the literal 4, so on
    this file the two disagree: layers 1 and 3 slide (and rope at
    100), layer 2 is full (10000), layers 0 and 4 are NoPE. The
    window is still pinned to 4096 so the mask never bites; the SWA
    base is what makes the pattern visible. No converter writes the
    pattern key for this architecture; this is a hand-written shape
    that pins WHICH of two periods each rule reads.

The router weights are drawn wide and the attention output wide, so
that routing on `inpL` and routing on `ffn_norm(ffn_inp)` pick
DIFFERENT top-2 experts on this prompt; the test measures that rather
than assuming it.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_smallthinker_fixture.py OUT.gguf \\
            [--no-window] [--swa-period N] [--declared-window W]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "smallthinker"

N_LAYER = 5
N_LAYER_NO_WINDOW = 4
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF_EXP = 16
N_EXPERT = 6
N_EXPERT_USED = 2
N_VOCAB = 48
CTX = 64
# Narrower than the six-token prompt, so that HONOURING it would mask.
SWA_WINDOW_DECLARED = 3
ROPE_BASE = 10000.0
# Two decades under the base, so which layers rope at it is visible
# on a six-token prompt with an 8-wide head.
ROPE_BASE_SWA = 100.0
RMS_EPS = 1e-5


def main(out_path: str, no_window: bool, swa_period: int | None, declared_window: int) -> None:
    rng = np.random.default_rng(0x5A11)
    n_layer = N_LAYER_NO_WINDOW if no_window else N_LAYER

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-smallthinker-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    # conversion/smallthinker.py:23-25 writes BOTH lengths from
    # `moe_ffn_hidden_size`; there is no dense FFN anywhere.
    w.add_feed_forward_length(N_FF_EXP)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    if no_window:
        w.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
    else:
        w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        w.add_sliding_window(declared_window)
        if swa_period is not None:
            w.add_sliding_window_pattern(swa_period)
            w.add_rope_freq_base_swa(ROPE_BASE_SWA)
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

    for il in range(n_layer):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        # Drawn wider than the rest of the file so the softmax over a
        # six-token prompt is not near-uniform: a near-uniform attention
        # cannot see whether a layer rotated.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        # Wide, so the attention branch moves the residual by enough that
        # `inpL` and `ffn_norm(inpL + attn)` are not near-parallel: that
        # is what makes the two routing operands pick different experts.
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q) * 4.0)

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))

        # smallthinker.cpp:61: `{n_embd, n_expert}`, REQUIRED. Wide, so
        # the sigmoid scores are spread and the top-2 is decisive.
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD) * 4.0)
        # gate/up: ne = [n_embd, n_ff_exp, n_expert]
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        # down: ne = [n_ff_exp, n_embd, n_expert]. Drawn wider so the
        # FFN branch is a real share of the residual: at the file's
        # default scale the whole branch moved the logits by 6e-3, and a
        # sabotage of the activation has to clear the tolerance by
        # orders of magnitude, not by a factor.
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP) * 4.0)

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="smallthinker-fixture.gguf")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--swa-period", type=int, default=None)
    ap.add_argument(
        "--declared-window",
        type=int,
        default=SWA_WINDOW_DECLARED,
        help="the value written to attention.sliding_window (llama.cpp ignores it)",
    )
    args = ap.parse_args()
    main(args.out, args.no_window, args.swa_period, args.declared_window)
