#!/usr/bin/env python3
"""Generate the tiny synthetic `plamo3` GGUF used by frink's PLaMo-3
coverage test.

`plamo3` is Preferred Networks' PLaMo-3. It sat on frink's generic GQA
path refusing as UNAUDITED, triaged FIXTURE-AWAY -- and that verdict was
WRONG by one tensor name, which building this fixture is what found.

**The name.** `LLM_TN` appends `.weight` only when it is given a suffix
(`src/llama-arch.cpp:898-910`). Every architecture that creates
`ATTN_POST_NORM` / `FFN_POST_NORM` passes one -- gemma2, gemma3, glm4,
exaone4, afmoe -- except `plamo3`, which uses the two-argument overload
(`src/models/plamo3.cpp:52,55`) and so asks for
`blk.N.post_attention_norm` and `blk.N.post_ffw_norm` with NO suffix.
The converter agrees, by a second accident that lines up:
`gguf-py/gguf/tensor_mapping.py:368,434` give the PLaMo entries as
`model.layers.layers.{bid}.post_mixer_norm.weight` and
`...post_mlp_norm.weight`, keys that already END in `.weight`, and
`TensorNameMap.get_type_and_name` (:2585-2594) tries an exact match
first, so nothing is appended. **This file therefore writes the
un-suffixed names, because that is what a real PLaMo-3 checkpoint
carries** -- and a `.weight`-spelled fixture is rejected by libllama with
"tensor 'blk.0.post_attention_norm' not found", which is how the
discrepancy was found rather than guessed.

Everything else it pins against `plamo3.cpp`:

  * The SANDWICH residual: `attn_norm` before attention (:104),
    `attn_post_norm` applied to the attention OUTPUT before its residual
    add (:152-155), `ffn_norm` before the FFN (:160), `ffn_post_norm`
    applied to the FFN output before its add (:171-174). All four norm
    weights here are centred near 1.0 the way the converter leaves them
    (`conversion/plamo.py:182-193` adds 1, 1/5, 1 and 1/5**1.5 to the
    stored deltas), and all four are non-zero so a dropped slot shows.
  * A FUSED `attn_qkv` (:47) sized `{n_embd, q + k + v}` from
    head_dim_q/head_dim_v explicitly, so `head_dim != n_embd / n_head`
    is allowed and this file uses 8 where dividing would give 6.
  * Per-head `attn_q_norm` / `attn_k_norm` of width `head_dim_q`
    (:49-50) applied BEFORE RoPE (:128-138).
  * A fused SwiGLU `ffn_up` of `n_ff * 2` with no `ffn_gate`, driven by
    `LLM_FFN_SWIGLU, LLM_FFN_SEQ` (:57, :163-168) -- the audited phi3
    path, first half gate, second half up.
  * NEOX RoPE (`LLM_ARCH_PLAMO3` in `llama_model_rope_type`'s NEOX
    group, llama-model.cpp:2641).
  * SLIDING-WINDOW ATTENTION with a real pattern AND a real phase. :5-11
    reads `attention.sliding_window` and a scalar
    `attention.sliding_window_pattern` (default 8) and calls
    `set_swa_pattern(period)` with `dense_first = false`, i.e.
    `is_swa(il) = il % period < period - 1`. This file sets period 2 and
    four layers, so layers 0 and 2 slide and layers 1 and 3 are full --
    and it sets the window to 3 over a six-token prompt, so the window
    actually bites at the position the comparison reads. A fixture with
    a window wider than its prompt would compare equal with SWA
    switched off entirely.
  * `rope.freq_base_swa` is NOT written, and plamo3.cpp never seeds
    `rope_freq_base_train_swa` from the model's own base, so llama.cpp's
    default of 10000 (`src/llama-hparams.h:127`) applies -- which is the
    same as this file's `rope_freq_base`, so the sliding and full layers
    rotate identically and the SWA test is about MASKING alone.
  * `attention.key_length == attention.value_length`, which the triage
    verdict asked to confirm on the fixture: llama.cpp carries
    head_dim_q and head_dim_v separately (:25-26) and frink has one
    head_dim, so a checkpoint where they differ is refused by name in
    `loader.rs` and is out of scope here.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_plamo3_fixture.py OUT.gguf

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "plamo3"

# Four layers with a period of 2: SWA, full, SWA, full.
N_LAYER = 4
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
# Deliberately NOT n_embd / n_head (which would be 6). plamo3.cpp:41-43
# sizes the fused QKV from head_dim_q / head_dim_v explicitly.
HEAD_DIM = 8
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5
# Narrower than the six-token prompt on purpose: a window that never
# bites is a window that cannot be tested.
SLIDING_WINDOW = 3
SWA_PATTERN = 2


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x91A403)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic values in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-plamo3-fixture")
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
    w.add_sliding_window(SLIDING_WINDOW)
    w.add_sliding_window_pattern(SWA_PATTERN)
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

        # Fused QKV, Q then K then V. Drawn WIDER than the rest of the
        # file: at one magnitude the attention scores over a six-token
        # prompt sit within a fraction of each other, softmax comes out
        # nearly uniform, and the layer stops caring which positions it
        # can see -- which would leave both the RoPE-variant and the
        # sliding-window sabotage tests unable to see their change.
        w.add_tensor(p + "attn_qkv.weight", rnd(n_embd_q + 2 * n_embd_kv, N_EMBD) * 4.0)
        # Per-head, width head_dim_q, applied before RoPE.
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.0)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.0)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        # NO `.weight` on either post-norm. See the module docstring:
        # this is the spelling llama.cpp asks for and the converter
        # emits, and it is the whole reason plamo3 was not actually a
        # fixture away.
        w.add_tensor(p + "post_attention_norm", rnd(N_EMBD) + 1.0)

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "post_ffw_norm", rnd(N_EMBD) + 1.0)

        # Fused SwiGLU: first half gate, second half up. No ffn_gate.
        w.add_tensor(p + "ffn_up.weight", rnd(2 * N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plamo3-fixture.gguf")
