#!/usr/bin/env python3
"""Generate the tiny synthetic `hrm_text` GGUF (DFM Mimir 1B's shape).

`hrm_text` landed upstream after the 2026-08-04 llama.cpp pin and was
triaged NEW CODE on 2026-09-19 for its schedule, which is two things at
once (`src/models/hrm-text.cpp`):

  * **Two stacks replayed.** `:47-87` creates `2 * layers_per_stack`
    blocks -- the LOW stack at `[0, lps)` and the HIGH stack at
    `[lps, 2*lps)` -- and every later pass ALIASES one of the two, so
    the file holds `2 * lps` blocks while `block_count` is the expanded
    slot count `lps * h_cycles * (l_cycles + 1)` (`:22-23` asserts it).
    Each slot keeps its own KV.
  * **Two residual streams.** `:183-196` runs `h_cycles` cycles of
    `l_cycles` LOW stacks and one HIGH stack; every stack reads
    `zH + zL` and replaces one of the two with its output. `zH` starts
    as the embeddings and `zL` as the learned `hrm_z_l_init` row
    (`:182`), and the lm_head reads `zH` (`:198`) with NO final norm --
    every stack ends with its own weightless RMS (`:162`) and the last
    one is the final norm.

Two fixtures. The default is `lps = 2, h = 2, l = 1`: eight slots over
four blocks, passes LOW HIGH LOW HIGH. `--deep` is `l = 2`: twelve
slots, passes LOW LOW HIGH LOW LOW HIGH, which is the case where the
LOW stack runs TWICE IN A ROW against two different cache slots -- a
schedule that aliased the wrong way would still alternate correctly in
the first file and not in the second.

Everything else it carries was already served: weightless RMS norms at
every layer slot (`:107,144,162`, `NormOp::RmsNoParams`), a per-element
sigmoid attention gate (`:77-78,113-134`), an `embedding_scale`
(`:8`), NEOX RoPE, and a SwiGLU FFN.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \
        python3 scripts/make_hrm_text_fixture.py OUT.gguf

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "hrm_text"

N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5
EMBEDDING_SCALE = 1.25

LPS = 2
H_CYCLES = 2
# `--deep` is the same weights with TWO low passes per cycle, so the
# LOW stack runs twice in a row against two different cache slots --
# the aliasing case the alternating schedule never reaches.
L_CYCLES = 1
DEEP_L_CYCLES = 2


def main(out_path: str, deep: bool = False) -> None:
    l_cycles = DEEP_L_CYCLES if deep else L_CYCLES
    n_slot = LPS * H_CYCLES * (l_cycles + 1)
    n_block = 2 * LPS
    rng = np.random.default_rng(0x8121)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-hrm-text-fixture")
    # The BLOCK COUNT is the expanded slot count (`:22-23`), not the
    # number of blocks the file holds.
    w.add_block_count(n_slot)
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
    w.add_embedding_scale(EMBEDDING_SCALE)
    w.add_uint32(f"{ARCH}.hrm.layers_per_stack", LPS)
    w.add_uint32(f"{ARCH}.hrm.h_cycles", H_CYCLES)
    w.add_uint32(f"{ARCH}.hrm.l_cycles", l_cycles)
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
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))
    # The learned LOW stream, one row broadcast over the tokens; drawn
    # away from zero so a decoder that forgot it would diverge.
    w.add_tensor("hrm.z_l_init", (rng.standard_normal(N_EMBD) * 0.8).astype(np.float32))

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM

    for bid in range(n_block):
        p = f"blk.{bid}."
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 3.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 3.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 3.0)
        w.add_tensor(p + "attn_gate.weight", rnd(n_embd_q, N_EMBD) * 6.0)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} ({n_slot} slots over {n_block} blocks)")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "hrm-text-fixture.gguf",
        "--deep" in sys.argv[2:],
    )
