#!/usr/bin/env python3
"""Generate the tiny synthetic `nanbeige` GGUFs used by ferrox's
layer-loop coverage test.

`nanbeige` refused as UNAUDITED, triaged NEW CODE, on one thing no other
graph has: it RUNS THE SAME PHYSICAL LAYERS MORE THAN ONCE
(`src/models/nanbeige.cpp`):

  * `:6-9` read `{arch}.num_loops` (default 1) and `:11-12`
    `{arch}.skip_loop_final_norm` (default false).
  * `:19-31` set `n_layer_all = n_layer_phys * n_loops` and COPY the
    per-layer head / kv-head / ff / swa arrays from physical layer `i`
    to every logical slot `i + j * n_phys`, so every logical layer has
    its own KV cache and its own entry in every per-layer table.
  * `:47-66` create tensors for the `n_phys` physical layers only, and
    `:69-73` alias `layers[i + j * n_phys] = layers[i]`: the weights are
    shared, the KV is not.
  * `:167-175`: after the LAST logical layer of each pass but the final
    one, the running residual is normed with `output_norm` (the same
    tensor the lm_head's norm uses) unless `skip_loop_final_norm`.
  * Everything inside a pass is plain Llama (`:52-63`, `:106-155`),
    which is what makes the tensor set look generic.

`crates/ferrox-models/src/layer_loops.rs` is the seam. Reach, measured:
`grep -l 'n_loops\\|n_layer_phys' src/models/*.cpp` over all 140 graphs
is `nanbeige.cpp`.

Shapes, one script:

  * (default) two physical layers, `num_loops = 2`: four logical
    layers 0,1,0,1, with a loop norm between logical layers 1 and 2 and
    NOT after logical layer 3 (`(il + 1) < n_layer` at `:170`, so the
    last pass hands its residual to the final norm untouched).
  * `--skip-loop-norm`: `skip_loop_final_norm = true`, the same four
    layers with nothing between the passes.
  * `--loops 1`: the key present at 1, which is a plain two-layer Llama
    (`:20` `if (n_loops > 1)`): the arm every real export without
    looping takes.

Weights are pseudo-random from a fixed seed so the file is byte-stable.
The FFN down projection is drawn wide so that the second pass over the
same weights moves the residual by far more than the tolerance, and
`output_norm` is drawn away from one so that a loop norm that is skipped
(or applied on the last pass) is visible.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_nanbeige_fixture.py OUT.gguf [--skip-loop-norm] [--loops N]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "nanbeige"

N_LAYER_PHYS = 2
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str, loops: int, skip_loop_norm: bool) -> None:
    rng = np.random.default_rng(0x9A9B)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-nanbeige-fixture")
    # block_count is the PHYSICAL count; the logical count is derived.
    w.add_block_count(N_LAYER_PHYS)
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
    # conversion/nanbeige.py:14-22 writes both, always.
    w.add_num_loops(loops)
    w.add_skip_loop_final_norm(skip_loop_norm)
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

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM
    for il in range(N_LAYER_PHYS):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q) * 4.0)
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 4.0)

    # Away from one: the loop norm's weight is this tensor too.
    w.add_tensor("output_norm.weight", (1.5 + rng.standard_normal(N_EMBD) * 0.5).astype(np.float32))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="nanbeige-fixture.gguf")
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--skip-loop-norm", action="store_true")
    args = ap.parse_args()
    main(args.out, args.loops, args.skip_loop_norm)
