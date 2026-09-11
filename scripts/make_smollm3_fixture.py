#!/usr/bin/env python3
"""Generate the tiny synthetic `smollm3` GGUF used by ferrox's
per-layer-RoPE coverage test.

`smollm3` was refused OUTRIGHT -- `ArchPath::DedicatedOnly`, in
`capability.rs`'s "No RoPE at all" group beside the ALiBi
architectures -- and its graph is otherwise the plain pre-norm llama
one, line for line (`src/models/smollm3.cpp:72-134`: `attn_norm`,
`create_tensor_qkv`, `wo`, `ffn_norm`, SwiGLU, both residuals, RMS
output norm). The ONE thing it needed was a way to say **which layers
rotate**:

    smollm3.cpp:5   hparams.n_no_rope_layer_step = 4;
    smollm3.cpp:69  const bool use_rope = (il + 1) % hparams.n_no_rope_layer_step != 0;

so 9 of a 36-layer SmolLM3-3B's layers get no rotation at all. Nine
layers, and NO GGUF KEY: the step is a literal in `load_arch_hparams`
and `grep -rn n_no_rope_layer_step src/` finds no reader for it
anywhere in llama.cpp. The tensor set is byte-for-byte the generic llama
set, so the file loads clean, runs at full speed, and answers fluently
from positions three layers in four never encode -- which is why it was
a refusal rather than a gate.

ferrox implements the rule once, in
`crates/ferrox-models/src/rope_layers.rs`, shared with `exaone4`,
`exaone-moe`, `smallthinker`, `afmoe` and `llama4`. `smollm3` is the row
where it is the ONLY thing: the other five each carry the rule plus
something else, so this is the fixture that isolates it.

**FOUR LAYERS ARE NOT AN ACCIDENT, and neither is the PHASE.** The step
is 4 and llama.cpp skips `(il + 1) % 4 == 0`, so layer 3 is the
unrotated one and layers 0-2 rotate. A three-layer fixture would rotate
everything and prove nothing; the OTHER phase (`il % 4 == 0`, which is
`smallthinker.cpp:109`) would skip layer 0 instead, and on this file the
two answers differ at both ends.

No sliding window, no QK-norm, no biases: `smollm3.cpp` creates none of
them, and a fixture that carried them would be evidence about a
different graph.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_smollm3_fixture.py OUT.gguf

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "smollm3"

# Four, so layer 3 is the unrotated one. See the docstring.
N_LAYER = 4
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x5A0113)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-smollm3-fixture")
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

        # Drawn wider than the rest of the file so the softmax over a
        # six-token prompt is not near-uniform: a near-uniform attention
        # cannot see whether a layer rotated.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smollm3-fixture.gguf")
