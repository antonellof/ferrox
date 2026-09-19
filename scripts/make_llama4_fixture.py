#!/usr/bin/env python3
"""Generate the tiny synthetic `llama4` GGUFs used by frink's Llama 4
coverage test (`crates/frink-models/tests/llama4_graphs.rs`).

`llama4` is Llama 4 Scout (16 experts) / Maverick (128 experts) and the
MobileLLM dense series (no experts). `.scratch/llama.cpp/src/models/
llama4.cpp`:

  * `:5-21`: `attention.sliding_window` PRESENT AND ZERO switches the
    whole chunked machinery off (`swa_type NONE`, RoPE on every layer);
    otherwise `swa_type CHUNKED` with `n_swa = 8192` FROM A LITERAL
    (the file's value is ignored), the attention-temperature constants
    `0.1 / 8192 / 1.0` from literals, a 3-chunked-1-full period of 4
    (`attention.sliding_window_pattern` may override) and
    `rope.freq_base_swa` seeded from the model's base.
  * `:43`: `use_kq_norm` unless the model is the 128-expert one: a
    WEIGHTLESS RMS norm per head on Q and K (`:182-186`), after RoPE,
    on the layers that rotate.
  * `:64`: layer `i` is MoE iff `(i + 1) % interleave_moe_layer_step ==
    0` -- the LOADER honours the step, unlike ERNIE's (`crate::
    moe_interleave`). A MoE layer routes SIGMOID with `norm_w = false`
    (`:228-230`) over `n_expert` experts and adds a shared expert
    sized `expert_feed_forward_length` (`:86-89,234-241`).
  * `:111-112`: layer `i` rotates iff `(i + 1) % n_no_rope_layer_step
    != 0` (4, or `n_layer` on the no-SWA branch); the NoPE layers get
    the temperature instead (`:175-176`).
  * NOT in `llama4.cpp`: `llama-graph.cpp:1947` (`weight_before_ffn =
    arch == LLM_ARCH_LLAMA4`) multiplies the sigmoid weight into the
    expert's INPUT before `mul_mat_id` where every other graph
    multiplies the output, and `:1999` selects the top-k on the raw
    logits. The router is written at unit scale so the top-1 sigmoid
    sits well below 1 and both `norm_w = false` and the weight's site
    are visible in the logits (at `* 4.0` the weight was 0.999996 and
    neither was).

Layers (four): with the default keys, layers 0-2 are chunked and
rotated, layer 3 is full attention with no RoPE and the temperature;
with `interleave_moe_layer_step 2`, layers 1 and 3 are MoE and 0 and 2
dense.

Variants:
  * default          16 experts (Scout's `use_kq_norm`), step 2, no window key
  * `--noswa`        `sliding_window 0` (what the converter writes when
                     every layer is full attention, `conversion/llama.py:
                     391-394`): no chunking, RoPE everywhere, no temperature
  * `--128e`         128 experts (Maverick): `use_kq_norm` off
  * `--dense`        no experts (MobileLLM's hparams type): a file
                     libllama REFUSES (`llama4.cpp:49-51`, "model cannot
                     have zero experts", measured), so frink refuses it too
  * `--output`       a separate `output.weight`

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_llama4_fixture.py OUT.gguf [--noswa | --128e | --dense] [--output]

Weights are pseudo-random from a fixed seed so the files are byte-stable.
The golden logits that go with them are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "llama4"

N_EMBD = 24
N_HEAD = 4
N_KV = 2
HEAD_DIM = N_EMBD // N_HEAD
N_FF = 40
N_FF_EXP = 16
N_VOCAB = 48
N_LAYER = 4
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5
STEP = 2
N_EXPERT_USED = 1


def main(out_path: str, dense: bool, noswa: bool, experts: int, separate_output: bool) -> None:
    rng = np.random.default_rng(0x11A4)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    n_expert = 0 if dense else experts
    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-llama4-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    # llama4.cpp:6 reads the step as REQUIRED, dense or not
    # (conversion/llama.py:389 writes it for every export).
    w.add_interleave_moe_layer_step(STEP)
    w.add_expert_feed_forward_length(N_FF_EXP)
    if noswa:
        # conversion/llama.py:391-394: every layer full attention.
        w.add_sliding_window(0)
    if not dense:
        w.add_expert_count(n_expert)
        w.add_expert_used_count(N_EXPERT_USED)
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

    for il in range(N_LAYER):
        p = f"blk.{il}."
        is_moe = n_expert > 0 and (il + 1) % STEP == 0
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "attn_q.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(N_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(N_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
        w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if is_moe:
            # Unit scale: the top-1 sigmoid weight then sits well below 1, so
            # `norm_w = false` (llama4.cpp:228) is visible in the logits.
            w.add_tensor(p + "ffn_gate_inp.weight", rnd(n_expert, N_EMBD))
            w.add_tensor(p + "ffn_gate_exps.weight", rnd(n_expert, N_FF_EXP, N_EMBD))
            w.add_tensor(p + "ffn_up_exps.weight", rnd(n_expert, N_FF_EXP, N_EMBD))
            w.add_tensor(p + "ffn_down_exps.weight", rnd(n_expert, N_EMBD, N_FF_EXP))
            w.add_tensor(p + "ffn_gate_shexp.weight", rnd(N_FF_EXP, N_EMBD))
            w.add_tensor(p + "ffn_up_shexp.weight", rnd(N_FF_EXP, N_EMBD))
            w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, N_FF_EXP))
        else:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
    if separate_output:
        w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} (experts={n_expert}, noswa={noswa}, output={separate_output})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "llama4-fixture.gguf",
        "--dense" in sys.argv[1:],
        "--noswa" in sys.argv[1:],
        128 if "--128e" in sys.argv[1:] else 16,
        "--output" in sys.argv[1:],
    )
