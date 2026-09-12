#!/usr/bin/env python3
"""Generate the tiny `llama` base GGUF and the two LoRA adapter GGUFs
used by ferrox's LoRA coverage test.

The adapters are NOT written by this script. It writes a PEFT adapter
directory (`adapter_config.json` + `adapter_model.safetensors`, the
files `peft` itself saves) and a `config.json` for the base, and then
runs llama.cpp's own `convert_lora_to_gguf.py` over them. So the
adapter GGUFs carry upstream's format -- `general.type = "adapter"`,
`adapter.type = "lora"`, `adapter.lora.alpha`, `<base>.lora_a` /
`<base>.lora_b` with the converter's own transposes and the llama
Q/K permutation applied to `lora_b` -- and not a spelling of it that
ferrox's loader and ferrox's fixture happen to agree on.

Two adapters, because `--lora` is repeatable and the server's
per-request `lora: [{id, scale}]` list addresses them by index:

  * `lora_a_tiny.gguf`, rank 4, alpha 8, targets EVERY projection
    llama.cpp's `build_lora_mm` can see on this graph -- Q, K, V, O,
    gate, up, down, plus `token_embd` (the `lora_embedding_A/B` pair,
    which the converter TRANSPOSES, `convert_lora_to_gguf.py:521-523`)
    and `output` (lm_head).
  * `lora_b_tiny.gguf`, rank 2, alpha 2, targets Q and V only, the
    PEFT default.
  * `lora_e_tiny.gguf`, rank 2, alpha 4, targets the token embedding
    ONLY: the pair a tied output head cannot take.

Weights are pseudo-random from fixed seeds so the files are byte-stable.
Both A and B are drawn non-zero (PEFT initialises B to zero, which
would make the adapter invisible and the test unable to fail).

Usage:
    python3 scripts/make_lora_fixture.py \\
        crates/ferrox-models/tests/fixtures $LLAMA

(`$LLAMA/gguf-py` is put on `sys.path` from the second argument, so no
`PYTHONPATH` is needed.)

writes `lora_base_tiny.gguf`, `lora_base_tied_tiny.gguf` (no
`output.weight`), `lora_a_tiny.gguf`, `lora_b_tiny.gguf` and
`lora_e_tiny.gguf` into the first argument. The golden logits that go with them are
produced by llama.cpp itself (`scripts/gptoss_reference_logits.cpp
--lora`), not by this script.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

if len(sys.argv) == 3:
    sys.path.insert(0, os.path.join(sys.argv[2], "gguf-py"))

import gguf  # noqa: E402

ARCH = "llama"

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = N_EMBD // N_HEAD  # 6
N_FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def write_base(out_path: str, tied: bool = False) -> None:
    rng = np.random.default_rng(0x10A4)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-lora-base-fixture")
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

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
    if not tied:
        w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


# The HF `config.json` the converter needs for the base: only the
# hyper-parameters, never the weights (`--base` is documented that way).
BASE_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": N_EMBD,
    "intermediate_size": N_FF,
    "num_hidden_layers": N_LAYER,
    "num_attention_heads": N_HEAD,
    "num_key_value_heads": N_HEAD_KV,
    "head_dim": HEAD_DIM,
    "vocab_size": N_VOCAB,
    "max_position_embeddings": CTX,
    "rms_norm_eps": RMS_EPS,
    "rope_theta": ROPE_BASE,
    "tie_word_embeddings": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "torch_dtype": "float32",
}


def write_peft_adapter(dir_path: str, seed: int, rank: int, alpha: float, targets: list[str]) -> None:
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM
    # HF name -> (n_out, n_in)
    shapes = {
        "self_attn.q_proj": (n_embd_q, N_EMBD),
        "self_attn.k_proj": (n_embd_kv, N_EMBD),
        "self_attn.v_proj": (n_embd_kv, N_EMBD),
        "self_attn.o_proj": (N_EMBD, n_embd_q),
        "mlp.gate_proj": (N_FF, N_EMBD),
        "mlp.up_proj": (N_FF, N_EMBD),
        "mlp.down_proj": (N_EMBD, N_FF),
    }
    tensors: dict[str, np.ndarray] = {}
    for il in range(N_LAYER):
        for short, (n_out, n_in) in shapes.items():
            mod = short.split(".")[-1]
            if mod not in targets:
                continue
            key = f"base_model.model.model.layers.{il}.{short}"
            tensors[key + ".lora_A.weight"] = rnd(rank, n_in)
            tensors[key + ".lora_B.weight"] = rnd(n_out, rank)
    if "embed_tokens" in targets:
        # PEFT's embedding LoRA: A is [rank, n_vocab] (a row per token
        # is gathered from A^T), B is [n_embd, rank].
        key = "base_model.model.model.embed_tokens"
        tensors[key + ".lora_embedding_A"] = rnd(rank, N_VOCAB)
        tensors[key + ".lora_embedding_B"] = rnd(N_EMBD, rank)
    if "lm_head" in targets:
        key = "base_model.model.lm_head"
        tensors[key + ".lora_A.weight"] = rnd(rank, N_EMBD)
        tensors[key + ".lora_B.weight"] = rnd(N_VOCAB, rank)

    os.makedirs(dir_path, exist_ok=True)
    save_file(tensors, os.path.join(dir_path, "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "ferrox-lora-base-fixture",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": targets,
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(dir_path, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def convert(llama_dir: str, base_dir: str, adapter_dir: str, out_path: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(llama_dir, "gguf-py") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            os.path.join(llama_dir, "convert_lora_to_gguf.py"),
            "--base",
            base_dir,
            "--outfile",
            out_path,
            "--outtype",
            "f32",
            adapter_dir,
        ],
        check=True,
        env=env,
    )
    print(f"wrote {out_path}")


def main(out_dir: str, llama_dir: str) -> None:
    write_base(os.path.join(out_dir, "lora_base_tiny.gguf"))
    # The same base with NO `output.weight`, so the head is the
    # embedding: what `lora_a_tiny.gguf`'s embedding pair is refused
    # against (llama.cpp aborts in `ggml_mul_mat` on it, measured).
    write_base(os.path.join(out_dir, "lora_base_tied_tiny.gguf"), tied=True)
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = os.path.join(tmp, "base")
        os.makedirs(base_dir)
        with open(os.path.join(base_dir, "config.json"), "w") as f:
            json.dump(BASE_CONFIG, f, indent=2)

        adapter_a = os.path.join(tmp, "adapter_a")
        write_peft_adapter(
            adapter_a,
            seed=0x10A4A,
            rank=4,
            alpha=8.0,
            targets=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "embed_tokens",
                "lm_head",
            ],
        )
        convert(llama_dir, base_dir, adapter_a, os.path.join(out_dir, "lora_a_tiny.gguf"))

        adapter_b = os.path.join(tmp, "adapter_b")
        write_peft_adapter(adapter_b, seed=0x10A4B, rank=2, alpha=2.0, targets=["q_proj", "v_proj"])
        convert(llama_dir, base_dir, adapter_b, os.path.join(out_dir, "lora_b_tiny.gguf"))

        # Embedding only: the pair the tied-head refusal is about.
        adapter_e = os.path.join(tmp, "adapter_e")
        write_peft_adapter(adapter_e, seed=0x10A4E, rank=2, alpha=4.0, targets=["embed_tokens"])
        convert(llama_dir, base_dir, adapter_e, os.path.join(out_dir, "lora_e_tiny.gguf"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
