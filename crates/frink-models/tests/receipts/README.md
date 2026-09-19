# Checkpoint receipts (pending)

Real-checkpoint oracle receipts for Qwen2-MoE / Mistral / Mixtral /
Gemma-2 follow the Llama-3.1 pattern in
[`llama31_8b_instruct_q4_k_m.json`](llama31_8b_instruct_q4_k_m.json).

To add a receipt once a local GGUF is available:

1. Compute `shasum -a 256` and byte size of the file.
2. Capture a forced-continuation prompt + token IDs via
   `frink inspect` / tokenizer roundtrip.
3. Cross-check top-1 / greedy decode against llama.cpp on the same file.
4. Drop `<id>.json` here and add an `#[ignore]` test in
   `checkpoint_receipts.rs` that sets `FRINK_TEST_RECEIPT_CHECKPOINT`.

Suite entries already exist in `benchmarks/suite.json`:

| Suite id | Arch focus |
|---|---|
| `qwen2moe_05b_q4km` | QKV bias + shared_expert_gate |
| `mistral_7b_q4km` | SWA dense |
| `mixtral_8x7b_q4km` | Grouped MoE + SWA |
| `gemma2_2b_q4km` | Attn + final logit softcap |
| `olmoe_q4` | MoE Metal expert placement |

Until a receipt JSON lands, MODELS.md keeps these under **Partial**.
