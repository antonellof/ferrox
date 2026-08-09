

<p align="center">
  <img src="docs/assets/ferrox-logo.png" alt="Ferrox — pure-Rust GGUF / MoE inference" width="520">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
</p>

**Ferrox** is a pure-Rust inference engine for GGUF models. It runs dense
and MoE checkpoints on CPU, Apple Metal, or CUDA, with a llama.cpp-style
CLI and an OpenAI-compatible HTTP server.

## Quick start

```bash
cargo build --release -p ferrox-cli -p ferrox-server --features metal
```

### 1. Download a model

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
(`pip install -U huggingface_hub`), then:

```bash
mkdir -p models
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q8_0.gguf --local-dir models
```

Other useful GGUFs:

| Model | Repo | File |
|---|---|---|
| TinyLlama 1.1B Chat Q8_0 | [TheBloke/…](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) | `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` |
| Llama 3.2 1B Instruct Q4_K_M | [bartowski/…](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` |
| Llama 3.1 8B Instruct Q4_K_M | [bartowski/…](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` |
| SmolLM2 135M Instruct Q8_0 | [bartowski/…](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF) | `SmolLM2-135M-Instruct-Q8_0.gguf` |

Browse [llama.cpp-compatible models](https://huggingface.co/models?apps=llama.cpp&sort=trending)
on Hugging Face. Prefer `Q4_K_M` for everyday use; `Q8_0` for tiny smokes.

### 2. Run the CLI

```bash
./target/release/ferrox -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

# Chat template + Metal
./target/release/ferrox -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p "What is 2+2?" -n 64 --temp 0 -dev metal -ngl all
```

### 3. Start the server

```bash
./target/release/ferrox-server \
  -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --host 127.0.0.1 --port 8383 -dev metal -ngl all &

curl -s -X POST http://127.0.0.1:8383/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"temperature":0}'
```

## Documentation

| Doc | Description |
|---|---|
| [docs/FEATURES.md](docs/FEATURES.md) | Capabilities overview |
| [docs/MODELS.md](docs/MODELS.md) | Supported models and benchmarks |
| [docs/CLI.md](docs/CLI.md) | CLI flags and examples |
| [docs/API.md](docs/API.md) | OpenAI-compatible API |
| [docs/CONFIG.md](docs/CONFIG.md) | Environment variables |
| [docs/AGENTS_COOKBOOK.md](docs/AGENTS_COOKBOOK.md) | Point IDEs / agents at the server |
| [benchmarks/RESULTS.md](benchmarks/RESULTS.md) | Speed vs llama.cpp (engine + serving) |
| [benchmarks/README.md](benchmarks/README.md) | How the two benchmark tracks are measured |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Planned work |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

## License

Apache-2.0 — see [LICENSE](LICENSE) and
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).
