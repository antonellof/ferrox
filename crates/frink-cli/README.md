# frink-cli

llama.cpp-style CLI for Frink (`frink` binary).

Completion (`-m` / `-p` / `-n` / …), chat templates, inspect / presets /
smoke helpers, and `frink bench`. See [`docs/CLI.md`](../../docs/CLI.md).

`frink serve` runs the OpenAI-compatible HTTP API from this binary. It
needs `--features serve`, which is off by default because it pulls in 98
crates a completion-only install does not need. Built without it, the
subcommand still exists and says so rather than reporting an unknown
subcommand. The prebuilt release binary is built with it. The standalone
[`frink-server`](https://crates.io/crates/frink-server) is the same
server through the same code.
