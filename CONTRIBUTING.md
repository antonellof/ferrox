# Contributing

## Workflow

Run before opening a PR (CI enforces all of these with `-D warnings`):

```
cargo fmt --all
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

On an Apple-silicon checkout those four are not enough, because several
kernels exist twice: a NEON path under `#[cfg(target_arch = "aarch64")]`
and a portable scalar fallback the local build never compiles. A lint
that fires only in the fallback is invisible here and red on CI, which
is how a Rust 1.98 fix passed every local gate and still broke the
build. Compile for a second architecture before pushing:

```
scripts/check-cross-target.sh    # clippy for x86_64-apple-darwin, ~2s warm
rustup target add x86_64-apple-darwin   # once, if it refuses
```

To run it automatically, call it from your local pre-push hook. That
directory is gitignored and may already hold a hook of its own, so add
the line rather than replacing the file:

```
echo 'exec "$(git rev-parse --show-toplevel)/scripts/check-cross-target.sh"' \
  >> .githooks/pre-push
```

CI's Linux `clippy` job compiles the same fallbacks, so this is about
finding them a round trip earlier, not about a check CI lacks.

The `cuda` feature must keep compiling without a GPU or CUDA toolkit
present: `cargo clippy -p frink-cli -p frink-server --features cuda`.
CI also builds that CLI/server chain and, on `macos-latest`, compiles
`cargo clippy -p frink-metal -p frink-cli --features metal`. Hardware
kernel tests stay `#[ignore]`d on hosted CI.

CI builds `frink-cli --features serve` too, which is what the release
tarball ships: `serve` on Linux and `serve metal` on macOS. That gate
was added after noticing a break confined to `serve` could reach users
through the install script with CI green.

The `ui/` job runs `npm run licenses`, `typecheck`, `lint` and `build`.
It does not run `npm test`, so run `npm run check` in `ui/` before a PR
that touches `ui/src/lib/`.

## Benchmarks & presets

- Preset fields: confirm each one against a primary source, or list it
  in `best_effort_fields`.
- New quant kernels need independent goldens, not only self-parity.
- A hardware claim names the machine it was measured on, or says
  compile-tested only.
- Speed numbers come from `frink bench --suite` against `llama-bench`,
  with no HTTP in the loop. `benchmarks/suite.json` drives the runs and
  `benchmarks/RESULTS.md` is generated from them, so never edit that
  table by hand. To measure a new model, add an entry to `suite.json`,
  put the GGUF under `models/`, then run
  `frink bench --suite --id <id> --fit-host --skip-missing`.
- A run that stops before timing anything is telling you the host is
  busy, hot, or short on free memory. Fix the host rather than reaching
  for `--max-load 0`, which turns off all three checks and produces a
  number nobody may publish. `benchmarks/README.md` has each one.
- Never force a thread count on either engine. llama.cpp defaults to
  performance cores and loses 2-4x above them, so pinning both to the
  same count flatters frink instead of making the comparison fair.
- Run-to-run spread on Apple Silicon is around 20%. A claim under that
  needs an interleaved A/B: alternate the two binaries round by round in
  one session and count rounds won. Two batches of runs will not do it.
- Commit the negative results too. `.scratch/NOTES_LLAMA_*.md` records
  what was tried and did not work, which is as useful as the wins.

## Documentation

| Doc | Role |
|---|---|
| `docs/FEATURES.md` | Capabilities overview |
| `docs/CLI.md` | CLI flags and examples |
| `docs/MODELS.md` | Supported models |
| `docs/API.md` | HTTP API matrix |
| `docs/CONFIG.md` | Environment variables |
| `docs/AGENTS_COOKBOOK.md` | Pointing an IDE or agent at the server |
| `benchmarks/RESULTS.md` | Speed vs llama.cpp (generated) |
| `benchmarks/README.md` | How those numbers are measured |
| `docs/ROADMAP.md` | Planned work |

Do not duplicate those. Plans belong in `ROADMAP.md`, and git holds the
history. Keep counts that go stale fast (test totals and the like) out
of prose.
