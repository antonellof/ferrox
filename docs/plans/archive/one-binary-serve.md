---
name: one binary, `frink serve` behind a feature
overview: "GOAL: stop shipping two binaries for one product. `frink` and `frink-server` are separate executables today, and a user reasonably asks why. Fold the server into the CLI as a `frink serve` subcommand gated on an OPTIONAL `serve` Cargo feature, so the download path gets one binary while `cargo install frink-cli` stays small. THE MEASUREMENT THAT DECIDES THE DESIGN (taken 2026-08-20 on Host B, release build): frink 7.8 MB / 78 crates, frink-server 14 MB / 175 crates, and the server's dependency set is a strict SUPERSET of the CLI's, diffing them, the server adds 98 crates and the CLI adds exactly one (`frink-cli` itself). So an unconditional merge costs the server user nothing and costs the CLI-only user +98 crates, +6 MB, and a C toolchain, because `aws-lc-sys` (C and assembly, via rustls via axum-server and ureq) enters the tree. That asymmetry is why this is a feature flag and not a plain merge. NOT A GOAL: deleting the `frink-server` crate, it is published on crates.io as of v0.8.0 and stays as a thin shim."
todos:
  - id: server-lib-target
    content: "DONE. `crates/frink-server` now has a `[lib]` target: the module tree, `ServerArgs` and the entry point live in `lib.rs`, `main.rs` is a four-line shim over `frink_server::run_server(ServerArgs::parse_llama_style(std::env::args()))`. Public surface is four items and no `pub mod`: `ServerArgs` (private fields, `PartialEq` so parity can be asserted), `run_server`, `BUILT_WITH_METAL`/`BUILT_WITH_CUDA`. All 225 existing tests moved with the file, unedited, and pass. NOTE ON PROVENANCE: the move itself was written by an earlier session of this plan (488ca4b, 2026-08-21) whose worktree then went idle without ever being merged. frink-server's sources had not changed on main in between, so this branch MERGED that commit rather than rebasing or redoing it, and everything below sits on top of it. Two behaviour bugs the move introduced were found and fixed here, not there: `tracing_subscriber::fmt::init()` PANICS on a second install and frink-cli installs one before it parses argv, so `frink serve` died inside `run` after accepting the command line (now `try_init`, with a test that calls the installer twice); and the inline instance registration is now `claim_instance(&args)` with a test that reads the record back, because 'a serving process calls itself `server`' had no coverage at all"
    status: completed
  - id: serve-feature
    content: "DONE. `serve = [\"dep:frink-server\"]` on frink-cli, default OFF, with the rationale copied into the manifest so the next reader does not 'fix' it. Dependency counts on THIS host (`cargo tree -e normal`, unique crates, not a benchmark): default 79, `--features serve` 168, and the default tree contains no frink-server, axum, tokio, rustls or ureq. aws-lc-sys is absent from BOTH, since tls-backend-cost already removed it. The feature-OFF build still has a `serve` subcommand: it swallows the server's flags and fails with a sentence naming the fix (`cargo install frink-cli --features serve`), because clap's 'unrecognized subcommand' reads like the feature does not exist rather than like it was compiled out. ADDED BEYOND THE TODO: `cuda`/`metal` on frink-cli forward with the weak-dep syntax (`frink-server?/metal`), or one binary answers two ways -- `frink run --device metal` uses the GPU while `frink serve --device metal` refuses. Deleting that forwarding compiles cleanly, so `const _: () = assert!(frink_server::BUILT_WITH_METAL, ..)` fails the build instead; verified by deleting it (E0080 naming the manifest fix) and by /health reporting `metal: available` from a `--features serve,metal` build"
    status: completed
  - id: serve-subcommand-parity
    content: "DONE, and structurally rather than by transcription: `Commands::Serve(frink_server::ServerArgs)` EMBEDS the struct, so there is one definition of the server's command line and no second list to drift. Nothing was ported from docs/CLI.md and no --ui-server/FRINK_UI was resurrected. Three tests hold it: the flag-set of `frink serve` is a superset of `ServerArgs::command()`'s (mutation-checked by reimplementing serve as a hand-rolled 3-flag struct, which fails naming the seven missing flags); a full 10-flag command line including `--port 0` parses to a ServerArgs EQUAL to the one frink-server's own argv path produces (mutation-checked by renaming --mcp-config, which fails); and `--list-devices` alone, which exits before serving and so no smoke test would catch it. `-ngl`/`-dev` survive inside `frink serve` through the CLI's own rewriter. The ready line is byte-identical: `frink serve --port 0` and `frink-server --port 0` were both started on this host and both printed `{\"event\":\"frink.server.ready\",..}` followed by an identical /health body"
    status: completed
  - id: argv-rewriter-collision
    content: "DONE, test first and the hazard confirmed before the fix: with `serve` absent from SUBCOMMANDS, `frink serve -m model.gguf` really is rewritten to `[frink, run, serve, -m, model.gguf]` (observed as a test failure). `SUBCOMMANDS` is now module-level and asserted against clap's OWN subcommand list, so the next subcommand cannot repeat this: removing `serve` fails three tests, one of them with 'subcommand `serve` is missing from SUBCOMMANDS'. Tested in both builds -- the feature-off build declares the subcommand too, so its rewriter behaviour is identical and its failure is the rebuild hint. FOUND A SECOND, PRE-EXISTING CASE while testing: the rewriter looked only at argv[1], so a global flag before the subcommand (`frink --allow-multiple-instances serve -m x`) became `frink run --allow-multiple-instances serve …` and died on 'unexpected argument serve'. That broke `bench` and `verify` the same way and has nothing to do with serve. Fixed by skipping known value-less root flags (GLOBAL_FLAGS) before deciding. The flag's VALUE reaching the server needs nothing from us: clap propagates the root global into ServerArgs's identically-named flag, which a mutation check proved (a merge helper written for it was deleted when the test passed with the merge disabled) and which a test now pins, since it is clap's behaviour and not ours"
    status: completed
  - id: tls-backend-cost
    content: "DONE, and it was one feature flag. `aws-lc-sys` (C + assembly) had exactly one source: axum-server's `tls-rustls`, which is defined as `[\"tls-rustls-no-provider\", \"rustls/aws-lc-rs\"]`. ureq was innocent, it already asks rustls for `ring` with default-features off, but cargo FEATURE UNIFICATION meant axum-server's choice was imposed on ureq's rustls too, which is why `cargo tree -i aws-lc-sys` listed ureq as a parent and made it look like two independent sources. Switched to `tls-rustls-no-provider` + an explicit `rustls` dep on `ring`, and `install_ring_crypto_provider()` runs unconditionally at startup because `no-provider` moves the failure to ACCEPT time. Measured on Host B: 175 -> 173 crates, release binary 14 MB -> 13 MB, aws-lc-rs/aws-lc-sys gone entirely (`cargo tree -i aws-lc-sys` now errors with no match). aws-lc-rs alone took 18.0s to compile. FOUND AND FIXED A SEPARATE PRE-EXISTING BUG while smoke-testing this: the TLS arm passed a BLOCKING `std::net::TcpListener` to `axum_server::from_tcp_rustls`, and tokio panics on registering one. TLS was therefore completely broken on main, it bound, printed its `frink.server.ready` line, then panicked on first accept, so it looked like a healthy start followed by a server that answered nothing. `set_nonblocking(true)` before handover. Verified end to end: `GET /health` over HTTPS returns 200 on TLS 1.3 with AEAD-CHACHA20-POLY1305, which is a ring suite and so also proves the provider is live"
    status: completed
  - id: server-shim-crate
    content: "DONE. crates/frink-server/src/main.rs is a 10-line shim over frink_server::run_server(ServerArgs::parse_llama_style(std::env::args())), everything else moved to lib.rs, and the crate keeps publishing so `cargo install frink-server` still produces the same executable. Both front ends parse the same argv through the same code, so there is no second startup path to drift."
    status: completed
  - id: release-and-install-path
    content: "DONE, option (a) with the transition kept. .github/workflows/release.yml builds frink-cli with `serve` (plus the platform's GPU feature), so the downloaded `frink` runs completions AND serves. frink-server is still built and still shipped in the tarball, so an existing one on a PATH keeps working after an upgrade rather than being shadowed by a binary that cannot serve. FOUND BY THE MAINTAINER READING THE README: the install text said the tarball contains both binaries, which was true, and left the reader to assume the `frink` in it could serve. It could not, because release.yml built the CLI without the feature. The one-binary story held for `cargo install` and broke for the download path most people use."
    status: completed
  - id: one-instance-registry
    content: "DONE, and the answer is that exactly one side must claim. `frink_server::run_server` registers as `server` before it binds, so frink-cli must NOT also register for `serve` -- both guards name the same per-pid file, and the inner one's Drop would delete the entry while the server was still holding the weights, making a live server invisible to the next `frink run`. The classification is now `instance_target(&Commands)`, tested: `serve` returns None while `frink -m model.gguf` still returns `run` (so the test cannot be satisfied by returning None for everything); mutation-checked by adding a `Serve => Some((\"serve\", ..))` arm, which fails it. On the server side `claim_instance(&args)` is extracted and tested against a temp FRINK_INSTANCE_DIR: the record's command field is `server` (mutation-checked by writing `run`, which fails with the raw record in the message) and the guard's Drop removes the file. Nothing about the policy changed, so the benchmarking rationale in docs/CLI.md still holds as written: a serving instance and a completion run are still two model holders and still refuse each other under Single"
    status: completed
  - id: docs-single-binary
    content: "DONE. README shows `frink serve` in quick start and lists all three install shapes (`frink-cli`, `frink-cli --features \"serve metal\"`, `frink-server`) with the reason `serve` is default-off next to them: it pulls in 98 crates the CLI does not otherwise need, including a C crypto library. docs/CLI.md's Server section covers both entry points and states that both parse identical arguments through the same code. docs/AGENTS_COOKBOOK.md opens with both ways to start a server. The feature-off build's error names the rebuild rather than reporting an unknown subcommand."
    status: completed
  - id: hygiene-cross-target-gate
    content: "HALF LANDED (branch wf/tooling), and which half matters. ADDED 2026-08-24 after it broke two releases in one week: CI tracks `stable` and a developer checkout drifts behind it, so a stable bump turns a green branch red at TAG time with lints nobody ran (1.97 brought manual_checked_ops and explicit_counter_loop; 1.98 rejected chunks_exact with a compile-time-known size, 58 errors). The 1.98 fix passed every local gate and STILL broke CI, because the sites it got wrong live in the scalar fallbacks compiled only when the NEON paths are not: on an Apple-silicon laptop `cargo build --workspace` never sees them. (1) DONE: `scripts/check-cross-target.sh` runs `cargo clippy --workspace --all-targets --target x86_64-apple-darwin -- -D warnings`, and CONTRIBUTING.md gives the one line that calls it from the local pre-push hook. The hook file itself is deliberately NOT checked in: `.githooks/` is gitignored as local-only, this checkout already has a pre-push there that scrubs machine usernames, and shipping a second one would have replaced it. x86_64-apple-darwin is the cheap second architecture: no cross linker, no sysroot, same libc, and it selects exactly the paths an aarch64 build skips. Measured 24s cold and ~2s warm on this host, and it passes on current main. clippy is run rather than clippy AND check, since clippy runs the compiler front end and doing both compiles the workspace twice to learn one thing. Note this is NOT a check CI lacks -- CI's Linux clippy job compiles the same fallbacks -- it moves the discovery one round trip earlier, which is the whole complaint. (2) STILL OWED: the rust-toolchain.toml pin, so CI and every developer share a compiler and lint set. It was attempted once and BACKED OUT after rustup fought the cargo processes several agents were holding and left two toolchains half-installed, and this branch was written with four other agents building on the same box, so it was not attempted again. It needs a quiet machine and its own commit with the resulting fixes"
    status: pending
  - id: hygiene-generate-tmp
    content: "DONE. `crates/frink-server/src/generate.rs.tmp`, a 7-byte checked-in stray, deleted"
    status: completed
isProject: false
---

# One binary, `frink serve` behind a feature

> Prompted by a fair question: *why two binaries for one product?*
> The answer below is measured, not argued: the merge is worth doing,
> but not unconditionally, and the reason is a C compiler.

## The measurement

Host B, release build, 2026-08-20:

| | `frink` | `frink-server` |
|---|---|---|
| Binary | 7.8 MB | 14 MB |
| Unique crates | 78 | 175 |

The number that decides the design is the **diff**, not the totals:

- crates in `frink-server` but not `frink`: **98**
- crates in `frink` but not `frink-server`: **1** (`frink-cli` itself)

The server's dependency set is a strict superset of the CLI's. So a
plain merge is **asymmetric**: it costs a server user nothing and costs a
CLI-only user everything. Concretely, +98 crates, +6 MB, and this:

```
aws-lc-sys v0.43.0
└── aws-lc-rs v1.17.3
    ├── rustls v0.23.43
    │   ├── axum-server v0.8.0  ← frink-server
    │   └── tokio-rustls v0.26.4 ← axum-server
    └── ureq v2.12.1            ← frink-server
```

`aws-lc-sys` is a C-and-assembly crypto library. Merging unconditionally
means somebody who types `cargo install frink-cli` to run one prompt
compiles a TLS stack, and needs a C toolchain to do it. That is a worse
first experience than the two binaries they were confused by.

Hence: **a feature, not a merge.**

## The shape

```bash
cargo install frink-cli                    # 78 crates, no C toolchain, no `serve`
cargo install frink-cli --features serve   # one binary, does both
```

The prebuilt tarballs and the install script ship the `--features serve`
build, so for the majority of users, who download rather than compile, there is exactly one binary and the original question disappears.

This fits what `frink` already is. It has fifteen subcommands, and one
of them (`chat --url`) is already a *client* of the server. A `serve`
sibling is the natural completion of that surface, not a new idea bolted
onto it.

### `serve-default-off-rationale`

The obvious choice is to default the feature ON so `cargo install
frink-cli` "just works". It is the wrong one, for a reason that only
shows up in someone else's build log: a default-on feature is what a
dependent gets unless they remember `default-features = false`, so every
crate that ever depends on `frink-cli` inherits a TLS stack it did not
ask for. Default off, documented loudly, and shipped on in the artifacts
people actually download.

## What could go wrong, ranked

1. **The argv rewriter silently swallows `serve`.** `frink` accepts
   llama.cpp-style flags by rewriting argv before clap sees it, keyed off
   a hardcoded `SUBCOMMANDS` list (`crates/frink-cli/src/main.rs:344-361`).
   Miss `serve` there and `frink serve -m model.gguf` is rewritten into
   an implicit `run`, it would start a *completion* instead of a server,
   print tokens, and exit. No error, no clue. This is the one that needs
   a test before it needs code.
2. **Flag parity quietly regressing.** `ServerArgs`
   (`crates/frink-server/src/main.rs:90-160`) carries eleven flags;
   `docs/CLI.md:231-233` documents eight and reads as exhaustive. A
   reimplementation that follows the docs instead of the struct ships a
   downgrade. Port from the struct.
3. **A stale `frink-server` shadowing the new binary.** If the tarball
   stops shipping it, an upgrading user still has the old one on PATH.
   `scripts/install.sh:91` loops `for bin in frink frink-server`. The
   symptom would be reported as "`--ui-server` does nothing".
4. **The one-instance registry misfiring.** `frink` refuses to be the
   second process holding a model
   (`crates/frink-cli/src/main.rs:387-423`). Under one binary a server
   and a completion run are the same executable; the registry must still
   tell them apart.

## What this is not

Not a deletion of `frink-server`. It was published to crates.io at
v0.8.0 on 2026-08-20, and it is named in README, five files under
`docs/`, `scripts/install.sh`, and all three GitHub workflows. It stays
as a thin shim over the library target for at least two minor versions,
with the deprecation stated in its crate description rather than
discovered.

## Order of work

1. `server-lib-target`, mechanical, unblocks everything, no behaviour change.
2. `tls-backend-cost`, independent, and it shrinks the worst case for
   everything after it. Worth doing even if the rest is abandoned.
3. `serve-feature` + `serve-subcommand-parity` + `argv-rewriter-collision`.
   The actual change, and the rewriter test comes first.
4. `one-instance-registry`, `server-shim-crate`.
5. `release-and-install-path`, then `docs-single-binary` once the shape
   is settled rather than while it is moving.

## Definition of done

- `cargo install frink-cli` builds with no C toolchain and has no
  `serve` subcommand, and says how to get one.
- `cargo install frink-cli --features serve` serves an API identical to
  today's `frink-server`, including the `frink.server.ready` stdout
  line supervisors parse.
- `cargo install frink-server` still works.
- The release tarball's story about how many binaries it contains
  matches what it actually contains.
- Crate count and binary size recorded before and after, on the same
  host, in the same session.
