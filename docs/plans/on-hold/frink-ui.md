---
name: frink UI — web, desktop, one frontend
overview: "GOAL: one web frontend, embedded in frink-server and served at `/`, plus a Tauri 2 shell that wraps the same frontend into a native app for macOS/Windows/Linux. First cut: chat + model manager + live metrics. Every UI feature must be reachable through the public HTTP API — the UI calls `/v1/chat/completions` like any other client, so the API cannot rot without the UI breaking first. Sourced from a read-only study of oMLX (Python/MLX, Swift menubar) and Unsloth Studio (React + FastAPI + Tauri 2) under `.scratch/`; both were read for contracts only. LICENCE RULE: Unsloth `studio/` is AGPL-3.0 and frink is Apache-2.0 — read the shapes, copy no code."
todos:
  - id: ui-api-contract-crate
    content: "Shared route-constant + DTO crate so app and server cannot drift (Unsloth has this implicitly; oMLX's Endpoints.swift is 106 lines and is the whole app-server contract). LANDED 2e8c214: `crates/frink-api` (serde-only) with routes / health / lifecycle / usage / request_id / progress. frink-server's router and Usage now come from it. Only implemented routes get a constant — a constant for a missing endpoint reads as a promise. The OpenAI request/response bodies deliberately stay in frink-server where they are validated"
    status: completed
  - id: ui-health-capability
    content: "GET /health with THREE states: ready, unavailable+reason, and `detecting` while backends are probed. A guessed 'CPU only' on first paint is visually identical to a measured one. Hard 1s budget, then answer provisionally. LANDED 21a175c: `/health` returns the handshake instead of the string 'ok'. Probe runs once in the background, handler never blocks; before the 1s budget no GPU verdict is offered at all, after it the answer is filled in with reason `detection_timed_out`, and a late probe replaces that. Capabilities cpu/metal/cuda/real_weights/continuous_batching each carry a machine reason + a human sentence, so metal_unavailable / metal_not_built / disabled stay three different answers. `last_request_age_seconds` lets a saturated backend vouch for its own liveness. NOW REACHABLE (1a59c15): POST /admin/models/unload leaves nothing loaded, and /health answers 503 `unavailable` with reason `model_not_loaded` and a null model, instead of the 200 `ready` a supervisor would have acted on"
    status: completed
  - id: ui-server-lifecycle
    content: "`--port 0` + a structured stdout line carrying the bound addr/pid, and stdin-close => exit. Deletes the entire port-conflict UI and all OS-specific signal reaping. LANDED ff8f10d: both serve paths (plain + TLS) bind first and read the address back off the socket, then print the `frink.server.ready` JSON line (addr/port/scheme/pid/version). `--exit-on-stdin-close` / FRINK_EXIT_ON_STDIN_CLOSE=1 is OPT-IN — a /dev/null stdin (systemd, cron, nohup) is EOF at startup, so defaulting it on would make the server exit as it starts. Graceful shutdown added to both paths; runtime is shut down in the background since the stdin watcher parks in a blocking read. Verified end to end. NOT DONE: the preflight disposition (managed/owned/attached/external) is supervisor-side and belongs with ui-tauri-shell"
    status: completed
  - id: ui-request-id-stream
    content: "Emit `request_id` in the first SSE chunk and key metrics by the same id. oMLX's chat page reverse-engineers this with a claiming heuristic; frink can just say it. LANDED 233328f: every /v1/chat/completions request gets a `chatcmpl-…` id (was the constant 'frink-demo-0' for everyone). Stated as `id` on every chunk and once as `request_id` on the first one, before any content. PARTIAL: metrics are not yet keyed by it — that needs ui-api-monitor's ring buffer"
    status: completed
  - id: ui-usage-timings
    content: "Extend `usage` with llama.cpp-style timings (ttft, prompt_eval_duration, generation_duration, tok/s both phases, cached_tokens) in both non-streaming and the include_usage final chunk. Server-reported beats client wall-clock. LANDED 233328f: prompt_eval_duration_ms / generation_duration_ms / time_to_first_token_ms / cached_tokens beside the existing per-phase rates, on the Decoder path AND the Kimi/MLA engine path (which reported no timings at all). TTFT is stamped inside the decode step closure — the real moment a token existed — not approximated as prefill time. cached_tokens distinguishes a prefix-cache miss (Some(0)) from no prefix cache (absent). Untimed usage still serializes to exactly the three OpenAI fields"
    status: completed
  - id: ui-cancel-two-tier
    content: "Cancellation: AbortSignal on the socket PLUS an explicit POST /cancel with keepalive. A dropped TCP connection does not reliably stop a decode loop. LANDED: both tiers end at one `CancelToken` (crates/frink-server/src/cancel.rs), so there is one stop path rather than two. TIER 1 was a real bug, not a hypothetical: the streaming path discarded the result of every `blocking_send`, so a closed browser tab left a CPU core decoding the remaining hundreds of tokens into a receiver nobody held; a failed send now flips the flag. TIER 2 is `POST /v1/cancel {request_id}` — under `/v1` rather than the root the plan sketched, because it acts on inference and belongs behind the same FRINK_API_KEY gate, and a root `/cancel` would sit inside the namespace the SPA fallback owns. It answers 200 `cancelled: true` only when a LIVE generation was signalled and 404 `cancelled: false` otherwise: 'stopped it' and 'there was nothing left to stop' are different facts, and a UI told ok for both would claim work it did not do. Registration is guard-scoped so a panicking decode thread cannot leak an id. A cancelled stream ends with finish_reason `cancelled` — not an OpenAI value, but *a* finish reason, so the plan's truncation rule still holds — and keeps the tokens it earned. The frontend does both tiers: Stop, New chat and leaving the Chat screen all abort AND POST; a `pagehide` listener POSTs with `keepalive: true`, which is the case an AbortSignal cannot cover. NOT DONE: the flag is read between decoded tokens, so a prefill already inside a forward pass still completes; continuous batching decodes on the shared batcher thread and is not covered; `/v1/completions` and `/v1/messages` are buffered and register no id"
    status: completed
  - id: ui-sse-hardening
    content: "SSE with id:/retry:/Last-Event-ID replay, a stall timeout, and a polling fallback beside every stream — reverse proxies buffer text/event-stream. Both reference projects hit this in production. LANDED, all four pieces, server and client together. Previously landed: `X-Accel-Buffering: no` on every streamed completion, and a client-side stall timeout armed against BYTES rather than tokens so the server's 15s keep-alive disarms it on a healthy-but-slow stream. NOW LANDED: `id:` / `retry:` / Last-Event-ID replay and the polling fallback. The blocker this item recorded was real and is resolved by a DESIGN DECISION rather than a patch: a dropped socket cancels the generation, so there was nothing to resume into — so resumability is now OPT-IN PER REQUEST (`stream_resumable: true`), and asking for it is also what makes a dropped socket stop cancelling. Neither answer suits both callers: a tab that navigated away wants the CPU back, a tab whose proxy dropped a 90-second answer wants the answer, so the request says which. `POST /v1/cancel` remains the stop path for a resumable stream, and it is the tier a page-unload keepalive POST can actually reach. Server (`crates/frink-server/src/resume.rs`): a bounded replay buffer per stream holding the exact `data:` payloads including `[DONE]`, numbered from zero; `GET /v1/stream/{id}` reconnects after a `Last-Event-ID`; `GET /v1/stream/{id}/poll?from=n` drains the same buffer over plain JSON and parks up to 10s for the next event, which is the actual answer to a proxy that buffers text/event-stream — it cannot buffer a short response that has already ended. Every edge fails closed: an evicted position is 410 rather than a stream with an undetectable hole in it, a `Last-Event-ID` naming another request is 400 rather than a confident replay of the wrong answer (ids are qualified by request id precisely because browsers resend the last id they saw), an unknown id is 404, and eviction never takes a live stream out from under its reader. Client (`ui/src/lib/api.ts`): reads `id:`/`retry:`, reconnects with `Last-Event-ID`, falls back to polling when the reconnect cannot be made, and goes STRAIGHT to polling when the stall detector fired — a stream that went quiet with the socket still open is a buffering proxy, and a second SSE connection would meet the same proxy. The stall timeout therefore no longer only reports; it can finish the answer another way without touching the generation. Chat says which transport is in use, because a silent reconnect leaves the user watching an indicator that stopped meaning anything. Tested server-side (8 HTTP-level tests) and client-side (7, `ui/src/lib/api.test.ts`, node's own test runner, no framework added). NOT DONE and stated: nothing verified in a browser against a live mid-generation reconnect — the tests drive the protocol, not a real proxy; a continuous-batching stream buffers its whole answer before emitting, so its replay buffer fills at the end and a mid-generation resume gets nothing until then; `/v1/completions` and Anthropic `/v1/messages` are buffered and have no stream to harden."
    status: completed
  - id: ui-admin-models
    content: "GET /admin/models: per model id, path, loaded/loading, on-disk + resident size, backend, context length, quant, and capability flags each PAIRED WITH A HUMAN-READABLE REASON so the UI never re-derives why a control is disabled. LANDED 1a59c15: header-only discovery over FRINK_MODEL_DIR + the FRINK_MODEL_PATH directory (GgufFile::open parses metadata and tensor descriptors, never a weight), non-recursive, split checkpoints folded into one entry, safetensors-index directories included. Every optional field is null-not-absent, because an unknown context length and a zero one are different facts. quant prefers general.file_type (the only place the _M in Q4_K_M is stated) and falls back to the measured dominant tensor dtype rather than guessing from the filename. NOT DONE: resident_bytes is always null and says so — checkpoints are mmap-resident, so the true figure is a page-cache property this process cannot read and the file size would be a lie in both directions. NOT DONE: per-model capability flags; the reason-carrying capability list is on /health, which is where the UI already reads it"
    status: completed
  - id: ui-model-swap
    content: "Model load/unload/swap. frink-server's model.rs::load() is single-shot from env today; this is the real backend work behind the model manager. LANDED 1a59c15: AppState.active is an RwLock<Option<Arc<ActiveModel>>> and the rule is in its doc comment — a reader clones the Arc under the read lock and then runs, so the lock guards the POINTER and is never held across a decode. An in-flight request finishes against the weights it started on and the old model is freed when the last holder lets go; no attempt is made to migrate a running request, because half a completion from each of two checkpoints is worse than either. Tested: the old handle still decodes after a swap, and strong_count shows the swapped-out weights outliving the registry reference. ActiveModel bundles the continuous batcher with the model because the batcher thread holds one specific Arc<Decoder>. model.rs::load_from_path() is the env-free entry point; activate_loaded_model() is shared by startup and the load task so an engine variant cannot be wired into one and forgotten in the other. Load is by discovered id only — there is deliberately NO load-by-path endpoint. Unload makes /health reach the `unavailable` state Phase 1 defined but could not exercise. NOT DONE: the KV pool is still sized from the startup model, and no warmup prefill runs after a swap (ui-warmup is unclaimed)"
    status: completed
  - id: ui-task-contract
    content: "ONE long-running-task contract reused by download/convert/bench: POST start -> {task_id}, GET tasks, POST cancel/{id}, fields {status, progress, bytes_done, bytes_total, error, timestamps, retry_count}, plus a per-task event stream. LANDED 1a59c15: one TaskView shape for both job kinds (download, load). Terminal is terminal — a late update from a worker that has not noticed its cancellation is dropped, so a UI that stopped polling is never wrong. Progress comes from frink_api::progress::RateEstimator and nowhere else, so a warming window reports null rate and null ETA by construction (ui-transfer-stats finally has its consumer). Cancellation is cooperative and honest: a download stops within a chunk and keeps its .part file, a model load cannot be interrupted mid-mmap and so discards its finished result rather than pretending it stopped early, and a task only reaches `cancelled` when a worker acknowledges it. A TaskGuard rides with every worker so a panic cannot leave a task at `running` forever — nothing awaits a spawn_blocking handle. NOT DONE: no per-task event stream (polling only) and no retry_count — nothing retries yet, and a field that is always 0 is a promise, not a contract. convert/bench are not task kinds because neither job exists"
    status: completed
  - id: ui-transfer-stats
    content: "Rolling-window rate/ETA smoothing: >=3 samples over >=3s before reporting `stable`, reset on counter regression, ETA clamped. Prevents the '123 GB/s' flash on the first tick. Reimplement from the description — the reference is AGPL. LANDED 2e8c214: `frink_api::progress::RateEstimator`, written from this description only. A warming report carries NO number at all rather than an untrusted one, which makes the flash structurally impossible; window trims by age and drops from the middle on overflow so a fast-ticking job still reaches `stable`; ETA needs a known total and a positive rate. CONSUMER LANDED 1a59c15: frink_api::TaskProgress can only be built from a RateReport, so every /admin/tasks progress block inherits the refusal by construction — verified end to end on a real 396 MB Hub download, which reported no rate for the first four seconds and then a rate and an ETA"
    status: completed
  - id: ui-frontend-chat
    content: "Chat: streaming, markdown+code, reasoning blocks, sampling controls, model switch mid-conversation, conversation tree persisted SERVER-side (SQLite), not localStorage. REBUILT on a real component stack (React 19 + Vite + Tailwind v4 + Radix, with `@assistant-ui/react` owning the transcript, composer, autoscroll and branching) and VERIFIED in a browser against a real checkpoint: tokens stream, markdown lists and fenced code render with a copy button, and the line under each answer is the server's `usage` -- TTFT, prefill tok/s and decode tok/s as separate numbers. There is still NO client stopwatch: assistant-ui offers a `useMessageTiming()` that measures the stream in the browser and it is deliberately unused, because it cannot separate prefill from decode. Untrusted text still never becomes markup, and this was verified rather than assumed -- a seeded answer containing `<script>`, `<img onerror>` and a `javascript:` link produced zero script and img elements, no globals, and an anchor whose href react-markdown had already emptied; the literal characters showed as text. NOW DONE that was not before: a model switcher inside Chat, honest that a swap is a server-side load affecting every client rather than a per-request parameter; edit/regenerate branching, which comes with assistant-ui's message repository; prompt starters on the empty state (they fill the composer rather than auto-sending, because one click should not spend a minute of CPU decode). STILL NOT DONE and not stubbed: reasoning blocks, math, and the server-side SQLite conversation tree -- this server has no conversation API, so the transcript is assistant-ui's exported repository (messages plus parent ids, so branches survive) in localStorage, and the screen says so on its face"
    status: pending
  - id: ui-api-monitor
    content: "API Monitor screen: ring-buffered live request log. Keep duration_ms and decode_ms SEPARATE (duration carries queue wait + prefill; conflating them reads a 50 tok/s model as 5) and flag external-vs-UI traffic. BACKEND LANDED 1a59c15 and SCREEN REBUILT on TanStack Table: 200-entry ring keyed by request_id, duration_ms and decode_ms separate all the way to the wire, decode_ms/ttft_ms null rather than a copy of the total, tok/s computed from `completion_tokens / decode_ms` and nothing else, three null-dropping sparklines, and a 404 from /admin/stats rendering as `not available in this build` on this screen alone. THE OPEN HALF IS NOW CLOSED. `via_api_key`: every row names the key that served it as a per-process-SALTED fingerprint — never the key, never anything the key can be recovered from, so a stats payload pasted into a bug report leaks nothing and cannot be used to test key guesses offline (the plan's do-not-inherit list names a reference product returning its plaintext key in a stats payload). Two keys are two callers; no key is null, not a fingerprint of nothing. External-vs-UI is answered HONESTLY rather than inferred: beside the fingerprint sits `client`, the caller's self-declared `X-Frink-Client` header, which Studio now sends and which the column, its tooltip and the card footer all label as self-declared, because nothing authenticates it and a monitor that overstates what it knows is worse than one that shows less. Queue gauge: `queue_depth` / `queue_rejected_total` from the continuous-batching scheduler's own queue, null — not 0 — when batching is off, because then nothing queues in front of anything and a 0 would claim an empty queue was measured; `generating_now` stays beside them, still named for work in progress. /v1/embeddings, /v1/tokenize and /v1/detokenize are recorded now, with their status on failure as well as success — and with honest token accounting: embeddings run a forward pass so their prompt tokens count toward tokens_prompt_total and decode_ms stays null (there is no decode loop to time), while tokenize/detokenize run the tokenizer only and contribute no tokens to a counter that means 'tokens put through a forward pass'. The `model` column landed too, and deliberately records the model that SERVED the request rather than the `model` field the request carried: this server decodes against whatever is loaded and ignores that string, so echoing it back would make the log agree with the caller's belief instead of with what happened; it is read off the handle the request decoded against, so a swap mid-flight cannot rewrite history, and it is null when nothing was loaded. NOT DONE: `context_usage` and `stop_reason`, which the plan's field list names and the ring still does not carry; no browser verification of the new columns this cycle (they are covered by server tests and by types, not by a screenshot)."
    status: completed
  - id: ui-embed-server
    content: "Embed the built frontend into the frink-server binary (rust-embed) and serve at `/` with an SPA fallback. LANDED 4fbfe62, then DELIBERATELY REVERSED: the studio is a standalone app again and frink-server serves the API only. `src/ui.rs`, the `static/` folder, the rust-embed dependency, `--ui-server`, `FRINK_UI` and `routes::{ROOT, UI}` are all gone, and `/` is a 404 like any other unknown path -- which is what it already was without the flag, so no reachable behaviour was removed. What the embedding bought was one binary with no JS toolchain in the release path; what it cost was a committed build output inside a published crate and a SPA fallback that had to be prevented, by rule and by test, from ever answering an API path with HTML. Neither cost is worth paying for a frontend that is a static bundle any file server can host. THE CONSEQUENCE, and it is real: two origins where there was one, so CORS now applies. `ui/`'s dev server proxies /v1, /admin, /health, /metrics and /cache to 127.0.0.1:8383, which keeps the browser on one origin in development and needs no server-side configuration at all; a deployment that serves the built bundle from elsewhere sets FRINK_CORS_ORIGINS to that exact origin, which security.rs already refuses to widen to `*`. The UI carries a base-URL setting and a bearer-key field on the Connect screen for exactly that case, because it can no longer assume the page's own origin is the API's"
    status: completed
  - id: ui-tauri-shell
    content: "Tauri 2 shell: bundles the same frontend, spawns frink-server as a child, targets app/dmg/deb/nsis. Backend state machine incl. `unresponsive` as distinct from `failed` — a hung server must not trigger a restart"
    status: pending
  - id: ui-settings-snippets
    content: "Settings > Connect: copy-pasteable curl / Python-SDK / IDE snippets prefilled with the live base URL, key and loaded model. LANDED 5b784fa, REBUILT and re-VERIFIED in a browser: four snippets (curl streaming, the official Python OpenAI SDK, an OPENAI_BASE_URL/OPENAI_API_KEY env block, and a probe trio), each with a copy button, each filled from the model id /v1/models reports right now. The API key is entered here, stored in localStorage and sent as an Authorization header, never in a URL. CHANGED by the standalone split, and it was a real bug caught in the browser rather than in review: the snippets used to read `window.location.origin`, which is now the Vite dev server or a static host and NOT the API. A curl line saying `http://localhost:5173/v1/...` works only in the tab it was copied from, which is the same failure as shipping YOUR_MODEL_HERE. Snippets now use the configured base URL, falling back to frink's default bind address, and the screen states which one it used. The screen also carries the base-URL setting itself and spells out that another origin means FRINK_CORS_ORIGINS on the server. NOT DONE: per-IDE config files (docs/AGENTS_COOKBOOK.md is linked instead of emitted), and the `frink launch <tool>` writer the plan calls better still"
    status: completed
  - id: ui-cross-platform-truth
    content: "BLOCKING HONESTY: Windows/Linux GPU means CUDA, which the parity plan keeps only at 'must compile' — no receipts, no host pin. A desktop app shipping to Windows today is CPU-only in practice. State it, do not imply otherwise. LANDED: stated in docs/FEATURES.md beside the backend table, and /health now says the same per capability with a reason string instead of silently greying a control. Still to state on a download page if one ever ships (ui-tauri-shell)"
    status: completed
isProject: false
---

# frink UI — web, desktop, one frontend

> One frontend. Served by `frink-server` at `/` (that is the web UI) and
> bundled by a Tauri 2 shell into a native app for macOS, Windows and
> Linux. Written **2026-08-13** from a read-only study of two shipped
> products under `.scratch/`: **oMLX** (Python/MLX, Swift menubar app)
> and **Unsloth Studio** (React 19 + FastAPI + Tauri 2).
>
> **Licence rule, non-negotiable.** Unsloth's `studio/` is **AGPL-3.0**;
> frink is Apache-2.0. Everything below is a description of a *contract*
> or a *behaviour*, arrived at by reading. No code is copied, and none may
> be. oMLX is Apache-2.0 but is MLX-bound, so nothing of it is liftable
> either — same rule applies by accident rather than by licence.

## Why this shape

Both reference products converged on it from opposite directions:

- **oMLX's native Swift app has no chat screen at all.** 37k lines of
  Swift across 122 files, and chat + dashboard are ordinary HTML pages
  served by the backend and opened in the user's *default browser* —
  there is not even a WKWebView. The app is a supervisor and a settings
  panel. Its own portable UI turned out to be the browser layer.
- **Unsloth Studio built exactly the target architecture**: one React SPA
  served by FastAPI for web, and a Tauri 2 shell that bundles the same
  SPA and spawns the backend as a child process. 215k LOC of frontend
  with five TODO markers in the whole tree — a shipped product, not a
  demo.

frink starts from a better position than either: **one static Rust
binary**. oMLX's packaging chapter — venvstacks, `PYTHONHOME` relocation,
`install_name_tool` surgery on broken wheels — and Unsloth's 285 KB
`install.sh` that provisions a Python interpreter at install time both
exist to solve a problem frink does not have. Nothing in this plan may
reintroduce it.

## Where this stands

The frontmatter above is the checklist and the only status worth
reading. Three dated snapshots used to live here (2026-08-14,
2026-08-21, and a strikethrough list of what was missing), which meant
three answers to one question and two of them wrong.

Shipped: the `frink-api` contract crate, the three-state `/health`
handshake, `--port 0` with the `frink.server.ready` line, `request_id`
and per-phase `usage` timings, the `/admin` control surface with runtime
model swap, and all four screens (Chat, Models, Activity, Connect)
verified in a browser against a real checkpoint.

The UI is a standalone React app under `ui/`. `frink-server` serves the
API and nothing else, and `GET /` is a 404. `npm run dev` proxies the
API so development is one origin; a hosted build needs
`FRINK_CORS_ORIGINS` set to its exact origin.

Still open, and each says why in its own todo: server-side conversation
storage (the transcript is in `localStorage` today and the screen says
so), and the Tauri shell.


## Architecture

```
frink-server (one Rust binary)          ui/  (React + Vite bundle)
  ├─ /v1/*        OpenAI + Anthropic API    Chat · Models · Activity · Connect
  ├─ /admin/*     control API          ←──  every screen, over the public API
  ├─ /metrics     Prometheus
  └─ /            404 — the UI is not served from here

Web     = `npm run dev` (proxies the API, one origin) or any static host
          serving `ui/dist` with FRINK_CORS_ORIGINS set to its origin
Desktop = Tauri 2 shell → bundles the same `ui/dist` → spawns the server
Both    = one frontend codebase, one streaming path
```

### The rule that makes it work

**The UI is just another API client.** It calls `/v1/chat/completions`
with the same token an external client would use — never a private
`/api/chat`. Unsloth mounts one router at two prefixes for exactly this
reason, and it is the single highest-value decision in their codebase:
every UI feature is automatically an API feature, the public contract
cannot rot without the UI breaking first, and there is one streaming code
path to debug instead of two.

## Phase 1 — backend contracts (before any frontend work)

### 1a. Health as a capability handshake, with three states

Not two. `ready` / `unavailable + reason` / **`detecting`**. The third is
the one both products learned the hard way: while backends are being
probed, rendering the *guess* blacks out GPU-dependent controls on first
paint in a way that is visually identical to a measured "unsupported".
The UI must be able to *hold*.

- Liveness (port bound) and readiness (model loaded) are separate answers.
  oMLX returns 503 `"loading"` while preloading, because the port binds
  before the engine is ready.
- Hard **1s budget** on detection, then answer provisionally — the
  desktop shell probes with a short timeout and does not retry.
- Capability bits carry a machine reason *and* a human sentence:
  `metal_unavailable`, `cuda_not_built`, `cpu_only`. The UI greys the
  control **with the reason in a tooltip**; it never silently hides it and
  never re-derives the explanation.
- **Auxiliary health.** A saturated GPU can starve `/health`. Let the
  cheap status endpoint the UI already polls *vouch* for liveness within a
  freshness window, so the app does not declare a busy backend dead. This
  is a real bug class in oMLX's own history, not polish.

### 1b. Process lifecycle that needs no OS-specific code

- **`--port 0`** plus a structured line on stdout carrying the actually
  bound address and pid. This deletes oMLX's entire port-conflict feature:
  the `lsof` shell-out, the owner identification, the alert dialog, and a
  `killExternal()` helper that was written and never wired to anything.
- **Stdin close ⇒ exit.** The one orphan-prevention mechanism that behaves
  identically on macOS, Windows and Linux, and survives a parent crash.
  oMLX needs POSIX signal handlers plus an `atexit` trampoline plus a
  synchronous reaper; Windows has no SIGTERM at all.
- Preflight disposition when a server is already on the port:
  `managed_ready | owned_ready | attached_ready | external_conflict`,
  distinguished by an opaque per-install id so "mine" and "a stranger's"
  are different answers.

### 1c. Streaming contract

- **`request_id` in the first chunk**, and metrics keyed by the same id.
  oMLX's chat page has to claim request ids out of a stats snapshot with a
  heuristic so concurrent chats do not steal each other's numbers. frink
  can simply state it.
- **Two-tier cancellation**: `AbortSignal`/socket close **plus** an
  explicit `POST /cancel` with `keepalive`. A dropped connection does not
  reliably stop a decode loop, and a proxy that swallows the abort leaves
  the backend generating into nothing.
- **Truncation is an error, not a completion.** EOF without `[DONE]` and
  without a `finish_reason` must surface as a retryable failure, never as
  a finished message.
- **SSE hardening**: `id:` + `retry:` + `Last-Event-ID` replay so a
  reconnect resumes rather than restarts; a stall timeout; and a polling
  fallback beside every stream, because reverse proxies buffer
  `text/event-stream`. Both reference projects shipped the fallback after
  hitting this in production.
- **`usage` timings**, both non-streaming and in the `include_usage` final
  chunk: `time_to_first_token`, `prompt_eval_duration`,
  `generation_duration`, prompt and generation tok/s, `cached_tokens`.
  frink's kernels already know these; emitting them means the UI never
  wall-clock-guesses. Keep **queue wait + prefill (`duration_ms`) separate
  from decode (`decode_ms`)** — conflating them reads a 50 tok/s model as
  5, and every tok/s number downstream becomes a lie.

### 1d. Admin surface

- `GET /admin/models` — per model: id, path, loaded/loading, on-disk size,
  resident size, backend, context length, quant, plus capability flags
  **each paired with a reason string**.
- `POST /admin/models/{id}/load|unload`, `PUT /admin/models/{id}/settings`.
  This requires making `model.rs::load()` swappable, which is the real
  work of the first cut.
- **One task contract** for every long job (download, convert, bench):
  `POST …/start → {task_id}`, `GET …/tasks`, `POST …/cancel/{id}`, with
  `{status, progress, bytes_done, bytes_total, error, timestamps,
  retry_count}` and a per-task event stream. Uniformity is what makes the
  UI cheap; both products reuse one shape across four job types.
- A **non-terminal `finalizing` phase** in every job state machine. For
  model load that is `mmap → dequant → warmup → ready`. A bar pinned at
  100% while work continues is indistinguishable from a hang.
- Rate/ETA smoothing over a rolling window: at least 3 samples spanning at
  least 3 s before a rate is reported `stable`, buffer reset when the byte
  counter goes backwards, ETA clamped at zero. Reimplemented from this
  description — the reference implementation is AGPL.

### 1e. Warmup

Neither product warms up after load, and both pay Metal shader/JIT
compilation on the first real request; both of their *benchmark* paths
warm up explicitly because of it. frink has the same exposure and should
do a warmup prefill as part of `finalizing`, not as a user's first token.

## Phase 2 — the frontend

**Stack, decided and shipped**: React 19 + Vite, Tailwind v4, Radix UI
primitives with shadcn-style wrappers, `@assistant-ui/react` +
`@assistant-ui/react-markdown` for the chat, TanStack Table for the
request log, lucide-react icons. Every dependency is MIT / Apache-2.0 /
ISC / BSD, and `npm run licenses` fails the build on anything else —
the bundle is distributed, so a copyleft dependency is not a lockfile
detail. It builds to static assets any file server can host.

Screens, first cut:

1. **Chat** — streaming, markdown + code + math, collapsible reasoning
   blocks, sampling controls, model switch mid-conversation, per-model
   settings that survive the switch. **Conversation state server-side in
   SQLite**, with a `parent_id` per message so edit/regenerate branches
   work. Not `localStorage`: oMLX's chat stores the entire history as one
   JSON blob and *pops conversations off the end* when the quota is
   exceeded — silent data loss by design.
2. **Models** — list with size/quant/backend/context, load/unload,
   download with progress, "will it fit?" **computed exactly from the
   GGUF header** rather than heuristically (frink knows `n_layers`,
   `n_kv_heads`, `head_dim`, KV dtype; both reference products guess).
3. **Metrics / API Monitor** — live request log, ring-buffered:
   `{id, endpoint, model, status, started_at, duration_ms, decode_ms,
   ttft_ms, tok_per_sec, prompt_tokens, completion_tokens, context_usage,
   stop_reason, via_api_key}` plus a queue gauge and a server timestamp so
   the browser clock is never trusted. `via_api_key` distinguishes an
   external caller (someone's editor) from the UI's own traffic — the most
   directly useful screen for a server that is already pointed at IDEs.
4. **Settings > Connect** — copy-pasteable curl / Python-SDK / IDE config
   snippets prefilled with the live base URL, key and loaded model.
   `docs/AGENTS_COOKBOOK.md` describes this today; a screen that *emits*
   it is strictly better, and a `frink launch <tool>` that writes the
   config file is better still.

## Phase 3 — the desktop shell

Tauri 2, bundling the same frontend, spawning `frink-server` as a child.
Targets `app` / `dmg` / `deb` / `nsis`. Single-instance, window-state,
CSP `connect-src` allowing `http://127.0.0.1:*`.

Backend state machine — **`unresponsive` must be a state distinct from
`failed`**, because a hung backend must not trigger a restart loop:

```
checking → starting → running(pid) → unresponsive(pid)
                   ↘ failed(msg)   ↘ stopping → stopped
```

Restart backoff 5/10/20 s, cap 3, counter reset after 60 s stable, and an
`expectingExit` flag so a user-initiated stop does not read as a crash.
A startup screen driven off the backend's own stderr lines
("Loading model… / Building Metal pipelines… / Warming up…").

**Do not inherit:** unix-domain control sockets (no clean Windows
equivalent), `lsof` for port ownership, `hdiutil`-based DMG self-update
with an unsigned swap, or an API key passed in a URL query string —
oMLX does the last one for its browser handoff, and also returns the
server's plaintext key in a stats payload.

## The honesty clause

**Windows and Linux GPU means CUDA.** The parity plan keeps CUDA at
"must compile" — no receipts, no host pin, and `frink-cli --features
cuda` did not compile at all at one point this cycle. A desktop app
shipped to Windows today is **CPU-only in practice**. That is a scope
decision to state on the download page, not something to imply otherwise
by shipping a Windows installer next to a Metal benchmark table.

Unsloth handles the mirror image of this honestly and it is worth copying:
their macOS build defaults to chat-only, and the UI greys the unavailable
features **with a reason string** rather than hiding them — while their
README says only "works on Windows, Linux, WSL and macOS".

## Sequencing

Phase 1 is entirely backend and is independently useful — `request_id`,
`usage` timings, two-tier cancel, task contract and `--port 0` improve
`frink-server` for IDE users whether or not a frontend ever ships.
Phase 2 depends on 1d (model swap) only — **1d has landed**, so Phase 2
is unblocked. Phase 3 depends on 1b.

Do Phase 1 first. It is the part that cannot be replaced later without
breaking clients.
