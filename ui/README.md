# Ferrox Studio

The web frontend for `ferrox-server`: Chat, Models, Activity, Connect.

It is a **standalone app**. `ferrox-server` serves the HTTP API and
nothing else, `GET /` on it is a 404, and this app reaches that API the
same way an editor would. That rule earns its keep. Every screen here
goes through the public API, so the API cannot rot without a screen
breaking first and showing you.

## Working on it

```bash
npm install

# Terminal 1, the real backend, no UI flag, nothing special
cargo run -p ferrox-server -- -m models/some-model.gguf

# Terminal 2, Vite on :5173
npm run dev
```

`npm run dev` talks to a real `ferrox-server`, not a mock. Point it
somewhere other than `127.0.0.1:8383` with `FERROX_BACKEND`:

```bash
FERROX_BACKEND=http://127.0.0.1:9001 npm run dev
```

```bash
npm run typecheck   # tsc, no emit
npm run lint        # eslint
npm run licenses    # fails the build on any non-permissive dependency
npm test            # node's own test runner, no framework installed
npm run build       # -> dist/, gitignored
npm run check       # typecheck + lint + test
```

`npm test` runs `node --test` over `src/lib/*.test.ts`, node strips the
types itself, so there is no test framework in the dependency tree and
nothing for the licence check to weigh. It covers the stream recovery
paths in `lib/api.ts`, which are the half of SSE hardening that cannot
be proven from the server: that a reconnect resumes from the last `id:`
without repeating a token, that a lost replay window surfaces as a
truncation error instead of a partial answer shown as a whole one, and
that a non-resumable stream never tries to reconnect into a buffer that
does not exist.

CI runs `npm run licenses`, `npm run typecheck`, `npm run lint` and
`npm run build`. It does not run `npm test`, so run `npm run check`
yourself before opening a PR that touches `lib/`.

`dist/` is **not committed**. Nothing built here ships inside a Rust
crate, so there is no artefact to keep in sync. Serve `dist/` with any
static file server, or bundle it into a desktop shell.

## CORS, and how to not need it

The app and the server are two origins, so the browser's cross-origin
rules apply. Two answers are supported:

- **Development, and the default.** The dev server proxies `/v1`,
  `/admin`, `/health`, `/metrics` and `/cache` to the backend. The
  browser sees one origin, no preflight happens, and no server
  configuration is needed. The app's API base URL stays empty and every
  request goes out as a same-origin path.
- **A different origin.** Set the API base URL on the **Connect**
  screen (or `VITE_FERROX_BASE_URL` at build time), and start the server
  with `FERROX_CORS_ORIGINS` set to **this app's exact origin**. A `*`
  wildcard is a startup error, enforced in
  `crates/ferrox-server/src/security.rs`, because a wildcard beside a
  bearer token is a credential-leak shape.

The Connect screen also holds the `FERROX_API_KEY` bearer token, stored
in `localStorage` and sent as an `Authorization` header, never in a
URL.

## Stack

React 19 · Vite · Tailwind v4 · Radix UI primitives (the shadcn/ui
foundation) · `@assistant-ui/react` for the chat transcript ·
TanStack Table for the Activity log · lucide-react icons. Every runtime
dependency is MIT / Apache-2.0 / ISC / BSD, see
[`docs/THIRD_PARTY_NOTICES.md`](../docs/THIRD_PARTY_NOTICES.md), and
`npm run licenses` enforces it in CI. The bundle is distributed, so a
copyleft dependency is not a lockfile detail.

## Layout

```
src/
  main.tsx              router: /, /ui and /ui/<screen> all resolve here
  index.css             design tokens + Tailwind theme (one light block,
                        one prefers-color-scheme block, nothing else),
                        and the three rules — type scale, spacing
                        rhythm, elevation — that every component obeys.
                        Read its header before changing anything visual.
  lib/api.ts            the ONLY place that talks HTTP
  lib/api.test.ts       stream recovery, against a stubbed fetch
  lib/format.ts         "unknown" is an em dash, never a zero
  components/           app shell, health pill, shadcn-style primitives
  screens/chat/         assistant-ui runtime, markdown, thread
  screens/{models,activity,connect}.tsx
```

## Streams that survive a proxy

Chat asks for a resumable stream (`stream_resumable: true`), so every
event carries an `id:` and the server keeps a replay buffer. Three
consequences, all deliberate:

- A connection that dies mid-answer is **reconnected** with
  `Last-Event-ID`, continuing rather than restarting. The banner says so;
  a reconnect that silently replaced the original connection would leave
  the user watching an indicator that stopped meaning anything.
- A stream that goes quiet for 45 s while the socket stays open is the
  signature of a proxy buffering `text/event-stream`. A second SSE
  connection would go through the same proxy, so that case **skips
  straight to polling** `GET /v1/stream/{id}/poll`, a short JSON
  response nothing can hold back.
- A resumable request is **not cancelled by its socket closing**. That
  is the point of asking for one, and it means `POST /v1/cancel` is the
  only stop path. This app already sends it on Stop, on New chat, on
  leaving the screen, and on `pagehide` with `keepalive`, which is
  exactly the set of cases the socket close was standing in for. If you
  ever remove one of those, remove `stream_resumable` in the same
  change.

Nothing here ever presents a partial answer as a finished one: if the
reconnect and the poll both fail, or the replay window has moved past
where the client stopped, it surfaces as a truncation error.

## Three things not to undo

- **No client stopwatch.** Every number under an answer comes from the
  server's `usage` block. assistant-ui offers a `useMessageTiming()`
  that measures the stream in the browser. It is deliberately unused. It
  has no way to separate prefill from decode, so it reads a 50 tok/s
  model as 5 on a long prompt.
- **Model output never becomes markup.** `react-markdown` builds a React
  element tree and has no raw-HTML path unless `rehype-raw` is added to
  the pipeline. Do not add it. A lint rule fails the build on
  `innerHTML` and `dangerouslySetInnerHTML`, so nobody has to remember
  the rule.
- **`duration_ms` and `decode_ms` are never combined.** Duration carries
  queue wait and prefill, so the `tok/s` column divides by `decode_ms`
  alone. Nothing here computes a download rate either. The server's
  estimator reports `null` until it is confident, and that null renders
  as words rather than a number.
