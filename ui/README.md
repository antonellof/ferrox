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
nothing for the licence check to weigh. It covers three things a browser
cannot show you:

- the stream recovery paths in `lib/api.ts`, the half of SSE hardening
  that cannot be proven from the server: that a reconnect resumes from
  the last `id:` without repeating a token, that a lost replay window
  surfaces as a truncation error instead of a partial answer shown as a
  whole one, and that a non-resumable stream never tries to reconnect
  into a buffer that does not exist;
- the transcript sync's pure half in `lib/conversations.ts`, where
  getting "which nodes are new" wrong duplicates or drops a message;
- the entry rule in `lib/entry-state.ts` (below), where being too eager
  throws away the conversation you were in and being too lazy resurrects
  one for ever. Both failures are silent.

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
                        one prefers-color-scheme block, nothing else)
  lib/api.ts            the ONLY place that talks HTTP
  lib/api.test.ts       stream recovery, against a stubbed fetch
  lib/entry-state.ts    what Chat opens with, and why
  lib/use-tab-activity  the heartbeat that measures "away"
  lib/format.ts         "unknown" is an em dash, never a zero
  components/           app shell, server status, shadcn-style primitives
  screens/chat/         assistant-ui runtime, markdown, thread
  screens/{models,activity,connect}.tsx
```

## The look, in four rules

They are written at the top of `src/index.css`, because a rule nobody can
quote is a rule that drifts. Three of them are asserted by
`lib/theme.test.ts` rather than promised.

- **Colour.** There is **no brand hue**. The neutral ramp is zero-chroma
  and the only chromatic tokens in the app are `ok`, `warn` and `err`, so
  chroma means exactly one thing: a state. The test fails if any other
  token grows chroma, or if a semantic one loses it.
- **Emphasis**, since no hue carries it. Four levers, loudest first: a
  solid **ink** fill (`--ink`, near-black in light and near-white in
  dark) for the one primary action in a view; **weight** for a selected
  row or a label; **surface** behind a hairline for grouping; and the
  **contrast** ramp `fg` / `muted` / `faint` for rank inside a block.
- **Type.** One six-step scale — `2xs` 11, `xs` 12, `sm` 14, `base` 15,
  `lg` 17, `xl` 20, plus `code` 13 — and no arbitrary sizes in
  components.
- **Elevation.** Borders separate; shadows only float. One shadow token,
  `--shadow-pop`, for popovers and the mobile drawer.

The mark is a body-centred cubic cell, which is the structure of α-iron.
Its geometry lives in `lib/logo-geometry.ts` and is drawn twice — by
`components/logo.tsx` in `currentColor`, and by `public/favicon.svg`,
which needs a colour of its own because a favicon cannot inherit one.
`lib/logo-geometry.test.ts` reads the favicon off disk and holds the two
to the same path strings.

## Where you land, and what it opens

**The URL says which conversation you are in.** `/ui/chat` is a new
chat, `/ui/chat/<id>` is that conversation, and `/` and `/ui` redirect to
the first of those. That is what every chat UI of this shape does, and it
is what makes the base URL a predictable entry point instead of "whatever
was open last" — which is what this app used to do, from any path, after
any amount of time, by opening the newest conversation it could find.

The id is stamped into the URL by the first message, with `replace`, so
Back leaves Chat rather than walking into the empty version of the
conversation you are looking at. Opening one from the picker and starting
a new chat are pushes, so Back undoes them.

**A tab that has been away comes back to a new chat.** The window is
`RESUME_WINDOW_MS` in `lib/entry-state.ts`, thirty minutes, and it is
measured rather than assumed: a heartbeat stamps `sessionStorage` only
while the document is visible, so a hidden tab, a closed lid and a
sleeping machine accumulate away-time and a tab you are sitting in front
of never does. `sessionStorage`, not `localStorage`, is the whole trick —
it is per tab and dies with the tab, so a reload of a tab that was away
all night still reads as away, while a fresh tab opened on a deep link
has no record at all and the link is simply honoured.

Two things keep this from being a rule that loses work. It never fires
over a running generation or a half-typed message. And when it does fire
it says so, in the number, and offers the conversation back in one click.
No other chat UI does this at all — the research behind it found not one
product with a staleness rule — so it is deliberately narrow, announced
and undoable rather than silent.

Local mode (a server with no `/v1/conversations`) has no ids to put in a
URL, so it asks the same function with the one slot it has and its own
saved-at stamp. One rule, two callers.

## One model selector

Choosing which model answers happens in **one** place: the menu in the
Chat header. **Models** is the library — what is on disk, downloads, and
`Unload`, which is the one verb the menu cannot express. It used to carry
a second selector (a `Load` button per row, the same `POST
/admin/models/load` for the same server-wide effect), a second `Unload`,
and an `active:` badge restating the row's own `state` column.

The sidebar's bottom control is **server status**, not a model picker. It
used to render the loaded model id under a chevron, in the slot an
account or settings control normally occupies, which made it read as a
third selector; the model id belongs where the model is chosen. It never
names a backend, either: `/health` says which backends are *available*,
never which one is running, so "on Metal" there would be invented.

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
