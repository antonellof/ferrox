// The only place in the frontend that talks HTTP.
//
// The UI is just another API client: it calls the same public
// `/v1/chat/completions` an editor would, with the same key, over the
// same streaming path. There is no private UI endpoint anywhere in this
// app, which is what stops the public contract from rotting silently.
//
// Studio is a STANDALONE app — `ferrox-server` does not serve it — so it
// cannot assume the page's own origin is the API's. Two supported ways
// to point it at one:
//
//  - **Dev, and the default**: `npm run dev` proxies `/v1`, `/admin`,
//    `/health`, `/metrics` and `/cache` to the backend, so requests go
//    out same-origin and CORS never applies. The base URL stays empty
//    and every path below is used as-is.
//  - **A different origin**: set the base URL on the Connect screen (or
//    `VITE_FERROX_BASE_URL` at build time). The operator must then set
//    `FERROX_CORS_ORIGINS` on the server to this app's exact origin —
//    the wildcard is rejected by design, because `*` alongside a bearer
//    token is a credential-leak shape.
//
// Route strings mirror `ferrox_api::routes` one-for-one. Keeping them
// in a single object means a rename shows up here as one diff rather
// than as a scattering of string literals.

export const routes = {
  health: "/health",
  metrics: "/metrics",
  cacheStats: "/cache/stats",
  models: "/v1/models",
  chatCompletions: "/v1/chat/completions",
  cancel: "/v1/cancel",
  stream: (requestId: string) => `/v1/stream/${encodeURIComponent(requestId)}`,
  streamPoll: (requestId: string) =>
    `/v1/stream/${encodeURIComponent(requestId)}/poll`,
  conversations: "/v1/conversations",
  conversation: (id: string) => `/v1/conversations/${encodeURIComponent(id)}`,
  // A POST, not a DELETE: the server's CORS allow-list is GET + POST, so
  // a DELETE would pass from curl and fail the preflight from a browser
  // on any other origin. See `conversations.rs`.
  conversationDelete: (id: string) =>
    `/v1/conversations/${encodeURIComponent(id)}/delete`,
  adminModels: "/admin/models",
  adminModelsLoad: "/admin/models/load",
  adminModelsUnload: "/admin/models/unload",
  adminDownload: "/admin/download",
  adminTasks: "/admin/tasks",
  adminStats: "/admin/stats",
  adminTaskCancel: (taskId: string) =>
    `/admin/tasks/${encodeURIComponent(taskId)}/cancel`,
} as const;

const KEY_STORAGE = "ferrox.studio.apiKey";
const BASE_STORAGE = "ferrox.studio.baseUrl";

/**
 * Build-time default, for a deployment that ships pre-pointed.
 *
 * Read defensively: `import.meta.env` is Vite's, and this module is
 * also imported outside a bundle by the test runner, where it does not
 * exist at all.
 */
const BUILT_IN_BASE = (
  import.meta.env?.VITE_FERROX_BASE_URL ?? ""
).replace(/\/+$/, "");

/**
 * Where the API lives.
 *
 * Empty means "this page's own origin", which under `npm run dev` is the
 * Vite proxy and therefore the backend. Anything else is used verbatim
 * as a prefix, with the trailing slash normalised away so
 * `http://host:8383` and `http://host:8383/` behave identically.
 */
export function apiBase(): string {
  try {
    const stored = localStorage.getItem(BASE_STORAGE);
    if (stored !== null) return stored.replace(/\/+$/, "");
  } catch {
    /* private browsing: fall through to the build-time default */
  }
  return BUILT_IN_BASE;
}

export function setApiBase(value: string): void {
  const normalised = value.trim().replace(/\/+$/, "");
  try {
    if (normalised) localStorage.setItem(BASE_STORAGE, normalised);
    else localStorage.removeItem(BASE_STORAGE);
  } catch {
    /* private browsing: the setting simply does not persist */
  }
}

/** Absolute or same-origin URL for one of the routes above. */
export function url(path: string): string {
  const base = apiBase();
  return base ? `${base}${path}` : path;
}

/** The API key, when the operator has set FERROX_API_KEY and told us. */
export function apiKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

export function setApiKey(value: string): void {
  try {
    if (value) localStorage.setItem(KEY_STORAGE, value);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {
    /* private browsing: the key simply does not persist */
  }
}

/** Where `ferrox-server` binds unless told otherwise. */
export const DEFAULT_SERVER_ORIGIN = "http://127.0.0.1:8383";

/**
 * The origin a snippet should tell some *other* tool to call.
 *
 * Deliberately NOT this page's origin. With the base URL unset, this
 * app's own requests go same-origin and the dev server proxies them --
 * which is right for the app and wrong for a snippet, because the thing
 * pasting it is an editor on the other side of the machine, and
 * `http://localhost:5173/v1/...` is a path only this browser can reach.
 * A snippet that works where it was copied and nowhere else is the same
 * failure as one that still says YOUR_MODEL_HERE.
 *
 * So: the configured base when there is one, and ferrox's default bind
 * address when there is not. The Connect screen says which it used.
 */
export const snippetBase = () => apiBase() || DEFAULT_SERVER_ORIGIN;

/** The origin THIS app's own requests reach, for error messages. */
export const baseUrl = () => apiBase() || window.location.origin;

/**
 * What this app calls itself on every request it makes.
 *
 * The server records it on the Activity row (`client`) so a request the
 * UI made is distinguishable from one an editor made. It is a CLAIM and
 * nothing else — any client can send this header, nothing authenticates
 * it, and the Activity screen says so on its face. It is still better
 * than the alternative, which is inferring "that must have been the UI"
 * from timing.
 */
export const CLIENT_LABEL = "ferrox-studio";

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const h: Record<string, string> = {
    ...extra,
    "X-Ferrox-Client": CLIENT_LABEL,
  };
  const key = apiKey();
  if (key) h.Authorization = `Bearer ${key}`;
  return h;
}

/**
 * An HTTP failure with the status kept intact.
 *
 * `status` is load-bearing, not decoration: a 404 from an `/admin/*`
 * route means "this build does not have the control surface" and each
 * screen renders a different, honest empty state for it. Collapsing
 * every failure into one message would make that impossible.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }

  get isMissingEndpoint(): boolean {
    return this.status === 404 || this.status === 405;
  }

  get isAuth(): boolean {
    return this.status === 401 || this.status === 403;
  }
}

/** An error body, in either of the two shapes this surface uses. */
type ErrorBody = {
  error?: { message?: string };
  message?: string;
};

async function parse<T>(response: Response): Promise<T> {
  const text = await response.text();
  let body: unknown = null;
  try {
    if (text) body = JSON.parse(text);
  } catch {
    // A non-JSON body is still an answer; the raw text becomes the
    // message below rather than being swallowed.
  }
  if (!response.ok) {
    const shaped = body as ErrorBody | null;
    const message =
      shaped?.error?.message ||
      shaped?.message ||
      text.slice(0, 300) ||
      response.statusText;
    throw new ApiError(response.status, message, body);
  }
  return body as T;
}

/** Whatever `fetch` rejected with, as a sentence. */
function reason(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

function isAbort(cause: unknown): boolean {
  return cause instanceof Error && cause.name === "AbortError";
}

export async function getJson<T = unknown>(
  path: string,
  { signal }: { signal?: AbortSignal } = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url(path), { headers: headers(), signal });
  } catch (cause) {
    if (isAbort(cause)) throw cause;
    throw new ApiError(0, `cannot reach ${baseUrl()}: ${reason(cause)}`, null);
  }
  return parse<T>(response);
}

export async function postJson<T = unknown>(
  path: string,
  payload?: unknown,
  { signal }: { signal?: AbortSignal } = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url(path), {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: payload === undefined ? "{}" : JSON.stringify(payload),
      signal,
    });
  } catch (cause) {
    if (isAbort(cause)) throw cause;
    throw new ApiError(0, `cannot reach ${baseUrl()}: ${reason(cause)}`, null);
  }
  return parse<T>(response);
}

/**
 * Ask the server to stop the generation behind `requestId`.
 *
 * The second tier of cancellation. Aborting the fetch closes the socket
 * and the server now notices that too, but a socket is not a reliable
 * signal: a reverse proxy can hold the connection open on the backend
 * side, and a page unload races the abort it is supposed to send.
 * `keepalive: true` is what makes this one outlive the page, which is
 * exactly the case the socket tier is worst at.
 *
 * Deliberately not awaited by callers and deliberately silent: it is
 * best-effort cleanup, and a 404 here only means the answer finished
 * first. Nothing the user did went wrong, so nothing is shown.
 */
export function cancelGeneration(requestId: string | null | undefined): void {
  if (!requestId) return;
  try {
    void fetch(url(routes.cancel), {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ request_id: requestId }),
      keepalive: true,
    }).catch(() => {});
  } catch {
    /* the stream is already being torn down; there is nothing to report */
  }
}

// --------------------------------------------------------------------
// Streaming chat
// --------------------------------------------------------------------

/**
 * How long a stream may go without delivering a single byte before the
 * UI says so.
 *
 * Measured against *bytes*, not tokens, which is what makes the number
 * safe to keep low-ish: the server sends an SSE keep-alive comment
 * every 15 s, so a healthy stream resets this timer four times a minute
 * even while a long prompt is still prefilling. A slow model therefore
 * never trips it; a proxy that swallowed the connection does.
 *
 * It never aborts a generation to show a tidier error. What it does do,
 * when the request was resumable, is stop *reading this socket* and
 * drain the same answer over the polling fallback instead — the
 * generation is untouched on the server and the tokens already paid for
 * are still delivered. See `recover`.
 */
const STALL_MS = 45000;

/** Reconnect delay used when the server's `retry:` was never seen. */
const DEFAULT_RETRY_MS = 1500;

/** SSE reconnects tried before falling back to polling. */
const SSE_RECONNECTS = 2;

/** Consecutive polling failures tolerated before giving up. */
const POLL_RETRIES = 3;

/** The per-phase timings the server states in `usage`. */
export type Usage = {
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  prompt_per_second?: number | null;
  predicted_per_second?: number | null;
  prompt_eval_duration_ms?: number | null;
  generation_duration_ms?: number | null;
  time_to_first_token_ms?: number | null;
  cached_tokens?: number | null;
};

export type ChatMessage = {
  role: string;
  content: string;
  /**
   * A replayed assistant turn's chain of thought, in the server's own
   * field. Sent only on the one turn being CONTINUED: for ordinary
   * history the template strips it anyway, and the DeepSeek guidance
   * is not to replay it.
   */
  reasoning_content?: string;
};

export type ChatRequest = {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  /** Omitted, the server bounds the answer by the context alone. */
  max_tokens?: number;
  /**
   * llama.cpp's field: tokens of thinking allowed before the server
   * forces the closing tag. Omitted, the server's `--reasoning-budget`
   * default applies; `0` ends the thought as soon as it opens.
   */
  reasoning_budget_tokens?: number;
  /**
   * llama.cpp's field: render the trailing assistant message as a turn
   * still being written, so the model carries on from where it stopped
   * rather than starting a new one. The server does this by default
   * for a trailing assistant message; Studio still says so explicitly.
   */
  continue_final_message?: boolean;
};

/** One SSE frame's payload, as far as this client reads it. */
type StreamChunk = {
  request_id?: string;
  usage?: Usage;
  choices?: {
    finish_reason?: string | null;
    delta?: {
      content?: string | null;
      /**
       * A reasoning model's thinking, which the server streams in its
       * OWN field rather than inside `content` (llama.cpp and DeepSeek
       * both do this). A client that reads only `content` shows
       * nothing at all while a reasoning model thinks, which on an R1
       * distill is most of the answer's wall-clock: the transcript sits
       * empty, the stream looks dead, and then the answer lands whole.
       */
      reasoning_content?: string | null;
    };
  }[];
};

export type StreamResult = {
  requestId: string | null;
  usage: Usage | null;
  finishReason: string | null;
};

/**
 * How the answer is currently arriving.
 *
 * Worth reporting rather than hiding: "still streaming" and "the
 * original connection died and this is a reconnect" are different
 * facts, and a UI that shows neither leaves the user watching a
 * progress indicator that means nothing.
 */
export type Transport = "stream" | "resumed" | "polling";

export type StreamHandlers = {
  signal?: AbortSignal;
  onToken?: (text: string) => void;
  /**
   * A reasoning model's thinking, as it arrives.
   *
   * Kept separate from `onToken` rather than concatenated into it: the
   * two are shown differently, and only one of them is the answer. A
   * caller that does not set this simply does not render thinking,
   * which is the old behaviour.
   */
  onReasoning?: (text: string) => void;
  onRequestId?: (id: string) => void;
  onStall?: (ms: number | null) => void;
  onTransport?: (transport: Transport) => void;
  stallMs?: number;
  /**
   * Ask the server for a replay buffer (`stream_resumable`).
   *
   * Default on, and that is a real trade rather than a free win: a
   * resumable request is NOT cancelled by its socket closing, so the
   * explicit `POST /v1/cancel` becomes the only stop path. This app
   * always sends that — on Stop, on New chat, on leaving the screen and
   * on `pagehide` with `keepalive` — which is exactly the set of cases
   * a socket close was standing in for.
   */
  resumable?: boolean;
};

/** Everything one logical answer accumulates, across reconnects. */
type StreamState = {
  requestId: string | null;
  usage: Usage | null;
  finishReason: string | null;
  /** `[DONE]` was seen. */
  done: boolean;
  /** Index of the first event NOT yet consumed, from the `id:` fields. */
  nextIndex: number;
  /** The most recent `id:` value, sent verbatim on a reconnect. */
  lastEventId: string | null;
  /** The server's `retry:`, when it stated one. */
  retryMs: number;
};

function newState(): StreamState {
  return {
    requestId: null,
    usage: null,
    finishReason: null,
    done: false,
    nextIndex: 0,
    lastEventId: null,
    retryMs: DEFAULT_RETRY_MS,
  };
}

/** A completed answer: `[DONE]` or a finish reason. Anything else is not. */
function complete(state: StreamState): boolean {
  return state.done || state.finishReason !== null;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function truncated(detail: string): ApiError {
  return new ApiError(
    0,
    `the stream ended without a finish reason — the response was truncated, not completed (${detail})`,
    null,
  );
}

/** Applies one `data:` payload. Shared by live, resumed and polled events. */
function applyEvent(
  state: StreamState,
  payload: string,
  handlers: StreamHandlers,
): void {
  if (payload === "[DONE]") {
    state.done = true;
    return;
  }
  let chunk: StreamChunk;
  try {
    chunk = JSON.parse(payload) as StreamChunk;
  } catch {
    return; // a keep-alive comment or a partial frame; not fatal
  }
  if (chunk.request_id && !state.requestId) {
    state.requestId = chunk.request_id;
    handlers.onRequestId?.(chunk.request_id);
  }
  if (chunk.usage) state.usage = chunk.usage;
  const choice = chunk.choices?.[0];
  if (choice?.finish_reason) state.finishReason = choice.finish_reason;
  // Reasoning first, because that is the order the server sends it in
  // and the order it has to be shown in: thinking, then the answer it
  // led to.
  const reasoning = choice?.delta?.reasoning_content;
  if (reasoning) handlers.onReasoning?.(reasoning);
  const text = choice?.delta?.content;
  if (text) handlers.onToken?.(text);
}

/**
 * Drains one SSE body into `state`.
 *
 * Reads `id:` as well as `data:`. The id is what makes a reconnect
 * resume rather than restart, and it is qualified by the request id
 * server-side precisely so it cannot be mistaken for a position in some
 * other stream.
 *
 * `onStall` reports a stream that has gone quiet and, when the caller
 * can fall back, `stallSwitch` is called so the reader can be abandoned
 * in favour of polling. Nothing here ever cancels the generation.
 */
async function readSse(
  body: ReadableStream<Uint8Array>,
  state: StreamState,
  handlers: StreamHandlers,
  stallMs: number,
  stallSwitch?: () => void,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  let stalled = false;
  let stallTimer: ReturnType<typeof setTimeout> | undefined;
  const armStallTimer = () => {
    clearTimeout(stallTimer);
    if (!handlers.onStall && !stallSwitch) return;
    stallTimer = setTimeout(() => {
      stalled = true;
      handlers.onStall?.(stallMs);
      stallSwitch?.();
    }, stallMs);
  };

  try {
    armStallTimer();
    for (;;) {
      const { value, done: streamDone } = await reader.read();
      if (streamDone) break;
      armStallTimer();
      if (stalled) {
        // It came back. Saying so matters as much as saying it stopped:
        // a banner left up after recovery is a lie with a long tail.
        stalled = false;
        handlers.onStall?.(null);
      }
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by a blank line; a field may be split
      // across reads, so only complete frames are consumed.
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        for (const line of frame.split("\n")) {
          if (line.startsWith("data:")) {
            applyEvent(state, line.slice(5).trimStart(), handlers);
          } else if (line.startsWith("id:")) {
            const id = line.slice(3).trim();
            state.lastEventId = id;
            const index = Number(id.slice(id.lastIndexOf(":") + 1));
            if (Number.isFinite(index)) state.nextIndex = index + 1;
          } else if (line.startsWith("retry:")) {
            const ms = Number(line.slice(6).trim());
            if (Number.isFinite(ms) && ms > 0) state.retryMs = ms;
          }
        }
        boundary = buffer.indexOf("\n\n");
      }
    }
  } finally {
    clearTimeout(stallTimer);
    if (stalled) handlers.onStall?.(null);
  }
}

type PollBody = {
  events?: { index: number; data: string }[];
  next_index?: number;
  done?: boolean;
};

/**
 * Drains the rest of the answer over `GET /v1/stream/{id}/poll`.
 *
 * The fallback the whole feature exists for. A reverse proxy that
 * buffers `text/event-stream` — nginx's default and that of everything
 * that copied it — turns a token-by-token stream into one long silence,
 * which from here is indistinguishable from a hung backend. It cannot
 * do that to a short JSON response that has already ended.
 *
 * `done` from the server is trusted over any local guess: it is `true`
 * only once the generation has ended AND the buffer is drained, so this
 * never stops holding events it has not been given.
 */
async function drainByPolling(
  requestId: string,
  state: StreamState,
  handlers: StreamHandlers,
): Promise<void> {
  handlers.onTransport?.("polling");
  let failures = 0;
  for (;;) {
    if (handlers.signal?.aborted) {
      throw new DOMException("aborted", "AbortError");
    }
    let body: PollBody;
    try {
      body = await getJson<PollBody>(
        `${routes.streamPoll(requestId)}?from=${state.nextIndex}`,
        { signal: handlers.signal },
      );
      failures = 0;
    } catch (cause) {
      if (isAbort(cause)) throw cause;
      if (cause instanceof ApiError && cause.status >= 400) {
        // 404 (forgotten), 410 (replay window passed), 400 (bad
        // cursor). All three mean the rest of this answer is
        // unreachable, and a partial answer must not be presented as a
        // whole one.
        throw truncated(cause.message);
      }
      if (++failures > POLL_RETRIES) throw truncated(cause instanceof Error ? cause.message : String(cause));
      await sleep(state.retryMs);
      continue;
    }
    for (const event of body.events ?? []) {
      applyEvent(state, event.data, handlers);
    }
    if (typeof body.next_index === "number") state.nextIndex = body.next_index;
    if (body.done || complete(state)) return;
  }
}

/** One reconnect attempt over SSE, continuing after the last id seen. */
async function resumeOverSse(
  requestId: string,
  state: StreamState,
  handlers: StreamHandlers,
  stallMs: number,
): Promise<boolean> {
  const extra: Record<string, string> = { Accept: "text/event-stream" };
  if (state.lastEventId) extra["Last-Event-ID"] = state.lastEventId;
  let response: Response;
  try {
    response = await fetch(url(routes.stream(requestId)), {
      headers: headers(extra),
      signal: handlers.signal,
    });
  } catch (cause) {
    if (isAbort(cause)) throw cause;
    return false; // the network refused; the caller tries polling next
  }
  if (response.status >= 400) {
    // The server refusing by status is a decided answer, not a hiccup:
    // 410 means the replay window has passed this position and resuming
    // would skip part of the answer without either end being able to
    // tell. Fail closed rather than retrying into it.
    await parse(response).catch((error: unknown) => {
      throw truncated(error instanceof Error ? error.message : String(error));
    });
    return false;
  }
  if (!response.body) return false;
  handlers.onTransport?.("resumed");
  await readSse(response.body, state, handlers, stallMs);
  return true;
}

/**
 * Finishes an answer whose live connection did not.
 *
 * Order is deliberate. A stall means the bytes stopped arriving while
 * the connection stayed open, which is the signature of a buffering
 * proxy — and a second SSE connection would go through the same proxy,
 * so that case skips straight to polling. Anything else gets SSE
 * reconnects first (cheaper, and it keeps streaming) before falling
 * back.
 */
async function recover(
  state: StreamState,
  handlers: StreamHandlers,
  stallMs: number,
  preferPolling: boolean,
): Promise<void> {
  const requestId = state.requestId;
  if (!requestId) return;
  if (!preferPolling) {
    for (let attempt = 0; attempt < SSE_RECONNECTS; attempt++) {
      if (complete(state)) return;
      await sleep(state.retryMs);
      await resumeOverSse(requestId, state, handlers, stallMs);
    }
  }
  if (complete(state)) return;
  await drainByPolling(requestId, state, handlers);
}

/**
 * POST a chat completion with `stream: true` and drive the callbacks.
 *
 * Four things the server states that this reads rather than guesses:
 *
 * - `request_id` arrives on the **first** chunk only, so it is captured
 *   once and never re-derived. A UI that instead claimed an id out of a
 *   stats snapshot would mis-attribute the moment two chats overlap.
 * - `usage` arrives on the **final** chunk and carries per-phase
 *   timings. Client wall-clock is never substituted for them.
 * - `id:` names the position of every event, so a reconnect resumes
 *   instead of restarting, and `retry:` says how long to wait first.
 * - A stream that ends without `[DONE]` and without a `finish_reason`
 *   was truncated. Recovery is attempted — reconnect, then poll — and
 *   only if that also fails is it surfaced as an error, never as a
 *   finished message.
 */
export async function streamChat(
  request: ChatRequest,
  handlers: StreamHandlers = {},
): Promise<StreamResult> {
  const { signal, stallMs = STALL_MS, resumable = true } = handlers;
  const state = newState();

  // A private controller, so the stall fallback can stop reading this
  // socket without touching the caller's signal — and so a caller
  // abort still reads as an abort.
  const controller = new AbortController();
  let userAborted = false;
  const onUserAbort = () => {
    userAborted = true;
    controller.abort(new DOMException("aborted", "AbortError"));
  };
  if (signal?.aborted) onUserAbort();
  signal?.addEventListener("abort", onUserAbort, { once: true });

  let switchedToPolling = false;
  const inner: StreamHandlers = { ...handlers, signal };

  try {
    const response = await fetch(url(routes.chatCompletions), {
      method: "POST",
      headers: headers({
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      }),
      body: JSON.stringify({
        ...request,
        stream: true,
        stream_resumable: resumable,
      }),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      // An error answer is JSON, not SSE — parse it as such. `parse`
      // throws for anything non-2xx, so control only continues past
      // here on the pathological "200 with no body", which is not a
      // completion either and is reported the same way.
      return parse<StreamResult>(response);
    }

    handlers.onTransport?.("stream");
    try {
      await readSse(response.body, state, inner, stallMs, () => {
        // Only worth abandoning the socket if there is another way in.
        if (!resumable || !state.requestId) return;
        switchedToPolling = true;
        controller.abort();
      });
    } catch (cause) {
      if (!switchedToPolling) throw cause;
    }

    if (!complete(state) && resumable && !userAborted) {
      await recover(state, inner, stallMs, switchedToPolling);
    }
  } finally {
    signal?.removeEventListener("abort", onUserAbort);
  }

  if (!complete(state)) {
    throw truncated("no reconnect or poll could finish it");
  }

  return {
    requestId: state.requestId,
    usage: state.usage,
    finishReason: state.finishReason,
  };
}

// --------------------------------------------------------------------
// Response shapes the screens read
// --------------------------------------------------------------------

export type Capability = {
  id: string;
  available: boolean;
  reason: string;
  detail: string;
};

export type Health = {
  state: "ready" | "detecting" | "unavailable" | string;
  detail?: string | null;
  reason?: string | null;
  version?: string;
  pid?: number;
  capabilities?: Capability[];
  model?: {
    id: string;
    tokenizer?: string;
    synthetic_weights?: boolean;
  } | null;
  last_request_age_seconds?: number | null;
};

export type ModelEntry = {
  id: string;
  path?: string;
  state?: string;
  error?: string | null;
  quant?: string | null;
  arch?: string | null;
  context_length?: number | null;
  param_count?: number | null;
  size_bytes?: number | null;
  resident_bytes?: number | null;
};

export type Inventory = {
  active?: string | null;
  model_dir?: string | null;
  models: ModelEntry[];
};

export type TaskProgress = {
  state?: string;
  fraction?: number | null;
  bytes_done?: number | null;
  bytes_total?: number | null;
  rate_bytes_per_s?: number | null;
  eta_seconds?: number | null;
};

export type TaskView = {
  task_id: string;
  label: string;
  status: string;
  error?: string | null;
  progress?: TaskProgress | null;
};

export type StatsRow = {
  at_ms: number;
  request_id: string;
  route: string;
  /**
   * The model that SERVED this request, as /v1/models names it — not
   * the `model` field the request carried, which this server ignores.
   * Null when nothing was loaded.
   */
  model?: string | null;
  status: number;
  stream: boolean;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  ttft_ms?: number | null;
  duration_ms?: number | null;
  decode_ms?: number | null;
  /**
   * Fingerprint of the bearer key that served this request, or null
   * when none was presented. Never the key itself — the server salts it
   * per process, so it is comparable between rows of one server run and
   * meaningless anywhere else.
   */
  via_api_key?: string | null;
  /** The caller's SELF-DECLARED label. Not authenticated. */
  client?: string | null;
};

export type Stats = {
  uptime_seconds?: number | null;
  requests_total?: number | null;
  errors_total?: number | null;
  cache_hits?: number | null;
  cache_misses?: number | null;
  tokens_prompt_total?: number | null;
  tokens_generated_total?: number | null;
  generating_now?: number | null;
  /**
   * Requests waiting for a decode slot. `null` — not 0 — when continuous
   * batching is off, because then there is no queue to measure.
   */
  queue_depth?: number | null;
  queue_rejected_total?: number | null;
  last_request_age_seconds?: number | null;
  recent?: StatsRow[];
};
