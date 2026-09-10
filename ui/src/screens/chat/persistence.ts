import { useCallback, useEffect, useRef, useState } from "react";
import {
  ExportedMessageRepository,
  type AssistantRuntime,
} from "@assistant-ui/react";
import { ApiError } from "@/lib/api";
import {
  createConversation,
  deleteConversation,
  getConversation,
  hasWork,
  listConversations,
  pendingAppend,
  storedIds,
  toBranchable,
  updateConversation,
  type ConversationSummary,
  type ExportedRepository,
} from "@/lib/conversations";
import {
  decideEntry,
  LOCAL_SLOT,
  readTabActiveAt,
  type EntryDecision,
} from "@/lib/entry-state";
import { useLatest } from "@/lib/use-latest";

// Where the transcript lives.
//
// Server-side when the server has `/v1/conversations`, which it now
// does: the tree is stored there, keyed by message id and parent id, so
// it survives a reload, a different browser and a cleared profile, and
// edit/regenerate branches survive with it.
//
// `localStorage` is still here as the fallback for a server that
// answers 404 on that route -- an older build, or this app pointed at
// something else that speaks the OpenAI API. The screen says which of
// the two is in use, because "your chats are saved" and "your chats are
// saved in this browser" are different promises and only one of them
// survives a laptop.
//
// The one thing neither mode does is guess. When the store cannot be
// reached, nothing is silently dropped and nothing is silently kept
// somewhere else: the mode is reported and the reason with it.
//
// **Which conversation is open is not decided here.** In server mode the
// URL decides it and this module follows; the whole rule, including what
// "after a while" means, lives in `lib/entry-state.ts` and is applied in
// exactly one place below. Local mode has no ids to put in a URL, so it
// asks the same function with the one slot it has.

const LOCAL_KEY = "ferrox.studio.thread.v3";
/** The shape before the transcript carried its own save time. */
const LEGACY_LOCAL_KEY = "ferrox.studio.thread.v2";

type LocalTranscript = {
  /** When this was written. The entry rule needs an age, and a blob that
   * cannot say how old it is cannot be aged out. */
  savedAt: number | null;
  exported: { messages: unknown[] };
};

/** Anything unparseable is dropped: a corrupt blob must not wedge Chat. */
function readLocal(): LocalTranscript | null {
  try {
    const raw = localStorage.getItem(LOCAL_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<LocalTranscript>;
      const messages = parsed?.exported?.messages;
      if (!Array.isArray(messages) || !messages.length) return null;
      return {
        savedAt:
          typeof parsed.savedAt === "number" && Number.isFinite(parsed.savedAt)
            ? parsed.savedAt
            : null,
        exported: { messages },
      };
    }
    // A transcript written before this key carried a timestamp. It has no
    // age, and `decideEntry` reads a missing age as "no evidence of being
    // away", so it is restored once and rewritten in the new shape.
    const legacy = localStorage.getItem(LEGACY_LOCAL_KEY);
    if (!legacy) return null;
    const parsed = JSON.parse(legacy) as { messages?: unknown[] };
    if (!Array.isArray(parsed?.messages) || !parsed.messages.length) return null;
    return { savedAt: null, exported: { messages: parsed.messages } };
  } catch {
    return null;
  }
}

function writeLocal(exported: { messages: unknown[] }) {
  try {
    // An empty transcript is never written, and never clears a stored
    // one. A new chat starts empty while the previous transcript is
    // still being offered back, and a write here would delete the thing
    // the offer points at. Clearing is an explicit act: `newChat`.
    if (!exported.messages.length) return;
    localStorage.setItem(
      LOCAL_KEY,
      JSON.stringify({ savedAt: Date.now(), exported } satisfies LocalTranscript),
    );
    localStorage.removeItem(LEGACY_LOCAL_KEY);
  } catch {
    /* quota or private browsing — the in-memory thread still works */
  }
}

export function clearLocalTranscript() {
  try {
    localStorage.removeItem(LOCAL_KEY);
    localStorage.removeItem(LEGACY_LOCAL_KEY);
  } catch {
    /* private browsing: nothing was persisted to begin with */
  }
}

/** How long the transcript sits still before it is written. A decode
 * loop notifies once per token; writing on each would put a request
 * between every pair of them. */
const DEBOUNCE_MS = 500;

export type TranscriptMode = "checking" | "server" | "local";

/**
 * A conversation that was NOT opened, and could be.
 *
 * Set only when the entry rule declined one — the tab had been away
 * longer than the resume window. It is what turns "you have been moved"
 * into "you have been moved, here is the way back".
 */
export type StaleOffer = {
  /** A conversation id, or `LOCAL_SLOT` in local mode. */
  id: string;
  label: string;
  awayMs: number | null;
};

export type Transcript = {
  mode: TranscriptMode;
  /** Why the transcript is browser-local, when it is. */
  reason: string | null;
  /** A write is in flight. */
  saving: boolean;
  /** The last write or read that failed, as a sentence. */
  error: string | null;
  current: { id: string; title: string | null } | null;
  summaries: ConversationSummary[];
  /** The conversation the entry rule declined to reopen, if any. */
  stale: StaleOffer | null;
  refresh: () => void;
  open: (id: string) => void;
  newChat: () => void;
  remove: (id: string) => void;
  /** Take up the declined conversation after all. */
  resumeStale: () => void;
  /** Drop the offer without taking it. */
  dismissStale: () => void;
  /** The tab came back after being away. Applied here rather than in the
   * screen so there is one implementation of "go fresh, offer the way
   * back", shared with the one that runs at load. */
  onReturnedAfterAway: (awayMs: number) => void;
};

type SyncState = {
  mode: TranscriptMode;
  conversationId: string | null;
  /** Ids the server is holding. Only ever grown from what the server
   * echoed back, never from what was sent -- a request that failed
   * halfway must not leave this claiming the messages landed. */
  stored: Set<string>;
  storedHead: string | null;
  busy: boolean;
  dirty: boolean;
  /** Writes are held off while the thread is being replaced from the
   * outside (a load, a reset), so an import is not immediately written
   * back as if the user had typed it. */
  suspended: boolean;
  migrating: boolean;
  /**
   * The conversation the entry rule DECLINED to open.
   *
   * The URL cannot be corrected synchronously — asking the router to
   * replace it is a state update, and the route effect runs once more on
   * the old id before the new one arrives. Without this the effect
   * loaded exactly the conversation the rule had just declined, which is
   * how the rule came to fire, print its banner, and then be undone by
   * the very next effect. Cleared the moment the route points anywhere
   * else.
   */
  declined: string | null;
};

function sentence(cause: unknown): string {
  if (cause instanceof ApiError && cause.isAuth)
    return `${cause.message} — set the API key on the Connect screen.`;
  return cause instanceof Error ? cause.message : String(cause);
}

/**
 * Keep the thread and the server's conversation store in step.
 *
 * The loop is one-directional by design: assistant-ui owns the
 * transcript in the tab, and every change is pushed to the server as an
 * append. Nothing is ever pulled back into a live thread, because two
 * writers on one transcript is how a message gets lost, and this server
 * serves one person's Chat screen.
 *
 * `routeId` is the conversation the URL points at, and `onRoute` is how
 * this asks for a different one. Opening a conversation is therefore a
 * navigation, never a direct load: one path in, so the address bar
 * cannot disagree with what is on screen.
 */
export function useTranscript(
  runtime: AssistantRuntime,
  deps: {
    model: () => string | null;
    routeId: string | null;
    onRoute: (id: string | null, opts?: { replace?: boolean }) => void;
  },
): Transcript {
  const [mode, setMode] = useState<TranscriptMode>("checking");
  const [reason, setReason] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [current, setCurrent] = useState<{
    id: string;
    title: string | null;
  } | null>(null);
  const [summaries, setSummaries] = useState<ConversationSummary[]>([]);
  const [stale, setStale] = useState<StaleOffer | null>(null);

  // Read during the FIRST RENDER, before any effect has run.
  //
  // The heartbeat in `useTabActivity` stamps the tab as awake from its
  // own mount effect, and the probe below reads this from inside a
  // promise — so by the time the probe asks, the heartbeat has already
  // overwritten the only evidence that the tab was away. Capturing it
  // here, synchronously, is what makes the load-time half of the entry
  // rule reachable at all. It was not, when this was a call inside the
  // probe: the tab always looked freshly active and the rule never
  // fired once.
  const [entryActiveAt] = useState<number | null>(readTabActiveAt);

  const depsRef = useLatest(deps);
  const sync = useRef<SyncState>({
    mode: "checking",
    conversationId: null,
    stored: new Set(),
    storedHead: null,
    busy: false,
    dirty: false,
    suspended: true,
    migrating: false,
    declined: null,
  });

  const setBoth = useCallback((next: TranscriptMode) => {
    sync.current.mode = next;
    setMode(next);
  }, []);

  const refresh = useCallback(() => {
    if (sync.current.mode !== "server") return;
    listConversations()
      .then(setSummaries)
      .catch(() => {
        // A failed listing is not worth an error banner: the store is
        // still writable, and the next refresh will say otherwise if it
        // is not.
      });
  }, []);

  const flush = useCallback(async () => {
    const s = sync.current;
    if (s.suspended) return;

    if (s.mode === "local") {
      writeLocal(runtime.thread.export() as { messages: unknown[] });
      return;
    }
    if (s.mode !== "server") return;
    if (s.busy) {
      s.dirty = true;
      return;
    }

    // assistant-ui's exported shape is wider than the four fields the
    // store reads; the cast names that rather than restating the
    // library's type here, where it would drift.
    const exported = runtime.thread.export() as unknown as ExportedRepository;
    const pending = pendingAppend(exported, s.stored);
    if (!hasWork(pending, s.storedHead)) return;

    s.busy = true;
    setSaving(true);
    try {
      const head = pending.headId ? { head_id: pending.headId } : {};
      const created = !s.conversationId;
      const conversation = s.conversationId
        ? await updateConversation(s.conversationId, {
            append: pending.messages,
            ...head,
          })
        : await createConversation({
            messages: pending.messages,
            ...head,
            // Recorded, not enforced. It says which checkpoint was
            // loaded when the conversation started, which is a fact
            // worth keeping when the answer is later read back.
            ...(depsRef.current.model() ? { model: depsRef.current.model()! } : {}),
          });
      s.conversationId = conversation.id;
      s.stored = storedIds(conversation);
      s.storedHead = conversation.head_id;
      setCurrent({ id: conversation.id, title: conversation.title });
      setError(null);
      if (created) {
        // The first send is what turns a new chat into a conversation,
        // so it is also what puts it in the address bar. `replace`, not
        // `push`: Back should leave Chat, not walk backwards into the
        // empty version of the conversation you are looking at.
        depsRef.current.onRoute(conversation.id, { replace: true });
      }
      if (s.migrating) {
        // The browser copy is dropped only once the server has echoed
        // the messages back. Clearing it on the way out would lose the
        // transcript if the write failed.
        clearLocalTranscript();
        s.migrating = false;
      }
      refresh();
    } catch (cause) {
      // `stored` is deliberately untouched, so the same nodes are
      // offered again on the next change rather than being counted as
      // saved.
      setError(sentence(cause));
    } finally {
      s.busy = false;
      setSaving(false);
      if (s.dirty) {
        s.dirty = false;
        void flush();
      }
    }
  }, [runtime, depsRef, refresh]);

  /** Empty the thread and forget which conversation it was. Does not
   * navigate and does not delete anything: the callers decide both. */
  const resetThread = useCallback(() => {
    const s = sync.current;
    s.suspended = true;
    runtime.thread.cancelRun();
    runtime.thread.reset();
    s.conversationId = null;
    s.stored = new Set();
    s.storedHead = null;
    s.migrating = false;
    setCurrent(null);
    setError(null);
    s.suspended = false;
  }, [runtime]);

  /** Pull a conversation out of the store and into the thread. Called
   * only from the route effect and the first landing, so there is one
   * way for a conversation to become the open one. */
  const load = useCallback(
    async (id: string) => {
      const s = sync.current;
      s.suspended = true;
      setError(null);
      try {
        const conversation = await getConversation(id);
        runtime.thread.cancelRun();
        const { items, headId } = toBranchable(conversation);
        runtime.thread.import(
          ExportedMessageRepository.fromBranchableArray(items, { headId }),
        );
        s.conversationId = conversation.id;
        s.stored = storedIds(conversation);
        s.storedHead = conversation.head_id;
        setCurrent({ id: conversation.id, title: conversation.title });
      } catch (cause) {
        setError(sentence(cause));
      } finally {
        s.suspended = false;
      }
    },
    [runtime],
  );

  const newChat = useCallback(() => {
    resetThread();
    setStale(null);
    sync.current.declined = null;
    if (sync.current.mode === "local") clearLocalTranscript();
    depsRef.current.onRoute(null);
  }, [resetThread, depsRef]);

  const remove = useCallback(
    async (id: string) => {
      try {
        await deleteConversation(id);
        if (sync.current.conversationId === id) newChat();
        setStale((offer) => (offer?.id === id ? null : offer));
        refresh();
      } catch (cause) {
        setError(sentence(cause));
      }
    },
    [newChat, refresh],
  );

  /**
   * Apply an entry decision.
   *
   * One implementation, three callers: the first landing in server mode,
   * the first landing in local mode, and the tab coming back after being
   * away. A second copy of "reset, navigate, remember what was declined"
   * is exactly the shape that drifts.
   */
  const applyEntry = useCallback(
    (decision: EntryDecision, label: (id: string) => string) => {
      if (decision.kind === "resume") return decision.conversationId;
      sync.current.declined = decision.resumable;
      if (decision.resumable) {
        setStale({
          id: decision.resumable,
          label: label(decision.resumable),
          awayMs: decision.awayMs,
        });
      }
      return null;
    },
    [],
  );

  const resumeStale = useCallback(() => {
    const offer = stale;
    if (!offer) return;
    setStale(null);
    // Explicitly asked for, so it is no longer declined -- and the route
    // effect below would otherwise skip the very navigation this makes.
    sync.current.declined = null;
    if (offer.id === LOCAL_SLOT) {
      const local = readLocal();
      if (!local) return;
      const s = sync.current;
      s.suspended = true;
      try {
        runtime.thread.import(local.exported as never);
      } catch {
        clearLocalTranscript();
      } finally {
        s.suspended = false;
      }
      return;
    }
    depsRef.current.onRoute(offer.id);
  }, [stale, runtime, depsRef]);

  const dismissStale = useCallback(() => setStale(null), []);

  const onReturnedAfterAway = useCallback(
    (awayMs: number) => {
      const s = sync.current;
      if (s.mode === "checking") return;
      const openId =
        s.mode === "local"
          ? readLocal()
            ? LOCAL_SLOT
            : null
          : s.conversationId;
      if (!openId) return;
      const label =
        openId === LOCAL_SLOT
          ? "your last chat"
          : (current?.title?.trim() || "the last conversation");
      resetThread();
      // Same call the first landing makes, so "go fresh and offer the way
      // back" has one implementation and the two triggers cannot drift.
      applyEntry({ kind: "fresh", awayMs, resumable: openId }, () => label);
      if (s.mode === "server") depsRef.current.onRoute(null, { replace: true });
    },
    [current, resetThread, applyEntry, depsRef],
  );

  // Probe once: does this server keep conversations at all? The answer
  // decides the mode, and the mode decides which half of the entry rule
  // applies.
  useEffect(() => {
    let cancelled = false;
    const s = sync.current;
    const entryRouteId = depsRef.current.routeId;

    listConversations()
      .then(async (list) => {
        if (cancelled) return;
        setBoth("server");
        setSummaries(list);

        const titleOf = (id: string) =>
          list.find((entry) => entry.id === id)?.title?.trim() ||
          "the last conversation";
        const open = applyEntry(
          decideEntry({
            conversationId: entryRouteId,
            lastActiveAt: entryActiveAt,
            now: Date.now(),
          }),
          titleOf,
        );

        if (open) {
          await load(open);
          return;
        }
        // A new chat. The URL is put back to the bare `/ui/chat` so the
        // address bar and the screen agree; without this a reload would
        // walk straight back into the conversation just declined.
        if (entryRouteId) depsRef.current.onRoute(null, { replace: true });

        const local = readLocal();
        if (local) {
          // A transcript from before this server had a store. It is
          // imported into the thread and then written through the
          // normal sync path, so there is one code path that creates a
          // conversation rather than two.
          try {
            runtime.thread.import(local.exported as never);
            s.migrating = true;
          } catch {
            clearLocalTranscript();
          }
        }
      })
      .catch((cause) => {
        if (cancelled) return;
        setBoth("local");
        setReason(
          cause instanceof ApiError && cause.isMissingEndpoint
            ? "This server has no conversation API, so the transcript is kept in this browser only."
            : `The conversation store could not be reached (${sentence(cause)}), so the transcript is kept in this browser only.`,
        );
        // No ids exist in this mode, so a conversation id in the URL
        // cannot mean anything: the address bar is put back to the bare
        // path rather than left pointing at something unreachable.
        if (entryRouteId) depsRef.current.onRoute(null, { replace: true });

        const local = readLocal();
        const open = applyEntry(
          decideEntry({
            conversationId: local ? LOCAL_SLOT : null,
            lastActiveAt: local?.savedAt ?? null,
            now: Date.now(),
          }),
          () => "your last chat",
        );
        if (open && local) {
          try {
            runtime.thread.import(local.exported as never);
          } catch {
            clearLocalTranscript();
          }
        }
      })
      .finally(() => {
        if (!cancelled) s.suspended = false;
      });

    return () => {
      cancelled = true;
    };
    // Runs once for the life of the runtime: a re-probe would replace a
    // thread the user is in the middle of.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runtime]);

  // The URL is the single source of truth for which conversation is
  // open. Every other path -- the picker, New chat, a created
  // conversation, the Back button -- moves the URL and lands here.
  //
  // The lint rule below is about effects that write state React already
  // knows; this one drives an EXTERNAL system, the assistant-ui runtime
  // holding the thread, and the state it touches is the label describing
  // what that runtime now contains. It fires only when the route id
  // actually changes, so there is no cascade.
  const routeId = deps.routeId;
  useEffect(() => {
    const s = sync.current;
    if (s.mode !== "server") return;
    if (routeId !== null && routeId === s.declined) return;
    s.declined = null;
    if (routeId === s.conversationId) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (routeId === null) resetThread();
    else void load(routeId);
  }, [routeId, mode, load, resetThread]);

  // Coalesced writes. The thread notifies once per decoded token, and a
  // request per token would stutter the stream it is recording.
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const unsubscribe = runtime.thread.subscribe(() => {
      clearTimeout(timer);
      timer = setTimeout(() => void flush(), DEBOUNCE_MS);
    });
    return () => {
      clearTimeout(timer);
      unsubscribe();
    };
  }, [runtime, flush]);

  return {
    mode,
    reason,
    saving,
    error,
    current,
    summaries,
    stale,
    refresh,
    open: (id) => depsRef.current.onRoute(id),
    newChat,
    remove: (id) => void remove(id),
    resumeStale,
    dismissStale,
    onReturnedAfterAway,
  };
}
