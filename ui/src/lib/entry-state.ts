// Where Chat lands when you arrive, and why.
//
// Two facts decide it, and nothing else: **what the URL points at**, and
// **how long this tab has been away**. Keeping both in one pure function
// is the point of this module — the rule used to be a single line buried
// in the sync effect (`else if (list.length) await open(list[0].id)`),
// which meant every entry into the app, from any path, after any amount
// of time, resurrected whatever conversation happened to be newest. That
// is not a rule anybody chose; it is what "load the list, show
// something" turns into.
//
// The rule, in one sentence:
//
//   `/ui/chat` is always a new chat, `/ui/chat/<id>` is that
//   conversation, and a tab that has been away longer than the resume
//   window drops back to a new chat when you return to it.
//
// The window is the only number here, it is named once, and when it
// fires the screen says so out loud and offers the conversation back.
// A rule that silently moves the user is indistinguishable from a bug.

/**
 * How long a tab may be away before Chat stops calling the conversation
 * it was in "the one you are in".
 *
 * Thirty minutes, which is the usual definition of a session boundary
 * everywhere else that has to draw this line. Short enough that coming
 * back the next morning is a fresh start, long enough that lunch, a
 * meeting, or a long read of an answer is not.
 *
 * "Away" is measured, not assumed: the timestamp is refreshed on a
 * heartbeat only while the document is visible, so a hidden tab, a
 * closed lid and a sleeping machine all accumulate away-time, and a tab
 * you are sitting in front of never does.
 */
export const RESUME_WINDOW_MS = 30 * 60 * 1000;

/** How often the "this tab is awake" timestamp is refreshed. */
export const ACTIVITY_HEARTBEAT_MS = 30 * 1000;

/**
 * The local-mode stand-in for a conversation id.
 *
 * A server without `/v1/conversations` has exactly one transcript and no
 * ids to put in a URL, so the same decision is asked with this sentinel
 * and the stored transcript's own save time. One rule, two callers,
 * rather than two rules that drift.
 */
export const LOCAL_SLOT = "local";

export type EntryDecision =
  /** Open this conversation. */
  | { kind: "resume"; conversationId: string }
  /**
   * Start empty. `awayMs` is how long the tab had been away when that is
   * the reason, and `resumable` is the conversation that was declined,
   * so the screen can offer it back instead of losing it.
   */
  | { kind: "fresh"; awayMs: number | null; resumable: string | null };

export type EntryInput = {
  /** What the URL (or, in local mode, the store) points at. `null` is
   * the bare base URL: a new chat, with nothing to decide. */
  conversationId: string | null;
  /** When this tab was last known to be awake, or `null` if there is no
   * record — a fresh tab, a deep link from a bookmark, a browser that
   * refused the storage. No record is not evidence of being away. */
  lastActiveAt: number | null;
  now: number;
  windowMs?: number;
};

/**
 * Decide what Chat opens with.
 *
 * Ordered, and the order is the argument:
 *
 *  1. **No conversation is pointed at** — the base URL. A new chat, and
 *     nothing was declined, so nothing is offered back.
 *  2. **The tab has been away longer than the window.** A new chat, with
 *     the declined conversation named so it is one click away.
 *  3. **Otherwise** the conversation is opened. A deep link with no
 *     activity record (a bookmark, a shared link, a new tab) lands here:
 *     an explicit navigation is explicit, whatever the clock says.
 */
export function decideEntry({
  conversationId,
  lastActiveAt,
  now,
  windowMs = RESUME_WINDOW_MS,
}: EntryInput): EntryDecision {
  if (!conversationId) return { kind: "fresh", awayMs: null, resumable: null };

  if (lastActiveAt !== null) {
    const awayMs = now - lastActiveAt;
    // A clock that went backwards reads as "not away", which is the safe
    // side of this comparison: it keeps the conversation rather than
    // throwing it away on the strength of a bad timestamp.
    if (awayMs > windowMs)
      return { kind: "fresh", awayMs, resumable: conversationId };
  }

  return { kind: "resume", conversationId };
}

/** The away time as a sentence fragment: "42 minutes", "3 hours". */
export function describeAway(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return "a while";
  const minutes = Math.round(ms / 60_000);
  if (minutes < 90) return `${Math.max(1, minutes)} minutes`;
  const hours = Math.round(minutes / 60);
  if (hours < 36) return `${hours} hours`;
  return `${Math.round(hours / 24)} days`;
}

// ---------------------------------------------------------------------
// The activity record.
//
// `sessionStorage`, not `localStorage`, and that is the whole design: it
// is per tab and it dies with the tab. So a reload of a tab that has
// been away all night still reads as away, while a brand-new tab opened
// on a deep link has no record at all and the link is honoured. One
// storage choice, and the two cases that used to need special-casing
// fall out of it.
// ---------------------------------------------------------------------

const ACTIVE_AT_KEY = "ferrox.studio.tabActiveAt";

export function readTabActiveAt(): number | null {
  try {
    const raw = sessionStorage.getItem(ACTIVE_AT_KEY);
    if (raw === null) return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  } catch {
    // Private browsing, or storage disabled. No record means "honour the
    // URL", which is the same answer a fresh tab gets.
    return null;
  }
}

export function markTabActive(now = Date.now()): void {
  try {
    sessionStorage.setItem(ACTIVE_AT_KEY, String(now));
  } catch {
    /* nothing is recorded, so nothing is ever considered stale */
  }
}
