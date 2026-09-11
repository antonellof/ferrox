// How long a reasoning model thought, as the stream saw it.
//
// This is the ONE client-side stopwatch in the UI, and it is here on
// purpose. Every other number under an answer is the server's `usage`,
// because a browser cannot tell prefill from decode. But "how long did
// the model think" is not a decode figure at all: the server's usage
// counts tokens and their rate, and a thought's length in seconds is
// the wall-clock between the first `reasoning_content` delta and the
// first `content` delta -- which only the consumer of the stream can
// see. That measurement is taken in `screens/chat/runtime.ts`, carried
// on the message as `metadata.custom.thought`, shown by
// `screens/chat/reasoning.tsx`, and stored beside `reasoning_content`
// as `reasoning_ms` by `lib/conversations.ts`. The shape is spelled
// here so those four agree by import rather than by convention.

/** Under `metadata.custom`, on an assistant message. */
export const THOUGHT_KEY = "thought";

export type Thought =
  /**
   * Still thinking. `startedAt` is an epoch-millisecond stamp so the
   * elapsed time can tick locally between deltas -- a stalled stream
   * must not show a frozen clock. A continuation resumed inside a
   * thought sets it BACK by the earlier portion, so the running total
   * counts both halves without a second field.
   */
  | { state: "thinking"; startedAt: number }
  /** The thought ended, either because the answer began or because
   * the stream did (cut off, stopped) with no answer after it. */
  | { state: "done"; ms: number };

/** Milliseconds thought so far, at `now`. */
export function thoughtMs(thought: Thought, now: number): number {
  return thought.state === "done"
    ? thought.ms
    : Math.max(0, now - thought.startedAt);
}

/**
 * The `Thought` carried on a message's metadata, or `undefined` when
 * there is none or it is not the shape this module writes.
 *
 * Validated field by field because metadata comes back from the store
 * and from `localStorage`, neither of which the type system reaches.
 */
export function readThought(
  metadata: Record<string, unknown> | null | undefined,
): Thought | undefined {
  const custom = metadata?.custom;
  if (!custom || typeof custom !== "object") return undefined;
  return parseThought((custom as Record<string, unknown>)[THOUGHT_KEY]);
}

/** The slot's value alone, for a reader that already has it in hand. */
export function parseThought(value: unknown): Thought | undefined {
  if (!value || typeof value !== "object") return undefined;
  const { state, startedAt, ms } = value as Record<string, unknown>;
  if (state === "thinking" && isFiniteNumber(startedAt))
    return { state: "thinking", startedAt };
  if (state === "done" && isFiniteNumber(ms) && ms >= 0)
    return { state: "done", ms };
  return undefined;
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

/**
 * A thought's length in words: "15 seconds" under a minute, "1 min
 * 20 s" above it, "1 h 5 min" above an hour. Whole seconds: a tenth
 * of a second is not a fact about a thought, and a live clock that
 * ticks in tenths is noise.
 */
export function fmtThought(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 1) return "under a second";
  if (s < 60) return s === 1 ? "1 second" : `${s} seconds`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const rest = s % 60;
    return rest ? `${m} min ${rest} s` : `${m} min`;
  }
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return m ? `${h} h ${m} min` : `${h} h`;
}
