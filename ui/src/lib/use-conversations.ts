import { useCallback, useEffect, useState } from "react";
import { ApiError } from "./api.ts";
import {
  CONVERSATIONS_CHANGED,
  deleteConversation,
  listConversations,
  type ConversationSummary,
} from "./conversations.ts";

/** Re-read this often even when nothing here wrote: another client may have. */
const REFRESH_MS = 30_000;

export type ConversationList = {
  /** `"checking"` until the first answer; `"unsupported"` when the server has no store. */
  mode: "checking" | "server" | "unsupported";
  summaries: ConversationSummary[];
  /** When `summaries` was read (ms); the clock recency buckets are drawn against. */
  fetchedAt: number;
  error: string | null;
  refresh: () => void;
  remove: (id: string) => Promise<void>;
};

/**
 * The saved conversations, for a listing that lives OUTSIDE the chat
 * screen (the sidebar) and so cannot read the chat's own transcript
 * state.
 *
 * Reads on mount, whenever the store is written to from this tab
 * (`CONVERSATIONS_CHANGED`), and on a slow timer for writes from other
 * clients. A 404 on the listing means the server keeps no
 * conversations at all -- local-transcript mode -- and the section
 * hides rather than showing an empty library.
 */
export function useConversationList(): ConversationList {
  const [mode, setMode] = useState<ConversationList["mode"]>("checking");
  const [summaries, setSummaries] = useState<ConversationSummary[]>([]);
  const [fetchedAt, setFetchedAt] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listConversations()
      .then((list) => {
        setSummaries(list);
        setFetchedAt(Date.now());
        setMode("server");
        setError(null);
      })
      .catch((e: unknown) => {
        if (e instanceof ApiError && e.isMissingEndpoint) {
          setMode("unsupported");
          return;
        }
        setError((e as Error).message);
      });
  }, []);

  useEffect(() => {
    refresh();
    const onChange = () => refresh();
    window.addEventListener(CONVERSATIONS_CHANGED, onChange);
    const timer = setInterval(refresh, REFRESH_MS);
    return () => {
      window.removeEventListener(CONVERSATIONS_CHANGED, onChange);
      clearInterval(timer);
    };
  }, [refresh]);

  const remove = useCallback(
    async (id: string) => {
      try {
        await deleteConversation(id);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [],
  );

  return { mode, summaries, fetchedAt, error, refresh, remove };
}

/** Sidebar groups, in display order. */
export type RecencyBucket = "Today" | "Yesterday" | "Previous 7 days" | "Older";

/**
 * Which bucket a conversation falls in by its `updated_at` (seconds),
 * against `now` (milliseconds). Local-midnight boundaries, as every
 * chat client draws them.
 */
export function recencyBucket(updatedAtSeconds: number, now: number): RecencyBucket {
  const startOfToday = new Date(now);
  startOfToday.setHours(0, 0, 0, 0);
  const day = 24 * 60 * 60 * 1000;
  const at = updatedAtSeconds * 1000;
  if (at >= startOfToday.getTime()) return "Today";
  if (at >= startOfToday.getTime() - day) return "Yesterday";
  if (at >= startOfToday.getTime() - 7 * day) return "Previous 7 days";
  return "Older";
}
