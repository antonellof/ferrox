import { useEffect } from "react";
import {
  ACTIVITY_HEARTBEAT_MS,
  markTabActive,
  readTabActiveAt,
  RESUME_WINDOW_MS,
} from "@/lib/entry-state";
import { useLatest } from "@/lib/use-latest";

/**
 * Keep the tab's "last awake" stamp current, and say when it comes back.
 *
 * The stamp is refreshed on a heartbeat **only while the document is
 * visible**, which is what makes the gap it leaves behind mean something:
 * a hidden tab, a closed lid and a suspended machine all stop the
 * heartbeat, so the gap is time the user was away rather than time that
 * merely passed. A tab someone is sitting in front of never accumulates
 * any.
 *
 * `onReturn` fires once per return, with how long the gap was, and only
 * when the gap is longer than the window. It is a report, not a
 * decision: what to do about it belongs to the caller, which is also the
 * only thing that knows whether a generation is in flight.
 */
export function useTabActivity(
  onReturn: (awayMs: number) => void,
  windowMs: number = RESUME_WINDOW_MS,
) {
  const onReturnRef = useLatest(onReturn);

  useEffect(() => {
    const visible = () =>
      typeof document === "undefined" || document.visibilityState === "visible";

    markTabActive();

    /**
     * One tick, and both triggers run it.
     *
     * A heartbeat that only stamped, and a visibility handler that only
     * compared, would be two halves of one rule that could disagree —
     * and they did: a machine that slept with this tab in front of the
     * user never fires `visibilitychange` at all, so only the heartbeat
     * can see that gap, and only if the heartbeat is the thing that
     * measures it.
     */
    const tick = () => {
      if (!visible()) {
        // Stamp the moment of leaving, so the gap starts now rather than
        // at the last heartbeat up to `ACTIVITY_HEARTBEAT_MS` earlier.
        markTabActive();
        return;
      }
      // Read before writing: the stamp is the evidence, and overwriting
      // it first would destroy the only record of how long this was.
      const last = readTabActiveAt();
      markTabActive();
      if (last === null) return;
      const awayMs = Date.now() - last;
      if (awayMs > windowMs) onReturnRef.current(awayMs);
    };

    const beat = setInterval(tick, ACTIVITY_HEARTBEAT_MS);
    document.addEventListener("visibilitychange", tick);
    return () => {
      clearInterval(beat);
      document.removeEventListener("visibilitychange", tick);
    };
  }, [windowMs, onReturnRef]);
}
