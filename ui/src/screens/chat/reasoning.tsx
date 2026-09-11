import { useEffect, useRef, useState } from "react";
import {
  useAuiState,
  type ReasoningMessagePartComponent,
} from "@assistant-ui/react";
import { ChevronRight } from "lucide-react";
import {
  fmtThought,
  parseThought,
  THOUGHT_KEY,
  thoughtMs,
  type Thought,
} from "@/lib/thought";
import { cn } from "@/lib/utils";

/**
 * A reasoning model's thinking, shown above the answer it led to.
 *
 * assistant-ui renders NOTHING for a reasoning part unless a component
 * is supplied for it, which is how this went unnoticed: the server
 * streamed `reasoning_content` correctly, the client dropped it, and an
 * R1 distill therefore spent most of its wall-clock producing tokens
 * that never reached the screen. The transcript looked frozen and the
 * answer arrived whole, which reads as "streaming is broken" rather
 * than "thinking is hidden".
 *
 * Three states, told apart by the clock the runtime carries on the
 * message (`lib/thought.ts`), not by the part's own status:
 *
 * - THINKING: open, the text streaming in and pinned to its last line,
 *   the header a live "Thinking for 12 seconds". A model that thinks
 *   for a minute on CPU is not frozen, and the clock is what says so.
 * - DONE, with an answer under it: collapsed to "Thought for 15
 *   seconds". The thought is working-out, secondary once the answer
 *   exists, and one click away.
 * - DONE, with NO answer: still open. This is the cut-off case from
 *   the Continue banner -- the budget ran out inside the thought, so
 *   the thought is all there is, and folding it would leave a banner
 *   over nothing.
 *
 * A record stored before the clock existed has a thought and no time;
 * it reads "Thought" with the length in characters, as it did.
 *
 * It is deliberately NOT markdown: thinking is where a model emits
 * half-open code fences and unbalanced brackets, and a renderer that
 * reflows them makes the text harder to read, not easier.
 */
export const ReasoningPart: ReasoningMessagePartComponent = ({ text }) => {
  // The raw slot is selected and decoded outside the selector: a
  // selector that returned a fresh object would re-render on every
  // store update, and `useSyncExternalStore` would then loop on it.
  const thought = parseThought(
    useAuiState((s) => s.message.metadata.custom?.[THOUGHT_KEY]),
  );
  const running = useAuiState((s) => s.message.status?.type === "running");
  const answered = useAuiState((s) =>
    s.message.content.some(
      (part) => part.type === "text" && part.text.trim().length > 0,
    ),
  );
  const thinking = thought?.state === "thinking" && running;
  // Until the user has clicked, the block follows the state above; one
  // click and it is theirs. `null` is "not clicked yet", so a thought
  // the user opened after it finished does not snap shut on the next
  // render, and one they closed while it streamed stays closed.
  const [chosen, setChosen] = useState<boolean | null>(null);
  const open = chosen ?? (thinking || !answered);

  const trimmed = text.trim();
  // A part that exists but has not been written to yet would otherwise
  // render an empty, clickable box under every answer.
  if (!trimmed) return null;
  return (
    <div className="mb-2 rounded-lg border border-line bg-inset">
      <button
        type="button"
        onClick={() => setChosen(!open)}
        aria-expanded={open}
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-xs text-muted transition-colors hover:text-fg"
      >
        <ChevronRight
          aria-hidden
          className={cn(
            "size-3.5 shrink-0 transition-transform",
            open && "rotate-90",
          )}
        />
        <Summary thought={thought} thinking={thinking} chars={trimmed.length} />
      </button>
      {open && <Body text={trimmed} thinking={thinking} />}
    </div>
  );
};

/**
 * The header line. A live clock while thinking, a total once done.
 *
 * The interval runs ONLY while thinking: a finished thought is a fixed
 * number and a transcript of fifty of them must not hold fifty timers.
 */
function Summary({
  thought,
  thinking,
  chars,
}: {
  thought: Thought | undefined;
  thinking: boolean;
  chars: number;
}) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!thinking) return;
    // Half-second ticks so the displayed second is never more than
    // half a second stale, which reads as a clock rather than a lag.
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [thinking]);

  if (!thought) {
    return (
      <>
        <span>{thinking ? "Thinking" : "Thought"}</span>
        <span className="text-muted/70">{chars.toLocaleString()} chars</span>
      </>
    );
  }
  const elapsed = fmtThought(thoughtMs(thought, now));
  return thinking ? (
    <>
      {/* The pulse is the "still alive" signal: a clock alone could be
          a stopped one that happens to show the right number. */}
      <span className="relative grid size-2.5 shrink-0 place-items-center">
        <span className="absolute inset-0 animate-ping rounded-full bg-muted/40" />
        <span className="size-1.5 rounded-full bg-muted" />
      </span>
      <span>
        Thinking for <span className="tabular-nums">{elapsed}</span>
      </span>
    </>
  ) : (
    <span>
      Thought for <span className="tabular-nums">{elapsed}</span>
    </span>
  );
}

/**
 * The thought itself. While streaming it follows its own last line,
 * the way the transcript follows the answer -- but only while the
 * reader is at the bottom. Scrolling up to reread something is a
 * choice, and a box that yanks back on every token overrides it.
 */
function Body({ text, thinking }: { text: string; thinking: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  useEffect(() => {
    const el = box.current;
    if (!el || !thinking || !pinned.current) return;
    el.scrollTop = el.scrollHeight;
  }, [text, thinking]);
  return (
    <div
      ref={box}
      onScroll={(e) => {
        const el = e.currentTarget;
        pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
      className="max-h-80 overflow-y-auto whitespace-pre-wrap border-t border-line px-2.5 py-2 text-xs leading-relaxed text-muted"
    >
      {text}
    </div>
  );
}
