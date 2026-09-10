import type { FC } from "react";
import {
  ActionBarPrimitive,
  BranchPickerPrimitive,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAuiState,
} from "@assistant-ui/react";
import {
  ArrowDown,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Copy,
  CornerDownLeft,
  Pencil,
  RefreshCw,
  Square,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { MarkdownText } from "@/screens/chat/markdown";
import { ReasoningPart } from "@/screens/chat/reasoning";
import type { AnswerStats } from "@/screens/chat/runtime";

// The transcript, the composer, autoscroll, branching and the abort
// signal are assistant-ui's. What is written here is presentation plus
// the one thing that is genuinely ferrox's: the stat line under an
// answer, which is the server's `usage` and nothing else.
//
// Note what is NOT used: `useMessageTiming()`. assistant-ui measures its
// own stream client-side and would happily hand over a `tokensPerSecond`.
// That number cannot separate prefill from decode and would read a
// 50 tok/s model as 5 on a long prompt. There is no client stopwatch in
// this UI, by construction.

function useStats(): AnswerStats | undefined {
  return useAuiState(
    (s) =>
      (s.message.metadata as { custom?: { stats?: AnswerStats } } | undefined)
        ?.custom?.stats,
  );
}

const OUTCOME_LABEL: Record<AnswerStats["outcome"], string | null> = {
  ok: null,
  "stopped-by-you": "stopped by you",
  "stopped-by-server": "stopped",
  error: "failed",
};

function StatLine() {
  const stats = useStats();
  if (!stats) return null;
  const outcome = OUTCOME_LABEL[stats.outcome];
  const pieces = [outcome, stats.line].filter(Boolean);
  if (!pieces.length) {
    return stats.requestId ? (
      <p className="mt-2 font-mono text-2xs text-faint">{stats.requestId}</p>
    ) : null;
  }
  return (
    <p
      className="mt-2 font-mono text-2xs leading-relaxed text-faint"
      title="Reported by the server in the final SSE chunk's usage block. The browser holds no stopwatch."
    >
      {pieces.join("  ·  ")}
    </p>
  );
}

function BranchPicker({ className }: { className?: string }) {
  return (
    <BranchPickerPrimitive.Root
      hideWhenSingleBranch
      className={cn("flex items-center gap-0.5 text-faint", className)}
    >
      <BranchPickerPrimitive.Previous asChild>
        <Button variant="ghost" size="iconSm" aria-label="Previous version">
          <ChevronLeft />
        </Button>
      </BranchPickerPrimitive.Previous>
      <span className="font-mono text-2xs">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild>
        <Button variant="ghost" size="iconSm" aria-label="Next version">
          <ChevronRight />
        </Button>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
}

// The user's turn is a NEUTRAL bubble. It used to be a solid accent
// block, which made the loudest thing in a transcript the text the user
// had just typed and already knew — and on a long conversation it turned
// the column into a stripe of orange. The references (Jan, Ollama) all
// land on a quiet raised bubble; ferrox's accent is better spent on the
// send button, which is the only thing in the composer worth pointing at.
const UserMessage: FC = () => (
  <MessagePrimitive.Root className="group flex w-full flex-col items-end gap-1">
    <div className="max-w-[min(44rem,88%)] rounded-xl rounded-br-sm bg-inset px-3.5 py-2 text-fg">
      <MessagePrimitive.Parts />
    </div>
    <div className="flex items-center gap-0.5 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
      <BranchPicker />
      <ActionBarPrimitive.Root>
        <ActionBarPrimitive.Edit asChild>
          <Button variant="ghost" size="iconSm" aria-label="Edit and resend">
            <Pencil />
          </Button>
        </ActionBarPrimitive.Edit>
      </ActionBarPrimitive.Root>
    </div>
  </MessagePrimitive.Root>
);

const EditComposer: FC = () => (
  <ComposerPrimitive.Root className="ml-auto w-full max-w-[min(44rem,88%)] rounded-xl border border-line bg-raised p-2">
    <ComposerPrimitive.Input
      autoFocus
      className="w-full resize-none bg-transparent px-1.5 py-1 text-sm outline-none"
    />
    <div className="mt-1.5 flex justify-end gap-2">
      <ComposerPrimitive.Cancel asChild>
        <Button variant="ghost" size="sm">
          Cancel
        </Button>
      </ComposerPrimitive.Cancel>
      <ComposerPrimitive.Send asChild>
        <Button variant="primary" size="sm">
          Send
        </Button>
      </ComposerPrimitive.Send>
    </div>
  </ComposerPrimitive.Root>
);

const AssistantMessage: FC = () => (
  <MessagePrimitive.Root className="group flex w-full flex-col gap-1">
    <div className="flex gap-3">
      <span
        aria-hidden
        className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-md border border-line bg-raised text-2xs font-semibold text-accent"
      >
        Fe
      </span>
      <div className="min-w-0 flex-1">
        <div className="min-w-0">
          <MessagePrimitive.Parts
            components={{ Text: MarkdownText, Reasoning: ReasoningPart }}
          />
        </div>

        <MessagePrimitive.Error>
          <div className="mt-2 flex items-start gap-2 rounded-md border border-err/30 bg-err-soft px-3 py-2 text-sm text-err">
            <CircleAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
            <ErrorPrimitive.Message className="min-w-0 whitespace-pre-wrap" />
          </div>
        </MessagePrimitive.Error>

        <StatLine />

        <div className="mt-1 flex items-center gap-0.5 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
          <ActionBarPrimitive.Root
            hideWhenRunning
            autohide="not-last"
            className="flex items-center gap-0.5"
          >
            <ActionBarPrimitive.Copy asChild>
              <Button variant="ghost" size="iconSm" aria-label="Copy answer">
                <Copy />
              </Button>
            </ActionBarPrimitive.Copy>
            <ActionBarPrimitive.Reload asChild>
              <Button variant="ghost" size="iconSm" aria-label="Regenerate">
                <RefreshCw />
              </Button>
            </ActionBarPrimitive.Reload>
          </ActionBarPrimitive.Root>
          <BranchPicker />
        </div>
      </div>
    </div>
  </MessagePrimitive.Root>
);

const STARTERS = [
  "Write a haiku about a quantized tensor.",
  "Explain KV caching in three sentences.",
  "Give me a Python snippet that streams from this server.",
];

function Empty({ disabledReason }: { disabledReason: string | null }) {
  return (
    <ThreadPrimitive.Empty>
      <div className="flex flex-col items-center gap-5 px-4 py-14 text-center">
        <span
          aria-hidden
          className="grid size-11 place-items-center rounded-xl bg-accent text-base font-semibold text-accent-fg"
        >
          Fe
        </span>
        <div className="space-y-1.5">
          <p className="text-lg font-semibold tracking-tight">
            Talk to your local model
          </p>
          <p className="mx-auto max-w-md text-xs leading-relaxed text-muted">
            This screen posts to{" "}
            <code className="font-mono">/v1/chat/completions</code> with{" "}
            <code className="font-mono">stream: true</code> — the same endpoint
            any other client uses. Timings under each answer are the server's
            own <code className="font-mono">usage</code>.
          </p>
        </div>
        {disabledReason ? null : (
          <div className="flex flex-wrap justify-center gap-2">
            {STARTERS.map((prompt) => (
              // Fills the composer rather than sending: a starter that
              // fires a generation on one click can spend a minute of a
              // CPU decode before the user has read what it asked.
              <ThreadPrimitive.Suggestion
                key={prompt}
                prompt={prompt}
                method="replace"
                asChild
              >
                <Button variant="default" size="sm" className="max-w-xs">
                  <span className="truncate">{prompt}</span>
                </Button>
              </ThreadPrimitive.Suggestion>
            ))}
          </div>
        )}
      </div>
    </ThreadPrimitive.Empty>
  );
}

function Composer({ disabledReason }: { disabledReason: string | null }) {
  const isRunning = useAuiState((s) => s.thread.isRunning);

  return (
    <ComposerPrimitive.Root
      className={cn(
        "flex w-full items-end gap-2 rounded-xl border border-line bg-raised p-2 transition-colors focus-within:border-accent",
        disabledReason && "opacity-70",
      )}
    >
      <ComposerPrimitive.Input
        rows={1}
        autoFocus
        disabled={!!disabledReason}
        placeholder={
          disabledReason ??
          "Message…  (Enter to send, Shift+Enter for a newline)"
        }
        className="max-h-56 min-h-8 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm leading-relaxed outline-none placeholder:text-faint disabled:cursor-not-allowed"
      />
      {isRunning ? (
        // Stop is both tiers: assistant-ui aborts the fetch, and the
        // adapter POSTs /v1/cancel with the request_id the server named
        // on the first chunk. It cancels on the server, not just here.
        <ComposerPrimitive.Cancel asChild>
          <Button variant="default" size="icon" aria-label="Stop generating">
            <Square className="fill-current" />
          </Button>
        </ComposerPrimitive.Cancel>
      ) : (
        <ComposerPrimitive.Send asChild>
          <Button
            variant="primary"
            size="icon"
            aria-label="Send"
            disabled={!!disabledReason}
          >
            <CornerDownLeft />
          </Button>
        </ComposerPrimitive.Send>
      )}
    </ComposerPrimitive.Root>
  );
}

export function Thread({
  disabledReason,
  footer,
}: {
  /** Why sending is blocked right now, or null. Always says why. */
  disabledReason: string | null;
  footer?: React.ReactNode;
}) {
  return (
    <ThreadPrimitive.Root className="flex h-full min-h-0 flex-col bg-bg">
      <ThreadPrimitive.Viewport
        autoScroll
        className="relative flex min-h-0 flex-1 flex-col overflow-y-auto"
      >
        <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-4 py-6 md:px-6">
          <Empty disabledReason={disabledReason} />
          <ThreadPrimitive.Messages
            components={{
              UserMessage,
              EditComposer,
              AssistantMessage,
            }}
          />
          <div className="min-h-4 flex-1" />
        </div>

        <ThreadPrimitive.ViewportFooter className="sticky bottom-0 z-10 mt-auto w-full bg-linear-to-t from-bg via-bg to-transparent pt-4">
          <div className="mx-auto w-full max-w-3xl px-4 pb-4 md:px-6">
            <div className="relative">
              <ThreadPrimitive.ScrollToBottom asChild>
                <Button
                  variant="default"
                  size="iconSm"
                  aria-label="Scroll to latest"
                  className="absolute -top-9 left-1/2 -translate-x-1/2 rounded-full shadow-pop disabled:invisible"
                >
                  <ArrowDown />
                </Button>
              </ThreadPrimitive.ScrollToBottom>
            </div>
            <Composer disabledReason={disabledReason} />
            {footer ? <div className="mt-2">{footer}</div> : null}
          </div>
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
}
