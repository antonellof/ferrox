import * as Popover from "@radix-ui/react-popover";
import { Check, ChevronDown, Minus, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { fmtDuration } from "@/lib/format";
import type { HealthState } from "@/lib/use-health";

// The bottom of the sidebar, which is where an app of this shape puts
// "how is my connection" and nothing else. This one used to put the
// LOADED MODEL ID there, under a chevron, in the slot an account or a
// settings control normally occupies. It read as a third model picker
// beside the two real ones, it was the widest thing in the sidebar, and
// the one word that would have explained it ("Backend status") was in a
// `title` attribute nobody hovers. The model id belongs where the model
// is chosen. This belongs to the server.
//
// Three states, and the third one is the point: while the server is
// still probing backends it answers `detecting`, and this shows a
// probing pill rather than a verdict. Rendering "CPU only" from a guess
// is pixel-identical to rendering it from a measurement, and the user
// cannot tell which they were shown. The line beside the state is the
// server's VERSION, which it states, and never a backend name: `/health`
// says which backends are *available*, never which one is running, so an
// "on Metal" here would be invented.
type Visual = {
  dot: string;
  ring: string;
  /** The state, in the server's own words wherever it has any. */
  label: string;
  /** What is worth knowing beside it, or nothing. */
  detail: string | null;
};

function visual({ health, error }: HealthState): Visual {
  if (error) {
    return {
      dot: "bg-err",
      ring: "bg-err/25",
      label: error.status === 503 ? "unavailable" : "unreachable",
      detail: error.status === 503 ? "answered, not serving" : "no answer",
    };
  }
  if (!health) {
    return {
      dot: "bg-faint",
      ring: "bg-faint/25",
      label: "connecting…",
      detail: null,
    };
  }
  const version = health.version ? `v${health.version}` : null;
  switch (health.state) {
    case "ready":
      return { dot: "bg-ok", ring: "bg-ok/25", label: "ready", detail: version };
    case "detecting":
      return {
        dot: "bg-warn",
        ring: "bg-warn/30",
        label: "detecting backends…",
        detail: version,
      };
    default:
      return {
        dot: "bg-err",
        ring: "bg-err/25",
        label: health.reason || "unavailable",
        detail: version,
      };
  }
}

export function HealthPill({ state, className }: { state: HealthState; className?: string }) {
  const v = visual(state);
  const health = state.health;

  return (
    <Popover.Root>
      <Popover.Trigger
        aria-label="Server status — open for backend and capability detail"
        className={cn(
          "group flex w-full items-center gap-2.5 rounded-lg border border-line bg-raised px-2.5 py-2 text-left text-xs transition-colors hover:border-line-strong hover:bg-inset",
          className,
        )}
      >
        <span className="relative grid size-2.5 shrink-0 place-items-center">
          {state.health?.state === "detecting" ? (
            <span
              className={cn(
                "absolute inset-0 animate-ping rounded-full",
                v.ring,
              )}
            />
          ) : null}
          <span className={cn("size-2 rounded-full", v.dot)} />
        </span>
        {/* The word first. A status control you have to click to find out
            is a status control is the thing being fixed here. */}
        <span className="min-w-0 flex-1">
          <span className="block text-2xs leading-tight text-faint">
            Server
          </span>
          <span
            className="block truncate leading-tight font-medium"
            aria-live="polite"
          >
            {v.label}
            {v.detail ? (
              <span className="font-normal text-faint"> · {v.detail}</span>
            ) : null}
          </span>
        </span>
        <ChevronDown className="size-3.5 shrink-0 text-faint transition-transform group-data-[state=open]:rotate-180" />
      </Popover.Trigger>

      <Popover.Portal>
        <Popover.Content
          side="top"
          align="start"
          sideOffset={8}
          collisionPadding={12}
          className="z-50 w-[min(26rem,calc(100vw-1.5rem))] rounded-xl border border-line bg-raised p-3 text-xs shadow-pop data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
        >
          {health ? (
            <div className="space-y-3">
              <div className="space-y-1">
                <p className="font-medium text-fg">
                  <code className="font-mono">/health</code> · {health.state}
                </p>
                {health.detail ? (
                  <p className="text-faint">{health.detail}</p>
                ) : null}
                {health.model ? (
                  <p className="text-muted">
                    model{" "}
                    <span className="font-mono">{health.model.id}</span>
                    {health.model.tokenizer
                      ? ` · tokenizer ${health.model.tokenizer}`
                      : ""}
                    {health.model.synthetic_weights ? (
                      <span className="text-err">
                        {" "}
                        — SYNTHETIC random weights, output is noise
                      </span>
                    ) : null}
                  </p>
                ) : null}
              </div>

              <ul className="space-y-1.5">
                {(health.capabilities ?? []).map((cap) => (
                  <li key={cap.id} className="flex items-start gap-2">
                    {/* The server pairs every flag with a machine reason *and*
                        a human sentence precisely so the UI never re-derives
                        the explanation. A greyed control without its reason is
                        the failure this replaces. */}
                    {cap.available ? (
                      <Check className="mt-px size-3.5 shrink-0 text-ok" />
                    ) : cap.reason === "detecting" ? (
                      <Minus className="mt-px size-3.5 shrink-0 text-warn" />
                    ) : (
                      <X className="mt-px size-3.5 shrink-0 text-faint" />
                    )}
                    <span className="min-w-0">
                      <span className="font-mono font-medium text-fg">
                        {cap.id}
                      </span>
                      <span className="text-faint">
                        {" — "}
                        {cap.available ? "available" : cap.reason}
                      </span>
                      <span className="block text-faint">{cap.detail}</span>
                    </span>
                  </li>
                ))}
              </ul>

              <p className="border-t border-line pt-2 text-faint">
                {health.version ? `version ${health.version}` : null}
                {health.pid ? ` · pid ${health.pid}` : null}
                {typeof health.last_request_age_seconds === "number"
                  ? ` · last request ${fmtDuration(health.last_request_age_seconds)} ago`
                  : null}
              </p>
            </div>
          ) : (
            <p className="text-muted">
              The server did not answer <code className="font-mono">/health</code>
              {state.error ? `: ${state.error.message}` : "."}
            </p>
          )}
          <Popover.Arrow className="fill-[var(--bg-raised)]" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
