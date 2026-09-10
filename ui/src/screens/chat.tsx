import { useCallback, useEffect, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import * as Popover from "@radix-ui/react-popover";
import { Link, useNavigate, useOutletContext, useParams } from "react-router";
import {
  Check,
  ChevronDown,
  History,
  Loader2,
  MessagesSquare,
  SlidersHorizontal,
  SquarePen,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Field, Input, Textarea } from "@/components/ui/field";
import { Notice } from "@/components/ui/feedback";
import { Badge } from "@/components/ui/badge";
import {
  ApiError,
  getJson,
  postJson,
  routes,
  type Inventory,
} from "@/lib/api";
import type { HealthState } from "@/lib/use-health";
import { fmtBytes } from "@/lib/format";
import { useLatest } from "@/lib/use-latest";
import { cn } from "@/lib/utils";
import { Thread } from "@/screens/chat/thread";
import {
  DEFAULT_SAMPLING,
  useFerroxRuntime,
  type Sampling,
} from "@/screens/chat/runtime";
import {
  useTranscript,
  type Transcript,
} from "@/screens/chat/persistence";
import { conversationLabel } from "@/lib/conversations";
import { describeAway, RESUME_WINDOW_MS } from "@/lib/entry-state";
import { useTabActivity } from "@/lib/use-tab-activity";

const SETTINGS_KEY = "ferrox.studio.sampling.v1";

function loadSampling(): Sampling {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw
      ? { ...DEFAULT_SAMPLING, ...JSON.parse(raw) }
      : { ...DEFAULT_SAMPLING };
  } catch {
    return { ...DEFAULT_SAMPLING };
  }
}

function saveSampling(value: Sampling) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(value));
  } catch {
    /* quota or private browsing — the in-memory settings still apply */
  }
}

type Loaded = {
  modelId: string | null;
  synthetic: boolean;
  error: string | null;
};

/**
 * What `/v1/models` says is serving right now, kept current.
 *
 * Re-read whenever `/health` reports a different model, which the shell
 * already polls every five seconds — so no second poller is added, and
 * a model loaded from the Models screen (or by anything else talking to
 * this server) un-gates the composer on its own. Reading it once at
 * mount was survivable when the id was only a label; now that an empty
 * one blocks Send, a stale "no model loaded" would strand the user on a
 * server that is ready.
 */
function useServingModel(healthModelId: string | null): [Loaded, () => void] {
  const [state, setState] = useState<Loaded>({
    modelId: null,
    synthetic: false,
    error: null,
  });
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getJson<{ data?: { id: string; ferrox_synthetic_weights?: boolean }[] }>(
      routes.models,
    )
      .then((body) => {
        if (cancelled) return;
        const first = body?.data?.[0];
        setState({
          modelId: first?.id ?? null,
          synthetic: !!first?.ferrox_synthetic_weights,
          error: null,
        });
      })
      .catch((error: Error) => {
        if (cancelled) return;
        setState({ modelId: null, synthetic: false, error: error.message });
      });
    return () => {
      cancelled = true;
    };
  }, [nonce, healthModelId]);

  return [state, useCallback(() => setNonce((n) => n + 1), [])];
}

/**
 * Switch the served model without leaving the conversation.
 *
 * A swap is a real server-side action (`POST /admin/models/load`), not a
 * per-request parameter — this server serves one checkpoint at a time.
 * The menu says so rather than implying the next message could pick a
 * different model on its own.
 */
function ModelSwitcher({
  active,
  onSwitched,
}: {
  active: string | null;
  onSwitched: () => void;
}) {
  const [inventory, setInventory] = useState<Inventory | null>(null);
  const [unsupported, setUnsupported] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    getJson<Inventory>(routes.adminModels)
      .then(setInventory)
      .catch((e) => {
        if (e instanceof ApiError && e.isMissingEndpoint) setUnsupported(true);
      });
  }, []);

  const swap = async (id: string) => {
    setBusy(id);
    setError(null);
    try {
      await postJson(routes.adminModelsLoad, { id });
      onSwitched();
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Popover.Root onOpenChange={(open) => open && refresh()}>
      <Popover.Trigger asChild>
        <Button variant="default" size="sm" className="max-w-[16rem]">
          <span className="truncate font-mono text-[0.6875rem]">
            {active ?? "no model loaded"}
          </span>
          <ChevronDown className="text-faint" />
        </Button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={6}
          collisionPadding={12}
          className="z-50 w-[min(24rem,calc(100vw-1.5rem))] rounded-card border border-line bg-raised p-1.5 shadow-pop data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
        >
          {unsupported ? (
            <p className="p-2 text-xs text-faint">
              This build has no <code className="font-mono">/admin</code>{" "}
              control surface, so the model cannot be swapped from here.
            </p>
          ) : !inventory ? (
            <p className="p-2 text-xs text-faint">Reading the inventory…</p>
          ) : !inventory.models.length ? (
            <p className="p-2 text-xs text-faint">
              No checkpoints found.{" "}
              <Link to="/ui/models" className="text-accent underline">
                Download one
              </Link>
              .
            </p>
          ) : (
            <ul className="max-h-72 space-y-0.5 overflow-y-auto">
              {inventory.models.map((entry) => {
                const isActive = entry.id === inventory.active;
                return (
                  <li key={entry.id}>
                    <button
                      type="button"
                      disabled={isActive || !!busy}
                      onClick={() => swap(entry.id)}
                      className={cn(
                        "flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors",
                        isActive
                          ? "bg-accent-soft text-accent"
                          : "hover:bg-inset disabled:opacity-50",
                      )}
                    >
                      {busy === entry.id ? (
                        <Loader2 className="size-3.5 shrink-0 animate-spin" />
                      ) : isActive ? (
                        <Check className="size-3.5 shrink-0" />
                      ) : (
                        <span className="size-3.5 shrink-0" />
                      )}
                      <span className="min-w-0 flex-1">
                        <span className="block truncate font-mono text-xs">
                          {entry.id}
                        </span>
                        <span className="block truncate text-[0.6875rem] text-faint">
                          {[entry.quant, entry.arch, fmtBytes(entry.size_bytes)]
                            .filter(Boolean)
                            .join(" · ")}
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
          {error ? (
            <p className="mt-1 rounded-lg bg-err-soft px-2 py-1.5 text-[0.6875rem] text-err">
              {error}
            </p>
          ) : null}
          <p className="mt-1 border-t border-line px-2 pt-1.5 text-[0.6875rem] text-faint">
            Loading a checkpoint swaps it for every client of this server. A
            request already in flight finishes on the weights it started on.
          </p>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function SamplingPanel({
  value,
  onChange,
}: {
  value: Sampling;
  onChange: (next: Sampling) => void;
}) {
  const set = <K extends keyof Sampling>(key: K, next: Sampling[K]) =>
    onChange({ ...value, [key]: next });

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <Button variant="default" size="sm">
          <SlidersHorizontal />
          <span className="hidden sm:inline">Sampling</span>
        </Button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={6}
          collisionPadding={12}
          className="z-50 w-[min(24rem,calc(100vw-1.5rem))] space-y-3 rounded-card border border-line bg-raised p-3 shadow-pop data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
        >
          <div className="grid grid-cols-3 gap-2">
            <Field label="temperature">
              <Input
                type="number"
                min="0"
                max="2"
                step="0.05"
                value={value.temperature}
                onChange={(e) => set("temperature", Number(e.target.value))}
              />
            </Field>
            <Field label="top_p">
              <Input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={value.topP}
                onChange={(e) => set("topP", Number(e.target.value))}
              />
            </Field>
            <Field label="max_tokens">
              <Input
                type="number"
                min="1"
                max="32768"
                step="1"
                value={value.maxTokens}
                onChange={(e) =>
                  set(
                    "maxTokens",
                    Math.max(1, Math.round(Number(e.target.value) || 1)),
                  )
                }
              />
            </Field>
          </div>
          <Field
            label="system prompt"
            hint="Sent as the first message of every request, not stored on the server."
          >
            <Textarea
              rows={3}
              placeholder="You are a helpful assistant."
              value={value.system}
              onChange={(e) => set("system", e.target.value)}
            />
          </Field>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

/**
 * The saved conversations, and the switch between them.
 *
 * Only rendered when the server actually keeps conversations. In local
 * mode there is exactly one transcript and a list of it would be a
 * menu with one entry pretending to be a library.
 */
function ConversationPicker({ transcript }: { transcript: Transcript }) {
  const { summaries, current } = transcript;

  return (
    <Popover.Root
      onOpenChange={(open) => {
        if (open) transcript.refresh();
      }}
    >
      <Popover.Trigger asChild>
        <Button variant="default" size="sm" className="max-w-[14rem]">
          <MessagesSquare className="text-faint" />
          <span className="truncate text-[0.6875rem]">
            {current ? conversationLabel(current) : "New conversation"}
          </span>
          <ChevronDown className="text-faint" />
        </Button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={6}
          collisionPadding={12}
          className="z-50 w-[min(26rem,calc(100vw-1.5rem))] rounded-card border border-line bg-raised p-1.5 shadow-pop data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
        >
          {!summaries.length ? (
            <p className="p-2 text-xs text-faint">
              Nothing saved yet. A conversation is created on this server the
              first time you send a message.
            </p>
          ) : (
            <ul className="max-h-80 space-y-0.5 overflow-y-auto">
              {summaries.map((entry) => {
                const isActive = entry.id === current?.id;
                return (
                  <li key={entry.id} className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => transcript.open(entry.id)}
                      className={cn(
                        "min-w-0 flex-1 rounded-lg px-2 py-1.5 text-left transition-colors",
                        isActive
                          ? "bg-accent-soft text-accent"
                          : "hover:bg-inset",
                      )}
                    >
                      <span className="block truncate text-xs">
                        {conversationLabel(entry)}
                      </span>
                      <span className="block truncate text-[0.6875rem] text-faint">
                        {[
                          `${entry.message_count} message${entry.message_count === 1 ? "" : "s"}`,
                          entry.model,
                        ]
                          .filter(Boolean)
                          .join(" · ")}
                      </span>
                    </button>
                    <Button
                      variant="ghost"
                      size="sm"
                      title="Delete this conversation from the server"
                      onClick={() => transcript.remove(entry.id)}
                    >
                      <Trash2 />
                    </Button>
                  </li>
                );
              })}
            </ul>
          )}
          <p className="mt-1 border-t border-line px-2 pt-1.5 text-[0.6875rem] text-faint">
            Stored on the server, not in this browser. Deleting one deletes it
            for every client of this server, and nothing is ever deleted to
            make room.
          </p>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function ChatInner({
  sampling,
  setSampling,
  serving,
  refreshServing,
  stall,
  transport,
  transcript,
}: {
  sampling: Sampling;
  setSampling: (next: Sampling) => void;
  serving: Loaded;
  refreshServing: () => void;
  stall: string | null;
  transport: string | null;
  transcript: Transcript;
}) {
  const disabledReason = serving.modelId
    ? null
    : serving.error
      ? `Could not read ${routes.models}: ${serving.error}`
      : "No model is loaded — pick one from the model menu above before sending.";

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex shrink-0 flex-wrap items-center gap-2 border-b border-line bg-raised/70 px-4 py-2.5 backdrop-blur">
        <h1 className="text-sm font-semibold tracking-tight">Chat</h1>
        {serving.synthetic ? (
          <Badge tone="err">synthetic weights</Badge>
        ) : null}
        {transcript.saving ? (
          <span className="text-[0.6875rem] text-faint">saving…</span>
        ) : null}
        <span className="flex-1" />
        {transcript.mode === "server" ? (
          <ConversationPicker transcript={transcript} />
        ) : null}
        <ModelSwitcher active={serving.modelId} onSwitched={refreshServing} />
        <SamplingPanel value={sampling} onChange={setSampling} />
        <Button variant="ghost" size="sm" onClick={transcript.newChat}>
          <SquarePen />
          <span className="hidden sm:inline">New chat</span>
        </Button>
      </header>

      {stall ||
      transport ||
      serving.synthetic ||
      disabledReason ||
      transcript.stale ||
      transcript.error ? (
        <div className="shrink-0 space-y-2 border-b border-line bg-raised/40 px-4 py-2.5">
          {transcript.stale ? (
            <Notice>
              <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                <span>
                  You were away for {describeAway(transcript.stale.awayMs)}, so
                  this is a new chat. Anything idle for over{" "}
                  {Math.round(RESUME_WINDOW_MS / 60_000)} minutes is not picked
                  back up on its own.
                </span>
                <Button
                  variant="default"
                  size="sm"
                  onClick={transcript.resumeStale}
                >
                  <History />
                  <span className="max-w-[14rem] truncate">
                    Reopen {transcript.stale.label}
                  </span>
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={transcript.dismissStale}
                >
                  Dismiss
                </Button>
              </span>
            </Notice>
          ) : null}
          {transcript.error ? (
            <Notice tone="err">
              This conversation is not being saved: {transcript.error} The
              messages on screen are unaffected, and the next change retries.
            </Notice>
          ) : null}
          {disabledReason ? (
            <Notice tone="warn">
              {disabledReason}{" "}
              <Link to="/ui/models" className="underline underline-offset-2">
                Or download one
              </Link>
            </Notice>
          ) : null}
          {serving.synthetic ? (
            <Notice tone="err">
              <code className="font-mono">{serving.modelId}</code> is running on
              synthetic random weights — the output is noise, not a bad model.
            </Notice>
          ) : null}
          {stall ? <Notice tone="warn">{stall}</Notice> : null}
          {transport ? <Notice>{transport}</Notice> : null}
        </div>
      ) : null}

      <div className="min-h-0 flex-1">
        <Thread
          disabledReason={disabledReason}
          footer={
            <p className="text-center text-[0.6875rem] text-faint">
              {transcript.mode === "server"
                ? "Transcript is stored on the server, branches included, and survives this browser."
                : transcript.mode === "local"
                  ? (transcript.reason ??
                    "Transcript is stored in this browser only.")
                  : "Checking where this transcript will be stored…"}
            </p>
          }
        />
      </div>
    </div>
  );
}

export function ChatScreen() {
  const health = useOutletContext<HealthState>();
  const navigate = useNavigate();
  // `/ui/chat` is a new chat and `/ui/chat/<id>` is that conversation.
  // The URL is the only place that says which one is open, so the
  // address bar cannot disagree with the screen — see `lib/entry-state`.
  const { conversationId } = useParams();
  const [sampling, setSamplingState] = useState<Sampling>(loadSampling);
  const [serving, refreshServing] = useServingModel(
    health?.health?.model?.id ?? null,
  );
  const [stall, setStall] = useState<string | null>(null);
  const [transport, setTransport] = useState<string | null>(null);

  const samplingRef = useLatest(sampling);
  const servingRef = useLatest(serving);

  const setSampling = useCallback((next: Sampling) => {
    setSamplingState(next);
    saveSampling(next);
  }, []);

  const runtime = useFerroxRuntime({
    modelId: () => servingRef.current.modelId,
    sampling: () => samplingRef.current,
    // A stream that has gone quiet is not the same as a slow model — the
    // server sends a keep-alive comment every 15 s, so silence on the
    // wire means the connection, not the decode. Said out loud rather
    // than left as a spinner that never resolves; `null` means it
    // recovered, and the banner comes down.
    onStall: (ms) =>
      setStall(
        ms === null
          ? null
          : `No data for ${Math.round(ms / 1000)}s. The generation may still be running — ` +
              "a proxy between you and the server may be buffering text/event-stream. " +
              "Stop cancels it on the server, not just here.",
      ),
    // Recovery is visible rather than silent. The answer is the same
    // one either way — the server kept generating into its replay
    // buffer — but "this is a reconnect" and "this is still the
    // original connection" are different facts about what just
    // happened, and only one of them explains a pause.
    onTransport: (mode) =>
      setTransport(
        mode === "stream"
          ? null
          : mode === "resumed"
            ? "The connection dropped. Reconnected and resumed from the last event received — no tokens repeated, none lost."
            : "Streaming is not getting through, so the rest of this answer is being polled from the server's replay buffer. The generation itself was never interrupted.",
      ),
  });

  // A push moves you somewhere you asked to go; a replace corrects an
  // address that was never a place. Opening a conversation and starting
  // a new chat are the first kind, so Back undoes them. The URL a
  // just-created conversation gets, and the URL a declined entry is put
  // back to, are the second.
  const onRoute = useCallback(
    (id: string | null, opts?: { replace?: boolean }) => {
      navigate(id ? `/ui/chat/${encodeURIComponent(id)}` : "/ui/chat", {
        replace: opts?.replace ?? false,
      });
    },
    [navigate],
  );

  // Above the provider on purpose: the sync loop needs the runtime
  // object itself (export/import/subscribe), not the React context a
  // component under the provider would read.
  const transcript = useTranscript(runtime, {
    model: () => servingRef.current.modelId,
    routeId: conversationId ?? null,
    onRoute,
  });

  // Coming back to a tab that has been away long enough drops to a new
  // chat — but never over the top of work in progress. A running
  // generation or a half-typed message means the tab was left mid-task,
  // and "you were away" is not a reason to throw that out.
  const transcriptRef = useLatest(transcript);
  useTabActivity(
    useCallback(
      (awayMs: number) => {
        if (runtime.thread.getState().isRunning) return;
        if (runtime.thread.composer.getState().text.trim()) return;
        transcriptRef.current.onReturnedAfterAway(awayMs);
      },
      [runtime, transcriptRef],
    ),
  );

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ChatInner
        sampling={sampling}
        setSampling={setSampling}
        serving={serving}
        refreshServing={refreshServing}
        stall={stall}
        transport={transport}
        transcript={transcript}
      />
    </AssistantRuntimeProvider>
  );
}
