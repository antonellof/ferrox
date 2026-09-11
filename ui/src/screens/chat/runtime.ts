// The bridge between assistant-ui's thread runtime and ferrox's public
// `/v1/chat/completions`.
//
// assistant-ui owns the transcript, the composer, autoscroll, branching
// and the abort signal. This file owns exactly one thing: turning a run
// into an SSE request and turning the server's answer back into message
// parts. Every speed the UI prints comes from the server's own `usage`
// block, carried on the message as `metadata.custom.stats`. The one
// clock this file holds is the thought's (`lib/thought.ts`): how long
// the model reasoned before it began to answer is the gap between two
// deltas of the stream, which `usage` does not measure and only the
// stream's consumer can.

import { useState } from "react";
import {
  useLocalRuntime,
  type ChatModelAdapter,
  type ChatModelRunOptions,
  type ChatModelRunResult,
  type ThreadMessage,
} from "@assistant-ui/react";
import {
  ApiError,
  cancelGeneration,
  streamChat,
  type ChatMessage,
  type StreamResult,
  type Transport,
  type Usage,
} from "@/lib/api";
import { fmtInt, fmtMs, fmtNum, isNum } from "@/lib/format";
import { samplingToWire } from "@/lib/sampling-wire";
import { THOUGHT_KEY, type Thought } from "@/lib/thought";
import { useLatest } from "@/lib/use-latest";

/** Not re-exported by the react package under its own name. */
type RunConfig = ChatModelRunOptions["runConfig"];

export type Sampling = {
  system: string;
  temperature: number;
  topP: number;
  /**
   * `null` sends no `max_tokens` at all, and the server then bounds the
   * answer by the context window alone -- llama.cpp's `n_predict: -1`,
   * and what its own web UI defaults to.
   *
   * This used to be 512. That is a decode-era number: a reasoning model
   * spends more than that THINKING on an ordinary question, the budget
   * runs out inside the thought, and the server correctly returns
   * `finish_reason: "length"` with no answer at all. OpenAI's semantics
   * count reasoning inside the completion budget, so the accounting was
   * right and the number was wrong. There is no number that is right
   * for every model, which is why the default is now no number: the
   * context is the only limit that is always true.
   */
  maxTokens: number | null;
  /**
   * llama.cpp's `reasoning_budget_tokens`: how many tokens the model
   * may think for before the server forces the closing tag and the
   * answer begins. `null` sends nothing and the server's own default
   * applies (unrestricted unless it was started with
   * `--reasoning-budget`). Unlike `max_tokens`, this never cuts the
   * answer: it moves the model out of its thought and into one.
   */
  reasoningBudget: number | null;
};

export const DEFAULT_SAMPLING: Sampling = {
  system: "",
  temperature: 0.7,
  topP: 0.95,
  maxTokens: null,
  reasoningBudget: null,
};

/** The `max_tokens` the previous default sent. A saved settings blob
 * still carrying it was never a choice, so it is not kept. */
export const LEGACY_MAX_TOKENS = 512;

/**
 * What the UI prints under an answer.
 *
 * `line` is the server's `usage`, formatted. `outcome` says how the
 * generation ended when that is not simply "it finished" -- a short
 * answer and a truncated one look identical otherwise, and `length` is
 * the one that used to render as silence: the model was cut off, most
 * often inside its own thinking, and nothing said so.
 */
export type AnswerStats = {
  line: string;
  requestId: string | null;
  outcome: "ok" | "length" | "stopped-by-you" | "stopped-by-server" | "error";
  usage: Usage | null;
};

/** Whether an answer with this outcome can be picked up where it stopped. */
export function canContinue(outcome: AnswerStats["outcome"]): boolean {
  return (
    outcome === "length" ||
    outcome === "stopped-by-you" ||
    outcome === "stopped-by-server"
  );
}

/**
 * A partial turn to carry on from: what the cut-off message already
 * holds, so the continuation starts from it rather than from nothing.
 *
 * `thoughtMs` is how long the first attempt thought. A continuation
 * that resumes inside the thought adds to it rather than restarting
 * the clock, so the summary at the end counts the whole thought and
 * not the second half of it.
 */
export type ContinueFrom = {
  reasoning: string;
  text: string;
  thoughtMs?: number;
};

const CONTINUE_KEY = "continueFrom";

/**
 * The run config that turns a reload into a continuation.
 *
 * The parts ride in `runConfig.custom` because that is the one channel
 * assistant-ui carries from the button that starts a run to the adapter
 * that serves it. Written here and read by `readContinuation` below, so
 * the key is spelled once.
 */
export function continuationRun(from: ContinueFrom): RunConfig {
  return { custom: { [CONTINUE_KEY]: from } };
}

function readContinuation(runConfig: RunConfig): ContinueFrom | null {
  const value = runConfig.custom?.[CONTINUE_KEY] as Partial<ContinueFrom> | undefined;
  if (!value || typeof value !== "object") return null;
  return {
    reasoning: typeof value.reasoning === "string" ? value.reasoning : "",
    text: typeof value.text === "string" ? value.text : "",
    ...(isNum(value.thoughtMs) ? { thoughtMs: value.thoughtMs } : {}),
  };
}

/** The reasoning and text parts of a message, concatenated by kind. */
export function partsText(
  content: readonly { type: string; text?: string }[],
): ContinueFrom {
  const pick = (kind: string) =>
    content
      .filter((part) => part.type === kind && typeof part.text === "string")
      .map((part) => part.text as string)
      .join("");
  return { reasoning: pick("reasoning"), text: pick("text") };
}

/** Turns the server's `usage` into one line, omitting anything absent. */
export function statLine(
  usage: Usage | null | undefined,
  requestId: string | null,
): string {
  const parts: string[] = [];
  if (isNum(usage?.time_to_first_token_ms))
    parts.push(`TTFT ${fmtMs(usage.time_to_first_token_ms)}`);
  if (usage) {
    const prefill = [`${fmtInt(usage.prompt_tokens)} tok`];
    if (isNum(usage.prompt_per_second))
      prefill.push(`${fmtNum(usage.prompt_per_second)} tok/s`);
    if (isNum(usage.prompt_eval_duration_ms))
      prefill.push(fmtMs(usage.prompt_eval_duration_ms));
    parts.push(`prefill ${prefill.join(" · ")}`);

    const decode = [`${fmtInt(usage.completion_tokens)} tok`];
    if (isNum(usage.predicted_per_second))
      decode.push(`${fmtNum(usage.predicted_per_second)} tok/s`);
    if (isNum(usage.generation_duration_ms))
      decode.push(fmtMs(usage.generation_duration_ms));
    parts.push(`decode ${decode.join(" · ")}`);

    if (isNum(usage.cached_tokens))
      parts.push(`cached ${fmtInt(usage.cached_tokens)} tok`);
  }
  if (requestId) parts.push(requestId);
  return parts.join("  ·  ");
}

/** Flattens assistant-ui's parts back into the wire format. */
function toWire(messages: readonly ThreadMessage[]): ChatMessage[] {
  const wire: ChatMessage[] = [];
  for (const message of messages) {
    if (message.role !== "user" && message.role !== "assistant") continue;
    // A message that failed carries no answer worth replaying.
    if (message.status?.type === "incomplete" && message.status.reason === "error")
      continue;
    // History carries the answer only. A turn that was cut off inside
    // its thought has no answer and is not replayed; `Continue` on that
    // turn is the path that sends the thought back.
    const { text } = partsText(message.content);
    if (!text) continue;
    wire.push({ role: message.role, content: text });
  }
  return wire;
}

/** A callback stream turned into something `for await` can drain. */
function pump<T>() {
  const queue: T[] = [];
  let wake: (() => void) | null = null;
  let ended = false;
  return {
    push(value: T) {
      queue.push(value);
      wake?.();
      wake = null;
    },
    end() {
      ended = true;
      wake?.();
      wake = null;
    },
    async *drain(): AsyncGenerator<T, void> {
      for (;;) {
        while (queue.length) yield queue.shift()!;
        if (ended) return;
        await new Promise<void>((resolve) => {
          wake = resolve;
        });
      }
    },
  };
}

export type ChatDeps = {
  /** The model id `/v1/models` reports right now. */
  modelId: () => string | null;
  sampling: () => Sampling;
  /** `ms` when the stream has gone quiet, `null` when it came back. */
  onStall: (ms: number | null) => void;
  /**
   * How the answer is arriving right now: live, over a reconnect, or
   * over the polling fallback. Reported rather than hidden — a
   * reconnect that silently replaces the original connection leaves the
   * user watching an indicator that has stopped meaning anything.
   */
  onTransport: (transport: Transport) => void;
};

function makeAdapter(deps: ChatDeps): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal, runConfig }) {
      const sampling = deps.sampling();
      const wire: ChatMessage[] = [];
      if (sampling.system.trim())
        wire.push({ role: "system", content: sampling.system.trim() });
      wire.push(...toWire(messages));

      // A continuation: the cut-off turn goes back as the trailing
      // assistant message, thought and all, and the server is asked to
      // keep writing it rather than to open a new one. The new message
      // starts out holding what the old one had, so the stream appends
      // to a visible answer rather than replaying it.
      const resume = readContinuation(runConfig);
      if (resume) {
        wire.push({
          role: "assistant",
          content: resume.text,
          ...(resume.reasoning ? { reasoning_content: resume.reasoning } : {}),
        });
      }

      // ONE pump, carrying tagged chunks, rather than one per kind.
      // Two queues drained in sequence would be two structures that
      // have to agree about ordering, and they would disagree the first
      // time a model interleaved thinking with its answer.
      const tokens = pump<{ kind: "text" | "reasoning"; text: string }>();
      let requestId: string | null = null;
      let result: StreamResult | null = null;
      let failure: unknown = null;

      // Both cancellation tiers, because one is not enough. assistant-ui
      // aborts the fetch, which closes the socket — and the server now
      // notices that. But a proxy can hold the backend connection open,
      // so the explicit `POST /v1/cancel` goes out too. Both end at the
      // same server-side flag, so doing both is never worse than either.
      const onAbort = () => cancelGeneration(requestId);
      abortSignal.addEventListener("abort", onAbort, { once: true });

      const task = streamChat(
        {
          model: deps.modelId() || "ferrox",
          messages: wire,
          ...samplingToWire(sampling),
          ...(resume ? { continue_final_message: true } : {}),
        },
        {
          signal: abortSignal,
          onRequestId: (id) => {
            // Named on the first chunk, which is what makes an explicit
            // cancel possible at all — there is nothing to cancel by
            // before the server has said what this generation is called.
            requestId = id;
            live.add(id);
          },
          onToken: (token) => tokens.push({ kind: "text", text: token }),
          onReasoning: (token) =>
            tokens.push({ kind: "reasoning", text: token }),
          onStall: deps.onStall,
          onTransport: deps.onTransport,
        },
      )
        .then((r) => {
          result = r;
        })
        .catch((error) => {
          failure = error;
        })
        .finally(() => {
          if (requestId) live.delete(requestId);
          tokens.end();
        });

      let text = resume?.text ?? "";
      let reasoning = resume?.reasoning ?? "";
      // Thinking is shown ABOVE the answer, which is also the order it
      // arrives in. An empty part is never emitted: a model that does
      // not think must not grow an empty block, and an answer that has
      // not started must not be an empty bubble under one.
      const parts = () => [
        ...(reasoning ? [{ type: "reasoning" as const, text: reasoning }] : []),
        ...(text ? [{ type: "text" as const, text }] : []),
      ];

      // The thought clock. It starts on the first reasoning delta and
      // stops on the first content delta, or on the end of the stream
      // when no answer ever came (cut off inside the thought, or
      // stopped). A continuation inherits the earlier attempt's time
      // and, if it is still thinking, resumes the clock BEHIND `now` by
      // that much, so one running total covers both halves.
      let thought: Thought | undefined = isNum(resume?.thoughtMs)
        ? { state: "done", ms: resume.thoughtMs }
        : undefined;
      const thoughtStarts = () => {
        if (thought?.state === "thinking") return;
        const before = thought?.ms ?? 0;
        thought = { state: "thinking", startedAt: Date.now() - before };
      };
      const thoughtEnds = () => {
        if (thought?.state !== "thinking") return;
        thought = { state: "done", ms: Math.max(0, Date.now() - thought.startedAt) };
      };
      // Every yield carries the whole `custom` block: assistant-ui
      // replaces it per yield rather than merging key by key, so a
      // yield that named only `stats` would drop the thought.
      const custom = (rest: Record<string, unknown> = {}) => ({
        ...(thought ? { [THOUGHT_KEY]: thought } : {}),
        ...rest,
      });

      try {
        for await (const chunk of tokens.drain()) {
          if (chunk.kind === "reasoning") {
            reasoning += chunk.text;
            thoughtStarts();
          } else {
            text += chunk.text;
            thoughtEnds();
          }
          yield { content: parts(), metadata: { custom: custom() } };
        }
        await task;
      } finally {
        thoughtEnds();
        abortSignal.removeEventListener("abort", onAbort);
      }

      const finished = result as StreamResult | null;
      const error = failure;

      if (!error) {
        const id = finished?.requestId || requestId;
        // The server won the race: it noticed the cancel and closed the
        // stream cleanly, so this arrives as a finished response rather
        // than as an AbortError. Saying so is the difference between a
        // short answer and a truncated one, which look identical.
        const cancelled = finished?.finishReason === "cancelled";
        // `length` is the server saying the budget ran out, not the
        // model saying it was done. For a reasoning model that most
        // often happens INSIDE the thought, so the message has thinking
        // and no answer -- which, marked complete, looked like a model
        // that chose to say nothing.
        const cutOff = finished?.finishReason === "length";
        const stats: AnswerStats = {
          line: statLine(finished?.usage, id),
          requestId: id,
          outcome: cancelled ? "stopped-by-server" : cutOff ? "length" : "ok",
          usage: finished?.usage ?? null,
        };
        yield {
          // The reasoning is kept on the finished message too. Dropping
          // it here would make the thinking vanish at the moment the
          // answer completes, which reads as a rendering bug.
          content: parts(),
          status: cancelled
            ? { type: "incomplete", reason: "cancelled" }
            : cutOff
              ? { type: "incomplete", reason: "length" }
              : { type: "complete", reason: "stop" },
          metadata: { custom: custom({ stats }) },
        } satisfies ChatModelRunResult;
        return;
      }

      if (error instanceof Error && error.name === "AbortError") {
        // A stopped generation is not a failure: the tokens that did
        // arrive are kept, and the line says why there are no timings.
        yield {
          content: [
            ...(reasoning
              ? [{ type: "reasoning" as const, text: reasoning }]
              : []),
            {
              type: "text" as const,
              text: text || "_(stopped before any token arrived)_",
            },
          ],
          status: { type: "incomplete", reason: "cancelled" },
          metadata: {
            custom: custom({
              stats: {
                line: "",
                requestId,
                outcome: "stopped-by-you",
                usage: null,
              } satisfies AnswerStats,
            }),
          },
        } satisfies ChatModelRunResult;
        return;
      }

      const message =
        error instanceof ApiError && error.isAuth
          ? `${error.message}\n\nThis server requires an API key. Set it on the Connect screen.`
          : error instanceof Error
            ? error.message
            : String(error);
      throw new Error(message);
    },
  };
}

/**
 * Ids of generations that are on the wire right now.
 *
 * The tab closing mid-answer is precisely the case an AbortSignal
 * cannot cover: the page is gone before the abort is delivered. A
 * `keepalive` POST is the one request that survives it.
 */
const live = new Set<string>();

if (typeof window !== "undefined") {
  window.addEventListener("pagehide", () => {
    for (const id of live) cancelGeneration(id);
  });
}

export function useFerroxRuntime(deps: ChatDeps) {
  // The adapter is built once and reads its inputs through a latest-value
  // box, so a sampling change mid-conversation applies to the next send
  // without tearing down the runtime (which would drop the transcript).
  const ref = useLatest(deps);

  // Read through the box, never captured: each arrow is called at send
  // time, so the adapter always sees the settings as they are then.
  // eslint-disable-next-line react-hooks/refs -- see lib/use-latest.ts
  const [adapter] = useState(() =>
    makeAdapter({
      modelId: () => ref.current.modelId(),
      sampling: () => ref.current.sampling(),
      onStall: (ms) => ref.current.onStall(ms),
      onTransport: (transport) => ref.current.onTransport(transport),
    }),
  );

  return useLocalRuntime(adapter);
}
