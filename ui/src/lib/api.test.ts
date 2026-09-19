// Tests for the streaming client, run by node's built-in test runner
// (`npm test`) against a stubbed `fetch`.
//
// No test framework is installed on purpose: node ships one, and it
// strips the types itself, so the recovery paths below cost the bundle
// nothing and the licence check nothing. What they buy is the half of
// SSE hardening that cannot be proven server-side — a replay id nothing
// consumes is not hardening, and "the client resumes from the last id
// without repeating a token" is a claim about this file.

import assert from "node:assert/strict";
import { after, beforeEach, test } from "node:test";

import { streamChat } from "./api.ts";

type Call = { url: string; init: RequestInit };

const calls: Call[] = [];
let responders: ((call: Call) => Promise<Response>)[] = [];

const realFetch = globalThis.fetch;

/** An SSE body, delivered as one chunk per string. */
function sse(...frames: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) controller.enqueue(encoder.encode(frame));
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** One `data:` frame carrying a content delta, numbered like the server's. */
function chunk(index: number, text: string, extra: object = {}): string {
  const payload = JSON.stringify({
    id: "chatcmpl-1",
    choices: [{ index: 0, delta: { content: text } }],
    ...extra,
  });
  // `retry:` deliberately small: the client honours the server's value,
  // and a test that waited the real 1.5 s per reconnect would be a test
  // nobody runs.
  const retry = index === 0 ? "retry: 5\n" : "";
  return `${retry}id: chatcmpl-1:${index}\ndata: ${payload}\n\n`;
}

function finalChunk(index: number): string {
  const payload = JSON.stringify({
    id: "chatcmpl-1",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: 3, completion_tokens: 4 },
  });
  return `id: chatcmpl-1:${index}\ndata: ${payload}\n\nid: chatcmpl-1:${index + 1}\ndata: [DONE]\n\n`;
}

beforeEach(() => {
  calls.length = 0;
  responders = [];
  globalThis.fetch = (input: RequestInfo | URL, init: RequestInit = {}) => {
    const call = { url: String(input), init };
    calls.push(call);
    const responder = responders.shift();
    if (!responder) throw new Error(`unexpected fetch: ${call.url}`);
    return responder(call);
  };
});

after(() => {
  globalThis.fetch = realFetch;
});

function collect() {
  let text = "";
  return {
    onToken: (t: string) => {
      text += t;
    },
    text: () => text,
  };
}

test("a resumable request asks the server for a replay buffer and names itself", async () => {
  responders = [
    async () =>
      sse(
        chunk(0, "he", { request_id: "chatcmpl-1" }),
        chunk(1, "llo"),
        finalChunk(2),
      ),
  ];
  const sink = collect();
  const result = await streamChat(
    { model: "m", messages: [{ role: "user", content: "hi" }] },
    { onToken: sink.onToken },
  );

  assert.equal(sink.text(), "hello");
  assert.equal(result.finishReason, "stop");
  assert.equal(result.requestId, "chatcmpl-1");
  const body = JSON.parse(String(calls[0].init.body)) as Record<string, unknown>;
  assert.equal(body.stream, true);
  assert.equal(
    body.stream_resumable,
    true,
    "without this the server keeps no buffer and an id would be a promise it cannot keep",
  );
  const sent = calls[0].init.headers as Record<string, string>;
  assert.equal(sent["X-Frink-Client"], "frink-studio");
  assert.equal(calls.length, 1, "a completed stream needs no recovery");
});

test("a reasoning model's thinking arrives on its own channel, not as answer text", async () => {
  // The bug this covers: the client typed the delta as `{ content }`
  // only, so every `reasoning_content` frame was dropped. On an R1
  // distill that is most of the answer's wall-clock, so the transcript
  // sat empty and the stream looked dead until the answer landed whole.
  const thinking = `id: chatcmpl-1:0\ndata: ${JSON.stringify({
    id: "chatcmpl-1",
    request_id: "chatcmpl-1",
    choices: [{ index: 0, delta: { reasoning_content: "17 x 3 = 51" } }],
  })}\n\n`;
  responders = [async () => sse(thinking, chunk(1, "51"), finalChunk(2))];
  const sink = collect();
  let reasoning = "";
  await streamChat(
    { model: "m", messages: [{ role: "user", content: "hi" }] },
    {
      onToken: sink.onToken,
      onReasoning: (t) => {
        reasoning += t;
      },
    },
  );

  assert.equal(reasoning, "17 x 3 = 51");
  assert.equal(
    sink.text(),
    "51",
    "thinking must not be concatenated into the answer",
  );
});

test("a stream that dies mid-answer resumes from the last id, repeating nothing", async () => {
  responders = [
    // The connection dies after two events: no finish reason, no [DONE].
    async () => sse(chunk(0, "he", { request_id: "chatcmpl-1" }), chunk(1, "ll")),
    async () => sse(chunk(2, "o w"), chunk(3, "orld"), finalChunk(4)),
  ];
  const sink = collect();
  const result = await streamChat(
    { model: "m", messages: [{ role: "user", content: "hi" }] },
    { onToken: sink.onToken },
  );

  assert.equal(
    sink.text(),
    "hello world",
    "a resume that repeated delivered tokens would be worse than restarting",
  );
  assert.equal(result.finishReason, "stop");
  assert.equal(calls.length, 2);
  assert.equal(calls[1].url, "/v1/stream/chatcmpl-1");
  const resumeHeaders = calls[1].init.headers as Record<string, string>;
  assert.equal(
    resumeHeaders["Last-Event-ID"],
    "chatcmpl-1:1",
    "the reconnect must name where it stopped, or the server replays from zero",
  );
});

test("when the reconnect cannot be made, the rest is drained by polling", async () => {
  responders = [
    async () => sse(chunk(0, "he", { request_id: "chatcmpl-1" }), chunk(1, "ll")),
    // Both SSE reconnects fail at the network level, the way a proxy
    // hostile to text/event-stream does.
    async () => {
      throw new TypeError("network error");
    },
    async () => {
      throw new TypeError("network error");
    },
    async (call) => {
      assert.ok(
        call.url.includes("from=2"),
        `the poll must continue from the last event seen: ${call.url}`,
      );
      return json({
        request_id: "chatcmpl-1",
        events: [
          { index: 2, data: JSON.stringify({ choices: [{ delta: { content: "o!" } }] }) },
        ],
        next_index: 3,
        done: false,
      });
    },
    async () =>
      json({
        request_id: "chatcmpl-1",
        events: [
          {
            index: 3,
            data: JSON.stringify({
              choices: [{ delta: {}, finish_reason: "stop" }],
              usage: { prompt_tokens: 1, completion_tokens: 3 },
            }),
          },
          { index: 4, data: "[DONE]" },
        ],
        next_index: 5,
        done: false,
      }),
  ];
  const transports: string[] = [];
  const sink = collect();
  const result = await streamChat(
    { model: "m", messages: [{ role: "user", content: "hi" }] },
    { onToken: sink.onToken, onTransport: (t) => transports.push(t) },
  );

  assert.equal(sink.text(), "hello!");
  assert.equal(result.finishReason, "stop");
  assert.deepEqual(result.usage, { prompt_tokens: 1, completion_tokens: 3 });
  assert.ok(
    transports.includes("polling"),
    "falling back is a fact about what happened and is reported, not hidden",
  );
});

test("a replay window that has moved on is a truncation, not a partial answer", async () => {
  responders = [
    async () => sse(chunk(0, "he", { request_id: "chatcmpl-1" }), chunk(1, "ll")),
    async () =>
      json(
        { error: { message: "the replay window has moved past the requested position", code: "replay_window_lost" } },
        410,
      ),
  ];
  const sink = collect();
  await assert.rejects(
    () =>
      streamChat(
        { model: "m", messages: [{ role: "user", content: "hi" }] },
        { onToken: sink.onToken },
      ),
    (error: Error) => {
      assert.match(error.message, /truncated/);
      return true;
    },
    "a partial answer must never be presented as a finished one",
  );
});

test("a poll that lands past the replay window is a truncation too", async () => {
  const fail = async () => {
    throw new TypeError("network error");
  };
  responders = [
    async () => sse(chunk(0, "he", { request_id: "chatcmpl-1" }), chunk(1, "ll")),
    fail,
    fail,
    async () =>
      json(
        {
          error: {
            message: "the replay window for 'chatcmpl-1' has moved past the requested position",
            code: "replay_window_lost",
          },
        },
        410,
      ),
  ];
  await assert.rejects(
    () => streamChat({ model: "m", messages: [{ role: "user", content: "hi" }] }),
    (error: Error) => {
      assert.match(error.message, /truncated/);
      assert.match(error.message, /replay window/);
      return true;
    },
    "the fallback must fail closed as loudly as the reconnect does",
  );
});

test("a non-resumable stream that ends early fails closed and never reconnects", async () => {
  responders = [
    async () => sse(chunk(0, "he", { request_id: "chatcmpl-1" }), chunk(1, "ll")),
  ];
  await assert.rejects(
    () =>
      streamChat(
        { model: "m", messages: [{ role: "user", content: "hi" }] },
        { resumable: false },
      ),
    /truncated/,
  );
  assert.equal(
    calls.length,
    1,
    "with no replay buffer there is nothing to reconnect into, so nothing is tried",
  );
  const body = JSON.parse(String(calls[0].init.body)) as Record<string, unknown>;
  assert.equal(body.stream_resumable, false);
});

test("an error answer is read as JSON rather than as an empty stream", async () => {
  responders = [
    async () => json({ error: { message: "no model is loaded" } }, 503),
  ];
  await assert.rejects(
    () => streamChat({ model: "m", messages: [{ role: "user", content: "hi" }] }),
    /no model is loaded/,
  );
});
