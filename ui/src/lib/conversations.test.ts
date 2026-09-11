// Tests for the transcript sync's pure half, run by node's built-in
// test runner (`npm test`).
//
// The network half is the server's tests. What is worth pinning here is
// the diff: a sync loop that gets "which nodes are new" wrong either
// duplicates a message or drops one, and neither is visible until a
// reload the next day. Every case below is one of those.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  hasWork,
  isStorable,
  pendingAppend,
  plainText,
  restoredStatus,
  storedIds,
  toBranchable,
  type Conversation,
  type ExportedRepository,
} from "./conversations.ts";

function item(
  id: string,
  parentId: string | null,
  role: string,
  text: string,
  status?: { type: string; reason?: string },
) {
  return {
    parentId,
    message: {
      id,
      role,
      content: [{ type: "text", text }],
      ...(status ? { status } : {}),
    },
  };
}

function repo(
  messages: ExportedRepository["messages"],
  headId?: string | null,
): ExportedRepository {
  return { headId, messages };
}

test("a turn is stored parent-before-child in one batch", () => {
  const pending = pendingAppend(
    repo(
      [item("u1", null, "user", "hi"), item("a1", "u1", "assistant", "hello")],
      "a1",
    ),
    new Set(),
  );
  assert.deepEqual(
    pending.messages.map((m) => m.id),
    ["u1", "a1"],
  );
  assert.equal(pending.messages[1].parent_id, "u1");
  assert.equal(pending.headId, "a1");
});

test("nothing already stored is sent twice", () => {
  const exported = repo([
    item("u1", null, "user", "hi"),
    item("a1", "u1", "assistant", "hello"),
  ]);
  const first = pendingAppend(exported, new Set());
  const stored = new Set(first.messages.map((m) => m.id));
  const second = pendingAppend(exported, stored);
  assert.equal(second.messages.length, 0, "a second pass must find no work");
});

test("a message still streaming is not stored yet", () => {
  const pending = pendingAppend(
    repo([
      item("u1", null, "user", "hi"),
      item("a1", "u1", "assistant", "hel", { type: "running" }),
    ]),
    new Set(),
  );
  assert.deepEqual(
    pending.messages.map((m) => m.id),
    ["u1"],
    "a half-decoded answer would have to be rewritten on every token",
  );
});

test("a child of a skipped message waits for its parent", () => {
  // The server refuses a dangling parent, so sending this child now
  // would be a refusal the user never asked for. It lands on the tick
  // after its parent finishes.
  const pending = pendingAppend(
    repo([
      item("u1", null, "user", "hi", { type: "running" }),
      item("a1", "u1", "assistant", "hello"),
    ]),
    new Set(),
  );
  assert.equal(pending.messages.length, 0);
});

test("a failed answer is never stored", () => {
  const pending = pendingAppend(
    repo([
      item("u1", null, "user", "hi"),
      item("a1", "u1", "assistant", "", {
        type: "incomplete",
        reason: "error",
      }),
    ]),
    new Set(),
  );
  assert.deepEqual(
    pending.messages.map((m) => m.id),
    ["u1"],
    "the next turn drops a failed answer anyway; storing it desynchronises the two",
  );
});

test("a cancelled answer keeps the tokens it earned", () => {
  const pending = pendingAppend(
    repo([
      item("u1", null, "user", "hi"),
      item("a1", "u1", "assistant", "part of an answer", {
        type: "incomplete",
        reason: "cancelled",
      }),
    ]),
    new Set(),
  );
  assert.equal(pending.messages.length, 2);
  assert.equal(pending.messages[1].content, "part of an answer");
});

test("regenerating branches rather than replacing", () => {
  const exported = repo(
    [
      item("u1", null, "user", "hi"),
      item("a1", "u1", "assistant", "first"),
      item("a2", "u1", "assistant", "second"),
    ],
    "a2",
  );
  const pending = pendingAppend(exported, new Set(["u1", "a1"]));
  assert.deepEqual(
    pending.messages.map((m) => [m.id, m.parent_id]),
    [["a2", "u1"]],
  );
  assert.equal(pending.headId, "a2");
});

test("a head that names a skipped message is withheld", () => {
  // Sending it would be a 400 for a message the user cannot see; the
  // head catches up on the next write instead.
  const pending = pendingAppend(
    repo(
      [
        item("u1", null, "user", "hi"),
        item("a1", "u1", "assistant", "…", { type: "running" }),
      ],
      "a1",
    ),
    new Set(),
  );
  assert.equal(pending.headId, null);
});

test("a branch switch alone is still work", () => {
  const pending = pendingAppend(
    repo(
      [item("u1", null, "user", "hi"), item("a1", "u1", "assistant", "hello")],
      "u1",
    ),
    new Set(["u1", "a1"]),
  );
  assert.equal(pending.messages.length, 0);
  assert.equal(hasWork(pending, "a1"), true);
  assert.equal(hasWork(pending, "u1"), false);
});

test("only text becomes transcript content", () => {
  assert.equal(
    plainText([
      { type: "text", text: "a" },
      { type: "reasoning", text: "hidden" },
      { type: "text", text: "b" },
    ]),
    "ab",
  );
});

test("a role the server refuses is never offered to it", () => {
  assert.equal(
    isStorable({ id: "t1", role: "tool", content: [{ type: "text" }] }),
    false,
  );
  assert.equal(
    isStorable({ id: "", role: "user", content: [{ type: "text" }] }),
    false,
  );
});

test("metadata rides along so the usage line survives a reload", () => {
  const stats = { custom: { stats: { line: "TTFT 40 ms" } } };
  const pending = pendingAppend(
    repo([
      {
        parentId: null,
        message: {
          id: "a1",
          role: "assistant",
          content: [{ type: "text", text: "hi" }],
          metadata: stats,
        },
      },
    ]),
    new Set(),
  );
  assert.deepEqual(pending.messages[0].metadata, stats);
});

const CONVERSATION: Conversation = {
  object: "conversation",
  id: "conv_1",
  title: "t",
  model: null,
  created_at: 1_700_000_000,
  updated_at: 1_700_000_000,
  head_id: "a1",
  messages: [
    {
      id: "u1",
      parent_id: null,
      role: "user",
      content: "hi",
      created_at: 1_700_000_000,
    },
    {
      id: "a1",
      parent_id: "u1",
      role: "assistant",
      content: "hello",
      created_at: 1_700_000_001,
      metadata: { custom: { stats: { line: "TTFT 40 ms" } } },
    },
  ],
};

test("a stored tree reloads with its parents, head and metadata", () => {
  const { items, headId } = toBranchable(CONVERSATION);
  assert.equal(headId, "a1");
  assert.deepEqual(
    items.map((i) => [i.message.id, i.parentId]),
    [
      ["u1", null],
      ["a1", "u1"],
    ],
  );
  assert.deepEqual(items[0].message.content, [{ type: "text", text: "hi" }]);
  // An assistant message with no status reads as still running, which
  // would leave a reloaded transcript spinning forever.
  assert.deepEqual(items[1].message.status, {
    type: "complete",
    reason: "stop",
  });
  assert.equal(items[0].message.status, undefined);
  assert.deepEqual(items[1].message.metadata, {
    custom: { stats: { line: "TTFT 40 ms" } },
  });
  assert.equal(
    items[1].message.createdAt.getTime(),
    1_700_000_001_000,
    "the server stores seconds and Date takes milliseconds",
  );
});

test("a withheld head is not work, or the sync loop would spin", () => {
  // There is nothing to send that would move the stored head, so
  // counting the difference would make every export report the same
  // work and never finish it.
  const pending = pendingAppend(
    repo(
      [
        item("u1", null, "user", "hi"),
        item("a1", "u1", "assistant", "…", { type: "running" }),
      ],
      "a1",
    ),
    new Set(["u1"]),
  );
  assert.equal(pending.messages.length, 0);
  assert.equal(pending.headId, null);
  assert.equal(hasWork(pending, "u1"), false);
});

test("a reloaded conversation is immediately in sync", () => {
  const { items, headId } = toBranchable(CONVERSATION);
  const exported = repo(
    items.map((i) => ({ parentId: i.parentId, message: i.message })),
    headId,
  );
  const pending = pendingAppend(exported, storedIds(CONVERSATION));
  assert.equal(pending.messages.length, 0);
  assert.equal(
    hasWork(pending, CONVERSATION.head_id),
    false,
    "opening a conversation must not write it straight back",
  );
});

test("thinking is stored beside the answer, never inside it", () => {
  // The user's case: an R1 turn cut off inside its thought is ALL
  // reasoning. Stored as `content: ""` alone it reloaded as an empty
  // bubble with the thinking gone.
  const pending = pendingAppend(
    repo([
      item("u1", null, "user", "why"),
      {
        parentId: "u1",
        message: {
          id: "a1",
          role: "assistant",
          content: [{ type: "reasoning", text: "Let me think" }],
          status: { type: "incomplete", reason: "length" },
        },
      },
    ]),
    new Set(),
  );
  assert.equal(pending.messages.length, 2, "a cut-off turn is stored");
  assert.equal(pending.messages[1].content, "");
  assert.equal(pending.messages[1].reasoning_content, "Let me think");
  // A turn that never thought carries no key at all.
  assert.equal("reasoning_content" in pending.messages[0], false);
});

test("a stored thought reloads above its answer, and a cut-off turn offers the way out", () => {
  const conversation: Conversation = {
    ...CONVERSATION,
    messages: [
      CONVERSATION.messages[0],
      {
        id: "a1",
        parent_id: "u1",
        role: "assistant",
        content: "",
        reasoning_content: "Let me think",
        created_at: 1_700_000_001,
        metadata: { custom: { stats: { outcome: "length" } } },
      },
      {
        id: "a2",
        parent_id: "u1",
        role: "assistant",
        content: "hello",
        created_at: 1_700_000_002,
        metadata: { custom: { stats: { outcome: "stopped-by-you" } } },
      },
    ],
  };
  const { items } = toBranchable(conversation);
  assert.deepEqual(items[1].message.content, [
    { type: "reasoning", text: "Let me think" },
    { type: "text", text: "" },
  ]);
  // The status comes back from the outcome the runtime recorded, so
  // Continue is offered after a reload exactly where it was before.
  assert.deepEqual(items[1].message.status, {
    type: "incomplete",
    reason: "length",
  });
  assert.deepEqual(items[2].message.status, {
    type: "incomplete",
    reason: "cancelled",
  });
  assert.deepEqual(
    items[2].message.content,
    [{ type: "text", text: "hello" }],
    "a turn that never thought grows no empty reasoning part",
  );
  // A record from before the field existed, or with no outcome at all,
  // is complete: the old shape reads exactly as it did.
  assert.deepEqual(restoredStatus(undefined), { type: "complete", reason: "stop" });
  assert.deepEqual(restoredStatus({ custom: { stats: { outcome: "ok" } } }), {
    type: "complete",
    reason: "stop",
  });
});
