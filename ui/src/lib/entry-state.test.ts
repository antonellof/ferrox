// Tests for the entry rule: what Chat opens with, and why.
//
// This is worth pinning without a browser because every failure mode is
// silent. A rule that is too eager throws away the conversation someone
// was in the middle of; a rule that is too lazy is the bug this replaces,
// where every entry into the app resurrected the newest conversation for
// ever. Neither shows up as an error anywhere.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  decideEntry,
  describeAway,
  RESUME_WINDOW_MS,
  type EntryDecision,
} from "./entry-state.ts";

const NOW = 1_700_000_000_000;

test("the base URL is always a new chat, however recently the tab was active", () => {
  const decision = decideEntry({
    conversationId: null,
    lastActiveAt: NOW - 1000,
    now: NOW,
  });
  assert.deepEqual(decision, { kind: "fresh", awayMs: null, resumable: null });
});

test("a deep link with no activity record is honoured", () => {
  // A bookmark, a shared link, a brand-new tab: `sessionStorage` is
  // empty, and an absent record must not read as "away for ever".
  const decision = decideEntry({
    conversationId: "conv_1",
    lastActiveAt: null,
    now: NOW,
  });
  assert.deepEqual(decision, { kind: "resume", conversationId: "conv_1" });
});

test("a tab that was active a moment ago keeps its conversation", () => {
  const decision = decideEntry({
    conversationId: "conv_1",
    lastActiveAt: NOW - RESUME_WINDOW_MS + 1,
    now: NOW,
  });
  assert.deepEqual(decision, { kind: "resume", conversationId: "conv_1" });
});

test("exactly at the window is still inside it", () => {
  // The boundary is a strict `>` on purpose: a rule that fires at
  // exactly the window would fire a heartbeat early on a tab that has
  // been away for precisely the allowed time.
  const decision = decideEntry({
    conversationId: "conv_1",
    lastActiveAt: NOW - RESUME_WINDOW_MS,
    now: NOW,
  });
  assert.deepEqual(decision, { kind: "resume", conversationId: "conv_1" });
});

test("past the window is a new chat that still names what it declined", () => {
  const awayMs = RESUME_WINDOW_MS + 60_000;
  const decision = decideEntry({
    conversationId: "conv_1",
    lastActiveAt: NOW - awayMs,
    now: NOW,
  });
  assert.deepEqual(decision, {
    kind: "fresh",
    awayMs,
    resumable: "conv_1",
  } satisfies EntryDecision);
});

test("a clock that went backwards keeps the conversation", () => {
  // Machine slept, NTP corrected, timestamp is in the future. The safe
  // side of this comparison is "not away": a bad clock must not delete
  // somebody's place.
  const decision = decideEntry({
    conversationId: "conv_1",
    lastActiveAt: NOW + 86_400_000,
    now: NOW,
  });
  assert.deepEqual(decision, { kind: "resume", conversationId: "conv_1" });
});

test("the window is a parameter, so local mode and tab mode share one rule", () => {
  const input = {
    conversationId: "local",
    lastActiveAt: NOW - 5_000,
    now: NOW,
  };
  assert.deepEqual(decideEntry({ ...input, windowMs: 10_000 }), {
    kind: "resume",
    conversationId: "local",
  });
  assert.deepEqual(decideEntry({ ...input, windowMs: 1_000 }), {
    kind: "fresh",
    awayMs: 5_000,
    resumable: "local",
  });
});

test("the away time reads as a sentence, and never as zero", () => {
  assert.equal(describeAway(null), "a while");
  assert.equal(describeAway(30_000), "1 minutes");
  assert.equal(describeAway(42 * 60_000), "42 minutes");
  assert.equal(describeAway(3 * 3_600_000), "3 hours");
  assert.equal(describeAway(3 * 86_400_000), "3 days");
});
