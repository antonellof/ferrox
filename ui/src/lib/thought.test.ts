// The thought clock's pure half.
//
// What is worth pinning: the words a duration turns into (the summary
// line is the feature), that a running clock ticks from its stamp
// rather than freezing at the last delta, and that a stored shape the
// module did not write is refused rather than rendered as NaN seconds.

import assert from "node:assert/strict";
import { test } from "node:test";

import { fmtThought, readThought, thoughtMs } from "./thought.ts";

test("seconds under a minute, minutes and seconds above it", () => {
  assert.equal(fmtThought(0), "under a second");
  assert.equal(fmtThought(499), "under a second");
  assert.equal(fmtThought(500), "1 second");
  assert.equal(fmtThought(1_000), "1 second");
  assert.equal(fmtThought(15_000), "15 seconds");
  assert.equal(fmtThought(59_400), "59 seconds");
  assert.equal(fmtThought(59_600), "1 min");
  assert.equal(fmtThought(80_000), "1 min 20 s");
  assert.equal(fmtThought(3_600_000), "1 h");
  assert.equal(fmtThought(3_900_000), "1 h 5 min");
});

test("a thought in progress is measured from its stamp, never negative", () => {
  assert.equal(thoughtMs({ state: "thinking", startedAt: 1_000 }, 16_000), 15_000);
  // A clock that went backwards (a resumed laptop) reads zero, not a
  // negative number of seconds.
  assert.equal(thoughtMs({ state: "thinking", startedAt: 5_000 }, 4_000), 0);
  assert.equal(thoughtMs({ state: "done", ms: 12_345 }, 0), 12_345);
});

test("the metadata round-trips, and anything else reads as no thought", () => {
  assert.deepEqual(readThought({ custom: { thought: { state: "done", ms: 15_000 } } }), {
    state: "done",
    ms: 15_000,
  });
  assert.deepEqual(
    readThought({ custom: { thought: { state: "thinking", startedAt: 7 } } }),
    { state: "thinking", startedAt: 7 },
  );
  for (const bad of [
    undefined,
    null,
    {},
    { custom: {} },
    { custom: { thought: null } },
    { custom: { thought: "15s" } },
    { custom: { thought: { state: "done" } } },
    { custom: { thought: { state: "done", ms: -1 } } },
    { custom: { thought: { state: "done", ms: "15000" } } },
    { custom: { thought: { state: "thinking", ms: 1 } } },
    { custom: { thought: { state: "paused", startedAt: 1 } } },
  ]) {
    assert.equal(readThought(bad as never), undefined, JSON.stringify(bad));
  }
});
