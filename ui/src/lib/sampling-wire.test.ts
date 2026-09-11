// Tests for the panel-to-request mapping.
//
// The failure these guard is silent on both sides: a `0` sent for an
// empty budget box makes every thought end the moment it opens, and a
// budget dropped on the way to the wire makes the box a decoration.
// Neither is an error anywhere.

import assert from "node:assert/strict";
import { test } from "node:test";

import { parseReasoningBudget, samplingToWire } from "./sampling-wire.ts";

test("an unset budget sends no field, and a set one sends the number", () => {
  const base = { temperature: 0.7, topP: 0.95, maxTokens: null };
  assert.deepEqual(samplingToWire({ ...base, reasoningBudget: null }), {
    temperature: 0.7,
    top_p: 0.95,
  });
  assert.deepEqual(samplingToWire({ ...base, reasoningBudget: 40 }), {
    temperature: 0.7,
    top_p: 0.95,
    reasoning_budget_tokens: 40,
  });
  // Zero is a value, not an absence: llama.cpp's "end immediately".
  assert.deepEqual(samplingToWire({ ...base, reasoningBudget: 0 }), {
    temperature: 0.7,
    top_p: 0.95,
    reasoning_budget_tokens: 0,
  });
  assert.deepEqual(
    samplingToWire({ ...base, maxTokens: 512, reasoningBudget: 40 }),
    {
      temperature: 0.7,
      top_p: 0.95,
      max_tokens: 512,
      reasoning_budget_tokens: 40,
    },
  );
});

test("the budget box reads empty and -1 as unrestricted and refuses below -1", () => {
  assert.equal(parseReasoningBudget("", 40), null);
  assert.equal(parseReasoningBudget("  ", 40), null);
  assert.equal(parseReasoningBudget("-1", 40), null);
  assert.equal(parseReasoningBudget("0", 40), 0);
  assert.equal(parseReasoningBudget("2000", null), 2000);
  assert.equal(parseReasoningBudget("12.6", null), 13);
  assert.equal(parseReasoningBudget("-2", 40), 40, "kept the last valid value");
  assert.equal(parseReasoningBudget("x", 40), 40);
});
