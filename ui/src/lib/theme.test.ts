// The palette, asserted rather than promised.
//
// Three invariants live in `index.css` as prose, and prose does not fail a
// build. Each of these has a silent failure mode: a token added to one
// theme and not the other renders as an unresolved `var()` (transparent,
// in exactly one theme, on exactly one screen); a `--color-*` pointing at
// a token that was renamed emits no CSS at all, so the utility class just
// stops working; and a hue creeping back into the neutral ramp is
// invisible in a diff of oklch triples but is the entire brief.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const css = readFileSync(
  join(dirname(fileURLToPath(import.meta.url)), "..", "index.css"),
  "utf8",
);

/** The tokens allowed to carry chroma, and what each one names. */
const SEMANTIC = ["ok", "ok-soft", "warn", "warn-soft", "err", "err-soft"];

/** `--name: value` pairs from one brace-delimited block. */
function declarations(block: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const [, name, value] of block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
    out.set(name, value.trim());
  }
  return out;
}

/** The `:root { … }` block that follows `from`, up to its closing brace. */
function rootBlock(source: string, from: number): string {
  const open = source.indexOf(":root {", from);
  assert.notEqual(open, -1, "expected a :root block");
  const close = source.indexOf("}", open);
  assert.notEqual(close, -1, "unterminated :root block");
  return source.slice(open, close);
}

const darkAt = css.indexOf("@media (prefers-color-scheme: dark)");
assert.notEqual(darkAt, -1, "the dark theme block is missing");

const light = declarations(rootBlock(css, 0));
const dark = declarations(rootBlock(css, darkAt));

test("the two themes define the same token names", () => {
  assert.deepEqual([...dark.keys()].sort(), [...light.keys()].sort());
  assert.ok(light.size > 20, "suspiciously few tokens parsed");
});

test("every Tailwind colour points at a token that exists", () => {
  const theme = css.slice(css.indexOf("@theme inline"));
  const refs = [...theme.matchAll(/--color-[\w-]+\s*:\s*var\((--[\w-]+)\)/g)];
  assert.ok(refs.length > 10, "suspiciously few --color-* mappings parsed");
  for (const [, token] of refs) {
    assert.ok(light.has(token), `--color-* points at unknown ${token}`);
  }
});

test("the neutral ramp is hueless — nothing carries a brand colour", () => {
  // This is the user-visible rule: chroma means "a state", and nothing
  // else. If a fourth hue is ever wanted, it has to answer what state it
  // names, and be added to SEMANTIC on purpose.
  for (const [theme, tokens] of [
    ["light", light],
    ["dark", dark],
  ] as const) {
    for (const [name, value] of tokens) {
      const oklch = /^oklch\(\s*[\d.]+\s+([\d.]+)\s/.exec(value);
      if (!oklch) continue;
      const chroma = Number(oklch[1]);
      const semantic = SEMANTIC.includes(name.slice(2));
      if (semantic) {
        assert.ok(chroma > 0, `${theme} ${name} lost its hue: ${value}`);
      } else {
        assert.equal(chroma, 0, `${theme} ${name} carries a hue: ${value}`);
      }
    }
  }
});

test("the semantic states are all still declared", () => {
  for (const name of SEMANTIC) {
    assert.ok(light.has(`--${name}`), `--${name} went missing`);
  }
});
