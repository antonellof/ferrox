// The mark is drawn twice — by `components/logo.tsx`, which imports the
// geometry, and by `public/favicon.svg`, which is a static file nothing
// links to the component. Two structures that must agree about one thing,
// with nothing enforcing it, is this repo's dominant defect shape, and the
// failure here is silent in the worst possible way: the tab would keep
// showing the old mark while the app showed the new one, and nobody looks
// at a favicon on purpose.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import {
  CELL_PATH,
  CORE_R,
  SPOKES_PATH,
  STROKE_WIDTH,
  VIEWBOX,
} from "./logo-geometry.ts";

const favicon = readFileSync(
  join(dirname(fileURLToPath(import.meta.url)), "..", "..", "public", "favicon.svg"),
  "utf8",
);

test("the favicon draws the same mark as the component", () => {
  assert.ok(favicon.includes(`viewBox="${VIEWBOX}"`), "viewBox");
  assert.ok(favicon.includes(`d="${CELL_PATH}"`), "cell outline");
  assert.ok(favicon.includes(`d="${SPOKES_PATH}"`), "spokes");
  assert.ok(favicon.includes(`r="${CORE_R}"`), "core radius");
  assert.ok(favicon.includes(`stroke-width="${STROKE_WIDTH}"`), "stroke width");
});

test("the favicon carries its own colour in both themes", () => {
  // It cannot inherit `currentColor` from the page, so it has to state a
  // light value and a dark one. A favicon with one colour disappears into
  // one of the two browser chromes.
  assert.match(favicon, /prefers-color-scheme:\s*dark/);
  assert.equal(favicon.match(/color:\s*#[0-9a-f]{6}/gi)?.length, 2);
});

test("the mark is hueless: it never names a colour of its own", () => {
  // The component inherits `currentColor`; the geometry module must not
  // start carrying a fill or stroke value, or the "no brand hue" rule
  // acquires an exception nobody can see from the stylesheet.
  const source = readFileSync(
    join(dirname(fileURLToPath(import.meta.url)), "logo-geometry.ts"),
    "utf8",
  );
  assert.doesNotMatch(source, /#[0-9a-fA-F]{3,8}\b|oklch\(|rgb\(|hsl\(/);
});
