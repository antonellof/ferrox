// Refuse a copyleft dependency before it reaches a shipped asset.
//
// frink is Apache-2.0 and this app is distributed as built JavaScript,
// so an AGPL or GPL package in `node_modules` would not merely sit in a
// lockfile — its code would be bundled into `dist/app.js` and handed to
// every viewer. This walks the installed tree and exits non-zero on
// anything that is not plainly permissive.
//
// Run with `npm run licenses`. CI runs it on every push.

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

/** SPDX ids we accept anywhere in the tree. */
const ALLOWED = new Set([
  "MIT",
  "MIT-0",
  "ISC",
  "0BSD",
  "BSD-2-Clause",
  "BSD-3-Clause",
  "Apache-2.0",
  "BlueOak-1.0.0",
  "Unlicense",
  "CC0-1.0",
  "Python-2.0",
  "WTFPL",
]);

/**
 * Build-time-only packages, matched by name prefix, and the one licence
 * each is allowed to carry.
 *
 * `lightningcss` (plus its per-platform binaries) is MPL-2.0 and
 * `caniuse-lite` is CC-BY-4.0. Both are Tailwind / browserslist
 * machinery that runs during `vite build` and contributes no code to
 * the bundle: MPL-2.0 is file-level copyleft on the MPL files
 * themselves, which are neither modified nor redistributed here, and
 * CC-BY-4.0 covers a browser-support data table that is read and
 * discarded. Neither reaches `dist/`.
 *
 * A prefix rather than an exact name because npm installs the native
 * half as `lightningcss-<platform>-<arch>`, which differs per machine
 * and would otherwise pass on one developer's laptop and fail on CI.
 */
const BUILD_TIME_ONLY = [
  ["lightningcss", "MPL-2.0"],
  ["caniuse-lite", "CC-BY-4.0"],
];

const found = new Map();

function idOf(pkg) {
  if (typeof pkg.license === "string") return pkg.license;
  if (pkg.license?.type) return pkg.license.type;
  if (Array.isArray(pkg.licenses))
    return pkg.licenses.map((l) => l.type ?? l).join(" OR ");
  return "UNKNOWN";
}

function walk(dir) {
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const path = join(dir, entry.name);
    if (entry.name.startsWith("@") || entry.name === "node_modules") {
      walk(path);
      continue;
    }
    try {
      const pkg = JSON.parse(readFileSync(join(path, "package.json"), "utf8"));
      if (pkg.name) found.set(pkg.name, idOf(pkg));
    } catch {
      /* not a package directory */
    }
    walk(join(path, "node_modules"));
  }
}

walk("node_modules");

const rejected = [];
for (const [name, license] of found) {
  // A dual licence is fine when any one of its terms is acceptable.
  const parts = license
    .replace(/[()]/g, "")
    .split(/\s+OR\s+/i)
    .map((s) => s.trim());
  if (parts.some((p) => ALLOWED.has(p))) continue;
  if (
    BUILD_TIME_ONLY.some(
      ([prefix, allowed]) => name.startsWith(prefix) && license === allowed,
    )
  )
    continue;
  rejected.push(`${name}: ${license}`);
}

const counts = new Map();
for (const license of found.values())
  counts.set(license, (counts.get(license) ?? 0) + 1);

console.log(`${found.size} installed packages`);
for (const [license, n] of [...counts].sort((a, b) => b[1] - a[1]))
  console.log(`  ${String(n).padStart(4)}  ${license}`);

if (rejected.length) {
  console.error(
    `\nREFUSED — not a permissive licence, and this tree is compiled into an Apache-2.0 binary:\n  ${rejected.join("\n  ")}`,
  );
  process.exit(1);
}
console.log("\nEvery dependency is permissively licensed.");
