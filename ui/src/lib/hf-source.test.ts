import test from "node:test";
import assert from "node:assert/strict";
import { parseHfSource } from "./hf-source.ts";

test("an owner/repo identifier takes the fallback file", () => {
  assert.deepEqual(parseHfSource("unsloth/Llama-3.2-3B-Instruct-GGUF", "*Q4_K_M.gguf"), {
    repo: "unsloth/Llama-3.2-3B-Instruct-GGUF",
    file: "*Q4_K_M.gguf",
  });
});

test("a :suffix names the file and beats the fallback", () => {
  assert.deepEqual(parseHfSource("owner/repo:model-q8_0.gguf", "*Q4_K_M.gguf"), {
    repo: "owner/repo",
    file: "model-q8_0.gguf",
  });
});

test("a repo URL, with or without /tree/main, is the repo", () => {
  for (const u of [
    "https://huggingface.co/owner/repo",
    "https://huggingface.co/owner/repo/",
    "https://huggingface.co/owner/repo/tree/main",
    "hf.co/owner/repo",
    "https://www.huggingface.co/owner/repo?not-for-all-audiences=true",
  ]) {
    assert.deepEqual(parseHfSource(u, "*.gguf"), { repo: "owner/repo", file: "*.gguf" }, u);
  }
});

test("a file URL names the file, resolve or blob, nested or not", () => {
  assert.deepEqual(
    parseHfSource("https://huggingface.co/owner/repo/resolve/main/sub/m-Q4_K_M.gguf?download=true", "*.gguf"),
    { repo: "owner/repo", file: "sub/m-Q4_K_M.gguf" },
  );
  assert.deepEqual(
    parseHfSource("https://huggingface.co/owner/repo/blob/main/m%20x.gguf", "*.gguf"),
    { repo: "owner/repo", file: "m x.gguf" },
  );
});

test("junk is refused with a reason rather than sent", () => {
  for (const bad of ["", "just-a-name", "https://huggingface.co/", "https://huggingface.co/owner/repo/discussions/3", "owner/repo:"]) {
    const out = parseHfSource(bad, "");
    assert.ok("error" in out, bad);
  }
});
