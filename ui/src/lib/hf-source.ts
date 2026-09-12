/**
 * What a user pastes into the download box, resolved to the
 * `{ repo, file }` pair `POST /admin/download` takes.
 *
 * Three spellings are accepted, because those are the three a person
 * has on their clipboard:
 *
 * - a Hugging Face identifier, `owner/repo`, with an optional
 *   `:file-or-glob` suffix (`unsloth/Llama-3.2-3B-Instruct-GGUF:*Q4_K_M.gguf`);
 * - a repo URL, `https://huggingface.co/owner/repo` (or `hf.co`, with
 *   or without `/tree/main`);
 * - a file URL, `https://huggingface.co/owner/repo/resolve/main/file.gguf`
 *   (or `/blob/main/`), which names the file too.
 *
 * `file` comes from the suffix or the URL when either names one, else
 * from the caller's pattern box. Nothing here validates that the file
 * exists: the server resolves a `*` glob against the repo's file list
 * and refuses anything that is not a plain `.gguf`.
 */
export type HfSource = { repo: string; file: string };

export function parseHfSource(
  input: string,
  fallbackFile: string,
): HfSource | { error: string } {
  const raw = input.trim();
  if (!raw) return { error: "Paste a Hugging Face repo or URL." };

  let path = raw;
  let fileFromUrl: string | null = null;
  const url = /^(?:https?:\/\/)?(?:www\.)?(?:huggingface\.co|hf\.co)\/(.+)$/i.exec(raw);
  if (url) {
    path = url[1].replace(/[?#].*$/, "").replace(/\/+$/, "");
    const parts = path.split("/").filter(Boolean);
    if (parts.length < 2) return { error: "That URL names no repo." };
    const [owner, repo, ...rest] = parts;
    path = `${owner}/${repo}`;
    // `/resolve/main/<file>` and `/blob/main/<file>` name a file;
    // `/tree/main` names the repo and nothing more.
    if ((rest[0] === "resolve" || rest[0] === "blob") && rest.length >= 3) {
      fileFromUrl = decodeURIComponent(rest.slice(2).join("/"));
    } else if (rest.length && rest[0] !== "tree") {
      return { error: `That URL is not a repo or file page (${rest.join("/")}).` };
    }
  }

  let repo = path;
  let fileFromSuffix: string | null = null;
  const colon = path.indexOf(":");
  if (colon > 0) {
    repo = path.slice(0, colon);
    fileFromSuffix = path.slice(colon + 1).trim() || null;
  }
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) {
    return {
      error: `“${repo}” is not an owner/repo identifier.`,
    };
  }
  const file = (fileFromUrl ?? fileFromSuffix ?? fallbackFile).trim();
  if (!file) return { error: "Say which file: a name or a glob like *Q4_K_M.gguf." };
  return { repo, file };
}
