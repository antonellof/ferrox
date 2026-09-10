// Connect: the snippets that point something else at this server.
//
// Everything here is filled from what is live right now — the model id
// `/v1/models` reports and the base URL this app is configured with —
// rather than from a placeholder the reader has to remember to edit.
// A snippet with `YOUR_MODEL_HERE` in it is a snippet that gets pasted
// with `YOUR_MODEL_HERE` still in it.
//
// One thing NOT taken from the live page: its own origin. Studio is a
// separate app now, so `window.location.origin` is usually a dev server
// or a static host, not the API. See `snippetBase()`.

import { useEffect, useMemo, useState } from "react";
import { KeyRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardBody,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { CopyButton } from "@/components/ui/copy-button";
import { Field, Input } from "@/components/ui/field";
import { Page, PageHeader } from "@/components/page";
import {
  apiBase,
  apiKey,
  DEFAULT_SERVER_ORIGIN,
  getJson,
  routes,
  setApiBase,
  setApiKey,
  snippetBase,
} from "@/lib/api";

function Snippet({
  heading,
  note,
  code,
}: {
  heading: string;
  note: string;
  code: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{heading}</CardTitle>
        <span className="flex-1" />
        <CopyButton getText={() => code} label="Copy snippet" showLabel />
      </CardHeader>
      <CardBody className="space-y-3">
        <p className="text-xs text-muted">{note}</p>
        {/* Same treatment a fenced block gets in the transcript: a
            recessed surface with its own hairline, so "this is verbatim
            text you paste" looks the same everywhere in the app. */}
        <pre className="overflow-x-auto rounded-md border border-line bg-sunken p-3 font-mono text-code">
          <code>{code}</code>
        </pre>
      </CardBody>
    </Card>
  );
}

export function ConnectScreen() {
  const [key, setKey] = useState(apiKey);
  const [savedKey, setSavedKey] = useState(apiKey);
  const [origin, setOrigin] = useState(apiBase);
  const [savedOrigin, setSavedOrigin] = useState(apiBase);
  const [modelId, setModelId] = useState<string | null>(null);
  const [status, setStatus] = useState("reading /v1/models…");
  const base = snippetBase();

  useEffect(() => {
    let cancelled = false;
    getJson<{ data?: { id: string }[] }>(routes.models)
      .then((body) => {
        if (cancelled) return;
        const id = body?.data?.[0]?.id ?? null;
        setModelId(id);
        setStatus(
          id
            ? `serving: ${id}`
            : 'no model loaded — snippets use the placeholder id "ferrox"',
        );
      })
      .catch((error: Error) => {
        if (cancelled) return;
        setStatus(`could not read ${routes.models}: ${error.message}`);
      });
    return () => {
      cancelled = true;
    };
  }, [savedKey, savedOrigin]);

  const snippets = useMemo(() => {
    const model = modelId || "ferrox";
    const authHeaderCurl = savedKey
      ? ` \\\n  -H "Authorization: Bearer ${savedKey}"`
      : "";
    const pyKey = savedKey
      ? `"${savedKey}"`
      : 'os.environ.get("FERROX_API_KEY", "not-needed")';

    return [
      {
        heading: "curl — streaming chat completion",
        note: "The same endpoint this Studio's Chat screen uses.",
        code: `curl ${base}${routes.chatCompletions} \\
  -H "Content-Type: application/json"${authHeaderCurl} \\
  -d '{
    "model": "${model}",
    "messages": [{"role": "user", "content": "Say hello in five words."}],
    "max_tokens": 64,
    "stream": true
  }'`,
      },
      {
        heading: "Python — the official OpenAI SDK",
        note: "Point base_url at /v1; nothing else changes.",
        code: `import os
from openai import OpenAI

client = OpenAI(
    base_url="${base}/v1",
    api_key=${pyKey},
)

stream = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Say hello in five words."}],
    max_tokens=64,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)`,
      },
      {
        heading: "Environment — editors and agents",
        note: "See docs/AGENTS_COOKBOOK.md for per-tool configuration.",
        code: `# Any OpenAI-compatible tool (editors, agents, SDKs)
OPENAI_BASE_URL=${base}/v1
OPENAI_API_KEY=${savedKey || "not-needed"}
# The model id this server is serving right now:
#   ${model}`,
      },
      {
        heading: "Probes",
        note: "Liveness, what is loaded, and the most recent request.",
        code: `curl -s ${base}${routes.health} | jq
curl -s ${base}${routes.models} | jq
curl -s ${base}${routes.adminStats} | jq '.recent[-1]'`,
      },
    ];
  }, [base, modelId, savedKey]);

  return (
    <Page>
      <PageHeader
        title="Connect"
        description="Copy-pasteable snippets, filled from the base URL below and the model id this server reports right now."
      />

      <Card>
        <CardHeader>
          <CardTitle>Connection</CardTitle>
          <span className="flex-1" />
          <span className="truncate font-mono text-2xs text-faint">
            {status}
          </span>
        </CardHeader>
        <CardBody>
          <form
            className="space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              const nextKey = key.trim();
              setApiKey(nextKey);
              setSavedKey(nextKey);
              setApiBase(origin);
              setSavedOrigin(apiBase());
            }}
          >
            {/* One row, and the hint lives UNDER it rather than inside
                the first field. As a field hint it was long enough to
                stretch that column, shove the key field onto its own
                line and leave the Save button stranded beside a gap. */}
            <div className="flex flex-wrap items-end gap-3">
              <Field
                label="API base URL"
                className="min-w-56 flex-1"
                htmlFor="apibase"
              >
                <Input
                  id="apibase"
                  value={origin}
                  onChange={(e) => setOrigin(e.target.value)}
                  placeholder="http://127.0.0.1:8383  (empty = same origin)"
                  className="font-mono text-xs"
                />
              </Field>
              <Field
                label="API key (FERROX_API_KEY)"
                className="min-w-56 flex-1"
                htmlFor="apikey"
              >
                <Input
                  id="apikey"
                  type="password"
                  autoComplete="off"
                  value={key}
                  onChange={(e) => setKey(e.target.value)}
                  placeholder="unset — this server may not require one"
                />
              </Field>
              <Button type="submit" variant="primary">
                <KeyRound />
                Save
              </Button>
            </div>
            <p className="text-2xs leading-relaxed text-faint">
              {origin
                ? "Cross-origin: the server needs FERROX_CORS_ORIGINS set to this app's origin."
                : `Empty — this app's own requests go to ${window.location.origin} and the dev server proxies them. Snippets below assume the server is at its default ${DEFAULT_SERVER_ORIGIN}.`}
            </p>
          </form>
        </CardBody>
        <CardFooter className="space-y-2">
          <p>
            The key is stored in this browser's localStorage and sent as an
            Authorization header, exactly as any other client would. Leave it
            empty when the server was started without{" "}
            <code className="font-mono">FERROX_API_KEY</code>. It is never put
            in a URL.
          </p>
          <p>
            Studio is a separate app from the server it talks to. Pointing it at
            another origin means the operator must start that server with{" "}
            <code className="font-mono">
              FERROX_CORS_ORIGINS={window.location.origin}
            </code>
            . The <code className="font-mono">*</code> wildcard is refused by
            design — a wildcard beside a bearer token is a credential-leak
            shape.
          </p>
        </CardFooter>
      </Card>

      {snippets.map((snippet) => (
        <Snippet key={snippet.heading} {...snippet} />
      ))}
    </Page>
  );
}
