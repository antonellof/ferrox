// Models: the library. What is on disk, how to get more, and how to give
// the memory back.
//
// **Choosing which model answers is not done here.** It is done in the
// picker in the Chat header, which is the only model selector in this
// app. This screen used to carry a second one — a `Load` button on every
// row, posting the same `/admin/models/load` to the same server for the
// same effect — plus an `Unload` in the header AND an `Unload` in the
// loaded row, and an `active:` badge restating what the row's own state
// column already said. Four controls and two badges for two verbs.
//
// The split it now follows is the one every UI of this shape converged
// on: a management screen installs, removes and reports; a picker beside
// the conversation selects. `Unload` stays because it is the one verb
// the picker cannot express — give the memory back without putting
// something else in it — and it exists exactly once.
//
// Two more rules this screen exists to respect.
//
// **A rate is shown only when the server calls the task `stable`.** The
// backend runs a rolling-window estimator that refuses to divide until
// it has enough samples, and sends `null` for rate and ETA until then.
// Recomputing either from `bytes_done` deltas on this side would put
// back exactly the "123 GB/s" first-tick flash the estimator exists to
// prevent, so nothing here ever divides.
//
// **A missing control surface is a state, not a crash.** `/admin/*` is
// only present in builds that have it; a 404 renders as a plain
// explanation rather than as a broken table.

import { useCallback, useEffect, useState } from "react";
import { Boxes, CloudDownload, RefreshCw, Search } from "lucide-react";
import { Link } from "react-router";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardBody,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, Input } from "@/components/ui/field";
import { EmptyState, Notice, Skeleton } from "@/components/ui/feedback";
import { Progress } from "@/components/ui/progress";
import { Table, TableScroll, Td, Th, Tr } from "@/components/ui/table";
import { Page, PageHeader } from "@/components/page";
import {
  ApiError,
  getJson,
  postJson,
  routes,
  type Inventory,
  type ModelEntry,
  type TaskView,
} from "@/lib/api";
import {
  fmtBytes,
  fmtDuration,
  fmtInt,
  fmtParams,
  fmtRate,
  isNum,
} from "@/lib/format";

/** Poll fast while something is moving, slowly when nothing is. */
const BUSY_POLL_MS = 1000;
const IDLE_POLL_MS = 5000;

type Banner = { text: string; tone: "info" | "warn" | "err" } | null;

function StateBadge({
  entry,
  activeId,
}: {
  entry: ModelEntry;
  activeId: string | null | undefined;
}) {
  const state = entry.id === activeId ? "loaded" : (entry.state ?? "available");
  const tone =
    state === "loaded"
      ? "ok"
      : state === "loading"
        ? "warn"
        : state === "error"
          ? "err"
          : "neutral";
  return (
    <Badge tone={tone} title={entry.error || undefined}>
      {state}
    </Badge>
  );
}

function TaskCard({
  task,
  onCancel,
}: {
  task: TaskView;
  onCancel: (id: string) => void;
}) {
  const p = task.progress ?? {};
  const fraction = isNum(p.fraction) ? p.fraction : null;
  const terminal = ["done", "error", "cancelled"].includes(task.status);

  const facts: string[] = [
    `${p.bytes_done ? fmtBytes(p.bytes_done) : "0 B"}${
      isNum(p.bytes_total) ? ` / ${fmtBytes(p.bytes_total)}` : ""
    }`,
  ];
  if (p.state === "stable") {
    // Only here. `warming` means the server declined to estimate, and
    // the honest render of that is the word, not a number.
    facts.push(fmtRate(p.rate_bytes_per_s));
    if (isNum(p.eta_seconds)) facts.push(`ETA ${fmtDuration(p.eta_seconds)}`);
  } else if (task.status === "running") {
    facts.push("measuring rate…");
  }
  if (task.error) facts.push(task.error);

  return (
    <li className="space-y-2 rounded-lg border border-line bg-inset/40 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-sm font-medium">
          {task.label}
        </span>
        <Badge
          tone={
            task.status === "error"
              ? "err"
              : task.status === "done"
                ? "ok"
                : task.status === "cancelled"
                  ? "neutral"
                  : "accent"
          }
        >
          {task.status}
        </Badge>
        {terminal ? null : (
          <Button
            variant="danger"
            size="sm"
            onClick={() => onCancel(task.task_id)}
          >
            Cancel
          </Button>
        )}
      </div>
      {terminal ? null : <Progress fraction={fraction} label={task.label} />}
      <p className="font-mono text-[0.6875rem] text-faint">
        {facts.join("  ·  ")}
      </p>
    </li>
  );
}

export function ModelsScreen() {
  const [inventory, setInventory] = useState<Inventory | null>(null);
  const [tasks, setTasks] = useState<TaskView[]>([]);
  const [unsupported, setUnsupported] = useState(false);
  const [banner, setBanner] = useState<Banner>(null);
  const [filter, setFilter] = useState("");
  const [repo, setRepo] = useState("");
  const [file, setFile] = useState("*Q4_K_M.gguf");
  const [queueing, setQueueing] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [models, taskList] = await Promise.all([
        getJson<Inventory>(routes.adminModels),
        getJson<{ tasks?: TaskView[] }>(routes.adminTasks),
      ]);
      setInventory(models);
      setTasks(taskList.tasks ?? []);
      setUnsupported(false);
    } catch (error) {
      if (error instanceof ApiError && error.isMissingEndpoint) {
        setUnsupported(true);
      } else {
        setBanner({
          text: `Could not read the control surface: ${(error as Error).message}`,
          tone: "err",
        });
      }
    }
  }, []);

  // Poll fast while something is moving, slowly when nothing is. `busy`
  // is derived from what the last answer said, so the effect re-arms at
  // the other rate the moment a download starts or finishes — no timer
  // is threaded through a ref to make that happen.
  const busy =
    tasks.some((t) => t.status === "queued" || t.status === "running") ||
    (inventory?.models ?? []).some((m) => m.state === "loading");

  useEffect(() => {
    // The lint rule below warns that an effect whose deps include state
    // this effect writes can cascade. That is what a poller IS, and the
    // cascade is bounded: `busy` and `unsupported` are booleans, so the
    // effect re-arms at most twice per real change of situation.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
    if (unsupported) return;
    const id = setInterval(refresh, busy ? BUSY_POLL_MS : IDLE_POLL_MS);
    return () => clearInterval(id);
  }, [refresh, busy, unsupported]);

  const act = async (label: string, run: () => Promise<unknown>) => {
    try {
      await run();
      setBanner(null);
    } catch (error) {
      setBanner({
        text: `${label} failed: ${(error as Error).message}`,
        tone: "err",
      });
    }
    await refresh();
  };

  const unloadModel = () =>
    act("Unload", () => postJson(routes.adminModelsUnload));
  const cancelTask = (taskId: string) =>
    act("Cancel", () => postJson(routes.adminTaskCancel(taskId)));

  const startDownload = async (event: React.FormEvent) => {
    event.preventDefault();
    setQueueing(true);
    try {
      // The server resolves a `*` glob against the repo's file list and
      // refuses anything that is not a plain `.gguf` child of the model
      // directory, so no validation is duplicated here.
      await postJson(routes.adminDownload, {
        repo: repo.trim(),
        file: file.trim(),
      });
      setBanner({ text: `Download queued for ${repo.trim()}.`, tone: "info" });
      await refresh();
    } catch (error) {
      setBanner({
        text: `Download refused: ${(error as Error).message}`,
        tone: "err",
      });
    } finally {
      setQueueing(false);
    }
  };

  if (unsupported) {
    return (
      <Page>
        <PageHeader
          title="Models"
          description="Inventory, load / unload, and Hugging Face downloads."
        />
        <Card>
          <CardBody>
            <Notice tone="warn">
              Not available in this build. This server answered{" "}
              <code className="font-mono">404</code> for the{" "}
              <code className="font-mono">/admin</code> control surface, so
              model inventory, loading and downloads cannot be driven from
              here. Chat and Activity are unaffected.
            </Notice>
          </CardBody>
        </Card>
      </Page>
    );
  }

  const active = inventory?.active ?? null;
  const needle = filter.trim().toLowerCase();
  const visible = (inventory?.models ?? []).filter(
    (m) =>
      !needle ||
      m.id.toLowerCase().includes(needle) ||
      (m.quant ?? "").toLowerCase().includes(needle) ||
      (m.arch ?? "").toLowerCase().includes(needle),
  );

  return (
    <Page>
      <PageHeader
        title="Models"
        description={
          inventory?.model_dir
            ? `Scanning ${inventory.model_dir}`
            : "What is on disk, Hugging Face downloads, and giving the memory back."
        }
        actions={
          <>
            {active ? (
              <Button variant="default" size="sm" onClick={unloadModel}>
                Unload {active}
              </Button>
            ) : null}
            <Button variant="ghost" size="sm" onClick={() => void refresh()}>
              <RefreshCw />
              Refresh
            </Button>
          </>
        }
      />

      {banner ? <Notice tone={banner.tone}>{banner.text}</Notice> : null}

      <Card>
        <CardHeader>
          <CardTitle>Inventory</CardTitle>
          {/* The active checkpoint is stated once on this screen. When
              there is one it is on the `Unload …` button, which names it
              and does something about it; only the empty case needs a
              badge of its own. The `state` column says the same thing
              per row and is where the eye goes anyway. */}
          {active ? null : <Badge tone="neutral">nothing loaded</Badge>}
          <span className="flex-1" />
          <div className="relative">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-faint" />
            <Input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter by id, quant or arch"
              aria-label="Filter models"
              className="h-8 w-56 pl-8 text-xs"
            />
          </div>
        </CardHeader>

        {!inventory ? (
          <CardBody className="space-y-2">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-9 w-full" />
            ))}
          </CardBody>
        ) : !inventory.models.length ? (
          <EmptyState
            icon={Boxes}
            title={
              inventory.model_dir
                ? "No .gguf checkpoints in the scanned directory"
                : "No model directory is configured"
            }
          >
            {inventory.model_dir ? (
              <>Download one below, or drop a file into that directory.</>
            ) : (
              <>
                Set <code className="font-mono">FERROX_MODEL_PATH</code> or{" "}
                <code className="font-mono">FERROX_MODEL_DIR</code> and restart
                the server.
              </>
            )}
          </EmptyState>
        ) : !visible.length ? (
          <EmptyState icon={Search} title={`Nothing matches “${filter}”`} />
        ) : (
          <TableScroll>
            <Table>
              <thead>
                <Tr className="hover:bg-transparent">
                  <Th>id</Th>
                  <Th>quant</Th>
                  <Th>arch</Th>
                  <Th numeric>context</Th>
                  <Th numeric>params</Th>
                  <Th numeric>on disk</Th>
                  <Th numeric>resident</Th>
                  <Th>state</Th>
                </Tr>
              </thead>
              <tbody>
                {visible.map((entry) => {
                  return (
                    <Tr key={entry.id}>
                      <Td mono className="max-w-[22rem]">
                        <span className="block truncate" title={entry.path}>
                          {entry.id}
                        </span>
                      </Td>
                      <Td>{entry.quant || "—"}</Td>
                      <Td>{entry.arch || "—"}</Td>
                      <Td numeric>
                        {isNum(entry.context_length)
                          ? fmtInt(entry.context_length)
                          : "—"}
                      </Td>
                      <Td numeric>{fmtParams(entry.param_count)}</Td>
                      <Td numeric>{fmtBytes(entry.size_bytes)}</Td>
                      {/* `resident_bytes` is null for anything the server
                          cannot measure; that is reported as unknown rather
                          than as the file size, which would be a guess
                          dressed as a measurement. */}
                      <Td numeric>
                        {isNum(entry.resident_bytes)
                          ? fmtBytes(entry.resident_bytes)
                          : "—"}
                      </Td>
                      <Td>
                        <StateBadge entry={entry} activeId={active} />
                      </Td>
                    </Tr>
                  );
                })}
              </tbody>
            </Table>
          </TableScroll>
        )}

        <CardFooter>
          Pick which of these answers in the model menu at the top of{" "}
          <Link to="/ui/chat" className="text-accent underline">
            Chat
          </Link>
          . A swap loads the checkpoint for every client of this server; an
          in-flight request finishes on the weights it started on.
        </CardFooter>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Download a checkpoint</CardTitle>
        </CardHeader>
        <CardBody>
          <form
            onSubmit={startDownload}
            className="flex flex-wrap items-end gap-3"
          >
            <Field
              label="Hugging Face repo"
              className="min-w-64 flex-1"
              htmlFor="repo"
            >
              <Input
                id="repo"
                required
                value={repo}
                onChange={(e) => setRepo(e.target.value)}
                placeholder="unsloth/Llama-3.2-3B-Instruct-GGUF"
              />
            </Field>
            <Field label="file (name or glob)" className="w-48" htmlFor="file">
              <Input
                id="file"
                required
                value={file}
                onChange={(e) => setFile(e.target.value)}
                placeholder="*Q4_K_M.gguf"
              />
            </Field>
            <Button type="submit" variant="primary" disabled={queueing}>
              <CloudDownload />
              Download
            </Button>
          </form>
        </CardBody>
        <CardFooter>
          <code className="font-mono">POST /admin/download</code> starts a task;
          progress appears below.
        </CardFooter>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Tasks</CardTitle>
          {tasks.length ? <Badge>{tasks.length}</Badge> : null}
        </CardHeader>
        {tasks.length ? (
          <CardBody>
            <ul className="space-y-2">
              {tasks.map((task) => (
                <TaskCard
                  key={task.task_id}
                  task={task}
                  onCancel={cancelTask}
                />
              ))}
            </ul>
          </CardBody>
        ) : (
          <EmptyState
            icon={CloudDownload}
            title="No downloads or loads have run yet"
          >
            A rate appears only once the server's estimator calls it stable —
            until then the progress line says so instead of guessing.
          </EmptyState>
        )}
      </Card>
    </Page>
  );
}
