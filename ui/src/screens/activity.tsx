// Activity: the live request log.
//
// `duration_ms` and `decode_ms` get their own columns and are never
// added, averaged or collapsed into one "latency". `duration_ms` is
// wall time for the whole request — queue wait, prefill and decode —
// while `decode_ms` is the decode loop alone. A screen that showed only
// the first would read a 50 tok/s model as 5 whenever the prompt is
// long, and every throughput number a user quoted from it would be
// wrong in the same direction.
//
// Rows are keyed by `request_id`, which the server states in the first
// SSE chunk of the response that produced them, so a chat message and
// its log line can be joined exactly rather than by a timing heuristic.
//
// Two attribution columns, and they are worth different amounts. `key`
// is a fingerprint of the bearer token that actually authenticated the
// request — real, checked, and never the token itself. `client` is what
// the caller SAID it was; this app sends `ferrox-studio` and so could
// anything else. The screen labels the second one as self-declared
// rather than dressing it up as identity, because a monitor that
// overstates what it knows is worse than one that shows less.

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { ArrowDown, ArrowUp, Inbox, ChevronsUpDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardBody,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { EmptyState, Notice, Skeleton } from "@/components/ui/feedback";
import { Sparkline } from "@/components/ui/sparkline";
import { StatTile, StatValue } from "@/components/ui/stat";
import { Table, TableScroll, Td, Tr } from "@/components/ui/table";
import { Page, PageHeader, SectionLabel } from "@/components/page";
import {
  ApiError,
  getJson,
  routes,
  type Stats,
  type StatsRow,
} from "@/lib/api";
import {
  fmtClock,
  fmtDuration,
  fmtInt,
  fmtMs,
  fmtNum,
  isNum,
} from "@/lib/format";
import { cn } from "@/lib/utils";

const POLL_MS = 2000;

function Counter({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <StatTile label={label} hint={hint}>
      <StatValue>{value}</StatValue>
    </StatTile>
  );
}

/** The one grid the counters and the trend lines share. */
const TILES = "grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5";

const column = createColumnHelper<StatsRow>();

/**
 * Decode throughput for one row, or null.
 *
 * Computed from `completion_tokens / decode_ms` and from nothing else.
 * `duration_ms` carries queue wait and prefill; dividing by it is
 * exactly the mistake this screen keeps two columns to avoid.
 */
function decodeRate(row: StatsRow): number | null {
  if (!isNum(row.decode_ms) || row.decode_ms <= 0) return null;
  if (!isNum(row.completion_tokens) || row.completion_tokens <= 0) return null;
  return (row.completion_tokens / row.decode_ms) * 1000;
}

const columns = [
  column.accessor("at_ms", {
    header: "at",
    cell: (c) => <span className="font-mono">{fmtClock(c.getValue())}</span>,
  }),
  column.accessor("request_id", {
    header: "request_id",
    cell: (c) => (
      <span
        className="block max-w-[16rem] truncate font-mono"
        title={c.getValue()}
      >
        {c.getValue()}
      </span>
    ),
  }),
  column.accessor("route", { header: "route" }),
  column.accessor((r) => r.model ?? "", {
    id: "model",
    header: "model",
    cell: (c) =>
      c.getValue() ? (
        <span
          className="block max-w-[12rem] truncate font-mono"
          title={`Served by ${c.getValue()} — not necessarily the model the request asked for.`}
        >
          {c.getValue()}
        </span>
      ) : (
        <span
          className="text-faint"
          title="Nothing was loaded, so nothing served it."
        >
          —
        </span>
      ),
  }),
  column.accessor("status", {
    header: "status",
    cell: (c) => (
      <Badge tone={c.getValue() >= 400 ? "err" : "ok"}>{c.getValue()}</Badge>
    ),
  }),
  column.accessor((r) => (r.stream ? "stream" : "once"), {
    id: "mode",
    header: "mode",
    cell: (c) => <span className="text-faint">{c.getValue()}</span>,
  }),
  column.accessor("prompt_tokens", {
    header: "prompt",
    cell: (c) => fmtInt(c.getValue()),
    meta: { numeric: true },
  }),
  column.accessor("completion_tokens", {
    header: "gen",
    cell: (c) => fmtInt(c.getValue()),
    meta: { numeric: true },
  }),
  column.accessor("ttft_ms", {
    header: "ttft",
    cell: (c) => (isNum(c.getValue()) ? fmtMs(c.getValue()) : "—"),
    meta: { numeric: true },
  }),
  column.accessor("duration_ms", {
    header: "duration",
    cell: (c) => fmtMs(c.getValue()),
    meta: { numeric: true },
  }),
  column.accessor("decode_ms", {
    header: "decode",
    cell: (c) => (isNum(c.getValue()) ? fmtMs(c.getValue()) : "—"),
    meta: { numeric: true },
  }),
  column.accessor(decodeRate, {
    id: "decode_rate",
    header: "tok/s",
    cell: (c) => (isNum(c.getValue()) ? fmtNum(c.getValue()) : "—"),
    meta: { numeric: true },
  }),
  column.accessor((r) => r.client ?? "", {
    id: "client",
    header: "client",
    cell: (c) =>
      c.getValue() ? (
        <span
          className="font-mono"
          title="Self-declared by the caller (X-Ferrox-Client). Nothing authenticates it."
        >
          {c.getValue()}
        </span>
      ) : (
        <span
          className="text-faint"
          title="This caller did not name itself. Ferrox Studio always does, so this request came from something else."
        >
          —
        </span>
      ),
  }),
  column.accessor((r) => r.via_api_key ?? "", {
    id: "via_api_key",
    header: "key",
    cell: (c) =>
      c.getValue() ? (
        <span
          className="font-mono"
          title="Fingerprint of the key that served this request — not the key, and only comparable within this server run."
        >
          {c.getValue()}
        </span>
      ) : (
        <span
          className="text-faint"
          title="No Authorization header was presented. On a server started without FERROX_API_KEY that is every request."
        >
          —
        </span>
      ),
  }),
];

export function ActivityScreen() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [unsupported, setUnsupported] = useState(false);
  const [stale, setStale] = useState<string | null>(null);
  const [sorting, setSorting] = useState<SortingState>([]);
  const refresh = useCallback(async () => {
    try {
      const body = await getJson<Stats>(routes.adminStats);
      setStats(body);
      setStale(null);
    } catch (error) {
      if (error instanceof ApiError && error.isMissingEndpoint) {
        setUnsupported(true);
        return;
      }
      // A transient failure must not wipe the last good table: keep it
      // and say the numbers are stale.
      setStale((error as Error).message);
    }
  }, []);

  useEffect(() => {
    void refresh();
    if (unsupported) return;
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh, unsupported]);

  // The ring buffer arrives newest-last; a log reads newest-first.
  const recent = useMemo(
    () => [...(stats?.recent ?? [])].reverse(),
    [stats?.recent],
  );

  // The React Compiler cannot see through TanStack Table's builder, so
  // this one call opts out rather than the whole file carrying a
  // suppression it does not need.
  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: recent,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (unsupported) {
    return (
      <Page>
        <PageHeader title="Activity" description="The server's request ring." />
        <Card>
          <CardBody>
            <Notice tone="warn">
              Not available in this build.{" "}
              <code className="font-mono">{routes.adminStats}</code> answered
              404, so there is no request log to show.{" "}
              <code className="font-mono">/metrics</code> may still carry
              Prometheus counters.
            </Notice>
          </CardBody>
        </Card>
      </Page>
    );
  }

  // Sparklines read the ring oldest-first, which is the order it arrived.
  const trend = stats?.recent ?? [];

  return (
    <Page>
      <PageHeader
        title="Activity"
        description="Every request this process served, keyed by the request_id the response carried."
      />

      {stale ? <Notice tone="err">Stale: {stale}</Notice> : null}

      <section>
        <SectionLabel>Server counters</SectionLabel>
        <div>
          {!stats ? (
            <div className={TILES}>
              {Array.from({ length: 9 }, (_, i) => (
                <Skeleton key={i} className="h-16" />
              ))}
            </div>
          ) : (
            <div className={TILES}>
              <Counter
                label="uptime"
                value={fmtDuration(stats.uptime_seconds)}
              />
              <Counter label="requests" value={fmtInt(stats.requests_total)} />
              <Counter label="errors" value={fmtInt(stats.errors_total)} />
              <Counter label="cache hits" value={fmtInt(stats.cache_hits)} />
              <Counter
                label="cache misses"
                value={fmtInt(stats.cache_misses)}
              />
              <Counter
                label="prompt tokens"
                value={fmtInt(stats.tokens_prompt_total)}
              />
              <Counter
                label="generated tokens"
                value={fmtInt(stats.tokens_generated_total)}
              />
              <Counter
                label="generating now"
                value={
                  isNum(stats.generating_now)
                    ? fmtInt(stats.generating_now)
                    : "—"
                }
                hint="Streamed generations decoding at this instant — the ones POST /v1/cancel could stop. Work in progress, not a queue depth: nothing waits in front of a decode here."
              />
              <Counter
                label="queued"
                value={
                  isNum(stats.queue_depth) ? fmtInt(stats.queue_depth) : "—"
                }
                hint="Requests waiting for a decode slot in the continuous-batching scheduler. “—” means there is no queue to measure: without batching every request gets its own thread and nothing waits in front of anything."
              />
              <Counter
                label="queue rejected"
                value={
                  isNum(stats.queue_rejected_total)
                    ? fmtInt(stats.queue_rejected_total)
                    : "—"
                }
                hint="Requests the scheduler's queue turned away because it was full, since this process started."
              />
              <Counter
                label="last request"
                value={
                  isNum(stats.last_request_age_seconds)
                    ? `${fmtDuration(stats.last_request_age_seconds)} ago`
                    : "—"
                }
                hint="Recent activity is positive evidence of liveness even when /health is slow to answer."
              />
            </div>
          )}
        </div>
      </section>

      {trend.length >= 2 ? (
        <section>
          <SectionLabel>
            Trend over the last {trend.length} requests
          </SectionLabel>
          <div className="grid gap-3 sm:grid-cols-3">
            {(
              [
                [
                  "TTFT (ms)",
                  trend.map((r) => (isNum(r.ttft_ms) ? r.ttft_ms : null)),
                ],
                [
                  "decode (ms)",
                  trend.map((r) => (isNum(r.decode_ms) ? r.decode_ms : null)),
                ],
                ["decode tok/s", trend.map(decodeRate)],
              ] as const
            ).map(([label, values]) => (
              <StatTile key={label} label={label}>
                <Sparkline label={label} values={[...values]} />
              </StatTile>
            ))}
          </div>
        </section>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Recent requests</CardTitle>
          <Badge>{recent.length}</Badge>
        </CardHeader>

        {!stats ? (
          <CardBody className="space-y-2">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-8 w-full" />
            ))}
          </CardBody>
        ) : !recent.length ? (
          <EmptyState icon={Inbox} title="No requests yet">
            Send a message on the Chat screen, or point an editor at this server
            — external traffic lands here too.
          </EmptyState>
        ) : (
          <TableScroll className="max-h-[32rem] overflow-y-auto">
            <Table>
              <thead>
                {table.getHeaderGroups().map((group) => (
                  <Tr key={group.id} className="hover:bg-transparent">
                    {group.headers.map((header) => {
                      const numeric = (
                        header.column.columnDef.meta as
                          { numeric?: boolean } | undefined
                      )?.numeric;
                      const sorted = header.column.getIsSorted();
                      const Icon =
                        sorted === "asc"
                          ? ArrowUp
                          : sorted === "desc"
                            ? ArrowDown
                            : ChevronsUpDown;
                      return (
                        <th
                          key={header.id}
                          scope="col"
                          className="sticky top-0 z-10 border-b border-line bg-raised p-0 text-left first:[&>button]:pl-4 last:[&>button]:pr-4"
                        >
                          <button
                            type="button"
                            onClick={header.column.getToggleSortingHandler()}
                            className={cn(
                              "flex w-full items-center gap-1 px-3 py-2 text-2xs font-medium tracking-wide text-faint uppercase transition-colors hover:text-fg",
                              numeric && "justify-end",
                            )}
                          >
                            {flexRender(
                              header.column.columnDef.header,
                              header.getContext(),
                            )}
                            <Icon
                              className={cn(
                                "size-3",
                                sorted ? "text-accent" : "opacity-40",
                              )}
                            />
                          </button>
                        </th>
                      );
                    })}
                  </Tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.map((row) => (
                  <Tr key={row.id}>
                    {row.getVisibleCells().map((cell) => (
                      <Td
                        key={cell.id}
                        numeric={
                          (
                            cell.column.columnDef.meta as
                              { numeric?: boolean } | undefined
                          )?.numeric
                        }
                      >
                        {flexRender(
                          cell.column.columnDef.cell,
                          cell.getContext(),
                        )}
                      </Td>
                    ))}
                  </Tr>
                ))}
              </tbody>
            </Table>
          </TableScroll>
        )}

        <CardFooter>
          <strong className="font-semibold text-muted">duration</strong> is the
          whole request — queue wait, prefill and decode.{" "}
          <strong className="font-semibold text-muted">decode</strong> is the
          decode loop alone; it is “—” when the engine did not time itself or
          the answer came from cache. The{" "}
          <strong className="font-semibold text-muted">tok/s</strong> column is{" "}
          <code className="font-mono">completion_tokens / decode_ms</code> and
          nothing else — dividing by duration is how a fast model gets reported
          as a slow one, so the two are never combined here.{" "}
          <strong className="font-semibold text-muted">model</strong> is the
          model that answered, not the one the request named — this server
          decodes against whatever is loaded, so the two differ whenever a
          client is configured for someone else’s model id.{" "}
          <strong className="font-semibold text-muted">key</strong> is a
          fingerprint of the bearer token that served the request — never the
          token, and comparable only between rows of this server run.{" "}
          <strong className="font-semibold text-muted">client</strong> is{" "}
          <em>self-declared</em>: this app sends{" "}
          <code className="font-mono">ferrox-studio</code> and any other caller
          could send the same thing, so read it as a label, not as identity.
        </CardFooter>
      </Card>
    </Page>
  );
}
