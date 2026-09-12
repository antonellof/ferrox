import { useMemo, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate, useParams } from "react-router";
import { PanelLeft, Search, SquarePen, Trash2, X } from "lucide-react";
import { SCREENS } from "@/lib/screens";
import { HealthPill } from "@/components/health-pill";
import { FerroxWordmark } from "@/components/logo";
import { useHealth } from "@/lib/use-health";
import {
  recencyBucket,
  useConversationList,
  type RecencyBucket,
} from "@/lib/use-conversations";
import { conversationLabel } from "@/lib/conversations";
import { cn } from "@/lib/utils";

const BUCKETS: RecencyBucket[] = ["Today", "Yesterday", "Previous 7 days", "Older"];

/** The three management screens; Chat is the sidebar's main body, not a row. */
function Nav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Screens" className="flex flex-col gap-0.5">
      {SCREENS.filter((s) => s.to !== "/ui/chat").map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          onClick={onNavigate}
          className={({ isActive }) =>
            cn(
              "group flex items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-sm transition-colors",
              isActive
                ? "bg-inset font-medium text-fg"
                : "text-muted hover:bg-inset/60 hover:text-fg",
            )
          }
        >
          {({ isActive }) => (
            <>
              <Icon
                className={cn(
                  "size-4 shrink-0",
                  isActive ? "text-fg" : "text-faint group-hover:text-muted",
                )}
              />
              <span className="min-w-0 flex-1 truncate">{label}</span>
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}

/** The one action that is not a screen: start over. Lit only on a NEW chat. */
function NewChatLink({ onNavigate }: { onNavigate?: () => void }) {
  const location = useLocation();
  const { conversationId } = useParams();
  const onNewChat = location.pathname.startsWith("/ui/chat") && !conversationId;
  return (
    <NavLink
      to="/ui/chat"
      onClick={onNavigate}
      className={cn(
        "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-colors",
        onNewChat ? "bg-inset font-medium text-fg" : "text-fg hover:bg-inset/60",
      )}
    >
      <SquarePen className="size-4 shrink-0" />
      <span>New chat</span>
    </NavLink>
  );
}

/**
 * The chat library: a search box and every saved
 * conversation grouped by when it was last touched, the way every
 * chat client lays its left rail out. Lives in the shell rather than
 * the chat header so it is there on every screen and never competes
 * with the model picker for the header.
 *
 * Deleting the conversation that is open sends the chat back to a new
 * one: the URL is the only thing that says which conversation is open,
 * so that is one navigation, not a second piece of state.
 */
function ConversationRail({ onNavigate }: { onNavigate?: () => void }) {
  const list = useConversationList();
  // `fetchedAt` is the clock the buckets are drawn against: a rail left
  // open overnight moves "Today" to "Yesterday" on its next refresh
  // rather than on every keystroke.
  const { summaries, fetchedAt: now } = list;
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const [query, setQuery] = useState("");

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const by = new Map<RecencyBucket, typeof summaries>();
    for (const entry of summaries) {
      if (needle && !conversationLabel(entry).toLowerCase().includes(needle)) {
        continue;
      }
      const bucket = recencyBucket(entry.updated_at, now);
      by.set(bucket, [...(by.get(bucket) ?? []), entry]);
    }
    return BUCKETS.filter((b) => by.has(b)).map((b) => ({
      bucket: b,
      entries: by.get(b)!,
    }));
  }, [summaries, query, now]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      {list.mode === "server" ? (
        <>
          <p className="px-2.5 pt-1 text-2xs font-medium tracking-wide text-faint uppercase">
            Chats
          </p>
          <div className="relative px-0.5">
            <Search className="pointer-events-none absolute top-1/2 left-3 size-3.5 -translate-y-1/2 text-faint" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search chats"
              aria-label="Search chats"
              className="h-8 w-full rounded-lg border border-line bg-raised pr-2 pl-8 text-xs text-fg placeholder:text-faint focus:border-fg/30 focus:outline-none"
            />
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto pr-0.5">
            {list.error ? (
              <p className="px-2.5 py-2 text-2xs text-err">{list.error}</p>
            ) : null}
            {!summaries.length ? (
              <p className="px-2.5 py-2 text-2xs text-faint">
                No saved chats yet. One is created the first time you send a
                message.
              </p>
            ) : !groups.length ? (
              <p className="px-2.5 py-2 text-2xs text-faint">
                Nothing matches “{query}”.
              </p>
            ) : (
              groups.map(({ bucket, entries }) => (
                <div key={bucket} className="mb-2">
                  <p className="px-2.5 pt-2 pb-1 text-2xs font-medium text-faint">
                    {bucket}
                  </p>
                  <ul className="space-y-px">
                    {entries.map((entry) => {
                      const isActive = entry.id === conversationId;
                      return (
                        <li key={entry.id} className="group relative">
                          <NavLink
                            to={`/ui/chat/${encodeURIComponent(entry.id)}`}
                            onClick={onNavigate}
                            title={conversationLabel(entry)}
                            className={cn(
                              "block truncate rounded-lg py-1.5 pr-8 pl-2.5 text-sm transition-colors",
                              isActive
                                ? "bg-inset font-medium text-fg"
                                : "text-muted hover:bg-inset/60 hover:text-fg",
                            )}
                          >
                            {conversationLabel(entry)}
                          </NavLink>
                          <button
                            type="button"
                            title="Delete this chat from the server"
                            aria-label={`Delete ${conversationLabel(entry)}`}
                            onClick={async () => {
                              await list.remove(entry.id);
                              if (isActive) navigate("/ui/chat");
                            }}
                            className={cn(
                              "absolute top-1/2 right-1.5 -translate-y-1/2 rounded-md p-1 text-faint transition-opacity hover:bg-raised hover:text-err",
                              "opacity-0 focus:opacity-100 group-hover:opacity-100",
                              isActive && "opacity-100",
                            )}
                          >
                            <Trash2 className="size-3.5" />
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))
            )}
          </div>
        </>
      ) : list.mode === "unsupported" ? (
        <p className="px-2.5 py-2 text-2xs text-faint">
          This server keeps no conversations; the transcript lives in this
          browser.
        </p>
      ) : (
        <div className="flex-1" />
      )}
    </div>
  );
}

export function AppShell() {
  const health = useHealth();
  const [drawer, setDrawer] = useState(false);

  const sidebar = (
    <div className="flex h-full flex-col gap-3 p-3">
      <div className="flex items-center justify-between px-1 pt-1">
        <FerroxWordmark />
        <button
          type="button"
          onClick={() => setDrawer(false)}
          className="rounded-md p-1 text-faint hover:bg-inset hover:text-fg md:hidden"
          aria-label="Close navigation"
        >
          <X className="size-4" />
        </button>
      </div>
      <NewChatLink onNavigate={() => setDrawer(false)} />
      <Nav onNavigate={() => setDrawer(false)} />
      <div className="border-t border-line" />
      <ConversationRail onNavigate={() => setDrawer(false)} />
      <div className="mt-auto border-t border-line pt-3">
        <HealthPill state={health} />
      </div>
    </div>
  );

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-bg">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50 focus:rounded-lg focus:bg-raised focus:px-3 focus:py-2 focus:text-sm focus:shadow-pop"
      >
        Skip to content
      </a>

      {/* Desktop sidebar */}
      <aside className="hidden w-64 shrink-0 border-r border-line bg-sunken md:block">
        {sidebar}
      </aside>

      {/* Mobile drawer */}
      {drawer ? (
        <div className="fixed inset-0 z-40 md:hidden">
          <button
            type="button"
            aria-label="Close navigation"
            className="absolute inset-0 bg-black/40"
            onClick={() => setDrawer(false)}
          />
          <aside className="animate-in slide-in-from-left absolute inset-y-0 left-0 w-72 border-r border-line bg-sunken shadow-pop">
            {sidebar}
          </aside>
        </div>
      ) : null}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b border-line bg-raised/80 px-3 backdrop-blur md:hidden">
          <button
            type="button"
            onClick={() => setDrawer(true)}
            className="rounded-md p-1.5 text-muted hover:bg-inset hover:text-fg"
            aria-label="Open navigation"
          >
            <PanelLeft className="size-4" />
          </button>
          <FerroxWordmark />
        </header>

        <main id="main" tabIndex={-1} className="min-h-0 flex-1 overflow-hidden">
          <Outlet context={health} />
        </main>
      </div>
    </div>
  );
}
