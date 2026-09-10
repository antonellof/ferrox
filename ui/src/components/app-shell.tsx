import { useState } from "react";
import { NavLink, Outlet } from "react-router";
import { PanelLeft, X } from "lucide-react";
import { SCREENS } from "@/lib/screens";
import { HealthPill } from "@/components/health-pill";
import { useHealth } from "@/lib/use-health";
import { cn } from "@/lib/utils";

function Brand() {
  return (
    <div className="flex min-w-0 items-center gap-2">
      <span
        aria-hidden
        className="grid size-6 shrink-0 place-items-center rounded-md bg-accent text-2xs font-semibold text-accent-fg"
      >
        Fe
      </span>
      <span className="truncate text-sm font-medium tracking-tight">
        Ferrox Studio
      </span>
    </div>
  );
}

/**
 * The nav rows.
 *
 * Active is a NEUTRAL fill plus a weight bump, not an accent-tinted
 * block. Four permanently-visible rows is the wrong place to spend the
 * one accent colour this app has: on a screen where nothing is wrong,
 * the loudest thing on it was "which tab am I on", which the user
 * already knows. The accent now marks actions, not location.
 */
function Nav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Screens" className="flex flex-col gap-0.5">
      {SCREENS.map(({ to, label, icon: Icon, blurb }) => (
        <NavLink
          key={to}
          to={to}
          onClick={onNavigate}
          className={({ isActive }) =>
            cn(
              "group flex h-8 items-center gap-2 rounded-md px-2 text-sm transition-colors",
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
              <span className="hidden text-2xs text-faint lg:group-hover:inline">
                {blurb}
              </span>
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}

export function AppShell() {
  const health = useHealth();
  const [drawer, setDrawer] = useState(false);

  // Chrome runs on the 8px grid: `p-2` around, `gap-1` between rows.
  const sidebar = (
    <div className="flex h-full flex-col gap-4 p-2">
      <div className="flex h-8 items-center justify-between gap-2 px-1">
        <Brand />
        <button
          type="button"
          onClick={() => setDrawer(false)}
          className="rounded-md p-1 text-faint transition-colors hover:bg-inset hover:text-fg md:hidden"
          aria-label="Close navigation"
        >
          <X className="size-4" />
        </button>
      </div>
      <Nav onNavigate={() => setDrawer(false)} />
      <div className="mt-auto">
        <HealthPill state={health} />
      </div>
    </div>
  );

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-bg">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50 focus:rounded-md focus:border focus:border-line focus:bg-raised focus:px-3 focus:py-2 focus:text-sm focus:shadow-pop"
      >
        Skip to content
      </a>

      {/* Desktop sidebar */}
      <aside className="hidden w-60 shrink-0 border-r border-line bg-sunken md:block">
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
          <aside className="animate-in slide-in-from-left absolute inset-y-0 left-0 w-64 border-r border-line bg-sunken shadow-pop">
            {sidebar}
          </aside>
        </div>
      ) : null}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b border-line bg-raised px-2 md:hidden">
          <button
            type="button"
            onClick={() => setDrawer(true)}
            className="rounded-md p-1.5 text-muted transition-colors hover:bg-inset hover:text-fg"
            aria-label="Open navigation"
          >
            <PanelLeft className="size-4" />
          </button>
          <Brand />
        </header>

        <main
          id="main"
          tabIndex={-1}
          className="min-h-0 flex-1 overflow-hidden"
        >
          <Outlet context={health} />
        </main>
      </div>
    </div>
  );
}
