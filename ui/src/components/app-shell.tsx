import { useState } from "react";
import { NavLink, Outlet } from "react-router";
import { PanelLeft, X } from "lucide-react";
import { SCREENS } from "@/lib/screens";
import { HealthPill } from "@/components/health-pill";
import { FerroxWordmark } from "@/components/logo";
import { useHealth } from "@/lib/use-health";
import { cn } from "@/lib/utils";

function Nav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Screens" className="flex flex-col gap-0.5">
      {SCREENS.map(({ to, label, icon: Icon, blurb }) => (
        <NavLink
          key={to}
          to={to}
          onClick={onNavigate}
          // Where you are is not news: you clicked it. A neutral fill
          // plus a weight bump says it without making the loudest thing
          // on a screen where nothing is wrong be "which tab am I on".
          className={({ isActive }) =>
            cn(
              "group flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-colors",
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

  const sidebar = (
    <div className="flex h-full flex-col gap-4 p-3">
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
      <Nav onNavigate={() => setDrawer(false)} />
      <div className="mt-auto space-y-2">
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
