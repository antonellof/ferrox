import type * as React from "react";
import { cn } from "@/lib/utils";

/** The header every screen shares: title, one sentence, right-hand actions. */
export function PageHeader({
  title,
  description,
  actions,
  className,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-start justify-between gap-x-4 gap-y-3",
        className,
      )}
    >
      <div className="min-w-0 space-y-1">
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        {description ? (
          <p className="max-w-2xl text-xs leading-relaxed text-muted">
            {description}
          </p>
        ) : null}
      </div>
      {actions ? (
        <div className="flex flex-wrap items-center gap-2">{actions}</div>
      ) : null}
    </div>
  );
}

/**
 * A label over a group of tiles that is not itself a card.
 *
 * The counters and trend lines on Activity used to be wrapped in cards
 * purely to get a title. This gives them the title without the box.
 */
export function SectionLabel({
  className,
  ...props
}: React.ComponentProps<"h2">) {
  return (
    <h2
      className={cn("mb-2 text-xs font-medium text-muted", className)}
      {...props}
    />
  );
}

/** A scrolling screen body with the standard gutters and max width. */
export function Page({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div className="h-full overflow-y-auto">
      <div
        className={cn(
          "mx-auto flex w-full max-w-6xl flex-col gap-4 p-4 md:gap-6 md:p-6",
          className,
        )}
        {...props}
      />
    </div>
  );
}
