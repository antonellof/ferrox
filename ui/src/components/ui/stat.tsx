import type * as React from "react";
import { cn } from "@/lib/utils";

/**
 * One labelled tile, for a counter or for a trend line.
 *
 * Activity had two of these written out separately — a counter box drawn
 * inside a `Card`, and a sparkline drawn inside a different `Card` — with
 * two spellings of the same uppercase micro-label and two different
 * paddings. Two structures that had to agree about one thing, with
 * nothing enforcing it. This is the one thing.
 *
 * It is a top-level tile, not a card body: a bordered box inside a
 * bordered box is the shape that makes a dashboard look busy, and the
 * grid of tiles reads as a row of facts on its own.
 */
export function StatTile({
  label,
  hint,
  className,
  children,
}: {
  label: React.ReactNode;
  /** Why this number means what it means, on hover. */
  hint?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      title={hint}
      className={cn(
        "rounded-lg border border-line bg-raised px-3 py-2.5",
        className,
      )}
    >
      <p className="text-2xs font-medium tracking-wide text-faint uppercase">
        {label}
      </p>
      <div className="mt-1.5">{children}</div>
    </div>
  );
}

/** The number in a `StatTile`. Tabular so a ticking counter cannot jitter. */
export function StatValue({ className, ...props }: React.ComponentProps<"p">) {
  return (
    <p
      className={cn(
        "font-mono text-lg leading-none tabular-nums text-fg",
        className,
      )}
      {...props}
    />
  );
}
