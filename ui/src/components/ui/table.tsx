import type * as React from "react";
import { cn } from "@/lib/utils";

/** Wide tables scroll inside their own box; the page never scrolls sideways. */
export function TableScroll({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      className={cn("w-full overflow-x-auto overscroll-x-contain", className)}
      {...props}
    />
  );
}

export function Table({ className, ...props }: React.ComponentProps<"table">) {
  return (
    <table
      className={cn("w-full border-collapse text-sm", className)}
      {...props}
    />
  );
}

/* A table inside a card shares the card's 16px gutter: the first column
 * lines up with the card title above it and the last with its right
 * edge. Cells are px-3 between those, which is where the density comes
 * from — dense but breathing, rather than dense and cramped. */
const gutter = "px-3 first:pl-4 last:pr-4";

export function Th({
  className,
  numeric,
  ...props
}: React.ComponentProps<"th"> & { numeric?: boolean }) {
  return (
    <th
      scope="col"
      className={cn(
        // A wrapped header doubles the height of every row under it, so
        // a narrow window scrolls the table instead of reflowing it.
        "sticky top-0 z-10 border-b border-line bg-raised py-2 text-left text-2xs font-medium tracking-wide whitespace-nowrap text-faint uppercase",
        gutter,
        numeric && "text-right tabular-nums",
        className,
      )}
      {...props}
    />
  );
}

export function Td({
  className,
  numeric,
  mono,
  ...props
}: React.ComponentProps<"td"> & { numeric?: boolean; mono?: boolean }) {
  return (
    <td
      className={cn(
        // `w-full` plus auto layout means a table too wide for its box
        // WRAPS its cells rather than overflowing — "1.04 GB" breaking
        // across two lines and doubling every row. Nowrap makes it
        // overflow instead, which is what `TableScroll` is for.
        "border-b border-line py-2 align-middle whitespace-nowrap",
        gutter,
        numeric && "text-right tabular-nums",
        mono && "font-mono text-code",
        className,
      )}
      {...props}
    />
  );
}

export function Tr({ className, ...props }: React.ComponentProps<"tr">) {
  return (
    <tr
      className={cn(
        "transition-colors last:[&>td]:border-b-0 hover:bg-inset/50",
        className,
      )}
      {...props}
    />
  );
}
