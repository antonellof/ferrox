import type * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Wide tables scroll inside their own box; the page never scrolls sideways.
 *
 * Every cell below is `whitespace-nowrap`, and that is what makes this box
 * work at all. A `w-full` table with auto layout does not overflow when it
 * is too wide — it WRAPS, so at 430px `708.7 MB` broke across two lines
 * and every row in the inventory was 57px instead of 35px, with nothing to
 * scroll. Let the cells refuse to wrap and the overflow lands here.
 */
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

export function Th({
  className,
  numeric,
  ...props
}: React.ComponentProps<"th"> & { numeric?: boolean }) {
  return (
    <th
      scope="col"
      className={cn(
        "sticky top-0 z-10 border-b border-line bg-raised px-3 py-2 text-left text-2xs font-semibold tracking-wide whitespace-nowrap text-faint uppercase",
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
        "border-b border-line/70 px-3 py-2 align-middle whitespace-nowrap",
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
      className={cn("transition-colors hover:bg-inset/60", className)}
      {...props}
    />
  );
}
