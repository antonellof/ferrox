import type * as React from "react";
import { cn } from "@/lib/utils";

// A card is a hairline and a surface, never a shadow. See the ELEVATION
// rule in `index.css`: what floats gets `shadow-pop`, and a card does
// not float — it sits on the page beside its neighbours.

export function Card({ className, ...props }: React.ComponentProps<"section">) {
  return (
    <section
      className={cn("rounded-lg border border-line bg-raised", className)}
      {...props}
    />
  );
}

export function CardHeader({
  className,
  ...props
}: React.ComponentProps<"header">) {
  return (
    <header
      className={cn(
        "flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-line px-4 py-3",
        className,
      )}
      {...props}
    />
  );
}

export function CardTitle({ className, ...props }: React.ComponentProps<"h2">) {
  return (
    <h2
      className={cn("text-sm font-semibold tracking-tight", className)}
      {...props}
    />
  );
}

export function CardDescription({
  className,
  ...props
}: React.ComponentProps<"p">) {
  return <p className={cn("text-xs text-muted", className)} {...props} />;
}

export function CardBody({ className, ...props }: React.ComponentProps<"div">) {
  return <div className={cn("p-4", className)} {...props} />;
}

/** The quiet strip under a card that explains what the card just showed. */
export function CardFooter({
  className,
  ...props
}: React.ComponentProps<"footer">) {
  return (
    <footer
      className={cn(
        "rounded-b-lg border-t border-line bg-sunken/70 px-4 py-3 text-xs leading-relaxed text-faint",
        className,
      )}
      {...props}
    />
  );
}
