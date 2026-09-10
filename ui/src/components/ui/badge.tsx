import type * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// Tinted fill, no border: a badge sits INSIDE something that already has
// a hairline, and a second outline 3px in reads as a box in a box.

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-2xs font-medium whitespace-nowrap",
  {
    variants: {
      tone: {
        neutral: "bg-inset text-muted",
        accent: "bg-accent-soft text-accent",
        ok: "bg-ok-soft text-ok",
        warn: "bg-warn-soft text-warn",
        err: "bg-err-soft text-err",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export type BadgeProps = React.ComponentProps<"span"> &
  VariantProps<typeof badgeVariants>;

export function Badge({ className, tone, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}
