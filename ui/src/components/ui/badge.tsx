import type * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-2xs font-medium leading-4 whitespace-nowrap",
  {
    variants: {
      // `ok`/`warn`/`err` are the only tones carrying a hue, and the hue
      // is the whole reason they exist. "In progress" is not a verdict,
      // so it gets contrast rather than a colour — a running download in
      // a brand hue reads as an alarm in a column of them.
      tone: {
        neutral: "border-line bg-inset text-muted",
        strong: "border-line-strong bg-inset text-fg",
        ok: "border-ok/35 bg-ok-soft text-ok",
        warn: "border-warn/35 bg-warn-soft text-warn",
        err: "border-err/35 bg-err-soft text-err",
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
