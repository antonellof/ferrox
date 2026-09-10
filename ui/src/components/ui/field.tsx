import type * as React from "react";
import * as LabelPrimitive from "@radix-ui/react-label";
import { cn } from "@/lib/utils";

// One control string, three controls. The hover/focus/disabled states
// live here and nowhere else, so an input and a textarea cannot come to
// disagree about what "focused" looks like.
const control =
  "w-full rounded-md border border-line bg-raised px-2.5 py-1.5 text-sm text-fg transition-colors placeholder:text-faint hover:border-line-strong focus:border-accent focus:outline-none focus-visible:outline-2 focus-visible:outline-accent focus-visible:outline-offset-1 disabled:cursor-not-allowed disabled:opacity-50";

export function Input({ className, ...props }: React.ComponentProps<"input">) {
  return <input className={cn(control, "h-8", className)} {...props} />;
}

export function Textarea({
  className,
  ...props
}: React.ComponentProps<"textarea">) {
  return (
    <textarea
      className={cn(control, "resize-y leading-relaxed", className)}
      {...props}
    />
  );
}

export function Label({
  className,
  ...props
}: React.ComponentProps<typeof LabelPrimitive.Root>) {
  return (
    <LabelPrimitive.Root
      className={cn("text-xs font-medium text-muted select-none", className)}
      {...props}
    />
  );
}

/** Label stacked above its control — the layout every form here uses. */
export function Field({
  label,
  hint,
  htmlFor,
  className,
  children,
}: {
  label: React.ReactNode;
  hint?: React.ReactNode;
  htmlFor?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-1.5", className)}>
      <Label htmlFor={htmlFor}>{label}</Label>
      {children}
      {hint ? (
        <p className="text-2xs leading-relaxed text-faint">{hint}</p>
      ) : null}
    </div>
  );
}
