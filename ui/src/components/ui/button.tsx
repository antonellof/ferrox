import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// The radius lives on the SIZE, not on the base, because `round` is a
// size whose whole point is a different radius. Two radius utilities on
// one element resolve by stylesheet order rather than class order, so
// "the later class wins" would be a coin flip; only one is ever emitted.
const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap font-medium transition-[background,color,border-color] disabled:pointer-events-none disabled:opacity-45 [&_svg]:pointer-events-none [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        // Ink. With no brand hue this is the loudest thing the app can
        // draw, so it is spent on ONE action per view and nothing else.
        primary: "bg-ink text-on-ink hover:bg-ink-hover",
        default:
          "border border-line bg-raised text-fg hover:bg-inset hover:border-line-strong",
        ghost: "text-muted hover:bg-inset hover:text-fg",
        danger: "border border-err/40 bg-err-soft text-err hover:border-err/70",
        // A link has no hue either; the underline is what marks it.
        link: "text-fg underline decoration-line-strong underline-offset-4 hover:decoration-fg",
      },
      size: {
        sm: "h-7 rounded-md px-2.5 text-xs [&_svg]:size-3.5",
        md: "h-9 rounded-lg px-3.5 text-sm [&_svg]:size-4",
        lg: "h-10 rounded-lg px-4 text-sm [&_svg]:size-4",
        icon: "size-9 rounded-lg [&_svg]:size-4",
        /** The composer's send / stop. A circle, and the only one. */
        round: "size-9 rounded-full [&_svg]:size-4",
        iconSm: "size-7 rounded-md [&_svg]:size-3.5",
      },
    },
    defaultVariants: { variant: "default", size: "md" },
  },
);

export type ButtonProps = React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & { asChild?: boolean };

export function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: ButtonProps) {
  const Comp = asChild ? Slot : "button";
  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  );
}
