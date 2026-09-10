import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// One accent-filled button per view, at most: the accent marks THE action
// on the screen, not every action. A row of twelve "Load" buttons is
// twelve neutral buttons — see the Models table, which used to be a wall
// of orange and read as an alarm rather than as an inventory.
//
// Heights are on the 4px grid and paired with a type step: sm 28/xs,
// md 32/sm, lg 36/sm. `md` is the default because chrome is dense here.

const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-1.5 rounded-md font-medium whitespace-nowrap transition-[background-color,color,border-color] disabled:pointer-events-none disabled:opacity-45 [&_svg]:pointer-events-none [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        primary: "bg-accent text-accent-fg hover:bg-accent-hover",
        default:
          "border border-line bg-raised text-fg hover:border-line-strong hover:bg-inset",
        ghost: "text-muted hover:bg-inset hover:text-fg",
        danger: "border border-err/35 bg-err-soft text-err hover:border-err/70",
        link: "text-accent underline-offset-4 hover:underline",
      },
      size: {
        sm: "h-7 gap-1 px-2.5 text-xs [&_svg]:size-3.5",
        md: "h-8 px-3 text-sm [&_svg]:size-4",
        lg: "h-9 px-4 text-sm [&_svg]:size-4",
        icon: "size-8 [&_svg]:size-4",
        iconSm: "size-7 [&_svg]:size-3.5",
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
