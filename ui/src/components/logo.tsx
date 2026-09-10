import {
  CELL_PATH,
  CORE_R,
  SPOKES_PATH,
  STROKE_WIDTH,
  VIEWBOX,
} from "@/lib/logo-geometry";
import { cn } from "@/lib/utils";

/**
 * The Ferrox mark. See `lib/logo-geometry.ts` for what it draws and why.
 *
 * It is inline SVG on `currentColor` rather than an image, so one file
 * serves the sidebar, the assistant avatar, the empty state and the
 * favicon, in both themes, with no second asset to keep in step.
 */
export function FerroxMark({
  className,
  title,
}: {
  className?: string;
  /** Give it one only where the mark is the sole naming of something. */
  title?: string;
}) {
  return (
    <svg
      viewBox={VIEWBOX}
      className={cn("shrink-0", className)}
      fill="none"
      stroke="currentColor"
      strokeWidth={STROKE_WIDTH}
      strokeLinejoin="round"
      strokeLinecap="round"
      role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true}
    >
      {title ? <title>{title}</title> : null}
      <path d={CELL_PATH} />
      <path d={SPOKES_PATH} />
      <circle cx="12" cy="12" r={CORE_R} fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Mark plus name. The weight split is the whole lockup: no hue does it. */
export function FerroxWordmark({ className }: { className?: string }) {
  return (
    <div className={cn("flex min-w-0 items-center gap-2", className)}>
      <FerroxMark className="size-5 text-fg" />
      <span className="truncate text-sm tracking-tight">
        <span className="font-semibold">Ferrox</span>{" "}
        <span className="text-faint">Studio</span>
      </span>
    </div>
  );
}
