import { cn } from "@/lib/utils";

/**
 * A determinate bar when the server knows the fraction, an indeterminate
 * sweep when it does not. There is no third case where this component
 * computes a fraction of its own.
 */
export function Progress({
  fraction,
  className,
  label,
}: {
  fraction: number | null;
  className?: string;
  label?: string;
}) {
  const pct = fraction === null ? null : Math.max(0, Math.min(1, fraction)) * 100;
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct === null ? undefined : Math.round(pct)}
      className={cn(
        "h-1.5 w-full overflow-hidden rounded-full bg-inset",
        className,
      )}
    >
      {pct === null ? (
        <div className="h-full w-1/3 animate-[indeterminate_1.4s_ease-in-out_infinite] rounded-full bg-ink/70" />
      ) : (
        <div
          className="h-full rounded-full bg-ink transition-[width] duration-500 ease-out"
          style={{ width: `${pct.toFixed(1)}%` }}
        />
      )}
    </div>
  );
}
