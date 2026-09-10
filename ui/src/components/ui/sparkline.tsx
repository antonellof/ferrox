import { useId } from "react";
import { cn } from "@/lib/utils";

/**
 * A trend line over a handful of points.
 *
 * Plain SVG on purpose — a chart library would be more code shipped in
 * the binary than the whole rest of this screen. Points that are `null`
 * are dropped rather than drawn as zero: a request the engine did not
 * time is missing data, and a line that dips to the axis for it would
 * read as "it got fast", which is the opposite of true.
 */
export function Sparkline({
  values,
  className,
  label,
}: {
  values: (number | null)[];
  className?: string;
  label: string;
}) {
  const id = useId();
  const points = values.filter((v): v is number => typeof v === "number");
  if (points.length < 2) {
    return (
      <div
        className={cn("h-8 w-full rounded bg-inset/60", className)}
        aria-label={`${label}: not enough samples`}
      />
    );
  }

  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const w = 100;
  const h = 28;
  const step = w / (points.length - 1);
  const coords = points.map(
    (v, i) => [i * step, h - 2 - ((v - min) / span) * (h - 4)] as const,
  );
  const path = coords
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`)
    .join(" ");
  const area = `${path} L${w},${h} L0,${h} Z`;
  const last = coords[coords.length - 1];

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={`${label}: ${points.length} samples, ${min.toFixed(1)} to ${max.toFixed(1)}`}
      className={cn("h-8 w-full overflow-visible text-fg", className)}
    >
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.25" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${id})`} />
      <path
        d={path}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
      <circle cx={last[0]} cy={last[1]} r="2" fill="currentColor" />
    </svg>
  );
}
