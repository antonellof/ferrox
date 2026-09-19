// Formatting.
//
// Every one of these answers "unknown" as an em dash rather than as a
// zero. The API is deliberate about `null` meaning "not established"
// (see frink-api's admin module); printing `0 B/s` for a rate the
// server refused to estimate would undo that on the last hop.

export const UNKNOWN = "—";

export function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function fmtInt(v: unknown): string {
  return isNum(v) ? Math.round(v).toLocaleString() : UNKNOWN;
}

export function fmtNum(v: unknown, digits = 1): string {
  return isNum(v) ? v.toFixed(digits) : UNKNOWN;
}

export function fmtBytes(v: unknown): string {
  if (!isNum(v)) return UNKNOWN;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let n = v;
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${n.toFixed(i === 0 ? 0 : n < 10 ? 2 : 1)} ${units[i]}`;
}

export function fmtRate(bytesPerSecond: unknown): string {
  return isNum(bytesPerSecond) ? `${fmtBytes(bytesPerSecond)}/s` : UNKNOWN;
}

export function fmtMs(v: unknown): string {
  if (!isNum(v)) return UNKNOWN;
  if (v < 1000) return `${v.toFixed(v < 10 ? 1 : 0)} ms`;
  return `${(v / 1000).toFixed(2)} s`;
}

export function fmtDuration(seconds: unknown): string {
  if (!isNum(seconds)) return UNKNOWN;
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  const h = Math.floor(s / 3600);
  return `${h}h ${Math.floor((s % 3600) / 60)}m`;
}

export function fmtParams(count: unknown): string {
  if (!isNum(count)) return UNKNOWN;
  if (count >= 1e12) return `${(count / 1e12).toFixed(2)}T`;
  if (count >= 1e9) return `${(count / 1e9).toFixed(count < 1e10 ? 2 : 1)}B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(0)}M`;
  return fmtInt(count);
}

/** Wall-clock time of a server-supplied epoch-millisecond stamp. */
export function fmtClock(unixMs: unknown): string {
  if (!isNum(unixMs)) return UNKNOWN;
  const d = new Date(unixMs);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
