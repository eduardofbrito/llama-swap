// Shared display formatters used across the UI.

export interface FormatDurationOptions {
  /** Number of decimal places when rendering seconds. Default 2. */
  precision?: number;
  /** When true, render durations under 1s as whole milliseconds (e.g. "850ms"). */
  subSecondMs?: boolean;
}

/** Format a millisecond duration as a human-readable string. */
export function formatDuration(ms: number, opts: FormatDurationOptions = {}): string {
  const { precision = 2, subSecondMs = false } = opts;
  if (subSecondMs && ms < 1000) {
    return `${ms.toFixed(0)}ms`;
  }
  return `${(ms / 1000).toFixed(precision)}s`;
}

/** Format a tokens-per-second value; negative values are reported as "unknown". */
export function formatSpeed(speed: number): string {
  return speed < 0 ? "unknown" : speed.toFixed(2) + " t/s";
}

/** Format a byte count as B / KB / MB. */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/** Format a hardware capacity using binary units through TiB. */
export function formatCapacity(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "Not detected";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  const precision = unit < 2 || value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(precision)} ${units[unit]}`;
}

/** Format a timestamp as a local "YYYY-MM-DD HH:mm:ss" string. */
export function formatAbsoluteTime(timestamp: string): string {
  const date = new Date(timestamp);
  const datePart = [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ].join("-");
  const timePart = [date.getHours(), date.getMinutes(), date.getSeconds()]
    .map((value) => String(value).padStart(2, "0"))
    .join(":");
  return `${datePart} ${timePart}`;
}

/** Format a timestamp as a relative time or local timestamp when older than a day. */
export function formatRelativeTime(timestamp: string): string {
  const now = new Date();
  const date = new Date(timestamp);
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  if (diffInSeconds < 5) return "now";
  if (diffInSeconds < 60) return `${diffInSeconds}s ago`;
  const diffInMinutes = Math.floor(diffInSeconds / 60);
  if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
  const diffInHours = Math.floor(diffInMinutes / 60);
  if (diffInHours < 24) return `${diffInHours}h ago`;
  return formatAbsoluteTime(timestamp);
}

/**
 * Renders a GPU's used/total memory as one compact pair sharing a unit, e.g.
 * "18.2 / 23.4 GiB". Both values come from the device in MiB.
 *
 * Returns null when there is no reading to show — the performance monitor is
 * off, or it has not sampled this device yet. That is "unknown", which the
 * caller must render differently from "empty".
 */
export function formatGpuMemory(usedMB?: number, totalMB?: number): string | null {
  if (!Number.isFinite(totalMB ?? NaN) || (totalMB ?? 0) <= 0) return null;
  const total = totalMB as number;
  const used = Number.isFinite(usedMB ?? NaN) ? Math.max(usedMB as number, 0) : 0;
  // GPUs are reported in MiB; anything past a gibibyte reads better in GiB.
  if (total >= 1024) {
    return `${(used / 1024).toFixed(1)} / ${(total / 1024).toFixed(1)} GiB`;
  }
  return `${Math.round(used)} / ${Math.round(total)} MiB`;
}

/**
 * Percentage of a GPU's memory in use, clamped to 0-100. Returns null when
 * there is no reading.
 */
export function gpuMemoryPct(usedMB?: number, totalMB?: number): number | null {
  if (!Number.isFinite(totalMB ?? NaN) || (totalMB ?? 0) <= 0) return null;
  const pct = ((usedMB ?? 0) / (totalMB as number)) * 100;
  return Math.min(Math.max(pct, 0), 100);
}
