import { describe, it, expect, vi, afterEach } from "vitest";
import {
  formatDuration,
  formatSpeed,
  formatFileSize,
  formatCapacity,
  formatGpuMemory,
  gpuMemoryPct,
  formatRelativeTime,
  formatAbsoluteTime,
} from "./format";

describe("formatDuration", () => {
  it("defaults to seconds with 2 decimals", () => {
    expect(formatDuration(1500)).toBe("1.50s");
    expect(formatDuration(0)).toBe("0.00s");
    expect(formatDuration(250)).toBe("0.25s");
  });

  it("honors custom precision", () => {
    expect(formatDuration(1500, { precision: 1 })).toBe("1.5s");
  });

  it("renders sub-second durations as ms when subSecondMs is set", () => {
    expect(formatDuration(850, { precision: 1, subSecondMs: true })).toBe("850ms");
    expect(formatDuration(1500, { precision: 1, subSecondMs: true })).toBe("1.5s");
    expect(formatDuration(999, { subSecondMs: true })).toBe("999ms");
  });
});

describe("formatSpeed", () => {
  it("formats tokens per second", () => {
    expect(formatSpeed(42.5)).toBe("42.50 t/s");
    expect(formatSpeed(0)).toBe("0.00 t/s");
  });

  it("reports negative values as unknown", () => {
    expect(formatSpeed(-1)).toBe("unknown");
  });
});

describe("formatFileSize", () => {
  it("formats bytes", () => {
    expect(formatFileSize(512)).toBe("512 B");
  });

  it("formats kilobytes", () => {
    expect(formatFileSize(2048)).toBe("2.0 KB");
  });

  it("formats megabytes", () => {
    expect(formatFileSize(5 * 1024 * 1024)).toBe("5.0 MB");
  });
});

describe("formatCapacity", () => {
  it("formats binary hardware capacities", () => {
    expect(formatCapacity(24 * 1024 ** 3)).toBe("24.0 GiB");
    expect(formatCapacity(1.5 * 1024 ** 4)).toBe("1.50 TiB");
  });

  it("handles unavailable capacities", () => {
    expect(formatCapacity(0)).toBe("Not detected");
  });
});

describe("formatRelativeTime", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("formats relative times", () => {
    const now = new Date("2026-06-28T12:00:00Z");
    vi.useFakeTimers();
    vi.setSystemTime(now);

    expect(formatRelativeTime("2026-06-28T11:59:58Z")).toBe("now");
    expect(formatRelativeTime("2026-06-28T11:59:30Z")).toBe("30s ago");
    expect(formatRelativeTime("2026-06-28T11:55:00Z")).toBe("5m ago");
    expect(formatRelativeTime("2026-06-28T09:00:00Z")).toBe("3h ago");
    const olderThanOneDay = new Date(2026, 5, 25, 12, 34, 56);
    expect(formatRelativeTime(olderThanOneDay.toISOString())).toBe("2026-06-25 12:34:56");
  });
});

describe("formatAbsoluteTime", () => {
  it("formats a timestamp in local time", () => {
    expect(formatAbsoluteTime(new Date(2026, 0, 2, 3, 4, 5).toISOString())).toBe(
      "2026-01-02 03:04:05"
    );
  });

  it("is unaffected by how recent the timestamp is", () => {
    expect(formatAbsoluteTime(new Date(2026, 11, 31, 23, 59, 59).toISOString())).toBe(
      "2026-12-31 23:59:59"
    );
  });
});

describe("formatGpuMemory", () => {
  it("renders a used/total pair sharing one unit", () => {
    expect(formatGpuMemory(18227, 24564)).toBe("17.8 / 24.0 GiB");
  });

  it("stays in MiB for small devices", () => {
    expect(formatGpuMemory(100, 512)).toBe("100 / 512 MiB");
  });

  it("treats a missing used value as zero", () => {
    expect(formatGpuMemory(undefined, 24564)).toBe("0.0 / 24.0 GiB");
  });

  it("returns null when there is no reading, which is not the same as empty", () => {
    expect(formatGpuMemory(0, 0)).toBeNull();
    expect(formatGpuMemory(1000, undefined)).toBeNull();
    expect(formatGpuMemory(undefined, undefined)).toBeNull();
  });
});

describe("gpuMemoryPct", () => {
  it("computes the percentage in use", () => {
    expect(gpuMemoryPct(12000, 24000)).toBe(50);
  });

  it("clamps out-of-range readings", () => {
    expect(gpuMemoryPct(30000, 24000)).toBe(100);
    expect(gpuMemoryPct(-5, 24000)).toBe(0);
  });

  it("returns null without a reading", () => {
    expect(gpuMemoryPct(1000, 0)).toBeNull();
  });
});
