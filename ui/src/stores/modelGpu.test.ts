import { describe, expect, it } from "vitest";
import { defaultGpuDisplay } from "./modelGpu";

describe("defaultGpuDisplay", () => {
  it("shows the configured GPU value prefixed with 'GPU Default'", () => {
    expect(defaultGpuDisplay("0")).toBe("GPU Default 0");
    expect(defaultGpuDisplay("3")).toBe("GPU Default 3");
  });

  it("shows multi-GPU comma lists prefixed with 'GPU Default'", () => {
    expect(defaultGpuDisplay("0,1,3")).toBe("GPU Default 0,1,3");
  });

  it("trims surrounding whitespace before formatting", () => {
    expect(defaultGpuDisplay(" 4 ")).toBe("GPU Default 4");
  });

  it('falls back to "GPU Default" when the model pins no GPU', () => {
    expect(defaultGpuDisplay(undefined)).toBe("GPU Default");
    expect(defaultGpuDisplay("")).toBe("GPU Default");
    expect(defaultGpuDisplay("   ")).toBe("GPU Default");
  });
});
