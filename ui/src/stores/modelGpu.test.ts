import { describe, expect, it } from "vitest";
import { defaultGpuDisplay } from "./modelGpu";

describe("defaultGpuDisplay", () => {
  it("shows the configured GPU value", () => {
    expect(defaultGpuDisplay("0")).toBe("0");
    expect(defaultGpuDisplay("3")).toBe("3");
  });

  it("shows multi-GPU comma lists verbatim", () => {
    expect(defaultGpuDisplay("0,1,3")).toBe("0,1,3");
  });

  it("trims surrounding whitespace", () => {
    expect(defaultGpuDisplay(" 4 ")).toBe("4");
  });

  it('falls back to "default" when the model pins no GPU', () => {
    expect(defaultGpuDisplay(undefined)).toBe("default");
    expect(defaultGpuDisplay("")).toBe("default");
    expect(defaultGpuDisplay("   ")).toBe("default");
  });
});
