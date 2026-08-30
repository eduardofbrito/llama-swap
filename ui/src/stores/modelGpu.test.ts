import { describe, expect, it } from "vitest";
import { defaultGpuDisplay } from "./modelGpu";

describe("defaultGpuDisplay", () => {
  it("shows the configured GPU value prefixed with 'default'", () => {
    expect(defaultGpuDisplay("0")).toBe("default 0");
    expect(defaultGpuDisplay("3")).toBe("default 3");
  });

  it("shows multi-GPU comma lists prefixed with 'default'", () => {
    expect(defaultGpuDisplay("0,1,3")).toBe("default 0,1,3");
  });

  it("trims surrounding whitespace before formatting", () => {
    expect(defaultGpuDisplay(" 4 ")).toBe("default 4");
  });

  it('falls back to "default" when the model pins no GPU', () => {
    expect(defaultGpuDisplay(undefined)).toBe("default");
    expect(defaultGpuDisplay("")).toBe("default");
    expect(defaultGpuDisplay("   ")).toBe("default");
  });
});
