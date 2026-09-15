import { describe, expect, it } from "vitest";
import { statusDotColor } from "./modelLoad";
import type { Model, ModelStatus } from "../lib/types";

function model(state: ModelStatus): Model {
  return { id: "m", state, name: "", description: "", unlisted: false } as Model;
}

describe("statusDotColor", () => {
  // A sleeping model is alive, parked, and one request away from serving.
  // Showing it in the same grey as a model that was never loaded hides the
  // whole feature: the operator cannot tell a warm model from a cold one.
  // This lived in three hand-copied versions and the sidebar's copy never
  // learned about sleeping, so the dot there stayed grey.
  it("gives every state its own colour", () => {
    expect(statusDotColor(model("ready"))).toBe("bg-success");
    expect(statusDotColor(model("sleeping"))).toBe("bg-warning");
    expect(statusDotColor(model("starting"))).toBe("bg-primary");
    expect(statusDotColor(model("stopping"))).toBe("bg-destructive");
  });

  it("greys out only what is genuinely not there", () => {
    const grey = "bg-muted-foreground/40";
    expect(statusDotColor(model("stopped"))).toBe(grey);
    expect(statusDotColor(model("shutdown"))).toBe(grey);
    expect(statusDotColor(model("unknown"))).toBe(grey);
    expect(statusDotColor(undefined)).toBe(grey);
  });

  it("does not paint a sleeping model the same as a stopped one", () => {
    expect(statusDotColor(model("sleeping"))).not.toBe(statusDotColor(model("stopped")));
  });
});
