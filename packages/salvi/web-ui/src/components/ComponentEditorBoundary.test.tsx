import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ComponentEditorBoundary } from "./ComponentEditorBoundary";

function BrokenEditor(): never {
  throw new Error("invalid component schema");
}

describe("ComponentEditorBoundary", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("keeps a component failure local to the drawer", () => {
    render(
      <main>
        <h1>Pipeline remains visible</h1>
        <ComponentEditorBoundary>
          <BrokenEditor />
        </ComponentEditorBoundary>
      </main>
    );

    expect(screen.getByRole("heading", { name: "Pipeline remains visible" })).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("invalid component schema");
  });
});
