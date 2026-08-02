import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ParameterEditor } from "./ParameterEditor";
import type { ComponentDescription } from "../types";

const component: ComponentDescription = {
  kind: "observer",
  name: "example",
  title: "Example",
  description: "Example component.",
  provides: [],
  requires: [],
  supported_patterns: [],
  conflicts: [],
  compatibility_notes: [],
  maturity: "STABLE",
  stage: "OUTPUT",
  order: 1,
  search_family: null,
  default_for_search_family: false,
  observer_view: null,
  parameters: [
    {
      name: "enabled",
      title: "Enabled",
      description: "Whether this setting is active.",
      required: true,
      default: true,
      value_schema: { type: "boolean" },
      applicable_patterns: [],
      widget: "BOOLEAN",
      unit: null,
      advanced: false
    }
  ]
};

describe("ParameterEditor", () => {
  it("edits catalog-backed primitive parameters", async () => {
    const onChange = vi.fn();
    render(
      <ParameterEditor
        component={component}
        parameters={{ enabled: true, count: 4 }}
        onChange={onChange}
      />
    );

    fireEvent.click(screen.getByRole("switch"));
    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ enabled: false }));
    });
  });
});
