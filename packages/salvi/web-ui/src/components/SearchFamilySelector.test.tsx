import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SearchFamilySelector } from "./SearchFamilySelector";

const families = [
  {
    family: "QUALITY_DIVERSITY" as const,
    title: "Quality diversity",
    description: "Optimize local trade-offs in descriptor regions.",
    order: 10,
    default_engine: "serial_mome"
  },
  {
    family: "CONVENTIONAL_MULTI_OBJECTIVE" as const,
    title: "Conventional multi-objective",
    description: "Optimize one global Pareto population.",
    order: 20,
    default_engine: "pymoo_nsga2"
  }
];

describe("SearchFamilySelector", () => {
  it("renders catalog descriptions and delegates family changes", () => {
    const onSelect = vi.fn();
    render(
      <SearchFamilySelector
        families={families}
        selected="QUALITY_DIVERSITY"
        busy={false}
        onSelect={onSelect}
      />
    );

    expect(screen.getByRole("radio", { name: /Quality diversity/i })).toHaveAttribute(
      "aria-checked",
      "true"
    );
    fireEvent.click(screen.getByRole("radio", { name: /Conventional multi-objective/i }));
    expect(onSelect).toHaveBeenCalledWith("CONVENTIONAL_MULTI_OBJECTIVE");
  });

  it("prevents transitions while another family is loading", () => {
    render(
      <SearchFamilySelector
        families={families}
        selected="QUALITY_DIVERSITY"
        busy
        onSelect={vi.fn()}
      />
    );
    expect(screen.getAllByRole("radio")).toEqual(
      expect.arrayContaining([expect.objectContaining({ disabled: true })])
    );
  });
});
