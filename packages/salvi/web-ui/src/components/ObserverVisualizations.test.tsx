import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Metric, ObserverView } from "../types";
import { ObserverMetricPanel } from "./ObserverVisualizations";

vi.mock("./Chart", () => ({
  Chart: ({ option, height }: { option: unknown; height: number }) => (
    <div data-height={height} data-option={JSON.stringify(option)} data-testid="chart" />
  )
}));

afterEach(cleanup);

const view: ObserverView = {
  view_kind: "QD_DIAGNOSTICS",
  title: "QD archive diagnostics",
  metric_patterns: ["qd.*"],
  empty_message: "No diagnostics yet.",
  x_axis_label: "First descriptor bin",
  y_axis_label: "Second descriptor bin",
  metrics: [
    {
      pattern: "qd.visited_cells",
      label: "Visited cells",
      description: "Visited descriptor cells.",
      unit: "cells",
      value_kind: "COUNTER",
      temporal_scope: "CUMULATIVE",
      population: "QD_CELLS",
      display_group: "coverage"
    },
    {
      pattern: "qd.unmapped_attempts",
      label: "Unmapped attempts",
      description: "Attempts without a descriptor cell.",
      unit: "candidates",
      value_kind: "COUNTER",
      temporal_scope: "CUMULATIVE",
      population: "ARCHIVE_DECISIONS",
      display_group: "coverage"
    },
    {
      pattern: "qd.cell_acceptance_ratio.*",
      label: "Cell-retention distribution",
      description: "Retention ratios across visited cells.",
      unit: "ratio",
      value_kind: "DISTRIBUTION",
      temporal_scope: "CUMULATIVE",
      population: "QD_CELLS",
      display_group: "cell summaries"
    },
    {
      pattern: "qd.cell_attempts.*",
      label: "Cell-attempt distribution",
      description: "Attempts across visited cells.",
      unit: "candidates",
      value_kind: "DISTRIBUTION",
      temporal_scope: "CUMULATIVE",
      population: "QD_CELLS",
      display_group: "cell summaries"
    },
    {
      pattern: "qd.cell.*.acceptance_ratio",
      label: "Retention ratio",
      description: "Retention ratio for every cell.",
      unit: "ratio",
      value_kind: "RATE",
      temporal_scope: "CUMULATIVE",
      population: "QD_CELLS",
      display_group: "cell map"
    }
  ],
  groups: [
    {
      name: "coverage",
      label: "Coverage",
      description: "Archive coverage."
    },
    {
      name: "cell summaries",
      label: "Cell summaries",
      description: "Summary distributions."
    },
    {
      name: "cell map",
      label: "Cell map",
      description: "Per-cell diagnostics."
    }
  ]
};

function metric(sequence: number, name: string, value: number, step = 1_000): Metric {
  return { sequence, event_sequence: null, name, value, step };
}

describe("QD archive diagnostics visualization", () => {
  it("renders aggregate cell statistics as a selectable temporal distribution", () => {
    const statistics = [
      ["minimum", 0.1],
      ["first_quartile", 0.3],
      ["median", 0.5],
      ["mean", 0.55],
      ["third_quartile", 0.7],
      ["maximum", 0.9]
    ] as const;
    const metrics = [
      metric(1, "qd.visited_cells", 48),
      metric(2, "qd.unmapped_attempts", 0),
      ...statistics.map(([name, value], index) =>
        metric(index + 3, `qd.cell_acceptance_ratio.${name}`, value)
      ),
      ...statistics.map(([name, value], index) =>
        metric(index + 9, `qd.cell_attempts.${name}`, value * 100)
      )
    ];

    render(
      <ObserverMetricPanel
        archiveAxes={["row_cardinality", "column_cardinality"]}
        evaluationBudget={50_000}
        metrics={metrics}
        view={view}
      />
    );

    expect(screen.getByRole("combobox", { name: "Cell summary" })).toHaveValue(
      "qd.cell_acceptance_ratio"
    );
    expect(screen.getByText("Retention ratios across visited cells.")).toBeVisible();
    expect(screen.getByTestId("chart").dataset.option).toContain("Interquartile range");
    expect(screen.queryByRole("columnheader", { name: "Metric" })).not.toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "Cell summary" }), {
      target: { value: "qd.cell_attempts" }
    });
    expect(screen.getByText("Attempts across visited cells.")).toBeVisible();
  });

  it("renders exact two-dimensional cell metrics as a heatmap", () => {
    render(
      <ObserverMetricPanel
        archiveAxes={["row_cardinality", "column_cardinality"]}
        evaluationBudget={50_000}
        metrics={[
          metric(1, "qd.visited_cells", 2),
          metric(2, "qd.cell.0_0.acceptance_ratio", 0.5),
          metric(3, "qd.cell.1_0.acceptance_ratio", 0.75)
        ]}
        view={view}
      />
    );

    expect(screen.getByRole("combobox", { name: "Cell diagnostic" })).toHaveValue(
      "acceptance_ratio"
    );
    expect(screen.getByText("Retention ratio for every cell.")).toBeVisible();
    expect(screen.getByTestId("chart").dataset.option).toContain("row_cardinality bin");
    expect(screen.queryByRole("combobox", { name: "Cell summary" })).not.toBeInTheDocument();
  });
});
