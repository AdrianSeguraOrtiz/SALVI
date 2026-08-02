import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import { appendMetricHistory, eventFailureMessage } from "../monitoring";
import type { Catalog, ComponentDescription, Metric, RunRecord } from "../types";
import { Monitor } from "./Monitor";

vi.mock("../api", () => ({
  api: {
    run: vi.fn(),
    metrics: vi.fn(),
    cancelRun: vi.fn(),
    deleteRun: vi.fn()
  }
}));

class EventSourceStub {
  addEventListener() {}
  close() {}
}

const failedRun: RunRecord = {
  identifier: "failed-run",
  dataset_identifier: "dataset",
  seed: 0,
  analyses: [],
  status: "failed",
  created_at: "2026-07-28T21:43:59Z",
  started_at: "2026-07-28T21:43:59Z",
  finished_at: "2026-07-28T21:44:02Z",
  error: "daemonic processes are not allowed to have children",
  has_events: true,
  has_raw_results: false,
  has_selected_results: false,
  monitoring: {
    observers: [],
    archive_axes: ["row_cardinality", "column_cardinality"],
    termination: { current: 0, limit: 50_000, unit: "evaluations" }
  }
};

const emptyCatalog: Catalog = {
  workflow_stages: [],
  roles: [],
  components: [],
  patterns: [],
  input_adapters: [],
  analyses: []
};

const searchProgress: ComponentDescription = {
  kind: "observer",
  name: "search_progress",
  title: "Search progress",
  description: "Persists search progress.",
  provides: ["observer"],
  requires: [],
  supported_patterns: ["CONSTANT", "ADDITIVE", "MULTIPLICATIVE"],
  conflicts: [],
  compatibility_notes: [],
  maturity: "STABLE",
  parameters: [],
  stage: "SEARCH",
  order: 0,
  observer_view: {
    view_kind: "KPI_SERIES",
    title: "Search progress",
    metric_patterns: ["search.*"],
    empty_message: "No progress yet.",
    x_axis_label: "Evaluations",
    y_axis_label: "Candidates",
    metrics: [
      {
        pattern: "search.evaluations",
        label: "Evaluations",
        description: "Completed candidate evaluations.",
        unit: "evaluations",
        value_kind: "COUNTER",
        temporal_scope: "CUMULATIVE",
        population: "RUN",
        display_group: "progress"
      }
    ],
    groups: [
      {
        name: "progress",
        label: "Evaluation progress",
        description: "Completed candidate evaluations."
      }
    ]
  }
};

describe("Monitor", () => {
  afterEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("extracts a durable failure from the event payload", () => {
    expect(
      eventFailureMessage({
        event_type: "run.failed",
        payload: { error: "worker failed" }
      })
    ).toBe("worker failed");
    expect(eventFailureMessage({ event_type: "run.completed", payload: {} })).toBeNull();
  });

  it("shows the full recorded error for a failed run", async () => {
    vi.stubGlobal("EventSource", EventSourceStub);
    vi.mocked(api.run).mockResolvedValue(failedRun);
    vi.mocked(api.metrics).mockResolvedValue({ names: [], items: [] });

    render(
      <Monitor
        catalog={emptyCatalog}
        runs={[failedRun]}
        selectedRun={failedRun.identifier}
        onSelectedRun={vi.fn()}
        onRunsChanged={vi.fn().mockResolvedValue(undefined)}
      />
    );

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        "daemonic processes are not allowed to have children"
      );
    });
    expect(screen.getByRole("alert")).toHaveTextContent("Execution failed");
  });

  it("preserves the beginning of metric history beyond the previous client limit", () => {
    const metric = (sequence: number): Metric => ({
      sequence,
      event_sequence: null,
      name: "search.evaluations",
      value: sequence,
      step: sequence
    });
    const current = Array.from({ length: 12_000 }, (_, index) => metric(index + 1));
    const combined = appendMetricHistory(current, [metric(12_000), metric(12_001)]);

    expect(combined).toHaveLength(12_001);
    expect(combined[0].sequence).toBe(1);
    expect(combined.at(-1)?.sequence).toBe(12_001);
  });

  it("moves minimized observers into a compact restorable dock", async () => {
    vi.stubGlobal("EventSource", EventSourceStub);
    const completedRun: RunRecord = {
      ...failedRun,
      status: "completed",
      error: null,
      monitoring: {
        ...failedRun.monitoring,
        observers: ["search_progress"]
      }
    };
    vi.mocked(api.run).mockResolvedValue(completedRun);
    vi.mocked(api.metrics).mockResolvedValue({ names: [], items: [] });
    const catalog = { ...emptyCatalog, components: [searchProgress] };

    const { container } = render(
      <Monitor
        catalog={catalog}
        runs={[completedRun]}
        selectedRun={completedRun.identifier}
        onSelectedRun={vi.fn()}
        onRunsChanged={vi.fn().mockResolvedValue(undefined)}
      />
    );

    await waitFor(() => {
      expect(container.querySelectorAll(".observer-grid .observer-panel")).toHaveLength(1);
    });
    expect(screen.getByText("Completed candidate evaluations.")).toBeVisible();
    fireEvent.click(screen.getByTitle("Minimize"));
    expect(container.querySelectorAll(".observer-grid .observer-panel")).toHaveLength(0);
    expect(screen.getByLabelText("Minimized observer panels")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Search progress" }));
    expect(container.querySelectorAll(".observer-grid .observer-panel")).toHaveLength(1);
  });
});
