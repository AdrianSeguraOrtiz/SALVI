import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PipelineWorkflow } from "./PipelineWorkflow";
import type { AnalysisDescription, CompositionResolution } from "../types";

const resolution: CompositionResolution = {
  valid: false,
  complete: false,
  allowed_patterns: ["CONSTANT"],
  errors: [],
  workflow_connections: [],
  roles: [
    {
      role: {
        kind: "search_engine",
        title: "Search engine",
        description: "Coordinates candidate generation and evaluation.",
        stage: "SEARCH",
        order: 10,
        icon: "search",
        repeatable: false,
        configuration_path: ["search", "engine"],
        incoming: [],
        accepts_pipeline_input: false,
        emits_pipeline_output: false
      },
      state: "REQUIRED",
      minimum: 1,
      maximum: 1,
      configured: [],
      reasons: ["Choose one search engine."],
      instances: []
    }
  ]
};

const searchStage = [
  {
    stage: "SEARCH",
    title: "Search",
    description: "Generate candidates.",
    order: 30,
    icon: "search",
    theme: "search",
    preferred_columns: 2
  }
];

describe("PipelineWorkflow", () => {
  it("hides unconfigured roles and exposes one catalog entry point per stage", () => {
    const onSelectStage = vi.fn();
    render(
      <PipelineWorkflow
        resolution={resolution}
        stages={searchStage}
        selectedRole=""
        selectedStage=""
        datasetLabel=""
        analyses={[]}
        availableAnalysisCount={0}
        onSelectRole={vi.fn()}
        onSelectStage={onSelectStage}
      />
    );

    expect(screen.getByText("Generate candidates.")).toBeInTheDocument();
    expect(screen.queryByText("Search engine")).not.toBeInTheDocument();
    expect(screen.getByText("1 required")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Add component").closest(".workflow-add-node")!);
    expect(onSelectStage).toHaveBeenCalledWith("SEARCH");
  });

  it("renders a configured role and delegates direct editing", () => {
    const onSelectRole = vi.fn();
    const configured: CompositionResolution = {
      ...resolution,
      workflow_connections: [
        {
          source: "search_engine",
          target: "__analysis__",
          kind: "PRIMARY"
        }
      ],
      roles: [
        {
          ...resolution.roles[0],
          state: "CONFIGURED",
          configured: ["serial_mome"],
          reasons: [],
          instances: [
            {
              available: true,
              reasons: [],
              component: {
                kind: "search_engine",
                name: "serial_mome",
                title: "Serial MOME",
                description: "Runs deterministic batched MOME.",
                provides: ["search-engine"],
                requires: [],
                supported_patterns: ["CONSTANT", "ADDITIVE", "MULTIPLICATIVE"],
                conflicts: [],
                compatibility_notes: [],
                maturity: "STABLE",
                parameters: [],
                stage: "SEARCH",
                order: 0,
                observer_view: null
              }
            }
          ]
        }
      ]
    };

    render(
      <PipelineWorkflow
        resolution={configured}
        stages={searchStage}
        selectedRole=""
        selectedStage=""
        datasetLabel=""
        analyses={[]}
        availableAnalysisCount={0}
        onSelectRole={onSelectRole}
        onSelectStage={vi.fn()}
      />
    );

    const node = screen.getByText("Serial MOME").closest(".workflow-node")!;
    expect(node.querySelector(".node-role")).toHaveTextContent("Search engine");
    expect(node.querySelector(".node-instance")).toHaveTextContent("Serial MOME");
    fireEvent.click(node);
    expect(onSelectRole).toHaveBeenCalledWith("search_engine");
  });

  it("renders configured post-run analyses as selectable workflow nodes", () => {
    const onSelectRole = vi.fn();
    const analyses: AnalysisDescription[] = [
      {
        name: "prelic_accuracy",
        title: "REC / REL / BE",
        description: "Compares detected biclusters with attached ground truth.",
        requires_ground_truth: true
      }
    ];

    render(
      <PipelineWorkflow
        resolution={{ ...resolution, roles: [] }}
        stages={[
          {
            stage: "ANALYSIS",
            title: "Analysis",
            description: "Inspect results.",
            order: 60,
            icon: "analysis",
            theme: "analysis",
            preferred_columns: 1
          }
        ]}
        selectedRole=""
        selectedStage=""
        datasetLabel="gbic-data"
        analyses={analyses}
        availableAnalysisCount={0}
        onSelectRole={onSelectRole}
        onSelectStage={vi.fn()}
      />
    );

    fireEvent.click(screen.getByText("REC / REL / BE").closest(".workflow-node")!);
    expect(onSelectRole).toHaveBeenCalledWith("__analysis__");
    expect(screen.getAllByText("1 active").length).toBeGreaterThan(0);
  });
});
