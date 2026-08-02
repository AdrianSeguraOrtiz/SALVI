import { memo, useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  Panel,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  type ReactFlowInstance
} from "@xyflow/react";
import {
  Activity,
  Archive,
  Boxes,
  ChartNoAxesCombined,
  ChartSpline,
  Check,
  CircleAlert,
  CirclePlus,
  Cpu,
  Crosshair,
  Database,
  Dna,
  Filter,
  Gauge,
  GitFork,
  ListFilter,
  Microscope,
  RadioTower,
  Radar,
  ScanSearch,
  ShieldAlert,
  ShieldCheck,
  Shuffle,
  SlidersHorizontal,
  Sparkles,
  TableProperties,
  Timer,
  TriangleAlert,
  UserRound,
  UsersRound,
  WandSparkles,
  Zap,
  type LucideIcon
} from "lucide-react";
import type {
  AnalysisDescription,
  CompositionResolution,
  RoleResolution,
  WorkflowStageDescription
} from "../types";

type WorkflowState = "CONFIGURED" | "INVALID" | "SPECIAL";
type ConnectionKind = "PRIMARY" | "SUPPORT" | "CONTROL" | "FEEDBACK";

interface WorkflowNodeData extends Record<string, unknown> {
  title: string;
  roleTitle: string;
  subtitle: string;
  state: WorkflowState;
  kind: string;
  icon: string;
  selected: boolean;
  onSelect: (kind: string) => void;
}

interface AddNodeData extends Record<string, unknown> {
  stage: string;
  title: string;
  subtitle: string;
  requiredCount: number;
  selected: boolean;
  onSelect: (stage: string) => void;
}

interface StageNodeData extends Record<string, unknown> {
  stage: string;
  sequence: string;
  title: string;
  description: string;
  configuredCount: number;
  requiredCount: number;
  icon: string;
  theme: string;
}

interface LayoutPoint {
  x: number;
  y: number;
}

const ROLE_WIDTH = 194;
const ROLE_HEIGHT = 72;
const ADD_HEIGHT = 64;
const STAGE_PADDING = 18;
const STAGE_HEADER = 88;
const STAGE_GAP = 22;
const COLUMN_GAP = 18;
const ROW_GAP = 20;
const MIN_STAGE_HEIGHT = 660;
const WORKFLOW_FOOTER_HEIGHT = 54;
const presentationIcons: Record<string, LucideIcon> = {
  alert: CircleAlert,
  archive: Archive,
  boxes: Boxes,
  chart: ChartSpline,
  "chart-combined": ChartNoAxesCombined,
  component: Activity,
  cpu: Cpu,
  crosshair: Crosshair,
  database: Database,
  dna: Dna,
  filter: Filter,
  fork: GitFork,
  gauge: Gauge,
  "list-filter": ListFilter,
  microscope: Microscope,
  radar: Radar,
  radio: RadioTower,
  scan: ScanSearch,
  "shield-alert": ShieldAlert,
  "shield-check": ShieldCheck,
  shuffle: Shuffle,
  sliders: SlidersHorizontal,
  sparkles: Sparkles,
  table: TableProperties,
  timer: Timer,
  user: UserRound,
  users: UsersRound,
  wand: WandSparkles,
  zap: Zap
};

function NodeHandles() {
  return (
    <>
      <Handle id="target-left" type="target" position={Position.Left} />
      <Handle id="target-right" type="target" position={Position.Right} />
      <Handle id="target-top" type="target" position={Position.Top} />
      <Handle id="target-bottom" type="target" position={Position.Bottom} />
      <Handle id="source-left" type="source" position={Position.Left} />
      <Handle id="source-right" type="source" position={Position.Right} />
      <Handle id="source-top" type="source" position={Position.Top} />
      <Handle id="source-bottom" type="source" position={Position.Bottom} />
    </>
  );
}

const WorkflowNode = memo(({ data }: NodeProps<Node<WorkflowNodeData>>) => {
  const Icon = presentationIcons[data.icon] ?? Activity;
  return (
    <div
      className={[
        "workflow-node",
        `state-${data.state.toLowerCase()}`,
        data.selected ? "is-selected" : ""
      ].join(" ")}
      role="button"
      tabIndex={0}
      aria-label={`${data.roleTitle}: ${data.title}`}
      onClick={() => data.onSelect(data.kind)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          data.onSelect(data.kind);
        }
      }}
    >
      <NodeHandles />
      <div className="workflow-node-heading">
        <span className="node-icon">
          <Icon size={17} />
        </span>
        <strong className="node-role">{data.roleTitle}</strong>
        {data.state === "INVALID" ? <TriangleAlert className="node-warning" size={14} /> : null}
        {data.state === "CONFIGURED" ? <Check className="node-check" size={14} /> : null}
      </div>
      <span className="node-instance">{data.title}</span>
      <small title={data.subtitle}>{data.subtitle}</small>
    </div>
  );
});

const AddNode = memo(({ data }: NodeProps<Node<AddNodeData>>) => (
  <div
    className={[
      "workflow-add-node",
      data.requiredCount > 0 ? "has-required" : "",
      data.selected ? "is-selected" : ""
    ].join(" ")}
    role="button"
    tabIndex={0}
    aria-label={`${data.title} in ${data.stage}`}
    onClick={() => data.onSelect(data.stage)}
    onKeyDown={(event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        data.onSelect(data.stage);
      }
    }}
  >
    <span className="add-icon">
      <CirclePlus size={20} />
    </span>
    <span>
      <strong>{data.title}</strong>
      <small>{data.subtitle}</small>
    </span>
    {data.requiredCount > 0 ? (
      <b>{data.requiredCount} required</b>
    ) : null}
  </div>
));

const StageNode = memo(({ data }: NodeProps<Node<StageNodeData>>) => {
  const Icon = presentationIcons[data.icon] ?? Activity;
  return (
    <div className={`workflow-stage stage-${data.theme}`}>
      <header>
        <span className="stage-icon">
          <Icon size={16} />
        </span>
        <span>
          <small>{data.sequence}</small>
          <strong>{data.title}</strong>
        </span>
      </header>
      <p>{data.description}</p>
      <div className="stage-summary">
        <span>{data.configuredCount} active</span>
        {data.requiredCount > 0 ? <b>{data.requiredCount} missing</b> : null}
      </div>
    </div>
  );
});

const nodeTypes = {
  workflow: WorkflowNode,
  add: AddNode,
  stage: StageNode
};

function stageWidth(stage: WorkflowStageDescription): number {
  const columns = stage.preferred_columns;
  return STAGE_PADDING * 2 + columns * ROLE_WIDTH + (columns - 1) * COLUMN_GAP;
}

function handlesBetween(source: LayoutPoint, target: LayoutPoint) {
  const deltaX = target.x - source.x;
  const deltaY = target.y - source.y;
  if (Math.abs(deltaX) >= Math.abs(deltaY) * 0.7) {
    return deltaX >= 0
      ? { sourceHandle: "source-right", targetHandle: "target-left" }
      : { sourceHandle: "source-left", targetHandle: "target-right" };
  }
  return deltaY >= 0
    ? { sourceHandle: "source-bottom", targetHandle: "target-top" }
    : { sourceHandle: "source-top", targetHandle: "target-bottom" };
}

function edgeFor(
  id: string,
  source: string,
  target: string,
  kind: ConnectionKind,
  positions: Map<string, LayoutPoint>,
  selectedRole: string
): Edge {
  const handles = handlesBetween(positions.get(source)!, positions.get(target)!);
  const related = !selectedRole || source === selectedRole || target === selectedRole;
  return {
    id,
    source,
    target,
    ...handles,
    type: "smoothstep",
    className: [
      "workflow-edge",
      `relation-${kind.toLowerCase()}`,
      selectedRole ? (related ? "is-highlighted" : "is-muted") : ""
    ].join(" "),
    markerEnd: {
      type: MarkerType.ArrowClosed,
      width: 14,
      height: 14
    }
  };
}

function configuredLabels(role: RoleResolution): string[] {
  const titles = new Map(
    role.instances.map((instance) => [instance.component.name, instance.component.title])
  );
  return role.configured.map((name) => titles.get(name) ?? name);
}

function nodeCopy(role: RoleResolution) {
  const labels = configuredLabels(role);
  if (role.role.repeatable) {
    return {
      title: `${labels.length} selected`,
      subtitle: labels.join(", ")
    };
  }
  return {
    title: labels[0] ?? role.role.title,
    subtitle: role.role.description
  };
}

interface Props {
  resolution: CompositionResolution | null;
  stages: WorkflowStageDescription[];
  selectedRole: string;
  selectedStage: string;
  datasetLabel: string;
  analyses: AnalysisDescription[];
  availableAnalysisCount: number;
  onSelectRole: (kind: string) => void;
  onSelectStage: (stage: string) => void;
}

export function PipelineWorkflow({
  resolution,
  stages,
  selectedRole,
  selectedStage,
  datasetLabel,
  analyses,
  availableAnalysisCount,
  onSelectRole,
  onSelectStage
}: Props) {
  const [flow, setFlow] = useState<ReactFlowInstance | null>(null);
  const initialFitComplete = useRef(false);
  const { nodes, edges, contentHeight } = useMemo(() => {
    const roles = resolution?.roles ?? [];
    const activeRoles = roles.filter((role) => role.configured.length > 0);
    const orderedStages = [...stages].sort((left, right) => left.order - right.order);
    const stagePositions = new Map<string, { x: number; width: number }>();
    const positions = new Map<string, LayoutPoint>();
    let nextX = 0;
    for (const stage of orderedStages) {
      const width = stageWidth(stage);
      stagePositions.set(stage.stage, { x: nextX, width });
      nextX += width + STAGE_GAP;
    }

    const requiredStageHeight = Math.max(
      MIN_STAGE_HEIGHT,
      ...orderedStages.map((stage) => {
        const columns = stage.preferred_columns;
        const roleCount = activeRoles.filter(
          (role) => role.role.stage === stage.stage
        ).length;
        const specialCount =
          (stage.stage === "INPUT" && datasetLabel ? 1 : 0) +
          (stage.stage === "ANALYSIS" ? analyses.length : 0);
        const rows = Math.ceil((roleCount + specialCount + 1) / columns);
        return (
          STAGE_HEADER +
          rows * ROLE_HEIGHT +
          Math.max(0, rows - 1) * ROW_GAP +
          STAGE_PADDING
        );
      })
    );

    const builtNodes: Node<WorkflowNodeData | AddNodeData | StageNodeData>[] = [];
    for (const [index, stage] of orderedStages.entries()) {
      const placement = stagePositions.get(stage.stage)!;
      const configuredCount =
        activeRoles.filter((role) => role.role.stage === stage.stage).length +
        (stage.stage === "INPUT" && datasetLabel ? 1 : 0) +
        (stage.stage === "ANALYSIS" ? analyses.length : 0);
      const requiredCount =
        roles.filter(
          (role) => role.role.stage === stage.stage && role.state === "REQUIRED"
        ).length + (stage.stage === "INPUT" && !datasetLabel ? 1 : 0);
      builtNodes.push({
        id: `__stage_${stage.stage.toLowerCase()}__`,
        type: "stage",
        position: { x: placement.x, y: 0 },
        draggable: false,
        selectable: false,
        focusable: false,
        zIndex: -10,
        style: { width: placement.width, height: requiredStageHeight },
        data: {
          stage: stage.stage,
          sequence: String(index + 1).padStart(2, "0"),
          title: stage.title,
          description: stage.description,
          configuredCount,
          requiredCount,
          icon: stage.icon,
          theme: stage.theme
        }
      });
    }

    const stageItems = new Map<string, RoleResolution[]>();
    for (const stage of orderedStages) {
      stageItems.set(
        stage.stage,
        activeRoles
          .filter((role) => role.role.stage === stage.stage)
          .sort((left, right) => left.role.order - right.role.order)
      );
    }

    if (datasetLabel) {
      const stage = stagePositions.get("INPUT");
      if (stage) {
        const position = { x: stage.x + STAGE_PADDING, y: STAGE_HEADER };
        positions.set("__input__", position);
        builtNodes.push({
          id: "__input__",
          type: "workflow",
          position,
          draggable: false,
          selectable: true,
          focusable: false,
          zIndex: 5,
          style: { width: ROLE_WIDTH, height: ROLE_HEIGHT },
          data: {
            title: "Dataset input",
            roleTitle: "Input",
            subtitle: datasetLabel,
            state: "SPECIAL",
            kind: "__input__",
            icon: "database",
            selected: selectedRole === "__input__",
            onSelect: onSelectRole
          }
        });
      }
    }

    for (const stage of orderedStages) {
      const placement = stagePositions.get(stage.stage)!;
      const items = stageItems.get(stage.stage) ?? [];
      const columns = stage.preferred_columns;
      const offset = stage.stage === "INPUT" && datasetLabel ? 1 : 0;
      items.forEach((role, index) => {
        const slot = index + offset;
        const column = slot % columns;
        const row = Math.floor(slot / columns);
        const position = {
          x: placement.x + STAGE_PADDING + column * (ROLE_WIDTH + COLUMN_GAP),
          y: STAGE_HEADER + row * (ROLE_HEIGHT + ROW_GAP)
        };
        positions.set(role.role.kind, position);
        const copy = nodeCopy(role);
        builtNodes.push({
          id: role.role.kind,
          type: "workflow",
          position,
          draggable: false,
          selectable: true,
          focusable: false,
          zIndex: 5,
          style: { width: ROLE_WIDTH, height: ROLE_HEIGHT },
          data: {
            ...copy,
            roleTitle: role.role.title,
            state: role.state === "INVALID" ? "INVALID" : "CONFIGURED",
            kind: role.role.kind,
            icon: role.role.icon,
            selected: selectedRole === role.role.kind,
            onSelect: onSelectRole
          }
        });
      });

      if (stage.stage === "ANALYSIS") {
        analyses.forEach((analysis, index) => {
          const position = {
            x: placement.x + STAGE_PADDING,
            y: STAGE_HEADER + index * (ROLE_HEIGHT + ROW_GAP)
          };
          const identifier = `__analysis_${analysis.name}__`;
          positions.set(identifier, position);
          builtNodes.push({
            id: identifier,
            type: "workflow",
            position,
            draggable: false,
            selectable: true,
            focusable: false,
            zIndex: 5,
            style: { width: ROLE_WIDTH, height: ROLE_HEIGHT },
            data: {
              title: analysis.title,
              roleTitle: "Analysis",
              subtitle: analysis.description,
              state: "SPECIAL",
              kind: "__analysis__",
              icon: "chart",
              selected: selectedRole === "__analysis__",
              onSelect: onSelectRole
            }
          });
        });
      }

      const specialOffset =
        (stage.stage === "INPUT" && datasetLabel ? 1 : 0) +
        (stage.stage === "ANALYSIS" ? analyses.length : 0);
      const slot = items.length + specialOffset;
      const column = slot % columns;
      const row = Math.floor(slot / columns);
      const requiredCount =
        roles.filter(
          (role) => role.role.stage === stage.stage && role.state === "REQUIRED"
        ).length + (stage.stage === "INPUT" && !datasetLabel ? 1 : 0);
      const compatibleCount =
        stage.stage === "ANALYSIS"
          ? availableAnalysisCount
          : roles
              .filter((role) => role.role.stage === stage.stage)
              .flatMap((role) =>
                role.instances.map((instance) => ({
                  instance,
                  active: role.configured.includes(instance.component.name)
                }))
              )
              .filter(({ instance, active }) => instance.available && !active).length;
      const addTitle =
        stage.stage === "INPUT"
          ? datasetLabel
            ? "Change input"
            : "Select input"
          : stage.stage === "ANALYSIS"
            ? "Add analysis"
            : "Add component";
      builtNodes.push({
        id: `__add_${stage.stage.toLowerCase()}__`,
        type: "add",
        position: {
          x: placement.x + STAGE_PADDING + column * (ROLE_WIDTH + COLUMN_GAP),
          y: STAGE_HEADER + row * (ROLE_HEIGHT + ROW_GAP)
        },
        draggable: false,
        selectable: true,
        focusable: false,
        zIndex: 5,
        style: { width: ROLE_WIDTH, height: ADD_HEIGHT },
        data: {
          stage: stage.stage,
          title: addTitle,
          subtitle:
            compatibleCount > 0
              ? `${compatibleCount} compatible option${compatibleCount === 1 ? "" : "s"}`
              : "Inspect this stage",
          requiredCount,
          selected: selectedStage === stage.stage,
          onSelect: onSelectStage
        }
      });
    }

    const builtEdges: Edge[] = [];
    for (const [index, connection] of (resolution?.workflow_connections ?? []).entries()) {
      const targets =
        connection.target === "__analysis__"
          ? analyses.map((analysis) => `__analysis_${analysis.name}__`)
          : [connection.target];
      for (const target of targets) {
        if (!positions.has(connection.source) || !positions.has(target)) continue;
        builtEdges.push(
          edgeFor(
            `${connection.source}-${target}-${index}`,
            connection.source,
            target,
            connection.kind,
            positions,
            selectedRole
          )
        );
      }
    }
    return {
      nodes: builtNodes,
      edges: builtEdges,
      contentHeight: requiredStageHeight + WORKFLOW_FOOTER_HEIGHT
    };
  }, [
    analyses,
    availableAnalysisCount,
    datasetLabel,
    onSelectRole,
    onSelectStage,
    resolution,
    selectedRole,
    selectedStage,
    stages
  ]);

  useEffect(() => {
    if (!flow || !resolution || initialFitComplete.current) return;
    initialFitComplete.current = true;
    const frame = window.requestAnimationFrame(() => {
      if (window.innerWidth <= 640) {
        void flow.setViewport({ x: 14, y: 54, zoom: 0.72 }, { duration: 250 });
      } else {
        void flow.fitView({ padding: 0.035, minZoom: 0.45, maxZoom: 1, duration: 250 });
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [flow, resolution]);

  return (
    <div className="workflow-canvas">
      <div className="workflow-flow-surface" style={{ height: contentHeight }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          nodesDraggable={false}
          nodesConnectable={false}
          edgesFocusable={false}
          elementsSelectable
          minZoom={0.2}
          maxZoom={1.35}
          fitView
          fitViewOptions={{ padding: 0.035, minZoom: 0.2, maxZoom: 1 }}
          onInit={setFlow}
          panOnScroll={false}
          zoomOnScroll={false}
          preventScrolling={false}
          zoomOnDoubleClick={false}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#d7dfdb" gap={20} size={1} />
          <Controls showInteractive={false} />
          <Panel position="bottom-center" className="workflow-legend">
            <span>
              <i className="legend-dot state-configured" />
              Configured
            </span>
            <span>
              <i className="legend-dot state-invalid" />
              Needs attention
            </span>
            <span>
              <i className="legend-add">
                <CirclePlus size={11} />
              </i>
              Add to stage
            </span>
          </Panel>
        </ReactFlow>
      </div>
    </div>
  );
}
