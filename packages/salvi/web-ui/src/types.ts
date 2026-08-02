export type JsonObject = Record<string, unknown>;

export interface ParameterDescription {
  name: string;
  title: string;
  description: string;
  required: boolean;
  default: unknown;
  value_schema: JsonObject;
  applicable_patterns: string[];
  widget: "BOOLEAN" | "NUMBER" | "SELECT" | "TEXT" | "STRUCTURED";
  unit: string | null;
  advanced: boolean;
}

export interface ComponentDescription {
  kind: string;
  name: string;
  title: string;
  description: string;
  provides: string[];
  requires: string[];
  supported_patterns: string[];
  conflicts: Array<{ kind: string; name: string }>;
  compatibility_notes: string[];
  maturity: string;
  parameters: ParameterDescription[];
  stage: string;
  order: number;
  observer_view: ObserverView | null;
}

export interface RoleDescription {
  kind: string;
  title: string;
  description: string;
  stage: string;
  order: number;
  icon: string;
  repeatable: boolean;
  configuration_path: string[];
  incoming: Array<{
    source: string;
    kind: "PRIMARY" | "SUPPORT" | "CONTROL" | "FEEDBACK";
  }>;
  accepts_pipeline_input: boolean;
  emits_pipeline_output: boolean;
}

export interface WorkflowStageDescription {
  stage: string;
  title: string;
  description: string;
  order: number;
  icon: string;
  theme: string;
  preferred_columns: number;
}

export interface ObserverView {
  view_kind: string;
  title: string;
  metric_patterns: string[];
  empty_message: string;
  x_axis_label: string | null;
  y_axis_label: string | null;
  metrics: ObserverMetricPresentation[];
  groups: ObserverMetricGroupPresentation[];
}

export interface ObserverMetricPresentation {
  pattern: string;
  label: string;
  description: string;
  unit: string;
  value_kind: "COUNTER" | "GAUGE" | "DELTA" | "RATE" | "DISTRIBUTION";
  temporal_scope: "CUMULATIVE" | "CURRENT" | "BATCH" | "WINDOW";
  population:
    | "RUN"
    | "EVALUATED_CANDIDATES"
    | "ARCHIVE_DECISIONS"
    | "REPERTOIRE"
    | "QD_CELLS"
    | "EMITTERS"
    | "PROCESS";
  display_group: string;
}

export interface ObserverMetricGroupPresentation {
  name: string;
  label: string;
  description: string;
}

export interface AdapterDescription {
  name: string;
  title: string;
  description: string;
  files: Array<{
    name: string;
    title: string;
    description: string;
    required: boolean;
    accepted_extensions: string[];
  }>;
  parameters: Array<{
    name: string;
    title: string;
    description: string;
    kind: "STRING" | "INTEGER" | "NUMBER" | "BOOLEAN";
    required: boolean;
    default: string | number | boolean | null;
    minimum: number | null;
    maximum: number | null;
  }>;
  supports_ground_truth: boolean;
  requires_confirmation: boolean;
}

export interface AnalysisDescription {
  name: string;
  title: string;
  description: string;
  requires_ground_truth: boolean;
}

export interface Catalog {
  workflow_stages: WorkflowStageDescription[];
  roles: RoleDescription[];
  components: ComponentDescription[];
  patterns: Array<Record<string, unknown>>;
  input_adapters: AdapterDescription[];
  analyses: AnalysisDescription[];
}

export interface RoleResolution {
  role: RoleDescription;
  state: "REQUIRED" | "AVAILABLE" | "CONFIGURED" | "UNAVAILABLE" | "INVALID";
  minimum: number;
  maximum: number | null;
  configured: string[];
  reasons: string[];
  instances: Array<{
    component: ComponentDescription;
    available: boolean;
    reasons: string[];
  }>;
}

export interface CompositionResolution {
  valid: boolean;
  complete: boolean;
  allowed_patterns: string[];
  roles: RoleResolution[];
  workflow_connections: Array<{
    source: string;
    target: string;
    kind: "PRIMARY" | "SUPPORT" | "CONTROL" | "FEEDBACK";
  }>;
  errors: string[];
}

export interface DatasetRecord {
  identifier: string;
  adapter: string;
  created_at: string;
  ground_truth_attached: boolean;
  clinical_annotations_attached: boolean;
}

export interface RunRecord {
  identifier: string;
  dataset_identifier: string;
  seed: number;
  analyses: string[];
  status: "created" | "running" | "completed" | "failed" | "cancelled";
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  has_events: boolean;
  has_raw_results: boolean;
  has_selected_results: boolean;
  monitoring: {
    observers: string[];
    archive_axes: string[];
    termination: {
      current: number;
      limit: number | null;
      unit: string;
    } | null;
  };
}

export interface Metric {
  sequence: number;
  event_sequence: number | null;
  name: string;
  value: number;
  step: number | null;
}

export interface ResultSummary {
  identifier: string;
  generation: number;
  row_count: number;
  column_count: number;
  objectives: Record<string, number>;
  constraints: Record<string, number>;
  descriptors: Record<string, number>;
  feasible: boolean;
  valid: boolean;
  archive_coordinate: number[] | null;
  selection_rank: number | null;
  quality_score: number | null;
  novelty_score: number | null;
  patterns: string[];
  provenance: JsonObject | null;
}

export interface ResultFilters {
  query: string;
  feasible: "" | "true" | "false";
  pattern: string;
  minRows: string;
  minColumns: string;
}
