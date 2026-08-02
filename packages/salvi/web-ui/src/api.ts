import type {
  Catalog,
  CompositionResolution,
  CompositionTransition,
  DatasetRecord,
  JsonObject,
  Metric,
  ResultFilters,
  ResultSummary,
  RunRecord,
  SearchFamily
} from "./types";

const API = "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({ detail: response.statusText }))) as {
      detail?: string;
    };
    throw new Error(payload.detail ?? response.statusText);
  }
  return (await response.json()) as T;
}

export const api = {
  catalog: () => request<Catalog>("/catalog"),
  defaultPipeline: () => request<{ yaml: string }>("/pipelines/default"),
  validatePipeline: (yaml: string) =>
    request<{ valid: boolean; yaml: string; configuration: JsonObject }>(
      "/pipelines/validate",
      { method: "POST", headers: { "Content-Type": "text/yaml" }, body: yaml }
    ),
  serializePipeline: (configuration: JsonObject) =>
    request<{ yaml: string }>("/pipelines/serialize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(configuration)
    }),
  resolve: (configuration: JsonObject) =>
    request<CompositionResolution>("/pipelines/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ configuration })
    }),
  switchSearchFamily: (configuration: JsonObject, searchFamily: SearchFamily) =>
    request<CompositionTransition>("/pipelines/search-family", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ configuration, search_family: searchFamily })
    }),
  datasets: async () => (await request<{ items: DatasetRecord[] }>("/datasets")).items,
  deleteDataset: (identifier: string) =>
    request(`/datasets/${encodeURIComponent(identifier)}`, { method: "DELETE" }),
  inspectImport: async (
    adapter: string,
    identifier: string,
    slots: Array<[string, File]>,
    parameters: Record<string, string | number | boolean>
  ) => {
    const body = new FormData();
    body.append("identifier", identifier);
    body.append("slot_names", JSON.stringify(slots.map(([slot]) => slot)));
    body.append("parameters", JSON.stringify(parameters));
    slots.forEach(([, file]) => body.append("files", file));
    return request<{
      identifier: string;
      status: string;
      preview: JsonObject & { confirmation_required: boolean; columns: JsonObject[] };
    }>(`/imports/${encodeURIComponent(adapter)}`, { method: "POST", body });
  },
  confirmImport: (
    identifier: string,
    columns: JsonObject[] | null,
    adapterConfiguration: JsonObject | null
  ) =>
    request<DatasetRecord>(`/imports/${encodeURIComponent(identifier)}/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        columns,
        adapter_configuration: adapterConfiguration
      })
    }),
  deleteImport: (identifier: string) =>
    request(`/imports/${encodeURIComponent(identifier)}`, { method: "DELETE" }),
  runs: async () => (await request<{ items: RunRecord[] }>("/runs")).items,
  run: (identifier: string) => request<RunRecord>(`/runs/${encodeURIComponent(identifier)}`),
  startRun: (
    pipeline: string,
    datasetIdentifier: string,
    runIdentifier: string,
    seed: number,
    analyses: string[]
  ) =>
    request<RunRecord>("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pipeline,
        dataset_identifier: datasetIdentifier,
        run_identifier: runIdentifier,
        seed,
        analyses
      })
    }),
  cancelRun: (identifier: string) =>
    request<RunRecord>(`/runs/${encodeURIComponent(identifier)}/cancel`, { method: "POST" }),
  deleteRun: (identifier: string) =>
    request(`/runs/${encodeURIComponent(identifier)}`, { method: "DELETE" }),
  metrics: async (identifier: string, after = 0) =>
    request<{ names: string[]; items: Metric[] }>(
      `/runs/${encodeURIComponent(identifier)}/metrics?after_sequence=${after}` +
        "&limit=25000&include_names=false"
    ),
  results: (
    identifier: string,
    kind: "raw" | "selected",
    offset: number,
    limit: number,
    filters: ResultFilters
  ) => {
    const query = new URLSearchParams({
      offset: String(offset),
      limit: String(limit)
    });
    if (filters.query.trim()) query.set("query", filters.query.trim());
    if (filters.feasible) query.set("feasible", filters.feasible);
    if (filters.pattern) query.set("pattern", filters.pattern);
    if (filters.minRows) query.set("min_rows", filters.minRows);
    if (filters.minColumns) query.set("min_columns", filters.minColumns);
    return (
    request<{ offset: number; limit: number; total: number; items: ResultSummary[] }>(
        `/runs/${encodeURIComponent(identifier)}/results/${kind}?${query.toString()}`
      )
    );
  },
  resultDetail: (identifier: string, kind: string, bicluster: string) =>
    request<JsonObject>(
      `/runs/${encodeURIComponent(identifier)}/results/${kind}/${encodeURIComponent(bicluster)}`
    ),
  matrix: (
    identifier: string,
    kind: string,
    bicluster: string,
    rowOffset = 0,
    columnOffset = 0
  ) =>
    request<JsonObject>(
      `/runs/${encodeURIComponent(identifier)}/results/${kind}/` +
        `${encodeURIComponent(bicluster)}/matrix?row_offset=${rowOffset}&row_limit=30` +
        `&column_offset=${columnOffset}&column_limit=20`
    ),
  accuracy: (identifier: string, kind: string, analysis: string) =>
    request<JsonObject>(
      `/runs/${encodeURIComponent(identifier)}/accuracy/${kind}/${encodeURIComponent(analysis)}`,
      { method: "POST" }
    )
};
