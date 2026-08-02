import { useEffect, useMemo, useState } from "react";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable
} from "@tanstack/react-table";
import {
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Download,
  Grid3X3,
  RotateCcw,
  Rows3,
  Search,
  Target,
  XCircle
} from "lucide-react";
import { api } from "../api";
import { humanizeMetricName } from "../monitoring";
import type {
  AnalysisDescription,
  DatasetRecord,
  JsonObject,
  ResultFilters,
  ResultSummary,
  RunRecord
} from "../types";
import { Chart } from "./Chart";

interface Props {
  runs: RunRecord[];
  datasets: DatasetRecord[];
  analyses: AnalysisDescription[];
  patterns: string[];
  selectedRun: string;
  onSelectedRun: (identifier: string) => void;
}

const columnHelper = createColumnHelper<ResultSummary>();
const columns = [
  columnHelper.accessor("selection_rank", {
    header: "Rank",
    cell: (info) => (
      <span className="rank-value">{info.getValue() == null ? "-" : `#${info.getValue()! + 1}`}</span>
    )
  }),
  columnHelper.accessor("identifier", {
    header: "Bicluster",
    cell: (info) => {
      const result = info.row.original;
      const producer =
        result.provenance && typeof result.provenance.producer === "string"
          ? result.provenance.producer
          : null;
      return (
        <div className="result-identity">
          <strong title={info.getValue()}>{shortIdentifier(info.getValue())}</strong>
          <span>
            generation {result.generation}
            {producer ? ` · ${humanizeMetricName(producer)}` : ""}
          </span>
          <div className="pattern-badges">
            {result.patterns.map((pattern) => (
              <i key={pattern}>{pattern}</i>
            ))}
          </div>
        </div>
      );
    }
  }),
  columnHelper.accessor("row_count", {
    header: "Structure",
    cell: (info) => (
      <div className="structure-value">
        <Rows3 size={14} />
        <strong>{info.getValue()}</strong>
        <span>×</span>
        <Grid3X3 size={14} />
        <strong>{info.row.original.column_count}</strong>
      </div>
    )
  }),
  columnHelper.accessor("objectives", {
    header: "Objectives",
    cell: (info) => <ValueChips values={info.getValue()} />
  }),
  columnHelper.accessor("quality_score", {
    header: "Selection",
    cell: (info) => {
      const result = info.row.original;
      if (info.getValue() == null && result.novelty_score == null) return "-";
      return (
        <div className="selection-scores">
          {info.getValue() == null ? null : (
            <span>
              <small>Quality</small>
              <strong>{info.getValue()!.toFixed(3)}</strong>
            </span>
          )}
          {result.novelty_score == null ? null : (
            <span>
              <small>Novelty</small>
              <strong>{result.novelty_score.toFixed(3)}</strong>
            </span>
          )}
        </div>
      );
    }
  }),
  columnHelper.accessor("feasible", {
    header: "Status",
    cell: (info) => (
      <span className={`result-status ${info.getValue() ? "feasible" : "infeasible"}`}>
        {info.getValue() ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
        {info.getValue() ? "Feasible" : "Infeasible"}
      </span>
    )
  })
];

function shortIdentifier(identifier: string): string {
  return identifier.length <= 26
    ? identifier
    : `${identifier.slice(0, 12)}…${identifier.slice(-9)}`;
}

function ValueChips({ values }: { values: Record<string, number> }) {
  return (
    <div className="value-chips">
      {Object.entries(values).map(([name, value]) => (
        <span key={name} title={name}>
          <small>{humanizeMetricName(name)}</small>
          <strong>{value.toFixed(3)}</strong>
        </span>
      ))}
    </div>
  );
}

export function Results({
  runs,
  datasets,
  analyses,
  patterns,
  selectedRun,
  onSelectedRun
}: Props) {
  const [kind, setKind] = useState<"raw" | "selected">("selected");
  const [items, setItems] = useState<ResultSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState<JsonObject | null>(null);
  const [matrix, setMatrix] = useState<JsonObject | null>(null);
  const [matrixRowOffset, setMatrixRowOffset] = useState(0);
  const [matrixColumnOffset, setMatrixColumnOffset] = useState(0);
  const [accuracy, setAccuracy] = useState<Record<string, JsonObject | null>>({});
  const [accuracyLoading, setAccuracyLoading] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [filters, setFilters] = useState<ResultFilters>({
    query: "",
    feasible: "",
    pattern: "",
    minRows: "",
    minColumns: ""
  });
  const pageSize = 50;
  const run = runs.find((item) => item.identifier === selectedRun);
  const dataset = datasets.find((item) => item.identifier === run?.dataset_identifier);
  const enabledAnalyses = analyses.filter((analysis) =>
    run?.analyses.includes(analysis.name)
  );
  const enabledAnalysisNames = enabledAnalyses.map((analysis) => analysis.name).join("\0");

  useEffect(() => {
    if (!selectedRun) return;
    setError("");
    api
      .results(selectedRun, kind, offset, pageSize, filters)
      .then((response) => {
        setItems(response.items);
        setTotal(response.total);
        if (selected && !response.items.some((item) => item.identifier === selected)) {
          setSelected("");
          setDetail(null);
          setMatrix(null);
        }
      })
      .catch((cause) => setError(cause instanceof Error ? cause.message : String(cause)));
  }, [filters, kind, offset, selected, selectedRun]);

  useEffect(() => setOffset(0), [filters]);

  useEffect(() => {
    if (!selectedRun || !selected) return;
    api
      .resultDetail(selectedRun, kind, selected)
      .then(setDetail)
      .catch((cause) => setError(cause instanceof Error ? cause.message : String(cause)));
  }, [kind, selected, selectedRun]);

  useEffect(() => {
    if (!selectedRun || !selected) return;
    api
      .matrix(selectedRun, kind, selected, matrixRowOffset, matrixColumnOffset)
      .then(setMatrix)
      .catch((cause) => setError(cause instanceof Error ? cause.message : String(cause)));
  }, [kind, matrixColumnOffset, matrixRowOffset, selected, selectedRun]);

  useEffect(() => {
    if (!run || !dataset?.ground_truth_attached || !enabledAnalysisNames) return;
    enabledAnalysisNames.split("\0").forEach((analysisName) => {
      const key = accuracyKey(run.identifier, kind, analysisName);
      if (key in accuracy || accuracyLoading.includes(key)) return;
      setAccuracyLoading((current) => [...current, key]);
      api
        .accuracy(run.identifier, kind, analysisName)
        .then((value) => {
          setAccuracy((current) => ({ ...current, [key]: value }));
        })
        .catch((cause) => {
          setAccuracy((current) => ({ ...current, [key]: null }));
          setError(cause instanceof Error ? cause.message : String(cause));
        })
        .finally(() =>
          setAccuracyLoading((current) => current.filter((item) => item !== key))
        );
    });
  }, [
    accuracy,
    accuracyLoading,
    dataset?.ground_truth_attached,
    enabledAnalysisNames,
    kind,
    run
  ]);

  const table = useReactTable({ data: items, columns, getCoreRowModel: getCoreRowModel() });
  const objectiveBars = useMemo(() => {
    if (!detail || !Array.isArray(detail.columns)) return [];
    const values: Array<{ column: string; objective: string; value: number }> = [];
    detail.columns.forEach((raw) => {
      const column = raw as JsonObject;
      const objectives = column.objectives as JsonObject;
      Object.entries(objectives ?? {}).forEach(([objective, item]) => {
        if (item && typeof item === "object" && "value" in item) {
          values.push({
            column: String(column.name),
            objective,
            value: Number((item as JsonObject).value)
          });
        }
      });
    });
    return values;
  }, [detail]);
  const objectiveColumns = useMemo(
    () => [...new Set(objectiveBars.map((item) => item.column))],
    [objectiveBars]
  );
  const objectiveNames = useMemo(
    () => [...new Set(objectiveBars.map((item) => item.objective))],
    [objectiveBars]
  );
  const denseObjectiveChart = objectiveColumns.length > 24;
  const objectiveWindowEnd = Math.max(
    10,
    Math.min(100, (24 / Math.max(1, objectiveColumns.length)) * 100)
  );

  return (
    <main className="workspace results-workspace">
      <header className="workspace-header">
        <div>
          <span className="eyebrow">Canonical output</span>
          <h1>Results</h1>
          <p>Compare the raw repertoire and final selection, then inspect each bicluster.</p>
        </div>
        <div className="results-controls">
          <label className="compact-select">
            <span>Run</span>
            <select value={selectedRun} onChange={(event) => onSelectedRun(event.target.value)}>
              <option value="">Select a completed run</option>
              {runs
                .filter((item) => item.has_selected_results)
                .map((item) => (
                  <option value={item.identifier} key={item.identifier}>
                    {item.identifier}
                  </option>
                ))}
            </select>
          </label>
          {selectedRun ? (
            <a
              className="button secondary"
              href={`/api/v1/runs/${encodeURIComponent(selectedRun)}/download`}
            >
              <Download size={17} /> Download
            </a>
          ) : null}
        </div>
      </header>

      {run ? (
        <>
          <section className="result-toolbar">
            <div className="segmented">
              <button
                className={kind === "raw" ? "active" : ""}
                disabled={!run.has_raw_results}
                onClick={() => {
                  setKind("raw");
                  setOffset(0);
                }}
              >
                Raw repertoire
              </button>
              <button
                className={kind === "selected" ? "active" : ""}
                onClick={() => {
                  setKind("selected");
                  setOffset(0);
                }}
              >
                Final selection
              </button>
            </div>
            <span>{total} biclusters</span>
          </section>
          <section className="result-filters" aria-label="Result filters">
            <label className="filter-search">
              <Search size={16} />
              <input
                value={filters.query}
                placeholder="Identifier, objective, descriptor, provenance..."
                onChange={(event) =>
                  setFilters((current) => ({ ...current, query: event.target.value }))
                }
              />
            </label>
            <label>
              <span>Feasibility</span>
              <select
                value={filters.feasible}
                onChange={(event) =>
                  setFilters((current) => ({
                    ...current,
                    feasible: event.target.value as ResultFilters["feasible"]
                  }))
                }
              >
                <option value="">All</option>
                <option value="true">Feasible</option>
                <option value="false">Infeasible</option>
              </select>
            </label>
            <label>
              <span>Pattern</span>
              <select
                value={filters.pattern}
                onChange={(event) =>
                  setFilters((current) => ({ ...current, pattern: event.target.value }))
                }
              >
                <option value="">All</option>
                {patterns.map((pattern) => (
                  <option value={pattern} key={pattern}>
                    {pattern}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Minimum rows</span>
              <input
                type="number"
                min={1}
                value={filters.minRows}
                onChange={(event) =>
                  setFilters((current) => ({ ...current, minRows: event.target.value }))
                }
              />
            </label>
            <label>
              <span>Minimum columns</span>
              <input
                type="number"
                min={1}
                value={filters.minColumns}
                onChange={(event) =>
                  setFilters((current) => ({ ...current, minColumns: event.target.value }))
                }
              />
            </label>
            <button
              className="icon-button"
              title="Clear result filters"
              onClick={() =>
                setFilters({
                  query: "",
                  feasible: "",
                  pattern: "",
                  minRows: "",
                  minColumns: ""
                })
              }
            >
              <RotateCcw size={17} />
            </button>
          </section>

          {dataset?.ground_truth_attached && enabledAnalyses.length > 0 ? (
            <section className="accuracy-panel">
              <header>
                <span className="accuracy-icon">
                  <Target size={19} />
                </span>
                <div>
                  <h2>Ground-truth accuracy</h2>
                  <p>
                    Fixed post-run assessment of the{" "}
                    {kind === "raw" ? "raw repertoire" : "final selection"}; these values do
                    not influence the search.
                  </p>
                </div>
              </header>
              <div className="accuracy-results">
                {enabledAnalyses.map((analysis) => {
                  const key = accuracyKey(run.identifier, kind, analysis.name);
                  const value = accuracy[key];
                  return (
                    <article key={key}>
                      <strong>{analysis.title}</strong>
                      {value ? (
                        <div className="accuracy-metrics">
                          <Metric label="REL" value={Number(value.relevance)} />
                          <Metric label="REC" value={Number(value.recovery)} />
                          <Metric label="BE" value={Number(value.biclustering_error)} />
                          <Metric
                            label="Detected"
                            value={Number(value.detected_count)}
                            digits={0}
                          />
                          <Metric
                            label="Ground truth"
                            value={Number(value.ground_truth_count)}
                            digits={0}
                          />
                        </div>
                      ) : value === null ? (
                        <span className="accuracy-loading">Accuracy unavailable</span>
                      ) : (
                        <span className="accuracy-loading">Calculating accuracy…</span>
                      )}
                    </article>
                  );
                })}
              </div>
            </section>
          ) : null}

          <section className="results-layout">
            <div className="result-table-wrap">
              <table className="data-table">
                <thead>
                  {table.getHeaderGroups().map((group) => (
                    <tr key={group.id}>
                      {group.headers.map((header) => (
                        <th key={header.id}>
                          {flexRender(header.column.columnDef.header, header.getContext())}
                        </th>
                      ))}
                    </tr>
                  ))}
                </thead>
                <tbody>
                  {table.getRowModel().rows.map((row) => (
                    <tr
                      key={row.id}
                      className={row.original.identifier === selected ? "selected" : ""}
                      onClick={() => {
                        setSelected(row.original.identifier);
                        setMatrixRowOffset(0);
                        setMatrixColumnOffset(0);
                      }}
                    >
                      {row.getVisibleCells().map((cell) => (
                        <td key={cell.id}>
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <footer className="pagination">
                <button
                  className="icon-button"
                  disabled={offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - pageSize))}
                  title="Previous page"
                >
                  <ChevronLeft size={18} />
                </button>
                <span>
                  {total === 0 ? 0 : offset + 1}-{Math.min(total, offset + pageSize)} of {total}
                </span>
                <button
                  className="icon-button"
                  disabled={offset + pageSize >= total}
                  onClick={() => setOffset(offset + pageSize)}
                  title="Next page"
                >
                  <ChevronRight size={18} />
                </button>
              </footer>
            </div>

            <aside className="bicluster-inspector">
              {detail ? (
                <>
                  <header>
                    <span className="eyebrow">Selected bicluster</span>
                    <h2>{String(detail.identifier)}</h2>
                    <p>
                      <Rows3 size={15} /> {String(detail.row_count)} rows
                      <Grid3X3 size={15} /> {String(detail.column_count)} columns
                    </p>
                  </header>
                  <MatrixFragment
                    matrix={matrix}
                    onRowsChanged={setMatrixRowOffset}
                    onColumnsChanged={setMatrixColumnOffset}
                  />
                  <section>
                    <h3>Column contribution</h3>
                    <p className="section-description">
                      Per-column contribution to each selected objective. Use zoom or hover to
                      inspect dense biclusters.
                    </p>
                    <Chart
                      height={330}
                      option={{
                        animation: false,
                        color: ["#0b7a75", "#ef765f", "#d69e2e"],
                        tooltip: { trigger: "axis" },
                        legend: { type: "scroll", bottom: 2 },
                        grid: {
                          top: 15,
                          right: 18,
                          bottom: denseObjectiveChart ? 130 : 96,
                          left: 56,
                          containLabel: true
                        },
                        xAxis: {
                          type: "category",
                          data: objectiveColumns,
                          name: "Selected columns",
                          nameLocation: "middle",
                          nameGap: denseObjectiveChart ? 52 : 46,
                          axisLabel: { rotate: 35, hideOverlap: true }
                        },
                        yAxis: {
                          type: "value",
                          name: "Contribution",
                          nameLocation: "middle",
                          nameGap: 38
                        },
                        dataZoom:
                          denseObjectiveChart
                            ? [
                                { type: "inside", start: 0, end: objectiveWindowEnd },
                                {
                                  type: "slider",
                                  height: 14,
                                  bottom: 36,
                                  start: 0,
                                  end: objectiveWindowEnd
                                }
                              ]
                            : undefined,
                        series: objectiveNames.map((objective) => ({
                            name: objective,
                            type: "bar",
                            data: objectiveColumns.map(
                              (column) =>
                                objectiveBars.find(
                                  (item) =>
                                    item.column === column && item.objective === objective
                                )?.value ?? null
                            )
                          }))
                      }}
                    />
                  </section>
                  <ColumnExplanations detail={detail} />
                  <EvaluationMetadata detail={detail} />
                </>
              ) : (
                <div className="empty-state compact">
                  <Grid3X3 size={30} />
                  <h3>Select a bicluster</h3>
                  <p>Its structure, values, patterns, and column scores will appear here.</p>
                </div>
              )}
            </aside>
          </section>
        </>
      ) : (
        <div className="empty-state">
          <Grid3X3 size={36} />
          <h2>Select a completed run</h2>
          <p>Both retained search repertoire and reported output remain inspectable.</p>
        </div>
      )}
      {error ? <div className="toast error">{error}</div> : null}
    </main>
  );
}

function Metric({ label, value, digits = 3 }: { label: string; value: number; digits?: number }) {
  return (
    <div>
      <small>{label}</small>
      <strong>{Number.isFinite(value) ? value.toFixed(digits) : "-"}</strong>
    </div>
  );
}

function accuracyKey(
  runIdentifier: string,
  kind: "raw" | "selected",
  analysisName: string
): string {
  return `${runIdentifier}:${kind}:${analysisName}`;
}

function MatrixFragment({
  matrix,
  onRowsChanged,
  onColumnsChanged
}: {
  matrix: JsonObject | null;
  onRowsChanged: (offset: number) => void;
  onColumnsChanged: (offset: number) => void;
}) {
  if (!matrix || !Array.isArray(matrix.values) || !Array.isArray(matrix.columns)) return null;
  const values = matrix.values as unknown[][];
  const columns = matrix.columns as JsonObject[];
  const rowOffset = Number(matrix.row_offset);
  const columnOffset = Number(matrix.column_offset);
  const totalRows = Number(matrix.total_rows);
  const totalColumns = Number(matrix.total_columns);
  return (
    <section>
      <div className="matrix-heading">
        <h3>Bicluster matrix</h3>
        <div className="matrix-pagers">
          <span>Rows</span>
          <button
            className="icon-button"
            disabled={rowOffset === 0}
            title="Previous rows"
            onClick={() => onRowsChanged(Math.max(0, rowOffset - 30))}
          >
            <ChevronLeft size={16} />
          </button>
          <button
            className="icon-button"
            disabled={rowOffset + values.length >= totalRows}
            title="Next rows"
            onClick={() => onRowsChanged(rowOffset + 30)}
          >
            <ChevronRight size={16} />
          </button>
          <span>Columns</span>
          <button
            className="icon-button"
            disabled={columnOffset === 0}
            title="Previous columns"
            onClick={() => onColumnsChanged(Math.max(0, columnOffset - 20))}
          >
            <ChevronLeft size={16} />
          </button>
          <button
            className="icon-button"
            disabled={columnOffset + columns.length >= totalColumns}
            title="Next columns"
            onClick={() => onColumnsChanged(columnOffset + 20)}
          >
            <ChevronRight size={16} />
          </button>
        </div>
      </div>
      <div
        className="matrix-grid"
        style={{ gridTemplateColumns: `70px repeat(${columns.length}, minmax(70px, 1fr))` }}
      >
        <span />
        {columns.map((column) => (
          <strong key={String(column.index)} title={String(column.kind)}>
            {String(column.name)}
          </strong>
        ))}
        {values.flatMap((row, rowIndex) => [
          <b key={`row-${rowIndex}`}>{String((matrix.row_indices as unknown[])[rowIndex])}</b>,
          ...row.map((value, columnIndex) => (
            <span
              key={`${rowIndex}-${columnIndex}`}
              className={`matrix-cell ${value == null ? "missing" : typeof value}`}
              title={value == null ? "Missing" : String(value)}
            >
              {value == null ? "NA" : String(value)}
            </span>
          ))
        ])}
      </div>
    </section>
  );
}

function ColumnExplanations({ detail }: { detail: JsonObject }) {
  if (!Array.isArray(detail.columns)) return null;
  const columns = detail.columns as JsonObject[];
  return <ColumnExplanationTable columns={columns} />;
}

function ColumnExplanationTable({ columns }: { columns: JsonObject[] }) {
  const [query, setQuery] = useState("");
  const [pattern, setPattern] = useState("");
  const patterns = [
    ...new Set(
      columns.map((column) => {
        const fit = column.pattern_fit as JsonObject | null;
        return fit ? String(fit.pattern) : "UNASSIGNED";
      })
    )
  ].sort();
  const patternCounts = new Map<string, number>();
  columns.forEach((column) => {
    const fit = column.pattern_fit as JsonObject | null;
    const name = fit ? String(fit.pattern) : "UNASSIGNED";
    patternCounts.set(name, (patternCounts.get(name) ?? 0) + 1);
  });
  const filtered = columns
    .filter((column) => {
      const fit = column.pattern_fit as JsonObject | null;
      const assigned = fit ? String(fit.pattern) : "UNASSIGNED";
      return (
        (!pattern || assigned === pattern) &&
        (!query ||
          String(column.name).toLowerCase().includes(query.toLowerCase()) ||
          String(column.kind).toLowerCase().includes(query.toLowerCase()))
      );
    })
    .sort((left, right) => {
      const leftFit = left.pattern_fit as JsonObject | null;
      const rightFit = right.pattern_fit as JsonObject | null;
      return Number(rightFit?.error ?? -1) - Number(leftFit?.error ?? -1);
    });
  return (
    <section className="column-diagnostics">
      <div className="column-diagnostics-heading">
        <div>
          <h3>Column pattern diagnostics</h3>
          <p className="section-description">
            Fit error measures mismatch with the inferred pattern (0 is best). Observed support
            reports usable selected rows. The parameter is the fitted prototype, offset, or
            factor for that pattern.
          </p>
        </div>
        <div className="pattern-summary">
          {[...patternCounts].map(([name, count]) => (
            <span key={name}>
              <i>{name}</i>
              <strong>{count}</strong>
            </span>
          ))}
        </div>
      </div>
      <div className="column-diagnostic-filters">
        <label>
          <Search size={14} />
          <input
            value={query}
            placeholder="Find a column"
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <select value={pattern} onChange={(event) => setPattern(event.target.value)}>
          <option value="">All patterns</option>
          {patterns.map((name) => (
            <option value={name} key={name}>
              {name}
            </option>
          ))}
        </select>
        <span>
          {filtered.length} of {columns.length} columns
        </span>
      </div>
      <div className="column-diagnostic-table-wrap">
        <table className="column-diagnostic-table">
          <thead>
            <tr>
              <th>Column</th>
              <th>Pattern</th>
              <th title="Pattern mismatch: 0 is a perfect fit and 1 is the worst valid score.">
                Fit error
              </th>
              <th title="Observed selected rows divided by rows available for this column.">
                Observed support
              </th>
              <th title="Pattern-specific fitted prototype, offset, or multiplicative factor.">
                Parameter
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((column) => {
              const fit = column.pattern_fit as JsonObject | null;
              const sourceSupport = Number(fit?.source_support ?? 0);
              const availableSupport = Number(fit?.available_support ?? 0);
              const supportRatio =
                availableSupport > 0 ? Math.min(1, sourceSupport / availableSupport) : 0;
              return (
                <tr key={String(column.index)}>
                  <td>
                    <strong>{String(column.name)}</strong>
                    <small>{humanizeMetricName(String(column.kind))}</small>
                  </td>
                  <td>
                    <span className="pattern-badge">
                      {fit ? String(fit.pattern) : "UNASSIGNED"}
                    </span>
                  </td>
                  <td>
                    <span className="fit-error">
                      <i
                        style={{
                          width: `${Math.max(0, Math.min(1, Number(fit?.error ?? 1))) * 100}%`
                        }}
                      />
                    </span>
                    <strong>{fit ? Number(fit.error).toFixed(4) : "-"}</strong>
                  </td>
                  <td>
                    <span className="support-track">
                      <i style={{ width: `${supportRatio * 100}%` }} />
                    </span>
                    <strong>
                      {fit ? `${sourceSupport} / ${availableSupport}` : "-"}
                    </strong>
                  </td>
                  <td title={fit?.parameter == null ? "" : String(fit.parameter)}>
                    {formatPatternParameter(fit?.parameter)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function formatPatternParameter(value: unknown): string {
  if (value == null) return "-";
  if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(5) : "-";
  if (typeof value === "string" || typeof value === "boolean") return String(value);
  const serialized = JSON.stringify(value);
  return serialized.length > 42 ? `${serialized.slice(0, 39)}…` : serialized;
}

function EvaluationMetadata({ detail }: { detail: JsonObject }) {
  return (
    <section>
      <h3>Evaluation and provenance</h3>
      <div className="metadata-grid">
        <MetadataBlock title="Objectives" value={detail.objectives} />
        <MetadataBlock title="Constraints" value={detail.constraints} />
        <MetadataBlock title="Descriptors" value={detail.descriptors} />
        <MetadataBlock title="Provenance" value={detail.provenance} />
      </div>
    </section>
  );
}

function MetadataBlock({ title, value }: { title: string; value: unknown }) {
  return (
    <div>
      <strong>{title}</strong>
      <pre>{value == null ? "Not recorded" : JSON.stringify(value, null, 2)}</pre>
    </div>
  );
}
