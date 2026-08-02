import { useEffect, useMemo, useState } from "react";
import {
  humanizeMetricName,
  latestMetricsByName,
  metricPresentation,
  seriesLabel
} from "../monitoring";
import type { Metric, ObserverMetricPresentation, ObserverView } from "../types";
import { Chart } from "./Chart";

const colors = ["#0b7a75", "#ef765f", "#d69e2e", "#416b79", "#7d8f88", "#8f5d9a"];
const distributionStatistics = new Set([
  "minimum",
  "first_quartile",
  "median",
  "mean",
  "third_quartile",
  "maximum"
]);

function MetricDescription({
  view,
  group,
  definition
}: {
  view: ObserverView;
  group: string;
  definition?: ObserverMetricPresentation;
}) {
  const groupPresentation = view.groups.find((item) => item.name === group);
  const description = definition?.description ?? groupPresentation?.description;
  return description ? <p className="metric-description">{description}</p> : null;
}

export function ObserverMetricPanel({
  metrics,
  view,
  archiveAxes,
  evaluationBudget
}: {
  metrics: Metric[];
  view: ObserverView;
  archiveAxes: string[];
  evaluationBudget: number | null;
}) {
  if (metrics.length === 0) {
    return (
      <div className="metric-panel-content">
        <MetricDescription view={view} group={view.groups[0]?.name ?? ""} />
        <div className="panel-empty">{view.empty_message}</div>
      </div>
    );
  }
  if (view.view_kind === "PROGRESS") {
    return <RunProgress metrics={metrics} view={view} budget={evaluationBudget} />;
  }
  if (view.view_kind === "TABLE") return <MetricTable metrics={metrics} view={view} />;
  if (view.view_kind === "QD_DIAGNOSTICS") {
    return <QdArchiveDiagnostics metrics={metrics} axes={archiveAxes} view={view} />;
  }
  if (view.view_kind === "DISTRIBUTION") {
    return (
      <DistributionEvolution
        metrics={metrics}
        view={view}
        xAxisLabel={view.x_axis_label ?? "Evaluations"}
        yAxisLabel={view.y_axis_label ?? "Value"}
      />
    );
  }
  if (view.view_kind === "GROUPED_SERIES") {
    return (
      <GroupedSeriesChart
        metrics={metrics}
        view={view}
        xAxisLabel={view.x_axis_label ?? "Evaluations"}
        yAxisLabel={view.y_axis_label ?? "Value"}
      />
    );
  }
  return (
    <SeriesChart
      metrics={metrics}
      view={view}
      showKpis={view.view_kind === "KPI_SERIES"}
      stacked={view.view_kind === "STACKED_SERIES"}
      xAxisLabel={view.x_axis_label ?? "Evaluations"}
      yAxisLabel={view.y_axis_label ?? "Value"}
    />
  );
}

function RunProgress({
  metrics,
  view,
  budget
}: {
  metrics: Metric[];
  view: ObserverView;
  budget: number | null;
}) {
  const evaluations = latestMetricsByName(metrics).get("search.evaluations")?.value ?? 0;
  const fraction = budget === null ? null : Math.min(1, evaluations / budget);
  return (
    <div className="run-progress-visualization">
      <div className="run-progress-values">
        <div>
          <small>Completed</small>
          <strong>{Math.round(evaluations).toLocaleString()}</strong>
          <span>evaluations</span>
        </div>
        {budget !== null ? (
          <>
            <div>
              <small>Budget</small>
              <strong>{budget.toLocaleString()}</strong>
              <span>evaluations</span>
            </div>
            <div>
              <small>Remaining</small>
              <strong>{Math.max(0, budget - evaluations).toLocaleString()}</strong>
              <span>{((fraction ?? 0) * 100).toFixed(1)}% complete</span>
            </div>
          </>
        ) : null}
      </div>
      {fraction !== null ? (
        <div
          className="run-progress-track"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={budget ?? 0}
          aria-valuenow={evaluations}
          aria-label="Evaluation budget progress"
        >
          <span style={{ width: `${fraction * 100}%` }} />
        </div>
      ) : null}
      <MetricDescription view={view} group="progress" />
    </div>
  );
}

function SeriesChart({
  metrics,
  view,
  showKpis,
  stacked,
  xAxisLabel,
  yAxisLabel
}: {
  metrics: Metric[];
  view: ObserverView;
  showKpis: boolean;
  stacked: boolean;
  xAxisLabel: string;
  yAxisLabel: string;
}) {
  const groups = useMemo(
    () => [
      ...new Set(
        metrics.map(
          (metric) =>
            metricPresentation(metric.name, view.metrics)?.display_group ?? "reported metrics"
        )
      )
    ],
    [metrics, view.metrics]
  );
  const [group, setGroup] = useState(groups[0] ?? "");
  useEffect(() => {
    if (!groups.includes(group)) setGroup(groups[0] ?? "");
  }, [group, groups]);
  const selectedMetrics = metrics.filter(
    (metric) =>
      (metricPresentation(metric.name, view.metrics)?.display_group ?? "reported metrics") === group
  );
  const names = [...new Set(selectedMetrics.map((metric) => metric.name))];
  const latest = latestMetricsByName(metrics);
  const definitions = view.metrics.filter((definition) => definition.display_group === group);
  const ratioOnly =
    definitions.length > 0 && definitions.every((definition) => definition.unit === "ratio");
  const snapshotOnly =
    names.length > 0 &&
    names.every(
      (name) => selectedMetrics.filter((metric) => metric.name === name).length === 1
    );
  return (
    <div className="metric-panel-content">
      <div className="metric-context-row">
        <MetricDescription view={view} group={group} />
        {groups.length > 1 ? (
          <label className="visualization-control">
            <span>View</span>
            <select value={group} onChange={(event) => setGroup(event.target.value)}>
              {groups.map((name) => (
                <option value={name} key={name}>
                  {view.groups.find((item) => item.name === name)?.label ??
                    humanizeMetricName(name)}
                </option>
              ))}
            </select>
          </label>
        ) : null}
      </div>
      {showKpis ? (
        <div className="metric-kpis">
          {names.slice(0, 4).map((name) => (
            <div key={name} title={humanizeMetricName(name)}>
              <small>{seriesLabel(name, view.metrics)}</small>
              <strong>{formatMetricValue(latest.get(name)?.value, ratioOnly)}</strong>
            </div>
          ))}
        </div>
      ) : null}
      {snapshotOnly ? (
        <SnapshotMetricChart
          names={names}
          latest={latest}
          definitions={view.metrics}
          valueAxisLabel={yAxisLabel}
        />
      ) : (
        <Chart
          height={280}
          option={{
            animation: false,
            color: colors,
            tooltip: { trigger: "axis" },
            legend: {
              type: "scroll",
              bottom: 0,
              formatter: (name: string) => name
            },
            grid: { top: 18, right: 28, bottom: 68, left: 92, containLabel: true },
            xAxis: {
              type: "value",
              name: xAxisLabel,
              nameLocation: "middle",
              nameGap: 30,
              nameTextStyle: { color: "#66736e" }
            },
            yAxis: {
              type: "value",
              name: yAxisLabel,
              nameLocation: "middle",
              nameGap: 62,
              nameTextStyle: { color: "#66736e" },
              scale: !ratioOnly,
              min: ratioOnly ? 0 : undefined,
              max: ratioOnly ? 1 : undefined,
              axisLabel: ratioOnly
                ? { formatter: (value: number) => `${Math.round(value * 100)}%` }
                : {}
            },
            series: names.map((name) => ({
              name: seriesLabel(name, view.metrics),
              type: "line",
              stack: stacked ? "pipeline-total" : undefined,
              areaStyle: stacked ? {} : undefined,
              symbol: "none",
              sampling: "lttb",
              data: selectedMetrics
                .filter((metric) => metric.name === name)
                .map((metric) => [metric.step ?? metric.sequence, metric.value])
            }))
          }}
        />
      )}
    </div>
  );
}

function SnapshotMetricChart({
  names,
  latest,
  definitions,
  valueAxisLabel
}: {
  names: string[];
  latest: Map<string, Metric>;
  definitions: ObserverMetricPresentation[];
  valueAxisLabel: string;
}) {
  const ordered = [...names].sort(
    (left, right) => (latest.get(right)?.value ?? 0) - (latest.get(left)?.value ?? 0)
  );
  const labels = new Map(
    ordered.map((name) => [
      name,
      humanizeMetricName(
        name.endsWith(".seconds") ? name.slice(0, -".seconds".length) : name,
        name.split(".").slice(0, 2).join(".")
      )
    ])
  );
  return (
    <div className="snapshot-metric-chart">
      <Chart
        height={Math.min(310, Math.max(190, ordered.length * 31 + 70))}
        option={{
          animation: false,
          color: [colors[0]],
          tooltip: {
            trigger: "axis",
            axisPointer: { type: "shadow" }
          },
          grid: { top: 8, right: 72, bottom: 48, left: 16, containLabel: true },
          xAxis: {
            type: "value",
            min: 0,
            name: valueAxisLabel,
            nameLocation: "middle",
            nameGap: 30,
            nameTextStyle: { color: "#66736e" }
          },
          yAxis: {
            type: "category",
            inverse: true,
            data: ordered.map((name) => labels.get(name) ?? seriesLabel(name, definitions)),
            axisLabel: {
              width: 180,
              overflow: "truncate"
            }
          },
          series: [
            {
              name: valueAxisLabel,
              type: "bar",
              barMaxWidth: 18,
              data: ordered.map((name) => latest.get(name)?.value ?? 0),
              itemStyle: { borderRadius: [0, 3, 3, 0] },
              label: {
                show: true,
                position: "right",
                color: "#45544f",
                formatter: (item) => `${Number(item.value).toPrecision(4)} s`
              }
            }
          ]
        }}
      />
    </div>
  );
}

function formatMetricValue(value: number | undefined, ratio: boolean): string {
  if (value === undefined) return "-";
  return ratio ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);
}

function MetricTable({ metrics, view }: { metrics: Metric[]; view: ObserverView }) {
  const latest = [...latestMetricsByName(metrics).values()].sort((left, right) =>
    left.name.localeCompare(right.name)
  );
  return (
    <div className="metric-table-wrap">
      <table className="metric-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Evaluation</th>
          </tr>
        </thead>
        <tbody>
          {latest.map((metric) => (
            <tr key={metric.name}>
              <td title={metric.name}>{seriesLabel(metric.name, view.metrics)}</td>
              <td>
                {formatMetricValue(
                  metric.value,
                  metricPresentation(metric.name, view.metrics)?.unit === "ratio"
                )}
              </td>
              <td>{metric.step ?? "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface DistributionPoint {
  step: number;
  minimum: number;
  firstQuartile: number;
  median: number;
  mean: number;
  thirdQuartile: number;
  maximum: number;
}

function distributionName(metricName: string): { name: string; statistic: string } | null {
  const parts = metricName.split(".");
  const statistic = parts.at(-1);
  if (!statistic || !distributionStatistics.has(statistic) || parts.length < 3) return null;
  return { name: parts.slice(0, -1).join("."), statistic };
}

function distributionPoints(metrics: Metric[], selected: string): DistributionPoint[] {
  const steps = new Map<number, Map<string, number>>();
  metrics.forEach((metric) => {
    const parsed = distributionName(metric.name);
    if (!parsed || parsed.name !== selected) return;
    const step = metric.step ?? metric.sequence;
    const values = steps.get(step) ?? new Map<string, number>();
    values.set(parsed.statistic, metric.value);
    steps.set(step, values);
  });
  return [...steps.entries()]
    .sort(([left], [right]) => left - right)
    .flatMap(([step, values]) => {
      const minimum = values.get("minimum");
      const median = values.get("median");
      const mean = values.get("mean");
      const maximum = values.get("maximum");
      if (
        minimum === undefined ||
        median === undefined ||
        mean === undefined ||
        maximum === undefined
      ) {
        return [];
      }
      return [
        {
          step,
          minimum,
          firstQuartile: values.get("first_quartile") ?? median,
          median,
          mean,
          thirdQuartile: values.get("third_quartile") ?? median,
          maximum
        }
      ];
    });
}

function DistributionEvolution({
  metrics,
  view,
  xAxisLabel,
  yAxisLabel
}: {
  metrics: Metric[];
  view: ObserverView;
  xAxisLabel: string;
  yAxisLabel: string;
}) {
  const distributions = useMemo(
    () =>
      [
        ...new Set(
          metrics
            .map((metric) => distributionName(metric.name))
            .filter((item) => item !== null)
            .map((item) => item.name)
        )
      ],
    [metrics]
  );
  const [selected, setSelected] = useState(distributions[0] ?? "");
  useEffect(() => {
    if (!distributions.includes(selected)) setSelected(distributions[0] ?? "");
  }, [distributions, selected]);
  const points = distributionPoints(metrics, selected);
  if (!selected || points.length === 0) return <MetricTable metrics={metrics} view={view} />;
  const definition = metricPresentation(`${selected}.mean`, view.metrics);
  return (
    <div className="metric-panel-content">
      <div className="metric-context-row">
        {definition ? (
          <MetricDescription
            view={view}
            group={definition.display_group}
            definition={definition}
          />
        ) : null}
        <label className="visualization-control">
          <span>Distribution</span>
          <select value={selected} onChange={(event) => setSelected(event.target.value)}>
            {distributions.map((name) => (
              <option value={name} key={name}>
                {humanizeMetricName(name, name.split(".")[0])}
              </option>
            ))}
          </select>
        </label>
      </div>
      <Chart
        height={310}
        option={{
          animation: false,
          color: [colors[0], colors[1]],
          tooltip: { trigger: "item" },
          legend: {
            bottom: 0,
            data: ["Batch distribution", "Mean"]
          },
          grid: { top: 18, right: 28, bottom: 72, left: 68, containLabel: true },
          xAxis: {
            type: "category",
            data: points.map((point) => String(point.step)),
            name: xAxisLabel,
            nameLocation: "middle",
            nameGap: 32,
            nameTextStyle: { color: "#66736e" }
          },
          yAxis: {
            type: "value",
            name: yAxisLabel,
            nameLocation: "middle",
            nameGap: 48,
            nameTextStyle: { color: "#66736e" },
            scale: true
          },
          dataZoom:
            points.length > 40
              ? [
                  { type: "inside", start: 0, end: 100 },
                  { type: "slider", height: 14, bottom: 34, start: 0, end: 100 }
                ]
              : undefined,
          series: [
            {
              name: "Batch distribution",
              type: "boxplot",
              data: points.map((point) => [
                point.minimum,
                point.firstQuartile,
                point.median,
                point.thirdQuartile,
                point.maximum
              ])
            },
            {
              name: "Mean",
              type: "scatter",
              symbolSize: 7,
              data: points.map((point) => point.mean)
            }
          ]
        }}
      />
    </div>
  );
}

function GroupedSeriesChart({
  metrics,
  view,
  xAxisLabel,
  yAxisLabel
}: {
  metrics: Metric[];
  view: ObserverView;
  xAxisLabel: string;
  yAxisLabel: string;
}) {
  const parsed = useMemo(
    () =>
      metrics.flatMap((metric) => {
        const match = /^emitter\.([^.]+)\.([^.]+)$/.exec(metric.name);
        return match ? [{ metric, group: match[1], measure: match[2] }] : [];
      }),
    [metrics]
  );
  const measures = useMemo(() => [...new Set(parsed.map((item) => item.measure))], [parsed]);
  const [measure, setMeasure] = useState(measures.includes("credit") ? "credit" : measures[0] ?? "");
  useEffect(() => {
    if (!measures.includes(measure)) {
      setMeasure(measures.includes("credit") ? "credit" : measures[0] ?? "");
    }
  }, [measure, measures]);
  const groups = [...new Set(parsed.filter((item) => item.measure === measure).map((item) => item.group))];
  if (!measure || groups.length === 0) return <MetricTable metrics={metrics} view={view} />;
  const definition = metricPresentation(`emitter.example.${measure}`, view.metrics);
  return (
    <div className="metric-panel-content">
      <div className="metric-context-row">
        {definition ? (
          <MetricDescription
            view={view}
            group={definition.display_group}
            definition={definition}
          />
        ) : null}
        <label className="visualization-control">
          <span>Measure</span>
          <select value={measure} onChange={(event) => setMeasure(event.target.value)}>
            {measures.map((name) => (
              <option value={name} key={name}>
                {humanizeMetricName(name)}
              </option>
            ))}
          </select>
        </label>
      </div>
      <Chart
        height={290}
        option={{
          animation: false,
          color: colors,
          tooltip: { trigger: "axis" },
          legend: {
            type: "scroll",
            bottom: 0,
            formatter: (name: string) => humanizeMetricName(name)
          },
          grid: { top: 18, right: 28, bottom: 68, left: 68, containLabel: true },
          xAxis: {
            type: "value",
            name: xAxisLabel,
            nameLocation: "middle",
            nameGap: 30,
            nameTextStyle: { color: "#66736e" }
          },
          yAxis: {
            type: "value",
            name: `${yAxisLabel}: ${definition?.label ?? humanizeMetricName(measure)}`,
            nameLocation: "middle",
            nameGap: 48,
            nameTextStyle: { color: "#66736e" },
            scale: true
          },
          series: groups.map((group) => ({
            name: group,
            type: "line",
            symbol: "none",
            sampling: "lttb",
            data: parsed
              .filter((item) => item.group === group && item.measure === measure)
              .map((item) => [item.metric.step ?? item.metric.sequence, item.metric.value])
          }))
        }}
      />
    </div>
  );
}

interface ArchiveCellMetric {
  coordinate: number[];
  coordinateLabel: string;
  measure: string;
  value: number;
}

function QdArchiveDiagnostics({
  metrics,
  axes,
  view
}: {
  metrics: Metric[];
  axes: string[];
  view: ObserverView;
}) {
  const latest = useMemo(() => [...latestMetricsByName(metrics).values()], [metrics]);
  const cells = useMemo(
    () =>
      latest.flatMap((metric): ArchiveCellMetric[] => {
        const match = /^qd\.cell\.([^.]+)\.([^.]+)$/.exec(metric.name);
        if (!match) return [];
        const coordinate = match[1].split("_").map(Number);
        if (coordinate.length !== 2 || coordinate.some((value) => !Number.isInteger(value))) {
          return [];
        }
        return [
          {
            coordinate,
            coordinateLabel: match[1],
            measure: match[2],
            value: metric.value
          }
        ];
      }),
    [latest]
  );
  return cells.length > 0 ? (
    <QdCellHeatmap metrics={metrics} axes={axes} view={view} cells={cells} />
  ) : (
    <QdCellSummary metrics={metrics} view={view} />
  );
}

function QdCellSummary({ metrics, view }: { metrics: Metric[]; view: ObserverView }) {
  const latest = useMemo(() => [...latestMetricsByName(metrics).values()], [metrics]);
  const distributions = useMemo(
    () =>
      [
        ...new Set(
          metrics.flatMap((metric) => {
            const item = distributionName(metric.name);
            return item?.name.startsWith("qd.cell_") ? [item.name] : [];
          })
        )
      ],
    [metrics]
  );
  const preferred = "qd.cell_acceptance_ratio";
  const [selected, setSelected] = useState(
    distributions.includes(preferred) ? preferred : distributions[0] ?? ""
  );
  useEffect(() => {
    if (!distributions.includes(selected)) {
      setSelected(distributions.includes(preferred) ? preferred : distributions[0] ?? "");
    }
  }, [distributions, selected]);
  const points = distributionPoints(metrics, selected);
  if (!selected || points.length === 0) return <MetricTable metrics={metrics} view={view} />;

  const definition = metricPresentation(`${selected}.mean`, view.metrics);
  const ratio = definition?.unit === "ratio";
  const summary = latest.filter((metric) =>
    ["qd.visited_cells", "qd.unmapped_attempts"].includes(metric.name)
  );
  const zoomed = points.length > 40;
  return (
    <div className="metric-panel-content">
      <div className="heatmap-toolbar">
        <div className="metric-kpis compact">
          {summary.map((metric) => (
            <div key={metric.name}>
              <small>{seriesLabel(metric.name, view.metrics)}</small>
              <strong>{metric.value.toFixed(0)}</strong>
            </div>
          ))}
        </div>
        <label className="visualization-control">
          <span>Cell summary</span>
          <select value={selected} onChange={(event) => setSelected(event.target.value)}>
            {distributions.map((name) => (
              <option value={name} key={name}>
                {humanizeMetricName(name, "qd")}
              </option>
            ))}
          </select>
        </label>
      </div>
      {definition ? (
        <MetricDescription
          view={view}
          group={definition.display_group}
          definition={definition}
        />
      ) : null}
      <Chart
        height={zoomed ? 360 : 330}
        option={{
          animation: false,
          tooltip: {
            trigger: "axis",
            formatter: (parameters: unknown) =>
              qdSummaryTooltip(parameters, points, ratio, definition?.unit ?? "value")
          },
          legend: {
            bottom: 0,
            data: ["Minimum to maximum", "Interquartile range", "Median", "Mean"]
          },
          grid: {
            top: 18,
            right: 28,
            bottom: zoomed ? 106 : 74,
            left: 72,
            containLabel: true
          },
          xAxis: {
            type: "category",
            data: points.map((point) => String(point.step)),
            name: "Evaluations",
            nameLocation: "middle",
            nameGap: 32,
            nameTextStyle: { color: "#66736e" }
          },
          yAxis: {
            type: "value",
            name: definition?.label ?? "Cell metric",
            nameLocation: "middle",
            nameGap: 52,
            nameTextStyle: { color: "#66736e" },
            axisLabel: ratio
              ? { formatter: (value: number) => `${(value * 100).toFixed(0)}%` }
              : undefined,
            scale: true
          },
          dataZoom: zoomed
            ? [
                { type: "inside", start: 0, end: 100 },
                { type: "slider", height: 14, bottom: 36, start: 0, end: 100 }
              ]
            : undefined,
          series: [
            {
              name: "Minimum baseline",
              type: "line",
              stack: "full-range",
              symbol: "none",
              silent: true,
              lineStyle: { opacity: 0 },
              areaStyle: { opacity: 0 },
              data: points.map((point) => point.minimum)
            },
            {
              name: "Minimum to maximum",
              type: "line",
              stack: "full-range",
              symbol: "none",
              silent: true,
              lineStyle: { opacity: 0 },
              areaStyle: { color: "#d8ebe8", opacity: 0.55 },
              data: points.map((point) => point.maximum - point.minimum)
            },
            {
              name: "Quartile baseline",
              type: "line",
              stack: "quartile-range",
              symbol: "none",
              silent: true,
              lineStyle: { opacity: 0 },
              areaStyle: { opacity: 0 },
              data: points.map((point) => point.firstQuartile)
            },
            {
              name: "Interquartile range",
              type: "line",
              stack: "quartile-range",
              symbol: "none",
              silent: true,
              lineStyle: { opacity: 0 },
              areaStyle: { color: "#75bdb4", opacity: 0.55 },
              data: points.map((point) => point.thirdQuartile - point.firstQuartile)
            },
            {
              name: "Median",
              type: "line",
              symbol: "none",
              lineStyle: { color: colors[0], width: 2 },
              itemStyle: { color: colors[0] },
              data: points.map((point) => point.median)
            },
            {
              name: "Mean",
              type: "line",
              symbol: "none",
              lineStyle: { color: colors[1], width: 2, type: "dashed" },
              itemStyle: { color: colors[1] },
              data: points.map((point) => point.mean)
            }
          ]
        }}
      />
    </div>
  );
}

function qdSummaryTooltip(
  parameters: unknown,
  points: DistributionPoint[],
  ratio: boolean,
  unit: string
): string {
  if (!Array.isArray(parameters) || parameters.length === 0) return "";
  const first = parameters[0];
  if (
    typeof first !== "object" ||
    first === null ||
    !("dataIndex" in first) ||
    typeof first.dataIndex !== "number"
  ) {
    return "";
  }
  const point = points[first.dataIndex];
  if (!point) return "";
  const value = (raw: number) =>
    ratio ? `${(raw * 100).toFixed(1)}%` : `${raw.toFixed(2)} ${unit}`;
  return [
    `<strong>Evaluation ${point.step.toLocaleString()}</strong>`,
    `Minimum: ${value(point.minimum)}`,
    `First quartile: ${value(point.firstQuartile)}`,
    `Median: ${value(point.median)}`,
    `Mean: ${value(point.mean)}`,
    `Third quartile: ${value(point.thirdQuartile)}`,
    `Maximum: ${value(point.maximum)}`
  ].join("<br/>");
}

function QdCellHeatmap({
  metrics,
  axes,
  view,
  cells
}: {
  metrics: Metric[];
  axes: string[];
  view: ObserverView;
  cells: ArchiveCellMetric[];
}) {
  const latest = useMemo(() => [...latestMetricsByName(metrics).values()], [metrics]);
  const measures = useMemo(() => [...new Set(cells.map((item) => item.measure))], [cells]);
  const [measure, setMeasure] = useState(
    measures.includes("acceptance_ratio") ? "acceptance_ratio" : measures[0] ?? ""
  );
  useEffect(() => {
    if (!measures.includes(measure)) {
      setMeasure(measures.includes("acceptance_ratio") ? "acceptance_ratio" : measures[0] ?? "");
    }
  }, [measure, measures]);
  if (!measure) return <MetricTable metrics={metrics} view={view} />;
  const selectedCells = cells.filter((cell) => cell.measure === measure);
  const definition = metricPresentation(`qd.cell.0_0.${measure}`, view.metrics);
  const xBins = [...new Set(selectedCells.map((cell) => cell.coordinate[0]))].sort((a, b) => a - b);
  const yBins = [...new Set(selectedCells.map((cell) => cell.coordinate[1]))].sort((a, b) => a - b);
  const values = selectedCells.map((cell) => cell.value);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const summary = latest.filter((metric) =>
    ["qd.visited_cells", "qd.unmapped_attempts"].includes(metric.name)
  );
  return (
    <div className="metric-panel-content">
      <div className="heatmap-toolbar">
        <div className="metric-kpis compact">
          {summary.map((metric) => (
            <div key={metric.name}>
              <small>{humanizeMetricName(metric.name, "qd")}</small>
              <strong>{metric.value.toFixed(0)}</strong>
            </div>
          ))}
        </div>
        <label className="visualization-control">
          <span>Cell diagnostic</span>
          <select value={measure} onChange={(event) => setMeasure(event.target.value)}>
            {measures.map((name) => (
              <option value={name} key={name}>
                {humanizeMetricName(name)}
              </option>
            ))}
          </select>
        </label>
      </div>
      {definition ? (
        <MetricDescription
          view={view}
          group={definition.display_group}
          definition={definition}
        />
      ) : null}
      <Chart
        height={340}
        option={{
          animation: false,
          tooltip: { position: "top" },
          grid: { top: 18, right: 32, bottom: 82, left: 86, containLabel: true },
          xAxis: {
            type: "category",
            data: xBins.map(String),
            name: `${axes[0] ?? "First descriptor"} bin`,
            nameLocation: "middle",
            nameGap: 34,
            nameTextStyle: { color: "#66736e" }
          },
          yAxis: {
            type: "category",
            data: yBins.map(String),
            name: `${axes[1] ?? "Second descriptor"} bin`,
            nameLocation: "middle",
            nameGap: 48,
            nameTextStyle: { color: "#66736e" }
          },
          visualMap: {
            min: minimum,
            max: minimum === maximum ? minimum + 1 : maximum,
            dimension: 2,
            calculable: true,
            orient: "horizontal",
            left: "center",
            bottom: 0,
            text: [definition?.label ?? humanizeMetricName(measure), "Low"],
            inRange: { color: ["#d5e9e5", "#69b9ae", "#08736e"] }
          },
          series: [
            {
              name: humanizeMetricName(measure),
              type: "heatmap",
              data: selectedCells.map((cell) => ({
                name:
                  `${axes[0] ?? "Descriptor 1"} bin ${cell.coordinate[0]}, ` +
                  `${axes[1] ?? "Descriptor 2"} bin ${cell.coordinate[1]}`,
                value: [
                  xBins.indexOf(cell.coordinate[0]),
                  yBins.indexOf(cell.coordinate[1]),
                  cell.value,
                  cell.coordinateLabel
                ]
              })),
              emphasis: {
                itemStyle: {
                  borderColor: "#18231f",
                  borderWidth: 1
                }
              },
              itemStyle: {
                borderColor: "#ffffff",
                borderWidth: 1
              }
            }
          ]
        }}
      />
    </div>
  );
}
