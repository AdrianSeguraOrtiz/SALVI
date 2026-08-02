import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  CircleAlert,
  Clock3,
  Maximize2,
  Minimize2,
  OctagonX,
  PanelTopClose,
  PanelTopOpen,
  RefreshCw,
  Trash2
} from "lucide-react";
import { api } from "../api";
import type {
  Catalog,
  ComponentDescription,
  JsonObject,
  Metric,
  ObserverView,
  RunRecord
} from "../types";
import {
  appendMetricHistory,
  eventFailureMessage,
  humanizeMetricName,
  latestMetricsByName
} from "../monitoring";
import { ObserverMetricPanel } from "./ObserverVisualizations";

interface Props {
  catalog: Catalog;
  runs: RunRecord[];
  selectedRun: string;
  onSelectedRun: (identifier: string) => void;
  onRunsChanged: () => Promise<void>;
}

type PanelMode = "normal" | "minimized" | "maximized";
const metricPageSize = 25_000;

const fallbackObserverView: ObserverView = {
  view_kind: "SERIES",
  title: "Observer metrics",
  metric_patterns: [],
  empty_message: "No metrics have been reported.",
  x_axis_label: "Evaluations",
  y_axis_label: "Value",
  metrics: [],
  groups: []
};

function observerView(component: ComponentDescription | undefined): ObserverView {
  return component?.observer_view ?? {
    ...fallbackObserverView,
    title: component?.title ?? fallbackObserverView.title
  };
}

export function Monitor({
  catalog,
  runs,
  selectedRun,
  onSelectedRun,
  onRunsChanged
}: Props) {
  const [run, setRun] = useState<RunRecord | null>(null);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [metricsSyncing, setMetricsSyncing] = useState(false);
  const [lastEvent, setLastEvent] = useState<JsonObject | null>(null);
  const [panelModes, setPanelModes] = useState<Record<string, PanelMode>>(() => {
    try {
      return JSON.parse(localStorage.getItem("salvi.monitor.panels") ?? "{}") as Record<
        string,
        PanelMode
      >;
    } catch {
      return {};
    }
  });
  const observers = run?.monitoring?.observers ?? [];
  const archiveAxes = (run?.monitoring?.archive_axes ?? []).map((name) =>
    humanizeMetricName(name)
  );
  const termination = run?.monitoring?.termination;
  const evaluationBudget =
    termination?.unit === "evaluations" && termination.limit != null ? termination.limit : null;
  const latest = useMemo(() => latestMetricsByName(metrics), [metrics]);
  const evaluationMetric = latest.get("search.evaluations");
  const observerItems = observers.map((name) => ({
    name,
    component: catalog.components.find(
      (item) => item.kind === "observer" && item.name === name
    )
  }));
  const minimizedItems = observerItems.filter(
    (item) => (panelModes[item.name] ?? "normal") === "minimized"
  );
  const visibleItems = observerItems.filter(
    (item) => (panelModes[item.name] ?? "normal") !== "minimized"
  );

  useEffect(() => {
    if (!selectedRun) {
      setRun(null);
      setMetrics([]);
      setMetricsSyncing(false);
      setLastEvent(null);
      return;
    }
    setRun(null);
    setMetrics([]);
    setMetricsSyncing(true);
    setLastEvent(null);
    let active = true;
    let refreshing = false;
    let metricSequence = 0;
    let source: EventSource | null = null;

    async function refresh() {
      if (refreshing) return;
      refreshing = true;
      if (active) setMetricsSyncing(true);
      try {
        const record = await api.run(selectedRun);
        if (!active) return;
        setRun(record);
        const incoming: Metric[] = [];
        while (active) {
          const response = await api.metrics(selectedRun, metricSequence);
          if (!active) return;
          if (response.items.length === 0) break;
          incoming.push(...response.items);
          metricSequence = response.items.at(-1)?.sequence ?? metricSequence;
          if (response.items.length < metricPageSize) break;
        }
        if (incoming.length > 0) {
          setMetrics((current) => appendMetricHistory(current, incoming));
        }
        if (record.status !== "running") await onRunsChanged();
      } finally {
        refreshing = false;
        if (active) setMetricsSyncing(false);
      }
    }

    refresh().catch(() => undefined);
    const timer = window.setInterval(() => refresh().catch(() => undefined), 1500);
    source = new EventSource(`/api/v1/runs/${encodeURIComponent(selectedRun)}/stream`);
    const receiveEvent = (event: Event) => {
      try {
        setLastEvent(JSON.parse((event as MessageEvent).data) as JsonObject);
      } catch {
        setLastEvent(null);
      }
    };
    source.addEventListener("run-event", receiveEvent);
    return () => {
      active = false;
      window.clearInterval(timer);
      source?.close();
    };
  }, [onRunsChanged, selectedRun]);

  function panelMode(name: string) {
    return panelModes[name] ?? "normal";
  }

  function setMode(name: string, mode: PanelMode) {
    setPanelModes((current) => {
      const next = { ...current, [name]: mode };
      localStorage.setItem("salvi.monitor.panels", JSON.stringify(next));
      return next;
    });
  }

  const elapsed =
    run?.started_at == null
      ? 0
      : ((run.finished_at ? Date.parse(run.finished_at) : Date.now()) -
          Date.parse(run.started_at)) /
        1000;
  const failureMessage =
    run?.status === "failed"
      ? run.error ??
        eventFailureMessage(lastEvent) ??
        "The run failed without recording an error message."
      : null;

  return (
    <main className="workspace monitor-workspace">
      <header className="workspace-header">
        <div>
          <span className="eyebrow">Live execution</span>
          <h1>Monitor</h1>
          <p>Durable events and observer metrics streamed from each run's SQLite store.</p>
        </div>
        <label className="compact-select">
          <span>Run</span>
          <select value={selectedRun} onChange={(event) => onSelectedRun(event.target.value)}>
            <option value="">Select a run</option>
            {runs.map((item) => (
              <option value={item.identifier} key={item.identifier}>
                {item.identifier} · {item.status}
              </option>
            ))}
          </select>
        </label>
      </header>

      {run ? (
        <>
          <section className="run-strip">
            <div className={`status-pulse ${run.status}`}>
              <Activity size={18} />
            </div>
            <div>
              <small>Status</small>
              <strong>{run.status}</strong>
            </div>
            <div>
              <small>Search evaluations</small>
              <strong>{evaluationMetric ? Math.round(evaluationMetric.value) : "..."}</strong>
            </div>
            <div>
              <small>Elapsed</small>
              <strong>
                <Clock3 size={15} /> {elapsed.toFixed(1)} s
              </strong>
            </div>
            <div>
              <small>Stored metric points</small>
              <strong>
                {metricsSyncing ? "Syncing..." : metrics.length.toLocaleString()}
              </strong>
            </div>
            <div className="run-message">
              <small>Latest event</small>
              <strong>{String(lastEvent?.event_type ?? "Waiting for events")}</strong>
            </div>
            {run.status === "running" ? (
              <button
                className="button danger"
                type="button"
                onClick={() => api.cancelRun(run.identifier).then(onRunsChanged)}
              >
                <OctagonX size={17} /> Cancel
              </button>
            ) : (
              <div className="button-row">
                <button className="icon-button" title="Refresh" onClick={onRunsChanged}>
                  <RefreshCw size={18} />
                </button>
                <button
                  className="icon-button"
                  title={`Delete ${run.identifier}`}
                  onClick={() => api.deleteRun(run.identifier).then(onRunsChanged)}
                >
                  <Trash2 size={17} />
                </button>
              </div>
            )}
          </section>

          {failureMessage ? (
            <section className="run-failure" role="alert" aria-live="assertive">
              <CircleAlert size={21} aria-hidden="true" />
              <div>
                <strong>Execution failed</strong>
                <p>{failureMessage}</p>
              </div>
            </section>
          ) : null}

          {minimizedItems.length > 0 ? (
            <section className="observer-dock" aria-label="Minimized observer panels">
              <div className="observer-dock-label">
                <PanelTopClose size={16} />
                <span>Minimized</span>
                <small>{minimizedItems.length}</small>
              </div>
              <div className="observer-dock-items">
                {minimizedItems.map(({ name, component }) => {
                  const view = observerView(component);
                  return (
                    <button
                      className="observer-dock-item"
                      type="button"
                      key={name}
                      onClick={() => setMode(name, "normal")}
                    >
                      <PanelTopOpen size={15} />
                      <span>{view.title}</span>
                    </button>
                  );
                })}
              </div>
            </section>
          ) : null}

          <section className="observer-grid">
            {visibleItems.map(({ name, component }) => {
              const mode = panelMode(name);
              const view = observerView(component);
              const selectedMetrics = metrics.filter((metric) =>
                view.metric_patterns.some((pattern) =>
                  metric.name.startsWith(pattern.replace("*", ""))
                )
              );
              return (
                <article
                  className={`observer-panel ${mode}`}
                  key={name}
                  aria-label={component?.title ?? name}
                >
                  <header>
                    <div className="observer-heading">
                      <h3>{view.title}</h3>
                    </div>
                    <div className="panel-actions">
                      <button
                        className="icon-button"
                        title="Minimize"
                        onClick={() => setMode(name, "minimized")}
                      >
                        <Minimize2 size={16} />
                      </button>
                      <button
                        className="icon-button"
                        title={mode === "maximized" ? "Restore" : "Maximize"}
                        onClick={() => setMode(name, mode === "maximized" ? "normal" : "maximized")}
                      >
                        <Maximize2 size={16} />
                      </button>
                    </div>
                  </header>
                  <ObserverMetricPanel
                    metrics={selectedMetrics}
                    view={view}
                    archiveAxes={archiveAxes}
                    evaluationBudget={evaluationBudget}
                  />
                </article>
              );
            })}
          </section>
        </>
      ) : (
        <div className="empty-state">
          <Activity size={36} />
          <h2>Select a run to inspect its event stream</h2>
          <p>Active and completed runs remain available in local history.</p>
        </div>
      )}
    </main>
  );
}
