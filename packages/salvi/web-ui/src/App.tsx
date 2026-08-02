import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Braces,
  CheckCircle2,
  CircleCheck,
  Download,
  FlaskConical,
  Grid3X3,
  Menu,
  Play,
  Plus,
  Settings2,
  Trash2,
  Upload,
  X
} from "lucide-react";
import { api } from "./api";
import { DatasetImport } from "./components/DatasetImport";
import { ComponentEditorBoundary } from "./components/ComponentEditorBoundary";
import { PipelineWorkflow } from "./components/PipelineWorkflow";
import { StageCatalog } from "./components/StageCatalog";
import type {
  Catalog,
  ComponentDescription,
  CompositionResolution,
  DatasetRecord,
  JsonObject,
  RoleResolution,
  RunRecord
} from "./types";

type View = "build" | "monitor" | "results";

const Monitor = lazy(() =>
  import("./components/Monitor").then((module) => ({ default: module.Monitor }))
);
const Results = lazy(() =>
  import("./components/Results").then((module) => ({ default: module.Results }))
);
const ParameterEditor = lazy(() =>
  import("./components/ParameterEditor").then((module) => ({
    default: module.ParameterEditor
  }))
);

function clone<T>(value: T): T {
  return structuredClone(value);
}

function getAtPath(root: JsonObject, path: string[]): unknown {
  let value: unknown = root;
  path.forEach((key) => {
    value = typeof value === "object" && value !== null ? (value as JsonObject)[key] : undefined;
  });
  return value;
}

function setAtPath(root: JsonObject, path: string[], value: unknown): JsonObject {
  const next = clone(root);
  let target = next;
  path.slice(0, -1).forEach((key) => {
    if (typeof target[key] !== "object" || target[key] === null) target[key] = {};
    target = target[key] as JsonObject;
  });
  target[path.at(-1)!] = value;
  return next;
}

function defaultParameters(component: ComponentDescription): JsonObject {
  return Object.fromEntries(
    component.parameters
      .filter((parameter) => parameter.default !== null && parameter.default !== undefined)
      .map((parameter) => [parameter.name, clone(parameter.default)])
  );
}

function configuredNames(configuration: JsonObject, role: RoleResolution): string[] {
  const value = getAtPath(configuration, role.role.configuration_path);
  const specifications = role.role.repeatable ? (Array.isArray(value) ? value : []) : [value];
  return specifications.flatMap((specification) =>
    typeof specification === "object" &&
    specification !== null &&
    "name" in specification &&
    typeof (specification as JsonObject).name === "string"
      ? [String((specification as JsonObject).name)]
      : []
  );
}

function downloadText(filename: string, content: string) {
  const url = URL.createObjectURL(new Blob([content], { type: "text/yaml" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function App() {
  const [view, setView] = useState<View>("build");
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [configuration, setConfiguration] = useState<JsonObject | null>(null);
  const [resolution, setResolution] = useState<CompositionResolution | null>(null);
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [dataset, setDataset] = useState("");
  const [selectedAnalyses, setSelectedAnalyses] = useState<string[]>([]);
  const [runIdentifier, setRunIdentifier] = useState("salvi-run");
  const [seed, setSeed] = useState(0);
  const [selectedRun, setSelectedRun] = useState("");
  const [selectedRole, setSelectedRole] = useState("");
  const [selectedStage, setSelectedStage] = useState("");
  const [selectedInstance, setSelectedInstance] = useState("");
  const [yamlOpen, setYamlOpen] = useState(false);
  const [yaml, setYaml] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<{ kind: "error" | "success"; text: string } | null>(null);
  const yamlUpload = useRef<HTMLInputElement>(null);

  const refreshDatasets = useCallback(async () => {
    const items = await api.datasets();
    setDatasets(items);
    setDataset((current) => current || items[0]?.identifier || "");
  }, []);
  const refreshRuns = useCallback(async () => {
    const items = await api.runs();
    setRuns(items);
    setSelectedRun((current) => current || items[0]?.identifier || "");
  }, []);

  useEffect(() => {
    Promise.all([api.catalog(), api.defaultPipeline(), api.datasets(), api.runs()])
      .then(async ([nextCatalog, defaultPipeline, nextDatasets, nextRuns]) => {
        const validated = await api.validatePipeline(defaultPipeline.yaml);
        setCatalog(nextCatalog);
        setConfiguration(validated.configuration);
        setYaml(validated.yaml);
        setDatasets(nextDatasets);
        setDataset(nextDatasets[0]?.identifier ?? "");
        setRuns(nextRuns);
        setSelectedRun(nextRuns[0]?.identifier ?? "");
      })
      .catch((cause) =>
        setNotice({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
      );
  }, []);

  useEffect(() => {
    if (!configuration) return;
    let current = true;
    const timer = window.setTimeout(() => {
      api
        .resolve(configuration)
        .then((next) => {
          if (current) setResolution(next);
        })
        .catch((cause) => {
          if (current) {
            setNotice({
              kind: "error",
              text: cause instanceof Error ? cause.message : String(cause)
            });
          }
        });
    }, 120);
    return () => {
      current = false;
      window.clearTimeout(timer);
    };
  }, [configuration]);

  const role = resolution?.roles.find((item) => item.role.kind === selectedRole) ?? null;
  const stage = catalog?.workflow_stages.find((item) => item.stage === selectedStage) ?? null;
  const stageRoles =
    resolution?.roles.filter((item) => item.role.stage === selectedStage) ?? [];
  const actualConfiguredNames = useMemo(
    () => (role && configuration ? configuredNames(configuration, role) : []),
    [configuration, role]
  );
  const configuredKey = actualConfiguredNames.join("\u0000");

  useEffect(() => {
    const configured = configuredKey ? configuredKey.split("\u0000") : [];
    setSelectedInstance((current) =>
      current && configured.includes(current) ? current : (configured[0] ?? "")
    );
  }, [configuredKey, selectedRole]);

  const configuredComponent = useMemo(() => {
    if (!role || !catalog || !configuration) return null;
    const name = actualConfiguredNames.includes(selectedInstance)
      ? selectedInstance
      : (actualConfiguredNames[0] ?? "");
    return catalog.components.find(
      (component) => component.kind === role.role.kind && component.name === name
    );
  }, [actualConfiguredNames, catalog, configuration, role, selectedInstance]);
  const analysisCatalog = useMemo(() => catalog?.analyses ?? [], [catalog]);
  const selectedDataset = datasets.find((item) => item.identifier === dataset) ?? null;
  const configuredAnalyses = analysisCatalog.filter((analysis) =>
    selectedAnalyses.includes(analysis.name)
  );

  useEffect(() => {
    setSelectedAnalyses((current) =>
      current.filter((name) => {
        const analysis = analysisCatalog.find((item) => item.name === name);
        return Boolean(
          analysis &&
            (!analysis.requires_ground_truth || selectedDataset?.ground_truth_attached)
        );
      })
    );
  }, [analysisCatalog, selectedDataset]);

  async function ensureYaml() {
    if (!configuration) throw new Error("No pipeline is loaded");
    const response = await api.serializePipeline(configuration);
    setYaml(response.yaml);
    return response.yaml;
  }

  async function startRun() {
    if (!configuration || !dataset) return;
    setBusy(true);
    setNotice(null);
    try {
      const effectiveYaml = await ensureYaml();
      const record = await api.startRun(
        effectiveYaml,
        dataset,
        runIdentifier,
        seed,
        selectedAnalyses
      );
      await refreshRuns();
      setSelectedRun(record.identifier);
      setView("monitor");
      setNotice({ kind: "success", text: `Run ${record.identifier} started.` });
    } catch (cause) {
      setNotice({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) });
    } finally {
      setBusy(false);
    }
  }

  function configureInstance(targetRole: RoleResolution, component: ComponentDescription) {
    if (!configuration) return;
    const path = targetRole.role.configuration_path;
    const spec = { name: component.name, parameters: defaultParameters(component) };
    const existing = getAtPath(configuration, path);
    if (targetRole.role.repeatable) {
      const values = Array.isArray(existing) ? existing : [];
      const without = values.filter(
        (item) =>
          !(typeof item === "object" && item !== null && (item as JsonObject).name === component.name)
      );
      setConfiguration(setAtPath(configuration, path, [...without, spec]));
    } else {
      setConfiguration(setAtPath(configuration, path, spec));
    }
    setSelectedInstance(component.name);
  }

  function removeConfigured(targetRole: RoleResolution, name?: string) {
    if (!configuration) return;
    const path = targetRole.role.configuration_path;
    if (targetRole.role.repeatable) {
      const existing = getAtPath(configuration, path);
      const values = Array.isArray(existing) ? existing : [];
      setConfiguration(
        setAtPath(
          configuration,
          path,
          values.filter(
            (item) =>
              !(typeof item === "object" && item !== null && (item as JsonObject).name === name)
          )
        )
      );
    } else {
      setConfiguration(setAtPath(configuration, path, null));
    }
    if (selectedInstance === name || !targetRole.role.repeatable) setSelectedInstance("");
  }

  function updateParameters(targetRole: RoleResolution, parameters: JsonObject) {
    if (!configuration || !configuredComponent) return;
    const path = targetRole.role.configuration_path;
    const existing = getAtPath(configuration, path);
    if (targetRole.role.repeatable) {
      const values = Array.isArray(existing) ? existing : [];
      setConfiguration(
        setAtPath(
          configuration,
          path,
          values.map((item) =>
            typeof item === "object" &&
            item !== null &&
            (item as JsonObject).name === configuredComponent.name
              ? { ...(item as JsonObject), parameters }
              : item
          )
        )
      );
    } else {
      setConfiguration(
        setAtPath(configuration, path, { name: configuredComponent.name, parameters })
      );
    }
  }

  function configuredParameters(targetRole: RoleResolution): JsonObject {
    if (!configuration || !configuredComponent) return {};
    const value = getAtPath(configuration, targetRole.role.configuration_path);
    const spec = Array.isArray(value)
      ? value.find(
          (item) =>
            typeof item === "object" &&
            item !== null &&
            (item as JsonObject).name === configuredComponent.name
        )
      : value;
    if (
      !Array.isArray(value) &&
      typeof spec === "object" &&
      spec !== null &&
      "name" in spec &&
      String((spec as JsonObject).name) !== configuredComponent.name
    ) {
      return {};
    }
    return typeof spec === "object" &&
      spec !== null &&
      typeof (spec as JsonObject).parameters === "object"
      ? ((spec as JsonObject).parameters as JsonObject)
      : {};
  }

  if (!catalog || !configuration) {
    return (
      <div className="boot-screen">
        <div className="salvi-mark">S</div>
        <p>Loading SALVI component catalog...</p>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <button className="brand" onClick={() => setView("build")}>
          <span className="salvi-mark">S</span>
          <span>
            <strong>SALVI</strong>
            <small>Quality-diversity biclustering</small>
          </span>
        </button>
        <nav>
          <NavButton active={view === "build"} onClick={() => setView("build")} icon={<Braces />}>
            Build
          </NavButton>
          <NavButton
            active={view === "monitor"}
            onClick={() => setView("monitor")}
            icon={<Activity />}
          >
            Monitor
          </NavButton>
          <NavButton
            active={view === "results"}
            onClick={() => setView("results")}
            icon={<Grid3X3 />}
          >
            Results
          </NavButton>
        </nav>
        <button className="icon-button mobile-menu" title="Menu">
          <Menu size={20} />
        </button>
      </header>

      {view === "build" ? (
        <main className="workspace build-workspace">
          <header className="workspace-header">
            <div>
              <span className="eyebrow">Algorithm composition</span>
              <h1>Build a SALVI pipeline</h1>
              <p>Add compatible components stage by stage, then configure the active pipeline.</p>
            </div>
            <div className="header-actions">
              <input
                ref={yamlUpload}
                hidden
                type="file"
                accept=".yaml,.yml"
                onChange={async (event) => {
                  const file = event.target.files?.[0];
                  if (!file) return;
                  try {
                    const validated = await api.validatePipeline(await file.text());
                    setConfiguration(validated.configuration);
                    setYaml(validated.yaml);
                    setNotice({ kind: "success", text: "Pipeline imported and validated." });
                  } catch (cause) {
                    setNotice({
                      kind: "error",
                      text: cause instanceof Error ? cause.message : String(cause)
                    });
                  }
                }}
              />
              <button className="button secondary" onClick={() => yamlUpload.current?.click()}>
                <Upload size={17} /> Import YAML
              </button>
              <button
                className="button secondary"
                onClick={() =>
                  ensureYaml().then((content) => downloadText("salvi-pipeline.yaml", content))
                }
              >
                <Download size={17} /> Export YAML
              </button>
              <button
                className="icon-button"
                title="Inspect YAML"
                onClick={() => ensureYaml().then(() => setYamlOpen(true))}
              >
                <Braces size={19} />
              </button>
            </div>
          </header>

          <section className="pattern-strip">
            <span>Allowed patterns</span>
            {catalog.patterns.map((pattern) => {
              const name = String(pattern.kind);
              const patterns = configuration.patterns as JsonObject;
              const allowed = (patterns.allowed as string[]) ?? [];
              const active = allowed.includes(name);
              return (
                <button
                  key={name}
                  className={active ? "active" : ""}
                  onClick={() => {
                    const next = active
                      ? allowed.filter((item) => item !== name)
                      : [...allowed, name];
                    if (next.length > 0) {
                      setConfiguration(
                        setAtPath(configuration, ["patterns", "allowed"], next)
                      );
                    }
                  }}
                >
                  {name}
                </button>
              );
            })}
            <span className={`composition-state ${resolution?.complete ? "valid" : "invalid"}`}>
              {resolution?.complete ? <CheckCircle2 size={16} /> : <FlaskConical size={16} />}
              {resolution?.complete ? "Runnable composition" : "Composition needs attention"}
            </span>
          </section>

          {resolution && !resolution.complete ? (
            <section className="composition-summary" aria-live="polite">
              <span>
                <strong>
                  {
                    resolution.roles.filter((item) => item.state === "REQUIRED").length
                  }
                </strong>{" "}
                required roles
              </span>
              <span>
                <strong>
                  {
                    resolution.roles.filter((item) => item.state === "INVALID").length
                  }
                </strong>{" "}
                invalid roles
              </span>
              {resolution.errors[0] ? <p>{resolution.errors[0]}</p> : null}
            </section>
          ) : null}

          <PipelineWorkflow
            resolution={resolution}
            stages={catalog.workflow_stages}
            selectedRole={selectedRole}
            selectedStage={selectedStage}
            datasetLabel={dataset}
            analyses={configuredAnalyses}
            availableAnalysisCount={
              catalog.analyses.filter(
                (analysis) =>
                  !selectedAnalyses.includes(analysis.name) &&
                  (!analysis.requires_ground_truth || selectedDataset?.ground_truth_attached)
              ).length
            }
            onSelectRole={(kind) => {
              setSelectedRole(kind);
              setSelectedStage("");
              setSelectedInstance("");
            }}
            onSelectStage={(nextStage) => {
              setSelectedStage(nextStage);
              setSelectedRole("");
              setSelectedInstance("");
            }}
          />

          <section className="launch-bar">
            <label>
              <span>Dataset</span>
              <select value={dataset} onChange={(event) => setDataset(event.target.value)}>
                <option value="">Choose a DatasetBundle</option>
                {datasets.map((item) => (
                  <option key={item.identifier} value={item.identifier}>
                    {item.identifier}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Run identifier</span>
              <input value={runIdentifier} onChange={(event) => setRunIdentifier(event.target.value)} />
            </label>
            <label className="seed-field">
              <span>Seed</span>
              <input
                type="number"
                min={0}
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value))}
              />
            </label>
            <button
              className="button primary run-button"
              disabled={busy || !dataset || !resolution?.complete}
              onClick={startRun}
            >
              <Play size={18} fill="currentColor" /> {busy ? "Starting..." : "Run pipeline"}
            </button>
          </section>
        </main>
      ) : (
        <Suspense fallback={<div className="view-loading">Loading view...</div>}>
          {view === "monitor" ? (
            <Monitor
              catalog={catalog}
              runs={runs}
              selectedRun={selectedRun}
              onSelectedRun={setSelectedRun}
              onRunsChanged={refreshRuns}
            />
          ) : (
            <Results
              runs={runs}
              datasets={datasets}
              analyses={catalog.analyses}
              patterns={catalog.patterns.map((pattern) => String(pattern.kind))}
              selectedRun={selectedRun}
              onSelectedRun={setSelectedRun}
            />
          )}
        </Suspense>
      )}

      {selectedRole || selectedStage ? (
        <aside className="drawer">
          <button
            className="drawer-close icon-button"
            title="Close"
            onClick={() => {
              setSelectedRole("");
              setSelectedStage("");
            }}
          >
            <X size={20} />
          </button>
          {selectedRole === "__input__" || selectedStage === "INPUT" ? (
            <DatasetImport
              adapters={catalog.input_adapters}
              datasets={datasets}
              selected={dataset}
              onSelected={setDataset}
              onRefresh={refreshDatasets}
            />
          ) : selectedRole === "__analysis__" || selectedStage === "ANALYSIS" ? (
            <div className="drawer-content">
              <div className="drawer-heading">
                <span className="eyebrow">Outside the search</span>
                <h2>Post-run analysis</h2>
                <p>These analyses consume canonical results and never influence optimization.</p>
              </div>
              {catalog.analyses.length ? (
                catalog.analyses.map((analysis) => {
                  const active = selectedAnalyses.includes(analysis.name);
                  const available =
                    !analysis.requires_ground_truth || Boolean(selectedDataset?.ground_truth_attached);
                  return (
                    <article
                      className={`instance-card ${active ? "active" : ""} ${
                        available ? "available" : "blocked"
                      }`}
                      key={analysis.name}
                    >
                      <header>
                        <span className="instance-title">
                          {active ? <CircleCheck size={17} /> : null}
                          <strong>{analysis.title}</strong>
                        </span>
                      </header>
                      <p>{analysis.description}</p>
                      {analysis.requires_ground_truth ? (
                        <small>
                          {available
                            ? `Ground truth attached to ${selectedDataset?.identifier}.`
                            : "Select a dataset with attached ground truth."}
                        </small>
                      ) : null}
                      <div className="button-row">
                        {active ? (
                          <button
                            className="button secondary compact"
                            onClick={() =>
                              setSelectedAnalyses((current) =>
                                current.filter((name) => name !== analysis.name)
                              )
                            }
                          >
                            <Trash2 size={15} /> Remove
                          </button>
                        ) : (
                          <button
                            className="button secondary compact"
                            disabled={!available}
                            onClick={() =>
                              setSelectedAnalyses((current) => [...current, analysis.name])
                            }
                          >
                            <Plus size={15} /> Use analysis
                          </button>
                        )}
                      </div>
                    </article>
                  );
                })
              ) : (
                <p className="empty-note">
                  Install salvi-experiments to enable scientific result analyses.
                </p>
              )}
            </div>
          ) : stage ? (
            <StageCatalog
              stage={stage}
              roles={stageRoles}
              onConfigure={(targetRole, component) => {
                configureInstance(targetRole, component);
                setSelectedStage("");
                setSelectedRole(targetRole.role.kind);
              }}
              onEdit={(targetRole, component) => {
                setSelectedStage("");
                setSelectedRole(targetRole.role.kind);
                setSelectedInstance(component.name);
              }}
              onRemove={(targetRole, component) =>
                removeConfigured(targetRole, component.name)
              }
            />
          ) : role ? (
            <div className="drawer-content">
              <div className="drawer-heading">
                <span className={`state-label ${role.state.toLowerCase()}`}>{role.state}</span>
                <h2>{role.role.title}</h2>
                <p>{role.role.description}</p>
                <div className="role-cardinality">
                  <span>
                    <strong>{role.configured.length}</strong> configured
                  </span>
                  <span>
                    {role.maximum === null
                      ? `At least ${role.minimum} required`
                      : role.minimum === role.maximum
                        ? `${role.minimum} required`
                        : `${role.minimum}-${role.maximum} allowed`}
                  </span>
                </div>
                {role.reasons.map((reason) => (
                  <div className="inline-warning" key={reason}>
                    {reason}
                  </div>
                ))}
              </div>
              <section className="drawer-section">
                <h3>Instances</h3>
                <div className="instance-list">
                  {role.instances.map((instance) => {
                    const active = role.configured.includes(instance.component.name);
                    const editing = active && configuredComponent?.name === instance.component.name;
                    return (
                      <article
                        className={`instance-card ${active ? "active" : ""} ${
                          editing ? "editing" : ""
                        } ${
                          instance.available ? "available" : "blocked"
                        }`}
                        key={instance.component.name}
                      >
                        <header>
                          <span className="instance-title">
                            {active ? <CircleCheck size={17} /> : null}
                            <strong>{instance.component.title}</strong>
                          </span>
                          <span className={`maturity ${instance.component.maturity.toLowerCase()}`}>
                            {instance.component.maturity}
                          </span>
                        </header>
                        <p>{instance.component.description}</p>
                        <div className="instance-tags">
                          {instance.component.supported_patterns.map((pattern) => (
                            <span key={pattern}>{pattern}</span>
                          ))}
                          {instance.component.requires.slice(0, 2).map((requirement) => (
                            <span className="capability" key={requirement}>
                              needs {requirement}
                            </span>
                          ))}
                        </div>
                        {instance.reasons.map((reason) => (
                          <small className="blocked-reason" key={reason}>
                            {reason}
                          </small>
                        ))}
                        <div className="button-row">
                          {active ? (
                            <>
                              <button
                                className={`button ${editing ? "primary" : "secondary"} compact`}
                                onClick={() => setSelectedInstance(instance.component.name)}
                              >
                                <Settings2 size={15} />
                                {editing ? "Editing" : "Edit parameters"}
                              </button>
                              <button
                                className="icon-button danger"
                                title={`Remove ${instance.component.title}`}
                                onClick={() => removeConfigured(role, instance.component.name)}
                              >
                                <Trash2 size={16} />
                              </button>
                            </>
                          ) : (
                            <button
                              className="button secondary compact"
                              disabled={!instance.available}
                              onClick={() => configureInstance(role, instance.component)}
                            >
                              <Plus size={15} /> Use instance
                            </button>
                          )}
                        </div>
                      </article>
                    );
                  })}
                </div>
              </section>
              {configuredComponent ? (
                <section className="drawer-section">
                  <h3>{configuredComponent.title} parameters</h3>
                  <ComponentEditorBoundary
                    key={`${configuredComponent.kind}:${configuredComponent.name}`}
                  >
                    <Suspense fallback={<p className="empty-note">Loading parameter editor...</p>}>
                      <ParameterEditor
                        component={configuredComponent}
                        parameters={configuredParameters(role)}
                        onChange={(parameters) => updateParameters(role, parameters)}
                      />
                    </Suspense>
                  </ComponentEditorBoundary>
                </section>
              ) : null}
            </div>
          ) : null}
        </aside>
      ) : null}

      {yamlOpen ? (
        <div className="modal-backdrop" onMouseDown={() => setYamlOpen(false)}>
          <section className="yaml-modal" onMouseDown={(event) => event.stopPropagation()}>
            <header>
              <div>
                <span className="eyebrow">Reusable algorithm definition</span>
                <h2>Pipeline YAML</h2>
              </div>
              <button className="icon-button" title="Close" onClick={() => setYamlOpen(false)}>
                <X size={20} />
              </button>
            </header>
            <textarea value={yaml} readOnly spellCheck={false} />
          </section>
        </div>
      ) : null}

      {notice ? (
        <button className={`toast ${notice.kind}`} onClick={() => setNotice(null)}>
          {notice.text}
        </button>
      ) : null}
    </div>
  );
}

function NavButton({
  active,
  onClick,
  icon,
  children
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <button className={active ? "active" : ""} onClick={onClick}>
      {icon}
      <span>{children}</span>
    </button>
  );
}
