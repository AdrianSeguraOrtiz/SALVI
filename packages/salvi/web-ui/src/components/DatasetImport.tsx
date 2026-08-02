import { useMemo, useState } from "react";
import { CheckCircle2, Database, Download, FileUp, Trash2 } from "lucide-react";
import { api } from "../api";
import type { AdapterDescription, DatasetRecord, JsonObject } from "../types";

interface Props {
  adapters: AdapterDescription[];
  datasets: DatasetRecord[];
  selected: string;
  onSelected: (identifier: string) => void;
  onRefresh: () => Promise<void>;
}

export function DatasetImport({
  adapters,
  datasets,
  selected,
  onSelected,
  onRefresh
}: Props) {
  const [adapterName, setAdapterName] = useState(adapters[0]?.name ?? "");
  const [identifier, setIdentifier] = useState("dataset");
  const [files, setFiles] = useState<Record<string, File>>({});
  const [parameters, setParameters] = useState<Record<string, string | number | boolean>>({});
  const [preview, setPreview] = useState<{
    identifier: string;
    preview: JsonObject & {
      confirmation_required: boolean;
      columns: JsonObject[];
      adapter_configuration?: JsonObject;
    };
  } | null>(null);
  const [columns, setColumns] = useState<JsonObject[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const adapter = useMemo(
    () => adapters.find((item) => item.name === adapterName),
    [adapterName, adapters]
  );

  function effectiveAdapterConfiguration(): JsonObject | null {
    const source = preview?.preview.adapter_configuration;
    if (!source) return null;
    const existing = Array.isArray(source.columns) ? source.columns : [];
    const byName = new Map(
      existing
        .filter((item): item is JsonObject => typeof item === "object" && item !== null)
        .map((item) => [String(item.name), item])
    );
    return {
      ...source,
      identifier,
      columns: columns.map((column) => ({
        ...(byName.get(String(column.name)) ?? { name: String(column.name) }),
        role: column.is_row_identifier ? "IDENTIFIER" : column.role,
        search_kind: column.role === "SEARCH" ? column.selected_kind : null
      }))
    };
  }

  async function inspect() {
    if (!adapter) return;
    setBusy(true);
    setError("");
    try {
      const slots = adapter.files
        .filter((slot) => files[slot.name])
        .map((slot) => [slot.name, files[slot.name]] as [string, File]);
      const result = await api.inspectImport(adapter.name, identifier, slots, parameters);
      setPreview({ identifier: result.identifier, preview: result.preview });
      setColumns(result.preview.columns);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  async function confirm() {
    if (!preview) return;
    setBusy(true);
    setError("");
    try {
      const dataset = await api.confirmImport(
        preview.identifier,
        preview.preview.confirmation_required ? columns : null,
        effectiveAdapterConfiguration()
      );
      await onRefresh();
      onSelected(dataset.identifier);
      setPreview(null);
      setFiles({});
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="drawer-content">
      <div className="drawer-heading">
        <span className="eyebrow">Input boundary</span>
        <h2>Dataset input</h2>
        <p>Uploads are converted to a canonical DatasetBundle before SALVI can see them.</p>
      </div>

      <section className="drawer-section">
        <h3>Available datasets</h3>
        <div className="dataset-list">
          {datasets.length === 0 ? <p className="empty-note">No imported datasets yet.</p> : null}
          {datasets.map((dataset) => (
            <div
              key={dataset.identifier}
              className={`dataset-row ${selected === dataset.identifier ? "selected" : ""}`}
            >
              <button type="button" onClick={() => onSelected(dataset.identifier)}>
                <Database size={17} />
                <span>
                  <strong>{dataset.identifier}</strong>
                  <small>
                    {dataset.adapter}
                    {dataset.ground_truth_attached ? " · ground truth" : ""}
                    {dataset.clinical_annotations_attached ? " · clinical annotations" : ""}
                  </small>
                </span>
                {selected === dataset.identifier ? <CheckCircle2 size={17} /> : null}
              </button>
              <button
                className="icon-button"
                type="button"
                title={`Delete ${dataset.identifier}`}
                onClick={() => {
                  setError("");
                  api
                    .deleteDataset(dataset.identifier)
                    .then(onRefresh)
                    .catch((cause) =>
                      setError(cause instanceof Error ? cause.message : String(cause))
                    );
                }}
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="drawer-section">
        <h3>Import new dataset</h3>
        <label className="field">
          <span className="field-label">Adapter</span>
          <select
            value={adapterName}
            onChange={(event) => {
              setAdapterName(event.target.value);
              setFiles({});
              const next = adapters.find((item) => item.name === event.target.value);
              setParameters(
                Object.fromEntries(
                  (next?.parameters ?? [])
                    .filter((item) => item.default !== null)
                    .map((item) => [item.name, item.default as string | number | boolean])
                )
              );
              setPreview(null);
            }}
          >
            {adapters.map((item) => (
              <option key={item.name} value={item.name}>
                {item.title}
              </option>
            ))}
          </select>
          <small>{adapter?.description}</small>
        </label>
        <label className="field">
          <span className="field-label">Dataset identifier</span>
          <input value={identifier} onChange={(event) => setIdentifier(event.target.value)} />
        </label>
        {adapter?.parameters.map((parameter) => (
          <label className="field" key={parameter.name}>
            <span className="field-label">{parameter.title}</span>
            {parameter.kind === "BOOLEAN" ? (
              <input
                type="checkbox"
                checked={Boolean(parameters[parameter.name] ?? parameter.default ?? false)}
                onChange={(event) =>
                  setParameters((current) => ({
                    ...current,
                    [parameter.name]: event.target.checked
                  }))
                }
              />
            ) : (
              <input
                type={parameter.kind === "STRING" ? "text" : "number"}
                required={parameter.required}
                min={parameter.minimum ?? undefined}
                max={parameter.maximum ?? undefined}
                value={String(parameters[parameter.name] ?? parameter.default ?? "")}
                onChange={(event) => {
                  const value =
                    parameter.kind === "STRING"
                      ? event.target.value
                      : Number(event.target.value);
                  setParameters((current) => ({ ...current, [parameter.name]: value }));
                }}
              />
            )}
            <small>{parameter.description}</small>
          </label>
        ))}
        {adapter?.files.map((slot) => (
          <label className="upload-slot" key={slot.name}>
            <FileUp size={18} />
            <span>
              <strong>{slot.title}</strong>
              <small>{slot.description}</small>
            </span>
            <input
              type="file"
              required={slot.required}
              accept={slot.accepted_extensions.join(",")}
              onChange={(event) => {
                const file = event.target.files?.[0];
                setFiles((current) => {
                  const next = { ...current };
                  if (file) next[slot.name] = file;
                  else delete next[slot.name];
                  return next;
                });
              }}
            />
          </label>
        ))}
        <button
          type="button"
          className="button secondary"
          disabled={
            busy ||
            !adapter ||
            adapter.files.some((slot) => slot.required && !files[slot.name]) ||
            adapter.parameters.some(
              (parameter) =>
                parameter.required &&
                parameters[parameter.name] === undefined &&
                parameter.default === null
            )
          }
          onClick={inspect}
        >
          Inspect upload
        </button>
      </section>

      {preview ? (
        <section className="drawer-section preview-section">
          <h3>Confirm inferred columns</h3>
          <p>
            {String(preview.preview.row_count)} rows · {String(preview.preview.column_count)} columns
          </p>
          {preview.preview.adapter_configuration ? (
            <button
              type="button"
              className="button secondary"
              onClick={() => {
                const blob = new Blob(
                  [JSON.stringify(effectiveAdapterConfiguration(), null, 2)],
                  { type: "text/yaml" }
                );
                const anchor = document.createElement("a");
                anchor.href = URL.createObjectURL(blob);
                anchor.download = "uci-import.yaml";
                anchor.click();
                URL.revokeObjectURL(anchor.href);
              }}
            >
              <Download size={16} />
              Export import recipe
            </button>
          ) : null}
          <div className="column-preview">
            {columns.map((column, index) => (
              <div className="column-map" key={String(column.source_index)}>
                <span>
                  <strong>{String(column.name)}</strong>
                  <small>
                    {Math.round(Number(column.missing_ratio) * 100)}% missing
                    {column.units ? ` · ${String(column.units)}` : ""}
                  </small>
                </span>
                {column.role ? (
                  <select
                    value={String(column.role)}
                    onChange={(event) =>
                      setColumns((current) =>
                        current.map((item, itemIndex) =>
                          itemIndex === index
                            ? {
                                ...item,
                                role: event.target.value,
                                is_row_identifier: event.target.value === "IDENTIFIER"
                              }
                            : item
                        )
                      )
                    }
                  >
                    {[
                      "IDENTIFIER",
                      "SEARCH",
                      "OUTCOME",
                      "COVARIATE",
                      "SUPPLEMENTARY",
                      "EXCLUDED"
                    ].map((role) => (
                      <option key={role} value={role}>
                        {role}
                      </option>
                    ))}
                  </select>
                ) : null}
                <select
                  value={String(column.selected_kind)}
                  disabled={
                    !preview.preview.confirmation_required ||
                    (column.role !== undefined && column.role !== "SEARCH")
                  }
                  onChange={(event) =>
                    setColumns((current) =>
                      current.map((item, itemIndex) =>
                        itemIndex === index
                          ? { ...item, selected_kind: event.target.value }
                          : item
                      )
                    )
                  }
                >
                  <option value="NUMERIC">Numeric</option>
                  <option value="BOOLEAN">Boolean</option>
                  <option value="CATEGORICAL">Categorical</option>
                </select>
              </div>
            ))}
          </div>
          <div className="button-row">
            <button type="button" className="button primary" disabled={busy} onClick={confirm}>
              Confirm DatasetBundle
            </button>
            <button
              type="button"
              className="icon-button"
              title="Discard import"
              onClick={() =>
                api.deleteImport(preview.identifier).finally(() => {
                  setPreview(null);
                  setFiles({});
                })
              }
            >
              <Trash2 size={18} />
            </button>
          </div>
        </section>
      ) : null}
      {error ? <div className="alert error">{error}</div> : null}
    </div>
  );
}
