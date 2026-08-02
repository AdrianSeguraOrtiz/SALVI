import { useEffect, useState } from "react";
import {
  isControl,
  isLayout,
  rankWith,
  type ControlProps,
  type Layout,
  type LayoutProps
} from "@jsonforms/core";
import {
  JsonForms,
  JsonFormsDispatch,
  withJsonFormsControlProps,
  withJsonFormsLayoutProps
} from "@jsonforms/react";
import type { ComponentDescription, JsonObject } from "../types";

function PrimitiveControl(props: ControlProps) {
  const { data, handleChange, path, label, required, schema, errors } = props;
  const [structured, setStructured] = useState(
    typeof data === "object" ? JSON.stringify(data ?? {}, null, 2) : ""
  );
  const variants = Array.isArray(schema.enum)
    ? schema.enum
    : Array.isArray(schema.anyOf)
      ? schema.anyOf
          .map((entry) =>
            typeof entry === "object" && entry !== null && "const" in entry
              ? (entry as { const: unknown }).const
              : null
          )
          .filter((value) => value !== null)
      : [];
  const schemaType = Array.isArray(schema.type)
    ? schema.type.find((item) => item !== "null")
    : schema.type;
  const isStructured =
    schemaType === "object" || schemaType === "array" || (!schemaType && variants.length === 0);

  useEffect(() => {
    if (isStructured) setStructured(JSON.stringify(data ?? {}, null, 2));
  }, [data, isStructured]);

  return (
    <label className="field">
      <span className="field-label">
        {label}
        {required ? <b aria-label="required">*</b> : null}
      </span>
      {schema.description ? <small>{String(schema.description)}</small> : null}
      {schemaType === "boolean" ? (
        <button
          type="button"
          className={`toggle ${data ? "active" : ""}`}
          role="switch"
          aria-checked={Boolean(data)}
          onClick={() => handleChange(path, !data)}
        >
          <span />
          {data ? "Enabled" : "Disabled"}
        </button>
      ) : variants.length > 0 ? (
        <select value={String(data ?? "")} onChange={(event) => handleChange(path, event.target.value)}>
          {variants.map((value) => (
            <option key={String(value)} value={String(value)}>
              {String(value)}
            </option>
          ))}
        </select>
      ) : isStructured ? (
        <textarea
          className="code-input"
          rows={6}
          value={structured}
          onChange={(event) => setStructured(event.target.value)}
          onBlur={() => {
            try {
              handleChange(path, JSON.parse(structured));
            } catch {
              // JSON Forms keeps the last valid value while the user repairs the draft.
            }
          }}
        />
      ) : (
        <input
          type={schemaType === "number" || schemaType === "integer" ? "number" : "text"}
          value={data == null ? "" : String(data)}
          min={typeof schema.minimum === "number" ? schema.minimum : undefined}
          max={typeof schema.maximum === "number" ? schema.maximum : undefined}
          step={schemaType === "integer" ? 1 : "any"}
          onChange={(event) => {
            const value =
              schemaType === "number" || schemaType === "integer"
                ? Number(event.target.value)
                : event.target.value;
            handleChange(path, value);
          }}
        />
      )}
      {errors ? <span className="field-error">{errors}</span> : null}
    </label>
  );
}

const renderer = {
  tester: rankWith(5, isControl),
  renderer: withJsonFormsControlProps(PrimitiveControl)
};

function VerticalLayout({ path, schema, uischema, visible, enabled }: LayoutProps) {
  if (!visible) return null;
  const elements = (uischema as Layout).elements;
  return (
    <div className="parameter-layout">
      {elements.map((element, index) => (
        <JsonFormsDispatch
          key={`${path}-${index}`}
          schema={schema}
          uischema={element}
          path={path}
          enabled={enabled}
        />
      ))}
    </div>
  );
}

const layoutRenderer = {
  tester: rankWith(5, isLayout),
  renderer: withJsonFormsLayoutProps(VerticalLayout)
};

interface Props {
  component: ComponentDescription;
  parameters: JsonObject;
  onChange: (parameters: JsonObject) => void;
}

export function ParameterEditor({ component, parameters, onChange }: Props) {
  const properties = Object.fromEntries(
    component.parameters.map((parameter) => [
      parameter.name,
      {
        ...parameter.value_schema,
        title: parameter.title,
        description: parameter.description
      }
    ])
  );
  const required = component.parameters.filter((item) => item.required).map((item) => item.name);
  const schema = { type: "object", properties, required };
  const uischema = {
    type: "VerticalLayout",
    elements: component.parameters.map((parameter) => ({
      type: "Control",
      scope: `#/properties/${parameter.name}`
    }))
  };
  if (component.parameters.length === 0) {
    return <p className="empty-note">This component has no configurable parameters.</p>;
  }
  return (
    <JsonForms
      schema={schema}
      uischema={uischema}
      data={parameters}
      renderers={[renderer, layoutRenderer]}
      onChange={({ data }) => {
        const next = (data ?? {}) as JsonObject;
        if (JSON.stringify(next) !== JSON.stringify(parameters)) onChange(next);
      }}
    />
  );
}
