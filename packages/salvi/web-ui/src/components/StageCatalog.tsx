import {
  CircleCheck,
  CirclePlus,
  LockKeyhole,
  Settings2,
  Trash2
} from "lucide-react";
import type {
  ComponentDescription,
  RoleResolution,
  WorkflowStageDescription
} from "../types";

interface CatalogEntry {
  role: RoleResolution;
  component: ComponentDescription;
  active: boolean;
  available: boolean;
  reasons: string[];
}

interface Props {
  stage: WorkflowStageDescription;
  roles: RoleResolution[];
  onConfigure: (role: RoleResolution, component: ComponentDescription) => void;
  onEdit: (role: RoleResolution, component: ComponentDescription) => void;
  onRemove: (role: RoleResolution, component: ComponentDescription) => void;
}

function entriesFor(roles: RoleResolution[]): CatalogEntry[] {
  return roles.flatMap((role) =>
    role.instances.map((instance) => ({
      role,
      component: instance.component,
      active: role.configured.includes(instance.component.name),
      available: instance.available && role.state !== "UNAVAILABLE",
      reasons:
        role.state === "UNAVAILABLE" || role.state === "INVALID"
          ? [...role.reasons, ...instance.reasons]
          : instance.reasons
    }))
  );
}

function cardinality(role: RoleResolution): string {
  if (role.maximum === null) return `At least ${role.minimum}`;
  if (role.minimum === role.maximum) return `${role.minimum} required`;
  return `${role.minimum}-${role.maximum} allowed`;
}

function CatalogCard({
  entry,
  action
}: {
  entry: CatalogEntry;
  action: React.ReactNode;
}) {
  return (
    <article
      className={[
        "instance-card",
        entry.active ? "active" : "",
        entry.available || entry.active ? "available" : "blocked"
      ].join(" ")}
    >
      <header>
        <span className="instance-title">
          {entry.active ? <CircleCheck size={17} /> : null}
          <span>
            <small className="component-role-name">{entry.role.role.title}</small>
            <strong>{entry.component.title}</strong>
          </span>
        </span>
        <span className={`maturity ${entry.component.maturity.toLowerCase()}`}>
          {entry.component.maturity}
        </span>
      </header>
      <p>{entry.component.description}</p>
      <div className="instance-tags">
        {entry.component.supported_patterns.map((pattern) => (
          <span key={pattern}>{pattern}</span>
        ))}
        <span className="capability">{cardinality(entry.role)}</span>
      </div>
      {entry.reasons.map((reason) => (
        <small className="blocked-reason" key={reason}>
          {reason}
        </small>
      ))}
      <div className="button-row">{action}</div>
    </article>
  );
}

export function StageCatalog({
  stage,
  roles,
  onConfigure,
  onEdit,
  onRemove
}: Props) {
  const entries = entriesFor(roles);
  const configured = entries.filter((entry) => entry.active);
  const required = entries.filter(
    (entry) => !entry.active && entry.available && entry.role.state === "REQUIRED"
  );
  const requiredRoleCount = roles.filter((role) => role.state === "REQUIRED").length;
  const compatible = entries.filter(
    (entry) => !entry.active && entry.available && entry.role.state !== "REQUIRED"
  );
  const unavailable = entries.filter((entry) => !entry.active && !entry.available);

  return (
    <div className="drawer-content stage-catalog">
      <div className="drawer-heading stage-catalog-heading">
        <span className="eyebrow">Workflow stage</span>
        <h2>{stage.title}</h2>
        <p>{stage.description}</p>
        <div className="stage-catalog-stats">
          <span>
            <strong>{configured.length}</strong> configured
          </span>
          <span className={requiredRoleCount ? "needs-attention" : ""}>
            <strong>{requiredRoleCount}</strong> required
          </span>
          <span>
            <strong>{compatible.length}</strong> compatible
          </span>
        </div>
      </div>

      {configured.length > 0 ? (
        <section className="drawer-section">
          <h3>Configured</h3>
          <div className="instance-list">
            {configured.map((entry) => (
              <CatalogCard
                key={`${entry.role.role.kind}:${entry.component.name}`}
                entry={entry}
                action={
                  <>
                    <button
                      className="button secondary compact"
                      onClick={() => onEdit(entry.role, entry.component)}
                    >
                      <Settings2 size={15} /> Edit parameters
                    </button>
                    <button
                      className="icon-button danger"
                      title={`Remove ${entry.component.title}`}
                      onClick={() => onRemove(entry.role, entry.component)}
                    >
                      <Trash2 size={16} />
                    </button>
                  </>
                }
              />
            ))}
          </div>
        </section>
      ) : null}

      {required.length > 0 ? (
        <section className="drawer-section required-catalog-group">
          <h3>Required now</h3>
          <p className="section-intro">
            Add one of these compatible instances to complete the current composition.
          </p>
          <div className="instance-list">
            {required.map((entry) => (
              <CatalogCard
                key={`${entry.role.role.kind}:${entry.component.name}`}
                entry={entry}
                action={
                  <button
                    className="button primary compact"
                    onClick={() => onConfigure(entry.role, entry.component)}
                  >
                    <CirclePlus size={15} /> Use instance
                  </button>
                }
              />
            ))}
          </div>
        </section>
      ) : null}

      {compatible.length > 0 ? (
        <section className="drawer-section">
          <h3>Compatible components</h3>
          <div className="instance-list">
            {compatible.map((entry) => (
              <CatalogCard
                key={`${entry.role.role.kind}:${entry.component.name}`}
                entry={entry}
                action={
                  <button
                    className="button secondary compact"
                    onClick={() => onConfigure(entry.role, entry.component)}
                  >
                    <CirclePlus size={15} />
                    {entry.role.configured.length && !entry.role.role.repeatable
                      ? "Replace current"
                      : "Use instance"}
                  </button>
                }
              />
            ))}
          </div>
        </section>
      ) : null}

      <details className="unavailable-catalog" open={entries.length > 0 && unavailable.length === entries.length}>
        <summary>
          <LockKeyhole size={15} />
          Unavailable for this composition
          <span>{unavailable.length}</span>
        </summary>
        {unavailable.length > 0 ? (
          <div className="instance-list">
            {unavailable.map((entry) => (
              <CatalogCard
                key={`${entry.role.role.kind}:${entry.component.name}`}
                entry={entry}
                action={null}
              />
            ))}
          </div>
        ) : (
          <p className="empty-note">Every catalog entry in this stage is currently compatible.</p>
        )}
      </details>
    </div>
  );
}
