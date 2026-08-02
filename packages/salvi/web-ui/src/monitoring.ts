import type { JsonObject, Metric, ObserverMetricPresentation } from "./types";

export function appendMetricHistory(current: Metric[], incoming: Metric[]): Metric[] {
  const lastSequence = current.at(-1)?.sequence ?? 0;
  const unseen = incoming.filter((metric) => metric.sequence > lastSequence);
  return unseen.length === 0 ? current : [...current, ...unseen];
}

export function eventFailureMessage(event: JsonObject | null): string | null {
  if (event?.event_type !== "run.failed") return null;
  const payload = event.payload;
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return null;
  const error = (payload as JsonObject).error;
  return typeof error === "string" && error.trim() ? error : null;
}

export function latestMetricsByName(metrics: Metric[]) {
  const values = new Map<string, Metric>();
  metrics.forEach((metric) => values.set(metric.name, metric));
  return values;
}

export function humanizeMetricName(name: string, prefix?: string): string {
  const withoutPrefix = prefix && name.startsWith(`${prefix}.`) ? name.slice(prefix.length + 1) : name;
  return withoutPrefix
    .split(".")
    .map((part) =>
      part
        .split("_")
        .filter(Boolean)
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ")
    )
    .join(" · ");
}

function metricPatternMatches(pattern: string, name: string): boolean {
  const expression = pattern
    .split("*")
    .map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join(".*");
  return new RegExp(`^${expression}$`).test(name);
}

export function metricPresentation(
  name: string,
  definitions: ObserverMetricPresentation[]
): ObserverMetricPresentation | undefined {
  return definitions
    .filter((definition) => metricPatternMatches(definition.pattern, name))
    .sort(
      (left, right) =>
        right.pattern.replaceAll("*", "").length - left.pattern.replaceAll("*", "").length
    )[0];
}

export function seriesLabel(name: string, definitions: ObserverMetricPresentation[]): string {
  const definition = metricPresentation(name, definitions);
  if (definition && !definition.pattern.includes("*")) return definition.label;
  return humanizeMetricName(name, name.split(".")[0]);
}
