import { Check } from "lucide-react";
import type { SearchFamily, SearchFamilyDescription } from "../types";

export function SearchFamilySelector({
  families,
  selected,
  busy,
  onSelect
}: {
  families: SearchFamilyDescription[];
  selected: SearchFamily | null;
  busy: boolean;
  onSelect: (family: SearchFamily) => void;
}) {
  return (
    <div className="search-family-control">
      <div className="control-label">
        <strong>Search architecture</strong>
        <span>The engine determines which component roles participate.</span>
      </div>
      <div className="family-options" role="radiogroup" aria-label="Search architecture">
        {families.map((family) => {
          const active = selected === family.family;
          return (
            <button
              key={family.family}
              type="button"
              role="radio"
              aria-checked={active}
              className={active ? "active" : ""}
              disabled={busy}
              title={family.description}
              onClick={() => onSelect(family.family)}
            >
              {active ? <Check size={14} strokeWidth={2.5} /> : null}
              <span>
                <strong>{family.title}</strong>
                <small>{family.description}</small>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
