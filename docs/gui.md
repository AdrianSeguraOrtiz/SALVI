# Local web application

SALVI's optional web application is a presentation layer over the same component
catalog, pipeline configuration, `RunService`, SQLite events and canonical
artifacts used by the Python API and CLI. React does not implement scientific
validation or search behavior.

Install and start it with:

```bash
python -m pip install "salvi[gui]"
salvi gui
```

The default address is `http://127.0.0.1:8765`. SALVI opens the browser when
possible. Use `--no-open` on headless machines:

```bash
salvi gui --no-open
```

With VS Code Remote SSH, open the **Ports** panel, forward remote port `8765`,
and open the generated local address. Direct non-loopback binding is rejected
because this single-user release has no authentication. The application does
not require X11, Wayland, Qt, VNC or Node.js at runtime.

Available launch settings are:

```text
--host 127.0.0.1
--port 8765
--no-open
--data-directory PATH
--max-upload-mib 2048
```

## Build

**Build** presents a fixed, zoomable workflow with Input, Preparation,
Evaluation, Search, Output and Analysis stages. The canvas represents the
effective pipeline rather than every catalog possibility: it shows only
configured roles plus one **Add component** entry point per stage. That entry
point groups the stage catalog into configured, currently required, compatible
and unavailable instances, with explicit reasons for blocked choices.

Every role, stage placement and effective connection comes from the public
component catalog and `CompositionResolutionService`, so stage backgrounds
remain aligned while panning or zooming and the browser contains no MOME- or
NSGA-II-specific rules. Unused optional roles do not appear. Configured
components that become incompatible remain visible as invalid until the user
resolves them; changing another component never silently removes YAML content.

Component parameters use catalog descriptions, units, schemas and recommended
widgets. A configured repeatable role is represented by one compact node with
its exact instance count, while the side panel exposes every instance
independently for editing or removal. Selecting a node highlights only its
effective relationships. The imported and exported pipeline is exactly the
YAML accepted by `salvi validate` and `salvi run`. Dataset choice, run
identifier, seed and output location are launch bindings and do not contaminate
the reusable pipeline.

The input boundary accepts:

- a canonical DatasetBundle ZIP;
- CSV or TSV, followed by explicit confirmation of inferred semantic types;
- G-Bic data when the optional `salvi-experiments` provider is installed;
- an official UCI dataset ID, optionally accompanied by an import recipe, when
  the same provider is installed.

G-Bic ground truth is optional. It is attached to the canonical dataset only for
post-run analysis and is never included in `RunContext`, pattern selection or
scientific search.

The UCI adapter retrieves official metadata, displays types, roles, units,
missingness and warnings, and lets the user edit or export the complete import
recipe before conversion. Only `SEARCH` columns enter SALVI. Outcomes,
covariates and supplementary annotations remain in the surrounding
`ClinicalDatasetBundle` for post-run clinical analysis.

## Monitor

**Monitor** starts SALVI in a separate spawned process and allows one active run.
It streams replayable events from `run.sqlite` using Server-Sent Events and polls
the same durable store for metrics. Reconnecting resumes from the SQLite event
sequence rather than process memory.

Only configured observers receive panels. Their titles, metric names, units and
view kinds come from catalog metadata. Panels can be minimized, maximized and
shown together; this layout is a browser preference and is not written to the
pipeline YAML. Cooperative cancellation uses the normal SALVI cancellation
contract and escalates after the configured grace period.

## Results

**Results** keeps raw search repertoire and final selection separate. It reads
canonical BiclusterSets in bounded pages and loads one bicluster and one matrix
fragment at a time. Server-side filters cover text and provenance, feasibility,
pattern and minimum row/column cardinality without limiting the search to the
visible page. The inspector shows structure, source values, missingness,
objectives, constraints, descriptors, provenance, inferred patterns, support,
parameters, diagnostics and per-column objective contributions.

When ground truth is attached and `salvi-experiments` is installed, REC, REL and
BE can be calculated independently for the raw repertoire and selected result.

## Storage and privacy

By default, private GUI data is stored in the platform application-data
directory under `salvi/web`. It contains:

```text
uploads/
datasets/
runs/
web.sqlite
```

Uploads are streamed with a 2 GiB default limit. ZIP members are checked for
path traversal, symbolic links, checksums and expanded size before import.
Temporary uploads are removed after confirmation or explicit discard.
DatasetBundles and completed runs remain available until explicitly deleted.
Runs can be downloaded as ZIP files without exposing server filesystem paths.

Use `--data-directory` to choose a different managed root. SALVI refuses to
delete paths outside that root.
