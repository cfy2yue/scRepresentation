# Prompt: Organize The CellClip Project

You are working in `/data/cyx/1030/CellClip`, an independent project. Do not
modify `/data/cyx/1030/scLatent` or `/data/cyx/1030/stock` unless explicitly
asked.

Goal: organize CellClip so its project story, data dependencies, and current
state are clear. This is a documentation and cleanup task first. Do not launch
training or GPU jobs unless I explicitly ask.

Read first:

```text
goal.md
docs/PROJECT_OVERVIEW.md
docs/PROJECT_REVIEW.md
docs/EXPERIMENT_INDEX.md
docs/RESULTS_SUMMARY.md
docs/DECISIONS.md
docs/BUGS_AND_FIXES.md
docs/Tahoe100M_USAGE.md
```

Then produce/update:

1. A top-level `README.md` or `docs/START_HERE.md` that explains:
   - project goal;
   - current best stage/result;
   - core model/data flow;
   - what depends on scLatent stack encodings or datasets;
   - what can be run independently.
2. A `docs/DATA_DEPENDENCIES.md` or equivalent mirror manifest:
   - source data paths;
   - stack/embedding dependencies from scLatent;
   - generated caches;
   - what should be copied/mirrored if CellClip becomes fully standalone.
3. A cleanup inventory:
   - keep;
   - archive;
   - safe delete;
   - unknown/needs user decision.
4. Remove only clearly reproducible noise such as pytest caches or empty temp
   files. Do not delete runs, reports, logs, data caches, manifests, or failed
   outputs unless the cleanup inventory explicitly proves they are redundant.
5. Update `docs/PROJECT_REVIEW.md` with a dated organization checkpoint.

Important coordination rule: if you want to edit shared scLatent files or code,
stop and ask. CellClip can consume scLatent artifacts, but it should not silently
rewrite them.

Final response should list changed files, deleted cache/noise files, unresolved
cleanup decisions, and the next concrete CellClip action.
