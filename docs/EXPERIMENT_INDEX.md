# Experiment Index

Last slimmed: 2026-07-01.

The full pre-slim experiment table is preserved server-local at:

```text
docs/local_archive/20260630_pre_slim/EXPERIMENT_INDEX.md
```

This Git-tracked file is intentionally a compact index. For a specific run,
inspect `runs/<run>/RUN_STATUS.md`, the corresponding report under `reports/`,
or the local archive.

| Area | Current status | Canonical evidence location | Git policy | Notes |
|---|---|---|---|---|
| Current default model | `xverse_8k_anchor` remains default until superseded | historical project review/reports | summary only | Do not promote a new default without a strict gate. |
| Scaling/NM package | CPU/report artifacts completed on 2026-06-25 | `reports/` server-local | no large reports in Git | Supports scaling-axis audit/failure map, not deployable scaling-law claim. |
| Chemical V2 / future GPU branch | not launched by cleanup | future `runs/<run>/RUN_STATUS.md` | status docs only | Requires fresh resource audit and written stop rule. |
| Benchmark infrastructure | tracked as source where small | `scFMBench/`, `ops/`, docs | yes for code/docs | Large refs and generated outputs stay ignored. |
| Negative evidence | preserved in local reports/archives | `reports/`, local archive | summarize | Do not delete just to simplify the tree. |

## Rules For New Entries

- Add only high-signal rows here.
- Put long per-run details in `runs/<run>/RUN_STATUS.md`.
- Keep generated tables, logs, figures, and checkpoints out of Git.
- Record enough path/provenance that another agent can find the server artifact.
