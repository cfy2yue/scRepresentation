# Operations Helpers

This folder is for workspace maintenance scripts that should not live inside
either code repo.

Current policy:

- Active root scripts stay in place while background jobs reference them.
- After sync and smoke tests finish, durable helper scripts can move here.
- Cleanup scripts must be conservative by default: archive and report first,
  delete only when the target is clearly temporary or user-approved.

Helpers:

- `hourly_progress_check.sh`: record one transfer progress row at most once per
  hour in `logs/progress_history.tsv`; if called too early, it exits without
  checking transfer state.
- `summarize_sync_progress.sh`: build `reports/SYNC_PROGRESS.md` from the
  hourly progress history without touching large data directories.
- `generate_dataset_inventory.sh`: after sync completion, write final dataset
  sizes, file counts, required-file checks, training-ready package boundaries,
  source/rebuild archive boundaries, and temporary-file candidates into
  `reports/DATASET_INVENTORY.md`.
- `generate_validation_report.sh`: summarize post-sync validation status from
  the current validation log into `reports/VALIDATION_REPORT.md`.
- `validate_dataset_package.sh`: lightweight presence validation for the
  training-ready dataset package; set `RUN_FM_BUNDLE_VALIDATION=1` to also run
  LatentFM bundle validators. Use `--dataset-root PATH` to validate a restored
  cloud/Zenodo package outside `/data/cyx/1030/dataset`.
- `stage_training_dataset_package.sh`: non-destructively stage only the
  training-ready dataset package for cloud/Zenodo upload. It intentionally
  excludes source/rebuild archives and never passes `rsync --delete`.
- `generate_dataset_package_manifest.sh`: write a lightweight TSV file manifest
  for the training/rebuild/legacy package. By default it records metadata only;
  `--with-sha256` is available for explicit full-content checksum runs.
- `compare_dataset_package_manifest.sh`: compare a restored package against a
  reference manifest by path, type, size, and symlink target while ignoring
  modification time.
- `check_latentfm_followup_posthoc_once.sh`: one-shot LatentFM follow-up status
  check that records tmux state, recent watcher log tail, posthoc JSON outputs,
  refreshes the alignment smoke report, and runs the read-only decision helper
  without continuous polling.
- `latentfm_followup_decision.py`: read-only gate helper that summarizes
  existing LatentFM follow-up JSON outputs into
  `reports/LATENTFM_FOLLOWUP_DECISION_STATUS_20260617.md` and
  `reports/latentfm_followup_decision_status_20260617.json`; it does not
  inspect tmux, tail logs, or launch jobs.
- `generate_workspace_status.py`: write a lightweight dashboard to
  `reports/WORKSPACE_STATUS.md` summarizing the active LatentFM branch,
  benchmark artifact layer, dataset package, key reports, and git status. It
  avoids tailing long-job logs and does not read large data contents.
- `validate_workspace_status.py`: pure-Python validation for key
  `generate_workspace_status.py` status-summary rules, including the case where
  four-run posthoc completion supersedes a stale launch `RUN_STATUS.md`.
- `check_relational_decision_once.sh`: one-shot status check for the active
  scFoundation relational-residual LatentFM branch. Use only after the
  scheduled 07:45/08:35/08:45/09:35/09:45/09:50 CST windows; it reads marker
  files and report JSON states only, and does not inspect tmux, GPU state, or
  training logs.
- `validate_handoff_docs.py`: read-only validation for the current handoff
  layer. It checks required docs/scripts exist, scans key Markdown files for
  token-like strings, and validates referenced `/data/cyx/1030/scLatent/...` paths
  without touching long-running jobs.
- `select_available_gpus.py`: read-only GPU availability sampler for this
  shared 8x4090 server. It samples `nvidia-smi`, maps compute PIDs to owners,
  applies the stable-light rule (`util<20%` and `memory.used<5120MiB` across
  samples), reports a lightweight CPU/RAM load snapshot, respects the
  four-physical-GPU user budget, and allows up to three LatentFM strategy jobs
  per physical GPU when safe. Example:
  `python ops/select_available_gpus.py --samples 3 --interval-seconds 10 --need 4`.
- `validate_gpu_availability_helper.py`: pure-Python validation for
  `select_available_gpus.py`; it does not call `nvidia-smi`.
- `check_latentfm_pertresid_once.sh`: one-shot status check for the active
  pert-residual target LatentFM smoke. Use only at the allowed 30-minute check
  window; it writes `reports/LATENTFM_PERTRESID_ONE_SHOT_STATUS_20260618.md`
  and refreshes `reports/WORKSPACE_STATUS.md`.
- `run_latentfm_strategy_probe_20260619.sh`: four-GPU capped LatentFM strategy
  probe after the full128 residual audit. It launches the active
  scFoundation/Stack short finetune matrix and writes
  `runs/latentfm_strategy_probe_20260619/RUN_STATUS.md`.
- `run_latentfm_strategy_probe_posthoc_20260619.sh`: detached watcher for the
  four-run strategy probe. It waits at 30-minute intervals, then runs
  split/family/full128 residual posthoc, checks each evaluation step exit code,
  and writes `reports/LATENTFM_STRATEGY_PROBE_20260619.md`.
- `summarize_latentfm_strategy_probe_20260619.py`: read-only summarizer for the
  four-run strategy probe posthoc JSON/CSV outputs.
- `run_latentfm_strategy_probe_expanded_20260619.sh`: expanded low-util
  strategy probe that stacks two extra LatentFM parameter/strategy jobs on
  each already-used physical GPU, respecting the three-jobs-per-GPU ceiling and
  writing `runs/latentfm_strategy_probe_expanded_20260619/RUN_STATUS.md`.
  Its posthoc phase checks each split/family/residual evaluation step exit code
  before writing per-label success markers.
- `summarize_latentfm_strategy_probe_expanded_20260619.py`: read-only summary
  builder for the expanded probe; expected outputs are
  `reports/LATENTFM_STRATEGY_PROBE_EXPANDED_20260619.md`,
  `reports/latentfm_strategy_probe_expanded_20260619.csv`, and
  `reports/latentfm_strategy_probe_expanded_20260619.json`.
- `summarize_latentfm_strategy_all_20260619.py`: read-only decision helper
  that merges the four-run and expanded strategy CSVs once they exist. It
  writes `reports/LATENTFM_STRATEGY_ALL_DECISION_20260619.md` plus CSV/JSON
  companions and remains `pending` while upstream posthoc reports are missing.
- `validate_latentfm_strategy_all_summary.py`: pure-Python validation for the
  combined strategy decision helper. It uses temporary CSVs to test no-input
  pending, empty-CSV-as-missing, partial-input pending, strict
  repeat-candidate, diagnostic-candidate, and reject branches without touching
  real reports, training logs, tmux, or GPUs.
- `run_latentfm_strategy_all_decision_watcher_20260619.sh`: detached watcher
  that checks only for the two strategy CSV files and upstream `RUN_STATUS.md`
  terminal states every 30 minutes. Once both CSVs exist, or both upstream jobs
  have finished, it runs `summarize_latentfm_strategy_all_20260619.py`; it does
  not tail training logs or inspect GPU state.
- `plot_latentfm_strategy_all_decision_20260619.py`: read-only plotter for the
  combined strategy-decision CSV. It writes
  `reports/latentfm_strategy_all_decision_20260619.{pdf,png,svg}` once complete
  strategy rows exist; before that it writes a small placeholder text file.
- `validate_latentfm_strategy_all_plotter.py`: pure-Python validation for the
  combined strategy decision plotter. It uses temporary CSVs to verify both
  PDF/PNG/SVG generation and the no-complete-row placeholder path.
- `summarize_latentfm_condition_residual_audit_20260619.py`: read-only
  summarizer for existing per-condition residual CSVs. It aggregates
  condition-level cosine/pearson/retrieval diagnostics across the strategy
  probes without rerunning inference or touching GPUs.
- `diagnose_latentfm_knn_additive_residual_20260619.py`: CPU-only diagnostic
  for zero-shot multi-perturbation splits. It tests whether scGPT gene-neighbor
  residuals plus train-single additive deltas can reconstruct held-out multi
  deltas, using capped HDF5 condition reads and no model training.
- `evaluate_latentfm_prior_correction_20260619.py`: capped evaluator for
  existing LatentFM checkpoints. It runs ODE inference, builds a train-single
  KNN/additive residual prior from the scGPT gene cache, interpolates checkpoint
  condition means with that prior, and writes direct/pc/pp reports without
  training or reading large raw data.
- `final_rsync_audit.sh`: after sync completion, run low-priority rsync
  dry-run comparisons and write `reports/FINAL_RSYNC_AUDIT.md`.
- `post_sync_cleanup.sh`: archive old logs and list temporary-file candidates
  after sync and validation are complete.

Dataset retention and cloud backup entry points:

- `reports/DATASET_INVENTORY.md`: current local data tree and required-file
  checks.
- `reports/DATASET_RETENTION_PLAN_20260617.md`: explains why `dataset/` is
  larger than the minimal training package and which directories are optional
  rebuild archives.
- `reports/DATASET_CLOUD_BACKUP_MANIFEST_20260617.md`: staging commands and
  validation checklist for cloud/Zenodo backup.
- `dataset/README.md`: portable README to include with the training-ready data
  package.
