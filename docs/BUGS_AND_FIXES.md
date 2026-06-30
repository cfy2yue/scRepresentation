# Bugs And Fixes

## Track C v2 pass-only scripts read wrong smoke decision status field

### Date

2026-06-23

### Symptom

The Track C routed-distill smoke summarizer writes the machine decision under
`decision.status` in its JSON payload.  Several newly prepared v2
support-context pass-only scripts initially read only a top-level `status`.
Once capped posthoc reports were produced, a true pass would have been misread
as `missing_status` or not passed.

### Fix / status

Patched the v2 original checker, v2 parallel checker, original v2 uncapped
launcher, and parameterized v2 uncapped launcher to read either top-level
`status` or nested `decision.status`.

Validation:

* `bash -n` passes for all four scripts;
* the parallel checker still refuses before its guard window with `RC=3`;
* the parameterized uncapped launcher still fails closed with `RC=2` when a
  capped decision JSON is absent.

## Track C v2 canonical posthoc needed forced support-absent mode

### Date

2026-06-23

### Symptom

The support-context v2 protocol requires canonical Track A evaluation to be
support-context absent.  The generic Track C routed-distill posthoc path loaded
the candidate checkpoint config and built a routed support-context bank whenever
the model used support residual/FiLM adapters.  For canonical family rows that
exactly collide with support-val conditions, that would make canonical posthoc
support-present and invalidate the v2 exact no-op gate.

### Fix / status

Added `--force-support-context-absent` to
`/data/cyx/1030/CoupledFM/model/latent/eval_split_groups.py` and
`/data/cyx/1030/CoupledFM/model/latent/eval_condition_families.py`.  The eval
path in `/data/cyx/1030/CoupledFM/model/latent/train.py` now builds the routed
support-context bank only when `trackc_support_context_source` is active; with
source `off`, support-context models receive zero context and therefore take
the exact support-absent path.  The generic Track C launcher now passes
`--force-support-context-absent` for canonical posthoc commands.

Validation:

* `py_compile` passed for `train.py`, both eval CLIs, and the v2 launcher gate;
* launcher `bash -n` checks passed;
* targeted support-context tests: `15 passed, 23 deselected`;
* both eval CLI help outputs expose `--force-support-context-absent`;
* launcher/provenance gate passed and recorded failed checks none.

## Track C support-context zero context was not guaranteed exact no-op

### Date

2026-06-23

### Symptom

The support-context v2 protocol requires canonical Track A evaluation to be
support-context absent and exact no-op.  Existing support residual/FiLM adapters
accepted zero support context, but `support_context_to_v` and
`support_context_to_v_scale` were trainable linear layers with bias.  After
training, zero context could still emit nonzero residual or FiLM output through
the bias term.  Thus zero context was not a valid exact no-op guarantee.

### Fix / status

`/data/cyx/1030/CoupledFM/model/latent/models/mlp.py` now uses biasless
support-context/residual/FiLM projections and accepts optional
`support_context_present`.  All support adapter outputs are multiplied by the
support-present mask; when the mask is absent, all-zero support context is
treated as support-absent.

Validation:

* targeted support-context tests: `16 passed, 22 deselected`;
* plan guards: `32 passed`;
* static code-boundary audit:
  `trackc_support_context_v2_code_boundary_pass_cpu_gate_next`.

## Anchor-gated support-teacher summarizer treated zero as missing

### Date

2026-06-23

### Symptom

After the condition-means family repair finished, the Track C anchor-gated
support-teacher CPU gate report showed canonical no-op deltas exactly zero:

```text
test_single max_abs_delta +0.000000
family_gene max_abs_delta +0.000000
```

but the decision still reported:

```text
test_single_not_exact_noop
family_gene_not_exact_noop
```

The cause was Python truthiness in the summarizer:

```python
float(row.get("max_abs_delta_pp") or 999.0)
```

which treats a valid `0.0` as missing. The same pattern also affected support
delta checks such as a valid Norman delta of `0.0`.

### Fix / status

Updated
`/data/cyx/1030/ops/summarize_latentfm_trackc_anchor_gated_support_teacher_cpu_gate_20260623.py`
to distinguish `None` from numeric zero. Added a regression test:

```text
/data/cyx/1030/ops/test_latentfm_trackc_anchor_gated_support_teacher_cpu_gate_20260623.py
```

Validation:

```text
PYTHONPATH=/data/cyx/1030/CoupledFM /data/cyx/software/miniconda3/envs/scdfm/bin/python -m pytest /data/cyx/1030/ops/test_latentfm_trackc_anchor_gated_support_teacher_cpu_gate_20260623.py -q
4 passed in 0.09s
```

The real summarizer was rerun and now reports:

```text
trackc_anchor_gated_support_teacher_cpu_gate_pass_code_gate_next
```

## Track C condition-means family eval rejected pert-means override

### Date

2026-06-23

### Symptom

The Track C condition-means artifact job successfully generated support split
and canonical `test_single` condition-mean artifacts, then failed at the
`eval_condition_families` family_gene stage:

```text
eval_condition_families.py: error: unrecognized arguments: --pert-means-file /data/cyx/1030/dataset/latentfm_full/xverse/pert_means.npz
```

The launcher also wrote the literal string `$?` to `EXIT_CODE` because the
tmux command over-escaped the shell status variable.

### Fix / status

Added `--pert-means-file` support to
`/data/cyx/1030/CoupledFM/model/latent/eval_condition_families.py`, using the
same resolver as `eval_split_groups`, and fixed the launcher tmux command to
write a numeric `rc`.

Validation:

```text
PYTHONPATH=/data/cyx/1030/CoupledFM /data/cyx/software/miniconda3/envs/scdfm/bin/python -m py_compile CoupledFM/model/latent/eval_condition_families.py CoupledFM/model/latent/eval_split_groups.py CoupledFM/model/latent/train.py CoupledFM/model/latent/config.py
bash -n ops/launch_latentfm_trackc_condition_means_artifacts_20260623.sh
python -m model.latent.eval_condition_families --help shows --pert-means-file
```

Recovery:

The four completed split/support artifacts were preserved. A targeted repair
job was launched to generate only the missing canonical `family_gene`
condition-mean artifacts and then run the fail-closed CPU gate summarizer:

```text
/data/cyx/1030/runs/latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/xverse_support_film_retry1_condition_means_family_repair/RUN_STATUS.md
```

## Endpoint 17:40 checker lacked Python fallback

### Date

2026-06-22

### Symptom

The 17:40 endpoint-routed checker successfully reported training
`EXIT_CODE=0`, posthoc `POSTHOC_EXIT_CODE=0`, and the decision artifact paths,
but then returned `127` before printing `decision_status`/`decision_action`
because it used `/data/cyx/1030/software/miniconda3/envs/scdfm/bin/python`
without the project-standard fallback to `/data/cyx/software/miniconda3/envs/scdfm/bin/python`.

### Fix / status

Updated `/data/cyx/1030/ops/check_latentfm_trackc_endpoint_routed_after_1740_20260622.sh`
to use the same Python fallback pattern as the other guards. Re-running the
checker after the artifact already existed printed:

```text
decision_status=trackc_smoke_fail_canonical_harm_close_branch
decision_action=close_branch_or_redesign_noharm_adapter
```

## Track C endpoint decision checklist status drift

### Date

2026-06-22

### Symptom

The endpoint-routed decision checklist listed support failure as
`trackc_smoke_fail_support_close_branch`, but the shared smoke summarizer uses
`trackc_smoke_fail_support_gate_close_branch`. The checklist also predeclared a
missing-metrics close status, while the summarizer previously folded missing
required metrics into the generic support-fail path.

This could confuse downstream branch closure bookkeeping after the posthoc
decision appears, although the pass gate remained exact and the uncapped/query
guards were already fail-closed unless the smoke status is precisely
`trackc_smoke_support_pass_needs_uncapped_noharm_before_query`.

### Fix / status

Updated `/data/cyx/1030/ops/summarize_latentfm_trackc_routed_distill_smoke_20260622.py`
so missing required metrics return
`trackc_smoke_missing_required_metrics_close_branch`, and updated the endpoint
decision checklist JSON/MD to use the actual support-fail status
`trackc_smoke_fail_support_gate_close_branch`. The checklist polling boundary
was also advanced from the completed 17:10 check to the next allowed 17:40
check.

Validation:

```text
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m py_compile ops/summarize_latentfm_trackc_routed_distill_smoke_20260622.py
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m json.tool reports/latentfm_trackc_endpoint_routed_decision_checklist_20260622.json
evaluate_gate({}) -> trackc_smoke_missing_required_metrics_close_branch
/data/cyx/software/miniconda3/envs/scdfm/bin/python ops/validate_latentfm_trackc_endpoint_decision_metadata_20260622.py
trailing whitespace scan clean
```

## LatentFM anchor-preserving finetunes could mix raw checkpoint base with EMA baselines

### Date

2026-06-22

### Symptom

The Track C routed-distill smoke used an xverse anchor selected and evaluated
with active EMA weights, but the candidate finetune warm-start path loaded only
the raw `ckpt["model"]` state. Because `finetune_trainable_scope` froze the
base model and trained only the condition-prior adapter, the candidate base
could remain raw-anchor while posthoc anchor baselines used EMA-anchor weights.

This can confound canonical no-harm interpretation, especially small MMD deltas
on frozen-base strata. The routed-distill branch still failed its support-val
material-gain gate (`test_multi` pp delta `+0.006434 < +0.02`), so the branch
closure stands, but future no-harm adapter smokes must make anchor weight
provenance explicit.

### Fix / status

Added default-off EMA loading controls in CoupledFM:

```text
Config.init_checkpoint_use_ema
Config.anchor_replay_checkpoint_use_ema
```

`load_model_weights_only(..., prefer_ema=True)` now loads active EMA
`shadow.*` tensors when explicitly requested, while default raw-model behavior
is unchanged. The full-stack launcher exposes the corresponding environment
flags:

```text
INIT_CHECKPOINT_USE_EMA=1
ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1
```

Audit report:

```text
/data/cyx/1030/reports/LATENTFM_EMA_ANCHOR_PROVENANCE_AUDIT_20260622.md
```

The audited Track C smoke candidate matched the raw anchor base exactly
(`max_abs=0`) and differed from the anchor EMA base (`max_abs=0.000983000`);
candidate EMA shadows covered only 8 trainable tensors and did not cover the
frozen base.

Validation:

```text
python -m py_compile model/latent/train.py model/latent/config.py model/tests/test_latent_eval_selection.py
pytest -q model/tests/test_latent_eval_selection.py model/tests/test_latent_condition_embedding_sources.py
bash -n /data/cyx/1030/CoupledFM/model/latent/scripts/run_full_stack_latentfm.sh
git diff --check
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m py_compile ops/audit_latentfm_ema_anchor_provenance_20260622.py
```

## GitHub HTTPS push required explicit token header

### Date

2026-06-16

### Symptom

`git push` failed with:

```text
fatal: could not read Username for 'https://github.com': No such device or address
```

### Fix

Use the sourced `GHTOKEN_1030` only as a temporary `http.extraheader` during push. Do not write the token to git remotes or logs.

## Rsync monitor log path bug

### Date

2026-06-16

### Symptom

The top3 data sync command used `tee -a ""`, so the intended log file was not written even though rsync continued.

### Fix

Documented the running session in `/data/cyx/1030/runs/sync_top3_training_data/RUN_STATUS.md`. Future rsync launches should use an explicit log path and status file before launch.

## Ambiguous NicheFormer resource validation

### Date

2026-06-16

### Symptom

`validate_resources.py --models nicheformer transcriptformer` reported that either the NicheFormer checkpoint or model mean was missing, even when the model mean file existed.

### Fix

Updated `/data/cyx/1030/scFMBench/fm/tools/model_registry.py` so NicheFormer validation reports the exact missing component. This was later superseded by the Hugging Face safetensor route in scFMBench commit `ae8bc45`; current NicheFormer resource validation passes, and remaining work is completing MCF7/xCellLine embeddings plus metrics.

## CoupledFM smoke tests depend on local raw metainfo fixtures

### Date

2026-06-16

### Symptom

`pytest -q model/tests/test_smoke_mains.py` currently fails in:

```text
test_smoke_metainfo_fallback
test_smoke_biflow_state_genepert
```

because `/data/cyx/1030/CoupledFM/data/raw/genepert_DE5000/metainfo.json` is missing or empty.

### Fix / status

This is a local fixture/data availability issue, not a scGPT default-cache regression. For config-only changes, use `pytest -q model/tests/test_plan_guards.py` plus explicit default-cache assertions until the raw metainfo fixture is restored.

## LatentFM alignment smoke launcher missed project environment

### Date

2026-06-17

### Symptom

`latentfm_alignment_smoke_20260617` launched tmux successfully, but the job exited immediately with:

```text
ModuleNotFoundError: No module named 'numpy'
```

The failed tag was:

```text
20260617_scfoundation_resid002_ctr0005_comp006_sum_pool_3k_smoke
```

### Cause

The launcher process was started from an activated environment, but the detached tmux command did not source `/data/cyx/1030/init-scdfm.sh`. The shell inside tmux therefore used the wrong Python environment.

### Fix / status

First attempted fix: update `/data/cyx/1030/runs/latentfm_alignment_smoke_20260617/launch_scfoundation_residual_sum_smoke.py` so the tmux command begins with:

```bash
source /data/cyx/1030/init-scdfm.sh >/dev/null
```

This was not sufficient for the retry1 launch; the detached run still resolved the wrong Python.

Final fix: update `/data/cyx/1030/CoupledFM/model/latent/scripts/run_full_stack_latentfm.sh` to use an explicit `PYTHON_BIN`, defaulting to:

```bash
/data/cyx/software/miniconda3/envs/scdfm/bin/python
```

`bash -n` passes. The fresh retry2 tag `20260617_scfoundation_resid002_ctr0005_comp006_sum_pool_3k_smoke_retry2` entered training successfully at 2026-06-17 17:57 CST and printed the expected scFoundation config; do not overwrite failed logs from the original or retry1 attempts.

## LatentFM scGPT condition source was implicit in configs

### Date

2026-06-17

### Symptom

Full-data LatentFM configs correctly pointed `pert_gene_emb_cache_dir` at:

```text
/data/cyx/1030/pretrainckpt/genepert_cache/scgpt_embed_gene
```

but `pert_condition_embedding_source` was empty in saved config JSONs, making it easy to misread the run as having ambiguous genetic perturbation condition semantics.

### Fix / status

This was a provenance issue, not a runtime conditioning bug: `pert_gene_emb_cache_dir` controls the loaded `GeneEmbeddingCache`.

CoupledFM commit `d724a87 fix: record latent perturbation embedding source` now exports `PERT_EMBED_SOURCE=scgpt_embed_gene` in the full/top3 LatentFM launchers and prints it in launch logs. Commit `c9b9395 fix: infer latent condition source from cache` also makes direct CLI training infer the source from `pert_gene_emb_cache_dir` when the field is omitted.

Lightweight validation:

```text
Config().pert_gene_emb_cache_dir -> /data/cyx/1030/pretrainckpt/genepert_cache/scgpt_embed_gene
fill_condition_embedding_source(Config(use_pert_condition=True)) -> scgpt_embed_gene
```

## LatentFM full-data runs did not checkpoint before epoch-end

### Date

2026-06-17

### Symptom

Full-data scFoundation LatentFM runs could train for many thousands of steps
without writing `latest.pt` because mid-epoch evaluation/checkpointing had been
removed. The interrupted `comp003_delta_w12` branch lost pre-epoch-end progress
and had to be treated as a fresh retry.

### Fix / status

CoupledFM commit `b31c725 fix: checkpoint long latentfm epochs` adds lightweight
periodic `latest.pt` saving every `eval_every` steps without changing training
loss, evaluation metrics, best-checkpoint selection, or final outputs.

Future long LatentFM runs should verify that `latest.pt` appears before
epoch-end. Interrupted runs that predate this fix should not be interpreted as
completed experiments.

## Warm-starting older LatentFM checkpoints failed after adding new modules

### Date

2026-06-17

### Symptom

The condition-delta head-injection smoke initially exposed a checkpoint-loading
gap: older warm-start checkpoints lacked newly introduced module weights such as
`condition_delta_to_c`, causing strict load assumptions to fail or making the
new branch brittle.

### Fix / status

CoupledFM commit `17d3106 fix: relax latentfm init checkpoint loading` allows
old checkpoint weights to load while initializing newly introduced modules
fresh. New experimental branches that change architecture should be clearly
labeled and must not claim checkpoint-equivalent comparability unless the new
modules are explicitly accounted for.

## GPU scheduling lock was held across long NicheFormer embedding work

### Date

2026-06-17

### Symptom

The NicheFormer chempert embedding watcher held the shared
`gpu_schedule.lock` while a long embedding export was running. This could block
other resource-aware launchers, including LatentFM posthoc, even when another
GPU was safely available.

### Fix / status

The NicheFormer launcher was updated to release the lock immediately after
dispatching the long embedding task. The currently documented workaround was to
manually launch a safe posthoc job after a 3-sample GPU audit. Future launchers
should hold the scheduling lock only while making the launch decision, not for
the full lifetime of the detached job.

## NicheFormer and TranscriptFormer must not duplicate log1p

### Date

2026-06-19

### Symptom

The benchmark `.X` matrices are already log1p-transformed. NicheFormer and
TranscriptFormer require raw/count-like inputs, so silently applying another
`log1p` or reconstructing pseudo-counts with `expm1(X)` would create an
incorrect benchmark.

### Fix / status

The current adapters require explicit count sources such as `raw.X` or
`layers['counts']`. The continuation check confirms both models are
resource-ready and do not duplicate `log1p` for the current chempert scope.

Evidence:

```text
/data/cyx/1030/reports/SCFMBENCH_CONTINUATION_CHECK_20260619.md
```

Keep this invariant for any future atlas/genepert expansion of these two
models.

## Synthetic condition-prior batches hard-coded nperts=2

### Date

2026-06-19

### Symptom

The new LatentFM condition-prior teacher helper built synthetic gene-combo
perturbation batches with `nperts_obs=2` unconditionally. The active dose
probes use `condition_prior_num_genes=2`, so their current behavior is not
changed, but the configuration supports larger synthetic gene counts and would
have produced inconsistent `nperts` metadata for future 3+ gene prior tests.

### Fix / status

`model.latent.train._make_gene_combo_perturbation_batch` now records
`nperts_obs=len(clean_genes)` after canonicalizing/deduplicating gene symbols.
A focused smoke test verifies both 2-gene and 3-gene synthetic batches:

```text
/data/cyx/1030/CoupledFM/model/tests/test_condition_prior_teacher.py
```

Validation commands run:

```bash
python model/tests/test_condition_prior_teacher.py
python -m py_compile model/latent/train.py model/latent/config.py model/tests/test_condition_prior_teacher.py
python -m model.tools.validate_repo
```

## Train-only internal-val eval needed explicit pert-means provenance

### Date

2026-06-22

### Symptom

`model.latent.eval_split_groups` always loaded `pert_means.npz` from the latent
bundle directory. That is appropriate for canonical posthoc, but train-only
internal-val audits need the pp reference to come from the same train-only
split used by their baseline gates. Otherwise an anchor-vs-baseline internal
audit can compare anchor `pearson_pert` computed against full-bundle perturbed
means with train-only residual baselines computed against train-only perturbed
means.

### Fix / status

`model.latent.eval_split_groups` now accepts an explicit `--pert-means-file`
override and records the loaded means files in the output JSON under
`means_files`. Default behavior is unchanged when the flag is omitted.

Validation:

```bash
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m pytest -q model/tests/test_latent_eval_selection.py
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m pytest -q model/tests/test_plan_guards.py
```

Future train-only internal-val anchor comparator jobs must pass the relevant
train-only pert means file explicitly and should not treat older internal-val
anchor pp summaries as final strict gates unless their means provenance is
verified.

## Cross-latent comparator launcher needed fail-closed parallel status and empty-GPU filtering

### Date

2026-06-22

### Symptom

The prepared cross-latent Track A anchor comparator launcher had two
orchestration risks before any launch. First, the parallel branch used a single
multi-PID `wait`, which can mask an earlier latent worker failure if a later
worker exits successfully. Second, the shared GPU helper may list
low-utilization GPUs with foreign compute processes as available candidates;
for this comparator, those should not be treated as empty launch targets.

### Fix / status

`ops/launch_latentfm_crosslatent_tracka_anchor_comparator_20260622.sh` now
collects each parallel latent worker's exit status and exits nonzero if any
worker fails. Its assignment step also requires chosen comparator GPUs to be
stable-light and to have zero own/foreign compute processes, while still
counting existing `cyx` active physical GPUs against the AGENTS budget.

Validation:

```bash
bash -n ops/launch_latentfm_crosslatent_tracka_anchor_comparator_20260622.sh
```

The launcher remains unlaunched until train-only baseline artifacts finish and
their report is reviewed.

## Track C smoke decision fallback treated empty support `test_multi` as usable

### Date

2026-06-22

### Symptom

The route-focused Track C routed-distill smoke posthoc completed, but the first
automatic decision reported `missing_or_bad_support_pp` and
`missing_or_bad_support_mmd` with zero support matched conditions. The support
posthoc JSONs did contain 24 support-val conditions under group `test`; the
route-dataset trainselect split intentionally had only `train`/`test` keys and
did not include a `test_multi` alias.

### Cause

`ops/summarize_latentfm_trackc_routed_distill_smoke_20260622.py` attempted a
Python `or` fallback from support `test_multi` to support `test`, but an
insufficient `test_multi` row is still a truthy dict. The fallback therefore
never reached the usable support `test` row.

### Fix / status

The summarizer now selects the first support row with `status == "ok"` and
`n_matched_conditions > 0`, falling back to `test` only when `test_multi` is
empty/unusable. The route-focused decision was regenerated from existing
support/canonical posthoc JSONs without rerunning model evaluation and without
reading held-out query. Corrected status:
`trackc_smoke_fail_canonical_harm_close_branch`.

Validation:

```bash
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m py_compile ops/summarize_latentfm_trackc_routed_distill_smoke_20260622.py
/data/cyx/software/miniconda3/envs/scdfm/bin/python ops/summarize_latentfm_trackc_routed_distill_smoke_20260622.py --run-root runs/latentfm_xverse_trackc_routefocused_distill_20260622/xverse_trackc_routefocus_condprior_w05_replay1_2k_seed42 --out-json reports/latentfm_trackc_routed_distill_smoke_decision_xverse_trackc_routefocus_condprior_w05_replay1_2k_seed42.json --out-md reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_xverse_trackc_routefocus_condprior_w05_replay1_2k_seed42.md --n-boot 2000 --seed 42 --python /data/cyx/software/miniconda3/envs/scdfm/bin/python
```

## Stack latent bundle manifest omitted condition_metadata_file

### Date

2026-06-22

### Symptom

The cross-latent train-only baseline build produced train-only pert means for
all three comparator latents and completed `scfoundation`/`scldm` baseline
gates, but failed the `stack` gate before metric computation:

```text
KeyError: 'condition_metadata_file'
```

`/data/cyx/1030/dataset/latentfm_full/stack/condition_metadata.json` exists,
but the stack manifest does not include the `condition_metadata_file` key that
the gene-reliability gate script expected.

### Fix / status

`ops/audit_latentfm_xverse_gene_reliability_router_gate_20260622.py` now
resolves condition metadata by first honoring `manifest.condition_metadata_file`
and then falling back to `<data-dir>/condition_metadata.json`. The gate report
records both `condition_metadata_file` and `condition_metadata_source` for
provenance.

A focused detached repair was launched instead of recomputing all latent
baselines:

```text
/data/cyx/1030/runs/latentfm_crosslatent_stack_baseline_repair_20260622/RUN_STATUS.md
```

Validation before launch:

```bash
python -m py_compile ops/audit_latentfm_xverse_gene_reliability_router_gate_20260622.py ops/repair_latentfm_crosslatent_stack_baseline_20260622.py
bash -n ops/launch_latentfm_crosslatent_stack_baseline_repair_20260622.sh
```
# Bug/Fix: Uncapped No-Harm Summarizer Overwrite Guard

## Date

2026-06-22 16:44 CST

## Bug

The shared Track C uncapped canonical no-harm summarizer did not refuse existing
final decision files or bootstrap directories before writing, even though later
query guards assume no accidental overwrite of frozen decision artifacts.

## Fix

`/data/cyx/1030/ops/summarize_latentfm_trackc_routefocus_uncapped_noharm_20260622.py`
now accepts parameterized report title/bootstrap dir arguments and refuses to
overwrite existing final JSON, final MD, or bootstrap directory. The endpoint
uncapped launcher also refuses to overwrite an existing manifest.

## Validation

`py_compile` passed. Endpoint launcher before smoke decision exists returned
`RC=2`; endpoint summarizer before uncapped index exists returned `RC=1`; no
endpoint uncapped artifacts were created.
# Bug: Track C Train-Only Memory Smoke Used 512-Cell Teacher Cap

## Date

2026-06-22 20:23 CST

## Context

The train-only memory CPU/preflight gate used `max_cells_per_condition=256`
for teacher target construction. The first memory-transfer GPU smoke launcher
did not explicitly set `CONDITION_PRIOR_BANK_MAX_CELLS`, so
`run_full_stack_latentfm.sh` inherited its default `512`.

## Impact

This is not query/canonical leakage, but it is a provenance mismatch: the
training teacher target vectors are not guaranteed to match the frozen CPU gate
teacher vectors. The first four `xverse_trackc_mem_*` runs trained to exit `0`,
but their posthoc watchers were stopped before decision generation, and the
runs are marked invalid for Track C gate decisions.

## Fix

`ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh` now accepts
and exports `LATENTFM_TRACKC_CONDITION_PRIOR_BANK_MAX_CELLS`.
`ops/launch_latentfm_trackc_trainonly_memory_parallel_20260622.sh` was switched
to an `mc256` run root/manifest and explicitly sets
`LATENTFM_TRACKC_CONDITION_PRIOR_BANK_MAX_CELLS=256`.

## Follow-Up

Use only the `mc256` relaunch block for train-only memory-transfer decisions.
Do not summarize or promote the earlier `xverse_trackc_mem_*` block.

# Bug/Fix: Track A Gene-Reliability Adapter Launcher GPU Override

## Date

2026-06-23 00:27 CST

## Bug

The first launch of
`ops/launch_latentfm_crosslatent_tracka_gene_reliability_adapter_20260623.sh`
assigned scFoundation to physical GPU0 and SCLDM to physical GPU1, but the
generated training script exported `GPU=0` for every run. The shared
`run_full_stack_latentfm.sh` then reset `CUDA_VISIBLE_DEVICES` from `GPU`,
putting both short-lived initial training processes on physical GPU0.

## Impact

No checkpoint completed and no posthoc/canonical decision was produced before
the issue was detected during the single allowed launch verification. This was
a resource-placement bug, not a data leakage issue.

## Fix

The launcher now writes `export GPU=<assigned physical gpu>` into each generated
training script. The SCLDM run script was patched from `GPU=0` to `GPU=1`, the
initial SCLDM training tmux was stopped, and SCLDM was restarted at
`2026-06-23 00:26:37 CST`.

Verification after restart showed:

```text
scfoundation_tracka_gene_shrink_k2_adapter_2k_seed42 -> CUDA_VISIBLE_DEVICES=0
scldm_tracka_gene_shrink_k4_adapter_2k_seed42 -> CUDA_VISIBLE_DEVICES=1
```

`nvidia-smi --query-compute-apps` confirmed the two training PIDs on GPU0 and
GPU1 respectively.

# Bug/Fix: Track C Residual Operator Wrapper Dropped Residual Flag

## Date

2026-06-23 02:12 CST

## Bug

The residual-operator wrapper set
`LATENTFM_TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL=1`, but the shared Track C
routed-distill launcher did not map that wrapper variable to the lower-level
`TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL` consumed by
`run_full_stack_latentfm.sh`.

The failed run was:

```text
xverse_trackc_residual_operator_memall_resid_ep050_replay2_2k_seed42
```

Its train log showed:

```text
trackc_support_context use_in_model=0 residual_use_in_model=0
RuntimeError: finetune_trainable_scope='support_residual_adapter' requires trackc_support_residual_use_in_model=True
```

## Impact

The run failed before training. No support-val/canonical decision was produced.
This is a launcher/config plumbing failure, not negative evidence against the
residual-operator mechanism.

## Fix

`ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh` now maps
`LATENTFM_TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL` into
`TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL`, exports it into generated train scripts,
and records `residual_use_in_model` in `RUN_STATUS.md`. The active exploration
GPU budget in the wrapper was also updated from `4` to the user-approved `5`.

Validation:

```bash
bash -n ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh
bash -n ops/launch_latentfm_trackc_residual_operator_smoke_20260623.sh
```

The corrected retry run:

```text
xverse_trackc_residual_operator_memall_resid_ep050_replay2_2k_seed42_retry1
```

verified at startup:

```text
trackc_support_context use_in_model=0 residual_use_in_model=1
Finetune scope: support_residual_adapter; trainable tensors=['support_context_to_v.weight', 'support_context_to_v.bias']
```

# Bug/Fix: GPU Helper Colocated Before Using Clean GPUs

## Date

2026-06-23 02:25 CST

## Symptom

Two low-util LatentFM jobs launched onto GPU0 even though several other GPUs
were clean.  The helper comment said clean/stably-light GPUs should be preferred
while physical GPU budget remains, but the rank function gave
`own_colocation_slot` a better rank than `clean`.

## Cause

`ops/select_available_gpus.py` ranked a stable own-occupied GPU before a clean
GPU.  `choose_job_gpus` also filled all free slots on the first candidate before
considering the next physical GPU.

## Fix

The helper now ranks clean GPUs before own low-util colocation while new
physical slots are available, and `choose_job_gpus` spreads one job across
eligible physical GPUs before filling remaining colocation slots.

Validation:

```bash
/data/cyx/software/miniconda3/envs/scdfm/bin/python ops/validate_gpu_availability_helper.py
```

Result:

```text
gpu availability helper validation passed
```

The subsequent broad Track A launch assigned GPU1 while GPU0 already contained
active `cyx` LatentFM jobs.

# Bug/Fix: Track C Support-FiLM Wrapper Missing Bank Split

## Date

2026-06-23 03:23 CST

## Symptom

The first support-FiLM smoke
`xverse_trackc_support_film_absroute_2k_seed42` failed before training with:

```text
support-context routed source requires trackc_routed_distill_bank_split_file so eval/posthoc cannot build context from the active evaluation split
```

## Cause

`ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh` only
received `LATENTFM_TRACKC_SUPPORT_FILM_USE_IN_MODEL=1`; it did not default
`TRACKC_ROUTED_DISTILL_BANK_SPLIT_FILE` to the safe trainselect split for the
new support-FiLM path.

## Fix

The wrapper now sets `TRACKC_BANK_SPLIT_FILE=${TRAINSELECT_SPLIT}` whenever any
support context/residual/film model path is active and the caller did not
provide an explicit bank split.

Validation:

```bash
bash -n ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh
```

This is launcher/config negative evidence only, not support-FiLM mechanism
evidence.

# Bug/Fix: Anchor-Gated Blend Launcher Lost Child PIDs

## Date

2026-06-23 06:39 CST

## Symptom

The first anchor-gated support-teacher blend posthoc launcher exited
immediately with:

```text
wait: pid ... is not a child of this shell
```

No evaluator processes remained alive afterward, so this consumed no GPU
experiment time and produced no mechanism evidence.

## Cause

The launcher captured `run_eval` output with command substitution.  Bash runs a
command substitution in a subshell, so the background eval processes were not
children of the parent shell that later called `wait`.

## Fix

The repaired launcher uses a global `RUN_EVAL_PID=$!` set by `run_eval` in the
current shell, then starts a provenance-preserving retry run named
`xverse_support_film_retry1_anchor_gated_blend_posthoc_ode20_retry1`.

Validation:

```bash
bash -n ops/launch_latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_20260623.sh
```

# Bug/Fix: Query Launcher Lacked 07:10 Not-Before Guard

## Date

2026-06-23 06:55 CST

## Symptom

The first pre-window dry run of
`ops/launch_latentfm_trackc_anchor_gated_blend_query_once_if_pass_20260623.sh`
did not immediately refuse before the scheduled `07:10 CST` posthoc-check
window.  It was interrupted before launching any tmux query job.  No query
processes or query outputs existed afterward, but a partial resource-audit
directory was created:

```text
runs/latentfm_trackc_anchor_gated_blend_query_once_20260623/resource_audit/
```

## Cause

The query launcher checked the full posthoc gate before enforcing the
not-before window.  If the posthoc report existed early, the script could enter
resource audit before the scheduled official check.

## Fix

The launcher now refuses before `2026-06-23 07:10:00 CST` before reading the
posthoc gate JSON or creating any retry run directory.  The launch label was
moved to `latentfm_trackc_anchor_gated_blend_query_once_20260623_retry1` to
preserve the partial dry-run evidence.

Validation:

```bash
bash -n ops/launch_latentfm_trackc_anchor_gated_blend_query_once_if_pass_20260623.sh
bash ops/launch_latentfm_trackc_anchor_gated_blend_query_once_if_pass_20260623.sh; rc=$?; echo RC=$rc
```

Result before `07:10 CST`:

```text
Refusing query launcher before 2026-06-23 07:10:00 CST
RC=3
```

# Bug/Fix: Track C CPU Gate Must Use scdfm Python

## Date

2026-06-23 14:47 CST

## Symptom

The first foreground run of
`ops/audit_latentfm_trackc_composition_subspace_operator_gate_20260623.py`
using bare `python` failed immediately:

```text
ModuleNotFoundError: No module named 'h5py'
```

No report was produced by the failed attempt and no GPU work was launched.

## Cause

The shell default `python` is not the LatentFM/scDFM environment. CPU gates
that import the support-route readiness module need packages such as `h5py`
from `/data/cyx/software/miniconda3/envs/scdfm/bin/python`.

## Fix

Use the project-standard interpreter and set `PYTHONPATH` when importing
CoupledFM modules:

```bash
PYTHONPATH=/data/cyx/1030/CoupledFM \
/data/cyx/software/miniconda3/envs/scdfm/bin/python \
ops/audit_latentfm_trackc_composition_subspace_operator_gate_20260623.py
```

Validation:

```bash
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m py_compile \
  ops/audit_latentfm_trackc_composition_subspace_operator_gate_20260623.py \
  ops/audit_latentfm_trackc_composition_module_coherence_gate_20260623.py \
  ops/audit_latentfm_trackc_composition_pair_geometry_gate_20260623.py
```

# Bug/Fix: Track A CORUM Rule Name Parser

## Date

2026-06-24 00:24 CST

## Symptom

The first run of
`ops/audit_latentfm_tracka_corum_complex_reliability_gate_20260624.py` failed
before writing a report:

```text
ValueError: invalid literal for int() with base 10: 'degree'
```

No GPU work was launched.

## Cause

The rule parser split `degree_ge1_use_gene` on `_` and attempted to parse the
first token, `degree`, instead of the full `degree_ge1` prefix.

## Fix

Parse thresholds by removing the `_use_gene` suffix first, then stripping
`degree_ge` or `degree_lt`.

Validation:

```bash
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m py_compile \
  ops/audit_latentfm_tracka_corum_complex_reliability_gate_20260624.py
/data/cyx/software/miniconda3/envs/scdfm/bin/python \
  ops/audit_latentfm_tracka_corum_complex_reliability_gate_20260624.py
```

The rerun completed and wrote
`/data/cyx/1030/reports/LATENTFM_TRACKA_CORUM_COMPLEX_RELIABILITY_GATE_20260624.md`.

# Bug/Fix: xverse Train-Strategy Launcher GPU Mapping

## Date

2026-06-24 00:59 CST

## Symptom

The first launch of
`ops/launch_latentfm_xverse_train_strategy_smokes_20260624.sh` assigned jobs to
physical GPUs `1,2,4`, but the child train scripts exported `GPU=0` before
calling `CoupledFM/model/latent/scripts/run_full_stack_latentfm.sh`. That
wrapper sets `CUDA_VISIBLE_DEVICES="${GPU}"`, so all three jobs started on
physical GPU0.

## Cause

The batch launcher mixed two conventions:

- direct eval scripts use `CUDA_VISIBLE_DEVICES=<physical>` plus `--gpu 0`;
- `run_full_stack_latentfm.sh` expects `GPU=<physical>` and sets
  `CUDA_VISIBLE_DEVICES` itself.

## Fix

The three just-started tmux sessions were stopped immediately and the partial
run/output directories were moved to:

```text
runs/latentfm_xverse_train_strategy_smokes_20260624/bad_gpu_mapping_20260624_005928
CoupledFM/output/latentfm_runs/xverse_train_strategy_smokes_20260624/bad_gpu_mapping_20260624_005928
```

The launcher now exports `GPU=${gpu}` for the train wrapper while keeping
posthoc direct-eval commands on `CUDA_VISIBLE_DEVICES=${gpu}` with `--gpu 0`.

Validation:

```bash
bash -n ops/launch_latentfm_xverse_train_strategy_smokes_20260624.sh
```

# Bug/Fix: LatentFM Full-Stack Launcher Did Not Propagate ds_alpha

## Date

2026-06-24 01:14 CST

## Symptom

`ops/launch_latentfm_xverse_train_strategy_smokes_20260624.sh` exported
per-run `DS_ALPHA` values, but
`CoupledFM/model/latent/scripts/run_full_stack_latentfm.sh` did not pass
`--ds-alpha` into `model.latent.train`.

The already launched train-strategy batch should therefore be interpreted from
the saved `config.json`/logs. In particular,
`xverse_trainstrat_balanced_dsalph045_floor48_cap6_3k_seed42` documents an
intended `ds_alpha=0.45`, but its actual training config may remain the
default `0.7`.

## Cause

`Config.ds_alpha` is a tyro CLI field, but the wrapper exposed only
`DS_LOSS_ALPHA`, `MIN_SELECTED_CONDITIONS_PER_DATASET`,
`CONDITION_VISIT_POWER`, and `CONDITION_VISIT_CAP`.

## Fix

The wrapper now defines `DS_ALPHA="${DS_ALPHA:-0.7}"`, echoes it in launcher
logs, and passes `--ds-alpha "${DS_ALPHA}"`.

Validation:

```bash
bash -n CoupledFM/model/latent/scripts/run_full_stack_latentfm.sh
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m py_compile \
  CoupledFM/model/latent/train.py CoupledFM/model/latent/config.py
```
# Bug: SciPlex Morgan Cache TSV Format Incompatible With DrugEmbeddingCache

## Date

2026-06-24

## Symptom

The first Morgan descriptor cache wrote `drug_index.tsv` as a rich metadata
table beginning with `index<TAB>drug...`. `DrugEmbeddingCache` loads TSV before
JSON and expects `key<TAB>index`; the metadata table would be parsed as an index
file and fail.

## Fix

Updated `/data/cyx/1030/ops/build_latentfm_sciplex_morgan_descriptor_cache_20260624.py`
to write a loader-compatible `drug_index.tsv` and move rich SMILES/scaffold/
dose/pathway/target annotations into `drug_metadata.tsv`. Rebuilt the cache and
verified `DrugEmbeddingCache` lookup for `A366` hits with dim `2048`.
# Bug: true cell-count capped-H5 manifests lacked top-level emb_dim

## Date

2026-06-24 22:52 CST

## Symptom

The first three true-cell-count exploratory smoke launches exited immediately
with exit code `2`. Launcher logs reported:

```text
Could not infer emb_dim from .../manifest.json; set EMB_DIM explicitly.
```

## Cause

`ops/materialize_latentfm_true_cell_count_capped_h5_20260624.py` produced valid
per-dataset capped H5 files, but the artifact-level `manifest.json` lacked the
top-level `emb_dim` field expected by
`CoupledFM/model/latent/scripts/run_full_stack_latentfm.sh`.

## Fix

`ops/backfill_latentfm_true_cell_count_sample_provenance_20260624.py` now
backfills `emb_dim` from the materialized capped H5 files while writing sampled
row provenance. The post-materialization gate runner was rerun and passed:
sample provenance, schema/provenance, dry-load, and design controls.

## Follow-up

Future materializer revisions should write `emb_dim` directly during artifact
creation. Current repaired artifacts report `emb_dim=384`.

# Bug: true cell-count capped-H5 manifests stored conditions as counts

## Date

2026-06-24 23:02 CST

## Symptom

The seed42 true cell-count trainings finished with `EXIT_CODE=0`, but all three
initial posthoc evaluations failed with:

```text
TypeError: 'int' object is not iterable
```

from `CoupledFM/model/latent/eval_split_groups.py` while reading
`manifest["datasets"][dataset]["conditions"]`.

## Cause

`ops/materialize_latentfm_true_cell_count_capped_h5_20260624.py` wrote each
dataset manifest entry as `"conditions": <count>`, while LatentFM dataset and
posthoc evaluators expect `"conditions"` to be an iterable list of condition
names.

## Fix

The materializer now writes `"conditions": [...]` plus `"n_conditions"`.
`ops/backfill_latentfm_true_cell_count_sample_provenance_20260624.py` repairs
already materialized artifacts from each H5 `conditions` dataset while
backfilling provenance. The capped-H5 schema gate now fails if manifest
conditions are not a list.

Validation:

```bash
bash /data/cyx/1030/ops/run_latentfm_true_cell_count_post_materialization_gates_20260624.sh
```

All post-materialization gates passed after repair, and the seed42 posthoc
evaluations were relaunched from existing checkpoints without retraining.

# Bug: nested true cell-count provenance backfill used non-nested sampling

## Date

2026-06-24 23:35 CST

## Symptom

The first nested-v2 post-materialization gate run reported design controls with
`nested sampling status: fail` even though the nested materializer dry-run and
artifact materialization were intended to use deterministic permutation-prefix
sampling.

## Cause

`ops/run_latentfm_true_cell_count_nested_post_materialization_gates_20260624.sh`
reused `backfill_latentfm_true_cell_count_sample_provenance_20260624.py`
without replacing its `sample_indices` function. That rewrote
`sampled_indices.npz` provenance using the non-nested sampling rule, so the
design-control nestedness check correctly failed.

## Fix

The nested gate runner now imports
`materialize_latentfm_true_cell_count_nested_capped_h5_20260624.py` and patches
the provenance backfill to use `nested_sample_indices`.

Validation:

```bash
bash /data/cyx/1030/ops/run_latentfm_true_cell_count_nested_post_materialization_gates_20260624.sh
```

The rerun reported `nested sampling status: ok` and `warnings: none` before any
nested-v2 GPU matrix was launched.

## Nested true-cell-count summary/control scripts treated missing metric values as numeric

### Date

2026-06-25

### Symptom

After all 9 nested true-cell-count train/posthoc runs exited `0`, both the nested matrix decision script and nested controls gate failed with:

```text
TypeError: float() argument must be a string or a real number, not 'NoneType'
```

The posthoc condition-level rows may contain `None` for a metric; the scripts attempted to convert every present key directly with `float(...)`.

### Fix / status

Patched:

```text
/data/cyx/1030/ops/summarize_latentfm_true_cell_count_nested_matrix_20260624.py
/data/cyx/1030/ops/audit_latentfm_true_cell_count_nested_controls_20260624.py
```

to skip missing or non-numeric condition-level metric values when constructing paired metric maps. This does not impute zeros or change decision thresholds.

Validation:

```bash
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m py_compile \
  /data/cyx/1030/ops/summarize_latentfm_true_cell_count_nested_matrix_20260624.py \
  /data/cyx/1030/ops/audit_latentfm_true_cell_count_nested_controls_20260624.py
/data/cyx/software/miniconda3/envs/scdfm/bin/python \
  /data/cyx/1030/ops/summarize_latentfm_true_cell_count_nested_matrix_20260624.py
/data/cyx/software/miniconda3/envs/scdfm/bin/python \
  /data/cyx/1030/ops/audit_latentfm_true_cell_count_nested_controls_20260624.py
```

The rerun produced:

```text
/data/cyx/1030/reports/LATENTFM_TRUE_CELL_COUNT_NESTED_MATRIX_DECISION_20260624.md
/data/cyx/1030/reports/LATENTFM_TRUE_CELL_COUNT_NESTED_CONTROLS_GATE_20260624.md
```

## Nested true-cell-count parser needed tail-stability run-name compatibility

### Date

2026-06-25 00:20 CST

### Symptom

The 6k budget128 tail-stability follow-up uses run names like:

```text
xverse_truecell_nested_budget128_tailstable_seed42_6000
```

The original nested matrix summarizer/control parser only recognized names
containing `_budget128_seed42`. A first broadening attempt using a greedy regex
would have incorrectly parsed old nested names such as:

```text
xverse_truecell_nested_gene_only_fixed256_budget64_128_256_budget128_seed42_3000
```

as budget64 rather than budget128.

### Fix / status

Patched:

```text
/data/cyx/1030/ops/summarize_latentfm_true_cell_count_nested_matrix_20260624.py
/data/cyx/1030/ops/audit_latentfm_true_cell_count_nested_controls_20260624.py
```

to parse the seed first, then use the last `_budget<N>` token before the seed.
This supports both the original nested matrix names and the new 6k
tail-stability names.

Validation:

```text
xverse_truecell_nested_budget128_tailstable_seed42_6000 -> (128, 42)
xverse_truecell_nested_gene_only_fixed256_budget64_128_256_budget128_seed42_3000 -> (128, 42)
xverse_truecell_nested_gene_only_fixed256_budget64_128_256_budget256_seed44_3000 -> (256, 44)
```

## All-modality dose-aware metadata backfill wrote gene perturbations as NULL-mapped type

### Date

2026-06-25 12:44 CST

### Symptom

The first all-modality dose-aware GPU smoke launch emitted:

```text
Unrecognized perturbation_type 'gene' -> PERT_TYPE_NULL
```

for gene perturbation rows. The smoke was otherwise training, which made this a
silent-semantics failure rather than an immediate crash.

### Cause

`ops/backfill_latentfm_true_cell_count_allmodality_doseaware_condition_metadata_20260625.py`
generated gene entries with the generic `perturbation_type_raw=gene` instead
of inheriting canonical xverse per-condition metadata. Canonical gene datasets
use concrete perturbation types such as `CRISPRi`, `CRISPRa`, or `CRISPRko`,
and the LatentFM mapper does not treat generic `gene` as a valid type.

### Fix / status

Patched the backfill script to strictly inherit gene entries from:

```text
/data/cyx/1030/dataset/latentfm_full/xverse/condition_metadata.json
```

and to fail the gate if a gene condition is missing from canonical metadata.
SciPlex dose-level chemical rows still use the dose-aware sidecar fields and
resolve through the Morgan512 projected drug cache.

Validation:

```text
Adamson sample -> CRISPRi
NormanWeissman2019_filtered sample -> CRISPRa
TianActivation sample -> CRISPRa
sciplex3_A549 sample -> drug with chem_obs_value=2Methoxyestradiol
```

The post-materialization wrapper reran successfully and the corrected smoke
launch reached early training steps with Morgan512 cache hits and no gene-type
warning.

### Affected runs

The first four all-modality dose-aware smoke sessions launched at 12:40 CST were
stopped at 12:42 CST and marked superseded in their `RUN_STATUS.md` files before
any gate decision. The corrected four-run slate was relaunched at 12:44 CST.

## 2026-06-25: all-modality dose-aware posthoc eval returned zero conditions

### Symptom

The first all-modality dose-aware smoke decision report showed
`allmodality_doseaware_smokes_fail_close`, but all metrics were `NA`. This was
not a valid biological/model failure: posthoc evaluation had selected zero
effective conditions.

### Cause

The materialized all-modality manifests expose per-dataset `path` and
`n_conditions`, while the posthoc evaluation path expected each manifest entry
to include an explicit `conditions` list. As a result,
`eval_split_groups._group_as_test_split` and downstream family evaluation
filtered requested split conditions against an empty allowed-condition set.

### Fix / status

Patched:

```text
/data/cyx/1030/CoupledFM/model/latent/eval_split_groups.py
```

so `_load_manifest` backfills missing `conditions` by reading the H5
`conditions` dataset from `<data_dir>/<dataset>.h5`. Because
`eval_condition_families.py` imports `_load_manifest`, family evaluation uses
the same fix.

Validation:

```text
python -m py_compile eval_split_groups.py eval_condition_families.py
Adamson conditions: 20
split test count: 484
family_gene: 24
family_drug/type_drug: 460
```

Detached posthoc rerun:

```text
/data/cyx/1030/runs/latentfm_true_cell_count_allmodality_doseaware_smokes_20260625/POSTHOC_MANIFEST_FIX_RUN_STATUS.md
```

The old `NA` decision must not be used to close the all-modality branch.

## 2026-06-26: Dixit GEO guide dictionary CSV hit Python field-size limit

### Symptom

The post-source reagent/read-support pipeline failed while extracting Dixit GEO
guide-barcode artifacts:

```text
_csv.Error: field larger than field limit (131072)
```

This happened after Frangieh extraction succeeded and before combined gates ran.

### Cause

`GSE90063_RAW.tar` contains guide-barcode dictionary rows with fields longer
than Python's default `csv` module field limit. The source file was complete;
this was a parser limit, not a biological/source gate failure.

### Fix / status

Patched:

```text
/data/cyx/1030/ops/extract_latentfm_dixit_geo_reagent_artifacts_20260626.py
```

to call:

```python
csv.field_size_limit(sys.maxsize)
```

Validation:

```text
python -m py_compile /data/cyx/1030/ops/extract_latentfm_dixit_geo_reagent_artifacts_20260626.py
bash /data/cyx/1030/ops/run_latentfm_reagent_read_support_post_source_pipeline_20260626.sh
```

The rerun completed all Frangieh/Dixit extractions and downstream CPU gates.

## 2026-06-27: allowlisted-tail posthoc failed after fail-closed metadata gate

### Symptom

The allowlisted-tail seed42/43 training runs both finished with `EXIT_CODE=0`,
but canonical Track A posthoc failed with:

```text
RuntimeError: condition_delta_in_model_filter requires pert_type_id and chem_mask
```

### Cause

The model-side allowlist gate was correctly changed to fail closed when
non-`all` condition-delta filters lack perturbation metadata. The eval path for
no-chemical gene conditions represented the chemical mask as `None`, so the
safer model gate treated the metadata as incomplete.

### Fix / status

Patched:

```text
/data/cyx/1030/CoupledFM/model/latent/train.py
```

so `_pert_to_device` creates an explicit false boolean `chem_mask` for
no-chemical conditions. The model-side fail-closed checks remain in place.

Validation:

```text
python -m py_compile CoupledFM/model/latent/train.py CoupledFM/model/latent/models/mlp.py
explicit no-chemical mask test: bool shape (batch, 1), sum 0
allowlisted gate tensor test: [True, False, False]
```

Posthoc was relaunched via:

```text
/data/cyx/1030/ops/launch_latentfm_allowtail_posthoc_rerun_after_chemmask_fix_20260627.sh
```

The first rerun launch also exposed a launcher bug: it used
`/data/cyx/1030/software/miniconda3/...` instead of
`/data/cyx/software/miniconda3/...`, and shell exit capture incorrectly returned
0 when earlier commands failed. The launcher now uses the correct Python path
and wraps the posthoc body in a fail-fast subshell before writing
`POSTHOC_RERUN_EXIT_CODE`.

## 2026-06-27: Track C support-set posthoc mixed eval seeds inflated seed43 delta

### Symptom

The seed43 shared-gene support-set smoke appeared to improve support-val
`pearson_pert` by about `+0.0089`, but actual/zero/shuffle/absent controls all
moved by the same amount. This looked like non-specific drift even though the
run was configured to train only `support_set_task_to_c.weight`.

### Cause

The posthoc script compared the anchor checkpoint using its checkpoint config
seed `42` against the seed43 candidate/control checkpoint using seed `43`.
`model.latent.train.evaluate()` uses `cfg.seed` for deterministic per-condition
eval cell subsampling, so candidate-minus-anchor deltas included eval-subset
mismatch.

Weight audit showed candidate raw non-support weights exactly matched the
anchor EMA baseline, and the optimizer had only one state entry for
`support_set_task_to_c.weight`. The issue was evaluation protocol, not backbone
training drift.

### Fix / status

Patched:

```text
/data/cyx/1030/CoupledFM/model/latent/eval_split_groups.py
/data/cyx/1030/CoupledFM/model/latent/eval_condition_families.py
```

Both CLIs now accept:

```text
--eval-seed <int>
```

which overrides `cfg.seed` for deterministic eval cell subsampling and records
`eval_seed_override` in the output JSON.

Validation:

```text
python -m py_compile CoupledFM/model/latent/eval_split_groups.py CoupledFM/model/latent/eval_condition_families.py
```

Corrected seed43 posthoc was rerun with `--eval-seed 43` for both anchor and
candidate/control evals:

```text
/data/cyx/1030/reports/LATENTFM_TRACKC_SUPPORT_SET_SHAREDGENE_DECISION_xverse_trackc_support_set_sharedgene_adapter_2k_seed43_EVALSEED43.md
```

Corrected result: actual pp `+0.000002`, zero pp `0`, absent pp `0`, and
anchor-vs-absent row-level max pp difference `0`. The branch remains
fail-closed, now with the stronger interpretation that the support adapter was
near-inert rather than non-specifically beneficial.
# Bug/Fix: Track C Focused Support-Set Summarizer Expected Original Split

Date: 2026-06-27 15:39 CST

Context: `xverse_trackc_support_set_focused_min2_adapter_2k_seed42` trained and
all posthoc eval JSONs were generated successfully, but the wrapper exited with
`POSTHOC_EXIT_CODE=1`.

Bug: `ops/summarize_latentfm_trackc_support_only_robustness_20260624.py`
defaults `--expected-split-file` to the original safe trainselect split. The
focused run correctly evaluated on
`split_seed42_multi_support_v2_trainselect_supportset_min2_focused.json`, so
the summarizer rejected all posthoc JSONs as split mismatches.

Fix: patched
`ops/launch_latentfm_trackc_support_set_smoke_20260627.sh` to pass
`--expected-split-file ${TRAIN_SPLIT}` to the summarizer. Manually reran the
summarizer on the existing posthoc JSONs with the focused split; decision rerun
exit code `0`.

Outcome: the focused support-set scientific gate still failed
(`actual pp +0.000130`, Wessels `-0.000005`), so no relaunch/seed extension is
authorized.

# Bug/Fix: True-Cell Count Launcher Needed Documented GPU Override

Date: 2026-06-28 00:56 CST

Context: budget256 seed42 was active on GPU2 and a fresh external three-sample
audit showed GPU3 was the cleanest physical GPU for budget256 seed43
(`28 MiB / 0%` in all samples). The generic GPU helper could still prefer
colocation or a less clear choice under tight mixed-occupancy conditions.

Fix: patched `ops/launch_latentfm_true_cell_count_single_smoke_20260624.sh` to
honor optional `LATENTFM_TRUE_CELL_COUNT_GPU_OVERRIDE=<physical_gpu>`. When set,
the launcher records an override audit JSON and writes the selected physical
GPU into `RUN_STATUS.md`. Default helper behavior is unchanged.

Validation:

```text
bash -n ops/launch_latentfm_true_cell_count_single_smoke_20260624.sh
```

Outcome: budget256 seed43 was launched on GPU3 with explicit override
provenance. This is an operational provenance fix only; it does not change
model semantics, split boundaries, or evaluation rules.

# Bug/Fix: ZSCAPE Metadata Download Needed Robust Resume

Date: 2026-06-28 04:14 CST

Context: the first detached ZSCAPE metadata coverage run
`zscape_metadata_coverage_20260628_0404` downloaded most of
`GSE202639_zperturb_full_cell_metadata.csv.gz` but exited with code `56`.

Bug: `curl` encountered an OpenSSL unexpected EOF near the end of the 303 MB
metadata file. The existing command used `-C -` and basic retry, but did not
retry this transport error.

Fix: patched `ops/run_zscape_metadata_coverage_audit_20260628.sh` to use
stronger resumable download settings:

```text
--retry 10 --retry-delay 20 --retry-all-errors --retry-connrefused
--speed-time 180 --speed-limit 1024 -C -
```

Relaunch: started
`zscape_metadata_coverage_20260628_0414_resume1`, which resumed the partial
file from byte `289761647`.

Outcome: this is an operational network/provenance fix only; it does not
change the metadata coverage gate, biological interpretation, or any model
training/evaluation semantics.

# Bug/Fix: ZSCAPE Reference Coverage Gate Must Not Require Perturbation Cells

Date: 2026-06-28 04:18 CST

Context: `ops/audit_zscape_metadata_coverage_20260628.py` scans both the
ZPERTURB perturbation atlas and the ZSCAPE reference atlas to identify shared
cell-type lineages for continuity/OT analysis.

Bug: the initial broad-cell-type candidate rule required perturbation cells and
perturbation targets for both datasets. That is correct for `zperturb_full`,
but wrong for the reference atlas, which is a control/developmental reference.
This would have caused a false metadata coverage failure even when the
reference atlas had strong multi-timepoint cell-type coverage.

Fix: patched the coverage audit so `zperturb_full` requires control and
perturbation coverage, while `reference` requires enough cells, multiple
timepoints, embryos, and subtype support. The shared-candidate gate still
requires the cell type to pass both dataset-specific rules.

Validation:

```text
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m py_compile \
  ops/audit_zscape_metadata_coverage_20260628.py \
  ops/plan_zscape_continuity_ot_gate_from_coverage_20260628.py
```

Outcome: fixes gate semantics only; no model training, inference, canonical
multi, Track C query, or GPU use.

# Bug/Fix: ZSCAPE Raw-Count Worker Failed After Full Transfer During Live Script Edit

Date: 2026-06-28 07:45 CST

Context: `zscape_raw_counts_minimal_download_20260628_044305` was a detached
network/disk-only download of `GSE202639_zperturb_full_raw_counts.RDS.gz`.
During the long download, the worker script was edited to reduce future curl
log noise.

Bug: the running bash process finished the full curl transfer, but then exited
with `EXIT_CODE=127` before writing the normal post-download SHA/gzip report.
The log showed transfer reached `100%`; the likely cause is a script-read
race from editing the worker while it was still running, not evidence of data
corruption.

Fix: preserved the original `EXIT_CODE=127`, then manually validated the
downloaded file with:

```text
sha256sum GSE202639_zperturb_full_raw_counts.RDS.gz
gzip -t GSE202639_zperturb_full_raw_counts.RDS.gz
stat -c '%n\t%s bytes' GSE202639_zperturb_full_raw_counts.RDS.gz
```

Manual validation passed. SHA256 and file size are recorded in:

```text
/data/cyx/1030/runs/zscape_raw_counts_minimal_download_20260628/zscape_raw_counts_minimal_download_20260628_044305/SHA256SUMS
/data/cyx/1030/runs/zscape_raw_counts_minimal_download_20260628/zscape_raw_counts_minimal_download_20260628_044305/outputs/downloaded_file_size.txt
```

Outcome: the raw-count source is treated as validated for the guarded
cell-manifest extraction, but the original worker exit code remains preserved
for provenance. Future long-running worker scripts should not be edited in
place while their bash process is still active; patch a copied/future launcher
or wait until the job exits.

# Bug/Fix: ZSCAPE Strict OT Gate Needed Cell-Level Manifest Alignment

Date: 2026-06-28 08:29 CST

Context: the first strict-controls launch
`zscape_expression_ot_strict_controls_gate_20260628_082527` was intended to run
the 500-null CPU expression OT gate after the external audit.

Bug: launch sanity check failed immediately because the expression matrix is
unique-cell indexed (`31245` cells), while the matched manifest keeps row-level
cell references for all biological hypothesis rows (`33847` rows). The script
used the row-level manifest as the cell-level mask for control-only HVG/SVD,
causing a boolean-index shape mismatch.

Fix: patched
`ops/audit_zscape_expression_ot_strict_controls_20260628.py` so embedding
fitting uses a unique cell-level manifest that exactly matches expression
matrix columns, while row-level entries are preserved for per-hypothesis
statistics and controls.

Validation:

```text
python -m py_compile ops/audit_zscape_expression_ot_strict_controls_20260628.py
bash -n ops/launch_zscape_expression_ot_strict_controls_gate_20260628.sh
/data/cyx/software/miniconda3/envs/scdfm/bin/python \
  ops/audit_zscape_expression_ot_strict_controls_20260628.py \
  --n-hvg 500 --n-pca 16 --ot-cells 64 --null-repeats 1 ...
```

The short software smoke wrote report/CSV/JSON outputs in
`runs/zscape_expression_ot_strict_controls_gate_20260628/_smoke_after_index_fix/`.
Its gate failure status is expected for a one-null software smoke and is not
used as biological evidence.

Outcome: relaunched the formal 500-null strict gate as
`zscape_expression_ot_strict_controls_gate_20260628_082748`. This fix changes
index alignment only; it does not alter the biological gate definition,
training data, canonical evaluation, or any model behavior.

# Bug/Fix: ZSCAPE Concordance Direction-Shuffle Used Stale DataFrame Index

Date: 2026-06-30 02:10 CST

Context: the first run of
`ops/audit_latentfm_zscape_condition_response_neighborhood_gate_20260630.py`
failed while constructing direction-shuffle nulls for the ZSCAPE-inspired
condition response-neighborhood gate.

Bug: `load_rows()` filters the condition-neighborhood table but preserves the
original DataFrame index. The null-shuffle code grouped by stratification keys
and used those labels as positional numpy indices, causing an out-of-bounds
index when the filtered table had fewer positional rows than original labels.

Fix: reset the DataFrame index at the start of `prepare_scores()` before
building positional permutation arrays:

```text
work = rows.copy().reset_index(drop=True)
```

Validation:

```text
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m py_compile \
  ops/audit_latentfm_zscape_condition_response_neighborhood_gate_20260630.py
env OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 \
  /data/cyx/software/miniconda3/envs/scdfm/bin/python \
  ops/audit_latentfm_zscape_condition_response_neighborhood_gate_20260630.py
```

The rerun completed and wrote
`/data/cyx/1030/reports/zscape_condition_response_neighborhood_gate_20260630/LATENTFM_ZSCAPE_CONDITION_RESPONSE_NEIGHBORHOOD_GATE_20260630.md`.

Outcome: reporting/provenance fix only. The completed gate status is
`zscape_condition_response_neighborhood_gate_blocks_gpu`; no training,
inference, canonical multi, Track C query, or GPU use was involved.

# Bug/Fix: ZSCAPE Pairability Atlas Strict-Context Count Misclassified

Date: 2026-06-30 03:20 CST

Context: the first completed
`ops/audit_zscape_dynamic_pairability_atlas_20260630.py` run expanded the
ZSCAPE OT pairability atlas to all 25 selected rows.

Bug: `has_strict_control_context` was defined from whether `strict_row_gate`
passed, not whether a row had strict-control columns at all. This misreported
`strict_context_rows=3` and incorrectly labeled strict-control failures as
`atlas_row_no_strict_context_yet`.

Fix: changed `has_strict_control_context` to `strict_row_gate.notna()` and
ordered `pairability_class` assignment so atlas-only labels apply only to rows
with no strict-control context.

Validation:

```text
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m py_compile \
  ops/audit_zscape_dynamic_pairability_atlas_20260630.py
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 \
  /data/cyx/software/miniconda3/envs/scdfm/bin/python \
  ops/audit_zscape_dynamic_pairability_atlas_20260630.py \
  --ot-cells 64 --n-hvg 1500 --n-pca 24 \
  --out-dir /data/cyx/1030/reports/zscape_dynamic_pairability_atlas_20260630 \
  --force
```

Outcome: corrected report now has `strict_context_rows=10` and
`atlas_only_rows_without_strict_controls=15`. This is a reporting/classification
fix only; no training, inference, GPU, canonical multi, or Track C query use was
involved.

# Bug/Fix: ZSCAPE Reusable Gate Provenance And Threshold Drift

Date: 2026-06-30 16:52 CST

Context: Dewey external code/provenance audit inspected the recent ZSCAPE
prospective strict-control and focused specificity branch after the user asked
for a solidity review.

Issues:

* `ops/audit_zscape_crossfit_residual_specificity_repair_gate_20260628.py`
  used a hard-coded biological pass condition that could disagree with
  downstream row-threshold decision wrappers in future reuse cases.
* Several report/decision scripts could overwrite nonempty output directories
  if run directly.
* Prospective posthoc wrappers did not validate that interpreted row IDs matched
  the frozen upstream candidate/pass rows.
* `ops/audit_zscape_pairability_strict_control_expansion_20260630.py` had
  defaults pointing to the older 25-row branch, making bare reruns ambiguous.

Fix:

* Added crossfit output overwrite guard plus `--row-pass-fraction` and
  `--min-pass-rows`, with row pass count/threshold reported in JSON/Markdown.
* Added overwrite guards and row-ID provenance validation to prospective strict
  and partial-specificity decision wrappers.
* Required explicit input paths and `--max-rows` for the strict-control
  expansion script.

Validation:

```text
/data/cyx/software/miniconda3/envs/scdfm/bin/python -m py_compile \
  ops/audit_zscape_crossfit_residual_specificity_repair_gate_20260628.py \
  ops/synthesize_zscape_prospective_partial_specificity_decision_20260630.py \
  ops/synthesize_zscape_prospective_strict_control_decision_20260630.py \
  ops/audit_zscape_pairability_strict_control_expansion_20260630.py
```

Outcome: tooling/provenance hardening only. The current ZSCAPE branch decisions
are unchanged because observed outcomes were clear fail/partial-fail: strict
control `3/31` below the predeclared broad gate and focused specificity `0/3`.
