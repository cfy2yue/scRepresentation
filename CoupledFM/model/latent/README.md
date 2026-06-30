# latent — latent-space flow matching

Conditional OT flow matching in the **latent (embedding) space** on
**control → GT** pairs. Used standalone as a baseline and as the frozen
latent FM referenced by `model/train.py` (teacher for `aux_emb`).

Top-level overview: [`../README.md`](../README.md).

## Data

1. Run `data/scFM/state/exp_emb/` steps 1–4 → produces `data/biFlow_data/`.
2. `python prepare_fm_data.py` → packs control/GT embeddings into
   `latent/fm_data/*.h5` (HDF5 keys `ctrl/emb`, `gt/emb`).

scFMBench exports can also be packed directly:

```bash
python -m model.latent.prepare_scfm_fm_data \
  --embeddings-root /data/cyx/1030/scLatent/scFM_output/embeddings \
  --model stack \
  --out-dir /data/cyx/1030/dataset/latentfm/stack
```

For chemical perturbation datasets, preserve condition semantics so drug names
are not parsed as gene symbols:

```bash
python -m model.latent.prepare_scfm_fm_data \
  --model state \
  --datasets sciplex3_A549 sciplex3_K562 sciplex3_MCF7 sciplex3_xCellLine \
  --out-dir /data/cyx/1030/dataset/latentfm/state \
  --perturbation-type drug
```

The converter writes `condition_metadata.json`; `CrossDatasetFMDataset` reads it
before falling back to h5ad metadata or string parsing.

Default `data_dir`: `${COUPLEDFM_ROOT}/latent/fm_data`.

## Training

```bash
cd /data2/cfy/FM/CoupledFM/latent

# single experiment
PYTHON=python GPU=0 RUNS=./runs/baseline bash scripts/run_baseline.sh

# baseline sweep (3 experiments on 3 GPUs)
GPU_1=0 GPU_2=1 GPU_3=2 bash scripts/run_baseline_v3_6exp.sh
```

Scripts accept `PYTHON`, `RUNS`, and per-run `GPU_*` env vars (see each
script's header).

## Features

- **Naming (OT / manifests)**：源池侧指标与字段以 `src` 为准（如
  `pearson_src`、`n_src`、`read_src`）。`pearson_ir`、`read_ir` 等仍可作为
  兼容别名（训练脚本会对弃用项告警或自动映射）。
- **EMA (decay 0.999)** for stability; shadow weights activated for
  evaluation via `ema.apply_to(model)`; serialised in checkpoint.
- **bf16 AMP** (`use_amp=True`, `amp_dtype="bf16"`) — forward passes run
  under `torch.autocast`.
- **Cosine-with-min-LR schedule** (`utils/train/schedulers.py` via train loop).
- **DiffPerceiver backbone** (`models/diff_perceiver.py`) — DiT-style
  perceiver with differential self/cross-attention; uses its own
  `MultiheadDiffAttn` (not the one in `utils/models/attention.py`).

See [`config.py`](config.py) for all hyperparameters.

## Checkpoints

Written under `runs/baseline/<exp_name>/`:

- `best.pt` — best by validation metric; includes an `"ema"` key when EMA is enabled.
- `latest.pt` / end-of-run saves — same dict layout as training checkpoints (`model`, `optimizer`, optional `ema`, `config`).
- `train_log.jsonl` — per-epoch metrics.

CoupledFM's :class:`~model.latent_utils.FrozenLatentFM` loads ``ckpt["model"]`` from any of these files (plus sidecar `config.json`).

## Using as teacher for CoupledFM

`model/latent_utils.py::FrozenLatentFM` loads a latent checkpoint (typically ``best.pt`` or ``latest.pt`` from above) and
exposes `ode_at_t(z_ctrl, t)` / `ode_step(z, z_ctrl, t, dt)` for raw FM
teacher forcing (latent_z_mode = `ode` or `curriculum`).

## Perturb condition + chem extension

Gene-level conditioning uses `condition_emb/genepert` (via `GeneEmbeddingCache` +
`use_pert_condition`). Optional chemical perturbation vectors are wired via
`chem_emb_source_dir`; enable with `pert_chem_enabled=true` — see [`config.py`](config.py).

For formal comparisons, keep the genetic perturbation cache fixed across
LatentFM, raw FM, and CoupledFM runs. The current primary branch uses
`scgpt_embed_gene` for all three; CellNavi remains supported only as an
explicit sensitivity / reproduction source. Mixing them within one experiment
matrix confounds the model comparison with condition-embedding-source effects.
Record both `pert_gene_emb_cache_dir` and `PERT_EMBED_SOURCE` (or the
equivalent config field) in the run status.

For State tx runs, convert `pert_onehot_map.pt` to a drug cache:

```bash
python model/tools/export_state_perturbation_cache.py \
  --pert-onehot-map /path/to/state_tx_run/pert_onehot_map.pt \
  --out-dir /data/cyx/1030/dataset/latentfm_drug_cache/state_tx \
  --drop-control DMSO
```

State tx perturbation vectors are label-based condition embeddings, not SMILES
encoders; use them for in-distribution drug conditions unless a molecular
encoder is added.

For the chem backbone package layout, see [`../condition_emb/chempert/README.md`](../condition_emb/chempert/README.md).

## Post-hoc family evaluation

Use `eval_condition_families.py` to reuse the training-time metrics while
splitting the canonical test set by perturbation family/type:

```bash
python -m model.latent.eval_condition_families \
  --checkpoint /path/to/best.pt \
  --groups test_all family_gene family_drug structure_single structure_multi \
           type_CRISPRi type_CRISPRa type_CRISPRko type_Cas13 type_drug \
  --out /path/to/condition_family_eval.json
```

This is useful when aggregate `pearson_pert` is weak: it separates genetic
perturbations, drug perturbations, single perturbations, multi perturbations,
and canonical multi seen/unseen split groups without changing metric
definitions.

## Prior-correction diagnostic

Use `evaluate_prior_correction.py` to test whether train-single genetic
responses contain a no-leakage KNN/additive prior that can rescue held-out
multi-perturbation `pp` after an existing LatentFM checkpoint has produced its
ODE prediction. This is an evaluator, not training; it reports direct/pc/pp on
condition means and intentionally does not report MMD.

```bash
SCFM_WORKSPACE_ROOT=/data/cyx/1030/scLatent \
CUDA_VISIBLE_DEVICES=0 \
python -m model.latent.evaluate_prior_correction \
  --checkpoint /data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k/best.pt \
  --data-dir /data/cyx/1030/dataset/latentfm_full/scfoundation \
  --datasets NormanWeissman2019_filtered Wessels \
  --groups test_multi_seen test_multi_unseen1 test_multi_unseen2 \
  --alphas 0 0.25 0.5 0.75 1 \
  --k-values 5 10
```

The prior is built only from canonical train-single conditions plus the fixed
gene embedding cache, so it is suitable as a leakage-aware diagnostic baseline
before adding any condition-prior teacher or distillation objective.

## Condition-prior teacher training

LatentFM can also train with a split-auditable condition-prior teacher:

```bash
python -m model.latent.train \
  --use-pert-condition \
  --condition-prior-delta-loss-weight 0.05 \
  --condition-prior-bank-max-cells 512 \
  --condition-prior-delta-loss-every 1 \
  --condition-prior-num-genes 2
```

When enabled, training builds a per-dataset prior bank from canonical
train-single gene conditions only. Each prior record is
`mean(gt) - mean(src)` and is never constructed from held-out multi-condition
GT. During training the model samples deterministic synthetic multi-gene
conditions, computes the t=0 velocity under that perturbation batch, and
matches the mean velocity to the summed train-single prior. Logs expose this
as `avg_prior_delta` with schedule field `lambda_prior_delta`.

To train the additive atom head from the same split-auditable prior, enable the
default-off additive-head variant:

```bash
python -m model.latent.train \
  --use-pert-condition \
  --condition-prior-additive-delta-loss-weight 0.05 \
  --condition-prior-bank-max-cells 512 \
  --condition-prior-delta-loss-every 1 \
  --condition-prior-num-genes 2
```

This supervises `predict_additive_condition_delta` on synthetic train-single
gene combinations and logs `avg_prior_add_delta`. It is intended to make the
additive head identifiable before interpreting `combo - additive` as an
interaction diagnostic.

This branch is intended as the trainable counterpart to
`evaluate_prior_correction.py`: use it after the no-leakage prior baseline
shows that train-single composition contains useful multi-perturbation signal.

## Training-composition sampling controls

The default epoch sampler remains condition-balanced by dataset:
`ds_alpha < 1` selects `ceil(n_conditions ** ds_alpha)` conditions per dataset,
then visits each selected condition `ceil(n_gt / batch_size)` times.  This is
useful for full-data training, but composition-relevant datasets such as
Norman/Wessels can still be under-exposed when they have few train-single
conditions.

Three default-off knobs support leakage-preserving sampling diagnostics:

```bash
python -m model.latent.train \
  --min-selected-conditions-per-dataset 32 \
  --condition-visit-power 0.5 \
  --condition-visit-cap 8
```

- `min_selected_conditions_per_dataset`: floor for selected train conditions
  per dataset after `ds_alpha`; capped by the dataset's valid train conditions.
- `condition_visit_power`: applies a power to `ceil(n_gt / batch_size)` before
  integer rounding, reducing cell-count dominance when set below 1.
- `condition_visit_cap`: caps per-condition visits per epoch.

Defaults (`0`, `1.0`, `0`) reproduce the legacy sampler exactly.  These knobs
do not change the canonical split, so held-out multi-condition zero-shot groups
remain held out.

## Condition-delta decomposition diagnostic

Use `eval_condition_delta_decomposition.py` on checkpoints with an enabled
`condition_delta_head` to inspect the head-only combo/additive/interaction
geometry without running ODE integration:

```bash
python -m model.latent.eval_condition_delta_decomposition \
  --checkpoint /path/to/best.pt \
  --groups test_multi_seen test_multi_unseen1 test_multi_unseen2 \
  --device cpu \
  --out-csv /path/to/condition_delta_decomposition.csv \
  --out-json /path/to/condition_delta_decomposition.json
```

The diagnostic reports direct combo-head prediction, additive atom prediction,
and `combo - additive` residual alignment to endpoint and perturbation-residual
targets. Under the canonical zero-shot split, this residual is a diagnostic
hypothesis, not supervised multi-condition interaction learning.
