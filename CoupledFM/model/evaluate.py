"""
Training-time evaluation for CoupledFM.

For each (dataset, condition) in the test set:
  1. Load IR cells, control cells, GT cells from dataset handles
  2. ODE-integrate IR -> predicted expression (Euler, configurable steps)
  3. Compute direct_pearson, corr_ctrl_mean, corr_pert_mean, MMD

Supports multi-GPU: conditions are distributed across ranks via
cost-balanced assignment (snake ordering by gene count) to prevent
NCCL timeout from workload imbalance.

Val / test evaluation: same task list as ``test_ds`` conditions (see explicit
pert split in ``data/pert_split.py``). Per-dataset metrics split by single-
vs multi-perturbation (``+`` in condition name).
"""

import math
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from model.inference import integrate
from model.metrics import pearson_delta, mmd2_multi_sigma


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _eta_str(seconds: float) -> str:
    if seconds <= 0 or not math.isfinite(seconds):
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def _fmt_metric4(x) -> str:
    """Format scalar metrics; NaN / non-finite as ``nan`` (not as 0.0)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(v) or math.isnan(v):
        return "nan"
    return f"{v:.4f}"


def _stable_hash32(text: str, salt: str = "") -> int:
    payload = f"{salt}:{text}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _collect_eval_tasks(test_ds) -> List[Tuple[str, str]]:
    """Flatten (dataset, condition) pairs from test_ds."""
    tasks = []
    for ds_name in test_ds.ds_names:
        for cond in test_ds.ds_conds[ds_name]:
            tasks.append((ds_name, cond))
    return tasks


def build_monitor_val_tasks(
    test_ds,
    fraction: float,
    seed: int,
    max_per_ds: int = 0,
    min_per_ds: int = 0,
    per_ds_target_range: bool = False,
) -> List[Tuple[str, str]]:
    """Deterministic subset of test conditions for monitoring or capped full test.

    If ``per_ds_target_range`` is False (full test schedule, or legacy):
      * ``k = min(ceil(n * fraction), max_per_ds or n, n)`` with ``k >= 1``.

    If True (step-level **val** monitoring):
      * Per dataset, ``n =`` number of test conditions.
      * ``k_raw = ceil(n * fraction)``.
      * If ``n < min_per_ds``: take all ``n`` conditions (dataset too small).
      * Else: ``k = clamp(k_raw, min_per_ds, min(max_per_ds, n))`` so each dataset
        contributes about ``ratio * n`` but stays in ``[min_per_ds, max_per_ds]``
        when possible.

    Returned list is shuffled for balanced distribution across ranks.
    """
    rng = np.random.RandomState(seed)

    per_ds: Dict[str, List[str]] = {}
    for ds_name in test_ds.ds_names:
        conds = list(test_ds.ds_conds[ds_name])
        per_ds[ds_name] = conds

    sampled: List[Tuple[str, str]] = []
    for ds_name, conds in per_ds.items():
        n = len(conds)
        if n == 0:
            continue
        if per_ds_target_range and min_per_ds > 0 and max_per_ds > 0:
            k_raw = max(1, int(math.ceil(n * fraction)))
            if n < min_per_ds:
                k = n
            else:
                k = max(min_per_ds, min(k_raw, max_per_ds, n))
        else:
            k = max(1, int(math.ceil(n * fraction)))
            if max_per_ds > 0:
                k = min(k, max_per_ds)
            k = min(k, n)
        chosen = rng.choice(n, size=k, replace=False)
        for i in chosen:
            sampled.append((ds_name, conds[int(i)]))

    rng.shuffle(sampled)
    return sampled


def _balanced_distribute(
    tasks: List[Tuple[str, str]],
    test_ds,
    world_size: int,
) -> List[List[Tuple[str, str]]]:
    """Distribute tasks across ranks balancing estimated compute cost.

    Cost proxy: n_genes² (dominates attention time).
    Snake-order assignment after descending-cost sort ensures each rank
    gets a roughly equal share of expensive conditions (e.g. gwps).
    """
    costs = []
    for ds_name, cond in tasks:
        h = test_ds.handles[ds_name]
        ng = len(h.gene_ids_valid)
        costs.append(ng * ng)

    order = sorted(range(len(tasks)), key=lambda i: -costs[i])

    assignments: List[List[Tuple[str, str]]] = [[] for _ in range(world_size)]
    for i, idx in enumerate(order):
        cycle = i // world_size
        pos = i % world_size
        r = pos if cycle % 2 == 0 else world_size - 1 - pos
        assignments[r].append(tasks[idx])

    return assignments


def _safe_eval_mb(
    n_genes: int, base_mb: int, attn_backend: str = "sdpa",
) -> int:
    """Reduce micro-batch for long sequences to avoid attention OOM (eval).

    ``linear`` / ``flash-attn`` / ``sparse`` allow micro-batches up to ~20 on
    gwps-length graphs (fits ~8GB with AMP); ``sdpa`` stays conservative.
    """
    N = n_genes + 1
    b = (attn_backend or "sdpa").lower()
    mem_ok = b in ("linear", "flash", "sparse")
    _cap = 20
    if N > 6000:
        if mem_ok:
            return min(base_mb, _cap)
        return min(base_mb, 2)
    return base_mb


def _safe_train_mb(
    n_genes: int, base_mb: int, attn_backend: str = "sdpa",
) -> int:
    """Reduce micro-batch for long sequences during training (with backward).

    ``linear`` / ``flash`` / ``sparse``: up to 20 cells/chunk on N≳5k (gwps);
    ``sdpa`` strict.
    """
    N = n_genes + 1
    b = (attn_backend or "sdpa").lower()
    mem_ok = b in ("linear", "flash", "sparse")
    _cap = 20
    if N > 6000:
        if mem_ok:
            return min(base_mb, _cap)
        return 1
    if N > 5000:
        if mem_ok:
            return min(base_mb, _cap)
        return min(base_mb, 2)
    return base_mb


def _pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D arrays."""
    a_m = a - a.mean()
    b_m = b - b.mean()
    numer = float((a_m * b_m).sum())
    denom = math.sqrt(float((a_m ** 2).sum() * (b_m ** 2).sum()) + 1e-12)
    return numer / denom if math.isfinite(numer) and denom > 0 else 0.0


@torch.no_grad()
def _eval_one_condition(
    model, h, cond, device,
    n_ode_steps, coupling_mode, latent_fm,
    use_amp, amp_dtype, eval_mb,
    max_cells=0,
    ctrl_mean_ds: Optional[np.ndarray] = None,
    pert_mean_ds: Optional[np.ndarray] = None,
    cfg_w: float = 1.0,
    use_residual_flow: bool = False,
    ode_method: str = "euler",
    test_ds=None,
    max_pert_genes: int = 16,
) -> Dict[str, float]:
    """Evaluate one condition by sampling control-center starts and averaging predictions.

    Evaluation intentionally does not redo OT matching. For each target
    condition, control-center rows provide x0/x_ctrl starts; the resulting
    predictions are averaged and compared with the condition GT pseudobulk.
    """
    in_vocab = h.in_vocab
    n_genes = int(in_vocab.sum())
    _m = model.module if hasattr(model, "module") else model
    ab = getattr(_m, "attn_backend", "sdpa")
    eval_mb = _safe_eval_mb(n_genes, eval_mb, attn_backend=ab)
    gene_ids_d = h.gene_ids_valid.to(device)

    src_idx = h.pert_cond2idx[cond]
    gt_idx = h.gt_cond2idx[cond]

    if max_cells > 0 and len(src_idx) > max_cells:
        rng = np.random.RandomState(_stable_hash32(cond, "src"))
        src_idx = rng.choice(src_idx, max_cells, replace=False)

    # GT 若不截断，gwps 等条件可达数万细胞；MMD 在 _median_sigmas 里做 cdist(Y,Y)，O(n²) 显存会爆。
    if max_cells > 0 and len(gt_idx) > max_cells:
        rng_gt = np.random.RandomState(_stable_hash32(cond, "gt"))
        gt_idx = rng_gt.choice(gt_idx, max_cells, replace=False)

    x_src_np = h.get_pert_rows(src_idx, in_vocab)
    x_gt_np = h.get_gt_rows(gt_idx, in_vocab)

    ctrl_idx = np.clip(h.pert_ctrl_map[src_idx], 0, h.n_ctrl - 1)
    x_ctrl_np = h.X_ctrl[ctrl_idx][:, in_vocab]
    budget_mask_np = None
    budget_keep_np = None
    if test_ds is not None and hasattr(test_ds, "budget_mask_for_eval"):
        budget_mask_np = test_ds.budget_mask_for_eval(h.ds_name)
    if budget_mask_np is not None:
        budget_mask_np = np.asarray(budget_mask_np, dtype=np.float32)
        if budget_mask_np.shape != (x_src_np.shape[1],):
            raise ValueError(
                f"[{h.ds_name} cond={cond!r}] budget mask shape {budget_mask_np.shape} "
                f"!= gene-space width {(x_src_np.shape[1],)}"
            )
        budget_keep_np = budget_mask_np < 0.5
        if not bool(budget_keep_np.any()):
            raise ValueError(f"[{h.ds_name} cond={cond!r}] budget mask keeps zero genes")
        x_src_np = x_src_np * budget_keep_np[None, :]
        x_ctrl_np = x_ctrl_np * budget_keep_np[None, :]

    x_src = torch.from_numpy(x_src_np).to(device)
    x_ctrl = torch.from_numpy(x_ctrl_np).to(device)
    gene_mask_t = None
    if budget_mask_np is not None:
        gm_np = np.broadcast_to(budget_mask_np[None, :], x_src_np.shape).astype(
            np.float32,
            copy=True,
        )
        gene_mask_t = torch.from_numpy(gm_np).to(device)

    dx_prior_t = None
    if use_residual_flow:
        dx_prior_t = torch.from_numpy(
            h.compute_dx_prior_gene(cond),
        ).to(device)
    eidx = h.edge_index.to(device) if h.edge_index is not None else None

    z_src_batch = None
    if coupling_mode == "coupled" and latent_fm is not None and h.has_latent:
        z_src_batch = torch.from_numpy(
            h.get_z_src_rows(src_idx)
        ).to(device)

    preds = []
    for ci in range(0, len(x_src), eval_mb):
        s, e = ci, min(ci + eval_mb, len(x_src))
        z_chunk = z_src_batch[s:e] if z_src_batch is not None else None

        pb_chunk = None
        if test_ds is not None:
            pb_chunk = test_ds.perturbation_batch_tensors(
                h.ds_name, cond, e - s, device=device,
            )

        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                pred_chunk = integrate(
                    model, x_src[s:e], x_ctrl[s:e],
                    gene_ids_d,
                    n_steps=n_ode_steps, method=ode_method,
                    latent_fm=latent_fm, z_src=z_chunk,
                    cfg_w=cfg_w,
                    edge_index=eidx,
                    dx_prior=dx_prior_t,
                    gene_mask=gene_mask_t[s:e] if gene_mask_t is not None else None,
                    perturbation_batch=pb_chunk,
                    max_pert_genes=max_pert_genes,
                )
        else:
            pred_chunk = integrate(
                model, x_src[s:e], x_ctrl[s:e],
                gene_ids_d,
                n_steps=n_ode_steps, method=ode_method,
                latent_fm=latent_fm, z_src=z_chunk,
                cfg_w=cfg_w,
                edge_index=eidx,
                dx_prior=dx_prior_t,
                gene_mask=gene_mask_t[s:e] if gene_mask_t is not None else None,
                perturbation_batch=pb_chunk,
                max_pert_genes=max_pert_genes,
            )
        preds.append(pred_chunk.float().cpu())

    x_pred = torch.cat(preds, dim=0).numpy()

    if budget_keep_np is not None:
        x_pred_eval = x_pred[:, budget_keep_np]
        x_gt_eval = x_gt_np[:, budget_keep_np]
        x_ctrl_eval = x_ctrl_np[:, budget_keep_np]
        pert_mean_eval = pert_mean_ds[budget_keep_np] if pert_mean_ds is not None else None
        ctrl_mean_eval = ctrl_mean_ds[budget_keep_np] if ctrl_mean_ds is not None else None
    else:
        x_pred_eval = x_pred
        x_gt_eval = x_gt_np
        x_ctrl_eval = x_ctrl_np
        pert_mean_eval = pert_mean_ds
        ctrl_mean_eval = ctrl_mean_ds

    pred_mean = x_pred_eval.mean(axis=0)
    gt_mean = x_gt_eval.mean(axis=0)
    ctrl_mean = x_ctrl_eval.mean(axis=0)

    n_g = int(pred_mean.shape[0])
    if pert_mean_eval is not None and pert_mean_eval.shape != (n_g,):
        raise ValueError(
            f"[{h.ds_name} cond={cond!r}] pert_mean_ds shape {pert_mean_eval.shape} "
            f"!= gene-space pred_mean ({n_g},). "
            f"``corr_pert_mean`` requires per-dataset means in **vocab-filtered gene** "
            f"space (same as ``train.py`` via ``compute_gt_mean_gene``). "
            f"Latent ``gt/emb`` vectors (e.g. length 2058) must not be passed."
        )
    if ctrl_mean_eval is not None and ctrl_mean_eval.shape != (n_g,):
        raise ValueError(
            f"[{h.ds_name} cond={cond!r}] ctrl_mean_ds shape {ctrl_mean_eval.shape} "
            f"!= gene-space pred_mean ({n_g},). "
            f"``corr_ctrl_mean`` requires **gene-space** ctrl means "
            f"(``ctrl_mean_gene``), not latent ctrl embeddings."
        )

    direct_pearson = _pearson_np(pred_mean, gt_mean)
    pd_ctrl = pearson_delta(pred_mean, gt_mean, ctrl_mean)
    # 无 per-dataset / gene 空间 pert 均值时，不误用 0.0 作为真实 corr（见 TrainConfig 与 split）
    if pert_mean_eval is not None:
        corr_pert_mean = _pearson_np(pred_mean - pert_mean_eval, gt_mean - pert_mean_eval)
    else:
        corr_pert_mean = float("nan")
    if ctrl_mean_eval is not None:
        corr_ctrl_mean = _pearson_np(pred_mean - ctrl_mean_eval, gt_mean - ctrl_mean_eval)
    else:
        corr_ctrl_mean = float("nan")

    x_pred_t = torch.from_numpy(x_pred_eval).to(device)
    x_gt_t = torch.from_numpy(x_gt_eval).to(device)
    mmd_val = mmd2_multi_sigma(x_pred_t, x_gt_t)
    del x_pred_t, x_gt_t, x_src, x_ctrl, z_src_batch, gene_mask_t
    torch.cuda.empty_cache()

    return {
        "direct_pearson": direct_pearson,
        "pearson_delta_ctrl": pd_ctrl,
        "corr_pert_mean": corr_pert_mean,
        "corr_ctrl_mean": corr_ctrl_mean,
        "mmd": mmd_val,
        "n_src": len(src_idx),
        "n_gt": len(gt_idx),
        "n_eval_genes": n_g,
    }


def _pert_row_mean(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {
            "direct_pearson": 0.0,
            "pearson_delta_ctrl": 0.0,
            "corr_pert_mean": 0.0,
            "corr_ctrl_mean": 0.0,
            "mmd": 0.0,
            "n_conds": 0,
        }
    return {
        "direct_pearson": float(np.mean([c["direct_pearson"] for c in rows])),
        "pearson_delta_ctrl": float(np.nanmean([c["pearson_delta_ctrl"] for c in rows])),
        "corr_pert_mean": float(np.nanmean([c["corr_pert_mean"] for c in rows])),
        "corr_ctrl_mean": float(np.nanmean([c["corr_ctrl_mean"] for c in rows])),
        "mmd": float(np.nanmean([c["mmd"] for c in rows])),
        "n_conds": len(rows),
    }


def _aggregate(all_results: List[Dict]) -> Dict:
    """Aggregate per-condition results into per-dataset and global metrics."""
    for r in all_results:
        c = r.get("cond", "")
        r["pert_type"] = "multi" if "+" in str(c) else "single"

    ds_groups: Dict[str, List[Dict]] = {}
    for r in all_results:
        ds_groups.setdefault(r["ds_name"], []).append(r)

    result = {"per_dataset": {}, "global": {}}
    all_direct, all_pd_ctrl, all_corr_pert, all_corr_ctrl, all_mmd = [], [], [], [], []

    for ds_name, conds in sorted(ds_groups.items()):
        ds_direct = [c["direct_pearson"] for c in conds]
        ds_pd_ctrl = [c["pearson_delta_ctrl"] for c in conds]
        ds_corr_pert = [c["corr_pert_mean"] for c in conds]
        ds_corr_ctrl = [c["corr_ctrl_mean"] for c in conds]
        ds_mmd = [c["mmd"] for c in conds]

        singles = [c for c in conds if c["pert_type"] == "single"]
        multis = [c for c in conds if c["pert_type"] == "multi"]

        entry = {
            "direct_pearson": float(np.mean(ds_direct)),
            "pearson_delta_ctrl": float(np.nanmean(ds_pd_ctrl)),
            "corr_pert_mean": float(np.nanmean(ds_corr_pert)),
            "corr_ctrl_mean": float(np.nanmean(ds_corr_ctrl)),
            "mmd": float(np.nanmean(ds_mmd)),
            "n_conds": len(conds),
        }
        if singles:
            entry["single"] = _pert_row_mean(singles)
        if multis:
            entry["multi"] = _pert_row_mean(multis)
        result["per_dataset"][ds_name] = entry

        all_direct.extend(ds_direct)
        all_pd_ctrl.extend(ds_pd_ctrl)
        all_corr_pert.extend(ds_corr_pert)
        all_corr_ctrl.extend(ds_corr_ctrl)
        all_mmd.extend(ds_mmd)

    result["global"] = {
        "direct_pearson": float(np.mean(all_direct)) if all_direct else 0.0,
        "pearson_delta_ctrl": float(np.nanmean(all_pd_ctrl)) if all_pd_ctrl else float("nan"),
        "corr_pert_mean": float(np.nanmean(all_corr_pert)) if all_corr_pert else float("nan"),
        "corr_ctrl_mean": float(np.nanmean(all_corr_ctrl)) if all_corr_ctrl else float("nan"),
        "mmd": float(np.nanmean(all_mmd)) if all_mmd else float("nan"),
        "n_conds": len(all_pd_ctrl),
    }
    g_all = list(all_results)
    g_s = [r for r in g_all if r["pert_type"] == "single"]
    g_m = [r for r in g_all if r["pert_type"] == "multi"]
    if g_s:
        result["global"]["single"] = _pert_row_mean(g_s)
    if g_m:
        result["global"]["multi"] = _pert_row_mean(g_m)
    return result


@torch.no_grad()
def evaluate(
    model,
    test_ds,
    device: torch.device,
    n_ode_steps: int = 20,
    coupling_mode: str = "coupled",
    latent_fm=None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    rank: int = 0,
    world_size: int = 1,
    eval_mb: int = 4,
    tasks: Optional[List[Tuple[str, str]]] = None,
    max_cells: int = 0,
    ctrl_means: Optional[Dict[str, np.ndarray]] = None,
    pert_means: Optional[Dict[str, np.ndarray]] = None,
    cfg_w: float = 1.0,
    use_residual_flow: bool = False,
    ode_method: str = "euler",
    max_pert_genes: int = 16,
) -> Optional[Dict]:
    """Run evaluation on test conditions with ODE integration.

    Args:
        max_cells: subsample control-center starts and GT cells per condition
            (0 = unlimited). Use small values for fast val monitoring.

    When ``tasks`` is None, evaluates all test conditions from ``test_ds``.
    Tasks are distributed via cost-balanced snake ordering (by gene count²)
    to prevent heavy conditions (e.g. gwps) from clustering on one rank.
    Only rank 0 returns the aggregated result; other ranks return None.
    """
    model.eval()

    if tasks is None:
        tasks = _collect_eval_tasks(test_ds)

    if world_size > 1:
        assignments = _balanced_distribute(tasks, test_ds, world_size)
        my_tasks = assignments[rank]
    else:
        my_tasks = list(tasks)

    n_my = len(my_tasks)
    n_total = len(tasks)

    local_results = []
    t_eval_start = time.time()
    t_last_log = t_eval_start

    for idx, (ds_name, cond) in enumerate(my_tasks):
        h = test_ds.handles[ds_name]
        t_cond_start = time.time()

        metrics = _eval_one_condition(
            model, h, cond, device,
            n_ode_steps, coupling_mode, latent_fm,
            use_amp, amp_dtype, eval_mb,
            max_cells=max_cells,
            ctrl_mean_ds=ctrl_means.get(ds_name) if ctrl_means else None,
            pert_mean_ds=pert_means.get(ds_name) if pert_means else None,
            cfg_w=cfg_w,
            use_residual_flow=use_residual_flow,
            ode_method=ode_method,
            test_ds=test_ds,
            max_pert_genes=max_pert_genes,
        )
        metrics["ds_name"] = ds_name
        metrics["cond"] = cond
        local_results.append(metrics)

        t_cond = time.time() - t_cond_start
        now = time.time()
        should_log = (
            rank == 0 and (
                idx + 1 == n_my
                or n_my <= 50
                or (idx + 1) % max(1, n_my // 20) == 0
                or now - t_last_log >= 30
            )
        )
        if should_log:
            elapsed = now - t_eval_start
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_my - idx - 1) / speed if speed > 0 else 0
            n_genes = len(h.gene_ids_valid)
            pct = (idx + 1) / n_my * 100
            print(
                f"    [{_now()}] eval: {idx+1}/{n_my} ({pct:.0f}%) "
                f"| {elapsed:.0f}s, {speed:.2f} cond/s, "
                f"ETA {_eta_str(eta)} "
                f"| last: {ds_name} G={n_genes} {t_cond:.1f}s "
                f"cells={metrics['n_src']}",
                flush=True,
            )
            t_last_log = now

    if world_size > 1:
        # Sequential broadcast avoids rank-0 allocating an oversized list of
        # per-rank result lists (PyTorch ``gather``-style allocation pattern).
        all_rows: List[Dict] = []
        for src in range(world_size):
            obj_list: List = [None]
            if rank == src:
                obj_list[0] = local_results
            dist.broadcast_object_list(obj_list, src=src)
            if rank == 0:
                recv_chunk = obj_list[0]
                assert recv_chunk is not None
                all_rows.extend(recv_chunk)
        if rank == 0:
            return _aggregate(all_rows)
        return None
    else:
        return _aggregate(local_results)


def print_eval_results(
    results: Dict,
    step: int,
    tag: str = "eval",
    per_dataset: bool = True,
):
    """Pretty-print evaluation results."""
    ts = _now()
    g = results["global"]
    print(
        f"  [{ts}] [{tag} step={step}] "
        f"dp={_fmt_metric4(g['direct_pearson'])}  "
        f"pc={_fmt_metric4(g['corr_ctrl_mean'])}  "
        f"pp={_fmt_metric4(g['corr_pert_mean'])}  "
        f"MMD={g['mmd']:.6f}  "
        f"({g['n_conds']} conds)",
        flush=True,
    )

    if not per_dataset:
        return
    for ds_name, ds in sorted(results["per_dataset"].items()):
        print(
            f"    {ds_name}: "
            f"dp={_fmt_metric4(ds['direct_pearson'])}  "
            f"pc={_fmt_metric4(ds['corr_ctrl_mean'])}  "
            f"pp={_fmt_metric4(ds['corr_pert_mean'])}  "
            f"MMD={ds['mmd']:.6f}  "
            f"({ds['n_conds']} conds)",
            flush=True,
        )
        if "single" in ds:
            s = ds["single"]
            print(
                f"    {ds_name} [single]: "
                f"dp={_fmt_metric4(s['direct_pearson'])}  "
                f"pc={_fmt_metric4(s['corr_ctrl_mean'])}  "
                f"pp={_fmt_metric4(s['corr_pert_mean'])}  "
                f"MMD={s['mmd']:.6f}  "
                f"({s['n_conds']} conds)",
                flush=True,
            )
        if "multi" in ds:
            m = ds["multi"]
            print(
                f"    {ds_name} [multi]: "
                f"dp={_fmt_metric4(m['direct_pearson'])}  "
                f"pc={_fmt_metric4(m['corr_ctrl_mean'])}  "
                f"pp={_fmt_metric4(m['corr_pert_mean'])}  "
                f"MMD={m['mmd']:.6f}  "
                f"({m['n_conds']} conds)",
                flush=True,
            )
    g = results["global"]
    if "single" in g:
        s = g["single"]
        print(
            f"    GLOBAL [single]: "
            f"dp={_fmt_metric4(s['direct_pearson'])}  "
            f"pc={_fmt_metric4(s['corr_ctrl_mean'])}  "
            f"pp={_fmt_metric4(s['corr_pert_mean'])}  "
            f"MMD={s['mmd']:.6f}  "
            f"({s['n_conds']} conds)",
            flush=True,
        )
    if "multi" in g:
        m = g["multi"]
        print(
            f"    GLOBAL [multi]: "
            f"dp={_fmt_metric4(m['direct_pearson'])}  "
            f"pc={_fmt_metric4(m['corr_ctrl_mean'])}  "
            f"pp={_fmt_metric4(m['corr_pert_mean'])}  "
            f"MMD={m['mmd']:.6f}  "
            f"({m['n_conds']} conds)",
            flush=True,
        )
