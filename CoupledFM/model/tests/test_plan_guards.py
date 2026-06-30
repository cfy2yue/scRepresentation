"""Regression checks for the raw_independent executable plan (CPU-only where possible)."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model.config import DataConfig  # noqa: E402
from model.config import Config  # noqa: E402
from model.inference import _apply_saved_config, _sample_control_indices  # noqa: E402
from model.metrics import mmd2_multi_sigma  # noqa: E402
from model.train import _select_metric_value, _validate_model_train_config  # noqa: E402
from model.utils.data.split import (  # noqa: E402
    classify_multi_perturbation_tests,
    split_condition_lists,
)


class TestPlanGuards(unittest.TestCase):
    def test_data_config_exposes_chem_pert_fields(self):
        for k in ("pert_chem_enabled", "drug_emb_cache_dir", "max_chem_keys"):
            self.assertIn(k, DataConfig.__dataclass_fields__)

    def test_h5ad_pert_metadata_is_safe_default(self):
        self.assertTrue(DataConfig().use_h5ad_pert_metadata)
        self.assertTrue(Config().inference.use_h5ad_pert_metadata)

    def test_vendor_biflow_paths_module(self):
        p = ROOT / "utils" / "data" / "biflow_paths.py"
        self.assertTrue(p.is_file(), f"missing vendored biflow_paths at {p}")

    def test_evaluate_ddp_uses_broadcast_not_gather(self):
        text = (ROOT / "evaluate.py").read_text(encoding="utf-8")
        self.assertIn("broadcast_object_list", text)
        self.assertNotIn("dist.gather_object", text)

    def test_pert_batch_utils_path_is_repo_local(self):
        text = (ROOT / "pert_batch_utils.py").read_text(encoding="utf-8")
        self.assertIn("_REPO_ROOT", text)
        self.assertNotIn("_COUPLED_FM_ROOT", text)
        self.assertNotIn("_LATENT_FM_ROOT", text)

    def test_smoke_test_paths_raw_only(self):
        text = (ROOT / "tools" / "smoke_test.py").read_text(encoding="utf-8")
        self.assertNotIn("_CF_ROOT", text)
        self.assertNotIn('_LATENT =', text)

    def test_no_coupled_import_strings_in_model_py(self):
        bad: list[str] = []
        for path in (ROOT / "model").rglob("*.py"):
            if "tests" in path.parts or "__pycache__" in path.parts:
                continue
            t = path.read_text(encoding="utf-8")
            if "from coupled" in t or "import coupled" in t:
                bad.append(str(path.relative_to(ROOT)))
        self.assertFalse(bad, msg="found coupled imports in: " + ", ".join(bad))

    def test_chempert_placeholder_package(self):
        init_p = ROOT / "condition_emb" / "chempert" / "__init__.py"
        readme = ROOT / "condition_emb" / "chempert" / "README.md"
        self.assertTrue(init_p.is_file())
        self.assertTrue(readme.is_file())

    def test_build_pert_batch_from_cond_uses_tuple_full(self):
        text = (ROOT / "pert_batch_utils.py").read_text(encoding="utf-8")
        self.assertIn("return pb.as_tuple_full()", text)

    def test_select_metric_nan_corr_pert_mean(self):
        results = {
            "global": {
                "corr_pert_mean": float("nan"),
                "mmd": 0.1,
                "pearson_delta_ctrl": 0.5,
            }
        }
        self.assertEqual(_select_metric_value(results, "corr_pert_mean"), float("-inf"))
        v = _select_metric_value(results, "corr_minus_mmd", mmd_lambda=0.5)
        self.assertTrue(math.isinf(v) and v < 0)

    def test_validate_sparse_sdpa_bias_errors(self):
        tc = SimpleNamespace(val_ode_method="euler")
        mc = SimpleNamespace(attn_backend="sparse", graph_bias_mode="sdpa_bias")
        with self.assertRaises(ValueError) as ctx:
            _validate_model_train_config(tc, mc, {})
        self.assertIn("sparse", str(ctx.exception).lower())

    def test_validate_sparse_no_edge_errors(self):
        tc = SimpleNamespace(val_ode_method="euler")
        mc = SimpleNamespace(attn_backend="sparse", graph_bias_mode="none")
        h = MagicMock()
        h.edge_index = None
        with self.assertRaises(ValueError) as ctx:
            _validate_model_train_config(tc, mc, {"ds1": h})
        self.assertIn("edge", str(ctx.exception).lower())

    def test_ot_assignment_resample_length(self):
        import torch

        from model.utils.data.ot_pairer import _resample_assignment_pairs  # noqa: E402

        i = torch.arange(10)
        j = torch.arange(10) % 8
        oi, oj = _resample_assignment_pairs(i, j, 16)
        self.assertEqual(oi.numel(), 16)
        self.assertEqual(oj.numel(), 16)

        i2 = torch.arange(80)
        j2 = torch.arange(80) % 64
        oi2, oj2 = _resample_assignment_pairs(i2, j2, 32)
        self.assertEqual(oi2.numel(), 32)
        self.assertEqual(oj2.numel(), 32)

    def test_torch_sinkhorn_pair_smoke(self):
        import torch

        from model.utils.data.ot_pairer import sinkhorn_pair

        x0 = torch.randn(8, 5)
        x1 = torch.randn(8, 5)
        i, j = sinkhorn_pair(x0, x1, n_samples=8, n_iter=3)
        self.assertEqual(i.numel(), 8)
        self.assertEqual(j.numel(), 8)

    def test_hungarian_pair_preserves_marginals(self):
        import torch

        from model.utils.data.ot_pairer import hungarian_pair

        x0 = torch.randn(8, 5)
        x1 = x0 + 0.01 * torch.randn(8, 5)
        i, j = hungarian_pair(x0, x1, n_samples=8)
        self.assertEqual(i.numel(), 8)
        self.assertEqual(j.numel(), 8)
        self.assertEqual(torch.unique(i).numel(), 8)
        self.assertEqual(torch.unique(j).numel(), 8)

    def test_gpu_sinkhorn_avoids_per_iteration_item_sync(self):
        text = (ROOT / "utils" / "data" / "ot_pairer.py").read_text(encoding="utf-8")
        self.assertIn('cost.device.type != "cuda"', text)
        self.assertIn("diff.item() < tol", text)

    def test_vendor_unified_encoder_and_six_pert_types(self):
        from model.condition_emb.genepert import perturbation as VP
        from model.condition_emb.genepert.perturbation_encoder import UnifiedConditionEncoder  # noqa: F401

        self.assertEqual(VP.num_perturbation_types(), 6)
        self.assertIsNotNone(UnifiedConditionEncoder)

    def test_vendor_drug_cache_module(self):
        p = ROOT / "condition_emb" / "chempert" / "drug_cache.py"
        self.assertTrue(p.is_file())

    def test_inference_restores_saved_config_fields(self):
        cfg = Config()
        saved = {
            "model": {
                "use_pert_condition": True,
                "pert_embed_mode": "pretrained_frozen",
                "pert_pool_aggregations": ["mean", "max", "min"],
            },
            "data": {
                "biflow_dir": "/tmp/biflow",
                "latent_backbone": "stack",
                "pert_gene_emb_cache_dir": "/tmp/cache",
                "use_h5ad_pert_metadata": True,
                "use_raw_cond": True,
            },
            "train": {"coupling_mode": "ot"},
            "inference": {"device": "cpu"},
        }
        _apply_saved_config(cfg, saved)
        self.assertTrue(cfg.model.use_pert_condition)
        self.assertEqual(cfg.model.pert_embed_mode, "pretrained_frozen")
        self.assertEqual(cfg.model.pert_pool_aggregations, ["mean", "max", "min"])
        self.assertEqual(cfg.data.biflow_dir, "/tmp/biflow")
        self.assertEqual(cfg.data.latent_backbone, "stack")
        self.assertEqual(cfg.data.pert_gene_emb_cache_dir, "/tmp/cache")
        self.assertTrue(cfg.data.use_h5ad_pert_metadata)
        self.assertTrue(cfg.data.use_raw_cond)
        self.assertEqual(cfg.train.coupling_mode, "ot")
        self.assertEqual(cfg.inference.device, "cpu")

    def test_inference_control_center_sampling_is_deterministic(self):
        a = _sample_control_indices(10, 4, "TP53")
        b = _sample_control_indices(10, 4, "TP53")
        c = _sample_control_indices(10, 4, "KRAS")
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(len(a), 4)
        self.assertNotEqual(a.tolist(), c.tolist())
        self.assertEqual(_sample_control_indices(3, 0, "TP53").tolist(), [0, 1, 2])

    def test_split_keeps_all_multi_gene_conditions_in_test(self):
        import numpy as np

        single = [f"S{i}" for i in range(10)]
        multi = [f"A{i}+B{i}" for i in range(40)]
        train, test, single_test, multi_test = split_condition_lists(
            single, multi, np.random.RandomState(7),
        )
        self.assertEqual(len(multi_test), len(multi))
        self.assertTrue(set(multi).issubset(set(test)))
        self.assertFalse(set(multi) & set(train))
        self.assertEqual(len(single_test), 4)

    def test_multi_gene_test_visibility_groups_are_component_based(self):
        groups = classify_multi_perturbation_tests(
            multi_test=["A+B", "A+X", "X+Y", "A+B+C", "A+B+X"],
            train_single=["A", "B", "C"],
        )
        self.assertEqual(groups["seen"], ["A+B", "A+B+C"])
        self.assertEqual(groups["unseen1"], ["A+X", "A+B+X"])
        self.assertEqual(groups["unseen2"], ["X+Y"])

    def test_latent_composition_audit_reports_zero_shot_multi(self):
        from model.latent.audit_composition_split import audit_composition_split

        manifest = {
            "datasets": {
                "ds1": {"conditions": ["A", "B", "C", "A+B", "A+X", "X+Y"]},
            }
        }
        split = {
            "ds1": {
                "train": ["A", "B", "C"],
                "test": ["A+B", "A+X", "X+Y"],
            }
        }
        report = audit_composition_split(manifest, split)
        self.assertEqual(report["totals"]["train_multi"], 0)
        self.assertFalse(report["totals"]["has_exact_multi_train_supervision"])
        self.assertEqual(report["totals"]["multi_seen"], 1)
        self.assertEqual(report["totals"]["multi_unseen1"], 1)
        self.assertEqual(report["totals"]["multi_unseen2"], 1)
        self.assertEqual(report["totals"]["test_multi_with_exact_train_leak"], 0)

    def test_latent_eval_split_group_filters_to_manifest_conditions(self):
        from model.latent.eval_split_groups import _group_as_test_split

        manifest = {
            "datasets": {
                "ds1": {"conditions": ["A", "A+B"]},
                "ds2": {"conditions": ["C"]},
            }
        }
        split = {
            "ds1": {"test_multi_seen": ["A+B", "MISSING"]},
            "ds2": {"test_multi_seen": ["C"]},
            "ds3": {"test_multi_seen": ["Z"]},
        }
        out = _group_as_test_split(split=split, manifest=manifest, group="test_multi_seen")
        self.assertEqual(out, {
            "ds1": {"train": [], "test": ["A+B"]},
            "ds2": {"train": [], "test": ["C"]},
        })

    def test_mmd_tiny_sample_is_nan_not_perfect_zero(self):
        import torch

        x = torch.zeros(1, 3)
        y = torch.ones(2, 3)
        self.assertTrue(math.isnan(mmd2_multi_sigma(x, y)))

    def test_latent_direction_loss_defaults_off_and_warms_up(self):
        from model.latent.config import Config as LatentConfig
        from model.latent.train import (
            composition_delta_loss_schedule,
            condition_prior_additive_delta_loss_schedule,
            anchor_replay_loss_schedule,
            direction_loss_schedule,
            endpoint_delta_loss_schedule,
            pert_residual_contrastive_loss_schedule,
            pert_residual_direction_loss_schedule,
            trackc_routed_distill_loss_schedule,
            trackc_routed_endpoint_loss_schedule,
        )

        cfg = LatentConfig()
        self.assertEqual(cfg.direction_loss_weight, 0.0)
        self.assertEqual(direction_loss_schedule(100, cfg), 0.0)
        self.assertEqual(cfg.endpoint_delta_loss_weight, 0.0)
        self.assertEqual(endpoint_delta_loss_schedule(100, cfg), 0.0)
        self.assertEqual(cfg.pert_residual_direction_loss_weight, 0.0)
        self.assertEqual(pert_residual_direction_loss_schedule(100, cfg), 0.0)
        self.assertEqual(cfg.pert_residual_contrastive_loss_weight, 0.0)
        self.assertEqual(pert_residual_contrastive_loss_schedule(100, cfg), 0.0)
        self.assertEqual(cfg.composition_delta_loss_weight, 0.0)
        self.assertEqual(composition_delta_loss_schedule(100, cfg), 0.0)
        self.assertEqual(cfg.condition_prior_additive_delta_loss_weight, 0.0)
        self.assertEqual(condition_prior_additive_delta_loss_schedule(100, cfg), 0.0)
        self.assertEqual(cfg.anchor_replay_loss_weight, 0.0)
        self.assertEqual(anchor_replay_loss_schedule(100, cfg), 0.0)
        self.assertEqual(cfg.trackc_routed_distill_loss_weight, 0.0)
        self.assertEqual(cfg.trackc_routed_distill_bank_split_file, "")
        self.assertEqual(trackc_routed_distill_loss_schedule(100, cfg), 0.0)
        self.assertEqual(cfg.trackc_routed_endpoint_loss_weight, 0.0)
        self.assertEqual(trackc_routed_endpoint_loss_schedule(100, cfg), 0.0)

        cfg.direction_loss_weight = 0.2
        cfg.direction_loss_warmup_start = 10
        cfg.direction_loss_warmup_end = 20
        self.assertEqual(direction_loss_schedule(0, cfg), 0.0)
        self.assertGreater(direction_loss_schedule(15, cfg), 0.0)
        self.assertLess(direction_loss_schedule(15, cfg), 0.2)
        self.assertEqual(direction_loss_schedule(20, cfg), 0.2)

        cfg.endpoint_delta_loss_weight = 0.3
        cfg.endpoint_delta_loss_warmup_start = 5
        cfg.endpoint_delta_loss_warmup_end = 15
        self.assertEqual(endpoint_delta_loss_schedule(0, cfg), 0.0)
        self.assertGreater(endpoint_delta_loss_schedule(10, cfg), 0.0)
        self.assertLess(endpoint_delta_loss_schedule(10, cfg), 0.3)
        self.assertEqual(endpoint_delta_loss_schedule(15, cfg), 0.3)

        cfg.pert_residual_direction_loss_weight = 0.4
        cfg.pert_residual_direction_loss_warmup_start = 30
        cfg.pert_residual_direction_loss_warmup_end = 50
        self.assertEqual(pert_residual_direction_loss_schedule(0, cfg), 0.0)
        self.assertGreater(pert_residual_direction_loss_schedule(40, cfg), 0.0)
        self.assertLess(pert_residual_direction_loss_schedule(40, cfg), 0.4)
        self.assertEqual(pert_residual_direction_loss_schedule(50, cfg), 0.4)

        cfg.pert_residual_contrastive_loss_weight = 0.5
        cfg.pert_residual_contrastive_loss_warmup_start = 60
        cfg.pert_residual_contrastive_loss_warmup_end = 80
        self.assertEqual(pert_residual_contrastive_loss_schedule(0, cfg), 0.0)
        self.assertGreater(pert_residual_contrastive_loss_schedule(70, cfg), 0.0)
        self.assertLess(pert_residual_contrastive_loss_schedule(70, cfg), 0.5)
        self.assertEqual(pert_residual_contrastive_loss_schedule(80, cfg), 0.5)

        cfg.composition_delta_loss_weight = 0.4
        cfg.composition_delta_loss_warmup_start = 0
        cfg.composition_delta_loss_warmup_end = 4
        cfg.composition_delta_loss_every = 2
        self.assertGreater(composition_delta_loss_schedule(2, cfg), 0.0)
        self.assertLess(composition_delta_loss_schedule(2, cfg), 0.4)
        self.assertEqual(composition_delta_loss_schedule(3, cfg), 0.0)
        self.assertEqual(composition_delta_loss_schedule(4, cfg), 0.4)

        cfg.condition_prior_additive_delta_loss_weight = 0.25
        cfg.condition_prior_additive_delta_loss_warmup_start = 0
        cfg.condition_prior_additive_delta_loss_warmup_end = 4
        cfg.condition_prior_delta_loss_every = 2
        self.assertGreater(condition_prior_additive_delta_loss_schedule(2, cfg), 0.0)
        self.assertLess(condition_prior_additive_delta_loss_schedule(2, cfg), 0.25)
        self.assertEqual(condition_prior_additive_delta_loss_schedule(3, cfg), 0.0)
        self.assertEqual(condition_prior_additive_delta_loss_schedule(4, cfg), 0.25)

        cfg.trackc_routed_distill_loss_weight = 0.6
        cfg.trackc_routed_distill_loss_warmup_start = 10
        cfg.trackc_routed_distill_loss_warmup_end = 20
        self.assertEqual(trackc_routed_distill_loss_schedule(0, cfg), 0.0)
        self.assertGreater(trackc_routed_distill_loss_schedule(15, cfg), 0.0)
        self.assertLess(trackc_routed_distill_loss_schedule(15, cfg), 0.6)
        self.assertEqual(trackc_routed_distill_loss_schedule(20, cfg), 0.6)

        cfg.trackc_routed_endpoint_loss_weight = 0.7
        cfg.trackc_routed_endpoint_loss_warmup_start = 10
        cfg.trackc_routed_endpoint_loss_warmup_end = 20
        self.assertEqual(trackc_routed_endpoint_loss_schedule(0, cfg), 0.0)
        self.assertGreater(trackc_routed_endpoint_loss_schedule(15, cfg), 0.0)
        self.assertLess(trackc_routed_endpoint_loss_schedule(15, cfg), 0.7)
        self.assertEqual(trackc_routed_endpoint_loss_schedule(20, cfg), 0.7)

    def test_trackc_routed_distill_route_and_target_helpers(self):
        import json
        import tempfile

        import torch

        from model.latent.config import Config as LatentConfig
        from model.latent.train import (
            _load_trackc_routed_distill_routes,
            _trackc_routed_distill_source_dataset,
            get_trackc_routed_distill_target,
        )
        from model.utils.conditioning.perturbation import ConditionMetadata

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "route.json"
            p.write_text(
                json.dumps(
                    {
                        "route": {
                            "ds1": "additive_single_sum",
                            "ds2": "dataset_multi_mean",
                            "ds3": "train_multi_memory",
                        }
                    }
                ),
                encoding="utf-8",
            )
            cfg = LatentConfig()
            cfg.trackc_routed_distill_loss_weight = 1.0
            cfg.trackc_routed_distill_route_file = str(p)
            self.assertEqual(
                _load_trackc_routed_distill_routes(cfg),
                {
                    "ds1": "additive_single_sum",
                    "ds2": "dataset_multi_mean",
                    "ds3": "train_multi_memory",
                },
            )
            dummy_dataset = object()
            source_dataset, source_kind, source_file = _trackc_routed_distill_source_dataset(dummy_dataset, cfg)
            self.assertIs(source_dataset, dummy_dataset)
            self.assertEqual(source_kind, "training_split")
            self.assertEqual(source_file, "")

        bank = {
            "routes": {"ds1": "additive_single_sum", "ds2": "dataset_multi_mean"},
            "gene_mean_by_dataset": {"ds1": {"A": torch.ones(3), "B": 2.0 * torch.ones(3)}},
            "global_gene_mean": {"C": 3.0 * torch.ones(3)},
            "dataset_multi_mean": {"ds2": 4.0 * torch.ones(3)},
        }
        meta = ConditionMetadata(genes=("B", "A"), perturbation_type_raw="CRISPRi", nperts_obs=2)
        target = get_trackc_routed_distill_target(bank, "ds1", meta)
        torch.testing.assert_close(target, 3.0 * torch.ones(3))
        meta_global = ConditionMetadata(genes=("A", "C"), perturbation_type_raw="CRISPRi", nperts_obs=2)
        target_global = get_trackc_routed_distill_target(bank, "ds1", meta_global)
        torch.testing.assert_close(target_global, 4.0 * torch.ones(3))
        target_ds = get_trackc_routed_distill_target(bank, "ds2", meta)
        torch.testing.assert_close(target_ds, 4.0 * torch.ones(3))

        mem_bank = {
            "routes": {"ds3": "train_multi_memory"},
            "multi_memory_by_dataset": {
                "ds3": [
                    {"dataset": "ds3", "condition": "A+B", "genes": ("A", "B"), "delta": torch.ones(3)},
                    {"dataset": "ds3", "condition": "A+C", "genes": ("A", "C"), "delta": 3.0 * torch.ones(3)},
                    {"dataset": "ds3", "condition": "D+E", "genes": ("D", "E"), "delta": 10.0 * torch.ones(3)},
                ],
                "other": [
                    {"dataset": "other", "condition": "A+B", "genes": ("A", "B"), "delta": 99.0 * torch.ones(3)}
                ],
            },
            "memory_mode": "jaccard",
            "memory_k": 3,
            "memory_min_score": 0.25,
            "memory_scope": "same_dataset",
        }
        mem_meta = ConditionMetadata(genes=("A", "B"), perturbation_type_raw="CRISPRi", nperts_obs=2)
        mem_target = get_trackc_routed_distill_target(mem_bank, "ds3", mem_meta)
        # Scores are 1.0 for A+B and 1/3 for A+C, so the weighted target is 1.5.
        torch.testing.assert_close(mem_target, 1.5 * torch.ones(3))
        mem_bank["memory_mode"] = "off"
        with self.assertRaises(ValueError):
            get_trackc_routed_distill_target(mem_bank, "ds3", mem_meta)

    def test_trackc_routed_distill_loss_is_differentiable(self):
        import torch

        from model.latent.config import Config as LatentConfig
        from model.latent.fm_ot import CondOTPath
        from model.latent.models.mlp import ControlMLPVelocityField
        from model.latent.train import train_step

        torch.manual_seed(7)
        cfg = LatentConfig()
        cfg.emb_dim = 4
        cfg.use_mmd = False
        cfg.use_pert_condition = True
        cfg.time_sampling = "uniform"
        cfg.use_amp = False
        model = ControlMLPVelocityField(
            emb_dim=4,
            d_model=8,
            n_layers=1,
            mlp_ratio=2.0,
            dropout=0.0,
            use_pert_condition=True,
            pert_embed_mode="random_learned",
            pert_cond_dim=8,
            pert_type_emb_dim=4,
            pert_encoder_num_embeddings=32,
            pert_gene_emb_dim=4,
            pert_encoder_dropout=0.0,
            pert_chem_emb_dim=0,
            pert_chem_projector_hidden=0,
            pert_gene_projector_hidden=0,
            pert_type_scale_init=(0.0, -1.0, -1.0, -1.0, 1.0, 1.0),
            pool_aggregations=("mean",),
            pool_scale_init=(1.0,),
            use_condition_delta_head=True,
            condition_delta_head_hidden=8,
        )
        bsz = 3
        src = torch.randn(bsz, 4)
        gt = src + torch.randn(bsz, 4) * 0.1
        gid = torch.tensor([[1, 2], [1, 2], [1, 2]], dtype=torch.long)
        mask = torch.ones(bsz, 2)
        tid = torch.zeros(bsz, dtype=torch.long)
        npt = torch.full((bsz,), 2, dtype=torch.long)
        cid = torch.zeros(bsz, dtype=torch.long)
        pb = (gid, mask, tid, npt, cid, None, None)
        target = torch.full((4,), 5.0)
        out = train_step(
            src,
            gt,
            model,
            CondOTPath(),
            cfg,
            torch.device("cpu"),
            trackc_routed_distill_weight_t=1.0,
            trackc_routed_distill_target=target,
            perturbation_batch=pb,
        )
        self.assertGreater(float(out["trackc_routed_distill"]), 0.0)
        out["loss"].backward()
        grad_norm = sum(
            p.grad.detach().abs().sum().item()
            for n, p in model.named_parameters()
            if n.startswith("condition_delta_head") and p.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_trackc_routed_endpoint_loss_is_differentiable(self):
        import torch

        from model.latent.config import Config as LatentConfig
        from model.latent.fm_ot import CondOTPath
        from model.latent.models.mlp import ControlMLPVelocityField
        from model.latent.train import train_step

        torch.manual_seed(11)
        cfg = LatentConfig()
        cfg.emb_dim = 4
        cfg.use_mmd = False
        cfg.use_pert_condition = True
        cfg.time_sampling = "uniform"
        cfg.use_amp = False
        model = ControlMLPVelocityField(
            emb_dim=4,
            d_model=8,
            n_layers=1,
            mlp_ratio=2.0,
            dropout=0.0,
            use_pert_condition=True,
            pert_embed_mode="random_learned",
            pert_cond_dim=8,
            pert_type_emb_dim=4,
            pert_encoder_num_embeddings=32,
            pert_gene_emb_dim=4,
            pert_encoder_dropout=0.0,
            pert_chem_emb_dim=0,
            pert_chem_projector_hidden=0,
            pert_gene_projector_hidden=0,
            pert_type_scale_init=(0.0, -1.0, -1.0, -1.0, 1.0, 1.0),
            pool_aggregations=("mean",),
            pool_scale_init=(1.0,),
        )
        bsz = 3
        src = torch.randn(bsz, 4)
        gt = src + torch.randn(bsz, 4) * 0.1
        gid = torch.tensor([[1, 2], [1, 2], [1, 2]], dtype=torch.long)
        mask = torch.ones(bsz, 2)
        tid = torch.zeros(bsz, dtype=torch.long)
        npt = torch.full((bsz,), 2, dtype=torch.long)
        cid = torch.zeros(bsz, dtype=torch.long)
        pb = (gid, mask, tid, npt, cid, None, None)
        target = torch.full((4,), 3.0)
        out = train_step(
            src,
            gt,
            model,
            CondOTPath(),
            cfg,
            torch.device("cpu"),
            trackc_routed_endpoint_weight_t=1.0,
            trackc_routed_distill_target=target,
            perturbation_batch=pb,
        )
        self.assertGreater(float(out["trackc_routed_endpoint"]), 0.0)
        out["loss"].backward()
        grad_norm = sum(
            p.grad.detach().abs().sum().item()
            for p in model.parameters()
            if p.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_latent_composition_key_accepts_single_gene_only(self):
        from model.latent.train import _single_gene_composition_key
        from model.utils.conditioning.perturbation import ConditionMetadata

        single = ConditionMetadata(genes=("TP53",), perturbation_type_raw="CRISPRi", nperts_obs=1)
        self.assertEqual(_single_gene_composition_key(single), ("TP53", "CRISPRi"))

        multi = ConditionMetadata(genes=("TP53", "MYC"), perturbation_type_raw="CRISPRi", nperts_obs=2)
        self.assertIsNone(_single_gene_composition_key(multi))

        drug = ConditionMetadata(genes=(), perturbation_type_raw="drug", nperts_obs=0, chem_emb_list=[])
        self.assertIsNone(_single_gene_composition_key(drug))

    def test_raw_pretrain_partial_accum_window_rescales_gradients(self):
        text = (ROOT / "raw_pretrain" / "train.py").read_text(encoding="utf-8")
        self.assertIn("scale_up = ga / float(accum_slot)", text)
        self.assertIn("p.grad.mul_(scale_up)", text)

    def test_latent_dataset_reads_selected_rows_with_duplicates(self):
        import tempfile

        import h5py
        import numpy as np

        from model.latent.dataset import _DatasetHandle

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "toy.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("conditions", data=np.array(["A"], dtype=object), dtype=h5py.string_dtype())
                f.create_dataset("ctrl/offsets", data=np.array([0, 5]))
                f.create_dataset("gt/offsets", data=np.array([0, 5]))
                f.create_dataset("ctrl/emb", data=np.arange(15, dtype=np.float32).reshape(5, 3))
                f.create_dataset("gt/emb", data=(100 + np.arange(15, dtype=np.float32)).reshape(5, 3))

            h = _DatasetHandle(str(path))
            idx = np.array([3, 1, 3, 0], dtype=np.int64)
            np.testing.assert_array_equal(h.read_src_rows("A", idx), h.read_src("A")[idx])
            np.testing.assert_array_equal(h.read_gt_rows("A", idx), h.read_gt("A")[idx])
            h.close()

    def test_latent_resume_recovers_stale_latest_best_score(self):
        import tempfile

        import torch

        from model.latent.train import recover_best_score_from_best_checkpoint

        with tempfile.TemporaryDirectory() as td:
            best = Path(td) / "best.pt"
            torch.save({"best_score": 0.123}, best)
            recovered = recover_best_score_from_best_checkpoint(
                latest_score=float("inf"),
                best_path=best,
                metric_name="test_mmd",
                device=torch.device("cpu"),
            )
            self.assertEqual(recovered, 0.123)

            finite = recover_best_score_from_best_checkpoint(
                latest_score=0.456,
                best_path=best,
                metric_name="test_mmd",
                device=torch.device("cpu"),
            )
            self.assertEqual(finite, 0.456)


if __name__ == "__main__":
    unittest.main()
