import math

from model.latent.train import (
    is_better_score,
    selection_metric_direction,
    selection_metric_value,
)


def test_composite_pearson_pert_minus_mmd_is_maximized():
    metrics = {"pearson_pert": 0.08, "test_mmd": 0.03}
    assert selection_metric_direction("pearson_pert_minus_mmd") == "max"
    assert math.isclose(
        selection_metric_value("pearson_pert_minus_mmd", metrics, mmd_lambda=0.5),
        0.065,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert is_better_score("pearson_pert_minus_mmd", 0.07, 0.06)


def test_composite_pearson_ctrl_minus_mmd_is_maximized():
    metrics = {"pearson_ctrl": 0.22, "test_mmd": 0.04}
    assert selection_metric_direction("pearson_ctrl_minus_mmd") == "max"
    assert math.isclose(
        selection_metric_value("pearson_ctrl_minus_mmd", metrics, mmd_lambda=2.0),
        0.14,
        rel_tol=0,
        abs_tol=1e-12,
    )
