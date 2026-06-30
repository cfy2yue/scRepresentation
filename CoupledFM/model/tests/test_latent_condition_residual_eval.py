import numpy as np

from model.latent.eval_condition_residuals import _residual_retrieval_metrics


def test_residual_retrieval_metrics_ranks_true_target_first():
    pred = np.array([0.0, 1.0], dtype=np.float32)
    targets = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = _residual_retrieval_metrics(pred, targets, true_index=1, top_ks=(1, 2))

    assert metrics["retrieval_rank"] == 1
    assert metrics["retrieval_top1"] is True
    assert metrics["retrieval_top2"] is True
    assert metrics["retrieval_best_index"] == 1
