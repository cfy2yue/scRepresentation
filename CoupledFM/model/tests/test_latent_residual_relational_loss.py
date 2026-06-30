import torch

from model.latent.train import _pert_residual_relational_loss


def test_residual_relational_loss_prefers_matching_residual_direction():
    target = torch.tensor([1.0, 0.0, 0.0])
    good_pred = torch.tensor([0.9, 0.1, 0.0], requires_grad=True)
    bad_pred = torch.tensor([-1.0, 0.0, 0.0], requires_grad=True)
    bank = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])

    good = _pert_residual_relational_loss(
        good_pred,
        target,
        bank,
        temperature=0.2,
        target_temperature=0.2,
    )
    bad = _pert_residual_relational_loss(
        bad_pred,
        target,
        bank,
        temperature=0.2,
        target_temperature=0.2,
    )

    assert good.item() < bad.item()
    good.backward()
    assert good_pred.grad is not None
    assert torch.isfinite(good_pred.grad).all()
