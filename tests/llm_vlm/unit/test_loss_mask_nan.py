import torch

from loongforge.train.get_loss_func import _masked_sum


def test_masked_sum_ignores_nan_in_masked_positions():
    losses = torch.tensor([1.0, float("nan"), 3.0], requires_grad=True)
    mask = torch.tensor([1.0, 0.0, 1.0])

    result = _masked_sum(losses, mask)

    assert result.item() == 4.0
    result.backward()
    assert losses.grad.tolist() == [1.0, 0.0, 1.0]


def test_masked_sum_handles_zero_weight_batch():
    losses = torch.tensor([float("nan"), float("nan")], requires_grad=True)
    weights = torch.zeros(2)

    result = _masked_sum(losses, weights)

    assert result.item() == 0.0
    result.backward()
    assert losses.grad.tolist() == [0.0, 0.0]
