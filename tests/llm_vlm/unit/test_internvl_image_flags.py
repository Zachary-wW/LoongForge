import torch

from loongforge.models.omni_models.omni_encoder_model import _filter_padded_images


def test_filter_padded_images_drops_zero_flag_slots():
    images = torch.arange(8).reshape(4, 2)
    flags = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    grid = torch.tensor([[1, 1, 1], [9, 9, 9], [2, 2, 2], [9, 9, 9]])

    filtered, filtered_grid = _filter_padded_images(images, grid, flags)

    assert torch.equal(filtered, images[[0, 2]])
    assert torch.equal(filtered_grid, grid[[0, 2]])


def test_filter_padded_images_keeps_non_padded_and_mismatched_inputs():
    images = torch.arange(6).reshape(3, 2)
    all_active = torch.ones(3, dtype=torch.long)

    kept, kept_grid = _filter_padded_images(images, None, all_active)
    assert kept is images
    assert kept_grid is None

    mismatched, mismatched_grid = _filter_padded_images(
        images, None, torch.ones(2, dtype=torch.long)
    )
    assert mismatched is images
    assert mismatched_grid is None
