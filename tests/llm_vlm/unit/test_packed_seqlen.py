import torch

from loongforge.train.pretrain.pretrain_vlm import _packed_max_seqlen


def test_packed_max_seqlen_includes_padding_segment():
    cu_seqlens = torch.tensor([0, 681, 1405, 2130, 2855, 32768], dtype=torch.int32)

    max_seqlen = _packed_max_seqlen(cu_seqlens)

    assert max_seqlen == 29913
