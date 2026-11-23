from loongforge.data.encoder_strided_sampler import EncoderStridedSampler
from megatron.training.datasets.data_samplers import MegatronPretrainingRandomSampler


def test_encoder_strided_sampler_filters_step_relative_positions():
    dataset = list(range(32))
    base_sampler = MegatronPretrainingRandomSampler(
        dataset,
        total_samples=len(dataset),
        consumed_samples=0,
        micro_batch_size=1,
        data_parallel_rank=0,
        data_parallel_size=1,
        data_sharding=True,
    )
    base_batches = list(base_sampler)
    expected = [
        batch
        for index, batch in enumerate(base_batches)
        if 0 <= index % 2 < 1
    ]
    sampler = EncoderStridedSampler(
        dataset,
        total_samples=len(dataset),
        consumed_samples=0,
        micro_batch_size=1,
        data_parallel_rank=0,
        data_parallel_size=1,
        data_sharding=True,
        pp_rank=0,
        tp_size=1,
        model_size=2,
        num_real_microbatch=4,
    )
    assert list(sampler) == expected
