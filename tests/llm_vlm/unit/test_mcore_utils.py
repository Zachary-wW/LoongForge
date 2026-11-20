from types import SimpleNamespace

from loongforge.train.mcore_utils import is_hf_checkpoint, resolve_convert_pp_layout


def test_is_hf_checkpoint_accepts_safetensors_index(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    assert is_hf_checkpoint(str(tmp_path))


def test_resolve_convert_pp_layout_reads_vlm_foundation():
    layout = SimpleNamespace(layout=[[0, 1]])
    config = SimpleNamespace(foundation=SimpleNamespace(pipeline_model_parallel_layout=layout))
    assert resolve_convert_pp_layout(config) is layout


def test_resolve_convert_pp_layout_ignores_empty_layout():
    layout = SimpleNamespace(layout=None)
    assert resolve_convert_pp_layout(SimpleNamespace(pipeline_model_parallel_layout=layout)) is None
