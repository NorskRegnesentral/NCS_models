"""Focused tests for inference pipeline dataset reconstruction helpers."""

from __future__ import annotations

from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
import torch
import xarray as xr
import zarr

from NCS.inference import pipeline
from NCS.inference.pipeline import _build_dataset_from_phase_sliced, _build_dataset_from_sliced


def _make_phase_features(
    tmp_path,
    primary_size: int,
    spatial_size: int,
    t_size: int,
    hidden_size: int,
    densify: int,
):
    phase_features = {}
    for sp_phase_idx in range(densify):
        sp_count = len(range(sp_phase_idx, spatial_size, densify))
        for t_phase_idx in range(densify):
            t_count = len(range(t_phase_idx, t_size, densify))
            arr = zarr.open_array(
                str(tmp_path / f"phase_{sp_phase_idx}_{t_phase_idx}.zarr"),
                mode="w",
                shape=(primary_size, sp_count, t_count, hidden_size),
                chunks=(1, sp_count, t_count, hidden_size),
                dtype=np.float32,
                fill_value=0,
            )
            values = np.zeros((primary_size, sp_count, t_count, hidden_size), dtype=np.float32)
            for primary_idx in range(primary_size):
                for sp_idx in range(sp_count):
                    for t_idx in range(t_count):
                        base = (
                            primary_idx * 1000
                            + (sp_phase_idx + sp_idx * densify) * 100
                            + (t_phase_idx + t_idx * densify) * 10
                        )
                        values[primary_idx, sp_idx, t_idx, 0] = base + 1
                        values[primary_idx, sp_idx, t_idx, 1] = base + 2
            arr[:] = values
            phase_features[(sp_phase_idx, t_phase_idx)] = arr
    return phase_features


def test_build_dataset_from_sliced_uses_dense_coords_for_dir0(tmp_path):
    arr = zarr.open_array(
        str(tmp_path / "features.zarr"),
        mode="w",
        shape=(3, 4, 2, 5),
        chunks=(1, 4, 2, 5),
        dtype=np.float32,
    )
    arr[:] = 1.0
    ds = _build_dataset_from_sliced(
        arr,
        "dir0",
        np.array([8, 16, 24], dtype=np.int32),
        np.array([8, 12, 16, 20], dtype=np.int32),
        np.array([8, 16], dtype=np.int32),
        5,
    )

    np.testing.assert_array_equal(ds.inline.values, np.array([8, 16, 24], dtype=np.int32))
    np.testing.assert_array_equal(ds.xline.values, np.array([8, 12, 16, 20], dtype=np.int32))
    np.testing.assert_array_equal(ds.time_depth.values, np.array([8, 16], dtype=np.int32))


def test_build_dataset_from_phase_sliced_reconstructs_dir0(tmp_path):
    primary_coords = np.array([10, 20, 30], dtype=np.int32)
    spatial_coords = np.array([100, 102, 104, 106, 108], dtype=np.int32)
    t_coords = np.array([1000, 1002, 1004, 1006], dtype=np.int32)
    densify = 2
    hidden_size = 2

    phase_features = _make_phase_features(
        tmp_path, len(primary_coords), len(spatial_coords), len(t_coords), hidden_size, densify
    )
    ds = _build_dataset_from_phase_sliced(
        phase_features, "dir0", primary_coords, spatial_coords, t_coords, hidden_size, densify
    )

    expected = np.zeros((len(primary_coords), len(spatial_coords), len(t_coords), hidden_size), dtype=np.float32)
    for (sp_phase_idx, t_phase_idx), phase_array in phase_features.items():
        expected[:, sp_phase_idx::densify, t_phase_idx::densify, :] = np.asarray(phase_array)

    np.testing.assert_array_equal(ds.inline.values, primary_coords)
    np.testing.assert_array_equal(ds.xline.values, spatial_coords)
    np.testing.assert_array_equal(ds.time_depth.values, t_coords)
    np.testing.assert_array_equal(ds.features.transpose("inline", "xline", "time_depth", "feature").values, expected)


def test_build_dataset_from_phase_sliced_reconstructs_dir90(tmp_path):
    primary_coords = np.array([11, 21, 31], dtype=np.int32)
    spatial_coords = np.array([201, 203, 205, 207, 209], dtype=np.int32)
    t_coords = np.array([3001, 3003, 3005, 3007], dtype=np.int32)
    densify = 2
    hidden_size = 2

    phase_features = _make_phase_features(
        tmp_path, len(primary_coords), len(spatial_coords), len(t_coords), hidden_size, densify
    )
    ds = _build_dataset_from_phase_sliced(
        phase_features, "dir90", primary_coords, spatial_coords, t_coords, hidden_size, densify
    )

    expected = np.zeros((len(primary_coords), len(spatial_coords), len(t_coords), hidden_size), dtype=np.float32)
    for (sp_phase_idx, t_phase_idx), phase_array in phase_features.items():
        expected[:, sp_phase_idx::densify, t_phase_idx::densify, :] = np.asarray(phase_array)

    np.testing.assert_array_equal(ds.inline.values, spatial_coords)
    np.testing.assert_array_equal(ds.xline.values, primary_coords)
    np.testing.assert_array_equal(ds.time_depth.values, t_coords)
    np.testing.assert_array_equal(
        ds.features.transpose("inline", "xline", "time_depth", "feature").values,
        np.transpose(expected, (1, 0, 2, 3)),
    )


def test_build_dataset_from_phase_sliced_can_save_to_zarr_with_uneven_phase_sizes(tmp_path):
    primary_coords = np.array([10, 20, 30], dtype=np.int32)
    spatial_coords = np.arange(119, dtype=np.int32)
    t_coords = np.arange(58, dtype=np.int32)
    densify = 2
    hidden_size = 2

    phase_features = _make_phase_features(
        tmp_path, len(primary_coords), len(spatial_coords), len(t_coords), hidden_size, densify
    )
    ds = _build_dataset_from_phase_sliced(
        phase_features, "dir0", primary_coords, spatial_coords, t_coords, hidden_size, densify
    )

    ds.to_zarr(str(tmp_path / "features.zarr"), mode="w")


def test_centered_tile_slices_clip_boundary_tiles():
    assert pipeline._centered_tile_slices(tile_center_patch=0, patches_per_crop=4, total_patches=9) == (
        slice(0, 2),
        slice(2, 4),
    )
    assert pipeline._centered_tile_slices(tile_center_patch=4, patches_per_crop=4, total_patches=9) == (
        slice(2, 6),
        slice(0, 4),
    )
    assert pipeline._centered_tile_slices(tile_center_patch=8, patches_per_crop=4, total_patches=9) == (
        slice(6, 9),
        slice(0, 3),
    )


def test_process_2d_batch_clips_centered_boundary_crop():
    token_grid = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
    last_hidden_state = torch.cat([torch.zeros((1, 1, 1), dtype=torch.float32), token_grid], dim=1)

    class _FakeModel:
        def __call__(self, pixel_values):
            return SimpleNamespace(last_hidden_state=last_hidden_state)

    feature_sum = torch.zeros((9, 9, 1), dtype=torch.float32)
    weight_sum = torch.zeros((9, 9), dtype=torch.float32)

    pipeline._process_2d_batch(
        pixel_values=torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        batch_meta=[(0, 0)],
        model=_FakeModel(),
        device="cpu",
        torch_dtype=torch.float32,
        non_blocking=False,
        patches_per_crop=4,
        w_sp=torch.ones((1, 4), dtype=torch.float32),
        w_t=torch.ones((1, 4), dtype=torch.float32),
        feature_sum=feature_sum,
        weight_sum=weight_sum,
        hidden_size=1,
        sp_tile_centers=[0],
        t_tile_centers=[0],
    )

    expected = torch.tensor([[10.0, 14.0], [11.0, 15.0]], dtype=torch.float32)
    torch.testing.assert_close(feature_sum[:2, :2, 0], expected)
    torch.testing.assert_close(weight_sum[:2, :2], torch.ones((2, 2), dtype=torch.float32))


def test_process_25d_batch_clips_centered_boundary_crop():
    view0 = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
    view1 = (100 + torch.arange(16, dtype=torch.float32)).reshape(1, 16, 1)
    last_hidden_state = torch.cat([torch.zeros((1, 1, 1), dtype=torch.float32), view0, view1], dim=1)

    class _FakeModel:
        def __call__(self, pixel_values, directions):
            return SimpleNamespace(last_hidden_state=last_hidden_state)

    feature_sum = torch.zeros((9, 9, 1), dtype=torch.float32)
    weight_sum = torch.zeros((9, 9), dtype=torch.float32)

    pipeline._process_25d_batch(
        pixel_values=torch.zeros((1, 2, 4, 4), dtype=torch.float32),
        batch_meta=[(0, 0)],
        model=_FakeModel(),
        device="cpu",
        torch_dtype=torch.float32,
        non_blocking=False,
        patches_per_crop=4,
        w_sp=torch.ones((1, 4), dtype=torch.float32),
        w_t=torch.ones((1, 4), dtype=torch.float32),
        feature_sum=feature_sum,
        weight_sum=weight_sum,
        hidden_size=1,
        direction_indices=torch.tensor([[0, 2]], dtype=torch.int32),
        travel_view_idx=1,
        num_views=2,
        patches_per_view=16,
        sp_tile_centers=[0],
        t_tile_centers=[0],
    )

    expected = torch.tensor([[110.0, 114.0], [111.0, 115.0]], dtype=torch.float32)
    torch.testing.assert_close(feature_sum[:2, :2, 0], expected)
    torch.testing.assert_close(weight_sum[:2, :2], torch.ones((2, 2), dtype=torch.float32))


@pytest.mark.parametrize(
    ("model_type", "runner_name"),
    [
        ("vit", "_run_2d"),
        ("vit25d", "_run_25d"),
        ("vit3d", "_run_3d"),
    ],
)
def test_run_inference_forwards_loader_speedup_settings_to_all_model_types(
    tmp_path, monkeypatch, model_type, runner_name
):
    captured = {}

    class _FakeModel:
        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return self

    def fake_runner(**kwargs):
        captured["runner"] = runner_name
        captured["densify"] = kwargs["densify"]
        captured["overlap_filter"] = kwargs["overlap_filter"]
        captured["num_workers"] = kwargs["num_workers"]
        captured["prefetch_factor"] = kwargs["prefetch_factor"]
        captured["pin_memory"] = kwargs["pin_memory"]
        captured["pad_before"] = kwargs["pad_before"]
        captured["non_blocking"] = kwargs["non_blocking"]
        if "max_preload" in kwargs:
            captured["max_preload"] = kwargs["max_preload"]
        return (
            xr.Dataset(
                {
                    "features": (
                        ["inline", "xline", "time_depth", "feature"],
                        da.from_array(np.ones((1, 1, 1, 2), dtype=np.float32), chunks=(1, 1, 1, 2)),
                    )
                },
                coords={"inline": [0], "xline": [0], "time_depth": [0], "feature": [0, 1]},
            ),
            None,
        )

    class _FakeSeismicFile:
        def close(self):
            return None

    monkeypatch.setattr(
        pipeline.AutoConfig,
        "from_pretrained",
        lambda _path: SimpleNamespace(model_type=model_type, hidden_size=2, directions=["dir0", "dir90"]),
    )

    class _FakeProcessor:
        def __init__(self, **kwargs):
            captured["processor_kwargs"] = kwargs

    monkeypatch.setattr(pipeline, "_load_model", lambda *args, **kwargs: _FakeModel())
    monkeypatch.setattr(pipeline, "open_seismic", lambda _path: _FakeSeismicFile())
    monkeypatch.setattr(pipeline, "cube_shape", lambda _f: (8, 8, 8))
    monkeypatch.setattr(pipeline, "compute_cube_stats", lambda _f, n_traces=None: (0.0, 1.0))
    monkeypatch.setattr(pipeline, "SeismicProcessor", _FakeProcessor)
    monkeypatch.setattr(pipeline, runner_name, fake_runner)

    class _FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def barrier(self):
            return None

    monkeypatch.setattr(pipeline, "AsyncWriter", _FakeWriter)

    output_path = tmp_path / f"{model_type}.zarr"
    pipeline.run_inference(
        model_path="model",
        input_path="cube.sgz",
        output_path=output_path,
        densify=4,
        overlap_filter="exponential",
        batch_size=4,
        device="cuda:0",
        num_workers=6,
        prefetch_factor=3,
        pin_memory=True,
        max_preload=123,
        pad_before=False,
        normalization_mode="zscore",
    )

    assert captured["runner"] == runner_name
    assert captured["device"] == "cuda:0"
    assert captured["densify"] == 4
    assert captured["overlap_filter"] == "exponential"
    assert captured["num_workers"] == 6
    assert captured["prefetch_factor"] == 3
    assert captured["pin_memory"] is True
    assert captured["non_blocking"] is True
    assert captured["pad_before"] is False
    assert captured["processor_kwargs"]["normalization_mode"] == "zscore"
    if runner_name == "_run_2d":
        assert "max_preload" not in captured
    else:
        assert captured["max_preload"] == 123


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_run_inference_writes_fms_metadata_from_base_model_type_and_25d_views(tmp_path, monkeypatch, suffix):
    if suffix == ".nc":
        pytest.importorskip("netCDF4")
    captured = {}

    class _FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    def fake_runner(**kwargs):
        captured["input_views"] = kwargs["input_views"]
        return (
            xr.Dataset(
                {
                    "features": (
                        ["inline", "xline", "time_depth", "feature"],
                        da.from_array(np.ones((1, 1, 1, 2), dtype=np.float32), chunks=(1, 1, 1, 2)),
                    )
                },
                coords={"inline": [0], "xline": [0], "time_depth": [0], "feature": [0, 1]},
            ),
            None,
        )

    class _FakeSeismicFile:
        def close(self):
            return None

    class _FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def barrier(self):
            return None

    monkeypatch.setattr(
        pipeline.AutoConfig,
        "from_pretrained",
        lambda _path: SimpleNamespace(model_type="vit25d_mae", hidden_size=2, directions=["dir0", "dir45", "dir90"]),
    )
    monkeypatch.setattr(pipeline, "_load_model", lambda *args, **kwargs: _FakeModel())
    monkeypatch.setattr(pipeline, "open_seismic", lambda _path: _FakeSeismicFile())
    monkeypatch.setattr(pipeline, "cube_shape", lambda _f: (8, 8, 8))
    monkeypatch.setattr(pipeline, "compute_cube_stats", lambda _f, n_traces=None: (0.0, 1.0))
    monkeypatch.setattr(pipeline, "SeismicProcessor", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(pipeline, "_run_25d", fake_runner)
    monkeypatch.setattr(pipeline, "AsyncWriter", _FakeWriter)

    output_path = tmp_path / f"vit25d{suffix}"
    pipeline.run_inference(
        model_path="model",
        input_path="cube.sgz",
        output_path=output_path,
        direction="dir0",
        input_views=["dir0", "dir90"],
    )

    with xr.open_dataset(output_path) as root:
        attrs = dict(root.attrs)

    assert captured["input_views"] == ["dir0", "dir90"]
    assert attrs["model_type"] == "vit25d_mae"
    if suffix == ".zarr":
        expected_directions = ["dir0:dir90"]
    else:
        expected_directions = "dir0:dir90"
    assert attrs["inference_directions"] == expected_directions


def test_run_inference_rejects_25d_input_views_without_primary_direction(tmp_path, monkeypatch):
    class _FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    class _FakeSeismicFile:
        def close(self):
            return None

    monkeypatch.setattr(
        pipeline.AutoConfig,
        "from_pretrained",
        lambda _path: SimpleNamespace(model_type="vit25d", hidden_size=2, directions=["dir0", "dir45", "dir90"]),
    )
    monkeypatch.setattr(pipeline, "_load_model", lambda *args, **kwargs: _FakeModel())
    monkeypatch.setattr(pipeline, "open_seismic", lambda _path: _FakeSeismicFile())
    monkeypatch.setattr(pipeline, "cube_shape", lambda _f: (8, 8, 8))
    monkeypatch.setattr(pipeline, "compute_cube_stats", lambda _f, n_traces=None: (0.0, 1.0))
    monkeypatch.setattr(pipeline, "SeismicProcessor", lambda **kwargs: SimpleNamespace(**kwargs))

    class _FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def barrier(self):
            return None

    monkeypatch.setattr(pipeline, "AsyncWriter", _FakeWriter)

    with pytest.raises(ValueError, match="must be included in input_views"):
        pipeline.run_inference(
            model_path="model",
            input_path="cube.sgz",
            output_path=tmp_path / "vit25d.zarr",
            direction="dir0",
            input_views=["dir90"],
        )
