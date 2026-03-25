"""Tests for inference DataLoader helpers."""

import os
from types import MethodType

import numpy as np
import torch
from torch.utils.data import IterableDataset

from NCS.inference.dataloading import (
    PreloadedSubcube,
    TwoDPositionDataset,
    _BaseInferenceDataset,
    build_inference_loader,
    build_processor_config,
    clamp_max_preload,
    clamp_prefetch_factor,
    clamp_worker_batch_size,
    expand_worker_preload_bounds,
    resolve_loader_settings,
    shape_aware_25d_primary_preload,
)
from NCS.inference.tiling import TilePositions
from NCS.processors import SeismicProcessor


class _SingleItemDataset(IterableDataset):
    def __iter__(self):
        yield {"value": 1}


class _PackableDataset(_BaseInferenceDataset):
    def __iter__(self):
        yield from ()


def test_resolve_loader_settings_defaults_for_cuda(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    num_workers, pin_memory = resolve_loader_settings(device="cuda:0", num_workers=None, pin_memory=None)
    assert num_workers == 24
    assert pin_memory is True


def test_resolve_loader_settings_defaults_for_cpu(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    num_workers, pin_memory = resolve_loader_settings(device="cpu", num_workers=None, pin_memory=None)
    assert num_workers == 24
    assert pin_memory is False


def test_clamp_prefetch_factor_never_drops_below_one():
    assert clamp_prefetch_factor(0) == 1
    assert clamp_prefetch_factor(4) == 4


def test_clamp_max_preload_never_drops_below_one():
    assert clamp_max_preload(0) == 1
    assert clamp_max_preload(128) == 128


def test_clamp_worker_batch_size_never_drops_below_one():
    assert clamp_worker_batch_size(0) == 1
    assert clamp_worker_batch_size(16) == 16


def test_shape_aware_25d_primary_preload_limits_primary_depth():
    assert shape_aware_25d_primary_preload(crop_size=224, patch_size=16, max_preload=500) == 352
    assert shape_aware_25d_primary_preload(crop_size=224, patch_size=16, max_preload=300) == 300


def test_build_processor_config_preserves_runtime_settings():
    processor = SeismicProcessor(
        do_normalize=False,
        normalize_mean=1.5,
        normalize_std=2.5,
        normalization_mode="zscore",
        clip_sigma=4.0,
        model_type="vit25d",
    )

    assert build_processor_config(processor) == {
        "do_normalize": False,
        "normalize_mean": 1.5,
        "normalize_std": 2.5,
        "normalization_mode": "zscore",
        "clip_sigma": 4.0,
        "model_type": "vit25d",
    }


def test_build_inference_loader_supports_single_work_units():
    loader = build_inference_loader(
        dataset=_SingleItemDataset(),
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
    )

    assert next(iter(loader)) == {"value": 1}


def test_pack_batches_flattens_worker_prebatches():
    dataset = _PackableDataset("tests/test_data/small_2bit_5x5x50.sgz", build_processor_config(SeismicProcessor()))
    batch_a = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    batch_b = torch.arange(24, 36, dtype=torch.float32).reshape(1, 3, 2, 2)

    pixel_values, meta, batch_splits = dataset._pack_batches(
        [batch_a, batch_b],
        [[("a", 0), ("a", 1)], [("b", 0)]],
    )

    assert tuple(pixel_values.shape) == (3, 3, 2, 2)
    assert meta == [("a", 0), ("a", 1), ("b", 0)]
    assert batch_splits == [2, 1]


def test_preloaded_subcube_reuses_cached_region():
    calls = []

    def read_fn(primary_start, primary_count, spatial_start, spatial_count, sample_start, sample_count):
        calls.append((primary_start, primary_count, spatial_start, spatial_count, sample_start, sample_count))
        return np.zeros((primary_count, spatial_count, sample_count), dtype=np.float32)

    cache = PreloadedSubcube(
        read_fn=read_fn,
        total_primary=200,
        total_spatial=200,
        total_sample=200,
        max_preload=32,
    )

    arr1 = cache.fetch(10, 20, 30, 40, 50, 60)
    arr2 = cache.fetch(12, 18, 32, 38, 52, 58)

    assert arr1.shape == (10, 10, 10)
    assert arr2.shape == (6, 6, 6)
    assert len(calls) == 1


def test_preloaded_subcube_caps_expansion_by_max_preload_when_request_is_smaller():
    calls = []

    def read_fn(primary_start, primary_count, spatial_start, spatial_count, sample_start, sample_count):
        calls.append((primary_start, primary_count, spatial_start, spatial_count, sample_start, sample_count))
        return np.zeros((primary_count, spatial_count, sample_count), dtype=np.float32)

    cache = PreloadedSubcube(
        read_fn=read_fn,
        total_primary=5000,
        total_spatial=5000,
        total_sample=5000,
        max_preload=100,
    )

    cache.fetch(1000, 1010, 2000, 2010, 3000, 3010)

    assert calls == [(955, 100, 1955, 100, 2955, 100)]


def test_preloaded_subcube_respects_worker_primary_bounds():
    calls = []

    def read_fn(primary_start, primary_count, spatial_start, spatial_count, sample_start, sample_count):
        calls.append((primary_start, primary_count, spatial_start, spatial_count, sample_start, sample_count))
        return np.zeros((primary_count, spatial_count, sample_count), dtype=np.float32)

    cache = PreloadedSubcube(
        read_fn=read_fn,
        total_primary=5000,
        total_spatial=5000,
        total_sample=5000,
        max_preload=100,
        primary_lower=1000,
        primary_upper=1100,
    )

    cache.fetch(1005, 1015, 2000, 2010, 3000, 3010)

    assert calls == [(1000, 100, 1955, 100, 2955, 100)]


def test_preloaded_subcube_fetch_context_supports_axis_specific_caps():
    calls = []

    def read_fn(primary_start, primary_count, spatial_start, spatial_count, sample_start, sample_count):
        calls.append((primary_start, primary_count, spatial_start, spatial_count, sample_start, sample_count))
        return np.zeros((primary_count, spatial_count, sample_count), dtype=np.float32)

    cache = PreloadedSubcube(
        read_fn=read_fn,
        total_primary=5000,
        total_spatial=5000,
        total_sample=5000,
        max_preload=500,
    )

    cache.fetch_context(
        1000,
        1010,
        2000,
        2100,
        3000,
        3200,
        primary_max_preload=32,
    )

    assert calls == [(989, 32, 1800, 500, 2850, 500)]


def test_expand_worker_preload_bounds_adds_halo_for_small_worker_spans():
    assert expand_worker_preload_bounds(total_primary=1000, worker_lower=200, worker_upper=220, max_preload=500) == (
        0,
        500,
    )


def test_expand_worker_preload_bounds_uses_full_axis_when_volume_is_small():
    assert expand_worker_preload_bounds(total_primary=300, worker_lower=50, worker_upper=70, max_preload=500) == (
        0,
        300,
    )


def test_two_d_position_dataset_centers_first_crop(monkeypatch):
    full_slice = np.arange(6 * 6, dtype=np.float32).reshape(6, 6)

    monkeypatch.setattr(
        "NCS.inference.dataloading._read_direction_slice",
        lambda *_args, **_kwargs: full_slice,
    )

    dataset = TwoDPositionDataset(
        input_path="cube.sgz",
        processor_config={},
        direction="dir0",
        patch_size=2,
        crop_size=4,
        densify=1,
        n_primary=10,
        primary_phase_idx=0,
        primary_phase_offset=0,
        primary_count=1,
        sp_phase_idx=0,
        sp_phase_offset=0,
        t_phase_idx=0,
        t_phase_offset=0,
        tile_sp=TilePositions(starts=[0], n_patches_total=3, pad_before=0, pad_after=0),
        sp_tail_pad=0,
        tile_t=TilePositions(starts=[0], n_patches_total=3, pad_before=0, pad_after=0),
        t_tail_pad=0,
        worker_batch_size=1,
    )
    dataset._get_file = MethodType(lambda self: object(), dataset)
    dataset._preprocess = MethodType(
        lambda self, crops: torch.as_tensor(np.stack(crops), dtype=torch.float32),
        dataset,
    )

    item = next(iter(dataset))

    expected = np.pad(full_slice, ((2, 2), (2, 2)), mode="constant")[:4, :4]
    np.testing.assert_array_equal(item["pixel_values"][0].numpy(), expected)
