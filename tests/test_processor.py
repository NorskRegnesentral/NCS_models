"""Tests for SeismicProcessor."""

import numpy as np
import pytest
import torch

from NCS.inference.views import extract_2d_crop_views, extract_2d_crop_views_from_cache
from NCS.processors import SeismicProcessor


def _expected_diagonal_crop(crop_2d: np.ndarray) -> np.ndarray:
    """Mirror the expected diagonal handling by center crop and resize along the spatial axis."""
    target_size = crop_2d.shape[0]
    diagonal_size = max(1, int(target_size / np.sqrt(2)))
    start = (target_size - diagonal_size) // 2
    cropped = crop_2d[start : start + diagonal_size]
    if cropped.shape[0] == target_size:
        return cropped.astype(np.float32, copy=False)

    src = np.arange(cropped.shape[0], dtype=np.float32)
    dst = np.linspace(0, cropped.shape[0] - 1, num=target_size, dtype=np.float32)
    resized = np.stack([np.interp(dst, src, cropped[:, col]) for col in range(cropped.shape[1])], axis=1)
    return resized.astype(np.float32, copy=False)


class TestSeismicProcessor:
    def test_init_defaults(self):
        proc = SeismicProcessor()
        assert proc.do_normalize is True
        assert proc.normalize_mean is None
        assert proc.normalize_std is None
        assert proc.normalization_mode == "std"
        assert proc.clip_sigma == 3.0
        assert proc.model_type == "vit3d"

    def test_preprocess_3d_shape(self):
        proc = SeismicProcessor(do_normalize=False, model_type="vit3d")
        crop = np.random.randn(4, 6, 8).astype(np.float32)  # (il, xl, t)
        result = proc.preprocess(crop)
        pv = result["pixel_values"]
        # Should be (B=1, C=1, T=8, H=4, W=6)
        assert pv.shape == (1, 1, 8, 4, 6)
        assert isinstance(pv, torch.Tensor)

    def test_preprocess_3d_batch(self):
        proc = SeismicProcessor(do_normalize=False, model_type="vit3d")
        crops = [np.random.randn(4, 6, 8).astype(np.float32) for _ in range(3)]
        result = proc.preprocess(crops)
        pv = result["pixel_values"]
        assert pv.shape == (3, 1, 8, 4, 6)

    def test_preprocess_2d_shape(self):
        proc = SeismicProcessor(do_normalize=False, model_type="vit")
        crop = np.random.randn(16, 8).astype(np.float32)  # (spatial, time)
        result = proc.preprocess(crop)
        pv = result["pixel_values"]
        # Should be (B=1, C=3, H=8, W=16) — channel repeated to 3 for standard ViT
        assert pv.shape == (1, 3, 8, 16)

    def test_preprocess_25d_shape(self):
        proc = SeismicProcessor(do_normalize=False, model_type="vit25d")
        crop = np.random.randn(4, 16, 8).astype(np.float32)  # (n_views, spatial, time)
        result = proc.preprocess(crop)
        pv = result["pixel_values"]
        # Should be (B=1, V=4, H=8, W=16)
        assert pv.shape == (1, 4, 8, 16)

    def test_zscore_normalization(self):
        proc = SeismicProcessor(
            do_normalize=True,
            normalize_mean=10.0,
            normalize_std=2.0,
            normalization_mode="zscore",
            clip_sigma=3.0,
            model_type="vit",
        )
        # Create data that after normalization: (14 - 10) / 2 = 2.0 (within clip)
        # (20 - 10) / 2 = 5.0 -> clipped to 3.0
        crop = np.array([[14.0, 20.0], [6.0, -10.0]], dtype=np.float32)
        result = proc.preprocess(crop)
        pv = result["pixel_values"].numpy()
        # Check clipping
        assert pv.max() <= 3.0
        assert pv.min() >= -3.0
        # Check correct normalization of in-range value
        np.testing.assert_almost_equal(pv[0, 0, 0, 0], 2.0, decimal=5)

    def test_std_normalization_is_default(self):
        proc = SeismicProcessor(
            do_normalize=True,
            normalize_mean=10.0,
            normalize_std=2.0,
            clip_sigma=3.0,
            model_type="vit",
        )
        crop = np.array([[14.0, 20.0], [6.0, -10.0]], dtype=np.float32)
        result = proc.preprocess(crop)
        pv = result["pixel_values"].numpy()
        np.testing.assert_almost_equal(pv[0, 0, 0, 0], 3.0, decimal=5)
        assert pv.max() <= 3.0
        assert pv.min() >= -3.0

    def test_std_normalization_requires_only_std(self):
        proc = SeismicProcessor(do_normalize=True, normalization_mode="std", model_type="vit")
        crop = np.random.randn(16, 16).astype(np.float32)
        with pytest.raises(ValueError, match="normalize_std must be set"):
            proc.preprocess(crop)

    def test_zscore_normalization_requires_mean(self):
        proc = SeismicProcessor(do_normalize=True, normalization_mode="zscore", normalize_std=1.0, model_type="vit")
        crop = np.random.randn(16, 16).astype(np.float32)
        with pytest.raises(ValueError, match="normalize_mean must be set"):
            proc.preprocess(crop)

    def test_preprocess_override_stats(self):
        proc = SeismicProcessor(
            do_normalize=True,
            normalize_mean=0.0,
            normalize_std=1.0,
            normalization_mode="zscore",
            model_type="vit",
        )
        crop = np.ones((8, 8), dtype=np.float32) * 5.0
        # Override with different stats
        result = proc.preprocess(crop, normalize_mean=5.0, normalize_std=1.0, normalization_mode="zscore")
        pv = result["pixel_values"].numpy()
        np.testing.assert_almost_equal(pv[0, 0], 0.0, decimal=5)

    def test_3d_axis_reordering(self):
        proc = SeismicProcessor(do_normalize=False, model_type="vit3d")
        # Create distinguishable data: shape (il=2, xl=3, t=4)
        crop = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        result = proc.preprocess(crop)
        pv = result["pixel_values"].numpy()
        # After reorder: (B=1, C=1, T=4, H=2, W=3)
        assert pv.shape == (1, 1, 4, 2, 3)
        # Verify data integrity: crop[0, 0, 0] should be at pv[0, 0, 0, 0, 0]
        assert pv[0, 0, 0, 0, 0] == crop[0, 0, 0]
        # crop[1, 2, 3] should be at pv[0, 0, 3, 1, 2]
        assert pv[0, 0, 3, 1, 2] == crop[1, 2, 3]

    @pytest.mark.parametrize(
        ("model_type", "shape"),
        [
            ("vit3d", (4, 6, 8)),
            ("vit", (16, 8)),
            ("vit25d", (3, 16, 8)),
        ],
    )
    def test_batched_preprocess_matches_individual_stack(self, model_type, shape):
        proc = SeismicProcessor(
            do_normalize=True,
            normalize_mean=0.5,
            normalize_std=2.0,
            normalization_mode="zscore",
            clip_sigma=3.0,
            model_type=model_type,
        )
        crops = [np.random.randn(*shape).astype(np.float32) for _ in range(3)]

        batch = proc.preprocess(crops)["pixel_values"]
        singles = torch.stack([proc.preprocess(crop)["pixel_values"][0] for crop in crops], dim=0)

        torch.testing.assert_close(batch, singles)


def test_extract_2d_crop_views_matches_expected_multiview_geometry():
    subvolume = np.arange(6 * 8 * 8, dtype=np.float32).reshape(6, 8, 8)
    views = ["dir0", "dir90", "dir45", "dir135"]
    result = extract_2d_crop_views(
        subvolume=subvolume,
        center_idx=3,
        spatial_start=2,
        time_start=1,
        crop_size=4,
        views=views,
        traversal_direction="dir0",
        circular_primary=False,
    )

    expected_dir0 = subvolume[3, 2:6, 1:5]
    expected_dir90 = subvolume[1:5, 4, 1:5]
    expected_dir45 = _expected_diagonal_crop(np.stack([subvolume[1 + i, 2 + i, 1:5] for i in range(4)], axis=0))
    expected_dir135 = _expected_diagonal_crop(np.stack([subvolume[1 + i, 5 - i, 1:5] for i in range(4)], axis=0))

    np.testing.assert_allclose(result[0], expected_dir0)
    np.testing.assert_allclose(result[1], expected_dir90)
    np.testing.assert_allclose(result[2], expected_dir45)
    np.testing.assert_allclose(result[3], expected_dir135)


@pytest.mark.parametrize(
    ("primary_idx", "crop_spatial_start", "crop_time_start"),
    [
        (5, 4, 5),
        (1, 2, 2),
        (1, 1, 2),
    ],
)
def test_extract_2d_crop_views_from_cache_matches_legacy_slab_path(primary_idx, crop_spatial_start, crop_time_start):
    cube = np.arange(10 * 12 * 14, dtype=np.float32).reshape(10, 12, 14)
    crop_size = 4
    half = crop_size // 2
    views = ["dir0", "dir90", "dir45", "dir135"]
    sp_phase_offset = 3
    sp_axis_length = 5
    t_phase_offset = 4
    t_axis_length = 6
    tile_sp_pad_before = 1
    tile_t_pad_before = 2
    sp_tail_pad = 1
    t_tail_pad = 0

    cache_primary_bounds = (0, 8)
    cache_spatial_bounds = (2, 10)
    cache_time_bounds = (1, 12)
    cache_data = cube[
        cache_primary_bounds[0] : cache_primary_bounds[1],
        cache_spatial_bounds[0] : cache_spatial_bounds[1],
        cache_time_bounds[0] : cache_time_bounds[1],
    ]

    source_start = primary_idx - half
    source_end = source_start + crop_size
    read_start = max(source_start, 0)
    read_end = min(source_end, cube.shape[0])
    slab_sp_size = half + tile_sp_pad_before + sp_axis_length + sp_tail_pad + half
    slab_t_size = half + tile_t_pad_before + t_axis_length + t_tail_pad + half
    data_sp_start = half + tile_sp_pad_before
    data_t_start = half + tile_t_pad_before
    slab = np.zeros((crop_size, slab_sp_size, slab_t_size), dtype=np.float32)
    dst_primary_start = read_start - source_start
    phase_data = cube[
        read_start:read_end,
        sp_phase_offset : sp_phase_offset + sp_axis_length,
        t_phase_offset : t_phase_offset + t_axis_length,
    ]
    slab[
        dst_primary_start : dst_primary_start + phase_data.shape[0],
        data_sp_start : data_sp_start + phase_data.shape[1],
        data_t_start : data_t_start + phase_data.shape[2],
    ] = phase_data
    center_in_slab = primary_idx - source_start
    legacy = extract_2d_crop_views(
        subvolume=slab,
        center_idx=center_in_slab,
        spatial_start=(crop_spatial_start - sp_phase_offset) + tile_sp_pad_before + half,
        time_start=(crop_time_start - t_phase_offset) + tile_t_pad_before + half,
        crop_size=crop_size,
        views=views,
        traversal_direction="dir0",
        circular_primary=True,
    )

    direct = extract_2d_crop_views_from_cache(
        cache_data=cache_data,
        cache_primary_bounds=cache_primary_bounds,
        cache_spatial_bounds=cache_spatial_bounds,
        cache_time_bounds=cache_time_bounds,
        center_primary=primary_idx,
        spatial_start=crop_spatial_start,
        time_start=crop_time_start,
        crop_size=crop_size,
        views=views,
        traversal_direction="dir0",
        valid_primary_bounds=(0, cube.shape[0]),
        valid_spatial_bounds=(sp_phase_offset, sp_phase_offset + sp_axis_length),
        valid_time_bounds=(t_phase_offset, t_phase_offset + t_axis_length),
    )

    np.testing.assert_allclose(direct, legacy)
