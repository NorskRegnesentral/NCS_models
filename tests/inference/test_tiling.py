"""Tests for tiling logic."""

import numpy as np
import pytest

from NCS.inference.tiling import (
    compute_centered_tile_centers,
    compute_centered_tile_patch_overlap_weights,
    compute_densified_axis,
    compute_patch_overlap_weights,
    compute_tile_patch_overlap_weights,
    compute_tile_positions,
)


def _accumulate_axis_weights(weights: np.ndarray, patches_per_crop: int, num_overlap_patches: int) -> np.ndarray:
    stride_patches = patches_per_crop - num_overlap_patches
    n_patches_total = patches_per_crop + stride_patches * (weights.shape[0] - 1)
    accumulated = np.zeros(n_patches_total, dtype=np.float32)
    for tile_idx in range(weights.shape[0]):
        start = tile_idx * stride_patches
        accumulated[start : start + patches_per_crop] += weights[tile_idx]
    return accumulated


class TestComputeTilePositions:
    def test_single_tile_uses_symmetric_padding(self):
        """Axis smaller than crop_size -> one centered tile with split pad."""
        tp = compute_tile_positions(axis_length=100, crop_size=224, patch_size=16, num_overlap_patches=0)
        assert tp.starts == [0]
        assert tp.pad_before == 48
        assert tp.pad_after == 64
        assert tp.n_patches_total == 14

    def test_exact_fit_no_overlap(self):
        tp = compute_tile_positions(axis_length=448, crop_size=224, patch_size=16, num_overlap_patches=0)
        assert tp.starts == [0, 224]
        assert tp.pad_before == 0
        assert tp.pad_after == 0
        assert tp.n_patches_total == 28

    def test_overlap_basic_is_centered_in_patch_space(self):
        tp = compute_tile_positions(axis_length=300, crop_size=224, patch_size=16, num_overlap_patches=7)
        assert tp.starts == [0, 112]
        assert tp.pad_before == 16
        assert tp.pad_after == 16
        assert tp.n_patches_total == 21

    def test_no_overlap_can_still_pad_both_sides(self):
        tp = compute_tile_positions(axis_length=500, crop_size=224, patch_size=16, num_overlap_patches=0)
        assert tp.starts == [0, 224, 448]
        assert tp.pad_before == 80
        assert tp.pad_after == 80
        assert tp.n_patches_total == 42

    def test_can_disable_pad_before(self):
        tp = compute_tile_positions(
            axis_length=500, crop_size=224, patch_size=16, num_overlap_patches=0, pad_before=False
        )
        assert tp.starts == [0, 224, 448]
        assert tp.pad_before == 0
        assert tp.pad_after == 160
        assert tp.n_patches_total == 42

    def test_crop_not_divisible_by_patch(self):
        with pytest.raises(ValueError, match="divisible"):
            compute_tile_positions(axis_length=100, crop_size=100, patch_size=16, num_overlap_patches=0)

    def test_overlap_too_large(self):
        with pytest.raises(ValueError, match="must be <"):
            compute_tile_positions(axis_length=100, crop_size=224, patch_size=16, num_overlap_patches=14)


class TestComputePatchOverlapWeights:
    def test_no_overlap(self):
        w = compute_patch_overlap_weights(14, 0)
        assert w.shape == (14,)
        np.testing.assert_array_equal(w, np.ones(14, dtype=np.float32))

    def test_overlap_symmetry(self):
        w = compute_patch_overlap_weights(14, 7)
        assert w.shape == (14,)
        np.testing.assert_array_almost_equal(w, w[::-1])
        assert np.all(np.diff(w[:7]) > 0)

    def test_complementary_weights(self):
        """Tile-aware weights should sum to 1 over the covered axis."""
        patches_per_crop = 14
        num_overlap = 7
        weights = compute_tile_patch_overlap_weights(2, patches_per_crop, num_overlap)
        totals = np.zeros(patches_per_crop + (patches_per_crop - num_overlap), dtype=np.float32)
        stride_patches = patches_per_crop - num_overlap
        for tile_idx, tile_weights in enumerate(weights):
            start = tile_idx * stride_patches
            totals[start : start + patches_per_crop] += tile_weights
        np.testing.assert_allclose(totals, np.ones_like(totals))

    def test_peak_reaches_one(self):
        w = compute_patch_overlap_weights(14, 3)
        np.testing.assert_almost_equal(w.max(), 1.0)

    def test_small_overlap(self):
        w = compute_patch_overlap_weights(4, 1)
        assert w.shape == (4,)
        # Ramp: [0.5, 1.0, 1.0, 0.5]
        np.testing.assert_almost_equal(w[0], 0.5)
        np.testing.assert_almost_equal(w[3], 0.5)
        np.testing.assert_almost_equal(w[1], 1.0)
        np.testing.assert_almost_equal(w[2], 1.0)

    @pytest.mark.parametrize("overlap_filter", ["uniform", "ramp", "exponential", "center"])
    @pytest.mark.parametrize("num_overlap", [1, 3, 7])
    def test_tile_weights_sum_to_one_for_any_valid_overlap(self, overlap_filter, num_overlap):
        patches_per_crop = 14
        weights = compute_tile_patch_overlap_weights(6, patches_per_crop, num_overlap, overlap_filter=overlap_filter)
        totals = np.zeros(patches_per_crop + (6 - 1) * (patches_per_crop - num_overlap), dtype=np.float32)
        stride_patches = patches_per_crop - num_overlap
        for tile_idx, tile_weights in enumerate(weights):
            start = tile_idx * stride_patches
            totals[start : start + patches_per_crop] += tile_weights
        np.testing.assert_allclose(totals, np.ones_like(totals))

    def test_uniform_filter_is_all_ones(self):
        w = compute_patch_overlap_weights(14, 7, overlap_filter="uniform")
        np.testing.assert_array_equal(w, np.ones(14, dtype=np.float32))

    def test_ramp_filter_increases_toward_center(self):
        w = compute_patch_overlap_weights(14, 7, overlap_filter="ramp")
        assert w[0] < w[1] < w[2] < w[3]
        np.testing.assert_array_almost_equal(w, w[::-1])

    def test_ramp_filter_raw_profile_is_overlap_independent(self):
        np.testing.assert_allclose(
            compute_patch_overlap_weights(14, 7, overlap_filter="ramp"),
            compute_patch_overlap_weights(14, 13, overlap_filter="ramp"),
        )

    def test_exponential_filter_tapers_more_aggressively_than_ramp(self):
        ramp = compute_patch_overlap_weights(14, 7, overlap_filter="ramp")
        exponential = compute_patch_overlap_weights(14, 7, overlap_filter="exponential")
        assert exponential[0] < ramp[0]
        assert exponential[1] < ramp[1]
        np.testing.assert_array_almost_equal(exponential, exponential[::-1])

    def test_exponential_filter_raw_profile_is_overlap_independent(self):
        np.testing.assert_allclose(
            compute_patch_overlap_weights(14, 7, overlap_filter="exponential"),
            compute_patch_overlap_weights(14, 13, overlap_filter="exponential"),
        )

    def test_more_overlap_softens_center_tile_weights(self):
        low_overlap = compute_tile_patch_overlap_weights(6, 14, 7, overlap_filter="ramp")[3]
        high_overlap = compute_tile_patch_overlap_weights(6, 14, 13, overlap_filter="ramp")[3]
        assert high_overlap[0] < low_overlap[0]

    def test_center_filter_assigns_each_patch_to_one_tile(self):
        weights = compute_tile_patch_overlap_weights(6, 14, 13, overlap_filter="center")
        assert np.all((weights == 0.0) | (weights == 1.0))
        assert np.all(weights.sum(axis=0) >= 1.0)

    def test_center_filter_raw_profile_is_binary(self):
        w = compute_patch_overlap_weights(14, 13, overlap_filter="center")
        assert np.all((w == 0.0) | (w == 1.0))

    def test_invalid_overlap_filter(self):
        with pytest.raises(ValueError, match="Unsupported overlap_filter"):
            compute_patch_overlap_weights(14, 7, overlap_filter="triangle")

    def test_centered_tile_centers_match_sampling_pattern(self):
        assert compute_centered_tile_centers(
            total_patches=59, patches_per_crop=14, num_overlap_patches=0, pad_before=False
        ) == [
            0,
            14,
            28,
            42,
            56,
        ]
        assert compute_centered_tile_centers(
            total_patches=59, patches_per_crop=14, num_overlap_patches=4, pad_before=False
        ) == [
            0,
            10,
            20,
            30,
            40,
            50,
            58,
        ]
        assert compute_centered_tile_centers(
            total_patches=59, patches_per_crop=14, num_overlap_patches=7, pad_before=False
        ) == [
            0,
            7,
            14,
            21,
            28,
            35,
            42,
            49,
            56,
        ]

    def test_centered_tile_centers_can_use_symmetric_pad_before(self):
        assert compute_centered_tile_centers(
            total_patches=59, patches_per_crop=14, num_overlap_patches=0, pad_before=True
        ) == [2, 16, 30, 44, 58]
        assert compute_centered_tile_centers(
            total_patches=59, patches_per_crop=14, num_overlap_patches=4, pad_before=True
        ) == [5, 15, 25, 35, 45, 55]

    @pytest.mark.parametrize("pad_before", [False, True])
    def test_centered_tile_weights_sum_to_one_over_output_axis(self, pad_before):
        centers = compute_centered_tile_centers(
            total_patches=59,
            patches_per_crop=14,
            num_overlap_patches=7,
            pad_before=pad_before,
        )
        weights = compute_centered_tile_patch_overlap_weights(
            centers,
            patches_per_crop=14,
            total_patches=59,
            overlap_filter="gaussian",
        )
        totals = np.zeros(59, dtype=np.float32)
        for tile_idx, center in enumerate(centers):
            start = center - 7
            for local_idx, weight in enumerate(weights[tile_idx]):
                patch_idx = start + local_idx
                if 0 <= patch_idx < 59:
                    totals[patch_idx] += weight
        np.testing.assert_allclose(totals, np.ones_like(totals))


class TestComputeDensifiedAxis:
    def test_default_matches_patch_grid(self):
        axis = compute_densified_axis(axis_length=100, patch_size=16, densify=1)

        assert axis.step_samples == 16
        assert axis.phase_offsets == [0]
        assert axis.phase_axis_lengths == [100]
        assert axis.phase_counts == [7]
        np.testing.assert_array_equal(axis.coords, np.array([8, 24, 40, 56, 72, 88, 104], dtype=np.int32))

    def test_full_densify_matches_input_size(self):
        axis = compute_densified_axis(axis_length=19, patch_size=16, densify=16)

        assert axis.step_samples == 1
        assert axis.size == 19
        np.testing.assert_array_equal(axis.coords, np.arange(8, 27, dtype=np.int32))

    def test_partial_densify_interleaves_phases(self):
        axis = compute_densified_axis(axis_length=100, patch_size=16, densify=4)

        assert axis.step_samples == 4
        assert axis.phase_offsets == [0, 4, 8, 12]
        assert axis.phase_counts == [7, 6, 6, 6]
        np.testing.assert_array_equal(axis.coords[:8], np.array([8, 12, 16, 20, 24, 28, 32, 36], dtype=np.int32))

    def test_invalid_densify_choice(self):
        with pytest.raises(ValueError, match="must be one of"):
            compute_densified_axis(axis_length=100, patch_size=16, densify=3)

    def test_patch_size_must_be_divisible_by_densify(self):
        with pytest.raises(ValueError, match="divisible by densify"):
            compute_densified_axis(axis_length=100, patch_size=12, densify=16)
