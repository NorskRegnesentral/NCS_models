"""Tiling utilities for patch-based inference with overlap."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


OverlapFilter = Literal["exponential", "gaussian", "ramp", "uniform", "center"]
OVERLAP_FILTER_CHOICES: tuple[OverlapFilter, ...] = ("exponential", "gaussian", "ramp", "uniform", "center")
DENSIFY_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16)


@dataclass
class DensifyAxis:
    """Dense output-grid metadata for one axis."""

    step_samples: int
    phase_offsets: list[int]
    phase_axis_lengths: list[int]
    phase_counts: list[int]
    size: int
    coords: np.ndarray


@dataclass
class TilePositions:
    """Tile start positions and derived quantities for one axis.

    Attributes:
        starts: 0-based start positions of each tile in the padded axis (in samples).
        n_patches_total: Total number of padded patch positions covered by the tiles.
        pad_before: Number of samples padded before the axis.
        pad_after: Number of samples padded after the axis.
    """

    starts: list[int]
    n_patches_total: int
    pad_before: int
    pad_after: int


def compute_tile_positions(
    axis_length: int,
    crop_size: int,
    patch_size: int,
    num_overlap_patches: int = 0,
    pad_before: bool = True,
) -> TilePositions:
    """Compute tile start positions along one axis.

    Tiles are placed in patch space so that each consecutive tile overlaps by
    ``num_overlap_patches`` patches with the previous tile. Any extra tiled
    extent is split between the start and end of the axis in patch-sized
    chunks so the coverage is centered instead of only padded at the end.

    Args:
        axis_length: Length of the axis in samples.
        crop_size: Size of each tile (crop) in samples. Must be divisible by ``patch_size``.
        patch_size: Size of each patch in samples.
        num_overlap_patches: Number of patches that overlap between adjacent tiles.
        pad_before: Whether to split extra padding before the axis. When ``False``,
            the tiled extent is anchored at the start of the axis and all extra
            padding is applied at the end.

    Returns:
        A :class:`TilePositions` with padded start positions, total patch count,
        and symmetric padding.
    """
    if crop_size % patch_size != 0:
        raise ValueError(f"crop_size ({crop_size}) must be divisible by patch_size ({patch_size})")

    patches_per_crop = crop_size // patch_size
    if num_overlap_patches >= patches_per_crop:
        raise ValueError(
            f"num_overlap_patches ({num_overlap_patches}) must be < patches_per_crop ({patches_per_crop})"
        )

    stride_patches = patches_per_crop - num_overlap_patches
    actual_patches = math.ceil(axis_length / patch_size)
    extra_patch_starts = max(actual_patches - patches_per_crop, 0)
    num_tiles = max(1, math.ceil(extra_patch_starts / stride_patches) + 1)
    n_patches_total = patches_per_crop + stride_patches * (num_tiles - 1)

    total_pad_patches = n_patches_total - actual_patches
    if pad_before:
        pad_before_samples = (total_pad_patches // 2) * patch_size
        pad_after = (total_pad_patches - total_pad_patches // 2) * patch_size
    else:
        pad_before_samples = 0
        pad_after = total_pad_patches * patch_size
    starts = [tile_idx * stride_patches * patch_size for tile_idx in range(num_tiles)]

    return TilePositions(
        starts=starts,
        n_patches_total=n_patches_total,
        pad_before=pad_before_samples,
        pad_after=pad_after,
    )


def compute_densified_axis(axis_length: int, patch_size: int, densify: int = 1) -> DensifyAxis:
    """Compute dense output-grid metadata for one axis."""
    if densify not in DENSIFY_CHOICES:
        raise ValueError(f"densify ({densify}) must be one of {DENSIFY_CHOICES}")
    if patch_size % densify != 0:
        raise ValueError(f"patch_size ({patch_size}) must be divisible by densify ({densify})")

    step_samples = patch_size // densify
    phase_offsets = [phase_idx * step_samples for phase_idx in range(densify)]
    phase_axis_lengths = [max(axis_length - phase_offset, 0) for phase_offset in phase_offsets]
    phase_counts = [
        0 if phase_length == 0 else math.ceil(phase_length / patch_size) for phase_length in phase_axis_lengths
    ]
    size = sum(phase_counts)

    coords = np.empty(size, dtype=np.int32)
    for phase_idx, (phase_offset, phase_count) in enumerate(zip(phase_offsets, phase_counts, strict=True)):
        if phase_count == 0:
            continue
        coords[phase_idx : phase_idx + densify * phase_count : densify] = (
            phase_offset + patch_size // 2 + np.arange(phase_count, dtype=np.int32) * patch_size
        )

    return DensifyAxis(
        step_samples=step_samples,
        phase_offsets=phase_offsets,
        phase_axis_lengths=phase_axis_lengths,
        phase_counts=phase_counts,
        size=size,
        coords=coords,
    )


def _validate_overlap_filter(overlap_filter: str) -> None:
    if overlap_filter not in OVERLAP_FILTER_CHOICES:
        raise ValueError(
            f"Unsupported overlap_filter {overlap_filter!r}. Expected one of {', '.join(OVERLAP_FILTER_CHOICES)}."
        )


def _compute_raw_patch_profile(patches_per_crop: int, overlap_filter: OverlapFilter) -> np.ndarray:
    if overlap_filter == "uniform":
        return np.ones(patches_per_crop, dtype=np.float32)

    center_span = max(math.ceil(patches_per_crop / 2), 1)
    edge_distance = np.minimum(
        np.arange(1, patches_per_crop + 1, dtype=np.float32),
        np.arange(patches_per_crop, 0, -1, dtype=np.float32),
    )
    edge_distance /= center_span

    if overlap_filter == "ramp":
        return edge_distance.astype(np.float32)

    if overlap_filter == "exponential":
        return ((np.exp(4.0 * edge_distance) - 1.0) / (np.exp(4.0) - 1.0)).astype(np.float32)

    if overlap_filter == "gaussian":
        positions = np.arange(patches_per_crop, dtype=np.float32)
        center = (patches_per_crop - 1) / 2
        distances = np.abs(positions - center)
        distances = distances / np.max(distances) if np.max(distances) > 0 else distances
        return (np.exp(-3.0 * distances**2)).astype(np.float32)

    raise ValueError(f"Raw profile is not defined for overlap_filter {overlap_filter!r}")


def compute_patch_overlap_weights(
    patches_per_crop: int,
    num_overlap_patches: int,
    overlap_filter: OverlapFilter = "ramp",
) -> np.ndarray:
    """Compute the raw per-tile overlap profile for inspection and plotting."""
    _validate_overlap_filter(overlap_filter)

    if num_overlap_patches == 0:
        return np.ones(patches_per_crop, dtype=np.float32)

    if overlap_filter != "center":
        return _compute_raw_patch_profile(patches_per_crop, overlap_filter)

    stride_patches = patches_per_crop - num_overlap_patches
    extra_tiles = math.ceil((patches_per_crop - 1) / stride_patches)
    num_tiles = 2 * extra_tiles + 1
    return compute_tile_patch_overlap_weights(num_tiles, patches_per_crop, num_overlap_patches, overlap_filter)[
        extra_tiles
    ]


def compute_centered_tile_centers(
    total_patches: int,
    patches_per_crop: int,
    num_overlap_patches: int,
    pad_before: bool = False,
) -> list[int]:
    """Compute crop-center patch indices.

    Centers are placed every ``stride_patches`` and an extra final center is added
    when the remaining uncovered tail is at least half a crop.

    When ``pad_before=True``, the centers instead follow a symmetrically padded
    tiled extent in patch space, analogous to :func:`compute_tile_positions`.
    """
    if total_patches <= 0:
        return []
    if num_overlap_patches >= patches_per_crop:
        raise ValueError(
            f"num_overlap_patches ({num_overlap_patches}) must be < patches_per_crop ({patches_per_crop})"
        )

    if pad_before:
        tile_positions = compute_tile_positions(
            axis_length=total_patches,
            crop_size=patches_per_crop,
            patch_size=1,
            num_overlap_patches=num_overlap_patches,
            pad_before=True,
        )
        half_crop = patches_per_crop // 2
        return [start - tile_positions.pad_before + half_crop for start in tile_positions.starts]

    stride_patches = patches_per_crop - num_overlap_patches
    centers = list(range(0, total_patches, stride_patches))
    if not centers:
        centers = [0]

    last_patch = total_patches - 1
    if centers[-1] != last_patch and last_patch - centers[-1] >= patches_per_crop // 2:
        centers.append(last_patch)

    return centers


def compute_tile_patch_overlap_weights(
    num_tiles: int,
    patches_per_crop: int,
    num_overlap_patches: int,
    overlap_filter: OverlapFilter = "ramp",
) -> np.ndarray:
    """Compute normalized per-tile blending weights along one axis."""
    _validate_overlap_filter(overlap_filter)

    if num_tiles <= 0:
        raise ValueError(f"num_tiles ({num_tiles}) must be positive")

    if num_overlap_patches == 0:
        return np.ones((num_tiles, patches_per_crop), dtype=np.float32)

    stride_patches = patches_per_crop - num_overlap_patches
    starts = np.arange(num_tiles, dtype=np.int32) * stride_patches
    n_patches_total = patches_per_crop + stride_patches * (num_tiles - 1)

    weights = np.zeros((num_tiles, patches_per_crop), dtype=np.float32)
    centers = starts.astype(np.float32) + (patches_per_crop - 1) / 2
    raw_profile = None if overlap_filter == "center" else _compute_raw_patch_profile(patches_per_crop, overlap_filter)

    for patch_idx in range(n_patches_total):
        covering_tiles = np.flatnonzero((starts <= patch_idx) & (patch_idx < starts + patches_per_crop))

        if overlap_filter == "uniform":
            scores = np.ones(covering_tiles.shape[0], dtype=np.float32)
        elif overlap_filter == "center":
            scores = np.zeros(covering_tiles.shape[0], dtype=np.float32)
            chosen_tile = np.argmin(np.abs(centers[covering_tiles] - patch_idx))
            scores[chosen_tile] = 1.0
        else:
            assert raw_profile is not None
            scores = raw_profile[patch_idx - starts[covering_tiles]]

        scores /= scores.sum()
        for local_tile_idx, tile_idx in enumerate(covering_tiles):
            weights[tile_idx, patch_idx - starts[tile_idx]] = scores[local_tile_idx]

    return weights


def compute_centered_tile_patch_overlap_weights(
    tile_centers: list[int],
    patches_per_crop: int,
    total_patches: int,
    overlap_filter: OverlapFilter = "ramp",
) -> np.ndarray:
    """Compute normalized overlap weights for center-positioned tiles.

    Unlike ``compute_tile_patch_overlap_weights``, the tile locations are specified
    by crop-center patch indices.
    """
    _validate_overlap_filter(overlap_filter)

    num_tiles = len(tile_centers)
    if num_tiles <= 0:
        raise ValueError("tile_centers must be non-empty")
    if total_patches <= 0:
        raise ValueError("total_patches must be positive")

    half_crop = patches_per_crop // 2
    starts = np.asarray(tile_centers, dtype=np.int32) - half_crop
    weights = np.zeros((num_tiles, patches_per_crop), dtype=np.float32)
    centers = starts.astype(np.float32) + (patches_per_crop - 1) / 2
    raw_profile = None if overlap_filter == "center" else _compute_raw_patch_profile(patches_per_crop, overlap_filter)

    for patch_idx in range(total_patches):
        covering_tiles = np.flatnonzero((starts <= patch_idx) & (patch_idx < starts + patches_per_crop))
        if covering_tiles.size == 0:
            continue

        if overlap_filter == "uniform":
            scores = np.ones(covering_tiles.shape[0], dtype=np.float32)
        elif overlap_filter == "center":
            scores = np.zeros(covering_tiles.shape[0], dtype=np.float32)
            chosen_tile = np.argmin(np.abs(centers[covering_tiles] - patch_idx))
            scores[chosen_tile] = 1.0
        else:
            assert raw_profile is not None
            scores = raw_profile[patch_idx - starts[covering_tiles]]

        scores /= scores.sum()
        for local_tile_idx, tile_idx in enumerate(covering_tiles):
            weights[tile_idx, patch_idx - starts[tile_idx]] = scores[local_tile_idx]

    return weights
