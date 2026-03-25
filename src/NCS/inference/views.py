"""2.5D multi-view extraction from a seismic subvolume."""

import numpy as np


def _resize_spatial_axis(crop_2d: np.ndarray, target_size: int) -> np.ndarray:
    """Resize the spatial axis with linear interpolation."""
    if crop_2d.shape[0] == target_size:
        return crop_2d.astype(np.float32, copy=False)

    src = np.arange(crop_2d.shape[0], dtype=np.float32)
    dst = np.linspace(0, crop_2d.shape[0] - 1, num=target_size, dtype=np.float32)
    resized = np.stack([np.interp(dst, src, crop_2d[:, col]) for col in range(crop_2d.shape[1])], axis=1)
    return resized.astype(np.float32, copy=False)


def _apply_diagonal_crop(crop_2d: np.ndarray) -> np.ndarray:
    """Diagonal handling by center-cropping to crop_size / sqrt(2) and resizing back."""
    target_size = crop_2d.shape[0]
    diagonal_size = max(1, int(target_size / np.sqrt(2)))
    if diagonal_size >= target_size:
        return crop_2d.astype(np.float32, copy=False)

    start = (target_size - diagonal_size) // 2
    stop = start + diagonal_size
    return _resize_spatial_axis(crop_2d[start:stop], target_size)


def extract_2d_crop_views_from_cache(
    cache_data: np.ndarray,
    cache_primary_bounds: tuple[int, int],
    cache_spatial_bounds: tuple[int, int],
    cache_time_bounds: tuple[int, int],
    center_primary: int,
    spatial_start: int,
    time_start: int,
    crop_size: int,
    views: list[str],
    traversal_direction: str = "dir0",
    valid_primary_bounds: tuple[int, int] | None = None,
    valid_spatial_bounds: tuple[int, int] | None = None,
    valid_time_bounds: tuple[int, int] | None = None,
) -> np.ndarray:
    """Extract 2.5D views directly from a cached subcube with zero-padding semantics.

    The cache may contain more data than the logical phase window. Values requested outside
    ``valid_*_bounds`` are treated as zeros even if they are present in ``cache_data``.
    """

    valid_primary_bounds = valid_primary_bounds or cache_primary_bounds
    valid_spatial_bounds = valid_spatial_bounds or cache_spatial_bounds
    valid_time_bounds = valid_time_bounds or cache_time_bounds

    p0, p1 = cache_primary_bounds
    s0, s1 = cache_spatial_bounds
    t0, t1 = cache_time_bounds
    half = crop_size // 2
    primary_start = center_primary - half
    primary_indices = primary_start + np.arange(crop_size)
    spatial_end = spatial_start + crop_size
    time_end = time_start + crop_size
    time_lo = max(time_start, valid_time_bounds[0], t0)
    time_hi = min(time_end, valid_time_bounds[1], t1)

    result = np.zeros((len(views), crop_size, crop_size), dtype=np.float32)

    center_primary_valid = (
        valid_primary_bounds[0] <= center_primary < valid_primary_bounds[1] and p0 <= center_primary < p1
    )
    primary_mask = (
        (valid_primary_bounds[0] <= primary_indices)
        & (primary_indices < valid_primary_bounds[1])
        & (p0 <= primary_indices)
        & (primary_indices < p1)
    )
    dst_time_start = time_lo - time_start
    dst_time_end = time_hi - time_start

    for v_idx, view in enumerate(views):
        if time_lo >= time_hi:
            continue

        if view == traversal_direction:
            spatial_lo = max(spatial_start, valid_spatial_bounds[0], s0)
            spatial_hi = min(spatial_end, valid_spatial_bounds[1], s1)
            if center_primary_valid and spatial_lo < spatial_hi:
                dst_spatial_start = spatial_lo - spatial_start
                dst_spatial_end = spatial_hi - spatial_start
                result[v_idx, dst_spatial_start:dst_spatial_end, dst_time_start:dst_time_end] = cache_data[
                    center_primary - p0,
                    spatial_lo - s0 : spatial_hi - s0,
                    time_lo - t0 : time_hi - t0,
                ]
            continue

        if view in ("dir0", "dir90") and view != traversal_direction:
            spatial_center = spatial_start + half
            spatial_valid = (
                valid_spatial_bounds[0] <= spatial_center < valid_spatial_bounds[1] and s0 <= spatial_center < s1
            )
            if spatial_valid and np.any(primary_mask):
                dst_rows = np.flatnonzero(primary_mask)
                local_primary = primary_indices[primary_mask] - p0
                result[v_idx, dst_rows, dst_time_start:dst_time_end] = cache_data[
                    local_primary,
                    spatial_center - s0,
                    time_lo - t0 : time_hi - t0,
                ]
            continue

        if view == "dir45":
            spatial_indices = spatial_start + np.arange(crop_size)
        elif view == "dir135":
            spatial_indices = spatial_end - 1 - np.arange(crop_size)
        else:
            raise ValueError(f"Unsupported view: {view}")

        spatial_mask = (
            (valid_spatial_bounds[0] <= spatial_indices)
            & (spatial_indices < valid_spatial_bounds[1])
            & (s0 <= spatial_indices)
            & (spatial_indices < s1)
        )
        valid_mask = primary_mask & spatial_mask
        if not np.any(valid_mask):
            continue
        dst_rows = np.flatnonzero(valid_mask)
        local_primary = primary_indices[valid_mask] - p0
        local_spatial = spatial_indices[valid_mask] - s0
        result[v_idx, dst_rows, dst_time_start:dst_time_end] = cache_data[
            local_primary,
            local_spatial,
            time_lo - t0 : time_hi - t0,
        ]
        result[v_idx] = _apply_diagonal_crop(result[v_idx])

    return result


def extract_2d_crop_views(
    subvolume: np.ndarray,
    center_idx: int,
    spatial_start: int,
    time_start: int,
    crop_size: int,
    views: list[str],
    traversal_direction: str = "dir0",
    circular_primary: bool = False,
) -> np.ndarray:
    """Extract multiple 2D views from a subvolume for 2.5D inference.

    The subvolume contains adjacent slices along the *primary* traversal axis.
    Each view extracts a 2D crop of shape ``(crop_size, crop_size)`` in
    ``(time, spatial)`` order.

    Args:
        subvolume: Array of shape ``(n_slices, n_spatial, n_samples)`` where
            ``n_slices >= crop_size`` and the center slice is at index ``center_idx``.
        center_idx: Index into ``subvolume`` axis-0 of the current primary slice.
        spatial_start: 0-based start position along the spatial axis.
        time_start: 0-based start position along the time (samples) axis.
        crop_size: Size of each 2D crop (square).
        views: List of view names, e.g. ``["dir0", "dir90", "dir45", "dir135"]``.
        traversal_direction: The primary traversal direction (``"dir0"`` or ``"dir90"``).
            When ``"dir0"``, subvolume axis-0 contains inlines.
            When ``"dir90"``, subvolume axis-0 contains xlines.
        circular_primary: Whether the primary axis should be treated as a ring buffer.

    Returns:
        Array of shape ``(len(views), crop_size, crop_size)`` in seismic
        ``(view, spatial, time)`` order, float32.  Note that
        ``SeismicProcessor.preprocess`` subsequently applies
        ``seismic_to_torch_2d`` (``np.moveaxis(data, -1, -2)``) to convert to
        PyTorch ``(view, time, spatial)`` order before passing to the model.
    """
    half = crop_size // 2
    sp_end = spatial_start + crop_size
    t_end = time_start + crop_size

    result = np.empty((len(views), crop_size, crop_size), dtype=np.float32)
    primary_size = subvolume.shape[0]
    primary_offsets = np.arange(crop_size)
    time_slice = slice(time_start, t_end)

    def _primary_index(idx: int | np.ndarray) -> int | np.ndarray:
        if circular_primary:
            return idx % primary_size
        return idx

    for v_idx, view in enumerate(views):
        crop_2d = np.empty((crop_size, crop_size), dtype=np.float32)

        # Determine extraction type based on view relative to traversal direction
        if view == traversal_direction:
            # View matches traversal: extract single slice at center_idx (primary slice)
            # subvolume[center_idx, spatial, time] -> (spatial, time)
            crop_2d[:] = subvolume[_primary_index(center_idx), spatial_start:sp_end, time_start:t_end]
            result[v_idx] = crop_2d

        elif view in ("dir0", "dir90") and view != traversal_direction:
            # Orthogonal view: extract perpendicular to axis-0
            # For each primary slice, take the center spatial position
            sp_center = spatial_start + half
            start_primary = center_idx - half
            primary_indices = _primary_index(start_primary + primary_offsets)
            crop_2d[:] = subvolume[primary_indices, sp_center, time_slice]
            # crop_2d is (primary, time)
            result[v_idx] = crop_2d

        elif view == "dir45":
            # Diagonal: step +1 along both primary and spatial
            start_primary = center_idx - half
            primary_indices = _primary_index(start_primary + primary_offsets)
            spatial_indices = spatial_start + primary_offsets
            crop_2d[:] = subvolume[primary_indices, spatial_indices, time_slice]
            result[v_idx] = _apply_diagonal_crop(crop_2d)

        elif view == "dir135":
            # Reverse diagonal: step +1 along primary, -1 along spatial
            start_primary = center_idx - half
            primary_indices = _primary_index(start_primary + primary_offsets)
            spatial_indices = sp_end - 1 - primary_offsets
            crop_2d[:] = subvolume[primary_indices, spatial_indices, time_slice]
            result[v_idx] = _apply_diagonal_crop(crop_2d)

        else:
            raise ValueError(f"Unsupported view: {view}")

    return result
