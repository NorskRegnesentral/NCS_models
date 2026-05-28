"""End-to-end inference pipeline for seismic foundation models.

Reads a seismic volume (SEG-Y / SGZ), runs a ViT model (2D, 2.5D, or 3D)
over tiled crops, and writes a patch-space feature cube via xarray + dask.
"""

import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import torch
import xarray as xr
import zarr
from tqdm import tqdm
from transformers import AutoConfig, AutoModel
from transformers.models.vit.modeling_vit import ViTModel

import NCS.models  # noqa: F401  — registers custom model types with AutoConfig/AutoModel
from NCS.inference.async_writer import AsyncWriter, write_array_slice
from NCS.inference.dataloading import (
    ThreeDPhaseDataset,
    TwoDPositionDataset,
    TwoPointFiveDPipelineDataset,
    build_inference_loader,
    build_processor_config,
    clamp_max_preload,
    clamp_prefetch_factor,
    resolve_loader_settings,
)
from NCS.inference.reader import compute_cube_stats, cube_shape, open_seismic
from NCS.inference.tiling import (
    OverlapFilter,
    TilePositions,
    compute_centered_tile_centers,
    compute_centered_tile_patch_overlap_weights,
    compute_densified_axis,
    compute_tile_patch_overlap_weights,
    compute_tile_positions,
)
from NCS.models.vit3d import ViT3DModel
from NCS.models.vit25d import ViT25DModel
from NCS.processors import SeismicProcessor
from NCS.processors.seismic_processor import NormalizationMode


logger = logging.getLogger(__name__)
_VOLUME_INDEX_CACHE: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


@dataclass(frozen=True)
class AxisPhaseConfig:
    phase_idx: int
    phase_offset: int
    axis_length: int
    count: int
    tile: TilePositions
    weights: np.ndarray
    patch_offset: int
    tail_pad: int


def _axis_tail_pad(axis_length: int, patch_size: int) -> int:
    return math.ceil(axis_length / patch_size) * patch_size - axis_length


def _phase_slice(phase_idx: int, phase_count: int, densify: int) -> slice:
    return slice(phase_idx, phase_idx + densify * phase_count, densify)


def _phase_coords(coords: np.ndarray, phase_idx: int, densify: int) -> np.ndarray:
    return coords[phase_idx::densify]


def _resolve_direction_axes(direction: str, n_il: int, n_xl: int) -> tuple[int, int]:
    if direction == "dir0":
        return n_il, n_xl
    if direction == "dir90":
        return n_xl, n_il
    raise ValueError(f"Unsupported direction: {direction}")


def _build_centered_axis_phase_configs(
    dense_axis,
    patch_size: int,
    num_overlap_patches: int,
    overlap_filter: OverlapFilter,
    patches_per_crop: int,
    pad_before: bool,
) -> list[AxisPhaseConfig]:
    configs: list[AxisPhaseConfig] = []
    for phase_idx, (phase_offset, axis_length, count) in enumerate(
        zip(dense_axis.phase_offsets, dense_axis.phase_axis_lengths, dense_axis.phase_counts, strict=True)
    ):
        if count == 0:
            continue
        tile_centers = compute_centered_tile_centers(
            count,
            patches_per_crop,
            num_overlap_patches,
            pad_before=pad_before,
        )
        configs.append(
            AxisPhaseConfig(
                phase_idx=phase_idx,
                phase_offset=phase_offset,
                axis_length=axis_length,
                count=count,
                tile=TilePositions(
                    starts=[center * patch_size for center in tile_centers],
                    n_patches_total=count,
                    pad_before=0,
                    pad_after=0,
                ),
                weights=compute_centered_tile_patch_overlap_weights(
                    tile_centers,
                    patches_per_crop,
                    count,
                    overlap_filter,
                ),
                patch_offset=0,
                tail_pad=0,
            )
        )
    return configs


def _nonzero_phase_counts(axis) -> list[int]:
    return [count for count in axis.phase_counts if count > 0]


def _iter_tensor_batches(pixel_values: torch.Tensor, meta: list[tuple], batch_size: int):
    for start in range(0, pixel_values.shape[0], batch_size):
        end = start + batch_size
        yield pixel_values[start:end], meta[start:end]


def _iter_model_batches(item: dict[str, Any], batch_size: int):
    batch_splits = item.get("batch_splits")
    if batch_splits is not None:
        pixel_values: torch.Tensor = item["pixel_values"]
        meta: list[tuple] = item["meta"]
        start = 0
        for split in batch_splits:
            end = start + split
            yield pixel_values[start:end], meta[start:end]
            start = end
        return
    yield from _iter_tensor_batches(item["pixel_values"], item["meta"], batch_size)


def _centered_tile_slices(
    tile_center_patch: int,
    patches_per_crop: int,
    total_patches: int,
) -> tuple[slice, slice] | None:
    """Map a centered crop's local patch grid onto the dense output patch grid.

    We sample 2D/2.5D crops by center position, so the first crop is centered at
    patch coordinate 0 and extends half a crop outside the valid patch grid. The
    corresponding token block must therefore be clipped against the output bounds.
    """
    half_crop = patches_per_crop // 2
    start = tile_center_patch - half_crop
    end = start + patches_per_crop
    dst_start = max(start, 0)
    dst_end = min(end, total_patches)
    if dst_start >= dst_end:
        return None
    src_start = dst_start - start
    src_end = src_start + (dst_end - dst_start)
    return slice(dst_start, dst_end), slice(src_start, src_end)


def _get_volume_index_tensors(
    device: str,
    input_shape: int,
    patch_size: int,
    patches_per_crop: int,
    batch_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (str(device), input_shape, patch_size, batch_len)
    cached = _VOLUME_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    centers = torch.arange(start=0, end=input_shape, step=patch_size, device=device) + patch_size // 2
    volume_indices = (
        torch.cartesian_prod(centers, centers, centers)
        .reshape(patches_per_crop, patches_per_crop, patches_per_crop, 3)
        .expand(batch_len, 1, -1, -1, -1, -1)
    )
    volume_size = torch.full((3,), input_shape, device=device, dtype=torch.long)
    _VOLUME_INDEX_CACHE[cache_key] = (volume_indices, volume_size)
    return volume_indices, volume_size


def _flush_3d_tile_accumulation(
    feature_sum,
    weight_sum,
    tile_feature_sum: np.ndarray,
    tile_weight_sum: np.ndarray,
    il_tile_idx: int,
    patches_per_crop: int,
    stride_patches: int,
) -> None:
    il_ps = il_tile_idx * stride_patches
    il_slice = slice(il_ps, il_ps + patches_per_crop)
    feat_chunk = np.array(feature_sum[il_slice, :, :, :])
    feat_chunk += tile_feature_sum
    feature_sum[il_slice, :, :, :] = feat_chunk

    w_chunk = np.array(weight_sum[il_slice, :, :])
    w_chunk += tile_weight_sum
    weight_sum[il_slice, :, :] = w_chunk


def _write_single_sliced_position(
    dest, src, primary_index: int, sp_phase_idx: int, t_phase_idx: int, densify: int
) -> None:
    dest[
        primary_index,
        _phase_slice(sp_phase_idx, src.shape[0], densify),
        _phase_slice(t_phase_idx, src.shape[1], densify),
    ] = src


def _submit_single_sliced_position(
    writer: AsyncWriter,
    dest,
    src,
    primary_index: int,
    sp_phase_idx: int,
    t_phase_idx: int,
    densify: int,
) -> None:
    writer.submit(_write_single_sliced_position, dest, src, primary_index, sp_phase_idx, t_phase_idx, densify)


def _submit_single_phase_position(writer: AsyncWriter, dest, src, primary_index: int) -> None:
    writer.submit(write_array_slice, dest, (primary_index, slice(None), slice(None), slice(None)), src)


def _load_model(
    model_path: str | Path, config, base_model_type: str, torch_dtype: torch.dtype
) -> ViTModel | ViT25DModel | ViT3DModel:
    """Load a pretrained model, if ViTMAEForPretraining type, get encoder."""

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if base_model_type.endswith("_mae"):
        encoder_type = base_model_type.removesuffix("_mae")
        logger.info(
            "MAE model detected (model_type=%s). Trying to loading encoder %s equivalent.",
            base_model_type,
            encoder_type,
        )
        # Build a base-class config so AutoModel dispatches to the non-MAE model.
        # Remove model_type from the dict so the class attribute (e.g. "vit") is used.
        base_config_cls = CONFIG_MAPPING[encoder_type]
        config_dict = config.to_dict()
        config_dict.pop("model_type", None)
        load_config = base_config_cls.from_dict(config_dict)
    else:
        load_config = config

    if base_model_type in ("vit3d_mae", "vit3d"):
        model = ViT3DModel.from_pretrained(
            model_path, config=load_config, torch_dtype=torch_dtype, ignore_mismatched_sizes=True
        )
    elif base_model_type in ("vit25d_mae", "vit25d"):
        model = ViT25DModel.from_pretrained(
            model_path, config=load_config, torch_dtype=torch_dtype, ignore_mismatched_sizes=True
        )
    else:
        model: ViTModel | ViT25DModel | ViT3DModel = AutoModel.from_pretrained(
            model_path, config=load_config, torch_dtype=torch_dtype
        )

    return model


def run_inference(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    direction: str = "dir0",
    input_views: list[str] | None = None,
    crop_size: int = 224,
    patch_size: int = 16,
    num_overlap_patches: int = 7,
    overlap_filter: OverlapFilter = "ramp",
    densify: int = 1,
    batch_size: int = 8,
    device: str = "cuda",
    dtype: str = "float32",
    clip_sigma: float = 3.0,
    do_normalize: bool = True,
    normalization_mode: NormalizationMode = "std",
    n_stat_traces: int | None = None,
    num_workers: int | None = None,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    max_preload: int = 500,
    pad_before: bool = True,
) -> xr.Dataset:
    """Run inference on a seismic volume and save the feature cube.

    Args:
        model_path: Path to the pretrained model directory (HuggingFace-style).
        input_path: Path to the input seismic file (.segy / .sgz).
        output_path: Path for the output file (.zarr or .nc).
        direction: Primary traversal direction — ``"dir0"`` (inline) or ``"dir90"`` (crossline).
        input_views: Views to use for 2.5D models (e.g. ``["dir0", "dir90"]``).
            Defaults to the model config's ``directions`` list.
        crop_size: Crop size in samples.
        patch_size: Patch size in samples.
        num_overlap_patches: Number of overlapping patches between adjacent tiles.
        overlap_filter: Overlap weighting mode for patch blending.
        densify: Output densification factor. ``1`` keeps patch-grid output;
            ``16`` with the default patch size yields one output per input sample.
        batch_size: Batch size for model forward passes.
        device: PyTorch device string.
        dtype: PyTorch dtype string (``"float32"``, ``"float16"``, ``"bfloat16"``).
        clip_sigma: Sigma clipping value after normalization.
        do_normalize: Whether to normalize the input.
        normalization_mode: ``"std"`` divides by the cube standard deviation only.
            ``"zscore"`` subtracts the cube mean first.
        n_stat_traces: Number of traces to sample for computing cube statistics. Defaults to 10% of total traces.
        num_workers: Torch DataLoader worker count for asynchronous input preparation.
        prefetch_factor: Number of prefetched work units per DataLoader worker.
        pin_memory: Whether DataLoader batches should use pinned host memory.
        max_preload: Worker-local preload cache edge cap in samples.
        pad_before: Whether to allow leading tile padding. Disable to anchor tile
            coverage at the start of each axis and place all extra padding after it.

    Returns:
        The output ``xr.Dataset``.
    """
    torch_dtype = getattr(torch, dtype)
    num_workers, pin_memory = resolve_loader_settings(device, num_workers, pin_memory)
    prefetch_factor = clamp_prefetch_factor(prefetch_factor)
    max_preload = clamp_max_preload(max_preload)
    non_blocking = pin_memory and str(device).startswith("cuda")

    # 1. Load model and detect type
    config = AutoConfig.from_pretrained(model_path)
    base_model_type = config.model_type  # "vit3d", "vit25d", "vit" or their "_mae" variants
    model_type = base_model_type.removesuffix("_mae")

    model = _load_model(model_path, config, base_model_type, torch_dtype)
    model = model.to(device).eval()  # ty:ignore[invalid-argument-type]
    hidden_size = config.hidden_size

    logger.info(
        "Loaded model type=%s (%s) hidden_size=%d from %s", model_type, str(type(model)), hidden_size, model_path
    )

    # 2. Open seismic file and compute stats
    f = open_seismic(input_path)
    n_il, n_xl, n_s = cube_shape(f)
    logger.info("Cube shape: ilines=%d xlines=%d samples=%d", n_il, n_xl, n_s)

    mean, std = compute_cube_stats(f, n_traces=n_stat_traces)
    logger.info("Cube stats: mean=%.4f std=%.4f", mean, std)
    f.close()

    # 3. Create processor
    processor = SeismicProcessor(
        do_normalize=do_normalize,
        normalize_mean=mean,
        normalize_std=std,
        normalization_mode=normalization_mode,
        clip_sigma=clip_sigma,
        model_type=model_type,
    )

    # 4. Dispatch to model-specific pipeline
    tmp_dir = None
    with AsyncWriter() as writer:
        if model_type == "vit3d":
            ds, tmp_dir = _run_3d(
                input_path=input_path,
                model=model,
                processor=processor,
                config=config,
                n_il=n_il,
                n_xl=n_xl,
                n_s=n_s,
                direction=direction,
                crop_size=crop_size,
                patch_size=patch_size,
                num_overlap_patches=num_overlap_patches,
                overlap_filter=overlap_filter,
                densify=densify,
                batch_size=batch_size,
                device=device,
                torch_dtype=torch_dtype,
                hidden_size=hidden_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                max_preload=max_preload,
                pad_before=pad_before,
                non_blocking=non_blocking,
                writer=writer,
            )
        elif model_type == "vit25d":
            if input_views is None:
                input_views = list(config.directions)
            if direction not in input_views:
                raise ValueError(
                    f"Primary direction {direction!r} must be included in input_views={input_views!r} "
                    "for 2.5D inference."
                )
            unsupported_views = [view for view in input_views if view not in config.directions]
            if unsupported_views:
                raise ValueError(
                    f"input_views contains directions not present in the model config: {unsupported_views!r}. "
                    f"Supported directions are {list(config.directions)!r}."
                )
            ds, tmp_dir = _run_25d(
                input_path=input_path,
                model=model,
                processor=processor,
                config=config,
                n_il=n_il,
                n_xl=n_xl,
                n_s=n_s,
                direction=direction,
                input_views=input_views,
                crop_size=crop_size,
                patch_size=patch_size,
                num_overlap_patches=num_overlap_patches,
                overlap_filter=overlap_filter,
                densify=densify,
                batch_size=batch_size,
                device=device,
                torch_dtype=torch_dtype,
                hidden_size=hidden_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                max_preload=max_preload,
                pad_before=pad_before,
                non_blocking=non_blocking,
                writer=writer,
            )
        elif model_type == "vit":
            ds, tmp_dir = _run_2d(
                input_path=input_path,
                model=model,
                processor=processor,
                n_il=n_il,
                n_xl=n_xl,
                n_s=n_s,
                direction=direction,
                crop_size=crop_size,
                patch_size=patch_size,
                num_overlap_patches=num_overlap_patches,
                overlap_filter=overlap_filter,
                densify=densify,
                batch_size=batch_size,
                device=device,
                torch_dtype=torch_dtype,
                hidden_size=hidden_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                pad_before=pad_before,
                non_blocking=non_blocking,
                writer=writer,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        writer.barrier()

    # 5. Compute per-channel mean and standard deviation over all spatial positions (lazy via dask)
    var_name = "features"
    feat_da = ds[var_name].data  # dask array
    spatial_axes = tuple(range(feat_da.ndim - 1))  # all axes except the last (feature)
    channel_mean = feat_da.mean(axis=spatial_axes).compute().astype(np.float32)
    channel_std = feat_da.std(axis=spatial_axes).compute().astype(np.float32)
    ds["mean"] = (["feature"], channel_mean)
    ds["std"] = (["feature"], channel_std)
    logger.info(
        "%s: channel mean range [%.4f, %.4f], channel std range [%.4f, %.4f]",
        var_name,
        channel_mean.min(),
        channel_mean.max(),
        channel_std.min(),
        channel_std.max(),
    )

    # 6. Save output
    output_path = Path(output_path)
    if output_path.suffix == ".zarr":
        ds.to_zarr(output_path, group=direction, mode="a")  # ty:ignore[invalid-argument-type]
    else:
        ds.to_netcdf(output_path, group=direction, mode="a")

    logger.info("Saved output to %s", output_path)

    # 6b. Write metadata
    try:
        root = zarr.open_group(str(output_path), mode="a")
        if model_type == "vit25d" and input_views is not None:
            inference_directions = [":".join(input_views)]
        else:
            inference_directions = sorted(k for k in root.group_keys() if k.startswith("dir"))
        existing = dict(root.attrs)
        existing.update(
            {
                "ncs_version": "0.1.0",
                "model_path": str(model_path),
                "model_type": base_model_type,
                "inference_directions": inference_directions,
                "crop_size": [crop_size, crop_size, crop_size],
                "extraction_mode": extraction_mode,
                "num_overlap_patches": num_overlap_patches,
                "inference_name": Path(str(model_path)).name,
                "seismic_file": str(input_path),
                "dtype": dtype,
                "is_valid": True,
            }
        )
        root.attrs.update(existing)
        logger.info(f"Wrote FMS-compatible metadata to root zarr group at {output_path}")
    except Exception as _e:
        logger.warning("Could not write FMS-compatible metadata to zarr root: %s", _e)
    
    # 7. Clean up temporary zarr storage
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return ds


# ---------------------------------------------------------------------------
# 3D pipeline
# ---------------------------------------------------------------------------


def _run_3d(
    input_path: str | Path,
    model,
    processor: SeismicProcessor,
    config,
    n_il: int,
    n_xl: int,
    n_s: int,
    direction: str,
    crop_size: int,
    patch_size: int,
    num_overlap_patches: int,
    overlap_filter: OverlapFilter,
    densify: int,
    batch_size: int,
    device: str,
    torch_dtype: torch.dtype,
    hidden_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    max_preload: int,
    pad_before: bool,
    non_blocking: bool,
    writer: AsyncWriter,
) -> tuple[xr.Dataset, str]:
    """3D ViT inference: tile all three axes, accumulate in patch space."""
    patches_per_crop = crop_size // patch_size

    dense_il = compute_densified_axis(n_il, patch_size, densify)
    dense_xl = compute_densified_axis(n_xl, patch_size, densify)
    dense_t = compute_densified_axis(n_s, patch_size, densify)

    tmp_dir = tempfile.mkdtemp(prefix="ncs_")
    output_zarr = zarr.open_array(
        os.path.join(tmp_dir, "output.zarr"),
        mode="w",
        shape=(dense_il.size, dense_xl.size, dense_t.size, hidden_size),
        chunks=(patches_per_crop, patches_per_crop, patches_per_crop, hidden_size),
        dtype=np.float32,
        fill_value=0,
    )
    stride_patches = patches_per_crop - num_overlap_patches
    total_crops = 0
    processor_config = build_processor_config(processor)
    # for il_count in _nonzero_phase_counts(dense_il):
    #     tile_il = compute_tile_positions(il_count, crop_size, patch_size, num_overlap_patches)
    #     for xl_count in _nonzero_phase_counts(dense_xl):
    #         tile_xl = compute_tile_positions(xl_count, crop_size, patch_size, num_overlap_patches)
    #         for t_count in _nonzero_phase_counts(dense_t):
    #             tile_t = compute_tile_positions(t_count, crop_size, patch_size, num_overlap_patches)
    for il_axis_length in [l for l, c in zip(dense_il.phase_axis_lengths, dense_il.phase_counts) if c > 0]:
        tile_il = compute_tile_positions(
            il_axis_length, crop_size, patch_size, num_overlap_patches, pad_before=pad_before
        )
        for xl_axis_length in [l for l, c in zip(dense_xl.phase_axis_lengths, dense_xl.phase_counts) if c > 0]:
            tile_xl = compute_tile_positions(
                xl_axis_length, crop_size, patch_size, num_overlap_patches, pad_before=pad_before
            )
            for t_axis_length in [l for l, c in zip(dense_t.phase_axis_lengths, dense_t.phase_counts) if c > 0]:
                tile_t = compute_tile_positions(
                    t_axis_length, crop_size, patch_size, num_overlap_patches, pad_before=pad_before
                )
                total_crops += len(tile_il.starts) * len(tile_xl.starts) * len(tile_t.starts)
    pbar = tqdm(total=total_crops, desc="3D inference")

    for il_phase_idx, (il_phase_offset, il_axis_length, il_count) in enumerate(
        zip(dense_il.phase_offsets, dense_il.phase_axis_lengths, dense_il.phase_counts, strict=True)
    ):
        if il_count == 0:
            continue
        for xl_phase_idx, (xl_phase_offset, xl_axis_length, xl_count) in enumerate(
            zip(dense_xl.phase_offsets, dense_xl.phase_axis_lengths, dense_xl.phase_counts, strict=True)
        ):
            if xl_count == 0:
                continue
            tile_xl = compute_tile_positions(
                xl_axis_length, crop_size, patch_size, num_overlap_patches, pad_before=pad_before
            )
            w_xl = compute_tile_patch_overlap_weights(
                len(tile_xl.starts), patches_per_crop, num_overlap_patches, overlap_filter
            )
            xl_patch_offset = tile_xl.pad_before // patch_size
            xl_tail_pad = _axis_tail_pad(xl_axis_length, patch_size)
            padded_xl = tile_xl.pad_before + xl_axis_length + xl_tail_pad + tile_xl.pad_after

            for t_phase_idx, (t_phase_offset, t_axis_length, t_count) in enumerate(
                zip(dense_t.phase_offsets, dense_t.phase_axis_lengths, dense_t.phase_counts, strict=True)
            ):
                if t_count == 0:
                    continue

                tile_il = compute_tile_positions(
                    il_axis_length, crop_size, patch_size, num_overlap_patches, pad_before=pad_before
                )
                tile_t = compute_tile_positions(
                    t_axis_length, crop_size, patch_size, num_overlap_patches, pad_before=pad_before
                )
                w_il = compute_tile_patch_overlap_weights(
                    len(tile_il.starts), patches_per_crop, num_overlap_patches, overlap_filter
                )
                w_t = compute_tile_patch_overlap_weights(
                    len(tile_t.starts), patches_per_crop, num_overlap_patches, overlap_filter
                )
                il_patch_offset = tile_il.pad_before // patch_size
                t_patch_offset = tile_t.pad_before // patch_size
                t_tail_pad = _axis_tail_pad(t_axis_length, patch_size)
                padded_t = tile_t.pad_before + t_axis_length + t_tail_pad + tile_t.pad_after

                feature_sum = zarr.open_array(
                    os.path.join(tmp_dir, f"feature_sum_3d_phase_{il_phase_idx}_{xl_phase_idx}_{t_phase_idx}.zarr"),
                    mode="w",
                    shape=(tile_il.n_patches_total, tile_xl.n_patches_total, tile_t.n_patches_total, hidden_size),
                    chunks=(patches_per_crop, patches_per_crop, patches_per_crop, hidden_size),
                    dtype=np.float32,
                    fill_value=0,
                )
                weight_sum = zarr.open_array(
                    os.path.join(tmp_dir, f"weight_sum_3d_phase_{il_phase_idx}_{xl_phase_idx}_{t_phase_idx}.zarr"),
                    mode="w",
                    shape=(tile_il.n_patches_total, tile_xl.n_patches_total, tile_t.n_patches_total),
                    chunks=(patches_per_crop, patches_per_crop, patches_per_crop),
                    dtype=np.float32,
                    fill_value=0,
                )

                loader = build_inference_loader(
                    ThreeDPhaseDataset(
                        input_path=input_path,
                        processor_config=processor_config,
                        crop_size=crop_size,
                        n_il=n_il,
                        n_xl=n_xl,
                        n_s=n_s,
                        il_phase_offset=il_phase_offset,
                        xl_phase_offset=xl_phase_offset,
                        xl_axis_length=xl_axis_length,
                        t_phase_offset=t_phase_offset,
                        t_axis_length=t_axis_length,
                        padded_xl=padded_xl,
                        padded_t=padded_t,
                        tile_il=tile_il,
                        tile_xl=tile_xl,
                        tile_t=tile_t,
                        worker_batch_size=batch_size,
                        max_preload=max_preload,
                    ),
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    prefetch_factor=prefetch_factor,
                )
                loader_iter = iter(loader)
                while True:
                    try:
                        item = next(loader_iter)
                    except StopIteration:
                        break
                    tile_feature_sum = np.zeros(
                        (patches_per_crop, tile_xl.n_patches_total, tile_t.n_patches_total, hidden_size),
                        dtype=np.float32,
                    )
                    tile_weight_sum = np.zeros(
                        (patches_per_crop, tile_xl.n_patches_total, tile_t.n_patches_total), dtype=np.float32
                    )
                    for pixel_chunk, meta_chunk in _iter_model_batches(item, batch_size):
                        _process_3d_batch(
                            pixel_chunk,
                            meta_chunk,
                            model,
                            device,
                            torch_dtype,
                            non_blocking,
                            patches_per_crop,
                            w_il,
                            w_xl,
                            w_t,
                            tile_feature_sum,
                            tile_weight_sum,
                            patch_size,
                            hidden_size,
                            stride_patches,
                        )
                        pbar.update(len(meta_chunk))
                    writer.submit(
                        _flush_3d_tile_accumulation,
                        feature_sum,
                        weight_sum,
                        tile_feature_sum,
                        tile_weight_sum,
                        item["tile_index"],
                        patches_per_crop,
                        stride_patches,
                    )

                writer.barrier()

                for il_start in range(0, il_count, patches_per_crop):
                    il_end = min(il_start + patches_per_crop, il_count)
                    for xl_start in range(0, xl_count, patches_per_crop):
                        xl_end = min(xl_start + patches_per_crop, xl_count)
                        for t_start in range(0, t_count, patches_per_crop):
                            t_end = min(t_start + patches_per_crop, t_count)
                            feat_chunk = np.array(
                                feature_sum[
                                    il_patch_offset + il_start : il_patch_offset + il_end,
                                    xl_patch_offset + xl_start : xl_patch_offset + xl_end,
                                    t_patch_offset + t_start : t_patch_offset + t_end,
                                ]
                            )
                            w_chunk = np.array(
                                weight_sum[
                                    il_patch_offset + il_start : il_patch_offset + il_end,
                                    xl_patch_offset + xl_start : xl_patch_offset + xl_end,
                                    t_patch_offset + t_start : t_patch_offset + t_end,
                                ]
                            )
                            mask = w_chunk > 0
                            feat_chunk[mask] /= w_chunk[mask, np.newaxis]
                            writer.submit(
                                write_array_slice,
                                output_zarr,
                                (
                                    _phase_slice(il_phase_idx + densify * il_start, il_end - il_start, densify),
                                    _phase_slice(xl_phase_idx + densify * xl_start, xl_end - xl_start, densify),
                                    _phase_slice(t_phase_idx + densify * t_start, t_end - t_start, densify),
                                ),
                                feat_chunk,
                            )

    pbar.close()
    writer.barrier()

    ds = xr.Dataset(
        {
            "features": (
                ["inline", "xline", "time_depth", "feature"],
                da.from_zarr(output_zarr),
            )
        },
        coords={
            "inline": dense_il.coords,
            "xline": dense_xl.coords,
            "time_depth": dense_t.coords,
        },
    )
    return ds, tmp_dir


def _process_3d_batch(
    pixel_values,
    batch_meta,
    model,
    device,
    torch_dtype,
    non_blocking,
    patches_per_crop,
    w_il,
    w_xl,
    w_t,
    tile_feature_sum,
    tile_weight_sum,
    patch_size,
    hidden_size,
    stride_patches,
):
    """Process a batch of 3D crops through the model and accumulate features."""
    pixel_values = pixel_values.to(device=device, dtype=torch_dtype, non_blocking=non_blocking)

    input_shape = pixel_values.shape[-1]

    volume_indices, volume_size = _get_volume_index_tensors(
        device=device,
        input_shape=input_shape,
        patch_size=patch_size,
        patches_per_crop=patches_per_crop,
        batch_len=pixel_values.shape[0],
    )

    with torch.inference_mode():
        outputs = model(
            pixel_values=pixel_values,
            volume_indices=volume_indices,
            volume_size=volume_size,
            output_hidden_states=True,
        )

    # last_hidden_state: (B, 1 + P^3, hidden_size) — skip CLS token
    feats = outputs.last_hidden_state[:, 1:, :].cpu().float().numpy()
    # Reshape to (B, P, P, P, hidden_size)
    feats = feats.reshape(-1, patches_per_crop, patches_per_crop, patches_per_crop, hidden_size)

    for b, (il_tile_idx, xl_tile_idx, t_tile_idx) in enumerate(batch_meta):
        xl_ps = xl_tile_idx * stride_patches
        t_ps = t_tile_idx * stride_patches
        il_slice = slice(0, patches_per_crop)
        xl_slice = slice(xl_ps, xl_ps + patches_per_crop)
        t_slice = slice(t_ps, t_ps + patches_per_crop)
        w3d = np.einsum("i,j,k->ijk", w_il[il_tile_idx], w_xl[xl_tile_idx], w_t[t_tile_idx])
        # feats[b] is (P_time, P_iline, P_xline, H) because processor transposed to (time, iline, xline)
        # but we want to accumulate as (xline, iline, time, H) to match the coord order
        tile_feature_sum[il_slice, xl_slice, t_slice] += feats[b].transpose(1, 2, 0, 3) * w3d[:, :, :, np.newaxis]
        tile_weight_sum[il_slice, xl_slice, t_slice] += w3d


# ---------------------------------------------------------------------------
# 2D pipeline
# ---------------------------------------------------------------------------


def _run_2d(
    input_path: str | Path,
    model,
    processor: SeismicProcessor,
    n_il: int,
    n_xl: int,
    n_s: int,
    direction: str,
    crop_size: int,
    patch_size: int,
    num_overlap_patches: int,
    overlap_filter: OverlapFilter,
    densify: int,
    batch_size: int,
    device: str,
    torch_dtype: torch.dtype,
    hidden_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    pad_before: bool,
    non_blocking: bool,
    writer: AsyncWriter,
) -> tuple[xr.Dataset, str]:
    """2D ViT inference: one representative slice per primary patch position.

    Output shape is always ``(ceil(n_il/ps), ceil(n_xl/ps), ceil(n_s/ps), hidden_size)``
    in ``(inline, xline, time_depth)`` order regardless of traversal direction.
    """
    patches_per_crop = crop_size // patch_size

    n_primary, n_spatial = _resolve_direction_axes(direction, n_il, n_xl)

    dense_primary = compute_densified_axis(n_primary, patch_size, densify)
    dense_sp = compute_densified_axis(n_spatial, patch_size, densify)
    dense_t = compute_densified_axis(n_s, patch_size, densify)
    sp_phase_configs = _build_centered_axis_phase_configs(
        dense_sp,
        patch_size,
        num_overlap_patches,
        overlap_filter,
        patches_per_crop,
        pad_before,
    )
    t_phase_configs = _build_centered_axis_phase_configs(
        dense_t,
        patch_size,
        num_overlap_patches,
        overlap_filter,
        patches_per_crop,
        pad_before,
    )

    # One entry per primary patch position, spatial and time patch-gridded (zarr on disk)
    tmp_dir = tempfile.mkdtemp(prefix="ncs_")
    all_features = zarr.open_array(
        os.path.join(tmp_dir, "features.zarr"),
        mode="w",
        shape=(dense_primary.size, dense_sp.size, dense_t.size, hidden_size),
        chunks=(1, max(dense_sp.size, 1), max(dense_t.size, 1), hidden_size),
        dtype=np.float32,
        fill_value=0,
    )

    logger.info(
        "Running 2D inference in direction=%s: primary=%d spatial=%d time=%d",
        direction,
        n_primary,
        n_spatial,
        n_s,
    )
    logger.info(
        "Densified feature grid shape will be (primary=%d, spatial=%d, time=%d, hidden_size=%d)",
        dense_primary.size,
        dense_sp.size,
        dense_t.size,
        hidden_size,
    )

    total_positions = (
        sum(_nonzero_phase_counts(dense_primary))
        * len(_nonzero_phase_counts(dense_sp))
        * len(_nonzero_phase_counts(dense_t))
    )
    processor_config = build_processor_config(processor)
    pbar = tqdm(total=total_positions, desc="2D inference")
    for primary_phase_idx, (primary_phase_offset, _, primary_count) in enumerate(
        zip(dense_primary.phase_offsets, dense_primary.phase_axis_lengths, dense_primary.phase_counts, strict=True)
    ):
        if primary_count == 0:
            continue

        for sp_phase in sp_phase_configs:
            for t_phase in t_phase_configs:
                loader = build_inference_loader(
                    TwoDPositionDataset(
                        input_path=input_path,
                        processor_config=processor_config,
                        direction=direction,
                        patch_size=patch_size,
                        crop_size=crop_size,
                        densify=densify,
                        n_primary=n_primary,
                        primary_phase_idx=primary_phase_idx,
                        primary_phase_offset=primary_phase_offset,
                        primary_count=primary_count,
                        sp_phase_idx=sp_phase.phase_idx,
                        sp_phase_offset=sp_phase.phase_offset,
                        t_phase_idx=t_phase.phase_idx,
                        t_phase_offset=t_phase.phase_offset,
                        tile_sp=sp_phase.tile,
                        sp_tail_pad=sp_phase.tail_pad,
                        tile_t=t_phase.tile,
                        t_tail_pad=t_phase.tail_pad,
                        worker_batch_size=batch_size,
                    ),
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    prefetch_factor=prefetch_factor,
                )
                loader_iter = iter(loader)
                while True:
                    try:
                        item = next(loader_iter)
                    except StopIteration:
                        break
                    feature_sum = torch.zeros(
                        (sp_phase.count, t_phase.count, hidden_size), device=device, dtype=torch.float32
                    )
                    weight_sum = torch.zeros((sp_phase.count, t_phase.count), device=device, dtype=torch.float32)
                    w_sp_t = torch.as_tensor(sp_phase.weights, device=device, dtype=torch.float32)
                    w_t_t = torch.as_tensor(t_phase.weights, device=device, dtype=torch.float32)
                    sp_tile_centers = [start // patch_size for start in sp_phase.tile.starts]
                    t_tile_centers = [start // patch_size for start in t_phase.tile.starts]

                    for pixel_chunk, meta_chunk in _iter_model_batches(item, batch_size):
                        _process_2d_batch(
                            pixel_chunk,
                            meta_chunk,
                            model,
                            device,
                            torch_dtype,
                            non_blocking,
                            patches_per_crop,
                            w_sp_t,
                            w_t_t,
                            feature_sum,
                            weight_sum,
                            hidden_size,
                            sp_tile_centers,
                            t_tile_centers,
                        )

                    mask = weight_sum > 0
                    feature_sum[mask] /= weight_sum[mask].unsqueeze(-1)
                    feature_sum_np = feature_sum.cpu().numpy()
                    _submit_single_sliced_position(
                        writer,
                        all_features,
                        feature_sum_np,
                        item["primary_index"],
                        item["sp_phase_idx"],
                        item["t_phase_idx"],
                        densify,
                    )
                    pbar.update(1)
    pbar.close()
    writer.barrier()

    # Build dataset in canonical (inline, xline, time_depth) order
    ds = _build_dataset_from_sliced(
        all_features,
        direction,
        dense_primary.coords,
        dense_sp.coords,
        dense_t.coords,
        hidden_size,
    )
    return ds, tmp_dir


def _process_2d_batch(
    pixel_values,
    batch_meta,
    model,
    device,
    torch_dtype,
    non_blocking,
    patches_per_crop,
    w_sp,
    w_t,
    feature_sum,
    weight_sum,
    hidden_size,
    sp_tile_centers,
    t_tile_centers,
):
    """Process a batch of 2D crops through the model and accumulate."""
    pixel_values = pixel_values.to(device=device, dtype=torch_dtype, non_blocking=non_blocking)

    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values)

    # (B, 1 + P^2, H) -> skip CLS
    feats = outputs.last_hidden_state[:, 1:, :].reshape(-1, patches_per_crop, patches_per_crop, hidden_size).float()

    for b, (sp_tile_idx, t_tile_idx) in enumerate(batch_meta):
        sp_slices = _centered_tile_slices(sp_tile_centers[sp_tile_idx], patches_per_crop, feature_sum.shape[0])
        t_slices = _centered_tile_slices(t_tile_centers[t_tile_idx], patches_per_crop, feature_sum.shape[1])
        if sp_slices is None or t_slices is None:
            continue
        sp_slice, sp_src = sp_slices
        t_slice, t_src = t_slices
        w2d = torch.outer(w_sp[sp_tile_idx], w_t[t_tile_idx])
        # feats[b] is (P_time, P_spatial, H) because processor transposed to (time, spatial)
        # but we want to accumulate as (spatial, time, H) to match the coord order
        tile_feats = feats[b].permute(1, 0, 2)[sp_src, t_src]
        tile_w = w2d[sp_src, t_src]
        feature_sum[sp_slice, t_slice] += tile_feats * tile_w.unsqueeze(-1)
        weight_sum[sp_slice, t_slice] += tile_w


def _build_dataset_from_sliced(features, direction, primary_coords, spatial_coords, t_coords, hidden_size):
    """Build xarray Dataset from slice-based inference features.

    ``features`` is a zarr array with shape ``(n_primary_positions, n_sp_positions, n_t_positions, hidden_size)``.
    Output is always in canonical ``(inline, xline, time_depth, feature)`` order.

    For ``dir0``:  primary=inline,  spatial=xline  → already ``(il, xl, t, H)``
    For ``dir90``: primary=xline,   spatial=inline → transpose to ``(il, xl, t, H)``
    """
    if direction == "dir0":
        # features is (primary=il, spatial=xl, t, H) — already canonical
        data_da = da.from_zarr(features)
        il_coords = primary_coords
        xl_coords = spatial_coords
    else:
        # features is (primary=xl, spatial=il, t, H) — transpose first two axes lazily
        data_da = da.from_zarr(features).transpose(1, 0, 2, 3)
        il_coords = spatial_coords
        xl_coords = primary_coords

    ds = xr.Dataset(
        {"features": (["inline", "xline", "time_depth", "feature"], data_da)},
        coords={
            "inline": il_coords,
            "xline": xl_coords,
            "time_depth": t_coords,
        },
    )
    return ds


def _build_dataset_from_phase_sliced(
    phase_features: dict[tuple[int, int], zarr.Array],
    direction: str,
    primary_coords: np.ndarray,
    spatial_coords: np.ndarray,
    t_coords: np.ndarray,
    hidden_size: int,
    densify: int,
) -> xr.Dataset:
    """Build a dense xarray Dataset from phase-contiguous 2.5D feature arrays."""
    if direction == "dir0":
        primary_dim, spatial_dim = "inline", "xline"
    else:
        primary_dim, spatial_dim = "xline", "inline"

    feature_coords = np.arange(hidden_size, dtype=np.int32)
    spatial_phase_indices = sorted({sp_phase_idx for sp_phase_idx, _ in phase_features})
    t_phase_indices = sorted({t_phase_idx for _, t_phase_idx in phase_features})
    spatial_rows: list[xr.DataArray] = []
    for sp_phase_idx in spatial_phase_indices:
        time_tiles: list[xr.DataArray] = []
        sp_phase_coords = _phase_coords(spatial_coords, sp_phase_idx, densify)
        for t_phase_idx in t_phase_indices:
            phase_array = phase_features[(sp_phase_idx, t_phase_idx)]
            time_tiles.append(
                xr.DataArray(
                    da.from_zarr(phase_array),
                    dims=(primary_dim, spatial_dim, "time_depth", "feature"),
                    coords={
                        primary_dim: primary_coords,
                        spatial_dim: sp_phase_coords,
                        "time_depth": _phase_coords(t_coords, t_phase_idx, densify),
                        "feature": feature_coords,
                    },
                )
            )
        spatial_rows.append(xr.concat(time_tiles, dim="time_depth").sortby("time_depth"))

    features = xr.concat(spatial_rows, dim=spatial_dim).sortby(spatial_dim)
    features = features.chunk(
        {dim: max(chunks) for dim, chunks in zip(features.dims, features.data.chunks, strict=True)}
    )
    if direction == "dir90":
        features = features.transpose("inline", "xline", "time_depth", "feature")

    return xr.Dataset({"features": features})


# ---------------------------------------------------------------------------
# 2.5D pipeline
# ---------------------------------------------------------------------------


def _run_25d(
    input_path: str | Path,
    model,
    processor: SeismicProcessor,
    config,
    n_il: int,
    n_xl: int,
    n_s: int,
    direction: str,
    input_views: list[str],
    crop_size: int,
    patch_size: int,
    num_overlap_patches: int,
    overlap_filter: OverlapFilter,
    densify: int,
    batch_size: int,
    device: str,
    torch_dtype: torch.dtype,
    hidden_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    max_preload: int,
    pad_before: bool,
    non_blocking: bool,
    writer: AsyncWriter,
) -> tuple[xr.Dataset, str]:
    """2.5D ViT inference with DataLoader-backed primary-slab preparation."""
    patches_per_crop = crop_size // patch_size

    # Determine primary/spatial axes
    n_primary, n_spatial = _resolve_direction_axes(direction, n_il, n_xl)

    # Map direction to view index for extracting travel-direction features
    all_config_dirs = list(config.directions)
    travel_view_idx = input_views.index(direction)

    # Build directions tensor for the model
    direction_indices = torch.tensor([all_config_dirs.index(v) for v in input_views], dtype=torch.int32).unsqueeze(
        0
    )  # (1, num_views)

    dense_primary = compute_densified_axis(n_primary, patch_size, densify)
    dense_sp = compute_densified_axis(n_spatial, patch_size, densify)
    dense_t = compute_densified_axis(n_s, patch_size, densify)
    sp_phase_configs = _build_centered_axis_phase_configs(
        dense_sp,
        patch_size,
        num_overlap_patches,
        overlap_filter,
        patches_per_crop,
        pad_before,
    )
    t_phase_configs = _build_centered_axis_phase_configs(
        dense_t,
        patch_size,
        num_overlap_patches,
        overlap_filter,
        patches_per_crop,
        pad_before,
    )

    tmp_dir = tempfile.mkdtemp(prefix="ncs_")
    phase_features: dict[tuple[int, int], zarr.Array] = {}

    logger.info(
        "Running 2.5D inference in direction=%s: primary=%d spatial=%d time=%d",
        direction,
        n_primary,
        n_spatial,
        n_s,
    )
    logger.info(
        "Densified feature grid shape will be (primary=%d, spatial=%d, time=%d, hidden_size=%d)",
        dense_primary.size,
        dense_sp.size,
        dense_t.size,
        hidden_size,
    )

    num_views = len(input_views)
    patches_per_view = patches_per_crop * patches_per_crop
    total_positions = (
        sum(_nonzero_phase_counts(dense_primary))
        * len(_nonzero_phase_counts(dense_sp))
        * len(_nonzero_phase_counts(dense_t))
    )
    processor_config = build_processor_config(processor)
    pbar = tqdm(total=total_positions, desc="2.5D inference")

    primary_phase_configs: list[dict[str, Any]] = []
    phase_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}

    for primary_phase_idx, (primary_phase_offset, _primary_axis_length, primary_count) in enumerate(
        zip(dense_primary.phase_offsets, dense_primary.phase_axis_lengths, dense_primary.phase_counts, strict=True)
    ):
        if primary_count == 0:
            continue

        phase_configs: list[dict[str, Any]] = []
        for sp_phase in sp_phase_configs:
            for t_phase in t_phase_configs:
                phase_config = {
                    "sp_phase_idx": sp_phase.phase_idx,
                    "sp_phase_offset": sp_phase.phase_offset,
                    "sp_axis_length": sp_phase.axis_length,
                    "sp_count": sp_phase.count,
                    "tile_sp": sp_phase.tile,
                    "w_sp": sp_phase.weights,
                    "sp_patch_offset": sp_phase.patch_offset,
                    "sp_tail_pad": sp_phase.tail_pad,
                    "t_phase_idx": t_phase.phase_idx,
                    "t_phase_offset": t_phase.phase_offset,
                    "t_axis_length": t_phase.axis_length,
                    "t_count": t_phase.count,
                    "tile_t": t_phase.tile,
                    "w_t": t_phase.weights,
                    "t_patch_offset": t_phase.patch_offset,
                    "t_tail_pad": t_phase.tail_pad,
                }
                phase_features[(sp_phase.phase_idx, t_phase.phase_idx)] = zarr.open_array(
                    os.path.join(tmp_dir, f"features_25d_phase_{sp_phase.phase_idx}_{t_phase.phase_idx}.zarr"),
                    mode="w",
                    shape=(dense_primary.size, sp_phase.count, t_phase.count, hidden_size),
                    chunks=(1, max(sp_phase.count, 1), max(t_phase.count, 1), hidden_size),
                    dtype=np.float32,
                    fill_value=0,
                )
                phase_configs.append(phase_config)
                phase_lookup[(primary_phase_idx, sp_phase.phase_idx, t_phase.phase_idx)] = phase_config

        primary_phase_configs.append(
            {
                "primary_phase_idx": primary_phase_idx,
                "primary_phase_offset": primary_phase_offset,
                "primary_count": primary_count,
                "phase_configs": phase_configs,
            }
        )

    loader = build_inference_loader(
        TwoPointFiveDPipelineDataset(
            input_path=input_path,
            processor_config=processor_config,
            direction=direction,
            input_views=input_views,
            patch_size=patch_size,
            crop_size=crop_size,
            densify=densify,
            n_primary=n_primary,
            n_spatial=n_spatial,
            n_samples=n_s,
            primary_phase_configs=primary_phase_configs,
            worker_batch_size=batch_size,
            max_preload=max_preload,
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    loader_iter = iter(loader)
    while True:
        try:
            item = next(loader_iter)
        except StopIteration:
            break
        phase_key = (item["primary_phase_idx"], item["sp_phase_idx"], item["t_phase_idx"])
        phase_config = phase_lookup[phase_key]
        tile_sp = phase_config["tile_sp"]
        tile_t = phase_config["tile_t"]
        feature_sum = torch.zeros(
            (phase_config["sp_count"], phase_config["t_count"], hidden_size), device=device, dtype=torch.float32
        )
        weight_sum = torch.zeros(
            (phase_config["sp_count"], phase_config["t_count"]), device=device, dtype=torch.float32
        )
        w_sp_t = torch.as_tensor(phase_config["w_sp"], device=device, dtype=torch.float32)
        w_t_t = torch.as_tensor(phase_config["w_t"], device=device, dtype=torch.float32)
        sp_tile_centers = [start // patch_size for start in tile_sp.starts]
        t_tile_centers = [start // patch_size for start in tile_t.starts]
        for pixel_chunk, meta_chunk in _iter_model_batches(item, batch_size):
            _process_25d_batch(
                pixel_chunk,
                meta_chunk,
                model,
                device,
                torch_dtype,
                non_blocking,
                patches_per_crop,
                w_sp_t,
                w_t_t,
                feature_sum,
                weight_sum,
                hidden_size,
                direction_indices,
                travel_view_idx,
                num_views,
                patches_per_view,
                sp_tile_centers,
                t_tile_centers,
            )

        nonzero = weight_sum > 0
        feature_sum /= (weight_sum + (~nonzero).to(weight_sum.dtype)).unsqueeze(-1)
        feature_sum *= nonzero.unsqueeze(-1)
        feature_sum_np = feature_sum.cpu().numpy()
        _submit_single_phase_position(
            writer,
            phase_features[(item["sp_phase_idx"], item["t_phase_idx"])],
            feature_sum_np,
            item["primary_index"],
        )
        pbar.update(1)
    pbar.close()
    writer.barrier()

    ds = _build_dataset_from_phase_sliced(
        phase_features,
        direction,
        dense_primary.coords,
        dense_sp.coords,
        dense_t.coords,
        hidden_size,
        densify,
    )
    return ds, tmp_dir


def _process_25d_batch(
    pixel_values,
    batch_meta,
    model,
    device,
    torch_dtype,
    non_blocking,
    patches_per_crop,
    w_sp,
    w_t,
    feature_sum,
    weight_sum,
    hidden_size,
    direction_indices,
    travel_view_idx,
    num_views,
    patches_per_view,
    sp_tile_centers,
    t_tile_centers,
):
    """Process a batch of 2.5D multi-view crops and accumulate travel-direction features."""
    pixel_values = pixel_values.to(device=device, dtype=torch_dtype, non_blocking=non_blocking)
    B = pixel_values.shape[0]

    # Expand directions to batch
    dirs = direction_indices.expand(B, -1).to(device)

    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values, directions=dirs)

    # last_hidden_state: (B, 1 + num_views * P^2, H)
    # Extract only the travel direction's patches
    # Token layout: [CLS, view0_patch0, ..., view0_patchN, view1_patch0, ..., viewK_patchN]
    start_tok = 1 + travel_view_idx * patches_per_view
    end_tok = start_tok + patches_per_view
    feats = outputs.last_hidden_state[:, start_tok:end_tok, :].float()
    # Reshape to (B, P_time, P_spatial, H)
    # SeismicProcessor applies seismic_to_torch_2d (np.moveaxis(-1,-2)) converting
    # (n_views, spatial, time) -> (n_views, time, spatial) before the model, so
    # the ViT tokenizes time-first. feats[b] is (P_time, P_spatial, H).
    # Transpose to (P_spatial, P_time, H) to match feature_sum[sp_slice, t_slice].
    feats = feats.reshape(-1, patches_per_crop, patches_per_crop, hidden_size)

    for b, (sp_tile_idx, t_tile_idx) in enumerate(batch_meta):
        sp_slices = _centered_tile_slices(sp_tile_centers[sp_tile_idx], patches_per_crop, feature_sum.shape[0])
        t_slices = _centered_tile_slices(t_tile_centers[t_tile_idx], patches_per_crop, feature_sum.shape[1])
        if sp_slices is None or t_slices is None:
            continue
        sp_slice, sp_src = sp_slices
        t_slice, t_src = t_slices
        w2d = torch.outer(w_sp[sp_tile_idx], w_t[t_tile_idx])
        # feats[b] is (P_time, P_spatial, H) -> swap to (P_spatial, P_time, H)
        tile_feats = feats[b].permute(1, 0, 2)[sp_src, t_src]
        tile_w = w2d[sp_src, t_src]
        feature_sum[sp_slice, t_slice] += tile_feats * tile_w.unsqueeze(-1)
        weight_sum[sp_slice, t_slice] += tile_w
