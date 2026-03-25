"""Torch DataLoader helpers for asynchronous inference input preparation."""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from NCS.inference.reader import open_seismic, read_inline_slab, read_xline_slab
from NCS.inference.views import extract_2d_crop_views_from_cache
from NCS.processors import SeismicProcessor


ProcessorConfig = dict[str, Any]


def build_inference_loader(
    dataset: IterableDataset,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    """Build a DataLoader for inference work units."""
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def build_processor_config(processor: SeismicProcessor) -> ProcessorConfig:
    """Serialize processor settings for worker-local reconstruction."""
    return {
        "do_normalize": processor.do_normalize,
        "normalize_mean": processor.normalize_mean,
        "normalize_std": processor.normalize_std,
        "normalization_mode": processor.normalization_mode,
        "clip_sigma": processor.clip_sigma,
        "model_type": processor.model_type,
    }


def resolve_loader_settings(
    device: str,
    num_workers: int | None,
    pin_memory: bool | None,
) -> tuple[int, bool]:
    """Choose conservative defaults for asynchronous inference loading."""
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    if pin_memory is None:
        pin_memory = str(device).startswith("cuda")
    return max(num_workers, 0), pin_memory


def _worker_bounds(length: int) -> tuple[int, int]:
    info = get_worker_info()
    if info is None:
        return 0, length
    chunk_size = (length + info.num_workers - 1) // info.num_workers
    start = min(info.id * chunk_size, length)
    end = min(start + chunk_size, length)
    return start, end


def _read_direction_slice(
    f,
    direction: str,
    idx: int,
    spatial_start: int = 0,
    spatial_count: int | None = None,
    sample_start: int = 0,
    sample_count: int | None = None,
) -> np.ndarray:
    if direction == "dir0":
        inline = np.asarray(f.iline[int(f.ilines[idx])], dtype=np.float32)
        spatial_end = inline.shape[0] if spatial_count is None else spatial_start + spatial_count
        sample_end = inline.shape[1] if sample_count is None else sample_start + sample_count
        return inline[spatial_start:spatial_end, sample_start:sample_end]
    if direction == "dir90":
        xline = np.asarray(f.xline[int(f.xlines[idx])], dtype=np.float32)
        spatial_end = xline.shape[0] if spatial_count is None else spatial_start + spatial_count
        sample_end = xline.shape[1] if sample_count is None else sample_start + sample_count
        return xline[spatial_start:spatial_end, sample_start:sample_end]
    raise ValueError(f"Unsupported direction: {direction}")


def _read_direction_slab(
    f,
    direction: str,
    start: int,
    count: int,
    spatial_start: int = 0,
    spatial_count: int | None = None,
    sample_start: int = 0,
    sample_count: int | None = None,
) -> np.ndarray:
    if direction == "dir0":
        return read_inline_slab(
            f,
            start,
            count,
            xline_start=spatial_start,
            xline_count=spatial_count,
            sample_start=sample_start,
            sample_count=sample_count,
        )
    if direction == "dir90":
        return read_xline_slab(
            f,
            start,
            count,
            inline_start=spatial_start,
            inline_count=spatial_count,
            sample_start=sample_start,
            sample_count=sample_count,
        )
    raise ValueError(f"Unsupported direction: {direction}")


def clamp_max_preload(max_preload: int) -> int:
    """Validate and normalize the worker preload size."""
    return max(max_preload, 1)


def clamp_worker_batch_size(batch_size: int) -> int:
    """Validate and normalize the worker-side preprocessing batch size."""
    return max(batch_size, 1)


def shape_aware_25d_primary_preload(crop_size: int, patch_size: int, max_preload: int) -> int:
    """Choose a shallower primary preload for 2.5D while preserving nearby-position reuse."""
    target = max(crop_size, crop_size + 8 * patch_size)
    return min(clamp_max_preload(max_preload), target)


def expand_worker_preload_bounds(
    total_primary: int,
    worker_lower: int,
    worker_upper: int,
    max_preload: int,
) -> tuple[int, int]:
    """Expand a worker span with halo so small worker chunks can still preload effectively."""
    preload = clamp_max_preload(max_preload)
    lower = max(worker_lower, 0)
    upper = min(worker_upper, total_primary)
    span = max(upper - lower, 0)
    if total_primary <= preload:
        return 0, total_primary
    target = min(total_primary, max(span, preload))
    extra = max(target - span, 0)
    start = max(0, lower - extra // 2)
    end = min(total_primary, upper + extra - extra // 2)
    current = end - start
    if current < target:
        if start == 0:
            end = min(total_primary, target)
        elif end == total_primary:
            start = max(0, total_primary - target)
    return start, end


class PreloadedSubcube:
    """Worker-local reusable subcube cache bounded by max_preload per axis."""

    def __init__(
        self,
        read_fn,
        total_primary: int,
        total_spatial: int,
        total_sample: int,
        max_preload: int,
        primary_lower: int = 0,
        primary_upper: int | None = None,
    ):
        self._read_fn = read_fn
        self._total_primary = total_primary
        self._total_spatial = total_spatial
        self._total_sample = total_sample
        self._max_preload = clamp_max_preload(max_preload)
        self._primary_lower = max(primary_lower, 0)
        self._primary_upper = total_primary if primary_upper is None else min(primary_upper, total_primary)
        self._data: np.ndarray | None = None
        self._bounds: tuple[int, int, int, int, int, int] | None = None

    @staticmethod
    def _axis_bounds(
        total: int, req_start: int, req_end: int, max_preload: int, lower: int = 0, upper: int | None = None
    ) -> tuple[int, int]:
        axis_upper = total if upper is None else min(upper, total)
        axis_lower = max(lower, 0)
        axis_total = max(axis_upper - axis_lower, 0)
        req_len = max(req_end - req_start, 0)
        preload_len = min(axis_total, max(req_len, max_preload))
        if preload_len >= axis_total:
            return axis_lower, axis_upper
        center = (req_start + req_end) // 2
        start = max(axis_lower, min(center - preload_len // 2, axis_upper - preload_len))
        end = start + preload_len
        if req_start < start:
            start = req_start
            end = start + preload_len
        if req_end > end:
            end = req_end
            start = end - preload_len
        return max(start, axis_lower), min(end, axis_upper)

    def fetch(
        self,
        primary_start: int,
        primary_end: int,
        spatial_start: int,
        spatial_end: int,
        sample_start: int,
        sample_end: int,
    ) -> np.ndarray:
        cache_data, cache_bounds = self.fetch_context(
            primary_start,
            primary_end,
            spatial_start,
            spatial_end,
            sample_start,
            sample_end,
        )
        cache_p0, _cache_p1, cache_s0, _cache_s1, cache_t0, _cache_t1 = cache_bounds
        req_primary_start = max(primary_start, 0)
        req_primary_end = min(primary_end, self._total_primary)
        req_spatial_start = max(spatial_start, 0)
        req_spatial_end = min(spatial_end, self._total_spatial)
        req_sample_start = max(sample_start, 0)
        req_sample_end = min(sample_end, self._total_sample)
        return cache_data[
            req_primary_start - cache_p0 : req_primary_end - cache_p0,
            req_spatial_start - cache_s0 : req_spatial_end - cache_s0,
            req_sample_start - cache_t0 : req_sample_end - cache_t0,
        ]

    def fetch_context(
        self,
        primary_start: int,
        primary_end: int,
        spatial_start: int,
        spatial_end: int,
        sample_start: int,
        sample_end: int,
        primary_max_preload: int | None = None,
        spatial_max_preload: int | None = None,
        sample_max_preload: int | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int, int, int]]:
        req_primary_start = max(primary_start, 0)
        req_primary_end = min(primary_end, self._total_primary)
        req_spatial_start = max(spatial_start, 0)
        req_spatial_end = min(spatial_end, self._total_spatial)
        req_sample_start = max(sample_start, 0)
        req_sample_end = min(sample_end, self._total_sample)
        primary_preload = self._max_preload if primary_max_preload is None else clamp_max_preload(primary_max_preload)
        spatial_preload = self._max_preload if spatial_max_preload is None else clamp_max_preload(spatial_max_preload)
        sample_preload = self._max_preload if sample_max_preload is None else clamp_max_preload(sample_max_preload)

        if self._bounds is not None:
            cache_p0, cache_p1, cache_s0, cache_s1, cache_t0, cache_t1 = self._bounds
            if (
                cache_p0 <= req_primary_start
                and req_primary_end <= cache_p1
                and cache_s0 <= req_spatial_start
                and req_spatial_end <= cache_s1
                and cache_t0 <= req_sample_start
                and req_sample_end <= cache_t1
            ):
                assert self._data is not None
                return self._data, self._bounds

        cache_primary_start, cache_primary_end = self._axis_bounds(
            self._total_primary,
            req_primary_start,
            req_primary_end,
            primary_preload,
            lower=self._primary_lower,
            upper=self._primary_upper,
        )
        cache_spatial_start, cache_spatial_end = self._axis_bounds(
            self._total_spatial, req_spatial_start, req_spatial_end, spatial_preload
        )
        cache_sample_start, cache_sample_end = self._axis_bounds(
            self._total_sample, req_sample_start, req_sample_end, sample_preload
        )

        self._data = self._read_fn(
            cache_primary_start,
            cache_primary_end - cache_primary_start,
            cache_spatial_start,
            cache_spatial_end - cache_spatial_start,
            cache_sample_start,
            cache_sample_end - cache_sample_start,
        )
        self._bounds = (
            cache_primary_start,
            cache_primary_end,
            cache_spatial_start,
            cache_spatial_end,
            cache_sample_start,
            cache_sample_end,
        )
        return self._data, self._bounds


class _BaseInferenceDataset(IterableDataset):
    def __init__(self, input_path: str | Path, processor_config: ProcessorConfig):
        super().__init__()
        self.input_path = str(input_path)
        self.processor_config = processor_config
        self._file = None
        self._processor = None

    def _get_file(self):
        if self._file is None:
            self._file = open_seismic(self.input_path)
        return self._file

    def _get_processor(self) -> SeismicProcessor:
        if self._processor is None:
            self._processor = SeismicProcessor(**self.processor_config)
        return self._processor

    def _preprocess(self, crops: list[np.ndarray]) -> torch.Tensor:
        return self._get_processor().preprocess(crops)["pixel_values"].contiguous()

    def _finalize_crop_batches(
        self,
        pixel_value_batches: list[torch.Tensor],
        meta_batches: list[list[tuple]],
        crops: list[np.ndarray],
        meta: list[tuple],
    ) -> None:
        if not crops:
            return
        pixel_value_batches.append(self._preprocess(crops))
        meta_batches.append(meta.copy())
        crops.clear()
        meta.clear()

    def _pack_batches(
        self,
        pixel_value_batches: list[torch.Tensor],
        meta_batches: list[list[tuple]],
    ) -> tuple[torch.Tensor, list[tuple], list[int]]:
        if not pixel_value_batches:
            raise ValueError("Expected at least one preprocessed batch.")
        if len(pixel_value_batches) != len(meta_batches):
            raise ValueError("Pixel-value batches and metadata batches must stay aligned.")
        pixel_values = torch.cat(pixel_value_batches, dim=0).contiguous()
        meta = [entry for batch in meta_batches for entry in batch]
        batch_splits = [batch.shape[0] for batch in pixel_value_batches]
        if sum(batch_splits) != len(meta):
            raise ValueError("Packed  inference batches produced mismatched tensor and metadata lengths.")
        return pixel_values, meta, batch_splits

    def _pack_generated_crops(
        self,
        crop_items: Iterable[tuple[np.ndarray, tuple]],
        worker_batch_size: int,
    ) -> tuple[torch.Tensor, list[tuple], list[int]]:
        crops: list[np.ndarray] = []
        meta: list[tuple] = []
        pixel_value_batches: list[torch.Tensor] = []
        meta_batches: list[list[tuple]] = []
        for crop, crop_meta in crop_items:
            crops.append(crop)
            meta.append(crop_meta)
            if len(crops) >= worker_batch_size:
                self._finalize_crop_batches(pixel_value_batches, meta_batches, crops, meta)
        self._finalize_crop_batches(pixel_value_batches, meta_batches, crops, meta)
        return self._pack_batches(pixel_value_batches, meta_batches)

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class ThreeDPhaseDataset(_BaseInferenceDataset):
    """Yield preprocessed 3D crop batches for one phase triple."""

    def __init__(
        self,
        input_path: str | Path,
        processor_config: ProcessorConfig,
        crop_size: int,
        n_il: int,
        n_xl: int,
        n_s: int,
        il_phase_offset: int,
        xl_phase_offset: int,
        xl_axis_length: int,
        t_phase_offset: int,
        t_axis_length: int,
        padded_xl: int,
        padded_t: int,
        tile_il,
        tile_xl,
        tile_t,
        worker_batch_size: int,
        max_preload: int,
    ):
        super().__init__(input_path, processor_config)
        self.crop_size = crop_size
        self.n_il = n_il
        self.n_xl = n_xl
        self.n_s = n_s
        self.il_phase_offset = il_phase_offset
        self.xl_phase_offset = xl_phase_offset
        self.xl_axis_length = xl_axis_length
        self.t_phase_offset = t_phase_offset
        self.t_axis_length = t_axis_length
        self.padded_xl = padded_xl
        self.padded_t = padded_t
        self.tile_il = tile_il
        self.tile_xl = tile_xl
        self.tile_t = tile_t
        self.worker_batch_size = clamp_worker_batch_size(worker_batch_size)
        self.max_preload = max_preload

    def __iter__(self):
        f = self._get_file()
        worker_start, worker_end = _worker_bounds(len(self.tile_il.starts))
        if worker_start >= worker_end:
            return
        worker_il_start = self.tile_il.starts[worker_start]
        worker_il_end = self.tile_il.starts[worker_end - 1] + self.crop_size
        primary_lower, primary_upper = expand_worker_preload_bounds(
            self.n_il,
            self.il_phase_offset + worker_il_start - self.tile_il.pad_before,
            self.il_phase_offset + worker_il_end - self.tile_il.pad_before,
            self.max_preload,
        )
        preload = PreloadedSubcube(
            read_fn=lambda primary_start, primary_count, spatial_start, spatial_count, sample_start, sample_count: (
                read_inline_slab(
                    f,
                    primary_start,
                    primary_count,
                    xline_start=spatial_start,
                    xline_count=spatial_count,
                    sample_start=sample_start,
                    sample_count=sample_count,
                )
            ),
            total_primary=self.n_il,
            total_spatial=self.n_xl,
            total_sample=self.n_s,
            max_preload=self.max_preload,
            primary_lower=primary_lower,
            primary_upper=primary_upper,
        )
        for il_tile_idx in range(worker_start, worker_end):
            il_start = self.tile_il.starts[il_tile_idx]
            slab = np.zeros((self.crop_size, self.padded_xl, self.padded_t), dtype=np.float32)
            source_il_start = self.il_phase_offset + il_start - self.tile_il.pad_before
            source_il_end = source_il_start + self.crop_size
            read_start = max(source_il_start, 0)
            read_end = min(source_il_end, self.n_il)
            if read_end > read_start:
                raw_slab = preload.fetch(
                    read_start,
                    read_end,
                    self.xl_phase_offset,
                    self.xl_phase_offset + self.xl_axis_length,
                    self.t_phase_offset,
                    self.t_phase_offset + self.t_axis_length,
                )
                dst_il_start = read_start - source_il_start
                slab[
                    dst_il_start : dst_il_start + raw_slab.shape[0],
                    self.tile_xl.pad_before : self.tile_xl.pad_before + raw_slab.shape[1],
                    self.tile_t.pad_before : self.tile_t.pad_before + raw_slab.shape[2],
                ] = raw_slab

            pixel_values, packed_meta, batch_splits = self._pack_generated_crops(
                (
                    (
                        slab[
                            : self.crop_size, xl_start : xl_start + self.crop_size, t_start : t_start + self.crop_size
                        ],
                        (il_tile_idx, xl_tile_idx, t_tile_idx),
                    )
                    for xl_tile_idx, xl_start in enumerate(self.tile_xl.starts)
                    for t_tile_idx, t_start in enumerate(self.tile_t.starts)
                ),
                self.worker_batch_size,
            )

            yield {
                "pixel_values": pixel_values,
                "meta": packed_meta,
                "batch_splits": batch_splits,
                "tile_index": il_tile_idx,
            }


class TwoDPositionDataset(_BaseInferenceDataset):
    """Yield preprocessed 2D tile batches for one phase pair."""

    def __init__(
        self,
        input_path: str | Path,
        processor_config: ProcessorConfig,
        direction: str,
        patch_size: int,
        crop_size: int,
        densify: int,
        n_primary: int,
        primary_phase_idx: int,
        primary_phase_offset: int,
        primary_count: int,
        sp_phase_idx: int,
        sp_phase_offset: int,
        t_phase_idx: int,
        t_phase_offset: int,
        tile_sp,
        sp_tail_pad: int,
        tile_t,
        t_tail_pad: int,
        worker_batch_size: int,
    ):
        super().__init__(input_path, processor_config)
        self.direction = direction
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.densify = densify
        self.n_primary = n_primary
        self.primary_phase_idx = primary_phase_idx
        self.primary_phase_offset = primary_phase_offset
        self.primary_count = primary_count
        self.sp_phase_idx = sp_phase_idx
        self.sp_phase_offset = sp_phase_offset
        self.t_phase_idx = t_phase_idx
        self.t_phase_offset = t_phase_offset
        self.tile_sp = tile_sp
        self.sp_tail_pad = sp_tail_pad
        self.tile_t = tile_t
        self.t_tail_pad = t_tail_pad
        self.worker_batch_size = clamp_worker_batch_size(worker_batch_size)

    def __iter__(self):
        f = self._get_file()
        half = self.crop_size // 2
        worker_start, worker_end = _worker_bounds(self.primary_count)
        for pp_idx in range(worker_start, worker_end):
            center_slice = min(
                self.primary_phase_offset + pp_idx * self.patch_size + self.patch_size // 2,
                self.n_primary - 1,
            )
            full_slice = _read_direction_slice(
                f,
                self.direction,
                center_slice,
                spatial_start=self.sp_phase_offset,
                sample_start=self.t_phase_offset,
            )
            slice_data = np.pad(
                full_slice,
                (
                    (half + self.tile_sp.pad_before, half + self.tile_sp.pad_after + self.sp_tail_pad),
                    (half + self.tile_t.pad_before, half + self.tile_t.pad_after + self.t_tail_pad),
                ),
                mode="constant",
            )

            def iter_crops():
                for sp_tile_idx, sp_start in enumerate(self.tile_sp.starts):
                    for t_tile_idx, t_start in enumerate(self.tile_t.starts):
                        crop = slice_data[sp_start : sp_start + self.crop_size, t_start : t_start + self.crop_size]
                        if crop.shape[0] < self.crop_size:
                            crop = np.pad(crop, ((0, self.crop_size - crop.shape[0]), (0, 0)), mode="constant")
                        if crop.shape[1] < self.crop_size:
                            crop = np.pad(crop, ((0, 0), (0, self.crop_size - crop.shape[1])), mode="constant")
                        yield crop, (sp_tile_idx, t_tile_idx)

            pixel_values, packed_meta, batch_splits = self._pack_generated_crops(iter_crops(), self.worker_batch_size)

            yield {
                "pixel_values": pixel_values,
                "meta": packed_meta,
                "batch_splits": batch_splits,
                "primary_index": self.primary_phase_idx + self.densify * pp_idx,
                "sp_phase_idx": self.sp_phase_idx,
                "t_phase_idx": self.t_phase_idx,
            }


class TwoPointFiveDPipelineDataset(_BaseInferenceDataset):
    """Yield preprocessed 2.5D multi-view tile batches across all configured phases."""

    def __init__(
        self,
        input_path: str | Path,
        processor_config: ProcessorConfig,
        direction: str,
        input_views: list[str],
        patch_size: int,
        crop_size: int,
        densify: int,
        n_primary: int,
        n_spatial: int,
        n_samples: int,
        primary_phase_configs: list[dict[str, Any]],
        worker_batch_size: int,
        max_preload: int,
    ):
        super().__init__(input_path, processor_config)
        self.direction = direction
        self.input_views = input_views
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.densify = densify
        self.n_primary = n_primary
        self.n_spatial = n_spatial
        self.n_samples = n_samples
        self.primary_phase_configs = primary_phase_configs
        self.worker_batch_size = clamp_worker_batch_size(worker_batch_size)
        self.max_preload = max_preload
        self.primary_preload = shape_aware_25d_primary_preload(crop_size, patch_size, max_preload)

    def __iter__(self):
        f = self._get_file()
        half = self.crop_size // 2
        for primary_phase in self.primary_phase_configs:
            primary_phase_idx = primary_phase["primary_phase_idx"]
            primary_phase_offset = primary_phase["primary_phase_offset"]
            primary_count = primary_phase["primary_count"]
            worker_start, worker_end = _worker_bounds(primary_count)
            if worker_start >= worker_end:
                continue
            first_worker_primary = max(
                primary_phase_offset + worker_start * self.patch_size + self.patch_size // 2 - half, 0
            )
            last_worker_primary = min(
                primary_phase_offset + (worker_end - 1) * self.patch_size + self.patch_size // 2 + half,
                self.n_primary,
            )
            primary_lower, primary_upper = expand_worker_preload_bounds(
                self.n_primary,
                first_worker_primary,
                last_worker_primary,
                self.max_preload,
            )
            preload = PreloadedSubcube(
                read_fn=lambda primary_start, primary_count, spatial_start, spatial_count, sample_start, sample_count: (
                    _read_direction_slab(
                        f,
                        self.direction,
                        primary_start,
                        primary_count,
                        spatial_start=spatial_start,
                        spatial_count=spatial_count,
                        sample_start=sample_start,
                        sample_count=sample_count,
                    )
                ),
                total_primary=self.n_primary,
                total_spatial=self.n_spatial,
                total_sample=self.n_samples,
                max_preload=self.max_preload,
                primary_lower=primary_lower,
                primary_upper=primary_upper,
            )

            for phase in primary_phase["phase_configs"]:
                tile_sp = phase["tile_sp"]
                tile_t = phase["tile_t"]
                sp_phase_idx = phase["sp_phase_idx"]
                sp_phase_offset = phase["sp_phase_offset"]
                sp_axis_length = phase["sp_axis_length"]
                t_phase_idx = phase["t_phase_idx"]
                t_phase_offset = phase["t_phase_offset"]
                t_axis_length = phase["t_axis_length"]

                for pp_idx in range(worker_start, worker_end):
                    primary_idx = min(
                        primary_phase_offset + pp_idx * self.patch_size + self.patch_size // 2,
                        self.n_primary - 1,
                    )
                    source_start = primary_idx - half
                    source_end = source_start + self.crop_size
                    read_start = max(source_start, 0)
                    read_end = min(source_end, self.n_primary)
                    if read_end > read_start:
                        cache_data, cache_bounds = preload.fetch_context(
                            read_start,
                            read_end,
                            sp_phase_offset,
                            sp_phase_offset + sp_axis_length,
                            t_phase_offset,
                            t_phase_offset + t_axis_length,
                            primary_max_preload=self.primary_preload,
                        )
                    else:
                        cache_data = np.empty((0, 0, 0), dtype=np.float32)
                        cache_bounds = (0, 0, 0, 0, 0, 0)

                    def iter_crops():
                        for sp_tile_idx, sp_start in enumerate(tile_sp.starts):
                            crop_spatial_start = sp_phase_offset + sp_start - tile_sp.pad_before - half
                            for t_tile_idx, t_start in enumerate(tile_t.starts):
                                crop_time_start = t_phase_offset + t_start - tile_t.pad_before - half
                                yield (
                                    extract_2d_crop_views_from_cache(
                                        cache_data=cache_data,
                                        cache_primary_bounds=(cache_bounds[0], cache_bounds[1]),
                                        cache_spatial_bounds=(cache_bounds[2], cache_bounds[3]),
                                        cache_time_bounds=(cache_bounds[4], cache_bounds[5]),
                                        center_primary=primary_idx,
                                        spatial_start=crop_spatial_start,
                                        time_start=crop_time_start,
                                        crop_size=self.crop_size,
                                        views=self.input_views,
                                        traversal_direction=self.direction,
                                        valid_primary_bounds=(0, self.n_primary),
                                        valid_spatial_bounds=(sp_phase_offset, sp_phase_offset + sp_axis_length),
                                        valid_time_bounds=(t_phase_offset, t_phase_offset + t_axis_length),
                                    ),
                                    (sp_tile_idx, t_tile_idx),
                                )

                    pixel_values, packed_meta, batch_splits = self._pack_generated_crops(
                        iter_crops(), self.worker_batch_size
                    )

                    yield {
                        "pixel_values": pixel_values,
                        "meta": packed_meta,
                        "batch_splits": batch_splits,
                        "primary_index": primary_phase_idx + self.densify * pp_idx,
                        "primary_phase_idx": primary_phase_idx,
                        "sp_phase_idx": sp_phase_idx,
                        "t_phase_idx": t_phase_idx,
                    }


def clamp_prefetch_factor(prefetch_factor: int) -> int:
    """Validate and normalize DataLoader prefetch factor."""
    return max(prefetch_factor, 1)
