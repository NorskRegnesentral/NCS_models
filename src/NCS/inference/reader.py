"""Seismic I/O utilities for segyio-like file objects."""

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from segyio import SegyFile


if TYPE_CHECKING:
    from seismic_zfp.segyio_emulator import SegyioEmulator


# Type alias — SegyFile-like object with .ilines, .xlines, .samples, .iline[], .xline[], .trace[]
SegyLike = Union[SegyFile, "SegyioEmulator"]


def open_seismic(path: str | Path) -> SegyLike:
    """Open a seismic file, auto-detecting format from extension.

    Returns a segyio.SegyFile or seismic_zfp.SegyioEmulator context manager.
    Caller must close it (or use as context manager).
    """
    suffix = Path(path).suffix.lower()
    if suffix in (".sgy", ".segy"):
        import segyio

        return segyio.open(str(path), mode="r", strict=False)
    elif suffix in (".sgz", ".zfp"):
        try:
            import seismic_zfp
        except ImportError as e:
            raise ImportError("seismic_zfp is required to read .sgz/.zfp files.") from e

        return seismic_zfp.open(str(path))
    else:
        raise ValueError(f"Unsupported seismic file extension: {suffix}")


def cube_shape(f: SegyLike) -> tuple[int, int, int]:
    """Return ``(n_ilines, n_xlines, n_samples)`` for a segyio-like file."""
    return (len(f.ilines), len(f.xlines), len(f.samples))


def _read_subvolume_slab(
    f: SegyLike,
    min_il: int,
    max_il: int,
    min_xl: int,
    max_xl: int,
    min_sample: int,
    max_sample: int,
) -> np.ndarray | None:
    """Use the SGZ bulk subvolume API when available."""
    read_subvolume = getattr(f, "read_subvolume", None)
    if read_subvolume is None:
        return None
    return np.asarray(
        read_subvolume(min_il, max_il, min_xl, max_xl, min_sample, max_sample),
        dtype=np.float32,
    )


def read_inline_slab(
    f: SegyLike,
    start_idx: int,
    count: int,
    xline_start: int = 0,
    xline_count: int | None = None,
    sample_start: int = 0,
    sample_count: int | None = None,
) -> np.ndarray:
    """Read contiguous inline slices by 0-based index.

    Returns array of shape ``(count, n_xlines, n_samples)`` in seismic order, float32.
    Reads fewer slices if ``start_idx + count`` exceeds ``n_ilines``.
    """
    n_il = len(f.ilines)
    end_idx = min(start_idx + count, n_il)
    actual = end_idx - start_idx
    n_xl = len(f.xlines)
    n_s = len(f.samples)
    xline_end = n_xl if xline_count is None else min(xline_start + xline_count, n_xl)
    sample_end = n_s if sample_count is None else min(sample_start + sample_count, n_s)
    fast_slab = _read_subvolume_slab(f, start_idx, end_idx, xline_start, xline_end, sample_start, sample_end)
    if fast_slab is not None:
        return fast_slab

    slab = np.empty((actual, xline_end - xline_start, sample_end - sample_start), dtype=np.float32)
    for i in range(actual):
        inline = np.asarray(f.iline[int(f.ilines[start_idx + i])], dtype=np.float32)
        slab[i] = inline[xline_start:xline_end, sample_start:sample_end]
    return slab


def read_xline_slab(
    f: SegyLike,
    start_idx: int,
    count: int,
    inline_start: int = 0,
    inline_count: int | None = None,
    sample_start: int = 0,
    sample_count: int | None = None,
) -> np.ndarray:
    """Read contiguous crossline slices by 0-based index.

    Returns array of shape ``(count, n_ilines, n_samples)`` in ``(xline, inline, time)``
    order, float32.  Reads fewer slices if ``start_idx + count`` exceeds ``n_xlines``.
    """
    n_xl = len(f.xlines)
    end_idx = min(start_idx + count, n_xl)
    actual = end_idx - start_idx
    n_il = len(f.ilines)
    n_s = len(f.samples)
    inline_end = n_il if inline_count is None else min(inline_start + inline_count, n_il)
    sample_end = n_s if sample_count is None else min(sample_start + sample_count, n_s)
    fast_slab = _read_subvolume_slab(f, inline_start, inline_end, start_idx, end_idx, sample_start, sample_end)
    if fast_slab is not None:
        return np.ascontiguousarray(fast_slab.transpose(1, 0, 2))

    slab = np.empty((actual, inline_end - inline_start, sample_end - sample_start), dtype=np.float32)
    for i in range(actual):
        xline = np.asarray(f.xline[int(f.xlines[start_idx + i])], dtype=np.float32)
        slab[i] = xline[inline_start:inline_end, sample_start:sample_end]
    return slab


def compute_cube_stats(
    f: SegyLike,
    n_traces: int | None = None,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute approximate mean and std from a random subset of traces.

    Uses a streaming two-pass (sum + sum-of-squares) approach for numerical
    stability at float64 precision.

    Args:
        f: Open segyio-like file.
        n_traces: Number of traces to sample. If ``>= total``, reads all.
        seed: Random seed for reproducibility.

    Returns:
        ``(mean, std)`` tuple.
    """
    total_traces = f.tracecount
    if total_traces <= 0:
        raise ValueError("Cannot compute cube statistics for an empty seismic volume.")

    rng = np.random.default_rng(seed)

    if n_traces is None:
        n_traces = max(
            total_traces // 10, min(100, total_traces)
        )  # 10% of total traces, minimum 100 traces or total_traces if less than 100
        indices = sorted(rng.choice(total_traces, size=n_traces, replace=False))
    elif n_traces <= 0:
        raise ValueError("n_traces must be positive when provided.")
    elif n_traces >= total_traces:
        indices = range(total_traces)
    else:
        indices = sorted(rng.choice(total_traces, size=n_traces, replace=False))

    running_sum = 0.0
    running_sq_sum = 0.0
    total_samples = 0
    for idx in indices:
        trace = np.array(f.trace[int(idx)], dtype=np.float64)
        running_sum += trace.sum()
        running_sq_sum += (trace**2).sum()
        total_samples += trace.size

    if total_samples == 0:
        raise ValueError("No trace samples were selected for cube statistics.")

    mean = running_sum / total_samples
    std = np.sqrt(running_sq_sum / total_samples - mean**2)
    return float(mean), float(std)
