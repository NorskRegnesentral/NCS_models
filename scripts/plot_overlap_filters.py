"""Generate overlap filter profile plots.

Example:
    UV_NO_PROGRESS=1 uv run --group dev python scripts/plot_overlap_filters.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from NCS.inference.tiling import (
    OVERLAP_FILTER_CHOICES,
    compute_patch_overlap_weights,
    compute_tile_patch_overlap_weights,
)


def _parse_overlap_counts(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _output_name(profile: str, overlap_count: int, default_overlap_count: int) -> str:
    if overlap_count == default_overlap_count:
        return f"{profile}.png"
    return f"{profile}_overlap{overlap_count}.png"


def _default_num_tiles(patches_per_crop: int, overlap_count: int) -> int:
    stride = patches_per_crop - overlap_count
    extra_tiles = (patches_per_crop - 1 + stride - 1) // stride
    return 2 * extra_tiles + 1


def _plot_overlap_profiles(
    output_dir: Path,
    crop_size: int,
    patch_size: int,
    overlap_counts: list[int],
    num_tiles: int | None,
) -> None:
    patches_per_crop = crop_size // patch_size
    default_overlap_count = overlap_counts[0]
    patch_indices = np.arange(patches_per_crop)

    for overlap_count in overlap_counts:
        stride = patches_per_crop - overlap_count
        effective_num_tiles = num_tiles or _default_num_tiles(patches_per_crop, overlap_count)
        interior_tile_idx = effective_num_tiles // 2
        for profile in OVERLAP_FILTER_CHOICES:
            raw = compute_patch_overlap_weights(patches_per_crop, overlap_count, profile)
            weights = compute_tile_patch_overlap_weights(effective_num_tiles, patches_per_crop, overlap_count, profile)
            representative = weights[interior_tile_idx]
            accumulated = np.zeros(patches_per_crop + stride * (effective_num_tiles - 1), dtype=np.float32)
            for tile_idx in range(effective_num_tiles):
                start = tile_idx * stride
                accumulated[start : start + patches_per_crop] += weights[tile_idx]

            fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True, height_ratios=(1.3, 3, 1.2))
            ax_raw, ax_norm, ax_sum = axes

            ax_raw.step(patch_indices, raw, where="mid", color="#111111", linewidth=2.2)
            ax_raw.scatter(patch_indices, raw, s=22, color="#111111")
            ax_raw.set_title(
                f"{profile} overlap profile\n"
                f"crop_size={crop_size}, patch_size={patch_size}, num_overlap_patches={overlap_count}"
            )
            ax_raw.set_ylabel("raw")
            ax_raw.set_xticks(patch_indices)
            ax_raw.set_ylim(-0.05, 1.05)
            ax_raw.grid(True, axis="y", alpha=0.3)

            ax_norm.axvspan(-0.5, overlap_count - 0.5, color="#d9d9d9", alpha=0.25)
            ax_norm.axvspan(
                patches_per_crop - overlap_count - 0.5, patches_per_crop - 0.5, color="#d9d9d9", alpha=0.25
            )
            ax_norm.step(
                patch_indices,
                representative,
                where="mid",
                label="representative interior tile",
                linewidth=2.2,
                color="#ff7f0e",
            )
            ax_norm.scatter(patch_indices, representative, s=24, color="#ff7f0e", zorder=3)
            ax_norm.set_ylabel("normalized")
            ax_norm.set_xticks(patch_indices)
            ax_norm.set_ylim(-0.05, 1.05)
            ax_norm.grid(True, axis="y", alpha=0.3)
            ax_norm.legend(frameon=False, loc="lower center")

            ax_sum.step(np.arange(accumulated.size), accumulated, where="mid", color="#444444", linewidth=2)
            ax_sum.scatter(np.arange(accumulated.size), accumulated, s=18, color="#444444")
            ax_sum.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.5)
            ax_sum.set_ylabel("sum")
            ax_sum.set_xlabel("Patch index")
            ax_sum.set_ylim(0.0, 1.1)
            ax_sum.grid(True, axis="y", alpha=0.3)

            output_path = output_dir / _output_name(profile, overlap_count, default_overlap_count)
            fig.savefig(output_path, dpi=160)
            plt.close(fig)
            print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw and normalized overlap filter profiles.")
    parser.add_argument("--crop-size", type=int, default=224, help="Crop size in samples.")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size in samples.")
    parser.add_argument(
        "--num-overlap-patches",
        default="7,13",
        help="Comma-separated overlap counts to plot. The first count uses the base '<profile>.png' filename.",
    )
    parser.add_argument(
        "--num-tiles",
        type=int,
        default=None,
        help="Number of tiles to use when computing normalized weights. Defaults to enough tiles for a true interior profile.",
    )
    parser.add_argument("--output-dir", default="test_imgs", help="Directory where the PNGs will be written.")
    args = parser.parse_args()

    if args.crop_size % args.patch_size != 0:
        raise ValueError("--crop-size must be divisible by --patch-size")

    overlap_counts = _parse_overlap_counts(args.num_overlap_patches)
    if not overlap_counts:
        raise ValueError("At least one overlap count must be provided")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_overlap_profiles(output_dir, args.crop_size, args.patch_size, overlap_counts, args.num_tiles)


if __name__ == "__main__":
    main()
