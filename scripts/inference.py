"""CLI for seismic feature inference.

Example:
    uv run python scripts/inference.py \
        --model-path /path/to/model \
        --input-path /data/volume.segy \
        --output-path /tmp/features.zarr \
        --crop-size 224 --patch-size 16 --densify 1 --num-overlap-patches 7 \
        --overlap-filter ramp \
        --device cpu
"""

import argparse
import logging
import shutil
from pathlib import Path

from NCS.inference import run_inference
from NCS.inference.tiling import DENSIFY_CHOICES, OVERLAP_FILTER_CHOICES


def _prepare_output_path(output_path: str | Path, force: bool) -> Path:
    path = Path(output_path)
    if path.exists():
        if not force:
            raise FileExistsError(f"Output path already exists: {path}. Remove it first or pass --force to overwrite.")
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    return path


def main():
    parser = argparse.ArgumentParser(description="Seismic feature inference using ViT foundation models.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to pretrained model directory (HuggingFace-style). Such as NorskRegnesentralSTI/NCS-v1-2.5d-base.",
    )
    parser.add_argument("--input-path", required=True, help="Path to input seismic file (.segy / .sgz).")
    parser.add_argument("--output-path", required=True, help="Output path (.zarr or .nc).")
    parser.add_argument("--direction", default="dir0", choices=["dir0", "dir90"], help="Primary traversal direction.")
    parser.add_argument(
        "--input-views",
        default=None,
        help="Comma-separated additional context view names for 2.5D ViTs (e.g. 'dir0,dir90,dir45,dir135'). Defaults to model config.",
    )
    parser.add_argument("--crop-size", type=int, default=224, help="Crop size in samples.")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size in samples.")
    parser.add_argument("--num-overlap-patches", type=int, default=7, help="Number of overlapping patches per tile.")
    parser.add_argument("--densify", type=int, default=1, choices=DENSIFY_CHOICES, help="Output densification factor.")
    parser.add_argument(
        "--overlap-filter",
        default="ramp",
        choices=OVERLAP_FILTER_CHOICES,
        help="Overlap weighting mode for blended patches.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for model forward passes.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Torch DataLoader worker count for asynchronous input preparation. Defaults to the available CPU count.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Prefetched work units per DataLoader worker.",
    )
    parser.add_argument(
        "--max-preload",
        type=int,
        default=500,
        help="Per-axis cap for the worker-local preload cache. Larger values trade memory for fewer rereads.",
    )
    parser.add_argument(
        "--no-pad-before",
        action="store_true",
        help="Anchor tile coverage at the start of each axis instead of splitting extra padding before and after.",
    )
    parser.add_argument("--device", default="cuda", help="PyTorch device (e.g. cuda, cpu).")
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pinned host memory for DataLoader batches.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output path before running.")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="PyTorch dtype."
    )
    parser.add_argument("--clip-sigma", type=float, default=3.0, help="Sigma clipping after normalization.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable seismic normalization.")
    parser.add_argument(
        "--normalization-mode",
        default="std",
        choices=["std", "zscore"],
        help="Normalization mode: divide by std only, or full z-score using mean and std.",
    )
    parser.add_argument(
        "--n-stat-traces",
        type=int,
        default=None,
        help="Traces to sample for mean/std estimation. Default, 10%% of total traces.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    try:
        output_path = _prepare_output_path(args.output_path, args.force)
    except FileExistsError as exc:
        parser.error(str(exc))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    input_views = None
    if args.input_views:
        input_views = [v.strip() for v in args.input_views.split(",")]

    run_inference(
        model_path=args.model_path,
        input_path=args.input_path,
        output_path=output_path,
        direction=args.direction,
        input_views=input_views,
        crop_size=args.crop_size,
        patch_size=args.patch_size,
        num_overlap_patches=args.num_overlap_patches,
        densify=args.densify,
        overlap_filter=args.overlap_filter,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        clip_sigma=args.clip_sigma,
        do_normalize=not args.no_normalize,
        normalization_mode=args.normalization_mode,
        n_stat_traces=args.n_stat_traces,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=False if args.no_pin_memory else None,
        max_preload=args.max_preload,
        pad_before=not args.no_pad_before,
    )


if __name__ == "__main__":
    main()
