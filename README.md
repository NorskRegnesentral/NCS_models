# NCS foundation models

[![arXiv](https://img.shields.io/badge/arXiv-2603.23211-b31b1b.svg)](https://arxiv.org/abs/2603.23211)
[![Model: 2D](https://img.shields.io/badge/%F0%9F%A4%97-NCS--v1--2d--base-yellow)](https://huggingface.co/NorskRegnesentralSTI/NCS-v1-2d-base)
[![Model: 2.5D](https://img.shields.io/badge/%F0%9F%A4%97-NCS--v1--2.5d--base-yellow)](https://huggingface.co/NorskRegnesentralSTI/NCS-v1-2.5d-base)
[![Model: 3D](https://img.shields.io/badge/%F0%9F%A4%97-NCS--v1--3d--base-yellow)](https://huggingface.co/NorskRegnesentralSTI/NCS-v1-3d-base)

Repository for the NCS foundation models.
Contains model code for 2.5D and 3D ViT models, as well as an inference pipeline for running seismic foundation models on seismic data.

## Installation

### Setup (uv)

1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone and install package
```bash
git clone https://github.com/NorskRegnesentral/NCS_models
cd NCS_models
uv sync
```
For model code and a inference pipeline without optional features:
```bash
uv sync --extra inference
```

For all optional features:
```bash
uv sync --extra all
```

To select a specific CUDA version for PyTorch, use the corresponding extra. For example, for CUDA 12.8:
```bash
uv sync --extra cu128
```
`cu128` and `cu130` extras are supported.

### Setup (pip)
1. Create a new environment, e.g. by using the following command (for linux based systems):
```
python -m virtualenv -p PATH_TO_PYTHON_BINARY PATH_TO_ENV
```
2. Clone and install package
```bash
source activate PATH_TO_ENV
git clone https://github.com/NorskRegnesentral/NCS_models

cd NCS_models
pip install -e ".[all]"
```

3. To get CUDA-enabled PyTorch, install with the appropriate extra and index URL. Choose exactly one CUDA version:

For CUDA 12.8:
```bash
pip install -e ".[all,cu128]" --extra-index-url https://download.pytorch.org/whl/cu128
```

For CUDA 13.0:
```bash
pip install -e ".[all,cu130]" --extra-index-url https://download.pytorch.org/whl/cu130
```

Without this step, pip installs CPU-only PyTorch from PyPI.

## Inference

The inference pipeline reads a seismic volume (SEG-Y or SGZ), runs a ViT model over tiled crops with overlap blending, and writes a patch-space feature cube to Zarr or NetCDF.

For GPU inference, the CLI prepares crops through a `torch.utils.data.DataLoader` so host-side reads and preprocessing can overlap with model execution. Internal array writes are also dispatched through a background writer so output persistence can overlap with compute. By default, `--num-workers` uses the available CPU count; tune `--num-workers`, `--prefetch-factor`, `--max-preload`, and `--no-pin-memory` if needed. `--max-preload` controls the worker-local reusable preload cache so nearby positions can reuse larger reads without letting each worker expand unbounded.

The model type (2D, 2.5D, or 3D) is detected automatically from the model config.

### Quick start

Download seismic data from the [F3 demo dataset](https://drive.google.com/drive/folders/0B7brcf-eGK8CbGhBdmZoUnhiTWs?resourcekey=0-0ZhV_OJ3TKN1ShFAGcrOzQ&usp=drive_link) and run inference on a pretrained model:

```bash
uv run scripts/inference.py \
  --model-path NorskRegnesentralSTI/NCS-v1-2d-base \
  --input-path /path/to/volume.segy \
  --output-path ./features.zarr \
  --direction dir0 \
  --densify 1 \
  --num-overlap-patches 7 \
  --overlap-filter ramp \
  --device cuda \
  --batch-size 8 \
  --verbose
```

### Examples by model type

**3D ViT:**
```bash
uv run scripts/inference.py \
  --model-path NorskRegnesentralSTI/NCS-v1-3d-base \
  --input-path /data/volume.segy \
  --output-path ./features_3d.zarr \
  --direction dir0 \
  --densify 1 \
  --num-overlap-patches 7 \
  --overlap-filter ramp \
  --batch-size 4 \
  --device cuda:0 \
  --dtype float16 \
  --verbose
```

**2.5D ViT:**
```bash
uv run scripts/inference.py \
  --model-path NorskRegnesentralSTI/NCS-v1-2.5d-base \
  --input-path /data/volume.segy \
  --output-path ./features_25d.zarr \
  --direction dir0 \
  --densify 1 \
  --num-overlap-patches 7 \
  --overlap-filter ramp \
  --batch-size 32 \
  --device cuda:0 \
  --dtype float16 \
  --verbose
```

By default, all views from the model config are used. To override which views the 2.5D model processes, pass
`--input-views`:
```bash
  --input-views dir0,dir90
```

**2D ViT:**
```bash
uv run scripts/inference.py \
  --model-path NorskRegnesentralSTI/NCS-v1-2d-base \
  --input-path /data/volume.segy \
  --output-path ./features_2d.zarr \
  --direction dir0 \
  --densify 1 \
  --num-overlap-patches 7 \
  --overlap-filter ramp \
  --batch-size 64 \
  --device cuda:0 \
  --dtype float16 \
  --verbose
```

To regenerate the overlap profile figures in `test_imgs/`:
```bash
UV_NO_PROGRESS=1 uv run --group dev python scripts/plot_overlap_filters.py
```

### CLI reference

| Argument | Default | Description |
|---|---|---|
| `--model-path` | *(required)* | Local pretrained model directory or Hugging Face model ID. |
| `--input-path` | *(required)* | Path to input seismic file (`.segy` or `.sgz`). |
| `--output-path` | *(required)* | Output path (`.zarr` or `.nc`). The command fails if it already exists unless `--force` is passed. |
| `--direction` | `dir0` | Primary traversal direction: `dir0` (inline) or `dir90` (crossline). |
| `--input-views` | model config | Comma-separated view names for 2.5D models (e.g. `dir0,dir90,dir45,dir135`). |
| `--crop-size` | `224` | Crop size in samples. Must be divisible by `--patch-size`. |
| `--patch-size` | `16` | Patch size in samples. |
| `--densify` | `1` | Output densification factor: `1`, `2`, `4`, `8`, or `16`. The output step becomes `patch_size / densify`; with the default patch size, `16` yields one output per input sample. |
| `--overlap-filter` | `ramp` | Overlap weighting mode: `exponential`, `ramp`, `uniform`, or `center`. All modes are normalized so tiled weights sum to 1 per patch. `center` assigns each patch to the covering tile whose center is closest. |
| `--num-overlap-patches` | `7` | Number of overlapping patches between adjacent tiles. Higher values give smoother results but are slower. |
| `--batch-size` | `8` | Batch size for model forward passes and worker-side preprocessing chunks. |
| `--num-workers` | available CPU count | Torch DataLoader worker count for asynchronous input preparation. |
| `--prefetch-factor` | `2` | Number of prefetched work units per DataLoader worker. |
| `--max-preload` | `500` | Per-axis cap for the worker-local preload cache used to reuse larger reads across nearby positions while limiting overlap between workers. |
| `--no-pad-before` | off | Anchor tile coverage at the start of each axis instead of splitting extra padding before and after. |
| `--device` | `cuda` | PyTorch device (e.g. `cuda`, `cuda:0`, `cpu`). |
| `--no-pin-memory` | off | Disable pinned host memory for DataLoader batches. |
| `--force` | off | Remove an existing output path before writing new results. |
| `--dtype` | `float32` | Compute dtype: `float32`, `float16`, or `bfloat16`. |
| `--clip-sigma` | `3.0` | Sigma clipping value after normalization. |
| `--no-normalize` | off | Disable z-score normalization. |
| `--normalization-mode` | `std` | Normalize by cube standard deviation only (`std`) or by cube mean and standard deviation (`zscore`). |
| `--n-stat-traces` | 10% of traces, min 100 | Number of traces to sample for mean/std estimation. |
| `-v`, `--verbose` | off | Enable verbose (DEBUG) logging. |

### Output format

The output is an `xarray.Dataset` saved as Zarr (`.zarr`) or NetCDF (`.nc`).

**Feature cube:** The data is stored under groups named <direction> (e.g. `dir0/`), with the feature variable being `features` (e.g. `dir0/features`). Shape is `(inline, xline, time_depth, feature)`. With `--densify 1`, the non-feature dimensions are in patch space as before. Higher densify values interleave sliding-window passes so the output spacing becomes `patch_size / densify` samples on every non-feature axis.

**Channel statistics:** Per-channel mean and standard deviation computed over all spatial positions are also stored under the same group  as `<direction>/mean` / `<direction>/std`, each a 1-D array of length `hidden_size`.

**Coordinates:** `inline`, `xline`, and `time_depth` coordinates give the center sample position of each patch.

Example of reading the output in Python:
```python
import xarray as xr

ds = xr.open_zarr("features.zarr", group="dir0")
print(ds)

# Access feature cube
features = ds["features"].values  # (n_il_patches, n_xl_patches, n_t_patches, hidden_size)

# Access channel statistics
channel_mean = ds["mean"].values  # (hidden_size,)
channel_std = ds["std"].values    # (hidden_size,)
```

For safety, the CLI refuses to write if `--output-path` already exists. Pass `--force` to remove the existing file/store before inference starts.

### Python API

The pipeline can also be called directly from Python:
```python
from NCS.inference import run_inference

ds = run_inference(
    model_path="NorskRegnesentralSTI/NCS-v1-2d-base",
    input_path="/data/volume.segy",
    output_path="./features.zarr",
    direction="dir0",
    densify=1,
    num_overlap_patches=7,
    overlap_filter="ramp",
    batch_size=32,
    device="cuda:0",
    dtype="float16",
)
```

An end-to-end walkthrough covering inference, output loading, and cosine-similarity visualization is available in
[`notebooks/visualization.ipynb`](notebooks/visualization.ipynb).

## Citation

If you use this code or the pretrained models, please cite:

```bibtex
@article{ordonez2025ncsmodel,
  title={The {NCS}-model: A seismic foundation model trained on the Norwegian repository of public seismic data},
  author={Ordo{\~n}ez, Alba and Forgaard, Theodor Johannes Line and Wade, David and Bugge, Aina Juell and Nese, H{\aa}kon and Waldeland, Anders Ueland},
  journal={arXiv preprint arXiv:2603.23211},
  year={2025}
}
```

## Acknowledgements

This work is funded by The Research Council of Norway through the SFI Visual Intelligence (Centre for Research-based Innovation), grant no. 309439, and the industry partners Equinor ASA and AkerBP ASA. We also thank Equinor and AkerBP for providing access to the seismic data used in the evaluation.

## License

Except where otherwise noted, this repository is licensed under the Apache License 2.0. See
[`LICENSE`](LICENSE) for the full license text.

The 3D ViT model uses [LieRE](https://github.com/StanfordMIMI/LieRE) positional encodings (MIT license, Stanford MIMI). The ViT model code is derived from [HuggingFace Transformers](https://github.com/huggingface/transformers) (Apache 2.0). See [NOTICE](NOTICE) for attribution details.
