import ast
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _notebook_cells():
    notebook = json.loads((ROOT / "notebooks/visualization.ipynb").read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def test_standard_inference_examples_recommend_overlap_13():
    readme = (ROOT / "README.md").read_text()
    cli = (ROOT / "scripts/inference.py").read_text()
    pipeline = (ROOT / "src/NCS/inference/pipeline.py").read_text()
    notebook_cells = _notebook_cells()

    assert readme.count("--num-overlap-patches 13") == 4
    assert readme.count("# 13 overlapping patches remove edge-effects.") == 5
    assert "num_overlap_patches=13" in readme
    assert "--num-overlap-patches 13" in cli
    assert "# 13 overlapping patches remove edge-effects." in cli
    assert any(
        "NUM_OVERLAP_PATCHES = 13" in cell and "# 13 overlapping patches remove edge-effects." in cell
        for cell in notebook_cells
    )
    assert "--num-overlap-patches` | `7`" in readme
    assert 'default=7, help="Number of overlapping patches per tile."' in cli
    assert "num_overlap_patches: int = 7" in pipeline


def test_notebook_normalizes_features_per_dimension_and_handles_small_std():
    notebook_cells = _notebook_cells()
    source = next(cell for cell in notebook_cells if "def normalize_features" in cell)
    module = ast.parse(source)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "normalize_features")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook>", "exec"), namespace)

    features = np.array([[[[4.0, 7.0, 9.0, 1e-6]]], [[[6.0, 11.0, 9.0, 2e-6]]]], dtype=np.float32)
    mean = np.array([2.0, 3.0, 9.0, 0.0], dtype=np.float32)
    std = np.array([2.0, 4.0, 0.0, 1e-8], dtype=np.float32)
    normalized = namespace["normalize_features"](features, mean, std)

    np.testing.assert_allclose(normalized[..., 0], np.array([[[1.0]], [[2.0]]]))
    np.testing.assert_allclose(normalized[..., 1], np.array([[[1.0]], [[2.0]]]))
    np.testing.assert_array_equal(normalized[..., 2], 0.0)
    np.testing.assert_allclose(normalized[..., 3], np.array([[[1.0]], [[2.0]]]))
    assert np.isfinite(normalized).all()

    readme = (ROOT / "README.md").read_text()
    assert "standardized_features = (features - mu) / sigma_safe" in readme
