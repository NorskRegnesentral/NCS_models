import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from scripts.inference import _prepare_output_path


def test_prepare_output_path_rejects_existing_file(tmp_path):
    output_path = tmp_path / "features.zarr"
    output_path.write_text("existing")

    with pytest.raises(FileExistsError, match="already exists"):
        _prepare_output_path(output_path, force=False)


def test_prepare_output_path_rejects_existing_directory(tmp_path):
    output_path = tmp_path / "features.zarr"
    output_path.mkdir()
    (output_path / "marker").write_text("existing")

    with pytest.raises(FileExistsError, match="already exists"):
        _prepare_output_path(output_path, force=False)


def test_prepare_output_path_force_removes_existing_file(tmp_path):
    output_path = tmp_path / "features.nc"
    output_path.write_text("existing")

    resolved = _prepare_output_path(output_path, force=True)

    assert resolved == output_path
    assert not output_path.exists()


def test_prepare_output_path_force_removes_existing_directory(tmp_path):
    output_path = tmp_path / "features.zarr"
    output_path.mkdir()
    nested = output_path / "group" / "array"
    nested.mkdir(parents=True)
    (nested / ".zarray").write_text("{}")

    resolved = _prepare_output_path(output_path, force=True)

    assert resolved == output_path
    assert not output_path.exists()
