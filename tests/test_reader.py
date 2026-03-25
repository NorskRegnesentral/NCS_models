"""Tests for seismic I/O reader utilities."""

from pathlib import Path

import numpy as np
import pytest

from NCS.inference.reader import compute_cube_stats, cube_shape, open_seismic, read_inline_slab, read_xline_slab


TEST_DATA = Path(__file__).parent / "test_data"
SMALL_SGZ = TEST_DATA / "small_2bit_5x5x50.sgz"


@pytest.fixture
def sgz_file():
    """Open the small test SGZ file."""
    f = open_seismic(SMALL_SGZ)
    yield f
    f.close()


class TestOpenSeismic:
    def test_open_sgz(self, sgz_file):
        assert hasattr(sgz_file, "ilines")
        assert hasattr(sgz_file, "xlines")
        assert hasattr(sgz_file, "samples")

    def test_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported"):
            open_seismic("test.xyz")


class TestCubeShape:
    def test_shape(self, sgz_file):
        shape = cube_shape(sgz_file)
        assert len(shape) == 3
        assert shape == (5, 5, 50)
        assert all(s > 0 for s in shape)


class TestReadInlineSlab:
    def test_basic_read(self, sgz_file):
        slab = read_inline_slab(sgz_file, start_idx=0, count=2)
        n_il, n_xl, n_s = cube_shape(sgz_file)
        assert slab.shape == (2, n_xl, n_s)
        assert slab.dtype == np.float32

    def test_clamp_at_boundary(self, sgz_file):
        n_il, _, _ = cube_shape(sgz_file)
        slab = read_inline_slab(sgz_file, start_idx=n_il - 1, count=10)
        assert slab.shape[0] == 1  # only 1 slice available

    def test_read_all(self, sgz_file):
        n_il, n_xl, n_s = cube_shape(sgz_file)
        slab = read_inline_slab(sgz_file, start_idx=0, count=n_il)
        assert slab.shape == (n_il, n_xl, n_s)

    def test_matches_inline_accessor(self, sgz_file):
        slab = read_inline_slab(sgz_file, start_idx=1, count=2)
        expected = np.stack(
            [np.asarray(sgz_file.iline[int(sgz_file.ilines[idx])], dtype=np.float32) for idx in range(1, 3)],
            axis=0,
        )
        np.testing.assert_allclose(slab, expected)

    def test_windowed_inline_read_matches_accessor_slice(self, sgz_file):
        slab = read_inline_slab(
            sgz_file, start_idx=1, count=2, xline_start=1, xline_count=3, sample_start=5, sample_count=10
        )
        expected = np.stack(
            [
                np.asarray(sgz_file.iline[int(sgz_file.ilines[idx])], dtype=np.float32)[1:4, 5:15]
                for idx in range(1, 3)
            ],
            axis=0,
        )
        assert slab.shape == (2, 3, 10)
        np.testing.assert_allclose(slab, expected)


class TestReadXlineSlab:
    def test_basic_read(self, sgz_file):
        slab = read_xline_slab(sgz_file, start_idx=0, count=2)
        n_il, n_xl, n_s = cube_shape(sgz_file)
        assert slab.shape == (2, n_il, n_s)
        assert slab.dtype == np.float32

    def test_matches_xline_accessor(self, sgz_file):
        slab = read_xline_slab(sgz_file, start_idx=1, count=2)
        expected = np.stack(
            [np.asarray(sgz_file.xline[int(sgz_file.xlines[idx])], dtype=np.float32) for idx in range(1, 3)],
            axis=0,
        )
        np.testing.assert_allclose(slab, expected)

    def test_windowed_xline_read_matches_accessor_slice(self, sgz_file):
        slab = read_xline_slab(
            sgz_file, start_idx=1, count=2, inline_start=1, inline_count=3, sample_start=5, sample_count=10
        )
        expected = np.stack(
            [
                np.asarray(sgz_file.xline[int(sgz_file.xlines[idx])], dtype=np.float32)[1:4, 5:15]
                for idx in range(1, 3)
            ],
            axis=0,
        )
        assert slab.shape == (2, 3, 10)
        np.testing.assert_allclose(slab, expected)


class TestComputeCubeStats:
    def test_default_n_traces_reads_at_least_one_trace_for_small_volume(self):
        class _FakeSeismicFile:
            tracecount = 5
            trace = [np.arange(4, dtype=np.float32) + idx for idx in range(tracecount)]

        mean, std = compute_cube_stats(_FakeSeismicFile())
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std >= 0

    def test_stats_reasonable(self, sgz_file):
        mean, std = compute_cube_stats(sgz_file, n_traces=100)
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std >= 0

    def test_default_n_traces(self, sgz_file):
        """Default n_traces=None uses 10% of total traces."""
        mean, std = compute_cube_stats(sgz_file)
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std >= 0

    def test_deterministic(self, sgz_file):
        m1, s1 = compute_cube_stats(sgz_file, n_traces=10, seed=42)
        m2, s2 = compute_cube_stats(sgz_file, n_traces=10, seed=42)
        assert m1 == m2
        assert s1 == s2

    def test_read_all_traces(self, sgz_file):
        n_il, n_xl, _ = cube_shape(sgz_file)
        total = n_il * n_xl
        mean, std = compute_cube_stats(sgz_file, n_traces=total + 100)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    @pytest.mark.parametrize("n_traces", [0, -1])
    def test_invalid_n_traces_raises(self, sgz_file, n_traces):
        with pytest.raises(ValueError, match="n_traces must be positive"):
            compute_cube_stats(sgz_file, n_traces=n_traces)
