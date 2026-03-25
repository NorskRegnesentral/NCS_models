"""Tests for asynchronous inference writing helpers."""

import numpy as np

from NCS.inference.async_writer import AsyncWriter, write_array_slice


def test_async_writer_writes_array_slice():
    dest = np.zeros((3, 3), dtype=np.float32)

    with AsyncWriter() as writer:
        writer.submit(write_array_slice, dest, (slice(1, 3), slice(0, 2)), np.ones((2, 2), dtype=np.float32))
        writer.barrier()

    np.testing.assert_array_equal(dest[1:3, 0:2], np.ones((2, 2), dtype=np.float32))
