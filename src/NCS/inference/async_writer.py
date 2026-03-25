"""Single-threaded asynchronous write helpers for inference outputs."""

from __future__ import annotations

from collections.abc import Callable
from queue import Queue
from threading import Thread
from typing import Any


WriteTask = tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]


class _StopSignal:
    """Sentinel used to stop the writer thread."""


class AsyncWriter:
    """Run write operations on a dedicated background thread."""

    _STOP = _StopSignal()

    def __init__(self, maxsize: int = 32):
        self._queue: Queue[WriteTask | _StopSignal] = Queue(maxsize=maxsize)
        self._error: BaseException | None = None
        self._thread = Thread(target=self._run, name="ncs-async-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if isinstance(item, _StopSignal):
                    return
                fn, args, kwargs = item
                fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
            finally:
                self._queue.task_done()

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Asynchronous writer failed.") from self._error

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._raise_if_error()
        self._queue.put((fn, args, kwargs))

    def barrier(self) -> None:
        self._queue.join()
        self._raise_if_error()

    def close(self) -> None:
        self.barrier()
        self._queue.put(self._STOP)
        self._queue.join()
        self._thread.join()
        self._raise_if_error()

    def __enter__(self) -> AsyncWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_array_slice(dest, key, value) -> None:
    """Assign a value into an array-like destination."""
    dest[key] = value
