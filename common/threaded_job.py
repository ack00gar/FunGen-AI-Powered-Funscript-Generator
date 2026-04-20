"""
BackgroundJob -- run a long-running callable on a worker thread, surface
progress events both via a user callback and a poll-able queue.

Usage:

    def my_long_task(x, y, progress_callback=None):
        for i in range(100):
            do_something(x, y)
            if progress_callback:
                progress_callback(i / 100, f"step {i}")
        return "done"

    job = BackgroundJob.run(my_long_task, 1, 2, ui_callback=ui_thread_safe_cb)
    # ... later, on the UI thread:
    for pct, msg in job.poll():
        update_status_bar(pct, msg)
    if job.is_done():
        if job.error():
            show_error(job.error())
        else:
            display(job.result())
"""

import logging
import queue
import threading
from typing import Any, Callable, List, Optional, Tuple


class BackgroundJob:
    """
    Long-running task on a worker thread. Two consumption modes:
      1. Pass `ui_callback` -- invoked on the worker thread (caller responsible
         for UI marshalling).
      2. Call `poll()` each frame from the UI thread to drain queued events.
    """

    @classmethod
    def run(cls,
            target: Callable[..., Any],
            *args: Any,
            ui_callback: Optional[Callable[[float, str], None]] = None,
            name: str = "BackgroundJob",
            progress_kwarg: str = "progress_callback",
            **kwargs: Any) -> "BackgroundJob":
        """
        Start `target(*args, **kwargs)` on a worker thread immediately.

        The job injects its own progress callback under `progress_kwarg`
        (default 'progress_callback'); the target should accept it and call it
        as `progress_callback(pct, message)`.
        """
        return cls(target, args, kwargs, ui_callback, name, progress_kwarg)

    def __init__(self,
                 target: Callable[..., Any],
                 args: Tuple[Any, ...],
                 kwargs: dict,
                 ui_callback: Optional[Callable[[float, str], None]],
                 name: str,
                 progress_kwarg: str):
        self._user_callback = ui_callback
        self._progress_queue: "queue.Queue[Tuple[float, str]]" = queue.Queue()
        self._result: Any = None
        self._error: Optional[BaseException] = None
        self._done = threading.Event()
        self._logger = logging.getLogger(__name__)

        kwargs = dict(kwargs)
        kwargs.setdefault(progress_kwarg, self._on_progress)

        self._thread = threading.Thread(
            target=self._run, args=(target, args, kwargs),
            daemon=True, name=name,
        )
        self._thread.start()

    def _on_progress(self, pct: float, message: str = "") -> None:
        try:
            self._progress_queue.put_nowait((pct, message))
        except queue.Full:
            pass
        if self._user_callback is not None:
            try:
                self._user_callback(pct, message)
            except Exception as e:
                self._logger.debug(f"User progress callback raised: {e}")

    def _run(self, target: Callable[..., Any], args: tuple, kwargs: dict) -> None:
        try:
            self._result = target(*args, **kwargs)
        except BaseException as e:
            self._error = e
            self._logger.exception(f"{self._thread.name} failed")
        finally:
            self._done.set()

    def poll(self) -> List[Tuple[float, str]]:
        """Drain queued (pct, message) events. Returns list (possibly empty)."""
        events: List[Tuple[float, str]] = []
        while True:
            try:
                events.append(self._progress_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def is_done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout)

    def error(self) -> Optional[BaseException]:
        return self._error if self._done.is_set() else None

    def result(self, timeout: Optional[float] = None) -> Any:
        if not self._done.wait(timeout):
            raise TimeoutError(f"{self._thread.name} did not complete in time")
        if self._error is not None:
            raise self._error
        return self._result
