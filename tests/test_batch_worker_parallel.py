"""Tests for BatchWorker parallel subprocess path (v0.9.0 feature).

The parallel path is exercised end-to-end by mocking subprocess.Popen so
no real main.py invocations happen. These tests validate:
  - max_parallel argument handling
  - eligibility filtering (offline + batch-compatible + has CLI alias)
  - concurrency cap enforcement
  - exit code -> queue status mapping
  - clean termination on stop()
"""
from __future__ import annotations

import time
import types
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from application.batch.batch_queue import BatchItemStatus, BatchQueue
from application.batch.batch_worker import BatchWorker


# ---------- helpers ----------

def _mock_app_no_tracker():
    app = types.SimpleNamespace()
    app.app_state_ui = types.SimpleNamespace(selected_tracker_name="")
    app.is_batch_processing_active = False
    app.stage_processor = None
    app.processor = None
    app.gui_instance = None
    app.file_manager = None
    return app


def _offline_tracker_name() -> Optional[str]:
    """Return the internal name of any offline batch-compatible tracker with a CLI alias."""
    from config.tracker_discovery import get_tracker_discovery
    disc = get_tracker_discovery()
    for info in disc.get_batch_compatible_trackers():
        if info.cli_aliases and not info.requires_intervention:
            return info.internal_name
    return None


def _intervention_tracker_name() -> Optional[str]:
    from config.tracker_discovery import get_tracker_discovery
    disc = get_tracker_discovery()
    for info in disc.get_all_trackers().values():
        if info.requires_intervention:
            return info.internal_name
    return None


def _mock_app_offline():
    name = _offline_tracker_name()
    if name is None:
        pytest.skip("no batch-compatible offline tracker available")
    app = _mock_app_no_tracker()
    app.app_state_ui.selected_tracker_name = name
    return app


def _wait_until(predicate, timeout_s: float = 5.0, step_s: float = 0.02) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step_s)
    return False


class _FakePopen:
    """Configurable subprocess stand-in. Each instance tracks its own lifecycle."""

    _registry = []  # populated by the subprocess.Popen patch side_effect

    def __init__(self, cmd, cwd=None, env=None, stdout=None, stderr=None,
                 start_new_session=False, **kw):
        self.cmd = cmd
        self.cwd = cwd
        self.env = dict(env) if env else {}
        self.pid = len(_FakePopen._registry) + 1000
        self._started = time.time()
        self._rc: Optional[int] = None
        self._run_duration_s: float = 0.2
        self._final_rc: int = 0
        _FakePopen._registry.append(self)

    def poll(self):
        if self._rc is not None:
            return self._rc
        if time.time() - self._started >= self._run_duration_s:
            self._rc = self._final_rc
        return self._rc

    @property
    def returncode(self):
        return self._rc if self._rc is not None else -1

    def terminate(self):
        if self._rc is None:
            self._rc = -15  # SIGTERM

    def wait(self, timeout=None):
        if self._rc is None:
            self._rc = -15

    def kill(self):
        if self._rc is None:
            self._rc = -9


@pytest.fixture(autouse=True)
def _reset_popen_registry():
    _FakePopen._registry.clear()
    yield
    _FakePopen._registry.clear()


# ---------- simple constructor tests ----------

def test_max_parallel_defaults_to_one():
    app = _mock_app_no_tracker()
    w = BatchWorker(app, BatchQueue())
    assert w.max_parallel == 1


@pytest.mark.parametrize("raw,expected", [(0, 1), (-5, 1), (1, 1), (2, 2), (4, 4)])
def test_max_parallel_clamped_to_at_least_one(raw, expected):
    app = _mock_app_no_tracker()
    w = BatchWorker(app, BatchQueue(), max_parallel=raw)
    assert w.max_parallel == expected


# ---------- eligibility ----------

def test_find_next_eligible_returns_none_when_no_tracker_selected():
    app = _mock_app_no_tracker()
    q = BatchQueue()
    q.add("/fake/a.mp4")
    w = BatchWorker(app, q, max_parallel=2)
    assert w._find_next_eligible_for_subprocess() is None


def test_find_next_eligible_skips_intervention_trackers():
    name = _intervention_tracker_name()
    if name is None:
        pytest.skip("no intervention tracker present in registry")
    app = _mock_app_no_tracker()
    app.app_state_ui.selected_tracker_name = name
    q = BatchQueue()
    q.add("/fake/a.mp4")
    w = BatchWorker(app, q, max_parallel=2)
    assert w._find_next_eligible_for_subprocess() is None


def test_find_next_eligible_returns_first_queued_offline_item():
    app = _mock_app_offline()
    q = BatchQueue()
    q.add("/fake/a.mp4")
    q.add("/fake/b.mp4")
    w = BatchWorker(app, q, max_parallel=2)
    idx = w._find_next_eligible_for_subprocess()
    assert idx == 0


# ---------- end-to-end parallel path with mocked Popen ----------

def _run_parallel_to_completion(app, q, max_parallel, timeout_s=15.0):
    with patch("subprocess.Popen", _FakePopen):
        with patch.object(BatchWorker, "_app_is_busy", return_value=False):
            w = BatchWorker(app, q, max_parallel=max_parallel)
            w.start()
            ok = _wait_until(
                lambda: all(it.status in (BatchItemStatus.COMPLETED, BatchItemStatus.FAILED)
                            for it in q.items),
                timeout_s=timeout_s,
            )
            w.stop()
            if w._thread:
                w._thread.join(timeout=5.0)
            return ok


def test_parallel_launches_subprocess_per_eligible_item():
    app = _mock_app_offline()
    q = BatchQueue()
    for i in range(4):
        q.add(f"/fake/vid_{i}.mp4")
    ok = _run_parallel_to_completion(app, q, max_parallel=2, timeout_s=15.0)
    assert ok, "queue did not drain in time"
    assert len(_FakePopen._registry) == 4
    for item in q.items:
        assert item.status == BatchItemStatus.COMPLETED


def test_parallel_nonzero_exit_marks_failed():
    """Items whose subprocess returns rc != 0 should end up FAILED."""

    class _FailingPopen(_FakePopen):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._final_rc = 42

    app = _mock_app_offline()
    q = BatchQueue()
    q.add("/fake/vid.mp4")
    with patch("subprocess.Popen", _FailingPopen):
        with patch.object(BatchWorker, "_app_is_busy", return_value=False):
            w = BatchWorker(app, q, max_parallel=2)
            w.start()
            _wait_until(lambda: q.items[0].status == BatchItemStatus.FAILED, timeout_s=5.0)
            w.stop()
            if w._thread:
                w._thread.join(timeout=5.0)
    assert q.items[0].status == BatchItemStatus.FAILED
    assert "42" in q.items[0].error_message


def test_parallel_cli_cmd_uses_overwrite_flag():
    """Parallel subprocesses must pass --overwrite (mirrors sequential always-reprocess)."""
    app = _mock_app_offline()
    q = BatchQueue()
    q.add("/fake/only.mp4")
    with patch("subprocess.Popen", _FakePopen):
        with patch.object(BatchWorker, "_app_is_busy", return_value=False):
            w = BatchWorker(app, q, max_parallel=2)
            w.start()
            _wait_until(lambda: len(_FakePopen._registry) >= 1, timeout_s=5.0)
            w.stop()
            if w._thread:
                w._thread.join(timeout=5.0)
    assert _FakePopen._registry, "no subprocess launched"
    cmd = _FakePopen._registry[0].cmd
    assert "--overwrite" in cmd
    assert "--quiet" in cmd
    assert "--mode" in cmd


def test_parallel_subprocess_env_carries_fungen_batch_parallel():
    """Subprocesses should see FUNGEN_BATCH_PARALLEL=<max_parallel> in env.

    Stage 1 in the child reads this to scale down its internal producer/
    consumer pool and avoid CPU oversubscription across the parallel pool.
    """
    app = _mock_app_offline()
    q = BatchQueue()
    q.add("/fake/only.mp4")
    with patch("subprocess.Popen", _FakePopen):
        with patch.object(BatchWorker, "_app_is_busy", return_value=False):
            w = BatchWorker(app, q, max_parallel=3)
            w.start()
            _wait_until(lambda: len(_FakePopen._registry) >= 1, timeout_s=5.0)
            w.stop()
            if w._thread:
                w._thread.join(timeout=5.0)
    assert _FakePopen._registry, "no subprocess launched"
    env = _FakePopen._registry[0].env
    assert env.get("FUNGEN_BATCH_PARALLEL") == "3"


def test_parallel_honors_concurrency_cap():
    """At no point should more than max_parallel subprocesses be 'running'."""

    class _SlowPopen(_FakePopen):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._run_duration_s = 0.6

    app = _mock_app_offline()
    q = BatchQueue()
    for i in range(6):
        q.add(f"/fake/slow_{i}.mp4")
    peak_concurrent = {"v": 0}

    def _running_count():
        return sum(1 for p in _FakePopen._registry if p._rc is None)

    with patch("subprocess.Popen", _SlowPopen):
        with patch.object(BatchWorker, "_app_is_busy", return_value=False):
            w = BatchWorker(app, q, max_parallel=2)
            w.start()
            deadline = time.time() + 10.0
            while time.time() < deadline:
                peak_concurrent["v"] = max(peak_concurrent["v"], _running_count())
                if all(it.status in (BatchItemStatus.COMPLETED, BatchItemStatus.FAILED)
                       for it in q.items):
                    break
                time.sleep(0.01)
            w.stop()
            if w._thread:
                w._thread.join(timeout=5.0)
    assert peak_concurrent["v"] <= 2, f"cap breached, saw {peak_concurrent['v']} concurrent"
    assert peak_concurrent["v"] >= 2, f"pool never filled, peak={peak_concurrent['v']}"
    assert len(_FakePopen._registry) == 6
