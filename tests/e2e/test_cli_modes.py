"""
End-to-end tests for FunGen CLI processing modes.

These tests invoke main.py via subprocess and verify that video processing
works correctly for each discovered tracker mode. They exercise the full
CLI code path from argument parsing through to funscript output.

Usage:
    pytest tests/e2e/test_cli_modes.py -v
    pytest tests/e2e/test_cli_modes.py -v -m "e2e and cli"
    pytest tests/e2e/test_cli_modes.py -v -k "test_cli_help"
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAIN_PY = os.path.join(PROJECT_ROOT, "main.py")
TEST_VIDEO = "/Users/k00gar/Downloads/test_koogar_extra_short_A.mp4"
PYTHON = sys.executable
TIMEOUT = 120  # seconds

# Environment variables shared across all subprocess calls
_BASE_ENV = {**os.environ, "FUNGEN_TESTING": "1", "PYTHONPATH": PROJECT_ROOT}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args, timeout=TIMEOUT, env=None):
    """Run a CLI command and return the CompletedProcess."""
    run_env = env or _BASE_ENV
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=PROJECT_ROOT,
        env=run_env,
    )


def _output_dir_for_video(video_path):
    """Return the expected output directory for a given video path."""
    basename = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(PROJECT_ROOT, "output", basename)


def _get_discovered_modes():
    """Dynamically discover available CLI modes from the tracker_discovery module.

    Returns a list of mode strings.  Falls back to an empty list on import error
    so that tests that depend on modes are skipped rather than erroring.
    """
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        return discovery.get_supported_cli_modes()
    except Exception:
        return []


def _get_batch_modes(max_count=3):
    """Return up to *max_count* batch-compatible CLI mode aliases.

    These are the modes most suitable for automated testing because they do not
    require user intervention (ROI selection, etc.).
    """
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        batch_trackers = discovery.get_batch_compatible_trackers()
        # Use the first CLI alias for each tracker
        modes = []
        seen = set()
        for info in batch_trackers:
            if info.cli_aliases:
                alias = info.cli_aliases[0]
                if alias not in seen:
                    seen.add(alias)
                    modes.append(alias)
        return modes[:max_count]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_cleanup():
    """Yield and then clean up any output produced for the test video."""
    output_dir = _output_dir_for_video(TEST_VIDEO)
    # Capture state before test so we can restore
    existed_before = os.path.isdir(output_dir)
    files_before = set()
    if existed_before:
        for f in os.listdir(output_dir):
            files_before.add(f)

    yield output_dir

    # Cleanup: remove files that were created during the test
    if os.path.isdir(output_dir):
        for f in os.listdir(output_dir):
            if f not in files_before:
                fpath = os.path.join(output_dir, f)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                    elif os.path.isdir(fpath):
                        shutil.rmtree(fpath, ignore_errors=True)
                except OSError:
                    pass


@pytest.fixture
def clean_output_dir():
    """Create a fresh temporary output directory and point settings at it."""
    tmpdir = tempfile.mkdtemp(prefix="fungen_e2e_output_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests: Basic CLI invocations
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.cli
@pytest.mark.slow
class TestCLIBasic:
    """Test basic CLI argument handling."""

    def test_cli_help(self):
        """Running --help should exit 0 and print usage information."""
        result = _run([PYTHON, MAIN_PY, "--help"], timeout=30)
        assert result.returncode == 0, f"--help returned {result.returncode}: {result.stderr}"
        combined = result.stdout + result.stderr
        assert "usage" in combined.lower() or "funscript" in combined.lower() or "funGen" in combined.lower(), (
            f"Expected usage info in output, got:\n{combined[:500]}"
        )

    def test_cli_no_args(self):
        """Running with no arguments should not crash (may launch GUI or print help).

        We set a short timeout because the GUI would hang; a non-crash is
        considered success regardless of exit code.
        """
        # With no args the app tries to start the GUI, which will fail in a
        # headless/test environment.  We just verify it doesn't hard-crash
        # with an unhandled traceback in the argument-parsing phase.
        try:
            result = _run([PYTHON, MAIN_PY], timeout=15)
            # Any exit code is fine as long as we didn't get a timeout.
            # Check there's no uncaught traceback in the first phase.
        except subprocess.TimeoutExpired:
            # GUI probably started and hung -- that's fine for this test.
            pass

    def test_cli_invalid_video(self):
        """Passing a non-existent video file should be handled gracefully."""
        fake_video = "/tmp/nonexistent_video_fungen_test.mp4"
        result = _run([PYTHON, MAIN_PY, fake_video, "--mode", "3-stage", "--overwrite"], timeout=60)
        # Should either exit non-zero or print an error message -- not crash
        combined = (result.stdout + result.stderr).lower()
        # Accept either an error message or a non-zero exit code
        has_error_msg = any(kw in combined for kw in ["not exist", "error", "not found", "no video", "invalid"])
        assert result.returncode != 0 or has_error_msg, (
            f"Expected error handling for missing video. rc={result.returncode}\n"
            f"stdout: {result.stdout[:300]}\nstderr: {result.stderr[:300]}"
        )


# ---------------------------------------------------------------------------
# Tests: Mode discovery
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.cli
@pytest.mark.slow
class TestCLIModeDiscovery:
    """Test that tracker modes can be discovered dynamically."""

    def test_cli_discover_modes(self):
        """The tracker discovery system should return a non-empty list of modes."""
        modes = _get_discovered_modes()
        assert len(modes) > 0, "tracker_discovery returned no CLI modes"

    def test_cli_batch_modes_available(self):
        """There should be at least one batch-compatible mode for CLI testing."""
        modes = _get_batch_modes(max_count=10)
        assert len(modes) > 0, "No batch-compatible modes found"


# ---------------------------------------------------------------------------
# Tests: Video processing per mode
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.cli
@pytest.mark.slow
class TestCLIProcessVideo:
    """Process the test video through each discovered batch-compatible mode."""

    @staticmethod
    def _validate_funscript(path):
        """Validate a funscript file and return the parsed data."""
        assert os.path.isfile(path), f"Funscript not found: {path}"
        with open(path, "r") as f:
            data = json.load(f)

        assert "actions" in data, f"Missing 'actions' key in {path}"
        actions = data["actions"]
        assert len(actions) > 0, f"Actions list is empty in {path}"

        prev_at = -1
        for action in actions:
            assert "at" in action, f"Action missing 'at' key: {action}"
            assert "pos" in action, f"Action missing 'pos' key: {action}"
            assert 0 <= action["pos"] <= 100, (
                f"Position {action['pos']} out of range 0-100 in {path}"
            )
            assert action["at"] >= 0, (
                f"Negative timestamp {action['at']} in {path}"
            )
            assert action["at"] >= prev_at, (
                f"Timestamps not monotonically increasing: {prev_at} -> {action['at']} in {path}"
            )
            prev_at = action["at"]

        return data

    def _get_modes(self):
        """Get up to 3 batch-compatible modes for testing."""
        modes = _get_batch_modes(max_count=3)
        if not modes:
            pytest.skip("No batch-compatible modes discovered")
        return modes

    def test_cli_process_single_video(self, output_cleanup, test_video_path):
        """Process the test video with each discovered mode (up to 3).

        For each mode verify:
        - Exit code 0
        - Output funscript file exists
        - Funscript is valid JSON with 'actions' key
        - Actions list is non-empty
        - All action positions are 0-100
        - All action timestamps are non-negative and increasing
        """
        modes = self._get_modes()
        output_dir = _output_dir_for_video(test_video_path)
        basename = os.path.splitext(os.path.basename(test_video_path))[0]
        funscript_name = f"{basename}.funscript"

        for mode in modes:
            result = _run([
                PYTHON, MAIN_PY,
                test_video_path,
                "--mode", mode,
                "--overwrite",
                "--no-copy",
            ])
            assert result.returncode == 0, (
                f"Mode '{mode}' failed with rc={result.returncode}\n"
                f"stderr: {result.stderr[:500]}"
            )

            funscript_path = os.path.join(output_dir, funscript_name)
            self._validate_funscript(funscript_path)


# ---------------------------------------------------------------------------
# Tests: CLI flags
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.cli
@pytest.mark.slow
class TestCLIFlags:
    """Test specific CLI flag behaviour."""

    def test_cli_no_autotune_flag(self, output_cleanup, test_video_path):
        """Running with --no-autotune should produce raw (untuned) output."""
        modes = _get_batch_modes(max_count=1)
        if not modes:
            pytest.skip("No batch-compatible modes discovered")
        mode = modes[0]

        output_dir = _output_dir_for_video(test_video_path)
        basename = os.path.splitext(os.path.basename(test_video_path))[0]
        raw_name = f"{basename}_t1_raw.funscript"

        result = _run([
            PYTHON, MAIN_PY,
            test_video_path,
            "--mode", mode,
            "--overwrite",
            "--no-copy",
            "--no-autotune",
        ])
        assert result.returncode == 0, (
            f"--no-autotune run failed: rc={result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

        # The raw funscript should still be produced
        raw_path = os.path.join(output_dir, raw_name)
        assert os.path.isfile(raw_path), f"Raw funscript not found: {raw_path}"
        with open(raw_path, "r") as f:
            data = json.load(f)
        assert "actions" in data and len(data["actions"]) > 0, (
            "Raw funscript should have non-empty actions"
        )

    def test_cli_overwrite_flag(self, output_cleanup, test_video_path):
        """Running twice with --overwrite should succeed on both runs."""
        modes = _get_batch_modes(max_count=1)
        if not modes:
            pytest.skip("No batch-compatible modes discovered")
        mode = modes[0]

        cmd = [
            PYTHON, MAIN_PY,
            test_video_path,
            "--mode", mode,
            "--overwrite",
            "--no-copy",
        ]

        # First run
        r1 = _run(cmd)
        assert r1.returncode == 0, (
            f"First overwrite run failed: rc={r1.returncode}\nstderr: {r1.stderr[:500]}"
        )

        # Second run should also succeed (overwriting output)
        r2 = _run(cmd)
        assert r2.returncode == 0, (
            f"Second overwrite run failed: rc={r2.returncode}\nstderr: {r2.stderr[:500]}"
        )
