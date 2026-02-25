"""
End-to-end tests for FunGen CLI funscript filter mode.

These tests exercise the ``--funscript-mode --filter <name>`` code path by
creating temporary funscript files, running filters through the CLI, and
verifying that the output is correct.

Usage:
    pytest tests/e2e/test_cli_funscript_filters.py -v
    pytest tests/e2e/test_cli_funscript_filters.py -v -m "e2e and cli and plugins"
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
PYTHON = sys.executable
TIMEOUT = 60  # seconds -- filters are fast

# All filters accepted by the CLI --filter argument
ALL_FILTERS = [
    "ultimate-autotune",
    "rdp-simplify",
    "savgol-filter",
    "speed-limiter",
    "anti-jerk",
    "amplify",
    "clamp",
    "invert",
    "keyframe",
]

# Environment for subprocess calls
_BASE_ENV = {**os.environ, "FUNGEN_TESTING": "1", "PYTHONPATH": PROJECT_ROOT}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args, timeout=TIMEOUT, env=None):
    """Run a CLI command and return the CompletedProcess."""
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=PROJECT_ROOT,
        env=env or _BASE_ENV,
    )


def _make_sample_funscript_data(num_actions=50):
    """Generate a realistic sample funscript data dict.

    Produces a zigzag pattern that alternates between 0 and 100
    with some mid-range values mixed in, spaced 100ms apart.
    """
    actions = []
    for i in range(num_actions):
        if i % 4 == 0:
            pos = 0
        elif i % 4 == 1:
            pos = 100
        elif i % 4 == 2:
            pos = 25
        else:
            pos = 75
        actions.append({"at": i * 100, "pos": pos})
    return {
        "version": "1.0",
        "inverted": False,
        "range": 100,
        "actions": actions,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def filter_work_dir():
    """Temporary directory for funscript filter tests."""
    tmpdir = tempfile.mkdtemp(prefix="fungen_filter_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_funscript(filter_work_dir):
    """Create a sample funscript file in the temp directory and return its path."""
    data = _make_sample_funscript_data(num_actions=50)
    path = os.path.join(filter_work_dir, "test_sample.funscript")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.cli
@pytest.mark.plugins
class TestCLIFunscriptFilters:
    """Test the CLI funscript filtering mode."""

    # -- Individual filter tests ------------------------------------------

    def test_cli_funscript_mode_invert(self, sample_funscript, filter_work_dir):
        """The invert filter should flip positions (pos -> 100 - pos)."""
        result = _run([
            PYTHON, MAIN_PY,
            sample_funscript,
            "--funscript-mode",
            "--filter", "invert",
        ])
        assert result.returncode == 0, (
            f"Invert filter failed: rc={result.returncode}\nstderr: {result.stderr[:500]}"
        )

        # Output should be <base>.invert.funscript since we didn't pass --overwrite
        base, ext = os.path.splitext(sample_funscript)
        output_path = f"{base}.invert{ext}"
        assert os.path.isfile(output_path), f"Inverted output not found: {output_path}"

        # Verify inversion: original 0 -> 100, 100 -> 0, etc.
        with open(sample_funscript, "r") as f:
            original = json.load(f)
        with open(output_path, "r") as f:
            inverted = json.load(f)

        orig_actions = original["actions"]
        inv_actions = inverted["actions"]
        assert len(inv_actions) > 0, "Inverted funscript has no actions"

        # Check that at least the first few are inverted (plugins may adjust edges)
        for orig, inv in zip(orig_actions[:5], inv_actions[:5]):
            expected_pos = 100 - orig["pos"]
            assert inv["pos"] == expected_pos, (
                f"Expected inverted pos {expected_pos}, got {inv['pos']} "
                f"(original was {orig['pos']})"
            )

    def test_cli_funscript_mode_amplify(self, sample_funscript, filter_work_dir):
        """The amplify filter should produce an output file."""
        result = _run([
            PYTHON, MAIN_PY,
            sample_funscript,
            "--funscript-mode",
            "--filter", "amplify",
        ])
        assert result.returncode == 0, (
            f"Amplify filter failed: rc={result.returncode}\nstderr: {result.stderr[:500]}"
        )

        base, ext = os.path.splitext(sample_funscript)
        output_path = f"{base}.amplify{ext}"
        assert os.path.isfile(output_path), f"Amplified output not found: {output_path}"

        with open(output_path, "r") as f:
            data = json.load(f)
        assert "actions" in data and len(data["actions"]) > 0

    def test_cli_funscript_mode_clamp(self, sample_funscript, filter_work_dir):
        """The clamp filter should keep all positions within 0-100."""
        result = _run([
            PYTHON, MAIN_PY,
            sample_funscript,
            "--funscript-mode",
            "--filter", "clamp",
        ])
        assert result.returncode == 0, (
            f"Clamp filter failed: rc={result.returncode}\nstderr: {result.stderr[:500]}"
        )

        base, ext = os.path.splitext(sample_funscript)
        output_path = f"{base}.clamp{ext}"
        assert os.path.isfile(output_path), f"Clamped output not found: {output_path}"

        with open(output_path, "r") as f:
            data = json.load(f)
        actions = data.get("actions", [])
        assert len(actions) > 0, "Clamped funscript has no actions"
        for action in actions:
            assert 0 <= action["pos"] <= 100, (
                f"Clamped position {action['pos']} out of range 0-100"
            )

    # -- All filters smoke test (parametrized) ----------------------------

    @pytest.mark.parametrize("filter_name", ALL_FILTERS)
    def test_cli_funscript_mode_filter(self, filter_name, filter_work_dir):
        """Run a single CLI filter and verify it produces valid output.

        Parametrized over ALL_FILTERS so each filter gets its own test node
        and failures are reported independently.
        """
        # Create a fresh funscript for this filter
        data = _make_sample_funscript_data(num_actions=50)
        input_path = os.path.join(filter_work_dir, f"test_{filter_name}.funscript")
        with open(input_path, "w") as f:
            json.dump(data, f)

        result = _run([
            PYTHON, MAIN_PY,
            input_path,
            "--funscript-mode",
            "--filter", filter_name,
        ])
        assert result.returncode == 0, (
            f"Filter '{filter_name}' failed: rc={result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

        # Verify output file exists
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}.{filter_name}{ext}"
        assert os.path.isfile(output_path), (
            f"Filter '{filter_name}' did not produce output at {output_path}\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

        # Verify output is valid JSON with actions
        with open(output_path, "r") as f:
            out_data = json.load(f)
        assert "actions" in out_data, (
            f"Filter '{filter_name}' output missing 'actions' key"
        )
        # Some filters (like RDP simplify) may reduce action count but should
        # not produce an empty result from 50 input actions.
        assert len(out_data["actions"]) > 0, (
            f"Filter '{filter_name}' produced empty actions list"
        )

    # -- Error handling ---------------------------------------------------

    def test_cli_funscript_invalid_filter(self, sample_funscript):
        """Using a bogus filter name should fail with a clear error."""
        result = _run([
            PYTHON, MAIN_PY,
            sample_funscript,
            "--funscript-mode",
            "--filter", "nonexistent-bogus-filter",
        ])
        # argparse should reject the invalid choice
        assert result.returncode != 0, (
            "Expected non-zero exit code for invalid filter name"
        )
        combined = (result.stdout + result.stderr).lower()
        assert "invalid" in combined or "error" in combined or "choice" in combined, (
            f"Expected error message about invalid filter.\n"
            f"stdout: {result.stdout[:300]}\nstderr: {result.stderr[:300]}"
        )

    def test_cli_funscript_mode_without_filter_flag(self, sample_funscript):
        """Using --funscript-mode without --filter should fail."""
        result = _run([
            PYTHON, MAIN_PY,
            sample_funscript,
            "--funscript-mode",
        ])
        assert result.returncode != 0, (
            "Expected non-zero exit code for --funscript-mode without --filter"
        )

    def test_cli_funscript_overwrite_mode(self, filter_work_dir):
        """Running with --overwrite should write output to the same file."""
        data = _make_sample_funscript_data(num_actions=30)
        input_path = os.path.join(filter_work_dir, "overwrite_test.funscript")
        with open(input_path, "w") as f:
            json.dump(data, f)

        original_mtime = os.path.getmtime(input_path)

        result = _run([
            PYTHON, MAIN_PY,
            input_path,
            "--funscript-mode",
            "--filter", "invert",
            "--overwrite",
        ])
        assert result.returncode == 0, (
            f"Overwrite invert failed: rc={result.returncode}\nstderr: {result.stderr[:500]}"
        )

        # The file should have been modified in-place
        new_mtime = os.path.getmtime(input_path)
        assert new_mtime >= original_mtime, "File was not updated by --overwrite"

        with open(input_path, "r") as f:
            out_data = json.load(f)
        assert "actions" in out_data and len(out_data["actions"]) > 0

    def test_cli_funscript_nonexistent_input(self):
        """Passing a non-existent funscript should be handled gracefully."""
        fake_path = "/tmp/fungen_nonexistent_test_file.funscript"
        result = _run([
            PYTHON, MAIN_PY,
            fake_path,
            "--funscript-mode",
            "--filter", "invert",
        ])
        # Should fail gracefully (file not found)
        combined = (result.stdout + result.stderr).lower()
        has_error = any(kw in combined for kw in ["not exist", "error", "not found", "no funscript"])
        assert result.returncode != 0 or has_error, (
            f"Expected error for nonexistent funscript. rc={result.returncode}\n"
            f"output: {combined[:500]}"
        )
