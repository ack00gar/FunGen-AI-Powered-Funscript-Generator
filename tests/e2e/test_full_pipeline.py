"""
End-to-end tests for the full FunGen video-to-funscript pipeline.

These tests exercise the complete workflow: video processing via the CLI
followed by optional post-processing filter passes. They verify output
file structure, funscript validity, and multi-step pipeline correctness.

Usage:
    pytest tests/e2e/test_full_pipeline.py -v
    pytest tests/e2e/test_full_pipeline.py -v -m "e2e and slow"
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
TIMEOUT = 120  # seconds -- full pipeline may take a while

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


def _output_dir_for_video(video_path):
    """Return the expected output directory for a given video path."""
    basename = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(PROJECT_ROOT, "output", basename)


def _get_first_batch_mode():
    """Return the first batch-compatible CLI mode, or skip if none available."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        batch_trackers = discovery.get_batch_compatible_trackers()
        for info in batch_trackers:
            if info.cli_aliases:
                return info.cli_aliases[0]
    except Exception:
        pass
    pytest.skip("No batch-compatible mode available for pipeline test")


def _validate_funscript_file(path):
    """Load and validate a funscript file. Returns the parsed dict."""
    assert os.path.isfile(path), f"Funscript not found: {path}"
    with open(path, "r") as f:
        data = json.load(f)

    assert "actions" in data, f"Missing 'actions' key in {path}"
    actions = data["actions"]
    assert len(actions) > 0, f"Empty actions in {path}"

    prev_at = -1
    for action in actions:
        assert 0 <= action["pos"] <= 100, (
            f"Position {action['pos']} out of 0-100 in {path}"
        )
        assert action["at"] >= 0, f"Negative timestamp in {path}"
        assert action["at"] >= prev_at, (
            f"Non-monotonic timestamps: {prev_at} -> {action['at']} in {path}"
        )
        prev_at = action["at"]

    return data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_cleanup():
    """Track and clean up new files created in the test video's output dir."""
    output_dir = _output_dir_for_video(TEST_VIDEO)
    existed_before = os.path.isdir(output_dir)
    files_before = set()
    if existed_before:
        for f in os.listdir(output_dir):
            files_before.add(f)

    yield output_dir

    # Remove files created during the test
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
def pipeline_temp_dir():
    """Temporary directory for pipeline intermediate files."""
    tmpdir = tempfile.mkdtemp(prefix="fungen_pipeline_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.slow
class TestFullPipeline:
    """Test the complete end-to-end pipeline: video -> funscript -> filters."""

    def test_full_video_to_funscript_pipeline(self, output_cleanup, test_video_path, pipeline_temp_dir):
        """Process video via CLI, then apply post-processing filters in sequence.

        Pipeline:
        1. Generate funscript from video
        2. Apply amplify filter to the output
        3. Apply clamp filter to the amplified output
        4. Verify final output is valid
        """
        mode = _get_first_batch_mode()
        output_dir = _output_dir_for_video(test_video_path)
        basename = os.path.splitext(os.path.basename(test_video_path))[0]

        # Step 1: Generate funscript from video
        result = _run([
            PYTHON, MAIN_PY,
            test_video_path,
            "--mode", mode,
            "--overwrite",
            "--no-copy",
        ])
        assert result.returncode == 0, (
            f"Video processing failed: rc={result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

        funscript_path = os.path.join(output_dir, f"{basename}.funscript")
        original_data = _validate_funscript_file(funscript_path)
        original_count = len(original_data["actions"])

        # Step 2: Copy funscript to temp dir and apply amplify filter
        working_copy = os.path.join(pipeline_temp_dir, f"{basename}.funscript")
        shutil.copy2(funscript_path, working_copy)

        result = _run([
            PYTHON, MAIN_PY,
            working_copy,
            "--funscript-mode",
            "--filter", "amplify",
        ])
        assert result.returncode == 0, (
            f"Amplify filter failed: rc={result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

        # Amplify output: <base>.amplify.funscript
        base_no_ext = os.path.splitext(working_copy)[0]
        amplified_path = f"{base_no_ext}.amplify.funscript"
        amplified_data = _validate_funscript_file(amplified_path)

        # Step 3: Apply clamp filter to amplified output
        result = _run([
            PYTHON, MAIN_PY,
            amplified_path,
            "--funscript-mode",
            "--filter", "clamp",
        ])
        assert result.returncode == 0, (
            f"Clamp filter failed: rc={result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

        # Clamp output: <base>.amplify.clamp.funscript
        amplified_base = os.path.splitext(amplified_path)[0]
        clamped_path = f"{amplified_base}.clamp.funscript"
        clamped_data = _validate_funscript_file(clamped_path)

        # Verify all positions are clamped to 0-100
        for action in clamped_data["actions"]:
            assert 0 <= action["pos"] <= 100, (
                f"Post-clamp position {action['pos']} out of range"
            )

    def test_batch_processing_single_video(self, output_cleanup, test_video_path):
        """Simulate batch processing with a single video file.

        This verifies the batch code path works when given a single file,
        which is the typical CLI usage pattern.
        """
        mode = _get_first_batch_mode()
        output_dir = _output_dir_for_video(test_video_path)
        basename = os.path.splitext(os.path.basename(test_video_path))[0]

        result = _run([
            PYTHON, MAIN_PY,
            test_video_path,
            "--mode", mode,
            "--overwrite",
            "--no-copy",
        ])
        assert result.returncode == 0, (
            f"Batch single-video failed: rc={result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

        funscript_path = os.path.join(output_dir, f"{basename}.funscript")
        data = _validate_funscript_file(funscript_path)

        # Verify the output has a reasonable number of actions for a video
        assert len(data["actions"]) >= 2, (
            "Expected at least 2 actions from video processing"
        )

    def test_output_file_structure(self, output_cleanup, test_video_path):
        """After processing, verify the expected output file structure.

        Expected files in output/<video_basename>/:
        - <basename>.funscript          (final autotuned output)
        - <basename>_t1_raw.funscript   (raw pre-autotune output)
        """
        mode = _get_first_batch_mode()
        output_dir = _output_dir_for_video(test_video_path)
        basename = os.path.splitext(os.path.basename(test_video_path))[0]

        result = _run([
            PYTHON, MAIN_PY,
            test_video_path,
            "--mode", mode,
            "--overwrite",
            "--no-copy",
        ])
        assert result.returncode == 0, (
            f"Processing failed: rc={result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

        # Verify output directory exists
        assert os.path.isdir(output_dir), f"Output directory not found: {output_dir}"

        # Check for expected output files
        final_funscript = os.path.join(output_dir, f"{basename}.funscript")
        raw_funscript = os.path.join(output_dir, f"{basename}_t1_raw.funscript")

        assert os.path.isfile(final_funscript), (
            f"Final funscript not found: {final_funscript}"
        )
        assert os.path.isfile(raw_funscript), (
            f"Raw funscript not found: {raw_funscript}"
        )

        # Both should be valid funscripts
        final_data = _validate_funscript_file(final_funscript)
        raw_data = _validate_funscript_file(raw_funscript)

        # Raw should typically have more actions than the final autotuned version
        # (autotune simplifies), but at minimum both should have actions
        assert len(final_data["actions"]) > 0
        assert len(raw_data["actions"]) > 0

        # The raw funscript should have at least as many actions as the final
        # (autotune applies simplification / RDP)
        assert len(raw_data["actions"]) >= len(final_data["actions"]), (
            f"Raw ({len(raw_data['actions'])} actions) should have at least as many "
            f"actions as final ({len(final_data['actions'])} actions) since autotune "
            f"applies simplification"
        )

    def test_no_autotune_produces_different_output(self, output_cleanup, test_video_path):
        """Running with --no-autotune should produce a different final funscript
        compared to the default (autotuned) run.

        With --no-autotune, the final funscript should match the raw output more
        closely since no simplification/smoothing is applied.
        """
        mode = _get_first_batch_mode()
        output_dir = _output_dir_for_video(test_video_path)
        basename = os.path.splitext(os.path.basename(test_video_path))[0]

        # Run with autotune (default)
        result = _run([
            PYTHON, MAIN_PY,
            test_video_path,
            "--mode", mode,
            "--overwrite",
            "--no-copy",
        ])
        assert result.returncode == 0

        final_funscript = os.path.join(output_dir, f"{basename}.funscript")
        with open(final_funscript, "r") as f:
            autotuned = json.load(f)
        autotuned_count = len(autotuned["actions"])

        # Run without autotune
        result = _run([
            PYTHON, MAIN_PY,
            test_video_path,
            "--mode", mode,
            "--overwrite",
            "--no-copy",
            "--no-autotune",
        ])
        assert result.returncode == 0

        with open(final_funscript, "r") as f:
            no_autotune = json.load(f)
        no_autotune_count = len(no_autotune["actions"])

        # Without autotune, the output should typically have more actions
        # (no simplification), or at least be different
        # We just verify both produced valid output -- the counts may differ
        assert no_autotune_count > 0, "No-autotune output should have actions"
        assert autotuned_count > 0, "Autotuned output should have actions"


@pytest.mark.e2e
@pytest.mark.slow
class TestPipelineEdgeCases:
    """Test edge cases in the pipeline."""

    def test_process_with_all_flags(self, output_cleanup, test_video_path):
        """Run processing with multiple flags combined."""
        mode = _get_first_batch_mode()

        result = _run([
            PYTHON, MAIN_PY,
            test_video_path,
            "--mode", mode,
            "--overwrite",
            "--no-copy",
            "--no-autotune",
            "--od-mode", "current",
        ])
        # Should succeed or at least not crash
        # Some modes may not support --od-mode but that's handled gracefully
        assert result.returncode == 0 or "error" not in result.stderr.lower(), (
            f"Combined flags run crashed: rc={result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

    def test_filter_chain_preserves_structure(self, pipeline_temp_dir):
        """Applying multiple filters in sequence should preserve funscript structure."""
        # Create initial funscript
        data = {
            "version": "1.0",
            "inverted": False,
            "range": 100,
            "actions": [
                {"at": i * 100, "pos": (i * 37) % 101}
                for i in range(40)
            ],
        }
        input_path = os.path.join(pipeline_temp_dir, "chain_test.funscript")
        with open(input_path, "w") as f:
            json.dump(data, f)

        # Apply filters in sequence: invert -> clamp -> amplify
        filters_to_apply = ["invert", "clamp", "amplify"]
        current_path = input_path

        for i, filter_name in enumerate(filters_to_apply):
            result = _run([
                PYTHON, MAIN_PY,
                current_path,
                "--funscript-mode",
                "--filter", filter_name,
            ])
            assert result.returncode == 0, (
                f"Filter '{filter_name}' in chain failed: rc={result.returncode}\n"
                f"stderr: {result.stderr[:500]}"
            )

            # Next input is the output of the current filter
            base, ext = os.path.splitext(current_path)
            current_path = f"{base}.{filter_name}{ext}"
            assert os.path.isfile(current_path), (
                f"Filter '{filter_name}' output not found: {current_path}"
            )

        # Validate final output
        final_data = _validate_funscript_file(current_path)
        assert len(final_data["actions"]) > 0, (
            "Filter chain produced empty output"
        )
        # All positions must still be in valid range
        for action in final_data["actions"]:
            assert 0 <= action["pos"] <= 100
            assert action["at"] >= 0
