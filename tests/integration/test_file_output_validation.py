#!/usr/bin/env python3
"""
Comprehensive file output validation tests.

This module tests that all processing modes produce valid, non-empty output files
with correct structure and content.
"""

import os
import sys
import pytest
import subprocess
import tempfile
import shutil
import json
import msgpack
from pathlib import Path
import time
from typing import Dict, List, Any

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import test configuration  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from test_config import get_test_video_path, skip_if_no_test_videos

pytestmark = [pytest.mark.integration, pytest.mark.file_output_validation]

class TestFileOutputValidation:
    """Comprehensive tests for file output validation across all processing modes."""
    
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Set up test environment with temp directories."""
        self.test_video_path = get_test_video_path(category="short")
        self.temp_test_dir = tempfile.mkdtemp(prefix="file_output_test_")
        
        yield
        
        # Cleanup
        if os.path.exists(self.temp_test_dir):
            shutil.rmtree(self.temp_test_dir, ignore_errors=True)
    
    def run_cli_and_validate_basic_output(self, mode: str, expected_files: List[str]) -> Dict[str, Any]:
        """Run CLI processing and validate basic output files exist."""
        if not os.path.exists(self.test_video_path):
            pytest.skip("Test video not available")
        
        # Copy video to temp directory
        temp_video_name = "test_output_validation.mp4"
        temp_video_path = os.path.join(self.temp_test_dir, temp_video_name)
        shutil.copy2(self.test_video_path, temp_video_path)
        
        # Run CLI processing
        cmd = [
            "python", "main.py",
            temp_video_path,
            "--mode", mode,
            "--overwrite"
        ]
        
        env = os.environ.copy()
        env['FUNGEN_TESTING'] = '1'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout
            env=env,
            cwd='/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator'
        )
        
        # Check that CLI completed successfully
        assert result.returncode == 0, f"CLI processing failed: {result.stderr}"
        
        # Determine output directory
        output_dir = os.path.join(self.temp_test_dir, "test_output_validation")
        
        # Validate expected files exist
        found_files = {}
        for expected_file in expected_files:
            file_path = os.path.join(output_dir, expected_file)
            assert os.path.exists(file_path), f"Expected output file not found: {expected_file}"
            found_files[expected_file] = file_path
        
        return {
            "output_dir": output_dir,
            "found_files": found_files,
            "cli_result": result
        }
    
    def validate_funscript_file(self, funscript_path: str) -> Dict[str, Any]:
        """Validate a .funscript file has correct structure and content."""
        assert os.path.exists(funscript_path), f"Funscript file not found: {funscript_path}"
        assert os.path.getsize(funscript_path) > 0, f"Funscript file is empty: {funscript_path}"
        
        with open(funscript_path, 'r') as f:
            data = json.load(f)
        
        # Validate basic JSON structure
        assert isinstance(data, dict), "Funscript must be a JSON object"
        assert "actions" in data, "Funscript must contain 'actions' field"
        assert isinstance(data["actions"], list), "Actions must be a list"
        
        actions = data["actions"]
        assert len(actions) > 0, "Funscript must contain at least one action"
        
        # Validate action structure
        for i, action in enumerate(actions):
            assert isinstance(action, dict), f"Action {i} must be an object"
            assert "at" in action, f"Action {i} missing 'at' field"
            assert "pos" in action, f"Action {i} missing 'pos' field"
            assert isinstance(action["at"], int), f"Action {i} 'at' must be integer"
            assert isinstance(action["pos"], int), f"Action {i} 'pos' must be integer"
            assert 0 <= action["pos"] <= 100, f"Action {i} 'pos' must be 0-100"
        
        # Validate actions are sorted by timestamp
        timestamps = [action["at"] for action in actions]
        assert timestamps == sorted(timestamps), "Actions must be sorted by timestamp"
        
        return {
            "action_count": len(actions),
            "duration": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            "position_range": (min(a["pos"] for a in actions), max(a["pos"] for a in actions)),
            "has_metadata": any(key not in ["at", "pos"] for action in actions for key in action.keys())
        }
    
    def validate_msgpack_file(self, msgpack_path: str) -> Dict[str, Any]:
        """Validate a .msgpack file has correct structure and content."""
        assert os.path.exists(msgpack_path), f"Msgpack file not found: {msgpack_path}"
        assert os.path.getsize(msgpack_path) > 0, f"Msgpack file is empty: {msgpack_path}"
        
        with open(msgpack_path, 'rb') as f:
            data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        
        assert isinstance(data, dict), "Msgpack data must be a dictionary"
        
        # Check for common Stage 2 data fields
        expected_fields = ["frame_objects", "segments"]
        found_fields = []
        
        for field in expected_fields:
            if field in data:
                found_fields.append(field)
        
        assert len(found_fields) > 0, f"Msgpack must contain at least one of: {expected_fields}"
        
        frame_count = 0
        segment_count = 0
        
        if "frame_objects" in data:
            frame_objects = data["frame_objects"]
            if isinstance(frame_objects, dict):
                frame_count = len(frame_objects)
            elif isinstance(frame_objects, list):
                frame_count = len(frame_objects)
        
        if "segments" in data:
            segments = data["segments"]
            assert isinstance(segments, list), "Segments must be a list"
            segment_count = len(segments)
        
        return {
            "frame_count": frame_count,
            "segment_count": segment_count,
            "found_fields": found_fields
        }
    
    def validate_video_file(self, video_path: str) -> Dict[str, Any]:
        """Validate a video file (usually preprocessed .mkv) exists and has content."""
        assert os.path.exists(video_path), f"Video file not found: {video_path}"
        assert os.path.getsize(video_path) > 1000, f"Video file too small: {video_path}"  # At least 1KB
        
        # Try to get basic info using ffprobe if available
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                format_info = info.get('format', {})
                return {
                    "size_bytes": os.path.getsize(video_path),
                    "duration": float(format_info.get('duration', 0)),
                    "format_name": format_info.get('format_name', 'unknown')
                }
        except Exception:
            pass
        
        return {
            "size_bytes": os.path.getsize(video_path),
            "duration": 0,
            "format_name": 'unknown'
        }
    
    @pytest.mark.slow
    def test_2stage_output_validation(self):
        """Test that 2-stage mode produces valid output files."""
        expected_files = [
            "test_output_validation.funscript",
            "test_output_validation_stage2_overlay.msgpack"
        ]
        
        result = self.run_cli_and_validate_basic_output("2-stage", expected_files)
        
        # Validate funscript file
        funscript_path = result["found_files"]["test_output_validation.funscript"]
        funscript_info = self.validate_funscript_file(funscript_path)
        
        assert funscript_info["action_count"] > 10, "2-stage should produce substantial funscript"
        assert funscript_info["duration"] > 1000, "Funscript should span significant time"
        
        # Validate stage 2 overlay
        msgpack_path = result["found_files"]["test_output_validation_stage2_overlay.msgpack"]
        msgpack_info = self.validate_msgpack_file(msgpack_path)
        
        assert msgpack_info["frame_count"] > 0, "Stage 2 overlay should contain frame data"
        
        print(f"✓ 2-stage output validation: {funscript_info['action_count']} actions, {msgpack_info['frame_count']} frames")
    
    @pytest.mark.slow
    def test_3stage_output_validation(self):
        """Test that 3-stage mode produces valid output files."""
        expected_files = [
            "test_output_validation.funscript",
            "test_output_validation_stage2_overlay.msgpack"
        ]
        
        # 3-stage might also produce preprocessed video
        result = self.run_cli_and_validate_basic_output("3-stage", expected_files)
        
        # Validate funscript file
        funscript_path = result["found_files"]["test_output_validation.funscript"]
        funscript_info = self.validate_funscript_file(funscript_path)
        
        assert funscript_info["action_count"] > 10, "3-stage should produce substantial funscript"
        
        # Validate stage 2 overlay
        msgpack_path = result["found_files"]["test_output_validation_stage2_overlay.msgpack"]
        msgpack_info = self.validate_msgpack_file(msgpack_path)
        
        # Check for preprocessed video
        preprocessed_video = os.path.join(result["output_dir"], "test_output_validation_preprocessed.mkv")
        if os.path.exists(preprocessed_video):
            video_info = self.validate_video_file(preprocessed_video)
            print(f"✓ Preprocessed video: {video_info['size_bytes']} bytes")
        
        print(f"✓ 3-stage output validation: {funscript_info['action_count']} actions, {msgpack_info['frame_count']} frames")
    
    @pytest.mark.slow
    def test_3stage_mixed_output_validation(self):
        """Test that 3-stage-mixed mode produces valid output files."""
        expected_files = [
            "test_output_validation.funscript"
        ]
        
        result = self.run_cli_and_validate_basic_output("3-stage-mixed", expected_files)
        
        # Validate funscript file
        funscript_path = result["found_files"]["test_output_validation.funscript"]
        funscript_info = self.validate_funscript_file(funscript_path)
        
        assert funscript_info["action_count"] > 10, "3-stage-mixed should produce substantial funscript"
        
        # Check if stage 2 overlay was created (might exist from previous runs or be created)
        msgpack_path = os.path.join(result["output_dir"], "test_output_validation_stage2_overlay.msgpack")
        if os.path.exists(msgpack_path):
            msgpack_info = self.validate_msgpack_file(msgpack_path)
            print(f"✓ Stage 2 overlay found: {msgpack_info['frame_count']} frames")
        
        print(f"✓ 3-stage-mixed output validation: {funscript_info['action_count']} actions")
    
    @pytest.mark.slow
    def test_oscillation_detector_output_validation(self):
        """Test that oscillation-detector mode produces valid output files."""
        expected_files = [
            "test_output_validation.funscript"
        ]
        
        result = self.run_cli_and_validate_basic_output("oscillation-detector", expected_files)
        
        # Validate funscript file
        funscript_path = result["found_files"]["test_output_validation.funscript"]
        funscript_info = self.validate_funscript_file(funscript_path)
        
        assert funscript_info["action_count"] > 5, "Oscillation detector should produce some actions"
        
        print(f"✓ Oscillation detector output validation: {funscript_info['action_count']} actions")
    
    def test_funscript_file_copy_validation(self):
        """Test that funscripts are copied next to video files when expected."""
        if not os.path.exists(self.test_video_path):
            pytest.skip("Test video not available")
        
        # Copy video to temp directory
        temp_video_name = "test_copy_validation.mp4"
        temp_video_path = os.path.join(self.temp_test_dir, temp_video_name)
        shutil.copy2(self.test_video_path, temp_video_path)
        
        # Run CLI processing with copy enabled (default)
        cmd = [
            "python", "main.py",
            temp_video_path,
            "--mode", "2-stage",
            "--overwrite"
        ]
        
        env = os.environ.copy()
        env['FUNGEN_TESTING'] = '1'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd='/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator'
        )
        
        assert result.returncode == 0, f"CLI processing failed: {result.stderr}"
        
        # Check that funscript was copied next to video
        expected_copy_path = os.path.join(self.temp_test_dir, "test_copy_validation.funscript")
        
        # Note: The copy behavior might depend on settings, so we'll check if it exists or not
        if os.path.exists(expected_copy_path):
            copy_info = self.validate_funscript_file(expected_copy_path)
            print(f"✓ Funscript copy found: {copy_info['action_count']} actions")
        else:
            print("ℹ Funscript copy not found (may be disabled in settings)")
        
        # The main output should still be in the output directory
        output_dir = os.path.join(self.temp_test_dir, "test_copy_validation")
        main_funscript = os.path.join(output_dir, "test_copy_validation.funscript")
        assert os.path.exists(main_funscript), "Main funscript output should exist"
        
        main_info = self.validate_funscript_file(main_funscript)
        print(f"✓ Main funscript output: {main_info['action_count']} actions")
    
    def test_batch_processing_output_validation(self):
        """Test that batch processing produces valid outputs for multiple files."""
        if not os.path.exists(self.test_video_path):
            pytest.skip("Test video not available")
        
        # Create multiple test videos
        batch_dir = os.path.join(self.temp_test_dir, "batch_test")
        os.makedirs(batch_dir)
        
        test_videos = ["video1.mp4", "video2.mp4"]
        for video_name in test_videos:
            video_path = os.path.join(batch_dir, video_name)
            shutil.copy2(self.test_video_path, video_path)
        
        # Run batch processing
        cmd = [
            "python", "main.py",
            batch_dir,
            "--mode", "2-stage",
            "--overwrite",
            "--recursive"
        ]
        
        env = os.environ.copy()
        env['FUNGEN_TESTING'] = '1'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 minutes for batch
            env=env,
            cwd='/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator'
        )
        
        assert result.returncode == 0, f"Batch processing failed: {result.stderr}"
        
        # Validate outputs for each video
        for video_name in test_videos:
            video_basename = os.path.splitext(video_name)[0]
            output_dir = os.path.join(batch_dir, video_basename)
            
            funscript_path = os.path.join(output_dir, f"{video_basename}.funscript")
            assert os.path.exists(funscript_path), f"Funscript not found for {video_name}"
            
            funscript_info = self.validate_funscript_file(funscript_path)
            assert funscript_info["action_count"] > 5, f"Insufficient actions for {video_name}"
            
            print(f"✓ Batch output for {video_name}: {funscript_info['action_count']} actions")
    
    def test_output_file_permissions_and_structure(self):
        """Test that output files have correct permissions and directory structure."""
        if not os.path.exists(self.test_video_path):
            pytest.skip("Test video not available")
        
        # Copy video to temp directory
        temp_video_name = "test_permissions.mp4"
        temp_video_path = os.path.join(self.temp_test_dir, temp_video_name)
        shutil.copy2(self.test_video_path, temp_video_path)
        
        # Run CLI processing
        cmd = [
            "python", "main.py",
            temp_video_path,
            "--mode", "2-stage",
            "--overwrite"
        ]
        
        env = os.environ.copy()
        env['FUNGEN_TESTING'] = '1'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd='/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator'
        )
        
        assert result.returncode == 0, f"CLI processing failed: {result.stderr}"
        
        # Check output directory structure
        output_dir = os.path.join(self.temp_test_dir, "test_permissions")
        assert os.path.isdir(output_dir), "Output directory should exist"
        
        # Check file permissions
        funscript_path = os.path.join(output_dir, "test_permissions.funscript")
        assert os.path.exists(funscript_path), "Funscript should exist"
        assert os.access(funscript_path, os.R_OK), "Funscript should be readable"
        assert os.access(funscript_path, os.W_OK), "Funscript should be writable"
        
        # Check that files are not executable
        stat_info = os.stat(funscript_path)
        assert not (stat_info.st_mode & 0o111), "Output files should not be executable"
        
        print("✓ File permissions and directory structure validated")


if __name__ == "__main__":
    pytest.main([__file__])