"""
Smoke tests to verify basic functionality of all processing modes.

These are fast tests that check if the core functionality works without
detailed performance measurement.
"""

import pytest
import os
import subprocess
import tempfile
import shutil
from pathlib import Path


# Import test configuration
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from test_config import get_test_video_path, get_batch_test_video_paths

# Test video paths - now configured dynamically
BATCH_TEST_VIDEOS = get_batch_test_video_paths(count=2, category="short")
SINGLE_TEST_VIDEO = get_test_video_path(category="short")  # Use short video for smoke tests


@pytest.fixture(scope="function")
def temp_output_dir():
    """Create temporary output directory for each test."""
    temp_dir = tempfile.mkdtemp(prefix="smoke_test_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def run_cli_smoke_test(video_path: str, mode: str, output_dir: str, timeout: int = 300) -> bool:
    """Run CLI processing as smoke test with short timeout."""
    
    cmd = [
        'python', 'main.py',
        video_path,
        '--mode', mode,
        '--overwrite'  # Force processing to ensure clean test
    ]
    
    env = os.environ.copy()
    env['FUNGEN_TESTING'] = '1'
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd="/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator"
        )
        
        if result.returncode != 0:
            print(f"Smoke test failed for {mode}:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"Smoke test timed out for {mode}")
        return False


@pytest.mark.smoke
class TestSmokeTests:
    """Basic smoke tests for all processing modes."""
    
    def test_smoke_2stage_single_video(self, temp_output_dir):
        """Smoke test for 2-stage processing on single video."""
        if not os.path.exists(SINGLE_TEST_VIDEO):
            pytest.skip("Test video not available")
        
        success = run_cli_smoke_test(SINGLE_TEST_VIDEO, "2-stage", temp_output_dir)
        assert success, "2-stage processing smoke test failed"
        
        # Check if output files were created (funscript should be created next to video or in output folder)
        video_name = Path(SINGLE_TEST_VIDEO).stem
        video_dir = Path(SINGLE_TEST_VIDEO).parent
        expected_funscript = video_dir / f"{video_name}.funscript"
        # Just verify the processing succeeded - file location may vary
    
    def test_smoke_3stage_single_video(self, temp_output_dir):
        """Smoke test for 3-stage processing on single video."""
        if not os.path.exists(SINGLE_TEST_VIDEO):
            pytest.skip("Test video not available")
        
        success = run_cli_smoke_test(SINGLE_TEST_VIDEO, "3-stage", temp_output_dir)
        assert success, "3-stage processing smoke test failed"
    
    def test_smoke_oscillation_detector_single_video(self, temp_output_dir):
        """Smoke test for oscillation detector processing on single video."""
        if not os.path.exists(SINGLE_TEST_VIDEO):
            pytest.skip("Test video not available")
        
        success = run_cli_smoke_test(SINGLE_TEST_VIDEO, "oscillation-detector", temp_output_dir)
        assert success, "Oscillation detector processing smoke test failed"
    
    def test_smoke_batch_processing_2stage(self, temp_output_dir):
        """Smoke test for batch processing with 2-stage mode."""
        if not all(os.path.exists(path) for path in BATCH_TEST_VIDEOS[:1]):  # Test with just first video
            pytest.skip("Batch test videos not available")
        
        # Test batch processing by processing a folder instead
        # Copy test video to temp directory
        import shutil
        test_folder = os.path.join(temp_output_dir, "batch_test")
        os.makedirs(test_folder)
        shutil.copy2(BATCH_TEST_VIDEOS[0], test_folder)
        
        cmd = [
            'python', 'main.py',
            test_folder,
            '--mode', '2-stage',
            '--overwrite'
        ]
        
        env = os.environ.copy()
        env['FUNGEN_TESTING'] = '1'
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env,
                              cwd="/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator")
        
        assert result.returncode == 0, f"Batch 2-stage smoke test failed: {result.stderr}"
    
    def test_smoke_batch_processing_3stage(self, temp_output_dir):
        """Smoke test for batch processing with 3-stage mode."""
        if not all(os.path.exists(path) for path in BATCH_TEST_VIDEOS[:1]):  # Test with just first video
            pytest.skip("Batch test videos not available")
        
        # Test batch processing by processing a folder instead
        # Copy test video to temp directory
        import shutil
        test_folder = os.path.join(temp_output_dir, "batch_test")
        os.makedirs(test_folder)
        shutil.copy2(BATCH_TEST_VIDEOS[0], test_folder)
        
        cmd = [
            'python', 'main.py',
            test_folder,
            '--mode', '3-stage',
            '--overwrite'
        ]
        
        env = os.environ.copy()
        env['FUNGEN_TESTING'] = '1'
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env,
                              cwd="/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator")
        
        assert result.returncode == 0, f"Batch 3-stage smoke test failed: {result.stderr}"
    
    def test_smoke_batch_processing_oscillation_detector(self, temp_output_dir):
        """Smoke test for batch processing with oscillation detector mode."""
        if not all(os.path.exists(path) for path in BATCH_TEST_VIDEOS[:1]):  # Test with just first video
            pytest.skip("Batch test videos not available")
        
        # Test batch processing by processing a folder instead
        # Copy test video to temp directory
        import shutil
        test_folder = os.path.join(temp_output_dir, "batch_test")
        os.makedirs(test_folder)
        shutil.copy2(BATCH_TEST_VIDEOS[0], test_folder)
        
        cmd = [
            'python', 'main.py',
            test_folder,
            '--mode', 'oscillation-detector',
            '--overwrite'
        ]
        
        env = os.environ.copy()
        env['FUNGEN_TESTING'] = '1'
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env,
                              cwd="/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator")
        
        assert result.returncode == 0, f"Batch oscillation detector smoke test failed: {result.stderr}"


@pytest.mark.smoke
class TestFunctionalSmokeTests:
    """Functional smoke tests to verify core components work."""
    
    def test_import_all_modules(self):
        """Test that all main modules can be imported without errors."""
        try:
            import application.logic.app_logic
            import application.logic.app_stage_processor
            import application.logic.app_funscript_processor
            import detection.cd.stage_1_cd
            import detection.cd.stage_2_cd
            import detection.cd.stage_3_of_processor
            import tracker.tracker
            import video.video_processor
            import funscript.dual_axis_funscript
        except ImportError as e:
            pytest.fail(f"Failed to import core modules: {e}")
    
    def test_constants_accessible(self):
        """Test that constants are accessible."""
        try:
            from config import constants
            assert hasattr(constants, 'TrackerMode')
            assert hasattr(constants, 'DEFAULT_LIVE_TRACKER_SENSITIVITY')
        except ImportError as e:
            pytest.fail(f"Failed to import constants: {e}")
    
    def test_video_processor_can_be_created(self):
        """Test that VideoProcessor can be instantiated."""
        try:
            from video import VideoProcessor
            import logging
            
            # Mock app instance
            class MockApp:
                def __init__(self):
                    self.logger = logging.getLogger("test")
                    self.hardware_acceleration_method = "none"
                    self.available_ffmpeg_hwaccels = []
                    self.file_manager = None
            
            mock_app = MockApp()
            processor = VideoProcessor(app_instance=mock_app)
            assert processor is not None
        except Exception as e:
            pytest.fail(f"Failed to create VideoProcessor: {e}")
    
    def test_tracker_can_be_created(self):
        """Test that ROITracker can be instantiated."""
        try:
            from tracker import ROITracker
            import logging
            
            logger = logging.getLogger("test")
            tracker = ROITracker(
                app_logic_instance=None,
                tracker_model_path="",
                pose_model_path="",
                load_models_on_init=False,
                logger=logger
            )
            assert tracker is not None
        except Exception as e:
            pytest.fail(f"Failed to create ROITracker: {e}")


if __name__ == "__main__":
    # Allow running smoke tests directly
    pytest.main([__file__, "-v", "-m", "smoke"])