"""
Performance benchmarks for different processing methods.

This module tests and tracks performance metrics for:
- 2-Stage processing
- 3-Stage processing  
- Oscillation Detector mode
- Batch processing vs single video processing

Results are stored for comparison and regression detection.
"""

import pytest
import time
import json
import os
import subprocess
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import tempfile
import shutil

# Import test configuration
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from test_config import get_test_video_path, get_batch_test_video_paths

# Test video paths - now configured dynamically
BATCH_TEST_VIDEOS = get_batch_test_video_paths(count=2, category="short")
SINGLE_TEST_VIDEO = get_test_video_path(category="medium")  # Use medium video for single tests

# Performance results storage
PERFORMANCE_LOG_FILE = Path(__file__).parent / "performance_results.json"


class PerformanceTracker:
    """Track and store performance metrics."""
    
    def __init__(self):
        self.results = self._load_existing_results()
    
    def _load_existing_results(self) -> Dict[str, List[Dict]]:
        """Load existing performance results."""
        if PERFORMANCE_LOG_FILE.exists():
            try:
                with open(PERFORMANCE_LOG_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {}
    
    def _save_results(self):
        """Save performance results to file."""
        PERFORMANCE_LOG_FILE.parent.mkdir(exist_ok=True)
        with open(PERFORMANCE_LOG_FILE, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def record_result(self, test_name: str, mode: str, duration: float, 
                     video_info: Dict[str, Any], metrics: Dict[str, Any] = None):
        """Record a performance test result."""
        if test_name not in self.results:
            self.results[test_name] = []
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "duration_seconds": duration,
            "video_info": video_info,
            "metrics": metrics or {},
            "git_commit": self._get_git_commit()
        }
        
        self.results[test_name].append(result)
        self._save_results()
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                  capture_output=True, text=True, timeout=5)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "unknown"
    
    def get_performance_summary(self, test_name: str) -> Dict[str, Any]:
        """Get performance summary for a test."""
        if test_name not in self.results:
            return {}
        
        results_by_mode = {}
        for result in self.results[test_name]:
            mode = result["mode"]
            if mode not in results_by_mode:
                results_by_mode[mode] = []
            results_by_mode[mode].append(result["duration_seconds"])
        
        summary = {}
        for mode, durations in results_by_mode.items():
            if durations:
                summary[mode] = {
                    "count": len(durations),
                    "avg_duration": statistics.mean(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0,
                    "latest": durations[-1],
                    "trend": "improving" if len(durations) > 1 and durations[-1] < durations[-2] else "stable"
                }
        
        return summary


@pytest.fixture(scope="module")
def performance_tracker():
    """Performance tracker fixture."""
    return PerformanceTracker()


@pytest.fixture(scope="function")
def temp_output_dir():
    """Create temporary output directory for each test."""
    temp_dir = tempfile.mkdtemp(prefix="perf_test_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video information using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), {})
            
            return {
                "duration": float(data.get('format', {}).get('duration', 0)),
                "size_mb": round(int(data.get('format', {}).get('size', 0)) / (1024 * 1024), 2),
                "fps": eval(video_stream.get('r_frame_rate', '30/1')),
                "width": video_stream.get('width', 0),
                "height": video_stream.get('height', 0),
                "frame_count": int(video_stream.get('nb_frames', 0)) if video_stream.get('nb_frames') else None
            }
    except Exception as e:
        print(f"Warning: Could not get video info for {video_path}: {e}")
    
    return {"duration": 0, "size_mb": 0, "fps": 30, "width": 0, "height": 0}


def run_cli_processing(video_path: str, mode: str, output_dir: str, force_rerun: bool = True) -> Tuple[float, Dict[str, Any]]:
    """Run CLI processing and measure performance."""
    
    # Prepare command
    cmd = [
        'python', 'main.py',
        video_path,
        '--mode', mode,
        '--overwrite'  # Force processing to ensure clean test
    ]
    
    # Set environment
    env = os.environ.copy()
    env['FUNGEN_TESTING'] = '1'
    
    # Run and measure time
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes timeout
            env=env,
            cwd="/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator"
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        metrics = {
            "success": result.returncode == 0,
            "stdout_lines": len(result.stdout.splitlines()) if result.stdout else 0,
            "stderr_lines": len(result.stderr.splitlines()) if result.stderr else 0
        }
        
        if result.returncode != 0:
            print(f"CLI processing failed for {mode}: {result.stderr}")
            metrics["error"] = result.stderr[-500:]  # Last 500 chars of error
        
        return duration, metrics
        
    except subprocess.TimeoutExpired:
        return float('inf'), {"success": False, "error": "timeout"}


# Performance Tests

@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_batch_processing_2stage(self, performance_tracker, temp_output_dir):
        """Test batch processing performance with 2-stage mode."""
        if not all(os.path.exists(path) for path in BATCH_TEST_VIDEOS):
            pytest.skip("Batch test videos not available")
        
        # Create test folder with videos
        import shutil
        test_folder = os.path.join(temp_output_dir, "batch_test")
        os.makedirs(test_folder)
        for video in BATCH_TEST_VIDEOS:
            if os.path.exists(video):
                shutil.copy2(video, test_folder)
        
        cmd = [
            'python', 'main.py',
            test_folder,
            '--mode', '2-stage',
            '--overwrite',
            '--recursive'
        ]
        
        env = os.environ.copy()
        env['FUNGEN_TESTING'] = '1'
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env,
                              cwd="/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator")
        duration = time.time() - start_time
        
        # Record results
        video_info = {
            "type": "batch",
            "count": len(BATCH_TEST_VIDEOS),
            "total_size_mb": sum(get_video_info(v)["size_mb"] for v in BATCH_TEST_VIDEOS)
        }
        
        metrics = {
            "success": result.returncode == 0,
            "videos_processed": len(BATCH_TEST_VIDEOS)
        }
        
        performance_tracker.record_result("batch_2stage", "2-stage", duration, video_info, metrics)
        
        assert result.returncode == 0, f"Batch 2-stage processing failed: {result.stderr}"
        
    def test_batch_processing_3stage(self, performance_tracker, temp_output_dir):
        """Test batch processing performance with 3-stage mode."""
        if not all(os.path.exists(path) for path in BATCH_TEST_VIDEOS):
            pytest.skip("Batch test videos not available")
        
        # Create test folder with videos
        import shutil
        test_folder = os.path.join(temp_output_dir, "batch_test")
        os.makedirs(test_folder)
        for video in BATCH_TEST_VIDEOS:
            if os.path.exists(video):
                shutil.copy2(video, test_folder)
        
        cmd = [
            'python', 'main.py',
            test_folder,
            '--mode', '3-stage',
            '--overwrite',
            '--recursive'
        ]
        
        env = os.environ.copy()
        env['FUNGEN_TESTING'] = '1'
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env,
                              cwd="/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator")
        duration = time.time() - start_time
        
        # Record results
        video_info = {
            "type": "batch",
            "count": len(BATCH_TEST_VIDEOS),
            "total_size_mb": sum(get_video_info(v)["size_mb"] for v in BATCH_TEST_VIDEOS)
        }
        
        metrics = {
            "success": result.returncode == 0,
            "videos_processed": len(BATCH_TEST_VIDEOS)
        }
        
        performance_tracker.record_result("batch_3stage", "3-stage", duration, video_info, metrics)
        
        assert result.returncode == 0, f"Batch 3-stage processing failed: {result.stderr}"
    
    def test_batch_processing_oscillation_detector(self, performance_tracker, temp_output_dir):
        """Test batch processing performance with oscillation detector mode."""
        if not all(os.path.exists(path) for path in BATCH_TEST_VIDEOS):
            pytest.skip("Batch test videos not available")
        
        # Create test folder with videos
        import shutil
        test_folder = os.path.join(temp_output_dir, "batch_test")
        os.makedirs(test_folder)
        for video in BATCH_TEST_VIDEOS:
            if os.path.exists(video):
                shutil.copy2(video, test_folder)
        
        cmd = [
            'python', 'main.py',
            test_folder,
            '--mode', 'oscillation-detector',
            '--overwrite',
            '--recursive'
        ]
        
        env = os.environ.copy()
        env['FUNGEN_TESTING'] = '1'
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env,
                              cwd="/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator")
        duration = time.time() - start_time
        
        # Record results
        video_info = {
            "type": "batch",
            "count": len(BATCH_TEST_VIDEOS),
            "total_size_mb": sum(get_video_info(v)["size_mb"] for v in BATCH_TEST_VIDEOS)
        }
        
        metrics = {
            "success": result.returncode == 0,
            "videos_processed": len(BATCH_TEST_VIDEOS)
        }
        
        performance_tracker.record_result("batch_oscillation", "oscillation-detector", duration, video_info, metrics)
        
        assert result.returncode == 0, f"Batch oscillation detector processing failed: {result.stderr}"
    
    def test_batch_processing_3stage_mixed(self, performance_tracker, temp_output_dir):
        """Test batch processing performance with 3-stage-mixed mode."""
        if not all(os.path.exists(path) for path in BATCH_TEST_VIDEOS):
            pytest.skip("Batch test videos not available")
        
        # Create test folder with videos
        import shutil
        test_folder = os.path.join(temp_output_dir, "batch_test")
        os.makedirs(test_folder)
        for video in BATCH_TEST_VIDEOS:
            if os.path.exists(video):
                shutil.copy2(video, test_folder)
        
        cmd = [
            'python', 'main.py',
            test_folder,
            '--mode', '3-stage-mixed',
            '--overwrite',
            '--recursive'
        ]
        
        env = os.environ.copy()
        env['FUNGEN_TESTING'] = '1'
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env,
                              cwd="/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator")
        duration = time.time() - start_time
        
        # Record results
        video_info = {
            "type": "batch",
            "count": len(BATCH_TEST_VIDEOS),
            "total_size_mb": sum(get_video_info(v)["size_mb"] for v in BATCH_TEST_VIDEOS)
        }
        
        metrics = {
            "success": result.returncode == 0,
            "videos_processed": len(BATCH_TEST_VIDEOS)
        }
        
        performance_tracker.record_result("batch_3stage_mixed", "3-stage-mixed", duration, video_info, metrics)
        
        assert result.returncode == 0, f"Batch 3-stage-mixed processing failed: {result.stderr}"
    
    @pytest.mark.slow
    def test_single_video_2stage(self, performance_tracker, temp_output_dir):
        """Test single video processing performance with 2-stage mode."""
        if not os.path.exists(SINGLE_TEST_VIDEO):
            pytest.skip("Single test video not available")
        
        duration, metrics = run_cli_processing(SINGLE_TEST_VIDEO, "2-stage", temp_output_dir)
        video_info = get_video_info(SINGLE_TEST_VIDEO)
        video_info["type"] = "single"
        video_info["path"] = os.path.basename(SINGLE_TEST_VIDEO)
        
        performance_tracker.record_result("single_2stage", "2-stage", duration, video_info, metrics)
        
        assert metrics["success"], f"Single 2-stage processing failed"
    
    @pytest.mark.slow
    def test_single_video_3stage(self, performance_tracker, temp_output_dir):
        """Test single video processing performance with 3-stage mode."""
        if not os.path.exists(SINGLE_TEST_VIDEO):
            pytest.skip("Single test video not available")
        
        duration, metrics = run_cli_processing(SINGLE_TEST_VIDEO, "3-stage", temp_output_dir)
        video_info = get_video_info(SINGLE_TEST_VIDEO)
        video_info["type"] = "single"
        video_info["path"] = os.path.basename(SINGLE_TEST_VIDEO)
        
        performance_tracker.record_result("single_3stage", "3-stage", duration, video_info, metrics)
        
        assert metrics["success"], f"Single 3-stage processing failed"
        
    @pytest.mark.slow
    def test_single_video_oscillation_detector(self, performance_tracker, temp_output_dir):
        """Test single video processing performance with oscillation detector mode."""
        if not os.path.exists(SINGLE_TEST_VIDEO):
            pytest.skip("Single test video not available")
        
        duration, metrics = run_cli_processing(SINGLE_TEST_VIDEO, "oscillation-detector", temp_output_dir)
        video_info = get_video_info(SINGLE_TEST_VIDEO)
        video_info["type"] = "single"
        video_info["path"] = os.path.basename(SINGLE_TEST_VIDEO)
        
        performance_tracker.record_result("single_oscillation", "oscillation-detector", duration, video_info, metrics)
        
        assert metrics["success"], f"Single oscillation detector processing failed"
    
    @pytest.mark.slow
    def test_single_video_3stage_mixed(self, performance_tracker, temp_output_dir):
        """Test single video processing performance with 3-stage-mixed mode."""
        if not os.path.exists(SINGLE_TEST_VIDEO):
            pytest.skip("Single test video not available")
        
        duration, metrics = run_cli_processing(SINGLE_TEST_VIDEO, "3-stage-mixed", temp_output_dir)
        video_info = get_video_info(SINGLE_TEST_VIDEO)
        video_info["type"] = "single"
        video_info["path"] = os.path.basename(SINGLE_TEST_VIDEO)
        
        performance_tracker.record_result("single_3stage_mixed", "3-stage-mixed", duration, video_info, metrics)
        
        assert metrics["success"], f"Single 3-stage-mixed processing failed"
    
    @pytest.mark.performance
    def test_stage2_to_stage3_carryover_performance(self, performance_tracker, temp_output_dir):
        """Test performance of Stage 2 to Stage 3 data carryover."""
        from test_config import config
        test_video = get_test_video_path(category="short")
        video_name = Path(test_video).name
        existing_project_dir = config.get_existing_project_dir(video_name)
        
        if not os.path.exists(test_video):
            pytest.skip("Test video not available")
        
        if not os.path.exists(existing_project_dir):
            pytest.skip("Existing project with Stage 2 data not available")
        
        # Check for existing Stage 2 data
        stage2_overlay = os.path.join(existing_project_dir, "test_koogar_extra_short_A_stage2_overlay.msgpack")
        if not os.path.exists(stage2_overlay):
            pytest.skip("Stage 2 overlay data not available")
        
        # Create temp setup
        temp_video_name = "carryover_test.mp4"
        temp_output_dir_path = os.path.join(temp_output_dir, "carryover_test")
        os.makedirs(temp_output_dir_path, exist_ok=True)
        
        # Copy Stage 2 artifacts
        stage2_files = [
            "test_koogar_extra_short_A_stage2_overlay.msgpack",
            "test_koogar_extra_short_A_preprocessed.mkv"
        ]
        
        for filename in stage2_files:
            src_path = os.path.join(existing_project_dir, filename)
            if os.path.exists(src_path):
                dest_filename = filename.replace("test_koogar_extra_short_A", "carryover_test")
                dest_path = os.path.join(temp_output_dir_path, dest_filename)
                shutil.copy2(src_path, dest_path)
        
        # Create symlink to video
        temp_video_path = os.path.join(temp_output_dir, temp_video_name)
        try:
            os.symlink(test_video, temp_video_path)
        except OSError:
            shutil.copy2(test_video, temp_video_path)
        
        # Test Stage 3 mixed with carryover
        start_time = time.time()
        duration_mixed, metrics_mixed = run_cli_processing(temp_video_path, "3-stage-mixed", temp_output_dir)
        
        video_info = get_video_info(test_video)
        video_info["type"] = "carryover_test"
        video_info["path"] = os.path.basename(test_video)
        
        metrics_mixed["has_stage2_carryover"] = True
        metrics_mixed["stage2_overlay_size_mb"] = round(os.path.getsize(stage2_overlay) / (1024 * 1024), 2) if os.path.exists(stage2_overlay) else 0
        
        performance_tracker.record_result("stage2_to_stage3_carryover", "3-stage-mixed-carryover", duration_mixed, video_info, metrics_mixed)
        
        # Performance should be reasonable even with carryover
        assert duration_mixed < 600, f"Carryover processing took too long: {duration_mixed:.1f}s"
        
        print(f"✓ Stage 2 to Stage 3 carryover completed in {duration_mixed:.1f}s")


@pytest.mark.performance
def test_performance_regression_check(performance_tracker):
    """Check for performance regressions."""
    
    summary = {}
    for test_name in ["batch_2stage", "batch_3stage", "batch_oscillation", "batch_3stage_mixed", "single_2stage", "single_3stage", "single_oscillation", "single_3stage_mixed", "stage2_to_stage3_carryover"]:
        summary[test_name] = performance_tracker.get_performance_summary(test_name)
    
    # Print performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    for test_name, modes in summary.items():
        if modes:
            print(f"\n{test_name.upper()}:")
            for mode, stats in modes.items():
                print(f"  {mode}: {stats['latest']:.1f}s (avg: {stats['avg_duration']:.1f}s, runs: {stats['count']})")
                if stats['count'] > 1:
                    change = ((stats['latest'] - stats['avg_duration']) / stats['avg_duration']) * 100
                    trend = "⬇️ FASTER" if change < -5 else "⬆️ SLOWER" if change > 5 else "➡️ STABLE"
                    print(f"      {trend} ({change:+.1f}%)")
    
    print("\n" + "="*60)
    
    # Simple regression check - fail if latest run is >50% slower than average
    for test_name, modes in summary.items():
        for mode, stats in modes.items():
            if stats['count'] > 2:  # Need at least 3 runs for comparison
                regression_threshold = stats['avg_duration'] * 1.5
                if stats['latest'] > regression_threshold:
                    pytest.fail(f"Performance regression detected in {test_name} {mode}: "
                              f"{stats['latest']:.1f}s > {regression_threshold:.1f}s threshold")


if __name__ == "__main__":
    # Allow running performance tests directly
    pytest.main([__file__, "-v", "-m", "performance"])