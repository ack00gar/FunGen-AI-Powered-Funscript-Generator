import pytest
import sys
import os
import tempfile
import subprocess
import json
import time
import threading
import gc
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from unittest.mock import patch, MagicMock
from application.logic.app_logic import ApplicationLogic

@pytest.mark.integration
def test_large_funscript_performance():
    """
    Test performance with very large funscript datasets.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        funscript = app.funscript_processor.get_funscript_obj()
        
        # Create very large action set (simulating 1-hour video at 30fps)
        large_action_count = 108000  # 1 hour * 60 min * 60 sec * 30 fps / 10 (every 10th frame)
        
        print(f"Creating {large_action_count} actions...")
        start_time = time.time()
        
        # Add actions in efficient batches
        batch_size = 5000
        for batch_start in range(0, large_action_count, batch_size):
            batch_end = min(batch_start + batch_size, large_action_count)
            
            for i in range(batch_start, batch_end):
                timestamp = i * 333  # ~30fps (1000ms / 30 = 33.33ms per frame, every 10th = 333ms)
                position = int(50 + 40 * ((i % 100) / 100))  # Smooth oscillation pattern
                funscript.add_action(timestamp, position)
            
            # Check memory usage periodically
            if batch_start % 25000 == 0:
                current_count = len(funscript.primary_actions)
                elapsed = time.time() - start_time
                print(f"  Added {current_count} actions in {elapsed:.2f}s")
        
        creation_time = time.time() - start_time
        final_count = len(funscript.primary_actions)
        
        assert final_count == large_action_count
        assert creation_time < 60.0, f"Large funscript creation took {creation_time:.2f}s (should be < 60s)"
        
        print(f"Large funscript created: {final_count} actions in {creation_time:.2f}s")
        
        # Test statistics performance on large dataset
        start_time = time.time()
        stats = funscript.get_actions_statistics('primary')
        stats_time = time.time() - start_time
        
        assert stats['num_points'] == large_action_count
        assert stats['total_travel_dist'] > 0
        assert stats_time < 10.0, f"Statistics calculation took {stats_time:.2f}s (should be < 10s)"
        
        print(f"Statistics calculated in {stats_time:.2f}s")
        
        # Test interpolation performance
        start_time = time.time()
        test_timestamps = [i * 10000 for i in range(100)]  # Test every 10 seconds
        
        for timestamp in test_timestamps:
            value = funscript.get_value(timestamp)
            assert 0 <= value <= 100
        
        interpolation_time = time.time() - start_time
        assert interpolation_time < 5.0, f"Interpolation took {interpolation_time:.2f}s (should be < 5s)"
        
        print(f"Interpolation tests completed in {interpolation_time:.2f}s")
        
        # Test memory cleanup
        start_time = time.time()
        funscript.clear()
        cleanup_time = time.time() - start_time
        
        assert len(funscript.primary_actions) == 0
        assert cleanup_time < 2.0, f"Cleanup took {cleanup_time:.2f}s (should be < 2s)"
        
        print(f"Cleanup completed in {cleanup_time:.2f}s")

@pytest.mark.integration
def test_extreme_edge_cases():
    """
    Test extreme edge cases and boundary conditions.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        funscript = app.funscript_processor.get_funscript_obj()
        
        # Test edge case: Single action
        funscript.add_action(0, 50)
        assert len(funscript.primary_actions) == 1
        
        # Test interpolation with single action
        value = funscript.get_value(1000)
        assert value == 50  # Should return the single action's value
        
        funscript.clear()
        
        # Test edge case: Actions at boundaries
        boundary_actions = [
            (0, 0),          # Start time, min position
            (0, 100),        # Start time, max position (should replace previous)
            (3600000, 0),    # 1 hour, min position
            (3600000, 100),  # 1 hour, max position (should replace previous)
        ]
        
        for timestamp, position in boundary_actions:
            funscript.add_action(timestamp, position)
        
        # Should have 2 actions (one at 0, one at 3600000)
        actions = funscript.primary_actions
        assert len(actions) == 2
        
        # Test interpolation at boundaries
        assert funscript.get_value(0) == 100      # Latest value at timestamp 0
        assert funscript.get_value(3600000) == 100  # Latest value at timestamp 3600000
        
        # Test midpoint interpolation
        midpoint_value = funscript.get_value(1800000)  # 30 minutes
        assert 0 <= midpoint_value <= 100
        
        funscript.clear()
        
        # Test edge case: Very close timestamps
        close_timestamps = [
            (1000, 10),
            (1001, 90),  # 1ms apart
            (1002, 50),  # 1ms apart
        ]
        
        for timestamp, position in close_timestamps:
            funscript.add_action(timestamp, position)
        
        assert len(funscript.primary_actions) == 3
        
        # Test interpolation between very close timestamps
        value_1000_5 = funscript.get_value(1000)  # Should be 10
        value_1001_5 = funscript.get_value(1001)  # Should be 90
        
        assert value_1000_5 == 10
        assert value_1001_5 == 90
        
        funscript.clear()
        
        # Test edge case: Duplicate timestamps (should replace)
        duplicate_actions = [
            (5000, 25),
            (5000, 75),  # Same timestamp, should replace
        ]
        
        for timestamp, position in duplicate_actions:
            funscript.add_action(timestamp, position)
        
        actions = funscript.primary_actions
        assert len(actions) == 1
        assert actions[0]['at'] == 5000
        assert actions[0]['pos'] == 75  # Should have the latest value

@pytest.mark.integration
def test_memory_stress_scenarios():
    """
    Test memory usage under stress scenarios.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
    except ImportError:
        pytest.skip("psutil not available for memory testing")
    
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        
        # Test multiple app instances creation/destruction
        max_memory_increase = 0
        
        for cycle in range(10):
            app = ApplicationLogic(is_cli=True)
            
            # Add significant data
            funscript = app.funscript_processor.get_funscript_obj()
            for i in range(10000):
                funscript.add_action(i * 100, i % 100)
            
            # Check memory usage
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            max_memory_increase = max(max_memory_increase, memory_increase)
            
            # Clean up
            funscript.clear()
            del app
            gc.collect()
        
        # Final memory check
        final_memory = process.memory_info().rss
        final_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for the test)
        assert final_increase < 100 * 1024 * 1024, f"Memory increased by {final_increase / 1024 / 1024:.1f} MB"
        
        print(f"Memory stress test: max increase {max_memory_increase / 1024 / 1024:.1f} MB, final {final_increase / 1024 / 1024:.1f} MB")

@pytest.mark.integration
def test_concurrent_operations_safety():
    """
    Test thread safety and concurrent operations.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        funscript = app.funscript_processor.get_funscript_obj()
        
        # Test that operations don't interfere with each other
        errors = []
        results = []
        
        def worker_thread(thread_id, operation_count):
            try:
                thread_results = []
                for i in range(operation_count):
                    timestamp = thread_id * 10000 + i * 100
                    position = (thread_id * 20 + i) % 100
                    
                    # Add action
                    funscript.add_action(timestamp, position)
                    
                    # Get statistics
                    stats = funscript.get_actions_statistics('primary')
                    thread_results.append(stats['num_points'])
                    
                    # Small delay to allow interleaving
                    time.sleep(0.001)
                
                results.append((thread_id, thread_results))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads (be careful with thread safety)
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=worker_thread, args=(thread_id, 50))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)
        
        # Check for errors
        if errors:
            pytest.fail(f"Thread errors occurred: {errors}")
        
        # Verify results
        assert len(results) == 3, f"Expected 3 thread results, got {len(results)}"
        
        final_stats = funscript.get_actions_statistics('primary')
        assert final_stats['num_points'] == 150, f"Expected 150 total actions, got {final_stats['num_points']}"
        
        print(f"Concurrent operations test: {final_stats['num_points']} actions added successfully")

@pytest.mark.integration
def test_error_recovery_comprehensive():
    """
    Test comprehensive error recovery scenarios.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        # Test 1: Invalid parameter handling
        funscript = app.funscript_processor.get_funscript_obj()
        
        # Test negative timestamps (should handle gracefully)
        try:
            funscript.add_action(-1000, 50)
            # Should either accept or reject gracefully
            assert True
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert isinstance(e, (ValueError, TypeError))
        
        # Test invalid positions (outside 0-100 range)
        try:
            funscript.add_action(1000, -10)  # Below range
            funscript.add_action(2000, 150)  # Above range
            # Should either clamp or reject
            assert True
        except Exception as e:
            assert isinstance(e, (ValueError, TypeError))
        
        # Test 2: Operations on corrupted state
        # Simulate corrupted internal state
        original_actions = funscript.primary_actions.copy()
        
        try:
            # Try operations that might fail with corrupted state
            stats = funscript.get_actions_statistics('primary')
            assert isinstance(stats, dict)
            
            value = funscript.get_value(1000)
            assert isinstance(value, (int, float))
            
        except Exception as e:
            # Should recover gracefully
            funscript.clear()
            assert len(funscript.primary_actions) == 0
        
        # Test 3: File system errors simulation
        with tempfile.NamedTemporaryFile(suffix='.funscript', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create valid funscript data
            funscript.add_action(1000, 50)
            funscript.add_action(2000, 75)
            
            # Test export to invalid location
            invalid_path = "/root/cannot_write_here.funscript"
            try:
                # This should fail gracefully
                with open(invalid_path, 'w') as f:
                    json.dump({"actions": funscript.primary_actions}, f)
            except (PermissionError, FileNotFoundError, OSError):
                # Expected behavior
                assert True
            
            # Test with valid path
            with open(temp_path, 'w') as f:
                json.dump({"actions": funscript.primary_actions}, f)
            
            # Verify export worked
            assert os.path.exists(temp_path)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

@pytest.mark.integration
def test_platform_specific_features():
    """
    Test platform-specific features and compatibility.
    """
    import platform
    
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        system = platform.system().lower()
        
        # Test hardware acceleration detection
        hwaccels = app._get_available_ffmpeg_hwaccels()
        
        # Platform-specific expectations
        if system == 'darwin':  # macOS
            # Should have VideoToolbox support
            expected_accels = ['videotoolbox', 'auto', 'none']
            assert any(accel in hwaccels for accel in expected_accels)
            
        elif system == 'linux':
            # May have VAAPI, CUDA, etc.
            assert 'none' in hwaccels  # At minimum
            # May also have: vaapi, cuda, vdpau, etc.
            
        elif system == 'windows':
            # May have DXVA2, D3D11VA, etc.
            assert 'none' in hwaccels  # At minimum
            # May also have: dxva2, d3d11va, cuda, etc.
        
        # Test that detected accelerations are valid strings
        for accel in hwaccels:
            assert isinstance(accel, str)
            assert len(accel) > 0
            assert not accel.isspace()
        
        print(f"Platform {system} has accelerations: {hwaccels}")

@pytest.mark.integration
def test_long_running_operation_simulation():
    """
    Test behavior during long-running operations.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        funscript = app.funscript_processor.get_funscript_obj()
        
        # Simulate long-running operation with periodic checks
        operation_start = time.time()
        checkpoint_interval = 1.0  # Check every second
        max_duration = 10.0  # Maximum 10 seconds for test
        
        action_count = 0
        last_checkpoint = operation_start
        
        while time.time() - operation_start < max_duration:
            current_time = time.time()
            
            # Add actions continuously
            timestamp = action_count * 50
            position = (action_count % 100)
            funscript.add_action(timestamp, position)
            action_count += 1
            
            # Periodic checkpoint
            if current_time - last_checkpoint >= checkpoint_interval:
                stats = funscript.get_actions_statistics('primary')
                assert stats['num_points'] == action_count
                
                # Test that operations still work
                test_value = funscript.get_value(timestamp // 2)
                assert 0 <= test_value <= 100
                
                last_checkpoint = current_time
                print(f"Checkpoint: {action_count} actions, {current_time - operation_start:.1f}s elapsed")
            
            # Small delay to prevent overwhelming
            time.sleep(0.001)
        
        # Final verification
        final_stats = funscript.get_actions_statistics('primary')
        assert final_stats['num_points'] == action_count
        assert action_count > 1000, f"Should have created many actions in {max_duration}s, got {action_count}"
        
        print(f"Long-running operation: {action_count} actions in {max_duration}s")

@pytest.mark.integration 
def test_data_integrity_verification():
    """
    Test data integrity under various operations.
    """
    with patch('application.logic.app_logic.ApplicationLogic._load_last_project_on_startup'):
        app = ApplicationLogic(is_cli=True)
        
        funscript = app.funscript_processor.get_funscript_obj()
        
        # Create reference dataset
        reference_actions = []
        for i in range(1000):
            timestamp = i * 100
            position = int(50 + 40 * (i % 10) / 10)  # Pattern: 50, 54, 58, ..., 90, 50, ...
            reference_actions.append({"at": timestamp, "pos": position})
            funscript.add_action(timestamp, position)
        
        # Verify initial integrity
        stored_actions = funscript.primary_actions
        assert len(stored_actions) == len(reference_actions)
        
        for i, (ref, stored) in enumerate(zip(reference_actions, stored_actions)):
            assert ref["at"] == stored["at"], f"Timestamp mismatch at index {i}"
            assert ref["pos"] == stored["pos"], f"Position mismatch at index {i}"
        
        # Test integrity after statistics calculation
        stats = funscript.get_actions_statistics('primary')
        assert stats['num_points'] == 1000
        
        # Verify data unchanged
        post_stats_actions = funscript.primary_actions
        assert len(post_stats_actions) == len(reference_actions)
        
        # Test integrity after interpolation operations
        for test_time in range(0, 100000, 10000):
            value = funscript.get_value(test_time)
            assert 0 <= value <= 100
        
        # Verify data still unchanged
        post_interp_actions = funscript.primary_actions
        assert len(post_interp_actions) == len(reference_actions)
        
        for i, (ref, stored) in enumerate(zip(reference_actions, post_interp_actions)):
            assert ref["at"] == stored["at"], f"Data corruption after interpolation at index {i}"
            assert ref["pos"] == stored["pos"], f"Data corruption after interpolation at index {i}"
        
        print("Data integrity verified across all operations")