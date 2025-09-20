#!/usr/bin/env python
"""
Test script to verify the optimizations made to Stage 2 functions.
"""

import time
import numpy as np
from detection.cd.data_structures.frame_objects import FrameObject, LockedPenisState

def create_test_frames(n_frames=1000):
    """Create test frame objects for performance testing."""
    frames = []
    for i in range(n_frames):
        # Create a frame object
        frame = FrameObject(frame_id=i, yolo_input_size=640)
        
        # Set some test values
        frame.funscript_distance = np.random.randint(0, 101)
        
        # Set up locked penis state for some frames
        if i % 3 == 0:  # Every third frame has active penis tracking
            frame.locked_penis_state.active = True
            frame.locked_penis_state.box = (100.0, 200.0, 150.0, 250.0)
        
        frames.append(frame)
    
    return frames

def test_pass_7_optimization():
    """Test the optimized pass_7_smooth_and_normalize_distances function."""
    print("Testing pass_7_smooth_and_normalize_distances optimization...")
    
    # Create test data
    frames = create_test_frames(5000)  # 5000 frames should be enough to see performance difference
    
    # Import the function
    from detection.cd.stage_2_cd import pass_7_smooth_and_normalize_distances
    import logging
    
    # Create a mock app object
    class MockApp:
        def __init__(self):
            self.app_settings = None
    
    mock_app = MockApp()
    
    # Prepare output lists
    funscript_frames = []
    funscript_distances = []
    
    # Time the function
    start_time = time.time()
    pass_7_smooth_and_normalize_distances(mock_app, frames, funscript_frames, funscript_distances, None)
    end_time = time.time()
    
    print(f"Function completed in {end_time - start_time:.4f} seconds")
    print(f"Processed {len(frames)} frames")
    print(f"Output frames list length: {len(funscript_frames)}")
    print(f"Output distances list length: {len(funscript_distances)}")
    
    return end_time - start_time

def test_normalize_funscript_optimization():
    """Test the optimized _normalize_funscript_sparse_per_segment function."""
    print("\nTesting _normalize_funscript_sparse_per_segment optimization...")
    
    # Create test data
    frames = create_test_frames(1000)
    
    # Import the function
    from detection.cd.stage_2_cd import _normalize_funscript_sparse_per_segment
    import logging
    
    # Create some mock segments for testing
    class MockSegment:
        def __init__(self, start_idx, end_idx, frames):
            self.segment_frame_objects = frames[start_idx:end_idx]
            self.major_position = "Handjob" if start_idx % 2 == 0 else "Blowjob"
    
    # Create a few segments
    segments = [
        MockSegment(0, 300, frames),
        MockSegment(300, 700, frames),
        MockSegment(700, 1000, frames)
    ]
    
    # Create a mock app object
    class MockApp:
        def __init__(self):
            self.app_settings = None
    
    mock_app = MockApp()
    
    # Time the function
    start_time = time.time()
    _normalize_funscript_sparse_per_segment(mock_app, frames, segments, None)
    end_time = time.time()
    
    print(f"Function completed in {end_time - start_time:.4f} seconds")
    print(f"Processed {len(frames)} frames across {len(segments)} segments")
    
    return end_time - start_time

def test_signal_enhancement_optimization():
    """Test the optimized _apply_signal_enhancement function."""
    print("\nTesting _apply_signal_enhancement optimization...")
    
    # Create test data
    frames = create_test_frames(5000)
    
    # Import the function
    from detection.cd.stage_2_cd import _apply_signal_enhancement
    import logging
    
    # Create a mock app object
    class MockApp:
        def __init__(self):
            self.app_settings = type('MockSettings', (), {'get': lambda self, key, default: default})()
    
    mock_app = MockApp()
    
    # Time the function
    start_time = time.time()
    _apply_signal_enhancement(mock_app, frames, None)
    end_time = time.time()
    
    print(f"Function completed in {end_time - start_time:.4f} seconds")
    print(f"Processed {len(frames)} frames")
    
    return end_time - start_time

def main():
    """Run all optimization tests."""
    print("Running Stage 2 optimization tests...\n")
    
    # Test each function
    time1 = test_pass_7_optimization()
    time2 = test_normalize_funscript_optimization()
    time3 = test_signal_enhancement_optimization()
    
    total_time = time1 + time2 + time3
    print(f"\nTotal time for all tests: {total_time:.4f} seconds")
    
    print("\nOptimization tests completed!")

if __name__ == "__main__":
    main()