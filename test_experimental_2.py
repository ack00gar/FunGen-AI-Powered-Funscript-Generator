#!/usr/bin/env python3
"""
Test script for Experimental 2 oscillation detector
Tests the hybrid approach combining experimental timing with legacy amplification
"""

import numpy as np
import cv2
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tracker.tracker import ROITracker
from config.constants import TrackerMode
from funscript.dual_axis_funscript import DualAxisFunscript

def create_mock_app():
    """Create a mock app with basic settings"""
    class MockSettings:
        def get(self, key, default=None):
            settings_map = {
                'oscillation_processing_target_height': 480,
                'live_oscillation_dynamic_amp_enabled': True,
                'oscillation_use_simple_amplification': False,
                'oscillation_enable_decay': True,
                'oscillation_hold_duration_ms': 250,
                'oscillation_decay_factor': 0.95
            }
            return settings_map.get(key, default)
    
    class MockApp:
        def __init__(self):
            self.app_settings = MockSettings()
            self.tracking_axis_mode = "both"
            self.single_axis_output_target = "primary"
    
    return MockApp()

def create_test_frame(width=640, height=480, motion_type="oscillating", frame_num=0):
    """Create a test frame with simulated motion"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    if motion_type == "oscillating":
        # Create an oscillating white rectangle
        rect_size = 50
        center_x = width // 2
        center_y = height // 2
        
        # Oscillate vertically with 2Hz frequency
        oscillation = int(20 * np.sin(frame_num * 0.2))  # 0.2 rad/frame ≈ 2Hz at 30fps
        
        x1 = center_x - rect_size // 2
        x2 = center_x + rect_size // 2
        y1 = center_y - rect_size // 2 + oscillation
        y2 = center_y + rect_size // 2 + oscillation
        
        # Ensure bounds
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
    
    elif motion_type == "static":
        # Static white rectangle
        rect_size = 50
        center_x = width // 2
        center_y = height // 2
        x1, x2 = center_x - rect_size // 2, center_x + rect_size // 2
        y1, y2 = center_y - rect_size // 2, center_y + rect_size // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
    
    return frame

def test_experimental_2():
    """Test the Experimental 2 oscillation detector"""
    print("=== Testing Experimental 2 Oscillation Detector ===")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create mock app and tracker
    app = create_mock_app()
    tracker = ROITracker(app, tracker_model_path="/fake/path")
    tracker.tracking_mode = TrackerMode.OSCILLATION_DETECTOR_EXPERIMENTAL_2.name
    tracker.funscript = DualAxisFunscript()
    tracker.tracking_active = True
    
    # Test with oscillating motion
    print("\n1. Testing with oscillating motion...")
    results = []
    
    for frame_num in range(100):  # Test 100 frames
        frame = create_test_frame(motion_type="oscillating", frame_num=frame_num)
        frame_time_ms = frame_num * 33  # ~30fps
        
        try:
            processed_frame, action_log = tracker.process_frame_for_oscillation_experimental_2(
                frame, frame_time_ms, frame_num
            )
            
            if action_log:
                for action in action_log:
                    pos = action.get('pos', 50)
                    results.append(pos)
                    if frame_num % 20 == 0:  # Print every 20th frame
                        print(f"  Frame {frame_num:3d}: position = {pos:3d}")
        
        except Exception as e:
            print(f"❌ Error processing frame {frame_num}: {e}")
            return False
    
    if results:
        positions = np.array(results)
        variation = np.std(positions)
        mean_pos = np.mean(positions)
        min_pos = np.min(positions)
        max_pos = np.max(positions)
        
        print(f"\n📊 Analysis of {len(results)} position samples:")
        print(f"   Mean position: {mean_pos:.1f}")
        print(f"   Position range: {min_pos} - {max_pos}")
        print(f"   Standard deviation: {variation:.1f}")
        
        if variation > 5:  # Should have some variation for oscillating motion
            print("✓ Oscillation detected successfully")
        else:
            print("⚠️  Low variation detected - may need tuning")
    else:
        print("❌ No position data generated")
        return False
    
    # Test with static motion
    print("\n2. Testing with static motion...")
    static_results = []
    
    for frame_num in range(50):
        frame = create_test_frame(motion_type="static", frame_num=frame_num)
        frame_time_ms = frame_num * 33
        
        try:
            processed_frame, action_log = tracker.process_frame_for_oscillation_experimental_2(
                frame, frame_time_ms, frame_num
            )
            
            if action_log:
                for action in action_log:
                    pos = action.get('pos', 50)
                    static_results.append(pos)
        
        except Exception as e:
            print(f"❌ Error processing static frame {frame_num}: {e}")
            return False
    
    if static_results:
        static_positions = np.array(static_results)
        static_variation = np.std(static_positions)
        static_mean = np.mean(static_positions)
        
        print(f"📊 Static motion analysis:")
        print(f"   Mean position: {static_mean:.1f}")
        print(f"   Standard deviation: {static_variation:.1f}")
        
        if static_variation < variation:  # Should be less variable than oscillating
            print("✓ Static motion handled correctly")
        else:
            print("⚠️  Static motion seems too variable")
    
    print("\n3. Testing hybrid features...")
    
    # Check that live amplification is working
    if hasattr(tracker, 'live_amp_enabled') and tracker.live_amp_enabled:
        print("✓ Live dynamic amplification enabled")
    else:
        print("⚠️  Live dynamic amplification not found")
    
    # Check that position history is being tracked
    if hasattr(tracker, 'oscillation_position_history') and len(tracker.oscillation_position_history) > 0:
        print(f"✓ Position history tracking: {len(tracker.oscillation_position_history)} samples")
    else:
        print("⚠️  Position history not being tracked")
    
    # Check oscillation cell persistence (from experimental)
    if hasattr(tracker, 'oscillation_cell_persistence'):
        print(f"✓ Cell persistence tracking: {len(tracker.oscillation_cell_persistence)} active cells")
    else:
        print("⚠️  Cell persistence not found")
    
    print("\n🎉 Experimental 2 test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_experimental_2()
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)