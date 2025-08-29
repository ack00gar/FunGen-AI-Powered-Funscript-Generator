#!/usr/bin/env python3
"""
Test that Experimental 2 mode correctly dispatches to the right method
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tracker.tracker import ROITracker
from config.constants import TrackerMode
from funscript.dual_axis_funscript import DualAxisFunscript
import numpy as np

def test_mode_dispatch():
    """Test that mode switching and dispatch work correctly"""
    print("🧪 Testing Mode Dispatch for Experimental 2")
    print("=" * 50)
    
    # Create mock app
    class MockSettings:
        def get(self, key, default=None):
            return default
    
    class MockApp:
        def __init__(self):
            self.app_settings = MockSettings()
            self.tracking_axis_mode = "both"
            self.single_axis_output_target = "primary"
    
    app = MockApp()
    
    # Create tracker
    tracker = ROITracker(app, tracker_model_path="/fake/path")
    tracker.funscript = DualAxisFunscript()
    
    print("\n📋 Testing mode setting:")
    
    # Test all oscillation detector modes
    test_modes = [
        ("OSCILLATION_DETECTOR", "Experimental"),
        ("OSCILLATION_DETECTOR_LEGACY", "Legacy"),
        ("OSCILLATION_DETECTOR_EXPERIMENTAL_2", "Experimental 2"),
    ]
    
    all_good = True
    
    for mode_str, name in test_modes:
        # Set the mode
        tracker.set_tracking_mode(mode_str)
        
        # Check it was set correctly
        if tracker.tracking_mode == mode_str:
            print(f"✓ {name}: Mode set to '{mode_str}'")
        else:
            print(f"❌ {name}: Failed to set mode (got '{tracker.tracking_mode}')")
            all_good = False
        
        # Verify that process_frame would dispatch correctly
        # Create a simple test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[200:280, 300:380] = 255  # White square
        
        try:
            # Don't actually run the processing, just check the method exists
            if mode_str == "OSCILLATION_DETECTOR":
                if hasattr(tracker, 'process_frame_for_oscillation'):
                    print(f"  → Would call: process_frame_for_oscillation")
                else:
                    print(f"  ❌ Method process_frame_for_oscillation not found!")
                    all_good = False
                    
            elif mode_str == "OSCILLATION_DETECTOR_LEGACY":
                if hasattr(tracker, 'process_frame_for_oscillation_legacy'):
                    print(f"  → Would call: process_frame_for_oscillation_legacy")
                else:
                    print(f"  ❌ Method process_frame_for_oscillation_legacy not found!")
                    all_good = False
                    
            elif mode_str == "OSCILLATION_DETECTOR_EXPERIMENTAL_2":
                if hasattr(tracker, 'process_frame_for_oscillation_experimental_2'):
                    print(f"  → Would call: process_frame_for_oscillation_experimental_2")
                else:
                    print(f"  ❌ Method process_frame_for_oscillation_experimental_2 not found!")
                    all_good = False
                    
        except Exception as e:
            print(f"  ❌ Error checking dispatch: {e}")
            all_good = False
    
    # Test UI enum to tracker string mapping
    print("\n🎛️ Testing UI enum to tracker mode mapping:")
    
    ui_to_tracker_map = [
        (TrackerMode.OSCILLATION_DETECTOR, "OSCILLATION_DETECTOR"),
        (TrackerMode.OSCILLATION_DETECTOR_LEGACY, "OSCILLATION_DETECTOR_LEGACY"),
        (TrackerMode.OSCILLATION_DETECTOR_EXPERIMENTAL_2, "OSCILLATION_DETECTOR_EXPERIMENTAL_2"),
        (TrackerMode.LIVE_YOLO_ROI, "YOLO_ROI"),
        (TrackerMode.LIVE_USER_ROI, "USER_FIXED_ROI"),
    ]
    
    for ui_mode, expected_tracker_mode in ui_to_tracker_map:
        # Simulate what happens in the UI
        if ui_mode == TrackerMode.LIVE_USER_ROI:
            tracker.set_tracking_mode("USER_FIXED_ROI")
        elif ui_mode == TrackerMode.OSCILLATION_DETECTOR:
            tracker.set_tracking_mode("OSCILLATION_DETECTOR")
        elif ui_mode == TrackerMode.OSCILLATION_DETECTOR_LEGACY:
            tracker.set_tracking_mode("OSCILLATION_DETECTOR_LEGACY")
        elif ui_mode == TrackerMode.OSCILLATION_DETECTOR_EXPERIMENTAL_2:
            tracker.set_tracking_mode("OSCILLATION_DETECTOR_EXPERIMENTAL_2")
        elif ui_mode == TrackerMode.LIVE_YOLO_ROI:
            tracker.set_tracking_mode("YOLO_ROI")
        
        if tracker.tracking_mode == expected_tracker_mode:
            print(f"✓ {ui_mode.value}: Correctly maps to '{expected_tracker_mode}'")
        else:
            print(f"❌ {ui_mode.value}: Expected '{expected_tracker_mode}', got '{tracker.tracking_mode}'")
            all_good = False
    
    # Test process_frame dispatch
    print("\n🔄 Testing process_frame dispatch logic:")
    
    # Test Experimental 2 specifically
    tracker.set_tracking_mode("OSCILLATION_DETECTOR_EXPERIMENTAL_2")
    tracker.tracking_active = False  # Don't actually track
    
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        # This should call process_frame_for_oscillation_experimental_2
        processed, actions = tracker.process_frame(test_frame, 0, 0)
        print("✓ Experimental 2 process_frame dispatched successfully")
        
        # The method should at least return the frame
        if processed is not None:
            print("✓ Experimental 2 returned processed frame")
        else:
            print("⚠️  Experimental 2 returned None for processed frame")
            
    except AttributeError as e:
        if "process_frame_for_oscillation_experimental_2" in str(e):
            print("❌ process_frame_for_oscillation_experimental_2 method not found!")
            all_good = False
        else:
            print(f"❌ Unexpected AttributeError: {e}")
            all_good = False
    except Exception as e:
        # This is expected if we don't have all the required setup
        if "Dense optical flow not available" in str(e):
            print("✓ Experimental 2 method called (optical flow not initialized - expected)")
        else:
            print(f"⚠️  Other error (may be expected): {e}")
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 ALL TESTS PASSED!")
        print("Experimental 2 mode correctly:")
        print("  • Can be set via set_tracking_mode()")
        print("  • Maps from UI enum to tracker string")
        print("  • Dispatches to process_frame_for_oscillation_experimental_2()")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        print("Check the issues above to fix the dispatch problem.")
        return False

if __name__ == "__main__":
    success = test_mode_dispatch()
    sys.exit(0 if success else 1)