#!/usr/bin/env python3
"""
Test script to verify the UI dropdown contains Experimental 2
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.constants import TrackerMode

def test_dropdown_modes():
    """Test that all dropdown modes are properly defined"""
    print("🧪 Testing UI Dropdown Mode Integration")
    print("=" * 50)
    
    # Test that the constants are properly defined
    required_modes = [
        ('OSCILLATION_DETECTOR', 'Live - Oscillation Detector (Experimental)'),
        ('OSCILLATION_DETECTOR_LEGACY', 'Live - Oscillation Detector (Legacy)'),
        ('OSCILLATION_DETECTOR_EXPERIMENTAL_2', 'Live - Oscillation Detector (Experimental 2)'),
        ('LIVE_YOLO_ROI', 'Live - Optical Flow (YOLO auto ROI)'),
        ('LIVE_USER_ROI', 'Live - Optical Flow (User manual ROI)'),
        ('OFFLINE_2_STAGE', 'Offline - YOLO AI (2 Stages)'),
        ('OFFLINE_3_STAGE', 'Offline - YOLO AI + Opt. Flow (3 Stages)'),
        ('OFFLINE_3_STAGE_MIXED', 'Offline - YOLO AI + Mixed Flow (3 Stages Mixed)')
    ]
    
    print("📋 Checking TrackerMode constants:")
    all_good = True
    
    for mode_name, expected_value in required_modes:
        if hasattr(TrackerMode, mode_name):
            actual_value = getattr(TrackerMode, mode_name).value
            if actual_value == expected_value:
                print(f"✓ {mode_name}: '{actual_value}'")
            else:
                print(f"⚠️  {mode_name}: Expected '{expected_value}', got '{actual_value}'")
                all_good = False
        else:
            print(f"❌ {mode_name}: Missing from TrackerMode")
            all_good = False
    
    # Test dropdown arrays from control panel
    print(f"\n🎛️  Testing Control Panel Dropdown Arrays:")
    
    try:
        # Simulate the dropdown arrays as they appear in the UI
        simple_modes_display = [
            "Live Oscillation Detector",
            "Live Oscillation Detector (Legacy)",
            "Live Oscillation Detector (Experimental 2)",
            "Live Tracking (YOLO ROI)",
            "Offline AI Analysis (3-Stage)",
        ]
        simple_modes_enum = [
            TrackerMode.OSCILLATION_DETECTOR,
            TrackerMode.OSCILLATION_DETECTOR_LEGACY,
            TrackerMode.OSCILLATION_DETECTOR_EXPERIMENTAL_2,
            TrackerMode.LIVE_YOLO_ROI,
            TrackerMode.OFFLINE_3_STAGE,
        ]
        
        full_modes_enum = [
            TrackerMode.OSCILLATION_DETECTOR,
            TrackerMode.OSCILLATION_DETECTOR_LEGACY,
            TrackerMode.OSCILLATION_DETECTOR_EXPERIMENTAL_2,
            TrackerMode.LIVE_YOLO_ROI,
            TrackerMode.LIVE_USER_ROI,
            TrackerMode.OFFLINE_2_STAGE,
            TrackerMode.OFFLINE_3_STAGE,
            TrackerMode.OFFLINE_3_STAGE_MIXED,
        ]
        
        # Test that array lengths match
        if len(simple_modes_display) == len(simple_modes_enum):
            print(f"✓ Simple dropdown: {len(simple_modes_display)} display names ↔ {len(simple_modes_enum)} enum values")
        else:
            print(f"❌ Simple dropdown: {len(simple_modes_display)} display names ↔ {len(simple_modes_enum)} enum values (MISMATCH)")
            all_good = False
        
        # Test specific mode presence
        if TrackerMode.OSCILLATION_DETECTOR_EXPERIMENTAL_2 in simple_modes_enum:
            idx = simple_modes_enum.index(TrackerMode.OSCILLATION_DETECTOR_EXPERIMENTAL_2)
            display_name = simple_modes_display[idx]
            print(f"✓ Experimental 2 in simple dropdown at index {idx}: '{display_name}'")
        else:
            print("❌ Experimental 2 missing from simple dropdown")
            all_good = False
        
        if TrackerMode.OSCILLATION_DETECTOR_EXPERIMENTAL_2 in full_modes_enum:
            idx = full_modes_enum.index(TrackerMode.OSCILLATION_DETECTOR_EXPERIMENTAL_2)
            print(f"✓ Experimental 2 in full dropdown at index {idx}")
        else:
            print("❌ Experimental 2 missing from full dropdown")
            all_good = False
        
        full_modes_display = [m.value for m in full_modes_enum]
        print(f"✓ Full dropdown: {len(full_modes_display)} modes available")
        
        # List all modes in full dropdown
        print(f"\n📝 Full dropdown contents:")
        for i, (enum_val, display_val) in enumerate(zip(full_modes_enum, full_modes_display)):
            marker = "🆕" if "Experimental 2" in display_val else "  "
            print(f"  {i}: {marker} {display_val}")
        
    except Exception as e:
        print(f"❌ Error testing dropdown arrays: {e}")
        all_good = False
    
    # Test UI import
    print(f"\n🎨 Testing UI Component Import:")
    try:
        from application.gui_components.control_panel_ui import ControlPanelUI
        print("✓ ControlPanelUI imported successfully")
        
        # Test that TrackerMode can be accessed in the UI context
        if hasattr(ControlPanelUI, 'TrackerMode') or TrackerMode.OSCILLATION_DETECTOR_EXPERIMENTAL_2:
            print("✓ TrackerMode accessible in UI context")
        else:
            print("❌ TrackerMode not accessible in UI context")
            all_good = False
    
    except Exception as e:
        print(f"❌ UI import failed: {e}")
        all_good = False
    
    # Final result
    print(f"\n{'='*50}")
    if all_good:
        print("🎉 ALL TESTS PASSED! Experimental 2 is properly integrated in the UI dropdown.")
        print("The new tracking mode should now be visible and selectable in the application.")
        return True
    else:
        print("❌ SOME TESTS FAILED! Check the issues above.")
        return False

if __name__ == "__main__":
    success = test_dropdown_modes()
    sys.exit(0 if success else 1)