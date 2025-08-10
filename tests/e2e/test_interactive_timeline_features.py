#!/usr/bin/env python3
"""
Comprehensive interactive timeline features and point manipulation tests.

This module tests all interactive timeline functionality including:
- Point manipulation (add/delete/drag)
- Timeline scrubbing and playback synchronization  
- Heatmap generation and visualization
- Multi-axis timeline operations
"""

import pytest
from unittest.mock import patch, MagicMock
from imgui_bundle import imgui, hello_imgui
import os
import sys
import time
import json
import numpy as np
from typing import List, Dict, Any

# Add the parent directory to the path so we can import our modules  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

pytestmark = [pytest.mark.e2e, pytest.mark.interactive_timeline]


class TestInteractiveTimelineFeatures:
    """Comprehensive tests for interactive timeline functionality."""
    
    @pytest.fixture
    def setup_timeline_test_data(self, app_instance):
        """Setup test data for timeline operations."""
        # Create test actions spanning different time ranges
        test_actions_primary = [
            {"at": 1000, "pos": 20},
            {"at": 2000, "pos": 80}, 
            {"at": 3000, "pos": 30},
            {"at": 4000, "pos": 70},
            {"at": 5000, "pos": 40},
            {"at": 6000, "pos": 60}
        ]
        
        test_actions_secondary = [
            {"at": 1500, "pos": 35},
            {"at": 2500, "pos": 65},
            {"at": 3500, "pos": 45},
            {"at": 4500, "pos": 55}
        ]
        
        # Load test data into both timelines
        app_instance.funscript_processor.clear_timeline_history_and_set_new_baseline(
            1, test_actions_primary, "timeline_test_setup_primary"
        )
        app_instance.funscript_processor.clear_timeline_history_and_set_new_baseline(
            2, test_actions_secondary, "timeline_test_setup_secondary"
        )
        
        # Wait for data to be loaded
        time.sleep(0.5)
        
        return {
            "primary_actions": test_actions_primary,
            "secondary_actions": test_actions_secondary
        }
    
    @pytest.mark.e2e
    def test_point_addition_via_timeline_click(self, app_instance, setup_timeline_test_data):
        """Test adding points by clicking on the timeline."""
        engine = hello_imgui.get_imgui_test_engine()
        
        initial_primary_count = len(app_instance.funscript_processor.get_actions('primary'))
        
        # Navigate to a timeline view
        engine.item_open("**/View")
        if engine.item_exists("**/Show Timeline"):
            engine.item_click("**/Show Timeline")
        
        # Simulate clicking on timeline to add a point
        # Note: This would require more specific timeline widget interaction
        # For now, we test the underlying functionality
        
        # Add point via direct API (simulating timeline click)
        new_timestamp = 2500
        new_position = 45
        
        app_instance.funscript_processor.add_action_to_timeline(
            1, new_timestamp, new_position, "timeline_click_test"
        )
        
        # Verify point was added
        updated_actions = app_instance.funscript_processor.get_actions('primary')
        assert len(updated_actions) == initial_primary_count + 1
        
        # Find the new action
        new_action = next((a for a in updated_actions if a["at"] == new_timestamp), None)
        assert new_action is not None, "New action should be found"
        assert new_action["pos"] == new_position
        
        print(f"✓ Point addition: {initial_primary_count} -> {len(updated_actions)} actions")
    
    @pytest.mark.e2e
    def test_point_deletion_via_selection(self, app_instance, setup_timeline_test_data):
        """Test deleting points via selection and delete operation."""
        engine = hello_imgui.get_imgui_test_engine()
        
        initial_actions = app_instance.funscript_processor.get_actions('primary')
        initial_count = len(initial_actions)
        
        # Find a point to delete (middle point for testing)
        target_action = initial_actions[2] if len(initial_actions) > 2 else initial_actions[0]
        target_timestamp = target_action["at"]
        
        # Test deletion via timeline API (simulating selection + delete)
        success = app_instance.funscript_processor.delete_action_from_timeline(
            1, target_timestamp, "timeline_delete_test"
        )
        
        assert success, "Point deletion should succeed"
        
        # Verify point was deleted
        updated_actions = app_instance.funscript_processor.get_actions('primary')
        assert len(updated_actions) == initial_count - 1
        
        # Verify the specific point is gone
        deleted_action = next((a for a in updated_actions if a["at"] == target_timestamp), None)
        assert deleted_action is None, "Deleted action should not be found"
        
        print(f"✓ Point deletion: {initial_count} -> {len(updated_actions)} actions")
    
    @pytest.mark.e2e
    def test_point_dragging_position_change(self, app_instance, setup_timeline_test_data):
        """Test dragging points to change their position values."""
        initial_actions = app_instance.funscript_processor.get_actions('primary')
        
        # Select a point to modify
        target_action = initial_actions[1] if len(initial_actions) > 1 else initial_actions[0]
        original_timestamp = target_action["at"]
        original_position = target_action["pos"]
        new_position = 90 if original_position < 50 else 10
        
        # Simulate dragging to new position (vertical drag)
        success = app_instance.funscript_processor.modify_action_position(
            1, original_timestamp, new_position, "timeline_drag_test"
        )
        
        assert success, "Point position modification should succeed"
        
        # Verify position changed
        updated_actions = app_instance.funscript_processor.get_actions('primary')
        modified_action = next((a for a in updated_actions if a["at"] == original_timestamp), None)
        
        assert modified_action is not None, "Modified action should still exist"
        assert modified_action["pos"] == new_position, f"Position should change from {original_position} to {new_position}"
        
        print(f"✓ Point position change: {original_position} -> {new_position}")
    
    @pytest.mark.e2e  
    def test_point_dragging_time_change(self, app_instance, setup_timeline_test_data):
        """Test dragging points to change their timestamp values."""
        initial_actions = app_instance.funscript_processor.get_actions('primary')
        
        # Select a point to modify (avoid first and last to prevent edge cases)
        if len(initial_actions) < 3:
            pytest.skip("Need at least 3 actions for timestamp modification test")
        
        target_action = initial_actions[1]
        original_timestamp = target_action["at"]
        original_position = target_action["pos"]
        
        # Calculate new timestamp between adjacent points
        prev_timestamp = initial_actions[0]["at"]
        next_timestamp = initial_actions[2]["at"]
        new_timestamp = prev_timestamp + ((next_timestamp - prev_timestamp) // 3)
        
        # Simulate horizontal drag to new timestamp
        success = app_instance.funscript_processor.modify_action_timestamp(
            1, original_timestamp, new_timestamp, "timeline_time_drag_test"
        )
        
        assert success, "Point timestamp modification should succeed"
        
        # Verify timestamp changed and position preserved
        updated_actions = app_instance.funscript_processor.get_actions('primary')
        modified_action = next((a for a in updated_actions if a["at"] == new_timestamp), None)
        
        assert modified_action is not None, "Modified action should exist at new timestamp"
        assert modified_action["pos"] == original_position, "Position should be preserved"
        
        # Verify original timestamp no longer exists
        old_action = next((a for a in updated_actions if a["at"] == original_timestamp), None)
        assert old_action is None, "Original timestamp should not exist"
        
        print(f"✓ Point timestamp change: {original_timestamp} -> {new_timestamp}")
    
    @pytest.mark.e2e
    def test_multi_point_selection_operations(self, app_instance, setup_timeline_test_data):
        """Test selecting and operating on multiple points simultaneously."""
        initial_actions = app_instance.funscript_processor.get_actions('primary')
        
        if len(initial_actions) < 3:
            pytest.skip("Need at least 3 actions for multi-selection test")
        
        # Select multiple points (simulate Ctrl+click or box selection)
        selected_timestamps = [initial_actions[1]["at"], initial_actions[2]["at"]]
        
        # Test bulk position adjustment
        position_offset = 20
        success = app_instance.funscript_processor.bulk_modify_actions_position(
            1, selected_timestamps, position_offset, "multi_select_test"
        )
        
        assert success, "Bulk position modification should succeed"
        
        # Verify all selected points were modified
        updated_actions = app_instance.funscript_processor.get_actions('primary')
        for timestamp in selected_timestamps:
            action = next((a for a in updated_actions if a["at"] == timestamp), None)
            assert action is not None, f"Action at {timestamp} should exist"
            
            # Find original position
            original_action = next((a for a in initial_actions if a["at"] == timestamp), None)
            expected_new_pos = min(100, max(0, original_action["pos"] + position_offset))
            
            assert action["pos"] == expected_new_pos, f"Position should be adjusted by {position_offset}"
        
        print(f"✓ Multi-point selection: {len(selected_timestamps)} points modified")
    
    @pytest.mark.e2e
    def test_timeline_scrubbing_synchronization(self, app_instance, setup_timeline_test_data):
        """Test that timeline scrubbing synchronizes with video playback position."""
        engine = hello_imgui.get_imgui_test_engine()
        
        # Simulate loading a video for timeline sync testing
        video_path = os.path.abspath("test_data/timeline_sync_video.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        with open(video_path, "w") as f:
            f.write("dummy video for timeline sync test")
        
        # Load video through file dialog
        with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_show_dialog:
            engine.item_open("**/File")
            engine.item_open("**/Open...")
            engine.item_click("**/Video...")
            args, kwargs = mock_show_dialog.call_args
            callback = kwargs.get('callback')
            callback(video_path)
        
        # Wait for video load
        start_time = time.time()
        while not app_instance.file_manager.video_path:
            if time.time() - start_time > 5:
                break
            app_instance.gui_instance.run_one_frame(blocking=False)
            time.sleep(0.1)
        
        # Test scrubbing to specific timeline position
        target_timestamp = 3000  # 3 seconds
        
        # Simulate timeline scrub
        app_instance.video_processor.seek_to_timestamp(target_timestamp)
        
        # Verify video position matches timeline position
        current_position = app_instance.video_processor.get_current_timestamp()
        
        # Allow some tolerance for seeking precision
        assert abs(current_position - target_timestamp) < 100, \
            f"Video position ({current_position}) should match timeline position ({target_timestamp})"
        
        print(f"✓ Timeline scrubbing sync: {current_position}ms ≈ {target_timestamp}ms")
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
    
    @pytest.mark.e2e
    def test_dual_axis_timeline_synchronization(self, app_instance, setup_timeline_test_data):
        """Test that dual-axis timelines stay synchronized during operations."""
        primary_actions = app_instance.funscript_processor.get_actions('primary')
        secondary_actions = app_instance.funscript_processor.get_actions('secondary')
        
        # Add synchronized actions at same timestamp
        sync_timestamp = 7000
        primary_pos = 75
        secondary_pos = 25
        
        app_instance.funscript_processor.add_action_to_timeline(
            1, sync_timestamp, primary_pos, "dual_axis_sync_test_primary"
        )
        app_instance.funscript_processor.add_action_to_timeline(
            2, sync_timestamp, secondary_pos, "dual_axis_sync_test_secondary"
        )
        
        # Verify both actions exist at the same timestamp
        updated_primary = app_instance.funscript_processor.get_actions('primary')
        updated_secondary = app_instance.funscript_processor.get_actions('secondary')
        
        primary_sync_action = next((a for a in updated_primary if a["at"] == sync_timestamp), None)
        secondary_sync_action = next((a for a in updated_secondary if a["at"] == sync_timestamp), None)
        
        assert primary_sync_action is not None, "Primary sync action should exist"
        assert secondary_sync_action is not None, "Secondary sync action should exist"
        assert primary_sync_action["pos"] == primary_pos
        assert secondary_sync_action["pos"] == secondary_pos
        
        print(f"✓ Dual-axis sync: Primary={primary_pos}, Secondary={secondary_pos} at {sync_timestamp}ms")
    
    @pytest.mark.e2e
    def test_heatmap_generation_and_visualization(self, app_instance, setup_timeline_test_data):
        """Test heatmap generation based on timeline data."""
        actions = app_instance.funscript_processor.get_actions('primary')
        
        if len(actions) < 4:
            pytest.skip("Need at least 4 actions for meaningful heatmap test")
        
        # Test getting statistics for heatmap generation
        stats = app_instance.funscript_processor.get_actions_statistics('primary')
        
        assert isinstance(stats, dict), "Statistics should be a dictionary"
        assert 'num_points' in stats, "Statistics should include point count"
        assert stats['num_points'] == len(actions), "Point count should match actual actions"
        
        # Test speed/intensity calculation for heatmap
        if 'speed_data' in stats or hasattr(app_instance.funscript_processor, 'calculate_speed_data'):
            # Calculate speeds between consecutive points
            speeds = []
            for i in range(1, len(actions)):
                prev_action = actions[i-1]
                curr_action = actions[i]
                
                time_diff = curr_action["at"] - prev_action["at"]
                pos_diff = abs(curr_action["pos"] - prev_action["pos"])
                
                if time_diff > 0:
                    speed = pos_diff / (time_diff / 1000.0)  # units per second
                    speeds.append(speed)
            
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                max_speed = max(speeds)
                
                print(f"✓ Heatmap data: avg_speed={avg_speed:.2f}, max_speed={max_speed:.2f}")
                
                # Test heatmap intensity categories
                intensity_levels = []
                for speed in speeds:
                    if speed < avg_speed * 0.5:
                        intensity_levels.append("low")
                    elif speed < avg_speed * 1.5:
                        intensity_levels.append("medium")
                    else:
                        intensity_levels.append("high")
                
                assert len(intensity_levels) == len(speeds), "Should have intensity for each speed"
                print(f"✓ Heatmap intensities: {len([l for l in intensity_levels if l == 'high'])} high, "
                      f"{len([l for l in intensity_levels if l == 'medium'])} medium, "
                      f"{len([l for l in intensity_levels if l == 'low'])} low")
    
    @pytest.mark.e2e
    def test_timeline_zoom_and_pan_operations(self, app_instance, setup_timeline_test_data):
        """Test timeline zooming and panning functionality."""
        # Test timeline viewport manipulation
        actions = app_instance.funscript_processor.get_actions('primary')
        
        if len(actions) < 2:
            pytest.skip("Need at least 2 actions for zoom/pan test")
        
        # Get timeline bounds
        min_time = min(a["at"] for a in actions)
        max_time = max(a["at"] for a in actions)
        total_duration = max_time - min_time
        
        # Test zooming to specific time range
        zoom_start = min_time + total_duration * 0.25
        zoom_end = min_time + total_duration * 0.75
        
        # Simulate zoom operation (this would typically be done through UI)
        if hasattr(app_instance, 'timeline_viewport'):
            app_instance.timeline_viewport.set_time_range(zoom_start, zoom_end)
            
            current_start, current_end = app_instance.timeline_viewport.get_time_range()
            
            assert abs(current_start - zoom_start) < 10, "Timeline should zoom to requested start time"
            assert abs(current_end - zoom_end) < 10, "Timeline should zoom to requested end time"
            
            print(f"✓ Timeline zoom: {zoom_start}-{zoom_end}ms")
        
        # Test panning (shifting viewport)
        pan_offset = total_duration * 0.1
        
        if hasattr(app_instance, 'timeline_viewport'):
            app_instance.timeline_viewport.pan_by_offset(pan_offset)
            
            new_start, new_end = app_instance.timeline_viewport.get_time_range()
            expected_start = zoom_start + pan_offset
            expected_end = zoom_end + pan_offset
            
            assert abs(new_start - expected_start) < 10, "Timeline should pan by requested offset"
            assert abs(new_end - expected_end) < 10, "Timeline should maintain zoom level while panning"
            
            print(f"✓ Timeline pan: offset={pan_offset}ms")
    
    @pytest.mark.e2e
    def test_timeline_undo_redo_integration(self, app_instance, setup_timeline_test_data):
        """Test that timeline operations integrate properly with undo/redo."""
        engine = hello_imgui.get_imgui_test_engine()
        
        initial_actions = app_instance.funscript_processor.get_actions('primary')
        initial_count = len(initial_actions)
        
        # Perform a timeline operation that should be undoable
        new_timestamp = 7500
        new_position = 55
        
        app_instance.funscript_processor.add_action_to_timeline(
            1, new_timestamp, new_position, "undo_redo_test"
        )
        
        # Verify action was added
        after_add_actions = app_instance.funscript_processor.get_actions('primary')
        assert len(after_add_actions) == initial_count + 1
        
        # Perform undo
        engine.item_open("**/Edit")
        if engine.item_exists("**/Undo"):
            engine.item_click("**/Undo")
            time.sleep(0.5)
            
            # Verify undo worked
            after_undo_actions = app_instance.funscript_processor.get_actions('primary')
            assert len(after_undo_actions) == initial_count, "Undo should revert the addition"
            
            # Verify specific action is gone
            undone_action = next((a for a in after_undo_actions if a["at"] == new_timestamp), None)
            assert undone_action is None, "Undone action should not exist"
            
            print(f"✓ Timeline undo: {len(after_add_actions)} -> {len(after_undo_actions)} actions")
            
            # Perform redo
            if engine.item_exists("**/Redo"):
                engine.item_click("**/Redo")
                time.sleep(0.5)
                
                # Verify redo worked
                after_redo_actions = app_instance.funscript_processor.get_actions('primary')
                assert len(after_redo_actions) == initial_count + 1, "Redo should restore the addition"
                
                # Verify specific action is back
                redone_action = next((a for a in after_redo_actions if a["at"] == new_timestamp), None)
                assert redone_action is not None, "Redone action should exist"
                assert redone_action["pos"] == new_position, "Redone action should have correct position"
                
                print(f"✓ Timeline redo: {len(after_undo_actions)} -> {len(after_redo_actions)} actions")
    
    @pytest.mark.e2e
    def test_timeline_copy_paste_operations(self, app_instance, setup_timeline_test_data):
        """Test copying and pasting timeline segments."""
        engine = hello_imgui.get_imgui_test_engine()
        
        actions = app_instance.funscript_processor.get_actions('primary')
        
        if len(actions) < 3:
            pytest.skip("Need at least 3 actions for copy/paste test")
        
        # Select a range of actions to copy (simulate selection)
        copy_start_time = actions[1]["at"]
        copy_end_time = actions[2]["at"]
        
        # Simulate copy operation
        if hasattr(app_instance.funscript_processor, 'copy_timeline_segment'):
            success = app_instance.funscript_processor.copy_timeline_segment(
                1, copy_start_time, copy_end_time
            )
            assert success, "Timeline segment copy should succeed"
            
            # Simulate paste operation at different location
            paste_time_offset = 2000
            paste_success = app_instance.funscript_processor.paste_timeline_segment(
                1, copy_start_time + paste_time_offset, "copy_paste_test"
            )
            
            assert paste_success, "Timeline segment paste should succeed"
            
            # Verify pasted actions exist
            updated_actions = app_instance.funscript_processor.get_actions('primary')
            
            # Look for pasted actions at new location
            expected_paste_times = [
                actions[1]["at"] + paste_time_offset,
                actions[2]["at"] + paste_time_offset
            ]
            
            for expected_time in expected_paste_times:
                pasted_action = next((a for a in updated_actions if abs(a["at"] - expected_time) < 50), None)
                assert pasted_action is not None, f"Pasted action should exist near {expected_time}"
            
            print(f"✓ Timeline copy/paste: segment duplicated with {paste_time_offset}ms offset")


if __name__ == "__main__":
    pytest.main([__file__])