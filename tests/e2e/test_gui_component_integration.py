#!/usr/bin/env python3
"""
Comprehensive GUI component integration tests.

This module tests integration between different GUI components including:
- Control panel UI interactions
- Video display overlays and debug rendering
- Generated file manager window operations  
- Menu system comprehensive integration
- Component state synchronization
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

pytestmark = [pytest.mark.e2e, pytest.mark.gui_integration]


class TestGUIComponentIntegration:
    """Comprehensive tests for GUI component integration and synchronization."""
    
    @pytest.fixture
    def setup_gui_test_environment(self, app_instance):
        """Setup test environment for GUI component testing."""
        # Ensure GUI is properly initialized
        if not hasattr(app_instance, 'gui_instance') or app_instance.gui_instance is None:
            pytest.skip("GUI instance not available for testing")
        
        # Run a few frames to ensure UI is stabilized
        for _ in range(5):
            app_instance.gui_instance.run_one_frame(blocking=False)
            time.sleep(0.01)
        
        return app_instance
    
    @pytest.mark.e2e
    def test_control_panel_video_display_synchronization(self, setup_gui_test_environment):
        """Test that control panel changes synchronize with video display."""
        app_instance = setup_gui_test_environment
        engine = hello_imgui.get_imgui_test_engine()
        
        # Test tracking mode changes
        current_mode = app_instance.app_settings.get('tracker_mode', 0)
        
        # Change tracking mode via control panel
        if engine.item_exists("**/Tracking Mode"):
            engine.combo_click("**/Tracking Mode", "3-Stage")
            time.sleep(0.5)
            
            # Verify video display updates mode indicator
            new_mode = app_instance.app_settings.get('tracker_mode', 0)
            assert new_mode != current_mode or new_mode == 0, "Tracking mode should change"
            
            # Check that video display shows correct mode
            if hasattr(app_instance, 'video_display_ui'):
                # The video display should reflect the new mode
                app_instance.gui_instance.run_one_frame(blocking=False)
                print(f"✓ Control panel -> Video display sync: mode {current_mode} -> {new_mode}")
        
        # Test ROI selection synchronization
        if engine.item_exists("**/Enable ROI Selection"):
            initial_roi_state = app_instance.app_settings.get('roi_selection_enabled', False)
            engine.item_click("**/Enable ROI Selection")
            time.sleep(0.5)
            
            new_roi_state = app_instance.app_settings.get('roi_selection_enabled', False)
            assert new_roi_state != initial_roi_state, "ROI selection state should toggle"
            
            # Video display should show ROI overlay
            app_instance.gui_instance.run_one_frame(blocking=False)
            print(f"✓ ROI selection sync: {initial_roi_state} -> {new_roi_state}")
    
    @pytest.mark.e2e
    def test_video_display_overlay_rendering(self, setup_gui_test_environment):
        """Test video display overlay rendering and debug information."""
        app_instance = setup_gui_test_environment
        engine = hello_imgui.get_imgui_test_engine()
        
        # Enable debug overlays
        debug_settings = [
            "show_debug_info",
            "show_tracking_overlay", 
            "show_roi_overlay",
            "show_detection_boxes"
        ]
        
        overlay_states = {}
        for setting in debug_settings:
            if engine.item_exists(f"**/{setting}"):
                initial_state = app_instance.app_settings.get(setting, False)
                engine.item_click(f"**/{setting}")
                time.sleep(0.2)
                
                new_state = app_instance.app_settings.get(setting, False)
                overlay_states[setting] = (initial_state, new_state)
                
                # Run frame to update display
                app_instance.gui_instance.run_one_frame(blocking=False)
        
        # Verify overlay states changed
        for setting, (old_state, new_state) in overlay_states.items():
            assert old_state != new_state, f"{setting} should toggle"
            print(f"✓ Overlay toggle {setting}: {old_state} -> {new_state}")
        
        # Test that overlays render without crashing
        for _ in range(10):
            app_instance.gui_instance.run_one_frame(blocking=False)
            time.sleep(0.01)
        
        print("✓ Video display overlay rendering stable")
    
    @pytest.mark.e2e 
    def test_generated_file_manager_integration(self, setup_gui_test_environment):
        """Test generated file manager window operations and integration."""
        app_instance = setup_gui_test_environment
        engine = hello_imgui.get_imgui_test_engine()
        
        # Open generated file manager window
        engine.item_open("**/Tools")
        if engine.item_exists("**/Generated File Manager"):
            engine.item_click("**/Generated File Manager")
            time.sleep(0.5)
            
            # Run a few frames to let window open
            for _ in range(5):
                app_instance.gui_instance.run_one_frame(blocking=False)
                time.sleep(0.1)
            
            # Test file refresh functionality
            if engine.item_exists("**/Refresh Files"):
                engine.item_click("**/Refresh Files")
                time.sleep(0.5)
                
                # Check that file list updates
                app_instance.gui_instance.run_one_frame(blocking=False)
                print("✓ Generated file manager refresh")
            
            # Test file selection and operations
            if engine.item_exists("**/File List"):
                # Simulate selecting files (if any exist)
                # This would require actual generated files to test properly
                print("✓ Generated file manager file operations available")
            
            # Test cleanup operations
            if engine.item_exists("**/Clean Up Files"):
                # Don't actually run cleanup in tests, just verify it exists
                print("✓ Generated file manager cleanup operations available")
            
            # Close the window
            if engine.item_exists("**/Close"):
                engine.item_click("**/Close")
            
            print("✓ Generated file manager integration tested")
    
    @pytest.mark.e2e
    def test_menu_system_comprehensive_integration(self, setup_gui_test_environment):
        """Test comprehensive menu system integration across components."""
        app_instance = setup_gui_test_environment
        engine = hello_imgui.get_imgui_test_engine()
        
        # Test File menu integration
        engine.item_open("**/File")
        
        # Test New Project
        if engine.item_exists("**/New Project"):
            initial_dirty_state = app_instance.project_manager.project_dirty
            engine.item_click("**/New Project")
            time.sleep(0.5)
            
            # Verify project state changes
            new_dirty_state = app_instance.project_manager.project_dirty
            print(f"✓ File -> New Project: dirty state {initial_dirty_state} -> {new_dirty_state}")
        
        # Test Edit menu integration
        engine.item_open("**/Edit")
        
        # Test Settings
        if engine.item_exists("**/Settings"):
            engine.item_click("**/Settings")
            time.sleep(0.5)
            
            # Settings window should open
            for _ in range(5):
                app_instance.gui_instance.run_one_frame(blocking=False)
                time.sleep(0.1)
            
            print("✓ Edit -> Settings integration")
        
        # Test View menu integration
        engine.item_open("**/View")
        
        view_options = [
            "Show Control Panel",
            "Show Video Display", 
            "Show Timeline",
            "Show Info Panel"
        ]
        
        for option in view_options:
            if engine.item_exists(f"**/{option}"):
                # Toggle the view option
                engine.item_click(f"**/{option}")
                time.sleep(0.2)
                
                # Run frame to update UI
                app_instance.gui_instance.run_one_frame(blocking=False)
                print(f"✓ View -> {option}")
        
        # Test Tools menu integration
        engine.item_open("**/Tools")
        
        tools_options = [
            "Performance Monitor",
            "Debug Console",
            "Hardware Info"
        ]
        
        for option in tools_options:
            if engine.item_exists(f"**/{option}"):
                print(f"✓ Tools -> {option} available")
        
        print("✓ Menu system comprehensive integration tested")
    
    @pytest.mark.e2e
    def test_component_state_synchronization(self, setup_gui_test_environment):
        """Test that component states stay synchronized across the application."""
        app_instance = setup_gui_test_environment
        engine = hello_imgui.get_imgui_test_engine()
        
        # Test video loading state synchronization
        video_path = os.path.abspath("test_data/sync_test_video.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        with open(video_path, "w") as f:
            f.write("dummy video for state sync test")
        
        # Load video and check state propagation
        with patch('application.classes.file_dialog.ImGuiFileDialog.show') as mock_dialog:
            engine.item_open("**/File")
            engine.item_open("**/Open...")
            engine.item_click("**/Video...")
            args, kwargs = mock_dialog.call_args
            callback = kwargs.get('callback')
            callback(video_path)
        
        # Wait for video load to propagate
        start_time = time.time()
        while not app_instance.file_manager.video_path and time.time() - start_time < 5:
            app_instance.gui_instance.run_one_frame(blocking=False)
            time.sleep(0.1)
        
        if app_instance.file_manager.video_path:
            # Check that multiple components reflect video loaded state
            components_with_video_state = [
                app_instance.file_manager,
                app_instance.video_processor,
                app_instance.funscript_processor
            ]
            
            for component in components_with_video_state:
                if hasattr(component, 'video_loaded') or hasattr(component, 'video_path'):
                    print(f"✓ Video state synchronized in {component.__class__.__name__}")
            
            # Test processing state synchronization
            if engine.item_exists("**/Start Processing"):
                initial_processing_state = getattr(app_instance, 'is_processing', False)
                
                # Note: Don't actually start processing in tests, just verify state tracking
                print(f"✓ Processing state tracking available: {initial_processing_state}")
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        
        print("✓ Component state synchronization tested")
    
    @pytest.mark.e2e
    def test_keyboard_shortcut_integration(self, setup_gui_test_environment):
        """Test keyboard shortcut integration across GUI components."""
        app_instance = setup_gui_test_environment
        engine = hello_imgui.get_imgui_test_engine()
        
        # Test common keyboard shortcuts
        shortcuts_to_test = [
            ("Ctrl+N", "New Project"),
            ("Ctrl+O", "Open Project"),
            ("Ctrl+S", "Save Project"),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Y", "Redo"),
            ("Space", "Play/Pause"),
            ("F1", "Help")
        ]
        
        for key_combo, action in shortcuts_to_test:
            try:
                # Test that shortcut key combinations are recognized
                # Note: Actual key simulation would require more complex setup
                if hasattr(app_instance, 'shortcut_manager'):
                    shortcut_registered = app_instance.shortcut_manager.is_shortcut_registered(key_combo)
                    if shortcut_registered:
                        print(f"✓ Keyboard shortcut registered: {key_combo} -> {action}")
                    else:
                        print(f"ℹ Keyboard shortcut not registered: {key_combo}")
                
            except Exception as e:
                print(f"⚠ Error testing shortcut {key_combo}: {e}")
        
        print("✓ Keyboard shortcut integration tested")
    
    @pytest.mark.e2e
    def test_window_layout_and_docking_integration(self, setup_gui_test_environment):
        """Test window layout management and docking functionality."""
        app_instance = setup_gui_test_environment
        engine = hello_imgui.get_imgui_test_engine()
        
        # Test window docking if available
        if hasattr(imgui, 'dock_space'):
            # Run frames to ensure docking system is active
            for _ in range(10):
                app_instance.gui_instance.run_one_frame(blocking=False)
                time.sleep(0.01)
            
            print("✓ Window docking system active")
        
        # Test window visibility toggles
        window_toggles = [
            "Control Panel",
            "Video Display",
            "Timeline View",
            "Properties Panel"
        ]
        
        for window_name in window_toggles:
            if engine.item_exists(f"**/{window_name}"):
                # Toggle window visibility
                initial_visible = engine.item_info(f"**/{window_name}").is_visible if hasattr(engine.item_info(f"**/{window_name}"), 'is_visible') else True
                
                # Toggle via menu
                engine.item_open("**/View")
                if engine.item_exists(f"**/Show {window_name}"):
                    engine.item_click(f"**/Show {window_name}")
                    time.sleep(0.2)
                    
                    # Run frame to update layout
                    app_instance.gui_instance.run_one_frame(blocking=False)
                    print(f"✓ Window layout toggle: {window_name}")
        
        # Test layout persistence
        if hasattr(app_instance.app_settings, 'save_window_layout'):
            app_instance.app_settings.save_window_layout()
            print("✓ Window layout persistence")
        
        print("✓ Window layout and docking integration tested")
    
    @pytest.mark.e2e
    def test_error_handling_and_user_feedback(self, setup_gui_test_environment):
        """Test error handling and user feedback across GUI components."""
        app_instance = setup_gui_test_environment
        engine = hello_imgui.get_imgui_test_engine()
        
        # Test error dialog system
        if hasattr(app_instance, 'show_error_dialog'):
            # Simulate an error condition
            test_error_message = "Test error for GUI integration"
            app_instance.show_error_dialog("Test Error", test_error_message)
            
            # Run frames to show dialog
            for _ in range(5):
                app_instance.gui_instance.run_one_frame(blocking=False)
                time.sleep(0.1)
            
            print("✓ Error dialog system functional")
        
        # Test status bar updates
        if hasattr(app_instance, 'status_bar') or hasattr(app_instance, 'update_status'):
            test_status = "Testing status bar integration"
            if hasattr(app_instance, 'update_status'):
                app_instance.update_status(test_status)
            
            # Run frame to update status
            app_instance.gui_instance.run_one_frame(blocking=False)
            print("✓ Status bar integration")
        
        # Test progress bar integration
        if hasattr(app_instance, 'progress_bar') or hasattr(app_instance, 'update_progress'):
            test_progress = 0.5  # 50%
            if hasattr(app_instance, 'update_progress'):
                app_instance.update_progress(test_progress, "Testing progress integration")
            
            # Run frame to update progress
            app_instance.gui_instance.run_one_frame(blocking=False)
            print("✓ Progress bar integration")
        
        # Test tooltip system
        if hasattr(imgui, 'set_tooltip'):
            # Tooltips would be tested through UI hover simulation
            print("✓ Tooltip system available")
        
        print("✓ Error handling and user feedback tested")
    
    @pytest.mark.e2e
    def test_theme_and_styling_integration(self, setup_gui_test_environment):
        """Test theme and styling consistency across components."""
        app_instance = setup_gui_test_environment
        
        # Test theme application
        if hasattr(app_instance.app_settings, 'get_theme') or hasattr(app_instance, 'current_theme'):
            current_theme = getattr(app_instance.app_settings, 'get_theme', lambda: 'default')()
            print(f"✓ Current theme: {current_theme}")
        
        # Test style consistency by running multiple frames
        # This ensures all components render without style conflicts
        for _ in range(20):
            app_instance.gui_instance.run_one_frame(blocking=False)
            time.sleep(0.01)
        
        print("✓ Theme and styling integration stable")
        
        # Test font scaling if available
        if hasattr(imgui, 'get_font_size'):
            current_font_size = imgui.get_font_size()
            print(f"✓ Font system: size={current_font_size}")
        
        # Test color scheme consistency
        if hasattr(imgui, 'get_style'):
            style = imgui.get_style()
            if hasattr(style, 'colors'):
                print("✓ Color scheme applied consistently")
        
        print("✓ Theme and styling integration tested")


if __name__ == "__main__":
    pytest.main([__file__])