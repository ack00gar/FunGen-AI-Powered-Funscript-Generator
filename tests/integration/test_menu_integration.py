"""
Integration tests for menu functionality to catch method name mismatches.
"""
import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from application.classes.menu import MainMenu
from application.logic.app_file_manager import AppFileManager


@pytest.fixture
def mock_app():
    """Create a mock app instance with required attributes."""
    app = Mock()
    app.file_manager = Mock(spec=AppFileManager)
    app.gui_instance = Mock()
    app.gui_instance.file_dialog = Mock()
    app.project_manager = Mock()
    app.project_manager.open_project_dialog = Mock()
    app.shutdown_app = Mock()
    app.reset_project_state = Mock()
    return app


@pytest.fixture
def menu_instance(mock_app):
    """Create a MainMenu instance with a mock app."""
    return MainMenu(mock_app)


class TestMenuIntegration:
    """Test menu functionality to catch method name mismatches."""
    
    def test_menu_export_methods_exist(self, mock_app):
        """Test that all export methods exist on AppFileManager."""
        fm = AppFileManager(mock_app)
        
        # Test that save_funscript_from_timeline exists with correct signature
        assert hasattr(fm, 'save_funscript_from_timeline')
        assert callable(fm.save_funscript_from_timeline)
        
        # Test that import methods exist
        assert hasattr(fm, 'import_funscript_to_timeline')
        assert callable(fm.import_funscript_to_timeline)
        
        assert hasattr(fm, 'import_stage2_overlay_data')
        assert callable(fm.import_stage2_overlay_data)

    def test_export_funscript_from_timeline_dialog_trigger(self, mock_app):
        """Test that export menu items trigger file dialogs correctly."""
        fm = AppFileManager(mock_app)
        
        # Test Timeline 1 export
        fm.import_funscript_to_timeline(1)
        mock_app.gui_instance.file_dialog.show.assert_called()
        call_args = mock_app.gui_instance.file_dialog.show.call_args
        assert call_args[1]['title'] == "Import Funscript to Timeline 1"
        assert call_args[1]['is_save'] is False
        
        # Reset mock
        mock_app.gui_instance.file_dialog.show.reset_mock()
        
        # Test Timeline 2 export
        fm.import_funscript_to_timeline(2)
        mock_app.gui_instance.file_dialog.show.assert_called()
        call_args = mock_app.gui_instance.file_dialog.show.call_args
        assert call_args[1]['title'] == "Import Funscript to Timeline 2"

    def test_import_stage2_overlay_data_dialog_trigger(self, mock_app):
        """Test that import stage2 overlay data triggers file dialog correctly."""
        fm = AppFileManager(mock_app)
        
        fm.import_stage2_overlay_data()
        mock_app.gui_instance.file_dialog.show.assert_called()
        call_args = mock_app.gui_instance.file_dialog.show.call_args
        assert call_args[1]['title'] == "Import Stage 2 Overlay Data"
        assert call_args[1]['extension_filter'] == "MessagePack Files (*.msgpack),*.msgpack"
        assert call_args[1]['is_save'] is False

    def test_save_funscript_from_timeline_callback(self, mock_app):
        """Test that save funscript callback works correctly."""
        # Mock the funscript processor and required methods
        mock_funscript_processor = Mock()
        mock_funscript_processor.get_actions.return_value = [{"at": 1000, "pos": 50}]
        mock_funscript_processor.video_chapters = None
        mock_app.funscript_processor = mock_funscript_processor
        mock_app.energy_saver = Mock()
        mock_app.energy_saver.reset_activity_timer = Mock()
        
        fm = AppFileManager(mock_app)
        
        # Mock the _save_funscript_file method to avoid file system operations
        fm._save_funscript_file = Mock()
        
        with tempfile.NamedTemporaryFile(suffix='.funscript', delete=False) as tf:
            test_path = tf.name
        
        try:
            # Test Timeline 1 save
            fm.save_funscript_from_timeline(test_path, 1)
            
            # Verify the method was called with correct parameters
            mock_funscript_processor.get_actions.assert_called_with('primary')
            fm._save_funscript_file.assert_called_once()
            
            # Verify file path was set for timeline 1
            assert fm.funscript_path == test_path
            assert fm.loaded_funscript_path == test_path
            
        finally:
            # Clean up
            if os.path.exists(test_path):
                os.unlink(test_path)

    @patch('application.classes.menu.imgui')
    def test_menu_render_file_menu_no_attribute_errors(self, mock_imgui, menu_instance, mock_app):
        """Test that menu rendering doesn't cause AttributeError."""
        # Setup mock imgui returns
        mock_imgui.begin_menu.return_value = True
        mock_imgui.menu_item.return_value = (True, None)  # Simulate menu item clicked
        mock_imgui.end_menu = Mock()
        
        # Mock app state
        mock_app_state = Mock()
        
        # Test that _render_file_menu can be called without AttributeError
        try:
            # This should not raise AttributeError
            menu_instance._render_file_menu(mock_app_state, mock_app.file_manager)
        except AttributeError as e:
            if "has no attribute" in str(e):
                pytest.fail(f"Menu rendering caused AttributeError: {e}")
        except Exception:
            # Other exceptions are ok for this test, we're only checking for AttributeError
            pass

    def test_all_menu_methods_have_required_attributes(self, mock_app):
        """Comprehensive test to ensure all referenced methods exist."""
        fm = AppFileManager(mock_app)
        
        # List of all methods that the menu expects to exist
        required_methods = [
            'import_funscript_to_timeline',
            'import_stage2_overlay_data', 
            'save_funscript_from_timeline',
            'open_video_dialog',
        ]
        
        for method_name in required_methods:
            assert hasattr(fm, method_name), f"AppFileManager missing method: {method_name}"
            assert callable(getattr(fm, method_name)), f"AppFileManager.{method_name} is not callable"

    def test_menu_dialog_callbacks_work(self, mock_app):
        """Test that file dialog callbacks execute without errors."""
        fm = AppFileManager(mock_app)
        
        # Mock required dependencies
        mock_app.funscript_processor = Mock()
        mock_app.energy_saver = Mock()
        mock_app.energy_saver.reset_activity_timer = Mock()
        
        # Test import callback doesn't crash
        try:
            fm.import_funscript_to_timeline(1)
            # Get the callback that was passed to file dialog
            call_args = mock_app.gui_instance.file_dialog.show.call_args
            callback = call_args[1]['callback']
            
            # Mock load_funscript_to_timeline to avoid file operations  
            fm.load_funscript_to_timeline = Mock()
            
            # Test callback execution
            callback("/fake/path.funscript")
            fm.load_funscript_to_timeline.assert_called_with("/fake/path.funscript", 1)
            
        except Exception as e:
            pytest.fail(f"Import callback failed: {e}")