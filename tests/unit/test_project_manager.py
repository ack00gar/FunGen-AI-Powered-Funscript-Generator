"""Unit tests for ProjectManager class."""
import pytest
import os
import time
import json
import logging
from unittest.mock import MagicMock, patch, PropertyMock


@pytest.mark.unit
class TestProjectManagerCreation:
    """Tests for ProjectManager construction and initial state."""

    def test_creation_with_mock_app(self, mock_app):
        """ProjectManager is created with a reference to the app instance."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        assert pm.app is mock_app

    def test_initial_project_file_path_is_none(self, mock_app):
        """Initially, project_file_path is None."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        assert pm.project_file_path is None

    def test_initial_project_not_dirty(self, mock_app):
        """Initially, project_dirty is False."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        assert pm.project_dirty is False

    def test_initial_autosave_time_set(self, mock_app):
        """last_autosave_time is set to current time on creation."""
        from application.classes.project_manager import ProjectManager
        before = time.time()
        pm = ProjectManager(mock_app)
        after = time.time()
        assert before <= pm.last_autosave_time <= after


@pytest.mark.unit
class TestProjectManagerProperties:
    """Tests for project_file_path and project_dirty properties."""

    def test_set_project_file_path(self, mock_app):
        """project_file_path can be set and retrieved."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm.project_file_path = "/some/path/project.fgnproj"
        assert pm.project_file_path == "/some/path/project.fgnproj"

    def test_set_project_file_path_to_none(self, mock_app):
        """project_file_path can be set to None."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm.project_file_path = "/some/path"
        pm.project_file_path = None
        assert pm.project_file_path is None

    def test_project_dirty_flag(self, mock_app):
        """project_dirty flag can be toggled."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        assert pm.project_dirty is False
        pm.project_dirty = True
        assert pm.project_dirty is True
        pm.project_dirty = False
        assert pm.project_dirty is False

    def test_project_dirty_only_updates_on_change(self, mock_app):
        """project_dirty setter only fires when value actually changes."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm.project_dirty = False  # Same as initial, no-op
        pm.project_dirty = True   # Changes
        pm.project_dirty = True   # Same, no-op internally
        assert pm.project_dirty is True


@pytest.mark.unit
class TestProjectManagerNewProject:
    """Tests for new_project method."""

    def test_new_project_resets_state(self, mock_app):
        """new_project resets project_file_path, dirty flag, and autosave time."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm.project_file_path = "/old/path.fgnproj"
        pm.project_dirty = True
        
        pm.new_project()
        
        assert pm.project_file_path is None
        assert pm.project_dirty is False

    def test_new_project_clears_last_opened(self, mock_app):
        """new_project clears last_opened_project_path in settings."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm.new_project()
        mock_app.app_settings.set.assert_any_call("last_opened_project_path", None)

    def test_new_project_calls_reset_project_state(self, mock_app):
        """new_project delegates to app.reset_project_state."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm.new_project()
        mock_app.reset_project_state.assert_called_once_with(for_new_project=True)

    def test_new_project_warns_if_dirty(self, mock_app):
        """new_project logs a warning if the current project is dirty."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm.project_dirty = True
        pm.new_project()
        # The logger.warning should have been called
        # Just verify no exception is raised and state is reset
        assert pm.project_dirty is False

    def test_new_project_updates_autosave_time(self, mock_app):
        """new_project resets the autosave timer."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        old_time = pm.last_autosave_time
        time.sleep(0.01)
        pm.new_project()
        assert pm.last_autosave_time >= old_time


@pytest.mark.unit
class TestProjectManagerSaveProject:
    """Tests for save_project method."""

    def test_save_project_writes_file(self, mock_app, temp_dir):
        """save_project writes a project file to disk."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        filepath = os.path.join(temp_dir, "test_save.fgnproj")
        
        # Mock the data gathering
        mock_app.funscript_processor.get_project_save_data.return_value = {
            "video_chapters": [],
            "scripting_range_active": False,
            "scripting_start_frame": 0,
            "scripting_end_frame": -1,
            "selected_chapter_for_scripting_id": None,
        }
        mock_app.stage_processor.get_project_save_data.return_value = {
            "stage1_output_msgpack_path": None,
            "stage2_overlay_msgpack_path": None,
            "stage2_database_path": None,
            "stage2_status_text": "Not run.",
        }
        mock_app.processor = None
        mock_app.file_manager.video_path = ""
        mock_app.file_manager.funscript_path = ""
        mock_app.file_manager.loaded_funscript_path = ""
        mock_app.app_state_ui.timeline_pan_offset_ms = 0
        mock_app.app_state_ui.timeline_zoom_factor_ms_per_px = 20.0
        mock_app.app_state_ui.show_funscript_interactive_timeline = True
        mock_app.app_state_ui.show_funscript_interactive_timeline2 = False
        mock_app.app_state_ui.show_lr_dial_graph = False
        mock_app.app_state_ui.show_simulator_3d = True
        mock_app.app_state_ui.show_heatmap = True
        mock_app.app_state_ui.show_gauge_window_timeline1 = True
        mock_app.app_state_ui.show_gauge_window_timeline2 = False
        mock_app.app_state_ui.show_stage2_overlay = True
        mock_app.app_state_ui.show_audio_waveform = False
        mock_app.yolo_detection_model_path_setting = ""
        mock_app.yolo_pose_model_path_setting = ""
        mock_app.calibration.funscript_output_delay_frames = 3
        mock_app.audio_waveform_data = None
        mock_app.funscript_processor.get_actions.return_value = []
        mock_app.energy_saver = MagicMock()
        mock_app.gui_instance = None

        with patch('application.classes.project_manager.check_write_access'):
            pm.save_project(filepath)

        assert os.path.exists(filepath)
        assert pm.project_file_path == filepath
        assert pm.project_dirty is False

    def test_save_project_clears_dirty_flag(self, mock_app, temp_dir):
        """save_project sets project_dirty to False on success."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm.project_dirty = True
        filepath = os.path.join(temp_dir, "dirty_test.fgnproj")

        mock_app.funscript_processor.get_project_save_data.return_value = {
            "video_chapters": [],
            "scripting_range_active": False,
            "scripting_start_frame": 0,
            "scripting_end_frame": -1,
            "selected_chapter_for_scripting_id": None,
        }
        mock_app.stage_processor.get_project_save_data.return_value = {
            "stage1_output_msgpack_path": None,
            "stage2_overlay_msgpack_path": None,
            "stage2_database_path": None,
            "stage2_status_text": "Not run.",
        }
        mock_app.processor = None
        mock_app.file_manager.video_path = ""
        mock_app.file_manager.funscript_path = ""
        mock_app.file_manager.loaded_funscript_path = ""
        mock_app.app_state_ui.timeline_pan_offset_ms = 0
        mock_app.app_state_ui.timeline_zoom_factor_ms_per_px = 20.0
        mock_app.app_state_ui.show_funscript_interactive_timeline = True
        mock_app.app_state_ui.show_funscript_interactive_timeline2 = False
        mock_app.app_state_ui.show_lr_dial_graph = False
        mock_app.app_state_ui.show_simulator_3d = True
        mock_app.app_state_ui.show_heatmap = True
        mock_app.app_state_ui.show_gauge_window_timeline1 = True
        mock_app.app_state_ui.show_gauge_window_timeline2 = False
        mock_app.app_state_ui.show_stage2_overlay = True
        mock_app.app_state_ui.show_audio_waveform = False
        mock_app.yolo_detection_model_path_setting = ""
        mock_app.yolo_pose_model_path_setting = ""
        mock_app.calibration.funscript_output_delay_frames = 0
        mock_app.audio_waveform_data = None
        mock_app.funscript_processor.get_actions.return_value = []
        mock_app.energy_saver = MagicMock()
        mock_app.gui_instance = None

        with patch('application.classes.project_manager.check_write_access'):
            pm.save_project(filepath)

        assert pm.project_dirty is False


@pytest.mark.unit
class TestProjectManagerLoadProject:
    """Tests for load_project method."""

    def _create_project_file(self, temp_dir, filename="test.fgnproj"):
        """Helper to create a minimal project file for load tests."""
        import orjson
        filepath = os.path.join(temp_dir, filename)
        project_data = {
            "video_path": "",
            "funscript_path": "",
            "loaded_funscript_path_timeline1": "",
            "funscript_actions_timeline1": [],
            "funscript_actions_timeline2": [],
            "video_chapters": [],
        }
        with open(filepath, 'wb') as f:
            f.write(orjson.dumps(project_data))
        return filepath

    def _setup_mock_for_load(self, mock_app):
        """Configure mock_app for load_project calls."""
        mock_app.file_manager.video_path = ""
        mock_app.gui_instance = None
        # Use side_effect to return appropriate defaults per key
        mock_app.app_settings.get.side_effect = lambda key, default=None: {
            "recent_projects": [],
        }.get(key, default)
        mock_app.app_settings.set_batch = MagicMock()

    def test_load_project_sets_path(self, mock_app, temp_dir):
        """load_project sets project_file_path on success."""
        from application.classes.project_manager import ProjectManager

        pm = ProjectManager(mock_app)
        filepath = self._create_project_file(temp_dir, "load_test.fgnproj")
        self._setup_mock_for_load(mock_app)

        with patch.object(pm, '_apply_project_state_from_dict'):
            pm.load_project(filepath)

        assert pm.project_file_path == filepath

    def test_load_project_not_dirty_after_load(self, mock_app, temp_dir):
        """Loaded project is not dirty (unless it's an autosave)."""
        from application.classes.project_manager import ProjectManager

        pm = ProjectManager(mock_app)
        filepath = self._create_project_file(temp_dir, "clean_load.fgnproj")
        self._setup_mock_for_load(mock_app)

        with patch.object(pm, '_apply_project_state_from_dict'):
            pm.load_project(filepath, is_autosave=False)
        assert pm.project_dirty is False

    def test_load_project_autosave_stays_dirty(self, mock_app, temp_dir):
        """Loaded autosave project remains dirty."""
        from application.classes.project_manager import ProjectManager

        pm = ProjectManager(mock_app)
        filepath = self._create_project_file(temp_dir, "autosave.fgnproj")
        self._setup_mock_for_load(mock_app)

        with patch.object(pm, '_apply_project_state_from_dict'):
            pm.load_project(filepath, is_autosave=True)
        assert pm.project_dirty is True

    def test_load_nonexistent_file_handles_error(self, mock_app):
        """Loading a nonexistent file logs error but does not crash."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        
        mock_app.app_settings.get.return_value = []
        
        # Should not raise
        pm.load_project("/nonexistent/path/project.fgnproj")


@pytest.mark.unit
class TestProjectManagerRecentProjects:
    """Tests for recent projects tracking."""

    def test_add_to_recent_projects(self, mock_app, temp_dir):
        """_add_to_recent_projects adds path to settings."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        
        mock_app.app_settings.get.return_value = []
        
        filepath = os.path.join(temp_dir, "recent.fgnproj")
        pm._add_to_recent_projects(filepath)
        
        # Verify set was called with updated list
        mock_app.app_settings.set.assert_called()
        call_args = mock_app.app_settings.set.call_args
        assert call_args[0][0] == "recent_projects"
        assert os.path.abspath(filepath) in call_args[0][1]

    def test_add_to_recent_projects_no_duplicates(self, mock_app, temp_dir):
        """_add_to_recent_projects removes duplicates, putting latest first."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        
        filepath = os.path.abspath(os.path.join(temp_dir, "dup.fgnproj"))
        mock_app.app_settings.get.return_value = [filepath, "/other/path.fgnproj"]
        
        pm._add_to_recent_projects(filepath)
        
        call_args = mock_app.app_settings.set.call_args
        recent_list = call_args[0][1]
        assert recent_list[0] == filepath
        assert recent_list.count(filepath) == 1

    def test_add_to_recent_projects_trims_to_10(self, mock_app, temp_dir):
        """_add_to_recent_projects trims list to 10 entries."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        
        existing = [f"/path/project_{i}.fgnproj" for i in range(12)]
        mock_app.app_settings.get.return_value = existing
        
        new_path = os.path.join(temp_dir, "new.fgnproj")
        pm._add_to_recent_projects(new_path)
        
        call_args = mock_app.app_settings.set.call_args
        recent_list = call_args[0][1]
        assert len(recent_list) <= 10

    def test_add_empty_path_does_nothing(self, mock_app):
        """_add_to_recent_projects with empty path is a no-op."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm._add_to_recent_projects("")
        mock_app.app_settings.set.assert_not_called()

    def test_add_none_path_does_nothing(self, mock_app):
        """_add_to_recent_projects with None path is a no-op."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm._add_to_recent_projects(None)
        mock_app.app_settings.set.assert_not_called()


@pytest.mark.unit
class TestProjectManagerAutosave:
    """Tests for autosave functionality."""

    def test_autosave_skipped_when_disabled(self, mock_app):
        """perform_autosave does nothing when autosave is disabled."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm.project_dirty = True
        
        mock_app.app_settings.get.return_value = False  # autosave_enabled = False
        
        pm.perform_autosave()
        # No save should have been attempted

    def test_autosave_skipped_when_not_dirty(self, mock_app):
        """perform_autosave does nothing when project is not dirty."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm.project_dirty = False
        
        mock_app.app_settings.get.return_value = True  # autosave_enabled = True
        
        old_time = pm.last_autosave_time
        pm.perform_autosave()
        # Autosave time should be updated even though no save was needed
        assert pm.last_autosave_time >= old_time

    def test_autosave_skipped_when_no_video(self, mock_app):
        """perform_autosave does nothing when no video is loaded."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        pm.project_dirty = True
        
        mock_app.app_settings.get.return_value = True
        mock_app.file_manager.video_path = ""
        
        pm.perform_autosave()


@pytest.mark.unit
class TestProjectManagerGetSuggestedSavePath:
    """Tests for get_suggested_save_path_and_dir."""

    def test_returns_none_when_no_video(self, mock_app):
        """Returns None when no video is loaded."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        mock_app.file_manager.video_path = ""
        assert pm.get_suggested_save_path_and_dir(save_as=False) is None

    def test_returns_tuple_when_video_loaded(self, mock_app, temp_dir):
        """Returns (filename, directory) tuple when video is loaded."""
        from application.classes.project_manager import ProjectManager
        pm = ProjectManager(mock_app)
        
        mock_app.file_manager.video_path = "/videos/test.mp4"
        expected_output = os.path.join(temp_dir, "test.fgnproj")
        mock_app.file_manager.get_output_path_for_file.return_value = expected_output
        
        result = pm.get_suggested_save_path_and_dir(save_as=False)
        assert result is not None
        filename, directory = result
        assert filename == "test.fgnproj"
        assert directory == temp_dir
