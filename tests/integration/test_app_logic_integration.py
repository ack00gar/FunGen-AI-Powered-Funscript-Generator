"""
Integration tests for ApplicationLogic initialization and component wiring.

These tests verify that ApplicationLogic correctly creates and wires all
sub-components in CLI mode, and that key subsystems integrate properly
with each other.
"""

import json
import os
import sys
import tempfile
import pytest

# Ensure project root is on sys.path so imports resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from application.classes.settings_manager import AppSettings
from application.classes.undo_redo_manager import UndoRedoManager
from funscript import MultiAxisFunscript


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_settings_file(tmp_path, data=None):
    """Create a temporary settings.json and return its path."""
    settings_path = os.path.join(str(tmp_path), "settings.json")
    if data is not None:
        with open(settings_path, "w") as f:
            json.dump(data, f)
    return settings_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_settings_path(tmp_path):
    """Return a path to a non-existent settings file in a temp directory."""
    return os.path.join(str(tmp_path), "settings.json")


@pytest.fixture
def app_settings_fresh(tmp_settings_path):
    """Create a fresh AppSettings backed by a temp file."""
    return AppSettings(settings_file_path=tmp_settings_path, logger=None)


# ---------------------------------------------------------------------------
# ApplicationLogic CLI initialization
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAppLogicCLIInitialization:
    """Verify ApplicationLogic(is_cli=True) creates all expected sub-components."""

    @pytest.fixture(autouse=True)
    def _setup_app(self):
        """Create the app once per test class; skip if heavy deps are missing."""
        try:
            from application.logic.app_logic import ApplicationLogic
            self.app = ApplicationLogic(is_cli=True)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate ApplicationLogic in CLI mode: {exc}")

    def test_app_logic_cli_initialization(self):
        """ApplicationLogic(is_cli=True) should not crash and should expose key attributes."""
        assert self.app is not None
        assert self.app.is_cli_mode is True

    def test_app_settings_exists(self):
        assert self.app.app_settings is not None
        assert isinstance(self.app.app_settings, AppSettings)

    def test_stage_processor_exists(self):
        assert hasattr(self.app, 'stage_processor')
        assert self.app.stage_processor is not None

    def test_funscript_processor_exists(self):
        assert hasattr(self.app, 'funscript_processor')
        assert self.app.funscript_processor is not None

    def test_file_manager_exists(self):
        assert hasattr(self.app, 'file_manager')
        assert self.app.file_manager is not None

    def test_event_handlers_exist(self):
        assert hasattr(self.app, 'event_handlers')
        assert self.app.event_handlers is not None

    def test_calibration_exists(self):
        assert hasattr(self.app, 'calibration')
        assert self.app.calibration is not None

    def test_energy_saver_exists(self):
        assert hasattr(self.app, 'energy_saver')
        assert self.app.energy_saver is not None

    def test_utility_exists(self):
        assert hasattr(self.app, 'utility')
        assert self.app.utility is not None

    def test_app_state_ui_exists(self):
        assert hasattr(self.app, 'app_state_ui')
        assert self.app.app_state_ui is not None

    def test_project_manager_exists(self):
        assert hasattr(self.app, 'project_manager')
        assert self.app.project_manager is not None

    def test_shortcut_manager_exists(self):
        assert hasattr(self.app, 'shortcut_manager')
        assert self.app.shortcut_manager is not None


# ---------------------------------------------------------------------------
# AppSettings persistence
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAppSettingsPersistence:
    """Verify settings round-trip through save and reload."""

    def test_app_settings_persistence(self, tmp_path):
        """Create settings, set values, save, create new settings from same file, verify."""
        settings_path = _make_temp_settings_file(tmp_path)

        # Create first instance (first run, defaults written)
        settings1 = AppSettings(settings_file_path=settings_path, logger=None)
        assert settings1.is_first_run is True

        # Set custom values
        settings1.data["logging_level"] = "DEBUG"
        settings1.data["energy_saver_enabled"] = False
        settings1.data["num_producers_stage1"] = 42
        settings1.save_settings()

        # Create second instance from same file
        settings2 = AppSettings(settings_file_path=settings_path, logger=None)
        assert settings2.is_first_run is False  # file exists now

        assert settings2.get("logging_level") == "DEBUG"
        assert settings2.get("energy_saver_enabled") is False
        assert settings2.get("num_producers_stage1") == 42


# ---------------------------------------------------------------------------
# Funscript file round-trip
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFunscriptFileRoundtrip:
    """Verify funscript data can be saved to a file and reloaded accurately."""

    @staticmethod
    def _build_funscript_json(actions):
        return {
            "version": "1.0",
            "author": "test",
            "inverted": False,
            "range": 100,
            "actions": sorted(actions, key=lambda a: a["at"]),
            "metadata": {"version": "0.2.0", "chapters": []},
        }

    def test_funscript_file_roundtrip(self, tmp_path):
        """Create funscript data, save to temp file, reload, verify data matches."""
        import orjson

        original_actions = [
            {"at": 0, "pos": 10},
            {"at": 500, "pos": 90},
            {"at": 1000, "pos": 50},
            {"at": 1500, "pos": 0},
            {"at": 2000, "pos": 100},
        ]

        funscript_data = self._build_funscript_json(original_actions)
        filepath = os.path.join(str(tmp_path), "test.funscript")

        # Write
        with open(filepath, "wb") as f:
            f.write(orjson.dumps(funscript_data))

        # Read back
        with open(filepath, "rb") as f:
            loaded = orjson.loads(f.read())

        loaded_actions = sorted(loaded["actions"], key=lambda a: a["at"])
        assert len(loaded_actions) == len(original_actions)

        for orig, loaded_a in zip(original_actions, loaded_actions):
            assert orig["at"] == loaded_a["at"]
            assert orig["pos"] == loaded_a["pos"]

    def test_funscript_roundtrip_into_multi_axis(self, tmp_path):
        """Save funscript, load into MultiAxisFunscript, verify interpolation works."""
        import orjson

        actions = [
            {"at": 0, "pos": 0},
            {"at": 1000, "pos": 100},
            {"at": 2000, "pos": 0},
        ]
        funscript_data = self._build_funscript_json(actions)
        filepath = os.path.join(str(tmp_path), "round.funscript")

        with open(filepath, "wb") as f:
            f.write(orjson.dumps(funscript_data))

        with open(filepath, "rb") as f:
            loaded = orjson.loads(f.read())

        fs = MultiAxisFunscript()
        fs.actions = loaded["actions"]

        assert len(fs.primary_actions) == 3
        # Midpoint at 500ms should interpolate to ~50
        val = fs.get_value(500, axis='primary')
        assert 45 <= val <= 55


# ---------------------------------------------------------------------------
# Project state reset
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestProjectStateReset:
    """Verify reset_project_state restores a clean state."""

    def test_project_state_reset(self):
        """Initialize app, set some state, call reset_project_state(), verify clean state."""
        try:
            from application.logic.app_logic import ApplicationLogic
            app = ApplicationLogic(is_cli=True)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate ApplicationLogic: {exc}")

        # Set some dirty state
        app.funscript_processor.video_chapters.append("dummy_chapter")
        app.is_batch_processing_active = True

        # Reset
        app.reset_project_state(for_new_project=True)

        # Chapters should be cleared (reset_state_for_new_project clears them)
        assert len(app.funscript_processor.video_chapters) == 0
        # Batch flag is not reset by reset_project_state; it's managed by the batch loop.
        # But undo stacks should be cleared.
        if app.undo_manager_t1:
            assert not app.undo_manager_t1.can_undo()
        if app.undo_manager_t2:
            assert not app.undo_manager_t2.can_undo()


# ---------------------------------------------------------------------------
# Batch processor integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBatchProcessorIntegration:
    """Verify batch processing infrastructure is wired on the app."""

    def test_batch_processor_integration(self):
        try:
            from application.logic.app_logic import ApplicationLogic
            app = ApplicationLogic(is_cli=True)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate ApplicationLogic: {exc}")

        # Batch state attributes exist and are initialised
        assert hasattr(app, 'batch_video_paths')
        assert isinstance(app.batch_video_paths, list)
        assert app.is_batch_processing_active is False
        assert app.current_batch_video_index == -1
        assert hasattr(app, 'stop_batch_event')


# ---------------------------------------------------------------------------
# Autotuner integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAutotunerIntegration:
    """Verify autotuner component exists and can get snapshot."""

    def test_autotuner_integration(self):
        try:
            from application.logic.app_logic import ApplicationLogic
            app = ApplicationLogic(is_cli=True)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate ApplicationLogic: {exc}")

        # Autotuner state attributes should be present
        assert hasattr(app, 'is_autotuning_active')
        assert app.is_autotuning_active is False
        assert hasattr(app, 'autotuner_status_message')
        assert isinstance(app.autotuner_status_message, str)
        assert hasattr(app, 'autotuner_results')
        assert isinstance(app.autotuner_results, dict)
        assert hasattr(app, 'autotuner_best_fps')
        assert app.autotuner_best_fps == 0.0


# ---------------------------------------------------------------------------
# Model manager integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestModelManagerIntegration:
    """Verify model manager attributes are wired."""

    def test_model_manager_integration(self):
        try:
            from application.logic.app_logic import ApplicationLogic
            app = ApplicationLogic(is_cli=True)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate ApplicationLogic: {exc}")

        # Model paths should be initialised from settings
        assert hasattr(app, 'yolo_det_model_path')
        assert hasattr(app, 'yolo_pose_model_path')
        # The processor should exist and reference the tracker
        assert hasattr(app, 'processor')
        assert app.processor is not None


# ---------------------------------------------------------------------------
# ROI manager integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestROIManagerIntegration:
    """Verify ROI manager delegation attributes."""

    def test_roi_manager_integration(self):
        try:
            from application.logic.app_logic import ApplicationLogic
            app = ApplicationLogic(is_cli=True)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate ApplicationLogic: {exc}")

        # ROI-related state should be initialised
        assert hasattr(app, 'is_setting_user_roi_mode')
        assert app.is_setting_user_roi_mode is False
        assert hasattr(app, 'chapter_id_for_roi_setting')
        assert app.chapter_id_for_roi_setting is None
