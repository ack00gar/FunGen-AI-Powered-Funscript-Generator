"""
Functional tests for settings workflows.

These tests exercise the full AppSettings lifecycle: creation, modification,
persistence, default handling, first-run detection, and reset.
"""

import json
import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from application.classes.settings_manager import AppSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_settings(tmp_path, initial_data=None, filename="settings.json"):
    """Create an AppSettings instance backed by a temp file."""
    path = os.path.join(str(tmp_path), filename)
    if initial_data is not None:
        with open(path, "w") as f:
            json.dump(initial_data, f)
    return AppSettings(settings_file_path=path, logger=None)


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.functional
class TestSettingsFullLifecycle:
    """Create settings, modify, save, reload, verify."""

    def test_settings_full_lifecycle(self, tmp_path):
        # 1. Create (first run)
        settings = _create_settings(tmp_path)
        assert settings.is_first_run is True

        # 2. Modify
        settings.data["logging_level"] = "DEBUG"
        settings.data["energy_saver_enabled"] = False
        settings.data["num_producers_stage1"] = 8
        settings.data["custom_test_key"] = "hello_world"
        settings.save_settings()

        # 3. Reload from same file
        settings2 = AppSettings(
            settings_file_path=settings.settings_file, logger=None
        )
        assert settings2.is_first_run is False

        # 4. Verify persisted values
        assert settings2.get("logging_level") == "DEBUG"
        assert settings2.get("energy_saver_enabled") is False
        assert settings2.get("num_producers_stage1") == 8
        assert settings2.get("custom_test_key") == "hello_world"

    def test_set_method_persists_immediately(self, tmp_path):
        """settings.set() should write to disk immediately."""
        settings = _create_settings(tmp_path)
        settings.set("logging_level", "ERROR")

        # Re-read from disk
        with open(settings.settings_file, "r") as f:
            on_disk = json.load(f)

        assert on_disk["logging_level"] == "ERROR"

    def test_set_batch_persists_once(self, tmp_path):
        """set_batch() should update multiple keys and save once."""
        settings = _create_settings(tmp_path)
        settings.set_batch(
            logging_level="WARNING",
            energy_saver_fps=5,
            num_producers_stage1=16,
        )

        settings2 = AppSettings(
            settings_file_path=settings.settings_file, logger=None
        )
        assert settings2.get("logging_level") == "WARNING"
        assert settings2.get("energy_saver_fps") == 5
        assert settings2.get("num_producers_stage1") == 16


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

@pytest.mark.functional
class TestSettingsDefaultValues:
    """Verify all important defaults exist."""

    def test_settings_default_values(self, tmp_path):
        settings = _create_settings(tmp_path)
        defaults = settings.get_default_settings()

        # Check that a broad set of important defaults are defined
        important_keys = [
            "yolo_det_model_path",
            "yolo_pose_model_path",
            "output_folder_path",
            "logging_level",
            "ui_view_mode",
            "window_width",
            "window_height",
            "num_producers_stage1",
            "num_consumers_stage1",
            "hardware_acceleration_method",
            "autosave_enabled",
            "autosave_interval_seconds",
            "energy_saver_enabled",
            "energy_saver_threshold_seconds",
            "main_loop_normal_fps_target",
            "funscript_output_delay_frames",
            "tracking_axis_mode",
            "single_axis_output_target",
            "live_tracker_confidence_threshold",
            "enable_auto_post_processing",
            "funscript_editor_shortcuts",
            "last_opened_project_path",
            "recent_projects",
            "updater_check_on_startup",
            "generate_roll_file",
        ]

        for key in important_keys:
            assert key in defaults, f"Default missing for key '{key}'"
            # Also verify the get() method returns the default
            val = settings.get(key)
            assert val is not None or defaults[key] is None, (
                f"get('{key}') returned None but default is {defaults[key]}"
            )

    def test_defaults_have_correct_types(self, tmp_path):
        """Verify some key defaults have the expected types."""
        settings = _create_settings(tmp_path)

        assert isinstance(settings.get("logging_level"), str)
        assert isinstance(settings.get("autosave_enabled"), bool)
        assert isinstance(settings.get("autosave_interval_seconds"), (int, float))
        assert isinstance(settings.get("num_producers_stage1"), int)
        assert isinstance(settings.get("energy_saver_threshold_seconds"), (int, float))
        assert isinstance(settings.get("recent_projects"), list)
        assert isinstance(settings.get("funscript_editor_shortcuts"), dict)


# ---------------------------------------------------------------------------
# First run detection
# ---------------------------------------------------------------------------

@pytest.mark.functional
class TestSettingsFirstRunDetection:
    """First creation is first_run, after save not first_run."""

    def test_settings_first_run_detection(self, tmp_path):
        path = os.path.join(str(tmp_path), "settings.json")

        # First time: no file exists
        assert not os.path.exists(path)
        settings1 = AppSettings(settings_file_path=path, logger=None)
        assert settings1.is_first_run is True

        # After __init__ of first-run, the file should now exist on disk
        # (AppSettings.load_settings calls save_settings on first run)
        assert os.path.exists(path)

        # Second time: file exists
        settings2 = AppSettings(settings_file_path=path, logger=None)
        assert settings2.is_first_run is False

    def test_first_run_creates_settings_file(self, tmp_path):
        """On first run, the settings file should be created with defaults."""
        path = os.path.join(str(tmp_path), "new_settings.json")
        settings = AppSettings(settings_file_path=path, logger=None)

        assert os.path.exists(path)
        with open(path, "r") as f:
            on_disk = json.load(f)

        # Should contain at least the default keys
        defaults = settings.get_default_settings()
        for key in defaults:
            assert key in on_disk, f"First-run file missing default key '{key}'"


# ---------------------------------------------------------------------------
# Reset to defaults
# ---------------------------------------------------------------------------

@pytest.mark.functional
class TestSettingsResetToDefaults:
    """Modify settings, reset, verify defaults restored."""

    def test_settings_reset_to_defaults(self, tmp_path):
        settings = _create_settings(tmp_path)
        defaults = settings.get_default_settings()

        # Modify several settings
        settings.data["logging_level"] = "CRITICAL"
        settings.data["num_producers_stage1"] = 999
        settings.data["energy_saver_enabled"] = not defaults["energy_saver_enabled"]
        settings.data["custom_garbage"] = "should_be_removed"
        settings.save_settings()

        # Reset
        settings.reset_to_defaults()

        # Verify defaults restored
        assert settings.get("logging_level") == defaults["logging_level"]
        assert settings.get("num_producers_stage1") == defaults["num_producers_stage1"]
        assert settings.get("energy_saver_enabled") == defaults["energy_saver_enabled"]

        # Custom key should be gone after reset_to_defaults (data is replaced)
        assert "custom_garbage" not in settings.data

    def test_reset_persists_to_disk(self, tmp_path):
        """reset_to_defaults should also persist the reset state to disk."""
        settings = _create_settings(tmp_path)
        settings.data["logging_level"] = "CRITICAL"
        settings.save_settings()

        settings.reset_to_defaults()

        # Re-read from disk
        with open(settings.settings_file, "r") as f:
            on_disk = json.load(f)

        defaults = settings.get_default_settings()
        assert on_disk["logging_level"] == defaults["logging_level"]


# ---------------------------------------------------------------------------
# Missing keys use defaults
# ---------------------------------------------------------------------------

@pytest.mark.functional
class TestSettingsMissingKeysUseDefaults:
    """Create settings with partial data, verify missing keys return defaults."""

    def test_settings_missing_keys_use_defaults(self, tmp_path):
        # Write a partial settings file with only a few keys
        partial_data = {
            "logging_level": "WARNING",
            "num_producers_stage1": 4,
        }
        settings = _create_settings(tmp_path, initial_data=partial_data)
        defaults = settings.get_default_settings()

        # Explicitly set keys should be as provided
        assert settings.get("logging_level") == "WARNING"
        assert settings.get("num_producers_stage1") == 4

        # Missing keys should fall back to defaults
        assert settings.get("energy_saver_enabled") == defaults["energy_saver_enabled"]
        assert settings.get("autosave_interval_seconds") == defaults["autosave_interval_seconds"]
        assert settings.get("output_folder_path") == defaults["output_folder_path"]
        assert settings.get("tracking_axis_mode") == defaults["tracking_axis_mode"]

    def test_get_unknown_key_returns_parameter_default(self, tmp_path):
        """get() for a totally unknown key returns the provided default."""
        settings = _create_settings(tmp_path)

        assert settings.get("nonexistent_key_xyz") is None
        assert settings.get("nonexistent_key_xyz", 42) == 42
        assert settings.get("nonexistent_key_xyz", "fallback") == "fallback"

    def test_partial_shortcuts_merged_with_defaults(self, tmp_path):
        """When loading a file with partial shortcuts, missing shortcuts get defaults."""
        partial_data = {
            "funscript_editor_shortcuts": {
                "play_pause": "SPACE",
            }
        }
        settings = _create_settings(tmp_path, initial_data=partial_data)
        defaults = settings.get_default_settings()

        shortcuts = settings.get("funscript_editor_shortcuts")
        assert shortcuts is not None
        assert isinstance(shortcuts, dict)
        # Our custom shortcut should be present
        assert shortcuts.get("play_pause") == "SPACE"

        # Default shortcuts that were NOT in the partial file should also be present
        default_shortcuts = defaults.get("funscript_editor_shortcuts", {})
        for key in default_shortcuts:
            assert key in shortcuts, (
                f"Default shortcut '{key}' missing after merge"
            )

    def test_settings_profiles_roundtrip(self, tmp_path):
        """Save and load a settings profile, verify values restored."""
        settings = _create_settings(tmp_path)

        # Modify a profiled key
        settings.data["num_producers_stage1"] = 12
        settings.data["hardware_acceleration_method"] = "cuda"
        settings.save_settings()

        # Save as profile
        result = settings.save_profile("test_profile")
        assert result is True

        # Reset the values
        settings.data["num_producers_stage1"] = 1
        settings.data["hardware_acceleration_method"] = "none"
        settings.save_settings()

        # Load profile back
        result = settings.load_profile("test_profile")
        assert result is True
        assert settings.get("num_producers_stage1") == 12
        assert settings.get("hardware_acceleration_method") == "cuda"

    def test_list_and_delete_profiles(self, tmp_path):
        """Verify profile listing and deletion."""
        settings = _create_settings(tmp_path)
        settings.save_profile("profile_a")
        settings.save_profile("profile_b")

        profiles = settings.list_profiles()
        names = [p["name"] for p in profiles]
        assert "profile_a" in names
        assert "profile_b" in names

        # Delete one
        settings.delete_profile("profile_a")
        profiles = settings.list_profiles()
        names = [p["name"] for p in profiles]
        assert "profile_a" not in names
        assert "profile_b" in names
