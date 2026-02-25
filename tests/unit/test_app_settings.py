"""Unit tests for AppSettings class."""
import pytest
import json
import os
import logging
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestAppSettingsCreation:
    """Tests for AppSettings construction and defaults."""

    def test_creation_loads_defaults(self, app_settings):
        """Newly created AppSettings has default values."""
        assert app_settings.data is not None
        assert len(app_settings.data) > 0

    def test_first_run_detected_when_no_file(self, temp_dir):
        """is_first_run is True when no settings file exists."""
        from application.classes import AppSettings
        path = os.path.join(temp_dir, "nonexistent_settings.json")
        with patch.object(AppSettings, 'auto_detect_hardware_acceleration'):
            settings = AppSettings(settings_file_path=path)
        assert settings.is_first_run is True

    def test_first_run_false_when_file_exists(self, app_settings, temp_settings_file):
        """is_first_run is False when settings file already exists."""
        # app_settings fixture creates the file on first run, reload it
        from application.classes import AppSettings
        settings2 = AppSettings(settings_file_path=temp_settings_file)
        assert settings2.is_first_run is False

    def test_settings_file_created_on_first_run(self, temp_dir):
        """Settings file is created on first run."""
        from application.classes import AppSettings
        path = os.path.join(temp_dir, "new_settings.json")
        assert not os.path.exists(path)
        with patch.object(AppSettings, 'auto_detect_hardware_acceleration'):
            AppSettings(settings_file_path=path)
        assert os.path.exists(path)

    def test_custom_logger(self, temp_dir):
        """AppSettings can accept a custom logger."""
        from application.classes import AppSettings
        logger = logging.getLogger("custom_test_logger")
        path = os.path.join(temp_dir, "settings_logger_test.json")
        with patch.object(AppSettings, 'auto_detect_hardware_acceleration'):
            settings = AppSettings(settings_file_path=path, logger=logger)
        assert settings.logger is logger


@pytest.mark.unit
class TestAppSettingsGetSet:
    """Tests for get/set operations."""

    def test_get_existing_key(self, app_settings):
        """get() returns the stored value for an existing key."""
        app_settings.data["test_key"] = "test_value"
        assert app_settings.get("test_key") == "test_value"

    def test_get_missing_key_returns_default(self, app_settings):
        """get() returns the default when key is missing."""
        assert app_settings.get("nonexistent_key", "fallback") == "fallback"

    def test_get_missing_key_checks_hardcoded_defaults(self, app_settings):
        """get() falls back to hardcoded defaults for known keys."""
        # Remove a known key from data to test fallback
        known_key = "autosave_enabled"
        if known_key in app_settings.data:
            del app_settings.data[known_key]
        result = app_settings.get(known_key)
        # Should be restored from defaults
        assert result is not None
        assert known_key in app_settings.data  # Should be added back

    def test_get_unknown_key_returns_none(self, app_settings):
        """get() returns None for completely unknown keys with no default."""
        assert app_settings.get("completely_unknown_key_xyz") is None

    def test_set_stores_value(self, app_settings):
        """set() stores the value in data."""
        app_settings.set("new_key", 42)
        assert app_settings.data["new_key"] == 42

    def test_set_persists_immediately(self, app_settings, temp_settings_file):
        """set() triggers save_settings to persist immediately."""
        app_settings.set("persist_test", "hello")
        # Verify file was written
        with open(temp_settings_file, 'r') as f:
            saved = json.load(f)
        assert saved["persist_test"] == "hello"

    def test_set_overwrites_existing(self, app_settings):
        """set() overwrites existing values."""
        app_settings.set("overwrite_key", "first")
        app_settings.set("overwrite_key", "second")
        assert app_settings.get("overwrite_key") == "second"

    def test_set_batch_saves_once(self, app_settings, temp_settings_file):
        """set_batch() sets multiple keys and saves only once."""
        app_settings.set_batch(key_a="val_a", key_b="val_b", key_c="val_c")
        assert app_settings.data["key_a"] == "val_a"
        assert app_settings.data["key_b"] == "val_b"
        assert app_settings.data["key_c"] == "val_c"
        # Verify all persisted
        with open(temp_settings_file, 'r') as f:
            saved = json.load(f)
        assert saved["key_a"] == "val_a"

    def test_set_various_types(self, app_settings):
        """set() handles various value types: int, float, bool, list, dict."""
        app_settings.set("int_val", 42)
        app_settings.set("float_val", 3.14)
        app_settings.set("bool_val", True)
        app_settings.set("list_val", [1, 2, 3])
        app_settings.set("dict_val", {"nested": True})
        assert app_settings.get("int_val") == 42
        assert app_settings.get("float_val") == 3.14
        assert app_settings.get("bool_val") is True
        assert app_settings.get("list_val") == [1, 2, 3]
        assert app_settings.get("dict_val") == {"nested": True}


@pytest.mark.unit
class TestAppSettingsPersistence:
    """Tests for save/load cycle."""

    def test_save_and_load_roundtrip(self, temp_dir):
        """Settings saved and then loaded match."""
        from application.classes import AppSettings
        path = os.path.join(temp_dir, "roundtrip.json")
        with patch.object(AppSettings, 'auto_detect_hardware_acceleration'):
            s1 = AppSettings(settings_file_path=path)
        s1.set("roundtrip_test", "value123")

        s2 = AppSettings(settings_file_path=path)
        assert s2.get("roundtrip_test") == "value123"

    def test_load_merges_with_defaults(self, temp_dir):
        """Loading merges saved data with defaults (new defaults are added)."""
        from application.classes import AppSettings
        path = os.path.join(temp_dir, "merge_test.json")
        # Write a minimal settings file
        with open(path, 'w') as f:
            json.dump({"custom_setting": "custom_value"}, f)
        
        settings = AppSettings(settings_file_path=path)
        # Custom setting should be present
        assert settings.get("custom_setting") == "custom_value"
        # Default settings should also be present
        assert "autosave_enabled" in settings.data

    def test_corrupt_file_falls_back_to_defaults(self, temp_dir):
        """Corrupt settings file causes fallback to defaults."""
        from application.classes import AppSettings
        path = os.path.join(temp_dir, "corrupt.json")
        with open(path, 'w') as f:
            f.write("NOT VALID JSON {{{")
        
        settings = AppSettings(settings_file_path=path)
        # Should have default values despite corrupt file
        assert "autosave_enabled" in settings.data

    def test_missing_file_creates_defaults(self, temp_dir):
        """Missing settings file creates file with defaults."""
        from application.classes import AppSettings
        path = os.path.join(temp_dir, "missing_settings.json")
        with patch.object(AppSettings, 'auto_detect_hardware_acceleration'):
            settings = AppSettings(settings_file_path=path)
        assert os.path.exists(path)
        with open(path, 'r') as f:
            saved_data = json.load(f)
        assert "autosave_enabled" in saved_data

    def test_save_creates_valid_json(self, app_settings, temp_settings_file):
        """save_settings creates valid JSON."""
        app_settings.set("valid_json_test", True)
        with open(temp_settings_file, 'r') as f:
            data = json.load(f)  # Should not raise
        assert isinstance(data, dict)


@pytest.mark.unit
class TestAppSettingsReset:
    """Tests for reset_to_defaults."""

    def test_reset_restores_defaults(self, app_settings):
        """reset_to_defaults restores all values to defaults."""
        app_settings.set("custom_key", "custom_value")
        app_settings.reset_to_defaults()
        # Custom key should be gone (not in defaults)
        assert "custom_key" not in app_settings.data
        # Default key should be restored
        assert "autosave_enabled" in app_settings.data

    def test_reset_persists(self, app_settings, temp_settings_file):
        """reset_to_defaults saves the default state to file."""
        app_settings.set("pre_reset", True)
        app_settings.reset_to_defaults()
        with open(temp_settings_file, 'r') as f:
            saved = json.load(f)
        assert "pre_reset" not in saved

    def test_reset_returns_fresh_defaults(self, app_settings):
        """After reset, get_default_settings matches current data."""
        app_settings.reset_to_defaults()
        defaults = app_settings.get_default_settings()
        for key in defaults:
            assert key in app_settings.data


@pytest.mark.unit
class TestAppSettingsDefaultValues:
    """Tests for specific default values."""

    def test_default_autosave_enabled(self, app_settings):
        """Autosave is enabled by default."""
        assert app_settings.get("autosave_enabled") is True

    def test_default_autosave_interval(self, app_settings):
        """Autosave interval defaults to 120 seconds."""
        assert app_settings.get("autosave_interval_seconds") == 120

    def test_default_output_folder(self, app_settings):
        """Output folder has a default value."""
        assert app_settings.get("output_folder_path") is not None

    def test_default_energy_saver(self, app_settings):
        """Energy saver is enabled by default."""
        assert app_settings.get("energy_saver_enabled") is True

    def test_default_recent_projects_empty(self, app_settings):
        """Recent projects list is empty by default."""
        assert app_settings.get("recent_projects") == []


@pytest.mark.unit
class TestAppSettingsProfiles:
    """Tests for settings profiles functionality."""

    def test_save_and_load_profile(self, app_settings):
        """Saving and loading a profile roundtrips correctly."""
        app_settings.set("live_tracker_sensitivity", 99.0)
        assert app_settings.save_profile("test_profile")
        
        # Change the value
        app_settings.data["live_tracker_sensitivity"] = 1.0
        
        # Load profile to restore
        assert app_settings.load_profile("test_profile")
        assert app_settings.get("live_tracker_sensitivity") == 99.0

    def test_list_profiles(self, app_settings):
        """list_profiles returns saved profiles."""
        app_settings.save_profile("profile_alpha")
        app_settings.save_profile("profile_beta")
        profiles = app_settings.list_profiles()
        names = [p["name"] for p in profiles]
        assert "profile_alpha" in names
        assert "profile_beta" in names

    def test_delete_profile(self, app_settings):
        """delete_profile removes the profile file."""
        app_settings.save_profile("to_delete")
        assert app_settings.delete_profile("to_delete") is True
        profiles = app_settings.list_profiles()
        names = [p["name"] for p in profiles]
        assert "to_delete" not in names

    def test_save_empty_name_fails(self, app_settings):
        """Saving a profile with empty name returns False."""
        assert app_settings.save_profile("") is False
        assert app_settings.save_profile("   ") is False

    def test_load_nonexistent_profile_fails(self, app_settings):
        """Loading a nonexistent profile returns False."""
        assert app_settings.load_profile("nonexistent_profile_xyz") is False
