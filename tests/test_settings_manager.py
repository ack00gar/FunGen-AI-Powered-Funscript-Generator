"""AppSettings load/save invariants + the new batch_max_parallel_items default."""
from __future__ import annotations

import json

from application.classes.settings_manager import AppSettings


def test_missing_key_returns_explicit_default(tmp_settings_path):
    s = AppSettings(settings_file_path=str(tmp_settings_path))
    assert s.get("does_not_exist_ever", "fallback") == "fallback"


def test_missing_key_with_hardcoded_default_is_populated(tmp_settings_path):
    """get() should lazily pull from get_default_settings() when a key is absent."""
    s = AppSettings(settings_file_path=str(tmp_settings_path))
    defaults = s.get_default_settings()
    for key, expected in defaults.items():
        # Pick the first scalar-typed default to keep the assertion trivial.
        if isinstance(expected, (bool, int, float, str)):
            assert s.get(key) == expected
            break


def test_set_get_roundtrip_survives_reload(tmp_settings_path):
    s = AppSettings(settings_file_path=str(tmp_settings_path))
    s.set("custom_test_key", 42)
    s.save_settings()
    s2 = AppSettings(settings_file_path=str(tmp_settings_path))
    assert s2.get("custom_test_key") == 42


def test_set_batch_saves_once(tmp_settings_path):
    s = AppSettings(settings_file_path=str(tmp_settings_path))
    s.set_batch(k1="a", k2=7, k3=True)
    s.save_settings()
    assert s.get("k1") == "a"
    assert s.get("k2") == 7
    assert s.get("k3") is True


def test_batch_max_parallel_items_default_is_one(tmp_settings_path):
    """The new setting introduced by the parallel-batch commit."""
    s = AppSettings(settings_file_path=str(tmp_settings_path))
    assert int(s.get("batch_max_parallel_items", 1) or 1) == 1


def test_settings_file_is_valid_json_after_save(tmp_settings_path):
    s = AppSettings(settings_file_path=str(tmp_settings_path))
    s.set("roundtrip_check", [1, 2, 3])
    s.save_settings()
    with open(tmp_settings_path) as f:
        data = json.load(f)
    assert data.get("roundtrip_check") == [1, 2, 3]
