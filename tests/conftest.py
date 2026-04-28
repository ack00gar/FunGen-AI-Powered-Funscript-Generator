"""Shared fixtures. Ensures repo root is importable and provides mock app + tmp paths."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_settings_path(tmp_path: Path) -> Path:
    return tmp_path / "settings.json"


@pytest.fixture
def mock_app():
    """Minimal stand-in for ApplicationLogic for BatchWorker tests.

    Exposes only the attributes BatchWorker actually touches:
      app_state_ui.selected_tracker_name, app_settings.get,
      is_batch_processing_active, stage_processor, processor, gui_instance,
      file_manager (only referenced by the sequential path; set to None).
    """
    app = types.SimpleNamespace()
    app.app_state_ui = types.SimpleNamespace(selected_tracker_name="")
    app.app_settings = types.SimpleNamespace(get=lambda k, d=None: d)
    app.is_batch_processing_active = False
    app.stage_processor = None
    app.processor = None
    app.gui_instance = None
    app.file_manager = None
    return app
