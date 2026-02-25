"""Shared test fixtures for the FunGen test suite."""
import pytest
import sys
import os
import json
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test video path
TEST_VIDEO_PATH = "/Users/k00gar/Downloads/test_koogar_extra_short_A.mp4"


@pytest.fixture
def project_root():
    """Return the project root path."""
    return PROJECT_ROOT


@pytest.fixture
def test_video_path():
    """Return the test video path, skip if not available."""
    if not os.path.exists(TEST_VIDEO_PATH):
        pytest.skip(f"Test video not found: {TEST_VIDEO_PATH}")
    return TEST_VIDEO_PATH


@pytest.fixture
def temp_dir():
    """Create a temporary directory, cleaned up after test."""
    d = tempfile.mkdtemp(prefix="fungen_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_settings_file(temp_dir):
    """Create a temporary settings file."""
    return os.path.join(temp_dir, "test_settings.json")


@pytest.fixture
def sample_funscript_data():
    """Return sample funscript data for testing."""
    return {
        "version": "1.0",
        "inverted": False,
        "range": 100,
        "metadata": {
            "creator": "FunGen Test",
            "type": "basic"
        },
        "actions": [
            {"at": 0, "pos": 50},
            {"at": 100, "pos": 0},
            {"at": 200, "pos": 100},
            {"at": 300, "pos": 0},
            {"at": 400, "pos": 100},
            {"at": 500, "pos": 50},
            {"at": 600, "pos": 0},
            {"at": 700, "pos": 100},
            {"at": 800, "pos": 25},
            {"at": 900, "pos": 75},
            {"at": 1000, "pos": 50},
        ]
    }


@pytest.fixture
def sample_actions():
    """Return sample action list."""
    return [
        {"at": 0, "pos": 50},
        {"at": 100, "pos": 0},
        {"at": 200, "pos": 100},
        {"at": 300, "pos": 0},
        {"at": 400, "pos": 100},
        {"at": 500, "pos": 50},
        {"at": 600, "pos": 0},
        {"at": 700, "pos": 100},
        {"at": 800, "pos": 25},
        {"at": 900, "pos": 75},
        {"at": 1000, "pos": 50},
    ]


@pytest.fixture
def large_sample_actions():
    """Return a larger set of actions for performance testing."""
    import random
    random.seed(42)
    actions = []
    for i in range(10000):
        actions.append({"at": i * 33, "pos": random.randint(0, 100)})
    return actions


@pytest.fixture
def funscript_file(temp_dir, sample_funscript_data):
    """Create a temporary funscript file."""
    path = os.path.join(temp_dir, "test.funscript")
    with open(path, 'w') as f:
        json.dump(sample_funscript_data, f)
    return path


@pytest.fixture
def multi_axis_funscript():
    """Create and return a MultiAxisFunscript instance with sample data."""
    from funscript.multi_axis_funscript import MultiAxisFunscript
    fs = MultiAxisFunscript()
    fs.primary_actions = [
        {"at": 0, "pos": 50},
        {"at": 100, "pos": 0},
        {"at": 200, "pos": 100},
        {"at": 300, "pos": 0},
        {"at": 400, "pos": 100},
        {"at": 500, "pos": 50},
    ]
    fs._invalidate_cache('primary')
    return fs


@pytest.fixture
def empty_multi_axis_funscript():
    """Create and return an empty MultiAxisFunscript instance."""
    from funscript.multi_axis_funscript import MultiAxisFunscript
    return MultiAxisFunscript()


@pytest.fixture
def app_settings(temp_settings_file):
    """Create AppSettings with a temp file."""
    from application.classes import AppSettings
    settings = AppSettings(settings_file_path=temp_settings_file)
    return settings


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = logging.getLogger("test_fungen")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def mock_app():
    """Create a minimal mock ApplicationLogic for testing components that need it."""
    mock = MagicMock()
    mock.logger = logging.getLogger("mock_app")
    mock.logger.setLevel(logging.DEBUG)
    mock.app_settings = MagicMock()
    mock.app_settings.get = MagicMock(return_value=None)
    mock.app_settings.set = MagicMock()
    mock.app_state_ui = MagicMock()
    mock.app_state_ui.selected_tracker_name = "live_yolo_roi"
    mock.app_state_ui.ui_view_mode = "expert"
    mock.app_state_ui.ui_layout_mode = "fixed"
    mock.app_state_ui.show_advanced_options = True
    mock.processor = None
    mock.tracker = None
    mock.stage_processor = MagicMock()
    mock.stage_processor.full_analysis_active = False
    mock.funscript_processor = MagicMock()
    mock.event_handlers = MagicMock()
    mock.calibration = MagicMock()
    mock.calibration.is_calibration_mode_active = False
    mock.project_manager = MagicMock()
    mock.project_manager.project_dirty = False
    mock.is_setting_user_roi_mode = False
    mock.is_cli_mode = False
    mock.cached_class_names = None
    mock.shortcut_manager = MagicMock()
    mock.undo_redo_manager_t1 = MagicMock()
    mock.undo_redo_manager_t2 = MagicMock()
    return mock


@pytest.fixture
def plugin_registry():
    """Create a fresh PluginRegistry for testing."""
    from funscript.plugins.base_plugin import PluginRegistry
    return PluginRegistry()


@pytest.fixture
def loaded_plugin_registry():
    """Create a PluginRegistry with all built-in plugins loaded."""
    from funscript.plugins.base_plugin import PluginRegistry
    from funscript.plugins.plugin_loader import PluginLoader
    registry = PluginRegistry()
    loader = PluginLoader(registry)
    loader.load_builtin_plugins()
    return registry


@pytest.fixture
def tracker_discovery():
    """Get the tracker discovery singleton."""
    from config.tracker_discovery import get_tracker_discovery
    return get_tracker_discovery()


@pytest.fixture
def undo_redo_manager():
    """Create a fresh UndoRedoManager."""
    from application.classes import UndoRedoManager
    return UndoRedoManager(max_history=50)


@pytest.fixture
def video_segment():
    """Create a sample VideoSegment."""
    from application.utils import VideoSegment
    return VideoSegment(
        start_frame_id=0,
        end_frame_id=300,
        class_id=1,
        class_name="person",
        segment_type="detection",
        position_short_name="BJ",
        position_long_name="Blowjob",
        duration=300,
        occlusions=0,
    )
