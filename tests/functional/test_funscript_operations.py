"""
Functional tests for full funscript workflows.

These tests exercise complete end-to-end scenarios: creating a MultiAxisFunscript,
editing it, saving/loading from disk, undo/redo, interpolation accuracy,
chapter management, and statistics computation.
"""

import copy
import json
import os
import sys
import tempfile
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from funscript import MultiAxisFunscript
from application.classes.undo_redo_manager import UndoRedoManager
from application.utils.video_segment import VideoSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_funscript_json(actions, chapters=None):
    """Build a standard funscript JSON structure from an actions list."""
    data = {
        "version": "1.0",
        "author": "test",
        "inverted": False,
        "range": 100,
        "actions": sorted(actions, key=lambda a: a["at"]),
        "metadata": {
            "version": "0.2.0",
            "chapters": chapters or [],
        },
    }
    return data


# ---------------------------------------------------------------------------
# Create / Edit / Save / Load
# ---------------------------------------------------------------------------

@pytest.mark.functional
class TestCreateEditSaveLoadFunscript:
    """Create MultiAxisFunscript, add actions, save to file, load from file, verify."""

    def test_create_edit_save_load_funscript(self, tmp_path):
        import orjson

        # 1. Create
        fs = MultiAxisFunscript()

        # 2. Add actions using add_action (the real production method)
        test_data = [
            (0, 10, 50),
            (100, 30, 60),
            (200, 70, 40),
            (300, 90, 20),
            (400, 50, 80),
            (500, 20, 55),
        ]
        for ts, prim, sec in test_data:
            fs.add_action(ts, prim, sec)

        assert len(fs.primary_actions) >= len(test_data) - 2  # simplification may remove some
        assert len(fs.secondary_actions) >= len(test_data) - 2

        # 3. Save to file
        filepath = os.path.join(str(tmp_path), "test_workflow.funscript")
        funscript_data = _build_funscript_json(fs.primary_actions)
        with open(filepath, "wb") as f:
            f.write(orjson.dumps(funscript_data))

        # 4. Load from file
        with open(filepath, "rb") as f:
            loaded = orjson.loads(f.read())

        loaded_actions = sorted(loaded["actions"], key=lambda a: a["at"])

        # 5. Verify
        fs2 = MultiAxisFunscript()
        fs2.actions = loaded_actions  # uses the actions setter

        assert len(fs2.primary_actions) == len(fs.primary_actions)
        for orig, loaded_a in zip(fs.primary_actions, fs2.primary_actions):
            assert orig['at'] == loaded_a['at']
            assert orig['pos'] == loaded_a['pos']


# ---------------------------------------------------------------------------
# Undo / Redo with funscript edits
# ---------------------------------------------------------------------------

@pytest.mark.functional
class TestUndoRedoWithFunscriptEdits:
    """Wire UndoRedoManager to a funscript, make edits, undo, redo, verify."""

    def test_undo_redo_with_funscript_edits(self):
        fs = MultiAxisFunscript()

        # Populate with initial data
        initial_actions = [
            {"at": 0, "pos": 0},
            {"at": 500, "pos": 50},
            {"at": 1000, "pos": 100},
        ]
        fs.actions = [d.copy() for d in initial_actions]

        # Wire undo manager to the primary actions list
        undo_mgr = UndoRedoManager(max_history=50)
        undo_mgr.set_actions_reference(fs.primary_actions)

        # Verify initial state
        assert len(fs.primary_actions) == 3
        assert not undo_mgr.can_undo()
        assert not undo_mgr.can_redo()

        # --- Edit 1: Modify a position ---
        undo_mgr.record_state_before_action("Modify position")
        fs.primary_actions[1]['pos'] = 75  # was 50
        assert undo_mgr.can_undo()

        # --- Edit 2: Add a new action ---
        undo_mgr.record_state_before_action("Add action")
        fs.primary_actions.append({"at": 1500, "pos": 25})
        assert len(fs.primary_actions) == 4

        # --- Undo Edit 2 ---
        desc = undo_mgr.undo()
        assert desc == "Add action"
        assert len(fs.primary_actions) == 3
        assert fs.primary_actions[1]['pos'] == 75  # Edit 1 still present

        # --- Undo Edit 1 ---
        desc = undo_mgr.undo()
        assert desc == "Modify position"
        assert len(fs.primary_actions) == 3
        assert fs.primary_actions[1]['pos'] == 50  # Restored to original

        # --- Redo Edit 1 ---
        assert undo_mgr.can_redo()
        desc = undo_mgr.redo()
        assert desc == "Modify position"
        assert fs.primary_actions[1]['pos'] == 75

        # --- Redo Edit 2 ---
        desc = undo_mgr.redo()
        assert desc == "Add action"
        assert len(fs.primary_actions) == 4
        assert fs.primary_actions[3]['pos'] == 25

    def test_undo_clears_redo_on_new_action(self):
        """After undo, a new action should clear the redo stack."""
        fs = MultiAxisFunscript()
        fs.actions = [{"at": 0, "pos": 0}, {"at": 1000, "pos": 100}]
        undo_mgr = UndoRedoManager()
        undo_mgr.set_actions_reference(fs.primary_actions)

        undo_mgr.record_state_before_action("First edit")
        fs.primary_actions[0]['pos'] = 50

        undo_mgr.undo()
        assert undo_mgr.can_redo()

        # New edit after undo clears redo
        undo_mgr.record_state_before_action("Second edit")
        fs.primary_actions[0]['pos'] = 75
        assert not undo_mgr.can_redo()


# ---------------------------------------------------------------------------
# Interpolation accuracy
# ---------------------------------------------------------------------------

@pytest.mark.functional
class TestFunscriptInterpolationAccuracy:
    """Add actions at known points, verify get_value interpolates correctly."""

    def test_funscript_interpolation_accuracy(self):
        fs = MultiAxisFunscript()
        fs.actions = [
            {"at": 0, "pos": 0},
            {"at": 1000, "pos": 100},
            {"at": 2000, "pos": 0},
            {"at": 3000, "pos": 100},
        ]

        # Exact points
        assert fs.get_value(0, 'primary') == 0
        assert fs.get_value(1000, 'primary') == 100
        assert fs.get_value(2000, 'primary') == 0
        assert fs.get_value(3000, 'primary') == 100

        # Midpoint interpolation (linear)
        val_500 = fs.get_value(500, 'primary')
        assert 48 <= val_500 <= 52, f"Expected ~50, got {val_500}"

        val_1500 = fs.get_value(1500, 'primary')
        assert 48 <= val_1500 <= 52, f"Expected ~50, got {val_1500}"

        # Quarter points
        val_250 = fs.get_value(250, 'primary')
        assert 23 <= val_250 <= 27, f"Expected ~25, got {val_250}"

        val_750 = fs.get_value(750, 'primary')
        assert 73 <= val_750 <= 77, f"Expected ~75, got {val_750}"

    def test_interpolation_before_first_action(self):
        """Before the first action, get_value returns the first action's position."""
        fs = MultiAxisFunscript()
        fs.actions = [{"at": 100, "pos": 42}, {"at": 500, "pos": 80}]
        assert fs.get_value(0, 'primary') == 42

    def test_interpolation_after_last_action(self):
        """After the last action, get_value returns the last action's position."""
        fs = MultiAxisFunscript()
        fs.actions = [{"at": 0, "pos": 10}, {"at": 500, "pos": 80}]
        assert fs.get_value(10000, 'primary') == 80

    def test_interpolation_empty_funscript(self):
        """Empty funscript returns default neutral position."""
        fs = MultiAxisFunscript()
        assert fs.get_value(500, 'primary') == 50

    def test_interpolation_secondary_axis(self):
        """Verify interpolation works on the secondary axis."""
        fs = MultiAxisFunscript()
        fs.secondary_actions = [
            {"at": 0, "pos": 0},
            {"at": 1000, "pos": 100},
        ]
        fs._invalidate_cache('secondary')

        val = fs.get_value(500, 'secondary')
        assert 48 <= val <= 52


# ---------------------------------------------------------------------------
# Chapter management
# ---------------------------------------------------------------------------

@pytest.mark.functional
class TestChapterManagement:
    """Create VideoSegments, verify chapter operations."""

    def test_chapter_creation_and_dict_roundtrip(self):
        """Create VideoSegments, convert to dict and back."""
        seg = VideoSegment(
            start_frame_id=0,
            end_frame_id=900,
            class_id=None,
            class_name="TestClass",
            segment_type="SexAct",
            position_short_name="CG",
            position_long_name="Cowgirl",
            duration=900,
        )

        assert seg.start_frame_id == 0
        assert seg.end_frame_id == 900
        assert seg.position_short_name == "CG"

        # Round-trip through dict
        d = seg.to_dict()
        seg2 = VideoSegment.from_dict(d)
        assert seg2.start_frame_id == seg.start_frame_id
        assert seg2.end_frame_id == seg.end_frame_id
        assert seg2.class_name == seg.class_name
        assert seg2.position_short_name == seg.position_short_name

    def test_chapter_funscript_timecode_conversion(self):
        """Convert segment to funscript chapter dict and back."""
        fps = 30.0
        seg = VideoSegment(
            start_frame_id=0,
            end_frame_id=299,  # ~10 seconds at 30fps (inclusive)
            class_id=None,
            class_name="TestClass",
            segment_type="SexAct",
            position_short_name="BJ",
            position_long_name="Blowjob",
        )

        chap_dict = seg.to_funscript_chapter_dict(fps)
        assert "startTime" in chap_dict
        assert "endTime" in chap_dict
        assert chap_dict["name"] == "Blowjob"

        # Verify timecodes are reasonable (startTime should be 00:00:00.000)
        assert chap_dict["startTime"] == "00:00:00.000"

    def test_chapter_list_operations(self):
        """Verify chapters can be stored and managed as a list."""
        chapters = []
        for i in range(5):
            seg = VideoSegment(
                start_frame_id=i * 300,
                end_frame_id=(i + 1) * 300 - 1,
                class_id=None,
                class_name=f"Chapter {i}",
                segment_type="SexAct",
                position_short_name="CG",
                position_long_name=f"Chapter {i}",
            )
            chapters.append(seg)

        assert len(chapters) == 5
        # Sort by start_frame_id
        chapters.sort(key=lambda s: s.start_frame_id)
        for i, ch in enumerate(chapters):
            assert ch.start_frame_id == i * 300

    def test_video_segment_is_valid_dict(self):
        """Verify static validation of segment dictionaries."""
        valid = {"start_frame_id": 0, "end_frame_id": 100, "class_name": "Test"}
        assert VideoSegment.is_valid_dict(valid) is True

        invalid = {"start_frame_id": 0}  # missing keys
        assert VideoSegment.is_valid_dict(invalid) is False

        assert VideoSegment.is_valid_dict("not a dict") is False


# ---------------------------------------------------------------------------
# Funscript statistics
# ---------------------------------------------------------------------------

@pytest.mark.functional
class TestFunscriptStatistics:
    """Create known data, verify statistics computations."""

    def test_funscript_statistics(self):
        fs = MultiAxisFunscript()
        # Create a simple oscillating pattern: 0 -> 100 -> 0 -> 100 -> 0
        # Each step is 1000ms apart
        fs.actions = [
            {"at": 0, "pos": 0},
            {"at": 1000, "pos": 100},
            {"at": 2000, "pos": 0},
            {"at": 3000, "pos": 100},
            {"at": 4000, "pos": 0},
        ]

        stats = fs.get_actions_statistics('primary')

        assert stats["num_points"] == 5
        assert stats["min_pos"] == 0
        assert stats["max_pos"] == 100
        assert stats["duration_scripted_s"] == pytest.approx(4.0, abs=0.01)

        # Total travel distance: 100 + 100 + 100 + 100 = 400
        assert stats["total_travel_dist"] == 400

        # Average speed: 400 pos / 4 seconds = 100 pos/s
        assert stats["avg_speed_pos_per_s"] == pytest.approx(100.0, abs=1.0)

        # Average interval: 1000ms
        assert stats["avg_interval_ms"] == pytest.approx(1000.0, abs=1.0)

        # Strokes: direction changes at each point (0->100->0->100->0 = 3 reversals)
        assert stats["num_strokes"] >= 3

    def test_statistics_empty_funscript(self):
        fs = MultiAxisFunscript()
        stats = fs.get_actions_statistics('primary')
        assert stats["num_points"] == 0
        assert stats["duration_scripted_s"] == 0.0

    def test_statistics_single_point(self):
        fs = MultiAxisFunscript()
        fs.actions = [{"at": 0, "pos": 50}]
        stats = fs.get_actions_statistics('primary')
        assert stats["num_points"] == 1
        assert stats["min_pos"] == 50
        assert stats["max_pos"] == 50
        assert stats["duration_scripted_s"] == 0.0

    def test_statistics_two_points(self):
        fs = MultiAxisFunscript()
        fs.actions = [{"at": 0, "pos": 0}, {"at": 1000, "pos": 100}]
        stats = fs.get_actions_statistics('primary')
        assert stats["num_points"] == 2
        assert stats["total_travel_dist"] == 100
        assert stats["duration_scripted_s"] == pytest.approx(1.0, abs=0.01)
