"""Unit tests for OFS-compatible axis file naming.

Tests cover:
- _get_funscript_path_for_axis() path generation
- discover_axis_funscripts() auto-detection (OFS + legacy naming)
"""
import os
import pytest
from unittest.mock import MagicMock, patch
from funscript.axis_registry import file_suffix_for_axis


@pytest.mark.unit
class TestGetFunscriptPathForAxis:
    """Test the path generation helper for axis-specific funscript files."""

    @staticmethod
    def _get_path(video_path: str, axis_name: str) -> str:
        """Replicate the _get_funscript_path_for_axis logic directly."""
        suffix = file_suffix_for_axis(axis_name)
        base, _ = os.path.splitext(video_path)
        return f"{base}{suffix}.funscript"

    def test_stroke_no_suffix(self):
        result = self._get_path("/videos/test.mp4", "stroke")
        assert result == "/videos/test.funscript"

    def test_roll_suffix(self):
        result = self._get_path("/videos/test.mp4", "roll")
        assert result == "/videos/test.roll.funscript"

    def test_pitch_suffix(self):
        result = self._get_path("/videos/test.mp4", "pitch")
        assert result == "/videos/test.pitch.funscript"

    def test_twist_suffix(self):
        result = self._get_path("/videos/test.mp4", "twist")
        assert result == "/videos/test.twist.funscript"

    def test_sway_suffix(self):
        result = self._get_path("/videos/test.mp4", "sway")
        assert result == "/videos/test.sway.funscript"

    def test_surge_suffix(self):
        result = self._get_path("/videos/test.mp4", "surge")
        assert result == "/videos/test.surge.funscript"

    def test_vib_suffix(self):
        result = self._get_path("/videos/test.mp4", "vib")
        assert result == "/videos/test.vib.funscript"

    def test_pump_suffix(self):
        result = self._get_path("/videos/test.mp4", "pump")
        assert result == "/videos/test.pump.funscript"

    def test_custom_axis(self):
        result = self._get_path("/videos/test.mp4", "my_custom")
        assert result == "/videos/test.my_custom.funscript"

    def test_mkv_extension(self):
        result = self._get_path("/videos/movie.mkv", "roll")
        assert result == "/videos/movie.roll.funscript"

    def test_path_with_spaces(self):
        result = self._get_path("/my videos/test file.mp4", "pitch")
        assert result == "/my videos/test file.pitch.funscript"


@pytest.mark.unit
class TestDiscoverAxisFunscripts:
    """Test auto-detection of axis funscript files next to a video."""

    def test_discover_ofs_named_files(self, tmp_path):
        """Test discovery of OFS-convention named files."""
        video = tmp_path / "video.mp4"
        video.touch()
        # Create OFS-named funscripts
        (tmp_path / "video.funscript").touch()
        (tmp_path / "video.roll.funscript").touch()
        (tmp_path / "video.pitch.funscript").touch()

        from application.logic.app_file_manager import AppFileManager
        fm = MagicMock(spec=AppFileManager)
        fm.discover_axis_funscripts = AppFileManager.discover_axis_funscripts.__get__(fm)

        result = fm.discover_axis_funscripts(str(video))
        assert "stroke" in result
        assert "roll" in result
        assert "pitch" in result
        assert result["stroke"] == str(tmp_path / "video.funscript")
        assert result["roll"] == str(tmp_path / "video.roll.funscript")
        assert result["pitch"] == str(tmp_path / "video.pitch.funscript")

    def test_discover_legacy_t_named_files(self, tmp_path):
        """Test discovery of legacy _tN named files."""
        video = tmp_path / "video.mp4"
        video.touch()
        (tmp_path / "video_t1.funscript").touch()
        (tmp_path / "video_t2.funscript").touch()
        (tmp_path / "video_t3.funscript").touch()

        from application.logic.app_file_manager import AppFileManager
        fm = MagicMock(spec=AppFileManager)
        fm.discover_axis_funscripts = AppFileManager.discover_axis_funscripts.__get__(fm)

        result = fm.discover_axis_funscripts(str(video))
        assert "stroke" in result
        assert "roll" in result
        assert "axis_3" in result

    def test_ofs_naming_takes_priority_over_legacy(self, tmp_path):
        """When both OFS and legacy files exist for the same axis, OFS wins."""
        video = tmp_path / "video.mp4"
        video.touch()
        (tmp_path / "video.roll.funscript").touch()  # OFS naming
        (tmp_path / "video_t2.funscript").touch()     # Legacy naming

        from application.logic.app_file_manager import AppFileManager
        fm = MagicMock(spec=AppFileManager)
        fm.discover_axis_funscripts = AppFileManager.discover_axis_funscripts.__get__(fm)

        result = fm.discover_axis_funscripts(str(video))
        assert result["roll"] == str(tmp_path / "video.roll.funscript")

    def test_no_video_returns_empty(self):
        from application.logic.app_file_manager import AppFileManager
        fm = MagicMock(spec=AppFileManager)
        fm.discover_axis_funscripts = AppFileManager.discover_axis_funscripts.__get__(fm)

        result = fm.discover_axis_funscripts("")
        assert result == {}

    def test_nonexistent_video_returns_empty(self):
        from application.logic.app_file_manager import AppFileManager
        fm = MagicMock(spec=AppFileManager)
        fm.discover_axis_funscripts = AppFileManager.discover_axis_funscripts.__get__(fm)

        result = fm.discover_axis_funscripts("/nonexistent/video.mp4")
        assert result == {}

    def test_mixed_naming(self, tmp_path):
        """Test with some OFS names and some legacy names."""
        video = tmp_path / "video.mp4"
        video.touch()
        (tmp_path / "video.funscript").touch()        # Primary (OFS)
        (tmp_path / "video.roll.funscript").touch()   # Roll (OFS)
        (tmp_path / "video_t3.funscript").touch()     # T3 (legacy)

        from application.logic.app_file_manager import AppFileManager
        fm = MagicMock(spec=AppFileManager)
        fm.discover_axis_funscripts = AppFileManager.discover_axis_funscripts.__get__(fm)

        result = fm.discover_axis_funscripts(str(video))
        assert len(result) == 3
        assert "stroke" in result
        assert "roll" in result
        assert "axis_3" in result
