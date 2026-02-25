"""Unit tests for funscript.axis_registry module."""
import pytest
from funscript.axis_registry import (
    FunscriptAxis, AXIS_FILE_SUFFIX, AXIS_TCODE, DEFAULT_TIMELINE_AXIS,
    axis_from_file_suffix, axis_from_tcode, file_suffix_for_axis, tcode_for_axis,
    all_known_suffixes,
)


@pytest.mark.unit
class TestFunscriptAxisEnum:
    def test_all_expected_members(self):
        expected = {"stroke", "roll", "pitch", "twist", "sway", "surge", "vib", "pump"}
        actual = {a.value for a in FunscriptAxis}
        assert actual == expected

    def test_enum_from_value(self):
        assert FunscriptAxis("stroke") is FunscriptAxis.STROKE
        assert FunscriptAxis("roll") is FunscriptAxis.ROLL


@pytest.mark.unit
class TestAxisFileSuffix:
    def test_stroke_has_empty_suffix(self):
        assert AXIS_FILE_SUFFIX[FunscriptAxis.STROKE] == ""

    def test_roll_suffix(self):
        assert AXIS_FILE_SUFFIX[FunscriptAxis.ROLL] == ".roll"

    def test_all_axes_have_suffix_entry(self):
        for axis in FunscriptAxis:
            assert axis in AXIS_FILE_SUFFIX


@pytest.mark.unit
class TestAxisTCode:
    def test_stroke_is_l0(self):
        assert AXIS_TCODE[FunscriptAxis.STROKE] == "L0"

    def test_roll_is_r1(self):
        assert AXIS_TCODE[FunscriptAxis.ROLL] == "R1"

    def test_all_axes_have_tcode(self):
        for axis in FunscriptAxis:
            assert axis in AXIS_TCODE


@pytest.mark.unit
class TestAxisFromFileSuffix:
    def test_roll_suffix(self):
        assert axis_from_file_suffix(".roll") is FunscriptAxis.ROLL

    def test_pitch_suffix(self):
        assert axis_from_file_suffix(".pitch") is FunscriptAxis.PITCH

    def test_unknown_suffix_returns_none(self):
        assert axis_from_file_suffix(".nonexistent") is None

    def test_round_trip_all_known(self):
        for axis in FunscriptAxis:
            suffix = AXIS_FILE_SUFFIX[axis]
            if suffix:  # Skip stroke (empty suffix)
                assert axis_from_file_suffix(suffix) is axis


@pytest.mark.unit
class TestAxisFromTCode:
    def test_l0_is_stroke(self):
        assert axis_from_tcode("L0") is FunscriptAxis.STROKE

    def test_r1_is_roll(self):
        assert axis_from_tcode("R1") is FunscriptAxis.ROLL

    def test_unknown_tcode_returns_none(self):
        assert axis_from_tcode("X9") is None

    def test_round_trip_all_known(self):
        for axis in FunscriptAxis:
            tcode = AXIS_TCODE[axis]
            assert axis_from_tcode(tcode) is axis


@pytest.mark.unit
class TestFileSuffixForAxis:
    def test_known_axis_stroke(self):
        assert file_suffix_for_axis("stroke") == ""

    def test_known_axis_roll(self):
        assert file_suffix_for_axis("roll") == ".roll"

    def test_custom_axis_name(self):
        assert file_suffix_for_axis("my_custom") == ".my_custom"


@pytest.mark.unit
class TestTCodeForAxis:
    def test_known_axis(self):
        assert tcode_for_axis("stroke") == "L0"
        assert tcode_for_axis("roll") == "R1"

    def test_unknown_axis_returns_none(self):
        assert tcode_for_axis("custom_axis") is None


@pytest.mark.unit
class TestAllKnownSuffixes:
    def test_does_not_include_empty_string(self):
        suffixes = all_known_suffixes()
        assert "" not in suffixes

    def test_includes_roll_and_pitch(self):
        suffixes = all_known_suffixes()
        assert ".roll" in suffixes
        assert ".pitch" in suffixes

    def test_count_matches_non_stroke_axes(self):
        # All axes except STROKE have non-empty suffixes
        suffixes = all_known_suffixes()
        assert len(suffixes) == len(FunscriptAxis) - 1


@pytest.mark.unit
class TestDefaultTimelineAxis:
    def test_timeline_1_is_stroke(self):
        assert DEFAULT_TIMELINE_AXIS[1] is FunscriptAxis.STROKE

    def test_timeline_2_is_roll(self):
        assert DEFAULT_TIMELINE_AXIS[2] is FunscriptAxis.ROLL
