"""Unit tests for VideoSegment class."""
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestVideoSegmentCreation:
    """Tests for VideoSegment construction."""

    def test_creation_with_required_params(self, video_segment):
        """Segment is created with all expected attributes."""
        seg = video_segment
        assert seg.start_frame_id == 0
        assert seg.end_frame_id == 300
        assert seg.class_id == 1
        assert seg.class_name == "person"
        assert seg.segment_type == "detection"
        assert seg.position_short_name == "BJ"
        assert seg.position_long_name == "Blowjob"
        assert seg.duration == 300
        assert seg.occlusions == 0
        assert seg.source == "manual"

    def test_creation_with_defaults(self):
        """Default values for optional params are applied."""
        from application.utils import VideoSegment
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            seg = VideoSegment(
                start_frame_id=10,
                end_frame_id=20,
                class_id=2,
                class_name="test",
                segment_type="test_type",
                position_short_name="HJ",
                position_long_name="Handjob",
            )
        assert seg.duration == 0
        assert seg.occlusions == 0
        assert seg.source == "manual"
        assert seg.user_roi_fixed is None
        assert seg.user_roi_initial_point_relative is None
        assert seg.refined_track_id is None

    def test_frame_ids_cast_to_int(self):
        """start_frame_id and end_frame_id are cast to int."""
        from application.utils import VideoSegment
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            seg = VideoSegment(
                start_frame_id=10.5,
                end_frame_id=20.7,
                class_id=1,
                class_name="test",
                segment_type="t",
                position_short_name="BJ",
                position_long_name="Blowjob",
            )
        assert isinstance(seg.start_frame_id, int)
        assert isinstance(seg.end_frame_id, int)

    def test_none_class_id_allowed(self):
        """class_id can be None."""
        from application.utils import VideoSegment
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            seg = VideoSegment(
                start_frame_id=0,
                end_frame_id=100,
                class_id=None,
                class_name="unknown",
                segment_type="test",
                position_short_name="BJ",
                position_long_name="Blowjob",
            )
        assert seg.class_id is None

    def test_custom_source(self):
        """Custom source value is stored."""
        from application.utils import VideoSegment
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            seg = VideoSegment(
                start_frame_id=0,
                end_frame_id=100,
                class_id=1,
                class_name="test",
                segment_type="t",
                position_short_name="BJ",
                position_long_name="Blowjob",
                source="auto_detection",
            )
        assert seg.source == "auto_detection"


@pytest.mark.unit
class TestVideoSegmentUniqueId:
    """Tests for unique_id generation."""

    def test_unique_id_generated(self, video_segment):
        """Each segment gets a unique_id starting with 'segment_'."""
        assert video_segment.unique_id.startswith("segment_")

    def test_unique_ids_differ(self):
        """Two segments have different unique_ids."""
        from application.utils import VideoSegment
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            seg1 = VideoSegment(0, 100, 1, "a", "t", "BJ", "Blowjob")
            seg2 = VideoSegment(0, 100, 1, "a", "t", "BJ", "Blowjob")
        assert seg1.unique_id != seg2.unique_id


@pytest.mark.unit
class TestVideoSegmentColor:
    """Tests for color assignment based on position_short_name."""

    def test_default_color_for_bj(self):
        """BJ position gets the BJ color from SegmentColors."""
        from application.utils import VideoSegment
        from config.element_group_colors import SegmentColors
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            seg = VideoSegment(0, 100, 1, "test", "t", "BJ", "Blowjob")
        assert seg.color == SegmentColors.BJ

    def test_default_color_for_hj(self):
        """HJ position gets the HJ color from SegmentColors."""
        from application.utils import VideoSegment
        from config.element_group_colors import SegmentColors
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            seg = VideoSegment(0, 100, 1, "test", "t", "HJ", "Handjob")
        assert seg.color == SegmentColors.HJ

    def test_default_color_for_cg(self):
        """CG position gets the CG color from SegmentColors."""
        from application.utils import VideoSegment
        from config.element_group_colors import SegmentColors
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            seg = VideoSegment(0, 100, 1, "test", "t", "CG", "Cowgirl")
        assert seg.color == SegmentColors.CG

    def test_custom_color_overrides_default(self):
        """Explicit color parameter overrides the position-based default."""
        from application.utils import VideoSegment
        custom_color = (1.0, 0.0, 0.0, 1.0)
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            seg = VideoSegment(
                0, 100, 1, "test", "t", "BJ", "Blowjob",
                color=custom_color,
            )
        assert seg.color == custom_color

    def test_unknown_position_gets_default_color(self):
        """Unknown position_short_name gets the DEFAULT color."""
        from application.utils import VideoSegment
        from config.element_group_colors import SegmentColors
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            seg = VideoSegment(0, 100, 1, "test", "t", "UNKNOWN_POS", "Unknown Position")
        assert seg.color == SegmentColors.DEFAULT

    def test_color_is_tuple(self):
        """Color is stored as a tuple."""
        from application.utils import VideoSegment
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            seg = VideoSegment(
                0, 100, 1, "test", "t", "BJ", "Blowjob",
                color=[0.5, 0.5, 0.5, 1.0],
            )
        assert isinstance(seg.color, tuple)


@pytest.mark.unit
class TestVideoSegmentTimecodeConversion:
    """Tests for frame-to-timecode and timecode-to-frame conversions."""

    def test_frames_to_timecode_basic(self):
        """Basic frame to timecode conversion at 30fps."""
        from application.utils import VideoSegment
        tc = VideoSegment._frames_to_timecode(900, 30.0)
        assert tc == "00:00:30.000"

    def test_frames_to_timecode_with_hours(self):
        """Frame to timecode with hours component."""
        from application.utils import VideoSegment
        # 30fps * 3600s = 108000 frames = 1 hour
        tc = VideoSegment._frames_to_timecode(108000, 30.0)
        assert tc == "01:00:00.000"

    def test_frames_to_timecode_zero_fps(self):
        """Zero fps returns default timecode."""
        from application.utils import VideoSegment
        tc = VideoSegment._frames_to_timecode(100, 0.0)
        assert tc == "00:00:00.000"

    def test_frames_to_timecode_negative_frames(self):
        """Negative frames are treated as 0."""
        from application.utils import VideoSegment
        tc = VideoSegment._frames_to_timecode(-10, 30.0)
        assert tc == "00:00:00.000"

    def test_timecode_to_frames_basic(self):
        """Basic timecode to frame conversion."""
        from application.utils import VideoSegment
        frames = VideoSegment._timecode_to_frames("00:00:30.000", 30.0)
        assert frames == 900

    def test_timecode_to_frames_roundtrip(self):
        """Frame -> timecode -> frame roundtrip is consistent."""
        from application.utils import VideoSegment
        original = 450
        fps = 30.0
        tc = VideoSegment._frames_to_timecode(original, fps)
        result = VideoSegment._timecode_to_frames(tc, fps)
        assert abs(result - original) <= 1  # Allow rounding tolerance

    def test_timecode_to_frames_zero_fps(self):
        """Zero fps returns 0 frames."""
        from application.utils import VideoSegment
        assert VideoSegment._timecode_to_frames("01:00:00.000", 0.0) == 0

    def test_timecode_to_frames_invalid_format(self):
        """Invalid timecode format returns 0."""
        from application.utils import VideoSegment
        assert VideoSegment._timecode_to_frames("invalid", 30.0) == 0


@pytest.mark.unit
class TestVideoSegmentToDict:
    """Tests for serialization to dictionary."""

    def test_to_dict_contains_all_fields(self, video_segment):
        """to_dict includes all expected keys."""
        d = video_segment.to_dict()
        expected_keys = [
            'start_frame_id', 'end_frame_id', 'class_id', 'class_name',
            'segment_type', 'position_short_name', 'position_long_name',
            'duration', 'occlusions', 'source', 'color', 'unique_id',
            'user_roi_fixed', 'user_roi_initial_point_relative', 'refined_track_id',
        ]
        for key in expected_keys:
            assert key in d

    def test_to_dict_color_is_list(self, video_segment):
        """to_dict converts color tuple to list for JSON serialization."""
        d = video_segment.to_dict()
        assert isinstance(d['color'], list)

    def test_to_dict_values_match(self, video_segment):
        """to_dict values match segment attributes."""
        d = video_segment.to_dict()
        assert d['start_frame_id'] == video_segment.start_frame_id
        assert d['end_frame_id'] == video_segment.end_frame_id
        assert d['class_name'] == video_segment.class_name
        assert d['unique_id'] == video_segment.unique_id


@pytest.mark.unit
class TestVideoSegmentFromDict:
    """Tests for deserialization from dictionary."""

    def test_from_dict_roundtrip(self, video_segment):
        """from_dict(to_dict()) produces equivalent segment."""
        d = video_segment.to_dict()
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            restored = type(video_segment).from_dict(d)
        assert restored.start_frame_id == video_segment.start_frame_id
        assert restored.end_frame_id == video_segment.end_frame_id
        assert restored.class_name == video_segment.class_name
        assert restored.unique_id == video_segment.unique_id

    def test_from_dict_restores_color(self, video_segment):
        """from_dict restores color as tuple."""
        d = video_segment.to_dict()
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            restored = type(video_segment).from_dict(d)
        assert isinstance(restored.color, tuple)

    def test_from_dict_missing_optional_keys(self):
        """from_dict handles missing optional keys with defaults."""
        from application.utils import VideoSegment
        minimal = {
            'start_frame_id': 0,
            'end_frame_id': 100,
            'class_name': 'test',
        }
        with patch('application.classes.chapter_type_manager.get_chapter_type_manager', return_value=None):
            seg = VideoSegment.from_dict(minimal)
        assert seg.start_frame_id == 0
        assert seg.end_frame_id == 100
        assert seg.class_name == 'test'


@pytest.mark.unit
class TestVideoSegmentValidation:
    """Tests for validation methods."""

    def test_is_valid_dict_with_required_keys(self):
        """is_valid_dict returns True with all required keys."""
        from application.utils import VideoSegment
        d = {
            "start_frame_id": 0,
            "end_frame_id": 100,
            "class_name": "test",
        }
        assert VideoSegment.is_valid_dict(d) is True

    def test_is_valid_dict_missing_key(self):
        """is_valid_dict returns False when required key is missing."""
        from application.utils import VideoSegment
        d = {"start_frame_id": 0, "end_frame_id": 100}
        assert VideoSegment.is_valid_dict(d) is False

    def test_is_valid_dict_non_dict(self):
        """is_valid_dict returns False for non-dict input."""
        from application.utils import VideoSegment
        assert VideoSegment.is_valid_dict("not a dict") is False
        assert VideoSegment.is_valid_dict(None) is False
        assert VideoSegment.is_valid_dict([]) is False


@pytest.mark.unit
class TestVideoSegmentRepr:
    """Tests for string representation."""

    def test_repr_contains_key_info(self, video_segment):
        """repr contains segment id, frame range, and name."""
        r = repr(video_segment)
        assert "VideoSegment" in r
        assert "segment_" in r
        assert "person" in r
        assert "BJ" in r


@pytest.mark.unit
class TestVideoSegmentTimeInputParsing:
    """Tests for the flexible time input parser."""

    def test_parse_frame_number(self):
        """Integer without decimal is parsed as frame number."""
        from application.utils import VideoSegment
        assert VideoSegment.parse_time_input_to_frames("150", 30.0) == 150

    def test_parse_seconds_with_decimal(self):
        """Decimal number is parsed as seconds and converted to frames."""
        from application.utils import VideoSegment
        result = VideoSegment.parse_time_input_to_frames("5.0", 30.0)
        assert result == 150  # 5.0 * 30 = 150

    def test_parse_mm_ss(self):
        """MM:SS format is parsed correctly."""
        from application.utils import VideoSegment
        result = VideoSegment.parse_time_input_to_frames("1:30", 30.0)
        assert result == 2700  # 90s * 30fps

    def test_parse_hh_mm_ss(self):
        """HH:MM:SS format is parsed correctly."""
        from application.utils import VideoSegment
        result = VideoSegment.parse_time_input_to_frames("0:01:30", 30.0)
        assert result == 2700  # 90s * 30fps

    def test_parse_empty_string(self):
        """Empty string returns -1."""
        from application.utils import VideoSegment
        assert VideoSegment.parse_time_input_to_frames("", 30.0) == -1

    def test_parse_zero_fps(self):
        """Zero fps returns -1."""
        from application.utils import VideoSegment
        assert VideoSegment.parse_time_input_to_frames("100", 0.0) == -1

    def test_ms_to_frame_idx(self):
        """ms_to_frame_idx correctly converts milliseconds to frame index."""
        from application.utils import VideoSegment
        # 1000ms at 30fps = frame 30
        assert VideoSegment.ms_to_frame_idx(1000, 1000, 30.0) == 30

    def test_ms_to_frame_idx_clamped(self):
        """ms_to_frame_idx is clamped to total_frames - 1."""
        from application.utils import VideoSegment
        assert VideoSegment.ms_to_frame_idx(999999, 100, 30.0) == 99
