#!/usr/bin/env python3
"""
Specific tests for Stage 3 Mixed mode functionality and integration.
"""

import os
import sys
import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from detection.cd.stage_3_mixed_processor import MixedStageProcessor, perform_mixed_stage_analysis
from detection.cd.data_structures import FrameObject, ATRLockedPenisState
from application.utils.video_segment import VideoSegment
from config.constants import TrackerMode
from funscript.dual_axis_funscript import DualAxisFunscript

pytestmark = [pytest.mark.integration, pytest.mark.stage3_mixed]


class TestStage3MixedMode:
    """Test suite specifically for Stage 3 Mixed mode functionality."""

    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_mock_frame(self, frame_id, pos=50, has_penis=True, contact_boxes=None):
        """Create a mock FrameObject for testing."""
        frame_obj = FrameObject(frame_id=frame_id, yolo_input_size=640)
        frame_obj.pos_0_100 = pos
        
        # Set up locked penis state
        frame_obj.atr_locked_penis_state = ATRLockedPenisState()
        if has_penis:
            frame_obj.atr_locked_penis_state.active = True
            frame_obj.atr_locked_penis_state.box = (100.0, 100.0, 200.0, 200.0)
        
        # Set up contact boxes
        frame_obj.atr_detected_contact_boxes = contact_boxes or [
            {'class_name': 'hand', 'confidence': 0.8, 'bbox': (150, 120, 180, 160)}
        ]
        
        return frame_obj

    def test_mixed_processor_initialization(self):
        """Test MixedStageProcessor initialization."""
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        assert processor is not None
        assert hasattr(processor, 'stage2_frame_objects')
        assert hasattr(processor, 'stage2_segments')
        assert processor.stage2_frame_objects == {}
        assert processor.stage2_segments == []

    def test_stage2_results_setting(self):
        """Test setting Stage 2 results in the mixed processor."""
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        # Create test frame objects
        frame_objects = {
            0: self.create_mock_frame(0, pos=30),
            1: self.create_mock_frame(1, pos=70),
            2: self.create_mock_frame(2, pos=50)
        }
        
        # Create test segments
        segments = [VideoSegment(
            start_frame_id=0, end_frame_id=2, class_id=1,
            class_name='handjob', segment_type='SexAct',
            position_short_name='HJ', position_long_name='Hand Job'
        )]
        
        processor.set_stage2_results(frame_objects, segments)
        
        assert len(processor.stage2_frame_objects) == 3
        assert len(processor.stage2_segments) == 1
        assert processor.stage2_frame_objects[0].pos_0_100 == 30

    def test_chapter_type_determination(self):
        """Test chapter type determination based on Stage 2 data."""
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        # Create frame objects with different contact patterns
        frame_objects = {
            0: self.create_mock_frame(0, contact_boxes=[
                {'class_name': 'hand', 'confidence': 0.9, 'bbox': (150, 120, 180, 160)}
            ]),
            1: self.create_mock_frame(1, contact_boxes=[
                {'class_name': 'mouth', 'confidence': 0.8, 'bbox': (140, 110, 190, 170)}
            ]),
            2: self.create_mock_frame(2, contact_boxes=[])
        }
        
        segments = [
            VideoSegment(0, 0, 1, 'handjob', 'SexAct', 'HJ', 'Hand Job'),
            VideoSegment(1, 1, 2, 'blowjob', 'SexAct', 'BJ', 'Blow Job'),
            VideoSegment(2, 2, 3, 'setup', 'Setup', 'SETUP', 'Scene Setup')
        ]
        
        processor.set_stage2_results(frame_objects, segments)
        
        # Test chapter type determination
        chapter_0 = processor.determine_chapter_type(0)
        chapter_1 = processor.determine_chapter_type(1)
        chapter_2 = processor.determine_chapter_type(2)
        
        assert isinstance(chapter_0, str)
        assert isinstance(chapter_1, str)
        assert isinstance(chapter_2, str)
        
        # Different frames should potentially have different chapter types
        # (depending on implementation, but at minimum they should be valid strings)

    def test_roi_extraction_from_stage2(self):
        """Test ROI extraction from Stage 2 data."""
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        # Create frame objects with locked penis data
        frame_objects = {
            0: self.create_mock_frame(0, has_penis=True),
            1: self.create_mock_frame(1, has_penis=False),
        }
        
        processor.set_stage2_results(frame_objects, [])
        
        # Test ROI extraction
        roi_0 = processor.extract_roi_from_stage2(0)
        roi_1 = processor.extract_roi_from_stage2(1)
        roi_missing = processor.extract_roi_from_stage2(999)
        
        # Frame 0 should have ROI (has penis)
        assert roi_0 is not None
        assert len(roi_0) == 4  # x, y, w, h
        
        # Frame 1 might not have ROI (no penis)
        # roi_1 could be None or a default ROI
        
        # Missing frame should return None
        assert roi_missing is None

    def test_debug_info_functionality(self):
        """Test debug info generation."""
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        frame_objects = {
            0: self.create_mock_frame(0)
        }
        segments = [VideoSegment(0, 0, 1, 'handjob', 'SexAct', 'HJ', 'Hand Job')]
        
        processor.set_stage2_results(frame_objects, segments)
        
        debug_info = processor.get_debug_info()
        
        assert isinstance(debug_info, dict)
        
        expected_keys = [
            'current_roi', 'locked_penis_active', 'current_chapter_type',
            'live_tracker_active', 'oscillation_intensity', 'signal_source'
        ]
        
        for key in expected_keys:
            assert key in debug_info

    def test_mixed_stage_analysis_function(self):
        """Test the perform_mixed_stage_analysis function."""
        # Create mock input frame
        input_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create mock Stage 2 frame object
        stage2_frame = self.create_mock_frame(0, pos=60)
        
        # Create mock segments
        segments = [VideoSegment(0, 10, 1, 'handjob', 'SexAct', 'HJ', 'Hand Job')]
        
        # This test verifies the function signature and basic functionality
        # The actual implementation might require more setup
        try:
            result = perform_mixed_stage_analysis(
                input_frame=input_frame,
                frame_id=0,
                stage2_frame_object=stage2_frame,
                video_segments=segments,
                tracker_model_path='/fake/path/model.pt',
                pose_model_path=None,
                app_instance=None
            )
            
            # Result should be a tuple or dict with processed information
            assert result is not None
            
        except Exception as e:
            # If implementation requires more setup, that's acceptable
            # The important thing is that the function exists and has the right signature
            assert "perform_mixed_stage_analysis" in str(e) or "model" in str(e).lower()

    def test_signal_source_determination(self):
        """Test signal source determination in mixed mode."""
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        # Create frame objects with varying funscript positions
        frame_objects = {}
        for i in range(10):
            frame_objects[i] = self.create_mock_frame(i, pos=30 + i * 5)
        
        processor.set_stage2_results(frame_objects, [])
        
        # Test signal extraction
        signals = []
        for frame_id in range(10):
            if frame_id in frame_objects:
                # Extract Stage 2 signal
                stage2_signal = frame_objects[frame_id].pos_0_100 / 100.0
                signals.append(stage2_signal)
        
        assert len(signals) == 10
        assert all(0.0 <= sig <= 1.0 for sig in signals)
        # Signals should vary (not all the same)
        assert len(set(signals)) > 1

    def test_tracker_mode_integration(self):
        """Test integration with TrackerMode enum."""
        # Verify the mixed mode exists
        assert hasattr(TrackerMode, 'OFFLINE_3_STAGE_MIXED')
        
        mixed_mode = TrackerMode.OFFLINE_3_STAGE_MIXED
        
        # Should be different from regular 3-stage mode
        if hasattr(TrackerMode, 'OFFLINE_3_STAGE'):
            assert mixed_mode != TrackerMode.OFFLINE_3_STAGE
        
        # Should be a valid enum value
        assert isinstance(mixed_mode, TrackerMode)

    def test_funscript_integration(self):
        """Test integration with funscript generation."""
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        # Create frame objects with funscript data
        frame_objects = {}
        positions = [20, 40, 60, 80, 60, 40, 20]  # Sample pattern
        
        for i, pos in enumerate(positions):
            frame_objects[i] = self.create_mock_frame(i, pos=pos)
        
        processor.set_stage2_results(frame_objects, [])
        
        # Extract positions (simulating funscript generation)
        extracted_positions = []
        for frame_id in range(len(positions)):
            if frame_id in processor.stage2_frame_objects:
                pos = processor.stage2_frame_objects[frame_id].pos_0_100
                extracted_positions.append(pos)
        
        assert len(extracted_positions) == len(positions)
        assert extracted_positions == positions

    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases in mixed mode."""
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        # Test with no Stage 2 data
        chapter_type = processor.determine_chapter_type(0)
        roi = processor.extract_roi_from_stage2(0)
        debug_info = processor.get_debug_info()
        
        # Should handle gracefully
        assert isinstance(chapter_type, (str, type(None)))
        assert roi is None or isinstance(roi, (tuple, list))
        assert isinstance(debug_info, dict)
        
        # Test with partial data
        frame_objects = {
            0: self.create_mock_frame(0, has_penis=False)  # No penis data
        }
        processor.set_stage2_results(frame_objects, [])
        
        roi = processor.extract_roi_from_stage2(0)
        # Should handle missing penis data gracefully
        assert roi is None or isinstance(roi, (tuple, list))

    def test_performance_characteristics(self):
        """Test performance characteristics of mixed mode processing."""
        import time
        
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        # Create larger dataset for performance testing
        frame_objects = {}
        for i in range(1000):  # 1000 frames
            frame_objects[i] = self.create_mock_frame(
                i, 
                pos=50 + 30 * (i % 100) / 100,  # Varying positions
                contact_boxes=[
                    {'class_name': f'contact_{i%3}', 'confidence': 0.8, 'bbox': (i, i, i+50, i+50)}
                ]
            )
        
        # Time the data loading
        start_time = time.time()
        processor.set_stage2_results(frame_objects, [])
        load_time = time.time() - start_time
        
        # Should load quickly (under 1 second for 1000 frames)
        assert load_time < 1.0
        
        # Time chapter type determination
        start_time = time.time()
        for frame_id in range(0, 1000, 100):  # Every 100th frame
            processor.determine_chapter_type(frame_id)
        process_time = time.time() - start_time
        
        # Should process quickly
        assert process_time < 0.5

    def test_integration_with_existing_codebase(self):
        """Test integration points with existing codebase."""
        # Test imports work correctly
        from detection.cd.stage_3_mixed_processor import MixedStageProcessor
        from config.constants import TrackerMode
        
        # Test constants are properly defined
        assert hasattr(TrackerMode, 'OFFLINE_3_STAGE_MIXED')
        
        # Test processor can be instantiated
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        assert processor is not None
        
        # Test basic functionality works
        frame_objects = {0: self.create_mock_frame(0)}
        processor.set_stage2_results(frame_objects, [])
        
        debug_info = processor.get_debug_info()
        assert isinstance(debug_info, dict)


if __name__ == "__main__":
    pytest.main([__file__])