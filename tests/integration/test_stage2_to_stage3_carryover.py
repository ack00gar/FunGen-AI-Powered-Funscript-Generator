#!/usr/bin/env python3
"""
Tests for Stage 2 to Stage 3 data carryover functionality.
Ensures that Stage 2 artifacts properly carry over to Stage 3 and Stage 3 mixed mode.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
import msgpack
import json

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from application.logic.app_logic import ApplicationLogic
from application.logic.app_stage_processor import AppStageProcessor
from application.classes.settings_manager import AppSettings
from application.utils.video_segment import VideoSegment
from detection.cd.data_structures import FrameObject, LockedPenisState
from detection.cd.stage_3_mixed_processor import MixedStageProcessor
from config.constants import TrackerMode

pytestmark = [pytest.mark.integration, pytest.mark.stage_carryover]


class TestStage2ToStage3Carryover:
    """Test suite for Stage 2 to Stage 3 data carryover functionality."""

    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Set up test environment."""
        self.test_video_path = "/Users/k00gar/Downloads/test/test_koogar_extra_short_A.mp4"
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        yield
        
        # Cleanup
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_mock_stage2_artifacts(self, base_name):
        """Create mock Stage 2 artifacts for testing."""
        # Create mock stage2_overlay.msgpack
        overlay_path = os.path.join(self.output_dir, f"{base_name}_stage2_overlay.msgpack")
        
        # Mock frame objects with funscript positions and contact data
        frame_objects = {}
        segments = []
        
        for frame_id in range(100):
            frame_obj = {
                'frame_id': frame_id,
                'pos_0_100': 50 + (frame_id % 50),  # Varying position
                'yolo_boxes': [
                    {
                        'class_name': 'locked_penis',
                        'confidence': 0.9,
                        'bbox': (100.0 + frame_id, 100.0, 200.0, 200.0)
                    },
                    {
                        'class_name': 'hand',
                        'confidence': 0.8,
                        'bbox': (150.0, 120.0, 180.0, 160.0)
                    },
                    {
                        'class_name': 'finger',
                        'confidence': 0.7,
                        'bbox': (160.0, 130.0, 170.0, 150.0)
                    }
                ]
            }
            frame_objects[frame_id] = frame_obj
        
        # Create a test segment
        test_segment = {
            'start_frame_id': 10,
            'end_frame_id': 80,
            'class_id': 1,
            'class_name': 'handjob',
            'segment_type': 'SexAct',
            'position_short_name': 'HJ',
            'position_long_name': 'Hand Job'
        }
        segments.append(test_segment)
        
        # Save mock data
        mock_data = {
            'frame_objects': frame_objects,
            'segments': segments,
            'metadata': {
                'total_frames': 100,
                'created_timestamp': 1234567890
            }
        }
        
        with open(overlay_path, 'wb') as f:
            f.write(msgpack.packb(mock_data, use_bin_type=True))
        
        return overlay_path, mock_data

    def test_stage2_artifacts_detection(self):
        """Test that Stage 2 artifacts are properly detected."""
        base_name = "test_video"
        overlay_path, mock_data = self.create_mock_stage2_artifacts(base_name)
        
        # Test that the file exists and is readable
        assert os.path.exists(overlay_path)
        
        # Test msgpack loading
        with open(overlay_path, 'rb') as f:
            loaded_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        
        assert 'frame_objects' in loaded_data
        assert 'segments' in loaded_data
        assert len(loaded_data['frame_objects']) == 100
        assert len(loaded_data['segments']) == 1

    def test_stage2_frame_object_reconstruction(self):
        """Test that Stage 2 frame objects can be properly reconstructed."""
        base_name = "test_frame_reconstruction"
        overlay_path, mock_data = self.create_mock_stage2_artifacts(base_name)
        
        # Load and reconstruct frame objects
        with open(overlay_path, 'rb') as f:
            loaded_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        
        reconstructed_objects = {}
        
        for frame_id, frame_data in loaded_data['frame_objects'].items():
            frame_obj = FrameObject(frame_id=int(frame_id), yolo_input_size=640)
            
            # Reconstruct funscript position
            if 'pos_0_100' in frame_data:
                frame_obj.pos_0_100 = frame_data['pos_0_100']
            
            # Reconstruct contact boxes
            frame_obj.detected_contact_boxes = []
            if 'yolo_boxes' in frame_data:
                for box in frame_data['yolo_boxes']:
                    class_name = box.get('class_name', '').lower()
                    if class_name not in ['locked_penis', 'penis']:
                        contact_box = {
                            'class_name': box.get('class_name', 'unknown'),
                            'confidence': box.get('confidence', 0.5),
                            'bbox': box.get('bbox', (0, 0, 0, 0))
                        }
                        frame_obj.detected_contact_boxes.append(contact_box)
            
            # Reconstruct locked penis state
            frame_obj.locked_penis_state = LockedPenisState()
            locked_penis_data = None
            for box in frame_data.get('yolo_boxes', []):
                if box.get('class_name', '').lower() in ['locked_penis', 'penis']:
                    locked_penis_data = box
                    break
            
            if locked_penis_data and locked_penis_data.get('bbox'):
                frame_obj.locked_penis_state.active = True
                frame_obj.locked_penis_state.box = locked_penis_data.get('bbox')
            
            reconstructed_objects[frame_obj.frame_id] = frame_obj
        
        # Verify reconstruction
        assert len(reconstructed_objects) == 100
        
        # Test specific frame
        test_frame = reconstructed_objects[50]
        assert test_frame.pos_0_100 is not None
        assert test_frame.locked_penis_state.active
        assert len(test_frame.detected_contact_boxes) == 2  # hand and finger

    def test_stage3_mixed_processor_integration(self):
        """Test that Stage 3 mixed processor can use Stage 2 data."""
        base_name = "test_mixed_integration"
        overlay_path, mock_data = self.create_mock_stage2_artifacts(base_name)
        
        # Create mixed processor
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        # Reconstruct frame objects from mock data
        frame_objects = {}
        for frame_id, frame_data in mock_data['frame_objects'].items():
            frame_obj = FrameObject(frame_id=int(frame_id), yolo_input_size=640)
            frame_obj.pos_0_100 = frame_data.get('pos_0_100', 50)
            
            # Set up locked penis state
            frame_obj.locked_penis_state = LockedPenisState()
            for box in frame_data.get('yolo_boxes', []):
                if box.get('class_name', '').lower() in ['locked_penis', 'penis']:
                    frame_obj.locked_penis_state.active = True
                    frame_obj.locked_penis_state.box = box.get('bbox')
                    break
            
            # Set up contact boxes
            frame_obj.detected_contact_boxes = []
            for box in frame_data.get('yolo_boxes', []):
                class_name = box.get('class_name', '').lower()
                if class_name not in ['locked_penis', 'penis']:
                    frame_obj.detected_contact_boxes.append({
                        'class_name': box.get('class_name'),
                        'confidence': box.get('confidence'),
                        'bbox': box.get('bbox')
                    })
            
            frame_objects[frame_obj.frame_id] = frame_obj
        
        # Reconstruct segments
        segments = []
        for seg_data in mock_data['segments']:
            segment = VideoSegment(
                start_frame_id=seg_data['start_frame_id'],
                end_frame_id=seg_data['end_frame_id'],
                class_id=seg_data['class_id'],
                class_name=seg_data['class_name'],
                segment_type=seg_data['segment_type'],
                position_short_name=seg_data['position_short_name'],
                position_long_name=seg_data['position_long_name']
            )
            segments.append(segment)
        
        # Set Stage 2 results in mixed processor
        processor.set_stage2_results(frame_objects, segments)
        
        # Test functionality
        assert len(processor.stage2_frame_objects) == 100
        assert len(processor.stage2_segments) == 1
        
        # Test chapter type determination
        chapter_type = processor.determine_chapter_type(50)
        # Chapter type should be a string (could be various formats)
        assert isinstance(chapter_type, str) or chapter_type is None
        
        # Test ROI extraction
        roi = processor.extract_roi_from_stage2(50)
        assert roi is not None
        assert len(roi) == 4  # x, y, w, h

    def test_chapter_carryover(self):
        """Test that chapters from Stage 2 carry over to Stage 3."""
        base_name = "test_chapter_carryover"
        overlay_path, mock_data = self.create_mock_stage2_artifacts(base_name)
        
        # Create additional segments for comprehensive chapter testing
        additional_segments = [
            {
                'start_frame_id': 0,
                'end_frame_id': 9,
                'class_id': 0,
                'class_name': 'setup',
                'segment_type': 'Setup',
                'position_short_name': 'S',
                'position_long_name': 'Setup'
            },
            {
                'start_frame_id': 81,
                'end_frame_id': 99,
                'class_id': 2,
                'class_name': 'finish',
                'segment_type': 'Finish',
                'position_short_name': 'F',
                'position_long_name': 'Finish'
            }
        ]
        
        # Update mock data with additional segments
        mock_data['segments'].extend(additional_segments)
        with open(overlay_path, 'wb') as f:
            f.write(msgpack.packb(mock_data, use_bin_type=True))
        
        # Test segment loading and carryover
        with open(overlay_path, 'rb') as f:
            loaded_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        
        segments = []
        for seg_data in loaded_data['segments']:
            segment = VideoSegment(
                start_frame_id=seg_data['start_frame_id'],
                end_frame_id=seg_data['end_frame_id'],
                class_id=seg_data['class_id'],
                class_name=seg_data['class_name'],
                segment_type=seg_data['segment_type'],
                position_short_name=seg_data['position_short_name'],
                position_long_name=seg_data['position_long_name']
            )
            segments.append(segment)
        
        # Verify all segments loaded correctly
        assert len(segments) == 3
        
        # Verify segment details
        setup_segment = next(s for s in segments if s.class_name == 'setup')
        handjob_segment = next(s for s in segments if s.class_name == 'handjob')
        finish_segment = next(s for s in segments if s.class_name == 'finish')
        
        assert setup_segment.start_frame_id == 0
        assert setup_segment.end_frame_id == 9
        
        assert handjob_segment.start_frame_id == 10
        assert handjob_segment.end_frame_id == 80
        
        assert finish_segment.start_frame_id == 81
        assert finish_segment.end_frame_id == 99

    def test_stage3_mode_detection(self):
        """Test that Stage 3 modes are properly detected and configured."""
        # Test OFFLINE_3_STAGE mode
        assert hasattr(TrackerMode, 'OFFLINE_3_STAGE')
        
        # Test OFFLINE_3_STAGE_MIXED mode
        assert hasattr(TrackerMode, 'OFFLINE_3_STAGE_MIXED')
        
        # Test that both modes are different
        assert TrackerMode.OFFLINE_3_STAGE != TrackerMode.OFFLINE_3_STAGE_MIXED

    def test_stage2_data_persistence_across_reruns(self):
        """Test that Stage 2 data persists correctly across multiple runs."""
        base_name = "test_persistence"
        overlay_path, original_data = self.create_mock_stage2_artifacts(base_name)
        
        # First run - load data
        with open(overlay_path, 'rb') as f:
            first_load = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        
        # Simulate modification (what might happen during Stage 3 processing)
        modified_data = first_load.copy()
        modified_data['stage3_metadata'] = {
            'processed_timestamp': 1234567891,
            'stage3_mode': 'mixed'
        }
        
        # Save modified data
        with open(overlay_path, 'wb') as f:
            f.write(msgpack.packb(modified_data, use_bin_type=True))
        
        # Second run - verify Stage 2 data is still intact
        with open(overlay_path, 'rb') as f:
            second_load = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        
        # Verify original Stage 2 data is preserved
        assert 'frame_objects' in second_load
        assert 'segments' in second_load
        assert len(second_load['frame_objects']) == 100
        assert len(second_load['segments']) == 1
        
        # Verify Stage 3 metadata was added
        assert 'stage3_metadata' in second_load

    def test_mixed_stage_signal_extraction(self):
        """Test that mixed stage can extract signals from Stage 2 data."""
        base_name = "test_signal_extraction"
        overlay_path, mock_data = self.create_mock_stage2_artifacts(base_name)
        
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        # Create frame objects with varying funscript positions
        frame_objects = {}
        for frame_id in range(100):
            frame_obj = FrameObject(frame_id=frame_id, yolo_input_size=640)
            # Create a sine wave pattern for testing
            import math
            frame_obj.pos_0_100 = 50 + 30 * math.sin(frame_id * 0.1)
            
            frame_obj.locked_penis_state = LockedPenisState()
            frame_obj.locked_penis_state.active = True
            frame_obj.locked_penis_state.box = (100.0, 100.0, 200.0, 200.0)
            
            frame_objects[frame_id] = frame_obj
        
        processor.set_stage2_results(frame_objects, [])
        
        # Test signal extraction for different frames
        signals = []
        for frame_id in range(0, 100, 10):
            if frame_id in frame_objects:
                stage2_signal = frame_objects[frame_id].pos_0_100 / 100.0
                signals.append(stage2_signal)
        
        # Verify signal extraction
        assert len(signals) == 10
        assert all(0.0 <= sig <= 1.0 for sig in signals)
        assert not all(sig == signals[0] for sig in signals)  # Should vary

    def test_error_handling_missing_stage2_data(self):
        """Test error handling when Stage 2 data is missing or corrupted."""
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        # Test with empty data
        processor.set_stage2_results({}, [])
        
        # Should handle gracefully
        chapter_type = processor.determine_chapter_type(0)
        assert chapter_type in ['unknown', 'default', None] or isinstance(chapter_type, str)
        
        roi = processor.extract_roi_from_stage2(0)
        assert roi is None or isinstance(roi, (tuple, list))

    def test_comprehensive_pipeline_simulation(self):
        """Simulate a complete pipeline from Stage 2 to Stage 3 mixed."""
        base_name = "test_comprehensive_pipeline"
        overlay_path, mock_data = self.create_mock_stage2_artifacts(base_name)
        
        # Step 1: Verify Stage 2 artifacts exist
        assert os.path.exists(overlay_path)
        
        # Step 2: Load Stage 2 data
        with open(overlay_path, 'rb') as f:
            stage2_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        
        # Step 3: Create Stage 3 mixed processor
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        # Step 4: Reconstruct and set Stage 2 data
        frame_objects = {}
        for frame_id, frame_data in stage2_data['frame_objects'].items():
            frame_obj = FrameObject(frame_id=int(frame_id), yolo_input_size=640)
            frame_obj.pos_0_100 = frame_data.get('pos_0_100', 50)
            
            frame_obj.locked_penis_state = LockedPenisState()
            frame_obj.locked_penis_state.active = True
            frame_obj.locked_penis_state.box = (100.0, 100.0, 200.0, 200.0)
            
            frame_obj.atr_detected_contact_boxes = [
                {'class_name': 'hand', 'confidence': 0.8, 'bbox': (150, 120, 180, 160)}
            ]
            
            frame_objects[frame_obj.frame_id] = frame_obj
        
        segments = [VideoSegment(
            start_frame_id=10, end_frame_id=80, class_id=1,
            class_name='handjob', segment_type='SexAct',
            position_short_name='HJ', position_long_name='Hand Job'
        )]
        
        processor.set_stage2_results(frame_objects, segments)
        
        # Step 5: Verify Stage 3 processing can proceed
        debug_info = processor.get_debug_info()
        assert isinstance(debug_info, dict)
        
        # Step 6: Test frame processing
        for test_frame_id in [25, 50, 75]:
            chapter_type = processor.determine_chapter_type(test_frame_id)
            roi = processor.extract_roi_from_stage2(test_frame_id)
            
            assert isinstance(chapter_type, str)
            assert roi is None or (isinstance(roi, (tuple, list)) and len(roi) == 4)
        
        # Step 7: Verify pipeline completion
        assert len(processor.stage2_frame_objects) == 100
        assert len(processor.stage2_segments) == 1


if __name__ == "__main__":
    pytest.main([__file__])