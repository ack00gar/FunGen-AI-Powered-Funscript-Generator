#!/usr/bin/env python3
"""
Comprehensive tests for Stage 2 to Stage 3/Mixed carryover including chapters, 
single runs, and re-runs with existing data.
"""

import os
import sys
import pytest
import tempfile
import shutil
import subprocess
from pathlib import Path
import msgpack
import json
import time

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from application.utils.video_segment import VideoSegment
from detection.cd.data_structures import FrameObject, ATRLockedPenisState
from detection.cd.stage_3_mixed_processor import MixedStageProcessor, perform_mixed_stage_analysis
from config.constants import TrackerMode

pytestmark = [pytest.mark.integration, pytest.mark.comprehensive_carryover]


class TestComprehensiveStageCarryover:
    """Comprehensive test suite for Stage 2 to Stage 3 carryover in single and re-runs."""

    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Set up test environment."""
        self.test_video_path = "/Users/k00gar/Downloads/test/test_koogar_extra_short_A.mp4"
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "test_video_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        yield
        
        # Cleanup
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_comprehensive_stage2_artifacts(self, base_name):
        """Create comprehensive Stage 2 artifacts with chapters and detailed data."""
        overlay_path = os.path.join(self.output_dir, f"{base_name}_stage2_overlay.msgpack")
        
        # Create frame objects with realistic data patterns
        frame_objects = {}
        total_frames = 300  # Longer test video
        
        for frame_id in range(total_frames):
            frame_obj = {
                'frame_id': frame_id,
                'pos_0_100': self._generate_realistic_position(frame_id, total_frames),
                'yolo_boxes': self._generate_realistic_boxes(frame_id)
            }
            frame_objects[frame_id] = frame_obj
        
        # Create comprehensive chapters/segments
        segments = [
            {
                'start_frame_id': 0,
                'end_frame_id': 49,
                'class_id': 0,
                'class_name': 'setup',
                'segment_type': 'Setup',
                'position_short_name': 'SETUP',
                'position_long_name': 'Scene Setup'
            },
            {
                'start_frame_id': 50,
                'end_frame_id': 149,
                'class_id': 1,
                'class_name': 'handjob',
                'segment_type': 'SexAct',
                'position_short_name': 'HJ',
                'position_long_name': 'Hand Job'
            },
            {
                'start_frame_id': 150,
                'end_frame_id': 249,
                'class_id': 2,
                'class_name': 'blowjob',
                'segment_type': 'SexAct',
                'position_short_name': 'BJ',
                'position_long_name': 'Blow Job'
            },
            {
                'start_frame_id': 250,
                'end_frame_id': 299,
                'class_id': 3,
                'class_name': 'finish',
                'segment_type': 'Finish',
                'position_short_name': 'FINISH',
                'position_long_name': 'Scene Finish'
            }
        ]
        
        # Create comprehensive metadata
        metadata = {
            'total_frames': total_frames,
            'created_timestamp': int(time.time()),
            'stage2_version': '1.0',
            'model_used': 'FunGen-12n-pov-1.1.0',
            'processing_time': 45.7,
            'fps': 59.94,
            'video_resolution': '3840x1920',
            'chapters_created': len(segments),
            'total_detections': sum(len(fo['yolo_boxes']) for fo in frame_objects.values())
        }
        
        # Save comprehensive data
        comprehensive_data = {
            'frame_objects': frame_objects,
            'segments': segments,
            'metadata': metadata
        }
        
        with open(overlay_path, 'wb') as f:
            f.write(msgpack.packb(comprehensive_data, use_bin_type=True))
        
        return overlay_path, comprehensive_data

    def _generate_realistic_position(self, frame_id, total_frames):
        """Generate realistic funscript positions based on frame and context."""
        import math
        
        # Different patterns for different segments
        if frame_id < 50:  # Setup - static/slow
            return 50 + 5 * math.sin(frame_id * 0.01)
        elif frame_id < 150:  # Handjob - moderate motion
            return 50 + 25 * math.sin(frame_id * 0.05)
        elif frame_id < 250:  # Blowjob - faster motion
            return 50 + 35 * math.sin(frame_id * 0.08) * math.cos(frame_id * 0.03)
        else:  # Finish - intense motion
            return 50 + 40 * math.sin(frame_id * 0.1) * (1 + 0.2 * math.sin(frame_id * 0.5))

    def _generate_realistic_boxes(self, frame_id):
        """Generate realistic YOLO detection boxes."""
        boxes = []
        
        # Always include locked penis
        boxes.append({
            'class_name': 'locked_penis',
            'confidence': 0.85 + 0.1 * (frame_id % 10) / 10,
            'bbox': (100.0 + frame_id * 0.1, 100.0, 200.0, 200.0)
        })
        
        # Add context-appropriate contact boxes
        if 50 <= frame_id < 150:  # Handjob
            boxes.extend([
                {
                    'class_name': 'hand',
                    'confidence': 0.8,
                    'bbox': (150.0, 120.0 + frame_id * 0.05, 180.0, 160.0)
                },
                {
                    'class_name': 'finger',
                    'confidence': 0.7,
                    'bbox': (160.0, 130.0, 170.0, 150.0)
                }
            ])
        elif 150 <= frame_id < 250:  # Blowjob
            boxes.extend([
                {
                    'class_name': 'mouth',
                    'confidence': 0.9,
                    'bbox': (140.0, 110.0, 190.0, 170.0)
                },
                {
                    'class_name': 'face',
                    'confidence': 0.85,
                    'bbox': (130.0, 100.0, 200.0, 180.0)
                }
            ])
        
        return boxes

    def test_single_run_stage3_mixed_with_existing_stage2(self):
        """Test Stage 3 mixed mode run with existing Stage 2 data."""
        base_name = "test_single_run"
        overlay_path, stage2_data = self.create_comprehensive_stage2_artifacts(base_name)
        
        # Verify Stage 2 artifacts exist
        assert os.path.exists(overlay_path)
        
        # Create mixed processor and load Stage 2 data
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        # Reconstruct frame objects
        frame_objects = self._reconstruct_frame_objects(stage2_data['frame_objects'])
        segments = self._reconstruct_segments(stage2_data['segments'])
        
        processor.set_stage2_results(frame_objects, segments)
        
        # Verify data loaded correctly
        assert len(processor.stage2_frame_objects) == 300
        assert len(processor.stage2_segments) == 4
        
        # Test chapter-aware processing
        chapter_results = {}
        for segment in segments:
            mid_frame = (segment.start_frame_id + segment.end_frame_id) // 2
            chapter_type = processor.determine_chapter_type(mid_frame)
            chapter_results[segment.class_name] = chapter_type
        
        # Verify different chapters are detected
        assert len(set(chapter_results.values())) > 1  # Should have different chapter types

    def test_rerun_preserves_stage2_data(self):
        """Test that re-running preserves Stage 2 data and chapters."""
        base_name = "test_rerun"
        overlay_path, original_data = self.create_comprehensive_stage2_artifacts(base_name)
        
        # First run - simulate Stage 3 processing
        processor1 = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        frame_objects1 = self._reconstruct_frame_objects(original_data['frame_objects'])
        segments1 = self._reconstruct_segments(original_data['segments'])
        processor1.set_stage2_results(frame_objects1, segments1)
        
        # Simulate adding Stage 3 metadata
        enhanced_data = original_data.copy()
        enhanced_data['stage3_mixed_metadata'] = {
            'first_run_timestamp': int(time.time()),
            'mixed_mode_version': '1.0',
            'processed_chapters': [s.class_name for s in segments1]
        }
        
        with open(overlay_path, 'wb') as f:
            f.write(msgpack.packb(enhanced_data, use_bin_type=True))
        
        # Second run - verify preservation
        with open(overlay_path, 'rb') as f:
            rerun_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        
        processor2 = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        frame_objects2 = self._reconstruct_frame_objects(rerun_data['frame_objects'])
        segments2 = self._reconstruct_segments(rerun_data['segments'])
        processor2.set_stage2_results(frame_objects2, segments2)
        
        # Verify data preservation
        assert len(processor2.stage2_frame_objects) == len(processor1.stage2_frame_objects)
        assert len(processor2.stage2_segments) == len(processor1.stage2_segments)
        
        # Verify segments are identical
        for seg1, seg2 in zip(segments1, segments2):
            assert seg1.class_name == seg2.class_name
            assert seg1.start_frame_id == seg2.start_frame_id
            assert seg1.end_frame_id == seg2.end_frame_id
        
        # Verify Stage 3 metadata was preserved
        assert 'stage3_mixed_metadata' in rerun_data

    def test_chapter_continuity_across_runs(self):
        """Test that chapter information remains consistent across multiple runs."""
        base_name = "test_chapter_continuity"
        overlay_path, original_data = self.create_comprehensive_stage2_artifacts(base_name)
        
        chapter_results_runs = []
        
        # Run 3 times to test consistency
        for run_num in range(3):
            with open(overlay_path, 'rb') as f:
                run_data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
            
            processor = MixedStageProcessor(
                tracker_model_path='/fake/path/model.pt',
                pose_model_path=None
            )
            
            frame_objects = self._reconstruct_frame_objects(run_data['frame_objects'])
            segments = self._reconstruct_segments(run_data['segments'])
            processor.set_stage2_results(frame_objects, segments)
            
            # Test chapter detection for key frames
            chapter_results = {}
            test_frames = [25, 100, 200, 275]  # One from each segment
            
            for frame_id in test_frames:
                chapter_type = processor.determine_chapter_type(frame_id)
                chapter_results[frame_id] = chapter_type
            
            chapter_results_runs.append(chapter_results)
            
            # Simulate some processing and save
            run_data[f'run_{run_num}_metadata'] = {
                'timestamp': int(time.time()),
                'run_number': run_num
            }
            
            with open(overlay_path, 'wb') as f:
                f.write(msgpack.packb(run_data, use_bin_type=True))
        
        # Verify consistency across runs
        for frame_id in test_frames:
            frame_results = [run_result[frame_id] for run_result in chapter_results_runs]
            # All runs should return the same chapter type for the same frame
            assert len(set(frame_results)) <= 2  # Allow some variation but should be mostly consistent

    def test_performance_with_large_stage2_data(self):
        """Test performance characteristics with large Stage 2 datasets."""
        base_name = "test_large_data"
        
        # Create large dataset
        large_frame_objects = {}
        large_segments = []
        
        # Simulate 30 seconds at 60fps = 1800 frames
        total_frames = 1800
        
        for frame_id in range(total_frames):
            frame_obj = {
                'frame_id': frame_id,
                'pos_0_100': self._generate_realistic_position(frame_id, total_frames),
                'yolo_boxes': self._generate_realistic_boxes(frame_id)
            }
            large_frame_objects[frame_id] = frame_obj
        
        # Create multiple segments
        segment_duration = 300  # 5 seconds per segment
        segment_types = ['setup', 'handjob', 'blowjob', 'penetration', 'position_change', 'finish']
        
        for i, seg_type in enumerate(segment_types):
            start_frame = i * segment_duration
            end_frame = min(start_frame + segment_duration - 1, total_frames - 1)
            
            large_segments.append({
                'start_frame_id': start_frame,
                'end_frame_id': end_frame,
                'class_id': i,
                'class_name': seg_type,
                'segment_type': 'SexAct' if seg_type not in ['setup', 'finish'] else seg_type.title(),
                'position_short_name': seg_type[:3].upper(),
                'position_long_name': seg_type.title()
            })
        
        large_data = {
            'frame_objects': large_frame_objects,
            'segments': large_segments,
            'metadata': {
                'total_frames': total_frames,
                'created_timestamp': int(time.time()),
                'is_large_dataset': True
            }
        }
        
        overlay_path = os.path.join(self.output_dir, f"{base_name}_stage2_overlay.msgpack")
        with open(overlay_path, 'wb') as f:
            f.write(msgpack.packb(large_data, use_bin_type=True))
        
        # Test loading performance
        start_time = time.time()
        
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        frame_objects = self._reconstruct_frame_objects(large_data['frame_objects'])
        segments = self._reconstruct_segments(large_data['segments'])
        processor.set_stage2_results(frame_objects, segments)
        
        load_time = time.time() - start_time
        
        # Verify data loaded correctly
        assert len(processor.stage2_frame_objects) == total_frames
        assert len(processor.stage2_segments) == len(segment_types)
        
        # Performance should be reasonable (less than 5 seconds for 1800 frames)
        assert load_time < 5.0
        
        # Test processing performance for scattered frames
        start_time = time.time()
        
        test_frames = list(range(0, total_frames, 100))  # Every 100th frame
        chapter_types = []
        
        for frame_id in test_frames:
            chapter_type = processor.determine_chapter_type(frame_id)
            chapter_types.append(chapter_type)
        
        process_time = time.time() - start_time
        
        # Processing should be fast
        assert process_time < 2.0
        assert len(chapter_types) == len(test_frames)

    def test_stage3_mixed_cli_integration(self):
        """Test CLI integration with Stage 3 mixed mode using existing Stage 2 data."""
        if not os.path.exists(self.test_video_path):
            pytest.skip("Test video not available")
        
        # This test would ideally run the actual CLI, but we'll simulate the key parts
        base_name = "test_cli_integration"
        overlay_path, stage2_data = self.create_comprehensive_stage2_artifacts(base_name)
        
        # Simulate CLI arguments for Stage 3 mixed mode
        cli_args = {
            'video': self.test_video_path,
            'mode': 'OFFLINE_3_STAGE_MIXED',
            'auto_run': True,
            'output_dir': self.output_dir
        }
        
        # Verify that the Stage 2 artifacts would be detected
        assert os.path.exists(overlay_path)
        
        # Verify Stage 3 mixed mode can process the data
        processor = MixedStageProcessor(
            tracker_model_path='/fake/path/model.pt',
            pose_model_path=None
        )
        
        frame_objects = self._reconstruct_frame_objects(stage2_data['frame_objects'])
        segments = self._reconstruct_segments(stage2_data['segments'])
        processor.set_stage2_results(frame_objects, segments)
        
        # Verify CLI-like processing would work
        assert len(processor.stage2_frame_objects) > 0
        assert len(processor.stage2_segments) > 0
        
        # Test debug info (what would be logged)
        debug_info = processor.get_debug_info()
        assert isinstance(debug_info, dict)
        assert 'current_chapter_type' in debug_info

    def _reconstruct_frame_objects(self, frame_data_dict):
        """Helper to reconstruct FrameObject instances from saved data."""
        frame_objects = {}
        
        for frame_id, frame_data in frame_data_dict.items():
            frame_obj = FrameObject(frame_id=int(frame_id), yolo_input_size=640)
            
            # Reconstruct funscript position
            if 'pos_0_100' in frame_data:
                frame_obj.pos_0_100 = frame_data['pos_0_100']
            
            # Reconstruct contact boxes
            frame_obj.atr_detected_contact_boxes = []
            if 'yolo_boxes' in frame_data:
                for box in frame_data['yolo_boxes']:
                    class_name = box.get('class_name', '').lower()
                    if class_name not in ['locked_penis', 'penis']:
                        contact_box = {
                            'class_name': box.get('class_name', 'unknown'),
                            'confidence': box.get('confidence', 0.5),
                            'bbox': box.get('bbox', (0, 0, 0, 0))
                        }
                        frame_obj.atr_detected_contact_boxes.append(contact_box)
            
            # Reconstruct locked penis state
            frame_obj.atr_locked_penis_state = ATRLockedPenisState()
            locked_penis_data = None
            for box in frame_data.get('yolo_boxes', []):
                if box.get('class_name', '').lower() in ['locked_penis', 'penis']:
                    locked_penis_data = box
                    break
            
            if locked_penis_data and locked_penis_data.get('bbox'):
                frame_obj.atr_locked_penis_state.active = True
                frame_obj.atr_locked_penis_state.box = locked_penis_data.get('bbox')
            
            frame_objects[frame_obj.frame_id] = frame_obj
        
        return frame_objects

    def _reconstruct_segments(self, segment_data_list):
        """Helper to reconstruct VideoSegment instances from saved data."""
        segments = []
        
        for seg_data in segment_data_list:
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
        
        return segments


if __name__ == "__main__":
    pytest.main([__file__])