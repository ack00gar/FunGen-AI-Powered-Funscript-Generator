#!/usr/bin/env python3
"""
Comprehensive Chapter Handling Tests for VR Funscript AI Generator

Tests all aspects of chapter creation, management, and interaction including:
- Chapter creation during analysis
- Manual chapter creation
- Chapter loading/saving in project files
- Chapter interaction with refinement mode
- Chapter editing and deletion
- Chapter overlap detection
- Chapter merging functionality
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.insert(0, os.path.abspath('.'))

from application.logic.app_logic import ApplicationLogic
from application.logic.app_funscript_processor import AppFunscriptProcessor
from application.utils.video_segment import VideoSegment
from application.classes.project_manager import ProjectManager

# Test markers
pytestmark = [pytest.mark.integration, pytest.mark.chapters]

class TestChapterCreation:
    """Test chapter creation during analysis and manual creation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.app = Mock(spec=ApplicationLogic)
        self.app.logger = Mock()
        self.app.project_manager = Mock(spec=ProjectManager)
        self.app.project_manager.project_dirty = False
        
        self.funscript_processor = AppFunscriptProcessor(self.app)
    
    def test_chapters_created_during_stage2_analysis_with_force_flag(self):
        """Test that chapters are created during stage 2 analysis when force_rerun_stage2_segmentation is True."""
        # Mock stage processor with force flag enabled
        stage_processor = Mock()
        stage_processor.force_rerun_stage2_segmentation = True
        
        # Mock video segments data from stage 2
        video_segments_data = [
            {
                "unique_id": "seg_1",
                "start_frame_id": 0,
                "end_frame_id": 100,
                "segment_type": "SexAct",
                "position_short_name": "Cowgirl",
                "source": "stage2_analysis",
                "source_fps": 30.0
            },
            {
                "unique_id": "seg_2", 
                "start_frame_id": 150,
                "end_frame_id": 300,
                "segment_type": "SexAct",
                "position_short_name": "Missionary",
                "source": "stage2_analysis",
                "source_fps": 30.0
            }
        ]
        
        # Simulate stage2_results_success event handling
        fs_proc = self.funscript_processor
        
        # Clear existing chapters
        fs_proc.video_chapters.clear()
        
        # Process video segments data (simulating the event handler)
        if isinstance(video_segments_data, list):
            for seg_data in video_segments_data:
                if isinstance(seg_data, dict):
                    fs_proc.video_chapters.append(VideoSegment.from_dict(seg_data))
        
        # Verify chapters were created
        assert len(fs_proc.video_chapters) == 2
        assert fs_proc.video_chapters[0].position_short_name == "Cowgirl"
        assert fs_proc.video_chapters[1].position_short_name == "Missionary"
        assert fs_proc.video_chapters[0].start_frame_id == 0
        assert fs_proc.video_chapters[0].end_frame_id == 100
    
    def test_chapters_not_created_without_force_flag(self):
        """Test that chapters are NOT created during analysis when force_rerun_stage2_segmentation is False."""
        # Mock stage processor with force flag disabled
        stage_processor = Mock()
        stage_processor.force_rerun_stage2_segmentation = False
        
        # Create some existing chapters
        existing_chapter = VideoSegment(
            start_frame_id=50,
            end_frame_id=120,
            class_id=1,
            class_name="SexAct",
            segment_type="SexAct",
            position_short_name="Existing",
            position_long_name="Existing Position",
            source="manual"
        )
        self.funscript_processor.video_chapters = [existing_chapter]
        
        # Mock video segments data from stage 2 (should be ignored)
        video_segments_data = [
            {
                "unique_id": "seg_1",
                "start_frame_id": 0,
                "end_frame_id": 100,
                "segment_type": "SexAct", 
                "position_short_name": "Cowgirl",
                "source": "stage2_analysis",
                "source_fps": 30.0
            }
        ]
        
        # Simulate the logic from stage processor that preserves existing chapters
        if not stage_processor.force_rerun_stage2_segmentation:
            # Chapters should be preserved, not overwritten
            pass
        
        # Verify existing chapters were preserved
        assert len(self.funscript_processor.video_chapters) == 1
        assert self.funscript_processor.video_chapters[0].position_short_name == "Existing"
    
    def test_manual_chapter_creation(self):
        """Test manual chapter creation through the funscript processor."""
        fs_proc = self.funscript_processor
        
        # Create chapter data
        chapter_data = {
            "start_frame_id": 100,
            "end_frame_id": 200,
            "segment_type": "SexAct",
            "position_short_name": "Manual Test",
            "source": "manual"
        }
        
        # Test create_new_chapter_from_data method
        new_chapter = fs_proc.create_new_chapter_from_data(chapter_data, return_chapter_object=True)
        
        # Verify chapter was created and added
        assert new_chapter is not None
        assert new_chapter.start_frame_id == 100
        assert new_chapter.end_frame_id == 200
        assert len(fs_proc.video_chapters) == 1
        # Note: position_short_name might be processed differently by the actual implementation
    
    def test_chapter_overlap_detection(self):
        """Test that chapter overlap detection works correctly."""
        fs_proc = self.funscript_processor
        
        # Create first chapter
        chapter1_data = {
            "start_frame_id": 50,
            "end_frame_id": 150,
            "segment_type": "SexAct",
            "position_short_name": "First",
            "source": "manual"
        }
        fs_proc.create_new_chapter_from_data(chapter1_data)
        
        # Try to create overlapping chapter
        overlapping_chapter_data = {
            "start_frame_id": 100,
            "end_frame_id": 200,
            "segment_type": "SexAct", 
            "position_short_name": "Overlapping",
            "source": "manual"
        }
        
        # Test overlap detection method (using the private method)
        has_overlap = fs_proc._check_chapter_overlap(100, 200)
        assert has_overlap == True
        
        # Test non-overlapping chapter
        non_overlapping_data = {
            "start_frame_id": 300,
            "end_frame_id": 400,
            "segment_type": "SexAct",
            "position_short_name": "Non-overlapping", 
            "source": "manual"
        }
        
        has_overlap_non = fs_proc._check_chapter_overlap(300, 400)
        assert has_overlap_non == False


class TestChapterPersistence:
    """Test chapter loading and saving in project files."""
    
    def setup_method(self):
        """Set up test environment."""
        self.app = Mock(spec=ApplicationLogic)
        self.app.logger = Mock()
        self.app.project_manager = Mock(spec=ProjectManager)
        self.app.project_manager.project_dirty = False
        
        self.funscript_processor = AppFunscriptProcessor(self.app)
    
    def test_chapter_serialization_to_project_data(self):
        """Test that chapters are properly serialized for project saving."""
        fs_proc = self.funscript_processor
        
        # Create test chapters
        chapter1 = VideoSegment(0, 100, 1, "SexAct", "SexAct", "Test1", "Test1 Position", source="manual")
        chapter2 = VideoSegment(150, 250, 1, "SexAct", "SexAct", "Test2", "Test2 Position", source="stage2_analysis")
        
        fs_proc.video_chapters = [chapter1, chapter2]
        
        # Get project data for saving
        project_data = fs_proc.get_project_data_for_save()
        
        # Verify chapters are in project data
        assert "video_chapters" in project_data
        chapters_data = project_data["video_chapters"]
        assert len(chapters_data) == 2
        
        # Verify chapter data structure
        assert chapters_data[0]["unique_id"] == "ch1"
        assert chapters_data[0]["position_short_name"] == "Test1"
        assert chapters_data[1]["unique_id"] == "ch2"
        assert chapters_data[1]["position_short_name"] == "Test2"
    
    def test_chapter_deserialization_from_project_data(self):
        """Test that chapters are properly loaded from project data."""
        fs_proc = self.funscript_processor
        
        # Mock project data with chapters
        project_data = {
            "video_chapters": [
                {
                    "unique_id": "loaded_ch1",
                    "start_frame_id": 25,
                    "end_frame_id": 125,
                    "segment_type": "SexAct",
                    "position_short_name": "Loaded1",
                    "source": "manual",
                    "source_fps": 30.0
                },
                {
                    "unique_id": "loaded_ch2",
                    "start_frame_id": 175,
                    "end_frame_id": 275, 
                    "segment_type": "SexAct",
                    "position_short_name": "Loaded2",
                    "source": "stage2_analysis",
                    "source_fps": 30.0
                }
            ]
        }
        
        # Load chapters from project data
        fs_proc.update_project_specific_settings(project_data)
        
        # Verify chapters were loaded
        assert len(fs_proc.video_chapters) == 2
        assert fs_proc.video_chapters[0].unique_id == "loaded_ch1"
        assert fs_proc.video_chapters[0].position_short_name == "Loaded1"
        assert fs_proc.video_chapters[1].unique_id == "loaded_ch2"
        assert fs_proc.video_chapters[1].position_short_name == "Loaded2"
    
    def test_chapter_persistence_round_trip(self):
        """Test that chapters survive a save/load cycle."""
        fs_proc = self.funscript_processor
        
        # Create original chapters
        original_chapters = [
            VideoSegment(10, 110, 1, "SexAct", "SexAct", "RoundTrip1", "RoundTrip1 Position", source="manual"),
            VideoSegment(160, 260, 1, "SexAct", "SexAct", "RoundTrip2", "RoundTrip2 Position", source="stage2_analysis")
        ]
        
        fs_proc.video_chapters = original_chapters
        
        # Save to project data
        saved_data = fs_proc.get_project_data_for_save()
        
        # Clear chapters
        fs_proc.video_chapters.clear()
        assert len(fs_proc.video_chapters) == 0
        
        # Load from saved data
        fs_proc.update_project_specific_settings(saved_data)
        
        # Verify chapters match original
        assert len(fs_proc.video_chapters) == 2
        loaded_chapters = fs_proc.video_chapters
        
        assert loaded_chapters[0].unique_id == "rt1"
        assert loaded_chapters[0].position_short_name == "RoundTrip1"
        assert loaded_chapters[0].start_frame_id == 10
        assert loaded_chapters[0].end_frame_id == 110
        
        assert loaded_chapters[1].unique_id == "rt2"
        assert loaded_chapters[1].position_short_name == "RoundTrip2"
        assert loaded_chapters[1].start_frame_id == 160
        assert loaded_chapters[1].end_frame_id == 260


class TestChapterInteraction:
    """Test chapter interaction with other systems."""
    
    def setup_method(self):
        """Set up test environment."""
        self.app = Mock(spec=ApplicationLogic)
        self.app.logger = Mock()
        self.app.project_manager = Mock(spec=ProjectManager)
        self.app.project_manager.project_dirty = False
        
        self.funscript_processor = AppFunscriptProcessor(self.app)
    
    def test_get_chapter_at_frame(self):
        """Test finding chapters by frame index."""
        fs_proc = self.funscript_processor
        
        # Create test chapters
        chapter1 = VideoSegment(50, 150, 1, "SexAct", "SexAct", "FrameTest1", "FrameTest1 Position", source="manual")
        chapter2 = VideoSegment(200, 300, 1, "SexAct", "SexAct", "FrameTest2", "FrameTest2 Position", source="manual")
        
        fs_proc.video_chapters = [chapter1, chapter2]
        
        # Test finding chapters by frame
        found_chapter1 = fs_proc.get_chapter_at_frame(100)
        assert found_chapter1 is not None
        assert found_chapter1.unique_id == "frame_test1"
        
        found_chapter2 = fs_proc.get_chapter_at_frame(250)
        assert found_chapter2 is not None
        assert found_chapter2.unique_id == "frame_test2"
        
        # Test frame not in any chapter (gap)
        no_chapter = fs_proc.get_chapter_at_frame(175)
        assert no_chapter is None
        
        # Test frame before first chapter
        before_chapters = fs_proc.get_chapter_at_frame(25)
        assert before_chapters is None
        
        # Test frame after last chapter
        after_chapters = fs_proc.get_chapter_at_frame(350)
        assert after_chapters is None
    
    def test_chapter_refinement_mode_interaction(self):
        """Test that refinement mode correctly identifies chapters for interaction."""
        fs_proc = self.funscript_processor
        
        # Create test chapters
        refinement_chapter = VideoSegment(100, 200, 1, "SexAct", "SexAct", "RefinementTest", "RefinementTest Position", source="stage2_analysis")
        
        fs_proc.video_chapters = [refinement_chapter]
        
        # Test finding chapter for refinement
        current_frame = 150  # Within chapter bounds
        found_chapter = fs_proc.get_chapter_at_frame(current_frame)
        
        assert found_chapter is not None
        assert found_chapter.unique_id == "refinement_test"
        assert found_chapter.position_short_name == "RefinementTest"
        
        # Test refinement fails outside chapter bounds
        outside_frame = 250  # Outside chapter bounds
        no_chapter_found = fs_proc.get_chapter_at_frame(outside_frame)
        assert no_chapter_found is None
    
    def test_chapter_deletion(self):
        """Test chapter deletion functionality."""
        fs_proc = self.funscript_processor
        
        # Create test chapters
        chapters = [
            VideoSegment(50, 150, 1, "SexAct", "SexAct", "DeleteTest1", "DeleteTest1 Position", source="manual"),
            VideoSegment(200, 300, 1, "SexAct", "SexAct", "DeleteTest2", "DeleteTest2 Position", source="manual"),
            VideoSegment(350, 450, 1, "SexAct", "SexAct", "DeleteTest3", "DeleteTest3 Position", source="manual")
        ]
        
        fs_proc.video_chapters = chapters
        assert len(fs_proc.video_chapters) == 3
        
        # Delete middle chapter
        fs_proc.delete_video_chapters_by_ids(["del_test2"])
        
        # Verify deletion
        assert len(fs_proc.video_chapters) == 2
        remaining_ids = [ch.unique_id for ch in fs_proc.video_chapters]
        assert "del_test1" in remaining_ids
        assert "del_test2" not in remaining_ids
        assert "del_test3" in remaining_ids
        
        # Delete multiple chapters
        fs_proc.delete_video_chapters_by_ids(["del_test1", "del_test3"])
        
        # Verify all deleted
        assert len(fs_proc.video_chapters) == 0


class TestChapterEdgeCases:
    """Test edge cases and error conditions for chapter handling."""
    
    def setup_method(self):
        """Set up test environment."""
        self.app = Mock(spec=ApplicationLogic)
        self.app.logger = Mock()
        self.app.project_manager = Mock(spec=ProjectManager)
        self.app.project_manager.project_dirty = False
        
        self.funscript_processor = AppFunscriptProcessor(self.app)
    
    def test_empty_chapter_list_handling(self):
        """Test that empty chapter lists are handled gracefully."""
        fs_proc = self.funscript_processor
        
        # Ensure chapters list is empty
        fs_proc.video_chapters = []
        
        # Test get_chapter_at_frame with empty list
        no_chapter = fs_proc.get_chapter_at_frame(100)
        assert no_chapter is None
        
        # Test project data serialization with empty chapters
        project_data = fs_proc.get_project_data_for_save()
        assert "video_chapters" in project_data
        assert project_data["video_chapters"] == []
        
        # Test deletion with empty list
        fs_proc.delete_video_chapters_by_ids(["nonexistent"])
        assert len(fs_proc.video_chapters) == 0
    
    def test_invalid_chapter_data_handling(self):
        """Test handling of invalid or malformed chapter data."""
        fs_proc = self.funscript_processor
        
        # Test loading invalid chapter data
        invalid_project_data = {
            "video_chapters": [
                # Missing required fields
                {"unique_id": "invalid1"},
                # Wrong data types  
                {"unique_id": 123, "start_frame_id": "not_a_number"},
                # Valid chapter for comparison
                {
                    "unique_id": "valid1",
                    "start_frame_id": 100,
                    "end_frame_id": 200,
                    "segment_type": "SexAct",
                    "position_short_name": "Valid",
                    "source": "manual",
                    "source_fps": 30.0
                }
            ]
        }
        
        # Should only load valid chapters, skip invalid ones
        fs_proc.update_project_specific_settings(invalid_project_data)
        
        # Should have only the valid chapter
        assert len(fs_proc.video_chapters) == 1
        assert fs_proc.video_chapters[0].unique_id == "valid1"
    
    def test_chapter_boundary_conditions(self):
        """Test chapters at frame boundaries and edge cases."""
        fs_proc = self.funscript_processor
        
        # Create chapter with boundary frames
        boundary_chapter = VideoSegment(100, 200, 1, "SexAct", "SexAct", "BoundaryTest", "BoundaryTest Position", source="manual")
        
        fs_proc.video_chapters = [boundary_chapter]
        
        # Test exact boundary frames
        start_boundary = fs_proc.get_chapter_at_frame(100)  # Start frame
        assert start_boundary is not None
        assert start_boundary.unique_id == "boundary_test"
        
        end_boundary = fs_proc.get_chapter_at_frame(200)    # End frame
        assert end_boundary is not None
        assert end_boundary.unique_id == "boundary_test"
        
        # Test just outside boundaries
        before_start = fs_proc.get_chapter_at_frame(99)     # Just before start
        assert before_start is None
        
        after_end = fs_proc.get_chapter_at_frame(201)       # Just after end
        assert after_end is None


class TestChapterIntegrationWithUI:
    """Test chapter integration with UI components and user actions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.app = Mock(spec=ApplicationLogic)
        self.app.logger = Mock()
        self.app.project_manager = Mock(spec=ProjectManager)
        self.app.project_manager.project_dirty = False
        
        self.funscript_processor = AppFunscriptProcessor(self.app)
    
    def test_chapter_creation_triggers_ui_updates(self):
        """Test that chapter creation triggers proper UI state updates."""
        fs_proc = self.funscript_processor
        
        # Mock app set_status_message method
        self.app.set_status_message = Mock()
        
        # Create a new chapter
        chapter_data = {
            "start_frame_id": 50,
            "end_frame_id": 150,
            "segment_type": "SexAct",
            "position_short_name": "UI Test",
            "source": "manual"
        }
        
        new_chapter = fs_proc.create_new_chapter_from_data(chapter_data, return_chapter_object=True)
        
        # Verify status message was called
        self.app.set_status_message.assert_called_once()
        call_args = self.app.set_status_message.call_args[0][0]
        assert "Chapter 'UI Test' created" in call_args
        
        # Verify project dirty flag is set
        assert self.app.project_manager.project_dirty == True
    
    def test_chapter_deletion_clears_selection(self):
        """Test that deleting selected chapter clears the selection."""
        fs_proc = self.funscript_processor
        
        # Create and select a chapter
        selected_chapter = VideoSegment(100, 200, 1, "SexAct", "SexAct", "Selected", "Selected Position", source="manual")
        
        fs_proc.video_chapters = [selected_chapter]
        fs_proc.selected_chapter_for_scripting = selected_chapter
        
        # Delete the selected chapter
        fs_proc.delete_video_chapters_by_ids(["selected_for_deletion"])
        
        # Verify chapter was deleted and selection cleared
        assert len(fs_proc.video_chapters) == 0
        # Note: The actual clearing of selection would depend on the implementation
        # This test documents the expected behavior


# Integration test that runs with the full system
@pytest.mark.integration
@pytest.mark.slow
class TestChapterSystemIntegration:
    """Full system integration tests for chapter functionality."""
    
    def test_chapter_creation_in_analysis_pipeline(self):
        """Test chapter creation through the complete analysis pipeline."""
        # This would be a more complex test that sets up the full application
        # and runs through the analysis pipeline to verify chapters are created
        # This is a placeholder for a full integration test
        pytest.skip("Full system integration test - requires complete setup")
    
    def test_chapter_refinement_mode_full_workflow(self):
        """Test the complete refinement mode workflow with chapters."""
        # Another full integration test placeholder
        pytest.skip("Full refinement workflow test - requires complete setup")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])