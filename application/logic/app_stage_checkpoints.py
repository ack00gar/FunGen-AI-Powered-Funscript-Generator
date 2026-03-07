"""Checkpoint management methods for the processing pipeline.

Extracted from AppStageProcessor to reduce file size.
Contains checkpoint creation, resumption, and cleanup logic.

Used as a mixin — all methods operate on the AppStageProcessor instance.
"""
import os
from typing import Optional, List, Dict, Any, Tuple

from application.utils.checkpoint_manager import (
    ProcessingStage, CheckpointData,
)


class StageCheckpointMixin:
    """Mixin with methods that manage processing checkpoints."""

    def check_resumable_tasks(self) -> List[Tuple[str, CheckpointData]]:
        """Check for tasks that can be resumed from checkpoints."""
        return self.checkpoint_manager.get_resumable_tasks()

    def can_resume_video(self, video_path: str) -> Optional[CheckpointData]:
        """Check if a specific video has resumable progress."""
        return self.checkpoint_manager.find_latest_checkpoint(video_path)

    def start_resume_from_checkpoint(self, checkpoint_data: CheckpointData) -> bool:
        """Resume processing from a checkpoint."""
        try:
            if self.full_analysis_active:
                self.logger.warning("Cannot resume: Analysis already running.", extra={'status_message': True})
                return False

            if not os.path.exists(checkpoint_data.video_path):
                self.logger.error(f"Cannot resume: Video file not found: {checkpoint_data.video_path}",
                                extra={'status_message': True})
                return False

            self.resume_data = checkpoint_data
            self.logger.info(f"Resuming {checkpoint_data.processing_stage.value} from {checkpoint_data.progress_percentage:.1f}%",
                           extra={'status_message': True})

            # Load the video first
            if self.app.file_manager.video_path != checkpoint_data.video_path:
                # Would need to trigger video loading through the app
                # For now, assume video is already loaded
                pass

            # Resume based on the stage
            if checkpoint_data.processing_stage == ProcessingStage.STAGE_1_OBJECT_DETECTION:
                return self._resume_stage1(checkpoint_data)
            elif checkpoint_data.processing_stage == ProcessingStage.STAGE_2_OPTICAL_FLOW:
                return self._resume_stage2(checkpoint_data)
            elif checkpoint_data.processing_stage == ProcessingStage.STAGE_3_FUNSCRIPT_GENERATION:
                return self._resume_stage3(checkpoint_data)
            else:
                self.logger.error(f"Cannot resume: Unknown stage {checkpoint_data.processing_stage}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to resume from checkpoint: {e}", exc_info=True)
            return False

    def _resume_stage1(self, checkpoint_data: CheckpointData) -> bool:
        """Resume Stage 1 processing from checkpoint."""
        settings = checkpoint_data.processing_settings
        self.logger.info("Stage 1 resume: Restarting with original settings")
        processing_mode = settings.get('processing_mode', 'OFFLINE_GUIDED_FLOW')
        return self._start_full_analysis_with_settings(processing_mode, settings)

    def _resume_stage2(self, checkpoint_data: CheckpointData) -> bool:
        """Resume Stage 2 processing from checkpoint."""
        settings = checkpoint_data.processing_settings
        self.logger.info(f"Stage 2 resume: Continuing from {checkpoint_data.progress_percentage:.1f}%")
        processing_mode = settings.get('processing_mode', 'OFFLINE_GUIDED_FLOW')
        return self._start_full_analysis_with_settings(processing_mode, settings)

    def _resume_stage3(self, checkpoint_data: CheckpointData) -> bool:
        """Resume Stage 3 processing from checkpoint."""
        settings = checkpoint_data.processing_settings
        stage_data = checkpoint_data.stage_data
        self.logger.info(f"Stage 3 resume: Continuing from segment {stage_data.get('current_segment', 0)}")
        processing_mode = settings.get('processing_mode', 'OFFLINE_GUIDED_FLOW')
        return self._start_full_analysis_with_settings(processing_mode, settings)

    def _start_full_analysis_with_settings(self, processing_mode: str, settings: Dict[str, Any]) -> bool:
        """Start full analysis with restored settings from checkpoint."""
        try:
            override_producers = settings.get('num_producers_override')
            override_consumers = settings.get('num_consumers_override')
            frame_range_override = settings.get('frame_range_override')

            self.start_full_analysis(
                processing_mode=processing_mode,
                override_producers=override_producers,
                override_consumers=override_consumers,
                frame_range_override=frame_range_override,
                is_autotune_run=settings.get('is_autotune_run', False)
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to start analysis with restored settings: {e}")
            return False

    def delete_checkpoint_for_video(self, video_path: str) -> bool:
        """Delete all checkpoints for a specific video."""
        try:
            count = self.checkpoint_manager.delete_video_checkpoints(video_path)
            if count > 0:
                self.logger.info(f"Deleted {count} checkpoints for video", extra={'status_message': True})
            return count > 0
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoints: {e}")
            return False

    def _create_checkpoint_if_needed(self, stage: ProcessingStage, frame_index: int,
                                   total_frames: int, stage_data: Dict[str, Any]) -> None:
        """Create a checkpoint if enough time has passed."""
        if not self.app.file_manager.video_path:
            return

        if not self.checkpoint_manager.should_create_checkpoint(self.app.file_manager.video_path):
            return

        try:
            progress_percentage = (frame_index / total_frames * 100) if total_frames > 0 else 0

            processing_settings = {
                'processing_mode': getattr(self, 'processing_mode_for_thread', 'OFFLINE_GUIDED_FLOW'),
                'num_producers_override': getattr(self, 'override_producers', None),
                'num_consumers_override': getattr(self, 'override_consumers', None),
                'frame_range_override': getattr(self, 'frame_range_override', None),
                'is_autotune_run': getattr(self, 'is_autotune_run_for_thread', False),
                'yolo_det_model_path': self.app.yolo_det_model_path,
                'yolo_pose_model_path': self.app.yolo_pose_model_path,
                'confidence_threshold': self.app.tracker.confidence_threshold if self.app.tracker else 0.4,
                'yolo_input_size': self.app.yolo_input_size
            }

            checkpoint_id = self.checkpoint_manager.create_checkpoint(
                video_path=self.app.file_manager.video_path,
                stage=stage,
                progress_percentage=progress_percentage,
                frame_index=frame_index,
                total_frames=total_frames,
                stage_data=stage_data,
                processing_settings=processing_settings
            )

            if checkpoint_id:
                self.current_checkpoint_id = checkpoint_id

        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")

    def _cleanup_checkpoints_on_completion(self):
        """Clean up checkpoints when processing completes successfully."""
        if self.app.file_manager.video_path:
            try:
                self.checkpoint_manager.delete_video_checkpoints(self.app.file_manager.video_path)
                self.current_checkpoint_id = None
            except Exception as e:
                self.logger.error(f"Failed to cleanup checkpoints: {e}")
