#!/usr/bin/env python3
"""
Stage 2 Contact Analysis - Offline contact detection and analysis.

This tracker implements Stage 2 of the offline processing pipeline, which analyzes
body part interactions using YOLO detection results from Stage 1. It performs
contact analysis, pose estimation, and generates funscript signals based on
detected interactions.

Author: Migrated from Stage 2 CD system
Version: 1.1.0
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from multiprocessing import Event

# Constants
DEFAULT_YOLO_INPUT_SIZE = 640
DEFAULT_NUM_WORKERS = 4
DEFAULT_CHUNK_SIZE = 100
DEFAULT_OF_RECOVERY_WINDOW = 10
DEFAULT_MIN_CONTACT_CONFIDENCE = 0.3
DEFAULT_CONTACT_SMOOTHING_WINDOW = 5
DEFAULT_ENHANCEMENT_STRENGTH = 1.0
PROCESSING_TIME_BUFFER_SECONDS = 30.0
CONSERVATIVE_BASE_FPS = 15.0
DEFAULT_TIME_ESTIMATE_FALLBACK = 300.0

try:
    from ..core.base_offline_tracker import BaseOfflineTracker, OfflineProcessingResult, OfflineProcessingStage
    from ..core.base_tracker import TrackerMetadata, StageDefinition
except ImportError:
    from tracker.tracker_modules.core.base_offline_tracker import BaseOfflineTracker, OfflineProcessingResult, OfflineProcessingStage
    from tracker.tracker_modules.core.base_tracker import TrackerMetadata, StageDefinition

# Import Stage 2 processing module
try:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import detection.cd.stage_2_cd as stage2_module
    STAGE2_MODULE_AVAILABLE = True
except ImportError as e:
    stage2_module = None
    STAGE2_MODULE_AVAILABLE = False
    import logging
    logging.getLogger(__name__).error(f"Stage 2 module not available: {e}. This tracker will not function.")


class Stage2ContactAnalysisTracker(BaseOfflineTracker):
    """
    Stage 2 contact analysis tracker.
    
    This tracker processes YOLO detection results from Stage 1 to:
    - Analyze contact between body parts
    - Perform pose estimation and tracking
    - Generate funscript signals based on detected interactions
    - Apply signal enhancement and smoothing
    - Create segmented output for Stage 3 processing
    """
    
    def __init__(self):
        super().__init__()

        # Stage 2 specific settings
        self.yolo_input_size = DEFAULT_YOLO_INPUT_SIZE
        self.video_type = "auto"
        self.vr_input_format = "he"
        self.vr_fov = 190
        self.vr_pitch = 0
        self.generate_funscript_actions = True
        self.generate_overlay_data = True

        # Processing parameters
        self.enable_optical_flow_recovery = True
        self.of_recovery_frame_window = DEFAULT_OF_RECOVERY_WINDOW
        self.min_contact_confidence = DEFAULT_MIN_CONTACT_CONFIDENCE
        self.contact_smoothing_window = DEFAULT_CONTACT_SMOOTHING_WINDOW

        # Output configuration
        self.output_msgpack_path = None
        self.output_overlay_path = None
        self.output_sqlite_path = None

        # Signal enhancement
        self.enable_signal_enhancement = True
        self.enhancement_strength = DEFAULT_ENHANCEMENT_STRENGTH

        # Performance settings
        self.num_workers = DEFAULT_NUM_WORKERS
        self.chunk_size = DEFAULT_CHUNK_SIZE
    
    @property
    def metadata(self) -> TrackerMetadata:
        """Return metadata describing this tracker."""
        return TrackerMetadata(
            name="OFFLINE_2_STAGE",
            display_name="Contact Analysis (2-Stage)",
            description="Offline contact detection and analysis using YOLO detection results",
            category="offline",
            version="1.0.0", 
            author="Stage 2 CD System",
            tags=["offline", "contact-analysis", "pose-estimation", "stage2", "batch"],
            requires_roi=False,
            supports_dual_axis=True,
            stages=[
                StageDefinition(
                    stage_number=1,
                    name="Detection",
                    description="Object detection and tracking",
                    produces_funscript=False,
                    requires_previous=False,
                    output_type="analysis"
                ),
                StageDefinition(
                    stage_number=2,
                    name="Contact Analysis & Funscript",
                    description="Contact analysis and funscript generation",
                    produces_funscript=True,
                    requires_previous=True,
                    output_type="funscript"
                )
            ],
            properties={
                "produces_funscript_in_stage2": True,
                "supports_batch": True,
                "requires_stage1_data": True,
                "is_stage2_tracker": True,  # Backward compatibility
                "num_stages": 2
            }
        )
    
    @property
    def processing_stages(self) -> List[OfflineProcessingStage]:
        """Return list of processing stages this tracker implements."""
        return [OfflineProcessingStage.STAGE_2]
    
    @property
    def stage_dependencies(self) -> Dict[OfflineProcessingStage, List[OfflineProcessingStage]]:
        """Return dependencies between processing stages."""
        return {
            OfflineProcessingStage.STAGE_2: [OfflineProcessingStage.STAGE_1]
        }
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the Stage 2 contact analysis tracker."""
        try:
            self.app = app_instance

            if not STAGE2_MODULE_AVAILABLE or stage2_module is None:
                self.logger.error("Stage 2 module not available - cannot initialize tracker. "
                                "Check that detection.cd.stage_2_cd module is installed.")
                return False
            
            # Load settings
            if hasattr(app_instance, 'app_settings'):
                settings = app_instance.app_settings
                
                self.yolo_input_size = settings.get('yolo_input_size', 640)
                self.video_type = settings.get('video_type', 'auto')
                self.vr_input_format = settings.get('vr_input_format', 'he')
                self.vr_fov = settings.get('vr_fov', 190)
                self.vr_pitch = settings.get('vr_pitch', 0)
                self.generate_funscript_actions = settings.get('generate_funscript_actions', True)
                self.generate_overlay_data = settings.get('generate_overlay_data', True)
                
                self.enable_optical_flow_recovery = settings.get('enable_optical_flow_recovery', True)
                self.of_recovery_frame_window = settings.get('of_recovery_frame_window', 10)
                self.min_contact_confidence = settings.get('min_contact_confidence', 0.3)
                self.contact_smoothing_window = settings.get('contact_smoothing_window', 5)
                
                self.enable_signal_enhancement = settings.get('enable_signal_enhancement', True)
                self.enhancement_strength = settings.get('enhancement_strength', 1.0)
                
                self.num_workers = settings.get('stage2_num_workers', 4)
                self.chunk_size = settings.get('stage2_chunk_size', 100)
                
                self.logger.info(f"Stage 2 settings: yolo_size={self.yolo_input_size}, "
                               f"workers={self.num_workers}, enhancement={self.enable_signal_enhancement}")
            
            # Validate Stage 2 module availability
            if not hasattr(stage2_module, 'perform_contact_analysis'):
                self.logger.error("Stage 2 module missing perform_contact_analysis function")
                return False
            
            self._initialized = True
            self.logger.info("Stage 2 contact analysis tracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Stage 2 initialization failed: {e}")
            return False
    
    def can_resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Check if processing can be resumed from checkpoint data."""
        try:
            # Check if checkpoint contains Stage 2 specific data
            if checkpoint_data.get('stage') != 'stage2':
                return False
            
            # Check if input files still exist
            stage1_output = checkpoint_data.get('stage1_output_path')
            if not stage1_output or not os.path.exists(stage1_output):
                return False
            
            # Check if partial results exist and are valid
            partial_results = checkpoint_data.get('partial_results_path')
            if partial_results and os.path.exists(partial_results):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Checkpoint validation error: {e}")
            return False
    
    def process_stage(self, 
                     stage: OfflineProcessingStage,
                     video_path: str,
                     input_data: Optional[Dict[str, Any]] = None,
                     input_files: Optional[Dict[str, str]] = None,
                     output_directory: Optional[str] = None,
                     progress_callback: Optional[Callable] = None,
                     frame_range: Optional[Tuple[int, int]] = None,
                     resume_data: Optional[Dict[str, Any]] = None,
                     **kwargs) -> OfflineProcessingResult:
        """
        Process Stage 2 contact analysis.
        """
        if stage != OfflineProcessingStage.STAGE_2:
            return OfflineProcessingResult(
                success=False,
                error_message=f"This tracker only supports Stage 2, got {stage}"
            )
        
        if not self._initialized:
            return OfflineProcessingResult(
                success=False, 
                error_message="Tracker not initialized"
            )
        
        if not stage2_module:
            return OfflineProcessingResult(
                success=False,
                error_message="Stage 2 module not available"
            )
        
        try:
            start_time = time.time()
            self.current_stage = OfflineProcessingStage.STAGE_2
            self.processing_active = True
            
            # Validate dependencies
            if not self.validate_dependencies(stage, input_data or {}, input_files or {}):
                return OfflineProcessingResult(
                    success=False,
                    error_message="Stage dependencies not satisfied"
                )
            
            # Get Stage 1 output path
            stage1_output_path = input_files.get('stage1') or input_files.get('stage1_output')
            if not stage1_output_path or not os.path.exists(stage1_output_path):
                return OfflineProcessingResult(
                    success=False,
                    error_message="Stage 1 output file not found"
                )

            # Validate Stage 1 output file format
            if not self._validate_stage1_output(stage1_output_path):
                return OfflineProcessingResult(
                    success=False,
                    error_message=f"Stage 1 output file is invalid or corrupted: {stage1_output_path}"
                )
            
            # Set up output paths
            if not output_directory:
                output_directory = os.path.dirname(stage1_output_path)

            # Use proper path handling to avoid issues with multiple dots in filename
            video_basename = os.path.basename(video_path)
            video_name_no_ext = os.path.splitext(video_basename)[0]

            self.output_msgpack_path = os.path.join(output_directory,
                                                   f"{video_name_no_ext}_stage2_output.msgpack")

            if self.generate_overlay_data:
                self.output_overlay_path = os.path.join(output_directory,
                                                       f"{video_name_no_ext}_stage2_overlay.msgpack")
            
            # Check for SQLite database option
            self.output_sqlite_path = kwargs.get('sqlite_output_path')
            
            # Get preprocessed video path from Stage 1
            preprocessed_video_path = input_files.get('preprocessed_video')
            
            # Prepare progress callback wrapper
            def progress_wrapper(task_name: str, current: int, total: int):
                if progress_callback:
                    progress_info = {
                        'stage': 'stage2',
                        'task': task_name,
                        'current': current,
                        'total': total,
                        'percentage': (current / total * 100) if total > 0 else 0
                    }
                    progress_callback(progress_info)
            
            # Execute Stage 2 analysis
            self.logger.info(f"Starting Stage 2 contact analysis on {video_path}")
            
            # Get frame range settings
            range_is_active, range_start_frame, range_end_frame = (False, None, None)
            if frame_range:
                range_is_active = True
                range_start_frame, range_end_frame = frame_range
            elif hasattr(self.app, 'funscript_processor'):
                range_is_active, range_start_frame, range_end_frame = self.app.funscript_processor.get_effective_scripting_range()
            
            # Call the Stage 2 module
            # Use multiprocessing.Event() for proper multiprocessing support
            stop_event = self.stop_event if self.stop_event is not None else Event()

            stage2_results = stage2_module.perform_contact_analysis(
                video_path_arg=video_path,
                msgpack_file_path_arg=stage1_output_path,
                preprocessed_video_path_arg=preprocessed_video_path,
                progress_callback=progress_wrapper,
                stop_event=stop_event,
                app=self.app,
                ml_model_dir_path_arg=getattr(self.app, 'pose_model_artifacts_dir', None),
                parent_logger_arg=self.logger,
                output_overlay_msgpack_path=self.output_overlay_path,
                yolo_input_size_arg=self.yolo_input_size,
                video_type_arg=self.video_type,
                vr_input_format_arg=self.vr_input_format,
                vr_fov_arg=self.vr_fov,
                vr_pitch_arg=self.vr_pitch,
                generate_funscript_actions=self.generate_funscript_actions,
                range_is_active=range_is_active,
                range_start_frame=range_start_frame,
                range_end_frame=range_end_frame,
                is_ranged_data_source=range_is_active,
                output_msgpack_path_override=self.output_msgpack_path,
                sqlite_output_path=self.output_sqlite_path,
                enable_of_recovery=self.enable_optical_flow_recovery,
                of_recovery_frame_window=self.of_recovery_frame_window,
                min_contact_confidence=self.min_contact_confidence,
                contact_smoothing_window=self.contact_smoothing_window,
                enable_signal_enhancement=self.enable_signal_enhancement,
                enhancement_strength=self.enhancement_strength
            )
            
            # Process results
            if not stage2_results or not stage2_results.get('success', False):
                error_msg = stage2_results.get('error', 'Stage 2 processing failed') if stage2_results else 'Stage 2 processing failed'
                return OfflineProcessingResult(
                    success=False,
                    error_message=error_msg
                )
            
            # Prepare output files
            output_files = {
                'stage2_output': self.output_msgpack_path
            }
            
            if self.output_overlay_path and os.path.exists(self.output_overlay_path):
                output_files['stage2_overlay'] = self.output_overlay_path
            
            if self.output_sqlite_path and os.path.exists(self.output_sqlite_path):
                output_files['stage2_sqlite'] = self.output_sqlite_path
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            performance_metrics = {
                'processing_time_seconds': processing_time,
                'frames_processed': stage2_results.get('frames_processed', 0),
                'contacts_detected': stage2_results.get('contacts_detected', 0),
                'segments_created': stage2_results.get('segments_created', 0),
                'fps': stage2_results.get('frames_processed', 0) / processing_time if processing_time > 0 else 0
            }
            
            # Prepare checkpoint data
            checkpoint_data = {
                'stage': 'stage2',
                'video_path': video_path,
                'stage1_output_path': stage1_output_path,
                'stage2_output_path': self.output_msgpack_path,
                'processing_complete': True,
                'frame_range': frame_range,
                'settings': {
                    'yolo_input_size': self.yolo_input_size,
                    'generate_funscript_actions': self.generate_funscript_actions,
                    'enable_signal_enhancement': self.enable_signal_enhancement
                }
            }
            
            self.processing_active = False
            self.current_stage = None
            
            self.logger.info(f"Stage 2 contact analysis completed in {processing_time:.1f}s")
            
            return OfflineProcessingResult(
                success=True,
                output_data=stage2_results,
                output_files=output_files,
                checkpoint_data=checkpoint_data,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.processing_active = False
            self.current_stage = None
            self.logger.error(f"Stage 2 processing error: {e}", exc_info=True)

            # Clean up partial outputs on failure
            self._cleanup_partial_outputs()

            return OfflineProcessingResult(
                success=False,
                error_message=f"Stage 2 processing failed: {e}"
            )
    
    def estimate_processing_time(self,
                               stage: OfflineProcessingStage,
                               video_path: str,
                               **kwargs) -> float:
        """Estimate processing time for Stage 2."""
        if stage != OfflineProcessingStage.STAGE_2:
            return 0.0
        
        try:
            # Get video properties
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if frame_count <= 0:
                return 0.0
            
            # Estimation based on empirical data:
            # Stage 2 processes approximately 10-30 frames per second depending on complexity
            # Contact analysis is CPU intensive
            base_fps = CONSERVATIVE_BASE_FPS

            # Adjust based on settings
            if self.enable_optical_flow_recovery:
                base_fps *= 0.7  # OF recovery adds overhead

            if self.enable_signal_enhancement:
                base_fps *= 0.9  # Signal enhancement adds overhead

            # Adjust based on resolution (YOLO input size)
            if self.yolo_input_size > DEFAULT_YOLO_INPUT_SIZE:
                base_fps *= 0.8
            elif self.yolo_input_size < DEFAULT_YOLO_INPUT_SIZE:
                base_fps *= 1.2

            estimated_time = frame_count / base_fps

            # Add buffer for initialization and cleanup
            estimated_time += PROCESSING_TIME_BUFFER_SECONDS

            return estimated_time

        except Exception as e:
            self.logger.warning(f"Could not estimate processing time: {e}")
            return DEFAULT_TIME_ESTIMATE_FALLBACK
    
    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validate Stage 2 specific settings."""
        # Call base validation first
        if not super().validate_settings(settings):
            return False
        
        try:
            # Validate YOLO input size
            yolo_size = settings.get('yolo_input_size', self.yolo_input_size)
            if not isinstance(yolo_size, int) or yolo_size < 320 or yolo_size > 1280:
                self.logger.error("YOLO input size must be between 320 and 1280")
                return False
            
            # Validate confidence threshold
            confidence = settings.get('min_contact_confidence', self.min_contact_confidence)
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                self.logger.error("Min contact confidence must be between 0.0 and 1.0")
                return False
            
            # Validate smoothing window
            smoothing = settings.get('contact_smoothing_window', self.contact_smoothing_window)
            if not isinstance(smoothing, int) or smoothing < 1 or smoothing > 20:
                self.logger.error("Contact smoothing window must be between 1 and 20")
                return False
            
            # Validate enhancement strength
            enhancement = settings.get('enhancement_strength', self.enhancement_strength)
            if not isinstance(enhancement, (int, float)) or enhancement < 0.1 or enhancement > 5.0:
                self.logger.error("Enhancement strength must be between 0.1 and 5.0")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Settings validation error: {e}")
            return False
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Get current checkpoint data for resumption."""
        base_data = super().get_checkpoint_data()
        
        stage2_data = {
            'output_msgpack_path': self.output_msgpack_path,
            'output_overlay_path': self.output_overlay_path,
            'output_sqlite_path': self.output_sqlite_path,
            'settings': {
                'yolo_input_size': self.yolo_input_size,
                'video_type': self.video_type,
                'generate_funscript_actions': self.generate_funscript_actions,
                'enable_signal_enhancement': self.enable_signal_enhancement,
                'enhancement_strength': self.enhancement_strength,
                'num_workers': self.num_workers
            }
        }
        
        base_data.update(stage2_data)
        return base_data
    
    def cleanup(self):
        """Clean up Stage 2 resources."""
        # Stop any ongoing processing
        self.stop_processing()

        # Clean up temporary files if configured
        # Note: Overlay files are preserved as they're needed for UI
        if self.intermediate_cleanup:
            temp_files = []
            # Add any temporary files that should be cleaned up here
            # Currently overlay is preserved intentionally

        super().cleanup()
        self.logger.debug("Stage 2 contact analysis tracker cleaned up")

    def _validate_stage1_output(self, stage1_output_path: str) -> bool:
        """
        Validate Stage 1 output file format and integrity.

        Args:
            stage1_output_path: Path to Stage 1 output file

        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            import msgpack

            # Check file size (should be > 0)
            file_size = os.path.getsize(stage1_output_path)
            if file_size == 0:
                self.logger.error(f"Stage 1 output file is empty: {stage1_output_path}")
                return False

            # Try to open and read header of msgpack file
            with open(stage1_output_path, 'rb') as f:
                unpacker = msgpack.Unpacker(f, raw=False)
                try:
                    # Try to read first item to verify format
                    _ = next(unpacker)
                    return True
                except StopIteration:
                    self.logger.error(f"Stage 1 output file has no data: {stage1_output_path}")
                    return False
                except msgpack.exceptions.ExtraData:
                    # This is actually OK - means there's valid data
                    return True

        except Exception as e:
            self.logger.error(f"Failed to validate Stage 1 output: {e}")
            return False

    def _cleanup_partial_outputs(self):
        """Clean up partial output files after processing failure."""
        files_to_cleanup = [
            self.output_msgpack_path,
            self.output_overlay_path,
            self.output_sqlite_path
        ]

        for file_path in files_to_cleanup:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.logger.debug(f"Cleaned up partial output: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up partial output {file_path}: {e}")