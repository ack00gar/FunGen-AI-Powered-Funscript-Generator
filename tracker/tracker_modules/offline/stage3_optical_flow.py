#!/usr/bin/env python3
"""
Stage 3 Optical Flow - Offline optical flow analysis and funscript generation.

This tracker implements Stage 3 of the offline processing pipeline, which applies
optical flow analysis to segments identified by Stage 2. It uses live tracker
algorithms as sub-components to generate high-quality funscript output with
precise timing and motion detection.

Author: Migrated from Stage 3 OF system
Version: 1.1.0
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from multiprocessing import Event

# Constants
DEFAULT_LIVE_TRACKER = "oscillation_experimental_2"
DEFAULT_CHUNK_OVERLAP_FRAMES = 30
DEFAULT_MIN_SEGMENT_DURATION = 2.0
DEFAULT_SEGMENT_PADDING_FRAMES = 15
DEFAULT_NUM_WORKERS = 4
DEFAULT_CHUNK_SIZE_FRAMES = 1000
DEFAULT_SMOOTHING_WINDOW = 5
DEFAULT_QUALITY_MODE = "balanced"
PROCESSING_TIME_BUFFER_SECONDS = 60.0
DEFAULT_TIME_ESTIMATE_FALLBACK = 600.0
ESTIMATED_SEGMENT_COVERAGE = 0.45
FAST_MODE_FPS = 45.0
BALANCED_MODE_FPS = 25.0
HIGH_QUALITY_MODE_FPS = 15.0

try:
    from ..core.base_offline_tracker import BaseOfflineTracker, OfflineProcessingResult, OfflineProcessingStage
    from ..core.base_tracker import TrackerMetadata, StageDefinition
except ImportError:
    from tracker.tracker_modules.core.base_offline_tracker import BaseOfflineTracker, OfflineProcessingResult, OfflineProcessingStage
    from tracker.tracker_modules.core.base_tracker import TrackerMetadata, StageDefinition

# Import Stage 3 processing module and live tracker
try:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import detection.cd.stage_3_of_processor as stage3_module
    from detection.cd.data_structures import Segment, FrameObject
    STAGE3_MODULE_AVAILABLE = True
except ImportError as e:
    stage3_module = None
    STAGE3_MODULE_AVAILABLE = False
    import logging
    logging.getLogger(__name__).error(f"Stage 3 module not available: {e}. This tracker will not function.")


class Stage3OpticalFlowTracker(BaseOfflineTracker):
    """
    Stage 3 optical flow analysis tracker.
    
    This tracker processes Stage 2 contact analysis results to:
    - Apply optical flow tracking to identified segments
    - Use live tracker algorithms (oscillation detectors) as sub-components
    - Generate high-precision funscript actions with proper timing
    - Support multiprocessing for performance
    - Handle video segment processing with overlapping chunks
    """
    
    def __init__(self):
        super().__init__()

        # Live tracker integration
        self.live_tracker_name = DEFAULT_LIVE_TRACKER
        self.tracker_manager = None

        # Optical flow settings
        self.chunk_overlap_frames = DEFAULT_CHUNK_OVERLAP_FRAMES
        self.min_segment_duration_seconds = DEFAULT_MIN_SEGMENT_DURATION
        self.segment_padding_frames = DEFAULT_SEGMENT_PADDING_FRAMES

        # Processing parameters
        self.num_workers = DEFAULT_NUM_WORKERS
        self.enable_sqlite = True  # Prefer SQLite for memory efficiency
        self.chunk_size_frames = DEFAULT_CHUNK_SIZE_FRAMES

        # Quality settings
        self.quality_mode = DEFAULT_QUALITY_MODE
        self.enable_motion_smoothing = True
        self.smoothing_window_size = DEFAULT_SMOOTHING_WINDOW
        
        # Output configuration
        self.output_funscript_path = None
        self.generate_debug_output = False
        
        # Live tracker configuration
        self.live_tracker_settings = {
            'oscillation_ema_alpha': 0.2,  # Less aggressive smoothing for offline processing
            'oscillation_sensitivity': 1.2,  # Slightly higher sensitivity
            'show_masks': False,  # Disable visual overlays for performance
            'show_roi': False
        }
    
    @property
    def metadata(self) -> TrackerMetadata:
        """Return metadata describing this tracker."""
        return TrackerMetadata(
            name="OFFLINE_3_STAGE", 
            display_name="Optical Flow Analysis (3-Stage)",
            description="Offline optical flow tracking using live tracker algorithms on Stage 2 segments",
            category="offline",
            version="1.0.0",
            author="Stage 3 OF System",
            tags=["offline", "optical-flow", "stage3", "batch", "live-tracker-integration"],
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
                    name="Segmentation",
                    description="Video segmentation and scene analysis",
                    produces_funscript=False,
                    requires_previous=True,
                    output_type="segmentation"
                ),
                StageDefinition(
                    stage_number=3,
                    name="Optical Flow & Funscript",
                    description="Optical flow analysis and funscript generation",
                    produces_funscript=True,
                    requires_previous=True,
                    output_type="funscript"
                )
            ],
            properties={
                "produces_funscript_in_stage3": True,
                "supports_batch": True,
                "requires_stage2_data": True,
                "is_stage3_tracker": True,  # Backward compatibility
                "num_stages": 3
            }
        )
    
    @property
    def processing_stages(self) -> List[OfflineProcessingStage]:
        """Return list of processing stages this tracker implements."""
        return [OfflineProcessingStage.STAGE_3]
    
    @property
    def stage_dependencies(self) -> Dict[OfflineProcessingStage, List[OfflineProcessingStage]]:
        """Return dependencies between processing stages."""
        return {
            OfflineProcessingStage.STAGE_3: [OfflineProcessingStage.STAGE_1, OfflineProcessingStage.STAGE_2]
        }
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the Stage 3 optical flow tracker."""
        try:
            self.app = app_instance

            if not STAGE3_MODULE_AVAILABLE or stage3_module is None:
                self.logger.error("Stage 3 module not available - cannot initialize tracker. "
                                "Check that detection.cd.stage_3_of_processor module is installed.")
                return False
            
            # Load settings
            if hasattr(app_instance, 'app_settings'):
                settings = app_instance.app_settings
                
                self.live_tracker_name = settings.get('stage3_live_tracker', 'oscillation_experimental_2')
                self.chunk_overlap_frames = settings.get('stage3_chunk_overlap', 30)
                self.min_segment_duration_seconds = settings.get('stage3_min_segment_duration', 2.0)
                self.segment_padding_frames = settings.get('stage3_segment_padding', 15)
                self.num_workers = settings.get('stage3_num_workers', 4)
                self.chunk_size_frames = settings.get('stage3_chunk_size', 1000)
                self.quality_mode = settings.get('stage3_quality_mode', 'balanced')
                self.enable_motion_smoothing = settings.get('stage3_enable_smoothing', True)
                self.smoothing_window_size = settings.get('stage3_smoothing_window', 5)
                self.generate_debug_output = settings.get('stage3_debug_output', False)
                
                # Update live tracker settings from app settings
                self.live_tracker_settings.update({
                    'oscillation_ema_alpha': settings.get('oscillation_ema_alpha', 0.2),
                    'oscillation_sensitivity': settings.get('oscillation_sensitivity', 1.2),
                    'oscillation_grid_size': settings.get('oscillation_grid_size', 8)
                })
                
                self.logger.info(f"Stage 3 settings: live_tracker={self.live_tracker_name}, "
                               f"workers={self.num_workers}, quality={self.quality_mode}")
            
            # Initialize live tracker manager
            try:
                # Import TrackerManager directly (avoids global variable pattern)
                from tracker.tracker_manager import TrackerManager

                # Get tracker model path from app instance
                tracker_model_path = getattr(app_instance, 'yolo_det_model_path', '') or ''
                if not tracker_model_path:
                    self.logger.error("No tracker model path available from app instance")
                    return False

                self.tracker_manager = TrackerManager(app_instance, tracker_model_path)
                if not self.tracker_manager.set_tracking_mode(self.live_tracker_name):
                    self.logger.warning(f"Failed to set live tracker {self.live_tracker_name}, using default")
                    # Try fallback trackers
                    fallback_trackers = ['oscillation_experimental', 'oscillation_legacy']
                    tracker_set = False
                    for fallback in fallback_trackers:
                        if self.tracker_manager.set_tracking_mode(fallback):
                            self.live_tracker_name = fallback
                            tracker_set = True
                            break
                    if not tracker_set:
                        self.logger.error("No suitable live tracker available for Stage 3")
                        return False
                else:
                    self.logger.info(f"Live tracker {self.live_tracker_name} configured for Stage 3")
            except Exception as e:
                self.logger.error(f"Failed to initialize live tracker manager: {e}")
                return False
            
            # Validate Stage 3 module availability
            if not hasattr(stage3_module, 'perform_stage3_analysis'):
                self.logger.error("Stage 3 module missing perform_stage3_analysis function")
                return False
            
            self._initialized = True
            self.logger.info("Stage 3 optical flow tracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Stage 3 initialization failed: {e}")
            return False
    
    def can_resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Check if processing can be resumed from checkpoint data."""
        try:
            # Check if checkpoint contains Stage 3 specific data
            if checkpoint_data.get('stage') != 'stage3':
                return False
            
            # Check if input files still exist
            stage2_output = checkpoint_data.get('stage2_output_path')
            if not stage2_output or not os.path.exists(stage2_output):
                return False
            
            video_path = checkpoint_data.get('video_path')
            if not video_path or not os.path.exists(video_path):
                return False
            
            # Check if we can resume from partial results
            partial_funscript = checkpoint_data.get('partial_funscript_path')
            if partial_funscript and os.path.exists(partial_funscript):
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
        Process Stage 3 optical flow analysis.
        """
        if stage != OfflineProcessingStage.STAGE_3:
            return OfflineProcessingResult(
                success=False,
                error_message=f"This tracker only supports Stage 3, got {stage}"
            )
        
        if not self._initialized:
            return OfflineProcessingResult(
                success=False,
                error_message="Tracker not initialized"
            )
        
        if not stage3_module:
            return OfflineProcessingResult(
                success=False,
                error_message="Stage 3 module not available"
            )
        try:
            start_time = time.time()
            self.current_stage = OfflineProcessingStage.STAGE_3
            self.processing_active = True
            
            # Validate dependencies
            if not self.validate_dependencies(stage, input_data or {}, input_files or {}):
                return OfflineProcessingResult(
                    success=False,
                    error_message="Stage dependencies not satisfied"
                )
            
            # Get required input files
            stage2_output_path = input_files.get('stage2') or input_files.get('stage2_output')
            if not stage2_output_path or not os.path.exists(stage2_output_path):
                return OfflineProcessingResult(
                    success=False,
                    error_message="Stage 2 output file not found"
                )
            
            # Get preprocessed video path
            preprocessed_video_path = input_files.get('preprocessed_video')
            
            # Get SQLite database path if available
            sqlite_db_path = input_files.get('stage2_sqlite')
            
            # Set up output paths
            if not output_directory:
                output_directory = os.path.dirname(stage2_output_path)
            
            self.output_funscript_path = os.path.join(output_directory,
                                                     os.path.basename(video_path).replace('.', '_') + '.funscript')
            
            # Load Stage 2 results
            stage2_data = self._load_stage2_results(stage2_output_path, sqlite_db_path)
            if not stage2_data:
                return OfflineProcessingResult(
                    success=False,
                    error_message="Failed to load Stage 2 results"
                )
            
            segments = stage2_data.get('segments', [])
            frame_objects = stage2_data.get('frame_objects', {})
            
            if not segments:
                return OfflineProcessingResult(
                    success=False,
                    error_message="No segments found in Stage 2 results"
                )
            
            self.logger.info(f"Loaded {len(segments)} segments and {len(frame_objects)} frame objects from Stage 2")
            
            # Prepare progress callback wrapper
            def progress_wrapper(progress_info: Dict[str, Any]):
                if progress_callback:
                    progress_info['stage'] = 'stage3'
                    progress_callback(progress_info)
            
            # Prepare tracker configuration
            tracker_config = self._build_tracker_config()
            
            # Prepare common app configuration
            common_app_config = self._build_common_app_config()
            
            # Execute Stage 3 analysis
            self.logger.info(f"Starting Stage 3 optical flow analysis on {len(segments)} segments")
            
            stage3_results = stage3_module.perform_stage3_analysis(
                video_path=video_path,
                preprocessed_video_path_arg=preprocessed_video_path,
                atr_segments_list=segments,
                s2_frame_objects_map=frame_objects,
                tracker_config=tracker_config,
                common_app_config=common_app_config,
                progress_callback=progress_wrapper,
                stop_event=self.stop_event or Event(),
                parent_logger=self.logger,
                num_workers=self.num_workers,
                sqlite_db_path=sqlite_db_path
            )
            
            # Process results
            if not stage3_results or not stage3_results.get('success', False):
                error_msg = stage3_results.get('error', 'Stage 3 processing failed') if stage3_results else 'Stage 3 processing failed'
                return OfflineProcessingResult(
                    success=False,
                    error_message=error_msg
                )
            
            # Save funscript output
            funscript_data = stage3_results.get('funscript_data')
            if funscript_data and self.output_funscript_path:
                try:
                    self._save_funscript_output(funscript_data, self.output_funscript_path)
                except Exception as e:
                    self.logger.warning(f"Failed to save funscript output: {e}")
            
            # Prepare output files
            output_files = {}
            if self.output_funscript_path and os.path.exists(self.output_funscript_path):
                output_files['funscript'] = self.output_funscript_path
            
            # Add debug output if generated
            if self.generate_debug_output:
                debug_output_path = self.output_funscript_path.replace('.funscript', '_debug.json')
                if os.path.exists(debug_output_path):
                    output_files['debug_output'] = debug_output_path
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            performance_metrics = {
                'processing_time_seconds': processing_time,
                'segments_processed': len(segments),
                'total_actions': stage3_results.get('total_actions', 0),
                'average_segment_time': processing_time / len(segments) if segments else 0,
                'live_tracker_used': self.live_tracker_name,
                'quality_mode': self.quality_mode
            }
            
            # Prepare checkpoint data
            checkpoint_data = {
                'stage': 'stage3',
                'video_path': video_path,
                'stage2_output_path': stage2_output_path,
                'funscript_output_path': self.output_funscript_path,
                'processing_complete': True,
                'live_tracker_name': self.live_tracker_name,
                'settings': {
                    'num_workers': self.num_workers,
                    'quality_mode': self.quality_mode,
                    'chunk_size_frames': self.chunk_size_frames,
                    'live_tracker_settings': self.live_tracker_settings
                }
            }
            
            self.processing_active = False
            self.current_stage = None
            
            self.logger.info(f"Stage 3 optical flow analysis completed in {processing_time:.1f}s")
            
            return OfflineProcessingResult(
                success=True,
                output_data=stage3_results,
                output_files=output_files,
                checkpoint_data=checkpoint_data,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.processing_active = False
            self.current_stage = None
            self.logger.error(f"Stage 3 processing error: {e}", exc_info=True)
            return OfflineProcessingResult(
                success=False,
                error_message=f"Stage 3 processing failed: {e}"
            )
    
    def estimate_processing_time(self,
                               stage: OfflineProcessingStage,
                               video_path: str,
                               **kwargs) -> float:
        """Estimate processing time for Stage 3."""
        if stage != OfflineProcessingStage.STAGE_3:
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
            
            # Estimation based on segments and quality settings
            # Stage 3 typically processes 30-60% of video frames (only segments)
            estimated_processed_frames = frame_count * ESTIMATED_SEGMENT_COVERAGE

            # Base processing rate depends on quality mode and live tracker
            if self.quality_mode == "fast":
                base_fps = FAST_MODE_FPS
            elif self.quality_mode == "high_quality":
                base_fps = HIGH_QUALITY_MODE_FPS
            else:  # balanced
                base_fps = BALANCED_MODE_FPS

            # Adjust for live tracker type
            if "experimental_2" in self.live_tracker_name:
                base_fps *= 0.85  # More sophisticated algorithm
            elif "legacy" in self.live_tracker_name:
                base_fps *= 0.75  # More intensive processing

            # Adjust for multiprocessing
            worker_efficiency = min(self.num_workers * 0.75, 4.0)  # Diminishing returns
            base_fps *= worker_efficiency

            # Adjust for chunking overhead
            base_fps *= 0.9

            estimated_time = estimated_processed_frames / base_fps

            # Add buffer for initialization, loading, and saving
            estimated_time += PROCESSING_TIME_BUFFER_SECONDS

            return estimated_time

        except Exception as e:
            self.logger.warning(f"Could not estimate processing time: {e}")
            return DEFAULT_TIME_ESTIMATE_FALLBACK
    
    def _load_stage2_results(self, stage2_output_path: str, sqlite_db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load Stage 2 results from msgpack or SQLite."""
        try:
            # Prefer SQLite if available for memory efficiency
            if sqlite_db_path and os.path.exists(sqlite_db_path):
                return self._load_from_sqlite(sqlite_db_path)
            
            # Fallback to msgpack
            if os.path.exists(stage2_output_path):
                return self._load_from_msgpack(stage2_output_path)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load Stage 2 results: {e}")
            return None
    
    def _load_from_sqlite(self, sqlite_path: str) -> Dict[str, Any]:
        """Load data from SQLite database."""
        storage = None
        try:
            from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage

            storage = Stage2SQLiteStorage(sqlite_path, self.logger)

            # Load segments
            segments_data = storage.get_segments()
            segments = []

            # Convert to Segment objects if needed
            for segment_data in segments_data:
                segments.append(segment_data)

            # For Stage 3 with SQLite, we pass the database path instead of loading
            # all frame objects into memory. The Stage 3 processor will query on-demand.
            # We keep a reference to indicate SQLite should be used
            frame_objects = {'_use_sqlite': True, '_sqlite_path': sqlite_path}

            return {
                'segments': segments,
                'frame_objects': frame_objects,
                'data_source': 'sqlite',
                'sqlite_path': sqlite_path
            }

        except Exception as e:
            self.logger.error(f"Failed to load from SQLite: {e}")
            raise
        finally:
            # Ensure storage connection is closed
            if storage is not None:
                try:
                    storage.close()
                except AttributeError:
                    # Storage may not have close method, that's OK
                    pass
    
    def _load_from_msgpack(self, msgpack_path: str) -> Dict[str, Any]:
        """Load data from msgpack file."""
        try:
            import msgpack
            
            with open(msgpack_path, 'rb') as f:
                data = msgpack.load(f, raw=False)
            
            # Convert to expected format
            segments = data.get('segments', [])
            frame_objects = data.get('frame_objects', {})
            
            return {
                'segments': segments,
                'frame_objects': frame_objects,
                'data_source': 'msgpack',
                'msgpack_path': msgpack_path
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load from msgpack: {e}")
            raise
    
    def _build_tracker_config(self) -> Dict[str, Any]:
        """Build tracker configuration for Stage 3."""
        return {
            'tracker_name': self.live_tracker_name,
            'tracker_settings': self.live_tracker_settings.copy(),
            'quality_mode': self.quality_mode,
            'enable_smoothing': self.enable_motion_smoothing,
            'smoothing_window': self.smoothing_window_size,
            'chunk_overlap_frames': self.chunk_overlap_frames,
            'segment_padding_frames': self.segment_padding_frames
        }
    
    def _build_common_app_config(self) -> Dict[str, Any]:
        """Build common app configuration for Stage 3."""
        config = {
            'hardware_acceleration_method': getattr(self.app, 'hardware_acceleration_method', 'none'),
            'available_ffmpeg_hwaccels': getattr(self.app, 'available_ffmpeg_hwaccels', []),
            'video_type': getattr(self.app.processor, 'video_type_setting', 'auto') if hasattr(self.app, 'processor') else 'auto',
            'tracking_axis_mode': getattr(self.app, 'tracking_axis_mode', 'both'),
            'single_axis_output_target': getattr(self.app, 'single_axis_output_target', 'primary')
        }
        
        # Add video processor specific settings if available
        if hasattr(self.app, 'processor'):
            processor = self.app.processor
            config.update({
                'vr_input_format': getattr(processor, 'vr_input_format', 'he'),
                'vr_fov': getattr(processor, 'vr_fov', 190),
                'vr_pitch': getattr(processor, 'vr_pitch', 0)
            })
        
        return config
    
    def _save_funscript_output(self, funscript_data: Any, output_path: str):
        """Save funscript data to file."""
        try:
            import json

            # If funscript_data is already a DualAxisFunscript object with save method
            if hasattr(funscript_data, 'save') and callable(funscript_data.save):
                funscript_data.save(output_path)
            elif isinstance(funscript_data, dict):
                # Handle dict format - ensure it has required funscript structure
                if 'actions' not in funscript_data:
                    funscript_data = {
                        'version': '1.0',
                        'inverted': False,
                        'range': 100,
                        'actions': funscript_data.get('data', [])
                    }
                with open(output_path, 'w') as f:
                    json.dump(funscript_data, f, indent=2)
            elif isinstance(funscript_data, list):
                # Handle list of actions
                output_data = {
                    'version': '1.0',
                    'inverted': False,
                    'range': 100,
                    'actions': funscript_data
                }
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
            else:
                # Unsupported format
                raise ValueError(f"Unsupported funscript data type: {type(funscript_data)}. "
                               f"Expected object with save() method, dict, or list.")

            self.logger.info(f"Saved funscript output to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save funscript output: {e}", exc_info=True)
            raise
    
    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validate Stage 3 specific settings."""
        # Call base validation first
        if not super().validate_settings(settings):
            return False
        
        try:
            # Validate live tracker name
            live_tracker = settings.get('stage3_live_tracker', self.live_tracker_name)
            if not isinstance(live_tracker, str) or not live_tracker:
                self.logger.error("Live tracker name must be a non-empty string")
                return False
            
            # Validate quality mode
            quality_mode = settings.get('stage3_quality_mode', self.quality_mode)
            if quality_mode not in ['fast', 'balanced', 'high_quality']:
                self.logger.error("Quality mode must be 'fast', 'balanced', or 'high_quality'")
                return False
            
            # Validate chunk settings
            chunk_size = settings.get('stage3_chunk_size', self.chunk_size_frames)
            if not isinstance(chunk_size, int) or chunk_size < 100 or chunk_size > 10000:
                self.logger.error("Chunk size must be between 100 and 10000 frames")
                return False
            
            overlap = settings.get('stage3_chunk_overlap', self.chunk_overlap_frames)
            if not isinstance(overlap, int) or overlap < 0 or overlap > chunk_size // 2:
                self.logger.error("Chunk overlap must be between 0 and half of chunk size")
                return False
            
            # Validate timing settings
            min_duration = settings.get('stage3_min_segment_duration', self.min_segment_duration_seconds)
            if not isinstance(min_duration, (int, float)) or min_duration < 0.5 or min_duration > 30.0:
                self.logger.error("Min segment duration must be between 0.5 and 30.0 seconds")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Settings validation error: {e}")
            return False
    
    def cleanup(self):
        """Clean up Stage 3 resources."""
        # Stop any ongoing processing
        self.stop_processing()

        # Cleanup live tracker manager
        if self.tracker_manager:
            try:
                self.tracker_manager.cleanup()
            except Exception as e:
                self.logger.warning(f"Error cleaning up tracker manager: {e}")
            self.tracker_manager = None

        super().cleanup()
        self.logger.debug("Stage 3 optical flow tracker cleaned up")
