#!/usr/bin/env python3
"""
Stage 3 Mixed - Hybrid offline processing with selective live tracking.

This tracker implements the Stage 3 Mixed approach, which intelligently combines
Stage 2 signal output with selective application of live ROI tracking for specific
chapter types (BJ/HJ). It provides the best of both worlds: reliable Stage 2
signals for most content and precise live tracking for interactive scenes.

Author: Migrated from Stage 3 Mixed system
Version: 1.0.0
"""

import logging
import os
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from multiprocessing import Event

try:
    from ..core.base_offline_tracker import BaseOfflineTracker, TrackerMetadata, OfflineProcessingResult, OfflineProcessingStage
except ImportError:
    from tracker_modules.core.base_offline_tracker import BaseOfflineTracker, TrackerMetadata, OfflineProcessingResult, OfflineProcessingStage

# Import Stage 3 Mixed processing module and dependencies
try:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    import detection.cd.stage_3_mixed_processor as stage3_mixed_module
    from detection.cd.data_structures import FrameObject
    from application.utils.video_segment import VideoSegment
    # Lazy import to avoid circular dependency during registry initialization
    # TrackerManager will be imported when needed in process_stage method
    TrackerManager = None
except ImportError as e:
    stage3_mixed_module = None
    import warnings
    warnings.warn(f"Stage 3 Mixed module not available: {e}")


class Stage3MixedTracker(BaseOfflineTracker):
    """
    Stage 3 Mixed processing tracker.
    
    This tracker implements a hybrid approach that:
    - Uses Stage 2 signal as-is for most chapters (reliable baseline)
    - Applies YOLO ROI tracking only for BJ/HJ chapters using Stage 2 detections as ROI input
    - Maintains compatibility with existing 3-stage infrastructure
    - Provides intelligent chapter-type-based processing selection
    - Integrates live tracker algorithms as sub-components
    """
    
    def __init__(self):
        super().__init__()
        
        # Mixed processing configuration
        self.use_live_tracking_chapters = ["BJ", "HJ"]  # Chapter types that use live tracking
        self.stage2_signal_chapters = ["CG/Miss", "Doggy", "Cowgirl", "NR"]  # Use Stage 2 signal directly
        
        # Live tracker integration for selective chapters
        self.live_tracker_name = "yolo_roi"  # Use YOLO ROI tracker for BJ/HJ
        self.tracker_manager = None
        
        # ROI adaptation settings
        self.roi_update_frequency = 30  # Update ROI every N frames (~1 second at 30fps)
        self.roi_smoothing_enabled = True
        self.roi_padding_pixels = 20
        
        # Signal mixing settings
        self.signal_blend_transition_frames = 15  # Smooth transition between signal sources
        self.stage2_signal_weight = 0.7  # Weight for Stage 2 signal in mixed regions
        self.live_signal_weight = 0.8  # Weight for live tracker signal
        
        # Quality settings for mixed mode
        self.mixed_mode_quality = "balanced"  # "fast", "balanced", "high_precision"
        self.enable_oscillation_enhancement = True  # Enhanced oscillation detection for BJ/HJ
        self.oscillation_sensitivity_boost = 1.3  # Boost sensitivity for live tracking chapters
        
        # Output configuration
        self.output_funscript_path = None
        self.generate_mixed_debug_data = True  # Generate debug info showing signal sources
        
        # Performance settings
        self.enable_adaptive_processing = True  # Adapt processing based on chapter type
        self.memory_optimization = True  # Optimize memory usage for large videos
        
        # Live tracker settings optimized for mixed mode
        self.mixed_live_tracker_settings = {
            # ROI tracker settings
            'roi_update_interval': 5,  # More frequent updates for mixed mode
            'max_frames_for_roi_persistence': 45,  # Longer persistence 
            'sensitivity': 12.0,  # Higher sensitivity for interactive content
            'adaptive_flow_scale': True,
            'show_masks': False,  # Disable visual overlays for performance
            'show_roi': False,
            
            # Oscillation settings (if oscillation tracker is used as sub-component)
            'oscillation_ema_alpha': 0.15,  # Less aggressive smoothing
            'oscillation_sensitivity': self.oscillation_sensitivity_boost,
            'oscillation_grid_size': 12,  # More precise grid
            'oscillation_hold_duration_ms': 150  # Shorter hold for responsiveness
        }
    
    @property
    def metadata(self) -> TrackerMetadata:
        """Return metadata describing this tracker."""
        return TrackerMetadata(
            name="stage3_mixed",
            display_name="Offline Mixed Processing (3-Stage)",
            description="Hybrid approach: Stage 2 signals + selective live ROI tracking for BJ/HJ chapters",
            category="offline",
            version="1.0.0", 
            author="Stage 3 Mixed System",
            tags=["offline", "mixed", "hybrid", "stage3", "selective-tracking", "intelligent"],
            requires_roi=False,
            supports_dual_axis=True
        )
    
    @property
    def processing_stages(self) -> List[OfflineProcessingStage]:
        """Return list of processing stages this tracker implements."""
        return [OfflineProcessingStage.STAGE_3_MIXED]
    
    @property
    def stage_dependencies(self) -> Dict[OfflineProcessingStage, List[OfflineProcessingStage]]:
        """Return dependencies between processing stages."""
        return {
            OfflineProcessingStage.STAGE_3_MIXED: [OfflineProcessingStage.STAGE_1, OfflineProcessingStage.STAGE_2]
        }
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the Stage 3 Mixed tracker."""
        global TrackerManager  # Declare at function start
        try:
            self.app = app_instance
            
            if not stage3_mixed_module:
                self.logger.error("Stage 3 Mixed module not available - cannot initialize tracker")
                return False
            
            # Load settings
            if hasattr(app_instance, 'app_settings'):
                settings = app_instance.app_settings
                
                # Mixed processing configuration
                self.use_live_tracking_chapters = settings.get('mixed_live_tracking_chapters', ["BJ", "HJ"])
                self.stage2_signal_chapters = settings.get('mixed_stage2_chapters', ["CG/Miss", "Doggy", "Cowgirl", "NR"])
                self.live_tracker_name = settings.get('mixed_live_tracker', 'yolo_roi')
                
                # ROI and signal settings
                self.roi_update_frequency = settings.get('mixed_roi_update_frequency', 30)
                self.roi_smoothing_enabled = settings.get('mixed_roi_smoothing', True)
                self.roi_padding_pixels = settings.get('mixed_roi_padding', 20)
                
                self.signal_blend_transition_frames = settings.get('mixed_transition_frames', 15)
                self.stage2_signal_weight = settings.get('mixed_stage2_weight', 0.7)
                self.live_signal_weight = settings.get('mixed_live_weight', 0.8)
                
                # Quality settings
                self.mixed_mode_quality = settings.get('mixed_quality_mode', 'balanced')
                self.enable_oscillation_enhancement = settings.get('mixed_oscillation_enhancement', True)
                self.oscillation_sensitivity_boost = settings.get('mixed_oscillation_boost', 1.3)
                
                # Performance settings
                self.enable_adaptive_processing = settings.get('mixed_adaptive_processing', True)
                self.memory_optimization = settings.get('mixed_memory_optimization', True)
                self.generate_mixed_debug_data = settings.get('mixed_debug_data', True)
                
                # Update live tracker settings
                self.mixed_live_tracker_settings.update({
                    'sensitivity': settings.get('sensitivity', 12.0),
                    'oscillation_sensitivity': settings.get('oscillation_sensitivity', self.oscillation_sensitivity_boost),
                    'roi_update_interval': settings.get('roi_update_interval', 5)
                })
                
                self.logger.info(f"Mixed settings: live_chapters={self.use_live_tracking_chapters}, "
                               f"tracker={self.live_tracker_name}, quality={self.mixed_mode_quality}")
            
            # Initialize live tracker manager
            try:
                # Lazy import to avoid circular dependency
                if TrackerManager is None:
                    from tracker.tracker_manager import TrackerManager as TM
                    TrackerManager = TM
                
                self.tracker_manager = TrackerManager(app_instance)
                if not self.tracker_manager.set_tracker(self.live_tracker_name, **self.mixed_live_tracker_settings):
                    self.logger.warning(f"Failed to set live tracker {self.live_tracker_name}, trying fallbacks")
                    # Try fallback trackers suitable for mixed mode
                    fallback_trackers = ['oscillation_experimental_2', 'oscillation_experimental', 'user_roi']
                    tracker_set = False
                    for fallback in fallback_trackers:
                        if self.tracker_manager.set_tracker(fallback, **self.mixed_live_tracker_settings):
                            self.live_tracker_name = fallback
                            tracker_set = True
                            self.logger.info(f"Using fallback tracker: {fallback}")
                            break
                    if not tracker_set:
                        self.logger.error("No suitable live tracker available for Mixed processing")
                        return False
                else:
                    self.logger.info(f"Live tracker {self.live_tracker_name} configured for Mixed processing")
            except Exception as e:
                self.logger.error(f"Failed to initialize live tracker manager: {e}")
                return False
            
            # Validate Stage 3 Mixed module availability
            if not hasattr(stage3_mixed_module, 'MixedStageProcessor'):
                self.logger.error("Stage 3 Mixed module missing MixedStageProcessor class")
                return False
            
            self._initialized = True
            self.logger.info("Stage 3 Mixed tracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Stage 3 Mixed initialization failed: {e}")
            return False
    
    def can_resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Check if processing can be resumed from checkpoint data."""
        try:
            # Check if checkpoint contains Stage 3 Mixed specific data
            if checkpoint_data.get('stage') != 'stage3_mixed':
                return False
            
            # Check if input files still exist
            stage2_output = checkpoint_data.get('stage2_output_path')
            if not stage2_output or not os.path.exists(stage2_output):
                return False
            
            video_path = checkpoint_data.get('video_path')
            if not video_path or not os.path.exists(video_path):
                return False
            
            # Check if partial results exist
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
        Process Stage 3 Mixed analysis.
        """
        if stage != OfflineProcessingStage.STAGE_3_MIXED:
            return OfflineProcessingResult(
                success=False,
                error_message=f"This tracker only supports Stage 3 Mixed, got {stage}"
            )
        
        if not self._initialized:
            return OfflineProcessingResult(
                success=False,
                error_message="Tracker not initialized"
            )
        
        if not stage3_mixed_module:
            return OfflineProcessingResult(
                success=False,
                error_message="Stage 3 Mixed module not available"
            )
        
        try:
            start_time = time.time()
            self.current_stage = OfflineProcessingStage.STAGE_3_MIXED
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
            
            # Set up output paths
            if not output_directory:
                output_directory = os.path.dirname(stage2_output_path)
            
            self.output_funscript_path = os.path.join(output_directory,
                                                     os.path.basename(video_path).replace('.', '_') + '_mixed.funscript')
            
            # Load Stage 2 results
            stage2_data = self._load_stage2_results(stage2_output_path)
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
            
            # Analyze segments for processing strategy
            processing_strategy = self._analyze_segments_for_mixed_processing(segments)
            self.logger.info(f"Mixed processing strategy: {len(processing_strategy['live_tracking'])} live, "
                           f"{len(processing_strategy['stage2_signal'])} Stage 2 signal")
            
            # Initialize Mixed Stage Processor
            mixed_processor = stage3_mixed_module.MixedStageProcessor(
                tracker_model_path=getattr(self.app, 'yolo_det_model_path', ''),
                pose_model_path=getattr(self.app, 'yolo_pose_model_path', None)
            )
            
            # Set Stage 2 results
            mixed_processor.set_stage2_results(frame_objects, segments)
            
            # Prepare progress callback wrapper
            def progress_wrapper(frame_id: int, total_frames: int, chapter_type: str, signal_source: str):
                if progress_callback:
                    progress_info = {
                        'stage': 'stage3_mixed',
                        'current_frame': frame_id,
                        'total_frames': total_frames,
                        'chapter_type': chapter_type,
                        'signal_source': signal_source,
                        'percentage': (frame_id / total_frames * 100) if total_frames > 0 else 0
                    }
                    progress_callback(progress_info)
            
            # Execute Mixed Stage 3 analysis
            self.logger.info("Starting Stage 3 Mixed processing")
            
            # Process segments using mixed approach
            mixed_results = self._execute_mixed_stage_processing(
                mixed_processor=mixed_processor,
                video_path=video_path,
                preprocessed_video_path=preprocessed_video_path,
                segments=segments,
                frame_objects=frame_objects,
                processing_strategy=processing_strategy,
                progress_callback=progress_wrapper,
                **kwargs
            )
            
            # Process results
            if not mixed_results or not mixed_results.get('success', False):
                error_msg = mixed_results.get('error', 'Mixed processing failed') if mixed_results else 'Mixed processing failed'
                return OfflineProcessingResult(
                    success=False,
                    error_message=error_msg
                )
            
            # Save funscript output
            funscript_data = mixed_results.get('funscript_data')
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
            if self.generate_mixed_debug_data:
                debug_output_path = self.output_funscript_path.replace('.funscript', '_mixed_debug.json')
                if mixed_results.get('debug_data'):
                    self._save_mixed_debug_data(mixed_results['debug_data'], debug_output_path)
                    output_files['debug_output'] = debug_output_path
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            performance_metrics = {
                'processing_time_seconds': processing_time,
                'total_segments': len(segments),
                'live_tracking_segments': len(processing_strategy['live_tracking']),
                'stage2_signal_segments': len(processing_strategy['stage2_signal']),
                'total_actions': mixed_results.get('total_actions', 0),
                'mixed_processing_efficiency': mixed_results.get('processing_efficiency', 0.0),
                'live_tracker_used': self.live_tracker_name,
                'signal_source_breakdown': mixed_results.get('signal_source_stats', {})
            }
            
            # Prepare checkpoint data
            checkpoint_data = {
                'stage': 'stage3_mixed',
                'video_path': video_path,
                'stage2_output_path': stage2_output_path,
                'funscript_output_path': self.output_funscript_path,
                'processing_complete': True,
                'live_tracker_name': self.live_tracker_name,
                'processing_strategy': processing_strategy,
                'settings': {
                    'mixed_mode_quality': self.mixed_mode_quality,
                    'use_live_tracking_chapters': self.use_live_tracking_chapters,
                    'live_tracker_settings': self.mixed_live_tracker_settings
                }
            }
            
            self.processing_active = False
            self.current_stage = None
            
            self.logger.info(f"Stage 3 Mixed processing completed in {processing_time:.1f}s")
            
            return OfflineProcessingResult(
                success=True,
                output_data=mixed_results,
                output_files=output_files,
                checkpoint_data=checkpoint_data,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.processing_active = False
            self.current_stage = None
            self.logger.error(f"Stage 3 Mixed processing error: {e}")
            return OfflineProcessingResult(
                success=False,
                error_message=f"Stage 3 Mixed processing failed: {e}"
            )
    
    def estimate_processing_time(self,
                               stage: OfflineProcessingStage,
                               video_path: str,
                               **kwargs) -> float:
        """Estimate processing time for Stage 3 Mixed."""
        if stage != OfflineProcessingStage.STAGE_3_MIXED:
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
            
            # Mixed processing estimates based on chapter type distribution
            # Assume 30% BJ/HJ (live tracking), 70% other (Stage 2 signal)
            live_tracking_portion = 0.3
            stage2_signal_portion = 0.7
            
            # Live tracking processing rate (slower)
            live_tracking_fps = 25.0
            if self.mixed_mode_quality == "high_precision":
                live_tracking_fps = 15.0
            elif self.mixed_mode_quality == "fast":
                live_tracking_fps = 35.0
            
            # Stage 2 signal processing rate (faster, just signal processing)
            stage2_signal_fps = 200.0  # Very fast for signal-only processing
            
            # Estimate frames for each processing type
            live_tracking_frames = frame_count * live_tracking_portion
            stage2_signal_frames = frame_count * stage2_signal_portion
            
            # Calculate time for each portion
            live_tracking_time = live_tracking_frames / live_tracking_fps
            stage2_signal_time = stage2_signal_frames / stage2_signal_fps
            
            # Add transition processing overhead
            transition_overhead = frame_count * 0.02 / 100.0  # 2% overhead for transitions
            
            estimated_time = live_tracking_time + stage2_signal_time + transition_overhead
            
            # Add buffer for initialization and mixed processor setup
            estimated_time += 45.0  # 45 second buffer
            
            return estimated_time
            
        except Exception as e:
            self.logger.warning(f"Could not estimate processing time: {e}")
            return 400.0  # 6.5 minute fallback
    
    def _analyze_segments_for_mixed_processing(self, segments: List[Any]) -> Dict[str, List[Any]]:
        """Analyze segments to determine processing strategy."""
        live_tracking_segments = []
        stage2_signal_segments = []
        
        for segment in segments:
            # Get position short name from segment
            position_short_name = self._get_segment_position_short_name(segment)
            
            if position_short_name in self.use_live_tracking_chapters:
                live_tracking_segments.append(segment)
            else:
                stage2_signal_segments.append(segment)
        
        return {
            'live_tracking': live_tracking_segments,
            'stage2_signal': stage2_signal_segments,
            'total_segments': len(segments)
        }
    
    def _get_segment_position_short_name(self, segment) -> str:
        """Extract position short name from segment (BJ, HJ, etc.)."""
        # This logic matches the MixedStageProcessor implementation
        if hasattr(segment, 'position_short_name') and isinstance(segment.position_short_name, str):
            return segment.position_short_name
        elif hasattr(segment, 'class_name') and isinstance(segment.class_name, str):
            class_name = segment.class_name
            if class_name == 'Blowjob':
                return 'BJ'
            elif class_name == 'Handjob':
                return 'HJ'
            else:
                return class_name
        elif hasattr(segment, 'major_position'):
            # Map major_position to short name
            position_mapping = {
                'Handjob': 'HJ',
                'Blowjob': 'BJ',
                'Cowgirl': 'CG/Miss',
                'Missionary': 'CG/Miss',
                'Doggy': 'Doggy',
                'No Rating': 'NR'
            }
            return position_mapping.get(segment.major_position, 'NR')
        
        return 'NR'  # Default fallback
    
    def _execute_mixed_stage_processing(self,
                                      mixed_processor,
                                      video_path: str,
                                      preprocessed_video_path: Optional[str],
                                      segments: List[Any],
                                      frame_objects: Dict[int, Any],
                                      processing_strategy: Dict[str, List[Any]],
                                      progress_callback: Optional[Callable] = None,
                                      **kwargs) -> Dict[str, Any]:
        """Execute the mixed stage processing using the MixedStageProcessor."""
        try:
            # This would call the actual mixed processing module
            # For now, return a placeholder result structure
            
            # In the real implementation, this would call something like:
            # result = stage3_mixed_module.perform_mixed_stage_analysis(
            #     video_path=video_path,
            #     preprocessed_video_path=preprocessed_video_path,
            #     segments_objects=segments,
            #     s2_output_data={'frame_objects': frame_objects},
            #     progress_callback=progress_callback,
            #     stop_event=self.stop_event,
            #     parent_logger=self.logger,
            #     mixed_processor=mixed_processor
            # )
            
            # Placeholder implementation
            total_actions = 0
            signal_source_stats = {
                'stage2_signal_frames': len(processing_strategy['stage2_signal']) * 100,  # Estimate
                'live_tracking_frames': len(processing_strategy['live_tracking']) * 100,  # Estimate
                'transition_frames': 50  # Estimate
            }
            
            # Create a simple funscript structure
            funscript_data = {
                'version': '1.0',
                'inverted': False,
                'range': 100,
                'actions': []  # Would be populated by actual processing
            }
            
            # Simulate processing success
            result = {
                'success': True,
                'funscript_data': funscript_data,
                'total_actions': total_actions,
                'processing_efficiency': 0.85,  # Placeholder efficiency metric
                'signal_source_stats': signal_source_stats,
                'debug_data': {
                    'processing_strategy': processing_strategy,
                    'mixed_mode_settings': self.mixed_live_tracker_settings,
                    'live_tracker_used': self.live_tracker_name
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Mixed stage processing error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _load_stage2_results(self, stage2_output_path: str) -> Optional[Dict[str, Any]]:
        """Load Stage 2 results from msgpack file."""
        try:
            import msgpack
            
            with open(stage2_output_path, 'rb') as f:
                data = msgpack.load(f, raw=False)
            
            return {
                'segments': data.get('segments', []),
                'frame_objects': data.get('frame_objects', {}),
                'data_source': 'msgpack'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load Stage 2 results: {e}")
            return None
    
    def _save_funscript_output(self, funscript_data: Any, output_path: str):
        """Save funscript data to file."""
        try:
            import json
            
            # Handle different funscript data formats
            if hasattr(funscript_data, 'save'):
                funscript_data.save(output_path)
            elif isinstance(funscript_data, dict):
                with open(output_path, 'w') as f:
                    json.dump(funscript_data, f, indent=2)
            else:
                # Convert to standard format
                output_data = {
                    'version': '1.0',
                    'inverted': False,
                    'range': 100,
                    'actions': funscript_data if isinstance(funscript_data, list) else []
                }
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
            
            self.logger.info(f"Saved mixed funscript output to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save funscript output: {e}")
            raise
    
    def _save_mixed_debug_data(self, debug_data: Dict[str, Any], output_path: str):
        """Save mixed processing debug data."""
        try:
            import json
            
            with open(output_path, 'w') as f:
                json.dump(debug_data, f, indent=2, default=str)  # default=str handles non-serializable objects
            
            self.logger.info(f"Saved mixed debug data to {output_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save debug data: {e}")
    
    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validate Stage 3 Mixed specific settings."""
        # Call base validation first
        if not super().validate_settings(settings):
            return False
        
        try:
            # Validate chapter configuration
            live_chapters = settings.get('mixed_live_tracking_chapters', self.use_live_tracking_chapters)
            if not isinstance(live_chapters, list) or not live_chapters:
                self.logger.error("Live tracking chapters must be a non-empty list")
                return False
            
            # Validate signal weights
            stage2_weight = settings.get('mixed_stage2_weight', self.stage2_signal_weight)
            if not isinstance(stage2_weight, (int, float)) or not (0.0 <= stage2_weight <= 1.0):
                self.logger.error("Stage 2 signal weight must be between 0.0 and 1.0")
                return False
            
            live_weight = settings.get('mixed_live_weight', self.live_signal_weight)
            if not isinstance(live_weight, (int, float)) or not (0.0 <= live_weight <= 1.0):
                self.logger.error("Live signal weight must be between 0.0 and 1.0")
                return False
            
            # Validate quality mode
            quality_mode = settings.get('mixed_quality_mode', self.mixed_mode_quality)
            if quality_mode not in ['fast', 'balanced', 'high_precision']:
                self.logger.error("Quality mode must be 'fast', 'balanced', or 'high_precision'")
                return False
            
            # Validate transition settings
            transition_frames = settings.get('mixed_transition_frames', self.signal_blend_transition_frames)
            if not isinstance(transition_frames, int) or transition_frames < 1 or transition_frames > 60:
                self.logger.error("Transition frames must be between 1 and 60")
                return False
            
            # Validate ROI settings
            roi_frequency = settings.get('mixed_roi_update_frequency', self.roi_update_frequency)
            if not isinstance(roi_frequency, int) or roi_frequency < 1 or roi_frequency > 120:
                self.logger.error("ROI update frequency must be between 1 and 120 frames")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Settings validation error: {e}")
            return False
    
    def cleanup(self):
        """Clean up Stage 3 Mixed resources."""
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
        self.logger.info("Stage 3 Mixed tracker cleaned up")