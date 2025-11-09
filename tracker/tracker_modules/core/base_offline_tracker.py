#!/usr/bin/env python3
"""
Base offline tracker interface for batch video processing.

This module provides the foundation for offline tracking algorithms that process
entire videos in batch mode, potentially across multiple processing stages with
dependencies, checkpointing, and multiprocessing support.

Author: Offline Tracker Architecture
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Callable
from multiprocessing import Event
import logging
from enum import Enum

try:
    from .base_tracker import TrackerMetadata, TrackerError
except ImportError:
    from base_tracker import TrackerMetadata, TrackerError


class OfflineProcessingStage(Enum):
    """Enumeration of offline processing stages."""
    STAGE_1 = "stage1"  # Object detection
    STAGE_2 = "stage2"  # Contact analysis  
    STAGE_3 = "stage3"  # Optical flow analysis
    STAGE_3_MIXED = "stage3_mixed"  # Mixed approach


class OfflineProcessingResult:
    """Result from offline processing stage."""
    
    def __init__(self,
                 success: bool,
                 output_data: Optional[Dict[str, Any]] = None,
                 output_files: Optional[Dict[str, str]] = None,
                 checkpoint_data: Optional[Dict[str, Any]] = None,
                 error_message: Optional[str] = None,
                 performance_metrics: Optional[Dict[str, Any]] = None):
        """
        Initialize offline processing result.
        
        Args:
            success: Whether processing completed successfully
            output_data: Processed data results
            output_files: Generated output files (name -> path mapping)
            checkpoint_data: Data for resuming interrupted processing
            error_message: Error description if processing failed
            performance_metrics: Performance statistics
        """
        self.success = success
        self.output_data = output_data or {}
        self.output_files = output_files or {}
        self.checkpoint_data = checkpoint_data or {}
        self.error_message = error_message
        self.performance_metrics = performance_metrics or {}


class BaseOfflineTracker(ABC):
    """
    Base class for offline tracking algorithms.
    
    Offline trackers process entire videos in batch mode and may have
    multiple processing stages with dependencies. They support features like:
    - Multi-stage processing pipelines
    - Checkpointing and resumption
    - Multiprocessing for performance
    - Progress callbacks
    - Intermediate file management
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"OfflineTracker.{self.__class__.__name__}")
        self._initialized = False
        
        # Processing state
        self.stop_event: Optional[Event] = None
        self.current_stage: Optional[OfflineProcessingStage] = None
        self.processing_active = False
        
        # Configuration
        self.num_workers = 4
        self.enable_checkpointing = True
        self.intermediate_cleanup = True
    
    @property
    @abstractmethod
    def metadata(self) -> TrackerMetadata:
        """
        Return tracker metadata for UI display and discovery.
        
        Returns:
            TrackerMetadata: Metadata describing this tracker
        """
        pass
    
    @property
    @abstractmethod
    def processing_stages(self) -> List[OfflineProcessingStage]:
        """
        Return list of processing stages this tracker implements.
        
        Returns:
            List[OfflineProcessingStage]: Stages this tracker can execute
        """
        pass
    
    @property
    @abstractmethod
    def stage_dependencies(self) -> Dict[OfflineProcessingStage, List[OfflineProcessingStage]]:
        """
        Return dependencies between processing stages.
        
        Returns:
            Dict: Mapping of stage -> list of required prerequisite stages
        """
        pass
    
    @abstractmethod
    def initialize(self, app_instance, **kwargs) -> bool:
        """
        Initialize the offline tracker.
        
        Args:
            app_instance: Main application instance
            **kwargs: Additional initialization parameters
        
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    def can_resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """
        Check if processing can be resumed from checkpoint data.
        
        Args:
            checkpoint_data: Previously saved checkpoint data
        
        Returns:
            bool: True if resumption is possible
        """
        pass
    
    @abstractmethod
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
        Process a specific stage of the offline analysis.
        
        Args:
            stage: Processing stage to execute
            video_path: Path to input video file
            input_data: Input data from previous stages
            input_files: Input files from previous stages  
            output_directory: Directory for output files
            progress_callback: Function to report progress
            frame_range: Optional frame range to process
            resume_data: Data for resuming interrupted processing
            **kwargs: Additional stage-specific parameters
        
        Returns:
            OfflineProcessingResult: Processing results
        """
        pass
    
    @abstractmethod
    def estimate_processing_time(self,
                               stage: OfflineProcessingStage,
                               video_path: str,
                               **kwargs) -> float:
        """
        Estimate processing time for a stage.
        
        Args:
            stage: Processing stage
            video_path: Path to video file
            **kwargs: Additional parameters
        
        Returns:
            float: Estimated processing time in seconds
        """
        pass
    
    def validate_dependencies(self,
                            stage: OfflineProcessingStage,
                            available_data: Dict[str, Any],
                            available_files: Dict[str, str]) -> bool:
        """
        Validate that dependencies for a stage are satisfied.

        Args:
            stage: Stage to validate
            available_data: Available input data
            available_files: Available input files

        Returns:
            bool: True if dependencies are satisfied
        """
        dependencies = self.stage_dependencies.get(stage, [])

        for dep_stage in dependencies:
            # Check if required stage outputs are available
            stage_key = dep_stage.value
            found_in_data = stage_key in available_data
            found_in_files = stage_key in available_files

            if not found_in_data and not found_in_files:
                self.logger.error(f"Missing dependency {stage_key} for stage {stage.value}")
                return False

            # Validate data content if found
            if found_in_data:
                data = available_data[stage_key]
                if data is None:
                    self.logger.error(f"Dependency {stage_key} data is None")
                    return False
                # Check if it's an empty dict/list (likely invalid)
                if isinstance(data, (dict, list)) and len(data) == 0:
                    self.logger.warning(f"Dependency {stage_key} has empty data structure")

            # Validate file exists and is readable if found
            if found_in_files:
                import os
                file_path = available_files[stage_key]
                if not file_path or not os.path.exists(file_path):
                    self.logger.error(f"Dependency {stage_key} file does not exist: {file_path}")
                    return False
                if not os.access(file_path, os.R_OK):
                    self.logger.error(f"Dependency {stage_key} file is not readable: {file_path}")
                    return False
                # Check file is not empty
                if os.path.getsize(file_path) == 0:
                    self.logger.error(f"Dependency {stage_key} file is empty: {file_path}")
                    return False

        return True
    
    def set_stop_event(self, stop_event: Event):
        """Set the stop event for canceling processing."""
        self.stop_event = stop_event
    
    def stop_processing(self):
        """Signal processing to stop."""
        if self.stop_event:
            self.stop_event.set()
        self.processing_active = False

    def should_stop(self) -> bool:
        """
        Check if processing should be stopped.

        Returns:
            bool: True if stop has been requested
        """
        if self.stop_event and self.stop_event.is_set():
            return True
        return not self.processing_active
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """
        Get current checkpoint data for resumption.
        
        Override this method to provide custom checkpoint data.
        
        Returns:
            Dict: Checkpoint data
        """
        return {
            'current_stage': self.current_stage.value if self.current_stage else None,
            'processing_active': self.processing_active,
            'class_name': self.__class__.__name__
        }
    
    def cleanup_intermediate_files(self, file_paths: List[str]):
        """
        Clean up intermediate files if cleanup is enabled.
        
        Args:
            file_paths: List of file paths to clean up
        """
        if not self.intermediate_cleanup:
            return
        
        import os
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.debug(f"Cleaned up intermediate file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up {file_path}: {e}")
    
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get current progress information for UI display.

        Returns:
            Dict: Progress information including stage, status, and settings
        """
        info = {
            'current_stage': self.current_stage.value if self.current_stage else None,
            'current_stage_name': self.current_stage.name if self.current_stage else None,
            'processing_active': self.processing_active,
            'tracker_name': self.metadata.display_name,
            'tracker_category': self.metadata.category,
            'num_workers': self.num_workers,
            'checkpointing_enabled': self.enable_checkpointing,
            'intermediate_cleanup': self.intermediate_cleanup
        }

        # Add stop event status if available
        if self.stop_event:
            info['stop_requested'] = self.stop_event.is_set()

        return info
    
    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Validate offline tracker settings.
        
        Args:
            settings: Settings to validate
        
        Returns:
            bool: True if settings are valid
        """
        # Validate common settings
        num_workers = settings.get('num_workers', self.num_workers)
        if not isinstance(num_workers, int) or num_workers < 1 or num_workers > 32:
            self.logger.error("Number of workers must be between 1 and 32")
            return False
        
        return True
    
    def cleanup(self):
        """Clean up resources when tracker is being destroyed."""
        self.stop_processing()
        # Reduced logging verbosity
        self.logger.debug(f"{self.__class__.__name__} cleanup complete")


class OfflineTrackerInitializationError(TrackerError):
    """Exception raised when offline tracker initialization fails."""
    pass


class OfflineTrackerProcessingError(TrackerError):
    """Exception raised when offline processing fails."""
    pass


class OfflineTrackerDependencyError(TrackerError):
    """Exception raised when stage dependencies are not satisfied."""
    pass