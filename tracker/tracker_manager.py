"""
Tracker Manager - Main interface for the modular tracker system.

This class replaces the monolithic ROITracker and provides a clean interface
for managing different tracking algorithms. It handles tracker lifecycle,
switching between trackers, and provides a unified API that matches the
original ROITracker interface for seamless integration.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from tracker_modules import tracker_registry, BaseTracker, TrackerMetadata, TrackerResult
from tracker_modules.base_tracker import TrackerError, TrackerInitializationError, TrackerProcessingError


class TrackerManager:
    """
    Manager for the modular tracker system.
    
    This class provides the same interface as the original ROITracker but
    delegates to pluggable tracker implementations. It handles:
    - Tracker discovery and selection
    - Tracker lifecycle management
    - Unified API for the application
    - Error handling and fallback
    """
    
    def __init__(self, app_instance, **kwargs):
        """
        Initialize the tracker manager.
        
        Args:
            app_instance: Main application instance
            **kwargs: Additional initialization parameters
        """
        self.app = app_instance
        self.logger = logging.getLogger("TrackerManager")
        
        # Current tracker state
        self.current_tracker: Optional[BaseTracker] = None
        self.current_tracker_name: Optional[str] = None
        
        # Compatibility properties with original ROITracker
        self.tracking_active = False
        self.show_masks = kwargs.get('show_masks', True)
        self.show_roi = kwargs.get('show_roi', True)
        
        # Default tracker (can be overridden by settings)
        self.default_tracker_name = kwargs.get('default_tracker', 'oscillation_experimental_2')
        
        self.logger.info(f"TrackerManager initialized with {len(tracker_registry.get_available_names())} available trackers")
        
        # Report any discovery errors
        errors = tracker_registry.get_discovery_errors()
        if errors:
            self.logger.warning(f"Tracker discovery errors: {len(errors)}")
            for error in errors:
                self.logger.debug(f"Discovery error: {error}")
    
    def set_tracking_mode(self, mode: str) -> bool:
        """
        Set the tracking mode (for compatibility with existing code).
        
        This method maps legacy mode strings to tracker names and switches
        to the appropriate tracker implementation.
        
        Args:
            mode: Legacy mode string (e.g., "OSCILLATION_DETECTOR")
        
        Returns:
            bool: True if mode was set successfully
        """
        # Map legacy mode strings to tracker names
        mode_mapping = {
            "OSCILLATION_DETECTOR": "oscillation_experimental",
            "OSCILLATION_DETECTOR_LEGACY": "oscillation_legacy", 
            "OSCILLATION_DETECTOR_EXPERIMENTAL_2": "oscillation_experimental_2",
            "YOLO_ROI": "yolo_roi",
            "USER_FIXED_ROI": "user_roi"
        }
        
        tracker_name = mode_mapping.get(mode)
        if not tracker_name:
            self.logger.error(f"Unknown tracking mode: {mode}")
            return False
        
        return self.set_tracker(tracker_name)
    
    def set_tracker(self, tracker_name: str, **settings) -> bool:
        """
        Switch to a different tracker implementation.
        
        Args:
            tracker_name: Name of the tracker to switch to
            **settings: Additional settings to pass to the tracker
        
        Returns:
            bool: True if tracker was set successfully
        """
        if tracker_name == self.current_tracker_name:
            self.logger.debug(f"Already using tracker: {tracker_name}")
            return True
        
        # Validate tracker exists
        if not tracker_registry.get_tracker(tracker_name):
            self.logger.error(f"Tracker not found: {tracker_name}")
            return False
        
        # Stop current tracker if running
        if self.current_tracker and self.tracking_active:
            self.stop_tracking()
        
        # Cleanup previous tracker
        if self.current_tracker:
            self.logger.debug(f"Cleaning up previous tracker: {self.current_tracker_name}")
            try:
                self.current_tracker.cleanup()
            except Exception as e:
                self.logger.warning(f"Error during tracker cleanup: {e}")
        
        # Create and initialize new tracker
        try:
            new_tracker = tracker_registry.create_tracker(tracker_name)
            if not new_tracker:
                raise TrackerInitializationError(f"Failed to create tracker instance: {tracker_name}")
            
            if not new_tracker.initialize(self.app, **settings):
                raise TrackerInitializationError(f"Tracker initialization failed: {tracker_name}")
            
            self.current_tracker = new_tracker
            self.current_tracker_name = tracker_name
            
            metadata = tracker_registry.get_metadata(tracker_name)
            self.logger.info(f"Switched to tracker: {metadata.display_name if metadata else tracker_name}")
            
            # Clear any overlays when switching trackers
            if hasattr(self.app, 'clear_all_overlays_and_ui_drawings'):
                self.app.clear_all_overlays_and_ui_drawings()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch to tracker {tracker_name}: {e}")
            self.current_tracker = None
            self.current_tracker_name = None
            return False
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int, 
                     frame_index: Optional[int] = None, 
                     min_write_frame_id: Optional[int] = None) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """
        Process a frame using the current tracker.
        
        This method maintains compatibility with the original ROITracker interface.
        
        Args:
            frame: Input video frame
            frame_time_ms: Frame timestamp in milliseconds
            frame_index: Optional frame index
            min_write_frame_id: Optional minimum frame ID for writing (unused)
        
        Returns:
            Tuple[np.ndarray, Optional[List[Dict]]]: Processed frame and action log
        """
        if not self.current_tracker:
            # No tracker selected - return frame unchanged
            self.logger.debug("No tracker selected, returning frame unchanged")
            return frame, None
        
        try:
            result = self.current_tracker.process_frame(frame, frame_time_ms, frame_index)
            
            if not isinstance(result, TrackerResult):
                raise TrackerProcessingError("Tracker must return TrackerResult instance")
            
            # Update status if provided
            if result.status_message:
                self.logger.debug(f"Tracker status: {result.status_message}")
            
            return result.processed_frame, result.action_log
            
        except Exception as e:
            self.logger.error(f"Frame processing error in {self.current_tracker_name}: {e}")
            # Return original frame on error to prevent crash
            return frame, None
    
    def start_tracking(self) -> bool:
        """
        Start tracking with the current tracker.
        
        Returns:
            bool: True if tracking started successfully
        """
        if not self.current_tracker:
            self.logger.error("Cannot start tracking: no tracker selected")
            return False
        
        try:
            if self.current_tracker.start_tracking():
                self.tracking_active = True
                metadata = tracker_registry.get_metadata(self.current_tracker_name)
                self.logger.info(f"Started tracking with: {metadata.display_name if metadata else self.current_tracker_name}")
                return True
            else:
                self.logger.error("Tracker failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting tracker: {e}")
            return False
    
    def stop_tracking(self) -> bool:
        """
        Stop tracking with the current tracker.
        
        Returns:
            bool: True if tracking stopped successfully
        """
        if not self.current_tracker:
            self.logger.debug("No tracker to stop")
            return True
        
        try:
            result = self.current_tracker.stop_tracking()
            self.tracking_active = False
            self.logger.info("Tracking stopped")
            return result
            
        except Exception as e:
            self.logger.error(f"Error stopping tracker: {e}")
            self.tracking_active = False
            return False
    
    def list_available_trackers(self, category: Optional[str] = None) -> List[TrackerMetadata]:
        """
        Get list of all available trackers for UI display.
        
        Args:
            category: Optional category filter
        
        Returns:
            List[TrackerMetadata]: Available tracker metadata
        """
        return tracker_registry.list_trackers(category)
    
    def get_current_tracker_info(self) -> Dict[str, Any]:
        """
        Get information about the current tracker.
        
        Returns:
            Dict: Current tracker information for UI display
        """
        if not self.current_tracker:
            return {
                "name": None,
                "display_name": "No tracker selected",
                "active": False,
                "status": "idle"
            }
        
        metadata = tracker_registry.get_metadata(self.current_tracker_name)
        status_info = self.current_tracker.get_status_info()
        
        return {
            "name": self.current_tracker_name,
            "display_name": metadata.display_name if metadata else self.current_tracker_name,
            "active": self.tracking_active,
            "status": "tracking" if self.tracking_active else "ready",
            "metadata": metadata,
            "custom_status": status_info
        }
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> bool:
        """
        Set region of interest for current tracker if supported.
        
        Args:
            roi: Region of interest as (x, y, width, height)
        
        Returns:
            bool: True if ROI was set successfully
        """
        if not self.current_tracker:
            return False
        
        return self.current_tracker.set_roi(roi)
    
    def cleanup(self):
        """Clean up all resources."""
        if self.tracking_active:
            self.stop_tracking()
        
        if self.current_tracker:
            try:
                self.current_tracker.cleanup()
            except Exception as e:
                self.logger.warning(f"Error during final cleanup: {e}")
        
        self.current_tracker = None
        self.current_tracker_name = None
        self.logger.info("TrackerManager cleanup complete")
    
    # Compatibility properties and methods for existing code
    @property
    def tracking_mode(self) -> str:
        """Get current tracking mode (for compatibility)."""
        if not self.current_tracker_name:
            return "NONE"
        
        # Map tracker names back to legacy mode strings
        name_to_mode = {
            "oscillation_experimental": "OSCILLATION_DETECTOR",
            "oscillation_legacy": "OSCILLATION_DETECTOR_LEGACY",
            "oscillation_experimental_2": "OSCILLATION_DETECTOR_EXPERIMENTAL_2",
            "yolo_roi": "YOLO_ROI",
            "user_roi": "USER_FIXED_ROI"
        }
        
        return name_to_mode.get(self.current_tracker_name, self.current_tracker_name.upper())
    
    def update_oscillation_grid_size(self):
        """Compatibility method - delegate to current tracker if applicable."""
        if self.current_tracker and hasattr(self.current_tracker, 'update_oscillation_grid_size'):
            self.current_tracker.update_oscillation_grid_size()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Avoid errors during destruction