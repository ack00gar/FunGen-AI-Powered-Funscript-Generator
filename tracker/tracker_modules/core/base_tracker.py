"""
Base tracker interface and common types for modular tracker system.

This module provides the foundation for the plugin-based tracker architecture,
allowing community developers to easily create new tracking algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import logging
import cv2


@dataclass
class StageDefinition:
    """Definition of a processing stage in a tracker."""
    stage_number: int
    name: str
    description: str
    produces_funscript: bool = False
    requires_previous: bool = False
    output_type: str = "analysis"  # "analysis", "funscript", "segmentation", "mixed"


class TrackerMetadata:
    """Metadata describing a tracker for UI display and discovery."""
    
    def __init__(self, 
                 name: str,
                 display_name: str, 
                 description: str,
                 category: str,
                 version: str,
                 author: str,
                 tags: Optional[List[str]] = None,
                 requires_roi: bool = False,
                 supports_dual_axis: bool = True,
                 stages: Optional[List[StageDefinition]] = None,
                 properties: Optional[Dict[str, Any]] = None):
        """
        Initialize tracker metadata.
        
        Args:
            name: Internal identifier (must be unique)
            display_name: Human-readable name for UI display
            description: Short description of the tracker's purpose
            category: Category ("live", "offline", "experimental", "community")
            version: Semantic version (e.g., "1.0.0")
            author: Author name or organization
            tags: Optional list of tags for filtering/search
            requires_roi: Whether tracker needs ROI selection
            supports_dual_axis: Whether tracker supports dual-axis output
            stages: List of processing stages this tracker uses
            properties: Dictionary of tracker-specific properties/capabilities
        """
        self.name = name
        self.display_name = display_name
        self.description = description
        self.category = category
        self.version = version
        self.author = author
        self.tags = tags or []
        self.requires_roi = requires_roi
        self.supports_dual_axis = supports_dual_axis
        self.stages = stages or []
        self.properties = properties or {}
        
        # Auto-generate properties from name for backward compatibility
        if not self.properties and self.stages:
            self._auto_generate_properties()
    
    def _auto_generate_properties(self):
        """Auto-generate properties based on stages for backward compatibility."""
        self.properties = {
            "num_stages": len(self.stages),
            "supports_batch": self.category in ["live", "offline"] and not self.requires_roi,
            "supports_realtime": self.category == "live",
            "requires_intervention": self.requires_roi
        }
        
        # Check which stages produce funscripts
        for stage in self.stages:
            if stage.produces_funscript:
                self.properties[f"produces_funscript_in_stage{stage.stage_number}"] = True
                
    def has_property(self, property_name: str) -> bool:
        """Check if tracker has a specific property."""
        return property_name in self.properties and self.properties[property_name]
    
    def get_property(self, property_name: str, default: Any = None) -> Any:
        """Get a property value with optional default."""
        return self.properties.get(property_name, default)
    
    def get_stage(self, stage_number: int) -> Optional[StageDefinition]:
        """Get a specific stage definition by number."""
        for stage in self.stages:
            if stage.stage_number == stage_number:
                return stage
        return None


class TrackerResult:
    """Result returned by tracker processing with backward compatibility for tuple access."""
    
    def __init__(self, 
                 processed_frame: np.ndarray,
                 action_log: Optional[List[Dict]] = None,
                 debug_info: Optional[Dict] = None,
                 status_message: Optional[str] = None):
        """
        Initialize tracker result.
        
        Args:
            processed_frame: Frame with visual overlays applied
            action_log: List of funscript actions generated this frame
            debug_info: Optional debug information for display
            status_message: Optional status message for UI
        """
        self.processed_frame = processed_frame
        self.action_log = action_log
        self.debug_info = debug_info
        self.status_message = status_message
    
    def __getitem__(self, index: int):
        """Support tuple-like access for backward compatibility with video processor."""
        if index == 0:
            return self.processed_frame
        elif index == 1:
            return self.action_log
        else:
            raise IndexError(f"TrackerResult index out of range: {index} (only 0 and 1 supported)")
    
    def __iter__(self):
        """Support tuple-like unpacking for backward compatibility."""
        yield self.processed_frame
        yield self.action_log
    
    def __len__(self):
        """Support len() for tuple-like behavior."""
        return 2


class BaseTracker(ABC):
    """
    Base class for all tracking algorithms.
    
    Community developers should inherit from this class and implement all
    abstract methods. The tracker will be automatically discovered and
    made available in the UI.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"Tracker.{self.__class__.__name__}")
        self.app = None
        self.tracking_active = False
        self._initialized = False
    
    @property
    @abstractmethod
    def metadata(self) -> TrackerMetadata:
        """
        Return tracker metadata for UI display and discovery.
        
        This property must be implemented as a class property that returns
        TrackerMetadata with all required information about this tracker.
        
        Returns:
            TrackerMetadata: Metadata describing this tracker
        """
        pass
    
    @abstractmethod
    def initialize(self, app_instance, **kwargs) -> bool:
        """
        Initialize the tracker with app instance and settings.
        
        This method is called once when the tracker is selected. Use it to:
        - Store reference to app instance
        - Initialize internal state
        - Set up required resources
        - Validate configuration
        
        Args:
            app_instance: Main application instance with access to settings,
                         funscript, video processor, etc.
            **kwargs: Additional initialization parameters
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod  
    def process_frame(self, frame: np.ndarray, frame_time_ms: int, 
                     frame_index: Optional[int] = None) -> TrackerResult:
        """
        Process a single video frame and return tracking results.
        
        This is the core method that processes each frame of video and
        generates funscript actions. The method should:
        - Analyze the frame for motion/features
        - Update internal tracking state
        - Generate funscript actions if tracking is active
        - Apply visual overlays to the frame
        - Return results via TrackerResult
        
        Args:
            frame: Video frame as numpy array (H, W, C) in BGR format
            frame_time_ms: Timestamp of frame in milliseconds
            frame_index: Optional frame number in sequence
        
        Returns:
            TrackerResult: Processing results including modified frame and actions
        """
        pass
    
    @abstractmethod
    def start_tracking(self) -> bool:
        """
        Start the tracking session.
        
        Called when user clicks "Start Live Tracking". Use this to:
        - Initialize tracking state
        - Reset internal buffers
        - Start generating funscript actions
        - Set tracking_active = True
        
        Returns:
            bool: True if tracking started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def stop_tracking(self) -> bool:
        """
        Stop the tracking session.
        
        Called when user clicks "Stop" or switches modes. Use this to:
        - Stop generating funscript actions  
        - Set tracking_active = False
        - Preserve state for potential restart
        
        Returns:
            bool: True if tracking stopped successfully, False otherwise
        """
        pass
    
    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Validate tracker-specific settings.
        
        Override this method to validate custom settings before they are
        applied to the tracker. This is called when settings change.
        
        Args:
            settings: Dictionary of setting name -> value pairs
        
        Returns:
            bool: True if settings are valid, False otherwise
        """
        return True
    
    def cleanup(self):
        """
        Clean up resources when tracker is being destroyed.
        
        Override this method to:
        - Release allocated memory
        - Close file handles
        - Clean up OpenCV objects
        - Stop background threads
        """
        pass
    
    def get_status_info(self) -> Dict[str, Any]:
        """
        Get current status information for UI display.
        
        Override this method to provide custom status information
        that will be displayed in the tracker status panel.
        
        Returns:
            Dict: Status information key-value pairs
        """
        return {
            "tracker": self.metadata.display_name,
            "active": self.tracking_active,
            "initialized": self._initialized
        }
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> bool:
        """
        Set region of interest for trackers that support it.
        
        Override this method if your tracker supports ROI selection.
        
        Args:
            roi: Region of interest as (x, y, width, height)
        
        Returns:
            bool: True if ROI was set successfully, False otherwise
        """
        return False
    
    def get_settings_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema describing tracker-specific settings.
        
        Override this method to define custom settings that will be
        automatically generated in the UI. Use JSON Schema format.
        
        Returns:
            Dict: JSON schema for tracker settings
        """
        return {}
    
    def _draw_tracking_indicator(self, frame: np.ndarray) -> None:
        """
        Draw simple 'Tracking' indicator in bottom right when active.
        
        Args:
            frame: Frame to draw on (modified in-place)
        """
        if self.tracking_active:
            h, w = frame.shape[:2]
            cv2.putText(frame, "Tracking", (w - 80, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


class TrackerError(Exception):
    """Exception raised by tracker operations."""
    pass


class TrackerInitializationError(TrackerError):
    """Exception raised when tracker initialization fails."""
    pass


class TrackerProcessingError(TrackerError):
    """Exception raised when frame processing fails."""
    pass