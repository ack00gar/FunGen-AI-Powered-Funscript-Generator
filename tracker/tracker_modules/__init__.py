"""
Modular tracker system with auto-discovery.

This module automatically discovers and registers all tracker implementations,
making them available for use in the application without requiring manual
configuration or hardcoded lists.
"""

import os
import sys
import importlib.util
import inspect
import logging
from typing import Dict, List, Type, Optional

try:
    from .core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult, TrackerError
    from .core.base_offline_tracker import BaseOfflineTracker
    from .core.security import (
        TrackerSecurityError, TrackerValidationError, TrackerSandboxError, 
        TrackerAPIViolationError, load_tracker_safely
    )
except ImportError:
    # Fallback for direct execution
    from core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult, TrackerError
    from core.base_offline_tracker import BaseOfflineTracker
    from core.security import (
        TrackerSecurityError, TrackerValidationError, TrackerSandboxError,
        TrackerAPIViolationError, load_tracker_safely
    )


class TrackerRegistry:
    """
    Registry that automatically discovers and manages tracker implementations.
    
    The registry scans the tracker_modules directory and community subdirectory
    for Python files containing BaseTracker subclasses, validates them, and
    makes them available for instantiation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TrackerRegistry")
        self._trackers: Dict[str, Type] = {}
        self._metadata_cache: Dict[str, TrackerMetadata] = {}
        self._folder_map: Dict[str, str] = {}  # tracker_name -> folder_name
        self._discovery_errors: List[str] = []
        
        # Perform initial discovery
        self._discover_trackers()
        
        if self._trackers:
            # Single concise summary - debug level to reduce startup noise
            self.logger.debug(f"Discovered {len(self._trackers)} trackers")
        else:
            self.logger.warning("No trackers discovered!")
            
        if self._discovery_errors:
            self.logger.warning(f"Discovery errors: {len(self._discovery_errors)}")
    
    def _discover_trackers(self):
        """Auto-discover tracker modules in the tracker_modules subdirectories."""
        tracker_dir = os.path.dirname(__file__)
        
        # Scan live trackers subdirectory
        live_dir = os.path.join(tracker_dir, 'live')
        if os.path.exists(live_dir):
            self._scan_directory(live_dir, folder_name='live', is_community=False)
        
        # Scan offline trackers subdirectory
        offline_dir = os.path.join(tracker_dir, 'offline')
        if os.path.exists(offline_dir):
            self._scan_directory(offline_dir, folder_name='offline', is_community=False)
        
        # Scan experimental trackers subdirectory
        experimental_dir = os.path.join(tracker_dir, 'experimental')
        if os.path.exists(experimental_dir):
            self._scan_directory(experimental_dir, folder_name='experimental', is_community=False)
        
        # Scan community subdirectory
        community_dir = os.path.join(tracker_dir, 'community')
        if os.path.exists(community_dir):
            self._scan_directory(community_dir, folder_name='community', is_community=True)
        
        self.logger.debug(f"Discovery complete. Found {len(self._trackers)} trackers.")
    
    def _scan_directory(self, directory: str, folder_name: str, is_community: bool = False):
        """Scan a directory for tracker modules."""
        try:
            for filename in os.listdir(directory):
                if filename.endswith('.py') and filename not in ['__init__.py']:
                    file_path = os.path.join(directory, filename)
                    self._load_tracker_module(file_path, filename, folder_name, is_community)
        except OSError as e:
            error_msg = f"Failed to scan directory {directory}: {e}"
            self._discovery_errors.append(error_msg)
            self.logger.error(error_msg)
    
    def _load_tracker_module(self, file_path: str, filename: str, folder_name: str, is_community: bool):
        """Load and validate a single tracker module with security checks."""
        try:
            # Use secure loading with validation and sandboxing
            tracker_class = load_tracker_safely(file_path, filename)
            if tracker_class:
                self._register_tracker(tracker_class, folder_name, is_community, file_path)
            else:
                self.logger.debug(f"No valid tracker classes found in {filename}")
                
        except TrackerSecurityError as e:
            # Security violations are critical - log as error
            error_msg = f"SECURITY VIOLATION in {filename}: {e}"
            self._discovery_errors.append(error_msg)
            self.logger.error(error_msg)
        except Exception as e:
            # Other errors - log as warnings for now
            error_msg = f"Failed to load tracker module {filename}: {e}"
            self._discovery_errors.append(error_msg)
            self.logger.warning(error_msg)
    
    
    def _register_tracker(self, tracker_class: Type, folder_name: str, is_community: bool, file_path: str):
        """Register a validated tracker class with resource management."""
        temp_instance = None
        try:
            # Validate by attempting to access metadata
            # Note: We create a temporary instance just to get metadata
            # This ensures the metadata property is properly implemented
            temp_instance = tracker_class()
            metadata = temp_instance.metadata
            
            if not isinstance(metadata, TrackerMetadata):
                raise TrackerValidationError(f"Tracker metadata must be TrackerMetadata instance")
            
            if not metadata.name:
                raise TrackerValidationError(f"Tracker name cannot be empty")
            
            # Check for name conflicts
            if metadata.name in self._trackers:
                existing_metadata = self._metadata_cache[metadata.name]
                self.logger.warning(
                    f"Tracker name conflict: '{metadata.name}' "
                    f"(existing: {existing_metadata.display_name}, "
                    f"new: {metadata.display_name}). Skipping new tracker."
                )
                return
            
            # Register the tracker
            self._trackers[metadata.name] = tracker_class
            self._metadata_cache[metadata.name] = metadata
            self._folder_map[metadata.name] = folder_name
            
            # Log at debug level to reduce verbosity
            category_prefix = "[Community] " if is_community else ""
            self.logger.debug(
                f"Registered: {category_prefix}{metadata.display_name} ({metadata.name})"
            )
            
        except TrackerSecurityError as e:
            error_msg = f"SECURITY VIOLATION during registration of {tracker_class.__name__}: {e}"
            self._discovery_errors.append(error_msg)
            self.logger.error(error_msg)
        except Exception as e:
            error_msg = f"Failed to register tracker {tracker_class.__name__}: {e}"
            self._discovery_errors.append(error_msg)
            self.logger.error(error_msg)
        finally:
            # Resource cleanup - ensure temp instance is properly cleaned up
            if temp_instance:
                try:
                    if hasattr(temp_instance, 'cleanup'):
                        temp_instance.cleanup()
                    # Clear references to help garbage collection
                    temp_instance = None
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup temp instance: {cleanup_error}")
    
    def get_tracker(self, name: str) -> Optional[Type]:
        """
        Get tracker class by name.
        
        Args:
            name: Internal tracker name (from metadata.name)
        
        Returns:
            Type: Tracker class, or None if not found
        """
        return self._trackers.get(name)
    
    def create_tracker(self, name: str) -> Optional:
        """
        Create a new instance of the named tracker.
        
        Args:
            name: Internal tracker name (from metadata.name)
        
        Returns:
            BaseTracker or BaseOfflineTracker: New tracker instance, or None if not found
        """
        tracker_class = self.get_tracker(name)
        if tracker_class:
            try:
                return tracker_class()
            except Exception as e:
                self.logger.error(f"Failed to create tracker instance {name}: {e}")
                return None
        return None
    
    def list_trackers(self, category: Optional[str] = None) -> List[TrackerMetadata]:
        """
        List all discovered tracker metadata.
        
        Args:
            category: Optional category filter ("live", "offline", etc.)
        
        Returns:
            List[TrackerMetadata]: List of tracker metadata
        """
        trackers = list(self._metadata_cache.values())
        
        if category:
            trackers = [t for t in trackers if t.category == category]
        
        # Custom category priority: live first, then offline, then community
        # Within each category, sort alphabetically by display name
        category_priority = {'live': 1, 'offline': 2, 'community': 3}
        return sorted(trackers, key=lambda t: (category_priority.get(t.category, 999), t.display_name))
    
    def get_metadata(self, name: str) -> Optional[TrackerMetadata]:
        """
        Get metadata for a specific tracker.
        
        Args:
            name: Internal tracker name
        
        Returns:
            TrackerMetadata: Tracker metadata, or None if not found
        """
        return self._metadata_cache.get(name)
    
    def get_available_names(self) -> List[str]:
        """Get list of all available tracker names."""
        return list(self._trackers.keys())
    
    def get_discovery_errors(self) -> List[str]:
        """Get list of errors encountered during discovery."""
        return self._discovery_errors.copy()
    
    def get_tracker_folder(self, name: str) -> Optional[str]:
        """Get folder name for a specific tracker."""
        return self._folder_map.get(name)
    
    def reload_trackers(self):
        """Reload all trackers (useful for development)."""
        self.logger.debug("Reloading trackers...")
        self._trackers.clear()
        self._metadata_cache.clear()
        self._folder_map.clear()
        self._discovery_errors.clear()
        self._discover_trackers()


# Global registry instance - automatically discovers trackers on import
tracker_registry = TrackerRegistry()


def get_tracker_registry() -> TrackerRegistry:
    """Get the global tracker registry instance."""
    return tracker_registry


def list_available_trackers(category: Optional[str] = None) -> List[TrackerMetadata]:
    """
    Convenience function to list available trackers.
    
    Args:
        category: Optional category filter
    
    Returns:
        List[TrackerMetadata]: Available trackers
    """
    return tracker_registry.list_trackers(category)


def create_tracker(name: str):
    """
    Convenience function to create a tracker instance.
    
    Args:
        name: Internal tracker name
    
    Returns:
        BaseTracker or BaseOfflineTracker: New tracker instance, or None if not found
    """
    return tracker_registry.create_tracker(name)


# Export commonly used classes for easy importing
__all__ = [
    'TrackerRegistry', 
    'tracker_registry',
    'BaseTracker',
    'BaseOfflineTracker',
    'TrackerMetadata', 
    'TrackerResult',
    'TrackerError',
    'TrackerSecurityError',
    'TrackerValidationError', 
    'TrackerSandboxError',
    'TrackerAPIViolationError',
    'get_tracker_registry',
    'list_available_trackers',
    'create_tracker'
]