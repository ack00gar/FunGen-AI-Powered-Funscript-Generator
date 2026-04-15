"""
Feature Detection Utilities

Detects available features based on folder presence and dependencies.
Used to enable/disable features based on user tier (free vs supporter).
"""

import os
import logging
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass


@dataclass
class FeatureInfo:
    """Information about a feature."""
    name: str
    display_name: str
    description: str
    folder_path: str
    required_files: List[str]
    dependencies: List[str]
    tier: str  # "free", "supporter", "dev"
    enabled: bool = False
    available: bool = False


class FeatureDetector:
    """
    Detects available features based on folder structure and dependencies.
    
    This enables a tiered feature system where:
    - Free users get basic features
    - Project supporters get premium features by adding folders
    - Developers get all features
    """
    
    def __init__(self, app_root: Optional[str] = None):
        self.app_root = Path(app_root) if app_root else Path.cwd()
        self.logger = logging.getLogger(__name__)
        
        # Feature definitions
        self.features = self._define_features()
        
        # Detection results
        self.available_features: Set[str] = set()
        self.enabled_features: Set[str] = set()
        
        # Run initial detection
        self.detect_all_features()
    
    def _define_features(self) -> Dict[str, FeatureInfo]:
        """Define all available features."""
        features = {}
        
        # Device Control (Supporter)
        features["device_control"] = FeatureInfo(
            name="device_control",
            display_name="Device Control",
            description="Control adult toys with live tracking and funscript playback",
            folder_path="device_control",
            required_files=[
                "device_control/__init__.py",
                "device_control/device_manager.py",
                "device_control/backends/buttplug_backend_direct.py",
                "device_control/backends/osr_backend.py"
            ],
            dependencies=[],
            tier="supporter"
        )

        # Video Streamer / VR Stream Mode (Supporter)
        features["streamer"] = FeatureInfo(
            name="streamer",
            display_name="Video Streamer",
            description="Stream video to browsers/VR headsets with frame-perfect sync, zoom/pan controls, and interactive device control",
            folder_path="streamer",
            required_files=[
                "streamer/__init__.py",
                "streamer/sync_server.py",
                "streamer/video_http_server.py",
                "streamer/sync_media_handlers.py",
            ],
            dependencies=[],
            tier="supporter"
        )
        
        # patreon_features modules (live trackers, live capture, batch queue) now
        # ship in core. The feature key is kept for back-compat with the many
        # is_feature_available("patreon_features") gates scattered in the UI;
        # it always resolves to True via the override in is_feature_available().

        # Subtitle Translation (Supporter)
        features["subtitle_translation"] = FeatureInfo(
            name="subtitle_translation",
            display_name="Subtitle Translation",
            description="Local speech-to-text transcription and translation for video subtitles",
            folder_path="subtitle_translation",
            required_files=[
                "subtitle_translation/__init__.py",
                "subtitle_translation/pipeline.py",
                "subtitle_translation/transcriber.py",
                "subtitle_translation/translator.py",
                "subtitle_translation/subtitle_track.py",
            ],
            dependencies=[],
            tier="supporter"
        )

        # Developer Tools (Dev only)
        features["dev_tools"] = FeatureInfo(
            name="dev_tools",
            display_name="Developer Tools",
            description="Development and debugging tools",
            folder_path="dev_tools",
            required_files=[
                "dev_tools/__init__.py"
            ],
            dependencies=[],
            tier="dev"
        )
        
        return features
    
    def detect_all_features(self):
        """Detect all available features."""
        self.available_features.clear()
        self.enabled_features.clear()
        
        for feature_name, feature_info in self.features.items():
            if self._detect_feature(feature_info):
                self.available_features.add(feature_name)
                
                # Check if feature should be enabled
                if self._should_enable_feature(feature_info):
                    self.enabled_features.add(feature_name)
                    feature_info.enabled = True
                else:
                    feature_info.enabled = False
                
                feature_info.available = True
            else:
                feature_info.available = False
                feature_info.enabled = False
        
        self.logger.debug(f"Detected {len(self.available_features)} available features")
        self.logger.debug(f"Enabled {len(self.enabled_features)} features")
        
        # Log enabled features
        for feature_name in self.enabled_features:
            feature = self.features[feature_name]
            # Try to get version from the module if available
            version_str = ""
            try:
                module = __import__(feature.folder_path)
                if hasattr(module, '__version__'):
                    version_str = f" (v{module.__version__})"
            except (ImportError, AttributeError):
                pass
            self.logger.info(f"  {feature.display_name}{version_str}")
    
    def _detect_feature(self, feature_info: FeatureInfo) -> bool:
        """Detect if a specific feature is available."""
        try:
            # Check if feature folder exists
            feature_path = self.app_root / feature_info.folder_path
            if not feature_path.exists() or not feature_path.is_dir():
                return False
            
            # Check required files (accept .py or compiled .so/.pyd)
            for required_file in feature_info.required_files:
                file_path = self.app_root / required_file
                if file_path.exists():
                    continue
                # Check for Cython-compiled variants (.so on Mac/Linux, .pyd on Windows)
                stem = file_path.stem  # e.g. "translator"
                parent = file_path.parent
                compiled = list(parent.glob(f"{stem}.cpython-*.so")) + list(parent.glob(f"{stem}.*.pyd"))
                if not compiled:
                    self.logger.debug(f"Missing required file for {feature_info.name}: {required_file}")
                    return False
            
            # Check dependencies
            for dependency in feature_info.dependencies:
                if dependency not in self.available_features:
                    # Try to detect dependency first
                    if dependency in self.features:
                        if not self._detect_feature(self.features[dependency]):
                            self.logger.debug(f"Missing dependency for {feature_info.name}: {dependency}")
                            return False
                        else:
                            self.available_features.add(dependency)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error detecting feature {feature_info.name}: {e}")
            return False
    
    def _should_enable_feature(self, feature_info: FeatureInfo) -> bool:
        """Determine if a feature should be enabled."""
        # For now, enable all available features
        # In the future, this could check license files, user settings, etc.
        
        # Free features are always enabled
        if feature_info.tier == "free":
            return True
        
        # Supporter features are enabled if folder exists
        if feature_info.tier == "supporter":
            return True  # Folder presence indicates user has access
        
        # Dev features only in development mode
        if feature_info.tier == "dev":
            return os.getenv("FUNGEN_DEV", "").lower() in ("1", "true", "yes")
        
        return False
    
    def is_feature_available(self, feature_name: str) -> bool:
        """Check if a feature is available."""
        # patreon_features modules now ship in core; legacy gates always pass.
        if feature_name == "patreon_features":
            return True
        return feature_name in self.available_features
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return feature_name in self.enabled_features
    
    def get_feature_info(self, feature_name: str) -> Optional[FeatureInfo]:
        """Get information about a feature."""
        return self.features.get(feature_name)
    
    def get_available_features(self) -> List[FeatureInfo]:
        """Get list of all available features."""
        return [
            feature for feature_name, feature in self.features.items()
            if feature_name in self.available_features
        ]
    
    def get_enabled_features(self) -> List[FeatureInfo]:
        """Get list of all enabled features."""
        return [
            feature for feature_name, feature in self.features.items()
            if feature_name in self.enabled_features
        ]
    
    def get_supporter_features(self) -> List[FeatureInfo]:
        """Get list of supporter-tier features."""
        return [
            feature for feature in self.features.values()
            if feature.tier == "supporter"
        ]
    
    def get_missing_supporter_features(self) -> List[FeatureInfo]:
        """Get list of supporter features that are not available."""
        return [
            feature for feature in self.features.values()
            if feature.tier == "supporter" and not feature.available
        ]
    
    def enable_feature(self, feature_name: str) -> bool:
        """Manually enable a feature (if available)."""
        if feature_name in self.available_features:
            self.enabled_features.add(feature_name)
            self.features[feature_name].enabled = True
            self.logger.info(f"Enabled feature: {feature_name}")
            return True
        return False
    
    def disable_feature(self, feature_name: str) -> bool:
        """Manually disable a feature."""
        if feature_name in self.enabled_features:
            self.enabled_features.remove(feature_name)
            self.features[feature_name].enabled = False
            self.logger.info(f"Disabled feature: {feature_name}")
            return True
        return False
    
    def get_feature_status_summary(self) -> Dict[str, Any]:
        """Get a summary of feature status."""
        return {
            "total_features": len(self.features),
            "available_features": len(self.available_features),
            "enabled_features": len(self.enabled_features),
            "supporter_features_available": len([
                f for f in self.features.values()
                if f.tier == "supporter" and f.available
            ]),
            "supporter_features_missing": len([
                f for f in self.features.values() 
                if f.tier == "supporter" and not f.available
            ]),
            "feature_list": {
                name: {
                    "available": info.available,
                    "enabled": info.enabled,
                    "tier": info.tier,
                    "display_name": info.display_name
                }
                for name, info in self.features.items()
            }
        }
    
    def create_supporter_info_message(self) -> str:
        """Create an informational message about supporter features."""
        missing_features = self.get_missing_supporter_features()
        
        if not missing_features:
            return "All premium features are available!"
        
        message = "Premium Features Available with Project Support:\\n\\n"
        
        for feature in missing_features:
            message += f"- {feature.display_name}: {feature.description}\\n"
        
        message += "\\nSupport the project to unlock these features!"
        message += "\\nSupporters receive folders to unlock premium functionality."
        
        return message


# Global feature detector instance
_feature_detector: Optional[FeatureDetector] = None


def get_feature_detector() -> FeatureDetector:
    """Get the global feature detector instance."""
    global _feature_detector
    if _feature_detector is None:
        _feature_detector = FeatureDetector()
    return _feature_detector


def is_feature_available(feature_name: str) -> bool:
    """Quick check if a feature is available."""
    return get_feature_detector().is_feature_available(feature_name)


def is_feature_enabled(feature_name: str) -> bool:
    """Quick check if a feature is enabled."""
    return get_feature_detector().is_feature_enabled(feature_name)


def get_supporter_features() -> List[FeatureInfo]:
    """Get list of supporter features."""
    return get_feature_detector().get_supporter_features()


def require_feature(feature_name: str):
    """Decorator to require a feature for a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_feature_enabled(feature_name):
                raise PermissionError(f"Feature '{feature_name}' is not available")
            return func(*args, **kwargs)
        return wrapper
    return decorator