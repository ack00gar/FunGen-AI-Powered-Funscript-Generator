"""
Feature Detection Utilities

Detects available features based on folder presence and dependencies.
Used to enable/disable features based on user tier (free vs supporter).
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
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
                "streamer/integration_manager.py"
            ],
            dependencies=[],
            tier="supporter"
        )
        
        # Advanced AI Processing (Supporter)
        features["advanced_ai"] = FeatureInfo(
            name="advanced_ai",
            display_name="Advanced AI Models",
            description="Access to premium AI models and processing modes",
            folder_path="advanced_ai",
            required_files=[
                "advanced_ai/__init__.py",
                "advanced_ai/models/premium_model.py"
            ],
            dependencies=[],
            tier="supporter"
        )
        
        # Premium Export Formats (Supporter)
        features["premium_export"] = FeatureInfo(
            name="premium_export",
            display_name="Premium Export",
            description="Export to premium formats and enhanced funscripts",
            folder_path="premium_export",
            required_files=[
                "premium_export/__init__.py",
                "premium_export/exporters.py"
            ],
            dependencies=[],
            tier="supporter"
        )
        
        # Live Streaming Integration (Supporter)
        features["live_streaming"] = FeatureInfo(
            name="live_streaming",
            display_name="Live Streaming",
            description="Stream live tracking to external services",
            folder_path="live_streaming",
            required_files=[
                "live_streaming/__init__.py",
                "live_streaming/streamers.py"
            ],
            dependencies=["device_control"],
            tier="supporter"
        )
        
        # Custom Models (Supporter)
        features["custom_models"] = FeatureInfo(
            name="custom_models",
            display_name="Custom Models",
            description="Load and use custom trained AI models",
            folder_path="custom_models",
            required_files=[
                "custom_models/__init__.py",
                "custom_models/model_loader.py"
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
        
        self.logger.info(f"Detected {len(self.available_features)} available features")
        self.logger.info(f"Enabled {len(self.enabled_features)} features")
        
        # Log enabled features
        for feature_name in self.enabled_features:
            feature = self.features[feature_name]
            self.logger.info(f"  ✓ {feature.display_name}")
    
    def _detect_feature(self, feature_info: FeatureInfo) -> bool:
        """Detect if a specific feature is available."""
        try:
            # Check if feature folder exists
            feature_path = self.app_root / feature_info.folder_path
            if not feature_path.exists() or not feature_path.is_dir():
                return False
            
            # Check required files
            for required_file in feature_info.required_files:
                file_path = self.app_root / required_file
                if not file_path.exists():
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
    
    def get_feature_status_summary(self) -> Dict[str, any]:
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
            message += f"• {feature.display_name}: {feature.description}\\n"
        
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
        def wrapper(*args, **kwargs):
            if not is_feature_enabled(feature_name):
                raise PermissionError(f"Feature '{feature_name}' is not available")
            return func(*args, **kwargs)
        return wrapper
    return decorator