"""
Dynamic Tracker UI Helper

This module provides UI helper methods that replace hardcoded TrackerMode enum
dependencies with dynamic tracker discovery. It allows GUI components to work
with trackers dynamically without hardcoded references.
"""

from typing import List, Tuple, Optional, Set
from config.tracker_discovery import get_tracker_discovery, TrackerCategory, TrackerDisplayInfo


class DynamicTrackerUI:
    """
    Helper class that provides dynamic tracker information for GUI components.
    
    This class replaces hardcoded TrackerMode enum checks with dynamic 
    category-based checks using the tracker discovery system.
    """
    
    def __init__(self):
        self.discovery = get_tracker_discovery()
    
    def get_gui_display_list(self) -> Tuple[List[str], List[str]]:
        """Get display names and internal names for GUI combo boxes."""
        return self.discovery.get_gui_display_list()
    
    def get_simple_mode_trackers(self) -> Tuple[List[str], List[str]]:
        """Get only live trackers for simple mode GUI."""
        all_trackers = self.discovery.get_all_trackers()
        
        display_names = []
        internal_names = []
        
        # Get live trackers only for simple mode
        for name, info in all_trackers.items():
            if info.category in [TrackerCategory.LIVE, TrackerCategory.LIVE_INTERVENTION]:
                display_names.append(info.display_name)
                internal_names.append(info.internal_name)
        
        return display_names, internal_names
    
    def get_batch_gui_compatible_trackers(self) -> Tuple[List[str], List[str]]:
        """Get trackers compatible with batch GUI (Live + Offline, no Live Intervention)."""
        all_trackers = self.discovery.get_all_trackers()
        
        display_names = []
        internal_names = []
        
        # Include Live (automatic) and Offline trackers
        # Exclude Live Intervention (requires user setup - not compatible with batch)
        for name, info in all_trackers.items():
            if info.category in [TrackerCategory.LIVE, TrackerCategory.OFFLINE]:
                display_names.append(info.display_name)
                internal_names.append(info.internal_name)
        
        # Sort by category: Live first, then Offline
        combined = list(zip(display_names, internal_names))
        combined.sort(key=lambda x: (
            0 if self.discovery.get_tracker_info(x[1]).category == TrackerCategory.LIVE else 1,
            x[0]  # Then alphabetically by display name
        ))
        
        display_names, internal_names = zip(*combined) if combined else ([], [])
        return list(display_names), list(internal_names)
    
    def is_live_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is a live tracker (any live category)."""
        info = self.discovery.get_tracker_info(tracker_name)
        if not info:
            return False
        return info.category in [TrackerCategory.LIVE, TrackerCategory.LIVE_INTERVENTION]
    
    def is_offline_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is an offline tracker."""
        info = self.discovery.get_tracker_info(tracker_name)
        if not info:
            return False
        return info.category == TrackerCategory.OFFLINE
    
    def requires_user_intervention(self, tracker_name: str) -> bool:
        """Check if tracker requires user intervention (ROI setup, etc)."""
        info = self.discovery.get_tracker_info(tracker_name)
        if not info:
            return False
        return info.requires_intervention
    
    def supports_batch_processing(self, tracker_name: str) -> bool:
        """Check if tracker supports batch processing."""
        info = self.discovery.get_tracker_info(tracker_name)
        if not info:
            return False
        return info.supports_batch
    
    def is_oscillation_detector(self, tracker_name: str) -> bool:
        """Check if tracker is an oscillation detector."""
        if not tracker_name:
            return False
        return "oscillation" in tracker_name.lower()
    
    def is_yolo_roi_tracker(self, tracker_name: str) -> bool:
        """Check if tracker uses YOLO ROI."""
        if not tracker_name:
            return False
        return "yolo" in tracker_name.lower()
    
    def is_user_roi_tracker(self, tracker_name: str) -> bool:
        """Check if tracker uses user-defined ROI."""
        if not tracker_name:
            return False
        return "user" in tracker_name.lower()
    
    def is_stage2_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is a 2-stage offline tracker."""
        if not tracker_name:
            return False
        
        info = self.discovery.get_tracker_info(tracker_name)
        if info and info.properties:
            # Use property if available
            if "is_stage2_tracker" in info.properties:
                return info.properties["is_stage2_tracker"]
            # Check if it produces funscript in stage 2 and has exactly 2 stages
            if info.properties.get("num_stages") == 2 and info.properties.get("produces_funscript_in_stage2"):
                return True
        
        return False
    
    def is_stage3_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is a 3-stage offline tracker."""
        if not tracker_name:
            return False
        
        info = self.discovery.get_tracker_info(tracker_name)
        if info and info.properties:
            # Use property if available
            if "is_stage3_tracker" in info.properties:
                return info.properties["is_stage3_tracker"]
            # Check if it has 3 stages but is not mixed
            if info.properties.get("num_stages") == 3 and not info.properties.get("uses_hybrid_approach"):
                return True
        
        return False
    
    def is_mixed_stage3_tracker(self, tracker_name: str) -> bool:
        """Check if tracker is a mixed 3-stage offline tracker."""
        if not tracker_name:
            return False
        
        info = self.discovery.get_tracker_info(tracker_name)
        if info and info.properties:
            # Use property if available
            if "is_mixed_stage3_tracker" in info.properties:
                return info.properties["is_mixed_stage3_tracker"]
            # Check for hybrid approach property
            if info.properties.get("uses_hybrid_approach"):
                return True
        
        return False
    
    def get_trackers_in_category(self, category: TrackerCategory) -> List[TrackerDisplayInfo]:
        """Get all trackers in a specific category."""
        return self.discovery.get_trackers_by_category(category)
    
    def get_tracker_display_name(self, tracker_name: str) -> str:
        """Get display name for a tracker."""
        info = self.discovery.get_tracker_info(tracker_name)
        return info.display_name if info else tracker_name
    
    def get_tracker_category_name(self, tracker_name: str) -> str:
        """Get category name for a tracker."""
        info = self.discovery.get_tracker_info(tracker_name)
        return info.category.value if info else "unknown"
    
    def get_default_tracker(self) -> str:
        """Get the default tracker name."""
        from config.constants import DEFAULT_TRACKER_NAME
        return DEFAULT_TRACKER_NAME
    
    def generate_tooltip_for_tracker(self, tracker_name: str) -> str:
        """Generate tooltip text for a tracker."""
        info = self.discovery.get_tracker_info(tracker_name)
        if not info:
            return f"Unknown tracker: {tracker_name}"
        
        tooltip_lines = [
            f"Name: {info.display_name}",
            f"Category: {info.category.value.title()}",
            f"Description: {info.description}",
        ]
        
        if info.requires_intervention:
            tooltip_lines.append("⚠️ Requires user setup (ROI, etc.)")
        
        if info.supports_batch:
            tooltip_lines.append("✅ Batch processing supported")
        
        if info.supports_realtime:
            tooltip_lines.append("🔴 Real-time processing supported")
        
        if info.cli_aliases:
            tooltip_lines.append(f"CLI aliases: {', '.join(info.cli_aliases)}")
        
        return "\n".join(tooltip_lines)
    
    def get_combined_tooltip(self, tracker_names: List[str]) -> str:
        """Generate combined tooltip for multiple trackers."""
        if not tracker_names:
            return "No trackers available"
        
        tooltip_sections = []
        for tracker_name in tracker_names[:5]:  # Limit to first 5 to avoid huge tooltips
            tooltip_sections.append(self.generate_tooltip_for_tracker(tracker_name))
        
        if len(tracker_names) > 5:
            tooltip_sections.append(f"... and {len(tracker_names) - 5} more trackers")
        
        return "\n\n".join(tooltip_sections)
    
    def get_trackers_with_config_support(self) -> Set[str]:
        """Get set of tracker names that have configuration UI support."""
        # For now, return all trackers - configuration support can be determined dynamically
        all_trackers = self.discovery.get_all_trackers()
        return set(all_trackers.keys())


# Global instance for GUI components
_dynamic_tracker_ui = None

def get_dynamic_tracker_ui() -> DynamicTrackerUI:
    """Get the global dynamic tracker UI instance."""
    global _dynamic_tracker_ui
    if _dynamic_tracker_ui is None:
        _dynamic_tracker_ui = DynamicTrackerUI()
    return _dynamic_tracker_ui