"""
Dynamic Tracker Discovery System

This module replaces the hardcoded TrackerMode enum with a fully dynamic
discovery system. It provides mapping between UI display modes, CLI modes,
and internal tracker implementations.

No legacy fallback - pure modular approach.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto

from tracker.tracker_modules import tracker_registry, TrackerMetadata


class TrackerCategory(Enum):
    """Tracker categories for UI organization.

    LIVE / OFFLINE: primary scripting strategies the user picks first.
    LIVE_INTERVENTION: live trackers that need user setup at runtime.
    COMMUNITY: third-party drop-ins.
    TOOL: accessory utilities (peak/beat marker, manual ROI, oscillation
        detector, chapter maker). Hidden by default; surfaced when the user
        opts into the Tools filter. Tools can be live or offline at runtime
        depending on which base class they inherit; the category is purely
        about UI grouping.
    """
    LIVE = "live"
    LIVE_INTERVENTION = "live_intervention"
    OFFLINE = "offline"
    COMMUNITY = "community"
    TOOL = "tool"


@dataclass 
class TrackerDisplayInfo:
    """Information for displaying tracker in UI and CLI."""
    display_name: str
    internal_name: str
    category: TrackerCategory
    description: str
    cli_aliases: List[str]  # CLI command aliases
    requires_intervention: bool = False  # User ROI, oscillation area setup
    supports_batch: bool = True
    supports_realtime: bool = False
    supports_dual_axis: bool = True
    primary_axis: str = "stroke"
    secondary_axis: str = "roll"
    additional_axes: List[str] = field(default_factory=list)  # Extra axes beyond primary/secondary
    stages: List = field(default_factory=list)  # List of StageDefinition objects
    properties: Dict[str, Any] = field(default_factory=dict)  # Tracker properties/capabilities
    version: str = ""  # Tracker version from metadata
    folder_name: str = ""  # Source folder name for prefixing (live, offline, experimental, community)


class DynamicTrackerDiscovery:
    """
    Fully dynamic tracker discovery system.
    
    This class eliminates all hardcoded tracker references by:
    1. Discovering available trackers from the modular system
    2. Categorizing them by type (Live, Live+Intervention, Offline)
    3. Providing unified mapping for GUI, CLI, and batch modes
    4. Supporting runtime addition/removal of trackers
    """
    
    def __init__(self):
        self.logger = logging.getLogger("DynamicTrackerDiscovery")
        self._display_info_cache: Dict[str, TrackerDisplayInfo] = {}
        self._cli_alias_map: Dict[str, str] = {}
        self._category_map: Dict[TrackerCategory, List[str]] = {}
        
        # Build initial mappings
        self._discover_and_categorize()
        
    def _discover_and_categorize(self):
        """Discover all available trackers and categorize them."""
        self._display_info_cache.clear()
        self._cli_alias_map.clear()
        self._category_map.clear()
        
        # Initialize category map
        for category in TrackerCategory:
            self._category_map[category] = []
        
        # Process each discovered tracker
        available_trackers = tracker_registry.list_trackers()
        
        for metadata in available_trackers:
            display_info = self._create_display_info(metadata)
            
            # Cache the display info
            self._display_info_cache[metadata.name] = display_info
            
            # Add to category map
            self._category_map[display_info.category].append(metadata.name)
            
            # Build CLI alias map
            for alias in display_info.cli_aliases:
                self._cli_alias_map[alias] = metadata.name
        
        # Single line summary
        categories_summary = ", ".join([f"{cat.value}:{len(trackers)}" for cat, trackers in self._category_map.items() if trackers])
        self.logger.debug(f"Tracker categories: {categories_summary}")
    
    def _create_display_info(self, metadata: TrackerMetadata) -> TrackerDisplayInfo:
        """Create display info from tracker metadata."""
        
        # Determine category based on metadata category and characteristics
        category = self._determine_category(metadata)
        
        # Generate CLI aliases based on name and category
        cli_aliases = self._generate_cli_aliases(metadata, category)
        
        # Determine if tracker requires user intervention
        requires_intervention = self._requires_user_intervention(metadata)
        
        # Determine capabilities. TOOL trackers (oscillation, chapter_maker,
        # beat_marker, user_roi) are batch-capable only when they do not
        # require user intervention: user_roi asks the user to draw a box at
        # runtime, so it cannot run unattended.
        supports_batch = (
            category in [TrackerCategory.LIVE, TrackerCategory.OFFLINE]
            or (category == TrackerCategory.TOOL and not requires_intervention)
        )
        supports_realtime = category in [TrackerCategory.LIVE, TrackerCategory.LIVE_INTERVENTION, TrackerCategory.TOOL]
        
        # Get stages and properties from metadata if available
        stages = getattr(metadata, 'stages', [])
        properties = getattr(metadata, 'properties', {})
        
        # Determine folder name for display prefixing
        folder_name = self._determine_folder_name(metadata, category)
        
        return TrackerDisplayInfo(
            display_name=metadata.display_name,
            internal_name=metadata.name,
            category=category,
            description=metadata.description,
            cli_aliases=cli_aliases,
            requires_intervention=requires_intervention,
            supports_batch=supports_batch,
            supports_realtime=supports_realtime,
            supports_dual_axis=metadata.supports_dual_axis,
            primary_axis=getattr(metadata, 'primary_axis', 'stroke'),
            secondary_axis=getattr(metadata, 'secondary_axis', 'roll'),
            additional_axes=getattr(metadata, 'additional_axes', []),
            stages=stages,
            properties=properties,
            version=getattr(metadata, 'version', ''),
            folder_name=folder_name
        )
    
    def _determine_folder_name(self, metadata: TrackerMetadata, category: TrackerCategory) -> str:
        """Determine the source folder name for display prefixing."""
        
        # Get the actual folder name from the tracker registry
        from tracker.tracker_modules import tracker_registry
        actual_folder = tracker_registry.get_tracker_folder(metadata.name)
        
        if actual_folder:
            return actual_folder
        
        # Fallback: use category as folder name for standard trackers
        if category == TrackerCategory.LIVE or category == TrackerCategory.LIVE_INTERVENTION:
            return "live"
        elif category == TrackerCategory.OFFLINE:
            return "offline"
        elif category == TrackerCategory.COMMUNITY:
            return "community"
        elif category == TrackerCategory.TOOL:
            return "tool"

        # Default fallback
        return "live"
    
    def _determine_category(self, metadata: TrackerMetadata) -> TrackerCategory:
        """Determine tracker category from metadata."""
        
        # Check if it's a community tracker
        if hasattr(metadata, 'is_community') and metadata.is_community:
            return TrackerCategory.COMMUNITY
        
        # Categorize by metadata category
        if metadata.category == "live":
            # Check if it requires user intervention
            if self._requires_user_intervention(metadata):
                return TrackerCategory.LIVE_INTERVENTION
            else:
                return TrackerCategory.LIVE
        elif metadata.category == "live_intervention":
            return TrackerCategory.LIVE_INTERVENTION
        elif metadata.category == "offline":
            return TrackerCategory.OFFLINE
        elif metadata.category == "community":
            return TrackerCategory.COMMUNITY
        elif metadata.category == "tool":
            return TrackerCategory.TOOL
        else:
            # Default community for unknown categories
            return TrackerCategory.COMMUNITY
    
    def _requires_user_intervention(self, metadata: TrackerMetadata) -> bool:
        """Check if tracker requires user intervention to set up."""
        # Only manual/user trackers require intervention
        intervention_keywords = [
            "user", "manual"
        ]
        
        name_lower = metadata.name.lower()
        display_lower = metadata.display_name.lower()
        
        return any(keyword in name_lower or keyword in display_lower 
                  for keyword in intervention_keywords)
    
    def _generate_cli_aliases(self, metadata: TrackerMetadata, category: TrackerCategory) -> List[str]:
        """Generate CLI aliases for the tracker."""
        aliases = []

        # Primary alias from internal name
        aliases.append(metadata.name)

        # Also add lowercase version of the internal name as an alias
        name_lower = metadata.name.lower()
        if name_lower != metadata.name:
            aliases.append(name_lower)

        # Determine if tracker is in legacy folder
        from tracker.tracker_modules import tracker_registry
        actual_folder = tracker_registry.get_tracker_folder(metadata.name)
        is_legacy = (actual_folder == "legacy")

        # Specific aliases by tracker name for backward compatibility and convenience
        if "GUIDED_FLOW" in metadata.name:
            aliases.extend(["3-stage", "stage3", "guided-flow", "offline-3stage"])
        elif "CONTACT_ANALYSIS" in metadata.name:
            aliases.extend(["2-stage", "stage2", "offline-2stage"])
        elif metadata.name == "LEGACY_STAGE3_MIXED":
            aliases.extend(["legacy-3-stage-mixed", "stage3-mixed", "mixed"])
        elif metadata.name == "LEGACY_STAGE3_OPTICAL_FLOW":
            aliases.extend(["legacy-3-stage", "legacy-stage3"])
        elif metadata.name == "LIVE_OSCILLATION":
            aliases.extend(["oscillation", "osc", "live-osc"])
        elif metadata.name == "LEGACY_OSCILLATION":
            aliases.extend(["oscillation-legacy"])
        elif metadata.name == "LIVE_YOLO_ROI":
            aliases.extend(["yolo", "yolo_roi", "live-yolo", "auto-roi"])
        elif metadata.name == "LIVE_USER_ROI":
            aliases.extend(["user_roi"])
        elif metadata.name == "LIVE_VR_CHAPTER_FLOW":
            aliases.extend(["vr_chapter_flow", "vr-chapter-flow"])
        elif metadata.name == "LIVE_VR_FOCUSED":
            aliases.extend(["vr_focused", "vr-focused"])
        # Note: LIVE_INTERVENTION trackers don't get CLI aliases since they're not batch compatible

        return aliases
    
    # Public API Methods
    
    def get_all_trackers(self) -> Dict[str, TrackerDisplayInfo]:
        """Get all discovered trackers with display information."""
        return self._display_info_cache.copy()
    
    def get_trackers_by_category(self, category: TrackerCategory) -> List[TrackerDisplayInfo]:
        """Get all trackers in a specific category."""
        tracker_names = self._category_map.get(category, [])
        return [self._display_info_cache[name] for name in tracker_names if name in self._display_info_cache]
    
    def get_tracker_info(self, identifier: str) -> Optional[TrackerDisplayInfo]:
        """Get tracker info by internal name or CLI alias."""
        # Try direct lookup
        if identifier in self._display_info_cache:
            return self._display_info_cache[identifier]
        
        # Try CLI alias lookup
        internal_name = self._cli_alias_map.get(identifier)
        if internal_name and internal_name in self._display_info_cache:
            return self._display_info_cache[internal_name]
        
        return None
    
    def resolve_cli_mode(self, cli_mode: str) -> Optional[str]:
        """Resolve CLI mode string to internal tracker name."""
        return self._cli_alias_map.get(cli_mode)

    def get_runtime_category(self, identifier: str) -> Optional[TrackerCategory]:
        """Resolve a tracker to its runtime dispatch category (LIVE or OFFLINE).

        LIVE / LIVE_INTERVENTION / OFFLINE / COMMUNITY trackers pass through
        unchanged. TOOL trackers are UI-grouped accessories whose actual runtime
        is determined by which base class they inherit: BaseOfflineTracker maps
        to OFFLINE, BaseTracker maps to LIVE. Batch dispatchers use this to pick
        the right processing path for tools like Oscillation Detector (live) and
        Chapter Maker (offline).
        """
        info = self.get_tracker_info(identifier)
        if not info:
            return None
        if info.category != TrackerCategory.TOOL:
            return info.category
        from tracker.tracker_modules.core.base_offline_tracker import BaseOfflineTracker
        tracker_class = tracker_registry.get_tracker(info.internal_name)
        if tracker_class and issubclass(tracker_class, BaseOfflineTracker):
            return TrackerCategory.OFFLINE
        return TrackerCategory.LIVE
    
    def get_gui_display_list(self) -> Tuple[List[str], List[str]]:
        """
        Get display lists for GUI combo boxes.
        
        Returns:
            Tuple[List[str], List[str]]: (display_names, internal_names)
        """
        display_names = []
        internal_names = []
        
        # Collect all trackers from all categories
        all_trackers = []
        for category in [TrackerCategory.LIVE, TrackerCategory.LIVE_INTERVENTION,
                        TrackerCategory.OFFLINE, TrackerCategory.COMMUNITY,
                        TrackerCategory.TOOL]:
            category_trackers = self.get_trackers_by_category(category)
            all_trackers.extend(category_trackers)
        
        # Filter out example trackers
        all_trackers = [t for t in all_trackers if "example" not in t.internal_name.lower() and "example" not in t.display_name.lower()]
        
        # Sort by folder priority: Live first, Offline second, then the rest
        _FOLDER_PRIORITY = {"live": 0, "offline": 1, "legacy": 2, "experimental": 3, "community": 4}

        def get_sort_key(tracker_info):
            folder = (tracker_info.folder_name or "").lower()
            priority = _FOLDER_PRIORITY.get(folder, 99)
            folder_prefix = tracker_info.folder_name.title() + " - " if tracker_info.folder_name else ""
            display_name = tracker_info.display_name
            if not display_name.startswith(folder_prefix):
                prefixed_display_name = folder_prefix + display_name
            else:
                prefixed_display_name = display_name
            return (priority, prefixed_display_name)

        all_trackers.sort(key=get_sort_key)
        
        # Build the final lists with proper prefixing
        for tracker_info in all_trackers:
            # Add folder prefix to display name (but avoid duplicates)
            folder_prefix = tracker_info.folder_name.title() + " - " if tracker_info.folder_name else ""
            display_name = tracker_info.display_name
            
            # Don't add prefix if display name already starts with it
            if not display_name.startswith(folder_prefix):
                prefixed_display_name = folder_prefix + display_name
            else:
                prefixed_display_name = display_name
                
            display_names.append(prefixed_display_name)
            internal_names.append(tracker_info.internal_name)
        
        return display_names, internal_names
    
    def get_gui_display_list_filtered(self, hidden_folders: Set[str]) -> Tuple[List[str], List[str]]:
        """
        Get display lists filtered by hidden folder names.

        Args:
            hidden_folders: Set of folder names to exclude (e.g. {"legacy", "experimental"}).

        Returns:
            Tuple[List[str], List[str]]: (display_names, internal_names)
        """
        display_names, internal_names = self.get_gui_display_list()
        if not hidden_folders:
            return display_names, internal_names

        filtered_display = []
        filtered_internal = []
        for display, internal in zip(display_names, internal_names):
            info = self._display_info_cache.get(internal)
            if info and info.folder_name.lower() in hidden_folders:
                continue
            filtered_display.append(display)
            filtered_internal.append(internal)
        return filtered_display, filtered_internal

    def get_batch_compatible_trackers(self) -> List[TrackerDisplayInfo]:
        """Get trackers that support batch processing (Live + Offline, no Live Intervention)."""
        return [info for info in self._display_info_cache.values() 
                if info.category in [TrackerCategory.LIVE, TrackerCategory.OFFLINE]]
    
    def get_realtime_compatible_trackers(self) -> List[TrackerDisplayInfo]:
        """Get trackers that support real-time processing."""
        return [info for info in self._display_info_cache.values() if info.supports_realtime]
    
    def get_supported_cli_modes(self) -> List[str]:
        """Get all supported CLI mode aliases, excluding example trackers."""
        # Filter out CLI aliases that map to example trackers
        filtered_aliases = []
        for alias, tracker_name in self._cli_alias_map.items():
            tracker_info = self.get_tracker_info(tracker_name)
            if (tracker_info and 
                "example" not in tracker_info.internal_name.lower() and
                "example" not in tracker_info.display_name.lower()):
                filtered_aliases.append(alias)
        return filtered_aliases
    
    def reload(self):
        """Reload tracker discovery (for development/testing)."""
        self.logger.debug("Reloading dynamic tracker discovery...")
        tracker_registry.reload_trackers()
        self._discover_and_categorize()
    
    def validate_setup(self) -> Tuple[bool, List[str]]:
        """
        Validate the tracker setup.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Check that we have at least one tracker in each main category
        live_count = len(self._category_map.get(TrackerCategory.LIVE, []))
        offline_count = len(self._category_map.get(TrackerCategory.OFFLINE, []))
        
        if live_count == 0:
            errors.append("No live trackers discovered")
        
        if offline_count == 0:
            errors.append("No offline trackers discovered")
        
        # Check for CLI alias conflicts
        alias_counts = {}
        for alias in self._cli_alias_map.keys():
            alias_counts[alias] = alias_counts.get(alias, 0) + 1
        
        conflicts = [alias for alias, count in alias_counts.items() if count > 1]
        if conflicts:
            errors.append(f"CLI alias conflicts: {conflicts}")
        
        return len(errors) == 0, errors


# Global discovery instance
_tracker_discovery = None

def get_tracker_discovery() -> DynamicTrackerDiscovery:
    """Get the global tracker discovery instance."""
    global _tracker_discovery
    if _tracker_discovery is None:
        _tracker_discovery = DynamicTrackerDiscovery()
    return _tracker_discovery


def validate_tracker_setup() -> Tuple[bool, List[str]]:
    """Convenience function to validate tracker setup."""
    return get_tracker_discovery().validate_setup()
