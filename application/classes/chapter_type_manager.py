"""
Chapter Type Manager - Manages built-in and user-defined chapter types

This module provides a centralized system for managing chapter/position types used
throughout the application. It handles:
- Built-in chapter types (BJ, HJ, CG/Miss, etc.)
- User-defined custom types
- Type persistence (JSON storage)
- Color management
- Usage tracking for quick access
- Import/Export of custom type libraries

Author: k00gar
Version: 1.0.0
"""

import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from config import constants


class ChapterTypeManager:
    """
    Manages both built-in and user-defined chapter types.

    Features:
    - Merges built-in types from constants.POSITION_INFO_MAPPING
    - Persists custom types to JSON file
    - Tracks usage statistics
    - Provides search and filtering
    - Supports import/export of type libraries
    """

    CUSTOM_TYPES_FILE = "custom_chapter_types.json"
    VERSION = "1.0"

    # Default categories for organizing custom types
    DEFAULT_CATEGORIES = [
        "Position",
        "Solo",
        "Group",
        "Transition",
        "Custom",
        "Other"
    ]

    def __init__(self, app):
        """
        Initialize the ChapterTypeManager.

        Args:
            app: Application instance for accessing settings and logging
        """
        self.app = app
        self.logger = logging.getLogger(__name__)

        # Built-in types from constants (immutable)
        self.builtin_types = constants.POSITION_INFO_MAPPING.copy()

        # Enrich built-in types with colors from VideoSegment color mapping
        self._enrich_builtin_types_with_colors()

        # Custom types (user-defined, mutable)
        self.custom_types: Dict[str, Dict[str, Any]] = {}

        # Usage tracking (counts how often each type is used)
        self.usage_stats: Dict[str, int] = {}

        # Recently used types (for quick access)
        self.recent_types: List[str] = []

        # Load custom types from file
        self.load_custom_types()

    def _enrich_builtin_types_with_colors(self):
        """Add color information to built-in types from VideoSegment color mapping."""
        # Import here to avoid circular dependency
        from application.utils.video_segment import VideoSegment
        from config.element_group_colors import SegmentColors

        for short_name, type_info in self.builtin_types.items():
            # Get color from VideoSegment's color map, fallback to default
            color = VideoSegment._POSITION_COLOR_MAP.get(short_name, SegmentColors.DEFAULT)
            # Convert to list for JSON serialization compatibility
            type_info["color"] = list(color)
            # Category should already be defined in POSITION_INFO_MAPPING, but provide fallback
            if "category" not in type_info:
                type_info["category"] = "Position"
            type_info["usage_count"] = 0

    # ==================== CORE TYPE ACCESS ====================

    def get_all_chapter_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Get merged dictionary of all chapter types (built-in + custom).

        Returns:
            Dict mapping short_name to type info dict
        """
        return {**self.builtin_types, **self.custom_types}

    def get_type_by_short_name(self, short_name: str) -> Optional[Dict[str, Any]]:
        """
        Get type information by short name.

        Args:
            short_name: The short identifier (e.g., 'BJ', '69', 'Trans')

        Returns:
            Type info dict or None if not found
        """
        all_types = self.get_all_chapter_types()
        return all_types.get(short_name)

    def is_builtin_type(self, short_name: str) -> bool:
        """Check if a type is built-in (non-editable)."""
        return short_name in self.builtin_types

    def is_custom_type(self, short_name: str) -> bool:
        """Check if a type is user-defined."""
        return short_name in self.custom_types

    # ==================== CUSTOM TYPE MANAGEMENT ====================

    def add_custom_type(self, short_name: str, long_name: str,
                       color: Tuple[float, float, float, float],
                       category: str = "Custom") -> bool:
        """
        Add a new custom chapter type.

        Args:
            short_name: Short identifier (2-10 chars, unique)
            long_name: Full display name
            color: RGBA color tuple (0-1 range)
            category: Category for organization

        Returns:
            True if successful, False if short_name already exists
        """
        # Validate short_name
        if not short_name or len(short_name) < 2 or len(short_name) > 10:
            self.logger.error(f"Invalid short_name: must be 2-10 characters")
            return False

        # Check for duplicates (case-sensitive)
        if short_name in self.get_all_chapter_types():
            self.logger.error(f"Chapter type '{short_name}' already exists")
            return False

        # Create type entry
        self.custom_types[short_name] = {
            "long_name": long_name,
            "short_name": short_name,
            "color": list(color),
            "category": category,
            "created_date": datetime.now().isoformat(),
            "usage_count": 0
        }

        self.logger.info(f"Added custom chapter type: {short_name} - {long_name}")
        self.save_custom_types()
        return True

    def edit_custom_type(self, short_name: str, new_data: Dict[str, Any]) -> bool:
        """
        Edit an existing custom chapter type.

        Args:
            short_name: Type to edit
            new_data: Dict with fields to update (long_name, color, category)

        Returns:
            True if successful, False if type not found or is built-in
        """
        if not self.is_custom_type(short_name):
            self.logger.error(f"Cannot edit: '{short_name}' is not a custom type")
            return False

        # Update fields
        type_entry = self.custom_types[short_name]

        if "long_name" in new_data:
            type_entry["long_name"] = new_data["long_name"]
        if "color" in new_data:
            type_entry["color"] = list(new_data["color"])
        if "category" in new_data:
            type_entry["category"] = new_data["category"]

        type_entry["modified_date"] = datetime.now().isoformat()

        self.logger.info(f"Updated custom chapter type: {short_name}")
        self.save_custom_types()
        return True

    def delete_custom_type(self, short_name: str) -> bool:
        """
        Delete a custom chapter type.

        Args:
            short_name: Type to delete

        Returns:
            True if successful, False if not found or is built-in
        """
        if not self.is_custom_type(short_name):
            self.logger.error(f"Cannot delete: '{short_name}' is not a custom type")
            return False

        del self.custom_types[short_name]

        # Clean up usage stats
        if short_name in self.usage_stats:
            del self.usage_stats[short_name]

        # Remove from recent types
        if short_name in self.recent_types:
            self.recent_types.remove(short_name)

        self.logger.info(f"Deleted custom chapter type: {short_name}")
        self.save_custom_types()
        return True

    # ==================== USAGE TRACKING ====================

    def increment_usage(self, short_name: str):
        """
        Increment usage count for a chapter type.
        Called when a chapter of this type is created.

        Args:
            short_name: Type being used
        """
        # Increment in-memory counter
        self.usage_stats[short_name] = self.usage_stats.get(short_name, 0) + 1

        # Update custom type's usage_count if applicable
        if short_name in self.custom_types:
            self.custom_types[short_name]["usage_count"] = self.usage_stats[short_name]

        # Update recent types list
        if short_name in self.recent_types:
            self.recent_types.remove(short_name)
        self.recent_types.insert(0, short_name)

        # Keep only recent N types
        max_recent = self.app.app_settings.get("chapter_type_recent_max", 5)
        self.recent_types = self.recent_types[:max_recent]

    def get_usage_count(self, short_name: str) -> int:
        """Get usage count for a type."""
        return self.usage_stats.get(short_name, 0)

    def get_most_used_types(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get list of most-used types.

        Args:
            limit: Maximum number to return

        Returns:
            List of (short_name, usage_count) tuples, sorted by usage
        """
        sorted_types = sorted(
            self.usage_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_types[:limit]

    def get_recently_used(self, limit: int = 5) -> List[str]:
        """
        Get list of recently-used type short names.

        Args:
            limit: Maximum number to return

        Returns:
            List of short_name strings
        """
        return self.recent_types[:limit]

    # ==================== FILTERING & SEARCH ====================

    def get_types_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all custom types in a specific category.

        Args:
            category: Category name

        Returns:
            Dict of types in that category
        """
        return {
            short_name: type_info
            for short_name, type_info in self.custom_types.items()
            if type_info.get("category") == category
        }

    def get_all_categories(self) -> List[str]:
        """
        Get list of all categories (built-in + used custom categories).

        Returns:
            Sorted list of unique category names
        """
        custom_categories = set(
            type_info.get("category", "Other")
            for type_info in self.custom_types.values()
        )

        all_categories = set(self.DEFAULT_CATEGORIES) | custom_categories
        return sorted(all_categories)

    def search_types(self, query: str) -> Dict[str, Dict[str, Any]]:
        """
        Search for types matching a query string.
        Searches both short_name and long_name.

        Args:
            query: Search string (case-insensitive)

        Returns:
            Dict of matching types
        """
        query_lower = query.lower()
        all_types = self.get_all_chapter_types()

        return {
            short_name: type_info
            for short_name, type_info in all_types.items()
            if (query_lower in short_name.lower() or
                query_lower in type_info.get("long_name", "").lower())
        }

    # ==================== PERSISTENCE ====================

    def save_custom_types(self):
        """Save custom types and usage stats to JSON file."""
        data = {
            "version": self.VERSION,
            "custom_types": self.custom_types,
            "usage_stats": self.usage_stats,
            "recent_types": self.recent_types,
            "categories": self.get_all_categories()
        }

        filepath = os.path.join(os.getcwd(), self.CUSTOM_TYPES_FILE)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved custom chapter types to {self.CUSTOM_TYPES_FILE}")
        except Exception as e:
            self.logger.error(f"Failed to save custom chapter types: {e}", exc_info=True)

    def load_custom_types(self):
        """Load custom types and usage stats from JSON file."""
        filepath = os.path.join(os.getcwd(), self.CUSTOM_TYPES_FILE)

        if not os.path.exists(filepath):
            self.logger.info("No custom chapter types file found, starting fresh")
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate version
            file_version = data.get("version", "1.0")
            if file_version != self.VERSION:
                self.logger.warning(f"Custom types file version mismatch: {file_version} vs {self.VERSION}")

            # Load data
            self.custom_types = data.get("custom_types", {})
            self.usage_stats = data.get("usage_stats", {})
            self.recent_types = data.get("recent_types", [])

            self.logger.info(f"Loaded {len(self.custom_types)} custom chapter types")

        except Exception as e:
            self.logger.error(f"Failed to load custom chapter types: {e}", exc_info=True)
            self.custom_types = {}
            self.usage_stats = {}
            self.recent_types = []

    # ==================== IMPORT / EXPORT ====================

    def export_types_to_file(self, filepath: str, include_builtin: bool = False) -> bool:
        """
        Export custom types to a shareable JSON file.

        Args:
            filepath: Destination file path
            include_builtin: Whether to include built-in types

        Returns:
            True if successful
        """
        export_data = {
            "version": self.VERSION,
            "author": "k00gar",
            "description": "FunGen custom chapter types library",
            "export_date": datetime.now().isoformat(),
            "types": self.custom_types.copy()
        }

        if include_builtin:
            export_data["builtin_types"] = self.builtin_types.copy()

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Exported chapter types to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export chapter types: {e}", exc_info=True)
            return False

    def import_types_from_file(self, filepath: str, merge_mode: str = "skip") -> Tuple[int, int]:
        """
        Import custom types from a JSON file.

        Args:
            filepath: Source file path
            merge_mode: How to handle conflicts:
                - "skip": Skip existing types
                - "replace": Replace existing types
                - "rename": Auto-rename conflicts

        Returns:
            Tuple of (imported_count, skipped_count)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            imported_types = data.get("types", {})
            imported_count = 0
            skipped_count = 0

            for short_name, type_info in imported_types.items():
                exists = short_name in self.get_all_chapter_types()

                if exists and merge_mode == "skip":
                    skipped_count += 1
                    continue

                if exists and merge_mode == "rename":
                    # Auto-rename with suffix
                    original_name = short_name
                    suffix = 1
                    while short_name in self.get_all_chapter_types():
                        short_name = f"{original_name}_{suffix}"
                        suffix += 1
                    type_info["short_name"] = short_name

                # Add/replace the type
                self.custom_types[short_name] = type_info
                imported_count += 1

            self.save_custom_types()
            self.logger.info(f"Imported {imported_count} chapter types ({skipped_count} skipped)")
            return (imported_count, skipped_count)

        except Exception as e:
            self.logger.error(f"Failed to import chapter types: {e}", exc_info=True)
            return (0, 0)

    # ==================== UTILITY ====================

    def get_type_color(self, short_name: str) -> Tuple[float, float, float, float]:
        """
        Get color for a chapter type.

        Args:
            short_name: Type identifier

        Returns:
            RGBA color tuple, or default gray if not found
        """
        type_info = self.get_type_by_short_name(short_name)

        if type_info and "color" in type_info:
            color = type_info["color"]
            # Ensure it's a tuple of 4 floats
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                if len(color) == 3:
                    return (*color, 1.0)
                return tuple(color[:4])

        # Fallback to default
        return (0.5, 0.5, 0.5, 0.7)

    def validate_type_data(self, short_name: str, long_name: str,
                          color: Tuple, category: str) -> Tuple[bool, str]:
        """
        Validate chapter type data before adding/editing.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate short_name
        if not short_name or len(short_name) < 2:
            return (False, "Short name must be at least 2 characters")

        if len(short_name) > 10:
            return (False, "Short name must be 10 characters or less")

        if not short_name.replace(".", "").replace("-", "").replace("_", "").isalnum():
            return (False, "Short name can only contain letters, numbers, '.', '-', '_'")

        # Validate long_name
        if not long_name or len(long_name) < 3:
            return (False, "Long name must be at least 3 characters")

        # Validate color
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            return (False, "Color must be a tuple/list of at least 3 values")

        if not all(0 <= c <= 1 for c in color[:4]):
            return (False, "Color values must be between 0 and 1")

        return (True, "")


# Global instance (initialized by ApplicationLogic)
_chapter_type_manager_instance: Optional[ChapterTypeManager] = None


def get_chapter_type_manager() -> Optional[ChapterTypeManager]:
    """Get global ChapterTypeManager instance."""
    return _chapter_type_manager_instance


def set_chapter_type_manager(instance: ChapterTypeManager):
    """Set global ChapterTypeManager instance."""
    global _chapter_type_manager_instance
    _chapter_type_manager_instance = instance
