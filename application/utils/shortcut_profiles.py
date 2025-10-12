"""
Keyboard Shortcut Profiles System

Allows users to create, save, and switch between different shortcut profiles
for different workflows (e.g., "Video Editor", "Funscript Expert", "Custom").
"""

import json
import os
from typing import Dict, List, Optional
from config.constants import DEFAULT_SHORTCUTS


class ShortcutProfile:
    """Represents a keyboard shortcut profile"""

    def __init__(self, name: str, shortcuts: Dict[str, str], description: str = ""):
        self.name = name
        self.shortcuts = shortcuts.copy()
        self.description = description
        self.is_builtin = False  # Whether this is a built-in profile

    def to_dict(self) -> dict:
        """Convert profile to dictionary for serialization"""
        return {
            "name": self.name,
            "shortcuts": self.shortcuts,
            "description": self.description,
            "is_builtin": self.is_builtin
        }

    @staticmethod
    def from_dict(data: dict) -> 'ShortcutProfile':
        """Create profile from dictionary"""
        profile = ShortcutProfile(
            name=data.get("name", "Unnamed"),
            shortcuts=data.get("shortcuts", {}),
            description=data.get("description", "")
        )
        profile.is_builtin = data.get("is_builtin", False)
        return profile


class ShortcutProfileManager:
    """Manages keyboard shortcut profiles"""

    def __init__(self, app_settings):
        self.app_settings = app_settings
        self.profiles: Dict[str, ShortcutProfile] = {}
        self.active_profile_name: Optional[str] = None

        # Initialize with built-in profiles
        self._create_builtin_profiles()

        # Load custom profiles from settings
        self._load_custom_profiles()

        # Load active profile name
        self.active_profile_name = self.app_settings.get("active_shortcut_profile", "Default")

    def _create_builtin_profiles(self):
        """Create built-in shortcut profiles"""

        # Default profile (current shortcuts)
        default_profile = ShortcutProfile(
            name="Default",
            shortcuts=DEFAULT_SHORTCUTS.copy(),
            description="Standard FunGen shortcuts with all features"
        )
        default_profile.is_builtin = True
        self.profiles["Default"] = default_profile

        # Video Editor profile (focused on video navigation)
        video_editor_shortcuts = DEFAULT_SHORTCUTS.copy()
        # Emphasize video controls, simplify other shortcuts
        video_editor_profile = ShortcutProfile(
            name="Video Editor",
            shortcuts=video_editor_shortcuts,
            description="Optimized for video navigation and playback control"
        )
        video_editor_profile.is_builtin = True
        self.profiles["Video Editor"] = video_editor_profile

        # Funscript Expert profile (focused on editing)
        expert_shortcuts = DEFAULT_SHORTCUTS.copy()
        # Keep all editing shortcuts, streamline view toggles
        expert_profile = ShortcutProfile(
            name="Funscript Expert",
            shortcuts=expert_shortcuts,
            description="Advanced shortcuts for experienced funscript editors"
        )
        expert_profile.is_builtin = True
        self.profiles["Funscript Expert"] = expert_profile

        # Minimal profile (only essential shortcuts)
        minimal_shortcuts = {
            # Only core functionality
            "toggle_playback": "SPACE",
            "seek_next_frame": "RIGHT_ARROW",
            "seek_prev_frame": "LEFT_ARROW",
            "save_project": DEFAULT_SHORTCUTS["save_project"],
            "undo_timeline1": DEFAULT_SHORTCUTS["undo_timeline1"],
            "redo_timeline1": DEFAULT_SHORTCUTS["redo_timeline1"],
            # Copy other defaults to avoid missing shortcuts
            **DEFAULT_SHORTCUTS
        }
        minimal_profile = ShortcutProfile(
            name="Minimal",
            shortcuts=minimal_shortcuts,
            description="Only essential shortcuts for beginners"
        )
        minimal_profile.is_builtin = True
        self.profiles["Minimal"] = minimal_profile

    def _load_custom_profiles(self):
        """Load custom profiles from settings"""
        custom_profiles_data = self.app_settings.get("custom_shortcut_profiles", [])

        for profile_data in custom_profiles_data:
            try:
                profile = ShortcutProfile.from_dict(profile_data)
                if not profile.is_builtin:  # Don't overwrite built-ins
                    self.profiles[profile.name] = profile
            except Exception as e:
                print(f"Error loading profile: {e}")

    def _save_custom_profiles(self):
        """Save custom profiles to settings"""
        custom_profiles = [
            profile.to_dict()
            for profile in self.profiles.values()
            if not profile.is_builtin
        ]
        self.app_settings.set("custom_shortcut_profiles", custom_profiles)

    def get_profile(self, name: str) -> Optional[ShortcutProfile]:
        """Get a profile by name"""
        return self.profiles.get(name)

    def get_active_profile(self) -> ShortcutProfile:
        """Get the currently active profile"""
        if self.active_profile_name and self.active_profile_name in self.profiles:
            return self.profiles[self.active_profile_name]
        return self.profiles["Default"]

    def set_active_profile(self, name: str) -> bool:
        """
        Set the active profile and apply its shortcuts.

        Returns:
            True if successful, False if profile doesn't exist
        """
        if name not in self.profiles:
            return False

        self.active_profile_name = name
        self.app_settings.set("active_shortcut_profile", name)

        # Apply the profile's shortcuts to current settings
        profile = self.profiles[name]
        self.app_settings.set("funscript_editor_shortcuts", profile.shortcuts.copy())

        return True

    def create_profile(self, name: str, shortcuts: Dict[str, str], description: str = "") -> bool:
        """
        Create a new custom profile.

        Returns:
            True if successful, False if name already exists
        """
        if name in self.profiles:
            return False

        profile = ShortcutProfile(name, shortcuts, description)
        profile.is_builtin = False
        self.profiles[name] = profile

        self._save_custom_profiles()
        return True

    def duplicate_profile(self, source_name: str, new_name: str) -> bool:
        """
        Duplicate an existing profile with a new name.

        Returns:
            True if successful, False if source doesn't exist or new name taken
        """
        if source_name not in self.profiles or new_name in self.profiles:
            return False

        source = self.profiles[source_name]
        new_profile = ShortcutProfile(
            name=new_name,
            shortcuts=source.shortcuts.copy(),
            description=f"Copy of {source_name}"
        )
        new_profile.is_builtin = False
        self.profiles[new_name] = new_profile

        self._save_custom_profiles()
        return True

    def delete_profile(self, name: str) -> bool:
        """
        Delete a custom profile (cannot delete built-in profiles).

        Returns:
            True if successful, False if profile is built-in or doesn't exist
        """
        if name not in self.profiles:
            return False

        profile = self.profiles[name]
        if profile.is_builtin:
            return False  # Cannot delete built-in profiles

        # Don't delete if it's the active profile
        if name == self.active_profile_name:
            self.set_active_profile("Default")

        del self.profiles[name]
        self._save_custom_profiles()
        return True

    def rename_profile(self, old_name: str, new_name: str) -> bool:
        """
        Rename a custom profile.

        Returns:
            True if successful, False if profile is built-in, doesn't exist, or new name taken
        """
        if old_name not in self.profiles or new_name in self.profiles:
            return False

        profile = self.profiles[old_name]
        if profile.is_builtin:
            return False  # Cannot rename built-in profiles

        profile.name = new_name
        self.profiles[new_name] = profile
        del self.profiles[old_name]

        # Update active profile name if needed
        if self.active_profile_name == old_name:
            self.active_profile_name = new_name
            self.app_settings.set("active_shortcut_profile", new_name)

        self._save_custom_profiles()
        return True

    def get_profile_names(self) -> List[str]:
        """Get list of all profile names"""
        return list(self.profiles.keys())

    def get_builtin_profile_names(self) -> List[str]:
        """Get list of built-in profile names"""
        return [name for name, profile in self.profiles.items() if profile.is_builtin]

    def get_custom_profile_names(self) -> List[str]:
        """Get list of custom profile names"""
        return [name for name, profile in self.profiles.items() if not profile.is_builtin]

    def export_profile(self, name: str, filepath: str) -> bool:
        """
        Export a profile to JSON file.

        Returns:
            True if successful, False if profile doesn't exist or export fails
        """
        if name not in self.profiles:
            return False

        try:
            profile_data = self.profiles[name].to_dict()
            with open(filepath, 'w') as f:
                json.dump(profile_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting profile: {e}")
            return False

    def import_profile(self, filepath: str) -> Optional[str]:
        """
        Import a profile from JSON file.

        Returns:
            Profile name if successful, None if import fails
        """
        try:
            with open(filepath, 'r') as f:
                profile_data = json.load(f)

            profile = ShortcutProfile.from_dict(profile_data)
            profile.is_builtin = False  # Imported profiles are always custom

            # Ensure unique name
            base_name = profile.name
            counter = 1
            while profile.name in self.profiles:
                profile.name = f"{base_name} ({counter})"
                counter += 1

            self.profiles[profile.name] = profile
            self._save_custom_profiles()

            return profile.name
        except Exception as e:
            print(f"Error importing profile: {e}")
            return None

    def detect_conflicts(self, shortcuts: Dict[str, str]) -> List[tuple]:
        """
        Detect conflicting shortcuts in a shortcuts dictionary.

        Returns:
            List of (shortcut_string, [action_names]) tuples for conflicts
        """
        conflicts = {}

        for action_name, shortcut_str in shortcuts.items():
            if shortcut_str:
                if shortcut_str not in conflicts:
                    conflicts[shortcut_str] = []
                conflicts[shortcut_str].append(action_name)

        # Return only actual conflicts (2+ actions with same shortcut)
        return [(shortcut, actions) for shortcut, actions in conflicts.items() if len(actions) > 1]
