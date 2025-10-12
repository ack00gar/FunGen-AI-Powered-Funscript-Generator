"""
Keyboard Layout Detection and Management

Detects keyboard layout (QWERTY, AZERTY, QWERTZ) and provides
layout-aware shortcuts for better international support.
"""

import platform
import locale
import subprocess
import glfw


class KeyboardLayout:
    """Represents a keyboard layout with its characteristics"""

    QWERTY = "QWERTY"
    AZERTY = "AZERTY"
    QWERTZ = "QWERTZ"

    def __init__(self, name: str):
        self.name = name
        self.physical_key_mappings = self._get_physical_mappings()

    def _get_physical_mappings(self):
        """
        Map logical shortcuts to physical keys for this layout.

        Key differences:
        - QWERTY: . and , are on their own keys
        - AZERTY: . is SHIFT+; and , is ; (semicolon key)
        - QWERTZ: . and , same as QWERTY, but Y/Z swapped
        """
        if self.name == self.AZERTY:
            return {
                # On AZERTY, the semicolon key is where period/comma are
                "period_key": "SEMICOLON",  # Requires SHIFT for .
                "comma_key": "SEMICOLON",   # No shift for ,
                "period_modifier": "SHIFT",
                "comma_modifier": None,
            }
        else:  # QWERTY and QWERTZ have same punctuation
            return {
                "period_key": "PERIOD",
                "comma_key": "COMMA",
                "period_modifier": None,
                "comma_modifier": None,
            }

    def get_shortcut_for_action(self, action_name: str, default_shortcut: str):
        """
        Get the appropriate shortcut for an action based on this layout.

        Args:
            action_name: The action identifier (e.g., "jump_to_next_point")
            default_shortcut: The QWERTY default shortcut

        Returns:
            Layout-appropriate shortcut string
        """
        # For AZERTY, we need to adjust period/comma shortcuts
        if self.name == self.AZERTY:
            if default_shortcut == ".":
                # On AZERTY, period is SHIFT+;
                return "SHIFT+SEMICOLON"
            elif default_shortcut == ",":
                # On AZERTY, comma is just ;
                return "SEMICOLON"

        # QWERTY and QWERTZ use the same shortcuts (for now)
        return default_shortcut


class KeyboardLayoutDetector:
    """Detects and manages keyboard layout"""

    def __init__(self, app_settings=None):
        self.app_settings = app_settings
        self.detected_layout = None
        self._detect_layout()

    def _detect_layout(self):
        """
        Attempt to detect keyboard layout from OS settings.

        Detection strategy:
        1. Check saved user preference in settings
        2. Check platform-specific keyboard APIs (macOS, Windows, Linux)
        3. Check system locale as fallback
        4. Default to QWERTY
        """
        # Priority 1: Check saved user preference
        if self.app_settings:
            saved_layout = self.app_settings.get("keyboard_layout", None)
            if saved_layout in [KeyboardLayout.QWERTY, KeyboardLayout.AZERTY, KeyboardLayout.QWERTZ]:
                self.detected_layout = KeyboardLayout(saved_layout)
                return

        # Priority 2: Platform-specific detection
        detected = self._detect_platform_specific()
        if detected:
            self.detected_layout = KeyboardLayout(detected)
            return

        # Priority 3: Try to detect from locale (less reliable)
        try:
            system_locale = locale.getdefaultlocale()[0] or ""

            # French-speaking regions typically use AZERTY
            if system_locale.startswith(('fr_', 'be_', 'FR_', 'BE_')):
                self.detected_layout = KeyboardLayout(KeyboardLayout.AZERTY)
                return

            # German-speaking regions typically use QWERTZ
            if system_locale.startswith(('de_', 'at_', 'ch_', 'DE_', 'AT_', 'CH_')):
                self.detected_layout = KeyboardLayout(KeyboardLayout.QWERTZ)
                return

        except Exception:
            pass  # Fall back to default

        # Priority 4: Default to QWERTY (most common)
        self.detected_layout = KeyboardLayout(KeyboardLayout.QWERTY)

    def _detect_platform_specific(self):
        """
        Detect keyboard layout using platform-specific methods.

        Returns:
            Layout name string or None if detection fails
        """
        system = platform.system()

        if system == "Darwin":  # macOS
            return self._detect_macos_layout()
        elif system == "Windows":
            return self._detect_windows_layout()
        elif system == "Linux":
            return self._detect_linux_layout()

        return None

    def _detect_macos_layout(self):
        """Detect keyboard layout on macOS using defaults command"""
        try:
            # Try to get keyboard layout from macOS system
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleKeyboardUIMode"],
                capture_output=True,
                text=True,
                timeout=1
            )

            # Alternative: check current input source
            result = subprocess.run(
                ["defaults", "read", "com.apple.HIToolbox", "AppleSelectedInputSources"],
                capture_output=True,
                text=True,
                timeout=1
            )

            output = result.stdout.lower()

            # Check for French keyboard indicators
            if 'french' in output or 'azerty' in output or 'fr_' in output:
                return KeyboardLayout.AZERTY

            # Check for German keyboard indicators
            if 'german' in output or 'qwertz' in output or 'de_' in output:
                return KeyboardLayout.QWERTZ

        except Exception:
            pass

        return None

    def _detect_windows_layout(self):
        """Detect keyboard layout on Windows"""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            layout_id = user32.GetKeyboardLayout(0) & 0xFFFF

            # French layouts: 0x040c (French - France), 0x080c (French - Belgium)
            if layout_id in [0x040c, 0x080c]:
                return KeyboardLayout.AZERTY

            # German layouts: 0x0407 (German - Germany), 0x0807 (German - Switzerland)
            if layout_id in [0x0407, 0x0807, 0x0c07, 0x1007, 0x1407]:
                return KeyboardLayout.QWERTZ

        except Exception:
            pass

        return None

    def _detect_linux_layout(self):
        """Detect keyboard layout on Linux"""
        try:
            # Try using setxkbmap to get current layout
            result = subprocess.run(
                ["setxkbmap", "-query"],
                capture_output=True,
                text=True,
                timeout=1
            )

            output = result.stdout.lower()

            # Check for layout indicators
            if 'fr' in output or 'azerty' in output:
                return KeyboardLayout.AZERTY

            if 'de' in output or 'qwertz' in output:
                return KeyboardLayout.QWERTZ

        except Exception:
            pass

        return None

    def get_layout(self) -> KeyboardLayout:
        """Get the detected keyboard layout"""
        return self.detected_layout

    def set_layout(self, layout_name: str):
        """Manually set the keyboard layout and save to settings"""
        if layout_name in [KeyboardLayout.QWERTY, KeyboardLayout.AZERTY, KeyboardLayout.QWERTZ]:
            self.detected_layout = KeyboardLayout(layout_name)
            # Save user preference
            if self.app_settings:
                self.app_settings.set("keyboard_layout", layout_name)

    def get_available_layouts(self):
        """Get list of available layout names"""
        return [KeyboardLayout.QWERTY, KeyboardLayout.AZERTY, KeyboardLayout.QWERTZ]

    def get_layout_adjusted_shortcuts(self, default_shortcuts: dict) -> dict:
        """
        Get a dictionary of shortcuts adjusted for the detected layout.

        Args:
            default_shortcuts: Dictionary of action_name -> shortcut_string (QWERTY defaults)

        Returns:
            Dictionary of action_name -> layout-adjusted shortcut_string
        """
        adjusted = {}
        layout = self.get_layout()

        for action_name, default_shortcut in default_shortcuts.items():
            adjusted[action_name] = layout.get_shortcut_for_action(action_name, default_shortcut)

        return adjusted

    def get_layout_info_text(self) -> str:
        """Get human-readable layout information"""
        layout = self.get_layout()

        info = f"Detected Layout: {layout.name}\n\n"

        if layout.name == KeyboardLayout.AZERTY:
            info += "AZERTY Layout Adjustments:\n"
            info += "- Jump to Next Point: SHIFT+; (instead of .)\n"
            info += "- Jump to Previous Point: ; (instead of ,)\n"
            info += "\nNote: These adjustments match your physical keyboard layout."
        elif layout.name == KeyboardLayout.QWERTZ:
            info += "QWERTZ Layout:\n"
            info += "- No punctuation adjustments needed\n"
            info += "- Period and comma work as expected\n"
        else:  # QWERTY
            info += "QWERTY Layout:\n"
            info += "- Standard US keyboard layout\n"
            info += "- No adjustments needed\n"

        return info


# Global instance
_layout_detector = None


def get_layout_detector() -> KeyboardLayoutDetector:
    """Get the global keyboard layout detector instance"""
    global _layout_detector
    if _layout_detector is None:
        _layout_detector = KeyboardLayoutDetector()
    return _layout_detector
