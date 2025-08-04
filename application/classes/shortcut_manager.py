import glfw
from typing import Optional, TYPE_CHECKING
import imgui

if TYPE_CHECKING:
    from logic import ApplicationLogic


class ShortcutManager:
    def __init__(self, app: 'ApplicationLogic'):
        self.app = app
        self.is_recording_shortcut_for: Optional[str] = None
        self._initialize_reverse_key_map() # Call initialization

    def _initialize_reverse_key_map(self):
        """Initializes a map from key names to GLFW key codes."""
        self._reverse_key_map = {}
        # Alphanumeric keys
        for i in range(ord('A'), ord('Z') + 1): self._reverse_key_map[chr(i)] = i
        for i in range(ord('0'), ord('9') + 1): self._reverse_key_map[chr(i)] = i

        # Based on the key_map in glfw_key_to_name
        key_map_direct = {
            "SPACE": glfw.KEY_SPACE, "'": glfw.KEY_APOSTROPHE, ",": glfw.KEY_COMMA,
            "-": glfw.KEY_MINUS, ".": glfw.KEY_PERIOD, "/": glfw.KEY_SLASH,
            ";": glfw.KEY_SEMICOLON, "=": glfw.KEY_EQUAL,
            "[": glfw.KEY_LEFT_BRACKET, "\\": glfw.KEY_BACKSLASH,
            "]": glfw.KEY_RIGHT_BRACKET, "`": glfw.KEY_GRAVE_ACCENT,
            "ENTER": glfw.KEY_ENTER, "TAB": glfw.KEY_TAB,
            "BACKSPACE": glfw.KEY_BACKSPACE, "INSERT": glfw.KEY_INSERT,
            "DELETE": glfw.KEY_DELETE,
            "RIGHT_ARROW": glfw.KEY_RIGHT, "LEFT_ARROW": glfw.KEY_LEFT,
            "DOWN_ARROW": glfw.KEY_DOWN, "UP_ARROW": glfw.KEY_UP,
            "PAGE_UP": glfw.KEY_PAGE_UP, "PAGE_DOWN": glfw.KEY_PAGE_DOWN,
            "HOME": glfw.KEY_HOME, "END": glfw.KEY_END,
            "CAPS_LOCK": glfw.KEY_CAPS_LOCK, "SCROLL_LOCK": glfw.KEY_SCROLL_LOCK,
            "NUM_LOCK": glfw.KEY_NUM_LOCK, "PRINT_SCREEN": glfw.KEY_PRINT_SCREEN,
            "PAUSE": glfw.KEY_PAUSE,
            "ESCAPE": glfw.KEY_ESCAPE,
        }
        self._reverse_key_map.update(key_map_direct)

        for i in range(1, 26):  # F1-F25
            glfw_key_const = getattr(glfw, f"KEY_F{i}", None)
            if glfw_key_const is not None:
                self._reverse_key_map[f"F{i}"] = glfw_key_const

        for i in range(10):  # KP_0-KP_9
            glfw_key_const = getattr(glfw, f"KEY_KP_{i}", None)
            if glfw_key_const is not None:
                self._reverse_key_map[f"KP_{i}"] = glfw_key_const

        keypad_extras = {
            "KP_DECIMAL": glfw.KEY_KP_DECIMAL, "KP_DIVIDE": glfw.KEY_KP_DIVIDE,
            "KP_MULTIPLY": glfw.KEY_KP_MULTIPLY, "KP_SUBTRACT": glfw.KEY_KP_SUBTRACT,
            "KP_ADD": glfw.KEY_KP_ADD, "KP_ENTER": glfw.KEY_KP_ENTER,  # Name used in glfw_key_to_name
            "KP_EQUAL": glfw.KEY_KP_EQUAL
        }
        for name, code in keypad_extras.items():
            if hasattr(glfw, name):  # Check if constant exists (it should for these)
                self._reverse_key_map[name] = getattr(glfw, name)

    def name_to_glfw_key(self, key_name: str) -> Optional[int]:
        """Converts a human-readable key name to its GLFW key code."""
        return self._reverse_key_map.get(key_name.upper())  # Use upper for case-insensitivity

    def start_shortcut_recording(self, action_name: str):
        """Initiates recording for a specific shortcut action."""
        # If already recording for another action, cancel it first
        if self.is_recording_shortcut_for and self.is_recording_shortcut_for != action_name:
            self.cancel_shortcut_recording()  # Cancel previous recording

        self.is_recording_shortcut_for = action_name
        self.app.logger.info(
            f"Recording for '{action_name.replace('_', ' ').title()}'. Press key combination or ESC to cancel.", extra={'status_message': True})

    def cancel_shortcut_recording(self):
        """Cancels the current shortcut recording."""
        if self.is_recording_shortcut_for:
            self.app.logger.info(
                f"Recording for '{self.is_recording_shortcut_for.replace('_', ' ').title()}' cancelled.", extra={'status_message': True})
        self.is_recording_shortcut_for = None

    def finalize_shortcut_recording(self, new_shortcut_str: str):
        """Finalizes recording and updates the shortcut in settings."""
        if self.is_recording_shortcut_for:
            current_shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})
            current_shortcuts[self.is_recording_shortcut_for] = new_shortcut_str
            self.app.app_settings.set("funscript_editor_shortcuts", current_shortcuts)
            self.app.logger.info(
                f"Shortcut for '{self.is_recording_shortcut_for.replace('_', ' ').title()}' set to '{new_shortcut_str}'. Save settings to persist.", extra={'status_message': True})
            if self.app.project_manager:
                self.app.project_manager.project_dirty = True  # Mark settings as changed
            self.is_recording_shortcut_for = None

    def glfw_key_to_name(self, glfw_key_code: int) -> Optional[str]:
        """Converts a GLFW key code to a human-readable string name."""
        # Alphanumeric keys (A-Z, 0-9)
        if glfw.KEY_A <= glfw_key_code <= glfw.KEY_Z:
            return chr(glfw_key_code)
        if glfw.KEY_0 <= glfw_key_code <= glfw.KEY_9:
            return chr(glfw_key_code)

        key_map = {
            glfw.KEY_SPACE: "SPACE", glfw.KEY_APOSTROPHE: "'", glfw.KEY_COMMA: ",",
            glfw.KEY_MINUS: "-", glfw.KEY_PERIOD: ".", glfw.KEY_SLASH: "/",
            glfw.KEY_SEMICOLON: ";", glfw.KEY_EQUAL: "=",
            glfw.KEY_LEFT_BRACKET: "[", glfw.KEY_BACKSLASH: "\\",
            glfw.KEY_RIGHT_BRACKET: "]", glfw.KEY_GRAVE_ACCENT: "`",
            glfw.KEY_ENTER: "ENTER", glfw.KEY_TAB: "TAB",
            glfw.KEY_BACKSPACE: "BACKSPACE", glfw.KEY_INSERT: "INSERT",
            glfw.KEY_DELETE: "DELETE",
            glfw.KEY_RIGHT: "RIGHT_ARROW", glfw.KEY_LEFT: "LEFT_ARROW",
            glfw.KEY_DOWN: "DOWN_ARROW", glfw.KEY_UP: "UP_ARROW",
            glfw.KEY_PAGE_UP: "PAGE_UP", glfw.KEY_PAGE_DOWN: "PAGE_DOWN",
            glfw.KEY_HOME: "HOME", glfw.KEY_END: "END",
            glfw.KEY_CAPS_LOCK: "CAPS_LOCK", glfw.KEY_SCROLL_LOCK: "SCROLL_LOCK",
            glfw.KEY_NUM_LOCK: "NUM_LOCK", glfw.KEY_PRINT_SCREEN: "PRINT_SCREEN",
            glfw.KEY_PAUSE: "PAUSE",
            **{getattr(glfw, f"KEY_F{i}", -i): f"F{i}" for i in range(1, 26)},  # F1-F25
            **{getattr(glfw, f"KEY_KP_{i}", -100 - i): f"KP_{i}" for i in range(10)},  # KP_0-KP_9
            glfw.KEY_KP_DECIMAL: "KP_DECIMAL", glfw.KEY_KP_DIVIDE: "KP_DIVIDE",
            glfw.KEY_KP_MULTIPLY: "KP_MULTIPLY", glfw.KEY_KP_SUBTRACT: "KP_SUBTRACT",
            glfw.KEY_KP_ADD: "KP_ADD", glfw.KEY_KP_ENTER: "KP_ENTER",
            glfw.KEY_KP_EQUAL: "KP_EQUAL"
        }
        name = key_map.get(glfw_key_code)
        if name:
            return name

        # Fallback for other printable keys using glfw.get_key_name
        # This is useful for symbols that vary by keyboard layout but are still "typed"
        try:
            # Check if it's in a range that glfw.get_key_name might handle (often printable range)
            # and not already covered by our more specific map (like SPACE, ENTER etc.)
            if 32 <= glfw_key_code <= 255:  # Broaden range slightly for potential extended ASCII
                key_name_from_glfw = glfw.get_key_name(glfw_key_code,
                                                       0)  # scancode 0 for layout-independent if possible
                if key_name_from_glfw:
                    return key_name_from_glfw.upper()  # Standardize to uppercase
        except Exception:
            pass  # glfw.get_key_name might fail or return None for some codes

        # print(f"Warning: ShortcutManager unmapped GLFW key code: {glfw_key_code}")
        return None

    def handle_shortcut_recording_input(self):
        """Handles keyboard input when in shortcut recording mode."""
        if not self.is_recording_shortcut_for:
            return

        io = imgui.get_io()

        if imgui.is_key_pressed(glfw.KEY_ESCAPE):
            self.cancel_shortcut_recording()
            return

        key_str_parts = []
        main_key_pressed_name = None
        captured_this_frame = False

        # Check for the main key that was pressed this frame
        for key_code in range(glfw.KEY_SPACE, glfw.KEY_LAST + 1):  # Iterate through relevant GLFW key codes
            if imgui.is_key_pressed(key_code):  # Checks for initial press this frame
                # Ignore modifier keys themselves as the "main" key for the shortcut
                is_modifier_key = key_code in [
                    glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL,
                    glfw.KEY_LEFT_ALT, glfw.KEY_RIGHT_ALT,
                    glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT,
                    glfw.KEY_ESCAPE  # Escape is for cancel only
                ]
                if is_modifier_key:
                    continue  # Skip if it's purely a modifier key

                key_name = self.glfw_key_to_name(key_code)
                if key_name:
                    main_key_pressed_name = key_name
                    captured_this_frame = True
                    break  # Found the primary key for this frame

        if captured_this_frame and main_key_pressed_name:
            # Modifiers are checked *after* a main key is confirmed to be pressed
            if io.key_ctrl: key_str_parts.append("CTRL")
            if io.key_alt: key_str_parts.append("ALT")
            if io.key_shift: key_str_parts.append("SHIFT")
            if io.key_super: key_str_parts.append("SUPER")

            key_str_parts.append(main_key_pressed_name)
            final_shortcut_str = "+".join(key_str_parts)
            self.finalize_shortcut_recording(final_shortcut_str)
