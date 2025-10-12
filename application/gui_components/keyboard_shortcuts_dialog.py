"""
Unified Keyboard Shortcuts Dialog

Provides both discovery and customization of keyboard shortcuts in one place:
- Organized by category for easy discovery
- Click "Customize" to rebind any shortcut
- Search/filter functionality
- Platform-aware display (CMD on macOS, CTRL elsewhere)
- Reset individual or all shortcuts to defaults

Access via:
- F1 key (global shortcut)
- Help menu → Keyboard Shortcuts
"""

import imgui
import glfw
import platform


class KeyboardShortcutsDialog:
    """Unified keyboard shortcuts dialog - discovery + customization."""

    def __init__(self, app):
        self.app = app
        self.is_open = False
        self.search_filter = ""
        self.shortcut_categories = self._organize_shortcuts()
        self._is_macos = platform.system() == "Darwin"

    def _organize_shortcuts(self):
        """Group shortcuts by category for organized display"""
        return {
            "File": [
                ("save_project", "Save Project"),
                ("open_project", "Open Project"),
            ],
            "Playback": [
                ("toggle_playback", "Toggle Play/Pause"),
                ("seek_next_frame", "Next Frame"),
                ("seek_prev_frame", "Previous Frame"),
            ],
            "Video Navigation": [
                ("jump_to_start", "Jump to Start"),
                ("jump_to_end", "Jump to End"),
                ("pan_timeline_left", "Pan Timeline Left"),
                ("pan_timeline_right", "Pan Timeline Right"),
            ],
            "Timeline View": [
                ("zoom_in_timeline", "Zoom In Timeline"),
                ("zoom_out_timeline", "Zoom Out Timeline"),
            ],
            "Window Toggles": [
                ("toggle_video_display", "Toggle Video Display"),
                ("toggle_timeline2", "Toggle Timeline 2"),
                ("toggle_gauge_window", "Toggle Gauge Window"),
                ("toggle_3d_simulator", "Toggle 3D Simulator"),
                ("toggle_movement_bar", "Toggle Movement Bar"),
                ("toggle_chapter_list", "Toggle Chapter List"),
            ],
            "Timeline Displays": [
                ("toggle_heatmap", "Toggle Heatmap"),
                ("toggle_funscript_preview", "Toggle Funscript Preview"),
            ],
            "Video Overlays": [
                ("toggle_video_feed", "Toggle Video Feed"),
                ("toggle_waveform", "Toggle Audio Waveform"),
            ],
            "View Controls": [
                ("reset_timeline_view", "Reset Timeline Zoom/Pan"),
            ],
            "Editing": [
                ("undo_timeline1", "Undo (Timeline 1)"),
                ("redo_timeline1", "Redo (Timeline 1)"),
                ("undo_timeline2", "Undo (Timeline 2)"),
                ("redo_timeline2", "Redo (Timeline 2)"),
                ("select_all_points", "Select All Points"),
                ("deselect_all_points", "Deselect All Points"),
                ("delete_selected_point", "Delete Point"),
                ("delete_selected_point_alt", "Delete Point (Alt)"),
                ("copy_selection", "Copy Selection"),
                ("paste_selection", "Paste Selection"),
            ],
            "Point Navigation": [
                ("jump_to_next_point", "Jump to Next Point"),
                ("jump_to_prev_point", "Jump to Previous Point"),
            ],
            "Chapters": [
                ("set_chapter_start", "Set Chapter Start (In-point)"),
                ("set_chapter_end", "Set Chapter End (Out-point)"),
                ("delete_selected_chapter", "Delete Chapter"),
                ("delete_selected_chapter_alt", "Delete Chapter (Alt)"),
            ],
            "Add Points": [
                ("add_point_0", "Add Point at 0%"),
                ("add_point_10", "Add Point at 10%"),
                ("add_point_20", "Add Point at 20%"),
                ("add_point_30", "Add Point at 30%"),
                ("add_point_40", "Add Point at 40%"),
                ("add_point_50", "Add Point at 50%"),
                ("add_point_60", "Add Point at 60%"),
                ("add_point_70", "Add Point at 70%"),
                ("add_point_80", "Add Point at 80%"),
                ("add_point_90", "Add Point at 90%"),
                ("add_point_100", "Add Point at 100%"),
            ],
        }

    def render(self):
        """Render the keyboard shortcuts dialog"""
        if not self.is_open:
            return

        # Center on screen
        viewport = imgui.get_main_viewport()
        dialog_width = 700
        dialog_height = 550
        imgui.set_next_window_size(dialog_width, dialog_height, imgui.ONCE)
        imgui.set_next_window_position(
            viewport.pos.x + (viewport.size.x - dialog_width) / 2,
            viewport.pos.y + (viewport.size.y - dialog_height) / 2,
            imgui.ONCE
        )

        expanded, opened = imgui.begin("Keyboard Shortcuts", True)

        if not opened:
            self.is_open = False
            imgui.end()
            return

        if expanded:
            # Help text
            imgui.text_wrapped(
                "View all keyboard shortcuts. Click 'Customize' to rebind any shortcut."
            )
            imgui.spacing()

            # Search filter
            changed, self.search_filter = imgui.input_text(
                "##ShortcutsSearch",
                self.search_filter,
                256,
                imgui.INPUT_TEXT_AUTO_SELECT_ALL
            )

            if imgui.is_item_hovered():
                imgui.set_tooltip("Filter shortcuts by name or key")

            imgui.same_line()
            if imgui.button("Clear##ClearSearch"):
                self.search_filter = ""

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Render categories
            shortcuts_settings = self.app.app_settings.get("funscript_editor_shortcuts", {})
            sm = self.app.shortcut_manager

            # Scrollable area for shortcuts list
            if imgui.begin_child("ShortcutsList", height=-40):
                for category_name, shortcuts_list in self.shortcut_categories.items():
                    # Filter shortcuts
                    visible_shortcuts = self._filter_shortcuts(shortcuts_list, shortcuts_settings)

                    if not visible_shortcuts:
                        continue

                    # Category header
                    if imgui.collapsing_header(
                        f"{category_name}##ShortcutCategory",
                        flags=imgui.TREE_NODE_DEFAULT_OPEN
                    )[0]:
                        imgui.spacing()
                        # Render each shortcut in category
                        for action_name, display_name in visible_shortcuts:
                            self._render_shortcut_row(
                                action_name,
                                display_name,
                                shortcuts_settings,
                                sm
                            )
                        imgui.spacing()

                imgui.end_child()

            imgui.separator()

            # Bottom buttons
            if imgui.button("Reset All to Defaults##ResetAllShortcuts", width=180):
                imgui.open_popup("ConfirmResetShortcuts")

            if imgui.is_item_hovered():
                imgui.set_tooltip("Reset all shortcuts to their default values")

            imgui.same_line()

            # Spacer
            imgui.dummy(imgui.get_content_region_available_width() - 100, 0)
            imgui.same_line()

            if imgui.button("Close##CloseShortcutsDialog", width=100):
                self.is_open = False

            # Confirmation popup for reset
            self._render_reset_confirmation_popup()

        imgui.end()

    def _filter_shortcuts(self, shortcuts_list, shortcuts_settings):
        """Filter shortcuts based on search query"""
        if not self.search_filter:
            return shortcuts_list

        search_lower = self.search_filter.lower()
        visible_shortcuts = []

        for action_name, display_name in shortcuts_list:
            # Check if search matches display name or action name
            if search_lower in display_name.lower() or search_lower in action_name.lower():
                visible_shortcuts.append((action_name, display_name))
                continue

            # Also check if search matches the current key binding
            current_key = shortcuts_settings.get(action_name, "")
            if search_lower in current_key.lower():
                visible_shortcuts.append((action_name, display_name))

        return visible_shortcuts

    def _render_shortcut_row(self, action_name, display_name, shortcuts_settings, sm):
        """Render a single shortcut row with customize button"""
        # Get current binding
        current_key = shortcuts_settings.get(action_name, "Not Set")

        # Check if currently recording this shortcut
        is_recording = (sm.is_recording_shortcut_for == action_name)

        # Display name (left aligned)
        imgui.text(display_name)

        # Current key binding (middle, colored)
        imgui.same_line(position=350)  # Align all keys at same position

        if is_recording:
            # Show recording indicator
            imgui.text_colored("🎹 PRESS KEY...", 1.0, 0.5, 0.0, 1.0)
        else:
            # Platform-aware display (show CMD instead of SUPER on macOS)
            display_key = self._platform_aware_key_display(current_key)
            imgui.text_colored(display_key, 0.6, 0.8, 1.0, 1.0)

        # Customize/Cancel button (right aligned)
        imgui.same_line(position=530)

        button_text = "Cancel" if is_recording else "Customize"
        button_width = 90

        if imgui.button(f"{button_text}##{action_name}", width=button_width):
            if is_recording:
                sm.cancel_shortcut_recording()
            else:
                sm.start_shortcut_recording(action_name)

    def _platform_aware_key_display(self, key_str):
        """Convert SUPER to CMD on macOS for display"""
        if self._is_macos:
            return key_str.replace("SUPER", "CMD")
        return key_str

    def _render_reset_confirmation_popup(self):
        """Confirmation dialog for resetting all shortcuts"""
        if imgui.begin_popup_modal(
            "ConfirmResetShortcuts",
            True,
            imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )[0]:
            imgui.text("Reset all keyboard shortcuts to default values?")
            imgui.spacing()
            imgui.text_colored("This cannot be undone.", 0.9, 0.6, 0.2, 1.0)
            imgui.spacing()

            # Show warning if customizations exist
            shortcuts_settings = self.app.app_settings.get("funscript_editor_shortcuts", {})
            from config.constants import DEFAULT_SHORTCUTS

            customized_count = sum(
                1 for action_name, key_str in shortcuts_settings.items()
                if key_str != DEFAULT_SHORTCUTS.get(action_name, "")
            )

            if customized_count > 0:
                imgui.text(f"You have {customized_count} customized shortcut(s).")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Buttons
            if imgui.button("Reset All##ConfirmResetBtn", width=120):
                # Reset to defaults from constants.py
                self.app.app_settings.set("funscript_editor_shortcuts", dict(DEFAULT_SHORTCUTS))
                self.app.logger.info("All keyboard shortcuts reset to defaults", extra={'status_message': True})
                imgui.close_current_popup()

            imgui.same_line()
            if imgui.button("Cancel##CancelResetBtn", width=120):
                imgui.close_current_popup()

            imgui.end_popup()

    def open(self):
        """Open the dialog"""
        self.is_open = True

    def close(self):
        """Close the dialog"""
        self.is_open = False

    def toggle(self):
        """Toggle dialog open/close"""
        self.is_open = not self.is_open
