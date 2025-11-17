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
from application.utils.keyboard_layout_detector import get_layout_detector, KeyboardLayout
from application.utils import get_icon_texture_manager


class KeyboardShortcutsDialog:
    """Unified keyboard shortcuts dialog - discovery + customization."""

    def __init__(self, app):
        self.app = app
        self.is_open = False
        self.search_filter = ""
        self.shortcut_categories = self._organize_shortcuts()
        self._is_macos = platform.system() == "Darwin"

        # Keyboard layout detection
        from application.utils.keyboard_layout_detector import KeyboardLayoutDetector
        self.layout_detector = KeyboardLayoutDetector(app.app_settings)
        self.selected_layout_idx = self._get_layout_index(self.layout_detector.get_layout().name)

        # Cheat sheet window state
        self.show_cheat_sheet = False

    def _get_layout_index(self, layout_name: str) -> int:
        """Get index of layout in available layouts list"""
        layouts = self.layout_detector.get_available_layouts()
        try:
            return layouts.index(layout_name)
        except ValueError:
            return 0

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
                ("jump_to_next_point", "Jump to Next Point (Primary)"),
                ("jump_to_next_point_alt", "Jump to Next Point (Alt)"),
                ("jump_to_prev_point", "Jump to Previous Point (Primary)"),
                ("jump_to_prev_point_alt", "Jump to Previous Point (Alt)"),
                ("nudge_selection_pos_up", "Raise Selected Point Value"),
                ("nudge_selection_pos_down", "Lower Selected Point Value"),
            ],
            "Chapters": [
                ("set_chapter_start", "Set Chapter Start (In-point)"),
                ("set_chapter_end", "Set Chapter End (Out-point)"),
                ("delete_selected_chapter", "Delete Chapter"),
                ("delete_selected_chapter_alt", "Delete Chapter (Alt)"),
                ("delete_points_in_chapter", "Delete Points in Chapter"),
                ("delete_points_in_chapter_alt", "Delete Points in Chapter (Alt)"),
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
        dialog_width = 800
        dialog_height = 600
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
            # Tab bar
            if imgui.begin_tab_bar("ShortcutTabs"):
                # Shortcuts Tab
                if imgui.begin_tab_item("Shortcuts")[0]:
                    self._render_shortcuts_tab()
                    imgui.end_tab_item()

                # Settings Tab
                if imgui.begin_tab_item("Settings")[0]:
                    self._render_settings_tab()
                    imgui.end_tab_item()

                imgui.end_tab_bar()

            # Confirmation popups
            self._render_reset_confirmation_popup()

        imgui.end()

        # Render cheat sheet in separate window if open
        if self.show_cheat_sheet:
            self._render_cheat_sheet()

    def _render_shortcuts_tab(self):
        """Render the main shortcuts list tab"""
        # Help text
        imgui.text_wrapped(
            "View all keyboard shortcuts. Click 'Customize' to rebind any shortcut."
        )
        imgui.spacing()

        # Cheat sheet button
        if imgui.button("Cheat Sheet"):
            self.show_cheat_sheet = True

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

        # Conflict detection warning
        shortcuts_settings = self.app.app_settings.get("funscript_editor_shortcuts", {})
        conflicts = self._detect_conflicts(shortcuts_settings)
        if conflicts:
            # Get warning icon
            icon_mgr = get_icon_texture_manager()
            warning_tex, _, _ = icon_mgr.get_icon_texture('warning.png')

            if warning_tex:
                imgui.image(warning_tex, 20, 20)
                imgui.same_line()

            imgui.text_colored("Warning: Shortcut conflicts detected!", 1.0, 0.6, 0.0, 1.0)

            if imgui.is_item_hovered():
                tooltip = "The following shortcuts are assigned to multiple actions:\n\n"
                for shortcut, actions in conflicts[:5]:  # Show first 5
                    tooltip += f"{shortcut}:\n"
                    for action in actions:
                        tooltip += f"  - {action}\n"
                    tooltip += "\n"
                if len(conflicts) > 5:
                    tooltip += f"...and {len(conflicts) - 5} more conflicts\n\n"
                tooltip += "Click 'Customize' on conflicting shortcuts to resolve them."
                imgui.set_tooltip(tooltip)

            imgui.spacing()

            # Show expandable list of conflicts
            if imgui.collapsing_header("View Conflicts##ConflictsList")[0]:
                imgui.spacing()
                for shortcut, actions in conflicts:
                    imgui.bullet_text(f"{shortcut}:")
                    imgui.indent()
                    for action in actions:
                        # Find display name for this action
                        display_name = action
                        for category_shortcuts in self.shortcut_categories.values():
                            for act_name, disp_name in category_shortcuts:
                                if act_name == action:
                                    display_name = disp_name
                                    break

                        imgui.text(f"- {display_name}")
                        imgui.same_line()
                        if imgui.small_button(f"Clear##{action}"):
                            # Clear this specific conflicting shortcut
                            shortcuts_settings[action] = ""
                            self.app.app_settings.set("funscript_editor_shortcuts", shortcuts_settings)
                            self.app.logger.info(f"Cleared shortcut for: {display_name}", extra={'status_message': True})
                    imgui.unindent()
                    imgui.spacing()
                imgui.spacing()

        imgui.separator()
        imgui.spacing()

        # Render categories
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

    def _detect_conflicts(self, shortcuts):
        """
        Detect conflicting shortcuts in a shortcuts dictionary.

        Returns:
            List of (shortcut_string, [action_names]) tuples for conflicts
        """
        conflicts_map = {}

        for action_name, shortcut_str in shortcuts.items():
            if shortcut_str:
                if shortcut_str not in conflicts_map:
                    conflicts_map[shortcut_str] = []
                conflicts_map[shortcut_str].append(action_name)

        # Return only actual conflicts (2+ actions with same shortcut)
        return [(shortcut, actions) for shortcut, actions in conflicts_map.items() if len(actions) > 1]

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
            # Show recording indicator with edit icon
            icon_mgr = get_icon_texture_manager()
            edit_tex, _, _ = icon_mgr.get_icon_texture('edit.png')

            if edit_tex:
                imgui.image(edit_tex, 16, 16)
                imgui.same_line()

            imgui.text_colored("PRESS KEY...", 1.0, 0.5, 0.0, 1.0)
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

    def _render_settings_tab(self):
        """Render the settings tab"""
        imgui.text_wrapped(
            "Configure keyboard layout and other shortcut-related settings."
        )
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Keyboard layout section
        imgui.text_colored("Keyboard Layout Configuration", 0.6, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text_wrapped(
            "Select your physical keyboard layout. This adjusts shortcuts to match "
            "your keyboard's key positions (e.g., period and comma keys on AZERTY)."
        )
        imgui.spacing()

        layouts = self.layout_detector.get_available_layouts()
        imgui.text("Your Keyboard Layout:")
        imgui.same_line()
        imgui.set_next_item_width(200)

        changed, self.selected_layout_idx = imgui.combo(
            "##LayoutSelector",
            self.selected_layout_idx,
            layouts
        )

        if changed:
            layout_name = layouts[self.selected_layout_idx]
            self.layout_detector.set_layout(layout_name)
            self.app.logger.info(f"Keyboard layout set to: {layout_name}", extra={'status_message': True})

        imgui.spacing()

        # Layout info
        layout_info = self.layout_detector.get_layout_info_text()
        imgui.text_wrapped(layout_info)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Apply layout adjustments button
        imgui.text_wrapped(
            "After changing your keyboard layout, apply the adjustments to update "
            "your shortcuts automatically:"
        )
        imgui.spacing()

        if imgui.button("Apply Layout Adjustments to Active Profile", width=300):
            current_shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})
            from config.constants import DEFAULT_SHORTCUTS
            adjusted_shortcuts = self.layout_detector.get_layout_adjusted_shortcuts(DEFAULT_SHORTCUTS)
            self.app.app_settings.set("funscript_editor_shortcuts", adjusted_shortcuts)
            self.app.logger.info("Applied layout adjustments to shortcuts", extra={'status_message': True})

        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Adjusts shortcuts like period/comma for your keyboard layout.\n"
                "Example: On AZERTY, '.' becomes 'SHIFT+;' and ',' becomes ';'\n"
                "This modifies the active profile's shortcuts."
            )

    def _render_cheat_sheet(self):
        """Render the keyboard shortcuts cheat sheet window"""
        imgui.set_next_window_size(600, 700, imgui.ONCE)

        expanded, opened = imgui.begin("Keyboard Shortcuts Cheat Sheet", True)

        if not opened:
            self.show_cheat_sheet = False
            imgui.end()
            return

        if expanded:
            imgui.text_wrapped(
                "Quick reference for all keyboard shortcuts. This can be printed or kept open while working."
            )
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            shortcuts_settings = self.app.app_settings.get("funscript_editor_shortcuts", {})

            if imgui.begin_child("CheatSheetContent", height=-40):
                for category_name, shortcuts_list in self.shortcut_categories.items():
                    # Category header
                    imgui.text_colored(category_name, 0.6, 0.8, 1.0, 1.0)
                    imgui.separator()
                    imgui.spacing()

                    # Shortcuts in this category
                    for action_name, display_name in shortcuts_list:
                        current_key = shortcuts_settings.get(action_name, "Not Set")
                        display_key = self._platform_aware_key_display(current_key)

                        # Format: Action name................Shortcut
                        imgui.text(f"{display_name}")
                        imgui.same_line(position=350)
                        imgui.text_colored(display_key, 0.8, 0.8, 0.2, 1.0)

                    imgui.spacing()
                    imgui.spacing()

                imgui.end_child()

            imgui.separator()

            # Bottom button
            if imgui.button("Close##CloseCheatSheet", width=100):
                self.show_cheat_sheet = False

        imgui.end()
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
