"""
Chapter Type Manager UI - Manage built-in and custom chapter types

This window allows users to:
- View all built-in chapter types (read-only)
- Create new custom chapter types
- Edit existing custom types (name, color, category)
- Delete custom types
- See usage statistics
- Import/export custom type libraries

Author: k00gar
Version: 1.0.0
"""

import imgui
import logging
from typing import Optional, Tuple, Dict, Any


class ChapterTypeManagerUI:
    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger(__name__)

        # UI state
        self.selected_type = None  # Currently selected type short_name
        self.edit_mode = False  # True when editing a type

        # Edit/Create form data
        self.form_short_name = ""
        self.form_long_name = ""
        self.form_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        self.form_category = "Custom"
        self.form_error = ""

        # Filter/view options
        self.show_builtin = True
        self.show_custom = True
        self.filter_text = ""

    def render(self):
        """Render the Chapter Type Manager window."""
        app_state = self.app.app_state_ui

        if not hasattr(app_state, "show_chapter_type_manager"):
            app_state.show_chapter_type_manager = False

        if not app_state.show_chapter_type_manager:
            return

        # Get chapter type manager
        from application.classes.chapter_type_manager import get_chapter_type_manager
        type_mgr = get_chapter_type_manager()

        if not type_mgr:
            return

        # Window flags
        window_flags = imgui.WINDOW_NO_COLLAPSE

        # Set window size on first open
        imgui.set_next_window_size(700, 500, imgui.FIRST_USE_EVER)

        opened, _ = imgui.begin("Chapter Type Manager", closable=True, flags=window_flags)

        if not opened:
            app_state.show_chapter_type_manager = False
            imgui.end()
            return

        # Main layout: Left panel (list) | Right panel (details/edit)
        avail_width = imgui.get_content_region_available()[0]
        left_panel_width = avail_width * 0.45

        # === LEFT PANEL: Type List ===
        imgui.begin_child("TypeList", width=left_panel_width, height=0, border=True)

        # Filter options
        imgui.text("Show:")
        imgui.same_line()
        clicked, self.show_builtin = imgui.checkbox("Built-in", self.show_builtin)
        imgui.same_line()
        clicked, self.show_custom = imgui.checkbox("Custom", self.show_custom)

        imgui.separator()

        # Render type list
        self._render_type_list(type_mgr)

        imgui.end_child()

        imgui.same_line()

        # === RIGHT PANEL: Details/Edit ===
        imgui.begin_child("TypeDetails", width=0, height=0, border=True)

        if self.edit_mode:
            self._render_edit_form(type_mgr)
        elif self.selected_type:
            self._render_type_details(type_mgr)
        else:
            self._render_welcome_message()

        imgui.end_child()

        imgui.end()

    def _render_type_list(self, type_mgr):
        """Render the list of chapter types."""
        all_types = type_mgr.get_all_chapter_types()

        # Group by built-in vs custom
        builtin_types = []
        custom_types = []

        for short_name, info in all_types.items():
            is_builtin = type_mgr.is_builtin_type(short_name)
            if is_builtin and self.show_builtin:
                builtin_types.append((short_name, info))
            elif not is_builtin and self.show_custom:
                custom_types.append((short_name, info))

        # Render built-in types
        if builtin_types:
            imgui.text_colored("Built-in Types", 0.7, 0.7, 0.7, 1.0)
            imgui.separator()

            for short_name, info in sorted(builtin_types, key=lambda x: x[1].get("long_name", x[0])):
                self._render_type_item(short_name, info, is_builtin=True)

            imgui.spacing()

        # Render custom types
        if custom_types:
            imgui.text_colored("Custom Types", 0.4, 0.8, 1.0, 1.0)
            imgui.separator()

            for short_name, info in sorted(custom_types, key=lambda x: x[1].get("long_name", x[0])):
                self._render_type_item(short_name, info, is_builtin=False)

            imgui.spacing()

        # Create new type button
        imgui.separator()
        if imgui.button("Create New Type", width=-1):
            self._start_create_new_type()

    def _render_type_item(self, short_name: str, info: Dict[str, Any], is_builtin: bool):
        """Render a single type item in the list."""
        long_name = info.get("long_name", short_name)
        color = info.get("color", [1.0, 1.0, 1.0, 1.0])
        usage_count = info.get("usage_count", 0)

        is_selected = self.selected_type == short_name

        # Color indicator
        imgui.color_button(f"##color_{short_name}", *color, width=20, height=20)
        imgui.same_line()

        # Selectable item
        clicked, _ = imgui.selectable(
            f"{long_name}##item_{short_name}",
            selected=is_selected
        )

        if clicked:
            self.selected_type = short_name
            self.edit_mode = False

        # Tooltip with details
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text(f"Short Name: {short_name}")
            imgui.text(f"Category: {info.get('category', 'N/A')}")
            imgui.text(f"Usage Count: {usage_count}")
            if is_builtin:
                imgui.text_colored("Built-in (read-only)", 0.7, 0.7, 0.7, 1.0)
            imgui.end_tooltip()

    def _render_type_details(self, type_mgr):
        """Render details for the selected type."""
        all_types = type_mgr.get_all_chapter_types()

        if self.selected_type not in all_types:
            self.selected_type = None
            return

        info = all_types[self.selected_type]
        is_builtin = type_mgr.is_builtin_type(self.selected_type)

        imgui.text_colored("Type Details", 0.4, 0.8, 1.0, 1.0)
        imgui.separator()
        imgui.spacing()

        # Display info
        imgui.text(f"Long Name: {info.get('long_name', 'N/A')}")
        imgui.text(f"Short Name: {self.selected_type}")
        imgui.text(f"Category: {info.get('category', 'N/A')}")
        imgui.text(f"Usage Count: {info.get('usage_count', 0)}")

        # Color preview
        imgui.text("Color:")
        imgui.same_line()
        color = info.get("color", [1.0, 1.0, 1.0, 1.0])
        imgui.color_button(f"##preview_color", *color, width=100, height=30)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Action buttons
        if is_builtin:
            imgui.text_colored("Built-in types cannot be edited or deleted", 0.7, 0.7, 0.7, 1.0)
        else:
            if imgui.button("Edit Type", width=120):
                self._start_edit_type()

            imgui.same_line()

            # Delete button (destructive)
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.2, 0.2, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 1.0, 0.3, 0.3, 1.0)
            if imgui.button("Delete Type", width=120):
                self._delete_type(type_mgr)
            imgui.pop_style_color(2)

    def _render_edit_form(self, type_mgr):
        """Render the edit/create form."""
        is_new = self.selected_type is None

        if is_new:
            imgui.text_colored("Create New Type", 0.4, 0.8, 1.0, 1.0)
        else:
            imgui.text_colored("Edit Type", 0.4, 0.8, 1.0, 1.0)

        imgui.separator()
        imgui.spacing()

        # Short name (read-only if editing)
        if is_new:
            imgui.text("Short Name:")
            imgui.push_item_width(-1)
            _, self.form_short_name = imgui.input_text("##short_name", self.form_short_name, 10)
            imgui.pop_item_width()
            if imgui.is_item_hovered():
                imgui.set_tooltip("2-10 characters, unique identifier")
        else:
            imgui.text(f"Short Name: {self.selected_type} (read-only)")

        imgui.spacing()

        # Long name
        imgui.text("Display Name:")
        imgui.push_item_width(-1)
        _, self.form_long_name = imgui.input_text("##long_name", self.form_long_name, 50)
        imgui.pop_item_width()

        imgui.spacing()

        # Category
        imgui.text("Category:")
        imgui.push_item_width(-1)
        _, self.form_category = imgui.input_text("##category", self.form_category, 30)
        imgui.pop_item_width()
        if imgui.is_item_hovered():
            imgui.set_tooltip("e.g., Custom, Position, Transition")

        imgui.spacing()

        # Color picker
        imgui.text("Color:")
        clicked, self.form_color = imgui.color_edit4("##color_picker", *self.form_color)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Error message
        if self.form_error:
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.3, 0.3, 1.0)
            imgui.text_wrapped(self.form_error)
            imgui.pop_style_color()
            imgui.spacing()

        # Action buttons
        if imgui.button("Save", width=120):
            self._save_type(type_mgr, is_new)

        imgui.same_line()

        if imgui.button("Cancel", width=120):
            self.edit_mode = False
            self.form_error = ""

    def _render_welcome_message(self):
        """Render welcome message when nothing is selected."""
        imgui.text_wrapped("Select a chapter type from the list to view details.")
        imgui.spacing()
        imgui.text_wrapped("Built-in types are read-only. Custom types can be created, edited, and deleted.")

    def _start_create_new_type(self):
        """Start creating a new custom type."""
        self.selected_type = None
        self.edit_mode = True
        self.form_short_name = ""
        self.form_long_name = ""
        self.form_color = [0.5, 0.5, 1.0, 1.0]  # Default blue
        self.form_category = "Custom"
        self.form_error = ""

    def _start_edit_type(self):
        """Start editing the selected type."""
        from application.classes.chapter_type_manager import get_chapter_type_manager
        type_mgr = get_chapter_type_manager()

        if not type_mgr or not self.selected_type:
            return

        all_types = type_mgr.get_all_chapter_types()
        if self.selected_type not in all_types:
            return

        info = all_types[self.selected_type]

        self.edit_mode = True
        self.form_long_name = info.get("long_name", "")
        self.form_category = info.get("category", "Custom")
        color = info.get("color", [1.0, 1.0, 1.0, 1.0])
        self.form_color = list(color)  # Copy
        self.form_error = ""

    def _save_type(self, type_mgr, is_new: bool):
        """Save the type (create or update)."""
        self.form_error = ""

        # Validate
        if is_new:
            if not self.form_short_name or len(self.form_short_name) < 2:
                self.form_error = "Short name must be at least 2 characters"
                return

        if not self.form_long_name:
            self.form_error = "Display name is required"
            return

        try:
            if is_new:
                # Create new type
                success = type_mgr.add_custom_type(
                    short_name=self.form_short_name,
                    long_name=self.form_long_name,
                    color=tuple(self.form_color),
                    category=self.form_category
                )

                if success:
                    self.logger.info(f"Created custom chapter type: {self.form_short_name}")
                    self.selected_type = self.form_short_name
                    self.edit_mode = False
                    self.app.project_manager.project_dirty = True
                else:
                    self.form_error = f"Failed to create type (may already exist)"
            else:
                # Update existing type
                success = type_mgr.edit_custom_type(
                    short_name=self.selected_type,
                    new_data={
                        "long_name": self.form_long_name,
                        "color": tuple(self.form_color),
                        "category": self.form_category
                    }
                )

                if success:
                    self.logger.info(f"Updated custom chapter type: {self.selected_type}")
                    self.edit_mode = False
                    self.app.project_manager.project_dirty = True
                else:
                    self.form_error = "Failed to update type"

        except Exception as e:
            self.logger.error(f"Error saving chapter type: {e}", exc_info=True)
            self.form_error = f"Error: {str(e)}"

    def _delete_type(self, type_mgr):
        """Delete the selected custom type."""
        if not self.selected_type:
            return

        # Confirm deletion
        if not hasattr(self, 'confirm_delete') or not self.confirm_delete:
            self.confirm_delete = True
            return

        success = type_mgr.delete_custom_type(self.selected_type)

        if success:
            self.logger.info(f"Deleted custom chapter type: {self.selected_type}")
            self.selected_type = None
            self.confirm_delete = False
            self.app.project_manager.project_dirty = True
        else:
            self.logger.error(f"Failed to delete type: {self.selected_type}")
