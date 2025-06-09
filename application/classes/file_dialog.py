import os
from typing import Callable, Optional
import imgui
import platform
import logging
import string


def get_common_dirs():
    """Returns a dictionary of common directory paths."""
    home = os.path.expanduser("~")
    dirs = {
        "Home": home,
        "Desktop": os.path.join(home, "Desktop"),
        "Documents": os.path.join(home, "Documents"),
        "Downloads": os.path.join(home, "Downloads"),
    }
    if platform.system() == "Windows":
        for drive in string.ascii_uppercase:
            path = f"{drive}:\\"
            if os.path.isdir(path):
                dirs[f"{drive}: Drive"] = path
    else:
        dirs["/"] = "/"
    return dirs


class ImGuiFileDialog:
    """A file dialog implementation for ImGui."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.open: bool = False
        self.callback: Optional[Callable[[str], None]] = None
        self.title: str = ""
        self.is_save_dialog: bool = False
        self.current_dir: str = os.getcwd()
        self.selected_file: str = ""
        self.entered_filename: str = ""
        self.extension_filter: str = ""
        self.extension_groups: list[tuple[str, list[str]]] = []
        self.active_extension_index: int = 0
        self.show_overwrite_confirm: bool = False
        self.overwrite_file_path: str = ""
        self.common_dirs = get_common_dirs()
        self.logger = logger or logging.getLogger(__name__)

    def show(
            self,
            title: str,
            is_save: bool = False,
            callback: Optional[Callable[[str], None]] = None,
            extension_filter: str = "",
            initial_path: Optional[str] = None,
            initial_filename: str = ""
    ) -> None:
        """Opens the file dialog window."""
        self.open = True
        self.title = title
        self.is_save_dialog = is_save
        self.callback = callback
        self.extension_filter = extension_filter
        self.extension_groups = self._parse_extension_filter(extension_filter)
        self.active_extension_index = 0
        self.selected_file = initial_filename or ""
        self.entered_filename = initial_filename or ""

        if initial_path and os.path.isdir(initial_path):
            self.current_dir = initial_path
        elif not os.path.isdir(self.current_dir):
            self.current_dir = os.path.expanduser("~")

    def draw(self) -> None:
        """Renders the file dialog window."""
        if not self.open:
            return

        imgui.set_next_window_size(750, 450, imgui.ONCE)
        expanded, self.open = imgui.begin(self.title, True, flags=imgui.WINDOW_NO_COLLAPSE)
        if not self.open:
            imgui.end()
            return

        if expanded:
            self._draw_common_dirs_sidebar()
            imgui.same_line()
            imgui.begin_group()
            self._draw_directory_navigation()
            self._draw_filter_selector()
            imgui.begin_child("Files", width=0, height=-80, border=True)
            self._draw_file_list()
            imgui.end_child()
            self._draw_bottom_bar()
            imgui.end_group()

        self._draw_overwrite_confirm()
        imgui.end()

    def _draw_common_dirs_sidebar(self):
        imgui.begin_child("sidebar", width=150, border=False)
        imgui.text("Quick Access")
        imgui.separator()
        for name, path in self.common_dirs.items():
            if os.path.isdir(path):
                if imgui.button(name, width=-1):
                    self.current_dir = path
                    self.selected_file = ""
        imgui.end_child()

    def _draw_directory_navigation(self):
        imgui.text_wrapped(f"Current: {self.current_dir}")
        imgui.separator()

    def _navigate_up(self):
        parent_dir = os.path.dirname(self.current_dir)
        if os.path.isdir(parent_dir) and parent_dir != self.current_dir:
            self.current_dir = parent_dir
            self.selected_file = ""

    def _draw_filter_selector(self):
        imgui.align_text_to_frame_padding()
        imgui.text("File Type:")
        imgui.same_line()

        if self.extension_groups:
            filter_names = [name for name, _ in self.extension_groups]
            imgui.set_next_item_width(imgui.get_content_region_available_width() - 60)
            changed, self.active_extension_index = imgui.combo(
                "##filter_combo", self.active_extension_index, filter_names
            )
        else:
            imgui.set_next_item_width(imgui.get_content_region_available_width() - 60)
            imgui.text("All Files")

        imgui.same_line()
        if imgui.button("Up", width=50):
            self._navigate_up()
        imgui.separator()

    def _draw_file_list(self):
        try:
            if not os.path.isdir(self.current_dir):
                imgui.text("Directory not found or is inaccessible.")
                return

            items = sorted(os.listdir(self.current_dir), key=str.lower)
            directories, files = [], []
            for item in items:
                full_path = os.path.join(self.current_dir, item)
                if os.path.isdir(full_path):
                    directories.append(item)
                elif os.path.isfile(full_path):
                    files.append(item)

            self._draw_directories(directories)
            self._draw_files(files)

        except PermissionError:
            imgui.text("Permission denied to access this directory.")
        except Exception as e:
            self.logger.error(f"Error reading directory '{self.current_dir}': {e}", exc_info=True)
            imgui.text(f"Error reading directory: {e}")

    def _draw_directories(self, directories: list[str]):
        for d in directories:
            imgui.selectable(f"[DIR] {d}", False)
            if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                self.current_dir = os.path.join(self.current_dir, d)
                self.selected_file = ""
                self.entered_filename = ""

    def _draw_files(self, files: list[str]):
        filtered_files = []
        if self.extension_groups and self.active_extension_index < len(self.extension_groups):
            _, active_exts = self.extension_groups[self.active_extension_index]
            if "*" in active_exts:
                filtered_files = files
            else:
                for f in files:
                    for ext in active_exts:
                        if f.lower().endswith(f".{ext.lower()}"):
                            filtered_files.append(f)
                            break
        else:
            filtered_files = files

        for f in filtered_files:
            is_selected = (self.selected_file == f)
            if imgui.selectable(f"[FILE] {f}", is_selected):
                # For any click, update the selection for highlighting.
                self.selected_file = f
                # **REMOVED FAULTY LOGIC**: We no longer overwrite the input box on click.

            if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                # Double-clicking a file in an "Open" dialog confirms it.
                if not self.is_save_dialog:
                    self._confirm_selection()

    def _draw_bottom_bar(self):
        imgui.separator()
        confirm = False
        if self.is_save_dialog:
            imgui.text("File name:")
            imgui.same_line()
            enter_pressed, self.entered_filename = imgui.input_text(
                "##filename_input",
                self.entered_filename,
                256,
                flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
            )
            if enter_pressed:
                confirm = True
        else:
            imgui.text("File name:")
            imgui.same_line()
            imgui.input_text("##filename_display", self.selected_file, 256, flags=imgui.INPUT_TEXT_READ_ONLY)

        button_width = 80
        cursor_x = imgui.get_cursor_pos().x + imgui.get_content_region_available().x - (
                    button_width * 2 + imgui.get_style().item_spacing.x)
        imgui.set_cursor_pos_x(cursor_x)

        action_button_text = "Save" if self.is_save_dialog else "Open"
        filename_for_action = self.entered_filename if self.is_save_dialog else self.selected_file
        is_enabled = bool(filename_for_action)

        if not is_enabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)

        if imgui.button(action_button_text, width=button_width):
            confirm = True

        if not is_enabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

        imgui.same_line()
        if imgui.button("Cancel", width=button_width):
            self.open = False

        if confirm and is_enabled:
            self._confirm_selection()

    def _confirm_selection(self):
        final_filename = self.entered_filename if self.is_save_dialog else self.selected_file
        if not final_filename: return
        file_path = os.path.join(self.current_dir, final_filename)
        if self.is_save_dialog and os.path.exists(file_path) and not os.path.isdir(file_path):
            self.show_overwrite_confirm = True
            self.overwrite_file_path = file_path
            return
        self._execute_callback(file_path)

    def _execute_callback(self, path: str):
        if self.callback:
            try:
                self.callback(path)
            except Exception as e:
                self.logger.error(f"Error in file dialog callback: {e}", exc_info=True)
        self.open = False

    def _draw_overwrite_confirm(self):
        if self.show_overwrite_confirm:
            imgui.open_popup("Confirm Overwrite")
        if imgui.begin_popup_modal("Confirm Overwrite", True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text(f"'{os.path.basename(self.overwrite_file_path)}' already exists.")
            imgui.text("Do you want to replace it?")
            imgui.separator()
            if imgui.button("Overwrite", width=120):
                self._execute_callback(self.overwrite_file_path)
                self.show_overwrite_confirm = False
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", width=120):
                self.show_overwrite_confirm = False
                imgui.close_current_popup()
            imgui.end_popup()

    def _parse_extension_filter(self, filter_string: str) -> list[tuple[str, list[str]]]:
        if not filter_string:
            return [("All Files (*)", ["*"])]
        groups = filter_string.split("|")
        result = []
        for group in groups:
            parts = group.split(",", 1)
            if len(parts) == 2:
                label, exts_str = parts
                exts = [e.strip().lstrip("*.") for e in exts_str.split(";") if e.strip()]
                if not exts or "*" in exts or "*.*" in exts:
                    result.append((label, ["*"]))
                else:
                    result.append((label, exts))
        if not any("*" in exts for _, exts in result):
            result.append(("All Files (*)", ["*"]))
        return result
