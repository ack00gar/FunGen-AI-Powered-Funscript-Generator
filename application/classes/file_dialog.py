import os
import json
import orjson
from typing import Callable, Optional
import imgui
import platform
import logging
import string

from config.constants import FUNSCRIPT_METADATA_VERSION # Added


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

def get_directory_size(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total


class ImGuiFileDialog:
    def __init__(self, app_logic_instance) -> None:
        self.app = app_logic_instance
        self.logger = self.app.logger
        self.open: bool = False
        self.path: str = ""
        self.selected_file: str = ""
        self.current_dir: str = os.getcwd()
        self.callback: Optional[Callable[[str], None]] = None
        self.title: str = ""
        self.is_save_dialog: bool = False
        self.is_folder_dialog: bool = False
        self.extension_filter: str = ""
        self.scroll_to_selected: bool = False
        self.common_dirs = get_common_dirs()
        self.active_extension_index = 0
        self.extension_groups: list[tuple[str, list[str]]] = []
        self.show_overwrite_confirm: bool = False
        self.overwrite_file_path: str = ""
        self.video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm'] # Added

        # Cached directory listing and metadata to avoid per-frame filesystem hits
        self._current_dir_cached: str = ""
        self._cached_directories: list[str] = []
        self._cached_files: list[str] = []
        self._cached_special_packages: list[str] = []
        self._needs_rescan: bool = True
        self._file_sizes_cache: dict = {}
        self._funscript_status_cache: dict = {}

    @staticmethod
    def get_funscript_status(video_path: str, logger: logging.Logger) -> Optional[str]:
        """Checks for an associated funscript and determines its origin."""
        funscript_path = os.path.splitext(video_path)[0] + ".funscript"
        if not os.path.exists(funscript_path):
            return None

        try:
            # Reverted to original, working method: text mode with standard json library
            with open(funscript_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            metadata = data.get('metadata', {})
            author = data.get('author', '')
            version = metadata.get('version', '') if isinstance(metadata, dict) else ''

            if author.startswith("FunGen") and version == FUNSCRIPT_METADATA_VERSION:
                return 'fungen'
            else:
                return 'other'
        except Exception as e:
            logger.warning(f"Could not parse funscript '{os.path.basename(funscript_path)}': {e}")
            return 'other'

    def _get_funscript_status(self, video_path: str) -> Optional[str]:
        """Instance method wrapper for the static funscript status checker."""
        return ImGuiFileDialog.get_funscript_status(video_path, self.logger)

    def show(
            self,
            title: str,
            is_save: bool = False,
            callback: Optional[Callable[[str], None]] = None,
            extension_filter: str = "",
            initial_path: Optional[str] = None,
            initial_filename: str = "",
            is_folder_dialog: bool = False
    ) -> None:
        self.open = True
        self.title = title
        self.is_save_dialog = is_save
        self.is_folder_dialog = is_folder_dialog
        self.callback = callback
        self.extension_filter = extension_filter
        self.extension_groups = self._parse_extension_filter(extension_filter)
        self.active_extension_index = 0
        self.selected_file = initial_filename or ""

        if initial_path and os.path.isdir(initial_path):
            self.current_dir = initial_path

        # Invalidate caches on open
        self._invalidate_listing_cache()

    def _invalidate_listing_cache(self) -> None:
        self._current_dir_cached = ""
        self._cached_directories = []
        self._cached_files = []
        self._cached_special_packages = []
        self._file_sizes_cache.clear()
        self._funscript_status_cache.clear()
        self._needs_rescan = True

    def _build_listing_if_needed(self) -> None:
        # Only (re)scan the filesystem when the directory actually changes or a refresh is requested
        if not self._needs_rescan and self._current_dir_cached == self.current_dir:
            return
        try:
            items = os.listdir(self.current_dir) if os.path.exists(self.current_dir) else []
        except Exception:
            items = []

        special_packages = [d for d in items if os.path.isdir(os.path.join(self.current_dir, d)) and d.lower().endswith('.mlpackage')]
        directories = [d for d in items if os.path.isdir(os.path.join(self.current_dir, d)) and not d.lower().endswith('.mlpackage')]
        files = [f for f in items if os.path.isfile(os.path.join(self.current_dir, f))]

        self._cached_directories = directories
        self._cached_files = files
        self._cached_special_packages = special_packages
        self._current_dir_cached = self.current_dir
        self._needs_rescan = False

    def _draw_common_dirs_sidebar(self):
        imgui.begin_child("sidebar", width=150, height=0, border=False)

        # Add a header for clarity
        imgui.text("Quick Access")
        imgui.spacing()

        # Show common directories as buttons
        for name, path in self.common_dirs.items():
            if imgui.button(name, width=130):
                if os.path.exists(path):
                    self.current_dir = path
                    self.scroll_to_selected = True
                    self._invalidate_listing_cache()
            imgui.spacing()

        # 1. Output Directory
        output_dir = self.app.app_settings.get("output_folder_path", "output")
        if output_dir and os.path.isdir(output_dir):
            abs_output_dir = os.path.abspath(output_dir)
            if imgui.button("Output Folder", width=130):
                self.current_dir = abs_output_dir
                self.scroll_to_selected = True
                self._invalidate_listing_cache()
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Go to the configured output folder:\n{abs_output_dir}")
            imgui.spacing()

        # 2. Current Video Directory
        video_path = self.app.file_manager.video_path
        if video_path and os.path.isfile(video_path):
            video_dir = os.path.dirname(video_path)
            if os.path.isdir(video_dir):
                if imgui.button("Curr. Video Folder", width=130):
                    self.current_dir = video_dir
                    self.scroll_to_selected = True
                    self._invalidate_listing_cache()
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Go to the current video's folder:\n{video_dir}")
                imgui.spacing()
        imgui.end_child()

    def _draw_filter_selector(self):
        # For file dialogs: show extension dropdown and up button
        # For folder dialogs: show only up button
        if not self.is_folder_dialog:
            filter_names = [name for name, _ in self.extension_groups]
            clicked, new_index = imgui.combo("File Type", self.active_extension_index, filter_names)
            if clicked and new_index != self.active_extension_index:
                self.active_extension_index = new_index
                # No rescan needed; filtering is applied on cached files
            imgui.same_line()
            if imgui.button("^ Up", width=50):
                self._navigate_up()
        else:
            if imgui.button("^ Up", width=50):
                self._navigate_up()

        imgui.same_line()
        if imgui.button("Refresh", width=70):
            self._invalidate_listing_cache()

    def _parse_extension_filter(self, filter_string: str) -> list[tuple[str, list[str]]]:
        if not filter_string or self.is_folder_dialog:
            return [("All Files", [""])]

        groups = filter_string.split("|")
        result = []
        all_exts = set()
        for group in groups:
            if "," in group:
                label, ext = group.split(",", 1)
                ext_list = [e.strip().lstrip("*.") for e in ext.split(";")]
                result.append((label.strip(), ext_list))
                all_exts.update(ext_list)

        # Add individual extension filters
        indiv_exts = sorted([e for e in all_exts if e and e.lower() != 'all files'])
        indiv_ext_groups = [(f"*.{ext}", [ext]) for ext in indiv_exts]

        # Compose final list: all extensions (first), individual extensions, all files (last)
        final = []
        if result:
            # Use the first group as 'all extensions' ('AI Models')
            final.append(result[0])
        final.extend(indiv_ext_groups)
        # Add 'All Files' at the end
        final.append(("All Files", [""]))
        return final

    def draw(self) -> None:
        if not self.open:
            return

        imgui.set_next_window_size(750, 400)
        is_open_current_frame, self.open = imgui.begin(self.title, self.open)

        if not self.open:
            imgui.end()
            return

        if is_open_current_frame:
            try:
                imgui.columns(2, 'main_columns', border=False)
                imgui.set_column_width(0, 150)
                self._draw_common_dirs_sidebar()
                imgui.next_column()
                self._draw_directory_navigation()
                self._draw_filter_selector()
                if imgui.begin_child("Files", width=0, height=-75, border=True):
                    self._draw_file_list()
                    imgui.end_child()
                should_close = self._draw_bottom_bar()
                if should_close:
                    self.open = False
                self._draw_overwrite_confirm()
            finally:
                imgui.columns(1)
                imgui.end()

    def _draw_directory_navigation(self) -> None:
        current_dir_text = f"{self.current_dir}\n"
        imgui.text(current_dir_text)

    def _navigate_up(self) -> None:
        try:
            parent = os.path.dirname(self.current_dir)
            if os.path.exists(parent) and os.path.isdir(parent):
                self.current_dir = parent
                self.scroll_to_selected = True
                self._invalidate_listing_cache()
        except Exception as e:
            imgui.text(f"Error navigating up: {str(e)}")

    def _draw_file_list(self) -> None:
        try:
            if not os.path.exists(self.current_dir):
                imgui.text("Directory not found")
            else:
                self._build_listing_if_needed()

                directories = self._cached_directories
                files = self._cached_files
                special_packages = self._cached_special_packages
                selectable_files = files + special_packages

                # Filter files based on selected extension group
                if not self.is_folder_dialog and self.extension_groups and self.active_extension_index < len(self.extension_groups):
                    _, active_exts = self.extension_groups[self.active_extension_index]
                    # Keep .mlpackage files if "mlpackage" is in the active extensions
                    selectable_files = [
                        f for f in selectable_files if
                        any(f.lower().endswith(ext.lower()) for ext in active_exts) or
                        (f.lower().endswith('.mlpackage') and "mlpackage" in active_exts)
                    ]

                self._draw_directories(directories)
                if not self.is_folder_dialog:
                    self._draw_files(selectable_files)
                if self.scroll_to_selected:
                    imgui.set_scroll_here_y()
                    self.scroll_to_selected = False

        except PermissionError:
            imgui.text("Permission denied to access this directory")
        except Exception as e:
            imgui.text(f"Error: {str(e)}")

    def _draw_directories(self, directories: list[str]) -> None:
        for i, d in enumerate(sorted(directories)):
            # Use platform-neutral directory label
            label = f"[DIR] {d}"
            imgui.push_id(f"dir_{i}")
            is_selected = self.selected_file == d and self.is_folder_dialog
            if imgui.selectable(label, is_selected, flags=imgui.SELECTABLE_DONT_CLOSE_POPUPS | imgui.SELECTABLE_ALLOW_DOUBLE_CLICK):
                if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
                    self.selected_file = d
                if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                    self.current_dir = os.path.join(self.current_dir, d)
                    self.selected_file = ""
                    self.scroll_to_selected = True
                    self._invalidate_listing_cache()
            imgui.pop_id()

    def _draw_files(self, files: list[str]) -> None:
        for i, f in enumerate(sorted(files)):
            is_selected = self.selected_file == f
            full_path = os.path.join(self.current_dir, f)

            try:
                size_bytes = None
                if f.lower().endswith('.mlpackage'):
                    # Avoid expensive recursive size computation for packages while browsing
                    size_bytes = None
                else:
                    if full_path in self._file_sizes_cache:
                        size_bytes = self._file_sizes_cache[full_path]
                    else:
                        size_bytes = os.path.getsize(full_path)
                        self._file_sizes_cache[full_path] = size_bytes

                if size_bytes is None:
                    size_str = "—"
                elif size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            except Exception:
                size_str = "N/A"

            # Check for funscript status if the file is a video
            funscript_status = None
            if any(f.lower().endswith(ext) for ext in self.video_extensions):
                if full_path in self._funscript_status_cache:
                    funscript_status = self._funscript_status_cache[full_path]
                else:
                    funscript_status = self._get_funscript_status(full_path)
                    self._funscript_status_cache[full_path] = funscript_status

            imgui.push_id(f"file_{i}")

            # Draw components sequentially to position the indicator correctly
            if f.lower().endswith('.mlpackage'):
                imgui.text("[ML]")
            else:
                imgui.text("[FILE]")

            imgui.same_line()
            # Draw the funscript indicator or a placeholder for alignment
            if funscript_status == 'fungen':
                imgui.text_colored("[FG]", 0.2, 0.9, 0.2, 1.0)  # Green
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Funscript created by this version of FunGen")
            elif funscript_status == 'other':
                imgui.text_colored("[FS]", 0.9, 0.9, 0.2, 1.0)  # Yellow
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Funscript exists (unknown or older version)")
            else:
                imgui.text("[N/A]")
                if imgui.is_item_hovered():
                    imgui.set_tooltip("No Funscript exists for this video")

            imgui.same_line()
            # The selectable label now only contains the filename and size
            selectable_label = f"{f:<40} {size_str:>8}"
            if imgui.selectable(selectable_label, is_selected, flags=imgui.SELECTABLE_ALLOW_DOUBLE_CLICK):
                if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
                    self.selected_file = f
                if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                    self.selected_file = f
                    self._handle_file_selection(f)
            imgui.pop_id()

    def _handle_file_selection(self, file: str) -> None:
        # Handle both regular files and .mlpackage special directories
        if self.is_save_dialog:
            file_path = os.path.join(self.current_dir, self.selected_file)
        else:
            file_path = os.path.join(self.current_dir, file)

        if self.is_save_dialog:
            if os.path.exists(file_path):
                self.show_overwrite_confirm = True
                self.overwrite_file_path = file_path
                return

        self.path = file_path
        if self.callback:
            self.callback(self.path)
        self.open = False

    def _confirm_folder_selection(self):
        # Only called for folder selection dialog
        if self.selected_file:
            self.path = os.path.join(self.current_dir, self.selected_file)
        else:
            self.path = self.current_dir
        if self.callback:
            self.callback(self.path)
        self.open = False

    def _draw_overwrite_confirm(self) -> None:
        if self.show_overwrite_confirm:
            imgui.open_popup("Confirm Overwrite")
            if imgui.begin_popup_modal("Confirm Overwrite", flags=imgui.WINDOW_NO_RESIZE)[0]:
                imgui.text(f"The file '{os.path.basename(self.overwrite_file_path)}' already exists.")
                imgui.text("Are you sure you want to overwrite it?")

                imgui.spacing()

                if imgui.button("Overwrite", width=100):
                    self.path = self.overwrite_file_path
                    if self.callback:
                        self.callback(self.path)
                    self.open = False
                    self.show_overwrite_confirm = False

                imgui.same_line()
                if imgui.button("Cancel", width=100):
                    self.show_overwrite_confirm = False

                imgui.end_popup()

    def _draw_bottom_bar(self) -> bool:
        """Draw the bottom bar with buttons. Returns True if window should be closed."""
        should_close = False

        # Add filename input for save dialogs
        if self.is_save_dialog and not self.is_folder_dialog:
            # Calculate width for filename input
            window_width = imgui.get_window_width()
            input_width = int(window_width - 170)

            # Set width for the next item (the input field)
            imgui.set_next_item_width(input_width)

            # Create input field
            changed, value = imgui.input_text("", self.selected_file, 256)
            if changed:
                self.selected_file = value

        # Button row
        cancel_button_width = 60
        action_button_width = 60
        spacing = 10

        # Calculate positioning to right-align the buttons
        window_width = imgui.get_window_width()
        total_buttons_width = cancel_button_width + action_button_width + spacing
        start_x = window_width - total_buttons_width - 10  # 10px padding from right edge

        imgui.set_cursor_pos_x(start_x)

        if imgui.button("Cancel", width=cancel_button_width):
            self.open = False
            should_close = True

        imgui.same_line(0, spacing)
        button_text = "Save" if self.is_save_dialog else "Open"
        if self.is_folder_dialog:
            button_text = "Select"

        # Enable the action button only if a file/folder is selected
        enabled = bool(self.selected_file) or self.is_save_dialog or self.is_folder_dialog
        if not enabled:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
        if imgui.button(button_text, width=action_button_width) and enabled:
            if self.is_save_dialog and not self.is_folder_dialog:
                if self.selected_file:
                    file_path = os.path.join(self.current_dir, self.selected_file)
                    if os.path.exists(file_path):
                        self.show_overwrite_confirm = True
                        self.overwrite_file_path = file_path
                    else:
                        # For save dialog, we use the entered filename
                        self._handle_file_selection(self.selected_file)
            elif self.is_folder_dialog:
                self._confirm_folder_selection()
            else:
                if self.selected_file:
                    # For open dialog, we use the selected file
                    self._handle_file_selection(self.selected_file)
        if not enabled:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()

        return should_close
