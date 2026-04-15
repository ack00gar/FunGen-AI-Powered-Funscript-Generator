import imgui
from application.utils.imgui_helpers import center_next_window_pivot
import os
import logging
from application.utils import GeneratedFileManager, destructive_button_style
from application.utils.imgui_helpers import DisabledScope, begin_modal_centered

# TODO: Comprehensive delete options and settings. by date/days old, extension, size, etc.

class GeneratedFileManagerWindow:
    def __init__(self, app_instance):
        self.app = app_instance
        self.output_folder = self.app.app_settings.get("output_folder_path", "output")
        self.sort_by = 'name'
        self.file_manager = GeneratedFileManager(self.output_folder, logger=self.app.logger, delete_funscript_files=False)
        self.file_manager._scan_files()
        self.expanded_folders = set()
        self.expand_all = False
        self.force_expand_collapse = False


    def render(self):
        """Orchestrates the rendering of the Generated File Manager window UI."""
        app_state = self.app.app_state_ui

        # Set window size constraints for better auto-resize behavior
        imgui.set_next_window_size_constraints((600, 300), (1200, 800))

        center_next_window_pivot()
        is_open, new_visibility = imgui.begin("Generated File Manager", closable=True, flags=imgui.WINDOW_NO_COLLAPSE)
        if new_visibility != app_state.show_generated_file_manager:
            app_state.show_generated_file_manager = new_visibility
        if is_open:
            # Render header and controls (sticky)
            self._render_header_controls()
            # Begin a scrollable child region for the file/folder list
            # Calculate dynamic height based on number of items
            folder_items = self.file_manager.get_sorted_file_tree(self.sort_by)
            # Estimate rows: each folder + its files
            estimated_rows = 0
            for folder_name, files in folder_items:
                estimated_rows += 1  # Folder row
                if folder_name in self.expanded_folders or self.expand_all:
                    estimated_rows += len(files)  # File rows if expanded

            # Calculate height: min 200px, max 600px, or based on content
            row_height = 25  # Approximate height per row
            content_height = min(max(estimated_rows * row_height, 200), 600)

            imgui.begin_child("FileListRegion", width=0, height=content_height, border=True)
            self._render_proxy_section()
            self._render_file_tree()
            imgui.end_child()
            # Render popups/modals
            self._render_delete_all_popup()
        imgui.end()

    def _render_proxy_section(self):
        """Render a collapsible 'Proxies' section listing all iframe proxies
        tracked in the global registry (they live next to source videos, not
        in the output folder, so they need a separate scan)."""
        try:
            from video.proxy_builder import registry_list, delete_proxy
        except Exception:
            return
        entries = registry_list()
        total_gb = sum(e.get("proxy_size_bytes", 0) for e in entries) / (1024 ** 3)
        label = f"Proxies ({len(entries)}, {total_gb:.1f} GB)##proxies_section"
        if not imgui.tree_node(label, imgui.TREE_NODE_DEFAULT_OPEN if entries else 0):
            return
        try:
            if not entries:
                imgui.text_disabled("  No proxies built yet.")
                return
            for e in entries:
                src = e.get("source_path", "")
                pp = e.get("proxy_path", "")
                size_gb = e.get("proxy_size_bytes", 0) / (1024 ** 3)
                src_missing = not e.get("source_exists", True)
                tag = " (source missing)" if src_missing else ""
                imgui.text(f"  {os.path.basename(pp)}  - {size_gb:.2f} GB{tag}")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        f"Proxy: {pp}\n"
                        f"Source: {src}"
                    )
                imgui.same_line()
                if imgui.small_button(f"Delete##proxy_{pp}"):
                    delete_proxy(pp)
        finally:
            imgui.tree_pop()

    def _render_header_controls(self):
        """Renders the header and control buttons for the file manager window."""
        imgui.text(f"Managing files in: {os.path.abspath(self.output_folder)}")
        imgui.text(f"Total Disk Space Used: {self.file_manager.total_size:.2f} MB")
        imgui.separator()
        if imgui.button("Refresh File List"): self.file_manager._scan_files()
        imgui.same_line()
        imgui.text("Sort by:")
        imgui.same_line()
        if imgui.radio_button("Name", self.sort_by == 'name'): self.sort_by = 'name'
        imgui.same_line()
        if imgui.radio_button("Size", self.sort_by == 'size'): self.sort_by = 'size'
        imgui.same_line()
        imgui.dummy(4, 1)
        imgui.same_line()
        expand_text = "Collapse All" if self.expand_all else " Expand All "
        if imgui.button(expand_text):
            self.expand_all = not self.expand_all
            if self.expand_all:
                self.expanded_folders = set(self.file_manager.file_tree.keys())
            else:
                self.expanded_folders = set()
            self.force_expand_collapse = True
        # Place the checkbox and button on the same line, aligned right
        button_text = "[DANGER] Delete All Generated Files"
        button_width = imgui.calc_text_size(button_text)[0] + imgui.get_style().frame_padding[0] * 2
        window_width = imgui.get_window_width()
        checkbox_label = "Delete .funscript files"
        checkbox_width = imgui.calc_text_size(checkbox_label)[0] + imgui.get_style().frame_padding[0] * 2 + 30
        imgui.same_line(max(0, window_width - button_width - checkbox_width - 15))
        changed, new_val = imgui.checkbox(checkbox_label, self.file_manager.delete_funscript_files)
        if changed:
            self.file_manager.delete_funscript_files = new_val
        imgui.same_line(max(0, window_width - button_width - 15))
        if imgui.button(button_text): imgui.open_popup("Delete All Files?###ConfirmDeleteAll")
        imgui.separator()

    def _delete_path_with_status(self, path, delete_func, type_name):
        """
        Delete a file or folder and set an appropriate status message.
        Args:
            path (str): Path to file or folder.
            delete_func (callable): Function to call for deletion.
            type_name (str): 'folder' or 'file' for messages.
        """
        type_name_cap = type_name.capitalize()
        result = delete_func(path)
        if result:
            if not os.path.exists(path):
                self.app.set_status_message(f"SUCCESS: Deleted {type_name} {os.path.basename(path)}", level=logging.INFO)
            elif self.file_manager.delete_funscript_files:
                self.app.set_status_message(f"ERROR: {type_name_cap} still exists.", level=logging.ERROR)
            else:
                self.app.set_status_message(f"INFO: {type_name_cap} not fully deleted (may contain .funscript files)", level=logging.INFO)
        else:
            self.app.set_status_message(f"INFO: {type_name_cap} not found or already deleted.", level=logging.INFO)
        self.file_manager._scan_files()

    def _render_file_tree(self):
        """Renders the file tree structure for generated files and folders."""
        folder_items = self.file_manager.get_sorted_file_tree(self.sort_by)
        if not folder_items:
            imgui.text("No generated files found in the output directory.")
        else:
            for video_dir, dir_data in folder_items:
                self._render_folder_node(video_dir, dir_data)
            # Reset force_expand_collapse after all folders are rendered
            self.force_expand_collapse = False

    def _render_folder_node(self, video_dir, dir_data):
        """Renders a single folder node and its files in the file tree."""
        imgui.push_id(video_dir)
        if self.force_expand_collapse:
            imgui.set_next_item_open(self.expand_all, imgui.ALWAYS)
        is_node_open = False
        if self.expand_all or video_dir in self.expanded_folders:
            is_node_open = imgui.tree_node(video_dir, imgui.TREE_NODE_DEFAULT_OPEN)
        else:
            is_node_open = imgui.tree_node(video_dir)
        if is_node_open:
            self.expanded_folders.add(video_dir)
        else:
            self.expanded_folders.discard(video_dir)
        imgui.same_line(imgui.get_window_width() - 200)
        imgui.text_disabled(f"({dir_data['total_size_mb']:.2f} MB)")
        imgui.same_line(imgui.get_window_width() - 80)
        # Folder deletion
        if imgui.button("Delete"):
            self._delete_path_with_status(dir_data['path'], self.file_manager.delete_folder, "folder")
        if is_node_open:
            imgui.indent()
            for file_info in dir_data['files']:
                self._render_file_node(file_info)
            imgui.tree_pop()
            imgui.unindent()
        imgui.separator()
        imgui.pop_id()

    def _render_file_node(self, file_info):
        """Renders a single file node (row) in the file tree."""
        imgui.bullet()
        imgui.same_line()
        imgui.text(f"{file_info['name']}")
        imgui.same_line(imgui.get_window_width() - 200)
        imgui.text_disabled(f"{file_info['size_mb']:.3f} MB")
        imgui.same_line(imgui.get_window_width() - 80)
        imgui.push_id(file_info['path'])
        is_funscript = file_info['name'].endswith('.funscript')
        delete_enabled = self.file_manager.delete_funscript_files or not is_funscript
        with DisabledScope(not delete_enabled):
            with destructive_button_style():
                if imgui.button("Delete"):
                    self._delete_path_with_status(file_info['path'], self.file_manager.delete_file, "file")
        imgui.pop_id()

    def _render_delete_all_popup(self):
        """Renders the modal popup for confirming deletion of all generated files."""
        if not imgui.is_popup_open("Delete All Files?###ConfirmDeleteAll"):
            return
        imgui.set_next_window_size(480, 0)
        center_next_window_pivot()
        opened, visible = imgui.begin_popup_modal("Delete All Files?###ConfirmDeleteAll")
        if opened:
            imgui.text_wrapped("WARNING: This will delete ALL subfolders and files in the output directory!")
            imgui.text_wrapped(f"Directory: {os.path.abspath(self.output_folder)}")
            imgui.text("Contents will be moved to the recycle bin.")
            imgui.separator()
            with destructive_button_style():
                if imgui.button("YES, DELETE EVERYTHING", width=200):
                    include_funscripts = self.file_manager.delete_funscript_files
                    if self.file_manager.delete_all(include_funscript_files=include_funscripts):
                        self.app.set_status_message("SUCCESS: All generated files have been deleted.", level=logging.INFO)
                    else:
                        self.app.set_status_message(f"ERROR deleting all files in {self.output_folder}", level=logging.ERROR)
                    self.file_manager._scan_files()
                    imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", width=120):
                imgui.close_current_popup()
            imgui.end_popup()
