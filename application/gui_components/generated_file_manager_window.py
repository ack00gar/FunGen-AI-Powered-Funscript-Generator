import imgui
import os
import logging
from application.utils.generated_file_manager import GeneratedFileManager

# TODO: Comprehansive delete options and settings. by date/days old, extension, size, etc.

class GeneratedFileManagerWindow:
    def __init__(self, app_instance):
        self.app = app_instance
        self.output_folder = self.app.app_settings.get("output_folder_path", "output")
        self.sort_by = 'name'
        self.delete_funscript_files = False  # Default: do NOT delete .funscript files
        self.file_manager = GeneratedFileManager(self.output_folder, logger=self.app.logger)
        self._refresh_file_tree()
        self.expanded_folders = set()
        self.expand_all = False
        self.force_expand_collapse = False

    def _refresh_file_tree(self):
        self.file_manager._scan_files()

    def render(self):
        app_state = self.app.app_state_ui
        # Make this a regular, non-blocking window
        is_visible, is_open = imgui.begin("Generated File Manager", True)
        if not is_open:
            app_state.show_generated_file_manager = False
        if is_visible:
            # --- Header and Controls ---
            imgui.text(f"Managing files in: {os.path.abspath(self.output_folder)}")
            imgui.text(f"Total Disk Space Used: {self.file_manager.total_size:.2f} MB")
            imgui.separator()
            if imgui.button("Refresh File List"): self._refresh_file_tree()
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
            _, self.delete_funscript_files = imgui.checkbox(checkbox_label, self.delete_funscript_files)
            imgui.same_line(max(0, window_width - button_width - 15))
            if imgui.button(button_text): imgui.open_popup("ConfirmDeleteAll")
            imgui.separator()

            # --- File Tree Display ---
            folder_items = self.file_manager.get_sorted_file_tree(self.sort_by)
            if not folder_items:
                imgui.text("No generated files found in the output directory.")
            else:
                for video_dir, dir_data in folder_items:
                    imgui.push_id(video_dir)
                    # Force open/close state if expand/collapse all was just pressed
                    if self.force_expand_collapse:
                        imgui.set_next_item_open(self.expand_all, imgui.ALWAYS)
                    is_node_open = False
                    if self.expand_all or video_dir in self.expanded_folders:
                        is_node_open = imgui.tree_node(video_dir, imgui.TREE_NODE_DEFAULT_OPEN)
                    else:
                        is_node_open = imgui.tree_node(video_dir)
                    # Track expanded/collapsed state
                    if is_node_open:
                        self.expanded_folders.add(video_dir)
                    else:
                        self.expanded_folders.discard(video_dir)
                    imgui.same_line(imgui.get_window_width() - 200)
                    imgui.text_disabled(f"({dir_data['total_size_mb']:.2f} MB)")
                    imgui.same_line(imgui.get_window_width() - 80)

                    # --- FOLDER DELETION ---
                    if imgui.button("Delete"):
                        path = dir_data['path']
                        if self.file_manager.delete_folder(path, include_funscript_files=self.delete_funscript_files):
                            if not os.path.exists(path):
                                self.app.set_status_message(f"SUCCESS: Deleted folder {os.path.basename(path)}", level=logging.INFO)
                            elif self.delete_funscript_files:
                                self.app.set_status_message(f"ERROR: Folder still exists.", level=logging.ERROR)
                            else:
                                self.app.set_status_message(f"INFO: Folder not fully deleted (may contain .funscript files)", level=logging.INFO)
                        else:
                            self.app.set_status_message("INFO: Folder not found or already deleted.", level=logging.INFO)
                        self._refresh_file_tree()  # Refresh UI

                    if is_node_open:
                        imgui.indent()
                        for file_info in dir_data['files']:
                            imgui.bullet()
                            imgui.same_line()
                            imgui.text(f"{file_info['name']}")
                            imgui.same_line(imgui.get_window_width() - 200)
                            imgui.text_disabled(f"{file_info['size_mb']:.3f} MB")
                            imgui.same_line(imgui.get_window_width() - 80)
                            imgui.push_id(file_info['path'])

                            # --- FILE DELETION ---
                            imgui.push_style_color(imgui.COLOR_BUTTON, 0.65, 0.15, 0.15)
                            is_funscript = file_info['name'].endswith('.funscript')
                            delete_enabled = self.delete_funscript_files or not is_funscript
                            if not delete_enabled:
                                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
                            if imgui.button("Delete"):
                                path = file_info['path']
                                if self.file_manager.delete_file(path):
                                    if not os.path.exists(path):
                                        self.app.set_status_message(f"SUCCESS: Deleted {os.path.basename(path)}", level=logging.INFO)
                                    else:
                                        self.app.set_status_message(f"ERROR: File still exists.", level=logging.ERROR)
                                else:
                                    self.app.set_status_message("INFO: File not found or already deleted.", level=logging.INFO)
                                self._refresh_file_tree()  # Refresh UI
                            if not delete_enabled:
                                imgui.pop_style_var()
                                imgui.internal.pop_item_flag()
                            imgui.pop_style_color()
                            imgui.pop_id()
                        imgui.tree_pop()
                        imgui.unindent()
                    imgui.separator()
                    imgui.pop_id()

                # Reset force_expand_collapse after all folders are rendered
                self.force_expand_collapse = False

            # --- "DELETE ALL" POPUP ---
            opened, visible = imgui.begin_popup_modal("ConfirmDeleteAll")
            if opened:
                imgui.text_ansi_colored("WARNING: This will delete ALL subfolders and files in the output directory!", 1.0, 0.2, 0.2)
                imgui.text(f"Directory: {os.path.abspath(self.output_folder)}")
                imgui.text("Contents will be moved to the recycle bin.")
                imgui.separator()
                if imgui.button("YES, DELETE EVERYTHING", width=200):
                    if self.file_manager.delete_all(include_funscript_files=self.delete_funscript_files):
                        self.app.set_status_message("SUCCESS: All generated files have been deleted.", level=logging.INFO)
                    else:
                        self.app.set_status_message(f"ERROR deleting all files in {self.output_folder}", level=logging.ERROR)
                    self._refresh_file_tree()
                    imgui.close_current_popup()
                imgui.same_line()
                if imgui.button("Cancel", width=120):
                    imgui.close_current_popup()
                imgui.end_popup()
        imgui.end()
