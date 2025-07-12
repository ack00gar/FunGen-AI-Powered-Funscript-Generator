import imgui
import os
import shutil
import logging
from send2trash import send2trash


class GeneratedFileManagerWindow:
    def __init__(self, app_instance):
        self.app = app_instance
        self.output_folder = self.app.app_settings.get("output_folder_path", "output")
        self.file_tree = {}
        self.total_size_mb = 0
        self.sort_by = 'name'
        self.delete_funscript_files = False  # Default: do NOT delete .funscript files
        self._scan_files()

    def _scan_files(self):
        self.file_tree = {}
        total_size_bytes = 0
        if not os.path.isdir(self.output_folder):
            return

        for video_dir_name in os.listdir(self.output_folder):
            video_dir_path = os.path.join(self.output_folder, video_dir_name)
            if os.path.isdir(video_dir_path):
                files_in_dir = []
                folder_total_size_bytes = 0
                for filename in os.listdir(video_dir_path):
                    file_path = os.path.join(video_dir_path, filename)
                    if os.path.isfile(file_path):
                        try:
                            size_bytes = os.path.getsize(file_path)
                            total_size_bytes += size_bytes
                            folder_total_size_bytes += size_bytes
                            files_in_dir.append(
                                {"name": filename, "path": file_path, "size_mb": size_bytes / (1024 * 1024)}
                            )
                        except OSError:
                            continue

                if files_in_dir or not os.listdir(video_dir_path):
                    self.file_tree[video_dir_name] = {
                        "path": video_dir_path,
                        "files": sorted(files_in_dir, key=lambda x: x['name']),
                        "total_size_mb": folder_total_size_bytes / (1024 * 1024)
                    }
        self.total_size_mb = total_size_bytes / (1024 * 1024)

    def render(self):
        app_state = self.app.app_state_ui
        # Make this a regular, non-blocking window
        is_visible, is_open = imgui.begin("Generated File Manager", True)

        if not is_open:
            app_state.show_generated_file_manager = False

        if is_visible:
            # --- Header and Controls ---
            imgui.text(f"Managing files in: {os.path.abspath(self.output_folder)}")
            imgui.text(f"Total Disk Space Used: {self.total_size_mb:.2f} MB")
            imgui.separator()
            if imgui.button("Refresh File List"): self._scan_files()
            imgui.same_line()
            imgui.text("Sort by:")
            imgui.same_line()
            if imgui.radio_button("Name", self.sort_by == 'name'): self.sort_by = 'name'
            imgui.same_line()
            if imgui.radio_button("Size", self.sort_by == 'size'): self.sort_by = 'size'

            button_text = "[DANGER] Delete All Generated Files"
            button_width = imgui.calc_text_size(button_text)[0] + imgui.get_style().frame_padding[0] * 2
            # Place the checkbox and button on the same line, aligned right
            window_width = imgui.get_window_width()
            checkbox_label = "Include .funscript files"
            checkbox_width = imgui.calc_text_size(checkbox_label)[0] + imgui.get_style().frame_padding[0] * 2 + 30
            imgui.same_line(max(0, window_width - button_width - checkbox_width - 15))
            _, self.delete_funscript_files = imgui.checkbox(checkbox_label, self.delete_funscript_files)
            imgui.same_line(max(0, window_width - button_width - 15))
            if imgui.button(button_text): imgui.open_popup("ConfirmDeleteAll")
            imgui.separator()

            # --- File Tree Display ---
            if not self.file_tree:
                imgui.text("No generated files found in the output directory.")
            else:
                folder_items = list(self.file_tree.items())
                if self.sort_by == 'size':
                    folder_items.sort(key=lambda item: item[1]['total_size_mb'], reverse=True)
                else:
                    folder_items.sort(key=lambda item: item[0].lower())

                for video_dir, dir_data in folder_items:
                    imgui.push_id(video_dir)
                    is_node_open = imgui.tree_node(video_dir)
                    imgui.same_line(imgui.get_window_width() - 200)
                    imgui.text_disabled(f"({dir_data['total_size_mb']:.2f} MB)")
                    imgui.same_line(imgui.get_window_width() - 80)

                    # --- IMMEDIATE FOLDER DELETION ---
                    if imgui.button("Delete"):
                        path = dir_data['path']
                        if path and os.path.isdir(path):
                            try:
                                if self.delete_funscript_files:
                                    send2trash(path)
                                else:
                                    # Delete all except .funscript files in the folder
                                    for dirpath, dirnames, filenames in os.walk(path):
                                        for filename in filenames:
                                            if filename.endswith('.funscript'):
                                                continue
                                            file_path = os.path.join(dirpath, filename)
                                            try:
                                                send2trash(file_path)
                                            except Exception as e:
                                                self.app.set_status_message(f"ERROR deleting file: {file_path}: {e}", level=logging.ERROR)
                                    # Remove empty folders (except those containing .funscript files)
                                    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
                                        # If only .funscript files remain, skip
                                        if all(f.endswith('.funscript') for f in filenames):
                                            continue
                                        try:
                                            if not os.listdir(dirpath):
                                                send2trash(dirpath)
                                        except Exception:
                                            pass
                                if not os.path.exists(path):
                                    self.app.set_status_message(f"SUCCESS: Deleted folder {os.path.basename(path)}", level=logging.INFO)
                                elif self.delete_funscript_files:
                                    self.app.set_status_message(f"ERROR: Folder still exists.", level=logging.ERROR)
                                else:
                                    self.app.set_status_message(f"INFO: Folder not fully deleted (may contain .funscript files)", level=logging.INFO)
                            except Exception as e:
                                self.app.set_status_message(f"ERROR deleting folder: {e}", level=logging.ERROR)
                        else:
                            self.app.set_status_message("INFO: Folder not found or already deleted.", level=logging.INFO)
                        self._scan_files()  # Refresh UI immediately

                    if is_node_open:
                        for file_info in dir_data['files']:
                            imgui.bullet()
                            imgui.same_line()
                            imgui.text(f"{file_info['name']}")
                            imgui.same_line(imgui.get_window_width() - 200)
                            imgui.text_disabled(f"{file_info['size_mb']:.3f} MB")
                            imgui.same_line(imgui.get_window_width() - 80)
                            imgui.push_id(file_info['path'])

                            # --- IMMEDIATE FILE DELETION ---
                            is_funscript = file_info['name'].endswith('.funscript')
                            delete_enabled = self.delete_funscript_files or not is_funscript
                            if not delete_enabled:
                                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
                            if imgui.button("Delete"):
                                path = file_info['path']
                                if path and os.path.exists(path):
                                    try:
                                        send2trash(path)
                                        if not os.path.exists(path):
                                            self.app.set_status_message(f"SUCCESS: Deleted {os.path.basename(path)}",
                                                                        level=logging.INFO)
                                        else:
                                            self.app.set_status_message(f"ERROR: File still exists.", level=logging.ERROR)
                                    except Exception as e:
                                        self.app.set_status_message(f"ERROR deleting file: {e}", level=logging.ERROR)
                                else:
                                    self.app.set_status_message("INFO: File not found or already deleted.", level=logging.INFO)
                                self._scan_files()  # Refresh UI immediately
                            if not delete_enabled:
                                imgui.pop_style_var()
                                imgui.internal.pop_item_flag()
                            imgui.pop_id()
                        imgui.tree_pop()
                    imgui.separator()
                    imgui.pop_id()

            # --- "DELETE ALL" POPUP DEFINITION ---
            # Move the popup definition here, inside is_visible, after all controls and file tree rendering
            if imgui.begin_popup_modal("ConfirmDeleteAll")[0]:
                imgui.text_ansi_colored("WARNING: This will delete ALL subfolders and files in the output directory!", 1.0, 0.2, 0.2)
                imgui.text(f"Directory: {os.path.abspath(self.output_folder)}")
                imgui.text("Contents will be moved to the recycle bin.")
                imgui.separator()
                if imgui.button("YES, DELETE EVERYTHING", width=200):
                    try:
                        root_folder = self.output_folder
                        if os.path.isdir(root_folder):
                            # Only delete contents, not the root folder itself
                            for entry in os.listdir(root_folder):
                                entry_path = os.path.join(root_folder, entry)
                                if self.delete_funscript_files:
                                    send2trash(entry_path)
                                else:
                                    if os.path.isdir(entry_path):
                                        # Delete all except .funscript files in subfolders
                                        for dirpath, dirnames, filenames in os.walk(entry_path):
                                            for filename in filenames:
                                                if filename.endswith('.funscript'):
                                                    continue
                                                file_path = os.path.join(dirpath, filename)
                                                try:
                                                    send2trash(file_path)
                                                except Exception as e:
                                                    self.app.set_status_message(f"ERROR deleting file: {file_path}: {e}", level=logging.ERROR)
                                        # Remove empty folders (except those containing .funscript files)
                                        for dirpath, dirnames, filenames in os.walk(entry_path, topdown=False):
                                            if all(f.endswith('.funscript') for f in filenames):
                                                continue
                                            try:
                                                if not os.listdir(dirpath):
                                                    send2trash(dirpath)
                                            except Exception:
                                                pass
                                    elif os.path.isfile(entry_path) and not entry_path.endswith('.funscript'):
                                        try:
                                            send2trash(entry_path)
                                        except Exception as e:
                                            self.app.set_status_message(f"ERROR deleting file: {entry_path}: {e}", level=logging.ERROR)
                        self.app.set_status_message("SUCCESS: All generated files have been deleted.", level=logging.INFO)
                    except Exception as e:
                        self.app.set_status_message(f"ERROR deleting all files: {e}", level=logging.ERROR)
                    finally:
                        self._scan_files()
                        imgui.close_current_popup()
                imgui.same_line()
                if imgui.button("Cancel", width=120):
                    imgui.close_current_popup()
                imgui.end_popup()

        imgui.end()
