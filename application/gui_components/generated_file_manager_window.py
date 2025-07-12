import imgui
import os
import shutil
import logging


class GeneratedFileManagerWindow:
    def __init__(self, app_instance):
        self.app = app_instance
        self.output_folder = self.app.app_settings.get("output_folder_path", "output")
        self.file_tree = {}
        self.total_size_mb = 0
        self.sort_by = 'name'
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
        imgui.set_next_window_focus()
        imgui.set_next_window_size(700, 500, condition=imgui.FIRST_USE_EVER)
        is_visible, is_open = imgui.begin("Generated File Manager", closable=True)

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
            imgui.same_line(max(0, imgui.get_window_width() - button_width - 15))
            if imgui.button(button_text): imgui.open_popup("ConfirmDeleteAll")  # This is the only popup left
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
                                shutil.rmtree(path)
                                if not os.path.exists(path):
                                    self.app.set_status_message(f"SUCCESS: Deleted folder {os.path.basename(path)}",
                                                                level=logging.INFO)
                                else:
                                    self.app.set_status_message(f"ERROR: Folder still exists.", level=logging.ERROR)
                            except Exception as e:
                                self.app.set_status_message(f"ERROR deleting folder: {e}", level=logging.ERROR)
                        else:
                            self.app.set_status_message("INFO: Folder not found or already deleted.",
                                                        level=logging.INFO)
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
                            if imgui.button("Delete"):
                                path = file_info['path']
                                if path and os.path.exists(path):
                                    try:
                                        os.remove(path)
                                        if not os.path.exists(path):
                                            self.app.set_status_message(f"SUCCESS: Deleted {os.path.basename(path)}",
                                                                        level=logging.INFO)
                                        else:
                                            self.app.set_status_message(f"ERROR: File still exists.",
                                                                        level=logging.ERROR)
                                    except Exception as e:
                                        self.app.set_status_message(f"ERROR deleting file: {e}", level=logging.ERROR)
                                else:
                                    self.app.set_status_message("INFO: File not found or already deleted.",
                                                                level=logging.INFO)
                                self._scan_files()  # Refresh UI immediately
                            imgui.pop_id()
                        imgui.tree_pop()
                    imgui.separator()
                    imgui.pop_id()

        # --- "DELETE ALL" POPUP DEFINITION ---
        # This is the only popup left, defined at the window's root level to ensure it works reliably.
        if imgui.begin_popup_modal("ConfirmDeleteAll")[0]:
            imgui.text_ansi_colored("WARNING: This will delete ALL subfolders and files in the output directory!", 1.0,
                                    0.2, 0.2)
            imgui.text(f"Directory: {os.path.abspath(self.output_folder)}")
            imgui.text("This action cannot be undone.")
            imgui.separator()
            if imgui.button("YES, DELETE EVERYTHING", width=200):
                try:
                    root_folder = self.output_folder
                    if os.path.isdir(root_folder):
                        shutil.rmtree(root_folder)
                        os.makedirs(root_folder, exist_ok=True)
                        self.app.set_status_message("SUCCESS: All generated files have been deleted.",
                                                    level=logging.INFO)
                    else:
                        self.app.set_status_message("INFO: Output folder did not exist.", level=logging.INFO)
                    self._scan_files()
                except Exception as e:
                    self.app.set_status_message(f"ERROR deleting all files: {e}", level=logging.ERROR)
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", width=120):
                imgui.close_current_popup()
            imgui.end_popup()

        imgui.end()