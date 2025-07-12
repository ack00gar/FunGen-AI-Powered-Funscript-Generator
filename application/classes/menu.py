import imgui
import os
import time
import glfw


class MainMenu:
    def __init__(self, app_instance):
        self.app = app_instance

    def render(self):

        if imgui.begin_main_menu_bar():
            # Cache frequently accessed sub-modules
            app_state = self.app.app_state_ui
            file_mgr = self.app.file_manager
            fs_proc = self.app.funscript_processor
            stage_proc = self.app.stage_processor

            # --- FILE MENU ---
            if imgui.begin_menu("File", True):

                # --- Project Creation ---
                if imgui.menu_item("New Project")[0]:
                    self.app.project_manager.new_project()
                if imgui.is_item_hovered(): imgui.set_tooltip("Create a new, empty project.")
                imgui.separator()

                # --- Open Sub-Menu ---
                if imgui.begin_menu("Open..."):
                    if imgui.menu_item("Project...")[0]:
                        self.app.project_manager.open_project_dialog()
                    if imgui.is_item_hovered(): imgui.set_tooltip("Open an existing project file (.fgnproj).")

                    if imgui.menu_item("Video...")[0]:
                        initial_video_dir = os.path.dirname(file_mgr.video_path) if file_mgr.video_path else None
                        if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(
                                self.app.gui_instance, 'file_dialog'):
                            self.app.gui_instance.file_dialog.show(
                                title="Open Video File", is_save=False,
                                callback=lambda fp: file_mgr.open_video_from_path(fp),
                                extension_filter="Video Files (*.mp4 *.mkv *.avi *.mov),*.mp4;*.mkv;*.avi;*.mov|All files (*.*),*.*",
                                initial_path=initial_video_dir
                            )
                    if imgui.is_item_hovered(): imgui.set_tooltip("Open a new video file for processing.")
                    imgui.end_menu()

                # Placeholder for a future "Open Recent" feature
                if imgui.begin_menu("Open Recent", enabled=False):
                    imgui.end_menu()
                if imgui.is_item_hovered(): imgui.set_tooltip("Feature not yet implemented.")
                imgui.separator()

                # --- Close and Save ---
                if imgui.menu_item("Close Project")[0]:
                    file_mgr.close_video_action(clear_funscript_unconditionally=True)
                if imgui.is_item_hovered(): imgui.set_tooltip("Close the current video and all associated data.")
                imgui.separator()

                can_save_project = self.app.project_manager.project_dirty or not self.app.project_manager.project_file_path
                if imgui.menu_item("Save Project", enabled=can_save_project)[0]:
                    self.app.project_manager.save_project_dialog()
                if imgui.is_item_hovered(): imgui.set_tooltip("Save the current project state.")

                if imgui.menu_item("Save Project As...")[0]:
                    self.app.project_manager.save_project_dialog(save_as=True)
                if imgui.is_item_hovered(): imgui.set_tooltip("Save the current project to a new file.")
                imgui.separator()

                # --- Import Sub-Menu ---
                if imgui.begin_menu("Import..."):
                    if imgui.menu_item("Funscript to Timeline 1...")[0]:
                        if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(self.app.gui_instance, 'file_dialog'):
                            initial_dir = os.path.dirname(file_mgr.video_path) if file_mgr.video_path else None
                            self.app.gui_instance.file_dialog.show(
                                title="Import Funscript to Timeline 1", is_save=False,
                                callback=lambda fp: file_mgr.load_funscript_to_timeline(fp, timeline_num=1),
                                extension_filter="Funscript Files (*.funscript),*.funscript|All files (*.*),*.*",
                                initial_path=initial_dir
                            )
                    if imgui.is_item_hovered(): imgui.set_tooltip("Load a .funscript file into the primary timeline.")

                    if imgui.menu_item("Funscript to Timeline 2...")[0]:
                        if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(self.app.gui_instance, 'file_dialog'):
                            initial_dir = os.path.dirname(file_mgr.video_path) if file_mgr.video_path else None
                            self.app.gui_instance.file_dialog.show(
                                title="Import Funscript to Timeline 2", is_save=False,
                                callback=lambda fp: file_mgr.load_funscript_to_timeline(fp, timeline_num=2),
                                extension_filter="Funscript Files (*.funscript),*.funscript|All files (*.*),*.*",
                                initial_path=initial_dir
                            )
                    if imgui.is_item_hovered(): imgui.set_tooltip("Load a .funscript file into the secondary timeline.")

                    imgui.separator()
                    if imgui.menu_item("Stage 2 Overlay Data...")[0]:
                        if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(self.app.gui_instance, 'file_dialog'):
                            initial_dir = None
                            if file_mgr.stage2_output_msgpack_path:
                                initial_dir = os.path.dirname(file_mgr.stage2_output_msgpack_path)
                            elif file_mgr.video_path:
                                path_in_output = file_mgr.get_output_path_for_file(file_mgr.video_path, "_stage2_overlay.msgpack")
                                initial_dir = os.path.dirname(path_in_output)
                            self.app.gui_instance.file_dialog.show(
                                title="Load Stage 2 Overlay Data", is_save=False,
                                callback=lambda fp: file_mgr.load_stage2_overlay_data(fp),
                                extension_filter="MsgPack Files (*.msgpack),*.msgpack|All files (*.*),*.*",
                                initial_path=initial_dir
                            )
                    if imgui.is_item_hovered(): imgui.set_tooltip("Load pre-computed Stage 2 analysis data for display.")
                    imgui.end_menu()

                # --- Export Sub-Menu ---
                if imgui.begin_menu("Export..."):
                    if imgui.menu_item("Funscript from Timeline 1...")[0]:
                        if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(self.app.gui_instance, 'file_dialog'):
                            suggested_filename = "output.funscript"
                            initial_dir = None
                            if file_mgr.video_path:
                                path_in_output = file_mgr.get_output_path_for_file(file_mgr.video_path, ".funscript")
                                suggested_filename = os.path.basename(path_in_output)
                                initial_dir = os.path.dirname(path_in_output)
                            self.app.gui_instance.file_dialog.show(
                                title="Export Funscript from Timeline 1", is_save=True,
                                callback=lambda fp: file_mgr.save_funscript_from_timeline(fp, timeline_num=1),
                                extension_filter="Funscript Files (*.funscript),*.funscript|All files (*.*),*.*",
                                initial_filename=suggested_filename,
                                initial_path=initial_dir
                            )
                    if imgui.is_item_hovered(): imgui.set_tooltip("Save the primary timeline as a .funscript file.")

                    if imgui.menu_item("Funscript from Timeline 2...")[0]:
                        if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(self.app.gui_instance, 'file_dialog'):
                            suggested_filename = "output.roll.funscript"
                            initial_dir = None
                            if file_mgr.video_path:
                                path_in_output = file_mgr.get_output_path_for_file(file_mgr.video_path, ".roll.funscript")
                                suggested_filename = os.path.basename(path_in_output)
                                initial_dir = os.path.dirname(path_in_output)
                            self.app.gui_instance.file_dialog.show(
                                title="Export Funscript from Timeline 2", is_save=True,
                                callback=lambda fp: file_mgr.save_funscript_from_timeline(fp, timeline_num=2),
                                extension_filter="Funscript Files (*.funscript),*.funscript|All files (*.*),*.*",
                                initial_filename=suggested_filename,
                                initial_path=initial_dir
                            )
                    if imgui.is_item_hovered(): imgui.set_tooltip("Save the secondary timeline as a .funscript file.")
                    imgui.end_menu()
                imgui.separator()

                # --- Exit ---
                if imgui.menu_item("Exit")[0]:
                    if hasattr(self.app, 'gui_instance') and self.app.gui_instance.window:
                        glfw.set_window_should_close(self.app.gui_instance.window, True)
                if imgui.is_item_hovered(): imgui.set_tooltip("Exit the application.")

                imgui.end_menu()

            # --- EDIT MENU ---
            if imgui.begin_menu("Edit", True):
                manager_t1 = fs_proc._get_undo_manager(1)
                can_undo_t1 = manager_t1.can_undo() if manager_t1 else False
                if imgui.menu_item("Undo T1 Change", "Ctrl+Z", selected=False, enabled=can_undo_t1)[0]:
                    fs_proc.perform_undo_redo(1, 'undo')

                can_redo_t1 = manager_t1.can_redo() if manager_t1 else False
                if imgui.menu_item("Redo T1 Change", "Ctrl+Y", selected=False, enabled=can_redo_t1)[0]:
                    fs_proc.perform_undo_redo(1, 'redo')
                imgui.separator()

                if app_state.show_funscript_interactive_timeline2:
                    manager_t2 = fs_proc._get_undo_manager(2)
                    can_undo_t2 = manager_t2.can_undo() if manager_t2 else False
                    if imgui.menu_item("Undo T2 Change", "Alt+Ctrl+Z", selected=False, enabled=can_undo_t2)[0]:
                        fs_proc.perform_undo_redo(2, 'undo')

                    can_redo_t2 = manager_t2.can_redo() if manager_t2 else False
                    if imgui.menu_item("Redo T2 Change", "Alt+Ctrl+Y", selected=False, enabled=can_redo_t2)[0]:
                        fs_proc.perform_undo_redo(2, 'redo')
                else:
                    imgui.text_disabled("Timeline 2 Undo/Redo (Timeline 2 not visible)")

                imgui.end_menu()

            # --- VIEW MENU (Simplified) ---
            if imgui.begin_menu("View", True):

                imgui.text("UI Layout Mode")
                imgui.indent()
                is_fixed_mode = app_state.ui_layout_mode == 'fixed'
                if imgui.radio_button("Fixed Panels##UILayoutModeFixed", is_fixed_mode):
                    if not is_fixed_mode: app_state.ui_layout_mode = 'fixed'; self.app.project_manager.project_dirty = True
                is_floating_mode = app_state.ui_layout_mode == 'floating'
                if imgui.radio_button("Floating Windows##UILayoutModeFloating", is_floating_mode):
                    if not is_floating_mode: app_state.ui_layout_mode = 'floating'; app_state.just_switched_to_floating = True; self.app.project_manager.project_dirty = True
                imgui.unindent()
                imgui.separator()

                imgui.text("Layout Options")
                imgui.indent()
                if not hasattr(app_state, 'full_width_nav'):
                    app_state.full_width_nav = False
                clicked, app_state.full_width_nav = imgui.menu_item("Full-Width Navigation Bar", selected=app_state.full_width_nav, enabled=is_fixed_mode)
                if clicked: self.app.project_manager.project_dirty = True
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Only available in 'Fixed Panels' layout mode.")
                imgui.unindent()
                imgui.separator()

                # --- Main Window Toggles (for Floating Mode) ---
                imgui.text("Main Panels")
                imgui.indent()
                if app_state.ui_layout_mode == 'floating':
                    clicked, app_state.show_control_panel_window = imgui.menu_item("Control Panel",
                                                                                   selected=app_state.show_control_panel_window)
                    if clicked: self.app.project_manager.project_dirty = True
                    clicked, app_state.show_info_graphs_window = imgui.menu_item("Info & Graphs",
                                                                                 selected=app_state.show_info_graphs_window)
                    if clicked: self.app.project_manager.project_dirty = True
                    clicked, app_state.show_video_display_window = imgui.menu_item("Video Display",
                                                                                   selected=app_state.show_video_display_window)
                    if clicked: self.app.project_manager.project_dirty = True
                    clicked, app_state.show_video_navigation_window = imgui.menu_item("Video Navigation",
                                                                                      selected=app_state.show_video_navigation_window)
                    if clicked: self.app.project_manager.project_dirty = True
                else:
                    imgui.text_disabled("Window toggles are for Floating Mode.")
                imgui.unindent()
                imgui.separator()

                imgui.text("Timeline Editors & Previews")
                imgui.indent()
                clicked, app_state.show_funscript_interactive_timeline = imgui.menu_item("Interactive Timeline 1",
                                                                                         selected=app_state.show_funscript_interactive_timeline)
                if clicked: self.app.project_manager.project_dirty = True
                clicked, app_state.show_funscript_interactive_timeline2 = imgui.menu_item("Interactive Timeline 2",
                                                                                          selected=app_state.show_funscript_interactive_timeline2)
                if clicked: self.app.project_manager.project_dirty = True
                imgui.separator()
                clicked, app_state.show_funscript_timeline = imgui.menu_item("Funscript Preview Bar",
                                                                             selected=app_state.show_funscript_timeline)
                if clicked: self.app.project_manager.project_dirty = True
                clicked, app_state.show_heatmap = imgui.menu_item("Heatmap", selected=app_state.show_heatmap)
                if clicked: self.app.project_manager.project_dirty = True
                imgui.unindent()
                imgui.separator()

                imgui.text("Overlays & Aux Windows")
                imgui.indent()
                clicked, app_state.show_gauge_window = imgui.menu_item("Script Gauge",
                                                                       selected=app_state.show_gauge_window)
                if clicked: self.app.project_manager.project_dirty = True
                clicked, app_state.show_lr_dial_graph = imgui.menu_item("L/R Dial Graph",
                                                                        selected=app_state.show_lr_dial_graph)
                if clicked: self.app.project_manager.project_dirty = True

                # Added defensive hasattr check
                if not hasattr(app_state, 'show_chapter_list_window'):
                    app_state.show_chapter_list_window = False
                clicked, app_state.show_chapter_list_window = imgui.menu_item("Chapter List",
                                                                              selected=app_state.show_chapter_list_window)
                if clicked: self.app.project_manager.project_dirty = True

                imgui.separator()
                if self.app.tracker:
                    clicked, current_val = imgui.menu_item("Show Detections/Masks", selected=app_state.ui_show_masks)
                    if clicked: app_state.set_tracker_ui_flag("show_masks", current_val)
                    clicked, current_val = imgui.menu_item("Show Optical Flow", selected=app_state.ui_show_flow)
                    if clicked: app_state.set_tracker_ui_flag("show_flow", current_val)
                can_show_s2 = stage_proc.stage2_overlay_data is not None
                clicked, current_val = imgui.menu_item("Show Stage 2 Overlay", selected=app_state.show_stage2_overlay,
                                                       enabled=can_show_s2)
                if clicked:
                    app_state.show_stage2_overlay = current_val
                    self.app.project_manager.project_dirty = True

                clicked, _ = imgui.menu_item("Audio Waveform", selected=app_state.show_audio_waveform)
                if clicked:
                    self.app.toggle_waveform_visibility()
                    self.app.project_manager.project_dirty = True
                imgui.unindent()
                imgui.end_menu()

            # --- TOOLS MENU ---
            if imgui.begin_menu("Tools", True):
                # The menu item is enabled only when a video is loaded.
                can_calibrate = file_mgr.video_path is not None
                if imgui.menu_item("Start Latency Calibration...", enabled=can_calibrate)[0]:
                    # Call the robust start method which includes its own checks
                    if hasattr(self.app, 'calibration'):
                        self.app.calibration.start_latency_calibration()
                if imgui.is_item_hovered():
                    tooltip = "Calibrate latency. Requires a video to be loaded and points on Timeline 1."
                    if not can_calibrate:
                        tooltip = "Please load a video to enable calibration."
                    imgui.set_tooltip(tooltip)

                # --- Generated File Manager ---
                clicked, selected = imgui.menu_item("Manage Generated Files...", selected=app_state.show_generated_file_manager)
                if clicked:
                    self.app.toggle_file_manager_window()

                # --- TensorRT Compiler Tool ---
                if not hasattr(self.app, 'tensorrt_compiler_window'):
                    self.app.tensorrt_compiler_window = None
                if imgui.menu_item("Compile YOLO Model to TensorRT (.engine)...")[0]:
                    from application.gui_components.tensorrt_compiler_window import TensorRTCompilerWindow
                    def on_close():
                        self.app.tensorrt_compiler_window = None
                    self.app.tensorrt_compiler_window = TensorRTCompilerWindow(self.app, on_close_callback=on_close)
                imgui.end_menu()

            # --- STATUS MESSAGE ---
            if app_state.status_message and time.time() < app_state.status_message_time:
                text_size_status = imgui.calc_text_size(app_state.status_message)
                menu_bar_width = imgui.get_window_width()
                cursor_x_after_menus = imgui.get_cursor_pos_x()
                padding_needed = menu_bar_width - cursor_x_after_menus - text_size_status[0] - \
                                 imgui.get_style().item_spacing[0] * 2
                if padding_needed > 0:
                    imgui.same_line(cursor_x_after_menus + padding_needed)
                imgui.text_colored(app_state.status_message, 0.9, 0.9, 0.3, 1.0)
            elif app_state.status_message:
                app_state.status_message = ""

            imgui.end_main_menu_bar()
