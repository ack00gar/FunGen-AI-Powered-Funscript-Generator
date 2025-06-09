# menu.py
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
            calibration_mgr = self.app.calibration
            energy_saver_mgr = self.app.energy_saver
            event_handlers = self.app.event_handlers

            # --- FILE MENU ---
            if imgui.begin_menu("File", True):
                if imgui.menu_item("New Project")[0]:
                    self.app.project_manager.new_project()
                if imgui.is_item_hovered(): imgui.set_tooltip("Create a new, empty project.")

                if imgui.menu_item("Open Project...")[0]:
                    self.app.project_manager.open_project_dialog()
                if imgui.is_item_hovered(): imgui.set_tooltip("Open an existing project file.")

                can_save_project = self.app.project_manager.project_dirty or not self.app.project_manager.project_file_path
                if imgui.menu_item("Save Project", enabled=can_save_project)[0]:
                    self.app.project_manager.save_project_dialog()
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Save the current project. Enabled if there are unsaved changes or no project path.")

                if imgui.menu_item("Save Project As...")[0]:
                    self.app.project_manager.save_project_dialog(save_as=True)
                if imgui.is_item_hovered(): imgui.set_tooltip("Save the current project to a new file.")
                imgui.separator()

                if imgui.menu_item("Open Video...")[0]:
                    initial_video_dir = os.path.dirname(file_mgr.video_path) if file_mgr.video_path else None
                    if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(
                            self.app.gui_instance, 'file_dialog'):
                        self.app.gui_instance.file_dialog.show(
                            title="Open Video File",
                            is_save=False,
                            callback=lambda fp: file_mgr.open_video_from_path(fp) if hasattr(file_mgr,
                                                                                             'open_video_from_path') else self.app.logger.error(
                                "file_mgr.open_video_from_path not implemented"),
                            extension_filter="Video Files (*.mp4 *.mkv *.avi *.mov),*.mp4;*.mkv;*.avi;*.mov|All files (*.*),*.*",
                            initial_path=initial_video_dir
                        )
                    else:
                        self.app.logger.error("Open Video: File dialog bridge not found on app (gui_instance).")
                if imgui.is_item_hovered(): imgui.set_tooltip("Open a video file for processing.")

                if imgui.menu_item("Close Video")[0]:
                    file_mgr.close_video_action()
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Close the currently loaded video, keeping Funscripts loaded.")

                if imgui.menu_item("Close Video and Unload Funscripts")[0]:
                    file_mgr.close_video_action(clear_funscript_unconditionally=True)
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Close the currently loaded video and unload all associated Funscript data.")
                imgui.separator()

                if imgui.menu_item("Save Funscript (Timeline 1)")[0]:
                    actions_t1 = fs_proc.get_actions('primary')
                    if actions_t1:
                        sugg_fn_t1 = "timeline1.funscript"
                        init_dir_t1 = None
                        if file_mgr.video_path:
                            base, _ = os.path.splitext(os.path.basename(file_mgr.video_path))
                            sugg_fn_t1 = base + "_t1.funscript"
                            init_dir_t1 = os.path.dirname(file_mgr.video_path)
                        elif file_mgr.funscript_path:
                            init_dir_t1 = os.path.dirname(file_mgr.funscript_path)

                        if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(
                                self.app.gui_instance, 'file_dialog'):
                            self.app.gui_instance.file_dialog.show(
                                title="Save Timeline 1 Funscript As", is_save=True,
                                callback=lambda fp: file_mgr.save_funscript_from_timeline(fp, 1),
                                extension_filter="Funscript files (*.funscript),*.funscript|All files (*.*),*.*",
                                initial_filename=sugg_fn_t1, initial_path=init_dir_t1
                            )
                        else:
                            self.app.logger.error("Save T1: File dialog bridge not found on app (gui_instance).")
                    else:
                        self.app.logger.info("No Funscript data in Timeline 1 to save.", extra={'status_message': True})
                if imgui.is_item_hovered(): imgui.set_tooltip("Save the Funscript data from Timeline 1.")

                if imgui.menu_item("Save Funscript (Timeline 2)")[0]:
                    actions_t2 = fs_proc.get_actions('secondary')
                    if actions_t2:
                        sugg_fn_t2 = "timeline2.funscript"
                        init_dir_t2 = None
                        if file_mgr.video_path:
                            base, _ = os.path.splitext(os.path.basename(file_mgr.video_path))
                            sugg_fn_t2 = base + "_t2.funscript"
                            init_dir_t2 = os.path.dirname(file_mgr.video_path)
                        if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(
                                self.app.gui_instance, 'file_dialog'):
                            self.app.gui_instance.file_dialog.show(
                                title="Save Timeline 2 Funscript As", is_save=True,
                                callback=lambda fp: file_mgr.save_funscript_from_timeline(fp, 2),
                                extension_filter="Funscript files (*.funscript),*.funscript|All files (*.*),*.*",
                                initial_filename=sugg_fn_t2, initial_path=init_dir_t2
                            )
                        else:
                            self.app.logger.error("Save T2: File dialog bridge not found on app (gui_instance).")
                    else:
                        self.app.logger.info("No Funscript data in Timeline 2 to save.", extra={'status_message': True})
                if imgui.is_item_hovered(): imgui.set_tooltip("Save the Funscript data from Timeline 2.")
                imgui.separator()

                if imgui.menu_item("Load Funscript (Timeline 1)...")[0]:
                    initial_load_dir = os.path.dirname(
                        file_mgr.loaded_funscript_path) if file_mgr.loaded_funscript_path else (
                        os.path.dirname(file_mgr.video_path) if file_mgr.video_path else None)
                    if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(self.app.gui_instance,
                                                                                               'file_dialog'):
                        self.app.gui_instance.file_dialog.show(
                            title="Load Funscript for Timeline 1", is_save=False,
                            callback=lambda fp: file_mgr.load_funscript_to_timeline(fp, 1),
                            extension_filter="Funscript files (*.funscript),*.funscript|All files (*.*),*.*",
                            initial_path=initial_load_dir
                        )
                    else:
                        self.app.logger.error("Load T1: File dialog bridge not found on app (gui_instance).")
                if imgui.is_item_hovered(): imgui.set_tooltip("Load a Funscript file into Timeline 1.")

                if imgui.menu_item("Load Funscript (Timeline 2)...")[0]:
                    initial_load_dir_t2 = os.path.dirname(
                        file_mgr.loaded_funscript_path) if file_mgr.loaded_funscript_path else (
                        os.path.dirname(file_mgr.video_path) if file_mgr.video_path else None)
                    if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(self.app.gui_instance,
                                                                                               'file_dialog'):
                        self.app.gui_instance.file_dialog.show(
                            title="Load Funscript for Timeline 2", is_save=False,
                            callback=lambda fp: file_mgr.load_funscript_to_timeline(fp, 2),
                            extension_filter="Funscript files (*.funscript),*.funscript|All files (*.*),*.*",
                            initial_path=initial_load_dir_t2
                        )
                    else:
                        self.app.logger.error("Load T2: File dialog bridge not found on app (gui_instance).")
                if imgui.is_item_hovered(): imgui.set_tooltip("Load a Funscript file into Timeline 2.")
                imgui.separator()

                if imgui.menu_item("Unload Funscript (Timeline 1)")[0]:
                    fs_proc.clear_timeline_history_and_set_new_baseline(1, [], "Funscript Unloaded (Menu T1)")
                    file_mgr.loaded_funscript_path = ""
                    self.app.logger.info("Funscript for Timeline 1 unloaded.", extra={'status_message': True})
                if imgui.is_item_hovered(): imgui.set_tooltip("Clear Funscript data from Timeline 1.")

                if imgui.menu_item("Unload Funscript (Timeline 2)")[0]:
                    fs_proc.clear_timeline_history_and_set_new_baseline(2, [], "Funscript Unloaded (Menu T2)")
                    self.app.logger.info("Funscript for Timeline 2 unloaded.", extra={'status_message': True})
                if imgui.is_item_hovered(): imgui.set_tooltip("Clear Funscript data from Timeline 2.")
                imgui.separator()

                if imgui.menu_item("Load Stage 2 Overlay Data...")[0]:
                    initial_overlay_dir = os.path.dirname(
                        file_mgr.stage2_output_msgpack_path) if file_mgr.stage2_output_msgpack_path else (
                        os.path.dirname(file_mgr.video_path) if file_mgr.video_path else None)
                    if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(self.app.gui_instance,
                                                                                               'file_dialog'):
                        self.app.gui_instance.file_dialog.show(
                            title="Load Stage 2 Overlay Data", is_save=False,
                            callback=file_mgr.load_stage2_overlay_data,
                            extension_filter="Msgpack files (*.msgpack),*.msgpack|All files (*.*),*.*",
                            initial_path=initial_overlay_dir
                        )
                    else:
                        self.app.logger.error("Load S2 Overlay: File dialog bridge not found on app (gui_instance).")
                if imgui.is_item_hovered(): imgui.set_tooltip("Load pre-computed Stage 2 analysis data for overlay.")

                can_unload_s2_overlay = stage_proc.stage2_overlay_data is not None
                if imgui.menu_item("Unload Stage 2 Overlay Data", enabled=can_unload_s2_overlay)[0]:
                    file_mgr.clear_stage2_overlay_data()
                    stage_proc.stage2_status_text = "Not run/loaded."
                    app_state.show_stage2_overlay = False
                    self.app.logger.info("Stage 2 overlay data unloaded.", extra={'status_message': True})
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Clear loaded Stage 2 overlay data. Enabled if data is loaded.")
                imgui.separator()

                if imgui.menu_item("Exit")[0]:
                    if hasattr(event_handlers, 'request_app_exit'):
                        event_handlers.request_app_exit()
                    elif hasattr(self.app, 'request_app_exit'):
                        self.app.request_app_exit()
                    else:
                        self.app.logger.warning("No app exit request method found, attempting direct GLFW close.")
                        if hasattr(self.app, 'gui_instance') and self.app.gui_instance and self.app.gui_instance.window:
                            glfw.set_window_should_close(self.app.gui_instance.window, True)
                if imgui.is_item_hovered(): imgui.set_tooltip("Exit the application.")

                imgui.end_menu()

            # --- VIEW MENU ---
            if imgui.begin_menu("View", True):

                imgui.text("UI Layout Mode")
                imgui.indent()

                is_fixed_mode = app_state.ui_layout_mode == 'fixed'
                # Assign the single boolean return value to one variable
                if imgui.radio_button("Fixed Panels##UILayoutModeFixed", is_fixed_mode):
                    if not is_fixed_mode:
                        app_state.ui_layout_mode = 'fixed'
                        self.app.project_manager.project_dirty = True
                        self.app.logger.info("UI Layout changed to Fixed Panels.", extra={'status_message': True})

                is_floating_mode = app_state.ui_layout_mode == 'floating'
                # Assign the single boolean return value to one variable
                if imgui.radio_button("Floating Windows##UILayoutModeFloating", is_floating_mode):
                    if not is_floating_mode:
                        app_state.ui_layout_mode = 'floating'
                        self.app.project_manager.project_dirty = True
                        self.app.logger.info("UI Layout changed to Floating Windows.", extra={'status_message': True})

                imgui.unindent()
                imgui.separator()

                if imgui.begin_menu("Panel Sections Visibility"):
                    if imgui.begin_menu("Control Panel Sections"):
                        if not hasattr(app_state,
                                       'show_general_tracking_settings_section'): app_state.show_general_tracking_settings_section = True
                        if not hasattr(app_state,
                                       'show_tracking_control_section'): app_state.show_tracking_control_section = True
                        if not hasattr(app_state,
                                       'show_range_selection_section'): app_state.show_range_selection_section = True
                        if not hasattr(app_state,
                                       'show_funscript_processing_tools_section'): app_state.show_funscript_processing_tools_section = True
                        if not hasattr(app_state,
                                       'show_funscript_window_controls_section'): app_state.show_funscript_window_controls_section = True
                        if not hasattr(app_state,
                                       'show_auto_post_processing_section'): app_state.show_auto_post_processing_section = True

                        clicked, app_state.show_general_tracking_settings_section = imgui.menu_item(
                            "General Tracking Settings", selected=app_state.show_general_tracking_settings_section)
                        if clicked: self.app.project_manager.project_dirty = True
                        clicked, app_state.show_tracking_control_section = imgui.menu_item("Tracking Control",
                                                                                           selected=app_state.show_tracking_control_section)
                        if clicked: self.app.project_manager.project_dirty = True
                        clicked, app_state.show_range_selection_section = imgui.menu_item("Range Selection",
                                                                                          selected=app_state.show_range_selection_section)
                        if clicked: self.app.project_manager.project_dirty = True
                        clicked, app_state.show_funscript_processing_tools_section = imgui.menu_item(
                            "Funscript Processing Tools", selected=app_state.show_funscript_processing_tools_section)
                        if clicked: self.app.project_manager.project_dirty = True

                        clicked, app_state.show_auto_post_processing_section = imgui.menu_item(
                            "Automatic Post-Processing", selected=app_state.show_auto_post_processing_section)
                        if clicked: self.app.project_manager.project_dirty = True

                        clicked, app_state.show_funscript_window_controls_section = imgui.menu_item(
                            "Funscript Window Controls", selected=app_state.show_funscript_window_controls_section)
                        if clicked: self.app.project_manager.project_dirty = True
                        imgui.end_menu()

                    if imgui.begin_menu("Info Panel Sections"):
                        # Initialize if not present (first run or new attributes)
                        if not hasattr(app_state, 'show_video_info_section'): app_state.show_video_info_section = True
                        if not hasattr(app_state,
                                       'show_video_settings_section'): app_state.show_video_settings_section = True
                        if not hasattr(app_state,
                                       'show_funscript_info_t1_section'): app_state.show_funscript_info_t1_section = True
                        if not hasattr(app_state,
                                       'show_funscript_info_t2_section'): app_state.show_funscript_info_t2_section = True
                        if not hasattr(app_state,
                                       'show_undo_redo_history_section'): app_state.show_undo_redo_history_section = True

                        clicked, app_state.show_video_info_section = imgui.menu_item("Video Information",
                                                                                     selected=app_state.show_video_info_section)
                        if clicked: self.app.project_manager.project_dirty = True
                        clicked, app_state.show_video_settings_section = imgui.menu_item("Video Settings",
                                                                                         selected=app_state.show_video_settings_section)
                        if clicked: self.app.project_manager.project_dirty = True
                        clicked, app_state.show_funscript_info_t1_section = imgui.menu_item("Funscript Info (T1)",
                                                                                            selected=app_state.show_funscript_info_t1_section)
                        if clicked: self.app.project_manager.project_dirty = True

                        t2_section_enabled = app_state.show_funscript_interactive_timeline2
                        clicked, new_val = imgui.menu_item("Funscript Info (T2)",
                                                           selected=app_state.show_funscript_info_t2_section,
                                                           enabled=t2_section_enabled)
                        if clicked:
                            app_state.show_funscript_info_t2_section = new_val
                            self.app.project_manager.project_dirty = True
                        if not t2_section_enabled and imgui.is_item_hovered(): imgui.set_tooltip(
                            "Enable Interactive Timeline 2 to control this section.")

                        clicked, app_state.show_undo_redo_history_section = imgui.menu_item("Undo-Redo History",
                                                                                            selected=app_state.show_undo_redo_history_section)
                        if clicked: self.app.project_manager.project_dirty = True
                        imgui.end_menu()
                    imgui.end_menu()
                imgui.separator()

                imgui.text("General UI Elements")
                imgui.indent()
                clicked_gauge, current_show_gauge = imgui.menu_item("Script Gauge",
                                                                    selected=app_state.show_gauge_window)
                if clicked_gauge:
                    app_state.show_gauge_window = current_show_gauge
                    self.app.project_manager.project_dirty = True
                if imgui.is_item_hovered(): imgui.set_tooltip("Toggle visibility of the script action gauge window.")

                clicked_lr_dial, current_show_lr_dial = imgui.menu_item("L/R Dial Graph",
                                                                        selected=app_state.show_lr_dial_graph)
                if clicked_lr_dial: app_state.show_lr_dial_graph = current_show_lr_dial; self.app.project_manager.project_dirty = True
                if imgui.is_item_hovered(): imgui.set_tooltip("Toggle visibility of the L/R dial graph window.")

                if not hasattr(app_state, 'show_chapter_list_window'): app_state.show_chapter_list_window = False
                clicked_chapter_list, current_show_chapter_list = imgui.menu_item("Chapter List",
                                                                                 selected=app_state.show_chapter_list_window)
                if clicked_chapter_list:
                    app_state.show_chapter_list_window = current_show_chapter_list
                    self.app.project_manager.project_dirty = True
                if imgui.is_item_hovered(): imgui.set_tooltip("Toggle visibility of the floating chapter list window.")

                imgui.separator()
                clicked_video_panel, show_video_panel_new = imgui.menu_item("Video Display Panel", selected=app_state.show_video_display_window)
                if clicked_video_panel:
                    app_state.show_video_display_window = show_video_panel_new
                    self.app.project_manager.project_dirty = True
                if imgui.is_item_hovered(): imgui.set_tooltip("Toggle visibility of the Video Display panel/window.")

                imgui.unindent()
                imgui.separator()

                imgui.text("Main Timeline (T1)")
                imgui.indent()
                clicked_it1, current_show_it1 = imgui.menu_item("Interactive Timeline 1",
                                                                selected=app_state.show_funscript_interactive_timeline)
                if clicked_it1: app_state.show_funscript_interactive_timeline = current_show_it1; self.app.project_manager.project_dirty = True
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Toggle visibility of the main interactive Funscript timeline editor.")

                clicked_fpb1, current_show_fpb1 = imgui.menu_item("Funscript Preview Bar (T1)",
                                                                  selected=app_state.show_funscript_timeline)
                if clicked_fpb1: app_state.show_funscript_timeline = current_show_fpb1; self.app.project_manager.project_dirty = True
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Toggle visibility of the Funscript preview bar for Timeline 1 under the video player.")

                clicked_hm1, current_show_hm1 = imgui.menu_item("Heatmap (T1)", selected=app_state.show_heatmap)
                if clicked_hm1: app_state.show_heatmap = current_show_hm1; self.app.project_manager.project_dirty = True
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Toggle visibility of the Funscript action heatmap for Timeline 1 under the video player.")
                imgui.unindent()
                imgui.separator()

                imgui.text("Secondary Timeline (T2)")
                imgui.indent()
                clicked_it2, current_show_it2 = imgui.menu_item("Interactive Timeline 2",
                                                                selected=app_state.show_funscript_interactive_timeline2)
                if clicked_it2: app_state.show_funscript_interactive_timeline2 = current_show_it2; self.app.project_manager.project_dirty = True
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Toggle visibility of the secondary interactive Funscript timeline editor.")
                imgui.unindent()
                imgui.separator()

                imgui.text("Live Tracker Visuals (on video)")
                imgui.indent()
                if self.app.tracker:
                    clicked_masks, current_ui_show_masks = imgui.menu_item("Show Detections/Masks",
                                                                           selected=app_state.ui_show_masks)
                    if clicked_masks: app_state.set_tracker_ui_flag("show_masks", current_ui_show_masks)
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "Toggle display of detection boxes and masks on the video frame.")

                    clicked_flow, current_ui_show_flow = imgui.menu_item("Show Optical Flow",
                                                                         selected=app_state.ui_show_flow)
                    if clicked_flow: app_state.set_tracker_ui_flag("show_flow", current_ui_show_flow)
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "Toggle display of optical flow visualization on the video frame.")

                else:
                    imgui.text_disabled("Tracker visuals N/A (No tracker or video)")
                imgui.unindent()
                imgui.separator()

                imgui.text("Analysis Visuals")
                imgui.indent()
                can_show_s2_overlay = stage_proc.stage2_overlay_data is not None
                clicked_s2_overlay, current_show_s2_overlay = imgui.menu_item("Show Stage 2 Overlay",
                                                                              selected=app_state.show_stage2_overlay,
                                                                              enabled=can_show_s2_overlay)
                if clicked_s2_overlay: app_state.show_stage2_overlay = current_show_s2_overlay; self.app.project_manager.project_dirty = True
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Toggle visibility of the Stage 2 analysis overlay on the video. Enabled if overlay data is loaded.")
                imgui.unindent()
                imgui.end_menu()

            # --- EDIT MENU ---
            if imgui.begin_menu("Edit", True):
                manager_t1 = fs_proc._get_undo_manager(1)
                can_undo_t1 = manager_t1.can_undo() if manager_t1 else False
                if imgui.menu_item("Undo T1 Change", "Ctrl+Z", selected=False, enabled=can_undo_t1)[0]:
                    fs_proc.perform_undo_redo(1, 'undo')
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Undo the last change in Timeline 1. Enabled if an undo action is available.")

                can_redo_t1 = manager_t1.can_redo() if manager_t1 else False
                if imgui.menu_item("Redo T1 Change", "Ctrl+Y", selected=False, enabled=can_redo_t1)[0]:
                    fs_proc.perform_undo_redo(1, 'redo')
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Redo the last undone change in Timeline 1. Enabled if a redo action is available.")
                imgui.separator()

                if app_state.show_funscript_interactive_timeline2:
                    manager_t2 = fs_proc._get_undo_manager(2)
                    can_undo_t2 = manager_t2.can_undo() if manager_t2 else False
                    if imgui.menu_item("Undo T2 Change", "Alt+Ctrl+Z", selected=False, enabled=can_undo_t2)[0]:
                        fs_proc.perform_undo_redo(2, 'undo')
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "Undo the last change in Timeline 2. Enabled if T2 is visible and an undo action is available.")

                    can_redo_t2 = manager_t2.can_redo() if manager_t2 else False
                    if imgui.menu_item("Redo T2 Change", "Alt+Ctrl+Y", selected=False, enabled=can_redo_t2)[0]:
                        fs_proc.perform_undo_redo(2, 'redo')
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "Redo the last undone change in Timeline 2. Enabled if T2 is visible and a redo action is available.")
                else:
                    imgui.text_disabled("Timeline 2 Undo/Redo (Timeline 2 not visible)")

                imgui.end_menu()

            # --- PROCESSING MENU ---
            if imgui.begin_menu("Processing", True):
                imgui.text("Video Decoding")
                imgui.indent()
                hw_accel_options = self.app.available_ffmpeg_hwaccels if hasattr(self.app,
                                                                                 'available_ffmpeg_hwaccels') else [
                    "auto", "none"]
                hw_accel_display = [opt.replace("videotoolbox", "VideoToolbox (macOS)") for opt in hw_accel_options]
                current_method_val = self.app.hardware_acceleration_method
                try:
                    current_hw_idx = hw_accel_options.index(current_method_val)
                except ValueError:
                    current_hw_idx = 0
                    if hw_accel_options: self.app.hardware_acceleration_method = hw_accel_options[0]

                imgui.push_item_width(200)  # Give combo more width
                changed_hw_accel, new_hw_idx = imgui.combo("Hardware Acceleration##HWAccelMethodProcessingMenu",
                                                           current_hw_idx, hw_accel_display)
                imgui.pop_item_width()
                if changed_hw_accel:
                    self.app.hardware_acceleration_method = hw_accel_options[new_hw_idx]
                    self.app.logger.info(
                        f"Hardware acceleration set to: {self.app.hardware_acceleration_method}. Reload video to apply.",
                        extra={'status_message': True})
                    energy_saver_mgr.reset_activity_timer()
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Select FFmpeg hardware acceleration for video decoding. Requires video reload/re-open to apply.")
                imgui.unindent()
                imgui.separator()

                imgui.text("AI Models & Batch Inference")
                imgui.indent()
                current_yolo_path_display = self.app.yolo_detection_model_path_setting if self.app.yolo_detection_model_path_setting else "Not set"
                imgui.input_text("Detection Model##S1YOLOPathProcessingMenu", current_yolo_path_display, 256,
                                 flags=imgui.INPUT_TEXT_READ_ONLY)
                imgui.same_line()
                if imgui.button("Browse##S1YOLOBrowseProcessingMenu"):
                    initial_model_dir = os.path.dirname(
                        self.app.yolo_detection_model_path_setting) if self.app.yolo_detection_model_path_setting and os.path.exists(
                        os.path.dirname(self.app.yolo_detection_model_path_setting)) else None
                    if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(self.app.gui_instance,
                                                                                               'file_dialog'):
                        self.app.gui_instance.file_dialog.show(
                            title="Select Stage 1 YOLO Detection Model", is_save=False,
                            callback=lambda fp: file_mgr._set_yolo_model_path_callback(fp, "detection"),
                            extension_filter="CoreML Model (*.mlpackage),*.mlpackage|ONNX Model (*.onnx),*.onnx|PyTorch Model (*.pt),*.pt|All files (*.*),*.*",
                            initial_path=initial_model_dir
                        )
                    else:
                        self.app.logger.error("Det Model Dialog: File dialog bridge not found.")
                if imgui.is_item_hovered(): imgui.set_tooltip("Path to the YOLO object detection model file.")

                current_pose_path_display = self.app.yolo_pose_model_path_setting if self.app.yolo_pose_model_path_setting else "Not set"
                imgui.input_text("Pose Model##PoseYOLOPathProcessingMenu", current_pose_path_display, 256,
                                 flags=imgui.INPUT_TEXT_READ_ONLY)
                imgui.same_line()
                if imgui.button("Browse##PoseYOLOBrowseProcessingMenu"):
                    initial_model_dir_pose = os.path.dirname(
                        self.app.yolo_pose_model_path_setting) if self.app.yolo_pose_model_path_setting and os.path.exists(
                        os.path.dirname(self.app.yolo_pose_model_path_setting)) else None
                    if hasattr(self.app, 'gui_instance') and self.app.gui_instance and hasattr(self.app.gui_instance,
                                                                                               'file_dialog'):
                        self.app.gui_instance.file_dialog.show(
                            title="Select YOLO Pose Model", is_save=False,
                            callback=lambda fp: file_mgr._set_yolo_model_path_callback(fp, "pose"),
                            extension_filter="CoreML Model (*.mlpackage),*.mlpackage|ONNX Model (*.onnx),*.onnx|PyTorch Model (*.pt),*.pt|All files (*.*),*.*",
                            initial_path=initial_model_dir_pose
                        )
                    else:
                        self.app.logger.error("Pose Model Dialog: File dialog bridge not found.")
                if imgui.is_item_hovered(): imgui.set_tooltip("Path to the YOLO pose estimation model file.")

                if imgui.button("Apply & Check Model Paths##ProcessingMenuAIApplyCheck"):
                    self.app._check_model_paths()
                    self.app.logger.info("Model paths re-checked. Relevant components will use current paths.",
                                         extra={'status_message': True})
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Verify that the specified model paths are valid and accessible.")
                imgui.separator()

                imgui.text("Stage 1 Settings:")
                changed_prod, new_prod_s1_val = imgui.input_int("Num Producers##S1ProcessingMenu",
                                                                stage_proc.num_producers_stage1)
                if changed_prod: stage_proc.num_producers_stage1 = max(1, new_prod_s1_val)
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Number of producer threads for Stage 1 processing (frame decoding/preprocessing).")

                changed_cons, new_cons_s1_val = imgui.input_int("Num Consumers##S1ProcessingMenu",
                                                                stage_proc.num_consumers_stage1)
                if changed_cons: stage_proc.num_consumers_stage1 = max(1, new_cons_s1_val)
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Number of consumer threads for Stage 1 processing (AI inference).")

                imgui.unindent()
                imgui.separator()

                video_loaded = self.app.processor and self.app.processor.is_video_open()
                processing_active = stage_proc.full_analysis_active or stage_proc.scene_detection_active
                if imgui.menu_item("Detect Scenes & Create Chapters", enabled=(video_loaded and not processing_active))[0]:
                    stage_proc.start_scene_detection_analysis()

                imgui.separator()

                if imgui.menu_item("Run Automatic Post-Processing")[0]:
                    if fs_proc and hasattr(fs_proc, 'apply_automatic_post_processing'):
                        fs_proc.apply_automatic_post_processing()
                    else:
                        self.app.logger.warning("Funscript Processor not available to run post-processing.")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Apply the configured automatic post-processing steps (SG, RDP, Clamping, Amplification) to the funscripts.")

                imgui.separator()

                if imgui.begin_menu("Live Tracker Configuration##MenuTrackerSubMenu", True):
                    if imgui.collapsing_header("Live Tracker calibration##LiveTrackerCalibration",
                                               flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                        if not self.app.tracker:
                            imgui.text_disabled("Tracker not initialized.")
                        else:
                            tracker_instance = self.app.tracker
                            if imgui.menu_item("Calibrate Funscript Latency...##MenuCalibTracker")[0]:
                                calibration_mgr.start_latency_calibration()
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "Start the Funscript output latency calibration process.")

                            imgui.text("Funscript Export Compensation:")
                            changed_fod_val, new_fod_val_int = imgui.input_int(
                                "Output Delay Shift (frames)##FunscriptDelayTrackerMenu",
                                calibration_mgr.funscript_output_delay_frames, 1)
                            if changed_fod_val:
                                calibration_mgr.funscript_output_delay_frames = max(0, min(new_fod_val_int, 20))
                                calibration_mgr.update_tracker_delay_params()
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "Adjusts generated Funscript timing (0-20). Positive values make actions happen earlier. Calibrate first.")
                            imgui.separator()


                        if imgui.collapsing_header("Detection & ROI Definition##ROIDetectionTrackerMenu",
                                                   flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                            changed_conf, new_conf_val = imgui.slider_float("Obj. Confidence##ROIConfTrackerMenu",
                                                                            tracker_instance.confidence_threshold, 0.1,
                                                                            0.95, "%.2f")
                            if changed_conf: tracker_instance.confidence_threshold = new_conf_val; self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "Minimum confidence score for an object detection to be considered valid.")

                            changed_pad_val_int, new_pad_val_int_input = imgui.input_int(
                                "ROI Padding##ROIPadTrackerMenu", tracker_instance.roi_padding)
                            if changed_pad_val_int: tracker_instance.roi_padding = max(0,
                                                                                       new_pad_val_int_input); self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "Padding (in pixels) to add around the detected object to define the Region of Interest (ROI).")

                            changed_interval_val_int, new_interval_int_input = imgui.input_int(
                                "ROI Update Interval (frames)##ROIIntervalTrackerMenu",
                                tracker_instance.roi_update_interval)
                            if changed_interval_val_int: tracker_instance.roi_update_interval = max(1,
                                                                                                    new_interval_int_input); self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "How often (in frames) to re-run object detection to update the ROI.")

                            changed_rsf_val_float, new_rsf_float_input = imgui.slider_float(
                                "ROI Smoothing Factor##ROISmoothTrackerMenu", tracker_instance.roi_smoothing_factor,
                                0.0, 1.0, "%.2f")
                            if changed_rsf_val_float: tracker_instance.roi_smoothing_factor = new_rsf_float_input; self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "Smoothing factor for ROI position. Higher = more stable, slower to react.")

                            changed_mfp_val_int, new_mfp_int_input = imgui.input_int(
                                "ROI Persistence (frames)##ROIPersistTrackerMenu",
                                tracker_instance.max_frames_for_roi_persistence)
                            if changed_mfp_val_int: tracker_instance.max_frames_for_roi_persistence = max(0,
                                                                                                          new_mfp_int_input); self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "How many frames ROI stays if primary target lost.")
                            # No separator here, part of the collapsing header content

                        if \
                                imgui.collapsing_header("Optical Flow##ROIFlowTrackerMenu",
                                                        flags=imgui.TREE_NODE_DEFAULT_OPEN)[
                                    0]:
                            changed_usf_val_bool, new_usf_bool_val = imgui.checkbox(
                                "Use Sparse Optical Flow##ROISparseFlowTrackerMenu", tracker_instance.use_sparse_flow)
                            if changed_usf_val_bool: tracker_instance.use_sparse_flow = new_usf_bool_val; self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "Uses Lucas-Kanade (sparse) instead of DIS (dense).")

                            imgui.text("DIS Dense Flow Settings:")
                            dis_controls_disabled_menu = tracker_instance.use_sparse_flow
                            if dis_controls_disabled_menu: imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED,
                                                                                         True); imgui.push_style_var(
                                imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

                            dis_presets_list_menu = ["ULTRAFAST", "FAST", "MEDIUM"]
                            current_dis_preset_idx_menu = dis_presets_list_menu.index(
                                tracker_instance.dis_flow_preset.upper()) if tracker_instance.dis_flow_preset.upper() in dis_presets_list_menu else 0
                            changed_dis_preset_menu, new_dis_preset_idx_menu = imgui.combo(
                                "DIS Preset##ROIDISPresetTrackerMenu", current_dis_preset_idx_menu,
                                dis_presets_list_menu)
                            if changed_dis_preset_menu: tracker_instance.update_dis_flow_config(
                                preset=dis_presets_list_menu[
                                    new_dis_preset_idx_menu]); self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "Preset for DIS optical flow algorithm. Disabled if Sparse Optical Flow is active.")

                            current_dis_finest_scale_menu = tracker_instance.dis_finest_scale if tracker_instance.dis_finest_scale is not None else 0
                            changed_dis_fs_menu, new_dis_fs_val_int = imgui.input_int(
                                "DIS Finest Scale (0-10, 0=auto)##ROIDISFineScaleTrackerMenu",
                                current_dis_finest_scale_menu)
                            if changed_dis_fs_menu: tracker_instance.update_dis_flow_config(
                                finest_scale=new_dis_fs_val_int); self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "Finest scale for DIS optical flow (0 for auto). Disabled if Sparse Optical Flow is active.")

                            if dis_controls_disabled_menu: imgui.pop_style_var(); imgui.internal.pop_item_flag()
                            # No separator here

                        if imgui.collapsing_header("Output Signal Generation##ROISignalTrackerMenu",
                                                   flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                            changed_sens_val_float, new_sens_float_val = imgui.slider_float(
                                "Output Sensitivity##ROISensTrackerMenu", tracker_instance.sensitivity, 0.0, 100.0,
                                "%.1f")
                            if changed_sens_val_float: tracker_instance.sensitivity = new_sens_float_val; self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip("Overall output signal sensitivity (0-100).")

                            changed_baf_val_float, new_baf_float_val = imgui.slider_float(
                                "Base Amplification##ROIBaseAmpTrackerMenu", tracker_instance.base_amplification_factor,
                                0.1, 5.0, "%.2f")
                            if changed_baf_val_float: tracker_instance.base_amplification_factor = max(0.1,
                                                                                                       new_baf_float_val); self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip("General multiplier for output signal.")

                            imgui.text("Class-Specific Amplification Multipliers:")
                            if tracker_instance.class_specific_amplification_multipliers is None: tracker_instance.class_specific_amplification_multipliers = {}
                            curr_face_amp_val_menu = tracker_instance.class_specific_amplification_multipliers.get(
                                "face", 1.0)
                            ch_face_amp_menu, new_face_amp_val_menu = imgui.slider_float(
                                "Face Amp. Mult.##ROIFaceAmpTrackerMenu", curr_face_amp_val_menu, 0.1, 5.0, "%.2f")
                            if ch_face_amp_menu: tracker_instance.class_specific_amplification_multipliers[
                                "face"] = max(0.1, new_face_amp_val_menu); self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "Amplification multiplier for 'face' detections.")

                            curr_hand_amp_val_menu = tracker_instance.class_specific_amplification_multipliers.get(
                                "hand", 1.0)
                            ch_hand_amp_menu, new_hand_amp_val_menu = imgui.slider_float(
                                "Hand Amp. Mult.##ROIHandAmpTrackerMenu", curr_hand_amp_val_menu, 0.1, 5.0, "%.2f")
                            if ch_hand_amp_menu: tracker_instance.class_specific_amplification_multipliers[
                                "hand"] = max(0.1, new_hand_amp_val_menu); self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "Amplification multiplier for 'hand' detections.")
                            imgui.separator()  # Separator within Output Signal

                            ch_afc_val_bool, new_afc_bool_val = imgui.checkbox(
                                "Adaptive Flow Scaling##ROIAdaptiveScaleTrackerMenu",
                                tracker_instance.adaptive_flow_scale)
                            if ch_afc_val_bool: tracker_instance.adaptive_flow_scale = new_afc_bool_val; self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip("Auto-adjusts scaling based on flow range.")

                            ch_fhs_val_int, new_fhs_int_val = imgui.input_int(
                                "Flow Smoothing Window##ROIFlowSmoothWinTrackerMenu",
                                tracker_instance.flow_history_window_smooth)
                            if ch_fhs_val_int: tracker_instance.flow_history_window_smooth = max(1,
                                                                                                 new_fhs_int_val); self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "Frames to average flow (median). Higher = smoother, more lag.")
                            # No separator here

                        if imgui.collapsing_header("Preprocessing##ROIPreprocessingTrackerMenu",
                                                   flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                            tp_w_val_menu, tp_h_val_menu = tracker_instance.target_size_preprocess
                            ch_tpw_menu, new_tpw_val_menu = imgui.input_int("Preproc. Width##ROITPWTrackerMenu",
                                                                            tp_w_val_menu)
                            if ch_tpw_menu: tracker_instance.target_size_preprocess = (max(64, new_tpw_val_menu),
                                                                                       tracker_instance.target_size_preprocess[
                                                                                           1]); self.app.project_manager.project_dirty = True
                            imgui.same_line()
                            ch_tph_menu, new_tph_val_menu = imgui.input_int("Preproc. Height##ROITPHTrackerMenu",
                                                                            tp_h_val_menu)
                            if ch_tph_menu: tracker_instance.target_size_preprocess = (
                                tracker_instance.target_size_preprocess[0],
                                max(64, new_tph_val_menu)); self.app.project_manager.project_dirty = True
                            if imgui.is_item_hovered(): imgui.set_tooltip(
                                "Target width and height for preprocessing image patches fed to optical flow.")
                        # Separators are handled by collapsing_header or manually outside if needed
                    imgui.end_menu()  # End Live Tracker Configuration sub-menu
                imgui.end_menu()  # End Processing Menu

            # --- SETTINGS MENU (Revised) ---
            if imgui.begin_menu("Settings", True):  # Application-wide preferences
                if imgui.collapsing_header("Interface & Performance##SettingsMenuPerfInterface", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    imgui.text("GUI Refresh Rate:")
                    imgui.push_item_width(100)
                    current_normal_fps = int(energy_saver_mgr.main_loop_normal_fps_target)
                    changed_normal_fps, new_normal_fps = imgui.input_int("Normal FPS Target##NormalFPSSettingsMenu",
                                                                         current_normal_fps)
                    if changed_normal_fps:
                        energy_saver_mgr.main_loop_normal_fps_target = max(10, min(300, new_normal_fps))
                    imgui.pop_item_width()
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "Target FPS for normal GUI operation. Higher values consume more resources.")

                    imgui.text("Font Scale")
                    imgui.same_line()
                    imgui.push_item_width(120)

                    font_scale_options_display = ["70%", "80%", "90%", "100%", "110%", "125%", "150%", "175%", "200%"]
                    font_scale_options_values = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]

                    current_scale_val = self.app.app_settings.get("global_font_scale", 1.0)

                    # Find the index of the closest value to the current setting
                    try:
                        current_scale_idx = min(range(len(font_scale_options_values)),
                                                key=lambda i: abs(font_scale_options_values[i] - current_scale_val))
                    except (ValueError, IndexError):
                        current_scale_idx = 3  # Default to 100%

                    changed_font_scale, new_idx = imgui.combo("##GlobalFontScaleSettingsMenu",
                                                              current_scale_idx,
                                                              font_scale_options_display)
                    if changed_font_scale:
                        new_scale_val = font_scale_options_values[new_idx]
                        self.app.app_settings.set("global_font_scale", new_scale_val)
                        energy_saver_mgr.reset_activity_timer()
                    imgui.pop_item_width()
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "Adjust the global font size for the entire application.\nThe change is applied instantly.\nRequires 'Apply & Save All Settings' to persist across sessions.")

                    imgui.separator()

                    imgui.text("Energy Saver Mode:")
                    changed_es_enabled, current_es_enabled_val = imgui.checkbox(
                        "Enable Energy Saver##EnableESSettingsMenu", energy_saver_mgr.energy_saver_enabled)
                    if changed_es_enabled:
                        energy_saver_mgr.energy_saver_enabled = current_es_enabled_val
                        energy_saver_mgr.reset_activity_timer()
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "Enable to reduce CPU/GPU usage after a period of inactivity.")

                    if energy_saver_mgr.energy_saver_enabled:
                        imgui.push_item_width(100)
                        current_es_threshold = int(energy_saver_mgr.energy_saver_threshold_seconds)
                        changed_es_thresh, new_es_threshold = imgui.input_int(
                            "Inactivity Threshold (s)##ESThresholdSettingsMenu", current_es_threshold)
                        if changed_es_thresh:
                            energy_saver_mgr.energy_saver_threshold_seconds = float(max(10, new_es_threshold))
                            energy_saver_mgr.reset_activity_timer()
                        if imgui.is_item_hovered(): imgui.set_tooltip(
                            "Seconds of inactivity before energy saver mode activates.")

                        current_es_fps = int(energy_saver_mgr.energy_saver_fps)
                        changed_es_fps, new_es_fps = imgui.input_int("Energy Saver FPS##ESFPSSettingsMenu",
                                                                     current_es_fps)
                        if changed_es_fps:
                            energy_saver_mgr.energy_saver_fps = max(1, min(30, new_es_fps))
                            energy_saver_mgr.reset_activity_timer()
                        if imgui.is_item_hovered(): imgui.set_tooltip(
                            "Target FPS when energy saver mode is active. Lower values save more power.")
                        imgui.pop_item_width()
                imgui.separator()  # After Interface & Performance

                if imgui.collapsing_header("File & Output##SettingsMenuOutput", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    # Checkbox for auto-saving final funscript
                    c_auto_save_loc, current_auto_save_loc_val = imgui.checkbox(
                        "Autosave final funscript next to video",
                        self.app.app_settings.get("autosave_final_funscript_to_video_location", True)
                    )
                    if c_auto_save_loc:
                        self.app.app_settings.set("autosave_final_funscript_to_video_location",
                                                  current_auto_save_loc_val)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(
                            "If enabled, batch processing and single analysis will save final .funscript files next to the source video.\nIf disabled, they will be saved in the video's subfolder within the main Output Folder.")

                    # Output Folder controls - Corrected
                    imgui.text("Output Folder:")
                    imgui.push_item_width(-1) # Use full available width
                    current_output_folder = self.app.app_settings.get("output_folder_path", "output")
                    changed_output_folder, new_output_folder = imgui.input_text("##OutputFolderSettingsMenu",
                                                                                current_output_folder, 256)
                    if changed_output_folder:
                        self.app.app_settings.set("output_folder_path", new_output_folder)
                    imgui.pop_item_width()
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("All generated files (msgpack, project files, etc.) will be saved in a subfolder here, named after the video.\nManually enter a relative (e.g. 'output') or absolute path.")

                imgui.separator()

                if imgui.collapsing_header("Logging##SettingsMenuLogging", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    logging_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                    try:
                        current_log_level_idx = logging_levels.index(self.app.logging_level_setting.upper())
                    except ValueError:
                        current_log_level_idx = logging_levels.index("INFO")

                    imgui.push_item_width(150)
                    changed_log_level, new_log_level_idx = imgui.combo("Logging Level##LoggingMenuSettingsCombo",
                                                                       current_log_level_idx, logging_levels)
                    imgui.pop_item_width()
                    if changed_log_level:
                        new_level_str = logging_levels[new_log_level_idx]
                        self.app.set_application_logging_level(new_level_str)
                        energy_saver_mgr.reset_activity_timer()
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "Set the application's logging verbosity. Requires 'Apply & Save All Settings' to persist.")
                imgui.separator()  # After Logging

                if imgui.collapsing_header("Autosave##SettingsMenuAutosave", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    c_auto_enable_menu, current_autosave_enabled_menu = imgui.checkbox(
                        "Enable Autosave##MenuSettingsAutosaveEnable",
                        self.app.app_settings.get("autosave_enabled", True))
                    if c_auto_enable_menu: self.app.app_settings.set("autosave_enabled", current_autosave_enabled_menu)
                    if imgui.is_item_hovered(): imgui.set_tooltip(
                        "Automatically save project state at intervals and on exit.")

                    imgui.push_item_width(100)
                    current_autosave_interval_val_menu = self.app.app_settings.get("autosave_interval_seconds", 300)
                    c_interval_val_menu, new_autosave_interval_menu = imgui.input_int(
                        "Interval (s)##autosaveMenuSettingsInterval", current_autosave_interval_val_menu)
                    if c_interval_val_menu: self.app.app_settings.set("autosave_interval_seconds",
                                                                      max(30, new_autosave_interval_menu))
                    imgui.pop_item_width()
                    if imgui.is_item_hovered(): imgui.set_tooltip("Time in seconds between automatic project saves.")
                imgui.separator()  # After Autosave

                if imgui.collapsing_header("View/Edit Hotkeys##FSHotkeysMenuSettingsDetail",
                                           flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    shortcuts_settings = self.app.app_settings.get("funscript_editor_shortcuts", {})
                    for action_name, key_str in list(shortcuts_settings.items()):
                        action_display_name = action_name.replace('_', ' ').title()
                        imgui.text(f"{action_display_name}: ")
                        imgui.same_line()
                        current_display_key = key_str
                        button_text_hotkey = "Record"
                        if self.app.shortcut_manager.is_recording_shortcut_for == action_name:
                            current_display_key = "PRESS KEY..."
                            button_text_hotkey = "Cancel"
                        imgui.text_colored(current_display_key, 0.2, 0.8, 1.0, 1.0)
                        imgui.same_line()
                        if imgui.button(f"{button_text_hotkey}##record_btn_menu_settings_{action_name}"):
                            if self.app.shortcut_manager.is_recording_shortcut_for == action_name:
                                self.app.shortcut_manager.cancel_shortcut_recording()
                            else:
                                self.app.shortcut_manager.start_shortcut_recording(action_name)
                        if imgui.is_item_hovered(): imgui.set_tooltip(
                            f"Click 'Record' then press desired key combination for '{action_display_name}'.")
                imgui.separator()  # After Hotkeys

                if imgui.button("Apply & Save All Application Settings##MenuSettingsSaveAll"):
                    self.app._check_model_paths()  # Still good to check before save
                    self.app.save_app_settings()
                if imgui.is_item_hovered(): imgui.set_tooltip(
                    "Apply and save all application settings (interface, performance, logging, autosave, hotkeys, core processing paths).")
                imgui.end_menu()

            # Global status message display
            if app_state.status_message and time.time() < app_state.status_message_time:
                text_size_status = imgui.calc_text_size(app_state.status_message)
                menu_bar_width = imgui.get_window_width()
                cursor_x_after_menus = imgui.get_cursor_pos_x()
                padding_needed = menu_bar_width - cursor_x_after_menus - text_size_status[0] - \
                                 imgui.get_style().item_spacing[0] * 2
                if padding_needed > 0:
                    imgui.same_line(cursor_x_after_menus + padding_needed)
                    imgui.text_colored(app_state.status_message, 0.9, 0.9, 0.3, 1.0)
                elif text_size_status[0] < (menu_bar_width - cursor_x_after_menus):
                    imgui.same_line(menu_bar_width - text_size_status[0] - imgui.get_style().item_spacing[0] * 4)
                    imgui.text_colored(app_state.status_message, 0.9, 0.9, 0.3, 1.0)
            elif app_state.status_message:
                app_state.status_message = ""

            imgui.end_main_menu_bar()
