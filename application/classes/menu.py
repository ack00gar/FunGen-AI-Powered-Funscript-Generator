import imgui
import os
import time
import glfw


class MainMenu:
    def __init__(self, app_instance):
        self.app = app_instance

    def _render_timeline_selection_popup(self):
        """Renders a popup to select the reference timeline for comparison."""
        app_state = self.app.app_state_ui
        if not app_state.show_timeline_selection_popup:
            return

        imgui.open_popup("Select Reference Timeline##TimelineSelectPopup")

        main_viewport = imgui.get_main_viewport()
        popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 450) * 0.5
        popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 220) * 0.5
        imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
        imgui.set_next_window_size(450, 0, condition=imgui.APPEARING)

        if imgui.begin_popup_modal("Select Reference Timeline##TimelineSelectPopup", True,
                                   flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text("Which timeline has the correct timing?")
            imgui.text_wrapped("The offset will be calculated for the other timeline and applied to it.")
            imgui.separator()

            # --- FIX: Assign the boolean return value to a single variable ---
            if imgui.radio_button("Timeline 1 is the Reference", app_state.timeline_comparison_reference_num == 1):
                app_state.timeline_comparison_reference_num = 1

            if imgui.radio_button("Timeline 2 is the Reference", app_state.timeline_comparison_reference_num == 2):
                app_state.timeline_comparison_reference_num = 2
            # --- END FIX ---

            imgui.separator()

            if imgui.button("Compare", width=120):
                self.app.run_and_display_comparison_results(app_state.timeline_comparison_reference_num)
                app_state.show_timeline_selection_popup = False
                imgui.close_current_popup()

            imgui.same_line()

            if imgui.button("Cancel", width=120):
                app_state.show_timeline_selection_popup = False
                imgui.close_current_popup()

            imgui.end_popup()

    def _render_timeline_comparison_results_popup(self):
        """Renders the popup showing the comparison results."""
        app_state = self.app.app_state_ui
        if not app_state.show_timeline_comparison_results_popup:
            return

        imgui.open_popup("Timeline Comparison Results##TimelineResultsPopup")

        main_viewport = imgui.get_main_viewport()
        popup_pos_x = main_viewport.pos[0] + (main_viewport.size[0] - 400) * 0.5
        popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 240) * 0.5  # Increased height slightly
        imgui.set_next_window_position(popup_pos_x, popup_pos_y, condition=imgui.APPEARING)
        imgui.set_next_window_size(400, 0, condition=imgui.APPEARING)

        if imgui.begin_popup_modal("Timeline Comparison Results##TimelineResultsPopup", True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            results = app_state.timeline_comparison_results
            if results:
                offset_ms = results.get("calculated_offset_ms", 0)
                ref_strokes = results.get("ref_stroke_count", 0)
                target_strokes = results.get("target_stroke_count", 0)
                target_timeline_num = results.get("target_timeline_num", "N/A")
                ref_timeline_num = 1 if target_timeline_num == 2 else 2

                # --- Get FPS and calculate frame offset ---
                fps = 0
                if self.app.processor and self.app.processor.fps > 0:
                    fps = self.app.processor.fps

                frame_offset_str = ""
                if fps > 0:
                    frame_offset = round((offset_ms / 1000.0) * fps)
                    frame_offset_str = f" (approx. {frame_offset:.0f} frames)"

                imgui.text(f"Reference: Timeline {ref_timeline_num} ({ref_strokes} strokes)")
                imgui.text(f"Target:    Timeline {target_timeline_num} ({target_strokes} strokes)")
                imgui.separator()

                imgui.text_wrapped(
                    f"The Target (T{target_timeline_num}) appears to be delayed relative to the Reference (T{ref_timeline_num}) by:")
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 0.0, 1.0)
                imgui.text(f"  {offset_ms} milliseconds{frame_offset_str}")
                imgui.pop_style_color()

                imgui.separator()

                if imgui.button(f"Apply Offset to Timeline {target_timeline_num}", width=-1):
                    fs_proc = self.app.funscript_processor
                    op_desc = f"Apply Timeline Offset ({offset_ms}ms)"

                    fs_proc._record_timeline_action(target_timeline_num, op_desc)
                    funscript_obj, axis_name = fs_proc._get_target_funscript_object_and_axis(target_timeline_num)

                    if funscript_obj and axis_name:
                        # Apply a NEGATIVE offset to shift the target script EARLIER to match the reference
                        funscript_obj.shift_points_time(axis=axis_name, time_delta_ms=-offset_ms)
                        fs_proc._finalize_action_and_update_ui(target_timeline_num, op_desc)
                        self.app.logger.info(f"Applied {offset_ms}ms offset to Timeline {target_timeline_num}.", extra={'status_message': True})

                    app_state.show_timeline_comparison_results_popup = False
                    imgui.close_current_popup()

            if imgui.button("Close", width=-1):
                app_state.show_timeline_comparison_results_popup = False
                imgui.close_current_popup()

            imgui.end_popup()

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

                # --- Open Recent Sub-Menu ---
                recent_projects = self.app.app_settings.get("recent_projects", [])
                can_open_recent = bool(recent_projects)

                if imgui.begin_menu("Open Recent", enabled=can_open_recent):
                    if not recent_projects:
                        imgui.menu_item("(No recent projects)", enabled=False)
                    else:
                        for project_path in recent_projects:
                            # Display a shorter, more readable version of the path
                            try:
                                # e.g., "my_video/my_video.fgn"
                                display_name = f"{os.path.basename(os.path.dirname(project_path))}{os.sep}{os.path.basename(project_path)}"
                            except Exception:
                                display_name = project_path  # Fallback

                            if imgui.menu_item(display_name)[0]:
                                self.app.project_manager.load_project(project_path)
                            if imgui.is_item_hovered():
                                imgui.set_tooltip(project_path)  # Show full path on hover
                    imgui.end_menu()

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

                imgui.text("UI Mode")
                imgui.indent()
                is_expert_mode = app_state.ui_view_mode == 'expert'
                if imgui.radio_button("Expert Mode##UIModeExpert", is_expert_mode):
                    if not is_expert_mode:
                        app_state.ui_view_mode = 'expert'
                        self.app.app_settings.set("ui_view_mode", "expert")

                is_simple_mode = app_state.ui_view_mode == 'simple'
                if imgui.radio_button("Simple Mode##UIModeSimple", is_simple_mode):
                    if not is_simple_mode:
                        app_state.ui_view_mode = 'simple'
                        self.app.app_settings.set("ui_view_mode", "simple")
                imgui.unindent()
                imgui.separator()

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
                    clicked, app_state.show_control_panel_window = imgui.menu_item("Control Panel", selected=app_state.show_control_panel_window)
                    if clicked: self.app.project_manager.project_dirty = True
                    clicked, app_state.show_info_graphs_window = imgui.menu_item("Info & Graphs", selected=app_state.show_info_graphs_window)
                    if clicked: self.app.project_manager.project_dirty = True
                    clicked, app_state.show_video_display_window = imgui.menu_item("Video Display", selected=app_state.show_video_display_window)
                    if clicked: self.app.project_manager.project_dirty = True
                    clicked, app_state.show_video_navigation_window = imgui.menu_item("Video Navigation", selected=app_state.show_video_navigation_window)
                    if clicked: self.app.project_manager.project_dirty = True
                else:
                    imgui.text_disabled("Window toggles are for Floating Mode.")
                imgui.unindent()
                imgui.separator()

                imgui.text("Timeline Editors & Previews")
                imgui.indent()
                clicked, app_state.show_funscript_interactive_timeline = imgui.menu_item("Interactive Timeline 1", selected=app_state.show_funscript_interactive_timeline)
                if clicked: self.app.project_manager.project_dirty = True
                clicked, app_state.show_funscript_interactive_timeline2 = imgui.menu_item("Interactive Timeline 2", selected=app_state.show_funscript_interactive_timeline2)
                if clicked: self.app.project_manager.project_dirty = True
                imgui.separator()
                clicked, app_state.show_funscript_timeline = imgui.menu_item("Funscript Preview Bar", selected=app_state.show_funscript_timeline)
                if clicked: self.app.project_manager.project_dirty = True
                clicked, app_state.show_heatmap = imgui.menu_item("Heatmap", selected=app_state.show_heatmap)
                if clicked: self.app.project_manager.project_dirty = True
                imgui.unindent()
                imgui.separator()

                imgui.text("Overlays & Aux Windows")
                imgui.indent()

                # Gauge T1
                clicked, current_val_t1 = imgui.menu_item("Script Gauge (Timeline 1)", selected=app_state.show_gauge_window_timeline1)
                if clicked:
                    app_state.show_gauge_window_timeline1 = current_val_t1
                    self.app.project_manager.project_dirty = True

                # Gauge T2
                clicked, current_val_t2 = imgui.menu_item("Script Gauge (Timeline 2)", selected=app_state.show_gauge_window_timeline2)
                if clicked:
                    app_state.show_gauge_window_timeline2 = current_val_t2
                    self.app.project_manager.project_dirty = True

                clicked, app_state.show_lr_dial_graph = imgui.menu_item("L/R Dial Graph", selected=app_state.show_lr_dial_graph)
                if clicked: self.app.project_manager.project_dirty = True

                # Added defensive hasattr check
                if not hasattr(app_state, 'show_chapter_list_window'):
                    app_state.show_chapter_list_window = False
                clicked, app_state.show_chapter_list_window = imgui.menu_item("Chapter List", selected=app_state.show_chapter_list_window)
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

                # --- Autotuner Tool ---
                if not hasattr(app_state, 'show_autotuner_window'):
                    app_state.show_autotuner_window = False
                clicked, selected = imgui.menu_item("Performance Autotuner...", selected=app_state.show_autotuner_window)
                if clicked:
                    app_state.show_autotuner_window = not app_state.show_autotuner_window

                # --- Timeline Comparison Tool ---
                # The menu item is enabled only if both timelines have actions to compare.
                can_compare = (
                    hasattr(self.app, 'funscript_processor') and
                    self.app.funscript_processor.get_actions('primary') and
                    self.app.funscript_processor.get_actions('secondary')
                )
                if imgui.menu_item("Compare Timelines...", enabled=can_compare)[0]:
                    if hasattr(self.app, 'trigger_timeline_comparison'):
                        self.app.trigger_timeline_comparison()
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Compares the signals on Timeline 1 and Timeline 2 to calculate the optimal time offset.")

                # --- TensorRT Compiler Tool ---
                if not hasattr(self.app, 'tensorrt_compiler_window'):
                    self.app.tensorrt_compiler_window = None
                if imgui.menu_item("Compile YOLO Model to TensorRT (.engine)...")[0]:
                    from application.gui_components.tensorrt_compiler_window import TensorRTCompilerWindow
                    def on_close():
                        self.app.tensorrt_compiler_window = None
                    self.app.tensorrt_compiler_window = TensorRTCompilerWindow(self.app, on_close_callback=on_close)


                # --- Update Controls Sub-Menu ---
                if imgui.begin_menu("Update Settings..."):
                    # Toggle for checking on startup
                    check_startup = self.app.app_settings.get("updater_check_on_startup", True)
                    clicked, new_val_startup = imgui.menu_item("Check for Updates on Startup", selected=check_startup)
                    if clicked:
                        self.app.app_settings.set("updater_check_on_startup", new_val_startup)

                    # Toggle for periodic background checks
                    check_periodic = self.app.app_settings.get("updater_check_periodically", True)
                    clicked, new_val_periodic = imgui.menu_item("Check Periodically in Background (Hourly)", selected=check_periodic)
                    if clicked:
                        self.app.app_settings.set("updater_check_periodically", new_val_periodic)

                    # Toggle for suppressing the update popup
                    suppress_popup = self.app.app_settings.get("updater_suppress_popup", False)
                    clicked, new_val_suppress = imgui.menu_item("Suppress Update Notification Popup", selected=suppress_popup)
                    if clicked:
                        self.app.app_settings.set("updater_suppress_popup", new_val_suppress)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("If suppressed, only the menu bar indicator will be shown.")

                    imgui.end_menu()

                # Manual trigger to apply a pending update
                can_apply_update = self.app.updater.update_available and not self.app.updater.update_in_progress
                if imgui.menu_item("Apply Pending Update...", enabled=can_apply_update)[0]:
                    self.app.updater.show_update_dialog = True  # Re-opens the confirmation dialog
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Shows the update dialog if an update has been detected.")

                imgui.end_menu()

            # --- UPDATE INDICATOR ---
            # This is now a non-interactive, colored text label placed after the menus.
            if self.app.updater.update_available and not self.app.updater.update_in_progress:
                imgui.same_line()
                imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + 15) # Add some padding
                rect_min = imgui.get_item_rect_min()
                rect_max = imgui.get_item_rect_max()
                if imgui.is_mouse_hovering_rect(rect_min[0], rect_min[1], rect_max[0], rect_max[1]):
                    imgui.set_tooltip("A new version is available! Find options in the Tools menu.")

                # Make the text clickable to re-open the dialog
                if imgui.button("Update Available!"):
                    self.app.updater.show_update_dialog = True
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Click to see update details and apply.")

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

            # --- RENDER POPUPS MANAGED BY THIS CLASS ---
            self._render_timeline_selection_popup()
            self._render_timeline_comparison_results_popup()
