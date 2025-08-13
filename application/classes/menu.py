import os
import imgui
from config.element_group_colors import MenuColors

def _center_popup(width, height):
    mv = imgui.get_main_viewport()
    # Avoid tuple creation and repeated attr lookups
    main_viewport_pos_x, main_viewport_pos_y = mv.pos[0], mv.pos[1]
    main_viewport_w, main_viewport_h = mv.size[0], mv.size[1]
    pos_x = main_viewport_pos_x + (main_viewport_w - width) * 0.5
    pos_y = main_viewport_pos_y + (main_viewport_h - height) * 0.5
    imgui.set_next_window_position(pos_x, pos_y, condition=imgui.APPEARING)
    # height=0 -> auto-resize vertical; we still set width for consistent centering
    imgui.set_next_window_size(width, 0, condition=imgui.APPEARING)

def _begin_modal_popup(name, width, height):
    imgui.open_popup(name)
    _center_popup(width, height)
    opened, _ = imgui.begin_popup_modal(
        name, True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE
    )
    return opened

def _menu_item_simple(label, enabled=True):
    clicked, _ = imgui.menu_item(label, enabled=enabled)
    return clicked

def _radio_line(label, is_selected):
    # Cheaper than f-strings in hot loops
    if imgui.radio_button(label, is_selected):
        return True
    return False

class MainMenu:
    __slots__ = ("app", "FRAME_OFFSET")

    def __init__(self, app_instance):
        self.app = app_instance
        self.FRAME_OFFSET = MenuColors.FRAME_OFFSET

    # ------------------------- POPUPS -------------------------

    def _render_timeline_selection_popup(self):
        app = self.app
        app_state = app.app_state_ui
        if not app_state.show_timeline_selection_popup:
            return

        name = "Select Reference Timeline##TimelineSelectPopup"
        # 450x220 used for centering; height auto
        if _begin_modal_popup(name, 450, 220):
            imgui.text("Which timeline has the correct timing?")
            imgui.text_wrapped(
                "The offset will be calculated for the other timeline "
                "and applied to it."
            )
            imgui.separator()

            ref_num = app_state.timeline_comparison_reference_num
            # Fixed range (1..2)
            if _radio_line("Timeline 1 is the Reference", ref_num == 1):
                app_state.timeline_comparison_reference_num = 1
            if _radio_line("Timeline 2 is the Reference", ref_num == 2):
                app_state.timeline_comparison_reference_num = 2
            imgui.separator()

            if imgui.button("Compare", width=120):
                app.run_and_display_comparison_results(
                    app_state.timeline_comparison_reference_num
                )
                app_state.show_timeline_selection_popup = False
                imgui.close_current_popup()

            imgui.same_line()
            if imgui.button("Cancel", width=120):
                app_state.show_timeline_selection_popup = False
                imgui.close_current_popup()
            imgui.end_popup()

    def _render_timeline_comparison_results_popup(self):
        app = self.app
        app_state = app.app_state_ui
        if not app_state.show_timeline_comparison_results_popup:
            return

        name = "Timeline Comparison Results##TimelineResultsPopup"
        if _begin_modal_popup(name, 400, 240): # 400x240 used for centering; height auto
            results = app_state.timeline_comparison_results
            if results:
                # Localize lookups
                offset_ms = results.get("calculated_offset_ms", 0)
                target_num = results.get("target_timeline_num", "N/A")
                ref_strokes = results.get("ref_stroke_count", 0)
                target_strokes = results.get("target_stroke_count", 0)
                ref_num = 1 if target_num == 2 else 2

                fps = 0
                processor = app.processor
                if processor:
                    fps = processor.fps
                    if fps > 0:
                        # Use int conversion instead of round+format for speed
                        frames = int((offset_ms / 1000.0) * fps + 0.5)
                        frame_suffix = " (approx. %d frames)" % frames
                    else:
                        frame_suffix = ""

                imgui.text("Reference: Timeline %d (%d strokes)" % (ref_num, ref_strokes))
                imgui.text("Target:    Timeline %s (%d strokes)" % (str(target_num), target_strokes))
                imgui.separator()

                imgui.text_wrapped(
                    "The Target (T%s) appears to be delayed relative to the "
                    "Reference (T%d) by:" % (str(target_num), ref_num)
                )
                imgui.push_style_color(imgui.COLOR_TEXT, *self.FRAME_OFFSET)
                imgui.text("  %d milliseconds%s" % (offset_ms, frame_suffix))
                imgui.pop_style_color()
                imgui.separator()

                if imgui.button(
                    "Apply Offset to Timeline %s" % str(target_num), width=-1
                ):
                    fs_proc = app.funscript_processor
                    op_desc = "Apply Timeline Offset (%dms)" % offset_ms

                    fs_proc._record_timeline_action(target_num, op_desc)
                    funscript_obj, axis_name = fs_proc._get_target_funscript_object_and_axis(
                        target_num
                    )

                    if funscript_obj and axis_name:
                        # Negative => shift earlier to match reference
                        funscript_obj.shift_points_time(axis=axis_name, time_delta_ms=-offset_ms)
                        fs_proc._finalize_action_and_update_ui(target_num, op_desc)
                        app.logger.info(
                            "Applied %dms offset to Timeline %s." % (offset_ms, str(target_num)),
                            extra={"status_message": True},
                        )

                    app_state.show_timeline_comparison_results_popup = False
                    imgui.close_current_popup()

            if imgui.button("Close", width=-1):
                app_state.show_timeline_comparison_results_popup = False
                imgui.close_current_popup()
            imgui.end_popup()

    # ------------------------- MAIN RENDER -------------------------

    def render(self):
        app = self.app
        app_state = app.app_state_ui
        file_mgr = app.file_manager
        stage_proc = app.stage_processor

        if imgui.begin_main_menu_bar():
            self._render_file_menu(app_state, file_mgr)
            self._render_edit_menu(app_state)
            self._render_view_menu(app_state, stage_proc)
            self._render_tools_menu(app_state, file_mgr)
            self._render_ai_menu()
            self._render_updates_menu()
            imgui.end_main_menu_bar()

        self._render_timeline_selection_popup()
        self._render_timeline_comparison_results_popup()

    # ------------------------- MENUS -------------------------

    def _render_file_menu(self, app_state, file_mgr):
        app = self.app
        pm = app.project_manager
        settings = app.app_settings
        fm = app.file_manager

        if imgui.begin_menu("File", True):
            # New/Project/Video/Open
            if _menu_item_simple("New Project"):
                app.reset_project_state(for_new_project=True)
                pm.project_dirty = True

            if _menu_item_simple("Project..."):
                pm.open_project_dialog()

            if _menu_item_simple("Video..."):
                fm.open_video_dialog()

            if _menu_item_simple("Close Project"):
                app.reset_project_state(for_new_project=True)

            # Open Recent
            recent = settings.get("recent_projects", [])
            can_open_recent = bool(recent)

            if imgui.begin_menu("Open Recent", enabled=can_open_recent):
                if _menu_item_simple("Clear Menu"):
                    settings.set("recent_projects", [])
                if recent:
                    imgui.separator()
                    for project_path in recent:
                        display_name = os.path.basename(project_path)
                        if _menu_item_simple(display_name):
                            pm.load_project(project_path)
                imgui.end_menu()
            imgui.separator()

            # Save options
            can_save = pm.project_file_path is not None

            if _menu_item_simple("Save Project", enabled=can_save):
                pm.save_project_dialog()
            if _menu_item_simple("Save Project As...", enabled=True):
                pm.save_project_dialog(save_as=True)
            imgui.separator()

            # Import/Export
            if imgui.begin_menu("Import..."):
                if _menu_item_simple("Funscript to Timeline 1..."):
                    fm.import_funscript_to_timeline(1)
                if _menu_item_simple("Funscript to Timeline 2..."):
                    fm.import_funscript_to_timeline(2)
                if _menu_item_simple("Stage 2 Overlay Data..."):
                    fm.import_stage2_overlay_data()
                imgui.end_menu()

            if imgui.begin_menu("Export..."):
                if _menu_item_simple("Funscript from Timeline 1..."):
                    if self.app.gui_instance and self.app.gui_instance.file_dialog:
                        video_path = self.app.file_manager.video_path
                        output_folder_base = self.app.app_settings.get("output_folder_path", "output")
                        initial_path = output_folder_base
                        initial_filename = "timeline1.funscript"

                        if video_path:
                            video_basename = os.path.splitext(os.path.basename(video_path))[0]
                            initial_path = os.path.join(output_folder_base, video_basename)
                            initial_filename = f"{video_basename}.funscript"

                        if not os.path.isdir(initial_path):
                            os.makedirs(initial_path, exist_ok=True)

                        self.app.gui_instance.file_dialog.show(
                            is_save=True,
                            title="Export Funscript from Timeline 1",
                            extension_filter="Funscript Files (*.funscript),*.funscript",
                            callback=lambda filepath: fm.save_funscript_from_timeline(filepath, 1),
                            initial_path=initial_path,
                            initial_filename=initial_filename
                        )
                if _menu_item_simple("Funscript from Timeline 2..."):
                    if self.app.gui_instance and self.app.gui_instance.file_dialog:
                        video_path = self.app.file_manager.video_path
                        output_folder_base = self.app.app_settings.get("output_folder_path", "output")
                        initial_path = output_folder_base
                        initial_filename = "timeline2.funscript"

                        if video_path:
                            video_basename = os.path.splitext(os.path.basename(video_path))[0]
                            initial_path = os.path.join(output_folder_base, video_basename)
                            initial_filename = f"{video_basename}_t2.funscript"

                        if not os.path.isdir(initial_path):
                            os.makedirs(initial_path, exist_ok=True)

                        self.app.gui_instance.file_dialog.show(
                            is_save=True,
                            title="Export Funscript from Timeline 2",
                            extension_filter="Funscript Files (*.funscript),*.funscript",
                            callback=lambda filepath: fm.save_funscript_from_timeline(filepath, 2),
                            initial_path=initial_path,
                            initial_filename=initial_filename
                        )
                imgui.end_menu()
            imgui.separator()

            if _menu_item_simple("Exit"):
                app.shutdown_app()
                # Close the GLFW window to exit the application
                if app.gui_instance and app.gui_instance.window:
                    import glfw
                    glfw.set_window_should_close(app.gui_instance.window, True)
            imgui.end_menu()

    def _render_edit_menu(self, app_state):
        app = self.app
        fs_proc = app.funscript_processor

        if imgui.begin_menu("Edit", True):
            # T1
            undo1 = fs_proc._get_undo_manager(1)
            can_undo1 = undo1.can_undo() if undo1 else False
            can_redo1 = undo1.can_redo() if undo1 else False
            if imgui.menu_item(
                "Undo T1 Change", "Ctrl+Z", selected=False, enabled=can_undo1
            )[0]:
                fs_proc.perform_undo_redo(1, "undo")
            if imgui.menu_item(
                "Redo T1 Change", "Ctrl+Y", selected=False, enabled=can_redo1
            )[0]:
                fs_proc.perform_undo_redo(1, "redo")
            imgui.separator()

            # T2
            undo2 = fs_proc._get_undo_manager(2)
            can_undo2 = undo2.can_undo() if undo2 else False
            can_redo2 = undo2.can_redo() if undo2 else False
            if imgui.menu_item(
                "Undo T2 Change", "Alt+Ctrl+Z", selected=False, enabled=can_undo2
            )[0]:
                fs_proc.perform_undo_redo(2, "undo")
            if imgui.menu_item(
                "Redo T2 Change", "Alt+Ctrl+Y", selected=False, enabled=can_redo2
            )[0]:
                fs_proc.perform_undo_redo(2, "redo")
            imgui.end_menu()

    def _render_view_menu(self, app_state, stage_proc):
        if imgui.begin_menu("View", True):
            self._render_ui_mode_section(app_state)
            imgui.separator()
            self._render_ui_layout_section(app_state)
            imgui.separator()
            self._render_panels_submenu(app_state)
            imgui.separator()
            self._render_layout_options_section(app_state)
            imgui.separator()
            self._render_timeline_editors_section(app_state)
            imgui.separator()
            self._render_overlays_section(app_state, stage_proc)
            imgui.end_menu()

    def _render_ui_mode_section(self, app_state):
        settings = self.app.app_settings
        imgui.text("UI Mode")
        imgui.indent()

        current = app_state.ui_view_mode
        if _radio_line("Expert Mode##UIModeExpertMode", current == "expert"):
            if current != "expert":
                app_state.ui_view_mode = "expert"
                settings.set("ui_view_mode", "expert")
        if _radio_line("Simple Mode##UIModeSimpleMode", current == "simple"):
            if current != "simple":
                app_state.ui_view_mode = "simple"
                settings.set("ui_view_mode", "simple")
        imgui.unindent()

    def _render_ui_layout_section(self, app_state):
        imgui.text("UI Layout Mode")
        imgui.indent()

        current = app_state.ui_layout_mode
        if _radio_line("Fixed Panels##UILayoutModeFixedPanels", current == "fixed"):
            if current != "fixed":
                app_state.ui_layout_mode = "fixed"
                self.app.project_manager.project_dirty = True

        if _radio_line("Floating Windows##UILayoutModeFloatingWindows", current == "floating"):
            if current != "floating":
                app_state.ui_layout_mode = "floating"
                app_state.just_switched_to_floating = True
                self.app.project_manager.project_dirty = True
        imgui.unindent()

    def _render_panels_submenu(self, app_state):
        pm = self.app.project_manager
        is_floating = app_state.ui_layout_mode == "floating"

        if imgui.begin_menu("Panels", enabled=is_floating):
            # Using getattr/setattr has minor overhead; acceptable given low count.
            for label, attr in (
                ("Control Panel", "show_control_panel_window"),
                ("Info & Graphs", "show_info_graphs_window"),
                ("Video Display", "show_video_display_window"),
                ("Video Navigation", "show_video_navigation_window"),
            ):
                cur = getattr(app_state, attr)
                clicked, val = imgui.menu_item(label, selected=cur)
                if clicked:
                    setattr(app_state, attr, val)
                    pm.project_dirty = True
            imgui.end_menu()

        # Tooltip only computed if hovered
        if imgui.is_item_hovered() and not is_floating:
            imgui.set_tooltip("Window toggles are for floating mode.")

    def _render_layout_options_section(self, app_state):
        pm = self.app.project_manager
        imgui.text("Layout Options")
        imgui.indent()

        if not hasattr(app_state, "full_width_nav"):
            app_state.full_width_nav = False

        is_fixed = app_state.ui_layout_mode == "fixed"
        clicked, selected = imgui.menu_item(
            "Full-Width Navigation Bar",
            selected=app_state.full_width_nav,
            enabled=is_fixed,
        )
        if clicked:
            app_state.full_width_nav = selected
            pm.project_dirty = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Only available in 'Fixed Panels' layout mode.")
        imgui.unindent()

    def _render_timeline_editors_section(self, app_state):
        app = self.app
        pm = app.project_manager
        settings = app.app_settings

        imgui.text("Timeline Editors & Previews")
        imgui.indent()

        # Editors
        for label, attr in (
            ("Interactive Timeline 1", "show_funscript_interactive_timeline"),
            ("Interactive Timeline 2", "show_funscript_interactive_timeline2"),
        ):
            cur = getattr(app_state, attr)
            clicked, val = imgui.menu_item(label, selected=cur)
            if clicked:
                setattr(app_state, attr, val)
                pm.project_dirty = True
        imgui.separator()

        # Previews
        for label, attr in (
            ("Funscript Preview Bar", "show_funscript_timeline"),
            ("Heatmap", "show_heatmap"),
        ):
            cur = getattr(app_state, attr)
            clicked, val = imgui.menu_item(label, selected=cur)
            if clicked:
                setattr(app_state, attr, val)
                pm.project_dirty = True

        use_simplified = settings.get("use_simplified_funscript_preview", False)
        clicked, new_val = imgui.menu_item(
            "Use Simplified Funscript Preview", selected=use_simplified
        )
        if clicked and new_val != use_simplified:
            settings.set("use_simplified_funscript_preview", new_val)
            app.app_state_ui.funscript_preview_dirty = True
        imgui.separator()

        for label, attr, key in (
            ("Show Timeline Editor Buttons", "show_timeline_editor_buttons", "show_timeline_editor_buttons"),
            ("Show Advanced Options", "show_advanced_options", "show_advanced_options"),
        ):
            cur = getattr(app_state, attr)
            clicked, val = imgui.menu_item(label, selected=cur)
            if clicked:
                setattr(app_state, attr, val)
                settings.set(key, val)
                pm.project_dirty = True
        imgui.unindent()

    def _render_overlays_section(self, app_state, stage_proc):
        pm = self.app.project_manager
        app = self.app

        imgui.text("Overlays & Aux Windows")
        imgui.indent()

        for label, attr in (
            ("Script Gauge (Timeline 1)", "show_gauge_window_timeline1"),
            ("Script Gauge (Timeline 2)", "show_gauge_window_timeline2"),
        ):
            cur = getattr(app_state, attr)
            clicked, val = imgui.menu_item(label, selected=cur)
            if clicked:
                setattr(app_state, attr, val)
                pm.project_dirty = True

        label = "L/R Dial Graph"
        cur = getattr(app_state, "show_lr_dial_graph")
        clicked, val = imgui.menu_item(label, selected=cur)
        if clicked:
            setattr(app_state, "show_lr_dial_graph", val)
            pm.project_dirty = True

        if not hasattr(app_state, "show_chapter_list_window"):
            app_state.show_chapter_list_window = False

        clicked, val = imgui.menu_item(
            "Chapter List", selected=app_state.show_chapter_list_window
        )
        if clicked:
            app_state.show_chapter_list_window = val
            pm.project_dirty = True
        imgui.separator()

        # Tracker flags
        tracker = app.tracker
        if tracker:
            for label, flag in (
                ("Show Detections/Masks", "ui_show_masks"),
                ("Show Optical Flow", "ui_show_flow"),
            ):
                cur = getattr(app_state, flag)
                clicked, val = imgui.menu_item(label, selected=cur)
                if clicked:
                    app_state.set_tracker_ui_flag(flag.replace("ui_", ""), val)

        can_show_s2 = stage_proc.stage2_overlay_data is not None
        clicked, val = imgui.menu_item(
            "Show Stage 2 Overlay",
            selected=app_state.show_stage2_overlay,
            enabled=can_show_s2,
        )
        if clicked:
            app_state.show_stage2_overlay = val
            pm.project_dirty = True

        clicked, _ = imgui.menu_item(
            "Audio Waveform", selected=app_state.show_audio_waveform
        )
        if clicked:
            app.toggle_waveform_visibility()
            pm.project_dirty = True

        clicked, val = imgui.menu_item(
            "Show Video Feed", selected=app_state.show_video_feed
        )
        if clicked:
            app_state.show_video_feed = val
            pm.project_dirty = True
            
        imgui.unindent()

    def _render_tools_menu(self, app_state, file_mgr):
        app = self.app

        if imgui.begin_menu("Tools", True):
            can_calibrate = file_mgr.video_path is not None
            if _menu_item_simple("Start Latency Calibration...", enabled=can_calibrate):
                calibration = getattr(app, "calibration", None)
                if calibration:
                    calibration.start_latency_calibration()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Calibrate latency. Requires a video to be loaded and points "
                    "on Timeline 1."
                    if can_calibrate
                    else "Please load a video to enable calibration."
                )

            clicked, _ = imgui.menu_item(
                "Manage Generated Files...",
                selected=app_state.show_generated_file_manager,
            )
            if clicked:
                app.toggle_file_manager_window()

            if not hasattr(app_state, "show_autotuner_window"):
                app_state.show_autotuner_window = False
            clicked, _ = imgui.menu_item(
                "Performance Autotuner...",
                selected=app_state.show_autotuner_window,
            )
            if clicked:
                app_state.show_autotuner_window = not app_state.show_autotuner_window

            fs_proc = getattr(app, "funscript_processor", None)
            can_compare = (
                fs_proc is not None
                and fs_proc.get_actions("primary")
                and fs_proc.get_actions("secondary")
            )
            if _menu_item_simple("Compare Timelines...", enabled=can_compare):
                trigger = getattr(app, "trigger_timeline_comparison", None)
                if trigger:
                    trigger()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Compares the signals on Timeline 1 and Timeline 2 to "
                    "calculate the optimal time offset."
                )

            if _menu_item_simple("Reload Plugin Filters"):
                # Reload plugins for all interactive timelines
                timeline1 = getattr(app, "interactive_funscript_timeline1", None)
                timeline2 = getattr(app, "interactive_funscript_timeline2", None)
                if timeline1:
                    timeline1._reload_plugins()
                if timeline2:
                    timeline2._reload_plugins()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Reload funscript filter plugins from the plugins directories. "
                    "Use this after adding new plugin files."
                )

            if not hasattr(app, "tensorrt_compiler_window"):
                app.tensorrt_compiler_window = None

            if _menu_item_simple("Compile YOLO Model to TensorRT (.engine)..."):
                from application.gui_components.engine_compiler.tensorrt_compiler_window import (  # noqa: E501
                    TensorRTCompilerWindow,
                )

                def on_close():
                    app.tensorrt_compiler_window = None

                tw = app.tensorrt_compiler_window
                if tw is None:
                    app.tensorrt_compiler_window = TensorRTCompilerWindow(
                        app, on_close_callback=on_close
                    )
                else:
                    tw._reset_state()
                    tw.is_open = True
            imgui.end_menu()

    def _render_ai_menu(self):
        app = self.app
        if imgui.begin_menu("AI", True):
            if _menu_item_simple("Download Default Models"):
                dl = getattr(app, "download_default_models", None)
                if dl:
                    dl()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Download default AI models if they don't already exist."
                )
            imgui.end_menu()

    def _render_updates_menu(self):
        app = self.app
        settings = app.app_settings
        updater = app.updater

        if imgui.begin_menu("Updates", True):
            # settings toggles
            for key, label, default in (
                ("updater_check_on_startup", "Check for Updates on Startup", True),
                ("updater_check_periodically", "Check Periodically in Background (Hourly)", True),  # noqa: E501
            ):
                cur = settings.get(key, default)
                clicked, new_val = imgui.menu_item(label, selected=cur)
                if clicked and new_val != cur:
                    settings.set(key, new_val)
                if key == "updater_suppress_popup" and imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "If suppressed, only the menu bar indicator will be shown."
                    )
            imgui.separator()

            if _menu_item_simple("Select Update Commit"):
                app.app_state_ui.show_update_settings_dialog = True
            if imgui.is_item_hovered():
                token = updater.token_manager.get_token()
                imgui.set_tooltip(
                    "GitHub token and version selection."
                    if token
                    else "GitHub token and version selection.\nNo token set."
                )

            can_apply = updater.update_available and not updater.update_in_progress
            if _menu_item_simple("Apply Pending Update...", enabled=can_apply):
                updater.show_update_dialog = True
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Shows the update dialog if an update has been detected."
                )
            imgui.end_menu()