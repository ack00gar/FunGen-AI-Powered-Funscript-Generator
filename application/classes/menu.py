import imgui
import os

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

        if imgui.begin_popup_modal("Select Reference Timeline##TimelineSelectPopup", True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text("Which timeline has the correct timing?")
            imgui.text_wrapped("The offset will be calculated for the other timeline and applied to it.")
            imgui.separator()

            for i in range(1, 3):
                if imgui.radio_button(f"Timeline {i} is the Reference", app_state.timeline_comparison_reference_num == i):
                    app_state.timeline_comparison_reference_num = i
                    break
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
        popup_pos_y = main_viewport.pos[1] + (main_viewport.size[1] - 240) * 0.5
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
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 0.0, 1.0) # TODO: move to theme, yellow
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
        """Render the main menu bar."""
        app_state = self.app.app_state_ui
        file_mgr = self.app.file_manager
        stage_proc = self.app.stage_processor

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

    def _render_file_menu(self, app_state, file_mgr):
        """Render the File menu."""
        if imgui.begin_menu("File", True):
            file_operations = [
                ("New Project", lambda: self.app.reset_project_state(for_new_project=True)),
                ("Project...", lambda: self.app.file_manager.open_project_dialog()),
                ("Video...", lambda: self.app.file_manager.open_video_dialog()),
                ("Close Project", lambda: self.app.reset_project_state(for_new_project=True)),
            ]
            
            for label, callback in file_operations:
                if imgui.menu_item(label)[0]:
                    callback()
                    if label == "New Project":
                        self.app.project_manager.project_dirty = True

            recent_projects = self.app.app_settings.get("recent_projects", [])
            can_open_recent = len(recent_projects) > 0
            if imgui.begin_menu("Open Recent", enabled=can_open_recent):
                if imgui.menu_item("Clear Menu")[0]:
                    self.app.app_settings.set("recent_projects", [])
                if can_open_recent:
                    imgui.separator()
                    for project_path in recent_projects:
                        display_name = os.path.basename(project_path)
                        if imgui.menu_item(display_name)[0]:
                            self.app.file_manager.load_project(project_path)
                imgui.end_menu()
            imgui.separator()

            can_save_project = self.app.project_manager.project_file_path is not None
            save_operations = [
                ("Save Project", lambda: self.app.project_manager.save_project_dialog(), can_save_project),
                ("Save Project As...", lambda: self.app.project_manager.save_project_dialog(save_as=True), True)
            ]
            
            for label, callback, enabled in save_operations:
                if imgui.menu_item(label, enabled=enabled)[0]:
                    callback()
            imgui.separator()

            import_export_menus = [
                ("Import...", [
                    ("Funscript to Timeline 1...", lambda: self.app.file_manager.import_funscript_to_timeline(1)),
                    ("Funscript to Timeline 2...", lambda: self.app.file_manager.import_funscript_to_timeline(2)),
                    ("Stage 2 Overlay Data...", lambda: self.app.file_manager.import_stage2_overlay_data()),
                ]),
                ("Export...", [
                    ("Funscript from Timeline 1...", lambda: self.app.file_manager.export_funscript_from_timeline(1)),
                    ("Funscript from Timeline 2...", lambda: self.app.file_manager.export_funscript_from_timeline(2)),
                ])
            ]
            
            for menu_label, items in import_export_menus:
                if imgui.begin_menu(menu_label):
                    for label, callback in items:
                        if imgui.menu_item(label)[0]:
                            callback()
                    imgui.end_menu()
            imgui.separator()

            if imgui.menu_item("Exit")[0]:
                self.app.shutdown_app()
            imgui.end_menu()

    def _render_edit_menu(self, app_state):
        """Render the Edit menu."""
        if imgui.begin_menu("Edit", True):
            timeline_operations = [
                (1, "T1", "Ctrl+Z", "Ctrl+Y"),
                (2, "T2", "Alt+Ctrl+Z", "Alt+Ctrl+Y"),
            ]
            
            for timeline_num, timeline_name, undo_shortcut, redo_shortcut in timeline_operations:
                undo_manager = self.app.funscript_processor._get_undo_manager(timeline_num)
                can_undo = undo_manager.can_undo() if undo_manager else False
                can_redo = undo_manager.can_redo() if undo_manager else False
                
                if imgui.menu_item(f"Undo {timeline_name} Change", undo_shortcut, selected=False, enabled=can_undo)[0]:
                    self.app.funscript_processor.perform_undo_redo(timeline_num, 'undo')
                if imgui.menu_item(f"Redo {timeline_name} Change", redo_shortcut, selected=False, enabled=can_redo)[0]:
                    self.app.funscript_processor.perform_undo_redo(timeline_num, 'redo')
                if timeline_num == 1:
                    imgui.separator()
            imgui.end_menu()

    def _render_view_menu(self, app_state, stage_proc):
        """Render the View menu."""
        if imgui.begin_menu("View", True):
            sections = [
                (self._render_ui_mode_section, [app_state]),
                (self._render_ui_layout_section, [app_state]),
                (self._render_panels_submenu, [app_state]),
                (self._render_layout_options_section, [app_state]),
                (self._render_timeline_editors_section, [app_state]),
                (self._render_overlays_section, [app_state, stage_proc]),
            ]
            for i, (section_method, args) in enumerate(sections):
                section_method(*args)
                if i < len(sections) - 1:  # Don't add separator after last section
                    imgui.separator()
            imgui.end_menu()

    def _render_ui_mode_section(self, app_state):
        """Render the UI Mode section of the View menu."""
        imgui.text("UI Mode")
        imgui.indent()
        ui_modes = [("expert", "Expert Mode"), ("simple", "Simple Mode")]
        
        for mode_value, mode_label in ui_modes:
            is_selected = app_state.ui_view_mode == mode_value
            if imgui.radio_button(f"{mode_label}##UIMode{mode_label.replace(' ', '')}", is_selected):
                if not is_selected:
                    app_state.ui_view_mode = mode_value
                    self.app.app_settings.set("ui_view_mode", mode_value)
        imgui.unindent()

    def _render_ui_layout_section(self, app_state):
        """Render the UI Layout Mode section of the View menu."""
        imgui.text("UI Layout Mode")
        imgui.indent()
        
        layout_modes = [("fixed", "Fixed Panels"), ("floating", "Floating Windows")]
        
        for mode_value, mode_label in layout_modes:
            is_selected = app_state.ui_layout_mode == mode_value
            if imgui.radio_button(f"{mode_label}##UILayoutMode{mode_label.replace(' ', '')}", is_selected):
                if not is_selected:
                    app_state.ui_layout_mode = mode_value
                    if mode_value == 'floating':
                        app_state.just_switched_to_floating = True
                    self.app.project_manager.project_dirty = True
        imgui.unindent()

    def _render_panels_submenu(self, app_state):
        """Render the Panels submenu of the View menu."""
        is_floating_mode = app_state.ui_layout_mode == 'floating'
        
        if imgui.begin_menu("Panels", enabled=is_floating_mode):
            panel_items = [
                ("Control Panel", "show_control_panel_window"),
                ("Info & Graphs", "show_info_graphs_window"),
                ("Video Display", "show_video_display_window"),
                ("Video Navigation", "show_video_navigation_window"),
            ]
            
            for label, attr_name in panel_items:
                clicked, selected = imgui.menu_item(label, selected=getattr(app_state, attr_name))
                if clicked:
                    setattr(app_state, attr_name, selected)
                    self.app.project_manager.project_dirty = True
            imgui.end_menu()
        
        if imgui.is_item_hovered() and not is_floating_mode:
            imgui.set_tooltip("Window toggles are for floating mode.")

    def _render_layout_options_section(self, app_state):
        """Render the Layout Options section of the View menu."""
        imgui.text("Layout Options")
        imgui.indent()
        
        if not hasattr(app_state, 'full_width_nav'):
            app_state.full_width_nav = False
        
        is_fixed_mode = app_state.ui_layout_mode == 'fixed'
        clicked, selected = imgui.menu_item("Full-Width Navigation Bar", selected=app_state.full_width_nav, enabled=is_fixed_mode)
        if clicked:
            app_state.full_width_nav = selected
            self.app.project_manager.project_dirty = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Only available in 'Fixed Panels' layout mode.")
        imgui.unindent()

    def _render_timeline_editors_section(self, app_state):
        """Render the Timeline Editors & Previews section of the View menu."""
        imgui.text("Timeline Editors & Previews")
        imgui.indent()
        
        timeline_items = [("Interactive Timeline 1", "show_funscript_interactive_timeline"), ("Interactive Timeline 2", "show_funscript_interactive_timeline2")]
        
        for label, attr_name in timeline_items:
            clicked, selected = imgui.menu_item(label, selected=getattr(app_state, attr_name))
            if clicked:
                setattr(app_state, attr_name, selected)
                self.app.project_manager.project_dirty = True
        imgui.separator()

        preview_items = [
            ("Funscript Preview Bar", "show_funscript_timeline"),
            ("Heatmap", "show_heatmap"),
        ]

        for label, attr_name in preview_items:
            clicked, selected = imgui.menu_item(label, selected=getattr(app_state, attr_name))
            if clicked:
                setattr(app_state, attr_name, selected)
                self.app.project_manager.project_dirty = True
        use_simplified = self.app.app_settings.get("use_simplified_funscript_preview", False)
        clicked, new_simplified_val = imgui.menu_item("Use Simplified Funscript Preview", selected=use_simplified)
        if clicked:
            self.app.app_settings.set("use_simplified_funscript_preview", new_simplified_val)
            self.app.app_state_ui.funscript_preview_dirty = True
        imgui.separator()

        advanced_items = [
            ("Show Timeline Editor Buttons", "show_timeline_editor_buttons"),
            ("Show Advanced Options", "show_advanced_options"),
        ]
        
        for label, attr_name in advanced_items:
            clicked, selected = imgui.menu_item(label, selected=getattr(app_state, attr_name))
            if clicked:
                setattr(app_state, attr_name, selected)
                if attr_name == "show_timeline_editor_buttons":
                    self.app.app_settings.set("show_timeline_editor_buttons", selected)
                elif attr_name == "show_advanced_options":
                    self.app.app_settings.set("show_advanced_options", selected)
                self.app.project_manager.project_dirty = True
        imgui.unindent()

    def _render_overlays_section(self, app_state, stage_proc):
        """Render the Overlays & Aux Windows section of the View menu."""
        imgui.text("Overlays & Aux Windows")
        imgui.indent()

        gauge_items = [
            ("Script Gauge (Timeline 1)", "show_gauge_window_timeline1"),
            ("Script Gauge (Timeline 2)", "show_gauge_window_timeline2"),
        ]
        
        for label, attr_name in gauge_items:
            clicked, selected = imgui.menu_item(label, selected=getattr(app_state, attr_name))
            if clicked:
                setattr(app_state, attr_name, selected)
                self.app.project_manager.project_dirty = True

        overlay_items = [("L/R Dial Graph", "show_lr_dial_graph")]

        for label, attr_name in overlay_items:
            clicked, selected = imgui.menu_item(label, selected=getattr(app_state, attr_name))
            if clicked:
                setattr(app_state, attr_name, selected)
                self.app.project_manager.project_dirty = True

        if not hasattr(app_state, 'show_chapter_list_window'):
            app_state.show_chapter_list_window = False
        clicked, selected = imgui.menu_item("Chapter List", selected=app_state.show_chapter_list_window)
        if clicked:
            app_state.show_chapter_list_window = selected
            self.app.project_manager.project_dirty = True
        imgui.separator()

        if self.app.tracker:
            tracker_items = [
                ("Show Detections/Masks", "ui_show_masks"),
                ("Show Optical Flow", "ui_show_flow"),
            ]
            
            for label, flag_name in tracker_items:
                clicked, selected = imgui.menu_item(label, selected=getattr(app_state, flag_name))
                if clicked: app_state.set_tracker_ui_flag(flag_name.replace("ui_", ""), selected)

        can_show_s2 = stage_proc.stage2_overlay_data is not None
        clicked, selected = imgui.menu_item("Show Stage 2 Overlay", selected=app_state.show_stage2_overlay, enabled=can_show_s2)
        if clicked:
            app_state.show_stage2_overlay = selected
            self.app.project_manager.project_dirty = True

        clicked, _ = imgui.menu_item("Audio Waveform", selected=app_state.show_audio_waveform)
        if clicked:
            self.app.toggle_waveform_visibility()
            self.app.project_manager.project_dirty = True
        imgui.unindent()

    def _render_tools_menu(self, app_state, file_mgr):
        """Render the Tools menu."""
        if imgui.begin_menu("Tools", True):
            can_calibrate = file_mgr.video_path is not None
            if imgui.menu_item("Start Latency Calibration...", enabled=can_calibrate)[0]:
                if hasattr(self.app, 'calibration'):
                    self.app.calibration.start_latency_calibration()
            if imgui.is_item_hovered():
                tooltip = "Calibrate latency. Requires a video to be loaded and points on Timeline 1."
                if not can_calibrate:
                    tooltip = "Please load a video to enable calibration."
                imgui.set_tooltip(tooltip)

            clicked, selected = imgui.menu_item("Manage Generated Files...", selected=app_state.show_generated_file_manager)
            if clicked:
                self.app.toggle_file_manager_window()
            if not hasattr(app_state, 'show_autotuner_window'):
                app_state.show_autotuner_window = False
            clicked, selected = imgui.menu_item("Performance Autotuner...", selected=app_state.show_autotuner_window)
            if clicked:
                app_state.show_autotuner_window = not app_state.show_autotuner_window

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

            if not hasattr(self.app, 'tensorrt_compiler_window'):
                self.app.tensorrt_compiler_window = None
            if imgui.menu_item("Compile YOLO Model to TensorRT (.engine)...")[0]:
                from application.gui_components.engine_compiler.tensorrt_compiler_window import TensorRTCompilerWindow
                def on_close():
                    self.app.tensorrt_compiler_window = None
                if self.app.tensorrt_compiler_window is None:
                    self.app.tensorrt_compiler_window = TensorRTCompilerWindow(self.app, on_close_callback=on_close)
                else:
                    self.app.tensorrt_compiler_window._reset_state()
                    self.app.tensorrt_compiler_window.is_open = True
            imgui.end_menu()

    def _render_ai_menu(self):
        """Render the AI menu."""
        if imgui.begin_menu("AI", True):
            if imgui.menu_item("Download Default Models")[0]:
                if hasattr(self.app, 'download_default_models'):
                    self.app.download_default_models()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Download default AI models if they don't already exist.")
            imgui.end_menu()

    def _render_updates_menu(self):
        """Render the Updates menu."""
        if imgui.begin_menu("Updates", True):
            update_settings = [
                ("updater_check_on_startup", "Check for Updates on Startup"),
                ("updater_check_periodically", "Check Periodically in Background (Hourly)"),
                ("updater_suppress_popup", "Suppress Update Notification Popup"),
            ]
            
            for setting_key, label in update_settings:
                current_value = self.app.app_settings.get(setting_key, True if "startup" in setting_key or "periodic" in setting_key else False)
                clicked, new_value = imgui.menu_item(label, selected=current_value)
                if clicked:
                    self.app.app_settings.set(setting_key, new_value)
                if setting_key == "updater_suppress_popup" and imgui.is_item_hovered():
                    imgui.set_tooltip("If suppressed, only the menu bar indicator will be shown.")
            imgui.separator()

            if imgui.menu_item("Select Update Commit")[0]:
                self.app.app_state_ui.show_update_settings_dialog = True
            if imgui.is_item_hovered():
                current_token = self.app.updater.token_manager.get_token()
                if current_token:
                    imgui.set_tooltip(f"GitHub token and version selection.")
                else:
                    imgui.set_tooltip("GitHub token and version selection.\nNo token set.")

            can_apply_update = self.app.updater.update_available and not self.app.updater.update_in_progress
            if imgui.menu_item("Apply Pending Update...", enabled=can_apply_update)[0]:
                self.app.updater.show_update_dialog = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Shows the update dialog if an update has been detected.")
            imgui.end_menu()
