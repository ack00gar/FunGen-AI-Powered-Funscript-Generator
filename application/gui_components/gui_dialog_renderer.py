"""Dialog and popup rendering mixin for GUI."""
import imgui
import os
import time


class DialogRendererMixin:
    """Mixin providing dialog and popup rendering methods."""

    def _render_batch_confirmation_dialog(self):
        app = self.app
        if not app.show_batch_confirmation_dialog:
            return

        colors = self.colors
        imgui.open_popup("Batch Processing Setup")
        main_viewport = imgui.get_main_viewport()
        imgui.set_next_window_size(main_viewport.size[0] * 0.7, main_viewport.size[1] * 0.8, condition=imgui.APPEARING)
        popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                     main_viewport.pos[1] + main_viewport.size[1] * 0.5)
        imgui.set_next_window_position(popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5, condition=imgui.APPEARING)

        if imgui.begin_popup_modal("Batch Processing Setup", True)[0]:
            imgui.text(f"Found {len(self.batch_videos_data)} videos for batch processing.")
            imgui.separator()

            imgui.text("Overwrite Strategy:")
            imgui.same_line()
            if imgui.radio_button("Skip existing FunGen scripts", self.batch_overwrite_mode_ui == 0): self.batch_overwrite_mode_ui = 0
            imgui.same_line()
            if imgui.radio_button("Skip if ANY script exists", self.batch_overwrite_mode_ui == 1): self.batch_overwrite_mode_ui = 1
            imgui.same_line()
            if imgui.radio_button("Overwrite all existing scripts", self.batch_overwrite_mode_ui == 2): self.batch_overwrite_mode_ui = 2

            if self.batch_overwrite_mode_ui != self.last_overwrite_mode_ui:
                for video in self.batch_videos_data:
                    status = video["funscript_status"]
                    if self.batch_overwrite_mode_ui == 0: video["selected"] = status != 'fungen'
                    elif self.batch_overwrite_mode_ui == 1: video["selected"] = status is None
                    elif self.batch_overwrite_mode_ui == 2: video["selected"] = True
                self.last_overwrite_mode_ui = self.batch_overwrite_mode_ui

            imgui.separator()

            if imgui.begin_child("VideoList", height=-120):
                table_flags = imgui.TABLE_BORDERS | imgui.TABLE_SIZING_STRETCH_PROP | imgui.TABLE_SCROLL_Y
                if imgui.begin_table("BatchVideosTable", 4, flags=table_flags):
                    imgui.table_setup_column("Process", init_width_or_weight=0.5)
                    imgui.table_setup_column("Video File", init_width_or_weight=4.0)
                    imgui.table_setup_column("Detected", init_width_or_weight=1.3)
                    imgui.table_setup_column("Override", init_width_or_weight=1.5)

                    imgui.table_headers_row()

                    video_format_options = ["Auto (Heuristic)", "2D", "VR (he_sbs)", "VR (he_tb)", "VR (fisheye_sbs)", "VR (fisheye_tb)"]

                    for i, video_data in enumerate(self.batch_videos_data):
                        imgui.table_next_row()
                        imgui.table_set_column_index(0); imgui.push_id(f"sel_{i}")
                        _, video_data["selected"] = imgui.checkbox("##select", video_data["selected"])
                        imgui.pop_id()

                        imgui.table_set_column_index(1)
                        status = video_data["funscript_status"]
                        if status == 'fungen': imgui.text_colored(os.path.basename(video_data["path"]), *colors.VIDEO_STATUS_FUNGEN)
                        elif status == 'other': imgui.text_colored(os.path.basename(video_data["path"]), *colors.VIDEO_STATUS_OTHER)
                        else: imgui.text(os.path.basename(video_data["path"]))

                        if imgui.is_item_hovered():
                            if status == 'fungen':
                                imgui.set_tooltip("Funscript created by this version of FunGen")
                            elif status == 'other':
                                imgui.set_tooltip("Funscript exists (unknown or older version)")
                            else:
                                imgui.set_tooltip("No Funscript exists for this video")

                        imgui.table_set_column_index(2); imgui.text(video_data["detected_format"])

                        imgui.table_set_column_index(3); imgui.push_id(f"ovr_{i}"); imgui.set_next_item_width(-1)
                        _, video_data["override_format_idx"] = imgui.combo("##override", video_data["override_format_idx"], video_format_options)
                        imgui.pop_id()

                    imgui.end_table()
                imgui.end_child()

            imgui.separator()
            imgui.text("Processing Method:")

            # Get available batch-compatible trackers dynamically
            from application.gui_components.dynamic_tracker_ui import get_dynamic_tracker_ui
            from config.tracker_discovery import TrackerCategory

            tracker_ui = get_dynamic_tracker_ui()
            discovery = tracker_ui.discovery

            # Get live (non-intervention) and offline trackers
            batch_compatible_trackers = []
            tracker_internal_names = []

            # Add offline trackers
            offline_trackers = discovery.get_trackers_by_category(TrackerCategory.OFFLINE)
            for tracker in offline_trackers:
                if tracker.supports_batch:
                    # Add prefix based on folder name
                    if tracker.folder_name and tracker.folder_name.lower() == "experimental":
                        display_name = f"Experimental: {tracker.display_name}"
                    else:
                        display_name = f"Offline: {tracker.display_name}"
                    batch_compatible_trackers.append(display_name)
                    tracker_internal_names.append(tracker.internal_name)

            # Add live trackers (non-intervention only)
            live_trackers = discovery.get_trackers_by_category(TrackerCategory.LIVE)
            for tracker in live_trackers:
                if tracker.supports_batch and not tracker.requires_intervention:
                    # Add prefix based on folder name
                    if tracker.folder_name and tracker.folder_name.lower() == "experimental":
                        display_name = f"Experimental: {tracker.display_name}"
                    else:
                        display_name = f"Live: {tracker.display_name}"
                    batch_compatible_trackers.append(display_name)
                    tracker_internal_names.append(tracker.internal_name)

            # Create dropdown
            imgui.set_next_item_width(300)
            changed, self.selected_batch_method_idx_ui = imgui.combo(
                "##batch_tracker",
                self.selected_batch_method_idx_ui,
                batch_compatible_trackers
            )

            # Store the selected tracker's internal name for later use
            if 0 <= self.selected_batch_method_idx_ui < len(tracker_internal_names):
                self.selected_batch_tracker_name = tracker_internal_names[self.selected_batch_method_idx_ui]
            else:
                self.selected_batch_tracker_name = None

            imgui.text("Output Options:")
            _, self.batch_apply_ultimate_autotune_ui = imgui.checkbox("Apply Ultimate Autotune", self.batch_apply_ultimate_autotune_ui)
            imgui.same_line()
            _, self.batch_copy_funscript_to_video_location_ui = imgui.checkbox("Save copy next to video", self.batch_copy_funscript_to_video_location_ui)
            imgui.same_line()

            # Check if selected tracker supports roll file generation (3-stage trackers)
            has_3_stages = False
            if hasattr(self, 'selected_batch_tracker_name') and self.selected_batch_tracker_name:
                tracker_info = discovery.get_tracker_info(self.selected_batch_tracker_name)
                if tracker_info and tracker_info.properties:
                    has_3_stages = tracker_info.properties.get("num_stages", 0) >= 3

            if not has_3_stages:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True); imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            _, self.batch_generate_roll_file_ui = imgui.checkbox("Generate .roll file", self.batch_generate_roll_file_ui if has_3_stages else False)
            if not has_3_stages:
                imgui.pop_style_var(); imgui.internal.pop_item_flag()

            # Adaptive performance tuning checkbox
            _, self.batch_adaptive_tuning_ui = imgui.checkbox("Adaptive performance tuning", self.batch_adaptive_tuning_ui)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Progressively optimizes pipeline thread settings during batch.\n"
                                  "Starts conservative, tests small improvements after each video.\n"
                                  "Best settings saved for future use.")
            cur_p = app.stage_processor.num_producers_stage1
            cur_c = app.stage_processor.num_consumers_stage1
            imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.5, 0.5, 1.0)
            imgui.text(f"  Current pipeline: {cur_p} producers / {cur_c} consumers")
            imgui.pop_style_color()

            imgui.separator()
            if imgui.button("Start Batch", width=120):
                app._initiate_batch_processing_from_confirmation()
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", width=120):
                app._cancel_batch_processing_from_confirmation()
                imgui.close_current_popup()

            imgui.end_popup()

    def _render_ai_models_dialog(self):
        """Render AI Models configuration dialog."""
        app = self.app
        app_state = app.app_state_ui

        window_flags = imgui.WINDOW_NO_COLLAPSE
        main_viewport = imgui.get_main_viewport()
        center_x = main_viewport.pos[0] + main_viewport.size[0] * 0.5
        center_y = main_viewport.pos[1] + main_viewport.size[1] * 0.5
        imgui.set_next_window_position(center_x, center_y, imgui.ONCE, 0.5, 0.5)
        imgui.set_next_window_size(700, 400, imgui.ONCE)

        is_open, app_state.show_ai_models_dialog = imgui.begin(
            "AI Models Configuration##AIModelsDialog",
            closable=True,
            flags=window_flags
        )

        if is_open:
            imgui.text("Configure AI Model Paths and Inference Settings")
            imgui.separator()
            imgui.spacing()

            # Use the same rendering as control panel
            if hasattr(self, 'control_panel_ui') and self.control_panel_ui:
                self.control_panel_ui._render_ai_model_settings()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Close button
            if imgui.button("Close", width=-1):
                app_state.show_ai_models_dialog = False

        imgui.end()

    def _render_error_popup(self):
        """Render error popup with early return to avoid expensive operations when not needed."""
        # Early return if no error popup is active - avoids expensive ImGui operations
        if not self.error_popup_active and not imgui.is_popup_open("ErrorPopup"):
            return

        if self.error_popup_active:
            imgui.open_popup("ErrorPopup")

        # Center the popup and set a normal size (compatibility for imgui versions)
        if hasattr(imgui, 'get_main_viewport'):
            main_viewport = imgui.get_main_viewport()
            popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                         main_viewport.pos[1] + main_viewport.size[1] * 0.5)
            imgui.set_next_window_position(popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5)
        else:
            # Fallback: center on window size if viewport not available
            popup_pos = (self.window_width * 0.5, self.window_height * 0.5)
            imgui.set_next_window_position(popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5)
        popup_width = 600
        imgui.set_next_window_size(popup_width, 0)  # Wider width, auto height
        if imgui.begin_popup_modal("ErrorPopup")[0]:
            # Center title
            window_width = imgui.get_window_width()
            title_width = imgui.calc_text_size(self.error_popup_title)[0]
            imgui.set_cursor_pos_x((window_width - title_width) * 0.5)
            imgui.text(self.error_popup_title)
            imgui.separator()
            # Center message
            message_lines = self.error_popup_message.split('\n')
            for line in message_lines:
                line_width = imgui.calc_text_size(line)[0]
                imgui.set_cursor_pos_x((window_width - line_width) * 0.5)
                imgui.text(line)
            imgui.spacing()
            # Center button
            button_width = 120
            imgui.set_cursor_pos_x((window_width - button_width) * 0.5)
            if imgui.button("Close", width=button_width):
                self.error_popup_active = False
                imgui.close_current_popup()
                if self.error_popup_action_callback:
                    self.error_popup_action_callback()
            imgui.end_popup()

    def _render_all_popups(self):
        """Optimized popup rendering - only renders visible/active popups."""
        app_state = self.app.app_state_ui

        # Only render gauge windows if they're shown AND not in overlay mode
        if getattr(app_state, 'show_gauge_window_timeline1', False) and not self.app.app_settings.get('gauge_overlay_mode', False):
            self.gauge_window_ui_t1.render()

        if getattr(app_state, 'show_gauge_window_timeline2', False) and not self.app.app_settings.get('gauge_overlay_mode', False):
            self.gauge_window_ui_t2.render()

        # Only render Movement Bar if shown AND not in overlay mode
        if getattr(app_state, 'show_lr_dial_graph', False) and not self.app.app_settings.get('movement_bar_overlay_mode', False):
            self.movement_bar_ui.render()

        if getattr(app_state, 'show_simulator_3d', False) and not self.app.app_settings.get('simulator_3d_overlay_mode', False):
            self.simulator_3d_window_ui.render()

        # Batch confirmation dialog (has internal visibility check)
        self._render_batch_confirmation_dialog()

        # File dialog only if open
        if self.file_dialog.open:
            self.file_dialog.draw()

        # Updater dialogs (have early returns to avoid expensive ImGui calls when not visible)
        self.app.updater.render_update_dialog()
        self.app.updater.render_update_error_dialog()
        self.app.updater.render_migration_warning_dialog()
        self.app.updater.render_update_settings_dialog()

        # Keyboard Shortcuts Dialog (accessible via F1 or Help menu)
        self.keyboard_shortcuts_dialog.render()

        # First-run setup wizard
        self._render_first_run_setup_popup()

    def _render_first_run_setup_popup(self):
        app = self.app
        if not app.show_first_run_setup_popup:
            return

        imgui.open_popup("First-Time Setup")
        main_viewport = imgui.get_main_viewport()
        popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                     main_viewport.pos[1] + main_viewport.size[1] * 0.5)
        imgui.set_next_window_position(popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5)

        status_msg = app.first_run_status_message.lower()
        is_complete = "complete" in status_msg
        is_failed = "failed" in status_msg
        closable = is_complete or is_failed
        popup_flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE

        opened, visible = imgui.begin_popup_modal("First-Time Setup", closable, flags=popup_flags)
        if opened:
            imgui.text("Welcome to FunGen!")
            imgui.spacing()
            imgui.text_wrapped(
                "FunGen generates funscripts from video using AI motion analysis. "
                "Before you can start, the required AI models need to be downloaded."
            )
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text_wrapped(f"Status: {app.first_run_status_message}")

            # Progress bar
            progress_percent = app.first_run_progress / 100.0
            imgui.progress_bar(progress_percent, size=(400, 0), overlay=f"{app.first_run_progress:.1f}%")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if is_complete:
                imgui.push_style_color(imgui.COLOR_TEXT, 0.2, 0.9, 0.2, 1.0)
                imgui.text("Setup complete! You're ready to go.")
                imgui.pop_style_color()
                imgui.spacing()
                if imgui.button("Get Started", width=150):
                    app.show_first_run_setup_popup = False
                    imgui.close_current_popup()
            elif is_failed:
                imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.3, 0.3, 1.0)
                imgui.text_wrapped(
                    "Setup failed. You can download models manually via AI menu > Download Models."
                )
                imgui.pop_style_color()
                imgui.spacing()
                if imgui.button("Close", width=150):
                    app.show_first_run_setup_popup = False
                    imgui.close_current_popup()
            else:
                imgui.text_wrapped("Please wait while models are being downloaded...")

            imgui.end_popup()


    # TODO: Move this to a separate class/error management module
    def show_error_popup(self, title, message, action_label=None, action_callback=None):
        self.error_popup_active = True
        self.error_popup_title = title
        self.error_popup_message = message
        self.error_popup_action_label = action_label
        self.error_popup_action_callback = action_callback
