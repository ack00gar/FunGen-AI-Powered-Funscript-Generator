"""Dialog and popup rendering mixin for GUI."""
import imgui
import os
from application.utils.imgui_helpers import center_next_window, begin_modal_centered


class DialogRendererMixin:
    """Mixin providing dialog and popup rendering methods."""

    def _render_batch_confirmation_dialog(self):
        app = self.app
        if not app.show_batch_confirmation_dialog:
            return

        colors = self.colors
        imgui.open_popup("Batch Processing Setup")
        mv = imgui.get_main_viewport()
        imgui.set_next_window_size(mv.size[0] * 0.85, mv.size[1] * 0.8, condition=imgui.APPEARING)
        imgui.set_next_window_position(
            mv.pos[0] + mv.size[0] * 0.5,
            mv.pos[1] + mv.size[1] * 0.5,
            pivot_x=0.5, pivot_y=0.5, condition=imgui.APPEARING
        )

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

            # Set all overrides dropdown + button
            video_format_options = ["Auto (Heuristic)", "2D", "VR (he_sbs)", "VR (he_tb)", "VR (fisheye_sbs)", "VR (fisheye_tb)"]
            imgui.text("Set all overrides:")
            imgui.same_line()
            imgui.set_next_item_width(160)
            _, self.batch_set_all_format_idx = imgui.combo("##set_all_format", self.batch_set_all_format_idx, video_format_options)
            imgui.same_line()
            if imgui.button("Apply to All"):
                for video_data in self.batch_videos_data:
                    video_data["override_format_idx"] = self.batch_set_all_format_idx

            if imgui.begin_child("VideoList", height=-120):
                table_flags = imgui.TABLE_BORDERS | imgui.TABLE_SIZING_STRETCH_PROP | imgui.TABLE_SCROLL_Y
                if imgui.begin_table("BatchVideosTable", 8, flags=table_flags):
                    imgui.table_setup_column("Process", init_width_or_weight=0.5)
                    imgui.table_setup_column("Video File", init_width_or_weight=3.0)
                    imgui.table_setup_column("Created", init_width_or_weight=1.0)
                    imgui.table_setup_column("Tracker", init_width_or_weight=1.2)
                    imgui.table_setup_column("Version", init_width_or_weight=0.7)
                    imgui.table_setup_column("Git Hash", init_width_or_weight=0.8)
                    imgui.table_setup_column("Detected", init_width_or_weight=0.8)
                    imgui.table_setup_column("Override", init_width_or_weight=1.5)

                    imgui.table_headers_row()

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

                        # Creation date column
                        imgui.table_set_column_index(2)
                        creation_date = video_data.get("creation_date", "")
                        if creation_date:
                            # Show date portion only (YYYY-MM-DD) for compactness
                            display_date = creation_date[:10] if len(creation_date) >= 10 else creation_date
                            imgui.text(display_date)
                            if imgui.is_item_hovered():
                                imgui.set_tooltip(creation_date)
                        else:
                            imgui.text_colored("-", 0.5, 0.5, 0.5, 1.0)

                        # Tracker/model column
                        imgui.table_set_column_index(3)
                        tracker_name = video_data.get("tracker_name", "")
                        if tracker_name:
                            imgui.text(tracker_name)
                        else:
                            imgui.text_colored("-", 0.5, 0.5, 0.5, 1.0)

                        # FunGen version column
                        imgui.table_set_column_index(4)
                        fungen_version = video_data.get("fungen_version", "")
                        if fungen_version:
                            imgui.text(fungen_version)
                        else:
                            imgui.text_colored("-", 0.5, 0.5, 0.5, 1.0)

                        # Git hash column
                        imgui.table_set_column_index(5)
                        git_hash = video_data.get("git_commit_hash", "")
                        if git_hash:
                            short_hash = git_hash[:7] if len(git_hash) > 7 else git_hash
                            imgui.text(short_hash)
                            if imgui.is_item_hovered():
                                imgui.set_tooltip(git_hash)
                        else:
                            imgui.text_colored("-", 0.5, 0.5, 0.5, 1.0)

                        imgui.table_set_column_index(6); imgui.text(video_data["detected_format"])

                        imgui.table_set_column_index(7); imgui.push_id(f"ovr_{i}"); imgui.set_next_item_width(-1)
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

            from application.utils.imgui_helpers import DisabledScope
            with DisabledScope(not has_3_stages):
                sec_axis_label = self.app.app_settings.get("default_secondary_axis", "roll") if hasattr(self.app, 'app_settings') else "roll"
                _, self.batch_generate_roll_file_ui = imgui.checkbox(f"Generate .{sec_axis_label} file", self.batch_generate_roll_file_ui if has_3_stages else False)

            # Adaptive performance tuning checkbox
            _, self.batch_adaptive_tuning_ui = imgui.checkbox("Adaptive performance tuning", self.batch_adaptive_tuning_ui)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Progressively optimizes pipeline thread settings during batch.\n"
                                  "Starts conservative, tests small improvements after each video.\n"
                                  "Best settings saved for future use.")

            # Save preprocessed video (offline trackers only, off by default)
            is_offline_tracker = False
            if hasattr(self, 'selected_batch_tracker_name') and self.selected_batch_tracker_name:
                t_info = discovery.get_tracker_info(self.selected_batch_tracker_name)
                if t_info and t_info.category == TrackerCategory.OFFLINE:
                    is_offline_tracker = True
            if is_offline_tracker:
                _, self.batch_save_preprocessed_video_ui = imgui.checkbox(
                    "Save preprocessed video", self.batch_save_preprocessed_video_ui)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Keep the preprocessed (resized/unwarped) video for each processed file.\n"
                                      "WARNING: Uses significant disk space (~200-500MB per video).\n"
                                      "Only enable if you plan to re-run analysis on the same videos.")
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
        center_next_window(700, 400)

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
        popup_id = "Error###ErrorPopup"
        if not self.error_popup_active and not imgui.is_popup_open(popup_id):
            return

        if self.error_popup_active:
            imgui.open_popup(popup_id)

        center_next_window(600)
        if imgui.begin_popup_modal(popup_id)[0]:
            # Title
            window_width = imgui.get_window_width()
            title_width = imgui.calc_text_size(self.error_popup_title)[0]
            imgui.set_cursor_pos_x((window_width - title_width) * 0.5)
            imgui.text(self.error_popup_title)
            imgui.separator()
            # Message (wrapped to fit window)
            imgui.text_wrapped(self.error_popup_message)
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

        if getattr(app_state, 'show_simulator_3d', False) and not self.app.app_settings.get('simulator_3d_overlay_mode', False):
            self.simulator_3d_window_ui.render()

        if getattr(app_state, 'show_script_gauge', False):
            self.script_gauge_ui.render()

        if getattr(app_state, 'show_plugin_pipeline', False):
            self.plugin_pipeline_ui.render()

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

        # Addon update checker dialog
        self.app.addon_checker.render_update_dialog()

        # Keyboard Shortcuts Dialog (accessible via F1 or Help menu)
        self.keyboard_shortcuts_dialog.render()

        # One-time shortcut migration notice
        self._render_shortcuts_reset_notice()

        # Legacy first-run popup (only shown if wizard was skipped, e.g. manual re-trigger from menu)
        self._render_first_run_setup_popup()

    def _render_shortcuts_reset_notice(self):
        """One-time notice shown after v0.7.0 shortcut migration."""
        if not self.app.app_settings.shortcuts_were_reset:
            return

        popup_id = "Shortcuts Updated###ShortcutsReset"
        if not imgui.is_popup_open(popup_id):
            imgui.open_popup(popup_id)

        center_next_window(450)
        if imgui.begin_popup_modal(popup_id)[0]:
            imgui.text("Keyboard Shortcuts Reset")
            imgui.separator()
            imgui.spacing()
            imgui.text_wrapped(
                "Your keyboard shortcuts have been reset to the new defaults. "
                "The shortcut system was restructured in v0.7.0 and old bindings "
                "could cause unexpected behavior (e.g. drag acting as pan, "
                "Shift+Arrow not working)."
            )
            imgui.spacing()
            imgui.text_wrapped(
                "Press F1 at any time to review and customize shortcuts."
            )
            imgui.spacing()
            button_width = 120
            imgui.set_cursor_pos_x((imgui.get_window_width() - button_width) * 0.5)
            if imgui.button("OK", width=button_width):
                self.app.app_settings.shortcuts_were_reset = False
                imgui.close_current_popup()
            imgui.end_popup()

    def _render_first_run_setup_popup(self):
        app = self.app
        if not app.show_first_run_setup_popup:
            return
        # Skip if the new wizard is active — it handles model download itself
        if hasattr(self, '_first_run_wizard') and self._first_run_wizard is not None:
            return

        status_msg = app.first_run_status_message.lower()
        is_complete = "complete" in status_msg
        is_failed = "failed" in status_msg
        closable = is_complete or is_failed

        imgui.open_popup("First-Time Setup")
        center_next_window(450)
        opened, visible = imgui.begin_popup_modal(
            "First-Time Setup", closable, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )
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
