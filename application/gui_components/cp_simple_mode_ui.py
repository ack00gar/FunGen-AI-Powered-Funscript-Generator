"""Simple Mode UI mixin for ControlPanelUI."""
import imgui
import os
import config
from application.utils import primary_button_style, destructive_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope, tooltip_if_hovered as _tooltip_if_hovered
from application.utils.section_card import section_card as _section_card
from config.tracker_discovery import TrackerCategory
from config.element_group_colors import ControlPanelColors as _CPColors


class SimpleModeMixin:
    """Mixin providing Simple Mode rendering methods."""

    def _render_simple_mode_ui(self):
        """Render Simple Mode UI with step-by-step workflow."""
        app = self.app
        app_state = app.app_state_ui
        processor = app.processor
        stage_proc = app.stage_processor
        fs_proc = app.funscript_processor

        flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE
        imgui.begin("FunGen Simple##SimpleControlPanel", flags=flags)

        # Title
        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.STATUS_INFO)
        imgui.text("Simple Mode")
        imgui.pop_style_color()
        imgui.text_wrapped("Easy 3-step workflow for beginners")
        imgui.spacing()

        # STEP 1: Load Video
        with _section_card("Step 1: Load Video##SimpleStep1", tier="primary") as is_open:
            if is_open:
                if processor and processor.video_info:
                    self._status_indicator("Video loaded", "ready", "Video is ready for analysis")
                    imgui.text_wrapped("File: %s" % os.path.basename(processor.video_path or "Unknown"))
                    video_info = processor.video_info
                    if video_info:
                        duration_str = "%.0f:%02.0f" % divmod(video_info.get('duration', 0), 60)
                        imgui.text_wrapped("Duration: %s | %dx%d | %.0f fps" % (
                            duration_str,
                            video_info.get('width', 0),
                            video_info.get('height', 0),
                            video_info.get('fps', 0)
                        ))
                else:
                    self._status_indicator("No video loaded", "info", "Drag and drop a video file onto the window")
                    imgui.text_wrapped("Supported formats: MP4, AVI, MOV, MKV")

        # STEP 2: Choose Analysis Method
        with _section_card("Step 2: Choose What to Track##SimpleStep2", tier="primary") as is_open:
            if is_open:
                # Auto-recommend tracker based on video properties
                if processor and processor.video_info and self.tracker_ui:
                    rec_name, rec_reason = self.tracker_ui.recommend_tracker(processor.video_info)
                    self._auto_recommended_tracker = rec_name
                    self._auto_recommendation_reason = rec_reason
                    # Auto-select on first video load if user hasn't manually picked
                    if not self._user_manually_picked_tracker:
                        if app_state.selected_tracker_name != rec_name:
                            app_state.selected_tracker_name = rec_name
                            if hasattr(app, 'app_settings') and hasattr(app.app_settings, 'set'):
                                app.app_settings.set("selected_tracker_name", rec_name)

                # Render card-based tracker selection
                self._render_simple_mode_tracker_selection()

        # STEP 3: Generate Funscript
        with _section_card("Step 3: Generate Funscript##SimpleStep3", tier="primary") as is_open:
            if is_open:
                # Show progress or start button
                if stage_proc.full_analysis_active:
                    self._simple_mode_post_processing_applied = False  # Reset for new analysis
                    self._render_simple_progress_display()

                    # Stop button
                    imgui.spacing()
                    with destructive_button_style():
                        if imgui.button("Stop Analysis", width=-1):
                            app.event_handlers.handle_abort_process_click()
                else:
                    acts = fs_proc.get_actions("primary")
                    if acts:
                        # Analysis complete - show completion state
                        self._status_indicator(
                            "Analysis Complete",
                            "ready",
                            "Generated %d motion points" % len(acts)
                        )
                        imgui.spacing()

                        # Post-processing prompt
                        self._render_simple_mode_post_processing_prompt()

                        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                        imgui.text_wrapped("What's next?")
                        imgui.pop_style_color()
                        imgui.spacing()

                        # Export button (primary action)
                        with primary_button_style():
                            if imgui.button("Export Funscript", width=-1):
                                # Trigger export for Timeline 1
                                self._export_funscript_timeline(app, 1)

                        # Fine-tune button (secondary action)
                        imgui.spacing()
                        if imgui.button("Fine-Tune Results (Switch to Expert Mode)", width=-1):
                            app_state.ui_view_mode = "expert"
                            app.logger.info("Switched to Expert Mode", extra={"status_message": True})
                    else:
                        # Ready to start
                        self._render_start_stop_buttons(stage_proc, fs_proc, app.event_handlers)

        imgui.spacing()

        # Switch to Expert Mode link
        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
        imgui.text_wrapped("Need more control?")
        imgui.pop_style_color()
        if imgui.button("Switch to Expert Mode", width=-1):
            app_state.ui_view_mode = "expert"
            app.logger.info("Switched to Expert Mode", extra={"status_message": True})

        imgui.end()

    def _get_simple_tracker_description(self, tracker_name):
        """Get a simple, user-friendly description for a tracker using discovery system."""
        info = self.tracker_ui.discovery.get_tracker_info(tracker_name)
        if info and info.description:
            return info.description
        return "Analyzes motion in your video"

    def _get_category_accent_color(self, category):
        """Return RGBA accent color for a tracker category."""
        if category == TrackerCategory.OFFLINE:
            return (0.3, 0.5, 0.9, 1.0)   # Blue
        elif category == TrackerCategory.LIVE:
            return (0.3, 0.8, 0.4, 1.0)   # Green
        elif category == TrackerCategory.LIVE_INTERVENTION:
            return (0.9, 0.6, 0.2, 1.0)   # Orange
        return (0.5, 0.5, 0.5, 1.0)       # Gray fallback

    def _handle_tracker_card_click(self, info):
        """Handle tracker card click — update selection and persist setting."""
        app = self.app
        app_state = app.app_state_ui
        self._user_manually_picked_tracker = True
        if app_state.selected_tracker_name != info.internal_name:
            if hasattr(app, 'logger') and app.logger:
                app.logger.info(f"UI(Simple): Tracker changed to {info.internal_name}")
            if hasattr(app, 'clear_all_overlays_and_ui_drawings'):
                app.clear_all_overlays_and_ui_drawings()
        app_state.selected_tracker_name = info.internal_name
        if hasattr(app, 'app_settings') and hasattr(app.app_settings, 'set'):
            app.app_settings.set("selected_tracker_name", info.internal_name)

    def _render_simple_mode_tracker_selection(self):
        """Render card-based tracker selection for Simple Mode."""
        app = self.app
        app_state = app.app_state_ui
        stage_proc = app.stage_processor
        fs_proc = app.funscript_processor

        modes_display, modes_enum, _ = self._get_tracker_lists_for_ui(simple_mode=True)
        if not modes_enum:
            imgui.text_disabled("No trackers available")
            return

        # Ensure selected tracker is valid
        if app_state.selected_tracker_name not in modes_enum:
            if self._auto_recommended_tracker and self._auto_recommended_tracker in modes_enum:
                app_state.selected_tracker_name = self._auto_recommended_tracker
            else:
                from config.constants import DEFAULT_TRACKER_NAME
                app_state.selected_tracker_name = modes_enum[0] if modes_enum else DEFAULT_TRACKER_NAME

        # When analysis is running or results exist, only show the selected card
        is_busy = stage_proc.full_analysis_active
        has_results = bool(fs_proc.get_actions("primary"))
        if is_busy or has_results:
            from config.tracker_discovery import get_tracker_discovery
            discovery = get_tracker_discovery()
            info = discovery.get_tracker_info(app_state.selected_tracker_name)
            if info:
                is_rec = (self._auto_recommended_tracker == info.internal_name)
                self._render_tracker_card(info, True, is_rec)
            return

        # Group trackers: Offline first, then Live
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        offline_trackers = []
        live_trackers = []
        for internal_name in modes_enum:
            info = discovery.get_tracker_info(internal_name)
            if not info:
                continue
            if info.category == TrackerCategory.OFFLINE:
                offline_trackers.append(info)
            else:
                live_trackers.append(info)

        # Calculate scrollable region height (65px per card, max ~260px)
        total_cards = len(offline_trackers) + len(live_trackers)
        card_height = 65
        region_height = min(total_cards * card_height + 40, 300)
        imgui.begin_child("##TrackerCardRegion", width=0, height=region_height, border=False)

        if offline_trackers:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.6, 0.8, 1.0)
            imgui.text("OFFLINE ANALYSIS")
            imgui.pop_style_color()
            imgui.spacing()
            for info in offline_trackers:
                is_selected = (app_state.selected_tracker_name == info.internal_name)
                is_rec = (self._auto_recommended_tracker == info.internal_name)
                if self._render_tracker_card(info, is_selected, is_rec):
                    self._handle_tracker_card_click(info)

        if live_trackers:
            imgui.spacing()
            imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.7, 0.5, 1.0)
            imgui.text("LIVE TRACKING")
            imgui.pop_style_color()
            imgui.spacing()
            for info in live_trackers:
                is_selected = (app_state.selected_tracker_name == info.internal_name)
                is_rec = (self._auto_recommended_tracker == info.internal_name)
                if self._render_tracker_card(info, is_selected, is_rec):
                    self._handle_tracker_card_click(info)

        imgui.end_child()

    def _render_tracker_card(self, info, is_selected, is_recommended):
        """Render a single tracker card. Returns True if clicked."""
        card_height = 60
        avail_w = imgui.get_content_region_available_width()
        accent = self._get_category_accent_color(info.category)

        # Background color for selected card
        if is_selected:
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.2, 0.3, 0.4, 0.6)
        else:
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.15, 0.15, 0.15, 0.4)

        imgui.begin_child(
            "##Card_%s" % info.internal_name,
            width=avail_w, height=card_height,
            border=True,
        )

        # Draw colored accent bar on the left
        draw_list = imgui.get_window_draw_list()
        pos = imgui.get_cursor_screen_position()
        bar_width = 4
        draw_list.add_rect_filled(
            pos[0] - 4, pos[1] - 4,
            pos[0] - 4 + bar_width, pos[1] + card_height - 8,
            imgui.get_color_u32_rgba(*accent),
        )

        # Indent past the accent bar
        imgui.indent(6)

        # Category badge + display name on first line
        badge_text = "[OFFLINE]" if info.category == TrackerCategory.OFFLINE else "[LIVE]"
        imgui.push_style_color(imgui.COLOR_TEXT, *accent)
        imgui.text(badge_text)
        imgui.pop_style_color()
        imgui.same_line()

        # Display name (brighter when selected)
        if is_selected:
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 1.0)
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.85, 0.85, 0.85, 1.0)
        imgui.text(info.display_name)
        imgui.pop_style_color()

        # Recommended badge
        if is_recommended:
            imgui.same_line()
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.SUCCESS_TEXT)
            imgui.text("[Recommended]")
            imgui.pop_style_color()

        # Description (gray, second line)
        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
        imgui.text_wrapped(info.description if info.description else "")
        imgui.pop_style_color()

        imgui.unindent(6)

        # Click detection
        clicked = imgui.is_window_hovered() and imgui.is_mouse_clicked(0)

        # Tooltip for recommended
        if is_recommended and imgui.is_window_hovered() and self._auto_recommendation_reason:
            imgui.set_tooltip("Recommended: %s" % self._auto_recommendation_reason)

        imgui.end_child()
        imgui.pop_style_color()  # CHILD_BACKGROUND

        return clicked

    def _render_simple_mode_post_processing_prompt(self):
        """Render optional post-processing prompt after analysis completion."""
        app = self.app
        fs_proc = app.funscript_processor

        imgui.spacing()
        if self._simple_mode_post_processing_applied:
            # Already applied - show status
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.SUCCESS_TEXT)
            imgui.text("Results polished")
            imgui.pop_style_color()
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
            imgui.text_wrapped("Optional: Improve results with automatic smoothing and optimization")
            imgui.pop_style_color()
            imgui.spacing()
            with primary_button_style():
                if imgui.button("Polish Results", width=-1):
                    # Temporarily enable auto post-processing, apply, then restore
                    original_setting = app.app_settings.get("enable_auto_post_processing", False)
                    app.app_settings.set("enable_auto_post_processing", True)
                    try:
                        fs_proc.apply_automatic_post_processing()
                        self._simple_mode_post_processing_applied = True
                        app.logger.info("Simple Mode: Post-processing applied", extra={"status_message": True})
                    except Exception as e:
                        app.logger.error("Post-processing failed: %s" % e, extra={"status_message": True})
                    finally:
                        app.app_settings.set("enable_auto_post_processing", original_setting)
            _tooltip_if_hovered(
                "Applies smoothing, simplification, clamping, and amplitude\n"
                "optimization to improve the generated funscript quality."
            )
        imgui.spacing()

    def _get_tracker_num_stages(self, tracker_name):
        """Get the number of processing stages for a tracker."""
        if not self.tracker_ui:
            return 1
        info = self.tracker_ui.discovery.get_tracker_info(tracker_name)
        if info and info.properties:
            n = info.properties.get("num_stages")
            if n:
                return n
        if info and info.stages:
            return len(info.stages)
        return 1

    def _get_friendly_stage_label(self, tracker_name, stage_number):
        """Get a beginner-friendly label for a processing stage."""
        if not self.tracker_ui:
            return "Processing"
        info = self.tracker_ui.discovery.get_tracker_info(tracker_name)
        if info and info.stages:
            for stage_def in info.stages:
                if stage_def.stage_number == stage_number:
                    name = stage_def.name.lower()
                    if "detection" in name or "detect" in name:
                        return "Scanning video"
                    if "segmentation" in name or "contact" in name:
                        return "Analyzing scenes"
                    if "optical flow" in name or "funscript" in name or "motion" in name:
                        return "Generating motion data"
                    return stage_def.name
        return "Processing"

    def _render_simple_progress_display(self):
        """Render consolidated progress display for Simple Mode."""
        app = self.app
        stage_proc = app.stage_processor
        tracker_name = app.app_state_ui.selected_tracker_name

        num_stages = self._get_tracker_num_stages(tracker_name)
        current_stage = stage_proc.current_analysis_stage

        # Header
        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.STATUS_INFO)
        imgui.text("Analyzing...")
        imgui.pop_style_color()

        # Get current stage progress, FPS, and ETA
        _stage_metrics = {
            1: (stage_proc.stage1_progress_value, stage_proc.stage1_processing_fps_str, stage_proc.stage1_eta_str),
            2: (stage_proc.stage2_main_progress_value, stage_proc.stage2_sub_processing_fps_str or "", stage_proc.stage2_sub_eta_str or "N/A"),
            3: (stage_proc.stage3_overall_progress_value, stage_proc.stage3_processing_fps_str, stage_proc.stage3_eta_str),
        }
        stage_progress, fps_str, eta_str = _stage_metrics.get(current_stage, (0.0, "", "N/A"))

        # Friendly stage label + step counter
        if num_stages > 1 and current_stage > 0:
            label = self._get_friendly_stage_label(tracker_name, current_stage)
            imgui.text("%s  (Step %d of %d)" % (label, current_stage, num_stages))
        else:
            imgui.text("Processing")

        # Compute overall progress: weight each stage equally
        if num_stages > 1 and current_stage > 0:
            completed_stages = max(0, current_stage - 1)
            weight = 1.0 / num_stages
            overall = completed_stages * weight + stage_progress * weight
        else:
            overall = stage_progress

        overall = max(0.0, min(1.0, overall))

        # Single progress bar with percentage overlay
        imgui.progress_bar(overall, (-1, 0), "%.0f%%" % (overall * 100))

        # FPS + ETA line
        status_parts = []
        if fps_str and fps_str != "0 FPS":
            status_parts.append(fps_str)
        if overall > 0.01 and eta_str and eta_str != "N/A":
            status_parts.append("ETA: %s" % eta_str)
        if status_parts:
            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
            imgui.text(" | ".join(status_parts))
            imgui.pop_style_color()

        imgui.spacing()

    def _render_processing_speed_controls(self, app_state):
        app = self.app
        processor = app.processor
        selected_mode = app_state.selected_tracker_name

        # Always show processing speed controls as they affect basic video playback
        # Check if current tracker is a live mode for tooltip information
        from config.tracker_discovery import get_tracker_discovery, TrackerCategory
        discovery = get_tracker_discovery()
        tracker_info = discovery.get_tracker_info(selected_mode)
        is_live_mode = tracker_info and tracker_info.category in [TrackerCategory.LIVE, TrackerCategory.LIVE_INTERVENTION]

        # Update tooltip based on context
        if is_live_mode:
            tooltip = "Control the processing speed for live analysis and video playback"
        else:
            tooltip = "Control the video playback speed"

        # Processing Speed section header removed
        current_speed_mode = app_state.selected_processing_speed_mode

        if imgui.radio_button("Real Time", current_speed_mode == config.ProcessingSpeedMode.REALTIME):
            app_state.selected_processing_speed_mode = config.ProcessingSpeedMode.REALTIME
        imgui.same_line()
        if imgui.radio_button("Slow-mo", current_speed_mode == config.ProcessingSpeedMode.SLOW_MOTION):
            app_state.selected_processing_speed_mode = config.ProcessingSpeedMode.SLOW_MOTION
        imgui.same_line()
        if imgui.radio_button("Max Speed", current_speed_mode == config.ProcessingSpeedMode.MAX_SPEED):
            app_state.selected_processing_speed_mode = config.ProcessingSpeedMode.MAX_SPEED
