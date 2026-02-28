"""Advanced Settings tab UI mixin for ControlPanelUI."""
import imgui
import os
import config
from application.utils import get_icon_texture_manager, destructive_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.section_card import section_card
from funscript.axis_registry import FunscriptAxis, AXIS_FILE_SUFFIX, AXIS_TCODE, file_suffix_for_axis, tcode_for_axis


def _tooltip_if_hovered(text):
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


def _readonly_input(label_id, value, width=-1):
    if width is not None and width >= 0:
        imgui.push_item_width(width)
    imgui.input_text(label_id, value or "Not set", 256, flags=imgui.INPUT_TEXT_READ_ONLY)
    if width is not None and width >= 0:
        imgui.pop_item_width()


class AdvancedSettingsMixin:
    """Mixin providing Advanced Settings tab rendering methods."""

    # ------- Model path updates -------

    def _update_detection_model_path(self, path):
        app = self.app
        tracker = app.tracker
        if not path or (tracker and path == tracker.det_model_path):
            return
        app.cached_class_names = None
        app.yolo_detection_model_path_setting = path
        app.app_settings.set("yolo_det_model_path", path)
        app.yolo_det_model_path = path
        app.project_manager.project_dirty = True
        app.logger.info("Detection model path updated to: %s. Reloading models." % path)
        if tracker:
            tracker.det_model_path = path
            tracker._load_models()

    def _update_pose_model_path(self, path):
        app = self.app
        tracker = app.tracker
        if not path or (tracker and path == tracker.pose_model_path):
            return
        app.cached_class_names = None
        app.yolo_pose_model_path_setting = path
        app.app_settings.set("yolo_pose_model_path", path)
        app.yolo_pose_model_path = path
        app.project_manager.project_dirty = True
        app.logger.info("Pose model path updated to: %s. Reloading models." % path)
        if tracker:
            tracker.pose_model_path = path
            tracker._load_models()

    def _update_artifacts_dir_path(self, path):
        app = self.app
        if not path or path == app.pose_model_artifacts_dir:
            return
        app.pose_model_artifacts_dir = path
        app.app_settings.set("pose_model_artifacts_dir", path)
        app.project_manager.project_dirty = True
        app.logger.info("Pose Model Artifacts directory updated to: %s." % path)

    # ------- Settings profiles -------

    def _render_settings_profiles_section(self):
        """Render settings profiles UI in the Advanced tab."""
        import time
        app = self.app
        settings = app.app_settings

        if imgui.collapsing_header("Settings Profiles##AdvancedProfiles", flags=0)[0]:
            # Cache profile list (refresh every 2 seconds)
            now = time.time()
            if self._profile_list_cache is None or (now - self._profile_list_cache_time) > 2.0:
                self._profile_list_cache = settings.list_profiles()
                self._profile_list_cache_time = now

            profiles = self._profile_list_cache
            profile_names = [p["name"] for p in profiles] if profiles else []

            # Load / Delete existing profile
            if profile_names:
                imgui.text("Load Profile:")
                imgui.push_item_width(imgui.get_content_region_available_width() - 140)
                # Use a simple combo for profile selection
                clicked, idx = imgui.combo("##ProfileCombo", self._selected_profile_idx, profile_names)
                if clicked:
                    self._selected_profile_idx = idx
                imgui.pop_item_width()

                imgui.same_line()
                sel_idx = min(self._selected_profile_idx, len(profile_names) - 1) if profile_names else 0
                sel_name = profile_names[sel_idx] if profile_names else ""
                if imgui.button("Load##LoadProfile"):
                    if sel_name and settings.load_profile(sel_name):
                        app.logger.info("Profile loaded: %s" % sel_name, extra={"status_message": True})
                        self._profile_list_cache = None  # Refresh cache
                imgui.same_line()
                with destructive_button_style():
                    if imgui.button("Delete##DeleteProfile"):
                        if sel_name:
                            imgui.open_popup("Confirm Delete Profile##DeleteProfilePopup")

                # Confirm delete popup
                if imgui.begin_popup_modal("Confirm Delete Profile##DeleteProfilePopup", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
                    imgui.text("Delete profile '%s'?" % sel_name)
                    avail_w = imgui.get_content_region_available_width()
                    pw = (avail_w - imgui.get_style().item_spacing[0]) / 2.0
                    with destructive_button_style():
                        if imgui.button("Delete##ConfirmDeleteProfile", width=pw):
                            if settings.delete_profile(sel_name):
                                app.logger.info("Profile deleted: %s" % sel_name, extra={"status_message": True})
                                self._profile_list_cache = None
                                self._selected_profile_idx = 0
                            imgui.close_current_popup()
                    imgui.same_line()
                    if imgui.button("Cancel##CancelDeleteProfile", width=pw):
                        imgui.close_current_popup()
                    imgui.end_popup()
            else:
                imgui.text_disabled("No saved profiles")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Save new profile
            imgui.text("Save Current Settings as Profile:")
            imgui.push_item_width(imgui.get_content_region_available_width() - 70)
            _, self._profile_name_input = imgui.input_text_with_hint(
                "##ProfileNameInput",
                "Profile name...",
                self._profile_name_input,
                128,
            )
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("Save##SaveProfile"):
                if self._profile_name_input.strip():
                    if settings.save_profile(self._profile_name_input):
                        app.logger.info("Profile saved: %s" % self._profile_name_input.strip(), extra={"status_message": True})
                        self._profile_name_input = ""
                        self._profile_list_cache = None  # Refresh cache
            if imgui.is_item_hovered():
                imgui.set_tooltip("Saves current processing settings as a reusable preset\n(tracking, post-processing, performance - not UI layout)")

    # ------- Advanced tab -------

    def _render_advanced_tab(self):
        """Render Advanced tab combining Configuration and Settings.

        Tier 1 (always visible): Settings Profiles, Processing, Output
        Tier 2 (behind toggle): Interface & Perf, Logging, Class Filtering,
                                 Live Tracker, Oscillation Detector
        """
        app = self.app
        app_state = app.app_state_ui
        tmode = app_state.selected_tracker_name

        imgui.text("Advanced settings for AI models, tracking, and performance.")
        imgui.spacing()

        # Show All Settings toggle (at top for visibility)
        _, self._show_all_advanced_settings = imgui.checkbox(
            "Show All Settings##AdvancedTier2Toggle",
            self._show_all_advanced_settings
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Show rarely changed settings like Interface, Logging, and Tracker configuration")

        imgui.spacing()

        # Settings Profiles section (Tier 1)
        self._render_settings_profiles_section()
        imgui.spacing()

        # Search box for filtering settings
        imgui.push_item_width(-1)
        _, self._advanced_search_query = imgui.input_text_with_hint(
            "##AdvancedSearch",
            "Search settings...",
            self._advanced_search_query,
            256
        )
        imgui.pop_item_width()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Filter settings by keyword")
        imgui.spacing()

        search_query = self._advanced_search_query.lower()
        # If user is searching, show all sections regardless of toggle
        show_all = self._show_all_advanced_settings or bool(search_query)

        # Define searchable keywords for each section (including sub-options)
        section_keywords = {
            "live_tracker": "live tracker settings roi detection optical flow confidence padding interval smoothing persistence sparse dis preset scale sensitivity amplification delay face hand class",
            "class_filter": "class filtering filter person face hand foot genitals body parts",
            "oscillation": "oscillation detector frequency amplitude threshold smoothing window peak valley timing",
            "interface": "interface performance gpu theme font scale dark light color vsync fps timeline rendering",
            "file_output": "file output save export path format funscript metadata json axis assignment tcode ofs naming",
            "logging": "logging autosave log debug verbose checkpoint interval backup"
        }

        # Helper to check if search matches section
        def matches_section(section_key):
            if not search_query:
                return True
            keywords = section_keywords.get(section_key, "")
            return any(term in keywords for term in search_query.split())

        # ---- Tier 1: Always visible settings ----

        # File & Output settings (Tier 1 - directly affects output quality)
        if matches_section("file_output"):
            _open_default = bool(search_query and matches_section("file_output"))
            with section_card("File & Output##AdvancedFileOutput",
                              tier="secondary", open_by_default=_open_default) as _open:
                if _open:
                    self._render_settings_file_output()

        # ---- Tier 2: Behind "Show All Settings" toggle ----

        if show_all:
            # Interface & Performance settings
            if matches_section("interface"):
                _open_default = bool(search_query and matches_section("interface"))
                with section_card("Interface & Performance##AdvancedInterfacePerf",
                                  tier="secondary", open_by_default=_open_default) as _open:
                    if _open:
                        self._render_settings_interface_perf()

            # Logging & Autosave settings
            if app_state.show_advanced_options:
                if matches_section("logging"):
                    _open_default = bool(search_query and matches_section("logging"))
                    with section_card("Logging & Autosave##AdvancedLogging",
                                      tier="secondary", open_by_default=_open_default) as _open:
                        if _open:
                            self._render_settings_logging_autosave()

            # Live Tracker Settings (dynamic dispatch)
            adv = app_state.show_advanced_options
            if self._is_live_tracker(tmode) and adv:
                if matches_section("live_tracker") or matches_section("oscillation"):
                    _open_default = bool(search_query and (matches_section("live_tracker") or matches_section("oscillation")))
                    with section_card("Live Tracker Settings##AdvancedLiveTracker",
                                      tier="secondary", open_by_default=_open_default) as _open:
                        if _open:
                            self._render_tracker_dynamic_settings()

            # Class filtering
            if (self._is_live_tracker(tmode) or self._is_offline_tracker(tmode)) and adv:
                if matches_section("class_filter"):
                    _open_default = bool(search_query and matches_section("class_filter"))
                    with section_card("Class Filtering##AdvancedClassFilter",
                                      tier="secondary", open_by_default=_open_default) as _open:
                        if _open:
                            self._render_class_filtering_content()

        imgui.spacing()

        # Reset All Settings button
        with destructive_button_style():
            if imgui.button("Reset All Settings to Default##ResetAllSettingsButton", width=-1):
                imgui.open_popup("Confirm Reset##ResetSettingsPopup")

        if imgui.begin_popup_modal(
            "Confirm Reset##ResetSettingsPopup", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )[0]:
            imgui.text(
                "This will reset all application settings to their defaults.\n"
                "Your projects will not be affected.\n"
                "This action cannot be undone."
            )

            avail_w = imgui.get_content_region_available_width()
            pw = (avail_w - imgui.get_style().item_spacing[0]) / 2.0

            # Confirm Reset button
            with destructive_button_style():
                if imgui.button("Confirm Reset", width=pw):
                    app.app_settings.reset_to_defaults()
                    app.logger.info("All settings have been reset to default.", extra={"status_message": True})
                    imgui.close_current_popup()

            imgui.same_line()
            if imgui.button("Cancel", width=pw):
                imgui.close_current_popup()
            imgui.end_popup()

    # ------- AI model settings -------

    def _render_ai_model_settings(self):
        app = self.app
        stage_proc = app.stage_processor
        settings = app.app_settings
        style = imgui.get_style()

        is_batch_mode = app.is_batch_processing_active
        is_analysis_running = stage_proc.full_analysis_active
        is_live_tracking_running = (app.processor and
                                    app.processor.is_processing and
                                    app.processor.enable_tracker_processing)
        is_setting_roi = app.is_setting_user_roi_mode
        is_any_process_active = is_batch_mode or is_analysis_running or is_live_tracking_running or is_setting_roi

        with _DisabledScope(is_any_process_active):
            def show_model_file_dialog(title, current_path, callback):
                gi = getattr(app, "gui_instance", None)
                if not gi:
                    return
                init_dir = os.path.dirname(current_path) if current_path else None
                gi.file_dialog.show(
                    title=title,
                    is_save=False,
                    callback=callback,
                    extension_filter=self.AI_modelExtensionsFilter,
                    initial_path=init_dir,
                )

            # Precompute widths
            tp = style.frame_padding.x * 2
            browse_w = imgui.calc_text_size("Browse").x + tp
            unload_w = imgui.calc_text_size("Unload").x + tp
            total_btn_w = browse_w + unload_w + style.item_spacing.x
            avail_w = imgui.get_content_region_available_width()
            input_w = avail_w - total_btn_w - style.item_spacing.x

            # Detection model
            imgui.text("Detection Model")
            _readonly_input("##S1YOLOPath", app.yolo_detection_model_path_setting, input_w)
            imgui.same_line()
            # Browse button with folder-open icon
            icon_mgr = get_icon_texture_manager()
            folder_open_tex, _, _ = icon_mgr.get_icon_texture('folder-open.png')
            btn_size = imgui.get_frame_height()
            if folder_open_tex and imgui.image_button(folder_open_tex, btn_size, btn_size):
                show_model_file_dialog(
                    "Select YOLO Detection Model",
                    app.yolo_detection_model_path_setting,
                    self._update_detection_model_path,
                )
            elif not folder_open_tex and imgui.button("Browse##S1YOLOBrowse"):
                show_model_file_dialog(
                    "Select YOLO Detection Model",
                    app.yolo_detection_model_path_setting,
                    self._update_detection_model_path,
                )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Browse for detection model file")
            imgui.same_line()
            # Unload button (DESTRUCTIVE - unloads model from memory)
            with destructive_button_style():
                if imgui.button("Unload##S1YOLOUnload"):
                    app.unload_model("detection")
            _tooltip_if_hovered("Path to the YOLO object detection model file (%s)." % self.AI_modelTooltipExtensions)

            # Pose model
            imgui.text("Pose Model")
            _readonly_input("##PoseYOLOPath", app.yolo_pose_model_path_setting, input_w)
            imgui.same_line()
            # Browse button with folder-open icon
            imgui.push_id("PoseYOLOBrowse")
            folder_open_tex, _, _ = icon_mgr.get_icon_texture('folder-open.png')
            if folder_open_tex and imgui.image_button(folder_open_tex, btn_size, btn_size):
                show_model_file_dialog(
                    "Select YOLO Pose Model",
                    app.yolo_pose_model_path_setting,
                    self._update_pose_model_path,
                )
            elif not folder_open_tex and imgui.button("Browse"):
                show_model_file_dialog(
                    "Select YOLO Pose Model",
                    app.yolo_pose_model_path_setting,
                    self._update_pose_model_path,
                )
            imgui.pop_id()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Browse for pose model file")
            imgui.same_line()
            # Unload button (DESTRUCTIVE - unloads model from memory)
            with destructive_button_style():
                if imgui.button("Unload##PoseYOLOUnload"):
                    app.unload_model("pose")
            _tooltip_if_hovered("Path to the YOLO pose estimation model file (%s). This model is optional." % self.AI_modelTooltipExtensions)

            imgui.text("Pose Model Artifacts Dir")
            dir_input_w = avail_w - browse_w - style.item_spacing.x if avail_w > browse_w else -1
            _readonly_input("##PoseArtifactsDirPath", app.pose_model_artifacts_dir, dir_input_w)
            imgui.same_line()
            # Browse button with folder-open icon
            imgui.push_id("PoseArtifactsDirBrowse")
            folder_open_tex, _, _ = icon_mgr.get_icon_texture('folder-open.png')
            if folder_open_tex and imgui.image_button(folder_open_tex, btn_size, btn_size):
                gi = getattr(app, "gui_instance", None)
                if gi:
                    gi.file_dialog.show(
                        title="Select Pose Model Artifacts Directory",
                        callback=self._update_artifacts_dir_path,
                        is_folder_dialog=True,
                        initial_path=app.pose_model_artifacts_dir,
                    )
            elif not folder_open_tex and imgui.button("Browse"):
                gi = getattr(app, "gui_instance", None)
                if gi:
                    gi.file_dialog.show(
                        title="Select Pose Model Artifacts Directory",
                        callback=self._update_artifacts_dir_path,
                        is_folder_dialog=True,
                        initial_path=app.pose_model_artifacts_dir,
                    )
            imgui.pop_id()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Browse for pose model artifacts directory")
            _tooltip_if_hovered(
                "Path to the folder containing your trained classifier,\n"
                "imputer, and other .joblib model artifacts."
            )

            mode = app.app_state_ui.selected_tracker_name
            if self._is_offline_tracker(mode):
                imgui.text("Stage 1 Inference Workers:")
                imgui.push_item_width(100)
                is_save_pre = getattr(stage_proc, "save_preprocessed_video", False)
                with _DisabledScope(is_save_pre):
                    ch_p, n_p = imgui.input_int(
                        "Producers##S1Producers", stage_proc.num_producers_stage1
                    )
                    if ch_p and not is_save_pre:
                        v = max(1, n_p)
                        if v != stage_proc.num_producers_stage1:
                            stage_proc.num_producers_stage1 = v
                            settings.set("num_producers_stage1", v)
                if is_save_pre:
                    _tooltip_if_hovered("Producers are forced to 1 when 'Save/Reuse Preprocessed Video' is enabled.")
                else:
                    _tooltip_if_hovered("Number of threads for video decoding & preprocessing.")

                imgui.same_line()
                ch_c, n_c = imgui.input_int("Consumers##S1Consumers", stage_proc.num_consumers_stage1)
                if ch_c:
                    v = max(1, n_c)
                    if v != stage_proc.num_consumers_stage1:
                        stage_proc.num_consumers_stage1 = v
                        settings.set("num_consumers_stage1", v)
                _tooltip_if_hovered("Number of threads for AI model inference. Match to available cores for best performance.")
                imgui.pop_item_width()

                imgui.text("Stage 2 OF Workers")
                imgui.same_line()
                imgui.push_item_width(120)
                cur_s2 = settings.get("num_workers_stage2_of", self.constants.DEFAULT_S2_OF_WORKERS)
                ch, new_s2 = imgui.input_int("##S2OFWorkers", cur_s2)
                if ch:
                    v = max(1, new_s2)
                    if v != cur_s2:
                        settings.set("num_workers_stage2_of", v)
                imgui.pop_item_width()
                _tooltip_if_hovered(
                    "Number of processes for Stage 2 Optical Flow gap recovery.\n"
                    "More may be faster on high-core CPUs."
                )

    # ------- Settings: interface/perf -------

    def _render_settings_interface_perf(self):
        app = self.app
        energy = app.energy_saver
        settings = app.app_settings

        imgui.text("Font Scale")
        imgui.same_line()
        imgui.push_item_width(120)
        labels = config.constants.FONT_SCALE_LABELS
        values = config.constants.FONT_SCALE_VALUES
        cur_val = settings.get("global_font_scale", config.constants.DEFAULT_FONT_SCALE)
        try:
            cur_idx = min(range(len(values)), key=lambda i: abs(values[i] - cur_val))
        except (ValueError, IndexError):
            cur_idx = 3
        ch, new_idx = imgui.combo("##GlobalFontScale", cur_idx, labels)
        if ch:
            nv = values[new_idx]
            if nv != cur_val:
                settings.set("global_font_scale", nv)
                # Disable auto system scaling when user manually changes font scale
                settings.set("auto_system_scaling_enabled", False)
                energy.reset_activity_timer()
        imgui.pop_item_width()
        _tooltip_if_hovered("Adjust the global UI font size. Applied instantly.")

        # Automatic system scaling option
        imgui.same_line()
        auto_scaling_enabled = settings.get("auto_system_scaling_enabled", True)
        ch, auto_scaling_enabled = imgui.checkbox("Auto System Scaling", auto_scaling_enabled)
        if ch:
            settings.set("auto_system_scaling_enabled", auto_scaling_enabled)
            if auto_scaling_enabled:
                # Apply system scaling immediately when enabled
                try:
                    from application.utils.system_scaling import apply_system_scaling_to_settings
                    scaling_applied = apply_system_scaling_to_settings(settings)
                    if scaling_applied:
                        app.logger.info("System scaling applied to application settings")
                        energy.reset_activity_timer()
                except Exception as e:
                    app.logger.warning(f"Failed to apply system scaling: {e}")
            else:
                app.logger.info("Automatic system scaling disabled")
        _tooltip_if_hovered("Automatically detect and apply system DPI/scaling settings at startup. "
                           "When enabled, the application will adjust the UI font size based on your "
                           "system's display scaling settings (e.g., 125%, 150%, etc.).")

        # Manual system scaling detection button
        if imgui.button("Detect System Scaling Now"):
            try:
                from application.utils.system_scaling import get_system_scaling_info, get_recommended_font_scale
                scaling_factor, dpi, platform_name = get_system_scaling_info()
                recommended_scale = get_recommended_font_scale(scaling_factor)
                current_scale = settings.get("global_font_scale", config.constants.DEFAULT_FONT_SCALE)

                app.logger.info(f"System scaling detected: {scaling_factor:.2f}x ({dpi:.0f} DPI on {platform_name})")
                app.logger.info(f"Recommended font scale: {recommended_scale} (current: {current_scale})")

                if abs(recommended_scale - current_scale) > 0.05:  # Only update if significantly different
                    settings.set("global_font_scale", recommended_scale)
                    # Disable auto system scaling when user manually detects scaling
                    settings.set("auto_system_scaling_enabled", False)
                    energy.reset_activity_timer()
                    app.logger.info(f"Font scale updated to {recommended_scale} based on system scaling")
                else:
                    app.logger.info("System scaling matches current font scale setting")
            except Exception as e:
                app.logger.warning(f"Failed to detect system scaling: {e}")
        _tooltip_if_hovered("Manually detect and apply current system DPI/scaling settings.")

        imgui.text("Timeline Pan Speed")
        imgui.same_line()
        imgui.push_item_width(120)
        cur_speed = settings.get("timeline_pan_speed_multiplier", config.constants.DEFAULT_TIMELINE_PAN_SPEED)
        ch, new_speed = imgui.slider_int("##TimelinePanSpeed", cur_speed, config.constants.TIMELINE_PAN_SPEED_MIN, config.constants.TIMELINE_PAN_SPEED_MAX)
        if ch and new_speed != cur_speed:
            settings.set("timeline_pan_speed_multiplier", new_speed)
        imgui.pop_item_width()
        _tooltip_if_hovered("Multiplier for keyboard-based timeline panning speed.")

        # --- Timeline Performance & GPU Settings ---
        imgui.text("Timeline Performance")

        # GPU Enable/Disable
        gpu_enabled = settings.get("timeline_gpu_enabled", False)
        changed, gpu_enabled = imgui.checkbox("Enable GPU Rendering##GPUTimeline", gpu_enabled)
        if changed:
            settings.set("timeline_gpu_enabled", gpu_enabled)
            app.energy_saver.reset_activity_timer()
            # Reinitialize GPU if being enabled
            if gpu_enabled and hasattr(app, '_initialize_gpu_timeline'):
                app._initialize_gpu_timeline()
            app.logger.info(f"GPU timeline rendering {'enabled' if gpu_enabled else 'disabled'}", extra={"status_message": True})
        _tooltip_if_hovered(
            "Enable GPU-accelerated timeline rendering for massive performance improvements.\n"
            "Best for datasets with 10,000+ points. Automatic fallback to CPU if GPU fails."
        )

        if gpu_enabled:
            imgui.text("GPU Threshold")
            imgui.same_line()
            imgui.push_item_width(120)
            gpu_threshold = settings.get("timeline_gpu_threshold_points", 5000)
            changed, gpu_threshold = imgui.input_int("##GPUThreshold", gpu_threshold)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Minimum points to use GPU rendering (higher = use CPU more often, lower = use GPU more often)")
            if changed:
                gpu_threshold = max(1000, min(100000, gpu_threshold))  # Clamp between 1k-100k
                settings.set("timeline_gpu_threshold_points", gpu_threshold)
            imgui.pop_item_width()
            _tooltip_if_hovered("Use GPU rendering when timeline has more than this many points")

        # Performance indicators
        show_perf = settings.get("show_timeline_optimization_indicator", False)
        changed, show_perf = imgui.checkbox("Show Performance Info##PerfIndicator", show_perf)
        if changed:
            settings.set("show_timeline_optimization_indicator", show_perf)
        _tooltip_if_hovered("Display performance indicators on timeline (render time, optimization modes)")

        # Performance stats (if GPU enabled and available)
        if gpu_enabled and hasattr(app, 'gpu_integration') and app.gpu_integration:
            try:
                stats = app.gpu_integration.get_performance_summary()
                imgui.text(f"GPU Backend: {stats.get('current_backend', 'Unknown')}")

                if 'gpu_details' in stats:
                    gpu_stats = stats['gpu_details']
                    render_time = gpu_stats.get('render_time_ms', 0)
                    points_rendered = gpu_stats.get('points_rendered', 0)
                    imgui.text(f"Last Render: {render_time:.2f}ms, {points_rendered:,} pts")

                    # Show GPU performance color coding
                    if render_time < 5.0:
                        imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 1.0, 0.0, 1.0)  # Green
                        imgui.text("Excellent Performance")
                    elif render_time < 16.67:  # 60fps threshold
                        imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 0.0, 1.0)  # Yellow
                        imgui.text("Good Performance")
                    else:
                        imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.5, 0.0, 1.0)  # Orange
                        imgui.text("High Load")
                    imgui.pop_style_color()
            except Exception as e:
                imgui.text_disabled(f"GPU stats unavailable: {str(e)[:30]}...")
        elif gpu_enabled:
            imgui.text_disabled("GPU not available - using CPU fallback")

        imgui.text("Video Decoding")
        imgui.same_line()
        imgui.push_item_width(180)
        opts = app.available_ffmpeg_hwaccels
        disp = [o.replace("videotoolbox", "VideoToolbox (macOS)") for o in opts]
        try:
            hw_idx = opts.index(app.hardware_acceleration_method)
        except ValueError:
            hw_idx = 0
        ch, nidx = imgui.combo("HW Acceleration##HWAccelMethod", hw_idx, disp)
        if ch:
            method = opts[nidx]
            if method != app.hardware_acceleration_method:
                app.hardware_acceleration_method = method
                settings.set("hardware_acceleration_method", method)
                app.logger.info("Hardware acceleration set to: %s. Reload video to apply." % method, extra={"status_message": True})
        imgui.pop_item_width()
        _tooltip_if_hovered("Select FFmpeg hardware acceleration. Requires video reload to apply.")

        imgui.text("Energy Saver Mode:")
        ch_es, v_es = imgui.checkbox("Enable##EnableES", energy.energy_saver_enabled)
        if ch_es and v_es != energy.energy_saver_enabled:
            energy.energy_saver_enabled = v_es
            settings.set("energy_saver_enabled", v_es)

        if energy.energy_saver_enabled:
            imgui.push_item_width(100)
            imgui.text("Normal FPS")
            imgui.same_line()
            nf = int(energy.main_loop_normal_fps_target)
            ch, val = imgui.input_int("##NormalFPS", nf)
            if ch:
                v = max(config.constants.ENERGY_SAVER_NORMAL_FPS_MIN, val)
                if v != nf:
                    energy.main_loop_normal_fps_target = v
                    settings.set("main_loop_normal_fps_target", v)

            imgui.text("Idle After (s)")
            imgui.same_line()
            th = int(energy.energy_saver_threshold_seconds)
            ch, val = imgui.input_int("##ESThreshold", th)
            if ch:
                v = float(max(config.constants.ENERGY_SAVER_THRESHOLD_MIN, val))
                if v != energy.energy_saver_threshold_seconds:
                    energy.energy_saver_threshold_seconds = v
                    settings.set("energy_saver_threshold_seconds", v)

            imgui.text("Idle FPS")
            imgui.same_line()
            ef = int(energy.energy_saver_fps)
            ch, val = imgui.input_int("##ESFPS", ef)
            if ch:
                v = max(config.constants.ENERGY_SAVER_IDLE_FPS_MIN, val)
                if v != ef:
                    energy.energy_saver_fps = v
                    settings.set("energy_saver_fps", v)
            imgui.pop_item_width()

    # ------- Settings: file/output -------

    def _render_settings_file_output(self):
        settings = self.app.app_settings

        imgui.text("Output Folder:")
        imgui.push_item_width(-1)
        cur = settings.get("output_folder_path", "output")
        ch, new_val = imgui.input_text("##OutputFolder", cur, 256)
        if ch and new_val != cur:
            settings.set("output_folder_path", new_val)
        imgui.pop_item_width()
        _tooltip_if_hovered("Root folder for all generated files (projects, analysis data, etc.).")

        imgui.text("Funscript Output:")
        ch, v = imgui.checkbox(
            "Autosave final script next to video",
            settings.get("autosave_final_funscript_to_video_location", True),
        )
        if ch:
            settings.set("autosave_final_funscript_to_video_location", v)

        ch, v = imgui.checkbox("Generate .roll file (from Timeline 2)", settings.get("generate_roll_file", True))
        if ch:
            settings.set("generate_roll_file", v)

        ch, v = imgui.checkbox("Export as .funscript (skip .raw prefix)", settings.get("export_raw_as_funscript", False))
        if ch:
            settings.set("export_raw_as_funscript", v)
        _tooltip_if_hovered("When no post-processing is applied, export directly as .funscript\ninstead of .raw.funscript next to the video file.")

        # Point simplification
        cur_simplify = settings.get("funscript_point_simplification_enabled", True)
        ch, nv_simplify = imgui.checkbox("On the fly funscript simplification##EnablePointSimplify", cur_simplify)
        if ch and nv_simplify != cur_simplify:
            settings.set("funscript_point_simplification_enabled", nv_simplify)
            # Apply to active funscript (used during live tracking)
            if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
                self.app.processor.tracker.funscript.enable_point_simplification = nv_simplify
                self.app.logger.info(f"Point simplification {'enabled' if nv_simplify else 'disabled'} for active funscript")
        _tooltip_if_hovered("Remove redundant points on-the-fly (collinear/flat sections)\nReduces file size by 50-80% with negligible CPU overhead")

        # Funscript export format
        imgui.spacing()
        imgui.text("Funscript Export Format:")
        _tooltip_if_hovered(
            "Separate Files (OFS): One .funscript per axis (standard OFS naming)\n"
            "Unified (embedded axes): All axes in a single .funscript file\n"
            "Both: Save both formats simultaneously")
        export_options = ["Separate Files (OFS)", "Unified (embedded axes)", "Both"]
        export_values = ["separate", "unified", "both"]
        cur_export = settings.get("funscript_export_format", "separate")
        try:
            cur_export_idx = export_values.index(cur_export)
        except ValueError:
            cur_export_idx = 0
        imgui.push_item_width(250)
        ch, new_export_idx = imgui.combo("##FunscriptExportFormat", cur_export_idx, export_options)
        imgui.pop_item_width()
        if ch and new_export_idx != cur_export_idx:
            settings.set("funscript_export_format", export_values[new_export_idx])
            self.app.logger.info(f"Funscript export format set to: {export_values[new_export_idx]}",
                                 extra={'status_message': True})

        imgui.spacing()
        imgui.text("Batch Processing Default:")
        cur = settings.get("batch_mode_overwrite_strategy", 0)
        if imgui.radio_button("Process All (skips own matching version)", cur == 0):
            if cur != 0:
                settings.set("batch_mode_overwrite_strategy", 0)
        if imgui.radio_button("Skip if Funscript Exists", cur == 1):
            if cur != 1:
                settings.set("batch_mode_overwrite_strategy", 1)

        # --- Axis Assignments Table ---
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.text("Axis Assignments (OFS Naming):")
        _tooltip_if_hovered("Maps each timeline to a semantic axis name.\nControls the file suffix (e.g. .roll.funscript) and TCode channel.")

        funscript_obj = None
        # TrackerManager.funscript is always available after app init
        if self.app.tracker and hasattr(self.app.tracker, 'funscript'):
            funscript_obj = self.app.tracker.funscript

        if funscript_obj:
            axis_names = [fa.value for fa in FunscriptAxis]
            assignments = funscript_obj.get_axis_assignments()

            # Table header
            imgui.columns(4, "##AxisAssignTable")
            imgui.separator()
            imgui.text("Timeline"); imgui.next_column()
            imgui.text("Axis"); imgui.next_column()
            imgui.text("File Suffix"); imgui.next_column()
            imgui.text("TCode"); imgui.next_column()
            imgui.separator()

            for tl_num in sorted(assignments.keys()):
                current_axis = assignments[tl_num]
                imgui.text(f"T{tl_num}"); imgui.next_column()

                # Combo dropdown for axis selection
                imgui.push_item_width(-1)
                try:
                    current_idx = axis_names.index(current_axis)
                except ValueError:
                    current_idx = -1  # Custom axis not in list

                display_items = axis_names if current_idx >= 0 else [current_axis] + axis_names
                adj_idx = current_idx if current_idx >= 0 else 0
                changed, new_idx = imgui.combo(f"##AxisCombo{tl_num}", adj_idx, display_items)
                if changed:
                    new_axis = display_items[new_idx]
                    funscript_obj.assign_axis(tl_num, new_axis)
                    self.app.project_manager.project_dirty = True
                imgui.pop_item_width()
                imgui.next_column()

                # Show file suffix
                suffix = file_suffix_for_axis(current_axis)
                imgui.text(f"{suffix}.funscript" if suffix else ".funscript"); imgui.next_column()

                # Show TCode
                tcode = tcode_for_axis(current_axis)
                imgui.text(tcode or "-"); imgui.next_column()

            imgui.columns(1)
            imgui.separator()
        else:
            imgui.text_colored("No funscript loaded.", 0.5, 0.5, 0.5, 1.0)

    # ------- Settings: logging/autosave -------

    def _render_settings_logging_autosave(self):
        app = self.app
        settings = app.app_settings

        imgui.text("Logging Level")
        imgui.same_line()
        imgui.push_item_width(150)
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        try:
            idx = levels.index(app.logging_level_setting.upper())
        except ValueError:
            idx = 1
        ch, nidx = imgui.combo("##LogLevel", idx, levels)
        if ch:
            new_level = levels[nidx]
            if new_level != app.logging_level_setting.upper():
                app.set_application_logging_level(new_level)
        imgui.pop_item_width()

        imgui.text("Project Autosave:")
        ch, v = imgui.checkbox(
            "Enable##EnableAutosave", settings.get("autosave_enabled", True)
        )
        if ch:
            settings.set("autosave_enabled", v)

        if settings.get("autosave_enabled"):
            imgui.push_item_width(100)
            imgui.text("Interval (s)")
            imgui.same_line()
            interval = settings.get("autosave_interval_seconds", 300)
            ch_int, new_interval = imgui.input_int("##AutosaveInterval", interval)
            if ch_int:
                nv = max(30, new_interval)
                if nv != interval:
                    settings.set("autosave_interval_seconds", nv)
            imgui.pop_item_width()
