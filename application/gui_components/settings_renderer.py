"""Standalone settings rendering module — shared across panels.

Owns all application settings UI and the Tracker tab rendering.
Not a mixin — takes `app` as constructor argument and owns its own state.
"""
import imgui
import time
import logging
import config
from application.utils import destructive_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope, tooltip_if_hovered as _tooltip_if_hovered
from application.utils.section_card import section_card
from application.utils.feature_detection import is_feature_available as _is_feature_available
from funscript.axis_registry import FunscriptAxis, file_suffix_for_axis, tcode_for_axis

_logger = logging.getLogger(__name__)

# Layout helpers — imported from shared module
from application.utils.imgui_layout_helpers import (
    begin_settings_columns as _begin_settings_columns,
    end_settings_columns as _end_settings_columns,
    row_label as _row_label,
    row_end as _row_end,
    row_separator as _row_separator,
)


# ------------------------------------------------------------------ #
#  Standalone tracker helpers                                         #
# ------------------------------------------------------------------ #

def _is_offline_tracker(tracker_name):
    try:
        from config.tracker_discovery import get_tracker_discovery, TrackerCategory
        info = get_tracker_discovery().get_tracker_info(tracker_name)
        return info and info.category == TrackerCategory.OFFLINE
    except Exception:
        return False


def _is_hybrid_tracker(tracker_name):
    try:
        from config.tracker_discovery import get_tracker_discovery
        info = get_tracker_discovery().get_tracker_info(tracker_name)
        return info and info.properties and info.properties.get('hybrid', False)
    except Exception:
        return False


def _get_current_tracker_instance(app):
    tr = getattr(app, 'tracker', None)
    if tr and hasattr(tr, '_current_tracker'):
        return tr._current_tracker
    return None


# ================================================================== #
#  SettingsRenderer                                                   #
# ================================================================== #

class SettingsRenderer:
    """Standalone settings renderer.

    Owns the Settings tab content and the Tracker tab content.
    """

    def __init__(self, app):
        self.app = app
        # Search state
        self._search_query = ""
        # Profile state
        self._profile_name_input = ""
        self._profile_list_cache = None
        self._profile_list_cache_time = 0
        self._selected_profile_idx = 0
        # Lazy-instantiated tracker cache for "all trackers" section
        self._tracker_instances = {}  # {internal_name: instance}
        self._tracker_schemas = {}    # {internal_name: schema_dict or None}

    def cleanup(self):
        """Release lazy-instantiated tracker instances."""
        for inst in self._tracker_instances.values():
            try:
                inst.cleanup()
            except Exception:
                pass
        self._tracker_instances.clear()
        self._tracker_schemas.clear()

    # ================================================================ #
    #  Tracker tab (active tracker)                                    #
    # ================================================================ #

    def render_tracker_tab(self):
        """Render the dedicated Tracker tab — active tracker only."""
        app = self.app
        tracker_inst = _get_current_tracker_instance(app)
        tmode = app.app_state_ui.selected_tracker_name

        if not tracker_inst:
            imgui.text_disabled("No tracker active.")
            imgui.spacing()
            imgui.text_disabled("Select and start a tracker from the Control Panel.")
            return

        # Header — active tracker name
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        info = discovery.get_tracker_info(tmode)
        display_name = info.display_name if info else tmode
        imgui.text_colored(display_name, 0.4, 0.8, 1.0, 1.0)
        if info and info.description:
            imgui.text_wrapped(info.description)
        imgui.spacing()

        # Tracker Settings
        with section_card("Settings##ActiveTrackerSettings", tier="primary") as s_open:
            if s_open:
                self._render_tracker_settings_ui(tracker_inst)

        # Class Filtering
        if getattr(tracker_inst, 'uses_class_detection', False):
            with section_card("Class Filtering##ActiveTrackerClassFilter", tier="primary") as cf_open:
                if cf_open:
                    self._render_class_filtering(app)

        # Debug
        self._render_tracker_debug(tracker_inst)

    # ================================================================ #
    #  Settings tab                                                    #
    # ================================================================ #

    def render(self):
        """Render the full Settings tab content."""
        app = self.app
        tmode = app.app_state_ui.selected_tracker_name

        # Search box
        imgui.push_item_width(-1)
        _, self._search_query = imgui.input_text_with_hint(
            "##SettingsSearch", "Search settings...", self._search_query, 256
        )
        imgui.pop_item_width()
        imgui.spacing()

        sq = self._search_query.lower()

        kw = {
            "profiles": "profiles preset save load delete",
            "interface": "interface font scale timeline pan speed performance overlay",
            "video": "video hw acceleration hardware decoding videotoolbox",
            "energy": "energy saver idle fps timeout power performance",
            "logging": "logging autosave log debug verbose interval",
            "output": "file output save export path format funscript axis assignment tcode ofs batch overwrite simplification",
            "processing": "analysis stage rerun force preprocessed video database cache output delay workers producers consumers",
            "api": "api websocket ws external tools control port",
            "trackers": "tracker settings detection optical flow confidence class filtering",
        }

        def matches(key):
            if not sq:
                return True
            return any(t in kw.get(key, "") for t in sq.split())

        def filtered(label, keys, fn, guard=True, accent=None):
            if not guard:
                return
            matched = any(matches(k) for k in keys)
            if not matched:
                return
            with section_card(label, tier="primary",
                              accent_color=accent,
                              open_by_default=bool(sq and matched)) as o:
                if o:
                    fn()

        # --- Profiles ---
        if matches("profiles"):
            self._render_profiles()
            imgui.spacing()

        # --- Interface ---
        filtered("Interface##SettingsInterface", ["interface"], self._render_interface)

        # --- Energy & Performance ---
        filtered("Energy & Performance##SettingsEnergy", ["energy"], self._render_energy_perf)

        # --- Subtitles (optional add-on) ---
        if _is_feature_available("subtitle_translation"):
            filtered("Subtitles##SettingsSubtitles", ["subtitle", "translation"], self._render_subtitle_settings)

        # --- Logging ---
        filtered("Logging##SettingsLogging", ["logging"], self._render_logging)

        # --- Output ---
        filtered("Output##SettingsOutput", ["output"], self._render_output)

        # --- WebSocket API ---
        filtered("WebSocket API##SettingsWSAPI", ["api", "websocket", "ws"], self._render_ws_api)

        # --- Processing ---
        filtered("Processing##SettingsProcessing", ["processing"],
                 lambda: self._render_processing(tmode))

        # --- All Trackers ---
        if matches("trackers"):
            with section_card("Trackers##SettingsAllTrackers", tier="primary",
                              open_by_default=False) as t_open:
                if t_open:
                    self._render_all_trackers()

        imgui.spacing()

        # --- Reset All ---
        with destructive_button_style():
            if imgui.button("Reset All Settings to Default##ResetAllSettings", width=-1):
                imgui.open_popup("Confirm Reset##ResetSettingsPopup")

        if imgui.begin_popup_modal("Confirm Reset##ResetSettingsPopup", True,
                                   imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            imgui.text("This will reset all application settings to their defaults.\n"
                       "Your projects will not be affected.\nThis action cannot be undone.")
            aw = imgui.get_content_region_available_width()
            pw = (aw - imgui.get_style().item_spacing[0]) / 2.0
            with destructive_button_style():
                if imgui.button("Confirm Reset", width=pw):
                    app.app_settings.reset_to_defaults()
                    app.logger.info("All settings have been reset to default.",
                                    extra={"status_message": True})
                    imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", width=pw):
                imgui.close_current_popup()
            imgui.end_popup()

    # ================================================================ #
    #  Profiles                                                        #
    # ================================================================ #

    def _render_profiles(self):
        app = self.app
        settings = app.app_settings

        with section_card("Settings Profiles##SettingsProfiles", tier="primary") as _open:
            if not _open:
                return

            now = time.time()
            if self._profile_list_cache is None or (now - self._profile_list_cache_time) > 30.0:
                self._profile_list_cache = settings.list_profiles()
                self._profile_list_cache_time = now

            profiles = self._profile_list_cache
            names = [p["name"] for p in profiles] if profiles else []

            if names:
                _begin_settings_columns("profile_cols")
                _row_label("Load Profile")
                imgui.push_item_width(imgui.get_content_region_available_width() - 120)
                clicked, idx = imgui.combo("##ProfileCombo", self._selected_profile_idx, names)
                if clicked:
                    self._selected_profile_idx = idx
                imgui.pop_item_width()

                sel_idx = min(self._selected_profile_idx, len(names) - 1)
                sel_name = names[sel_idx] if names else ""
                imgui.same_line()
                if imgui.button("Load##LP"):
                    if sel_name and settings.load_profile(sel_name):
                        app.logger.info("Profile loaded: %s" % sel_name, extra={"status_message": True})
                        app.notify("Profile loaded: %s" % sel_name, "success", 2.0)
                        self._profile_list_cache = None
                imgui.same_line()
                with destructive_button_style():
                    if imgui.button("Del##DP"):
                        if sel_name:
                            imgui.open_popup("Confirm Delete Profile##DPP")

                if imgui.begin_popup_modal("Confirm Delete Profile##DPP", True,
                                           imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
                    imgui.text("Delete profile '%s'?" % sel_name)
                    aw = imgui.get_content_region_available_width()
                    pw = (aw - imgui.get_style().item_spacing[0]) / 2.0
                    with destructive_button_style():
                        if imgui.button("Delete##CDP", width=pw):
                            if settings.delete_profile(sel_name):
                                app.logger.info("Profile deleted: %s" % sel_name,
                                                extra={"status_message": True})
                                app.notify("Profile deleted: %s" % sel_name, "info", 2.0)
                                self._profile_list_cache = None
                                self._selected_profile_idx = 0
                            imgui.close_current_popup()
                    imgui.same_line()
                    if imgui.button("Cancel##CDP", width=pw):
                        imgui.close_current_popup()
                    imgui.end_popup()
                _row_end()
                _end_settings_columns()
            else:
                imgui.text_disabled("No saved profiles")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _begin_settings_columns("profile_save_cols")
            _row_label("Save as Profile")
            imgui.push_item_width(imgui.get_content_region_available_width() - 60)
            _, self._profile_name_input = imgui.input_text_with_hint(
                "##PNI", "Profile name...", self._profile_name_input, 128)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("Save##SP"):
                if self._profile_name_input.strip():
                    if settings.save_profile(self._profile_name_input):
                        app.logger.info("Profile saved: %s" % self._profile_name_input.strip(),
                                        extra={"status_message": True})
                        app.notify("Profile saved: %s" % self._profile_name_input.strip(), "success", 2.0)
                        self._profile_name_input = ""
                        self._profile_list_cache = None
            _tooltip_if_hovered("Saves current processing settings as a reusable preset\n"
                                "(tracking, post-processing, performance — not UI layout)")
            _row_end()
            _end_settings_columns()

    # ================================================================ #
    #  Interface                                                       #
    # ================================================================ #

    def _render_interface(self):
        app = self.app
        settings = app.app_settings

        _begin_settings_columns("iface_cols")

        # Font Scale
        _row_label("Font Scale", "Adjust the global UI font size. Applied instantly.")
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
                settings.set("auto_system_scaling_enabled", False)
                app.energy_saver.reset_activity_timer()
        imgui.pop_item_width()
        imgui.same_line()
        auto_sc = settings.get("auto_system_scaling_enabled", True)
        ch, auto_sc = imgui.checkbox("Auto##AutoScaling", auto_sc)
        if ch:
            settings.set("auto_system_scaling_enabled", auto_sc)
            if auto_sc:
                try:
                    from application.utils.system_scaling import apply_system_scaling_to_settings
                    if apply_system_scaling_to_settings(settings):
                        app.energy_saver.reset_activity_timer()
                except Exception:
                    pass
        _tooltip_if_hovered("Auto-detect and apply system DPI scaling at startup.")
        _row_end()

        # Timeline Pan Speed
        _row_label("Timeline Pan Speed", "Multiplier for keyboard-based timeline panning speed.")
        imgui.push_item_width(-1)
        cur_speed = settings.get("timeline_pan_speed_multiplier", config.constants.DEFAULT_TIMELINE_PAN_SPEED)
        ch, new_speed = imgui.slider_int("##TPS", cur_speed,
                                         config.constants.TIMELINE_PAN_SPEED_MIN,
                                         config.constants.TIMELINE_PAN_SPEED_MAX,
                                         format="%dx")
        if ch and new_speed != cur_speed:
            settings.set("timeline_pan_speed_multiplier", new_speed)
        imgui.pop_item_width()
        _row_end()

        # Timeline perf overlay
        _row_label("Timeline Perf Overlay", "Show render time and optimization mode overlay on timeline editors.")
        ch, v = imgui.checkbox("Show##PerfInd", settings.get("show_timeline_optimization_indicator", False))
        if ch:
            settings.set("show_timeline_optimization_indicator", v)
        _row_end()

        _end_settings_columns()

    # ================================================================ #
    #  Energy & Performance                                            #
    # ================================================================ #

    def _render_energy_perf(self):
        app = self.app
        energy = app.energy_saver
        settings = app.app_settings

        _begin_settings_columns("energy_cols")

        _row_label("Energy Saver", "Reduce rendering FPS when the app is idle to save power and CPU.")
        ch, v = imgui.checkbox("Enabled##ES", energy.energy_saver_enabled)
        if ch and v != energy.energy_saver_enabled:
            energy.energy_saver_enabled = v
            settings.set("energy_saver_enabled", v)
        _row_end()

        if energy.energy_saver_enabled:
            for label, tooltip, imgui_id, attr, min_val, unit in [
                ("  Normal FPS", "Target FPS during normal (active) usage.", "##NFPS",
                 "main_loop_normal_fps_target", config.constants.ENERGY_SAVER_NORMAL_FPS_MIN, " fps"),
                ("  Idle Timeout", "Seconds of inactivity before switching to idle FPS.", "##EST",
                 "energy_saver_threshold_seconds", config.constants.ENERGY_SAVER_THRESHOLD_MIN, " sec"),
                ("  Idle FPS", "Target FPS when the app is idle.", "##EFPS",
                 "energy_saver_fps", config.constants.ENERGY_SAVER_IDLE_FPS_MIN, " fps"),
            ]:
                _row_label(label, tooltip)
                imgui.push_item_width(80)
                cur = int(getattr(energy, attr))
                ch, val = imgui.input_int(imgui_id, cur)
                if ch:
                    v = max(min_val, val)
                    if v != cur:
                        setattr(energy, attr, float(v) if "threshold" in attr else v)
                        settings.set(attr, float(v) if "threshold" in attr else v)
                imgui.pop_item_width()
                imgui.same_line()
                imgui.text_disabled(unit)
                _row_end()

        _end_settings_columns()

    # ================================================================ #
    #  Logging                                                         #
    # ================================================================ #

    def _render_subtitle_settings(self):
        settings = self.app.app_settings

        _begin_settings_columns("sub_cols")

        # LLM model size
        _row_label("Translation Quality", "LLM model size for translation.\nLarger = better quality but more RAM.")
        sizes = ["Good (3B LLM, ~2GB)", "Best (7B Uncensored, ~4GB)"]
        cur_size = settings.get("subtitle_llm_size", "large")
        cur_idx = {"medium": 0, "large": 1}.get(cur_size, 1)
        imgui.push_item_width(-1)
        ch, new_idx = imgui.combo("##SubQuality", cur_idx, sizes)
        imgui.pop_item_width()
        if ch:
            settings.set("subtitle_llm_size", ["medium", "large"][new_idx])
        _row_end()

        _end_settings_columns()

    # ================================================================ #

    def _render_logging(self):
        app = self.app
        settings = app.app_settings

        _begin_settings_columns("log_cols")

        _row_label("Logging Level", "Controls the verbosity of console and file logs.")
        imgui.push_item_width(-1)
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        try:
            idx = levels.index(app.logging_level_setting.upper())
        except ValueError:
            idx = 1
        ch, nidx = imgui.combo("##LogLvl", idx, levels)
        if ch:
            nl = levels[nidx]
            if nl != app.logging_level_setting.upper():
                app.set_application_logging_level(nl)
        imgui.pop_item_width()
        _row_end()

        _row_label("Autosave Project", "Automatically save the project at regular intervals.")
        ch, v = imgui.checkbox("Enabled##AutoS", settings.get("autosave_enabled", True))
        if ch:
            settings.set("autosave_enabled", v)
        _row_end()

        if settings.get("autosave_enabled"):
            _row_label("  Autosave Interval", "Time between automatic project saves.")
            imgui.push_item_width(80)
            interval = settings.get("autosave_interval_seconds", 300)
            ch, ni = imgui.input_int("##ASI", interval)
            if ch:
                nv = max(30, ni)
                if nv != interval:
                    settings.set("autosave_interval_seconds", nv)
            imgui.pop_item_width()
            imgui.same_line()
            imgui.text_disabled(" sec")
            _row_end()

        _end_settings_columns()

    # ================================================================ #
    #  WebSocket API                                                   #
    # ================================================================ #

    def _render_ws_api(self):
        settings = self.app.app_settings
        _begin_settings_columns("wsapi_cols")

        _row_label("WebSocket API", "Allow external tools to control FunGen via WebSocket (localhost only).")
        ch, v = imgui.checkbox("Enabled##WSAPI", settings.get("ws_api_enabled", False))
        if ch:
            settings.set("ws_api_enabled", v)
            if v and not getattr(self.app, '_ws_api', None):
                try:
                    from common.ws_api import FunGenWSAPI
                    port = settings.get("ws_api_port", 8769)
                    self.app._ws_api = FunGenWSAPI(self.app, port=port)
                    self.app._ws_api.start()
                except Exception as e:
                    self.app.logger.warning(f"WS API start failed: {e}")
            elif not v and getattr(self.app, '_ws_api', None):
                self.app._ws_api.stop()
                self.app._ws_api = None
        _row_end()

        if settings.get("ws_api_enabled", False):
            _row_label("  Port", "WebSocket port (requires restart to change).")
            imgui.push_item_width(80)
            port = settings.get("ws_api_port", 8769)
            ch, np_ = imgui.input_int("##WSPort", port)
            if ch:
                nv = max(1024, min(65535, np_))
                if nv != port:
                    settings.set("ws_api_port", nv)
            imgui.pop_item_width()
            _row_end()

            api = getattr(self.app, '_ws_api', None)
            if api and api._running:
                _row_label("  Status", "")
                imgui.text_colored("Active", 0.3, 0.9, 0.3, 1.0)
                imgui.same_line()
                imgui.text_disabled(f"ws://127.0.0.1:{settings.get('ws_api_port', 8769)}")
                _row_end()

        _end_settings_columns()

    # ================================================================ #
    #  Output                                                          #
    # ================================================================ #

    def _render_output(self):
        settings = self.app.app_settings

        _begin_settings_columns("output_cols")

        _row_label("Output Folder", "Root folder for all generated files.")
        imgui.push_item_width(-1)
        cur = settings.get("output_folder_path", "output")
        ch, nv = imgui.input_text("##OutDir", cur, 256)
        if ch and nv != cur:
            settings.set("output_folder_path", nv)
        imgui.pop_item_width()
        _row_end()

        _row_label("Save Next to Video", "Automatically save a copy of the final .funscript next to the video file.")
        ch, v = imgui.checkbox("Enabled##AutoSaveVid",
                               settings.get("autosave_final_funscript_to_video_location", True))
        if ch:
            settings.set("autosave_final_funscript_to_video_location", v)
        _row_end()

        sec_axis = settings.get("default_secondary_axis", "roll")
        _row_label(f"Generate .{sec_axis} File", f"Generate a separate .{sec_axis}.funscript file from Timeline 2.")
        ch, v = imgui.checkbox("Enabled##GenRoll", settings.get("generate_roll_file", True))
        if ch:
            settings.set("generate_roll_file", v)
        _row_end()

        _row_label("Skip .raw Prefix", "Export directly as .funscript instead of .raw.funscript\nwhen no post-processing is applied.")
        ch, v = imgui.checkbox("Enabled##ExpRaw", settings.get("export_raw_as_funscript", False))
        if ch:
            settings.set("export_raw_as_funscript", v)
        _row_end()

        _row_label("Point Simplification", "Remove redundant collinear/flat points on-the-fly.\nReduces file size by 50-80% with negligible CPU overhead.")
        cur_s = settings.get("funscript_point_simplification_enabled", True)
        ch, nv = imgui.checkbox("Enabled##PtSimp", cur_s)
        if ch and nv != cur_s:
            settings.set("funscript_point_simplification_enabled", nv)
            if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
                self.app.processor.tracker.funscript.enable_point_simplification = nv
        _row_end()

        _row_label("Export Format", "Separate Files (per-axis): one .funscript per axis\n"
                    "Unified: all axes embedded in a single file\n"
                    "Both: save both formats simultaneously")
        imgui.push_item_width(-1)
        export_opts = ["Separate Files (per-axis)", "Unified (embedded axes)", "Both"]
        export_vals = ["separate", "unified", "both"]
        cur_exp = settings.get("funscript_export_format", "separate")
        try:
            ei = export_vals.index(cur_exp)
        except ValueError:
            ei = 0
        ch, ni = imgui.combo("##ExpFmt", ei, export_opts)
        if ch and ni != ei:
            settings.set("funscript_export_format", export_vals[ni])
            self.app.logger.info(f"Export format: {export_vals[ni]}", extra={'status_message': True})
        imgui.pop_item_width()
        _row_end()

        _row_label("Batch Default", "Default behavior when batch processing encounters existing funscripts.")
        cur_b = settings.get("batch_mode_overwrite_strategy", 0)
        if imgui.radio_button("Process All##BO0", cur_b == 0):
            if cur_b != 0:
                settings.set("batch_mode_overwrite_strategy", 0)
        imgui.same_line()
        if imgui.radio_button("Skip Existing##BO1", cur_b == 1):
            if cur_b != 1:
                settings.set("batch_mode_overwrite_strategy", 1)
        _row_end()

        _row_separator()
        _begin_settings_columns("output_axis_cols")

        # Default Secondary Axis
        _row_label("T2 Default Axis",
                    "Which axis Timeline 2 defaults to when a tracker is activated.\n"
                    "e.g. 'twist' for SSR2, 'roll' for OSR2.")
        imgui.push_item_width(-1)
        sec_options = [fa.value for fa in FunscriptAxis if fa != FunscriptAxis.STROKE]
        cur_sec = settings.get("default_secondary_axis", "roll")
        try:
            si = sec_options.index(cur_sec)
        except ValueError:
            si = 0
        ch, ni = imgui.combo("##DefSecAxis", si, sec_options)
        if ch:
            na = sec_options[ni]
            settings.set("default_secondary_axis", na)
            if self.app.tracker and hasattr(self.app.tracker, 'funscript') and self.app.tracker.funscript:
                self.app.tracker.funscript.assign_axis(2, na)
        imgui.pop_item_width()
        _row_end()

        _end_settings_columns()

        # Axis Assignments Table
        imgui.spacing()
        imgui.text("Axis Assignments (Per-axis Naming)")
        _tooltip_if_hovered("Maps each timeline to a semantic axis name.\n"
                            "Controls the file suffix and TCode channel.")

        fs_obj = None
        if self.app.tracker and hasattr(self.app.tracker, 'funscript'):
            fs_obj = self.app.tracker.funscript

        if fs_obj:
            axis_names = [fa.value for fa in FunscriptAxis]
            assignments = fs_obj.get_axis_assignments()

            imgui.columns(4, "##AxisTbl")
            imgui.separator()
            imgui.text("TL"); imgui.next_column()
            imgui.text("Axis"); imgui.next_column()
            imgui.text("Suffix"); imgui.next_column()
            imgui.text("TCode"); imgui.next_column()
            imgui.separator()

            for tl in sorted(assignments.keys()):
                cur_ax = assignments[tl]
                imgui.text(f"T{tl}"); imgui.next_column()
                imgui.push_item_width(-1)
                try:
                    ci = axis_names.index(cur_ax)
                except ValueError:
                    ci = -1
                items = axis_names if ci >= 0 else [cur_ax] + axis_names
                adj = ci if ci >= 0 else 0
                ch, ni = imgui.combo(f"##AC{tl}", adj, items)
                if ch:
                    fs_obj.assign_axis(tl, items[ni])
                    self.app.project_manager.project_dirty = True
                imgui.pop_item_width()
                imgui.next_column()
                suf = file_suffix_for_axis(cur_ax)
                imgui.text(f"{suf}.funscript" if suf else ".funscript"); imgui.next_column()
                tc = tcode_for_axis(cur_ax)
                imgui.text(tc or "-"); imgui.next_column()

            imgui.columns(1)
            imgui.separator()
        else:
            imgui.text_colored("No funscript loaded.", 0.5, 0.5, 0.5, 1.0)

    # ================================================================ #
    #  Processing                                                      #
    # ================================================================ #

    def _render_processing(self, tmode):
        app = self.app
        settings = app.app_settings
        stage_proc = app.stage_processor
        busy = stage_proc.full_analysis_active

        _begin_settings_columns("proc_cols")

        # Stage Reruns
        if _is_offline_tracker(tmode):
            if _is_hybrid_tracker(tmode):
                _row_label("Force Re-run",
                           "Re-run analysis even if cached results exist\n(preprocessed video, detection data).")
                with _DisabledScope(busy):
                    _, stage_proc.force_rerun_stage1 = imgui.checkbox(
                        "Force Re-run##FRHybrid", stage_proc.force_rerun_stage1)
                _row_end()
            else:
                _row_label("Force Re-run",
                           "Re-run stages even if cached results exist.")
                with _DisabledScope(busy):
                    _, stage_proc.force_rerun_stage1 = imgui.checkbox("Stage 1##FR1", stage_proc.force_rerun_stage1)
                    imgui.same_line()
                    _, stage_proc.force_rerun_stage2_segmentation = imgui.checkbox(
                        "Stage 2##FR2", stage_proc.force_rerun_stage2_segmentation)
                _row_end()

                _row_label("Keep Stage 2 DB",
                           "Keep the database file after processing.\nDisable to save disk space.")
                with _DisabledScope(busy):
                    ch, nv = imgui.checkbox("##RetDB", settings.get("retain_stage2_database", True))
                    if ch:
                        settings.set("retain_stage2_database", nv)
                _row_end()

        # Preprocessed video
        if _is_offline_tracker(tmode):
            _row_label("Preprocessed Video",
                       "Save a preprocessed video for faster re-runs.")
            with _DisabledScope(busy):
                if not hasattr(stage_proc, "save_preprocessed_video"):
                    stage_proc.save_preprocessed_video = settings.get("save_preprocessed_video", False)
                ch, nv = imgui.checkbox("##SavePre", stage_proc.save_preprocessed_video)
                if ch:
                    stage_proc.save_preprocessed_video = nv
                    settings.set("save_preprocessed_video", nv)
                    if nv and not _is_hybrid_tracker(tmode):
                        stage_proc.num_producers_stage1 = 1
                        settings.set("num_producers_stage1", 1)
            _row_end()

        # Workers
        if _is_offline_tracker(tmode):
            _row_separator()
            _begin_settings_columns("proc_workers_cols")

            _row_label("S1 Producers",
                       "Threads for video decoding & preprocessing.")
            imgui.push_item_width(80)
            is_pre = getattr(stage_proc, "save_preprocessed_video", False)
            with _DisabledScope(is_pre):
                ch, np_ = imgui.input_int("##S1P", stage_proc.num_producers_stage1)
                if ch and not is_pre:
                    v = max(1, np_)
                    if v != stage_proc.num_producers_stage1:
                        stage_proc.num_producers_stage1 = v
                        settings.set("num_producers_stage1", v)
            imgui.pop_item_width()
            _row_end()

            _row_label("S1 Consumers",
                       "Threads for AI model inference.")
            imgui.push_item_width(80)
            ch, nc = imgui.input_int("##S1C", stage_proc.num_consumers_stage1)
            if ch:
                v = max(1, nc)
                if v != stage_proc.num_consumers_stage1:
                    stage_proc.num_consumers_stage1 = v
                    settings.set("num_consumers_stage1", v)
            imgui.pop_item_width()
            _row_end()

            _row_label("S2 OF Workers",
                       "Processes for Stage 2 Optical Flow gap recovery.")
            imgui.push_item_width(80)
            cur_s2 = settings.get("num_workers_stage2_of", config.constants.DEFAULT_S2_OF_WORKERS)
            ch, ns2 = imgui.input_int("##S2W", cur_s2)
            if ch:
                v = max(1, ns2)
                if v != cur_s2:
                    settings.set("num_workers_stage2_of", v)
            imgui.pop_item_width()
            _row_end()

        _end_settings_columns()

    # ================================================================ #
    #  All Trackers (lazy-instantiated)                                #
    # ================================================================ #

    def _render_all_trackers(self):
        """Render collapsed subsections for every discovered tracker."""
        try:
            from config.tracker_discovery import get_tracker_discovery, TrackerCategory
            discovery = get_tracker_discovery()
            all_trackers = discovery.get_all_trackers()
        except Exception:
            imgui.text_disabled("Tracker discovery not available.")
            return

        if not all_trackers:
            imgui.text_disabled("No trackers discovered.")
            return

        # Group by category
        cat_order = [TrackerCategory.LIVE, TrackerCategory.LIVE_INTERVENTION,
                     TrackerCategory.OFFLINE, TrackerCategory.COMMUNITY]
        cat_labels = {
            TrackerCategory.LIVE: "Live",
            TrackerCategory.LIVE_INTERVENTION: "Live (Intervention)",
            TrackerCategory.OFFLINE: "Offline",
            TrackerCategory.COMMUNITY: "Community",
        }

        active_name = self.app.app_state_ui.selected_tracker_name

        for cat in cat_order:
            trackers = discovery.get_trackers_by_category(cat)
            if not trackers:
                continue

            cat_label = cat_labels.get(cat, str(cat))
            imgui.text_colored(cat_label, 0.6, 0.6, 0.6, 1.0)
            imgui.spacing()

            for tinfo in trackers:
                name = tinfo.internal_name
                is_active = (name == active_name)
                label = tinfo.display_name
                if is_active:
                    label += " [active]"

                if imgui.tree_node(f"{label}##{name}"):
                    if tinfo.description:
                        imgui.text_wrapped(tinfo.description)
                        imgui.spacing()

                    # Lazy-instantiate to get schema
                    inst = self._get_or_create_tracker_instance(name)
                    if inst is not None:
                        self._render_tracker_settings_ui(inst, id_suffix=name)
                    else:
                        imgui.text_disabled("Could not load tracker settings.")

                    imgui.tree_pop()

            imgui.spacing()

    def _get_or_create_tracker_instance(self, internal_name):
        """Lazy-create a tracker instance for settings rendering."""
        if internal_name in self._tracker_instances:
            return self._tracker_instances[internal_name]

        # If this is the active tracker, use the live instance
        active_inst = _get_current_tracker_instance(self.app)
        if active_inst and getattr(active_inst, 'metadata', None):
            if getattr(active_inst.metadata, 'name', None) == internal_name:
                self._tracker_instances[internal_name] = active_inst
                return active_inst

        # Create a new instance
        try:
            from tracker.tracker_modules import create_tracker
            inst = create_tracker(internal_name)
            if inst:
                self._tracker_instances[internal_name] = inst
                return inst
        except Exception as e:
            _logger.debug(f"Failed to instantiate tracker '{internal_name}': {e}")

        return None

    # ================================================================ #
    #  Shared tracker rendering helpers                                #
    # ================================================================ #

    def _render_tracker_settings_ui(self, tracker_inst, id_suffix=""):
        """Render settings for a tracker instance (custom UI or schema)."""
        active_inst = _get_current_tracker_instance(self.app)
        is_active = (tracker_inst is active_inst)

        # Active tracker: try custom UI first (it has full app context)
        if is_active and hasattr(tracker_inst, 'render_settings_ui'):
            try:
                if tracker_inst.render_settings_ui():
                    return
            except Exception as exc:
                imgui.text_colored("Settings UI error: %s" % exc, 1.0, 0.3, 0.3, 1.0)
                return

        # Schema-based rendering (works for both active and lazy instances)
        if hasattr(tracker_inst, 'get_settings_schema'):
            try:
                schema = tracker_inst.get_settings_schema()
                if schema and schema.get('properties'):
                    from application.utils.schema_settings_renderer import render_schema_settings
                    render_schema_settings(schema, self.app.app_settings, tracker_inst)
                    return
            except Exception:
                pass

        if is_active:
            imgui.text_disabled("No configurable settings.")
        else:
            imgui.text_disabled("Settings only available when tracker is active.")

    def _render_tracker_debug(self, tracker_inst):
        """Render debug UI if tracker provides it."""
        method = getattr(tracker_inst, 'render_debug_ui', None)
        if method is None:
            return
        try:
            from tracker.tracker_modules.core.base_tracker import BaseTracker
            if method.__func__ is BaseTracker.render_debug_ui:
                return
        except (ImportError, AttributeError):
            return
        with section_card("Debug##TrackerDebug", tier="primary", open_by_default=False) as d_open:
            if d_open:
                tracker_inst.render_debug_ui()

    def _render_class_filtering(self, app):
        """Render class filtering checkboxes."""
        classes = app.get_available_tracking_classes()
        if not classes:
            imgui.text_disabled("No classes available.")
            return

        imgui.text_wrapped("Select classes to DISCARD from tracking and analysis.")
        discarded = set(app.discarded_tracking_classes)
        changed_any = False
        num_cols = 3
        if imgui.begin_table("ClassFilterTbl", num_cols, flags=imgui.TABLE_SIZING_STRETCH_SAME):
            col = 0
            for cls in classes:
                if col == 0:
                    imgui.table_next_row()
                imgui.table_set_column_index(col)
                is_disc = (cls in discarded)
                imgui.push_id("dc_%s" % cls)
                clicked, nv = imgui.checkbox(" %s" % cls, is_disc)
                imgui.pop_id()
                if clicked:
                    changed_any = True
                    if nv:
                        discarded.add(cls)
                    else:
                        discarded.discard(cls)
                col = (col + 1) % num_cols
            imgui.end_table()

        if changed_any:
            new_list = sorted(list(discarded))
            if new_list != app.discarded_tracking_classes:
                app.discarded_tracking_classes = new_list
                app.app_settings.set("discarded_tracking_classes", new_list)
                app.project_manager.project_dirty = True
                app.logger.info("Discarded classes updated: %s" % new_list,
                                extra={"status_message": True})
                app.energy_saver.reset_activity_timer()

        imgui.spacing()
        if imgui.button("Clear All Discards##ClearDF", width=-1):
            if app.discarded_tracking_classes:
                app.discarded_tracking_classes.clear()
                app.app_settings.set("discarded_tracking_classes", [])
                app.project_manager.project_dirty = True
                app.logger.info("All class filters cleared.", extra={"status_message": True})
                app.energy_saver.reset_activity_timer()
        _tooltip_if_hovered("Uncheck all - enable all classes for tracking/analysis.")
