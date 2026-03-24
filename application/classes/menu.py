import os
import webbrowser
import platform
import imgui
from config.element_group_colors import MenuColors
from application.utils import get_logo_texture_manager
from application.utils.feature_detection import is_feature_available as _is_feature_available
from application.utils.timeline_constants import EXTRA_TIMELINE_RANGE
from application.utils.imgui_helpers import begin_modal_centered

def _menu_item_simple(label, enabled=True):
    clicked, _ = imgui.menu_item(label, enabled=enabled)
    return clicked

def _radio_line(label, is_selected):
    # Cheaper than f-strings in hot loops
    if imgui.radio_button(label, is_selected):
        return True
    return False

class MainMenu:
    __slots__ = ("app", "gui", "FRAME_OFFSET", "_last_menu_log_time", "_show_about_dialog",
                 "_kofi_texture_id", "_kofi_width", "_kofi_height", "_is_macos",
                 "_feat_supporter", "_feat_device", "_feat_streamer")

    def __init__(self, app_instance, gui_instance=None):
        self.app = app_instance
        self.gui = gui_instance
        self.FRAME_OFFSET = MenuColors.FRAME_OFFSET
        self._last_menu_log_time = 0
        self._show_about_dialog = False
        self._kofi_texture_id = None
        self._kofi_width = 0
        self._kofi_height = 0
        self._is_macos = platform.system() == "Darwin"

    # ------------------------- HELPER METHODS -------------------------

    def _axis_label_for(self, t_num):
        """Get axis label for a timeline number."""
        if self.app.tracker and hasattr(self.app.tracker, 'funscript'):
            return self.app.tracker.funscript.get_axis_for_timeline(t_num)
        return ""

    def _tl_label(self, t_num):
        """Short timeline label with axis, e.g. 'T1 (stroke)'."""
        axis = self._axis_label_for(t_num)
        return f"T{t_num} ({axis})" if axis else f"Timeline {t_num}"

    def _get_focused_timeline_editor(self):
        """Get the timeline editor that was last edited."""
        if not self.gui:
            return None
        fs_proc = getattr(self.app, 'funscript_processor', None)
        focused = fs_proc._last_edited_timeline if fs_proc else 1
        if focused == 1:
            return getattr(self.gui, 'timeline_editor1', None)
        elif focused == 2:
            return getattr(self.gui, 'timeline_editor2', None)
        return getattr(self.gui, 'timeline_editor1', None)

    def _get_active_bookmark_manager(self):
        """Get bookmark manager from the active timeline editor."""
        gui = self.gui
        if gui and hasattr(gui, 'timeline_editors'):
            editors = gui.timeline_editors
            if editors:
                return getattr(editors[0], '_bookmark_manager', None)
        return None

    def _add_bookmark_at_playhead(self):
        """Add bookmark at current playhead position."""
        bm_mgr = self._get_active_bookmark_manager()
        if not bm_mgr:
            return
        proc = self.app.processor
        if proc and proc.is_video_open():
            time_ms = proc.get_position_ms()
            bm_mgr.add(time_ms)

    def _seek_to_bookmark(self, time_ms):
        """Seek video to bookmark time."""
        proc = self.app.processor
        if proc and proc.is_video_open():
            proc.seek_to_ms(time_ms)

    def _format_bookmark_time(self, time_ms):
        """Format bookmark time for display."""
        secs = time_ms / 1000.0
        m, s = divmod(secs, 60)
        return f"{int(m)}:{s:05.2f}"

    def _get_shortcut_display(self, action_name: str) -> str:
        """
        Get formatted shortcut string for display in menus.

        Args:
            action_name: Internal action name (e.g., "save_project", "toggle_playback")

        Returns:
            Formatted shortcut string for menu display (e.g., "Cmd+S", "Ctrl+Z")
            Returns empty string if no shortcut is defined.
        """
        shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})
        shortcut_str = shortcuts.get(action_name, "")

        if not shortcut_str:
            return ""

        # Format for display: SUPER→Cmd/Win, CTRL→Ctrl, ALT→Alt, SHIFT→Shift
        display_str = shortcut_str

        if self._is_macos:
            display_str = display_str.replace("SUPER", "Cmd")
        else:
            display_str = display_str.replace("SUPER", "Win")

        display_str = display_str.replace("CTRL", "Ctrl")
        display_str = display_str.replace("ALT", "Alt")
        display_str = display_str.replace("SHIFT", "Shift")

        # Format arrow keys
        display_str = display_str.replace("RIGHT_ARROW", "Right")
        display_str = display_str.replace("LEFT_ARROW", "Left")
        display_str = display_str.replace("UP_ARROW", "Up")
        display_str = display_str.replace("DOWN_ARROW", "Down")

        # Format other keys
        display_str = display_str.replace("SPACE", "Space")
        display_str = display_str.replace("ENTER", "Enter")
        display_str = display_str.replace("BACKSPACE", "Backspace")
        display_str = display_str.replace("DELETE", "Del")
        display_str = display_str.replace("HOME", "Home")
        display_str = display_str.replace("END", "End")
        display_str = display_str.replace("PAGE_UP", "PgUp")
        display_str = display_str.replace("PAGE_DOWN", "PgDn")
        display_str = display_str.replace("EQUAL", "=")
        display_str = display_str.replace("MINUS", "-")

        return display_str

    # ------------------------- POPUPS -------------------------

    def _load_kofi_texture(self):
        """Load Ko-fi support image as OpenGL texture (once)."""
        if self._kofi_texture_id is not None:
            return self._kofi_texture_id

        try:
            import cv2
            import numpy as np
            import OpenGL.GL as gl

            kofi_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'branding', 'kofi_support.png')

            if not os.path.exists(kofi_path):
                return None

            # Load image
            kofi_img = cv2.imread(kofi_path, cv2.IMREAD_UNCHANGED)
            if kofi_img is None:
                return None

            # Convert BGR(A) to RGB(A)
            if kofi_img.shape[2] == 4:
                kofi_rgb = cv2.cvtColor(kofi_img, cv2.COLOR_BGRA2RGBA)
            else:
                kofi_rgb = cv2.cvtColor(kofi_img, cv2.COLOR_BGR2RGB)
                alpha = np.full((kofi_rgb.shape[0], kofi_rgb.shape[1], 1), 255, dtype=np.uint8)
                kofi_rgb = np.concatenate([kofi_rgb, alpha], axis=2)

            self._kofi_height, self._kofi_width = kofi_rgb.shape[:2]

            # Create OpenGL texture
            self._kofi_texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._kofi_texture_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA,
                self._kofi_width, self._kofi_height, 0,
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, kofi_rgb
            )

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            return self._kofi_texture_id

        except Exception as e:
            if hasattr(self.app, 'logger') and self.app.logger:
                self.app.logger.debug(f"Failed to load Ko-fi texture: {e}")
            return None

    def _render_timeline_selection_popup(self):
        app = self.app
        app_state = app.app_state_ui
        if not app_state.show_timeline_selection_popup:
            return

        name = "Select Reference Timeline##TimelineSelectPopup"
        # 450x220 used for centering; height auto
        if begin_modal_centered(name, 450, 220):
            imgui.text("Which timeline has the correct timing?")
            imgui.text_wrapped(
                "The offset will be calculated for the other timeline "
                "and applied to it."
            )
            imgui.separator()

            ref_num = app_state.timeline_comparison_reference_num
            # Fixed range (1..2)
            if _radio_line(f"{self._tl_label(1)} is the Reference", ref_num == 1):
                app_state.timeline_comparison_reference_num = 1
            if _radio_line(f"{self._tl_label(2)} is the Reference", ref_num == 2):
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
        if begin_modal_centered(name, 400, 240): # 400x240 used for centering; height auto
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

    def _render_about_dialog(self):
        """Render About FunGen dialog with logo and Ko-fi support."""
        if not self._show_about_dialog:
            return

        # Import constants here to get version info
        from config import constants

        # Center and open popup
        opened = begin_modal_centered("About FunGen##AboutDialog", 450)

        if opened:
            # App name and version
            app_name = constants.APP_NAME
            app_version = constants.APP_VERSION
            title_text = f"{app_name} v{app_version}"

            # Center text
            text_width = imgui.calc_text_size(title_text)[0]
            avail_width = imgui.get_content_region_available_width()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_width - text_width) * 0.5)

            imgui.push_style_color(imgui.COLOR_TEXT, 0.4, 0.8, 1.0, 1.0)
            imgui.text(title_text)
            imgui.pop_style_color()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text_wrapped("AI-powered funscript generation using computer vision")
            imgui.spacing()

            imgui.text("Created by k00gar")
            imgui.spacing()

            # GitHub link button
            if imgui.button("GitHub Repository", width=-1):
                try:
                    webbrowser.open("https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator")
                except Exception as e:
                    if hasattr(self.app, 'logger') and self.app.logger:
                        self.app.logger.warning(f"Could not open GitHub link: {e}")

            imgui.spacing()

            # Donate button
            from application.utils import primary_button_style
            with primary_button_style():
                if imgui.button("Donate on Ko-fi", width=-1):
                    try:
                        webbrowser.open("https://ko-fi.com/k00gar")
                    except Exception as e:
                        if hasattr(self.app, 'logger') and self.app.logger:
                            self.app.logger.warning(f"Could not open Ko-fi link: {e}")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Donate, become a supporter, unlock features!")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Close button
            if imgui.button("Close", width=-1):
                self._show_about_dialog = False
                imgui.close_current_popup()

            imgui.end_popup()
        else:
            # Popup was closed (X button)
            self._show_about_dialog = False

    # ------------------------- MAIN RENDER -------------------------

    def render(self):
        app = self.app
        app_state = app.app_state_ui
        file_mgr = app.file_manager
        stage_proc = app.stage_processor

        # Cache feature detection flags for this frame
        self._feat_supporter = _is_feature_available("patreon_features")
        self._feat_device = _is_feature_available("device_control")
        self._feat_streamer = _is_feature_available("streamer")

        if imgui.begin_main_menu_bar():
            # Render logo at the start of menu bar
            self._render_menu_bar_logo()

            self._render_file_menu(app_state, file_mgr)
            self._render_edit_menu(app_state)
            self._render_markers_menu(app_state)
            self._render_view_menu(app_state, stage_proc)
            self._render_tools_menu(app_state, file_mgr)
            self._render_help_menu()
            self._render_support_menu()

            # Render device control indicator after Support menu
            self._render_device_control_indicator()

            # Render Streamer indicator
            self._render_native_sync_indicator()

            # Render Supporter badge
            self._render_supporter_badge()

            imgui.end_main_menu_bar()

        self._render_timeline_selection_popup()
        self._render_timeline_comparison_results_popup()
        self._render_about_dialog()

    # ------------------------- MENUS -------------------------

    def _render_file_menu(self, app_state, file_mgr):
        app = self.app
        pm = app.project_manager
        settings = app.app_settings
        fm = app.file_manager
        fs_proc = app.funscript_processor

        if imgui.begin_menu("File", True):
            # New/Project/Video/Open
            if _menu_item_simple("New Project"):
                app.reset_project_state(for_new_project=True)
                pm.project_dirty = True

            if imgui.menu_item("Open Project...", self._get_shortcut_display("open_project"))[0]:
                pm.open_project_dialog()

            if _menu_item_simple("Open Video..."):
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

            if imgui.menu_item("Save Project", self._get_shortcut_display("save_project"),
                               selected=False, enabled=can_save)[0]:
                pm.save_project_dialog()
            if _menu_item_simple("Save Project As...", enabled=True):
                pm.save_project_dialog(save_as=True)
            imgui.separator()

            # Import/Export
            has_video = fm.video_path is not None
            if imgui.begin_menu("Import..."):
                if _menu_item_simple("Multi-Axis Funscript (Single File)..."):
                    fm.import_unified_funscript()
                if _menu_item_simple("All Axis Files (OFS naming)...", enabled=has_video):
                    fm.import_all_axes_ofs()
                imgui.separator()
                if _menu_item_simple(f"Funscript to {self._tl_label(1)}..."):
                    fm.import_funscript_to_timeline(1)
                if _menu_item_simple(f"Funscript to {self._tl_label(2)}..."):
                    fm.import_funscript_to_timeline(2)
                imgui.separator()
                if _menu_item_simple("Stage 2 Overlay Data..."):
                    fm.import_stage2_overlay_data()
                imgui.end_menu()

            if imgui.begin_menu("Export..."):
                if _menu_item_simple("Multi-Axis Funscript (Single File)..."):
                    fm.export_unified_funscript()
                if _menu_item_simple("All Axis Files (OFS naming)...", enabled=has_video):
                    fm.export_all_axes_ofs()
                imgui.separator()
                if _menu_item_simple(f"Funscript from {self._tl_label(1)}..."):
                    fm.export_funscript_from_timeline(1)
                if _menu_item_simple(f"Funscript from {self._tl_label(2)}..."):
                    fm.export_funscript_from_timeline(2)
                imgui.separator()
                if _menu_item_simple("Heatmap PNG..."):
                    fm.export_heatmap_png(1)
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
        mgr = app.undo_manager

        if imgui.begin_menu("Edit", True):
            # Unified undo/redo
            undo_desc = mgr.peek_undo()
            redo_desc = mgr.peek_redo()

            undo_label = f"Undo: {undo_desc}" if undo_desc else "Undo"
            redo_label = f"Redo: {redo_desc}" if redo_desc else "Redo"

            if imgui.menu_item(
                undo_label, self._get_shortcut_display("undo_timeline1"),
                selected=False, enabled=bool(undo_desc)
            )[0]:
                desc = mgr.undo(app)
                if desc:
                    app.notify(f"Undo: {desc}", "info", 1.5)
            if imgui.menu_item(
                redo_label, self._get_shortcut_display("redo_timeline1"),
                selected=False, enabled=bool(redo_desc)
            )[0]:
                desc = mgr.redo(app)
                if desc:
                    app.notify(f"Redo: {desc}", "info", 1.5)

            imgui.separator()

            # Point editing (dispatched to focused timeline)
            tl = self._get_focused_timeline_editor()
            has_sel = tl and tl.multi_selected_action_indices
            has_actions = tl and tl._get_actions()

            if imgui.menu_item("Select All", self._get_shortcut_display("select_all_points"),
                               selected=False, enabled=bool(has_actions))[0]:
                if tl:
                    tl.multi_selected_action_indices = set(range(len(tl._get_actions())))
            if imgui.menu_item("Deselect All", self._get_shortcut_display("deselect_all_points"),
                               selected=False, enabled=bool(has_sel))[0]:
                if tl:
                    tl.multi_selected_action_indices.clear()
                    tl.selected_action_idx = -1
            if imgui.menu_item("Delete Selected", self._get_shortcut_display("delete_selected_point"),
                               selected=False, enabled=bool(has_sel))[0]:
                if tl:
                    tl._delete_selected_points()
            if imgui.menu_item("Copy", self._get_shortcut_display("copy_selection"),
                               selected=False, enabled=bool(has_sel))[0]:
                if tl:
                    tl._copy_selection()
            if imgui.menu_item("Paste", self._get_shortcut_display("paste_selection"),
                               selected=False, enabled=bool(tl))[0]:
                if tl:
                    tl._paste_selection()

            imgui.separator()

            video_loaded = self.app.processor and self.app.processor.video_info
            if imgui.menu_item("Go to Frame...", self._get_shortcut_display("go_to_frame"),
                               selected=False, enabled=video_loaded)[0]:
                if self.gui:
                    self.gui._go_to_frame_open = True
                    self.gui._go_to_frame_input = ""
                    self.gui._go_to_frame_focus = True

            imgui.end_menu()


    def _render_view_menu(self, app_state, stage_proc):
        if imgui.begin_menu("View", True):
            # Layout submenu
            self._render_layout_submenu(app_state)

            # Panels submenu (floating mode only) - right after Layout
            self._render_panels_submenu(app_state)

            # Show Toolbar
            if not hasattr(app_state, 'show_toolbar'):
                app_state.show_toolbar = True
            clicked, val = imgui.menu_item(
                "Show Toolbar",
                selected=app_state.show_toolbar
            )
            if clicked:
                app_state.show_toolbar = val
                self.app.project_manager.project_dirty = True

            # Show Video Controls overlay
            clicked, val = imgui.menu_item(
                "Show Video Controls",
                selected=app_state.show_video_controls_overlay
            )
            if clicked:
                app_state.show_video_controls_overlay = val
                self.app.project_manager.project_dirty = True

            # Enter Fullscreen (mpv - supporter exclusive)
            mpv = getattr(self.app, '_mpv_controller', None)
            is_fs_active = mpv is not None and mpv.is_active
            from application.utils.feature_detection import is_feature_available as _is_feature_available
            fs_available = _is_feature_available("patreon_features") and mpv is not None
            fs_label = "Exit Fullscreen" if is_fs_active else "Enter Fullscreen (Patreon)"
            clicked, _ = imgui.menu_item(fs_label, "F11", selected=is_fs_active, enabled=fs_available)
            if clicked and fs_available:
                if is_fs_active:
                    mpv.stop()
                else:
                    file_manager = getattr(self.app, 'file_manager', None)
                    video_path = file_manager.video_path if file_manager else None
                    if video_path:
                        processor = self.app.processor
                        start_frame = processor.current_frame_index if processor else 0
                        mpv.start(video_path, start_frame=start_frame, fullscreen=True)

            imgui.separator()

            # --- Display toggles ---
            pm = self.app.project_manager
            settings = self.app.app_settings

            for label, attr, sc_key in (
                ("Funscript Preview", "show_funscript_timeline", "toggle_funscript_preview"),
                ("Heatmap", "show_heatmap", "toggle_heatmap"),
                ("Audio Waveform", None, "toggle_waveform"),
                ("3D Simulator", "show_simulator_3d", "toggle_3d_simulator"),
                ("Gauge", "show_script_gauge", "toggle_script_gauge"),
            ):
                if attr:
                    cur = getattr(app_state, attr)
                    clicked, val = imgui.menu_item(label, self._get_shortcut_display(sc_key), selected=cur)
                    if clicked:
                        setattr(app_state, attr, val)
                        pm.project_dirty = True
                else:
                    # Audio waveform has special toggle
                    clicked, _ = imgui.menu_item(label, self._get_shortcut_display(sc_key),
                                                 selected=app_state.show_audio_waveform)
                    if clicked:
                        self.app.toggle_waveform_visibility()
                        pm.project_dirty = True

            imgui.separator()

            # Timeline visibility
            self._render_timelines_submenu(app_state)

            # Video Overlays
            self._render_video_overlays_submenu(app_state, stage_proc)

            imgui.separator()

            # Preview options
            enhanced_preview = settings.get("enable_enhanced_funscript_preview", True)
            clicked, new_val = imgui.menu_item("Hover Zoom Preview", selected=enhanced_preview)
            if clicked and new_val != enhanced_preview:
                settings.set("enable_enhanced_funscript_preview", new_val)

            use_simplified = settings.get("use_simplified_funscript_preview", False)
            clicked, new_val = imgui.menu_item("Low-Detail Preview (faster)", selected=use_simplified)
            if clicked and new_val != use_simplified:
                settings.set("use_simplified_funscript_preview", new_val)
                app_state.funscript_preview_dirty = True

            if imgui.menu_item("Reset Timeline View", self._get_shortcut_display("reset_timeline_view"))[0]:
                app_state.timeline_zoom_factor_ms_per_px = settings.get_default_settings().get(
                    "timeline_zoom_factor_ms_per_px", 1.0)
                app_state.timeline_pan_offset_ms = 0.0

            imgui.separator()

            # GUI Scale
            _settings = self.app.app_settings
            if imgui.begin_menu("GUI Scale"):
                import config.constants as _cfg
                cur_scale = _settings.get("global_font_scale", _cfg.DEFAULT_FONT_SCALE)
                for label, val in zip(_cfg.FONT_SCALE_LABELS, _cfg.FONT_SCALE_VALUES):
                    is_selected = abs(val - cur_scale) < 0.01
                    if imgui.menu_item(label, selected=is_selected)[0] and not is_selected:
                        _settings.set("global_font_scale", val)
                        _settings.set("auto_system_scaling_enabled", False)
                imgui.end_menu()

            imgui.separator()

            # Toast notifications toggle
            toast_on = _settings.get("show_toast_notifications", True)
            clicked, val = imgui.menu_item(
                "Toast Notifications", selected=toast_on
            )
            if clicked:
                _settings.set("show_toast_notifications", val)

            # Show Advanced Options
            clicked, val = imgui.menu_item(
                "Show Advanced Options",
                selected=app_state.show_advanced_options
            )
            if clicked:
                app_state.show_advanced_options = val
                _settings.set("show_advanced_options", val)
                self.app.project_manager.project_dirty = True

            imgui.end_menu()

    def _render_layout_submenu(self, app_state):
        pm = self.app.project_manager
        if imgui.begin_menu("Layout"):
            # Layout mode selection
            current = app_state.ui_layout_mode
            if _radio_line("Fixed Panels", current == "fixed"):
                if current != "fixed":
                    app_state.ui_layout_mode = "fixed"
                    pm.project_dirty = True

            if _radio_line("Floating Windows", current == "floating"):
                if current != "floating":
                    app_state.ui_layout_mode = "floating"
                    app_state.just_switched_to_floating = True
                    pm.project_dirty = True

            imgui.end_menu()

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

    def _render_timelines_submenu(self, app_state):
        app = self.app
        pm = app.project_manager
        settings = app.app_settings

        if imgui.begin_menu("Timelines"):
            # Helper to get axis label from tracker's funscript
            def _axis_label_for(t_num):
                if app.tracker and hasattr(app.tracker, 'funscript'):
                    return app.tracker.funscript.get_axis_for_timeline(t_num)
                return ""

            # Interactive editors
            for t_num, attr, sc_key in (
                (1, "show_funscript_interactive_timeline", None),
                (2, "show_funscript_interactive_timeline2", "toggle_timeline2"),
            ):
                axis_label = _axis_label_for(t_num)
                label = f"Timeline {t_num} ({axis_label})" if axis_label else f"Interactive Timeline {t_num}"
                cur = getattr(app_state, attr)
                hint = self._get_shortcut_display(sc_key) if sc_key else ""
                clicked, val = imgui.menu_item(label, hint, selected=cur)
                if clicked:
                    setattr(app_state, attr, val)
                    pm.project_dirty = True

            # Extra timelines (T3+) — supporter only
            if self._feat_supporter:
                for t_num in EXTRA_TIMELINE_RANGE:
                    vis_attr = f"show_funscript_interactive_timeline{t_num}"
                    cur = getattr(app_state, vis_attr, False)
                    axis_label = _axis_label_for(t_num)
                    label = f"Timeline {t_num} ({axis_label})" if axis_label else f"Interactive Timeline {t_num}"
                    clicked, val = imgui.menu_item(label, selected=cur)
                    if clicked:
                        setattr(app_state, vis_attr, val)
                        pm.project_dirty = True

            imgui.end_menu()

    def _render_markers_menu(self, app_state):
        """Combined bookmarks + chapters menu."""
        app = self.app
        pm = app.project_manager
        fm = app.file_manager
        fs_proc = app.funscript_processor

        if imgui.begin_menu("Markers", True):
            has_video = fm.video_path is not None
            has_chapters = has_video and len(fs_proc.video_chapters) > 0

            # --- Bookmarks ---
            if imgui.menu_item("Add Bookmark", self._get_shortcut_display("add_bookmark"))[0]:
                self._add_bookmark_at_playhead()

            if not hasattr(app_state, 'show_bookmark_list_window'):
                app_state.show_bookmark_list_window = False
            clicked, val = imgui.menu_item(
                "Bookmark List", selected=app_state.show_bookmark_list_window
            )
            if clicked:
                app_state.show_bookmark_list_window = val

            bm_mgr = self._get_active_bookmark_manager()
            has_bookmarks = bm_mgr and bm_mgr.bookmarks
            if imgui.begin_menu("Go to Bookmark", enabled=bool(has_bookmarks)):
                for bm in bm_mgr.bookmarks:
                    time_str = self._format_bookmark_time(bm.time_ms)
                    label = f"{bm.name or 'Bookmark'} ({time_str})##{bm.id}"
                    if imgui.menu_item(label)[0]:
                        self._seek_to_bookmark(bm.time_ms)
                imgui.end_menu()

            imgui.separator()

            # --- Chapters Windows ---
            if not hasattr(app_state, "show_chapter_list_window"):
                app_state.show_chapter_list_window = False
            clicked, val = imgui.menu_item(
                "Chapter List", selected=app_state.show_chapter_list_window
            )
            if clicked:
                app_state.show_chapter_list_window = val
                pm.project_dirty = True

            if not hasattr(app_state, "show_chapter_type_manager"):
                app_state.show_chapter_type_manager = False
            clicked, val = imgui.menu_item(
                "Chapter Type Manager", selected=app_state.show_chapter_type_manager
            )
            if clicked:
                app_state.show_chapter_type_manager = val
                pm.project_dirty = True

            # Go to Chapter submenu
            if imgui.begin_menu("Go to Chapter", enabled=bool(fs_proc.video_chapters)):
                if fs_proc.video_chapters:
                    for ch in fs_proc.video_chapters:
                        label = f"{ch.position_short_name} ({ch.start_frame_id}-{ch.end_frame_id})##{ch.unique_id}"
                        if imgui.menu_item(label)[0]:
                            if self.app.processor:
                                self.app.processor.seek_video(ch.start_frame_id)
                else:
                    imgui.menu_item("(no chapters)", enabled=False)
                imgui.end_menu()

            # Chapter editing shortcuts
            video_loaded = self.app.processor and self.app.processor.video_info
            if imgui.menu_item("Set Chapter Start", self._get_shortcut_display("set_chapter_start"),
                               enabled=video_loaded)[0]:
                if self.gui and hasattr(self.gui, '_handle_set_chapter_start_shortcut'):
                    self.gui._handle_set_chapter_start_shortcut()
            if imgui.menu_item("Set Chapter End", self._get_shortcut_display("set_chapter_end"),
                               enabled=video_loaded)[0]:
                if self.gui and hasattr(self.gui, '_handle_set_chapter_end_shortcut'):
                    self.gui._handle_set_chapter_end_shortcut()
            if imgui.menu_item("Select Points in Chapter", self._get_shortcut_display("select_points_in_chapter"),
                               enabled=has_chapters)[0]:
                pass  # Handled by timeline shortcut handler when focused

            imgui.separator()

            # --- Chapters I/O ---
            if _menu_item_simple("Save Chapters...", enabled=has_chapters):
                if self.app.gui_instance and self.app.gui_instance.file_dialog and has_chapters:
                    chapter_mgr = self.app.chapter_manager
                    default_path = chapter_mgr.get_default_chapter_filepath(fm.video_path)
                    initial_dir = os.path.dirname(default_path)
                    initial_filename = os.path.basename(default_path)
                    self.app.gui_instance.file_dialog.show(
                        is_save=True,
                        title="Save Chapters",
                        extension_filter="Chapter Files (*.json),*.json",
                        callback=lambda filepath: self._save_chapters_callback(filepath),
                        initial_path=initial_dir,
                        initial_filename=initial_filename
                    )

            if _menu_item_simple("Save Chapters As...", enabled=has_chapters):
                if self.app.gui_instance and self.app.gui_instance.file_dialog and has_chapters:
                    initial_dir = os.path.dirname(fm.video_path) if fm.video_path else os.getcwd()
                    initial_filename = "chapters.json"
                    self.app.gui_instance.file_dialog.show(
                        is_save=True,
                        title="Save Chapters As",
                        extension_filter="Chapter Files (*.json),*.json",
                        callback=lambda filepath: self._save_chapters_callback(filepath),
                        initial_path=initial_dir,
                        initial_filename=initial_filename
                    )

            if _menu_item_simple("Load Chapters...", enabled=has_video):
                if self.app.gui_instance and self.app.gui_instance.file_dialog and has_video:
                    initial_dir = os.path.dirname(fm.video_path) if fm.video_path else os.getcwd()
                    self.app.gui_instance.file_dialog.show(
                        is_save=False,
                        title="Load Chapters",
                        extension_filter="Chapter Files (*.json),*.json",
                        callback=lambda filepath: self._load_chapters_callback(filepath),
                        initial_path=initial_dir
                    )

            imgui.separator()

            if _menu_item_simple("Backup Chapters Now", enabled=has_chapters):
                chapter_mgr = self.app.chapter_manager
                success = chapter_mgr.backup_chapters_manually(fs_proc.video_chapters, fm.video_path)
                if not success:
                    self.app.logger.error("Failed to create chapter backup", extra={'status_message': True})

            if _menu_item_simple("Clear All Chapters", enabled=has_chapters):
                if hasattr(self.app, 'confirmation_needed'):
                    self.app.confirmation_needed = ('clear_chapters', len(fs_proc.video_chapters))
                else:
                    fs_proc.video_chapters.clear()
                    self.app.logger.info("All chapters cleared", extra={'status_message': True})

            imgui.end_menu()

    def _render_video_overlays_submenu(self, app_state, stage_proc):
        pm = self.app.project_manager
        app = self.app

        if imgui.begin_menu("Video Overlays"):
            # Video feed toggle
            clicked, val = imgui.menu_item(
                "Show Video Feed", self._get_shortcut_display("toggle_video_feed"),
                selected=app_state.show_video_feed
            )
            if clicked:
                app_state.show_video_feed = val
                app.app_settings.set("show_video_feed", val)
                pm.project_dirty = True

            imgui.separator()

            # Stage 2 overlay
            can_show_s2 = stage_proc.stage2_overlay_data is not None
            clicked, val = imgui.menu_item(
                "Show Stage 2 Overlay",
                selected=app_state.show_stage2_overlay,
                enabled=can_show_s2,
            )
            if clicked:
                app_state.show_stage2_overlay = val
                pm.project_dirty = True

            # Tracker overlays (only if tracker exists)
            tracker = app.tracker
            if tracker:
                clicked, val = imgui.menu_item(
                    "Show Detections/Masks",
                    selected=app_state.ui_show_masks
                )
                if clicked:
                    app_state.set_tracker_ui_flag("show_masks", val)

                clicked, val = imgui.menu_item(
                    "Show Optical Flow",
                    selected=app_state.ui_show_flow
                )
                if clicked:
                    app_state.set_tracker_ui_flag("show_flow", val)

            imgui.separator()

            # Overlay mode for 3D Simulator
            clicked, val = imgui.menu_item(
                "3D Simulator on Video",
                selected=app.app_settings.get('simulator_3d_overlay_mode', False)
            )
            if clicked:
                app.app_settings.set('simulator_3d_overlay_mode', val)

            imgui.end_menu()

    def _render_tools_menu(self, app_state, file_mgr):
        app = self.app

        if imgui.begin_menu("Tools", True):
            # AI Models dialog
            if not hasattr(app_state, "show_ai_models_dialog"):
                app_state.show_ai_models_dialog = False
            clicked, _ = imgui.menu_item(
                "AI Models...",
                selected=app_state.show_ai_models_dialog,
            )
            if clicked:
                app_state.show_ai_models_dialog = not app_state.show_ai_models_dialog
            if imgui.is_item_hovered():
                imgui.set_tooltip("Configure AI model paths and download default models")

            imgui.separator()

            # Compare Timelines
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

            imgui.separator()

            # TensorRT compilation (CUDA only)
            if not hasattr(app, "tensorrt_compiler_window"):
                app.tensorrt_compiler_window = None
            if _menu_item_simple("Compile CUDA .engine..."):
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

            imgui.separator()

            # Manage Generated Files (standalone)
            clicked, _ = imgui.menu_item(
                "Manage Generated Files...",
                selected=app_state.show_generated_file_manager,
            )
            if clicked:
                app.toggle_file_manager_window()

            imgui.separator()

            # Setup Wizard
            if _menu_item_simple("Setup Wizard..."):
                gui = getattr(app, "gui_instance", None)
                if gui is not None:
                    gui._show_setup_wizard = True
                    app.logger.info("Setup Wizard triggered from menu")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Re-run the first-time setup wizard (display scale, mode, models)")

            imgui.end_menu()

    def _render_help_menu(self):
        app = self.app
        settings = app.app_settings
        updater = app.updater

        if imgui.begin_menu("Help", True):
            # About
            if _menu_item_simple("About FunGen..."):
                self._show_about_dialog = True

            imgui.separator()

            # Keyboard Shortcuts
            clicked, _ = imgui.menu_item("Keyboard Shortcuts...", "F1")
            if clicked:
                if hasattr(app, 'gui_instance') and app.gui_instance:
                    app.gui_instance.keyboard_shortcuts_dialog.open()

            imgui.separator()

            # Updates submenu
            if imgui.begin_menu("Updates"):
                # Settings toggles
                for key, label, default in (
                    ("updater_check_on_startup", "Check for Updates on Startup", True),
                    ("updater_check_periodically", "Check Periodically (Hourly)", True),
                ):
                    cur = settings.get(key, default)
                    clicked, new_val = imgui.menu_item(label, selected=cur)
                    if clicked and new_val != cur:
                        settings.set(key, new_val)
                imgui.separator()

                if _menu_item_simple("Select Update Commit..."):
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

            imgui.separator()

            # System Report — copy to clipboard
            if _menu_item_simple("Copy System Report"):
                try:
                    from application.utils.system_report import generate_report
                    report = generate_report()
                    try:
                        import pyperclip
                        pyperclip.copy(report)
                    except ImportError:
                        import subprocess as _sp
                        proc = _sp.Popen(['pbcopy'], stdin=_sp.PIPE, text=True)
                        proc.communicate(report)
                    if hasattr(app, 'logger') and app.logger:
                        app.logger.info("System report copied to clipboard.",
                                        extra={'status_message': True, 'duration': 3.0})
                except Exception as e:
                    if hasattr(app, 'logger') and app.logger:
                        app.logger.warning(f"Could not generate system report: {e}",
                                           extra={'status_message': True})
            if imgui.is_item_hovered():
                imgui.set_tooltip("Generate a system report and copy it to the clipboard.\nUseful for bug reports.")

            imgui.end_menu()

    def _render_support_menu(self):
        """Render top-level Support FunGen menu."""
        app = self.app
        if imgui.begin_menu("Support FunGen"):
            if _menu_item_simple("Get Add-ons (Ko-fi)"):
                try:
                    webbrowser.open("https://ko-fi.com/k00gar")
                except Exception as e:
                    if hasattr(app, 'logger') and app.logger:
                        app.logger.warning(f"Could not open Ko-fi link: {e}")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Add-ons available at ko-fi.com/k00gar:\n"
                    "- Device Control (Handy, OSR2, etc.)\n"
                    "- Video Streamer (XBVR, Stash)\n"
                    "- Patreon (batch processing, early access)\n\n"
                    "After purchase, use Discord bot commands to receive your files."
                )

            if _menu_item_simple("Join Discord Community"):
                try:
                    webbrowser.open("https://discord.com/invite/WYkjMbtCZA")
                except Exception as e:
                    if hasattr(app, 'logger') and app.logger:
                        app.logger.warning(f"Could not open Discord link: {e}")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Join the FunGen Discord community\n"
                    "Get help, share results, and discuss features!"
                )

            imgui.separator()

            if _menu_item_simple("Report Issue on GitHub"):
                try:
                    webbrowser.open("https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/issues")
                except Exception as e:
                    if hasattr(app, 'logger') and app.logger:
                        app.logger.warning(f"Could not open GitHub issues link: {e}")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Report bugs or request features on GitHub"
                )

            imgui.end_menu()

    def _render_menu_bar_logo(self):
        """Render FunGen logo at the start of menu bar."""
        # Load logo texture
        logo_manager = get_logo_texture_manager()
        logo_texture = logo_manager.get_texture_id()
        logo_width, logo_height = logo_manager.get_dimensions()

        if logo_texture and logo_width > 0 and logo_height > 0:
            # Scale logo to menu bar height (typically ~20px)
            menu_bar_height = imgui.get_frame_height()
            logo_display_h = menu_bar_height - 4  # Small padding
            logo_display_w = int(logo_width * (logo_display_h / logo_height))

            # Add small padding on left
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + 8)

            # Draw logo
            imgui.image(logo_texture, logo_display_w, logo_display_h)

            # Add spacing after logo before menus
            imgui.same_line(spacing=8)

    def _render_supporter_badge(self):
        """Render Patreon status indicator in menu bar, consistent with Device/Streamer."""
        if self._feat_supporter:
            # Green for active (same pattern as Device: ON / Streamer: ON)
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 0.2, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.7, 0.3, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.15, 0.5, 0.15, 1.0)
            imgui.small_button("Patreon: ON")
            imgui.pop_style_color(3)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Patreon features active\nBatch processing, early access trackers")
        else:
            # Red for inactive (same pattern as Device: OFF / Streamer: OFF)
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.7, 0.3, 0.3, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.8, 0.4, 0.4, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.6, 0.2, 0.2, 1.0)
            imgui.small_button("Patreon: OFF")
            imgui.pop_style_color(3)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Patreon features not activated\nMonthly subscription at ko-fi.com/k00gar")

    def _render_device_control_indicator(self):
        """Render simple device control status indicator button."""
        app = self.app

        # Check if device manager exists and is connected
        device_manager = getattr(app, 'device_manager', None)
        device_count = 0

        if device_manager and device_manager.is_connected():
            # Count connected devices
            device_count = len(device_manager.connected_devices)

            # Get active control source
            control_source = device_manager.get_active_control_source()

            # Choose color based on control source
            if control_source == 'streamer':
                # Blue for streamer control
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.5, 0.9, 1.0)  # Blue
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.6, 1.0, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.1, 0.4, 0.8, 1.0)
                button_label = f"[S] Device: {device_count}"
            elif control_source == 'desktop':
                # Green for desktop control
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.7, 0.2, 1.0)  # Green
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.8, 0.3, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.1, 0.6, 0.1, 1.0)
                button_label = f"[D] Device: {device_count}"
            else:
                # Yellow for idle/no control
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.7, 0.7, 0.2, 1.0)  # Yellow
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.8, 0.8, 0.3, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.6, 0.6, 0.1, 1.0)
                button_label = f"[-] Device: {device_count}"

            button_clicked = imgui.small_button(button_label)
            imgui.pop_style_color(3)

            if imgui.is_item_hovered():
                connected_devices = list(device_manager.connected_devices.keys())

                # Build tooltip with control source info
                if control_source == 'streamer':
                    control_info = "Controlled by: Streamer (Browser)"
                elif control_source == 'desktop':
                    control_info = "Controlled by: Desktop (FunGen)"
                else:
                    control_info = "Controlled by: None (Idle)"

                if device_count == 1:
                    device_name = connected_devices[0] if connected_devices else "Unknown"
                    imgui.set_tooltip(f"Device: {device_name}\n{control_info}\n\n[S] = Streamer  [D] = Desktop  [-] = Idle")
                else:
                    device_list = ", ".join(connected_devices[:3])  # Show up to 3
                    if device_count > 3:
                        device_list += f" (+{device_count - 3} more)"
                    imgui.set_tooltip(f"{device_count} devices connected\n{device_list}\n{control_info}\n\n[S] = Streamer  [D] = Desktop  [-] = Idle")
        else:
            # Red button for inactive/disconnected
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.7, 0.3, 0.3, 1.0)  # Red
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.8, 0.4, 0.4, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.6, 0.2, 0.2, 1.0)
            button_clicked = imgui.small_button("Device: OFF")
            imgui.pop_style_color(3)

            if imgui.is_item_hovered():
                if device_manager:
                    imgui.set_tooltip("No device connected\nGo to Device Control tab to connect")
                else:
                    # Check if device_control feature is available (folder exists)
                    if self._feat_device:
                        imgui.set_tooltip("Device control not initialized\nCheck Device Control tab in Control Panel")
                    else:
                        imgui.set_tooltip("Add-on available at ko-fi.com/k00gar")

    def _render_native_sync_indicator(self):
        """Render Streamer status indicator button."""
        app = self.app

        # Check if Streamer manager exists and is running
        sync_manager = None
        is_running = False
        client_count = 0

        # Check if we have access to GUI and control panel
        has_gui = self.gui is not None
        has_control_panel = has_gui and hasattr(self.gui, 'control_panel_ui')

        if has_control_panel:
            control_panel = self.gui.control_panel_ui
            sync_manager = getattr(control_panel, '_native_sync_manager', None)

            if sync_manager:
                # Get status to check is_running
                try:
                    status = sync_manager.get_status()
                    is_running = status.get('is_running', False)
                    client_count = status.get('connected_clients', 0)
                except Exception as e:
                    # Fallback if get_status fails
                    is_running = False
                    client_count = 0

        if is_running and client_count > 0:
            # Green button showing client count
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.7, 0.2, 1.0)  # Green
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.8, 0.3, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.1, 0.6, 0.1, 1.0)
            button_label = f"Streamer: {client_count}"
            button_clicked = imgui.small_button(button_label)
            imgui.pop_style_color(3)

            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Streaming to {client_count} client{'s' if client_count != 1 else ''}\nServing video to browsers/VR headsets")
        elif is_running:
            # Yellow button for running but no clients
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.7, 0.2, 1.0)  # Yellow
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.9, 0.8, 0.3, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.7, 0.6, 0.1, 1.0)
            button_clicked = imgui.small_button("Streamer: 0")
            imgui.pop_style_color(3)

            if imgui.is_item_hovered():
                imgui.set_tooltip("Streamer running\nWaiting for clients to connect")
        else:
            # Red button for inactive
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.7, 0.3, 0.3, 1.0)  # Red
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.8, 0.4, 0.4, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.6, 0.2, 0.2, 1.0)
            button_clicked = imgui.small_button("Streamer: OFF")
            imgui.pop_style_color(3)

            if imgui.is_item_hovered():
                if sync_manager:
                    imgui.set_tooltip("Streamer not running\nGo to Streamer tab to start")
                else:
                    # Check if sync_server feature is available (folder exists)
                    if self._feat_streamer:
                        imgui.set_tooltip("Streamer not initialized\nCheck Streamer tab in Control Panel")
                    else:
                        imgui.set_tooltip("Add-on available at ko-fi.com/k00gar")

    # ==================== CHAPTER FILE OPERATION CALLBACKS ====================

    def _save_chapters_callback(self, filepath):
        """Callback for Save Chapters to File operation."""
        chapter_mgr = self.app.chapter_manager
        fs_proc = self.app.funscript_processor
        fm = self.app.file_manager

        video_info = {
            "path": fm.video_path,
            "fps": self.app.processor.fps if self.app.processor else 30.0,
            "total_frames": self.app.processor.total_frames if self.app.processor else 0
        }

        success = chapter_mgr.save_chapters_to_file(filepath, fs_proc.video_chapters, video_info)
        if success:
            self.app.logger.info(f"Saved {len(fs_proc.video_chapters)} chapters to {os.path.basename(filepath)}",
                               extra={'status_message': True})
        else:
            self.app.logger.error("Failed to save chapters", extra={'status_message': True})

    def _load_chapters_callback(self, filepath):
        """Callback for Load Chapters from File operation."""
        chapter_mgr = self.app.chapter_manager
        fs_proc = self.app.funscript_processor

        chapters, metadata = chapter_mgr.load_chapters_from_file(filepath)
        if chapters:
            # Replace mode by default
            fs_proc.video_chapters = chapters
            self.app.logger.info(f"Loaded {len(chapters)} chapters from {os.path.basename(filepath)}",
                               extra={'status_message': True})

            # Mark project as dirty
            if hasattr(self.app, 'project_manager'):
                self.app.project_manager.project_dirty = True
        else:
            self.app.logger.error("Failed to load chapters", extra={'status_message': True})

