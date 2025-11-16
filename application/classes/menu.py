import os
import webbrowser
import platform
import imgui
from config.element_group_colors import MenuColors
from application.utils import get_logo_texture_manager

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
    __slots__ = ("app", "gui", "FRAME_OFFSET", "_last_menu_log_time", "_show_about_dialog",
                 "_kofi_texture_id", "_kofi_width", "_kofi_height", "_is_macos")

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
        display_str = display_str.replace("RIGHT_ARROW", "→")
        display_str = display_str.replace("LEFT_ARROW", "←")
        display_str = display_str.replace("UP_ARROW", "↑")
        display_str = display_str.replace("DOWN_ARROW", "↓")

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

    def _render_about_dialog(self):
        """Render About FunGen dialog with logo and Ko-fi support."""
        if not self._show_about_dialog:
            return

        # Import constants here to get version info
        from config import constants

        # Center and open popup
        imgui.open_popup("About FunGen##AboutDialog")

        # Center on main viewport
        mv = imgui.get_main_viewport()
        main_viewport_pos_x, main_viewport_pos_y = mv.pos[0], mv.pos[1]
        main_viewport_w, main_viewport_h = mv.size[0], mv.size[1]
        dialog_width = 450
        pos_x = main_viewport_pos_x + (main_viewport_w - dialog_width) * 0.5
        pos_y = main_viewport_pos_y + main_viewport_h * 0.3  # Center vertically (slightly higher)
        imgui.set_next_window_position(pos_x, pos_y, condition=imgui.ONCE)
        imgui.set_next_window_size(dialog_width, 0, condition=imgui.ONCE)

        opened, _ = imgui.begin_popup_modal(
            "About FunGen##AboutDialog",
            True,
            flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE
        )

        if opened:
            # Load logo
            logo_manager = get_logo_texture_manager()
            logo_texture = logo_manager.get_texture_id()
            logo_width, logo_height = logo_manager.get_dimensions()

            # Center and display logo
            if logo_texture and logo_width > 0 and logo_height > 0:
                # Scale logo to reasonable size (max 150px)
                max_size = 150
                if logo_width > logo_height:
                    display_w = min(logo_width, max_size)
                    display_h = int(logo_height * (display_w / logo_width))
                else:
                    display_h = min(logo_height, max_size)
                    display_w = int(logo_width * (display_h / logo_height))

                # Center horizontally
                avail_width = imgui.get_content_region_available_width()
                imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_width - display_w) * 0.5)
                imgui.image(logo_texture, display_w, display_h)
                imgui.spacing()

            # App name and version
            app_name = constants.APP_NAME
            app_version = constants.APP_VERSION
            title_text = f"{app_name} v{app_version}"

            # Center text
            text_width = imgui.calc_text_size(title_text)[0]
            avail_width = imgui.get_content_region_available_width()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_width - text_width) * 0.5)

            imgui.push_style_color(imgui.COLOR_TEXT, 0.4, 0.8, 1.0, 1.0)  # Nice blue
            imgui.text(title_text)
            imgui.pop_style_color()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Description
            imgui.text_wrapped("AI-powered funscript generation using computer vision")
            imgui.spacing()

            # GitHub link button
            if imgui.button("GitHub Repository", width=-1):
                try:
                    webbrowser.open("https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator")
                except Exception as e:
                    if hasattr(self.app, 'logger') and self.app.logger:
                        self.app.logger.warning(f"Could not open GitHub link: {e}")

            imgui.spacing()

            # Ko-fi support section with image button
            kofi_texture = self._load_kofi_texture()
            if kofi_texture and self._kofi_width > 0 and self._kofi_height > 0:
                # Scale to dialog width
                avail_width = imgui.get_content_region_available_width()
                scale = avail_width / self._kofi_width
                display_w = avail_width
                display_h = int(self._kofi_height * scale)

                # Image button with no background/border for clean appearance
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.0, 0.0, 0.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.0, 0.0, 0.0, 0.1)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.0, 0.0, 0.0, 0.2)

                if imgui.image_button(kofi_texture, display_w, display_h):
                    try:
                        webbrowser.open("https://ko-fi.com/k00gar")
                    except Exception as e:
                        if hasattr(self.app, 'logger') and self.app.logger:
                            self.app.logger.warning(f"Could not open Ko-fi link: {e}")

                imgui.pop_style_color(3)

                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Support development and unlock device control!\n"
                        "• Hardware device integration (Handy, OSR2, etc.)\n"
                        "• Live tracking with device control\n"
                        "• Synchronized playback"
                    )
            else:
                # Fallback to text button if image fails to load
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.7, 0.2, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.8, 0.3, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.15, 0.6, 0.15, 1.0)

                if imgui.button("Support on Ko-fi", width=-1):
                    try:
                        webbrowser.open("https://ko-fi.com/k00gar")
                    except Exception as e:
                        if hasattr(self.app, 'logger') and self.app.logger:
                            self.app.logger.warning(f"Could not open Ko-fi link: {e}")

                imgui.pop_style_color(3)

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Credits
            imgui.text("Created by k00gar")
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

        if imgui.begin_main_menu_bar():
            # Render logo at the start of menu bar
            self._render_menu_bar_logo()

            self._render_file_menu(app_state, file_mgr)
            self._render_edit_menu(app_state)
            self._render_view_menu(app_state, stage_proc)
            self._render_tools_menu(app_state, file_mgr)
            self._render_help_menu()

            # Render device control indicator after Support menu
            self._render_device_control_indicator()

            # Render Streamer indicator
            self._render_native_sync_indicator()

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

            if imgui.menu_item("Save Project", self._get_shortcut_display("save_project"),
                               selected=False, enabled=can_save)[0]:
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

            # Chapters submenu
            has_video = fm.video_path is not None
            has_chapters = has_video and len(fs_proc.video_chapters) > 0
            if imgui.begin_menu("Chapters..."):
                # Save Chapters (default location next to video)
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

                # Save Chapters As (custom location)
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

                # Load Chapters
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

                # Backup Chapters Now
                if _menu_item_simple("Backup Chapters Now", enabled=has_chapters):
                    chapter_mgr = self.app.chapter_manager
                    success = chapter_mgr.backup_chapters_manually(fs_proc.video_chapters, fm.video_path)
                    if not success:
                        self.app.logger.error("Failed to create chapter backup", extra={'status_message': True})

                # Clear All Chapters
                if _menu_item_simple("Clear All Chapters", enabled=has_chapters):
                    # Confirm before clearing
                    if hasattr(self.app, 'confirmation_needed'):
                        self.app.confirmation_needed = ('clear_chapters', len(fs_proc.video_chapters))
                    else:
                        fs_proc.video_chapters.clear()
                        self.app.logger.info("All chapters cleared", extra={'status_message': True})

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
                "Undo T1 Change", self._get_shortcut_display("undo_timeline1"),
                selected=False, enabled=can_undo1
            )[0]:
                fs_proc.perform_undo_redo(1, "undo")
            if imgui.menu_item(
                "Redo T1 Change", self._get_shortcut_display("redo_timeline1"),
                selected=False, enabled=can_redo1
            )[0]:
                fs_proc.perform_undo_redo(1, "redo")
            imgui.separator()

            # T2
            undo2 = fs_proc._get_undo_manager(2)
            can_undo2 = undo2.can_undo() if undo2 else False
            can_redo2 = undo2.can_redo() if undo2 else False
            if imgui.menu_item(
                "Undo T2 Change", self._get_shortcut_display("undo_timeline2"),
                selected=False, enabled=can_undo2
            )[0]:
                fs_proc.perform_undo_redo(2, "undo")
            if imgui.menu_item(
                "Redo T2 Change", self._get_shortcut_display("redo_timeline2"),
                selected=False, enabled=can_redo2
            )[0]:
                fs_proc.perform_undo_redo(2, "redo")
            imgui.end_menu()

    def _render_view_menu(self, app_state, stage_proc):
        if imgui.begin_menu("View", True):
            # UI Mode submenu
            self._render_ui_mode_submenu(app_state)

            # Layout submenu
            self._render_layout_submenu(app_state)

            imgui.separator()

            # Panels submenu (floating mode only)
            self._render_panels_submenu(app_state)

            imgui.separator()

            # Navigation submenu
            self._render_navigation_submenu(app_state)

            imgui.separator()

            # Timelines submenu
            self._render_timelines_submenu(app_state)

            imgui.separator()

            # Windows submenu
            self._render_windows_submenu(app_state)

            imgui.separator()

            # Video Overlays submenu
            self._render_video_overlays_submenu(app_state, stage_proc)

            imgui.separator()

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

            imgui.separator()

            # Show Advanced Options (last item in View menu)
            clicked, val = imgui.menu_item(
                "Show Advanced Options",
                selected=app_state.show_advanced_options
            )
            if clicked:
                app_state.show_advanced_options = val
                self.app.app_settings.set("show_advanced_options", val)
                self.app.project_manager.project_dirty = True

            imgui.end_menu()

    def _render_ui_mode_submenu(self, app_state):
        settings = self.app.app_settings
        if imgui.begin_menu("UI Mode"):
            current = app_state.ui_view_mode
            if _radio_line("Simple Mode", current == "simple"):
                if current != "simple":
                    app_state.ui_view_mode = "simple"
                    settings.set("ui_view_mode", "simple")
            if _radio_line("Expert Mode", current == "expert"):
                if current != "expert":
                    app_state.ui_view_mode = "expert"
                    settings.set("ui_view_mode", "expert")
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

    def _render_navigation_submenu(self, app_state):
        app = self.app
        pm = app.project_manager
        settings = app.app_settings

        if imgui.begin_menu("Navigation"):
            # Preview displays
            for label, attr in (
                ("Funscript Preview Bar", "show_funscript_timeline"),
                ("Heatmap", "show_heatmap"),
            ):
                cur = getattr(app_state, attr)
                clicked, val = imgui.menu_item(label, selected=cur)
                if clicked:
                    setattr(app_state, attr, val)
                    pm.project_dirty = True

            imgui.separator()

            # Preview options
            enhanced_preview = settings.get("enable_enhanced_funscript_preview", True)
            clicked, new_val = imgui.menu_item(
                "Enhanced Preview (Zoom + Frame)", selected=enhanced_preview
            )
            if clicked and new_val != enhanced_preview:
                settings.set("enable_enhanced_funscript_preview", new_val)

            use_simplified = settings.get("use_simplified_funscript_preview", False)
            clicked, new_val = imgui.menu_item(
                "Use Simplified Preview", selected=use_simplified
            )
            if clicked and new_val != use_simplified:
                settings.set("use_simplified_funscript_preview", new_val)
                app.app_state_ui.funscript_preview_dirty = True

            imgui.separator()

            # Full Width Navigation
            if not hasattr(app_state, 'full_width_nav'):
                app_state.full_width_nav = False
            clicked, val = imgui.menu_item(
                "Full Width Navigation",
                selected=app_state.full_width_nav
            )
            if clicked:
                app_state.full_width_nav = val
                pm.project_dirty = True

            imgui.end_menu()

    def _render_timelines_submenu(self, app_state):
        app = self.app
        pm = app.project_manager
        settings = app.app_settings

        if imgui.begin_menu("Timelines"):
            # Interactive editors only
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

            # Timeline editor buttons
            clicked, val = imgui.menu_item(
                "Show Timeline Editor Buttons",
                selected=app_state.show_timeline_editor_buttons
            )
            if clicked:
                app_state.show_timeline_editor_buttons = val
                settings.set("show_timeline_editor_buttons", val)
                pm.project_dirty = True

            imgui.end_menu()

    def _render_windows_submenu(self, app_state):
        pm = self.app.project_manager
        if imgui.begin_menu("Windows"):
            # Gauge windows
            for label, attr in (
                ("Script Gauge (Timeline 1)", "show_gauge_window_timeline1"),
                ("Script Gauge (Timeline 2)", "show_gauge_window_timeline2"),
            ):
                cur = getattr(app_state, attr)
                clicked, val = imgui.menu_item(label, selected=cur)
                if clicked:
                    setattr(app_state, attr, val)
                    pm.project_dirty = True

            # Movement bar
            clicked, val = imgui.menu_item(
                "Movement Bar", selected=app_state.show_lr_dial_graph
            )
            if clicked:
                app_state.show_lr_dial_graph = val
                pm.project_dirty = True

            # 3D Simulator
            clicked, val = imgui.menu_item(
                "3D Simulator", selected=app_state.show_simulator_3d
            )
            if clicked:
                app_state.show_simulator_3d = val
                pm.project_dirty = True

            # Chapter list
            if not hasattr(app_state, "show_chapter_list_window"):
                app_state.show_chapter_list_window = False

            clicked, val = imgui.menu_item(
                "Chapter List", selected=app_state.show_chapter_list_window
            )
            if clicked:
                app_state.show_chapter_list_window = val
                pm.project_dirty = True

            # Audio waveform
            clicked, _ = imgui.menu_item(
                "Audio Waveform", selected=app_state.show_audio_waveform
            )
            if clicked:
                self.app.toggle_waveform_visibility()
                pm.project_dirty = True

            imgui.end_menu()

    def _render_video_overlays_submenu(self, app_state, stage_proc):
        pm = self.app.project_manager
        app = self.app

        if imgui.begin_menu("Video Overlays"):
            # Video feed toggle
            clicked, val = imgui.menu_item(
                "Show Video Feed", selected=app_state.show_video_feed
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

            # Overlay modes (render windows on video)
            clicked, val = imgui.menu_item(
                "Gauges as Overlay",
                selected=app.app_settings.get('gauge_overlay_mode', False)
            )
            if clicked:
                app.app_settings.set('gauge_overlay_mode', val)

            clicked, val = imgui.menu_item(
                "Movement Bar as Overlay",
                selected=app.app_settings.get('movement_bar_overlay_mode', False)
            )
            if clicked:
                app.app_settings.set('movement_bar_overlay_mode', val)

            clicked, val = imgui.menu_item(
                "3D Simulator as Overlay",
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

            # Calibration & Analysis submenu
            if imgui.begin_menu("Calibration && Analysis"):
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

                if not hasattr(app_state, "show_autotuner_window"):
                    app_state.show_autotuner_window = False
                clicked, _ = imgui.menu_item(
                    "Performance Autotuner...",
                    selected=app_state.show_autotuner_window,
                )
                if clicked:
                    app_state.show_autotuner_window = not app_state.show_autotuner_window

                imgui.end_menu()

            imgui.separator()

            # Compilation submenu
            if imgui.begin_menu("Compilation"):
                if not hasattr(app, "tensorrt_compiler_window"):
                    app.tensorrt_compiler_window = None

                if _menu_item_simple("Compile YOLO to TensorRT (.engine)..."):
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

            imgui.separator()

            # Manage Generated Files (standalone)
            clicked, _ = imgui.menu_item(
                "Manage Generated Files...",
                selected=app_state.show_generated_file_manager,
            )
            if clicked:
                app.toggle_file_manager_window()

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

            # Support submenu
            if imgui.begin_menu("Support"):
                if _menu_item_simple("Become a Supporter"):
                    try:
                        webbrowser.open("https://ko-fi.com/k00gar")
                    except Exception as e:
                        if hasattr(app, 'logger') and app.logger:
                            app.logger.warning(f"Could not open Ko-fi link: {e}")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Unlock device control features and support development!\n"
                        "Supporters get access to:\n"
                        "• Hardware device integration (Handy, OSR2, etc.)\n"
                        "• Live tracking with device control\n"
                        "• Video + funscript synchronized playback\n"
                        "• Advanced device parameterization\n\n"
                        "After supporting, use !device_control command in Discord to get your folder!"
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
                    from application.utils.feature_detection import is_feature_available
                    if is_feature_available("device_control"):
                        imgui.set_tooltip("Device control not initialized\nCheck Device Control tab in Control Panel")
                    else:
                        imgui.set_tooltip("Supporter only feature\nCheck the Support menu for details")

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
                    from application.utils.feature_detection import is_feature_available
                    if is_feature_available("streamer"):
                        imgui.set_tooltip("Streamer not initialized\nCheck Streamer tab in Control Panel")
                    else:
                        imgui.set_tooltip("Supporter only feature\nCheck the Support menu for details")

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

