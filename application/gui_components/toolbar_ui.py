"""
Toolbar UI Component

Provides a horizontal toolbar with common actions organized in 7 sections:

1. File Operations: New, Open, Save, Export
2. Edit Operations: Undo/Redo for Timeline 1 & 2
3. Playback Controls: Play/Pause, Previous/Next Frame
4. Navigation: Previous/Next Point (↑↓)
5. Tracking Controls: Start/Stop Tracking (🤖 robot - red when active)
6. Funscript Actions: Auto-Simplify (🔧), Auto Post-Processing (✨), Ultimate Autotune (🚀)
7. Features: Streamer control (📡 satellite - only if module available)
8. View Toggles: Timeline 1/2 (1️⃣2️⃣), Chapter List (📚), 3D Simulator (📈)

Toggle visibility via View menu > Show Toolbar.

Required Icons:
All toolbar icons are defined in config/constants.py under UI_CONTROL_ICON_URLS.
The dependency checker automatically downloads missing icons on startup.

Displays at the top of the application, below the menu bar.
"""

import imgui
from application.utils import get_icon_texture_manager
from application.utils.button_styles import primary_button_style, destructive_button_style


class ToolbarUI:
    """Main application toolbar with common actions."""

    def __init__(self, app):
        self.app = app
        self._icon_size = 24  # Base icon size
        self._button_padding = 4

    def render(self):
        """Render the toolbar below the menu bar."""
        app = self.app
        app_state = app.app_state_ui

        # Check if toolbar should be shown
        if not hasattr(app_state, 'show_toolbar'):
            app_state.show_toolbar = True
        if not app_state.show_toolbar:
            return

        # Get viewport for positioning
        viewport = imgui.get_main_viewport()
        toolbar_height = self._icon_size + (self._button_padding * 2) + 8

        # Create an invisible full-width window for the toolbar
        imgui.set_next_window_position(viewport.pos.x, viewport.pos.y + imgui.get_frame_height())
        imgui.set_next_window_size(viewport.size.x, toolbar_height)

        # Window flags to make it look like a toolbar, not a floating window
        flags = (imgui.WINDOW_NO_TITLE_BAR |
                imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_NO_SCROLLBAR |
                imgui.WINDOW_NO_SCROLL_WITH_MOUSE |
                imgui.WINDOW_NO_COLLAPSE |
                imgui.WINDOW_NO_SAVED_SETTINGS |
                imgui.WINDOW_NO_BACKGROUND)

        imgui.begin("##MainToolbar", flags=flags)

        # Draw background manually
        draw_list = imgui.get_window_draw_list()
        win_pos = imgui.get_window_position()
        win_size = imgui.get_window_size()
        bg_color = imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 0.95)
        draw_list.add_rect_filled(
            win_pos[0], win_pos[1],
            win_pos[0] + win_size[0], win_pos[1] + win_size[1],
            bg_color
        )

        # Style for toolbar buttons
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (self._button_padding, self._button_padding))
        imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (8, 4))
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.2, 0.2, 0.5)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.3, 0.3, 0.7)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.15, 0.15, 0.15, 0.9)

        # Add small padding at start
        imgui.dummy(8, 0)
        imgui.same_line()

        icon_mgr = get_icon_texture_manager()
        btn_size = self._icon_size

        # --- FILE OPERATIONS SECTION ---
        self._render_file_section(icon_mgr, btn_size)

        imgui.same_line(spacing=12)
        self._render_separator()
        imgui.same_line(spacing=12)

        # --- EDIT OPERATIONS SECTION (Undo/Redo T1/T2) ---
        self._render_edit_section(icon_mgr, btn_size)

        imgui.same_line(spacing=12)
        self._render_separator()
        imgui.same_line(spacing=12)

        # --- PLAYBACK CONTROLS SECTION ---
        self._render_playback_section(icon_mgr, btn_size)

        imgui.same_line(spacing=12)
        self._render_separator()
        imgui.same_line(spacing=12)

        # --- NAVIGATION SECTION (Points) ---
        self._render_navigation_section(icon_mgr, btn_size)

        imgui.same_line(spacing=12)
        self._render_separator()
        imgui.same_line(spacing=12)

        # --- TRACKING CONTROLS SECTION ---
        self._render_tracking_section(icon_mgr, btn_size)

        imgui.same_line(spacing=12)
        self._render_separator()
        imgui.same_line(spacing=12)

        # --- FUNSCRIPT ACTIONS SECTION (Auto-simplify, Ultimate Autotune) ---
        self._render_funscript_actions_section(icon_mgr, btn_size)

        imgui.same_line(spacing=12)
        self._render_separator()
        imgui.same_line(spacing=12)

        # --- FEATURES SECTION (Streamer, Device Control - conditional) ---
        has_features = self._render_features_section(icon_mgr, btn_size)
        if has_features:
            imgui.same_line(spacing=12)
            self._render_separator()
            imgui.same_line(spacing=12)

        # --- VIEW TOGGLES SECTION ---
        self._render_view_section(icon_mgr, btn_size)

        imgui.pop_style_color(3)
        imgui.pop_style_var(2)

        imgui.end()

    def _render_separator(self):
        """Render a vertical separator line."""
        draw_list = imgui.get_window_draw_list()
        cursor_pos = imgui.get_cursor_screen_pos()
        height = self._icon_size + (self._button_padding * 2)

        # Draw vertical line
        color = imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 0.5)
        draw_list.add_line(
            cursor_pos[0], cursor_pos[1],
            cursor_pos[0], cursor_pos[1] + height,
            color, 1.0
        )

        # Advance cursor by 1 pixel for the line
        imgui.dummy(1, height)

    def _render_file_section(self, icon_mgr, btn_size):
        """Render file operation buttons."""
        app = self.app
        pm = app.project_manager
        fm = app.file_manager

        # New Project
        if self._toolbar_button(icon_mgr, 'document-new.png', btn_size, "New Project"):
            app.reset_project_state(for_new_project=True)
            pm.project_dirty = True

        imgui.same_line()

        # Open Project - use hyphen, not underscore!
        if self._toolbar_button(icon_mgr, 'folder-open.png', btn_size, "Open Project"):
            pm.open_project_dialog()

        imgui.same_line()

        # Save Project
        can_save = pm.project_file_path is not None
        if can_save:
            if self._toolbar_button(icon_mgr, 'save.png', btn_size, "Save Project"):
                pm.save_project_dialog()
        else:
            # Disabled state
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            self._toolbar_button(icon_mgr, 'save.png', btn_size, "Save Project (No project loaded)")
            imgui.pop_style_var()

        imgui.same_line()

        # Export Menu (dropdown)
        if self._toolbar_button(icon_mgr, 'export.png', btn_size, "Export Funscript"):
            imgui.open_popup("ExportPopup##Toolbar")

        # Export popup menu
        if imgui.begin_popup("ExportPopup##Toolbar"):
            if imgui.menu_item("Timeline 1...")[0]:
                self._export_timeline(1)
            if imgui.menu_item("Timeline 2...")[0]:
                self._export_timeline(2)
            imgui.end_popup()

    def _render_edit_section(self, icon_mgr, btn_size):
        """Render edit operation buttons (undo/redo for both timelines)."""
        app = self.app
        fs_proc = app.funscript_processor

        # Undo Timeline 1
        undo1 = fs_proc._get_undo_manager(1) if fs_proc else None
        can_undo1 = undo1.can_undo() if undo1 else False

        if can_undo1:
            if self._toolbar_button(icon_mgr, 'undo.png', btn_size, "Undo T1"):
                fs_proc.perform_undo_redo(1, "undo")
        else:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            self._toolbar_button(icon_mgr, 'undo.png', btn_size, "Undo T1 (Nothing to undo)")
            imgui.pop_style_var()

        imgui.same_line()

        # Redo Timeline 1
        can_redo1 = undo1.can_redo() if undo1 else False

        if can_redo1:
            if self._toolbar_button(icon_mgr, 'redo.png', btn_size, "Redo T1"):
                fs_proc.perform_undo_redo(1, "redo")
        else:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            self._toolbar_button(icon_mgr, 'redo.png', btn_size, "Redo T1 (Nothing to redo)")
            imgui.pop_style_var()

        imgui.same_line(spacing=4)
        imgui.text("|")  # Simple text separator
        imgui.same_line(spacing=4)

        # Undo Timeline 2
        undo2 = fs_proc._get_undo_manager(2) if fs_proc else None
        can_undo2 = undo2.can_undo() if undo2 else False

        if can_undo2:
            if self._toolbar_button(icon_mgr, 'undo.png', btn_size, "Undo T2"):
                fs_proc.perform_undo_redo(2, "undo")
        else:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            self._toolbar_button(icon_mgr, 'undo.png', btn_size, "Undo T2 (Nothing to undo)")
            imgui.pop_style_var()

        imgui.same_line()

        # Redo Timeline 2
        can_redo2 = undo2.can_redo() if undo2 else False

        if can_redo2:
            if self._toolbar_button(icon_mgr, 'redo.png', btn_size, "Redo T2"):
                fs_proc.perform_undo_redo(2, "redo")
        else:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            self._toolbar_button(icon_mgr, 'redo.png', btn_size, "Redo T2 (Nothing to redo)")
            imgui.pop_style_var()

    def _render_playback_section(self, icon_mgr, btn_size):
        """Render playback control buttons."""
        app = self.app
        processor = app.processor

        has_video = processor and processor.is_video_open() if processor else False
        is_playing = processor.is_processing and not processor.pause_event.is_set() if has_video else False

        # Play/Pause button
        if has_video:
            icon_name = 'pause.png' if is_playing else 'play.png'
            tooltip = "Pause" if is_playing else "Play"
            if self._toolbar_button(icon_mgr, icon_name, btn_size, tooltip):
                app.event_handlers.handle_playback_control("play_pause")
        else:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            self._toolbar_button(icon_mgr, 'play.png', btn_size, "Play (No video loaded)")
            imgui.pop_style_var()

        imgui.same_line()

        # Previous Frame
        if has_video:
            if self._toolbar_button(icon_mgr, 'prev-frame.png', btn_size, "Previous Frame"):
                app.event_handlers.handle_playback_control("prev_frame")
        else:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            self._toolbar_button(icon_mgr, 'prev-frame.png', btn_size, "Previous Frame (No video)")
            imgui.pop_style_var()

        imgui.same_line()

        # Next Frame
        if has_video:
            if self._toolbar_button(icon_mgr, 'next-frame.png', btn_size, "Next Frame"):
                app.event_handlers.handle_playback_control("next_frame")
        else:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            self._toolbar_button(icon_mgr, 'next-frame.png', btn_size, "Next Frame (No video)")
            imgui.pop_style_var()

    def _render_navigation_section(self, icon_mgr, btn_size):
        """Render navigation buttons (points and chapters)."""
        app = self.app
        has_video = app.processor and app.processor.is_video_open() if app.processor else False

        # Previous Point
        if has_video:
            if self._toolbar_button(icon_mgr, 'jump-start.png', btn_size, "Previous Point (↓)"):
                app.event_handlers.handle_jump_to_point("prev")
        else:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            self._toolbar_button(icon_mgr, 'jump-start.png', btn_size, "Previous Point (No video)")
            imgui.pop_style_var()

        imgui.same_line()

        # Next Point
        if has_video:
            if self._toolbar_button(icon_mgr, 'jump-end.png', btn_size, "Next Point (↑)"):
                app.event_handlers.handle_jump_to_point("next")
        else:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            self._toolbar_button(icon_mgr, 'jump-end.png', btn_size, "Next Point (No video)")
            imgui.pop_style_var()

    def _render_tracking_section(self, icon_mgr, btn_size):
        """Render tracking control buttons - Robot emoji with red when tracking."""
        app = self.app
        processor = app.processor

        if not processor:
            # No processor - show disabled
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            self._toolbar_button(icon_mgr, 'robot.png', btn_size, "Processor not initialized")
            imgui.pop_style_var()
            return

        # Check if live tracking is running (same logic as control_panel_ui.py)
        is_tracking = (processor.is_processing and
                      hasattr(processor, 'enable_tracker_processing') and
                      processor.enable_tracker_processing)

        if not is_tracking:
            # Start button - no special background, just normal state
            if self._toolbar_button(icon_mgr, 'robot.png', btn_size, "Start Tracking"):
                app.event_handlers.handle_start_live_tracker_click()
        else:
            # Tracking active - show with RED background to indicate "click to stop"
            imgui.pop_style_color(3)  # Remove default button colors
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.7, 0.0, 0.0, 0.7)  # Red background
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.85, 0.0, 0.0, 0.85)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.6, 0.0, 0.0, 0.9)

            if self._toolbar_button(icon_mgr, 'robot.png', btn_size, "Stop Tracking (Active)"):
                app.event_handlers.handle_reset_live_tracker_click()

            imgui.pop_style_color(3)
            # Restore default toolbar button colors
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.2, 0.2, 0.5)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.3, 0.3, 0.7)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.15, 0.15, 0.15, 0.9)

    def _render_funscript_actions_section(self, icon_mgr, btn_size):
        """Render funscript action buttons (auto-simplify, ultimate autotune)."""
        app = self.app
        settings = app.app_settings

        # Auto-Simplification Toggle - synced with Advanced tab
        auto_simplify = settings.get('funscript_point_simplification_enabled', True)
        if self._toolbar_toggle_button(icon_mgr, 'wrench.png', btn_size,
                                       "On-the-fly Funscript Simplification", auto_simplify):
            new_value = not auto_simplify
            settings.set('funscript_point_simplification_enabled', new_value)
            # Apply to active funscript if tracking
            if app.processor and hasattr(app.processor, 'active_funscript') and app.processor.active_funscript:
                app.processor.active_funscript.simplification_enabled = new_value

        imgui.same_line()

        # Auto Post-Processing Toggle - Sparkles emoji (✨)
        auto_post_proc = settings.get('enable_auto_post_processing', False)
        if self._toolbar_toggle_button(icon_mgr, 'sparkles.png', btn_size,
                                       "Automatic Post-Processing on Completion", auto_post_proc):
            new_value = not auto_post_proc
            settings.set('enable_auto_post_processing', new_value)
            app.logger.info(f"Automatic post-processing {'enabled' if new_value else 'disabled'}", extra={"status_message": True})

        imgui.same_line()

        # Ultimate Autotune button - Rocket emoji (🚀)
        has_video = app.processor and app.processor.is_video_open() if app.processor else False
        if has_video:
            if self._toolbar_button(icon_mgr, 'rocket.png', btn_size, "Ultimate Autotune (Timeline 1)"):
                app.trigger_ultimate_autotune_with_defaults(timeline_num=1)
        else:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
            self._toolbar_button(icon_mgr, 'rocket.png', btn_size, "Ultimate Autotune (No video)")
            imgui.pop_style_var()

    def _render_features_section(self, icon_mgr, btn_size):
        """Render supporter feature toggles (streamer, device control).

        Returns:
            bool: True if any features were rendered, False otherwise.
        """
        app = self.app
        rendered_any = False

        # Check for Streamer module (supporter feature)
        from application.utils.feature_detection import is_feature_available
        has_streamer = is_feature_available("streamer")
        has_device_control = is_feature_available("device_control")

        if has_streamer:
            control_panel = self.app.gui_instance.control_panel_ui if hasattr(self.app, 'gui_instance') else None

            # Initialize sync manager if not already done
            if control_panel and not hasattr(control_panel, '_native_sync_manager'):
                control_panel._native_sync_manager = None

            sync_mgr = getattr(control_panel, '_native_sync_manager', None) if control_panel else None

            # Initialize sync manager on first access if needed
            if control_panel and sync_mgr is None:
                try:
                    from streamer.integration_manager import NativeSyncManager
                    try:
                        # Try with app_logic parameter (newer streamer versions)
                        sync_mgr = NativeSyncManager(
                            self.app.processor,
                            logger=self.app.logger,
                            app_logic=self.app
                        )
                    except TypeError:
                        # Fall back to old signature (older streamer versions)
                        sync_mgr = NativeSyncManager(
                            self.app.processor,
                            logger=self.app.logger
                        )
                    control_panel._native_sync_manager = sync_mgr
                    self.app.logger.debug("Toolbar: Initialized NativeSyncManager")
                except Exception as e:
                    self.app.logger.debug(f"Toolbar: Could not initialize NativeSyncManager: {e}")

            # Show the button if we have the module (even if sync manager failed to init)
            is_running = False
            if sync_mgr:
                try:
                    status = sync_mgr.get_status()
                    is_running = status.get('is_running', False)
                except Exception as e:
                    self.app.logger.debug(f"Error getting streamer status: {e}")

            # Satellite emoji - clickable to start/stop streaming
            tooltip = "Stop Streaming Server" if is_running else "Start Streaming Server"
            if self._toolbar_toggle_button(icon_mgr, 'satellite.png', btn_size, tooltip, is_running):
                # Toggle streaming server
                if sync_mgr:
                    try:
                        if is_running:
                            self.app.logger.info("Toolbar: Stopping streaming server...")
                            sync_mgr.stop()
                        else:
                            self.app.logger.info("Toolbar: Starting streaming server...")
                            # Enable default settings
                            sync_mgr.enable_heresphere = True
                            sync_mgr.enable_xbvr_browser = True
                            sync_mgr.start()
                    except Exception as e:
                        self.app.logger.error(f"Toolbar: Failed to toggle streaming: {e}")
                        import traceback
                        self.app.logger.error(traceback.format_exc())
                else:
                    self.app.logger.warning("Toolbar: Streamer module available but NativeSyncManager failed to initialize")

            rendered_any = True

        # Device Control - hidden for now (not very useful as a toolbar button)
        # Users should use the Device Control tab instead

        return rendered_any

    def _render_view_section(self, icon_mgr, btn_size):
        """Render view toggle buttons."""
        app_state = self.app.app_state_ui

        # Timeline 1 Toggle - Keycap 1 emoji
        active = app_state.show_funscript_interactive_timeline if hasattr(app_state, 'show_funscript_interactive_timeline') else True
        if self._toolbar_toggle_button(icon_mgr, 'keycap-1.png', btn_size, "Toggle Timeline 1", active):
            app_state.show_funscript_interactive_timeline = not active
            self.app.project_manager.project_dirty = True

        imgui.same_line()

        # Timeline 2 Toggle - Keycap 2 emoji
        active = app_state.show_funscript_interactive_timeline2 if hasattr(app_state, 'show_funscript_interactive_timeline2') else False
        if self._toolbar_toggle_button(icon_mgr, 'keycap-2.png', btn_size, "Toggle Timeline 2", active):
            app_state.show_funscript_interactive_timeline2 = not active
            self.app.project_manager.project_dirty = True

        imgui.same_line()

        # Chapter List Toggle - Books emoji (📚)
        if not hasattr(app_state, 'show_chapter_list_window'):
            app_state.show_chapter_list_window = False
        active = app_state.show_chapter_list_window
        if self._toolbar_toggle_button(icon_mgr, 'books.png', btn_size, "Chapter List", active):
            app_state.show_chapter_list_window = not active
            self.app.project_manager.project_dirty = True

        imgui.same_line()

        # 3D Simulator Toggle - Chart emoji (📈)
        active = app_state.show_simulator_3d if hasattr(app_state, 'show_simulator_3d') else False
        if self._toolbar_toggle_button(icon_mgr, 'chart-increasing.png', btn_size, "3D Simulator", active):
            app_state.show_simulator_3d = not active
            self.app.project_manager.project_dirty = True

    def _toolbar_button(self, icon_mgr, icon_name, size, tooltip):
        """
        Render a toolbar button with icon.

        Returns:
            bool: True if button was clicked
        """
        icon_tex, _, _ = icon_mgr.get_icon_texture(icon_name)

        if icon_tex:
            clicked = imgui.image_button(icon_tex, size, size)
        else:
            # Fallback to small labeled button if icon fails to load
            # Extract a short label from the icon name (e.g., "folder-open.png" -> "Open")
            label = icon_name.replace('.png', '').replace('-', ' ').title().split()[0][:4]
            clicked = imgui.button(f"{label}###{icon_name}", size, size)

        if imgui.is_item_hovered():
            imgui.set_tooltip(tooltip)

        return clicked

    def _toolbar_toggle_button(self, icon_mgr, icon_name, size, tooltip, is_active):
        """
        Render a toggle button with active state indication.

        Returns:
            bool: True if button was clicked
        """
        # Highlight active buttons with a different tint
        if is_active:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 0.8)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.4, 0.6, 0.8, 0.9)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.2, 0.4, 0.6, 1.0)

        clicked = self._toolbar_button(icon_mgr, icon_name, size,
                                      f"{tooltip} ({'Active' if is_active else 'Inactive'})")

        if is_active:
            imgui.pop_style_color(3)

        return clicked

    def _export_timeline(self, timeline_num):
        """Export funscript from specified timeline."""
        import os

        if not self.app.gui_instance or not self.app.gui_instance.file_dialog:
            self.app.logger.warning("File dialog not available", extra={"status_message": True})
            return

        video_path = self.app.file_manager.video_path
        output_folder_base = self.app.app_settings.get("output_folder_path", "output")
        initial_path = output_folder_base

        if timeline_num == 1:
            initial_filename = "timeline1.funscript"
        else:
            initial_filename = "timeline2.funscript"

        if video_path:
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            initial_path = os.path.join(output_folder_base, video_basename)
            if timeline_num == 1:
                initial_filename = f"{video_basename}.funscript"
            else:
                initial_filename = f"{video_basename}_t2.funscript"

        if not os.path.isdir(initial_path):
            os.makedirs(initial_path, exist_ok=True)

        self.app.gui_instance.file_dialog.show(
            is_save=True,
            title=f"Export Funscript from Timeline {timeline_num}",
            extension_filter="Funscript Files (*.funscript),*.funscript",
            callback=lambda filepath: self.app.file_manager.save_funscript_from_timeline(filepath, timeline_num),
            initial_path=initial_path,
            initial_filename=initial_filename
        )
