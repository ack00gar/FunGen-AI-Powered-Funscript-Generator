"""
Toolbar UI Component

Provides a horizontal toolbar with common actions organized into labeled sections:

Sections (each with a label above the icons):
- PROJECT: New, Open, Save, Export operations
- PLAYBACK: Play/Pause, Previous/Next Frame, Speed modes
- TIMELINE EDIT: Axis dropdown + contextual Undo/Redo/Autotune
- VIEW: Chapter List, 3D Simulator
- TOOLS: Streamer, Device Control, Batch Processing (add-ons)

Toggle visibility via View menu > Show Toolbar.

Required Icons:
All toolbar icons are defined in config/constants.py under UI_CONTROL_ICON_URLS.
The dependency checker automatically downloads missing icons on startup.

Displays at the top of the application, below the menu bar.
"""

import imgui
from application.utils import get_icon_texture_manager
from application.utils.feature_detection import is_feature_available as _is_feature_available
from config.element_group_colors import ToolbarColors

try:
    from video.audio_player import SOUNDDEVICE_AVAILABLE
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class ToolbarUI:
    """Main application toolbar with common actions."""

    def __init__(self, app):
        self.app = app
        self._icon_size = 24  # Base icon size
        self._button_padding = 4
        self._label_height = 14  # Height for section labels
        self._label_spacing = 2  # Space between label and buttons
        self._selected_edit_timeline = 1
        # Pre-computed u32 colors (set on first render when imgui context exists)
        self._separator_color_u32 = None
        self._label_color_u32 = None
        # Cached axis items for edit section (never None — initialized with defaults)
        self._cached_axis_assignments = {}
        self._cached_axis_items = [(1, "Stroke (T1)")]
        self._cached_axis_labels = ["Stroke (T1)"]
        self._cached_axis_valid_tls = [1]

    def _get_button_row_height(self):
        """Total pixel height of a toolbar image button (icon + frame padding)."""
        return self._icon_size + self._button_padding * 2

    def _begin_vcenter(self):
        """Begin vertical centering context for a non-button widget in the toolbar.

        Call before the widget (after same_line if needed).
        Starts a group and adds a top dummy to push the widget down
        so it's vertically centered within the button row.
        """
        row_h = self._get_button_row_height()
        frame_h = imgui.get_frame_height()
        self._vcenter_offset = max(0.0, (row_h - frame_h) / 2.0)
        imgui.begin_group()
        if self._vcenter_offset > 0:
            imgui.dummy(0, self._vcenter_offset)

    def _end_vcenter(self):
        """End vertical centering context. Ensures the group occupies full button row height."""
        imgui.end_group()

    def get_toolbar_height(self):
        """Get the total height of the toolbar including labels.

        Returns:
            int: Total toolbar height in pixels
        """
        return self._label_height + self._label_spacing + self._icon_size + (self._button_padding * 2) + 10

    def _apply_button_active(self):
        """Apply active/toggled highlight — matches sidebar accent style."""
        imgui.pop_style_color(3)
        imgui.push_style_color(imgui.COLOR_BUTTON, *ToolbarColors.ACTIVE_BUTTON)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *ToolbarColors.ACTIVE_BUTTON_HOVERED)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *ToolbarColors.ACTIVE_BUTTON_PRESSED)

    def _apply_button_default(self):
        """Restore default transparent button colors."""
        imgui.pop_style_color(3)
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.5, 0.5, 0.55, 0.2)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.4, 0.4, 0.45, 0.35)

    def render(self):
        """Render the toolbar below the menu bar."""
        app = self.app
        app_state = app.app_state_ui

        # Cache feature detection flags for this frame
        self._feat_streamer = _is_feature_available("streamer")
        self._feat_device = _is_feature_available("device_control")
        self._feat_supporter = _is_feature_available("patreon_features")

        # Check if toolbar should be shown
        if not hasattr(app_state, 'show_toolbar'):
            app_state.show_toolbar = True
        if not app_state.show_toolbar:
            return

        # Lazy-init u32 colors (needs imgui context)
        if self._separator_color_u32 is None:
            self._separator_color_u32 = imgui.get_color_u32_rgba(*ToolbarColors.SEPARATOR)
            self._label_color_u32 = imgui.get_color_u32_rgba(*ToolbarColors.LABEL_TEXT)

        # Get viewport for positioning
        viewport = imgui.get_main_viewport()
        # Get toolbar height (includes label space)
        toolbar_height = self.get_toolbar_height()

        # Create an invisible full-width window for the toolbar
        imgui.set_next_window_position(viewport.pos.x, viewport.pos.y + imgui.get_frame_height())
        imgui.set_next_window_size(viewport.size.x, toolbar_height)

        # Window flags — standard docked window (matches other UI components)
        flags = (imgui.WINDOW_NO_TITLE_BAR |
                imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_NO_SCROLLBAR |
                imgui.WINDOW_NO_SCROLL_WITH_MOUSE |
                imgui.WINDOW_NO_COLLAPSE |
                imgui.WINDOW_NO_SAVED_SETTINGS)

        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
        imgui.begin("##MainToolbar", flags=flags)
        imgui.pop_style_var()

        # Style for toolbar buttons — transparent floating style
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (self._button_padding, self._button_padding))
        imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (8, 4))
        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 6.0)
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.5, 0.5, 0.55, 0.2)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.4, 0.4, 0.45, 0.35)

        # Add small padding at start
        imgui.dummy(8, 0)
        imgui.same_line()

        icon_mgr = get_icon_texture_manager()
        btn_size = self._icon_size

        # --- FILE OPERATIONS SECTION ---
        self._begin_toolbar_section("Project")
        self._render_file_section(icon_mgr, btn_size)
        self._end_toolbar_section()

        imgui.same_line(spacing=12)
        self._render_separator()
        imgui.same_line(spacing=12)

        # --- PLAYBACK CONTROLS SECTION ---
        self._begin_toolbar_section("Playback")
        self._render_playback_section(icon_mgr, btn_size)
        self._end_toolbar_section()

        imgui.same_line(spacing=12)
        self._render_separator()
        imgui.same_line(spacing=12)

        # --- AUDIO CONTROLS SECTION ---
        self._begin_toolbar_section("Audio")
        self._render_audio_section(icon_mgr, btn_size)
        self._end_toolbar_section()

        imgui.same_line(spacing=12)
        self._render_separator()
        imgui.same_line(spacing=12)

        # --- EDIT OPERATIONS SECTION (Undo/Redo T1/T2) ---
        self._begin_toolbar_section("Timeline Edit")
        self._render_edit_section(icon_mgr, btn_size)
        self._end_toolbar_section()

        imgui.same_line(spacing=12)
        self._render_separator()
        imgui.same_line(spacing=12)

        # --- VIEW TOGGLES SECTION ---
        self._begin_toolbar_section("View")
        self._render_view_section(icon_mgr, btn_size)
        self._end_toolbar_section()

        # --- TOOLS SECTION (always visible — grayed out if locked) ---
        imgui.same_line(spacing=12)
        self._render_separator()
        imgui.same_line(spacing=12)

        self._begin_toolbar_section("Tools")
        self._render_features_section(icon_mgr, btn_size)
        self._end_toolbar_section()

        imgui.pop_style_color(3)
        imgui.pop_style_var(3)

        imgui.end()

    def _render_separator(self):
        """Render a vertical separator line."""
        draw_list = imgui.get_window_draw_list()
        cursor_pos = imgui.get_cursor_screen_pos()
        height = self._label_height + self._label_spacing + self._icon_size + (self._button_padding * 2)

        draw_list.add_line(
            cursor_pos[0], cursor_pos[1],
            cursor_pos[0], cursor_pos[1] + height,
            self._separator_color_u32, 1.0
        )
        imgui.dummy(1, height)

    def _begin_toolbar_section(self, label_text):
        """Begin a toolbar section with a centered label above the buttons."""
        # Start outer group for the entire section
        imgui.begin_group()

        # Store starting cursor position
        self._section_start_x = imgui.get_cursor_pos_x()

        # We'll draw the label after we know the section width
        # For now, just reserve space for the label
        imgui.dummy(0, self._label_height)  # Reserve vertical space for label

        # Store this label text for later rendering
        self._pending_section_label = label_text

        # Start inner group for buttons (this will give us the section width)
        imgui.begin_group()

    def _end_toolbar_section(self):
        """End a toolbar section and render the centered label."""
        # End the button group
        imgui.end_group()

        # Get the width of the button group we just rendered
        section_size = imgui.get_item_rect_size()
        section_width = section_size[0]

        # End outer group
        imgui.end_group()

        # Now render the label centered above the section
        if hasattr(self, '_pending_section_label') and self._pending_section_label:
            label_text = self._pending_section_label.upper()
            text_size = imgui.calc_text_size(label_text)

            # Calculate centered position
            label_x = self._section_start_x + (section_width - text_size[0]) / 2

            # Get current cursor position to restore later
            current_pos = imgui.get_cursor_pos()

            # Draw label at calculated centered position (above the buttons)
            draw_list = imgui.get_window_draw_list()
            # Position is relative to window, need screen coordinates
            window_pos = imgui.get_window_position()
            label_y = window_pos[1] + 4  # Small top padding

            # Render centered label text
            draw_list.add_text(window_pos[0] + label_x, label_y,
                             self._label_color_u32,
                             label_text)

            # Clear the pending label
            self._pending_section_label = None

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
            self._toolbar_button_disabled(icon_mgr, 'save.png', btn_size, "Save Project (No project loaded)")

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
        """Render axis dropdown + contextual Undo/Redo/Autotune."""
        app = self.app
        fs_proc = app.funscript_processor
        has_video = app.processor and app.processor.is_video_open() if app.processor else False

        # --- Axis dropdown (cached) ---
        assignments = {}
        if hasattr(app, 'tracker') and app.tracker and hasattr(app.tracker, 'funscript'):
            try:
                assignments = app.tracker.funscript.get_axis_assignments()
            except Exception:
                pass
        if not assignments:
            assignments = {1: "stroke"}

        if assignments != self._cached_axis_assignments:
            self._cached_axis_assignments = assignments.copy()
            axis_items = []
            for tl_num in sorted(assignments):
                axis_name = assignments[tl_num].capitalize()
                axis_items.append((tl_num, f"{axis_name} (T{tl_num})"))
            self._cached_axis_items = axis_items
            self._cached_axis_valid_tls = [item[0] for item in axis_items]
            self._cached_axis_labels = [item[1] for item in axis_items]

        valid_tls = self._cached_axis_valid_tls
        labels = self._cached_axis_labels

        if self._selected_edit_timeline not in valid_tls:
            self._selected_edit_timeline = valid_tls[0]

        current_idx = valid_tls.index(self._selected_edit_timeline)

        self._begin_vcenter()
        imgui.push_item_width(120)
        changed, new_idx = imgui.combo("##EditAxisCombo", current_idx, labels)
        imgui.pop_item_width()
        self._end_vcenter()
        if changed:
            self._selected_edit_timeline = valid_tls[new_idx]

        selected_tl = self._selected_edit_timeline

        imgui.same_line(spacing=8)

        # --- Undo/Redo for selected timeline ---
        undo_mgr = fs_proc._get_undo_manager(selected_tl) if fs_proc else None
        can_undo = undo_mgr.can_undo() if undo_mgr else False
        can_redo = undo_mgr.can_redo() if undo_mgr else False

        if can_undo:
            if self._toolbar_button(icon_mgr, 'undo.png', btn_size, f"Undo (T{selected_tl})"):
                fs_proc.perform_undo_redo(selected_tl, "undo")
        else:
            self._toolbar_button_disabled(icon_mgr, 'undo.png', btn_size, "Undo (Nothing to undo)")

        imgui.same_line()

        if can_redo:
            if self._toolbar_button(icon_mgr, 'redo.png', btn_size, f"Redo (T{selected_tl})"):
                fs_proc.perform_undo_redo(selected_tl, "redo")
        else:
            self._toolbar_button_disabled(icon_mgr, 'redo.png', btn_size, "Redo (Nothing to redo)")

        imgui.same_line(spacing=8)

        # --- Autotune for selected timeline ---
        if has_video:
            if self._toolbar_button(icon_mgr, 'magic-wand.png', btn_size, f"Ultimate Autotune (T{selected_tl})"):
                app.trigger_ultimate_autotune_with_defaults(timeline_num=selected_tl)
        else:
            self._toolbar_button_disabled(icon_mgr, 'magic-wand.png', btn_size, "Ultimate Autotune (No video)")

    def _render_playback_section(self, icon_mgr, btn_size):
        """Render playback control buttons."""
        app = self.app
        processor = app.processor

        has_video = processor and processor.is_video_open() if processor else False
        is_playing = processor.is_processing and not processor.pause_event.is_set() if has_video else False

        # Jump Start
        if has_video:
            if self._toolbar_button(icon_mgr, 'jump-start.png', btn_size, "Jump to Start (HOME)"):
                app.event_handlers.handle_playback_control("jump_start")
        else:
            self._toolbar_button_disabled(icon_mgr, 'jump-start.png', btn_size, "Jump to Start (No video)")

        imgui.same_line()

        # Previous Frame
        if has_video:
            if self._toolbar_button(icon_mgr, 'prev-frame.png', btn_size, "Previous Frame (LEFT)"):
                app.event_handlers.handle_playback_control("prev_frame")
        else:
            self._toolbar_button_disabled(icon_mgr, 'prev-frame.png', btn_size, "Previous Frame (No video)")

        imgui.same_line()

        # Play/Pause button (green when playing)
        if has_video:
            if is_playing:
                self._apply_button_active()

            icon_name = 'pause.png' if is_playing else 'play.png'
            tooltip = "Pause (SPACE)" if is_playing else "Play (SPACE)"
            if self._toolbar_button(icon_mgr, icon_name, btn_size, tooltip):
                app.event_handlers.handle_playback_control("play_pause")

            if is_playing:
                self._apply_button_default()
        else:
            self._toolbar_button_disabled(icon_mgr, 'play.png', btn_size, "Play (No video loaded)")

        imgui.same_line()

        # Stop button
        if has_video:
            if self._toolbar_button(icon_mgr, 'stop.png', btn_size, "Stop"):
                app.event_handlers.handle_playback_control("stop")
        else:
            self._toolbar_button_disabled(icon_mgr, 'stop.png', btn_size, "Stop (No video)")

        imgui.same_line()

        # Next Frame
        if has_video:
            if self._toolbar_button(icon_mgr, 'next-frame.png', btn_size, "Next Frame (RIGHT)"):
                app.event_handlers.handle_playback_control("next_frame")
        else:
            self._toolbar_button_disabled(icon_mgr, 'next-frame.png', btn_size, "Next Frame (No video)")

        imgui.same_line()

        # Jump End
        if has_video:
            if self._toolbar_button(icon_mgr, 'jump-end.png', btn_size, "Jump to End (END)"):
                app.event_handlers.handle_playback_control("jump_end")
        else:
            self._toolbar_button_disabled(icon_mgr, 'jump-end.png', btn_size, "Jump to End (No video)")

        # Separator before video/speed controls (same style as between Timeline 1 and 2)
        imgui.same_line(spacing=12)
        self._render_separator()
        imgui.same_line(spacing=12)

        # Show/Hide Video toggle — highlight only when video is visible
        app_state = app.app_state_ui
        show_video = app_state.show_video_feed if hasattr(app_state, 'show_video_feed') else True

        if show_video:
            self._apply_button_active()

        # Icon shows the action: 18+ icon to hide video, camera icon to show video
        icon_name = 'video-hide.png' if show_video else 'video-show.png'
        tooltip = "Hide Video (F)" if show_video else "Show Video (F)"
        if self._toolbar_button(icon_mgr, icon_name, btn_size, tooltip):
            if hasattr(app_state, 'show_video_feed'):
                app_state.show_video_feed = not app_state.show_video_feed
                app.app_settings.set("show_video_feed", app_state.show_video_feed)

        if show_video:
            self._apply_button_default()

        imgui.same_line()

        # Playback Speed Mode buttons (highlight when active)
        from config.constants import ProcessingSpeedMode
        current_speed_mode = app_state.selected_processing_speed_mode

        # Real Time button
        if current_speed_mode == ProcessingSpeedMode.REALTIME:
            self._apply_button_active()

        if self._toolbar_button(icon_mgr, 'speed-realtime.png', btn_size, "Real Time Speed (matches video FPS)"):
            app_state.selected_processing_speed_mode = ProcessingSpeedMode.REALTIME

        if current_speed_mode == ProcessingSpeedMode.REALTIME:
            self._apply_button_default()

        imgui.same_line()

        # Slow-mo button
        if current_speed_mode == ProcessingSpeedMode.SLOW_MOTION:
            self._apply_button_active()

        slo_mo_fps = getattr(app_state, 'slow_motion_fps', 10.0)
        if self._toolbar_button(icon_mgr, 'speed-slowmo.png', btn_size, f"Slow Motion ({slo_mo_fps:.0f} FPS)"):
            app_state.selected_processing_speed_mode = ProcessingSpeedMode.SLOW_MOTION

        if current_speed_mode == ProcessingSpeedMode.SLOW_MOTION:
            self._apply_button_default()

        # Slow-mo FPS slider (only visible when slow-mo is active)
        if current_speed_mode == ProcessingSpeedMode.SLOW_MOTION:
            imgui.same_line()
            imgui.push_item_width(80)
            changed, new_fps = imgui.slider_float("##SloMoFPS", slo_mo_fps, 1.0, 30.0, "%.0f FPS")
            if changed:
                app_state.slow_motion_fps = new_fps
            imgui.pop_item_width()
            # Reset to default on double-click
            if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                app_state.slow_motion_fps = 10.0

        imgui.same_line()

        # Max Speed button
        if current_speed_mode == ProcessingSpeedMode.MAX_SPEED:
            self._apply_button_active()

        if self._toolbar_button(icon_mgr, 'speed-max.png', btn_size, "Max Speed (no frame delay)"):
            app_state.selected_processing_speed_mode = ProcessingSpeedMode.MAX_SPEED

        if current_speed_mode == ProcessingSpeedMode.MAX_SPEED:
            self._apply_button_default()

    def _render_audio_section(self, icon_mgr, btn_size):
        """Render audio mute toggle and volume slider."""
        app = self.app
        settings = app.app_settings

        sd_ok = SOUNDDEVICE_AVAILABLE
        player_ok = getattr(app, '_audio_player', None) is not None
        has_audio_system = sd_ok and player_ok
        is_muted = settings.get("audio_muted", False)
        vol = getattr(app, '_audio_volume_live', settings.get("audio_volume", 0.8))

        disabled = not has_audio_system
        if disabled:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)

        # --- Mute / unmute toggle ---
        icon = 'speaker-muted.png' if is_muted else 'speaker-high.png'

        if is_muted and not disabled:
            self._apply_button_active()

        # Tooltip
        if disabled:
            reasons = []
            if not sd_ok:
                reasons.append("sounddevice not installed")
            if not player_ok:
                reasons.append("audio player not initialized")
            tooltip = f"Audio unavailable ({', '.join(reasons)})"
        else:
            tooltip = "Unmute Audio" if is_muted else "Mute Audio"

        if self._toolbar_button(icon_mgr, icon, btn_size, tooltip):
            if not disabled:
                is_muted = not is_muted
                settings.set("audio_muted", is_muted)
                if app._audio_sync:
                    app._audio_sync.update_settings(vol, is_muted)

        if is_muted and not disabled:
            self._apply_button_default()

        # --- Volume slider (always visible) ---
        imgui.same_line()
        self._begin_vcenter()
        imgui.push_item_width(100)
        pct_label = f"{int(vol * 100)}%%"
        changed, vol = imgui.slider_float("##vol", vol, 0.0, 1.0, pct_label)
        if changed and not disabled:
            # Auto-mute when slider hits 0, auto-unmute when dragged above 0
            if vol <= 0.0 and not is_muted:
                settings.set("audio_muted", True)
            elif vol > 0.0 and is_muted:
                settings.set("audio_muted", False)
            # Update the player live — volume is persisted to settings on quit only
            app._audio_volume_live = vol
            if app._audio_sync:
                app._audio_sync.update_settings(vol, settings.get("audio_muted", False))
        imgui.pop_item_width()
        self._end_vcenter()

        if disabled:
            imgui.pop_style_var()

    def _render_navigation_section(self, icon_mgr, btn_size):
        """Render navigation buttons (points and chapters)."""
        app = self.app
        has_video = app.processor and app.processor.is_video_open() if app.processor else False

        # Previous Point
        if has_video:
            if self._toolbar_button(icon_mgr, 'jump-start.png', btn_size, "Previous Point (Down)"):
                app.event_handlers.handle_jump_to_point("prev")
        else:
            self._toolbar_button_disabled(icon_mgr, 'jump-start.png', btn_size, "Previous Point (No video)")

        imgui.same_line()

        # Next Point
        if has_video:
            if self._toolbar_button(icon_mgr, 'jump-end.png', btn_size, "Next Point (Up)"):
                app.event_handlers.handle_jump_to_point("next")
        else:
            self._toolbar_button_disabled(icon_mgr, 'jump-end.png', btn_size, "Next Point (No video)")

    def _render_features_section(self, icon_mgr, btn_size):
        """Render add-on feature toggles (streamer, device control, batch).

        Always renders all 3 tool buttons. Unavailable features show as grayed-out
        with ko-fi tooltips for feature discovery.
        """
        has_streamer = self._feat_streamer
        has_device_control = self._feat_device
        has_supporter = self._feat_supporter
        rendered_any = False

        # --- Streamer button ---
        if has_streamer:
            self._render_streamer_button_active(icon_mgr, btn_size, is_first=True)
            rendered_any = True
        else:
            self._toolbar_button_disabled(
                icon_mgr, 'satellite.png', btn_size,
                "Streamer - Stream from XBVR & Stash.\nAdd-on available at ko-fi.com/k00gar"
            )
            rendered_any = True

        # --- Device Control button ---
        imgui.same_line()
        if has_device_control:
            self._render_device_button_active(icon_mgr, btn_size)
        else:
            self._toolbar_button_disabled(
                icon_mgr, 'flashlight.png', btn_size,
                "Device Control - Control OSR2, Handy & more.\nAdd-on available at ko-fi.com/k00gar"
            )

        # --- Batch Processing button ---
        imgui.same_line()
        if has_supporter:
            self._apply_button_active()
            self._toolbar_button(icon_mgr, 'sidebar-batch.png', btn_size,
                                 "Batch Processing - Early access trackers, batch queue.")
            self._apply_button_default()
        else:
            self._toolbar_button_disabled(
                icon_mgr, 'sidebar-batch.png', btn_size,
                "Batch Processing - Early access trackers, batch queue.\nMonthly ko-fi.com/k00gar subscription."
            )

    def _render_streamer_button_active(self, icon_mgr, btn_size, is_first=False):
        """Render the active (available) streamer button with full toggle logic."""
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
                    sync_mgr = NativeSyncManager(
                        self.app.processor,
                        logger=self.app.logger,
                        app_logic=self.app
                    )
                except TypeError:
                    sync_mgr = NativeSyncManager(
                        self.app.processor,
                        logger=self.app.logger
                    )
                control_panel._native_sync_manager = sync_mgr
                self.app.logger.debug("Toolbar: Initialized NativeSyncManager")
            except Exception as e:
                self.app.logger.debug(f"Toolbar: Could not initialize NativeSyncManager: {e}")

        is_running = False
        if sync_mgr:
            try:
                status = sync_mgr.get_status()
                is_running = status.get('is_running', False)
            except Exception as e:
                self.app.logger.debug(f"Error getting streamer status: {e}")

        # Satellite emoji — highlight when active
        if is_running:
            self._apply_button_active()

        tooltip = "Stop Streaming Server" if is_running else "Start Streaming Server"
        if self._toolbar_button(icon_mgr, 'satellite.png', btn_size, tooltip):
            if sync_mgr:
                try:
                    if is_running:
                        self.app.logger.info("Toolbar: Stopping streaming server...")
                        sync_mgr.stop()
                    else:
                        self.app.logger.info("Toolbar: Starting streaming server...")
                        sync_mgr.enable_heresphere = True
                        sync_mgr.enable_xbvr_browser = True
                        sync_mgr.start()
                except Exception as e:
                    self.app.logger.error(f"Toolbar: Failed to toggle streaming: {e}")
                    import traceback
                    self.app.logger.error(traceback.format_exc())
            else:
                self.app.logger.warning("Toolbar: Streamer module available but NativeSyncManager failed to initialize")

        if is_running:
            self._apply_button_default()

    def _render_device_button_active(self, icon_mgr, btn_size):
        """Render the active (available) device control button with full toggle logic."""
        control_panel_ui = getattr(self.app.gui_instance, 'control_panel_ui', None) if hasattr(self.app, 'gui_instance') else None
        device_manager = getattr(control_panel_ui, 'device_manager', None) if control_panel_ui else None

        is_connected = False
        if device_manager:
            try:
                is_connected = bool(device_manager.is_connected())
            except Exception as e:
                self.app.logger.error(f"Toolbar: Error checking device connection status: {e}")
                import traceback
                self.app.logger.error(traceback.format_exc())

        # Flashlight emoji — highlight when connected
        if is_connected:
            self._apply_button_active()

        tooltip = "Disconnect Device" if is_connected else "Connect Device"
        if self._toolbar_button(icon_mgr, 'flashlight.png', btn_size, tooltip):
            if device_manager:
                try:
                    if is_connected:
                        self.app.logger.info("Toolbar: Disconnecting device...")
                        import asyncio
                        import threading

                        def run_disconnect():
                            try:
                                # Use device manager's worker loop if available
                                worker_loop = getattr(device_manager, '_worker_loop', None)
                                if worker_loop and worker_loop.is_running():
                                    future = asyncio.run_coroutine_threadsafe(device_manager.stop(), worker_loop)
                                    future.result(timeout=10)
                                else:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    try:
                                        loop.run_until_complete(device_manager.stop())
                                    finally:
                                        loop.close()
                            except Exception as e:
                                self.app.logger.error(f"Toolbar: Error during disconnect: {e}")

                            self.app.logger.info("Toolbar: Device disconnected successfully")

                        thread = threading.Thread(target=run_disconnect, daemon=True)
                        thread.start()
                    else:
                        self.app.logger.info("Toolbar: Auto-connecting to Handy device...")
                        self._auto_connect_handy()
                except Exception as e:
                    self.app.logger.error(f"Toolbar: Failed to toggle device connection: {e}")
                    import traceback
                    self.app.logger.error(traceback.format_exc())
            else:
                self.app.logger.info("Toolbar: DeviceManager not initialized, auto-connecting Handy...")
                self._auto_connect_handy()

        if is_connected:
            self._apply_button_default()

        # Script Loaded indicator — only show when device is connected
        if is_connected and device_manager:
            imgui.same_line()

            script_loaded = device_manager.has_prepared_handy_devices() if hasattr(device_manager, 'has_prepared_handy_devices') else False

            # Script loaded → active highlight; not loaded → default
            if script_loaded:
                self._apply_button_active()
            tooltip = "Script Loaded - Click to reload" if script_loaded else "No Script Loaded - Click to upload"

            if self._toolbar_button(icon_mgr, 'page-facing-up.png', btn_size, tooltip):
                self._upload_script_to_device(device_manager)

            if script_loaded:
                self._apply_button_default()

                # Sync toggle button for Handy
                imgui.same_line()
                is_handy_playing = device_manager.is_handy_playing() if hasattr(device_manager, 'is_handy_playing') else False

                if is_handy_playing:
                    self._apply_button_active()
                tooltip = "Handy Synced - Click to Pause" if is_handy_playing else "Handy Paused - Click to Resume"

                if self._toolbar_button(icon_mgr, 'counterclockwise-arrows.png', btn_size, tooltip):
                    self._toggle_handy_playback(device_manager)

                if is_handy_playing:
                    self._apply_button_default()

    def _render_view_section(self, icon_mgr, btn_size):
        """Render view toggle buttons (chapter list, simulator)."""
        app_state = self.app.app_state_ui

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

    def _toolbar_button_disabled(self, icon_mgr, icon, btn_size, tooltip):
        """Render a disabled (grayed-out) toolbar button."""
        imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
        self._toolbar_button(icon_mgr, icon, btn_size, tooltip)
        imgui.pop_style_var()

    def _toolbar_toggle_button(self, icon_mgr, icon_name, size, tooltip, is_active):
        """
        Render a toggle button with active state indication.

        Returns:
            bool: True if button was clicked
        """
        if is_active:
            self._apply_button_active()

        clicked = self._toolbar_button(icon_mgr, icon_name, size,
                                      f"{tooltip} ({'Active' if is_active else 'Inactive'})")

        if is_active:
            self._apply_button_default()

        return clicked

    def _export_timeline(self, timeline_num):
        """Export funscript from specified timeline."""
        self.app.file_manager.export_funscript_from_timeline(timeline_num)

    def _auto_connect_handy(self):
        """Auto-connect to last used device type, or fall back to configured preferred backend."""
        import asyncio
        import threading

        # Use last connected device type if available, otherwise fall back to preferred backend
        last_device_type = self.app.app_settings.get('device_control_last_connected_device_type', '')
        preferred_backend = self.app.app_settings.get('device_control_preferred_backend', 'handy')

        # Use last connected device type if set, otherwise use preferred backend
        device_type = last_device_type if last_device_type else preferred_backend
        handy_key = self.app.app_settings.get('device_control_handy_connection_key', '')

        self.app.logger.info(f"Toolbar: Auto-connecting to {device_type} (last: {last_device_type}, preferred: {preferred_backend})")

        # For Handy, we need the connection key
        if device_type == 'handy' and not handy_key:
            self.app.logger.warning("Toolbar: No Handy connection key configured. Opening Device Control settings...")
            self.app.app_state_ui.active_control_panel_tab = 4
            return

        def run_connect():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._connect_device_async(device_type, handy_key))
                finally:
                    loop.close()
            except Exception as e:
                self.app.logger.error(f"Toolbar: Failed to connect device: {e}")

        # Run in background thread
        thread = threading.Thread(target=run_connect, daemon=True)
        thread.start()

    async def _connect_device_async(self, preferred_backend: str, handy_key: str):
        """Async helper to connect to configured device type."""
        try:
            from device_control import DeviceManager, DeviceControlConfig

            # Get or create device manager
            control_panel_ui = getattr(self.app.gui_instance, 'control_panel_ui', None) if hasattr(self.app, 'gui_instance') else None

            if control_panel_ui and hasattr(control_panel_ui, 'device_manager') and control_panel_ui.device_manager:
                device_manager = control_panel_ui.device_manager
            else:
                # Create new DeviceManager
                config = DeviceControlConfig(
                    handy_connection_key=handy_key,
                    preferred_backend=preferred_backend
                )
                device_manager = DeviceManager(
                    config=config,
                    app_instance=self.app,
                    app_settings=self.app.app_settings
                )

                # Store it in control_panel_ui if available
                if control_panel_ui:
                    control_panel_ui.device_manager = device_manager

            # Connect based on backend type
            if preferred_backend == 'handy':
                success = await device_manager.connect_handy(handy_key)
                device_name = "Handy"
            elif preferred_backend == 'osr':
                # OSR uses auto-discovery
                devices = await device_manager.discover_devices_with_backend('osr')
                if devices:
                    device_id = next(iter(devices))
                    success = await device_manager.connect(device_id)
                    device_name = "OSR"
                else:
                    self.app.logger.warning("Toolbar: No OSR devices found")
                    return
            elif preferred_backend == 'buttplug':
                # Buttplug uses auto-discovery
                devices = await device_manager.discover_devices_with_backend('buttplug')
                if devices:
                    device_id = next(iter(devices))
                    success = await device_manager.connect(device_id)
                    device_name = "Buttplug device"
                else:
                    self.app.logger.warning("Toolbar: No Buttplug devices found")
                    return
            else:
                # Auto discovery across all backends
                devices = await device_manager.discover_devices()
                if devices:
                    device_id = next(iter(devices))
                    success = await device_manager.connect(device_id)
                    device_name = devices[device_id].name
                else:
                    self.app.logger.warning("Toolbar: No devices found")
                    return

            if success:
                self.app.logger.info(f"Toolbar: {device_name} connected successfully!")
                # Save last connected device type for future auto-connect
                self.app.app_settings.set('device_control_last_connected_device_type', preferred_backend)
            else:
                self.app.logger.error(f"Toolbar: Failed to connect to {device_name}")

        except Exception as e:
            self.app.logger.error(f"Toolbar: Error connecting to device: {e}")

    def _upload_script_to_device(self, device_manager):
        """Upload current funscript to connected device."""
        import asyncio
        import threading

        def run_upload():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._upload_script_async(device_manager))
                finally:
                    loop.close()
            except Exception as e:
                self.app.logger.error(f"Toolbar: Failed to upload script: {e}")

        # Run in background thread
        thread = threading.Thread(target=run_upload, daemon=True)
        thread.start()

    async def _upload_script_async(self, device_manager):
        """Async helper to upload script to device."""
        try:
            # Get funscript data from the app
            if not hasattr(self.app, 'funscript_processor') or not self.app.funscript_processor:
                self.app.logger.warning("Toolbar: No funscript processor available")
                return

            primary_actions = self.app.funscript_processor.get_actions('primary')
            if not primary_actions:
                self.app.logger.warning("Toolbar: No funscript actions to upload")
                return

            self.app.logger.info(f"Toolbar: Uploading script with {len(primary_actions)} actions...")

            # Reset streaming state to force re-upload
            device_manager.reset_handy_streaming_state()

            # Prepare device for playback (upload + setup)
            success = await device_manager.prepare_handy_for_video_playback(primary_actions)

            if success:
                self.app.logger.info("Toolbar: Script uploaded successfully!", extra={"status_message": True})
                # Sync upload revision with device control UI so stale-script detection stays consistent
                cp_ui = getattr(self.app.gui_instance, 'control_panel_ui', None) if hasattr(self.app, 'gui_instance') else None
                if cp_ui is not None:
                    if not hasattr(cp_ui, '_handy_uploaded_timelines'):
                        cp_ui._handy_uploaded_timelines = {}
                    upload_rev = getattr(self.app.funscript_processor, '_revision', 0)
                    cp_ui._handy_uploaded_timelines[1] = upload_rev
                    cp_ui._handy_last_upload_hash = upload_rev
            else:
                self.app.logger.error("Toolbar: Failed to upload script", extra={"status_message": True})

        except Exception as e:
            self.app.logger.error(f"Toolbar: Error uploading script: {e}")

    def _toggle_handy_playback(self, device_manager):
        """Toggle Handy playback between paused and playing states."""
        import asyncio
        import threading

        def run_toggle():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._toggle_handy_playback_async(device_manager))
                finally:
                    loop.close()
            except Exception as e:
                self.app.logger.error(f"Toolbar: Failed to toggle Handy playback: {e}")

        thread = threading.Thread(target=run_toggle, daemon=True)
        thread.start()

    async def _toggle_handy_playback_async(self, device_manager):
        """Async helper to toggle Handy playback."""
        try:
            # Get current video position
            video_position_ms = 0
            if self.app.processor and hasattr(self.app.processor, 'current_frame_index'):
                fps = self.app.processor.fps
                if fps > 0:
                    video_position_ms = int((self.app.processor.current_frame_index / fps) * 1000)

            # Get sync offset from settings
            sync_offset_ms = self.app.app_settings.get("device_control_handy_sync_offset_ms", 0)

            is_playing = device_manager.is_handy_playing() if hasattr(device_manager, 'is_handy_playing') else False

            if is_playing:
                self.app.logger.info("Pausing Handy...", extra={"status_message": True})
                await device_manager.pause_handy_playback()
            else:
                self.app.logger.info(f"Resuming Handy at {video_position_ms}ms...", extra={"status_message": True})
                await device_manager.start_handy_video_sync(
                    video_position_ms, allow_restart=True, sync_offset_ms=sync_offset_ms
                )

        except Exception as e:
            self.app.logger.error(f"Toolbar: Error toggling Handy playback: {e}")
