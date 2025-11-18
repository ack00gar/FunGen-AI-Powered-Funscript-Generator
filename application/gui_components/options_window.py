"""
Options Window - Comprehensive settings and configuration interface.

This module provides a unified Options window with vertical and horizontal tabs
for all application settings. Replaces the Advanced tab from control panel.

Features:
- 11 vertical tabs for main categories
- Dynamic horizontal tabs for subcategories
- Search functionality across all settings
- Auto-save to settings.json
- Feature-gated tabs (Device Control, Streamer for supporters)
"""

import imgui
import os
from typing import Optional, Dict, List, Tuple, Callable
from application.utils import get_icon_texture_manager, primary_button_style


def _tooltip_if_hovered(text):
    """Show tooltip when item is hovered."""
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


class _DisabledScope:
    """Context manager for disabled UI elements."""
    __slots__ = ("active",)

    def __init__(self, active):
        self.active = active
        if active:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.active:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()


class OptionsWindow:
    """
    Options Window - Unified settings interface.

    Provides comprehensive settings organized into:
    - General (Interface, Performance, Autosave, Energy Saver)
    - Display (Video, Gauges, Timelines, Panels)
    - AI Models (Models, Inference, TensorRT)
    - Tracking (General, ROI, Optical Flow, Sensitivity, Oscillation, Filtering)
    - Funscript Generation (General, User ROI, Refinement, Simplification, Calibration)
    - Post-Processing (Auto Processing, Profiles, Dynamic Plugin Tabs)
    - Output (General, Batch Mode, Advanced)
    - Device Control (Connection, Handy, OSR2, Buttplug, Live) [Supporters]
    - Streamer (XBVR, Sync, Advanced) [Supporters]
    - Keyboard Shortcuts (Navigation, Editing, Project, Chapters)
    - About (Info, Support, Updates)
    """

    __slots__ = (
        "app",
        "visible",
        "window_width",
        "window_height",
        "selected_vertical_tab",
        "selected_horizontal_tab",
        "search_query",
        "_vertical_tab_width",
        "_last_selected_plugin",
        # Cached references
        "_app_settings",
        "_app_state",
    )

    # Vertical tab definitions
    VERTICAL_TABS = [
        {"id": "general", "icon": "🎨", "name": "General", "count": 25},
        {"id": "display", "icon": "🖥️", "name": "Display", "count": 30},
        {"id": "ai", "icon": "🤖", "name": "AI Models", "count": 15},
        {"id": "tracking", "icon": "🎯", "name": "Tracking", "count": 50},
        {"id": "funscript", "icon": "📊", "name": "Funscript", "count": 35},
        {"id": "postproc", "icon": "🔧", "name": "Post-Processing", "count": "20+"},
        {"id": "output", "icon": "💾", "name": "Output", "count": 15},
        {"id": "device", "icon": "🕹️", "name": "Device Control", "count": 25, "supporter": True},
        {"id": "streamer", "icon": "📡", "name": "Streamer", "count": 10, "supporter": True},
        {"id": "keyboard", "icon": "⌨️", "name": "Keyboard", "count": 30},
        {"id": "about", "icon": "ℹ️", "name": "About", "count": None},
    ]

    # Horizontal tab definitions for each vertical tab
    HORIZONTAL_TABS = {
        "general": ["Interface", "Performance", "Autosave", "Energy Saver"],
        "display": ["Video", "Gauges", "Timelines", "Panels"],
        "ai": ["Models", "Inference", "TensorRT"],
        "tracking": ["General", "ROI", "Optical Flow", "Sensitivity", "Oscillation", "Class Filter"],
        "funscript": ["General", "User ROI", "Refinement", "Simplification", "Calibration"],
        "postproc": ["Auto Processing", "Profiles"],  # Will be extended with plugin tabs dynamically
        "output": ["General", "Batch Mode", "Advanced"],
        "device": ["Connection", "Handy", "OSR2", "Buttplug", "Live Tracking"],
        "streamer": ["XBVR", "Sync", "Advanced"],
        "keyboard": ["Navigation", "Editing", "Project", "Chapters"],
        "about": ["Info", "Support", "Updates"],
    }

    def __init__(self, app):
        """Initialize Options Window."""
        self.app = app
        self.visible = False
        self.window_width = 1400
        self.window_height = 800
        self.selected_vertical_tab = "general"
        self.selected_horizontal_tab = {}  # Dict[vertical_tab_id, horizontal_tab_name]
        self.search_query = ""
        self._vertical_tab_width = 180
        self._last_selected_plugin = None

        # Initialize horizontal tab selections
        for tab_id in self.HORIZONTAL_TABS:
            self.selected_horizontal_tab[tab_id] = self.HORIZONTAL_TABS[tab_id][0] if self.HORIZONTAL_TABS[tab_id] else None

        # Cached references
        self._app_settings = app.app_settings
        self._app_state = app.app_state_ui

    def show(self):
        """Show the Options window."""
        self.visible = True

    def hide(self):
        """Hide the Options window."""
        self.visible = False

    def toggle(self):
        """Toggle Options window visibility."""
        self.visible = not self.visible

    def is_visible(self):
        """Check if Options window is visible."""
        return self.visible

    def render(self):
        """Render the Options window."""
        if not self.visible:
            return

        # Set window size and position (centered)
        io = imgui.get_io()
        display_w, display_h = io.display_size
        window_x = (display_w - self.window_width) * 0.5
        window_y = (display_h - self.window_height) * 0.5

        imgui.set_next_window_position(window_x, window_y, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.window_width, self.window_height, imgui.FIRST_USE_EVER)

        # Begin Options window
        expanded, opened = imgui.begin(
            "⚙️ Options",
            closable=True,
            flags=imgui.WINDOW_NO_COLLAPSE
        )

        if not opened:
            self.visible = False

        if expanded:
            # Main layout: vertical tabs | content area
            self._render_main_layout()

            imgui.end()

    def _render_main_layout(self):
        """Render the main layout with vertical tabs and content area."""
        # Calculate available space
        avail_width, avail_height = imgui.get_content_region_available()

        # Left side: Vertical tabs
        imgui.begin_child("VerticalTabs", self._vertical_tab_width, avail_height - 60, border=True)
        self._render_vertical_tabs()
        imgui.end_child()

        imgui.same_line()

        # Right side: Content area with horizontal tabs
        imgui.begin_child("ContentArea", avail_width - self._vertical_tab_width - 10, avail_height - 60, border=True)
        self._render_content_area()
        imgui.end_child()

        # Bottom: Search and action buttons
        self._render_footer()

    def _render_vertical_tabs(self):
        """Render vertical tabs on the left side."""
        for tab in self.VERTICAL_TABS:
            # Check if tab should be visible (supporter feature gating)
            if tab.get("supporter", False) and not self._is_supporter():
                continue

            # Check if this tab is selected
            is_selected = self.selected_vertical_tab == tab["id"]

            # Apply selection styling
            if is_selected:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.4, 0.8, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.0, 0.5, 0.9, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.0, 0.3, 0.7, 1.0)

            # Render tab button
            button_label = f"{tab['icon']} {tab['name']}"
            if tab.get("supporter", False):
                button_label += " ⭐"

            if imgui.button(button_label, width=self._vertical_tab_width - 10):
                self.selected_vertical_tab = tab["id"]

            if is_selected:
                imgui.pop_style_color(3)

            # Show feature count tooltip
            if imgui.is_item_hovered() and tab["count"] is not None:
                imgui.set_tooltip(f"{tab['count']} settings")

    def _render_content_area(self):
        """Render the content area with horizontal tabs and settings."""
        # Get horizontal tabs for current vertical tab
        h_tabs = self._get_horizontal_tabs()

        if not h_tabs:
            imgui.text("No content available")
            return

        # Render horizontal tab bar
        if imgui.begin_tab_bar("HorizontalTabs"):
            for h_tab in h_tabs:
                if imgui.begin_tab_item(h_tab)[0]:
                    self.selected_horizontal_tab[self.selected_vertical_tab] = h_tab

                    # Render content for this tab
                    imgui.begin_child("TabContent", 0, 0, border=False)
                    self._render_tab_content(self.selected_vertical_tab, h_tab)
                    imgui.end_child()

                    imgui.end_tab_item()

            imgui.end_tab_bar()

    def _render_tab_content(self, vertical_tab: str, horizontal_tab: str):
        """Render content for a specific tab combination."""
        # Route to appropriate render method based on vertical tab
        render_methods = {
            "general": self._render_general_content,
            "display": self._render_display_content,
            "ai": self._render_ai_content,
            "tracking": self._render_tracking_content,
            "funscript": self._render_funscript_content,
            "postproc": self._render_postproc_content,
            "output": self._render_output_content,
            "device": self._render_device_content,
            "streamer": self._render_streamer_content,
            "keyboard": self._render_keyboard_content,
            "about": self._render_about_content,
        }

        render_method = render_methods.get(vertical_tab)
        if render_method:
            render_method(horizontal_tab)
        else:
            imgui.text(f"Content for {vertical_tab} > {horizontal_tab}")
            imgui.text("Coming soon...")

    def _render_footer(self):
        """Render the footer with search and action buttons."""
        # Search bar
        imgui.push_item_width(-200)
        _, self.search_query = imgui.input_text_with_hint(
            "##OptionsSearch",
            "🔍 Search all settings...",
            self.search_query,
            256
        )
        imgui.pop_item_width()

        if imgui.is_item_hovered():
            imgui.set_tooltip("Search settings across all categories")

        imgui.same_line()

        # Clear button
        if imgui.button("Clear"):
            self.search_query = ""

        imgui.same_line()

        # Reset to Defaults button
        if imgui.button("Reset to Defaults"):
            self._reset_to_defaults()

        imgui.same_line()

        # OK button (close window)
        if imgui.button("OK"):
            self.hide()

    def _get_horizontal_tabs(self) -> List[str]:
        """Get horizontal tabs for the current vertical tab."""
        tabs = self.HORIZONTAL_TABS.get(self.selected_vertical_tab, []).copy()

        # For post-processing, add dynamic plugin tabs
        if self.selected_vertical_tab == "postproc":
            plugin_tabs = self._get_plugin_tabs()
            tabs.extend(plugin_tabs)

        return tabs

    def _get_plugin_tabs(self) -> List[str]:
        """Get dynamic plugin tabs for post-processing."""
        # TODO: Discover plugins dynamically
        # For now, return static list
        return ["Smoother", "RDP Simplifier", "Limiter", "Half Speed", "Double Speed"]

    def _is_supporter(self) -> bool:
        """Check if user has supporter/buyer status."""
        # TODO: Implement actual supporter check
        # For now, check if feature exists in app
        return hasattr(self.app, 'is_buyer') and self.app.is_buyer

    def _reset_to_defaults(self):
        """Reset settings to defaults."""
        # TODO: Implement reset functionality
        imgui.open_popup("Reset Confirmation")

    # ============================================================
    # CONTENT RENDERING METHODS - General Tab
    # ============================================================

    def _render_general_content(self, horizontal_tab: str):
        """Render General tab content."""
        if horizontal_tab == "Interface":
            self._render_general_interface()
        elif horizontal_tab == "Performance":
            self._render_general_performance()
        elif horizontal_tab == "Autosave":
            self._render_general_autosave()
        elif horizontal_tab == "Energy Saver":
            self._render_general_energy_saver()

    def _render_general_interface(self):
        """Render General > Interface settings."""
        imgui.text_colored("General interface settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # UI Mode
        imgui.text("UI Mode:")
        imgui.same_line(200)
        current_mode = settings.get("ui_view_mode", "simple")
        mode_index = 0 if current_mode == "simple" else 1
        changed, new_index = imgui.combo("##UIMode", mode_index, ["Simple", "Expert"])
        if changed:
            new_mode = "simple" if new_index == 0 else "expert"
            settings.set("ui_view_mode", new_mode)
            self._app_state.ui_view_mode = new_mode
        _tooltip_if_hovered("Switch between Simple (beginner) and Expert (advanced) mode")

        # Layout Mode
        imgui.text("Layout Mode:")
        imgui.same_line(200)
        current_layout = settings.get("ui_layout_mode", "vertical")
        layout_options = ["vertical", "horizontal", "floating"]
        layout_index = layout_options.index(current_layout) if current_layout in layout_options else 0
        changed, new_index = imgui.combo("##LayoutMode", layout_index, ["Vertical (Stacked)", "Horizontal (Split)", "Floating Panels"])
        if changed:
            new_layout = layout_options[new_index]
            settings.set("ui_layout_mode", new_layout)
        _tooltip_if_hovered("Choose how panels are arranged in the application")

        # Font Scale
        imgui.text("Font Scale:")
        imgui.same_line(200)
        font_scale = settings.get("global_font_scale", 1.0)
        changed, new_scale = imgui.slider_float("##FontScale", font_scale, 0.5, 2.0, "%.1f")
        if changed:
            settings.set("global_font_scale", new_scale)
        _tooltip_if_hovered("Adjust the size of all UI text")

        # Auto System Scaling
        imgui.text("Auto System Scaling:")
        imgui.same_line(200)
        auto_scaling = settings.get("auto_system_scaling_enabled", True)
        changed, new_val = imgui.checkbox("##AutoScaling", auto_scaling)
        if changed:
            settings.set("auto_system_scaling_enabled", new_val)
        _tooltip_if_hovered("Automatically detect and apply system DPI scaling")

        # Show Toolbar
        imgui.text("Show Toolbar:")
        imgui.same_line(200)
        show_toolbar = settings.get("show_toolbar", True)
        changed, new_val = imgui.checkbox("##ShowToolbar", show_toolbar)
        if changed:
            settings.set("show_toolbar", new_val)
        _tooltip_if_hovered("Show or hide the main toolbar")

        # Timeline Pan Speed
        imgui.text("Timeline Pan Speed:")
        imgui.same_line(200)
        pan_speed = settings.get("timeline_pan_speed_multiplier", 20)
        changed, new_speed = imgui.slider_int("##PanSpeed", pan_speed, 1, 50)
        if changed:
            settings.set("timeline_pan_speed_multiplier", new_speed)
        _tooltip_if_hovered("Adjust how fast the timeline pans when dragging")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Info box
        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.1, 0.3, 0.5, 0.3)
        imgui.begin_child("InfoBox", 0, 60, border=True)
        imgui.text_colored("💡 Auto-Save Enabled", 0.4, 0.8, 1.0, 1.0)
        imgui.text("Changes are automatically saved to settings.json")
        imgui.text("No need to click Apply or OK")
        imgui.end_child()
        imgui.pop_style_color()

    def _render_general_performance(self):
        """Render General > Performance settings."""
        imgui.text_colored("Performance and hardware settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # GPU Rendering
        imgui.text("GPU Timeline Rendering:")
        imgui.same_line(250)
        gpu_enabled = settings.get("timeline_gpu_enabled", True)
        changed, new_val = imgui.checkbox("##GPUEnabled", gpu_enabled)
        if changed:
            settings.set("timeline_gpu_enabled", new_val)
        _tooltip_if_hovered("Use GPU for rendering timelines (better performance)")

        # GPU Threshold (only if GPU enabled)
        with _DisabledScope(not gpu_enabled):
            imgui.text("GPU Threshold (points):")
            imgui.same_line(250)
            threshold = settings.get("timeline_gpu_threshold_points", 5000)
            changed, new_val = imgui.input_int("##GPUThreshold", threshold)
            if changed:
                settings.set("timeline_gpu_threshold_points", max(100, new_val))
            _tooltip_if_hovered("Use GPU rendering when timeline has more than this many points")

        imgui.spacing()

        # Hardware Acceleration
        imgui.text("Hardware Acceleration:")
        imgui.same_line(250)
        hw_accel = settings.get("hardware_acceleration_method", "none")
        accel_options = ["none", "cuda", "opencl", "metal"]
        accel_index = accel_options.index(hw_accel) if hw_accel in accel_options else 0
        changed, new_index = imgui.combo("##HWAccel", accel_index, ["None (CPU)", "CUDA (NVIDIA)", "OpenCL", "Metal (macOS)"])
        if changed:
            settings.set("hardware_acceleration_method", accel_options[new_index])
        _tooltip_if_hovered("Hardware acceleration for AI inference")

        # Video Decoding Method
        imgui.text("Video Decoding Method:")
        imgui.same_line(250)
        decode_method = settings.get("ffmpeg_hw_decode", "auto")
        decode_options = ["auto", "cpu", "nvdec", "videotoolbox", "qsv"]
        decode_index = decode_options.index(decode_method) if decode_method in decode_options else 0
        changed, new_index = imgui.combo("##DecodeMethod", decode_index, ["Auto", "CPU", "NVDEC (NVIDIA)", "VideoToolbox (macOS)", "Quick Sync (Intel)"])
        if changed:
            settings.set("ffmpeg_hw_decode", decode_options[new_index])
        _tooltip_if_hovered("Video decoding hardware acceleration")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Stage Reruns (Expert mode settings)
        imgui.text_colored("Processing Control", 0.8, 0.8, 0.3, 1.0)
        imgui.spacing()

        imgui.text("Force Rerun Stage 1:")
        imgui.same_line(250)
        force_s1 = settings.get("force_rerun_stage1", False)
        changed, new_val = imgui.checkbox("##ForceS1", force_s1)
        if changed:
            settings.set("force_rerun_stage1", new_val)
        _tooltip_if_hovered("Force Stage 1 (detection) to run even if cache exists")

        imgui.text("Force Rerun Stage 2:")
        imgui.same_line(250)
        force_s2 = settings.get("force_rerun_stage2", False)
        changed, new_val = imgui.checkbox("##ForceS2", force_s2)
        if changed:
            settings.set("force_rerun_stage2", new_val)
        _tooltip_if_hovered("Force Stage 2 (tracking) to run even if cache exists")

    def _render_general_autosave(self):
        """Render General > Autosave settings."""
        imgui.text_colored("Project and file autosave settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Autosave Enabled
        imgui.text("Enable Autosave:")
        imgui.same_line(250)
        autosave_enabled = settings.get("autosave_enabled", False)
        changed, new_val = imgui.checkbox("##AutosaveEnabled", autosave_enabled)
        if changed:
            settings.set("autosave_enabled", new_val)
        _tooltip_if_hovered("Automatically save project at regular intervals")

        # Autosave Interval (only if enabled)
        with _DisabledScope(not autosave_enabled):
            imgui.text("Autosave Interval (seconds):")
            imgui.same_line(250)
            interval = settings.get("autosave_interval_seconds", 300)
            changed, new_val = imgui.input_int("##AutosaveInterval", interval)
            if changed:
                settings.set("autosave_interval_seconds", max(30, new_val))
            _tooltip_if_hovered("How often to autosave the project")

        imgui.spacing()

        # Autosave Final Funscript
        imgui.text("Save to Video Location:")
        imgui.same_line(250)
        save_to_video = settings.get("autosave_final_funscript_to_video_location", True)
        changed, new_val = imgui.checkbox("##SaveToVideo", save_to_video)
        if changed:
            settings.set("autosave_final_funscript_to_video_location", new_val)
        _tooltip_if_hovered("Automatically save final funscript to video file location")

    def _render_general_energy_saver(self):
        """Render General > Energy Saver settings."""
        imgui.text_colored("Energy saver and power management", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Energy Saver Enabled
        imgui.text("Enable Energy Saver:")
        imgui.same_line(250)
        energy_enabled = settings.get("energy_saver_enabled", False)
        changed, new_val = imgui.checkbox("##EnergySaverEnabled", energy_enabled)
        if changed:
            settings.set("energy_saver_enabled", new_val)
        _tooltip_if_hovered("Reduce frame rate when idle to save energy")

        # Energy Saver settings (only if enabled)
        with _DisabledScope(not energy_enabled):
            imgui.text("Normal FPS:")
            imgui.same_line(250)
            normal_fps = settings.get("energy_saver_normal_fps", 60)
            changed, new_val = imgui.input_int("##NormalFPS", normal_fps)
            if changed:
                settings.set("energy_saver_normal_fps", max(30, min(144, new_val)))
            _tooltip_if_hovered("Target frame rate during active use")

            imgui.text("Idle Threshold (seconds):")
            imgui.same_line(250)
            idle_threshold = settings.get("energy_saver_idle_threshold_seconds", 10)
            changed, new_val = imgui.input_int("##IdleThreshold", idle_threshold)
            if changed:
                settings.set("energy_saver_idle_threshold_seconds", max(5, new_val))
            _tooltip_if_hovered("Time before considering application idle")

            imgui.text("Idle FPS:")
            imgui.same_line(250)
            idle_fps = settings.get("energy_saver_idle_fps", 15)
            changed, new_val = imgui.input_int("##IdleFPS", idle_fps)
            if changed:
                settings.set("energy_saver_idle_fps", max(5, min(60, new_val)))
            _tooltip_if_hovered("Target frame rate when idle")

    # ============================================================
    # CONTENT RENDERING METHODS - Display Tab
    # ============================================================

    def _render_display_content(self, horizontal_tab: str):
        """Render Display tab content."""
        if horizontal_tab == "Video":
            self._render_display_video()
        elif horizontal_tab == "Gauges":
            self._render_display_gauges()
        elif horizontal_tab == "Timelines":
            self._render_display_timelines()
        elif horizontal_tab == "Panels":
            self._render_display_panels()

    def _render_display_video(self):
        """Render Display > Video settings."""
        imgui.text_colored("Video display and overlay settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Show Video Feed
        imgui.text("Show Video Feed:")
        imgui.same_line(250)
        show_video = settings.get("show_video_feed", True)
        changed, new_val = imgui.checkbox("##ShowVideo", show_video)
        if changed:
            settings.set("show_video_feed", new_val)
        _tooltip_if_hovered("Show or hide the video display area")

        # Show Stage 2 Overlay
        imgui.text("Show Stage 2 Overlay:")
        imgui.same_line(250)
        show_overlay = settings.get("show_stage2_overlay", True)
        changed, new_val = imgui.checkbox("##ShowStage2Overlay", show_overlay)
        if changed:
            settings.set("show_stage2_overlay", new_val)
        _tooltip_if_hovered("Show tracking overlay on video during processing")

        # Use Simplified Preview
        imgui.text("Simplified Preview:")
        imgui.same_line(250)
        simplified = settings.get("use_simplified_funscript_preview", False)
        changed, new_val = imgui.checkbox("##SimplifiedPreview", simplified)
        if changed:
            settings.set("use_simplified_funscript_preview", new_val)
        _tooltip_if_hovered("Use simplified rendering for funscript preview")

    def _render_display_gauges(self):
        """Render Display > Gauges settings."""
        imgui.text_colored("Gauge windows and simulator settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings
        app_state = self._app_state

        # Timeline 1 Gauge
        imgui.text("Timeline 1 Gauge:")
        imgui.same_line(250)
        show_t1 = settings.get("show_gauge_window_timeline1", False)
        changed, new_val = imgui.checkbox("##ShowGaugeT1", show_t1)
        if changed:
            settings.set("show_gauge_window_timeline1", new_val)
            app_state.show_gauge_window_timeline1 = new_val
        _tooltip_if_hovered("Show gauge window for Timeline 1")

        # Timeline 2 Gauge
        imgui.text("Timeline 2 Gauge:")
        imgui.same_line(250)
        show_t2 = settings.get("show_gauge_window_timeline2", False)
        changed, new_val = imgui.checkbox("##ShowGaugeT2", show_t2)
        if changed:
            settings.set("show_gauge_window_timeline2", new_val)
            app_state.show_gauge_window_timeline2 = new_val
        _tooltip_if_hovered("Show gauge window for Timeline 2")

        # Movement Bar / LR Dial
        imgui.text("Movement Bar (LR Dial):")
        imgui.same_line(250)
        show_lr = settings.get("show_lr_dial_graph", False)
        changed, new_val = imgui.checkbox("##ShowLRDial", show_lr)
        if changed:
            settings.set("show_lr_dial_graph", new_val)
            app_state.show_lr_dial_graph = new_val
        _tooltip_if_hovered("Show movement bar with roll angle visualization")

        # LR Dial Window Size
        imgui.spacing()
        imgui.text_colored("Movement Bar Window Size", 0.8, 0.8, 0.3, 1.0)
        imgui.text("Width:")
        imgui.same_line(250)
        lr_width = settings.get("lr_dial_window_size_w", 180)
        changed, new_val = imgui.input_int("##LRDialWidth", lr_width)
        if changed:
            settings.set("lr_dial_window_size_w", max(100, new_val))

        imgui.text("Height:")
        imgui.same_line(250)
        lr_height = settings.get("lr_dial_window_size_h", 220)
        changed, new_val = imgui.input_int("##LRDialHeight", lr_height)
        if changed:
            settings.set("lr_dial_window_size_h", max(100, new_val))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # 3D Simulator
        imgui.text("3D Simulator:")
        imgui.same_line(250)
        show_3d = settings.get("show_simulator_3d", True)
        changed, new_val = imgui.checkbox("##Show3DSim", show_3d)
        if changed:
            settings.set("show_simulator_3d", new_val)
            app_state.show_simulator_3d = new_val
        _tooltip_if_hovered("Show 3D funscript simulator")

        # 3D Simulator Logo
        imgui.text("Show 3D Simulator Logo:")
        imgui.same_line(250)
        show_logo = settings.get("show_3d_simulator_logo", True)
        changed, new_val = imgui.checkbox("##Show3DLogo", show_logo)
        if changed:
            settings.set("show_3d_simulator_logo", new_val)
        _tooltip_if_hovered("Display logo texture on 3D simulator cylinder")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Overlay Modes
        imgui.text_colored("Overlay Modes (Render on Video)", 0.8, 0.8, 0.3, 1.0)

        imgui.text("Gauge Overlay Mode:")
        imgui.same_line(250)
        gauge_overlay = settings.get("gauge_overlay_mode", False)
        changed, new_val = imgui.checkbox("##GaugeOverlay", gauge_overlay)
        if changed:
            settings.set("gauge_overlay_mode", new_val)
        _tooltip_if_hovered("Render gauges as overlay on video display")

        imgui.text("Movement Bar Overlay:")
        imgui.same_line(250)
        lr_overlay = settings.get("movement_bar_overlay_mode", False)
        changed, new_val = imgui.checkbox("##LROverlay", lr_overlay)
        if changed:
            settings.set("movement_bar_overlay_mode", new_val)
        _tooltip_if_hovered("Render movement bar as overlay on video display")

        imgui.text("3D Simulator Overlay:")
        imgui.same_line(250)
        sim_overlay = settings.get("simulator_3d_overlay_mode", True)
        changed, new_val = imgui.checkbox("##SimOverlay", sim_overlay)
        if changed:
            settings.set("simulator_3d_overlay_mode", new_val)
        _tooltip_if_hovered("Render 3D simulator as overlay on video display")

    def _render_display_timelines(self):
        """Render Display > Timelines settings."""
        imgui.text_colored("Timeline and heatmap display settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings
        app_state = self._app_state

        # Show Funscript Timeline
        imgui.text("Show Funscript Timeline:")
        imgui.same_line(250)
        show_timeline = settings.get("show_funscript_timeline", True)
        changed, new_val = imgui.checkbox("##ShowTimeline", show_timeline)
        if changed:
            settings.set("show_funscript_timeline", new_val)
            app_state.show_funscript_timeline = new_val
        _tooltip_if_hovered("Show the main funscript timeline")

        # Show Interactive Timeline 1
        imgui.text("Interactive Timeline 1:")
        imgui.same_line(250)
        show_int1 = settings.get("show_funscript_interactive_timeline", True)
        changed, new_val = imgui.checkbox("##ShowIntTimeline1", show_int1)
        if changed:
            settings.set("show_funscript_interactive_timeline", new_val)
            app_state.show_funscript_interactive_timeline = new_val
        _tooltip_if_hovered("Show interactive editing timeline for Timeline 1")

        # Show Interactive Timeline 2
        imgui.text("Interactive Timeline 2:")
        imgui.same_line(250)
        show_int2 = settings.get("show_funscript_interactive_timeline2", False)
        changed, new_val = imgui.checkbox("##ShowIntTimeline2", show_int2)
        if changed:
            settings.set("show_funscript_interactive_timeline2", new_val)
            app_state.show_funscript_interactive_timeline2 = new_val
        _tooltip_if_hovered("Show interactive editing timeline for Timeline 2")

        # Show Heatmap
        imgui.text("Show Heatmap:")
        imgui.same_line(250)
        show_heatmap = settings.get("show_heatmap", True)
        changed, new_val = imgui.checkbox("##ShowHeatmap", show_heatmap)
        if changed:
            settings.set("show_heatmap", new_val)
            app_state.show_heatmap = new_val
        _tooltip_if_hovered("Show heatmap visualization of funscript intensity")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Show Timeline Editor Buttons
        imgui.text("Timeline Editor Buttons:")
        imgui.same_line(250)
        show_buttons = settings.get("show_timeline_editor_buttons", False)
        changed, new_val = imgui.checkbox("##ShowEditorButtons", show_buttons)
        if changed:
            settings.set("show_timeline_editor_buttons", new_val)
        _tooltip_if_hovered("Show editing buttons on timeline")

        # Show Performance Indicators
        imgui.text("Performance Indicators:")
        imgui.same_line(250)
        show_perf = settings.get("show_timeline_optimization_indicator", False)
        changed, new_val = imgui.checkbox("##ShowPerfIndicators", show_perf)
        if changed:
            settings.set("show_timeline_optimization_indicator", new_val)
        _tooltip_if_hovered("Show performance optimization indicators")

        # Timeline Performance Logging
        imgui.text("Performance Logging:")
        imgui.same_line(250)
        perf_log = settings.get("timeline_performance_logging", False)
        changed, new_val = imgui.checkbox("##TimelinePerfLog", perf_log)
        if changed:
            settings.set("timeline_performance_logging", new_val)
        _tooltip_if_hovered("Log timeline rendering performance statistics")

    def _render_display_panels(self):
        """Render Display > Panels settings."""
        imgui.text_colored("Panel and window visibility settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings
        app_state = self._app_state

        # Chapter List Window
        imgui.text("Chapter List Window:")
        imgui.same_line(250)
        show_chapters = settings.get("show_chapter_list_window", False)
        changed, new_val = imgui.checkbox("##ShowChapterList", show_chapters)
        if changed:
            settings.set("show_chapter_list_window", new_val)
            app_state.show_chapter_list_window = new_val
        _tooltip_if_hovered("Show chapter list management window")

        # Show Advanced Options
        imgui.text("Show Advanced Options:")
        imgui.same_line(250)
        show_advanced = settings.get("show_advanced_options", False)
        changed, new_val = imgui.checkbox("##ShowAdvanced", show_advanced)
        if changed:
            settings.set("show_advanced_options", new_val)
        _tooltip_if_hovered("Show advanced options in UI")

        # Full Width Navigation
        imgui.text("Full Width Navigation:")
        imgui.same_line(250)
        full_width = settings.get("full_width_nav", True)
        changed, new_val = imgui.checkbox("##FullWidthNav", full_width)
        if changed:
            settings.set("full_width_nav", new_val)
        _tooltip_if_hovered("Use full width for navigation elements")

    # ============================================================
    # CONTENT RENDERING METHODS - AI Models Tab
    # ============================================================

    def _render_ai_content(self, horizontal_tab: str):
        """Render AI Models tab content."""
        if horizontal_tab == "Models":
            self._render_ai_models()
        elif horizontal_tab == "Inference":
            self._render_ai_inference()
        elif horizontal_tab == "TensorRT":
            self._render_ai_tensorrt()

    def _render_ai_models(self):
        """Render AI Models > Models settings."""
        imgui.text_colored("AI model paths and configuration", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # YOLO Detection Model Path
        imgui.text("Detection Model Path:")
        yolo_det_path = settings.get("yolo_det_model_path", "")
        imgui.push_item_width(-80)
        changed, new_val = imgui.input_text("##YOLODetPath", yolo_det_path, 512)
        imgui.pop_item_width()
        if changed:
            settings.set("yolo_det_model_path", new_val)
        imgui.same_line()
        if imgui.button("Browse##DetPath"):
            # TODO: Open file dialog
            pass
        _tooltip_if_hovered("Path to YOLO detection model (.pt file)")

        imgui.spacing()

        # YOLO Pose Model Path
        imgui.text("Pose Model Path:")
        yolo_pose_path = settings.get("yolo_pose_model_path", "")
        imgui.push_item_width(-80)
        changed, new_val = imgui.input_text("##YOLOPosePath", yolo_pose_path, 512)
        imgui.pop_item_width()
        if changed:
            settings.set("yolo_pose_model_path", new_val)
        imgui.same_line()
        if imgui.button("Browse##PosePath"):
            # TODO: Open file dialog
            pass
        _tooltip_if_hovered("Path to YOLO pose estimation model (.pt file)")

        imgui.spacing()

        # Pose Model Artifacts Directory
        imgui.text("Artifacts Directory:")
        artifacts_dir = settings.get("pose_model_artifacts_dir", "")
        imgui.push_item_width(-80)
        changed, new_val = imgui.input_text("##ArtifactsDir", artifacts_dir, 512)
        imgui.pop_item_width()
        if changed:
            settings.set("pose_model_artifacts_dir", new_val)
        imgui.same_line()
        if imgui.button("Browse##ArtifactsDir"):
            # TODO: Open directory dialog
            pass
        _tooltip_if_hovered("Directory for pose model artifacts and cache")

    def _render_ai_inference(self):
        """Render AI Models > Inference settings."""
        imgui.text_colored("Inference worker configuration", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        imgui.text_colored("Stage 1 Workers (Detection)", 0.8, 0.8, 0.3, 1.0)
        imgui.spacing()

        # Stage 1 Producers
        imgui.text("Producers (threads):")
        imgui.same_line(250)
        s1_producers = settings.get("num_producers_stage1", 2)
        changed, new_val = imgui.input_int("##S1Producers", s1_producers)
        if changed:
            settings.set("num_producers_stage1", max(1, min(16, new_val)))
        _tooltip_if_hovered("Number of producer threads for Stage 1 detection")

        # Stage 1 Consumers
        imgui.text("Consumers (threads):")
        imgui.same_line(250)
        s1_consumers = settings.get("num_consumers_stage1", 2)
        changed, new_val = imgui.input_int("##S1Consumers", s1_consumers)
        if changed:
            settings.set("num_consumers_stage1", max(1, min(16, new_val)))
        _tooltip_if_hovered("Number of consumer threads for Stage 1 detection")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored("Stage 2 Workers (Tracking)", 0.8, 0.8, 0.3, 1.0)
        imgui.spacing()

        # Stage 2 Workers
        imgui.text("Workers (threads):")
        imgui.same_line(250)
        s2_workers = settings.get("num_workers_stage2_of", 4)
        changed, new_val = imgui.input_int("##S2Workers", s2_workers)
        if changed:
            settings.set("num_workers_stage2_of", max(1, min(16, new_val)))
        _tooltip_if_hovered("Number of worker threads for Stage 2 optical flow tracking")

    def _render_ai_tensorrt(self):
        """Render AI Models > TensorRT settings."""
        imgui.text_colored("TensorRT optimization (optional)", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text("TensorRT compilation is an optional feature that can")
        imgui.text("significantly improve inference performance on NVIDIA GPUs.")
        imgui.spacing()

        imgui.text_colored("Note:", 1.0, 0.6, 0.0, 1.0)
        imgui.text("TensorRT settings are managed through the")
        imgui.text("Tools > TensorRT Compiler menu.")

    def _render_tracking_content(self, horizontal_tab: str):
        """Render Tracking tab content."""
        imgui.text(f"Tracking > {horizontal_tab}")
        imgui.text("Coming soon...")

    def _render_funscript_content(self, horizontal_tab: str):
        """Render Funscript Generation tab content."""
        imgui.text(f"Funscript Generation > {horizontal_tab}")
        imgui.text("Coming soon...")

    def _render_postproc_content(self, horizontal_tab: str):
        """Render Post-Processing tab content."""
        imgui.text(f"Post-Processing > {horizontal_tab}")
        imgui.text("Coming soon...")
        if horizontal_tab not in ["Auto Processing", "Profiles"]:
            imgui.spacing()
            imgui.text(f"Plugin: {horizontal_tab}")
            imgui.bullet_text("Plugin parameters will be loaded dynamically")

    # ============================================================
    # CONTENT RENDERING METHODS - Output Tab
    # ============================================================

    def _render_output_content(self, horizontal_tab: str):
        """Render Output tab content."""
        if horizontal_tab == "General":
            self._render_output_general()
        elif horizontal_tab == "Batch Mode":
            self._render_output_batch()
        elif horizontal_tab == "Advanced":
            self._render_output_advanced()

    def _render_output_general(self):
        """Render Output > General settings."""
        imgui.text_colored("Output folder and file generation settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Output Folder Path
        imgui.text("Output Folder:")
        output_folder = settings.get("output_folder_path", "")
        imgui.push_item_width(-80)
        changed, new_val = imgui.input_text("##OutputFolder", output_folder, 512)
        imgui.pop_item_width()
        if changed:
            settings.set("output_folder_path", new_val)
        imgui.same_line()
        if imgui.button("Browse##OutputFolder"):
            # TODO: Open directory dialog
            pass
        _tooltip_if_hovered("Default folder for generated funscript files")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Generate .funscript file (always true, but kept for consistency)
        imgui.text("Generate .funscript:")
        imgui.same_line(250)
        generate_funscript = True  # Always generate funscript
        imgui.text("✓ Always enabled")
        _tooltip_if_hovered("Funscript files are always generated")

        # Generate .roll file
        imgui.text("Generate .roll file:")
        imgui.same_line(250)
        generate_roll = settings.get("generate_roll_file", True)
        changed, new_val = imgui.checkbox("##GenerateRoll", generate_roll)
        if changed:
            settings.set("generate_roll_file", new_val)
        _tooltip_if_hovered("Generate .roll file for rotational movements")

    def _render_output_batch(self):
        """Render Output > Batch Mode settings."""
        imgui.text_colored("Batch processing output settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Batch Mode Overwrite Strategy
        imgui.text("Overwrite Strategy:")
        imgui.same_line(250)
        strategy = settings.get("batch_mode_overwrite_strategy", 0)
        strategies = ["Process All", "Skip Existing"]
        changed, new_index = imgui.combo("##BatchStrategy", strategy, strategies)
        if changed:
            settings.set("batch_mode_overwrite_strategy", new_index)
        _tooltip_if_hovered("How to handle existing output files in batch mode:\n"
                           "Process All: Overwrite existing files\n"
                           "Skip Existing: Skip files that already have output")

        imgui.spacing()

        # Autosave Final Funscript to Video Location
        imgui.text("Save to Video Location:")
        imgui.same_line(250)
        autosave_to_video = settings.get("autosave_final_funscript_to_video_location", True)
        changed, new_val = imgui.checkbox("##AutosaveToVideo", autosave_to_video)
        if changed:
            settings.set("autosave_final_funscript_to_video_location", new_val)
        _tooltip_if_hovered("Automatically save funscript to video file location")

    def _render_output_advanced(self):
        """Render Output > Advanced settings."""
        imgui.text_colored("Advanced output and caching settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Retain Stage 2 Database
        imgui.text("Keep Stage 2 Database:")
        imgui.same_line(250)
        retain_s2 = settings.get("retain_stage2_database", False)
        changed, new_val = imgui.checkbox("##RetainS2DB", retain_s2)
        if changed:
            settings.set("retain_stage2_database", new_val)
        _tooltip_if_hovered("Keep Stage 2 tracking database for future use")

        # Save/Reuse Preprocessed Video (mentioned in Performance but logically fits here)
        imgui.text("Save Preprocessed Video:")
        imgui.same_line(250)
        save_preprocessed = settings.get("save_preprocessed_video", False)
        changed, new_val = imgui.checkbox("##SavePreprocessed", save_preprocessed)
        if changed:
            settings.set("save_preprocessed_video", new_val)
        _tooltip_if_hovered("Save preprocessed video for reuse in future runs")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # FFmpeg Path
        imgui.text("FFmpeg Path:")
        ffmpeg_path = settings.get("ffmpeg_path", "ffmpeg")
        imgui.push_item_width(-80)
        changed, new_val = imgui.input_text("##FFmpegPath", ffmpeg_path, 512)
        imgui.pop_item_width()
        if changed:
            settings.set("ffmpeg_path", new_val)
        imgui.same_line()
        if imgui.button("Browse##FFmpegPath"):
            # TODO: Open file dialog
            pass
        _tooltip_if_hovered("Path to ffmpeg executable (leave as 'ffmpeg' to use system PATH)")

    def _render_device_content(self, horizontal_tab: str):
        """Render Device Control tab content (supporters only)."""
        imgui.text(f"Device Control > {horizontal_tab}")
        imgui.text_colored("⭐ Supporter Feature", 1.0, 0.6, 0.0, 1.0)
        imgui.text("Coming soon...")

    def _render_streamer_content(self, horizontal_tab: str):
        """Render Streamer tab content (supporters only)."""
        imgui.text(f"Streamer > {horizontal_tab}")
        imgui.text_colored("⭐ Supporter Feature", 1.0, 0.6, 0.0, 1.0)
        imgui.text("Coming soon...")

    def _render_keyboard_content(self, horizontal_tab: str):
        """Render Keyboard Shortcuts tab content."""
        imgui.text(f"Keyboard Shortcuts > {horizontal_tab}")
        imgui.text("Coming soon...")

    def _render_about_content(self, horizontal_tab: str):
        """Render About tab content."""
        if horizontal_tab == "Info":
            imgui.text_colored("FunGen - AI-Powered Funscript Generator", 0.4, 0.8, 1.0, 1.0)
            imgui.spacing()
            imgui.text("Version: 0.1.0")  # TODO: Get from app
            imgui.text("Build Date: 2025-01-18")  # TODO: Get from app
            imgui.spacing()
            if imgui.button("Visit GitHub"):
                # TODO: Open browser to GitHub
                pass
        elif horizontal_tab == "Support":
            imgui.text_colored("Support Development", 0.4, 0.8, 1.0, 1.0)
            imgui.spacing()
            imgui.text("If you find FunGen useful, consider supporting development:")
            imgui.spacing()
            if imgui.button("☕ Support on Ko-fi"):
                # TODO: Open browser to Ko-fi
                pass
        elif horizontal_tab == "Updates":
            imgui.text_colored("Check for Updates", 0.4, 0.8, 1.0, 1.0)
            imgui.spacing()
            if imgui.button("Check for Updates"):
                # TODO: Check for updates
                pass
