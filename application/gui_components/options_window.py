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

    # ============================================================
    # CONTENT RENDERING METHODS - Tracking Tab
    # ============================================================

    def _render_tracking_content(self, horizontal_tab: str):
        """Render Tracking tab content."""
        if horizontal_tab == "General":
            self._render_tracking_general()
        elif horizontal_tab == "ROI":
            self._render_tracking_roi()
        elif horizontal_tab == "Optical Flow":
            self._render_tracking_flow()
        elif horizontal_tab == "Sensitivity":
            self._render_tracking_sensitivity()
        elif horizontal_tab == "Oscillation":
            self._render_tracking_oscillation()
        elif horizontal_tab == "Class Filter":
            self._render_tracking_class_filter()

    def _render_tracking_general(self):
        """Render Tracking > General settings."""
        imgui.text_colored("General tracking settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Confidence Threshold
        imgui.text("Confidence Threshold:")
        imgui.same_line(250)
        conf_threshold = settings.get("live_tracker_confidence_threshold", 0.45)
        changed, new_val = imgui.slider_float("##ConfThreshold", conf_threshold, 0.1, 1.0, "%.2f")
        if changed:
            settings.set("live_tracker_confidence_threshold", new_val)
        _tooltip_if_hovered("Minimum confidence for detection (0.1-1.0)")

    def _render_tracking_roi(self):
        """Render Tracking > ROI settings."""
        imgui.text_colored("Region of Interest (ROI) settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # ROI Padding
        imgui.text("ROI Padding (pixels):")
        imgui.same_line(250)
        roi_padding = settings.get("live_tracker_roi_padding", 50)
        changed, new_val = imgui.input_int("##ROIPadding", roi_padding)
        if changed:
            settings.set("live_tracker_roi_padding", max(0, new_val))
        _tooltip_if_hovered("Padding around detected region")

        # ROI Update Interval
        imgui.text("Update Interval (frames):")
        imgui.same_line(250)
        update_interval = settings.get("live_tracker_roi_update_interval", 5)
        changed, new_val = imgui.input_int("##UpdateInterval", update_interval)
        if changed:
            settings.set("live_tracker_roi_update_interval", max(1, new_val))
        _tooltip_if_hovered("How often to update ROI")

        # ROI Smoothing Factor
        imgui.text("Smoothing Factor:")
        imgui.same_line(250)
        smoothing = settings.get("live_tracker_roi_smoothing_factor", 0.2)
        changed, new_val = imgui.slider_float("##ROISmoothing", smoothing, 0.0, 1.0, "%.2f")
        if changed:
            settings.set("live_tracker_roi_smoothing_factor", new_val)
        _tooltip_if_hovered("ROI movement smoothing (0=no smoothing, 1=max smoothing)")

        # ROI Persistence Frames
        imgui.text("Persistence (frames):")
        imgui.same_line(250)
        persistence = settings.get("live_tracker_roi_persistence_frames", 30)
        changed, new_val = imgui.input_int("##ROIPersistence", persistence)
        if changed:
            settings.set("live_tracker_roi_persistence_frames", max(0, new_val))
        _tooltip_if_hovered("How long to keep ROI without detection")

    def _render_tracking_flow(self):
        """Render Tracking > Optical Flow settings."""
        imgui.text_colored("Optical flow tracking settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Use Sparse Flow
        imgui.text("Use Sparse Flow:")
        imgui.same_line(250)
        use_sparse = settings.get("live_tracker_use_sparse_flow", False)
        changed, new_val = imgui.checkbox("##UseSparse", use_sparse)
        if changed:
            settings.set("live_tracker_use_sparse_flow", new_val)
        _tooltip_if_hovered("Use sparse optical flow (faster)")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored("DIS Flow Settings", 0.8, 0.8, 0.3, 1.0)
        imgui.spacing()

        # DIS Flow Preset
        imgui.text("DIS Flow Preset:")
        imgui.same_line(250)
        preset = settings.get("live_tracker_dis_flow_preset", "medium")
        presets = ["ultrafast", "fast", "medium", "fine"]
        preset_index = presets.index(preset) if preset in presets else 2
        changed, new_index = imgui.combo("##DISPreset", preset_index, ["Ultra Fast", "Fast", "Medium", "Fine"])
        if changed:
            settings.set("live_tracker_dis_flow_preset", presets[new_index])
        _tooltip_if_hovered("Quality preset for DIS optical flow")

        # DIS Finest Scale
        imgui.text("Finest Scale:")
        imgui.same_line(250)
        finest_scale = settings.get("live_tracker_dis_finest_scale", 1)
        changed, new_val = imgui.input_int("##FinestScale", finest_scale)
        if changed:
            settings.set("live_tracker_dis_finest_scale", max(0, min(5, new_val)))
        _tooltip_if_hovered("Finest scale for DIS (0-5)")

    def _render_tracking_sensitivity(self):
        """Render Tracking > Sensitivity settings."""
        imgui.text_colored("Output sensitivity and amplification", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Output Sensitivity
        imgui.text("Output Sensitivity:")
        imgui.same_line(250)
        sensitivity = settings.get("live_tracker_output_sensitivity", 1.0)
        changed, new_val = imgui.slider_float("##OutputSensitivity", sensitivity, 0.1, 5.0, "%.2f")
        if changed:
            settings.set("live_tracker_output_sensitivity", new_val)
        _tooltip_if_hovered("Global output sensitivity multiplier")

        # Signal Amplification
        imgui.text("Signal Amplification:")
        imgui.same_line(250)
        amplification = settings.get("live_tracker_signal_amplification", 1.0)
        changed, new_val = imgui.slider_float("##SignalAmp", amplification, 0.1, 5.0, "%.2f")
        if changed:
            settings.set("live_tracker_signal_amplification", new_val)
        _tooltip_if_hovered("Amplify tracking signal")

        # Output Delay
        imgui.text("Output Delay (frames):")
        imgui.same_line(250)
        delay = settings.get("funscript_output_delay_frames", 0)
        changed, new_val = imgui.input_int("##OutputDelay", delay)
        if changed:
            settings.set("funscript_output_delay_frames", new_val)
        _tooltip_if_hovered("Frame delay for funscript output")

    def _render_tracking_oscillation(self):
        """Render Tracking > Oscillation settings."""
        imgui.text_colored("Oscillation detector settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Grid Size
        imgui.text("Grid Size:")
        imgui.same_line(250)
        grid_size = settings.get("oscillation_detector_grid_size", 8)
        changed, new_val = imgui.slider_int("##GridSize", grid_size, 4, 32)
        if changed:
            settings.set("oscillation_detector_grid_size", new_val)
        _tooltip_if_hovered("Grid size for oscillation detection")

        # Detection Sensitivity
        imgui.text("Detection Sensitivity:")
        imgui.same_line(250)
        osc_sens = settings.get("oscillation_detector_sensitivity", 0.5)
        changed, new_val = imgui.slider_float("##OscSensitivity", osc_sens, 0.1, 1.0, "%.2f")
        if changed:
            settings.set("oscillation_detector_sensitivity", new_val)
        _tooltip_if_hovered("Sensitivity for oscillation detection")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Enable Decay
        imgui.text("Enable Decay:")
        imgui.same_line(250)
        enable_decay = settings.get("oscillation_enable_decay", True)
        changed, new_val = imgui.checkbox("##EnableDecay", enable_decay)
        if changed:
            settings.set("oscillation_enable_decay", new_val)
        _tooltip_if_hovered("Enable signal decay mechanism")

        # Hold Duration (only if decay enabled)
        with _DisabledScope(not enable_decay):
            imgui.text("Hold Duration (ms):")
            imgui.same_line(250)
            hold_duration = settings.get("oscillation_hold_duration_ms", 100)
            changed, new_val = imgui.input_int("##HoldDuration", hold_duration)
            if changed:
                settings.set("oscillation_hold_duration_ms", max(0, new_val))
            _tooltip_if_hovered("How long to hold signal before decay")

            imgui.text("Decay Factor:")
            imgui.same_line(250)
            decay_factor = settings.get("oscillation_decay_factor", 0.95)
            changed, new_val = imgui.slider_float("##DecayFactor", decay_factor, 0.5, 1.0, "%.2f")
            if changed:
                settings.set("oscillation_decay_factor", new_val)
            _tooltip_if_hovered("Signal decay rate")

    def _render_tracking_class_filter(self):
        """Render Tracking > Class Filter settings."""
        imgui.text_colored("Detection class filtering", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text("Filter which detection classes to track:")
        imgui.spacing()

        # This would typically be dynamic based on available classes
        # For now, show common classes
        imgui.text("Common detection classes:")
        imgui.bullet_text("Person")
        imgui.bullet_text("Hand")
        imgui.bullet_text("Toy/Object")
        imgui.spacing()
        imgui.text_colored("Note:", 1.0, 0.6, 0.0, 1.0)
        imgui.text("Class filtering is configured in the Advanced tab")
        imgui.text("in the main control panel during tracking setup.")

    # ============================================================
    # CONTENT RENDERING METHODS - Funscript Generation Tab
    # ============================================================

    def _render_funscript_content(self, horizontal_tab: str):
        """Render Funscript Generation tab content."""
        if horizontal_tab == "General":
            self._render_funscript_general()
        elif horizontal_tab == "User ROI":
            self._render_funscript_user_roi()
        elif horizontal_tab == "Refinement":
            self._render_funscript_refinement()
        elif horizontal_tab == "Simplification":
            self._render_funscript_simplification()
        elif horizontal_tab == "Calibration":
            self._render_funscript_calibration()

    def _render_funscript_general(self):
        """Render Funscript > General settings."""
        imgui.text_colored("General funscript generation settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Tracking Axis Mode
        imgui.text("Tracking Axis Mode:")
        imgui.same_line(250)
        axis_mode = settings.get("tracking_axis_mode", "auto")
        modes = ["auto", "vertical", "horizontal", "both"]
        mode_index = modes.index(axis_mode) if axis_mode in modes else 0
        changed, new_index = imgui.combo("##AxisMode", mode_index, ["Auto", "Vertical (Y)", "Horizontal (X)", "Both"])
        if changed:
            settings.set("tracking_axis_mode", modes[new_index])
        _tooltip_if_hovered("Which axis to track for funscript generation")

        # Range Processing
        imgui.text("Range Processing:")
        imgui.same_line(250)
        range_processing = settings.get("enable_range_processing", False)
        changed, new_val = imgui.checkbox("##RangeProcessing", range_processing)
        if changed:
            settings.set("enable_range_processing", new_val)
        _tooltip_if_hovered("Process only a specific frame range")

    def _render_funscript_user_roi(self):
        """Render Funscript > User ROI settings."""
        imgui.text_colored("User-defined ROI settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text("User ROI allows manual selection of tracking region.")
        imgui.spacing()
        imgui.text("Use the video display to:")
        imgui.bullet_text("Select ROI area with mouse")
        imgui.bullet_text("Adjust amplification and scale")
        imgui.bullet_text("Configure output range")

    def _render_funscript_refinement(self):
        """Render Funscript > Refinement settings."""
        imgui.text_colored("Interactive refinement settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Scale
        imgui.text("Scale:")
        imgui.same_line(250)
        scale = settings.get("refinement_scale", 1.0)
        changed, new_val = imgui.slider_float("##RefScale", scale, 0.1, 5.0, "%.2f")
        if changed:
            settings.set("refinement_scale", new_val)
        _tooltip_if_hovered("Scale funscript values")

        # Center
        imgui.text("Center:")
        imgui.same_line(250)
        center = settings.get("refinement_center", 50.0)
        changed, new_val = imgui.slider_float("##RefCenter", center, 0.0, 100.0, "%.1f")
        if changed:
            settings.set("refinement_center", new_val)
        _tooltip_if_hovered("Center point for funscript")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Smoothing
        imgui.text_colored("Smoothing (Savitzky-Golay)", 0.8, 0.8, 0.3, 1.0)

        imgui.text("Window Size:")
        imgui.same_line(250)
        window_size = settings.get("smoothing_window_size", 5)
        changed, new_val = imgui.slider_int("##SmoothWindow", window_size, 3, 21)
        # Ensure odd number
        if changed:
            if new_val % 2 == 0:
                new_val += 1
            settings.set("smoothing_window_size", new_val)
        _tooltip_if_hovered("Smoothing window size (must be odd)")

    def _render_funscript_simplification(self):
        """Render Funscript > Simplification settings."""
        imgui.text_colored("RDP simplification settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Enable Simplification
        imgui.text("Enable Simplification:")
        imgui.same_line(250)
        enable_simp = settings.get("enable_simplification", True)
        changed, new_val = imgui.checkbox("##EnableSimp", enable_simp)
        if changed:
            settings.set("enable_simplification", new_val)
        _tooltip_if_hovered("Apply RDP simplification to funscript")

        # RDP Epsilon
        with _DisabledScope(not enable_simp):
            imgui.text("RDP Epsilon:")
            imgui.same_line(250)
            epsilon = settings.get("rdp_epsilon", 1.0)
            changed, new_val = imgui.slider_float("##RDPEpsilon", epsilon, 0.1, 10.0, "%.1f")
            if changed:
                settings.set("rdp_epsilon", new_val)
            _tooltip_if_hovered("RDP simplification threshold (higher = more simplification)")

    def _render_funscript_calibration(self):
        """Render Funscript > Calibration settings."""
        imgui.text_colored("Latency calibration settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Latency Offset
        imgui.text("Latency Offset (ms):")
        imgui.same_line(250)
        latency = settings.get("latency_offset_ms", 0)
        changed, new_val = imgui.input_int("##LatencyOffset", latency)
        if changed:
            settings.set("latency_offset_ms", new_val)
        _tooltip_if_hovered("Latency compensation in milliseconds")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text("Use Tools > Calibration & Analysis menu for:")
        imgui.bullet_text("Latency calibration wizard")
        imgui.bullet_text("Timeline comparison")
        imgui.bullet_text("Auto-calibration")

    # ============================================================
    # CONTENT RENDERING METHODS - Post-Processing Tab
    # ============================================================

    def _render_postproc_content(self, horizontal_tab: str):
        """Render Post-Processing tab content."""
        if horizontal_tab == "Auto Processing":
            self._render_postproc_auto()
        elif horizontal_tab == "Profiles":
            self._render_postproc_profiles()
        else:
            # Dynamic plugin tabs
            self._render_postproc_plugin(horizontal_tab)

    def _render_postproc_auto(self):
        """Render Post-Processing > Auto Processing settings."""
        imgui.text_colored("Automatic post-processing settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Enable Auto Post-Processing
        imgui.text("Enable Auto Post-Processing:")
        imgui.same_line(250)
        enable_auto = settings.get("enable_auto_post_processing", False)
        changed, new_val = imgui.checkbox("##EnableAuto", enable_auto)
        if changed:
            settings.set("enable_auto_post_processing", new_val)
        _tooltip_if_hovered("Automatically apply post-processing after tracking")

        # Apply Per-Chapter Profiles
        imgui.text("Per-Chapter Profiles:")
        imgui.same_line(250)
        per_chapter = settings.get("auto_processing_use_chapter_profiles", False)
        changed, new_val = imgui.checkbox("##PerChapter", per_chapter)
        if changed:
            settings.set("auto_processing_use_chapter_profiles", new_val)
        _tooltip_if_hovered("Use different profiles for each chapter type")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Final RDP Pass
        imgui.text_colored("Final RDP Pass", 0.8, 0.8, 0.3, 1.0)

        imgui.text("Enable Final RDP:")
        imgui.same_line(250)
        enable_rdp = settings.get("auto_post_proc_final_rdp_enabled", False)
        changed, new_val = imgui.checkbox("##EnableFinalRDP", enable_rdp)
        if changed:
            settings.set("auto_post_proc_final_rdp_enabled", new_val)
        _tooltip_if_hovered("Apply final RDP simplification pass")

        with _DisabledScope(not enable_rdp):
            imgui.text("Final RDP Epsilon:")
            imgui.same_line(250)
            rdp_epsilon = settings.get("auto_post_proc_final_rdp_epsilon", 1.0)
            changed, new_val = imgui.slider_float("##FinalRDPEps", rdp_epsilon, 0.1, 10.0, "%.1f")
            if changed:
                settings.set("auto_post_proc_final_rdp_epsilon", new_val)
            _tooltip_if_hovered("Epsilon for final RDP pass")

    def _render_postproc_profiles(self):
        """Render Post-Processing > Profiles settings."""
        imgui.text_colored("Per-position processing profiles", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text("Configure different processing profiles for:")
        imgui.bullet_text("Blowjob scenes")
        imgui.bullet_text("Penetration scenes")
        imgui.bullet_text("Handjob scenes")
        imgui.bullet_text("Other/custom scenes")
        imgui.spacing()
        imgui.text("Profiles are managed in the Post-Processing tab")
        imgui.text("of the main control panel.")

    def _render_postproc_plugin(self, plugin_name: str):
        """Render dynamic plugin settings."""
        imgui.text_colored(f"Plugin: {plugin_name}", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text("Plugin parameters are loaded dynamically.")
        imgui.text("Configure plugins in the Post-Processing tab")
        imgui.text("of the main control panel.")
        imgui.spacing()
        imgui.text(f"Selected plugin: {plugin_name}")

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

    # ============================================================
    # CONTENT RENDERING METHODS - Device Control Tab
    # ============================================================

    def _render_device_content(self, horizontal_tab: str):
        """Render Device Control tab content (supporters only)."""
        if horizontal_tab == "Connection":
            self._render_device_connection()
        elif horizontal_tab == "Handy":
            self._render_device_handy()
        elif horizontal_tab == "OSR2":
            self._render_device_osr2()
        elif horizontal_tab == "Buttplug":
            self._render_device_buttplug()
        elif horizontal_tab == "Live Tracking":
            self._render_device_live_tracking()

    def _render_device_connection(self):
        """Render Device Control > Connection settings."""
        imgui.text_colored("Device connection and discovery", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Enable Device Control
        imgui.text("Enable Device Control:")
        imgui.same_line(250)
        device_enabled = settings.get("device_control_enabled", True)
        changed, new_val = imgui.checkbox("##DeviceEnabled", device_enabled)
        if changed:
            settings.set("device_control_enabled", new_val)
        _tooltip_if_hovered("Enable device control features (requires supporter status)")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Backend Selection
        imgui.text_colored("Backend Selection", 0.8, 0.8, 0.3, 1.0)

        imgui.text("Preferred Backend:")
        imgui.same_line(250)
        backend = settings.get("device_control_preferred_backend", "buttplug")
        backends = ["auto", "buttplug", "handy", "osr"]
        backend_index = backends.index(backend) if backend in backends else 0
        changed, new_index = imgui.combo("##PreferredBackend", backend_index, ["Auto-detect", "Buttplug.io", "Handy", "OSR2/SR6"])
        if changed:
            settings.set("device_control_preferred_backend", backends[new_index])
        _tooltip_if_hovered("Choose which device backend to use")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Device Discovery
        imgui.text_colored("Device Discovery", 0.8, 0.8, 0.3, 1.0)
        imgui.text("Use the Device Control panel to:")
        imgui.bullet_text("Discover available devices")
        imgui.bullet_text("Connect to your device")
        imgui.bullet_text("Configure device parameters")

    def _render_device_handy(self):
        """Render Device Control > Handy settings."""
        imgui.text_colored("Handy device settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text("Handy-specific configuration:")
        imgui.spacing()
        imgui.bullet_text("Connection key management")
        imgui.bullet_text("Stroke zone calibration")
        imgui.bullet_text("Speed and position limits")
        imgui.spacing()
        imgui.text_colored("Note:", 1.0, 0.6, 0.0, 1.0)
        imgui.text("Handy settings are configured through the Device Control")
        imgui.text("panel when a Handy device is connected.")

    def _render_device_osr2(self):
        """Render Device Control > OSR2 settings."""
        imgui.text_colored("OSR2/SR6 device settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text("OSR2/SR6-specific configuration:")
        imgui.spacing()
        imgui.bullet_text("Serial port selection")
        imgui.bullet_text("Multi-axis configuration")
        imgui.bullet_text("Speed and range limits")
        imgui.spacing()
        imgui.text_colored("Note:", 1.0, 0.6, 0.0, 1.0)
        imgui.text("OSR2 settings are configured through the Device Control")
        imgui.text("panel when an OSR2/SR6 device is connected.")

    def _render_device_buttplug(self):
        """Render Device Control > Buttplug settings."""
        imgui.text_colored("Buttplug.io server settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Server Address
        imgui.text("Server Address:")
        imgui.same_line(250)
        server_address = settings.get("buttplug_server_address", "localhost")
        imgui.push_item_width(200)
        changed, new_val = imgui.input_text("##ButtplugAddress", server_address, 128)
        imgui.pop_item_width()
        if changed:
            settings.set("buttplug_server_address", new_val)
        _tooltip_if_hovered("Buttplug.io server address or IP")

        # Server Port
        imgui.text("Server Port:")
        imgui.same_line(250)
        server_port = settings.get("buttplug_server_port", 12345)
        changed, new_val = imgui.input_int("##ButtplugPort", server_port)
        if changed:
            settings.set("buttplug_server_port", max(1, min(65535, new_val)))
        _tooltip_if_hovered("Buttplug.io server port (default: 12345)")

        imgui.spacing()

        # Auto Connect
        imgui.text("Auto-connect on Startup:")
        imgui.same_line(250)
        auto_connect = settings.get("buttplug_auto_connect", False)
        changed, new_val = imgui.checkbox("##ButtplugAutoConnect", auto_connect)
        if changed:
            settings.set("buttplug_auto_connect", new_val)
        _tooltip_if_hovered("Automatically connect to Buttplug.io server on startup")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Info box
        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.1, 0.3, 0.5, 0.3)
        imgui.begin_child("ButtplugInfoBox", 0, 80, border=True)
        imgui.text_colored("💡 Buttplug.io Setup", 0.4, 0.8, 1.0, 1.0)
        imgui.text("1. Download and run Intiface Central")
        imgui.text("2. Start the Buttplug.io server")
        imgui.text("3. Use the Device Control panel to connect devices")
        imgui.end_child()
        imgui.pop_style_color()

    def _render_device_live_tracking(self):
        """Render Device Control > Live Tracking settings."""
        imgui.text_colored("Live tracking with device sync", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Max Rate
        imgui.text("Max Update Rate (Hz):")
        imgui.same_line(250)
        max_rate = settings.get("device_control_max_rate_hz", 20.0)
        changed, new_val = imgui.slider_float("##MaxRateHz", max_rate, 1.0, 60.0, "%.1f Hz")
        if changed:
            settings.set("device_control_max_rate_hz", new_val)
        _tooltip_if_hovered("Maximum update rate for device commands during live tracking")

        imgui.spacing()

        # Log Commands
        imgui.text("Log Device Commands:")
        imgui.same_line(250)
        log_commands = settings.get("device_control_log_commands", False)
        changed, new_val = imgui.checkbox("##LogCommands", log_commands)
        if changed:
            settings.set("device_control_log_commands", new_val)
        _tooltip_if_hovered("Log all device commands to console (debug)")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Info
        imgui.text_colored("Live Tracking", 0.8, 0.8, 0.3, 1.0)
        imgui.text("Live tracking allows real-time device sync during video playback:")
        imgui.bullet_text("Connect your device via Device Control panel")
        imgui.bullet_text("Enable live tracking in the tracking settings")
        imgui.bullet_text("Device will respond in real-time to tracking data")

    # ============================================================
    # CONTENT RENDERING METHODS - Streamer Tab
    # ============================================================

    def _render_streamer_content(self, horizontal_tab: str):
        """Render Streamer tab content (supporters only)."""
        if horizontal_tab == "XBVR":
            self._render_streamer_xbvr()
        elif horizontal_tab == "Sync":
            self._render_streamer_sync()
        elif horizontal_tab == "Advanced":
            self._render_streamer_advanced()

    def _render_streamer_xbvr(self):
        """Render Streamer > XBVR settings."""
        imgui.text_colored("XBVR integration settings", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        settings = self._app_settings

        # Enable XBVR
        imgui.text("Enable XBVR Integration:")
        imgui.same_line(250)
        xbvr_enabled = settings.get("xbvr_enabled", True)
        changed, new_val = imgui.checkbox("##XBVREnabled", xbvr_enabled)
        if changed:
            settings.set("xbvr_enabled", new_val)
        _tooltip_if_hovered("Enable XBVR streaming integration")

        imgui.spacing()

        # XBVR Host
        with _DisabledScope(not xbvr_enabled):
            imgui.text("XBVR Server Host:")
            imgui.same_line(250)
            xbvr_host = settings.get("xbvr_host", "localhost")
            imgui.push_item_width(200)
            changed, new_val = imgui.input_text("##XBVRHost", xbvr_host, 128)
            imgui.pop_item_width()
            if changed:
                settings.set("xbvr_host", new_val)
            _tooltip_if_hovered("XBVR server hostname or IP address")

            # XBVR Port
            imgui.text("XBVR Server Port:")
            imgui.same_line(250)
            xbvr_port = settings.get("xbvr_port", 9999)
            changed, new_val = imgui.input_int("##XBVRPort", xbvr_port)
            if changed:
                settings.set("xbvr_port", max(1, min(65535, new_val)))
            _tooltip_if_hovered("XBVR server port (default: 9999)")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Info box
        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.1, 0.3, 0.5, 0.3)
        imgui.begin_child("XBVRInfoBox", 0, 80, border=True)
        imgui.text_colored("💡 XBVR Integration", 0.4, 0.8, 1.0, 1.0)
        imgui.text("XBVR allows FunGen to sync with your VR video library")
        imgui.text("and automatically serve funscripts to connected clients.")
        imgui.text("Visit xbvr.app for more information.")
        imgui.end_child()
        imgui.pop_style_color()

    def _render_streamer_sync(self):
        """Render Streamer > Sync settings."""
        imgui.text_colored("Sync status and client monitoring", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text("Connection Status:")
        imgui.same_line(250)
        imgui.text_colored("Not Connected", 0.8, 0.3, 0.3, 1.0)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text("Connected Clients: 0")
        imgui.spacing()

        # Client list placeholder
        imgui.text_colored("Client Monitoring", 0.8, 0.8, 0.3, 1.0)
        imgui.text("No clients connected.")

    def _render_streamer_advanced(self):
        """Render Streamer > Advanced settings."""
        imgui.text_colored("Advanced streaming options", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text("Additional streaming options will be added here:")
        imgui.bullet_text("Client authentication")
        imgui.bullet_text("Bandwidth optimization")
        imgui.bullet_text("Compression settings")
        imgui.bullet_text("Synchronization tuning")

    # ============================================================
    # CONTENT RENDERING METHODS - Keyboard Shortcuts Tab
    # ============================================================

    def _render_keyboard_content(self, horizontal_tab: str):
        """Render Keyboard Shortcuts tab content."""
        if horizontal_tab == "Navigation":
            self._render_keyboard_navigation()
        elif horizontal_tab == "Editing":
            self._render_keyboard_editing()
        elif horizontal_tab == "Project":
            self._render_keyboard_project()
        elif horizontal_tab == "Chapters":
            self._render_keyboard_chapters()

    def _render_keyboard_navigation(self):
        """Render Keyboard > Navigation shortcuts."""
        imgui.text_colored("Navigation and playback shortcuts", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        # Table of shortcuts
        imgui.text_colored("Playback Control", 0.8, 0.8, 0.3, 1.0)
        self._render_shortcut_row("Play/Pause", "Space")
        self._render_shortcut_row("Next Frame", "Right Arrow")
        self._render_shortcut_row("Previous Frame", "Left Arrow")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored("Video Navigation", 0.8, 0.8, 0.3, 1.0)
        self._render_shortcut_row("Jump to Start", "Home")
        self._render_shortcut_row("Jump to End", "End")
        self._render_shortcut_row("Pan Timeline Left", "Shift+Left")
        self._render_shortcut_row("Pan Timeline Right", "Shift+Right")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored("Timeline View", 0.8, 0.8, 0.3, 1.0)
        self._render_shortcut_row("Zoom In", "+")
        self._render_shortcut_row("Zoom Out", "-")
        self._render_shortcut_row("Reset View", "Ctrl+0")

    def _render_keyboard_editing(self):
        """Render Keyboard > Editing shortcuts."""
        imgui.text_colored("Editing and point manipulation shortcuts", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text_colored("Undo/Redo", 0.8, 0.8, 0.3, 1.0)
        self._render_shortcut_row("Undo", "Ctrl+Z")
        self._render_shortcut_row("Redo", "Ctrl+Y")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored("Selection", 0.8, 0.8, 0.3, 1.0)
        self._render_shortcut_row("Select All", "Ctrl+A")
        self._render_shortcut_row("Deselect All", "Ctrl+D")
        self._render_shortcut_row("Copy", "Ctrl+C")
        self._render_shortcut_row("Paste", "Ctrl+V")
        self._render_shortcut_row("Delete Point", "Delete")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored("Point Navigation", 0.8, 0.8, 0.3, 1.0)
        self._render_shortcut_row("Next Point", "Ctrl+Right")
        self._render_shortcut_row("Previous Point", "Ctrl+Left")
        self._render_shortcut_row("Raise Value", "Ctrl+Up")
        self._render_shortcut_row("Lower Value", "Ctrl+Down")

    def _render_keyboard_project(self):
        """Render Keyboard > Project shortcuts."""
        imgui.text_colored("Project and file management shortcuts", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text_colored("File Operations", 0.8, 0.8, 0.3, 1.0)
        self._render_shortcut_row("New Project", "Ctrl+N")
        self._render_shortcut_row("Open Project", "Ctrl+O")
        self._render_shortcut_row("Save Project", "Ctrl+S")
        self._render_shortcut_row("Save As", "Ctrl+Shift+S")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored("Export", 0.8, 0.8, 0.3, 1.0)
        self._render_shortcut_row("Export Funscript", "Ctrl+E")
        self._render_shortcut_row("Quick Export", "Ctrl+Shift+E")

    def _render_keyboard_chapters(self):
        """Render Keyboard > Chapters shortcuts."""
        imgui.text_colored("Chapter management shortcuts", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text_colored("Chapter Creation", 0.8, 0.8, 0.3, 1.0)
        self._render_shortcut_row("Set Chapter Start", "I")
        self._render_shortcut_row("Set Chapter End", "O")
        self._render_shortcut_row("Create Chapter", "Ctrl+I")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored("Chapter Management", 0.8, 0.8, 0.3, 1.0)
        self._render_shortcut_row("Delete Chapter", "Ctrl+Delete")
        self._render_shortcut_row("Delete Points in Chapter", "Ctrl+Shift+Delete")
        self._render_shortcut_row("Next Chapter", "Page Down")
        self._render_shortcut_row("Previous Chapter", "Page Up")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Info box
        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.1, 0.3, 0.5, 0.3)
        imgui.begin_child("KeyboardInfoBox", 0, 60, border=True)
        imgui.text_colored("💡 Customize Shortcuts", 0.4, 0.8, 1.0, 1.0)
        imgui.text("Press F1 or use Help → Keyboard Shortcuts to view and")
        imgui.text("customize all keyboard shortcuts.")
        imgui.end_child()
        imgui.pop_style_color()

    def _render_shortcut_row(self, action: str, shortcut: str):
        """Render a single shortcut row in a table-like format."""
        imgui.text(action)
        imgui.same_line(300)
        imgui.text_colored(shortcut, 0.4, 1.0, 0.4, 1.0)

    # ============================================================
    # CONTENT RENDERING METHODS - About Tab
    # ============================================================

    def _render_about_content(self, horizontal_tab: str):
        """Render About tab content."""
        if horizontal_tab == "Info":
            self._render_about_info()
        elif horizontal_tab == "Support":
            self._render_about_support()
        elif horizontal_tab == "Updates":
            self._render_about_updates()

    def _render_about_info(self):
        """Render About > Info content."""
        imgui.text_colored("FunGen - AI-Powered Funscript Generator", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        # Version info
        version = getattr(self.app, 'version', '0.1.0')
        imgui.text(f"Version: {version}")
        imgui.text("Build Date: 2025-01-18")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Links
        if imgui.button("🌐 Visit GitHub Repository", width=300):
            import webbrowser
            webbrowser.open("https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator")

        imgui.spacing()

        if imgui.button("📖 View Documentation", width=300):
            import webbrowser
            webbrowser.open("https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/wiki")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # License
        imgui.text_colored("License", 0.8, 0.8, 0.3, 1.0)
        imgui.text("This software is open source.")
        imgui.text("See LICENSE file for details.")

    def _render_about_support(self):
        """Render About > Support content."""
        imgui.text_colored("Support Development", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        imgui.text("If you find FunGen useful, please consider supporting")
        imgui.text("the development through Ko-fi:")
        imgui.spacing()

        if imgui.button("☕ Support on Ko-fi", width=300):
            import webbrowser
            webbrowser.open("https://ko-fi.com/fungendev")  # Replace with actual Ko-fi link

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored("Why Support?", 0.8, 0.8, 0.3, 1.0)
        imgui.bullet_text("Helps maintain and improve FunGen")
        imgui.bullet_text("Enables new feature development")
        imgui.bullet_text("Supports AI model training and optimization")
        imgui.bullet_text("Access to supporter-only features")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored("Supporter Features", 0.8, 0.8, 0.3, 1.0)
        imgui.bullet_text("Device Control (Handy, OSR2, Buttplug.io)")
        imgui.bullet_text("XBVR Streaming Integration")
        imgui.bullet_text("Advanced live tracking features")
        imgui.bullet_text("Priority support and feature requests")

    def _render_about_updates(self):
        """Render About > Updates content."""
        imgui.text_colored("Check for Updates", 0.4, 0.8, 1.0, 1.0)
        imgui.spacing()

        version = getattr(self.app, 'version', '0.1.0')
        imgui.text(f"Current Version: {version}")
        imgui.spacing()

        if imgui.button("🔍 Check for Updates", width=300):
            # TODO: Implement update check
            imgui.open_popup("UpdateCheckPopup")

        # Update check popup
        if imgui.begin_popup_modal("UpdateCheckPopup")[0]:
            imgui.text("Checking for updates...")
            imgui.spacing()
            imgui.text("You are running the latest version!")
            imgui.spacing()
            if imgui.button("OK", width=100):
                imgui.close_current_popup()
            imgui.end_popup()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored("Update Settings", 0.8, 0.8, 0.3, 1.0)

        settings = self._app_settings
        auto_check = settings.get("auto_check_updates", True)
        changed, new_val = imgui.checkbox("Automatically check for updates on startup", auto_check)
        if changed:
            settings.set("auto_check_updates", new_val)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored("Release Notes", 0.8, 0.8, 0.3, 1.0)
        if imgui.button("📝 View Release Notes", width=300):
            import webbrowser
            webbrowser.open("https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/releases")
