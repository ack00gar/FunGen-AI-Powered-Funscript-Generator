"""Device Control — AxisConfig methods."""
import asyncio
import threading
import time
import imgui
from application.utils.imgui_helpers import tooltip_if_hovered as _tooltip_if_hovered
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.section_card import section_card as _section_card
from application.utils import primary_button_style, destructive_button_style

# Canonical TCode channel → friendly name mapping (used across axis config UI)
_CHANNEL_FRIENDLY = {
    "L0": "Stroke", "L1": "Sway", "L2": "Surge",
    "R0": "Twist", "R1": "Roll", "R2": "Pitch",
    "V0": "Vibration", "V1": "Pump",
}



class AxisConfigMixin:
    """Mixin fragment for DeviceControlMixin."""

    def _render_quick_controls(self):
        """Render quick controls for connected device (always visible when connected)."""
        # Bookmark navigation
        self._render_bookmark_nav()


    def _render_bookmark_nav(self):
        """Render prev/next bookmark navigation buttons."""
        try:
            # Get bookmark manager from primary timeline editor
            bm_mgr = None
            gui = getattr(self.app, 'gui', None)
            if gui and hasattr(gui, 'timeline_editors'):
                editors = gui.timeline_editors
                if editors:
                    bm_mgr = getattr(editors[0], '_bookmark_manager', None)

            if not bm_mgr or not bm_mgr.bookmarks:
                return

            proc = self.app.processor
            if not proc or not proc.is_video_open():
                return

            fps = proc.fps or 30.0
            current_time_ms = (proc.current_frame_index / fps) * 1000.0

            imgui.text("Bookmarks:")
            imgui.same_line()

            if imgui.small_button("<< Prev##BmkPrev"):
                bm = bm_mgr.get_nearest(current_time_ms, direction=-1)
                if bm:
                    frame_idx = int(bm.time_ms / 1000.0 * fps)
                    self.app.event_handlers.seek_video_with_sync(frame_idx)
            _tooltip_if_hovered("Jump to previous bookmark")

            imgui.same_line()
            if imgui.small_button("Next >>##BmkNext"):
                bm = bm_mgr.get_nearest(current_time_ms, direction=1)
                if bm:
                    frame_idx = int(bm.time_ms / 1000.0 * fps)
                    self.app.event_handlers.seek_video_with_sync(frame_idx)
            _tooltip_if_hovered("Jump to next bookmark")

        except Exception:
            pass  # Bookmarks unavailable — silently skip

    # ── Unified Axis Configuration Panel ──────────────────────────────


    def _render_axis_configuration(self):
        """Render the unified axis configuration panel for all device types."""
        from device_control.axis_routing import (
            DEVICE_CHANNELS, TCODE_DEFAULT_AXIS, AxisRoute,
        )
        # Fallback for device_control < 5.4
        try:
            from device_control.axis_routing import CHANNEL_FRIENDLY_NAMES
        except ImportError:
            CHANNEL_FRIENDLY_NAMES = _CHANNEL_FRIENDLY

        # Get current timeline axis assignments
        axis_assignments = {}
        if hasattr(self, 'device_video_integration') and self.device_video_integration:
            axis_assignments = self.device_video_integration.get_axis_assignments()
        if not axis_assignments:
            axis_assignments = {1: "stroke"}

        # Build source options: "(none)" + all assigned axes
        source_options = ["(none)"]
        source_labels = ["(none)"]
        for tl_num, axis_name in sorted(axis_assignments.items()):
            source_options.append(axis_name)
            source_labels.append(f"{axis_name} (TL {tl_num})")

        router = self.device_manager.axis_router

        for device_id, backend in self.device_manager.connected_devices.items():
            device_type = self.device_manager.get_device_type_for_id(device_id)
            config = router.get_config(device_id)

            if not config:
                config = router.auto_detect(device_id, device_type, axis_assignments)

            channels = DEVICE_CHANNELS.get(device_type, ["L0"])
            is_osr = device_type == "osr"
            is_multi_axis = is_osr  # OSR has multiple TCode channels

            # Device header
            device_name = self.device_manager.get_connected_device_name()
            imgui.text_colored(f"{device_name}", 0.7, 0.7, 0.9)
            imgui.same_line()
            if imgui.small_button(f"Auto-Detect##{device_id}"):
                router.auto_detect(device_id, device_type, axis_assignments)
                config = router.get_config(device_id)
                router.save_to_settings(self.app.app_settings)

            # OSR profile selector (only for OSR devices)
            if is_osr:
                imgui.same_line()
                imgui.text("Profile:")
                imgui.same_line()
                osr_profiles = self.app.app_settings.get("device_control_osr_profiles", {})
                profile_names = list(osr_profiles.keys())
                current_profile_name = self.app.app_settings.get("device_control_selected_profile", "Balanced")
                current_idx = profile_names.index(current_profile_name) if current_profile_name in profile_names else 0
                imgui.push_item_width(120)
                changed, new_idx = imgui.combo(f"##profile_{device_id}", current_idx, profile_names)
                imgui.pop_item_width()
                if changed and 0 <= new_idx < len(profile_names):
                    new_profile = profile_names[new_idx]
                    self.app.app_settings.set("device_control_selected_profile", new_profile)
                    self._load_osr_profile_to_routes(device_id, new_profile)
                    router.save_to_settings(self.app.app_settings)

            # On first render for OSR: sync profile → routes if routes have defaults
            if is_osr and not getattr(self, f'_osr_routes_synced_{device_id}', False):
                setattr(self, f'_osr_routes_synced_{device_id}', True)
                current_profile_name = self.app.app_settings.get("device_control_selected_profile", "Balanced")
                osr_profiles = self.app.app_settings.get("device_control_osr_profiles", {})
                if current_profile_name in osr_profiles:
                    all_default = all(
                        getattr(r, 'min_value', 0.0) == 0.0 and getattr(r, 'max_value', 100.0) == 100.0
                        for r in config.routes.values()
                    )
                    if all_default:
                        self._load_osr_profile_to_routes(device_id, current_profile_name)

            imgui.spacing()

            # Ensure all channels have routes
            for ch in channels:
                if ch not in config.routes:
                    config.routes[ch] = AxisRoute(device_channel=ch, source_axis="none", enabled=False)

            if is_multi_axis:
                # Multi-axis table: 4 columns (Channel, Source, Range, En)
                imgui.columns(4, f"axis_config_table_{device_id}", True)
                imgui.set_column_width(0, 80)   # Channel
                imgui.set_column_width(1, 140)  # Source
                imgui.set_column_width(2, 150)  # Range (dual slider)
                imgui.set_column_width(3, 30)   # En

                imgui.text("Channel")
                imgui.next_column()
                imgui.text("Source")
                imgui.next_column()
                imgui.text("Range")
                imgui.next_column()
                imgui.text("En")
                imgui.next_column()
                imgui.separator()

                for ch in channels:
                    route = config.routes[ch]
                    self._render_axis_row_multi(device_id, ch, route, source_options, source_labels, router, is_osr)

                imgui.columns(1)
            else:
                # Single-axis: compact inline layout (no table)
                ch = channels[0]
                route = config.routes[ch]
                self._render_axis_row_single(device_id, ch, route, source_options, source_labels, router)

            # Expandable details per axis (includes Invert checkbox)
            for ch in channels:
                route = config.routes.get(ch)
                if not route:
                    continue
                friendly = CHANNEL_FRIENDLY_NAMES.get(ch, ch)
                detail_key = f"{device_id}_{ch}"
                if imgui.tree_node(f"{ch} {friendly} Details##{detail_key}"):
                    self._axis_details_expanded[detail_key] = True
                    self._render_axis_details(device_id, ch, route, router, is_osr, friendly)
                    imgui.tree_pop()
                else:
                    self._axis_details_expanded[detail_key] = False


    def _render_axis_row_single(self, device_id, ch, route, source_options, source_labels, router):
        """Render compact single-axis configuration (Handy, Buttplug, OSSM)."""
        friendly = _CHANNEL_FRIENDLY.get(ch, ch)

        # Source dropdown
        imgui.text("Source:")
        imgui.same_line()
        current_idx = 0
        if route.source_axis in source_options:
            current_idx = source_options.index(route.source_axis)
        imgui.push_item_width(-1)
        changed, new_idx = imgui.combo(f"##{device_id}_{ch}_src", current_idx, source_labels)
        imgui.pop_item_width()
        if changed:
            route.source_axis = source_options[new_idx]
            if route.source_axis == "(none)":
                route.enabled = False
            router.save_to_settings(self.app.app_settings)

        # Range: dual-handle slider (full width)
        imgui.text("Range:")
        imgui.same_line()
        min_val = getattr(route, 'min_value', 0.0)
        max_val = getattr(route, 'max_value', 100.0)
        imgui.push_item_width(-1)
        changed, new_min, new_max = imgui.drag_float_range2(
            f"##{device_id}_{ch}_range", min_val, max_val,
            0.5, 0.0, 100.0, "Min %.0f%%", "Max %.0f%%"
        )
        imgui.pop_item_width()
        if changed and hasattr(route, 'min_value'):
            route.min_value = new_min
            route.max_value = new_max
            router.save_to_settings(self.app.app_settings)

        # Enabled + Inverted on same line
        changed_en, new_en = imgui.checkbox(f"Enabled##{device_id}_{ch}_en", route.enabled)
        if changed_en:
            route.enabled = new_en
            router.save_to_settings(self.app.app_settings)
        imgui.same_line()
        changed_inv, new_inv = imgui.checkbox(f"Inverted##{device_id}_{ch}_inv", route.invert)
        if changed_inv:
            route.invert = new_inv
            router.save_to_settings(self.app.app_settings)


    def _render_axis_row_multi(self, device_id, ch, route, source_options, source_labels, router, is_osr):
        """Render a single axis row in the multi-axis configuration table (OSR)."""
        friendly = _CHANNEL_FRIENDLY.get(ch, ch)

        # Channel label
        imgui.text(f"{ch} {friendly}")
        imgui.next_column()

        # Source dropdown
        current_idx = 0
        if route.source_axis in source_options:
            current_idx = source_options.index(route.source_axis)
        imgui.push_item_width(-1)
        changed, new_idx = imgui.combo(f"##{device_id}_{ch}_src", current_idx, source_labels)
        imgui.pop_item_width()
        if changed:
            route.source_axis = source_options[new_idx]
            if route.source_axis == "(none)":
                route.enabled = False
            router.save_to_settings(self.app.app_settings)
            if is_osr:
                self._save_routes_to_osr_profile(device_id)
        imgui.next_column()

        # Range: dual-handle slider
        min_val = getattr(route, 'min_value', 0.0)
        max_val = getattr(route, 'max_value', 100.0)
        imgui.push_item_width(-1)
        changed, new_min, new_max = imgui.drag_float_range2(
            f"##{device_id}_{ch}_range", min_val, max_val,
            0.5, 0.0, 100.0, "%.0f%%", "%.0f%%"
        )
        imgui.pop_item_width()
        if changed and hasattr(route, 'min_value'):
            route.min_value = new_min
            route.max_value = new_max
            router.save_to_settings(self.app.app_settings)
            if is_osr:
                self._save_routes_to_osr_profile(device_id)
        imgui.next_column()

        # Enable checkbox
        changed, new_val = imgui.checkbox(f"##{device_id}_{ch}_en", route.enabled)
        if changed:
            route.enabled = new_val
            router.save_to_settings(self.app.app_settings)
            if is_osr:
                self._save_routes_to_osr_profile(device_id)
        imgui.next_column()


    def _render_axis_details(self, device_id, ch, route, router, is_osr, friendly):
        """Render expandable detail section for one axis."""
        imgui.indent(10)
        settings_changed = False
        _has_extended = hasattr(route, 'speed_multiplier')

        # Invert checkbox (moved here from table to save column space)
        changed_inv, new_inv = imgui.checkbox(f"Invert##{device_id}_{ch}_inv", route.invert)
        if changed_inv:
            route.invert = new_inv
            settings_changed = True

        # Speed multiplier
        speed = getattr(route, 'speed_multiplier', 1.0)
        changed, new_speed = imgui.slider_float(
            f"Speed Multiplier##{device_id}_{ch}", speed, 0.1, 3.0, "%.2f"
        )
        if changed and _has_extended:
            route.speed_multiplier = new_speed
            settings_changed = True

        # Smoothing
        smooth = getattr(route, 'smoothing_factor', 0.3)
        changed, new_smooth = imgui.slider_float(
            f"Smoothing##{device_id}_{ch}", smooth, 0.0, 1.0, "%.2f"
        )
        if changed and _has_extended:
            route.smoothing_factor = new_smooth
            settings_changed = True

        # Pattern / motion provider
        pattern_types = ["disabled", "wave", "follow", "auto", "random_noise"]
        pattern_labels = ["Disabled", "Wave (Smooth)", "Follow Primary", "Auto-Select", "Random Noise"]
        cur_pat = getattr(route, 'motion_provider_pattern', "disabled")
        cur_pat_idx = pattern_types.index(cur_pat) if cur_pat in pattern_types else 0
        changed, new_pat_idx = imgui.combo(
            f"Pattern##{device_id}_{ch}", cur_pat_idx, pattern_labels
        )
        if changed and 0 <= new_pat_idx < len(pattern_types) and _has_extended:
            route.motion_provider_pattern = pattern_types[new_pat_idx]
            settings_changed = True

        if cur_pat != "disabled":
            intensity = getattr(route, 'motion_provider_intensity', 1.0)
            changed, new_int = imgui.slider_float(
                f"Intensity##{device_id}_{ch}", intensity, 0.0, 2.0, "%.2f"
            )
            if changed and _has_extended:
                route.motion_provider_intensity = new_int
                settings_changed = True

            freq = getattr(route, 'motion_provider_frequency', 1.0)
            changed, new_freq = imgui.slider_float(
                f"Frequency##{device_id}_{ch}", freq, 0.1, 5.0, "%.2f"
            )
            if changed and _has_extended:
                route.motion_provider_frequency = new_freq
                settings_changed = True

            if cur_pat in ("follow", "auto"):
                follow = getattr(route, 'motion_provider_follow_strength', 0.5)
                changed, new_fs = imgui.slider_float(
                    f"Follow Strength##{device_id}_{ch}", follow, 0.0, 1.0, "%.2f"
                )
                if changed and _has_extended:
                    route.motion_provider_follow_strength = new_fs
                    settings_changed = True
                _tooltip_if_hovered("How closely this axis follows the primary axis movement")

        # Test / demo buttons
        imgui.spacing()
        imgui.text("Test:")
        imgui.same_line()
        min_pct = getattr(route, 'min_value', 0.0)
        max_pct = getattr(route, 'max_value', 100.0)
        if imgui.small_button(f"Min##{device_id}_{ch}_tmin"):
            self._preview_axis_position_pct(ch, min_pct, f"Testing {friendly} min")
        imgui.same_line()
        if imgui.small_button(f"Max##{device_id}_{ch}_tmax"):
            self._preview_axis_position_pct(ch, max_pct, f"Testing {friendly} max")
        imgui.same_line()
        if imgui.small_button(f"Center##{device_id}_{ch}_tctr"):
            self._preview_axis_position_pct(ch, (min_pct + max_pct) / 2.0, f"Centering {friendly}")

        # Simulation patterns
        imgui.same_line()
        if imgui.small_button(f"Demo##{device_id}_{ch}"):
            self._demo_axis_range_pct(ch, min_pct, max_pct, route.invert, friendly)
        imgui.same_line()
        if imgui.small_button(f"Sine##{device_id}_{ch}"):
            self._simulate_axis_pattern_pct(ch, "sine_wave", min_pct, max_pct, route.invert, friendly)
        imgui.same_line()
        if imgui.small_button(f"Square##{device_id}_{ch}"):
            self._simulate_axis_pattern_pct(ch, "square_wave", min_pct, max_pct, route.invert, friendly)
        if imgui.small_button(f"Triangle##{device_id}_{ch}"):
            self._simulate_axis_pattern_pct(ch, "triangle_wave", min_pct, max_pct, route.invert, friendly)
        imgui.same_line()
        if imgui.small_button(f"Random##{device_id}_{ch}"):
            self._simulate_axis_pattern_pct(ch, "random", min_pct, max_pct, route.invert, friendly)
        imgui.same_line()
        if imgui.small_button(f"Pulse##{device_id}_{ch}"):
            self._simulate_axis_pattern_pct(ch, "pulse", min_pct, max_pct, route.invert, friendly)

        if settings_changed:
            router.save_to_settings(self.app.app_settings)
            if is_osr:
                self._save_routes_to_osr_profile(device_id)

        imgui.unindent(10)


    def _load_osr_profile_to_routes(self, device_id, profile_name):
        """Load OSR profile TCode values into AxisRoute 0-100% values."""
        osr_profiles = self.app.app_settings.get("device_control_osr_profiles", {})
        profile_data = osr_profiles.get(profile_name, {})
        if not profile_data:
            return

        router = self.device_manager.axis_router
        config = router.get_config(device_id)
        if not config:
            return

        # Map OSR profile axis keys → TCode channel
        _PROFILE_KEY_TO_CH = {
            'up_down': 'L0', 'left_right': 'L1', 'front_back': 'L2',
            'twist': 'R0', 'roll': 'R1', 'pitch': 'R2',
            'vibration': 'V0', 'aux_vibration': 'V1',
        }

        for axis_key, ch in _PROFILE_KEY_TO_CH.items():
            axis_data = profile_data.get(axis_key)
            route = config.routes.get(ch)
            if not axis_data or not route:
                continue
            route.enabled = axis_data.get("enabled", False)
            route.invert = axis_data.get("invert", False)
            # Extended fields — safe setattr for compat with device_control < 5.4
            _ext = {
                'min_value': (axis_data.get("min_position", 0) / 9999.0) * 100.0,
                'max_value': (axis_data.get("max_position", 9999) / 9999.0) * 100.0,
                'speed_multiplier': axis_data.get("speed_multiplier", 1.0),
                'smoothing_factor': axis_data.get("smoothing_factor", 0.3),
                'motion_provider_pattern': axis_data.get("pattern_type", "disabled"),
                'motion_provider_intensity': axis_data.get("pattern_intensity", 1.0),
                'motion_provider_frequency': axis_data.get("pattern_frequency", 1.0),
                'motion_provider_follow_strength': axis_data.get("follow_strength", 0.5),
            }
            for attr, val in _ext.items():
                if hasattr(route, attr):
                    setattr(route, attr, val)

        # Also load profile to the device backend
        self._load_osr_profile_to_device(profile_name, profile_data)


    def _save_routes_to_osr_profile(self, device_id):
        """Write AxisRoute values back to the active OSR profile."""
        router = self.device_manager.axis_router
        config = router.get_config(device_id)
        if not config:
            return

        current_profile_name = self.app.app_settings.get("device_control_selected_profile", "Balanced")
        osr_profiles = self.app.app_settings.get("device_control_osr_profiles", {})
        profile_data = osr_profiles.get(current_profile_name)
        if not profile_data:
            return

        _CH_TO_PROFILE_KEY = {
            'L0': 'up_down', 'L1': 'left_right', 'L2': 'front_back',
            'R0': 'twist', 'R1': 'roll', 'R2': 'pitch',
            'V0': 'vibration', 'V1': 'aux_vibration',
        }

        for ch, route in config.routes.items():
            axis_key = _CH_TO_PROFILE_KEY.get(ch)
            if not axis_key or axis_key not in profile_data:
                continue
            axis_data = profile_data[axis_key]
            axis_data["enabled"] = route.enabled
            axis_data["invert"] = route.invert
            axis_data["min_position"] = int((getattr(route, 'min_value', 0.0) / 100.0) * 9999)
            axis_data["max_position"] = int((getattr(route, 'max_value', 100.0) / 100.0) * 9999)
            axis_data["speed_multiplier"] = getattr(route, 'speed_multiplier', 1.0)
            axis_data["smoothing_factor"] = getattr(route, 'smoothing_factor', 0.3)
            axis_data["pattern_type"] = getattr(route, 'motion_provider_pattern', "disabled")
            axis_data["pattern_intensity"] = getattr(route, 'motion_provider_intensity', 1.0)
            axis_data["pattern_frequency"] = getattr(route, 'motion_provider_frequency', 1.0)
            axis_data["follow_strength"] = getattr(route, 'motion_provider_follow_strength', 0.5)

        osr_profiles[current_profile_name] = profile_data
        self.app.app_settings.set("device_control_osr_profiles", osr_profiles)
        self.app.app_settings.save_settings()

    # ── Device-agnostic preview/demo helpers (0-100% based) ──────────


    def _preview_axis_position_pct(self, channel, position_pct, message):
        """Preview a specific axis position using 0-100% (device-agnostic)."""
        try:
            if not self.device_manager.is_connected():
                return

            backend = self.device_manager.get_connected_backend()
            if not backend or not backend.is_connected():
                return

            def run_preview():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    success = loop.run_until_complete(backend.set_axis_position(channel, position_pct))
                    if success:
                        self.app.logger.debug(f"{message}: {channel} to {position_pct:.1f}%")
                except Exception as e:
                    self.app.logger.error(f"Failed to preview axis position: {e}")
                finally:
                    loop.close()

            thread = threading.Thread(target=run_preview, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to preview axis position: {e}")


    def _demo_axis_range_pct(self, channel, min_pct, max_pct, inverted, label):
        """Demonstrate the full range of an axis using 0-100% values."""
        try:
            if not self.device_manager.is_connected():
                return

            def run_demo():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    backend = self.device_manager.get_connected_backend()
                    if not backend:
                        return

                    center = (min_pct + max_pct) / 2.0
                    if inverted:
                        sequence = [(max_pct, "0% (inv)"), (min_pct, "100% (inv)"), (center, "center")]
                    else:
                        sequence = [(min_pct, "0%"), (max_pct, "100%"), (center, "center")]

                    self.app.logger.info(f"Demonstrating {label} range...")
                    for pos, desc in sequence:
                        loop.run_until_complete(backend.set_axis_position(channel, pos))
                        self.app.logger.info(f"{label} demo: {desc} -> {channel} at {pos:.1f}%")
                        time.sleep(2.0)
                    self.app.logger.info(f"{label} range demo complete")
                except Exception as e:
                    self.app.logger.error(f"Failed to demo axis range: {e}")
                finally:
                    loop.close()

            thread = threading.Thread(target=run_demo, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to start axis demo: {e}")


    def _simulate_axis_pattern_pct(self, channel, pattern_type, min_pct, max_pct, inverted, label):
        """Simulate motion patterns using 0-100% values (device-agnostic)."""
        try:
            import math
            import random

            def run_pattern():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    backend = self.device_manager.get_connected_backend()
                    if not backend:
                        return

                    center = (min_pct + max_pct) / 2.0
                    amplitude = (max_pct - min_pct) / 2.0
                    duration = 10.0
                    steps = 50
                    dt = duration / steps

                    positions = []
                    if pattern_type == "sine_wave":
                        for i in range(steps):
                            t = (i / steps) * 4 * math.pi
                            pos = center + amplitude * math.sin(t)
                            positions.append(pos)
                    elif pattern_type == "square_wave":
                        for i in range(steps):
                            t = (i / steps) * 4
                            pos = max_pct if (t % 2) < 1 else min_pct
                            positions.append(pos)
                    elif pattern_type == "triangle_wave":
                        for i in range(steps):
                            t = (i / steps) * 4
                            cycle_pos = t % 2
                            if cycle_pos < 1:
                                pos = min_pct + (max_pct - min_pct) * cycle_pos
                            else:
                                pos = max_pct - (max_pct - min_pct) * (cycle_pos - 1)
                            positions.append(pos)
                    elif pattern_type == "random":
                        for _ in range(20):
                            positions.append(random.uniform(min_pct, max_pct))
                        dt = duration / 20
                    elif pattern_type == "pulse":
                        for _ in range(10):
                            positions.extend([max_pct, center])
                        dt = duration / 20

                    self.app.logger.info(f"Starting {pattern_type} for {label}...")
                    for pos in positions:
                        if inverted:
                            pos = max_pct + min_pct - pos
                        loop.run_until_complete(backend.set_axis_position(channel, pos))
                        time.sleep(dt)

                    # Return to center
                    loop.run_until_complete(backend.set_axis_position(channel, center))
                    self.app.logger.info(f"{label} {pattern_type} complete")
                except Exception as e:
                    self.app.logger.error(f"Error in {pattern_type} pattern: {e}")
                finally:
                    loop.close()

            thread = threading.Thread(target=run_pattern, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to start {pattern_type} pattern: {e}")

