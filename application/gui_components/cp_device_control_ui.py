"""Device Control tab UI mixin for ControlPanelUI."""
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


class DeviceControlMixin:
    """Mixin providing Device Control tab rendering methods."""

    def _render_device_control_tab(self):
        """Render device control tab content."""
        try:
            # Safety check: Don't initialize during first frame to avoid segfault
            # The app needs to be fully initialized before creating device manager
            if not hasattr(self, '_first_frame_rendered'):
                self._first_frame_rendered = False

            if not self._first_frame_rendered:
                imgui.text("Device Control initializing...")
                imgui.text("Please wait for application to fully load.")
                self._first_frame_rendered = True
                return

            # Initialize device control system lazily
            if not self._device_control_initialized:
                self._initialize_device_control()

            # If device control is available, render the UI
            if self.device_manager and self.param_manager:
                self._render_device_control_content()
            else:
                imgui.text("Device Control system failed to initialize.")
                err = getattr(self, '_device_control_init_error', None)
                if err is not None:
                    imgui.text_colored(f"Error: {err}", 1.0, 0.4, 0.4, 1.0)
                    if isinstance(err, (AttributeError, TypeError)):
                        imgui.spacing()
                        imgui.text_colored("This looks like a version mismatch.", 1.0, 0.85, 0.2, 1.0)
                        imgui.text_colored("Did you update to the latest Device Control version?", 1.0, 0.85, 0.2, 1.0)
                        imgui.text_colored("Install the latest device_control zip from your purchase.", 0.8, 0.8, 0.8, 1.0)
                    elif isinstance(err, (ImportError, ModuleNotFoundError)):
                        imgui.spacing()
                        imgui.text_colored("Device Control addon not found or incomplete.", 1.0, 0.85, 0.2, 1.0)
                        imgui.text_colored("Install the latest device_control zip from your purchase.", 0.8, 0.8, 0.8, 1.0)
                    else:
                        imgui.text_colored("Check logs for details.", 1.0, 0.5, 0.0, 1.0)
                else:
                    imgui.text_colored("Check logs for details.", 1.0, 0.5, 0.0, 1.0)
                imgui.spacing()
                if imgui.button("Retry Initialization"):
                    self._device_control_initialized = False
                    self._device_control_init_error = None

        except Exception as e:
            imgui.text_colored(f"Error in Device Control: {e}", 1.0, 0.4, 0.4, 1.0)
            if isinstance(e, (AttributeError, TypeError)):
                imgui.spacing()
                imgui.text_colored("This looks like a version mismatch.", 1.0, 0.85, 0.2, 1.0)
                imgui.text_colored("Did you update to the latest Device Control version?", 1.0, 0.85, 0.2, 1.0)
                imgui.text_colored("Install the latest device_control zip from your purchase.", 0.8, 0.8, 0.8, 1.0)
            elif isinstance(e, (ImportError, ModuleNotFoundError)):
                imgui.text_colored("Device Control addon not found or incomplete.", 1.0, 0.85, 0.2, 1.0)
                imgui.text_colored("Install the latest device_control zip from your purchase.", 0.8, 0.8, 0.8, 1.0)
            imgui.text_colored("See logs for full details.", 0.7, 0.7, 0.7, 1.0)

    def _initialize_device_control(self):
        """Initialize device control system for the control panel."""
        try:
            from device_control.device_manager import DeviceManager, DeviceControlConfig
            from device_control.device_parameterization import DeviceParameterManager

            self.app.logger.info("Device Control: Starting initialization...")

            # Create device manager with default config
            config = DeviceControlConfig(
                enable_live_tracking=True,
                enable_funscript_playback=True,
                preferred_backend="auto",
                log_device_commands=False  # Disable excessive logging in production
            )

            self.app.logger.info("Device Control: Creating DeviceManager...")
            self.device_manager = DeviceManager(config, app_instance=self.app)

            # Share device manager with app for TrackerManager integration
            self.app.device_manager = self.device_manager
            self.app.logger.info("Device Control: DeviceManager created and shared with app")

            # Initialize video integration (observer pattern for desktop video playback)
            self.app.logger.info("Device Control: Setting up video playback integration...")
            from device_control.video_integration import DeviceControlVideoIntegration
            from device_control.bridges.video_playback_bridge import VideoPlaybackBridge

            # Create integration (connects to video_processor via observer pattern)
            self.device_video_integration = DeviceControlVideoIntegration(
                self.app.processor,
                self.device_manager,
                app_instance=self.app,
                logger=self.app.logger
            )

            # Create video playback bridge (polls integration at device update rate)
            self.device_video_bridge = VideoPlaybackBridge(
                self.device_manager,
                video_integration=self.device_video_integration
            )

            # Start integration (registers callbacks with video_processor)
            self.device_video_integration.start()

            # Start bridge in background thread with its own event loop
            import threading
            import asyncio

            def run_bridge_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.device_video_bridge.start())
                try:
                    loop.run_forever()
                except KeyboardInterrupt:
                    pass
                finally:
                    loop.close()

            self.device_bridge_thread = threading.Thread(
                target=run_bridge_loop,
                daemon=True,
                name="DeviceVideoBridge"
            )
            self.device_bridge_thread.start()

            self.app.logger.info("Device Control: Video playback integration active")

            # Update existing tracker managers to use the shared device manager
            self._update_existing_tracker_managers()

            self.app.logger.info("Device Control: Creating DeviceParameterManager...")
            self.param_manager = DeviceParameterManager()
            self.app.logger.info("Device Control: DeviceParameterManager created successfully")

            # Initialize OSR profiles if not already present
            self._initialize_osr_profiles()

            # UI state already initialized in __init__

            # Initialize Live Preview Bridge (editing-time haptic feedback)
            try:
                from device_control.bridges.live_preview_bridge import LivePreviewBridge
                self.live_preview_bridge = LivePreviewBridge()
                self.app.logger.info("Device Control: Live Preview Bridge initialized")
            except Exception as e_preview:
                self.app.logger.debug(f"Live Preview Bridge init skipped: {e_preview}")
                self.live_preview_bridge = None

            self._device_control_initialized = True
            self.app.logger.info("Device Control initialized in Control Panel successfully")

        except Exception as e:
            self.app.logger.error(f"Failed to initialize Device Control: {e}")
            import traceback
            self.app.logger.error(f"Full traceback: {traceback.format_exc()}")
            self._device_control_init_error = e
            self._device_control_initialized = True  # Mark as attempted

    def _update_existing_tracker_managers(self):
        """Update existing TrackerManagers to use the shared device manager."""
        try:
            # Check if app has tracker managers
            found_any = False
            for timeline_id in range(1, 3):  # Timeline 1 and 2
                tracker_manager = getattr(self.app, f'tracker_manager_{timeline_id}', None)
                if tracker_manager:
                    found_any = True
                    self.app.logger.info(f"Updating TrackerManager {timeline_id} with shared device manager")
                    # Re-initialize the device bridge with shared device manager
                    tracker_manager._init_device_bridge()

                    # Also update live device control setting from current settings
                    live_tracking_enabled = self.app.app_settings.get("device_control_live_tracking", False)
                    if live_tracking_enabled:
                        tracker_manager.set_live_device_control_enabled(True)
                        self.app.logger.info(f"TrackerManager {timeline_id} live control enabled from settings")

            if not found_any:
                self.app.logger.info("No existing TrackerManagers found to update")

        except Exception as e:
            self.app.logger.warning(f"Failed to update existing tracker managers: {e}")
            import traceback
            self.app.logger.warning(f"Traceback: {traceback.format_exc()}")

    def _initialize_osr_profiles(self):
        """Initialize OSR profiles in app settings if not present."""
        try:
            from device_control.axis_control import DEFAULT_PROFILES, save_profile_to_settings

            # Check if profiles already exist
            existing_profiles = self.app.app_settings.get("device_control_osr_profiles", {})

            if not existing_profiles:
                self.app.logger.info("Initializing OSR profiles from defaults...")

                # Convert DEFAULT_PROFILES to settings format
                profiles_dict = {}
                for profile_name, profile_obj in DEFAULT_PROFILES.items():
                    profiles_dict[profile_name] = save_profile_to_settings(profile_obj)

                # Save to settings
                self.app.app_settings.set("device_control_osr_profiles", profiles_dict)

                # Set default selected profile if not set
                if not self.app.app_settings.get("device_control_selected_profile"):
                    self.app.app_settings.set("device_control_selected_profile", "Balanced")

                self.app.logger.info(f"Initialized {len(profiles_dict)} OSR profiles")
            else:
                self.app.logger.info(f"OSR profiles already initialized ({len(existing_profiles)} profiles)")

        except Exception as e:
            self.app.logger.error(f"Failed to initialize OSR profiles: {e}")
            import traceback
            self.app.logger.error(f"Traceback: {traceback.format_exc()}")

    def _get_connected_device_type(self):
        """Return the device type string for the currently connected device, or ''."""
        if not self.device_manager.is_connected():
            return ""
        for did in self.device_manager.connected_devices:
            return self.device_manager.get_device_type_for_id(did)
        return ""

    def _render_device_control_content(self):
        """Render the main device control interface with improved UX."""
        # Guard: skip rendering while background disconnect thread is tearing down state
        if getattr(self, '_device_disconnecting', False):
            imgui.text("Disconnecting device...")
            return
        # Version info (top of tab, consistent with other supporter modules)
        self._render_addon_version_label("device_control", "Device Control")

        # Compact connection status (always visible, no card needed)
        self._render_compact_connection_status()

        imgui.separator()

        _conn_type = self._get_connected_device_type()

        # Quick controls when connected
        if _conn_type:
            with _section_card("Quick Controls##QuickCtrl", tier="primary") as is_open:
                if is_open:
                    self._render_quick_controls()

        # Device type sections
        _osr_open = _conn_type == "osr" or not _conn_type
        with _section_card("OSR2/OSR6 (USB)##OSRDevices", tier="primary", open_by_default=_osr_open) as is_open:
            if is_open:
                self._render_osr_controls()

        _bp_open = _conn_type in ("buttplug_linear", "buttplug_vibrator")
        with _section_card("Buttplug.io (Universal)##ButtplugDevices", tier="primary", open_by_default=_bp_open) as is_open:
            if is_open:
                self._render_buttplug_controls()

        _handy_open = _conn_type == "handy"
        with _section_card("Handy : Direct / Streaming##HandyDirect", tier="primary", open_by_default=_handy_open) as is_open:
            if is_open:
                self._render_handy_controls()

        _ossm_open = _conn_type == "ossm"
        with _section_card("OSSM (Bluetooth)##OSSMDevices", tier="primary", open_by_default=_ossm_open) as is_open:
            if is_open:
                self._render_ossm_controls()

        # Axis Configuration (shown when connected)
        if _conn_type:
            with _section_card("Axis Configuration##AxisConfig", tier="primary") as is_open:
                if is_open:
                    self._render_axis_configuration()

        # Advanced Settings
        if _conn_type:
            with _section_card("Advanced Settings##DeviceAdvancedAll", tier="secondary", open_by_default=False) as is_open:
                if is_open:
                    self._render_all_advanced_settings()

    def _render_compact_connection_status(self):
        """Render compact connection status (always visible)."""
        if self.device_manager.is_connected():
            device_name = self.device_manager.get_connected_device_name()
            control_source = self.device_manager.get_active_control_source()

            # Status line with color indicator
            if control_source == 'streamer':
                imgui.text_colored("[STREAMER]", 0.2, 0.5, 0.9)  # Blue
                imgui.same_line()
                imgui.text(f"{device_name}")
            elif control_source == 'desktop':
                imgui.text_colored("[DESKTOP]", 0.2, 0.7, 0.2)  # Green
                imgui.same_line()
                imgui.text(f"{device_name}")
            else:
                imgui.text_colored("[IDLE]", 0.7, 0.7, 0.2)  # Yellow
                imgui.same_line()
                imgui.text(f"{device_name}")

            if imgui.is_item_hovered():
                imgui.set_tooltip("Blue = Streamer Control | Green = Desktop Control | Yellow = Idle")

            imgui.same_line()
            if imgui.small_button("Disconnect"):
                self._disconnect_current_device()
        else:
            imgui.text_colored("Device: Not Connected", 0.7, 0.3, 0.3)

    def _render_quick_controls(self):
        """Render quick controls for connected device (always visible when connected)."""
        imgui.text("Quick Controls:")
        imgui.spacing()

        # Global stroke range — device-agnostic via AxisRoutes (0-100%)
        imgui.text("Global Range (All Active Axes):")

        router = self.device_manager.axis_router
        # Collect enabled routes from all connected devices
        enabled_routes = []
        connected_device_id = None
        is_osr = False
        for device_id, _backend in self.device_manager.connected_devices.items():
            connected_device_id = device_id
            device_type = self.device_manager.get_device_type_for_id(device_id)
            is_osr = device_type == "osr"
            config = router.get_config(device_id)
            if config:
                for _ch, route in config.get_enabled_routes().items():
                    enabled_routes.append(route)

        if enabled_routes:
            avg_min = sum(getattr(r, 'min_value', 0.0) for r in enabled_routes) / len(enabled_routes)
            avg_max = sum(getattr(r, 'max_value', 100.0) for r in enabled_routes) / len(enabled_routes)

            changed_min, new_min = imgui.slider_float("Global Min %##GlobalMin", avg_min, 0.0, 50.0, "%.0f%%")
            if changed_min:
                for route in enabled_routes:
                    if hasattr(route, 'min_value'):
                        route.min_value = new_min
                router.save_to_settings(self.app.app_settings)
                if is_osr and connected_device_id:
                    self._save_routes_to_osr_profile(connected_device_id)
                self.device_manager.update_position(new_min, 50.0)

            changed_max, new_max = imgui.slider_float("Global Max %##GlobalMax", avg_max, 50.0, 100.0, "%.0f%%")
            if changed_max:
                for route in enabled_routes:
                    if hasattr(route, 'max_value'):
                        route.max_value = new_max
                router.save_to_settings(self.app.app_settings)
                if is_osr and connected_device_id:
                    self._save_routes_to_osr_profile(connected_device_id)
                self.device_manager.update_position(new_max, 50.0)

            _tooltip_if_hovered("Adjust min/max for all active axes at once. Drag to feel the limits in real-time.")
        else:
            imgui.text_colored("No active axes configured", 0.7, 0.5, 0.0)

        imgui.spacing()

        # Bookmark navigation
        self._render_bookmark_nav()

        imgui.spacing()

        # Quick position test
        imgui.text("Test Position:")
        current_pos = self.device_manager.current_position
        changed, new_pos = imgui.slider_float("##QuickTestPos", current_pos, 0.0, 100.0, "%.1f%%")
        if changed:
            self.device_manager.update_position(new_pos, 50.0)
        _tooltip_if_hovered("Drag to test device movement")



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

            import threading
            def run_preview():
                import asyncio
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

            import threading
            def run_demo():
                import asyncio
                import time
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
            import threading
            import math
            import random

            def run_pattern():
                import asyncio
                import time
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

    def _render_all_advanced_settings(self):
        """Render all advanced settings in one section."""
        # Performance Settings
        imgui.text_colored("Performance:", 0.8, 0.8, 0.2)
        config = self.device_manager.config

        changed, new_rate = imgui.slider_float("Update Rate##DeviceRate", config.max_position_rate_hz, 1.0, 120.0, "%.1f Hz")
        if changed:
            config.max_position_rate_hz = new_rate
        _tooltip_if_hovered("How often device position is updated per second")

        changed, new_smoothing = imgui.slider_float("Smoothing##DeviceSmooth", config.position_smoothing, 0.0, 1.0, "%.2f")
        if changed:
            config.position_smoothing = new_smoothing
        _tooltip_if_hovered("Smooths position changes (0=no smoothing, 1=maximum smoothing)")

        changed, new_latency = imgui.slider_int("Latency Comp.##DeviceLatency", config.latency_compensation_ms, 0, 200, "%d ms")
        if changed:
            config.latency_compensation_ms = new_latency
        _tooltip_if_hovered("Compensates for device response delay")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Integration Settings
        imgui.text_colored("Integration:", 0.8, 0.8, 0.2)

        live_tracking_enabled = self.app.app_settings.get("device_control_live_tracking", False)
        changed, new_live_tracking = imgui.checkbox("Live Tracking Control##DeviceLiveTracking", live_tracking_enabled)
        if changed:
            self.app.app_settings.set("device_control_live_tracking", new_live_tracking)
            self.app.app_settings.save_settings()
            self._update_live_tracking_control(new_live_tracking)
        _tooltip_if_hovered("Stream live tracker data directly to device in real-time")

        video_playback_enabled = self.app.app_settings.get("device_control_video_playback", False)
        changed, new_video_playback = imgui.checkbox("Video Playback Control##DeviceVideoPlayback", video_playback_enabled)
        if changed:
            self.app.app_settings.set("device_control_video_playback", new_video_playback)
            self.app.app_settings.save_settings()
            self._update_video_playback_control(new_video_playback)
        _tooltip_if_hovered("Sync device with video timeline and funscript playback")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Speed Limiting
        imgui.text_colored("Speed Limiting:", 0.8, 0.8, 0.2)

        speed_limit_enabled = getattr(self.device_manager, '_speed_limit_enabled', False)
        changed, new_enabled = imgui.checkbox("Enable Speed Limit##SpeedLimit", speed_limit_enabled)
        if changed:
            self.device_manager._speed_limit_enabled = new_enabled
        _tooltip_if_hovered("Limit maximum device movement speed to prevent dangerous acceleration")

        if speed_limit_enabled:
            max_speed = getattr(self.device_manager, '_max_speed_pct_per_second', 400.0)
            changed, new_speed = imgui.slider_float("Max Speed##SpeedLimitVal", max_speed, 50.0, 500.0, "%.0f %%/s")
            if changed:
                self.device_manager._max_speed_pct_per_second = new_speed
            _tooltip_if_hovered("Maximum position change per second (400%%/s = full stroke in 0.25s)")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Interpolation Mode
        imgui.text_colored("Interpolation:", 0.8, 0.8, 0.2)

        try:
            from device_control.bridges.funscript_player_bridge import InterpolationMode
            mode_names = ["Linear", "Cosine", "PCHIP (Recommended)", "Step"]
            mode_values = [InterpolationMode.LINEAR, InterpolationMode.COSINE,
                           InterpolationMode.PCHIP, InterpolationMode.STEP]

            # Get current mode from video bridge config
            bridge = getattr(self, 'device_video_bridge', None)
            current_mode = InterpolationMode.PCHIP
            if bridge:
                current_mode = getattr(bridge.config, 'interpolation_mode', InterpolationMode.PCHIP)

            current_idx = mode_values.index(current_mode) if current_mode in mode_values else 2
            changed, new_idx = imgui.combo("##InterpMode", current_idx, mode_names)
            if changed and bridge:
                bridge.config.interpolation_mode = mode_values[new_idx]
            _tooltip_if_hovered(
                "LINEAR: Simple straight-line interpolation\n"
                "COSINE: Smoother acceleration/deceleration\n"
                "PCHIP: Best quality, prevents overshoot\n"
                "STEP: No interpolation, snap to keyframes"
            )
        except Exception:
            pass

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Auto-Home Settings
        imgui.text_colored("Auto-Home:", 0.8, 0.8, 0.2)

        auto_home_enabled = getattr(self.device_manager, '_auto_home_enabled', True)
        changed, new_enabled = imgui.checkbox("Enable Auto-Home##AutoHome", auto_home_enabled)
        if changed:
            self.device_manager._auto_home_enabled = new_enabled
        _tooltip_if_hovered("Return device to center position after idle period")

        if auto_home_enabled:
            auto_home_delay = getattr(self.device_manager, '_auto_home_delay_s', 5.0)
            changed, new_delay = imgui.slider_float("Idle Delay##AutoHomeDelay", auto_home_delay, 1.0, 30.0, "%.1f s")
            if changed:
                self.device_manager._auto_home_delay_s = new_delay
            _tooltip_if_hovered("How long to wait after last movement before homing starts")

            auto_home_duration = getattr(self.device_manager, '_auto_home_duration_s', 3.0)
            changed, new_duration = imgui.slider_float("Home Duration##AutoHomeDur", auto_home_duration, 0.5, 10.0, "%.1f s")
            if changed:
                self.device_manager._auto_home_duration_s = new_duration
            _tooltip_if_hovered("How long the homing transition takes (ease-in curve)")

        # OSR-specific performance settings (only when OSR connected)
        if self._get_connected_device_type() == "osr":
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            self._render_osr_performance_settings()

        imgui.spacing()

    def _render_osr_controls(self):
        """Render OSR device controls."""
        # Check OSR connection status
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        is_osr_connected = self._get_connected_device_type() == "osr"

        if is_osr_connected:
            self._status_indicator(f"Connected to {connected_device.device_id}", "ready", "OSR device connected and ready")

            # Uniform action row
            imgui.spacing()
            if imgui.button("Test Movement##OSR"):
                self._test_osr_movement()
            _tooltip_if_hovered("Test OSR device with predefined movement sequence")
            imgui.same_line()
            with destructive_button_style():
                if imgui.button("Disconnect##OSRDisconnect"):
                    self._disconnect_current_device()
            _tooltip_if_hovered("Disconnect OSR device")

        else:
            imgui.text("Connect your OSR2/OSR6 device via USB cable.")

            imgui.separator()
            if imgui.button("Scan for OSR Devices##OSRScan"):
                self._scan_osr_devices()
            _tooltip_if_hovered("Search for connected OSR devices on serial ports")

            # Show available ports
            if self._available_osr_ports:
                imgui.spacing()
                imgui.text("Available devices:")
                for port_info in self._available_osr_ports:
                    port_name = port_info.get('device', 'Unknown')
                    description = port_info.get('description', 'No description')

                    with primary_button_style():
                        if imgui.button(f"Connect##OSR_{port_name}"):
                            self._connect_osr_device(port_name)
                    imgui.same_line()
                    imgui.text(f"{port_name} ({description})")

            elif self._osr_scan_performed:
                imgui.spacing()
                self._status_indicator("No OSR devices found", "warning", "Try troubleshooting steps below")
                imgui.text("Troubleshooting:")
                imgui.bullet_text("Ensure OSR2/OSR6 is connected via USB")
                imgui.bullet_text("Check device is powered on")
                imgui.bullet_text("Try different USB cable or port")

    def _render_handy_controls(self):
        """Render Handy direct API controls."""

        # Check Handy connection status
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        is_handy_connected = self._get_connected_device_type() == "handy"

        if is_handy_connected:
            # Connected state + measured RTD
            rtd_ms = self.device_manager.get_handy_rtd_ms() if hasattr(self.device_manager, 'get_handy_rtd_ms') else 0
            rtd_label = f"  (RTD: {rtd_ms}ms)" if rtd_ms > 0 else ""
            self._status_indicator(f"Connected to {connected_device.name}{rtd_label}", "ready", "Handy connected and ready")

            # Mode selector (HDSP vs HSSP)
            handy_mode = self.app.app_settings.get("device_control_handy_mode", "HSSP (Script Sync)")
            mode_options = ["HSSP (Script Sync)", "HDSP (Experimental)"]
            mode_idx = mode_options.index(handy_mode) if handy_mode in mode_options else 0
            imgui.text("Mode:")
            imgui.same_line()
            avail_w = imgui.get_content_region_available()[0]
            imgui.push_item_width(avail_w)
            changed_mode, new_mode_idx = imgui.combo("##HandyMode", mode_idx, mode_options)
            imgui.pop_item_width()
            if changed_mode:
                self.app.app_settings.set("device_control_handy_mode", mode_options[new_mode_idx])
                self.app.app_settings.save_settings()
                # Reset preparation state so the new mode starts fresh
                if hasattr(self, 'device_video_integration') and self.device_video_integration:
                    self.device_video_integration.reset_handy_preparation()
            _tooltip_if_hovered("HSSP: uploads script to device (recommended)\nHDSP: sends positions in real-time (experimental)\n\nNote: HDSP plays existing funscripts only.\nLive tracking is not supported in HDSP mode.")

            is_hdsp_mode = mode_options[new_mode_idx if changed_mode else mode_idx].startswith("HDSP")

            imgui.spacing()

            # Upload only applies to direct Handy (HSSP), not BT/Intiface (Buttplug)
            is_direct_handy = getattr(self.device_manager, '_handy_backend', None) is not None

            # Upload Funscript button (auto-uploads on play, but manual option if script changed)
            has_funscript = (hasattr(self.app, 'funscript_processor') and
                           self.app.funscript_processor and
                           self.app.funscript_processor.get_actions('primary'))

            if is_direct_handy and not is_hdsp_mode:
                if has_funscript:
                    # Timeline selector for upload (Handy is single-axis, let user choose which)
                    axis_assignments = {}
                    funscript_obj = self.app.funscript_processor.get_funscript_obj()
                    if funscript_obj and hasattr(funscript_obj, 'get_axis_assignments'):
                        axis_assignments = funscript_obj.get_axis_assignments()

                    # Build list of timelines that have actions
                    upload_timelines = []
                    upload_labels = []
                    for tl_num, axis_name in sorted(axis_assignments.items()):
                        actions = funscript_obj.get_axis_actions(
                            'primary' if tl_num == 1 else ('secondary' if tl_num == 2 else axis_name))
                        if actions:
                            label = f"Timeline {tl_num} ({axis_name})"
                            # Show upload indicator (uses revision counter for reliable change detection)
                            uploaded_tls = getattr(self, '_handy_uploaded_timelines', {})
                            current_hash = getattr(self.app.funscript_processor, '_revision', 0)
                            if tl_num in uploaded_tls and uploaded_tls[tl_num] == current_hash:
                                label += " [uploaded]"
                            upload_timelines.append(tl_num)
                            upload_labels.append(label)

                    if not upload_timelines:
                        upload_timelines = [1]
                        upload_labels = ["Timeline 1 (stroke)"]

                    # Timeline combo
                    selected_tl_idx = getattr(self, '_handy_upload_tl_idx', 0)
                    if selected_tl_idx >= len(upload_labels):
                        selected_tl_idx = 0
                    imgui.text("Upload timeline:")
                    imgui.same_line()
                    avail_w = imgui.get_content_region_available()[0]
                    imgui.push_item_width(avail_w)
                    changed_tl, selected_tl_idx = imgui.combo(
                        "##HandyUploadTimeline", selected_tl_idx, upload_labels)
                    imgui.pop_item_width()
                    self._handy_upload_tl_idx = selected_tl_idx

                    # Get selected timeline's actions for hash check
                    selected_tl_num = upload_timelines[selected_tl_idx] if upload_timelines else 1
                    current_actions = self.app.funscript_processor.get_actions(
                        'primary' if selected_tl_num == 1 else ('secondary' if selected_tl_num == 2
                        else axis_assignments.get(selected_tl_num, 'primary')))
                    current_hash = getattr(self.app.funscript_processor, '_revision', 0)

                    # Stale-script detection
                    uploaded_tls = getattr(self, '_handy_uploaded_timelines', {})
                    last_hash = uploaded_tls.get(selected_tl_num)
                    script_changed = last_hash is not None and current_hash != last_hash

                    if script_changed:
                        imgui.text_colored("Funscript modified since last upload", 1.0, 0.7, 0.0, 1.0)

                        # Auto re-upload on play: invalidate prepared state
                        auto_reupload = self.app.app_settings.get("handy_auto_reupload_on_play", True)
                        if auto_reupload and self.device_manager.has_prepared_handy_devices():
                            self.device_manager.reset_handy_streaming_state()

                    # Auto re-upload toggle
                    auto_reupload = self.app.app_settings.get("handy_auto_reupload_on_play", True)
                    ch, new_auto = imgui.checkbox("Auto upload on Play##HandyAutoUpload", auto_reupload)
                    if ch:
                        self.app.app_settings.set("handy_auto_reupload_on_play", new_auto)
                    _tooltip_if_hovered("Automatically re-upload funscript when pressing Play if it changed since last upload")

                    btn_label = "Re-upload Funscript##HandyUpload" if last_hash is not None else "Upload Funscript##HandyUpload"
                    if imgui.button(btn_label, width=-1):
                        self._upload_funscript_to_handy(timeline_num=selected_tl_num)
                    _tooltip_if_hovered("Upload selected timeline's funscript to Handy for HSSP playback")
                else:
                    imgui.text_colored("No funscript loaded", 0.7, 0.5, 0.0)
                    _tooltip_if_hovered("Load a funscript first")

            imgui.spacing()

            # Disconnect button
            with destructive_button_style():
                if imgui.button("Disconnect##HandyDisconnect"):
                    self._disconnect_handy()
            _tooltip_if_hovered("Disconnect from Handy device")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Sync settings (HSSP only — not applicable to HDSP mode)
            if not is_hdsp_mode:
                imgui.text("Sync Offset:")
                imgui.same_line()
                current_offset = self.app.app_settings.get("device_control_handy_sync_offset_ms", 0)
                imgui.push_item_width(-1)
                changed, new_offset = imgui.drag_int(
                    "##HandySyncOffset", current_offset, 1.0, 0, 2500, "%d ms"
                )
                imgui.pop_item_width()
                if changed:
                    self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                    self._apply_handy_hstp_offset(new_offset)
                _tooltip_if_hovered(
                    "Drag to adjust, Ctrl+Click for direct input.\n"
                    "Higher = device moves earlier to compensate lag.\n"
                    "Network latency is auto-compensated; adjust if movement feels late."
                )

        else:
            # Disconnected state - show connection controls
            imgui.text("Enter your Handy connection key:")

            # Connection key input
            connection_key = self.app.app_settings.get("handy_connection_key", "")
            changed, new_key = imgui.input_text(
                "##HandyConnectionKey",
                connection_key,
                256
            )
            if changed:
                self.app.app_settings.set("handy_connection_key", new_key)
            _tooltip_if_hovered("Your Handy connection key (e.g., 'DH7Hc')")

            imgui.spacing()

            # Connect button (PRIMARY - positive action)
            if connection_key and len(connection_key) > 0:
                with primary_button_style():
                    if imgui.button("Connect to Handy##HandyConnect"):
                        self._connect_handy(connection_key)
                _tooltip_if_hovered("Connect to your Handy device")
            else:
                imgui.text_disabled("Enter connection key to enable connect button")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Help text
            imgui.text("How to get your connection key:")
            imgui.indent(10)
            imgui.bullet_text("Open the Handy app")
            imgui.bullet_text("Go to Settings > Connection")
            imgui.bullet_text("Copy the connection key")
            imgui.unindent(10)

            imgui.spacing()

            # Advanced settings even when disconnected
            if imgui.collapsing_header("Advanced Settings##HandyAdvanced")[0]:
                imgui.indent(10)

                # Minimum interval setting
                changed, value = imgui.slider_int(
                    "Min Command Interval (ms)##HandyMinIntervalAdv",
                    self.app.app_settings.get("handy_min_interval", 60),
                    20, 200
                )
                if changed:
                    self.app.app_settings.set("handy_min_interval", value)
                _tooltip_if_hovered("Minimum time between position commands (60ms recommended)")

                imgui.unindent(10)

    def _render_buttplug_controls(self):
        """Render Buttplug.io device controls."""

        # Check Buttplug connection status
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        is_buttplug_connected = self._get_connected_device_type() in ("buttplug_linear", "buttplug_vibrator")

        if is_buttplug_connected:
            self._status_indicator(f"Connected to {connected_device.name}", "ready", "Buttplug device connected and ready")

            # Device capabilities
            caps = getattr(connected_device, 'capabilities', None)
            if caps:
                imgui.text("Device capabilities:")
                imgui.indent(10)
                if caps.supports_linear:
                    imgui.bullet_text(f"Linear motion: {caps.linear_channels} axis")
                if caps.supports_vibration:
                    imgui.bullet_text(f"Vibration: {caps.vibration_channels} motors")
                if caps.supports_rotation:
                    imgui.bullet_text(f"Rotation: {caps.rotation_channels} axis")
                imgui.bullet_text(f"Update rate: {caps.max_position_rate_hz} Hz")
                imgui.unindent(10)

            # BLE Latency Compensation
            if caps and caps.supports_linear:
                imgui.spacing()
                imgui.separator()
                imgui.text("BLE Latency Compensation")
                _tooltip_if_hovered(
                    "Compensates for Bluetooth transport delay.\n"
                    "Higher values make the device move earlier/faster to stay in sync.\n"
                    "Start at 100ms and adjust based on your setup."
                )
                current_ble_comp = self.app.app_settings.get("ble_latency_compensation_ms", 100)
                changed, new_ble_comp = imgui.slider_int(
                    "##BLELatencyComp", current_ble_comp, 0, 300, f"{current_ble_comp} ms"
                )
                if changed:
                    self.app.app_settings.set("ble_latency_compensation_ms", new_ble_comp)
                    if hasattr(self.device_manager, 'update_ble_latency_compensation'):
                        self.device_manager.update_ble_latency_compensation(new_ble_comp)

            # Uniform action row
            imgui.spacing()
            if imgui.button("Test Movement##Buttplug"):
                self._test_buttplug_movement()
            _tooltip_if_hovered("Test device with predefined movement sequence")
            imgui.same_line()
            with destructive_button_style():
                if imgui.button("Disconnect##ButtplugDisconnect"):
                    self._disconnect_current_device()
            _tooltip_if_hovered("Disconnect Buttplug device")

        else:
            imgui.text("Connect devices via Intiface Central")
            imgui.text("Supports 100+ devices: Handy, Lovense, Kiiroo, OSR2, and more")

            # Server configuration
            if imgui.collapsing_header("Buttplug Server Configuration##ButtplugServer")[0]:
                imgui.indent(10)

                # Server address
                current_address = self.app.app_settings.get("buttplug_server_address", "localhost")
                changed, new_address = imgui.input_text("Server Address##ButtplugAddr", current_address, 256)
                if changed:
                    self.app.app_settings.set("buttplug_server_address", new_address)
                _tooltip_if_hovered("IP address or hostname of Intiface Central server")

                # Server port
                current_port = self.app.app_settings.get("buttplug_server_port", 12345)
                changed, new_port = imgui.input_int("Port##ButtplugPort", current_port)
                if changed and 1024 <= new_port <= 65535:
                    self.app.app_settings.set("buttplug_server_port", new_port)
                _tooltip_if_hovered("WebSocket port (default: 12345)")

                imgui.unindent(10)

            imgui.separator()
            with primary_button_style():
                if imgui.button("Discover Devices##ButtplugDiscover"):
                    self._discover_buttplug_devices()
            _tooltip_if_hovered("Search for devices through Intiface Central")

            imgui.same_line()
            if imgui.button("Check Server##ButtplugStatus"):
                self._check_buttplug_server_status()
            _tooltip_if_hovered("Test connection to Intiface Central server")

            # Show discovered devices
            if hasattr(self, '_discovered_buttplug_devices') and self._discovered_buttplug_devices:
                imgui.spacing()
                imgui.text(f"Found {len(self._discovered_buttplug_devices)} device(s):")

                for i, device_info in enumerate(self._discovered_buttplug_devices):
                    with primary_button_style():
                        if imgui.button(f"Connect##buttplug_{i}"):
                            self._connect_specific_buttplug_device(device_info.device_id)
                    imgui.same_line()
                    imgui.text(f"{device_info.name} ({device_info.device_type.name})")

            elif hasattr(self, '_buttplug_discovery_performed') and self._buttplug_discovery_performed:
                imgui.spacing()
                self._status_indicator("No devices found", "warning", "Check troubleshooting steps below")
                imgui.text("Troubleshooting:")
                imgui.bullet_text("Start Intiface Central application")
                imgui.bullet_text("Enable Server Mode in Intiface")
                imgui.bullet_text("Connect and pair your devices")

    def _disconnect_current_device(self):
        """Disconnect the currently connected device."""
        try:
            import threading
            import asyncio

            self._device_disconnecting = True

            def run_disconnect():
                try:
                    # Try to use existing event loop first
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Schedule disconnect in the existing loop
                            future = asyncio.run_coroutine_threadsafe(self.device_manager.stop(), loop)
                            future.result(timeout=10)  # Wait up to 10 seconds
                        else:
                            # Use the existing loop if not running
                            loop.run_until_complete(self.device_manager.stop())
                    except RuntimeError:
                        # No event loop exists, create a new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(self.device_manager.stop())
                        finally:
                            loop.close()

                    self.app.logger.info("Device disconnected successfully")
                except Exception as e:
                    self.app.logger.error(f"Error during disconnect: {e}")
                finally:
                    self._device_disconnecting = False

            thread = threading.Thread(target=run_disconnect, daemon=True)
            thread.start()
        except Exception as e:
            self._device_disconnecting = False
            self.app.logger.error(f"Failed to disconnect device: {e}")

    def _scan_osr_devices(self):
        """Scan for OSR devices specifically."""
        try:
            import threading
            def run_osr_scan():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Get OSR backend and scan
                    osr_backend = self.device_manager.available_backends.get('osr')
                    if osr_backend:
                        devices = loop.run_until_complete(osr_backend.discover_devices())
                        # Convert to simple format for UI
                        self._available_osr_ports = []
                        for device in devices:
                            self._available_osr_ports.append({
                                'device': device.device_id,
                                'description': device.name,
                                'manufacturer': getattr(device, 'manufacturer', 'Unknown')
                            })
                        self.app.logger.info(f"Found {len(devices)} potential OSR devices")
                        self._osr_scan_performed = True
                finally:
                    loop.close()

            thread = threading.Thread(target=run_osr_scan, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to scan OSR devices: {e}")

    def _connect_osr_device(self, port_name):
        """Connect to specific OSR device."""
        try:
            import threading
            import asyncio

            def run_osr_connect_and_loop():
                """Connect to OSR device and keep the async loop running."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def connect_and_run():
                    try:
                        success = await self.device_manager.connect(port_name)
                        if success:
                            self.app.logger.info(f"Connected to OSR device on {port_name}")
                            self.app.logger.info("Async loop running for device control - keeping alive for live tracking")

                            # Keep the loop running forever to maintain the position update task
                            # This will only end when the application shuts down
                            try:
                                while True:
                                    await asyncio.sleep(1)  # Keep loop alive
                            except asyncio.CancelledError:
                                self.app.logger.info("Device manager loop cancelled")
                        else:
                            self.app.logger.error(
                                f"Failed to connect to OSR device on {port_name} - is it plugged in?",
                                extra={'status_message': True, 'duration': 5.0})
                    except Exception as e:
                        self.app.logger.error(
                            f"OSR connection error: {e}",
                            extra={'status_message': True, 'duration': 5.0})

                try:
                    # Store loop reference for potential cleanup
                    self.device_manager.loop = loop
                    loop.run_until_complete(connect_and_run())
                finally:
                    loop.close()

            # Start the persistent connection thread
            thread = threading.Thread(target=run_osr_connect_and_loop, daemon=True)
            thread.start()

        except Exception as e:
            self.app.logger.error(f"Failed to connect OSR device: {e}")

    def _render_osr_performance_settings(self):
        """Render OSR performance tuning controls."""
        try:
            imgui.separator()
            imgui.text("Performance Settings:")

            # Get current settings or defaults
            sensitivity = self.app.app_settings.get("osr_sensitivity", 2.0)
            speed = self.app.app_settings.get("osr_speed", 2.0)

            # Sensitivity slider
            imgui.text("Sensitivity (how small movements trigger device):")
            changed_sens, new_sensitivity = imgui.slider_float("##osr_sensitivity", sensitivity, 0.5, 5.0, "%.1fx")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Higher = more responsive to small position changes\nLower = only responds to large movements")

            if changed_sens:
                self.app.app_settings.set("osr_sensitivity", new_sensitivity)
                self._update_osr_performance(new_sensitivity, speed)

            # Speed slider
            imgui.text("Speed (how fast the device moves):")
            changed_speed, new_speed = imgui.slider_float("##osr_speed", speed, 0.5, 5.0, "%.1fx")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Higher = faster movements\nLower = slower, smoother movements")

            if changed_speed:
                self.app.app_settings.set("osr_speed", new_speed)
                self._update_osr_performance(sensitivity, new_speed)

            # Video playback amplification
            imgui.separator()
            imgui.text("Video Playback Amplification:")
            video_amp = self.app.app_settings.get("video_playback_amplification", 1.5)
            changed_amp, new_amp = imgui.slider_float("##video_amp", video_amp, 1.0, 3.0, "%.1fx")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Amplifies funscript movement during video playback\nHigher = more dramatic movement\n1.0x = original funscript range")

            if changed_amp:
                self.app.app_settings.set("video_playback_amplification", new_amp)
                self.app.logger.info(f"Video playback amplification set to {new_amp:.1f}x")

            # Reset button
            if imgui.button("Reset to Defaults##OSR_Performance"):
                self.app.app_settings.set("osr_sensitivity", 2.0)
                self.app.app_settings.set("osr_speed", 2.0)
                self.app.app_settings.set("video_playback_amplification", 1.5)
                self._update_osr_performance(2.0, 2.0)

        except Exception as e:
            self.app.logger.error(f"Error rendering OSR performance settings: {e}")

    def _update_osr_performance(self, sensitivity: float, speed: float):
        """Update OSR device performance settings."""
        try:
            # Get the OSR backend
            osr_backend = self.device_manager.available_backends.get('osr')
            if osr_backend and hasattr(osr_backend, 'set_performance_settings'):
                osr_backend.set_performance_settings(sensitivity, speed)
                self.app.logger.info(f"Updated OSR performance: sensitivity={sensitivity:.1f}x, speed={speed:.1f}x")
            else:
                self.app.logger.debug("OSR backend not available for performance update")

        except Exception as e:
            self.app.logger.error(f"Failed to update OSR performance: {e}")

    def _test_osr_movement(self):
        """Test OSR movement with a simple pattern."""
        try:
            import threading
            def run_test():
                import asyncio
                import time
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Check device manager state
                    if not self.device_manager:
                        self.app.logger.error("Device manager not initialized")
                        return

                    # Check if any device is connected
                    if not self.device_manager.is_connected():
                        self.app.logger.error("No device connected. Please connect an OSR device first.")
                        return

                    backend = self.device_manager.get_connected_backend()
                    if not backend:
                        self.app.logger.error("No connected backend available")
                        return

                    # Check if the backend is actually connected
                    if not backend.is_connected():
                        self.app.logger.error("Backend reports not connected")
                        return

                    self.app.logger.info("Starting OSR test movement pattern...")
                    self.app.logger.info(f"Using backend: {type(backend).__name__}")

                    # Test pattern: center -> up -> center -> down -> center
                    test_positions = [
                        (50, "Center"),
                        (10, "Up"),
                        (50, "Center"),
                        (90, "Down"),
                        (50, "Center")
                    ]

                    for position, label in test_positions:
                        # Use the correct backend method
                        self.app.logger.info(f"Sending {label} position ({position}%) to device...")
                        success = loop.run_until_complete(backend.set_position(position, 50))
                        if success:
                            self.app.logger.debug(f"OSR test: {label} position ({position}%) - Success")
                        else:
                            self.app.logger.error(f"\u274c OSR test: {label} position ({position}%) - Failed")
                        time.sleep(1.0)  # Hold position for 1 second

                    self.app.logger.info("OSR test movement completed")

                except Exception as e:
                    self.app.logger.error(f"Error during OSR test: {e}")
                finally:
                    loop.close()

            thread = threading.Thread(target=run_test, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to start OSR test movement: {e}")

    def _update_live_tracking_control(self, enabled: bool):
        """Update live tracking control setting in tracker manager."""
        try:
            # Get tracker manager from app
            tracker_manager = getattr(self.app, 'tracker_manager', None)
            self.app.logger.info(f"Updating live tracking control: enabled={enabled}, tracker_manager={tracker_manager is not None}")

            if tracker_manager and hasattr(tracker_manager, 'set_live_device_control_enabled'):
                tracker_manager.set_live_device_control_enabled(enabled)
                self.app.logger.info(f"Live tracking device control {'enabled' if enabled else 'disabled'}")
            else:
                self.app.logger.warning(f"Tracker manager not available for live device control: {tracker_manager}")

                # Try to find tracker managers by timeline ID
                for timeline_id in range(1, 3):
                    tm = getattr(self.app, f'tracker_manager_{timeline_id}', None)
                    if tm:
                        self.app.logger.info(f"Found tracker_manager_{timeline_id}, updating...")
                        tm.set_live_device_control_enabled(enabled)

        except Exception as e:
            self.app.logger.error(f"Failed to update live tracking control: {e}")
            import traceback
            self.app.logger.error(f"Traceback: {traceback.format_exc()}")

    def _update_video_playback_control(self, enabled: bool):
        """Update video playback control setting."""
        try:
            # Setting is automatically picked up by timeline during video playback
            self.app.logger.info(f"Video playback device control {'enabled' if enabled else 'disabled'}")

            if enabled:
                # Verify device manager is available
                device_manager = getattr(self.app, 'device_manager', None)
                if device_manager and device_manager.is_connected():
                    self.app.logger.info("Device control ready for video playback")
                else:
                    self.app.logger.warning("No connected devices - video playback control will be inactive")

        except Exception as e:
            self.app.logger.error(f"Failed to update video playback control: {e}")

    def _load_osr_profile_to_device(self, profile_name: str, profile_data: dict):
        """Load OSR profile to the connected device."""
        try:
            # Import axis control here to avoid circular imports
            from device_control.axis_control import load_profile_from_settings

            # Convert settings to OSRControlProfile
            profile = load_profile_from_settings(profile_data)

            # Get the OSR backend and load the profile
            backend = self.device_manager.get_connected_backend()
            if backend and hasattr(backend, 'load_axis_profile'):
                success = backend.load_axis_profile(profile)
                if success:
                    self.app.logger.info(f"Loaded OSR profile '{profile_name}' to device")
                else:
                    self.app.logger.error(f"Failed to load OSR profile '{profile_name}' to device")

        except Exception as e:
            self.app.logger.error(f"Error loading OSR profile to device: {e}")

    def _open_intiface_download(self):
        """Open Intiface Central download page."""
        try:
            import webbrowser
            webbrowser.open("https://intiface.com/central/")
            self.app.logger.info("Opened Intiface Central download page")
        except Exception as e:
            self.app.logger.error(f"Failed to open Intiface download page: {e}")

    def _discover_buttplug_devices(self):
        """Discover available Buttplug devices using current server settings."""
        try:
            import threading
            def run_buttplug_discovery():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Use the device manager's existing Buttplug backend to ensure consistency
                    if self.device_manager and 'buttplug' in self.device_manager.available_backends:
                        backend = self.device_manager.available_backends['buttplug']
                        server_url = backend.server_address
                        self.app.logger.info(f"Discovering Buttplug devices at {server_url}...")
                        devices = loop.run_until_complete(backend.discover_devices())
                    else:
                        # Fallback: Create temporary backend for discovery
                        from device_control.backends.buttplug_backend_direct import DirectButtplugBackend

                        server_address = self.app.app_settings.get("buttplug_server_address", "localhost")
                        server_port = self.app.app_settings.get("buttplug_server_port", 12345)
                        server_url = f"ws://{server_address}:{server_port}"

                        self.app.logger.info(f"Discovering Buttplug devices at {server_url}...")
                        backend = DirectButtplugBackend(server_url)
                        devices = loop.run_until_complete(backend.discover_devices())

                    # Store discovered devices for UI display
                    self._discovered_buttplug_devices = devices
                    self._buttplug_discovery_performed = True

                    if devices:
                        self.app.logger.debug(f"Found {len(devices)} Buttplug device(s):")
                        for device in devices:
                            caps = []
                            if device.capabilities.supports_linear:
                                caps.append(f"Linear({device.capabilities.linear_channels}ch)")
                            if device.capabilities.supports_vibration:
                                caps.append(f"Vibration({device.capabilities.vibration_channels}ch)")
                            if device.capabilities.supports_rotation:
                                caps.append(f"Rotation({device.capabilities.rotation_channels}ch)")

                            self.app.logger.info(f"  \u2022 {device.name} - {', '.join(caps) if caps else 'No capabilities'}")
                    else:
                        self.app.logger.info("\u274c No Buttplug devices found")
                        self.app.logger.info("Make sure Intiface Central is running and devices are connected")

                except Exception as e:
                    self._buttplug_discovery_performed = True
                    if "Connection refused" in str(e) or "Connect call failed" in str(e):
                        self.app.logger.info(f"\u274c Cannot connect to Intiface Central at {server_url}")
                        self.app.logger.info("Please start Intiface Central and enable server mode")
                    else:
                        self.app.logger.error(f"Buttplug discovery error: {e}")
                finally:
                    loop.close()

            thread = threading.Thread(target=run_buttplug_discovery, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to start Buttplug discovery: {e}")

    def _connect_specific_buttplug_device(self, device_id):
        """Connect to a specific Buttplug device by ID."""
        try:
            import threading
            def run_buttplug_connection():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    success = loop.run_until_complete(self.device_manager.connect(device_id))
                    if success:
                        # Find the device name for logging
                        device_name = "Unknown Device"
                        if hasattr(self, '_discovered_buttplug_devices'):
                            for device in self._discovered_buttplug_devices:
                                if device.device_id == device_id:
                                    device_name = device.name
                                    break

                        self.app.logger.info(f"Connected to {device_name}")
                    else:
                        self.app.logger.error(f"\u274c Failed to connect to device {device_id}")

                except Exception as e:
                    self.app.logger.error(f"Buttplug connection failed: {e}")
                finally:
                    loop.close()

            thread = threading.Thread(target=run_buttplug_connection, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to connect to Buttplug device: {e}")

    def _check_buttplug_server_status(self):
        """Check if Buttplug server is running at configured address/port."""
        try:
            import threading
            def run_status_check():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def check_server():
                    try:
                        server_address = self.app.app_settings.get("buttplug_server_address", "localhost")
                        server_port = self.app.app_settings.get("buttplug_server_port", 12345)
                        server_url = f"ws://{server_address}:{server_port}"

                        # Try to connect briefly to check status
                        try:
                            import websockets
                            import json

                            import asyncio
                            websocket = await asyncio.wait_for(
                                websockets.connect(server_url), timeout=5
                            )

                            # Send handshake
                            handshake = {
                                "RequestServerInfo": {
                                    "Id": 1,
                                    "ClientName": "VR-Funscript-AI-Generator-StatusCheck",
                                    "MessageVersion": 3
                                }
                            }

                            await websocket.send(json.dumps([handshake]))
                            response = await websocket.recv()
                            response_data = json.loads(response)

                            await websocket.close()

                            if response_data and len(response_data) > 0 and 'ServerInfo' in response_data[0]:
                                server_info = response_data[0]['ServerInfo']
                                server_name = server_info.get('ServerName', 'Unknown')
                                server_version = server_info.get('MessageVersion', 'Unknown')

                                self.app.logger.debug(f"Buttplug server running at {server_url}")
                                self.app.logger.info(f"   Server: {server_name} (Protocol v{server_version})")
                            else:
                                self.app.logger.debug(f"Connected to {server_url} but unexpected response")

                        except Exception as connection_error:
                            if "Connection refused" in str(connection_error):
                                self.app.logger.info(f"\u274c Buttplug server not running at {server_url}")
                                self.app.logger.info("Please start Intiface Central and enable server mode")
                            else:
                                self.app.logger.error(f"Server status check failed: {connection_error}")

                    except Exception as e:
                        self.app.logger.error(f"Failed to check server status: {e}")

                try:
                    loop.run_until_complete(check_server())
                finally:
                    loop.close()

            thread = threading.Thread(target=run_status_check, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to start server status check: {e}")

    def _test_buttplug_movement(self):
        """Test movement for connected Buttplug device."""
        try:
            import threading
            def run_movement_test():
                import asyncio
                import time
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    if not self.device_manager.is_connected():
                        self.app.logger.info("No device connected for movement test")
                        return

                    self.app.logger.debug("Testing Buttplug device movement...")

                    # Test sequence with good timing
                    positions = [0, 100, 25, 75, 50]
                    for i, pos in enumerate(positions):
                        loop.run_until_complete(asyncio.sleep(0.8))  # Wait between positions
                        self.device_manager.update_position(pos, 50.0)
                        self.app.logger.info(f"   Step {i+1}/{len(positions)}: Position {pos}%")

                    # Return to center
                    loop.run_until_complete(asyncio.sleep(0.8))
                    self.device_manager.update_position(50.0, 50.0)
                    self.app.logger.debug("Movement test complete")

                except Exception as e:
                    self.app.logger.error(f"Movement test failed: {e}")
                finally:
                    loop.close()

            thread = threading.Thread(target=run_movement_test, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to start movement test: {e}")

    def _connect_handy(self, connection_key: str):
        """Connect to Handy device with given connection key."""
        import threading
        import asyncio

        def connect_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                success = loop.run_until_complete(self.device_manager.connect_handy(connection_key))
                if success:
                    self.app.logger.info("Connected to Handy", extra={'status_message': True})
                    # Auto-upload funscript on connect (like Heresphere)
                    self._auto_upload_on_connect()
                else:
                    self.app.logger.error(
                        "Failed to connect to Handy - is the device turned on?",
                        extra={'status_message': True, 'duration': 5.0})
            except Exception as e:
                self.app.logger.error(
                    f"Handy connection error: {e}",
                    extra={'status_message': True, 'duration': 5.0})
            finally:
                loop.close()

        threading.Thread(target=connect_async, daemon=True).start()

    def _auto_upload_on_connect(self):
        """Auto-upload funscript when Handy connects (if available)."""
        try:
            if not hasattr(self.app, 'funscript_processor') or not self.app.funscript_processor:
                return
            actions = self.app.funscript_processor.get_actions('primary')
            if not actions:
                return
            self.app.logger.info(
                "Auto-uploading funscript to Handy...",
                extra={'status_message': True})
            self._upload_funscript_to_handy(timeline_num=1)
        except Exception as e:
            self.app.logger.warning(f"Auto-upload on connect failed: {e}")

    def _disconnect_handy(self):
        """Disconnect from Handy device."""
        import threading
        import asyncio

        def disconnect_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.device_manager.disconnect_handy())
            finally:
                loop.close()

        threading.Thread(target=disconnect_async, daemon=True).start()

    def _apply_handy_hstp_offset(self, offset_ms: int):
        """Apply sync offset instantly via Handy's /hstp/offset API."""
        import threading
        import asyncio

        if not self.device_manager or not self.device_manager.is_connected():
            return

        def set_offset_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.device_manager.set_handy_hstp_offset(offset_ms))
            finally:
                loop.close()

        threading.Thread(target=set_offset_async, daemon=True).start()

    def _apply_handy_sync_offset(self):
        """Apply sync offset via Handy's /hstp/offset API."""
        sync_offset = self.app.app_settings.get("device_control_handy_sync_offset_ms", 0)
        self._apply_handy_hstp_offset(sync_offset)

    def _upload_funscript_to_handy(self, timeline_num: int = 1):
        """Upload funscript from specified timeline to Handy for HSSP streaming."""
        import threading
        import asyncio

        # Get funscript actions for the requested timeline
        if not hasattr(self.app, 'funscript_processor') or not self.app.funscript_processor:
            self.app.logger.error("No funscript loaded", extra={'status_message': True})
            return

        funscript_obj = self.app.funscript_processor.get_funscript_obj()
        if not funscript_obj:
            self.app.logger.error("No funscript loaded", extra={'status_message': True})
            return

        # Resolve axis name for the timeline
        axis_assignments = funscript_obj.get_axis_assignments() if hasattr(funscript_obj, 'get_axis_assignments') else {}
        if timeline_num == 1:
            axis_key = 'primary'
        elif timeline_num == 2:
            axis_key = 'secondary'
        else:
            axis_key = axis_assignments.get(timeline_num, 'primary')

        actions = funscript_obj.get_axis_actions(axis_key)
        if not actions:
            axis_name = axis_assignments.get(timeline_num, f'timeline {timeline_num}')
            self.app.logger.error(
                f"No actions on timeline {timeline_num} ({axis_name})",
                extra={'status_message': True})
            return

        # Track upload revision per timeline for stale-script detection
        if not hasattr(self, '_handy_uploaded_timelines'):
            self._handy_uploaded_timelines = {}
        upload_rev = getattr(self.app.funscript_processor, '_revision', 0)
        self._handy_uploaded_timelines[timeline_num] = upload_rev
        self._handy_last_upload_hash = upload_rev

        axis_name = axis_assignments.get(timeline_num, 'stroke')
        self.app.logger.info(
            f"Uploading timeline {timeline_num} ({axis_name}, {len(actions)} actions) to Handy...",
            extra={'status_message': True})

        def upload_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self.device_manager.prepare_handy_for_video_playback(actions)
                )
                self.app.logger.info(
                    f"Funscript uploaded to Handy ({len(actions)} actions)",
                    extra={'status_message': True})
            except Exception as e:
                self.app.logger.error(
                    f"Failed to upload funscript: {e}",
                    extra={'status_message': True, 'duration': 5.0})
            finally:
                loop.close()

        threading.Thread(target=upload_async, daemon=True).start()

    # ── OSSM BLE Controls ───────────────────────────────────────────────

    def _render_ossm_controls(self):
        """Render OSSM BLE device controls."""
        try:
            # Check if bleak is available
            ossm_available = 'ossm' in (self.device_manager.available_backends if self.device_manager else {})

            if not ossm_available:
                imgui.text_colored("OSSM backend unavailable", 0.7, 0.5, 0.0)
                imgui.text("Install bleak: pip install bleak>=0.21.0")
                return

            # Check connection status
            connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
            is_ossm_connected = connected_device and connected_device.device_id.startswith("ossm_")

            if is_ossm_connected:
                self._render_ossm_connected(connected_device)
            else:
                self._render_ossm_disconnected()

        except Exception as e:
            imgui.text_colored(f"OSSM error: {e}", 1.0, 0.3, 0.3)

    def _render_ossm_connected(self, device_info):
        """Render OSSM controls when connected."""
        self._status_indicator(f"Connected to {device_info.name}", "ready", "OSSM connected via BLE")

        # Device state from BLE notifications
        ossm_backend = self.device_manager.available_backends.get('ossm')
        if ossm_backend and hasattr(ossm_backend, 'device_state'):
            state = ossm_backend.device_state
            imgui.text(f"Mode: {state.mode}")
            imgui.same_line(150)
            imgui.text(f"Speed: {state.speed}")
            imgui.same_line(250)
            imgui.text(f"Stroke: {state.stroke}")

        imgui.spacing()

        # Speed knob override checkbox
        knob_override = self.app.app_settings.get("ossm_speed_knob_override", True)
        changed_knob, new_knob = imgui.checkbox("Speed Knob Override##OSSMKnob", knob_override)
        if changed_knob:
            self.app.app_settings.set("ossm_speed_knob_override", new_knob)
            self._set_ossm_speed_knob(new_knob)
        _tooltip_if_hovered("When enabled, BLE has full speed control.\nWhen disabled, the physical knob limits BLE speed.")

        imgui.spacing()

        # Manual sliders
        imgui.text("Manual Controls:")

        # Speed slider
        speed_val = getattr(self, '_ossm_manual_speed', 50)
        changed_s, new_speed = imgui.slider_int("Speed##OSSMSpeed", speed_val, 0, 100)
        if changed_s:
            self._ossm_manual_speed = new_speed
            self._send_ossm_command(f"set:speed:{new_speed}")

        # Stroke slider
        stroke_val = getattr(self, '_ossm_manual_stroke', 50)
        changed_st, new_stroke = imgui.slider_int("Stroke##OSSMStroke", stroke_val, 0, 100)
        if changed_st:
            self._ossm_manual_stroke = new_stroke
            self._send_ossm_command(f"set:stroke:{new_stroke}")

        # Depth slider
        depth_val = getattr(self, '_ossm_manual_depth', 50)
        changed_d, new_depth = imgui.slider_int("Depth##OSSMDepth", depth_val, 0, 100)
        if changed_d:
            self._ossm_manual_depth = new_depth
            self._send_ossm_command(f"set:depth:{new_depth}")

        # Sensation slider
        sens_val = getattr(self, '_ossm_manual_sensation', 0)
        changed_sn, new_sens = imgui.slider_int("Sensation##OSSMSensation", sens_val, 0, 100)
        if changed_sn:
            self._ossm_manual_sensation = new_sens
            self._send_ossm_command(f"set:sensation:{new_sens}")

        imgui.spacing()

        # Movement test button
        if imgui.button("Test Movement##OSSMTest"):
            self._test_ossm_movement()
        _tooltip_if_hovered("Run a short streaming mode test sequence")

        imgui.same_line()

        # Disconnect button
        with destructive_button_style():
            if imgui.button("Disconnect##OSSMDisconnect"):
                self._disconnect_ossm()
        _tooltip_if_hovered("Disconnect from OSSM device")

    def _render_ossm_disconnected(self):
        """Render OSSM controls when disconnected."""
        # Scan button
        if imgui.button("Scan for OSSM Devices##OSSMScan", width=-1):
            self._scan_ossm_devices()

        imgui.spacing()

        # Show discovered devices
        if self._ossm_scan_performed:
            if self._discovered_ossm_devices:
                for i, device in enumerate(self._discovered_ossm_devices):
                    name = device.get('name', 'Unknown')
                    rssi = device.get('rssi', '')
                    rssi_text = f" (RSSI: {rssi})" if rssi else ""
                    imgui.bullet_text(f"{name}{rssi_text}")
                    imgui.same_line()
                    if imgui.small_button(f"Connect##{i}"):
                        address = device.get('address', '')
                        if address:
                            self._connect_ossm_device(address)
            else:
                imgui.text_colored("No OSSM devices found", 0.7, 0.5, 0.0)
                imgui.spacing()
                imgui.text("Troubleshooting:")
                imgui.bullet_text("Ensure OSSM is powered on")
                imgui.bullet_text("Check Bluetooth is enabled")
                imgui.bullet_text("Move closer to the device")
                imgui.bullet_text("Try scanning again")

        imgui.spacing()

        # Advanced settings
        rate_hz = self.app.app_settings.get("ossm_max_command_rate_hz", 40)
        changed_rate, new_rate = imgui.slider_int("Max Rate (Hz)##OSSMRate", rate_hz, 10, 50)
        if changed_rate:
            self.app.app_settings.set("ossm_max_command_rate_hz", new_rate)
        _tooltip_if_hovered("Maximum BLE command rate. Higher = smoother movement.")

        auto_reconnect = self.app.app_settings.get("ossm_auto_reconnect", True)
        changed_ar, new_ar = imgui.checkbox("Auto-Reconnect##OSSMAuto", auto_reconnect)
        if changed_ar:
            self.app.app_settings.set("ossm_auto_reconnect", new_ar)
        _tooltip_if_hovered("Automatically reconnect if BLE connection drops")

    # ── OSSM helper methods ──────────────────────────────────────────────

    def _scan_ossm_devices(self):
        """Scan for OSSM devices via BLE."""
        import threading

        def run_ossm_scan():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ossm_backend = self.device_manager.available_backends.get('ossm')
                if ossm_backend:
                    devices = loop.run_until_complete(ossm_backend.discover_devices())
                    self._discovered_ossm_devices = []
                    for device in devices:
                        self._discovered_ossm_devices.append({
                            'name': device.name,
                            'address': device.metadata.get('ble_address', ''),
                            'rssi': device.metadata.get('rssi', ''),
                            'device_id': device.device_id,
                        })
                    self.app.logger.info(f"Found {len(devices)} OSSM devices")
                    self._ossm_scan_performed = True
            except Exception as e:
                self.app.logger.error(f"OSSM scan failed: {e}")
                self._ossm_scan_performed = True
            finally:
                loop.close()

        threading.Thread(target=run_ossm_scan, daemon=True).start()

    def _connect_ossm_device(self, ble_address):
        """Connect to an OSSM device by BLE address."""
        import threading

        def connect_async():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Apply settings to backend before connecting
                ossm_backend = self.device_manager.available_backends.get('ossm')
                if ossm_backend:
                    rate_hz = self.app.app_settings.get("ossm_max_command_rate_hz", 40)
                    ossm_backend.set_max_rate_hz(rate_hz)
                    ossm_backend._reconnect_enabled = self.app.app_settings.get("ossm_auto_reconnect", True)

                success = loop.run_until_complete(self.device_manager.connect_ossm(ble_address))
                if success:
                    self.app.logger.info(
                        "Connected to OSSM",
                        extra={'status_message': True})
                    # Apply speed knob override
                    knob_override = self.app.app_settings.get("ossm_speed_knob_override", True)
                    if ossm_backend:
                        loop.run_until_complete(ossm_backend.set_speed_knob_override(knob_override))
                else:
                    self.app.logger.error(
                        "Failed to connect to OSSM - is it powered on?",
                        extra={'status_message': True, 'duration': 5.0})
            except Exception as e:
                self.app.logger.error(
                    f"OSSM connection error: {e}",
                    extra={'status_message': True, 'duration': 5.0})
            finally:
                loop.close()

        threading.Thread(target=connect_async, daemon=True).start()

    def _disconnect_ossm(self):
        """Disconnect from OSSM device."""
        import threading

        def disconnect_async():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.device_manager.disconnect_ossm())
            finally:
                loop.close()

        threading.Thread(target=disconnect_async, daemon=True).start()

    def _send_ossm_command(self, cmd):
        """Send a command to the OSSM device."""
        import threading

        def send_async():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ossm_backend = self.device_manager.available_backends.get('ossm')
                if ossm_backend:
                    loop.run_until_complete(ossm_backend._send_command(cmd))
            finally:
                loop.close()

        threading.Thread(target=send_async, daemon=True).start()

    def _set_ossm_speed_knob(self, enabled):
        """Set OSSM speed knob override."""
        import threading

        def set_async():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ossm_backend = self.device_manager.available_backends.get('ossm')
                if ossm_backend:
                    loop.run_until_complete(ossm_backend.set_speed_knob_override(enabled))
            finally:
                loop.close()

        threading.Thread(target=set_async, daemon=True).start()

    def _test_ossm_movement(self):
        """Run a short streaming mode test sequence on the OSSM."""
        import threading

        def test_async():
            import asyncio

            async def run_test():
                ossm_backend = self.device_manager.available_backends.get('ossm')
                if not ossm_backend or not ossm_backend.is_connected():
                    self.app.logger.error("OSSM not connected")
                    return

                self.app.logger.info("OSSM test: starting streaming sequence...")
                # Enter streaming mode and do a few movements
                positions = [(10, 500), (90, 500), (50, 300), (80, 400), (20, 400), (50, 500)]
                for pos, dur in positions:
                    await ossm_backend.set_position_enhanced(pos, duration_ms=dur)
                    await asyncio.sleep(dur / 1000.0)

                # Return to center
                await ossm_backend.set_position_enhanced(50, duration_ms=500)
                self.app.logger.info("OSSM test: complete")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_test())
            finally:
                loop.close()

        threading.Thread(target=test_async, daemon=True).start()
