"""Device Control tab UI mixin for ControlPanelUI."""
import imgui
from funscript.axis_registry import AXIS_TCODE

# Shared TCode axis mapping derived from the canonical axis registry.
# Maps UI axis keys (matching OSRControlProfile field names) to TCode IDs.
_DEVICE_AXIS_TCODE = {
    'up_down': 'L0',
    'left_right': 'L1',
    'front_back': 'L2',
    'twist': 'R0',
    'roll': 'R1',
    'pitch': 'R2',
    'vibration': 'V0',
    'aux_vibration': 'V1',
}

# Module-level helper used by methods in the mixin
def _tooltip_if_hovered(text):
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


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
                imgui.text_colored("Check logs for details.", 1.0, 0.5, 0.0)
                if imgui.button("Retry Initialization"):
                    # Reset initialization flag to try again
                    self._device_control_initialized = False

        except Exception as e:
            imgui.text(f"Error in Device Control: {e}")
            imgui.text_colored("See logs for full details.", 1.0, 0.0, 0.0)

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

    def _render_device_control_content(self):
        """Render the main device control interface with improved UX."""
        # Version info (top of tab, consistent with other supporter modules)
        try:
            import device_control
            version = getattr(device_control, '__version__', 'unknown')
            imgui.text_colored(f"Device Control v{version}", 0.5, 0.5, 0.5, 1.0)
            imgui.spacing()
        except Exception:
            pass

        # SIMPLIFIED: Compact connection status (always visible)
        self._render_compact_connection_status()

        imgui.separator()

        # SIMPLIFIED: Quick controls when connected (always visible)
        if self.device_manager.is_connected():
            self._render_quick_controls()
            imgui.separator()

        # Device Types (collapsible)
        if not self.device_manager.is_connected():
            imgui.text("Connect a Device:")
            imgui.spacing()

        # OSR2/OSR6 Devices
        if imgui.collapsing_header("OSR2/OSR6 (USB)##OSRDevices", flags=0 if self.device_manager.is_connected() else imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._render_osr_controls()

        # Buttplug.io Universal Devices
        if imgui.collapsing_header("Buttplug.io (Universal)##ButtplugDevices", flags=0 if self.device_manager.is_connected() else imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._render_buttplug_controls()

        # Handy Direct Control
        if imgui.collapsing_header("Handy (Direct)##HandyDirect", flags=0 if self.device_manager.is_connected() else imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._render_handy_controls()

        # OSSM BLE Direct Control
        if imgui.collapsing_header("OSSM (Bluetooth)##OSSMDevices", flags=0 if self.device_manager.is_connected() else imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            self._render_ossm_controls()

        # Axis Routing (shown when connected)
        if self.device_manager.is_connected():
            imgui.separator()
            if imgui.collapsing_header("Axis Routing##AxisRouting")[0]:
                self._render_axis_routing()

        # SIMPLIFIED: All settings in one collapsible section
        if self.device_manager.is_connected():
            imgui.separator()
            if imgui.collapsing_header("Advanced Settings##DeviceAdvancedAll")[0]:
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

        # Global stroke range for all active axes
        imgui.text("Stroke Range (All Active Axes):")

        # Get current profile settings
        current_profile_name = self.app.app_settings.get("device_control_selected_profile", "Balanced")
        osr_profiles = self.app.app_settings.get("device_control_osr_profiles", {})

        if current_profile_name in osr_profiles:
            profile_data = osr_profiles[current_profile_name]

            # Calculate global min/max from enabled axes
            active_axes = []
            for axis_key in ["up_down", "left_right", "front_back", "twist", "roll", "pitch"]:
                if axis_key in profile_data and profile_data[axis_key].get("enabled", False):
                    active_axes.append(axis_key)

            if active_axes:
                # Get average min/max from active axes
                avg_min = int(sum(profile_data[axis].get("min_position", 0) for axis in active_axes) / len(active_axes))
                avg_max = int(sum(profile_data[axis].get("max_position", 9999) for axis in active_axes) / len(active_axes))

                # Global min slider
                changed_min, new_min = imgui.slider_int("Min Extent##GlobalMin", avg_min, 0, 5000, "%d")
                if changed_min:
                    # Apply to all active axes
                    for axis_key in active_axes:
                        profile_data[axis_key]["min_position"] = new_min
                    osr_profiles[current_profile_name] = profile_data
                    self.app.app_settings.set("device_control_osr_profiles", osr_profiles)
                    self._preview_global_extent(new_min, "min")

                # Global max slider
                changed_max, new_max = imgui.slider_int("Max Extent##GlobalMax", avg_max, 5000, 9999, "%d")
                if changed_max:
                    # Apply to all active axes
                    for axis_key in active_axes:
                        profile_data[axis_key]["max_position"] = new_max
                    osr_profiles[current_profile_name] = profile_data
                    self.app.app_settings.set("device_control_osr_profiles", osr_profiles)
                    self._preview_global_extent(new_max, "max")

                _tooltip_if_hovered("Adjust min/max for all active axes at once. Drag to feel the limits in real-time.")
            else:
                imgui.text_colored("No active axes configured", 0.7, 0.5, 0.0)

        imgui.spacing()

        # Movement recording
        self._render_recording_controls()

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

    def _preview_global_extent(self, value, extent_type):
        """Preview global min or max extent by moving device to that position."""
        try:
            # Convert T-code value (0-9999) to percentage (0-100)
            percentage = (value / 9999.0) * 100.0
            self.device_manager.update_position(percentage, 50.0)
        except Exception as e:
            self.app.logger.error(f"Error previewing global extent: {e}")

    def _render_recording_controls(self):
        """Render movement recording controls in Quick Controls."""
        try:
            recorder = getattr(self.device_manager, 'recorder', None)
            if not recorder:
                return

            if recorder.is_recording:
                # Recording active — show red indicator + elapsed time + stop button
                elapsed_s = recorder.duration_ms / 1000.0
                imgui.text_colored("REC", 1.0, 0.2, 0.2)
                imgui.same_line()
                imgui.text(f"{elapsed_s:.1f}s ({recorder.action_count} pts)")
                imgui.same_line()
                if imgui.small_button("Stop Recording##RecStop"):
                    recorder.stop()
                    self._export_recording()
            else:
                if imgui.small_button("Record Movement##RecStart"):
                    proc = self.app.processor
                    video_time_ms = 0
                    if proc and proc.is_video_open():
                        fps = proc.fps or 30.0
                        video_time_ms = int((proc.current_frame_index / fps) * 1000)
                    recorder.start(video_time_ms)
                _tooltip_if_hovered("Record device positions during playback and export as .funscript")
        except Exception:
            pass

    def _export_recording(self):
        """Export recorded movement as funscript file."""
        try:
            recorder = self.device_manager.recorder
            if recorder.action_count == 0:
                self.app.logger.warning("No actions recorded — nothing to export")
                return

            # Build default filename from video name
            import os
            proc = self.app.processor
            base_name = "recorded"
            if proc and proc.is_video_open() and hasattr(proc, 'video_path'):
                base_name = os.path.splitext(os.path.basename(proc.video_path))[0]

            default_path = os.path.join(
                os.path.dirname(getattr(proc, 'video_path', '.') or '.'),
                f"{base_name}_recorded.funscript"
            )

            recorder.save_funscript(default_path)
            self.app.logger.info(
                f"Recording exported: {default_path} ({recorder.action_count} actions)",
                extra={'status_message': True, 'duration': 5.0}
            )
        except Exception as e:
            self.app.logger.error(f"Failed to export recording: {e}")

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

    def _render_axis_routing(self):
        """Render axis routing configuration for connected device(s)."""
        from device_control.axis_routing import DEVICE_CHANNELS, TCODE_DEFAULT_AXIS

        imgui.indent(10)

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
            source_labels.append(f"{axis_name} (Timeline {tl_num})")

        router = self.device_manager.axis_router

        for device_id, backend in self.device_manager.connected_devices.items():
            device_type = self.device_manager.get_device_type_for_id(device_id)
            config = router.get_config(device_id)

            if not config:
                # Auto-detect on first render
                config = router.auto_detect(device_id, device_type, axis_assignments)

            channels = DEVICE_CHANNELS.get(device_type, ["L0"])
            is_single = len(channels) == 1

            device_name = self.device_manager.get_connected_device_name()
            imgui.text_colored(f"{device_name}", 0.7, 0.7, 0.9)
            imgui.same_line()
            if imgui.small_button(f"Auto-Detect##{device_id}"):
                router.auto_detect(device_id, device_type, axis_assignments)
                config = router.get_config(device_id)
                router.save_to_settings(self.app.app_settings)

            if is_single:
                # Single-axis device (Handy): just a source dropdown
                route = config.routes.get("L0")
                if route:
                    current_idx = 0
                    if route.source_axis in source_options:
                        current_idx = source_options.index(route.source_axis)
                    changed, new_idx = imgui.combo(
                        f"Source##{device_id}_L0", current_idx, source_labels
                    )
                    if changed:
                        route.source_axis = source_options[new_idx]
                        route.enabled = route.source_axis != "(none)"
                        router.save_to_settings(self.app.app_settings)
            else:
                # Multi-axis device (OSR/SR6): full table
                # Table headers
                imgui.columns(4, f"axis_routing_table_{device_id}", True)
                imgui.set_column_width(0, 50)
                imgui.set_column_width(1, 180)
                imgui.set_column_width(2, 35)
                imgui.set_column_width(3, 35)

                imgui.text("Ch")
                imgui.next_column()
                imgui.text("Source")
                imgui.next_column()
                imgui.text("En")
                imgui.next_column()
                imgui.text("Inv")
                imgui.next_column()
                imgui.separator()

                for ch in channels:
                    route = config.routes.get(ch)
                    if not route:
                        from device_control.axis_routing import AxisRoute
                        route = AxisRoute(device_channel=ch, source_axis="none", enabled=False)
                        config.routes[ch] = route

                    # Channel name
                    default_axis = TCODE_DEFAULT_AXIS.get(ch, "")
                    imgui.text(ch)
                    if imgui.is_item_hovered() and default_axis:
                        imgui.set_tooltip(f"Default: {default_axis}")
                    imgui.next_column()

                    # Source dropdown
                    current_idx = 0
                    if route.source_axis in source_options:
                        current_idx = source_options.index(route.source_axis)
                    imgui.push_item_width(-1)
                    changed, new_idx = imgui.combo(
                        f"##{device_id}_{ch}_src", current_idx, source_labels
                    )
                    imgui.pop_item_width()
                    if changed:
                        route.source_axis = source_options[new_idx]
                        if route.source_axis == "(none)":
                            route.enabled = False
                        router.save_to_settings(self.app.app_settings)
                    imgui.next_column()

                    # Enable checkbox
                    changed, new_val = imgui.checkbox(f"##{device_id}_{ch}_en", route.enabled)
                    if changed:
                        route.enabled = new_val
                        router.save_to_settings(self.app.app_settings)
                    imgui.next_column()

                    # Invert checkbox
                    changed, new_val = imgui.checkbox(f"##{device_id}_{ch}_inv", route.invert)
                    if changed:
                        route.invert = new_val
                        router.save_to_settings(self.app.app_settings)
                    imgui.next_column()

                imgui.columns(1)

        imgui.unindent(10)

    def _render_all_advanced_settings(self):
        """Render all advanced settings in one section."""
        imgui.indent(10)

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

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Per-Axis Configuration (for OSR devices)
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        if connected_device and "osr" in connected_device.device_id.lower():
            if imgui.tree_node("Per-Axis Configuration##PerAxis"):
                imgui.text_colored("Configure individual axes:", 0.7, 0.7, 0.7)
                self._render_osr_axis_configuration()
                imgui.tree_pop()

        imgui.unindent(10)

    def _render_connection_status_section(self):
        """Render connection status section with consistent UX."""
        imgui.indent(15)

        if self.device_manager.is_connected():
            device_name = self.device_manager.get_connected_device_name()
            self._status_indicator(f"Connected to {device_name}", "ready", "Device is connected and ready")

            # Connection info
            device_info = self.device_manager.get_connected_device_info()
            if device_info:
                imgui.text(f"Device ID: {device_info.device_id}")
                imgui.text(f"Type: {device_info.device_type.value.title()}")

                # Quick position test
                imgui.separator()
                imgui.text("Quick Test:")
                current_pos = self.device_manager.current_position
                changed, new_pos = imgui.slider_float("Position##QuickTest", current_pos, 0.0, 100.0, "%.1f")
                if changed:
                    self.device_manager.update_position(new_pos, 50.0)

                _tooltip_if_hovered("Drag to test device movement")

            imgui.separator()
            if imgui.button("Disconnect Device"):
                self._disconnect_current_device()
        else:
            self._status_indicator("No device connected", "warning", "Connect a device below")
            imgui.text("Select and connect a device from the types below.")

        imgui.unindent(15)

    def _render_device_types_section(self):
        """Render device types section with consistent UX."""
        imgui.indent(15)

        # OSR2/OSR6 Devices
        if imgui.collapsing_header("OSR2/OSR6 Devices (USB/Serial)##OSRDevices")[0]:
            self._render_osr_controls()

        # Buttplug.io Universal Devices
        if imgui.collapsing_header("Buttplug.io Devices (Universal)##ButtplugDevices")[0]:
            self._render_buttplug_controls()

        # Handy Direct Control
        if imgui.collapsing_header("Handy (Direct API)##HandyDirect")[0]:
            self._render_handy_controls()

        imgui.unindent(15)

    def _render_osr_controls(self):
        """Render OSR device controls."""
        imgui.indent(10)

        # Check OSR connection status
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        is_osr_connected = connected_device and "osr" in connected_device.device_id.lower()

        if is_osr_connected:
            self._status_indicator(f"Connected to {connected_device.device_id}", "ready", "OSR device connected and ready")

            # Advanced OSR Settings
            if imgui.collapsing_header("OSR Performance Settings##OSRPerformance")[0]:
                self._render_osr_performance_settings()

            if imgui.collapsing_header("OSR Axis Configuration##OSRAxis")[0]:
                self._render_osr_axis_configuration()

            if imgui.collapsing_header("OSR Test Functions##OSRTest")[0]:
                imgui.indent(10)
                if imgui.button("Run Movement Test##OSR"):
                    self._test_osr_movement()
                _tooltip_if_hovered("Test OSR device with predefined movement sequence")
                imgui.unindent(10)

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

        imgui.unindent(10)

    def _render_handy_controls(self):
        """Render Handy direct API controls."""
        imgui.indent(10)

        # Check Handy connection status
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        is_handy_connected = connected_device and "handy" in connected_device.device_id.lower()

        if is_handy_connected:
            # Connected state + measured RTD
            rtd_ms = self.device_manager.get_handy_rtd_ms() if hasattr(self.device_manager, 'get_handy_rtd_ms') else 0
            rtd_label = f"  (RTD: {rtd_ms}ms)" if rtd_ms > 0 else ""
            self._status_indicator(f"Connected to {connected_device.name}{rtd_label}", "ready", "Handy connected and ready")

            # Mode selector (HDSP vs HSSP)
            handy_mode = self.app.app_settings.get("device_control_handy_mode", "HDSP (Direct)")
            mode_options = ["HDSP (Direct)", "HSSP (Script Sync)"]
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
            _tooltip_if_hovered("HDSP: sends positions in real-time (recommended)\nHSSP: uploads script to device (use if HDSP has issues)")

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
                            # Show upload indicator
                            uploaded_tls = getattr(self, '_handy_uploaded_timelines', {})
                            current_hash = len(actions)
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
                    current_hash = len(current_actions) if current_actions else 0

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
            if imgui.button("Disconnect Handy##HandyDisconnect"):
                self._disconnect_handy()
            _tooltip_if_hovered("Disconnect from Handy device")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Sync settings (HSSP only — not applicable to HDSP mode)
            if not is_hdsp_mode:
                imgui.text("Sync Settings:")
                imgui.indent(10)

                current_offset = self.app.app_settings.get("device_control_handy_sync_offset_ms", 0)

                # Row 1: -50 / -10 / -1 buttons, then slider, then +1 / +10 / +50 buttons
                # Fine adjustment buttons (left side - decrease compensation)
                if imgui.button("-50##SyncMinus50"):
                    new_offset = max(0, current_offset - 50)
                    self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                    self._apply_handy_hstp_offset(new_offset)
                _tooltip_if_hovered("-50ms")
                imgui.same_line()

                if imgui.button("-10##SyncMinus10"):
                    new_offset = max(0, current_offset - 10)
                    self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                    self._apply_handy_hstp_offset(new_offset)
                _tooltip_if_hovered("-10ms")
                imgui.same_line()

                if imgui.button("-1##SyncMinus1"):
                    new_offset = max(0, current_offset - 1)
                    self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                    self._apply_handy_hstp_offset(new_offset)
                _tooltip_if_hovered("-1ms")
                imgui.same_line()

                # Sync offset slider
                imgui.push_item_width(100)
                changed, value = imgui.slider_int(
                    "##HandySyncOffset",
                    current_offset,
                    0, 2500
                )
                imgui.pop_item_width()
                if changed:
                    self.app.app_settings.set("device_control_handy_sync_offset_ms", value)
                    self._apply_handy_hstp_offset(value)
                _tooltip_if_hovered("Fine-tune sync feel (ms): higher = device moves earlier\nNetwork latency is auto-compensated; adjust if movement feels late")
                imgui.same_line()

                # Fine adjustment buttons (right side - increase compensation)
                if imgui.button("+1##SyncPlus1"):
                    new_offset = min(2500, current_offset + 1)
                    self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                    self._apply_handy_hstp_offset(new_offset)
                _tooltip_if_hovered("+1ms")
                imgui.same_line()

                if imgui.button("+10##SyncPlus10"):
                    new_offset = min(2500, current_offset + 10)
                    self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                    self._apply_handy_hstp_offset(new_offset)
                _tooltip_if_hovered("+10ms")
                imgui.same_line()

                if imgui.button("+50##SyncPlus50"):
                    new_offset = min(2500, current_offset + 50)
                    self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                    self._apply_handy_hstp_offset(new_offset)
                _tooltip_if_hovered("+50ms")

                # Row 2: Direct numeric input + current value display
                imgui.push_item_width(80)
                changed, input_value = imgui.input_int("##SyncOffsetInput", current_offset, 0, 0)
                imgui.pop_item_width()
                if changed:
                    clamped_value = max(0, min(2500, input_value))
                    self.app.app_settings.set("device_control_handy_sync_offset_ms", clamped_value)
                    self._apply_handy_hstp_offset(clamped_value)
                _tooltip_if_hovered("Enter lag compensation directly (ms)\n0 to 2500")
                imgui.same_line()
                imgui.text("ms")
                imgui.same_line()
                if current_offset > 0:
                    imgui.text_colored(f"(compensating {current_offset}ms lag)", 0.5, 0.8, 0.5, 1.0)
                else:
                    imgui.text_colored("(no compensation)", 0.6, 0.6, 0.6, 1.0)

                imgui.unindent(10)

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
                from application.utils import primary_button_style
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

        imgui.unindent(10)

    def _render_buttplug_controls(self):
        """Render Buttplug.io device controls."""
        imgui.indent(10)

        # Check Buttplug connection status
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        is_buttplug_connected = connected_device and "buttplug" in connected_device.device_id.lower()

        if is_buttplug_connected:
            self._status_indicator(f"Connected to {connected_device.name}", "ready", "Buttplug device connected and ready")

            # Device capabilities
            if hasattr(connected_device, 'capabilities') and connected_device.capabilities:
                caps = connected_device.capabilities
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

            # Advanced Buttplug Settings
            if imgui.collapsing_header("Buttplug Test Functions##ButtplugTest")[0]:
                imgui.indent(10)
                if imgui.button("Run Movement Test##Buttplug"):
                    self._test_buttplug_movement()
                _tooltip_if_hovered("Test device with predefined movement sequence")
                imgui.unindent(10)

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


        imgui.unindent(10)

    def _render_device_settings_section(self):
        """Render device settings section with consistent UX."""
        imgui.indent(15)

        # Performance Settings
        if imgui.collapsing_header("Device Performance##DevicePerformance")[0]:
            imgui.indent(10)
            config = self.device_manager.config

            # Update rate
            changed, new_rate = imgui.slider_float("Update Rate##DeviceRate", config.max_position_rate_hz, 1.0, 120.0, "%.1f Hz")
            if changed:
                config.max_position_rate_hz = new_rate
            _tooltip_if_hovered("How often device position is updated per second (T-Code devices can handle 60-120Hz)")

            # Position smoothing
            changed, new_smoothing = imgui.slider_float("Position Smoothing##DeviceSmooth", config.position_smoothing, 0.0, 1.0, "%.2f")
            if changed:
                config.position_smoothing = new_smoothing
            _tooltip_if_hovered("Smooths position changes to reduce jerkiness (0=no smoothing, 1=maximum smoothing)")

            # Latency compensation
            changed, new_latency = imgui.slider_int("Latency Compensation##DeviceLatency", config.latency_compensation_ms, 0, 200)
            if changed:
                config.latency_compensation_ms = new_latency
            _tooltip_if_hovered("Compensates for device response delay in milliseconds")

            imgui.unindent(10)


        # Live Control Integration
        if imgui.collapsing_header("Live Control Integration##DeviceLiveControl")[0]:
            imgui.indent(10)

            # Live tracking device control
            live_tracking_enabled = self.app.app_settings.get("device_control_live_tracking", False)
            changed, new_live_tracking = imgui.checkbox("Live Tracking Control##DeviceLiveTracking", live_tracking_enabled)
            if changed:
                self.app.app_settings.set("device_control_live_tracking", new_live_tracking)
                self.app.app_settings.save_settings()
                self._update_live_tracking_control(new_live_tracking)
            _tooltip_if_hovered("Stream live tracker data directly to device in real-time")

            # Video playback device control
            video_playback_enabled = self.app.app_settings.get("device_control_video_playback", False)
            changed, new_video_playback = imgui.checkbox("Video Playback Control##DeviceVideoPlayback", video_playback_enabled)
            if changed:
                self.app.app_settings.set("device_control_video_playback", new_video_playback)
                self.app.app_settings.save_settings()
                self._update_video_playback_control(new_video_playback)
            _tooltip_if_hovered("Sync device with video timeline and funscript playback")

            # Live Preview (editing-time haptic feedback)
            live_preview_bridge = getattr(self, 'live_preview_bridge', None)
            if live_preview_bridge:
                preview_enabled = live_preview_bridge.config.enabled
                changed, new_preview = imgui.checkbox("Live Preview##DeviceLivePreview", preview_enabled)
                if changed:
                    live_preview_bridge.config.enabled = new_preview
                    if new_preview:
                        live_preview_bridge.start()
                    else:
                        live_preview_bridge.stop()
                _tooltip_if_hovered("Send device position as you scrub through the timeline (even when video is paused)")

            imgui.unindent(10)


        # Advanced Settings (only show if live control enabled)
        live_tracking_enabled = self.app.app_settings.get("device_control_live_tracking", False)
        video_playback_enabled = self.app.app_settings.get("device_control_video_playback", False)

        if live_tracking_enabled or video_playback_enabled:
            if imgui.collapsing_header("Advanced Control Settings##DeviceAdvanced")[0]:
                imgui.indent(10)

                # Control intensity
                live_intensity = self.app.app_settings.get("device_control_live_intensity", 1.0)
                changed, new_intensity = imgui.slider_float("Control Intensity##DeviceIntensity", live_intensity, 0.1, 2.0, "%.2fx")
                if changed:
                    self.app.app_settings.set("device_control_live_intensity", new_intensity)
                    self.app.app_settings.save_settings()
                _tooltip_if_hovered("Multiplier for device movement intensity")

                # Rolling Autotune (for live tracking → device)
                if live_tracking_enabled:
                    imgui.spacing()
                    imgui.separator()
                    imgui.spacing()
                    imgui.text("Rolling Autotune:")

                    settings = self.app.app_settings
                    tr = getattr(self.app, 'tracker', None)

                    if not tr:
                        imgui.text_colored("Tracker not initialized", 1.0, 0.5, 0.0, 1.0)
                    else:
                        cur_enabled = settings.get("live_tracker_rolling_autotune_enabled", False)
                        ch, new_enabled = imgui.checkbox("Enable##RollingAutotuneDevice", cur_enabled)
                        if imgui.is_item_hovered():
                            imgui.set_tooltip(
                                "Apply Ultimate Autotune to the last N seconds of live tracking data\n"
                                "before sending to the device. Cleans up noise for smoother motion.")
                        if ch:
                            settings.set("live_tracker_rolling_autotune_enabled", new_enabled)
                            tr.rolling_autotune_enabled = new_enabled

                        if cur_enabled:
                            cur_interval = settings.get("live_tracker_rolling_autotune_interval_ms", 5000)
                            imgui.push_item_width(120)
                            ch, new_interval = imgui.input_int("Interval (ms)##RAIntervalDev", cur_interval, 1000)
                            imgui.pop_item_width()
                            if imgui.is_item_hovered():
                                imgui.set_tooltip("How often to apply autotune (default: 5000ms)")
                            if ch:
                                v = max(1000, min(30000, new_interval))
                                if v != cur_interval:
                                    settings.set("live_tracker_rolling_autotune_interval_ms", v)
                                    tr.rolling_autotune_interval_ms = v

                            cur_window = settings.get("live_tracker_rolling_autotune_window_ms", 5000)
                            imgui.push_item_width(120)
                            ch, new_window = imgui.input_int("Window (ms)##RAWindowDev", cur_window, 1000)
                            imgui.pop_item_width()
                            if imgui.is_item_hovered():
                                imgui.set_tooltip("Size of data window to process (default: 5000ms)")
                            if ch:
                                v = max(1000, min(30000, new_window))
                                if v != cur_window:
                                    settings.set("live_tracker_rolling_autotune_window_ms", v)
                                    tr.rolling_autotune_window_ms = v

                imgui.unindent(10)

        imgui.unindent(15)

    def _disconnect_current_device(self):
        """Disconnect the currently connected device."""
        try:
            import threading
            import asyncio

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

            thread = threading.Thread(target=run_disconnect, daemon=True)
            thread.start()
        except Exception as e:
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

    def _preview_axis_position(self, axis_key, tcode_position, message):
        """Preview a specific axis position in real-time."""
        try:
            if not self.device_manager.is_connected():
                self.app.logger.warning("No device connected for preview")
                return

            connected_device = self.device_manager.get_connected_device_info()
            if not connected_device or "osr" not in connected_device.device_id.lower():
                self.app.logger.warning("Preview only available for OSR devices")
                return

            # Get the OSR backend
            backend = self.device_manager.get_connected_backend()
            if not backend:
                self.app.logger.warning("No connected backend available for preview")
                return

            # Check backend connection status
            if not backend.is_connected():
                self.app.logger.warning("Backend not connected for preview")
                return

            self.app.logger.debug(f"Using backend: {type(backend).__name__} for axis preview")

            tcode_axis = _DEVICE_AXIS_TCODE.get(axis_key)
            if not tcode_axis:
                self.app.logger.warning(f"Unknown axis key: {axis_key}")
                return

            # Convert TCode position (0-9999) to percentage (0-100) for backend
            position_percent = (tcode_position / 9999.0) * 100.0

            # Send command through backend's standardized axis method
            import threading
            def run_preview():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Log what we're about to send
                    self.app.logger.debug(f"Sending command: {tcode_axis} to {position_percent:.1f}% via {type(backend).__name__}")

                    # Check if backend is still connected before sending
                    if not backend.is_connected():
                        self.app.logger.error(f"Backend disconnected before sending {tcode_axis} command")
                        return

                    # Use backend's set_axis_position method instead of direct TCode
                    success = loop.run_until_complete(backend.set_axis_position(tcode_axis, position_percent))

                    if success:
                        self.app.logger.info(f"{message}: {tcode_axis} axis to {position_percent:.1f}%")
                    else:
                        self.app.logger.error(f"Failed to set {tcode_axis} axis position - backend returned False")

                except Exception as e:
                    self.app.logger.error(f"Failed to preview axis position: {e}")
                finally:
                    loop.close()

            thread = threading.Thread(target=run_preview, daemon=True)
            thread.start()

        except Exception as e:
            self.app.logger.error(f"Failed to preview axis position: {e}")

    def _demo_axis_range(self, axis_key, min_pos, max_pos, inverted, axis_label):
        """Demonstrate the full range of an axis with current settings."""
        try:
            if not self.device_manager.is_connected():
                self.app.logger.warning("No device connected for demo")
                return

            import threading
            def run_demo():
                import asyncio
                import time
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Get the OSR backend
                    backend = self.device_manager.get_connected_backend()
                    if not backend:
                        return

                    tcode_axis = _DEVICE_AXIS_TCODE.get(axis_key)
                    if not tcode_axis:
                        self.app.logger.warning(f"Unknown axis: {axis_key}")
                        return

                    self.app.logger.info(f"Demonstrating {axis_label} range...")

                    # Demo sequence: min -> max -> center (respecting inversion)
                    # Convert TCode positions (0-9999) to percentages (0-100) for backend
                    if inverted:
                        sequence = [
                            ((max_pos / 9999.0) * 100.0, "0% (inverted)"),
                            ((min_pos / 9999.0) * 100.0, "100% (inverted)"),
                            (((min_pos + max_pos) / 2 / 9999.0) * 100.0, "50% (center)")
                        ]
                    else:
                        sequence = [
                            ((min_pos / 9999.0) * 100.0, "0% (normal)"),
                            ((max_pos / 9999.0) * 100.0, "100% (normal)"),
                            (((min_pos + max_pos) / 2 / 9999.0) * 100.0, "50% (center)")
                        ]

                    for position_percent, label in sequence:
                        # Use backend's standardized axis method instead of direct TCode
                        success = loop.run_until_complete(backend.set_axis_position(tcode_axis, position_percent))
                        if success:
                            self.app.logger.info(f"{axis_label} demo: {label} \u2192 {tcode_axis} axis to {position_percent:.1f}%")
                        else:
                            self.app.logger.error(f"Failed to set {tcode_axis} axis to {position_percent:.1f}%")
                        time.sleep(2.0)  # Wait between movements

                    self.app.logger.info(f"{axis_label} range demonstration complete")

                except Exception as e:
                    self.app.logger.error(f"Failed to demo axis range: {e}")
                finally:
                    loop.close()

            thread = threading.Thread(target=run_demo, daemon=True)
            thread.start()

        except Exception as e:
            self.app.logger.error(f"Failed to start axis demo: {e}")

    def _simulate_axis_pattern(self, axis_key: str, pattern_type: str, min_pos: int, max_pos: int, inverted: bool, axis_label: str):
        """Simulate various motion patterns for axis testing."""
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

                    tcode_axis = _DEVICE_AXIS_TCODE.get(axis_key)
                    if not tcode_axis:
                        self.app.logger.warning(f"Unknown axis: {axis_key}")
                        return

                    self.app.logger.info(f"Starting {pattern_type} pattern for {axis_label}...")

                    # Generate pattern positions
                    center_pos = (min_pos + max_pos) // 2
                    amplitude = (max_pos - min_pos) // 2
                    duration = 10.0  # 10 seconds
                    steps = 50  # Number of steps
                    dt = duration / steps

                    positions = []

                    if pattern_type == "sine_wave":
                        for i in range(steps):
                            t = (i / steps) * 4 * math.pi  # 2 full cycles
                            offset = amplitude * math.sin(t)
                            pos = center_pos + offset
                            positions.append((int(pos), f"Sine {i}/{steps}"))

                    elif pattern_type == "square_wave":
                        for i in range(steps):
                            t = (i / steps) * 4  # 2 full cycles
                            pos = max_pos if (t % 2) < 1 else min_pos
                            positions.append((int(pos), f"Square {i}/{steps}"))

                    elif pattern_type == "triangle_wave":
                        for i in range(steps):
                            t = (i / steps) * 4  # 2 full cycles
                            cycle_pos = t % 2
                            if cycle_pos < 1:
                                # Rising
                                pos = min_pos + (max_pos - min_pos) * cycle_pos
                            else:
                                # Falling
                                pos = max_pos - (max_pos - min_pos) * (cycle_pos - 1)
                            positions.append((int(pos), f"Triangle {i}/{steps}"))

                    elif pattern_type == "random":
                        for i in range(20):  # Shorter for random
                            pos = random.randint(min_pos, max_pos)
                            positions.append((int(pos), f"Random {i}/20"))

                    elif pattern_type == "pulse":
                        for i in range(10):  # 10 pulses
                            # Pulse out and back
                            positions.append((max_pos, f"Pulse {i} - Out"))
                            positions.append((center_pos, f"Pulse {i} - Back"))

                    # Execute pattern
                    for pos, label in positions:
                        if inverted:
                            # Invert the position mapping
                            display_pos = max_pos + min_pos - pos
                        else:
                            display_pos = pos

                        # Convert to percentage for backend
                        position_percent = (display_pos / 9999.0) * 100.0

                        success = loop.run_until_complete(backend.set_axis_position(tcode_axis, position_percent))
                        if success:
                            self.app.logger.info(f"{axis_label} {pattern_type}: {label} \u2192 {tcode_axis} at {position_percent:.1f}%")
                        else:
                            self.app.logger.error(f"Failed to set {tcode_axis} to {position_percent:.1f}%")

                        time.sleep(dt)

                    # Return to center
                    center_percent = ((center_pos if not inverted else center_pos) / 9999.0) * 100.0
                    loop.run_until_complete(backend.set_axis_position(tcode_axis, center_percent))
                    self.app.logger.info(f"{axis_label} {pattern_type} pattern complete - returned to center")

                except Exception as e:
                    self.app.logger.error(f"Error in {pattern_type} pattern: {e}")
                finally:
                    loop.close()

            thread = threading.Thread(target=run_pattern, daemon=True)
            thread.start()

        except Exception as e:
            self.app.logger.error(f"Failed to start {pattern_type} pattern: {e}")

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

    def _initialize_video_playback_bridge(self):
        """Initialize video playback bridge."""
        try:
            if self.device_manager:
                from device_control.bridges.video_playback_bridge import create_video_playback_bridge
                self.video_playback_bridge = create_video_playback_bridge(self.device_manager)
                self.app.logger.info("Video playback bridge initialized")
            else:
                self.app.logger.warning("Device manager not available for video playback bridge")
        except Exception as e:
            self.app.logger.error(f"Failed to initialize video playback bridge: {e}")
            self.video_playback_bridge = None


    def _render_osr_axis_configuration(self):
        """Render OSR axis configuration UI."""
        try:
            imgui.separator()
            imgui.text("OSR Axis Configuration")

            # Load current OSR settings
            current_profile_name = self.app.app_settings.get("device_control_selected_profile", "Balanced")
            osr_profiles = self.app.app_settings.get("device_control_osr_profiles", {})

            if current_profile_name not in osr_profiles:
                imgui.text_colored("No OSR profile found in settings", 1.0, 0.5, 0.0)
                return

            profile_data = osr_profiles[current_profile_name]

            # Profile selection
            imgui.text("Profile:")
            imgui.same_line()
            profile_names = list(osr_profiles.keys())
            current_index = profile_names.index(current_profile_name) if current_profile_name in profile_names else 0

            changed, new_index = imgui.combo("##profile_selector", current_index, profile_names)
            if changed and 0 <= new_index < len(profile_names):
                new_profile_name = profile_names[new_index]
                self.app.app_settings.set("device_control_selected_profile", new_profile_name)
                profile_data = osr_profiles[new_profile_name]
                self._load_osr_profile_to_device(new_profile_name, profile_data)

            imgui.text(f"Description: {profile_data.get('description', 'No description')}")

            # Axis configurations
            imgui.separator()
            imgui.text("Axis Settings:")

            axes_to_show = [
                # Linear axes
                ("up_down", "Up/Down Stroke", "L0"),
                ("left_right", "Left/Right", "L1"),
                ("front_back", "Front/Back", "L2"),
                # Rotation axes
                ("twist", "Twist", "R0"),
                ("roll", "Roll", "R1"),
                ("pitch", "Pitch", "R2"),
                # Vibration axes
                ("vibration", "Vibration", "V0"),
                ("aux_vibration", "Aux Vibration", "V1")
            ]

            settings_changed = False

            for axis_key, axis_label, tcode in axes_to_show:
                if axis_key not in profile_data:
                    continue

                axis_data = profile_data[axis_key]

                # Axis header with enable checkbox
                enabled = axis_data.get("enabled", False)
                changed, new_enabled = imgui.checkbox(f"{axis_label} ({tcode})", enabled)
                if changed:
                    axis_data["enabled"] = new_enabled
                    settings_changed = True

                if enabled:
                    imgui.indent(20)

                    # Min/Max position sliders with real-time preview
                    min_pos = axis_data.get("min_position", 0)
                    max_pos = axis_data.get("max_position", 9999)

                    imgui.text(f"{axis_label} Range:")
                    imgui.text_colored("Drag sliders to feel the limits in real-time", 0.7, 0.7, 0.7)

                    changed, new_min = imgui.slider_int(f"Min Position##{axis_key}", min_pos, 0, 9999, f"%d (0%% limit)")
                    if changed:
                        axis_data["min_position"] = new_min
                        settings_changed = True
                        # Real-time preview: move to min position
                        self._preview_axis_position(axis_key, new_min, f"Previewing {axis_label} minimum")

                    changed, new_max = imgui.slider_int(f"Max Position##{axis_key}", max_pos, 0, 9999, f"%d (100%% limit)")
                    if changed:
                        axis_data["max_position"] = new_max
                        settings_changed = True
                        # Real-time preview: move to max position
                        self._preview_axis_position(axis_key, new_max, f"Previewing {axis_label} maximum")

                    # Range validation
                    if new_min >= new_max:
                        imgui.text_colored("Warning: Min must be less than Max", 1.0, 0.5, 0.0)

                    # Preview buttons for testing limits
                    imgui.text("Test Range:")
                    if imgui.button(f"Test Min##{axis_key}"):
                        self._preview_axis_position(axis_key, new_min, f"Testing {axis_label} minimum (0%)")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Move to minimum position (funscript 0%)")

                    imgui.same_line()
                    if imgui.button(f"Test Max##{axis_key}"):
                        self._preview_axis_position(axis_key, new_max, f"Testing {axis_label} maximum (100%)")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Move to maximum position (funscript 100%)")

                    imgui.same_line()
                    if imgui.button(f"Center##{axis_key}"):
                        center_pos = (new_min + new_max) // 2
                        self._preview_axis_position(axis_key, center_pos, f"Centering {axis_label} (50%)")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Move to center position (funscript 50%)")

                    # Speed multiplier
                    speed_mult = axis_data.get("speed_multiplier", 1.0)
                    changed, new_speed = imgui.slider_float(f"Speed Multiplier##{axis_key}", speed_mult, 0.1, 3.0, "%.2f")
                    if changed:
                        axis_data["speed_multiplier"] = new_speed
                        settings_changed = True

                    # Invert checkbox with preview
                    invert = axis_data.get("invert", False)
                    changed, new_invert = imgui.checkbox(f"Invert Direction##{axis_key}", invert)
                    if changed:
                        axis_data["invert"] = new_invert
                        settings_changed = True
                        # Preview inversion by showing the effect
                        if new_invert:
                            # Show inverted max (funscript 0% -> device max)
                            self._preview_axis_position(axis_key, new_max, f"Previewing {axis_label} INVERTED: funscript 0% \u2192 device max")
                        else:
                            # Show normal min (funscript 0% -> device min)
                            self._preview_axis_position(axis_key, new_min, f"Previewing {axis_label} NORMAL: funscript 0% \u2192 device min")

                    # Pattern simulation buttons
                    imgui.separator()
                    imgui.text(f"{axis_label} Simulation Patterns:")

                    # Row 1: Basic patterns
                    if imgui.button(f"Demo Range##{axis_key}"):
                        self._demo_axis_range(axis_key, new_min, new_max, new_invert, axis_label)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Test min \u2192 max \u2192 center positions")

                    imgui.same_line()
                    if imgui.button(f"Sine Wave##{axis_key}"):
                        self._simulate_axis_pattern(axis_key, "sine_wave", new_min, new_max, new_invert, axis_label)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Smooth sine wave motion")

                    imgui.same_line()
                    if imgui.button(f"Square Wave##{axis_key}"):
                        self._simulate_axis_pattern(axis_key, "square_wave", new_min, new_max, new_invert, axis_label)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Sharp min/max transitions")

                    # Row 2: Complex patterns
                    if imgui.button(f"Triangle Wave##{axis_key}"):
                        self._simulate_axis_pattern(axis_key, "triangle_wave", new_min, new_max, new_invert, axis_label)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Linear ramp up/down motion")

                    imgui.same_line()
                    if imgui.button(f"Random Pattern##{axis_key}"):
                        self._simulate_axis_pattern(axis_key, "random", new_min, new_max, new_invert, axis_label)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Random positions for testing")

                    imgui.same_line()
                    if imgui.button(f"Pulse Pattern##{axis_key}"):
                        self._simulate_axis_pattern(axis_key, "pulse", new_min, new_max, new_invert, axis_label)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Quick pulses from center")

                    # Smoothing
                    smoothing = axis_data.get("smoothing_factor", 0.8)
                    changed, new_smoothing = imgui.slider_float(f"Smoothing##{axis_key}", smoothing, 0.0, 1.0, "%.2f")
                    if changed:
                        axis_data["smoothing_factor"] = new_smoothing
                        settings_changed = True

                    # Pattern generation settings for this axis
                    imgui.separator()
                    imgui.text(f"{axis_label} Pattern Generation:")

                    # Pattern type dropdown (generalized for all axes)
                    pattern_types = ["disabled", "wave", "follow", "auto", "random_noise"]
                    pattern_labels = ["Disabled", "Wave (Smooth)", "Follow Primary", "Auto-Select", "Random Noise"]
                    current_pattern = axis_data.get("pattern_type", "disabled")
                    current_pattern_index = pattern_types.index(current_pattern) if current_pattern in pattern_types else 0

                    changed, new_pattern_index = imgui.combo(f"Pattern Type##{axis_key}", current_pattern_index, pattern_labels)
                    if changed and 0 <= new_pattern_index < len(pattern_types):
                        axis_data["pattern_type"] = pattern_types[new_pattern_index]
                        settings_changed = True

                    # Pattern intensity (only if not disabled)
                    if axis_data.get("pattern_type", "disabled") != "disabled":
                        intensity = axis_data.get("pattern_intensity", 1.0)
                        changed, new_intensity = imgui.slider_float(f"Pattern Intensity##{axis_key}", intensity, 0.0, 2.0, "%.2f")
                        if changed:
                            axis_data["pattern_intensity"] = new_intensity
                            settings_changed = True

                        frequency = axis_data.get("pattern_frequency", 1.0)
                        changed, new_frequency = imgui.slider_float(f"Pattern Frequency##{axis_key}", frequency, 0.1, 5.0, "%.2f")
                        if changed:
                            axis_data["pattern_frequency"] = new_frequency
                            settings_changed = True

                        # Follow strength (only for follow and auto modes)
                        if axis_data.get("pattern_type") in ("follow", "auto"):
                            follow_strength = axis_data.get("follow_strength", 0.5)
                            changed, new_fs = imgui.slider_float(f"Follow Strength##{axis_key}", follow_strength, 0.0, 1.0, "%.2f")
                            if changed:
                                axis_data["follow_strength"] = new_fs
                                settings_changed = True
                            _tooltip_if_hovered("How closely this axis follows the primary axis movement")

                    imgui.unindent(20)

                imgui.separator()

            # Global settings
            imgui.text("Global Settings:")

            # Update rate
            update_rate = profile_data.get("update_rate_hz", 20.0)
            changed, new_rate = imgui.slider_float("Update Rate (Hz)", update_rate, 5.0, 50.0, "%.1f")
            if changed:
                profile_data["update_rate_hz"] = new_rate
                settings_changed = True

            # Safety limits
            safety_enabled = profile_data.get("safety_limits_enabled", True)
            changed, new_safety = imgui.checkbox("Safety Limits Enabled", safety_enabled)
            if changed:
                profile_data["safety_limits_enabled"] = new_safety
                settings_changed = True

            # Apply button
            imgui.separator()
            if imgui.button("Apply Configuration"):
                self._load_osr_profile_to_device(current_profile_name, profile_data)
                settings_changed = True

            imgui.same_line()
            if imgui.button("Test Axis Movement"):
                self._test_osr_axes()

            # Save settings if changed
            if settings_changed:
                osr_profiles[current_profile_name] = profile_data
                self.app.app_settings.set("device_control_osr_profiles", osr_profiles)
                self.app.app_settings.save_settings()

        except Exception as e:
            imgui.text_colored(f"Error in OSR configuration: {e}", 1.0, 0.0, 0.0)

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

    def _test_osr_axes(self):
        """Test OSR axes with a simple movement pattern."""
        try:
            import threading
            def run_test():
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._test_osr_movement_async())
                finally:
                    loop.close()

            thread = threading.Thread(target=run_test, daemon=True)
            thread.start()
        except Exception as e:
            self.app.logger.error(f"Failed to start OSR test: {e}")

    async def _test_osr_movement_async(self):
        """Test OSR movement pattern."""
        try:
            backend = self.device_manager.get_connected_backend()
            if backend and hasattr(backend, 'set_position_with_profile'):
                # Test pattern: 0 -> 50 -> 100 -> 50 -> 0
                test_positions = [0.0, 50.0, 100.0, 50.0, 0.0]

                import asyncio
                for pos in test_positions:
                    await backend.set_position_with_profile(pos)
                    await asyncio.sleep(1.0)  # Hold position for 1 second

                self.app.logger.info("OSR axis test completed")

        except Exception as e:
            self.app.logger.error(f"OSR test movement failed: {e}")


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

    def _test_handy_movement(self):
        """Test Handy device movement."""
        try:
            import threading
            def test_handy_async():
                import asyncio

                async def run_test():
                    try:
                        self.app.logger.info(f"Device manager has _handy_backend: {hasattr(self.device_manager, '_handy_backend')}")
                        if hasattr(self.device_manager, '_handy_backend'):
                            self.app.logger.info(f"_handy_backend value: {self.device_manager._handy_backend}")

                        if not hasattr(self.device_manager, '_handy_backend') or not self.device_manager._handy_backend:
                            self.app.logger.error("No Handy connected")
                            return

                        backend = self.device_manager._handy_backend
                        self.app.logger.info(f"Backend type: {type(backend)}")
                        self.app.logger.info(f"Backend connected: {backend.is_connected()}")
                        self.app.logger.info("Testing Handy movement...")

                        # Test sequence: position, duration_ms (short durations for immediate testing)
                        positions = [(20, 50), (80, 50), (50, 50), (30, 50), (70, 50), (50, 50)]

                        for i, (pos, duration) in enumerate(positions):
                            try:
                                self.app.logger.info(f"   Calling set_position_enhanced({pos}, duration_ms={duration})")
                                success = await backend.set_position_enhanced(
                                    primary=pos,
                                    duration_ms=duration,
                                    movement_type="test"
                                )
                                self.app.logger.info(f"   set_position_enhanced returned: {success}")

                                if success:
                                    self.app.logger.info(f"   Step {i+1}/{len(positions)}: Position {pos}% in {duration}ms")
                                else:
                                    self.app.logger.error(f"   Step {i+1} failed")

                                # Wait for movement to complete
                                await asyncio.sleep(duration / 1000.0 + 0.2)

                            except Exception as e:
                                self.app.logger.error(f"   Step {i+1} error: {e}")
                                import traceback
                                self.app.logger.error(f"   Traceback: {traceback.format_exc()}")

                        # Return to center
                        try:
                            await backend.stop()
                            self.app.logger.info("Handy movement test complete")
                        except Exception as e:
                            self.app.logger.error(f"Failed to stop Handy: {e}")

                    except Exception as e:
                        self.app.logger.error(f"Handy test failed: {e}")

                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(run_test())
                finally:
                    loop.close()

            thread = threading.Thread(target=test_handy_async, daemon=True)
            thread.start()

        except Exception as e:
            self.app.logger.error(f"Failed to start Handy test: {e}")

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

        # Track upload hash per timeline for stale-script detection
        if not hasattr(self, '_handy_uploaded_timelines'):
            self._handy_uploaded_timelines = {}
        self._handy_uploaded_timelines[timeline_num] = len(actions)
        # Legacy compat
        self._handy_last_upload_hash = len(actions)

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
        imgui.indent(10)

        try:
            # Check if bleak is available
            ossm_available = 'ossm' in (self.device_manager.available_backends if self.device_manager else {})

            if not ossm_available:
                imgui.text_colored("OSSM backend unavailable", 0.7, 0.5, 0.0)
                imgui.text("Install bleak: pip install bleak>=0.21.0")
                imgui.unindent(10)
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

        imgui.unindent(10)

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
