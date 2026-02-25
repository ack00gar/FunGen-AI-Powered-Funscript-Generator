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
            self.device_manager = DeviceManager(config)

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
            # Connected state
            self._status_indicator(f"Connected to {connected_device.name}", "ready", "Handy connected and ready")

            # Upload Funscript button (auto-uploads on play, but manual option if script changed)
            has_funscript = (hasattr(self.app, 'funscript_processor') and
                           self.app.funscript_processor and
                           self.app.funscript_processor.get_actions('primary'))

            if has_funscript:
                if imgui.button("Re-upload Funscript##HandyUpload", width=-1):
                    self._upload_funscript_to_handy()
                _tooltip_if_hovered("Re-upload funscript if you made changes (auto-uploads on first play)")
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

            # Sync settings
            imgui.text("Sync Settings:")
            imgui.indent(10)

            current_offset = self.app.app_settings.get("device_control_handy_sync_offset_ms", 0)

            # Row 1: -50 / -10 / -1 buttons, then slider, then +1 / +10 / +50 buttons
            # Fine adjustment buttons (left side)
            if imgui.button("-50##SyncMinus50"):
                new_offset = max(-1000, current_offset - 50)
                self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                self._apply_handy_hstp_offset(new_offset)
            _tooltip_if_hovered("-50ms")
            imgui.same_line()

            if imgui.button("-10##SyncMinus10"):
                new_offset = max(-1000, current_offset - 10)
                self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                self._apply_handy_hstp_offset(new_offset)
            _tooltip_if_hovered("-10ms")
            imgui.same_line()

            if imgui.button("-1##SyncMinus1"):
                new_offset = max(-1000, current_offset - 1)
                self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                self._apply_handy_hstp_offset(new_offset)
            _tooltip_if_hovered("-1ms")
            imgui.same_line()

            # Sync offset slider
            imgui.push_item_width(100)
            changed, value = imgui.slider_int(
                "##HandySyncOffset",
                current_offset,
                -1000, 1000
            )
            imgui.pop_item_width()
            if changed:
                self.app.app_settings.set("device_control_handy_sync_offset_ms", value)
                self._apply_handy_hstp_offset(value)
            _tooltip_if_hovered("Sync Offset (ms): + = Handy moves later, - = Handy moves earlier\nChanges apply instantly via Handy API")
            imgui.same_line()

            # Fine adjustment buttons (right side)
            if imgui.button("+1##SyncPlus1"):
                new_offset = min(1000, current_offset + 1)
                self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                self._apply_handy_hstp_offset(new_offset)
            _tooltip_if_hovered("+1ms")
            imgui.same_line()

            if imgui.button("+10##SyncPlus10"):
                new_offset = min(1000, current_offset + 10)
                self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                self._apply_handy_hstp_offset(new_offset)
            _tooltip_if_hovered("+10ms")
            imgui.same_line()

            if imgui.button("+50##SyncPlus50"):
                new_offset = min(1000, current_offset + 50)
                self.app.app_settings.set("device_control_handy_sync_offset_ms", new_offset)
                self._apply_handy_hstp_offset(new_offset)
            _tooltip_if_hovered("+50ms")

            # Row 2: Direct numeric input + current value display
            imgui.push_item_width(80)
            changed, input_value = imgui.input_int("##SyncOffsetInput", current_offset, 0, 0)
            imgui.pop_item_width()
            if changed:
                clamped_value = max(-1000, min(1000, input_value))
                self.app.app_settings.set("device_control_handy_sync_offset_ms", clamped_value)
                self._apply_handy_hstp_offset(clamped_value)
            _tooltip_if_hovered("Enter offset directly (ms)\n-1000 to +1000")
            imgui.same_line()
            imgui.text("ms")
            imgui.same_line()
            if current_offset >= 0:
                imgui.text_colored(f"(Handy +{current_offset}ms later)", 0.5, 0.8, 0.5, 1.0)
            else:
                imgui.text_colored(f"(Handy {current_offset}ms earlier)", 0.8, 0.5, 0.5, 1.0)

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
                            self.app.logger.error(f"Failed to connect to OSR device on {port_name}")
                    except Exception as e:
                        self.app.logger.error(f"Error in device connection loop: {e}")

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
                    pattern_types = ["disabled", "wave", "follow", "auto"]
                    pattern_labels = ["Disabled", "Wave (Smooth)", "Follow Primary", "Auto-Select"]
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

                        # Pattern frequency (only if not disabled)
                        frequency = axis_data.get("pattern_frequency", 1.0)
                        changed, new_frequency = imgui.slider_float(f"Pattern Frequency##{axis_key}", frequency, 0.1, 5.0, "%.2f")
                        if changed:
                            axis_data["pattern_frequency"] = new_frequency
                            settings_changed = True

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
                loop.run_until_complete(self.device_manager.connect_handy(connection_key))
            finally:
                loop.close()

        threading.Thread(target=connect_async, daemon=True).start()

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

    def _upload_funscript_to_handy(self):
        """Upload current funscript to Handy for HSSP streaming."""
        import threading
        import asyncio

        # Get funscript actions
        if not hasattr(self.app, 'funscript_processor') or not self.app.funscript_processor:
            self.app.logger.error("No funscript loaded")
            return

        primary_actions = self.app.funscript_processor.get_actions('primary')
        if not primary_actions:
            self.app.logger.error("No funscript actions available")
            return

        def upload_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self.device_manager.prepare_handy_for_video_playback(primary_actions)
                )
            finally:
                loop.close()

        threading.Thread(target=upload_async, daemon=True).start()
