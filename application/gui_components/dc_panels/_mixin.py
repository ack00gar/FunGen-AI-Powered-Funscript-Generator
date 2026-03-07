"""Device Control — DeviceControlCore methods."""
import asyncio
import threading
import traceback
import imgui
from application.utils.imgui_helpers import tooltip_if_hovered as _tooltip_if_hovered
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.section_card import section_card as _section_card
from application.utils import primary_button_style, destructive_button_style


class DeviceControlCoreMixin:
    """Mixin fragment for DeviceControlMixin."""

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

            # Auto-scan for serial (TCode) devices in the background so they
            # appear immediately in Quick Controls without a manual "Scan" click.
            self._auto_scan_serial_devices()

            self._device_control_initialized = True
            self.app.logger.info("Device Control initialized in Control Panel successfully")

        except Exception as e:
            self.app.logger.error(f"Failed to initialize Device Control: {e}")
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
            self.app.logger.error(f"Traceback: {traceback.format_exc()}")


    def _get_connected_device_type(self):
        """Return the device type string for the first connected device, or ''."""
        if not self.device_manager.is_connected():
            return ""
        for did in self.device_manager.connected_devices:
            return self.device_manager.get_device_type_for_id(did)
        return ""

    def _is_device_type_connected(self, device_type):
        """Check if any connected device matches the given type (multi-device aware)."""
        for did in self.device_manager.connected_devices:
            if self.device_manager.get_device_type_for_id(did) == device_type:
                return True
        return False


    def _render_device_control_content(self):
        """Render the main device control interface with improved UX."""
        # Guard: skip rendering while background disconnect thread is tearing down state
        if getattr(self, '_device_disconnecting', False):
            imgui.text("Disconnecting device...")
            return
        # Version info (top of tab, consistent with other supporter modules)
        self._render_addon_version_label("device_control", "Device Control")

        imgui.separator()

        _conn_type = self._get_connected_device_type()

        # Device Hub — always visible, replaces old compact status + quick controls
        with _section_card("Devices##DeviceHub", tier="primary", open_by_default=True) as is_open:
            if is_open:
                self._render_device_hub()

        # Quick controls when connected (test slider, bookmarks)
        if _conn_type:
            with _section_card("Quick Controls##QuickCtrl", tier="primary") as is_open:
                if is_open:
                    self._render_quick_controls()

        # Device type sections (detailed config per type)
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


    def _auto_scan_serial_devices(self):
        """Auto-scan serial ports for TCode devices in background thread."""

        def _scan():
            try:
                osr_backend = self.device_manager.available_backends.get('osr')
                if not osr_backend:
                    return
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    devices = loop.run_until_complete(osr_backend.discover_devices())
                    self._available_osr_ports = [
                        {
                            'device': d.device_id,
                            'description': d.name,
                            'manufacturer': getattr(d, 'manufacturer', 'Unknown'),
                        }
                        for d in devices
                    ]
                    self._osr_scan_performed = True
                    if devices:
                        self.app.logger.info(f"Auto-detected {len(devices)} serial device(s)")
                finally:
                    loop.close()
            except Exception as e:
                self.app.logger.debug(f"Auto-scan serial devices: {e}")

        threading.Thread(target=_scan, daemon=True).start()

    def _render_device_hub(self):
        """Render device hub: connected devices + detected (unconnected) devices."""
        connected = self.device_manager.get_connected_devices()  # Dict[str, DeviceInfo]
        control_source = self.device_manager.get_active_control_source()

        # ── Status line ──
        if control_source == 'streamer':
            imgui.text_colored("[STREAMER CONTROL]", 0.2, 0.5, 0.9)
        elif control_source == 'desktop':
            imgui.text_colored("[DESKTOP CONTROL]", 0.2, 0.7, 0.2)
        elif connected:
            imgui.text_colored("[IDLE]", 0.7, 0.7, 0.2)
        else:
            imgui.text_colored("No devices connected", 0.5, 0.5, 0.5)

        imgui.spacing()

        # ── Connected devices ──
        avail_w = imgui.get_content_region_available_width()

        if connected:
            for device_id, device_info in connected.items():
                device_type = self.device_manager.get_device_type_for_id(device_id)
                name = getattr(device_info, 'name', device_id)

                # Status dot + name
                imgui.text_colored("*", 0.2, 0.8, 0.2)  # green dot
                imgui.same_line()
                imgui.text(f"{name}")

                # RTD for Handy
                if device_type == "handy":
                    rtd = self.device_manager.get_handy_rtd_ms() if hasattr(self.device_manager, 'get_handy_rtd_ms') else 0
                    if rtd > 0:
                        imgui.same_line()
                        imgui.text_colored(f"({rtd}ms)", 0.6, 0.6, 0.6)

                # Disconnect button — right-aligned
                btn_label = f"Disconnect##{device_id}"
                btn_w = imgui.calc_text_size(btn_label.split("##")[0])[0] + imgui.get_style().frame_padding[0] * 2
                imgui.same_line(avail_w - btn_w)
                with destructive_button_style():
                    if imgui.small_button(btn_label):
                        if device_type == "handy":
                            self._disconnect_handy()
                        else:
                            self._disconnect_device_by_id(device_id)

            imgui.spacing()

        # ── Detected but not connected devices ──
        # Serial (TCode) devices
        connected_port_ids = set(connected.keys())
        unconnected_serial = [
            p for p in getattr(self, '_available_osr_ports', [])
            if p['device'] not in connected_port_ids
        ]

        if unconnected_serial:
            imgui.text_colored("Detected:", 0.6, 0.6, 0.6)
            for port_info in unconnected_serial:
                port_name = port_info.get('device', 'Unknown')
                description = port_info.get('description', '')
                imgui.text_colored("*", 0.6, 0.6, 0.2)  # yellow dot
                imgui.same_line()
                imgui.text(f"{description}")
                btn_label = f"Connect##{port_name}"
                btn_w = imgui.calc_text_size(btn_label.split("##")[0])[0] + imgui.get_style().frame_padding[0] * 2
                imgui.same_line(avail_w - btn_w)
                with primary_button_style():
                    if imgui.small_button(btn_label):
                        self._connect_osr_device(port_name)

        # Buttplug discovered but unconnected
        bp_devices = getattr(self, '_discovered_buttplug_devices', [])
        unconnected_bp = [d for d in bp_devices if d.device_id not in connected_port_ids]
        if unconnected_bp:
            for device_info in unconnected_bp:
                imgui.text_colored("*", 0.6, 0.6, 0.2)
                imgui.same_line()
                imgui.text(f"{device_info.name}")
                btn_label = f"Connect##{device_info.device_id}"
                btn_w = imgui.calc_text_size(btn_label.split("##")[0])[0] + imgui.get_style().frame_padding[0] * 2
                imgui.same_line(avail_w - btn_w)
                with primary_button_style():
                    if imgui.small_button(btn_label):
                        self._connect_specific_buttplug_device(device_info.device_id)

        # Handy (always show connect option if not connected)
        if not self._is_device_type_connected("handy"):
            connection_key = self.app.app_settings.get("handy_connection_key", "")
            if connection_key:
                imgui.text_colored("*", 0.6, 0.6, 0.2)
                imgui.same_line()
                imgui.text(f"Handy ({connection_key[:4]}...)")
                btn_label = "Connect##HandyHub"
                btn_w = imgui.calc_text_size("Connect")[0] + imgui.get_style().frame_padding[0] * 2
                imgui.same_line(avail_w - btn_w)
                with primary_button_style():
                    if imgui.small_button(btn_label):
                        self._connect_handy(connection_key)

        # ── Re-scan button ──
        imgui.spacing()
        if imgui.small_button("Rescan Serial##HubRescan"):
            self._scan_osr_devices()
        _tooltip_if_hovered("Re-scan serial ports for OSR/TCode devices")
        imgui.same_line()
        if imgui.small_button("Scan Buttplug##HubBPScan"):
            self._discover_buttplug_devices()
        _tooltip_if_hovered("Discover devices through Intiface Central")

    def _disconnect_device_by_id(self, device_id: str):
        """Disconnect a specific device by its ID."""
        try:
            self._device_disconnecting = True

            def run_disconnect():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            self.device_manager.disconnect(device_id))
                    finally:
                        loop.close()
                    self.app.logger.info(f"Disconnected {device_id}")
                except Exception as e:
                    self.app.logger.error(f"Error disconnecting {device_id}: {e}")
                finally:
                    self._device_disconnecting = False

            threading.Thread(target=run_disconnect, daemon=True).start()
        except Exception as e:
            self._device_disconnecting = False
            self.app.logger.error(f"Failed to disconnect {device_id}: {e}")

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


    def _disconnect_current_device(self):
        """Disconnect the currently connected device."""
        try:

            self._device_disconnecting = True

            def run_disconnect():
                try:
                    # Use device manager's worker loop if available
                    worker_loop = getattr(self.device_manager, '_worker_loop', None)
                    if worker_loop and worker_loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(self.device_manager.stop(), worker_loop)
                        future.result(timeout=10)
                    else:
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

