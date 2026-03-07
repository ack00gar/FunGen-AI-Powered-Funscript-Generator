"""Device Control — OSRPanel methods."""
import asyncio
import threading
import time
import imgui
from application.utils.imgui_helpers import tooltip_if_hovered as _tooltip_if_hovered
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.section_card import section_card as _section_card
from application.utils import primary_button_style, destructive_button_style


class OSRPanelMixin:
    """Mixin fragment for DeviceControlMixin."""

    def _render_osr_controls(self):
        """Render OSR device controls."""
        # Check OSR connection status
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        is_osr_connected = self._is_device_type_connected("osr")

        if is_osr_connected:
            self._status_indicator(f"Connected to {connected_device.device_id}", "ready", "OSR device connected and ready")

            imgui.spacing()
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
            def run_test():
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


    def _scan_osr_devices(self):
        """Scan for OSR devices specifically."""
        try:
            def run_osr_scan():
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

