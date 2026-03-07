"""Device Control — OSSMPanel methods."""
import asyncio
import threading
import imgui
from application.utils.imgui_helpers import tooltip_if_hovered as _tooltip_if_hovered
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.section_card import section_card as _section_card
from application.utils import primary_button_style, destructive_button_style


class OSSMPanelMixin:
    """Mixin fragment for DeviceControlMixin."""

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

        def run_ossm_scan():
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

        def connect_async():
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

        def disconnect_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.device_manager.disconnect_ossm())
            finally:
                loop.close()

        threading.Thread(target=disconnect_async, daemon=True).start()


    def _send_ossm_command(self, cmd):
        """Send a command to the OSSM device."""

        def send_async():
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

        def set_async():
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

        def test_async():

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
