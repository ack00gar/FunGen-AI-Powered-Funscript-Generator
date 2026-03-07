"""Device Control — ButtplugPanel methods."""
import asyncio
import threading
import imgui
from application.utils.imgui_helpers import tooltip_if_hovered as _tooltip_if_hovered
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.section_card import section_card as _section_card
from application.utils import primary_button_style, destructive_button_style


class ButtplugPanelMixin:
    """Mixin fragment for DeviceControlMixin."""

    def _render_buttplug_controls(self):
        """Render Buttplug.io device controls."""

        # Check Buttplug connection status
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        is_buttplug_connected = (self._is_device_type_connected("buttplug_linear")
                                  or self._is_device_type_connected("buttplug_vibrator"))

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


    def _discover_buttplug_devices(self):
        """Discover available Buttplug devices using current server settings."""
        try:
            def run_buttplug_discovery():
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
            def run_buttplug_connection():
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
            def run_status_check():
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


    def _open_intiface_download(self):
        """Open Intiface Central download page."""
        try:
            import webbrowser
            webbrowser.open("https://intiface.com/central/")
            self.app.logger.info("Opened Intiface Central download page")
        except Exception as e:
            self.app.logger.error(f"Failed to open Intiface download page: {e}")

