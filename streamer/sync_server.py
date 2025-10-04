#!/usr/bin/env python3
"""
Universal Sync Server

WebSocket server that broadcasts VideoProcessor state to all clients.
Handles bidirectional control (clients can play/pause/seek).
Sends funscript actions as T-code commands for devices.

Protocol:
- Server broadcasts: {type: "sync", frame_index, is_playing, timestamp_ms, fps}
- Server sends: {type: "tcode", frame_index, command, timestamp_ms}
- Client sends: {type: "control", action: "play/pause/seek/stop", frame: 1234}
"""

import asyncio
import websockets
import json
import logging
import threading
import time
from typing import Set, Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class SyncState:
    """Current playback synchronization state."""
    frame_index: int
    timestamp_ms: float
    is_playing: bool
    is_paused: bool
    fps: float
    total_frames: int
    video_path: str = ""
    video_filename: str = ""
    tracking_mode: str = "off"  # "off", "funscript", or tracker name
    tracker_name: str = ""
    device_name: str = ""
    device_connected: bool = False
    device_protocol: str = ""
    vr_projection: str = ""  # "fisheye", "he" (equirectangular hemisphere), etc.
    vr_fov: int = 0  # Field of view (180, 190, 200)


class UniversalSyncServer:
    """
    Universal synchronization server for video playback and device control.

    Broadcasts VideoProcessor state to all connected clients (browsers, devices, etc.)
    Handles control commands from any client with proper conflict resolution.
    """

    def __init__(self, video_processor, host: str = '0.0.0.0', port: int = 8765, xbvr_host: str = None, xbvr_port: int = 9999, video_server=None):
        """
        Initialize sync server.

        Args:
            video_processor: VideoProcessor instance (master timeline)
            host: Server bind address
            port: WebSocket port
            xbvr_host: XBVR server host (optional)
            xbvr_port: XBVR server port
            video_server: VideoHTTPServer instance for XBVR path updates (optional)
        """
        self.video_processor = video_processor
        self.logger = video_processor.logger

        # Server configuration
        self.host = host
        self.port = port

        # Video HTTP server reference (for XBVR path updates)
        self.video_server = video_server

        # XBVR integration
        self.xbvr_client = None
        if xbvr_host:
            from streamer.xbvr_client import XBVRClient
            self.xbvr_client = XBVRClient(host=xbvr_host, port=xbvr_port, logger=self.logger)

        # Server state
        self.is_running = False
        self.server_thread = None
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.client_metadata: Dict[str, Dict[str, Any]] = {}  # client_id → metadata

        # Synchronization state
        self.broadcast_interval = 1.0 / 30.0  # 30 Hz sync updates
        self.sync_task = None
        self.device_control_task = None
        self.loop = None

        # Frame drift tracking
        self.target_frame_index = None  # Target frame from browser
        self.last_drift_update_time = 0

        # Device control state
        self.last_device_position = None

        # Funscript update tracking
        self.last_funscript_update_time = 0
        self.last_funscript_action_count = 0

        # Device control bridge (optional - supporters only)
        self.streamer_device_bridge = None

        # Statistics
        self.stats = {
            'clients_connected': 0,
            'sync_messages_sent': 0,
            'tcode_commands_sent': 0,
            'control_commands_received': 0,
            'frames_skipped': 0
        }

    async def start_server(self):
        """Start the WebSocket server."""
        try:
            self.is_running = True

            # Start WebSocket server
            ws_server = await websockets.serve(
                self._handle_websocket_client,
                self.host,
                self.port,
                ping_interval=2,
                ping_timeout=10,
                max_size=1024 * 1024  # 1MB max message
            )

            self.logger.info(f"🚀 Universal Sync Server started on ws://{self.host}:{self.port}")
            self.logger.info(f"📡 Clients can connect to: ws://{self._get_local_ip()}:{self.port}")

            # Start sync broadcast loop
            self.sync_task = asyncio.create_task(self._sync_broadcast_loop())

            # Start device control loop (browser-time-based)
            self.device_control_task = asyncio.create_task(self._device_control_loop())

            # Keep server running
            await asyncio.Future()  # Run forever

        except Exception as e:
            self.logger.error(f"Failed to start sync server: {e}")
            self.is_running = False
            raise

    def start(self):
        """Start server in background thread."""
        if self.is_running:
            self.logger.warning("Sync server already running")
            return

        def run_server():
            """Run server in asyncio event loop."""
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_until_complete(self.start_server())
            except Exception as e:
                self.logger.error(f"Sync server error: {e}")
            finally:
                self.loop.close()

        self.server_thread = threading.Thread(target=run_server, daemon=True, name='SyncServer')
        self.server_thread.start()

        self.logger.info("✅ Universal Sync Server started in background")

    def stop(self):
        """Stop the sync server."""
        if not self.is_running:
            return

        self.logger.info("Stopping Universal Sync Server...")
        self.is_running = False

        # Close all connections
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._close_all_connections(), self.loop)

        # Wait for thread
        if self.server_thread and self.server_thread.is_alive():
            if self.loop:
                self.loop.call_soon_threadsafe(self.loop.stop)
            self.server_thread.join(timeout=2.0)

        self.logger.info("✅ Universal Sync Server stopped")

    async def _close_all_connections(self):
        """Close all active WebSocket connections."""
        for client in list(self.websocket_clients):
            try:
                await client.close()
            except:
                pass
        self.websocket_clients.clear()

    async def _handle_websocket_client(self, websocket):
        """Handle WebSocket client connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.websocket_clients.add(websocket)
        self.stats['clients_connected'] = len(self.websocket_clients)

        self.logger.info(f"🔌 Client connected: {client_id}")

        try:
            # Send initial state
            await self._send_sync_state(websocket)

            # Handle incoming messages (control commands)
            async for message in websocket:
                await self._handle_client_message(websocket, client_id, message)

        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"🔌 Client disconnected: {client_id}")
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.websocket_clients.discard(websocket)
            self.stats['clients_connected'] = len(self.websocket_clients)
            if client_id in self.client_metadata:
                del self.client_metadata[client_id]

    async def _handle_client_message(self, websocket, client_id: str, message):
        """Handle control messages from client."""
        try:
            if isinstance(message, str):
                cmd = json.loads(message)
                cmd_type = cmd.get('type')

                if cmd_type == 'control':
                    await self._handle_control_command(websocket, client_id, cmd)
                    self.stats['control_commands_received'] += 1
                elif cmd_type == 'ping':
                    await websocket.send(json.dumps({'type': 'pong', 'timestamp': time.time()}))
                elif cmd_type == 'metadata':
                    # Store client metadata (device type, capabilities, etc.)
                    self.client_metadata[client_id] = cmd.get('data', {})
                    self.logger.info(f"Client {client_id} metadata: {self.client_metadata[client_id]}")
                elif cmd_type == 'request_capabilities':
                    # Send available trackers and devices
                    await self._send_capabilities(websocket)
                    # Also send funscript data if available
                    await self._send_funscript_data(websocket)
                elif cmd_type == 'live_track':
                    # Handle live tracker control
                    await self._handle_live_track(websocket, client_id, cmd)
                elif cmd_type == 'device_control':
                    # Handle device enable/disable
                    await self._handle_device_control(websocket, client_id, cmd)
                elif cmd_type == 'transform':
                    # Handle zoom/pan transform
                    await self._handle_transform(websocket, client_id, cmd)
                elif cmd_type == 'set_speed_mode':
                    # Handle speed mode change
                    await self._handle_speed_mode(websocket, client_id, cmd)
                elif cmd_type == 'start_live_tracking':
                    # Handle start live tracking from browser
                    await self._handle_start_live_tracking(websocket, client_id, cmd)
                elif cmd_type == 'stop_live_tracking':
                    # Handle stop live tracking from browser
                    await self._handle_stop_live_tracking(websocket, client_id, cmd)
                elif cmd_type == 'toggle_video_playback_control':
                    # Handle toggle video playback control setting
                    await self._handle_toggle_video_playback_control(websocket, client_id, cmd)
                elif cmd_type == 'xbvr_discover':
                    # Handle XBVR auto-discovery
                    await self._handle_xbvr_discover(websocket, client_id, cmd)
                elif cmd_type == 'xbvr_get_scenes':
                    # Handle XBVR scene list request
                    await self._handle_xbvr_get_scenes(websocket, client_id, cmd)
                elif cmd_type == 'xbvr_get_scene_details':
                    # Handle XBVR scene details request
                    await self._handle_xbvr_get_scene_details(websocket, client_id, cmd)
                elif cmd_type == 'xbvr_load_video':
                    # Handle load video from XBVR
                    await self._handle_xbvr_load_video(websocket, client_id, cmd)
                elif cmd_type == 'frame_position':
                    # Handle frame position update from browser for device sync
                    frame_index = cmd.get('frame_index')
                    is_playing = cmd.get('is_playing', True)

                    if frame_index is not None:
                        self.update_target_frame(frame_index)

                        # Log large frame jumps (indicates seek) for debugging
                        if self.target_frame_index is not None:
                            drift = abs(frame_index - self.video_processor.current_frame_index)
                            if drift > 300:  # More than 5 seconds at 60fps
                                self.logger.debug(f"📍 Browser seek detected: frame {frame_index} (drift: {drift} frames, playing: {is_playing})")
                elif cmd_type == 'query_funscript':
                    # Handle funscript data query for time range
                    await self._handle_funscript_query(websocket, client_id, cmd)

        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON from client {client_id}")
        except Exception as e:
            self.logger.error(f"Error handling message from {client_id}: {e}")

    async def _handle_control_command(self, websocket, client_id: str, cmd: dict):
        """Handle playback control commands from client."""
        action = cmd.get('action')
        self.logger.info(f"🎮 Control command from {client_id}: {action}")

        try:
            if action == 'play':
                # Ensure VideoProcessor is in the correct speed mode before playing
                speed_mode = cmd.get('speed_mode', 'realtime')  # Get current browser speed mode
                await self._ensure_speed_mode(speed_mode)

                # Start/resume playback
                frame = cmd.get('frame')
                if frame is not None:
                    self.video_processor.start_processing(start_frame=frame)
                elif self.video_processor.is_processing and self.video_processor.pause_event.is_set():
                    self.video_processor.pause_event.clear()
                else:
                    self.video_processor.start_processing()
                self.logger.info(f"▶️ Play command executed (speed mode: {speed_mode})")

            elif action == 'pause':
                if self.video_processor.is_processing:
                    self.video_processor.pause_event.set()
                self.logger.info(f"⏸️ Pause command executed")

            elif action == 'seek':
                frame = cmd.get('frame')
                if frame is not None:
                    # Check if live tracking is active before seeking
                    tracking_was_active = False
                    tracker_name = None

                    if hasattr(self.video_processor, 'app'):
                        app = self.video_processor.app
                        if hasattr(app, 'app_state_ui') and hasattr(app, 'tracker'):
                            # Check if tracking is active by checking tracker.tracking_active property
                            tracking_was_active = (
                                self.video_processor.is_processing and
                                hasattr(app.tracker, 'tracking_active') and
                                app.tracker.tracking_active
                            )
                            tracker_name = app.app_state_ui.selected_tracker_name if tracking_was_active else None

                            self.logger.info(f"🔍 Seek tracking check: is_processing={self.video_processor.is_processing}, "
                                           f"has_tracker_obj={hasattr(app, 'tracker')}, "
                                           f"has_tracking_active={hasattr(app.tracker, 'tracking_active') if hasattr(app, 'tracker') else False}, "
                                           f"tracking_active={app.tracker.tracking_active if hasattr(app, 'tracker') and hasattr(app.tracker, 'tracking_active') else False}, "
                                           f"tracking_was_active={tracking_was_active}, tracker_name={tracker_name}")

                    # Perform the seek (this will stop tracking)
                    self.video_processor.seek_video(frame)
                    self.logger.info(f"⏩ Seek to frame {frame}")

                    # If tracking was active, restart it from new position
                    if tracking_was_active and tracker_name:
                        self.logger.info(f"🔄 Restarting tracking from frame {frame} with {tracker_name}")
                        if hasattr(app, 'event_handlers'):
                            app.event_handlers.handle_start_live_tracker_click()
                            self.logger.info(f"✅ Tracking restarted after seek")
                    else:
                        self.logger.info(f"ℹ️  Not restarting tracking (was_active={tracking_was_active}, tracker={tracker_name})")

            elif action == 'stop':
                self.video_processor.stop_processing()
                self.video_processor.seek_video(0)
                self.logger.info(f"⏹️ Stop command executed")

            # Immediately broadcast updated state
            await self._broadcast_sync_state()

        except Exception as e:
            self.logger.error(f"Error executing control command: {e}")

    async def _sync_broadcast_loop(self):
        """Background task that broadcasts sync state periodically."""
        self.logger.info("🔄 Sync broadcast loop started")

        while self.is_running:
            try:
                await asyncio.sleep(self.broadcast_interval)

                if self.websocket_clients:
                    await self._broadcast_sync_state()

                    # Periodically update funscript data if live tracking is active
                    # Update every 1 second during live tracking for near-realtime graph updates
                    current_time = time.time()
                    if current_time - self.last_funscript_update_time >= 1.0:  # Every 1 second
                        await self._broadcast_funscript_updates()
                        self.last_funscript_update_time = current_time

            except Exception as e:
                self.logger.error(f"Error in sync broadcast loop: {e}")
                await asyncio.sleep(1.0)

        self.logger.info("🔄 Sync broadcast loop stopped")

    async def _broadcast_sync_state(self):
        """Broadcast current sync state to all clients."""
        if not self.websocket_clients:
            # No clients connected - disable streamer device control
            if self.streamer_device_bridge:
                self.streamer_device_bridge.set_streaming_active(False)
            return

        state = self._get_sync_state()
        message = json.dumps(asdict(state))

        # Update device bridge with current playback state
        if self.streamer_device_bridge and len(self.websocket_clients) > 0:
            # Clients are connected - enable streamer device control
            self.streamer_device_bridge.set_streaming_active(True)

            # Update browser playback state for device sync
            current_time_ms = int((state.frame_index / state.fps) * 1000) if state.fps > 0 else 0
            self.streamer_device_bridge.update_browser_state(state.is_playing, current_time_ms)

        # Broadcast to all clients
        disconnected = []
        for client in list(self.websocket_clients):
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(client)

        # Remove disconnected clients
        for client in disconnected:
            self.websocket_clients.discard(client)

        if len(self.websocket_clients) > 0:
            self.stats['sync_messages_sent'] += 1
        else:
            # All clients just disconnected - disable streamer device control
            if self.streamer_device_bridge:
                self.streamer_device_bridge.set_streaming_active(False)

    async def _send_sync_state(self, websocket):
        """Send current sync state to a specific client."""
        state = self._get_sync_state()
        message = json.dumps(asdict(state))
        await websocket.send(message)

    def _get_sync_state(self) -> SyncState:
        """Get current synchronization state from VideoProcessor."""
        is_playing = self.video_processor.is_processing and not self.video_processor.pause_event.is_set()
        is_paused = self.video_processor.is_processing and self.video_processor.pause_event.is_set()

        # Get video filename
        video_filename = ""
        if self.video_processor.video_path:
            import os
            video_filename = os.path.basename(self.video_processor.video_path)

        # Get tracking mode and tracker info
        tracking_mode = "off"
        tracker_name = ""
        if hasattr(self.video_processor, 'app'):
            app = self.video_processor.app
            if hasattr(app, 'tracker') and app.tracker:
                if hasattr(app.tracker, 'tracking_active') and app.tracker.tracking_active:
                    # Live tracking
                    if hasattr(app, 'app_state_ui'):
                        tracker_name = app.app_state_ui.selected_tracker_name or "Tracking"
                        tracking_mode = tracker_name
                elif hasattr(app.tracker, 'funscript') and app.tracker.funscript:
                    # Funscript mode
                    tracking_mode = "funscript"

        # Get device info
        device_name = ""
        device_connected = False
        device_protocol = ""
        if hasattr(self.video_processor, 'app'):
            app = self.video_processor.app
            if hasattr(app, 'device_manager') and app.device_manager:
                device_connected = app.device_manager.is_connected()
                if device_connected:
                    # Get device info from backend
                    backend = app.device_manager.backend
                    if backend and hasattr(backend, 'connected_devices'):
                        devices = backend.connected_devices
                        if devices and len(devices) > 0:
                            first_device = devices[0]
                            device_name = first_device.get('name', 'Unknown')
                            # Determine protocol from backend type
                            backend_type = type(backend).__name__
                            if 'Buttplug' in backend_type:
                                device_protocol = 'Buttplug'
                            elif 'OSR' in backend_type:
                                device_protocol = 'OSR'
                            else:
                                device_protocol = 'Unknown'

        # Get VR metadata from video processor
        vr_projection = ""
        vr_fov = 0
        if hasattr(self.video_processor, 'vr_input_format') and self.video_processor.vr_input_format:
            # Extract base projection type (remove _sbs, _tb suffixes)
            vr_format = self.video_processor.vr_input_format.replace('_sbs', '').replace('_tb', '')
            vr_projection = vr_format  # fisheye, he (hemisphere equirect), etc.
        if hasattr(self.video_processor, 'vr_fov'):
            vr_fov = self.video_processor.vr_fov

        return SyncState(
            frame_index=self.video_processor.current_frame_index,
            timestamp_ms=time.time() * 1000,
            is_playing=is_playing,
            is_paused=is_paused,
            fps=self.video_processor.fps if hasattr(self.video_processor, 'fps') else 30.0,
            total_frames=self.video_processor.total_frames if hasattr(self.video_processor, 'total_frames') else 0,
            video_path=self.video_processor.video_path if hasattr(self.video_processor, 'video_path') else "",
            video_filename=video_filename,
            tracking_mode=tracking_mode,
            tracker_name=tracker_name,
            device_name=device_name,
            device_connected=device_connected,
            device_protocol=device_protocol,
            vr_projection=vr_projection,
            vr_fov=vr_fov
        )

    def update_target_frame(self, target_frame: int):
        """
        Update target frame from browser for drift tracking.

        Args:
            target_frame: Frame index that browser is currently at
        """
        self.target_frame_index = target_frame
        self.last_drift_update_time = time.time()

    def get_frame_drift(self) -> int:
        """
        Calculate frame drift (how many frames behind we are).

        Returns:
            Positive number if VideoProcessor is behind, negative if ahead, 0 if no target
        """
        if self.target_frame_index is None:
            return 0

        current_frame = self.video_processor.current_frame_index
        drift = self.target_frame_index - current_frame
        return drift

    def should_skip_frame(self) -> bool:
        """
        Determine if VideoProcessor should skip the next frame based on drift.

        Returns:
            True if frame should be skipped to catch up
        """
        drift = self.get_frame_drift()

        # Skip if behind by 3 or more frames
        if drift >= 3:
            self.stats['frames_skipped'] += 1
            return True

        return False

    async def _send_capabilities(self, websocket):
        """Send available trackers and devices to client."""
        trackers = []
        devices = []

        # Get available live trackers (same method as Control Panel UI)
        try:
            from application.gui_components.dynamic_tracker_ui import DynamicTrackerUI
            tracker_ui = DynamicTrackerUI()

            # Use the same method as Control Panel for consistency
            display_names, internal_names = tracker_ui.get_simple_mode_trackers()

            for display_name, internal_name in zip(display_names, internal_names):
                trackers.append({
                    'name': display_name,
                    'display_name': display_name,
                    'internal_name': internal_name,
                    'description': ''  # Description not needed for dropdown
                })

            self.logger.debug(f"Found {len(trackers)} live trackers")
        except Exception as e:
            self.logger.error(f"Failed to get trackers: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        # Get connected devices (if device_control available)
        try:
            from device_control.device_manager import DeviceManager

            # Try multiple sources for device_manager (same as Control Panel)
            device_manager = None

            # 1. Check app reference (primary source, same as Control Panel)
            if hasattr(self.video_processor, 'app') and hasattr(self.video_processor.app, 'device_manager'):
                device_manager = self.video_processor.app.device_manager
                self.logger.debug("Using device_manager from app")

            # 2. Check video_processor (fallback)
            elif hasattr(self.video_processor, 'device_manager') and self.video_processor.device_manager:
                device_manager = self.video_processor.device_manager
                self.logger.debug("Using device_manager from video_processor")

            # 3. Check Control Panel UI (last resort)
            elif (hasattr(self.video_processor, 'app') and
                  hasattr(self.video_processor.app, 'gui') and
                  hasattr(self.video_processor.app.gui, 'control_panel_ui') and
                  hasattr(self.video_processor.app.gui.control_panel_ui, 'device_manager')):
                device_manager = self.video_processor.app.gui.control_panel_ui.device_manager
                self.logger.debug("Using device_manager from Control Panel UI")
            else:
                self.logger.debug("No device_manager found")

            if device_manager:
                is_connected = device_manager.is_connected()

                if is_connected:
                    connected = device_manager.connected_devices

                    # Map device backend types to icons
                    device_icons = {
                        'handy': '🎯',
                        'osr': '🔧',
                        'buttplug': '🔌',
                        'intiface': '🔌'
                    }

                    # connected_devices[device_id] = backend instance (not DeviceInfo)
                    for device_id, backend in connected.items():
                        # Get device info from backend
                        device_info = backend.get_device_info() if hasattr(backend, 'get_device_info') else None

                        if device_info:
                            device_name = device_info.name
                        else:
                            device_name = device_id

                        # Determine backend type from backend class name
                        backend_class = backend.__class__.__name__.lower()
                        if 'osr' in backend_class:
                            backend_type = 'osr'
                        elif 'buttplug' in backend_class:
                            backend_type = 'buttplug'
                        elif 'handy' in backend_class:
                            backend_type = 'handy'
                        else:
                            backend_type = 'unknown'

                        devices.append({
                            'id': device_id,
                            'name': device_name,
                            'icon': device_icons.get(backend_type, '🎮'),
                            'backend': backend_type
                        })

                    self.logger.debug(f"Found {len(devices)} connected devices: {[d['name'] for d in devices]}")
                else:
                    self.logger.debug("Device manager exists but no devices connected")
            else:
                self.logger.debug("Device manager not available")
        except ImportError:
            self.logger.debug("Device control module not available (supporter feature)")
        except Exception as e:
            self.logger.error(f"Error getting devices: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        # Send capabilities response
        await websocket.send(json.dumps({
            'type': 'capabilities',
            'trackers': trackers,
            'devices': devices
        }))

    async def _send_funscript_data(self, websocket):
        """Send funscript data to browser if available."""
        try:
            primary_actions = None
            secondary_actions = None

            # Source 1: Live tracking funscript (highest priority)
            tracker = self.video_processor.tracker
            if tracker and hasattr(tracker, 'funscript'):
                try:
                    # Access the actions directly from the DualAxisFunscript object
                    primary_actions = tracker.funscript.primary_actions if tracker.funscript.primary_actions else None
                    secondary_actions = tracker.funscript.secondary_actions if tracker.funscript.secondary_actions else None
                    if primary_actions or secondary_actions:
                        self.logger.debug(f"Using live tracking funscript: {len(primary_actions) if primary_actions else 0} primary, {len(secondary_actions) if secondary_actions else 0} secondary actions")
                except Exception as e:
                    self.logger.error(f"Could not get live tracking funscript: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())

            # Source 2: Pre-loaded funscript (fallback if no live tracking)
            if not primary_actions and not secondary_actions:
                if hasattr(self.video_processor, 'app') and hasattr(self.video_processor.app, 'funscript_processor'):
                    fs_proc = self.video_processor.app.funscript_processor
                    primary_actions = fs_proc.get_actions('primary')
                    secondary_actions = fs_proc.get_actions('secondary')
                    if primary_actions or secondary_actions:
                        self.logger.info(f"📊 Using pre-loaded funscript: {len(primary_actions) if primary_actions else 0} primary, {len(secondary_actions) if secondary_actions else 0} secondary actions")

            # Check if we have any data
            has_data = bool(primary_actions) or bool(secondary_actions)

            if has_data:
                # Send funscript data
                await websocket.send(json.dumps({
                    'type': 'funscript_data',
                    'primary': primary_actions if primary_actions else [],
                    'secondary': secondary_actions if secondary_actions else [],
                    'has_primary': bool(primary_actions),
                    'has_secondary': bool(secondary_actions)
                }))
            else:
                self.logger.info("📊 No funscript data available")
                # Send empty funscript data
                await websocket.send(json.dumps({
                    'type': 'funscript_data',
                    'primary': [],
                    'secondary': [],
                    'has_primary': False,
                    'has_secondary': False
                }))

        except Exception as e:
            self.logger.error(f"Error sending funscript data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    async def _broadcast_funscript_updates(self):
        """Broadcast funscript updates to all clients (for live tracking)."""
        if not self.websocket_clients:
            return

        try:
            # Check if live tracking is active
            tracker = self.video_processor.tracker
            if not tracker or not hasattr(tracker, 'funscript'):
                return

            # Get current action count from direct attribute access
            primary_actions = tracker.funscript.primary_actions
            current_count = len(primary_actions) if primary_actions else 0

            # Broadcast if action count changed OR during active live tracking (to show partial data)
            # This ensures the graph updates frequently during live generation
            if current_count != self.last_funscript_action_count or current_count > 0:
                self.last_funscript_action_count = current_count

                # Broadcast to all clients
                for client in list(self.websocket_clients):
                    try:
                        await self._send_funscript_data(client)
                    except Exception as e:
                        self.logger.debug(f"Could not send funscript update to client: {e}")

        except Exception as e:
            self.logger.debug(f"Error broadcasting funscript updates: {e}")

    async def _handle_live_track(self, websocket, client_id: str, cmd: dict):
        """Handle live tracker enable/disable."""
        data = cmd.get('data', {})
        enabled = data.get('enabled', False)
        tracker = data.get('tracker')

        self.logger.info(f"🔴 Live track from {client_id}: {enabled}, tracker: {tracker}")

        # Enable/disable live tracking
        # The actual implementation depends on how live tracking is managed in VideoProcessor
        # For now, just log the command - this will be implemented when bridging to device_control
        if enabled and tracker:
            self.logger.info(f"Would enable live tracking with tracker: {tracker}")
            # Future: self.video_processor.enable_live_tracking(tracker)
        else:
            self.logger.info("Would disable live tracking")
            # Future: self.video_processor.disable_live_tracking()

    async def _handle_device_control(self, websocket, client_id: str, cmd: dict):
        """Handle device enable/disable."""
        data = cmd.get('data', {})
        device = data.get('device')
        enabled = data.get('enabled', False)

        self.logger.info(f"🎮 Device control from {client_id}: {device} = {enabled}")

        # Enable/disable device
        # The device_control module manages devices independently
        # This command can be used to enable/disable specific devices for live tracking
        try:
            if hasattr(self.video_processor, 'device_manager') and self.video_processor.device_manager:
                # For now, just log - actual per-device enable/disable
                # would need to be implemented in device_manager
                if enabled:
                    self.logger.info(f"Would enable device: {device}")
                    # Future: self.video_processor.device_manager.enable_device(device)
                else:
                    self.logger.info(f"Would disable device: {device}")
                    # Future: self.video_processor.device_manager.disable_device(device)
            else:
                self.logger.warning("Device manager not available on video_processor")
        except Exception as e:
            self.logger.error(f"Error controlling device: {e}")

    async def _handle_transform(self, websocket, client_id: str, cmd: dict):
        """Handle zoom/pan transform from client."""
        zoom = cmd.get('zoom', 1.0)
        pan_x = cmd.get('pan_x', 0)
        pan_y = cmd.get('pan_y', 0)

        self.logger.debug(f"🔍 Transform from {client_id}: zoom={zoom:.2f}, pan=({pan_x}, {pan_y})")

        # Store transform state in client metadata for potential broadcasting to other clients
        # (e.g., synchronizing zoom/pan across multiple viewers)
        if client_id not in self.client_metadata:
            self.client_metadata[client_id] = {}

        self.client_metadata[client_id].update({
            'zoom': zoom,
            'pan_x': pan_x,
            'pan_y': pan_y
        })

        # Future: Could broadcast transform to other clients for synchronized viewing
        # or store in VideoProcessor for display in main UI

    async def _ensure_speed_mode(self, mode: str):
        """Ensure VideoProcessor is in the specified speed mode."""
        try:
            from config.constants import ProcessingSpeedMode

            if hasattr(self.video_processor, 'app') and hasattr(self.video_processor.app, 'app_state_ui'):
                app_state_ui = self.video_processor.app.app_state_ui

                # Map browser mode to ProcessingSpeedMode constant
                if mode == 'realtime':
                    target_mode = ProcessingSpeedMode.REALTIME
                elif mode == 'slomo':
                    target_mode = ProcessingSpeedMode.SLOW_MOTION
                else:
                    self.logger.warning(f"⚠️ Unknown speed mode: {mode}, defaulting to realtime")
                    target_mode = ProcessingSpeedMode.REALTIME

                # Only update if different
                if app_state_ui.selected_processing_speed_mode != target_mode:
                    app_state_ui.selected_processing_speed_mode = target_mode
                    self.logger.info(f"✅ VideoProcessor speed mode set to {target_mode.name}")
            else:
                self.logger.warning("⚠️ Cannot set speed mode: app_state_ui not accessible")

        except Exception as e:
            self.logger.error(f"Error ensuring speed mode: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    async def _handle_speed_mode(self, websocket, client_id: str, cmd: dict):
        """Handle speed mode change from client."""
        mode = cmd.get('mode', 'realtime')  # 'realtime' or 'slomo'

        self.logger.info(f"⚡ Speed mode from {client_id}: {mode}")

        await self._ensure_speed_mode(mode)

    async def _handle_start_live_tracking(self, websocket, client_id: str, cmd: dict):
        """Handle start live tracking command from browser."""
        tracker = cmd.get('tracker')
        device = cmd.get('device')

        self.logger.info(f"▶️ Start live tracking from {client_id}: tracker={tracker}, device={device}")

        try:
            if not tracker:
                self.logger.warning("⚠️ No tracker specified for live tracking")
                return

            # Access app and event handlers
            if hasattr(self.video_processor, 'app'):
                app = self.video_processor.app

                # Set the selected tracker
                if hasattr(app, 'app_state_ui'):
                    app.app_state_ui.selected_tracker_name = tracker
                    self.logger.info(f"✅ Set tracker to: {tracker}")

                # Call the start live tracking handler
                if hasattr(app, 'event_handlers'):
                    app.event_handlers.handle_start_live_tracker_click()
                    self.logger.info("✅ Live tracking started")

                    # Wait a moment for tracker to initialize, then send funscript data
                    await asyncio.sleep(0.5)
                    await self._send_funscript_data(websocket)
                else:
                    self.logger.error("⚠️ Event handlers not available")
            else:
                self.logger.error("⚠️ App not accessible from video_processor")

        except Exception as e:
            self.logger.error(f"Error starting live tracking: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    async def _handle_stop_live_tracking(self, websocket, client_id: str, cmd: dict):
        """Handle stop live tracking command from browser."""
        self.logger.info(f"⏹️ Stop live tracking from {client_id}")

        try:
            # Access app and event handlers
            if hasattr(self.video_processor, 'app'):
                app = self.video_processor.app

                # Call the abort/stop process handler
                if hasattr(app, 'event_handlers'):
                    app.event_handlers.handle_abort_process_click()
                    self.logger.info("✅ Live tracking stopped")
                else:
                    self.logger.error("⚠️ Event handlers not available")
            else:
                self.logger.error("⚠️ App not accessible from video_processor")

        except Exception as e:
            self.logger.error(f"Error stopping live tracking: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    async def _handle_toggle_video_playback_control(self, websocket, client_id: str, cmd: dict):
        """Handle toggle video playback control setting from browser."""
        enabled = cmd.get('enabled', False)

        self.logger.info(f"🎮 Toggle video playback control from {client_id}: {enabled}")

        try:
            # Access app settings
            if hasattr(self.video_processor, 'app'):
                app = self.video_processor.app

                if hasattr(app, 'app_settings'):
                    # Update the setting
                    app.app_settings.set("device_control_video_playback", enabled)
                    app.app_settings.save_settings()
                    self.logger.info(f"✅ Video playback control {'enabled' if enabled else 'disabled'}")

                    # Update the video playback control (same as Control Panel does)
                    if hasattr(app, 'gui') and hasattr(app.gui, 'control_panel_ui'):
                        control_panel = app.gui.control_panel_ui
                        if hasattr(control_panel, '_update_video_playback_control'):
                            control_panel._update_video_playback_control(enabled)
                            self.logger.info("✅ Applied video playback control update")
                else:
                    self.logger.error("⚠️ App settings not available")
            else:
                self.logger.error("⚠️ App not accessible from video_processor")

        except Exception as e:
            self.logger.error(f"Error toggling video playback control: {e}")

    async def _handle_funscript_query(self, websocket, client_id: str, cmd: dict):
        """Handle funscript data query for a specific time range."""
        start_ms = cmd.get('start_ms', 0)
        end_ms = cmd.get('end_ms', 0)
        axis = cmd.get('axis', 'primary')

        try:
            # Get tracker and funscript
            tracker = self.video_processor.tracker
            if not tracker or not hasattr(tracker, 'funscript'):
                # No funscript data available
                await websocket.send(json.dumps({
                    'type': 'funscript_response',
                    'actions': [],
                    'tracked_up_to_ms': 0,
                    'tracking_active': False
                }))
                return

            # Get actions in requested range
            actions = tracker.funscript.get_actions_in_range(start_ms, end_ms, axis)

            # Calculate how far tracking has progressed
            fps = self.video_processor.fps if self.video_processor.fps > 0 else 30.0
            tracked_up_to_ms = int(self.video_processor.current_frame_index * (1000.0 / fps))

            # Send response
            await websocket.send(json.dumps({
                'type': 'funscript_response',
                'actions': actions,
                'tracked_up_to_ms': tracked_up_to_ms,
                'tracking_active': tracker.tracking_active if hasattr(tracker, 'tracking_active') else False,
                'axis': axis
            }))

        except Exception as e:
            self.logger.error(f"Error handling funscript query: {e}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())

    async def _device_control_loop(self):
        """
        Device control loop synchronized to browser playback position.

        This loop runs continuously and sends T-code commands to connected devices
        based on the browser's current playback position, NOT the VideoProcessor's
        frame index. This ensures devices stay in sync with what the user sees in VR.

        Works for both:
        - Live tracking: Reads funscript being generated in real-time
        - Pre-existing funscript: Reads loaded funscript data
        """
        self.logger.info("🎮 Device control loop started (browser-time-based)")

        while self.is_running:
            try:
                # Dynamic update rate based on video FPS (realtime sync)
                fps = self.video_processor.fps if self.video_processor.fps > 0 else 30.0
                update_interval = 1.0 / fps

                await asyncio.sleep(update_interval)

                # Only send commands if browser is providing position updates
                if self.target_frame_index is None:
                    continue

                # Calculate browser's current time in milliseconds
                browser_time_ms = int((self.target_frame_index / fps) * 1000)

                # Get funscript positions at browser's current time (primary + secondary)
                positions = await self._get_funscript_positions(browser_time_ms)

                if positions['primary'] is not None or positions['secondary'] is not None:
                    # Only send if position changed (avoid spamming devices)
                    if self.last_device_position is None or abs(positions.get('primary', 50) - self.last_device_position) > 1:
                        await self._send_to_devices(positions, browser_time_ms)
                        self.last_device_position = positions.get('primary', 50)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Device control loop error: {e}")
                await asyncio.sleep(0.5)  # Backoff on error

        self.logger.info("🎮 Device control loop stopped")

    async def _get_funscript_positions(self, time_ms: int) -> Dict[str, Optional[int]]:
        """
        Get funscript positions at specific time for both axes.

        Checks multiple sources:
        1. Live tracking funscript (tracker.funscript)
        2. Pre-loaded funscript (app.funscript_processor)

        Returns:
            Dict with 'primary' (up/down) and 'secondary' (roll) positions (0-100) or None
        """
        result = {'primary': None, 'secondary': None}

        try:
            # Source 1: Live tracking funscript
            tracker = self.video_processor.tracker
            if tracker and hasattr(tracker, 'funscript'):
                primary_pos = tracker.funscript.get_value(time_ms, axis='primary')
                secondary_pos = tracker.funscript.get_value(time_ms, axis='secondary')

                if primary_pos is not None:
                    result['primary'] = primary_pos
                if secondary_pos is not None:
                    result['secondary'] = secondary_pos

                # If we got data from live tracking, return immediately
                if result['primary'] is not None or result['secondary'] is not None:
                    return result

            # Source 2: Pre-loaded funscript
            if hasattr(self.video_processor, 'app'):
                app = self.video_processor.app
                if hasattr(app, 'funscript_processor'):
                    fs_proc = app.funscript_processor

                    # Get primary axis
                    primary_actions = fs_proc.get_actions('primary')
                    if primary_actions:
                        result['primary'] = self._find_closest_position(primary_actions, time_ms)

                    # Get secondary axis (roll)
                    secondary_actions = fs_proc.get_actions('secondary')
                    if secondary_actions:
                        result['secondary'] = self._find_closest_position(secondary_actions, time_ms)

            return result

        except Exception as e:
            self.logger.error(f"Error getting funscript positions: {e}")
            return result

    def _find_closest_position(self, actions: list, time_ms: int, max_time_diff: int = 500) -> Optional[int]:
        """
        Find position of action closest to given time.

        Args:
            actions: List of actions [{'at': ms, 'pos': 0-100}, ...]
            time_ms: Target time in milliseconds
            max_time_diff: Maximum time difference to consider (ms)

        Returns:
            Position (0-100) or None if no action within max_time_diff
        """
        if not actions:
            return None

        # Binary search for closest action (actions are sorted by 'at')
        import bisect
        timestamps = [action['at'] for action in actions]
        idx = bisect.bisect_left(timestamps, time_ms)

        # Check action at idx and idx-1 to find closest
        candidates = []
        if idx < len(actions):
            candidates.append(actions[idx])
        if idx > 0:
            candidates.append(actions[idx - 1])

        if not candidates:
            return None

        # Find closest action
        closest = min(candidates, key=lambda a: abs(a['at'] - time_ms))
        time_diff = abs(closest['at'] - time_ms)

        if time_diff <= max_time_diff:
            return closest['pos']

        return None

    async def _send_to_devices(self, positions: Dict[str, Optional[int]], timestamp_ms: int):
        """
        Send position commands to connected T-Code devices (dual-axis support).

        Args:
            positions: Dict with 'primary' (up/down L0) and 'secondary' (roll R1) positions (0-100)
            timestamp_ms: Browser playback timestamp
        """
        try:
            # Get device manager from app
            device_manager = None
            if hasattr(self.video_processor, 'app'):
                device_manager = getattr(self.video_processor.app, 'device_manager', None)

            if not device_manager:
                return

            if not device_manager.is_connected():
                return

            # Get connected backend to check capabilities
            backend = device_manager.get_connected_backend() if hasattr(device_manager, 'get_connected_backend') else None

            # Send primary axis (up/down - L0)
            primary_pos = positions.get('primary')
            if primary_pos is not None:
                if asyncio.iscoroutinefunction(device_manager.update_position):
                    await device_manager.update_position(primary_pos)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, device_manager.update_position, primary_pos)

                self.stats['tcode_commands_sent'] += 1

            # Send secondary axis (roll - R1) if backend supports it
            secondary_pos = positions.get('secondary')
            if secondary_pos is not None and backend and hasattr(backend, 'set_position_with_profile'):
                # Multi-axis device (OSR2, SR6, etc.)
                # Use set_position_with_profile for secondary axis
                try:
                    if asyncio.iscoroutinefunction(backend.set_position_with_profile):
                        await backend.set_position_with_profile({
                            'primary': primary_pos if primary_pos is not None else 50,
                            'secondary': secondary_pos
                        })
                    else:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, backend.set_position_with_profile, {
                            'primary': primary_pos if primary_pos is not None else 50,
                            'secondary': secondary_pos
                        })

                    self.stats['tcode_commands_sent'] += 1

                except Exception as e:
                    # Secondary axis error (non-critical)
                    pass

        except Exception as e:
            # Don't spam logs with device errors
            if not hasattr(self, '_last_device_error_time'):
                self._last_device_error_time = 0

            current_time = time.time()
            if current_time - self._last_device_error_time > 5.0:  # Log every 5s max
                self.logger.error(f"Error sending to devices: {e}")
                self._last_device_error_time = current_time

    def _get_local_ip(self) -> str:
        """Get local network IP address."""
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"

    async def _handle_xbvr_discover(self, websocket, client_id: str, cmd: dict):
        """Handle XBVR auto-discovery request."""
        try:
            from streamer.xbvr_client import XBVRClient

            self.logger.info("🔍 Starting XBVR auto-discovery...")
            discovered = await XBVRClient.discover_xbvr_instances(timeout=1.0)

            self.logger.info(f"✅ Found {len(discovered)} XBVR instance(s)")

            await websocket.send(json.dumps({
                'type': 'xbvr_discovered',
                'instances': discovered
            }))

        except Exception as e:
            self.logger.error(f"Error during XBVR discovery: {e}")
            await websocket.send(json.dumps({
                'type': 'xbvr_discovered',
                'instances': [],
                'error': str(e)
            }))

    async def _handle_xbvr_get_scenes(self, websocket, client_id: str, cmd: dict):
        """Handle request for XBVR scene list."""
        if not self.xbvr_client:
            await websocket.send(json.dumps({
                'type': 'xbvr_scenes',
                'scenes': [],
                'error': 'XBVR not configured'
            }))
            return

        try:
            limit = cmd.get('limit', 50)
            offset = cmd.get('offset', 0)
            search = cmd.get('search', None)

            if search:
                scenes = await self.xbvr_client.search_scenes(search, limit=limit)
            else:
                scenes = await self.xbvr_client.get_scenes(limit=limit, offset=offset)

            # Simplify scene data for browser and check funscript availability
            # Show all scenes (XBVR's files array is often empty even when videos exist)
            simplified_scenes = []
            total_scenes = len(scenes)
            scenes_with_files = 0
            scenes_without_files = 0

            # Debug: Log first scene structure to understand XBVR response
            if scenes and len(scenes) > 0:
                sample = scenes[0]
                self.logger.info(f"📊 XBVR scene sample keys: {list(sample.keys())}")
                self.logger.info(f"   has 'file': {('file' in sample)}, has 'files': {('files' in sample)}")
                if 'file' in sample:
                    self.logger.info(f"   file type: {type(sample['file'])}, length: {len(sample['file']) if isinstance(sample['file'], list) else 'N/A'}")
                if 'files' in sample:
                    self.logger.info(f"   files type: {type(sample['files'])}, length: {len(sample['files']) if isinstance(sample['files'], list) else 'N/A'}")

            for scene in scenes:
                # Check if scene has file info
                # XBVR uses 'file' (singular) not 'files' (plural)
                files = scene.get('file', scene.get('files', []))
                has_file_info = files and len(files) > 0

                # Find first video file (type='video')
                video_file = None
                if has_file_info:
                    for f in files:
                        if f.get('type') == 'video':
                            video_file = f
                            break

                # Construct full path from video file
                video_path = None
                if video_file:
                    path = video_file.get('path', '')
                    filename = video_file.get('filename', '')
                    if path and filename:
                        video_path = f"{path}/{filename}"
                        has_file_info = True
                    else:
                        has_file_info = False

                if has_file_info:
                    scenes_with_files += 1
                else:
                    scenes_without_files += 1

                # Get preview URLs (try multiple field names XBVR might use)
                preview_url = (
                    scene.get('preview_video_url') or
                    scene.get('video_preview') or
                    scene.get('scene_preview') or
                    scene.get('preview_url')
                )

                # If scene has video preview path, construct full URL
                if preview_url and not preview_url.startswith('http'):
                    preview_url = f"{self.xbvr_client.base_url}{preview_url}"

                # First, search XBVR's file array for actual funscript files
                # This is the most reliable method - if there's a file, we know we have a funscript
                funscript_filename = None
                funscript_path = None
                has_funscript = False

                if files:
                    for f in files:
                        file_type = f.get('type', '')
                        filename = f.get('filename', '')
                        # Check if this is a funscript file
                        if file_type == 'script' or filename.endswith('.funscript'):
                            has_funscript = True
                            funscript_filename = filename
                            # Construct full path
                            path = f.get('path', '')
                            if path and filename:
                                funscript_path = f"{path}/{filename}"
                            break

                # Do NOT use XBVR's is_scripted/script_published fields as fallback
                # Those indicate scripts exist online (SLR, etc.) but not necessarily locally
                # Only trust the file array - if no funscript file found, scene has no script

                # Debug logging for specific scene
                if scene.get('title') == 'Queen of the Sculpture':
                    self.logger.info(f"🔍 DEBUG Queen scene: has_funscript={has_funscript}, filename={funscript_filename}, path={funscript_path}")

                scene_data = {
                    'id': scene.get('id'),
                    'title': scene.get('title'),
                    'studio': scene.get('studio'),
                    'site': scene.get('site'),
                    'release_date': scene.get('release_date'),
                    'duration': scene.get('duration'),
                    'cover_url': scene.get('cover_url') or scene.get('cover_image') or scene.get('image_url'),
                    'preview_video_url': preview_url,
                    'cast': scene.get('cast', []),
                    'performers': scene.get('performers', []),
                    'tags': scene.get('tags', []),
                    'star_rating': scene.get('star_rating') or scene.get('rating'),
                    'synopsis': scene.get('synopsis') or scene.get('description'),
                    'has_file': has_file_info,
                    'has_funscript': has_funscript,
                    'has_funscript_secondary': False,  # XBVR doesn't distinguish secondary
                    'is_scripted': has_funscript,  # Add XBVR's native field too
                    'funscript_filename': funscript_filename,  # Add funscript filename
                    'funscript_path': funscript_path,  # Add full funscript path
                    'file_info': {}
                }

                # Add file info if available
                if video_path and video_file:
                    scene_data['file_info'] = {
                        'path': video_path,
                        'resolution': video_file.get('video_width', 0),
                        'height': video_file.get('video_height', 0),
                        'size': video_file.get('size', 0),
                        'codec': video_file.get('video_codec_name', 'Unknown'),
                        'projection': video_file.get('projection', 'Unknown')
                    }

                simplified_scenes.append(scene_data)

            # Log first scene structure for debugging (if we have scenes)
            if scenes and len(scenes) > 0:
                first_scene = scenes[0]
                self.logger.info(f"📊 XBVR Scene structure (first scene keys): {list(first_scene.keys())}")
                # Log ALL files and their types for the first scene
                files = first_scene.get('file', first_scene.get('files', []))
                if files:
                    self.logger.info(f"📁 XBVR Files for first scene ({len(files)} files):")
                    for i, f in enumerate(files):
                        file_type = f.get('type', 'unknown')
                        filename = f.get('filename', 'unknown')
                        self.logger.info(f"   File {i+1}: type={file_type}, filename={filename}")
                        if i == 0:
                            self.logger.info(f"   File keys: {list(f.keys())}")
                # Log funscript detection for first scene
                self.logger.info(f"📜 Funscript detection (first scene): is_scripted={first_scene.get('is_scripted')}, script_published={first_scene.get('script_published')}, ai_script={first_scene.get('ai_script')}, human_script={first_scene.get('human_script')}")

                # Log details for a scene with is_scripted=True if we find one
                scripted_scene = next((s for s in scenes if s.get('is_scripted')), None)
                if scripted_scene:
                    self.logger.info(f"📜 Found scripted scene: {scripted_scene.get('title')}")
                    scripted_files = scripted_scene.get('file', scripted_scene.get('files', []))
                    self.logger.info(f"   Files for scripted scene ({len(scripted_files)} files):")
                    for i, f in enumerate(scripted_files):
                        file_type = f.get('type', 'unknown')
                        filename = f.get('filename', 'unknown')
                        self.logger.info(f"      File {i+1}: type={file_type}, filename={filename}")

            # Log statistics (only at debug level to reduce noise)
            self.logger.debug(f"XBVR scenes: {total_scenes} total, {scenes_with_files} with file info, {scenes_without_files} without, {len(simplified_scenes)} sent")

            # Determine if there are more scenes to load
            # If we fetched a full page (limit), there might be more
            has_more = len(scenes) >= limit

            await websocket.send(json.dumps({
                'type': 'xbvr_scenes',
                'scenes': simplified_scenes,
                'has_more': has_more,
                'offset': offset
            }))

        except Exception as e:
            self.logger.error(f"Error fetching XBVR scenes: {e}")
            await websocket.send(json.dumps({
                'type': 'xbvr_scenes',
                'scenes': [],
                'error': str(e)
            }))

    async def _handle_xbvr_get_scene_details(self, websocket, client_id: str, cmd: dict):
        """Handle request for detailed XBVR scene information including file paths."""
        if not self.xbvr_client:
            await websocket.send(json.dumps({
                'type': 'xbvr_scene_details',
                'error': 'XBVR not configured'
            }))
            return

        try:
            scene_id = cmd.get('scene_id')
            if not scene_id:
                await websocket.send(json.dumps({
                    'type': 'xbvr_scene_details',
                    'error': 'No scene_id provided'
                }))
                return

            self.logger.info(f"🔍 Fetching details for scene {scene_id}")

            # Get full scene details from XBVR
            scene = await self.xbvr_client.get_scene_details(scene_id)
            if not scene:
                await websocket.send(json.dumps({
                    'type': 'xbvr_scene_details',
                    'error': f'Scene {scene_id} not found'
                }))
                return

            # Process file info
            files = scene.get('file', scene.get('files', []))
            video_file = None
            if files:
                for f in files:
                    if f.get('type') == 'video':
                        video_file = f
                        break

            # Construct full path
            video_path = None
            file_info = {}
            if video_file:
                path = video_file.get('path', '')
                filename = video_file.get('filename', '')
                if path and filename:
                    video_path = f"{path}/{filename}"
                    file_info = {
                        'path': video_path,
                        'resolution': video_file.get('video_width', 0),
                        'height': video_file.get('video_height', 0),
                        'size': video_file.get('size', 0),
                        'codec': video_file.get('video_codec_name', 'Unknown')
                    }

            # First, search XBVR's file array for actual funscript files
            funscript_filename = None
            funscript_path = None
            has_funscript = False

            if files:
                for f in files:
                    file_type = f.get('type', '')
                    filename = f.get('filename', '')
                    # Check if this is a funscript file
                    if file_type == 'script' or filename.endswith('.funscript'):
                        has_funscript = True
                        funscript_filename = filename
                        path = f.get('path', '')
                        if path and filename:
                            funscript_path = f"{path}/{filename}"
                        self.logger.debug(f"📜 Funscript detected in XBVR files for scene {scene_id}: {filename}")
                        break

            # If no file found, fallback to XBVR's is_scripted field
            if not has_funscript:
                has_funscript = scene.get('is_scripted', False)
                # Also check for script_published, ai_script, or human_script fields
                if not has_funscript:
                    has_funscript = (
                        scene.get('script_published', False) or
                        scene.get('ai_script', False) or
                        scene.get('human_script', False)
                    )

            # Build response
            scene_data = {
                'id': scene.get('id'),
                'title': scene.get('title'),
                'studio': scene.get('studio'),
                'site': scene.get('site'),
                'release_date': scene.get('release_date'),
                'duration': scene.get('duration'),
                'cover_url': scene.get('cover_url') or scene.get('cover_image') or scene.get('image_url'),
                'cast': scene.get('cast', []),
                'performers': scene.get('performers', []),
                'tags': scene.get('tags', []),
                'star_rating': scene.get('star_rating') or scene.get('rating'),
                'synopsis': scene.get('synopsis') or scene.get('description'),
                'has_file': bool(video_path),
                'has_funscript': has_funscript,
                'has_funscript_secondary': False,  # XBVR doesn't distinguish secondary
                'is_scripted': has_funscript,
                'funscript_filename': funscript_filename,
                'funscript_path': funscript_path,
                'file_info': file_info
            }

            self.logger.info(f"✅ Scene {scene_id} details: has_file={bool(video_path)}, path={video_path}")

            await websocket.send(json.dumps({
                'type': 'xbvr_scene_details',
                'scene': scene_data
            }))

        except Exception as e:
            self.logger.error(f"Error fetching scene details: {e}")
            await websocket.send(json.dumps({
                'type': 'xbvr_scene_details',
                'error': str(e)
            }))

    async def _handle_xbvr_load_video(self, websocket, client_id: str, cmd: dict):
        """Handle load video from XBVR and optionally start live tracking."""
        if not self.xbvr_client:
            self.logger.error("XBVR not configured")
            await websocket.send(json.dumps({
                'type': 'xbvr_load_result',
                'success': False,
                'error': 'XBVR not configured'
            }))
            return

        try:
            scene_id = cmd.get('scene_id')
            tracker_name = cmd.get('tracker', None)
            video_path = cmd.get('video_path', None)
            funscript_path = cmd.get('funscript_path', None)  # Get funscript path from command

            # If video_path is provided, we don't need scene_id (URL-based flow)
            # Otherwise we need scene_id to query XBVR API
            if not video_path and not scene_id:
                self.logger.error("No scene_id or video_path provided")
                await websocket.send(json.dumps({
                    'type': 'xbvr_load_result',
                    'success': False,
                    'error': 'No scene_id or video_path provided'
                }))
                return

            # Use video_path from browser if provided (new behavior - URL-based flow)
            # Otherwise fall back to querying XBVR API (old behavior - WebSocket flow)
            if video_path:
                self.logger.info(f"📹 Using video path from browser: {video_path}")
            else:
                # Get scene files from XBVR API (fallback)
                self.logger.info(f"🔍 Fetching scene files for scene {scene_id}")
                files = await self.xbvr_client.get_scene_files(scene_id)
                if not files:
                    self.logger.error(f"No files found for scene {scene_id}")
                    await websocket.send(json.dumps({
                        'type': 'xbvr_load_result',
                        'success': False,
                        'error': 'No files found for scene'
                    }))
                    return

                # Use first available file (use full_path constructed by get_scene_files)
                video_path = files[0].get('full_path') if files else None
                if not video_path:
                    self.logger.error(f"No video path for scene {scene_id}")
                    await websocket.send(json.dumps({
                        'type': 'xbvr_load_result',
                        'success': False,
                        'error': 'No video path in scene data'
                    }))
                    return

            # Check if file exists
            from pathlib import Path
            if not Path(video_path).exists():
                self.logger.error(f"Video file not found: {video_path}")
                await websocket.send(json.dumps({
                    'type': 'xbvr_load_result',
                    'success': False,
                    'error': f'Video file not found: {video_path}'
                }))
                return

            self.logger.info(f"📹 Loading video from XBVR: {video_path}")

            # Load video in FunGen
            if hasattr(self.video_processor, 'app'):
                app = self.video_processor.app

                # Load video file using file_manager
                if hasattr(app, 'file_manager'):
                    success = app.file_manager.open_video_from_path(video_path)

                    if success:
                        self.logger.info(f"✅ Video loaded: {video_path}")

                        # Update video server path if video server is available
                        if self.video_server:
                            self.logger.info(f"📹 Updating video server path to: {video_path}")
                            update_success = self.video_server.update_video_path(video_path)

                            if update_success:
                                self.logger.info(f"✅ Video server path updated successfully")
                                # Delay AFTER confirmation to ensure browser receives updated path
                                await asyncio.sleep(0.2)  # Increased to 200ms for safety margin
                            else:
                                self.logger.error(f"❌ Failed to update video server path")
                                # Notify client of failure
                                await websocket.send(json.dumps({
                                    'type': 'xbvr_load_result',
                                    'success': False,
                                    'error': 'Failed to update video server path'
                                }))
                                return
                        else:
                            self.logger.warning(f"⚠️ Video HTTP server not available")
                            await websocket.send(json.dumps({
                                'type': 'xbvr_load_result',
                                'success': False,
                                'error': 'Video HTTP server not running'
                            }))
                            return

                        # Load funscript if provided or auto-detect
                        funscript_loaded = False
                        if hasattr(app, 'file_manager'):
                            # First try: Use funscript_path if provided (from XBVR browser)
                            if funscript_path:
                                funscript_file = Path(funscript_path)
                                if funscript_file.exists():
                                    try:
                                        app.file_manager.load_funscript_to_timeline(str(funscript_file), timeline_num=1)
                                        self.logger.info(f"✅ Loaded funscript from XBVR: {funscript_file.name}")
                                        funscript_loaded = True
                                    except Exception as e:
                                        self.logger.error(f"❌ Failed to load funscript {funscript_path}: {e}")
                                else:
                                    self.logger.warning(f"⚠️ Funscript file not found: {funscript_path}")

                            # Second try: Auto-detect based on video filename (fallback)
                            if not funscript_loaded:
                                video_path_obj = Path(video_path)
                                auto_funscript_path = video_path_obj.with_suffix('.funscript')
                                if auto_funscript_path.exists():
                                    try:
                                        app.file_manager.load_funscript_to_timeline(str(auto_funscript_path), timeline_num=1)
                                        self.logger.info(f"✅ Auto-loaded funscript: {auto_funscript_path.name}")
                                        funscript_loaded = True
                                    except Exception as e:
                                        self.logger.warning(f"Failed to auto-load funscript {auto_funscript_path}: {e}")

                            if not funscript_loaded and funscript_path:
                                self.logger.warning(f"⚠️ Could not load funscript - file may not exist or is invalid")

                        # Enable interactive control if funscript was loaded
                        if funscript_loaded and hasattr(app, 'app_state_ui'):
                            app.app_state_ui.interactive_control_enabled = True
                            self.logger.info(f"✅ Interactive control enabled automatically")

                        # Start live tracking if tracker specified
                        if tracker_name and hasattr(app, 'app_state_ui') and hasattr(app, 'event_handlers'):
                            self.logger.info(f"🎯 Setting up live tracking with tracker: {tracker_name}")

                            # Set selected tracker
                            app.app_state_ui.selected_tracker_name = tracker_name
                            self.logger.info(f"  ✓ Selected tracker set to: {app.app_state_ui.selected_tracker_name}")

                            # Ensure tracker is initialized
                            if hasattr(app, 'tracker') and app.tracker:
                                app.tracker.set_tracking_mode(tracker_name)
                                self.logger.info(f"  ✓ Tracker mode explicitly set to: {tracker_name}")
                            else:
                                self.logger.warning(f"  ⚠️ Tracker not found in app")

                            # Start live tracking
                            app.event_handlers.handle_start_live_tracker_click()
                            self.logger.info(f"✅ Started live tracking with {tracker_name}")

                            # Broadcast success - only XBVR browser needs to know (to open new tab)
                            # Don't reload existing streamer tabs - would interrupt tracking
                            success_msg = json.dumps({
                                'type': 'xbvr_load_result',
                                'success': True,
                                'video_path': video_path,
                                'tracker': tracker_name,
                                'message': f'Video loaded and tracking started with {tracker_name}',
                                'reload_video': False,  # Don't reload - would stop tracking
                                'open_streamer': True   # Signal XBVR browser to open new tab
                            })
                        else:
                            # Broadcast success - only XBVR browser needs to know (to open new tab)
                            success_msg = json.dumps({
                                'type': 'xbvr_load_result',
                                'success': True,
                                'video_path': video_path,
                                'message': 'Video loaded successfully',
                                'reload_video': False,
                                'open_streamer': True
                            })

                        # Broadcast to all connected clients
                        for client in list(self.websocket_clients):
                            try:
                                await client.send(success_msg)
                            except Exception as e:
                                self.logger.debug(f"Failed to send to client: {e}")
                    else:
                        self.logger.error(f"Failed to load video: {video_path}")
                        await websocket.send(json.dumps({
                            'type': 'xbvr_load_result',
                            'success': False,
                            'error': 'Failed to load video in FunGen'
                        }))
                else:
                    self.logger.error("file_manager not available")
                    await websocket.send(json.dumps({
                        'type': 'xbvr_load_result',
                        'success': False,
                        'error': 'file_manager not available'
                    }))
            else:
                self.logger.error("app not available on video_processor")
                await websocket.send(json.dumps({
                    'type': 'xbvr_load_result',
                    'success': False,
                    'error': 'app not available'
                }))

        except Exception as e:
            self.logger.error(f"Error loading video from XBVR: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            await websocket.send(json.dumps({
                'type': 'xbvr_load_result',
                'success': False,
                'error': str(e)
            }))
