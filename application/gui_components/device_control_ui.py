"""
Device Control UI Panel

Comprehensive device control interface for FunGen.
Provides device discovery, connection, parameterization, live tracking,
and synchronized video playback with device control.

Only visible when device_control folder is present (supporter feature).
"""

import imgui
import asyncio
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Device control imports (conditional)
try:
    from device_control.device_manager import DeviceManager, DeviceControlConfig
    from device_control.device_parameterization import (
        get_parameter_manager, DeviceProfile, AxisType, AxisRange
    )
    from device_control.video_player_integration import create_video_player
    from device_control.live_tracking_integration import (
        create_live_tracking_processor, LiveTrackingConfig, TrackingMode
    )
    from device_control.ui_integration import create_device_control_system
    DEVICE_CONTROL_AVAILABLE = True
except ImportError:
    DEVICE_CONTROL_AVAILABLE = False


class DeviceControlUI:
    """
    Device Control UI Panel for FunGen.
    
    Features:
    - Device discovery and connection
    - Device parameter configuration (ranges, speeds, safety)
    - Live tracking integration
    - Video + funscript synchronized playback
    - Performance monitoring
    - Profile management
    """
    
    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger(__name__)
        
        # Check if device control is available
        self.available = self._check_availability()
        
        if not self.available:
            return
        
        # Initialize device control system
        self.device_manager: Optional[DeviceManager] = None
        self.parameter_manager = None
        self.video_player = None
        self.live_tracker = None
        
        # UI state
        self.show_device_panel = True
        self.show_parameter_panel = False
        self.show_live_tracking_panel = False
        self.show_video_player_panel = False
        self.show_debug_panel = False
        
        # Device discovery state
        self.discovered_devices: List = []
        self.discovery_in_progress = False
        self.last_discovery_time = 0
        
        # Connection state
        self.connection_status = "Disconnected"
        self.connected_device_info: Optional[Dict] = None
        
        # Parameter editing
        self.editing_profile: Optional[DeviceProfile] = None
        self.selected_axis = AxisType.STROKE
        
        # Live tracking state
        self.live_tracking_active = False
        self.tracking_stats = {}
        
        # Video player state
        self.video_loaded = False
        self.funscript_loaded = False
        self.playback_status = None
        
        # Performance monitoring
        self.device_stats = {}
        self.last_stats_update = 0
        
        # Initialize device control system
        self._initialize_device_system()
    
    def _check_availability(self) -> bool:
        """Check if device control features are available."""
        if not DEVICE_CONTROL_AVAILABLE:
            return False
        
        # Check if device_control folder exists
        device_control_path = Path("device_control")
        if not device_control_path.exists():
            return False
        
        return True
    
    def _initialize_device_system(self):
        """Initialize the device control system."""
        try:
            # Create device control system
            self.device_manager, live_bridge, funscript_bridge = create_device_control_system()
            self.parameter_manager = get_parameter_manager()
            
            # Create video player and live tracker
            self.video_player = create_video_player(self.device_manager)
            self.live_tracker = create_live_tracking_processor(self.device_manager)
            
            self.logger.info("Device control system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize device control system: {e}")
            self.available = False
    
    def render(self):
        """Render the device control UI."""
        # Check if window should be shown
        app_state = self.app.app_state_ui
        should_show = getattr(app_state, 'show_device_control_window', False)
        
        if not should_show:
            return
        
        # Always render window, but show different content based on availability
        if not self.available:
            self._render_supporter_only_message()
            return
        
        # Set window properties to ensure it's visible
        imgui.set_next_window_position(150, 150, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(900, 650, imgui.FIRST_USE_EVER)
        
        self.app.logger.debug("Device Control: About to call imgui.begin")
        
        # Main device control window with flags to ensure visibility
        window_flags = imgui.WINDOW_NO_COLLAPSE
        expanded, opened = imgui.begin("🎮 Device Control", True, window_flags)
        
        self.app.logger.debug(f"Device Control: imgui.begin returned expanded={expanded}, opened={opened}")
        
        if not opened:
            app_state.show_device_control_window = False
            self.app.logger.debug("Device Control: Window was closed by user")
        
        if expanded:
            self.app.logger.debug("Device Control: Rendering main panel")
            self._render_main_panel()
        else:
            self.app.logger.debug("Device Control: Window not expanded")
            
        imgui.end()
        self.app.logger.debug("Device Control: Finished rendering")
        
        # Individual panels (if enabled)
        if self.show_parameter_panel:
            self._render_parameter_panel()
        
        if self.show_live_tracking_panel:
            self._render_live_tracking_panel()
        
        if self.show_video_player_panel:
            self._render_video_player_panel()
        
        if self.show_debug_panel:
            self._render_debug_panel()
    
    def _render_main_panel(self):
        """Render the main device control panel."""
        # Panel tabs
        if imgui.begin_tab_bar("DeviceControlTabs"):
            
            # Device Connection Tab
            if imgui.begin_tab_item("Connection")[0]:
                self._render_connection_tab()
                imgui.end_tab_item()
            
            # Device Parameters Tab
            if imgui.begin_tab_item("Parameters")[0]:
                self._render_parameters_tab()
                imgui.end_tab_item()
            
            # Live Tracking Tab
            if imgui.begin_tab_item("Live Tracking")[0]:
                self._render_live_tracking_tab()
                imgui.end_tab_item()
            
            # Video Player Tab
            if imgui.begin_tab_item("Video Player")[0]:
                self._render_video_player_tab()
                imgui.end_tab_item()
            
            # Statistics Tab
            if imgui.begin_tab_item("Stats")[0]:
                self._render_statistics_tab()
                imgui.end_tab_item()
            
            imgui.end_tab_bar()
    
    def _render_connection_tab(self):
        """Render device connection tab."""
        # Connection status
        imgui.text("Connection Status:")
        imgui.same_line()
        
        if self.device_manager and self.device_manager.is_connected():
            imgui.text_colored("Connected", 0.0, 1.0, 0.0)
            if self.connected_device_info:
                imgui.text(f"Device: {self.connected_device_info.get('name', 'Unknown')}")
                imgui.text(f"Type: {self.connected_device_info.get('device_type', 'Unknown')}")
        else:
            imgui.text_colored("Disconnected", 1.0, 0.0, 0.0)
        
        imgui.separator()
        
        # Device discovery
        imgui.text("Device Discovery:")
        
        # Discovery controls
        if imgui.button("Discover Devices" if not self.discovery_in_progress else "Discovering..."):
            if not self.discovery_in_progress:
                asyncio.create_task(self._discover_devices())
        
        imgui.same_line()
        if imgui.button("Quick Handy Scan"):
            asyncio.create_task(self._quick_handy_scan())
        
        imgui.same_line()
        if self.discovery_in_progress:
            imgui.text_colored("Scanning network...", 1.0, 1.0, 0.0)
        else:
            imgui.text(f"Last scan: {time.time() - self.last_discovery_time:.1f}s ago" if self.last_discovery_time > 0 else "Never")
        
        # Handy setup help
        if not self.discovered_devices:
            imgui.separator()
            imgui.text_colored("Handy Setup Instructions:", 0.0, 1.0, 1.0)
            imgui.text("1. Make sure your Handy is connected to the same WiFi network")
            imgui.text("2. Check that local mode is enabled on your Handy")
            imgui.text("3. Try the 'Quick Handy Scan' button for faster detection")
            imgui.text("4. If not found, check your Handy's IP address and network")
            imgui.separator()
        
        # Discovered devices list
        if self.discovered_devices:
            imgui.text_colored("Discovered Devices:", 0.0, 1.0, 0.0)
            
            for i, device in enumerate(self.discovered_devices):
                device_name = device.name
                device_type = device.device_type.value
                
                # Device info with visual indicators
                icon = "🎮" if device_type == "stroker" else "🔧"
                if imgui.tree_node(f"{icon} {device_name} ({device_type})##device_{i}"):
                    imgui.text(f"ID: {device.device_id}")
                    imgui.text(f"Manufacturer: {device.manufacturer}")
                    imgui.text(f"Model: {device.model}")
                    
                    # Show connection info for Handy
                    if "handy" in device.device_id.lower():
                        if "local" in device.metadata.get("connection_mode", ""):
                            imgui.text_colored("✓ Local WiFi Connection", 0.0, 1.0, 0.0)
                            if "local_ip" in device.metadata:
                                imgui.text(f"IP: {device.metadata['local_ip']}")
                        firmware = device.metadata.get("firmware", "unknown")
                        imgui.text(f"Firmware: {firmware}")
                    
                    if device.capabilities:
                        caps = device.capabilities
                        imgui.text(f"Linear: {caps.supports_linear}")
                        imgui.text(f"Rotation: {caps.supports_rotation}")
                        imgui.text(f"Vibration: {caps.supports_vibration}")
                        imgui.text(f"Max Rate: {caps.max_position_rate_hz:.1f} Hz")
                    
                    # Connect button
                    if imgui.button(f"Connect to {device_name}##connect_{i}"):
                        asyncio.create_task(self._connect_device(device.device_id))
                    
                    imgui.tree_pop()
        else:
            imgui.text("No devices discovered yet.")
        
        imgui.separator()
        
        # Connection controls
        if self.device_manager and self.device_manager.is_connected():
            if imgui.button("Disconnect"):
                asyncio.create_task(self._disconnect_device())
            
            imgui.same_line()
            if imgui.button("Test Device"):
                asyncio.create_task(self._test_device())
    
    def _render_parameters_tab(self):
        """Render device parameters tab."""
        if not self.device_manager or not self.device_manager.is_connected():
            imgui.text("Connect a device to configure parameters")
            return
        
        # Active profile info
        active_profile = self.parameter_manager.get_active_profile()
        if active_profile:
            imgui.text(f"Active Profile: {active_profile.device_name}")
            imgui.text(f"Model: {active_profile.model}")
        else:
            imgui.text("No active profile")
            return
        
        imgui.separator()
        
        # Axis selection
        imgui.text("Select Axis:")
        axis_names = [axis.value for axis in AxisType if axis in active_profile.axes]
        current_axis_index = axis_names.index(self.selected_axis.value) if self.selected_axis.value in axis_names else 0
        
        changed, new_index = imgui.combo("##axis_select", current_axis_index, axis_names)
        if changed:
            self.selected_axis = AxisType(axis_names[new_index])
        
        # Axis configuration
        if self.selected_axis in active_profile.axes:
            axis_range = active_profile.axes[self.selected_axis]
            
            imgui.separator()
            imgui.text(f"Configuring: {self.selected_axis.value.title()}")
            
            # Range settings
            imgui.text("Range Settings:")
            changed, new_min = imgui.slider_float("Min Value", axis_range.min_value, 0.0, 100.0, "%.1f%%")
            if changed:
                axis_range.min_value = new_min
            
            changed, new_max = imgui.slider_float("Max Value", axis_range.max_value, 0.0, 100.0, "%.1f%%")
            if changed:
                axis_range.max_value = new_max
            
            changed, new_center = imgui.slider_float("Center Value", axis_range.center_value, 0.0, 100.0, "%.1f%%")
            if changed:
                axis_range.center_value = new_center
            
            # Safety limits
            imgui.separator()
            imgui.text("Safety Limits:")
            changed, new_safe_min = imgui.slider_float("Safe Min", axis_range.safe_min, 0.0, 100.0, "%.1f%%")
            if changed:
                axis_range.safe_min = new_safe_min
            
            changed, new_safe_max = imgui.slider_float("Safe Max", axis_range.safe_max, 0.0, 100.0, "%.1f%%")
            if changed:
                axis_range.safe_max = new_safe_max
            
            # Motion settings
            imgui.separator()
            imgui.text("Motion Settings:")
            changed, new_speed = imgui.slider_float("Max Speed", axis_range.max_speed, 10.0, 500.0, "%.1f%%/s")
            if changed:
                axis_range.max_speed = new_speed
            
            changed, new_accel = imgui.slider_float("Acceleration", axis_range.acceleration, 50.0, 1000.0, "%.1f%%/s²")
            if changed:
                axis_range.acceleration = new_accel
            
            changed, new_smoothing = imgui.slider_float("Smoothing", axis_range.smoothing, 0.0, 1.0, "%.3f")
            if changed:
                axis_range.smoothing = new_smoothing
            
            # Advanced settings
            imgui.separator()
            imgui.text("Advanced Settings:")
            changed, new_invert = imgui.checkbox("Invert Axis", axis_range.invert)
            if changed:
                axis_range.invert = new_invert
            
            # Response curve
            curve_types = ["linear", "exponential", "logarithmic"]
            current_curve_index = curve_types.index(axis_range.curve_type) if axis_range.curve_type in curve_types else 0
            changed, new_curve_index = imgui.combo("Response Curve", current_curve_index, curve_types)
            if changed:
                axis_range.curve_type = curve_types[new_curve_index]
            
            if axis_range.curve_type != "linear":
                changed, new_factor = imgui.slider_float("Curve Factor", axis_range.curve_factor, 0.1, 3.0, "%.2f")
                if changed:
                    axis_range.curve_factor = new_factor
        
        imgui.separator()
        
        # Profile management
        if imgui.button("Save Profile"):
            if active_profile:
                self.parameter_manager.save_profile(active_profile)
                self.app.logger.info("Device profile saved")
        
        imgui.same_line()
        if imgui.button("Reset to Defaults"):
            # Reset to default profile
            if self.connected_device_info:
                default_profile = self.parameter_manager.create_profile_for_device(self.connected_device_info)
                self.parameter_manager.profiles[active_profile.device_id] = default_profile
                self.parameter_manager.set_active_profile(active_profile.device_id)
    
    def _render_live_tracking_tab(self):
        """Render live tracking tab."""
        imgui.text("Live Tracking Integration")
        
        if not self.device_manager or not self.device_manager.is_connected():
            imgui.text("Connect a device to enable live tracking")
            return
        
        imgui.separator()
        
        # Live tracking controls
        if self.live_tracking_active:
            imgui.text_colored("Live Tracking: ACTIVE", 0.0, 1.0, 0.0)
            if imgui.button("Stop Live Tracking"):
                asyncio.create_task(self._stop_live_tracking())
        else:
            imgui.text("Live Tracking: INACTIVE")
            if imgui.button("Start Live Tracking"):
                asyncio.create_task(self._start_live_tracking())
        
        imgui.separator()
        
        # Live device control toggle
        self._render_live_device_control_toggle()
        
        imgui.separator()
        
        # Live tracking configuration
        imgui.text("Configuration:")
        
        # Tracking mode
        mode_names = [mode.value for mode in TrackingMode]
        current_mode = getattr(self.live_tracker.config, 'tracking_mode', TrackingMode.POSITION_ONLY)
        current_mode_index = mode_names.index(current_mode.value)
        
        changed, new_mode_index = imgui.combo("Tracking Mode", current_mode_index, mode_names)
        if changed:
            self.live_tracker.config.tracking_mode = TrackingMode(mode_names[new_mode_index])
        
        # Update rate
        changed, new_rate = imgui.slider_float("Update Rate", self.live_tracker.config.update_rate_hz, 10.0, 60.0, "%.1f Hz")
        if changed:
            self.live_tracker.config.update_rate_hz = new_rate
        
        # Smoothing
        changed, new_smoothing = imgui.slider_float("Smoothing", self.live_tracker.config.smoothing_factor, 0.0, 1.0, "%.3f")
        if changed:
            self.live_tracker.config.smoothing_factor = new_smoothing
        
        # Safety settings
        imgui.separator()
        imgui.text("Safety Settings:")
        
        changed, stop_on_loss = imgui.checkbox("Emergency Stop on Tracking Loss", self.live_tracker.config.emergency_stop_on_loss)
        if changed:
            self.live_tracker.config.emergency_stop_on_loss = stop_on_loss
        
        changed, new_confidence = imgui.slider_float("Min Confidence", self.live_tracker.config.confidence_threshold, 0.0, 1.0, "%.3f")
        if changed:
            self.live_tracker.config.confidence_threshold = new_confidence
        
        # Statistics
        if self.live_tracking_active and self.tracking_stats:
            imgui.separator()
            imgui.text("Live Tracking Statistics:")
            
            stats = self.tracking_stats
            imgui.text(f"Frames Processed: {stats.get('frames_processed', 0)}")
            imgui.text(f"Device Commands: {stats.get('device_commands_sent', 0)}")
            imgui.text(f"Avg Confidence: {stats.get('average_confidence', 0.0):.3f}")
            imgui.text(f"Avg Latency: {stats.get('average_latency_ms', 0.0):.1f}ms")
            imgui.text(f"Tracking Loss Events: {stats.get('tracking_loss_events', 0)}")
    
    def _render_video_player_tab(self):
        """Render video player tab."""
        imgui.text("Video + Funscript Player")
        
        if not self.device_manager or not self.device_manager.is_connected():
            imgui.text("Connect a device to enable video playback")
            return
        
        imgui.separator()
        
        # File loading
        imgui.text("Load Files:")
        
        if imgui.button("Load Video"):
            # This would integrate with the app's file dialog
            # For now, just show the concept
            imgui.text("Video loading would integrate with file dialog")
        
        imgui.same_line()
        if imgui.button("Load Funscript"):
            imgui.text("Funscript loading would integrate with file dialog")
        
        # Status
        imgui.separator()
        imgui.text("Status:")
        imgui.text(f"Video: {'Loaded' if self.video_loaded else 'Not loaded'}")
        imgui.text(f"Funscript: {'Loaded' if self.funscript_loaded else 'Not loaded'}")
        
        # Playback controls
        if self.video_loaded and self.funscript_loaded:
            imgui.separator()
            imgui.text("Playback Controls:")
            
            if imgui.button("Play"):
                asyncio.create_task(self._play_video())
            
            imgui.same_line()
            if imgui.button("Pause"):
                asyncio.create_task(self._pause_video())
            
            imgui.same_line()
            if imgui.button("Stop"):
                asyncio.create_task(self._stop_video())
            
            # Playback status
            if self.playback_status:
                status = self.playback_status
                imgui.text(f"State: {status.state.value}")
                imgui.text(f"Time: {status.current_time_ms/1000:.1f}s / {status.duration_ms/1000:.1f}s")
                imgui.text(f"Actions: {status.actions_processed} / {status.total_actions}")
                imgui.text(f"Device Updates: {status.position_updates_sent}")
    
    def _render_statistics_tab(self):
        """Render statistics and monitoring tab."""
        imgui.text("Device Control Statistics")
        imgui.separator()
        
        # Device stats
        if self.device_manager:
            device_stats = self.device_manager.get_status()
            
            imgui.text("Device Manager:")
            for key, value in device_stats.items():
                imgui.text(f"  {key}: {value}")
        
        imgui.separator()
        
        # Parameter manager stats
        if self.parameter_manager:
            imgui.text("Parameter Manager:")
            profiles = self.parameter_manager.list_profiles()
            imgui.text(f"  Loaded Profiles: {len(profiles)}")
            active = self.parameter_manager.get_active_profile()
            imgui.text(f"  Active Profile: {active.device_name if active else 'None'}")
        
        imgui.separator()
        
        # Performance monitoring
        imgui.text("Performance:")
        
        current_time = time.time()
        if current_time - self.last_stats_update > 1.0:  # Update every second
            if self.live_tracker:
                self.tracking_stats = self.live_tracker.get_stats()
            self.last_stats_update = current_time
        
        if self.tracking_stats:
            imgui.text("Live Tracking Performance:")
            for key, value in self.tracking_stats.items():
                if isinstance(value, float):
                    imgui.text(f"  {key}: {value:.3f}")
                else:
                    imgui.text(f"  {key}: {value}")
    
    def _render_parameter_panel(self):
        """Render standalone parameter panel."""
        if imgui.begin("Device Parameters", True)[1]:
            self._render_parameters_tab()
            imgui.end()
    
    def _render_live_tracking_panel(self):
        """Render standalone live tracking panel."""
        if imgui.begin("Live Tracking", True)[1]:
            self._render_live_tracking_tab()
            imgui.end()
    
    def _render_video_player_panel(self):
        """Render standalone video player panel."""
        if imgui.begin("Video Player", True)[1]:
            self._render_video_player_tab()
            imgui.end()
    
    def _render_live_device_control_toggle(self):
        """Render toggle for live device control during tracking."""
        imgui.text("Live Device Control:")
        imgui.text_colored("Stream tracking positions to device in real-time", 0.7, 0.7, 0.7)
        
        # Get current state from tracker manager
        current_enabled = False
        if hasattr(self.app, 'tracker_manager') and self.app.tracker_manager:
            current_enabled = getattr(self.app.tracker_manager, 'live_device_control_enabled', False)
        
        # Checkbox for toggling
        changed, new_enabled = imgui.checkbox("Enable Live Device Control", current_enabled)
        if changed:
            self._set_live_device_control_enabled(new_enabled)
        
        # Status indicator
        if current_enabled:
            imgui.same_line()
            imgui.text_colored("● ENABLED", 0.0, 1.0, 0.0)
        else:
            imgui.same_line()
            imgui.text_colored("● DISABLED", 0.7, 0.7, 0.7)
        
        # Help text
        if current_enabled:
            imgui.text_colored("Tracker positions will stream to this device during live tracking", 0.0, 1.0, 0.0)
        else:
            imgui.text("Enable to stream live tracking positions to connected device")
    
    def _set_live_device_control_enabled(self, enabled: bool):
        """Enable/disable live device control in tracker manager."""
        try:
            if hasattr(self.app, 'tracker_manager') and self.app.tracker_manager:
                self.app.tracker_manager.set_live_device_control_enabled(enabled)
                self.app.logger.info(f"Live device control set to: {'enabled' if enabled else 'disabled'}")
            else:
                self.app.logger.warning("Tracker manager not available for live device control")
        except Exception as e:
            self.app.logger.error(f"Error setting live device control: {e}")
    
    def _render_debug_panel(self):
        """Render debug information panel."""
        if imgui.begin("Device Debug", True)[1]:
            imgui.text("Debug Information")
            imgui.separator()
            
            # Raw device info
            if self.connected_device_info:
                imgui.text("Connected Device Details:")
                for key, value in self.connected_device_info.items():
                    imgui.text(f"  {key}: {value}")
            
            imgui.separator()
            
            # System info
            imgui.text(f"Device Control Available: {DEVICE_CONTROL_AVAILABLE}")
            imgui.text(f"System Available: {self.available}")
            
            imgui.end()
    
    # Async device operations
    async def _discover_devices(self):
        """Discover available devices."""
        try:
            self.discovery_in_progress = True
            self.discovered_devices = await self.device_manager.discover_devices()
            self.last_discovery_time = time.time()
            self.logger.info(f"Discovered {len(self.discovered_devices)} devices")
        except Exception as e:
            self.logger.error(f"Device discovery failed: {e}")
        finally:
            self.discovery_in_progress = False
    
    async def _quick_handy_scan(self):
        """Quick scan specifically for Handy devices on WiFi."""
        try:
            self.discovery_in_progress = True
            self.app.logger.info("Starting quick Handy scan...")
            
            # Use the Handy backend directly for faster scanning
            handy_backend = self.device_manager.available_backends.get('handy')
            if handy_backend:
                devices = await handy_backend.discover_devices()
                
                # Update discovered devices list
                device_dict = {}
                for device in devices:
                    device_dict[device.device_id] = device
                    
                # Add to existing discoveries or replace if empty
                if not self.discovered_devices:
                    self.discovered_devices = devices
                else:
                    # Merge with existing discoveries
                    existing_ids = {d.device_id for d in self.discovered_devices}
                    for device in devices:
                        if device.device_id not in existing_ids:
                            self.discovered_devices.append(device)
                
                self.last_discovery_time = time.time()
                self.app.logger.info(f"Quick Handy scan found {len(devices)} devices")
            else:
                self.app.logger.error("Handy backend not available")
                
        except Exception as e:
            self.logger.error(f"Quick Handy scan failed: {e}")
        finally:
            self.discovery_in_progress = False
    
    async def _connect_device(self, device_id: str):
        """Connect to a specific device."""
        try:
            success = await self.device_manager.connect(device_id)
            if success:
                self.connected_device_info = self.device_manager.get_connected_devices()
                # Set up parameter profile
                device_info = next((d for d in self.discovered_devices if d.device_id == device_id), None)
                if device_info:
                    profile = self.parameter_manager.create_profile_for_device(device_info)
                    self.parameter_manager.save_profile(profile)
                    self.parameter_manager.set_active_profile(device_id)
                
                self.logger.info(f"Connected to device: {device_id}")
            else:
                self.logger.error(f"Failed to connect to device: {device_id}")
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
    
    async def _disconnect_device(self):
        """Disconnect from current device."""
        try:
            await self.device_manager.stop()
            self.connected_device_info = None
            self.logger.info("Disconnected from device")
        except Exception as e:
            self.logger.error(f"Disconnection error: {e}")
    
    async def _test_device(self):
        """Test current device."""
        try:
            # Simple test sequence
            positions = [30, 70, 30, 70, 50]
            for pos in positions:
                await self.device_manager.update_position(pos)
                await asyncio.sleep(0.5)
            self.logger.info("Device test completed")
        except Exception as e:
            self.logger.error(f"Device test failed: {e}")
    
    async def _start_live_tracking(self):
        """Start live tracking."""
        try:
            success = await self.live_tracker.start()
            if success:
                self.live_tracking_active = True
                self.logger.info("Live tracking started")
        except Exception as e:
            self.logger.error(f"Failed to start live tracking: {e}")
    
    async def _stop_live_tracking(self):
        """Stop live tracking."""
        try:
            await self.live_tracker.stop()
            self.live_tracking_active = False
            self.logger.info("Live tracking stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop live tracking: {e}")
    
    async def _play_video(self):
        """Start video playback."""
        try:
            await self.video_player.play()
            self.logger.info("Video playback started")
        except Exception as e:
            self.logger.error(f"Video playback failed: {e}")
    
    async def _pause_video(self):
        """Pause video playback."""
        try:
            await self.video_player.pause()
            self.logger.info("Video playback paused")
        except Exception as e:
            self.logger.error(f"Video pause failed: {e}")
    
    async def _stop_video(self):
        """Stop video playback."""
        try:
            await self.video_player.stop()
            self.logger.info("Video playback stopped")
        except Exception as e:
            self.logger.error(f"Video stop failed: {e}")
    
    def update_tracking_result(self, tracking_result):
        """Update device with tracking result (called from app)."""
        if self.live_tracking_active and self.live_tracker:
            asyncio.create_task(self.live_tracker.process_tracking_result(tracking_result))
    
    def _render_supporter_only_message(self):
        """Render supporter-only message when device control is not available."""
        # Set window properties
        imgui.set_next_window_position(150, 150, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(500, 300, imgui.FIRST_USE_EVER)
        
        # Main window
        window_flags = imgui.WINDOW_NO_COLLAPSE
        expanded, opened = imgui.begin("🎮 Device Control", True, window_flags)
        
        if not opened:
            self.app.app_state_ui.show_device_control_window = False
        
        if expanded:
            # Center the content
            imgui.dummy(0, 20)
            
            # Title
            imgui.push_font(None)  # Use default font but we can make it bold
            imgui.text_colored("🎮 Device Control", 0.2, 0.8, 1.0)
            imgui.pop_font()
            
            imgui.separator()
            imgui.dummy(0, 10)
            
            # Main message
            imgui.text_colored("Supporter Feature Only", 1.0, 0.6, 0.0)
            imgui.dummy(0, 10)
            
            imgui.text_wrapped(
                "Device control features are available to supporters only. "
                "These features include:"
            )
            
            imgui.dummy(0, 5)
            imgui.bullet_text("Hardware device integration (Handy, OSR2, etc.)")
            imgui.bullet_text("Live tracking with device control")
            imgui.bullet_text("Video + funscript synchronized playback")
            imgui.bullet_text("Advanced device parameterization")
            imgui.bullet_text("Multi-axis device support")
            
            imgui.dummy(0, 15)
            
            # Support button
            if imgui.button("Become a Supporter", width=200, height=30):
                # This would open the support/patreon page
                self._open_support_page()
            
            imgui.same_line()
            imgui.text_colored("Support the project!", 0.7, 0.7, 0.7)
            
            imgui.dummy(0, 10)
            imgui.separator()
            imgui.dummy(0, 5)
            
            imgui.text_colored("How to activate:", 0.0, 1.0, 0.0)
            imgui.text_wrapped(
                "Once you become a supporter, you'll receive a device_control "
                "folder. Simply copy it to your FunGen directory and restart "
                "the application to activate all device control features!"
            )
        
        imgui.end()
    
    def _open_support_page(self):
        """Open the support/patreon page in browser."""
        try:
            import webbrowser
            # This would be the actual support URL
            support_url = "https://patreon.com/your_project"  # Replace with actual URL
            webbrowser.open(support_url)
            self.logger.info("Opened supporter page in browser")
        except Exception as e:
            self.logger.error(f"Failed to open support page: {e}")

    def cleanup(self):
        """Clean up device control resources."""
        if not self.available:
            return
        
        try:
            if self.live_tracking_active:
                asyncio.create_task(self.live_tracker.stop())
            
            if self.device_manager and self.device_manager.is_connected():
                asyncio.create_task(self.device_manager.stop())
            
            if self.video_player:
                self.video_player.cleanup()
            
            self.logger.info("Device control cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during device control cleanup: {e}")


# Factory function for integration
def create_device_control_ui(app) -> DeviceControlUI:
    """Create device control UI for the app."""
    return DeviceControlUI(app)