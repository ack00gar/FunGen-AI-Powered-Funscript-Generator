"""Device Control — AdvancedSettings methods."""
import traceback
import imgui
from application.utils.imgui_helpers import tooltip_if_hovered as _tooltip_if_hovered
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.section_card import section_card as _section_card
from application.utils import primary_button_style, destructive_button_style


class AdvancedSettingsMixin:
    """Mixin fragment for DeviceControlMixin."""

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
        if self._is_device_type_connected("osr"):
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            self._render_osr_performance_settings()

        imgui.spacing()


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

