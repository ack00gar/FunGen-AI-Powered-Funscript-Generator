"""Device Control — HandyPanel methods."""
import asyncio
import threading
import imgui
from application.utils.imgui_helpers import tooltip_if_hovered as _tooltip_if_hovered
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.section_card import section_card as _section_card
from application.utils import primary_button_style, destructive_button_style


class HandyPanelMixin:
    """Mixin fragment for DeviceControlMixin."""

    def _render_handy_controls(self):
        """Render Handy direct API controls."""

        # Check Handy connection status
        connected_device = self.device_manager.get_connected_device_info() if self.device_manager.is_connected() else None
        is_handy_connected = self._is_device_type_connected("handy")

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


    def _connect_handy(self, connection_key: str):
        """Connect to Handy device with given connection key."""

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

