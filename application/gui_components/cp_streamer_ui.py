"""Streamer/Native Sync tab UI mixin for ControlPanelUI."""
import time
import imgui
from application.utils import get_icon_texture_manager, primary_button_style, destructive_button_style
from application.utils.imgui_helpers import DisabledScope as _DisabledScope, tooltip_if_hovered as _tooltip_if_hovered
from application.utils.section_card import section_card as _section_card
from config.element_group_colors import ControlPanelColors as _CPColors


class StreamerMixin:
    """Mixin providing Streamer tab rendering methods."""

    def _render_streamer_error(self, err):
        """Render streamer initialization error with version mismatch hints."""
        if err is not None:
            imgui.text_colored(f"Streamer failed to initialize: {err}", 1.0, 0.4, 0.4, 1.0)
            if isinstance(err, (AttributeError, TypeError)):
                imgui.spacing()
                imgui.text_colored("This looks like a version mismatch.", 1.0, 0.85, 0.2, 1.0)
                imgui.text_colored("Did you update to the latest Streamer version?", 1.0, 0.85, 0.2, 1.0)
                imgui.text_colored("Install the latest streamer zip from your purchase.", 0.8, 0.8, 0.8, 1.0)
            elif isinstance(err, (ImportError, ModuleNotFoundError)):
                imgui.spacing()
                imgui.text_colored("Streamer addon not found or incomplete.", 1.0, 0.85, 0.2, 1.0)
                imgui.text_colored("Install the latest streamer zip from your purchase.", 0.8, 0.8, 0.8, 1.0)
            else:
                imgui.text_colored("Check logs for details.", 1.0, 0.5, 0.0, 1.0)
        else:
            imgui.text("Streamer initializing...")
        imgui.spacing()
        if imgui.button("Retry##streamer_retry"):
            self._streamer_init_error = None

    def _handle_streamer_auto_hide(self, is_running, client_count):
        """Handle auto-hide/show video feed based on client connections."""
        if not is_running:
            return

        if not hasattr(self.app.app_settings, '_streamer_auto_hide_video'):
            self.app.app_settings._streamer_auto_hide_video = True

        auto_hide_enabled = getattr(self.app.app_settings, '_streamer_auto_hide_video', True)

        if not hasattr(self, '_prev_client_count'):
            self._prev_client_count = 0

        # Client connected (0 -> >0)
        if auto_hide_enabled and client_count > 0 and self._prev_client_count == 0:
            self.app.app_state_ui.show_video_feed = False
            self.app.app_settings.set("show_video_feed", False)
            self.app.logger.info("Auto-hiding video feed (streamer active)")

        # All clients disconnected (>0 -> 0)
        elif client_count == 0 and self._prev_client_count > 0:
            self.app.app_state_ui.show_video_feed = True
            self.app.app_settings.set("show_video_feed", True)
            self.app.logger.info("Restoring video feed (no clients)")

        self._prev_client_count = client_count

    def _render_heresphere_urls(self, status):
        """Render HereSphere integration URLs and copy button."""
        if not status.get('heresphere_enabled', False):
            return

        imgui.spacing()
        imgui.separator()
        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.STATUS_INFO)
        imgui.text("HereSphere Integration:")
        imgui.pop_style_color()
        imgui.spacing()

        local_ip = self._get_local_ip()
        heresphere_api_port = status.get('heresphere_api_port', 8091)

        api_url = f"http://{local_ip}:{heresphere_api_port}/heresphere"
        imgui.text("API Server (POST):")
        imgui.same_line()
        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.SUCCESS_TEXT)
        imgui.text(api_url)
        imgui.pop_style_color()

        if imgui.button("Copy API URL", width=-1):
            self._copy_to_clipboard(api_url)

        imgui.spacing()
        imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
        imgui.text(
            "Configure HereSphere to use the API URL above as a library source. "
            "The Event URL is automatically provided to HereSphere via video metadata."
        )
        imgui.pop_style_color()
        imgui.pop_text_wrap_pos()

    def _render_native_sync_tab(self):
        """Render streamer tab content."""
        try:
            # Initialize streamer manager lazily
            if self._native_sync_manager is None:
                try:
                    from streamer.integration_manager import NativeSyncManager
                    try:
                        self._native_sync_manager = NativeSyncManager(
                            self.app.processor,
                            logger=self.app.logger,
                            app_logic=self.app
                        )
                    except TypeError:
                        self.app.logger.warning("Streamer version doesn't support app_logic parameter - using backward-compatible initialization")
                        self._native_sync_manager = NativeSyncManager(
                            self.app.processor,
                            logger=self.app.logger
                        )
                    self._streamer_init_error = None
                except Exception as e_init:
                    import traceback
                    self.app.logger.error(f"Failed to initialize Streamer: {e_init}")
                    self.app.logger.error(traceback.format_exc())
                    self._streamer_init_error = e_init

            # If init failed, show error and bail out
            if self._native_sync_manager is None:
                self._render_streamer_error(getattr(self, '_streamer_init_error', None))
                return

            # Cache status to avoid expensive lookups every frame (throttle to 500ms)
            current_time = time.time()
            if self._native_sync_status_cache is None or (current_time - self._native_sync_status_time) > 0.5:
                self._native_sync_status_cache = self._native_sync_manager.get_status()
                self._native_sync_status_time = current_time

            status = self._native_sync_status_cache
            is_running = status.get('is_running', False)
            client_count = status.get('connected_clients', 0)

            # Version info
            self._render_addon_version_label("streamer", "Streamer")

            # Auto-hide/show video feed
            self._handle_streamer_auto_hide(is_running, client_count)

            # --- Server Control ---
            with _section_card("Server Control##NativeSyncControl", tier="primary") as is_open:
                if is_open:
                    imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                    imgui.text(
                        "Stream video to browsers/VR headsets with frame-perfect synchronization. "
                        "Supports zoom/pan controls, speed modes, and interactive device control."
                    )
                    imgui.pop_style_color()
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()

                    if is_running:
                        with destructive_button_style():
                            if imgui.button("Stop Streaming Server", width=-1):
                                self._stop_native_sync()
                    else:
                        with primary_button_style():
                            if imgui.button("Start Streaming Server", width=-1):
                                self._start_native_sync()

            # --- Connection Info (only when running) ---
            if is_running:
                with _section_card("Connection Info##NativeSyncConnection", tier="secondary") as is_open:
                    if is_open:
                        viewer_url = f"http://{self._get_local_ip()}:{status.get('http_port', 8080)}/fungen"
                        imgui.text("FunGen Viewer URL:")
                        imgui.same_line()
                        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.SUCCESS_TEXT)
                        imgui.text(viewer_url)
                        imgui.pop_style_color()

                        button_width = imgui.get_content_region_available_width() / 2 - 5
                        with primary_button_style():
                            if imgui.button("Open in Browser", width=button_width):
                                self._open_in_browser(viewer_url)
                        imgui.same_line()
                        if imgui.button("Copy URL", width=button_width):
                            self._copy_to_clipboard(viewer_url)

                # --- Status ---
                with _section_card("Status##NativeSyncStatus", tier="secondary") as is_open:
                    if is_open:
                        sync_active = status.get('sync_server_active', False)
                        video_active = status.get('video_server_active', False)

                        imgui.text("Streamer:")
                        imgui.same_line()
                        if sync_active:
                            imgui.text_colored("Active", *_CPColors.SUCCESS_TEXT)
                        else:
                            imgui.text_colored("Inactive", *_CPColors.ERROR_TEXT)

                        imgui.text("Video Server:")
                        imgui.same_line()
                        if video_active:
                            imgui.text_colored("Active", *_CPColors.SUCCESS_TEXT)
                        else:
                            imgui.text_colored("Inactive", *_CPColors.ERROR_TEXT)

                        imgui.text(f"Connected Clients:")
                        imgui.same_line()
                        if client_count > 0:
                            imgui.text_colored(str(client_count), *_CPColors.SUCCESS_TEXT)
                        else:
                            imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                            imgui.text("0")
                            imgui.pop_style_color()

                        imgui.spacing()

                        # Browser Playback Progress
                        if client_count > 0:
                            imgui.separator()
                            imgui.text("Browser Playback Position:")

                            if self._native_sync_manager and self._native_sync_manager.sync_server:
                                sync_server = self._native_sync_manager.sync_server
                                browser_frame = sync_server.target_frame_index
                                processor_frame = self.app.processor.current_frame_index
                                total_frames = self.app.processor.total_frames

                                if browser_frame is not None and total_frames > 0:
                                    browser_progress = (browser_frame / total_frames) * 100.0
                                    processor_progress = (processor_frame / total_frames) * 100.0

                                    imgui.text("  Browser:")
                                    imgui.same_line(120)
                                    imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *_CPColors.STATUS_INFO)
                                    imgui.progress_bar(browser_progress / 100.0, (200, 0), f"{browser_frame} / {total_frames}")
                                    imgui.pop_style_color()

                                    imgui.text("  Processing:")
                                    imgui.same_line(120)
                                    if processor_frame > browser_frame:
                                        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *_CPColors.SUCCESS_TEXT)
                                    else:
                                        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *_CPColors.WARNING_TEXT)
                                    imgui.progress_bar(processor_progress / 100.0, (200, 0), f"{processor_frame} / {total_frames}")
                                    imgui.pop_style_color()

                                    drift = processor_frame - browser_frame
                                    if abs(drift) > 10:
                                        imgui.spacing()
                                        if drift > 0:
                                            imgui.text_colored(f"  Processing is {drift} frames ahead", *_CPColors.SUCCESS_TEXT)
                                        else:
                                            imgui.text_colored(f"  Processing is {abs(drift)} frames behind!", *_CPColors.WARNING_TEXT)
                                else:
                                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                                    imgui.text("  Waiting for browser position updates...")
                                    imgui.pop_style_color()
                            else:
                                imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                                imgui.text("  No sync data available")
                                imgui.pop_style_color()

                        # HereSphere URLs
                        self._render_heresphere_urls(status)

                # --- Display Options ---
                with _section_card("Display Options##NativeSyncDisplay", tier="secondary",
                                   open_by_default=False) as is_open:
                    if is_open:
                        imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                        imgui.text(
                            "When streaming to browsers, you can hide the local video feed to save GPU resources."
                        )
                        imgui.pop_style_color()
                        imgui.pop_text_wrap_pos()
                        imgui.spacing()

                        if not hasattr(self.app.app_settings, '_streamer_auto_hide_video'):
                            self.app.app_settings._streamer_auto_hide_video = True

                        auto_hide = getattr(self.app.app_settings, '_streamer_auto_hide_video', True)
                        clicked, new_val = imgui.checkbox("Auto-hide Video Feed while streaming", auto_hide)
                        if clicked:
                            self.app.app_settings._streamer_auto_hide_video = new_val
                            if new_val and client_count > 0:
                                self.app.app_state_ui.show_video_feed = False
                                self.app.app_settings.set("show_video_feed", False)
                            elif not new_val:
                                self.app.app_state_ui.show_video_feed = True
                                self.app.app_settings.set("show_video_feed", True)
                        _tooltip_if_hovered(
                            "When enabled, the video feed will be hidden\n"
                            "when clients are connected, and restored when\n"
                            "all clients disconnect."
                        )

                # --- Rolling Autotune ---
                with _section_card("Rolling Autotune (Live Tracking)##RollingAutotuneStreamer",
                                   tier="secondary", open_by_default=False) as is_open:
                    if is_open:
                        imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                        imgui.text(
                            "Automatically apply Ultimate Autotune to live tracking data every N seconds. "
                            "Perfect for streaming with a 5+ second buffer - ensures the cleanest possible "
                            "funscript signal reaches viewers/devices by the time they play it."
                        )
                        imgui.pop_style_color()
                        imgui.pop_text_wrap_pos()
                        imgui.spacing()

                        settings = self.app.app_settings
                        tr = self.app.tracker

                        if not tr:
                            imgui.text_colored("Tracker not initialized", *_CPColors.WARNING_TEXT)
                        else:
                            streamer_available = False
                            clients_connected = False
                            try:
                                from application.utils.feature_detection import is_feature_enabled
                                streamer_available = is_feature_enabled("streamer")
                                if streamer_available and self._native_sync_manager:
                                    if self._native_sync_manager.sync_server:
                                        clients_connected = len(self._native_sync_manager.sync_server.websocket_clients) > 0
                                    if not clients_connected and self._native_sync_manager.heresphere_event_bridge:
                                        heresphere = self._native_sync_manager.heresphere_event_bridge
                                        if heresphere.is_running and heresphere.last_event_time > 0:
                                            time_since_last_event = time.time() - heresphere.last_event_time
                                            if time_since_last_event < 30.0:
                                                clients_connected = True
                            except Exception as e:
                                self.app.logger.debug(f"Error checking streamer availability: {e}")

                            can_enable = streamer_available and clients_connected

                            if not can_enable:
                                icon_mgr = get_icon_texture_manager()
                                warning_tex, _, _ = icon_mgr.get_icon_texture('warning.png')
                                imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                                if warning_tex:
                                    imgui.image(warning_tex, 20, 20)
                                    imgui.same_line()
                                if not streamer_available:
                                    imgui.text_colored("Requires Streamer module to be available", *_CPColors.WARNING_TEXT)
                                elif not clients_connected:
                                    imgui.text_colored("Requires an active streamer session with connected clients", *_CPColors.WARNING_TEXT)
                                imgui.pop_text_wrap_pos()
                                imgui.spacing()

                            cur_enabled = settings.get("live_tracker_rolling_autotune_enabled", False)

                            with _DisabledScope(not can_enable):
                                ch, new_enabled = imgui.checkbox("Enable Rolling Autotune##RollingAutotuneEnable", cur_enabled)
                            _tooltip_if_hovered(
                                "Apply Ultimate Autotune to the last N seconds of funscript data every N seconds\n"
                                "Recommended: Keep processing ahead of browser playback by at least the window size"
                                if can_enable else
                                "Rolling autotune requires:\n- Streamer module available\n- Active streamer session with connected clients"
                            )

                            if ch and can_enable:
                                settings.set("live_tracker_rolling_autotune_enabled", new_enabled)
                                tr.rolling_autotune_enabled = new_enabled
                                if new_enabled:
                                    self.app.logger.info("Rolling autotune enabled for live tracking", extra={'status_message': True})
                                else:
                                    self.app.logger.info("Rolling autotune disabled", extra={'status_message': True})

                            if cur_enabled:
                                imgui.spacing()
                                imgui.separator()
                                imgui.spacing()

                                imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.STATUS_INFO)
                                imgui.text("Advanced Settings:")
                                imgui.pop_style_color()
                                imgui.spacing()

                                cur_interval = settings.get("live_tracker_rolling_autotune_interval_ms", 5000)
                                imgui.text("Autotune Interval (ms):")
                                imgui.push_item_width(150)
                                ch, new_interval = imgui.input_int("##RollingAutotuneInterval", cur_interval, 1000)
                                imgui.pop_item_width()
                                _tooltip_if_hovered("How often to apply autotune (in milliseconds). Default: 5000ms (5 seconds)")
                                if ch:
                                    v = max(1000, min(30000, new_interval))
                                    if v != cur_interval:
                                        settings.set("live_tracker_rolling_autotune_interval_ms", v)
                                        tr.rolling_autotune_interval_ms = v

                                cur_window = settings.get("live_tracker_rolling_autotune_window_ms", 5000)
                                imgui.text("Processing Window (ms):")
                                imgui.push_item_width(150)
                                ch, new_window = imgui.input_int("##RollingAutotuneWindow", cur_window, 1000)
                                imgui.pop_item_width()
                                _tooltip_if_hovered(
                                    "Size of data window to process each time (in milliseconds).\n"
                                    "Should match your buffer size. Default: 5000ms (5 seconds)"
                                )
                                if ch:
                                    v = max(1000, min(30000, new_window))
                                    if v != cur_window:
                                        settings.set("live_tracker_rolling_autotune_window_ms", v)
                                        tr.rolling_autotune_window_ms = v

                                imgui.spacing()
                                imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                                imgui.text_colored(
                                    "Tip: Keep your processing position at least 5-10 seconds ahead of "
                                    "browser playback to ensure cleaned data is ready when needed.",
                                    *_CPColors.SUCCESS_TEXT
                                )
                                imgui.pop_text_wrap_pos()

            # --- XBVR Integration ---
            with _section_card("XBVR Integration##XBVRSettings", tier="primary",
                               open_by_default=not is_running) as is_open:
                if is_open:
                    imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                    imgui.text(
                        "Browse and load videos from your XBVR library directly in the VR viewer. "
                        "Displays scene thumbnails, funscript availability, and enables remote playback control."
                    )
                    imgui.pop_style_color()
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()

                    xbvr_host = self.app.app_settings.get('xbvr_host', 'localhost')
                    xbvr_port = self.app.app_settings.get('xbvr_port', 9999)

                    imgui.text("XBVR Host/IP:")
                    imgui.push_item_width(200)
                    changed, new_host = imgui.input_text("##xbvr_host", str(xbvr_host), 256)
                    imgui.pop_item_width()
                    if changed or imgui.is_item_deactivated_after_edit():
                        self.app.app_settings.set('xbvr_host', new_host)
                        self.app.app_settings.save_settings()

                    imgui.text("XBVR Port:")
                    imgui.push_item_width(100)
                    changed, new_port_str = imgui.input_text("##xbvr_port", str(xbvr_port), 256)
                    imgui.pop_item_width()
                    if changed or imgui.is_item_deactivated_after_edit():
                        try:
                            new_port = int(new_port_str)
                            self.app.app_settings.set('xbvr_port', new_port)
                            self.app.app_settings.save_settings()
                        except ValueError:
                            pass

                    imgui.spacing()
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.STATUS_INFO)
                    imgui.text(f"XBVR URL: http://{xbvr_host}:{xbvr_port}")
                    imgui.pop_style_color()

                    imgui.spacing()
                    with primary_button_style():
                        if imgui.button("Discover XBVR Address", width=-1):
                            self._discover_xbvr_address()

                    imgui.spacing()
                    with primary_button_style():
                        if imgui.button("Open XBVR Browser", width=-1):
                            import webbrowser
                            local_ip = self._get_local_ip()
                            xbvr_browser_url = f"http://{local_ip}:8080/xbvr"
                            webbrowser.open(xbvr_browser_url)
                            self.app.logger.info(f"Opening XBVR browser: {xbvr_browser_url}")

            # --- Stash Integration ---
            with _section_card("Stash Integration##StashSettings", tier="primary",
                               open_by_default=not is_running) as is_open:
                if is_open:
                    imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                    imgui.text(
                        "Browse and load videos from your Stash library directly in the VR viewer. "
                        "Access scene markers, organized collections, and interactive funscripts."
                    )
                    imgui.pop_style_color()
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()

                    stash_host = self.app.app_settings.get('stash_host', 'localhost')
                    stash_port = self.app.app_settings.get('stash_port', 9999)
                    stash_api_key = self.app.app_settings.get('stash_api_key', '')

                    imgui.text("Stash Host/IP:")
                    imgui.push_item_width(200)
                    changed, new_host = imgui.input_text("##stash_host", str(stash_host), 256)
                    imgui.pop_item_width()
                    if changed or imgui.is_item_deactivated_after_edit():
                        self.app.app_settings.set('stash_host', new_host)
                        self.app.app_settings.save_settings()

                    imgui.text("Stash Port:")
                    imgui.push_item_width(100)
                    changed, new_port_str = imgui.input_text("##stash_port", str(stash_port), 256)
                    imgui.pop_item_width()
                    if changed or imgui.is_item_deactivated_after_edit():
                        try:
                            new_port = int(new_port_str)
                            self.app.app_settings.set('stash_port', new_port)
                            self.app.app_settings.save_settings()
                        except ValueError:
                            pass

                    imgui.text("API Key:")
                    imgui.same_line()
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
                    imgui.text("(required for authentication)")
                    imgui.pop_style_color()
                    imgui.push_item_width(300)
                    changed, new_api_key = imgui.input_text(
                        "##stash_api_key", str(stash_api_key), 256,
                        imgui.INPUT_TEXT_PASSWORD
                    )
                    imgui.pop_item_width()
                    if changed or imgui.is_item_deactivated_after_edit():
                        self.app.app_settings.set('stash_api_key', new_api_key)
                        self.app.app_settings.save_settings()

                    imgui.spacing()
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.STATUS_INFO)
                    imgui.text(f"Stash URL: http://{stash_host}:{stash_port}")
                    imgui.pop_style_color()
                    imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.LABEL_TEXT)
                    imgui.text("Find your API key in Stash: Settings -> Security -> API Key")
                    imgui.pop_style_color()

                    imgui.spacing()
                    with primary_button_style():
                        if imgui.button("Open Stash Browser", width=-1):
                            import webbrowser
                            local_ip = self._get_local_ip()
                            stash_browser_url = f"http://{local_ip}:8080/stash"
                            webbrowser.open(stash_browser_url)
                            self.app.logger.info(f"Opening Stash browser: {stash_browser_url}")

            # --- Quest3 VR Bridge (stalled) ---
            with _section_card("Quest3 VR Bridge##Quest3Bridge", tier="secondary",
                               open_by_default=False) as is_open:
                if is_open:
                    with _DisabledScope(True):
                        imgui.text("Direct Quest 3 streaming via Wi-Fi bridge.")
                        imgui.spacing()
                        imgui.text("Status: Backend implemented")
                        imgui.text("Latency target: < 20ms")
                        imgui.spacing()
                        if imgui.button("Connect Quest3##Quest3Connect", width=-1):
                            pass
                    _tooltip_if_hovered("Feature stalled - backend exists but not yet exposed")

            # --- Video Cache (stalled) ---
            with _section_card("Video Cache##VideoCacheMgmt", tier="secondary",
                               open_by_default=False) as is_open:
                if is_open:
                    with _DisabledScope(True):
                        imgui.text("Manage transcoded video cache for streaming.")
                        imgui.spacing()
                        imgui.text("Cache entries: --")
                        imgui.text("Total size: --")
                        imgui.spacing()
                        with destructive_button_style():
                            if imgui.button("Clear Cache##ClearVideoCache", width=-1):
                                pass
                    _tooltip_if_hovered("Feature stalled - backend exists but not yet exposed")

            # --- Info sections (only when not running) ---
            if not is_running:
                with _section_card("Requirements##NativeSyncRequirements", tier="secondary") as is_open:
                    if is_open:
                        imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                        imgui.push_style_color(imgui.COLOR_TEXT, *_CPColors.HINT_TEXT)
                        imgui.text(
                            "This will start HTTP and WebSocket servers for native video playback "
                            "in browsers and VR headsets. Your video will be served at full quality "
                            "with frame-perfect synchronization."
                        )
                        imgui.pop_style_color()
                        imgui.pop_text_wrap_pos()
                        imgui.spacing()

                        imgui.bullet_text("Ports 8080 (HTTP) and 8765 (WebSocket) available")
                        imgui.bullet_text("Browser with HTML5 video support")
                        imgui.bullet_text("Video can be loaded before or after starting the server")

                with _section_card("Features##NativeSyncFeatures", tier="secondary",
                                   open_by_default=False) as is_open:
                    if is_open:
                        imgui.bullet_text("Native hardware H.265/AV1 decode")
                        imgui.bullet_text("Zoom/Pan controls (+/- and WASD keys)")
                        imgui.bullet_text("Speed modes (Real Time / Slo Mo)")
                        imgui.bullet_text("Real-time FPS and resolution stats")
                        imgui.bullet_text("Interactive device control")
                        imgui.bullet_text("Funscript visualization graph")

        except Exception as e:
            import traceback
            self.app.logger.error(f"Streamer tab error: {e}")
            self.app.logger.error(traceback.format_exc())
            imgui.text_colored(f"Error in Streamer: {e}", 1.0, 0.4, 0.4, 1.0)
            if isinstance(e, (AttributeError, TypeError)):
                imgui.spacing()
                imgui.text_colored("This looks like a version mismatch.", 1.0, 0.85, 0.2, 1.0)
                imgui.text_colored("Did you update to the latest Streamer version?", 1.0, 0.85, 0.2, 1.0)
                imgui.text_colored("Install the latest streamer zip from your purchase.", 0.8, 0.8, 0.8, 1.0)
            elif isinstance(e, (ImportError, ModuleNotFoundError)):
                imgui.text_colored("Streamer addon not found or incomplete.", 1.0, 0.85, 0.2, 1.0)
                imgui.text_colored("Install the latest streamer zip from your purchase.", 0.8, 0.8, 0.8, 1.0)
            imgui.text_colored("See logs for full details.", 0.7, 0.7, 0.7, 1.0)

    def _start_native_sync(self):
        """Start streamer servers."""
        try:
            self.app.logger.info("Starting streamer...")
            if self._native_sync_manager:
                self._native_sync_manager.enable_heresphere = True
                self._native_sync_manager.enable_xbvr_browser = True
            self._native_sync_manager.start()
            self.app._streamer_active = True
        except Exception as e:
            self.app.logger.error(f"Failed to start streamer: {e}")
            self.app._streamer_active = False
            import traceback
            self.app.logger.error(traceback.format_exc())

    def _stop_native_sync(self):
        """Stop streamer servers."""
        try:
            self.app.logger.info("Stopping streamer...")
            self._native_sync_manager.stop()
            self.app._streamer_active = False
        except Exception as e:
            self.app.logger.error(f"Failed to stop streamer: {e}")
            self.app._streamer_active = False

    def _open_in_browser(self, url: str):
        """Open URL in system default browser."""
        try:
            import webbrowser
            webbrowser.open(url)
            self.app.logger.info(f"Opening in browser: {url}")
        except Exception as e:
            self.app.logger.error(f"Failed to open browser: {e}")

    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard."""
        try:
            import pyperclip
            pyperclip.copy(text)
            self.app.logger.info(f"Copied to clipboard: {text}")
        except Exception:
            self.app.logger.info(f"URL to copy: {text}")

    def _discover_xbvr_address(self):
        """Attempt to discover XBVR on the local network."""
        import socket
        import requests
        from threading import Thread

        def scan_network():
            local_ip = self._get_local_ip()
            if not local_ip or not local_ip.startswith('192.168.'):
                self.app.logger.info("Could not determine local network subnet", extra={'status_message': True})
                return

            subnet = '.'.join(local_ip.split('.')[:-1])
            self.app.logger.info(f"Scanning {subnet}.0/24 for XBVR on port 9999...", extra={'status_message': True})

            found = False
            for i in range(1, 255):
                if found:
                    break
                test_ip = f"{subnet}.{i}"
                try:
                    response = requests.get(f"http://{test_ip}:9999", timeout=0.5)
                    if response.status_code == 200 or 'xbvr' in response.text.lower():
                        self.app.logger.info(f"Found XBVR at {test_ip}:9999", extra={'status_message': True})
                        self.app.app_settings.set('xbvr_host', test_ip)
                        self.app.app_settings.set('xbvr_port', 9999)
                        self.app.app_settings.save_settings()
                        if hasattr(self.app, 'integration_manager') and self.app.integration_manager:
                            self.app.integration_manager.update_xbvr_client(test_ip, 9999)
                        found = True
                except Exception:
                    pass

            if not found:
                self.app.logger.info("No XBVR server found on local network", extra={'status_message': True})

        Thread(target=scan_network, daemon=True).start()

    def _get_local_ip(self) -> str:
        """Get local network IP address (prefer 192.168.x.x over VPN)."""
        import socket
        try:
            hostname = socket.gethostname()
            addresses = socket.getaddrinfo(hostname, None, socket.AF_INET)
            for addr in addresses:
                ip = addr[4][0]
                if ip.startswith('192.168.'):
                    return ip
            for addr in addresses:
                ip = addr[4][0]
                if not ip.startswith('127.'):
                    return ip
            return "localhost"
        except Exception:
            return "localhost"

    def _export_funscript_timeline(self, app, timeline_num):
        """Export funscript from specified timeline."""
        app.file_manager.export_funscript_from_timeline(timeline_num)
