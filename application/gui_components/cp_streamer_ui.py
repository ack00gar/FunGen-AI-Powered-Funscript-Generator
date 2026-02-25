"""Streamer/Native Sync tab UI mixin for ControlPanelUI."""
import imgui
from application.utils import get_icon_texture_manager, primary_button_style, destructive_button_style


class StreamerMixin:
    """Mixin providing Streamer tab rendering methods."""

    def _render_native_sync_tab(self):
        """Render streamer tab content."""
        try:
            # Initialize streamer manager lazily
            if self._native_sync_manager is None:
                from streamer.integration_manager import NativeSyncManager
                try:
                    # Try with app_logic parameter (newer streamer versions)
                    self._native_sync_manager = NativeSyncManager(
                        self.app.processor,
                        logger=self.app.logger,
                        app_logic=self.app  # For HereSphere auto-load functionality
                    )
                except TypeError:
                    # Fall back to old signature (older streamer versions)
                    self.app.logger.warning("Streamer version doesn't support app_logic parameter - using backward-compatible initialization")
                    self._native_sync_manager = NativeSyncManager(
                        self.app.processor,
                        logger=self.app.logger
                    )

            # Cache status to avoid expensive lookups every frame (throttle to 500ms)
            import time
            current_time = time.time()

            # Update cache if stale (> 500ms)
            if self._native_sync_status_cache is None or (current_time - self._native_sync_status_time) > 0.5:
                self._native_sync_status_cache = self._native_sync_manager.get_status()
                self._native_sync_status_time = current_time

            # Use cached status
            status = self._native_sync_status_cache
            is_running = status.get('is_running', False)
            client_count = status.get('connected_clients', 0)

            # Version info (top of tab, consistent with other supporter modules)
            try:
                import streamer
                version = getattr(streamer, '__version__', 'unknown')
                imgui.text_colored(f"Streamer v{version}", 0.5, 0.5, 0.5, 1.0)
                imgui.spacing()
            except Exception:
                pass

            # Auto-hide/show video feed based on client connections
            if is_running:
                # Initialize setting if not exists
                if not hasattr(self.app.app_settings, '_streamer_auto_hide_video'):
                    self.app.app_settings._streamer_auto_hide_video = True

                auto_hide_enabled = getattr(self.app.app_settings, '_streamer_auto_hide_video', True)

                # Track previous client count
                if not hasattr(self, '_prev_client_count'):
                    self._prev_client_count = 0

                # Client connected (0 -> >0)
                if auto_hide_enabled and client_count > 0 and self._prev_client_count == 0:
                    self.app.app_state_ui.show_video_feed = False
                    self.app.app_settings.set("show_video_feed", False)
                    self.app.logger.info("🎥 Auto-hiding video feed (streamer active)")

                # All clients disconnected (>0 -> 0)
                elif client_count == 0 and self._prev_client_count > 0:
                    self.app.app_state_ui.show_video_feed = True
                    self.app.app_settings.set("show_video_feed", True)
                    self.app.logger.info("🎥 Restoring video feed (no clients)")

                self._prev_client_count = client_count

            # Control Section
            open_, _ = imgui.collapsing_header(
                "Server Control##NativeSyncControl",
                flags=imgui.TREE_NODE_DEFAULT_OPEN,
            )
            if open_:
                # Description
                imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                imgui.text_colored(
                    "Stream video to browsers/VR headsets with frame-perfect synchronization. "
                    "Supports zoom/pan controls, speed modes, and interactive device control.",
                    0.7, 0.7, 0.7
                )
                imgui.pop_text_wrap_pos()
                imgui.spacing()

                # Start/Stop button
                if is_running:
                    # Running - show stop button (DESTRUCTIVE - stops server)
                    with destructive_button_style():
                        if imgui.button("Stop Streaming Server", width=-1):
                            self._stop_native_sync()
                else:
                    # Not running - show start button (PRIMARY - positive action)
                    with primary_button_style():
                        if imgui.button("Start Streaming Server", width=-1):
                            self._start_native_sync()

            # Connection Info Section (only when running)
            if is_running:
                open_, _ = imgui.collapsing_header(
                    "Connection Info##NativeSyncConnection",
                    flags=imgui.TREE_NODE_DEFAULT_OPEN,
                )
                if open_:
                    # FunGen Viewer URL
                    viewer_url = f"http://{self._get_local_ip()}:{status.get('http_port', 8080)}/fungen"
                    imgui.text("FunGen Viewer URL:")
                    imgui.same_line()
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 1.0, 0.5)
                    imgui.text(viewer_url)
                    imgui.pop_style_color()

                    # Buttons row
                    button_width = imgui.get_content_region_available_width() / 2 - 5
                    # Open in Browser button (PRIMARY - positive action)
                    with primary_button_style():
                        if imgui.button("Open in Browser", width=button_width):
                            self._open_in_browser(viewer_url)
                    imgui.same_line()
                    if imgui.button("Copy URL", width=button_width):
                        self._copy_to_clipboard(viewer_url)

                # Status Section
                open_, _ = imgui.collapsing_header(
                    "Status##NativeSyncStatus",
                    flags=imgui.TREE_NODE_DEFAULT_OPEN,
                )
                if open_:
                    # Server status
                    sync_active = status.get('sync_server_active', False)
                    video_active = status.get('video_server_active', False)

                    imgui.text("Streamer:")
                    imgui.same_line()
                    if sync_active:
                        imgui.text_colored("Active", 0.0, 1.0, 0.0)
                    else:
                        imgui.text_colored("Inactive", 1.0, 0.0, 0.0)

                    imgui.text("Video Server:")
                    imgui.same_line()
                    if video_active:
                        imgui.text_colored("Active", 0.0, 1.0, 0.0)
                    else:
                        imgui.text_colored("Inactive", 1.0, 0.0, 0.0)

                    # Connected clients
                    client_count = status.get('connected_clients', 0)
                    imgui.text(f"Connected Clients:")
                    imgui.same_line()
                    if client_count > 0:
                        imgui.text_colored(str(client_count), 0.0, 1.0, 0.5)
                    else:
                        imgui.text_colored("0", 0.7, 0.7, 0.7)

                    imgui.spacing()

                    # Browser Playback Progress (when clients connected)
                    if client_count > 0:
                        imgui.separator()
                        imgui.text("Browser Playback Position:")

                        # Get sync server for browser position
                        if self._native_sync_manager and self._native_sync_manager.sync_server:
                            sync_server = self._native_sync_manager.sync_server
                            browser_frame = sync_server.target_frame_index
                            processor_frame = self.app.processor.current_frame_index
                            total_frames = self.app.processor.total_frames

                            if browser_frame is not None and total_frames > 0:
                                # Calculate progress percentages
                                browser_progress = (browser_frame / total_frames) * 100.0
                                processor_progress = (processor_frame / total_frames) * 100.0

                                # Browser progress bar
                                imgui.text("  Browser:")
                                imgui.same_line(120)
                                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 0.0, 0.7, 1.0)
                                imgui.progress_bar(browser_progress / 100.0, (200, 0), f"{browser_frame} / {total_frames}")
                                imgui.pop_style_color()

                                # Processor progress bar
                                imgui.text("  Processing:")
                                imgui.same_line(120)
                                if processor_frame > browser_frame:
                                    imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 0.0, 1.0, 0.5)
                                else:
                                    imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 1.0, 0.5, 0.0)
                                imgui.progress_bar(processor_progress / 100.0, (200, 0), f"{processor_frame} / {total_frames}")
                                imgui.pop_style_color()

                                # Show drift if exists
                                drift = processor_frame - browser_frame
                                if abs(drift) > 10:
                                    imgui.spacing()
                                    if drift > 0:
                                        imgui.text_colored(f"  Processing is {drift} frames ahead", 0.0, 1.0, 0.5)
                                    else:
                                        imgui.text_colored(f"  Processing is {abs(drift)} frames behind!", 1.0, 0.5, 0.0)
                            else:
                                imgui.text_colored("  Waiting for browser position updates...", 0.7, 0.7, 0.7)
                        else:
                            imgui.text_colored("  No sync data available", 0.7, 0.7, 0.7)

                # Connection Info Section (continued)
                if open_:
                    # HereSphere URLs (if enabled)
                    if status.get('heresphere_enabled', False):
                        imgui.spacing()
                        imgui.separator()
                        imgui.text_colored("HereSphere Integration:", 0.5, 0.8, 1.0)
                        imgui.spacing()

                        # HereSphere API URL
                        local_ip = self._get_local_ip()
                        heresphere_api_port = status.get('heresphere_api_port', 8091)
                        heresphere_event_port = status.get('heresphere_event_port', 8090)

                        api_url = f"http://{local_ip}:{heresphere_api_port}/heresphere"
                        imgui.text("API Server (POST):")
                        imgui.same_line()
                        imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 1.0, 0.8)
                        imgui.text(api_url)
                        imgui.pop_style_color()

                        # Copy button for API URL
                        if imgui.button("Copy API URL", width=-1):
                            self._copy_to_clipboard(api_url)

                        imgui.spacing()
                        imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                        imgui.text_colored(
                            "Configure HereSphere to use the API URL above as a library source. "
                            "The Event URL is automatically provided to HereSphere via video metadata.",
                            0.6, 0.6, 0.6
                        )
                        imgui.pop_text_wrap_pos()

                # Video Display Options (when streaming)
                open_, _ = imgui.collapsing_header(
                    "Display Options##NativeSyncDisplay",
                    flags=0,  # Collapsed by default
                )
                if open_:
                    imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                    imgui.text_colored(
                        "When streaming to browsers, you can hide the local video feed to save GPU resources.",
                        0.7, 0.7, 0.7
                    )
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()

                    # Auto-hide video feed checkbox
                    if not hasattr(self.app.app_settings, '_streamer_auto_hide_video'):
                        self.app.app_settings._streamer_auto_hide_video = True

                    auto_hide = getattr(self.app.app_settings, '_streamer_auto_hide_video', True)
                    clicked, new_val = imgui.checkbox("Auto-hide Video Feed while streaming", auto_hide)
                    if clicked:
                        self.app.app_settings._streamer_auto_hide_video = new_val
                        # Apply immediately
                        if new_val and client_count > 0:
                            # Hide video
                            self.app.app_state_ui.show_video_feed = False
                            self.app.app_settings.set("show_video_feed", False)
                        elif not new_val:
                            # Show video
                            self.app.app_state_ui.show_video_feed = True
                            self.app.app_settings.set("show_video_feed", True)

                    imgui.same_line()
                    imgui.text_colored("(?)", 0.7, 0.7, 0.7)
                    if imgui.is_item_hovered():
                        imgui.begin_tooltip()
                        imgui.text("When enabled, the video feed will be hidden\nwhen clients are connected, and restored when\nall clients disconnect.")
                        imgui.end_tooltip()

                # Rolling Autotune Section (when streaming)
                imgui.spacing()
                open_, _ = imgui.collapsing_header(
                    "Rolling Autotune (Live Tracking)##RollingAutotuneStreamer",
                    flags=0,  # Collapsed by default
                )
                if open_:
                    imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                    imgui.text_colored(
                        "Automatically apply Ultimate Autotune to live tracking data every N seconds. "
                        "Perfect for streaming with a 5+ second buffer - ensures the cleanest possible "
                        "funscript signal reaches viewers/devices by the time they play it.",
                        0.7, 0.7, 0.7
                    )
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()

                    settings = self.app.app_settings
                    tr = self.app.tracker

                    if not tr:
                        imgui.text_colored("Tracker not initialized", 1.0, 0.5, 0.0)
                    else:
                        # Check if streamer is available and has connected clients
                        streamer_available = False
                        clients_connected = False
                        try:
                            from application.utils.feature_detection import is_feature_enabled
                            streamer_available = is_feature_enabled("streamer")
                            if streamer_available and self._native_sync_manager:
                                # Check browser websocket clients
                                if self._native_sync_manager.sync_server:
                                    clients_connected = len(self._native_sync_manager.sync_server.websocket_clients) > 0

                                # Also check HereSphere connections (active within last 30 seconds)
                                if not clients_connected and self._native_sync_manager.heresphere_event_bridge:
                                    import time
                                    heresphere = self._native_sync_manager.heresphere_event_bridge
                                    if heresphere.is_running and heresphere.last_event_time > 0:
                                        time_since_last_event = time.time() - heresphere.last_event_time
                                        if time_since_last_event < 30.0:  # Active within last 30 seconds
                                            clients_connected = True
                        except Exception as e:
                            self.app.logger.debug(f"Error checking streamer availability: {e}")

                        can_enable = streamer_available and clients_connected

                        # Show requirement message if conditions not met
                        if not can_enable:
                            # Get warning icon
                            icon_mgr = get_icon_texture_manager()
                            warning_tex, _, _ = icon_mgr.get_icon_texture('warning.png')

                            imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                            if warning_tex:
                                imgui.image(warning_tex, 20, 20)
                                imgui.same_line()

                            if not streamer_available:
                                imgui.text_colored("Requires Streamer module to be available", 1.0, 0.7, 0.0)
                            elif not clients_connected:
                                imgui.text_colored("Requires an active streamer session with connected clients", 1.0, 0.7, 0.0)
                            imgui.pop_text_wrap_pos()
                            imgui.spacing()

                        # Enable/disable toggle (disabled by default, requires streamer + connected session)
                        cur_enabled = settings.get("live_tracker_rolling_autotune_enabled", False)

                        # Disable checkbox if requirements not met
                        if not can_enable:
                            imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.5, 0.5, 1.0)
                            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, 0.2, 0.2, 0.2, 0.5)
                            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, 0.2, 0.2, 0.2, 0.5)
                            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, 0.2, 0.2, 0.2, 0.5)
                            imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.5, 0.5, 0.5, 0.5)

                        ch, new_enabled = imgui.checkbox("Enable Rolling Autotune##RollingAutotuneEnable", cur_enabled)

                        if not can_enable:
                            imgui.pop_style_color(5)

                        if imgui.is_item_hovered():
                            if not can_enable:
                                tooltip_msg = "Rolling autotune requires:\n• Streamer module available\n• Active streamer session with connected clients"
                            else:
                                tooltip_msg = "Apply Ultimate Autotune to the last N seconds of funscript data every N seconds\nRecommended: Keep processing ahead of browser playback by at least the window size"
                            imgui.set_tooltip(tooltip_msg)

                        if ch and can_enable:
                            settings.set("live_tracker_rolling_autotune_enabled", new_enabled)
                            tr.rolling_autotune_enabled = new_enabled
                            if new_enabled:
                                self.app.logger.info("Rolling autotune enabled for live tracking", extra={'status_message': True})
                            else:
                                self.app.logger.info("Rolling autotune disabled", extra={'status_message': True})

                        # Only show advanced settings if enabled
                        if cur_enabled:
                            imgui.spacing()
                            imgui.separator()
                            imgui.spacing()

                            imgui.text_colored("Advanced Settings:", 0.5, 0.8, 1.0)
                            imgui.spacing()

                            # Interval setting
                            cur_interval = settings.get("live_tracker_rolling_autotune_interval_ms", 5000)
                            imgui.text("Autotune Interval (ms):")
                            imgui.push_item_width(150)
                            ch, new_interval = imgui.input_int("##RollingAutotuneInterval", cur_interval, 1000)
                            imgui.pop_item_width()
                            if imgui.is_item_hovered():
                                imgui.set_tooltip("How often to apply autotune (in milliseconds). Default: 5000ms (5 seconds)")
                            if ch:
                                v = max(1000, min(30000, new_interval))  # 1-30 seconds
                                if v != cur_interval:
                                    settings.set("live_tracker_rolling_autotune_interval_ms", v)
                                    tr.rolling_autotune_interval_ms = v

                            # Window size setting
                            cur_window = settings.get("live_tracker_rolling_autotune_window_ms", 5000)
                            imgui.text("Processing Window (ms):")
                            imgui.push_item_width(150)
                            ch, new_window = imgui.input_int("##RollingAutotuneWindow", cur_window, 1000)
                            imgui.pop_item_width()
                            if imgui.is_item_hovered():
                                imgui.set_tooltip(
                                    "Size of data window to process each time (in milliseconds).\n"
                                    "Should match your buffer size. Default: 5000ms (5 seconds)"
                                )
                            if ch:
                                v = max(1000, min(30000, new_window))  # 1-30 seconds
                                if v != cur_window:
                                    settings.set("live_tracker_rolling_autotune_window_ms", v)
                                    tr.rolling_autotune_window_ms = v

                            imgui.spacing()
                            imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                            imgui.text_colored(
                                "💡 Tip: Keep your processing position at least 5-10 seconds ahead of "
                                "browser playback to ensure cleaned data is ready when needed.",
                                0.5, 1.0, 0.5
                            )
                            imgui.pop_text_wrap_pos()

            # XBVR Configuration Section
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            open_, _ = imgui.collapsing_header(
                "XBVR Integration##XBVRSettings",
                flags=imgui.TREE_NODE_DEFAULT_OPEN if not is_running else 0,
            )
            if open_:
                imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                imgui.text_colored(
                    "Browse and load videos from your XBVR library directly in the VR viewer. "
                    "Displays scene thumbnails, funscript availability, and enables remote playback control.",
                    0.7, 0.7, 0.7
                )
                imgui.pop_text_wrap_pos()
                imgui.spacing()

                # Get current settings (XBVR always enabled by default)
                xbvr_host = self.app.app_settings.get('xbvr_host', 'localhost')
                xbvr_port = self.app.app_settings.get('xbvr_port', 9999)

                if True:
                    imgui.spacing()

                    # XBVR Host
                    imgui.text("XBVR Host/IP:")
                    imgui.push_item_width(200)
                    changed, new_host = imgui.input_text(
                        "##xbvr_host",
                        str(xbvr_host),
                        256
                    )
                    imgui.pop_item_width()
                    if changed or imgui.is_item_deactivated_after_edit():
                        self.app.app_settings.set('xbvr_host', new_host)
                        self.app.app_settings.save_settings()

                    # XBVR Port
                    imgui.text("XBVR Port:")
                    imgui.push_item_width(100)
                    changed, new_port_str = imgui.input_text(
                        "##xbvr_port",
                        str(xbvr_port),
                        256
                    )
                    imgui.pop_item_width()
                    if changed or imgui.is_item_deactivated_after_edit():
                        try:
                            new_port = int(new_port_str)
                            self.app.app_settings.set('xbvr_port', new_port)
                            self.app.app_settings.save_settings()
                        except ValueError:
                            pass  # Ignore invalid port input

                    imgui.spacing()
                    imgui.text_colored(
                        f"XBVR URL: http://{xbvr_host}:{xbvr_port}",
                        0.5, 0.8, 1.0
                    )

                    imgui.spacing()
                    # Discover XBVR button (PRIMARY - positive action)
                    with primary_button_style():
                        if imgui.button("Discover XBVR Address", width=-1):
                            self._discover_xbvr_address()

                    imgui.spacing()
                    # Open XBVR Browser button (PRIMARY - positive action)
                    with primary_button_style():
                        if imgui.button("Open XBVR Browser", width=-1):
                            # Open XBVR browser in default browser
                            import webbrowser
                            local_ip = self._get_local_ip()
                            xbvr_browser_url = f"http://{local_ip}:8080/xbvr"
                            webbrowser.open(xbvr_browser_url)
                            self.app.logger.info(f"Opening XBVR browser: {xbvr_browser_url}")

            # Stash Configuration Section
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            open_, _ = imgui.collapsing_header(
                "Stash Integration##StashSettings",
                flags=imgui.TREE_NODE_DEFAULT_OPEN if not is_running else 0,
            )
            if open_:
                imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                imgui.text_colored(
                    "Browse and load videos from your Stash library directly in the VR viewer. "
                    "Access scene markers, organized collections, and interactive funscripts.",
                    0.7, 0.7, 0.7
                )
                imgui.pop_text_wrap_pos()
                imgui.spacing()

                # Get current settings (Stash default port is 9999, same as XBVR)
                stash_host = self.app.app_settings.get('stash_host', 'localhost')
                stash_port = self.app.app_settings.get('stash_port', 9999)
                stash_api_key = self.app.app_settings.get('stash_api_key', '')

                imgui.spacing()

                # Stash Host
                imgui.text("Stash Host/IP:")
                imgui.push_item_width(200)
                changed, new_host = imgui.input_text(
                    "##stash_host",
                    str(stash_host),
                    256
                )
                imgui.pop_item_width()
                if changed or imgui.is_item_deactivated_after_edit():
                    self.app.app_settings.set('stash_host', new_host)
                    self.app.app_settings.save_settings()

                # Stash Port
                imgui.text("Stash Port:")
                imgui.push_item_width(100)
                changed, new_port_str = imgui.input_text(
                    "##stash_port",
                    str(stash_port),
                    256
                )
                imgui.pop_item_width()
                if changed or imgui.is_item_deactivated_after_edit():
                    try:
                        new_port = int(new_port_str)
                        self.app.app_settings.set('stash_port', new_port)
                        self.app.app_settings.save_settings()
                    except ValueError:
                        pass  # Ignore invalid port input

                # Stash API Key
                imgui.text("API Key:")
                imgui.same_line()
                imgui.text_colored("(required for authentication)", 0.6, 0.6, 0.6)
                imgui.push_item_width(300)
                changed, new_api_key = imgui.input_text(
                    "##stash_api_key",
                    str(stash_api_key),
                    256,
                    imgui.INPUT_TEXT_PASSWORD
                )
                imgui.pop_item_width()
                if changed or imgui.is_item_deactivated_after_edit():
                    self.app.app_settings.set('stash_api_key', new_api_key)
                    self.app.app_settings.save_settings()

                imgui.spacing()
                imgui.text_colored(
                    f"Stash URL: http://{stash_host}:{stash_port}",
                    0.5, 0.8, 1.0
                )
                imgui.text_colored(
                    "Find your API key in Stash: Settings -> Security -> API Key",
                    0.5, 0.5, 0.5
                )

                imgui.spacing()
                # Open Stash Browser button (PRIMARY - positive action)
                with primary_button_style():
                    if imgui.button("Open Stash Browser", width=-1):
                        # Open Stash browser in default browser
                        import webbrowser
                        local_ip = self._get_local_ip()
                        stash_browser_url = f"http://{local_ip}:8080/stash"
                        webbrowser.open(stash_browser_url)
                        self.app.logger.info(f"Opening Stash browser: {stash_browser_url}")

            # Info Section (only when not running)
            if not is_running:
                open_, _ = imgui.collapsing_header(
                    "Requirements##NativeSyncRequirements",
                    flags=imgui.TREE_NODE_DEFAULT_OPEN,
                )
                if open_:
                    imgui.push_text_wrap_pos(imgui.get_content_region_available_width())
                    imgui.text_colored(
                        "This will start HTTP and WebSocket servers for native video playback "
                        "in browsers and VR headsets. Your video will be served at full quality "
                        "with frame-perfect synchronization.",
                        0.7, 0.7, 0.7
                    )
                    imgui.pop_text_wrap_pos()
                    imgui.spacing()

                    imgui.bullet_text("Ports 8080 (HTTP) and 8765 (WebSocket) available")
                    imgui.bullet_text("Browser with HTML5 video support")
                    imgui.bullet_text("Video can be loaded before or after starting the server")

                # Features Section
                open_, _ = imgui.collapsing_header(
                    "Features##NativeSyncFeatures",
                    flags=0,  # Collapsed by default
                )
                if open_:
                    imgui.bullet_text("Native hardware H.265/AV1 decode")
                    imgui.bullet_text("Zoom/Pan controls (+/- and WASD keys)")
                    imgui.bullet_text("Speed modes (Real Time / Slo Mo)")
                    imgui.bullet_text("Real-time FPS and resolution stats")
                    imgui.bullet_text("Interactive device control")
                    imgui.bullet_text("Funscript visualization graph")

        except Exception as e:
            imgui.text(f"Error in Streamer: {e}")
            imgui.text_colored("See logs for details.", 1.0, 0.0, 0.0)
            import traceback
            self.app.logger.error(f"Streamer tab error: {e}")
            self.app.logger.error(traceback.format_exc())

    def _start_native_sync(self):
        """Start streamer servers."""
        try:
            # Streamer can start without a video loaded (video can be loaded later)
            self.app.logger.info("Starting streamer...")

            # Enable HereSphere and XBVR browser by default
            if self._native_sync_manager:
                self._native_sync_manager.enable_heresphere = True
                self._native_sync_manager.enable_xbvr_browser = True

            self._native_sync_manager.start()

        except Exception as e:
            self.app.logger.error(f"Failed to start streamer: {e}")
            import traceback
            self.app.logger.error(traceback.format_exc())

    def _stop_native_sync(self):
        """Stop streamer servers."""
        try:
            self.app.logger.info("Stopping streamer...")
            self._native_sync_manager.stop()

        except Exception as e:
            self.app.logger.error(f"Failed to stop streamer: {e}")

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
            # Fallback - just log
            self.app.logger.info(f"URL to copy: {text}")

    def _discover_xbvr_address(self):
        """Attempt to discover XBVR on the local network."""
        import socket
        import requests
        from threading import Thread

        def scan_network():
            # Get local IP to determine subnet
            local_ip = self._get_local_ip()
            if not local_ip or not local_ip.startswith('192.168.'):
                self.app.logger.info("Could not determine local network subnet", extra={'status_message': True})
                return

            # Extract subnet (e.g., 192.168.1.x)
            subnet = '.'.join(local_ip.split('.')[:-1])
            self.app.logger.info(f"Scanning {subnet}.0/24 for XBVR on port 9999...", extra={'status_message': True})

            found = False
            for i in range(1, 255):
                if found:
                    break
                test_ip = f"{subnet}.{i}"
                try:
                    # Quick connection test on port 9999
                    response = requests.get(f"http://{test_ip}:9999", timeout=0.5)
                    if response.status_code == 200 or 'xbvr' in response.text.lower():
                        self.app.logger.info(f"✅ Found XBVR at {test_ip}:9999", extra={'status_message': True})
                        self.app.app_settings.set('xbvr_host', test_ip)
                        self.app.app_settings.set('xbvr_port', 9999)  # Also save the port
                        self.app.app_settings.save_settings()

                        # Notify integration_manager to update its XBVR client
                        if hasattr(self.app, 'integration_manager') and self.app.integration_manager:
                            self.app.integration_manager.update_xbvr_client(test_ip, 9999)

                        found = True
                except Exception:
                    pass  # Connection failed, continue

            if not found:
                self.app.logger.info("No XBVR server found on local network", extra={'status_message': True})

        # Run scan in background thread
        Thread(target=scan_network, daemon=True).start()

    def _get_local_ip(self) -> str:
        """Get local network IP address (prefer 192.168.x.x over VPN)."""
        import socket
        try:
            # Get all network interfaces
            hostname = socket.gethostname()
            addresses = socket.getaddrinfo(hostname, None, socket.AF_INET)

            # Prefer 192.168.x.x addresses
            for addr in addresses:
                ip = addr[4][0]
                if ip.startswith('192.168.'):
                    return ip

            # Fallback to any non-loopback address
            for addr in addresses:
                ip = addr[4][0]
                if not ip.startswith('127.'):
                    return ip

            return "localhost"
        except Exception:
            return "localhost"

    def _export_funscript_timeline(self, app, timeline_num):
        """Export funscript from specified timeline.

        Args:
            app: Application instance
            timeline_num: Timeline number to export (1 for primary, 2 for secondary)
        """
        app.file_manager.export_funscript_from_timeline(timeline_num)
