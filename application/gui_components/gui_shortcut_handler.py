"""Keyboard shortcut handler mixin for GUI."""
import imgui
import time

# Arrow key forward hold: ramp from real-time to max decode speed over this duration
_ARROW_ACCEL_RAMP_S = 3.0
_ARROW_TICK_BUDGET_S = 0.006  # max time to spend reading frames per GUI tick (6ms)


class ShortcutHandlerMixin:
    """Mixin providing keyboard shortcut handling methods."""

    def _handle_global_shortcuts(self):
        # CRITICAL: Check if shortcuts should be processed
        # This prevents shortcuts from firing when user is typing in text inputs
        if not self.app.shortcut_manager.should_handle_shortcuts():
            return

        io = imgui.get_io()
        app_state = self.app.app_state_ui

        current_shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})
        fs_proc = self.app.funscript_processor
        video_loaded = self.app.processor and self.app.processor.video_info and self.app.processor.total_frames > 0

        def check_and_run_shortcut(shortcut_name, action_func, *action_args):
            shortcut_str = current_shortcuts.get(shortcut_name)
            if not shortcut_str:
                return False

            map_result = self.app._map_shortcut_to_glfw_key(shortcut_str)
            if not map_result:
                return False

            mapped_key, mapped_mods_from_string = map_result

            # Check key press state ONCE and reuse (calling is_key_pressed multiple times can consume the event)
            key_pressed = imgui.is_key_pressed(mapped_key)

            if key_pressed:
                mods_match = (mapped_mods_from_string['ctrl'] == io.key_ctrl
                    and mapped_mods_from_string['alt'] == io.key_alt
                    and mapped_mods_from_string['shift'] == io.key_shift
                    and mapped_mods_from_string['super'] == io.key_super)
                if mods_match:
                    action_func(*action_args)
                    return True
            return False

        def check_key_held(shortcut_name):
            """Check if a key is being held down (for continuous navigation)"""
            shortcut_str = current_shortcuts.get(shortcut_name)
            if not shortcut_str:
                return False
            map_result = self.app._map_shortcut_to_glfw_key(shortcut_str)
            if not map_result:
                return False
            mapped_key, mapped_mods_from_string = map_result
            return (imgui.is_key_down(mapped_key) and
                   mapped_mods_from_string['ctrl'] == io.key_ctrl and
                   mapped_mods_from_string['alt'] == io.key_alt and
                   mapped_mods_from_string['shift'] == io.key_shift and
                   mapped_mods_from_string['super'] == io.key_super)

        # F1 key - Open Keyboard Shortcuts Dialog (no modifiers)
        f1_map = self.app._map_shortcut_to_glfw_key("F1")
        if f1_map:
            f1_key, f1_mods = f1_map
            if (imgui.is_key_pressed(f1_key) and
                not io.key_ctrl and not io.key_alt and not io.key_shift and not io.key_super):
                self.keyboard_shortcuts_dialog.toggle()
                return

        # Handle non-repeating shortcuts first

        # File Operations
        if check_and_run_shortcut("save_project", self._handle_save_project_shortcut):
            pass
        elif check_and_run_shortcut("open_project", self._handle_open_project_shortcut):
            pass

        # Editing — focused undo/redo (operates on last-edited timeline)
        elif check_and_run_shortcut("undo_timeline1", fs_proc.perform_undo_redo_focused, 'undo'):
            pass
        elif check_and_run_shortcut("redo_timeline1", fs_proc.perform_undo_redo_focused, 'redo'):
            pass
        # Per-timeline overrides for power users
        elif self.app.app_state_ui.show_funscript_interactive_timeline2 and (
            check_and_run_shortcut("undo_timeline2", fs_proc.perform_undo_redo, 2, 'undo')
            or check_and_run_shortcut("redo_timeline2", fs_proc.perform_undo_redo, 2, 'redo')
        ): pass

        # Playback & Navigation
        elif check_and_run_shortcut("toggle_playback", self.app.event_handlers.handle_playback_control, "play_pause"):
            pass
        elif check_and_run_shortcut("jump_to_next_point", self.app.event_handlers.handle_jump_to_point, 'next'):
            pass
        elif check_and_run_shortcut("jump_to_next_point_alt", self.app.event_handlers.handle_jump_to_point, 'next'):
            pass
        elif check_and_run_shortcut("jump_to_prev_point", self.app.event_handlers.handle_jump_to_point, 'prev'):
            pass
        elif check_and_run_shortcut("jump_to_prev_point_alt", self.app.event_handlers.handle_jump_to_point, 'prev'):
            pass
        elif video_loaded and check_and_run_shortcut("jump_to_start", self._handle_jump_to_start_shortcut):
            pass
        elif video_loaded and check_and_run_shortcut("jump_to_end", self._handle_jump_to_end_shortcut):
            pass
        elif video_loaded and check_and_run_shortcut("go_to_frame", self._handle_go_to_frame_shortcut):
            pass

        # Timeline View Controls
        elif check_and_run_shortcut("zoom_in_timeline", self._handle_zoom_in_timeline_shortcut):
            pass
        elif check_and_run_shortcut("zoom_out_timeline", self._handle_zoom_out_timeline_shortcut):
            pass

        # Window Toggles
        elif check_and_run_shortcut("toggle_video_display", self._handle_toggle_video_display_shortcut):
            pass
        elif check_and_run_shortcut("toggle_timeline2", self._handle_toggle_timeline2_shortcut):
            pass
        elif check_and_run_shortcut("toggle_3d_simulator", self._handle_toggle_3d_simulator_shortcut):
            pass
        elif check_and_run_shortcut("toggle_chapter_list", self._handle_toggle_chapter_list_shortcut):
            pass

        # Timeline Displays
        elif check_and_run_shortcut("toggle_heatmap", self._handle_toggle_heatmap_shortcut):
            pass
        elif check_and_run_shortcut("toggle_funscript_preview", self._handle_toggle_funscript_preview_shortcut):
            pass

        # Video Overlays
        elif check_and_run_shortcut("toggle_video_feed", self._handle_toggle_video_feed_shortcut):
            pass
        elif check_and_run_shortcut("toggle_waveform", self._handle_toggle_waveform_shortcut):
            pass

        # View Controls
        elif check_and_run_shortcut("reset_timeline_view", self._handle_reset_timeline_view_shortcut):
            pass

        # Video Zoom Controls
        elif check_and_run_shortcut("zoom_in_video", self._handle_zoom_in_video_shortcut):
            pass
        elif check_and_run_shortcut("zoom_out_video", self._handle_zoom_out_video_shortcut):
            pass
        elif check_and_run_shortcut("reset_video_view", self._handle_reset_video_view_shortcut):
            pass
        elif check_and_run_shortcut("toggle_fullscreen", self._handle_toggle_fullscreen_shortcut):
            pass

        # Tracking Tools
        elif check_and_run_shortcut("set_oscillation_area", self._handle_toggle_oscillation_area_mode):
            pass
        elif check_and_run_shortcut("set_user_roi", self._handle_toggle_user_roi_mode):
            pass

        # Chapters
        elif check_and_run_shortcut("set_chapter_start", self._handle_set_chapter_start_shortcut):
            pass
        elif check_and_run_shortcut("set_chapter_end", self._handle_set_chapter_end_shortcut):
            pass

        # Add Points at specific values (Number keys 0-9 and = for 100%)
        # These add a point at the current video time to the active timeline
        if video_loaded and check_and_run_shortcut("add_point_0", self._handle_add_point_at_value, 0):
            pass
        elif video_loaded and check_and_run_shortcut("add_point_10", self._handle_add_point_at_value, 10):
            pass
        elif video_loaded and check_and_run_shortcut("add_point_20", self._handle_add_point_at_value, 20):
            pass
        elif video_loaded and check_and_run_shortcut("add_point_30", self._handle_add_point_at_value, 30):
            pass
        elif video_loaded and check_and_run_shortcut("add_point_40", self._handle_add_point_at_value, 40):
            pass
        elif video_loaded and check_and_run_shortcut("add_point_50", self._handle_add_point_at_value, 50):
            pass
        elif video_loaded and check_and_run_shortcut("add_point_60", self._handle_add_point_at_value, 60):
            pass
        elif video_loaded and check_and_run_shortcut("add_point_70", self._handle_add_point_at_value, 70):
            pass
        elif video_loaded and check_and_run_shortcut("add_point_80", self._handle_add_point_at_value, 80):
            pass
        elif video_loaded and check_and_run_shortcut("add_point_90", self._handle_add_point_at_value, 90):
            pass
        elif video_loaded and check_and_run_shortcut("add_point_100", self._handle_add_point_at_value, 100):
            pass

        # Handle continuous arrow key navigation
        if video_loaded:
            self._handle_arrow_navigation()

    def _handle_arrow_navigation(self):
        """Optimized arrow key navigation with continuous scrolling support"""
        io = imgui.get_io()
        current_shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})
        current_time = time.time()

        # Update seek interval based on video FPS for natural navigation speed
        # Use FPS override if enabled, otherwise fall back to video FPS
        app_state = self.app.app_state_ui
        if app_state.fps_override_enabled and app_state.fps_override_value > 0:
            target_nav_fps = max(15, min(240, app_state.fps_override_value))
            self.arrow_key_state['seek_interval'] = 1.0 / target_nav_fps
        elif self.app.processor and self.app.processor.fps > 0:
            # Use video FPS but cap at reasonable limits for responsiveness
            video_fps = self.app.processor.fps
            # Allow faster navigation for high FPS videos, slower for low FPS
            target_nav_fps = max(15, min(60, video_fps))
            self.arrow_key_state['seek_interval'] = 1.0 / target_nav_fps

        # Get arrow key mappings
        left_shortcut = current_shortcuts.get("seek_prev_frame", "LEFT_ARROW")
        right_shortcut = current_shortcuts.get("seek_next_frame", "RIGHT_ARROW")

        left_map = self.app._map_shortcut_to_glfw_key(left_shortcut)
        right_map = self.app._map_shortcut_to_glfw_key(right_shortcut)

        if not left_map or not right_map:
            return

        left_key, left_mods = left_map
        right_key, right_mods = right_map

        # Check if keys are held down (no modifier keys for arrow navigation)
        left_held = (imgui.is_key_down(left_key) and
                    left_mods['ctrl'] == io.key_ctrl and
                    left_mods['alt'] == io.key_alt and
                    left_mods['shift'] == io.key_shift and
                    left_mods['super'] == io.key_super)

        right_held = (imgui.is_key_down(right_key) and
                     right_mods['ctrl'] == io.key_ctrl and
                     right_mods['alt'] == io.key_alt and
                     right_mods['shift'] == io.key_shift and
                     right_mods['super'] == io.key_super)

        # Update key state
        self.arrow_key_state['left_pressed'] = left_held
        self.arrow_key_state['right_pressed'] = right_held

        # Determine seek direction and apply rate limiting
        seek_direction = 0
        if left_held and not right_held:
            seek_direction = -1
        elif right_held and not left_held:
            seek_direction = 1

        # Reset direction tracking and acceleration when key is released
        if seek_direction == 0:
            self.arrow_key_state['last_direction'] = 0
            self.arrow_key_state['continuous_start_time'] = 0
            self._reading_fps_frames.clear()
            self._reading_fps_display = 0.0

        # Apply navigation with proper frame-by-frame then continuous logic
        if seek_direction != 0:
            time_since_last = current_time - self.arrow_key_state['last_seek_time']
            key_just_pressed = (left_held and imgui.is_key_pressed(left_key)) or (right_held and imgui.is_key_pressed(right_key))

            should_navigate = False
            frames_this_tick = 1  # how many frames to read this GUI tick

            if key_just_pressed:
                # INITIAL KEY PRESS: Only navigate if this is a new direction
                if self.arrow_key_state['last_direction'] != seek_direction:
                    should_navigate = True
                    self.arrow_key_state['initial_press_time'] = current_time
                    self.arrow_key_state['last_direction'] = seek_direction
                    self.arrow_key_state['continuous_start_time'] = 0
            else:
                if self.arrow_key_state['last_direction'] != seek_direction:
                    # DIRECTION CHANGED while key already held (e.g., released one of two held keys)
                    should_navigate = True
                    self.arrow_key_state['initial_press_time'] = current_time
                    self.arrow_key_state['last_direction'] = seek_direction
                    self.arrow_key_state['continuous_start_time'] = 0
                else:
                    # KEY HELD DOWN in same direction: continuous navigation
                    time_since_initial_press = current_time - self.arrow_key_state['initial_press_time']

                    if time_since_initial_press >= self.arrow_key_state['continuous_delay']:
                        # Mark when continuous mode begins
                        if self.arrow_key_state['continuous_start_time'] == 0:
                            self.arrow_key_state['continuous_start_time'] = current_time

                        base_interval = self.arrow_key_state['seek_interval']

                        if seek_direction > 0:
                            # Forward: ramp from 1 frame/tick to many frames/tick
                            # over _ARROW_ACCEL_RAMP_S seconds (pipe decode is the ceiling)
                            hold_duration = current_time - self.arrow_key_state['continuous_start_time']
                            t = min(1.0, hold_duration / _ARROW_ACCEL_RAMP_S)
                            # Speed multiplier: 1x at t=0, unbounded at t=1
                            # Use 1/(1-t) curve: 1→2→5→∞ (clamped by time budget)
                            speed = 1.0 / max(0.01, 1.0 - t)
                            video_fps = 60.0
                            if self.app.processor and self.app.processor.fps > 0:
                                video_fps = self.app.processor.fps
                            frames_this_tick = max(1, round(speed * video_fps * time_since_last))
                            should_navigate = True
                        else:
                            # Backward: 1 frame/tick at real-time (buffer is small)
                            if time_since_last >= base_interval:
                                should_navigate = True
                    # else: Still in the delay period, don't navigate (allows precise frame-by-frame)

            if should_navigate:
                if frames_this_tick <= 1:
                    self._perform_frame_seek(seek_direction)
                else:
                    # Read multiple frames from pipe, display only the last
                    self._perform_accelerated_forward_seek(frames_this_tick)
                self.arrow_key_state['last_seek_time'] = current_time

    def _perform_accelerated_forward_seek(self, frames_target):
        """Read multiple frames from pipe in one GUI tick for accelerated seeking."""
        proc = self.app.processor
        if not proc or not proc.video_info or proc.seek_in_progress:
            return

        is_actively_playing = (proc.is_processing and not proc.pause_event.is_set())
        is_tracking = self.app.tracker and self.app.tracker.tracking_active
        if is_actively_playing or is_tracking:
            # Fall back to single frame during playback/tracking
            self._perform_frame_seek(1)
            return

        total_frames = proc.total_frames
        max_frame = total_frames - 1 if total_frames > 0 else 0
        t0 = time.perf_counter()
        budget_end = t0 + _ARROW_TICK_BUDGET_S
        last_frame = None
        frames_read = 0

        for _ in range(frames_target):
            new_frame = proc.current_frame_index + 1
            if new_frame > max_frame:
                break
            frame = proc.arrow_nav_forward(new_frame)
            if frame is None:
                break
            last_frame = frame
            frames_read += 1
            # Check time budget — don't block GUI too long
            if time.perf_counter() > budget_end:
                break

        if last_frame is not None:
            with proc.frame_lock:
                proc.current_frame = last_frame
                proc._frame_version += 1

        elapsed = time.perf_counter() - t0
        self.track_frame_seek_time(elapsed * 1000, path="arrow")
        # Report frames read this tick for status bar averaging
        self._reading_fps_frames.append((time.time(), frames_read))
        self.app.app_state_ui.force_timeline_pan_to_current_frame = True
        if self.app.project_manager:
            self.app.project_manager.project_dirty = True
        self.app.energy_saver.reset_activity_timer()

    def _perform_frame_seek(self, delta_frames):
        """Arrow key navigation with rolling backward buffer"""
        if not self.app.processor or not self.app.processor.video_info:
            return

        proc = self.app.processor

        # Skip if already seeking to avoid frame jump issues
        if proc.seek_in_progress:
            return

        new_frame = proc.current_frame_index + delta_frames
        total_frames = proc.total_frames
        new_frame = max(0, min(new_frame, total_frames - 1 if total_frames > 0 else 0))

        if new_frame == proc.current_frame_index:
            return  # No change needed

        t0 = time.perf_counter()

        # Use fast pipe/buffer path when idle or paused (paused = pipe available)
        is_actively_playing = (proc.is_processing
                               and not proc.pause_event.is_set())
        is_tracking = self.app.tracker and self.app.tracker.tracking_active
        if not is_actively_playing and not is_tracking:
            if delta_frames > 0:
                frame = proc.arrow_nav_forward(new_frame)
                if frame is not None:
                    with proc.frame_lock:
                        proc.current_frame = frame
                        proc._frame_version += 1
            else:
                frame = proc.arrow_nav_backward(new_frame)
                if frame is not None:
                    with proc.frame_lock:
                        proc.current_frame = frame
                        proc._frame_version += 1
        else:
            # During tracking/processing: use standard cache-based seek
            frame_from_cache = None
            with proc.frame_cache_lock:
                if new_frame in proc.frame_cache:
                    frame_from_cache = proc.frame_cache[new_frame]
                    proc.frame_cache.move_to_end(new_frame)

            if frame_from_cache is not None:
                proc.current_frame_index = new_frame
                with proc.frame_lock:
                    proc.current_frame = frame_from_cache
                    proc._frame_version += 1
            else:
                proc.current_frame_index = new_frame
                proc.seek_video(new_frame)

        # Report to perf monitor so it shows as "ArrowNavDecode" instead of
        # being lumped into "GlobalShortcuts"
        elapsed = time.perf_counter() - t0
        self.track_frame_seek_time(elapsed * 1000, path="arrow")
        # Report 1 frame read this tick for status bar averaging
        self._reading_fps_frames.append((time.time(), 1))

        # Update UI
        self.app.app_state_ui.force_timeline_pan_to_current_frame = True
        if self.app.project_manager:
            self.app.project_manager.project_dirty = True
        self.app.energy_saver.reset_activity_timer()

    # Removed complex predictive caching - it was blocking the UI
    # Keep navigation simple: cache check first, then single frame fetch if needed

    def _handle_set_chapter_start_shortcut(self):
        """Handle keyboard shortcut for setting chapter start (I key)"""
        current_frame = self._get_current_frame_for_chapter()
        if hasattr(self, 'video_navigation_ui') and self.video_navigation_ui:
            # If chapter dialog is open, update it
            if self.video_navigation_ui.show_create_chapter_dialog or self.video_navigation_ui.show_edit_chapter_dialog:
                self.video_navigation_ui.chapter_edit_data["start_frame_str"] = str(current_frame)
                self.app.logger.info(f"Chapter start set to frame {current_frame}", extra={'status_message': True})
            else:
                # Store for future chapter creation
                self._stored_chapter_start_frame = current_frame
                self.app.logger.info(f"Chapter start marked at frame {current_frame} (Press O to set end, then Shift+C to create)", extra={'status_message': True})

    def _handle_set_chapter_end_shortcut(self):
        """Handle keyboard shortcut for setting chapter end (O key)"""
        current_frame = self._get_current_frame_for_chapter()
        if hasattr(self, 'video_navigation_ui') and self.video_navigation_ui:
            # If chapter dialog is open, update it
            if self.video_navigation_ui.show_create_chapter_dialog or self.video_navigation_ui.show_edit_chapter_dialog:
                self.video_navigation_ui.chapter_edit_data["end_frame_str"] = str(current_frame)
                self.app.logger.info(f"Chapter end set to frame {current_frame}", extra={'status_message': True})
            else:
                # Store for future chapter creation and auto-create if start is set
                self._stored_chapter_end_frame = current_frame
                if hasattr(self, '_stored_chapter_start_frame'):
                    self._auto_create_chapter_from_stored_frames()
                else:
                    self.app.logger.info(f"Chapter end marked at frame {current_frame} (Press I to set start, then Shift+C to create)", extra={'status_message': True})

    def _handle_add_point_at_value(self, value: int):
        """Add a point at the current video playhead position with the specified value (0-100).

        The point is added to the active timeline (the one with the green border).
        Uses the timeline's _add_point() method which handles snapping, undo, and cache invalidation.
        """
        if not self.app.processor or not self.app.processor.video_info:
            return

        # Get current video time
        current_frame = self.app.processor.current_frame_index
        fps = self.app.processor.fps
        if fps <= 0:
            return

        current_time_ms = int((current_frame / fps) * 1000)

        # Get the active timeline and add the point
        app_state = self.app.app_state_ui
        timeline_num = getattr(app_state, 'active_timeline_num', 1)

        # Get timeline from GUI instance (timelines are stored as timeline_editor1/2 in AppGUI)
        if timeline_num == 1:
            timeline = self.timeline_editor1
        elif timeline_num == 2:
            timeline = self.timeline_editor2
        elif timeline_num >= 3:
            timeline = self._extra_timeline_editors.get(timeline_num)
        else:
            timeline = None

        if timeline:
            timeline._add_point(current_time_ms, value)
            self.app.logger.info(f"Added point: {value}% at {current_time_ms}ms (Timeline {timeline_num})", extra={'status_message': True})
        else:
            self.app.logger.warning(f"Timeline {timeline_num} not found")

    def _get_current_frame_for_chapter(self) -> int:
        """Get current video frame for chapter operations"""
        if self.app.processor and hasattr(self.app.processor, 'current_frame_index'):
            return max(0, self.app.processor.current_frame_index)
        return 0

    def _auto_create_chapter_from_stored_frames(self):
        """Automatically create chapter when both start and end frames are marked"""
        if not (hasattr(self, '_stored_chapter_start_frame') and hasattr(self, '_stored_chapter_end_frame')):
            return

        start_frame = self._stored_chapter_start_frame
        end_frame = self._stored_chapter_end_frame

        # Ensure start is before end
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame

        # Create chapter data
        if hasattr(self, 'video_navigation_ui') and self.video_navigation_ui and self.app.funscript_processor:
            default_pos_key = self.video_navigation_ui.position_short_name_keys[0] if self.video_navigation_ui.position_short_name_keys else "N/A"
            chapter_data = {
                "start_frame_str": str(start_frame),
                "end_frame_str": str(end_frame),
                "segment_type": "SexAct",
                "position_short_name_key": default_pos_key,
                "source": "keyboard_shortcut"
            }

            self.app.funscript_processor.create_new_chapter_from_data(chapter_data)
            self.app.logger.info(f"Chapter created: frames {start_frame} to {end_frame}", extra={'status_message': True})

            # Clear stored frames
            if hasattr(self, '_stored_chapter_start_frame'):
                delattr(self, '_stored_chapter_start_frame')
            if hasattr(self, '_stored_chapter_end_frame'):
                delattr(self, '_stored_chapter_end_frame')

    # --- New Shortcut Handlers ---

    def _handle_save_project_shortcut(self):
        """Handle keyboard shortcut for saving project (CMD+S / CTRL+S)"""
        self.app.project_manager.save_project_dialog()

    def _handle_open_project_shortcut(self):
        """Handle keyboard shortcut for opening project (CMD+O / CTRL+O)"""
        self.app.project_manager.open_project_dialog()

    def _handle_jump_to_start_shortcut(self):
        """Handle keyboard shortcut for jumping to video start (HOME)"""
        if self.app.processor:
            self.app.processor.seek_video(0)
            self.app.app_state_ui.force_timeline_pan_to_current_frame = True
            if self.app.project_manager:
                self.app.project_manager.project_dirty = True
            self.app.energy_saver.reset_activity_timer()

    def _handle_jump_to_end_shortcut(self):
        """Handle keyboard shortcut for jumping to video end (END)"""
        if self.app.processor:
            last_frame = max(0, self.app.processor.total_frames - 1) if self.app.processor.total_frames > 0 else 0
            self.app.processor.seek_video(last_frame)
            self.app.app_state_ui.force_timeline_pan_to_current_frame = True

    def _handle_go_to_frame_shortcut(self):
        """Open Go to Frame popup (Ctrl+G)."""
        self._go_to_frame_open = True
        self._go_to_frame_input = ""
        self._go_to_frame_focus = True

    def _handle_zoom_in_timeline_shortcut(self):
        """Handle keyboard shortcut for zooming in timeline (CMD+= / CTRL+=)"""
        # Apply zoom in with scale factor (0.85 = zoom in)
        app_state = self.app.app_state_ui
        scale_factor = 0.85

        # Zoom around current time (center of view)
        effective_total_duration_s, _, _ = self.app.get_effective_video_duration_params()
        effective_total_duration_ms = effective_total_duration_s * 1000.0

        # Get current center time
        if self.timeline_editor1:
            # Use timeline 1's center marker position
            center_time_ms = app_state.timeline_pan_offset_ms
        else:
            center_time_ms = 0.0

        # Apply zoom
        min_ms_per_px, max_ms_per_px = 0.01, 2000.0
        old_zoom = app_state.timeline_zoom_factor_ms_per_px
        app_state.timeline_zoom_factor_ms_per_px = max(
            min_ms_per_px,
            min(app_state.timeline_zoom_factor_ms_per_px * scale_factor, max_ms_per_px),
        )

        # Adjust pan offset to keep center time roughly in place
        if old_zoom != app_state.timeline_zoom_factor_ms_per_px:
            self.app.energy_saver.reset_activity_timer()

    def _handle_zoom_out_timeline_shortcut(self):
        """Handle keyboard shortcut for zooming out timeline (CMD+- / CTRL+-)"""
        # Apply zoom out with scale factor (1.15 = zoom out)
        app_state = self.app.app_state_ui
        scale_factor = 1.15

        # Zoom around current time (center of view)
        effective_total_duration_s, _, _ = self.app.get_effective_video_duration_params()
        effective_total_duration_ms = effective_total_duration_s * 1000.0

        # Get current center time
        if self.timeline_editor1:
            # Use timeline 1's center marker position
            center_time_ms = app_state.timeline_pan_offset_ms
        else:
            center_time_ms = 0.0

        # Apply zoom
        min_ms_per_px, max_ms_per_px = 0.01, 2000.0
        old_zoom = app_state.timeline_zoom_factor_ms_per_px
        app_state.timeline_zoom_factor_ms_per_px = max(
            min_ms_per_px,
            min(app_state.timeline_zoom_factor_ms_per_px * scale_factor, max_ms_per_px),
        )

        # Adjust pan offset to keep center time roughly in place
        if old_zoom != app_state.timeline_zoom_factor_ms_per_px:
            self.app.energy_saver.reset_activity_timer()

    def _handle_toggle_video_display_shortcut(self):
        """Handle keyboard shortcut for toggling video display (V)"""
        app_state = self.app.app_state_ui
        # Only allow toggle in floating mode - in fixed mode video display is always shown
        if app_state.ui_layout_mode == "floating":
            app_state.show_video_display_window = not app_state.show_video_display_window
            if self.app.project_manager:
                self.app.project_manager.project_dirty = True
            status = "shown" if app_state.show_video_display_window else "hidden"
            self.app.logger.info(f"Video display {status}", extra={'status_message': True})
        else:
            self.app.logger.info("Video display toggle only available in floating mode", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def _handle_toggle_timeline2_shortcut(self):
        """Handle keyboard shortcut for toggling timeline 2 (T)"""
        app_state = self.app.app_state_ui
        app_state.show_funscript_interactive_timeline2 = not app_state.show_funscript_interactive_timeline2
        if self.app.project_manager:
            self.app.project_manager.project_dirty = True
        status = "shown" if app_state.show_funscript_interactive_timeline2 else "hidden"
        self.app.logger.info(f"Timeline 2 {status}", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def _handle_toggle_3d_simulator_shortcut(self):
        """Handle keyboard shortcut for toggling 3D simulator (S)"""
        app_state = self.app.app_state_ui
        app_state.show_simulator_3d = not app_state.show_simulator_3d
        if self.app.project_manager:
            self.app.project_manager.project_dirty = True
        status = "shown" if app_state.show_simulator_3d else "hidden"
        self.app.logger.info(f"3D Simulator {status}", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def _handle_toggle_chapter_list_shortcut(self):
        """Handle keyboard shortcut for toggling chapter list (L)"""
        app_state = self.app.app_state_ui
        if not hasattr(app_state, 'show_chapter_list_window'):
            app_state.show_chapter_list_window = False
        app_state.show_chapter_list_window = not app_state.show_chapter_list_window
        if self.app.project_manager:
            self.app.project_manager.project_dirty = True
        status = "shown" if app_state.show_chapter_list_window else "hidden"
        self.app.logger.info(f"Chapter List {status}", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def _handle_toggle_heatmap_shortcut(self):
        """Handle keyboard shortcut for toggling heatmap (H)"""
        app_state = self.app.app_state_ui
        app_state.show_heatmap = not app_state.show_heatmap
        if self.app.project_manager:
            self.app.project_manager.project_dirty = True
        status = "shown" if app_state.show_heatmap else "hidden"
        self.app.logger.info(f"Heatmap {status}", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def _handle_toggle_funscript_preview_shortcut(self):
        """Handle keyboard shortcut for toggling funscript preview bar (P)"""
        app_state = self.app.app_state_ui
        app_state.show_funscript_timeline = not app_state.show_funscript_timeline
        if self.app.project_manager:
            self.app.project_manager.project_dirty = True
        status = "shown" if app_state.show_funscript_timeline else "hidden"
        self.app.logger.info(f"Funscript Preview {status}", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def _handle_toggle_video_feed_shortcut(self):
        """Handle keyboard shortcut for toggling video feed overlay (F)"""
        app_state = self.app.app_state_ui
        app_state.show_video_feed = not app_state.show_video_feed
        self.app.app_settings.set("show_video_feed", app_state.show_video_feed)
        if self.app.project_manager:
            self.app.project_manager.project_dirty = True
        status = "shown" if app_state.show_video_feed else "hidden"
        self.app.logger.info(f"Video Feed {status}", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def _handle_toggle_waveform_shortcut(self):
        """Handle keyboard shortcut for toggling audio waveform (W)"""
        app_state = self.app.app_state_ui
        app_state.show_audio_waveform = not app_state.show_audio_waveform
        if self.app.project_manager:
            self.app.project_manager.project_dirty = True
        status = "shown" if app_state.show_audio_waveform else "hidden"
        self.app.logger.info(f"Audio Waveform {status}", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def _handle_reset_timeline_view_shortcut(self):
        """Handle keyboard shortcut for resetting timeline zoom/pan (R)"""
        app_state = self.app.app_state_ui

        # Reset zoom to default (20.0 ms per pixel is a good default)
        app_state.timeline_zoom_factor_ms_per_px = 20.0

        # Reset pan to start
        app_state.timeline_pan_offset_ms = 0.0

        # Force timeline to pan to current frame
        app_state.force_timeline_pan_to_current_frame = True

        self.app.logger.info("Timeline view reset to default", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def _handle_toggle_oscillation_area_mode(self):
        """Handle keyboard shortcut for toggling oscillation area drawing mode (X)"""
        # Only available when an oscillation tracker is active
        tracker = self.app.tracker
        if not tracker:
            return

        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        tracker_info = discovery.get_tracker_info(self.app.app_state_ui.selected_tracker_name)
        if not tracker_info or 'oscillation' not in tracker_info.display_name.lower():
            self.app.logger.info("Oscillation area shortcut requires an oscillation tracker.", extra={'status_message': True})
            return

        if self.app.is_setting_oscillation_area_mode:
            self.app.exit_set_oscillation_area_mode()
        else:
            self.app.enter_set_oscillation_area_mode()

    def _handle_toggle_user_roi_mode(self):
        """Handle keyboard shortcut for toggling User ROI drawing mode (U)"""
        tracker = self.app.tracker
        if not tracker:
            return

        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        tracker_info = discovery.get_tracker_info(self.app.app_state_ui.selected_tracker_name)
        if not tracker_info or not tracker_info.requires_intervention:
            self.app.logger.info("User ROI shortcut requires a User ROI tracker.", extra={'status_message': True})
            return

        if self.app.is_setting_user_roi_mode:
            self.app.exit_set_user_roi_mode()
        else:
            self.app.enter_set_user_roi_mode()

    def _handle_zoom_in_video_shortcut(self):
        """Handle keyboard shortcut for zooming in video (Cmd+Shift+= / Ctrl+Shift+=)"""
        self.app.app_state_ui.adjust_video_zoom(1.2)

    def _handle_zoom_out_video_shortcut(self):
        """Handle keyboard shortcut for zooming out video (Cmd+Shift+- / Ctrl+Shift+-)"""
        self.app.app_state_ui.adjust_video_zoom(1.0 / 1.2)

    def _handle_reset_video_view_shortcut(self):
        """Handle keyboard shortcut for resetting video zoom/pan (Cmd+Shift+R / Ctrl+Shift+R)"""
        self.app.app_state_ui.reset_video_zoom_pan()
        self.app.logger.info("Video zoom/pan reset", extra={'status_message': True})

    def _handle_toggle_fullscreen_shortcut(self):
        """Handle keyboard shortcut for toggling fullscreen (F11) - mpv supporter feature."""
        from application.utils.feature_detection import is_feature_available as _is_feature_available
        if not _is_feature_available("patreon_features"):
            return
        mpv = getattr(self.app, '_mpv_controller', None)
        if mpv is None:
            return
        if mpv.is_active:
            mpv.stop()
        else:
            file_manager = getattr(self.app, 'file_manager', None)
            video_path = file_manager.video_path if file_manager else None
            if not video_path:
                return
            processor = self.app.processor
            start_frame = processor.current_frame_index if processor else 0
            mpv.start(video_path, start_frame=start_frame, fullscreen=True)

    def _handle_energy_saver_interaction_detection(self):
        io = imgui.get_io()
        interaction_detected_this_frame = False
        current_mouse_pos = io.mouse_pos
        if current_mouse_pos[0] != self.last_mouse_pos_for_energy_saver[0] or current_mouse_pos[1] != self.last_mouse_pos_for_energy_saver[1]:
            interaction_detected_this_frame = True
            self.last_mouse_pos_for_energy_saver = current_mouse_pos

        # REFACTORED for readability and maintainability
        buttons = (0, 1, 2)
        if (any(imgui.is_mouse_clicked(b) or imgui.is_mouse_double_clicked(b) for b in buttons)
            or io.mouse_wheel != 0.0
            or io.want_text_input
            or imgui.is_mouse_dragging(0)
            or imgui.is_any_item_active()
            or imgui.is_any_item_focused()):
                interaction_detected_this_frame = True
        if hasattr(io, 'keys_down'):
            for i in range(len(io.keys_down)):
                if imgui.is_key_pressed(i): interaction_detected_this_frame = True; break
        if interaction_detected_this_frame:
            self.app.energy_saver.reset_activity_timer()
