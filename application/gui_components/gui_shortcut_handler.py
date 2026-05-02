"""Keyboard shortcut handler mixin for GUI."""
import imgui
import time

from common.frame_utils import frame_to_ms

_ARROW_ACCEL_RAMP_S = 3.0
_ARROW_TICK_BUDGET_S = 0.006
# Right-arrow hold timings: tap < REALTIME, then REALTIME playback, then
# MAX_SPEED at 3s so the user gets a solid window of normal-speed playback
# before we flip into fast-scrub.
_ARROW_REALTIME_HOLD_S = 0.25
_ARROW_MAXSPEED_HOLD_S = 3.0

# Time-based key-repeat for held shortcuts. We do our own pacing off
# imgui.is_key_down instead of relying on imgui.is_key_pressed(key, True),
# which needs GLFW to forward OS KEY_REPEAT events — BLE HID keyboards and
# some remap tools (e.g. Contour Shuttle Pro V2 profiles, DaVinci Speed
# Editor unlockers) never emit those, so "hold Left to pan frame-by-frame"
# otherwise requires a tap per frame. Values chosen to roughly match macOS
# default auto-repeat (~400ms to first repeat, ~25 Hz after).
_REPEAT_INITIAL_DELAY_S = 0.40
_REPEAT_INTERVAL_S = 0.040


def _nav_dbg_enabled(app) -> bool:
    try:
        return app.app_settings.config.navigation.debug_logging
    except Exception:
        return False


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

        def check_and_run_shortcut(shortcut_name, action_func, *action_args, repeat=False):
            shortcut_str = current_shortcuts.get(shortcut_name)
            if not shortcut_str:
                return False

            map_result = self.app._map_shortcut_to_glfw_key(shortcut_str)
            if not map_result:
                return False

            mapped_key, mapped_mods_from_string = map_result

            mods_match = (mapped_mods_from_string['ctrl'] == io.key_ctrl
                and mapped_mods_from_string['alt'] == io.key_alt
                and mapped_mods_from_string['shift'] == io.key_shift
                and mapped_mods_from_string['super'] == io.key_super)

            if repeat:
                # Held-key path: own the repeat cadence so devices that do
                # not forward OS KEY_REPEAT events still fire continuously
                # while the key is down.
                if self._time_based_repeat_fire(
                        shortcut_name, mapped_key, mods_match):
                    action_func(*action_args)
                    return True
                return False

            # Non-repeat path: single fire on initial press.
            try:
                key_pressed = imgui.is_key_pressed(mapped_key, False)
            except TypeError:
                key_pressed = imgui.is_key_pressed(mapped_key)

            if key_pressed and mods_match:
                action_func(*action_args)
                return True

            # Char-input fallback (BLE HID, remapped hardware that reaches
            # the OS via WM_CHAR / text-input events only, never key events
            # — matches the community-reported Shuttle Pro V2 / Speed Editor
            # pattern where Ctrl-combos work but "F" / "Space" don't). Only
            # eligible when:
            #   - shortcut has no modifiers AND no modifiers are currently
            #     held (so a partial Ctrl+F, Ctrl via key path and 'F' via
            #     char path, does not spuriously fire the plain-F shortcut),
            #   - imgui doesn't already see the key as down: if key events
            #     are flowing normally, holding the key would produce OS
            #     char-repeat which would otherwise re-fire the shortcut
            #     every char event and regress the "tap-only" semantics of
            #     non-repeat shortcuts.
            # Populated by app_gui._on_char_input during glfw.poll_events();
            # cleared at the end of this dispatcher.
            virtual = getattr(self, '_virtual_pressed_keys', None)
            if (virtual
                    and mapped_key in virtual
                    and not imgui.is_key_down(mapped_key)
                    and not any(mapped_mods_from_string.values())
                    and not (io.key_ctrl or io.key_alt
                             or io.key_shift or io.key_super)):
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

        # Editing, unified undo/redo (single chronological stack)
        elif check_and_run_shortcut("undo_timeline1", self._handle_unified_undo):
            pass
        elif check_and_run_shortcut("redo_timeline1", self._handle_unified_redo):
            pass

        # Playback & Navigation
        elif check_and_run_shortcut("toggle_playback", self.app.event_handlers.handle_playback_control, "play_pause"):
            pass
        elif check_and_run_shortcut("jump_to_next_point", self.app.event_handlers.handle_jump_to_point, 'next', repeat=True):
            pass
        elif check_and_run_shortcut("jump_to_next_point_alt", self.app.event_handlers.handle_jump_to_point, 'next', repeat=True):
            pass
        elif check_and_run_shortcut("jump_to_prev_point", self.app.event_handlers.handle_jump_to_point, 'prev', repeat=True):
            pass
        elif check_and_run_shortcut("jump_to_prev_point_alt", self.app.event_handlers.handle_jump_to_point, 'prev', repeat=True):
            pass
        elif check_and_run_shortcut("jump_to_next_point_any", self.app.event_handlers.handle_jump_to_point, 'next', True, repeat=True):
            pass
        elif check_and_run_shortcut("jump_to_prev_point_any", self.app.event_handlers.handle_jump_to_point, 'prev', True, repeat=True):
            pass
        elif video_loaded and check_and_run_shortcut("jump_to_start", self._handle_jump_to_start_shortcut):
            pass
        elif video_loaded and check_and_run_shortcut("jump_to_end", self._handle_jump_to_end_shortcut):
            pass
        elif video_loaded and check_and_run_shortcut("go_to_frame", self._handle_go_to_frame_shortcut):
            pass
        elif video_loaded and check_and_run_shortcut("seek_next_n_frames", self._handle_seek_n_frames, 1, repeat=True):
            pass
        elif video_loaded and check_and_run_shortcut("seek_prev_n_frames", self._handle_seek_n_frames, -1, repeat=True):
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
        elif check_and_run_shortcut("toggle_script_gauge", self._handle_toggle_script_gauge_shortcut):
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
        elif check_and_run_shortcut("toggle_timeline_smooth_curve", self._handle_toggle_timeline_smooth_curve_shortcut):
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

        # Snap nearest point to playhead
        elif video_loaded and check_and_run_shortcut("snap_nearest_to_playhead", self._handle_snap_nearest_to_playhead):
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

        if video_loaded:
            self._handle_arrow_navigation()

        # Drop virtual key presses now that this frame's dispatcher has had
        # a chance to consume them. Next frame's glfw.poll_events() will
        # repopulate from fresh char events.
        virtual = getattr(self, '_virtual_pressed_keys', None)
        if virtual:
            virtual.clear()

    def _time_based_repeat_fire(self, state_key, mapped_key, mods_match,
                                initial_delay=_REPEAT_INITIAL_DELAY_S,
                                interval=_REPEAT_INTERVAL_S):
        """Return True if a held shortcut should fire this frame.

        Keyed off `imgui.is_key_down` + per-shortcut timestamps rather
        than `imgui.is_key_pressed(key, repeat=True)`. The latter needs
        GLFW to forward OS KEY_REPEAT events; some devices never emit
        them, so the built-in auto-repeat silently fails to fire.

        Fires once on fresh keydown, then again after `initial_delay`,
        then paced by `interval`. Releasing the key (or mods no longer
        matching) clears the timer so the next press starts clean.
        """
        state_dict = getattr(self, '_shortcut_repeat_state', None)
        if state_dict is None:
            self._shortcut_repeat_state = {}
            state_dict = self._shortcut_repeat_state

        if not (mods_match and imgui.is_key_down(mapped_key)):
            state_dict.pop(state_key, None)
            return False

        now = time.monotonic()
        state = state_dict.get(state_key)
        if state is None:
            state_dict[state_key] = {'pressed_at': now, 'last_fire': now}
            return True

        if (now - state['pressed_at']) < initial_delay:
            return False
        if (now - state['last_fire']) < interval:
            return False
        state['last_fire'] = now
        return True

    def _nav_log(self, action, frm, to, path, dur_ms, **extra):
        """One-line nav debug trace. Gated on debug_nav_logging setting."""
        if not _nav_dbg_enabled(self.app):
            return
        kv = " ".join(f"{k}={v}" for k, v in extra.items())
        self.app.logger.info(
            f"NAV {action:<14} from={frm} to={to} delta={to - frm:+d} "
            f"path={path:<10} dur={dur_ms:.1f}ms {kv}")

    def _handle_arrow_navigation(self):
        """Right arrow: tap = one frame forward. Hold >= HOLD_PLAYBACK_S
        = forward playback. Hold >= HOLD_MAXSPEED_S = MAX_SPEED playback.
        Release = stop playback.

        Left arrow: tap = one frame back. Hold = repeated step-back paced
        by our own time-based repeat (we can't reverse-play the engine,
        and the OS repeat path is unreliable on some BLE keyboards)."""
        io = imgui.get_io()
        current_shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})
        current_time = time.time()

        left_shortcut = current_shortcuts.get("seek_prev_frame", "LEFT_ARROW")
        right_shortcut = current_shortcuts.get("seek_next_frame", "RIGHT_ARROW")

        left_map = self.app._map_shortcut_to_glfw_key(left_shortcut)
        right_map = self.app._map_shortcut_to_glfw_key(right_shortcut)
        if not left_map or not right_map:
            return

        left_key, left_mods = left_map
        right_key, right_mods = right_map

        def _mods_match(m):
            return (m['ctrl'] == io.key_ctrl and m['alt'] == io.key_alt
                    and m['shift'] == io.key_shift and m['super'] == io.key_super)

        left_held = imgui.is_key_down(left_key) and _mods_match(left_mods)
        right_held = imgui.is_key_down(right_key) and _mods_match(right_mods)

        self.arrow_key_state['left_pressed'] = left_held
        self.arrow_key_state['right_pressed'] = right_held

        seek_direction = 0
        if left_held and not right_held:
            seek_direction = -1
        elif right_held and not left_held:
            seek_direction = 1

        # Release: stop any playback we started and reset timing.
        if seek_direction == 0:
            if self.arrow_key_state.get('arrow_triggered_playback'):
                self._end_arrow_playback()
            self.arrow_key_state['last_direction'] = 0
            self.arrow_key_state['nav_phase'] = 'idle'
            self._reading_fps_frames.clear()
            self._reading_fps_display = 0.0
            return

        # Direction changed (swapped keys, or keydown from rest). Mark the
        # fresh press, fire a single frame-step (tap ergonomics), and reset
        # the hold timer. If we were driving playback the other way, end
        # it first so we don't seek backwards through a live pipe.
        if self.arrow_key_state.get('last_direction') != seek_direction:
            if self.arrow_key_state.get('arrow_triggered_playback'):
                self._end_arrow_playback()
            self.arrow_key_state['last_direction'] = seek_direction
            self.arrow_key_state['initial_press_time'] = current_time
            self._perform_frame_seek(seek_direction)
            self.arrow_key_state['last_seek_time'] = current_time
            return

        hold_time = current_time - self.arrow_key_state.get(
            'initial_press_time', current_time)

        # Left held: repeated step-back on our own timer. Previously this
        # used imgui.is_key_pressed(left_key, True), which needed OS
        # KEY_REPEAT events — BLE HID / remapped-hardware keyboards don't
        # always emit them, so "hold Left to pan" would degrade to one
        # step per physical keypress. Time-based pacing fires identically
        # on all devices as long as is_key_down stays True, and reuses
        # the arrow_key_state timestamps already tracked for the tap +
        # hold-to-play state machine above.
        if seek_direction < 0:
            initial = self.arrow_key_state.get(
                'initial_press_time', current_time)
            last_fire = self.arrow_key_state.get('last_seek_time', 0.0)
            if ((current_time - initial) >= _REPEAT_INITIAL_DELAY_S
                    and (current_time - last_fire) >= _REPEAT_INTERVAL_S):
                self._perform_frame_seek(-1)
                self.arrow_key_state['last_seek_time'] = current_time
            return

        # Right held: transition to hold-to-play once the tap window is past.
        # Before HOLD_PLAYBACK_S: do nothing extra (the single step fired on
        # the initial keydown; we don't step on auto-repeat because that
        # would race with the hold-to-play decision).
        if hold_time < _ARROW_REALTIME_HOLD_S:
            return

        # Past threshold: engage playback (REALTIME first, MAX_SPEED at the
        # longer threshold). _maybe_drive_arrow_playback is idempotent and
        # keeps transitioning as hold_time grows.
        if self._maybe_drive_arrow_playback(current_time):
            self.arrow_key_state['last_seek_time'] = current_time

    def _arrow_playback_available(self) -> bool:
        proc = self.app.processor
        if not proc or not proc.video_info:
            return False
        tracker = getattr(self.app, 'tracker', None)
        if tracker is not None and getattr(tracker, 'tracking_active', False):
            return False
        mpv_ctl = getattr(self.app, '_mpv_controller', None)
        if mpv_ctl is not None and getattr(mpv_ctl, 'is_active', False):
            return False
        already_playing = proc.is_processing and not proc.pause_event.is_set()
        if already_playing and not self.arrow_key_state.get('arrow_triggered_playback'):
            return False
        return True

    def _maybe_drive_arrow_playback(self, current_time: float) -> bool:
        if not self._arrow_playback_available():
            return False

        initial = self.arrow_key_state.get('initial_press_time') or current_time
        hold = current_time - initial
        if hold < _ARROW_REALTIME_HOLD_S:
            return False

        try:
            from config.constants import ProcessingSpeedMode
        except Exception:
            return False

        target_mode = (ProcessingSpeedMode.MAX_SPEED
                       if hold >= _ARROW_MAXSPEED_HOLD_S
                       else ProcessingSpeedMode.REALTIME)
        target_phase = 'maxspeed' if hold >= _ARROW_MAXSPEED_HOLD_S else 'realtime'

        if not self.arrow_key_state.get('arrow_triggered_playback'):
            self._begin_arrow_playback(target_mode, target_phase)
        elif self.arrow_key_state.get('nav_phase') != target_phase:
            self.app.app_state_ui.selected_processing_speed_mode = target_mode
            self.arrow_key_state['nav_phase'] = target_phase
        return True

    def _begin_arrow_playback(self, mode, phase: str) -> None:
        app_state = self.app.app_state_ui
        self.arrow_key_state['saved_speed_mode'] = getattr(
            app_state, 'selected_processing_speed_mode', None)
        app_state.selected_processing_speed_mode = mode
        self.arrow_key_state['arrow_triggered_playback'] = True
        self.arrow_key_state['nav_phase'] = phase
        proc = self.app.processor
        already_playing = proc.is_processing and not proc.pause_event.is_set()
        if not already_playing:
            proc.start_processing()

    def _end_arrow_playback(self) -> None:
        proc = self.app.processor
        saved_mode = self.arrow_key_state.get('saved_speed_mode')
        self.arrow_key_state['arrow_triggered_playback'] = False
        self.arrow_key_state['saved_speed_mode'] = None
        self.arrow_key_state['nav_phase'] = 'idle'
        if proc and proc.is_processing and not proc.pause_event.is_set():
            try:
                proc.pause_processing()
            except Exception as e:
                self.logger.debug(f"arrow playback pause failed: {e}")
        if saved_mode is not None:
            try:
                self.app.app_state_ui.selected_processing_speed_mode = saved_mode
            except Exception:
                pass

    def _auto_pause_for_nav(self, proc, reason: str) -> None:
        """Pause playback so arrow-nav can run on the well-behaved paused
        path. Without this, every arrow press mid-playback called seek_video,
        which tore down the ffmpeg pipe and sometimes lost the subprocess
        entirely. Triggered by left arrow during play, or by any nav during
        right-held playback that we ourselves engaged."""
        if proc.is_processing and not proc.pause_event.is_set():
            self._nav_log('auto_pause', proc.current_frame_index,
                          proc.current_frame_index, 'pause_only', 0.0,
                          reason=reason)
            try:
                proc.pause_processing()
            except Exception as e:
                self.app.logger.debug(f"auto-pause for nav failed: {e}")

    def _perform_accelerated_seek(self, frames_delta):
        # Small delta: step one frame at a time. Large delta: one seek.
        proc = self.app.processor
        if not proc or not proc.video_info:
            return

        is_actively_playing = (proc.is_processing and not proc.pause_event.is_set())
        is_tracking = self.app.tracker and self.app.tracker.tracking_active
        if is_tracking:
            self._perform_frame_seek(1 if frames_delta >= 0 else -1)
            return
        if is_actively_playing:
            # If the user is running their own playback (spacebar) and holds
            # an arrow, don't hijack the transport with a scrub ramp. Just
            # let playback run. Progressive-hold-forward already has its own
            # path that took over when appropriate.
            if not self.arrow_key_state.get('arrow_triggered_playback'):
                self._nav_log('accel_skip', proc.current_frame_index,
                              proc.current_frame_index, 'user_playing',
                              0.0, reason='not_hijacking')
                return
            # Otherwise this is our own arrow-triggered playback; leave it be.
            return

        total_frames = proc.total_frames
        max_frame = total_frames - 1 if total_frames > 0 else 0
        cur = proc.current_frame_index
        target = max(0, min(max_frame, cur + int(frames_delta)))
        if target == cur:
            return

        t0 = time.perf_counter()
        forward = target > cur
        jump_style = "single-jump"

        # Async path: advances cursor, enqueues background fetch, returns None
        # on cache miss. Cache hit returns the frame synchronously.
        if forward:
            last_frame = proc.arrow_nav_forward(target)
        else:
            last_frame = proc.arrow_nav_backward(target)
        frames_read = 1 if last_frame is not None else 0

        elapsed = time.perf_counter() - t0

        if last_frame is not None:
            with proc.frame_lock:
                proc.current_frame = last_frame
                proc._frame_version += 1

        self._nav_log('accel_seek', cur, target, jump_style, elapsed * 1000,
                      frames_read=frames_read, ok=bool(last_frame is not None))

        self.track_frame_seek_time(elapsed * 1000, path="arrow")
        self._reading_fps_frames.append((time.time(), frames_read))
        self.app.app_state_ui.force_timeline_pan_to_current_frame = True
        if self.app.project_manager:
            self.app.project_manager.project_dirty = True
        self.app.energy_saver.reset_activity_timer()

    def _perform_frame_seek(self, delta_frames):
        if not self.app.processor or not self.app.processor.video_info:
            return

        proc = self.app.processor

        new_frame = proc.current_frame_index + delta_frames
        total_frames = proc.total_frames
        new_frame = max(0, min(new_frame, total_frames - 1 if total_frames > 0 else 0))

        if new_frame == proc.current_frame_index:
            return

        try:
            self.app.app_state_ui.last_nav_activity_time = time.monotonic()
        except Exception:
            pass

        t0 = time.perf_counter()
        cur = proc.current_frame_index

        is_actively_playing = (proc.is_processing
                               and not proc.pause_event.is_set())
        is_tracking = self.app.tracker and self.app.tracker.tracking_active
        # Auto-pause on arrow nav mid-playback so we can run the paused
        # path (cache lookup + disposable frame fetch) instead of tearing
        # down the live ffmpeg pipe with a seek_video.
        if is_actively_playing and not is_tracking:
            self._auto_pause_for_nav(proc, reason='frame_seek')
            is_actively_playing = False

        if not is_actively_playing and not is_tracking:
            if delta_frames > 0:
                frame = proc.arrow_nav_forward(new_frame)
                path_taken = 'nav_fwd'
            else:
                frame = proc.arrow_nav_backward(new_frame)
                path_taken = 'nav_back'
            if frame is not None:
                with proc.frame_lock:
                    proc.current_frame = frame
                    proc._frame_version += 1
            try:
                mpv_disp = getattr(self.app.gui_instance,
                                   'mpv_display', None) if self.app.gui_instance else None
                if mpv_disp is not None and getattr(mpv_disp, 'is_loaded', False):
                    if delta_frames == 1 and hasattr(mpv_disp, 'step_forward'):
                        mpv_disp.step_forward()
                    elif delta_frames == -1 and hasattr(mpv_disp, 'step_backward'):
                        mpv_disp.step_backward()
                    elif hasattr(proc, '_mpv_seek_to_frame'):
                        proc._mpv_seek_to_frame(new_frame)
            except Exception:
                pass
            self._nav_log('frame_seek', cur, new_frame, path_taken,
                          (time.perf_counter() - t0) * 1000,
                          ok=bool(frame is not None))
        else:
            # Only reached during active tracking. Cache or fall through to
            # a seek_video, which the tracker's pipe handles via its own
            # seek plumbing.
            frame_from_cache = proc.get_cached_frame(new_frame)
            if frame_from_cache is not None:
                proc.current_frame_index = new_frame
                with proc.frame_lock:
                    proc.current_frame = frame_from_cache
                    proc._frame_version += 1
                self._nav_log('frame_seek', cur, new_frame, 'cache_hit',
                              (time.perf_counter() - t0) * 1000)
            else:
                proc.current_frame_index = new_frame
                proc.seek_video(new_frame)
                self._nav_log('frame_seek', cur, new_frame, 'seek_video',
                              (time.perf_counter() - t0) * 1000)

        elapsed = time.perf_counter() - t0
        self.track_frame_seek_time(elapsed * 1000, path="arrow")
        self._reading_fps_frames.append((time.time(), 1))

        self.app.app_state_ui.force_timeline_pan_to_current_frame = True
        if self.app.project_manager:
            self.app.project_manager.project_dirty = True
        self.app.energy_saver.reset_activity_timer()


    def _handle_unified_undo(self):
        desc = self.app.undo_manager.undo(self.app)
        if desc:
            self.app.notify(f"Undo: {desc}", "info", 1.5)

    def _handle_unified_redo(self):
        desc = self.app.undo_manager.redo(self.app)
        if desc:
            self.app.notify(f"Redo: {desc}", "info", 1.5)

    def _handle_snap_nearest_to_playhead(self):
        """Handle snap nearest point to playhead shortcut (delegates to active timeline)."""
        active_tl_num = getattr(self.app.app_state_ui, 'active_timeline_num', 1)
        tl = None
        if active_tl_num == 1:
            tl = self.timeline_editor1
        elif active_tl_num == 2:
            tl = getattr(self, 'timeline_editor2', None)
        else:
            tl = self._extra_timeline_editors.get(active_tl_num)
        if tl is not None:
            from application.classes.timeline_ops import snap_to_playhead
            snap_to_playhead(tl)

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

        current_time_ms = getattr(self.app.processor, 'playhead_override_ms', None)
        if current_time_ms is None:
            current_time_ms = frame_to_ms(current_frame, fps)

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
            # If the user has manually selected points, move those instead of
            # adding/moving at the playhead. Playhead acts only as a fallback.
            selected_idxs = []
            if getattr(timeline, 'multi_selected_action_indices', None):
                try:
                    selected_idxs = timeline._resolve_selected_indices() or []
                except Exception:
                    selected_idxs = []
            if selected_idxs:
                actions = timeline._get_actions()
                from application.classes.undo_manager import MovePointCmd
                changed = 0
                for sidx in selected_idxs:
                    if 0 <= sidx < len(actions):
                        old_value = actions[sidx]['pos']
                        if old_value == value:
                            continue
                        actions[sidx]['pos'] = value
                        self.app.undo_manager.push_done(MovePointCmd(
                            timeline.timeline_num, sidx,
                            actions[sidx]['at'], old_value,
                            actions[sidx]['at'], value))
                        changed += 1
                if changed:
                    fs, axis = timeline._get_target_funscript_details()
                    if fs:
                        fs._invalidate_cache(axis or 'both')
                    timeline.multi_selected_action_indices = {
                        timeline._action_key(actions[i]) for i in selected_idxs if 0 <= i < len(actions)
                    }
                    self.app.funscript_processor._post_mutation_refresh(timeline_num, "Move Point")
                    timeline.invalidate_cache()
                    self.app.logger.info(f"Moved {changed} selected point(s) to {value}% (Timeline {timeline_num})", extra={'status_message': True})
                return

            # Check if a point already exists at this time - move it instead of adding
            actions = timeline._get_actions()
            if actions:
                from bisect import bisect_left
                timestamps = [a['at'] for a in actions]
                # Snap tolerance: half a frame in ms
                tol_ms = max(1, int(500 / fps))
                idx = bisect_left(timestamps, current_time_ms)
                existing_idx = None
                for candidate in (idx - 1, idx):
                    if 0 <= candidate < len(actions):
                        if abs(actions[candidate]['at'] - current_time_ms) <= tol_ms:
                            existing_idx = candidate
                            break

                if existing_idx is not None:
                    old_value = actions[existing_idx]['pos']
                    actions[existing_idx]['pos'] = value
                    fs, axis = timeline._get_target_funscript_details()
                    if fs:
                        fs._invalidate_cache(axis or 'both')
                    from application.classes.undo_manager import MovePointCmd
                    self.app.undo_manager.push_done(MovePointCmd(
                        timeline.timeline_num,
                        existing_idx,
                        actions[existing_idx]['at'], old_value,
                        actions[existing_idx]['at'], value
                    ))
                    self.app.funscript_processor._post_mutation_refresh(timeline_num, "Move Point")
                    timeline.invalidate_cache()
                    self.app.logger.info(f"Moved point: {old_value}% -> {value}% at {current_time_ms}ms (Timeline {timeline_num})", extra={'status_message': True})
                    return

            from application.classes.timeline_ops import add_point
            add_point(timeline, current_time_ms, value, snap_time=False)
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

    def _handle_seek_n_frames(self, direction: int):
        """Step `direction * seek_n_frames` frames (default N=5). Uses
        seek_video_with_sync so timelines stay aligned."""
        proc = self.app.processor
        if not proc or proc.total_frames <= 0:
            return
        n = self.app.app_settings.config.navigation.seek_n_frames
        target = max(0, min(proc.total_frames - 1,
                            proc.current_frame_index + direction * n))
        if target == proc.current_frame_index:
            return
        self.app.event_handlers.seek_video_with_sync(target)

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
        self.app.logger.info(f"Funscript 2 {status}", extra={'status_message': True})
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

    def _handle_toggle_script_gauge_shortcut(self):
        """Handle keyboard shortcut for toggling gauge (G)"""
        app_state = self.app.app_state_ui
        app_state.show_script_gauge = not getattr(app_state, 'show_script_gauge', False)
        if self.app.project_manager:
            self.app.project_manager.project_dirty = True
        status = "shown" if app_state.show_script_gauge else "hidden"
        self.app.logger.info(f"Gauge {status}", extra={'status_message': True})
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
        self.app.app_settings.config.ui.show_video_feed = app_state.show_video_feed
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

    def _handle_toggle_timeline_smooth_curve_shortcut(self):
        """Toggle timeline smooth-curve rendering (C)."""
        settings = self.app.app_settings
        new_val = not bool(settings.get("timeline_smooth_curve", True))
        settings.set("timeline_smooth_curve", new_val)
        gui = getattr(self.app, 'gui_instance', None)
        if gui is not None:
            for tl_attr in ("timeline_editor1", "timeline_editor2"):
                tl = getattr(gui, tl_attr, None)
                if tl is not None:
                    tl._show_smooth_curve = new_val
        status = "smoothed" if new_val else "straight"
        self.app.logger.info(f"Timeline curve: {status}", extra={'status_message': True})
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
        """Toggle the mpv-backed fullscreen review window."""
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
        # Modifiers + common nav keys as an interaction proxy: 0.2 us vs
        # 120 us for any(io.keys_down), which iterates 512 pyimgui proxies.
        if not interaction_detected_this_frame:
            if (io.key_ctrl or io.key_alt or io.key_shift or io.key_super
                    or imgui.is_key_down(imgui.KEY_SPACE)
                    or imgui.is_key_down(imgui.KEY_LEFT_ARROW)
                    or imgui.is_key_down(imgui.KEY_RIGHT_ARROW)
                    or imgui.is_key_down(imgui.KEY_UP_ARROW)
                    or imgui.is_key_down(imgui.KEY_DOWN_ARROW)):
                interaction_detected_this_frame = True
        if interaction_detected_this_frame:
            self.app.energy_saver.reset_activity_timer()
