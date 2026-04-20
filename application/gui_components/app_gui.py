import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import numpy as np
import cv2
import time
import threading
import queue
import ctypes
import os
import platform
from typing import List, Dict, Tuple
from collections import deque

from common import paths
from config import constants, element_group_colors
from config.constants_colors import CurrentTheme
from application.classes import ImGuiFileDialog, InteractiveFunscriptTimeline, MainMenu, Simulator3DWindow, ScriptGaugeWindow
from application.classes.batch_state import BatchState
from application.gui_components import ControlPanelUI, VideoDisplayUI, VideoNavigationUI, ChapterListWindow, InfoGraphsUI, GeneratedFileManagerWindow, KeyboardShortcutsDialog, ToolbarUI, ChapterTypeManagerUI
from application.gui_components.bookmark_list_window import BookmarkListWindow
from application.utils import _format_time, ProcessingThreadManager, TaskType, TaskPriority, get_icon_texture_manager
from application.utils.feature_detection import is_feature_available as _is_feature_available
from application.utils.timeline_constants import EXTRA_TIMELINE_RANGE
from application.utils.timeline_modes import TimelineMode
from application.gui_components.gui_preview_manager import PreviewManagerMixin
from application.gui_components.gui_shortcut_handler import ShortcutHandlerMixin
from application.gui_components.gui_dialog_renderer import DialogRendererMixin
from application.gui_components.plugin_pipeline_ui import PluginPipelineUI
from application.gui_components.side_blocks import (
    LeftBottomBlock, RightBottomBlock, render_collapsed_chevron,
)
from application.utils.notifications import NotificationManager
from application.gui_components.first_run_wizard import FirstRunWizard
from config import ui_metrics

_STATUS_STRIP_HEIGHT = ui_metrics.STATUS_STRIP_PX
_HINT_ROTATION_INTERVAL_S = 10.0
_FIXED_PANEL_BASE_WIDTH = 450  # Base width for control panel and info graphs (scaled by font_scale)
_VIDEO_NAV_BAR_HEIGHT = 150


class GUI(DialogRendererMixin, ShortcutHandlerMixin, PreviewManagerMixin):
    def __init__(self, app_logic):
        self.app = app = app_logic
        self.window = None
        self.impl = None
        self.window_width = app.app_settings.config.ui.window_width
        self.window_height = app.app_settings.config.ui.window_height
        self.main_menu_bar_height = 0

        self.constants = constants
        self.colors = element_group_colors.AppGUIColors

        # Feature-detection is runtime-stable (per-session addon state);
        # resolve once here so render_gui stops calling _is_feature_available
        # every frame. _visible_extra_timelines is still recomputed per frame
        # because it reflects live user toggles.
        self._feat_supporter = _is_feature_available("patreon_features")

        self.frame_texture_id = 0
        self.heatmap_texture_id = 0
        self.funscript_preview_texture_id = 0
        self.enhanced_preview_texture_id = 0  # Dedicated texture for enhanced preview tooltips

        # Texture upload bookkeeping. Initialized once here so update_texture's
        # hot path doesn't need a per-frame hasattr guard.
        self._texture_sizes: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
        self._texture_pbos: Dict[int, int] = {}
        self._pbo_upload_enabled = True

        # --- Advanced Threading Architecture ---
        # Decoupled, non-blocking preview/heatmap pipeline with larger queues and background workers
        max_queue = 8
        self.preview_task_queue = queue.Queue(maxsize=max_queue)
        self.preview_results_queue = queue.Queue(maxsize=max_queue)
        self.shutdown_event = threading.Event()
        # Start 2 workers to avoid stalls under load
        self.preview_worker_threads = [
            threading.Thread(target=self._preview_generation_worker, daemon=True, name="PreviewWorker-1"),
            threading.Thread(target=self._preview_generation_worker, daemon=True, name="PreviewWorker-2")
        ]
        for t in self.preview_worker_threads: t.start()
        
        # New ProcessingThreadManager for GPU-intensive operations
        self.processing_thread_manager = ProcessingThreadManager(
            max_worker_threads=2,
            logger=app.logger
        )
        
        # Progress tracking for threaded operations
        self.active_threaded_operations: Dict[str, Dict] = {}
        self.processing_thread_manager.set_global_progress_callback(self._handle_threaded_progress)

        # --- State for incremental texture generation ---
        self.last_submitted_action_count_timeline: int = 0
        self.last_submitted_action_count_heatmap: int = 0

        # Performance monitoring
        self.component_render_times = {}
        self.perf_log_interval = 10  # Log performance every 10 seconds
        self.last_perf_log_time = time.time()
        self.perf_frame_count = 0
        self.perf_accumulated_times = {}

        # Profiler mode: when FUNGEN_PROFILE_FRAMES is set to a positive int,
        # every _time_render() call appends its duration to _profile_samples[name]
        # and after N rendered frames the session writes a JSON summary and
        # exits. Off by default; zero per-frame cost when unset.
        try:
            _pf = int(os.environ.get("FUNGEN_PROFILE_FRAMES", "0") or "0")
        except ValueError:
            _pf = 0
        self._profile_target_frames = _pf
        self._profile_samples: Dict[str, list] = {} if _pf > 0 else {}
        self._profile_enabled = _pf > 0
        
        # Frontend data queue - maintains continuous data flow
        self._frontend_perf_queue = deque(maxlen=2)  # Keep last 2 data points
        self._frontend_perf_queue.append({
            'accumulated_times': {},
            'frame_count': 0,
            'timestamp': time.time()
        })
        
        # Extended monitoring capabilities
        self.video_decode_times = deque(maxlen=100)  # Track video decoding
        self.gpu_memory_usage = 0
        self.last_gpu_check = 0
        self.disk_io_times = deque(maxlen=50)  # Track file operations
        self.network_operation_times = deque(maxlen=30)  # Track network calls

        # Notification system
        self.notification_manager = NotificationManager(app=app)

        # Standard Components (owned by GUI)
        self.file_dialog = ImGuiFileDialog(app_logic_instance=app)
        self.main_menu = MainMenu(app, gui_instance=self)
        self.toolbar_ui = ToolbarUI(app)
        self.simulator_3d_window_ui = Simulator3DWindow(app)
        self.script_gauge_ui = ScriptGaugeWindow(app)
        self.plugin_pipeline_ui = PluginPipelineUI(app)

        self.timeline_editor1 = InteractiveFunscriptTimeline(app_instance=app, timeline_num=1)
        self.timeline_editor2 = InteractiveFunscriptTimeline(app_instance=app, timeline_num=2)

        # Extra timelines (supporter feature, created lazily)
        self._extra_timeline_editors: Dict[int, InteractiveFunscriptTimeline] = {}

        # Subtitle timeline (optional addon)
        try:
            from subtitle_translation.interactive_timeline import InteractiveSubtitleTimeline
            self._subtitle_timeline = InteractiveSubtitleTimeline(app)
        except ImportError:
            pass

        # Modularized UI Panel Components
        self.control_panel_ui = ControlPanelUI(app)
        self.video_display_ui = VideoDisplayUI(app, self)  # Pass self for texture updates
        self.video_navigation_ui = VideoNavigationUI(app, self)  # Pass self for texture methods
        self.info_graphs_ui = InfoGraphsUI(app)
        self.chapter_list_window_ui = ChapterListWindow(app, nav_ui=self.video_navigation_ui)
        self.chapter_type_manager_ui = ChapterTypeManagerUI(app)
        self.bookmark_list_window_ui = BookmarkListWindow(app, gui=self)
        self.left_bottom_block = LeftBottomBlock(app, gui=self)
        self.right_bottom_block = RightBottomBlock(app, gui=self)
        self.generated_file_manager_ui = GeneratedFileManagerWindow(app)
        self.keyboard_shortcuts_dialog = KeyboardShortcutsDialog(app)

        # Proxy (edit-time transcode) controller. Owns suggestion + progress
        # modals; exposed on app.proxy_controller so the file manager hook
        # and the Tools menu entry can drive it.
        from application.gui_components.proxy_dialog import ProxyController
        app.proxy_controller = ProxyController(app)

        # First-run wizard (full-window overlay, replaces old popup)
        self._first_run_wizard = (
            FirstRunWizard(app)
            if app.app_settings.is_first_run and not app.is_cli_mode
            else None
        )
        # Flag for menu-triggered wizard re-launch
        self._show_setup_wizard = False

        self.control_panel_ui.timeline_editor1 = self.timeline_editor1
        self.control_panel_ui.timeline_editor2 = self.timeline_editor2

        self.last_preview_update_time_timeline = 0.0
        self.last_preview_update_time_heatmap = 0.0
        self.preview_update_interval_seconds = constants.UI_PREVIEW_UPDATE_INTERVAL_S

        self.last_mouse_pos_for_energy_saver = (0, 0)
        self.app.energy_saver.reset_activity_timer()
        
        current_time = time.time()
        self.arrow_key_state = {
            'last_seek_time': current_time,
            'seek_interval': 0.033,
            'initial_press_time': current_time,
            'continuous_delay': 0.2,
            'last_direction': 0,
            'playback_active': False,
            'continuous_start_time': 0,
            'arrow_triggered_playback': False,
            'saved_speed_mode': None,
            'nav_phase': 'idle',
        }
        # Per-shortcut time-based repeat state (see gui_shortcut_handler.py).
        # Lets held shortcuts repeat on our own timer instead of depending on
        # OS KEY_REPEAT events, which some BLE / remapped devices never emit.
        self._shortcut_repeat_state: Dict[str, Dict[str, float]] = {}
        # GLFW char-input fallback: codepoints -> GLFW keycodes "virtually
        # pressed" this frame via the char callback. Consumed (and cleared)
        # by the shortcut dispatcher. See _install_char_input_fallback().
        self._virtual_pressed_keys: set = set()
        self._prev_char_callback = None
        self.arrow_nav_reading_fps = 0.0
        self._reading_fps_frames = deque()
        self._reading_fps_display = 0.0
        self._reading_fps_last_update = 0.0
        self._tracker_fps_display = 0.0
        self._tracker_fps_last_update = 0.0
        # Rolling samples of raw tracker fps with timestamps for a 3s mean.
        self._tracker_fps_samples = deque()

        self.batch_state = BatchState()

        # Go to Frame popup
        self._go_to_frame_open = False
        self._go_to_frame_input = ""
        self._go_to_frame_focus = False

        # TODO: Move this to a separate class/error management module
        self.error_popup_active = False
        self.error_popup_title = ""
        self.error_popup_message = ""
        self.error_popup_action_label = None
        self.error_popup_action_callback = None

        # Shortcut hint rotation state (status strip)
        self._hint_rotate_index = 0
        self._hint_last_rotate = 0.0
        self._hint_pool_cache = []
        self._hint_last_context = None

    def _handle_threaded_progress(self, task_id: str, progress: float, message: str):
        """Handle progress updates from threaded operations."""
        if task_id in self.active_threaded_operations:
            self.active_threaded_operations[task_id].update({
                'progress': progress,
                'message': message,
                'last_update': time.time()
            })
            
            # Update UI status if this is a high-priority operation
            operation_info = self.active_threaded_operations[task_id]
            if operation_info.get('show_in_status', False):
                status_msg = f"{operation_info.get('name', 'Processing')}: {message} ({progress*100:.1f}%)"
                self.app.set_status_message(status_msg, duration=1.0)

    def _get_or_create_timeline_editor(self, timeline_num: int) -> InteractiveFunscriptTimeline:
        """Get or lazily create a timeline editor for timeline 3+."""
        if timeline_num not in self._extra_timeline_editors:
            editor = InteractiveFunscriptTimeline(app_instance=self.app, timeline_num=timeline_num)
            self._extra_timeline_editors[timeline_num] = editor
        return self._extra_timeline_editors[timeline_num]

    def _dump_profile_snapshot(self):
        """Write a JSON summary of _time_render samples to disk, then log it.

        Output path: bench_results/gui_profile_<timestamp>.json. Each entry
        has n / mean_ms / p50_ms / p95_ms / p99_ms / max_ms / total_ms so
        subsequent diff-against-baseline runs are possible.
        """
        import json, statistics
        from pathlib import Path

        out_dir = Path("bench_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"gui_profile_{stamp}.json"

        payload = {
            "frames": self.perf_frame_count,
            "components": {},
        }
        for name, samples in self._profile_samples.items():
            if not samples:
                continue
            s = sorted(samples)
            n = len(s)
            def _pct(p):
                k = max(0, min(n - 1, int(round((n - 1) * p))))
                return s[k]
            payload["components"][name] = {
                "n": n,
                "mean_ms": statistics.fmean(samples),
                "p50_ms": _pct(0.50),
                "p95_ms": _pct(0.95),
                "p99_ms": _pct(0.99),
                "max_ms": s[-1],
                "total_ms": sum(samples),
            }

        try:
            path.write_text(json.dumps(payload, indent=2))
            self.app.logger.debug(f"GUI profile dumped to {path}")
        except Exception as e:
            self.app.logger.error(f"failed to write profile JSON: {e}")

    def _time_render(self, component_name: str, render_func, *args, **kwargs):
        """Helper to time a render function and store its duration."""
        start_time = time.perf_counter()
        render_func(*args, **kwargs)
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        self.component_render_times[component_name] = duration_ms

        # Accumulate for averaging
        if component_name not in self.perf_accumulated_times:
            self.perf_accumulated_times[component_name] = 0.0
        self.perf_accumulated_times[component_name] += duration_ms

        # Profile-mode sample record (no-op unless FUNGEN_PROFILE_FRAMES set).
        if self._profile_enabled:
            lst = self._profile_samples.get(component_name)
            if lst is None:
                lst = []
                self._profile_samples[component_name] = lst
            lst.append(duration_ms)

    def get_performance_summary(self):
        """Get comprehensive performance analysis."""
        if not self.component_render_times:
            return {"status": "No performance data available"}
        
        current_total = sum(self.component_render_times.values())
        sorted_components = sorted(
            self.component_render_times.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Categorize performance
        if current_total < 8.33:  # 120 FPS
            status = "EXCELLENT"
            color = (0.0, 1.0, 0.0, 1.0)
        elif current_total < 16.67:  # 60 FPS
            status = "GOOD"
            color = (0.4, 0.8, 0.4, 1.0)
        elif current_total < 33.33:  # 30 FPS
            status = "OK"
            color = (1.0, 0.8, 0.2, 1.0)
        else:
            status = "SLOW"
            color = (1.0, 0.2, 0.2, 1.0)
        
        return {
            "status": status,
            "color": color,
            "total_ms": current_total,
            "components": sorted_components,
            "frame_budget_60fps": 16.67,
            "frame_budget_used_percent": (current_total / 16.67) * 100
        }

    def track_video_decode_time(self, decode_time_ms):
        """Track video decoding performance."""
        self.video_decode_times.append(decode_time_ms)
        self.component_render_times["VideoDecoding"] = decode_time_ms

    def track_frame_seek_time(self, seek_time_ms, path: str = "arrow"):
        """Track frame seek/decode time for arrow nav and scrubbing.

        Args:
            seek_time_ms: Time in milliseconds for the seek+decode operation.
            path: "arrow" for arrow-key navigation, "scrub" for timeline scrub.
        """
        key = "ArrowNavDecode" if path == "arrow" else "ScrubDecode"
        self.component_render_times[key] = seek_time_ms

    def track_disk_io_time(self, operation_name, io_time_ms):
        """Track disk I/O operations."""
        self.disk_io_times.append(io_time_ms)
        self.component_render_times[f"DiskIO_{operation_name}"] = io_time_ms

    def track_network_time(self, operation_name, network_time_ms):
        """Track network operations."""
        self.network_operation_times.append(network_time_ms)
        self.component_render_times[f"Network_{operation_name}"] = network_time_ms

    def update_gpu_memory_usage(self):
        """Update GPU memory usage (called periodically)."""
        # GPUtil only reports NVIDIA GPUs via nvidia-smi. Skip on macOS
        # (no NVIDIA) and cache an unavailable flag after the first miss so
        # non-NVIDIA Windows/Linux boxes don't re-import+probe every 2s.
        if getattr(self, '_gpu_util_unavailable', False):
            self.gpu_memory_usage = 0
            return
        if platform.system() == "Darwin":
            self._gpu_util_unavailable = True
            self.gpu_memory_usage = 0
            return
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self.gpu_memory_usage = gpu.memoryUsed / gpu.memoryTotal * 100
                self.component_render_times["GPU_MemoryUsage"] = self.gpu_memory_usage
            else:
                self._gpu_util_unavailable = True
                self.gpu_memory_usage = 0
        except (ImportError, Exception):
            self._gpu_util_unavailable = True
            self.gpu_memory_usage = 0

    def _log_performance(self):
        """Log performance statistics and reset accumulators."""
        if not self.perf_accumulated_times or self.perf_frame_count == 0:
            return

        # Calculate averages
        total_time = sum(self.perf_accumulated_times.values())
        avg_time = total_time / self.perf_frame_count if self.perf_frame_count > 0 else 0

        # Build detailed log message
        log_parts = [f"Performance: {avg_time:.2f}ms avg ({self.perf_frame_count} frames)"]
        
        # Sort components by time (most expensive first)
        sorted_components = sorted(
            self.perf_accumulated_times.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Add top 3 most expensive components
        for i, (component, total_time) in enumerate(sorted_components[:3]):
            avg_component_time = total_time / self.perf_frame_count if self.perf_frame_count > 0 else 0
            log_parts.append(f"{component}: {avg_component_time:.2f}ms")
        
        log_message = " | ".join(log_parts)
        self.app.logger.debug(log_message)  # Use debug to avoid spamming info logs

        # Store the current stats in frontend queue before clearing backend
        self._frontend_perf_queue.append({
            'accumulated_times': self.perf_accumulated_times.copy(),
            'frame_count': self.perf_frame_count,
            'timestamp': time.time()
        })
        
        # Clear the backend accumulators for the next interval
        self.perf_accumulated_times.clear()
        self.perf_frame_count = 0
        
        self.last_perf_log_time = time.time()

    def init_glfw(self) -> bool:
        constants = self.constants
        if not glfw.init():
            self.app.logger.error("Could not initialize GLFW")
            return False
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        self.window = glfw.create_window(
            self.window_width, self.window_height, constants.APP_WINDOW_TITLE, None, None
        )
        if not self.window:
            glfw.terminate()
            self.app.logger.error("Could not create GLFW window")
            return False

        # Reinitialize keyboard layout map now that GLFW is ready
        # (ShortcutManager was created before glfw.init, so layout detection was wrong)
        if self.app.shortcut_manager:
            self.app.shortcut_manager.reinitialize_key_map()

        # Set window icon (macOS doesn't support window icons in GLFW, skip on macOS)
        try:
            import platform
            if platform.system() != "Darwin":  # Skip on macOS
                icon_path = str(paths.asset('logo.png'))
                if os.path.exists(icon_path):
                    # Load icon with cv2 (already imported)
                    icon_img = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                    if icon_img is not None:
                        # Convert BGR(A) to RGB(A) for GLFW
                        if len(icon_img.shape) == 3 and icon_img.shape[2] == 4:  # Has alpha channel
                            icon_rgb = cv2.cvtColor(icon_img, cv2.COLOR_BGRA2RGBA)
                        else:
                            icon_rgb = cv2.cvtColor(icon_img, cv2.COLOR_BGR2RGB)
                            # Add alpha channel (fully opaque)
                            alpha = np.full((icon_rgb.shape[0], icon_rgb.shape[1], 1), 255, dtype=np.uint8)
                            icon_rgb = np.concatenate([icon_rgb, alpha], axis=2)

                        height, width = icon_rgb.shape[:2]
                        pixels = icon_rgb.tobytes()

                        # pyGLFW expects list of GLFWimage objects
                        # Create image tuple: (width, height, pixels)
                        from glfw import _GLFWimage as GLFWimage
                        icon_image = GLFWimage(width, height, pixels)

                        glfw.set_window_icon(self.window, 1, [icon_image])
                        self.app.logger.debug(f"Window icon set from {icon_path}")
                    else:
                        self.app.logger.warning(f"Failed to load icon image: {icon_path}")
        except Exception as e:
            self.app.logger.debug(f"Window icon not set: {e}")  # Debug level since it's non-critical

        glfw.make_context_current(self.window)
        glfw.set_drop_callback(self.window, self.handle_drop)
        glfw.set_window_close_callback(self.window, self.handle_window_close)

        imgui.create_context()

        from config.constants_colors import get_active_theme
        from config.theme_manager import apply_imgui_theme
        apply_imgui_theme(get_active_theme())

        io = imgui.get_io()
        self._cjk_font_loaded = False
        self._icon_font_loaded = False
        io.fonts.add_font_default()

        # Merge Symbols Nerd Font so icon glyphs (\ue000-\uf8ff, \uf0000+) render.
        # Single TTF shipped under assets/fonts/, no runtime install needed.
        try:
            icon_font_path = str(paths.ICON_FONT_PATH)
            if os.path.exists(icon_font_path):
                icon_cfg = imgui.FontConfig(merge_mode=True)
                # Private-use ranges covering Font Awesome, Material, Octicons,
                # Weather, Devicons, Codicons, Powerline, all inside the
                # Symbols Nerd Font.
                icon_ranges = imgui.GlyphRanges([
                    0x23fb, 0x23fe,   # IEC power symbols
                    0x2665, 0x2665,   # heart
                    0x26a1, 0x26a1,   # flash
                    0x2b58, 0x2b58,   # circle
                    0xe000, 0xe00a,   # Pomicons
                    0xe0a0, 0xe0a2,   # Powerline
                    0xe0a3, 0xe0a3,   # Powerline Extra
                    0xe0b0, 0xe0b3,   # Powerline
                    0xe0b4, 0xe0c8,   # Powerline Extra
                    0xe0ca, 0xe0ca,
                    0xe0cc, 0xe0d4,
                    0xe200, 0xe2a9,   # Font Awesome Extension
                    0xe300, 0xe3e3,   # Weather
                    0xe5fa, 0xe6b5,   # Seti UI + Custom
                    0xe700, 0xe8ef,   # Devicons
                    0xea60, 0xec1e,   # Codicons
                    0xed00, 0xf2ff,   # Font Awesome (v6 block 1)
                    0xf000, 0xf2e0,   # Font Awesome legacy
                    0xf300, 0xf372,   # Font Logos
                    0xf400, 0xf533,   # Octicons
                    0xf0001, 0xf1af0, # Material Design
                    0, 0,
                ])
                io.fonts.add_font_from_file_ttf(
                    icon_font_path, 13.0,
                    font_config=icon_cfg,
                    glyph_ranges=icon_ranges,
                )
                self._icon_font_loaded = True
                self.app.logger.info("Icon font merged: icons.ttf")
            else:
                self.app.logger.debug(f"Icon font not found at {icon_font_path}")
        except Exception as e:
            self.app.logger.debug(f"Icon font load failed: {e}")

        if _is_feature_available("subtitle_translation"):
            try:
                from subtitle_translation.model_downloader import get_cjk_font_path, ensure_cjk_font
                cjk_font_path = get_cjk_font_path() or ensure_cjk_font()
            except Exception:
                cjk_font_path = ""
            if cjk_font_path:
                try:
                    merge_cfg = imgui.FontConfig(merge_mode=True)
                    io.fonts.add_font_from_file_ttf(
                        cjk_font_path, 14.0,
                        font_config=merge_cfg,
                        glyph_ranges=io.fonts.get_glyph_ranges_japanese(),
                    )
                    self._cjk_font_loaded = True
                    self.app.logger.info(f"CJK font loaded: {os.path.basename(cjk_font_path)}")
                except Exception as e:
                    self.app.logger.debug(f"CJK font load failed: {e}")

        self.impl = GlfwRenderer(self.window)
        self._install_char_input_fallback()
        style = imgui.get_style()
        style.window_rounding = 6.0
        style.frame_rounding = 4.0
        style.child_rounding = 6.0
        style.popup_rounding = 6.0
        style.tab_rounding = 4.0
        style.scrollbar_rounding = 6.0
        style.grab_rounding = 4.0

        self.frame_texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.frame_texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # mpv display FBO + texture; shared by GL and SW backends.
        self.mpv_display = None
        self.mpv_display_texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.mpv_display_texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                           gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_SWIZZLE_A, gl.GL_ONE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, 1920, 1080, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self.mpv_display_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.mpv_display_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, self.mpv_display_texture_id, 0)
        _fbo_status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if _fbo_status != gl.GL_FRAMEBUFFER_COMPLETE:
            self.app.logger.error(
                f"mpv display FBO incomplete: status=0x{_fbo_status:x}")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        self.mpv_display_w, self.mpv_display_h = 1920, 1080

        # VR dewarp output FBO + texture (shader_dewarp mode only).
        self.vr_dewarp_texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.vr_dewarp_texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_SWIZZLE_A, gl.GL_ONE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, 1920, 1080, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self.vr_dewarp_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.vr_dewarp_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, self.vr_dewarp_texture_id, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        self.vr_dewarp_w, self.vr_dewarp_h = 1920, 1080

        self._init_mpv_display()
        self._init_vr_dewarp_shader()

        # Tracker debug overlay texture (community trackers can set tracker.debug_frame)
        self._debug_frame_texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._debug_frame_texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self._show_debug_frame_window = False

        # Initialize heatmap with a dummy texture
        self.heatmap_texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.heatmap_texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        dummy_pixel = np.array([0, 0, 0, 0], dtype=np.uint8).tobytes()
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 1, 1, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, dummy_pixel)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Initialize funscript preview with a dummy texture
        self.funscript_preview_texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.funscript_preview_texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        dummy_pixel_fs_preview = np.array([0, 0, 0, 0], dtype=np.uint8).tobytes()
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 1, 1, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, dummy_pixel_fs_preview)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Initialize dedicated enhanced preview texture (isolated from main video display)
        self.enhanced_preview_texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.enhanced_preview_texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return True

    def handle_window_close(self, window):
        """Handle window close event (X button clicked)."""
        self.app.logger.info("Window close requested via window controls")
        self.app.shutdown_app()

    def handle_drop(self, window, paths):
        if not paths:
            return

        constants = self.constants

        # Separate files by type
        project_files = [p for p in paths if p.lower().endswith(constants.PROJECT_FILE_EXTENSION)]
        funscript_files = [p for p in paths if p.lower().endswith('.funscript')]
        other_files = [p for p in paths if p not in project_files and p not in funscript_files]

        # 1. Handle Project Files (highest priority)
        if project_files:
            project_to_load = project_files[0]
            self.app.logger.info(f"Project file dropped. Loading: {os.path.basename(project_to_load)}")
            self.app.project_manager.load_project(project_to_load)
            # Typically, loading a project handles everything, so we can stop.
            return

        # 2. Handle Video/Other Files via FileManager
        if other_files:
            self.app.logger.info(f"Video/other files dropped. Passing to FileManager: {len(other_files)} files")
            # This will handle loading the video and preparing the processor
            self.app.file_manager.handle_drop_event(other_files)

        # 3. Handle Funscript Files
        if funscript_files:
            self.app.logger.info(f"Funscript files dropped: {len(funscript_files)} files")
            # If timeline 1 is empty or has no loaded script, load the first funscript there.

            if not self.app.funscript_processor.get_actions('primary'):
                self.app.logger.info(f"Loading '{os.path.basename(funscript_files[0])}' into Funscript 1.")
                self.app.file_manager.load_funscript_to_timeline(funscript_files[0], timeline_num=1)

                if len(funscript_files) > 1:
                    self.app.logger.info(f"Loading '{os.path.basename(funscript_files[1])}' into Funscript 2.")
                    self.app.file_manager.load_funscript_to_timeline(funscript_files[1], timeline_num=2)
                    self.app.app_state_ui.show_funscript_interactive_timeline2 = True
            else:
                self.app.logger.info(f"Funscript 1 has data. Loading '{os.path.basename(funscript_files[0])}' into Funscript 2.")
                self.app.file_manager.load_funscript_to_timeline(funscript_files[0], timeline_num=2)
                self.app.app_state_ui.show_funscript_interactive_timeline2 = True

            # Mark previews as dirty to force a redraw
            self.app.app_state_ui.funscript_preview_dirty = True
            self.app.app_state_ui.heatmap_dirty = True


    def _init_mpv_display(self):
        """Construct the session-wide mpv display (GL or SW backend)."""
        try:
            from video.mpv_loader import mpv_available, mpv_load_error
            if not mpv_available:
                self.app.logger.info(
                    f"libmpv not loaded: {mpv_load_error or 'unavailable'}; "
                    "video display will use the CPU upload path.")
                return

            import ctypes as _ctypes

            def _get_proc_address(_ctx, name):
                nm = name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else name
                addr = glfw.get_proc_address(nm)
                if not addr:
                    return 0
                return _ctypes.cast(addr, _ctypes.c_void_p).value or 0

            # 'auto' probes all hwdec options and falls back to SW only if
            # none work. 'auto-safe' was too conservative and left some 8K
            # h264 files on SW.
            backend = self.app.app_settings.config.mpv.render_backend
            _default_hwdec = "auto"
            _sys = __import__("platform").system()
            hwdec_override = self.app.app_settings.config.mpv.hwdec_override or _default_hwdec
            disp = None

            if backend == "gl":
                try:
                    from video.mpv_display_gl import MpvDisplayGL
                    refresh_hz = 60.0
                    try:
                        monitor = glfw.get_primary_monitor()
                        if monitor:
                            vmode = glfw.get_video_mode(monitor)
                            if vmode and getattr(vmode, 'refresh_rate', 0):
                                refresh_hz = float(vmode.refresh_rate)
                    except Exception:
                        pass
                    disp = MpvDisplayGL(
                        video_path="",
                        hwdec=hwdec_override,
                        display_fps=refresh_hz,
                        logger=self.app.logger,
                    )
                    self.app.logger.debug(
                        f"MpvDisplayGL backend selected (hwdec={hwdec_override}, "
                        f"display_fps={refresh_hz:.1f})")
                    if not disp.open(_get_proc_address):
                        self.app.logger.warning(
                            "MpvDisplayGL.open() failed; falling back to SW backend.")
                        disp = None
                except Exception as e:
                    self.app.logger.warning(
                        f"MpvDisplayGL init failed: {e}; falling back to SW backend.")
                    disp = None

            if disp is None:
                from video.mpv_display import MpvDisplay
                disp = MpvDisplay(video_path="", hwdec=hwdec_override, logger=self.app.logger)
                self.app.logger.info(f"MpvDisplay (SW) hwdec={hwdec_override}")
                if not disp.open(_get_proc_address):
                    self.app.logger.warning(
                        "MpvDisplay.open() failed; falling back to CPU upload path.")
                    return

            self.mpv_display = disp
            try:
                from video.thumbnail_mpv import ThumbnailMpv, ThumbnailMpvQueue
                tmb = ThumbnailMpv(logger=self.app.logger)
                if tmb.open(_get_proc_address):
                    self.thumbnail_mpv = tmb
                    self.thumbnail_mpv_queue = ThumbnailMpvQueue(tmb)
                else:
                    self.thumbnail_mpv = None
                    self.thumbnail_mpv_queue = None
            except Exception as e:
                self.app.logger.debug(f"ThumbnailMpv init failed: {e}")
                self.thumbnail_mpv = None
                self.thumbnail_mpv_queue = None
            def _mpv_pos_cb(idx):
                proc = getattr(self.app, 'processor', None)
                if proc is not None:
                    try:
                        proc.on_mpv_position(idx)
                    except Exception:
                        pass
            try:
                disp.register_position_callback(_mpv_pos_cb)
            except Exception as e:
                self.app.logger.debug(f"mpv position callback register failed: {e}")
            self.app.logger.debug(
                f"{type(disp).__name__} ready (libmpv render API)")
            # Project-restore catch-up: if open_video already ran (project
            # loaded at app start before gui_instance existed), the prior
            # _load_into_mpv_display() call hit a None disp and bailed.
            # Now that disp exists, fire it so mpv actually loads the file.
            proc = getattr(self.app, 'processor', None)
            if proc is not None and getattr(proc, 'video_path', None):
                self.app.logger.debug(
                    "Project-restore: loading current video into MpvDisplay")
                try:
                    proc._load_into_mpv_display()
                except Exception as e:
                    self.app.logger.warning(
                        f"project-restore mpv load failed: {e}")
        except Exception as e:
            self.app.logger.warning(f"MpvDisplay init error: {e}; CPU upload fallback in use.")
            self.mpv_display = None

    def _init_vr_dewarp_shader(self) -> None:
        """Compile the VR dewarp GLSL program once, after the GL context
        exists. Failure leaves vr_dewarp_shader=None; the painter falls
        back to whatever vr_display_mode it can without the shader."""
        try:
            from video.vr_dewarp_shader import VRDewarpShader
            shader = VRDewarpShader(logger=self.app.logger)
            if shader.compile():
                self.vr_dewarp_shader = shader
            else:
                self.vr_dewarp_shader = None
                # If the user had selected shader_dewarp but compile failed,
                # roll them back to passthrough so they see something.
                try:
                    if self.app.app_settings.config.vr_display.mode == "shader_dewarp":
                        self.app.app_settings.config.vr_display.mode = "passthrough"
                        self.app.logger.warning(
                            "VR dewarp shader unavailable; reverting "
                            "vr_display_mode to passthrough.")
                except Exception:
                    pass
        except Exception as e:
            self.app.logger.warning(f"VR dewarp shader init error: {e}")
            self.vr_dewarp_shader = None

    def resize_vr_dewarp_target(self, w: int, h: int) -> None:
        """Reallocate the VR-dewarp FBO + texture (fresh IDs) if the size changed."""
        w = max(64, int(w))
        h = max(64, int(h))
        if w == self.vr_dewarp_w and h == self.vr_dewarp_h:
            return
        try:
            gl.glDeleteFramebuffers(1, [self.vr_dewarp_fbo])
        except Exception:
            pass
        try:
            gl.glDeleteTextures(1, [self.vr_dewarp_texture_id])
        except Exception:
            pass
        new_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, new_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                           gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_SWIZZLE_A, gl.GL_ONE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        new_fbo = gl.glGenFramebuffers(1)
        _prev_fbo = gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, new_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, new_tex, 0)
        _status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if _status != gl.GL_FRAMEBUFFER_COMPLETE:
            self.app.logger.error(
                f"VR dewarp FBO incomplete after fresh alloc at {w}x{h}: "
                f"status=0x{_status:x}")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, int(_prev_fbo))
        self.vr_dewarp_texture_id = int(new_tex)
        self.vr_dewarp_fbo = int(new_fbo)
        self.vr_dewarp_w, self.vr_dewarp_h = w, h

    def reallocate_mpv_display_storage(self, w: int, h: int) -> None:
        """Per-frame texture realloc, SW backend only (dfaker demo workaround)."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.mpv_display_fbo)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.mpv_display_texture_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, int(w), int(h), 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self.mpv_display_w, self.mpv_display_h = int(w), int(h)

    def resize_mpv_display_target(self, w: int, h: int) -> None:
        """Reallocate the mpv display FBO + texture (fresh IDs) if the size changed."""
        w = max(64, int(w))
        h = max(64, int(h))
        if w == self.mpv_display_w and h == self.mpv_display_h:
            return
        try:
            gl.glDeleteFramebuffers(1, [self.mpv_display_fbo])
        except Exception:
            pass
        try:
            gl.glDeleteTextures(1, [self.mpv_display_texture_id])
        except Exception:
            pass
        # Fresh texture.
        new_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, new_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                           gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_SWIZZLE_A, gl.GL_ONE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        new_fbo = gl.glGenFramebuffers(1)
        _prev_fbo = gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, new_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, new_tex, 0)
        _status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if _status != gl.GL_FRAMEBUFFER_COMPLETE:
            self.app.logger.error(
                f"mpv display FBO incomplete after fresh alloc at {w}x{h}: "
                f"status=0x{_status:x}")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, int(_prev_fbo))
        self.mpv_display_texture_id = int(new_tex)
        self.mpv_display_fbo = int(new_fbo)
        self.mpv_display_w, self.mpv_display_h = w, h
        self.app.logger.debug(
            f"mpv display FBO re-created fresh at {w}x{h} "
            f"(tex={self.mpv_display_texture_id}, fbo={self.mpv_display_fbo})")

    def update_texture(self, texture_id: int, image: np.ndarray):
        if image is None or image.size == 0: return
        if not texture_id:
            return
        h, w = image.shape[:2]
        if w == 0 or h == 0: return

        # Skip the per-frame gl.glIsTexture(texture_id) driver round-trip:
        # texture_ids come from our own glGenTextures bookkeeping and are
        # zeroed on cleanup, so truthiness-check plus truthiness on the
        # arg above is sufficient. glIsTexture was a ~0.05 ms pipeline-sync
        # call on some drivers; skipping it wins ~3 ms/s at 60 fps.

        # Upload BGR frames natively via GL_BGR instead of a CPU-side
        # cv2.cvtColor(BGR->RGB) that allocates a second copy of the whole
        # frame. GL_BGR is core since OpenGL 1.2.
        if len(image.shape) == 2:
            internal_fmt = gl.GL_RED; fmt = gl.GL_RED; payload = image
        elif image.shape[2] == 3:
            internal_fmt = gl.GL_RGB; fmt = gl.GL_BGR; payload = image
        else:
            internal_fmt = gl.GL_RGBA; fmt = gl.GL_RGBA; payload = image

        last_size = self._texture_sizes.get(texture_id)
        size_changed = (last_size is None) or (last_size != (w, h, internal_fmt))

        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

        # Streaming texture upload via PBO: glBufferData orphans the backing
        # store so the driver can start the DMA copy without waiting for the
        # prior frame's upload to finish. Fall back to direct glTexSubImage2D
        # on any error so a PBO quirk can't wedge rendering.
        used_pbo = False
        if self._pbo_upload_enabled and not size_changed and payload.flags['C_CONTIGUOUS']:
            try:
                nbytes = int(payload.nbytes)
                pbo = self._texture_pbos.get(texture_id)
                if pbo is None:
                    pbo = int(gl.glGenBuffers(1))
                    self._texture_pbos[texture_id] = pbo
                gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
                # Pass data directly: implicit orphan via STREAM_DRAW hint.
                gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, nbytes, payload, gl.GL_STREAM_DRAW)
                gl.glTexSubImage2D(
                    gl.GL_TEXTURE_2D, 0, 0, 0, w, h, fmt, gl.GL_UNSIGNED_BYTE,
                    ctypes.c_void_p(0))
                gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
                used_pbo = True
            except Exception as e:
                self.app.logger.warning(
                    f"PBO texture upload failed, falling back to direct: {e}")
                self._pbo_upload_enabled = False
                try:
                    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
                except Exception:
                    pass

        if not used_pbo:
            if not size_changed:
                gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h, fmt, gl.GL_UNSIGNED_BYTE, payload)
            else:
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_fmt, w, h, 0, fmt, gl.GL_UNSIGNED_BYTE, payload)
                self._texture_sizes[texture_id] = (w, h, internal_fmt)
        elif size_changed:
            self._texture_sizes[texture_id] = (w, h, internal_fmt)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _get_active_preview_axis(self) -> str:
        """Return the axis name for the currently focused timeline (for heatmap/preview)."""
        tl_num = getattr(self.app.app_state_ui, 'active_timeline_num', 1)
        if tl_num == 1:
            return 'primary'
        elif tl_num == 2:
            return 'secondary'
        else:
            fs_proc = self.app.funscript_processor
            fs, axis = fs_proc._get_target_funscript_object_and_axis(tl_num)
            return axis or 'primary'

    # --- This function now submits a task to the worker thread ---
    def render_funscript_timeline_preview(self, total_duration_s: float, graph_height: int):
        app_state = self.app.app_state_ui
        colors = self.colors
        style = imgui.get_style()

        current_bar_width_float = imgui.get_content_region_available()[0]
        current_bar_width_int = int(round(current_bar_width_float))

        if current_bar_width_int <= 0 or graph_height <= 0 or not self.funscript_preview_texture_id:
            imgui.dummy(current_bar_width_float if current_bar_width_float > 0 else 1, graph_height + 5)
            return

        preview_axis = self._get_active_preview_axis()
        current_action_count = len(self.app.funscript_processor.get_actions(preview_axis))
        is_live_tracking = self.app.processor and self.app.processor.tracker and self.app.processor.tracker.tracking_active

        # Determine if a redraw is needed (also dirty when active timeline changes)
        axis_changed = getattr(self, '_last_preview_axis', 'primary') != preview_axis
        self._last_preview_axis = preview_axis
        full_redraw_needed = (app_state.funscript_preview_dirty or axis_changed
            or current_bar_width_int != app_state.last_funscript_preview_bar_width
            or abs(total_duration_s - app_state.last_funscript_preview_duration_s) > 0.01)

        incremental_update_needed = current_action_count != self.last_submitted_action_count_timeline

        # For this async model, we always do a full redraw. Incremental drawing is complex with threading.
        # The performance gain from async outweighs the loss of incremental drawing.
        needs_regen = (full_redraw_needed
            or (incremental_update_needed
            and (not is_live_tracking
            or (time.time() - self.last_preview_update_time_timeline >= self.preview_update_interval_seconds))))

        # Non-blocking submit: try_put; if queue full, skip this frame without blocking UI
        if needs_regen:
            actions_copy = self.app.funscript_processor.get_actions(preview_axis).copy()
            task = {
                'type': 'timeline',
                'target_width': current_bar_width_int,
                'target_height': graph_height,
                'total_duration_s': total_duration_s,
                'actions': actions_copy
            }
            try:
                self.preview_task_queue.put_nowait(task)
            except queue.Full:
                pass

            # Update state after submission
            app_state.funscript_preview_dirty = False
            app_state.last_funscript_preview_bar_width = current_bar_width_int
            app_state.last_funscript_preview_duration_s = total_duration_s
            self.last_submitted_action_count_timeline = current_action_count
            if is_live_tracking and incremental_update_needed:
                self.last_preview_update_time_timeline = time.time()

        # --- Rendering Logic (uses the existing texture until a new one is ready) ---
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 20)
        canvas_p1_x = imgui.get_cursor_screen_pos()[0]
        canvas_p1_y_offset = imgui.get_cursor_screen_pos()[1]

        imgui.image(self.funscript_preview_texture_id, current_bar_width_float, graph_height, uv0=(0, 0), uv1=(1, 1))

        # --- Add seeking capability to the funscript preview bar ---
        if imgui.is_item_hovered():
            mouse_x = imgui.get_mouse_pos()[0] - canvas_p1_x
            normalized_pos = np.clip(mouse_x / current_bar_width_float, 0.0, 1.0)
            if self.app.processor and self.app.processor.video_info:
                total_frames = self.app.processor.video_info.get('total_frames', 0)
                if total_frames > 0:
                    if (imgui.is_mouse_dragging(0) or imgui.is_mouse_down(0)):
                        # Use time-based calculation consistent with timeline and funscript timing
                        click_time_s = normalized_pos * total_duration_s
                        fps = self.app.processor.fps if self.app.processor and self.app.processor.fps > 0 else 30.0
                        seek_frame = int(round(click_time_s * fps))
                        seek_frame = max(0, min(seek_frame, total_frames - 1))  # Clamp to valid range
                        
                        self.app.event_handlers.handle_seek_bar_drag(seek_frame)
                    else:
                        # Show enhanced tooltip with zoom preview and video frame
                        total_duration = total_duration_s  # Use the parameter passed to this function
                        if total_duration > 0:
                            hover_time_s = normalized_pos * total_duration
                            # Use consistent time-based frame calculation for hover too
                            fps = self.app.processor.fps if self.app.processor and self.app.processor.fps > 0 else 30.0
                            hover_frame = int(round(hover_time_s * fps))
                            hover_frame = max(0, min(hover_frame, total_frames - 1))
                            
                            # Initialize hover tracking attributes if needed
                            if not hasattr(self, '_preview_hover_start_time'):
                                self._preview_hover_start_time = None
                                self._preview_hover_pos = None
                                self._preview_cached_tooltip_data = None
                                self._preview_cached_pos = None
                            
                            # Tolerance trades cache stability for frame precision.
                            # 0.0001 (1 px on a 10k-wide bar) was over-strict:
                            # every sub-pixel jitter invalidated the cache and
                            # queued a fresh decode. 0.002 is ~2-3 px on a
                            # typical bar width and still within a frame for
                            # normal-length videos.
                            position_tolerance = 0.002
                            position_changed = (self._preview_hover_pos is None or
                                              abs(self._preview_hover_pos - normalized_pos) > position_tolerance)
                            
                            if position_changed:
                                self._preview_hover_pos = normalized_pos
                                self._preview_hover_start_time = time.time()
                                # Clear cached data when position changes
                                self._preview_cached_tooltip_data = None
                                self._preview_cached_pos = None
                            
                            # Show enhanced preview logic
                            hover_duration = time.time() - self._preview_hover_start_time if self._preview_hover_start_time else 0
                            enhanced_preview_enabled = self.app.app_settings.config.funscript.enable_enhanced_preview
                            
                            if enhanced_preview_enabled:
                                # Always show timestamp + script zoom instantly, then add video frame after delay
                                show_video_frame = hover_duration > 0.1  # Video frame after 100ms hover stability
                                
                                # Check if we have cached data for this position (with tight tolerance)
                                if (self._preview_cached_tooltip_data is not None and 
                                    self._preview_cached_pos is not None and 
                                    abs(self._preview_cached_pos - normalized_pos) <= position_tolerance):
                                    
                                    # If we need video frame but cached data doesn't have it, fetch async
                                    cached_frame_data = self._preview_cached_tooltip_data.get('frame_data')
                                    if show_video_frame and (cached_frame_data is None or cached_frame_data.size == 0):
                                        # Check if we already have a pending fetch for this frame
                                        if not hasattr(self, '_preview_frame_fetch_pending'):
                                            self._preview_frame_fetch_pending = False

                                        if not self._preview_frame_fetch_pending:
                                            # Start async fetch in background thread
                                            self._preview_frame_fetch_pending = True
                                            self._preview_cached_tooltip_data['frame_loading'] = True  # Set immediately so tooltip shows "Loading..."
                                            cached_hover_frame = self._preview_cached_tooltip_data.get('hover_frame', hover_frame)

                                            def fetch_frame_async():
                                                try:
                                                    frame_data, actual_frame = self._get_frame_direct_cv2(cached_hover_frame)
                                                    if frame_data is not None and frame_data.size > 0:
                                                        self._preview_cached_tooltip_data['frame_data'] = frame_data
                                                        self._preview_cached_tooltip_data['actual_frame'] = actual_frame
                                                        self._preview_cached_tooltip_data['frame_loading'] = False
                                                except Exception:
                                                    self._preview_cached_tooltip_data['frame_loading'] = False
                                                finally:
                                                    self._preview_frame_fetch_pending = False

                                            threading.Thread(target=fetch_frame_async, daemon=True).start()

                                    # Use cached tooltip data (frame number and video frame are consistent)
                                    self._render_instant_enhanced_tooltip(self._preview_cached_tooltip_data, show_video_frame)
                                else:
                                    # Generate new tooltip data (without slow video frame extraction initially)
                                    try:
                                        tooltip_data = self._generate_instant_tooltip_data(
                                            hover_time_s, hover_frame, total_duration, normalized_pos, show_video_frame
                                        )
                                        self._preview_cached_tooltip_data = tooltip_data
                                        self._preview_cached_pos = normalized_pos

                                        # Launch async frame fetch if needed
                                        if show_video_frame and tooltip_data.get('frame_loading', False):
                                            if not hasattr(self, '_preview_frame_fetch_pending'):
                                                self._preview_frame_fetch_pending = False

                                            if not self._preview_frame_fetch_pending:
                                                self._preview_frame_fetch_pending = True

                                                def fetch_frame_async():
                                                    try:
                                                        frame_data, actual_frame = self._get_frame_direct_cv2(hover_frame)
                                                        if frame_data is not None and frame_data.size > 0:
                                                            self._preview_cached_tooltip_data['frame_data'] = frame_data
                                                            self._preview_cached_tooltip_data['actual_frame'] = actual_frame
                                                            self._preview_cached_tooltip_data['frame_loading'] = False
                                                    except Exception:
                                                        self._preview_cached_tooltip_data['frame_loading'] = False
                                                    finally:
                                                        self._preview_frame_fetch_pending = False

                                                threading.Thread(target=fetch_frame_async, daemon=True).start()

                                        self._render_instant_enhanced_tooltip(tooltip_data, show_video_frame)
                                    except Exception as e:
                                        # Fallback to simple tooltip
                                        imgui.set_tooltip(f"{_format_time(self.app, hover_time_s)} / {_format_time(self.app, total_duration)}")
                            else:
                                # Show simple time tooltip immediately
                                imgui.set_tooltip(f"{_format_time(self.app, hover_time_s)} / {_format_time(self.app, total_duration)}")
        else:
            # Reset hover tracking when mouse leaves
            if hasattr(self, '_preview_hover_start_time'):
                self._preview_hover_start_time = None
                self._preview_hover_pos = None
                self._preview_frame_fetch_pending = False

        # Draw playback marker over the image
        if self.app.file_manager.video_path and self.app.processor and self.app.processor.video_info and self.app.processor.current_frame_index >= 0:
            total_frames = self.app.processor.video_info.get('total_frames', 0)
            if total_frames > 0:
                # Time-based normalization for accurate end-of-video positioning
                fps = self.app.processor.fps if self.app.processor.fps > 0 else 30.0
                current_time_s = self.app.processor.current_frame_index / fps
                normalized_pos = current_time_s / total_duration_s if total_duration_s > 0 else 0
                marker_x = canvas_p1_x + normalized_pos * current_bar_width_float
                marker_color = imgui.get_color_u32_rgba(*CurrentTheme.RED)
                draw_list_marker = imgui.get_window_draw_list()

                # Draw triangle
                triangle_p1 = (marker_x - 5, canvas_p1_y_offset)
                triangle_p2 = (marker_x + 5, canvas_p1_y_offset)
                triangle_p3 = (marker_x, canvas_p1_y_offset + 5)
                draw_list_marker.add_triangle_filled(triangle_p1[0], triangle_p1[1], triangle_p2[0], triangle_p2[1], triangle_p3[0], triangle_p3[1], marker_color)

                # Vertical playhead line drawn once in _core.render() spanning all nav bars.

                # Draw text
                current_frame = self.app.processor.current_frame_index
                current_time_s = self.app.processor.current_frame_index / self.app.processor.video_info.get('fps', 30.0)
                text = f"{_format_time(self.app, current_time_s)} ({current_frame})"
                text_size = imgui.calc_text_size(text)
                text_pos_x = marker_x - text_size[0] / 2
                if text_pos_x < canvas_p1_x:
                    text_pos_x = canvas_p1_x
                if text_pos_x + text_size[0] > canvas_p1_x + current_bar_width_float:
                    text_pos_x = canvas_p1_x + current_bar_width_float - text_size[0]
                text_pos = (text_pos_x, canvas_p1_y_offset - text_size[1] - 2)
                draw_list_marker.add_text(text_pos[0], text_pos[1], imgui.get_color_u32_rgba(*colors.WHITE), text)


    # --- This function now submits a task to the worker thread ---
    def render_funscript_heatmap_preview(self, total_video_duration_s: float, bar_width_float: float, bar_height_float: float):
        app_state = self.app.app_state_ui
        current_bar_width_int = int(round(bar_width_float))
        if current_bar_width_int <= 0 or app_state.heatmap_texture_fixed_height <= 0 or not self.heatmap_texture_id:
            imgui.dummy(bar_width_float, bar_height_float)
            return

        heatmap_axis = self._get_active_preview_axis()
        current_action_count = len(self.app.funscript_processor.get_actions(heatmap_axis))
        is_live_tracking = self.app.processor and self.app.processor.tracker and self.app.processor.tracker.tracking_active

        axis_changed = getattr(self, '_last_heatmap_axis', 'primary') != heatmap_axis
        self._last_heatmap_axis = heatmap_axis
        # Texture is fixed-width (1 x HEATMAP_TEX_WIDTH); display width changes
        # are pure GPU stretching and do NOT trigger regeneration anymore.
        full_redraw_needed = (
            app_state.heatmap_dirty or axis_changed
            or abs(total_video_duration_s - app_state.last_heatmap_video_duration_s) > 0.01)

        incremental_update_needed = current_action_count != self.last_submitted_action_count_heatmap

        needs_regen = full_redraw_needed or (incremental_update_needed and (not is_live_tracking or (time.time() - self.last_preview_update_time_heatmap >= self.preview_update_interval_seconds)))

        if needs_regen:
            actions_copy = self.app.funscript_processor.get_actions(heatmap_axis).copy()
            task = {
                'type': 'heatmap',
                'target_width': current_bar_width_int,  # ignored by worker, kept for API
                'target_height': app_state.heatmap_texture_fixed_height,
                'total_duration_s': total_video_duration_s,
                'actions': actions_copy
            }
            try:
                self.preview_task_queue.put_nowait(task)
            except queue.Full:
                pass

            # Update state after submission
            app_state.heatmap_dirty = False
            app_state.last_heatmap_bar_width = current_bar_width_int
            app_state.last_heatmap_video_duration_s = total_video_duration_s
            self.last_submitted_action_count_heatmap = current_action_count
            if is_live_tracking and incremental_update_needed:
                self.last_preview_update_time_heatmap = time.time()

        # Render the existing texture
        imgui.image(self.heatmap_texture_id, bar_width_float, bar_height_float, uv0=(0, 0), uv1=(1, 1))

    # All other methods from the original file from this point are included below without modification
    # for completeness, except for the `run` method's `finally` block which now handles thread shutdown.

    def _draw_fps_marks_on_slider(self, draw_list, min_rect, max_rect, current_target_fps, tracker_fps, processor_fps):
        if not imgui.is_item_visible():
            return

        app_state = self.app.app_state_ui
        colors = self.colors
        marks = [(current_target_fps, colors.FPS_TARGET_MARKER, "Target"), (tracker_fps, colors.FPS_TRACKER_MARKER, "Tracker"), (processor_fps, colors.FPS_PROCESSOR_MARKER, "Processor")]
        slider_x_start, slider_x_end = min_rect.x, max_rect.x
        slider_width = slider_x_end - slider_x_start
        slider_y = (min_rect.y + max_rect.y) / 2
        for mark_fps, color_rgb, label_text in marks:
            if not (app_state.fps_slider_min_val <= mark_fps <= app_state.fps_slider_max_val): continue
            norm = (mark_fps - app_state.fps_slider_min_val) / (
                    app_state.fps_slider_max_val - app_state.fps_slider_min_val)
            x_pos = slider_x_start + norm * slider_width
            color_u32 = imgui.get_color_u32_rgba(color_rgb[0] / 255, color_rgb[1] / 255, color_rgb[2] / 255, 1.0)
            draw_list.add_line(x_pos, slider_y - 6, x_pos, slider_y + 6, color_u32, thickness=1.5)

    # --- Shortcut handler methods moved to gui_shortcut_handler.py (ShortcutHandlerMixin) ---
    # --- Methods: _handle_global_shortcuts, _handle_arrow_navigation, _perform_frame_seek,
    #     _handle_set_chapter_start_shortcut, _handle_set_chapter_end_shortcut,
    #     _handle_add_point_at_value, _get_current_frame_for_chapter,
    #     _auto_create_chapter_from_stored_frames, _handle_save_project_shortcut,
    #     _handle_open_project_shortcut, _handle_jump_to_start_shortcut,
    #     _handle_jump_to_end_shortcut, _handle_zoom_in_timeline_shortcut,
    #     _handle_zoom_out_timeline_shortcut, _handle_toggle_video_display_shortcut,
    #     _handle_toggle_timeline2_shortcut, _handle_toggle_3d_simulator_shortcut,
    #     _handle_toggle_chapter_list_shortcut, _handle_toggle_heatmap_shortcut,
    #     _handle_toggle_funscript_preview_shortcut, _handle_toggle_video_feed_shortcut,
    #     _handle_toggle_waveform_shortcut, _handle_reset_timeline_view_shortcut,
    #     _handle_toggle_oscillation_area_mode, _handle_energy_saver_interaction_detection ---

    # ---- Shortcut hint helpers (status strip) ----

    def _format_shortcut_hint(self, key_str):
        """Platform-aware formatting: SUPER->Cmd/Ctrl, arrow symbols, etc."""
        d = key_str
        is_mac = platform.system() == "Darwin"
        d = d.replace("SUPER", "Cmd" if is_mac else "Ctrl")
        d = d.replace("CTRL", "Ctrl")
        d = d.replace("ALT", "Alt")
        d = d.replace("SHIFT", "Shift")
        d = d.replace("RIGHT_ARROW", "Right")
        d = d.replace("LEFT_ARROW", "Left")
        d = d.replace("UP_ARROW", "Up")
        d = d.replace("DOWN_ARROW", "Down")
        d = d.replace("SPACE", "Space")
        d = d.replace("BACKSPACE", "Backspace")
        d = d.replace("DELETE", "Del")
        d = d.replace("HOME", "Home")
        d = d.replace("END", "End")
        d = d.replace("EQUAL", "=")
        d = d.replace("MINUS", "-")
        return d

    def _get_contextual_hints(self, shortcuts, timeline_hovered, has_selection, active_mode=None):
        """Return 1-2 fixed hints based on current app state.

        Each hint is (formatted_key, label).
        """
        proc = self.app.processor
        has_video = proc and proc.is_video_open()

        if not has_video:
            key = shortcuts.get("open_project", "")
            return [(self._format_shortcut_hint(key), "Open Project")] if key else []

        # Timeline hovered with selection -> editing hints
        if has_selection:
            hints = []
            k = shortcuts.get("delete_selected_point", "")
            if k:
                hints.append((self._format_shortcut_hint(k), "Delete"))
            k_up = shortcuts.get("nudge_selection_pos_up", "")
            k_dn = shortcuts.get("nudge_selection_pos_down", "")
            if k_up and k_dn:
                hints.append((self._format_shortcut_hint(k_up) + "/" + self._format_shortcut_hint(k_dn), "Nudge Value"))
            k = shortcuts.get("copy_selection", "")
            if k:
                hints.append((self._format_shortcut_hint(k), "Copy"))
            return hints

        # Timeline hovered, no selection -> mode-specific or canvas interaction hints
        if timeline_hovered:
            if active_mode == TimelineMode.ALTERNATING:
                return [("Click", "Add Alt Point"), ("Alt+Drag", "Range Select")]
            elif active_mode == TimelineMode.RECORDING:
                k = shortcuts.get("toggle_playback", "")
                hints = [("Record", "Start/Stop")]
                if k:
                    hints.append((self._format_shortcut_hint(k), "Play/Pause"))
                return hints
            elif active_mode == TimelineMode.INJECTION:
                return [("Right-click", "Inject Points"), ("Alt+Drag", "Range Select")]

            # Default Select mode
            hints = []
            hints.append(("Alt+Drag", "Range Select"))
            hints.append(("Click", "Add Point"))
            k = shortcuts.get("select_all_points", "")
            if k:
                hints.append((self._format_shortcut_hint(k), "Select All"))
            return hints

        # Default: video loaded, not hovering timeline
        hints = []
        k = shortcuts.get("toggle_playback", "")
        if k:
            hints.append((self._format_shortcut_hint(k), "Play/Pause"))
        k_l = shortcuts.get("seek_prev_frame", "")
        k_r = shortcuts.get("seek_next_frame", "")
        if k_l and k_r:
            hints.append((self._format_shortcut_hint(k_l) + "/" + self._format_shortcut_hint(k_r), "Navigate"))
        return hints

    def _build_rotating_hint_pool(self, shortcuts, fixed_actions):
        """Build pool of discovery hints from shortcut categories, excluding fixed ones.

        Returns list of (formatted_key, label) tuples.
        """
        pool = []
        fixed_set = set(fixed_actions)
        proc = self.app.processor
        has_video = proc and proc.is_video_open()

        # Categories to skip entirely when no video
        skip_no_video = {"Editing", "Point Navigation", "Playback", "Add Points",
                         "Chapters", "Tracking Tools", "Bookmarks"}

        for cat_name, cat_shortcuts in self.keyboard_shortcuts_dialog.shortcut_categories.items():
            if not has_video and cat_name in skip_no_video:
                continue
            for action_name, display_name in cat_shortcuts:
                if action_name in fixed_set:
                    continue
                # Skip _alt duplicates
                if action_name.endswith("_alt"):
                    continue
                # Skip number-pad point shortcuts (too many)
                if action_name.startswith("add_point_"):
                    continue
                key_str = shortcuts.get(action_name, "")
                if not key_str:
                    continue
                pool.append((self._format_shortcut_hint(key_str), display_name))
        return pool

    def _render_fixed_layout(self, app_state, font_scale, toolbar_height, status_strip_h):
        """Render fixed-position layout with calculated panel geometry."""
        panel_y_start = self.main_menu_bar_height + toolbar_height
        # Give the single visible interactive timeline more vertical room so
        # short strokes are easier to edit. With multiple timelines we keep the
        # compact base height so they all fit.
        single_tl_visible = (
            app_state.show_funscript_interactive_timeline
            and not app_state.show_funscript_interactive_timeline2
            and not self._visible_extra_timelines
        )
        base_h = app_state.timeline_base_height
        single_h = int(base_h * 1.7)
        timeline1_render_h = (single_h if single_tl_visible else base_h) if app_state.show_funscript_interactive_timeline else 0
        timeline2_render_h = base_h if app_state.show_funscript_interactive_timeline2 else 0
        extra_timelines_total_height = len(self._visible_extra_timelines) * base_h
        sub_timeline_h = 65 if getattr(app_state, 'show_subtitle_timeline', False) else 0
        interactive_timelines_total_height = timeline1_render_h + timeline2_render_h + extra_timelines_total_height + sub_timeline_h
        max_timeline_area_h = int(self.window_height * 0.45)
        capped_timelines_h = min(interactive_timelines_total_height, max_timeline_area_h)
        timelines_need_scroll = interactive_timelines_total_height > max_timeline_area_h
        available_height_for_main_panels = max(100, self.window_height - panel_y_start - capped_timelines_h - status_strip_h)
        if not hasattr(app_state, 'fixed_layout_geometry') or app_state.fixed_layout_geometry is None:
            app_state.fixed_layout_geometry = {}
        else:
            app_state.fixed_layout_geometry.clear()
        is_full_width_nav = getattr(app_state, 'full_width_nav', False)
        control_panel_w = _FIXED_PANEL_BASE_WIDTH * font_scale
        graphs_panel_w = _FIXED_PANEL_BASE_WIDTH * font_scale
        video_nav_bar_h = _VIDEO_NAV_BAR_HEIGHT

        if is_full_width_nav:
            top_panels_h = max(50, available_height_for_main_panels - video_nav_bar_h)
            nav_y_start = panel_y_start + top_panels_h
            if True:
                CHEVRON_W = 16
                left_show_top = bool(getattr(app_state, 'show_left_top_block', True))
                left_show_bot = bool(getattr(app_state, 'show_left_bottom_block', True))
                right_show_top = bool(getattr(app_state, 'show_right_top_block', True))
                right_show_bot = bool(getattr(app_state, 'show_right_bottom_block', True))
                left_collapsed = not (left_show_top or left_show_bot)
                right_collapsed = not (right_show_top or right_show_bot)
                left_col_w = CHEVRON_W if left_collapsed else control_panel_w
                right_col_w = CHEVRON_W if right_collapsed else graphs_panel_w
                video_panel_w = self.window_width - left_col_w - right_col_w
                if video_panel_w < 100:
                    video_panel_w = 100
                    right_col_w = max(CHEVRON_W, self.window_width - left_col_w - video_panel_w)
                video_area_x_start = left_col_w
                graphs_area_x_start = left_col_w + video_panel_w
                self._render_left_column(app_state, 0, panel_y_start, left_col_w,
                                         top_panels_h, left_show_top, left_show_bot,
                                         left_collapsed, CHEVRON_W)
                app_state.fixed_layout_geometry['VideoDisplay'] = {'pos': (video_area_x_start, panel_y_start), 'size': (video_panel_w, top_panels_h)}
                imgui.set_next_window_position(video_area_x_start, panel_y_start)
                imgui.set_next_window_size(video_panel_w, top_panels_h)
                self._time_render("VideoDisplayUI", self.video_display_ui.render)
                self._render_right_column(app_state, graphs_area_x_start, panel_y_start,
                                          right_col_w, top_panels_h, right_show_top,
                                          right_show_bot, right_collapsed, CHEVRON_W)
            else:
                control_panel_w_no_vid = self.window_width / 2
                graphs_panel_w_no_vid = self.window_width - control_panel_w_no_vid
                graphs_area_x_start_no_vid = control_panel_w_no_vid
                app_state.fixed_layout_geometry['ControlPanel'] = {'pos': (0, panel_y_start), 'size': (control_panel_w_no_vid, top_panels_h)}
                imgui.set_next_window_position(0, panel_y_start)
                imgui.set_next_window_size(control_panel_w_no_vid, top_panels_h)
                self._time_render("ControlPanelUI", self.control_panel_ui.render)
                app_state.fixed_layout_geometry['InfoGraphs'] = {'pos': (graphs_area_x_start_no_vid, panel_y_start), 'size': (graphs_panel_w_no_vid, top_panels_h)}
                imgui.set_next_window_position(graphs_area_x_start_no_vid, panel_y_start)
                imgui.set_next_window_size(graphs_panel_w_no_vid, top_panels_h)
                self._time_render("InfoGraphsUI", self.info_graphs_ui.render)
            app_state.fixed_layout_geometry['VideoNavigation'] = {'pos': (0, nav_y_start), 'size': (self.window_width, video_nav_bar_h)}
            imgui.set_next_window_position(0, nav_y_start)
            imgui.set_next_window_size(self.window_width, video_nav_bar_h)
            self._time_render("VideoNavigationUI", self.video_navigation_ui.render, self.window_width)
        else:
            if True:
                # Quad-block layout: left column (control panel top, chapters/bookmarks bottom),
                # right column (info graphs top, plugins bottom). Each block independently
                # collapsible. Collapsed columns shrink to a thin chevron strip.
                CHEVRON_W = 16
                left_show_top = bool(getattr(app_state, 'show_left_top_block', True))
                left_show_bot = bool(getattr(app_state, 'show_left_bottom_block', True))
                right_show_top = bool(getattr(app_state, 'show_right_top_block', True))
                right_show_bot = bool(getattr(app_state, 'show_right_bottom_block', True))
                left_collapsed = not (left_show_top or left_show_bot)
                right_collapsed = not (right_show_top or right_show_bot)
                left_col_w = CHEVRON_W if left_collapsed else control_panel_w
                right_col_w = CHEVRON_W if right_collapsed else graphs_panel_w
                video_panel_w = self.window_width - left_col_w - right_col_w
                if video_panel_w < 100:
                    video_panel_w = 100
                    right_col_w = max(CHEVRON_W, self.window_width - left_col_w - video_panel_w)
                video_render_h = max(50, available_height_for_main_panels - video_nav_bar_h)
                video_area_x_start = left_col_w
                graphs_area_x_start = left_col_w + video_panel_w

                self._render_left_column(app_state, 0, panel_y_start, left_col_w,
                                         available_height_for_main_panels,
                                         left_show_top, left_show_bot, left_collapsed, CHEVRON_W)

                # ----- MIDDLE (video + nav) -----
                app_state.fixed_layout_geometry['VideoDisplay'] = {'pos': (video_area_x_start, panel_y_start), 'size': (video_panel_w, video_render_h)}
                imgui.set_next_window_position(video_area_x_start, panel_y_start)
                imgui.set_next_window_size(video_panel_w, video_render_h)
                self._time_render("VideoDisplayUI", self.video_display_ui.render)
                app_state.fixed_layout_geometry['VideoNavigation'] = {
                    'pos': (video_area_x_start, panel_y_start + video_render_h),
                    'size': (video_panel_w, video_nav_bar_h)}
                imgui.set_next_window_position(video_area_x_start, panel_y_start + video_render_h)
                imgui.set_next_window_size(video_panel_w, video_nav_bar_h)
                self._time_render("VideoNavigationUI", self.video_navigation_ui.render, video_panel_w)

                self._render_right_column(app_state, graphs_area_x_start, panel_y_start,
                                          right_col_w, available_height_for_main_panels,
                                          right_show_top, right_show_bot, right_collapsed, CHEVRON_W)
            else:
                control_panel_w_no_vid = self.window_width / 2
                graphs_panel_w_no_vid = self.window_width - control_panel_w_no_vid
                graphs_area_x_start_no_vid = control_panel_w_no_vid
                app_state.fixed_layout_geometry['ControlPanel'] = {'pos': (0, panel_y_start), 'size': (control_panel_w_no_vid, available_height_for_main_panels)}
                imgui.set_next_window_position(0, panel_y_start)
                imgui.set_next_window_size(control_panel_w_no_vid, available_height_for_main_panels)
                self._time_render("ControlPanelUI", self.control_panel_ui.render)
                app_state.fixed_layout_geometry['InfoGraphs'] = {'pos': (graphs_area_x_start_no_vid, panel_y_start), 'size': (graphs_panel_w_no_vid, available_height_for_main_panels)}
                imgui.set_next_window_position(graphs_area_x_start_no_vid, panel_y_start)
                imgui.set_next_window_size(graphs_panel_w_no_vid, available_height_for_main_panels)
                self._time_render("InfoGraphsUI", self.info_graphs_ui.render)

        timeline_area_y = panel_y_start + available_height_for_main_panels
        per_tl_h = app_state.timeline_base_height

        if timelines_need_scroll:
            imgui.set_next_window_position(0, timeline_area_y)
            imgui.set_next_window_size(self.window_width, capped_timelines_h)
            container_flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                               imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)
            if imgui.begin("##TimelineScrollContainer", True, container_flags):
                # Subtitle timeline first (above funscript timelines)
                if getattr(app_state, 'show_subtitle_timeline', False) and hasattr(self, '_subtitle_timeline'):
                    self._time_render("SubtitleTimeline", self._subtitle_timeline.render,
                                      0, 65, container_mode=True)
                if app_state.show_funscript_interactive_timeline:
                    self._time_render("TimelineEditor1", self.timeline_editor1.render,
                                      0, per_tl_h, container_mode=True)
                if app_state.show_funscript_interactive_timeline2:
                    self._time_render("TimelineEditor2", self.timeline_editor2.render,
                                      0, per_tl_h, container_mode=True)
                for t_num in self._visible_extra_timelines:
                    editor = self._get_or_create_timeline_editor(t_num)
                    self._time_render(f"TimelineEditor{t_num}", editor.render,
                                      0, per_tl_h, container_mode=True)
            imgui.end()
            app_state.fixed_layout_geometry['TimelineContainer'] = {
                'pos': (0, timeline_area_y), 'size': (self.window_width, capped_timelines_h)}
        else:
            timeline_current_y_start = timeline_area_y
            # Subtitle timeline first (above funscript timelines)
            if getattr(app_state, 'show_subtitle_timeline', False) and hasattr(self, '_subtitle_timeline'):
                app_state.fixed_layout_geometry['SubtitleTimeline'] = {'pos': (0, timeline_current_y_start), 'size': (self.window_width, sub_timeline_h)}
                self._time_render("SubtitleTimeline", self._subtitle_timeline.render, timeline_current_y_start, sub_timeline_h)
                timeline_current_y_start += sub_timeline_h
            if app_state.show_funscript_interactive_timeline:
                app_state.fixed_layout_geometry['Timeline1'] = {'pos': (0, timeline_current_y_start), 'size': (self.window_width, timeline1_render_h)}
                self._time_render("TimelineEditor1", self.timeline_editor1.render, timeline_current_y_start, timeline1_render_h)
                timeline_current_y_start += timeline1_render_h
            if app_state.show_funscript_interactive_timeline2:
                app_state.fixed_layout_geometry['Timeline2'] = {'pos': (0, timeline_current_y_start), 'size': (self.window_width, timeline2_render_h)}
                self._time_render("TimelineEditor2", self.timeline_editor2.render, timeline_current_y_start, timeline2_render_h)
                timeline_current_y_start += timeline2_render_h
            for t_num in self._visible_extra_timelines:
                editor = self._get_or_create_timeline_editor(t_num)
                extra_h = per_tl_h
                app_state.fixed_layout_geometry[f'Timeline{t_num}'] = {'pos': (0, timeline_current_y_start), 'size': (self.window_width, extra_h)}
                self._time_render(f"TimelineEditor{t_num}", editor.render, timeline_current_y_start, extra_h)
                timeline_current_y_start += extra_h

    _SASH_THICKNESS = ui_metrics.SASH_PX

    def _render_block_sash(self, side: str, x: float, y: float, w: float,
                           h: float, total_h: float, frac_attr: str,
                           app_state) -> float:
        """Invisible-button sash between top/bottom blocks. Drag adjusts the
        bottom-block height fraction. Returns the new frac (0..1)."""
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE
                 | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_BACKGROUND
                 | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)
        imgui.set_next_window_position(x, y)
        imgui.set_next_window_size(w, h)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
        imgui.begin(f"##sash_{side}", flags=flags)
        imgui.invisible_button(f"##sash_btn_{side}", w, h)
        if imgui.is_item_hovered():
            try:
                imgui.set_mouse_cursor(imgui.MOUSE_CURSOR_RESIZE_NS)
            except Exception:
                pass
        cur = float(getattr(app_state, frac_attr, 0.45))
        if imgui.is_item_active():
            io = imgui.get_io()
            cur = max(0.15, min(0.85, cur - io.mouse_delta[1] / max(1.0, total_h)))
            setattr(app_state, frac_attr, cur)
        # Hairline separator
        dl = imgui.get_window_draw_list()
        col = imgui.get_color_u32_rgba(0.4, 0.4, 0.45, 0.6 if imgui.is_item_active() or imgui.is_item_hovered() else 0.25)
        dl.add_line(x, y + h * 0.5, x + w, y + h * 0.5, col, 1.0)
        imgui.end()
        imgui.pop_style_var()
        return cur

    def _render_left_column(self, app_state, x, y, w, h,
                            show_top, show_bot, collapsed, chevron_w):
        if collapsed:
            render_collapsed_chevron('left', x, y, chevron_w, h, app_state)
            return
        lb_frac = float(getattr(app_state, 'left_bottom_block_height_frac', 0.45))
        lb_frac = max(0.15, min(0.85, lb_frac))
        sash_h = self._SASH_THICKNESS if (show_top and show_bot) else 0
        if show_top and show_bot:
            usable = max(0, h - sash_h)
            top_h = int(usable * (1.0 - lb_frac))
            bot_h = usable - top_h
        elif show_top:
            top_h, bot_h = h, 0
        else:
            top_h, bot_h = 0, h
        if show_top:
            app_state.fixed_layout_geometry['ControlPanel'] = {'pos': (x, y), 'size': (w, top_h)}
            imgui.set_next_window_position(x, y)
            imgui.set_next_window_size(w, top_h)
            self._time_render("ControlPanelUI", self.control_panel_ui.render)
        if show_top and show_bot:
            self._render_block_sash('left', x, y + top_h, w, sash_h, h,
                                    'left_bottom_block_height_frac', app_state)
        if show_bot:
            by = y + top_h + sash_h
            app_state.fixed_layout_geometry['LeftBottomBlock'] = {'pos': (x, by), 'size': (w, bot_h)}
            imgui.set_next_window_position(x, by)
            imgui.set_next_window_size(w, bot_h)
            self._time_render("LeftBottomBlock", self.left_bottom_block.render)

    def _render_right_column(self, app_state, x, y, w, h,
                             show_top, show_bot, collapsed, chevron_w):
        if collapsed:
            render_collapsed_chevron('right', x, y, chevron_w, h, app_state)
            return
        rb_frac = float(getattr(app_state, 'right_bottom_block_height_frac', 0.45))
        rb_frac = max(0.15, min(0.85, rb_frac))
        sash_h = self._SASH_THICKNESS if (show_top and show_bot) else 0
        if show_top and show_bot:
            usable = max(0, h - sash_h)
            top_h = int(usable * (1.0 - rb_frac))
            bot_h = usable - top_h
        elif show_top:
            top_h, bot_h = h, 0
        else:
            top_h, bot_h = 0, h
        if show_top:
            app_state.fixed_layout_geometry['InfoGraphs'] = {'pos': (x, y), 'size': (w, top_h)}
            imgui.set_next_window_position(x, y)
            imgui.set_next_window_size(w, top_h)
            self._time_render("InfoGraphsUI", self.info_graphs_ui.render)
        if show_top and show_bot:
            self._render_block_sash('right', x, y + top_h, w, sash_h, h,
                                    'right_bottom_block_height_frac', app_state)
        if show_bot:
            by = y + top_h + sash_h
            app_state.fixed_layout_geometry['RightBottomBlock'] = {'pos': (x, by), 'size': (w, bot_h)}
            imgui.set_next_window_position(x, by)
            imgui.set_next_window_size(w, bot_h)
            self._time_render("RightBottomBlock", self.right_bottom_block.render)

    def _render_floating_layout(self, app_state):
        """Render floating-window layout."""
        if app_state.just_switched_to_floating:
            if 'ControlPanel' in app_state.fixed_layout_geometry:
                geom = app_state.fixed_layout_geometry['ControlPanel']
                imgui.set_next_window_position(geom['pos'][0], geom['pos'][1], condition=imgui.APPEARING)
                imgui.set_next_window_size(geom['size'][0], geom['size'][1], condition=imgui.APPEARING)
            if 'VideoDisplay' in app_state.fixed_layout_geometry:
                geom = app_state.fixed_layout_geometry['VideoDisplay']
                imgui.set_next_window_position(geom['pos'][0], geom['pos'][1], condition=imgui.APPEARING)
                imgui.set_next_window_size(geom['size'][0], geom['size'][1], condition=imgui.APPEARING)

        self._time_render("ControlPanelUI", self.control_panel_ui.render)
        self._time_render("InfoGraphsUI", self.info_graphs_ui.render)
        self._time_render("VideoDisplayUI", self.video_display_ui.render)
        self._time_render("VideoNavigationUI", self.video_navigation_ui.render)
        self._time_render("TimelineEditor1", self.timeline_editor1.render)
        self._time_render("TimelineEditor2", self.timeline_editor2.render)

        # Side blocks as floating closable+resizable windows.
        self._time_render("LeftBottomBlock", self.left_bottom_block.render, floating=True)
        self._time_render("RightBottomBlock", self.right_bottom_block.render, floating=True)

        for t_num in self._visible_extra_timelines:
            editor = self._get_or_create_timeline_editor(t_num)
            self._time_render(f"TimelineEditor{t_num}", editor.render)

        if app_state.just_switched_to_floating:
            app_state.just_switched_to_floating = False

    def _render_dialogs_and_overlays(self, app_state):
        """Render floating dialogs, popups, and overlay windows."""
        if hasattr(app_state, 'show_chapter_list_window') and app_state.show_chapter_list_window:
            self._time_render("ChapterListWindow", self.chapter_list_window_ui.render)
        if hasattr(app_state, 'show_chapter_type_manager') and app_state.show_chapter_type_manager:
            self._time_render("ChapterTypeManager", self.chapter_type_manager_ui.render)
        if getattr(app_state, 'show_bookmark_list_window', False):
            self._time_render("BookmarkListWindow", self.bookmark_list_window_ui.render)
        self._time_render("Popups", self._render_all_popups)
        self._time_render("ErrorPopup", self._render_error_popup)

        if hasattr(self.app, 'tensorrt_compiler_window') and self.app.tensorrt_compiler_window:
            self._time_render("TensorRTCompiler", self.app.tensorrt_compiler_window.render)

        # Subtitle editor floating window (opened from control panel tab)
        if _is_feature_available("subtitle_translation"):
            cp = self.control_panel_ui
            tool = getattr(cp, '_subtitle_tool', None)
            if tool and tool.is_open:
                self._time_render("SubtitleEditor", tool.render)

        if self.app.app_state_ui.show_generated_file_manager:
            self._time_render("GeneratedFileManager", self.generated_file_manager_ui.render)

        # Proxy (iframe transcode) suggestion + progress modals
        pc = getattr(self.app, "proxy_controller", None)
        if pc is not None:
            self._time_render("ProxyDialogs", pc.render)

        if hasattr(app_state, 'show_ai_models_dialog') and app_state.show_ai_models_dialog:
            self._time_render("AIModelsDialog", self._render_ai_models_dialog)

        self._render_go_to_frame_popup()
        self.app.addon_checker.tick_status_ads()
        self._render_tracker_debug_frame()

    def _render_tracker_debug_frame(self):
        """Render a debug overlay window if the current tracker provides a debug_frame."""
        tracker = self.app.tracker
        if not tracker:
            return
        debug_frame = getattr(tracker, 'debug_frame', None)

        # Toggle visibility: show window when debug_frame exists, hide when it doesn't
        if debug_frame is None:
            self._show_debug_frame_window = False
            return
        self._show_debug_frame_window = True

        h, w = debug_frame.shape[:2]
        if w == 0 or h == 0:
            return

        imgui.set_next_window_size(min(w + 16, 520), min(h + 40, 520), imgui.FIRST_USE_EVER)
        expanded, opened = imgui.begin("FunGen: Tracker Debug##DebugFrame", True)
        if not opened:
            self._show_debug_frame_window = False
            # Tracker can check this to skip expensive debug frame generation
            tracker.debug_frame = None
            imgui.end()
            return
        if expanded:
            self.update_texture(self._debug_frame_texture_id, debug_frame)
            avail_w = imgui.get_content_region_available_width()
            scale = avail_w / w if w > 0 else 1.0
            imgui.image(self._debug_frame_texture_id, avail_w, int(h * scale))
        imgui.end()

    def _render_status_strip(self, strip_h):
        """Render a unified status strip at the very bottom of the window."""
        from config.element_group_colors import StatusStripColors

        strip_y = self.window_height - strip_h
        imgui.set_next_window_position(0, strip_y)
        imgui.set_next_window_size(self.window_width, strip_h)
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                 imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR |
                 imgui.WINDOW_NO_SCROLL_WITH_MOUSE |
                 imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS |
                 imgui.WINDOW_NO_NAV | imgui.WINDOW_NO_FOCUS_ON_APPEARING)

        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *StatusStripColors.BG)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (8, 2))

        if imgui.begin("##StatusStrip", True, flags):
            app_state = self.app.app_state_ui
            stage_proc = self.app.stage_processor
            proc = self.app.processor

            self._render_status_left(proc, stage_proc, StatusStripColors)
            self._render_status_center(proc, app_state, stage_proc, StatusStripColors)
            self._render_status_right(proc)

        imgui.end()
        imgui.pop_style_var()
        imgui.pop_style_color()

    def _render_status_left(self, proc, stage_proc, colors):
        """Render left section: energy saver state or workflow state."""
        if self.app.energy_saver.energy_saver_active:
            icon_mgr = get_icon_texture_manager()
            leaf_tex, _, _ = icon_mgr.get_icon_texture('energy-leaf.png')
            if leaf_tex:
                imgui.image(leaf_tex, 16, 16)
                imgui.same_line()
            imgui.push_style_color(imgui.COLOR_TEXT, *colors.ENERGY_SAVER)
            imgui.text("Energy Saver")
            imgui.pop_style_color()
        else:
            if not proc or not proc.is_video_open():
                state_text = "No video loaded"
                state_color = colors.TEXT
            elif stage_proc.full_analysis_active:
                state_text = "Tracking..."
                state_color = colors.ACCENT
            elif proc and getattr(proc, 'is_processing', False) and getattr(proc, 'enable_tracker_processing', False):
                if hasattr(proc, 'pause_event') and proc.pause_event.is_set():
                    state_text = "Live Tracking Paused"
                    state_color = colors.WARNING
                else:
                    state_text = "Live Tracking"
                    state_color = colors.ACCENT
            elif proc and getattr(proc, 'is_processing', False) and not getattr(proc, 'enable_tracker_processing', False):
                if hasattr(proc, 'pause_event') and proc.pause_event.is_set():
                    state_text = "Paused"
                    state_color = colors.TEXT
                else:
                    state_text = "Playing"
                    state_color = colors.ACCENT
            else:
                state_text = "Ready"
                state_color = colors.TEXT
            imgui.push_style_color(imgui.COLOR_TEXT, *state_color)
            imgui.text(state_text)
            imgui.pop_style_color()

    def _render_status_center(self, proc, app_state, stage_proc, colors):
        """Render center section: progress %, status message, or shortcut hints."""
        _status_msg, _status_expire = app_state.peek_status()
        _has_status_msg = _status_msg and time.time() < _status_expire

        if stage_proc.full_analysis_active:
            progress = getattr(stage_proc, 'overall_progress', 0.0)
            progress_text = f"{int(progress * 100)}%"
            center_x = self.window_width * 0.5
            text_w = imgui.calc_text_size(progress_text)[0]
            imgui.same_line(position=center_x - text_w * 0.5)
            imgui.push_style_color(imgui.COLOR_TEXT, *colors.ACCENT)
            imgui.text(progress_text)
            imgui.pop_style_color()
        elif _has_status_msg:
            center_x = self.window_width * 0.5
            text_w = imgui.calc_text_size(_status_msg)[0]
            imgui.same_line(position=center_x - text_w * 0.5)
            imgui.push_style_color(imgui.COLOR_TEXT, *colors.ACCENT)
            imgui.text(_status_msg)
            imgui.pop_style_color()
        else:
            if _status_msg and time.time() >= _status_expire:
                app_state.set_status("", 0.0)
            # Dynamic shortcut hints
            shortcuts = self.app.app_settings.get("funscript_editor_shortcuts", {})

            # Detect context: video, timeline hover, selection, active mode
            has_video = proc and proc.is_video_open()
            tl_hovered = False
            has_sel = False
            active_tl_mode = TimelineMode.SELECT
            try:
                t1 = self.timeline_editor1
                t2 = self.timeline_editor2
                if t1 and t1.is_hovered:
                    tl_hovered = True
                    active_tl_mode = getattr(t1, '_mode', TimelineMode.SELECT)
                if t2 and t2.is_hovered:
                    tl_hovered = True
                    active_tl_mode = getattr(t2, '_mode', TimelineMode.SELECT)
                if t1 and t1.multi_selected_action_indices:
                    has_sel = True
                if not has_sel and t2 and t2.multi_selected_action_indices:
                    has_sel = True
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Timeline hover detection error: {e}")

            fixed_hints = self._get_contextual_hints(shortcuts, tl_hovered, has_sel, active_tl_mode)
            ctx_key = (has_video, tl_hovered, has_sel, active_tl_mode)

            # Rebuild rotating pool on context change
            if ctx_key != self._hint_last_context:
                fixed_actions = set()
                if not has_video:
                    fixed_actions.add("open_project")
                elif has_sel:
                    fixed_actions.update(("delete_selected_point", "nudge_selection_pos_up",
                                          "nudge_selection_pos_down", "copy_selection"))
                elif tl_hovered:
                    fixed_actions.add("select_all_points")
                else:
                    fixed_actions.update(("toggle_playback", "seek_prev_frame", "seek_next_frame"))
                self._hint_pool_cache = self._build_rotating_hint_pool(shortcuts, fixed_actions)
                self._hint_last_context = ctx_key
                self._hint_rotate_index = 0

            # Rotate discovery tip every 10s
            now = time.time()
            if self._hint_pool_cache and now - self._hint_last_rotate >= _HINT_ROTATION_INTERVAL_S:
                self._hint_rotate_index = (self._hint_rotate_index + 1) % len(self._hint_pool_cache)
                self._hint_last_rotate = now

            # Build display string: "Key Action . Key Action . Key Action"
            parts = []
            for key_disp, label in fixed_hints:
                parts.append(f"{key_disp} {label}")
            if self._hint_pool_cache:
                rot = self._hint_pool_cache[self._hint_rotate_index % len(self._hint_pool_cache)]
                parts.append(f"{rot[0]} {rot[1]}")

            if parts:
                hint_text = "  \u00b7  ".join(parts)
                center_x = self.window_width * 0.5
                text_w = imgui.calc_text_size(hint_text)[0]
                imgui.same_line(position=center_x - text_w * 0.5)
                imgui.push_style_color(imgui.COLOR_TEXT, 0.50, 0.50, 0.55, 0.75)
                imgui.text(hint_text)
                imgui.pop_style_color()
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Press F1 to see all keyboard shortcuts")

    def _render_status_right(self, proc):
        """Render right section: GUI FPS, Video FPS, Frame Buffer visualization."""
        io = imgui.get_io()
        dim_color = (0.50, 0.50, 0.55, 0.75)
        sep = "  -  "
        if not hasattr(self, '_sep_w'):
            self._sep_w = imgui.calc_text_size(sep)[0]
        sep_w = self._sep_w

        # --- Compute Video FPS (1-second windowed, updated once/sec) ---
        now_t = time.time()
        # Evict stale samples from front of deque (O(k) where k = expired count)
        while self._reading_fps_frames and now_t - self._reading_fps_frames[0][0] >= 1.0:
            self._reading_fps_frames.popleft()
        if now_t - self._reading_fps_last_update >= 1.0:
            if self._reading_fps_frames:
                total_frames = sum(n for _, n in self._reading_fps_frames)
                window = now_t - self._reading_fps_frames[0][0]
                self._reading_fps_display = total_frames / window if window > 0.01 else 0.0
            else:
                self._reading_fps_display = 0.0
            self._reading_fps_last_update = now_t

        video_fps_text = ""
        if proc and proc.is_video_open():
            if self._reading_fps_display > 0.5:
                video_fps_text = f"Video FPS {self._reading_fps_display:.0f}"
            elif proc.fps > 0:
                video_fps_text = f"Video FPS {proc.fps:.0f}"

        # --- Tracker FPS (only while a live tracker is active) ---
        # 3s rolling mean of raw tracker.current_fps to smooth the 1/dt
        # instantaneous samples; displayed value refreshes every 500ms.
        tracker_fps_text = ""
        tracker_fps_color = dim_color
        tracking_now = False
        cur_tracker_fps = 0.0
        if proc and getattr(proc, 'enable_tracker_processing', False):
            tr = getattr(proc, 'tracker', None)
            if tr is not None and getattr(tr, 'tracking_active', False):
                tracking_now = True
                cur_tracker_fps = float(
                    getattr(getattr(tr, '_current_tracker', None),
                            'current_fps', 0.0) or 0.0)
        if not tracking_now:
            self._tracker_fps_display = 0.0
            self._tracker_fps_last_update = 0.0
            self._tracker_fps_samples.clear()
        else:
            if cur_tracker_fps > 0.0:
                self._tracker_fps_samples.append((now_t, cur_tracker_fps))
            cutoff = now_t - 3.0
            while self._tracker_fps_samples and self._tracker_fps_samples[0][0] < cutoff:
                self._tracker_fps_samples.popleft()
            if now_t - self._tracker_fps_last_update >= 0.5 and self._tracker_fps_samples:
                self._tracker_fps_display = (
                    sum(v for _, v in self._tracker_fps_samples)
                    / len(self._tracker_fps_samples))
                self._tracker_fps_last_update = now_t
        if tracking_now:
            tracker_fps_text = f"Tracker FPS {self._tracker_fps_display:.0f}"
            src_fps = float(getattr(proc, 'fps', 0.0) or 0.0)
            if src_fps > 0 and self._tracker_fps_display > 0:
                ratio = self._tracker_fps_display / src_fps
                if ratio < 0.6:
                    tracker_fps_color = (0.95, 0.45, 0.25, 0.9)   # red-ish: can't keep up
                elif ratio < 0.9:
                    tracker_fps_color = (0.95, 0.85, 0.30, 0.9)   # yellow: borderline

        # --- Compute Frame Buffer bar ---
        buf_label = "Frame Buffer "
        buf_label_w = imgui.calc_text_size(buf_label)[0]
        bar_w = 60
        has_buf = False
        green_start_frac = 0.0
        green_end_frac = 0.0
        cursor_frac = 0.0
        buf_size = 0
        buf_capacity = 1
        buf_start = buf_end = buf_current = 0
        if proc and proc.is_video_open():
            has_buf = True
            if hasattr(proc, 'buffer_info'):
                buf = proc.buffer_info
                buf_size = buf['size']
                buf_capacity = buf['capacity'] if buf['capacity'] > 0 else 1
                if buf_size > 0:
                    buf_start = buf['start']
                    buf_end = buf['end']
                    buf_current = buf.get('current', buf_start)
                    bar_left = buf_end - buf_capacity + 1
                    green_start_frac = max(0.0, (buf_start - bar_left) / buf_capacity)
                    green_end_frac = min(1.0, (buf_end - bar_left + 1) / buf_capacity)
                    cursor_frac = max(0.0, min(1.0, (buf_current - bar_left + 0.5) / buf_capacity))

        # --- Layout: measure total width from right edge ---
        right_margin = 28
        total_w = 0
        gui_fps_text = f"GUI FPS {io.framerate:.0f}"
        gui_fps_w = imgui.calc_text_size(gui_fps_text)[0]
        total_w += gui_fps_w
        if video_fps_text:
            total_w += sep_w + imgui.calc_text_size(video_fps_text)[0]
        if tracker_fps_text:
            total_w += sep_w + imgui.calc_text_size(tracker_fps_text)[0]
        if has_buf:
            total_w += sep_w + buf_label_w + bar_w

        start_x = self.window_width - total_w - right_margin

        # --- Render left to right ---
        imgui.same_line(position=start_x)
        imgui.push_style_color(imgui.COLOR_TEXT, *dim_color)
        imgui.text(gui_fps_text)
        imgui.pop_style_color()

        if video_fps_text:
            imgui.same_line()
            imgui.push_style_color(imgui.COLOR_TEXT, *dim_color)
            imgui.text(sep + video_fps_text)
            imgui.pop_style_color()

        if tracker_fps_text:
            imgui.same_line()
            imgui.push_style_color(imgui.COLOR_TEXT, *tracker_fps_color)
            imgui.text(sep + tracker_fps_text)
            imgui.pop_style_color()

        if has_buf:
            imgui.same_line()
            imgui.push_style_color(imgui.COLOR_TEXT, *dim_color)
            imgui.text(sep + buf_label)
            imgui.pop_style_color()

            # Mini bar via draw_list
            imgui.same_line()
            bar_cursor = imgui.get_cursor_screen_pos()
            bar_h = imgui.get_text_line_height() - 2
            bar_y = bar_cursor[1] + 1
            bx = bar_cursor[0]
            draw_list = imgui.get_window_draw_list()

            bg_color = imgui.get_color_u32_rgba(0.25, 0.25, 0.28, 1.0)
            draw_list.add_rect_filled(bx, bar_y, bx + bar_w, bar_y + bar_h, bg_color, 3.0)

            gx0 = bx + bar_w * green_start_frac
            gx1 = bx + bar_w * green_end_frac
            if gx1 - gx0 > 0.5:
                fill_color = imgui.get_color_u32_rgba(0.2, 0.7, 0.3, 0.8)
                draw_list.add_rect_filled(gx0, bar_y, gx1, bar_y + bar_h, fill_color, 3.0)

            mx = bx + bar_w * cursor_frac
            marker_color = imgui.get_color_u32_rgba(*CurrentTheme.HIGHLIGHT_OVERLAY)
            draw_list.add_line(mx, bar_y, mx, bar_y + bar_h, marker_color, 1.5)

            imgui.dummy(bar_w, bar_h)

            if imgui.is_item_hovered():
                if buf_size > 0:
                    buf_pct = int(100 * (buf_current - (buf_end - buf_capacity + 1)) / buf_capacity) if buf_capacity > 0 else 0
                    imgui.set_tooltip(
                        f"Frame Buffer: {buf_size}/{buf_capacity}\n"
                        f"Range: {buf_start}-{buf_end}\n"
                        f"Current Frame: {buf_current}\n"
                        f"Position: {buf_pct}%"
                    )
                else:
                    imgui.set_tooltip(f"Frame Buffer: empty ({buf_capacity} capacity)")

    def _render_go_to_frame_popup(self):
        """Render the Go to Frame popup (Ctrl+G)."""
        if not self._go_to_frame_open:
            return

        from application.utils.video_segment import VideoSegment

        # Center the popup
        main_vp = imgui.get_main_viewport()
        cx = main_vp.pos[0] + main_vp.size[0] * 0.5
        cy = main_vp.pos[1] + main_vp.size[1] * 0.4
        imgui.set_next_window_position(cx, cy, imgui.APPEARING, 0.5, 0.5)
        imgui.set_next_window_size(320, 0)

        flags = imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SAVED_SETTINGS
        _, self._go_to_frame_open = imgui.begin("FunGen: Go to Frame##GoToFrame", closable=True, flags=flags)

        if not self._go_to_frame_open:
            imgui.end()
            return

        proc = self.app.processor
        fps = proc.fps if proc and proc.fps > 0 else 30.0

        imgui.text("Frame number or timecode (MM:SS.ms):")
        imgui.spacing()

        # Auto-focus input on first frame
        if self._go_to_frame_focus:
            imgui.set_keyboard_focus_here()
            self._go_to_frame_focus = False

        enter_pressed = False
        changed, self._go_to_frame_input = imgui.input_text(
            "##go_to_frame_input", self._go_to_frame_input, 64,
            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
        )
        if changed:
            enter_pressed = True

        imgui.same_line()
        if imgui.button("Go"):
            enter_pressed = True

        # Live preview: show what the input resolves to
        input_text = self._go_to_frame_input.strip()
        if input_text and proc and proc.video_info:
            resolved = VideoSegment.parse_time_input_to_frames(input_text, fps)
            if resolved >= 0:
                total = proc.total_frames
                clamped = max(0, min(resolved, total - 1 if total > 0 else 0))
                timecode = VideoSegment._frames_to_timecode(clamped, fps)
                imgui.text_disabled(f"-> Frame {clamped}  |  {timecode}")
            else:
                imgui.text_colored("Invalid input", *CurrentTheme.RED_LIGHT)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Show current position
        if proc and proc.video_info:
            total = proc.total_frames
            current = proc.current_frame_index
            cur_tc = VideoSegment._frames_to_timecode(current, fps)
            total_tc = VideoSegment._frames_to_timecode(total - 1 if total > 0 else 0, fps)
            imgui.text_disabled(f"Current: {current} ({cur_tc})")
            imgui.text_disabled(f"Total:   {total} ({total_tc})")

        if enter_pressed and input_text:
            if proc and proc.video_info:
                target = VideoSegment.parse_time_input_to_frames(input_text, fps)
                if target >= 0:
                    total = proc.total_frames
                    target = max(0, min(target, total - 1 if total > 0 else 0))
                    self.app.event_handlers.seek_video_with_sync(target)
                    self._go_to_frame_open = False

        # Escape to close
        if imgui.is_key_pressed(imgui.KEY_ESCAPE):
            self._go_to_frame_open = False

        imgui.end()

    def render_gui(self):
        self.component_render_times.clear()

        # _feat_supporter resolved in __init__; session-stable.
        # Cache visible extra timeline numbers (avoids per-frame getattr + f-string)
        if self._feat_supporter:
            app_state = self.app.app_state_ui
            self._visible_extra_timelines = [
                t for t in EXTRA_TIMELINE_RANGE
                if getattr(app_state, f"show_funscript_interactive_timeline{t}", False)
            ]
        else:
            self._visible_extra_timelines = []

        # One-shot metadata nudge: on new project, focus the metadata tab
        # once so the user is reminded to fill creator/tags. Cleared after.
        pm = getattr(self.app, 'project_manager', None)
        if pm is not None and getattr(pm, 'nudge_metadata', False):
            try:
                if hasattr(self, 'control_panel_ui'):
                    self.control_panel_ui._active_section = "metadata"
            except Exception:
                pass
            pm.nudge_metadata = False

        # Energy detection can be done before new_frame
        self._time_render("EnergyDetection", self._handle_energy_saver_interaction_detection)

        self._time_render("StageProcessorEvents", self.app.stage_processor.process_gui_events)

        # --- Process preview results queue every frame ---
        self._process_preview_results()

        imgui.new_frame()

        # Launch wizard on demand (menu trigger)
        if self._show_setup_wizard and self._first_run_wizard is None:
            self._first_run_wizard = FirstRunWizard(self.app)
            self._show_setup_wizard = False

        # First-run wizard, full-window overlay, skips all other UI
        if self._first_run_wizard is not None:
            font_scale = self.app.app_settings.config.ui.global_font_scale
            imgui.get_io().font_global_scale = font_scale
            wizard_done = self._first_run_wizard.render()
            if wizard_done:
                self._first_run_wizard = None
            self.perf_frame_count += 1
            self._time_render("ImGuiRender", imgui.render)
            if self.impl:
                draw_data = imgui.get_draw_data()
                self.impl.render(draw_data)
            return

        # IMPORTANT: Global shortcuts must be called AFTER new_frame() because
        # imgui.is_key_pressed() relies on KeysDownDuration which is updated by new_frame().
        # Previously shortcuts were called before new_frame, causing is_key_pressed to always return False.
        self._time_render("GlobalShortcuts", self._handle_global_shortcuts)

        if self.app.shortcut_manager.is_recording_shortcut_for:
            self._time_render("ShortcutRecordingInput", self.app.shortcut_manager.handle_shortcut_recording_input)
            self.app.energy_saver.reset_activity_timer()

        # Keep video texture fresh even when normal video panel is skipped
        self.video_display_ui.update_frame_texture_if_needed()

        main_viewport = imgui.get_main_viewport()
        self.window_width, self.window_height = main_viewport.size
        app_state = self.app.app_state_ui
        app_state.window_width = int(self.window_width)
        app_state.window_height = int(self.window_height)

        self._time_render("MainMenu", self.main_menu.render)

        # Render toolbar
        self._time_render("Toolbar", self.toolbar_ui.render)

        font_scale = self.app.app_settings.config.ui.global_font_scale
        imgui.get_io().font_global_scale = font_scale

        if hasattr(app_state, 'main_menu_bar_height_from_menu_class'):
            self.main_menu_bar_height = app_state.main_menu_bar_height_from_menu_class
        else:
            self.main_menu_bar_height = imgui.get_frame_height_with_spacing() if self.main_menu else 0

        # Account for toolbar height (includes section labels) - only if shown
        if not hasattr(app_state, 'show_toolbar'):
            app_state.show_toolbar = True
        toolbar_height = self.toolbar_ui.get_toolbar_height() if app_state.show_toolbar else 0

        app_state.update_current_script_display_values()

        status_strip_h = _STATUS_STRIP_HEIGHT

        if app_state.ui_layout_mode == 'fixed':
            self._render_fixed_layout(app_state, font_scale, toolbar_height, status_strip_h)
        else:
            self._render_floating_layout(app_state)

        self._render_dialogs_and_overlays(app_state)

        # Render status strip at bottom of window
        self._render_status_strip(status_strip_h)

        # Render toast notifications (foreground, aligned to top of content area)
        self.notification_manager.render(top_y_offset=self.main_menu_bar_height + toolbar_height)

        self.perf_frame_count += 1
        if time.time() - self.last_perf_log_time > self.perf_log_interval:
            self._log_performance()
        
        # Continuously update frontend queue with current data
        self._update_frontend_perf_queue()

        # Track final rendering operations
        self._time_render("ImGuiRender", imgui.render)
        if self.impl:
            # Only measure OpenGL render time if it's likely to be significant
            # Skip timing for very simple frames to reduce overhead
            draw_data = imgui.get_draw_data()
            if draw_data.total_vtx_count > 100 or draw_data.cmd_lists_count > 5:
                self._time_render("OpenGLRender", self.impl.render, draw_data)
            else:
                # Simple frame - render without timing overhead
                self.impl.render(draw_data)
                self.component_render_times["OpenGLRender"] = 0.0

    def _update_frontend_perf_queue(self):
        """
        Updates the frontend performance queue with the current accumulated times and frame count.
        Only updates if there is valid data to prevent empty entries.
        """
        # Only update queue if we have valid data
        if self.perf_accumulated_times and self.perf_frame_count > 0:
            current_perf_data = {
                'accumulated_times': self.perf_accumulated_times.copy(),
                'frame_count': self.perf_frame_count,
                'timestamp': time.time()
            }
            self._frontend_perf_queue.append(current_perf_data)

    def _install_char_input_fallback(self):
        """Attach a GLFW char callback that supplements pyimgui's key-event
        path with character-event-driven shortcut dispatch.

        Some devices (BLE HID keyboards, keymacs with remapped profiles, the
        Contour Shuttle Pro V2 / DaVinci Speed Editor variants the community
        asked about) reach the OS only through WM_CHAR / text-input events
        and never raise key events. pyimgui's GlfwRenderer therefore sees
        `io.keys_down[KEY_F]` stay False forever, so `imgui.is_key_pressed`
        never fires and single-letter shortcuts ("F", Space, digit-shortcuts)
        look dead to the user while Ctrl-combos (which DO come through as
        key events on those devices, because modifiers travel a different
        HID path) keep working. SDL2 handles this transparently; GLFW does
        not, which is why the same devices work in OFS and fail here.

        We (1) forward to the renderer's own char handler so text-input
        widgets keep receiving typed characters, then (2) when no text
        input is focused, translate the character to a GLFW keycode via
        the shortcut manager's reverse map and stash it in
        `_virtual_pressed_keys` for one frame. The shortcut dispatcher
        consults that set as a fallback for modifier-less shortcuts.
        """
        if self.impl is None or self.window is None:
            return
        # Grab pyimgui's bound char_callback directly. glfw.set_char_callback
        # returns the previously-installed Python callable on most builds,
        # but taking the method explicitly is robust across glfw-python
        # versions and avoids subtle CFUNCTYPE lifetime issues if GLFW ever
        # hands us back the C wrapper instead of the Python callable.
        self._prev_char_callback = getattr(self.impl, 'char_callback', None)
        try:
            glfw.set_char_callback(self.window, self._on_char_input)
        except Exception as e:
            self.app.logger.debug(f"char-callback install failed: {e}")

    def _on_char_input(self, window, codepoint):
        """Char callback installed by `_install_char_input_fallback`.

        Runs during `glfw.poll_events()`. Forwards first so pyimgui keeps
        feeding typed characters into text inputs, then (if no text widget
        wants them) marks the corresponding keycode as virtually-pressed
        for the shortcut dispatcher to pick up this same frame.
        """
        # 1. Forward to pyimgui so text inputs still receive characters.
        #    The renderer's own callback just calls io.add_input_character.
        prev = self._prev_char_callback
        if prev is not None:
            try:
                prev(window, codepoint)
            except Exception:
                # Fall through to the inline equivalent; we must not let a
                # broken forward drop text input for the user entirely.
                try:
                    io = imgui.get_io()
                    if 0 < codepoint < 0x10000:
                        io.add_input_character(codepoint)
                except Exception:
                    pass
        else:
            try:
                io = imgui.get_io()
                if 0 < codepoint < 0x10000:
                    io.add_input_character(codepoint)
            except Exception:
                pass

        # 2. Shortcut fallback. Skip if a text input is focused — in that
        #    case the character belongs to the user's typing, not to a
        #    global shortcut.
        try:
            io = imgui.get_io()
            if io.want_text_input:
                return
            if codepoint < 32 or codepoint > 126:
                return
            char = chr(codepoint).upper()
            # Space char has no entry in the reverse map; use the named alias.
            lookup = 'SPACE' if char == ' ' else char
            sm = self.app.shortcut_manager
            if sm is None:
                return
            key_code = sm.name_to_glfw_key(lookup)
            if key_code is None:
                return
            self._virtual_pressed_keys.add(key_code)
        except Exception:
            # Never let a callback exception bubble through GLFW — it will
            # segfault the process on some platforms.
            pass

    def run(self):
        colors = self.colors
        if not self.init_glfw(): return
        target_normal_fps = self.app.energy_saver.main_loop_normal_fps_target
        target_energy_fps = self.app.energy_saver.energy_saver_fps
        if target_normal_fps <= 0: target_normal_fps = 60
        if target_energy_fps <= 0: target_energy_fps = 1
        if target_energy_fps > target_normal_fps: target_energy_fps = target_normal_fps
        target_frame_duration_normal = 1.0 / target_normal_fps
        target_frame_duration_energy_saver = 1.0 / target_energy_fps
        # Passive throttle between active and energy-saver states: when
        # nothing animates (paused, no user input for ~1.5s, no tracker)
        # drop to this rate. Still responsive (any event wakes us via
        # glfw.wait_events_timeout), and cuts idle CPU dramatically vs
        # running the imgui loop at 60 Hz.
        target_passive_fps = 15
        target_frame_duration_passive = 1.0 / target_passive_fps
        passive_idle_seconds = 1.5
        glfw.swap_interval(0)

        try:
            while not glfw.window_should_close(self.window):
                frame_start_time = time.time()
                
                # Track frame setup operations
                event_start = time.perf_counter()
                # Active = playing / tracking / recent user activity. Anything
                # else blocks on wait_events_timeout so we don't burn CPU
                # redrawing a static imgui UI at 60 Hz.
                _idle_for = time.time() - self.app.energy_saver.last_activity_time
                _is_active = (
                    _idle_for < passive_idle_seconds
                    or (self.app.processor and getattr(self.app.processor,
                                                        'is_processing', False))
                    or self.app.stage_processor.full_analysis_active
                )
                if _is_active:
                    glfw.poll_events()
                else:
                    # Timeout equals the passive frame budget, so we still
                    # tick ~15 Hz for time-based updates (autosave check,
                    # toast expiry, etc.) even with zero input events.
                    glfw.wait_events_timeout(target_frame_duration_passive)
                if self.impl: self.impl.process_inputs()
                gl.glClearColor(*colors.BACKGROUND_CLEAR)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                event_time = (time.perf_counter() - event_start) * 1000
                self.component_render_times["FrameSetup"] = event_time

                # Detect external mpv-fullscreen exit (user closed the window
                # via the X or pressed q inside mpv). Without this poll, the
                # bridge stays "active" in app state and FunGen's controls
                # remain disabled even though the window is gone.
                mpv_ctl = getattr(self.app, '_mpv_controller', None)
                if mpv_ctl is not None:
                    mpv_ctl.poll_external_exit()
                
                # GUI rendering (internally timed)
                self.render_gui()
                _as_cfg = self.app.app_settings.config.autosave
                if (
                    _as_cfg.enabled
                    and time.time() - self.app.project_manager.last_autosave_time > _as_cfg.interval_seconds
                ):
                    self.app.project_manager.perform_autosave()
                self.app.energy_saver.check_and_update_energy_saver()
                
                # Track buffer swap (GPU synchronization)
                swap_start = time.perf_counter()
                glfw.swap_buffers(self.window)
                if self.mpv_display is not None:
                    try:
                        _rs = getattr(self.mpv_display, "report_swap", None)
                        if _rs is not None:
                            _rs()
                    except Exception:
                        pass
                _tq = getattr(self, 'thumbnail_mpv_queue', None)
                if _tq is not None:
                    try:
                        _tq.tick()
                    except Exception:
                        pass
                swap_time = (time.perf_counter() - swap_start) * 1000
                self.component_render_times["BufferSwap"] = swap_time
                
                # Update GPU memory usage every 120 frames (~2s at 60fps) - reduced frequency  
                if self.perf_frame_count % 120 == 0:
                    gpu_start = time.perf_counter()
                    self.update_gpu_memory_usage()
                    gpu_time = (time.perf_counter() - gpu_start) * 1000
                    # Only track if it's actually expensive (>1ms)
                    if gpu_time > 1.0:
                        self.component_render_times["GPU_Monitor"] = gpu_time
                if self.app.energy_saver.energy_saver_active:
                    current_target_duration = target_frame_duration_energy_saver
                elif not _is_active:
                    # Passive idle: wait_events_timeout already slept the
                    # budget, so don't double-sleep afterwards.
                    current_target_duration = 0.0
                else:
                    current_target_duration = target_frame_duration_normal
                elapsed_time_for_frame = time.time() - frame_start_time
                sleep_duration = current_target_duration - elapsed_time_for_frame
                
                _updater_cfg = self.app.app_settings.config.updater
                if ( # Periodic update checks
                    _updater_cfg.check_on_startup
                    and _updater_cfg.check_periodically
                    and time.time() - self.app.updater.last_check_time > 3600  # 1 hour
                ):
                    self.app.updater.check_for_updates_async()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

                # Profile-mode: when the requested frame count is reached,
                # dump timings and exit.
                if (self._profile_enabled
                        and self.perf_frame_count >= self._profile_target_frames):
                    self._dump_profile_snapshot()
                    glfw.set_window_should_close(self.window, True)
        finally:
            # Safety net: dump whatever we collected even if the user closed
            # the window before hitting the target frame count.
            if self._profile_enabled and self._profile_samples:
                try:
                    self._dump_profile_snapshot()
                except Exception:
                    pass
            self.app.shutdown_app()

            # --- Cleanly shut down all worker threads ---
            self.shutdown_event.set()
            
            # Shutdown ProcessingThreadManager first

    def cleanup(self):
        try:
            # Stop native sync servers if running (managed by control panel now)
            if hasattr(self.control_panel_ui, '_native_sync_manager'):
                try:
                    self.control_panel_ui._native_sync_manager.stop()
                except Exception as e:
                    self.app.logger.error(f"Error stopping native sync: {e}")

            self.app.logger.info("Shutting down ProcessingThreadManager...")
            self.processing_thread_manager.shutdown(timeout=3.0)

            # Shutdown legacy preview worker thread
            for _ in self.preview_worker_threads:
                try:
                    self.preview_task_queue.put_nowait({'type': 'shutdown'})
                except queue.Full:
                    pass
            for t in self.preview_worker_threads:
                t.join(timeout=5.0)

            if self.frame_texture_id: gl.glDeleteTextures([self.frame_texture_id]); self.frame_texture_id = 0
            if self.heatmap_texture_id: gl.glDeleteTextures([self.heatmap_texture_id]); self.heatmap_texture_id = 0
            if self.funscript_preview_texture_id: gl.glDeleteTextures(
                [self.funscript_preview_texture_id]); self.funscript_preview_texture_id = 0

            if self.impl: self.impl.shutdown()
            if self.window: glfw.destroy_window(self.window)
            glfw.terminate()
            self.app.logger.info("GUI terminated.", extra={'status_message': False})
        except Exception as e:
            print(f"Error during cleanup: {e}")
            