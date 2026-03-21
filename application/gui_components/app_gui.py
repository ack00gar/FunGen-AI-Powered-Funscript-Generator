import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import numpy as np
import cv2
import time
import threading
import queue
import os
import platform
from typing import List, Dict
from collections import deque

from config import constants, element_group_colors
from application.classes import GaugeWindow, ImGuiFileDialog, InteractiveFunscriptTimeline, LRDialWindow, MainMenu, Simulator3DWindow
from application.gui_components import ControlPanelUI, VideoDisplayUI, VideoNavigationUI, ChapterListWindow, InfoGraphsUI, GeneratedFileManagerWindow, KeyboardShortcutsDialog, ToolbarUI, ChapterTypeManagerUI
from application.gui_components.bookmark_list_window import BookmarkListWindow
from application.utils import _format_time, ProcessingThreadManager, TaskType, TaskPriority, get_icon_texture_manager
from application.utils.feature_detection import is_feature_available as _is_feature_available
from application.utils.timeline_constants import EXTRA_TIMELINE_RANGE
from application.utils.timeline_modes import TimelineMode
from application.gui_components.gui_preview_manager import PreviewManagerMixin
from application.gui_components.gui_shortcut_handler import ShortcutHandlerMixin
from application.gui_components.gui_dialog_renderer import DialogRendererMixin
from application.utils.notifications import NotificationManager
from application.gui_components.first_run_wizard import FirstRunWizard

_STATUS_STRIP_HEIGHT = 22
_HINT_ROTATION_INTERVAL_S = 10.0
_FIXED_PANEL_BASE_WIDTH = 450  # Base width for control panel and info graphs (scaled by font_scale)
_VIDEO_NAV_BAR_HEIGHT = 150


class GUI(DialogRendererMixin, ShortcutHandlerMixin, PreviewManagerMixin):
    def __init__(self, app_logic):
        self.app = app = app_logic
        self.window = None
        self.impl = None
        self.window_width = app.app_settings.get("window_width", 1800)
        self.window_height = app.app_settings.get("window_height", 1000)
        self.main_menu_bar_height = 0

        self.constants = constants
        self.colors = element_group_colors.AppGUIColors

        self.frame_texture_id = 0
        self.heatmap_texture_id = 0
        self.funscript_preview_texture_id = 0
        self.enhanced_preview_texture_id = 0  # Dedicated texture for enhanced preview tooltips

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
        self.gauge_window_ui_t1 = GaugeWindow(app, timeline_num=1)
        self.gauge_window_ui_t2 = GaugeWindow(app, timeline_num=2)
        self.movement_bar_ui = LRDialWindow(app)  # Movement Bar (backward compatible name)
        self.simulator_3d_window_ui = Simulator3DWindow(app)

        self.timeline_editor1 = InteractiveFunscriptTimeline(app_instance=app, timeline_num=1)
        self.timeline_editor2 = InteractiveFunscriptTimeline(app_instance=app, timeline_num=2)

        # Extra timelines (supporter feature, created lazily)
        self._extra_timeline_editors: Dict[int, InteractiveFunscriptTimeline] = {}

        # Modularized UI Panel Components
        self.control_panel_ui = ControlPanelUI(app)
        self.video_display_ui = VideoDisplayUI(app, self)  # Pass self for texture updates
        self.video_navigation_ui = VideoNavigationUI(app, self)  # Pass self for texture methods
        self.info_graphs_ui = InfoGraphsUI(app)
        self.chapter_list_window_ui = ChapterListWindow(app, nav_ui=self.video_navigation_ui)
        self.chapter_type_manager_ui = ChapterTypeManagerUI(app)
        self.bookmark_list_window_ui = BookmarkListWindow(app, gui=self)
        self.generated_file_manager_ui = GeneratedFileManagerWindow(app)
        self.keyboard_shortcuts_dialog = KeyboardShortcutsDialog(app)

        # First-run wizard (full-window overlay, replaces old popup)
        self._first_run_wizard = (
            FirstRunWizard(app)
            if app.app_settings.is_first_run and not app.is_cli_mode
            else None
        )
        # Flag for menu-triggered wizard re-launch
        self._show_setup_wizard = False

        # UI state for the dialog's radio buttons
        self.selected_batch_method_idx_ui = 0
        self.batch_overwrite_mode_ui = 0  # 0: Process All, 1: Skip Existing
        self.batch_apply_post_processing_ui = True
        self.batch_copy_funscript_to_video_location_ui = True
        self.batch_generate_roll_file_ui = True
        self.batch_apply_ultimate_autotune_ui = True
        self.batch_adaptive_tuning_ui: bool = True

        self.control_panel_ui.timeline_editor1 = self.timeline_editor1
        self.control_panel_ui.timeline_editor2 = self.timeline_editor2

        self.last_preview_update_time_timeline = 0.0
        self.last_preview_update_time_heatmap = 0.0
        self.preview_update_interval_seconds = constants.UI_PREVIEW_UPDATE_INTERVAL_S

        self.last_mouse_pos_for_energy_saver = (0, 0)
        self.app.energy_saver.reset_activity_timer()
        
        # Simple arrow key navigation state
        current_time = time.time()
        self.arrow_key_state = {
            'last_seek_time': current_time,
            'seek_interval': 0.033,  # Will be updated based on video FPS
            'initial_press_time': current_time,
            'continuous_delay': 0.2,  # 200ms delay before continuous playback (allows frame-by-frame taps)
            'last_direction': 0,  # Track direction to prevent double-navigation
            'playback_active': False,  # True when hold-forward has engaged the processing loop
            'continuous_start_time': 0,  # When continuous (accelerating) mode began
        }
        self.arrow_nav_reading_fps = 0.0    # instantaneous frames/sec during arrow key seeking
        self._reading_fps_frames = deque()  # (timestamp, frame_count) per tick
        self._reading_fps_display = 0.0   # value shown in status bar (updated once/sec)
        self._reading_fps_last_update = 0.0  # last time display value was refreshed

        self.batch_videos_data: List[Dict] = []
        self.batch_overwrite_mode_ui: int = 0  # 0: Skip own, 1: Skip any, 2: Overwrite all
        self.batch_set_all_format_idx: int = 1  # Default to "2D" in the set-all dropdown
        self.batch_processing_method_idx_ui: int = 0
        self.batch_copy_funscript_to_video_location_ui: bool = True
        self.batch_generate_roll_file_ui: bool = True
        self.batch_apply_ultimate_autotune_ui: bool = True
        self.batch_adaptive_tuning_ui: bool = True
        self.batch_save_preprocessed_video_ui: bool = False
        self.last_overwrite_mode_ui: int = -1 # Used to trigger auto-selection logic

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
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                # Get memory usage from first GPU
                gpu = gpus[0]
                self.gpu_memory_usage = gpu.memoryUsed / gpu.memoryTotal * 100
                self.component_render_times["GPU_MemoryUsage"] = self.gpu_memory_usage
            else:
                self.gpu_memory_usage = 0
        except (ImportError, Exception):
            # GPUtil not available or GPU not accessible
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

        # Set window icon (macOS doesn't support window icons in GLFW, skip on macOS)
        try:
            import platform
            if platform.system() != "Darwin":  # Skip on macOS
                icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')
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
        self.impl = GlfwRenderer(self.window)
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
            # If timeline 1 is empty or has no AI-generated script, load the first funscript there.

            if not self.app.funscript_processor.get_actions('primary'):
                self.app.logger.info(f"Loading '{os.path.basename(funscript_files[0])}' into Timeline 1.")
                self.app.file_manager.load_funscript_to_timeline(funscript_files[0], timeline_num=1)

                if len(funscript_files) > 1:
                    self.app.logger.info(f"Loading '{os.path.basename(funscript_files[1])}' into Timeline 2.")
                    self.app.file_manager.load_funscript_to_timeline(funscript_files[1], timeline_num=2)
                    self.app.app_state_ui.show_funscript_interactive_timeline2 = True
            else:
                self.app.logger.info(f"Timeline 1 has data. Loading '{os.path.basename(funscript_files[0])}' into Timeline 2.")
                self.app.file_manager.load_funscript_to_timeline(funscript_files[0], timeline_num=2)
                self.app.app_state_ui.show_funscript_interactive_timeline2 = True

            # Mark previews as dirty to force a redraw
            self.app.app_state_ui.funscript_preview_dirty = True
            self.app.app_state_ui.heatmap_dirty = True


    def update_texture(self, texture_id: int, image: np.ndarray):
        if image is None or image.size == 0: return
        h, w = image.shape[:2]
        if w == 0 or h == 0: return

        # Ensure we have a valid texture ID
        if not gl.glIsTexture(texture_id):
            self.app.logger.error(f"Attempted to update an invalid texture ID: {texture_id}")
            return

        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

        # Cache last texture sizes to prefer glTexSubImage2D when dimensions unchanged
        if not hasattr(self, '_texture_sizes'):
            self._texture_sizes = {}

        last_size = self._texture_sizes.get(texture_id)

        # Determine format and upload
        if len(image.shape) == 2:
            internal_fmt = gl.GL_RED; fmt = gl.GL_RED; payload = image
        elif image.shape[2] == 3:
            internal_fmt = gl.GL_RGB; fmt = gl.GL_RGB; payload = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            internal_fmt = gl.GL_RGBA; fmt = gl.GL_RGBA; payload = image

        if last_size and last_size == (w, h, internal_fmt):
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h, fmt, gl.GL_UNSIGNED_BYTE, payload)
        else:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_fmt, w, h, 0, fmt, gl.GL_UNSIGNED_BYTE, payload)
            self._texture_sizes[texture_id] = (w, h, internal_fmt)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

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

        current_action_count = len(self.app.funscript_processor.get_actions('primary'))
        is_live_tracking = self.app.processor and self.app.processor.tracker and self.app.processor.tracker.tracking_active

        # Determine if a redraw is needed
        full_redraw_needed = (app_state.funscript_preview_dirty
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
            actions_copy = self.app.funscript_processor.get_actions('primary').copy()
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
                            
                            # Use much tighter tolerance to avoid frame/video mismatches
                            # With 10k frames, 0.0001 = ~1 frame tolerance instead of ~50 frames
                            position_tolerance = 0.0001  # ~0.01% tolerance for position stability
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
                            enhanced_preview_enabled = self.app.app_settings.get("enable_enhanced_funscript_preview", True)
                            
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
                marker_x = (canvas_p1_x + style.frame_padding[0]) + (normalized_pos * (current_bar_width_float - style.frame_padding[0] * 2))
                marker_color = imgui.get_color_u32_rgba(*colors.MARKER)
                draw_list_marker = imgui.get_window_draw_list()

                # Draw triangle
                triangle_p1 = (marker_x - 5, canvas_p1_y_offset)
                triangle_p2 = (marker_x + 5, canvas_p1_y_offset)
                triangle_p3 = (marker_x, canvas_p1_y_offset + 5)
                draw_list_marker.add_triangle_filled(triangle_p1[0], triangle_p1[1], triangle_p2[0], triangle_p2[1], triangle_p3[0], triangle_p3[1], marker_color)

                # Draw line
                draw_list_marker.add_line(marker_x, canvas_p1_y_offset, marker_x, canvas_p1_y_offset + graph_height, marker_color, 1.0)

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

        current_action_count = len(self.app.funscript_processor.get_actions('primary'))
        is_live_tracking = self.app.processor and self.app.processor.tracker and self.app.processor.tracker.tracking_active

        full_redraw_needed = (
            app_state.heatmap_dirty
            or current_bar_width_int != app_state.last_heatmap_bar_width
            or abs(total_video_duration_s - app_state.last_heatmap_video_duration_s) > 0.01)

        incremental_update_needed = current_action_count != self.last_submitted_action_count_heatmap

        needs_regen = full_redraw_needed or (incremental_update_needed and (not is_live_tracking or (time.time() - self.last_preview_update_time_heatmap >= self.preview_update_interval_seconds)))

        if needs_regen:
            actions_copy = self.app.funscript_processor.get_actions('primary').copy()
            task = {
                'type': 'heatmap',
                'target_width': current_bar_width_int,
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
    #     _handle_toggle_timeline2_shortcut, _handle_toggle_gauge_window_shortcut,
    #     _handle_toggle_3d_simulator_shortcut, _handle_toggle_movement_bar_shortcut,
    #     _handle_toggle_chapter_list_shortcut, _handle_toggle_heatmap_shortcut,
    #     _handle_toggle_funscript_preview_shortcut, _handle_toggle_video_feed_shortcut,
    #     _handle_toggle_waveform_shortcut, _handle_reset_timeline_view_shortcut,
    #     _handle_toggle_oscillation_area_mode, _handle_energy_saver_interaction_detection ---

    # ---- Shortcut hint helpers (status strip) ----

    def _format_shortcut_hint(self, key_str):
        """Platform-aware formatting: SUPER→Cmd/Ctrl, arrow symbols, etc."""
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

        # Timeline hovered with selection → editing hints
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

        # Timeline hovered, no selection → mode-specific or canvas interaction hints
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
        timeline1_render_h = app_state.timeline_base_height if app_state.show_funscript_interactive_timeline else 0
        timeline2_render_h = app_state.timeline_base_height if app_state.show_funscript_interactive_timeline2 else 0
        extra_timelines_total_height = len(self._visible_extra_timelines) * app_state.timeline_base_height
        interactive_timelines_total_height = timeline1_render_h + timeline2_render_h + extra_timelines_total_height
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
                video_panel_w = self.window_width - control_panel_w - graphs_panel_w
                if video_panel_w < 100:
                    video_panel_w = 100
                    graphs_panel_w = max(100, self.window_width - control_panel_w - video_panel_w)
                video_area_x_start = control_panel_w
                graphs_area_x_start = control_panel_w + video_panel_w
                app_state.fixed_layout_geometry['ControlPanel'] = {'pos': (0, panel_y_start), 'size': (control_panel_w, top_panels_h)}
                imgui.set_next_window_position(0, panel_y_start)
                imgui.set_next_window_size(control_panel_w, top_panels_h)
                self._time_render("ControlPanelUI", self.control_panel_ui.render)
                app_state.fixed_layout_geometry['VideoDisplay'] = {'pos': (video_area_x_start, panel_y_start), 'size': (video_panel_w, top_panels_h)}
                imgui.set_next_window_position(video_area_x_start, panel_y_start)
                imgui.set_next_window_size(video_panel_w, top_panels_h)
                self._time_render("VideoDisplayUI", self.video_display_ui.render)
                app_state.fixed_layout_geometry['InfoGraphs'] = {'pos': (graphs_area_x_start, panel_y_start), 'size': (graphs_panel_w, top_panels_h)}
                imgui.set_next_window_position(graphs_area_x_start, panel_y_start)
                imgui.set_next_window_size(graphs_panel_w, top_panels_h)
                self._time_render("InfoGraphsUI", self.info_graphs_ui.render)
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
                video_panel_w = self.window_width - control_panel_w - graphs_panel_w
                if video_panel_w < 100:
                    video_panel_w = 100
                    graphs_panel_w = max(100, self.window_width - control_panel_w - video_panel_w)
                video_render_h = max(50, available_height_for_main_panels - video_nav_bar_h)
                video_area_x_start = control_panel_w
                graphs_area_x_start = control_panel_w + video_panel_w
                app_state.fixed_layout_geometry['ControlPanel'] = {'pos': (0, panel_y_start), 'size': (control_panel_w, available_height_for_main_panels)}
                imgui.set_next_window_position(0, panel_y_start)
                imgui.set_next_window_size(control_panel_w, available_height_for_main_panels)
                self._time_render("ControlPanelUI", self.control_panel_ui.render)
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
                app_state.fixed_layout_geometry['InfoGraphs'] = {'pos': (graphs_area_x_start, panel_y_start), 'size': (graphs_panel_w, available_height_for_main_panels)}
                imgui.set_next_window_position(graphs_area_x_start, panel_y_start)
                imgui.set_next_window_size(graphs_panel_w, available_height_for_main_panels)
                self._time_render("InfoGraphsUI", self.info_graphs_ui.render)
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
                if app_state.show_funscript_interactive_timeline:
                    self._time_render("TimelineEditor1", self.timeline_editor1.render,
                                      0, per_tl_h, view_mode=app_state.ui_view_mode, container_mode=True)
                if app_state.show_funscript_interactive_timeline2:
                    self._time_render("TimelineEditor2", self.timeline_editor2.render,
                                      0, per_tl_h, view_mode=app_state.ui_view_mode, container_mode=True)
                for t_num in self._visible_extra_timelines:
                    editor = self._get_or_create_timeline_editor(t_num)
                    self._time_render(f"TimelineEditor{t_num}", editor.render,
                                      0, per_tl_h, view_mode=app_state.ui_view_mode, container_mode=True)
            imgui.end()
            app_state.fixed_layout_geometry['TimelineContainer'] = {
                'pos': (0, timeline_area_y), 'size': (self.window_width, capped_timelines_h)}
        else:
            timeline_current_y_start = timeline_area_y
            if app_state.show_funscript_interactive_timeline:
                app_state.fixed_layout_geometry['Timeline1'] = {'pos': (0, timeline_current_y_start), 'size': (self.window_width, timeline1_render_h)}
                self._time_render("TimelineEditor1", self.timeline_editor1.render, timeline_current_y_start, timeline1_render_h, view_mode=app_state.ui_view_mode)
                timeline_current_y_start += timeline1_render_h
            if app_state.show_funscript_interactive_timeline2:
                app_state.fixed_layout_geometry['Timeline2'] = {'pos': (0, timeline_current_y_start), 'size': (self.window_width, timeline2_render_h)}
                self._time_render("TimelineEditor2", self.timeline_editor2.render, timeline_current_y_start, timeline2_render_h, view_mode=app_state.ui_view_mode)
                timeline_current_y_start += timeline2_render_h
            for t_num in self._visible_extra_timelines:
                editor = self._get_or_create_timeline_editor(t_num)
                extra_h = per_tl_h
                app_state.fixed_layout_geometry[f'Timeline{t_num}'] = {'pos': (0, timeline_current_y_start), 'size': (self.window_width, extra_h)}
                self._time_render(f"TimelineEditor{t_num}", editor.render, timeline_current_y_start, extra_h, view_mode=app_state.ui_view_mode)
                timeline_current_y_start += extra_h

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

        if self.app.app_state_ui.show_generated_file_manager:
            self._time_render("GeneratedFileManager", self.generated_file_manager_ui.render)

        if hasattr(app_state, 'show_ai_models_dialog') and app_state.show_ai_models_dialog:
            self._time_render("AIModelsDialog", self._render_ai_models_dialog)

        self._render_go_to_frame_popup()
        self.app.addon_checker.tick_status_ads()

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
        _has_status_msg = app_state.status_message and time.time() < app_state.status_message_time

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
            msg = app_state.status_message
            center_x = self.window_width * 0.5
            text_w = imgui.calc_text_size(msg)[0]
            imgui.same_line(position=center_x - text_w * 0.5)
            imgui.push_style_color(imgui.COLOR_TEXT, *colors.ACCENT)
            imgui.text(msg)
            imgui.pop_style_color()
        else:
            # Clear expired messages
            if app_state.status_message and time.time() >= app_state.status_message_time:
                app_state.status_message = ""
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

            # Build display string: "Key Action · Key Action · Key Action"
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
        # Cache separator width (constant string, font doesn't change mid-session)
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
            marker_color = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.9)
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
        _, self._go_to_frame_open = imgui.begin("Go to Frame##GoToFrame", closable=True, flags=flags)

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
                imgui.text_colored("Invalid input", 1.0, 0.4, 0.4, 1.0)

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

        # Cache feature detection flags for this frame
        self._feat_supporter = _is_feature_available("patreon_features")

        # Cache visible extra timeline numbers (avoids per-frame getattr + f-string)
        if self._feat_supporter:
            app_state = self.app.app_state_ui
            self._visible_extra_timelines = [
                t for t in EXTRA_TIMELINE_RANGE
                if getattr(app_state, f"show_funscript_interactive_timeline{t}", False)
            ]
        else:
            self._visible_extra_timelines = []

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

        # First-run wizard — full-window overlay, skips all other UI
        if self._first_run_wizard is not None:
            font_scale = self.app.app_settings.get("global_font_scale", 1.0)
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

        font_scale = self.app.app_settings.get("global_font_scale", 1.0)
        imgui.get_io().font_global_scale = font_scale

        if hasattr(app_state, 'main_menu_bar_height_from_menu_class'):
            self.main_menu_bar_height = app_state.main_menu_bar_height_from_menu_class
        else:
            self.main_menu_bar_height = imgui.get_frame_height_with_spacing() if self.main_menu else 0

        # Account for toolbar height (includes section labels) - only if shown
        if not hasattr(app_state, 'show_toolbar'):
            app_state.show_toolbar = True
        toolbar_height = self.toolbar_ui.get_toolbar_height() if app_state.show_toolbar else 0

        if not app_state.gauge_pos_initialized and self.main_menu_bar_height > 0:
            app_state.initialize_gauge_default_y(self.main_menu_bar_height + toolbar_height)

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
        glfw.swap_interval(0)

        try:
            while not glfw.window_should_close(self.window):
                frame_start_time = time.time()
                
                # Track frame setup operations
                event_start = time.perf_counter()
                glfw.poll_events()
                if self.impl: self.impl.process_inputs()
                gl.glClearColor(*colors.BACKGROUND_CLEAR)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                event_time = (time.perf_counter() - event_start) * 1000
                self.component_render_times["FrameSetup"] = event_time
                
                # GUI rendering (internally timed)
                self.render_gui()
                if (
                    self.app.app_settings.get("autosave_enabled", True)
                    and time.time() - self.app.project_manager.last_autosave_time > self.app.app_settings.get("autosave_interval_seconds", constants.DEFAULT_AUTOSAVE_INTERVAL_SECONDS)
                ):
                    self.app.project_manager.perform_autosave()
                self.app.energy_saver.check_and_update_energy_saver()
                
                # Track buffer swap (GPU synchronization)
                swap_start = time.perf_counter()
                glfw.swap_buffers(self.window)
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
                current_target_duration = target_frame_duration_energy_saver if self.app.energy_saver.energy_saver_active else target_frame_duration_normal
                elapsed_time_for_frame = time.time() - frame_start_time
                sleep_duration = current_target_duration - elapsed_time_for_frame
                
                if ( # Periodic update checks
                    self.app.app_settings.get("updater_check_on_startup", True)
                    and self.app.app_settings.get("updater_check_periodically", True)
                    and time.time() - self.app.updater.last_check_time > 3600  # 1 hour
                ):
                    self.app.updater.check_for_updates_async()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
        finally:
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
            