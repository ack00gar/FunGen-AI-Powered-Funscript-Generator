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
from typing import List, Dict
from collections import deque

from config import constants, element_group_colors
from application.classes import GaugeWindow, ImGuiFileDialog, InteractiveFunscriptTimeline, LRDialWindow, MainMenu
from application.gui_components import ControlPanelUI, VideoDisplayUI, VideoNavigationUI, ChapterListWindow, InfoGraphsUI, GeneratedFileManagerWindow, AutotunerWindow
from application.utils import _format_time, ProcessingThreadManager, TaskType, TaskPriority


class GUI:
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

        # Standard Components (owned by GUI)
        self.file_dialog = ImGuiFileDialog(app_logic_instance=app)
        self.main_menu = MainMenu(app)
        self.gauge_window_ui_t1 = GaugeWindow(app, timeline_num=1)
        self.gauge_window_ui_t2 = GaugeWindow(app, timeline_num=2)
        self.lr_dial_window_ui = LRDialWindow(app)

        self.timeline_editor1 = InteractiveFunscriptTimeline(app_instance=app, timeline_num=1)
        self.timeline_editor2 = InteractiveFunscriptTimeline(app_instance=app, timeline_num=2)

        # Modularized UI Panel Components
        self.control_panel_ui = ControlPanelUI(app)
        self.video_display_ui = VideoDisplayUI(app, self)  # Pass self for texture updates
        self.video_navigation_ui = VideoNavigationUI(app, self)  # Pass self for texture methods
        self.info_graphs_ui = InfoGraphsUI(app)
        self.chapter_list_window_ui = ChapterListWindow(app, nav_ui=self.video_navigation_ui)
        self.generated_file_manager_ui = GeneratedFileManagerWindow(app)
        self.autotuner_window_ui = AutotunerWindow(app)

        # UI state for the dialog's radio buttons
        self.selected_batch_method_idx_ui = 0
        self.batch_overwrite_mode_ui = 0  # 0: Process All, 1: Skip Existing
        self.batch_apply_post_processing_ui = True
        self.batch_copy_funscript_to_video_location_ui = True
        self.batch_generate_roll_file_ui = True
        self.batch_apply_ultimate_autotune_ui = True

        self.control_panel_ui.timeline_editor1 = self.timeline_editor1
        self.control_panel_ui.timeline_editor2 = self.timeline_editor2

        self.last_preview_update_time_timeline = 0.0
        self.last_preview_update_time_heatmap = 0.0
        self.preview_update_interval_seconds = constants.UI_PREVIEW_UPDATE_INTERVAL_S

        self.last_mouse_pos_for_energy_saver = (0, 0)
        self.app.energy_saver.reset_activity_timer()
        
        # Simple arrow key navigation state
        self.arrow_key_state = {
            'last_seek_time': 0.0,
            'seek_interval': 0.033,  # Will be updated based on video FPS
            'initial_press_time': 0.0,  # When key was first pressed
            'continuous_delay': 0.5,  # Delay before continuous scrolling starts (500ms)
        }

        self.batch_videos_data: List[Dict] = []
        self.batch_overwrite_mode_ui: int = 0  # 0: Skip own, 1: Skip any, 2: Overwrite all
        self.batch_processing_method_idx_ui: int = 0
        self.batch_copy_funscript_to_video_location_ui: bool = True
        self.batch_generate_roll_file_ui: bool = True
        self.batch_apply_ultimate_autotune_ui: bool = True
        self.last_overwrite_mode_ui: int = -1 # Used to trigger auto-selection logic

        # TODO: Move this to a separate class/error management module
        self.error_popup_active = False
        self.error_popup_title = ""
        self.error_popup_message = ""
        self.error_popup_action_label = None
        self.error_popup_action_callback = None

    # --- Worker thread for generating preview images ---
    def _preview_generation_worker(self):
        """
        Runs in a background thread. Waits for tasks and processes them.
        """
        while not self.shutdown_event.is_set():
            try:
                task = self.preview_task_queue.get(timeout=0.1)
                task_type = task['type']

                if task_type == 'timeline':
                    image_data = self._generate_funscript_preview_data(
                        task['target_width'],
                        task['target_height'],
                        task['total_duration_s'],
                        task['actions']
                    )
                    self.preview_results_queue.put({'type': 'timeline', 'image_data': image_data})

                elif task_type == 'heatmap':
                    image_data = self._generate_heatmap_data(
                        task['target_width'],
                        task['target_height'],
                        task['total_duration_s'],
                        task['actions']
                    )
                    self.preview_results_queue.put({'type': 'heatmap', 'image_data': image_data})

                self.preview_task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.app.logger.error(f"Error in preview generation worker: {e}", exc_info=True)

    # --- Method to handle completed preview data from the queue ---
    def _process_preview_results(self):
        """
        Called in the main render loop to process any completed preview images.
        """
        try:
            while not self.preview_results_queue.empty():
                result = self.preview_results_queue.get_nowait()

                result_type = result.get('type')
                image_data = result.get('image_data')

                if image_data is None:
                    continue

                if result_type == 'timeline':
                    self.update_texture(self.funscript_preview_texture_id, image_data)
                elif result_type == 'heatmap':
                    self.update_texture(self.heatmap_texture_id, image_data)

                self.preview_results_queue.task_done()

        except queue.Empty:
            pass  # No results to process
        except Exception as e:
            self.app.logger.error(f"Error processing preview results: {e}", exc_info=True)

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

    def submit_async_processing_task(
        self,
        task_id: str,
        task_type: TaskType,
        function,
        args=(),
        kwargs=None,
        priority: TaskPriority = TaskPriority.NORMAL,
        name: str = "Processing",
        show_in_status: bool = True
    ):
        """
        Submit a processing task to run asynchronously without blocking the UI.
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of processing task
            function: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority
            name: Human-readable name for progress display
            show_in_status: Whether to show progress in status bar
        """
        # Track the operation
        self.active_threaded_operations[task_id] = {
            'name': name,
            'show_in_status': show_in_status,
            'progress': 0.0,
            'message': 'Starting...',
            'started_time': time.time()
        }
        
        # Submit to processing thread manager
        def completion_callback(result):
            # Clean up tracking
            if task_id in self.active_threaded_operations:
                del self.active_threaded_operations[task_id]
            self.app.logger.info(f"Async task {task_id} completed successfully")
        
        def error_callback(error):
            # Clean up tracking and show error
            if task_id in self.active_threaded_operations:
                del self.active_threaded_operations[task_id]
            self.app.logger.error(f"Async task {task_id} failed: {error}")
            self.app.set_status_message(f"Error in {name}: {str(error)}", duration=5.0)
        
        self.processing_thread_manager.submit_task(
            task_id=task_id,
            task_type=task_type,
            function=function,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            completion_callback=completion_callback,
            error_callback=error_callback
        )
        
        self.app.logger.info(f"Submitted async task: {task_id} ({name})")

    # --- Extracted CPU-intensive drawing logic for timeline ---
    def _generate_funscript_preview_data(self, target_width, target_height, total_duration_s, actions):
        """
        Performs the numpy/cv2 operations to create the timeline image.
        This is called by the worker thread.
        """
        use_simplified_preview = self.app.app_settings.get("use_simplified_funscript_preview", False)

        # Create background
        image_data = np.full((target_height, target_width, 4), (38, 31, 31, 255), dtype=np.uint8)
        center_y_px = target_height // 2
        cv2.line(image_data, (0, center_y_px), (target_width - 1, center_y_px), (77, 77, 77, 179), 1)

        if not actions or total_duration_s <= 0.001:
            return image_data

        if use_simplified_preview:
            if len(actions) < 2: return image_data

            # --- Simplified Min/Max Envelope Drawing ---
            min_vals = np.full(target_width, target_height, dtype=np.int32)
            max_vals = np.full(target_width, -1, dtype=np.int32)

            # Pre-calculate x coordinates and values
            times_s = np.array([a['at'] for a in actions]) / 1000.0
            positions = np.array([a['pos'] for a in actions])
            x_coords = np.round((times_s / total_duration_s) * (target_width - 1)).astype(np.int32)
            y_coords = np.round((1.0 - positions / 100.0) * (target_height - 1)).astype(np.int32)

            # Find min/max y for each x
            for i in range(len(actions) - 1):
                x1, x2 = x_coords[i], x_coords[i+1]
                y1, y2 = y_coords[i], y_coords[i+1]

                if x1 == x2:
                    min_vals[x1] = min(min_vals[x1], y1, y2)
                    max_vals[x1] = max(max_vals[x1], y1, y2)
                else:
                    # Interpolate for line segments
                    dx = x2 - x1
                    dy = y2 - y1
                    for x in range(x1, x2 + 1):
                        y = y1 + dy * (x - x1) / dx
                        y_int = int(round(y))
                        min_vals[x] = min(min_vals[x], y_int)
                        max_vals[x] = max(max_vals[x], y_int)

            # Create polygon points
            min_points = []
            max_points_rev = []
            for x in range(target_width):
                if max_vals[x] != -1: # Only add points where there is data
                    min_points.append([x, min_vals[x]])
                    max_points_rev.append([x, max_vals[x]])

            if not min_points: return image_data

            # Combine to form a closed polygon
            poly_points = np.array(min_points + max_points_rev[::-1], dtype=np.int32)

            # Draw the semi-transparent polygon
            overlay = image_data.copy()
            envelope_color_rgba = self.app.utility.get_speed_color_from_map(500) # Use a mid-range speed color
            envelope_color_bgra = (int(envelope_color_rgba[2] * 255), int(envelope_color_rgba[1] * 255), int(envelope_color_rgba[0] * 255), 100) # 100 for alpha
            cv2.fillPoly(overlay, [poly_points], envelope_color_bgra)
            cv2.addWeighted(overlay, 0.5, image_data, 0.5, 0, image_data) # Blend with background

        else:
            # --- Detailed, Speed-Colored Line Drawing (Original Logic) ---
            if len(actions) > 1:
                ats = np.array([a['at'] for a in actions], dtype=np.float64) / 1000.0
                pos = np.array([a['pos'] for a in actions], dtype=np.float32) / 100.0
                x = np.clip(((ats / total_duration_s) * (target_width - 1)).astype(np.int32), 0, target_width - 1)
                y = np.clip(((1.0 - pos) * target_height).astype(np.int32), 0, target_height - 1)
                dt = np.diff(ats)
                dpos = np.abs(np.diff(pos * 100.0))  # back to 0..100 for speed calc
                speeds = np.divide(dpos, dt, out=np.zeros_like(dpos), where=dt > 1e-6)
                colors_u8 = self.app.utility.get_speed_colors_vectorized_u8(speeds)  # RGBA uint8
                # Draw per segment; allow OpenCV to optimize internally
                for i in range(len(speeds)):
                    if x[i] == x[i+1] and y[i] == y[i+1]:
                        continue
                    c = colors_u8[i]
                    cv2.line(image_data, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), (int(c[2]), int(c[1]), int(c[0]), int(c[3])), 1)

        return image_data

    # --- Extracted CPU-intensive drawing logic for heatmap ---
    def _generate_heatmap_data(self, target_width, target_height, total_duration_s, actions):
        """
        Performs the numpy/cv2 operations to create the heatmap image.
        This is called by the worker thread.
        """

        colors = self.colors
        image_data = np.full((target_height, target_width, 4), (colors.HEATMAP_BACKGROUND), dtype=np.uint8)

        if len(actions) > 1 and total_duration_s > 0.001:
            ats = np.array([a['at'] for a in actions], dtype=np.float64) / 1000.0
            poss = np.array([a['pos'] for a in actions], dtype=np.float32)
            # Segment starts/ends in pixel space
            x_coords = ((ats / total_duration_s) * (target_width - 1)).astype(np.int32)
            x_coords = np.clip(x_coords, 0, target_width - 1)
            # Compute speeds per segment
            dt = np.diff(ats)
            dpos = np.abs(np.diff(poss))
            speeds = np.divide(dpos, dt, out=np.zeros_like(dpos), where=dt > 1e-6)
            # Vectorized color mapping (prebuilt cache to uint8)
            colors_u8 = self.app.utility.get_speed_colors_vectorized_u8(speeds)
            # For each column, find its segment index via searchsorted
            cols = np.arange(target_width, dtype=np.int32)
            # Map columns to times then to segment indices
            # In pixel domain we can use x_coords boundaries directly
            seg_idx_for_col = np.searchsorted(x_coords, cols, side='right') - 1
            # Columns before first segment are not-yet-fixed; set fully transparent (alpha=0)
            valid_mask = seg_idx_for_col >= 0
            seg_idx_for_col = np.clip(seg_idx_for_col, 0, len(speeds) - 1)
            col_colors = np.zeros((target_width, 4), dtype=np.uint8)
            if np.any(valid_mask):
                col_colors[valid_mask] = colors_u8[seg_idx_for_col[valid_mask]]  # shape (W,4)
                # Ensure fully opaque for fixed columns
                col_colors[valid_mask, 3] = 255
            # Ensure not-yet-fixed columns remain fully transparent
            if np.any(~valid_mask):
                col_colors[~valid_mask, 3] = 0
            # Broadcast to image rows
            image_data[:] = col_colors[np.newaxis, :, :]

        return image_data

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
        glfw.make_context_current(self.window)
        glfw.set_drop_callback(self.window, self.handle_drop)
        glfw.set_window_close_callback(self.window, self.handle_window_close)

        imgui.create_context()
        self.impl = GlfwRenderer(self.window)
        style = imgui.get_style()
        style.window_rounding = 5.0
        style.frame_rounding = 3.0

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

    def _render_energy_saver_indicator(self):
        colors = self.colors
        """Renders a constant indicator when energy saver mode is active."""
        if self.app.energy_saver.energy_saver_active:
            indicator_text = "⚡ Energy Saver Active"
            main_viewport = imgui.get_main_viewport()
            style = imgui.get_style()
            text_size = imgui.calc_text_size(indicator_text)
            win_size = (text_size[0] + style.window_padding[0] * 2, text_size[1] + style.window_padding[1] * 2)
            position = (main_viewport.pos[0] + 10, main_viewport.pos[1] + main_viewport.size[1] - win_size[1] - 10)

            imgui.set_next_window_position(position[0], position[1])
            imgui.set_next_window_bg_alpha(0.65)

            window_flags = (imgui.WINDOW_NO_DECORATION |
                            imgui.WINDOW_NO_MOVE |
                            imgui.WINDOW_ALWAYS_AUTO_RESIZE |
                            imgui.WINDOW_NO_INPUTS |
                            imgui.WINDOW_NO_FOCUS_ON_APPEARING |
                            imgui.WINDOW_NO_NAV)

            imgui.begin("EnergySaverIndicator", closable=False, flags=window_flags)
            imgui.text_colored(indicator_text, *colors.ENERGY_SAVER_INDICATOR)
            imgui.end()

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
                        seek_frame = int(normalized_pos * (total_frames - 1))
                        self.app.event_handlers.handle_seek_bar_drag(seek_frame)
                    else:
                        # Show tooltip with time
                        total_duration = self.app.processor.video_info.get('duration_s', 0)
                        if total_duration > 0:
                            hover_time_s = normalized_pos * total_duration
                            imgui.set_tooltip(f"{_format_time(self.app, hover_time_s)} / {_format_time(self.app, total_duration)}")

        # Draw playback marker over the image
        if self.app.file_manager.video_path and self.app.processor and self.app.processor.video_info and self.app.processor.current_frame_index >= 0:
            total_frames = self.app.processor.video_info.get('total_frames', 0)
            if total_frames > 0:
                normalized_pos = self.app.processor.current_frame_index / (total_frames - 1.0)
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

    def _render_first_run_setup_popup(self):
        # Disabled automatic model downloading - now handled manually via AI menu
        pass
        # app = self.app
        # if app.show_first_run_setup_popup:
        #     imgui.open_popup("First-Time Setup")
        #     main_viewport = imgui.get_main_viewport()
        #     popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
        #                  main_viewport.pos[1] + main_viewport.size[1] * 0.5)
        #     imgui.set_next_window_position(popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5)

        #     # Make the popup non-closable by the user until setup is done or fails.
        #     closable = "complete" in app.first_run_status_message or "failed" in app.first_run_status_message
        #     popup_flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE | (0 if not closable else imgui.WINDOW_CLOSABLE)

        #     if imgui.begin_popup_modal("First-Time Setup", closable, flags=popup_flags)[0]:
        #         imgui.text("Welcome to FunGen!")
        #         imgui.text_wrapped("For the application to work, some default AI models need to be downloaded.")
        #         imgui.separator()

        #         imgui.text_wrapped(f"Status: {app.first_run_status_message}")

        #         # Progress Bar
        #         progress_percent = app.first_run_progress / 100.0
        #         imgui.progress_bar(progress_percent, size=(350, 0), overlay=f"{app.first_run_progress:.1f}%")

        #         imgui.separator()

        #         if closable:
        #             if imgui.button("Close", width=120):
        #                 app.show_first_run_setup_popup = False
        #                 imgui.close_current_popup()

        #         imgui.end_popup()


    # TODO: Move this to a separate class/error management module
    def show_error_popup(self, title, message, action_label=None, action_callback=None):
        self.error_popup_active = True
        self.error_popup_title = title
        self.error_popup_message = message
        self.error_popup_action_label = action_label
        self.error_popup_action_callback = action_callback

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

    def _handle_global_shortcuts(self):
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

            if imgui.is_key_pressed(mapped_key):
                if (mapped_mods_from_string['ctrl'] == io.key_ctrl
                    and mapped_mods_from_string['alt'] == io.key_alt
                    and mapped_mods_from_string['shift'] == io.key_shift
                    and mapped_mods_from_string['super'] == io.key_super):
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

        # Handle non-repeating shortcuts first  
        if check_and_run_shortcut("undo_timeline1", fs_proc.perform_undo_redo, 1, 'undo'):
            pass
        elif check_and_run_shortcut("redo_timeline1", fs_proc.perform_undo_redo, 1, 'redo'):
            pass
        elif self.app.app_state_ui.show_funscript_interactive_timeline2 and (
            check_and_run_shortcut("undo_timeline2", fs_proc.perform_undo_redo, 2, 'undo')
            or check_and_run_shortcut("redo_timeline2", fs_proc.perform_undo_redo, 2, 'redo')
        ): pass
        elif check_and_run_shortcut("toggle_playback", self.app.event_handlers.handle_playback_control, "play_pause"):
            pass
        elif check_and_run_shortcut("jump_to_next_point", self.app.event_handlers.handle_jump_to_point, 'next'):
            pass
        elif check_and_run_shortcut("jump_to_prev_point", self.app.event_handlers.handle_jump_to_point, 'prev'):
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
        if self.app.processor and self.app.processor.fps > 0:
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
        
        # Apply navigation with proper frame-by-frame then continuous logic
        if seek_direction != 0:
            time_since_last = current_time - self.arrow_key_state['last_seek_time']
            key_just_pressed = (left_held and imgui.is_key_pressed(left_key)) or (right_held and imgui.is_key_pressed(right_key))
            
            should_navigate = False
            
            if key_just_pressed:
                # INITIAL KEY PRESS: Always navigate immediately (frame-by-frame response)
                should_navigate = True
                self.arrow_key_state['initial_press_time'] = current_time
            else:
                # KEY HELD DOWN: Check if enough time passed for continuous scrolling
                time_since_initial_press = current_time - self.arrow_key_state['initial_press_time']
                
                if time_since_initial_press >= self.arrow_key_state['continuous_delay']:
                    # Continuous scrolling: only navigate after interval
                    if time_since_last >= self.arrow_key_state['seek_interval']:
                        should_navigate = True
                # else: Still in the delay period, don't navigate (allows precise frame-by-frame)
            
            if should_navigate:
                self._perform_frame_seek(seek_direction)
                self.arrow_key_state['last_seek_time'] = current_time

    def _perform_frame_seek(self, delta_frames):
        """SIMPLE frame seeking - cache first, then direct VideoProcessor method"""
        if not self.app.processor or not self.app.processor.video_info:
            return
            
        new_frame = self.app.processor.current_frame_index + delta_frames
        total_frames = self.app.processor.total_frames
        new_frame = max(0, min(new_frame, total_frames - 1 if total_frames > 0 else 0))

        if new_frame == self.app.processor.current_frame_index:
            return  # No change needed
            
        # SIMPLE STRATEGY: Check cache first, otherwise use VideoProcessor's optimized method
        frame_from_cache = None
        with self.app.processor.frame_cache_lock:
            if new_frame in self.app.processor.frame_cache:
                frame_from_cache = self.app.processor.frame_cache[new_frame]
                self.app.processor.frame_cache.move_to_end(new_frame)
        
        if frame_from_cache is not None:
            # Cache hit: instant update
            self.app.processor.current_frame_index = new_frame
            self.app.processor.current_frame = frame_from_cache
        else:
            # Cache miss: let VideoProcessor handle it optimally
            self.app.processor.seek_video(new_frame)
        
        # Update UI
        self.app.app_state_ui.force_timeline_pan_to_current_frame = True
        if self.app.project_manager: 
            self.app.project_manager.project_dirty = True
        self.app.energy_saver.reset_activity_timer()

    # Removed complex predictive caching - it was blocking the UI
    # Keep navigation simple: cache check first, then single frame fetch if needed

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

    def _render_batch_confirmation_dialog(self):
        app = self.app
        if not app.show_batch_confirmation_dialog:
            return

        colors = self.colors
        imgui.open_popup("Batch Processing Setup")
        main_viewport = imgui.get_main_viewport()
        imgui.set_next_window_size(main_viewport.size[0] * 0.7, main_viewport.size[1] * 0.8, condition=imgui.APPEARING)
        popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                     main_viewport.pos[1] + main_viewport.size[1] * 0.5)
        imgui.set_next_window_position(popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5, condition=imgui.APPEARING)

        if imgui.begin_popup_modal("Batch Processing Setup", True)[0]:
            imgui.text(f"Found {len(self.batch_videos_data)} videos for batch processing.")
            imgui.separator()

            imgui.text("Overwrite Strategy:")
            imgui.same_line()
            if imgui.radio_button("Skip existing FunGen scripts", self.batch_overwrite_mode_ui == 0): self.batch_overwrite_mode_ui = 0
            imgui.same_line()
            if imgui.radio_button("Skip if ANY script exists", self.batch_overwrite_mode_ui == 1): self.batch_overwrite_mode_ui = 1
            imgui.same_line()
            if imgui.radio_button("Overwrite all existing scripts", self.batch_overwrite_mode_ui == 2): self.batch_overwrite_mode_ui = 2

            if self.batch_overwrite_mode_ui != self.last_overwrite_mode_ui:
                for video in self.batch_videos_data:
                    status = video["funscript_status"]
                    if self.batch_overwrite_mode_ui == 0: video["selected"] = status != 'fungen'
                    elif self.batch_overwrite_mode_ui == 1: video["selected"] = status is None
                    elif self.batch_overwrite_mode_ui == 2: video["selected"] = True
                self.last_overwrite_mode_ui = self.batch_overwrite_mode_ui

            imgui.separator()

            if imgui.begin_child("VideoList", height=-120):
                table_flags = imgui.TABLE_BORDERS | imgui.TABLE_SIZING_STRETCH_PROP | imgui.TABLE_SCROLL_Y
                if imgui.begin_table("BatchVideosTable", 4, flags=table_flags):
                    imgui.table_setup_column("Process", init_width_or_weight=0.5)
                    imgui.table_setup_column("Video File", init_width_or_weight=4.0)
                    imgui.table_setup_column("Detected", init_width_or_weight=1.3)
                    imgui.table_setup_column("Override", init_width_or_weight=1.5)

                    imgui.table_headers_row()

                    video_format_options = ["Auto (Heuristic)", "2D", "VR (he_sbs)", "VR (he_tb)", "VR (fisheye_sbs)", "VR (fisheye_tb)"]

                    for i, video_data in enumerate(self.batch_videos_data):
                        imgui.table_next_row()
                        imgui.table_set_column_index(0); imgui.push_id(f"sel_{i}")
                        _, video_data["selected"] = imgui.checkbox("##select", video_data["selected"])
                        imgui.pop_id()

                        imgui.table_set_column_index(1)
                        status = video_data["funscript_status"]
                        if status == 'fungen': imgui.text_colored(os.path.basename(video_data["path"]), *colors.VIDEO_STATUS_FUNGEN)
                        elif status == 'other': imgui.text_colored(os.path.basename(video_data["path"]), *colors.VIDEO_STATUS_OTHER)
                        else: imgui.text(os.path.basename(video_data["path"]))

                        if imgui.is_item_hovered():
                            if status == 'fungen':
                                imgui.set_tooltip("Funscript created by this version of FunGen")
                            elif status == 'other':
                                imgui.set_tooltip("Funscript exists (unknown or older version)")
                            else:
                                imgui.set_tooltip("No Funscript exists for this video")

                        imgui.table_set_column_index(2); imgui.text(video_data["detected_format"])

                        imgui.table_set_column_index(3); imgui.push_id(f"ovr_{i}"); imgui.set_next_item_width(-1)
                        _, video_data["override_format_idx"] = imgui.combo("##override", video_data["override_format_idx"], video_format_options)
                        imgui.pop_id()

                    imgui.end_table()
                imgui.end_child()

            imgui.separator()
            imgui.text("Processing Method:")
            
            # Get available batch-compatible trackers dynamically
            from application.gui_components.dynamic_tracker_ui import get_dynamic_tracker_ui
            from config.tracker_discovery import TrackerCategory
            
            tracker_ui = get_dynamic_tracker_ui()
            discovery = tracker_ui.discovery
            
            # Get live (non-intervention) and offline trackers
            batch_compatible_trackers = []
            tracker_internal_names = []
            
            # Add offline trackers
            offline_trackers = discovery.get_trackers_by_category(TrackerCategory.OFFLINE)
            for tracker in offline_trackers:
                if tracker.supports_batch:
                    batch_compatible_trackers.append(tracker.display_name)
                    tracker_internal_names.append(tracker.internal_name)
            
            # Add live trackers (non-intervention only)
            live_trackers = discovery.get_trackers_by_category(TrackerCategory.LIVE)
            for tracker in live_trackers:
                if tracker.supports_batch and not tracker.requires_intervention:
                    batch_compatible_trackers.append(tracker.display_name)
                    tracker_internal_names.append(tracker.internal_name)
            
            # Create dropdown
            imgui.set_next_item_width(300)
            changed, self.selected_batch_method_idx_ui = imgui.combo(
                "##batch_tracker", 
                self.selected_batch_method_idx_ui,
                batch_compatible_trackers
            )
            
            # Store the selected tracker's internal name for later use
            if 0 <= self.selected_batch_method_idx_ui < len(tracker_internal_names):
                self.selected_batch_tracker_name = tracker_internal_names[self.selected_batch_method_idx_ui]
            else:
                self.selected_batch_tracker_name = None

            imgui.text("Output Options:")
            _, self.batch_apply_ultimate_autotune_ui = imgui.checkbox("Apply Ultimate Autotune", self.batch_apply_ultimate_autotune_ui)
            imgui.same_line()
            _, self.batch_copy_funscript_to_video_location_ui = imgui.checkbox("Save copy next to video", self.batch_copy_funscript_to_video_location_ui)
            imgui.same_line()
            
            # Check if selected tracker supports roll file generation (3-stage trackers)
            has_3_stages = False
            if hasattr(self, 'selected_batch_tracker_name') and self.selected_batch_tracker_name:
                tracker_info = discovery.get_tracker_info(self.selected_batch_tracker_name)
                if tracker_info and tracker_info.properties:
                    has_3_stages = tracker_info.properties.get("num_stages", 0) >= 3
            
            if not has_3_stages:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True); imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            _, self.batch_generate_roll_file_ui = imgui.checkbox("Generate .roll file", self.batch_generate_roll_file_ui if has_3_stages else False)
            if not has_3_stages:
                imgui.pop_style_var(); imgui.internal.pop_item_flag()

            imgui.separator()
            if imgui.button("Start Batch", width=120):
                app._initiate_batch_processing_from_confirmation()
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", width=120):
                app._cancel_batch_processing_from_confirmation()
                imgui.close_current_popup()

            imgui.end_popup()

    def render_gui(self):
        self.component_render_times.clear()

        # Separate timing to identify performance bottlenecks
        self._time_render("EnergyDetection", self._handle_energy_saver_interaction_detection)
        self._time_render("GlobalShortcuts", self._handle_global_shortcuts)

        if self.app.shortcut_manager.is_recording_shortcut_for:
            self._time_render("ShortcutRecordingInput", self.app.shortcut_manager.handle_shortcut_recording_input)
            self.app.energy_saver.reset_activity_timer()

        self._time_render("StageProcessorEvents", self.app.stage_processor.process_gui_events)

        # --- Process preview results queue every frame ---
        self._process_preview_results()

        imgui.new_frame()
        main_viewport = imgui.get_main_viewport()
        self.window_width, self.window_height = main_viewport.size
        app_state = self.app.app_state_ui
        app_state.window_width = int(self.window_width)
        app_state.window_height = int(self.window_height)

        self._time_render("MainMenu", self.main_menu.render)

        font_scale = self.app.app_settings.get("global_font_scale", 1.0)
        imgui.get_io().font_global_scale = font_scale

        if hasattr(app_state, 'main_menu_bar_height_from_menu_class'):
            self.main_menu_bar_height = app_state.main_menu_bar_height_from_menu_class
        else:
            self.main_menu_bar_height = imgui.get_frame_height_with_spacing() if self.main_menu else 0

        if not app_state.gauge_pos_initialized and self.main_menu_bar_height > 0:
            app_state.initialize_gauge_default_y(self.main_menu_bar_height)

        app_state.update_current_script_display_values()

        if app_state.ui_layout_mode == 'fixed':
            panel_y_start = self.main_menu_bar_height
            timeline1_render_h = app_state.timeline_base_height if app_state.show_funscript_interactive_timeline else 0
            timeline2_render_h = app_state.timeline_base_height if app_state.show_funscript_interactive_timeline2 else 0
            interactive_timelines_total_height = timeline1_render_h + timeline2_render_h
            available_height_for_main_panels = max(100, self.window_height - panel_y_start - interactive_timelines_total_height)
            app_state.fixed_layout_geometry = {}
            is_full_width_nav = getattr(app_state, 'full_width_nav', False)
            control_panel_w = 450 * font_scale
            graphs_panel_w = 450 * font_scale
            video_nav_bar_h = 150

            if is_full_width_nav:
                top_panels_h = max(50, available_height_for_main_panels - video_nav_bar_h)
                nav_y_start = panel_y_start + top_panels_h
                if app_state.show_video_display_window:
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
                if app_state.show_video_display_window:
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

            timeline_current_y_start = panel_y_start + available_height_for_main_panels
            if app_state.show_funscript_interactive_timeline:
                app_state.fixed_layout_geometry['Timeline1'] = {'pos': (0, timeline_current_y_start), 'size': (self.window_width, timeline1_render_h)}
                self._time_render("TimelineEditor1", self.timeline_editor1.render, timeline_current_y_start, timeline1_render_h, view_mode=app_state.ui_view_mode)
                timeline_current_y_start += timeline1_render_h
            if app_state.show_funscript_interactive_timeline2:
                app_state.fixed_layout_geometry['Timeline2'] = {'pos': (0, timeline_current_y_start), 'size': (self.window_width, timeline2_render_h)}
                self._time_render("TimelineEditor2", self.timeline_editor2.render, timeline_current_y_start, timeline2_render_h, view_mode=app_state.ui_view_mode)
        else:
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
            if app_state.just_switched_to_floating:
                app_state.just_switched_to_floating = False

        if hasattr(app_state, 'show_chapter_list_window') and app_state.show_chapter_list_window:
            self._time_render("ChapterListWindow", self.chapter_list_window_ui.render)
        self._time_render("Popups", lambda: (
            self.gauge_window_ui_t1.render(),
            self.gauge_window_ui_t2.render(),
            self.lr_dial_window_ui.render(),
            self._render_batch_confirmation_dialog(),
            self.file_dialog.draw() if self.file_dialog.open else None,
            self._render_status_message(app_state),
            self.app.updater.render_update_dialog(),
            self.app.updater.render_update_error_dialog(),
            self.app.updater.render_update_settings_dialog()
        ))
        self._time_render("EnergySaverIndicator", self._render_energy_saver_indicator)

        # TODO: Move this to a separate class/error management module
        if self.error_popup_active:
            imgui.open_popup("ErrorPopup")
        # Center the popup and set a normal size (compatibility for imgui versions)
        if hasattr(imgui, 'get_main_viewport'):
            main_viewport = imgui.get_main_viewport()
            popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                         main_viewport.pos[1] + main_viewport.size[1] * 0.5)
            imgui.set_next_window_position(popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5)
        else:
            # Fallback: center on window size if viewport not available
            popup_pos = (self.window_width * 0.5, self.window_height * 0.5)
            imgui.set_next_window_position(popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5)
        popup_width = 600
        imgui.set_next_window_size(popup_width, 0)  # Wider width, auto height
        if imgui.begin_popup_modal("ErrorPopup")[0]:
            # Center title
            window_width = imgui.get_window_width()
            title_width = imgui.calc_text_size(self.error_popup_title)[0]
            imgui.set_cursor_pos_x((window_width - title_width) * 0.5)
            imgui.text(self.error_popup_title)
            imgui.separator()
            # Center message
            message_lines = self.error_popup_message.split('\n')
            for line in message_lines:
                line_width = imgui.calc_text_size(line)[0]
                imgui.set_cursor_pos_x((window_width - line_width) * 0.5)
                imgui.text(line)
            imgui.spacing()
            # Center button
            button_width = 120
            imgui.set_cursor_pos_x((window_width - button_width) * 0.5)
            if imgui.button("Close", width=button_width):
                self.error_popup_active = False
                imgui.close_current_popup()
                if self.error_popup_action_callback:
                    self.error_popup_action_callback()
            imgui.end_popup()

        # Render TensorRT Compiler Window if open
        if hasattr(self.app, 'tensorrt_compiler_window') and self.app.tensorrt_compiler_window:
            self.app.tensorrt_compiler_window.render()

        # --- Render Generated File Manager window ---
        if self.app.app_state_ui.show_generated_file_manager:
            self.generated_file_manager_ui.render()

        # --- Render Autotuner Window ---
        self._time_render("AutotunerWindow", self.autotuner_window_ui.render)

        self.perf_frame_count += 1
        if time.time() - self.last_perf_log_time > self.perf_log_interval:
            self._log_performance()
        
        # Continuously update frontend queue with current data
        self._update_frontend_perf_queue()

        imgui.render()
        if self.impl:
            self.impl.render(imgui.get_draw_data())

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

    def _render_status_message(self, app_state):
        if app_state.status_message and time.time() < app_state.status_message_time:
            imgui.set_next_window_position(self.window_width - 310, self.window_height - 40)
            imgui.begin("StatusMessage", flags=(
                imgui.WINDOW_NO_DECORATION |
                imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_ALWAYS_AUTO_RESIZE |
                imgui.WINDOW_NO_INPUTS |
                imgui.WINDOW_NO_FOCUS_ON_APPEARING |
                imgui.WINDOW_NO_NAV))
            imgui.text(app_state.status_message)
            imgui.end()
        elif app_state.status_message:
            app_state.status_message = ""



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
                glfw.poll_events()
                if self.impl: self.impl.process_inputs()
                gl.glClearColor(*colors.BACKGROUND_CLEAR)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                self.render_gui()
                if (
                    self.app.app_settings.get("autosave_enabled", True)
                    and time.time() - self.app.project_manager.last_autosave_time > self.app.app_settings.get("autosave_interval_seconds", constants.DEFAULT_AUTOSAVE_INTERVAL_SECONDS)
                ):
                    self.app.project_manager.perform_autosave()
                self.app.energy_saver.check_and_update_energy_saver()
                glfw.swap_buffers(self.window)
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
            self.app.logger.info("Shutting down ProcessingThreadManager...")
            self.processing_thread_manager.shutdown(timeout=3.0)
            
            # Shutdown legacy preview worker thread
            for _ in self.preview_worker_threads:
                try:
                    self.preview_task_queue.put_nowait({'type': 'shutdown'})
                except queue.Full:
                    pass
            for t in self.preview_worker_threads:
                t.join()

            if self.frame_texture_id: gl.glDeleteTextures([self.frame_texture_id]); self.frame_texture_id = 0
            if self.heatmap_texture_id: gl.glDeleteTextures([self.heatmap_texture_id]); self.heatmap_texture_id = 0
            if self.funscript_preview_texture_id: gl.glDeleteTextures(
                [self.funscript_preview_texture_id]); self.funscript_preview_texture_id = 0

            if self.impl: self.impl.shutdown()
            if self.window: glfw.destroy_window(self.window)
            glfw.terminate()
            self.app.logger.info("GUI terminated.", extra={'status_message': False})
            