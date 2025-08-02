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
from typing import Optional, List, Dict

from application.classes.gauge import GaugeWindow
from application.classes.file_dialog import ImGuiFileDialog
from application.classes.interactive_timeline import InteractiveFunscriptTimeline
from application.classes.lr_dial import LRDialWindow
from application.classes.menu import MainMenu

from application.gui_components.control_panel_ui import ControlPanelUI
from application.gui_components.video_display_ui import VideoDisplayUI
from application.gui_components.video_navigation_ui import VideoNavigationUI, ChapterListWindow
from application.gui_components.info_graphs_ui import InfoGraphsUI
from application.gui_components.generated_file_manager_window import GeneratedFileManagerWindow
from application.gui_components.autotuner_window import AutotunerWindow

from config import constants
from config.element_group_colors import AppGUIColors, UpdateSettingsColors


class GUI:
    def __init__(self, app_logic):
        self.app = app_logic  # app_logic is ApplicationLogic instance
        self.window = None
        self.impl = None
        self.window_width = self.app.app_settings.get("window_width", 1800)
        self.window_height = self.app.app_settings.get("window_height", 1000)
        self.main_menu_bar_height = 0

        self.frame_texture_id = 0
        self.heatmap_texture_id = 0
        self.funscript_preview_texture_id = 0

        # --- Threading for asynchronous preview generation ---
        self.preview_task_queue = queue.Queue()
        self.preview_results_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.preview_worker_thread = threading.Thread(target=self._preview_generation_worker, daemon=True)
        self.preview_worker_thread.start()

        # --- State for incremental texture generation ---
        self.last_submitted_action_count_timeline: int = 0
        self.last_submitted_action_count_heatmap: int = 0

        # Performance monitoring
        self.component_render_times = {}
        self.perf_log_interval = 5  # Log performance every 5 seconds
        self.last_perf_log_time = time.time()
        self.perf_frame_count = 0
        self.perf_accumulated_times = {}

        # Standard Components (owned by GUI)
        self.file_dialog = ImGuiFileDialog(app_logic_instance=self.app)
        self.main_menu = MainMenu(self.app)
        self.gauge_window_ui_t1 = GaugeWindow(self.app, timeline_num=1)
        self.gauge_window_ui_t2 = GaugeWindow(self.app, timeline_num=2)
        self.lr_dial_window_ui = LRDialWindow(self.app)

        self.timeline_editor1 = InteractiveFunscriptTimeline(app_instance=self.app, timeline_num=1)
        self.timeline_editor2 = InteractiveFunscriptTimeline(app_instance=self.app, timeline_num=2)

        # Modularized UI Panel Components
        self.control_panel_ui = ControlPanelUI(self.app)
        self.video_display_ui = VideoDisplayUI(self.app, self)  # Pass self for texture updates
        self.video_navigation_ui = VideoNavigationUI(self.app, self)  # Pass self for texture methods
        self.info_graphs_ui = InfoGraphsUI(self.app)
        self.chapter_list_window_ui = ChapterListWindow(self.app, nav_ui=self.video_navigation_ui)
        self.generated_file_manager_ui = GeneratedFileManagerWindow(self.app)
        self.autotuner_window_ui = AutotunerWindow(self.app)

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
        self.preview_update_interval_seconds = 1.0

        self.last_mouse_pos_for_energy_saver = (0, 0)
        self.app.energy_saver.reset_activity_timer()

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
                task = self.preview_task_queue.get(timeout=1)
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

    # --- Extracted CPU-intensive drawing logic for timeline ---
    def _generate_funscript_preview_data(self, target_width, target_height, total_duration_s, actions):
        """
        Performs the numpy/cv2 operations to create the timeline image.
        This is called by the worker thread.
        """
        # Create background
        bg_color_cv_bgra = (int(0.15 * 255), int(0.12 * 255), int(0.12 * 255), 255)
        image_data = np.full((target_height, target_width, 4), bg_color_cv_bgra, dtype=np.uint8)
        center_y_px = target_height // 2
        cv2.line(image_data, (0, center_y_px), (target_width - 1, center_y_px),
                 (int(0.3 * 255), int(0.3 * 255), int(0.3 * 255), int(0.7 * 255)), 1)

        # Draw funscript lines
        if len(actions) > 1 and total_duration_s > 0.001:
            for i in range(len(actions) - 1):
                p1_action, p2_action = actions[i], actions[i + 1]
                time1_s, pos1_norm = p1_action["at"] / 1000.0, p1_action["pos"] / 100.0
                px1, py1 = int(round((time1_s / total_duration_s) * target_width)), int(
                    round((1.0 - pos1_norm) * target_height))
                time2_s, pos2_norm = p2_action["at"] / 1000.0, p2_action["pos"] / 100.0
                px2, py2 = int(round((time2_s / total_duration_s) * target_width)), int(
                    round((1.0 - pos2_norm) * target_height))

                px1, py1 = np.clip(px1, 0, target_width - 1), np.clip(py1, 0, target_height - 1)
                px2, py2 = np.clip(px2, 0, target_width - 1), np.clip(py2, 0, target_height - 1)
                if px1 == px2 and py1 == py2: continue

                delta_pos = abs(p2_action["pos"] - p1_action["pos"])
                delta_time_ms = p2_action["at"] - p1_action["at"]
                speed_pps = delta_pos / (delta_time_ms / 1000.0) if delta_time_ms > 0 else 0.0
                segment_color_float_rgba = self.app.utility.get_speed_color_from_map(speed_pps)
                segment_color_cv_bgra = (
                    int(segment_color_float_rgba[2] * 255), int(segment_color_float_rgba[1] * 255),
                    int(segment_color_float_rgba[0] * 255), int(segment_color_float_rgba[3] * 255)
                )
                cv2.line(image_data, (px1, py1), (px2, py2), segment_color_cv_bgra, thickness=1)

        return image_data

    # --- Extracted CPU-intensive drawing logic for heatmap ---
    def _generate_heatmap_data(self, target_width, target_height, total_duration_s, actions):
        """
        Performs the numpy/cv2 operations to create the heatmap image.
        This is called by the worker thread.
        """
        bg_color_heatmap_texture_rgba255 = (int(0.08 * 255), int(0.08 * 255), int(0.10 * 255), 255)
        image_data = np.full((target_height, target_width, 4), bg_color_heatmap_texture_rgba255, dtype=np.uint8)

        if len(actions) > 1 and total_duration_s > 0.001:
            for i in range(len(actions) - 1):
                p1, p2 = actions[i], actions[i + 1]
                start_time_s, end_time_s = p1["at"] / 1000.0, p2["at"] / 1000.0
                if end_time_s <= start_time_s: continue
                seg_start_x_px = int(round((start_time_s / total_duration_s) * target_width))
                seg_end_x_px = int(round((end_time_s / total_duration_s) * target_width))
                seg_start_x_px, seg_end_x_px = max(0, seg_start_x_px), min(target_width, seg_end_x_px)

                if seg_end_x_px <= seg_start_x_px:
                    if seg_start_x_px < target_width:
                        seg_end_x_px = seg_start_x_px + 1
                    else:
                        continue

                delta_pos = abs(p2["pos"] - p1["pos"])
                delta_time_s_seg = (p2["at"] - p1["at"]) / 1000.0
                speed_pps = delta_pos / delta_time_s_seg if delta_time_s_seg > 0.001 else 0.0
                segment_color_float_rgba = self.app.utility.get_speed_color_from_map(speed_pps)
                segment_color_byte_rgba = np.array([int(c * 255) for c in segment_color_float_rgba], dtype=np.uint8)
                image_data[:, seg_start_x_px:seg_end_x_px] = segment_color_byte_rgba

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
        """Logs the average performance of components."""
        if self.perf_frame_count == 0:
            self.app.logger.debug("No frames rendered (yet).")
            return

        log_message = "Avg Render Times (ms) over {} frames:".format(self.perf_frame_count)
        for name, total_time in self.perf_accumulated_times.items():
            avg_time = total_time / self.perf_frame_count
            log_message += f"\n  - {name}: {avg_time:.3f}"

        self.app.logger.debug(log_message)  # Use debug to avoid spamming info logs

        # Reset accumulators for next interval
        self.perf_accumulated_times.clear()
        self.perf_frame_count = 0
        self.last_perf_log_time = time.time()

    def init_glfw(self) -> bool:
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

    def handle_drop(self, window, paths):
        if not paths:
            return

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

        # Determine format and upload
        if len(image.shape) == 2:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RED, w, h, 0, gl.GL_RED, gl.GL_UNSIGNED_BYTE, image)
        elif image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb_image)
        elif image.shape[2] == 4:
            # The worker already produces RGBA, no conversion needed
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _render_energy_saver_indicator(self):
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
            imgui.text_colored(indicator_text, *AppGUIColors.ENERGY_SAVER_INDICATOR)
            imgui.end()

    # --- This function now submits a task to the worker thread ---
    def render_funscript_timeline_preview(self, total_duration_s: float, graph_height: int):
        app_state = self.app.app_state_ui
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

        if needs_regen and self.preview_task_queue.empty():
            actions_copy = self.app.funscript_processor.get_actions('primary').copy()
            task = {
                'type': 'timeline',
                'target_width': current_bar_width_int,
                'target_height': graph_height,
                'total_duration_s': total_duration_s,
                'actions': actions_copy
            }
            self.preview_task_queue.put(task)

            # Update state after submission
            app_state.funscript_preview_dirty = False
            app_state.last_funscript_preview_bar_width = current_bar_width_int
            app_state.last_funscript_preview_duration_s = total_duration_s
            self.last_submitted_action_count_timeline = current_action_count
            if is_live_tracking and incremental_update_needed:
                self.last_preview_update_time_timeline = time.time()

        # --- Rendering Logic (uses the existing texture until a new one is ready) ---
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 5)
        canvas_p1_x = imgui.get_cursor_screen_pos()[0]
        canvas_p1_y_offset = imgui.get_cursor_screen_pos()[1]

        imgui.image(self.funscript_preview_texture_id, current_bar_width_float, graph_height, uv0=(0, 0), uv1=(1, 1))

        # Draw playback marker over the image
        if self.app.file_manager.video_path and self.app.processor and self.app.processor.video_info and self.app.processor.current_frame_index >= 0:
            total_frames = self.app.processor.video_info.get('total_frames', 0)
            if total_frames > 0:
                normalized_pos = self.app.processor.current_frame_index / (total_frames - 1.0)
                marker_x = (canvas_p1_x + style.frame_padding[0]) + (normalized_pos * (current_bar_width_float - style.frame_padding[0] * 2))
                marker_color = imgui.get_color_u32_rgba(*AppGUIColors.MARKER)
                draw_list_marker = imgui.get_window_draw_list()
                draw_list_marker.add_line(marker_x, canvas_p1_y_offset, marker_x, canvas_p1_y_offset + graph_height, marker_color, 1.0)

        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 5)

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

        if needs_regen and self.preview_task_queue.empty():
            actions_copy = self.app.funscript_processor.get_actions('primary').copy()
            task = {
                'type': 'heatmap',
                'target_width': current_bar_width_int,
                'target_height': app_state.heatmap_texture_fixed_height,
                'total_duration_s': total_video_duration_s,
                'actions': actions_copy
            }
            self.preview_task_queue.put(task)

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
        app = self.app
        if app.show_first_run_setup_popup:
            imgui.open_popup("First-Time Setup")
            main_viewport = imgui.get_main_viewport()
            popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                         main_viewport.pos[1] + main_viewport.size[1] * 0.5)
            imgui.set_next_window_position(popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5)

            # Make the popup non-closable by the user until setup is done or fails.
            closable = "complete" in app.first_run_status_message or "failed" in app.first_run_status_message
            popup_flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE | (0 if not closable else imgui.WINDOW_CLOSABLE)

            if imgui.begin_popup_modal("First-Time Setup", closable, flags=popup_flags)[0]:
                imgui.text("Welcome to FunGen!")
                imgui.text_wrapped("For the application to work, some default AI models need to be downloaded.")
                imgui.separator()

                imgui.text_wrapped(f"Status: {app.first_run_status_message}")

                # Progress Bar
                progress_percent = app.first_run_progress / 100.0
                imgui.progress_bar(progress_percent, size=(350, 0), overlay=f"{app.first_run_progress:.1f}%")

                imgui.separator()

                if closable:
                    if imgui.button("Close", width=120):
                        app.show_first_run_setup_popup = False
                        imgui.close_current_popup()

                imgui.end_popup()


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
        app_state = self.app.app_state_ui
        if not imgui.is_item_visible():
            return
        marks = [(current_target_fps, AppGUIColors.FPS_TARGET_MARKER, "Target"), (tracker_fps, AppGUIColors.FPS_TRACKER_MARKER, "Tracker"), (processor_fps, AppGUIColors.FPS_PROCESSOR_MARKER, "Processor")]
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
        elif video_loaded:
            seek_delta_frames = 0
            processed_seek = False
            if check_and_run_shortcut("seek_prev_frame", lambda: None):
                seek_delta_frames = -1
                processed_seek = True
            elif check_and_run_shortcut("seek_next_frame", lambda: None):
                seek_delta_frames = 1
                processed_seek = True
            elif check_and_run_shortcut("jump_to_next_point", self.app.event_handlers.handle_jump_to_point, 'next'):
                pass
            elif check_and_run_shortcut("jump_to_prev_point", self.app.event_handlers.handle_jump_to_point, 'prev'):
                pass

            # Allow seeking if video is loaded, regardless of play/pause state
            if processed_seek and seek_delta_frames != 0:
                if self.app.processor and self.app.processor.video_info:
                    paused_state = False
                    if hasattr(self.app.processor, 'pause_event'):
                        paused_state = self.app.processor.pause_event.is_set()
                    self.app.logger.debug(f"Shortcut seek triggered: {seek_delta_frames} frames (paused={paused_state}, is_processing={self.app.processor.is_processing})")
                    new_frame = self.app.processor.current_frame_index + seek_delta_frames
                    total_frames_vid = self.app.processor.total_frames
                    new_frame = np.clip(new_frame, 0, total_frames_vid - 1 if total_frames_vid > 0 else 0)

                    if new_frame != self.app.processor.current_frame_index:
                        self.app.processor.seek_video(new_frame)
                        app_state.force_timeline_pan_to_current_frame = True
                        if self.app.project_manager: self.app.project_manager.project_dirty = True
                        self.app.energy_saver.reset_activity_timer()

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
            or imgui.is_any_item_focused()
            or (self.file_dialog and self.file_dialog.open)):
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
                        if status == 'fungen': imgui.text_colored(os.path.basename(video_data["path"]), *AppGUIColors.VIDEO_STATUS_FUNGEN)
                        elif status == 'other': imgui.text_colored(os.path.basename(video_data["path"]), *AppGUIColors.VIDEO_STATUS_OTHER)
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
            if imgui.radio_button("3-Stage", self.selected_batch_method_idx_ui == 0):
                self.selected_batch_method_idx_ui = 0
            imgui.same_line()
            if imgui.radio_button("2-Stage", self.selected_batch_method_idx_ui == 1):
                self.selected_batch_method_idx_ui = 1
            imgui.same_line()
            if imgui.radio_button("Oscillation Detector", self.selected_batch_method_idx_ui == 2):
                self.selected_batch_method_idx_ui = 2

            imgui.text("Output Options:")
            _, self.batch_apply_ultimate_autotune_ui = imgui.checkbox("Apply Ultimate Autotune", self.batch_apply_ultimate_autotune_ui)
            imgui.same_line()
            _, self.batch_copy_funscript_to_video_location_ui = imgui.checkbox("Save copy next to video", self.batch_copy_funscript_to_video_location_ui)
            imgui.same_line()
            is_3_stage = self.selected_batch_method_idx_ui == 0
            if not is_3_stage:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True); imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
            _, self.batch_generate_roll_file_ui = imgui.checkbox("Generate .roll file", self.batch_generate_roll_file_ui if is_3_stage else False)
            if not is_3_stage:
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

        self._time_render("EnergySaver+Shortcuts", lambda: (
            self._handle_energy_saver_interaction_detection(),
            self._handle_global_shortcuts()))

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
            self._render_update_settings_dialog()
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
        popup_width = 480
        imgui.set_next_window_size(popup_width, 0)  # Normal width, auto height
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

        imgui.render()
        if self.impl:
            self.impl.render(imgui.get_draw_data())

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

    def _render_update_settings_dialog(self):
        """Renders the combined update commit & GitHub token dialog with tabs."""
        if self.app.app_state_ui.show_update_settings_dialog:
            imgui.open_popup("Updates & GitHub Token")
            self.app.app_state_ui.show_update_settings_dialog = False
            # Load versions when dialog opens
            self.app.updater.load_available_versions_async()

        # Initialize buffers if needed
        if not hasattr(self, '_github_token_buffer'):
            self._github_token_buffer = self.app.updater.token_manager.get_token()
        if not hasattr(self, '_updates_active_tab'):
            self._updates_active_tab = 0  # 0 = Version, 1 = Token

        # Set initial size and make resizable
        if not hasattr(self, '_update_settings_window_size'):
            self._update_settings_window_size = (800, 600)
        
        # Set initial position for first time
        if not hasattr(self, '_update_settings_window_pos'):
            main_viewport = imgui.get_main_viewport()
            popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                         main_viewport.pos[1] + main_viewport.size[1] * 0.5)
            self._update_settings_window_pos = (popup_pos[0] - 400, popup_pos[1] - 300)  # Center the window
        
        imgui.set_next_window_size(*self._update_settings_window_size, condition=imgui.ONCE)
        imgui.set_next_window_size_constraints((600, 400), (1200, 800))
        imgui.set_next_window_position(*self._update_settings_window_pos, condition=imgui.ONCE)

        if imgui.begin_popup_modal("Updates & GitHub Token", True)[0]:
            # Save window size and position for persistence
            window_size = imgui.get_window_size()
            window_pos = imgui.get_window_position()
            if window_size[0] > 0 and window_size[1] > 0:
                self._update_settings_window_size = window_size
            if window_pos[0] > 0 and window_pos[1] > 0:
                self._update_settings_window_pos = window_pos
            
            # Tab bar
            if imgui.begin_tab_bar("Updates & GitHub Token Tabs"):
                # Version Selection Tab
                if imgui.begin_tab_item("Choose FunGen Version")[0]:
                    self._updates_active_tab = 0
                    imgui.end_tab_item()
                
                # GitHub Token Tab
                if imgui.begin_tab_item("GitHub Token")[0]:
                    self._updates_active_tab = 1
                    imgui.end_tab_item()
                
                imgui.end_tab_bar()

            # Tab content
            if self._updates_active_tab == 0:
                # Version Selection Tab
                self._render_version_picker_content()
            else:
                # GitHub Token Tab
                self._render_github_token_content()

            imgui.separator()
            
            # Close button positioned at bottom right
            imgui.set_cursor_pos_x(imgui.get_window_width() - 130)  # Position from right edge
            if imgui.button("Close", width=120):
                imgui.close_current_popup()

            imgui.end_popup()

    def _render_version_picker_content(self):
        """Renders the version picker content within the tabbed dialog."""
        if self.app.updater.version_picker_loading:
            imgui.text("Loading available commits...")
            spinner_chars = "|/-\\"
            spinner_index = int(time.time() * 4) % 4
            imgui.text(f"Please wait... {spinner_chars[spinner_index]}")
        elif self.app.updater.update_in_progress:
            imgui.text(self.app.updater.status_message)
            spinner_chars = "|/-\\"
            spinner_index = int(time.time() * 4) % 4
            imgui.text(f"Processing... {spinner_chars[spinner_index]}")
        else:
            target_branch = self.app.updater.BRANCH
            imgui.text(f"Select a commit from branch '{target_branch}' to switch to:")
            imgui.separator()

            # Version list with inline changelogs
            child_width = imgui.get_content_region_available()[0]
            child_height = 400
            imgui.begin_child("VersionList", child_width, child_height, border=True, flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)

            current_hash = self.app.updater._get_local_commit_hash()
            
            for version in self.app.updater.available_versions:
                commit_hash = version['commit_hash']
                is_expanded = commit_hash in self.app.updater.expanded_commits
                
                # Highlight current version
                is_current = current_hash and commit_hash.startswith(current_hash[:7])
                
                if is_current:
                    imgui.push_style_color(imgui.COLOR_TEXT, *AppGUIColors.VERSION_CURRENT_HIGHLIGHT)
                
                # Create expand/collapse button
                expand_icon = "▼" if is_expanded else "▶"
                button_text = f"{expand_icon} {commit_hash[:7]}"
                if imgui.button(button_text, width=80):
                    if is_expanded:
                        self.app.updater.expanded_commits.discard(commit_hash)
                    else:
                        self.app.updater.expanded_commits.add(commit_hash)
                        # Load changelog if not cached
                        if commit_hash not in self.app.updater.commit_changelogs:
                            try:
                                changelog = self.app.updater._get_version_diff(commit_hash)
                                self.app.updater.commit_changelogs[commit_hash] = changelog
                            except Exception as e:
                                self.app.updater.logger.error(f"Error loading changelog for {commit_hash[:7]}: {e}")
                                self.app.updater.commit_changelogs[commit_hash] = [f"Error loading changelog: {str(e)}"]
                
                imgui.same_line()
                
                # Version display - show commit message and hash
                commit_msg = version['name']
                if len(commit_msg) > 60:
                    commit_msg = commit_msg[:57] + "..."
                
                version_text = f"{commit_msg} - {commit_hash[:7]}"
                if imgui.selectable(version_text, self.app.updater.selected_version == version)[0]:
                    self.app.updater.selected_version = version
                
                if is_current:
                    imgui.pop_style_color()
                    imgui.same_line()
                    imgui.text("(Current)")
                
                # Show inline changelog if expanded
                if is_expanded:
                    imgui.indent(30)
                    imgui.push_style_color(imgui.COLOR_TEXT, *AppGUIColors.VERSION_CHANGELOG_TEXT)
                    
                    changelog = self.app.updater.commit_changelogs.get(commit_hash, [])
                    if not changelog:
                        imgui.text_wrapped("Loading changelog...")
                    else:
                        for line in changelog:
                            imgui.text_wrapped(self.app.updater.clean_text(line))
                    
                    imgui.pop_style_color()
                    imgui.unindent(30)
                    imgui.separator()

            imgui.end_child()
            imgui.separator()

            # Action buttons
            if self.app.updater.selected_version:
                if imgui.button("Switch to Commit", width=200):
                    self.app.updater.apply_version_change(self.app.updater.selected_version['commit_hash'], self.app.updater.selected_version['name'])
            else:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                imgui.button("Switch to Commit", width=200)
                imgui.pop_style_var()

    def _render_github_token_content(self):
        """Renders the GitHub token content within the tabbed dialog."""
        imgui.text("GitHub Personal Access Token")
        imgui.text_wrapped("A GitHub token increases the API rate limit from 60 to 5000 requests per hour.")
        imgui.separator()
        
        current_token = self.app.updater.token_manager.get_token()
        
        # Show current token status
        if current_token:
            masked_token = self.app.updater.token_manager.get_masked_token()
            imgui.text(f"Current token: {masked_token}")
            imgui.text_colored("✓ Token is set", *UpdateSettingsColors.TOKEN_SET)
        else:
            imgui.text_colored("No token set", *UpdateSettingsColors.TOKEN_NOT_SET)
        
        imgui.separator()
        
        # Token input field
        imgui.text("Enter GitHub Personal Access Token:")
        imgui.text_wrapped("Get a token from: GitHub → Settings → Developer settings → Personal access tokens")
        imgui.text_wrapped("Required scope: public_repo (for public repositories)")
        
        changed, self._github_token_buffer = imgui.input_text("Token", self._github_token_buffer, 100, imgui.INPUT_TEXT_PASSWORD)
        
        imgui.separator()
        
        # Buttons
        if imgui.button("Save Token", width=120):
            self.app.updater.token_manager.set_token(self._github_token_buffer)
        
        imgui.same_line()
        
        if imgui.button("Test Token", width=120):
            # Test the current token in the buffer
            test_token = self._github_token_buffer if self._github_token_buffer else self.app.updater.token_manager.get_token()
            validation_result = self.app.updater.token_manager.validate_token(test_token)
            
            if validation_result['valid']:
                imgui.open_popup("Token Validation")
                self._token_validation_result = validation_result
            else:
                imgui.open_popup("Token Validation")
                self._token_validation_result = validation_result
        
        imgui.same_line()
        
        if imgui.button("Remove Token", width=120):
            self.app.updater.token_manager.remove_token()
            self._github_token_buffer = ""
        
        # Token validation result popup
        if hasattr(self, '_token_validation_result'):
            if imgui.begin_popup_modal("Token Validation", True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
                result = self._token_validation_result
                
                if result['valid']:
                    imgui.text_colored("Token is valid!", *UpdateSettingsColors.TOKEN_VALID)
                    if result['user_info']:
                        imgui.text(f"Username: {result['user_info'].get('login', 'Unknown')}")
                else:
                    imgui.text_colored("✗ Token validation failed", *UpdateSettingsColors.TOKEN_INVALID)
                    imgui.text(result['message'])
                
                imgui.separator()
                
                if imgui.button("OK", width=100):
                    imgui.close_current_popup()
                    delattr(self, '_token_validation_result')
                
                imgui.end_popup()

    def run(self):
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
                gl.glClearColor(*AppGUIColors.BACKGROUND_CLEAR)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                self.render_gui()
                if self.app.app_settings.get("autosave_enabled", True) and time.time() - self.app.project_manager.last_autosave_time > self.app.app_settings.get("autosave_interval_seconds", 300):
                    self.app.project_manager.perform_autosave()
                self.app.energy_saver.check_and_update_energy_saver()
                glfw.swap_buffers(self.window)
                current_target_duration = target_frame_duration_energy_saver if self.app.energy_saver.energy_saver_active else target_frame_duration_normal
                elapsed_time_for_frame = time.time() - frame_start_time
                sleep_duration = current_target_duration - elapsed_time_for_frame
                # Periodic update checks
                if self.app.app_settings.get("updater_check_periodically", True) and time.time() - self.app.updater.last_check_time > 3600:  # 1 hour
                    self.app.updater.check_for_updates_async()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
        finally:
            self.app.shutdown_app()

            # --- Cleanly shut down the worker thread ---
            self.shutdown_event.set()
            # Unblock the queue in case the worker is waiting
            try:
                self.preview_task_queue.put_nowait({'type': 'shutdown'})
            except queue.Full:
                pass
            self.preview_worker_thread.join()

            if self.frame_texture_id: gl.glDeleteTextures([self.frame_texture_id]); self.frame_texture_id = 0
            if self.heatmap_texture_id: gl.glDeleteTextures([self.heatmap_texture_id]); self.heatmap_texture_id = 0
            if self.funscript_preview_texture_id: gl.glDeleteTextures(
                [self.funscript_preview_texture_id]); self.funscript_preview_texture_id = 0

            if self.impl: self.impl.shutdown()
            if self.window: glfw.destroy_window(self.window)
            glfw.terminate()
            self.app.logger.info("GUI terminated.", extra={'status_message': False})
            