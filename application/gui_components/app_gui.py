import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import numpy as np
import cv2
import time


from application.classes.gauge import GaugeWindow
from application.classes.file_dialog import ImGuiFileDialog
from application.classes.interactive_timeline import InteractiveFunscriptTimeline
from application.classes.lr_dial import LRDialWindow
from application.classes.menu import MainMenu


from application.gui_components.control_panel_ui import ControlPanelUI
from application.gui_components.video_display_ui import VideoDisplayUI
from application.gui_components.video_navigation_ui import VideoNavigationUI, ChapterListWindow
from application.gui_components.info_graphs_ui import InfoGraphsUI


from config import constants


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

        # Performance monitoring
        self.component_render_times = {}
        self.perf_log_interval = 5  # Log performance every 5 seconds
        self.last_perf_log_time = time.time()
        self.perf_frame_count = 0
        self.perf_accumulated_times = {}

        # Standard Components (owned by GUI)
        self.file_dialog = ImGuiFileDialog(logger=self.app.logger)
        self.main_menu = MainMenu(self.app)
        self.gauge_window_ui = GaugeWindow(self.app)
        self.lr_dial_window_ui = LRDialWindow(self.app)

        self.timeline_editor1 = InteractiveFunscriptTimeline(app_instance=self.app, timeline_num=1)
        self.timeline_editor2 = InteractiveFunscriptTimeline(app_instance=self.app, timeline_num=2)

        # Modularized UI Panel Components
        self.control_panel_ui = ControlPanelUI(self.app)
        self.video_display_ui = VideoDisplayUI(self.app, self)  # Pass self for texture updates
        self.video_navigation_ui = VideoNavigationUI(self.app, self)  # Pass self for texture methods
        self.info_graphs_ui = InfoGraphsUI(self.app)

        # Modularized UI Panel Components
        self.control_panel_ui = ControlPanelUI(self.app)
        self.video_display_ui = VideoDisplayUI(self.app, self)  # Pass self for texture updates
        self.video_navigation_ui = VideoNavigationUI(self.app, self)  # Pass self for texture methods
        self.info_graphs_ui = InfoGraphsUI(self.app)
        self.chapter_list_window_ui = ChapterListWindow(self.app, nav_ui=self.video_navigation_ui)

        # UI state for the dialog's radio buttons
        self.selected_batch_method_idx_ui = 0
        self.batch_overwrite_mode_ui = 0  # 0: Process All, 1: Skip Existing
        self.batch_apply_post_processing_ui = True
        self.batch_copy_funscript_to_video_location_ui = True
        self.batch_generate_roll_file_ui = True

        self.control_panel_ui.timeline_editor1 = self.timeline_editor1
        self.control_panel_ui.timeline_editor2 = self.timeline_editor2

        self.last_preview_update_time_timeline = 0.0
        self.last_preview_update_time_heatmap = 0.0
        self.preview_update_interval_seconds = 1.0

        self.last_mouse_pos_for_energy_saver = (0, 0)
        self.app.energy_saver.reset_activity_timer()

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

        self.heatmap_texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.heatmap_texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        dummy_pixel = np.array([0, 0, 0, 0], dtype=np.uint8).tobytes()
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 1, 1, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, dummy_pixel)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self.funscript_preview_texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.funscript_preview_texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        dummy_pixel_fs_preview = np.array([0, 0, 0, 0], dtype=np.uint8).tobytes()
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 1, 1, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                        dummy_pixel_fs_preview)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return True

    def handle_drop(self, window, paths):
        if not paths: return
        self.app.file_manager.handle_drop_event(paths)

    def update_texture(self, texture_id: int, image: np.ndarray):
        if image is None or image.size == 0: return
        h, w = image.shape[:2]
        if w == 0 or h == 0: return
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        if len(image.shape) == 2:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RED, w, h, 0, gl.GL_RED, gl.GL_UNSIGNED_BYTE, image)
        elif image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb_image)
        elif image.shape[2] == 4:
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba_image)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _update_funscript_preview_texture(self, target_width: int, target_height: int,
                                          total_duration_s: float):
        app_state = self.app.app_state_ui
        if target_width <= 0 or target_height <= 0 or total_duration_s <= 0.001:
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.funscript_preview_texture_id)
            w_gl, h_gl = max(1, target_width), max(1, target_height)
            clear_color_gl = np.array([0, 0, 0, 0], dtype=np.uint8)
            clear_data_gl = np.tile(clear_color_gl, (h_gl, w_gl, 1))
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w_gl, h_gl, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                            clear_data_gl)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            app_state.last_funscript_preview_bar_width = target_width
            app_state.last_funscript_preview_duration_s = total_duration_s
            app_state.last_funscript_preview_action_count = 0
            app_state.funscript_preview_dirty = False
            return

        actions = self.app.funscript_processor.get_actions('primary')  # This is now a direct reference
        bg_color_cv_bgra = (int(0.15 * 255), int(0.12 * 255), int(0.12 * 255), 255)
        preview_image_data_bgra = np.full((target_height, target_width, 4), bg_color_cv_bgra, dtype=np.uint8)
        center_y_px = target_height // 2
        cv2.line(preview_image_data_bgra, (0, center_y_px), (target_width - 1, center_y_px),
                 (int(0.3 * 255), int(0.3 * 255), int(0.3 * 255), int(0.7 * 255)), 1)

        if len(actions) >= 2:
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
                # --- COLOR CONVERSION ---
                segment_color_cv_bgra = (
                    int(segment_color_float_rgba[2] * 255),  # Blue
                    int(segment_color_float_rgba[1] * 255),  # Green
                    int(segment_color_float_rgba[0] * 255),  # Red
                    int(segment_color_float_rgba[3] * 255)  # Alpha
                )
                # --- END COLOR CONVERSION ---
                cv2.line(preview_image_data_bgra, (px1, py1), (px2, py2), segment_color_cv_bgra, thickness=1)
        elif len(actions) == 1:
            action = actions[0]
            time_s, pos_norm = action["at"] / 1000.0, action["pos"] / 100.0
            px, py = int(round((time_s / total_duration_s) * target_width)), int(
                round((1.0 - pos_norm) * target_height))
            px, py = np.clip(px, 0, target_width - 1), np.clip(py, 0, target_height - 1)

            default_color_tuple_rgba = self.app.utility.get_speed_color_from_map(constants.TIMELINE_COLOR_SPEED_STEP * 2.5)
            # --- COLOR CONVERSION ---
            point_color_cv_bgra = (
                int(default_color_tuple_rgba[2] * 255),  # Blue
                int(default_color_tuple_rgba[1] * 255),  # Green
                int(default_color_tuple_rgba[0] * 255),  # Red
                int(default_color_tuple_rgba[3] * 255)  # Alpha
            )
            # --- END COLOR CONVERSION ---
            cv2.circle(preview_image_data_bgra, (px, py), 2, point_color_cv_bgra, -1)

        preview_image_data_rgba = cv2.cvtColor(preview_image_data_bgra, cv2.COLOR_BGRA2RGBA)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.funscript_preview_texture_id)
        if app_state.last_funscript_preview_bar_width != target_width or \
                app_state.funscript_preview_texture_fixed_height != target_height:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, target_width, target_height, 0, gl.GL_RGBA,
                            gl.GL_UNSIGNED_BYTE, preview_image_data_rgba)
        else:
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, target_width, target_height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                               preview_image_data_rgba)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        app_state.funscript_preview_dirty = False
        app_state.last_funscript_preview_bar_width = target_width
        app_state.last_funscript_preview_duration_s = total_duration_s
        app_state.last_funscript_preview_action_count = len(actions)

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
            imgui.text_colored(indicator_text, 0.4, 0.9, 0.4, 1.0)  # Greenish text
            imgui.end()

    def render_funscript_timeline_preview(self, total_duration_s: float, graph_height: int):
        app_state = self.app.app_state_ui
        style = imgui.get_style()  # Get style for frame_padding

        current_bar_width_float = imgui.get_content_region_available()[0]
        current_bar_width_int = int(round(current_bar_width_float))

        if current_bar_width_int <= 0 or graph_height <= 0 or not self.funscript_preview_texture_id:
            # Ensure dummy takes the full intended width to maintain layout consistency
            imgui.dummy(current_bar_width_float if current_bar_width_float > 0 else 1,
                        graph_height + 5)  # Use current_bar_width_float
            return

        current_action_count = len(self.app.funscript_processor.get_actions('primary'))
        is_live_tracking = self.app.processor and self.app.processor.tracker and self.app.processor.tracker.tracking_active
        force_update = app_state.funscript_preview_dirty or \
                       current_bar_width_int != app_state.last_funscript_preview_bar_width or \
                       abs(total_duration_s - app_state.last_funscript_preview_duration_s) > 0.01
        action_count_changed = current_action_count != app_state.last_funscript_preview_action_count
        needs_regen = force_update or (action_count_changed and (not is_live_tracking or (
                time.time() - self.last_preview_update_time_timeline >= self.preview_update_interval_seconds)))

        if needs_regen:
            if current_bar_width_int > 0 and graph_height > 0 and total_duration_s > 0.001:
                self._update_funscript_preview_texture(current_bar_width_int, graph_height, total_duration_s)
                if is_live_tracking and action_count_changed and (
                        not force_update or app_state.funscript_preview_dirty):
                    self.last_preview_update_time_timeline = time.time()
            elif app_state.funscript_preview_dirty:  # Ensure dirty flag alone can trigger regen
                self._update_funscript_preview_texture(current_bar_width_int, graph_height, total_duration_s)
                if is_live_tracking: self.last_preview_update_time_timeline = time.time()

        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 5)
        canvas_p1_x = imgui.get_cursor_screen_pos()[0]  # This is the start of the full width element
        canvas_p1_y_offset = imgui.get_cursor_screen_pos()[1]

        if app_state.last_funscript_preview_bar_width > 0:  # Check if texture has valid dimensions from last update
            imgui.image(self.funscript_preview_texture_id, current_bar_width_float, graph_height, uv0=(0, 0),
                        uv1=(1, 1))
        else:  # Fallback if texture somehow not ready/valid but we need to occupy space
            draw_list_fallback = imgui.get_window_draw_list()
            p_min_fallback = (canvas_p1_x, canvas_p1_y_offset)
            p_max_fallback = (canvas_p1_x + current_bar_width_float, canvas_p1_y_offset + graph_height)
            bg_col_fallback = imgui.get_color_u32_rgba(0.12, 0.12, 0.15, 1.0)
            draw_list_fallback.add_rect_filled(p_min_fallback[0], p_min_fallback[1], p_max_fallback[0],
                                               p_max_fallback[1], bg_col_fallback)
            imgui.dummy(current_bar_width_float, graph_height)  # Ensure space is occupied

        # Marker Drawing Logic
        if self.app.file_manager.video_path and self.app.processor and \
                self.app.processor.video_info and self.app.processor.current_frame_index >= 0:

            total_frames = self.app.processor.video_info.get('total_frames', 0)
            current_frame_idx = self.app.processor.current_frame_index

            if total_frames > 0:  # Proceed only if we have total_frames
                normalized_pos = 0.0
                if total_frames > 1:
                    normalized_pos = current_frame_idx / (total_frames - 1.0)
                # If total_frames == 1, current_frame_idx must be 0, so normalized_pos remains 0.0

                normalized_pos = max(0.0, min(1.0, normalized_pos))  # Clamp

                # Calculate effective drawing area for the marker to align with other padded elements
                effective_timeline_start_x = canvas_p1_x + style.frame_padding[0]
                effective_timeline_width = current_bar_width_float - (style.frame_padding[0] * 2)

                if effective_timeline_width > 0:  # Ensure drawable width is positive
                    marker_x = effective_timeline_start_x + normalized_pos * effective_timeline_width
                    marker_color = imgui.get_color_u32_rgba(0.9, 0.2, 0.2, 0.85)
                    draw_list_marker = imgui.get_window_draw_list()
                    draw_list_marker.add_line(marker_x, canvas_p1_y_offset, marker_x, canvas_p1_y_offset + graph_height,
                                              marker_color, 1.0)  # Thickness 1.0, you can make it 2.0 if preferred

        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 5)  # Consistent spacing after the element

    def _update_heatmap_texture(self, target_width: int, target_height: int,
                                total_video_duration_s: float):
        app_state = self.app.app_state_ui
        if target_width <= 0 or target_height <= 0 or total_video_duration_s <= 0.001:
            # ... (Clear texture logic) ...
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.heatmap_texture_id);
            w_gl, h_gl = max(1, target_width), max(1, target_height);
            clear_color = np.array([0, 0, 0, 0], dtype=np.uint8);
            clear_data = np.tile(clear_color, (h_gl, w_gl, 1))
            if app_state.last_heatmap_bar_width != target_width or app_state.heatmap_texture_fixed_height != target_height:
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w_gl, h_gl, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                                clear_data)
            else:
                gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w_gl, h_gl, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, clear_data)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            app_state.last_heatmap_bar_width = target_width
            app_state.last_heatmap_video_duration_s = total_video_duration_s
            app_state.last_heatmap_action_count = 0
            app_state.heatmap_dirty = False
            return

        # Use get_actions
        actions = self.app.funscript_processor.get_actions('primary')
        actions_to_render = actions  # Potentially simplified later

        bg_color_heatmap_texture_rgba255 = (int(0.08 * 255), int(0.08 * 255), int(0.10 * 255), 255)
        heatmap_image_data = np.full((target_height, target_width, 4), bg_color_heatmap_texture_rgba255, dtype=np.uint8)

        if len(actions_to_render) >= 1:
            center_y_px = target_height // 2
            cv2.line(heatmap_image_data, (0, center_y_px), (target_width - 1, center_y_px), (70, 70, 70, 180), 1)

        if len(actions_to_render) >= 2:
            for i in range(len(actions_to_render) - 1):
                p1, p2 = actions_to_render[i], actions_to_render[i + 1]
                start_time_s, end_time_s = p1["at"] / 1000.0, p2["at"] / 1000.0
                if end_time_s <= start_time_s: continue
                seg_start_x_px = int(round((start_time_s / total_video_duration_s) * target_width))
                seg_end_x_px = int(round((end_time_s / total_video_duration_s) * target_width))
                seg_start_x_px = max(0, seg_start_x_px)
                seg_end_x_px = min(target_width, seg_end_x_px)
                if seg_end_x_px <= seg_start_x_px:
                    if seg_start_x_px < target_width:
                        seg_end_x_px = seg_start_x_px + 1
                    else:
                        continue
                seg_end_x_px = min(target_width, seg_end_x_px)
                delta_pos = abs(p2["pos"] - p1["pos"])
                delta_time_s_seg = (p2["at"] - p1["at"]) / 1000.0
                speed_pps = delta_pos / delta_time_s_seg if delta_time_s_seg > 0.001 else 0.0
                segment_color_float_rgba = self.app.utility.get_speed_color_from_map(speed_pps)
                segment_color_byte_rgba = np.array([int(c * 255) for c in segment_color_float_rgba], dtype=np.uint8)
                heatmap_image_data[:, seg_start_x_px:seg_end_x_px] = segment_color_byte_rgba

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.heatmap_texture_id)
        if app_state.last_heatmap_bar_width != target_width or app_state.heatmap_texture_fixed_height != target_height:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, target_width, target_height, 0, gl.GL_RGBA,
                            gl.GL_UNSIGNED_BYTE, heatmap_image_data)
        else:
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, target_width, target_height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                               heatmap_image_data)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        app_state.heatmap_dirty = False
        app_state.last_heatmap_bar_width = target_width
        app_state.last_heatmap_video_duration_s = total_video_duration_s
        app_state.last_heatmap_action_count = len(actions_to_render)

    def render_funscript_heatmap_preview(self, total_video_duration_s: float, bar_width_float: float,
                                         bar_height_float: float):  # Identical
        app_state = self.app.app_state_ui
        current_bar_width_int = int(round(bar_width_float))
        if current_bar_width_int <= 0 or app_state.heatmap_texture_fixed_height <= 0 or not self.heatmap_texture_id: imgui.dummy(
            bar_width_float, bar_height_float); return
        current_action_count = len(self.app.funscript_processor.get_actions('primary'))  # Use for_read
        is_live_tracking = self.app.processor and self.app.processor.tracker and self.app.processor.tracker.tracking_active
        force_update = app_state.heatmap_dirty or \
                       current_bar_width_int != app_state.last_heatmap_bar_width or \
                       abs(total_video_duration_s - app_state.last_heatmap_video_duration_s) > 0.01
        action_count_changed = current_action_count != app_state.last_heatmap_action_count
        needs_regen = force_update or (action_count_changed and (not is_live_tracking or (
                    time.time() - self.last_preview_update_time_heatmap >= self.preview_update_interval_seconds)))

        if needs_regen:
            if current_bar_width_int > 0 and app_state.heatmap_texture_fixed_height > 0 and total_video_duration_s > 0.001:
                self._update_heatmap_texture(current_bar_width_int, app_state.heatmap_texture_fixed_height,
                                             total_video_duration_s)
                if is_live_tracking and action_count_changed and (not force_update or app_state.heatmap_dirty):
                    self.last_preview_update_time_heatmap = time.time()
            elif app_state.heatmap_dirty:
                self._update_heatmap_texture(current_bar_width_int, app_state.heatmap_texture_fixed_height,
                                             total_video_duration_s)
                if is_live_tracking: self.last_preview_update_time_heatmap = time.time()

        if app_state.last_heatmap_bar_width > 0:
            imgui.image(self.heatmap_texture_id, bar_width_float, bar_height_float, uv0=(0, 0), uv1=(1, 1))
        else:
            draw_list = imgui.get_window_draw_list()
            cursor_screen_pos = imgui.get_cursor_screen_pos()
            p_min = cursor_screen_pos
            p_max = (cursor_screen_pos[0] + bar_width_float, cursor_screen_pos[1] + bar_height_float)
            bg_col = imgui.get_color_u32_rgba(0.1, 0.1, 0.12, 1.0)
            draw_list.add_rect_filled(p_min[0], p_min[1], p_max[0], p_max[1], bg_col)
            imgui.dummy(bar_width_float, bar_height_float)

    def _draw_fps_marks_on_slider(self, draw_list, min_rect, max_rect, current_target_fps, tracker_fps,
                                  processor_fps):
        app_state = self.app.app_state_ui
        if not imgui.is_item_visible(): return
        marks = [(current_target_fps, (255, 255, 0), "Target"), (tracker_fps, (0, 255, 0), "Tracker"),
                 (processor_fps, (255, 0, 0), "Processor")]
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

        # video_fps = self.app.processor.fps if video_loaded and self.app.processor.fps > 0 else 30.0 # Not directly used here now

        # Simplified helper function
        def check_and_run_shortcut(shortcut_name, action_func, *action_args):
            shortcut_str = current_shortcuts.get(shortcut_name)
            if not shortcut_str:
                # self.app.logger.debug(f"Shortcut '{shortcut_name}' not found in settings.")
                return False

            map_result = self.app._map_shortcut_to_glfw_key(shortcut_str)
            if not map_result:
                # self.app.logger.debug(f"Could not map shortcut string '{shortcut_str}' for '{shortcut_name}'.")
                return False

            mapped_key, mapped_mods_from_string = map_result

            # Check if the main key for the shortcut is pressed in the current frame
            if imgui.is_key_pressed(mapped_key):
                # self.app.logger.debug(f"Key {mapped_key} for '{shortcut_name}' PRESSED.")
                # self.app.logger.debug(f"  Expected mods: {mapped_mods_from_string}")
                # self.app.logger.debug(f"  Current IO mods: Ctrl={io.key_ctrl}, Alt={io.key_alt}, Shift={io.key_shift}, Super={io.key_super}")

                # Ensure CURRENT io modifier state EXACTLY matches what the shortcut string DEFINED.
                # If shortcut needs CTRL (mapped_mods_from_string['ctrl'] is True), io.key_ctrl must be True.
                # If shortcut does NOT need CTRL (mapped_mods_from_string['ctrl'] is False), io.key_ctrl must be False.
                # Same logic applies to Alt, Shift, Super.
                if (mapped_mods_from_string['ctrl'] == io.key_ctrl and
                        mapped_mods_from_string['alt'] == io.key_alt and
                        mapped_mods_from_string['shift'] == io.key_shift and
                        mapped_mods_from_string['super'] == io.key_super):
                    # self.app.logger.debug(f"Executing action for '{shortcut_name}'.")
                    action_func(*action_args)
                    return True
                # else:
                # self.app.logger.debug(f"Modifier mismatch for '{shortcut_name}'.")
            return False

        # --- Shortcut Execution Order ---
        # Try to process shortcuts. The 'elif' structure ensures only one global shortcut triggers per frame.

        # Undo/Redo (typically Ctrl/Super involved, less likely to clash with simple keys)
        if check_and_run_shortcut("undo_timeline1", fs_proc.perform_undo_redo, 1, 'undo'):
            pass
        elif check_and_run_shortcut("redo_timeline1", fs_proc.perform_undo_redo, 1, 'redo'):
            pass
        elif self.app.app_state_ui.show_funscript_interactive_timeline2 and \
                (check_and_run_shortcut("undo_timeline2", fs_proc.perform_undo_redo, 2, 'undo') or \
                 check_and_run_shortcut("redo_timeline2", fs_proc.perform_undo_redo, 2, 'redo')):
            pass
        elif check_and_run_shortcut("toggle_playback", self.app.event_handlers.handle_playback_control, "play_pause"):
            pass
        # Video Seeking (Arrow keys - ensure these are checked without specific modifiers unless defined)
        elif video_loaded:
            seek_delta_frames = 0
            processed_seek = False
            if check_and_run_shortcut("seek_prev_frame", lambda: None):  # Lambda as placeholder, action is below
                seek_delta_frames = -1
                processed_seek = True
            elif check_and_run_shortcut("seek_next_frame", lambda: None):  # Lambda as placeholder
                seek_delta_frames = 1
                processed_seek = True
            elif check_and_run_shortcut("jump_to_next_point", self.app.event_handlers.handle_jump_to_point, 'next'):
                pass
            elif check_and_run_shortcut("jump_to_prev_point", self.app.event_handlers.handle_jump_to_point, 'prev'):
                pass

            if processed_seek and seek_delta_frames != 0:
                if self.app.processor and self.app.processor.video_info:
                    new_frame = self.app.processor.current_frame_index + seek_delta_frames

                if self.app.processor and self.app.processor.video_info:
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
        if current_mouse_pos[0] != self.last_mouse_pos_for_energy_saver[0] or \
                current_mouse_pos[1] != self.last_mouse_pos_for_energy_saver[1]:
            interaction_detected_this_frame = True
            self.last_mouse_pos_for_energy_saver = current_mouse_pos
        if imgui.is_mouse_clicked(0) or imgui.is_mouse_clicked(1) or imgui.is_mouse_clicked(2) or \
                imgui.is_mouse_double_clicked(0) or imgui.is_mouse_double_clicked(1) or imgui.is_mouse_double_clicked(
            2) or \
                io.mouse_wheel != 0.0 or io.want_text_input or \
                imgui.is_mouse_dragging(0) or imgui.is_any_item_active() or imgui.is_any_item_focused() or \
                (self.file_dialog and self.file_dialog.open):
            interaction_detected_this_frame = True
        if hasattr(io, 'keys_down'):
            for i in range(len(io.keys_down)):
                if imgui.is_key_pressed(i): interaction_detected_this_frame = True; break
        if interaction_detected_this_frame:
            self.app.energy_saver.reset_activity_timer()

    def _render_batch_confirmation_dialog(self):
        """Renders a modal popup to confirm the start of a batch process."""
        app = self.app
        if app.show_batch_confirmation_dialog:
            imgui.open_popup("Batch Confirmation")
            # Center the popup on the screen
            main_viewport = imgui.get_main_viewport()
            popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                         main_viewport.pos[1] + main_viewport.size[1] * 0.5)

            imgui.set_next_window_position(popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5)

            if imgui.begin_popup_modal("Batch Confirmation", True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
                imgui.text_wrapped(app.batch_confirmation_message)
                imgui.separator()

                # Radio buttons for method selection
                imgui.text("Select Batch Processing Method:")
                if imgui.radio_button("3-Stage (Detect + Segment + Flow)", self.selected_batch_method_idx_ui == 0):
                    self.selected_batch_method_idx_ui = 0
                imgui.same_line()
                if imgui.radio_button("2-Stage (Detect + Segment Only)", self.selected_batch_method_idx_ui == 1):
                    self.selected_batch_method_idx_ui = 1

                # --- Overwrite Mode Selection ---
                imgui.separator()
                imgui.text("File Handling:")
                if imgui.radio_button("Process All Videos (Skips own matching version)",
                                      self.batch_overwrite_mode_ui == 0):
                    self.batch_overwrite_mode_ui = 0
                if imgui.radio_button("Process Only if Funscript is Missing", self.batch_overwrite_mode_ui == 1):
                    self.batch_overwrite_mode_ui = 1

                # --- Output Options ---
                imgui.separator()
                imgui.text("Output Options:")
                _, self.batch_apply_post_processing_ui = imgui.checkbox("Apply Auto Post-Processing",
                                                                        self.batch_apply_post_processing_ui)
                imgui.same_line()

                # Disable roll file generation for 2-stage mode which doesn't produce it
                is_3_stage = self.selected_batch_method_idx_ui == 0
                if not is_3_stage:
                    imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                    imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

                _, self.batch_generate_roll_file_ui = imgui.checkbox("Generate .roll file (Timeline 2)",
                                                                     self.batch_generate_roll_file_ui if is_3_stage else False)

                if not is_3_stage:
                    imgui.pop_style_var()
                    imgui.internal.pop_item_flag()

                imgui.same_line()
                _, self.batch_copy_funscript_to_video_location_ui = imgui.checkbox("Save copy next to video",
                                                                                   self.batch_copy_funscript_to_video_location_ui)

                imgui.separator()

                if imgui.button("Yes", width=100):
                    app._initiate_batch_processing_from_confirmation(
                        self.selected_batch_method_idx_ui,
                        self.batch_apply_post_processing_ui,
                        self.batch_copy_funscript_to_video_location_ui,
                        self.batch_overwrite_mode_ui,
                        self.batch_generate_roll_file_ui
                    )
                    imgui.close_current_popup()

                imgui.same_line()

                if imgui.button("No", width=100):
                    app._cancel_batch_processing_from_confirmation()
                    imgui.close_current_popup()

                imgui.end_popup()

    # --- Method to render the progress modal ---
    def _render_scene_detection_progress_modal(self):
        stage_proc = self.app.stage_processor
        if stage_proc.scene_detection_active:
            # This keeps the popup open as long as the detection is active
            imgui.open_popup("Scene Detection Progress")

        # Center the popup on the screen
        main_viewport = imgui.get_main_viewport()
        popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                     main_viewport.pos[1] + main_viewport.size[1] * 0.5)
        imgui.set_next_window_position(popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5)

        popup_flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        if imgui.begin_popup_modal("Scene Detection Progress", None, flags=popup_flags)[0]:
            imgui.text("Detecting Video Scenes...")
            imgui.text_wrapped(f"Status: {stage_proc.scene_detection_status}")

            # FIXED: Display an animated spinner if progress is not updating,
            # otherwise show the actual progress bar.
            progress = stage_proc.scene_detection_progress
            if progress > 0.001:
                imgui.progress_bar(progress, size=(250, 0), overlay=f"{progress * 100:.0f}%")
            else:
                spinner_chars = "|/-\\"
                spinner_index = int(time.time() * 4) % 4
                imgui.text(f"Processing... {spinner_chars[spinner_index]}")
                if imgui.is_item_hovered():
                    imgui.set_tooltip("The process is running. Progress reporting is currently unavailable.")

            imgui.separator()
            if imgui.button("Cancel", width=120):
                stage_proc.stop_stage_event.set()  # Signal the thread to stop
                imgui.close_current_popup()

            # Close the popup automatically once the task is no longer active
            if not stage_proc.scene_detection_active:
                imgui.close_current_popup()

            imgui.end_popup()

    def render_gui(self):
        self.component_render_times.clear()

        self._time_render("EnergySaver+Shortcuts", lambda: (
            self._handle_energy_saver_interaction_detection(),
            self._handle_global_shortcuts()
        ))

        if self.app.shortcut_manager.is_recording_shortcut_for:
            self._time_render("ShortcutRecordingInput", self.app.shortcut_manager.handle_shortcut_recording_input)
            self.app.energy_saver.reset_activity_timer()

        self._time_render("StageProcessorEvents", self.app.stage_processor.process_gui_events)

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

        # --- LAYOUT MODE SWITCH ---
        if app_state.ui_layout_mode == 'fixed':
            # --- FIXED PANEL LAYOUT ---
            panel_y_start = self.main_menu_bar_height
            num_interactive_timelines_shown = (1 if app_state.show_funscript_interactive_timeline else 0) + \
                                              (1 if app_state.show_funscript_interactive_timeline2 else 0)
            timeline1_render_h = app_state.timeline_base_height if app_state.show_funscript_interactive_timeline else 0
            timeline2_render_h = app_state.timeline_base_height if app_state.show_funscript_interactive_timeline2 else 0
            interactive_timelines_total_height = timeline1_render_h + timeline2_render_h
            available_height_for_main_panels = max(100,
                                                   self.window_height - panel_y_start - interactive_timelines_total_height)

            # Clear previous geometry to ensure it's fresh for the current layout
            app_state.fixed_layout_geometry = {}
            is_full_width_nav = getattr(app_state, 'full_width_nav', False)

            control_panel_w = 450 * font_scale
            graphs_panel_w = 450 * font_scale
            video_nav_bar_h = 150

            if is_full_width_nav:
                # --- FULL-WIDTH NAVIGATION BAR LAYOUT ---
                top_panels_h = max(50, available_height_for_main_panels - video_nav_bar_h)
                nav_y_start = panel_y_start + top_panels_h

                if app_state.show_video_display_window:
                    video_panel_w = self.window_width - control_panel_w - graphs_panel_w
                    if video_panel_w < 100:
                        video_panel_w = 100
                        graphs_panel_w = max(100, self.window_width - control_panel_w - video_panel_w)

                    video_area_x_start = control_panel_w
                    graphs_area_x_start = control_panel_w + video_panel_w

                    # Capture and Render shorter top panels
                    app_state.fixed_layout_geometry['ControlPanel'] = {'pos': (0, panel_y_start),
                                                                       'size': (control_panel_w, top_panels_h)}
                    imgui.set_next_window_position(0, panel_y_start)
                    imgui.set_next_window_size(control_panel_w, top_panels_h)
                    self._time_render("ControlPanelUI", self.control_panel_ui.render)

                    app_state.fixed_layout_geometry['VideoDisplay'] = {'pos': (video_area_x_start, panel_y_start),
                                                                       'size': (video_panel_w, top_panels_h)}
                    imgui.set_next_window_position(video_area_x_start, panel_y_start)
                    imgui.set_next_window_size(video_panel_w, top_panels_h)
                    self._time_render("VideoDisplayUI", self.video_display_ui.render)

                    app_state.fixed_layout_geometry['InfoGraphs'] = {'pos': (graphs_area_x_start, panel_y_start),
                                                                     'size': (graphs_panel_w, top_panels_h)}
                    imgui.set_next_window_position(graphs_area_x_start, panel_y_start)
                    imgui.set_next_window_size(graphs_panel_w, top_panels_h)
                    self._time_render("InfoGraphsUI", self.info_graphs_ui.render)
                else:
                    # Layout without video panel
                    control_panel_w_no_vid = self.window_width / 2
                    graphs_panel_w_no_vid = self.window_width - control_panel_w_no_vid
                    graphs_area_x_start_no_vid = control_panel_w_no_vid

                    app_state.fixed_layout_geometry['ControlPanel'] = {'pos': (0, panel_y_start),
                                                                       'size': (control_panel_w_no_vid, top_panels_h)}
                    imgui.set_next_window_position(0, panel_y_start)
                    imgui.set_next_window_size(control_panel_w_no_vid, top_panels_h)
                    self._time_render("ControlPanelUI", self.control_panel_ui.render)

                    app_state.fixed_layout_geometry['InfoGraphs'] = {'pos': (graphs_area_x_start_no_vid, panel_y_start),
                                                                     'size': (graphs_panel_w_no_vid, top_panels_h)}
                    imgui.set_next_window_position(graphs_area_x_start_no_vid, panel_y_start)
                    imgui.set_next_window_size(graphs_panel_w_no_vid, top_panels_h)
                    self._time_render("InfoGraphsUI", self.info_graphs_ui.render)

                # Capture and Render the full-width navigation bar
                app_state.fixed_layout_geometry['VideoNavigation'] = {'pos': (0, nav_y_start),
                                                                      'size': (self.window_width, video_nav_bar_h)}
                imgui.set_next_window_position(0, nav_y_start)
                imgui.set_next_window_size(self.window_width, video_nav_bar_h)
                self._time_render("VideoNavigationUI", self.video_navigation_ui.render, self.window_width)

            else:
                # --- ORIGINAL VERTICAL-COLUMN LAYOUT ---
                if app_state.show_video_display_window:
                    video_panel_w = self.window_width - control_panel_w - graphs_panel_w
                    if video_panel_w < 100:
                        video_panel_w = 100
                        graphs_panel_w = max(100, self.window_width - control_panel_w - video_panel_w)

                    video_render_h = max(50, available_height_for_main_panels - video_nav_bar_h)
                    video_area_x_start = control_panel_w
                    graphs_area_x_start = control_panel_w + video_panel_w

                    app_state.fixed_layout_geometry['ControlPanel'] = {'pos': (0, panel_y_start),
                                                                       'size': (control_panel_w,
                                                                                available_height_for_main_panels)}
                    imgui.set_next_window_position(0, panel_y_start)
                    imgui.set_next_window_size(control_panel_w, available_height_for_main_panels)
                    self._time_render("ControlPanelUI", self.control_panel_ui.render)

                    app_state.fixed_layout_geometry['VideoDisplay'] = {'pos': (video_area_x_start, panel_y_start),
                                                                       'size': (video_panel_w, video_render_h)}
                    imgui.set_next_window_position(video_area_x_start, panel_y_start)
                    imgui.set_next_window_size(video_panel_w, video_render_h)
                    self._time_render("VideoDisplayUI", self.video_display_ui.render)

                    app_state.fixed_layout_geometry['VideoNavigation'] = {
                        'pos': (video_area_x_start, panel_y_start + video_render_h),
                        'size': (video_panel_w, video_nav_bar_h)}
                    imgui.set_next_window_position(video_area_x_start, panel_y_start + video_render_h)
                    imgui.set_next_window_size(video_panel_w, video_nav_bar_h)
                    self._time_render("VideoNavigationUI", self.video_navigation_ui.render, video_panel_w)

                    app_state.fixed_layout_geometry['InfoGraphs'] = {'pos': (graphs_area_x_start, panel_y_start),
                                                                     'size': (graphs_panel_w,
                                                                              available_height_for_main_panels)}
                    imgui.set_next_window_position(graphs_area_x_start, panel_y_start)
                    imgui.set_next_window_size(graphs_panel_w, available_height_for_main_panels)
                    self._time_render("InfoGraphsUI", self.info_graphs_ui.render)
                else:
                    control_panel_w_no_vid = self.window_width / 2
                    graphs_panel_w_no_vid = self.window_width - control_panel_w_no_vid
                    graphs_area_x_start_no_vid = control_panel_w_no_vid

                    app_state.fixed_layout_geometry['ControlPanel'] = {'pos': (0, panel_y_start),
                                                                       'size': (control_panel_w_no_vid,
                                                                                available_height_for_main_panels)}
                    imgui.set_next_window_position(0, panel_y_start)
                    imgui.set_next_window_size(control_panel_w_no_vid, available_height_for_main_panels)
                    self._time_render("ControlPanelUI", self.control_panel_ui.render)

                    app_state.fixed_layout_geometry['InfoGraphs'] = {'pos': (graphs_area_x_start_no_vid, panel_y_start),
                                                                     'size': (graphs_panel_w_no_vid,
                                                                              available_height_for_main_panels)}
                    imgui.set_next_window_position(graphs_area_x_start_no_vid, panel_y_start)
                    imgui.set_next_window_size(graphs_panel_w_no_vid, available_height_for_main_panels)
                    self._time_render("InfoGraphsUI", self.info_graphs_ui.render)

            # --- RENDER TIMELINES (Common to all fixed layouts) ---
            timeline_current_y_start = panel_y_start + available_height_for_main_panels
            if app_state.show_funscript_interactive_timeline:
                app_state.fixed_layout_geometry['Timeline1'] = {'pos': (0, timeline_current_y_start),
                                                                'size': (self.window_width, timeline1_render_h)}
                self._time_render("TimelineEditor1", self.timeline_editor1.render, timeline_current_y_start,
                                  timeline1_render_h)
                timeline_current_y_start += timeline1_render_h
            if app_state.show_funscript_interactive_timeline2:
                app_state.fixed_layout_geometry['Timeline2'] = {'pos': (0, timeline_current_y_start),
                                                                'size': (self.window_width, timeline2_render_h)}
                self._time_render("TimelineEditor2", self.timeline_editor2.render, timeline_current_y_start,
                                  timeline2_render_h)

        else:  # 'floating' mode
            # --- FLOATING WINDOWS LAYOUT ---
            if app_state.just_switched_to_floating:
                # Apply geometry from the dictionary, checking if each key exists
                if 'ControlPanel' in app_state.fixed_layout_geometry:
                    geom = app_state.fixed_layout_geometry['ControlPanel']
                    imgui.set_next_window_position(geom['pos'][0], geom['pos'][1], condition=imgui.APPEARING)
                    imgui.set_next_window_size(geom['size'][0], geom['size'][1], condition=imgui.APPEARING)

                if 'VideoDisplay' in app_state.fixed_layout_geometry:
                    geom = app_state.fixed_layout_geometry['VideoDisplay']
                    imgui.set_next_window_position(geom['pos'][0], geom['pos'][1], condition=imgui.APPEARING)
                    imgui.set_next_window_size(geom['size'][0], geom['size'][1], condition=imgui.APPEARING)

                if 'VideoNavigation' in app_state.fixed_layout_geometry:
                    geom = app_state.fixed_layout_geometry['VideoNavigation']
                    imgui.set_next_window_position(geom['pos'][0], geom['pos'][1], condition=imgui.APPEARING)
                    imgui.set_next_window_size(geom['size'][0], geom['size'][1], condition=imgui.APPEARING)

                if 'InfoGraphs' in app_state.fixed_layout_geometry:
                    geom = app_state.fixed_layout_geometry['InfoGraphs']
                    imgui.set_next_window_position(geom['pos'][0], geom['pos'][1], condition=imgui.APPEARING)
                    imgui.set_next_window_size(geom['size'][0], geom['size'][1], condition=imgui.APPEARING)

                if 'Timeline1' in app_state.fixed_layout_geometry:
                    geom = app_state.fixed_layout_geometry['Timeline1']
                    imgui.set_next_window_position(geom['pos'][0], geom['pos'][1], condition=imgui.APPEARING)
                    imgui.set_next_window_size(geom['size'][0], geom['size'][1], condition=imgui.APPEARING)

                if 'Timeline2' in app_state.fixed_layout_geometry:
                    geom = app_state.fixed_layout_geometry['Timeline2']
                    imgui.set_next_window_position(geom['pos'][0], geom['pos'][1], condition=imgui.APPEARING)
                    imgui.set_next_window_size(geom['size'][0], geom['size'][1], condition=imgui.APPEARING)

            # Render all the floating windows
            self._time_render("ControlPanelUI", self.control_panel_ui.render)
            self._time_render("InfoGraphsUI", self.info_graphs_ui.render)
            self._time_render("VideoDisplayUI", self.video_display_ui.render)
            self._time_render("VideoNavigationUI", self.video_navigation_ui.render)
            self._time_render("TimelineEditor1", self.timeline_editor1.render)
            self._time_render("TimelineEditor2", self.timeline_editor2.render)

            # Reset the flag at the end of the frame
            if app_state.just_switched_to_floating:
                app_state.just_switched_to_floating = False

        # --- Common Components ---
        # This section and the new lines should be at the end of render_gui()
        if hasattr(app_state, 'show_chapter_list_window') and app_state.show_chapter_list_window:
            self._time_render("ChapterListWindow", self.chapter_list_window_ui.render)

        self._time_render("Popups", lambda: (
            self.gauge_window_ui.render(),
            self.lr_dial_window_ui.render(),
            self._render_batch_confirmation_dialog(),
            self.file_dialog.draw() if self.file_dialog.open else None,
            self._render_status_message(app_state)
        ))

        self._time_render("EnergySaverIndicator", self._render_energy_saver_indicator)

        self.perf_frame_count += 1
        if time.time() - self.last_perf_log_time > self.perf_log_interval:
            self._log_performance()

        imgui.render()
        if self.impl:
            self.impl.render(imgui.get_draw_data())

    def _render_status_message(self, app_state):  # Helper for status message
        if app_state.status_message and time.time() < app_state.status_message_time:
            imgui.set_next_window_position(self.window_width - 310, self.window_height - 40)
            imgui.begin("StatusMessage", flags=imgui.WINDOW_NO_DECORATION | imgui.WINDOW_NO_MOVE | \
                                               imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_INPUTS | \
                                               imgui.WINDOW_NO_FOCUS_ON_APPEARING | imgui.WINDOW_NO_NAV)
            imgui.text(app_state.status_message)
            imgui.end()
        elif app_state.status_message:  # Clear if expired
            app_state.status_message = ""

    def run(self):
        if not self.init_glfw(): return
        target_normal_fps = self.app.energy_saver.main_loop_normal_fps_target
        target_energy_fps = self.app.energy_saver.energy_saver_fps
        if target_normal_fps <= 0: target_normal_fps = 60
        if target_energy_fps <= 0: target_energy_fps = 1
        if target_energy_fps > target_normal_fps: target_energy_fps = target_normal_fps
        target_frame_duration_normal = 1.0 / target_normal_fps
        target_frame_duration_energy_saver = 1.0 / target_energy_fps
        glfw.swap_interval(0)  # Important for manual frame rate control

        try:
            while not glfw.window_should_close(self.window):
                frame_start_time = time.time()

                glfw.poll_events()
                if self.impl: self.impl.process_inputs()

                gl.glClearColor(0.06, 0.06, 0.06, 1)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)

                self.render_gui()

                # Autosave logic (ensure app.project_manager.perform_autosave() is efficient)
                if self.app.app_settings.get("autosave_enabled", True) and \
                        time.time() - self.app.project_manager.last_autosave_time > self.app.app_settings.get(
                    "autosave_interval_seconds", 300):
                    self.app.project_manager.perform_autosave()

                self.app.energy_saver.check_and_update_energy_saver()

                glfw.swap_buffers(self.window)

                current_target_duration = target_frame_duration_energy_saver if self.app.energy_saver.energy_saver_active else target_frame_duration_normal
                elapsed_time_for_frame = time.time() - frame_start_time
                sleep_duration = current_target_duration - elapsed_time_for_frame
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
        finally:
            self.app.shutdown_app()

            if self.frame_texture_id: gl.glDeleteTextures([self.frame_texture_id]); self.frame_texture_id = 0
            if self.heatmap_texture_id: gl.glDeleteTextures([self.heatmap_texture_id]); self.heatmap_texture_id = 0
            if self.funscript_preview_texture_id: gl.glDeleteTextures(
                [self.funscript_preview_texture_id]); self.funscript_preview_texture_id = 0

            if self.impl: self.impl.shutdown()
            if self.window: glfw.destroy_window(self.window)
            glfw.terminate()
            self.app.logger.info("GUI terminated.", extra={'status_message': False})
