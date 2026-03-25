"""Preview and heatmap generation mixin for GUI."""
import numpy as np
import cv2
import time
import queue
import imgui

from application.utils import _format_time, TaskType, TaskPriority


class PreviewManagerMixin:
    """Mixin providing preview generation and heatmap rendering methods."""

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

    def _generate_instant_tooltip_data(self, hover_time_s: float, hover_frame: int, total_duration: float, normalized_pos: float, include_frame: bool = False):
        """Generate cached tooltip data with instant funscript zoom and optional delayed video frame."""
        tooltip_data = {
            'hover_time_s': hover_time_s,
            'hover_frame': hover_frame,
            'total_duration': total_duration,
            'zoom_actions': [],
            'zoom_start_s': 0,
            'zoom_end_s': 0,
            'frame_data': None,
            'frame_loading': include_frame,  # Set to True immediately if frame will be fetched
            'actual_frame': None  # Track actual frame if different from requested
        }

        # Funscript zoom preview (±2 seconds window around hover point) - INSTANT
        zoom_window_s = 4.0  # Total window size in seconds
        zoom_start_s = max(0, hover_time_s - zoom_window_s/2)
        zoom_end_s = min(total_duration, hover_time_s + zoom_window_s/2)
        tooltip_data['zoom_start_s'] = zoom_start_s
        tooltip_data['zoom_end_s'] = zoom_end_s

        # Get funscript actions in zoom window - FAST operation
        actions = self.app.funscript_processor.get_actions('primary')
        if actions:
            zoom_actions = []
            for action in actions:
                action_time_s = action['at'] / 1000.0
                if zoom_start_s <= action_time_s <= zoom_end_s:
                    zoom_actions.append(action)
            tooltip_data['zoom_actions'] = zoom_actions

        # Frame extraction is handled asynchronously by the caller to avoid blocking UI
        # frame_loading is already set to True in dict initialization if include_frame is True

        return tooltip_data

    def _get_frame_direct_cv2(self, frame_index: int):
        """Fast direct frame extraction using video processor's existing infrastructure.
        Returns: (frame_data, actual_frame_index) or (None, None) on error
        """
        if not self.app.processor or not self.app.processor.video_path:
            return None, None

        import numpy as np

        try:
            # Use OpenCV-based thumbnail extractor for fast seeking (no FFmpeg process spawning!)
            # This is much faster than spawning FFmpeg for each tooltip hover
            # Note: This still blocks briefly for OpenCV seek, but much faster than FFmpeg
            frame = self.app.processor.get_thumbnail_frame(frame_index, use_gpu_unwarp=False)

            if frame is None:
                return None, None

            # Frame is exact, no mismatch
            actual_frame = frame_index
            # Keep frame in BGR format - update_texture will handle BGR→RGB conversion

            # Apply VR panel selection if enabled (user override controls)
            # Note: ThumbnailExtractor already crops VR to one panel (left for SBS, top for TB)
            # This section allows user to override and select right panel for SBS content
            if hasattr(self.app, 'app_settings') and self.app.app_settings:
                vr_enabled = self.app.app_settings.get('vr_mode_enabled', False)
                vr_panel = self.app.app_settings.get('vr_panel_selection', 'full')  # 'left', 'right', 'full'

                # Only apply panel selection for SBS content (not TB)
                # TB content is already cropped to top panel by ThumbnailExtractor
                vr_format = getattr(self.app.processor, 'vr_input_format', '') if self.app.processor else ''
                is_sbs = '_sbs' in vr_format.lower() or '_lr' in vr_format.lower() or '_rl' in vr_format.lower()

                if vr_enabled and vr_panel != 'full' and is_sbs:
                    height, width = frame.shape[:2]

                    if vr_panel == 'left':
                        # Take left half for preview
                        frame = frame[:, :width//2]
                    elif vr_panel == 'right':
                        # Take right half for preview
                        frame = frame[:, width//2:]

            # Crop out black padding if frame is padded (non-HD 640x640 mode only)
            height, width = frame.shape[:2]
            is_hd = self.app.processor and self.app.processor.is_hd_active
            if not is_hd and hasattr(self.app, 'app_state_ui'):
                c_l, c_t, c_r, c_b = self.app.app_state_ui.get_processing_content_uv_rect()
                if (c_l, c_t, c_r, c_b) != (0.0, 0.0, 1.0, 1.0):
                    y0, y1 = int(c_t * height), int(c_b * height)
                    x0, x1 = int(c_l * width), int(c_r * width)
                    frame = frame[y0:y1, x0:x1]
                    height, width = frame.shape[:2]

            # Resize for preview (keep aspect ratio)
            aspect_ratio = width / height if height > 0 else 16/9

            # For VR content, make preview larger since it's typically wider
            if aspect_ratio > 1.5:  # Likely VR content (wider than standard 16:9)
                preview_width = 240  # Bigger for VR
                preview_height = int(preview_width / aspect_ratio)
            else:
                preview_width = 200  # Standard content
                preview_height = int(preview_width / aspect_ratio)

            # Resize frame for preview
            frame_resized = cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)

            return frame_resized, actual_frame

        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.debug(f"Direct cv2 frame extraction failed: {e}")
            return None, None

    def _render_instant_enhanced_tooltip(self, tooltip_data: dict, show_video_frame: bool = True):
        """Render enhanced tooltip using cached data."""
        imgui.begin_tooltip()

        try:
            # Time information header
            imgui.text(f"{_format_time(self.app, tooltip_data['hover_time_s'])} / {_format_time(self.app, tooltip_data['total_duration'])}")

            # Show frame number with visual indicator if video frame is available and matching
            frame_text = f"Frame: {tooltip_data['hover_frame']}"
            has_frame_data = tooltip_data.get('frame_data') is not None and tooltip_data.get('frame_data').size > 0
            actual_frame = tooltip_data.get('actual_frame', tooltip_data['hover_frame'])
            frames_match = actual_frame == tooltip_data['hover_frame']

            if show_video_frame and has_frame_data and frames_match:
                imgui.text_colored(frame_text, 0.0, 1.0, 0.0, 1.0)  # Green = frame and video are synchronized
            elif show_video_frame and has_frame_data and not frames_match:
                imgui.text_colored(f"{frame_text} (video: {actual_frame})", 1.0, 1.0, 0.0, 1.0)  # Yellow = mismatch warning
            else:
                imgui.text(frame_text)  # Normal color = no video preview yet

            imgui.separator()

            # Funscript zoom preview
            zoom_actions = tooltip_data.get('zoom_actions', [])
            if zoom_actions:
                # Get tooltip window width for centering
                window_width = imgui.get_window_width()
                zoom_width = min(300, window_width - 20)  # Match popup width with padding
                zoom_height = 80
                draw_list = imgui.get_window_draw_list()
                graph_pos = imgui.get_cursor_screen_pos()

                # Background
                draw_list.add_rect_filled(
                    graph_pos[0], graph_pos[1],
                    graph_pos[0] + zoom_width, graph_pos[1] + zoom_height,
                    imgui.get_color_u32_rgba(0.1, 0.1, 0.1, 1.0)
                )

                # Draw funscript curve
                zoom_start_s = tooltip_data['zoom_start_s']
                zoom_end_s = tooltip_data['zoom_end_s']
                hover_time_s = tooltip_data['hover_time_s']

                if len(zoom_actions) > 1:
                    for i in range(len(zoom_actions) - 1):
                        # Calculate positions
                        t1 = (zoom_actions[i]['at'] / 1000.0 - zoom_start_s) / (zoom_end_s - zoom_start_s)
                        t2 = (zoom_actions[i+1]['at'] / 1000.0 - zoom_start_s) / (zoom_end_s - zoom_start_s)
                        y1 = 1.0 - (zoom_actions[i]['pos'] / 100.0)
                        y2 = 1.0 - (zoom_actions[i+1]['pos'] / 100.0)

                        x1 = graph_pos[0] + t1 * zoom_width
                        x2 = graph_pos[0] + t2 * zoom_width
                        py1 = graph_pos[1] + y1 * zoom_height
                        py2 = graph_pos[1] + y2 * zoom_height

                        # Draw line segment
                        draw_list.add_line(
                            x1, py1, x2, py2,
                            imgui.get_color_u32_rgba(0.2, 0.8, 0.2, 1.0),
                            2.0
                        )

                    # Draw points
                    for action in zoom_actions:
                        t = (action['at'] / 1000.0 - zoom_start_s) / (zoom_end_s - zoom_start_s)
                        y = 1.0 - (action['pos'] / 100.0)
                        x = graph_pos[0] + t * zoom_width
                        py = graph_pos[1] + y * zoom_height

                        # Highlight if near hover position
                        is_near_hover = abs(action['at'] / 1000.0 - hover_time_s) < 0.1
                        color = imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 1.0) if is_near_hover else imgui.get_color_u32_rgba(0.4, 1.0, 0.4, 1.0)
                        radius = 4 if is_near_hover else 3

                        draw_list.add_circle_filled(x, py, radius, color)

                # Draw vertical line at hover position
                hover_x = graph_pos[0] + ((hover_time_s - zoom_start_s) / (zoom_end_s - zoom_start_s)) * zoom_width
                draw_list.add_line(
                    hover_x, graph_pos[1],
                    hover_x, graph_pos[1] + zoom_height,
                    imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 0.8),
                    1.0
                )

                imgui.dummy(zoom_width, zoom_height)

            # Video frame preview (only show if requested after delay)
            if show_video_frame:
                imgui.separator()

                frame_data = tooltip_data.get('frame_data')
                frame_loading = tooltip_data.get('frame_loading', False)

                if frame_data is not None:
                    # Display cached frame using dedicated enhanced preview texture
                    if hasattr(self, 'enhanced_preview_texture_id') and self.enhanced_preview_texture_id:
                        # Only update texture once per cached tooltip data
                        if not hasattr(tooltip_data, '_frame_texture_updated'):
                            self.update_texture(self.enhanced_preview_texture_id, frame_data)
                            tooltip_data['_frame_texture_updated'] = True

                        # Calculate dimensions to fit popup width
                        frame_height, frame_width = frame_data.shape[:2]
                        window_width = imgui.get_window_width()
                        max_width = min(300, window_width - 20)  # Match graph width

                        # Scale frame to fit if needed
                        if frame_width > max_width:
                            scale = max_width / frame_width
                            display_width = max_width
                            display_height = int(frame_height * scale)
                        else:
                            display_width = frame_width
                            display_height = frame_height

                        # Center the image
                        available_width = imgui.get_content_region_available()[0]
                        if display_width < available_width:
                            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (available_width - display_width) / 2)

                        imgui.image(self.enhanced_preview_texture_id, display_width, display_height)

                        # Show VR panel info only if relevant
                        if hasattr(self.app, 'app_settings') and self.app.app_settings:
                            vr_enabled = self.app.app_settings.get('vr_mode_enabled', False)
                            vr_panel = self.app.app_settings.get('vr_panel_selection', 'full')
                            if vr_enabled and vr_panel != 'full':
                                imgui.text(f"[{vr_panel.upper()} panel]")
                    else:
                        imgui.text_disabled(f"[Frame {tooltip_data['hover_frame']} - no texture available]")
                elif frame_loading:
                    imgui.text_disabled(f"[Loading...]")
                else:
                    # More helpful error message
                    if not self.app.processor:
                        imgui.text_disabled("[No video processor available]")
                    elif not self.app.processor.video_path:
                        imgui.text_disabled("[No video loaded]")
                    else:
                        imgui.text_disabled(f"[Frame extraction failed]")

        except Exception as e:
            imgui.text(f"Preview Error: {str(e)[:50]}")
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Enhanced preview tooltip error: {e}")

        imgui.end_tooltip()
