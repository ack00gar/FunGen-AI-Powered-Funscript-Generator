import time
import imgui
import logging
from typing import Optional, Tuple

import config.constants as constants
from config.element_group_colors import VideoDisplayColors
from application.utils import get_logo_texture_manager, get_icon_texture_manager
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.feature_detection import is_feature_available as _is_feature_available

# Module-level logger for Handy debug output (disabled by default)
_handy_debug_logger = logging.getLogger(__name__ + '.handy')



class VideoOverlaysMixin:
    """Mixin fragment for VideoDisplayUI."""

    def _render_handy_sync_overlay(self):
        """Render a small floating pill showing Handy sync state (top-right of video)."""
        # Check if Handy is connected
        if not hasattr(self.app, 'device_manager') or not self.app.device_manager:
            return
        dm = self.app.device_manager
        if not dm.is_connected():
            return

        # Find a connected Handy backend
        handy_backend = None
        try:
            from device_control.backends.handy_backend import HandyBackend
            for backend in dm.connected_devices.values():
                if isinstance(backend, HandyBackend):
                    handy_backend = backend
                    break
        except (ImportError, AttributeError):
            return
        if handy_backend is None:
            return

        # Auto-fade with same timer as playback controls
        now = time.monotonic()
        elapsed = now - self._controls_last_activity_time
        if elapsed > self._CONTROLS_HIDE_TIMEOUT:
            return
        fade_start = self._CONTROLS_HIDE_TIMEOUT - 1.0
        alpha = 1.0 - (elapsed - fade_start) / 1.0 if elapsed > fade_start else 1.0

        img_rect = self._actual_video_image_rect_on_screen
        if img_rect['w'] <= 0 or img_rect['h'] <= 0:
            return

        # Build status text
        streaming_state = dm._handy_streaming_state
        point_count = 0
        is_uploading = False
        is_streaming = False
        for state in streaming_state.values():
            is_streaming = state.get('streaming_active', False)
            is_uploading = state.get('prepared', False) is False and not state.get('setup_complete', False)

        # Get point count from the funscript data if available
        try:
            cp = self.gui_instance
            if hasattr(cp, 'device_video_integration') and cp.device_video_integration:
                actions, _ = cp.device_video_integration.get_funscript_data()
                if actions:
                    point_count = len(actions)
        except Exception:
            pass

        # Get last sync age
        sync_age_str = ""
        if handy_backend._last_sync_time > 0:
            age_s = int(now - handy_backend._last_sync_time)
            if age_s < 60:
                sync_age_str = f"Synced {age_s}s ago"
            else:
                sync_age_str = f"Synced {age_s // 60}m ago"

        # Determine status line
        processor = self.app.processor
        is_playing = processor and processor.is_processing and not getattr(processor, 'is_paused', True)

        if is_uploading:
            status = f"Handy | Uploading..."
        elif not point_count:
            status = f"Handy | No script"
        elif not is_playing:
            status = f"Handy | {point_count} pts | Paused"
        elif sync_age_str:
            status = f"Handy | {point_count} pts | {sync_age_str}"
        else:
            status = f"Handy | {point_count} pts | Ready"

        # Position: top-right of video area
        padding = 6.0
        text_size = imgui.calc_text_size(status)
        pill_w = text_size[0] + padding * 4
        pill_h = text_size[1] + padding * 2
        overlay_x = img_rect['max_x'] - pill_w - padding
        overlay_y = img_rect['min_y'] + padding

        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
                 imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS |
                 imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_FOCUS_ON_APPEARING |
                 imgui.WINDOW_NO_NAV | imgui.WINDOW_NO_BACKGROUND | imgui.WINDOW_NO_INPUTS)

        imgui.set_next_window_position(overlay_x, overlay_y)
        imgui.set_next_window_size(pill_w, pill_h)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (padding * 2, padding))

        imgui.begin("##HandySyncOverlay", flags=flags)

        draw_list = imgui.get_window_draw_list()
        win_pos = imgui.get_window_position()
        win_size = imgui.get_window_size()
        draw_list.add_rect_filled(
            win_pos[0], win_pos[1],
            win_pos[0] + win_size[0], win_pos[1] + win_size[1],
            imgui.get_color_u32_rgba(0.1, 0.1, 0.1, 0.65 * alpha),
            rounding=pill_h / 2
        )

        imgui.push_style_var(imgui.STYLE_ALPHA, alpha)
        imgui.text_colored(status, 0.5, 0.9, 0.5, alpha)
        imgui.pop_style_var()  # STYLE_ALPHA

        imgui.end()
        imgui.pop_style_var()  # STYLE_WINDOW_PADDING


    def _render_pose_skeleton(self, draw_list, pose_data: dict, is_dominant: bool):
        """Draws the skeleton, highlighting the dominant pose."""
        keypoints = pose_data.get("keypoints", [])
        if not isinstance(keypoints, list) or len(keypoints) < 17: return

        # --- Color based on whether this is the dominant pose ---
        if is_dominant:
            limb_color = imgui.get_color_u32_rgba(*VideoDisplayColors.DOMINANT_LIMB)  # Bright Green
            kpt_color = imgui.get_color_u32_rgba(*VideoDisplayColors.DOMINANT_KEYPOINT)  # Bright Orange
            thickness = 2
        else:
            limb_color = imgui.get_color_u32_rgba(*VideoDisplayColors.MUTED_LIMB)  # Muted Cyan
            kpt_color = imgui.get_color_u32_rgba(*VideoDisplayColors.MUTED_KEYPOINT)  # Muted Red
            thickness = 1

        skeleton = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 11], [6, 12], [11, 12], [5, 7], [7, 9], [6, 8], [8, 10], [11, 13], [13, 15], [12, 14], [14, 16]]

        for conn in skeleton:
            idx1, idx2 = conn
            if not (idx1 < len(keypoints) and idx2 < len(keypoints)): continue
            kp1, kp2 = keypoints[idx1], keypoints[idx2]
            if kp1[2] > 0.5 and kp2[2] > 0.5:
                p1_screen = self._video_to_screen_coords(int(kp1[0]), int(kp1[1]))
                p2_screen = self._video_to_screen_coords(int(kp2[0]), int(kp2[1]))
                if p1_screen and p2_screen:
                    draw_list.add_line(p1_screen[0], p1_screen[1], p2_screen[0], p2_screen[1], limb_color, thickness=thickness)

        for kp in keypoints:
            if kp[2] > 0.5:
                p_screen = self._video_to_screen_coords(int(kp[0]), int(kp[1]))
                if p_screen:
                    draw_list.add_circle_filled(p_screen[0], p_screen[1], 3.0, kpt_color)


    def _render_motion_mode_overlay(self, draw_list, motion_mode: Optional[str], interaction_class: Optional[str], roi_video_coords: Tuple[int, int, int, int]):
        """Renders the motion mode text (Thrusting, Riding, etc.) as an ImGui overlay."""
        if not motion_mode or motion_mode == 'undetermined':
            return

        mode_color = imgui.get_color_u32_rgba(*VideoDisplayColors.MOTION_UNDETERMINED)
        mode_text = "Undetermined"

        if motion_mode == 'thrusting':
            mode_text = "Thrusting"
            mode_color = imgui.get_color_u32_rgba(*VideoDisplayColors.MOTION_THRUSTING)
        elif motion_mode == 'riding':
            mode_color = imgui.get_color_u32_rgba(*VideoDisplayColors.MOTION_RIDING)
            if interaction_class == 'face':
                mode_text = "Blowing"
            elif interaction_class == 'hand':
                mode_text = "Stroking"
            else:
                mode_text = "Riding"

        if mode_text == "Undetermined":
            return

        # Anchor point in video coordinates: top-left of the box
        box_x, box_y, _, _ = roi_video_coords
        anchor_vid_x = box_x
        anchor_vid_y = box_y

        anchor_screen_pos = self._video_to_screen_coords(int(anchor_vid_x), int(anchor_vid_y))

        if anchor_screen_pos:
            # Position text inside the top-left corner with padding
            text_pos_x = anchor_screen_pos[0] + 5  # 5 pixels of padding from the left
            text_pos_y = anchor_screen_pos[1] + 5  # 5 pixels of padding from the top

            img_rect = self._actual_video_image_rect_on_screen
            text_size = imgui.calc_text_size(mode_text) # Calculate text size to check bounds
            if (text_pos_x + text_size[0]) < img_rect['max_x'] and (text_pos_y + text_size[1]) < img_rect['max_y']:
                draw_list.add_text(text_pos_x, text_pos_y, mode_color, mode_text)



    def _render_live_tracker_overlay(self, draw_list):
        """Renders overlays specific to the live tracker, like motion mode and live_overlay data."""
        tracker = self.app.tracker
        if not tracker:
            return

        # Existing motion mode rendering (requires ROI and tracking active)
        if tracker.tracking_active and getattr(tracker, 'roi', None):
            is_vr_video = tracker._is_vr_video() if hasattr(tracker, '_is_vr_video') else False

            if getattr(tracker, 'enable_inversion_detection', False) and is_vr_video:
                interaction_class = tracker.main_interaction_class
                roi_video_coords = tracker.roi
                motion_mode = tracker.motion_mode

                self._render_motion_mode_overlay(
                    draw_list=draw_list,
                    motion_mode=motion_mode,
                    interaction_class=interaction_class,
                    roi_video_coords=roi_video_coords
                )

        # Render live_overlay data from tracker
        overlay = getattr(tracker, 'live_overlay', None)
        if not overlay:
            return

        app_state = self.app.app_state_ui
        img_rect = self._actual_video_image_rect_on_screen
        if img_rect['w'] <= 0 or img_rect['h'] <= 0:
            return

        c_left, c_top, c_right, c_bottom = app_state.get_content_uv_rect()
        c_w, c_h = c_right - c_left, c_bottom - c_top
        buf_size = self.app.yolo_input_size
        uv_span_x = c_w / app_state.video_zoom_factor
        uv_span_y = c_h / app_state.video_zoom_factor
        if uv_span_x <= 0 or uv_span_y <= 0:
            return

        scale_x = img_rect['w'] / (uv_span_x * buf_size)
        scale_y = img_rect['h'] / (uv_span_y * buf_size)
        off_x = img_rect['min_x'] - (c_left + app_state.video_pan_normalized[0] * c_w) * buf_size * scale_x
        off_y = img_rect['min_y'] - (c_top + app_state.video_pan_normalized[1] * c_h) * buf_size * scale_y

        def to_screen(vx, vy):
            return vx * scale_x + off_x, vy * scale_y + off_y

        # Clip to video area
        draw_list.push_clip_rect(img_rect['min_x'], img_rect['min_y'], img_rect['max_x'], img_rect['max_y'], True)

        for rect in overlay.get('filled_rects', []):
            sx1, sy1 = to_screen(rect['x1'], rect['y1'])
            sx2, sy2 = to_screen(rect['x2'], rect['y2'])
            r, g, b, a = rect['color']
            draw_list.add_rect_filled(sx1, sy1, sx2, sy2, imgui.get_color_u32_rgba(r, g, b, a))

        for rect in overlay.get('rects', []):
            sx1, sy1 = to_screen(rect['x1'], rect['y1'])
            sx2, sy2 = to_screen(rect['x2'], rect['y2'])
            r, g, b, a = rect['color']
            draw_list.add_rect(sx1, sy1, sx2, sy2, imgui.get_color_u32_rgba(r, g, b, a),
                              thickness=rect.get('thickness', 1.0))
            label = rect.get('label')
            if label:
                draw_list.add_text(sx1 + 3, sy1 + 3, imgui.get_color_u32_rgba(r, g, b, a), label)

        for circle in overlay.get('circles', []):
            sx, sy = to_screen(circle['x'], circle['y'])
            r, g, b, a = circle['color']
            radius = circle.get('radius', 3.0)
            if circle.get('filled', False):
                draw_list.add_circle_filled(sx, sy, radius, imgui.get_color_u32_rgba(r, g, b, a))
            else:
                draw_list.add_circle(sx, sy, radius, imgui.get_color_u32_rgba(r, g, b, a))

        for text in overlay.get('texts', []):
            sx, sy = to_screen(text['x'], text['y'])
            r, g, b, a = text['color']
            draw_list.add_text(sx, sy, imgui.get_color_u32_rgba(r, g, b, a), text['text'])

        draw_list.pop_clip_rect()


    def _render_stage2_overlay(self, stage_proc, app_state):
        frame_overlay_data = stage_proc.stage2_overlay_data_map.get(self.app.processor.current_frame_index)
        if not frame_overlay_data: return

        current_chapter = self.app.funscript_processor.get_chapter_at_frame(self.app.processor.current_frame_index)

        draw_list = imgui.get_window_draw_list()
        img_rect = self._actual_video_image_rect_on_screen
        draw_list.push_clip_rect(img_rect['min_x'], img_rect['min_y'], img_rect['max_x'], img_rect['max_y'], True)

        dominant_pose_id = frame_overlay_data.get("dominant_pose_id")
        active_track_id = frame_overlay_data.get("active_interaction_track_id")
        is_occluded = frame_overlay_data.get("is_occluded", False)
        # Get the list of aligned fallback candidate IDs for this frame
        aligned_fallback_ids = set(frame_overlay_data.get("atr_aligned_fallback_candidate_ids", []))


        for pose in frame_overlay_data.get("poses", []):
            is_dominant = (pose.get("id") == dominant_pose_id)
            self._render_pose_skeleton(draw_list, pose, is_dominant)

        for box in frame_overlay_data.get("yolo_boxes", []):
            if not box or "bbox" not in box: continue

            p1 = self._video_to_screen_coords(box["bbox"][0], box["bbox"][1])
            p2 = self._video_to_screen_coords(box["bbox"][2], box["bbox"][3])

            if p1 and p2:
                track_id = box.get("track_id")
                is_active_interactor = (track_id is not None and track_id == active_track_id)
                is_locked_penis = (box.get("class_name") == "locked_penis")
                is_inferred_status = (box.get("status") == constants.STATUS_INFERRED_RELATIVE or box.get("status") == constants.STATUS_POSE_INFERRED)
                is_of_recovered = (box.get("status") == constants.STATUS_OF_RECOVERED or box.get("status") == constants.STATUS_OF_RECOVERED)

                # Check if this box is an aligned fallback candidate
                is_aligned_candidate = (track_id is not None and track_id in aligned_fallback_ids)

                is_refined_track = False
                if current_chapter and current_chapter.refined_track_id is not None:
                    if track_id == current_chapter.refined_track_id:
                        is_refined_track = True

                # --- HIERARCHICAL HIGHLIGHTING LOGIC ---
                if is_refined_track:
                    color = VideoDisplayColors.PERSISTENT_REFINED_TRACK  # Bright Cyan for the persistent refined track
                    thickness = 3.0
                elif is_active_interactor:
                    color = VideoDisplayColors.ACTIVE_INTERACTOR  # Bright Yellow for the ACTIVE interactor
                    thickness = 3.0
                elif is_locked_penis:
                    color = VideoDisplayColors.LOCKED_PENIS  # Bright Green for LOCKED PENIS
                    thickness = 2.0
                    # If it's a locked penis and has a visible part, draw the solid fill first.
                    if "visible_bbox" in box and box["visible_bbox"]:
                        vis_bbox = box["visible_bbox"]
                        p1_vis = self._video_to_screen_coords(vis_bbox[0], vis_bbox[1])
                        p2_vis = self._video_to_screen_coords(vis_bbox[2], vis_bbox[3])
                        if p1_vis and p2_vis:
                            # Use a semi-transparent fill of the same base color
                            fill_color = VideoDisplayColors.FILL_COLOR
                            fill_color_u32 = imgui.get_color_u32_rgba(*fill_color)
                            draw_list.add_rect_filled(p1_vis[0], p1_vis[1], p2_vis[0], p2_vis[1], fill_color_u32, rounding=2.0)
                elif is_aligned_candidate:
                    color = VideoDisplayColors.ALIGNED_FALLBACK  # Orange for ALIGNED FALLBACK candidates
                    thickness = 1.5
                elif is_inferred_status:
                    color = VideoDisplayColors.INFERRED_BOX # A distinct purple for inferred boxes
                    thickness = 1.0
                else:
                    color, thickness, _ = self.app.utility.get_box_style(box)

                color_u32 = imgui.get_color_u32_rgba(*color)
                draw_list.add_rect(p1[0], p1[1], p2[0], p2[1], color_u32, thickness=thickness, rounding=2.0)

                track_id_str = f" (id: {track_id})" if track_id is not None else ""
                label = f'{box.get("class_name", "?")}{track_id_str}'

                if is_active_interactor:
                    label += " (ACTIVE)"
                elif is_aligned_candidate:
                    label += " (Aligned)"
                elif is_inferred_status:
                    label += " (Inferred)"

                if is_of_recovered:
                    label += " [OF]"

                draw_list.add_text(p1[0] + 3, p1[1] + 3, imgui.get_color_u32_rgba(*VideoDisplayColors.BOX_LABEL), label)

        if is_occluded:
            draw_list.add_text(img_rect['min_x'] + 10, img_rect['max_y'] - 30, imgui.get_color_u32_rgba(*VideoDisplayColors.OCCLUSION_WARNING), "OCCLUSION (FALLBACK)")

        motion_mode = frame_overlay_data.get("motion_mode")
        is_vr_video = self.app.processor and self.app.processor.determined_video_type == 'VR'

        if motion_mode and is_vr_video:
            roi_to_use = None
            locked_penis_box = next((b for b in frame_overlay_data.get("yolo_boxes", []) if b.get("class_name") == "locked_penis"), None)
            if locked_penis_box and "bbox" in locked_penis_box:
                x1, y1, x2, y2 = locked_penis_box["bbox"]
                roi_to_use = (x1, y1, x2 - x1, y2 - y1)

            if roi_to_use:
                interaction_class_proxy = None
                position = frame_overlay_data.get("atr_assigned_position")
                if position:
                    if "Blowjob" in position:
                        interaction_class_proxy = "face"
                    elif "Handjob" in position:
                        interaction_class_proxy = "hand"

                self._render_motion_mode_overlay(
                    draw_list=draw_list,
                    motion_mode=motion_mode,
                    interaction_class=interaction_class_proxy,
                    roi_video_coords=roi_to_use
                )

        draw_list.pop_clip_rect()


    def _render_mixed_mode_debug_overlay(self, draw_list):
        """
        Render debug overlay for Mixed Stage 3 mode.
        Shows current processing state, ROI info, and signal source.
        """
        debug_info = None
        
        # Check if we have debug data loaded from msgpack (during video playback)
        if (hasattr(self.app, 'stage3_mixed_debug_frame_map') and 
            self.app.stage3_mixed_debug_frame_map and
            self.app.processor and hasattr(self.app.processor, 'current_frame_index')):
            
            current_frame = self.app.processor.current_frame_index
            debug_info = self.app.stage3_mixed_debug_frame_map.get(current_frame, {})
            
        # Fallback to live processor debug info (during processing)
        elif (hasattr(self.app, 'mixed_stage_processor') and self.app.mixed_stage_processor):
            debug_info = self.app.mixed_stage_processor.get_debug_info()
        
        if not debug_info:
            return
        
        # Position overlay text in top-left corner of video area
        img_rect = self._actual_video_image_rect_on_screen
        overlay_x = img_rect['min_x'] + 10
        overlay_y = img_rect['min_y'] + 10
        
        # Background for text
        text_bg_color = (0, 0, 0, 180)  # Semi-transparent black
        text_color = (255, 255, 255, 255)  # White text
        
        # Build debug text
        debug_lines = [
            f"Mixed Stage 3 Debug",
            f"Chapter: {debug_info.get('current_chapter_type', 'Unknown')}",
            f"Signal: {debug_info.get('signal_source', 'Unknown')}",
            f"Live Tracker: {'Active' if debug_info.get('live_tracker_active', False) else 'Inactive'}",
        ]
        
        # Add ROI info if available
        roi = debug_info.get('current_roi')
        if roi:
            debug_lines.append(f"ROI: ({roi[0]}, {roi[1]}) - ({roi[2]}, {roi[3]})")
            # Add ROI update info
            roi_updated = debug_info.get('roi_updated', False)
            roi_counter = debug_info.get('roi_update_counter', 0)
            debug_lines.append(f"ROI Updated: {roi_updated} (Frame #{roi_counter})")
        
        # Add oscillation details if live tracker is active
        if debug_info.get('live_tracker_active', False):
            intensity = debug_info.get('oscillation_intensity', 0.0)
            debug_lines.append(f"Oscillation: {intensity:.2f}")
            
            # Add oscillation position if available
            osc_pos = debug_info.get('oscillation_pos')
            if osc_pos is not None:
                debug_lines.append(f"Osc Pos: {osc_pos}/100")
            
            # Add EMA alpha setting
            ema_alpha = debug_info.get('ema_alpha')
            if ema_alpha is not None:
                debug_lines.append(f"EMA Alpha: {ema_alpha:.2f}")
            
            # Add last known position for debugging smoothing
            last_known = debug_info.get('oscillation_last_known')
            if last_known is not None:
                debug_lines.append(f"Last Known: {last_known:.1f}")
        
        # Add frame ID for debugging
        frame_id = debug_info.get('frame_id')
        if frame_id is not None:
            debug_lines.append(f"Frame: {frame_id}")
        
        # Render each line
        line_height = 16
        for i, line in enumerate(debug_lines):
            text_y = overlay_y + (i * line_height)
            
            # Calculate text size for background
            text_size = imgui.calc_text_size(line)
            
            # Draw background rectangle
            draw_list.add_rect_filled(
                overlay_x - 5, text_y - 2,
                overlay_x + text_size.x + 5, text_y + text_size.y + 2,
                imgui.get_color_u32_rgba(*text_bg_color)
            )
            
            # Draw text
            draw_list.add_text(
                overlay_x, text_y,
                imgui.get_color_u32_rgba(*text_color),
                line
            )
        
        # Render ROI box if available
        roi = debug_info.get('current_roi')
        if roi:
            p1 = self._video_to_screen_coords(roi[0], roi[1])
            p2 = self._video_to_screen_coords(roi[2], roi[3])
            
            if p1 and p2:
                # ROI box color based on chapter type
                chapter_type = debug_info.get('current_chapter_type', 'Other')
                if chapter_type in ['BJ', 'HJ']:
                    roi_color = (0, 255, 0, 255)  # Green for BJ/HJ (ROI tracking active)
                else:
                    roi_color = (255, 255, 0, 255)  # Yellow for other (Stage 2 signal)
                
                # Draw ROI rectangle
                draw_list.add_rect(
                    p1[0], p1[1], p2[0], p2[1],
                    imgui.get_color_u32_rgba(*roi_color),
                    thickness=2.0
                )
                
                # Add ROI label
                roi_label = f"ROI ({chapter_type})"
                draw_list.add_text(
                    p1[0], p1[1] - 20,
                    imgui.get_color_u32_rgba(*roi_color),
                    roi_label
                )
    

    def _render_component_overlays(self, app_state):
        """Render 3D simulator overlay on video display."""
        simulator_3d_overlay = self.app.app_settings.get('simulator_3d_overlay_mode', False)
        if not simulator_3d_overlay or not app_state.show_simulator_3d:
            return

        img_rect = self._actual_video_image_rect_on_screen
        if not img_rect:
            return

        self._render_simulator_3d_overlay(
            app_state, img_rect['min_x'], img_rect['min_y'],
            img_rect['max_x'], img_rect['max_y'])


    def _render_simulator_3d_overlay(self, app_state, video_min_x, video_min_y, video_max_x, video_max_y):
        """Render 3D simulator as overlay on video display (bottom-right, fixed aspect)."""
        video_width = video_max_x - video_min_x
        video_height = video_max_y - video_min_y

        # Height-based sizing with square aspect ratio to prevent stretching on widescreen
        overlay_height = int(min(video_height * 0.45, video_width * 0.35))
        overlay_width = overlay_height  # Square for 3D cylinder rendering

        # Position at bottom-right of video (aligned to corner)
        overlay_x = video_max_x - overlay_width
        overlay_y = video_max_y - overlay_height

        imgui.set_next_window_position(overlay_x, overlay_y, condition=imgui.ALWAYS)
        imgui.set_next_window_size(overlay_width, overlay_height, condition=imgui.ALWAYS)

        # Fully transparent background, no border
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_BORDER, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)

        window_flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR |
                       imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SAVED_SETTINGS |
                       imgui.WINDOW_NO_RESIZE)

        imgui.begin("3D Simulator (Overlay)##Simulator3DOverlay", flags=window_flags)

        # Get current window content size for rendering
        content_w, content_h = imgui.get_content_region_available()

        # Get simulator instance and render 3D content
        if hasattr(self.gui_instance, 'simulator_3d_window_ui'):
            simulator = self.gui_instance.simulator_3d_window_ui
            # Render the 3D content with overlay window size
            if content_w > 50 and content_h > 50:  # Minimum size check
                simulator.render_3d_content(width=int(content_w), height=int(content_h))
            else:
                imgui.text("Window too small")
        else:
            imgui.text("3D Simulator Unavailable")

        imgui.end()
        imgui.pop_style_var(1)
        imgui.pop_style_color(2)
