"""ROI (Region of Interest) management for ApplicationLogic."""
from typing import Tuple


class AppROIManager:
    """Handles ROI mode operations and overlay management."""

    def __init__(self, app_logic):
        self.app = app_logic

    def enter_set_user_roi_mode(self):
        if self.app.processor and self.app.processor.is_processing:
            self.app.processor.pause_processing()  # Pause if playing/tracking
            self.app.logger.info("Video paused to set User ROI.")

        self.app.is_setting_user_roi_mode = True
        if self.app.gui_instance and hasattr(self.app.gui_instance, 'video_display_ui'):  # Reset drawing state in UI
            self.app.gui_instance.video_display_ui.is_drawing_user_roi = False
            self.app.gui_instance.video_display_ui.drawn_user_roi_video_coords = None
            self.app.gui_instance.video_display_ui.waiting_for_point_click = False

        self.app.logger.info("Setting User Defined ROI: Draw rectangle on video, then click point inside.", extra={'status_message': True, 'duration': 5.0})
        self.app.energy_saver.reset_activity_timer()

    def exit_set_user_roi_mode(self):
        self.app.is_setting_user_roi_mode = False
        if self.app.gui_instance and hasattr(self.app.gui_instance, 'video_display_ui'):
            self.app.gui_instance.video_display_ui.is_drawing_user_roi = False
            self.app.gui_instance.video_display_ui.drawn_user_roi_video_coords = None
            self.app.gui_instance.video_display_ui.waiting_for_point_click = False

    def user_roi_and_point_set(self, roi_rect_video_coords: Tuple[int, int, int, int], point_video_coords: Tuple[int, int]):
        if self.app.chapter_id_for_roi_setting:
            # --- Logic for setting chapter-specific ROI ---
            target_chapter = next((ch for ch in self.app.funscript_processor.video_chapters if ch.unique_id == self.app.chapter_id_for_roi_setting), None)
            if target_chapter:
                target_chapter.user_roi_fixed = roi_rect_video_coords

                # Calculate the point's position relative to the new ROI
                rx, ry, _, _ = roi_rect_video_coords
                px_rel = float(point_video_coords[0] - rx)
                py_rel = float(point_video_coords[1] - ry)
                target_chapter.user_roi_initial_point_relative = (px_rel, py_rel)

                self.app.logger.info(
                    f"ROI and point set for chapter: {target_chapter.position_short_name} ({target_chapter.unique_id[:8]})", extra={'status_message': True})
                self.app.project_manager.project_dirty = True
            else:
                self.app.logger.error(f"Could not find the target chapter ({self.app.chapter_id_for_roi_setting}) to set ROI.", extra={'status_message': True})

            # Reset the state variable
            self.app.chapter_id_for_roi_setting = None

        else:
            if self.app.tracker and self.app.processor:
                current_display_frame = None
                # We need the raw frame buffer that corresponds to the video_coords.
                # processor.current_frame is usually the one passed to tracker (e.g. 640x640 BGR)
                with self.app.processor.frame_lock:
                    if self.app.processor.current_frame is not None:
                        current_display_frame = self.app.processor.current_frame.copy()

                if current_display_frame is not None:
                    # Legacy USER_FIXED_ROI mode removed - ModularTrackerBridge doesn't use this mode
                    if hasattr(self.app.tracker, 'set_user_defined_roi_and_point'):
                        self.app.tracker.set_user_defined_roi_and_point(roi_rect_video_coords, point_video_coords, current_display_frame)
                        self.app.logger.info("User defined ROI and point have been set in the tracker.", extra={'status_message': True})
                    else:
                        self.app.logger.info("Current tracker doesn't support user-defined ROI functionality.", extra={'status_message': True})
                else:
                    self.app.logger.error("Could not get current frame to set user ROI patch. ROI not set.", extra={'status_message': True})
            else:
                self.app.logger.error("Tracker or Processor not available to set user ROI.", extra={'status_message': True})

        self.exit_set_user_roi_mode()
        self.app.energy_saver.reset_activity_timer()

    def clear_all_overlays_and_ui_drawings(self) -> None:
        """Clears all drawn visuals on the video regardless of current mode.
        This includes: manual ROI & point, oscillation area & grid, YOLO ROI box,
        and any in-progress UI drawing states.
        """
        # Clear tracker-side overlays/state
        if self.app.tracker and hasattr(self.app.tracker, 'clear_all_drawn_overlays'):
            self.app.tracker.clear_all_drawn_overlays()

        # Clear any UI-side drawing state (ROI/oscillation drawing in progress)
        if self.app.gui_instance and hasattr(self.app.gui_instance, 'video_display_ui'):
            vdui = self.app.gui_instance.video_display_ui
            # User ROI drawing state
            vdui.is_drawing_user_roi = False
            vdui.drawn_user_roi_video_coords = None
            vdui.waiting_for_point_click = False
            vdui.user_roi_draw_start_screen_pos = (0, 0)
            vdui.user_roi_draw_current_screen_pos = (0, 0)

            # Oscillation area drawing state
            if hasattr(vdui, 'is_drawing_oscillation_area'):
                vdui.is_drawing_oscillation_area = False
            if hasattr(vdui, 'drawn_oscillation_area_video_coords'):
                vdui.drawn_oscillation_area_video_coords = None
            if hasattr(vdui, 'waiting_for_oscillation_point_click'):
                vdui.waiting_for_oscillation_point_click = False
            if hasattr(vdui, 'oscillation_area_draw_start_screen_pos'):
                vdui.oscillation_area_draw_start_screen_pos = (0, 0)
            if hasattr(vdui, 'oscillation_area_draw_current_screen_pos'):
                vdui.oscillation_area_draw_current_screen_pos = (0, 0)

    def enter_set_oscillation_area_mode(self):
        if self.app.processor and self.app.processor.is_processing:
            self.app.processor.pause_processing()  # Pause if playing/tracking
            self.app.logger.info("Video paused to set oscillation area.")

        self.app.is_setting_oscillation_area_mode = True
        if self.app.gui_instance and hasattr(self.app.gui_instance, 'video_display_ui'):  # Reset drawing state in UI
            self.app.gui_instance.video_display_ui.is_drawing_oscillation_area = False
            self.app.gui_instance.video_display_ui.drawn_oscillation_area_video_coords = None
            self.app.gui_instance.video_display_ui.waiting_for_oscillation_point_click = False

        self.app.logger.info("Setting Oscillation Area: Draw rectangle on video to define detection region.", extra={'status_message': True, 'duration': 5.0})
        self.app.energy_saver.reset_activity_timer()

    def exit_set_oscillation_area_mode(self):
        self.app.is_setting_oscillation_area_mode = False
        if self.app.gui_instance and hasattr(self.app.gui_instance, 'video_display_ui'):
            self.app.gui_instance.video_display_ui.is_drawing_oscillation_area = False
            self.app.gui_instance.video_display_ui.drawn_oscillation_area_video_coords = None
            self.app.gui_instance.video_display_ui.waiting_for_oscillation_point_click = False
            # Clear drawing position variables to prevent showing both rectangles
            self.app.gui_instance.video_display_ui.oscillation_area_draw_start_screen_pos = (0, 0)
            self.app.gui_instance.video_display_ui.oscillation_area_draw_current_screen_pos = (0, 0)

    def oscillation_area_and_point_set(self, area_rect_video_coords: Tuple[int, int, int, int], point_video_coords: Tuple[int, int]):
        if self.app.tracker and self.app.processor:
            current_display_frame = None
            # We need the raw frame buffer that corresponds to the video_coords.
            # processor.current_frame is usually the one passed to tracker (e.g. 640x640 BGR)
            with self.app.processor.frame_lock:
                if self.app.processor.current_frame is not None:
                    current_display_frame = self.app.processor.current_frame.copy()

            if current_display_frame is not None:
                self.app.tracker.set_oscillation_area_and_point(area_rect_video_coords, point_video_coords, current_display_frame)
                self.app.logger.info("Oscillation area and point have been set in the tracker.", extra={'status_message': True})
            else:
                self.app.logger.error("Could not get current frame to set oscillation area patch. Area not set.", extra={'status_message': True})
        else:
            self.app.logger.error("Tracker or Processor not available to set oscillation area.", extra={'status_message': True})

        self.exit_set_oscillation_area_mode()
        self.app.energy_saver.reset_activity_timer()
