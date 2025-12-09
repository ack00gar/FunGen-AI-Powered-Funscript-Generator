import os
import copy
import logging
from typing import List, Dict, Optional, Tuple
from bisect import bisect_left, bisect_right
import numpy as np
from scipy.signal import correlate, find_peaks

from application.utils import VideoSegment, _format_time
from funscript import DualAxisFunscript
from config import constants
from config.constants import ChapterSource


class AppFunscriptProcessor:
    def __init__(self, app_logic):
        self.app = app_logic
        self.logger = self.app.logger if hasattr(self.app, 'logger') else logging.getLogger("AppFunscriptProcessor_fallback")

        # Chapters and Scripting Range
        self.video_chapters: List[VideoSegment] = []
        self.chapter_bar_height = 20

        # These would be managed and potentially loaded by ProjectManager
        self.selected_chapter_for_scripting: Optional[VideoSegment] = None
        self.scripting_range_active: bool = False
        self.scripting_start_frame: int = 0
        self.scripting_end_frame: int = 0

        # Funscript Attributes (stats are per timeline)
        self.funscript_stats_t1: Dict = self._get_default_funscript_stats()
        self.funscript_stats_t2: Dict = self._get_default_funscript_stats()

        # Selection state for operations (indices for the currently active_axis_for_processing)
        self.current_selection_indices: List[int] = []

        # Funscript Operations Parameters
        self.selected_axis_for_processing: str = 'primary'  # 'primary' or 'secondary'
        self.operation_target_mode: str = 'apply_to_scripting_range'  # or 'apply_to_selected_points'
        self.sg_window_length_input: int = 5
        self.sg_polyorder_input: int = 2
        self.rdp_epsilon_input: float = 8.0
        self.amplify_factor_input: float = 1.1
        self.amplify_center_input: int = 50

        # Clipboard
        self.clipboard_actions_data: List[Dict] = []

    def compare_funscript_signals(self, actions_ref: List[Dict], actions_target: List[Dict],
                                  prominence: int = 5) -> Dict:
        """
        Compares a target funscript (e.g., Stage 3) to a reference (e.g., Stage 2).

        This method uses cross-correlation on detected signal peaks to find the optimal
        time offset and gathers key comparative statistics.

        Args:
            actions_ref (List[Dict]): The reference signal with correct timing (e.g., Stage 2).
            actions_target (List[Dict]): The signal to compare and align (e.g., Stage 3).
            prominence (int): The prominence used for peak/valley detection. A higher value
                              detects only more significant strokes.

        Returns:
            Dict: A dictionary of comparison statistics, including the calculated time offset.
        """
        stats = {
            "calculated_offset_ms": 0,
            "ref_stroke_count": 0,
            "target_stroke_count": 0,
            "error": None
        }

        if not actions_ref or not actions_target:
            stats["error"] = "One or both action lists are empty."
            self.logger.warning(stats["error"])
            return stats

        # --- 1. Feature Extraction: Get Peaks and Valleys Timestamps ---
        def get_extrema_times(actions: List[Dict]) -> np.ndarray:
            if len(actions) < 3:
                return np.array([], dtype=int)

            positions = np.array([a['pos'] for a in actions])

            # Find peaks (maxima)
            peaks, _ = find_peaks(positions, prominence=prominence)

            # Find valleys (minima) by inverting the signal
            valleys, _ = find_peaks(-positions, prominence=prominence)

            # Combine, sort, and get the timestamps
            extrema_indices = np.unique(np.concatenate((peaks, valleys)))

            if len(extrema_indices) == 0:
                return np.array([], dtype=int)

            return np.array([actions[i]['at'] for i in extrema_indices], dtype=int)

        ref_extrema_times = get_extrema_times(actions_ref)
        target_extrema_times = get_extrema_times(actions_target)

        stats["ref_stroke_count"] = len(ref_extrema_times)
        stats["target_stroke_count"] = len(target_extrema_times)

        if len(ref_extrema_times) < 5 or len(target_extrema_times) < 5:
            stats["error"] = "Not enough significant peaks/valleys found to perform a reliable correlation."
            self.logger.warning(stats["error"])
            return stats

        # --- 2. Offset Calculation using Cross-Correlation ---
        # Determine the total duration for the binary signals
        duration = max(actions_ref[-1]['at'], actions_target[-1]['at']) + 1

        # Create binary event signals where '1' marks a peak/valley
        ref_signal = np.zeros(duration)
        target_signal = np.zeros(duration)
        ref_signal[ref_extrema_times] = 1
        target_signal[target_extrema_times] = 1

        # Compute the cross-correlation
        correlation = correlate(target_signal, ref_signal, mode='full', method='fft')

        # The lag is the offset from the center of the correlation array where the peak occurs
        delay_array_index = np.argmax(correlation)
        # The center of the 'full' correlation result corresponds to a lag of 0
        center_index = len(ref_signal) - 1
        lag = delay_array_index - center_index

        stats["calculated_offset_ms"] = int(lag)
        self.logger.info( f"Signal comparison complete. Calculated offset: {lag} ms. Ref strokes: {stats['ref_stroke_count']}, Target strokes: {stats['target_stroke_count']}.")

        return stats

    def get_default_ultimate_autotune_params(self) -> Dict:
        """
        Constructs a dictionary of the default parameters for the Ultimate Autotune pipeline.
        This avoids needing to instantiate a UI class in the logic layer.
        """
        # Note: These setting keys match the ones saved by the UI in interactive_timeline.py
        return {
            'presmoothing': {
                'enabled': self.app.app_settings.get("timeline1_ultimate_presmoothing_enabled", True),
                'max_window_size': self.app.app_settings.get("timeline1_ultimate_presmoothing_max_window", 15)
            },
            'peaks': {
                'enabled': self.app.app_settings.get("timeline1_ultimate_peaks_enabled", True),
                'prominence': self.app.app_settings.get("timeline1_ultimate_peaks_prominence", 10),
                'distance': 1
            },
            'recovery': {
                'enabled': self.app.app_settings.get("timeline1_ultimate_recovery_enabled", True),
                'threshold_factor': self.app.app_settings.get("timeline1_ultimate_recovery_threshold", 1.8)
            },
            'normalization': {
                'enabled': self.app.app_settings.get("timeline1_ultimate_normalization_enabled", True)
            },
            # Regeneration is disabled
            'speed_limiter': {
                'enabled': self.app.app_settings.get("timeline1_ultimate_speed_limit_enabled", True),
                'speed_threshold': self.app.app_settings.get("timeline1_ultimate_speed_threshold", 500.0)
            }
        }

    def get_chapter_at_frame(self, frame_index: int) -> Optional[VideoSegment]:
        """
        Efficiently finds the chapter that contains the given frame index.
        Returns None if the frame is not within any chapter (i.e., in a gap).
        Assumes chapters are sorted by start_frame_id.
        """
        # This is a simple linear scan. For a huge number of chapters,
        # a binary search (bisect_right) would be more efficient.
        # For typical use cases, this is fast enough and simpler.
        for chapter in self.video_chapters:
            if chapter.start_frame_id <= frame_index <= chapter.end_frame_id:
                return chapter
        return None

    def get_funscript_obj(self) -> Optional[DualAxisFunscript]:
        if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
            return self.app.processor.tracker.funscript
        self.logger.warning("Funscript object not available.")
        return None

    def _sync_chapters_to_funscript(self):
        """Sync app-level chapters to funscript object chapters."""
        try:
            funscript_obj = self.get_funscript_obj()
            if not funscript_obj:
                self.logger.debug("No funscript object available for chapter sync")
                return
                
            if not hasattr(funscript_obj, 'clear_chapters') or not hasattr(funscript_obj, 'add_chapter'):
                self.logger.warning("Funscript object missing chapter methods")
                return
                
            funscript_obj.clear_chapters()
            fps = self._get_current_fps()
            if fps <= 0:
                self.logger.warning("Invalid FPS for chapter sync, using default 30.0")
                fps = 30.0
                
            for segment in self.video_chapters:
                if hasattr(segment, 'start_frame_id') and hasattr(segment, 'end_frame_id'):
                    start_time_ms = int((segment.start_frame_id / fps) * 1000)
                    end_time_ms = int((segment.end_frame_id / fps) * 1000)
                    funscript_obj.add_chapter(
                        start_time_ms,
                        end_time_ms,
                        getattr(segment, 'position_long_name', segment.class_name),
                        getattr(segment, 'position_short_name', ''),
                        getattr(segment, 'position_long_name', '')
                    )
            self.logger.debug(f"Synced {len(self.video_chapters)} chapters to funscript object")
        except Exception as e:
            self.logger.error(f"Error syncing chapters to funscript: {e}", exc_info=True)

    def _sync_chapters_from_funscript(self):
        """Sync funscript object chapters to app-level chapters."""
        try:
            funscript_obj = self.get_funscript_obj()
            if not funscript_obj:
                self.logger.debug("No funscript object available for chapter sync")
                return
                
            if not hasattr(funscript_obj, 'chapters'):
                self.logger.debug("Funscript object has no chapters attribute")
                return
                
            if not funscript_obj.chapters:
                self.logger.debug("Funscript object has empty chapters list")
                return
                
            fps = self._get_current_fps()
            if fps <= 0:
                self.logger.warning("Invalid FPS for chapter sync, using default 30.0")
                fps = 30.0
                
            self.video_chapters = []
            for chapter in funscript_obj.chapters:
                try:
                    # Convert timestamps back to frame IDs
                    start_frame_id = int((chapter.get('start', 0) / 1000) * fps)
                    end_frame_id = int((chapter.get('end', 0) / 1000) * fps)
                    segment = VideoSegment(
                        start_frame_id=start_frame_id,
                        end_frame_id=end_frame_id,
                        class_id=None,
                        class_name=chapter.get('name', 'Unknown'),
                        segment_type='SexAct',
                        position_short_name=chapter.get('position_short', ''),
                        position_long_name=chapter.get('position_long', chapter.get('name', ''))
                    )
                    self.video_chapters.append(segment)
                except Exception as chapter_e:
                    self.logger.warning(f"Error converting chapter to VideoSegment: {chapter_e}")
                    
            self.video_chapters.sort(key=lambda c: c.start_frame_id)
            self.logger.debug(f"Synced {len(self.video_chapters)} chapters from funscript object")
        except Exception as e:
            self.logger.error(f"Error syncing chapters from funscript: {e}", exc_info=True)

    def _get_current_fps(self) -> float:
        """Get current video FPS for chapter conversions."""
        if self.app.processor:
            return self.app.processor.fps
        return 30.0  # Fallback

    def get_actions(self, axis: str) -> List[dict]:
        funscript_obj = self.get_funscript_obj()
        if funscript_obj:
            if axis == 'primary':
                return funscript_obj.primary_actions
            elif axis == 'secondary':
                return funscript_obj.secondary_actions
        return []

    def _get_default_funscript_stats(self) -> Dict:
        return {
            "source_type": "N/A", "path": "N/A", "num_points": 0,
            "duration_scripted_s": 0.0, "avg_speed_pos_per_s": 0.0,
            "avg_intensity_percent": 0.0, "min_pos": -1, "max_pos": -1,
            "avg_interval_ms": 0.0, "min_interval_ms": -1, "max_interval_ms": -1,
            "total_travel_dist": 0, "num_strokes": 0
        }

    def _get_target_funscript_object_and_axis(self, timeline_num: int) -> Tuple[Optional[object], Optional[str]]:
        """Returns the funscript object and axis name ('primary' or 'secondary')."""
        if self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript:
            funscript_obj = self.app.processor.tracker.funscript
            if timeline_num == 1:
                return funscript_obj, 'primary'
            elif timeline_num == 2:
                return funscript_obj, 'secondary'
        return None, None

    def _ensure_undo_managers_linked(self):
        """Ensures undo managers are created and linked if they weren't at init."""
        if not (self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript):
            # self.logger.warning("Cannot ensure undo manager linking: Funscript object not available.")
            return

        funscript_obj = self.app.processor.tracker.funscript
        if self.app.undo_manager_t1 is None:
            from application.classes import UndoRedoManager  # Corrected import path
            self.app.undo_manager_t1 = UndoRedoManager(max_history=50)
            self.logger.info("UndoManager T1 created dynamically.")
        if self.app.undo_manager_t1._actions_list_reference is not funscript_obj.primary_actions:
            self.app.undo_manager_t1.set_actions_reference(funscript_obj.primary_actions)
            # self.logger.debug("UndoManager T1 re-linked to primary_actions.")

        if self.app.undo_manager_t2 is None:
            from application.classes import UndoRedoManager  # Corrected import path
            self.app.undo_manager_t2 = UndoRedoManager(max_history=50)
            self.logger.info("UndoManager T2 created dynamically.")
        if self.app.undo_manager_t2._actions_list_reference is not funscript_obj.secondary_actions:
            self.app.undo_manager_t2.set_actions_reference(funscript_obj.secondary_actions)
            # self.logger.debug("UndoManager T2 re-linked to secondary_actions.")

    def _get_undo_manager(self, timeline_num: int) -> Optional[object]:  # Actually UndoRedoManager
        self._ensure_undo_managers_linked()
        if timeline_num == 1: return self.app.undo_manager_t1
        if timeline_num == 2: return self.app.undo_manager_t2
        self.logger.warning(f"Requested undo manager for invalid timeline_num: {timeline_num}")
        return None

    def _get_current_fps(self) -> float:  # Duplicated for internal use, could centralize in app_logic
        fps = 30.0
        if self.app.processor and hasattr(self.app.processor, 'video_info') and \
                self.app.processor.video_info and self.app.processor.video_info.get('fps', 0) > 0:
            fps = self.app.processor.video_info['fps']
        elif self.app.processor and hasattr(self.app.processor, 'fps') and self.app.processor.fps > 0:  # Fallback
            fps = self.app.processor.fps
        return fps

    def _check_chapter_overlap(self, start_frame: int, end_frame: int,
                               existing_chapter_id: Optional[str] = None) -> bool:
        """Checks if the given frame range overlaps with any existing chapters.
           Overlap is defined as sharing one or more frames. [s,e] includes s and e.
        """
        for chapter in self.video_chapters:
            if existing_chapter_id and chapter.unique_id == existing_chapter_id:
                continue  # Skip self when checking for an update

            # Overlap if max(start_frame, chapter.start_frame_id) <= min(end_frame, chapter.end_frame_id)
            if max(start_frame, chapter.start_frame_id) <= min(end_frame, chapter.end_frame_id):
                self.logger.warning(
                    f"Overlap detected: Proposed [{start_frame}-{end_frame}] with existing '{chapter.unique_id}' [{chapter.start_frame_id}-{chapter.end_frame_id}]")
                return True
        return False

    def _repair_overlapping_chapters(self):
        """Repair overlapping chapters from old projects before exclusive endTime fix.

        Adjusts adjacent chapters to have proper boundaries:
        - Chapter N end_frame_id should be < Chapter N+1 start_frame_id
        - Or they should be adjacent (end + 1 = start)
        """
        if len(self.video_chapters) <= 1:
            return  # Nothing to repair

        # Sort chapters by start frame
        self.video_chapters.sort(key=lambda ch: ch.start_frame_id)

        repaired_count = 0
        for i in range(len(self.video_chapters) - 1):
            curr_chapter = self.video_chapters[i]
            next_chapter = self.video_chapters[i + 1]

            # Check if chapters overlap (share frames)
            if curr_chapter.end_frame_id >= next_chapter.start_frame_id:
                # Fix: Make them adjacent by adjusting current chapter's end
                old_end = curr_chapter.end_frame_id
                curr_chapter.end_frame_id = next_chapter.start_frame_id - 1
                repaired_count += 1
                self.logger.info(
                    f"Repaired overlapping chapters: '{curr_chapter.position_short_name}' "
                    f"end adjusted from {old_end} to {curr_chapter.end_frame_id}"
                )

        if repaired_count > 0:
            self.logger.info(f"Repaired {repaired_count} overlapping chapter(s) from project load")

    def _auto_adjust_chapter_range(self, start_frame: int, end_frame: int) -> tuple[int, int]:
        """Auto-adjust chapter range to avoid overlaps, keeping as close as possible to original location."""
        if not self.video_chapters:
            return start_frame, end_frame
        
        chapters_sorted = sorted(self.video_chapters, key=lambda c: c.start_frame_id)
        original_duration = end_frame - start_frame
        
        # Find overlapping chapters
        overlapping_chapters = []
        for chapter in chapters_sorted:
            if max(start_frame, chapter.start_frame_id) <= min(end_frame, chapter.end_frame_id):
                overlapping_chapters.append(chapter)
        
        if not overlapping_chapters:
            return start_frame, end_frame  # No overlaps, keep original
        
        # Strategy 1: Try to fit right before the first overlapping chapter
        first_overlapping = min(overlapping_chapters, key=lambda c: c.start_frame_id)
        if first_overlapping.start_frame_id >= original_duration:
            adjusted_end = first_overlapping.start_frame_id - 1
            adjusted_start = adjusted_end - original_duration + 1
            if adjusted_start >= 0:
                return adjusted_start, adjusted_end
        
        # Strategy 2: Try to fit right after the last overlapping chapter  
        last_overlapping = max(overlapping_chapters, key=lambda c: c.end_frame_id)
        adjusted_start = last_overlapping.end_frame_id + 1
        adjusted_end = adjusted_start + original_duration - 1
        
        # Check if this position conflicts with any other chapters
        conflicts = False
        for chapter in chapters_sorted:
            if chapter in overlapping_chapters:
                continue  # Skip the chapters we're trying to avoid
            if max(adjusted_start, chapter.start_frame_id) <= min(adjusted_end, chapter.end_frame_id):
                conflicts = True
                break
        
        if not conflicts:
            return adjusted_start, adjusted_end
        
        # Strategy 3: Find the first available gap that can fit our duration
        for i in range(len(chapters_sorted) - 1):
            current_chapter = chapters_sorted[i]
            next_chapter = chapters_sorted[i + 1]
            
            gap_start = current_chapter.end_frame_id + 1
            gap_end = next_chapter.start_frame_id - 1
            gap_size = gap_end - gap_start + 1
            
            if gap_size >= original_duration:
                return gap_start, gap_start + original_duration - 1
        
        # Strategy 4: Place after the last chapter
        last_chapter = chapters_sorted[-1]
        final_start = last_chapter.end_frame_id + 1
        return final_start, final_start + original_duration - 1

    def _add_chapter_if_unique(self, chapter: 'VideoSegment') -> bool:
        """Add a chapter only if it doesn't duplicate an existing one. Returns True if added."""
        for existing_chapter in self.video_chapters:
            if (existing_chapter.start_frame_id == chapter.start_frame_id and 
                existing_chapter.end_frame_id == chapter.end_frame_id and
                existing_chapter.position_short_name == chapter.position_short_name):
                self.logger.debug(f"Skipping duplicate chapter at frames {chapter.start_frame_id}-{chapter.end_frame_id} ({chapter.position_short_name})")
                return False
        
        self.video_chapters.append(chapter)
        return True

    def create_new_chapter_from_data(self, data: Dict,
                                     return_chapter_object: bool = False):  # Added return_chapter_object
        self.logger.info(f"Attempting to create new chapter with data: {data}")
        new_chapter = None  # Initialize
        try:
            start_frame = int(data.get("start_frame_str", "0"))
            end_frame = int(data.get("end_frame_str", "0"))

            if start_frame < 0 or end_frame < start_frame:
                self.logger.error(f"Invalid frame range for new chapter: Start={start_frame}, End={end_frame}")
                return None if return_chapter_object else None  # Explicitly return None

            if self._check_chapter_overlap(start_frame, end_frame):
                # Auto-adjust the chapter range to avoid overlap
                original_start, original_end = start_frame, end_frame
                start_frame, end_frame = self._auto_adjust_chapter_range(start_frame, end_frame)

                # Verify adjustment was successful - check again for overlap
                if self._check_chapter_overlap(start_frame, end_frame):
                    self.logger.error(f"Cannot create chapter - no valid position found to avoid overlap. "
                                    f"Attempted range: [{original_start}-{original_end}]",
                                    extra={'status_message': True, 'duration': 4.0})
                    return None if return_chapter_object else None

                self.logger.info(f"Auto-adjusted chapter range to avoid overlap: [{original_start}-{original_end}] → [{start_frame}-{end_frame}]",
                               extra={'status_message': True})

            # Check for duplicate chapters (same frame range and position)
            pos_short_key = data.get("position_short_name_key")
            pos_info = constants.POSITION_INFO_MAPPING.get(pos_short_key, {})
            pos_short_name = pos_info.get("short_name", pos_short_key if pos_short_key else "N/A")
            
            for existing_chapter in self.video_chapters:
                if (existing_chapter.start_frame_id == start_frame and 
                    existing_chapter.end_frame_id == end_frame and
                    existing_chapter.position_short_name == pos_short_name):
                    self.logger.warning(f"Duplicate chapter detected - skipping creation of identical chapter at frames {start_frame}-{end_frame}")
                    return existing_chapter if return_chapter_object else None

            pos_long_name = pos_info.get("long_name", "Unknown Position")

            derived_class_name = pos_short_key if pos_short_key else "DefaultChapterType"

            new_chapter = VideoSegment(
                start_frame_id=start_frame,
                end_frame_id=end_frame,
                class_id=None,
                class_name=derived_class_name,
                segment_type=data.get("segment_type", "default"),
                position_short_name=pos_short_name,
                position_long_name=pos_long_name,
                source=data.get("source", ChapterSource.MANUAL.value),
                color=None
            )
            self.video_chapters.append(new_chapter)
            self.video_chapters.sort(key=lambda c: c.start_frame_id)

            # Sync chapters to funscript object
            self._sync_chapters_to_funscript()

            self.app.project_manager.project_dirty = True
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True
            self.logger.info(
                f"Successfully created new chapter: {new_chapter.unique_id} ({new_chapter.position_short_name}) with color {new_chapter.color}")
            self.app.set_status_message(f"Chapter '{new_chapter.position_short_name}' created.")
            # TODO: Add undo action
        except ValueError:
            self.logger.error("Invalid frame number format for new chapter.")
            self.app.set_status_message("Error: Frame numbers must be integers.", level=logging.ERROR)
        except Exception as e:
            self.logger.error(f"Error creating new chapter: {e}", exc_info=True)
            self.app.set_status_message("Error creating chapter.", level=logging.ERROR)

        return new_chapter if return_chapter_object else None

    def update_chapter_from_data(self, chapter_id: str, new_data: Dict):
        self.logger.info(f"Attempting to update chapter {chapter_id} with data: {new_data}")
        chapter_to_update = next((ch for ch in self.video_chapters if ch.unique_id == chapter_id), None)

        if not chapter_to_update:
            self.logger.error(f"Chapter ID {chapter_id} not found for update.")
            self.app.set_status_message("Error: Chapter not found.", level=logging.ERROR)
            return

        try:
            start_frame = int(new_data.get("start_frame_str", str(chapter_to_update.start_frame_id)))
            end_frame = int(new_data.get("end_frame_str", str(chapter_to_update.end_frame_id)))

            if start_frame < 0 or end_frame < start_frame:
                self.logger.error(f"Invalid frame range for chapter update: Start={start_frame}, End={end_frame}")
                return

            if self._check_chapter_overlap(start_frame, end_frame, chapter_id):
                self.logger.error("Updated chapter overlaps with another chapter.")
                return

            chapter_to_update.start_frame_id = start_frame
            chapter_to_update.end_frame_id = end_frame

            pos_short_key = new_data.get("position_short_name_key")
            pos_info = constants.POSITION_INFO_MAPPING.get(pos_short_key, {})
            chapter_to_update.position_short_name = pos_info.get("short_name",
                                                                 pos_short_key if pos_short_key else chapter_to_update.position_short_name)
            chapter_to_update.position_long_name = pos_info.get("long_name", chapter_to_update.position_long_name)

            # Update class_name based on the new position key
            chapter_to_update.class_name = pos_short_key if pos_short_key else "DefaultChapterType"

            chapter_to_update.segment_type = new_data.get("segment_type", chapter_to_update.segment_type)
            chapter_to_update.source = new_data.get("source", chapter_to_update.source)

            # Force color re-evaluation by calling a helper or reconstructing part of __init__'s color logic
            # This is a simplified re-evaluation. A dedicated method in VideoSegment would be cleaner.
            current_color_before_update = chapter_to_update.color
            temp_segment_for_color = VideoSegment(
                start_frame_id=chapter_to_update.start_frame_id,
                end_frame_id=chapter_to_update.end_frame_id,
                class_id=chapter_to_update.class_id,
                class_name=chapter_to_update.class_name,
                segment_type=chapter_to_update.segment_type,
                position_short_name=chapter_to_update.position_short_name,
                position_long_name=chapter_to_update.position_long_name,
                color=None
            )
            chapter_to_update.color = temp_segment_for_color.color
            if chapter_to_update.color is None:
                chapter_to_update.color = current_color_before_update if current_color_before_update else (0.5, 0.5,
                                                                                                           0.5, 0.7)

            self.video_chapters.sort(key=lambda c: c.start_frame_id)

            # Sync chapters to funscript object
            self._sync_chapters_to_funscript()

            self.app.project_manager.project_dirty = True
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True
            self.logger.info(f"Successfully updated chapter: {chapter_id}")
            self.app.set_status_message(f"Chapter '{chapter_to_update.position_short_name}' updated.")
            # TODO: Add undo action using original_data_for_undo
        except ValueError:
            self.logger.error("Invalid frame number format for chapter update.")
            self.app.set_status_message("Error: Frame numbers must be integers.", level=logging.ERROR)
        except Exception as e:
            self.logger.error(f"Error updating chapter {chapter_id}: {e}", exc_info=True)
            self.app.set_status_message("Error updating chapter.", level=logging.ERROR)

    def _record_timeline_action(self, timeline_num: int, action_description: str):
        undo_manager = self._get_undo_manager(timeline_num)
        if undo_manager:
            try:
                # Ensure reference is current (already handled by _ensure_undo_managers_linked)
                undo_manager.record_state_before_action(action_description)
                self.logger.debug(f"UndoRec: T{timeline_num} - '{action_description}'")
            except Exception as e:
                self.logger.error(f"Error recording undo for T{timeline_num} ('{action_description}'): {e}", exc_info=True)
        else:
            self.logger.warning(f"Could not record undo for T{timeline_num}: Undo manager not found.")

    def perform_undo_redo(self, timeline_num: int, operation: str):  # operation is 'undo' or 'redo'
        undo_manager = self._get_undo_manager(timeline_num)
        if not undo_manager:
            self.logger.info(f"Cannot {operation} on Timeline {timeline_num}: Manager missing.", extra={'status_message': False})
            return

        action_description = None
        success = False
        if operation == 'undo' and undo_manager.can_undo():
            action_description = undo_manager.undo()
            success = action_description is not None
        elif operation == 'redo' and undo_manager.can_redo():
            action_description = undo_manager.redo()
            success = action_description is not None

        if success:
            target_funscript, axis_name = self._get_target_funscript_object_and_axis(timeline_num)
            if target_funscript and axis_name:  # Update funscript object's internal state like last_timestamp
                actions_list = getattr(target_funscript, f"{axis_name}_actions", [])
                last_ts = actions_list[-1]['at'] if actions_list else 0
                setattr(target_funscript, f"last_timestamp_{axis_name}", last_ts)

            self._finalize_action_and_update_ui(timeline_num, f"{operation.capitalize()}: {action_description}")
            self.logger.info(f"Performed {operation.capitalize()} on Timeline {timeline_num}: {action_description}",
                             extra={'status_message': True})
            self.app.energy_saver.reset_activity_timer()
        else:
            self.logger.info(
                f"Cannot {operation} on Timeline {timeline_num}: No actions in history or operation failed.",
                extra={'status_message': False})

    def _finalize_action_and_update_ui(self, timeline_num: int, change_description: str):
        self.update_funscript_stats_for_timeline(timeline_num, change_description)
        self.app.project_manager.project_dirty = True

        # Invalidate the ultimate autotune preview for the affected timeline
        timeline_instance = None
        if self.app.gui_instance:
            if timeline_num == 1:
                timeline_instance = self.app.gui_instance.timeline_editor1
            elif timeline_num == 2:
                timeline_instance = self.app.gui_instance.timeline_editor2

        if timeline_instance:
            timeline_instance.invalidate_ultimate_preview()

            # Also invalidate editor caches to reflect changes immediately (per 9337108)
            if hasattr(timeline_instance, 'invalidate_cache'):
                try:
                    timeline_instance.invalidate_cache()
                except Exception:
                    pass
            # Also invalidate editor caches to reflect changes immediately (per commit 9337108)
            if hasattr(timeline_instance, 'invalidate_cache'):
                try:
                    timeline_instance.invalidate_cache()
                except Exception:
                    pass

        if timeline_num == 1:
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True

        # Clear selection if it was for the timeline/axis that just changed
        current_timeline_num_for_selection = 1 if self.selected_axis_for_processing == 'primary' else 2
        if timeline_num == current_timeline_num_for_selection:
            self.current_selection_indices.clear()

    def clear_timeline_history_and_set_new_baseline(self, timeline_num: int, new_actions: list,
                                                    loaded_from_description: str):
        target_funscript, axis_name = self._get_target_funscript_object_and_axis(timeline_num)
        if not target_funscript or not axis_name:
            self.logger.warning(f"Cannot clear/set baseline for T{timeline_num}: Target funscript or axis not found.")
            return

        self._record_timeline_action(timeline_num, f"Replace T{timeline_num} with: {loaded_from_description}")

        live_actions_list_attr_name = f"{axis_name}_actions"
        live_actions_list = getattr(target_funscript, live_actions_list_attr_name, None)
        if live_actions_list is None:  # Should not happen if target_funscript is valid
            self.logger.error(f"Could not get actions list '{live_actions_list_attr_name}' for T{timeline_num}")
            return

        live_actions_list.clear()
        live_actions_list.extend(copy.deepcopy(new_actions))

        last_ts_attr_name = f"last_timestamp_{axis_name}"
        setattr(target_funscript, last_ts_attr_name, new_actions[-1]['at'] if new_actions else 0)

        # Undo manager's redo stack is auto-cleared by record_state_before_action.
        # For major events, explicitly clear full history.
        if any(kw in loaded_from_description for kw in
               ["New Project", "Project Loaded", "Video Closed", "Stage 1 Pending", "Stage 2"]):
            undo_manager = self._get_undo_manager(timeline_num)
            if undo_manager:
                undo_manager.clear_history()
                # After clearing, the state recorded by _record_timeline_action IS the new baseline's undo.
                self.logger.debug(f"Undo history cleared for T{timeline_num} due to: {loaded_from_description}")

        self._finalize_action_and_update_ui(timeline_num, loaded_from_description)
        self.app.energy_saver.reset_activity_timer()

    def update_funscript_stats_for_timeline(self, timeline_num: int, source_type_description: str = "N/A"):
        stats_dict_to_update = self.funscript_stats_t1 if timeline_num == 1 else self.funscript_stats_t2
        default_app_stats = self._get_default_funscript_stats()

        for key in default_app_stats:
            stats_dict_to_update[key] = default_app_stats[key]

        stats_dict_to_update["source_type"] = source_type_description
        if "Loaded T1" in source_type_description and self.app.file_manager.loaded_funscript_path:
            stats_dict_to_update["path"] = os.path.basename(self.app.file_manager.loaded_funscript_path)
        elif "Loaded T2" in source_type_description:
            try:
                stats_dict_to_update["path"] = source_type_description.split(": ", 1)[1]
            except:
                stats_dict_to_update["path"] = "Loaded Funscript (T2)"
        elif "Stage 2" in source_type_description:
            stats_dict_to_update["path"] = "Generated by Stage 2" + (" (Secondary)" if timeline_num == 2 else "")
        elif "Live Tracker" in source_type_description:
            stats_dict_to_update["path"] = "From Live Tracker"
        # else path remains N/A

        target_funscript, axis_name = self._get_target_funscript_object_and_axis(timeline_num)
        if target_funscript and axis_name:
            core_stats = target_funscript.get_actions_statistics(axis=axis_name)
            for key, value in core_stats.items():
                if key in stats_dict_to_update:
                    stats_dict_to_update[key] = value

        if timeline_num == 1:
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True
        # self.app.project_manager.project_dirty = True # Already set by finalize or calling context

    def set_clipboard_actions(self, actions_data: List[Dict]):
        self.clipboard_actions_data = copy.deepcopy(actions_data)
        self.app.energy_saver.reset_activity_timer()
        if actions_data:
            self.logger.info(f"Copied {len(actions_data)} point(s) to clipboard.", extra={'status_message': True})
        else:
            self.logger.info("Clipboard cleared (no points selected to copy).", extra={'status_message': True})

    def get_clipboard_actions(self) -> List[Dict]:
        return copy.deepcopy(self.clipboard_actions_data)

    def clipboard_has_actions(self) -> bool:
        return bool(self.clipboard_actions_data)

    def apply_interactive_refinement(self, chapter: VideoSegment, new_actions: List[Dict]):
        """
        Applies the results from an interactive refinement session to the funscript,
        ensuring the action is recorded for undo/redo.
        """
        if not chapter or not new_actions:
            self.logger.warning("Attempted to apply refinement with invalid chapter or actions.")
            return

        # Determine the time range of the chapter to be replaced.
        start_ms = self.frame_to_ms(chapter.start_frame_id)
        end_ms = self.frame_to_ms(chapter.end_frame_id)

        # Define a clear description for the undo/redo history.
        op_desc = f"Refine Chapter: {chapter.position_short_name}"

        # The existing clear_actions_in_range_and_inject_new method already handles
        # undo recording, replacing the data, and updating the UI. We can call it directly.
        self.clear_actions_in_range_and_inject_new(
            timeline_num=1,  # Refinement currently targets the primary timeline
            new_actions_for_range=new_actions,
            range_start_ms=start_ms,
            range_end_ms=end_ms,
            operation_description=op_desc
        )

        self.logger.info(f"Successfully applied and recorded refinement for chapter '{chapter.position_short_name}'.")

    def clear_actions_in_range_and_inject_new(self, timeline_num: int,
                                              new_actions_for_range: List[Dict],
                                              range_start_ms: int, range_end_ms: int,  # Actual ms for clearing
                                              operation_description: str):
        target_funscript, axis_name = self._get_target_funscript_object_and_axis(timeline_num)
        if not target_funscript or not axis_name:
            self.logger.warning(f"Cannot clear and inject for T{timeline_num}: Target funscript or axis not found.")
            return

        self._record_timeline_action(timeline_num, operation_description)  # Record state BEFORE modification

        live_actions_list_attr_name = f"{axis_name}_actions"
        # Make a copy of current actions to work with for filtering and merging
        original_actions_copy: List[Dict] = list(
            copy.deepcopy(getattr(target_funscript, live_actions_list_attr_name, [])))

        # 1. Preserve actions outside the specified range
        actions_before_range = [action for action in original_actions_copy if action['at'] < range_start_ms]
        actions_after_range = [action for action in original_actions_copy if action['at'] > range_end_ms]

        # 2. Prepare new actions (ensure they are sorted and deepcopied)
        # Stage 2 should provide actions with absolute timestamps already correct for the range.
        processed_new_actions = sorted(copy.deepcopy(new_actions_for_range), key=lambda x: x['at'])

        # 3. Combine the three parts: before, new (for the range), after
        merged_actions = actions_before_range + processed_new_actions + actions_after_range

        # 4. Sort the final combined list by time and ensure unique timestamps
        merged_actions.sort(key=lambda x: x['at'])

        unique_final_actions = []
        if merged_actions:
            unique_final_actions.append(merged_actions[0])
            for i in range(1, len(merged_actions)):
                if merged_actions[i]['at'] > merged_actions[i - 1]['at']:
                    unique_final_actions.append(merged_actions[i])
                else:  # Timestamps are the same, potentially overwrite previous with this one if different pos.
                    if unique_final_actions and unique_final_actions[-1]['at'] == merged_actions[i]['at']:
                        unique_final_actions[-1] = merged_actions[i]


        # Update the actual live list in the funscript object
        live_actions_list_ref = getattr(target_funscript, live_actions_list_attr_name)
        live_actions_list_ref.clear()
        live_actions_list_ref.extend(unique_final_actions)

        last_ts_attr_name = f"last_timestamp_{axis_name}"
        new_last_ts = unique_final_actions[-1]['at'] if unique_final_actions else 0
        setattr(target_funscript, last_ts_attr_name, new_last_ts)

        self._finalize_action_and_update_ui(timeline_num, operation_description)
        self.logger.info(
            f"Funscript T{timeline_num}: Range [{range_start_ms}-{range_end_ms}]ms updated. Injected {len(processed_new_actions)} new. Total: {len(unique_final_actions)}.",
            extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def get_effective_scripting_range(self) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Returns if the scripting range is active and the effective start and end frames.
        -1 for scripting_end_frame is resolved to the total number of video frames if possible.
        If video info is not available, -1 for end_frame results in None (no upper bound).
        """
        if not self.scripting_range_active:
            return False, None, None

        start_f = self.scripting_start_frame
        end_f = self.scripting_end_frame

        if end_f == -1:
            if self.app.processor and self.app.processor.video_info:
                total_frames = self.app.processor.video_info.get('total_frames', 0)
                if total_frames > 0:
                    end_f = total_frames - 1  # 0-indexed
                else:
                    # No video info with total_frames, so -1 means no upper bound effectively
                    self.app.logger.warning(
                        "Scripting range end is -1, but no video total_frames info. Treating as no upper bound.")
                    end_f = None
            else:
                self.logger.warning(
                    "Scripting range end is -1, but no video processor/info. Treating as no upper bound.")
                end_f = None  # No video info, -1 means no upper bound

        # Basic validation for the range itself
        if start_f is not None and end_f is not None and start_f > end_f:
            self.logger.warning(
                f"Scripting range start_frame {start_f} is after end_frame {end_f}. Effective range will be empty for filtering.")
            pass  # Allow Stage 2 to see the invalid range and produce no points

        return True, start_f, end_f

    def frame_to_ms(self, frame_id: int) -> int:
        fps = self._get_current_fps()
        if fps > 0:
            return int(round((frame_id / fps) * 1000))
        # Try to get from Chapters FPS if video not loaded but chapters are
        if self.video_chapters and hasattr(self.video_chapters[0], 'source_fps') and self.video_chapters[
            0].source_fps > 0:
            return int(round((frame_id / self.video_chapters[0].source_fps) * 1000))
        return 0  # Fallback

    def get_script_end_time_ms(self, axis_name: str) -> int:
        actions_list = self.get_actions(axis_name)
        return actions_list[-1]['at'] if actions_list else 0

    def get_processing_args_for_operation(self) -> Tuple[Optional[int], Optional[int], Optional[List[int]]]:
        """Determines start_time, end_time, or selected_indices for a funscript operation."""
        start_time_ms: Optional[int] = None
        end_time_ms: Optional[int] = None
        selected_indices: Optional[List[int]] = None

        if self.operation_target_mode == 'apply_to_selected_points':
            if self.current_selection_indices:
                selected_indices = list(self.current_selection_indices)
            else:
                return None, None, None
        elif self.operation_target_mode == 'apply_to_scripting_range':
            if self.scripting_range_active:
                start_time_ms = self.frame_to_ms(self.scripting_start_frame)
                if self.scripting_end_frame == -1:  # Means end of video/script
                    if self.app.processor and self.app.processor.video_info and self.app.processor.video_info.get(
                            'duration', 0) > 0:
                        end_time_ms = int(self.app.processor.video_info['duration'] * 1000)
                    else:  # Fallback to script end if no video
                        end_time_ms = self.get_script_end_time_ms(self.selected_axis_for_processing)
                else:
                    end_time_ms = self.frame_to_ms(self.scripting_end_frame)

                if start_time_ms is not None and end_time_ms is not None and start_time_ms > end_time_ms:
                    end_time_ms = start_time_ms  # Ensure end is not before start
            # If not scripting_range_active, it implies full script (None, None, None for time means full)
        return start_time_ms, end_time_ms, selected_indices

    def handle_funscript_operation(self, operation_name: str):
        if not (self.app.processor and self.app.processor.tracker and self.app.processor.tracker.funscript):
            self.logger.info("Funscript processor not ready for operation.", extra={'status_message': True})
            return

        s_time, e_time, sel_idx = self.get_processing_args_for_operation()

        if self.operation_target_mode == 'apply_to_selected_points' and not sel_idx:
            self.logger.info("Operation requires selected points, but none are selected.",
                             extra={'status_message': True})
            return

        timeline_num_map = {'primary': 1, 'secondary': 2}
        timeline_num = timeline_num_map.get(self.selected_axis_for_processing)
        if timeline_num is None:
            self.logger.error("Invalid axis for processing.")
            return

        target_fs_obj, axis = self._get_target_funscript_object_and_axis(timeline_num)
        if not target_fs_obj or not axis:
            self.logger.info(f"Target funscript/axis ({self.selected_axis_for_processing}) not available.",
                             extra={'status_message': True})
            return

        action_desc_map = {
            'clamp_0': f"Clamp to 0 ({axis})", 'clamp_100': f"Clamp to 100 ({axis})",
            'invert': f"Invert values ({axis})", 'clear': f"Clear points ({axis})",
            'amplify': f"Amplify (F:{self.amplify_factor_input:.2f}, C:{self.amplify_center_input}) ({axis})",
            'apply_sg': f"Apply SG (W:{self.sg_window_length_input}, P:{self.sg_polyorder_input}) ({axis})",
            'apply_rdp': f"Apply RDP (Eps:{self.rdp_epsilon_input:.2f}) ({axis})",
            'apply_dynamic_amp': f"Apply Dyn. Amplify (Win:{self.dynamic_amp_window_ms_input}ms) ({axis})"

        }
        action_desc = action_desc_map.get(operation_name)
        if not action_desc:
            self.logger.info(f"Unknown funscript operation: {operation_name}")
            return

        # Validation for specific ops
        if operation_name == 'apply_sg':
            if self.sg_window_length_input < 3 or self.sg_window_length_input % 2 == 0:
                self.logger.info("SG: Window must be odd & >= 3.", extra={'status_message': True})
                return
            if self.sg_polyorder_input < 1 or self.sg_polyorder_input >= self.sg_window_length_input:
                self.logger.info("SG: Polyorder invalid.", extra={'status_message': True})
                return
        if operation_name == 'apply_rdp' and self.rdp_epsilon_input <= 0:
            self.logger.info("RDP: Epsilon must be > 0.", extra={'status_message': True})
            return

        self._record_timeline_action(timeline_num, action_desc)  # Record state BEFORE

        op_dispatch = {
            'clamp_0': lambda: target_fs_obj.apply_plugin('Value Clamp', axis=axis, clamp_value=0, start_time_ms=s_time, end_time_ms=e_time, selected_indices=sel_idx),
            'clamp_100': lambda: target_fs_obj.apply_plugin('Value Clamp', axis=axis, clamp_value=100, start_time_ms=s_time, end_time_ms=e_time, selected_indices=sel_idx),
            'invert': lambda: target_fs_obj.apply_plugin('Invert', axis=axis, start_time_ms=s_time, end_time_ms=e_time, selected_indices=sel_idx),
            'clear': lambda: target_fs_obj.clear_points(axis, s_time, e_time, sel_idx),
            'amplify': lambda: target_fs_obj.apply_plugin('Amplify', axis=axis, scale_factor=self.amplify_factor_input,
                                                         center_value=self.amplify_center_input, start_time_ms=s_time, end_time_ms=e_time, selected_indices=sel_idx),
            'apply_sg': lambda: target_fs_obj.apply_plugin('Savitzky-Golay Filter', axis=axis, window_length=self.sg_window_length_input,
                                                           polyorder=self.sg_polyorder_input, start_time_ms=s_time, end_time_ms=e_time, selected_indices=sel_idx),
            'apply_rdp': lambda: target_fs_obj.apply_plugin('Simplify (RDP)', axis=axis, epsilon=self.rdp_epsilon_input, start_time_ms=s_time, end_time_ms=e_time, selected_indices=sel_idx),
            'apply_dynamic_amp': lambda: target_fs_obj.apply_plugin('Dynamic Amplify', axis=axis, window_ms=self.dynamic_amp_window_ms_input,
                                                                    start_time_ms=s_time, end_time_ms=e_time, selected_indices=sel_idx)

        }
        op_func = op_dispatch.get(operation_name)
        if op_func:
            op_func()
        else:
            self.logger.error(f"Dispatch failed for {operation_name}")
            return

        self._finalize_action_and_update_ui(timeline_num, action_desc)
        self.logger.info(f"Applied: {action_desc}", extra={'status_message': True})
        self.app.energy_saver.reset_activity_timer()

    def get_time_range_ms_from_scripting_frames(self) -> Tuple[Optional[float], Optional[float]]:
        fps = self._get_current_fps()
        if not (fps > 0):
            self.logger.info("Video/FPS info needed for time range calculation.", extra={'status_message': True})
            return None, None

        start_ms = (self.scripting_start_frame / fps) * 1000.0

        eff_end_frame = self.scripting_end_frame
        if eff_end_frame == -1:  # To end of video
            eff_end_frame = (
                    self.app.processor.total_frames - 1) if self.app.processor and self.app.processor.total_frames > 0 else self.scripting_start_frame

        end_ms = (eff_end_frame / fps) * 1000.0
        if end_ms < start_ms:
            self.logger.info("Scripting range end time < start time.", extra={'status_message': True})
            return None, None  # Or (start_ms, start_ms)
        return start_ms, end_ms

    def get_scripting_range_display_text(self) -> Tuple[str, str]:
        """Returns display strings for scripting start and end frames."""
        start_display = str(self.scripting_start_frame)
        end_display = str(self.scripting_end_frame)

        if self.scripting_end_frame == -1:
            if self.app.processor and self.app.processor.total_frames > 0:
                end_display = f"{self.app.processor.total_frames - 1} (Video End)"
            elif self.get_actions('primary'):  # Check primary script if no video
                end_display = f"Script End (T1)"
            else:
                end_display = "End (No Media)"
        return start_display, end_display

    def get_operation_target_range_label(self) -> str:
        if self.operation_target_mode == 'apply_to_selected_points':
            return f"{len(self.current_selection_indices)} Selected Point(s)"
        if self.scripting_range_active:
            start_d, end_d = self.get_scripting_range_display_text()
            return f"Frames: {start_d} to {end_d}"
        return "Full Script"  # Default if not selected points and not scripting range

    def reset_scripting_range(self):
        self.scripting_range_active = False
        self.scripting_start_frame = 0
        self.scripting_end_frame = -1
        self.selected_chapter_for_scripting = None

    def update_project_specific_settings(self, project_data: Dict):
        """Called when a project is loaded to update relevant settings."""
        self.video_chapters = [VideoSegment.from_dict(data) for data in project_data.get("video_chapters", []) if
                               VideoSegment.is_valid_dict(data)]

        # Repair any overlapping chapters from old projects (before exclusive endTime fix)
        self._repair_overlapping_chapters()

        # Sync loaded chapters to funscript object
        self._sync_chapters_to_funscript()

        self.scripting_range_active = project_data.get("scripting_range_active", False)
        self.scripting_start_frame = project_data.get("scripting_start_frame", 0)
        self.scripting_end_frame = project_data.get("scripting_end_frame", -1)

        selected_chapter_id = project_data.get("selected_chapter_for_scripting_id")
        if selected_chapter_id and self.video_chapters:
            self.selected_chapter_for_scripting = next(
                (ch for ch in self.video_chapters if hasattr(ch, 'unique_id') and ch.unique_id == selected_chapter_id),
                None)
        else:
            self.selected_chapter_for_scripting = None

    def get_project_save_data(self) -> Dict:
        """Returns data from this module to be saved in a project file."""
        chapters_serializable = []
        if self.video_chapters and all(hasattr(ch, 'to_dict') for ch in self.video_chapters):  # Check all have method
            chapters_serializable = [chapter.to_dict() for chapter in self.video_chapters]
        elif self.video_chapters:
            self.logger.warning("Some VideoSegment objects lack to_dict() method. Chapters may not be fully saved.")
            chapters_serializable = [chapter.to_dict() for chapter in self.video_chapters if
                                     hasattr(chapter, 'to_dict')]

        return {
            "video_chapters": chapters_serializable,
            "scripting_range_active": self.scripting_range_active,
            "scripting_start_frame": self.scripting_start_frame,
            "scripting_end_frame": self.scripting_end_frame,
            "selected_chapter_for_scripting_id": self.selected_chapter_for_scripting.unique_id if self.selected_chapter_for_scripting and hasattr(
                self.selected_chapter_for_scripting, 'unique_id') else None,
        }

    # --- Functions for Context Menu Actions ---

    def request_create_new_chapter(self):
        # This would typically open a dialog. For now, just log.
        self.logger.info("Request to create a new chapter received (UI Dialog Needed).")
        self.app.set_status_message("Create New Chapter: Not fully implemented (needs UI dialog).")

    def request_edit_chapter(self, chapter_to_edit: VideoSegment):
        if not chapter_to_edit:
            self.logger.warning("Request to edit chapter received, but no chapter provided.")
            return
        self.logger.info(f"Request to edit chapter '{chapter_to_edit.unique_id}' received (UI Dialog Needed).")
        self.app.set_status_message(
            f"Edit Chapter {chapter_to_edit.position_short_name}: Not fully implemented (needs UI dialog).")

    def delete_video_chapters_by_ids(self, chapter_ids: List[str]):
        if not chapter_ids:
            self.logger.info("No chapter IDs provided for deletion.")
            return

        initial_count = len(self.video_chapters)
        self.video_chapters = [ch for ch in self.video_chapters if ch.unique_id not in chapter_ids]
        deleted_count = initial_count - len(self.video_chapters)

        if deleted_count > 0:
            self.logger.info(f"Deleted {deleted_count} chapter(s): {chapter_ids}")
            
            # Sync chapters to funscript object
            self._sync_chapters_to_funscript()
            
            self.app.project_manager.project_dirty = True
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True
            # TODO: Add Undo/Redo record
            self.reset_scripting_range()
            self.logger.info("Scripting range was reset after chapter deletion to prevent stale state.")

            self.app.set_status_message(f"Deleted {deleted_count} chapter(s).")
        else:
            self.logger.info(f"No chapters found matching IDs for deletion: {chapter_ids}")
            self.app.set_status_message("No matching chapters found to delete.")

    def clear_script_points_in_selected_chapters(self, selected_chapters: List[VideoSegment]):
        if not selected_chapters:
            self.logger.info("No chapters selected to clear points from.")
            return

        funscript_obj = self.get_funscript_obj()
        if not funscript_obj:
            self.logger.error("Funscript object not found. Cannot clear points.")
            return

        op_desc = f"Delete Points in {len(selected_chapters)} Chapter(s)"

        # Record the state for each timeline that has actions, as 'axis=both' affects both.
        if funscript_obj.primary_actions:
            self._record_timeline_action(1, op_desc)

        if funscript_obj.secondary_actions:
            self._record_timeline_action(2, op_desc)

        fps = self._get_current_fps()
        if fps == 30.0 and not (
                self.app.processor and hasattr(self.app.processor, 'video_info') and self.app.processor.video_info):
            self.logger.warning(
                f"Valid FPS not found, using default {fps}fps for point clearing. Accuracy may be affected.")

        cleared_any_points = False
        for chapter in selected_chapters:
            start_ms = int(round((chapter.start_frame_id / fps) * 1000.0))
            end_ms = int(round((chapter.end_frame_id / fps) * 1000.0))

            if start_ms >= end_ms:
                self.logger.warning(
                    f"Chapter {chapter.unique_id} has invalid time range for point clearing: {start_ms}ms - {end_ms}ms. Skipping.")
                continue

            self.logger.info(
                f"Clearing script points in chapter '{chapter.unique_id}' (Frames: {chapter.start_frame_id}-{chapter.end_frame_id}, Time: {start_ms}ms-{end_ms}ms)")
            funscript_obj.clear_actions_in_time_range(start_ms, end_ms, axis='both')
            cleared_any_points = True

        if cleared_any_points:
            self.app.project_manager.project_dirty = True
            self.app.app_state_ui.heatmap_dirty = True
            self.app.app_state_ui.funscript_preview_dirty = True
            self.update_funscript_stats_for_timeline(1, "Points Cleared in Chapter")
            self.update_funscript_stats_for_timeline(2, "Points Cleared in Chapter")
            self.app.set_status_message(f"Cleared script points in {len(selected_chapters)} chapter(s).")

    def merge_selected_chapters(self, chapter1: VideoSegment, chapter2: VideoSegment,
                                return_chapter_object: bool = False):
        if not chapter1 or not chapter2:
            self.logger.error("Two chapters must be provided for merging.")
            return
        if chapter1.unique_id == chapter2.unique_id:
            self.logger.warning("Cannot merge a chapter with itself.")
            return

        # Ensure chapter1 is the earlier one
        if chapter1.start_frame_id > chapter2.start_frame_id:
            chapter1, chapter2 = chapter2, chapter1  # Ensure chapter1 is earlier

        new_start_frame = chapter1.start_frame_id
        new_end_frame = max(chapter1.end_frame_id, chapter2.end_frame_id)

        # Check for overlap with *other* chapters BEFORE creating the new one
        # Exclude the two chapters being merged from this specific check
        ids_to_ignore_for_overlap_check = {chapter1.unique_id, chapter2.unique_id}
        temp_chapters_for_check = [ch for ch in self.video_chapters if
                                   ch.unique_id not in ids_to_ignore_for_overlap_check]

        for other_ch in temp_chapters_for_check:
            if max(new_start_frame, other_ch.start_frame_id) <= min(new_end_frame, other_ch.end_frame_id):
                self.logger.error(
                    f"Merge failed: Resulting chapter [{new_start_frame}-{new_end_frame}] would overlap with existing chapter '{other_ch.unique_id}' [{other_ch.start_frame_id}-{other_ch.end_frame_id}].")
                self.app.set_status_message("Error: Merge would cause overlap with another chapter.",
                                            level=logging.ERROR)
                return

        merged_pos_short_key = chapter1.position_short_name
        merged_pos_info = constants.POSITION_INFO_MAPPING.get(merged_pos_short_key, {})
        merged_pos_short_name = chapter1.position_short_name # Set directly from chapter1
        merged_pos_long_name = chapter1.position_long_name # Set directly from chapter1

        # Derived class_name from position key
        merged_derived_class_name = merged_pos_short_key if merged_pos_short_key else "MergedChapter"

        merged_chapter = VideoSegment(
            start_frame_id=new_start_frame,
            end_frame_id=new_end_frame,
            class_id=chapter1.class_id,
            class_name=merged_derived_class_name,
            segment_type=chapter1.segment_type,
            position_short_name=merged_pos_short_name,
            position_long_name=merged_pos_long_name,
            source=ChapterSource.MANUAL_MERGE.value,
            color=chapter1.color
        )

        # Duration will be calculated by VideoSegment if not passed, or we can set it
        merged_chapter.duration = new_end_frame - new_start_frame

        ids_to_delete = {chapter1.unique_id, chapter2.unique_id}
        self.video_chapters = [ch for ch in self.video_chapters if ch.unique_id not in ids_to_delete]
        self.video_chapters.append(merged_chapter)
        self.video_chapters.sort(key=lambda c: c.start_frame_id)

        # Sync chapters to funscript object
        self._sync_chapters_to_funscript()

        self.logger.info(
            f"Merged chapters '{chapter1.unique_id}' and '{chapter2.unique_id}' into new chapter '{merged_chapter.unique_id}'.")
        self.app.project_manager.project_dirty = True
        self.app.app_state_ui.heatmap_dirty = True
        self.app.app_state_ui.funscript_preview_dirty = True
        # TODO: Add Undo/Redo record
        self.app.set_status_message("Chapters merged successfully.")
        self.reset_scripting_range()
        self.app.set_status_message("Chapters merged. Scripting range cleared.")

        return merged_chapter if return_chapter_object else None

    def finalize_merge_after_gap_tracking(self, chapter1_id: str, chapter2_id: str):
        self.logger.info(f"Finalizing merge after gap tracking for chapters: {chapter1_id}, {chapter2_id}")

        chapter1 = next((ch for ch in self.video_chapters if ch.unique_id == chapter1_id), None)
        chapter2 = next((ch for ch in self.video_chapters if ch.unique_id == chapter2_id), None)

        if not chapter1 or not chapter2:
            self.logger.error(f"Could not find original chapters for final merge: C1={chapter1_id}, C2={chapter2_id}")
            # Attempt to clean up funscript even if chapters are gone, though unlikely.
            funscript_obj_check = self.get_funscript_obj()
            if funscript_obj_check:
                funscript_obj_check.primary_actions.sort(key=lambda x: x['at'])  # Basic sort at least
            return

        funscript_obj = self.get_funscript_obj()
        if not funscript_obj:
            self.logger.error("Funscript object not available for finalizing merge.")
            return

        actions_list_ref = funscript_obj.primary_actions

        if actions_list_ref:
            actions_list_ref.sort(key=lambda x: x['at'])
            unique_actions = []
            if actions_list_ref:
                unique_actions.append(actions_list_ref[0])
                for i in range(1, len(actions_list_ref)):
                    if actions_list_ref[i]['at'] > unique_actions[-1]['at']:
                        unique_actions.append(actions_list_ref[i])

            actions_list_ref[:] = unique_actions  # Update in place

            if funscript_obj.primary_actions:
                funscript_obj.last_timestamp_primary = funscript_obj.primary_actions[-1]['at']
            else:
                funscript_obj.last_timestamp_primary = 0
        # Similar logic if secondary axis was tracked for the gap.

        # Now, perform the chapter merge logic (similar to merge_chapters_across_gap or merge_selected_chapters)
        # This will create one new chapter spanning the old C1, the gap, and the old C2.

        # Ensure chapter1 is the earlier one for definition
        if chapter1.start_frame_id > chapter2.start_frame_id:
            chapter1, chapter2 = chapter2, chapter1

        new_start_frame = chapter1.start_frame_id
        new_end_frame = chapter2.end_frame_id

        # --- Instead of aborting on overlap, collect all chapters to be replaced ---
        ids_to_delete = {chapter1_id, chapter2_id}
        chapters_to_keep = []

        for other_ch in self.video_chapters:
            # If the chapter is one of the selected ones, it's already marked for deletion.
            if other_ch.unique_id in ids_to_delete:
                continue
          # If this 'other' chapter is fully contained within the new merged range, mark it for deletion too.
            if other_ch.start_frame_id >= new_start_frame and other_ch.end_frame_id <= new_end_frame:
                self.logger.info(f"Gap merge will consume existing chapter: {other_ch.unique_id} ({other_ch.position_long_name})")
                ids_to_delete.add(other_ch.unique_id)

        # Use properties from the first chapter for the merged chapter metadata
        merged_pos_short_key = chapter1.position_short_name
        merged_pos_short_name = chapter1.position_short_name
        merged_pos_long_name = chapter1.position_long_name
        merged_derived_class_name = chapter1.class_name

        merged_chapter = VideoSegment(
            start_frame_id = new_start_frame,
            end_frame_id = new_end_frame,
            class_id=chapter1.class_id,
            class_name=merged_derived_class_name,
            segment_type=chapter1.segment_type,
            position_short_name=merged_pos_short_name,
            position_long_name=merged_pos_long_name,
            source=ChapterSource.MANUAL_GAP_TRACK_MERGE.value
            # Color will be auto-assigned by VideoSegment, or could be chapter1.color
        )

        # Update chapter list: remove old chapters, add new merged one
        self.video_chapters = [ch for ch in self.video_chapters if ch.unique_id not in ids_to_delete]
        self.video_chapters.append(merged_chapter)
        self.video_chapters.sort(key=lambda c: c.start_frame_id)

        self.app.project_manager.project_dirty = True
        self.app.app_state_ui.heatmap_dirty = True
        self.app.app_state_ui.funscript_preview_dirty = True
        self.update_funscript_stats_for_timeline(1, "Gap Tracked & Chapters Merged")
        # self.update_funscript_stats_for_timeline(2, "Gap Tracked & Chapters Merged") # If T2 involved

        self.logger.info(
            f"Successfully finalized tracking of gap and merged into new chapter '{merged_chapter.unique_id}'.")

        self.reset_scripting_range()
        self.logger.info("Scripting range was reset after finalizing gap merge.")

        self.app.set_status_message("Gap tracked and chapters merged successfully.")
        self.app.energy_saver.reset_activity_timer()

    def merge_chapters_across_gap(self, chapter1: VideoSegment, chapter2: VideoSegment) -> Optional[
        VideoSegment]:  # Add return type hint
        if not chapter1 or not chapter2:
            self.logger.error("Two chapters must be provided for merging across a gap.")
            self.app.set_status_message("Error: Two chapters needed for merge.", level=logging.ERROR)
            return None  # Explicitly return None
        if chapter1.unique_id == chapter2.unique_id:
            self.logger.warning("Cannot merge a chapter with itself (across gap).")
            # Optionally set a status message if desired
            return None  # Explicitly return None

        # Ensure chapter1 is the earlier one, already handled by UI sort before call usually
        if chapter1.start_frame_id > chapter2.start_frame_id:
            chapter1, chapter2 = chapter2, chapter1

        new_start_frame = chapter1.start_frame_id
        new_end_frame = chapter2.end_frame_id  # This is the key difference: always use chapter2's end.

        # Check for overlap with *other* chapters BEFORE creating the new one
        ids_to_ignore_for_overlap_check = {chapter1.unique_id, chapter2.unique_id}
        temp_chapters_for_check = [ch for ch in self.video_chapters if
                                   ch.unique_id not in ids_to_ignore_for_overlap_check]

        for other_ch in temp_chapters_for_check:
            if max(new_start_frame, other_ch.start_frame_id) <= min(new_end_frame, other_ch.end_frame_id):
                self.logger.error(
                    f"Merge across gap failed: Resulting chapter [{new_start_frame}-{new_end_frame}] would overlap with existing chapter '{other_ch.unique_id}' [{other_ch.start_frame_id}-{other_ch.end_frame_id}].")
                self.app.set_status_message("Error: Merge would cause overlap with another chapter.",
                                            level=logging.ERROR)
                return None  # Return None on overlap failure

        # Use properties from the first chapter
        merged_pos_short_key = chapter1.position_short_name
        merged_pos_info = constants.POSITION_INFO_MAPPING.get(merged_pos_short_key, {})
        merged_pos_short_name = chapter1.position_short_name # Set directly
        merged_pos_long_name = chapter1.position_long_name # Set directly

        merged_derived_class_name = merged_pos_short_key if merged_pos_short_key else "GapFilledChapter"

        merged_chapter = VideoSegment(
            start_frame_id=new_start_frame,
            end_frame_id=new_end_frame,
            class_id=chapter1.class_id,
            class_name=merged_derived_class_name,
            segment_type=chapter1.segment_type,
            position_short_name=merged_pos_short_name,
            position_long_name=merged_pos_long_name,
            source=ChapterSource.MANUAL_MERGE_GAP_FILL.value,
            color=chapter1.color # Inherit color
        )
        merged_chapter.duration = new_end_frame - new_start_frame

        ids_to_delete = {chapter1.unique_id, chapter2.unique_id}
        self.video_chapters = [ch for ch in self.video_chapters if ch.unique_id not in ids_to_delete]
        self.video_chapters.append(merged_chapter)
        self.video_chapters.sort(key=lambda c: c.start_frame_id)

        self.logger.info(
            f"Filled gap and merged chapters '{chapter1.unique_id}' and '{chapter2.unique_id}' into new chapter '{merged_chapter.unique_id}'.")
        self.app.project_manager.project_dirty = True
        self.app.app_state_ui.heatmap_dirty = True
        self.app.app_state_ui.funscript_preview_dirty = True
        self.app.set_status_message("Chapters merged across gap successfully.")

        return merged_chapter  # Return the newly created chapter

    def set_scripting_range_from_chapter(self, chapter: VideoSegment):
        if not chapter:
            self.logger.warning("Attempted to set scripting range from None chapter.")
            return

        self.scripting_start_frame = chapter.start_frame_id
        self.scripting_end_frame = chapter.end_frame_id
        self.scripting_range_active = True
        self.selected_chapter_for_scripting = chapter

        self.app.project_manager.project_dirty = True
        current_fps_for_log = self._get_current_fps()
        start_t_str = _format_time(self.app,
                                   chapter.start_frame_id / current_fps_for_log if current_fps_for_log > 0 else 0)
        end_t_str = _format_time(self.app, chapter.end_frame_id / current_fps_for_log if current_fps_for_log > 0 else 0)
        self.logger.info(
            f"Scripting range auto-set to chapter: {chapter.position_short_name} [{start_t_str} - {end_t_str}] (Frames: {self.scripting_start_frame}-{self.scripting_end_frame})",
            extra={'status_message': True}
        )
        if hasattr(self.app, 'energy_saver'):
            self.app.energy_saver.reset_activity_timer()

    def reset_state_for_new_project(self):  # Added from app_logic context
        self.logger.debug("AppFunscriptProcessor resetting state for new project.")
        self.video_chapters = []
        self.selected_chapter_for_scripting = None
        self.scripting_range_active = False
        self.scripting_start_frame = 0
        self.scripting_end_frame = 0
        # funscript_obj = self.get_funscript_obj()
        # if funscript_obj:
            # funscript_obj.clear() # Clearing funscript usually handled by FileManage.close_video or tracker.reset
            # pass
        self.update_funscript_stats_for_timeline(1, "Project Reset")
        self.update_funscript_stats_for_timeline(2, "Project Reset")

    def apply_automatic_post_processing(self, frame_range: Optional[Tuple[int, int]] = None):
        """
        Applies a series of post-processing steps to the funscript(s).
        This implementation is now context-aware, applying different settings
        based on video chapters. If no chapters exist, it uses default settings.
        If frame_range is provided, processing is limited to that range.
        """
        funscript_obj = self.get_funscript_obj()
        if not funscript_obj:
            self.logger.warning("Post-Processing: Funscript object not available.")
            return

        self.logger.info("--- Starting Context-Aware Post-Processing ---")

        # --- Get Configurations ---
        processing_config = self.app.app_settings.get("auto_post_processing_amplification_config", {})

        # Create a robust default_params dictionary by merging user settings with system defaults.
        # 1. Start with the hardcoded system defaults from constants.
        base_defaults = constants.DEFAULT_AUTO_POST_AMP_CONFIG.get("Default", {})
        # 2. Get the user's configured defaults, if they exist.
        user_defaults = processing_config.get("Default", {})
        # 3. Create a new dictionary starting with the base defaults and update it with the user's settings.
        #    This ensures all keys are present, with user values overriding where specified.
        default_params = base_defaults.copy()
        default_params.update(user_defaults)

        # --- Record State for Undo ---
        op_desc = "Auto Post-Process" + (f" on range" if frame_range else "")
        if funscript_obj.primary_actions: self._record_timeline_action(1, f"{op_desc} (T1)")
        if funscript_obj.secondary_actions: self._record_timeline_action(2, f"{op_desc} (T2)")

        # --- Determine Time Range ---
        range_start_ms, range_end_ms = None, None
        if frame_range:
            current_fps = self._get_current_fps()
            if current_fps > 0:
                start_frame, end_frame = frame_range
                range_start_ms = self.frame_to_ms(start_frame)
                range_end_ms = self.frame_to_ms(end_frame) if end_frame != -1 else None
                self.logger.info(
                    f"Processing limited to range: Frames {start_frame}-{end_frame if end_frame != -1 else 'End'}")

        # --- Process Each Timeline (Axis) ---
        for axis in ['primary', 'secondary']:
            if not getattr(funscript_obj, f"{axis}_actions", []):
                continue

            self.logger.info(f"Post-Processing: {axis.capitalize()} Axis...")

            if self.video_chapters:
                self.logger.info(f"Applying chapter-based settings for {axis} axis...")
                for chapter in self.video_chapters:
                    chapter_start_ms = self.frame_to_ms(chapter.start_frame_id)
                    chapter_end_ms = self.frame_to_ms(chapter.end_frame_id)

                    effective_start_ms = max(range_start_ms,
                                             chapter_start_ms) if range_start_ms is not None else chapter_start_ms
                    effective_end_ms = min(range_end_ms, chapter_end_ms) if range_end_ms is not None else chapter_end_ms

                    if effective_end_ms <= effective_start_ms:
                        continue

                    # Get chapter-specific params, falling back to the robust default_params
                    params = processing_config.get(chapter.position_long_name, default_params)

                    # With the corrected default_params, these .get() calls are now safe.
                    # If a key is missing from the chapter-specific `params`, it will be found in `default_params`.
                    sg_win = params.get("sg_window", default_params.get("sg_window"))
                    sg_poly = params.get("sg_polyorder", default_params.get("sg_polyorder"))
                    rdp_eps = params.get("rdp_epsilon", default_params.get("rdp_epsilon"))
                    amp_scale = params.get("scale_factor", default_params.get("scale_factor"))
                    amp_center = params.get("center_value", default_params.get("center_value"))
                    clamp_low = params.get("clamp_lower", default_params.get("clamp_lower"))
                    clamp_high = params.get("clamp_upper", default_params.get("clamp_upper"))
                    # output_min = params.get("output_min", default_params.get("output_min"))
                    # output_max = params.get("output_max", default_params.get("output_max"))


                    self.logger.debug(
                        f"Processing {axis} in '{chapter.position_long_name}' ({effective_start_ms}-{effective_end_ms}ms) with params: {params}")

                    funscript_obj.apply_plugin('Savitzky-Golay Filter', axis=axis, window_length=sg_win, polyorder=sg_poly, start_time_ms=effective_start_ms, end_time_ms=effective_end_ms)
                    funscript_obj.apply_plugin('Simplify (RDP)', axis=axis, epsilon=rdp_eps, start_time_ms=effective_start_ms, end_time_ms=effective_end_ms)
                    if axis == 'primary':
                        funscript_obj.apply_plugin('Threshold Clamp', axis=axis, lower_threshold=clamp_low, upper_threshold=clamp_high, start_time_ms=effective_start_ms, end_time_ms=effective_end_ms)
                    funscript_obj.apply_plugin('Amplify', axis=axis, scale_factor=amp_scale, center_value=amp_center, start_time_ms=effective_start_ms, end_time_ms=effective_end_ms)
                    # if output_min != 0 or output_max != 100: # Only apply if it's not the default 0-100
                    #     funscript_obj.scale_points_to_range(axis, output_min, output_max, effective_start_ms, effective_end_ms)

            else:
                self.logger.info(f"No chapters found. Applying default settings to {axis} axis for the full range.")
                params = default_params  # Use the robust defaults
                # Direct access is now safe because we guaranteed the keys exist.
                funscript_obj.apply_plugin('Savitzky-Golay Filter', axis=axis, window_length=params["sg_window"], polyorder=params["sg_polyorder"], start_time_ms=range_start_ms, end_time_ms=range_end_ms)
                funscript_obj.apply_plugin('Simplify (RDP)', axis=axis, epsilon=params["rdp_epsilon"], start_time_ms=range_start_ms, end_time_ms=range_end_ms)
                if axis == 'primary':
                    funscript_obj.apply_plugin('Threshold Clamp', axis=axis, lower_threshold=params["clamp_lower"], upper_threshold=params["clamp_upper"], start_time_ms=range_start_ms, end_time_ms=range_end_ms)
                funscript_obj.apply_plugin('Amplify', axis=axis, scale_factor=params["scale_factor"], center_value=params["center_value"], start_time_ms=range_start_ms, end_time_ms=range_end_ms)
                # output_min = params.get("output_min", 0)
                # output_max = params.get("output_max", 100)
                # if output_min != 0 or output_max != 100:
                #     funscript_obj.scale_points_to_range(axis, output_min, output_max, range_start_ms, range_end_ms)


            timeline_num = 1 if axis == 'primary' else 2
            self._finalize_action_and_update_ui(timeline_num, op_desc)

        # --- Final RDP Pass to Seam Chapters ---
        final_rdp_enabled = self.app.app_settings.get("auto_post_proc_final_rdp_enabled", False)
        if final_rdp_enabled:
            final_rdp_epsilon = self.app.app_settings.get("auto_post_proc_final_rdp_epsilon", 10.0)
            self.logger.info(f"Applying final RDP pass with epsilon={final_rdp_epsilon} to seam chapter joints.")

            final_op_desc = op_desc + " + Final RDP"
            for axis in ['primary', 'secondary']:
                if getattr(funscript_obj, f"{axis}_actions", []):
                    funscript_obj.apply_plugin('Simplify (RDP)', axis=axis, epsilon=final_rdp_epsilon, start_time_ms=None, end_time_ms=None, selected_indices=None)
            if funscript_obj.primary_actions:
                self._finalize_action_and_update_ui(1, final_op_desc)
            if funscript_obj.secondary_actions:
                self._finalize_action_and_update_ui(2, final_op_desc)

        self.logger.info("--- Context-Aware Post-Processing Finished ---")
        self.app.set_status_message("Post-processing applied.", duration=5.0)
        self.app.energy_saver.reset_activity_timer()



    def select_points_in_chapters(self, chapters_to_select_in: List[VideoSegment], target_timeline: str = 'both'):
        """
        Selects all funscript points within the time range of the given chapters
        for the specified timeline(s). This will clear any previous selection on the affected timelines.
        :param chapters_to_select_in: A list of VideoSegment objects.
        :param target_timeline: 'primary', 'secondary', or 'both'.
        """
        if not chapters_to_select_in:
            self.logger.warning("No chapters provided to select points from.")
            return

        if not self.app.gui_instance:
            self.logger.error("Cannot select points, GUI instance not available.")
            return

        # Determine which timeline numbers to process based on the target_timeline parameter
        timelines_to_process_nums = []
        if target_timeline == 'primary':
            timelines_to_process_nums.append(1)
        elif target_timeline == 'secondary':
            timelines_to_process_nums.append(2)
        elif target_timeline == 'both':
            timelines_to_process_nums.extend([1, 2])

        # Create a list of timeline UI instances to process, ensuring they are visible
        timelines_to_process = []
        for timeline_num in timelines_to_process_nums:
            is_visible_flag_name = f"show_funscript_interactive_timeline{'' if timeline_num == 1 else '2'}"
            is_visible = getattr(self.app.app_state_ui, is_visible_flag_name, False)
            ui_instance = getattr(self.app.gui_instance, f"timeline_editor{timeline_num}", None)
            if is_visible and ui_instance:
                timelines_to_process.append({"num": timeline_num, "ui_instance": ui_instance})

        if not timelines_to_process:
            self.logger.info("No relevant interactive timelines are visible to select points on.")
            return

        fps = self._get_current_fps()
        if fps <= 0:
            self.app.set_status_message("Cannot select points: Invalid video FPS.", level=logging.ERROR)
            return

        total_points_selected_overall = 0

        for timeline_info in timelines_to_process:
            timeline_num = timeline_info["num"]
            timeline_ui = timeline_info["ui_instance"]

            funscript_obj, axis_name = self._get_target_funscript_object_and_axis(timeline_num)
            if not funscript_obj or not axis_name:
                continue

            actions_list = getattr(funscript_obj, f"{axis_name}_actions", [])

            # Clear previous selection on this timeline
            timeline_ui.multi_selected_action_indices.clear()

            if not actions_list:
                continue

            action_timestamps = [a['at'] for a in actions_list]
            newly_selected_indices_for_timeline = set()

            for chapter in chapters_to_select_in:
                start_ms = int(round((chapter.start_frame_id / fps) * 1000.0))
                end_ms = int(round((chapter.end_frame_id / fps) * 1000.0))

                start_idx = bisect_left(action_timestamps, start_ms)
                end_idx = bisect_right(action_timestamps, end_ms)

                if start_idx < end_idx:
                    indices_in_range = set(range(start_idx, end_idx))
                    newly_selected_indices_for_timeline.update(indices_in_range)

            # Update the timeline UI with the new selection
            if newly_selected_indices_for_timeline:
                timeline_ui.multi_selected_action_indices = newly_selected_indices_for_timeline
                timeline_ui.selected_action_idx = min(newly_selected_indices_for_timeline)
                total_points_selected_overall += len(newly_selected_indices_for_timeline)
            else:
                timeline_ui.selected_action_idx = -1

        if total_points_selected_overall > 0:
            self.app.set_status_message(f"Selected {total_points_selected_overall} points across targeted timelines.")
        else:
            self.app.set_status_message("No points found in the selected chapter(s).")

        self.app.energy_saver.reset_activity_timer()
