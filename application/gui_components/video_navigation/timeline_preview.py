import imgui
from typing import Optional

from application.utils import _format_time, VideoSegment, get_icon_texture_manager, primary_button_style, destructive_button_style
from config.constants import POSITION_INFO_MAPPING, DEFAULT_CHAPTER_FPS
from config.element_group_colors import VideoNavigationColors
from config.constants_colors import CurrentTheme



class TimelinePreviewMixin:
    """Mixin fragment for VideoNavigationUI."""

    def _render_funscript_timeline_preview(self, total_duration_s: float, graph_height: int):
        self.gui_instance.render_funscript_timeline_preview(total_duration_s, graph_height)


    def _render_funscript_heatmap_preview(self, total_video_duration_s: float, bar_width_float: float, bar_height_float: float):
        # bar_width_float here is nav_content_width
        self.gui_instance.render_funscript_heatmap_preview(total_video_duration_s, bar_width_float, bar_height_float)


    def _render_chapter_tooltip(self):
        # Make sure chapter_tooltip_segment is valid before trying to access its attributes
        if not self.chapter_tooltip_segment or not hasattr(self.chapter_tooltip_segment, 'class_name'):
            return

        imgui.begin_tooltip()
        segment = self.chapter_tooltip_segment

        fs_proc = self.app.funscript_processor
        chapter_number_str = "N/A"
        if fs_proc and fs_proc.video_chapters:
            sorted_chapters = sorted(fs_proc.video_chapters, key=lambda c: c.start_frame_id)
            try:
                chapter_index = sorted_chapters.index(segment)
                chapter_number_str = str(chapter_index + 1)
            except ValueError:
                # Fallback to ID search if object identity fails
                for i, chap in enumerate(sorted_chapters):
                    if chap.unique_id == segment.unique_id:
                        chapter_number_str = str(i + 1)
                        break

        imgui.text(f"Chapter #{chapter_number_str}: {segment.position_short_name} ({segment.segment_type})")
        imgui.text(f"Pos:  {segment.position_long_name}")
        imgui.text(f"Source: {segment.source}")
        imgui.text(f"Frames: {segment.start_frame_id} - {segment.end_frame_id}")

        fps_tt = self._get_current_fps()
        start_t_tt = segment.start_frame_id / fps_tt if fps_tt > 0 else 0
        end_t_tt = segment.end_frame_id / fps_tt if fps_tt > 0 else 0
        imgui.text(f"Time: {_format_time(self.app, start_t_tt)} - {_format_time(self.app, end_t_tt)}")

        # Duration
        duration_frames = segment.end_frame_id - segment.start_frame_id + 1
        duration_s = duration_frames / fps_tt if fps_tt > 0 else 0
        duration_timecode = _format_time(self.app, duration_s)
        imgui.text(f"Duration: {duration_timecode} ({duration_frames} frames)")

        imgui.end_tooltip()

