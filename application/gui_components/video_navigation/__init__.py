"""Video Navigation UI — decomposed from monolithic video_navigation_ui.py."""
from application.gui_components.video_navigation._core import VideoNavigationCoreMixin
from application.gui_components.video_navigation.chapter_bar import ChapterBarMixin
from application.gui_components.video_navigation.chapter_editor import ChapterEditorMixin
from application.gui_components.video_navigation.timeline_preview import TimelinePreviewMixin
from application.gui_components.video_navigation.chapter_plugins import ChapterPluginsMixin
from application.gui_components.video_navigation.chapter_list_window import ChapterListWindow


class VideoNavigationUI(
    VideoNavigationCoreMixin,
    ChapterBarMixin,
    ChapterEditorMixin,
    TimelinePreviewMixin,
    ChapterPluginsMixin,
):
    """Video navigation panel — composed from focused sub-mixins."""
    pass
