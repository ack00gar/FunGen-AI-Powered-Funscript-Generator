"""Video Display UI — decomposed from monolithic video_display_ui.py."""
from application.gui_components.video_display._core import VideoDisplayCoreMixin
from application.gui_components.video_display.controls import VideoControlsMixin
from application.gui_components.video_display.overlays import VideoOverlaysMixin
from application.gui_components.video_display.handy_integration import HandyVideoMixin


class VideoDisplayUI(
    VideoDisplayCoreMixin,
    VideoControlsMixin,
    VideoOverlaysMixin,
    HandyVideoMixin,
):
    """Video display panel — composed from focused sub-mixins."""
    pass
