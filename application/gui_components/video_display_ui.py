"""Video Display UI — re-exports from decomposed subpackage.

The actual implementation lives in application/gui_components/video_display/.
This file preserves the original import path for backwards compatibility.
"""
from application.gui_components.video_display import VideoDisplayUI

__all__ = ['VideoDisplayUI']
