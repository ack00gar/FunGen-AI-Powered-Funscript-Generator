"""Video Navigation UI — re-exports from decomposed subpackage.

The actual implementation lives in application/gui_components/video_navigation/.
This file preserves the original import path for backwards compatibility.
"""
from application.gui_components.video_navigation import VideoNavigationUI, ChapterListWindow

__all__ = ['VideoNavigationUI', 'ChapterListWindow']
