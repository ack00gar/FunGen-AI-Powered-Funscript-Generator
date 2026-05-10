"""Chapter actions invokable from shortcuts and context menus.

Each function takes the app and resolves the chapter at the current
playhead, so a hotkey works without the user right-clicking a chapter
first. Returns True on success for the dispatcher.
"""
from __future__ import annotations

from typing import Optional


def _chapter_at_playhead(app):
    proc = getattr(app, 'processor', None)
    fs_proc = getattr(app, 'funscript_processor', None)
    if proc is None or fs_proc is None:
        return None, None
    frame = int(getattr(proc, 'current_frame_index', 0) or 0)
    chapter = fs_proc.get_chapter_at_frame(frame)
    return chapter, frame


def split_chapter_at_cursor(app) -> bool:
    """Split the chapter that contains the playhead, at the playhead frame.
    No-op if the playhead is at a chapter boundary or no chapter contains it."""
    chapter, frame = _chapter_at_playhead(app)
    if chapter is None or frame is None:
        return False
    if not (chapter.start_frame_id < frame < chapter.end_frame_id):
        return False

    fs_proc = app.funscript_processor
    from config.constants import POSITION_INFO_MAPPING
    pos_key = chapter.position_short_name
    for key, info in POSITION_INFO_MAPPING.items():
        if info.get('short_name') == chapter.position_short_name:
            pos_key = key
            break

    original_end = chapter.end_frame_id
    old_fields = {
        'start_frame_id': chapter.start_frame_id,
        'end_frame_id': chapter.end_frame_id,
        'position_short_name': chapter.position_short_name,
        'position_long_name': chapter.position_long_name,
        'class_name': chapter.class_name,
        'segment_type': chapter.segment_type,
        'source': chapter.source,
        'color': chapter.color,
    }
    fs_proc.update_chapter_from_data(
        chapter.unique_id,
        {
            'start_frame_str': str(chapter.start_frame_id),
            'end_frame_str': str(frame),
            'position_short_name_key': pos_key,
        },
        _skip_undo_record=True,
    )
    new_fields = {
        'start_frame_id': chapter.start_frame_id,
        'end_frame_id': chapter.end_frame_id,
        'position_short_name': chapter.position_short_name,
        'position_long_name': chapter.position_long_name,
        'class_name': chapter.class_name,
        'segment_type': chapter.segment_type,
        'source': chapter.source,
        'color': chapter.color,
    }
    new_chapter_data = {
        'start_frame_str': str(frame + 1),
        'end_frame_str': str(original_end),
        'position_short_name_key': pos_key,
        'segment_type': chapter.segment_type,
        'source': chapter.source,
    }
    new_chapter = fs_proc.create_new_chapter_from_data(
        new_chapter_data,
        return_chapter_object=True,
        _skip_undo_record=True,
    )
    undo = getattr(app, 'undo_manager', None)
    if undo is not None and new_chapter is not None:
        from application.classes.undo_manager import (
            CompoundCmd, CreateChapterCmd, UpdateChapterCmd,
        )
        undo.push_done(CompoundCmd([
            UpdateChapterCmd(chapter.unique_id, old_fields, new_fields),
            CreateChapterCmd(new_chapter.unique_id, new_chapter_data),
        ], 'Split Chapter'))
    app.notify(f'Chapter split at frame {frame}', 'success', 1.5)
    return True


def seek_to_chapter_start(app) -> bool:
    chapter, _ = _chapter_at_playhead(app)
    if chapter is None:
        return False
    proc = app.processor
    if proc is None:
        return False
    proc.seek_video(chapter.start_frame_id)
    app.app_state_ui.force_timeline_pan_to_current_frame = True
    return True


def seek_to_chapter_end(app) -> bool:
    chapter, _ = _chapter_at_playhead(app)
    if chapter is None:
        return False
    proc = app.processor
    if proc is None:
        return False
    proc.seek_video(chapter.end_frame_id)
    app.app_state_ui.force_timeline_pan_to_current_frame = True
    return True


def snap_chapter_start_to_playhead(app) -> bool:
    """Move the start of the chapter at the playhead to the playhead frame.
    Refuses if the playhead is past the chapter's end (would invert the range)."""
    chapter, frame = _chapter_at_playhead(app)
    if chapter is None or frame is None:
        return False
    if frame >= chapter.end_frame_id:
        return False
    fs_proc = app.funscript_processor
    fs_proc.update_chapter_from_data(
        chapter.unique_id,
        {
            'start_frame_str': str(frame),
            'end_frame_str': str(chapter.end_frame_id),
        },
    )
    return True


def snap_chapter_end_to_playhead(app) -> bool:
    """Move the end of the chapter at the playhead to the playhead frame."""
    chapter, frame = _chapter_at_playhead(app)
    if chapter is None or frame is None:
        return False
    if frame <= chapter.start_frame_id:
        return False
    fs_proc = app.funscript_processor
    fs_proc.update_chapter_from_data(
        chapter.unique_id,
        {
            'start_frame_str': str(chapter.start_frame_id),
            'end_frame_str': str(frame),
        },
    )
    return True
