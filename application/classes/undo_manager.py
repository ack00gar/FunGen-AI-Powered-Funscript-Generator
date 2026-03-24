"""Unified command-pattern undo/redo manager.

Single instance on app. All mutations (points + chapters) go through this.
Each Command stores the minimal delta needed to undo/redo.
"""
import struct
import time
from collections import deque
from typing import Optional, List, Dict, Any


# ------------------------------------------------------------------ #
#  Packed action helpers (8 bytes per action vs ~232 for dict)        #
# ------------------------------------------------------------------ #

_ACTION_FMT = '<ii'
_ACTION_SIZE = struct.calcsize(_ACTION_FMT)


def pack_actions(actions: list) -> bytes:
    buf = bytearray(len(actions) * _ACTION_SIZE)
    offset = 0
    for a in actions:
        struct.pack_into(_ACTION_FMT, buf, offset, a['at'], a['pos'])
        offset += _ACTION_SIZE
    return bytes(buf)


def unpack_actions(data: bytes) -> list:
    count = len(data) // _ACTION_SIZE
    result = []
    offset = 0
    for _ in range(count):
        at, pos = struct.unpack_from(_ACTION_FMT, data, offset)
        result.append({'at': at, 'pos': pos})
        offset += _ACTION_SIZE
    return result


# ------------------------------------------------------------------ #
#  Finalize helper                                                    #
# ------------------------------------------------------------------ #

def _finalize_timeline(app, timeline_num, description):
    """Post-mutation cleanup for a timeline: cache invalidation, stats, UI."""
    fs_proc = app.funscript_processor
    fs, axis = fs_proc._get_target_funscript_object_and_axis(timeline_num)
    if fs:
        fs._invalidate_cache(axis or 'both')
    fs_proc._revision += 1
    fs_proc.update_funscript_stats_for_timeline(timeline_num, description)
    app.project_manager.project_dirty = True
    app.app_state_ui.heatmap_dirty = True
    app.app_state_ui.funscript_preview_dirty = True
    gui = app.gui_instance
    if gui:
        if timeline_num == 1:
            tl = gui.timeline_editor1
        elif timeline_num == 2:
            tl = getattr(gui, 'timeline_editor2', None)
        else:
            tl = gui._extra_timeline_editors.get(timeline_num)
        if tl:
            tl.invalidate_cache()


def _finalize_chapters(app, description):
    """Post-mutation cleanup for chapters."""
    fs_proc = app.funscript_processor
    fs_proc._sync_chapters_to_funscript()
    app.project_manager.project_dirty = True
    app.app_state_ui.heatmap_dirty = True
    app.app_state_ui.funscript_preview_dirty = True


def _get_actions_list(app, timeline_num):
    """Get the mutable actions list for a timeline."""
    fs_proc = app.funscript_processor
    fs, axis = fs_proc._get_target_funscript_object_and_axis(timeline_num)
    if fs and axis:
        return fs.get_axis_actions(axis)
    return None


# ------------------------------------------------------------------ #
#  Command base class                                                 #
# ------------------------------------------------------------------ #

class Command:
    """Base class for all undoable operations."""
    __slots__ = ('description', 'timestamp')

    def __init__(self, description: str):
        self.description = description
        self.timestamp = time.monotonic()

    def execute(self, app):
        """Apply the change (used for initial execution and redo)."""
        raise NotImplementedError

    def undo(self, app):
        """Reverse the change."""
        raise NotImplementedError

    def finalize(self, app):
        """Post-apply cleanup (cache invalidation, UI update)."""
        pass


# ------------------------------------------------------------------ #
#  Point commands                                                     #
# ------------------------------------------------------------------ #

class AddPointCmd(Command):
    __slots__ = ('description', 'timestamp', 'tl', 'action')

    def __init__(self, tl: int, action: dict):
        super().__init__(f"Add Point (T{tl})")
        self.tl = tl
        self.action = action.copy()

    def execute(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions is not None:
            from bisect import insort_left
            insort_left(actions, self.action, key=lambda a: a['at'])

    def undo(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions is not None:
            for i, a in enumerate(actions):
                if a['at'] == self.action['at'] and a['pos'] == self.action['pos']:
                    actions.pop(i)
                    break

    def finalize(self, app):
        _finalize_timeline(app, self.tl, self.description)


class DeletePointsCmd(Command):
    __slots__ = ('description', 'timestamp', 'tl', 'deleted')

    def __init__(self, tl: int, deleted: list):
        n = len(deleted)
        desc = f"Delete {n} Point{'s' if n != 1 else ''} (T{tl})"
        super().__init__(desc)
        self.tl = tl
        self.deleted = [(d['index'], d['action'].copy()) for d in deleted]

    def execute(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions is not None:
            for idx, _ in sorted(self.deleted, key=lambda x: x[0], reverse=True):
                if idx < len(actions):
                    actions.pop(idx)

    def undo(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions is not None:
            for idx, action in sorted(self.deleted, key=lambda x: x[0]):
                actions.insert(idx, action.copy())

    def finalize(self, app):
        _finalize_timeline(app, self.tl, self.description)


class MovePointCmd(Command):
    __slots__ = ('description', 'timestamp', 'tl', 'index', 'old_at', 'old_pos', 'new_at', 'new_pos')

    def __init__(self, tl: int, index: int, old_at: int, old_pos: int, new_at: int, new_pos: int):
        super().__init__(f"Drag Point (T{tl})")
        self.tl = tl
        self.index = index
        self.old_at = old_at
        self.old_pos = old_pos
        self.new_at = new_at
        self.new_pos = new_pos

    def execute(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions and self.index < len(actions):
            actions[self.index]['at'] = self.new_at
            actions[self.index]['pos'] = self.new_pos

    def undo(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions and self.index < len(actions):
            actions[self.index]['at'] = self.old_at
            actions[self.index]['pos'] = self.old_pos

    def finalize(self, app):
        _finalize_timeline(app, self.tl, self.description)


class NudgeValuesCmd(Command):
    __slots__ = ('description', 'timestamp', 'tl', 'indices', 'delta')

    def __init__(self, tl: int, indices: list, delta: int):
        super().__init__(f"Nudge Value (T{tl})")
        self.tl = tl
        self.indices = list(indices)
        self.delta = delta

    def execute(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions:
            for idx in self.indices:
                if idx < len(actions):
                    actions[idx]['pos'] = max(0, min(100, actions[idx]['pos'] + self.delta))

    def undo(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions:
            for idx in self.indices:
                if idx < len(actions):
                    actions[idx]['pos'] = max(0, min(100, actions[idx]['pos'] - self.delta))

    def finalize(self, app):
        _finalize_timeline(app, self.tl, self.description)


class NudgeTimesCmd(Command):
    __slots__ = ('description', 'timestamp', 'tl', 'indices', 'delta_ms')

    def __init__(self, tl: int, indices: list, delta_ms: int, label: str = "Nudge Time"):
        super().__init__(f"{label} (T{tl})")
        self.tl = tl
        self.indices = list(indices)
        self.delta_ms = delta_ms

    def execute(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions:
            for idx in self.indices:
                if idx < len(actions):
                    actions[idx]['at'] += self.delta_ms

    def undo(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions:
            for idx in self.indices:
                if idx < len(actions):
                    actions[idx]['at'] -= self.delta_ms

    def finalize(self, app):
        _finalize_timeline(app, self.tl, self.description)


class SnapToPlayheadCmd(Command):
    __slots__ = ('description', 'timestamp', 'tl', 'index', 'old_at', 'new_at')

    def __init__(self, tl: int, index: int, old_at: int, new_at: int):
        super().__init__(f"Snap to Playhead (T{tl})")
        self.tl = tl
        self.index = index
        self.old_at = old_at
        self.new_at = new_at

    def execute(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions and self.index < len(actions):
            actions[self.index]['at'] = self.new_at

    def undo(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions and self.index < len(actions):
            actions[self.index]['at'] = self.old_at

    def finalize(self, app):
        _finalize_timeline(app, self.tl, self.description)


class PasteActionsCmd(Command):
    __slots__ = ('description', 'timestamp', 'tl', 'pasted_actions')

    def __init__(self, tl: int, pasted_actions: list):
        n = len(pasted_actions)
        super().__init__(f"Paste {n} Point{'s' if n != 1 else ''} (T{tl})")
        self.tl = tl
        self.pasted_actions = [a.copy() for a in pasted_actions]

    def execute(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions is not None:
            actions.extend(self.pasted_actions)
            actions.sort(key=lambda a: a['at'])

    def undo(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions is not None:
            pasted_set = {(a['at'], a['pos']) for a in self.pasted_actions}
            i = len(actions) - 1
            while i >= 0 and pasted_set:
                key = (actions[i]['at'], actions[i]['pos'])
                if key in pasted_set:
                    actions.pop(i)
                    pasted_set.discard(key)
                i -= 1

    def finalize(self, app):
        _finalize_timeline(app, self.tl, self.description)


class BulkReplaceCmd(Command):
    """Full snapshot replace for bulk operations (stage load, plugin apply, file load)."""
    __slots__ = ('description', 'timestamp', 'tl', 'old_packed', 'new_packed')

    def __init__(self, tl: int, old_actions: list, new_actions: list, description: str):
        super().__init__(description)
        self.tl = tl
        self.old_packed = pack_actions(old_actions)
        self.new_packed = pack_actions(new_actions)

    def execute(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions is not None:
            actions.clear()
            actions.extend(unpack_actions(self.new_packed))

    def undo(self, app):
        actions = _get_actions_list(app, self.tl)
        if actions is not None:
            actions.clear()
            actions.extend(unpack_actions(self.old_packed))

    def finalize(self, app):
        _finalize_timeline(app, self.tl, self.description)


# ------------------------------------------------------------------ #
#  Chapter commands                                                   #
# ------------------------------------------------------------------ #

class CreateChapterCmd(Command):
    __slots__ = ('description', 'timestamp', 'chapter_id', 'chapter_data')

    def __init__(self, chapter_id: str, chapter_data: dict):
        name = chapter_data.get('position_short_name_key', 'chapter')
        super().__init__(f"Create Chapter: {name}")
        self.chapter_id = chapter_id
        self.chapter_data = chapter_data

    def execute(self, app):
        app.funscript_processor.create_new_chapter_from_data(self.chapter_data)

    def undo(self, app):
        app.funscript_processor.delete_video_chapters_by_ids(
            [self.chapter_id], _skip_undo_record=True)

    def finalize(self, app):
        _finalize_chapters(app, self.description)


class DeleteChaptersCmd(Command):
    __slots__ = ('description', 'timestamp', 'chapters')

    def __init__(self, chapters: list):
        n = len(chapters)
        super().__init__(f"Delete {n} Chapter{'s' if n != 1 else ''}")
        # Store full chapter objects for restoration
        self.chapters = list(chapters)

    def execute(self, app):
        ids = [ch.unique_id for ch in self.chapters]
        app.funscript_processor.delete_video_chapters_by_ids(ids, _skip_undo_record=True)

    def undo(self, app):
        for ch in self.chapters:
            app.funscript_processor.video_chapters.append(ch)
        app.funscript_processor.video_chapters.sort(key=lambda c: c.start_frame_id)

    def finalize(self, app):
        _finalize_chapters(app, self.description)


class UpdateChapterCmd(Command):
    __slots__ = ('description', 'timestamp', 'chapter_id', 'old_fields', 'new_fields')

    def __init__(self, chapter_id: str, old_fields: dict, new_fields: dict):
        super().__init__("Update Chapter")
        self.chapter_id = chapter_id
        self.old_fields = old_fields.copy()
        self.new_fields = new_fields.copy()

    def _apply_fields(self, app, fields):
        for ch in app.funscript_processor.video_chapters:
            if ch.unique_id == self.chapter_id:
                for k, v in fields.items():
                    setattr(ch, k, v)
                break

    def execute(self, app):
        self._apply_fields(app, self.new_fields)

    def undo(self, app):
        self._apply_fields(app, self.old_fields)

    def finalize(self, app):
        _finalize_chapters(app, self.description)


class ReplaceAllChaptersCmd(Command):
    """Full chapter list replacement (stage results, file load)."""
    __slots__ = ('description', 'timestamp', 'old_chapters', 'new_chapters')

    def __init__(self, old_chapters: list, new_chapters: list, description: str):
        super().__init__(description)
        self.old_chapters = list(old_chapters)
        self.new_chapters = list(new_chapters)

    def execute(self, app):
        fs_proc = app.funscript_processor
        fs_proc.video_chapters.clear()
        fs_proc.video_chapters.extend(self.new_chapters)

    def undo(self, app):
        fs_proc = app.funscript_processor
        fs_proc.video_chapters.clear()
        fs_proc.video_chapters.extend(self.old_chapters)

    def finalize(self, app):
        _finalize_chapters(app, self.description)


# ------------------------------------------------------------------ #
#  Compound command                                                   #
# ------------------------------------------------------------------ #

class CompoundCmd(Command):
    """Groups multiple commands into a single undoable operation."""
    __slots__ = ('description', 'timestamp', 'commands')

    def __init__(self, commands: list, description: str):
        super().__init__(description)
        self.commands = list(commands)

    def execute(self, app):
        for cmd in self.commands:
            cmd.execute(app)

    def undo(self, app):
        for cmd in reversed(self.commands):
            cmd.undo(app)

    def finalize(self, app):
        for cmd in self.commands:
            cmd.finalize(app)


# ------------------------------------------------------------------ #
#  Unified UndoManager                                                #
# ------------------------------------------------------------------ #

class UndoManager:
    """Single unified undo/redo manager for the entire application."""

    def __init__(self, max_history: int = 100):
        self.undo_stack: deque = deque(maxlen=max_history)
        self.redo_stack: deque = deque(maxlen=max_history)

    def execute(self, command: Command, app):
        """Execute a command and push it to the undo stack."""
        command.execute(app)
        command.finalize(app)
        self.undo_stack.append(command)
        self.redo_stack.clear()

    def push_done(self, command: Command):
        """Push an already-executed command (for mutations that happen inline)."""
        self.undo_stack.append(command)
        self.redo_stack.clear()

    def undo(self, app) -> Optional[str]:
        """Undo the most recent command. Returns description or None."""
        if not self.undo_stack:
            return None
        cmd = self.undo_stack.pop()
        cmd.undo(app)
        cmd.finalize(app)
        self.redo_stack.append(cmd)
        return cmd.description

    def redo(self, app) -> Optional[str]:
        """Redo the most recently undone command. Returns description or None."""
        if not self.redo_stack:
            return None
        cmd = self.redo_stack.pop()
        cmd.execute(app)
        cmd.finalize(app)
        self.undo_stack.append(cmd)
        return cmd.description

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def clear(self):
        """Clear all undo/redo history."""
        self.undo_stack.clear()
        self.redo_stack.clear()

    def peek_undo(self) -> Optional[str]:
        """Get description of next undo without performing it."""
        return self.undo_stack[-1].description if self.undo_stack else None

    def peek_redo(self) -> Optional[str]:
        """Get description of next redo without performing it."""
        return self.redo_stack[-1].description if self.redo_stack else None

    def undo_to(self, index: int, app) -> int:
        """Undo multiple steps. index=0 means undo 1 (most recent), index=N means undo N+1.
        Returns number of steps actually undone."""
        count = index + 1
        done = 0
        for _ in range(count):
            if not self.undo_stack:
                break
            cmd = self.undo_stack.pop()
            cmd.undo(app)
            cmd.finalize(app)
            self.redo_stack.append(cmd)
            done += 1
        return done

    def redo_to(self, index: int, app) -> int:
        """Redo multiple steps. index=0 means redo 1 (most recent), index=N means redo N+1.
        Returns number of steps actually redone."""
        count = index + 1
        done = 0
        for _ in range(count):
            if not self.redo_stack:
                break
            cmd = self.redo_stack.pop()
            cmd.execute(app)
            cmd.finalize(app)
            self.undo_stack.append(cmd)
            done += 1
        return done

    def get_undo_history(self) -> List[str]:
        """Get descriptions for display (most recent first)."""
        return [cmd.description for cmd in reversed(self.undo_stack)]

    def get_redo_history(self) -> List[str]:
        """Get descriptions for display (most recent first)."""
        return [cmd.description for cmd in reversed(self.redo_stack)]
