import collections
import struct
from typing import Optional, List, Tuple

# Each action is (at: int32, pos: int32) = 8 bytes
_ACTION_FMT = '<ii'  # little-endian, two signed int32
_ACTION_SIZE = struct.calcsize(_ACTION_FMT)


def _pack_actions(actions: list) -> bytes:
    """Pack a list of {'at': int, 'pos': int} dicts into compact bytes.

    ~29x smaller than list-of-dicts: 8 bytes per action vs ~232 bytes.
    """
    buf = bytearray(len(actions) * _ACTION_SIZE)
    offset = 0
    for a in actions:
        struct.pack_into(_ACTION_FMT, buf, offset, a['at'], a['pos'])
        offset += _ACTION_SIZE
    return bytes(buf)


def _unpack_actions(data: bytes) -> list:
    """Unpack bytes back into a list of {'at': int, 'pos': int} dicts."""
    count = len(data) // _ACTION_SIZE
    result = []
    offset = 0
    for _ in range(count):
        at, pos = struct.unpack_from(_ACTION_FMT, data, offset)
        result.append({'at': at, 'pos': pos})
        offset += _ACTION_SIZE
    return result


class UndoRedoManager:
    def __init__(self, max_history: int = 50):
        self.max_history: int = max_history
        # Stacks store (description, packed_bytes) instead of (description, list[dict])
        self.undo_stack: collections.deque[Tuple[str, bytes]] = collections.deque(maxlen=max_history)
        self.redo_stack: collections.deque[Tuple[str, bytes]] = collections.deque(maxlen=max_history)

        self._actions_list_reference: Optional[list] = None
        # Cache last packed state to avoid redundant packing on dedup check
        self._last_packed: Optional[bytes] = None
        self._last_packed_len: int = -1

    def set_actions_reference(self, actions_list_ref: list):
        self._actions_list_reference = actions_list_ref
        self._last_packed = None
        self._last_packed_len = -1
        self.clear_history()

    def record_state_before_action(self, action_description: str):
        """Call this BEFORE the actions list is modified."""
        if self._actions_list_reference is None:
            return

        packed = _pack_actions(self._actions_list_reference)

        # Dedup: skip if identical to last pushed state with same description
        if self.undo_stack:
            prev_desc, prev_packed = self.undo_stack[-1]
            if prev_desc == action_description and prev_packed == packed:
                return

        self.undo_stack.append((action_description, packed))
        self.redo_stack.clear()

    def undo(self) -> Optional[str]:
        """Undo: push current state to redo, restore previous state."""
        if not self.undo_stack or self._actions_list_reference is None:
            return None

        action_desc, prev_packed = self.undo_stack.pop()

        # Save current state for redo
        current_packed = _pack_actions(self._actions_list_reference)
        self.redo_stack.append((action_desc, current_packed))

        # Restore
        restored = _unpack_actions(prev_packed)
        self._actions_list_reference.clear()
        self._actions_list_reference.extend(restored)

        return action_desc

    def redo(self) -> Optional[str]:
        """Redo: push current state to undo, restore redo state."""
        if not self.redo_stack or self._actions_list_reference is None:
            return None

        action_desc, redo_packed = self.redo_stack.pop()

        # Save current state for undo
        current_packed = _pack_actions(self._actions_list_reference)
        self.undo_stack.append((action_desc, current_packed))

        # Restore
        restored = _unpack_actions(redo_packed)
        self._actions_list_reference.clear()
        self._actions_list_reference.extend(restored)

        return action_desc

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def clear_history(self):
        self.undo_stack.clear()
        self.redo_stack.clear()

    def get_undo_history_for_display(self) -> List[str]:
        return [item[0] for item in reversed(self.undo_stack)]

    def get_redo_history_for_display(self) -> List[str]:
        return [item[0] for item in reversed(self.redo_stack)]
