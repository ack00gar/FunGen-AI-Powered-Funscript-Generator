import collections
from typing import Optional, List, Tuple


class UndoRedoManager:
    def __init__(self, max_history: int = 50):
        self.max_history: int = max_history
        # undo_stack: Stores (description_of_action_that_led_AWAY_from_this_state, state_snapshot)
        # So, if state S0 was changed by "Add Point" to S1, undo_stack gets ("Add Point", S0)
        self.undo_stack: collections.deque[Tuple[str, list]] = collections.deque(maxlen=max_history)
        # redo_stack: Stores (description_of_action_to_REAPPLY, state_snapshot_that_results_from_reapply)
        self.redo_stack: collections.deque[Tuple[str, list]] = collections.deque(maxlen=max_history)

        self._actions_list_reference: Optional[list] = None

    def set_actions_reference(self, actions_list_ref: list):
        self._actions_list_reference = actions_list_ref
        self.clear_history()

    def record_state_before_action(self, action_description: str):
        """
        Call this *BEFORE* the actions list is modified.
        'action_description' describes the action that is *about to happen*.
        """
        if self._actions_list_reference is None:
            return

        # Action dicts contain only immutable values (int), so shallow
        # dict copies are safe and ~20x faster than deepcopy.
        state_before_action = [d.copy() for d in self._actions_list_reference]

        # Avoid pushing identical states if the description is also the same (less likely but possible)
        # Short-circuit with length check to avoid O(n) list comparison in most cases
        if self.undo_stack:
            prev_desc, prev_state = self.undo_stack[-1]
            if prev_desc == action_description and len(prev_state) == len(state_before_action):
                if prev_state == state_before_action:
                    return

        self.undo_stack.append((action_description, state_before_action))
        self.redo_stack.clear()  # A new action clears the redo stack

    def undo(self) -> Optional[str]:  # Returns description of the action that was undone
        """
        Performs an undo. The current state is pushed to redo.
        The state from the top of the undo stack is restored.
        """
        if not self.undo_stack or self._actions_list_reference is None:
            return None

        # action_description is "what was done to get from prev_state to current_state"
        # prev_state is the state we are restoring TO.
        action_description_that_was_done, previous_state_to_restore = self.undo_stack.pop()

        # The current live state is the result of 'action_description_that_was_done'
        current_live_state_for_redo = [d.copy() for d in self._actions_list_reference]
        # When redoing, we re-apply 'action_description_that_was_done' to get 'current_live_state_for_redo'
        self.redo_stack.append((action_description_that_was_done, current_live_state_for_redo))

        self._actions_list_reference.clear()
        self._actions_list_reference.extend([d.copy() for d in previous_state_to_restore])

        return action_description_that_was_done  # This is the action that was just "undone"

    def redo(self) -> Optional[str]:  # Returns description of the action that was redone
        """
        Performs a redo. The state before redoing is pushed to undo.
        The state from the top of the redo stack is restored.
        """
        if not self.redo_stack or self._actions_list_reference is None:
            return None

        # action_to_reapply_desc is "what will be done to get from current_state to next_state"
        # state_to_restore_via_redo is the "next_state"
        action_to_reapply_desc, state_to_restore_via_redo = self.redo_stack.pop()

        # The current live state is the one *before* this redo operation.
        current_live_state_for_undo = [d.copy() for d in self._actions_list_reference]
        # If we undo this redo, we revert 'action_to_reapply_desc', going back to 'current_live_state_for_undo'.
        # So, on undo_stack, we store (action_to_reapply_desc, current_live_state_for_undo)
        self.undo_stack.append((action_to_reapply_desc, current_live_state_for_undo))

        self._actions_list_reference.clear()
        self._actions_list_reference.extend([d.copy() for d in state_to_restore_via_redo])

        return action_to_reapply_desc  # This is the action that was just "redone"

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def clear_history(self):
        self.undo_stack.clear()
        self.redo_stack.clear()

    def get_undo_history_for_display(self) -> List[str]:
        """Returns a list of descriptions for actions that can be undone."""
        # The last item pushed to undo_stack is the most recent action taken.
        return [item[0] for item in reversed(self.undo_stack)]

    def get_redo_history_for_display(self) -> List[str]:
        """Returns a list of descriptions for actions that can be redone."""
        # The last item pushed to redo_stack is the most recent action undone.
        return [item[0] for item in reversed(self.redo_stack)]
