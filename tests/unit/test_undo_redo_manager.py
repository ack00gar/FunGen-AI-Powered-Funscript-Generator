"""Unit tests for UndoRedoManager class."""
import pytest


@pytest.mark.unit
class TestUndoRedoManagerCreation:
    """Tests for UndoRedoManager construction and initial state."""

    def test_creation_with_default_max_history(self):
        """Default max_history is 50."""
        from application.classes import UndoRedoManager
        mgr = UndoRedoManager()
        assert mgr.max_history == 50

    def test_creation_with_custom_max_history(self):
        """Custom max_history is respected."""
        from application.classes import UndoRedoManager
        mgr = UndoRedoManager(max_history=10)
        assert mgr.max_history == 10

    def test_initial_state_empty(self, undo_redo_manager):
        """Initially, both undo and redo stacks are empty."""
        mgr = undo_redo_manager
        assert mgr.can_undo() is False
        assert mgr.can_redo() is False

    def test_initial_no_actions_reference(self, undo_redo_manager):
        """Initially, no actions reference is set."""
        mgr = undo_redo_manager
        assert mgr._actions_list_reference is None


@pytest.mark.unit
class TestUndoRedoManagerSetReference:
    """Tests for setting the actions reference."""

    def test_set_actions_reference(self, undo_redo_manager):
        """set_actions_reference stores the reference."""
        mgr = undo_redo_manager
        actions = [{"at": 100, "pos": 50}]
        mgr.set_actions_reference(actions)
        assert mgr._actions_list_reference is actions

    def test_set_actions_reference_clears_history(self, undo_redo_manager):
        """set_actions_reference clears both undo and redo stacks."""
        mgr = undo_redo_manager
        actions = [{"at": 100, "pos": 50}]
        mgr.set_actions_reference(actions)
        # Record some state
        mgr.record_state_before_action("test action")
        actions.append({"at": 200, "pos": 75})
        assert mgr.can_undo()
        
        # Setting new reference should clear everything
        new_actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(new_actions)
        assert mgr.can_undo() is False
        assert mgr.can_redo() is False


@pytest.mark.unit
class TestUndoRedoManagerRecordAndUndo:
    """Tests for recording state and undoing."""

    def test_record_state_enables_undo(self, undo_redo_manager):
        """Recording state before an action enables undo."""
        mgr = undo_redo_manager
        actions = [{"at": 100, "pos": 50}]
        mgr.set_actions_reference(actions)
        mgr.record_state_before_action("Add Point")
        actions.append({"at": 200, "pos": 75})
        assert mgr.can_undo() is True

    def test_undo_restores_previous_state(self, undo_redo_manager):
        """Undo restores the actions list to the state before the action."""
        mgr = undo_redo_manager
        actions = [{"at": 100, "pos": 50}]
        mgr.set_actions_reference(actions)
        
        # Record state before modification
        mgr.record_state_before_action("Add Point")
        actions.append({"at": 200, "pos": 75})
        assert len(actions) == 2
        
        # Undo
        desc = mgr.undo()
        assert desc == "Add Point"
        assert len(actions) == 1
        assert actions[0] == {"at": 100, "pos": 50}

    def test_undo_returns_action_description(self, undo_redo_manager):
        """Undo returns the description of the undone action."""
        mgr = undo_redo_manager
        actions = [{"at": 100, "pos": 50}]
        mgr.set_actions_reference(actions)
        mgr.record_state_before_action("Delete Point")
        actions.pop()
        result = mgr.undo()
        assert result == "Delete Point"

    def test_undo_empty_returns_none(self, undo_redo_manager):
        """Undo on empty stack returns None."""
        mgr = undo_redo_manager
        actions = []
        mgr.set_actions_reference(actions)
        assert mgr.undo() is None

    def test_undo_without_reference_returns_none(self, undo_redo_manager):
        """Undo without actions reference returns None."""
        mgr = undo_redo_manager
        assert mgr.undo() is None

    def test_multiple_undos(self, undo_redo_manager):
        """Multiple sequential undos restore progressively older states."""
        mgr = undo_redo_manager
        actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(actions)
        
        # Action 1: add point
        mgr.record_state_before_action("Action 1")
        actions.append({"at": 100, "pos": 50})
        
        # Action 2: add another point
        mgr.record_state_before_action("Action 2")
        actions.append({"at": 200, "pos": 100})
        
        assert len(actions) == 3
        
        # Undo Action 2
        desc = mgr.undo()
        assert desc == "Action 2"
        assert len(actions) == 2
        
        # Undo Action 1
        desc = mgr.undo()
        assert desc == "Action 1"
        assert len(actions) == 1


@pytest.mark.unit
class TestUndoRedoManagerRedo:
    """Tests for redo functionality."""

    def test_redo_after_undo(self, undo_redo_manager):
        """Redo restores the state that was undone."""
        mgr = undo_redo_manager
        actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(actions)
        
        mgr.record_state_before_action("Add Point")
        actions.append({"at": 100, "pos": 50})
        
        mgr.undo()
        assert len(actions) == 1
        
        desc = mgr.redo()
        assert desc == "Add Point"
        assert len(actions) == 2
        assert actions[1] == {"at": 100, "pos": 50}

    def test_redo_empty_returns_none(self, undo_redo_manager):
        """Redo on empty redo stack returns None."""
        mgr = undo_redo_manager
        actions = []
        mgr.set_actions_reference(actions)
        assert mgr.redo() is None

    def test_redo_without_reference_returns_none(self, undo_redo_manager):
        """Redo without actions reference returns None."""
        mgr = undo_redo_manager
        assert mgr.redo() is None

    def test_new_action_clears_redo_stack(self, undo_redo_manager):
        """Recording a new action after undo clears the redo stack."""
        mgr = undo_redo_manager
        actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(actions)
        
        mgr.record_state_before_action("Action 1")
        actions.append({"at": 100, "pos": 50})
        
        mgr.undo()
        assert mgr.can_redo() is True
        
        # New action should clear redo
        mgr.record_state_before_action("Action 2 (new)")
        actions.append({"at": 200, "pos": 75})
        assert mgr.can_redo() is False

    def test_undo_then_redo_full_cycle(self, undo_redo_manager):
        """Full undo-redo cycle preserves state integrity."""
        mgr = undo_redo_manager
        actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(actions)
        
        # Record and modify
        mgr.record_state_before_action("Modify")
        original_state = [d.copy() for d in actions]
        actions[0]["pos"] = 100
        modified_state = [d.copy() for d in actions]
        
        # Undo
        mgr.undo()
        assert actions == original_state
        
        # Redo
        mgr.redo()
        assert actions == modified_state

    def test_multiple_redo(self, undo_redo_manager):
        """Multiple sequential redos restore in correct order."""
        mgr = undo_redo_manager
        actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(actions)
        
        mgr.record_state_before_action("Action 1")
        actions.append({"at": 100, "pos": 50})
        
        mgr.record_state_before_action("Action 2")
        actions.append({"at": 200, "pos": 100})
        
        # Undo both
        mgr.undo()  # Undo Action 2
        mgr.undo()  # Undo Action 1
        assert len(actions) == 1
        
        # Redo both
        mgr.redo()  # Redo Action 1
        assert len(actions) == 2
        mgr.redo()  # Redo Action 2
        assert len(actions) == 3


@pytest.mark.unit
class TestUndoRedoManagerMaxHistory:
    """Tests for maximum history limit."""

    def test_max_history_limit_enforced(self):
        """Undo stack does not exceed max_history."""
        from application.classes import UndoRedoManager
        mgr = UndoRedoManager(max_history=5)
        actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(actions)
        
        for i in range(10):
            mgr.record_state_before_action(f"Action {i}")
            actions.append({"at": (i + 1) * 100, "pos": (i + 1) * 10})
        
        assert len(mgr.undo_stack) == 5

    def test_oldest_history_evicted(self):
        """When max_history is reached, oldest entries are evicted."""
        from application.classes import UndoRedoManager
        mgr = UndoRedoManager(max_history=3)
        actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(actions)
        
        for i in range(5):
            mgr.record_state_before_action(f"Action {i}")
            actions.append({"at": (i + 1) * 100, "pos": (i + 1) * 10})
        
        # Only the last 3 actions should be in the stack
        descriptions = [item[0] for item in mgr.undo_stack]
        assert "Action 0" not in descriptions
        assert "Action 1" not in descriptions
        assert "Action 4" in descriptions


@pytest.mark.unit
class TestUndoRedoManagerClear:
    """Tests for clearing history."""

    def test_clear_history(self, undo_redo_manager):
        """clear_history empties both stacks."""
        mgr = undo_redo_manager
        actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(actions)
        
        mgr.record_state_before_action("Action 1")
        actions.append({"at": 100, "pos": 50})
        mgr.undo()
        
        assert mgr.can_undo() is False  # Undid only action
        assert mgr.can_redo() is True
        
        mgr.clear_history()
        assert mgr.can_undo() is False
        assert mgr.can_redo() is False


@pytest.mark.unit
class TestUndoRedoManagerDisplayHistory:
    """Tests for history display methods."""

    def test_get_undo_history_for_display(self, undo_redo_manager):
        """get_undo_history_for_display returns descriptions in most-recent-first order."""
        mgr = undo_redo_manager
        actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(actions)
        
        mgr.record_state_before_action("First Action")
        actions.append({"at": 100, "pos": 50})
        
        mgr.record_state_before_action("Second Action")
        actions.append({"at": 200, "pos": 75})
        
        mgr.record_state_before_action("Third Action")
        actions.append({"at": 300, "pos": 100})
        
        history = mgr.get_undo_history_for_display()
        assert history == ["Third Action", "Second Action", "First Action"]

    def test_get_redo_history_for_display(self, undo_redo_manager):
        """get_redo_history_for_display returns descriptions in most-recent-first order."""
        mgr = undo_redo_manager
        actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(actions)
        
        mgr.record_state_before_action("Action A")
        actions.append({"at": 100, "pos": 50})
        
        mgr.record_state_before_action("Action B")
        actions.append({"at": 200, "pos": 75})
        
        mgr.undo()  # Undo B
        mgr.undo()  # Undo A
        
        history = mgr.get_redo_history_for_display()
        assert history == ["Action A", "Action B"]

    def test_empty_history_display(self, undo_redo_manager):
        """Display history returns empty list when no history."""
        mgr = undo_redo_manager
        assert mgr.get_undo_history_for_display() == []
        assert mgr.get_redo_history_for_display() == []

    def test_can_undo_can_redo_flags(self, undo_redo_manager):
        """can_undo and can_redo correctly reflect stack states."""
        mgr = undo_redo_manager
        actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(actions)
        
        assert mgr.can_undo() is False
        assert mgr.can_redo() is False
        
        mgr.record_state_before_action("Action")
        actions.append({"at": 100, "pos": 50})
        assert mgr.can_undo() is True
        assert mgr.can_redo() is False
        
        mgr.undo()
        assert mgr.can_undo() is False
        assert mgr.can_redo() is True
        
        mgr.redo()
        assert mgr.can_undo() is True
        assert mgr.can_redo() is False


@pytest.mark.unit
class TestUndoRedoManagerDuplicateStates:
    """Tests for duplicate state handling."""

    def test_duplicate_state_not_pushed(self, undo_redo_manager):
        """Recording the same state with same description is not pushed again."""
        mgr = undo_redo_manager
        actions = [{"at": 0, "pos": 0}]
        mgr.set_actions_reference(actions)
        
        mgr.record_state_before_action("Same Action")
        # Don't modify actions - state is the same
        mgr.record_state_before_action("Same Action")
        
        # Should only have one entry since state + description are identical
        assert len(mgr.undo_stack) == 1

    def test_record_without_reference_does_nothing(self, undo_redo_manager):
        """Recording state without actions reference is a no-op."""
        mgr = undo_redo_manager
        mgr.record_state_before_action("No Reference")
        assert mgr.can_undo() is False
