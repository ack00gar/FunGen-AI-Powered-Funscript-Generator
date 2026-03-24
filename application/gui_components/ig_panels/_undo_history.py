"""Undo/redo history display mixin for InfoGraphsUI."""
import imgui
from application.utils.section_card import section_card


class UndoHistoryMixin:

    def _render_content_undo_redo_history(self):
        mgr = self.app.undo_manager

        imgui.columns(2, "UndoRedoUnifiedCols", border=False)

        # Undo column
        imgui.text_colored("Undo", 0.5, 0.7, 1.0, 1.0)
        undo_list = mgr.get_undo_history()
        if undo_list:
            for i, desc in enumerate(undo_list):
                if i == 0:
                    imgui.text(f"  {desc}")
                else:
                    imgui.text_disabled(f"  {desc}")
                if i >= 19:
                    imgui.text_disabled(f"  ... and {len(undo_list) - 20} more")
                    break
        else:
            imgui.text_disabled("  (empty)")

        imgui.next_column()

        # Redo column
        imgui.text_colored("Redo", 0.5, 0.7, 1.0, 1.0)
        redo_list = mgr.get_redo_history()
        if redo_list:
            for i, desc in enumerate(redo_list):
                if i == 0:
                    imgui.text(f"  {desc}")
                else:
                    imgui.text_disabled(f"  {desc}")
                if i >= 19:
                    imgui.text_disabled(f"  ... and {len(redo_list) - 20} more")
                    break
        else:
            imgui.text_disabled("  (empty)")

        imgui.next_column()
        imgui.columns(1)
