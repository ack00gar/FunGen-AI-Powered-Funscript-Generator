"""Undo/redo history display mixin for InfoGraphsUI."""
import imgui
from application.utils.timeline_constants import EXTRA_TIMELINE_RANGE


class UndoHistoryMixin:

    def _render_content_undo_redo_history(self):
        fs_proc = self.app.funscript_processor
        imgui.begin_child("UndoRedoChild", height=-1, border=True)

        def render_history_for_timeline(num):
            manager = fs_proc._get_undo_manager(num)
            if not manager:
                return

            imgui.text(f"T{num} Undo History:")
            imgui.next_column()
            imgui.text(f"T{num} Redo History:")
            imgui.next_column()

            undo_history = manager.get_undo_history_for_display()
            redo_history = manager.get_redo_history_for_display()

            if undo_history:
                for i, desc in enumerate(undo_history):
                    imgui.text(f"  {i}: {desc}")
            else:
                imgui.text_disabled("  (empty)")

            imgui.next_column()

            if redo_history:
                for i, desc in enumerate(redo_history):
                    imgui.text(f"  {i}: {desc}")
            else:
                imgui.text_disabled("  (empty)")
            imgui.next_column()

        # T1 always
        imgui.columns(2, "UndoRedoColumnsT1")
        render_history_for_timeline(1)
        imgui.columns(1)

        # T2 if visible
        if self.app.app_state_ui.show_funscript_interactive_timeline2:
            imgui.separator()
            imgui.columns(2, "UndoRedoColumnsT2")
            render_history_for_timeline(2)
            imgui.columns(1)

        # T3+ if visible
        for tl_num in EXTRA_TIMELINE_RANGE:
            vis_attr = f"show_funscript_interactive_timeline{tl_num}"
            if getattr(self.app.app_state_ui, vis_attr, False):
                imgui.separator()
                imgui.columns(2, f"UndoRedoColumnsT{tl_num}")
                render_history_for_timeline(tl_num)
                imgui.columns(1)

        imgui.end_child()
