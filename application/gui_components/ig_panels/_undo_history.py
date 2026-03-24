"""Undo/redo history display mixin for InfoGraphsUI."""
import imgui
from application.utils.timeline_constants import EXTRA_TIMELINE_RANGE
from application.utils.section_card import section_card


class UndoHistoryMixin:

    def _render_content_undo_redo_history(self):
        fs_proc = self.app.funscript_processor
        last_edited = getattr(fs_proc, '_last_edited_timeline', 1)

        def render_timeline_history(num):
            manager = fs_proc._get_undo_manager(num)
            if not manager:
                return

            is_active = (num == last_edited)
            label = f"Timeline {num}"
            if is_active:
                label += " (active)"

            with section_card(f"{label}##UndoT{num}", tier="primary") as _open:
                if not _open:
                    return

                undo_history = manager.get_undo_history_for_display()
                redo_history = manager.get_redo_history_for_display()

                imgui.columns(2, f"UndoRedoCols{num}", border=False)

                # Undo column
                imgui.text_colored("Undo", 0.5, 0.7, 1.0, 1.0)
                if undo_history:
                    for i, desc in enumerate(undo_history):
                        if i == 0:
                            imgui.text(f"  {desc}")
                        else:
                            imgui.text_disabled(f"  {desc}")
                else:
                    imgui.text_disabled("  (empty)")

                imgui.next_column()

                # Redo column
                imgui.text_colored("Redo", 0.5, 0.7, 1.0, 1.0)
                if redo_history:
                    for i, desc in enumerate(redo_history):
                        if i == 0:
                            imgui.text(f"  {desc}")
                        else:
                            imgui.text_disabled(f"  {desc}")
                else:
                    imgui.text_disabled("  (empty)")

                imgui.next_column()
                imgui.columns(1)

        # T1 always
        render_timeline_history(1)

        # T2 if visible
        if self.app.app_state_ui.show_funscript_interactive_timeline2:
            render_timeline_history(2)

        # T3+ if visible
        for tl_num in EXTRA_TIMELINE_RANGE:
            vis_attr = f"show_funscript_interactive_timeline{tl_num}"
            if getattr(self.app.app_state_ui, vis_attr, False):
                render_timeline_history(tl_num)
