"""Undo/redo history display mixin for InfoGraphsUI."""
import imgui
from application.utils.section_card import section_card


class UndoHistoryMixin:

    def _render_content_undo_redo_history(self):
        mgr = self.app.undo_manager
        undo_list = mgr.get_undo_history()
        redo_list = mgr.get_redo_history()

        undo_count = len(undo_list)
        redo_count = len(redo_list)

        with section_card(f"Undo ({undo_count})##UndoCard", tier="primary") as u_open:
            if u_open:
                self._render_history_list(undo_list, "undo", mgr)

        with section_card(f"Redo ({redo_count})##RedoCard", tier="primary",
                          open_by_default=bool(redo_list)) as r_open:
            if r_open:
                self._render_history_list(redo_list, "redo", mgr)

    def _render_history_list(self, entries, direction, mgr):
        if not entries:
            imgui.text_disabled("  Empty")
            return

        max_visible = 30
        for i, desc in enumerate(entries):
            if i >= max_visible:
                imgui.text_disabled(f"  ... {len(entries) - max_visible} more")
                break

            is_next = (i == 0)

            # Clickable selectable row
            if is_next:
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 1.0)
            else:
                imgui.push_style_color(imgui.COLOR_TEXT, 0.55, 0.55, 0.6, 1.0)

            clicked, _ = imgui.selectable(f"  {desc}##{direction}_{i}", False)
            imgui.pop_style_color()

            if clicked:
                if direction == "undo":
                    n = mgr.undo_to(i, self.app)
                else:
                    n = mgr.redo_to(i, self.app)
                if n > 0:
                    verb = "Undid" if direction == "undo" else "Redid"
                    self.app.notify(f"{verb} {n} action{'s' if n > 1 else ''}", "info", 1.5)
                break  # List changed, stop iterating

            if imgui.is_item_hovered():
                steps = i + 1
                verb = "undo" if direction == "undo" else "redo"
                imgui.set_tooltip(f"Click to {verb} {steps} step{'s' if steps > 1 else ''}")
