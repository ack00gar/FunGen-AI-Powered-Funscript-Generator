"""Undo/redo history display mixin for InfoGraphsUI."""
import imgui


class UndoHistoryMixin:

    def _render_content_undo_redo_history(self):
        mgr = self.app.undo_manager
        undo_list = mgr.get_undo_history()
        redo_list = mgr.get_redo_history()

        avail = imgui.get_content_region_available()
        col_w = avail[0] * 0.5 - 4

        # --- Undo column ---
        imgui.begin_child("##UndoCol", width=col_w, height=-1, border=False)

        count_text = f"({len(undo_list)})" if undo_list else ""
        imgui.text_colored(f"Undo {count_text}", 0.5, 0.7, 1.0, 1.0)
        imgui.separator()
        imgui.spacing()

        if undo_list:
            for i, desc in enumerate(undo_list):
                if i >= 30:
                    imgui.text_disabled(f"  ... {len(undo_list) - 30} more")
                    break

                # Clickable selectable row
                is_next = (i == 0)
                if is_next:
                    imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 1.0)
                else:
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.55, 0.55, 0.6, 1.0)

                clicked, _ = imgui.selectable(f"  {desc}##undo_{i}", False)
                imgui.pop_style_color()

                if clicked:
                    n = mgr.undo_to(i, self.app)
                    if n > 0:
                        self.app.notify(f"Undid {n} action{'s' if n > 1 else ''}", "info", 1.5)
                    break  # List changed, stop iterating

                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Click to undo to this point ({i + 1} step{'s' if i > 0 else ''})")
        else:
            imgui.text_disabled("  Nothing to undo")

        imgui.end_child()

        imgui.same_line(spacing=8)

        # --- Redo column ---
        imgui.begin_child("##RedoCol", width=col_w, height=-1, border=False)

        count_text = f"({len(redo_list)})" if redo_list else ""
        imgui.text_colored(f"Redo {count_text}", 0.5, 0.7, 1.0, 1.0)
        imgui.separator()
        imgui.spacing()

        if redo_list:
            for i, desc in enumerate(redo_list):
                if i >= 30:
                    imgui.text_disabled(f"  ... {len(redo_list) - 30} more")
                    break

                is_next = (i == 0)
                if is_next:
                    imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 1.0)
                else:
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.55, 0.55, 0.6, 1.0)

                clicked, _ = imgui.selectable(f"  {desc}##redo_{i}", False)
                imgui.pop_style_color()

                if clicked:
                    n = mgr.redo_to(i, self.app)
                    if n > 0:
                        self.app.notify(f"Redid {n} action{'s' if n > 1 else ''}", "info", 1.5)
                    break

                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Click to redo to this point ({i + 1} step{'s' if i > 0 else ''})")
        else:
            imgui.text_disabled("  Nothing to redo")

        imgui.end_child()
