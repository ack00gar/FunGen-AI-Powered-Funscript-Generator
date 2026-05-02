"""Side blocks for the quad-block layout.

Each block is a self-contained panel rendered inside a fixed-position imgui
window. Window position and size are set by the caller via set_next_window_*.

- LeftBottomBlock: chapters + bookmarks (compact list with seek on click).
- RightBottomBlock: plugins listed by category with access to the pipeline.
- render_collapsed_chevron: thin strip shown when a whole column is collapsed.
"""
from typing import Dict, List

import imgui

from application.utils.section_card import section_card
from config.constants_colors import CurrentTheme


# Mirrors the flag set used by ControlPanelUI / InfoGraphsUI in fixed mode:
# no title bar, no move, no collapse, but resize *grip* is allowed (purely
# visual — set_next_window_size still pins the geometry every frame).
_WIN_FLAGS = (
    imgui.WINDOW_NO_TITLE_BAR
    | imgui.WINDOW_NO_MOVE
    | imgui.WINDOW_NO_COLLAPSE
    | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
)

_FLOAT_FLAGS = imgui.WINDOW_NO_COLLAPSE  # closable + resizable + movable


def _render_panel_label(text: str) -> None:
    """Dim, centered, uppercase panel title — matches AI ANALYSIS / EXPERT style."""
    label = text.upper()
    text_size = imgui.calc_text_size(label)
    avail_w = imgui.get_content_region_available_width()
    x_offset = (avail_w - text_size[0]) * 0.5
    if x_offset > 0:
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + x_offset)
    imgui.text_colored(label, *CurrentTheme.GRAY_SUBDUED)
    imgui.spacing()


def _playhead_time_ms(app) -> float:
    """Return the video's real playhead time in ms (the frame currently shown)."""
    proc = app.processor
    if not proc or not proc.fps or proc.fps <= 0:
        return 0.0
    return (proc.current_frame_index / proc.fps) * 1000.0


class LeftBottomBlock:
    def __init__(self, app, gui):
        self.app = app
        self.gui = gui

    def render(self, floating: bool = False):
        app_state = self.app.app_state_ui
        if floating:
            if not getattr(app_state, 'show_left_bottom_block', True):
                return
            imgui.set_next_window_size(380, 480, condition=imgui.FIRST_USE_EVER)
            opened, vis = imgui.begin("FunGen: Chapters & Bookmarks##LeftBottomFloat",
                                       closable=True, flags=_FLOAT_FLAGS)
            if vis != getattr(app_state, 'show_left_bottom_block', True):
                app_state.show_left_bottom_block = vis
                self.app.app_settings.config.ui.show_left_bottom_block = vis
            if not opened:
                imgui.end()
                return
        else:
            imgui.begin("Chapters & Bookmarks##LeftBottomBlock", flags=_WIN_FLAGS)
            _render_panel_label("Chapters & Bookmarks")
        imgui.begin_child("##lb_scroll", 0, 0, border=False)
        self._render_chapters(app_state)
        self._render_bookmarks(app_state)
        imgui.end_child()
        imgui.end()

    def _render_chapters(self, app_state):
        fs_proc = getattr(self.app, 'funscript_processor', None)
        chapters = getattr(fs_proc, 'video_chapters', None) or () if fs_proc else ()
        with section_card(f"Chapters ({len(chapters)})##lb_ch", tier="primary") as open_:
            if not open_:
                return
            if imgui.small_button("Open list...##lb_ch_open"):
                app_state.show_chapter_list_window = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Open the full chapter list window (rename, re-time, delete, seek to any chapter).")
            imgui.same_line()
            if imgui.small_button("Types...##lb_ch_types"):
                app_state.show_chapter_type_manager = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Manage chapter types (add custom categories, change colors).")

            if not chapters:
                imgui.text_disabled("No chapters")
                return

            proc = self.app.processor
            fps = (proc.fps if proc and proc.fps and proc.fps > 0 else 30.0)
            cur_frame = (proc.current_frame_index if proc else 0)
            child_h = max(60, min(260, len(chapters) * 20 + 8))
            imgui.begin_child("##lb_ch_list", 0, child_h, border=True)

            # Any chapter >= 1h? Use HH:MM:SS uniformly for alignment.
            max_t = max((ch.end_frame_id / fps for ch in chapters), default=0) if fps else 0
            use_hours = max_t >= 3600

            def _fmt(t: float) -> str:
                t = int(t)
                if use_hours:
                    return f"{t // 3600:d}:{(t % 3600) // 60:02d}:{t % 60:02d}"
                return f"{t // 60:02d}:{t % 60:02d}"

            # Cache labels + per-label widths by chapter id; calc_text_size
            # is an IPC call and was firing N x per-frame.
            label_cache = getattr(self, '_lb_ch_label_cache', None)
            ckey = (id(chapters), len(chapters))
            if label_cache is None or label_cache.get('__key') != ckey:
                label_cache = {'__key': ckey}
                self._lb_ch_label_cache = label_cache
            labels = []
            label_widths = []
            for ch in chapters:
                cid = ch.unique_id
                lb = ch.position_short_name or ch.position_long_name or "Chapter"
                cached = label_cache.get(cid)
                if cached is None or cached[0] != lb:
                    w = imgui.calc_text_size(lb)[0]
                    label_cache[cid] = (lb, w)
                else:
                    w = cached[1]
                labels.append(lb)
                label_widths.append(w)
            max_label_w = max(label_widths, default=0.0)
            time_col_x = max_label_w + 12.0

            for ch, label in zip(chapters, labels):
                t = ch.start_frame_id / fps if fps else 0
                is_current = ch.start_frame_id <= cur_frame <= ch.end_frame_id
                if is_current:
                    imgui.push_style_color(imgui.COLOR_TEXT, *CurrentTheme.REFERENCE_MATCH_GOLD)
                row_x, row_y = imgui.get_cursor_pos()
                clicked = imgui.selectable(f"##lb_ch_{ch.unique_id}", False)[0]
                if imgui.is_item_hovered():
                    long_name = ch.position_long_name or ch.position_short_name or "Chapter"
                    dur = max(0, (ch.end_frame_id - ch.start_frame_id) / fps) if fps else 0
                    imgui.set_tooltip(f"{long_name}\n{_fmt(t)} - {_fmt(t + dur)}  ({_fmt(dur)})")
                next_y = imgui.get_cursor_pos_y()
                imgui.set_cursor_pos((row_x + 4, row_y))
                imgui.text(label)
                imgui.same_line(position=row_x + time_col_x + 4)
                imgui.text_disabled(_fmt(t))
                imgui.set_cursor_pos((row_x, next_y))
                if is_current:
                    imgui.pop_style_color()
                if clicked:
                    try:
                        self.app.event_handlers.seek_video_with_sync(int(ch.start_frame_id))
                    except Exception:
                        pass
            imgui.end_child()

    def _render_bookmarks(self, app_state):
        bm_ui = getattr(self.gui, 'bookmark_list_window_ui', None)
        try:
            all_bm = bm_ui._collect_all_bookmarks() if bm_ui else []
        except Exception:
            all_bm = []
        with section_card(f"Bookmarks ({len(all_bm)})##lb_bm", tier="primary") as open_:
            if not open_:
                return

            if imgui.small_button("Add at playhead##lb_bm_add"):
                if bm_ui:
                    t_ms = _playhead_time_ms(self.app)
                    mgrs = bm_ui._get_all_bookmark_managers()
                    if mgrs:
                        mgrs[0][1].add(t_ms)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Place a bookmark on Funscript 1 at the current playback time.")
            imgui.same_line()
            if imgui.small_button("Open list...##lb_bm_open"):
                app_state.show_bookmark_list_window = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Open the full bookmark list window (rename, delete, jump to any bookmark).")

            if not all_bm:
                imgui.text_disabled("No bookmarks")
                return

            child_h = max(60, min(220, len(all_bm) * 20 + 8))
            imgui.begin_child("##lb_bm_list", 0, child_h, border=True)

            max_secs = max((bm.time_ms / 1000.0 for _, _, bm in all_bm), default=0)
            use_hours_bm = max_secs >= 3600

            def _fmt_bm(secs: float) -> str:
                if use_hours_bm:
                    h = int(secs // 3600)
                    m = int((secs % 3600) // 60)
                    s = secs % 60
                    return f"{h:d}:{m:02d}:{s:05.2f}"
                return f"{int(secs // 60):02d}:{secs % 60:05.2f}"

            tl_labels = [f"F{tl_num}" for tl_num, _, _ in all_bm]
            max_tl_w = max((imgui.calc_text_size(lb)[0] for lb in tl_labels), default=0.0)
            # Probe a representative time string for column width.
            sample_time = _fmt_bm(max_secs if all_bm else 0)
            time_w = imgui.calc_text_size(sample_time)[0]
            time_col_x_bm = max_tl_w + 10.0
            name_col_x_bm = time_col_x_bm + time_w + 10.0

            for (tl_num, mgr, bm), tl_lbl in zip(all_bm, tl_labels):
                secs = bm.time_ms / 1000.0
                name = (bm.name or "")
                row_x, row_y = imgui.get_cursor_pos()
                clicked = imgui.selectable(f"##lb_bm_{bm.id}", False)[0]
                next_y = imgui.get_cursor_pos_y()
                imgui.set_cursor_pos((row_x + 4, row_y))
                imgui.text_disabled(tl_lbl)
                imgui.same_line(position=row_x + time_col_x_bm + 4)
                imgui.text(_fmt_bm(secs))
                if name:
                    imgui.same_line(position=row_x + name_col_x_bm + 4)
                    imgui.text_disabled(name)
                imgui.set_cursor_pos((row_x, next_y))
                if clicked:
                    try:
                        bm_ui._seek(bm.time_ms)
                    except Exception:
                        pass
            imgui.end_child()


class RightBottomBlock:
    def __init__(self, app, gui):
        self.app = app
        self.gui = gui
        self._plugin_cache = None

    def invalidate(self):
        self._plugin_cache = None

    def _open_plugin_on_active_timeline(self, plugin_name: str) -> None:
        """Open a plugin dialog on the active timeline so 'apply to selection'
        respects the current point selection. Falls back to the Pipeline editor
        if no timeline can be reached."""
        try:
            tl_num = getattr(self.app.app_state_ui, 'active_timeline_num', 1) or 1
            tl = getattr(self.gui, f'timeline_editor{tl_num}', None)
            if tl is None:
                tl = getattr(self.gui, 'timeline_editor1', None)
            if tl is not None and hasattr(tl, '_activate_plugin'):
                has_sel = bool(getattr(tl, 'multi_selected_action_indices', None))
                tl._activate_plugin(plugin_name, apply_to_selection=has_sel)
                return
        except Exception:
            pass
        # Fallback: open the Pipeline editor preselected (legacy behavior).
        self.app.app_state_ui.show_plugin_pipeline = True
        try:
            self.gui.plugin_pipeline_ui.preselect_plugin = plugin_name
        except Exception:
            pass

    def _build_cache(self):
        try:
            from funscript.plugins.base_plugin import plugin_registry
            plugins = plugin_registry.list_plugins()
        except Exception:
            plugins = []
        grouped: Dict[str, List[Dict]] = {}
        for p in plugins:
            cat = (p.get('category') or 'misc').strip() or 'misc'
            grouped.setdefault(cat, []).append(p)
        for cat in grouped:
            grouped[cat].sort(key=lambda x: x.get('name', ''))
        self._plugin_cache = sorted(grouped.items(), key=lambda kv: kv[0].lower())

    def render(self, floating: bool = False):
        app_state = self.app.app_state_ui
        if floating:
            if not getattr(app_state, 'show_right_bottom_block', True):
                return
            imgui.set_next_window_size(380, 480, condition=imgui.FIRST_USE_EVER)
            opened, vis = imgui.begin("FunGen: Plugins##RightBottomFloat",
                                       closable=True, flags=_FLOAT_FLAGS)
            if vis != getattr(app_state, 'show_right_bottom_block', True):
                app_state.show_right_bottom_block = vis
                self.app.app_settings.config.ui.show_right_bottom_block = vis
            if not opened:
                imgui.end()
                return
        else:
            imgui.begin("Plugins##RightBottomBlock", flags=_WIN_FLAGS)
            _render_panel_label("Plugins")
        imgui.begin_child("##rb_scroll", 0, 0, border=False)

        if imgui.button("Open Pipeline...##rb_pipe", width=-1, height=24):
            app_state.show_plugin_pipeline = True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Open the Plugin Pipeline window to chain post-processing plugins and run them as a single sequence.")

        if self._plugin_cache is None:
            self._build_cache()

        for cat, plist in self._plugin_cache:
            with section_card(f"{cat.title()} ({len(plist)})##rb_cat_{cat}",
                              tier="primary") as open_:
                if not open_:
                    continue
                avail_w = imgui.get_content_region_available_width()
                btn_w_target = 135.0
                n_cols = max(1, int(avail_w // (btn_w_target + 6)))
                btn_w = (avail_w - (n_cols - 1) * 6) / n_cols if n_cols > 0 else avail_w
                for idx, p in enumerate(plist):
                    name = p.get('name', '?')
                    desc = p.get('description', '')
                    disp = name if len(name) <= 18 else name[:17] + "..."
                    if imgui.button(f"{disp}##rb_p_{name}", width=btn_w, height=24):
                        self._open_plugin_on_active_timeline(name)
                    if imgui.is_item_hovered():
                        tip = name if not desc else f"{name}\n\n{desc}"
                        imgui.set_tooltip(tip)
                    if (idx + 1) % n_cols != 0 and idx < len(plist) - 1:
                        imgui.same_line(spacing=6)
        imgui.end_child()
        imgui.end()


def render_collapsed_chevron(side: str, x: float, y: float, w: float, h: float, app_state):
    imgui.set_next_window_position(x, y)
    imgui.set_next_window_size(w, h)
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (2, 4))
    imgui.begin(f"##chev_{side}", flags=_WIN_FLAGS | imgui.WINDOW_NO_SCROLLBAR)
    label = ">" if side == 'left' else "<"
    if imgui.button(f"{label}##chev_btn_{side}", width=max(8, w - 4), height=24):
        ui_cfg = app_state.app_settings.config.ui
        if side == 'left':
            app_state.show_left_top_block = True
            app_state.show_left_bottom_block = True
            ui_cfg.show_left_top_block = True
            ui_cfg.show_left_bottom_block = True
        else:
            app_state.show_right_top_block = True
            app_state.show_right_bottom_block = True
            ui_cfg.show_right_top_block = True
            ui_cfg.show_right_bottom_block = True
    if imgui.is_item_hovered():
        imgui.set_tooltip("Show panel")
    imgui.end()
    imgui.pop_style_var()
