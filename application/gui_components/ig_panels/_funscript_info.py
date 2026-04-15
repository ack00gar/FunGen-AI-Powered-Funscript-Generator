"""Funscript information and quality display mixin for InfoGraphsUI."""
import imgui
import os
import time
from application.utils.timeline_constants import EXTRA_TIMELINE_RANGE
from funscript.quality_validator import FunscriptQualityValidator, IssueSeverity


class FunscriptInfoMixin:

    def _get_timeline_editor(self, timeline_num):
        """Get the timeline editor instance for a given timeline number."""
        gi = self.app.gui_instance
        if not gi:
            return None
        if timeline_num == 1:
            return gi.timeline_editor1
        elif timeline_num == 2:
            return getattr(gi, 'timeline_editor2', None)
        else:
            return gi._extra_timeline_editors.get(timeline_num)

    def _maybe_recompute_quality(self, timeline_num):
        """Recompute quality report with time + delta throttle to avoid per-frame revalidation."""
        try:
            fs_proc = self.app.funscript_processor
            funscript_obj = fs_proc.get_funscript_obj() if fs_proc else None

            # Detect funscript object swap (new project), clear all caches, recompute immediately
            obj_id = id(funscript_obj) if funscript_obj else None
            force = False
            if obj_id != self._quality_last_funscript_id:
                self._quality_last_funscript_id = obj_id
                self._quality_reports.clear()
                self._quality_last_action_counts.clear()
                self._quality_last_compute_time.clear()
                force = True

            if not funscript_obj:
                return

            target_obj, axis_name = fs_proc._get_target_funscript_object_and_axis(timeline_num)
            if not target_obj or not axis_name:
                return

            actions = target_obj.get_axis_actions(axis_name)
            count = len(actions) if actions else 0
            last_count = self._quality_last_action_counts.get(timeline_num, -1)

            # Skip if unchanged
            if count == last_count:
                return

            if count < 2:
                self._quality_last_action_counts[timeline_num] = count
                self._quality_reports.pop(timeline_num, None)
                return

            if not force:
                # Time-based throttle: minimum 5 seconds between recomputations
                now = time.monotonic()
                last_time = self._quality_last_compute_time.get(timeline_num, 0.0)
                if now - last_time < 5.0:
                    # Delta threshold: only bypass throttle if change is significant
                    delta_threshold = max(50, int(last_count * 0.02)) if last_count > 0 else 0
                    if abs(count - last_count) < delta_threshold:
                        return

            self._quality_last_action_counts[timeline_num] = count

            # Get video duration
            duration_ms = 0
            if self.app.processor and self.app.processor.video_info:
                fps = self.app.processor.fps
                total_frames = self.app.processor.video_info.get('total_frames', 0)
                if fps > 0 and total_frames > 0:
                    duration_ms = (total_frames / fps) * 1000.0

            self._quality_reports[timeline_num] = FunscriptQualityValidator().validate(actions, duration_ms)
            self._quality_last_compute_time[timeline_num] = time.monotonic()
        except Exception:
            pass  # Fail silently, quality score is non-critical

    def _render_quality_gauge(self, report):
        """Render an 8px quality gauge bar. Called unconditionally so gauge is always visible."""
        if report is None:
            return

        score = report.score
        if score >= 80:
            bar_color = imgui.get_color_u32_rgba(0.2, 0.8, 0.2, 1.0)
        elif score >= 50:
            bar_color = imgui.get_color_u32_rgba(0.9, 0.7, 0.1, 1.0)
        else:
            bar_color = imgui.get_color_u32_rgba(0.9, 0.2, 0.2, 1.0)

        bg_color = imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0)
        draw_list = imgui.get_window_draw_list()
        cursor = imgui.get_cursor_screen_position()
        avail_w = imgui.get_content_region_available()[0]
        bar_h = 8

        # Background
        draw_list.add_rect_filled(
            cursor[0], cursor[1],
            cursor[0] + avail_w, cursor[1] + bar_h,
            bg_color
        )
        # Foreground fill
        fill_w = avail_w * (score / 100.0)
        if fill_w > 0:
            draw_list.add_rect_filled(
                cursor[0], cursor[1],
                cursor[0] + fill_w, cursor[1] + bar_h,
                bar_color
            )
        # Score text right-aligned
        score_text = f"{score}%"
        text_size = imgui.calc_text_size(score_text)
        text_x = cursor[0] + avail_w - text_size[0] - 4
        text_y = cursor[1] + (bar_h - text_size[1]) * 0.5
        draw_list.add_text(text_x, text_y, imgui.get_color_u32_rgba(1, 1, 1, 0.9), score_text)

        # Advance cursor past the gauge
        imgui.dummy(avail_w, bar_h + 2)

    def _render_funscript_info_section(self, timeline_num):
        """Render funscript info as tree_node with always-visible quality gauge."""
        fs_proc = self.app.funscript_processor

        # Build axis label
        axis_label = ""
        funscript_obj = fs_proc.get_funscript_obj() if fs_proc else None
        if funscript_obj and hasattr(funscript_obj, '_axis_assignments'):
            ax = funscript_obj._axis_assignments.get(timeline_num)
            if ax:
                axis_label = ax

        # Recompute quality (throttled)
        self._maybe_recompute_quality(timeline_num)
        report = self._quality_reports.get(timeline_num)

        # Build header text
        if axis_label:
            header_label = f"Timeline {timeline_num}: {axis_label}"
        else:
            header_label = f"Timeline {timeline_num}"

        if report is not None:
            score = report.score
            header = f"{header_label} | {score}/100##FSInfoT{timeline_num}"
            if score >= 80:
                score_color = (0.2, 0.8, 0.2, 1.0)
            elif score >= 50:
                score_color = (0.9, 0.7, 0.1, 1.0)
            else:
                score_color = (0.9, 0.2, 0.2, 1.0)
            imgui.push_style_color(imgui.COLOR_TEXT, *score_color)
            expanded = imgui.tree_node(header)
            imgui.pop_style_color()
        else:
            header = f"{header_label}##FSInfoT{timeline_num}"
            expanded = imgui.tree_node(header)

        # Gauge bar, always visible regardless of expanded/collapsed
        self._render_quality_gauge(report)

        if expanded:
            # --- Quality KPI row ---
            if report is not None:
                stats = report.stats
                if stats:
                    kpi_parts = [
                        f"Avg speed: {stats.get('avg_speed', 0):.0f} u/s",
                        f"Max: {stats.get('max_speed', 0):.0f} u/s",
                    ]
                    if report.issues:
                        issue_parts = []
                        wc = report.warning_count
                        ec = report.error_count
                        if ec > 0:
                            issue_parts.append(f"{ec}E")
                        if wc > 0:
                            issue_parts.append(f"{wc}W")
                        if issue_parts:
                            kpi_parts.append("Issues: " + " ".join(issue_parts))
                    imgui.text(" | ".join(kpi_parts))
                imgui.spacing()

            # --- Original funscript info content ---
            self._render_content_funscript_info(timeline_num)

            # --- Expandable issue details (collapsed by default) ---
            if report is not None and report.issues:
                if imgui.tree_node(f"Issues ({len(report.issues)})##QualityIssuesT{timeline_num}"):
                    if imgui.begin_child(f"##QualityIssueListT{timeline_num}", 0, 100, border=True):
                        for issue in report.issues:
                            if issue.severity == IssueSeverity.ERROR:
                                icon, color = "[E]", (0.9, 0.2, 0.2, 1.0)
                            elif issue.severity == IssueSeverity.WARNING:
                                icon, color = "[W]", (0.9, 0.7, 0.1, 1.0)
                            else:
                                icon, color = "[i]", (0.5, 0.5, 0.5, 1.0)
                            imgui.push_style_color(imgui.COLOR_TEXT, *color)
                            imgui.text(icon)
                            imgui.pop_style_color()
                            imgui.same_line()
                            imgui.text_wrapped(issue.message)
                    imgui.end_child()
                    imgui.tree_pop()

            imgui.tree_pop()

    def _render_content_funscript_info(self, timeline_num):
        self.funscript_info_perf.start_timing()
        fs_proc = self.app.funscript_processor

        # Get stats with caching, avoid O(N) recomputation every frame
        target_funscript, axis_name = fs_proc._get_target_funscript_object_and_axis(timeline_num)
        if target_funscript and axis_name:
            # Detect funscript object swap, clear all caches
            obj_id = id(target_funscript)
            if obj_id != self._stats_last_funscript_id:
                self._stats_last_funscript_id = obj_id
                self._stats_cache.clear()
                self._stats_last_action_counts.clear()

            actions = target_funscript.get_axis_actions(axis_name)
            count = len(actions) if actions else 0
            last_count = self._stats_last_action_counts.get(timeline_num, -1)

            if count != last_count or timeline_num not in self._stats_cache:
                # Recompute, action count changed
                stats = target_funscript.get_actions_statistics(axis=axis_name)
                self._stats_cache[timeline_num] = stats
                self._stats_last_action_counts[timeline_num] = count
            else:
                stats = self._stats_cache[timeline_num]

            # Get source info from cached stats (for display purposes only)
            stats_attr = f'funscript_stats_t{timeline_num}'
            cached_stats = getattr(fs_proc, stats_attr, {})
            stats["source_type"] = cached_stats.get("source_type", "N/A")
            stats["path"] = cached_stats.get("path", "N/A")
        else:
            # Fallback to cached stats if funscript object not available
            stats_attr = f'funscript_stats_t{timeline_num}'
            stats = getattr(fs_proc, stats_attr, {})

        source_text = stats.get("source_type", "N/A")

        if source_text == "File" and stats.get("path", "N/A") != "N/A":
            source_text = f"File: {os.path.basename(stats['path'])}"
        elif stats.get("path", "N/A") != "N/A":
            source_text = stats["path"]

        imgui.text_wrapped(f"Source: {source_text}")
        imgui.separator()

        imgui.columns(2, f"fs_stats_{timeline_num}", border=False)
        imgui.set_column_width(0, 180 * imgui.get_io().font_global_scale)

        def stat_row(label, value):
            imgui.text(label)
            imgui.next_column()
            imgui.text(str(value))
            imgui.next_column()

        stat_row("Points:", stats.get("num_points", 0))
        stat_row("Duration (s):", f"{stats.get('duration_scripted_s', 0.0):.2f}")
        stat_row("Total Travel:", stats.get("total_travel_dist", 0))
        stat_row("Strokes:", stats.get("num_strokes", 0))
        imgui.separator()

        # Live stroke stats: the segment the playhead is currently inside.
        proc = self.app.processor
        if (proc and proc.fps and proc.fps > 0 and target_funscript and axis_name and actions and len(actions) >= 2):
            ph_ms = (proc.current_frame_index / proc.fps) * 1000.0
            try:
                lo, _ = target_funscript.range_indices(axis_name, ph_ms, ph_ms)
            except Exception:
                lo = 0
            i_after = max(1, min(len(actions) - 1, lo if lo > 0 else 1))
            a_prev = actions[i_after - 1]
            a_next = actions[i_after]
            dt_ms = max(0, a_next['at'] - a_prev['at'])
            d_pos = a_next['pos'] - a_prev['pos']
            speed = (abs(d_pos) / (dt_ms / 1000.0)) if dt_ms > 0 else 0.0
            stat_row("Stroke interval:", f"{dt_ms} ms")
            stat_row("Stroke speed:", f"{speed:.0f} u/s")
            stat_row("Stroke delta:", f"{a_prev['pos']} → {a_next['pos']} ({d_pos:+d})")
            imgui.separator()
        imgui.next_column()
        imgui.separator()
        imgui.next_column()
        stat_row("Avg Speed (pos/s):", f"{stats.get('avg_speed_pos_per_s', 0.0):.2f}")
        stat_row("Avg Intensity (%):", f"{stats.get('avg_intensity_percent', 0.0):.1f}")
        imgui.separator()
        imgui.next_column()
        imgui.separator()
        imgui.next_column()
        stat_row(
            "Position Range:",
            f"{stats.get('min_pos', 'N/A')} - {stats.get('max_pos', 'N/A')}",
        )
        imgui.separator()
        imgui.next_column()
        imgui.separator()
        imgui.next_column()
        stat_row(
            "Min/Max Interval (ms):",
            f"{stats.get('min_interval_ms', 'N/A')} / {stats.get('max_interval_ms', 'N/A')}",
        )
        stat_row("Avg Interval (ms):", f"{stats.get('avg_interval_ms', 0.0):.2f}")

        imgui.columns(1)
        imgui.spacing()

        self.funscript_info_perf.end_timing()
