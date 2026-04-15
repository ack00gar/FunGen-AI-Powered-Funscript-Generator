"""Reference comparison and segment statistics mixin for InfoGraphsUI."""
import imgui
from bisect import bisect_left
from application.utils.timeline_constants import EXTRA_TIMELINE_RANGE


class ComparisonMixin:

    def _has_any_reference_loaded(self):
        """Check if any visible timeline has a reference overlay loaded."""
        if self._get_timeline_editor(1) and getattr(self._get_timeline_editor(1), 'reference_overlay_actions', None):
            return True
        if self.app.app_state_ui.show_funscript_interactive_timeline2:
            tl2 = self._get_timeline_editor(2)
            if tl2 and getattr(tl2, 'reference_overlay_actions', None):
                return True
        for tl_num in EXTRA_TIMELINE_RANGE:
            if getattr(self.app.app_state_ui, f"show_funscript_interactive_timeline{tl_num}", False):
                tl = self._get_timeline_editor(tl_num)
                if tl and getattr(tl, 'reference_overlay_actions', None):
                    return True
        return False

    def _render_reference_comparison_standalone(self):
        """Render the dedicated Funscript Comparison section for all timelines with references."""
        # Collect timelines with references
        timelines = []
        tl1 = self._get_timeline_editor(1)
        if tl1 and tl1.reference_overlay_actions:
            timelines.append((1, tl1))
        if self.app.app_state_ui.show_funscript_interactive_timeline2:
            tl2 = self._get_timeline_editor(2)
            if tl2 and getattr(tl2, 'reference_overlay_actions', None):
                timelines.append((2, tl2))
        for tl_num in EXTRA_TIMELINE_RANGE:
            if getattr(self.app.app_state_ui, f"show_funscript_interactive_timeline{tl_num}", False):
                tl = self._get_timeline_editor(tl_num)
                if tl and getattr(tl, 'reference_overlay_actions', None):
                    timelines.append((tl_num, tl))

        for timeline_num, tl in timelines:
            metrics = tl._reference_metrics
            if not metrics:
                continue

            ref_name = tl.reference_overlay_name or "Reference"
            ps = metrics.get('peak_stats', {})
            classes = ps.get('classes', {})
            good = classes.get('gold', 0) + classes.get('green', 0)
            total = ps.get('total', 0)
            pct = f"{good/total:.0%}" if total > 0 else "-"

            # Header: T1: filename — summary
            label = f"T{timeline_num}: {ref_name}"
            if len(timelines) == 1:
                label = ref_name

            imgui.text_colored(label, 0.3, 0.65, 1.0, 1.0)

            # --- Always-visible overview ---
            mae = metrics.get('mae', 0)
            corr = metrics.get('correlation', 0)

            # Score color based on peak accuracy
            if total > 0:
                ratio = good / total
                if ratio >= 0.8:
                    score_col = (0.2, 0.8, 0.2, 1.0)
                elif ratio >= 0.5:
                    score_col = (0.9, 0.7, 0.1, 1.0)
                else:
                    score_col = (0.9, 0.2, 0.2, 1.0)
            else:
                score_col = (0.5, 0.5, 0.5, 1.0)

            imgui.text("Accuracy:")
            imgui.same_line()
            imgui.text_colored(f"{pct}", *score_col)
            imgui.same_line()
            imgui.text(f"({good}/{total} peaks)")
            imgui.same_line()
            imgui.text_colored(f"  MAE: {mae:.1f}", 0.7, 0.7, 0.75, 1.0)
            imgui.same_line()
            imgui.text_colored(f"  Corr: {corr:.2f}", 0.7, 0.7, 0.75, 1.0)

            # Peak breakdown: colored legend
            imgui.text("  ")
            imgui.same_line()
            imgui.text_colored(f"exact:{classes.get('gold', 0)}", 1.0, 0.85, 0.0, 1.0)
            imgui.same_line()
            imgui.text_colored(f"close:{classes.get('green', 0)}", 0.2, 1.0, 0.3, 1.0)
            imgui.same_line()
            imgui.text_colored(f"off:{classes.get('yellow', 0)}", 1.0, 0.9, 0.2, 1.0)
            imgui.same_line()
            imgui.text_colored(f"wrong:{classes.get('red', 0)}", 1.0, 0.2, 0.2, 1.0)
            unmatched_m = ps.get('unmatched_main', 0)
            unmatched_r = ps.get('unmatched_ref', 0)
            if unmatched_m or unmatched_r:
                imgui.same_line()
                imgui.text_colored(f"missing:{unmatched_m + unmatched_r}", 0.5, 0.5, 0.5, 1.0)

            # --- Expandable details ---
            if imgui.tree_node(f"Details##RefDetT{timeline_num}"):
                speed_mae = metrics.get('speed_mae', 0)
                coverage = metrics.get('coverage', 0)
                imgui.text(f"Speed MAE: {speed_mae:.1f}  |  Coverage: {coverage:.0%}")

                # Per-chapter breakdown
                per_ch = metrics.get('per_chapter', {})
                if per_ch:
                    imgui.spacing()
                    if imgui.tree_node(f"Per Chapter ({len(per_ch)})##RefChT{timeline_num}"):
                        for ch_name, ch_info in per_ch.items():
                            ch_mae = ch_info.get('mae', 0)
                            ch_acc = ch_info.get('peaks_accuracy', 0)
                            ch_good = ch_info.get('peaks_good', 0)
                            ch_total = ch_info.get('peaks_total', 0)
                            if ch_acc >= 0.8:
                                col = (0.2, 0.8, 0.2, 1.0)
                            elif ch_acc >= 0.5:
                                col = (0.9, 0.7, 0.1, 1.0)
                            else:
                                col = (0.9, 0.2, 0.2, 1.0)
                            imgui.text_colored(f"{ch_name}", *col)
                            imgui.same_line()
                            imgui.text(f"MAE:{ch_mae:.1f}  {ch_good}/{ch_total} ({ch_acc:.0%})")
                        imgui.tree_pop()

                # Problem sections
                problems = tl._reference_problem_sections
                if problems:
                    imgui.spacing()
                    n = len(problems)
                    if imgui.tree_node(f"Problem Sections ({n})##RefProbT{timeline_num}"):
                        for sec in problems:
                            start_s = sec['start_ms'] / 1000.0
                            end_s = sec['end_ms'] / 1000.0
                            dur_s = sec['duration_ms'] / 1000.0
                            sec_mae = sec.get('mae', 0)
                            imgui.text_colored(
                                f"{start_s:.1f}s - {end_s:.1f}s ({dur_s:.1f}s) MAE:{sec_mae:.1f}",
                                0.9, 0.4, 0.4, 1.0
                            )
                        imgui.tree_pop()

                imgui.tree_pop()

            if len(timelines) > 1:
                imgui.separator()

    def _render_segment_statistics(self):
        """Render segment statistics for the current playhead position."""
        processor = self.app.processor
        fs_proc = self.app.funscript_processor
        if not processor or not fs_proc or not processor.video_info:
            imgui.text_colored("No video loaded", 0.5, 0.5, 0.5, 1.0)
            return

        fps = processor.fps
        if fps <= 0:
            imgui.text_colored("No video loaded", 0.5, 0.5, 0.5, 1.0)
            return

        current_time_ms = (processor.current_frame_index / fps) * 1000.0
        active_tl = getattr(self.app.app_state_ui, 'active_timeline_num', 1)

        fs_obj, axis_name = fs_proc._get_target_funscript_object_and_axis(active_tl)
        if not fs_obj or not axis_name:
            imgui.text_colored("No funscript data", 0.5, 0.5, 0.5, 1.0)
            return

        actions = fs_obj.get_axis_actions(axis_name)
        if not actions or len(actions) < 2:
            imgui.text_colored("Not enough actions", 0.5, 0.5, 0.5, 1.0)
            return

        # Find bounding actions using bisect on cached timestamps
        timestamps = fs_obj._get_timestamps_for_axis(axis_name)
        idx = bisect_left(timestamps, current_time_ms)

        # Clamp to valid segment range
        if idx <= 0:
            idx = 1
        if idx >= len(actions):
            idx = len(actions) - 1

        behind = actions[idx - 1]
        front = actions[idx]

        seg_duration_ms = front['at'] - behind['at']
        interval_ms = max(0, current_time_ms - behind['at'])
        pos_delta = front['pos'] - behind['pos']
        abs_delta = abs(pos_delta)

        if seg_duration_ms > 0:
            speed = abs_delta / (seg_duration_ms / 1000.0)
        else:
            speed = 0.0

        arrow = "UP" if pos_delta > 0 else ("DN" if pos_delta < 0 else "--")

        imgui.text(f"Timeline {active_tl}")
        imgui.separator()
        imgui.text(f"Interval:  {interval_ms:.0f} ms")
        imgui.text(f"Duration:  {seg_duration_ms:.0f} ms")
        imgui.text(f"Speed:     {speed:.0f} units/s")
        imgui.text(f"Direction: {behind['pos']} -> {front['pos']} = {abs_delta} {arrow}")
